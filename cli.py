"""CLI — drive auto-research from the command line.

Usage:
    auto-research run --source synthetic --max-iterations 10
    auto-research status
    auto-research positions --source synthetic
    auto-research export ./my-strategy/
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    import polars as pl

import numpy as np
import typer
from loguru import logger

from datasource import load_raw

app = typer.Typer(
    name="auto-research",
    help="Autonomous trading strategy discovery with pluggable data sources.",
    no_args_is_help=True,
)

STRATEGY_FILE = Path("strategy.py")
TEMPLATE_FILE = Path("strategy_template.py")
EXPERIMENTS_LOG = Path("experiments.jsonl")
SESSION_FILE = Path("session.md")
FIREWALL_KEY_FILE = Path(".firewall_key")


def _parse_source_args(source_args: list[str]) -> dict:
    """Parse key=value source args into a dict."""
    kwargs = {}
    for arg in source_args:
        if "=" not in arg:
            logger.error("[CLI] --source-arg must be key=value, got: {}", arg)
            raise typer.Exit(1)
        k, v = arg.split("=", 1)
        kwargs[k] = v
    return kwargs


def _load_experiments() -> list[dict]:
    """Load all experiments from JSONL log."""
    if not EXPERIMENTS_LOG.exists():
        return []
    experiments = []
    for line in EXPERIMENTS_LOG.read_text().strip().split("\n"):
        if line.strip():
            experiments.append(json.loads(line))
    return experiments


def _load_firewalled_data(source: str, source_kwargs: dict) -> pl.DataFrame:
    """Load data using the persisted firewall key (same as run command).

    This ensures all CLI commands evaluate strategies on the same data
    that the run command trained on.
    """
    from firewall import anonymize_dataset, generate_key, load_key, save_key

    raw_df = load_raw(source, **source_kwargs)

    if FIREWALL_KEY_FILE.exists():
        fw_key = load_key(FIREWALL_KEY_FILE)
    else:
        fw_key = generate_key()
        save_key(fw_key, FIREWALL_KEY_FILE)
        logger.info("[DATA] New firewall key saved to {}", FIREWALL_KEY_FILE)

    data_df, _ = anonymize_dataset(raw_df, fw_key)
    return data_df


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    source: Annotated[str, typer.Option(help="Data source name")] = "synthetic",
    source_arg: Annotated[
        list[str] | None, typer.Option(help="Source args as key=value")
    ] = None,
    symbol: Annotated[str | None, typer.Option(help="Filter to anonymized symbol")] = None,
    max_iterations: Annotated[int, typer.Option(help="Max iterations (0=infinite)")] = 0,
    model: Annotated[str, typer.Option(help="LLM model")] = "claude-sonnet-4-20250514",
) -> None:
    """Run the autonomous strategy discovery loop."""
    import anthropic
    import polars as pl

    from loop import (
        STRATEGY_FILE,
        TEMPLATE_FILE,
        log_experiment,
        polars_to_numpy_bars,
        run_iteration,
    )

    source_kwargs = _parse_source_args(source_arg or [])

    logger.info("[CLI] run: source={} model={}", source, model)
    try:
        data_df = _load_firewalled_data(source, source_kwargs)
    except Exception as e:
        logger.error("[DATA] Failed to load: {}", e)
        raise typer.Exit(1) from None

    symbols = data_df["symbol"].unique().to_list()
    if symbol:
        data_df = data_df.filter(pl.col("symbol") == symbol)
    else:
        data_df = data_df.filter(pl.col("symbol") == symbols[0])
    logger.info("[DATA] Using symbol: {} ({} bars)", data_df["symbol"][0], len(data_df))

    if len(data_df) < 50:
        logger.error("[DATA] Too few bars ({}). Need at least 50.", len(data_df))
        raise typer.Exit(1)

    bars_np = polars_to_numpy_bars(data_df)
    close_returns = bars_np["close"]

    if not Path(".git").exists():
        subprocess.run(["git", "init"], capture_output=True, check=False)

    if not STRATEGY_FILE.exists():
        shutil.copy(TEMPLATE_FILE, STRATEGY_FILE)
        subprocess.run(["git", "add", "strategy.py"], capture_output=True, check=False)
        subprocess.run(
            ["git", "commit", "-m", "init: baseline strategy"],
            capture_output=True,
            check=False,
        )

    client = anthropic.Anthropic()
    best_sharpe = float("-inf")
    iteration = 0

    logger.info("[LOOP] {} bars, model={}", len(close_returns), model)

    while True:
        iteration += 1
        if 0 < max_iterations < iteration:
            logger.info("[LOOP] Reached max iterations ({})", max_iterations)
            break

        logger.info("[LOOP] Iteration {} (best Sharpe: {:.4f})", iteration, best_sharpe)

        try:
            result = run_iteration(client, bars_np, close_returns, best_sharpe, model)
        except KeyboardInterrupt:
            logger.info("[LOOP] Interrupted")
            break
        except Exception as e:
            logger.error("[LOOP] {}", e)
            log_experiment({"status": "crash", "error": str(e), "description": str(e)})
            continue

        status = result["status"]
        sharpe = result.get("sharpe", 0)
        trades = result.get("trades", 0)

        if status == "keep":
            best_sharpe = max(best_sharpe, sharpe)
            logger.success(
                "[KEEP] Sharpe={:.4f}, trades={}, best={:.4f}", sharpe, trades, best_sharpe
            )
        elif status == "crash":
            logger.warning("[CRASH] {}", result.get("error", "unknown")[:100])
        else:
            logger.info("[DISCARD] Sharpe={:.4f}, trades={}", sharpe, trades)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show session stats, best strategy, and recent results."""
    experiments = _load_experiments()

    if not experiments:
        logger.info("[STATUS] No experiments yet. Run 'auto-research run' first.")
        raise typer.Exit()

    kept = [e for e in experiments if e.get("status") == "keep"]
    discarded = [e for e in experiments if e.get("status") == "discard"]
    crashed = [e for e in experiments if e.get("status") == "crash"]

    best = max(kept, key=lambda e: e.get("sharpe", 0)) if kept else None

    typer.echo(f"Total runs:  {len(experiments)}")
    typer.echo(f"Kept:        {len(kept)}")
    typer.echo(f"Discarded:   {len(discarded)}")
    typer.echo(f"Crashed:     {len(crashed)}")
    typer.echo(f"Best Sharpe: {best['sharpe'] if best else 'N/A'}")
    typer.echo(f"Best DD:     {best.get('max_drawdown', 'N/A') if best else 'N/A'}")
    typer.echo("")
    typer.echo("Recent results:")
    for exp in experiments[-5:]:
        s = exp.get("status", "?")
        sh = exp.get("sharpe", "?")
        tr = exp.get("trades", "?")
        typer.echo(f"  [{s:>8}] Sharpe={sh}, trades={tr}")


# ---------------------------------------------------------------------------
# positions
# ---------------------------------------------------------------------------


@app.command()
def positions(
    source: Annotated[str, typer.Option(help="Data source name")] = "synthetic",
    source_arg: Annotated[
        list[str] | None, typer.Option(help="Source args as key=value")
    ] = None,
    symbol: Annotated[str | None, typer.Option(help="Filter to anonymized symbol")] = None,
    output: Annotated[str, typer.Option(help="Output format: json or csv")] = "json",
) -> None:
    """Run best committed strategy on latest data and output position weights."""
    import polars as pl

    from backtest import backtest
    from loop import polars_to_numpy_bars
    from sandbox import run_strategy

    if not STRATEGY_FILE.exists():
        logger.error("[POSITIONS] No strategy.py found. Run 'auto-research run' first.")
        raise typer.Exit(1)

    source_kwargs = _parse_source_args(source_arg or [])
    data_df = _load_firewalled_data(source, source_kwargs)

    symbols = data_df["symbol"].unique().to_list()
    if symbol:
        data_df = data_df.filter(pl.col("symbol") == symbol)
    else:
        data_df = data_df.filter(pl.col("symbol") == symbols[0])

    bars_np = polars_to_numpy_bars(data_df)
    bars_json = {k: v.tolist() for k, v in bars_np.items()}

    result = run_strategy(STRATEGY_FILE, bars_json)
    if "error" in result:
        logger.error("[POSITIONS] Strategy failed: {}", result["error"])
        raise typer.Exit(1)

    pos = np.array(result["positions"])
    close = bars_np["close"]
    bt = backtest(close, pos)

    if output == "json":
        out = {
            "symbol": data_df["symbol"][0],
            "bars": len(pos),
            "current_position": float(pos[-1]),
            "metrics": bt,
            "last_10_positions": [float(p) for p in pos[-10:]],
        }
        typer.echo(json.dumps(out, indent=2))
    else:
        typer.echo("bar,position")
        for i, p in enumerate(pos):
            typer.echo(f"{i},{p:.4f}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command()
def export(
    dest: Annotated[str, typer.Argument(help="Destination directory")] = "export",
    no_docker: Annotated[bool, typer.Option(help="Skip Dockerfile generation")] = False,
) -> None:
    """Export best strategy as a standalone project."""
    dest_path = Path(dest)

    if not STRATEGY_FILE.exists():
        logger.error("[EXPORT] No strategy.py found. Run 'auto-research run' first.")
        raise typer.Exit(1)

    dest_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(STRATEGY_FILE, dest_path / "strategy.py")

    # Write a minimal runner
    runner = '''\
"""Minimal runner for an exported auto-research strategy."""

import json
import sys

import numpy as np

from strategy import strategy


def main():
    """Load data from stdin (JSON) and output positions."""
    raw = json.loads(sys.stdin.read())
    bars = {k: np.array(v) for k, v in raw.items()}
    positions = strategy(bars)
    print(json.dumps({"positions": [float(p) for p in positions]}))


if __name__ == "__main__":
    main()
'''
    (dest_path / "run.py").write_text(runner)
    (dest_path / "requirements.txt").write_text("numpy\npandas\n")

    # Write Dockerfile unless --no-docker
    if not no_docker:
        dockerfile = """\
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY strategy.py run.py ./

CMD ["python", "run.py"]
"""
        (dest_path / "Dockerfile").write_text(dockerfile)

    logger.success("[EXPORT] Strategy exported to {}/", dest_path)
    files = [f"{dest_path}/strategy.py", f"{dest_path}/run.py", f"{dest_path}/requirements.txt"]
    if not no_docker:
        files.append(f"{dest_path}/Dockerfile")
    typer.echo(f"Files: {', '.join(files)}")
    typer.echo(f"Run:   echo '{{\"open\":[...],\"close\":[...],...}}' | python {dest_path}/run.py")
    if not no_docker:
        typer.echo(
            f"Docker: docker build -t my-strategy {dest_path}/ && "
            "echo '{\"open\":[...],...}' | docker run -i my-strategy"
        )


# ---------------------------------------------------------------------------
# returns
# ---------------------------------------------------------------------------


@app.command()
def returns(
    source: Annotated[str, typer.Option(help="Data source name")] = "synthetic",
    source_arg: Annotated[
        list[str] | None, typer.Option(help="Source args as key=value")
    ] = None,
    symbol: Annotated[str | None, typer.Option(help="Filter to symbol")] = None,
) -> None:
    """Show performance report and ASCII equity curve for best strategy."""
    import polars as pl

    from backtest import backtest
    from loop import polars_to_numpy_bars
    from sandbox import run_strategy
    from torture import deflation_test, noise_test, walkforward_test

    if not STRATEGY_FILE.exists():
        logger.error("[RETURNS] No strategy.py found.")
        raise typer.Exit(1)

    source_kwargs = _parse_source_args(source_arg or [])
    data_df = _load_firewalled_data(source, source_kwargs)

    symbols = data_df["symbol"].unique().to_list()
    if symbol:
        data_df = data_df.filter(pl.col("symbol") == symbol)
    else:
        data_df = data_df.filter(pl.col("symbol") == symbols[0])

    bars_np = polars_to_numpy_bars(data_df)
    bars_json = {k: v.tolist() for k, v in bars_np.items()}

    result = run_strategy(STRATEGY_FILE, bars_json)
    if "error" in result:
        logger.error("[RETURNS] Strategy failed: {}", result["error"])
        raise typer.Exit(1)

    pos = np.array(result["positions"])
    close = bars_np["close"]

    bt = backtest(close, pos)
    noise = noise_test(close, pos)
    deflation = deflation_test(close, pos)
    walkforward = walkforward_test(close, pos)

    typer.echo("=== Performance Report ===")
    typer.echo(f"  Sharpe:       {bt['sharpe']}")
    typer.echo(f"  Max Drawdown: {bt['max_drawdown']}")
    typer.echo(f"  Trades:       {bt['trades']}")
    typer.echo(f"  Win Rate:     {bt['win_rate']}")
    typer.echo(f"  Exposure:     {bt['exposure']}")
    typer.echo(f"  Final Equity: {bt['final_equity']}")
    typer.echo("")
    typer.echo("=== Torture Tests ===")
    typer.echo(
        f"  Noise:       {'PASS' if noise['passed'] else 'FAIL'}"
        f" (real={noise['real_sharpe']},"
        f" shuffled={noise['mean_shuffled_sharpe']})"
    )
    typer.echo(
        f"  Deflation:   {'PASS' if deflation['passed'] else 'FAIL'}"
        f" (base={deflation['base_sharpe']},"
        f" deflated={deflation['deflated_sharpe']})"
    )
    typer.echo(
        f"  Walkforward: {'PASS' if walkforward['passed'] else 'FAIL'}"
        f" (pass_rate={walkforward.get('pass_rate', 0.0)},"
        f" folds={len(walkforward.get('folds', []))})"
    )

    # ASCII equity curve (with costs, matching reported metrics)
    cost_frac = 10.0 / 10_000  # same default as backtest
    position_changes = np.abs(np.diff(pos, prepend=0.0))
    strategy_returns = pos * close - position_changes * cost_frac
    equity = np.cumprod(1 + strategy_returns)
    _print_ascii_equity(equity)


def _print_ascii_equity(equity: np.ndarray, width: int = 60, height: int = 10) -> None:
    """Print a simple ASCII equity curve."""
    if len(equity) < 2:
        return

    indices = np.linspace(0, len(equity) - 1, width, dtype=int)
    sampled = equity[indices]

    lo, hi = float(np.min(sampled)), float(np.max(sampled))
    if hi == lo:
        hi = lo + 0.01

    typer.echo("\n=== Equity Curve ===")
    for row in range(height - 1, -1, -1):
        threshold = lo + (hi - lo) * row / (height - 1)
        line = "".join("#" if val >= threshold else " " for val in sampled)
        label = f"{threshold:.2f}" if row in (0, height - 1) else ""
        typer.echo(f"  {label:>6} |{line}|")
    typer.echo(f"         {'_' * width}")


# ---------------------------------------------------------------------------
# reveal
# ---------------------------------------------------------------------------


@app.command()
def reveal(
    source: Annotated[str, typer.Option(help="Data source name")] = "synthetic",
    source_arg: Annotated[
        list[str] | None, typer.Option(help="Source args as key=value")
    ] = None,
    key: Annotated[str | None, typer.Option(help="HMAC key as hex string")] = None,
    key_file: Annotated[
        str | None, typer.Option(help="Path to hex-encoded key file")
    ] = None,
    symbol: Annotated[str | None, typer.Option(help="Filter to anonymized symbol")] = None,
    output: Annotated[str, typer.Option(help="Output format: json or csv")] = "json",
) -> None:
    """De-anonymize strategy output — map positions back to real tickers.

    This is a PRIVILEGED operation that breaks the epistemic firewall.
    Requires the HMAC key used during the original run.
    """
    import polars as pl

    from backtest import backtest
    from firewall import anonymize_dataset, load_key
    from loop import polars_to_numpy_bars
    from sandbox import run_strategy

    if not STRATEGY_FILE.exists():
        logger.error("[REVEAL] No strategy.py found. Run 'auto-research run' first.")
        raise typer.Exit(1)

    # Resolve key
    if key:
        fw_key = bytes.fromhex(key)
    elif key_file:
        fw_key = load_key(Path(key_file))
    elif FIREWALL_KEY_FILE.exists():
        fw_key = load_key(FIREWALL_KEY_FILE)
    else:
        logger.error(
            "[REVEAL] No key provided. Use --key, --key-file, or run 'auto-research run' first."
        )
        raise typer.Exit(1)

    source_kwargs = _parse_source_args(source_arg or [])

    # Load RAW data (before firewall)
    try:
        raw_df = load_raw(source, **source_kwargs)
    except Exception as e:
        logger.error("[REVEAL] Failed to load raw data: {}", e)
        raise typer.Exit(1) from None

    # Anonymize with the provided key to get reverse_map
    anon_df, reverse_map = anonymize_dataset(raw_df, fw_key)

    # Warn prominently
    typer.echo("=" * 60)
    typer.echo("  WARNING: EPISTEMIC FIREWALL BREACHED")
    typer.echo("  Real ticker symbols will be shown below.")
    typer.echo("  Strategy code never saw these — only the output layer")
    typer.echo("  is translating anonymized symbols to real tickers.")
    typer.echo("=" * 60)
    typer.echo("")

    # Run strategy per symbol
    anon_symbols = anon_df["symbol"].unique().to_list()
    if symbol:
        anon_symbols = [s for s in anon_symbols if s == symbol]
        if not anon_symbols:
            logger.error("[REVEAL] Symbol {} not found in anonymized data.", symbol)
            raise typer.Exit(1)

    results = []
    for anon_sym in sorted(anon_symbols):
        sym_df = anon_df.filter(pl.col("symbol") == anon_sym)
        bars_np = polars_to_numpy_bars(sym_df)
        bars_json = {k: v.tolist() for k, v in bars_np.items()}

        result = run_strategy(STRATEGY_FILE, bars_json)
        if "error" in result:
            logger.warning("[REVEAL] Strategy failed on {}: {}", anon_sym, result["error"])
            continue

        pos = np.array(result["positions"])
        close = bars_np["close"]
        bt = backtest(close, pos)

        real_sym = reverse_map.get(anon_sym, anon_sym)
        results.append({
            "real_symbol": real_sym,
            "anon_symbol": anon_sym,
            "current_position": float(pos[-1]),
            "sharpe": bt["sharpe"],
            "trades": bt["trades"],
        })

    if output == "json":
        typer.echo(json.dumps(results, indent=2))
    else:
        typer.echo("real_symbol,anon_symbol,current_position,sharpe,trades")
        for r in results:
            typer.echo(
                f"{r['real_symbol']},{r['anon_symbol']},"
                f"{r['current_position']:.4f},{r['sharpe']},{r['trades']}"
            )


# ---------------------------------------------------------------------------
# compare (multi-symbol)
# ---------------------------------------------------------------------------


@app.command()
def compare(
    source: Annotated[str, typer.Option(help="Data source name")] = "synthetic",
    source_arg: Annotated[
        list[str] | None, typer.Option(help="Source args as key=value")
    ] = None,
    symbols: Annotated[str | None, typer.Option(help="Comma-separated symbols")] = None,
) -> None:
    """Run strategy on multiple symbols and compare results."""
    import polars as pl

    from backtest import backtest
    from loop import polars_to_numpy_bars
    from sandbox import run_strategy

    if not STRATEGY_FILE.exists():
        logger.error("[COMPARE] No strategy.py found.")
        raise typer.Exit(1)

    source_kwargs = _parse_source_args(source_arg or [])
    data_df = _load_firewalled_data(source, source_kwargs)

    available = data_df["symbol"].unique().to_list()
    selected = [s.strip() for s in symbols.split(",")] if symbols else available

    results = []
    for sym in selected:
        sym_df = data_df.filter(pl.col("symbol") == sym)
        if len(sym_df) < 50:
            logger.warning("[COMPARE] {} — {} bars, skipping", sym, len(sym_df))
            continue

        bars_np = polars_to_numpy_bars(sym_df)
        bars_json = {k: v.tolist() for k, v in bars_np.items()}
        result = run_strategy(STRATEGY_FILE, bars_json)

        if "error" in result:
            logger.warning("[COMPARE] {} failed: {}", sym, result["error"][:80])
            continue

        pos = np.array(result["positions"])
        bt = backtest(bars_np["close"], pos)
        bt["symbol"] = sym
        results.append(bt)

    if not results:
        logger.error("[COMPARE] No symbols produced results.")
        raise typer.Exit(1)

    results.sort(key=lambda r: r["sharpe"], reverse=True)

    typer.echo(
        f"{'Symbol':<20} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} "
        f"{'WinRate':>8} {'Exposure':>8} {'Equity':>8}"
    )
    typer.echo("-" * 77)
    for r in results:
        typer.echo(
            f"{r['symbol']:<20} {r['sharpe']:>8.4f} "
            f"{r['max_drawdown']:>8.4f} {r['trades']:>7} "
            f"{r['win_rate']:>8.4f} {r['exposure']:>8.4f} "
            f"{r['final_equity']:>8.4f}"
        )


if __name__ == "__main__":
    app()
