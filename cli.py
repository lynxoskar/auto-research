"""CLI — drive auto-research from the command line.

Usage:
    auto-research run --source synthetic --max-iterations 10
    auto-research status
    auto-research sources
    auto-research positions --source synthetic
    auto-research export ./my-strategy/
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from loguru import logger

from datasource import list_sources, load_data

app = typer.Typer(
    name="auto-research",
    help="Autonomous trading strategy discovery with pluggable data sources.",
    no_args_is_help=True,
)

STRATEGY_FILE = Path("strategy.py")
TEMPLATE_FILE = Path("strategy_template.py")
EXPERIMENTS_LOG = Path("experiments.jsonl")
SESSION_FILE = Path("session.md")


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
        data_df = load_data(source, **source_kwargs)
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
# sources
# ---------------------------------------------------------------------------


@app.command()
def sources() -> None:
    """List available data sources."""
    for name in list_sources():
        typer.echo(name)


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
    data_df = load_data(source, **source_kwargs)

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
    data = json.loads(sys.stdin.read())
    positions = strategy(
        np.array(data["open"]),
        np.array(data["high"]),
        np.array(data["low"]),
        np.array(data["close"]),
        np.array(data["volume"]),
    )
    print(json.dumps({"positions": [float(p) for p in positions]}))


if __name__ == "__main__":
    main()
'''
    (dest_path / "run.py").write_text(runner)
    (dest_path / "requirements.txt").write_text("numpy\npandas\n")

    logger.success("[EXPORT] Strategy exported to {}/", dest_path)
    typer.echo(f"Files: {dest_path}/strategy.py, {dest_path}/run.py, {dest_path}/requirements.txt")
    typer.echo(f"Run:   echo '{{\"open\":[...],\"close\":[...],...}}' | python {dest_path}/run.py")


if __name__ == "__main__":
    app()
