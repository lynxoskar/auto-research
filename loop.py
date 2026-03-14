"""Main loop — autonomous strategy discovery on anonymized data.

Inspired by Karpathy's autoresearch: try idea, measure, keep/discard, repeat.
Uses Polars for data handling. Strategies receive numpy arrays (zero-copy from Polars).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import numpy as np
import polars as pl
from loguru import logger

from backtest import backtest
from datasource import list_sources, load_data
from sandbox import run_strategy
from torture import deflation_test, noise_test

SKILLS_DIR = Path("skills")

STRATEGY_FILE = Path("strategy.py")
TEMPLATE_FILE = Path("strategy_template.py")
EXPERIMENTS_LOG = Path("experiments.jsonl")
SESSION_FILE = Path("session.md")

BASE_SYSTEM_PROMPT = """\
You are an autonomous trading strategy researcher. You work on anonymized market data \
(symbols are hashed, dates are shifted, prices are normalized to returns). You CANNOT \
identify real tickers or dates. Do not try.

Your job: write a Python strategy function that generates profitable entry/exit signals.

The function signature is:
```python
def strategy(open, high, low, close, volume):
    # open/high/low/close: numpy arrays of percentage returns
    # volume: numpy array of z-score normalized volume
    # Returns: (entry_signals, exit_signals) as boolean arrays
```

You may use: numpy, pandas, math, statistics.
You may NOT use: any other imports (no sklearn, scipy, ta-lib, etc).

Each iteration:
1. I show you the current strategy.py, recent results, and session context.
2. You write an improved strategy.py.
3. I run it, backtest it, torture-test it, and report results.
4. If Sharpe improved AND noise test passed → keep. Otherwise → discard.

Be creative. Try different approaches: momentum, mean reversion, volatility breakouts, \
volume-weighted signals, regime detection via rolling stats. When stuck, try something \
structurally different.

IMPORTANT: Do not overfit to the benchmark. Strategies must survive noise testing \
(shuffled returns should produce worse Sharpe than real returns).\
"""


def build_system_prompt() -> str:
    """Build system prompt with any loaded skill files appended."""
    prompt = BASE_SYSTEM_PROMPT
    if SKILLS_DIR.exists():
        for skill_path in sorted(SKILLS_DIR.glob("*.md")):
            skill_content = skill_path.read_text().strip()
            if skill_content:
                prompt += f"\n\n---\n## Skill: {skill_path.stem}\n\n{skill_content}"
                logger.debug("[SKILL] Loaded {}", skill_path.name)
    return prompt


def load_session_context() -> str:
    """Load session.md if it exists."""
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text()
    return "No previous session. This is the first run."


def load_recent_experiments(n: int = 10) -> list[dict]:
    """Load the last n experiments from the JSONL log."""
    if not EXPERIMENTS_LOG.exists():
        return []
    lines = EXPERIMENTS_LOG.read_text().strip().split("\n")
    experiments = []
    for line in lines[-n:]:
        if line.strip():
            experiments.append(json.loads(line))
    return experiments


def log_experiment(result: dict) -> None:
    """Append a result to the JSONL log."""
    result["timestamp"] = datetime.now(UTC).isoformat()
    with open(EXPERIMENTS_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")


def update_session(result: dict, strategy_code: str) -> None:
    """Update session.md with latest result."""
    experiments = load_recent_experiments(100)
    kept = [e for e in experiments if e.get("status") == "keep"]
    discarded = [e for e in experiments if e.get("status") == "discard"]

    best = max(kept, key=lambda e: e.get("sharpe", 0)) if kept else None

    content = f"""# Auto-Research: Strategy Discovery

## Status
- Total runs: {len(experiments)}
- Kept: {len(kept)}
- Discarded: {len(discarded)}
- Best Sharpe: {best['sharpe'] if best else 'N/A'}
- Best max drawdown: {best.get('max_drawdown', 'N/A') if best else 'N/A'}

## Current Best Strategy
```python
{best.get('strategy_code', 'None yet') if best else 'None yet'}
```

## Recent Results (last 10)
"""
    for exp in experiments[-10:]:
        status = exp.get("status", "?")
        sharpe = exp.get("sharpe", "?")
        trades = exp.get("trades", "?")
        noise = "pass" if exp.get("noise_passed") else "fail"
        content += (
            f"- [{status}] Sharpe={sharpe}, trades={trades}, "
            f"noise={noise}: {exp.get('description', '')}\n"
        )

    content += "\n## What's Been Tried\n"
    descriptions = [e.get("description", "") for e in experiments if e.get("description")]
    for desc in descriptions[-20:]:
        content += f"- {desc}\n"

    SESSION_FILE.write_text(content)


def build_user_message(current_strategy: str, recent: list[dict], session: str) -> str:
    """Build the user message for the LLM."""
    recent_text = ""
    for exp in recent[-5:]:
        recent_text += (
            f"  [{exp.get('status')}] Sharpe={exp.get('sharpe')}, "
            f"trades={exp.get('trades')}, "
            f"noise={'pass' if exp.get('noise_passed') else 'fail'}: "
            f"{exp.get('description', '')}\n"
        )

    return f"""## Current strategy.py
```python
{current_strategy}
```

## Recent results
{recent_text if recent_text else '  No results yet — this is the first run.'}

## Session context
{session}

Write an improved strategy.py. Output ONLY the Python code (no markdown fences, \
no explanation). The file must define `def strategy(open, high, low, close, volume)` \
returning `(entry, exit_)` boolean arrays."""


def extract_strategy_code(response_text: str) -> str:
    """Extract Python code from LLM response, stripping markdown fences if present."""
    text = response_text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def polars_to_numpy_bars(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Convert Polars DataFrame columns to numpy arrays."""
    return {
        "open": df["open"].to_numpy(allow_copy=True),
        "high": df["high"].to_numpy(allow_copy=True),
        "low": df["low"].to_numpy(allow_copy=True),
        "close": df["close"].to_numpy(allow_copy=True),
        "volume": df["volume"].to_numpy(allow_copy=True),
    }


def run_iteration(
    client: anthropic.Anthropic,
    bars_np: dict[str, np.ndarray],
    close_returns: np.ndarray,
    best_sharpe: float,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run one iteration of the loop."""
    current_code = (
        STRATEGY_FILE.read_text() if STRATEGY_FILE.exists() else TEMPLATE_FILE.read_text()
    )
    session = load_session_context()
    recent = load_recent_experiments(10)

    user_msg = build_user_message(current_code, recent, session)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": user_msg}],
    )

    if not response.content or not hasattr(response.content[0], "text"):
        raise RuntimeError("LLM returned empty or non-text response")
    new_code = extract_strategy_code(response.content[0].text)
    STRATEGY_FILE.write_text(new_code)

    # Prepare bars as JSON-serializable lists for sandbox subprocess
    bars_json = {k: v.tolist() for k, v in bars_np.items()}
    result = run_strategy(STRATEGY_FILE, bars_json, timeout_seconds=30)

    if "error" in result:
        experiment = {
            "status": "crash",
            "error": result["error"],
            "description": "Strategy crashed or timed out",
            "sharpe": 0,
            "trades": 0,
            "strategy_code": new_code[:500],
        }
        _revert_strategy()
        log_experiment(experiment)
        update_session(experiment, new_code)
        return experiment

    entry = np.array(result["entry"], dtype=bool)
    exit_ = np.array(result["exit"], dtype=bool)

    bt = backtest(close_returns, entry, exit_)
    noise = noise_test(close_returns, entry, exit_)
    deflation = deflation_test(close_returns, entry, exit_)

    improved = bt["sharpe"] > best_sharpe
    keep = (
        improved
        and noise["passed"]
        and deflation["passed"]
        and bt["trades"] >= 10
    )

    experiment = {
        "status": "keep" if keep else "discard",
        "sharpe": bt["sharpe"],
        "max_drawdown": bt["max_drawdown"],
        "trades": bt["trades"],
        "final_equity": bt["final_equity"],
        "win_rate": bt["win_rate"],
        "noise_passed": noise["passed"],
        "noise_ratio": noise["ratio"],
        "deflation_passed": deflation["passed"],
        "deflation_sharpe": deflation["deflated_sharpe"],
        "improved": improved,
        "description": (
            f"Sharpe={bt['sharpe']}, trades={bt['trades']}, "
            f"noise={'pass' if noise['passed'] else 'fail'}, "
            f"deflation={'pass' if deflation['passed'] else 'fail'}"
        ),
        "strategy_code": new_code[:1000],
    }

    if keep:
        add_result = subprocess.run(
            ["git", "add", "strategy.py"], capture_output=True, check=False
        )
        if add_result.returncode != 0:
            logger.warning("[GIT] git add failed: {}", add_result.stderr.decode()[:100])
        commit_result = subprocess.run(
            ["git", "commit", "-m", f"keep: Sharpe={bt['sharpe']:.4f} trades={bt['trades']}"],
            capture_output=True,
            check=False,
        )
        if commit_result.returncode != 0:
            logger.warning("[GIT] git commit failed: {}", commit_result.stderr.decode()[:100])
    else:
        _revert_strategy()

    log_experiment(experiment)
    update_session(experiment, new_code)
    return experiment


def _revert_strategy() -> None:
    """Revert strategy.py to last committed version or template."""
    result = subprocess.run(
        ["git", "checkout", "--", "strategy.py"], capture_output=True
    )
    if result.returncode != 0 and TEMPLATE_FILE.exists():
        shutil.copy(TEMPLATE_FILE, STRATEGY_FILE)


def main() -> None:
    """Run the autonomous strategy discovery loop."""
    import argparse

    available = list_sources()

    parser = argparse.ArgumentParser(description="Autonomous strategy discovery")
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help=f"Data source name ({', '.join(available)})",
    )
    parser.add_argument("--symbol", type=str, help="Filter to single anonymized symbol")
    parser.add_argument(
        "--source-arg",
        action="append",
        default=[],
        help="Extra source args as key=value (e.g. --source-arg path=data.parquet)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=0, help="Max iterations (0=infinite)"
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    # Parse source kwargs
    source_kwargs = {}
    for arg in args.source_arg:
        if "=" not in arg:
            logger.error("[DATA] --source-arg must be key=value, got: {}", arg)
            sys.exit(1)
        k, v = arg.split("=", 1)
        source_kwargs[k] = v

    # Load firewalled data via pluggable source
    logger.info("[DATA] Loading from source '{}' {}", args.source, source_kwargs or "")
    try:
        data_df = load_data(args.source, **source_kwargs)
    except Exception as e:
        logger.error("[DATA] Failed to load data: {}", e)
        sys.exit(1)

    # Filter to single symbol (use anonymized names since data is already anonymized)
    symbols = data_df["symbol"].unique().to_list()
    if args.symbol:
        # --symbol takes an anonymized name (e.g. Asset_B071EBDB)
        data_df = data_df.filter(pl.col("symbol") == args.symbol)
    else:
        data_df = data_df.filter(pl.col("symbol") == symbols[0])
    logger.info("[DATA] Using symbol: {} ({} bars)", data_df["symbol"][0], len(data_df))

    if len(data_df) < 50:
        logger.error("[DATA] Too few bars ({}) after filtering. Need at least 50.", len(data_df))
        sys.exit(1)

    # Convert to numpy for backtest/sandbox
    # Note: zero_copy_only=False because pct_change columns may have been cloned.
    # After firewall transform, columns are null-free floats — copies are cheap.
    bars_np = polars_to_numpy_bars(data_df)
    close_returns = bars_np["close"]

    # Ensure git repo exists
    if not Path(".git").exists():
        subprocess.run(["git", "init"], capture_output=True, check=False)

    # Initialize strategy file
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

    logger.info("[LOOP] Auto-Research: Strategy Discovery")
    logger.info("[LOOP] Data: {} bars, model: {}", len(close_returns), args.model)
    logger.info("[LOOP] Loop forever. Ctrl+C to stop.")

    while True:
        iteration += 1
        if 0 < args.max_iterations < iteration:
            logger.info("[LOOP] Reached max iterations ({}). Stopping.", args.max_iterations)
            break

        logger.info("[LOOP] Iteration {} (best Sharpe: {:.4f})", iteration, best_sharpe)

        try:
            result = run_iteration(client, bars_np, close_returns, best_sharpe, args.model)
        except KeyboardInterrupt:
            logger.info("[LOOP] Interrupted. Results saved to experiments.jsonl")
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


if __name__ == "__main__":
    main()
