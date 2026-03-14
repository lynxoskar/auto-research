"""Main loop — autonomous strategy discovery on anonymized data.

Inspired by Karpathy's autoresearch: try idea, measure, keep/discard, repeat.
Uses Polars for data handling. Strategies receive numpy arrays (zero-copy from Polars).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import numpy as np
import polars as pl
from loguru import logger

from backtest import backtest
from sandbox import run_strategy
from torture import deflation_test, noise_test, walkforward_test

SKILLS_DIR = Path("skills")

STRATEGY_FILE = Path("strategy.py")
TEMPLATE_FILE = Path("strategy_template.py")
EXPERIMENTS_LOG = Path("experiments.jsonl")
SESSION_FILE = Path("session.md")

BASE_SYSTEM_PROMPT = """\
You are an autonomous trading strategy researcher. You work on anonymized market data \
(symbols are hashed, dates are shifted, prices are normalized to returns). You CANNOT \
identify real tickers or dates. Do not try.

Your job: write a Python strategy function that returns position weights.

The function signature is:
```python
def strategy(bars):
    # bars: dict of numpy arrays (keys depend on data source / skill file)
    # Typical keys: open, high, low, close, volume
    # Each array is 1D (single symbol) or 2D (n_bars x n_symbols)
    # Returns: 1D positions array (length = number of bars)
    #   -1.0 = full short, 0.0 = flat, 1.0 = full long
```

You may use any installed Python packages (numpy, pandas, scipy, sklearn, etc).

Each iteration:
1. I show you the current strategy.py, recent results, and session context.
2. You write an improved strategy.py.
3. I run it, backtest it, torture-test it, and report results.
4. If Sharpe improved AND noise test passed → keep. Otherwise → discard.

Position weights give you full expressiveness: go long, short, partial, or flat. \
Transaction costs scale with position changes — smooth transitions are cheaper than \
binary flips. Exposure (fraction of time positioned) is tracked.

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
returning a 1D positions array (floats: -1.0 short to 1.0 long, 0.0 flat). \
The function receives a single `bars` dict argument."""


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
    new_code = extract_strategy_code(str(response.content[0].text))
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

    positions = np.array(result["positions"], dtype=np.float64)

    bt = backtest(close_returns, positions)
    noise = noise_test(close_returns, positions)
    deflation = deflation_test(close_returns, positions)
    walkforward = walkforward_test(close_returns, positions)

    improved = bt["sharpe"] > best_sharpe
    keep = (
        improved
        and noise["passed"]
        and deflation["passed"]
        and walkforward["passed"]
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
        "walkforward_passed": walkforward["passed"],
        "walkforward_pass_rate": walkforward.get("pass_rate", 0.0),
        "improved": improved,
        "description": (
            f"Sharpe={bt['sharpe']}, trades={bt['trades']}, "
            f"noise={'pass' if noise['passed'] else 'fail'}, "
            f"deflation={'pass' if deflation['passed'] else 'fail'}, "
            f"walkforward={'pass' if walkforward['passed'] else 'fail'}"
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


if __name__ == "__main__":
    from cli import app

    app()
