"""Sandbox -- execute strategy code in a subprocess with timeout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

RUNNER_TEMPLATE = '''
import sys, json
import numpy as np
import pandas as pd
import math, statistics, functools, itertools

# bars: dict of numpy arrays. Each value is 1D (single symbol) or 2D (n_bars x n_symbols).
# The data source and skill file describe the shape. The strategy decides its own output.
raw = json.loads(sys.stdin.read())
bars = {k: np.array(v) for k, v in raw.items()}

with open(sys.argv[1]) as f:
    code = f.read()

g = {}
exec(compile(code, sys.argv[1], "exec"), g)

strategy = g.get("strategy")
if strategy is None:
    print(json.dumps({"error": "No strategy() function defined"}))
    sys.exit(1)

positions = strategy(bars)
positions = np.asarray(positions, dtype=float)

# Positions must be 1D with length matching the first dimension of any bar array
n_bars = next(iter(bars.values())).shape[0]
if positions.ndim != 1 or positions.shape[0] != n_bars:
    print(json.dumps({
        "error": f"positions shape {positions.shape} incompatible with {n_bars} bars"
    }))
    sys.exit(1)
print(json.dumps({"positions": [float(x) for x in positions]}))
'''


def run_strategy(
    strategy_path: str | Path,
    bars: dict,
    timeout_seconds: int = 30,
) -> dict:
    """Execute a strategy file in a subprocess and return position weights.

    Args:
        strategy_path: Path to the strategy .py file.
        bars: Dict of OHLCV arrays (1D for single symbol, 2D for multi-symbol).
        timeout_seconds: Max execution time.

    Returns:
        Dict with "positions" (list of floats) or "error" (str).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(RUNNER_TEMPLATE)
        runner_path = tmp.name

    # Serialize — handle both 1D and 2D numpy arrays
    serializable = {}
    for k, v in bars.items():
        if hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = [float(x) for x in v]
    data_json = json.dumps(serializable)

    try:
        result = subprocess.run(
            [sys.executable, runner_path, str(strategy_path)],
            input=data_json,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={"PATH": "", "PYTHONPATH": "", "HOME": ""},
        )

        if result.returncode != 0:
            stderr = result.stderr[-500:] if result.stderr else "unknown error"
            return {"error": f"Strategy crashed: {stderr}"}

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return {"error": f"Strategy timed out after {timeout_seconds}s"}
    except json.JSONDecodeError:
        stdout = result.stdout[-200:] if result.stdout else "empty"
        return {"error": f"Strategy produced invalid output: {stdout}"}
    except Exception as e:
        return {"error": f"Sandbox error: {e}"}
    finally:
        Path(runner_path).unlink(missing_ok=True)
