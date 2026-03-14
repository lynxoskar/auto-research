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

data = json.loads(sys.stdin.read())
close = np.array(data["close"])
open_ = np.array(data["open"])
high = np.array(data["high"])
low = np.array(data["low"])
volume = np.array(data["volume"])

with open(sys.argv[1]) as f:
    code = f.read()

g = {}
exec(compile(code, sys.argv[1], "exec"), g)

strategy = g.get("strategy")
if strategy is None:
    print(json.dumps({"error": "No strategy() function defined"}))
    sys.exit(1)

positions = strategy(open_, high, low, close, volume)
positions = np.asarray(positions, dtype=float)
if positions.shape != close.shape:
    print(json.dumps({
        "error": f"positions shape {positions.shape} != data shape {close.shape}"
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

    Data is passed via stdin to avoid ARG_MAX limits.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(RUNNER_TEMPLATE)
        runner_path = tmp.name

    data_json = json.dumps({k: [float(x) for x in v] for k, v in bars.items()})

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
