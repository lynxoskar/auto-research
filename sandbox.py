"""Sandbox — execute strategy code in a subprocess with timeout and restricted builtins."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = frozenset({
    "numpy", "np", "pandas", "pd", "math", "statistics", "functools", "itertools",
})

# Runner script uses restricted builtins to prevent __import__ escape.
# Data is passed via stdin (not argv) to avoid ARG_MAX limits and /proc visibility.
RUNNER_TEMPLATE = '''
import sys, json

# Import allowed modules FIRST (before any restrictions)
import numpy as np
import pandas as pd
import math, statistics, functools, itertools

# Read data from stdin
data = json.loads(sys.stdin.read())
close = np.array(data["close"])
open_ = np.array(data["open"])
high = np.array(data["high"])
low = np.array(data["low"])
volume = np.array(data["volume"])

# Load strategy code
with open(sys.argv[1]) as f:
    code = f.read()

# Restricted import: only allows modules that are already loaded
_BLOCKED_TOPLEVEL = frozenset({
    "os", "subprocess", "socket", "http", "urllib", "requests", "httpx",
    "shutil", "pathlib", "glob", "importlib", "ctypes", "signal",
    "multiprocessing", "threading", "asyncio", "anthropic", "openai",
})

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

def _guarded_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _BLOCKED_TOPLEVEL:
        raise ImportError(f"Import of '{name}' is blocked in strategy sandbox")
    return _real_import(name, *args, **kwargs)

# Execute strategy with restricted globals (blocks dangerous imports + open())
restricted_globals = {
    "__builtins__": {
        "range": range, "len": len, "int": int, "float": float, "bool": bool,
        "str": str, "list": list, "dict": dict, "tuple": tuple, "set": set,
        "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
        "abs": abs, "max": max, "min": min, "sum": sum, "round": round,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed, "any": any, "all": all,
        "isinstance": isinstance, "issubclass": issubclass, "type": type,
        "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
        "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
        "super": super, "object": object, "slice": slice,
        "print": print, "repr": repr, "id": id, "hash": hash,
        "True": True, "False": False, "None": None,
        "__import__": _guarded_import,
        "__name__": "__main__",
    },
    "np": np, "numpy": np, "pd": pd, "pandas": pd,
    "math": math, "statistics": statistics,
    "functools": functools, "itertools": itertools,
}

exec(compile(code, sys.argv[1], "exec"), restricted_globals)

strategy = restricted_globals.get("strategy")
if strategy is None:
    print(json.dumps({"error": "No strategy() function defined"}))
    sys.exit(1)

entry, exit_ = strategy(open_, high, low, close, volume)
result = {
    "entry": [bool(x) for x in entry],
    "exit": [bool(x) for x in exit_],
}
print(json.dumps(result))
'''


def check_imports(strategy_path: str | Path) -> list[str]:
    """Static check for obviously disallowed imports. Not a security boundary."""
    violations = []
    with open(strategy_path) as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                if stripped.startswith("from "):
                    module = stripped.split()[1].split(".")[0]
                else:
                    module = stripped.split()[1].split(".")[0].rstrip(",")
                if module not in ALLOWED_IMPORTS:
                    violations.append(f"Line {i}: disallowed import '{module}'")
    return violations


def run_strategy(
    strategy_path: str | Path,
    bars: dict,
    timeout_seconds: int = 30,
) -> dict:
    """Execute a strategy file in a subprocess and return entry/exit signals.

    Data is passed via stdin to avoid ARG_MAX limits. Strategy runs with restricted
    builtins that block dynamic imports of unauthorized modules.
    """
    violations = check_imports(strategy_path)
    if violations:
        return {"error": f"Import violations: {'; '.join(violations)}"}

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
