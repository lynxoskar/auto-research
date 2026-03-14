# auto-research — Project Instructions

## Overview

Autonomous trading strategy discovery. LLM writes Python strategy functions,
sandbox executes them, backtest scores them, torture tests catch overfitting,
loop keeps what works. Data sources are pluggable — each decides its own
transforms (firewall, raw, etc).

## Tech Stack

- Python 3.12+ with uv
- polars, duckdb, numpy (data + backtest)
- anthropic SDK (LLM)
- typer (CLI)
- loguru (logging)
- ruff (lint), ty (type check), pytest (tests)

## Mandatory Quality Gates

**Run after EVERY change:**

```bash
uv run ruff check .          # lint — must pass with zero errors
uv run ty check              # type check — must pass with zero errors
uv run pytest test_smoke.py -v  # smoke tests — all must pass
```

Do not commit if any of these fail.

## Coding Standards

- ruff: line-length=100, select E/F/I/UP/B/SIM
- loguru for logging with prefixes: [DATA], [LOOP], [KEEP], [DISCARD], etc.
- polars over pandas (pandas allowed in sandbox for strategy code)
- Zero-copy awareness: use `allow_copy=True` for polars→numpy, document why
- No print() — use loguru or typer.echo

## Architecture

```
data source (pluggable) → [datasource.py] → data + skill
    → [loop.py] LLM writes strategy.py
    → [sandbox.py] subprocess execution
    → [backtest.py] numpy vectorized
    → [torture.py] noise + deflation + holdout + stress
    → keep (git commit) / discard (git revert)
```

Data sources decide their own transforms. The firewall (firewall.py) is a
utility available to sources — not mandatory in the pipeline.

## CLI Commands

```
auto-research run         # main discovery loop
auto-research status      # session stats
auto-research sources     # list data sources
auto-research positions   # current position weights
auto-research returns     # performance report + equity curve
auto-research profile     # strategy position profiling
auto-research compare     # multi-symbol comparison
auto-research reveal      # de-anonymize positions (privileged)
auto-research export      # standalone project + Dockerfile
auto-research test        # run smoke tests
```

## Strategy Interface

```python
def strategy(bars):
    # bars: dict of numpy arrays (keys depend on data source / skill)
    # Each array is 1D (single symbol) or 2D (n_bars x n_symbols)
    # Returns: 1D positions array, -1.0 = full short, 0.0 = flat, 1.0 = full long
    return positions
```

## Issue Tracking

Use br (beads_rust) for all work tracking:

```bash
br list              # see all issues
br ready             # find unblocked work
br show <id>         # view issue details
br close <id>        # mark complete
```
