# auto-research — System Plan

## What This Is

LLM writes Python strategy functions → sandboxed backtest on market data → measure →
keep/discard → loop forever. Data sources are pluggable. Skills teach the LLM about
available data.

Inspired by Karpathy's autoresearch and pi-autoresearch.

## Architecture

```
data source (pluggable, each decides own transforms)
    ↓
[datasource.py] — registry of sources, each returns Polars DataFrame
    ↓
data + skill (markdown describing schema for LLM)
    ↓
[loop.py] — ask LLM to write/improve strategy.py
    ↓
[sandbox.py] — execute in subprocess, restricted builtins
    ↓
[backtest.py] — numpy: Sharpe, drawdown, trades, win rate
    ↓
[torture.py] — noise test + deflation test
    ↓
keep (git commit) or discard (git revert)
    ↓
repeat forever
```

## Current Stack

- **Data:** polars, duckdb, pyarrow
- **LLM:** anthropic SDK (direct, no CLI wrappers)
- **Execution:** subprocess sandbox with restricted builtins
- **Backtest:** numpy (bar-by-bar, ~100 lines)
- **Logging:** loguru
- **Tests:** pytest (24 smoke tests)

## Design Principles

1. **The LLM is the optimizer.** No tournament selection, no population, no genetic
   algorithms. Serial hill-climbing with memory (session.md) is sufficient.

2. **Code enforces what code must.** Safety (sandbox), measurement (backtest, torture),
   and data delivery (sources). Everything else is the LLM's job.

3. **Skills over code.** Teach the LLM via markdown skill files instead of building
   elaborate Rust/Python machinery. The LLM handles selection, mutation, and knowledge
   management natively through session.md.

4. **Data sources own their schema.** Each source decides whether to anonymize, what
   columns to expose, and what skill file to provide. The loop is source-agnostic.
