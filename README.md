# auto-research

Autonomous trading strategy discovery with pluggable data sources.

An LLM agent explores data, forms hypotheses, tests strategies, and iterates —
keeping what works, discarding what doesn't. Data sources are pluggable and
each decides its own schema and transforms.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
and [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch).

## How it works

### Agent mode (default)

Each iteration is a multi-step **research session**. The LLM gets three tools:

```
explore(code)     → run analysis code on 70% of data, see output
backtest(code)    → test a strategy variant, get metrics (max 5 per iteration)
submit(code)      → commit best strategy for full torture testing on 100% of data
```

```
data source (pluggable)
    ↓
[datasource] — source decides its own transforms
    ↓
[agent loop] — LLM explores, hypothesizes, tests, refines
    ↓
    ├── explore → see data patterns (70% of bars)
    ├── backtest → test strategy variants (full data)
    └── submit → final strategy
         ↓
    [torture] — noise + deflation + walk-forward (100% of data)
         ↓
    keep (git commit) or discard (git revert)
         ↓
    repeat forever
```

### Oneshot mode

Single-shot code generation for cheaper iterations: `--oneshot` flag.

## Quick start

```bash
# Install
uv sync

# Run with agent mode (default) on synthetic data
export ANTHROPIC_API_KEY=sk-...
uv run python cli.py run --source synthetic

# Oneshot mode (cheaper, less effective)
uv run python cli.py run --source synthetic --oneshot

# Local parquet file
uv run python cli.py run --source parquet --source-arg path=data/bars.parquet

# DuckLake (needs AWS credentials)
uv run python cli.py run --source ducklake

# Limit iterations
uv run python cli.py run --source synthetic --max-iterations 5
```

## CLI commands

```
auto-research run         # main loop (agent mode by default, --oneshot available)
auto-research status      # session stats, best strategy, recent results
auto-research positions   # current position weights (JSON/CSV)
auto-research returns     # performance report + ASCII equity curve + torture results
auto-research compare     # run strategy on multiple symbols, comparison table
auto-research reveal      # de-anonymize positions (requires firewall key)
auto-research export      # standalone project + Dockerfile
```

## Pluggable data sources

Each source returns a Polars DataFrame. Sources decide their own transforms —
some apply the epistemic firewall, others deliver raw data. The skill file
tells the LLM what schema to expect.

```python
from datasource import register_source, load_data

def my_source(**kwargs):
    return pl.read_parquet("my_data.parquet")

register_source("mydata", my_source)
df = load_data("mydata")
```

Built-in sources:

| Source | Description | Firewall | Credentials |
|--------|-------------|----------|-------------|
| `synthetic` | Generated trending/mean-reverting data | Yes (default) | None |
| `parquet` | Local parquet file | No (raw) | None |
| `ducklake` | DuckLake catalog on S3 (Lynx universe) | Yes (default) | AWS |

## Skills

Markdown files in `skills/` are injected into the LLM system prompt. Each data
source should have a skill describing its schema. Drop a `.md` in `skills/` and
it's loaded on next iteration.

## Strategy format

The LLM writes a Python function that receives a `bars` dict and returns position
weights. The data source decides what's in `bars` — single series or multiple.

```python
def strategy(bars):
    """
    bars: dict of numpy arrays (1D or 2D, schema from data source skill).
    Returns: 1D positions array. -1.0 = short, 0.0 = flat, 1.0 = long.
    """
    close = bars["close"]
    prices = np.cumprod(1 + close)
    fast = pd.Series(prices).rolling(20).mean().values
    slow = pd.Series(prices).rolling(50).mean().values
    return np.where(fast > slow, 1.0, 0.0)
```

Strategies can use any installed Python package.

## Torture tests

Three tests, all must pass:

- **Noise test** — shuffle returns, re-run with same positions. Real Sharpe must
  beat shuffled significantly. Random seed (not gameable).
- **Deflation test** — double transaction costs. Must still be profitable.
- **Walk-forward test** — 5-fold rolling validation. Must be profitable in >50% of
  test windows.

## Anti-overfitting safeguards

In agent mode:
- **Explore sees 70% of data**, torture tests run on 100%
- **Max 5 backtests per iteration** — prevents hill-climbing
- **Random noise seed** — can't game the shuffle test
- **Walk-forward validation** — catches time-dependent overfitting

## Project structure

```
auto-research/
  cli.py               # Typer CLI: 7 commands
  loop.py              # Agent + oneshot loops, tool-use, system prompts
  backtest.py          # Numpy vectorized backtest engine
  datasource.py        # Pluggable data source registry
  firewall.py          # Epistemic firewall (optional per-source)
  sandbox.py           # Subprocess execution + explore runner
  torture.py           # Noise, deflation, walk-forward tests
  strategy_template.py # Baseline SMA crossover strategy
  test_smoke.py        # 38 smoke tests
  skills/              # LLM skill files (injected into system prompt)
  CLAUDE.md            # Agent instructions + mandatory quality gates
```

## Quality gates

```bash
uv run ruff check .             # lint
uv run ty check                 # type check
uv run pytest test_smoke.py -v  # 38 smoke tests
```

All three must pass before every commit.

## Session persistence

- `experiments.jsonl` — append-only log of every iteration
- `session.md` — living document: what's been tried, dead ends, best result
- `git history` — each kept strategy is a commit; discards are reverted
- `.firewall_key` — HMAC key for de-anonymization (gitignored, 0600 perms)

## Design philosophy

The LLM handles exploration, selection, and knowledge management natively.
Code enforces only what code must enforce:

1. **Measurement** — backtest and torture tests must be correct
2. **Safety** — subprocess timeout and crash isolation
3. **Data delivery** — sources provide data, skills describe it
4. **Anti-overfitting** — explore/backtest data split, random seeds, walk-forward

## Planned features

| Feature | Status | Description |
|---------|--------|-------------|
| Token oracle | Deferred (Mar 28) | Multi-provider LLM budget management |
| Real-time positions | Planned | Live data feeds, scheduled position output |
