# auto-research

Autonomous trading strategy discovery with pluggable data sources.

An LLM writes Python strategy functions, a sandbox executes them, a backtest engine
scores them, torture tests catch overfitting, and the loop keeps what works. Data
sources are pluggable — each source decides how to deliver data and describes its
own schema via skill files.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
and [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch).

## How it works

```
data source (pluggable)
    |
    v
[datasource.py] -- source decides its own transforms (firewall, raw, etc.)
    |
    v
data + skill file (describes what the LLM sees)
    |
    v
[loop] -- ask LLM to write strategy.py
    |
    v
[sandbox] -- execute in subprocess with restricted builtins
    |
    v
[backtest] -- Sharpe, drawdown, trades, win rate
    |
    v
[torture] -- noise test + deflation test
    |
    v
keep (git commit) or discard (git revert)
    |
    v
repeat forever
```

## Quick start

```bash
# Install
uv sync

# Run on synthetic data (needs ANTHROPIC_API_KEY for LLM)
export ANTHROPIC_API_KEY=sk-...
uv run python loop.py --source synthetic

# Run on a local parquet file
uv run python loop.py --source parquet --source-arg path=data/bars.parquet

# Run on DuckLake (needs AWS credentials, applies epistemic firewall)
uv run python loop.py --source ducklake

# Limit iterations
uv run python loop.py --source synthetic --max-iterations 5
```

## Pluggable data sources

Each source is a function that returns a Polars DataFrame with OHLCV columns.
Sources decide their own transforms — some apply the epistemic firewall (anonymize
symbols, shift dates, normalize to returns), others deliver raw data. The
accompanying skill file tells the LLM what schema to expect.

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

Skills are markdown files in `skills/` that get injected into the LLM system prompt.
They teach the LLM about available data, schemas, and constraints. Each data source
should have a corresponding skill that describes what the LLM receives.

Drop a `.md` file in `skills/` and it's automatically loaded on next iteration.

See `skills/ducklake_data.md` for an example.

## The epistemic firewall

Available as an optional transform for data sources that need it. LLMs have memorized
financial history — if they see real tickers and dates, they exploit hindsight.

| Transform | What it does |
|-----------|-------------|
| Symbol anonymization | HMAC-SHA256 hash: `AAPL` becomes `Asset_B071EBDB` |
| Date shifting | All dates offset by a configurable amount |
| Price normalization | Absolute prices replaced with percentage returns |
| Volume normalization | Raw volume replaced with z-score (mean=0, std=1) |

Sources opt into the firewall individually. Synthetic and DuckLake apply it by default.
Raw parquet does not.

## Strategy format

The LLM writes a Python function that receives a `bars` dict and returns position
weights. The data source decides what's in `bars` — single series or multiple.

```python
def strategy(bars):
    """
    Args: bars dict of numpy arrays (schema described by data source skill).
          Each array is 1D (single symbol) or 2D (n_bars x n_symbols).
    Returns: 1D positions array (length = number of bars).
             -1.0 = full short, 0.0 = flat, 1.0 = full long.
    """
    close = bars["close"]
    prices = np.cumprod(1 + close)
    fast = pd.Series(prices).rolling(20).mean().values
    slow = pd.Series(prices).rolling(50).mean().values
    return np.where(fast > slow, 1.0, 0.0)
```

The strategy can use any installed Python package. The subprocess provides
timeout and crash isolation.

## Torture tests

Every strategy must survive two tests before being kept:

**Noise test:** Shuffle the return series, re-run the strategy with the same signals.
If the real Sharpe isn't significantly better than the shuffled Sharpe, the strategy
is just finding patterns in noise — discard.

**Deflation test:** Double the transaction costs. If the strategy is no longer
profitable, it doesn't have enough edge to survive real-world friction — discard.

## Project structure

```
auto-research/
  cli.py               # Typer CLI: 10 commands
  loop.py              # Main loop: LLM -> sandbox -> backtest -> torture
  backtest.py          # Numpy vectorized backtest engine
  datasource.py        # Pluggable data source registry
  firewall.py          # Epistemic firewall (optional per-source)
  sandbox.py           # Restricted subprocess execution
  profiler.py          # Strategy position profiling
  torture.py           # Noise, deflation, holdout, stress tests
  strategy_template.py # Baseline SMA crossover strategy
  test_smoke.py        # 43 smoke tests
  skills/              # LLM skill files (injected into system prompt)
    ducklake_data.md   # Data schema and source documentation
  CLAUDE.md            # Agent instructions + mandatory quality gates
  pyproject.toml       # Dependencies
```

## Quality gates

```bash
uv run ruff check .             # lint
uv run ty check                 # type check
uv run pytest test_smoke.py -v  # 43 smoke tests
```

All three must pass before every commit.

## Session persistence

The system persists state across context resets:

- `experiments.jsonl` — append-only log of every iteration
- `session.md` — living document: what's been tried, dead ends, best result
- `git history` — each kept strategy is a commit; discards are reverted
- `.firewall_key` — HMAC key for de-anonymization (gitignored, 0600 perms)

A fresh LLM can read `session.md` and continue where the last one left off.

## Design philosophy

The LLM handles selection, mutation, and knowledge management natively. Code enforces
only what code must enforce:

1. **Execution safety** — strategy code must not escape the sandbox
2. **Measurement** — the backtest and torture tests must be correct
3. **Data delivery** — sources provide data, skills describe it

Everything else is the LLM's job.

## Planned features

| Feature | Status | Description |
|---------|--------|-------------|
| Token oracle | Deferred (Mar 28) | Multi-provider LLM budget management, end-of-week token harvesting |
| OS-level sandbox | Planned | nsjail/bubblewrap for strategy execution (current: restricted builtins) |
| Walk-forward v2 | Planned | Multiple holdout windows instead of single 80/20 split |
| Real-time positions | Planned | Connect to live data feeds, output positions on schedule |
