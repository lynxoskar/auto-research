# EvoTradeForge Lite — Implementation Plan

## Core Idea

LLM writes Python strategy functions → sandboxed backtest on anonymized data → measure → keep/discard → loop.

Inspired by Karpathy's autoresearch ("one file, one metric") and pi-autoresearch ("try, measure, keep/discard, repeat forever").

## Architecture

```
Raw OHLCV Parquet
    ↓
[firewall.py] — HMAC anonymize symbols, shift dates, normalize to returns
    ↓
anonymized_bars.parquet (what the LLM sees)
    ↓
[loop.py] — autonomous optimization loop
    ↓
    ├── LLM writes strategy.py (a Python function)
    ├── [sandbox.py] — exec strategy in subprocess with timeout
    ├── [backtest.py] — vectorbt or simple numpy backtest
    ├── [torture.py] — shuffle returns, re-run, compare (noise test)
    ├── Measure: Sharpe, max drawdown, trade count, noise resilience
    ├── Keep / Discard decision
    ├── Log to experiments.jsonl
    └── Loop forever
```

## Files (target: ~500 lines total)

| File | Purpose | Est. Lines |
|------|---------|------------|
| `firewall.py` | HMAC symbol anonymization, date shift, return normalization | ~120 |
| `backtest.py` | Numpy-based vectorized backtest, Sharpe/drawdown/trade metrics | ~100 |
| `torture.py` | Noise test (shuffle returns), deflation test (2x costs) | ~50 |
| `sandbox.py` | Run strategy.py in subprocess with timeout, capture result | ~40 |
| `loop.py` | Main CLI: load data, call LLM, backtest, score, keep/discard | ~120 |
| `strategy_template.py` | Template the LLM fills in / rewrites | ~20 |
| `experiments.jsonl` | Append-only log of every run | generated |
| `session.md` | Living doc: what's been tried, dead ends, best so far | generated |
| `pyproject.toml` | Dependencies: anthropic, numpy, pandas, pyarrow | config |

## Key Design Decisions

1. **No JSON schema.** LLM writes a Python function: `def strategy(bars) -> (entry_signals, exit_signals)`. Full expressiveness.

2. **Firewall is the IP.** HMAC anonymization + date shifting + return normalization. This is the one thing that makes this different from "just ask GPT to write a trading strategy."

3. **Numpy backtest, not vectorbt.** Keep deps minimal. A simple long-only bar-by-bar backtest in numpy is ~60 lines and fast enough.

4. **Subprocess sandbox.** Strategy code runs in a separate process with timeout. No network access. Import allowlist enforced.

5. **LLM decides keep/discard.** Like pi-autoresearch: the LLM sees the metrics, compares to best, decides. No tournament selection. No population. Serial hill-climbing with memory (session.md).

6. **Noise test as the only torture.** Shuffle daily returns, re-run strategy, compare Sharpe. If shuffled Sharpe is close to real Sharpe → strategy is random → discard. Simple, effective, ~20 lines.

7. **Anthropic SDK directly.** No CLI wrappers. `anthropic.Anthropic().messages.create()` with tool use for `run_strategy` and `log_result`.

## Anti-Cheating (Epistemic Firewall)

The one novel idea worth preserving from EvoTradeForge:

```python
# firewall.py
import hmac, hashlib

def anonymize_symbol(symbol: str, key: bytes) -> str:
    h = hmac.new(key, symbol.encode(), hashlib.sha256).hexdigest()[:8]
    return f"Asset_{h.upper()}"

def anonymize_dataset(bars_df, key, date_offset_days=1000):
    df = bars_df.copy()
    df["symbol"] = df["symbol"].map(lambda s: anonymize_symbol(s, key))
    df["timestamp"] = df["timestamp"] - pd.Timedelta(days=date_offset_days)
    # Normalize to returns (destroy absolute price levels)
    for col in ["open", "high", "low", "close"]:
        df[col] = df.groupby("symbol")[col].pct_change().fillna(0)
    df["volume"] = df.groupby("symbol")["volume"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df
```

## Strategy Template

What the LLM writes/modifies:

```python
# strategy.py — LLM edits this file
import numpy as np

def strategy(open, high, low, close, volume):
    """
    Args: numpy arrays of OHLCV returns (anonymized).
    Returns: (entry_signals, exit_signals) as boolean arrays.
    """
    # Example: simple SMA crossover on cumulative returns
    prices = np.cumprod(1 + close)  # reconstruct price-like series
    sma_fast = pd.Series(prices).rolling(20).mean().values
    sma_slow = pd.Series(prices).rolling(50).mean().values

    entry = (sma_fast > sma_slow) & (np.roll(sma_fast, 1) <= np.roll(sma_slow, 1))
    exit = (sma_fast < sma_slow)

    return entry, exit
```

## Backtest Engine

~60 lines of numpy:

```python
def backtest(close_returns, entry, exit, cost_bps=10):
    position = False
    equity = [1.0]
    trades = 0

    for i in range(len(close_returns)):
        if not position and entry[i]:
            position = True
            equity.append(equity[-1] * (1 - cost_bps/10000))
            trades += 1
        elif position and exit[i]:
            position = False
            equity.append(equity[-1] * (1 - cost_bps/10000))
        elif position:
            equity.append(equity[-1] * (1 + close_returns[i]))
        else:
            equity.append(equity[-1])

    returns = np.diff(np.log(equity))
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_dd = np.min(np.array(equity) / np.maximum.accumulate(equity)) - 1
    return {"sharpe": sharpe, "max_drawdown": max_dd, "trades": trades, "final_equity": equity[-1]}
```

## Noise Test

~15 lines:

```python
def noise_test(close_returns, entry_fn, exit_fn, n_shuffles=5, cost_bps=10):
    real = backtest(close_returns, entry_fn(close_returns), exit_fn(close_returns), cost_bps)
    shuffled_sharpes = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(close_returns)
        result = backtest(shuffled, entry_fn(shuffled), exit_fn(shuffled), cost_bps)
        shuffled_sharpes.append(result["sharpe"])
    return real["sharpe"] > np.mean(shuffled_sharpes) * 1.5
```

## Loop (Karpathy/pi-autoresearch style)

```
while True:
    1. Read session.md for context
    2. Ask LLM to improve strategy.py (or write new one)
    3. Run strategy in sandbox with timeout
    4. Backtest on anonymized data
    5. Run noise test
    6. If Sharpe improved AND noise test passed → keep (git commit)
    7. Else → discard (git checkout -- strategy.py)
    8. Log to experiments.jsonl
    9. Update session.md with what was tried
```

## Dependencies

```toml
[project]
name = "auto-research"
requires-python = ">=3.12"
dependencies = [
    "anthropic",
    "numpy",
    "pandas",
    "pyarrow",
]
```

## What We're NOT Building

- No Rust
- No JSON schema / TradePlan
- No Golden Registry
- No indicator engine
- No tournament selection / population
- No skill evolution
- No archive system
- No regime detection
- No multiple agent modes
- No CLI framework
- No research investigations
