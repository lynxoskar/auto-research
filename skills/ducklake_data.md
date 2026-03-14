# Skill: DuckLake Anonymized Market Data

## What This Is

You have access to anonymized market data from a DuckLake catalog. The data has been passed
through an **epistemic firewall** — you cannot identify real tickers, real dates, or real prices.
Do not attempt to reverse the anonymization.

## Data Schema

After firewalling, you receive a Polars DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Anonymized symbol, e.g. `Asset_B071EBDB` (HMAC hash, not real ticker) |
| `timestamp` | date | Shifted date (offset from real dates, do not try to reverse) |
| `open` | f64 | Open-to-previous-close percentage return |
| `high` | f64 | High-to-previous-high percentage return |
| `low` | f64 | Low-to-previous-low percentage return |
| `close` | f64 | Close-to-previous-close percentage return |
| `volume` | f64 | Z-score normalized volume (mean=0, std=1) |

**Critical:** These are **returns**, not prices. To reconstruct a price-like series for
indicators that need prices (SMA, Bollinger, etc.), use `np.cumprod(1 + close)`.

## Available Data Sources

Use the `load_data` tool to load firewalled data. Sources:

### `synthetic` (default, no credentials needed)
```
load_data(source="synthetic", symbols=["A", "B"], bars_per_symbol=500)
```
Generated trending/mean-reverting data. Good for testing strategies.

### `parquet` (local file)
```
load_data(source="parquet", path="/path/to/bars.parquet")
```
Expects columns: symbol, timestamp, open, high, low, close, volume (raw prices).
Automatically firewalled before you see it.

### `ducklake` (production, needs AWS credentials)
```
load_data(source="ducklake")
load_data(source="ducklake", symbol="AAPL")  # single symbol
```
Connects to `s3://lynx-sandbox-agent-datalake/catalog.ducklake` (eu-north-1).
Table: `lake.main.lynx_minutebars` — ~14,000 anonymized symbols, 20+ years, ~1.16B rows.

**You receive the data AFTER firewalling.** You never see real tickers or prices.

## What You Can Do

1. **Explore the data** — look at return distributions, autocorrelation, volume patterns
2. **Write strategy functions** — `def strategy(open, high, low, close, volume) -> positions`
3. **Use standard indicators** — SMA, EMA, RSI, Bollinger, MACD, ATR on reconstructed prices
4. **Use numpy and pandas** — no other libraries allowed in strategy code

## What You CANNOT Do

- Identify real tickers (symbols are HMAC hashes)
- Reverse the date shift (you don't know the offset)
- Access absolute price levels (only returns)
- Import os, subprocess, socket, or any network/filesystem libraries
- Call external APIs from strategy code

## Strategy Evaluation

Your strategy returns **position weights** per bar:
- `-1.0` = full short, `0.0` = flat, `1.0` = full long
- Fractional values allowed (e.g. `0.5` = half position)
- Transaction costs scale with position changes — smooth transitions are cheaper

Evaluation criteria:
- **Sharpe ratio** — risk-adjusted return (higher is better)
- **Noise test** — shuffle returns, re-run: real Sharpe must beat shuffled significantly
- **Deflation test** — double transaction costs: strategy must still be profitable
- **Trade count** — minimum 10 position changes required
- **Exposure** — fraction of time with a non-trivial position
