"""Epistemic Firewall — anonymize market data so LLMs cannot cheat.

Transforms raw OHLCV data into anonymized returns:
- HMAC-based symbol anonymization (deterministic, non-reversible without key)
- Date shifting by a fixed offset
- Price normalization to percentage returns (destroys absolute levels)
- Volume z-score normalization

Uses Polars for zero-copy columnar operations and DuckDB for parquet ingestion.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl


def generate_key() -> bytes:
    """Generate a random 32-byte HMAC key."""
    return secrets.token_bytes(32)


def anonymize_symbol(symbol: str, key: bytes) -> str:
    """Deterministic symbol anonymization via HMAC-SHA256."""
    h = hmac.new(key, symbol.encode(), hashlib.sha256).hexdigest()[:8]
    return f"Asset_{h.upper()}"


def anonymize_dataset(
    df: pl.DataFrame,
    key: bytes,
    date_offset_days: int = 1000,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Anonymize a raw OHLCV Polars DataFrame.

    Expects columns: symbol, timestamp, open, high, low, close, volume.
    Returns (anonymized_df, reverse_map).
    """
    # Build symbol mapping
    unique_symbols = df["symbol"].unique().to_list()
    symbol_map = {s: anonymize_symbol(s, key) for s in unique_symbols}
    reverse_map = {v: k for k, v in symbol_map.items()}

    # Sort for consistent pct_change
    df = df.sort(["symbol", "timestamp"])

    # Apply transformations using Polars expressions (zero-copy where possible)
    result = df.with_columns(
        # Anonymize symbols
        pl.col("symbol").replace_strict(symbol_map).alias("symbol"),
        # Shift dates
        (pl.col("timestamp") - timedelta(days=date_offset_days)).alias("timestamp"),
    ).with_columns(
        # Normalize prices to returns per symbol
        pl.col("open").pct_change().over("symbol").fill_null(0.0).alias("open"),
        pl.col("high").pct_change().over("symbol").fill_null(0.0).alias("high"),
        pl.col("low").pct_change().over("symbol").fill_null(0.0).alias("low"),
        pl.col("close").pct_change().over("symbol").fill_null(0.0).alias("close"),
        # Z-score normalize volume per symbol
        ((pl.col("volume") - pl.col("volume").mean().over("symbol"))
         / pl.col("volume").std().over("symbol")).fill_null(0.0).alias("volume"),
    )

    return result, reverse_map


def load_parquet_duckdb(parquet_path: str | Path) -> pl.DataFrame:
    """Load a parquet file via DuckDB, returning a Polars DataFrame (zero-copy via Arrow)."""
    con = duckdb.connect()
    arrow_table = con.execute(
        "SELECT * FROM read_parquet(?)", [str(parquet_path)]
    ).to_arrow_table()
    con.close()
    return pl.DataFrame(pl.from_arrow(arrow_table))


def load_and_anonymize(
    parquet_path: str | Path,
    key: bytes | None = None,
    date_offset_days: int = 1000,
) -> tuple[pl.DataFrame, bytes, dict[str, str]]:
    """Load a parquet file via DuckDB and anonymize it."""
    df = load_parquet_duckdb(parquet_path)

    # Ensure expected columns exist
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if key is None:
        key = generate_key()

    anon_df, reverse_map = anonymize_dataset(df, key, date_offset_days)
    return anon_df, key, reverse_map


def save_key(key: bytes, path: Path) -> None:
    """Write an HMAC key as hex to a file (owner-only permissions)."""
    import os

    path.write_text(key.hex() + "\n")
    os.chmod(path, 0o600)


def load_key(path: Path) -> bytes:
    """Read a hex-encoded HMAC key from a file."""
    return bytes.fromhex(path.read_text().strip())


def generate_synthetic_data(
    symbols: list[str] | None = None,
    bars_per_symbol: int = 500,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    symbols = symbols or ["AAPL", "GOOGL", "MSFT"]
    rows = []

    from datetime import date
    from datetime import timedelta as td

    for sym in symbols:
        price = 100.0
        base_date = date(2020, 1, 1)
        for i in range(bars_per_symbol):
            daily_return = rng.normal(0.0005, 0.02)
            price *= 1 + daily_return
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            volume = max(1000, int(rng.normal(1_000_000, 200_000)))
            rows.append({
                "symbol": sym,
                "timestamp": base_date + td(days=i),
                "open": price * (1 + rng.normal(0, 0.002)),
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            })

    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Date)
    )
