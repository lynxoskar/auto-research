"""Pluggable data sources — each source decides its own transforms.

Data sources are simple callables that return a Polars DataFrame with OHLCV columns.
Each source decides whether to anonymize (firewall), normalize, or deliver raw data.
The accompanying skill file describes the schema the LLM receives.

Built-in sources:
- "synthetic" — generated data, firewalled by default
- "parquet" — local parquet file, raw (no firewall)
- "ducklake" — DuckLake catalog over S3, firewalled by default

Usage:
    from datasource import load_data, list_sources

    df = load_data("synthetic")
    df = load_data("parquet", path="data/bars.parquet")
    df = load_data("ducklake", symbol="AAPL")
"""

from __future__ import annotations

import re
from collections.abc import Callable

import duckdb
import polars as pl
from loguru import logger

from firewall import anonymize_dataset, generate_key, generate_synthetic_data

# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------


# Any callable that takes kwargs and returns a Polars DataFrame
RawDataSource = Callable[..., pl.DataFrame]


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCES: dict[str, RawDataSource] = {}


def register_source(name: str, source: RawDataSource) -> None:
    """Register a data source by name."""
    _SOURCES[name] = source


def list_sources() -> list[str]:
    """Return names of all registered data sources."""
    return list(_SOURCES.keys())


def load_raw(name: str, **kwargs) -> pl.DataFrame:
    """Load raw (pre-firewall) OHLCV data from a named source."""
    if name not in _SOURCES:
        raise ValueError(f"Unknown data source '{name}'. Available: {list_sources()}")
    df = _SOURCES[name](**kwargs)
    _validate_columns(df)
    return df


def load_data(name: str, **kwargs) -> pl.DataFrame:
    """Load data from a named source. The source decides its own schema and transforms.

    Some sources (like ducklake) apply the epistemic firewall internally.
    Others (like synthetic) return raw data as-is.
    The skill file for each source describes what the LLM receives.
    """
    if name not in _SOURCES:
        raise ValueError(f"Unknown data source '{name}'. Available: {list_sources()}")
    df = _SOURCES[name](**kwargs)
    _validate_columns(df)
    logger.info(
        "[DATASOURCE] {} → {} bars, {} symbols",
        name,
        len(df),
        df["symbol"].n_unique(),
    )
    return df


def _validate_columns(df: pl.DataFrame) -> None:
    """Ensure the DataFrame has required OHLCV columns."""
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data source missing required columns: {missing}")
    if df.is_empty():
        raise ValueError("Data source returned empty DataFrame")


# ---------------------------------------------------------------------------
# Built-in: synthetic
# ---------------------------------------------------------------------------


def _source_synthetic(*, firewall: bool = True, **kwargs) -> pl.DataFrame:
    """Generate synthetic OHLCV data, optionally firewalled.

    By default applies the epistemic firewall (anonymize + returns).
    Pass firewall=False to get raw synthetic prices.
    """
    df = generate_synthetic_data(**kwargs)
    if firewall:
        key = generate_key()
        df, _ = anonymize_dataset(df, key)
    return df


register_source("synthetic", _source_synthetic)


# ---------------------------------------------------------------------------
# Built-in: parquet (local file)
# ---------------------------------------------------------------------------


def _source_parquet(*, path: str, **_kwargs) -> pl.DataFrame:
    """Load OHLCV from a local parquet file via DuckDB (zero-copy Arrow path)."""
    con = duckdb.connect()
    try:
        arrow = con.execute("SELECT * FROM read_parquet(?)", [path]).to_arrow_table()
        return pl.DataFrame(pl.from_arrow(arrow))
    finally:
        con.close()


register_source("parquet", _source_parquet)


# ---------------------------------------------------------------------------
# Built-in: ducklake
# ---------------------------------------------------------------------------

_DUCKLAKE_DEFAULTS = {
    "catalog_url": "s3://lynx-sandbox-agent-datalake/catalog.ducklake",
    "region": "eu-north-1",
}

# SQL injection guard
_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|COPY|LOAD|CALL|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Alphanumeric + hyphens only for region/catalog identifiers
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9._:/@\-]+$")


def _source_ducklake(
    *,
    catalog_url: str = _DUCKLAKE_DEFAULTS["catalog_url"],
    region: str = _DUCKLAKE_DEFAULTS["region"],
    symbol: str | None = None,
    query: str | None = None,
    firewall: bool = True,
    **_kwargs,
) -> pl.DataFrame:
    """Load OHLCV bars from a DuckLake catalog, firewalled by default.

    The epistemic firewall anonymizes symbols, shifts dates, and normalizes
    prices to returns. The skill file describes the schema the LLM receives.

    Args:
        catalog_url: S3 URL to the DuckLake catalog file.
        region: AWS region for S3 access.
        symbol: Load a single symbol (parameterized query, safe from injection).
        query: Custom SELECT query. Must start with SELECT, no DDL/DML.
        firewall: Apply epistemic firewall (default True). Set False for raw data.
    """
    if not _SAFE_IDENTIFIER.match(region):
        raise ValueError(f"Invalid region: {region!r}")
    if not _SAFE_IDENTIFIER.match(catalog_url):
        raise ValueError(f"Invalid catalog_url: {catalog_url!r}")

    con = duckdb.connect()
    try:
        # Install extensions in order
        for ext in ("ducklake", "httpfs", "aws"):
            con.execute(f"INSTALL {ext}; LOAD {ext};")

        con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain);")
        con.execute(f"SET s3_region='{region}';")
        con.execute(f"ATTACH 'ducklake:{catalog_url}' AS lake (READ_ONLY);")

        logger.info("[DUCKLAKE] Connected to {}", catalog_url)

        if query:
            if not query.strip().upper().startswith("SELECT"):
                raise ValueError("Custom query must start with SELECT")
            if _FORBIDDEN_SQL.search(query):
                raise ValueError("Custom query contains forbidden SQL keywords")
            arrow = con.execute(query).to_arrow_table()
        elif symbol:
            arrow = con.execute(
                "SELECT symbol, timestamp, open, high, low, close, volume "
                "FROM lake.main.lynx_minutebars WHERE symbol = $1 ORDER BY timestamp",
                [symbol],
            ).to_arrow_table()
        else:
            arrow = con.execute(
                "SELECT symbol, timestamp, open, high, low, close, volume "
                "FROM lake.main.lynx_minutebars ORDER BY symbol, timestamp"
            ).to_arrow_table()

        df = pl.DataFrame(pl.from_arrow(arrow))

        if firewall:
            key = generate_key()
            df, _ = anonymize_dataset(df, key)
            logger.info(
                "[DUCKLAKE] Firewall applied: {} symbols anonymized", df["symbol"].n_unique()
            )

        return df
    finally:
        con.close()


register_source("ducklake", _source_ducklake)
