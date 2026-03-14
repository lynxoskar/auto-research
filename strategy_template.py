"""Example strategy — LLM rewrites this file each iteration.

Returns position weights per bar: -1.0 (short) to 1.0 (long), 0.0 = flat.
"""

import numpy as np
import pandas as pd


def strategy(bars):
    """Generate position weights from market data.

    Args:
        bars: dict of numpy arrays. Keys depend on the data source.
              Typical keys: open, high, low, close, volume.
              Each array is 1D (single symbol) or 2D (n_bars x n_symbols).
              The skill file describes the exact schema.

    Returns:
        positions: 1D numpy array of floats, length = number of bars.
                   -1.0 = full short, 0.0 = flat, 1.0 = full long.
    """
    close = bars["close"]
    n = len(close)
    positions = np.zeros(n)

    # Reconstruct price-like series from cumulative returns
    prices = np.cumprod(1 + close)

    # SMA crossover: long when fast > slow, flat otherwise
    fast = pd.Series(prices).rolling(20, min_periods=1).mean().values
    slow = pd.Series(prices).rolling(50, min_periods=1).mean().values

    positions = np.where(fast > slow, 1.0, 0.0)

    # Don't position on first bar (insufficient data)
    positions[0] = 0.0

    return positions
