"""Example strategy — LLM rewrites this file each iteration.

Returns position weights per bar: -1.0 (short) to 1.0 (long), 0.0 = flat.
"""

import numpy as np
import pandas as pd


def strategy(open, high, low, close, volume):
    """Generate position weights from anonymized OHLCV returns.

    Args:
        open, high, low, close: numpy arrays of percentage returns.
        volume: numpy array of z-score normalized volume.

    Returns:
        positions: numpy array of floats, same length as inputs.
                   -1.0 = full short, 0.0 = flat, 1.0 = full long.
    """
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
