"""Example strategy — LLM rewrites this file each iteration."""

import numpy as np
import pandas as pd


def strategy(open, high, low, close, volume):
    """Generate entry and exit signals from anonymized OHLCV returns.

    Args:
        open, high, low, close: numpy arrays of percentage returns (anonymized).
        volume: numpy array of z-score normalized volume.

    Returns:
        (entry_signals, exit_signals): boolean numpy arrays of same length.
    """
    # Reconstruct price-like series from cumulative returns
    prices = np.cumprod(1 + close)

    # Simple SMA crossover
    fast = pd.Series(prices).rolling(20, min_periods=1).mean().values
    slow = pd.Series(prices).rolling(50, min_periods=1).mean().values

    entry = (fast > slow) & (np.roll(fast, 1) <= np.roll(slow, 1))
    exit_ = (fast < slow) & (np.roll(fast, 1) >= np.roll(slow, 1))

    # Don't signal on first bar (roll artifact)
    entry[0] = False
    exit_[0] = False

    return entry, exit_
