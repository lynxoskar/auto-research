"""Strategy profiler -- basic stats on a positions array.

Since strategies are opaque Python functions, we can't decompose their signals.
Instead we profile the *output*: how often positioned, long vs short, churn, etc.
"""

from __future__ import annotations

import numpy as np


def profile_strategy(positions: np.ndarray, threshold: float = 0.01) -> dict:
    """Profile a positions array and return summary statistics.

    Args:
        positions: Float array of target positions per bar (-1 to 1).
        threshold: Minimum absolute position to count as "positioned".

    Returns:
        Dict with bars_total, bars_positioned, bars_long, bars_short, bars_flat,
        avg_position, avg_abs_position, position_changes, max_consecutive_positioned,
        max_consecutive_flat.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if n == 0:
        return _empty_profile()

    positioned = np.abs(positions) > threshold
    long_mask = positions > threshold
    short_mask = positions < -threshold
    flat_mask = ~positioned

    # Position change count (number of bars where position differs from prior bar)
    changes = np.sum(np.abs(np.diff(positions)) > threshold) if n > 1 else 0

    return {
        "bars_total": n,
        "bars_positioned": int(np.sum(positioned)),
        "bars_long": int(np.sum(long_mask)),
        "bars_short": int(np.sum(short_mask)),
        "bars_flat": int(np.sum(flat_mask)),
        "avg_position": round(float(np.mean(positions)), 4),
        "avg_abs_position": round(float(np.mean(np.abs(positions))), 4),
        "position_changes": int(changes),
        "max_consecutive_positioned": _max_run(positioned),
        "max_consecutive_flat": _max_run(flat_mask),
    }


def _max_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive True run in a boolean array."""
    if len(mask) == 0 or not np.any(mask):
        return 0
    # Pad with False on both ends to detect runs at boundaries
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int(np.max(ends - starts))


def _empty_profile() -> dict:
    return {
        "bars_total": 0,
        "bars_positioned": 0,
        "bars_long": 0,
        "bars_short": 0,
        "bars_flat": 0,
        "avg_position": 0.0,
        "avg_abs_position": 0.0,
        "position_changes": 0,
        "max_consecutive_positioned": 0,
        "max_consecutive_flat": 0,
    }
