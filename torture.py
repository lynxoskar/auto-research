"""Torture tests — detect overfitting and randomness.

Two tests:
1. Noise test: shuffle returns, re-run strategy, compare Sharpe.
2. Deflation test: double transaction costs, check if still profitable.
"""

from __future__ import annotations

import numpy as np

from backtest import backtest


def noise_test(
    close_returns: np.ndarray,
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    n_shuffles: int = 5,
    cost_bps: float = 10.0,
    threshold: float = 1.5,
) -> dict:
    """Test if strategy Sharpe is significantly better than on shuffled data.

    Returns dict with passed, real_sharpe, mean_shuffled_sharpe, ratio.
    """
    real = backtest(close_returns, entry_signals, exit_signals, cost_bps)
    real_sharpe = real["sharpe"]

    rng = np.random.default_rng(42)
    shuffled_sharpes = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(close_returns)
        # Use same entry/exit signals on shuffled data — tests if signal timing matters
        result = backtest(shuffled, entry_signals, exit_signals, cost_bps)
        shuffled_sharpes.append(result["sharpe"])

    mean_shuffled = float(np.mean(shuffled_sharpes))

    # Compute ratio: real vs shuffled. Handle negative/zero mean_shuffled correctly.
    # When mean_shuffled <= 0, a positive real_sharpe is infinitely better.
    # When mean_shuffled > 0, require real_sharpe to be threshold * mean_shuffled.
    if mean_shuffled > 0:
        ratio = real_sharpe / mean_shuffled
        passed = real_sharpe > mean_shuffled * threshold
    elif mean_shuffled == 0:
        ratio = float("inf") if real_sharpe > 0 else 0.0
        passed = real_sharpe > 0
    else:
        # mean_shuffled < 0: real must be positive and better than shuffled by threshold margin
        ratio = float("inf") if real_sharpe > 0 else real_sharpe / abs(mean_shuffled)
        passed = real_sharpe > 0 and real_sharpe > abs(mean_shuffled) * threshold

    return {
        "passed": passed,
        "real_sharpe": round(real_sharpe, 4),
        "mean_shuffled_sharpe": round(mean_shuffled, 4),
        "ratio": round(ratio, 4) if ratio != float("inf") else "inf",
    }


def deflation_test(
    close_returns: np.ndarray,
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    base_cost_bps: float = 10.0,
    cost_multiplier: float = 2.0,
) -> dict:
    """Test if strategy survives with doubled transaction costs.

    Returns dict with passed, base_sharpe, deflated_sharpe.
    """
    base = backtest(close_returns, entry_signals, exit_signals, base_cost_bps)
    deflated = backtest(
        close_returns, entry_signals, exit_signals, base_cost_bps * cost_multiplier
    )

    return {
        "passed": deflated["sharpe"] > 0,
        "base_sharpe": round(base["sharpe"], 4),
        "deflated_sharpe": round(deflated["sharpe"], 4),
    }
