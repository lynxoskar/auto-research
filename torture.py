"""Torture tests — detect overfitting and randomness.

Two tests:
1. Noise test: shuffle returns, re-run backtest with same positions, compare Sharpe.
2. Deflation test: double transaction costs, check if still profitable.
"""

from __future__ import annotations

import numpy as np

from backtest import backtest


def noise_test(
    close_returns: np.ndarray,
    positions: np.ndarray,
    n_shuffles: int = 5,
    cost_bps: float = 10.0,
    threshold: float = 1.5,
) -> dict:
    """Test if strategy Sharpe is significantly better than on shuffled data.

    Keeps the same position sequence but shuffles the return series.
    If the strategy's timing doesn't matter, it's not a real signal.

    Returns dict with passed, real_sharpe, mean_shuffled_sharpe, ratio.
    """
    real = backtest(close_returns, positions, cost_bps)
    real_sharpe = real["sharpe"]

    rng = np.random.default_rng(42)
    shuffled_sharpes = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(close_returns)
        result = backtest(shuffled, positions, cost_bps)
        shuffled_sharpes.append(result["sharpe"])

    mean_shuffled = float(np.mean(shuffled_sharpes))

    # Handle negative/zero mean_shuffled correctly
    if mean_shuffled > 0:
        ratio = real_sharpe / mean_shuffled
        passed = real_sharpe > mean_shuffled * threshold
    elif mean_shuffled == 0:
        ratio = float("inf") if real_sharpe > 0 else 0.0
        passed = real_sharpe > 0
    else:
        ratio = float("inf") if real_sharpe > 0 else real_sharpe / abs(mean_shuffled)
        passed = real_sharpe > 0 and real_sharpe > abs(mean_shuffled) * threshold

    return {
        "passed": passed,
        "real_sharpe": round(real_sharpe, 4),
        "mean_shuffled_sharpe": round(mean_shuffled, 4),
        "ratio": round(ratio, 4) if ratio != float("inf") else "inf",
    }


def holdout_test(
    close_returns: np.ndarray,
    positions: np.ndarray,
    train_frac: float = 0.8,
    cost_bps: float = 10.0,
) -> dict:
    """Walk-forward temporal holdout: train on first 80%, validate on last 20%.

    Uses the SAME position weights on both halves — catches time-dependent overfitting
    that the noise shuffle test misses.

    Returns dict with passed, train_sharpe, holdout_sharpe, split_index.
    """
    split = int(len(close_returns) * train_frac)
    train = backtest(close_returns[:split], positions[:split], cost_bps)
    holdout = backtest(close_returns[split:], positions[split:], cost_bps)

    return {
        "passed": holdout["sharpe"] > 0,
        "train_sharpe": round(train["sharpe"], 4),
        "holdout_sharpe": round(holdout["sharpe"], 4),
        "split_index": split,
    }


def deflation_test(
    close_returns: np.ndarray,
    positions: np.ndarray,
    base_cost_bps: float = 10.0,
    cost_multiplier: float = 2.0,
) -> dict:
    """Test if strategy survives with doubled transaction costs.

    Returns dict with passed, base_sharpe, deflated_sharpe.
    """
    base = backtest(close_returns, positions, base_cost_bps)
    deflated = backtest(close_returns, positions, base_cost_bps * cost_multiplier)

    return {
        "passed": deflated["sharpe"] > 0,
        "base_sharpe": round(base["sharpe"], 4),
        "deflated_sharpe": round(deflated["sharpe"], 4),
    }
