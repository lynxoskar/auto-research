"""Torture tests — detect overfitting and randomness.

Three tests:
1. Noise test: shuffle returns, compare Sharpe (is timing real?)
2. Deflation test: double costs, check profitability (enough edge?)
3. Walk-forward test: rolling folds, check temporal stability
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

    # Simple comparison: real must be positive and meaningfully better than shuffled
    if mean_shuffled > 0:
        ratio = real_sharpe / mean_shuffled
        passed = real_sharpe > mean_shuffled * threshold
    else:
        # Shuffled is zero or negative — real just needs to be positive
        ratio = float("inf") if real_sharpe > 0 else 0.0
        passed = real_sharpe > 0

    return {
        "passed": passed,
        "real_sharpe": round(real_sharpe, 4),
        "mean_shuffled_sharpe": round(mean_shuffled, 4),
        "ratio": round(ratio, 4) if ratio != float("inf") else "inf",
    }


def walkforward_test(
    close_returns: np.ndarray,
    positions: np.ndarray,
    n_folds: int = 5,
    train_frac: float = 0.6,
    cost_bps: float = 10.0,
) -> dict:
    """Rolling temporal validation with multiple train/test windows.

    Slides n_folds windows across the data. Each window uses train_frac
    for training, the rest for testing. Strategy must be profitable in
    >50% of test windows.

    Returns {passed, folds: [{fold, train_sharpe, test_sharpe}], pass_rate}.
    """
    n = len(close_returns)
    if n < 50:
        return {"passed": True, "folds": [], "pass_rate": 0.0, "skipped": "need >= 50 bars"}

    window_size = n // n_folds
    if window_size < 10:
        return {"passed": True, "folds": [], "pass_rate": 0.0, "skipped": "window too small"}

    train_size = int(window_size * train_frac)
    folds = []

    for i in range(n_folds):
        start = i * window_size
        train_end = start + train_size
        test_end = start + window_size

        if test_end > n or train_end >= test_end:
            break

        train_ret = close_returns[start:train_end]
        train_pos = positions[start:train_end]
        test_ret = close_returns[train_end:test_end]
        test_pos = positions[train_end:test_end]

        if len(train_ret) == 0 or len(test_ret) == 0:
            break

        train_bt = backtest(train_ret, train_pos, cost_bps)
        test_bt = backtest(test_ret, test_pos, cost_bps)

        folds.append({
            "fold": i,
            "train_sharpe": round(train_bt["sharpe"], 4),
            "test_sharpe": round(test_bt["sharpe"], 4),
        })

    if not folds:
        return {"passed": True, "folds": [], "pass_rate": 0.0, "skipped": "no valid folds"}

    pass_rate = sum(1 for f in folds if f["test_sharpe"] > 0) / len(folds)

    return {
        "passed": pass_rate > 0.5,
        "folds": folds,
        "pass_rate": round(pass_rate, 4),
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
