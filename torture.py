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


def walkforward_test(
    close_returns: np.ndarray,
    positions: np.ndarray,
    n_folds: int = 5,
    train_frac: float = 0.6,
    cost_bps: float = 10.0,
) -> dict:
    """Rolling walk-forward validation with multiple train/test windows.

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


def stress_test(
    close_returns: np.ndarray,
    positions: np.ndarray,
    cost_bps: float = 10.0,
    seed: int = 42,
) -> dict:
    """Test strategy against 3 hardcoded stress scenarios.

    1. Flash crash: inject -20% return at a random point
    2. Dead market: replace 50 bars with 0.0 returns
    3. Volatility spike: multiply 30 bars by 5x

    Returns {passed, scenarios: [{name, passed, max_drawdown}]}.
    Requires at least 100 bars.
    """
    n = len(close_returns)
    if n < 100:
        return {"passed": True, "scenarios": [], "skipped": "need >= 100 bars"}

    rng = np.random.default_rng(seed)
    scenarios = []

    # 1. Flash crash
    crash_returns = close_returns.copy()
    crash_idx = rng.integers(n // 4, 3 * n // 4)
    crash_returns[crash_idx] = -0.20
    crash_bt = backtest(crash_returns, positions, cost_bps)
    scenarios.append({
        "name": "flash_crash",
        "passed": crash_bt["max_drawdown"] > -0.50,
        "max_drawdown": crash_bt["max_drawdown"],
    })

    # 2. Dead market
    dead_returns = close_returns.copy()
    dead_start = rng.integers(0, max(1, n - 50))
    dead_returns[dead_start : dead_start + 50] = 0.0
    dead_bt = backtest(dead_returns, positions, cost_bps)
    scenarios.append({
        "name": "dead_market",
        "passed": dead_bt["final_equity"] > 0.5,
        "max_drawdown": dead_bt["max_drawdown"],
    })

    # 3. Volatility spike
    vol_returns = close_returns.copy()
    vol_start = rng.integers(0, max(1, n - 30))
    vol_returns[vol_start : vol_start + 30] *= 5.0
    vol_bt = backtest(vol_returns, positions, cost_bps)
    scenarios.append({
        "name": "volatility_spike",
        "passed": vol_bt["max_drawdown"] > -0.50,
        "max_drawdown": vol_bt["max_drawdown"],
    })

    return {
        "passed": all(s["passed"] for s in scenarios),
        "scenarios": scenarios,
    }
