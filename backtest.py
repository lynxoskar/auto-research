"""Vectorized backtest engine — position weights instead of boolean signals.

The strategy returns a position array: -1.0 (full short) to 1.0 (full long), 0.0 = flat.
Transaction costs are proportional to position changes.
"""

from __future__ import annotations

import numpy as np


def backtest(
    close_returns: np.ndarray,
    positions: np.ndarray,
    cost_bps: float = 10.0,
    bars_per_year: float = 252.0,
    risk_free_rate: float = 0.02,
) -> dict:
    """Run a vectorized backtest from position weights.

    Args:
        close_returns: Array of close-to-close percentage returns.
        positions: Float array of target positions per bar.
                   -1.0 = full short, 0.0 = flat, 1.0 = full long.
                   Clipped to [-1, 1].
        cost_bps: One-way transaction cost in basis points per unit of position change.
        bars_per_year: For annualizing Sharpe.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict with sharpe, max_drawdown, trades, final_equity, win_rate, exposure.
    """
    n = len(close_returns)
    if n == 0:
        return _empty_result()

    positions = np.asarray(positions, dtype=np.float64)[:n]
    positions = np.clip(positions, -1.0, 1.0)

    cost_frac = cost_bps / 10_000

    # Position changes incur transaction costs
    position_changes = np.abs(np.diff(positions, prepend=0.0))
    costs = position_changes * cost_frac

    # Strategy returns: position * market return - costs
    strategy_returns = positions * close_returns - costs

    # Equity curve
    equity = np.cumprod(1 + strategy_returns)

    # Sharpe
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = 0.0
    if std_ret > 0:
        excess = mean_ret - risk_free_rate / bars_per_year
        sharpe = excess / std_ret * np.sqrt(bars_per_year)

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))

    # Trade count: number of times position changes significantly (> 0.01)
    trades = int(np.sum(position_changes > 0.01))

    # Win rate: fraction of bars with positive strategy return while positioned
    positioned = np.abs(positions) > 0.01
    win_rate = float(np.mean(strategy_returns[positioned] > 0)) if np.any(positioned) else 0.0

    # Exposure: fraction of bars with a non-trivial position
    exposure = float(np.mean(positioned))

    return {
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "trades": trades,
        "final_equity": round(float(equity[-1]), 4),
        "win_rate": round(win_rate, 4),
        "exposure": round(exposure, 4),
    }


def _empty_result() -> dict:
    return {
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "trades": 0,
        "final_equity": 1.0,
        "win_rate": 0.0,
        "exposure": 0.0,
    }
