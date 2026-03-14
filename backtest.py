"""Simple numpy-based backtest engine.

Evaluates a strategy's entry/exit signals against OHLCV return data.
Returns Sharpe, max drawdown, trade count, and equity curve.
"""

from __future__ import annotations

import numpy as np


def backtest(
    close_returns: np.ndarray,
    entry_signals: np.ndarray,
    exit_signals: np.ndarray,
    cost_bps: float = 10.0,
    bars_per_year: float = 252.0,
    risk_free_rate: float = 0.02,
) -> dict:
    """Run a simple long-only backtest.

    Args:
        close_returns: Array of close-to-close percentage returns.
        entry_signals: Boolean array — True where strategy enters long.
        exit_signals: Boolean array — True where strategy exits.
        cost_bps: One-way transaction cost in basis points (applied on entry AND exit).
        bars_per_year: For annualizing Sharpe/returns.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict with sharpe, max_drawdown, trades, final_equity, win_rate.
    """
    n = len(close_returns)
    if n == 0:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "final_equity": 1.0,
                "win_rate": 0.0}

    entry_signals = np.asarray(entry_signals, dtype=bool)[:n]
    exit_signals = np.asarray(exit_signals, dtype=bool)[:n]

    cost_frac = cost_bps / 10_000

    # Compute strategy returns: position * market return, with costs on transitions
    position = np.zeros(n, dtype=bool)
    strategy_returns = np.zeros(n)
    trades = 0
    wins = 0
    entry_equity = 1.0
    equity = 1.0

    for i in range(n):
        was_long = bool(position[i - 1]) if i > 0 else False
        if not was_long:
            # Was flat
            if entry_signals[i]:
                position[i] = True
                strategy_returns[i] = -cost_frac  # entry cost
                entry_equity = equity * (1 + strategy_returns[i])
                trades += 1
        else:
            # Was long
            if exit_signals[i]:
                position[i] = False
                strategy_returns[i] = close_returns[i] - cost_frac  # exit cost + return
                exit_equity = equity * (1 + strategy_returns[i])
                if exit_equity > entry_equity:
                    wins += 1
            else:
                position[i] = True
                strategy_returns[i] = close_returns[i]

        equity *= 1 + strategy_returns[i]

    # Compute equity curve from strategy returns
    equity_curve = np.cumprod(1 + strategy_returns)

    # Sharpe computed over strategy returns (not raw equity)
    # This correctly handles flat periods as 0 return
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)

    sharpe = 0.0
    if std_ret > 0:
        excess = mean_ret - risk_free_rate / bars_per_year
        sharpe = excess / std_ret * np.sqrt(bars_per_year)

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))

    win_rate = wins / trades if trades > 0 else 0.0

    return {
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "trades": trades,
        "final_equity": round(float(equity_curve[-1]), 4),
        "win_rate": round(win_rate, 4),
    }
