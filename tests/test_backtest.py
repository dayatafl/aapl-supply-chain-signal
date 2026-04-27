"""
test_backtest.py
----------------
Unit tests for backtest utility functions (no model needed).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import sharpe_ratio, max_drawdown, compute_strategy_returns


class TestBacktestMetrics:

    def test_sharpe_positive_returns(self):
        returns = pd.Series([0.01] * 252)
        s = sharpe_ratio(returns)
        assert s > 0

    def test_sharpe_zero_std(self):
        returns = pd.Series([0.0] * 100)
        s = sharpe_ratio(returns)
        assert s == 0.0

    def test_sharpe_negative_returns(self):
        returns = pd.Series([-0.01] * 252)
        s = sharpe_ratio(returns)
        assert s < 0

    def test_max_drawdown_no_loss(self):
        equity = pd.Series([1.0, 1.1, 1.2, 1.3])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_known_value(self):
        # Goes from 1.0 → 2.0 → 1.0 → 50% drawdown
        equity = pd.Series([1.0, 2.0, 1.0])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(-0.5, abs=1e-4)

    def test_strategy_returns_shape(self):
        n = 50
        results = pd.DataFrame({
            "prediction": np.random.randint(0, 2, n),
            "actual_direction": np.random.randint(0, 2, n),
            "actual_return_7d": np.random.normal(0.01, 0.05, n),
            "close": np.random.uniform(150, 200, n),
        }, index=pd.date_range("2023-01-01", periods=n, freq="B"))

        out = compute_strategy_returns(results)
        assert "strategy_return" in out.columns
        assert "buyhold_return" in out.columns
        assert "strategy_equity" in out.columns
        assert "buyhold_equity" in out.columns

    def test_strategy_long_only(self):
        """Strategy return = 0 when prediction = 0 (flat)."""
        results = pd.DataFrame({
            "prediction": [0, 0, 0],
            "actual_direction": [1, 0, 1],
            "actual_return_7d": [0.05, -0.03, 0.04],
            "close": [180, 175, 179],
        }, index=pd.date_range("2023-01-01", periods=3, freq="B"))
        out = compute_strategy_returns(results)
        assert (out["strategy_return"] == 0).all()
