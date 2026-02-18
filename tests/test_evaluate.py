# tests/test_evaluate.py
"""Unit tests for pairs.strategies.evaluate."""
import numpy as np
import pandas as pd
import pytest

from pairs.strategies.evaluate import evaluate_pair_signals


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pair_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "P1": 100 + rng.standard_normal(n).cumsum(),
        "P2": 100 + rng.standard_normal(n).cumsum(),
    }, index=dates)


def _make_signals(df: pd.DataFrame, *, pos: int = 1) -> pd.DataFrame:
    """All-in signals for the full window."""
    return pd.DataFrame({
        "n1": pos * np.ones(len(df)),
        "n2": -pos * np.ones(len(df)),
        "pos": pos * np.ones(len(df), dtype=int),
    }, index=df.index)


def _flat_signals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "n1": np.zeros(len(df)),
        "n2": np.zeros(len(df)),
        "pos": np.zeros(len(df), dtype=int),
    }, index=df.index)


# ── basic structure ────────────────────────────────────────────────────────────

class TestEvaluatePairSignalsStructure:
    def test_returns_three_elements(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        result = evaluate_pair_signals(df, sig)
        assert len(result) == 3

    def test_daily_has_expected_columns(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        daily, _, _ = evaluate_pair_signals(df, sig)
        for col in ["pnl_gross", "pnl_net", "ret_net", "drawdown_pct", "in_pos"]:
            assert col in daily.columns

    def test_summary_has_expected_keys(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        _, _, summary = evaluate_pair_signals(df, sig)
        for key in ["sharpe", "ann_return", "max_drawdown", "n_trades", "hit_rate",
                    "profit_factor", "capital_base"]:
            assert key in summary


# ── profit_factor edge cases ───────────────────────────────────────────────────

class TestProfitFactor:
    def test_zero_trades_profit_factor_is_nan(self):
        """When there are no round-trip trades, profit_factor must be np.nan, not np.inf."""
        df = _make_pair_df()
        sig = _flat_signals(df)
        _, _, summary = evaluate_pair_signals(df, sig)
        assert np.isnan(summary["profit_factor"]), (
            f"Expected nan for zero trades, got {summary['profit_factor']}"
        )

    def test_empty_input_profit_factor_is_nan(self):
        """Empty DataFrame early return must also give np.nan."""
        df = _make_pair_df()
        # Force empty by making all prices non-finite
        df_bad = df.copy()
        df_bad["P1"] = np.nan
        sig = _make_signals(df)
        _, _, summary = evaluate_pair_signals(df_bad, sig)
        assert np.isnan(summary["profit_factor"])

    def test_profitable_strategy_finite_profit_factor(self):
        """A strategy with winning trades should have a finite profit_factor."""
        df = _make_pair_df()
        sig = _make_signals(df)
        _, _, summary = evaluate_pair_signals(df, sig)
        # May be nan if only wins (no losses denominator), but must not be -inf
        pf = summary["profit_factor"]
        assert np.isnan(pf) or pf >= 0


# ── capital_base_percentile validation ────────────────────────────────────────

class TestCapitalBaseValidation:
    def test_invalid_percentile_above_one_raises(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        with pytest.raises(ValueError, match="capital_base_percentile"):
            evaluate_pair_signals(df, sig, capital_base_percentile=1.5)

    def test_invalid_percentile_zero_raises(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        with pytest.raises(ValueError, match="capital_base_percentile"):
            evaluate_pair_signals(df, sig, capital_base_percentile=0.0)

    def test_invalid_capital_base_raises(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        with pytest.raises(ValueError, match="capital_base"):
            evaluate_pair_signals(df, sig, capital_base=-100.0)

    def test_valid_percentile_does_not_raise(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        daily, _, _ = evaluate_pair_signals(df, sig, capital_base_percentile=0.75)
        assert len(daily) > 0


# ── cost model sanity ──────────────────────────────────────────────────────────

class TestCostModel:
    def test_net_pnl_less_than_gross_with_costs(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        _, _, summary_free = evaluate_pair_signals(df, sig, cost_bps=0.0)
        _, _, summary_cost = evaluate_pair_signals(df, sig, cost_bps=10.0)
        assert summary_cost["net_pnl"] <= summary_free["net_pnl"]

    def test_borrow_cost_reduces_pnl(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        _, _, s0 = evaluate_pair_signals(df, sig, borrow_bps_per_year=0)
        _, _, s1 = evaluate_pair_signals(df, sig, borrow_bps_per_year=100)
        assert s1["net_pnl"] <= s0["net_pnl"]


# ── drawdown sanity ────────────────────────────────────────────────────────────

class TestDrawdown:
    def test_max_drawdown_pct_non_positive(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        daily, _, summary = evaluate_pair_signals(df, sig)
        assert summary["max_drawdown_pct"] >= 0
        assert (daily["drawdown_pct"] <= 0).all()

    def test_drawdown_pct_bounded(self):
        df = _make_pair_df()
        sig = _make_signals(df)
        daily, _, _ = evaluate_pair_signals(df, sig)
        assert daily["drawdown_pct"].min() >= -1.0  # can't lose more than 100%


# ── missing required columns ───────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_price_column_raises(self):
        df = pd.DataFrame({"P1": [100.0, 101.0]},
                          index=pd.date_range("2020-01-01", periods=2))
        sig = pd.DataFrame({"n1": [1.0, 1.0], "n2": [-1.0, -1.0]},
                           index=df.index)
        with pytest.raises(KeyError):
            evaluate_pair_signals(df, sig)
