# tests/test_stability.py
"""Tests for pairs.stats.stability — hedge ratio stability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.stats.stability import (
    cusum_beta_stability,
    rolling_beta_drift,
    summarize_hedge_ratio_stability,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_stable_beta(n: int = 252, seed: int = 0) -> pd.Series:
    """Stable beta — constant value with tiny noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    beta = 1.0 + rng.standard_normal(n) * 0.001
    return pd.Series(beta, index=dates, name="beta")


def _make_step_change_beta(n: int = 252, shift_at: int = 126, seed: int = 0) -> pd.Series:
    """Beta with a clear step change at shift_at — should fail CUSUM."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    beta = np.ones(n)
    beta[shift_at:] = 2.5    # large level shift
    beta += rng.standard_normal(n) * 0.01
    return pd.Series(beta, index=dates, name="beta")


def _make_volatile_beta(n: int = 252, seed: int = 0) -> pd.Series:
    """Beta with very high rolling volatility — should fail rolling test."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    beta = rng.standard_normal(n) * 5.0   # large global std
    # Inject a burst of extreme variance in the last quarter
    burst_start = 3 * n // 4
    beta[burst_start:] = rng.standard_normal(n - burst_start) * 50.0
    return pd.Series(beta, index=dates, name="beta")


def _make_kf_results(n_pairs: int = 3, n: int = 252, seed: int = 0) -> dict:
    """Synthetic kf_results with stable beta."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    results = {}
    tickers = [f"T{i}" for i in range(n_pairs + 1)]
    for i in range(n_pairs):
        k1, k2 = tickers[i], tickers[i + 1]
        resid = np.zeros(n)
        for t in range(1, n):
            resid[t] = 0.8 * resid[t - 1] + rng.standard_normal()
        beta = 1.0 + rng.standard_normal(n) * 0.01
        df = pd.DataFrame({"resid": resid, "beta": beta, "alpha": np.zeros(n)}, index=dates)
        results[(k1, k2)] = df
    return results


# ---------------------------------------------------------------------------
# TestCusumBetaStability
# ---------------------------------------------------------------------------

class TestCusumBetaStability:

    def test_stable_beta_is_stable(self):
        beta = _make_stable_beta()
        result = cusum_beta_stability(beta)
        assert result["is_stable"] is True

    def test_step_change_is_unstable(self):
        beta = _make_step_change_beta()
        result = cusum_beta_stability(beta)
        assert result["is_stable"] is False

    def test_returns_all_required_keys(self):
        beta = _make_stable_beta()
        result = cusum_beta_stability(beta)
        for key in ("cusum_stat", "critical_val", "is_stable", "alpha", "cusum_series"):
            assert key in result

    def test_cusum_series_same_length_as_non_nan_input(self):
        beta = _make_stable_beta(n=200)
        result = cusum_beta_stability(beta)
        assert len(result["cusum_series"]) == 200

    def test_alpha_01_gives_higher_critical_val(self):
        beta = _make_stable_beta()
        r05 = cusum_beta_stability(beta, alpha=0.05)
        r01 = cusum_beta_stability(beta, alpha=0.01)
        assert r01["critical_val"] > r05["critical_val"]

    def test_alpha_is_stored_in_result(self):
        beta = _make_stable_beta()
        result = cusum_beta_stability(beta, alpha=0.01)
        assert result["alpha"] == 0.01

    def test_invalid_alpha_raises(self):
        beta = _make_stable_beta()
        with pytest.raises(ValueError, match="alpha"):
            cusum_beta_stability(beta, alpha=0.25)

    def test_short_series_handled_gracefully(self):
        beta = pd.Series([1.0])
        result = cusum_beta_stability(beta)
        assert np.isnan(result["cusum_stat"])
        # is_stable = True (don't flag insufficient data)
        assert result["is_stable"] is True

    def test_nan_in_beta_handled(self):
        beta = _make_stable_beta(n=100)
        beta.iloc[10:20] = np.nan
        result = cusum_beta_stability(beta)
        # Should not raise; dropna handles NaNs
        assert isinstance(result["cusum_stat"], float)

    def test_constant_beta_gives_zero_stat(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        beta = pd.Series(np.ones(100), index=dates)
        result = cusum_beta_stability(beta)
        assert result["cusum_stat"] == 0.0
        assert result["is_stable"] is True

    def test_cusum_series_is_pandas_series(self):
        beta = _make_stable_beta()
        result = cusum_beta_stability(beta)
        assert isinstance(result["cusum_series"], pd.Series)

    def test_step_change_cusum_stat_exceeds_critical(self):
        beta = _make_step_change_beta()
        result = cusum_beta_stability(beta)
        assert result["cusum_stat"] > result["critical_val"]


# ---------------------------------------------------------------------------
# TestRollingBetaDrift
# ---------------------------------------------------------------------------

class TestRollingBetaDrift:

    def test_stable_beta_is_stable(self):
        beta = _make_stable_beta()
        result = rolling_beta_drift(beta)
        assert result["is_stable"] is True

    def test_volatile_beta_flags_instability(self):
        """Beta with a late burst of high volatility should be flagged."""
        beta = _make_volatile_beta()
        result = rolling_beta_drift(beta, window=30, threshold_sigma=2.0)
        assert result["is_stable"] is False
        assert len(result["flagged_dates"]) > 0

    def test_returns_all_required_keys(self):
        beta = _make_stable_beta()
        result = rolling_beta_drift(beta)
        for key in ("max_roll_std_ratio", "threshold_sigma", "is_stable",
                    "flagged_dates", "roll_std_series"):
            assert key in result

    def test_flagged_dates_is_datetime_index(self):
        beta = _make_volatile_beta()
        result = rolling_beta_drift(beta, window=30, threshold_sigma=1.5)
        assert isinstance(result["flagged_dates"], pd.DatetimeIndex)

    def test_roll_std_series_same_length_as_input(self):
        beta = _make_stable_beta(n=200)
        result = rolling_beta_drift(beta)
        assert len(result["roll_std_series"]) == 200

    def test_threshold_sigma_stored_in_result(self):
        beta = _make_stable_beta()
        result = rolling_beta_drift(beta, threshold_sigma=3.0)
        assert result["threshold_sigma"] == 3.0

    def test_all_nan_beta_handled_gracefully(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        beta = pd.Series(np.full(50, np.nan), index=dates)
        result = rolling_beta_drift(beta)
        assert result["is_stable"] is True
        assert result["max_roll_std_ratio"] == 0.0

    def test_constant_beta_is_stable(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        beta = pd.Series(np.ones(100), index=dates)
        result = rolling_beta_drift(beta)
        assert result["is_stable"] is True

    def test_no_flagged_dates_on_stable_series(self):
        beta = _make_stable_beta()
        result = rolling_beta_drift(beta)
        assert len(result["flagged_dates"]) == 0

    def test_roll_std_series_is_pandas_series(self):
        beta = _make_stable_beta()
        result = rolling_beta_drift(beta)
        assert isinstance(result["roll_std_series"], pd.Series)


# ---------------------------------------------------------------------------
# TestSummarizeHedgeRatioStability
# ---------------------------------------------------------------------------

class TestSummarizeHedgeRatioStability:

    def test_returns_dataframe_with_all_columns(self):
        kf = _make_kf_results(n_pairs=3)
        stab = summarize_hedge_ratio_stability(kf)
        expected_cols = [
            "cusum_stat", "cusum_critical", "cusum_stable",
            "roll_max_ratio", "roll_stable", "n_flagged_bars", "overall_stable",
        ]
        for col in expected_cols:
            assert col in stab.columns

    def test_index_is_multiindex_with_ticker_names(self):
        kf = _make_kf_results(n_pairs=3)
        stab = summarize_hedge_ratio_stability(kf)
        assert isinstance(stab.index, pd.MultiIndex)
        assert stab.index.names == ["ticker1", "ticker2"]

    def test_stable_pairs_overall_stable_true(self):
        kf = _make_kf_results(n_pairs=3)
        stab = summarize_hedge_ratio_stability(kf)
        assert stab["overall_stable"].all()

    def test_unstable_pair_sorted_first(self):
        """An unstable pair should appear at the top of the result."""
        n = 252
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rng = np.random.default_rng(1)
        # Stable pair
        stable_beta = 1.0 + rng.standard_normal(n) * 0.001
        df_stable = pd.DataFrame({"beta": stable_beta, "resid": rng.standard_normal(n)}, index=dates)
        # Unstable pair: large step change
        unstable_beta = np.ones(n)
        unstable_beta[126:] = 3.0
        df_unstable = pd.DataFrame({"beta": unstable_beta, "resid": rng.standard_normal(n)}, index=dates)

        kf = {
            ("S1", "S2"): df_stable,
            ("U1", "U2"): df_unstable,
        }
        stab = summarize_hedge_ratio_stability(kf)
        # Unstable pair (U1,U2) should be first (sorted ascending by overall_stable)
        first = stab.index[0]
        assert first == ("U1", "U2")

    def test_overall_stable_is_and_of_both(self):
        kf = _make_kf_results(n_pairs=3)
        stab = summarize_hedge_ratio_stability(kf)
        expected = (stab["cusum_stable"] & stab["roll_stable"]).values
        np.testing.assert_array_equal(stab["overall_stable"].values, expected)

    def test_min_obs_filters_short_series(self):
        """Pairs with fewer obs than min_obs get is_stable=True (not tested / skipped)."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df_short = pd.DataFrame({"beta": np.ones(10), "resid": np.zeros(10)}, index=dates)
        kf = {("A", "B"): df_short}
        stab = summarize_hedge_ratio_stability(kf, min_obs=30)
        # Should still return a row
        assert len(stab) == 1
        # overall_stable = True for short series (insufficient data → don't flag)
        assert bool(stab["overall_stable"].iloc[0]) is True

    def test_n_rows_equals_n_pairs(self):
        kf = _make_kf_results(n_pairs=5)
        stab = summarize_hedge_ratio_stability(kf)
        assert len(stab) == 5

    def test_empty_kf_returns_empty_df(self):
        stab = summarize_hedge_ratio_stability({})
        assert isinstance(stab, pd.DataFrame)
        assert stab.empty

    def test_missing_beta_column_skipped(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        df_no_beta = pd.DataFrame({"resid": np.zeros(100)}, index=dates)
        kf = {("A", "B"): df_no_beta}
        stab = summarize_hedge_ratio_stability(kf)
        assert stab.empty

    def test_cusum_stat_nonnegative(self):
        kf = _make_kf_results(n_pairs=3)
        stab = summarize_hedge_ratio_stability(kf)
        assert (stab["cusum_stat"] >= 0).all()
