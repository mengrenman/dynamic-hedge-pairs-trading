# tests/test_signals.py
"""Unit tests for pairs.strategies.signals."""
import numpy as np
import pandas as pd
import pytest

from pairs.strategies.signals import (
    estimate_halflife_window,
    zscore_from_spread,
    generate_pair_signals,
)
from pairs.stats.stationarity import estimate_halflife


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_df_pair(n: int = 300, phi: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """DataFrame with 'resid', 'beta', 'P1', 'P2' columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = phi * resid[i - 1] + rng.standard_normal()
    return pd.DataFrame({
        "resid": resid,
        "beta": np.ones(n),
        "P1": 100 + rng.standard_normal(n).cumsum(),
        "P2": 100 + rng.standard_normal(n).cumsum(),
    }, index=dates)


# ── estimate_halflife_window ───────────────────────────────────────────────────

class TestEstimateHalflifeWindow:
    def test_returns_int(self):
        df = _make_df_pair()
        w = estimate_halflife_window(df["resid"])
        assert isinstance(w, int)

    def test_clipped_to_min_win(self):
        # Very fast-reverting series → HL tiny → clipped to min_win
        s = pd.Series(np.random.default_rng(0).standard_normal(500) * 0.001)
        w = estimate_halflife_window(s, min_win=30)
        assert w >= 30

    def test_clipped_to_max_win(self):
        # Near-random-walk → HL huge → clipped to max_win
        s = pd.Series(np.random.default_rng(0).standard_normal(500).cumsum())
        w = estimate_halflife_window(s, max_win=252)
        assert w <= 252

    def test_empty_series_returns_min_win(self):
        w = estimate_halflife_window(pd.Series([], dtype=float), min_win=30)
        assert w == 30


# ── zscore_from_spread ────────────────────────────────────────────────────────

class TestZscoreFromSpread:
    def test_rolling_no_inf(self):
        s = pd.Series(np.random.randn(200))
        z = zscore_from_spread(s, method="rolling", window=30)
        assert not np.isinf(z.dropna()).any()

    def test_ewm_no_inf(self):
        s = pd.Series(np.random.randn(200))
        z = zscore_from_spread(s, method="ewm", halflife=20)
        assert not np.isinf(z.dropna()).any()

    def test_robust_no_inf(self):
        s = pd.Series(np.random.randn(200))
        z = zscore_from_spread(s, method="robust", window=30)
        assert not np.isinf(z.dropna()).any()

    def test_output_length_matches_input(self):
        s = pd.Series(np.random.randn(150))
        z = zscore_from_spread(s, method="rolling", window=20)
        assert len(z) == len(s)

    def test_preserves_index(self):
        idx = pd.date_range("2020-01-01", periods=100)
        s = pd.Series(np.random.randn(100), index=idx)
        z = zscore_from_spread(s, method="rolling", window=20)
        assert (z.index == idx).all()


# ── generate_pair_signals ─────────────────────────────────────────────────────

class TestGeneratePairSignals:
    def test_returns_expected_columns(self):
        df = _make_df_pair()
        sig = generate_pair_signals(df, z_window=30)
        for col in ["z", "pos", "n1", "n2", "entry", "exit", "stop"]:
            assert col in sig.columns

    def test_length_matches_input(self):
        df = _make_df_pair()
        sig = generate_pair_signals(df, z_window=30)
        assert len(sig) == len(df)

    def test_flat_when_no_signal(self):
        df = _make_df_pair()
        # Very high entry threshold → strategy stays flat
        sig = generate_pair_signals(df, z_entry=100.0, z_window=30)
        assert (sig["pos"] == 0).all()
        assert (sig["n1"] == 0).all()
        assert (sig["n2"] == 0).all()

    def test_dollar_neutral_sizing(self):
        """When in position, |n1*P1| should approximately equal |n2*P2| (beta=1)."""
        df = _make_df_pair()
        sig = generate_pair_signals(df, z_window=30, capital_per_pair=10_000.0)
        in_pos = sig["pos"] != 0
        if in_pos.any():
            notional1 = (sig.loc[in_pos, "n1"].abs() * df.loc[in_pos, "P1"])
            notional2 = (sig.loc[in_pos, "n2"].abs() * df.loc[in_pos, "P2"])
            # Dollar-neutral: both should be roughly equal (beta=1)
            ratio = (notional1 / notional2).dropna()
            assert ((ratio > 0.5) & (ratio < 2.0)).all()

    def test_missing_columns_raises(self):
        df = _make_df_pair().drop(columns=["beta"])
        with pytest.raises(ValueError, match="missing"):
            generate_pair_signals(df, z_window=30)

    def test_exec_lag_zero_ok(self):
        df = _make_df_pair()
        sig = generate_pair_signals(df, z_window=30, exec_lag=0)
        assert len(sig) == len(df)

    def test_negative_exec_lag_raises(self):
        df = _make_df_pair()
        with pytest.raises(ValueError, match="exec_lag"):
            generate_pair_signals(df, z_window=30, exec_lag=-1)

    def test_no_future_lookahead_with_default_lag(self):
        """With exec_lag=1, first bar of executed signals should always be flat."""
        df = _make_df_pair()
        sig = generate_pair_signals(df, z_window=30, exec_lag=1)
        assert sig["pos"].iloc[0] == 0


# ── zscore_from_spread: argument validation ──────────────────────────────────

def test_zscore_unknown_method_raises():
    s = pd.Series(np.random.default_rng(0).standard_normal(100))
    with pytest.raises(ValueError, match="method"):
        zscore_from_spread(s, method="bogus")


# ── zscore_from_spread / generate_pair_signals: history warm-up (no in-window look-ahead) ──

def _ar1(n, phi, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + scale * rng.standard_normal()
    return pd.Series(x)


class TestZscoreHistory:
    def test_history_removes_leading_nans_and_matches_concatenated_calc(self):
        hist = _ar1(300, 0.9, seed=1)
        s = _ar1(60, 0.9, seed=2)
        s.index = pd.RangeIndex(1000, 1060)                      # distinct index from history
        z_plain = zscore_from_spread(s, method="rolling", window=30)
        z_hist = zscore_from_spread(s, method="rolling", window=30, history=hist)
        assert z_plain.iloc[:29].isna().all()                      # no warm-up: first bars undefined
        assert z_hist.notna().all()                                # warmed up on history
        assert z_hist.index.equals(s.index)
        full = zscore_from_spread(pd.concat([hist, s], ignore_index=True), method="rolling", window=30)
        np.testing.assert_allclose(z_hist.to_numpy(), full.iloc[-60:].to_numpy())

    @pytest.mark.parametrize("method", ["rolling", "robust", "ewm"])
    def test_lookback_is_chosen_from_history_not_from_sample(self, method):
        hist = _ar1(400, 0.5, seed=3)          # short half-life
        s = _ar1(80, 0.98, seed=4)             # long half-life — must not influence the look-back
        z_hist = zscore_from_spread(s, method=method, history=hist)
        if method == "ewm":
            hl = estimate_halflife(hist)
            expected = zscore_from_spread(s, method=method, halflife=float(hl), history=hist)
        else:
            expected = zscore_from_spread(s, method=method, window=estimate_halflife_window(hist), history=hist)
        pd.testing.assert_series_equal(z_hist, expected)

    def test_generate_pair_signals_uses_history(self):
        n, m = 300, 60
        rng = np.random.default_rng(5)
        resid_all = _ar1(n + m, 0.9, seed=6)
        p2 = pd.Series(100 + rng.standard_normal(n + m).cumsum())
        p1 = 0.8 * p2 + resid_all
        idx = pd.date_range("2020-01-01", periods=n + m, freq="B")
        df = pd.DataFrame({"resid": resid_all.values, "beta": 0.8, "P1": p1.values, "P2": p2.values}, index=idx)
        df_hist, df_test = df.iloc[:n], df.iloc[n:]
        sig_plain = generate_pair_signals(df_test, z_method="rolling", z_window=30)
        sig_hist = generate_pair_signals(df_test, z_method="rolling", z_window=30, z_history=df_hist["resid"])
        assert sig_plain["z"].iloc[:29].isna().all()
        assert sig_hist["z"].notna().all()
        assert sig_hist.index.equals(df_test.index)
