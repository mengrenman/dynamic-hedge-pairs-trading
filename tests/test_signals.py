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
