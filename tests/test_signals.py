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


# ── session masks: force_flat / block_entry ───────────────────────────────────

def _two_session_pair(n_per_session: int = 6) -> pd.DataFrame:
    """
    Two sessions of minute bars. The residual trends up one unit per bar, so with a 2-bar
    rolling z-score z = +1 on every bar after the first: a z_entry of 1 opens a short spread
    at once and, with z_exit below 1, nothing ever closes it — unless a mask says so.
    """
    idx = pd.DatetimeIndex(
        list(pd.date_range("2024-01-02 09:30", periods=n_per_session, freq="min"))
        + list(pd.date_range("2024-01-03 09:30", periods=n_per_session, freq="min"))
    )
    resid = np.arange(len(idx), dtype=float)
    return pd.DataFrame({"resid": resid, "beta": 1.0, "P1": 100.0, "P2": 50.0}, index=idx)


def _last_bar_of_session(index: pd.DatetimeIndex) -> pd.Series:
    session = index.normalize()
    nxt = np.append(session[1:], [session[-1] + pd.Timedelta(days=99)])
    return pd.Series(session != nxt, index=index)


class TestSessionMasks:
    kw = dict(z_method="rolling", z_window=2, z_entry=1.0, z_exit=0.5, z_stop=100.0)

    def test_default_carries_position_across_sessions(self):
        df = _two_session_pair()
        sig = generate_pair_signals(df, **self.kw)
        assert (sig["pos"].iloc[2:] == -1).all()          # entered at bar 1, executed bar 2, never closed

    def test_force_flat_on_last_bar_closes_overnight_exposure(self):
        df = _two_session_pair()
        flat = _last_bar_of_session(df.index)
        sig = generate_pair_signals(df, force_flat=flat, **self.kw)
        assert sig["pos"].iloc[5] == -1                    # held through the last bar of session 1
        assert sig["pos"].iloc[6] == 0                     # flat decision at bar 5 -> executed at bar 6
        assert bool(sig["exit"].iloc[6])
        assert sig["pos"].iloc[7] == -1                    # re-entered on bar 6, executed bar 7
        assert (sig["pos"].iloc[7:] == -1).all()

    def test_force_flat_covers_exec_lag(self):
        df = _two_session_pair()
        session = df.index.normalize()
        last = _last_bar_of_session(df.index)
        two_last = last | last.shift(-1, fill_value=False)   # last two bars of each session
        sig = generate_pair_signals(df, force_flat=two_last, exec_lag=2, **self.kw)
        assert sig["pos"].iloc[6] == 0 and sig["pos"].iloc[7] == 0   # first two bars of session 2 flat
        assert sig["pos"].iloc[5] == -1

    def test_block_entry_prevents_new_positions_only(self):
        df = _two_session_pair()
        block = pd.Series(False, index=df.index)
        block.iloc[:6] = True                              # no entries in session 1
        sig = generate_pair_signals(df, block_entry=block, **self.kw)
        assert (sig["pos"].iloc[:7] == 0).all()            # session-1 decisions (executed through bar 6) flat
        assert (sig["pos"].iloc[7:] == -1).all()           # session 2 enters at once
        block2 = pd.Series(False, index=df.index)
        block2.iloc[6:] = True                             # blocking entries does not close an open position
        sig2 = generate_pair_signals(df, block_entry=block2, **self.kw)
        assert (sig2["pos"].iloc[2:] == -1).all()

    def test_masks_align_by_index_and_validate_length(self):
        df = _two_session_pair()
        partial = pd.Series(True, index=df.index[:3])      # reindexed; missing entries mean False
        sig = generate_pair_signals(df, force_flat=partial, **self.kw)
        assert (sig["pos"].iloc[4:] == -1).all()           # bars 0-2 flat decisions; entry at bar 3, executed bar 4
        with pytest.raises(ValueError, match="force_flat"):
            generate_pair_signals(df, force_flat=np.ones(3, dtype=bool), **self.kw)

    def test_masks_default_to_no_change(self):
        df = _two_session_pair()
        a = generate_pair_signals(df, **self.kw)
        b = generate_pair_signals(df, force_flat=None, block_entry=None, **self.kw)
        pd.testing.assert_frame_equal(a, b)


# ── session_masks ─────────────────────────────────────────────────────────────

from pairs.strategies.signals import session_masks


class TestSessionMasksHelper:
    def _idx(self):
        return pd.DatetimeIndex(
            list(pd.date_range("2024-01-02 09:30", periods=390, freq="min"))
            + list(pd.date_range("2024-01-03 09:30", periods=210, freq="min")))   # second day an early close

    def test_flatten_flags_last_exec_lag_bars(self):
        idx = self._idx()
        flat, block = session_masks(idx, exec_lag=1)
        assert flat.sum() == 2 and flat.loc["2024-01-02 15:59"] and flat.loc["2024-01-03 12:59"]
        flat2, _ = session_masks(idx, exec_lag=2)
        assert flat2.sum() == 4 and flat2.loc["2024-01-02 15:58"]
        assert (block == flat).all()                              # no cutoff: block only where flat

    def test_no_entry_after_cutoff(self):
        idx = self._idx()
        flat, block = session_masks(idx, exec_lag=1, no_entry_after="15:30")
        assert not flat.loc["2024-01-02 15:30"] and block.loc["2024-01-02 15:30"]
        assert not block.loc["2024-01-02 15:29"]
        assert block.loc["2024-01-02 15:30":"2024-01-02 15:59"].all()
        assert not block.loc["2024-01-03 12:00"]                  # early-close day never reaches the cutoff

    def test_no_flatten(self):
        idx = self._idx()
        flat, block = session_masks(idx, flatten=False, no_entry_after="15:45")
        assert not flat.any() and block.sum() == 15

    def test_feeds_generate_pair_signals(self):
        df = _two_session_pair()
        flat, block = session_masks(df.index, exec_lag=1)
        sig = generate_pair_signals(df, force_flat=flat, block_entry=block, **TestSessionMasks.kw)
        assert sig["pos"].iloc[6] == 0 and sig["pos"].iloc[5] == -1


# ── initial_position: stitching walk-forward folds ────────────────────────────

class TestInitialPosition:
    kw = dict(z_method="rolling", z_window=2, z_entry=1.0, z_exit=0.5, z_stop=100.0)

    def test_default_starts_flat(self):
        df = _two_session_pair()
        sig = generate_pair_signals(df, **self.kw)
        assert sig["pos"].iloc[0] == 0 and sig["n1"].iloc[0] == 0.0

    def test_carried_position_executes_from_the_first_bar(self):
        df = _two_session_pair()
        hist = pd.Series([-1.0])                                   # warm-up so z is defined on the first bar
        sig = generate_pair_signals(df, z_history=hist, initial_position=(-1, -40.0, 40.0), **self.kw)
        assert sig["pos"].iloc[0] == -1 and sig["n1"].iloc[0] == -40.0 and sig["n2"].iloc[0] == 40.0
        # from bar 1 the sizes are the window's own (dollar-neutral at its prices), same side
        assert sig["pos"].iloc[1] == -1 and sig["n1"].iloc[1] != -40.0
        assert not sig["entry"].any()                              # never re-entered: it was already short

    def test_carried_position_is_managed_by_the_rules(self):
        df = _two_session_pair()
        df["resid"] = 0.0                                          # z = NaN (zero std) -> "no information" -> flat
        sig = generate_pair_signals(df, initial_position=(1, 50.0, -50.0), **self.kw)
        assert sig["pos"].iloc[0] == 1 and (sig["pos"].iloc[1:] == 0).all()

    def test_stitching_two_folds_equals_one_run(self):
        df = _two_session_pair(n_per_session=8)
        whole = generate_pair_signals(df, **self.kw)
        a = generate_pair_signals(df.iloc[:8], **self.kw)
        last = a.iloc[-1]
        b = generate_pair_signals(df.iloc[8:], z_history=df["resid"].iloc[:8],      # the look-back carries over too
                                  initial_position=(int(last["pos"]), float(last["n1"]), float(last["n2"])), **self.kw)
        stitched = pd.concat([a, b])
        pd.testing.assert_series_equal(stitched["pos"], whole["pos"])
        # holdings agree except on the boundary bar, where the second fold re-sizes to its own first prices
        assert np.allclose(stitched["n1"].iloc[:8], whole["n1"].iloc[:8]) and np.allclose(stitched["n1"].iloc[9:], whole["n1"].iloc[9:])

    def test_zero_initial_position_ignores_sizes(self):
        df = _two_session_pair()
        a = generate_pair_signals(df, initial_position=(0, 12.0, -3.0), **self.kw)
        b = generate_pair_signals(df, **self.kw)
        pd.testing.assert_frame_equal(a, b)


def test_robust_zscore_needs_two_windows_of_history():
    rng = np.random.default_rng(1)
    hist = pd.Series(rng.normal(size=200))
    s = pd.Series(rng.normal(size=50))
    z_one = zscore_from_spread(s, "robust", window=20, history=hist.iloc[-20:])
    z_two = zscore_from_spread(s, "robust", window=20, history=hist.iloc[-39:])
    assert z_one.iloc[:18].isna().all() and z_one.iloc[18:].notna().all()   # window-2 bars lost with one window of history
    assert z_two.notna().all()
    z_roll = zscore_from_spread(s, "rolling", window=20, history=hist.iloc[-20:])
    assert z_roll.notna().all()


class TestNoEntryBeyondStop:
    def _df(self, z_path):
        # a residual whose 2-bar rolling z reproduces z_path is awkward; use z_history=None with a
        # long flat history so rolling mean 0 / std 1 and the residual IS the z-score
        n = len(z_path)
        idx = pd.date_range("2024-01-02 09:30", periods=n, freq="min")
        return pd.DataFrame({"resid": z_path, "beta": 1.0, "P1": 100.0, "P2": 50.0}, index=idx)

    def _hist(self):
        rng = np.random.default_rng(0)
        h = rng.normal(size=400)
        return pd.Series((h - h.mean()) / h.std(ddof=0))                 # mean 0, std 1 -> z ≈ resid

    def test_entry_at_or_beyond_stop_is_not_taken(self):
        df = self._df([0.0, 4.5, 4.6, 4.7, 0.0, 0.0])
        sig = generate_pair_signals(df, z_method="rolling", z_window=400, z_history=self._hist(),
                                    z_entry=2.0, z_exit=0.5, z_stop=4.0)
        assert not sig["entry"].any() and (sig["pos"] == 0).all()

    def test_reentry_only_once_back_inside_the_band(self):
        # enter at 2.5, blow out to 4.5 (stop), stay at 4.5 (no re-entry), come back to 3 (re-entry), revert to 0 (exit)
        df = self._df([0.0, 2.5, 4.5, 4.5, 4.5, 3.0, 3.0, 0.0, 0.0])
        sig = generate_pair_signals(df, z_method="rolling", z_window=400, z_history=self._hist(),
                                    z_entry=2.0, z_exit=0.5, z_stop=4.0)
        ent = list(np.flatnonzero(sig["entry"].to_numpy()))
        stp = list(np.flatnonzero(sig["stop"].to_numpy()))
        assert stp == [3]                                   # stop decided at bar 2, executed bar 3
        assert ent == [2, 6]                                # entries decided at bars 1 and 5 (executed +1)
        assert (sig["pos"].iloc[4:6] == 0).all()            # flat while |z| >= stop
