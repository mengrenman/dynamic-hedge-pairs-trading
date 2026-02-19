# tests/test_circuit_breaker.py
"""
Unit and integration tests for pairs.strategies.circuit_breaker.

Test classes
------------
TestCircuitBreakerConfig    - dataclass construction and defaults
TestZHaltWindows            - _find_z_halt_windows() helper
TestDrawdownHaltWindows     - _find_drawdown_halt_windows() helper
TestApplyCircuitBreaker     - apply_circuit_breaker() integration tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.strategies.circuit_breaker import (
    CircuitBreakerConfig,
    apply_circuit_breaker,
    _find_z_halt_windows,
    _find_drawdown_halt_windows,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df_pair(n: int = 300, phi: float = 0.8, seed: int = 0) -> pd.DataFrame:
    """Synthetic df_pair with resid, beta, P1, P2 on a business-day index."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = phi * resid[i - 1] + rng.standard_normal()
    return pd.DataFrame(
        {
            "resid": resid,
            "beta":  np.ones(n),
            "P1":    100 + rng.standard_normal(n).cumsum(),
            "P2":    100 + rng.standard_normal(n).cumsum(),
        },
        index=dates,
    )


def _make_signals(
    pos_array: np.ndarray,
    *,
    n1_array: np.ndarray | None = None,
    n2_array: np.ndarray | None = None,
    z_array: np.ndarray | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a minimal signals DataFrame from arrays.

    pos_array : int array of {-1, 0, +1}, length n.
    """
    n = len(pos_array)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if n1_array is None:
        n1_array = pos_array.astype(float) * 10.0
    if n2_array is None:
        n2_array = -pos_array.astype(float) * 10.0
    if z_array is None:
        z_array = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "z":     z_array.astype(float),
            "pos":   pos_array.astype(int),
            "n1":    n1_array.astype(float),
            "n2":    n2_array.astype(float),
            "entry": np.zeros(n, dtype=bool),
            "exit":  np.zeros(n, dtype=bool),
            "stop":  np.zeros(n, dtype=bool),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# TestCircuitBreakerConfig
# ---------------------------------------------------------------------------

class TestCircuitBreakerConfig:
    """Tests for the CircuitBreakerConfig dataclass."""

    def test_default_construction(self):
        cfg = CircuitBreakerConfig()
        assert cfg.z_halt is None
        assert cfg.cb_cooldown_bars == 10
        assert cfg.z_reentry is None
        assert cfg.max_drawdown_pct is None
        assert cfg.drawdown_window_bars == 63
        assert cfg.halt_bars == 20
        assert cfg.capital_base is None

    def test_custom_construction(self):
        cfg = CircuitBreakerConfig(
            z_halt=4.5,
            cb_cooldown_bars=15,
            z_reentry=2.0,
            max_drawdown_pct=0.10,
            halt_bars=25,
            capital_base=10_000.0,
        )
        assert cfg.z_halt == 4.5
        assert cfg.z_reentry == 2.0
        assert cfg.max_drawdown_pct == 0.10
        assert cfg.capital_base == 10_000.0

    def test_evaluate_kwargs_default_empty(self):
        cfg = CircuitBreakerConfig()
        assert cfg.evaluate_kwargs == {}

    def test_evaluate_kwargs_independent_instances(self):
        """Two separate instances must not share the same evaluate_kwargs dict."""
        cfg1 = CircuitBreakerConfig()
        cfg2 = CircuitBreakerConfig()
        cfg1.evaluate_kwargs["cost_bps"] = 5
        assert "cost_bps" not in cfg2.evaluate_kwargs


# ---------------------------------------------------------------------------
# TestZHaltWindows
# ---------------------------------------------------------------------------

class TestZHaltWindows:
    """Unit tests for _find_z_halt_windows()."""

    def test_spike_triggers_halt_on_trigger_and_cooldown_bars(self):
        """Trigger bar + cb_cooldown_bars subsequent bars should be halted."""
        n = 20
        z = np.ones(n) * 0.5
        z[5] = 5.0          # spike at bar 5, z_halt=4.0, cooldown=3
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=3, z_reentry=None)
        # bars 5, 6, 7, 8 halted (trigger + 3 cooldown)
        assert not halted.iloc[4]
        for i in range(5, 9):
            assert halted.iloc[i], f"bar {i} should be halted"
        assert not halted.iloc[9]

    def test_negative_spike_also_triggers(self):
        """Negative z blow-out (|z| > z_halt) must also trigger a halt."""
        n = 20
        z = np.ones(n) * 0.5
        z[10] = -5.0
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=2, z_reentry=None)
        assert not halted.iloc[9]
        for i in range(10, 13):
            assert halted.iloc[i], f"bar {i} should be halted"
        assert not halted.iloc[13]

    def test_no_spike_all_false(self):
        """No z above threshold: entire halted series must be False."""
        n = 50
        z = np.clip(np.random.default_rng(0).standard_normal(n), -2.0, 2.0)
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=5, z_reentry=None)
        assert halted.sum() == 0

    def test_overlapping_spikes_merge(self):
        """Two close spikes whose cooldown windows overlap must merge into one."""
        n = 30
        z = np.ones(n) * 0.5
        z[5] = 5.0   # trigger 1 → halt 5–8
        z[7] = 5.0   # trigger 2 → halt 7–10; merged → 5–10
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=3, z_reentry=None)
        assert halted.iloc[5:11].all()
        assert not halted.iloc[11]

    def test_cooldown_bars_zero(self):
        """With cb_cooldown_bars=0, only the trigger bar itself is halted."""
        n = 20
        z = np.ones(n) * 0.5
        z[10] = 5.0
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=0, z_reentry=None)
        assert not halted.iloc[9]
        assert halted.iloc[10]
        assert not halted.iloc[11]

    def test_trigger_on_last_bar_does_not_raise(self):
        """Trigger on the last bar must not raise and the last bar must be halted."""
        n = 10
        z = np.ones(n) * 0.5
        z[-1] = 5.0
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=5, z_reentry=None)
        assert halted.iloc[-1]
        assert not halted.iloc[-2]

    def test_z_reentry_extends_halt(self):
        """After cooldown, |z| >= z_reentry should extend the halt bar-by-bar."""
        n = 30
        z = np.ones(n) * 0.5
        z[5] = 5.0   # trigger; cooldown_bars=1 → mandatory halt on bars 5, 6
        z[7] = 3.0   # still >= z_reentry=2.0 → extend to bar 7
        z[8] = 1.0   # below z_reentry → stop extending
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=1, z_reentry=2.0)
        assert halted.iloc[5]
        assert halted.iloc[6]
        assert halted.iloc[7]
        assert not halted.iloc[8]

    def test_nan_z_never_triggers(self):
        """NaN z values must not trigger a halt."""
        n = 20
        z = np.ones(n) * 0.5
        z[5] = np.nan
        sig = _make_signals(np.zeros(n, dtype=int), z_array=z)
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=3, z_reentry=None)
        assert halted.sum() == 0

    def test_return_type_and_index(self):
        """Return value must be a boolean Series with the same index as signals."""
        n = 30
        sig = _make_signals(np.zeros(n, dtype=int))
        halted = _find_z_halt_windows(sig, z_halt=4.0, cb_cooldown_bars=3, z_reentry=None)
        assert isinstance(halted, pd.Series)
        assert halted.dtype == bool
        assert (halted.index == sig.index).all()


# ---------------------------------------------------------------------------
# TestDrawdownHaltWindows
# ---------------------------------------------------------------------------

class TestDrawdownHaltWindows:
    """Unit tests for _find_drawdown_halt_windows()."""

    def _make_losing_pair_and_signals(self, n: int = 200, seed: int = 42):
        """Synthetic losing scenario: long leg1 while leg1 price falls steadily."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        P1 = 100 - np.linspace(0, 30, n) + rng.standard_normal(n) * 0.5
        P2 = np.ones(n) * 100 + rng.standard_normal(n) * 0.5
        df_pair = pd.DataFrame(
            {
                "resid": P1 - P2,
                "beta":  np.ones(n),
                "P1": P1,
                "P2": P2,
            },
            index=dates,
        )
        pos = np.zeros(n, dtype=int)
        pos[:100] = 1   # long for first half → continuous losses as P1 falls
        sig = _make_signals(pos, n1_array=pos.astype(float), n2_array=-pos.astype(float))
        return df_pair, sig

    def test_deep_drawdown_triggers_halt(self):
        df_pair, sig = self._make_losing_pair_and_signals()
        halted, _ = _find_drawdown_halt_windows(
            sig, df_pair,
            max_drawdown_pct=0.03,
            drawdown_window_bars=30,
            halt_bars=10,
            capital_base=100.0,
            evaluate_kwargs={},
        )
        assert halted.sum() > 0, "Expected at least one halted bar from deep drawdown"

    def test_profitable_signals_no_halt(self):
        """Monotonically profitable signals should never fire the drawdown trigger."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        P1 = 100 + np.linspace(0, 50, n)  # steady gain
        P2 = np.ones(n) * 100
        df_pair = pd.DataFrame(
            {"resid": P1 - P2, "beta": np.ones(n), "P1": P1, "P2": P2},
            index=dates,
        )
        pos = np.ones(n, dtype=int)
        sig = _make_signals(pos)
        halted, _ = _find_drawdown_halt_windows(
            sig, df_pair,
            max_drawdown_pct=0.05,
            drawdown_window_bars=30,
            halt_bars=10,
            capital_base=1_000.0,
            evaluate_kwargs={},
        )
        assert halted.sum() == 0

    def test_returns_tuple_of_series_and_dataframe(self):
        df_pair, sig = self._make_losing_pair_and_signals()
        result = _find_drawdown_halt_windows(
            sig, df_pair,
            max_drawdown_pct=0.05,
            drawdown_window_bars=30,
            halt_bars=10,
            capital_base=100.0,
            evaluate_kwargs={},
        )
        assert len(result) == 2
        halted, daily = result
        assert isinstance(halted, pd.Series)
        assert halted.dtype == bool
        assert isinstance(daily, pd.DataFrame)

    def test_halted_index_aligned_to_signals(self):
        df_pair, sig = self._make_losing_pair_and_signals()
        halted, _ = _find_drawdown_halt_windows(
            sig, df_pair,
            max_drawdown_pct=0.05,
            drawdown_window_bars=30,
            halt_bars=10,
            capital_base=100.0,
            evaluate_kwargs={},
        )
        assert (halted.index == sig.index).all()


# ---------------------------------------------------------------------------
# TestApplyCircuitBreaker
# ---------------------------------------------------------------------------

class TestApplyCircuitBreaker:
    """Integration tests for apply_circuit_breaker()."""

    # ---- basic API -------------------------------------------------------

    def test_returns_two_element_tuple(self):
        df_pair = _make_df_pair(n=50)
        sig = _make_signals(np.zeros(50, dtype=int))
        result = apply_circuit_breaker(sig, df_pair, z_halt=4.0)
        assert len(result) == 2

    def test_returns_signals_df_and_audit_df(self):
        df_pair = _make_df_pair(n=50)
        sig = _make_signals(np.zeros(50, dtype=int))
        sig_cb, audit = apply_circuit_breaker(sig, df_pair, z_halt=4.0)
        assert isinstance(sig_cb, pd.DataFrame)
        assert isinstance(audit, pd.DataFrame)

    def test_output_signals_same_columns(self):
        df_pair = _make_df_pair(n=50)
        sig = _make_signals(np.zeros(50, dtype=int))
        sig_cb, _ = apply_circuit_breaker(sig, df_pair, z_halt=4.0)
        assert set(sig_cb.columns) == set(sig.columns)

    def test_output_signals_same_index(self):
        df_pair = _make_df_pair(n=50)
        sig = _make_signals(np.zeros(50, dtype=int))
        sig_cb, _ = apply_circuit_breaker(sig, df_pair, z_halt=4.0)
        assert (sig_cb.index == sig.index).all()

    def test_does_not_mutate_input_signals(self):
        df_pair = _make_df_pair(n=50)
        n = 50
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[20] = 5.0
        sig = _make_signals(pos, z_array=z)
        pos_before = sig["pos"].copy()
        apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=5)
        pd.testing.assert_series_equal(sig["pos"], pos_before)

    # ---- z-halt trigger --------------------------------------------------

    def test_z_spike_forces_flat_in_halt_window(self):
        """Bars within a z-halt window must have pos=0, n1=0, n2=0."""
        n = 50
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[20] = 5.0   # trigger at bar 20, cooldown=5 → halt bars 20-25
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        sig_cb, _ = apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=5)
        assert (sig_cb["pos"].iloc[20:26] == 0).all()
        assert (sig_cb["n1"].iloc[20:26]  == 0.0).all()
        assert (sig_cb["n2"].iloc[20:26]  == 0.0).all()
        # Bars outside the window are unchanged
        assert sig_cb["pos"].iloc[19] == 1
        assert sig_cb["pos"].iloc[26] == 1

    def test_entry_exit_stop_preserved_in_halt_window(self):
        """entry/exit/stop flags must NOT be zeroed by the circuit breaker."""
        n = 30
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[10] = 5.0
        df_pair = _make_df_pair(n=n)
        entry = np.zeros(n, dtype=bool)
        entry[10] = True   # coincides with trigger
        sig = _make_signals(pos, z_array=z)
        sig["entry"] = entry
        sig_cb, _ = apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=3)
        # entry flag on the trigger bar is preserved
        assert sig_cb["entry"].iloc[10] == True

    # ---- audit DataFrame -------------------------------------------------

    def test_audit_has_required_columns(self):
        n = 50
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[10] = 6.0
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        _, audit = apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=3)
        for col in [
            "trigger_type", "z_at_trigger", "dd_pct_at_trigger",
            "halt_start", "halt_end", "n_halted_bars",
        ]:
            assert col in audit.columns, f"Missing column: {col}"

    def test_audit_trigger_type_is_z_halt_when_only_z_trigger(self):
        n = 50
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[15] = 6.0
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        _, audit = apply_circuit_breaker(
            sig, df_pair, z_halt=4.0, cb_cooldown_bars=3, max_drawdown_pct=None
        )
        assert len(audit) > 0
        assert (audit["trigger_type"] == "z_halt").all()

    def test_audit_halt_start_end_are_timestamps(self):
        n = 50
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[10] = 6.0
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        _, audit = apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=3)
        for ts in list(audit["halt_start"]) + list(audit["halt_end"]):
            assert isinstance(ts, pd.Timestamp), f"Expected Timestamp, got {type(ts)}"

    def test_overlapping_z_windows_produce_single_audit_row(self):
        """Two spikes with overlapping cooldowns → one contiguous window in audit."""
        n = 30
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[5] = 5.0   # halt 5–8
        z[7] = 5.0   # halt 7–10; merged → 5–10
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        _, audit = apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=3)
        assert len(audit) == 1
        assert audit["n_halted_bars"].iloc[0] == 6  # bars 5,6,7,8,9,10

    # ---- no-op / disabled paths ------------------------------------------

    def test_no_trigger_returns_signals_unchanged(self):
        n = 100
        z = np.clip(np.random.default_rng(7).standard_normal(n), -2.0, 2.0)
        pos = np.zeros(n, dtype=int)
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        sig_cb, audit = apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=5)
        pd.testing.assert_frame_equal(sig_cb, sig)
        assert len(audit) == 0

    def test_no_trigger_empty_audit(self):
        n = 100
        z = np.clip(np.random.default_rng(7).standard_normal(n), -2.0, 2.0)
        pos = np.zeros(n, dtype=int)
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        _, audit = apply_circuit_breaker(sig, df_pair, z_halt=4.0)
        assert len(audit) == 0

    def test_all_flat_signals_is_noop(self):
        """All-flat input: both triggers disabled even with high z."""
        n = 50
        pos = np.zeros(n, dtype=int)
        z = np.ones(n) * 10.0
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        sig_cb, audit = apply_circuit_breaker(
            sig, df_pair, z_halt=4.0, cb_cooldown_bars=5, max_drawdown_pct=0.05
        )
        pd.testing.assert_frame_equal(sig_cb, sig)
        assert len(audit) == 0

    def test_both_triggers_disabled_noop(self):
        n = 100
        pos = np.ones(n, dtype=int)
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos)
        sig_cb, audit = apply_circuit_breaker(
            sig, df_pair, z_halt=None, max_drawdown_pct=None
        )
        pd.testing.assert_frame_equal(sig_cb, sig)
        assert len(audit) == 0

    def test_z_halt_none_ignores_large_z(self):
        n = 50
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 10.0
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        sig_cb, audit = apply_circuit_breaker(
            sig, df_pair, z_halt=None, max_drawdown_pct=None
        )
        assert (sig_cb["pos"] == sig["pos"]).all()
        assert len(audit) == 0

    # ---- edge cases ------------------------------------------------------

    def test_trigger_on_last_bar_no_raise(self):
        n = 20
        pos = np.ones(n, dtype=int)
        z = np.ones(n) * 1.0
        z[-1] = 6.0
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(pos, z_array=z)
        sig_cb, audit = apply_circuit_breaker(
            sig, df_pair, z_halt=4.0, cb_cooldown_bars=5
        )
        assert sig_cb["pos"].iloc[-1] == 0
        assert len(audit) >= 1

    # ---- drawdown trigger ------------------------------------------------

    def test_drawdown_trigger_fires_on_losing_strategy(self):
        """A steadily losing strategy must trigger the drawdown halt.

        Uses a large, deterministic loss (P1 falls 60 points over 100 bars) and
        a generous threshold (3 % of $100 capital = $3) so the trigger reliably
        fires regardless of seeding.
        """
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Strictly linear decline — no randomness — guarantees trigger fires
        P1 = 100 - np.linspace(0, 60, n)
        P2 = np.ones(n) * 100
        df_pair = pd.DataFrame(
            {"resid": P1 - P2, "beta": np.ones(n), "P1": P1, "P2": P2},
            index=dates,
        )
        pos = np.zeros(n, dtype=int)
        pos[:100] = 1   # long for first half while P1 falls → large loss
        z = np.ones(n) * 0.5
        sig = _make_signals(pos, n1_array=pos.astype(float),
                            n2_array=-pos.astype(float), z_array=z)
        sig_cb, audit = apply_circuit_breaker(
            sig, df_pair,
            z_halt=None,
            max_drawdown_pct=0.03,   # 3 % of $100 = $3; easily exceeded
            drawdown_window_bars=20,
            halt_bars=10,
            capital_base=100.0,
        )
        assert len(audit) > 0, "Expected drawdown trigger to fire"
        assert (audit["trigger_type"] == "drawdown").all()

    def test_drawdown_trigger_type_in_audit(self):
        """When only the drawdown trigger fires, trigger_type must be 'drawdown'."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        P1 = 100 - np.linspace(0, 60, n)
        P2 = np.ones(n) * 100
        df_pair = pd.DataFrame(
            {"resid": P1 - P2, "beta": np.ones(n), "P1": P1, "P2": P2},
            index=dates,
        )
        pos = np.zeros(n, dtype=int)
        pos[:100] = 1
        # keep z well within z_halt to isolate drawdown trigger
        z = np.ones(n) * 0.5
        sig = _make_signals(pos, n1_array=pos.astype(float),
                            n2_array=-pos.astype(float), z_array=z)
        sig_cb, audit = apply_circuit_breaker(
            sig, df_pair,
            z_halt=10.0,          # effectively disabled (z never reaches 10)
            cb_cooldown_bars=2,
            max_drawdown_pct=0.03,
            drawdown_window_bars=20,
            halt_bars=10,
            capital_base=100.0,
        )
        dd_rows = audit[audit["trigger_type"] == "drawdown"]
        assert len(dd_rows) > 0

    # ---- validation / error handling ------------------------------------

    def test_missing_pos_column_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        bad = pd.DataFrame({"z": np.ones(n), "n1": np.ones(n), "n2": np.ones(n)},
                           index=pd.date_range("2020-01-01", periods=n, freq="B"))
        with pytest.raises(ValueError, match="missing required columns"):
            apply_circuit_breaker(bad, df_pair, z_halt=4.0)

    def test_invalid_z_halt_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="z_halt"):
            apply_circuit_breaker(sig, df_pair, z_halt=-1.0)

    def test_z_halt_zero_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="z_halt"):
            apply_circuit_breaker(sig, df_pair, z_halt=0.0)

    def test_invalid_max_drawdown_pct_over_one_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            apply_circuit_breaker(sig, df_pair, max_drawdown_pct=1.5)

    def test_invalid_max_drawdown_pct_zero_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            apply_circuit_breaker(sig, df_pair, max_drawdown_pct=0.0)

    def test_drawdown_trigger_without_df_pair_raises(self):
        n = 30
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="df_pair"):
            apply_circuit_breaker(sig, None, max_drawdown_pct=0.05)

    def test_halt_bars_less_than_one_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="halt_bars"):
            apply_circuit_breaker(sig, df_pair, max_drawdown_pct=0.05, halt_bars=0)

    def test_negative_cooldown_bars_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        sig = _make_signals(np.zeros(n, dtype=int))
        with pytest.raises(ValueError, match="cb_cooldown_bars"):
            apply_circuit_breaker(sig, df_pair, z_halt=4.0, cb_cooldown_bars=-1)

    def test_missing_z_column_with_z_halt_raises(self):
        n = 30
        df_pair = _make_df_pair(n=n)
        pos = np.ones(n, dtype=int)
        sig = _make_signals(pos)
        sig_no_z = sig.drop(columns=["z"])
        with pytest.raises(ValueError, match="'z' column"):
            apply_circuit_breaker(sig_no_z, df_pair, z_halt=4.0)
