# tests/test_kalman.py
"""Unit tests for pairs.models.kalman."""
import numpy as np
import pandas as pd
import pytest

from pairs.models.kalman import (
    _align_pair_multiindex,
    _kalman_dynamic_hedge,
    kalman_dynamic_hedge_joblib,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_multiindex_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    frames = []
    for t in ["A", "B"]:
        prices = 100 + rng.standard_normal(n).cumsum()
        idx = pd.MultiIndex.from_product([[t], dates], names=["ticker", "datetime"])
        frames.append(pd.DataFrame({"close": prices}, index=idx))
    return pd.concat(frames).sort_index()


def _make_aligned_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "P1": 100 + rng.standard_normal(n).cumsum(),
        "P2": 100 + rng.standard_normal(n).cumsum(),
    }, index=dates)


# ── _align_pair_multiindex ────────────────────────────────────────────────────

class TestAlignPairMultiindex:
    def test_returns_p1_p2_columns(self):
        data = _make_multiindex_df()
        df = _align_pair_multiindex(data, "A", "B")
        assert "P1" in df.columns and "P2" in df.columns

    def test_no_nans_in_output(self):
        data = _make_multiindex_df()
        df = _align_pair_multiindex(data, "A", "B")
        assert not df.isna().any().any()

    def test_raises_without_close_column(self):
        data = _make_multiindex_df().rename(columns={"close": "price"})
        with pytest.raises(ValueError, match="close"):
            _align_pair_multiindex(data, "A", "B")

    def test_raises_without_multiindex(self):
        data = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="MultiIndex"):
            _align_pair_multiindex(data, "A", "B")


# ── _kalman_dynamic_hedge ─────────────────────────────────────────────────────

class TestKalmanDynamicHedge:
    def test_returns_states_dataframe(self):
        df = _make_aligned_df(100)
        k1, k2, states, _ = _kalman_dynamic_hedge("A", "B", df)
        assert states is not None
        assert isinstance(states, pd.DataFrame)

    def test_states_has_expected_columns(self):
        df = _make_aligned_df(100)
        _, _, states, _ = _kalman_dynamic_hedge("A", "B", df)
        for col in ["alpha", "beta", "y_hat", "resid"]:
            assert col in states.columns

    def test_states_length_matches_input(self):
        df = _make_aligned_df(100)
        _, _, states, _ = _kalman_dynamic_hedge("A", "B", df)
        assert len(states) == len(df)

    def test_too_few_rows_returns_none(self):
        df = _make_aligned_df(3)
        _, _, states, _ = _kalman_dynamic_hedge("A", "B", df)
        assert states is None

    def test_invalid_q_raises(self):
        df = _make_aligned_df(50)
        with pytest.raises(ValueError, match="q"):
            _kalman_dynamic_hedge("A", "B", df, q=-1e-5)

    def test_invalid_r_raises(self):
        df = _make_aligned_df(50)
        with pytest.raises(ValueError, match="r"):
            _kalman_dynamic_hedge("A", "B", df, r=0.0)

    def test_invalid_init_cov_raises(self):
        df = _make_aligned_df(50)
        with pytest.raises(ValueError, match="init_cov"):
            _kalman_dynamic_hedge("A", "B", df, init_cov=-100.0)

    def test_return_params_returns_tuple(self):
        df = _make_aligned_df(50)
        _, _, states, params = _kalman_dynamic_hedge("A", "B", df, return_params=True)
        assert params is not None
        assert "F" in params and "Q" in params and "R" in params

    def test_no_nan_in_resid(self):
        df = _make_aligned_df(100)
        _, _, states, _ = _kalman_dynamic_hedge("A", "B", df)
        assert not states["resid"].isna().any()


# ── kalman_dynamic_hedge_joblib ───────────────────────────────────────────────

class TestKalmanDynamicHedgeJoblib:
    def test_returns_dict_of_dataframes(self):
        data = _make_multiindex_df(100)
        result = kalman_dynamic_hedge_joblib(data, pairs=[("A", "B")],
                                             show_progress=False, n_workers=1)
        assert isinstance(result, dict)
        assert ("A", "B") in result
        assert isinstance(result[("A", "B")], pd.DataFrame)

    def test_missing_close_column_raises(self):
        data = _make_multiindex_df().rename(columns={"close": "price"})
        with pytest.raises(ValueError, match="close"):
            kalman_dynamic_hedge_joblib(data, show_progress=False)

    def test_flat_index_raises(self):
        data = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="MultiIndex"):
            kalman_dynamic_hedge_joblib(data, show_progress=False)

    def test_return_params_true(self):
        data = _make_multiindex_df(100)
        states, params = kalman_dynamic_hedge_joblib(
            data, pairs=[("A", "B")], return_params=True,
            show_progress=False, n_workers=1
        )
        assert ("A", "B") in params
        assert "F" in params[("A", "B")]

    def test_invalid_q_raises(self):
        data = _make_multiindex_df(100)
        with pytest.raises(ValueError, match="q"):
            kalman_dynamic_hedge_joblib(data, pairs=[("A", "B")], q=0.0,
                                        show_progress=False, n_workers=1)


# ── the fitting default must be causal ────────────────────────────────────────

class TestCausalByDefault:
    """The smoother conditions each state on the whole sample; it must never be the default."""

    def _prices(self, n=400, seed=7):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        p2 = 50 + np.cumsum(rng.normal(0, 0.4, n))
        p1 = 1.3 * p2 + np.cumsum(rng.normal(0, 0.15, n))
        return pd.DataFrame({"P1": p1, "P2": p2}, index=idx)

    def test_default_mode_is_filter(self):
        import inspect
        from pairs.models.kalman import _kalman_dynamic_hedge, kalman_dynamic_hedge_joblib
        for fn in (_kalman_dynamic_hedge, kalman_dynamic_hedge_joblib):
            assert inspect.signature(fn).parameters["mode"].default == "filter", fn.__name__

    def test_default_states_do_not_use_future_data(self):
        # truncating the sample must not change the states on the bars that remain
        from pairs.models.kalman import _kalman_dynamic_hedge
        df = self._prices()
        _, _, full, _ = _kalman_dynamic_hedge("P1", "P2", df, q=1e-5, return_params=True)
        _, _, part, _ = _kalman_dynamic_hedge("P1", "P2", df.iloc[:300], q=1e-5, return_params=True)
        pd.testing.assert_frame_equal(full.iloc[:300], part)

    def test_the_smoother_does_use_future_data(self):
        from pairs.models.kalman import _kalman_dynamic_hedge
        df = self._prices()
        _, _, full, _ = _kalman_dynamic_hedge("P1", "P2", df, q=1e-5, mode="smooth", return_params=True)
        _, _, part, _ = _kalman_dynamic_hedge("P1", "P2", df.iloc[:300], q=1e-5, mode="smooth", return_params=True)
        assert not np.allclose(full["resid"].to_numpy()[:300], part["resid"].to_numpy())

    def test_exported_params_are_identical_in_both_modes(self):
        # so an OOS continuation started from them is causal whichever mode produced the states
        from pairs.models.kalman import _kalman_dynamic_hedge
        df = self._prices()
        _, _, _, pf = _kalman_dynamic_hedge("P1", "P2", df, q=1e-5, em_iters=3, mode="filter", return_params=True)
        _, _, _, ps = _kalman_dynamic_hedge("P1", "P2", df, q=1e-5, em_iters=3, mode="smooth", return_params=True)
        for k in ("F", "Q", "R", "last_state_mean", "last_state_cov"):
            assert np.allclose(pf[k], ps[k]), k


class TestIndefiniteNoiseCovariance:
    """EM can return a transition covariance with a negative eigenvalue; pykalman filters with it
    silently, so the package must refuse it."""

    def _constant_hedge_logs(self, n=60, seed=13):
        # log prices with a constant hedge ratio, so the true Q is zero; with the noise scales set
        # relative to the OLS residual variance r0 (the log-price convention), EM drives Q indefinite
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        x = np.log(50) + 0.02 * rng.standard_normal(n).cumsum()
        y = 1.0 + 0.8 * x + 0.05 * rng.standard_normal(n)
        X = np.column_stack([x, np.ones(n)])
        r0 = float(np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]))
        return pd.DataFrame({"P1": y, "P2": x}, index=idx), r0

    def test_case_is_indefinite_under_raw_pykalman(self):
        # guards the fixture: if a pykalman release stops producing the bad Q, the test below
        # would pass for the wrong reason
        from pykalman import KalmanFilter
        df, r0 = self._constant_hedge_logs()
        n = len(df)
        H = np.zeros((n, 1, 2)); H[:, 0, 0] = df["P2"].to_numpy(); H[:, 0, 1] = 1.0
        kf = KalmanFilter(transition_matrices=np.eye(2), observation_matrices=H,
                          initial_state_mean=np.zeros(2), initial_state_covariance=np.eye(2) * 1e6,
                          transition_covariance=np.eye(2) * 1e-5 * r0,
                          observation_covariance=np.array([[r0]]))
        Q = kf.em(df["P1"].to_numpy().reshape(-1, 1), n_iter=5).transition_covariance
        assert np.linalg.eigvalsh((Q + Q.T) / 2).min() < 0

    def test_em_fit_with_indefinite_q_raises(self):
        df, r0 = self._constant_hedge_logs()
        with pytest.raises(ValueError, match="transition_covariance Q is not positive definite"):
            _kalman_dynamic_hedge("P1", "P2", df, q=1e-5 * r0, r=r0, em_iters=5, return_params=True)

    def test_same_data_without_em_fits(self):
        # the check is on what EM learns; the user-supplied q and r are already validated positive
        df, r0 = self._constant_hedge_logs()
        _, _, states, _ = _kalman_dynamic_hedge("P1", "P2", df, q=1e-5 * r0, r=r0, em_iters=0)
        assert states is not None and np.isfinite(states["beta"]).all()

    def test_joblib_wrapper_propagates(self):
        df, r0 = self._constant_hedge_logs()
        data = pd.concat({"A": df["P1"], "B": df["P2"]}, names=["ticker", "datetime"]).to_frame("close")
        with pytest.raises(ValueError, match="not positive definite"):
            kalman_dynamic_hedge_joblib(data, pairs=[("A", "B")], q=1e-5 * r0, r=r0, em_iters=5,
                                        n_workers=1, show_progress=False)

    @pytest.mark.parametrize("Q, R, what", [
        (np.array([[1e-5, 0.0], [0.0, -1e-7]]), np.array([[1.0]]), "transition_covariance Q"),
        (np.array([[1e-5, 2e-5], [2e-5, 1e-5]]), np.array([[1.0]]), "transition_covariance Q"),  # positive diagonal, indefinite
        (np.zeros((2, 2)), np.array([[1.0]]), "transition_covariance Q"),                       # semidefinite is not enough
        (np.eye(2) * 1e-5, np.array([[0.0]]), "observation_covariance R"),
        (np.eye(2) * 1e-5, np.array([[np.nan]]), "observation_covariance R"),
    ])
    def test_filter_kf_on_new_rejects_bad_frozen_params(self, Q, R, what):
        from pairs.models.kalman import filter_kf_on_new
        df = _make_aligned_df()
        with pytest.raises(ValueError, match=what):
            filter_kf_on_new(df["P1"], df["P2"], frozen={"F": np.eye(2), "Q": Q, "R": R})
