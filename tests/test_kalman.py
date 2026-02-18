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
