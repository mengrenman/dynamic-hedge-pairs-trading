# tests/test_portfolio.py
"""Tests for pairs.stats.portfolio — pair correlation & diversification analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.stats.portfolio import (
    pair_return_correlations,
    portfolio_diversification_score,
    suggest_position_weights,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_kf_results(
    n_pairs: int = 3,
    n: int = 252,
    phi: float = 0.8,
    seed: int = 0,
) -> dict:
    """
    Synthetic kf_results dict: {(f'T{i}', f'T{j}'): df}.
    Each df has 'resid' (AR(1)) and 'beta' (constant 1.0) columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    results = {}
    tickers = [f"T{i}" for i in range(n_pairs + 1)]
    for i in range(n_pairs):
        k1, k2 = tickers[i], tickers[i + 1]
        resid = np.zeros(n)
        for t in range(1, n):
            resid[t] = phi * resid[t - 1] + rng.standard_normal()
        df = pd.DataFrame(
            {"resid": resid, "beta": np.ones(n), "alpha": np.zeros(n)},
            index=dates,
        )
        results[(k1, k2)] = df
    return results


def _make_identical_kf_results(n: int = 200, seed: int = 42) -> dict:
    """Two pairs with identical resid series → correlation = 1.0."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    resid = rng.standard_normal(n).cumsum()
    df = pd.DataFrame({"resid": resid, "beta": np.ones(n)}, index=dates)
    return {
        ("A", "B"): df.copy(),
        ("A", "C"): df.copy(),
    }


# ---------------------------------------------------------------------------
# TestPairReturnCorrelations
# ---------------------------------------------------------------------------

class TestPairReturnCorrelations:

    def test_returns_square_symmetric_dataframe(self):
        kf = _make_kf_results(n_pairs=3)
        corr = pair_return_correlations(kf)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1] == 3

    def test_symmetric(self):
        kf = _make_kf_results(n_pairs=4)
        corr = pair_return_correlations(kf)
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_diagonal_is_one(self):
        kf = _make_kf_results(n_pairs=3)
        corr = pair_return_correlations(kf)
        diag = np.diag(corr.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_identical_resid_gives_corr_one(self):
        kf = _make_identical_kf_results()
        corr = pair_return_correlations(kf, min_overlap=5)
        # Both off-diagonal entries should be ≈ 1.0
        labels = list(corr.columns)
        assert len(labels) == 2
        off = corr.loc[labels[0], labels[1]]
        assert abs(off - 1.0) < 1e-9

    def test_labels_contain_slash(self):
        kf = _make_kf_results(n_pairs=2)
        corr = pair_return_correlations(kf)
        for col in corr.columns:
            assert "/" in col

    def test_single_pair_returns_1x1(self):
        kf = _make_kf_results(n_pairs=1)
        corr = pair_return_correlations(kf)
        assert corr.shape == (1, 1)
        assert abs(corr.iloc[0, 0] - 1.0) < 1e-10

    def test_non_overlapping_index_gives_nan(self):
        """Pairs with non-overlapping date ranges should produce NaN off-diagonals."""
        dates1 = pd.date_range("2020-01-01", periods=100, freq="B")
        dates2 = pd.date_range("2022-01-01", periods=100, freq="B")
        rng = np.random.default_rng(0)
        df1 = pd.DataFrame({"resid": rng.standard_normal(100), "beta": np.ones(100)}, index=dates1)
        df2 = pd.DataFrame({"resid": rng.standard_normal(100), "beta": np.ones(100)}, index=dates2)
        kf = {("A", "B"): df1, ("C", "D"): df2}
        corr = pair_return_correlations(kf, min_overlap=30)
        labels = list(corr.columns)
        assert np.isnan(corr.loc[labels[0], labels[1]])
        assert np.isnan(corr.loc[labels[1], labels[0]])

    def test_spearman_method_works(self):
        kf = _make_kf_results(n_pairs=3)
        corr = pair_return_correlations(kf, method="spearman")
        assert corr.shape == (3, 3)
        diag = np.diag(corr.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_invalid_method_raises(self):
        kf = _make_kf_results(n_pairs=2)
        with pytest.raises(ValueError, match="method"):
            pair_return_correlations(kf, method="kendall")

    def test_empty_kf_results_returns_empty(self):
        corr = pair_return_correlations({})
        assert corr.empty


# ---------------------------------------------------------------------------
# TestPortfolioDiversificationScore
# ---------------------------------------------------------------------------

class TestPortfolioDiversificationScore:

    def test_single_pair_returns_nan(self):
        corr = pd.DataFrame([[1.0]], index=["A/B"], columns=["A/B"])
        score = portfolio_diversification_score(corr)
        assert np.isnan(score)

    def test_identity_matrix_returns_inf(self):
        """Two uncorrelated pairs (off-diag = 0) → infinite diversification."""
        corr = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A/B", "C/D"],
            columns=["A/B", "C/D"],
        )
        score = portfolio_diversification_score(corr)
        assert score == float("inf")

    def test_all_correlated_gives_low_score(self):
        """Off-diagonal ≈ 1.0 → score ≈ 1."""
        val = 0.99
        corr = pd.DataFrame(
            [[1.0, val], [val, 1.0]],
            index=["A/B", "C/D"],
            columns=["A/B", "C/D"],
        )
        score = portfolio_diversification_score(corr)
        assert score < 1.1

    def test_uncorrelated_pairs_give_high_score(self):
        kf = _make_kf_results(n_pairs=4, seed=99)
        corr = pair_return_correlations(kf)
        score = portfolio_diversification_score(corr)
        assert np.isfinite(score)
        assert score > 0

    def test_score_positive(self):
        kf = _make_kf_results(n_pairs=3)
        corr = pair_return_correlations(kf)
        score = portfolio_diversification_score(corr)
        assert score > 0

    def test_empty_corr_returns_nan(self):
        corr = pd.DataFrame()
        score = portfolio_diversification_score(corr)
        assert np.isnan(score)


# ---------------------------------------------------------------------------
# TestSuggestPositionWeights
# ---------------------------------------------------------------------------

class TestSuggestPositionWeights:

    def _get_corr(self, kf):
        return pair_return_correlations(kf, min_overlap=2)

    def test_weights_sum_to_one(self):
        kf = _make_kf_results(n_pairs=3)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr)
        assert abs(w["weight"].sum() - 1.0) < 1e-10

    def test_max_weight_clipping_enforced(self):
        kf = _make_kf_results(n_pairs=4)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, max_weight=0.30)
        assert w["weight"].max() <= 0.30 + 1e-10

    def test_inv_var_gives_lower_weight_to_high_var_pair(self):
        """One pair with very large residual variance should get lower weight."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        low_var = pd.DataFrame({"resid": rng.standard_normal(300) * 0.01, "beta": np.ones(300)}, index=dates)
        high_var = pd.DataFrame({"resid": rng.standard_normal(300) * 10.0, "beta": np.ones(300)}, index=dates)
        kf = {("A", "B"): low_var, ("C", "D"): high_var}
        corr = pair_return_correlations(kf, min_overlap=2)
        w = suggest_position_weights(kf, corr, method="inv_var")
        w = w.set_index("pair")
        assert w.loc["A/B", "weight"] > w.loc["C/D", "weight"]

    def test_equal_method_gives_equal_weights(self):
        kf = _make_kf_results(n_pairs=4)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, method="equal")
        weights = w["weight"].values
        np.testing.assert_allclose(weights, weights[0], atol=1e-10)

    def test_required_columns_present(self):
        kf = _make_kf_results(n_pairs=3)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr)
        for col in ["pair", "resid_var", "inv_var_weight", "weight", "suggested_capital_pct"]:
            assert col in w.columns

    def test_single_pair_weight_is_one(self):
        kf = _make_kf_results(n_pairs=1)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr)
        assert abs(w["weight"].iloc[0] - 1.0) < 1e-10

    def test_capital_pct_equals_weight_times_100(self):
        kf = _make_kf_results(n_pairs=3)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr)
        np.testing.assert_allclose(w["suggested_capital_pct"].values, w["weight"].values * 100, atol=1e-10)

    def test_invalid_method_raises(self):
        kf = _make_kf_results(n_pairs=2)
        corr = self._get_corr(kf)
        with pytest.raises(ValueError, match="method"):
            suggest_position_weights(kf, corr, method="risk_parity")

    def test_invalid_max_weight_raises(self):
        kf = _make_kf_results(n_pairs=2)
        corr = self._get_corr(kf)
        with pytest.raises(ValueError, match="max_weight"):
            suggest_position_weights(kf, corr, max_weight=0.0)

    def test_empty_kf_returns_empty_df(self):
        w = suggest_position_weights({}, pd.DataFrame())
        assert w.empty
