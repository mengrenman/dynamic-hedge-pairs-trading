# tests/test_portfolio.py
"""Tests for pairs.stats.portfolio — pair correlation & diversification analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pairs.stats.portfolio as portfolio_module
from pairs.stats.portfolio import (
    ERCConvergenceError,
    allocation_weights,
    pair_return_correlations,
    portfolio_diversification_score,
    spread_returns,
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


# ---------------------------------------------------------------------------
# TestSpreadReturns
# ---------------------------------------------------------------------------

class TestSpreadReturns:

    def test_hand_computed_example(self):
        idx = pd.bdate_range("2021-03-01", periods=3)
        P1 = pd.Series([100.0, 102.0, 101.0], index=idx)
        P2 = pd.Series([50.0, 49.0, 50.0], index=idx)
        r = spread_returns(P1, P2, 2.0)
        # (dP1 - 2 dP2) / (P1 + 2 P2) at the previous close
        assert r.iloc[0] == pytest.approx((2.0 - 2.0 * -1.0) / (100.0 + 2.0 * 50.0))      # 0.02
        assert r.iloc[1] == pytest.approx((-1.0 - 2.0 * 1.0) / (102.0 + 2.0 * 49.0))      # -0.015
        assert list(r.index) == list(idx[1:])                 # first row dropped, dated by the later close

    def test_negative_beta_uses_the_absolute_hedge_in_the_denominator(self):
        idx = pd.bdate_range("2021-03-01", periods=3)
        P1 = pd.Series([100.0, 102.0, 101.0], index=idx)
        P2 = pd.Series([50.0, 49.0, 50.0], index=idx)
        r = spread_returns(P1, P2, -1.0)
        assert r.iloc[0] == pytest.approx((2.0 + -1.0) / (100.0 + 50.0))
        assert r.iloc[1] == pytest.approx((-1.0 + 1.0) / (102.0 + 49.0))

    def test_sessions_where_a_leg_has_no_close_are_dropped_before_differencing(self):
        idx = pd.bdate_range("2021-03-01", periods=4)
        P1 = pd.Series([100.0, 102.0, 101.0, 103.0], index=idx)
        P2 = pd.Series([50.0, np.nan, 50.0, 52.0], index=idx)
        r = spread_returns(P1, P2, 1.0)
        assert list(r.index) == [idx[2], idx[3]]
        assert r.iloc[0] == pytest.approx((1.0 - 0.0) / (100.0 + 50.0))    # spans the gap: 100 -> 101, 50 -> 50

    def test_matches_a_direct_pandas_construction_on_random_walks(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2020-01-01", periods=300)
        P1 = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
        P2 = pd.Series(40 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
        beta = 1.7
        want = (P1.diff() - beta * P2.diff()).iloc[1:] / (P1.shift(1) + beta * P2.shift(1)).iloc[1:]
        np.testing.assert_allclose(spread_returns(P1, P2, beta).to_numpy(), want.to_numpy(), rtol=1e-12)

    @pytest.mark.parametrize("beta", [np.nan, np.inf])
    def test_non_finite_beta_raises(self, beta):
        idx = pd.bdate_range("2021-03-01", periods=3)
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="beta"):
            spread_returns(s, s, beta)

    def test_too_few_sessions_or_non_positive_prices_raise(self):
        idx = pd.bdate_range("2021-03-01", periods=3)
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="two sessions"):
            spread_returns(s.iloc[:1], s.iloc[:1], 1.0)
        with pytest.raises(ValueError, match="positive"):
            spread_returns(s, pd.Series([1.0, 0.0, 3.0], index=idx), 1.0)


# ---------------------------------------------------------------------------
# TestAllocationWeights
# ---------------------------------------------------------------------------

_SIGMA3 = np.array([0.20, 0.10, 0.15])
_COV3 = np.array([[0.0400, 0.0060, 0.0000],
                  [0.0060, 0.0100, 0.0020],
                  [0.0000, 0.0020, 0.0225]])


def _risk_contributions(cov, w):
    rc = w * (cov @ w)
    return rc / rc.sum()


class TestAllocationWeights:

    def _inputs(self):
        return dict(sigma=_SIGMA3, cov=_COV3, pairs=[("A", "B"), ("A", "C"), ("D", "E")])

    @pytest.mark.parametrize("method", ["equal", "inv_vol", "inv_var", "erc", "leg_split"])
    def test_weights_are_a_probability_vector(self, method):
        w = allocation_weights(method, **self._inputs())
        assert w.shape == (3,)
        assert (w >= 0).all() and abs(w.sum() - 1.0) < 1e-12

    @pytest.mark.parametrize("method", ["equal", "inv_vol", "inv_var", "erc", "leg_split"])
    def test_order_of_the_input_is_the_order_of_the_output(self, method):
        inp = self._inputs()
        w = allocation_weights(method, **inp)
        perm = np.array([2, 0, 1])
        w_perm = allocation_weights(method, sigma=inp["sigma"][perm], cov=inp["cov"][np.ix_(perm, perm)],
                                    pairs=[inp["pairs"][i] for i in perm])
        np.testing.assert_allclose(w_perm, w[perm], atol=1e-9)

    @pytest.mark.parametrize("method", ["equal", "inv_vol", "inv_var", "erc", "leg_split"])
    def test_a_single_pair_gets_everything(self, method):
        w = allocation_weights(method, sigma=[0.1], cov=[[0.01]], pairs=[("A", "B")])
        np.testing.assert_allclose(w, [1.0])

    def test_equal(self):
        np.testing.assert_allclose(allocation_weights("equal", sigma=[1.0, 2.0, 3.0, 4.0]), 0.25)
        assert allocation_weights("equal", pairs=[("A", "B"), ("C", "D")]).tolist() == [0.5, 0.5]
        assert allocation_weights("equal", cov=np.eye(5)).shape == (5,)

    def test_inverse_volatility_and_inverse_variance(self):
        s = np.array([0.1, 0.2, 0.4])
        np.testing.assert_allclose(allocation_weights("inv_vol", sigma=s), (1 / s) / (1 / s).sum())
        np.testing.assert_allclose(allocation_weights("inv_var", sigma=s), (1 / s ** 2) / (1 / s ** 2).sum())
        # halving volatility doubles the weight under inv_vol, quadruples it under inv_var
        assert allocation_weights("inv_vol", sigma=s)[0] / allocation_weights("inv_vol", sigma=s)[1] == pytest.approx(2.0)
        assert allocation_weights("inv_var", sigma=s)[0] / allocation_weights("inv_var", sigma=s)[1] == pytest.approx(4.0)

    def test_inputs_may_be_pandas_objects(self):
        w = allocation_weights("inv_vol", sigma=pd.Series([0.1, 0.2], index=["x", "y"]))
        assert isinstance(w, np.ndarray) and w.tolist() == pytest.approx([2 / 3, 1 / 3])
        cov = pd.DataFrame(_COV3, index=list("abc"), columns=list("abc"))
        np.testing.assert_allclose(allocation_weights("erc", cov=cov), allocation_weights("erc", cov=_COV3))

    # ---- equal risk contribution ----

    def test_erc_equalizes_risk_contributions_on_a_known_matrix(self):
        w = allocation_weights("erc", cov=_COV3)
        np.testing.assert_allclose(_risk_contributions(_COV3, w), 1 / 3, atol=1e-6)
        assert abs(w.sum() - 1.0) < 1e-12
        # correlated pairs are not simply inverse-volatility weighted
        iv = allocation_weights("inv_vol", sigma=np.sqrt(np.diag(_COV3)))
        assert np.abs(w - iv).max() > 1e-3

    def test_erc_on_a_larger_matrix(self):
        rng = np.random.default_rng(11)
        A = rng.standard_normal((20, 60))
        C = np.corrcoef(A)
        d = np.exp(rng.uniform(np.log(0.002), np.log(0.05), 20))
        S = np.diag(d) @ C @ np.diag(d)
        w = allocation_weights("erc", cov=S, sigma=d)
        np.testing.assert_allclose(_risk_contributions(S, w), 1 / 20, atol=1e-6)

    def test_erc_converges_on_many_shrunk_twenty_pair_covariances(self):
        # the shape of the study's inputs: 20 pairs, strongly correlated, volatilities over a decade
        rng = np.random.default_rng(2)
        for _ in range(40):
            n = int(rng.integers(2, 21))
            F = rng.standard_normal((n, 3)) @ rng.standard_normal((3, 600))
            C = np.corrcoef(F + rng.standard_normal((n, 600)))
            C = 0.9 * C + 0.1 * np.eye(n)
            d = np.exp(rng.uniform(np.log(0.002), np.log(0.03), n))
            S = np.diag(d) @ C @ np.diag(d)
            w = allocation_weights("erc", cov=S, sigma=d)
            np.testing.assert_allclose(_risk_contributions(S, w), 1 / n, atol=1e-5 / n)

    def test_erc_reduces_to_inverse_volatility_on_a_diagonal_matrix(self):
        s = np.array([0.05, 0.10, 0.02, 0.30])
        w = allocation_weights("erc", cov=np.diag(s ** 2))
        np.testing.assert_allclose(w, allocation_weights("inv_vol", sigma=s), atol=1e-9)

    def test_erc_equals_inverse_volatility_under_constant_correlation(self):
        s = np.array([0.05, 0.10, 0.02, 0.30])
        C = np.full((4, 4), 0.6) + 0.4 * np.eye(4)
        S = np.diag(s) @ C @ np.diag(s)
        np.testing.assert_allclose(allocation_weights("erc", cov=S), allocation_weights("inv_vol", sigma=s), atol=1e-6)

    def test_erc_is_scale_free_and_starts_from_sigma_without_changing_the_answer(self):
        w = allocation_weights("erc", cov=_COV3)
        np.testing.assert_allclose(allocation_weights("erc", cov=1e-4 * _COV3), w, atol=1e-6)
        np.testing.assert_allclose(allocation_weights("erc", cov=_COV3, sigma=[1.0, 2.0, 3.0]), w, atol=1e-6)

    def test_erc_raises_a_clear_error_when_the_solve_does_not_converge(self, monkeypatch):
        monkeypatch.setattr(portfolio_module, "_ERC_MAXITER", 1)
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 30))
        S = np.cov(A)
        with pytest.raises(ERCConvergenceError, match="did not converge"):
            allocation_weights("erc", cov=S)
        assert issubclass(ERCConvergenceError, RuntimeError)

    # ---- shared legs ----

    def test_leg_split_on_a_star_is_equal(self):
        star = [("HUB", x) for x in "ABCDE"]
        np.testing.assert_allclose(allocation_weights("leg_split", pairs=star), 0.2)
        # the hub can sit on either side of the pair
        mixed = [("HUB", "A"), ("B", "HUB"), ("HUB", "C"), ("D", "HUB")]
        np.testing.assert_allclose(allocation_weights("leg_split", pairs=mixed), 0.25)

    def test_leg_split_on_disjoint_pairs_is_equal(self):
        w = allocation_weights("leg_split", pairs=[("A", "B"), ("C", "D"), ("E", "F")])
        np.testing.assert_allclose(w, 1 / 3)

    def test_leg_split_gives_the_separate_pair_half_when_three_pairs_share_a_hub(self):
        w = allocation_weights("leg_split", pairs=[("H", "A"), ("H", "B"), ("H", "C"), ("D", "E")])
        np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2])

    def test_leg_split_uses_the_more_shared_of_the_two_legs(self):
        # X is in three pairs, Y in one: the (X, Y) pair is weighted by X's count
        w = allocation_weights("leg_split", pairs=[("X", "Y"), ("X", "A"), ("X", "B"), ("C", "D")])
        np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2])

    # ---- validation ----

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            allocation_weights("min_var", sigma=[0.1, 0.2])

    def test_missing_inputs_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            allocation_weights("equal")
        with pytest.raises(ValueError, match="sigma"):
            allocation_weights("inv_vol", pairs=[("A", "B")])
        with pytest.raises(ValueError, match="sigma"):
            allocation_weights("inv_var", cov=np.eye(2))
        with pytest.raises(ValueError, match="cov"):
            allocation_weights("erc", sigma=[0.1, 0.2])
        with pytest.raises(ValueError, match="pairs"):
            allocation_weights("leg_split", sigma=[0.1, 0.2])

    @pytest.mark.parametrize("sigma", [[0.1, 0.0], [0.1, -0.2], [0.1, np.nan], [0.1, np.inf], [], [[0.1, 0.2]]])
    def test_bad_sigma_raises(self, sigma):
        with pytest.raises(ValueError, match="sigma"):
            allocation_weights("inv_vol", sigma=sigma)

    def test_bad_cov_raises(self):
        with pytest.raises(ValueError, match="square"):
            allocation_weights("erc", cov=np.ones((2, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            allocation_weights("erc", cov=np.array([[1.0, 0.5], [0.1, 1.0]]))
        with pytest.raises(ValueError, match="positive definite"):
            allocation_weights("erc", cov=np.array([[1.0, 2.0], [2.0, 1.0]]))
        with pytest.raises(ValueError, match="finite"):
            allocation_weights("erc", cov=np.array([[1.0, np.nan], [np.nan, 1.0]]))
        with pytest.raises(ValueError, match="positive definite"):
            allocation_weights("erc", cov=[[0.0]])

    def test_bad_pairs_raise(self):
        with pytest.raises(ValueError, match="pairs"):
            allocation_weights("leg_split", pairs=[])
        with pytest.raises(ValueError, match="pairs"):
            allocation_weights("leg_split", pairs=[("A", "A")])
        with pytest.raises(ValueError, match="pairs"):
            allocation_weights("leg_split", pairs=[("A", "B", "C")])

    def test_lengths_must_agree(self):
        with pytest.raises(ValueError, match="same length"):
            allocation_weights("inv_vol", sigma=[0.1, 0.2], pairs=[("A", "B")])
