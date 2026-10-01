# tests/test_portfolio.py
"""Tests for pairs.stats.portfolio — pair correlation, diversification and capital-allocation analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

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


def _make_prices(kf_results: dict, seed: int = 1, vol: float = 0.01) -> pd.DataFrame:
    """Wide closes, one positive random-walk column per ticker of kf_results, on the frames' dates."""
    rng = np.random.default_rng(seed)
    idx = next(iter(kf_results.values())).index
    tickers = sorted({t for key in kf_results for t in key})
    return pd.DataFrame(
        {t: 50.0 * np.exp(np.cumsum(rng.normal(0.0, vol, len(idx)))) for t in tickers}, index=idx
    )


def _alternating_pairs(vols, levels=None, n: int = 41):
    """
    kf_results and prices whose per-dollar spread return alternates +vol, -vol exactly.

    Leg one grows by (1 + vol) and shrinks by (1 - vol) in turn; leg two is flat and the hedge ratio is
    zero, so the spread return per dollar is +-vol and its sample variance is vol**2 * m / (m - 1) for
    every pair (m = n - 1 returns, even). Weights by inverse variance are then exactly proportional to
    1 / vol**2, which lets a test choose the uncapped weights it wants. ``levels`` are the starting prices,
    which set the pair's price-unit variance but not its per-dollar one. The frame's 'resid' is leg one's
    price (alpha and beta are zero), in price units.
    """
    assert (n - 1) % 2 == 0
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    levels = [100.0] * len(vols) if levels is None else list(levels)
    kf, cols = {}, {}
    for k, (v, level) in enumerate(zip(vols, levels)):
        r = np.where(np.arange(1, n) % 2 == 1, v, -v)
        p1 = level * np.concatenate([[1.0], np.cumprod(1.0 + r)])
        cols[f"X{k}"] = p1
        cols[f"Y{k}"] = np.full(n, level / 2.0)
        kf[(f"X{k}", f"Y{k}")] = pd.DataFrame(
            {"alpha": 0.0, "beta": 0.0, "y_hat": 0.0, "resid": p1}, index=idx
        )
    return kf, pd.DataFrame(cols, index=idx)


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

# Expected values of the one-pass tests below, taken from the helper as shipped before 2026-10-01
# (`git show 96edc19:pairs/stats/portfolio.py`) on `_det_price_unit_frames([1.0, 1.2, 8.0, 10.0])`.
_PREFIX_RESID_VAR = [0.399517931064487, 1.80451097016098, 122.977914761211, 196.871927838594]
_PREFIX_INV_VAR_WEIGHT = [0.815210136200441, 0.180487163770771, 0.00264837038121845, 0.00165432964756975]
_PREFIX_WEIGHT_CAP_040 = [0.684006383765062, 0.308635930517145, 0.00452875561831932, 0.00282893009947379]
_PREFIX_WEIGHT_CAP_030 = [0.618824819580051, 0.372299788523209, 0.00546292441112885, 0.00341246748561094]


def _det_price_unit_frames(scales, n: int = 60) -> dict:
    """Deterministic frames (no random numbers) for the pre-fix comparison; 'resid' is in price units."""
    t = np.arange(n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    out = {}
    for j, s in enumerate(scales):
        resid = s * (np.sin(0.9 * (j + 1) * t) + 0.5 * np.cos(0.37 * t + j))
        out[(f"A{j}", f"B{j}")] = pd.DataFrame({"resid": resid, "beta": 1.0}, index=idx)
    return out


def _random_pairs(n_pairs: int = 3, n: int = 300, seed: int = 11):
    """Frames with a constant hedge ratio and the prices behind them: resid = P1 - beta * P2 in price units."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    kf, cols = {}, {}
    for k in range(n_pairs):
        p1 = 80.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
        p2 = 30.0 * np.exp(np.cumsum(rng.normal(0.0, 0.010, n)))
        beta = 0.6 + 0.3 * k
        cols[f"X{k}"], cols[f"Y{k}"] = p1, p2
        kf[(f"X{k}", f"Y{k}")] = pd.DataFrame(
            {"alpha": 0.0, "beta": beta, "y_hat": beta * p2, "resid": p1 - beta * p2}, index=idx
        )
    return kf, pd.DataFrame(cols, index=idx)


class TestSuggestPositionWeights:

    def _get_corr(self, kf):
        return pair_return_correlations(kf, min_overlap=2)

    def test_weights_sum_to_one(self):
        kf = _make_kf_results(n_pairs=3)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, prices=_make_prices(kf))
        assert abs(w["weight"].sum() - 1.0) < 1e-10

    def test_max_weight_clipping_enforced(self):
        kf = _make_kf_results(n_pairs=4)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, max_weight=0.30, prices=_make_prices(kf))
        assert w["weight"].max() <= 0.30 + 1e-10

    def test_inv_var_gives_lower_weight_to_high_var_pair(self):
        """One pair with very large residual variance should get lower weight.

        This is the price-unit property (the weight follows the variance of Δresid), so it is asked of
        risk='price_units'; the per-dollar default weighs by the spread return and is tested below.
        max_weight=1.0 because two pairs cannot sit under the default 0.40 cap (the effective cap is
        1/n = 0.5, so the default would give 0.5 and 0.5), which is not what this test is about.
        """
        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        low_var = pd.DataFrame({"resid": rng.standard_normal(300) * 0.01, "beta": np.ones(300)}, index=dates)
        high_var = pd.DataFrame({"resid": rng.standard_normal(300) * 10.0, "beta": np.ones(300)}, index=dates)
        kf = {("A", "B"): low_var, ("C", "D"): high_var}
        corr = pair_return_correlations(kf, min_overlap=2)
        w = suggest_position_weights(kf, corr, method="inv_var", risk="price_units", max_weight=1.0)
        w = w.set_index("pair")
        assert w.loc["A/B", "weight"] > w.loc["C/D", "weight"]

    def test_equal_method_gives_equal_weights(self):
        kf = _make_kf_results(n_pairs=4)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, method="equal", prices=_make_prices(kf))
        weights = w["weight"].values
        np.testing.assert_allclose(weights, weights[0], atol=1e-10)

    def test_required_columns_present(self):
        kf = _make_kf_results(n_pairs=3)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, prices=_make_prices(kf))
        assert list(w.columns) == ["pair", "resid_var", "var_per_dollar", "inv_var_weight", "weight",
                                   "suggested_capital_pct"]

    def test_single_pair_weight_is_one(self):
        kf = _make_kf_results(n_pairs=1)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, prices=_make_prices(kf))
        assert abs(w["weight"].iloc[0] - 1.0) < 1e-10

    def test_capital_pct_equals_weight_times_100(self):
        kf = _make_kf_results(n_pairs=3)
        corr = self._get_corr(kf)
        w = suggest_position_weights(kf, corr, prices=_make_prices(kf))
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
        assert list(w.columns) == ["pair", "resid_var", "var_per_dollar", "inv_var_weight", "weight",
                                   "suggested_capital_pct"]

    # ---- the new options: validation ----

    def test_invalid_risk_and_cap_raise(self):
        kf = _make_kf_results(n_pairs=2)
        with pytest.raises(ValueError, match="risk"):
            suggest_position_weights(kf, risk="variance")
        with pytest.raises(ValueError, match="cap"):
            suggest_position_weights(kf, cap="hard")

    def test_corr_matrix_is_optional_and_never_used(self):
        kf = _make_kf_results(n_pairs=3)
        prices = _make_prices(kf)
        none = suggest_position_weights(kf, prices=prices)
        given = suggest_position_weights(kf, pair_return_correlations(kf, min_overlap=2), prices=prices)
        junk = suggest_position_weights(kf, "not a matrix", prices=prices)
        pd.testing.assert_frame_equal(none, given)
        pd.testing.assert_frame_equal(none, junk)

    # ---- risk per dollar ----

    def test_per_dollar_weights_do_not_move_when_a_pairs_price_level_is_multiplied_by_ten(self):
        kf, prices = _random_pairs(n_pairs=3)
        before = suggest_position_weights(kf, prices=prices, max_weight=1.0).set_index("pair")
        before_pu = suggest_position_weights(kf, risk="price_units", max_weight=1.0).set_index("pair")

        # Pair X1/Y1 at ten times the price level: both legs scale, so the hedge ratio, the percentage
        # moves and the per-dollar risk stay put while the residual, in price units, scales by ten.
        kf10 = {k: v.copy() for k, v in kf.items()}
        prices10 = prices.copy()
        prices10[["X1", "Y1"]] *= 10.0
        kf10[("X1", "Y1")][["y_hat", "resid"]] *= 10.0
        after = suggest_position_weights(kf10, prices=prices10, max_weight=1.0).set_index("pair")
        after_pu = suggest_position_weights(kf10, risk="price_units", max_weight=1.0).set_index("pair")

        np.testing.assert_allclose(after["var_per_dollar"], before["var_per_dollar"].reindex(after.index), rtol=1e-10)
        np.testing.assert_allclose(after["weight"], before["weight"].reindex(after.index), rtol=1e-10)
        # the price-unit weights do move: the repriced pair is down-weighted a hundredfold in variance terms
        assert after_pu.loc["X1/Y1", "weight"] < 0.1 * before_pu.loc["X1/Y1", "weight"]
        assert after_pu.loc["X1/Y1", "resid_var"] == pytest.approx(100.0 * before_pu.loc["X1/Y1", "resid_var"])

    def test_the_hedge_ratio_is_lagged_one_row(self):
        idx = pd.bdate_range("2021-03-01", periods=4)
        prices = pd.DataFrame({"A": [100.0, 102.0, 101.0, 104.0], "B": [50.0, 49.0, 50.0, 52.0]}, index=idx)
        frame = pd.DataFrame({"beta": [1.0, 1.5, 2.0, 0.5], "resid": [0.0, 1.0, 0.0, 2.0]}, index=idx)
        # day t is priced with the hedge held at the close of day t-1: 1.0, then 1.5, then 2.0
        r = [(2.0 - 1.0 * -1.0) / (100.0 + 1.0 * 50.0),
             (-1.0 - 1.5 * 1.0) / (102.0 + 1.5 * 49.0),
             (3.0 - 2.0 * 2.0) / (101.0 + 2.0 * 50.0)]
        got = suggest_position_weights({("A", "B"): frame}, prices=prices)["var_per_dollar"].iloc[0]
        assert got == pytest.approx(np.var(r, ddof=1), rel=1e-12)
        # the same-row hedge would give a different number, so the lag is what the test sees
        same_row = [(2.0 - 1.5 * -1.0) / (100.0 + 1.5 * 50.0),
                    (-1.0 - 2.0 * 1.0) / (102.0 + 2.0 * 49.0),
                    (3.0 - 0.5 * 2.0) / (101.0 + 0.5 * 50.0)]
        assert abs(np.var(same_row, ddof=1) - got) > 1e-6

    def test_a_hedge_ratio_that_changes_on_the_last_row_does_not_change_the_risk(self):
        kf, prices = _random_pairs(n_pairs=1)
        frame = kf[("X0", "Y0")]
        moved = frame.copy()
        moved.iloc[-1, moved.columns.get_loc("beta")] = 9.0       # re-estimated on the last day: prices no return
        a = suggest_position_weights({("X0", "Y0"): frame}, prices=prices)["var_per_dollar"].iloc[0]
        b = suggest_position_weights({("X0", "Y0"): moved}, prices=prices)["var_per_dollar"].iloc[0]
        assert a == b
        # and a change on an earlier row does move it
        earlier = frame.copy()
        earlier.iloc[100, earlier.columns.get_loc("beta")] = 9.0
        c = suggest_position_weights({("X0", "Y0"): earlier}, prices=prices)["var_per_dollar"].iloc[0]
        assert abs(c - a) > 1e-9

    def test_a_constant_hedge_matches_spread_returns_including_across_a_missing_close(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2020-01-01", periods=120)
        P1 = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=idx)
        P2 = pd.Series(40 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))), index=idx)
        P2.iloc[[10, 11, 57]] = np.nan                              # sessions on which one leg does not price
        beta = 1.7
        # the frame covers the last 100 sessions only; the prices have 20 more rows before it
        frame = pd.DataFrame({"beta": beta, "resid": 0.0}, index=idx[20:])
        got = suggest_position_weights({("A", "B"): frame},
                                       prices=pd.DataFrame({"A": P1, "B": P2}))["var_per_dollar"].iloc[0]
        want = spread_returns(P1.iloc[20:], P2.iloc[20:], beta).var(ddof=1)
        assert got == pytest.approx(want, rel=1e-12)

    def test_pairs_without_a_usable_risk_get_zero_weight(self):
        kf, prices = _random_pairs(n_pairs=3)
        kf[("X2", "Y2")] = kf[("X2", "Y2")].iloc[:2]                # two rows: one return, no variance
        w = suggest_position_weights(kf, prices=prices).set_index("pair")
        assert np.isnan(w.loc["X2/Y2", "var_per_dollar"]) and w.loc["X2/Y2", "weight"] == 0.0
        assert w["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        # with every pair unusable the weights are all zero, not an error
        none = suggest_position_weights({k: v.iloc[:2] for k, v in kf.items()}, prices=prices)
        assert (none["weight"] == 0.0).all()

    def test_per_dollar_needs_prices_and_says_how_to_proceed(self):
        kf, prices = _random_pairs(n_pairs=2)
        with pytest.raises(ValueError, match=r"prices=.*price_units"):
            suggest_position_weights(kf)
        with pytest.raises(ValueError, match=r"prices=.*price_units"):
            suggest_position_weights(kf, method="equal")           # which pairs are usable is decided by the risk
        with pytest.raises(ValueError, match=r"no column for \['Y1'\].*price_units"):
            suggest_position_weights(kf, prices=prices.drop(columns="Y1"))
        with pytest.raises(ValueError, match="DataFrame"):
            suggest_position_weights(kf, prices=prices["X0"])
        # price units need none
        assert len(suggest_position_weights(kf, risk="price_units")) == 2

    def test_a_frame_without_beta_or_a_non_positive_close_raises_under_per_dollar(self):
        kf, prices = _random_pairs(n_pairs=2)
        with pytest.raises(ValueError, match="beta"):
            suggest_position_weights({k: v.drop(columns="beta") for k, v in kf.items()}, prices=prices)
        bad = prices.copy()
        bad.iloc[5, bad.columns.get_loc("X0")] = 0.0
        with pytest.raises(ValueError, match="positive"):
            suggest_position_weights(kf, prices=bad)

    def test_prices_that_cannot_be_lined_up_with_a_frame_raise_naming_the_pair(self):
        kf, prices = _random_pairs(n_pairs=2)
        ok = suggest_position_weights(kf, prices=prices)
        assert ok["var_per_dollar"].notna().all()
        misaligned = {
            "a timezone-aware index against naive frames": prices.tz_localize("UTC"),
            "a calendar shifted by more than the span": prices.set_axis(prices.index + pd.Timedelta(days=500)),
            "a RangeIndex": prices.reset_index(drop=True),
            "two shared dates of 300": prices.iloc[:2],
        }
        for what, bad in misaligned.items():
            with pytest.raises(ValueError, match=r"X0/Y0.*indexed like the frames"):
                suggest_position_weights(kf, prices=bad)
        # one misaligned pair among aligned ones is named, and is not answered with a zero weight
        kf_late = {**kf, ("X1", "Y1"): kf[("X1", "Y1")].set_axis(kf[("X1", "Y1")].index + pd.Timedelta(days=500))}
        with pytest.raises(ValueError, match=r"X1/Y1"):
            suggest_position_weights(kf_late, prices=prices)
        # the price-unit risk reads no prices, so it does not look at them
        assert len(suggest_position_weights(kf, risk="price_units", prices=prices.tz_localize("UTC"))) == 2

    def test_missing_closes_and_tiny_frames_are_an_unusable_pair_not_a_misaligned_index(self):
        kf, prices = _random_pairs(n_pairs=3)
        # exactly three shared dates is enough to line up: two returns, a variance
        three = suggest_position_weights({("X0", "Y0"): kf[("X0", "Y0")]}, prices=prices.iloc[:3])
        assert np.isfinite(three["var_per_dollar"].iloc[0]) and three["weight"].iloc[0] == 1.0
        # the indexes share every date but one leg has no close on all but two of them: NaN, weight 0, no error
        gappy = prices.copy()
        gappy.iloc[2:, gappy.columns.get_loc("Y2")] = np.nan
        w = suggest_position_weights(kf, prices=gappy).set_index("pair")
        assert np.isnan(w.loc["X2/Y2", "var_per_dollar"]) and w.loc["X2/Y2", "weight"] == 0.0
        assert w["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        # a frame with fewer than three rows cannot show a misalignment it cannot fill: it is unusable, not an error
        for rows in (0, 1, 2):
            tiny = suggest_position_weights({("X0", "Y0"): kf[("X0", "Y0")].iloc[:rows]}, prices=prices)
            assert np.isnan(tiny["var_per_dollar"].iloc[0]) and tiny["weight"].iloc[0] == 0.0

    def test_per_dollar_inverse_variance_follows_the_spread_return(self):
        # three pairs with spread returns of 1%, 2% and 4% a day, whatever their price levels
        kf, prices = _alternating_pairs([0.01, 0.02, 0.04], levels=[500.0, 20.0, 3000.0])
        w = suggest_position_weights(kf, prices=prices, max_weight=1.0).set_index("pair")
        m = 40
        np.testing.assert_allclose(w["var_per_dollar"].sort_index().to_numpy(),
                                   np.array([0.01, 0.02, 0.04]) ** 2 * m / (m - 1), rtol=1e-9)
        base = 1.0 / np.array([1.0, 4.0, 16.0])
        np.testing.assert_allclose(w["weight"].sort_index().to_numpy(), base / base.sum(), rtol=1e-9)
        # the price-unit variance, by contrast, is driven by the price level
        assert w["resid_var"].idxmax() == "X2/Y2" and w["resid_var"].idxmin() == "X1/Y1"

    # ---- inverse volatility ----

    def test_inv_vol_equals_inv_var_on_equal_variance_inputs_and_differs_otherwise(self):
        kf, prices = _alternating_pairs([0.02, 0.02, 0.02, 0.02])
        iv = suggest_position_weights(kf, prices=prices, method="inv_var")
        vol = suggest_position_weights(kf, prices=prices, method="inv_vol")
        np.testing.assert_allclose(iv["weight"], vol["weight"], atol=1e-12)
        np.testing.assert_allclose(vol["weight"], 0.25, atol=1e-12)

        kf, prices = _alternating_pairs([0.01, 0.02])
        var_w = suggest_position_weights(kf, prices=prices, method="inv_var", max_weight=1.0).set_index("pair")["weight"]
        vol_w = suggest_position_weights(kf, prices=prices, method="inv_vol", max_weight=1.0).set_index("pair")["weight"]
        assert var_w["X0/Y0"] == pytest.approx(0.8) and vol_w["X0/Y0"] == pytest.approx(2.0 / 3.0)   # 4:1 against 2:1
        assert abs(var_w["X0/Y0"] - vol_w["X0/Y0"]) > 0.1

    def test_inv_vol_under_price_units_uses_the_standard_deviation_of_the_residual_difference(self):
        kf = _det_price_unit_frames([1.0, 1.2, 8.0, 10.0])
        w = suggest_position_weights(kf, method="inv_vol", risk="price_units", max_weight=1.0)
        sd = np.sqrt(np.array(_PREFIX_RESID_VAR))
        base = (1.0 / sd) / (1.0 / sd).sum()
        np.testing.assert_allclose(w.set_index("pair").sort_index()["inv_var_weight"], base, rtol=1e-9)

    # ---- the cap ----

    def test_iterative_cap_is_a_cap(self):
        # uncapped weights 0.50, 0.35, 0.10, 0.05: one clip-and-renormalize is not enough, two rounds are
        base = np.array([0.50, 0.35, 0.10, 0.05])
        kf, prices = _alternating_pairs(0.02 / np.sqrt(base))
        w = suggest_position_weights(kf, prices=prices, max_weight=0.40)
        np.testing.assert_allclose(w.set_index("pair").sort_index()["inv_var_weight"], base, rtol=1e-9)
        assert w["weight"].max() == 0.40                              # exactly the cap
        assert abs(w["weight"].sum() - 1.0) < 1e-12
        np.testing.assert_allclose(w.sort_values("pair")["weight"].to_numpy(),
                                   [0.40, 0.40, 0.20 * 0.10 / 0.15, 0.20 * 0.05 / 0.15], rtol=1e-9)
        assert (w["weight"] <= 0.40 + 1e-12).all()

    def test_the_weights_under_the_cap_keep_their_ratios(self):
        base = np.array([0.50, 0.35, 0.10, 0.04, 0.01])
        kf, prices = _alternating_pairs(0.02 / np.sqrt(base))
        w = suggest_position_weights(kf, prices=prices, max_weight=0.40).set_index("pair").sort_index()
        free = w["weight"] < 0.40 - 1e-9
        assert free.sum() == 3
        ratios = (w.loc[free, "weight"] / w.loc[free, "inv_var_weight"]).to_numpy()
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
        assert ratios[0] > 1.0                                         # they took up what the capped pairs gave up

    def test_one_pass_leaves_the_cap_broken_where_the_iterative_cap_holds(self):
        base = np.array([0.50, 0.35, 0.10, 0.05])
        kf, prices = _alternating_pairs(0.02 / np.sqrt(base))
        one = suggest_position_weights(kf, prices=prices, max_weight=0.40, cap="one_pass")
        assert one["weight"].max() == pytest.approx(0.40 / 0.90, rel=1e-9)         # 0.444: clipped, then renormalized
        assert one["weight"].max() > 0.40 + 0.04
        it = suggest_position_weights(kf, prices=prices, max_weight=0.40, cap="iterative")
        assert it["weight"].max() == 0.40

    def test_a_cap_that_cannot_hold_gives_equal_weights(self):
        kf, prices = _alternating_pairs([0.01, 0.05])               # very unequal: weights 25:1 before the cap
        w = suggest_position_weights(kf, prices=prices)             # n=2, cap 0.40: 2 * 0.40 < 1
        assert w["weight"].tolist() == pytest.approx([0.5, 0.5], abs=1e-12)
        kf, prices = _alternating_pairs([0.01, 0.02, 0.05])
        w = suggest_position_weights(kf, prices=prices, max_weight=0.20)
        np.testing.assert_allclose(w["weight"], 1.0 / 3.0, atol=1e-12)
        # exactly n * cap = 1 is feasible and is also the equal split
        kf, prices = _alternating_pairs([0.01, 0.02, 0.05, 0.08])
        w = suggest_position_weights(kf, prices=prices, max_weight=0.25)
        np.testing.assert_allclose(w["weight"], 0.25, atol=1e-12)

    def test_a_cap_at_one_changes_nothing(self):
        kf, prices = _alternating_pairs([0.01, 0.02, 0.05])
        w = suggest_position_weights(kf, prices=prices, max_weight=1.0)
        np.testing.assert_allclose(w["weight"], w["inv_var_weight"], atol=0.0)

    def test_the_cap_is_met_on_many_random_weight_vectors(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 15))
            base = rng.dirichlet(np.full(n, rng.uniform(0.1, 2.0)))
            base[rng.random(n) < 0.15] = 0.0
            if base.sum() == 0.0:
                continue
            base /= base.sum()
            cap = float(rng.uniform(0.05, 1.0))
            out = portfolio_module._capped_weights(base, cap)
            n_pos = int((base > 0).sum())
            eff = max(cap, 1.0 / n_pos)
            assert abs(out.sum() - 1.0) < 1e-12
            assert out.max() <= eff + 1e-12
            assert (out[base == 0.0] == 0.0).all()
            # water-filling is w_i = min(cap, lam * base_i) for the lam that makes the weights sum to one;
            # when n_pos * eff = 1 every weight sits at the cap and any large lam does
            if n_pos * eff <= 1.0 + 1e-12:
                want = np.where(base > 0.0, eff, 0.0)
            else:
                hi = 2.0 * eff / base[base > 0].min()                  # every positive weight is capped at hi
                lam = brentq(lambda x: np.minimum(eff, x * base).sum() - 1.0, 0.0, hi, xtol=1e-14)
                want = np.minimum(eff, lam * base)
            np.testing.assert_allclose(out, want, atol=1e-9)

    def test_zero_weight_pairs_stay_at_zero_and_do_not_count_toward_the_cap(self):
        kf, prices = _random_pairs(n_pairs=3)
        kf[("X2", "Y2")] = kf[("X2", "Y2")].iloc[:2]
        w = suggest_position_weights(kf, prices=prices).set_index("pair")   # two usable pairs, cap 0.40 cannot hold
        assert w.loc["X2/Y2", "weight"] == 0.0
        np.testing.assert_allclose(w.loc[["X0/Y0", "X1/Y1"], "weight"], 0.5, atol=1e-12)

    @pytest.mark.parametrize("method", ["inv_var", "inv_vol", "equal"])
    @pytest.mark.parametrize("risk", ["per_dollar", "price_units"])
    def test_a_pair_with_exactly_zero_variance_gets_weight_zero_and_the_rest_stay_finite(self, risk, method):
        kf, prices = _random_pairs(n_pairs=4)
        idx = prices.index
        rng = np.random.default_rng(5)
        walk = lambda start: pd.Series(start * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx)))), index=idx)
        # R: a residual that never moves (price units) while its legs do (so its risk per dollar is positive)
        prices["R0"], prices["R1"] = walk(60.0), walk(40.0)
        kf[("R0", "R1")] = pd.DataFrame({"alpha": 0.0, "beta": 1.0, "y_hat": 0.0, "resid": 10.0}, index=idx)
        # T: a spread whose return is exactly 100% a day (leg one doubles, leg two is flat, no hedge), so its
        # risk per dollar has exactly zero variance while its residual, in price units, moves
        prices["T0"], prices["T1"] = 100.0 * 2.0 ** np.arange(len(idx)), 50.0
        kf[("T0", "T1")] = pd.DataFrame({"alpha": 0.0, "beta": 0.0, "y_hat": 0.0,
                                         "resid": rng.standard_normal(len(idx)) * 3.0}, index=idx)
        zero, decoy = ("R0/R1", "T0/T1") if risk == "price_units" else ("T0/T1", "R0/R1")

        w = suggest_position_weights(kf, prices=prices, method=method, risk=risk).set_index("pair")
        zero_var = w.loc[zero, "resid_var" if risk == "price_units" else "var_per_dollar"]
        assert zero_var == 0.0                                          # the fixture is exactly zero, not merely small
        assert w.loc[zero, "weight"] == 0.0 and w.loc[zero, "inv_var_weight"] == 0.0
        assert np.isfinite(w[["inv_var_weight", "weight", "suggested_capital_pct"]].to_numpy()).all()
        assert w["weight"].sum() == pytest.approx(1.0, abs=1e-12)
        assert w["weight"].max() <= 0.40 + 1e-12                        # five weighted pairs: the cap can hold
        assert w.loc[decoy, "weight"] > 0.0                             # zero is judged on the chosen risk only
        if method == "equal":
            np.testing.assert_allclose(w.drop(index=zero)["weight"], 0.2, atol=1e-12)

    def test_result_is_sorted_by_weight_and_ties_keep_the_input_order(self):
        kf, prices = _alternating_pairs([0.02, 0.02, 0.01, 0.02])
        w = suggest_position_weights(kf, prices=prices, max_weight=1.0)
        assert w["weight"].is_monotonic_decreasing
        assert w["pair"].tolist() == ["X2/Y2", "X0/Y0", "X1/Y1", "X3/Y3"]
        # many ties: an unstable sort would scramble them (numpy only sorts small arrays stably)
        kf, prices = _alternating_pairs([0.02] * 60)
        w = suggest_position_weights(kf, prices=prices)
        assert w["pair"].tolist() == [f"X{k}/Y{k}" for k in range(60)]

    # ---- the pre-fix behavior, kept for notebook 22 ----

    def test_price_units_and_one_pass_reproduce_the_helper_as_shipped(self):
        kf = _det_price_unit_frames([1.0, 1.2, 8.0, 10.0])
        corr = pair_return_correlations(kf, min_overlap=2)
        w = suggest_position_weights(kf, corr, method="inv_var", max_weight=0.40,
                                     risk="price_units", cap="one_pass")
        assert w["pair"].tolist() == ["A0/B0", "A1/B1", "A2/B2", "A3/B3"]
        np.testing.assert_allclose(w["resid_var"], _PREFIX_RESID_VAR, rtol=1e-9)
        np.testing.assert_allclose(w["inv_var_weight"], _PREFIX_INV_VAR_WEIGHT, rtol=1e-9)
        np.testing.assert_allclose(w["weight"], _PREFIX_WEIGHT_CAP_040, rtol=1e-9)
        np.testing.assert_allclose(w["suggested_capital_pct"], 100.0 * np.array(_PREFIX_WEIGHT_CAP_040), rtol=1e-9)
        assert w["var_per_dollar"].isna().all()
        assert w["weight"].max() > 0.40                                # the defect: the cap is not a cap
        w30 = suggest_position_weights(kf, corr, max_weight=0.30, risk="price_units", cap="one_pass")
        np.testing.assert_allclose(w30["weight"], _PREFIX_WEIGHT_CAP_030, rtol=1e-9)

    def test_price_units_with_the_iterative_cap_is_a_real_cap_on_the_same_weights(self):
        kf = _det_price_unit_frames([1.0, 1.2, 8.0, 10.0])
        w = suggest_position_weights(kf, risk="price_units", cap="iterative")
        np.testing.assert_allclose(w["inv_var_weight"], _PREFIX_INV_VAR_WEIGHT, rtol=1e-9)   # same weights before the cap
        assert w["weight"].max() == 0.40 and abs(w["weight"].sum() - 1.0) < 1e-12
        small = _PREFIX_INV_VAR_WEIGHT[2] + _PREFIX_INV_VAR_WEIGHT[3]          # the two pairs under the cap share 0.20
        np.testing.assert_allclose(w["weight"], [0.4, 0.4, 0.2 * _PREFIX_INV_VAR_WEIGHT[2] / small,
                                                 0.2 * _PREFIX_INV_VAR_WEIGHT[3] / small], rtol=1e-9)

    def test_the_old_options_give_the_same_weights_by_label_and_list_ties_in_input_order(self):
        # twenty identical pairs: the earlier unstable sort scrambled tied rows beyond sixteen pairs; the values by
        # label were never different, and notebook 22's A4 reads them by label
        idx = pd.bdate_range("2020-01-01", periods=60)
        r = np.random.default_rng(2).standard_normal(60)
        kf = {(f"A{k}", f"B{k}"): pd.DataFrame({"resid": r, "beta": 1.0}, index=idx) for k in range(20)}
        w = suggest_position_weights(kf, risk="price_units", cap="one_pass")
        assert w["pair"].tolist() == [f"A{k}/B{k}" for k in range(20)]
        np.testing.assert_allclose(w["weight"], 1.0 / 20.0, rtol=0, atol=1e-15)

    def test_the_equal_method_is_unchanged_under_the_old_options(self):
        kf = _det_price_unit_frames([1.0, 1.2, 8.0, 10.0])
        w = suggest_position_weights(kf, method="equal", risk="price_units", cap="one_pass")
        np.testing.assert_allclose(w["weight"], 0.25)
        assert sorted(w["resid_var"]) == pytest.approx(sorted(_PREFIX_RESID_VAR), rel=1e-9)

    def test_resid_var_is_reported_under_both_risks(self):
        kf, prices = _random_pairs(n_pairs=2)
        a = suggest_position_weights(kf, prices=prices).set_index("pair")
        b = suggest_position_weights(kf, risk="price_units").set_index("pair")
        np.testing.assert_allclose(a["resid_var"], b["resid_var"].reindex(a.index), rtol=0, atol=0)
        assert a["var_per_dollar"].notna().all() and b["var_per_dollar"].isna().all()


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
