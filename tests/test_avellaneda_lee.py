"""Tests for the Avellaneda-Lee (2010) engine: sector assignment, trading time, the ETF regression,
the OU fit, the s-score, the bang-bang rule and position sizing.

Each test targets one equation or one interpretive decision documented in
``pairs/strategies/avellaneda_lee.py``'s docstrings: eq. 11-12 (the model), Appendix A / eq. A1-A2
(the OU fit and centered s-score), eq. 16 (bang-bang), eq. 20 (trading time), and this module's own
reading of how eq. 20's volume weighting enters a two-sided regression (weighted ``etf_residuals``).
"""
import numpy as np
import pandas as pd
import pytest

from pairs.strategies.avellaneda_lee import (
    assign_sector_etf,
    bang_bang_update,
    etf_residuals,
    ou_fit,
    positions_from_state,
    s_score,
    trading_time_factor,
)


def _simulate_ou_resid(W, kappa, m, sigma, seed, x0=None, n=1):
    """Discretize eq. 12-13 into the AR(1) residual increments :func:`ou_fit` expects.

    b = exp(-kappa/252), a = (1-b)m, Var(zeta) = sigma^2 (1-b^2) / (2 kappa) -- eq. A1 read in
    reverse: these are the population parameters a real fit would try to recover from ``resid``.
    """
    dt = 1.0 / 252.0
    b = np.exp(-kappa * dt)
    a = (1.0 - b) * m
    var_zeta = sigma ** 2 * (1.0 - b ** 2) / (2.0 * kappa)
    rng = np.random.default_rng(seed)
    X = np.zeros((W, n))
    X[0] = m if x0 is None else x0
    noise = rng.normal(0.0, np.sqrt(var_zeta), size=(W - 1, n))
    for k in range(1, W):
        X[k] = a + b * X[k - 1] + noise[k - 1]
    resid = np.diff(np.concatenate([np.zeros((1, n)), X], axis=0), axis=0)
    return resid


# ── ou_fit: recovering the generating process ──────────────────────────────────

class TestOuFitRecovery:
    def test_recovers_kappa_and_sigma_eq_on_a_long_window(self):
        """eq. A1's kappa and sigma_eq, recovered from a simulated path of known parameters."""
        kappa, m, sigma = 20.0, 0.01, 0.05
        sigma_eq_true = sigma / np.sqrt(2.0 * kappa)
        resid = _simulate_ou_resid(5000, kappa, m, sigma, seed=11)
        fit = ou_fit(resid)
        assert fit["model_ok"][0]
        assert abs(fit["kappa"][0] - kappa) / kappa < 0.25
        assert abs(fit["sigma_eq"][0] - sigma_eq_true) / sigma_eq_true < 0.15
        assert abs(fit["m"][0] - m) < 3.0 * sigma_eq_true

    def test_s_raw_equals_minus_m_over_sigma_eq_on_an_intercept_regression(self):
        """eq. A1's own remark: X_W = sum(eps) = 0 for an OLS-with-intercept residual, so
        s_raw = (X_W - m)/sigma_eq collapses to -m/sigma_eq exactly. Checked via the actual
        regression path (etf_residuals with weights=None), not by constructing X_W=0 by hand.
        """
        rng = np.random.default_rng(3)
        W, N = 60, 200
        etf_win = rng.normal(size=(W, N))
        stock_win = 0.6 * etf_win + rng.normal(scale=0.4, size=(W, N))
        resid = etf_residuals(stock_win, etf_win, weights=None)["resid"]

        assert np.allclose(resid.sum(axis=0), 0.0, atol=1e-10), "intercept regression must zero the sum"

        fit = ou_fit(resid)
        ok = np.isfinite(fit["s_raw"])
        assert ok.sum() > N * 0.3, "need enough surviving fits for the identity to be checked"
        expected = -fit["m"][ok] / fit["sigma_eq"][ok]
        np.testing.assert_allclose(fit["s_raw"][ok], expected, atol=1e-10)

    def test_bad_fits_are_nan_across_every_field_including_a_and_b(self):
        """b outside (0, 1) must NaN the whole column, not just the derived quantities."""
        # column 0: b > 1 (explosive, X grows without bound); column 1: a well-behaved fit
        w, kappa, m, sigma = 60, 15.0, 0.0, 0.01
        good = _simulate_ou_resid(w, kappa, m, sigma, seed=1, n=1)
        rng = np.random.default_rng(0)
        explosive = np.cumsum(rng.normal(size=w)).reshape(-1, 1) * np.arange(1, w + 1).reshape(-1, 1)
        resid = np.column_stack([explosive[:, 0], good[:, 0]])
        fit = ou_fit(resid)
        for key in ("a", "b", "kappa", "m", "sigma", "sigma_eq", "s_raw"):
            assert not np.isfinite(fit[key][0]), key
            assert np.isfinite(fit[key][1]), key
        assert fit["model_ok"][0] == False  # noqa: E712 (explicit bool, not just falsy-NaN)


# ── s-score sign convention and centering ───────────────────────────────────────

class TestSScore:
    def test_x60_below_m_gives_a_negative_s_and_a_buy_signal(self):
        """s < -1.25 with a passing model must open a long from flat (eq. 16's buy-to-open)."""
        kappa, m, sigma = 15.0, 0.0, 0.01
        sigma_eq = sigma / np.sqrt(2.0 * kappa)
        resid = _simulate_ou_resid(60, kappa, m, sigma, seed=0, x0=-60.0 * sigma_eq)
        fit = ou_fit(resid)
        assert fit["model_ok"][0]
        assert fit["s_raw"][0] < -1.25, fit["s_raw"]

        state = bang_bang_update(np.array([0]), fit["s_raw"], fit["model_ok"])
        assert state[0] == 1

    def test_centering_removes_the_cross_sectional_mean_of_m(self):
        """eq. 18 / A2: s = -(m - <m>)/sigma_eq when x_last = 0 (OLS-with-intercept case).
        nanmean(m_bar) must be exactly zero (up to fp)."""
        rng = np.random.default_rng(4)
        m = rng.normal(0.02, 0.01, size=50)
        m[::7] = np.nan                       # a scattered set of names with no valid fit
        sigma_eq = np.full(50, 0.01)
        fit = {"m": m, "sigma_eq": sigma_eq, "x_last": np.zeros(50)}

        s = s_score(fit, center=True)
        m_bar_implied = -s * sigma_eq         # invert s = -m_bar/sigma_eq (valid since x_last = 0)
        assert np.nanmean(m_bar_implied) == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(s[np.isnan(m)]).all()

    def test_uncentered_matches_raw_definition(self):
        """center=False must reproduce ou_fit's own s_raw = (x_last - m)/sigma_eq exactly, for
        both a zero x_last (the OLS-intercept case) and a non-zero one (the general case)."""
        fit = {
            "m": np.array([0.02, -0.01, np.nan, 0.05]),
            "sigma_eq": np.array([0.01, 0.02, 0.01, 0.02]),
            "x_last": np.array([0.0, 0.0, 0.0, 0.03]),
        }
        fit["s_raw"] = (fit["x_last"] - fit["m"]) / fit["sigma_eq"]

        s = s_score(fit, center=False)
        np.testing.assert_allclose(s, fit["s_raw"], equal_nan=True)
        np.testing.assert_allclose(s[:2], [-2.0, 0.5])   # x_last = 0 here: matches -m/sigma_eq
        assert np.isnan(s[2])
        assert s[3] == pytest.approx((0.03 - 0.05) / 0.02)   # x_last != 0: shortcut would be wrong

    def test_wls_nonzero_x_last_moves_the_score_away_from_the_shortcut(self):
        """With a weighted (trading-time) regression the residuals need not sum to zero, so
        x_last is materially non-zero and s_score must differ from the OLS-intercept shortcut
        -m_bar/sigma_eq by exactly x_last/sigma_eq -- the whole point of DEFECT 1's fix."""
        rng = np.random.default_rng(21)
        W, N = 60, 40
        etf_win = rng.normal(size=(W, N))
        stock_win = 0.5 * etf_win + rng.normal(scale=0.3, size=(W, N))
        weights = rng.uniform(0.2, 3.0, size=(W, N))     # non-constant -> WLS, not plain OLS

        resid = etf_residuals(stock_win, etf_win, weights=weights)["resid"]
        assert not np.allclose(resid.sum(axis=0), 0.0, atol=1e-6), \
            "weighted residuals should not sum to zero in general"

        fit = ou_fit(resid)
        ok = fit["model_ok"] & np.isfinite(fit["x_last"])
        assert ok.sum() > N * 0.3, "need enough surviving fits to check the identity"
        assert np.nanmedian(np.abs(fit["x_last"][ok])) > 0, "x_last should be materially non-zero"

        s_full = s_score(fit, center=True)
        m_bar = fit["m"] - np.nanmean(fit["m"])
        shortcut = -m_bar / fit["sigma_eq"]

        diff = s_full[ok] - shortcut[ok]
        expected_diff = fit["x_last"][ok] / fit["sigma_eq"][ok]
        np.testing.assert_allclose(diff, expected_diff, atol=1e-10)
        assert np.median(np.abs(diff)) > 1e-6

    def test_s_score_requires_x_last(self):
        """No silent default of 0 for a missing x_last -- that default is exactly the bug."""
        fit = {"m": np.array([0.02]), "sigma_eq": np.array([0.01])}
        with pytest.raises(KeyError):
            s_score(fit, center=False)


# ── bang-bang transition table (eq. 16) ────────────────────────────────────────

class TestBangBang:
    """Every row of the table in bang_bang_update's docstring, one name per row."""

    def test_transition_table(self):
        #                state, s,      model_ok   expected  why
        rows = [
            (0, -2.00, False, 0, "not model_ok blocks an open even though s would open it"),
            (0, -2.00, True, 1, "flat, s < -open -> long"),
            (0, 2.00, True, -1, "flat, s > +open -> short"),
            (0, 0.00, True, 0, "flat, s inside the band -> stays flat"),
            (1, -0.40, True, 0, "long, s > close_long (-0.5) -> flat"),
            (1, -0.60, True, 1, "long, s <= close_long -> stays long"),
            (-1, 0.70, True, 0, "short, s < close_short (0.75) -> flat"),
            (-1, 0.80, True, -1, "short, s >= close_short -> stays short"),
            (1, 3.00, False, 0, "model_ok False closes an open long"),
            (-1, -3.00, False, 0, "model_ok False closes an open short"),
            (1, 3.00, True, 0, "long, s far above open -> close to flat, never straight to short"),
            (-1, -3.00, True, 0, "short, s far below -open -> close to flat, never straight to long"),
        ]
        state = np.array([r[0] for r in rows])
        s = np.array([r[1] for r in rows], dtype=float)
        model_ok = np.array([r[2] for r in rows], dtype=bool)
        expected = np.array([r[3] for r in rows])

        new = bang_bang_update(state, s, model_ok)
        for i, r in enumerate(rows):
            assert new[i] == expected[i], r[4]

    def test_nan_s_is_treated_like_a_failed_model(self):
        state = np.array([0, 1, -1])
        s = np.array([np.nan, np.nan, np.nan])
        model_ok = np.array([True, True, True])
        new = bang_bang_update(state, s, model_ok)
        assert (new == 0).all()

    def test_output_dtype_matches_input_state(self):
        state = np.array([0, 1, -1], dtype=np.int64)
        new = bang_bang_update(state, np.array([0.0, 0.0, 0.0]), np.array([True, True, True]))
        assert new.dtype == state.dtype

    def test_unsigned_state_dtype_raises_instead_of_wrapping(self):
        """An unsigned dtype cannot hold -1; a caller passing one must get a clear error, not a
        silently wrapped position code (uint8's -1 wraps to 255, uint16's to 65535)."""
        for dtype in (np.uint8, np.uint16):
            state = np.array([0], dtype=dtype)
            with pytest.raises(ValueError):
                bang_bang_update(state, np.array([2.0]), np.array([True]))

    def test_signed_narrow_dtype_still_works(self):
        """int8 is signed and can hold -1; this must keep working (only unsigned is rejected)."""
        state = np.array([0], dtype=np.int8)
        new = bang_bang_update(state, np.array([2.0]), np.array([True]))
        assert new.dtype == np.int8
        assert new[0] == -1


# ── trading_time_factor (eq. 20) ────────────────────────────────────────────────

class TestTradingTimeFactor:
    def test_is_exactly_one_for_constant_volume(self):
        idx = pd.date_range("2020-01-01", periods=40, freq="B")
        vol = pd.DataFrame(5.0, index=idx, columns=["A", "B"])
        f = trading_time_factor(vol, window=10)
        valid = f.iloc[10:]
        assert (valid == 1.0).all().all()

    def test_nan_for_the_first_window_rows(self):
        idx = pd.date_range("2020-01-01", periods=40, freq="B")
        rng = np.random.default_rng(2)
        vol = pd.DataFrame(rng.uniform(1, 10, size=(40, 1)), index=idx, columns=["A"])
        f = trading_time_factor(vol, window=10)
        assert f["A"].iloc[:10].isna().all()
        assert f["A"].iloc[10:].notna().all()

    def test_no_look_ahead_tomorrows_volume_does_not_move_todays_factor(self):
        idx = pd.date_range("2020-01-01", periods=80, freq="B")
        rng = np.random.default_rng(5)
        vol = pd.DataFrame(rng.uniform(1, 10, size=(80, 1)), index=idx, columns=["A"])
        f_before = trading_time_factor(vol, window=10)

        vol_perturbed = vol.copy()
        vol_perturbed.iloc[50, 0] = vol_perturbed.iloc[50, 0] * 7.0
        f_after = trading_time_factor(vol_perturbed, window=10)

        pd.testing.assert_series_equal(f_before["A"].iloc[:50], f_after["A"].iloc[:50])
        assert f_before["A"].iloc[50] != pytest.approx(f_after["A"].iloc[50])

    def test_todays_own_volume_is_excluded_from_the_trailing_average(self):
        """Directly exercises the 'exclusive of t' contract: perturbing V_t itself must change only
        the denominator of the ratio, not <dV> (the numerator). A mutant that folds t into the
        trailing average (e.g. rolling(window+1) with no shift, or a missing .shift(1)) would move
        the implied <dV> when V_t changes -- this is not caught by perturbing a *future* row, which
        only tests that tomorrow doesn't leak into today.
        """
        idx = pd.date_range("2020-01-01", periods=80, freq="B")
        rng = np.random.default_rng(5)
        vol = pd.DataFrame(rng.uniform(1, 10, size=(80, 1)), index=idx, columns=["A"])
        f_before = trading_time_factor(vol, window=10)
        avg_prev_before = f_before["A"].iloc[50] * vol["A"].iloc[50]        # implied <dV> at row 50

        vol_perturbed = vol.copy()
        vol_perturbed.iloc[50, 0] = vol_perturbed.iloc[50, 0] * 7.0    # perturb V_t itself, not a future row
        f_after = trading_time_factor(vol_perturbed, window=10)
        avg_prev_after = f_after["A"].iloc[50] * vol_perturbed["A"].iloc[50]  # recover <dV> using the NEW V_t

        # <dV> (the trailing average) must be unchanged by today's own volume...
        assert avg_prev_before == pytest.approx(avg_prev_after)
        # ...so the factor itself, which divides by the now-different V_t, does change.
        assert f_before["A"].iloc[50] != pytest.approx(f_after["A"].iloc[50])
        # Rows before 50 never touch row 50 at all; rows 61+ have row 50 outside their trailing
        # window (window=10, so only t = 51..60 include row 50 in their <dV>) -- both must be
        # untouched. Rows 51..60 are excluded from this check: they legitimately DO change, since
        # row 50 is inside their own trailing window.
        unaffected = [i for i in range(10, 80) if i < 50 or i > 60]
        pd.testing.assert_series_equal(f_before["A"].iloc[unaffected], f_after["A"].iloc[unaffected])

    def test_clipped_and_nan_on_zero_volume(self):
        idx = pd.date_range("2020-01-01", periods=20, freq="B")
        vol = pd.DataFrame(5.0, index=idx, columns=["A"])
        vol.iloc[15, 0] = 0.0
        vol.iloc[5, 0] = 1e-6            # would blow the ratio far past the clip
        f = trading_time_factor(vol, window=10, clip=(0.1, 10.0))
        assert np.isnan(f["A"].iloc[15])
        assert f["A"].iloc[16] <= 10.0   # the huge-ratio day two rows later is clipped, not NaN


# ── assign_sector_etf ────────────────────────────────────────────────────────────

class TestAssignSectorEtf:
    def test_recovers_the_generating_etf(self):
        rng = np.random.default_rng(1)
        T = 300
        idx = pd.date_range("2020-01-01", periods=T, freq="B")
        etfs = ["XLF", "XLK", "XLE"]
        etf_ret = pd.DataFrame(rng.normal(scale=0.01, size=(T, 3)), index=idx, columns=etfs)

        names = [f"S{i}" for i in range(15)]
        truth = {n: etfs[i % 3] for i, n in enumerate(names)}
        stock_ret = pd.DataFrame(index=idx, columns=names, dtype=float)
        for n in names:
            beta = rng.uniform(0.5, 1.5)
            stock_ret[n] = beta * etf_ret[truth[n]] + rng.normal(scale=0.002, size=T)

        out = assign_sector_etf(stock_ret, etf_ret, min_obs=200)
        assert (out["etf"] == pd.Series(truth)).all()
        assert (out["r2"] > 0.8).all()

    def test_nan_when_no_etf_clears_min_obs(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        rng = np.random.default_rng(6)
        etf_ret = pd.DataFrame(rng.normal(size=(50, 2)), index=idx, columns=["X", "Y"])
        stock_ret = pd.DataFrame(rng.normal(size=(50, 1)), index=idx, columns=["S0"])
        out = assign_sector_etf(stock_ret, etf_ret, min_obs=200)
        assert out.loc["S0", "etf"] is None
        assert np.isnan(out.loc["S0", "r2"])


# ── etf_residuals ──────────────────────────────────────────────────────────────

class TestEtfResiduals:
    def test_weights_of_ones_equals_plain_ols(self):
        rng = np.random.default_rng(7)
        W, N = 80, 10
        etf_win = rng.normal(size=(W, N))
        stock_win = 0.6 * etf_win + rng.normal(scale=0.3, size=(W, N))

        plain = etf_residuals(stock_win, etf_win, weights=None)
        ones = etf_residuals(stock_win, etf_win, weights=np.ones((W, N)))

        np.testing.assert_allclose(plain["alpha"], ones["alpha"])
        np.testing.assert_allclose(plain["beta"], ones["beta"])
        np.testing.assert_allclose(plain["resid"], ones["resid"])
        np.testing.assert_allclose(plain["r2"], ones["r2"])

    def test_plain_ols_matches_numpy_polyfit(self):
        rng = np.random.default_rng(8)
        W, N = 50, 4
        etf_win = rng.normal(size=(W, N))
        stock_win = -0.3 * etf_win + 0.02 + rng.normal(scale=0.2, size=(W, N))
        out = etf_residuals(stock_win, etf_win, weights=None)
        for i in range(N):
            beta_np, alpha_np = np.polyfit(etf_win[:, i], stock_win[:, i], deg=1)
            assert out["beta"][i] == pytest.approx(beta_np, abs=1e-8)
            assert out["alpha"][i] == pytest.approx(alpha_np, abs=1e-8)

    def test_r2_is_nan_not_inflated_for_a_near_degenerate_stale_name(self):
        """A stale/illiquid name (almost all daily returns exactly zero) must not report a
        fabricated high r2 just because its return variance happens to land near float precision.
        Regression guard for a fixed +1e-18 denominator epsilon, which used to turn an honest
        'no relationship' (r2 ~ 0.02 without the fudge) into a fabricated r2 = 0.72."""
        rng = np.random.default_rng(42)
        stock_win = np.zeros((60, 1))
        stock_win[59, 0] = 5e-9                       # one tiny genuine tick, rest exactly flat
        etf_win = rng.normal(scale=0.01, size=(60, 1))
        out = etf_residuals(stock_win, etf_win, weights=None)
        assert np.isnan(out["r2"][0]), out["r2"]
        # beta/alpha are unaffected by the r2 floor and stay near zero (no genuine relationship)
        assert abs(out["beta"][0]) < 1e-3

    def test_r2_finite_and_correct_for_ordinary_scale_returns(self):
        """The near-zero-variance floor must not touch names with realistic return variance."""
        rng = np.random.default_rng(9)
        W, N = 80, 5
        etf_win = rng.normal(scale=0.01, size=(W, N))
        stock_win = 0.8 * etf_win + rng.normal(scale=0.005, size=(W, N))
        out = etf_residuals(stock_win, etf_win, weights=None)
        assert np.isfinite(out["r2"]).all()
        assert (out["r2"] > 0.5).all()

    def test_weighted_matches_lstsq_on_the_scaled_system(self):
        """weights == f: fitting [f, f*x] against f*y (OLS) is the WLS-with-weight-f^2 identity
        this module's docstring claims."""
        rng = np.random.default_rng(0)
        W, N = 80, 6
        etf_win = rng.normal(size=(W, N))
        stock_win = 0.7 * etf_win + rng.normal(scale=0.3, size=(W, N)) + 0.01
        f = rng.uniform(0.5, 2.0, size=(W, N))

        out = etf_residuals(stock_win, etf_win, weights=f)
        for i in range(N):
            design = np.column_stack([f[:, i], f[:, i] * etf_win[:, i]])
            target = f[:, i] * stock_win[:, i]
            a_lstsq, b_lstsq = np.linalg.lstsq(design, target, rcond=None)[0]
            assert out["alpha"][i] == pytest.approx(a_lstsq, abs=1e-8)
            assert out["beta"][i] == pytest.approx(b_lstsq, abs=1e-8)
            resid_lstsq = target - design @ [a_lstsq, b_lstsq]
            np.testing.assert_allclose(out["resid"][:, i], resid_lstsq, atol=1e-8)


# ── positions_from_state ──────────────────────────────────────────────────────

class TestPositionsFromState:
    def test_nets_etf_legs_across_names(self):
        state = pd.Series({"A": 1, "B": -1, "C": 1, "D": 0, "E": 1})
        beta = pd.Series({"A": 0.5, "B": 0.8, "C": 1.2, "D": 0.9, "E": np.nan})
        etf_of = pd.Series({"A": "XLF", "B": "XLF", "C": "XLK", "D": "XLK", "E": "XLF"})
        notional = 1000.0

        pos = positions_from_state(state, beta, etf_of, notional=notional)

        assert pos["A"] == pytest.approx(1000.0)
        assert pos["B"] == pytest.approx(-1000.0)
        assert pos["C"] == pytest.approx(1000.0)
        assert pos["D"] == pytest.approx(0.0)
        assert "E" not in pos.index                       # NaN beta: contributes nothing

        # XLF: -(1*0.5*1000 + (-1)*0.8*1000) = -(500 - 800) = 300; E excluded (NaN beta)
        assert pos["XLF"] == pytest.approx(300.0)
        # XLK: -(1*1.2*1000 + 0*0.9*1000) = -1200
        assert pos["XLK"] == pytest.approx(-1200.0)
        assert len(pos) == 6

    def test_missing_etf_assignment_also_contributes_nothing(self):
        state = pd.Series({"A": 1})
        beta = pd.Series({"A": 0.5})
        etf_of = pd.Series({"A": np.nan})
        pos = positions_from_state(state, beta, etf_of, notional=1000.0)
        assert pos.empty
