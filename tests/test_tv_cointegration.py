"""Tests for the time-varying cointegration model and the dynamic-hedge gate.

The gate exists because of a specific failure documented in notebook 07 §1: the package's
Kalman-residual stationarity test declares two *independent random walks* "stationary" 90% of the
time at the default q=1e-5. A time-varying-coefficient filter manufactures a stationary residual
for any two I(1) series, so nothing downstream of it can tell you whether a dynamic hedge was
warranted. These tests pin the three things that matter: the model recovers the regime it was
generated from, the gate declines the trap, and the decision tree maps regimes to hedges the way
the paper's §4.2 tables say.

The bootstrap is the slow part, so most tests use a reduced B. The regime-recovery test is marked
slow and uses the paper's own DGP and seed, because changing either changes what "correct" means.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("statsmodels")
from pairs.stats.tv_cointegration import (        # noqa: E402
    FIXED, NO_COINTEGRATION, TIME_VARYING, UNDECIDED,
    HedgeVerdict, bootstrap_test, classify_cointegration, fit_tvssm, recommend_hedge,
    simulate_tvssm, t_sigma, t_theta,
)

warnings.filterwarnings("ignore")


# ── the DGP itself ───────────────────────────────────────────────────────────

class TestSimulator:
    def test_follows_the_papers_section_4_parameterization(self):
        """These constants are not arbitrary — Table 1's size and power are measured on them."""
        d = simulate_tvssm(200, theta=0.8, sigma_eta=0.0, seed=11)
        assert {"y", "x", "beta_true", "w_true"} == set(d.columns)
        # mu = 1/(1-T) = 10/3 and beta_0 = mu/(1-T), so beta sits near 11 and does not wander
        assert 10.0 < d["beta_true"].mean() < 12.0
        assert d["beta_true"].std() < 1e-9, "sigma_eta=0 must give a constant coefficient"

    def test_sigma_eta_is_what_makes_the_coefficient_move(self):
        still = simulate_tvssm(200, 0.8, 0.0, seed=11)["beta_true"]
        moving = simulate_tvssm(200, 0.8, 0.3, seed=11)["beta_true"]
        assert still.std() < 1e-9 < moving.std()

    def test_x_is_a_random_walk_so_the_levels_are_integrated(self):
        x = simulate_tvssm(400, 0.8, 0.0, seed=3)["x"].to_numpy()
        steps = np.diff(x)
        assert abs(steps.mean()) < 0.2 and 0.8 < steps.std() < 1.2


# ── the model ────────────────────────────────────────────────────────────────

class TestFit:
    def test_recovers_the_parameters_it_was_generated_from(self):
        d = simulate_tvssm(300, theta=0.8, sigma_eta=0.3, seed=11)
        res = fit_tvssm(d["y"], d["x"])
        theta, sigma_eta = res.params[3], abs(res.params[4])
        assert 0.5 < theta < 1.0, theta
        assert 0.15 < sigma_eta < 0.5, sigma_eta

    def test_a_restriction_is_actually_imposed(self):
        d = simulate_tvssm(150, 0.8, 0.3, seed=11)
        assert fit_tvssm(d["y"], d["x"], fixed={"theta": 1.0}).params[3] == pytest.approx(1.0)
        assert abs(fit_tvssm(d["y"], d["x"], fixed={"sigma_eta": 0.0}).params[4]) == pytest.approx(0.0)

    def test_the_t_statistics_point_the_way_the_hypotheses_do(self):
        d = simulate_tvssm(200, 0.8, 0.3, seed=11)
        res = fit_tvssm(d["y"], d["x"])
        assert t_theta(res) < 0, "H0 theta=1 is rejected from below"
        assert t_sigma(res) > 0, "sigma_eta enters squared, so its statistic is unsigned"

    @pytest.mark.parametrize("bad, err", [
        (("short",), "at least 20"),
        (("ragged",), "same length"),
        (("nan",), "finite"),
    ])
    def test_unusable_input_is_refused_with_a_reason(self, bad, err):
        kind = bad[0]
        if kind == "short":
            y, x = np.arange(10.0), np.arange(10.0)
        elif kind == "ragged":
            y, x = np.arange(50.0), np.arange(40.0)
        else:
            y, x = np.arange(50.0), np.concatenate([np.arange(49.0), [np.nan]])
        with pytest.raises(ValueError, match=err):
            fit_tvssm(y, x)


# ── the bootstrap ────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_returns_a_null_distribution_and_a_one_sided_p_value(self):
        d = simulate_tvssm(80, 0.8, 0.0, seed=11)
        out = bootstrap_test(d["y"], d["x"], {"theta": 1.0}, B=25, seed=0, n_jobs=1)
        assert out["boot"].size > 5 and np.isfinite(out["boot"]).all()
        assert 0.0 <= out["p_value"] <= 1.0
        # left tail: the p-value is the share of the null at or below the observed statistic
        assert out["p_value"] == pytest.approx((out["boot"] <= out["stat"]).mean())

    def test_only_the_two_documented_restrictions_are_accepted(self):
        d = simulate_tvssm(60, 0.8, 0.0, seed=11)
        with pytest.raises(ValueError, match="theta.*sigma_eta"):
            bootstrap_test(d["y"], d["x"], {"T": 0.5}, B=5, n_jobs=1)

    def test_it_is_reproducible_from_its_seed(self):
        d = simulate_tvssm(60, 0.8, 0.0, seed=11)
        kw = dict(B=15, seed=4, n_jobs=1)
        a = bootstrap_test(d["y"], d["x"], {"theta": 1.0}, **kw)
        b = bootstrap_test(d["y"], d["x"], {"theta": 1.0}, **kw)
        assert a["p_value"] == b["p_value"] and np.allclose(a["boot"], b["boot"])


# ── the gate ─────────────────────────────────────────────────────────────────

class TestGate:
    def test_two_independent_random_walks_do_not_get_a_dynamic_hedge(self):
        """Notebook 07 §1's trap: the Kalman-residual test calls these stationary 90% of the time."""
        rng = np.random.default_rng(7)
        x = np.cumsum(rng.standard_normal(100))
        y = 100 + np.cumsum(rng.standard_normal(100))
        v = recommend_hedge(y, x, B=99, seed=1)
        assert v.hedge != "dynamic", v
        assert not v.dynamic_warranted

    @pytest.mark.slow
    @pytest.mark.parametrize("theta, sigma_eta, want_hedge, want_verdict", [
        (1.0, 0.0, "none", NO_COINTEGRATION),
        (0.8, 0.0, "static", FIXED),
        (0.8, 0.3, "dynamic", TIME_VARYING),
    ])
    def test_each_regime_gets_the_hedge_it_should(self, theta, sigma_eta, want_hedge, want_verdict):
        """The paper's DGP and notebook 07's seed: changing either changes what 'correct' means.

        Single draws at n=100, so this pins behavior rather than proving the test's power —
        notebook 07 §4 measures that properly and finds the theta test weak at this sample size.
        """
        d = simulate_tvssm(100, theta, sigma_eta, seed=11)
        v = recommend_hedge(d["y"], d["x"], B=199, seed=0)
        assert (v.hedge, v.verdict) == (want_hedge, want_verdict), v

    def test_an_undecided_test_falls_back_to_the_safer_hedge(self, monkeypatch):
        """Conservative by design: fewer moving parts when the evidence cannot resolve."""
        import pairs.stats.tv_cointegration as tv
        monkeypatch.setattr(tv, "classify_cointegration", lambda *a, **k: {
            "verdict": UNDECIDED, "theta_hat": np.nan, "sigma_eta_hat": np.nan,
            "p_theta": np.nan, "p_sigma": np.nan, "t_theta": np.nan, "t_sigma": np.nan,
            "converged": True})
        v = tv.recommend_hedge(np.arange(50.0), np.arange(50.0))
        assert v.hedge == "static" and "conservative" in v.reason

    @pytest.mark.parametrize("theta, expect_hedge, expect_plausible, why", [
        (0.58, "dynamic", True, "a decaying error is what a dynamic hedge is for"),
        (-0.65, "static", False, "alternating sign each bar is oscillation, not a long-run relation"),
        (1.10, "static", False, "outside the stationary region, so the verdict is uninterpretable"),
    ])
    def test_a_time_varying_verdict_is_downgraded_on_an_implausible_theta(
            self, monkeypatch, theta, expect_hedge, expect_plausible, why):
        """The optimizer leaves theta unconstrained and it wanders.

        On this repository's 2,151 screen survivors, 14% of fits return |theta| > 1 and 30% of the
        time-varying verdicts do. The paper's classification is reported unchanged; the hedge
        recommendation is what gets downgraded.
        """
        import pairs.stats.tv_cointegration as tv
        monkeypatch.setattr(tv, "classify_cointegration", lambda *a, **k: {
            "verdict": TIME_VARYING, "theta_hat": theta, "sigma_eta_hat": 0.3,
            "p_theta": 0.001, "p_sigma": 0.004, "t_theta": -3.0, "t_sigma": 3.0,
            "converged": True})
        v = tv.recommend_hedge(np.arange(50.0), np.arange(50.0))
        assert (v.hedge, v.theta_plausible) == (expect_hedge, expect_plausible), (v, why)
        assert v.verdict == TIME_VARYING, "the paper's classification must be reported unchanged"

    def test_a_non_converged_fit_yields_no_verdict_at_all(self, monkeypatch):
        """A theta that never left its 0.9 starting value is an input, not an estimate.

        6% of daily fits over this repository's screen survivors come back exactly 0.9, the
        optimizer's own start, on a flat likelihood. statsmodels reports convergence and the gate
        must read it.
        """
        import pairs.stats.tv_cointegration as tv
        monkeypatch.setattr(tv, "classify_cointegration", lambda *a, **k: {
            "verdict": TIME_VARYING, "theta_hat": 0.9, "sigma_eta_hat": 0.3,
            "p_theta": 0.01, "p_sigma": 0.01, "t_theta": -3.0, "t_sigma": 3.0,
            "converged": False})
        v = tv.recommend_hedge(np.arange(50.0), np.arange(50.0))
        assert v.hedge == "static" and v.verdict == UNDECIDED
        assert v.converged is False and "did not converge" in v.reason

    def test_convergence_is_reported_on_a_fit_that_does_converge(self):
        d = simulate_tvssm(80, 0.8, 0.0, seed=11)
        v = recommend_hedge(d["y"], d["x"], B=49, seed=0, n_jobs=1)
        assert v.converged is True

    def test_a_no_cointegration_verdict_is_never_upgraded_by_the_theta_check(self, monkeypatch):
        """Downgrading must not accidentally turn 'do not trade' into 'trade statically'."""
        import pairs.stats.tv_cointegration as tv
        monkeypatch.setattr(tv, "classify_cointegration", lambda *a, **k: {
            "verdict": NO_COINTEGRATION, "theta_hat": 1.4, "sigma_eta_hat": 0.0,
            "p_theta": 0.6, "p_sigma": 0.7, "t_theta": -0.2, "t_sigma": 0.1,
            "converged": True})
        assert tv.recommend_hedge(np.arange(50.0), np.arange(50.0)).hedge == "none"

    def test_the_verdict_carries_its_own_evidence(self):
        d = simulate_tvssm(80, 0.8, 0.0, seed=11)
        v = recommend_hedge(d["y"], d["x"], B=49, seed=0, n_jobs=1)
        assert isinstance(v, HedgeVerdict)
        for field in ("theta_hat", "sigma_eta_hat", "p_theta", "p_sigma"):
            assert getattr(v, field) is not None
        assert v.reason and v.verdict in {NO_COINTEGRATION, FIXED, TIME_VARYING, UNDECIDED}
        assert str(v).startswith(v.verdict)


def test_the_gate_is_reachable_from_the_package_root():
    import pairs
    assert callable(pairs.recommend_hedge) and callable(pairs.classify_cointegration)
