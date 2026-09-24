"""Time-varying cointegration tests, and the gate that decides whether a dynamic hedge is warranted.

Eroğlu, Miller and Yiğit (2021), *Time-varying cointegration and the Kalman filter*, Econometric
Reviews 40(5). Notebook 07 developed this against the paper; the model and its bootstrap live here
so that the pipeline can use them rather than only the notebook.

**Why this exists.** The rest of this package will happily fit a Kalman dynamic hedge to any two
series, and notebook 07 §1 shows what that costs: on two *independent random walks* at price-like
levels, the package's Kalman-residual stationarity test declares the pair "stationary" 90% of the
time at the default ``q=1e-5``, while Engle-Granger on levels stays near its nominal size. A
time-varying-coefficient filter manufactures a stationary residual for any two I(1) series, so any
test applied to that residual inherits the manufacture. Nothing in the pipeline previously asked
whether a *dynamic* hedge was warranted at all.

**What the model does instead.** Put the regression error in the state and estimate its persistence:

.. math::
    y_t = \\alpha + x_t \\beta_t + w_t, \\qquad
    \\beta_t = \\mu + T\\beta_{t-1} + \\eta_t, \\qquad
    w_t = \\theta w_{t-1} + \\varepsilon_t

Two parameters carry the decision. :math:`\\theta` is the persistence of the error: at
:math:`\\theta = 1` the error is a random walk and there is no long-run relation, so rejecting
:math:`\\theta = 1` *downwards* is the cointegration test. :math:`\\sigma_\\eta` is how much the
coefficient actually moves: at :math:`\\sigma_\\eta = 0` the relationship is fixed, and rejecting
that *upwards* is what licenses a dynamic hedge.

Together they give three regimes, which is the whole point — "fixed or time-varying" becomes a
testable question rather than a modelling assumption:

===========================  ======================  ==================================
reject :math:`\\theta = 1`?    reject :math:`\\sigma_\\eta = 0`?   verdict
===========================  ======================  ==================================
no                           --                      no cointegration
yes                          no                      fixed cointegration
yes                          yes                     time-varying cointegration
===========================  ======================  ==================================

**Use the bootstrap, not the asymptotic critical values.** Notebook 07 §4 reproduces the paper's
size study: the asymptotic :math:`t` values are badly sized for the :math:`\\theta` test at
practical sample sizes, and the Hessian is degenerate at :math:`\\sigma_\\eta = 0`, so the
Hessian-based standard error there is not trustworthy on its own. The paper's innovation bootstrap
costs a few seconds a pair and is the default here.

**What this does not do.** It does not silently change any hedge. :func:`recommend_hedge` returns a
recommendation and the evidence behind it; wiring it into a screen is the caller's choice, because
forcing it would move every result in notebooks 01-17. Notebook 07 §6 argues it belongs *upstream*
of the Kalman step, alongside the Engle-Granger and Johansen gates, not downstream of it.

Scope, following the notebook: one regressor and :math:`k = 0` extra :math:`\\Delta w` lags, where
the paper selects lags by BIC; Gaussian ML with a fixed diffuse prior on :math:`\\beta_0`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    "TVCointModel",
    "fit_tvssm",
    "t_theta",
    "t_sigma",
    "bootstrap_test",
    "classify_cointegration",
    "recommend_hedge",
    "HedgeVerdict",
]

PARAM_NAMES = ["alpha", "mu", "T", "theta", "sigma_eta", "sigma_eps"]

NO_COINTEGRATION = "no cointegration"
FIXED = "fixed cointegration"
TIME_VARYING = "time-varying cointegration"
UNDECIDED = "undecided"


def _mle_model():
    """statsmodels is imported lazily: this module is only needed by callers who ask for it."""
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    return MLEModel


def _build_model_class():
    MLEModel = _mle_model()

    class _TVCointModel(MLEModel):
        """Eroğlu-Miller-Yiğit TVSSM with one regressor and k = 0."""

        param_names = PARAM_NAMES

        def __init__(self, y, x, beta0_var: float = 1e6):
            import statsmodels.api as sm
            y = np.asarray(y, float)
            x = np.asarray(x, float)
            if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
                raise ValueError(f"y and x must be 1-D and the same length, got {y.shape} and {x.shape}")
            if len(y) < 20:
                raise ValueError(f"need at least 20 observations to fit the TVSSM, got {len(y)}")
            if not (np.isfinite(y).all() and np.isfinite(x).all()):
                raise ValueError("y and x must be finite; drop or fill missing values first")
            super().__init__(y, k_states=2, k_posdef=2, initialization="known",
                             initial_state=np.zeros(2),
                             initial_state_cov=np.diag([beta0_var, 0.0]))
            design = np.zeros((1, 2, len(y)))
            design[0, 0, :] = x                      # time-varying design row [x_t, 1]
            design[0, 1, :] = 1.0
            self.ssm["design"] = design
            self.ssm["selection"] = np.eye(2)
            self.ssm["obs_cov"] = np.zeros((1, 1))
            ols = sm.OLS(y, sm.add_constant(x)).fit()          # starting values from the static fit
            self._start = np.array([ols.params[0], 0.1 * ols.params[1], 0.9, 0.9,
                                    0.05 * abs(ols.params[1]) + 1e-3, np.std(ols.resid)])

        @property
        def start_params(self):
            return self._start

        def transform_params(self, u):
            from scipy.special import expit
            c = u.copy()
            c[2] = expit(u[2])                                  # T constrained to (0, 1)
            return c

        def untransform_params(self, c):
            u = c.copy()
            T = np.clip(c[2], 1e-6, 1 - 1e-6)
            u[2] = np.log(T / (1 - T))
            return u

        def update(self, params, **kw):
            params = super().update(params, **kw)
            alpha, mu, T, theta, s_eta, s_eps = params
            self.ssm["obs_intercept"] = np.array([[alpha]])
            self.ssm["state_intercept"] = np.array([[mu], [0.0]])
            self.ssm["transition"] = np.array([[T, 0.0], [0.0, theta]])
            self.ssm["state_cov"] = np.diag([s_eta ** 2, s_eps ** 2])

    return _TVCointModel


_MODEL = None


def TVCointModel(y, x, beta0_var: float = 1e6):                 # noqa: N802 — it is a class factory
    """The TVSSM state-space model for one pair. See the module docstring for the specification."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model_class()
    return _MODEL(y, x, beta0_var=beta0_var)


def fit_tvssm(y, x, fixed: Optional[Dict[str, float]] = None, maxiter: int = 500):
    """Maximum-likelihood fit. ``fixed`` imposes a restriction, e.g. ``{"theta": 1.0}``.

    ``cov_type="oim"`` gives the Hessian-based standard errors the paper uses. They are the input
    to the t-statistics, and at ``sigma_eta = 0`` the Hessian is degenerate — which is exactly why
    the p-values come from the bootstrap rather than from these standard errors.
    """
    model = TVCointModel(y, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # optimiser chatter; joblib resets kernel filters
        if fixed:
            with model.fix_params(fixed):
                return model.fit(disp=False, maxiter=maxiter, cov_type="oim")
        return model.fit(disp=False, maxiter=maxiter, cov_type="oim")


def t_theta(res) -> float:
    """H0: theta = 1, left-sided. Below the bootstrap 5% quantile rejects no-cointegration."""
    return float((res.params[3] - 1.0) / res.bse[3])


def t_sigma(res) -> float:
    """H0: sigma_eta = 0, right-sided. sigma_eta enters squared, so its sign carries no meaning."""
    return float(abs(res.params[4]) / res.bse[4])


def _refit_stat(y_b, x, left_tail):
    warnings.filterwarnings("ignore")              # this runs in a joblib worker process
    try:
        res = fit_tvssm(y_b, x)
        return t_theta(res) if left_tail else t_sigma(res)
    except Exception:
        return np.nan


def bootstrap_test(y, x, null: Dict[str, float], *, B: int = 199, seed: int = 0,
                   n_jobs: int = -1) -> dict:
    """The paper's §3.1 innovation bootstrap.

    ``null`` is ``{"theta": 1.0}`` (left-tailed) or ``{"sigma_eta": 0.0}`` (right-tailed). The
    restricted fit supplies the data-generating process: its predicted beta path and its own
    residuals are resampled, so the null distribution is conditioned on the pair at hand rather
    than assumed.

    A NaN p-value means the test could not be decided — the unrestricted standard error failed, or
    every bootstrap replication did. It never counts as a rejection.
    """
    from joblib import Parallel, delayed

    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    key = next(iter(null))
    if key not in ("theta", "sigma_eta"):
        raise ValueError(f"null must restrict 'theta' or 'sigma_eta', got {key!r}")
    left_tail = key == "theta"
    stat = t_theta if left_tail else t_sigma

    res_U = fit_tvssm(y, x)
    res_R = fit_tvssm(y, x, fixed=null)
    tau = stat(res_U)

    alpha_R, theta_R = res_R.params[0], res_R.params[3]
    beta_pred = res_R.predicted_state[0, :n]                    # t|t-1 for t = 1..n
    w_pred = res_R.predicted_state[1, :n]
    eps = w_pred[1:] - theta_R * w_pred[:-1]
    eps = eps - eps.mean()

    rng = np.random.default_rng(seed)
    ys = []
    for _ in range(B):
        e_b = rng.choice(eps, size=n, replace=True)
        w_b = np.empty(n)
        wv = 0.0
        for t in range(n):
            wv = theta_R * wv + e_b[t]
            w_b[t] = wv
        ys.append(alpha_R + x * beta_pred + w_b)
    boot = np.asarray(Parallel(n_jobs=n_jobs)(delayed(_refit_stat)(y_b, x, left_tail) for y_b in ys),
                      dtype=float)
    boot = boot[np.isfinite(boot)]

    if boot.size == 0 or not np.isfinite(tau):
        p, crit = np.nan, np.nan
    elif left_tail:
        p, crit = float((boot <= tau).mean()), float(np.quantile(boot, 0.05))
    else:
        p, crit = float((boot >= tau).mean()), float(np.quantile(boot, 0.95))
    return {"stat": tau, "p_value": p, "crit_5pct": crit, "boot": boot,
            "res_U": res_U, "res_R": res_R}


def classify_cointegration(y, x, *, B: int = 199, seed: int = 0, level: float = 0.05,
                           n_jobs: int = -1) -> dict:
    """Run both bootstrap tests and return the paper's §4.2 verdict.

    Returns ``theta_hat``, ``sigma_eta_hat``, both t-statistics and bootstrap p-values, and a
    ``verdict`` of ``"no cointegration"`` / ``"fixed cointegration"`` /
    ``"time-varying cointegration"`` / ``"undecided"``.
    """
    th = bootstrap_test(y, x, {"theta": 1.0}, B=B, seed=seed, n_jobs=n_jobs)
    sg = bootstrap_test(y, x, {"sigma_eta": 0.0}, B=B, seed=seed + 1, n_jobs=n_jobs)
    reject_theta = np.isfinite(th["p_value"]) and th["p_value"] < level
    reject_sigma = np.isfinite(sg["p_value"]) and sg["p_value"] < level

    if not np.isfinite(th["p_value"]):
        verdict = UNDECIDED
    elif not reject_theta:
        verdict = NO_COINTEGRATION
    elif reject_sigma:
        verdict = TIME_VARYING
    else:
        verdict = FIXED

    p = dict(zip(PARAM_NAMES, th["res_U"].params))
    # A fit that never left its starting value is not an estimate. statsmodels reports it and the
    # gate must read it: on this repository's screen survivors 6% of daily fits return theta
    # exactly 0.9, the optimiser's own start, with a flat likelihood behind them.
    converged = bool(th["res_U"].mle_retvals.get("converged", False))
    return {"theta_hat": float(p["theta"]), "t_theta": th["stat"], "p_theta": th["p_value"],
            "sigma_eta_hat": float(abs(p["sigma_eta"])), "t_sigma": sg["stat"],
            "p_sigma": sg["p_value"], "verdict": verdict, "converged": converged}


@dataclass(frozen=True)
class HedgeVerdict:
    """What the gate concluded, and the evidence for it."""

    hedge: str                  # "dynamic" | "static" | "none"
    verdict: str                # the paper's classification, unmodified
    theta_plausible: bool       # is theta_hat in (0, 1), a decaying non-oscillating error?
    converged: bool             # did the unrestricted optimiser actually converge?
    theta_hat: float
    sigma_eta_hat: float
    p_theta: float
    p_sigma: float
    reason: str

    @property
    def dynamic_warranted(self) -> bool:
        return self.hedge == "dynamic"

    def __str__(self) -> str:                                    # pragma: no cover - display only
        return (f"{self.verdict}: use a {self.hedge} hedge "
                f"(theta={self.theta_hat:.3f}, p={self.p_theta:.3f}; "
                f"sigma_eta={self.sigma_eta_hat:.4f}, p={self.p_sigma:.3f})")


def recommend_hedge(y, x, *, B: int = 199, seed: int = 0, level: float = 0.05,
                    n_jobs: int = -1) -> HedgeVerdict:
    """The gate: should this pair get a dynamic hedge, a static one, or neither?

    A dynamic hedge is warranted only when the coefficient is shown to move — when
    :math:`\\theta = 1` is rejected (there is a long-run relation) *and* :math:`\\sigma_\\eta = 0`
    is rejected (it is not a fixed one). Everything else gets a static hedge or no trade:

    * ``"none"`` — no cointegration. Neither hedge describes a long-run relation, and fitting a
      Kalman filter here is what produces the spurious stationary residual of notebook 07 §1.
    * ``"static"`` — fixed cointegration. The relationship is real and the coefficient does not
      move, so a frozen regression has less estimation noise than a filter chasing nothing. This
      is not merely theoretical: notebook 11 §4 finds the frozen per-fold OLS hedge beats the
      Kalman hedge 0.402 against 0.315 on twenty years, trading 1,698 round trips against 723.
    * ``"dynamic"`` — time-varying cointegration, the only case the Kalman hedge is for.

    ``"undecided"`` also returns ``"static"``, deliberately: the conservative default when the test
    cannot resolve is the hedge with fewer moving parts, not the more flexible one.

    A ``"dynamic"`` verdict is downgraded to ``"static"`` when :math:`\\hat\\theta` falls outside
    ``(0, 1)`` — see ``theta_plausible``. This matters in practice rather than in principle: run
    over this repository's 2,151 screen survivors, 7.3% classify as time-varying but 73% of those
    rest on a negative :math:`\\hat\\theta` and 30% on one outside the stationary region
    altogether, leaving 1.8% standing.
    """
    c = classify_cointegration(y, x, B=B, seed=seed, level=level, n_jobs=n_jobs)
    v = c["verdict"]
    th = c["theta_hat"]
    # The model constrains T to (0, 1) but leaves theta free, so the optimiser can and does wander
    # outside the stationary region: on this repository's own screen survivors, 14% of fits return
    # |theta| > 1, and 30% of the "time-varying" verdicts do. A theta at or beyond 1 describes a
    # non-stationary error, which is the null the test is supposed to reject, so the classification
    # built on it is not interpretable. A negative theta is stationary but alternates sign every
    # bar -- high-frequency oscillation rather than a long-run relation, which is not what a
    # dynamic hedge is for. Either way the paper's verdict is reported unchanged and the *hedge*
    # recommendation is downgraded, because that is the part this repository is responsible for.
    plausible = bool(0.0 < th < 1.0)
    converged = bool(c.get("converged", True))
    if not converged:
        # nothing downstream of a non-converged fit means anything, including the verdict
        return HedgeVerdict(
            hedge="static", verdict=UNDECIDED, theta_plausible=False, converged=False,
            theta_hat=th, sigma_eta_hat=c["sigma_eta_hat"], p_theta=c["p_theta"],
            p_sigma=c["p_sigma"],
            reason="the likelihood optimiser did not converge, so neither parameter is an "
                   "estimate; the conservative default is the hedge with fewer moving parts")
    if v == TIME_VARYING and plausible:
        hedge, reason = "dynamic", ("the coefficient moves: sigma_eta = 0 is rejected, so a filter "
                                    "is tracking something real")
    elif v == TIME_VARYING:
        hedge, reason = "static", (
            f"sigma_eta = 0 is rejected, but theta_hat = {th:.3f} lies outside (0, 1): "
            + ("the error process is non-stationary, so the classification is not interpretable"
               if abs(th) >= 1.0 else
               "the error alternates sign each bar, which is oscillation rather than a long-run "
               "relation") + " -- not a case for a dynamic hedge")
    elif v == FIXED:
        hedge, reason = "static", ("cointegrated but with a fixed coefficient: a frozen regression "
                                   "has less estimation noise than a filter chasing nothing")
    elif v == NO_COINTEGRATION:
        hedge, reason = "none", ("theta = 1 is not rejected, so there is no long-run relation to "
                                 "hedge; a Kalman residual here is manufactured, not measured")
    else:
        hedge, reason = "static", ("the theta test could not be decided; the conservative default "
                                   "is the hedge with fewer moving parts")
    return HedgeVerdict(hedge=hedge, verdict=v, theta_plausible=plausible, converged=converged,
                        theta_hat=c["theta_hat"],
                        sigma_eta_hat=c["sigma_eta_hat"], p_theta=c["p_theta"],
                        p_sigma=c["p_sigma"], reason=reason)


def simulate_tvssm(n: int, theta: float, sigma_eta: float, *, T: float = 0.7,
                   alpha: float = 0.0, sigma_eps: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Generate one pair from the model — the DGP of the paper's Section 4.

    ``x_t`` is a random walk with unit-variance steps, ``sigma_eps = 1``, ``T = 0.7``,
    ``mu = 1/(1 - T)`` and ``beta_0`` at its unconditional mean ``mu/(1 - T)`` (about 11), with
    ``w_0 = 0``. These values are not arbitrary: the size and power figures in the paper's Table 1,
    which notebook 07 §4 reproduces a slice of, are all measured on this DGP, so changing them
    changes what "correctly classified" means.

    ``theta=1, sigma_eta=0`` is no cointegration; ``theta<1, sigma_eta=0`` fixed; ``theta<1,
    sigma_eta>0`` time-varying.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    mu = 1.0 / (1.0 - T)
    beta = np.empty(n)
    w = np.empty(n)
    b, wv = mu / (1.0 - T), 0.0
    for t in range(n):
        b = mu + T * b + sigma_eta * rng.standard_normal()
        wv = theta * wv + sigma_eps * rng.standard_normal()
        beta[t], w[t] = b, wv
    return pd.DataFrame({"y": alpha + x * beta + w, "x": x, "beta_true": beta, "w_true": w})
