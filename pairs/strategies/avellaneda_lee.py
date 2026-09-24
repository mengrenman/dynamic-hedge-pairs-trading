# pairs/strategies/avellaneda_lee.py
"""Avellaneda & Lee (2010), *Statistical arbitrage in the US equities market*, Quant. Finance
10(7) 761-782 -- the ETF-factor construction, in pure functions over windows the caller supplies.

No I/O, no global state, no cadence logic (rebuild schedules, universe screens, caching) -- all of
that belongs to the notebook that calls this module. Every function operates on a single window or
a single cross-section and returns arrays/Series the caller assembles into a time series.

**The model (eq. 11-12).** For stock :math:`i` and its assigned sector ETF :math:`I`,

.. math::
    \\frac{\\mathrm dS_i}{S_i} = \\alpha_i\\,\\mathrm dt + \\beta_i\\,\\frac{\\mathrm dI}{I} + \\mathrm dX_i,
    \\qquad
    \\mathrm dX_i = \\kappa_i (m_i - X_i)\\,\\mathrm dt + \\sigma_i\\,\\mathrm dW_i,\\quad \\kappa_i > 0.

**Estimation (Appendix A).** Regress the stock's daily returns on its ETF's daily returns over the
trailing 60 sessions *with an intercept*; :math:`X_k = \\sum_{j \\le k} \\epsilon_j` for
:math:`k = 1,\\dots,60` is the cumulative residual (an OLS intercept forces
:math:`\\sum\\epsilon_j = 0`, so :math:`X_{60} = 0` by construction -- eq. A1's own observation, not
a bug); fit the AR(1) :math:`X_{n+1} = a + bX_n + \\zeta_{n+1}` for :math:`n = 1,\\dots,59`; then

.. math::
    \\kappa = -\\log(b)\\times 252, \\quad m = \\frac{a}{1-b}, \\quad
    \\sigma_{\\mathrm{eq}} = \\sqrt{\\frac{\\mathrm{Var}(\\zeta)}{1-b^2}}, \\quad
    s = \\frac{X_{60}-m}{\\sigma_{\\mathrm{eq}}}.

That last identity is only :math:`-m/\\sigma_{\\mathrm{eq}}` when :math:`X_{60} = 0`, i.e. for the
OLS-with-intercept residual above; :func:`s_score` computes the general form from :math:`X_{60}`
(``ou_fit``'s ``x_last``), not the shortcut, because a residual from a regression whose intercept
column is scaled by the trading-time factor (:func:`etf_residuals` with ``weights`` and its scaled
intercept, see that function's docstring) need not sum to zero. The reading adopted downstream for
the paper's own "trading time" (its section 6) is different and keeps :math:`X_{60}=0` even under
volume weighting: multiply both sides of the regression by the trading-time factor :math:`f_t` and
fit an *unscaled* intercept, i.e. call :func:`etf_residuals` with ``stock_win, etf_win`` already
scaled by :math:`f_t` and ``weights=None`` -- an ordinary OLS-with-intercept fit of the modified
returns, whose residuals sum to zero by the same OLS identity as the calendar-clock case. Only the
scaled-intercept WLS reading has a nonzero :math:`X_{60}` in general.

**Centered means (eq. 18 / A2).** :math:`\\bar m = m - \\langle m\\rangle`, the cross-sectional mean
over the names estimated that day, then :math:`s = (X_{60} - \\bar m)/\\sigma_{\\mathrm{eq}}`, which
reduces to :math:`-\\bar m/\\sigma_{\\mathrm{eq}}` only when :math:`X_{60}=0` (OLS with intercept).
The paper reports centering "consistently" helped and centers everywhere; this module defaults to it.

**Model rejection.** :math:`\\kappa > 252/30` (mean-reversion faster than 30 days, i.e. :math:`b <
0.9672`) and :math:`b \\in (0,1)`; a name failing this opens no new trade and closes any open one.

**Trading rule (eq. 16), bang-bang.** buy-to-open if :math:`s < -1.25`; sell-to-open if
:math:`s > +1.25`; close a long if :math:`s > -0.50`; close a short if :math:`s < +0.75`. Full size
on open, held until the close condition fires -- no partial sizing on the way in or out.

**Trading time (section 6, eq. 20).** :math:`\\bar R_t = R_t \\times \\langle\\delta V\\rangle /
V_t`, with :math:`\\langle\\delta V\\rangle` the trailing 10-session average daily volume. The paper
folds this into the *residual estimation*, not into a separate return series downstream; this
module follows that reading (see :func:`etf_residuals`).

Interpretive decisions are called out in each function's docstring; the top-level summary lives in
the module that built this file (search history / HANDOFF.md), not here, so this file stays a pure
reference to the paper.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "assign_sector_etf",
    "trading_time_factor",
    "etf_residuals",
    "ou_fit",
    "s_score",
    "bang_bang_update",
    "positions_from_state",
]


# ── 1. sector assignment ──────────────────────────────────────────────────────

def assign_sector_etf(stock_ret: pd.DataFrame, etf_ret: pd.DataFrame, *,
                       min_obs: int = 200) -> pd.DataFrame:
    """Assign each name to the sector ETF whose daily returns explain it best.

    The paper hand-assigns each of its ~1,400 names to one of the 15 sector ETFs by GICS industry
    (section 5.1). No sector classification exists in this repository (``pairs/universes/tickers/*``
    is empty), so the assignment here is data-driven and point-in-time instead: for each name, pick
    the candidate ETF with the highest :math:`R^2` of daily returns over the window the caller
    supplies (intended to be the trailing 252 sessions), using simple linear regression with an
    intercept -- for one regressor, :math:`R^2` is exactly the squared correlation, which is how
    this is computed. This is the interpretive substitute for the paper's fixed classification and
    is expected to drift a name between ETFs across refits, which the caller controls by choosing
    how often to call this (the spec this module was built against rebuilds monthly).

    Parameters
    ----------
    stock_ret : DataFrame, T x N
        Daily returns, one column per name.
    etf_ret : DataFrame, T x K
        Daily returns, one column per candidate ETF. Reindexed to ``stock_ret``'s row index.
    min_obs : int
        An ETF is only a candidate for a name if they have at least this many overlapping
        non-missing sessions; an ETF with fewer is treated as untested for that name, not as a
        zero-:math:`R^2` candidate.

    Returns
    -------
    DataFrame indexed by name, columns ``['etf', 'r2']``. Both are NaN (``etf`` is ``None``) for a
    name with fewer than ``min_obs`` overlapping sessions against *every* candidate ETF, or where
    every candidate that did clear ``min_obs`` produced a non-finite :math:`R^2` (e.g. constant
    returns on one side).
    """
    etf_ret = etf_ret.reindex(stock_ret.index)
    names = stock_ret.columns
    best_etf = pd.Series([None] * len(names), index=names, dtype=object)
    best_r2 = pd.Series(np.nan, index=names, dtype=float)

    stock_ok = stock_ret.notna()
    for e in etf_ret.columns:
        col = etf_ret[e]
        overlap = stock_ok.mul(col.notna(), axis=0).sum(axis=0)
        r2 = stock_ret.corrwith(col) ** 2
        r2 = r2.where(overlap >= min_obs)
        better = r2.notna() & (best_r2.isna() | (r2 > best_r2))
        best_etf = best_etf.mask(better, e)
        best_r2 = best_r2.mask(better, r2)

    return pd.DataFrame({"etf": best_etf, "r2": best_r2}, index=names)


# ── 2. trading time ────────────────────────────────────────────────────────────

def trading_time_factor(volume: pd.DataFrame, *, window: int = 10,
                         clip: tuple = (0.1, 10.0)) -> pd.DataFrame:
    """The trading-time scaling factor :math:`\\langle\\delta V\\rangle / V_t` of eq. 20.

    :math:`\\langle\\delta V\\rangle` is the mean daily volume over the *previous* ``window``
    sessions (:math:`t-\\text{window}, \\dots, t-1` -- exclusive of :math:`t`, so the factor at
    :math:`t` uses no information from :math:`t` or later). NaN for the first ``window`` rows (no
    full look-back yet) and wherever :math:`V_t` itself is missing or zero.

    **Deviation from the paper.** Avellaneda-Lee do not discuss zero- or near-zero-volume days,
    which would send the factor to infinity; this implementation clips the ratio to ``clip``
    (default 0.1-10x) so a single illiquid session cannot dominate a name's regression weights
    downstream. That clip has no basis in the paper and is this module's own guard.

    Parameters
    ----------
    volume : DataFrame, T x N
        Daily share volume, one column per name.
    window : int
        Trailing look-back for :math:`\\langle\\delta V\\rangle` (paper: 10 sessions, section 6).
    clip : (float, float)
        Bounds applied to the ratio after computation; NaN entries stay NaN.

    Returns
    -------
    DataFrame, T x N, aligned to ``volume``.
    """
    avg_prev = volume.shift(1).rolling(window, min_periods=window).mean()
    factor = avg_prev / volume
    factor = factor.where(volume.notna() & (volume != 0))
    lo, hi = clip
    return factor.clip(lower=lo, upper=hi)


# ── 3. residuals ───────────────────────────────────────────────────────────────

def etf_residuals(stock_win: np.ndarray, etf_win: np.ndarray, *,
                   weights: Optional[np.ndarray] = None) -> dict:
    """Per-name OLS of a stock's returns on its assigned ETF's returns (eq. 11's regression step).

    ``stock_win`` and ``etf_win`` are both W x N: column :math:`i` of ``etf_win`` is the return
    series of the ETF *already assigned* to name :math:`i` (the caller builds this by looking up
    each name's ETF and gathering that ETF's column -- this function does not know about sector
    assignment). Each column is regressed independently: :math:`y = \\alpha + \\beta x + \\epsilon`.

    **Trading time -- two readings, one adopted downstream.** The paper's eq. 20 replaces calendar
    returns with volume-modified returns :math:`\\bar R_t = R_t \\times \\langle\\delta
    V\\rangle/V_t` "in the residual estimation" but does not spell out the mechanics for a
    two-sided regression. The reading the calling notebook actually uses (paper-faithful) is to
    scale ``stock_win``/``etf_win`` by the trading-time factor *before* calling this function and
    fit with ``weights=None`` -- an ordinary OLS-with-intercept regression of the modified returns,
    :math:`f_ty_t = \\alpha + \\beta(f_tx_t) + \\epsilon_t`, whose residuals sum to zero by the same
    identity as the calendar-clock fit (Appendix A2), exactly as the paper's own remark that
    modified returns "are equal to the classical returns if the daily trading volume is typical"
    implies: a mild reweighting of the same intercept regression, not a different estimator.

    This function's own ``weights`` parameter instead implements a second, *rejected* reading,
    kept here only because it was this module's original interpretation and is exercised by
    :mod:`tests.test_avellaneda_lee`: the modification is applied *consistently* to both sides of
    each name's regression **and** to the intercept column itself, so with ``weights`` :math:`f`
    (typically :func:`trading_time_factor`'s output for that window), both :math:`y_t` and
    :math:`x_t` are scaled by :math:`f_t` before the fit *and* the constant column becomes
    :math:`f_t` rather than 1, i.e. the model actually estimated is

    .. math::
        f_t y_t = \\alpha f_t + \\beta (f_t x_t) + f_t\\epsilon_t,

    which is an ordinary regression of :math:`f y` on :math:`[f,\\, fx]` (the intercept column
    becomes :math:`f_t` rather than 1) and is algebraically identical to a weighted least squares
    fit of the *unscaled* model with weight :math:`f_t^2` -- :math:`\\alpha,\\beta` come out the
    same either way. What is returned as ``resid`` is the residual of that scaled regression,
    :math:`f_t(y_t - \\alpha - \\beta x_t)`, i.e. the modified-return residual that feeds
    :func:`ou_fit`'s cumulative sum, not the plain regression residual (though when ``weights`` is
    ``None``, or an array of ones, the two coincide). Because the constant column is :math:`f_t`
    rather than 1, this regression's normal equations force :math:`\\sum f_t\\epsilon_t = 0`, not
    :math:`\\sum\\epsilon_t = 0`: the cumulative residual at the end of the window need not vanish,
    and on real data is typically of order one :math:`\\sigma_{\\mathrm{eq}}` -- this is what made
    the un-general (Appendix A2 shortcut) form of :func:`s_score` silently wrong for this reading
    (the original form of the bug this module was rewritten to fix), and is also why this reading
    is not the one downstream notebooks use for "trading time".

    Parameters
    ----------
    stock_win : ndarray, W x N
    etf_win : ndarray, W x N
        Column :math:`i` is name :math:`i`'s assigned ETF's return over the same W sessions.
    weights : ndarray, W x N, or None
        The trading-time factor per name per session; ``None`` is plain OLS.

    Returns
    -------
    dict with keys ``alpha`` (N,), ``beta`` (N,), ``resid`` (W x N), ``r2`` (N,). ``r2`` is
    :math:`1 - \\mathrm{Var}(\\text{resid})/\\mathrm{Var}(y_{\\text{used}})`, where
    :math:`y_{\\text{used}}` is the actual left-hand side fit (:math:`fy` when weighted, :math:`y`
    otherwise) -- population variance, matching notebook 12's ``factor_day``. NaN where
    :math:`\\mathrm{Var}(y_{\\text{used}}) \\le 10^{-18}` (the same absolute floor used to gate
    ``beta`` on ``sxx``, so a name whose regressor or regressand is degenerate never produces a
    numeric ``r2``): a fixed additive epsilon in the denominator was tried and rejected, because for
    a stale/illiquid name whose return variance lands within a few orders of magnitude of that
    epsilon (e.g. 59 of 60 daily returns exactly zero, one tiny non-zero tick), it inflates ``r2``
    by an order of magnitude or more relative to the same formula without the fudge -- a diagnostic
    silently reading "strong relationship" for a name that has essentially none.
    """
    stock_win = np.asarray(stock_win, dtype=float)
    etf_win = np.asarray(etf_win, dtype=float)

    f = np.ones_like(stock_win) if weights is None else np.asarray(weights, dtype=float)
    w = f ** 2

    ws = w.sum(axis=0)
    xw = (w * etf_win).sum(axis=0) / ws
    yw = (w * stock_win).sum(axis=0) / ws
    xc = etf_win - xw
    yc = stock_win - yw
    sxx = (w * xc * xc).sum(axis=0)
    sxy = (w * xc * yc).sum(axis=0)

    beta = np.divide(sxy, sxx, out=np.full(sxx.shape, np.nan), where=sxx > 1e-18)
    alpha = yw - beta * xw
    eps = stock_win - alpha - beta * etf_win                 # residual of the unscaled model
    resid = eps if weights is None else f * eps              # residual of the fitted (scaled) model

    y_used = stock_win if weights is None else f * stock_win
    var_y = y_used.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(var_y > 1e-18, 1.0 - resid.var(axis=0) / var_y, np.nan)

    return dict(alpha=alpha, beta=beta, resid=resid, r2=r2)


# ── 4. the OU fit ──────────────────────────────────────────────────────────────

def ou_fit(resid: np.ndarray) -> dict:
    """Fit the Ornstein-Uhlenbeck residual model of Appendix A, per column of ``resid``.

    :math:`X_k = \\sum_{j\\le k}\\epsilon_j` (``resid``'s cumulative sum down each column); the
    AR(1) :math:`X_{n+1} = a + bX_n + \\zeta_{n+1}` is fit by OLS over the :math:`W-1` consecutive
    pairs; then, exactly per eq. A1,

    .. math::
        \\kappa = -\\log(b)\\times 252,\\quad m = \\frac{a}{1-b},\\quad
        \\sigma_{\\mathrm{eq}} = \\sqrt{\\frac{\\mathrm{Var}(\\zeta)}{1-b^2}},\\quad
        \\sigma = \\sqrt{\\frac{2\\kappa\\,\\mathrm{Var}(\\zeta)}{1-b^2}},\\quad
        s_{\\mathrm{raw}} = \\frac{X_W - m}{\\sigma_{\\mathrm{eq}}}.

    :math:`\\mathrm{Var}(\\zeta)` uses the same estimator as notebook 12's ``s_score_al``:
    :math:`\\sum\\zeta^2 / \\max(W-3,\\,1)` (two fitted parameters, :math:`W-1` observations). When
    ``resid`` comes from an intercept regression (:func:`etf_residuals` with ``weights=None``),
    :math:`X_W = \\sum\\epsilon_j = 0` exactly, so :math:`s_{\\mathrm{raw}} = -m/\\sigma_{\\mathrm{eq}}`
    -- this is eq. A1's own remark, reproduced here as a test, not an assumption this function
    makes; with a weighted residual the sum need not vanish and :math:`s_{\\mathrm{raw}}` is
    computed from the actual :math:`X_W`.

    Every array is NaN wherever :math:`b \\le 0`, :math:`b \\ge 1`, or the fit is otherwise
    degenerate (near-zero variance in :math:`X_{1:W-1}`, or a non-finite downstream quantity) --
    including ``a`` and ``b`` themselves, so a caller never sees a coefficient from a fit this
    function has flagged as unusable.

    Parameters
    ----------
    resid : ndarray, W x N
        Residual returns for the estimation window (typically 60 sessions per the paper).

    Returns
    -------
    dict of ndarrays, each shape (N,): ``a``, ``b``, ``kappa``, ``m``, ``sigma``, ``sigma_eq``,
    ``s_raw``, ``x_last`` (:math:`X_W`, the cumulative residual at the end of the window --
    :func:`s_score` needs this to compute the general form of the s-score rather than the
    OLS-with-intercept shortcut), plus ``model_ok`` (bool): ``kappa`` finite and
    ``kappa > 252/30`` (eq. "fast mean-reversion" gate, section 3 / Appendix A).
    """
    resid = np.asarray(resid, dtype=float)
    x = np.cumsum(resid, axis=0)
    x0, x1 = x[:-1], x[1:]
    n = x0.shape[0]

    mx0, mx1 = x0.mean(axis=0), x1.mean(axis=0)
    cov = ((x0 - mx0) * (x1 - mx1)).sum(axis=0)
    var = ((x0 - mx0) ** 2).sum(axis=0)
    b = np.divide(cov, var, out=np.full(var.shape, np.nan), where=var > 1e-18)
    a = mx1 - b * mx0

    zeta = x1 - (a + b * x0)
    dof = max(n - 2, 1)
    var_z = (zeta ** 2).sum(axis=0) / dof

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = -np.log(np.clip(b, 1e-9, 0.999999)) * 252.0
        m = a / (1.0 - b)
        denom = 1.0 - b ** 2
        sigma_eq = np.sqrt(var_z / denom)
        sigma = np.sqrt(var_z * 2.0 * kappa / denom)
        s_raw = (x[-1] - m) / sigma_eq

    bad = ((b <= 0) | (b >= 1) | ~np.isfinite(kappa) | ~np.isfinite(m)
           | ~np.isfinite(sigma_eq) | (sigma_eq <= 1e-12) | ~np.isfinite(s_raw))

    out = {}
    for key, arr in (("a", a), ("b", b), ("kappa", kappa), ("m", m),
                      ("sigma", sigma), ("sigma_eq", sigma_eq), ("s_raw", s_raw),
                      ("x_last", x[-1])):
        out[key] = np.where(bad, np.nan, arr)
    out["model_ok"] = np.isfinite(out["kappa"]) & (out["kappa"] > 252.0 / 30.0)
    return out


# ── 5. the s-score ─────────────────────────────────────────────────────────────

def s_score(fit: dict, *, center: bool = True) -> np.ndarray:
    """The s-score of eq. 15, optionally cross-sectionally centered per eq. 18 / A2.

    .. math::
        s = \\frac{X_W - \\bar m}{\\sigma_{\\mathrm{eq}}}, \\qquad
        \\bar m = \\begin{cases} m - \\langle m\\rangle & \\text{center=True} \\\\ m &
        \\text{center=False} \\end{cases}

    with :math:`X_W` = ``fit['x_last']`` (:func:`ou_fit`'s cumulative residual at the end of the
    window) and :math:`\\langle m\\rangle` the cross-sectional mean of ``fit['m']`` over the names
    present (``np.nanmean``, so names already NaN from :func:`ou_fit` neither shift the mean nor
    themselves become non-NaN).

    This reduces to eq. A2's shortcut, :math:`s = -\\bar m/\\sigma_{\\mathrm{eq}}`, **only** when
    :math:`X_W = 0`, which holds for an OLS-with-intercept residual (:func:`etf_residuals` with
    ``weights=None``) but not in general: a weighted (trading-time) regression's residuals need not
    sum to zero, and on real data the median :math:`|X_W|/\\sigma_{\\mathrm{eq}}` is about 0.97, so
    silently dropping the :math:`X_W` term there discards where the residual actually sits rather
    than reproducing a valid simplification. Accordingly ``fit`` must carry ``x_last``; there is no
    silent default of 0 for a missing key, because that default is exactly the bug this function
    used to have.

    Parameters
    ----------
    fit : dict
        A dict as returned by :func:`ou_fit` (needs ``m``, ``sigma_eq``, and ``x_last``).
    center : bool

    Returns
    -------
    ndarray, shape (N,). NaN wherever ``fit['m']``, ``fit['sigma_eq']`` or ``fit['x_last']`` is NaN.

    Raises
    ------
    KeyError
        If ``fit`` has no ``x_last`` entry.
    """
    m = np.asarray(fit["m"], dtype=float)
    sigma_eq = np.asarray(fit["sigma_eq"], dtype=float)
    x_last = np.asarray(fit["x_last"], dtype=float)
    m_bar = m - np.nanmean(m) if center else m
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x_last - m_bar) / sigma_eq


# ── 6. the bang-bang trading rule ──────────────────────────────────────────────

def bang_bang_update(state: np.ndarray, s: np.ndarray, model_ok: np.ndarray, *,
                      open: float = 1.25, close_long: float = -0.5,
                      close_short: float = 0.75) -> np.ndarray:
    """One step of the bang-bang trading rule (eq. 16): full size on open, hold to the close signal.

    Applied in this priority order, per name:

    1. ``not model_ok`` or ``s`` is NaN -> flat (0), whatever the incoming state -- the paper closes
       any open trade in a name whose mean-reversion no longer clears the :math:`\\kappa > 252/30`
       gate, it does not hold it open on stale evidence.
    2. flat and :math:`s < -\\text{open}` -> long (+1, "buy to open": long $1 stock, short
       :math:`\\beta` dollars of its ETF).
    3. flat and :math:`s > +\\text{open}` -> short (-1, "sell to open").
    4. long and :math:`s > \\text{close\\_long}` -> flat.
    5. short and :math:`s < \\text{close\\_short}` -> flat.
    6. otherwise unchanged.

    Rules 2-5 only fire where rule 1 does not, so a name can go from open to flat in one call but
    never directly from +1 to -1 (or -1 to +1): closing and re-opening in one step is not how this
    function is wired, even though the (state, s) combination could in principle satisfy both a
    close and an open condition at once.

    Parameters
    ----------
    state : ndarray, (N,), values in {-1, 0, +1}
        Yesterday's position.
    s : ndarray, (N,)
        Today's s-score (:func:`s_score`'s output).
    model_ok : ndarray, (N,), bool
        Today's mean-reversion gate (:func:`ou_fit`'s ``model_ok``).
    open, close_long, close_short : float
        The paper's cutoffs (1.25, -0.50, +0.75).

    Returns
    -------
    ndarray, (N,), the next state, same dtype as ``state`` -- ``state`` must have a signed dtype
    (or be plain Python ints, which ``np.asarray`` makes ``int64``): the state alphabet is
    ``{-1, 0, +1}`` and an unsigned dtype cannot represent ``-1`` at all, so passing one raises
    ``ValueError`` rather than silently wrapping a short position into a huge positive integer
    (e.g. 255 on ``uint8``) that :func:`positions_from_state` would then multiply straight into
    dollar notional.

    Raises
    ------
    ValueError
        If ``state``'s dtype is unsigned.
    """
    state_arr = np.asarray(state)
    if np.issubdtype(state_arr.dtype, np.unsignedinteger):
        raise ValueError(
            f"bang_bang_update requires a signed dtype for `state` (got {state_arr.dtype}); "
            "an unsigned dtype cannot represent the -1 (short) state and would silently wrap."
        )
    s = np.asarray(s, dtype=float)
    model_ok = np.asarray(model_ok, dtype=bool)

    ok = model_ok & np.isfinite(s)
    new = state_arr.astype(float).copy()

    open_long = ok & (state_arr == 0) & (s < -open)
    open_short = ok & (state_arr == 0) & (s > open)
    close_long_mask = ok & (state_arr == 1) & (s > close_long)
    close_short_mask = ok & (state_arr == -1) & (s < close_short)

    new = np.where(open_long, 1.0, new)
    new = np.where(open_short, -1.0, new)
    new = np.where(close_long_mask, 0.0, new)
    new = np.where(close_short_mask, 0.0, new)
    new = np.where(~ok, 0.0, new)

    return new.astype(state_arr.dtype)


# ── 7. positions ───────────────────────────────────────────────────────────────

def positions_from_state(state: pd.Series, beta: pd.Series, etf_of: pd.Series, *,
                          notional: float) -> pd.Series:
    """Dollar positions for a cross-section of states: stock legs plus netted ETF hedge legs.

    A "buy" is long $1 of stock and short :math:`\\beta` dollars of its ETF (eq. 16's description
    of a trade); scaled here to ``notional`` dollars of stock per open name. Every open name that
    shares an ETF nets into a single position in that ETF, since the strategy is long/short many
    names against the same handful of sector ETFs simultaneously.

    .. math::
        \\text{stock}_i = \\text{state}_i \\times \\text{notional}, \\qquad
        \\text{etf}_e = -\\sum_{i:\\,\\text{etf\\_of}_i = e} \\text{state}_i\\,\\beta_i\\,\\text{notional}.

    A name with NaN ``beta`` or NaN ``etf_of`` contributes to neither leg (no stock position and no
    hedge contribution) -- there is no way to size or hedge a position without both, so it is
    treated as absent from the book, not as an unhedged stock position.

    Parameters
    ----------
    state : Series, name -> {-1, 0, +1}
    beta : Series, name -> float
        The name's regression beta against its assigned ETF (:func:`etf_residuals`'s ``alpha``/
        ``beta`` output, keyed back to names by the caller).
    etf_of : Series, name -> str
        The name's assigned ETF ticker (:func:`assign_sector_etf`'s ``etf`` column).
    notional : float
        Dollars of stock per fully-open position.

    Returns
    -------
    Series indexed by instrument (stock tickers, then ETF tickers), dollar position. A name with
    ``state == 0`` and valid beta/etf still appears, at $0.
    """
    idx = state.index
    beta = beta.reindex(idx)
    etf_of = etf_of.reindex(idx)
    valid = state.notna() & beta.notna() & etf_of.notna()

    s = state[valid].astype(float)
    b = beta[valid].astype(float)
    e = etf_of[valid]

    stock_legs = s * notional
    etf_legs = -(s * b * notional).groupby(e).sum()

    return pd.concat([stock_legs, etf_legs])
