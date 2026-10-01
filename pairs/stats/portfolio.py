# pairs/stats/portfolio.py
"""
Portfolio-level analytics for multi-pair trading strategies.

Exports:
- pair_return_correlations(kf_results, ...) : N×N cross-pair spread-return correlation matrix
- portfolio_diversification_score(corr_matrix) : scalar diversification ratio
- suggest_position_weights(kf_results, prices=..., ...) : capital allocation weights per pair,
  weighted by the spread's risk per dollar of gross notional and held to a cap that holds
  (fixed 2026-10-01; ``risk="price_units"`` and ``cap="one_pass"`` keep the earlier behavior)
- spread_returns(P1, P2, beta) : the spread's return per dollar of gross notional
- allocation_weights(method, ...) : equal / inverse-volatility / equal-risk-contribution /
  shared-leg splits computed from numbers (sigma, a covariance matrix, a list of pairs)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "pair_return_correlations",
    "portfolio_diversification_score",
    "suggest_position_weights",
    "spread_returns",
    "allocation_weights",
    "ERCConvergenceError",
]


def pair_return_correlations(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    *,
    min_overlap: int = 30,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlations of spread daily returns (Δresid) across pairs.

    Parameters
    ----------
    kf_results : dict mapping (ticker1, ticker2) → DataFrame with a 'resid' column
    min_overlap : minimum overlapping bars required; pairs with fewer overlap get NaN
    method : 'pearson' (default) or 'spearman'

    Returns
    -------
    Symmetric N×N DataFrame, index and columns labeled "T1/T2".
    Diagonal entries are 1.0 (or NaN if a pair's series is all NaN).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")

    # Build spread-return series dict: label → pd.Series of Δresid
    returns: dict[str, pd.Series] = {}
    for (k1, k2), df in kf_results.items():
        label = f"{k1}/{k2}"
        if "resid" not in df.columns:
            continue
        ret = df["resid"].diff().dropna()
        if not ret.empty:
            returns[label] = ret

    if len(returns) == 0:
        return pd.DataFrame()

    labels = list(returns.keys())

    # Align all series on a common DatetimeIndex
    combined = pd.concat(returns, axis=1)
    combined.columns = labels

    # Compute correlation matrix
    corr = combined.corr(method=method)

    # NaN-out off-diagonal entries whose pairwise overlap is below min_overlap.
    # overlap[i, j] = number of bars where both series are non-NaN, computed
    # with one matrix product instead of an O(N^2) loop of dropna() calls.
    present = combined.notna().to_numpy(dtype=np.int64)
    overlap = present.T @ present
    too_few = (overlap < min_overlap) & ~np.eye(len(labels), dtype=bool)
    corr = corr.mask(too_few)

    return corr


def portfolio_diversification_score(
    corr_matrix: pd.DataFrame,
) -> float:
    """
    Portfolio diversification ratio.

    Defined as 1 / mean(|off-diagonal correlations|).

    Range: 1 (all perfectly correlated) → ∞ (all uncorrelated).
    A score > 3 indicates good diversification across pairs.
    Returns np.nan if fewer than 2 pairs (no off-diagonal entries exist).

    Parameters
    ----------
    corr_matrix : square symmetric correlation DataFrame (output of pair_return_correlations)
    """
    n = len(corr_matrix)
    if n < 2:
        return float("nan")

    # Extract off-diagonal elements only
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr_matrix.values[mask]

    # Drop NaNs from incomplete pair overlaps
    off_diag = off_diag[~np.isnan(off_diag)]

    if len(off_diag) == 0:
        return float("nan")

    mean_abs_corr = float(np.mean(np.abs(off_diag)))
    if mean_abs_corr == 0.0:
        return float("inf")

    return 1.0 / mean_abs_corr


_WEIGHT_COLUMNS = ["pair", "resid_var", "var_per_dollar", "inv_var_weight", "weight", "suggested_capital_pct"]
_CAP_TOL = 1e-12          # a weight within this of the cap counts as at the cap


def _per_dollar_returns(px: pd.DataFrame) -> pd.Series:
    """
    Per-dollar spread returns from complete rows of ``P1``, ``P2`` and ``beta``.

    ``r_t = (dP1_t - beta_{t-1} dP2_t) / (P1_{t-1} + |beta_{t-1}| P2_{t-1})``: the hedge ratio that
    prices day ``t`` is the one held at the previous close, so a hedge re-estimated on day ``t``
    does not enter day ``t``'s return. ``px`` must already be free of missing values and have
    positive prices; the first row is dropped. This is the one place the formula is written, shared
    by :func:`spread_returns` (a constant ``beta`` column) and :func:`suggest_position_weights`
    (the frame's own time-varying ``beta`` column).
    """
    d = px[["P1", "P2"]].diff().iloc[1:]
    prev = px.shift(1).iloc[1:]
    b = prev["beta"]
    r = (d["P1"] - b * d["P2"]) / (prev["P1"] + b.abs() * prev["P2"])
    r.name = "spread_return"
    return r


def _variance_per_dollar(df: pd.DataFrame, p1: pd.Series, p2: pd.Series, label: str) -> float:
    """Variance (ddof=1) of a pair's per-dollar spread return on the rows where its frame and both
    prices are present; NaN when fewer than two returns exist.

    Raises ValueError when ``prices`` cannot be lined up with the frame at all: the frame has rows
    but ``prices`` shares fewer than ``min(3, len(df))`` of its dates (a timezone-aware index against
    a naive one, a shifted calendar and a RangeIndex all give zero). That is a caller bug, and
    answering it with a NaN variance and a zero weight would look like a pair with no usable risk.
    Missing closes on dates the two indexes do share are not an error: they leave fewer returns, and
    a pair with fewer than two gets NaN.
    """
    if "beta" not in df.columns:
        raise ValueError(
            f"frame for {label} has no 'beta' column, which risk='per_dollar' needs; "
            "pass the Kalman state frame or use risk='price_units'")
    shared = int(df.index.isin(p1.index).sum())
    if shared < min(3, len(df)):
        raise ValueError(
            f"prices shares {shared} of the {len(df)} dates of the frame for {label}; "
            "it must be indexed like the frames (the same dates, with the same timezone or none)")
    px = pd.DataFrame({"P1": p1.reindex(df.index), "P2": p2.reindex(df.index), "beta": df["beta"]})
    px = px.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 3:
        return float("nan")
    if not (px[["P1", "P2"]] > 0).all().all():
        raise ValueError(f"prices for {label} must be positive")
    r = _per_dollar_returns(px)
    return float(r.var(ddof=1))


def _capped_weights(w: np.ndarray, max_weight: float) -> np.ndarray:
    """
    Cap weights by water-filling: clip what exceeds the cap, scale the weights still under it to
    share the remaining mass, and repeat until none exceeds the cap (tolerance ``_CAP_TOL``).

    ``w`` is non-negative and sums to one (or is all zeros). The effective cap is
    ``max(max_weight, 1/n)`` with ``n`` the number of positive weights, so a cap that cannot hold
    (``n * max_weight < 1``) gives equal weights over those pairs. Zero weights stay zero, and the
    weights that were never clipped keep their ratios to one another.
    """
    base = np.asarray(w, dtype=float)
    pos = base > 0.0
    n = int(pos.sum())
    if n == 0:
        return base.copy()
    cap = max(float(max_weight), 1.0 / n)
    out = base.copy()
    capped = np.zeros(len(base), dtype=bool)
    for _ in range(n + 1):
        over = pos & ~capped & (out > cap + _CAP_TOL)
        if not over.any():
            break
        capped |= over
        free = pos & ~capped
        out[capped] = cap
        # `free` is never empty here: n weights summing to 1 cannot all exceed a cap of at least 1/n
        out[free] = base[free] * (1.0 - cap * capped.sum()) / base[free].sum()
    return out


def suggest_position_weights(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    corr_matrix: Optional[pd.DataFrame] = None,
    *,
    method: str = "inv_var",
    max_weight: float = 0.40,
    prices: Optional[pd.DataFrame] = None,
    risk: str = "per_dollar",
    cap: str = "iterative",
) -> pd.DataFrame:
    """
    Suggest capital allocation weights across pairs.

    By default a pair's weight is proportional to the inverse variance of its spread's return
    *per dollar of gross notional* and no weight exceeds ``max_weight``.

    Changed 2026-10-01. Notebook 22 scored this helper as challenger A4 and documented two
    defects, both fixed here: (1) it weighted by the variance of the first difference of the
    residual in *price units*, which is not a risk per dollar, so a $500 stock pair was
    down-weighted against a $20 one whatever its percentage volatility; (2) it clipped at
    ``max_weight`` and renormalized once, so the cap was not a cap (an average effective 2.8 pairs
    on notebook 11's book, and 1.00 in one formation of 20 pairs). ``risk="price_units"`` and
    ``cap="one_pass"`` reproduce the earlier weights exactly, pair by pair (every ``resid_var``,
    ``inv_var_weight`` and ``weight`` is the same number for the same label); notebook 22's A4
    calls them and reads the weights by label. The one difference is the row order among tied
    weights: they are now listed in input order, where the earlier unstable sort scrambled them
    once there were more than 16 pairs.

    Parameters
    ----------
    kf_results : dict mapping (ticker1, ticker2) → state DataFrame. ``risk="per_dollar"`` needs
        the ``beta`` column (the Kalman frame's own, time-varying hedge ratio); ``risk="price_units"``
        needs ``resid``. A pair with no usable variance gets weight 0.
    corr_matrix : accepted only for backward compatibility. It was never used and still is not.
    method : 'inv_var' (default) — weight ∝ 1/variance of the spread return;
        'inv_vol' — weight ∝ 1/standard deviation; or 'equal' — equal weights over the pairs that
        have a usable variance. Under 'equal', ``prices`` is still needed with
        ``risk="per_dollar"``, because that is what decides which pairs are usable.
    max_weight : largest weight for any one pair (default 0.40), in (0, 1].
    prices : wide DataFrame of closes, indexed like the frames, one column per ticker (the shape
        the notebooks hold). Needed when ``risk="per_dollar"``; a frame state has no prices.
    risk : 'per_dollar' (default) — the variance of
        ``r_t = (dP1_t − β_{t−1} dP2_t) / (P1_{t−1} + |β_{t−1}| P2_{t−1})``, with the frame's own
        ``beta`` lagged one row, over the rows where the frame and both prices are present and
        differenced across them as :func:`spread_returns` does, sample variance (ddof=1);
        or 'price_units' — the variance of the first difference of ``resid``, as shipped before
        2026-10-01.
    cap : 'iterative' (default) — a true cap by water-filling: clip at the cap, share the
        remaining mass among the weights still under it in proportion, repeat until none exceeds
        it (tolerance 1e-12). The effective cap is ``max(max_weight, 1/n)`` with ``n`` the number
        of weighted pairs, so a cap that cannot hold (n=2 and 0.40) gives equal weights;
        or 'one_pass' — clip at ``max_weight`` then renormalize once, as shipped before
        2026-10-01 (renormalizing can lift weights above the cap again).

    Returns
    -------
    DataFrame with columns
      ["pair", "resid_var", "var_per_dollar", "inv_var_weight", "weight", "suggested_capital_pct"]
    sorted by weight descending (ties keep the order of ``kf_results``). ``resid_var`` is the
    price-unit variance and is always reported; ``var_per_dollar`` is NaN under
    ``risk="price_units"``; ``inv_var_weight`` is the weight by the chosen risk before the cap
    (it holds ∝ 1/standard deviation under 'inv_vol', and equal weights under 'equal'); ``weight``
    is after the cap.

    Raises
    ------
    ValueError : for an unknown ``method``, ``risk`` or ``cap``, a ``max_weight`` outside (0, 1],
        and, under ``risk="per_dollar"``, when ``prices`` is None, lacks a ticker of ``kf_results``,
        holds a non-positive close, shares fewer than ``min(3, rows)`` dates with a frame's index
        (a timezone-aware index against a naive one, a shifted calendar or a RangeIndex: a caller
        bug, not a pair with no usable risk), or a frame has no ``beta`` column. An empty
        ``kf_results`` returns an empty frame.
    """
    if method not in ("inv_var", "inv_vol", "equal"):
        raise ValueError(f"method must be 'inv_var', 'inv_vol' or 'equal'; got {method!r}")
    if risk not in ("per_dollar", "price_units"):
        raise ValueError(f"risk must be 'per_dollar' or 'price_units'; got {risk!r}")
    if cap not in ("iterative", "one_pass"):
        raise ValueError(f"cap must be 'iterative' or 'one_pass'; got {cap!r}")
    if not (0.0 < max_weight <= 1.0):
        raise ValueError(f"max_weight must be in (0, 1]; got {max_weight}")

    if not kf_results:
        return pd.DataFrame(columns=_WEIGHT_COLUMNS)

    if risk == "per_dollar":
        if prices is None:
            raise ValueError(
                "risk='per_dollar' needs prices: pass prices= (a wide DataFrame of closes, one "
                "column per ticker, indexed like the frames) or risk='price_units' for the "
                "price-unit variance")
        if not isinstance(prices, pd.DataFrame):
            raise ValueError(f"prices must be a wide DataFrame with one column per ticker; got {type(prices).__name__}")
        missing = sorted({t for k in kf_results for t in k if t not in prices.columns})
        if missing:
            raise ValueError(
                f"prices has no column for {missing}; pass prices= with every ticker of kf_results "
                "or risk='price_units'")

    rows = []
    for (k1, k2), df in kf_results.items():
        label = f"{k1}/{k2}"
        if "resid" not in df.columns:
            var = float("nan")
        else:
            ret = df["resid"].diff().dropna()
            var = float(ret.var(ddof=1)) if len(ret) >= 2 else float("nan")
        var_pd = float("nan")
        if risk == "per_dollar":
            var_pd = _variance_per_dollar(df, prices[k1], prices[k2], label)
        rows.append({"pair": label, "resid_var": var, "var_per_dollar": var_pd})

    result = pd.DataFrame(rows)

    # Raw weights by the chosen risk; a pair with no positive variance gets none
    risk_var = result["var_per_dollar" if risk == "per_dollar" else "resid_var"].to_numpy(dtype=float)
    valid = np.isfinite(risk_var) & (risk_var > 0)

    raw = np.zeros(len(result))
    if method == "inv_var":
        raw[valid] = 1.0 / risk_var[valid]
    elif method == "inv_vol":
        raw[valid] = 1.0 / np.sqrt(risk_var[valid])
    else:  # equal
        raw[valid] = 1.0

    total_raw = raw.sum()
    inv_var_weight = raw / total_raw if total_raw > 0 else raw

    if cap == "iterative":
        weight = _capped_weights(inv_var_weight, max_weight)
    else:  # one_pass: clip, then renormalize once
        clipped = np.clip(inv_var_weight, 0.0, max_weight)
        clipped_sum = clipped.sum()
        weight = clipped / clipped_sum if clipped_sum > 0 else clipped

    result["inv_var_weight"] = inv_var_weight
    result["weight"] = weight
    result["suggested_capital_pct"] = weight * 100.0

    return (result[_WEIGHT_COLUMNS]
            .sort_values("weight", ascending=False, kind="stable").reset_index(drop=True))


# ---- capital splits across pairs ---------------------------------------------------------------

_ALLOCATION_METHODS = ("equal", "inv_vol", "inv_var", "erc", "leg_split")

# L-BFGS-B settings of the equal-risk-contribution solve.  gtol is the tolerance the study fixes.
# scipy's default ftol (2.2e-9) would stop the solve first, at a risk-contribution error up to
# 1.6e-3 on real 20-pair covariances, so ftol is 0 and the solve runs until the line search can
# make no further progress.  That stop is reported as ABNORMAL even when the solution is exact to
# 1e-7, so scipy's success flag is not used: a solve is accepted when the largest relative error
# of the risk contributions is below _ERC_TOL, and rejected otherwise.
_ERC_GTOL, _ERC_FTOL, _ERC_MAXITER, _ERC_TOL = 1e-12, 0.0, 1000, 1e-5


class ERCConvergenceError(RuntimeError):
    """The equal-risk-contribution solve did not converge.

    Raised by ``allocation_weights(method="erc")`` so that the caller can fall back to another
    split and log the fact; a silently unconverged split would be reported as ERC.
    """


def spread_returns(P1: pd.Series, P2: pd.Series, beta: float) -> pd.Series:
    """
    Return of the spread per dollar of gross notional, one value per session.

    .. math::  r_t = (\\Delta P_{1,t} - \\beta\\,\\Delta P_{2,t}) / (P_{1,t-1} + |\\beta|\\,P_{2,t-1})

    The numerator is the one-day P&L of a hold of one share of leg one against ``-beta`` shares of
    leg two; the denominator is that hold's gross notional at the previous close, which is what
    ``generate_pair_signals`` divides its capital by, so ``r_t`` is the return on a trade sized by
    that function. Risk is measured on the spread, not on the position: the sign of a pair's
    return depends on which ticker is leg one.

    Parameters
    ----------
    P1, P2 : close-price Series on a common DatetimeIndex. Sessions on which either close is
        missing are dropped before differencing (the pair's own frame), so a return can span more
        than one session where a leg did not price.
    beta : hedge ratio (shares of leg two per share of leg one, frozen), a finite scalar; it may be
        negative, in which case both legs are held on the same side.

    Returns
    -------
    Series of ``len(rows) - 1`` returns, indexed by the later session of each difference (the first
    row is dropped).
    """
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError(f"beta must be a finite number; got {beta}")
    px = pd.concat({"P1": pd.Series(P1, dtype=float), "P2": pd.Series(P2, dtype=float)}, axis=1)
    px = px.dropna()
    if len(px) < 2:
        raise ValueError(f"need at least two sessions with both closes; got {len(px)}")
    if not (px > 0).all().all():
        raise ValueError("P1 and P2 must be positive")
    px = px.assign(beta=beta)
    return _per_dollar_returns(px)


def _erc_weights(cov: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Spinu (2013): minimize 1/2 x'Sx - (1/n) sum(log x) over x > 0, on y = log x."""
    n = len(sigma)
    if n == 1:
        return np.ones(1)

    def fun(y):
        x = np.exp(y)
        sx = cov @ x
        return 0.5 * float(x @ sx) - float(y.sum()) / n, x * sx - 1.0 / n

    res = minimize(fun, np.log(1.0 / sigma), jac=True, method="L-BFGS-B",
                   options={"gtol": _ERC_GTOL, "ftol": _ERC_FTOL, "maxiter": _ERC_MAXITER})
    x = np.exp(res.x)
    w = x / x.sum()
    rc = w * (cov @ w)
    err = float(np.max(np.abs(rc / rc.sum() * n - 1.0))) if np.all(np.isfinite(rc)) else np.inf
    if not np.isfinite(err) or err > _ERC_TOL:
        raise ERCConvergenceError(
            f"equal-risk-contribution solve did not converge for {n} pairs "
            f"({res.message!s}, {res.nit} iterations, largest relative risk-contribution "
            f"error {err:.2e} against {_ERC_TOL:.0e})")
    return w


def allocation_weights(
    method: str,
    *,
    sigma: Optional[Sequence[float]] = None,
    cov: Optional[np.ndarray] = None,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> np.ndarray:
    """
    Split one unit of capital across pairs, from numbers rather than prices.

    Parameters
    ----------
    method : one of

        - ``"equal"``    : ``1/n``.
        - ``"inv_vol"``  : ``w_i`` proportional to ``1/sigma_i`` (naive risk parity).
        - ``"inv_var"``  : ``w_i`` proportional to ``1/sigma_i**2``.
        - ``"erc"``      : equal risk contributions under ``cov``, long-only: the minimizer of
          ``1/2 x'Sx - (1/n) sum(log x_i)`` (Spinu 2013; Maillard, Roncalli & Teiletche 2010),
          found by L-BFGS-B on ``log x`` from ``1/sigma`` (``gtol=1e-12``, ``ftol=0``), then
          ``w = x/sum(x)``. The solve is accepted when every risk contribution is within a
          relative ``1e-5`` of ``1/n``; otherwise it raises :class:`ERCConvergenceError` and the
          caller decides the fallback and logs it.
        - ``"leg_split"``: ``w_i`` proportional to ``1 / max(c(t1_i), c(t2_i))`` where ``c(t)`` is
          the number of listed pairs that contain ticker ``t``. Pairs that share a leg share that
          leg's exposure; where no ticker repeats it equals ``"equal"``.
    sigma : per-pair volatility (positive, finite), needed by ``inv_vol`` / ``inv_var``; for
        ``erc`` it is the starting point and defaults to ``sqrt(diag(cov))``.
    cov : ``n x n`` covariance matrix of the pairs' returns (symmetric, positive definite), needed
        by ``erc``. A DataFrame is accepted; its labels are ignored and the row order is used.
    pairs : list of ``(ticker1, ticker2)`` tuples, needed by ``leg_split``.

    ``"equal"`` needs only the count and takes it from whichever of the three inputs is given. When
    several are given they must have the same length.

    Returns
    -------
    ndarray of ``n`` weights, in the order of the input, non-negative and summing to one.
    """
    if method not in _ALLOCATION_METHODS:
        raise ValueError(f"method must be one of {_ALLOCATION_METHODS}; got {method!r}")

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim != 1 or sigma.size == 0:
            raise ValueError(f"sigma must be a non-empty 1-D array; got shape {sigma.shape}")
        if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
            raise ValueError("sigma must be finite and positive")
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
            raise ValueError(f"cov must be a non-empty square matrix; got shape {cov.shape}")
        if not np.all(np.isfinite(cov)):
            raise ValueError("cov must be finite")
        if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-12 * np.abs(cov).max()):
            raise ValueError("cov must be symmetric")
    if pairs is not None:
        pairs = [tuple(p) for p in pairs]
        if not pairs or any(len(p) != 2 or p[0] == p[1] for p in pairs):
            raise ValueError("pairs must be a non-empty list of (ticker1, ticker2) tuples "
                             "with two different tickers each")

    sizes = {name: len(v) for name, v in (("sigma", sigma), ("cov", cov), ("pairs", pairs))
             if v is not None}
    if not sizes:
        raise ValueError("give at least one of sigma, cov or pairs")
    if len(set(sizes.values())) > 1:
        raise ValueError(f"sigma, cov and pairs must have the same length; got {sizes}")
    n = next(iter(sizes.values()))

    if method == "equal":
        return np.full(n, 1.0 / n)
    if method in ("inv_vol", "inv_var"):
        if sigma is None:
            raise ValueError(f"method {method!r} needs sigma")
        raw = 1.0 / sigma if method == "inv_vol" else 1.0 / sigma ** 2
        return raw / raw.sum()
    if method == "erc":
        if cov is None:
            raise ValueError("method 'erc' needs cov")
        if np.linalg.eigvalsh(cov)[0] <= 0.0:
            raise ValueError("cov must be positive definite")
        start = np.sqrt(np.diag(cov)) if sigma is None else sigma
        return _erc_weights(cov, start)
    # leg_split
    if pairs is None:
        raise ValueError("method 'leg_split' needs pairs")
    count = Counter(t for p in pairs for t in p)
    raw = np.array([1.0 / max(count[a], count[b]) for a, b in pairs])
    return raw / raw.sum()
