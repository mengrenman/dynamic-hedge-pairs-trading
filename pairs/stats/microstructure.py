# pairs/stats/microstructure.py
"""
Microstructure diagnostics for intraday prices and spreads.

- roll_spread(prices): Roll (1984) effective bid-ask spread from the serial covariance of price
  changes. Bid-ask bounce makes consecutive price changes negatively autocorrelated; for a
  constant half-spread c, Cov(Δp_t, Δp_{t-1}) = -c², so the effective spread is 2·sqrt(-Cov).
- realized_variance_signature(prices, intervals): realized variance of a series per session when
  it is sampled every k bars ("signature plot", Andersen et al. 2000). For a random walk the
  curve is flat; microstructure noise inflates it at fine sampling.
- epps_correlation(p1, p2, intervals): return correlation of two series sampled every k bars.
  Asynchronous trading and noise drive the correlation towards zero at fine sampling
  (Epps 1979); the interval at which it plateaus is the finest sampling that sees the
  "true" comovement.
- autocorr_by_interval(prices, intervals): first-order autocorrelation of changes at each
  sampling interval — negative values at fine intervals are the bounce, not mean reversion
  one can trade.

All functions take a Series indexed by datetime; ``session`` groups bars into sessions (a
callable on the index or an array of keys, default calendar date) so that no interval ever
spans the overnight gap.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "roll_spread",
    "realized_variance_signature",
    "epps_correlation",
    "autocorr_by_interval",
]


def _session_keys(index: pd.Index, session) -> np.ndarray:
    if session is None:
        return pd.DatetimeIndex(index).normalize().to_numpy()
    if callable(session):
        return np.asarray(session(index))
    return np.asarray(session)


def _sample_every(s: pd.Series, k: int, session) -> pd.Series:
    """Every k-th bar within each session, starting at the session's first bar."""
    if k == 1:
        return s
    keys = _session_keys(s.index, session)
    pos = pd.Series(np.arange(len(s))).groupby(keys, sort=False).cumcount().to_numpy()
    return s[pos % k == 0]


def roll_spread(prices: pd.Series, *, as_bps: bool = True, min_obs: int = 30,
                signed: bool = False, session=None, every: int = 1) -> float:
    """
    Roll (1984) effective spread of a price series (consecutive observations, no session split).

    Returns the spread in basis points of the mean price (``as_bps=True``) or in price units.
    NaN when the serial covariance of price changes is non-negative (no measurable bounce) or
    there are fewer than ``min_obs`` changes.

    ``signed=True`` returns ``-2*sqrt(cov)`` instead of NaN when the covariance comes out
    positive. Dropping those cases looks tidy but biases every average upward: the estimator is
    noisy, so a name whose true spread is small produces a positive covariance a good fraction of
    the time, and keeping only the draws that happened to land negative keeps only the draws that
    happened to look expensive. The signed root is meaningless for a single name -- a negative
    spread does not exist -- but its average over many names is not systematically wrong, which is
    what a cost assumption needs.

    The quantity returned is the *full* spread. A marketable order crosses half of it, so the
    per-transaction cost to compare against a ``cost_bps`` assumption is half this number.

    ``session`` groups bars so that no price change spans the overnight gap -- an overnight return
    is not a bid-ask bounce -- and ``every`` samples every k-th bar within each session before
    differencing. Changes are pooled across sessions into a single covariance rather than
    estimated per session and averaged, which is what makes a coarse ``every`` usable: sampling
    every 15th bar leaves only ~26 changes in a session but tens of thousands across a window.

    The interval matters more than the textbook suggests. Sampled too finely, order-flow
    continuation offsets part of the bounce and the estimate collapses -- at one minute it falls
    below the minimum-tick floor for a third of the names in this repository's universe, which is
    arithmetically impossible rather than merely low. Sampled too coarsely, genuine reversal that
    a strategy could trade gets counted as a cost. Sweep ``every`` and use the plateau.
    """
    p = pd.Series(prices, dtype=float).dropna()
    if session is None and every == 1:
        d = p.diff().dropna()
        pairs_ = (d.to_numpy()[1:], d.to_numpy()[:-1])
    else:
        keys = _session_keys(p.index, session)
        lags, leads = [], []
        for _, g in p.groupby(keys, sort=False):
            d = g.iloc[::every].diff().dropna().to_numpy()
            if len(d) >= 2:
                leads.append(d[1:]); lags.append(d[:-1])
        if not lags:
            return float("nan")
        pairs_ = (np.concatenate(leads), np.concatenate(lags))
    if len(pairs_[0]) < min_obs:
        return float("nan")
    cov = float(np.cov(pairs_[0], pairs_[1], ddof=0)[0, 1])
    if cov < 0:
        spread = 2.0 * np.sqrt(-cov)
    elif signed:
        spread = -2.0 * np.sqrt(cov)
    else:
        return float("nan")
    return float(spread / p.mean() * 1e4) if as_bps else float(spread)


def realized_variance_signature(
    prices: pd.Series,
    intervals: Sequence[int] = (1, 2, 5, 10, 15, 30, 60),
    *,
    session=None,
    log: bool = True,
) -> pd.Series:
    """
    Realized variance of ``prices`` per *bar of the original grid* when the series is sampled
    every ``k`` bars: for each session the squared k-bar changes are summed and divided by the
    number of original bars they span, then averaged over sessions. A random walk gives a flat
    curve (the per-bar variance); microstructure noise inflates it at fine sampling. Indexed by k.
    """
    p = pd.Series(prices, dtype=float).dropna()
    x = np.log(p) if log else p
    out = {}
    for k in intervals:
        xs = _sample_every(x, int(k), session)
        keys = _session_keys(xs.index, session)
        d = xs.groupby(keys, sort=False).diff().dropna()
        if d.empty:
            out[int(k)] = float("nan")
            continue
        sess = _session_keys(d.index, session)
        rv = (d ** 2).groupby(sess, sort=False).sum()
        span = pd.Series(k, index=d.index).groupby(sess, sort=False).sum()   # bars covered
        out[int(k)] = float((rv / span).mean())
    return pd.Series(out, name="rv_per_bar").rename_axis("interval")


def epps_correlation(
    p1: pd.Series,
    p2: pd.Series,
    intervals: Sequence[int] = (1, 2, 5, 10, 15, 30, 60),
    *,
    session=None,
    log: bool = True,
) -> pd.Series:
    """Correlation of the two series' changes when both are sampled every k bars (indexed by k)."""
    df = pd.concat({"a": pd.Series(p1, dtype=float), "b": pd.Series(p2, dtype=float)}, axis=1).dropna()
    x = np.log(df) if log else df
    out = {}
    for k in intervals:
        xs = x.iloc[_sample_every(pd.Series(np.arange(len(x)), index=x.index), int(k), session).to_numpy()]
        keys = _session_keys(xs.index, session)
        d = xs.groupby(keys, sort=False).diff().dropna()
        out[int(k)] = float(d["a"].corr(d["b"])) if len(d) > 2 else float("nan")
    return pd.Series(out, name="corr").rename_axis("interval")


def autocorr_by_interval(
    prices: pd.Series,
    intervals: Sequence[int] = (1, 2, 5, 10, 15, 30, 60),
    *,
    session=None,
    log: bool = False,
) -> pd.Series:
    """First-order autocorrelation of within-session changes at each sampling interval."""
    p = pd.Series(prices, dtype=float).dropna()
    x = np.log(p) if log else p
    out = {}
    for k in intervals:
        xs = _sample_every(x, int(k), session)
        keys = _session_keys(xs.index, session)
        d = xs.groupby(keys, sort=False).diff()
        lag = d.groupby(keys, sort=False).shift(1)
        m = d.notna() & lag.notna()
        out[int(k)] = float(np.corrcoef(d[m], lag[m])[0, 1]) if m.sum() > 2 else float("nan")
    return pd.Series(out, name="acf1").rename_axis("interval")
