# pairs/strategies/spread_accounting.py
"""
Split a time-varying-hedge spread's P&L into what a position actually earns
and what the filter's own re-marking contributes.

Background
----------
For a spread built from a hedge that moves over time -- a rolling-LS or
Kalman hedge ratio, say -- the usual backtest P&L is booked as the day-to-day
change in the spread itself::

    spread_t = w_t' y_t - c_t
    book_t   = signal_{t-1} * (spread_t - spread_{t-1})

where ``y_t`` are the leg (log) prices, ``w_t`` the hedge weights, and ``c_t``
a centering term (e.g. ``mu_t / (1 + gamma_t)`` for a Kalman mu/gamma spread).
That identity decomposes as::

    spread_t - spread_{t-1}
        = w_{t-1}' (y_t - y_{t-1})                          <- tradable
        + [ (w_t - w_{t-1})' y_t - (c_t - c_{t-1}) ]         <- revaluation

The first term is the return a trader who entered at the close of ``t-1``
with weights ``w_{t-1}`` actually earns from the price move over ``t-1`` to
``t`` -- it only uses information available when the position was formed.
The second term (bracket) is the filter re-marking *its own parameters*
against *today's* price level; nobody can trade it, because ``w_t`` and
``c_t`` are only known at the close of ``t``, after the price move they are
being multiplied into has already happened.

Under a contrarian signal (short the spread when it is rich, long when it is
cheap) the bracket term is positive in expectation: the Kalman/rolling-LS
update at ``t`` is proportional to the innovation at ``t-1``, which is itself
proportional to the spread the contrarian signal is betting against, so the
sign of the bracket lines up with the sign of the position built from that
same spread.

This module makes the split explicit so a backtest can report the tradable
line (what a trader could actually earn) alongside the book line (what a
naive diff(spread) P&L would report), rather than conflating the two.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["decompose_spread_pnl", "turnover_and_fees"]


def decompose_spread_pnl(
    y: np.ndarray,
    w: np.ndarray,
    c: np.ndarray,
    signal: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Split a spread's mark-to-market P&L into a tradable line and a revaluation line.

    Parameters
    ----------
    y : ndarray, shape (T, 2)
        Leg log prices (or any linear price basis the caller is using), one
        row per bar.
    w : ndarray, shape (T, 2)
        Spread weights. ``w[t]`` is the weight vector used to *compute the
        spread at bar t*, i.e. ``spread[t] = w[t] @ y[t] - c[t]``. Whether
        that weight was itself known at the close of bar ``t`` or of bar
        ``t-1`` is the caller's modeling choice (a causal Kalman filter's
        one-step-ahead predicted state, say, is already information through
        ``t-1`` even though it indexes bar ``t``) -- document that choice
        where ``w`` is built. This function applies its own, additional one-bar
        lag when forming the tradable line (see below), matching the
        convention that the position entered at the close of ``t-1`` is what
        earns the return from ``t-1`` to ``t``.
    c : ndarray, shape (T,)
        Spread centering term at each bar (e.g. ``mu_t / (1 + gamma_t)`` for a
        Kalman mu/gamma spread; zero if the spread has no intercept).
    signal : ndarray, shape (T,)
        Position size *decided at the close of bar t* (e.g. +1/0/-1). The
        position that earns the move from ``t-1`` to ``t`` is
        ``signal[t-1]`` -- this function performs that lag internally, so
        pass the undelayed decision series.

    Returns
    -------
    dict of ndarrays, each shape (T,):
        ``spread``      : ``w[t] @ y[t] - c[t]``.
        ``book``        : ``signal[t-1] * (spread[t] - spread[t-1])`` -- the
                           P&L a naive "diff the spread" backtest would book.
        ``tradable``     : ``signal[t-1] * w[t-1] @ (y[t] - y[t-1])`` -- the
                           P&L actually earned by holding the weights known
                           when the position was formed through the price
                           move that followed.
        ``revaluation``  : ``book - tradable``, equal by identity to
                           ``signal[t-1] * ((w[t]-w[t-1]) @ y[t] - (c[t]-c[t-1]))``
                           -- the filter re-marking its own parameters
                           against bar ``t``'s price level; not earnable by
                           any trader.

    ``book``, ``tradable`` and ``revaluation`` are all 0 at t=0 (there is no
    prior bar to diff against or to have decided a position at).
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"y must have shape (T, 2), got {y.shape}")
    T = y.shape[0]
    if w.shape != (T, 2):
        raise ValueError(f"w must have shape {(T, 2)}, got {w.shape}")
    if c.shape != (T,):
        raise ValueError(f"c must have shape {(T,)}, got {c.shape}")
    if signal.shape != (T,):
        raise ValueError(f"signal must have shape {(T,)}, got {signal.shape}")

    spread = (w * y).sum(axis=1) - c

    diff_spread = np.zeros(T, dtype=float)
    diff_spread[1:] = spread[1:] - spread[:-1]

    dy = np.zeros_like(y)
    dy[1:] = y[1:] - y[:-1]

    w_lag = np.zeros_like(w)
    w_lag[1:] = w[:-1]          # w[t-1]; row 0 is unused since dy[0] == 0

    signal_lag = np.zeros(T, dtype=float)
    signal_lag[1:] = signal[:-1]  # signal[t-1]; no prior decision at t=0

    book = signal_lag * diff_spread
    tradable = signal_lag * (w_lag * dy).sum(axis=1)
    revaluation = book - tradable

    return {
        "spread": spread,
        "book": book,
        "tradable": tradable,
        "revaluation": revaluation,
    }


def turnover_and_fees(
    signal: np.ndarray,
    w: np.ndarray,
    cost_bps_per_leg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dollars traded per leg and the resulting fee line for a spread position.

    The position held at bar ``t`` is ``signal[t] * w[t]`` (a 2-vector); the
    dollars traded on leg ``i`` at bar ``t`` is
    ``abs(position[t, i] - position[t-1, i])``, with the ``t=0`` row taken to
    be 0 (no prior position to have traded out of, matching the convention
    used throughout this module that diff-based quantities start at 0).

    Parameters
    ----------
    signal : ndarray, shape (T,)
        Position size held at each bar (already at the caller's execution
        timing -- this function does not lag it).
    w : ndarray, shape (T, 2)
        Spread weights at each bar.
    cost_bps_per_leg : ndarray, shape (T, 2)
        One-way cost in basis points of notional, per leg, per bar.

    Returns
    -------
    dollars_traded : ndarray, shape (T, 2)
        Absolute dollars traded on each leg at each bar.
    fees : ndarray, shape (T,)
        Total fee charged at each bar, summed across legs:
        ``sum_i dollars_traded[t, i] * cost_bps_per_leg[t, i] / 1e4``.
    """
    signal = np.asarray(signal, dtype=float)
    w = np.asarray(w, dtype=float)
    cost_bps_per_leg = np.asarray(cost_bps_per_leg, dtype=float)

    T = signal.shape[0]
    if w.shape != (T, 2):
        raise ValueError(f"w must have shape {(T, 2)}, got {w.shape}")
    if cost_bps_per_leg.shape != (T, 2):
        raise ValueError(
            f"cost_bps_per_leg must have shape {(T, 2)}, got {cost_bps_per_leg.shape}"
        )

    position = signal[:, None] * w  # shape (T, 2)

    dollars_traded = np.zeros((T, 2), dtype=float)
    dollars_traded[1:] = np.abs(position[1:] - position[:-1])

    fees = (dollars_traded * (cost_bps_per_leg / 1e4)).sum(axis=1)

    return dollars_traded, fees
