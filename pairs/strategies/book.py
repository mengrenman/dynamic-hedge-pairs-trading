# pairs/strategies/book.py
"""
Leg-level book simulator for one pair whose size is set by a dollar target.

Exports
-------
- simulate_pair_book(prices, pos, beta, target_notional, ...)

``generate_pair_signals`` sizes every trade at one fixed ``capital_per_pair``, so a study that
asks what a different size would have earned has to re-run signals it does not want to change.
Position size never feeds back into a signal, so this module takes the executed position path
(``pos``, the signal generator's output) as given and re-sizes it: it owns the share counts, the
ledger and the costs, and nothing else.

Timing and ledger conventions are exactly those of ``generate_pair_signals(exec_lag=1)`` followed
by ``evaluate_pair_signals`` and the dividend line of notebook 11:

- The decision for ledger row t+1 is made at the close of row t, with row t's prices and row t's
  target: ``n1 = pos_{t+1} * T_t / (P1_t + |beta_t| * P2_t)`` and ``n2 = -beta_t * n1``.
- Those shares are booked on row t+1, earn the move from close t to close t+1, and pay cost on
  ``|delta shares| * close_{t+1}`` for each leg.
- Row 0 is flat; a decision on the last row is never executed; a position open on the last row is
  marked to market and never closed, with no closing cost; a row whose prices are not finite and
  positive is dropped from the ledger, not zero-filled.
- Borrow on row t is ``borrow_bps / 1e4 / days_per_year * short shares * close_t``; dividends are
  ``shares * dividend`` on the ex-date row.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["simulate_pair_book"]

_RESETS = ("daily", "frozen", "band")

Numeric = Union[float, pd.Series, Sequence[float], np.ndarray]


def _per_row(x: Numeric, idx: pd.Index, name: str) -> np.ndarray:
    """A scalar, a Series on ``idx`` or an array of ``len(idx)``, as a float array."""
    if np.ndim(x) == 0:
        return np.full(len(idx), float(x))
    if isinstance(x, pd.Series):
        if not x.index.equals(idx):
            raise ValueError(f"{name} is not aligned to prices: its index differs from the ledger rows. "
                             "Reindex it onto prices.index first.")
        return x.to_numpy(dtype=float)
    a = np.asarray(x, dtype=float)
    if a.shape != (len(idx),):
        raise ValueError(f"{name} must be a scalar or have one entry per ledger row ({len(idx)}), "
                         f"got shape {a.shape}")
    return a


def simulate_pair_book(
    prices: pd.DataFrame,
    pos: Union[pd.Series, Sequence[int], np.ndarray],
    beta: Numeric,
    target_notional: Numeric,
    *,
    reset: str = "daily",
    band: float = 0.10,
    cost_bps: Union[float, Sequence[float]] = 0.0,
    borrow_bps_per_year: float = 50.0,
    dividends: Optional[pd.DataFrame] = None,
    days_per_year: int = 252,
) -> pd.DataFrame:
    """
    Re-size one pair's executed position path to a dollar target and book it leg by leg.

    Parameters
    ----------
    prices : DataFrame with columns ``P1``, ``P2`` on the ledger rows (unique, increasing index).
    pos : executed position on each ledger row, in ``{-1, 0, +1}`` -- the ``pos`` column of
        ``generate_pair_signals``. The decision for row t+1 is ``pos[t+1]``, taken at the close of
        row t. Row 0 must be flat.
    beta : hedge ratio (shares of leg two per share of leg one) at each decision close: a scalar
        or a Series on the ledger rows. May be negative.
    target_notional : ``T``, the gross dollar notional a held trade is sized to
        (``|n1| P1 + |n2| P2 = T`` at the sizing close): a scalar or a Series giving the target in
        force at each DECISION close (row t's value sizes row t+1's shares).
    reset : how a trade that is held at two consecutive decision closes is re-sized.
        ``"daily"``  -- shares are reset to the target at every decision close, so a held trade's
        shares change on nearly every row (notebook 11's behavior).
        ``"frozen"`` -- shares are set by the entry decision and kept until the position goes
        flat; a change in ``target_notional`` while held is ignored.
        ``"band"``   -- shares are reset to the target at a decision close only when the gross
        notional of the held shares, at that close's prices, differs from that close's target by
        more than ``band * T``. An entry is always sized at the target.
    band : tolerance of ``reset="band"`` as a fraction of the target (``0`` equals ``"daily"``;
        a very large value equals ``"frozen"``).
    cost_bps : transaction cost in bps of traded notional per leg-side: a scalar for both legs or
        ``(c1, c2)`` for leg one and leg two (measured costs differ by ticker).
    borrow_bps_per_year : annual borrow on the market value of short shares.
    dividends : optional DataFrame with columns ``D1``, ``D2`` (cash dividend per share on the
        ex-date row, zero elsewhere) covering the ledger rows; long shares receive, short pay.
    days_per_year : divisor of the borrow accrual.

    Returns
    -------
    DataFrame on the ledger rows (rows with non-finite or non-positive prices are dropped, as
    ``evaluate_pair_signals`` does), with columns

    - ``pos``                          executed position (as given)
    - ``n1``, ``n2``                   executed shares held on the row
    - ``dn1``, ``dn2``                 signed change in shares booked on the row (net them across
      pairs by ticker to charge cost on a netted order)
    - ``pnl_price``                    ``n1 dP1 + n2 dP2`` (= ``pnl_gross`` of ``evaluate_pair_signals``)
    - ``cost``                         transaction cost only, ``c1 |dn1| P1 + c2 |dn2| P2`` in dollars
    - ``borrow``                       borrow accrued on the row
    - ``dividends``                    dividend cash flow (zero when ``dividends`` is None)
    - ``pnl``                          ``pnl_price - cost - borrow + dividends``
    - ``traded_notional_1``, ``traded_notional_2``   ``|dn_k| P_k``
    - ``notional_1``, ``notional_2``   ``|n_k| P_k`` held at the row's close
    - ``gross_notional``, ``net_notional``   ``|n1| P1 + |n2| P2`` and ``n1 P1 + n2 P2``
    - ``entry``                        the position was opened on this row (a reversal opens one)
    - ``exit``                         the position held on the previous ledger row ended on this row
      (flat, or reversed); the closing print's cost is booked here. Completed round trips are
      ``exit.sum()``; a position open on the last row has no exit.
    - ``resize``                       a held trade's shares changed on this row (not an entry)
    - ``trade_id``                     1, 2, ... for the rows on which a trade is held, 0 when flat

    With ``reset="daily"``, a scalar ``target_notional`` equal to ``capital_per_pair`` and a scalar
    ``cost_bps``, ``pnl - dividends`` equals ``evaluate_pair_signals(...)[0]["pnl_net"]`` and
    ``cost + borrow`` its ``cost`` column, to floating-point error.
    """
    if reset not in _RESETS:
        raise ValueError(f"reset must be one of {_RESETS}; got {reset!r}")
    if not (np.isfinite(band) and band >= 0.0):
        raise ValueError(f"band must be a non-negative finite number; got {band}")
    if days_per_year <= 0:
        raise ValueError(f"days_per_year must be positive; got {days_per_year}")
    if not (np.isfinite(borrow_bps_per_year) and borrow_bps_per_year >= 0.0):
        raise ValueError(f"borrow_bps_per_year must be a non-negative finite number; got {borrow_bps_per_year}")
    if not {"P1", "P2"}.issubset(prices.columns):
        raise ValueError(f"prices must contain columns ['P1', 'P2']; got {list(prices.columns)}")
    idx = prices.index
    if not (idx.is_unique and idx.is_monotonic_increasing):
        raise ValueError("prices must have a unique, increasing index (one row per ledger row)")

    c = np.asarray(cost_bps, dtype=float)
    if c.ndim == 0:
        c = np.repeat(c, 2)
    if c.shape != (2,) or not (np.all(np.isfinite(c)) and np.all(c >= 0.0)):
        raise ValueError(f"cost_bps must be a non-negative scalar or a (c1, c2) pair; got {cost_bps!r}")
    c1, c2 = float(c[0]), float(c[1])

    n = len(idx)
    cols = ["pos", "n1", "n2", "dn1", "dn2", "pnl_price", "cost", "borrow", "dividends", "pnl",
            "traded_notional_1", "traded_notional_2", "notional_1", "notional_2",
            "gross_notional", "net_notional", "entry", "exit", "resize", "trade_id"]
    if n == 0:
        return pd.DataFrame(columns=cols, index=idx)

    p = _per_row(pos, idx, "pos")
    if not np.all(np.isin(p, (-1.0, 0.0, 1.0))):
        raise ValueError("pos must take values in {-1, 0, +1}")
    p = p.astype(int)
    if p[0] != 0:
        raise ValueError("pos[0] must be 0: the first ledger row is flat, there is no decision "
                         "close before it")
    b = _per_row(beta, idx, "beta")
    T = _per_row(target_notional, idx, "target_notional")

    P1 = prices["P1"].to_numpy(dtype=float)
    P2 = prices["P2"].to_numpy(dtype=float)
    ok = np.isfinite(P1) & np.isfinite(P2) & (P1 > 0) & (P2 > 0)

    if dividends is None:
        D1 = D2 = np.zeros(n)
    else:
        if not {"D1", "D2"}.issubset(dividends.columns):
            raise ValueError("dividends must contain columns ['D1', 'D2']")
        missing = idx.difference(dividends.index)
        if len(missing):
            raise ValueError(f"dividends does not cover the ledger: {len(missing)} of {n} rows "
                             f"missing (first: {missing[0]!r})")
        d = dividends.reindex(idx)
        D1, D2 = d["D1"].fillna(0.0).to_numpy(float), d["D2"].fillna(0.0).to_numpy(float)

    # ---- shares: the decision at the close of row t sizes row t+1 ----------------------------
    n1 = np.zeros(n)
    n2 = np.zeros(n)
    for i in range(1, n):
        side, t = p[i], i - 1
        if side == 0:
            continue
        held = p[t] == side                      # held at the previous decision close and at this one
        if held and reset == "frozen":
            n1[i], n2[i] = n1[t], n2[t]
            continue
        if not (ok[t] and np.isfinite(b[t]) and np.isfinite(T[t]) and T[t] >= 0.0):
            raise ValueError(f"cannot size the position held on row {i} ({idx[i]!r}): the decision "
                             f"close {idx[t]!r} needs finite positive prices, a finite beta and a "
                             f"finite non-negative target")
        if held and reset == "band":
            gross = abs(n1[t]) * P1[t] + abs(n2[t]) * P2[t]
            if abs(gross - T[t]) <= band * T[t]:
                n1[i], n2[i] = n1[t], n2[t]
                continue
        s = side * T[t] / (P1[t] + abs(b[t]) * P2[t])
        n1[i], n2[i] = s, -b[t] * s

    # ---- the ledger, on rows with usable prices -----------------------------------------------
    keep = ok
    out_idx = idx[keep]
    P1, P2, n1, n2, p = P1[keep], P2[keep], n1[keep], n2[keep], p[keep]
    D1, D2 = D1[keep], D2[keep]

    dP1 = np.diff(P1, prepend=P1[:1])
    dP2 = np.diff(P2, prepend=P2[:1])
    dn1 = np.diff(n1, prepend=0.0)
    dn2 = np.diff(n2, prepend=0.0)
    tn1, tn2 = np.abs(dn1) * P1, np.abs(dn2) * P2
    pnl_price = n1 * dP1 + n2 * dP2
    cost = (c1 * tn1 + c2 * tn2) / 1e4
    short = np.where(n1 < 0, -n1 * P1, 0.0) + np.where(n2 < 0, -n2 * P2, 0.0)
    borrow = short * (borrow_bps_per_year / 1e4) / float(days_per_year)
    div = n1 * D1 + n2 * D2

    prev = np.concatenate([[0], p[:-1]])
    entry = (p != 0) & (p != prev)
    exit_ = (prev != 0) & (p != prev)
    resize = (p != 0) & (p == prev) & ((dn1 != 0.0) | (dn2 != 0.0))
    trade_id = np.cumsum(entry) * (p != 0)

    return pd.DataFrame({
        "pos": p, "n1": n1, "n2": n2, "dn1": dn1, "dn2": dn2,
        "pnl_price": pnl_price, "cost": cost, "borrow": borrow, "dividends": div,
        "pnl": pnl_price - cost - borrow + div,
        "traded_notional_1": tn1, "traded_notional_2": tn2,
        "notional_1": np.abs(n1) * P1, "notional_2": np.abs(n2) * P2,
        "gross_notional": np.abs(n1) * P1 + np.abs(n2) * P2, "net_notional": n1 * P1 + n2 * P2,
        "entry": entry, "exit": exit_, "resize": resize, "trade_id": trade_id.astype(int),
    }, index=out_idx)[cols]
