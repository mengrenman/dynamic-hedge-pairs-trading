"""Measure what a transaction costs, instead of assuming it.

Every backtest in this repository charges a flat cost in basis points of traded notional, and
several of its conclusions turn on that number: notebook 12's break-even costs run from 0.9 to
5.7 bps against an assumed 5, which is close enough that the assumption decides the verdict. The
minute lake can settle it, because a Roll (1984) effective spread is computable for any name over
any window, and notebook 08 established that the minute and day lakes share a price basis.

Three choices in here are not the textbook defaults, and each one changes the answer materially:

**Halve the Roll spread.** Roll estimates the full bid-ask spread. A marketable order crosses half
of it, and ``cost_bps`` in :func:`pairs.evaluate_pair_signals` is charged once per transaction on
the notional of that transaction. So the comparable quantity is half the Roll number. Forgetting
this doubles every cost in sight.

**Sample every five minutes, not every one.** Roll assumes the only source of negative serial
covariance is the bounce. At one-minute sampling that is false: order-flow continuation offsets
part of the bounce, and the estimate collapses below the *minimum tick* for a third of the names
in this repository's universe -- a half-spread of 1.4 bps on a stock that traded at $20 in 2009
was not merely cheap, it was arithmetically impossible on a one-cent tick. Sweeping the interval,
the estimate rises from 1.0 bps at one minute to about 2 bps and then flattens, and the share of
impossible values bottoms at five. Coarser sampling starts measuring genuine reversal, which a
strategy earns rather than pays.

**Keep the cells whose covariance comes out positive.** Returning NaN there, as the textbook
estimator does, keeps only the draws that happened to look expensive; on pure random walks with no
spread at all the surviving cells average 1.8 bps. The signed root averages 0.2.

Corwin-Schultz (2012) was tried as an independent anchor and abandoned. On true daily highs and
lows from the day lake, with the paper's overnight adjustment applied, it returns spreads of about
*minus* 10 bps and correlates -0.14 with Roll. The minimum-tick floor is the anchor instead: it
needs no estimator and cannot be argued with.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pairs.stats.microstructure import roll_spread

__all__ = ["CostSpec", "measure_ticker_window_costs", "pair_fold_costs"]


@dataclass(frozen=True)
class CostSpec:
    every: int = 5               # sampling interval in minutes; 5 is where the sweep flattens
    min_obs: int = 200           # pooled lead/lag pairs required before a verdict is offered
    tick: float = 0.01           # the minimum tick, used as an arithmetic floor
    min_bars: int = 2_000        # a six-month window holds roughly 49,000 one-minute bars


def measure_ticker_window_costs(
    windows: Mapping[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]],
    tickers: Mapping[pd.Timestamp, Sequence[str]],
    minute_root,
    *,
    raw_close: Optional[pd.DataFrame] = None,
    spec: CostSpec = CostSpec(),
    loader: Optional[Callable] = None,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """One per-transaction cost per (formation, ticker), in bps of notional.

    ``windows`` maps a formation date to the ``(start, end)`` of its trading window and
    ``tickers`` maps the same key to the names held over it, so only the bars actually needed are
    read. ``raw_close`` is the day lake's unadjusted close, wide by ticker: it supplies the price
    the name really traded at, which is what the tick floor must be computed against -- a
    split-adjusted price of $3 on a stock that traded at $90 would imply a floor thirty times too
    high.

    Returns one row per cell with ``cost_bps`` (the raw estimate), ``tick_floor`` and ``cost_used``
    (the estimate floored), plus an ``err`` column that is empty when the cell was measured.
    """
    from joblib import Parallel, delayed

    if loader is None:                                  # imported here to keep the module light
        from pairs.market_data.minute_bars import load_minute_bars as loader

    minute_root = Path(minute_root)

    def _floor(t: str, lo, hi) -> Tuple[float, float]:
        if raw_close is None or t not in raw_close.columns:
            return np.nan, np.nan
        px = raw_close.loc[(raw_close.index > lo) & (raw_close.index <= hi), t].dropna()
        if not len(px):
            return np.nan, np.nan
        med = float(px.median())
        return med, 0.5 * spec.tick / med * 1e4

    def _window(key):
        lo, hi = windows[key]
        want = list(tickers.get(key, ()))
        if not want:
            return []
        try:
            m = loader(want, str(pd.Timestamp(lo).date()), str(pd.Timestamp(hi).date()),
                       minute_root, price="close_split", freq="1min", layout="market", n_jobs=1)
        except Exception as exc:                        # a missing year is a fact, not a crash
            return [{"formation": key, "ticker": t, "err": f"{type(exc).__name__}: {exc}"}
                    for t in want]
        have = set(m.index.get_level_values("ticker")) if not m.empty else set()
        rows = []
        for t in want:
            px, floor = _floor(t, lo, hi)
            base = {"formation": key, "ticker": t, "raw_px": px, "tick_floor": floor}
            if t not in have:
                rows.append({**base, "err": "absent from the minute lake"})
                continue
            s = m.xs(t, level="ticker")["close"].dropna()
            if len(s) < spec.min_bars:
                rows.append({**base, "err": f"only {len(s)} bars"})
                continue
            full = roll_spread(s, signed=True, session=s.index.normalize(),
                               every=spec.every, min_obs=spec.min_obs)
            if not np.isfinite(full):
                rows.append({**base, "err": "too few usable changes"})
                continue
            half = full / 2.0                           # one side of the spread
            rows.append({**base, "bars": len(s), "cost_bps": half,
                         "cost_used": half if not np.isfinite(floor) else max(half, floor),
                         "err": ""})
        return rows

    out = Parallel(n_jobs=n_jobs)(delayed(_window)(k) for k in windows)
    cols = ["formation", "ticker", "raw_px", "tick_floor", "bars", "cost_bps", "cost_used", "err"]
    df = pd.DataFrame([r for chunk in out for r in chunk])
    return df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=cols)


def pair_fold_costs(cells: pd.DataFrame,
                    selections: Mapping[pd.Timestamp, Iterable[Tuple[str, str]]],
                    *, weights: Optional[Mapping[Tuple, float]] = None,
                    fallback: Optional[float] = None) -> pd.DataFrame:
    """Combine the two legs' costs into one cost per pair-fold.

    A pair trade turns over both legs on entry and again on exit, and the backtest charges
    ``cost_bps`` on each transaction's own notional, so the rate a pair pays per unit of turnover
    is the average of its two legs' rates *weighted by the notional each leg turns over*.

    ``weights`` gives leg one's share of the pair's notional, keyed ``(formation, t1, t2)``.
    Omitting it falls back to a plain mean, which is only correct when the two legs are equally
    sized -- and in this repository they are usually not. ``generate_pair_signals`` sizes the
    trade as ``capital / (P1 + |beta| * P2)``, so leg one's share is ``P1 / (P1 + |beta| * P2)``,
    which for notebook 11's book sits outside 40/60 in two thirds of pair-folds and outside 30/70
    in half of them. The plain mean over-weights whichever leg is smaller; on that book it
    reports 3.45 bps where the notional-weighted figure is 2.44, a 29% overstatement.

    ``fallback`` fills pair-folds where neither leg could be measured; left as None they come back
    NaN, which is the honest default because a name absent from the minute lake is usually a name
    that was thinly traded.
    """
    look = (cells[cells["err"].eq("")].set_index(["formation", "ticker"])["cost_used"]
            if len(cells) else pd.Series(dtype=float))
    rows = []
    for key, pairs in selections.items():
        for a, b in pairs:
            ca, cb = look.get((key, a), np.nan), look.get((key, b), np.nan)
            w1 = np.nan if weights is None else weights.get((key, a, b), np.nan)
            if not np.isfinite(w1):
                w1 = 0.5
            if np.isfinite(ca) and np.isfinite(cb):
                cost = float(w1 * ca + (1.0 - w1) * cb)
            elif np.isfinite(ca) or np.isfinite(cb):
                cost = float(ca if np.isfinite(ca) else cb)   # one leg is better than none
            else:
                cost = float("nan")
            if not np.isfinite(cost) and fallback is not None:
                cost = float(fallback)
            rows.append({"formation": key, "t1": a, "t2": b,
                         "c1": ca, "c2": cb, "w1": w1, "cost_bps": cost})
    return pd.DataFrame(rows, columns=["formation", "t1", "t2", "c1", "c2", "w1", "cost_bps"])
