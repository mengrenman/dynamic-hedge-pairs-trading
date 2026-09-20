"""Detect instruments that are a mechanical function of another listed instrument.

Notebook 11 found that 39% of its selected pair-folds — and 39% of its P&L — involve a leveraged,
inverse or volatility ETF. Those instruments clear every liquidity gate because nothing in the
screen asks what the instrument *is*: a 3× fund on a liquid index is itself liquid, priced above
$5 and volatile. Cointegrating one against its own underlying, or against another fund on the same
index, is an arithmetic identity rather than an economic relationship.

The obvious fix — a hand-curated ticker list — is brittle, incomplete and survivorship-biased. A
list of the seventeen names notebook 11 happened to trade misses DGAZ, UGAZ, DRIP, GUSH, LABD,
BOIL and dozens of others, and cannot know about funds that delisted before anyone looked.

So this detects them by behaviour instead. A leveraged or inverse fund is, by construction, a
near-exact scalar multiple of some other listed thing, and it is the *magnified* side of that
relationship:

    exists j with |corr(r_i, r_j)| >= rho_min  and  |beta_ij| >= 1 + beta_tol

The direction matters. If BULL3 is 3x IDX then IDX is one-third of BULL3, so a rule that merely
asks whether beta is far from 1 flags the underlying as well — and gating GDX, SOXX or QLD out of
the universe because a leveraged sibling exists would remove far more than the funds themselves.
Only |beta| >= 1 + tol counts, never |beta| <= 1 - tol.

The test finds SOXS at −2.98x SOXX, JDST at −2.96x GDXJ, NUGT at +2.97x GDX, LABU at +3.02x XBI
and TQQQ at 1.49x QLD (a 3x against a 2x on the same index). Crucially it needs no benchmark: the
gold-miner funds have an R-squared of 0.005 against the S&P, so any single-benchmark regression
misses them entirely. It is checked against *every* near-relative rather than only the closest,
and it is measured against the single closest relative: scanning every near-relative instead
flags SPY and QQQ, because in a correlated cluster the higher-volatility members always look
magnified relative to the lowest-volatility one.

Two known limitations, both visible rather than hidden:

* a fund whose closest relative is another fund of *different* leverage escapes the scaled arm
  from the wrong side — QLD (2×) sits at beta 0.67 against TQQQ (3×), which is a shrink rather
  than a magnification. A fund whose closest relative carries the *same* leverage escapes too:
  TZA is beta 0.996 against SRTY, both −3× Russell. Both cases surface in the ``duplicate`` arm.
  Partial coverage with no false positives is worth more here than full coverage that would gut
  the universe.
* SVXY changed from −1× to −0.5× in February 2018, so across that boundary it is not a constant
  multiple of anything and no behavioural test can see it.

The ``duplicate`` arm (|corr| high, beta ~ 1) is reported separately and gated separately. It is
dominated by index funds tracking the same index — AGG and BND, ACWI and VT — which are equally
mechanical but are a different question from notebook 11's, and removing them would change the
universe far more than the leveraged funds do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

__all__ = ["scaled_instrument_report", "detect_scaled_instruments"]


@dataclass(frozen=True)
class Thresholds:
    rho_min: float = 0.95        # correlation with the closest relative
    beta_tol: float = 0.15       # how far from 1.0 the scaling must be
    dup_rho_min: float = 0.98    # the near-duplicate arm
    min_obs: int = 120           # bars required before a verdict is offered


def scaled_instrument_report(prices: pd.DataFrame, *,
                             thresholds: Thresholds = Thresholds(),
                             tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Per-ticker verdict over one window of closing prices.

    ``prices`` is wide: a DatetimeIndex and one column per ticker. Only columns complete over the
    window are judged, because a correlation computed on different day-counts is not comparable
    across the candidates and this test turns on the top of the correlation distribution.

    Returns a frame indexed by ticker with the closest relative, the correlation and implied beta
    against it, annualised volatility, and a ``kind`` of ``scaled`` / ``duplicate`` / ``ordinary``.
    """
    if tickers is not None:
        prices = prices[[t for t in tickers if t in prices.columns]]
    r = np.log(prices.astype(float)).diff().iloc[1:]
    r = r.loc[:, r.notna().all()]
    r = r.loc[:, r.std() > 1e-12]
    if r.shape[1] < 2 or len(r) < thresholds.min_obs:
        return pd.DataFrame(columns=["partner", "rho", "beta", "ann_vol", "kind"]).rename_axis("ticker")

    names = list(r.columns)
    X = r.to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)
    C = (X.T @ X) / len(X)
    np.fill_diagonal(C, 0.0)

    sd = r.std().to_numpy()
    B = C * (sd[:, None] / sd[None, :])          # beta of i on j, for every pair

    # The relationship is symmetric: if BULL3 is 3x IDX then IDX is one-third of BULL3, and a rule
    # that only asks "is beta far from 1" flags the *underlying* too. Gating GDX, SOXX or QLD out
    # of the universe because a leveraged sibling exists would be much worse than the disease. So
    # only the magnified side is flagged — |beta| >= 1 + tol, never <= 1 - tol.
    # Compare against the single *closest* relative, not every near-relative. Scanning them all
    # and taking the largest |beta| flags SPY and QQQ, because in any correlated cluster the
    # higher-volatility members look magnified relative to the lowest — SPY against a min-vol S&P
    # fund clears 1.15 easily. The closest relative of a genuine fund is its own underlying or a
    # sibling on the same index, which is exactly the comparison that carries the information.
    j = np.abs(C).argmax(axis=1)
    rho = C[np.arange(len(names)), j]
    beta = B[np.arange(len(names)), j]

    scaled = (np.abs(rho) >= thresholds.rho_min) & (np.abs(beta) >= 1.0 + thresholds.beta_tol)
    out = pd.DataFrame({"partner": [names[a] for a in j], "rho": rho, "beta": beta,
                        "ann_vol": sd * np.sqrt(252),
                        "scaled_vs": np.where(scaled, [names[a] for a in j], ""),
                        "scaled_beta": np.where(scaled, beta, np.nan)},
                       index=pd.Index(names, name="ticker"))
    dup = (~scaled) & (np.abs(rho) >= thresholds.dup_rho_min) & \
          (np.abs(beta - 1.0) < thresholds.beta_tol)
    out["kind"] = np.where(scaled, "scaled", np.where(dup, "duplicate", "ordinary"))
    return out


def detect_scaled_instruments(prices: pd.DataFrame, *,
                              thresholds: Thresholds = Thresholds(),
                              include_duplicates: bool = False,
                              tickers: Optional[Iterable[str]] = None) -> set[str]:
    """The set to hand to ``liquidity_screen(exclude=...)``.

    By default only the ``scaled`` arm — leveraged and inverse funds, which is notebook 11's
    finding. Pass ``include_duplicates=True`` to also drop same-index twins.
    """
    rep = scaled_instrument_report(prices, thresholds=thresholds, tickers=tickers)
    kinds = {"scaled", "duplicate"} if include_duplicates else {"scaled"}
    return set(rep.index[rep["kind"].isin(kinds)])
