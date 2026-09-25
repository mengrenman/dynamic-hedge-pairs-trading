"""The three compute tiers the UI is built around.

Measured on this machine, one pair, 1508 daily bars:

    tier 1   generate_pair_signals + evaluate_pair_signals + render     ~430 ms
    tier 2   fit_kalman_hedge (em_iters=5)                             ~1770 ms
    tier 3   the dual-gate screen over 44,850 pairs x 39 formations       ~1 h

Tier 1 is live under HTMX, tier 2 is a button, tier 3 is a background job or a cached artifact.
Keeping that split explicit in the code is what stops the UI promising interactivity it cannot
deliver.

Everything here is a thin wrapper over `pairs`, deliberately: the app must not become a second
implementation of the research. Where a notebook fixed a parameter, the same value is used and the
notebook is named.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pairs import (estimate_halflife, estimate_halflife_window, evaluate_pair_signals,
                   filter_kf_on_new, fit_kalman_hedge, generate_pair_signals)
from pairs.models.kalman import _kalman_dynamic_hedge

from .state import REGISTRY

# notebook 11's engine settings, so a day-lake run here matches the notebook
FORM_YEARS, CAP, COST_BPS, BORROW_BPS = 2, 10_000, 5.0, 50


@dataclass(frozen=True)
class HedgeSpec:
    """Tier 2 — changing any of these means refitting."""
    source: str = "yahoo"
    t1: str = "BKNG"
    t2: str = "MA"
    model: str = "kalman"            # kalman | static | rolling
    q: float = 1e-5
    em_iters: int = 5
    rolling_window: int = 252

    def key(self) -> str:
        return "|".join(str(v) for v in asdict(self).values())


@dataclass(frozen=True)
class SignalSpec:
    """Tier 1 — these only re-run signals and evaluation, never the hedge."""
    z_method: str = "robust"
    z_window: Optional[int] = None   # None -> from the residual's half-life
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    max_hold_bars: Optional[int] = None
    cooldown_bars: int = 0
    capital: float = CAP
    cost_bps: float = COST_BPS
    borrow_bps: float = BORROW_BPS


def _ols(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(a), float(b)


def fit_hedge(spec: HedgeSpec) -> Dict[str, object]:
    """Tier 2. Returns the state frame the signal layer needs, plus what refitting produced.

    Cached on the spec, so the tier-1 knobs never pay for this.
    """
    def run():
        px = REGISTRY.prices(spec.source)
        if spec.t1 not in px.columns or spec.t2 not in px.columns:
            raise ValueError(f"{spec.t1} or {spec.t2} is not in the {spec.source} price frame")
        d = pd.DataFrame({"P1": px[spec.t1], "P2": px[spec.t2]}).dropna()
        if len(d) < 250:
            raise ValueError(f"only {len(d)} overlapping bars for {spec.t1}/{spec.t2}; need 250")

        if spec.model == "static":
            a, b = _ols(d["P1"].to_numpy(), d["P2"].to_numpy())
            states = d.assign(beta=b, resid=d["P1"] - a - b * d["P2"])
            extra = {"alpha": a, "beta": b}
        elif spec.model == "rolling":
            w = spec.rolling_window
            cov = d["P1"].rolling(w).cov(d["P2"])
            var = d["P2"].rolling(w).var()
            beta = (cov / var).shift(1)                      # strictly past data
            alpha = (d["P1"].rolling(w).mean() - beta * d["P2"].rolling(w).mean()).shift(1)
            states = d.assign(beta=beta, resid=d["P1"] - alpha - beta * d["P2"]).dropna()
            extra = {"window": w}
        else:
            frame = (pd.concat({spec.t1: d["P1"], spec.t2: d["P2"]}, names=["ticker", "datetime"])
                       .rename("close").to_frame())
            st, params = fit_kalman_hedge(frame, pairs=[(spec.t1, spec.t2)], mode="filter",
                                          em_iters=spec.em_iters, q=spec.q,
                                          show_progress=False, return_params=True)
            p = params[(spec.t1, spec.t2)]
            states = d.join(st[(spec.t1, spec.t2)][["beta", "resid"]], how="inner")
            extra = {"frozen": {k: p[k] for k in ("F", "Q", "R")},
                     "last_state": (p["last_state_mean"], p["last_state_cov"])}

        hl = float(estimate_halflife(states["resid"].dropna()))
        return {"states": states, "halflife": hl,
                "z_window": estimate_halflife_window(states["resid"]), **extra}
    return REGISTRY._memo(f"hedge:{spec.key()}", run)


def run_signals(states: pd.DataFrame, sig: SignalSpec) -> Dict[str, object]:
    """Tier 1. ~130 ms for a six-year daily frame."""
    z_window = sig.z_window or estimate_halflife_window(states["resid"])
    signals = generate_pair_signals(
        states, z_method=sig.z_method, z_window=z_window,
        z_entry=sig.z_entry, z_exit=sig.z_exit, z_stop=sig.z_stop,
        max_hold_bars=sig.max_hold_bars, cooldown_bars=sig.cooldown_bars,
        capital_per_pair=sig.capital,
    )
    daily, trades, summary = evaluate_pair_signals(
        states[["P1", "P2"]], signals, cost_bps=sig.cost_bps,
        borrow_bps_per_year=sig.borrow_bps, days_per_year=252, bars_per_year=252,
        capital_base=sig.capital,
    )
    return {"signals": signals, "daily": daily, "trades": trades,
            "summary": summary, "z_window": z_window}


def walk_forward(states: pd.DataFrame, sig: SignalSpec, *, train: int = 504,
                 test: int = 126, step: int = 63) -> pd.DataFrame:
    """Fold-by-fold Sharpe on the same hedge, so a single-window number is never shown alone.

    The hedge is *not* refitted per fold — that would be tier 2 many times over. This answers
    "how stable is this configuration across sub-periods", which is the question the sliders
    provoke, not "does the whole procedure hold up", which notebooks 04 and 05 answer properly.
    """
    n = len(states)
    rows = []
    start = 0
    while start + train + test <= n:
        te = states.iloc[start + train: start + train + test]
        if len(te) >= 20:
            r = run_signals(te, sig)
            s = r["summary"]
            rows.append({"start": te.index[0], "end": te.index[-1],
                         "sharpe": float(s["sharpe"]), "net_pnl": float(s["net_pnl"]),
                         "trades": int(s["n_trades"])})
        start += step
    return pd.DataFrame(rows)


def holdout(spec: HedgeSpec, sig: SignalSpec) -> Optional[Dict[str, object]]:
    """The 2026 window, Yahoo only, continuing the filter from the end of training.

    Every call increments a counter the UI displays. Notebooks 04, 05 and 14 all found that a
    window this short cannot rank configurations, so the point of the counter is to make repeated
    looks visible rather than to prevent them.
    """
    if spec.source != "yahoo":
        return None
    te = REGISTRY.yahoo_oos(spec.t1, spec.t2)
    if te is None:
        return None
    fitted = fit_hedge(spec)
    if spec.model == "kalman":
        st, _ = filter_kf_on_new(te[spec.t1], te[spec.t2], frozen=fitted["frozen"],
                                 last_state=fitted["last_state"], init_cov=1e6, mode="filter")
        states_te = st[["beta", "resid"]].join(
            te[[spec.t1, spec.t2]].rename(columns={spec.t1: "P1", spec.t2: "P2"}), how="inner")
    else:
        a, b = fitted.get("alpha", 0.0), fitted["states"]["beta"].iloc[-1]
        states_te = (te[[spec.t1, spec.t2]].rename(columns={spec.t1: "P1", spec.t2: "P2"})
                       .assign(beta=b))
        states_te["resid"] = states_te["P1"] - a - b * states_te["P2"]

    # Warm the z-score exactly as notebook 01 does: re-filter the *training* window under the
    # frozen parameters from a diffuse start, rather than reusing the EM-fitted in-sample states.
    # The two differ, and using the wrong one moves the hold-out Sharpe from 2.64 to 2.26.
    if spec.model == "kalman":
        px = REGISTRY.prices(spec.source)
        d = pd.DataFrame({"P1": px[spec.t1], "P2": px[spec.t2]}).dropna()
        warm_states, _ = filter_kf_on_new(d["P1"], d["P2"], frozen=fitted["frozen"],
                                          last_state=None, init_cov=1e6, mode="filter")
        warm = warm_states["resid"]
    else:
        warm = fitted["states"]["resid"]
    z_window = sig.z_window or estimate_halflife_window(warm)
    signals = generate_pair_signals(
        states_te, z_method=sig.z_method, z_window=z_window, z_history=warm,
        z_entry=sig.z_entry, z_exit=sig.z_exit, z_stop=sig.z_stop,
        max_hold_bars=sig.max_hold_bars, cooldown_bars=sig.cooldown_bars,
        capital_per_pair=sig.capital)
    daily, trades, summary = evaluate_pair_signals(
        states_te[["P1", "P2"]], signals, cost_bps=sig.cost_bps,
        borrow_bps_per_year=sig.borrow_bps, days_per_year=252, bars_per_year=252,
        capital_base=sig.capital)
    touches = REGISTRY.touch_holdout()
    return {"states": states_te, "signals": signals, "daily": daily, "trades": trades,
            "summary": summary, "z_window": z_window, "touches": touches}


# ── tier 3 ────────────────────────────────────────────────────────────────────

LEVERED = frozenset("DUST FAZ LABU TVIX UVXY NUGT TZA SQQQ TQQQ SPXU UPRO VXX SVXY JNUG "
                    "JDST SOXL SOXS".split())


def _formation_windows(sessions: pd.DatetimeIndex, first="2006-06-30", freq="6MS"):
    anchors = pd.date_range(first, sessions[-1], freq=freq)
    forms = pd.DatetimeIndex(sorted({sessions[sessions <= a][-1] for a in anchors
                                     if (sessions <= a).any()}))
    return forms, {d: (d, n) for d, n in zip(forms, list(forms[1:]) + [sessions[-1]])}


def portfolio_backtest(rule: str, sig: SignalSpec, *, hedge: str = "static",
                       max_pairs: int = 20, exclude_levered: bool = False,
                       job=None) -> Dict[str, object]:
    """Notebook 11's engine, driven from the UI. Minutes, so this is always a background job.

    `exclude_levered` is open issue #6: 39% of notebook 11's P&L comes from 17 leveraged, inverse
    and volatility ETFs that clear every liquidity gate because nothing in the screen asks what the
    instrument is. The switch is here so the question can be answered rather than argued about.
    """
    # The day lake is 803 MB and takes a few seconds on the first call of the process. Tick around
    # it so the progress bar is not dead on arrival, and so a cancel during the load is noticed.
    if job is not None:
        job.tick(0.0, "loading the day lake (803 MB, first use only)…")
    px = REGISTRY.day_bars()
    if job is not None:
        job.tick(0.02, "loading dividends…")
    div = REGISTRY.day_dividends()
    sessions = pd.DatetimeIndex(px.index)
    forms, windows = _formation_windows(sessions)
    if job is not None:
        job.tick(0.04, "reading the selection rule…")
    sel = REGISTRY.day_rule(rule)
    order = {"bh_dual": "eg_p_fdr", "raw_dual_top20": "eg_p", "distance_top20": "ssd"}[rule]

    jobs_list = []
    for f in forms:
        g = sel[pd.to_datetime(sel["formation"]) == f].nsmallest(max_pairs, order)
        for a, b in zip(g["ticker1"], g["ticker2"]):
            if exclude_levered and (a in LEVERED or b in LEVERED):
                continue
            jobs_list.append((a, b, f))

    pnl = pd.Series(0.0, index=sessions)
    active = pd.Series(0, index=sessions, dtype=int)
    rows, n_trades = [], 0
    for i, (a, b, f) in enumerate(jobs_list):
        if job is not None and i % 5 == 0:          # also how often a cancel is noticed
            job.tick(0.05 + 0.95 * i / max(len(jobs_list), 1),
                     f"{i}/{len(jobs_list)} pair-folds")
        if a not in px.columns or b not in px.columns:
            continue
        lo, hi = windows[f]
        form = pd.DataFrame({"P1": px[a], "P2": px[b]}).loc[
            (px.index > f - pd.DateOffset(years=FORM_YEARS)) & (px.index <= f)].dropna()
        trade = pd.DataFrame({"P1": px[a], "P2": px[b], "D1": div[a], "D2": div[b]}).loc[
            (px.index > lo) & (px.index <= hi)].dropna()
        if len(form) < 250 or len(trade) < 20:
            continue
        al, be = _ols(form["P1"].to_numpy(), form["P2"].to_numpy())
        rf = form["P1"] - al - be * form["P2"]
        states = trade[["P1", "P2"]].assign(beta=be, resid=trade["P1"] - al - be * trade["P2"])
        hl = estimate_halflife(rf.dropna())
        win = sig.z_window or (int(np.clip(3 * hl, 20, 250)) if np.isfinite(hl) else 60)
        s = generate_pair_signals(states, z_method=sig.z_method, z_window=win,
                                  z_history=rf.dropna(), z_entry=sig.z_entry, z_exit=sig.z_exit,
                                  z_stop=sig.z_stop, capital_per_pair=sig.capital)
        daily, trades, summ = evaluate_pair_signals(
            states[["P1", "P2"]], s, cost_bps=sig.cost_bps, borrow_bps_per_year=sig.borrow_bps,
            days_per_year=252, bars_per_year=252, capital_base=sig.capital)
        d = trade.loc[daily.index]
        p = daily["pnl_net"] + s["n1"] * d["D1"] + s["n2"] * d["D2"]
        pnl = pnl.add(p.reindex(sessions).fillna(0.0), fill_value=0.0)
        active.loc[p.index] += 1
        n_trades += int(len(trades))
        rows.append({"pair": f"{a}/{b}", "formation": f, "pnl": float(p.sum()),
                     "trades": int(len(trades)),
                     "levered": a in LEVERED or b in LEVERED})

    live = active > 0
    r = (pnl / (active.replace(0, np.nan) * sig.capital)).fillna(0.0)
    rl = r[live]
    folds_df = pd.DataFrame(rows)
    by_year = pnl.groupby(pnl.index.year).sum()
    return {
        "pnl": pnl, "active": active, "folds": folds_df,
        "n_folds": len(folds_df), "n_trades": n_trades,
        "total_pnl": float(pnl.sum()),
        "sharpe": float(rl.mean() / rl.std(ddof=1) * np.sqrt(252)) if len(rl) > 1 else float("nan"),
        "live_sessions": int(live.sum()),
        "by_year": by_year,
        "levered_share": (float(folds_df.loc[folds_df["levered"], "pnl"].sum() / pnl.sum())
                          if len(folds_df) and pnl.sum() else 0.0),
        "levered_folds": int(folds_df["levered"].sum()) if len(folds_df) else 0,
    }
