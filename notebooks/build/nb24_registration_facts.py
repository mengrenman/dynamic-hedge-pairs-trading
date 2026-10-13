"""Formation-window facts for notebook 24's pre-registration (the hedge in logs).

    python notebooks/build/nb24_registration_facts.py [--cache DIR] [--out DIR] [--jobs 4] [--parts ABC]

Reads only the formation-window closes (the ``form`` field) of notebook 22's ``alloc_folds.pkl``,
with the pair, formation and tracker/period flags, and notebook 10's ``day_screen.parquet``. It never
reads a trading-window field. It writes ``loghedge_reg_{A,B,C}.pkl`` to ``--out`` and prints every
number the pre-registration's "Facts established while drafting" section quotes.

Part A, facts 1-4: the level, log and returns hedges, their residuals, z windows, the opening
    transient, and the Engle-Granger/Johansen screen on log prices (with checks that the level fits,
    z windows and p-values reproduce the cache).
Part B, fact 5: EM stepped five and twenty iterations from three starts (H3 levels with notebook 11's
    settings; H4 as first drafted, P0 = 1e6; H4 with every variance scaled by r0).
Part C, fact 6: EM run to convergence under the rule the registration fixes (``em_converge``),
    for H3 and for H4 with scaled variances.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FORM_FIELDS = ("pair", "formation", "alpha", "beta", "z_window", "same", "period", "form")
ITERS_B = 20
# The convergence rule (part C). Differences of log-likelihood are invariant to rescaling the
# observations, so an absolute tolerance in nats is scale-free.
TOL_NATS, MAX_ITERS = 1e-3, 500


# ---------------------------------------------------------------- shared helpers
def ols(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(a), float(b)


def zwin(resid):
    from pairs.stats.stationarity import estimate_halflife
    hl = estimate_halflife(pd.Series(resid))
    return (int(np.clip(3 * hl, 20, 250)) if np.isfinite(hl) else 60), hl


def transient(x):
    a = np.abs(x - x.mean()) / np.std(x)
    return a[0], a[20], a[100:].mean()


def make_kf(y, x, q0, r0, p0):
    from pykalman import KalmanFilter
    H = np.zeros((len(y), 1, 2)); H[:, 0, 0] = x; H[:, 0, 1] = 1.0
    return KalmanFilter(transition_matrices=np.eye(2), observation_matrices=H,
                        initial_state_mean=np.zeros(2), initial_state_covariance=np.eye(2) * p0,
                        transition_covariance=np.eye(2) * q0, observation_covariance=np.array([[r0]]))


def loglik(kf, yy):
    """Gaussian log-likelihood from the predicted innovations; NaN if any innovation variance <= 0
    (pykalman's own ``loglikelihood`` raises instead when Q is indefinite)."""
    from pykalman.standard import _filter
    pm, pc, *_ = _filter(kf.transition_matrices, kf.observation_matrices, kf.transition_covariance,
                         kf.observation_covariance, np.zeros(2), np.zeros(1),
                         kf.initial_state_mean, kf.initial_state_covariance, yy)
    H = kf.observation_matrices
    S = np.einsum("tij,tjk,tlk->til", H, pc, H)[:, 0, 0] + kf.observation_covariance[0, 0]
    v = yy[:, 0] - np.einsum("tij,tj->ti", H, pm)[:, 0]
    if np.any(S <= 0) or not np.all(np.isfinite(S)):
        return np.nan
    return float(-0.5 * np.sum(np.log(2 * np.pi * S) + v * v / S))


def qmin(kf):
    return float(np.linalg.eigvalsh(kf.transition_covariance).min())


def starts(P1, P2):
    """(y, x, q0, r0, p0) for H3 (levels, notebook 11), H4a (logs, first draft), H4 (logs, scaled)."""
    L1, L2 = np.log(P1), np.log(P2)
    a1, b1 = ols(L1, L2)
    r0 = float(np.var(L1 - a1 - b1 * L2))
    return {"H3": (P1, P2, 1e-5, 1.0, 1e6),
            "H4a": (L1, L2, 1e-5 * r0, r0, 1e6),
            "H4": (L1, L2, 1e-5 * r0, r0, 1e6 * r0)}, r0


# ---------------------------------------------------------------- part A: facts 1-4
def part_a(f, cutoff):
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    P1, P2 = f["form"]["P1"].to_numpy(float), f["form"]["P2"].to_numpy(float)
    L1, L2 = np.log(P1), np.log(P2)
    n = len(P1)
    o = {k: f[k] for k in ("pair", "formation", "same", "period")}
    o["n_form"] = n
    a0, b0 = ols(P1, P2)
    e0 = P1 - a0 - b0 * P2
    o.update(b_level=b0, check_alpha=abs(a0 - f["alpha"]), check_beta=abs(b0 - f["beta"]),
             dollar_H0_close=b0 * P2[-1] / P1[-1])
    ratio = P2 / P1
    o["dollar_H0_range_rel"] = (ratio.max() - ratio.min()) / np.median(ratio)
    a1, b1 = ols(L1, L2)
    e1 = L1 - a1 - b1 * L2
    o["b_log"] = b1
    r1, r2 = np.diff(L1), np.diff(L2)                       # Eq. (1) as estimated in §3 / App. A
    a2, b2 = ols(r1, r2)
    x2 = (L1 - L1[0]) - a2 * np.arange(n) - b2 * (L2 - L2[0])
    o.update(beta_ret=b2, drift_126_over_sd=abs(a2) * 126 / np.std(x2),
             corr_e0_e1=float(np.corrcoef(e0, e1)[0, 1]), corr_e1_x2=float(np.corrcoef(e1, x2)[0, 1]))
    for name, e in (("H0", e0), ("H1", e1), ("H2", x2)):
        o[f"zwin_{name}"], o[f"hl_{name}"] = zwin(e)
        o[f"tr0_{name}"], o[f"tr20_{name}"], o[f"tr100_{name}"] = transient(e)
    o["check_zwin"] = o["zwin_H0"] - f["z_window"]
    o["eg_p_level"] = coint(P1, P2, trend="c", autolag="aic")[1]
    o["eg_p_log"] = coint(L1, L2, trend="c", autolag="aic")[1]
    jl = coint_johansen(np.column_stack([L1, L2]), det_order=0, k_ar_diff=1)
    o["joh_log_pass"] = bool(jl.lr1[0] > jl.cvt[0, 1])             # 95%, as the screen
    o["bh_cutoff"] = cutoff.get(f["formation"], np.nan)
    o["log_passes"] = bool(o["eg_p_log"] <= o["bh_cutoff"] and o["joh_log_pass"])
    return o


# ---------------------------------------------------------------- part B: fact 5
def part_b(f):
    P1, P2 = f["form"]["P1"].to_numpy(float), f["form"]["P2"].to_numpy(float)
    st, r0 = starts(P1, P2)
    o = {k: f[k] for k in ("pair", "formation", "same", "period")}
    for v, (y, x, q0, rr, p0) in st.items():
        kf, yy = make_kf(y, x, q0, rr, p0), y.reshape(-1, 1)
        ll, qm, bend = [loglik(kf, yy)], [], []
        for _ in range(ITERS_B):
            kf = kf.em(yy, n_iter=1)
            ll.append(loglik(kf, yy)); qm.append(qmin(kf))
            b = float(kf.filter(yy)[0][-1, 0])
            bend.append(b * P2[-1] / P1[-1] if v == "H3" else b)     # as a dollar ratio
        o[v] = {"ll": ll, "qmin": qm, "dollar_end": bend}
    return o


# ---------------------------------------------------------------- part C: fact 6
def em_converge(kf, yy, tol=TOL_NATS, max_iters=MAX_ITERS):
    """The registered rule: step EM one iteration at a time; after each step require Q positive
    definite and R > 0, and a log-likelihood that did not fall. Stop at the first of: an indefinite
    Q or R <= 0 ("indefinite", keep the previous iterate); a fall of more than ``tol`` nats
    ("decrease", keep the previous iterate); a change below ``tol`` nats ("converged");
    ``max_iters`` ("cap"). Returns (kf, status, iterations kept)."""
    ll_prev = loglik(kf, yy)
    for it in range(1, max_iters + 1):
        nxt = kf.em(yy, n_iter=1)
        if qmin(nxt) <= 0 or nxt.observation_covariance[0, 0] <= 0:
            return kf, "indefinite", it - 1
        ll = loglik(nxt, yy)
        if not np.isfinite(ll) or (np.isfinite(ll_prev) and ll < ll_prev - tol):
            return kf, "decrease", it - 1
        kf = nxt
        if np.isfinite(ll_prev) and ll - ll_prev < tol:
            return kf, "converged", it
        ll_prev = ll
    return kf, "cap", max_iters


def part_c(f):
    P1, P2 = f["form"]["P1"].to_numpy(float), f["form"]["P2"].to_numpy(float)
    st, r0 = starts(P1, P2)
    o = {k: f[k] for k in ("pair", "formation", "same", "period")}
    for v in ("H3", "H4"):
        y, x, q0, rr, p0 = st[v]
        yy = y.reshape(-1, 1)
        t0 = time.perf_counter()
        kf, status, it = em_converge(make_kf(y, x, q0, rr, p0), yy)
        means = kf.filter(yy)[0]
        conv = P2 / P1 if v == "H3" else np.ones_like(P1)           # to a dollar ratio
        d = means[:, 0] * conv
        var_x = np.var(x)
        o[v] = {"status": status, "iters": it, "secs": time.perf_counter() - t0,
                "dollar_end": float(d[-1]),
                "dollar_daily_change": float(np.median(np.abs(np.diff(d[20:])))),
                # scale-free smoothing: the beta state's variance as observation noise, over R
                "snr_beta": float(kf.transition_covariance[0, 0] * var_x / kf.observation_covariance[0, 0])}
    return o


# ---------------------------------------------------------------- summary
def q(s):
    s = pd.Series(s, dtype=float)
    return " | ".join(f"{k} {s.quantile(p):.3g}" for k, p in
                      (("p10", .1), ("p25", .25), ("med", .5), ("p75", .75), ("p90", .9)))


def summary_a(d):
    print("\n==== Part A: facts 1-4")
    print("checks: max |alpha|, |beta| diff", d.check_alpha.max(), d.check_beta.max(),
          "| z windows differing", int((d.check_zwin != 0).sum()),
          "| max rel diff of level EG p vs screen", d.check_eg_rel.max(),
          "| missing in screen", int(d.eg_p_level_cached.isna().sum()))
    for label, x in (("all 287", d), ("non-tracker", d[~d.same])):
        print(f"\n-- {label} (n={len(x)})")
        gap = (x.b_log - x.dollar_H0_close).abs() / x.dollar_H0_close.abs()
        print("1. hedge gap:", q(gap), "| >10%", round((gap > .1).mean(), 3),
              "| sign differs", int((np.sign(x.b_log) != np.sign(x.b_level)).sum()))
        neg = x.b_level < 0
        print(pd.DataFrame({"n": neg.value_counts(), "median gap": gap.groupby(neg).median(),
                            "share >10%": (gap > .1).groupby(neg).mean(),
                            "median corr": x.corr_e0_e1.groupby(neg).median(),
                            "log passes": x.log_passes.groupby(neg).sum()})
              .rename(index={False: "positive", True: "negative"}).round(3).to_string())
        print("   H0 dollar-ratio range / median:", q(x.dollar_H0_range_rel))
        print("   corr(H0, H1 residual):", q(x.corr_e0_e1))
        print("   z windows differ:", int((x.zwin_H0 != x.zwin_H1).sum()), "| H1/H0:", q(x.zwin_H1 / x.zwin_H0))
        print("2. log screen passes:", int(x.log_passes.sum()), "| log EG p > 0.05:", int((x.eg_p_log > .05).sum()),
              "| median p levels", f"{x.eg_p_level.median():.2g}", "logs", f"{x.eg_p_log.median():.2g}")
        print("3. returns beta / b_log:", q(x.beta_ret / x.b_log),
              "| opposite sign", int((np.sign(x.beta_ret) != np.sign(x.b_log)).sum()))
        print("   drift*126/sd(X):", q(x.drift_126_over_sd), "| >1:", round((x.drift_126_over_sd > 1).mean(), 3),
              "| corr(H1, H2):", f"{x.corr_e1_x2.median():.3g}",
              "| z window H2 med", x.zwin_H2.median(), "H1 med", x.zwin_H1.median(), "H2 p90", x.zwin_H2.quantile(.9))
        print("4. transient row0/row20/rows100+:",
              "; ".join(f"{h} {x[f'tr0_{h}'].mean():.2f}/{x[f'tr20_{h}'].mean():.2f}/{x[f'tr100_{h}'].mean():.2f}"
                        for h in ("H0", "H1", "H2")))


def summary_b(rows):
    print("\n==== Part B: fact 5 (all 287)")
    for v in ("H3", "H4a", "H4"):
        q5 = np.array([r[v]["qmin"][4] for r in rows])
        q20 = np.array([r[v]["qmin"][-1] for r in rows])
        any5 = np.array([min(r[v]["qmin"][:5]) for r in rows])
        share = []
        for r in rows:
            ll = np.array(r[v]["ll"], float)
            if np.all(np.isfinite(ll[[1, 5, 20]])) and ll[20] > ll[1]:
                share.append((ll[20] - ll[5]) / (ll[20] - ll[1]))
        mv = np.array([abs(r[v]["dollar_end"][-1] - r[v]["dollar_end"][4]) / max(abs(r[v]["dollar_end"][-1]), 1e-12)
                       for r in rows])
        print(f"{v:4s} Q indefinite after 5: {(q5 < 0).sum()} | at any <=5: {(any5 < 0).sum()} | after 20: {(q20 < 0).sum()}"
              f" | gain still to come after 5: med {np.median(share):.2f} (n={len(share)})"
              f" | end hedge moves >25% 5->20: {np.mean(mv > .25):.2f}")


def summary_c(rows, a):
    print(f"\n==== Part C: fact 6, EM to convergence (tol {TOL_NATS} nats, cap {MAX_ITERS})")
    blog = a.set_index(["formation", "pair"])["b_log"]
    for label, keep in (("all 287", lambda r: True), ("non-tracker", lambda r: not r["same"])):
        R = [r for r in rows if keep(r)]
        print(f"-- {label} (n={len(R)})")
        for v in ("H3", "H4"):
            st = pd.Series([r[v]["status"] for r in R]).value_counts().to_dict()
            it = [r[v]["iters"] for r in R]
            secs = sum(r[v]["secs"] for r in R)
            static = np.array([blog[(r["formation"], r["pair"])] for r in R])
            de = np.array([r[v]["dollar_end"] for r in R])
            print(f"   {v}: status {st} | iterations {q(it)} | cpu {secs / 60:.1f} min")
            print(f"       end dollar hedge vs H1's static b, |diff|/|b|: {q(np.abs(de - static) / np.abs(static))}")
            print(f"       median daily |change| of the dollar hedge: {q([r[v]['dollar_daily_change'] for r in R])}")
            print(f"       beta-state SNR (q_beta var(x) / R): {q([r[v]['snr_beta'] for r in R])}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "notebooks" / "cache"))
    ap.add_argument("--out", default=None, help="output directory (default: --cache)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--limit", type=int, default=None, help="first N pair-folds only (development)")
    args = ap.parse_args()
    cache, out = Path(args.cache), Path(args.out or args.cache)
    folds = [{k: f[k] for k in FORM_FIELDS} for f in pd.read_pickle(cache / "alloc_folds.pkl")["folds"]]
    folds = folds[: args.limit] if args.limit else folds
    par = Parallel(n_jobs=args.jobs)
    a = None
    if "A" in args.parts or "C" in args.parts:
        scr = pd.read_parquet(cache / "day_screen.parquet",
                              columns=["formation", "ticker1", "ticker2", "eg_p", "eg_p_fdr"])
        scr["formation"] = pd.to_datetime(scr["formation"])
        cutoff = scr[scr["eg_p_fdr"] <= 0.05].groupby("formation")["eg_p"].max().to_dict()
        a = pd.DataFrame(par(delayed(part_a)(f, cutoff) for f in folds))
        idx = scr.set_index(["formation", "ticker1", "ticker2"])["eg_p"]
        a["eg_p_level_cached"] = [idx.get((fm, p[0], p[1]), np.nan) for fm, p in zip(a.formation, a.pair)]
        a["check_eg_rel"] = (a.eg_p_level - a.eg_p_level_cached).abs() / a.eg_p_level_cached
        a.to_pickle(out / "loghedge_reg_A.pkl")
        if "A" in args.parts:
            summary_a(a)
    if "B" in args.parts:
        b = par(delayed(part_b)(f) for f in folds)
        pd.to_pickle(b, out / "loghedge_reg_B.pkl")
        summary_b(b)
    if "C" in args.parts:
        c = par(delayed(part_c)(f) for f in folds)
        pd.to_pickle(c, out / "loghedge_reg_C.pkl")
        summary_c(c, a)
    print(f"\npeak RSS of the parent process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
