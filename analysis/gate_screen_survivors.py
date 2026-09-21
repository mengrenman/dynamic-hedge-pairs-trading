"""Run the θ-persistence gate over every one of notebook 10's BH dual-gate survivors.

    python analysis/gate_screen_survivors.py        # ~84 min on 12 cores, resumable

Answers: of the pairs that cleared Engle-Granger + Johansen under Benjamini-Hochberg, how many
actually warrant a *dynamic* hedge? Writes cache/tv_gate_survivors_weekly.parquet. The findings
are in SYNTHESIS.md; the short version is 7.3% classify as time-varying and 1.8% survive a check
that theta lands in (0, 1).

Weekly log prices over each pair's own 2-year formation window — notebook 07's form for real
pairs, and the sample size (~104 bars) its Monte Carlo calibrated. A probe found weekly and
daily-levels disagreeing on half of a small sample, so this is a choice, not a neutral default.

Two things learned the hard way. The series are sliced in the PARENT and only the ~104-point
arrays are dispatched: closing over the unstacked day lake made joblib pickle 1.45 GB to each of
twelve workers, which is why the first attempt produced no checkpoint in six minutes. And it
checkpoints every chunk, because the attempt before that lost an hour when its session ended.
"""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
# Pin BLAS before numpy/statsmodels load. Twelve joblib workers each spawning their own thread
# pool oversubscribes 16 cores badly -- the unpinned first attempt ran ~7x slower than the probe
# predicted, at load average 17. Every notebook in this repo sets these for the same reason.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
ROOT = "/Users/mengren/Documents/projects/coding_finance/dynamic-hedge-pairs-trading"
sys.path.insert(0, ROOT); os.chdir(ROOT + "/notebooks")
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from pairs.stats.tv_cointegration import recommend_hedge

B, FORM_YEARS, CHUNK = 199, 2, 200
OUT = Path("cache/tv_gate_survivors_weekly.parquet")

t_load = time.time()
PX = pd.read_parquet("cache/day_market_bars.parquet", columns=["close"])["close"].unstack("ticker")
surv = pd.read_parquet("cache/day_screen.parquet")
surv = surv[surv["verdict"].eq("pass")].reset_index(drop=True)
surv["formation"] = pd.to_datetime(surv["formation"])

done = set()
if OUT.exists():
    prev = pd.read_parquet(OUT)
    done = set(zip(pd.to_datetime(prev["formation"]), prev["t1"], prev["t2"]))
    print(f"resuming: {len(done)} already done", flush=True)

# slice here, in the parent: each payload is two ~104-point float arrays
jobs, skipped = [], []
for r in surv.itertuples():
    key = (r.formation, r.ticker1, r.ticker2)
    if key in done:
        continue
    meta = {"formation": r.formation, "t1": r.ticker1, "t2": r.ticker2,
            "eg_p": r.eg_p, "eg_p_fdr": r.eg_p_fdr}
    if r.ticker1 not in PX.columns or r.ticker2 not in PX.columns:
        skipped.append({**meta, "err": "absent from the day lake"}); continue
    w = (PX.index > r.formation - pd.DateOffset(years=FORM_YEARS)) & (PX.index <= r.formation)
    d = pd.DataFrame({"y": PX.loc[w, r.ticker1], "x": PX.loc[w, r.ticker2]}).dropna()
    if len(d) < 250:
        skipped.append({**meta, "err": f"only {len(d)} daily bars"}); continue
    wk = np.log(d.resample("W-FRI").last().dropna())
    if len(wk) < 60:
        skipped.append({**meta, "err": f"only {len(wk)} weekly bars"}); continue
    jobs.append(({**meta, "n_weekly": len(wk)}, wk["y"].to_numpy(), wk["x"].to_numpy()))
del PX
print(f"{len(surv)} survivors: {len(jobs)} to run, {len(skipped)} unusable "
      f"(sliced in {time.time()-t_load:.0f}s)", flush=True)


def one(meta, y, x):
    out = dict(meta, err="")
    try:
        v = recommend_hedge(y, x, B=B, seed=0, n_jobs=1)
        out.update(hedge=v.hedge, verdict=v.verdict, theta=v.theta_hat,
                   sigma_eta=v.sigma_eta_hat, p_theta=v.p_theta, p_sigma=v.p_sigma)
    except Exception as exc:
        out["err"] = f"{type(exc).__name__}: {exc}"[:80]
    return out


t0 = time.time()
for i in range(0, len(jobs), CHUNK):
    part = jobs[i:i + CHUNK]
    res = Parallel(n_jobs=12)(delayed(one)(m, y, x) for m, y, x in part)
    df = pd.DataFrame(res + (skipped if i == 0 else []))
    if OUT.exists():
        df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
    df.to_parquet(OUT)
    n, el = i + len(part), time.time() - t0
    print(f"  {n}/{len(jobs)} in {el/60:.1f} min (eta {el/n*(len(jobs)-n)/60:.0f} min)", flush=True)

d = pd.read_parquet(OUT)
ok = d[d["err"].eq("")]
print(f"\n=== {len(ok)} of {len(d)} classified in {(time.time()-t0)/60:.1f} min ===", flush=True)
if len(d) > len(ok):
    print("unusable:", d[~d["err"].eq("")]["err"].str.replace(r"\d+", "N", regex=True).value_counts().to_dict())
print("\nhedge recommended:")
for k, v in ok["hedge"].value_counts().items():
    print(f"  {k:8s} {v:5d}  ({100*v/len(ok):5.1f}%)")
