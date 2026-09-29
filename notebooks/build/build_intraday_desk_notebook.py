"""Build notebooks/pairs_trading_21_intraday_desk_alphas.ipynb (cells only; outputs from execute.py).

    python notebooks/build/build_intraday_desk_notebook.py [--out PATH]
"""
import argparse
import nbformat as nbf
from pathlib import Path

PREREG_PATH = Path(__file__).resolve().parent / "nb21_intraday_desk_preregistration.md"
PREREG_TEXT = PREREG_PATH.read_text()

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "stat-arb", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ═══════════════════════════════ 0. Title ═══════════════════════════════
md(r"""
# Pairs trading on minute bars — IV. The desk at intraday frequency

## `pairs_trading_21_intraday_desk_alphas.ipynb`

Notebook 17 combined eleven daily alphas into one neutralized book and found that a fitted blend did
not clear the equal-weight one. Notebooks 18 to 20 measured what an intraday round trip costs. This
notebook puts the two together: **500 names, thirty-minute bars, horizons from one bar to a day, and
eight pre-registered alphas** built from intraday price and volume — the overnight gap, the opening
move, the last half hour, abnormal volume and range, and the time-of-day seasonality of Heston,
Korajczyk and Sadka — with notebook 17's two daily alphas kept as controls. The question is whether any
of it clears the measured half-spread, how much holding overnight and damping turnover buy, and whether
the intraday alphas add anything once both sets are judged on the same book.

Everything below was fixed before any cell in this notebook ran, in
`notebooks/build/nb21_intraday_desk_preregistration.md`, reproduced verbatim in §0.1. No alpha, sign,
book, formation time, window, period, cost rule, universe rule, blend or decision rule was changed after
the first development-period run; every place this build could not do something exactly as written, and
every implementation choice the pre-registration left open, is logged in §10, Deviations.

**What it found.** Both pre-registered rules were applied as written, in §9, and the hold-out was looked at once.

* **Rule one: the intraday desk does not work.** Book B, equal-weight, phi = 0.25, net of the primary (Roll-spread)
  costs, had a test-period Sharpe of 0.144 (t = 0.29 on 1008 sessions, against the threshold of 2) and a hold-out
  Sharpe of 0.675 on 655 sessions, the same sign as the test period. It failed on the t-statistic. The sign condition
  was met, but the sign does not credit the intraday alphas: the eight-alpha blend's hold-out IC was 0.0019 (s.e.
  0.0089), while `mom12_1` alone had 0.0150 (t = 2.1558), and the book of the two daily controls alone earned a
  hold-out net Sharpe of 0.691 against the eight-alpha book's 0.675. "Failed on the t-statistic, not on the sign"
  would overstate what the sign shows.
* **Rule two: intraday alphas add to the daily ones, in the two periods the rule reads, and the addition did not
  persist.** The eight-alpha equal-weight blend's IC exceeded that of the two daily controls alone by +0.0156 in
  development (s.e. 0.0045; twice that is 0.0090) and by +0.0113 in the test period (s.e. 0.0073), the same sign but a
  ratio of only +1.56. In the hold-out the difference was −0.0124 (s.e. 0.0087, ratio −1.42): the intraday alphas
  subtracted from the daily controls there. A leave-one-out shows a different alpha carrying the addition each period:
  `irev30` in development (+0.0084, t = 3.4567, and −0.0001 in the test period) and `iopen` in the test period
  (+0.0096, t = 2.1907, against +0.0031 in development); in the hold-out no intraday alpha added and `perio`
  subtracted (−0.0059, t = −2.0214).
* **Book B is above the cost cliff on the primary basis, by a margin that is not established.** Its gross Sharpe was
  1.130 over the full span (s.e. 0.253) with 0.263 of the gross traded each session, about a quarter of the book. Net of
  the primary costs (one basis point of commission plus half the Roll spread) it was 0.476; net of the secondary costs
  (notebook 14's day-lake measured costs) it was 0.548, and the two differ by less than a third of a standard error. The
  medians of the two bases are close as well (2.64 and 2.08 bps per name per side in 2018). The profit per bet was 5.48
  bps gross against 3.21 bps of cost on the primary basis (1.71x, hold-out included), and clause (a) of the scorecard
  came out at t = 1.71 on 786 non-overlapping blocks, positive and not significant.
* **Indistinguishable from notebooks 17 and 11.** On the secondary basis Book B's 0.548 compares with 0.596 for notebook
  17's equal-weight book (difference −0.048, 0.14 combined standard errors) and with 0.490 for notebook 11 (+0.058, 0.16
  combined standard errors).
* **Books A and C show the cliff at the desk's own frequency.** Their gross Sharpe ratios over the full span were 3.298
  and 8.608, their net Sharpe ratios on the primary basis −4.951 and −25.411, and their break-even costs 1.138 and 0.736
  bps per side, the second below the one basis point of commission alone.
* **Turnover control was the lever that moved the net Sharpe.** Through the test period Book B's net Sharpe rose from
  −0.310 at phi = 1.00 to 0.433 at 0.25 and 0.518 at 0.10 while its gross Sharpe fell from 2.506 to 1.095 and 0.812.
* **A grouping defect changed the primary-basis numbers.** The first full run grouped the minute rows by `id` after
  filtering them by ticker-and-id pair, which interleaved the minutes of two names that share an id (JCI and TYC, ACE
  and CB, LBTYA and LBTYK) and put thousands of basis points of Roll spread on those six names, where the largest
  correct cell is 30.2 bps. Every primary-basis figure above and below is from the rebuilt caches (§1.3, Deviation 13).
  The test-period and hold-out verdicts were unchanged by the rebuild, but the development, full-span, scorecard and
  cost-basis conclusions of the first run were not, and the conclusions above replace them.

**Data findings that changed the run.** The minute lake's `id` is not unique to a ticker inside a session file (191 of
6,988 ids in the first file read labeled more than one ticker), and an id filter spliced two price series into one and
manufactured a spurious next-day reversal; rows are kept by the ticker-and-id pair instead (§1.3). The first full run
also had BGZ, a triple-inverse fund whose day-lake closes sit near `1e13`, in the universe, where its junk dividend
applied to a real minute-lake price produced window returns of the order of `1e9` and a fictitious equal-weight P&L.
That added a fifth universe exclusion for corrupted day-lake series (1.73 names flagged per month, 16 distinct
tickers, 0.81 per month removed by that rule alone), a rule that treats window returns beyond ±100% as missing, and a
rule that drops day-lake dividends above half the prior close (340 cells). None of them changed an alpha, a sign, a
book or a decision rule; all are logged in §10, Deviations 1, 4 and 13.

**How the periods are used.** Every fit, selection and look happens in the development period. The test
period is shown alongside it from §4 on, because it is walk-forward output that no choice depends on.
The hold-out is computed with everything else but is **displayed only in §9**, once, at the end: the
tables in §4 to §8 stop at the end of the test period.

**Caching.** Everything this notebook writes goes under `notebooks/cache/` (gitignored) with the prefix
`intra_` (`intra_smoke_` in the reduced smoke mode the build uses to debug): the 30-minute panel of the
market layout (`intra_bars30_<year>.parquet` chunks, resumable, and the per-name Roll statistics
`intra_rollstats_<year>.parquet`), the assembled Roll table (`intra_roll.parquet`), the point-in-time
universe (`intra_universe.pkl`), the daily risk-model loop (`intra_loadings.pkl`) and the alpha and
target panels of the three books (`intra_panels.pkl`). Every other cache in this repository, and both
parquet lakes, are read-only from here.
""")

md("### 0.1 The pre-registration, verbatim")
md(PREREG_TEXT)

# ═══════════════════════════════ 1. Setup ═══════════════════════════════
md(r"""
## 1. Setup, data and universe

The day-lake constants are notebook 12's and 17's (`ANN`, `BORROW_BPS`, the liquidity rules, the 252-session
formation window, the 60-session regression window, k = 15 PCA factors), plus this notebook's own
period boundaries and its intraday constants: a 30-minute bar grid anchored at 09:30 Eastern (thirteen
bars on a regular session, seven on a 13:00 early-close session), and a one-basis-point commission.
""")
code(r"""
from pathlib import Path
import os, re, sys, time, pickle, warnings

repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from joblib import Parallel, delayed

import pairs
from pairs import load_daily_bars, liquidity_screen, benjamini_hochberg_fdr
from pairs.market_data.minute_bars import nyse_early_closes
from pairs.market_data.instruments import detect_scaled_instruments
from pairs.stats.microstructure import roll_spread

SMOKE = os.environ.get("INTRA_SMOKE", "") == "1"
PFX = "intra_smoke_" if SMOKE else "intra_"
DAY_LAKE    = Path(os.environ.get("DAY_LAKE", Path.home() / "local/parquet_lake/day_adj"))
MINUTE_LAKE = Path(os.environ.get("MINUTE_LAKE", Path.home() / "local/parquet_lake/minute_adj"))
DAY_MARKET, MIN_MARKET = DAY_LAKE / "all_adjusted", MINUTE_LAKE / "all_adjusted"
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
BUILD_DIR = repo_root / "notebooks" / "build"

START, END = "2004-01-01", "2025-08-13"
if SMOKE:
    FIRST_SESSION, LAST_SESSION = pd.Timestamp("2015-01-02"), pd.Timestamp("2016-12-30")
    DEV_END, TEST_START, TEST_END, HOLD_START = (pd.Timestamp(x) for x in
        ("2015-12-31", "2016-01-04", "2016-06-30", "2016-07-01"))
else:
    FIRST_SESSION, LAST_SESSION = pd.Timestamp("2010-01-04"), pd.Timestamp(END)
    DEV_END, TEST_START, TEST_END, HOLD_START = (pd.Timestamp(x) for x in
        ("2018-12-31", "2019-01-02", "2022-12-30", "2023-01-03"))

MIN_PRICE, MIN_DV, MIN_VOL, MAX_ABS_RET = 5.0, 20e6, 0.15, 1.0
UNIV_N, FORM_DAYS, N_FACTORS, REG_DAYS = 500, 252, 15, 60
CAP, ANN, BORROW_BPS, COMMISSION_BPS = 1_000_000.0, 252, 50.0, 1.0
R2_TRACKER = 0.90
PHI_PRIMARY = 0.25
PHIS = [0.25, 1.0, 0.5, 0.1]                # primary first, then the sensitivity grid
RIDGE_TRAIL_MONTHS = 36
RIDGE_GRID = [0.1, 1.0, 10.0, 100.0]
BAR_MIN, N_SLOT, OPEN_MIN, EARLY_SLOTS = 30, 13, 570, 7
LOOKBACK, MIN_PRIOR = 20, 10                # prior sessions for the same-clock alphas; fewest valid ones accepted
WARM = 25                                   # warm-up sessions before the first formation, history only
N_JOBS = min(16, os.cpu_count() or 4)
N_EW_DRAWS = 5 if SMOKE else 40
N_RIDGE_DRAWS = 3 if SMOKE else 40
RIDGE_PLACEBO_BUDGET_S = 300 if SMOKE else 1800

ALPHAS = ["perio", "irev30", "ovn", "iopen", "ivol_abn", "irange", "rev5", "mom12_1"]
SIGN = {"perio": 1, "irev30": -1, "ovn": -1, "iopen": -1, "ivol_abn": 1, "irange": -1,
        "rev5": -1, "mom12_1": 1}
NA = len(ALPHAS)
DAILY_CONTROLS = ["rev5", "mom12_1"]
NB17_SHARPE, NB17_SE = 0.596, 0.23         # notebook 17's equal-weight book, day-lake measured costs
NB11_SHARPE, NB11_SE = 0.49, 0.26          # notebook 11, as re-evaluated in notebook 14

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})
rng = np.random.default_rng(0)
sharpe = lambda x: x.mean() / x.std(ddof=0) * np.sqrt(ANN) if len(x) > 1 and x.std(ddof=0) > 0 else np.nan
DEVIATIONS = []
print("pairs", pairs.__version__)
print(f"mode: {'SMOKE (reduced span, caches prefixed ' + PFX + ')' if SMOKE else 'FULL (caches prefixed ' + PFX + ')'}")
print(f"formation sessions {FIRST_SESSION.date()} -> {LAST_SESSION.date()}; development to {DEV_END.date()}, "
      f"test {TEST_START.date()} -> {TEST_END.date()}, hold-out from {HOLD_START.date()}")
print(f"{N_JOBS} worker processes; placebo draws: {N_EW_DRAWS} equal-weight, up to {N_RIDGE_DRAWS} ridge "
      f"within {RIDGE_PLACEBO_BUDGET_S} s")
""")

md(r"""
### 1.1 Daily bars, dividends, sessions

Notebook 12's cached total-return day-lake frame (read-only from here): split-adjusted closes, the dividend
recovered from the adjustment factors and credited on the ex-date, winsorized at ±100% a day, exchange test
symbols dropped. The frame is re-sorted date-major once so that every point-in-time window below is a
contiguous slice. The day lake's ticker-to-instrument map is single-valued over the whole span (printed: no
ticker carries more than one `id`), but it is not one-to-one: 14 ids carry two union tickers (printed in §1.3),
eleven of them renames that never trade in the same session and three of them (JCI and TYC, ACE and CB, LBTYA and
LBTYK) pairs of names that trade at the same time under one id. The minute lake is therefore read by the
ticker-and-id pair, not by `id`.
""")
code(r"""
f_bars = CACHE / "day_market_bars.parquet"
if f_bars.exists():
    bars = pd.read_parquet(f_bars)
else:
    bars = load_daily_bars(None, START, END, DAY_MARKET, with_dividends=True)
    bars.to_parquet(CACHE / f"{PFX}day_bars.parquet")
bars_dt = bars.swaplevel("ticker", "datetime").sort_index()
del bars

px  = bars_dt["close"].unstack("ticker")
div = bars_dt["dividend"].unstack("ticker").reindex_like(px).fillna(0.0)
_big_div = div > 0.5 * px.shift(1)                 # a "dividend" above half the prior close is a factor glitch, not a payout
N_DIV_DROPPED = int(_big_div.to_numpy().sum())
div = div.mask(_big_div, 0.0)
dv  = bars_dt["dollar_volume"].unstack("ticker").reindex_like(px)
ret = ((px + div) / px.shift(1) - 1.0).clip(-1.0, 1.0)
TESTS = sorted(t for t in px.columns if re.fullmatch(r"Z[A-Z]ZZT", str(t)))
ret = ret.drop(columns=TESTS, errors="ignore")
sessions = ret.index
id_by_ticker = bars_dt.reset_index().groupby("ticker")["id"].agg(["last", "nunique"])
ID_OF = id_by_ticker["last"].to_dict()
print(f"{ret.shape[1]:,} tickers x {ret.shape[0]:,} sessions, {sessions[0].date()} -> {sessions[-1].date()}")
print(f"exchange test symbols dropped: {len(TESTS)}; tickers mapped to more than one id: "
      f"{int((id_by_ticker['nunique'] > 1).sum())}")

# the nine classic Select Sector SPDRs (notebook 17's reading of 'the nine sector ETFs'), plus SPY as the market
ETF9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
rows = []
for t in ETF9:
    s = ret[t].dropna()
    pos = sessions.get_indexer(s.index)
    gaps = np.diff(pos) if len(pos) > 1 else np.array([0])
    rows.append({"etf": t, "first": s.index.min().date(), "last": s.index.max().date(),
                 "max_gap_sessions": int(gaps.max()), "n_obs": len(s)})
etf_coverage = pd.DataFrame(rows).set_index("etf")
display(etf_coverage)
BAD9 = etf_coverage.index[(etf_coverage["max_gap_sessions"] > 5)
                          | (pd.to_datetime(etf_coverage["last"]) < pd.Timestamp("2025-01-01"))].tolist()
print(f"dropped from the nine for discontinuity: {BAD9}")
ETF9 = [t for t in ETF9 if t not in BAD9]
ETF10 = ["SPY"] + ETF9
assert "SPY" in ret.columns
etf_ret = ret[ETF9]
print(f"risk-model sector ETFs ({len(ETF9)}): {ETF9}; market factor for exposures and exclusions: SPY")
""")

md(r"""
### 1.2 Universe, exclusions and the monthly PCA loadings

Notebook 09's liquidity rules capped at the 500 most traded eligible names — exactly notebook 12's
`universe_at` — rebuilt **monthly**: the list for a calendar month is the screen on the 400 or so calendar
days ending at the last session of the month before, so nothing in it uses the month's own data. The four
pre-registered exclusions are then applied to that list, in this order, each with its count:

1. the ten factor ETFs (SPY and the nine SPDRs), which serve as factors and are never selected;
2. names `detect_scaled_instruments` flags in the 252-session formation window (leveraged and inverse funds),
   judged against every other candidate and the factor ETFs;
3. names whose formation-window daily-return R² against SPY or any one sector SPDR is at least 0.90 (index
   trackers);
4. the 79 tickers frozen in `nb21_etf_exclusions.txt`.

The survivors with a complete formation-window return history feed notebook 12's `eigenportfolios` (15
factors). A name dropped only for lacking a complete window is counted separately. The cap is applied
first and the exclusions second, as pre-registered, so the survivors number fewer than 500.
""")
code(r"""
def universe_raw_at(end, n=UNIV_N):
    # notebook 12's universe_at, verbatim rules; the factor ETFs are removed later, as counted exclusion 1
    start = end - pd.DateOffset(days=int(FORM_DAYS * 1.6))
    w = bars_dt.loc[start + pd.Timedelta(days=1):end]
    st = liquidity_screen(w, min_price=MIN_PRICE, min_dollar_volume=MIN_DV, min_ann_vol=MIN_VOL,
                          max_abs_return=MAX_ABS_RET, exclude=TESTS, top_n=n)
    return list(st[st["eligible"]].index)

def eigenportfolios(R, k=N_FACTORS):
    # verbatim from notebook 12
    sd = R.std(axis=0); ok = sd > 1e-12
    R, sd = R.loc[:, ok], sd[ok]
    C = np.nan_to_num(np.corrcoef(R.to_numpy(), rowvar=False), nan=0.0)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:k]
    Q = vecs[:, idx] / sd.to_numpy()[:, None]
    Q = Q / np.abs(Q).sum(axis=0, keepdims=True)
    return pd.DataFrame(Q, index=R.columns, columns=[f"f{i}" for i in range(len(idx))]), vals[idx] / vals.sum()

def neutralize(w, B):
    # notebook 17's corrected form: project orthogonal to [loadings, 1] JOINTLY, then scale to unit gross
    B1 = np.column_stack([B, np.ones(B.shape[0])])
    BtB = B1.T @ B1 + 1e-8 * np.eye(B1.shape[1])
    w = w - B1 @ np.linalg.solve(BtB, B1.T @ w)
    g = np.abs(w).sum()
    return w / g if g > 1e-12 else w

def robust_z(x):
    # median/MAD z-score winsorized at +/-3; non-finite input stays NaN
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isfinite(x).sum() < 3:
        return np.full_like(x, np.nan)
    med = np.nanmedian(x)
    scale = 1.4826 * np.nanmedian(np.abs(x - med))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.full_like(x, np.nan)
    return np.clip((x - med) / scale, -3.0, 3.0)

FROZEN_ETFS = [ln.strip() for ln in (BUILD_DIR / "nb21_etf_exclusions.txt").read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
print(f"{len(FROZEN_ETFS)} frozen ETF-like tickers read from nb21_etf_exclusions.txt")
""")

code(r"""
# the monthly schedule: one rebuild per calendar month, on the last session before the month starts
all_days = sessions[(sessions >= sessions[sessions.get_loc(sessions[sessions >= FIRST_SESSION][0]) - WARM])
                    & (sessions <= LAST_SESSION)]
MONTHS = pd.period_range(all_days[0].to_period("M"), all_days[-1].to_period("M"), freq="M")
f_univ = CACHE / f"{PFX}universe.pkl"

def build_universe():
    flags, QM, RB = [], {}, {}
    t0 = time.time()
    for m in MONTHS:
        first = sessions[sessions >= m.to_timestamp()][0]
        i_first = sessions.get_loc(first)
        r = sessions[i_first - 1]                                   # last session before the month
        RB[m] = r
        raw = universe_raw_at(r)
        hist_all = ret.iloc[i_first - FORM_DAYS:i_first]            # the 252 sessions ending at r
        px_w = px.iloc[i_first - FORM_DAYS:i_first]
        cands = [t for t in raw if t not in ETF10]
        is_factor = pd.Series([t in ETF10 for t in raw], index=raw)
        is_frozen = pd.Series([t in set(FROZEN_ETFS) for t in raw], index=raw)
        judged = px_w[[t for t in raw if t in px_w.columns] + [t for t in ETF10 if t not in raw]]
        scaled = detect_scaled_instruments(judged)
        is_scaled = pd.Series([t in scaled for t in raw], index=raw)
        H = hist_all[[t for t in raw if t not in ETF10]]
        r2 = pd.DataFrame({e: H.corrwith(hist_all[e]) ** 2 for e in ETF10})
        cnt = pd.DataFrame({e: H.notna().mul(hist_all[e].notna(), axis=0).sum() for e in ETF10})
        r2max = r2.where(cnt >= 120).max(axis=1)                 # best single-ETF R-squared, 120 paired days at least
        is_tracker = (r2max >= R2_TRACKER).fillna(False)
        # (v) data quality, added after the first full run and logged as a deviation: a name whose day-lake
        # series is corrupted in the window (a close above $100,000, or a recovered "dividend" above half the
        # prior close) cannot be judged by rules (ii) and (iii) and its dividends cannot be trusted -- BGZ, a
        # triple-inverse fund with day-lake closes near 1e13, passed every other rule this way
        bd_w = _big_div.iloc[i_first - FORM_DAYS:i_first]
        is_corrupt = pd.Series([(t in px_w.columns) and (float(px_w[t].max()) > 1e5 or bool(bd_w[t].any()))
                                for t in raw], index=raw)
        f = pd.DataFrame({"month": str(m), "ticker": raw, "factor_etf": is_factor.to_numpy(),
                          "frozen_etf": is_frozen.to_numpy(), "scaled": is_scaled.to_numpy(),
                          "tracker": is_tracker.reindex(raw).fillna(False).astype(bool).to_numpy(), "r2_max": r2max.reindex(raw).to_numpy(),
                          "corrupt": is_corrupt.to_numpy()})
        f["excluded"] = f[["factor_etf", "frozen_etf", "scaled", "tracker", "corrupt"]].any(axis=1)
        keep = f.loc[~f["excluded"], "ticker"].tolist()
        hist = hist_all.reindex(columns=keep).dropna(axis=1)
        f["no_window"] = (~f["excluded"]) & (~f["ticker"].isin(hist.columns))
        Q, _ = eigenportfolios(hist)
        QM[m] = Q
        flags.append(f)
    flags = pd.concat(flags, ignore_index=True)
    print(f"universe schedule built for {len(MONTHS)} months in {time.time() - t0:.0f}s")
    return flags, QM, RB

if f_univ.exists():
    with open(f_univ, "rb") as fh:
        UNIV_FLAGS, QM, REBUILD = pickle.load(fh)
    print("loaded the cached universe schedule")
else:
    UNIV_FLAGS, QM, REBUILD = build_universe()
    with open(f_univ, "wb") as fh:
        pickle.dump((UNIV_FLAGS, QM, REBUILD), fh, protocol=5)
UNIV = {m: list(QM[m].index) for m in MONTHS}
UNION_TICKERS = sorted(set().union(*[set(v) for v in UNIV.values()]) | set(ETF10))
print(f"{len(MONTHS)} monthly universes {MONTHS[0]} -> {MONTHS[-1]}; {len(UNION_TICKERS):,} distinct tickers "
      f"including the ten factor ETFs")
""")

code(r"""
# the count excluded under each rule, printed
fl = UNIV_FLAGS
n_m = fl["month"].nunique()
rules = [("(i) factor ETFs (SPY and the nine SPDRs)", "factor_etf"),
         ("(ii) leveraged and inverse funds (detect_scaled_instruments)", "scaled"),
         ("(iii) index trackers (R-squared >= 0.90 against SPY or a SPDR)", "tracker"),
         ("(iv) frozen ETF list (79 tickers)", "frozen_etf"),
         ("(v) corrupted day-lake series (data quality; added after the first full run, logged)", "corrupt")]
rows = []
alone = fl[["factor_etf", "frozen_etf", "scaled", "tracker", "corrupt"]]
for label, c in rules:
    others = [o for _, o in rules if o != c]
    rows.append({"rule": label, "flagged, mean per month": fl.groupby("month")[c].sum().mean(),
                 "removed by this rule alone, mean per month": (alone[c] & ~alone[others].any(axis=1)).sum() / n_m,
                 "distinct tickers ever flagged": fl.loc[fl[c], "ticker"].nunique()})
rows.append({"rule": "any rule", "flagged, mean per month": fl.groupby("month")["excluded"].sum().mean(),
             "removed by this rule alone, mean per month": np.nan,
             "distinct tickers ever flagged": fl.loc[fl["excluded"], "ticker"].nunique()})
excl_table = pd.DataFrame(rows).set_index("rule")
display(excl_table.round(2))
sizes = pd.Series({str(m): len(UNIV[m]) for m in MONTHS})
print(f"raw top-500 lists: {fl.groupby('month').size().mean():.0f} names per month before exclusions; "
      f"dropped only for lacking a complete formation window: {fl.groupby('month')['no_window'].sum().mean():.1f} per month")
print(f"final monthly universe: mean {sizes.mean():.0f} names, min {sizes.min()}, max {sizes.max()}")
""")

md(r"""
### 1.3 Reading the minute lake

The market layout holds one file per session with every ticker (about 1.5 million minute rows). Reading it
name by name through the sidecar index costs seconds per session for 500 names, so each session file is
instead read **whole** (eight columns), filtered with Arrow to the session's ticker-and-id pairs, moved from UTC
to Eastern, cut to the regular session (09:30 to 16:00, or to 13:00 on the NYSE early closes that
`nyse_early_closes` lists) and aggregated to 30-minute bars labeled by their start minute: open is the first
traded minute's open, high and low the extremes, close the last traded minute's close, volume the sum. A bar
with no trade repeats the previous close with zero volume, forward-filled **within the session only**;
minutes before a name's first trade of the session stay missing. Prices are the lake's split-adjusted
columns. **The lake's `id` cannot be trusted on its own:** inside a session file a few percent of ids label two tickers (the cell below counts them in the first session file read; in the 2015-01-02 file the id of `BBT` also labels `BHLB`, that of `BBBY` also labels `OSTK`), so an id filter silently splices two price series into one and manufactures a spurious next-day reversal (an information coefficient of the order of a fifth on a first trial, gone once fixed). Rows are therefore kept only where the ticker and the id are the pair the day lake resolved, **and every grouping of those rows (the 30-minute bars, the Roll statistics) is by the same pair**: the filter alone is not enough, because two kept tickers can share one id inside a session (JCI and TYC, ACE and CB, LBTYA and LBTYK), and grouping the filtered rows by `id` would interleave their minutes into one series. The first full run of this notebook did exactly that, and the six names' Roll spreads came out at thousands of basis points; the results below are from the rebuilt caches (Deviation 13). The pairs kept for a session are those of the point-in-time universes of its month and of the two
following months (so a name's history is on hand the month it enters) plus the ten factor ETFs. The same
pass records, per name and session, the sufficient statistics of the Roll estimator from the traded
one-minute closes (the sums of the lagged and lead price changes and of their product, within the session
only), so that the per-name, per-calendar-year Roll spread of the cost basis comes out of the same read.

**A note on clock times.** An instant is named by its clock time, and the price *at* an instant is the close of
the 30-minute bar that ends there, that is, the bar labeled thirty minutes earlier. Book B forms at 15:30, the
close of the bar labeled 15:00; Book A forms at 10:30, the close of the bar labeled 10:00; Book C forms at the
close of every bar from 10:00 to 15:30. On a 13:00 early-close session the same relative instant, thirty
minutes before the close (12:30), plays the role of 15:30.
""")
code(r"""
early_days = nyse_early_closes(all_days[0], all_days[-1])
EARLY_SET = set(early_days)
def month_keys(m):
    # the minute lake's `id` is NOT unique to a ticker inside a session file (215 of 7,295 ids in one 2015 file
    # carry two tickers: BBT's id also labels BHLB, BBBY's labels OSTK), so rows are kept by the ticker|id pair
    # the day lake resolved, never by id alone
    tk = set(ETF10)
    for k in range(3):
        if m + k in UNIV:
            tk |= set(UNIV[m + k])
    return sorted(f"{t}|{ID_OF[t]}" for t in tk if t in ID_OF)
MKEYS = {m: month_keys(m) for m in MONTHS}
# the dense axis is the (ticker, id) PAIR, not the id: the first full run showed two names whose id is
# shared with junk-priced ETN rows in the day lake (window returns of 2e8 and 2e9 once their prices and
# dividends were merged onto one code); keying by the pair keeps every instrument's own series
ALL_IDS = sorted({k for v in MKEYS.values() for k in v})            # "ticker|id" keys
CODE = {i: j for j, i in enumerate(ALL_IDS)}
N_ID = len(ALL_IDS)
code_of_ticker = {t: CODE[f"{t}|{ID_OF[t]}"] for t in UNION_TICKERS if f"{t}|{ID_OF.get(t)}" in CODE}
print(f"{len(all_days):,} sessions read from the minute lake ({all_days[0].date()} -> {all_days[-1].date()}, the first "
      f"{WARM} are warm-up history), {N_ID:,} distinct (ticker, id) instruments, {len(early_days)} early-close sessions")

def session_bars(path, date, ids, early):
    # one session file -> (30-minute bars on the full session grid, Roll sufficient statistics)
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    cols = ["datetime", "id", "ticker", "open_split", "high_split", "low_split", "close_split", "volume_split"]
    tb = pq.read_table(path, columns=cols)
    key = pc.binary_join_element_wise(pc.cast(tb["ticker"], pa.string()), pc.cast(tb["id"], pa.string()), "|")
    tb = tb.filter(pc.is_in(key, value_set=pa.array(list(ids))))
    if tb.num_rows == 0:
        return None, None
    df = tb.to_pandas()
    ts = pd.Timestamp(date)
    off = int(-pd.Timestamp(ts.year, ts.month, ts.day, 12).tz_localize("US/Eastern").utcoffset().total_seconds() // 60)
    et = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute).to_numpy() - off
    close_min = 13 * 60 if early else 16 * 60
    keep = ((et >= 570) & (et < close_min) & np.isfinite(df["close_split"].to_numpy())
            & (df["datetime"].dt.normalize() == ts).to_numpy())
    df, et = df[keep], et[keep]
    if len(df) == 0:
        return None, None
    # the group is the ticker|id PAIR: two kept tickers can share one id inside a session (JCI and TYC, ACE and CB,
    # LBTYA and LBTYK), and grouping by id alone would interleave their minutes into one series
    pair = df["ticker"].astype(str).to_numpy() + "|" + df["id"].astype(str).to_numpy()
    g, uniq_pair = pd.factorize(pair, sort=False)
    uniq = np.array([p_.split("|")[-1] for p_ in uniq_pair], dtype=object)
    utick = np.array([p_.rsplit("|", 1)[0] for p_ in uniq_pair], dtype=object)
    order = np.lexsort((et, g))
    g, et = g[order], et[order]
    c = df["close_split"].to_numpy(float)[order]
    o = df["open_split"].to_numpy(float)[order]; h = df["high_split"].to_numpy(float)[order]
    l = df["low_split"].to_numpy(float)[order]; v = df["volume_split"].to_numpy(float)[order]
    o = np.where(np.isfinite(o), o, c); h = np.where(np.isfinite(h), h, c); l = np.where(np.isfinite(l), l, c)
    v = np.where(np.isfinite(v), v, 0.0)
    G, nb = len(uniq), (close_min - 570) // 30
    k = (et - 570) // 30
    key = g * 32 + k
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    end = np.r_[start[1:], len(key)] - 1
    gg, kk = key[start] // 32, key[start] % 32
    Om, Hm, Lm, Cm = (np.full((G, nb), np.nan) for _ in range(4))
    Vm = np.zeros((G, nb))
    Om[gg, kk], Cm[gg, kk] = o[start], c[end]
    Hm[gg, kk], Lm[gg, kk] = np.maximum.reduceat(h, start), np.minimum.reduceat(l, start)
    Vm[gg, kk] = np.add.reduceat(v, start)
    valid = np.isfinite(Cm)
    fill = np.maximum.accumulate(np.where(valid, np.arange(nb)[None, :], 0), axis=1)
    Cf = np.take_along_axis(Cm, fill, axis=1)                 # forward fill inside the session only
    empty = ~valid & np.isfinite(Cf)
    Om, Hm, Lm = np.where(empty, Cf, Om), np.where(empty, Cf, Hm), np.where(empty, Cf, Lm)
    bars = pd.DataFrame({
        "session": ts,
        "bar_start": ts + pd.to_timedelta(570 + 30 * np.tile(np.arange(nb), G), unit="m"),
        "id": np.repeat(uniq, nb), "ticker": np.repeat(utick, nb),
        "open": Om.ravel().astype(np.float32), "high": Hm.ravel().astype(np.float32),
        "low": Lm.ravel().astype(np.float32), "close": Cf.ravel().astype(np.float32),
        "volume": Vm.ravel().astype(np.float32)})
    # Roll sufficient statistics: consecutive traded-minute price changes, pairs kept inside the name-session
    d = np.r_[np.nan, np.diff(c)]
    first = np.r_[True, g[1:] != g[:-1]]
    d[first] = np.nan
    lag = np.r_[np.nan, d[:-1]]
    pm = np.isfinite(d) & np.isfinite(lag)
    gp = g[pm]
    roll = pd.DataFrame({
        "session": ts, "id": uniq, "ticker": utick,
        "n_pairs": np.bincount(gp, minlength=G).astype(np.int64),
        "s_ab": np.bincount(gp, weights=(d * lag)[pm], minlength=G),
        "s_a": np.bincount(gp, weights=d[pm], minlength=G),
        "s_b": np.bincount(gp, weights=lag[pm], minlength=G),
        "s_p": np.bincount(g, weights=c, minlength=G), "n_p": np.bincount(g, minlength=G).astype(np.int64)})
    return bars, roll

def session_task(d):
    m = d.to_period("M")
    return (MIN_MARKET / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}.parquet", d, MKEYS[m], d in EARLY_SET)

tasks = [session_task(d) for d in all_days]
n_full = int(((sessions >= sessions[sessions.get_loc(sessions[sessions >= pd.Timestamp("2010-01-04")][0]) - WARM])
              & (sessions <= pd.Timestamp(END))).sum())
_ids = pq.read_table(tasks[0][0], columns=["id", "ticker"]).to_pandas()
_multi = int((_ids.groupby("id")["ticker"].nunique() > 1).sum())
print(f"id uniqueness check on the {tasks[0][1].date()} file: {_multi:,} of {_ids['id'].nunique():,} ids label more than one ticker; "
      f"{int((_ids.groupby('ticker')['id'].nunique() > 1).sum())} tickers carry more than one id")
t0 = time.time()
for tk in tasks[:10]:
    b_, r_ = session_bars(*tk)
per_session = (time.time() - t0) / 10
print(f"timing: first ten sessions read one after another in {per_session * 10:.1f}s ({per_session:.2f}s per session)")
print(f"projected minute-lake build for the full span ({n_full:,} sessions) on {N_JOBS} workers: "
      f"{per_session * n_full / N_JOBS / 60:.1f} minutes; sequentially {per_session * n_full / 60:.0f} minutes")
_pair_tab = pd.Series({t: ID_OF[t] for t in UNION_TICKERS if t in ID_OF}).rename("id").reset_index().rename(columns={"index": "ticker"})
_shared = _pair_tab.groupby("id")["ticker"].agg(list)
_shared = _shared[_shared.map(len) > 1]
print(f"day-lake ids that carry more than one union ticker: {len(_shared)} ({'; '.join('/'.join(v) for v in _shared)})")
print(f"this run builds {len(tasks):,} sessions, projected {per_session * len(tasks) / N_JOBS / 60:.1f} minutes on {N_JOBS} workers")
""")

code(r"""
def chunk_files(y):
    return CACHE / f"{PFX}bars30_{y}.parquet", CACHE / f"{PFX}rollstats_{y}.parquet"

years = sorted({d.year for d in all_days})
t_build = time.time()
built, reused = [], []
for y in years:
    fb, fr = chunk_files(y)
    if fb.exists() and fr.exists():
        reused.append(y); continue
    ty = time.time()
    todo = [tk for tk in tasks if tk[1].year == y]
    res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(session_bars)(*tk) for tk in todo)
    bars_y = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    roll_y = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    bars_y.to_parquet(fb); roll_y.to_parquet(fr)
    built.append(y)
    print(f"  {y}: {len(todo)} sessions, {len(bars_y):,} bar rows, {time.time() - ty:.0f}s")
print(f"minute-lake pass: built {built if built else 'nothing'}, reused cached chunks for {reused if reused else 'none'}; "
      f"{time.time() - t_build:.0f}s this run")
""")

md(r"""
#### The per-name, per-year Roll spread

`pairs.stats.microstructure.roll_spread` pools the within-session pairs of consecutive price changes over a
window into one covariance and reports twice the square root of its negative, in basis points of the mean
price; it is NaN when the pooled covariance is not negative or there are fewer than 30 pairs. Because the
estimator pools, its sufficient statistics add across sessions, so the calendar-year spread of a name can be
assembled from the per-session statistics recorded above without holding a year of minute closes for every
name at once. The cell below assembles the table and **checks it against `roll_spread` itself** on a handful
of names over a few weeks of raw minute closes read by an independent route.
""")
code(r"""
f_roll = CACHE / f"{PFX}roll.parquet"
rs = pd.concat([pd.read_parquet(chunk_files(y)[1]) for y in years], ignore_index=True)
rs["year"] = rs["session"].dt.year

def roll_from_stats(g):
    n = g["n_pairs"].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = g["s_ab"].sum() / n - (g["s_a"].sum() / n) * (g["s_b"].sum() / n)
        mean_p = g["s_p"].sum() / g["n_p"].sum()
        spread = 2.0 * np.sqrt(-cov)
    ok = (n >= 30) & (cov < 0)
    return pd.Series({"roll_bps": spread / mean_p * 1e4 if ok else np.nan, "n_obs": n})

# the unit of a Roll cell is the ticker|id pair; grouping by id alone would pool two names that share an id
assert not rs.duplicated(["session", "ticker", "id"]).any(), "a (session, ticker, id) row appears twice"
grp = rs.groupby(["ticker", "id", "year"])
roll_tab = grp[["n_pairs", "s_ab", "s_a", "s_b", "s_p", "n_p"]].sum()
with np.errstate(invalid="ignore", divide="ignore"):
    n_ = roll_tab["n_pairs"]
    cov_ = roll_tab["s_ab"] / n_ - (roll_tab["s_a"] / n_) * (roll_tab["s_b"] / n_)
    roll_tab["roll_bps"] = (2.0 * np.sqrt(-cov_) / (roll_tab["s_p"] / roll_tab["n_p"]) * 1e4).where((n_ >= 30) & (cov_ < 0))
roll_tab["n_obs"] = n_
roll_tab = roll_tab[["roll_bps", "n_obs"]].reset_index()
_sess_year = rs.groupby("year")["session"].nunique()
_cap = roll_tab["year"].map(_sess_year) * 391
assert (roll_tab["n_obs"] <= _cap).all(), "a (name, year) Roll cell has more pairs than sessions x 391"
print(f"Roll-cell check: largest n_obs / (sessions x 391) = {(roll_tab['n_obs'] / _cap).max():.3f}; "
      f"{int((rs.groupby(['session', 'id'])['ticker'].nunique() > 1).sum()):,} (session, id) cells carry two kept tickers, "
      f"each now its own Roll series; largest Roll cell {roll_tab['roll_bps'].max():.1f} bps")
roll_tab.to_parquet(f_roll)
ROLL_MED = roll_tab.groupby("year")["roll_bps"].median()
print(f"Roll table: {len(roll_tab):,} (ticker, id, year) cells, {roll_tab['roll_bps'].notna().mean():.1%} measured "
      f"(the rest take that year's median); median Roll spread {roll_tab['roll_bps'].median():.2f} bps")
display(ROLL_MED.to_frame("median Roll spread, bps").T.round(2))

# validation: the pooled statistics against roll_spread on raw minute closes, read by a different route
val_days = [d for d in all_days if d.year == all_days[WARM].year][:15]
val_names = list(dict.fromkeys(["SPY"] + [t for t in UNIV[val_days[0].to_period("M")] if t in ID_OF][:5]))
val_ids = {ID_OF[t]: t for t in val_names}
val_tickers = set(val_names)
frames = []
for d in val_days:
    tb = pq.read_table(MIN_MARKET / f"{d.year:04d}/{d.month:02d}/{d.day:02d}.parquet",
                       columns=["datetime", "id", "ticker", "close_split"])
    df = tb.to_pandas()
    df = df[df["id"].isin(list(val_ids)) & df["ticker"].isin(val_tickers)].copy()
    et = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.tz_localize(None)
    mod = et.dt.hour * 60 + et.dt.minute
    cm = 13 * 60 if d in EARLY_SET else 16 * 60
    df = df[(mod >= 570) & (mod < cm) & (et.dt.normalize() == d) & df["close_split"].notna()].copy()
    df["et"], df["session"] = et[df.index], d
    frames.append(df)
raw_min = pd.concat(frames).sort_values(["id", "et"])
vt = rs[rs["session"].isin(val_days) & rs["id"].isin(list(val_ids)) & rs["ticker"].isin(val_tickers)]
rows = []
for i_, t in val_ids.items():
    s_ = raw_min[raw_min["id"] == i_]
    a = roll_spread(pd.Series(s_["close_split"].to_numpy(), index=pd.DatetimeIndex(s_["et"])),
                    as_bps=True, session=s_["session"].to_numpy(), every=1)
    b = roll_from_stats(vt[vt["id"] == i_])["roll_bps"]
    rows.append({"ticker": t, "roll_spread(minute closes), bps": a, "assembled from statistics, bps": b})
val = pd.DataFrame(rows).set_index("ticker")
display(val.round(4))
gap = (val.iloc[:, 0] - val.iloc[:, 1]).abs().max()
print(f"Roll validation over {len(val_days)} sessions and {len(val)} names: largest absolute difference {gap:.2e} bps "
      f"({'agree' if (gap < 1e-6 or np.isnan(gap)) else 'DISAGREE'}); NaN cells agree: "
      f"{bool((val.iloc[:, 0].isna() == val.iloc[:, 1].isna()).all())}")
""")

md(r"""
### 1.4 The dense 30-minute panel

The yearly chunks are scattered into dense arrays indexed by session, bar slot (0 is the bar labeled 09:30) and
instrument code, so that every same-clock window in the alphas below is a slice. The range-so-far array (the
running high minus the running low from the open) is built once. Dividends, needed for the overnight leg, come
from the day lake's `dividend` column by ticker and are placed on the same instrument codes.
""")
code(r"""
S_W = all_days
S = len(S_W)
NSLOT = np.where(S_W.isin(early_days), EARLY_SLOTS, N_SLOT)
Cc, Oo, Hh, Ll, Vv = (np.full((S, N_SLOT, N_ID), np.nan, np.float32) for _ in range(5))
t0 = time.time()
all_id_index = pd.Index(ALL_IDS)
for y in years:
    df = pd.read_parquet(chunk_files(y)[0], columns=["session", "bar_start", "ticker", "id", "open", "high", "low", "close", "volume"])
    si = S_W.get_indexer(df["session"])
    k = (((df["bar_start"] - df["session"]).dt.total_seconds().to_numpy() // 60) - OPEN_MIN).astype(int) // BAR_MIN
    ci = all_id_index.get_indexer(df["ticker"].astype(str) + "|" + df["id"].astype(str))
    ok = (si >= 0) & (ci >= 0)
    si, k, ci = si[ok], k[ok], ci[ok]
    for arr, col in ((Cc, "close"), (Oo, "open"), (Hh, "high"), (Ll, "low"), (Vv, "volume")):
        arr[si, k, ci] = df[col].to_numpy()[ok]
RNG = np.fmax.accumulate(Hh, axis=1) - np.fmin.accumulate(Ll, axis=1)
for s in np.flatnonzero(NSLOT == EARLY_SLOTS):
    RNG[s, EARLY_SLOTS:, :] = np.nan
del Hh, Ll
DIVc = np.zeros((S, N_ID), np.float32)
dsub = div.reindex(S_W)
TICK_OF_CODE = {}
for t in UNION_TICKERS:
    if t in code_of_ticker:
        DIVc[:, code_of_ticker[t]] += dsub[t].to_numpy(np.float32)
        TICK_OF_CODE[code_of_ticker[t]] = t
ETF_CODES = np.array([code_of_ticker[t] for t in ETF10])           # SPY first, then the nine SPDRs
print(f"dense panel: {S:,} sessions x {N_SLOT} slots x {N_ID:,} ids, {Cc.nbytes / 1e9:.2f} GB per price array, "
      f"filled in {time.time() - t0:.0f}s; {100 * np.isfinite(Cc).mean():.1f}% of cells populated")
""")

code(r"""
# first-bar close-to-open check on the names whose id is shared inside a session: a spliced series would show
# open and close from different securities
def first_bar_ratio(t):
    c = code_of_ticker.get(t)
    if c is None:
        return None
    o_, c_ = Oo[:, 0, c], Cc[:, 0, c]
    ok_ = np.isfinite(o_) & np.isfinite(c_) & (o_ > 0)
    if ok_.sum() == 0:
        return {"ticker": t, "sessions with a first bar": 0, "share |C/O-1| > 20%": np.nan}
    return {"ticker": t, "sessions with a first bar": int(ok_.sum()),
            "share |C/O-1| > 20%": float((np.abs(c_[ok_] / o_[ok_] - 1) > 0.20).mean())}
_fb = [r for r in (first_bar_ratio(t) for t in ["JCI", "TYC", "ACE", "CB", "LBTYA", "LBTYK"]) if r]
display(pd.DataFrame(_fb).set_index("ticker").round(4))
""")

code(r"""
# coverage by year: sessions, early closes handled, names per session, bars per session, names with a price at each instant
span = np.flatnonzero((S_W >= FIRST_SESSION) & (S_W <= LAST_SESSION))
kB = NSLOT - 2                                                    # the 15:30 instant (12:30 on early closes)
rows = []
for s in span:
    m = S_W[s].to_period("M")
    cd = np.array([code_of_ticker[t] for t in UNIV[m] if t in code_of_ticker])
    rows.append({"year": S_W[s].year, "early": NSLOT[s] == EARLY_SLOTS, "names": len(cd), "slots": NSLOT[s],
                 "px_1030": np.isfinite(Cc[s, 1, cd]).mean(), "px_1530": np.isfinite(Cc[s, kB[s], cd]).mean(),
                 "px_close": np.isfinite(Cc[s, NSLOT[s] - 1, cd]).mean()})
cov = pd.DataFrame(rows)
coverage = cov.groupby("year").agg(sessions=("names", "size"), early_close=("early", "sum"),
                                   names_per_session=("names", "mean"), bars_per_session=("slots", "mean"),
                                   with_price_at_1030=("px_1030", "mean"), with_price_at_1530=("px_1530", "mean"),
                                   with_price_at_close=("px_close", "mean"))
display(coverage.round(3))
print(f"{int(cov['early'].sum())} early-close sessions in the span, each on the seven-bar grid")

# sanity checks on the calendar: a regular session whose last six bars carry almost no volume would be a missed early close
tot = np.nansum(Vv[span][:, :, :], axis=(1, 2))
late = np.nansum(Vv[span][:, EARLY_SLOTS:, :], axis=(1, 2))
share_late = pd.Series(late / np.where(tot > 0, tot, np.nan), index=S_W[span])
reg = share_late[NSLOT[span] == N_SLOT]
print(f"share of session volume after 13:00 on regular sessions: min {reg.min():.3f}, median {reg.median():.3f}; "
      f"sessions below 0.05: {int((reg < 0.05).sum())} {list(reg.index[reg < 0.05].date)[:5]}")
raw_late = []
for d in early_days[(early_days >= FIRST_SESSION) & (early_days <= LAST_SESSION)]:
    if d not in EARLY_SET or d not in S_W:
        continue
    df = pq.read_table(MIN_MARKET / f"{d.year:04d}/{d.month:02d}/{d.day:02d}.parquet",
                       columns=["datetime", "volume_split"]).to_pandas()
    et = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    mod = et.dt.hour * 60 + et.dt.minute
    v = df["volume_split"].fillna(0)
    raw_late.append(float(v[(mod >= 13 * 60) & (mod < 16 * 60)].sum() / v[(mod >= 570) & (mod < 16 * 60)].sum()))
if raw_late:
    print(f"raw lake volume between 13:00 and 16:00 on the {len(raw_late)} early-close sessions, as a share of regular-hours "
          f"volume: max {max(raw_late):.4f}, median {np.median(raw_late):.4f}")
""")

# ═══════════════════════════════ 2. Risk model and targets ═══════════════════════════════
md(r"""
## 2. Risk model and hedged targets

**Risk model.** Notebook 12's monthly PCA loadings from daily returns (15 eigenportfolios of the formation
window), the nine sector SPDRs as explicit factors, and size (log dollar volume, a style column used for
neutralization only, since it has no return series). For every session, each name's trailing 60 daily returns
**ending at the prior close** are regressed jointly, with an intercept, on the 15 PCA factor returns and the
nine SPDR returns; the loadings are the coefficients, and the daily residual of that regression is what the
daily control `rev5` sums. The intercept is kept out of everything that is forecast.

**Hedged targets.** The target of a book is the name's total return over the book's window minus its factor
betas times the factor returns over *the same window*: the SPDRs' own returns over that window, and the PCA
portfolios' returns (the eigenportfolio weights applied to the names' returns over that window, missing names
counted as zero for the factor and left missing for themselves). There is **no in-sample intercept**. The
window is 15:30 to the next session's 15:30 for Book B (with the dividend of the next session's ex-date
credited to the overnight leg), 10:30 to the close for Book A, and the next 30-minute bar for Book C.
""")
code(r"""
f_load = CACHE / f"{PFX}loadings.pkl"
retU = ret.reindex(columns=UNION_TICKERS)
mom_panel = np.expm1(np.log1p(retU).rolling(231).sum().shift(21))        # sessions t-251..t-21, notebook 17's definition
dv21 = dv.reindex(columns=UNION_TICKERS).rolling(21).mean()
span = np.flatnonzero((S_W >= FIRST_SESSION) & (S_W <= LAST_SESSION))

if f_load.exists():
    with open(f_load, "rb") as fh:
        LOAD, N_SKIPPED, R2_BY_SESSION = pickle.load(fh)
    print(f"loaded the cached risk-model loop: {len(LOAD):,} sessions")
else:
    t0 = time.time()
    LOAD, N_SKIPPED, R2_BY_SESSION = {}, 0, {}
    for s in span:
        d = S_W[s]; i = sessions.get_loc(d); Q = QM[d.to_period("M")]
        win = ret.iloc[i - REG_DAYS:i].reindex(columns=Q.index).dropna(axis=1)       # 60 sessions ending at the prior close
        if win.shape[1] < 50:
            N_SKIPPED += 1
            continue
        cols = win.columns
        Qw = Q.reindex(cols).to_numpy()
        Rw = win.to_numpy()
        Fm = np.column_stack([np.ones(len(Rw)), Rw @ Qw, etf_ret.iloc[i - REG_DAYS:i].to_numpy()])
        beta_full, *_ = np.linalg.lstsq(Fm, Rw, rcond=None)
        resid = Rw - Fm @ beta_full
        r2 = 1.0 - resid.var(0) / np.maximum(Rw.var(0), 1e-18)
        size_z = robust_z(np.log(dv21.iloc[i - 1].reindex(cols).to_numpy()))
        mom_raw = mom_panel.iloc[i - 1].reindex(cols).to_numpy()
        bpca, betf = beta_full[1:1 + N_FACTORS], beta_full[1 + N_FACTORS:]
        LOAD[s] = {"tick": np.array(cols), "codes": np.array([code_of_ticker[t] for t in cols]),
                   "Qw": Qw, "bpca": bpca, "betf": betf,
                   "B": np.column_stack([bpca.T, betf.T, np.nan_to_num(size_z)]).astype(np.float32),
                   "rev5": resid[-5:].sum(0), "mom": mom_raw, "momz": robust_z(mom_raw), "sizez": size_z}
        R2_BY_SESSION[s] = float(np.nanmean(r2))
    with open(f_load, "wb") as fh:
        pickle.dump((LOAD, N_SKIPPED, R2_BY_SESSION), fh, protocol=5)
    print(f"risk-model loop: {len(LOAD):,} sessions, {N_SKIPPED} skipped for too few names, {time.time() - t0:.0f}s")

r2_s = pd.Series(R2_BY_SESSION); r2_s.index = S_W[r2_s.index]
n_names = pd.Series({S_W[s]: len(L["codes"]) for s, L in LOAD.items()})
print(f"factor count in the risk model: {N_FACTORS} PCA + {len(ETF9)} SPDRs = {N_FACTORS + len(ETF9)} return factors, "
      f"plus one size column for neutralization")
print(f"mean joint-model R-squared over the 60-session regressions: development {r2_s[r2_s.index <= DEV_END].mean():.3f}, "
      f"all sessions shown here {r2_s[r2_s.index <= TEST_END].mean():.3f}")
print(f"names per session in the risk model: mean {n_names.mean():.0f}, min {n_names.min()}, max {n_names.max()}")
""")

md(r"""
## 3. The eight alphas

Each alpha is computed at the book's formation instant from data through the close of the formation bar and no
later, standardized cross-sectionally at every formation (median and MAD, winsorized at ±3) and then multiplied
by its pre-registered sign, so that a positive coefficient or a positive IC always means "agrees with the
pre-registration". The eight, and the clock they run on:

* `perio` (+): the mean of the total returns of the same clock window over the previous 20 sessions (for
  Book B the 24-hour windows starting 15:30 on each of those sessions, for Book A 10:30 to the close, for
  Book C the same 30-minute bar).
* `irev30` (−): the factor-hedged return of the formation bar; `iopen` (−): the factor-hedged return from the
  09:30 open to the formation instant; both use the same betas and the same-interval factor returns as the
  target.
* `ovn` (−): prior close to today's 09:30 open, with any dividend credited.
* `ivol_abn` (+): log of the formation bar's volume over the mean volume of the same clock bar on the prior 20
  sessions; `irange` (−): log of the range so far today, high minus low from the open, over its mean at the same
  clock time on the prior 20 sessions.
* `rev5` (−) and `mom12_1` (+): notebook 17's daily controls, taken at the prior close from the same daily
  data and the same risk-model residual (five-session residual return; return over sessions t-251 to t-21).

A same-clock alpha needs at least ten valid sessions among the previous 20 and is missing otherwise; a
non-finite value (a log of zero volume or zero range) is missing. The construction is in one function that
takes the arrays it may look at as an argument, so that the look-ahead audit two cells down can hand it
arrays with everything after the formation instant blanked and check that the alphas do not move.
""")
code(r"""
SIGN_VEC = np.array([SIGN[a] for a in ALPHAS], dtype=float)[:, None]

def robust_z_rows(X):
    # median/MAD z-score winsorized at +/-3, row by row; rows with fewer than three finite values stay NaN
    X = np.where(np.isfinite(X), X, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=1, keepdims=True)
        scale = 1.4826 * np.nanmedian(np.abs(X - med), axis=1, keepdims=True)
        Z = np.clip((X - med) / scale, -3.0, 3.0)
    bad = (np.isfinite(X).sum(1) < 3) | ~np.isfinite(scale[:, 0]) | (scale[:, 0] < 1e-12)
    Z[bad] = np.nan
    return Z

def cap(a):
    # notebook 12's daily rule at every horizon: a return beyond +/-100% is a data error, treated as missing
    return np.where(np.abs(a) > MAX_ABS_RET, np.nan, a)

def make_D(C, O, V, RG, DIV, NS):
    # the arrays a session's construction may look at, plus the same-clock window returns by start session
    n_s = C.shape[0]
    ar = np.arange(n_s)
    D = {"C": C, "O": O, "V": V, "RNG": RG, "DIV": DIV, "NS": NS}
    CB = C[ar, NS - 2, :].astype(np.float64)                    # price at the 15:30 instant (12:30 on early closes)
    WB = np.full((n_s, C.shape[2]), np.nan)
    DIVw = np.where(DIV[1:] > 0.5 * CB[:-1], 0.0, DIV[1:])      # a dividend above half the price is a glitch
    WB[:-1] = (CB[1:] + DIVw) / CB[:-1] - 1.0                   # the 24-hour window that starts on session s
    with np.errstate(all="ignore"):
        WA = C[ar, NS - 1, :].astype(np.float64) / C[:, 1, :].astype(np.float64) - 1.0   # 10:30 to the close
    n_cap = 0
    for arr in (WA, WB):
        bad = np.isfinite(arr) & (np.abs(arr) > MAX_ABS_RET)
        n_cap += int(bad.sum()); arr[bad] = np.nan
    D["WA"], D["WB"], D["n_capped"] = WA, WB, n_cap
    return D

def prior_mean(arr):
    ok = np.isfinite(arr)
    cnt = ok.sum(0)
    m = np.where(cnt > 0, np.where(ok, arr, 0.0).sum(0) / np.maximum(cnt, 1), np.nan)
    return np.where(cnt >= MIN_PRIOR, m, np.nan)

def hedge(X, Fe, L):
    # interval returns of the names (K x n) minus betas times the factor returns over the same interval
    Fp = np.nan_to_num(X) @ L["Qw"]
    return X - (Fp @ L["bpca"] + Fe @ L["betf"])

def session_rows(s, L, D):
    # everything one session contributes to the three books: signed z-scores, hedged targets, raw window returns
    codes, n, nsl = L["codes"], len(L["codes"]), int(D["NS"][s])
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    C, O0 = f64(D["C"][s][:, codes]), f64(D["O"][s, 0][codes])
    V, RG = f64(D["V"][s][:, codes]), f64(D["RNG"][s][:, codes])
    Ce, Oe0 = f64(D["C"][s][:, ETF_CODES]), f64(D["O"][s, 0][ETF_CODES])
    lo = s - LOOKBACK
    with np.errstate(all="ignore"):
        Vm = prior_mean(f64(D["V"][lo:s][:, :, codes]))
        Rm = prior_mean(f64(D["RNG"][lo:s][:, :, codes]))
        ivol = np.log(V / Vm)
        irng = np.log(RG / Rm)
        Cp = f64(D["C"][lo:s][:, :, codes])
        perio_A = prior_mean(f64(D["WA"][lo:s][:, codes]))
        perio_B = prior_mean(f64(D["WB"][lo:s][:, codes]))
        perio_C = prior_mean(cap(Cp[:, 1:] / Cp[:, :-1] - 1.0))               # (12, n): the return of bar k+1
        Cprev = f64(D["C"][s - 1, int(D["NS"][s - 1]) - 1][codes])
        dv0 = f64(D["DIV"][s][codes]); dv0 = np.where(dv0 > 0.5 * Cprev, 0.0, dv0)
        ovn = cap((O0 + dv0) / Cprev - 1.0)
        Xbar = np.full((N_SLOT, n), np.nan)
        Xbar[0] = cap(C[0] / O0 - 1.0)
        Xbar[1:nsl] = cap(C[1:nsl] / C[:nsl - 1] - 1.0)
        Xop = cap(C / O0 - 1.0)
        Ebar = np.full((N_SLOT, len(ETF_CODES)), np.nan)
        Ebar[0] = cap(Ce[0] / Oe0 - 1.0)
        Ebar[1:nsl] = cap(Ce[1:nsl] / Ce[:nsl - 1] - 1.0)
        Eop = cap(Ce / Oe0 - 1.0)
    Rbar = hedge(Xbar[:nsl], Ebar[:nsl, 1:], L)
    Rop = hedge(Xop[:nsl], Eop[:nsl, 1:], L)
    rev5, mom = L["rev5"], L["mom"]

    def alpha_rows(perio, k):
        raw = np.vstack([perio, Rbar[k], ovn, Rop[k], ivol[k], irng[k], rev5, mom])
        return robust_z_rows(raw) * SIGN_VEC

    out = {"A": [], "B": [], "C": []}
    # Book A: form at 10:30 (close of the bar labeled 10:00), hold to the close
    XA = C[nsl - 1] / C[1] - 1.0
    EA = Ce[nsl - 1] / Ce[1] - 1.0
    out["A"].append(dict(k=1, Z=alpha_rows(perio_A, 1), Y=hedge(XA[None], EA[None, 1:], L)[0], R=XA, E=EA,
                         ok=np.isfinite(C[1])))
    # Book B: form at 15:30 (12:30 on early closes), hold to the same instant of the next session
    if s + 1 < D["C"].shape[0]:
        kb, s1 = nsl - 2, s + 1
        kb1 = int(D["NS"][s1]) - 2
        dvB = f64(D["DIV"][s1][codes]); dvB = np.where(dvB > 0.5 * C[kb], 0.0, dvB)      # a dividend above half the price is a glitch
        XB = cap((f64(D["C"][s1, kb1][codes]) + dvB) / C[kb] - 1.0)
        EB = cap((f64(D["C"][s1, kb1][ETF_CODES]) + f64(D["DIV"][s1][ETF_CODES])) / Ce[kb] - 1.0)
        out["B"].append(dict(k=kb, Z=alpha_rows(perio_B, kb), Y=hedge(XB[None], EB[None, 1:], L)[0], R=XB, E=EB,
                             ok=np.isfinite(C[kb])))
    # Book C: form at every bar close from 10:00, hold one bar
    XC = cap(C[1:nsl] / C[:nsl - 1] - 1.0)
    EC = Ce[1:nsl] / Ce[:nsl - 1] - 1.0
    YC = hedge(XC, EC[:, 1:], L)
    for k in range(nsl - 1):
        out["C"].append(dict(k=k, Z=alpha_rows(perio_C[k], k), Y=YC[k], R=XC[k], E=EC[k], ok=np.isfinite(C[k])))
    return out

D_FULL = make_D(Cc, Oo, Vv, RNG, DIVc, NSLOT)
print(f"data sanity: {D_FULL['n_capped']:,} window returns beyond +/-{MAX_ABS_RET:.0%} treated as missing; "
      f"{N_DIV_DROPPED:,} day-lake dividend cells above half the prior close dropped")
DEVIATIONS.append(f"Data sanity, not in the pre-registration: window and bar returns beyond +/-100% are treated as "
                  f"missing (notebook 12's daily rule; {D_FULL['n_capped']:,} window cells), day-lake dividends above "
                  f"half the prior close are dropped ({N_DIV_DROPPED:,} cells), and the dense arrays and cost vectors "
                  f"are keyed by the (ticker, id) pair rather than the id; and a fifth universe exclusion removes names "
                  f"whose day-lake series is corrupted in the formation window (a close above $100,000 or a recovered "
                  f"dividend above half the prior close). The first full run had BGZ, a triple-inverse fund whose "
                  f"day-lake closes sit near 1e13, in the universe: rules (ii) and (iii) judged it on garbage and its "
                  f"junk dividend applied to a real minute-lake price produced window returns of the order of 1e9 and "
                  f"a fictitious equal-weight P&L.")
print(f"same-clock window arrays built: {np.isfinite(D_FULL['WB']).mean():.1%} of (session, id) cells carry a 24-hour window, "
      f"{np.isfinite(D_FULL['WA']).mean():.1%} an intraday one")
""")

md(r"""
### 3.1 Look-ahead audit

A signal at a bar close must be a function of data through that close. The audit takes a sample of sessions
and, for each book's formation instant, hands `session_rows` copies of the arrays in which **everything after
that instant is blanked** (the later bars of the session, every later session, later dividends) and compares the
eight z-scores with the ones computed from the full arrays. Anything that reads the future would move. The
targets are expected to change (they are built from the future by definition) and are not compared.
""")
code(r"""
def blanked(D, s, k):
    lo = s - LOOKBACK - 1
    hi = min(s + 3, D["C"].shape[0])
    C, O, V, RG, DIV, NS = (D[x][lo:hi].copy() for x in ("C", "O", "V", "RNG", "DIV", "NS"))
    sl = s - lo
    for a in (C, O, V, RG):
        a[sl, k + 1:] = np.nan
        a[sl + 1:] = np.nan
    DIV[sl + 1:] = 0.0
    return make_D(C, O, V, RG, DIV, NS), sl

rng_a = np.random.default_rng(1)
pool = [s for s in sorted(LOAD) if s + 3 < S and s - LOOKBACK - 1 >= 0]
picks = [pool[i] for i in rng_a.choice(len(pool), size=min(12, len(pool)), replace=False)]
worst, n_checked = 0.0, 0
for s in picks:
    full = session_rows(s, LOAD[s], D_FULL)
    for book in ("A", "B", "C"):
        for row in full[book][::3] if book == "C" else full[book]:
            Db, sl = blanked(D_FULL, s, row["k"])
            part = session_rows(sl, LOAD[s], Db)
            twin = [r for r in part[book] if r["k"] == row["k"]][0]
            a, b = row["Z"], twin["Z"]
            same_nan = np.array_equal(np.isnan(a), np.isnan(b))
            diff = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0.0
            worst = max(worst, diff if same_nan else np.inf)
            n_checked += 1
print(f"look-ahead audit: {n_checked} formations on {len(picks)} sessions; largest change in any z-score when all later data "
      f"is blanked: {worst:.2e} ({'no dependence on the future' if worst < 1e-6 else 'LOOK-AHEAD FOUND'})")
assert worst < 1e-6
""")

code(r"""
f_pan = CACHE / f"{PFX}panels.pkl"
NMAX = max(len(L["codes"]) for L in LOAD.values())
sess_list = sorted(LOAD)

if f_pan.exists():
    with open(f_pan, "rb") as fh:
        PAN = pickle.load(fh)
    print("loaded the cached panels")
else:
    t0 = time.time()
    counts = {"A": len(sess_list), "B": sum(1 for s in sess_list if s + 1 < S),
              "C": sum(int(NSLOT[s]) - 1 for s in sess_list)}
    PAN = {}
    for b in "ABC":
        F = counts[b]
        PAN[b] = {"sess": np.zeros(F, np.int32), "slot": np.zeros(F, np.int8), "n": np.zeros(F, np.int16),
                  "Z": np.full((F, NA, NMAX), np.nan, np.float32), "Y": np.full((F, NMAX), np.nan, np.float32),
                  "R": np.full((F, NMAX), np.nan, np.float32), "ok": np.zeros((F, NMAX), bool),
                  "E": np.zeros((F, len(ETF_CODES)))}
    at = {b: 0 for b in "ABC"}
    for s in sess_list:
        rows = session_rows(s, LOAD[s], D_FULL)
        n = len(LOAD[s]["codes"])
        for b in "ABC":
            P = PAN[b]
            for r in rows[b]:
                f = at[b]; at[b] += 1
                P["sess"][f], P["slot"][f], P["n"][f] = s, r["k"], n
                P["Z"][f, :, :n], P["Y"][f, :n], P["R"][f, :n] = r["Z"], r["Y"], r["R"]
                P["ok"][f, :n], P["E"][f] = r["ok"], r["E"]
    assert all(at[b] == counts[b] for b in "ABC")
    with open(f_pan, "wb") as fh:
        pickle.dump(PAN, fh, protocol=5)
    print(f"panels built in {time.time() - t0:.0f}s")
for b, name in (("A", "10:30 to the close"), ("B", "15:30 to the next 15:30"), ("C", "each 30-minute bar")):
    P = PAN[b]
    print(f"Book {b} ({name}): {len(P['sess']):,} formations, {np.isfinite(P['Y']).sum():,} name-target pairs, "
          f"mean {np.isfinite(P['Y']).sum(1).mean():.0f} names with a target per formation")
""")

md(r"""
### 3.2 Coverage and correlation

Coverage is the share of names at a formation with a defined signed z-score, pooled over the development
period; the daily controls are defined whenever the risk model is, the intraday alphas only for names that
traded, and the same-clock alphas need ten prior sessions. The correlation matrix pools every Book B
development formation.
""")
code(r"""
def period_of(sess_idx):
    d = S_W[sess_idx]
    return np.where(d <= DEV_END, "dev", np.where((d >= TEST_START) & (d <= TEST_END), "test",
                    np.where(d >= HOLD_START, "hold", "gap")))

for b in "ABC":
    PAN[b]["period"] = period_of(PAN[b]["sess"])
    PAN[b]["date"] = S_W[PAN[b]["sess"]]

rows = []
for b in "ABC":
    P = PAN[b]; dev = P["period"] == "dev"
    valid = np.arange(NMAX)[None, :] < P["n"][dev][:, None]
    for j, a in enumerate(ALPHAS):
        z = P["Z"][dev][:, j, :]
        rows.append({"book": b, "alpha": a, "coverage": np.isfinite(z)[valid].mean()})
cov_a = pd.DataFrame(rows).pivot(index="alpha", columns="book", values="coverage").loc[ALPHAS]
display(cov_a.round(3))

PB = PAN["B"]; dev = PB["period"] == "dev"
Zdev = PB["Z"][dev]                                                          # (F, 8, NMAX)
flat = np.moveaxis(Zdev, 1, 0).reshape(NA, -1)
keep = np.isfinite(flat).sum(0) >= 2
corr = pd.DataFrame(flat[:, keep]).T.corr().to_numpy()
corr = pd.DataFrame(corr, index=ALPHAS, columns=ALPHAS)
display(corr.round(2))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(NA)); ax.set_xticklabels(ALPHAS, rotation=45, ha="right")
ax.set_yticks(range(NA)); ax.set_yticklabels(ALPHAS)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlation of the eight signed z-scores, Book B, development")
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════ 4. Univariate IC ═══════════════════════════════
md(r"""
## 4. Univariate information coefficients

For each alpha, book window and period, the Pearson correlation between the signed z-score and the hedged
target is computed formation by formation (pairwise-complete, more than 30 names), and averaged over the
**probe** formations: every fifth session of the trading range, and for Book C every bar of every fifth
session. Probe sessions are five sessions apart, so the windows they score do not overlap (a 24-hour window
of Book B on one probe session cannot reach the next), and the standard error is the standard deviation of
those formation-level correlations over the square root of their count. The table also gives the mean number
of names in the pairwise-complete cross-section. The Benjamini–Hochberg verdict at q = 0.10 is computed once,
across the eight alphas' development p-values on Book B's window. Only the development and test periods are
shown here; the hold-out waits for §9.
""")
code(r"""
def row_corr(X, Y, min_n=30, chunk=4000):
    # Pearson correlation of X and Y along axis 1, pairwise-complete; NaN unless more than min_n valid pairs
    F = X.shape[0]
    out, cnt = np.full(F, np.nan), np.zeros(F, int)
    for a in range(0, F, chunk):
        x, y = X[a:a + chunk].astype(np.float64), Y[a:a + chunk].astype(np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        n = m.sum(1)
        with np.errstate(all="ignore"):
            xm = np.where(m, x, 0.0); ym = np.where(m, y, 0.0)
            xc = np.where(m, x - (xm.sum(1) / n)[:, None], 0.0)
            yc = np.where(m, y - (ym.sum(1) / n)[:, None], 0.0)
            c = (xc * yc).sum(1) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1))
        c[n <= min_n] = np.nan
        out[a:a + chunk], cnt[a:a + chunk] = c, n
    return out, cnt

trade_pos = np.array(sess_list)
PROBE_SESS = set(trade_pos[::5])
ICF, ICN = {}, {}
t0 = time.time()
for b in "ABC":
    P = PAN[b]
    P["probe"] = np.isin(P["sess"], list(PROBE_SESS))
    ICF[b] = np.full((len(P["sess"]), NA), np.nan); ICN[b] = np.zeros((len(P["sess"]), NA), int)
    for j in range(NA):
        ICF[b][:, j], ICN[b][:, j] = row_corr(P["Z"][:, j, :], P["Y"])
print(f"formation-level ICs for all three books and eight alphas in {time.time() - t0:.0f}s; "
      f"{len(PROBE_SESS):,} probe sessions")

def ic_stats(b, j, period, ic_arr=None, cnt_arr=None):
    P = PAN[b]
    arr = ICF[b][:, j] if ic_arr is None else ic_arr
    m = P["probe"] & (P["period"] == period) & np.isfinite(arr)
    a = arr[m]
    if len(a) < 2:
        return dict(ic=np.nan, se=np.nan, t=np.nan, n=len(a), names=np.nan)
    ic, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    names = (ICN[b][m, j].mean() if cnt_arr is None else cnt_arr[m].mean())
    return dict(ic=ic, se=se, t=ic / se if se > 0 else np.nan, n=len(a), names=names)

def ic_table(b, periods=("dev", "test")):
    rows = {}
    for j, a in enumerate(ALPHAS):
        r = {}
        for p in periods:
            g = ic_stats(b, j, p)
            r[f"{p}_ic"], r[f"{p}_se"], r[f"{p}_t"], r[f"{p}_n"], r[f"{p}_names"] = g["ic"], g["se"], g["t"], g["n"], g["names"]
        rows[a] = r
    return pd.DataFrame(rows).T

icB = ic_table("B")
icB["sign_agrees_dev"] = icB["dev_ic"] > 0
dev_p = 2 * stats.t.sf(np.abs(icB["dev_t"].astype(float)), df=(icB["dev_n"].astype(float) - 1))
reject, p_adj = benjamini_hochberg_fdr(dev_p, alpha=0.10)
icB["bh_pass_dev"] = reject
print("Book B (15:30 to the next 15:30), h = one session:")
display(icB.round(4))
print(f"Book B development ICs with the pre-registered sign: {int(icB['sign_agrees_dev'].sum())} of {NA}")
print(f"Benjamini-Hochberg (q=0.10) survivors among the eight Book B development ICs: {int(icB['bh_pass_dev'].sum())} of {NA}")
""")

code(r"""
icA, icC = ic_table("A"), ic_table("C")
print("Book A (10:30 to the close):")
display(icA.round(4))
print("Book C (each 30-minute bar):")
display(icC.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, per in zip(axes, ("dev", "test")):
    w = 0.26
    for q, (b, lab, col) in enumerate((("C", "next 30-minute bar", "tab:green"), ("A", "10:30 to the close", "tab:orange"),
                                       ("B", "15:30 to the next 15:30", "tab:blue"))):
        tab = {"A": icA, "B": icB, "C": icC}[b]
        ax.bar(np.arange(NA) + (q - 1) * w, tab[f"{per}_ic"].astype(float), width=w, label=lab, color=col,
               yerr=1.96 * tab[f"{per}_se"].astype(float), capsize=2, ecolor="0.4")
    ax.set_xticks(range(NA)); ax.set_xticklabels(ALPHAS, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{'Development' if per == 'dev' else 'Test'}: mean IC by window (bars: 95% interval)")
axes[0].set_ylabel("mean IC on the hedged target"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════ 5. Blends ═══════════════════════════════
md(r"""
**What the tables say.** On Book B's window six of the eight development ICs had the pre-registered sign (`perio` and
`ivol_abn` did not) and two survived Benjamini–Hochberg at q = 0.10: `irev30` (IC 0.0246, s.e. 0.0063, t = 3.9328) and
`irange` (0.0090, s.e. 0.0028, t = 3.1846). Neither carried into the test period (0.0036, t = 0.3719, and 0.0024,
t = 0.5470). The only alpha above two standard errors there was `iopen` (0.0248, t = 2.1967), which had not passed in
development (0.0121, t = 1.7861). Five of the eight test ICs were positive; `perio`, `ovn` and `mom12_1` were not. The
two daily controls were small in both periods (`rev5` 0.0021 and 0.0074, `mom12_1` 0.0025 and −0.0029).

On Book A's window (10:30 to the close) `iopen` (0.0141, t = 2.0521), `ivol_abn` (0.0084, t = 2.9300) and `rev5` (0.0098,
t = 2.6901) were the development ICs above two standard errors, and `irev30` (0.0245, t = 2.6167) and `iopen` (0.0266,
t = 2.5707) the ones in the test period; no rule was applied to Book A's table. On Book C's window, the next 30-minute
bar, `irev30` had an IC of 0.0401 (s.e. 0.0020, t = 20.4198) in development and 0.0264 (t = 8.8502) in the test period,
and `iopen` 0.0173 (t = 8.5190) and 0.0147 (t = 4.8225); the other alphas were near zero in the test period, `ivol_abn`
(0.0007) and `perio` (−0.0005) among them. Book C's standard errors treat every bar of a probe session as independent,
which the bars of one session are not, so its t-statistics are the least reliable in the tables. What the alphas carry
is a short-horizon reversal (`irev30`, `iopen`), largest on the next 30-minute bar. The hold-out counterparts of these
tables are in §9.0.
""")

md(r"""
## 5. Blends

**Equal-weight (the decision blend)**: the mean of the eight signed z-scores at each formation, computed over the
alphas a name has (a name missing an alpha is blended from the others rather than dropped). A second
equal-weight blend of alphas 7 and 8 alone (`rev5` and `mom12_1`, the daily controls) is kept for the
secondary question.

**Ridge**: the book's hedged target regressed on the eight signed z-scores pooled over (name, formation) rows of
the trailing 36 months, refit at the first session of every month, walk-forward. The penalty is chosen from
{0.1, 1, 10, 100} times the mean diagonal of X'X by leave-one-calendar-year-out cross-validation inside the
trailing window, and the final fit for the month uses the whole window at the chosen penalty. A formation
enters the history only once its target has realized: for Book B (24-hour target) a session's formation is
admitted at the refit only if the session is at least one session older than the refit date, and for Books A
and C (targets realized the same day) any earlier session is. Missing z-scores enter the design as zero. A
refit needs at least a year of admitted history. Because the ridge only needs the cross-products of the design
and the target, they are computed once per formation and added up, which makes every refit, every
cross-validation fold and every placebo draw a small linear-algebra problem.

**Where the blend disagrees with the univariate sign.** The coefficients act on already-signed z-scores, so a
negative coefficient means the blend wants that alpha to point the other way from its pre-registered sign that
month. For each book the table gives, per alpha, the share of monthly refits with a negative coefficient and
whether the **median** coefficient over the refits disagrees in sign with the pre-registration.
""")
code(r"""
def blend_ew(P, cols=None):
    Z = P["Z"] if cols is None else P["Z"][:, cols, :]
    out = np.empty((Z.shape[0], NMAX), np.float32)
    for a in range(0, Z.shape[0], 4000):
        with np.errstate(all="ignore"):
            out[a:a + 4000] = np.nanmean(Z[a:a + 4000], axis=1)
    return out

date_arr = S_W[trade_pos]
year_arr = np.asarray(date_arr.year)
pos_of_sess = {int(s): i for i, s in enumerate(trade_pos)}
MIN_RIDGE_SESSIONS = 100 if SMOKE else 252

def formation_stats(P, Zr, chunk=2000):
    # per-session sums of X'X, X'y, y'y and the row count, over the rows that have a target
    Sn = len(trade_pos)
    XTX, XTY = np.zeros((Sn, NA, NA)), np.zeros((Sn, NA))
    YTY, NN = np.zeros(Sn), np.zeros(Sn)
    pos = np.array([pos_of_sess[int(s)] for s in P["sess"]])
    for a in range(0, len(pos), chunk):
        z = np.nan_to_num(Zr[a:a + chunk], nan=0.0).astype(np.float64)
        y = P["Y"][a:a + chunk].astype(np.float64)
        m = np.isfinite(y)
        z = z * m[:, None, :]; y = np.where(m, y, 0.0)
        np.add.at(XTX, pos[a:a + chunk], np.einsum("fan,fbn->fab", z, z))
        np.add.at(XTY, pos[a:a + chunk], np.einsum("fan,fn->fa", z, y))
        np.add.at(YTY, pos[a:a + chunk], (y ** 2).sum(1))
        np.add.at(NN, pos[a:a + chunk], m.sum(1))
    return XTX, XTY, YTY, NN

first_of_month = pd.Series(np.arange(len(date_arr)), index=date_arr).groupby(date_arr.to_period("M")).min().to_numpy()

def ridge_walk(st, lag):
    XTX, XTY, YTY, NN = st
    cs = [np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)]) for x in (XTX, XTY, YTY, NN)]
    take = lambda a, b: tuple(c[b] - c[a] for c in cs)
    coef_at = np.zeros((len(date_arr), NA))
    refits, mult_log = {}, []
    for pos in first_of_month:
        d = date_arr[pos]
        lo = int(date_arr.searchsorted(d - pd.DateOffset(months=RIDGE_TRAIL_MONTHS), side="left"))
        hi = pos - lag                                                     # sessions p with p + lag < pos
        if hi - lo < MIN_RIDGE_SESSIONS:
            continue
        tot = take(lo, hi)
        pieces = []
        for y in np.unique(year_arr[lo:hi]):
            idx = np.flatnonzero(year_arr == y)
            pieces.append(take(max(lo, idx[0]), min(hi, idx[-1] + 1)))
        mult = RIDGE_GRID[1]
        if len(pieces) >= 2:
            scores = {}
            for mu in RIDGE_GRID:
                mses = []
                for h in pieces:
                    tr = tuple(t_ - h_ for t_, h_ in zip(tot, h))
                    if tr[3] < 50 or h[3] < 10:
                        continue
                    lam = mu * np.mean(np.diag(tr[0]))
                    c = np.linalg.solve(tr[0] + lam * np.eye(NA), tr[1])
                    mses.append((h[2] - 2 * c @ h[1] + c @ h[0] @ c) / h[3])
                if mses:
                    scores[mu] = float(np.mean(mses))
            mult = min(scores, key=scores.get) if scores else RIDGE_GRID[1]
        lam = mult * np.mean(np.diag(tot[0]))
        coef = np.linalg.solve(tot[0] + lam * np.eye(NA), tot[1])
        refits[int(pos)] = coef
        mult_log.append({"refit": d, "lambda_mult": mult, "rows": int(tot[3])})
        coef_at[pos:] = coef
    return coef_at, refits, pd.DataFrame(mult_log)

def ridge_scores(P, coef_at, Zr=None, chunk=2000):
    pos = np.array([pos_of_sess[int(s)] for s in P["sess"]])
    out = np.zeros((len(pos), NMAX), np.float32)
    for a in range(0, len(pos), chunk):
        z = np.nan_to_num((P["Z"] if Zr is None else Zr)[a:a + chunk], nan=0.0)
        out[a:a + chunk] = np.einsum("fan,fa->fn", z, coef_at[pos[a:a + chunk]]).astype(np.float32)
    return out

t0 = time.time()
SCORE, RIDGE_COEF, RIDGE_LOG, STATS = {}, {}, {}, {}
LAG = {"A": 0, "B": 1, "C": 0}
for b in "ABC":
    P = PAN[b]
    SCORE[(b, "ew")] = blend_ew(P)
    STATS[b] = formation_stats(P, P["Z"])
    coef_at, refits, log = ridge_walk(STATS[b], LAG[b])
    RIDGE_COEF[b], RIDGE_LOG[b] = refits, log
    SCORE[(b, "ridge")] = ridge_scores(P, coef_at)
    n_first = min(refits) if refits else None
    print(f"Book {b}: {len(refits)} monthly ridge refits, first {date_arr[n_first].date() if n_first is not None else 'none'}")
SCORE[("B", "ew78")] = blend_ew(PAN["B"], cols=[ALPHAS.index("rev5"), ALPHAS.index("mom12_1")])
print(f"blends built in {time.time() - t0:.0f}s")
display(RIDGE_LOG["B"]["lambda_mult"].value_counts().to_frame("Book B refits at this penalty multiple").T)
""")

code(r"""
fig, ax = plt.subplots(figsize=(11, 4))
cB = pd.DataFrame({date_arr[p]: c for p, c in RIDGE_COEF["B"].items()}, index=ALPHAS).T
(cB * 1e4).plot(ax=ax, lw=1.0)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Book B ridge weights over time (monthly refits), bps of next-session hedged return per unit z-score")
ax.legend(fontsize=7, ncol=4)
plt.tight_layout(); plt.show()

def sign_table(b):
    c = pd.DataFrame({date_arr[p]: v for p, v in RIDGE_COEF[b].items()}, index=ALPHAS).T
    c = c[c.index <= TEST_END]                                             # the hold-out refits wait for section 9
    t = pd.concat([(c < 0).mean().rename("share of refits with coef < 0"),
                   (c.median() * 1e4).rename("median coefficient (bps per z)")], axis=1)
    t["median coef disagrees with SIGN"] = t["median coefficient (bps per z)"] < 0
    return t.sort_values("share of refits with coef < 0", ascending=False), len(c)

for b in "BAC":
    t, n_ref = sign_table(b)
    print(f"Book {b}: ridge coefficients over {n_ref} monthly refits up to the end of the test period")
    display(t.round(3))
    flipped = t.index[t["share of refits with coef < 0"] > 0.25].tolist()
    print(f"  {len(flipped)} of {NA} alphas have a negative coefficient (disagree with the pre-registered sign) in more than a "
          f"quarter of refits: {flipped}")
    print(f"  alphas whose MEDIAN ridge coefficient disagrees in sign with the pre-registered sign: "
          f"{t.index[t['median coef disagrees with SIGN']].tolist()}")
""")

md(r"""
### 5.1 Blend information coefficients, and the intraday-versus-daily question

The blend IC is the correlation of the blend score with the book's hedged target, on the same probe
formations as §4. For the secondary question, the eight-alpha equal-weight IC is compared with that of the
two daily controls alone **formation by formation**, so the standard error of the difference uses the paired
differences (the two blends share the controls and are highly correlated, so this s.e. is far smaller than
either blend's own).
""")
code(r"""
BLEND_IC = {}
for key, sc in SCORE.items():
    BLEND_IC[key] = row_corr(sc, PAN[key[0]]["Y"])
rows = []
for key in [("B", "ew"), ("B", "ridge"), ("B", "ew78"), ("A", "ew"), ("A", "ridge"), ("C", "ew"), ("C", "ridge")]:
    row = {"book": key[0], "blend": key[1]}
    for p in ("dev", "test"):
        g = ic_stats(key[0], 0, p, ic_arr=BLEND_IC[key][0], cnt_arr=BLEND_IC[key][1])
        row[f"{p}_ic"], row[f"{p}_se"], row[f"{p}_t"] = g["ic"], g["se"], g["t"]
    rows.append(row)
blend_ic_tab = pd.DataFrame(rows).set_index(["book", "blend"])
display(blend_ic_tab.round(4))

P = PAN["B"]
diff = BLEND_IC[("B", "ew")][0] - BLEND_IC[("B", "ew78")][0]
ADD = {}
for p in ("dev", "test"):
    m = P["probe"] & (P["period"] == p) & np.isfinite(diff)
    a = diff[m]
    ADD[p] = dict(diff=a.mean(), se=a.std(ddof=1) / np.sqrt(len(a)), n=len(a))
    ADD[p]["t"] = ADD[p]["diff"] / ADD[p]["se"]
print(f"Book B, eight-alpha equal-weight IC minus the IC of alphas 7 and 8 alone (paired over probe formations):")
for p in ("dev", "test"):
    print(f"  {p}: difference {ADD[p]['diff']:+.4f}, s.e. {ADD[p]['se']:.4f}, ratio {ADD[p]['t']:+.2f}, {ADD[p]['n']} formations")
print(f"development difference exceeds twice its s.e.: {ADD['dev']['diff'] > 2 * ADD['dev']['se']}; "
      f"same sign in the test period: {np.sign(ADD['test']['diff']) == np.sign(ADD['dev']['diff'])}")
""")

# ═══════════════════════════════ 6. Books ═══════════════════════════════
md(r"""
**What the tables say.** On Book B the equal-weight blend's IC was 0.0184 (s.e. 0.0047) in development and 0.0147
(s.e. 0.0076) in the test period. The ridge blend's was higher in development, 0.0221 (s.e. 0.0065), and lower in the
test period, 0.0079 (s.e. 0.0083). The blend of the two daily controls alone had 0.0028 (s.e. 0.0034) and 0.0034
(s.e. 0.0057). The paired difference between the eight-alpha blend and the controls was +0.0156 (s.e. 0.0045, ratio
+3.46, 453 formations) in development and +0.0113 (s.e. 0.0073, ratio +1.56, 202 formations) in the test period:
significant in the first, and of the same sign but not significant on its own in the second, which is what rule two
asks for. §9.0 adds the hold-out, where the sign reverses. On Book A the ridge blend's development IC was lower than
the equal-weight one (0.0120 against 0.0174), on Book C higher (0.0381 against 0.0254), and on C the ridge leaned on
`irev30` (median coefficient 0.857 bps per z-score).

**Where the ridge disagreed with the pre-registered signs.** On Book B the median `perio` coefficient was negative
(−0.214 bps per z-score, negative in 0.923 of the refits); its development IC was slightly negative too (−0.0014,
t = −0.3465), not distinguishable from zero, so the disagreement is a weak one. `ovn` and `mom12_1` were negative in
0.441 and 0.413 of the refits with medians near zero (0.060 and 0.049), so the fitted blend gave them no stable weight,
while `irev30` and `iopen` were positive in nearly every refit (medians 1.151 and 0.833). On Books A and C no median
disagreed; `mom12_1` was negative in 0.424 and 0.451 of the refits with medians of 0.018 and 0.004.
""")

md(r"""
## 6. Books

Target weights are `neutralize(blend score, B) * CAP` with notebook 17's corrected, joint projection against the
loadings and a constant column (the loadings here are the session's 15 PCA betas, nine SPDR betas and the size
column), applied to the names that have a price at the formation instant, missing scores counted as zero. The
gross is notebook 12's one million dollars.

* **Book B, the decision instrument.** One formation per session at 15:30. Held weights move a fraction phi of
  the gap to target once per session (Garleanu–Pedersen partial adjustment) and the book is rescaled to the
  full gross; positions are held overnight and earn the window return to the next session's 15:30 with the
  ex-dividend credited, and pay 50 bp a year of borrow on the short notional. phi = 0.25 is primary; 1, 0.5 and
  0.1 are reported and never chosen from.
* **Book A.** One formation per session at 10:30, a full trade to target, flat at the close.
* **Book C.** A formation at every bar close from 10:00 to 15:30, a full trade to target each time, flat at
  the close. It exists to put the fundamental-law arithmetic and the cost cliff on the page at the desk's own
  frequency and is never selected on.

Rows are labeled by the **realization** date, the session the position's profit lands on (the next session for
Book B), while costs and turnover are charged on the session the trade is placed, as in notebook 17.
Books A and C hold nothing overnight and pay no borrow.
""")
code(r"""
cost_cells = pd.read_parquet(CACHE / "xs_costs.parquet")
cost_ok = cost_cells[cost_cells["err"].eq("")]
COST_DAY = {int(y): g.set_index("ticker")["cost_used"] for y, g in cost_ok.groupby("year")}
MED_DAY = {y: float(v.median()) for y, v in COST_DAY.items()}
tick_arr = np.array([TICK_OF_CODE.get(c, "") for c in range(N_ID)])
book_years = sorted({int(y) for y in S_W[trade_pos].year})

def cost_vectors(y):
    r = np.full(N_ID, np.nan)
    sub = roll_tab[(roll_tab["year"] == y) & roll_tab["roll_bps"].notna()]
    ci = all_id_index.get_indexer(sub["ticker"].astype(str) + "|" + sub["id"].astype(str))
    ok = ci >= 0
    r[ci[ok]] = sub["roll_bps"].to_numpy()[ok]
    med = ROLL_MED.get(y, float(roll_tab["roll_bps"].median()))
    prim = COMMISSION_BPS + 0.5 * np.where(np.isfinite(r), r, med)        # primary basis: commission + half the Roll spread
    ser = COST_DAY.get(y)
    dflt = MED_DAY.get(y, float(cost_ok["cost_used"].median()))
    sec = dflt if ser is None else np.where(np.isfinite(ser.reindex(tick_arr).to_numpy()), ser.reindex(tick_arr).to_numpy(), dflt)
    return prim, np.broadcast_to(sec, (N_ID,)).astype(float)

CV_P, CV_D = {}, {}
for y in book_years:
    CV_P[y], CV_D[y] = cost_vectors(y)
mid = book_years[len(book_years) // 2]
print(f"primary cost basis (commission plus half the Roll spread): median {np.median(CV_P[mid]):.2f} bps per name per side in {mid}; "
      f"secondary basis (notebook 14's day-lake measured costs): median {np.median(CV_D[mid]):.2f}")
print(f"stock costs reused from notebook 12/14: {len(cost_ok):,} of {len(cost_cells):,} (ticker, year) cells measured")

def neutral_targets(b, score):
    P = PAN[b]
    F = len(P["sess"])
    W = np.zeros((F, NMAX))
    for f in range(F):
        n = int(P["n"][f]); ok = P["ok"][f, :n]
        if ok.sum() < 30:
            continue
        sc = np.nan_to_num(score[f, :n].astype(np.float64))
        W[f, :n][ok] = neutralize(sc[ok], LOAD[int(P["sess"][f])]["B"][ok].astype(np.float64))
    return W

def _frame(rows, cols):
    return pd.DataFrame(rows, columns=cols).set_index("date")
BOOK_COLS = ["date", "s", "slot", "pnl_gross", "cost_p", "cost_d", "borrow", "traded", "gross", "n"]

def run_B(W, phi):
    P = PAN["B"]
    w_prev, rows = np.zeros(N_ID), []
    for f in range(len(P["sess"])):
        s, n = int(P["sess"][f]), int(P["n"][f]); codes = LOAD[s]["codes"]
        wt = np.zeros(N_ID); wt[codes] = W[f, :n] * CAP
        w = w_prev + phi * (wt - w_prev)
        g = np.abs(w).sum()
        if g > 0:
            w = w * (CAP / g)
        w[np.abs(w) < 1e-9] = 0.0
        dw = np.abs(w - w_prev)
        y = S_W[s].year
        r = np.zeros(N_ID); r[codes] = np.nan_to_num(P["R"][f, :n].astype(np.float64))
        rows.append((S_W[s + 1], s, int(P["slot"][f]), float((w * r).sum()), float((dw * CV_P[y]).sum() / 1e4),
                     float((dw * CV_D[y]).sum() / 1e4), float(-w[w < 0].sum() * BORROW_BPS / 1e4 / ANN),
                     float(dw.sum()), float(np.abs(w).sum()), int((w != 0).sum())))
        w_prev = w
    return _frame(rows, BOOK_COLS)

def run_A(W):
    P = PAN["A"]
    rows = []
    for f in range(len(P["sess"])):
        s, n = int(P["sess"][f]), int(P["n"][f]); codes = LOAD[s]["codes"]
        w = np.zeros(N_ID); w[codes] = W[f, :n] * CAP
        y = S_W[s].year
        r = np.zeros(N_ID); r[codes] = np.nan_to_num(P["R"][f, :n].astype(np.float64))
        aw = np.abs(w)
        rows.append((S_W[s], s, int(P["slot"][f]), float((w * r).sum()), float(2 * (aw * CV_P[y]).sum() / 1e4), np.nan, 0.0,
                     float(2 * aw.sum()), float(aw.sum()), int((w != 0).sum())))
    return _frame(rows, BOOK_COLS)

def run_C(W):
    # returns the formation-level frame; the session-level book is its daily sum
    P = PAN["C"]
    sess = P["sess"]; F = len(sess)
    rows, w_prev = [], np.zeros(N_ID)
    for f in range(F):
        s, n = int(sess[f]), int(P["n"][f]); codes = LOAD[s]["codes"]
        if f == 0 or sess[f - 1] != s:
            w_prev = np.zeros(N_ID)
        w = np.zeros(N_ID); w[codes] = W[f, :n] * CAP
        y = S_W[s].year
        dw = np.abs(w - w_prev)
        traded, cost = float(dw.sum()), float((dw * CV_P[y]).sum() / 1e4)
        if f == F - 1 or sess[f + 1] != s:                                    # flat at the close
            traded += float(np.abs(w).sum()); cost += float((np.abs(w) * CV_P[y]).sum() / 1e4)
        r = np.zeros(N_ID); r[codes] = np.nan_to_num(P["R"][f, :n].astype(np.float64))
        rows.append((S_W[s], s, int(P["slot"][f]), float((w * r).sum()), cost, np.nan, 0.0, traded,
                     float(np.abs(w).sum()), int((w != 0).sum())))
        w_prev = w
    return _frame(rows, BOOK_COLS)

def to_daily(bf):
    g = bf.groupby(level=0)
    out = g[["pnl_gross", "cost_p", "borrow", "traded"]].sum()
    out["cost_d"] = np.nan
    out["gross"] = g["gross"].mean()
    out["n"] = g["n"].mean()
    out["s"] = g["s"].first()
    return out

t0 = time.time()
TARGETS, BOOKS, BOOKS_F = {}, {}, {}
for blend in ("ew", "ridge", "ew78"):
    TARGETS[("B", blend)] = neutral_targets("B", SCORE[("B", blend)])
for blend in ("ew", "ridge"):
    for phi in PHIS:
        BOOKS[("B", blend, phi)] = run_B(TARGETS[("B", blend)], phi)
BOOKS[("B", "ew78", PHI_PRIMARY)] = run_B(TARGETS[("B", "ew78")], PHI_PRIMARY)
for blend in ("ew", "ridge"):
    TARGETS[("A", blend)] = neutral_targets("A", SCORE[("A", blend)])
    BOOKS[("A", blend, 1.0)] = run_A(TARGETS[("A", blend)])
    TARGETS[("C", blend)] = neutral_targets("C", SCORE[("C", blend)])
    BOOKS_F[("C", blend, 1.0)] = run_C(TARGETS[("C", blend)])
    BOOKS[("C", blend, 1.0)] = to_daily(BOOKS_F[("C", blend, 1.0)])
print(f"{len(BOOKS)} books built in {time.time() - t0:.0f}s: Book B two blends x four phi plus the daily-controls blend, "
      f"Books A and C two blends each at a full trade")
""")

# ═══════════════════════════════ 7. Costs and metrics ═══════════════════════════════
md(r"""
## 7. Costs and metrics

The **primary cost basis** is notebook 18's: one basis point of commission plus half the Roll spread per name per
side, the Roll spread estimated per name and calendar year from that year's one-minute closes (§1.3), with
unmeasured cells at that year's median, and 50 bp a year of borrow on overnight shorts. The **secondary basis**
is notebook 14's day-lake measured cost per (ticker, year), with the same borrow, so that Book B can be set
beside notebook 17's book on the same footing in §9. Sharpe ratios are annualized, with standard error
$\sqrt{252/\text{sessions}}$ on all sessions and again on the sessions where the book holds a position;
turnover per session is traded notional over the gross (both sides, notebook 17's convention); break-even cost
is gross P&L over traded notional in basis points, a per-side figure. Everything in this section stops at the
end of the test period.
""")
code(r"""
def se_sharpe(n):
    return float(np.sqrt(ANN / n)) if n > 0 else np.nan

def vis(df):
    return df[df.index <= TEST_END]

def summarize(key, df=None, with_hold=False):
    d = vis(BOOKS[key]) if df is None else df
    net = d["pnl_gross"] - d["cost_p"] - d["borrow"]
    net_d = d["pnl_gross"] - d["cost_d"] - d["borrow"]
    pos = d["gross"] > 0
    n = len(d)
    seg = lambda lo=None, hi=None: net[((d.index >= lo) if lo is not None else True) & ((d.index <= hi) if hi is not None else True)]
    s_dev, s_test = sharpe(seg(hi=DEV_END) / CAP), sharpe(seg(TEST_START, TEST_END) / CAP)
    out = {"book": key[0], "blend": key[1], "phi": key[2],
           "gross Sharpe": sharpe(d["pnl_gross"] / CAP), "net Sharpe": sharpe(net / CAP),
           "net Sharpe, day-lake costs": sharpe(net_d / CAP) if net_d.notna().all() else np.nan,
           "s.e.": se_sharpe(n), "sessions": n, "with a position": int(pos.sum()),
           "net Sharpe, positioned sessions": sharpe(net[pos] / CAP), "s.e., positioned": se_sharpe(int(pos.sum())),
           "turnover / session": float((d["traded"] / d["gross"].replace(0, np.nan)).mean()),
           "break-even bps": float(d["pnl_gross"].sum() / d["traded"].sum() * 1e4),
           "net Sharpe dev": s_dev, "net Sharpe test": s_test,
           "t, test": s_test * np.sqrt(len(seg(TEST_START, TEST_END)) / ANN) if np.isfinite(s_test) else np.nan}
    out["net P&L dev ($k)"] = seg(hi=DEV_END).sum() / 1e3
    out["net P&L test ($k)"] = seg(TEST_START, TEST_END).sum() / 1e3
    if with_hold:
        out["net Sharpe hold-out"] = sharpe(seg(HOLD_START) / CAP)
        out["net P&L hold-out ($k)"] = seg(HOLD_START).sum() / 1e3
    return out

sum_keys = [("B", b, p) for b in ("ew", "ridge") for p in PHIS] + [("B", "ew78", PHI_PRIMARY)] + \
           [(b, bl, 1.0) for b in "AC" for bl in ("ew", "ridge")]
SUMMARY = pd.DataFrame([summarize(k) for k in sum_keys]).set_index(["book", "blend", "phi"])
display(SUMMARY.round(3))
print(f"sessions shown: {int(SUMMARY['sessions'].iloc[0]):,} of the book's, up to {TEST_END.date()}")
""")

md(r"""
**What the table says.** Through the end of the test period Book B's equal-weight, phi = 0.25 book had a gross Sharpe of
1.095 and a net Sharpe of 0.433 on the primary basis and 0.515 on the day-lake basis (s.e. 0.278 on 3271 sessions),
with 0.263 of the gross traded each session and a break-even cost of 5.308 bps per side. On the primary basis the net
Sharpe rose monotonically as phi fell: −0.310 at 1.00, 0.429 at 0.50, 0.433 at 0.25 and 0.518 at 0.10, while the gross
fell from 2.506 to 0.812, so damping gave up less gross than it saved in cost (the four are reported, not chosen from).
On the day-lake basis phi = 0.50, 0.25 and 0.10 were indistinguishable (0.608, 0.515, 0.548) and only phi = 1.00
(0.077) was clearly worse. The ridge blend was lower than equal-weight at every phi on both bases through the test
period (0.333 against 0.433 at phi = 0.25 on the primary basis). The book of the two daily controls alone had a gross
Sharpe of 0.589 and a net of 0.018, so the intraday alphas lifted the gross to 1.095 and the net to 0.433: consistent
with rule two through the test period (§9.0 shows the hold-out). Books A and C turned a gross Sharpe of 3.684 and 9.259
into a net of −4.782 and −24.703, with break-even costs of 1.237 and 0.792 bps per side, below the median primary cost
of 2.64 bps (2018), at 2.000 and 11.197 of the gross traded per session.
""")

md(r"""
### 7.1 P&L by year, and five-year blocks

Book B at the primary phi, both blends, on the primary cost basis. Years and blocks stop at the end of the test
period; the partial last block is labeled by the years it holds.
""")
code(r"""
def by_year_table(key, upto=None):
    d = vis(BOOKS[key]) if upto is None else BOOKS[key][BOOKS[key].index <= upto]
    net = d["pnl_gross"] - d["cost_p"] - d["borrow"]
    g = pd.DataFrame({"gross": d["pnl_gross"], "net": net}).groupby(d.index.year)
    return pd.DataFrame({"gross Sharpe": g["gross"].apply(lambda x: sharpe(x / CAP)),
                         "net Sharpe": g["net"].apply(lambda x: sharpe(x / CAP)),
                         "net P&L ($k)": g["net"].sum() / 1e3})

def block_table(key, upto=None):
    d = vis(BOOKS[key]) if upto is None else BOOKS[key][BOOKS[key].index <= upto]
    net = d["pnl_gross"] - d["cost_p"] - d["borrow"]
    yrs = d.index.year
    blk = (yrs - FIRST_SESSION.year) // 5
    lab = pd.Series([f"{FIRST_SESSION.year + 5 * b}-{min(FIRST_SESSION.year + 5 * b + 4, yrs.max())}" for b in blk], index=d.index)
    g = pd.DataFrame({"gross": d["pnl_gross"], "net": net}).groupby(lab.to_numpy())
    return pd.DataFrame({"sessions": g.size(), "gross Sharpe": g["gross"].apply(lambda x: sharpe(x / CAP)),
                         "net Sharpe": g["net"].apply(lambda x: sharpe(x / CAP)),
                         "s.e.": [se_sharpe(n) for n in g.size()]})

prim_keys = [("B", "ew", PHI_PRIMARY), ("B", "ridge", PHI_PRIMARY)]
for k in prim_keys:
    print(f"Book B, {k[1]}, phi={k[2]}, by year:")
    display(by_year_table(k).round(3))
    print(f"Book B, {k[1]}, phi={k[2]}, by five-year block:")
    display(block_table(k).round(3))

fig, ax = plt.subplots(figsize=(11, 4))
for k, col in zip(prim_keys, ("tab:blue", "tab:orange")):
    d = vis(BOOKS[k])
    ax.plot(d.index, (d["pnl_gross"] - d["cost_p"] - d["borrow"]).cumsum() / 1e3, color=col, lw=1.2, label=f"{k[1]}, net")
    ax.plot(d.index, d["pnl_gross"].cumsum() / 1e3, color=col, lw=0.8, ls="--", label=f"{k[1]}, gross")
ax.axvline(DEV_END, color="0.4", ls=":", lw=1)
ax.axhline(0, color="k", lw=0.8); ax.legend()
ax.set_title(f"Book B, phi = {PHI_PRIMARY}: cumulative P&L ($k), development and test")
plt.tight_layout(); plt.show()
""")

md(r"""
**What the tables say.** Book B's equal-weight net Sharpe by year ran from 2.416 in 2010 to −1.337 in 2016, and its
gross from 3.324 to −0.702 in 2016. The year 2012, which the first run's interleaved Roll series had turned into the
one year that decided the sign of the period, lost 12.625 thousand dollars net on a gross Sharpe of 0.297, and no single year
decides the sign of the period. By five-year block the equal-weight net Sharpe was 1.103, −0.004 and 0.282 (s.e. 0.448,
0.448 and 0.577); only the first was more than two standard errors from zero, and the gross fell from 1.884 to 0.720
and then 0.806, the decay of an edge that was strong early in the sample. The ridge blend's blocks were 0.922, 0.019
and 0.220, the first also just over two standard errors.
""")

md(r"""
### 7.2 Exposures

Each book's net return regressed on SPY, the nine SPDRs, a size factor-mimicking return and a 12-1 momentum
factor-mimicking return, **all measured over the book's own realization window**: the ETFs' returns from the
formation instant to the end of the window (Book B: 15:30 to the next 15:30, dividends credited), and the two
mimicking returns are the characteristic's demeaned, unit-gross z-score at formation applied to the window
returns of the names. Aligning the regressors to the window matters for Book B, whose profit lands overnight:
regressing it on close-to-close returns would compare it with the wrong hours. Momentum is not neutralized (it
is an alpha under test), so its exposure is a measurement, not a zero by construction; SPY is not a factor of
the risk model, so its exposure tests whether the sector hedge left a market beta. Book C's regression is at
the formation level, one row per bar.
""")
code(r"""
def formation_regressors(b):
    P = PAN[b]
    F = len(P["sess"])
    size_f, mom_f = np.zeros(F), np.zeros(F)
    for f in range(F):
        L, n = LOAD[int(P["sess"][f])], int(P["n"][f])
        okm = P["ok"][f, :n]
        r = np.nan_to_num(P["R"][f, :n].astype(np.float64))
        for zv, arr in ((L["sizez"], size_f), (L["momz"], mom_f)):
            zz = np.where(okm & np.isfinite(zv), zv, np.nan)
            m = np.isfinite(zz)
            if m.sum() < 10:
                continue
            w = zz[m] - zz[m].mean()
            g = np.abs(w).sum()
            if g > 1e-12:
                arr[f] = float((w / g * r[m]).sum())
    X = pd.DataFrame(P["E"], columns=ETF10)
    X["size"], X["mom12_1"] = size_f, mom_f
    return X

REGRESSORS = {b: formation_regressors(b) for b in "ABC"}

def exposure_table(key, upto=None):
    upto = TEST_END if upto is None else upto
    bf = BOOKS_F[key] if key[0] == "C" else BOOKS[key]
    X = REGRESSORS[key[0]]
    assert len(bf) == len(X)
    y = ((bf["pnl_gross"] - bf["cost_p"] - bf["borrow"]) / CAP).to_numpy()
    m = np.asarray(bf.index <= upto)
    fit = sm.OLS(y[m], sm.add_constant(X[m].to_numpy())).fit()
    return pd.DataFrame({"beta": fit.params, "t": fit.tvalues}, index=["const"] + list(X.columns))

EXPOSURE = {k: exposure_table(k) for k in [("B", "ew", PHI_PRIMARY), ("B", "ridge", PHI_PRIMARY), ("A", "ew", 1.0), ("C", "ew", 1.0)]}
for k, tab in EXPOSURE.items():
    print(f"exposures, Book {k[0]}, {k[1]}, phi={k[2]} (regressors over the realization window):")
    display(tab.round(3))
# the ten ETF regressors are almost collinear (SPY against the sum of the nine SPDRs), so their joint coefficients are not
# separately interpretable; the market exposure is read from SPY alone and from the sum of the ten betas
for k in EXPOSURE:
    bf = BOOKS_F[k] if k[0] == "C" else BOOKS[k]
    Xk = REGRESSORS[k[0]]
    yk = ((bf["pnl_gross"] - bf["cost_p"] - bf["borrow"]) / CAP).to_numpy()
    mk = np.asarray(bf.index <= TEST_END)
    f_spy = sm.OLS(yk[mk], sm.add_constant(Xk.loc[mk, ["SPY"]].to_numpy())).fit()
    tab = EXPOSURE[k]
    print(f"Book {k[0]} {k[1]}: SPY-only beta {f_spy.params[1]:+.3f} (t {f_spy.tvalues[1]:+.2f}); sum of the ten ETF betas in the "
          f"joint regression {tab.loc[ETF10, 'beta'].sum():+.3f}; corr(SPY window return, sum of nine SPDR returns) "
          f"{np.corrcoef(Xk['SPY'], Xk[ETF9].sum(axis=1))[0, 1]:.3f}")
big = pd.DataFrame({f"{k[0]}-{k[1]}": v["t"].drop("const").abs() for k, v in EXPOSURE.items()}).max()
print("largest absolute t-statistic on any factor exposure, by book: " + ", ".join(f"{i} {v:.2f}" for i, v in big.items()))
""")

md(r"""
**What the tables say.** The joint regression's coefficients on SPY and the nine SPDRs are not separately interpretable:
the window returns of SPY and of the nine SPDRs together correlate at 0.971, so the equal-weight Book B's −0.202 on SPY
(t = −5.327) is offset by positive coefficients on the sector ETFs (largest 0.068 on XLK and 0.066 on XLY), and reading
it as a net short of a fifth of the gross would be wrong. The market exposure is what the two summary lines under the
tables give: a SPY-only beta of +0.030 (t = +9.07) and a sum of the ten joint betas of +0.032, a small net long, and the
same reading holds for the other books (SPY-only +0.020, +0.013 and +0.007 for the ridge Book B, A and C). The
equal-weight Book B's exposure to the 12-1 momentum mimicking return was large and positive (0.150, t = 24.654). That
is the unneutralized `mom12_1` alpha, one eighth of the blend score, showing up as an exposure by construction (its IC
was 0.0025 in development and −0.0029 in the test period), not as an edge. The intercept was −0.000 (t = −0.008). The
ridge Book B had a small SPY coefficient (0.035, t = 0.959), a size exposure of −0.061 (t = −3.066) and a momentum
exposure of −0.040 (t = −6.802), the opposite sign to the equal-weight book's. Books A and C had joint SPY coefficients
of −0.048 (t = −1.230) and −0.031 (t = −3.159) and a momentum exposure of 0.061 (t = 9.759) and 0.055 (t = 27.435);
their intercepts, −0.000 with t-statistics of −17.410 and −103.408, are their net loss, which is the cost, since no
factor exposure explains it.
""")

md(r"""
### 7.3 The five-clause scorecard for Book B

Notebook 14 section 9's five clauses, adapted as notebook 17 adapted them: the independent-bet unit is one
non-overlapping block of five sessions (positions persist for roughly four sessions at phi = 0.25, so trades
inside a block share most of their information), clause (b) is scored against the eight pre-registered alphas
(no other search was run), and clause (c) uses the test period here; the hold-out figure is added in §9.
""")
code(r"""
def scorecard(key, upto=None, hold=False):
    d = (vis(BOOKS[key]) if upto is None else BOOKS[key][BOOKS[key].index <= upto]).copy()
    d["block"] = np.arange(len(d)) // 5
    net = d["pnl_gross"] - d["cost_p"] - d["borrow"]
    tr = d["traded"].replace(0, np.nan)
    gross_bps, cost_bps = d["pnl_gross"] / tr * 1e4, (d["cost_p"] + d["borrow"]) / tr * 1e4
    block_net = (net / tr * 1e4).groupby(d["block"]).mean().dropna()
    t_bets = float(block_net.mean() / (block_net.std(ddof=1) / np.sqrt(len(block_net))))
    t_needed = float(stats.norm.ppf(1 - (1 - 0.95 ** (1 / NA)) / 2))
    s_test = sharpe(net[(net.index >= TEST_START) & (net.index <= TEST_END)] / CAP)
    c_text = f"test-period net Sharpe {s_test:.3f}"
    if hold:
        c_text += f", hold-out net Sharpe {sharpe(net[net.index >= HOLD_START] / CAP):.3f}"
    return pd.DataFrame([
        {"clause": "(0) profit per bet beats cost per bet",
         "value": f"{gross_bps.mean():.2f} bps gross vs {cost_bps.mean():.2f} bps cost ({gross_bps.mean() / cost_bps.mean():.2f}x)"},
        {"clause": "(a) survives its own sampling error",
         "value": f"t = {t_bets:.2f} over {len(block_net):,} non-overlapping 5-session blocks"},
        {"clause": "(b) survives the search that found it",
         "value": f"t = {t_needed:.2f} needed for one of {NA} pre-registered alphas (Sidak); achieved t = {t_bets:.2f}"},
        {"clause": "(c) survives out of sample", "value": c_text},
        {"clause": "(d) large enough to be worth running",
         "value": f"${net.sum():,.0f} net P&L over {len(d) / ANN:.1f} years on a ${CAP:,.0f} book"},
    ]).set_index("clause")

for k in prim_keys:
    print(f"five-clause scorecard, Book B, {k[1]}, phi={k[2]} (development and test):")
    display(scorecard(k))
""")

md(r"""
**What the table says.** For the equal-weight book (through the test period) the profit per bet was 5.25 bps gross
against 3.21 bps of cost (1.63x), so clause (0) passed: the book earned more per traded dollar than it paid. Clause (a)
came out at t = 1.49 over 655 non-overlapping blocks, positive and short of two. Clause (b) asked t = 2.73 of one of
eight alphas and the book's was 1.49. Clause (c) was a test-period net Sharpe of 0.144 (t = 0.29). Clause (d) was a net
profit of 180,406 dollars over 13.0 years on a one-million-dollar book. The ridge book scored lower on every clause that
varies: 4.43 gross against 3.20 cost (1.39x), t = 1.11, a test-period net Sharpe of 0.076 and a profit of 122,120
dollars.
""")

md(r"""
### 7.4 The fundamental law, for every book

Information ratio is roughly the information coefficient times the square root of the number of independent
bets a year. The realized IC is the blend's mean IC on the book's own target over probe formations
(development and test), with its standard error. Bets per year is breadth times how much of it turns over: the
average number of names held, times 252, times the one-way turnover per session (half of the two-sided
turnover of §7, so a book that replaces its whole portfolio each session counts one bet per name per
session). This is a rough check, not a proof: the IC is per formation, and on Books A and C the formations
within a session are treated as adding to the session's turnover rather than as separate bets.
""")
code(r"""
rows = []
for key in [("B", "ew", PHI_PRIMARY), ("B", "ridge", PHI_PRIMARY), ("A", "ew", 1.0), ("A", "ridge", 1.0),
            ("C", "ew", 1.0), ("C", "ridge", 1.0)]:
    d = vis(BOOKS[key])
    P = PAN[key[0]]
    arr = BLEND_IC[(key[0], key[1])][0]
    m = P["probe"] & np.isin(P["period"], ["dev", "test"]) & np.isfinite(arr)
    a = arr[m]
    ic, ic_se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    names = d.loc[d["gross"] > 0, "n"].mean()
    one_way = float((d["traded"] / d["gross"].replace(0, np.nan)).mean() / 2)
    bets = float(names * ANN * one_way)
    rows.append({"book": key[0], "blend": key[1], "realized IC": ic, "IC s.e.": ic_se, "names held": names,
                 "one-way turnover / session": one_way, "bets / year": bets, "IC x sqrt(bets)": ic * np.sqrt(bets),
                 "realized gross Sharpe": sharpe(d["pnl_gross"] / CAP)})
FLAW = pd.DataFrame(rows).set_index(["book", "blend"])
display(FLAW.round(3))
for (b, bl), r in FLAW.iterrows():
    print(f"Book {b}, {bl}: IC x sqrt(bets) = {r['IC x sqrt(bets)']:.3f} against a realized gross Sharpe of "
          f"{r['realized gross Sharpe']:.3f} (IC {r['realized IC']:.4f}, s.e. {r['IC s.e.']:.4f}, {r['bets / year']:,.0f} bets a year)")
""")

md(r"""
### 7.5 The cost cliff

The same gross P&L and traded notional, charged a flat per-side cost from zero upward, for the equal-weight
books. This is an illustration of where each book's net Sharpe crosses zero, not a cost rule: the pre-registered
costs are the measured ones above. Book B also pays its borrow.
""")
code(r"""
flat = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
rows = {}
for key, lab in ((("A", "ew", 1.0), "A, 10:30 to the close"), (("B", "ew", PHI_PRIMARY), "B, overnight, phi=0.25"),
                 (("C", "ew", 1.0), "C, every bar")):
    d = vis(BOOKS[key])
    rows[lab] = {c: sharpe((d["pnl_gross"] - d["traded"] * c / 1e4 - d["borrow"]) / CAP) for c in flat}
cliff = pd.DataFrame(rows).T
cliff.columns = [f"{c:g} bps" for c in flat]
display(cliff.round(2))
be = {lab: float(vis(BOOKS[k])["pnl_gross"].sum() / vis(BOOKS[k])["traded"].sum() * 1e4)
      for k, lab in ((("A", "ew", 1.0), "A"), (("B", "ew", PHI_PRIMARY), "B"), (("C", "ew", 1.0), "C"))}
print("break-even cost per side (bps of traded notional), gross of the flat cost, by book: "
      + ", ".join(f"{k} {v:.2f}" for k, v in be.items()))
med_cost = {y: float(np.median(CV_P[y])) for y in (book_years[0], book_years[-1])}
print("median measured cost per name per side on the primary basis: "
      + ", ".join(f"{y} {v:.2f} bps" for y, v in med_cost.items()))

fig, ax = plt.subplots(figsize=(8, 4))
for lab, r in cliff.iterrows():
    ax.plot(flat, r.to_numpy(), marker="o", label=lab)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("flat cost per side, bps"); ax.set_ylabel("net Sharpe"); ax.legend()
ax.set_title("Net Sharpe against a flat per-side cost, development and test")
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════ 8. Placebo ═══════════════════════════════
md(r"""
**What the table says.** Charged a flat cost per side, Book B's net Sharpe fell by about a tenth of a unit for every half
basis point (1.02 at zero, 0.91 at 0.5, 0.81 at 1, 0.60 at 2, 0.40 at 3) and crossed zero at about 5 bps (−0.01),
against break-evens of 5.31 bps for B, 1.24 for A and 0.79 for C. Book A fell by about 1.5 Sharpe units per half basis
point at first (3.68, 2.19, 0.71 at 1 bp) and Book C by about 5.8 (9.26, 3.43, −2.45 at 1 bp): the slope of the cliff is
the turnover, 0.263 for B, 2.000 for A and 11.197 for C per session. The median measured cost per name per side was 2.48
bps in 2010 and 2.82 bps in 2025, so B's break-even sat well above the median cost and A's and C's below it. The charge
the primary basis actually levied on B was 3.21 bps per traded dollar in clause (0) of §7.3, borrow included (well under
a basis point of it at 50 bp a year and this turnover), a little above the median name's 2.48 to 2.82: the traded dollars
sit in somewhat costlier names than the median name, not in a heavy tail. The book's net Sharpe at that charge, 0.433,
is close to the flat-cost row at 3 bps (0.40).
""")

md(r"""
## 8. Placebo, turnover-matched

Each alpha's name-to-signal map is permuted **once per calendar month** and held fixed within it: within a
month, name i receives the signed z-score of a fixed other name, drawn independently for each alpha, at every
formation of the month. A placebo book therefore turns over about as the real one does (the signal a name
carries changes as slowly as the real signal does), which the shuffle-every-session placebo of notebook 17
did not achieve. Each draw runs the whole Book B pipeline: equal-weight blend, neutralization, partial
adjustment at phi = 0.25, the primary costs. The ridge placebo also refits the walk-forward ridge on the
permuted signals. Draws use `np.random.default_rng(0)`. Gross and net Sharpe percentiles are both reported;
the gross one is the comparison the null supports, since a placebo has no reason to pay the real book's
costs identically. Sessions here stop at the end of the test period.
""")
code(r"""
PBK = PAN["B"]
months_B = PBK["date"].to_period("M")
month_groups = {m: np.flatnonzero(months_B == m) for m in months_B.unique()}
month_codes = {m: np.array(sorted(set(np.concatenate([LOAD[int(s)]["codes"] for s in np.unique(PBK["sess"][fi])]))))
               for m, fi in month_groups.items()}

def permute_B(rg):
    Z = PBK["Z"]
    Zp = np.full_like(Z, np.nan)
    for m, fidx in month_groups.items():
        mc = month_codes[m]
        maps = []
        for a in range(NA):
            pm = np.arange(N_ID)
            pm[mc] = mc[rg.permutation(len(mc))]
            maps.append(pm)
        for f in fidx:
            codes = LOAD[int(PBK["sess"][f])]["codes"]
            n = len(codes)
            posmap = np.full(N_ID, -1)
            posmap[codes] = np.arange(n)
            for a in range(NA):
                sp = posmap[maps[a][codes]]
                v = Z[f, a, :n]
                Zp[f, a, :n] = np.where(sp >= 0, v[np.maximum(sp, 0)], np.nan)
    return Zp

def book_stats(df):
    d = vis(df)
    net = d["pnl_gross"] - d["cost_p"] - d["borrow"]
    return sharpe(d["pnl_gross"] / CAP), sharpe(net / CAP), float((d["traded"] / d["gross"].replace(0, np.nan)).mean())

real_ew = book_stats(BOOKS[("B", "ew", PHI_PRIMARY)])
real_ridge = book_stats(BOOKS[("B", "ridge", PHI_PRIMARY)])

t0 = time.time()
ew_null = []
for draw in range(N_EW_DRAWS):
    Zp = permute_B(rng)
    W = neutral_targets("B", blend_ew({"Z": Zp}))
    ew_null.append(book_stats(run_B(W, PHI_PRIMARY)))
ew_null = pd.DataFrame(ew_null, columns=["gross", "net", "turnover"])
print(f"equal-weight placebo: {len(ew_null)} draws in {time.time() - t0:.0f}s")
""")

code(r"""
t0 = time.time()
ridge_null, draw = [], 0
while draw < N_RIDGE_DRAWS and (time.time() - t0) < RIDGE_PLACEBO_BUDGET_S:
    Zp = permute_B(rng)
    st_p = formation_stats(PBK, Zp)
    coef_p, _, _ = ridge_walk(st_p, LAG["B"])
    W = neutral_targets("B", ridge_scores(PBK, coef_p, Zr=Zp))
    ridge_null.append(book_stats(run_B(W, PHI_PRIMARY)))
    draw += 1
ridge_null = pd.DataFrame(ridge_null, columns=["gross", "net", "turnover"])
elapsed = time.time() - t0
print(f"ridge placebo: {len(ridge_null)} of {N_RIDGE_DRAWS} draws in {elapsed:.0f}s "
      f"(budget {RIDGE_PLACEBO_BUDGET_S} s{'; the budget cut it short' if len(ridge_null) < N_RIDGE_DRAWS else ''})")
if len(ridge_null) < N_RIDGE_DRAWS:
    DEVIATIONS.append(
        f"Section 8: the ridge placebo completed {len(ridge_null)} of the pre-registered {N_RIDGE_DRAWS} draws before its "
        f"{RIDGE_PLACEBO_BUDGET_S}-second budget ({elapsed:.0f}s elapsed); the shortfall is logged, nothing else was changed. "
        f"With this few draws the ridge percentile has coarse resolution.")

def null_row(name, real, null):
    if len(null) < 2:
        return {"book": name, "draws": len(null)}
    return {"book": name, "draws": len(null),
            "real gross": real[0], "null gross mean": null["gross"].mean(), "null gross sd": null["gross"].std(ddof=1),
            "gross percentile": float((null["gross"] < real[0]).mean() * 100),
            "gross z": (real[0] - null["gross"].mean()) / null["gross"].std(ddof=1),
            "real net": real[1], "null net mean": null["net"].mean(), "net percentile": float((null["net"] < real[1]).mean() * 100),
            "real turnover": real[2], "null turnover": null["turnover"].mean()}
null_tab = pd.DataFrame([null_row("equal-weight", real_ew, ew_null), null_row("ridge", real_ridge, ridge_null)]).set_index("book")
display(null_tab.round(3).T)
print(f"equal-weight Book B (phi={PHI_PRIMARY}): gross Sharpe {real_ew[0]:.3f} sits at the "
      f"{null_tab.loc['equal-weight', 'gross percentile']:.1f}th percentile of its {len(ew_null)}-draw gross null "
      f"(mean {ew_null['gross'].mean():.3f}, sd {ew_null['gross'].std(ddof=1):.3f}); net Sharpe {real_ew[1]:.3f} at the "
      f"{null_tab.loc['equal-weight', 'net percentile']:.1f}th percentile of the net null")
print(f"turnover per session, real {real_ew[2]:.3f} against the placebo's mean {ew_null['turnover'].mean():.3f}: "
      f"{'matched' if abs(real_ew[2] - ew_null['turnover'].mean()) < 0.5 * real_ew[2] else 'not matched'}")
if len(ridge_null) > 1:
    print(f"ridge Book B: gross Sharpe {real_ridge[0]:.3f} against its {len(ridge_null)}-draw gross null "
          f"(mean {ridge_null['gross'].mean():.3f}, sd {ridge_null['gross'].std(ddof=1):.3f}), "
          f"{null_tab.loc['ridge', 'gross z']:.1f} null standard deviations; on {len(ridge_null)} draws that is a z, not a percentile")

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.hist(ew_null["gross"], bins=max(6, len(ew_null) // 3), color="0.7", label="placebo, gross")
ax.axvline(real_ew[0], color="tab:blue", lw=2, label="real, gross")
ax.axvline(real_ew[1], color="tab:blue", lw=1.2, ls="--", label="real, net")
ax.set_xlabel("Sharpe, development and test"); ax.legend()
ax.set_title("Equal-weight Book B against its turnover-matched placebo")
plt.tight_layout(); plt.show()
""")

# ═══════════════════════════════ 9. Decision ═══════════════════════════════
md(r"""
**What the tables say.** Both real books sat above their nulls before costs: the equal-weight gross Sharpe of 1.095 was
above every one of 40 draws (null mean 0.020, s.d. 0.298, z = 3.610) and the ridge book's 1.131 was 4.561 null standard
deviations above its own (mean −0.063, s.d. 0.262). With 40 draws a 100.0th percentile means only that the real book
beat all of them, so the z is the better summary (it assumes a roughly normal null, which 40 draws cannot test). The
null gross means sit near zero, so the placebo pipeline (neutralization, partial adjustment, the blend) did not
manufacture a Sharpe on its own. Turnover was matched for the equal-weight book (0.263 against 0.266) but not exactly
for the ridge (0.305 against 0.262). The net comparison is not one the null supports and is reported for completeness:
the placebo's net Sharpe averaged −1.335 on a gross of 0.020, the real book's 0.433 on 1.095, so at matched turnover the
placebo paid about twice as much in cost per unit of turnover as the real book did, and every placebo draw's net
Sharpe sat below the real book's. That is the 100.0th net percentile, and it says nothing about the alphas: a permuted
signal map trades a different set of names from the real one, and the cost of the names traded is what separates the
nets. The notebook did not decompose the difference, and the gross comparison is the one the null supports.
""")

md(r"""
## 9. Decision, and the hold-out

Both pre-registered rules, applied verbatim, and then the hold-out for the first time. Nothing above depended
on it: the ridge refits and every table stopped at the end of the test period.

**Rule one.** The intraday desk **works** if Book B's equal-weight, phi = 0.25, net-of-primary-cost Sharpe on
the test period is positive with t > 2 and the hold-out has the same sign. **Rule two.** Intraday alphas
**add to the daily ones** if the eight-alpha equal-weight blend on Book B has a higher development IC than the
blend of alphas 7 and 8 alone by more than twice the difference's standard error, and the same sign of
difference in the test period; reported, not decided, if only one holds.
""")
code(r"""
key = ("B", "ew", PHI_PRIMARY)
dfull = BOOKS[key]
net_full = dfull["pnl_gross"] - dfull["cost_p"] - dfull["borrow"]
seg = lambda lo, hi: net_full[(net_full.index >= lo) & (net_full.index <= hi)]
s_test = sharpe(seg(TEST_START, TEST_END) / CAP)
n_test = len(seg(TEST_START, TEST_END))
t_test = s_test * np.sqrt(n_test / ANN)
s_hold = sharpe(seg(HOLD_START, pd.Timestamp("2100-01-01")) / CAP)
n_hold = len(seg(HOLD_START, pd.Timestamp("2100-01-01")))
works = bool(np.isfinite(s_test) and s_test > 0 and t_test > 2 and np.sign(s_hold) == np.sign(s_test))
print(f"RULE ONE: the intraday desk {'works' if works else 'does not work'} by the pre-registered rule. Book B equal-weight, "
      f"phi={PHI_PRIMARY}, net of the primary costs: test-period Sharpe {s_test:.3f} (t = {t_test:.2f} on {n_test} sessions, "
      f"against the threshold of 2), hold-out Sharpe {s_hold:.3f} on {n_hold} sessions, "
      f"{'the same sign as' if np.sign(s_hold) == np.sign(s_test) else 'the opposite sign to'} the test period.")

dev_ok = ADD["dev"]["diff"] > 2 * ADD["dev"]["se"]
test_ok = bool(np.sign(ADD["test"]["diff"]) == np.sign(ADD["dev"]["diff"]))
adds = dev_ok and test_ok
verdict2 = "add to the daily ones" if adds else ("do not add to the daily ones" if not (dev_ok or test_ok)
                                                 else "are reported only (exactly one of the two conditions holds)")
print(f"RULE TWO: intraday alphas {verdict2}. Development IC difference {ADD['dev']['diff']:+.4f} against twice its s.e. "
      f"{2 * ADD['dev']['se']:.4f} ({'exceeds' if dev_ok else 'does not exceed'} it); test-period difference "
      f"{ADD['test']['diff']:+.4f} ({'same' if test_ok else 'opposite'} sign).")
""")

md(r"""
### 9.0 Hold-out information coefficients

Sections 4 and 5 stopped at the test period. The same formation-level ICs, on the same probe formations, for the
hold-out, computed here for the first time: every alpha on every book, the blends, and rule two's paired
difference, so that the decision rules can be read against the period they did not use.
""")
code(r"""
hold_rows = []
for b in "ABC":
    for j, a in enumerate(ALPHAS):
        g = ic_stats(b, j, "hold")
        hold_rows.append({"book": b, "series": a, "hold_ic": g["ic"], "hold_se": g["se"], "hold_t": g["t"], "n": g["n"]})
    for bl in ("ew", "ridge") + (("ew78",) if b == "B" else ()):
        g = ic_stats(b, 0, "hold", ic_arr=BLEND_IC[(b, bl)][0], cnt_arr=BLEND_IC[(b, bl)][1])
        hold_rows.append({"book": b, "series": f"blend {bl}", "hold_ic": g["ic"], "hold_se": g["se"], "hold_t": g["t"], "n": g["n"]})
hold_ic_tab = pd.DataFrame(hold_rows).set_index(["book", "series"])
display(hold_ic_tab.round(4))
m_h = PAN["B"]["probe"] & (PAN["B"]["period"] == "hold") & np.isfinite(diff)
a_h = diff[m_h]
ADD["hold"] = dict(diff=a_h.mean(), se=a_h.std(ddof=1) / np.sqrt(len(a_h)), n=len(a_h))
ADD["hold"]["t"] = ADD["hold"]["diff"] / ADD["hold"]["se"]
print(f"Book B, eight-alpha equal-weight IC minus the IC of alphas 7 and 8 alone, hold-out: difference {ADD['hold']['diff']:+.4f}, "
      f"s.e. {ADD['hold']['se']:.4f}, ratio {ADD['hold']['t']:+.2f}, {ADD['hold']['n']} formations")
# leave-one-out: which alpha carries the intraday addition in each period (eight-alpha equal-weight blend without alpha j)
loo = {}
for j, a in enumerate(ALPHAS[:6]):
    keep_j = [i for i in range(NA) if i != j]
    sc_j = blend_ew(PAN["B"], cols=keep_j)
    ic_j = row_corr(sc_j, PAN["B"]["Y"])[0]
    d_j = BLEND_IC[("B", "ew")][0] - ic_j
    row = {}
    for p in ("dev", "test", "hold"):
        m_ = PAN["B"]["probe"] & (PAN["B"]["period"] == p) & np.isfinite(d_j)
        x_ = d_j[m_]
        row[f"{p}: IC gain from alpha"] = x_.mean(); row[f"{p}_t"] = x_.mean() / (x_.std(ddof=1) / np.sqrt(len(x_)))
    loo[a] = row
print("what each intraday alpha adds to the eight-alpha equal-weight blend's IC (blend minus blend without it), Book B:")
display(pd.DataFrame(loo).T.round(4))
""")

md(r"""
**What the tables say.** The hold-out ICs do not support the intraday alphas. On Book B's window the eight-alpha
equal-weight blend's IC was 0.0019 (s.e. 0.0089, t = 0.2111) on 131 probe formations, against 0.0142 (s.e. 0.0077,
t = 1.8585) for the blend of the two daily controls alone, so rule two's paired difference was −0.0124 (s.e. 0.0087,
ratio −1.42): the sign that held in development (+0.0156) and in the test period (+0.0113) reversed, though the
hold-out difference is within about one and a half standard errors of zero. Among the single alphas on Book B the
hold-out ICs were `perio` −0.0159 (t = −2.1160), `mom12_1` +0.0150 (t = 2.1558), `irev30` 0.0068, `rev5` 0.0060 and
near zero for the rest (`ovn` −0.0013, `iopen` −0.0013, `ivol_abn` −0.0064, `irange` −0.0002), so the two alphas above
two standard errors are one intraday alpha pointing the wrong way and one daily control. The leave-one-out table shows
that the addition rule two credits was carried by a different alpha in each period: `irev30` in development (its
removal cost the blend 0.0084 of IC, t = 3.4567) and `iopen` in the test period (0.0096, t = 2.1907), each contributing
almost nothing in the other, and in the hold-out no intraday alpha added and removing `perio` helped by 0.0059 (t =
−2.0214). On Book A the eight-alpha blend's hold-out IC was 0.0091 (t = 0.9382) and on Book C 0.0069 (t = 2.5120, on
1,572 bar formations whose standard errors are the least reliable); Book C's `irev30` kept a positive hold-out IC
(0.0092, t = 2.4649).
""")

md(r"""
### 9.1 The hold-out, all books

The full-span versions of the §7 tables, with the hold-out column. The hold-out is 2023 to the end of the data,
the window of notebooks 11 and 17, and was computed once, here.
""")
code(r"""
full_summary = pd.DataFrame([summarize(k, df=BOOKS[k], with_hold=True) for k in sum_keys]).set_index(["book", "blend", "phi"])
display(full_summary[["gross Sharpe", "net Sharpe", "net Sharpe, day-lake costs", "s.e.", "sessions",
                      "net Sharpe dev", "net Sharpe test", "net Sharpe hold-out", "net P&L dev ($k)", "net P&L test ($k)",
                      "net P&L hold-out ($k)", "turnover / session", "break-even bps"]].round(3))
for k in prim_keys:
    print(f"Book B, {k[1]}, phi={k[2]}, by year including the hold-out:")
    display(by_year_table(k, upto=LAST_SESSION + pd.Timedelta(days=5)).round(3))
    print(f"Book B, {k[1]}, phi={k[2]}, by five-year block including the hold-out:")
    display(block_table(k, upto=LAST_SESSION + pd.Timedelta(days=5)).round(3))
    print(f"five-clause scorecard with the hold-out, Book B, {k[1]}, phi={k[2]}:")
    display(scorecard(k, upto=LAST_SESSION + pd.Timedelta(days=5), hold=True))
for k in [("A", "ew", 1.0), ("C", "ew", 1.0)]:
    print(f"Book {k[0]}, {k[1]}, by year including the hold-out:")
    display(by_year_table(k, upto=LAST_SESSION + pd.Timedelta(days=5)).round(3))
    print(f"Book {k[0]}, {k[1]}, by five-year block including the hold-out:")
    display(block_table(k, upto=LAST_SESSION + pd.Timedelta(days=5)).round(3))

fig, ax = plt.subplots(figsize=(11, 4))
for k, col in zip(prim_keys, ("tab:blue", "tab:orange")):
    d = BOOKS[k]
    ax.plot(d.index, (d["pnl_gross"] - d["cost_p"] - d["borrow"]).cumsum() / 1e3, color=col, lw=1.2, label=f"{k[1]}, net")
    ax.plot(d.index, d["pnl_gross"].cumsum() / 1e3, color=col, lw=0.8, ls="--", label=f"{k[1]}, gross")
ax.axvline(DEV_END, color="0.4", ls=":", lw=1); ax.axvline(TEST_END, color="0.4", ls=":", lw=1)
ax.axhline(0, color="k", lw=0.8); ax.legend()
ax.set_title(f"Book B, phi = {PHI_PRIMARY}: cumulative P&L ($k), full span (dotted lines end development and test)")
plt.tight_layout(); plt.show()
""")

md(r"""
**What the tables say.** Over the full span Book B's equal-weight, phi = 0.25 book had a gross Sharpe of 1.130 and a net
of 0.476 on the primary basis (0.548 on the day-lake basis, s.e. 0.253). Its hold-out net Sharpe was 0.675 (net P&L
+60.732 thousand dollars), higher than the test figure (0.144) and above the development one (0.634), and by year 0.762
in 2023, 0.829 in 2024 and 0.330 in 2025. That is the sign rule one asked for, and it is not evidence of an edge in the
intraday alphas: the hold-out is 655 sessions long, clause (a) with the hold-out included was t = 1.71 over 786 blocks,
the profit per bet was 5.48 bps gross against 3.21 bps of cost (1.71x, against 1.63x through the test period), and the
book of the two daily controls alone earned 0.691 in the hold-out, with the eight-alpha blend's own hold-out IC at
0.0019 (§9.0). The ridge book's hold-out (1.197) was higher still but rested on 2023 (1.678) and 2025 (2.266), with 2024
at −0.103. The lowest-turnover books did best in the hold-out (phi = 0.10: 1.189 equal-weight, 1.614 ridge) and the
fastest worst (phi = 1.00: −1.337 and −1.735), the order of the cost cliff. Books A and C lost in every period
(hold-out −5.746 and −29.852), in every calendar year (the best years were 2011 for A at −0.386 and 2020 for C at
−11.927) and in every five-year block (A −5.340, −5.689, −4.270; C −28.736, −32.336, −20.749).
""")

md(r"""
### 9.2 Like-for-like against notebooks 17 and 11

Reported, not decided. Notebook 17's equal-weight book scored a net Sharpe of 0.596 at notebook 14's measured
day-lake costs, with a standard error of 0.23 (its span is 2006 to 2025); notebook 11's pairs book scored 0.49 with
0.26 on the same footing. The comparison here is Book B's equal-weight, phi = 0.25 book on the **secondary cost
basis only**, over its own span (which starts in 2010, so the two spans differ and the difference is not a
clean like-for-like in time), with the difference divided by the combined standard error.
""")
code(r"""
net_day = dfull["pnl_gross"] - dfull["cost_d"] - dfull["borrow"]
s_day, se_day = sharpe(net_day / CAP), se_sharpe(len(net_day))
print(f"Book B equal-weight, phi={PHI_PRIMARY}, secondary basis (day-lake measured costs, {BORROW_BPS:g} bp borrow), "
      f"{len(net_day):,} sessions {net_day.index.min().date()} -> {net_day.index.max().date()}: Sharpe {s_day:.3f}, s.e. {se_day:.3f}")
for name, ref, ref_se in (("notebook 17's equal-weight book", NB17_SHARPE, NB17_SE), ("notebook 11", NB11_SHARPE, NB11_SE)):
    comb = float(np.sqrt(se_day ** 2 + ref_se ** 2))
    print(f"  against {name} (Sharpe {ref:.3f}, s.e. {ref_se:.2f}): difference {s_day - ref:+.3f}, combined s.e. {comb:.3f}, "
          f"{abs(s_day - ref) / comb:.2f} combined standard errors")
print(f"for reference, the same book on the primary (Roll) basis: Sharpe {sharpe(net_full / CAP):.3f}")
""")

md(r"""
**What the table says.** On the secondary basis Book B's full-span Sharpe of 0.548 (s.e. 0.253) sat 0.048 below notebook
17's 0.596 and 0.058 above notebook 11's 0.490, that is 0.14 and 0.16 combined standard errors (combined s.e. 0.342 and
0.363). The three are not distinguishable, and the comparison could not have distinguished a difference of several
tenths: two combined standard errors span about 0.7. It is also not a like-for-like in construction: notebook 17's book
blended eleven daily alphas, Book B blends eight (six intraday), and the spans differ. Inside this notebook, on the
same span and basis, the eight-alpha book's 0.548 compares with 0.199 for the two daily controls alone (§9.1); the two
share their controls, so that difference is tested only by rule two's paired IC, not here. The same book on the primary
basis had 0.476, within a third of a standard error of the secondary figure, so here the two bases tell the same story.
""")

md(r"""
### 9.3 What the pieces say

**The rules.** Rule one failed: Book B's test-period net Sharpe was 0.144 (t = 0.29 on 1008 sessions, threshold 2), with
a hold-out Sharpe of 0.675 of the same sign. The sign condition was met, but by the daily controls and not by the intraday
alphas: the book of `rev5` and `mom12_1` alone earned 0.691 in the hold-out, the eight-alpha blend's hold-out IC was
0.0019, and `mom12_1` alone had +0.0150 (t = 2.1558). Rule two held as pre-registered: the eight-alpha blend's development
IC exceeded that of the daily controls alone by +0.0156 against twice its standard error of 0.0090, and the test-period
difference, +0.0113, had the same sign (ratio +1.56, not significant by itself). The addition did not persist into the
hold-out (−0.0124, ratio −1.42) and was carried by `irev30` in development and by `iopen` in the test period (§9.0), so
"the intraday alphas add to the daily ones" is a statement about two periods, each resting on a different alpha, and
not about a stable source of information. The desk as pre-registered does not work at the measured costs, and the
evidence that the intraday alphas carry information the daily ones lack is weaker than rule two's verdict alone
suggests. The gross placebo (below) is the stronger evidence that the signals carry something, on the periods it covers.

**Which alphas kept their sign.** On Book B's window six of the eight development ICs had the pre-registered sign (`perio`
at −0.0014 and `ivol_abn` at −0.0026 did not, neither distinguishable from zero) and two survived the
Benjamini–Hochberg step, `irev30` (t = 3.9328) and `irange` (t = 3.1846). In the test period five of the eight were
positive, both survivors fell under one standard error (t = 0.3719 and 0.5470), and the one alpha above two, `iopen`
(t = 2.1967), had missed the development cut (t = 1.7861); that t is below the 2.73 clause (b) of §7.3 asks of one of
eight alphas, so it is not evidence either. Selecting on development would have kept the two alphas that faded, and
the equal-weight blend, which selects nothing, held an IC of 0.0147 in the test period. The ridge disagreed with the
pre-registered sign only on `perio` (Book B), and was lower than equal-weight at every phi on both cost bases through
the test period (§7). Equal-weight is the decision blend, and the fitted one added nothing to it in that period.

**The fundamental law, book by book.** IC x sqrt(bets) was 2.252 for Book B's equal-weight blend against a realized gross
Sharpe of 1.095 (IC 0.0173, s.e. 0.0040, 16,991 bets a year) and 2.523 against 1.131 for its ridge blend (IC 0.0173,
s.e. 0.0052, 21,258 bets); 6.344 against 3.684 and 5.776 against 3.164 for Book A (112,603 and 112,978 bets); 18.024
against 9.259 and 33.474 against 9.904 for Book C (630,434 and 937,443 bets). The arithmetic overstated the realized
gross Sharpe by a factor of between about 1.7 and 2.2 in five of the six cases and by about 3.4 for Book C's ridge, the
book with the highest IC (0.0346) and the highest one-way turnover (8.298 per session). The sign of the overstatement is
the expected one: the law counts every unit of one-way turnover as an independent bet, while positions that persist for
roughly four sessions at phi = 0.25, alphas that share a reversal horizon, and the formations inside one session (Books
A and C) share information, and the IC is measured on the blend score before neutralization and scaling. The law still
explains the cliff. The Sharpe ratio grows with the square root of the bets and the cost bill grows with their number:
Book C's gross Sharpe of 9.259 was about eight and a half times Book B's 1.095 on more than forty times the traded
notional (11.197 against 0.263 of the gross per session), and only Book B's profit per traded dollar beat its cost.

**The placebo.** Read gross, the alphas carry information: 1.095 against a null of mean 0.020 and s.d. 0.298 (z = 3.610)
for the equal-weight book, and 1.131 against −0.063 and 0.262 (z = 4.561) for the ridge book, from 40 draws each at
matched turnover for equal-weight (0.263 against 0.266). Read net, the placebo says nothing about the alphas: its net
Sharpe averaged −1.335 against the real book's 0.433, and permuting the signals does not preserve which names are
traded, which is what the cost per traded dollar depends on; the placebo paid about twice as much in cost per unit of
turnover as the real book, and the notebook did not decompose that. That is why the pre-registration named the gross
comparison as the one the null supports. The placebo separates "the signal has information" (yes, at between three and
five null standard deviations, on the development and test periods it covers) from "the information pays for its own
trading" (yes on the primary basis over the full span, not established by any test the notebook ran), and only the
second was the question rule one asked.

**Which cost basis decides.** The primary basis decides because the pre-registration made it the decision basis before
any number existed: rule one is written on the net-of-primary-cost Sharpe, and the secondary basis was reserved for the
comparison with notebooks 17 and 11, so that the choice between them could not be made after seeing which one flattered
the book. It is also the basis measured on this notebook's own one-minute closes for the names and years the book
traded, and the only one computed for all three books (Deviation 12). It happens not to matter for Book B here. The
secondary basis, notebook 14's day-lake measured cost per name and year, gave Book B a net Sharpe of 0.548 (s.e. 0.253)
over the full span against 0.476 on the primary, a difference of less than a third of a standard error, and the
first run's apparent gap (a primary net Sharpe below zero) came from six names whose interleaved minutes had given them
Roll spreads in the thousands of basis points, not from the cost model (Deviation 13). On the secondary basis Book B was
positive at phi = 0.50, 0.25 and 0.10 (0.502, 0.548, 0.667) and negative at phi = 1.00 (−0.142); the best of the eight
Book B cells there, 0.667, sat more than two and a half standard errors from zero and is one of eight reported, not
chosen from. On the primary basis the same cells were 0.343, 0.476 and 0.640 and −0.488. The two bases have similar
medians (2.64 against 2.08 bps per name per side in 2018) and, on the corrected data, similar charges on what the book
traded. The remaining uncertainty is not which basis to use but whether either measures what a resting or crossing
order would really cost, which needs quotes or fills this notebook does not have: Book B's gross Sharpe of 1.130 is well
outside the placebo null and its net Sharpe is positive on both bases, with a full-span standard error of 0.253.

**Distance to the measured cost.** Break-even cost per side over the full span was 5.547 bps for Book B, 1.138 for Book A
and 0.736 for Book C, against a median measured cost of 2.48 to 2.82 bps per name per side (primary basis, 2010 and
2025) and a charge of 3.21 bps per bet on Book B (clause (0), §9.1, against 5.48 bps of gross profit per bet: 1.71x).
Book B's gross profit per bet was about one and seven-tenths times its cost. Books A and C were below half the median
cost, and Book C's break-even was below the one basis point of commission alone.

**What a desk with passive execution would face.** The primary basis charges every traded dollar the full half spread
plus a basis point of commission, the cost of crossing. A passive desk would pay less on the orders that filled, and
Book B's flat-cost row bounds the gain: 0.60 at 2 bps a side and 0.40 at 3 bps through the test period, against 0.433 at
the measured charge, if every order filled and none was adversely selected. Neither assumption is safe for these alphas,
which are mostly reversals: a resting bid fills when the price is still falling, before the reversal the signal
predicted, so the fill-conditional edge is smaller than the unconditional one this notebook measured, and the orders
that do not fill are the ones that would have profited. The minute lake carries prices and volume only, with no quotes or queue
information, so this notebook cannot measure either effect.
For Books A and C passive execution cannot close the gap on these numbers: break-evens of 1.138 bps (A) and 0.736 bps (C)
leave little or nothing after one basis point of commission, so they would need commission and spread near zero or a
rebate, and the fill-selection problem is strongest at C's 30-minute reversal (`irev30`, the largest IC in the tables).
Book B is the only book on which passive execution is a live question, and it is a question about fills, not about the
alphas.
""")

# ═══════════════════════════════ 10. Deviations ═══════════════════════════════
md("## 10. Deviations")
code(r"""
if SMOKE:
    DEVIATIONS.append("SMOKE MODE: this execution used the reduced span 2015-01-02 to 2016-12-30, fewer placebo draws and "
                      "periods re-cut inside that span (development 2015, test to mid-2016, hold-out after). Its numbers are "
                      "a debugging run and are not the recorded result.")
DEVIATIONS += [
    "Clock times. The pre-registration names instants by clock time ('15:30', '10:30'). This build reads the price at an "
    "instant as the close of the 30-minute bar that ends there (the bar labeled thirty minutes earlier), so a signal at a "
    "bar close uses that bar and the target starts at that close, as the timing rule requires. On the roughly three "
    "13:00 early-close sessions a year the instant thirty minutes before the close (12:30) plays the role of 15:30 for "
    "Book B, and the same relative slot is compared with the prior sessions' same clock slot.",
    "Universe. The monthly list is built on the last session before the month starts (nothing from the month itself), "
    "the exclusions are applied after the 500-name cap (each rule is judged on the full capped list, so their counts overlap and the table prints both the flags and the names only that rule removes), and a name must also have "
    "a complete 252-session daily return history to enter the PCA loadings and the risk model, as in notebooks 12 and 17. "
    "The nine sector SPDRs of the risk model are read as the nine classic Select Sector SPDRs, as in notebook 17.",
    "Minute-lake keys. The build notes say to key the market layout by `id`; the lake's `id` is not unique to a ticker inside a "
    "session file (see section 1.3), so rows are kept by the ticker-and-id pair the day lake resolved, which is what "
    "the ticker-reuse resolution was meant to achieve. The pair key fixed the day-lake merge and the row filter, but the first "
    "full run still grouped the filtered rows by `id` alone (see Deviation 13), which the pair key did not cover.",
    "Minute-lake history. The 25 sessions before the first formation session are read as history only (the 20-session "
    "same-clock alphas need them); no book forms on them. The ids kept per session are those of the month's universe and "
    "the two following months, so that a name entering the universe already has its history on hand.",
    "Alphas where the pre-registration was silent. A same-clock alpha (perio, ivol_abn, irange) needs at least "
    f"{MIN_PRIOR} valid sessions among the previous {LOOKBACK}; a log of zero volume or zero range is left missing; the "
    "range of irange is in price units (high minus low), as written, so its log ratio carries the ratio of prices over "
    "the twenty sessions, which is small next to the ratio of ranges. rev5 and mom12_1 are notebook 17's definitions "
    "taken at the prior close (rev5 sums the last five residuals of the 60-session regression with its intercept, as "
    "in notebook 17, where it is a signal and not a target).",
    "Risk-model betas are estimated on the 60 sessions ending at the prior close, not at the formation instant, so a "
    "15:30 formation uses no same-day data in its hedge. Neutralization is applied to the names that have a price at the "
    "formation instant.",
    "Roll spread. The per-name, per-year Roll spread is assembled from per-session pooled sufficient statistics of the "
    "traded one-minute closes rather than by handing a year of minute closes for every name to `roll_spread`; the "
    "assembly is checked against `roll_spread` itself in section 1.3 and agrees.",
    "Periods. Information coefficients, blends and ridge history are assigned to a period by the formation session; profit "
    "and loss rows by the realization session, so the single Book B position formed on the last session of a period "
    "realizes in the next period. The hold-out is computed together with everything else but displayed only in section 9.",
    "Ridge. It is fit for Books A and C as well as B (the build notes allowed it if time permitted; the cross-product "
    "formulation made it cheap). The admission rule is that a formation is history only once its target has realized: "
    "Book B admits sessions at least one session before the refit, Books A and C any earlier session.",
    "Placebo scope. The turnover-matched placebo runs for Book B only (the pre-registration asks for it there); Books A "
    "and C have none. The placebo sessions stop at the end of the test period.",
    "Bets per year in the fundamental law use the one-way turnover per session (half the two-sided figure of section 7); "
    "the pre-registration says only 'from breadth and turnover'. The secondary cost basis is computed for Book B only.",
    "Session grouping by ticker-and-id pair (a defect found after the first full run and corrected before the results were "
    "written up). Deviation 3 kept rows by the ticker-and-id pair, but the first full run then grouped the kept rows by `id` "
    "alone, so two kept tickers sharing one day-lake id inside a session were interleaved into one minute series. Fourteen day-lake ids "
    "carry two union tickers; eleven are renames that never trade in the same session and were harmless, and three pairs "
    "(JCI/TYC, ACE/CB, LBTYA/LBTYK) trade together. The interleaved series corrupted the Roll spread of those names "
    "(18 (name, year) cells above 50 bps, the largest 9,968 bps, against 30 bps for the largest other cell) and spliced their "
    "bars (a JCI first bar with a close-to-open move above 20% on 17% of sessions). The bars, Roll statistics, Roll table and panels "
    "were rebuilt with the pair as the grouping key, with an assertion that no (session, ticker, id) row repeats and that no Roll "
    "cell holds more pairs than its sessions times 391; every primary-basis number in this notebook is from the rebuild. The "
    "defect was not a pre-registered choice, but it moved the decision-basis numbers, so it is logged here rather than "
    "corrected silently: the primary-basis net Sharpe of Book B, the cost per traded dollar, the cost-basis comparison and "
    "Books A and C's net figures all changed, while the test-period and hold-out verdicts did not.",
    "Hold-out figures added beyond the pre-registered display. The pre-registration asks for the hold-out on rule one only. "
    "Section 9 also prints the hold-out information coefficient of every alpha and blend on all three books, rule two's paired "
    "difference in the hold-out, a leave-one-out of which intraday alpha carries the blend's addition in each period, and "
    "the by-year and by-block P&L of Books A and C. They were computed with the rest and added after the audit found their absence "
    "material; none of them changes either pre-registered verdict, and the rule-two verdict is read on development and test "
    "as pre-registered.",
    "Market exposure reading. The joint regression of section 7.2 has SPY and the nine SPDRs together, whose window "
    "returns are almost collinear, so the notebook now also prints the SPY-only beta and the sum of the ten betas; the "
    "joint coefficients on individual ETFs are not separately interpretable.",
]
print(f"{len(DEVIATIONS)} deviation(s) and clarification(s) logged:")
for k, dtext in enumerate(DEVIATIONS, 1):
    print(f"{k}. {dtext}")
""")

nb.cells = cells

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "pairs_trading_21_intraday_desk_alphas.ipynb")
    args = ap.parse_args()
    nbf.write(nb, args.out)
    print(f"wrote {args.out} ({len(cells)} cells)")
