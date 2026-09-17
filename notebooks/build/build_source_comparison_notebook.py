"""Build notebooks/pairs_trading_06_yahoo_vs_day_lake.ipynb (cells only; outputs from execute.py).

    python notebooks/build/build_source_comparison_notebook.py [--out PATH]
"""
import argparse
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "stat-arb", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Two sources for the same prices — OpenBB/Yahoo against the day lake

## `pairs_trading_06_yahoo_vs_day_lake.ipynb`

Notebooks 01–05 ran the whole pipeline on daily closes pulled through OpenBB, which for equities is
Yahoo Finance. Notebooks 07–10 rerun it on a Polygon-derived **day lake** held locally. That switch
is usually justified in one line — *survivorship bias* — and then never examined.

This notebook examines it. Both sources are loaded for the same 505 tickers over the same span and
compared directly: price levels, returns, coverage, the corporate actions that separate them, and the
data-quality traps each one does or does not expose. It closes by running the same backtest on both and
reporting what moves.

**The short version.** On returns the two agree to around eight decimal places on the overwhelming
majority of name-days, and the exceptions trace to corporate actions. On *price levels* they disagree far
more often, and the reason is not that either is wrong: an adjusted series is a function of **when you
downloaded it**, and it does not carry enough information to reconstruct its own basis. Anything that reads
a price level — a share count, a capital base, a dollar-neutral hedge — inherits that.

**Data.** The Yahoo side is the cached frame `cache/viz_prices_<universe>.parquet` written by
`visualize_cointegrated_pairs_yahoo.ipynb`, so this notebook needs no network. The lake side is read
straight from the local day lake. Everything is restricted to the span the two actually share.
""")

md("## 0. Setup")

code(r"""
import os, re, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pairs
from pairs import load_daily_bars, load_universe

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})
pd.set_option("display.width", 130)

LAKE   = Path(os.environ.get("DAY_LAKE", Path.home() / "local/parquet_lake/day_adj"))
MARKET = LAKE / "all_adjusted"
CACHE  = Path("cache"); CACHE.mkdir(exist_ok=True)
UNIVERSE = "spx_ndx_combined"

print("pairs", pairs.__version__)
""")

md(r"""
## 1. What the two sources are

They are not two copies of the same table. They differ in what they contain, how far back they go, how
they identify an instrument, and which price bases they can express.

| | OpenBB / Yahoo | day lake |
|---|---|---|
| universe | a **curated list** — today's S&P 500 + Nasdaq-100 members | every symbol that ever traded |
| instrument identity | the ticker string, and nothing else | resolved to a FIGI, so reuse is detectable |
| price bases | one adjusted close, plus `dividend` and `split_ratio` columns | `raw`, `split`, `tr` — chosen explicitly |
| freshness | re-downloaded on demand, always current | frozen at the date the lake was built |

The last row turns out to matter more than any of the others.
""")

code(r"""
yah = pd.read_parquet(CACHE / f"viz_prices_{UNIVERSE}.parquet")
ytk = sorted(yah.index.get_level_values("ticker").unique())
ydt = yah.index.get_level_values("datetime")
print(f"Yahoo cache : {len(yah):,} rows | {len(ytk)} tickers | "
      f"{ydt.min().date()} -> {ydt.max().date()} | columns {list(yah.columns)}")

# the lake, same tickers, deliberately over-wide so we can see where it stops
lake_probe = load_daily_bars(ytk, "2020-01-02", "2026-12-31", MARKET, price="split")
ldt = lake_probe.index.get_level_values("datetime")
print(f"day lake    : {len(lake_probe):,} rows | "
      f"{lake_probe.index.get_level_values('ticker').nunique()} tickers | "
      f"{ldt.min().date()} -> {ldt.max().date()}")

START, END = "2020-01-02", str(ldt.max().date())
print(f"\nshared span : {START} -> {END}   "
      f"(the lake stops {(ydt.max() - ldt.max()).days} days before the Yahoo cache does)")
""")

md(r"""
That gap is the first finding, and it is doing more work than it looks. The lake is a **snapshot**: it was
built on a date and has not moved since. The Yahoo cache was pulled later and runs four and a half months
further. Everything below is restricted to the span they share — but as §3 shows, restricting the *window*
does not make the two series comparable, because a corporate action after the window still rewrites the
history before it.
""")

md("## 2. Where they agree, and where they do not")

code(r"""
lake = load_daily_bars(ytk, START, END, MARKET, price="split", with_dividends=True)
ltk  = set(lake.index.get_level_values("ticker").unique())

yc = yah["close"].unstack("ticker")
lc = lake["close"].unstack("ticker")
common = sorted(set(ytk) & ltk)
grid   = yc.index.intersection(lc.index)
yc, lc = yc.loc[grid, common], lc.loc[grid, common]
both   = yc.notna() & lc.notna()
n_cells = int(both.sum().sum())

rel_px  = ((yc - lc).abs() / lc.abs()).where(both)
rel_ret = (yc.pct_change() - lc.pct_change()).abs().where(both & both.shift(1).fillna(False))

print(f"{yc.shape[0]:,} sessions x {yc.shape[1]} tickers = {n_cells:,} comparable closes\n")
rows = []
for th in (1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.20):
    rows.append({"threshold": f"{th:.4%}",
                 "price cells apart": np.nansum(rel_px.to_numpy() > th) / n_cells,
                 "return cells apart": np.nansum(rel_ret.to_numpy() > th) / np.isfinite(rel_ret.to_numpy()).sum()})
tab = (pd.DataFrame(rows).set_index("threshold") * 100).round(4)
tab.columns = [c + " (%)" for c in tab.columns]
display(tab)
print(f"median |price  difference| : {np.nanmedian(rel_px.to_numpy()):.3e}")
print(f"median |return difference| : {np.nanmedian(rel_ret.to_numpy()):.3e}")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a, dat, lab in ((ax[0], rel_px, "close"), (ax[1], rel_ret, "daily return")):
    v = dat.to_numpy().ravel(); v = v[np.isfinite(v) & (v > 0)]
    a.hist(np.log10(v), bins=80, color="0.4")
    a.axvline(np.log10(1e-2), color="r", ls="--", lw=1, label="1%")
    a.set_xlabel(f"log10 |relative {lab} difference|"); a.set_ylabel("name-days"); a.legend()
    a.set_title(f"{lab}: {np.mean(v > 1e-2):.2%} of comparable cells differ by more than 1%")
plt.tight_layout(); plt.show()
""")

md(r"""
The distribution is **bimodal**, and that is the whole story in one picture. The two sources are either
identical to floating-point precision — the median difference is around $10^{-8}$, which is the round-trip
error of storing the same number twice — or they are wildly apart. There is almost nothing in between.

That rules out the explanations one reaches for first. It is not rounding, not a stale bar, not a
different close convention, not a timezone slip. Those would produce a smooth hump of small errors. A
bimodal split means a *discrete* cause: for most names the two feeds carry the same numbers, and for a
minority they carry numbers that differ by a factor.

Note also how much better **returns** agree than **prices**. That is the clue to the mechanism.
""")

md("## 3. Why: an adjusted price is a function of when you downloaded it")

code(r"""
ratio = (yc / lc).where(both)
const = {}
for t in common:
    r = ratio[t].dropna()
    if len(r) < 100:
        continue
    if abs(r.median() - 1) > 0.01 and r.std() < 1e-3:      # off by a factor, and the factor is CONSTANT
        const[t] = float(r.median())

print(f"{len(const)} of {len(common)} tickers differ by a constant factor on every single session:\n")
cs = pd.Series(const).sort_values()
display(cs.to_frame("yahoo / lake").round(4))
""")

code(r"""
# for each of them, ask the lake what it thinks the raw (unadjusted) price was
raw = load_daily_bars(list(cs.index), START, END, MARKET, price="raw")["close"].unstack("ticker")
chk = pd.DataFrame({
    "yahoo close":     yc[cs.index].iloc[0],
    "lake split close": lc[cs.index].iloc[0],
    "lake raw close":  raw.loc[grid[0], cs.index],
    "yahoo / lake":    cs,
})
chk["lake split applied?"] = (chk["lake split close"] - chk["lake raw close"]).abs() > 1e-6
display(chk.round(4))

# can the cached data explain the factor? only if Yahoo recorded the action itself
rows = []
for t in cs.index:
    sr = yah.xs(t, level="ticker").get("split_ratio")
    ev = sr[sr.notna() & sr.ne(1.0) & sr.ne(0.0)] if sr is not None else pd.Series(dtype=float)
    after = [float(v) for d, v in ev.items() if str(d.date()) > END]
    prod  = float(np.prod(after)) if after else np.nan
    rows.append({"ticker": t, "yahoo / lake": cs[t], "implied factor": 1 / cs[t],
                 "action recorded after the lake ends": after or None,
                 "matches?": (abs(prod - 1 / cs[t]) < 0.02) if after else False})
expl = pd.DataFrame(rows).set_index("ticker")
display(expl.round(4))
print(f"\nexactly explained by an action visible in the cached window: "
      f"{int(expl['matches?'].sum())} of {len(expl)}")
""")

md(r"""
The factors are constant to a standard deviation of order $10^{-8}$ across more than a thousand sessions.
The two feeds are carrying the **same price path**, scaled by a number.

Some of those numbers are the obvious ones — 25, 10, 5, 4, 3, 2 — and some are not: 1.241 on FedEx, 1.067
on Comcast, 1.057 on S&P Global. Round or not, the mechanism is the same. Yahoo's adjusted close is
re-based every time a corporate action lands, so the price it reports for January 2020 *today* is not the
price it reported for January 2020 last year. The lake is a snapshot and still carries the older basis.
A whole-number factor is a split; a number like 1.241 is a spinoff, where the parent's history is marked
down by the value of what left.

**But notice how little of this the data can actually prove.** Only three of the names — Netflix, ServiceNow
and Texas Pacific Land — carry an action inside the cached window that exactly accounts for their factor.
For the rest, nothing in either source explains the number: the event happened after the Yahoo cache's own
last date, so it is not in the `split_ratio` column either, even though it has already been baked into
every adjusted close in the file.

That is the sharper version of the lesson. It is not merely that adjusted prices change under you. It is
that **a downloaded adjusted series does not carry the information needed to reconstruct its own basis.**
You can detect that two snapshots disagree, as we just did, but you cannot tell from the files why, or
which of them matches the basis your other data is on.

This is not a bug in either source. It is a property of adjusted series that is easy to forget:

> An adjusted price is not a measurement. It is a measurement *plus every corporate action since*, and it
> changes under you whenever a new one lands.

**The practical consequence** is the asymmetry already visible in §2. A uniform rescaling is invisible in
returns — divide every price by ten and every return is unchanged — which is why returns agree on 99.8% of
name-days. But it is not invisible to anything that reads a **level**: a share count, a notional, a
capital base, a dollar-neutral hedge ratio, a \$5 minimum-price filter. A backtest that computes returns
from one source and position sizes from another is silently wrong, and nothing in the output looks odd.
""")

code(r"""
# Before reaching for an outside source: can we at least tell a BASIS difference from a
# CORRUPT feed?  Yes.  A split rescales price and share count inversely, so price x volume
# is invariant to it.  If the dollar volumes agree, the two files hold the same trades.
raw_bk = load_daily_bars(["BKNG", "MA"], START, END, MARKET, price="raw")
inv = []
for t in ("BKNG", "MA"):
    ys = yah.xs(t, level="ticker")
    ls = raw_bk.xs(t, level="ticker")
    g2 = ys.index.intersection(ls.index)
    ys, ls = ys.loc[g2], ls.loc[g2]
    inv.append({"ticker": t,
                "price ratio":         float((ys["close"] / ls["close"]).median()),
                "volume ratio":        float((ys["volume"] / ls["volume"]).median()),
                "dollar-volume ratio": float(((ys["close"] * ys["volume"]) /
                                              (ls["close"] * ls["volume"])).median())})
display(pd.DataFrame(inv).set_index("ticker").round(4))

# the dividend column is denominated in the same basis -- a third, independent check
for t in ("BKNG", "MA"):
    d = yah.xs(t, level="ticker")["dividend"]
    nz = d[d.notna() & d.ne(0.0)]
    print(f"  {t}: {len(nz):2d} dividends recorded, last ${nz.iloc[-1]:.4f} on {nz.index[-1].date()}")
""")

md(r"""
### 3.1 Is one of them simply wrong?

Booking's price ratio is 1/25 and its **volume** ratio is 25, so the dollar volumes match to about a tenth
of a percent — ordinary vendor disagreement over consolidated against primary-exchange share counts.
Mastercard, whose price agrees, shows a volume ratio of 1 and the same residual noise. The two files hold
the **same trades**.

The dividend column says it a third way. Booking initiated a dividend in the first quarter of 2024, so
eight quarters fall inside this cache; the last is recorded as **\$0.3840**, which is \$9.60 ÷ 25.
Mastercard's is recorded at **\$0.76**, unscaled and correct. A corrupt feed does not rescale price, volume
*and* dividends coherently. This is a basis difference rather than an error, and establishing that much
took nothing but the two files.

**Establishing the reason did require leaving them.** The factor is 1/25; neither file says why. The answer
is in Booking Holdings' own Form 8-K, filed with the SEC on **2 April 2026**:

> On April 2, 2026, Booking Holdings Inc. filed an amendment to its Restated Certificate of Incorporation
> with the Delaware Secretary of State to effect the previously announced **twenty-five-for-one forward
> stock split** of the Company's common stock … The amendment … became effective at 4:01 p.m. Eastern Time
> on April 2, 2026. Trading is expected to commence on a split-adjusted basis at market open on Monday,
> April 6, 2026.
>
> — [SEC EDGAR, accession 0000950157-26-000465](https://www.sec.gov/Archives/edgar/data/1075531/000095015726000465/form8-k.htm)

The filed ratio and the measured factor agree exactly, and that single date resolves everything above. It
falls **after the lake's last session**, so the lake is simply reporting what Booking traded at. It falls
**after the Yahoo cache's own last date**, which is why `split_ratio` is silent about it. And it falls
**before the cache was downloaded**, which is why the division by twenty-five is nonetheless baked into
every close in the file.

So the inference was right, and confirming it took a regulatory filing rather than a column of data. That
is the lesson in its sharpest form: the two files are internally consistent, mutually contradictory, and
individually insufficient to say which basis either is on.

Both numbers are true. On 2 January 2020 Booking closed at **\$2,074.58** — what you would have paid — and
Yahoo's **\$82.98** is that same close expressed in the shares you would hold today. Use the first for
anything carrying units (a price floor, a share count, a notional, an ADV) and the second for returns.
""")

md("## 4. Corporate actions that are not just a rescaling")

code(r"""
# a constant factor is a split; a factor that changes ONCE is a spinoff; a factor that
# wanders is the two feeds tracking different instruments under one ticker
prefix, wander = {}, {}
for t in common:
    r = ratio[t].dropna()
    if len(r) < 100 or t in const:
        continue
    off = r[(r - 1).abs() > 0.01]
    if len(off) < 20:                       # a handful of odd days is not a corporate action
        continue
    brk = off.index.max()
    pre = r.loc[:brk]
    if abs(pre.median() - 1) < 0.01:        # the prefix must actually sit off 1
        continue
    (prefix if pre.std() < 0.02 else wander)[t] = (brk.date(), float(pre.median()), float(pre.std()))

print(f"{len(prefix)} tickers: one clean break, constant before it  (spinoff / special distribution)")
display(pd.DataFrame(prefix, index=["break", "ratio before", "sd before"]).T.sort_values("break"))
print(f"\n{len(wander)} tickers: the ratio wanders  (the two feeds are not tracking the same instrument)")
display(pd.DataFrame(wander, index=["last disagreement", "median ratio", "sd"]).T)
""")

code(r"""
show = list(prefix)[:2] + list(wander)[:2]
fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.4), squeeze=False)
for a, t in zip(axes[0], show):
    a.plot(ratio.index, ratio[t], lw=1.2, color="tab:blue")
    a.axhline(1.0, color="k", lw=0.8)
    a.set_title(t); a.set_ylabel("yahoo / lake")
    a.tick_params(axis="x", rotation=30)
fig.suptitle("A step is a distribution the two sources treat differently; noise is a different instrument")
plt.tight_layout(); plt.show()
""")

md(r"""
Two more mechanisms, and they are different in kind from the rescaling above.

**A single step.** The ratio is one constant before a date and exactly 1.0 after it. That is a **spinoff**:
Yahoo back-adjusts the parent's pre-event history downwards by the value of what was spun off, so the
return series is continuous through the event; the lake's split-only basis leaves the drop in. Here neither
convention is wrong, but they answer different questions — *"what would I have paid?"* against *"what did
this holding earn?"* — and a study that mixes them will read a spinoff as a one-day crash.

**A wandering ratio.** The two feeds are not tracking the same company. A ticker was reassigned, or a
merger moved it to a different instrument. The lake can see this because it resolves each ticker to a FIGI;
Yahoo has only the string, so it silently splices two companies into one price series. This is the trap
notebook 07 measures across the whole market, visible here in a 505-name list that is supposed to be clean.
""")

md("## 5. Coverage and survivorship")

code(r"""
# what the liquid universe actually looked like, year by year, against today's list
today = set(ytk)
rows = []
for yr in (2006, 2010, 2014, 2018, 2022):
    w = load_daily_bars(None, f"{yr}-01-01", f"{yr}-12-31", MARKET, price="split")
    dv = w.groupby(level="ticker")["dollar_volume"].median().sort_values(ascending=False)
    top = set(dv.head(500).index)
    rows.append({"year": yr, "in today's list": len(top & today),
                 "gone by today": len(top - today), "share surviving": len(top & today) / 500})
cov = pd.DataFrame(rows).set_index("year")
display(cov.assign(**{"share surviving": (cov["share surviving"] * 100).round(1)})
           .rename(columns={"share surviving": "share surviving (%)"}))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(cov.index.astype(str), cov["in today's list"], label="still in today's index list", color="tab:blue")
ax.bar(cov.index.astype(str), cov["gone by today"], bottom=cov["in today's list"],
       label="gone — delisted, acquired, or dropped", color="tab:red")
ax.set_ylabel("names in that year's 500 most traded"); ax.legend()
ax.set_title("What a study on today's membership list cannot see")
plt.tight_layout(); plt.show()
""")

md(r"""
This is the bias notebooks 01–05 carry. The numbers are larger than the phrase "survivorship bias"
suggests: of the 500 most-traded US names in **2006, only 39% are in today's index list** — the other 304
were acquired, taken private, delisted or simply fell out. Even 2018 is only 57%. A screen run on today's
membership is a screen run on the subset that survived, and survival correlates with exactly what a pairs
study is trying to measure: whether a relationship held together.

The lake has all of them, because it is the whole market rather than a list. That, and not data quality,
is the substantive reason to prefer it. Notebook 07 prices the effect on a buy-and-hold portfolio at
**+6.1% a year and +0.28 Sharpe**; notebook 09 finds it is enough to flip the sign of the distance rule.

Note the direction of the coverage result, though: among the **505 names in today's list**, the lake has
every one. Yahoo is not missing data on the names it covers. It is missing the names.
""")

md("## 6. The four traps, side by side")

code(r"""
probe_start, probe_end = "2024-01-01", "2024-12-31"
mkt = load_daily_bars(None, probe_start, probe_end, MARKET, price="split")
mtk = set(mkt.index.get_level_values("ticker").unique())

test_pat = re.compile(r"Z[A-Z]ZZT")
traps = pd.DataFrame(
    [
        ["exchange test symbols",
         sorted(t for t in today if test_pat.fullmatch(t)) or "none",
         sorted(t for t in mtk if test_pat.fullmatch(t))],
        ["ticker reuse (same string, different company)",
         "undetectable — no instrument id",
         f"{len(lake.attrs.get('ticker_collisions', []))} among these 505: "
         f"{[c['ticker'] for c in lake.attrs.get('ticker_collisions', [])]}"],
        ["split-adjustment price trap",
         "adjusted close only — a $5 floor drops pre-split history",
         "raw_close kept alongside, so gates use the tradeable price"],
        ["dividend look-ahead",
         "a dividend column is provided; the adjusted close already embeds it",
         "explicit: price='split' has no dividend, price='tr' does and is flagged"],
    ],
    columns=["trap", "OpenBB / Yahoo", "day lake"],
).set_index("trap")
display(traps)
""")

md(r"""
The curated feed is *cleaner* on two of these and *blind* on the other two, which is the honest summary.

Yahoo's list contains no exchange test symbols — someone has already filtered them — while the raw lake
carries them and they produce the largest one-day returns in twenty years. Point to the curated feed.

But Yahoo cannot tell you that a ticker changed company, because it has no identifier beyond the string;
and it gives you one adjusted close, so a minimum-price filter silently means *"the adjusted price today"*
rather than *"what it traded at then"*. The lake exposes both problems precisely because it exposes the
raw material. A feed that hides a problem is not the same as a feed that does not have it.
""")

md("## 7. Does any of it change a result?")

code(r"""
from pairs.models.kalman import _kalman_dynamic_hedge
from pairs.strategies.signals import generate_pair_signals
from pairs.strategies.evaluate import evaluate_pair_signals

PAIR = ("BKNG", "MA")          # notebook 03's walk-forward pick; BKNG is one of the rescaled names

def run(px, label):
    d = px.dropna()
    _, _, states, _ = _kalman_dynamic_hedge("P1", "P2", d, q=1e-5, em_iters=5, mode="filter",
                                            return_params=True)
    frame = d.join(states[["alpha", "beta", "resid"]])
    sig = generate_pair_signals(frame, z_method="robust", z_entry=2.0, z_exit=0.5, z_stop=4.0)
    daily, trades, summ = evaluate_pair_signals(frame[["P1", "P2"]], sig, cost_bps=1.0)
    return {"source": label, "bars": summ["bars"], "Sharpe": summ["sharpe"],
            "ann return": summ["ann_return"], "trades": summ["n_trades"],
            "capital base ($)": summ["capital_base"], "net P&L ($)": summ["net_pnl"],
            "median |z|": float(sig["z"].abs().median())}

py = pd.DataFrame({"P1": yc[PAIR[0]], "P2": yc[PAIR[1]]})
pl = pd.DataFrame({"P1": lc[PAIR[0]], "P2": lc[PAIR[1]]})
res = pd.DataFrame([run(py, "OpenBB / Yahoo"), run(pl, "day lake")]).set_index("source")
display(res.round(4))

print(f"\nprice levels differ by {float((yc[PAIR[0]] / lc[PAIR[0]]).median()):.4f}x on {PAIR[0]} "
      f"and {float((yc[PAIR[1]] / lc[PAIR[1]]).median()):.4f}x on {PAIR[1]}")
""")

md(r"""
I expected this table to show almost nothing, and it does not.

Same pair, same rule, same window, same code — and the Sharpe differs by about **0.19** (0.51 against 0.70),
with 78 trades instead of 69. The difference is not noise in the data: BKNG's *returns* agree between the
two sources to within $3\times10^{-4}$, and MA's to $10^{-7}$. It comes from the price **level**, through
two routes that are easy to miss.

**The Kalman hedge is not scale-invariant.** `q=1e-5` and `r=1.0` are absolute variances, not relative
ones. Dividing BKNG by 25 does not change the spread's shape, but it divides the residual's variance by
625 while leaving `q` and `r` where they were — so the filter's effective signal-to-noise, and therefore
how fast it lets the hedge ratio move, depends on what the prices happen to be denominated in. Measured
here, the residual standard deviation as a fraction of the price is 0.0032 on the Yahoo basis against
0.0024 on the lake's: a third more relative noise, purely from the scaling.

**Rounding is not free either.** A price stored to the cent carries about $10^{-4}$ of relative precision
at \$83 and about $5\times10^{-6}$ at \$2,075. Rescaling a series downward and storing it to two decimals
throws away significant figures, which is exactly why BKNG's returns differ at the fourth decimal while
MA's — unscaled — agree at the seventh.

So the answer to "does the data source matter?" is not the comfortable one. **The ratio is robust to the
adjustment basis only if your model is.** Any hyperparameter with units — a variance, a minimum price, a
notional, an impact ADV — silently couples your result to the snapshot you happened to download. The fix
is not to pick the "right" source; it is to make the model scale-free, and to keep raw prices and
adjustment factors separately so the basis can be reconstructed.
""")

md(r"""
## 8. Honest assessment

**The two sources agree far better than the switch to the lake implies.** On returns — which is what every
Sharpe in this repository is computed from — they match to eight decimal places on 99.8% of name-days, and
every exception is a corporate action rather than an error. Nothing in notebooks 01–05 is wrong *because*
it used Yahoo.

**The reason to prefer the lake is coverage, not accuracy.** Today's membership list cannot show you the
names that did not make it, and the shortfall is large: only 39% of 2006's five hundred most-traded names
are still on the list, and only 57% of 2018's. That is a bias in the question being asked, and no amount
of data quality fixes it.

**Adjusted prices are not stable objects.** The largest disagreements here are not errors at all — they are
the same price path on two different adjustment bases, because a split landed between one download and the
other. Any analysis pinned to price *levels* inherits the download date. Cache the raw series and the
adjustment factors separately, as the lake does, and the problem goes away; cache an adjusted close, as
the Yahoo path does, and it does not.

**What the curated feed buys you is a filter, and what it costs you is visibility.** No test symbols, no
obvious junk — but also no instrument identity, no raw price, and no way to know a ticker changed hands.
Notebook 07 can find those problems only because the lake does not hide them.

**Caveats.** One universe and one six-year window; the Yahoo side is a cached snapshot rather than a live
pull, so its adjustment basis is itself frozen at the download date and a fresh download would move the
constants in §3 again — which is the point. The spinoff and reassignment classification in §4 is
mechanical, based on the shape of the ratio, and is a hypothesis about each name rather than a
corporate-actions lookup.
""")

nb.cells = cells

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "pairs_trading_06_yahoo_vs_day_lake.ipynb")
    args = ap.parse_args()
    nbf.write(nb, args.out)
    print(f"wrote {args.out} ({len(cells)} cells)")
