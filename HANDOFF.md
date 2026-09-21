# Handover notes

`README.md` describes what this repository *contains*. This file describes what it **found**, what
will **bite you**, and what is **still open** — the things a feature list cannot carry.

Read §1 and §4 before touching anything.

---

## 1. The bottom line

Seventeen notebooks, twenty years of US equities, two data lakes. **No tradeable edge was
established.** That is the result, not a failure to finish.

Notebook 14 §9 makes "tradeable edge" a measurement rather than a judgement — five clauses, each
arithmetic — and scores the main strategy on all five:

| clause | measured | verdict |
|---|---|---|
| (0) profit per bet beats cost per bet | 33.9 bps earned vs 2.2 bps paid (15.2×) | **passes** |
| (a) survives its own sampling error | t = 2.16 clustered on 30 formations; 18.3 bps observed vs 16.6 detectable | marginal |
| (b) survives the search that found it | Šidák on 1,738,998 screened tests wants 5.54 | **fails** |
| (c) survives out of sample | hold-out 13.1 bps vs development 19.3; 14% power | unconfirmed |
| (d) large enough to run | $48.5k over 19 years, 2.6%/yr | too small |

**Costs are not the problem.** That was the working assumption for most of the project's life and it
is measurably wrong. Execution was measured on the minute lake (nb14 §9.1, nb12 §5.5) at **~2 bps a
transaction, not the 5 bps every backtest assumed**. Running the whole book at *zero* cost adds
0.055 of Sharpe. The binding constraints are **breadth** and **the selection search**.

The breadth arithmetic is the most transferable thing here. Net edge per bet is 0.145 of its own
standard deviation; at 38 bets a year that gives IR 0.89, which is what the book delivers. IR 1.0
needs 47 bets a year, IR 2.0 needs 190.

**Two half-strategies, failing differently.** nb11 (pairs) has the per-bet edge and no breadth.
nb12 (cross-sectional, 500 positions daily, no discovery search so no clause-(b) problem) has the
breadth and lost the signal: its break-even cost fell from **10.8 bps before 2016 to 0.33 bps
after**, with zero of ten configurations net-positive in the later era. Nothing in the repo
combines the two, and that gap is the most interesting place to pick up.

**The benchmark that keeps everything honest:** equal-weight buy-and-hold on the same 240 tickers
scores Sharpe 0.476 over the same sessions. The pairs book scores 0.490 at measured costs. All of
this bought 0.014 of Sharpe over doing nothing clever. (They are not identical products — the book
is market-neutral, beta 0.052, alpha 1.59%/yr at t = 1.39; the foil is market beta with *negative*
alpha. But on raw Sharpe it is a wash.)

---

## 2. Running it

```bash
conda env create -f environment.yml && conda activate stat-arb   # python 3.12, env name "stat-arb"
pip install -e .
pytest -q                                                        # 445 tests, ~15 s
```

**The data lakes are read-only and not in the repo.** Adjusted Polygon.io (now Massive.com) parquet:

| lake | default path | override |
|---|---|---|
| day | `~/local/parquet_lake/day_adj/all_adjusted` | `DAY_LAKE` |
| minute | `~/local/parquet_lake/minute_adj/all_adjusted` | `MINUTE_LAKE` |

Both have a *market* layout (all tickers per file, ~10k symbols/day) and a *ticker* layout. Most
notebooks use market layout; nb15–17 use the minute lake's `spx_ndx_combined_adjusted` ticker
layout — see the survivorship warning in §4.

**The web explorer** (three pages over the same package, so it cannot diverge from the notebooks):

```bash
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8077
```

### The notebook build cycle

Most notebooks are **generated from Python**, so code and prose diff like source. Never hand-edit a
generated `.ipynb` except for markdown-only patches (see §4).

```bash
python notebooks/build/build_<name>_notebook.py     # writes cells, clears outputs
cd notebooks && python build/execute.py pairs_trading_NN_<name>.ipynb
```

`notebooks/build/README.md` has the script→notebook table with cold-run times. Rebuild order and the
expensive steps:

```
09 (~3 min) ──> 10 (~1 HOUR, 1.74M tests) ──> 11 (~5 min) ──> 13 (~13 min)
                                          └─> 14 (~6 min)
12 (~11 min, independent)      15 (~3 min) ──> 16 (~10 min) ──> 17 (~5 min)
```

`notebooks/cache/` is **1.8 GB and gitignored**, so a fresh clone rebuilds everything. Notebook 10
is the one that hurts; protect its `day_rule_*.parquet` and `day_screen.parquet`.

`notebooks/build/renumber.py` renumbers the whole series atomically (`--dry-run`, then `--apply`).
It is **not idempotent** — apply exactly once per mapping.

---

## 3. What each notebook concluded

| nb | conclusion |
|---|---|
| 01–03 | Pipeline demo on OpenBB/Yahoo. All three select BKNG/MA, in-sample Sharpe 0.516, OOS 2.642 on 12 trades. **nb03's finding is real and general**: a significance filter placed upstream of a stricter rank cut is decorative — BH keeps 279 of 118,828 pairs, but the top-200 rank cut downstream makes the gate change nothing. Contrast nb11, where FDR *is* the selection and is worth +0.40 Sharpe. |
| 04 | Tuning the Kalman hedge is not worth it. Best pooled OOF Sharpe 1.338 vs default 0.508, but split-half Spearman **ρ = 0.044** (indistinguishable from noise), it degrades all three runner-up pairs, and it is a dead heat with the default out of sample (2.635 vs 2.642). |
| 05 | Same at portfolio scale over 40 pairs. Tune→validation rank correlation 0.36 (Kalman), −0.03 (static). The static hedge goes best-on-tune → mid-table-on-validation → first-on-hold-out. Winner's curse, not a bad grid. |
| 06 | Universe-wide screening is hub-dominated: 279 survivors among 239 tickers, but two hubs (NCLH, CCL) carry 250 of 279 edges. Only 23 of 279 have a tradeable static half-life. |
| 07 | **The most important methodological result.** The package's Kalman-residual stationarity test declares two *independent random walks* "stationary" **90% of the time**. Engle–Granger on levels stays near its nominal ~8%. Of 8 real pairs tested, only V/MA is genuinely cointegrated — and its persistence variance is ~0, i.e. *fixed*, not time-varying. A dynamic hedge is not demonstrably warranted anywhere in this repo. |
| 08 | Yahoo vs day lake: **returns agree** to floating point on 99.8% of name-days; **price levels do not** (4.6% of raw cells differ >1%). Since the Kalman hedge is not scale-free, absolute-variance parameters (`q`, `r`) do not transfer between sources. |
| 09 | Data quality. Survivorship is worth **+6.05%/yr and +0.28 Sharpe** before any strategy. `close_tr` embeds dividend look-ahead (AAPL marked down 15.9% in 2003). 10% of lake tickers name more than one instrument. 13 NASDAQ test symbols produce 3 of the 4 largest one-day returns in 20 years. |
| 10 | 1,738,998 dual-gate tests. Storey's π̂₀ = **0.931** — 93% of pairs indistinguishable from noise. BH keeps **2,151**; the median formation yields **3** survivors and **9 of 39 yield zero**. The most reliable survivors are ETF index clones (IVV/SPY), not economic pairs. |
| 11 | The main portfolio. BH dual-gate Sharpe **0.402 ± 0.26** (0.490 at measured costs); raw-p 0.04; distance −0.41. FDR control is the whole edge. 39% of pair-folds involve a leveraged/inverse/vol ETF, proportionate to count. Kalman hedge *underperforms* frozen static OLS (0.28 vs 0.40). 2022 alone is **52%** of 20-year P&L. |
| 12 | Cross-sectional Avellaneda–Lee. Breadth instead of pair selection — and it is not enough. Gross Sharpe 0.77 (2006–15) vs **0.02** (2016–25). Break-even collapses 10.8 → **0.33 bps**. Best full-span net Sharpe 0.18 (t = 0.8). |
| 13 | Grande & Borondo (2025) replication. Our `pairs.stats.network` reproduces all 21 published Fig. 2 values **exactly** (ties must be floor-truncated, not mid-ranked). Their peripheral-beats-central claim **does not replicate**: TMFG t = 0.24, p = 0.811. PMFG shows +0.087 (p = 0.008) but the notebook self-flags it as one sign out of two. |
| 14 | What "alpha" means, four ways, measured. Headline IC 0.197/fold is reproduced by **random uncointegrated pairs** (0.205). Fundamental Law predicts IR 1.14–3.38; book delivers 0.40. §9 is the tradeable-edge definition in §1 above. |
| 15–17 | Intraday. Half-life is ~16.6 days — **the pair does not revert intraday**, so minute bars buy timing, not alpha. 39% of spread variance is overnight. Re-estimating the hedge faster than daily catastrophically whitens the spread. Hold-out: all four designs within 1 s.e. of zero. |

---

## 4. Traps

These each caused a real bug, some more than once.

**Prose drifts from outputs, and the drift inverts conclusions.** Commit `3a1965f` re-executed
notebooks on a changed shortlist without rewriting nb04 §4b.5. Three days later *three of its six
conclusions were backwards* — it said the tuned config traded more when it traded a third less, that
two of three pairs improved when none did. `tests/test_notebook_prose.py` now asserts every number
in prose was printed by some notebook. **It is not sufficient**, and nb01 proved it twice: it catches
changed values, not reversed directions, not claims with no numbers in them, and not a stale figure
that happens to round to *some* number printed elsewhere in the repo (nb01 carried "Sharpe 2.00 on 4
trades" against an actual 2.642 on 12, and the test passed). After any re-execution, re-read the
prose for *direction words* — more/less, improves/worsens — against the new output.

**`mode="smooth"` is a look-ahead trap.** The RTS smoother conditions every state on the whole
sample. Always `mode="filter"` (the default) for anything feeding a signal, backtest, stationarity
test or selection score.

**`em_iters > 0` silently discards your `init_cov`.** pykalman's `em_vars` default re-estimates
`initial_state_covariance` too, collapsing a deliberate 1e6 diffuse prior to ~1e-4. Pass
`em_vars=("transition_covariance", "observation_covariance")` to keep it.

**The Sharpe column is spelled `shapre`.** Deliberately, and notebooks depend on it. Reading it with
`.get("sharpe_train", default)` silently returned zeros for four notebooks, so a documented
five-metric composite score only ever had four live terms. **Never read a column with `.get` and a
default** — index it directly so a rename fails loudly.

**`cost_bps` is charged per transaction on `|Δn1|·P1 + |Δn2|·P2`** — both legs, each side. Not per
round trip, not per pair-notional. A pair's effective rate is its two legs' rates weighted by the
notional each *turns over*; a plain mean over-weights the smaller leg and overstates cost by ~30%
(67% of pair-folds are outside a 40/60 split).

**Kalman-residual stationarity is a filter artefact.** See nb07. Never read ADF/KPSS on a Kalman
residual, or a half-life computed from one, as evidence of cointegration.

**Notebooks 01–05 are pinned. The rule is narrower than "never re-execute".** Their prices live in
`notebooks/cache/nb0N_train_*.parquet` and the notebooks read them when present, so re-executing
with the cache in place is offline and reproducible — verified on nb01, where every substantive
output is byte-identical across runs. What must never happen is **deleting the cache**: a fresh
OpenBB/Yahoo download re-bases adjusted prices for any name that has since split, and moves every
number for reasons unrelated to whatever you were fixing.

Two caveats if you do re-execute. Snapshot the outputs first and diff cell by cell afterwards —
that is how the seeded-sample problem below was found. And expect two benign diffs: tqdm timing
lines, and anything genuinely random. **Markdown-only patches avoid all of this** — patch the
`.ipynb` cell in place *and* the builder from the same string so they cannot drift.

**The instrument gate applies only inside nb11 §7–8.** nb10, nb12, nb13 and nb14 all describe the
**ungated** universe. Only nb11 §8 has the gated and held-out figures.

**Clustering changes conclusions, not just error bars.** nb14's clause (a): t = 3.89 treating 723
trades as independent, **2.16** clustered on the 30 formations. Applying the wrong one to the
hold-out made a 32% decline look like a 90% collapse. Trades inside a formation share a hedge, a
universe and a market — always cluster on formations.

**An unseeded `.sample()` makes a notebook un-reproducible for no benefit.** nb01 displayed
`summary_dual.sample(20)` with no `random_state`, so every re-execution produced a spurious diff on
that cell alone — noise in exactly the notebook whose pinning is meant to keep re-execution quiet.
Now seeded. nb02–17 were checked and have none. If you add an illustrative sample, seed it.

**`renumber.py` does not scan `webapp/` or `figures/`.** It covers `notebooks/build/*.{py,md}`,
`pairs/**/*.py`, `tests/*.py`, `README.md`, `HANDOFF.md` and the notebooks. Grep the other two by
hand after `--apply`.

**A triple-quoted docstring inside `code(r"""...""")` closes the outer string** and the builder dies
with a confusing `SyntaxError`. Use `#` comments inside builder code cells. This has bitten three
times.

---

## 5. Known open

| what | where | why it is still open |
|---|---|---|
| nb10 and nb12 never re-run with the behavioural instrument gate | `build_daily_screen_notebook.py`, `build_cross_sectional_notebook.py` | Would require regenerating nb10's full 1.74M-test screen (~1 h) and everything downstream. The principled version of issue #6. |
| Trade log omits the exit bar's cost; `filter_kf_on_new` skips the fold-boundary predict step | `pairs/strategies/evaluate.py`, `pairs/models/kalman.py` | Documented in README. Affects per-trade stats (~2% on profit factor), not Sharpe/return/drawdown, which come from the daily ledger. |
| `TVCointModel` still lives in nb07, not in `pairs/stats/` | `pairs_trading_07_...ipynb` | Promotion with tests was planned and not done. |
| The synthesis document | — | Never written. §1 and §3 of this file are the closest thing. |
| GitHub issue #1 | — | **Not actionable.** An unsolicited third-party pitch to integrate an external forecasting API, not a defect or a design ask. Treated as untrusted data; close it whenever. |

---

## 6. Settled — do not redo

- **Hyperparameter tuning of the Kalman hedge.** nb04 and nb05, single-pair and portfolio scale. The
  ranking is noise (ρ = 0.044); the winner's curse is the reason.
- **Whether costs are the binding constraint.** No. Measured at ~2 bps against a 33.9 bps edge.
- **Whether a Kalman dynamic hedge beats a frozen static OLS hedge.** It does not — 0.28 vs 0.40 on
  the day lake (nb11 §4), and worse intraday (nb16 §3.1). nb07 shows the dynamic hedge is not even
  demonstrably warranted.
- **Whether the pair reverts intraday.** It does not; half-life is ~16.6 days.
- **Whether Grande & Borondo's peripheral-pairs claim replicates here.** It does not (nb13 §6).
- **Whether survivorship bias matters.** +6.05%/yr and +0.28 Sharpe (nb09 §7).

---

## 7. Where to pick up

In rough order of expected value:

1. **Combine nb12's breadth with an economically anchored signal.** The measurement apparatus
   already exists — cost measurement, placebo nulls, walk-forward, the five-clause table — so a new
   signal can be scored in days, not weeks. That apparatus is the durable asset in this repo, and
   pointing it at a new domain is cheaper than refining this one.
2. **Add a θ-persistence gate** to the Kalman path, per nb07's finding — currently nothing stops the
   pipeline using a dynamic hedge on a pair with no time-varying cointegration.
3. **Re-screen nb10/nb12 with the instrument gate**, if you want the principled version of issue #6
   and are willing to spend the hour.

A closing caution the record supports: seventeen notebooks of careful measurement produced zero
clear edges, and most of the value realised here came from *measuring things the design had assumed*
— costs, survivorship, the spuriousness of the Kalman residual, the meaning of the hold-out. Expect
that ratio to continue.
