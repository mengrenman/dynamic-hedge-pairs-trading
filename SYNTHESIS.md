# What twenty years of this data established

Seventeen notebooks, two parquet lakes, 2006–2025 US equities. `README.md` lists what the
repository contains; `HANDOFF.md` says how to run it and what will bite you. This document is the
argument: organized by claim, not by notebook.

Every number here is one the notebooks printed. Where a claim rests on a single sample, a single
year or an underpowered test, it says so — several of the most quotable figures in this project do,
and the difference between "we showed X" and "we failed to show not-X" is most of what the work
turned out to be about.

---

## 1. The bar, and the score against it

Notebook 14 §9 replaces the judgment call with five arithmetic clauses. A tradeable edge is a rule
whose expected profit per bet exceeds the cost of placing it, by a margin surviving (a) its own
sampling error counted in *independent* bets, (b) the number of rules searched to find it, and (c)
evaluation on data not used to choose it — at a size where the profit exceeds the cost of running
the operation.

The device that makes it measurable is putting both sides in the same units: **basis points of the
notional a bet turns over**. The backtest already charges `cost_bps` on traded notional, so gross
P&L on that base subtracts directly.

| clause | measured | verdict |
|---|---|---|
| (0) profit per bet beats cost per bet | 33.9 bps earned vs 2.2 paid (15.2×) | passes |
| (a) survives its own sampling error | t = 2.16 clustered on 30 formations; 18.3 bps observed vs 16.6 detectable | marginal |
| (b) survives the search that found it | Šidák over 1,738,998 screened tests wants 5.54 | **fails** |
| (c) survives out of sample | hold-out 13.1 bps vs development 19.3, at 14% power | unconfirmed |
| (d) large enough to run | $48,514 over 19 years, 2.6%/yr on capital deployed | too small |

**Four qualifications, all of which matter:**

*Clause (0) passes on the mean, and the mean is not the trade you would place.* The median round
trip earns **2.9 bps gross against a 1.9 bps cost** — near break-even. 5% of round trips (36 of 723)
carry **87%** of gross P&L, and **42% of round trips fail to cover their own cost**. The 15× ratio is
real and it describes a distribution carried by its tail.

*Clause (b) is the only decisive failure*, and it is decisive. No further data fixes it; only a rule
chosen in advance escapes a 1.74-million-test search.

*Clause (c) is not a failure.* The hold-out spans five formations with a 21.8 bps standard error, so
the smallest edge it could have called significant is 42.8 bps — more than twice the development
estimate. Its power against that effect is **14%**. A test that would miss a real effect six times
in seven has found nothing either way. (At the trade level the same comparison reads 3.7 against
36.6, which looks catastrophic; that construction is the one clause (a) rejects, because a
25-formation development sample containing a few enormous winners will always dwarf a five-fold
hold-out containing none.)

*The scorecard was run on the ungated book.* Notebook 11 §7–8's instrument gate removes the
leveraged and inverse funds and takes Sharpe from 0.402 to 0.309 and P&L from $43.6k to $32.5k.
Whether the more defensible gated book clears the same five clauses **was never scored**. It would
not clear them by more.

**So the honest headline is:** by this project's own bar, no tradeable edge is established —
decisively on (b), marginally on (a), and for want of statistical power on (c). That is not the same
as "there is no edge," and the difference is the point.

---

## 2. Costs were never the constraint — which reverses this project's own assumption

Every backtest here charged a flat 5 bps a leg-side, chosen as a plausible round number. Notebook 12's
break-even costs run from 0.7 to 5.7 bps, so the assumption sat *inside* the answer's range and was
deciding it. Measured on the minute lake instead:

| book | measured cost | against |
|---|---|---|
| nb11 pairs (287 pair-folds) | **2.2 bps** a transaction, notional-weighted | 33.9 bps of edge |
| nb12 cross-section (11,913 cells) | **2.94 bps** turnover-weighted | break-even 0.7–5.7 |

Running notebook 11's whole book at three cost levels: **0.407** at the assumed flat 5 bps,
**0.490** at measured costs, **0.545** entirely free. Measuring the cost is worth +0.083 of Sharpe;
abolishing execution altogether buys **+0.055 more**. Both sit well inside the 0.26 standard error a
Sharpe over this span carries.

**Hedge this number rather than banking it.** It is a single estimator (Roll 1984 half-spread) at one
sampling interval, and **a third of the cells fall below the minimum tick and are replaced by it** —
a floor-assisted measurement, not a clean one. The raw estimate is interval-sensitive (1.18 bps at
one minute, 1.63 at five, 2.71 at fifteen); five minutes is chosen because that is where the share of
arithmetically impossible estimates stops falling (44% → 33% → 33%), not because the estimate
converges. Corwin–Schultz was implemented as an independent check and abandoned: on true daily
OHLC with the overnight adjustment it returns about −10 bps and correlates −0.14 with Roll.

What survives the hedging is the direction and the order of magnitude: execution costs roughly half
what was assumed, and is an order of magnitude below the edge. **Cheaper execution does not rescue
this strategy.**

---

## 3. Breadth is the constraint, and two books fail in opposite directions

Net edge per bet is **0.145** of its own standard deviation. The Fundamental Law then gives the
annual information ratio as that times the square root of bets per year: 0.145 × √38 = **0.89**,
which is what the book delivers. Read as a specification: **IR 1.0 needs 47 independent bets a year,
IR 2.0 needs 190.** The book places 38.

Notebook 14 §6 reaches the same conclusion from the opposite direction — the Law predicts IR
1.14–3.38 from the measured IC and the book returns 0.40 — but both derivations run on the same
723 trades, so they corroborate the arithmetic, not the precision of the point estimate.

The repository contains both halves of the problem and never combines them:

| | edge per unit turnover | breadth | what binds |
|---|---|---|---|
| **nb11** pairs | 33.9 bps | 38 bets/yr | breadth |
| **nb12** cross-section | 0.33 bps *(post-2015)* | ~500 positions daily | the signal |

Notebook 12 has no discovery search, so no clause-(b) problem, and 500 positions a day. It should
have escaped every constraint above. Instead its **break-even cost collapsed from 10.8 bps before
2016 to 0.33 bps after** — four of ten configurations turn *negative* break-even, and zero of ten are
net-positive in the later era, against seven of ten before. That is a widely-known short-horizon
reversal signal being competed away on a datable timeline, and it is the best-evidenced decay result
in the repository.

### Notebook 15: the paper's own machinery doesn't change the verdict

Notebook 15 runs Avellaneda & Lee's actual configuration — sector-ETF residuals, the section 6
trading-time volume correction, and the bang-bang rule — as six books, reproducing notebook 12's own
baseline live (gross Sharpe 0.77/0.02, exact). None beats notebook 12 on a per-turnover basis when stock and ETF legs are pooled: the
best paper variant by break-even (ETF, trading time, continuous) earns **4.423 bps** against notebook
12's **5.703** — but notebook 12's book trades only stock legs, and counted on stock legs alone both
trading-time books clear it (**6.857** and **5.837 bps**), since the pooled column charges the hedged
books' ETF legs at the stock rate. Getting trading time right took fixing two independent bugs, not one; corrected, it
raises gross Sharpe +0.195 (bang-bang) to +0.157 (continuous) with turnover essentially unchanged,
and net of *measured* costs the trading-time bang-bang book is the single best-performing run in the
notebook — net Sharpe **0.366** against notebook 12's **0.180**. Post-2016 survival is mixed: five of
six runs decay on the usual split, only calendar/bang-bang does not (0.533 → 0.548) — but five-year
blocks show that is a 2016-cutline artifact (that run has the most blocks below 0.15 Sharpe, 2 of 4),
while trading-time/bang-bang is the steadiest run on the finer cut (zero blocks below 0.15) despite
an unremarkable two-decade drop (0.784 → 0.698).

---

## 4. The signal was thin before any of this

Notebook 10 ran **1,738,998** dual-gate cointegration tests. Storey's π̂₀ = **0.931**: 93% of pairs are
indistinguishable from noise. Benjamini–Hochberg keeps **2,151**; the median formation yields **three**
survivors and **nine of thirty-nine yield zero**. Without correction, 86,950 would pass.

The survivors are not what the method advertises. The most reliable are **ETF index clones** —
IVV/SPY in 15 of 39 formations, SPY/VOO in 10 — arithmetic identities rather than economic
relationships. On the Yahoo universe, notebook 6 finds the screen hub-dominated: two names (NCLH,
CCL) carry 250 of 279 edges, and only 23 of 279 survivors have a tradeable static half-life.

**And the diagnostic that made pairs look stationary was measuring itself.** Notebook 7 applies the
package's own Kalman-residual stationarity test to two *independent random walks* at price-like
levels: it declares them "stationary" **90% of the time** at the default q=1e-5, while Engle–Granger
on levels stays near its nominal ~8%. A time-varying-coefficient filter manufactures a stationary
residual for any two I(1) series. Of eight real pairs, only V/MA is genuinely cointegrated — and its
persistence variance is ≈0, meaning *fixed*, not time-varying, cointegration. Those eight are a
hand-picked convenience sample, so they establish that the test is unreliable, not what fraction of
pairs are cointegrated; notebook 10's π̂₀ is the population number.

**Nothing in this repository demonstrates that a dynamic hedge is warranted.** Notebook 11 confirms it
empirically: the frozen static OLS hedge beats the Kalman hedge **0.402 against 0.315**, trading 723
round trips against 1,698 — the filter re-estimates faster than the spread reverts and absorbs the
signal into its state.

---

## 4b. The dynamic hedge is the wrong model almost everywhere

Notebook 07 showed the package's Kalman-residual test calls two independent random walks
"stationary" 90% of the time, and inferred from eight hand-picked pairs that a dynamic hedge is
rarely warranted. `pairs.recommend_hedge` now tests that properly, and it has been run over **all
2,151 BH dual-gate survivors** (weekly log prices, each pair's own 2-year formation window,
B = 199 — `analysis/gate_screen_survivors.py`, 84 minutes on 12 cores).

| hedge the evidence supports | survivors | notebook 11's 287 traded pair-folds |
|---|---|---|
| none — no cointegration | 1,191 (55.4%) | 165 (57%) |
| static — fixed coefficient | 802 (37.3%) | 87 (30%) |
| **dynamic** — coefficient demonstrably moves | **158 (7.3%)** | **35 (12%)** |

**And most of the `dynamic` verdicts do not survive inspection.** The model constrains `T` to
(0, 1) but leaves θ free, and the optimizer wanders: **73% of the time-varying verdicts rest on a
negative θ̂** — an error that alternates sign every week, which is oscillation rather than a
long-run relation — and **30% on a |θ̂| > 1**, outside the stationary region altogether, which is
the null the test exists to reject. Requiring θ̂ ∈ (0, 1) leaves **38 of 2,151 survivors (1.8%)**
and **6 of notebook 11's 287 traded pair-folds (2.1%)**.

So notebook 11 applies a Kalman dynamic hedge to all 287 pair-folds it trades, and the evidence
supports one on about six of them. That is the direct explanation of §4's empirical result, where
the frozen OLS hedge beats the Kalman hedge 0.402 to 0.315 while trading 1,698 round trips against
723: the filter is chasing a coefficient that, for all but a handful of pairs, is not moving.
`recommend_hedge` now downgrades a time-varying verdict to `static` when θ̂ leaves (0, 1), and
reports the paper's classification unchanged alongside it.

**Two cautions, and the first one cuts against reading too much into the 55% `none`.** Among those
verdicts the median θ̂ is 0.782 — visibly below 1 — yet the median bootstrap p is 0.291, and **77%
have θ̂ < 0.9 while failing to reject θ = 1**. That is the test seeing decay it has no power to
certify, exactly as notebook 07 §4's size study predicts at n ≈ 104. The `none` column is mostly
"cannot tell", not "demonstrably absent", and the disagreement with Engle–Granger is flat across
the EG p-value quartiles (57.6% `none` in the strongest quartile against 55.0% in the weakest),
which is what low power looks like rather than a systematic contradiction.

**Second — and this is the part that turned out to matter — both series were run, and they agree
on the rate while disagreeing on the pairs.** Notebook 10 screened *daily levels* and notebook 11
fits its hedge to them, so the weekly form is a departure from the pipeline, taken because it is
notebook 07's choice for real pairs and its Monte Carlo's sample size. Running both over the 287
traded pair-folds:

| on the 287 traded folds | none | static | dynamic (raw) | dynamic, θ̂ ∈ (0,1) |
|---|---|---|---|---|
| daily levels (~504 bars) | 79.4% | 17.8% | 8 (2.8%) | **8 (2.8%)** |
| weekly logs (~104 bars) | 57% | 30% | 35 (12.2%) | **6 (2.1%)** |

**They agree on 52% of folds — a coin flip — and exactly one fold is `dynamic` under both.** The
*identity* of the pairs that warrant a filter is not established at all. The *rate* is: two
estimators, on differently transformed data, disagreeing pair by pair, both land at 2–3%.

Neither estimator is well behaved, and they fail in opposite directions, which is worth knowing
before anyone leans on θ̂ itself:

| | median θ̂ | θ̂ < 0 | \|θ̂\| > 1 | θ̂ ∈ (0,1) |
|---|---|---|---|---|
| daily levels | +0.999 | 2% | **44%** | 55% |
| weekly logs | +0.776 | 19% | 22% | 68% |

Weekly wanders negative; **daily pins against the unit root** — median 0.999, with 44% overshooting
past 1, twice the weekly rate. That is the familiar near-unit-root estimation problem: a two-year
daily spread is persistent enough that the MLE sits on the boundary, θ = 1 cannot be rejected, and
the daily form returns `none` for 79% of folds close to by construction. Weekly's 0.776 is roughly
what a daily θ ≈ 0.95 implies under five-day aggregation, so the weekly estimate is arguably
reading the persistence *more* informatively despite the shorter sample.

So the defensible claim is narrow and holds regardless of the choice: **about 2–3% of the pair-folds
notebook 11 trades show a coefficient that moves, and notebook 11 applies a dynamic hedge to all of
them.** Anything stronger — which pairs, or how much the rate varies by era — the data does not
support.

**The full population, both forms.** The daily-levels run has since finished over all 2,151
survivors (129 minutes on 12 cores; `converged` recorded for the last 751 cells, and θ̂ sitting
exactly on its 0.9 start flagged post hoc on the first 1,400):

| all 2,151 survivors | none | static | dynamic (raw) | dynamic, θ̂ ∈ (0,1), converged |
|---|---|---|---|---|
| daily levels | 62.4% | 34.7% | 63 (2.9%) | **63 (2.9%)** |
| weekly logs | 55.4% | 37.3% | 158 (7.3%) | **38 (1.8%)** |

The rate holds at population scale — 2–3% on either clock — and the population and the traded
subset barely differ on it (daily, untraded cells: 59.8 / 37.3 / 3.0%). The identity does not
hold: the two forms agree on **52.8%** of cells (51.6% traded, 53.0% untraded), 210 cells are
`dynamic` on one clock, and **11 (0.5%) on both**. θ̂'s pathology is milder over the population
than on the traded folds but has the same shape: on daily levels |θ̂| > 1 for 33.2% of cells, θ̂
stuck exactly at its 0.9 start for 4.6%, non-convergence reported on 6.4% of the cells that
recorded it — 38.7% untrustworthy in all, median θ̂ 0.995; on weekly logs |θ̂| > 1 for 13.9%,
negative for 13.5%, 15.6% untrustworthy, median 0.743. The bootstrap p-value for θ = 1 has median
0.30 on daily levels against 0.07 on weekly: the daily form mostly cannot reject the unit root,
which is why `none` rises to 62% over the population and to 79% on the traded folds — the most
persistent spreads by selection.

**Notebook 16 adds external support, from outside this repository's own pipeline.** It reproduces
Palomar (2025) ch. 15's own EWA-EWC / KO-PEP Kalman pairs backtest — a textbook example built to
showcase the Kalman filter's advantage — and finds **75–85%** of that advantage (EWA-EWC) and
**~90–108%** (KO-PEP) is the filter re-marking its own parameters against today's price, not P&L a
trader could collect: EWA-EWC's tradable line sits at **0.528–0.550** under all three hedges tested
(rolling LS, basic Kalman, momentum Kalman; SE 0.18–0.23) even as the book line ranges
0.634–3.290, and KO-PEP's tradable line (0.076 / 0.120 / -0.046) is zero within noise under all
three. The tradable line is **hedge-invariant** — this section's own finding, that the dynamic
hedge buys almost nothing real, reproduced on someone else's pair and someone else's code. And this
repository's own accounting is clean for the right reason: `evaluate_pair_signals` applies executed
holdings to price changes directly, confirmed by showing a naive diff-of-spread line on the same
holdings would overstate real P&L by 2.13x.

---

## 5. What is actually inside the P&L that exists

- **2022 alone is 52%** of twenty years of P&L ($22.7k of $43.6k).
- **87% of gross P&L comes from 5% of round trips** (36 of 723).
- **39% of pair-folds involve a leveraged, inverse or volatility ETF** (112 of 287), contributing a
  proportionate 39% of P&L — they are not where the edge is concentrated, but over a third of what a
  cointegration screen finds in liquid US equities is mechanical rather than economic.
- Gating those instruments costs **about a quarter of the P&L and a fifth of the Sharpe**
  (0.402 → 0.309, $43.6k → $32.5k).

A strategy whose average rests on 36 observations out of 723, and whose twenty-year record is one
good year, does not have twenty years of evidence for itself.

---

## 6. Every out-of-sample test came back weaker, and none of them settles anything

| test | development | hold-out |
|---|---|---|
| nb14 §9 clause (c), clustered | 19.3 bps, t = 2.06 | 13.1 bps, t = 0.60 — **14% power** |
| nb11 §8, six rule × gate cells | all positive | **all six decline** |
| nb18→17 intraday | pooled OOF 1.50 | −0.27 on 153 sessions, s.e. ≈ 1.3 |
| nb04 tuning | OOF 1.338 vs default 0.508 | dead heat, 2.635 vs 2.642 |
| nb05 tuning, 40 pairs | tuned wins on validation | static hedge first on hold-out |

The direction is consistent and the individual tests are all underpowered. Notebook 11 §8 notes that
six of six declining has probability 2⁻⁶ = 1.6% under independence — but the six cells are three
rules crossed with gate on/off and share most of their trades, so treat that as illustrative rather
than a test. Notebook 19 states its own version plainly: ten pairs and two and a half years "cannot
separate 1 from zero."

**The honest summary is not that the edge decayed. It is that every attempt to confirm it out of
sample produced a weaker number, and no single one of those attempts could have confirmed it even
if the edge were entirely intact.**

---

## 7. The comparison that keeps it all honest

An equal-weight buy-and-hold foil on the same 240 tickers scores **Sharpe 0.476** over the same
sessions. The pairs book scores **0.490** at measured costs.

Twenty years of data, 1.74 million cointegration tests, a Kalman hedge, a walk-forward harness and
723 trades bought **0.014 of Sharpe** over holding the same stocks.

The two are not the same product — the book is market-neutral (beta 0.052) with alpha 1.59%/yr at
t = 1.39, the foil is market beta with *negative* alpha — so what the book demonstrably bought, if
anything durable, is low market correlation rather than excess return. On raw risk-adjusted return
it is a wash.

---

## 8. What transfers beyond this repository

Strictly, in descending order of confidence:

1. **Breadth arithmetic is arithmetic.** IR ≈ (edge/σ per bet) × √(bets per year). Nothing in it is
   negotiable, and it caps any design that places few bets regardless of execution quality.
2. **Search-based discovery carries a multiple-testing tax that is usually fatal.** Screening a
   million pairs to find twenty is not a detail of this implementation — it is what "find
   cointegrated pairs" *means*, and it is what clause (b) charges for.
3. **Measure the assumption that sits inside your answer's range.** Notebook 12's verdict turned on
   an assumed 5 bps against break-evens of 0.7–5.7. An hour of measurement changed six of ten
   configurations from negative to positive — and left the conclusion intact for a different reason.
4. **The unit of independence is rarely the row.** Clustering the same trades on 30 formations
   instead of 723 trades moves t from 3.89 to 2.16, and turns an apparent 90% out-of-sample collapse
   into a 32% decline.
5. **A single statistic is not a verdict.** Against a matched placebo of random uncointegrated pairs,
   the same strategy sits at the **97th percentile** of the null by median per-fold Sharpe and the
   **2nd** by mean. Both are correct. Cointegration screening widens the outcome distribution rather
   than shifting it, and which estimator you report decides what you conclude.
6. **Diagnostics can measure themselves.** A time-varying-coefficient filter produces a stationary
   residual from pure noise; any test applied to its output inherits that.

Method findings that are real but narrower: a survivor-only universe can flip the *sign* of a
Sharpe for a strategy that loses money either way (notebook 11 §9), so check dollar P&L alongside
ratios; and hyperparameter tuning on this pipeline is noise — split-half Spearman ρ = 0.044.

---

## 9. What this work does not tell you

- **Capacity, except for one book.** Notebook 19 models square-root impact on the *intraday* book:
  24 bps of capital at $10k a pair, 77 at $100k, **244 at $1M**, so the practical ceiling is nearer
  $100k a pair than the $135M that a participation-only measure suggests. The two measures disagree
  because participation asks whether the order fits in the bar and impact asks what it costs to
  insist on it. **The day-lake book's capacity was never measured this way** and the intraday number
  does not transfer to it.
- **Borrow availability and cost, financing, taxes, margin, operational risk, crisis behavior.**
  None of these were modeled. Borrow is a flat 50 bp/yr assumption throughout.
- **Anything about the future.** Every result here is in-sample or from an underpowered hold-out.
- **The gated book's standing** against the five clauses, as noted in §1.
- Two known accounting defects, documented in `README.md` and `HANDOFF.md`: the per-trade log omits
  the exit bar's cost (~2% on profit factor) and `filter_kf_on_new` skips the fold-boundary predict
  step. Neither touches the daily ledger that every Sharpe, return and drawdown comes from.

Nothing in this repository is investment advice, and none of it was written by anyone licensed to
give it.

---

## 10. What would change the answer

The two halves exist separately and have never been combined: notebook 12's breadth with a signal
that has an economic anchor rather than a statistical one. That design would score differently on
every clause — no discovery search (b), hundreds of bets a year (a, d), and a reason for the return
that competition erodes slowly rather than quickly.

The measurement apparatus is the durable asset here: the cost measurement, the placebo nulls, the
walk-forward harness, the clause table, and a test that checks the prose against the outputs.
Pointing it at a new domain is cheaper than refining this one.

A closing observation the record supports. Seventeen notebooks of careful work produced zero clear
edges, and nearly all the value realized came from **measuring things the design had assumed** —
costs, survivorship, the spuriousness of the Kalman residual, what a hold-out can and cannot see.
That ratio is not a sign something went wrong. It is what the work is.
