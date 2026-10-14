# Pre-registration: notebook 24, the hedge in logs (Avellaneda–Lee's Eq. 1 against the level hedge)

**DRAFT, 2026-10-13. Not registered.** Sections marked **[OPEN]** are decisions to make before
this file is committed as the registration. "Facts established while drafting" records the
formation-window measurements taken on 2026-10-13, and what they changed. They are reproduced by
`notebooks/build/nb24_registration_facts.py`. Once registered, everything in this file is fixed,
and anything the notebook does differently goes in a "Deviations" section at its end, with the
reason.

## Question

Every pairs book in this repo writes the pair relation in price **levels**,
P1 = α + β·P2 + ε, with β in shares of leg two per share of leg one. Avellaneda & Lee (2010),
Eq. (1), write it in **log prices**,

    ln(P1_t / P1_t0) = α·(t − t0) + β·ln(P2_t / P2_t0) + X_t,

with β a dollar ratio (long $1 of P1, short $β of P2), a deterministic drift α, and X_t the
mean-reverting residual that is traded. The two forms hedge differently whenever the price ratio
P1/P2 moves. The level form holds a fixed share ratio, the log form a fixed dollar ratio. The log
form is also scale-free, while the level form's Kalman variances are in dollars² (notebook 06:
Sharpe 0.51 against 0.70 on BKNG/MA from the same returns at two price bases).

On notebook 11's pool, signals and accounting, does writing the hedge in logs change the book's
performance, (a) for the frozen OLS hedge and (b) for the Kalman hedge?

The log books here drop Eq. (1)'s drift, as the paper does in §4.1. Eq. (1) as the paper
*estimates* it (a returns regression with a drift, residuals cumulated) is not in scope; see
"Not in scope".

## What is held fixed

Notebook 22's frame, unchanged:

- **Pool:** notebook 11's `selections("bh_dual", f)`, as in notebook 22 (`nsmallest(20,
  "eg_p_fdr")`, cached row order and orientation): 39 formations, 30 live, 287 pair-folds.
- **Windows:** two-year formation, six-month trading, each pair-fold starting flat.
- **Signal machine:** the robust z-score with a look-back of three half-lives (clipped to 20–250
  sessions), warmed up on the formation residual, entry at 2 ≤ |z| < 4, exit at |z| ≤ 0.5, stop at
  |z| ≥ 4. The machine is fixed. Its *input*, the residual, is what changes, and with it the
  half-life and the z window.
- **Accounting:** K = $200,000, `equal` split, `full` deployment, `daily` reset (notebook 22's
  baseline), decision-close fills, dividends on the ex-date, 50 bp a year borrow, measured cost
  per (ticker, formation) as primary and flat 5 bps as comparison.
- **Data:** the day-lake caches of notebooks 09–11 and notebook 14's `cost_per_ticker_window.parquet`
  (`cost_used`). Read, never rewritten. New caches are prefixed `loghedge_`.

**The books hold different trades.** Unlike notebook 22, a hedge change moves the residual and
therefore every entry and exit. As in notebook 23, the 99% interval of a pooled difference spans
roughly ±0.75, so "no material difference" is out of reach and **"undetermined" is the expected
outcome under no effect.**

## What has already been seen

- **Notebook 11 §4:** on this pool, under notebook 11's accounting, the level Kalman hedge at five
  EM iterations (H3₅ below) scored Sharpe 0.315 against the static level hedge's 0.402, with 1,698
  round trips against 723. H3₅'s result is therefore a restatement, not a test.
- **Notebook 22:** all of its facts about this book (118 of 287 hedges negative; 36 tracker
  pair-folds; BAC/CNX 2008 and the 2021-12-31 BDX pairs carry about 70% of the evaluation P&L; the
  effective number of formations is about four; the baseline's evaluation-span Sharpe is 0.514 at
  measured cost).
- **Notebook 23's registration:** the formation-window level residual starts in a transient that
  the Engle–Granger screen selected.
- **θ-gate population study:** a dynamic hedge is warranted for only 2–3% of BH survivors.
- **Notebook 16:** a log-price Kalman (Palomar ch. 15) on a different pool. Its spread accounting
  applies to H4.
- **The formation-window facts below.** No trading-window field was read to establish them.
- **No log-form hedge has been run on this pool.**

**The pool was selected on levels.** Notebook 10's screen runs Engle–Granger and Johansen on
split-adjusted price **levels**. Every pair in the pool passed because its *level* residual looked
stationary over the formation window, so the comparison is tilted toward H0 before any data is
seen. The notebook must say so beside every result. Fact 2 measures the tilt: only 91 of the 287
pair-folds would pass the same screen run on log prices. **[OPEN]** Whether to add a selection arm
(re-screen the 300-name universe on log prices, BH as before, top 20 by q-value) or to name it as
the follow-up. It is the larger compute item (about 44,850 pairs × 39 formations), and like
notebook 23's S family it can only be compared where the pools differ. This draft names it as the
follow-up, and it reports C1 on the pair-folds that pass both screens (see "Reported").

## The books

Every hedge is fitted on the formation window only and frozen (static) or continued with frozen
parameters (Kalman) into the trading window, exactly as notebook 11 does.

| Book | Form | Hedge | Residual traded |
|---|---|---|---|
| **H0** `level_ols` (baseline) | levels | OLS of P1 on P2, frozen | P1 − a − b·P2 |
| **H1** `log_ols` | logs | OLS of ln P1 on ln P2, frozen | ln P1 − a − b·ln P2 |
| **H3** `level_kalman` | levels | `_kalman_dynamic_hedge`, starting q = 1e-5, r = 1, P₀ = 1e6, EM under the convergence rule below, filter | the filter's `resid` |
| **H4** `log_kalman` | logs | the same on ln P1, ln P2, every starting variance scaled by r₀ = formation variance of H1's residual (q₀ = 1e-5·r₀, r = r₀, P₀ = 1e6·r₀), EM under the same rule, filter | the filter's `resid` |
| **H3₅** `level_kalman_5` (notebook 11 §4, restated) | levels | as H3 but `em_iters = 5` | the filter's `resid` |

(The label H2 is retired with the book it named; see "Not in scope".)

**From hedge to shares.** `simulate_pair_book` takes a share ratio per decision close. For the
log books it receives β_shares,t = β_t · P1_t / P2_t, so that with the daily reset the position
holds dollars in the ratio 1 : β_t, which is Eq. (1)'s portfolio. Gross notional is unchanged:
n1 = T / (P1_t + |β_shares,t|·P2_t) = T / (P1_t·(1 + |β_t|)).

**H4's starting values are scaled, not copied.** The first draft set q₀ = 1e-5·r₀ and r = r₀ but
kept P₀ = 1e6, a dollars² value. On log prices, EM then returned an indefinite transition
covariance in about 110 of 287 pair-folds, and the count moved from run to run (fact 5). Scaling
every variance by r₀ makes H4 the unit translation of H3's settings, and its results are
reproducible.

**The convergence rule** (`em_converge` in `nb24_registration_facts.py`, the same for H3 and H4).
Step EM one iteration at a time. After each step, require Q positive definite and R > 0, and
compute the log-likelihood from the predicted innovations. Stop at the first of:
- an indefinite Q or R ≤ 0: "indefinite", keep the previous iterate;
- a fall of more than 0.001 nats: "decrease", keep the previous iterate;
- a change below 0.001 nats: "converged";
- 500 iterations: "cap".

Log-likelihood differences do not change when the observations are rescaled, so the tolerance is
scale-free. Each pair-fold's status is recorded and reported.

**[OPEN] Fact 6: under this rule EM does not converge for most log pair-folds.** H4 converges in
107 of 287. It stops on a genuinely indefinite Q in 108, after a median of 10 iterations, and hits
the cap in 72. H3 converges in 179 and hits the cap in 104. C2 would therefore compare two books
whose pair-folds are a mixture of converged and unconverged fits, in different proportions. The
options:
- (a) Keep the rule, and report C2 with each book's status counts beside it.
- (b) No EM: fix one scale-free signal-to-noise ratio for both books. But at convergence the level
  filter's β-state ratio is ten times the log filter's (fact 6), so any single value favors one form.
- (c) Take C2 out of the family. C1 is then the only test (Holm's step 0.05), and H3, H4 and H3₅
  are reported as descriptions.

Recommendation: (c). The frozen comparison is clean, and the Kalman comparison is not like-for-like
with the EM machinery available. The log Kalman can return as its own study once
`_kalman_dynamic_hedge` checks Q (separate work in progress), possibly with the observation
centered (x − x̄) to separate β from the intercept.

## Comparisons

| Test | Challenger vs comparator | What it isolates |
|---|---|---|
| **C1** | H1 vs H0 | logs against levels, frozen hedge |
| **C2** | H4 vs H3 | logs against levels, Kalman hedge (subject to the [OPEN] decision above) |

Reported outside the family: H3₅ vs H0 (a restatement of notebook 11 §4 under notebook 22's
accounting), H3 vs H3₅ (what convergence changes), and H4 vs H1 (time-varying against frozen, in
logs).

## Statistics

Notebook 23's machinery, unchanged where it applies:

- **Primary test, paired by formation:** δ_f = SR_f(challenger) − SR_f(comparator) over the 17
  evaluation formations that are not tracker-only. An exact two-sided Wilcoxon signed-rank test,
  with the exact sign test beside it. Formations with an undefined SR_f are dropped and counted.
- **Holm** at a family-wise 5%, two-sided, across the tests in the family. With C1 and C2 the
  strictest step is 0.025.
- **Pooled D** on the evaluation span, with 95% and 99% cluster-bootstrap intervals over the 24
  live evaluation formations (`np.random.default_rng(24)`, 10,000 draws, one index matrix shared
  by all books), the leave-one-formation-out range and the hold-out value.
- **Power:** the s.d. of δ_f is printed for every comparison, with the effect size the paired test
  could detect at 80% power at each Holm step.

## Periods

| Period | Sessions | Formations | Use |
|---|---|---|---|
| Evaluation | 2006-07-03 to 2022-12-30 | 24 live (17 not tracker-only), 225 pair-folds | the decision |
| Hold-out | 2023-01-03 to 2025-08-13 | 6 live, 62 pair-folds | sign check on an exposed window |

Nothing is fitted beyond this file, so there is no development period. Halves (2016-01-01) and the
span without 2008-07-01, 2021-12-31 and 2022 are reported for every book.

## Reported for every book, whatever the outcome

Notebook 22's list (net and gross Sharpe on live and all sessions, dollar P&L, cost, turnover,
drawdown, concentration), plus, per pair-fold:

- **The hedges:** the dollar ratio implied at the formation close (H0: b·P2/P1; H1: b) and the
  number of pair-folds where the sign differs between H0 and H1.
- **The residual's statistics:** half-life and z window for each residual; the Engle–Granger
  statistic of H1's formation residual next to H0's.
- **The trades:** completed round trips, the median hold, and the share of trips ended by the stop.
- **How the hedges move:** for H3, H3₅ and H4, the median absolute daily change in the dollar
  hedge ratio and the EM status and iteration count. For H0, the drift in the implied dollar ratio
  over the trading window, which H1 holds fixed by construction.
- **C1 and C2 on two subsets, as descriptions and not tests:** the pair-folds whose level hedge is
  positive and those whose level hedge is negative (fact 1 puts nearly all of the hedge gap in the
  second), and the pair-folds that pass the screen on both levels and logs (fact 2). These are
  pooled D and its 99% interval only. Each subset leaves too few formations for the paired test.
  C2 is also reported on the pair-folds where both H3 and H4 converged.

## Decision rules, fixed now

Notebook 23's ordered outcomes, unchanged: (1) no material difference (99% interval inside
±0.10), (2) adopted (Holm-positive, pooled D positive on the evaluation span and in every
leave-one-out sample and in the hold-out), (3) worse, (4) undetermined. No definition, grid, seed,
starting value or convergence setting changes after the first run.

## Not in scope, and why

**Eq. (1) as the paper estimates it** (the former H2: OLS of daily log returns on daily log
returns, residuals cumulated, drift extrapolated into the trading window). Fact 3 shows it is a
different strategy on this pool, not a different hedge for the same one. The paper fits that
regression on a stock against its own sector factor, over a 60-day window refitted daily, and in
§4.1 it drops the drift. Frozen for six months on a pair chosen by a level screen, the estimator
answers a different question. Notebook 15 runs the paper's estimator in its own setting.

## Facts established while drafting (2026-10-13; formation windows only, no outcomes)

Reproduced by `python notebooks/build/nb24_registration_facts.py` (parts A–C, about an hour on four
workers, peak 0.5 GB in the parent process; part C is about four CPU-hours). It reads only the `form`
field of notebook 22's `alloc_folds.pkl` (the 287 formation windows, 501–506 rows each), with the
pair, formation and flags, and notebook 10's `day_screen.parquet`. No trading-window field is read.
Figures are for the 251 non-tracker pair-folds unless marked.

**The harness reproduces the cached book.** Recomputed level OLS α and β match the cache exactly
(max difference 0), so do all 287 z windows, and all 287 recomputed Engle–Granger p-values equal
the screen's (max relative difference 0). The level residual's opening transient comes out at
3.93 / 1.47 / 0.57 (row 0 / row 20 / rows 100+, all 287), notebook 23's 3.9 / 1.5 / 0.5–0.6.

1. **The hedges differ, and the difference is in the negative hedges.** |b_log − H0's dollar ratio
   at the formation close| / |H0's| has a median of 0.43 (p25 0.10, p75 0.79). 75% differ by more
   than 10%. Only 5 differ in sign. By the sign of the level hedge:

   | | n | median gap | share > 10% | median corr(H0, H1 residual) |
   |---|---|---|---|---|
   | positive | 133 | 0.14 | 0.58 | 0.96 |
   | negative | 118 | 0.64 | 0.95 | 0.91 |

   H0's implied dollar ratio b·P2/P1 ranges over 1.66 times its median within a formation window
   (median; recomputed from raw closes). The level hedge's intercept makes the dollar ratio move
   even with b fixed. The residuals stay close (median correlation 0.95, p10 0.79), but the z
   windows differ in 195 of 251 (H1/H0 median 1.10, p75 1.51). **C1 is not a non-test.**
2. **The selection tilt is large.** On log prices, the formation's own BH cutoff and the Johansen
   gate pass **62 of 251** non-tracker pair-folds (91 of 287 with the trackers): 49 of 133
   positive hedges and 13 of 118 negative ones. 79 have a log Engle–Granger p above 0.05. Median
   p is 7.6e-7 in levels and 4.5e-3 in logs. This is what selection on the level statistic does:
   the pool is extreme on that statistic and regresses on a correlated one. A log-screened pool
   would be equally extreme in logs.
3. **Eq. (1) as the paper estimates it is a different strategy here.** The returns β̂ over b_log
   has a median of 0.34 (p25 −0.05). Opposite sign in 74 of 251. |α̂|·126 / s.d.(X) has a median
   of 1.27, and exceeds 1 in 61%. corr(H1 residual, X) has a median of 0.50. X's z window has a
   median of 106 against H1's 40, and hits the 250 clip at p90. This is why it was dropped.
4. **The log residual has the transient too, smaller:** 3.26 / 1.48 / 0.65 against the level
   residual's 4.43 / 1.63 / 0.57 (non-trackers).
5. **Five EM iterations are not scale-free.** EM stepped one iteration at a time, log-likelihood
   from the predicted innovations (all 287):

   | | Q indefinite after iteration 5 | at any iteration ≤ 5 | LL gain from iteration 1 to 20 still to come after iteration 5 (median) | end dollar hedge moves > 25% from iteration 5 to 20 |
   |---|---|---|---|---|
   | H3, levels, notebook 11's settings | 0 | 0 | 0.23 | 5% |
   | H4 as first drafted (P₀ = 1e6) | **110** | 177 | 0.60 (158 defined) | 47% |
   | H4, every variance scaled by r₀ | 0 (11 by iteration 20) | 0 | **0.86** | 45% |

   The first-draft start is numerically unstable. Between two runs of the same computation (the
   script's and an earlier scratch run, compared separately) its iteration-5 count moved from 114
   to 110, and the sign of Q's smallest eigenvalue flipped in 14
   pair-folds, by up to 1.1. The negative eigenvalues are not rounding: the median is −6e-5·r₀. The
   scaled start gave identical signs in both runs. pykalman's EM can return an indefinite Q, and
   `_kalman_dynamic_hedge` then filters with it without raising.
6. **EM to convergence is not reachable for most log pair-folds** (the rule above):

   | | converged | indefinite | cap (500) | decrease | median iterations |
   |---|---|---|---|---|---|
   | H3, all 287 | 179 | 3 | 104 | 1 | 171 |
   | H4, all 287 | 107 | **108** | 72 | 0 | 258 |
   | H3, non-trackers | 147 | 3 | 100 | 1 | 237 |
   | H4, non-trackers | 106 | 74 | 71 | 0 | 300 |

   H4's indefinite stops come after a median of 10 iterations (6–21). On a sample of ten,
   re-stepped separately (`np.random.default_rng(0)`; not part of the script), the rejected step's
   Q has a smallest-to-largest eigenvalue ratio of −0.06 to −0.85, so these are genuine
   indefinite matrices, not rounding at zero. A hypothesis, not tested: in log units ln P2 varies
   little around its mean, so the β and intercept states are nearly collinear.

   Where they stop, the two filters differ by an order of magnitude (non-trackers, medians):

   | | β-state signal-to-noise q_β·var(x)/R | daily absolute change in the dollar hedge | gap from H1's static b at the formation close |
   |---|---|---|---|
   | H3 | 0.11 | 0.012 | 0.81 |
   | H4 | 0.011 | 0.0033 | 0.63 |

## Forecasts, recorded before the run

**[OPEN]** Provisional, revised after the facts:

1. **C1:** undetermined, and the pooled D is negative. The pool was selected on the level statistic
   (fact 2), which tilts toward H0. The draft's first forecast, |D| < 0.10 because the hedges
   coincide, is withdrawn: fact 1 shows they do not.
2. **C1 by hedge sign:** |D| on the negative-hedge pair-folds exceeds |D| on the positive ones,
   since that is where the hedges differ.
3. **C2 (if kept):** undetermined. H3 completes more round trips than H4, since its hedge moves
   about 3.5 times as fast per day (fact 6).
4. **H3 vs H3₅:** H3 completes more round trips than H3₅ (its β-state ratio rises with
   convergence).
5. **H4 vs H1:** H4 behind, as H3₅ was behind H0 in notebook 11.

## Code

- `pairs/models/kalman.py`: the log books pass ln P1, ln P2 to the same functions. A check on Q
  after EM is being added in separate work in progress, and notebook 24 builds on it. If C2 stays in
  the family, `_kalman_dynamic_hedge` also gains the convergence rule (`em_converge` as written in
  `nb24_registration_facts.py`), tested on a pair-fold that triggers each stopping status.
- One helper, `log_hedge_to_shares(beta_log, p1, p2)`, beside `simulate_pair_book` in
  `pairs/strategies/book.py`, tested on hand-built paths so that the dollar ratio is held at
  1 : β under the daily reset and the gross notional equals T.
- The notebook is built by `notebooks/build/build_log_hedge_notebook.py`. It reuses notebook 22's
  simulator and statistics. Development runs in smoke mode (`LOGHEDGE_SMOKE=1`) on notebook 23's
  smoke pool, the distance-method pairs with every registered (formation, unordered pair) removed,
  and the builder asserts the intersection with the registered 287 is empty. Smoke caches are
  prefixed `loghedge_smoke_`.
- `notebooks/build/nb24_registration_facts.py` stays as the record of the facts above.

## Numbering

`notebooks/pairs_trading_24_log_hedge_day_lake.ipynb`, a day-lake study numbered after 23.
Notebook 23 is registered and not yet run, and the two share a pool, so **[OPEN]** whether
notebook 24's first run waits for notebook 23's.
