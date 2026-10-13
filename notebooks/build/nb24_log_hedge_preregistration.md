# Pre-registration: notebook 24, the hedge in logs (Avellaneda–Lee's Eq. 1 against the level hedge)

**DRAFT, 2026-10-13. Not registered.** Sections marked **[OPEN]** are decisions to make before
this file is committed as the registration. "Facts established while drafting" records the
formation-window measurements taken on 2026-10-13, and what they changed. Once registered,
everything in this file is fixed, and anything the notebook does differently goes in a "Deviations"
section at its end, with the reason.

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

Primary: on notebook 11's pool, signals and accounting, does writing the hedge in logs change the
book's performance, (a) for the frozen OLS hedge and (b) for the Kalman hedge?

Secondary: does Eq. (1) as the paper estimates it (a returns regression with a drift, residuals
cumulated) differ from the log OLS hedge?

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

- **Notebook 11 §4:** on this pool, under notebook 11's accounting, the level Kalman hedge (H3
  below) scored Sharpe 0.315 against the static level hedge's 0.402, with 1,698 round trips against
  723. H3's result is therefore a restatement, not a test.
- **Notebook 22:** all of its facts about this book (118 of 287 hedges negative; 36 tracker
  pair-folds; BAC/CNX 2008 and the 2021-12-31 BDX pairs carry about 70% of the evaluation P&L; the
  effective number of formations is about four; the baseline's evaluation-span Sharpe is 0.514 at
  measured cost).
- **Notebook 23's registration:** the formation-window level residual starts in a transient that
  the Engle–Granger screen selected.
- **θ-gate population study:** a dynamic hedge is warranted for only 2–3% of BH survivors.
- **Notebook 16:** a log-price Kalman (Palomar ch. 15) on a different pool. Its spread accounting
  applies to H4.
- **No log-form hedge has been run on this pool.**

**The pool was selected on levels.** Notebook 10's screen runs Engle–Granger and Johansen on
split-adjusted price **levels**. Every pair in the pool passed because its *level* residual looked
stationary over the formation window, so the comparison is tilted toward H0 before any data is
seen. The notebook must say so beside every result. Fact 2 below measures the tilt: only 91 of
the 287 pair-folds would pass the same screen run on log prices. **[OPEN]** Whether to add a
selection arm (re-screen the 300-name universe on log prices, BH as before, top 20 by q-value) or
to name it as the follow-up. It is the larger compute item (about 44,850 pairs × 39 formations),
and like notebook 23's S family it can only be compared where the pools differ. This draft names
it as the follow-up, and it reports C1 on the pair-folds that pass both screens (see "Reported").

## The books

Every hedge is fitted on the formation window only and frozen (static) or continued with frozen
parameters (Kalman) into the trading window, exactly as notebook 11 does.

| Book | Form | Hedge | Residual traded |
|---|---|---|---|
| **H0** `level_ols` (baseline) | levels | OLS of P1 on P2, frozen | P1 − a − b·P2 |
| **H1** `log_ols` | logs | OLS of ln P1 on ln P2, frozen | ln P1 − a − b·ln P2 |
| **H2** `log_drift` (Eq. 1 as estimated in §3 / App. A) | logs | OLS of daily log returns r1 on r2 → α̂, β̂ | ln(P1_t/P1_t0) − α̂·(t − t0) − β̂·ln(P2_t/P2_t0), t in sessions from the formation's first row, drift extrapolated into the trading window |
| **H3** `level_kalman` (notebook 11 §4, restated) | levels | `_kalman_dynamic_hedge`, q = 1e-5, r = 1, em_iters = 5, filter | the filter's `resid` |
| **H4** `log_kalman` | logs | the same function on ln P1, ln P2, filter, every starting variance scaled by r₀ = formation variance of H1's residual (q₀ = 1e-5·r₀, r = r₀, P₀ = 1e6·r₀); **[OPEN]** how EM is run, see fact 5 | the filter's `resid` |

**From hedge to shares.** `simulate_pair_book` takes a share ratio per decision close. For the
log books it receives β_shares,t = β_t · P1_t / P2_t, so that with the daily reset the position
holds dollars in the ratio 1 : β_t, which is Eq. (1)'s portfolio. Gross notional is unchanged:
n1 = T / (P1_t + |β_shares,t|·P2_t) = T / (P1_t·(1 + |β_t|)).

**[OPEN] H2 as drafted is neither Eq. (1) as the paper uses it nor a fair test of it.** Fact 3
shows the returns β̂ is a different hedge from H1's b (median ratio 0.43, opposite sign in 74 of
287), and extrapolating the drift across a six-month window moves the residual's center by more
than one formation s.d. in 54% of pair-folds. The paper fits the returns regression on a stock
against its own sector factor over a 60-day window refitted daily, and in §4.1 it drops the drift.
Frozen for six months on a pair chosen by a level screen, the estimator answers a different
question. Recommendation: remove H2 and C3, which leaves a family of two (Holm's strictest step
0.025), and leave the paper's estimator to notebook 15, which runs it in its own setting. The
alternative is H2′: the returns β̂ with the drift dropped, reported outside the family.

**H4's starting values are scaled, not copied.** The first draft set q₀ = 1e-5·r₀ and r = r₀ but
kept P₀ = 1e6, a dollars² value. On log prices, EM then returned an indefinite transition
covariance in 114 of 287 pair-folds (fact 5). Scaling every variance by r₀ removes that failure,
which makes H4 the unit translation of H3's settings.

## Comparisons

| Test | Challenger vs comparator | What it isolates |
|---|---|---|
| **C1** | H1 vs H0 | logs against levels, frozen hedge |
| **C2** | H4 vs H3 | logs against levels, Kalman hedge |
| **C3** | H2 vs H1 | Eq. (1)'s drift and returns estimation against log OLS |

Reported outside the family: H3 vs H0 (a restatement of notebook 11 §4 under notebook 22's
accounting) and H4 vs H1 (time-varying against frozen, in logs).

## Statistics

Notebook 23's machinery, unchanged where it applies:

- **Primary test, paired by formation:** δ_f = SR_f(challenger) − SR_f(comparator) over the 17
  evaluation formations that are not tracker-only. An exact two-sided Wilcoxon signed-rank test,
  with the exact sign test beside it. Formations with an undefined SR_f are dropped and counted.
- **Holm** at a family-wise 5%, two-sided, across C1–C3. The strictest step is 0.0167.
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

- **The hedges:** the dollar ratio implied at the formation close (H0: b·P2/P1; H1, H2: b, β̂),
  the number of pair-folds where the sign differs between H0 and H1, and the drift share
  |α̂|·(sessions)/s.d.(X) for H2.
- **The residual's statistics:** half-life and z window for each residual; the Engle–Granger
  statistic of H1's formation residual next to H0's.
- **The trades:** completed round trips, the median hold, and the share of trips ended by the stop.
- **How the hedges move:** for H3 and H4, the median absolute daily change in the dollar hedge
  ratio. For H0, the drift in the implied dollar ratio over the trading window, which H1 holds
  fixed by construction.
- **C1 and C2 on two subsets, as descriptions and not tests:** the pair-folds whose level hedge is
  positive and those whose level hedge is negative (fact 1 puts nearly all of the hedge gap in the
  second), and the pair-folds that pass the screen on both levels and logs (fact 2). These are
  pooled D and its 99% interval only. Each subset leaves too few formations for the paired test.

## Decision rules, fixed now

Notebook 23's ordered outcomes, unchanged: (1) no material difference (99% interval inside
±0.10), (2) adopted (Holm-positive, pooled D positive on the evaluation span and in every
leave-one-out sample and in the hold-out), (3) worse, (4) undetermined. No definition, grid, seed
or starting value changes after the first run.

## Facts established while drafting (2026-10-13; formation windows only, no outcomes)

Computed from the `form` field of notebook 22's `alloc_folds.pkl` (the 287 formation windows,
501–506 rows each) and notebook 10's `day_screen.parquet`. No trading-window field was read.
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
3. **H2 is a different strategy.** The returns β̂ over b_log has a median of 0.34 (p25 −0.05).
   Opposite sign in 74 of 251. |α̂|·126 / s.d.(X) has a median of 1.27, and exceeds 1 in 61%.
   corr(H1 residual, H2's X) has a median of 0.50. H2's z window has a median of 106 against
   H1's 40, and hits the 250 clip at p90.
4. **The log residual has the transient too, smaller:** 3.26 / 1.48 / 0.65 against the level
   residual's 4.43 / 1.63 / 0.57 (non-trackers). H2's X starts at zero by construction, so its
   row-0 figure (1.53) measures something else.
5. **EM on log prices does not behave like EM on levels.** EM stepped one iteration at a time,
   log-likelihood from the predicted innovations (all 287):

   | | Q indefinite after iteration 5 | at any iteration ≤ 5 | LL gain from iteration 1 to 20 still to come after iteration 5 (median) | end dollar hedge moves > 25% from iteration 5 to 20 |
   |---|---|---|---|---|
   | H3, levels, notebook 11's settings | 0 | 0 | 0.23 | 5% |
   | H4 as first drafted (P₀ = 1e6) | **114** | 180 | 0.59 (158 defined) | 47% |
   | H4, every variance scaled by r₀ | 0 (11 by iteration 20) | 0 | **0.86** | 46% |

   pykalman's EM can return a transition covariance with a negative variance, and
   `_kalman_dynamic_hedge` then filters with it without raising: its log-likelihood is undefined,
   and the filter runs anyway. Scaling P₀ removes that on this pool, but five iterations are far
   from convergence in logs, while in levels they are nearly converged. **"em_iters = 5" is not
   scale-free**, so C2 as first drafted would compare a converged fit with an unconverged one.

**[OPEN] What fact 5 leaves for H3/H4.** (a) Run EM to convergence in both books with one rule (a
relative LL tolerance, an iteration cap, and Q's smallest eigenvalue checked after every step, with
a pair-fold whose Q turns indefinite recorded and run static). H3 at five iterations stays as the
restatement of notebook 11 §4, outside the family. (b) No EM: fix a scale-free signal-to-noise
ratio for both books. That removes the convergence question, but the ratio has to be chosen and
justified. Recommendation: (a). It keeps the model family notebook 11 used, and makes C2 about
levels against logs rather than about iteration counts.

## Forecasts, recorded before the run

**[OPEN]** Provisional, revised after the facts:

1. **C1:** undetermined, and the pooled D is negative. The pool was selected on the level statistic
   (fact 2), which tilts toward H0. The draft's first forecast, |D| < 0.10 because the hedges
   coincide, is withdrawn: fact 1 shows they do not.
2. **C1 by hedge sign:** |D| on the negative-hedge pair-folds exceeds |D| on the positive ones,
   since that is where the hedges differ.
3. **C2:** undetermined. H4's round trips are within ±25% of H3's.
4. **H4 vs H1:** H4 behind, as H3 was behind H0 in notebook 11.
5. **(If H2 is kept)** C3's pooled D is negative.

## Code

- `pairs/models/kalman.py`: the log books pass ln P1, ln P2 to the same functions. Under option
  (a) of fact 5, `_kalman_dynamic_hedge` also gains a convergence rule and a check on Q's smallest
  eigenvalue after EM, tested on a pair-fold that triggers it.
- One helper, `log_hedge_to_shares(beta_log, p1, p2)`, beside `simulate_pair_book` in
  `pairs/strategies/book.py`, tested on hand-built paths so that the dollar ratio is held at
  1 : β under the daily reset and the gross notional equals T.
- The notebook is built by `notebooks/build/build_log_hedge_notebook.py`. It reuses notebook 22's
  simulator and statistics. Development runs in smoke mode (`LOGHEDGE_SMOKE=1`) on notebook 23's
  smoke pool, the distance-method pairs with every registered (formation, unordered pair) removed,
  and the builder asserts the intersection with the registered 287 is empty. Smoke caches are
  prefixed `loghedge_smoke_`.

## Numbering

`notebooks/pairs_trading_24_log_hedge_day_lake.ipynb`, a day-lake study numbered after 23.
Notebook 23 is registered and not yet run, and the two share a pool, so **[OPEN]** whether
notebook 24's first run waits for notebook 23's.
