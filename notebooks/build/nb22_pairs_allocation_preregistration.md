# Pre-registration: notebook 22, allocating capital across a pool of pairs

Written 2026-09-30, before any of the code below exists or any number from a challenger has been
seen. Everything in this file is fixed. Anything the notebook does differently must be listed in a
"Deviations" section at its end, with the reason.

## Question

Given a pool of pairs that has already been chosen and signals that are already fixed, how should
capital be split among the pairs, how much of it should be deployed when the pool is thin, and how
should positions be rebalanced inside a holding window? Notebook 11 answered all three by default
(the same dollars for every pair, reset to constant dollars at every close) and never examined the
answer. This notebook asks whether any standard alternative beats that default by more than the
noise, and how large a difference it could have seen.

Secondary: does the helper the package ships for this, `pairs.suggest_position_weights`, do what
its name says?

## What is held fixed

Notebook 11's book, unchanged:

- The pool is exactly notebook 11's `selections("bh_dual", f)`, that is
  `DataFrame.nsmallest(20, "eg_p_fdr")` on `day_rule_bh_dual.parquet` with ties resolved by the
  cached row order and the (ticker1, ticker2) orientation as stored. It is not re-derived by
  sorting. 39 six-monthly formations from 2006-06-30, 30 of them live, 287 pair-folds.
- Windows are notebook 11's `WINDOWS` over all 39 formations, a two-year formation window, an OLS
  hedge of P1 on P2 frozen for the trading window, the robust z-score with a look-back of three
  half-lives (clipped to 20–250 sessions) warmed up on the formation window, entry at
  2 ≤ |z| < 4, exit at |z| ≤ 0.5, stop at |z| ≥ 4, every pair-fold starting flat, dividends
  accrued on the ex-date, 50 bp a year borrow on short notional.

Position size never feeds back into a signal, so **every book in this study holds the same pairs
on the same sessions in the same direction.** Only the share counts differ.

Data: the day-lake caches notebooks 09–11 wrote (`day_market_bars.parquet`,
`day_rule_bh_dual.parquet`, `day_scaled_by_formation.pkl`) and notebook 14's measured costs
(`cost_per_ticker_window.parquet`, column `cost_used`; not `cost_by_pair_fold.parquet`, which is
an unweighted leg mean). All are read and none rewritten. The lakes are read-only. New caches are
prefixed `alloc_`.

## What has already been seen

The author has seen notebook 11's and notebook 14's results for this book under equal dollars, by
pair-fold: that 2022 carries about half of twenty years of dollar P&L; that leveraged and
volatility products are 39% of the pair-folds and 39% of the P&L; that 87% of gross P&L comes from
5% of round trips; notebook 11 §3.5's list of the largest contributing pair-folds; notebook 14
§8's per-fold Sharpe table (mean −0.123, median +0.210, pooled 0.402), which previews the
direction of any inverse-volatility tilt; notebook 14's panel of z against forward spread moves;
and the weaker 2023–2025 window, which notebooks 11, 14, 17 and 21 all display. The hold-out is
therefore a consistency check on an exposed window, not an untouched test, and the notebook must
say so.

`per_pair` (B1 below) is notebook 11's published dollar series. Its whole P&L path has been seen,
so it is reported as a restatement and is outside the test family. No other challenger has been
run on this book.

**Design history.** Two drafts of this file were reviewed, read-only, before registration. The
reviewers computed on notebook 11's own allocation, on positions and on ticker counts only, and
what they found changed the design:

- The baseline was misdescribed. Notebook 11 resets every held position to constant dollars at
  each close (shares change on 5,688 of 5,691 held rows); it does not size once at entry.
- The pooled Sharpe difference was replaced as the test statistic. Under the baseline one
  three-pair formation, 2008-07-01, carries 45% of the evaluation-span variance and the effective
  number of formations is about four.
- The pool contains 36 **same-underlying** pair-folds (IVV/SPY, SPY/VOO and the like), 31 of
  which lose money. Their spreads are about a hundred times less volatile than an ordinary
  pair's. Seven of the 24 evaluation formations hold nothing else, and a Sharpe ratio there is
  cost drag over a vanishing standard deviation. Those formations are excluded from the primary
  tests, and an ex-tracker pool was added.
- A split that recognizes shared legs replaced minimum variance. In five of the nine evaluation
  formations with 19 or 20 pairs, one ticker is a leg of 13 to 19 of them.
- `reuse` was given a per-pair cap in place of a multiplier cap of 3, which bound on 76% of held
  decision closes in multi-pair formations; `reweight` was given a 63-session window, because a
  monthly update of a two-year window shares 96% to 79% of its values with the formation window.

Other facts from the reviews, all under notebook 11's allocation:

- The 30 live formations hold 1 to 20 pairs (median 6); six evaluation formations hold one pair.
- The hedge is negative in 118 of the 287 pair-folds. Both legs are then held on the same side,
  and the "spread" is a basket.
- A pair-fold is in a position on 18% of its sessions on average.
- On the evaluation span under K the baseline's Sharpe is 0.567 at zero transaction cost, 0.514
  at measured cost and 0.440 at flat 5 bps, so all measured cost is worth 0.053. Its
  leave-one-formation-out range at measured cost is 0.37 to 0.58.
- The P&L of 2008-07-01 is one pair-fold (BAC/CNX, the head of notebook 11 §3.5's list), beside
  an IVV/SPY pair; that of 2021-12-31 is its 18 BDX pairs, beside SPY/VOO and IVV/VOO.
- The `z_size` multiplier has mean 1.30.

Two things follow. The weights `inv_vol`, `erc` and `leg_split` put on those two formations'
pair-folds can be worked out from tickers and tracker volatility, and B2's pooled statistic is
arithmetic on per-formation baseline sums already computed; **the results of A1, A2, A3 and B2
weigh less as tests than those of A5 and C1–C3.** And forecasts 1–4 below are informed.

## Accounting

A fund with constant capital **K = $200,000** (20 × $10,000). P&L is not reinvested, as in
notebook 11, so a return is P&L divided by K.

- In formation f with n_f selected pairs, pair i has **split weight** w_i ≥ 0 with Σ w_i = 1 and
  **allocation** a_i = A_f · w_i, where A_f is the fold's **deployment**.
- Every held trade has a **target notional** T_i = a_i · m · K, where m is a multiplier that is 1
  unless a rule below changes it.
- **Timing, identical for entries, exits and every resize, and identical to
  `generate_pair_signals(exec_lag=1)` plus `evaluate_pair_signals`:** shares decided at the close
  of session t are n1 = pos · T_i / (P1_t + |β| P2_t) and n2 = −β · n1. They are booked on ledger
  row t+1, earn the move from close t to close t+1, and pay cost on |Δshares| × close_{t+1} for
  each leg. The fill is therefore the decision close, notebook 11's optimistic convention; it is
  held fixed, not corrected.
- **Ledger conventions held fixed** (all as in notebook 11): row 0 of a pair-fold is flat; a
  decision on a fold's last row is never executed; a row on which either leg has no close is
  dropped, not zero-filled; borrow on row t is 50/1e4/252 × short shares × close_t, per ledger
  row; dividends are shares × dividend on the ex-date row; a position open on a fold's last row is
  marked to market and never closed, with no closing cost; a pair with no row on a session has no
  P&L, cost or borrow, its allocation idles and is not redistributed. Shares are fractional. No
  financing is charged on deployment above K.
- **Live sessions** are the sessions inside the trading window of a formation that selected at
  least one pair. A Sharpe is mean over standard deviation (ddof = 0) × √252 of daily P&L / K on
  live sessions. The same ratio with the sessions of empty formations counted as zeros is
  reported beside it for every book.

**Notebook 11 reports two numbers that imply two different deployment rules.** Its dollar P&L
($43.6k) is $10,000 for every pair, so the fund is larger when the screen finds more. Its Sharpe
(0.402) divides each session's P&L by the number of pairs selected, which is the return of a fund
that puts **all** of K into however many pairs the screen found, one pair included. The baseline
of this study is the second, because that is the Sharpe every later notebook quotes.

**Baseline:** `equal` split (w_i = 1/n_f), `full` deployment (A_f = 1), `daily` reset (at every
decision close on which a trade is held, its shares are reset to T_i at that close's prices).

**Validation gates, passed before any challenger is computed:**

- G1. `equal` / `per_pair` / `daily` at flat 5 bps equals notebook 11 §3's `bh_dual` dollar
  series on all 3,677 live sessions to 1e-6 dollars (total $43,587.11), with 723 completed round
  trips and 52 positions open at a fold's end. Dividing by notebook 11's denominator gives 0.4019;
  the baseline under K gives 0.4021 (the two differ on 18 sessions where one pair stops pricing).
- G2. With measured costs charged leg by leg, restricted to the 251 pair-folds that complete at
  least one round trip and with notebook 14's denominator, the simulator gives notebook 14 §9.4's
  0.407 at flat 5 bps and 0.490 at measured cost, each within 0.001. On all 287 pair-folds with
  notebook 11's denominator it gives 0.483 within 0.001.
- If a gate fails, no challenger is computed until the simulator is fixed. A change to the
  simulator is not a deviation; a change to a gate's target is.

## Pools

- **Full pool**: the 287 pair-folds. Every outcome label comes from it.
- **Same-underlying pair**: both legs in one of {SPY, IVV, VOO, SPLG}, {IJH, MDY},
  {GOOG, GOOGL}. There are 36 (18 in the evaluation span, 18 in the hold-out).
- **Tracker-only formation**: every pair is same-underlying. Evaluation: 2011-12-30, 2012-06-29,
  2012-12-31, 2017-06-30, 2019-12-31, 2020-07-01, 2022-07-01. Hold-out: 2024-07-01, 2025-07-01.
- **Ex-tracker pool**: same-underlying pairs removed after the top-20 cut with no backfill; n_f,
  every risk input and M_f rebuilt. 251 pair-folds: 207 in 17 evaluation formations and 44 in
  four hold-out formations.
- **Gated pool**: notebook 11 §8's filter applied after the top-20 cut with no backfill, n_f and
  every risk input rebuilt. 206 pair-folds (173 evaluation, 33 hold-out; the 2024-12-31
  formation is empty and 2017-12-29 becomes a single pair).

## Risk inputs (formation window only)

For pair i with its frozen hedge β_i, the **spread return per dollar** is

  r_i,t = (ΔP1_t − β_i ΔP2_t) / (P1_{t−1} + |β_i| P2_{t−1}),

computed on the pair's own formation frame (both closes present, first differences, first row
dropped). On the sessions where every pair of the formation has a value (at least 500 in every
live formation), σ_i is the standard deviation of r_i (ddof = 0), and
Σ_f = D Ĉ D with D = diag(σ_i) and Ĉ the covariance from
`sklearn.covariance.LedoitWolf(assume_centered=False)` fitted on the columns divided by σ_i. The
notebook asserts that every σ_i is finite and positive and stops otherwise.

Risk is measured on the spread, not on the position: a pair is long the spread, short it or flat
as its z-score dictates, and the sign of a spread correlation depends on which ticker is P1.

**Validity rule for the risk model, fixed now.**

- Pair level: inside each evaluation formation with at least five pairs that are not
  same-underlying (12 formations), the Spearman correlation between σ_i and the realized standard
  deviation of r_i over the trading-window rows. The statistic is the median of the 12.
- Formation level: the Spearman correlation between s_f (B2 below) and the realized standard
  deviation of the baseline's daily return, over the 17 evaluation formations that are not
  tracker-only.
- If a statistic is below 0.5, the outcome of the challengers that rest on it (A1 and A2 at pair
  level, B2 at formation level) keeps its label and carries the qualifier "the risk model did not
  forecast", whether the outcome is an adoption or not.

## The challengers

Each changes one decision and leaves the other two at the baseline.

### A. The split across pairs (set at formation, fixed for the fold)

| # | Name | Rule | Source |
|---|---|---|---|
| — | `equal` | w_i = 1/n_f | baseline; DeMiguel, Garlappi & Uppal 2009 |
| A1 | `inv_vol` | w_i ∝ 1/σ_i | naive risk parity |
| A2 | `erc` | equal risk contributions under Σ_f, long-only | Maillard, Roncalli & Teiletche 2010 |
| A3 | `leg_split` | w_i ∝ 1 / max(c(t1_i), c(t2_i)), where c(t) is the number of the formation's pairs that contain ticker t | shared-leg exposure |
| A4 | `shipped` | the package helper, as shipped | this repository |
| A5 | `z_size` | `equal`, with m = clip(\|z\| / 2, 1, 2) fixed for the life of the trade | conviction sizing |

- `erc` minimizes ½x′Σ_f x − (1/n_f) Σ log x_i over x > 0 (Spinu 2013) by `scipy` L-BFGS-B on
  log x, started at 1/σ_i, gtol 1e-12, then w = x / Σx. A failure to converge falls back to
  `inv_vol` for that formation and is logged.
- `shipped` is `suggest_position_weights({(t1, t2): frame}, corr_matrix=pair_return_correlations(…),
  method="inv_var", max_weight=0.40)` on the formation-window residual P1 − α − βP2, using its
  `weight` column re-keyed by pair label. Two properties are known now and are not corrected: it
  weights by the variance of the residual's first difference in **price units**, which is not a
  risk per dollar, and it clips then renormalizes once, so its cap is not a cap.
- `z_size` uses the z-score on the decision row, `sig["z"].shift(1)` read on the entry row.
  Entries need |z| < 4, so m lies in [1, 2). It is a sizing rule, not a split: its deployment
  exceeds one and only relative sizes matter for a Sharpe.
- One diagnostic outside the test family: `inv_var_dollar`, w_i ∝ 1/σ_i². The gap between it and
  `shipped` is what the price units cost.

### B. How much to deploy (on `equal`, `daily`)

| # | Name | Rule |
|---|---|---|
| — | `full` | A_f = 1 |
| B1 | `per_pair` | A_f = n_f / 20 (notebook 11's dollars; a restatement, not tested) |
| B2 | `vol_target` | A_f = clip(M_f / s_f, 1/3, 3), where s_f = √(w′ Σ_f w) is the forecast volatility of the fold's book with every pair open and M_f is the median of s over all earlier live formations that are not tracker-only, in date order. The first live formation has A_f = 1. |

M_f for 2008-07-01 is a single value, s of 2006-06-30. The notebook says so beside B2's result
and reports, as a sensitivity, B2 with A_f = 1 until three earlier formations enter M_f.

### C. Rebalancing inside the fold (`full` deployment)

| # | Name | On | Rule |
|---|---|---|---|
| — | `daily` | any | shares reset to T_i at every decision close |
| C1 | `frozen` | `equal` | shares set by the entry decision and kept until the exit |
| C2 | `reweight` | `inv_vol` | σ_i is the standard deviation (ddof = 0) of the pair's own last 63 values of r_i, taken at the formation close and again at the close of sessions 21, 42, 63, 84, 105 and 126 of the trading window; the split is 1/σ_i over all n_f pairs |
| C3 | `reuse` | `equal` | at every decision close, after the entries and exits decided at that close, U_t = Σ w_j over the pairs that will be held and m_t = max(1, min(1 / U_t, n_f / 3)) |

- `reweight`: sessions are counted 1-based on the window's session calendar (session 1 is ledger
  row 0), and r_i runs over the pair's concatenated formation and trading rows with the frozen
  hedge. The new split is used by the decision made at that same close, for held trades and for
  later entries; a pair with no row on that session keeps its last σ_i; the `daily` reset is
  unchanged. It changes both the length and the age of the estimate and is compared with
  `inv_vol` as a package.
- `reuse` is full reinvestment among the held pairs, with no pair above a third of K. An entering
  trade gets T_i = w_i m_t K. A continuing trade (held at the previous decision close and at this
  one; an exit followed by an entry on the next close is a new trade) has its T_i moved to
  w_i m_t K only when the two differ by more than 25% of T_i. When nothing is held, nothing is
  sized. The `daily` reset is unchanged. For n_f ≤ 3 it is the baseline.
- One diagnostic outside the test family: `band`, in which shares are reset at a decision close
  only when the gross notional of the held shares at that close's prices differs from T_i by more
  than 10% of T_i. The median hold is six rows, so it will mostly coincide with `frozen`; the
  notebook reports the share of round trips in which it triggers.

**Comparators.** A1–A5, B2, C1 and C3 are compared with the baseline. C2 is compared with
`inv_vol` / `full` / `daily`.

**The grid.** The 6 × 3 table of split × deployment under `daily` is reported as pooled Sharpe
(for B2 each cell uses its own split in s_f). Nothing is selected from it.

## Not in scope, and why

- Re-estimating the hedge: settled in notebooks 11 §4, 16 and 19 (the frozen hedge wins).
- The formation cadence: changing it means re-running notebook 10's screen.
- Rules that need an expected return per pair (mean-variance, Kelly): 287 pair-folds cannot
  estimate one. Minimum variance was dropped for `leg_split`: with a floor of zero it selects
  pairs, which changes the pool, and notebook 17 already made the fitted-against-naive comparison.
- Market impact, capacity and financing of leverage.
- Entry, exit and stop thresholds, and any change to the pool.
- Combinations. Adoption is of one change at a time; a book that combines two needs its own
  pre-registration.

## Costs

Primary: notebook 14 §9's measured cost per (ticker, formation), charged on each leg's traded
notional (all 360 cells are measured). Comparison: flat 5 bps a leg-side. Each pair's orders are
charged separately, as in notebook 11. A **netted** line on the measured schedule, in which
Δshares in the same ticker on the same session are summed across pairs before cost (borrow and
dividends are not netted), is reported for every book and nothing is selected on it.

## Periods

| Period | Sessions | Formations | Use |
|---|---|---|---|
| Evaluation | 2006-07-03 to 2022-12-30 | 24 live (17 not tracker-only), 225 pair-folds, 3,022 live sessions | the decision |
| Hold-out | 2023-01-03 to 2025-08-13 | 6 live (2022-12-30 onward), 62 pair-folds, 655 live sessions | sign check on an exposed window |

There is no development period because nothing is fitted or chosen: every definition is in this
file. Every book is also reported by half (sessions before and from 2016-01-01), with the
2008-07-01 formation removed, with the 2021-12-31 formation removed, and with the 2022 sessions
removed.

## Statistics

All statistics are net of measured costs on the evaluation span unless stated.

- **Primary test for A1–A5 and C1–C3: paired by formation.** For each of the 17 evaluation
  formations that are not tracker-only, SR_f is mean / s.d. (ddof = 0) × √252 of the book's daily
  return over that formation's live sessions, and δ_f = SR_f(challenger) − SR_f(comparator). A
  formation with |δ_f| < 1e-9 is dropped. The test is the exact two-sided Wilcoxon signed-rank
  test of δ_f (`scipy.stats.wilcoxon`, `method="exact"`); its direction is positive when the sum
  of positive ranks exceeds the sum of negative ranks. The exact sign test on the same δ_f is
  reported beside it, and the δ_f of the tracker-only formations are printed separately. Each
  formation tested is one draw of a pool and counts once.
- **Formations that can inform each test.** A1, A2 and A4: at most 17. `leg_split`: 13 (it equals
  `equal` where the pairs share no ticker: 2008-07-01, 2015-07-01, 2020-12-31, 2021-07-01).
  `reuse`: at most 13 (it is the baseline for n_f ≤ 3). The notebook prints the number tested,
  and the median, mean and standard deviation of δ_f over them, for every challenger.
- **Primary test for B2: permutation.** Deployment is constant inside a formation, so every δ_f
  is zero. The null is that the multipliers are unrelated to formation outcomes: the A_f of the
  17 evaluation formations that are not tracker-only are permuted among those 17 (tracker-only
  formations keep their own) 10,000 times (`np.random.default_rng(23)`), pooled D is recomputed,
  and p = (1 + #{|D_π − mean(D_π)| ≥ |D − mean(D_π)| − 1e-12}) / 10,001. Beside p the notebook
  reports the same p with 2008-07-01 removed from the statistic and from the permutation, and the
  share of the 17 circular shifts of the multiplier sequence at least as extreme.
- **Pooled D, reported for every challenger and not the test.** D = Sharpe(challenger) −
  Sharpe(comparator) on the span's live sessions, with 95% and 99% percentile intervals from a
  cluster bootstrap over formations: one index matrix
  `np.random.default_rng(22).integers(0, 24, size=(10000, 24))` shared by all books. The
  leave-one-formation-out range of D is reported beside it. In every bootstrap and
  leave-one-out sample the books keep their full-sample A_f and M_f; only sessions are resampled
  or removed. The decision rules use the **99%** interval, because on the baseline the
  24-cluster percentile interval covers about 87% to 90% at a nominal 95% and about 95% at a
  nominal 99%. For `reuse`, pooled D includes a movement of capital between formations, which is
  described and not tested.
- **Multiple testing.** Holm at a family-wise 5%, two-sided, across the nine p-values of A1–A5,
  B2 and C1–C3.
- **Power, stated now.** At the strictest Holm step the paired test reaches 80% power for an
  effect of about one standard deviation of δ_f with 17 formations and about 1.2 with 13.
- **Random-split scale (A1–A4).** 1,000 books, each with one independent Dirichlet(1, …, 1)
  split per evaluation formation with n_f ≥ 2 (`np.random.default_rng(24)`, formations in date
  order, pairs in `selections` order). The median and 5th–95th percentiles of their median δ_f
  (over the same 17 formations) and pooled D are shown beside each challenger's. It is a scale
  for how far an arbitrary split moves the statistic, not a test.
- **Other pools.** A1, A2, A4, B2 and C2 are re-run on the ex-tracker pool, and every book on the
  gated pool. Neither is Holm-tested.
- Hold-out and sub-period Sharpe ratios carry the standard error √(252 / sessions) only.

## Reported for every book, whatever the outcome

- Net and gross Sharpe (gross is before transaction cost, after borrow and dividends), on live
  sessions and with idle sessions counted; dollar P&L; measured cost paid, separate and netted.
- Turnover: traded gross notional / (K × live sessions / 252).
- At **matched volatility** (one scalar per book, fitted on the evaluation span so that its
  daily volatility equals the baseline's, applied unchanged to the hold-out): maximum drawdown of
  the cumulative sum of returns, in units of K, and the worst formation (the smallest sum of
  returns over a formation's live sessions).
- Concentration: the share of round-trip gross P&L from the top floor(5%) of round trips (a round
  trip runs from an entry to the following exit or stop; resizes inside it belong to it), and the
  shares of P&L from 2022, from the 2008-07-01 formation and from the 2021-12-31 formation, each
  with its dollar amount.
- Effective number of pairs, 1 / Σ w_i²; the weight each split gives to same-underlying pairs,
  by formation; the largest single-ticker share of gross notional, averaged over sessions with a
  position and at its peak; utilization (open gross notional / K, averaged over live sessions and
  over sessions with a position); net notional / K; for B2 and C3 the share of formations or held
  decision closes at each bound.
- Per pair-fold: realized against forecast volatility, and the realized risk contribution against
  the one Σ_f implies for the book's own weights.

## Decision rules, fixed now

Outcomes are tested in this order, and the first that applies is the outcome. The interval is
the 99% percentile interval of pooled D on the evaluation span.

1. **No material difference**: the interval lies strictly inside ±0.10. The notebook adds
   "consistently better" or "consistently worse" when the primary test passes Holm in that
   direction or the interval excludes zero.
2. **Adopted**: the primary test passes Holm in the positive direction; pooled D is positive on
   the evaluation span and in every leave-one-formation-out sample; pooled D is positive on the
   gated pool over the evaluation span; and pooled D is positive in the hold-out. For B2,
   D − mean(D_π) must also be positive with and without 2008-07-01. For C2, the composite
   (`inv_vol` with `reweight`) must also have a positive pooled D against the baseline with an
   interval that excludes zero.
3. **Worse**: the primary test passes Holm in the negative direction; or the interval lies below
   zero and pooled D is negative in every leave-one-formation-out sample.
4. **Undetermined**: anything else.

- `per_pair`, `band` and `inv_var_dollar` receive no outcome; their pooled D and intervals are
  reported.
- The hold-out and gated signs are necessary conditions only. Under no effect a hold-out sign
  agrees about half the time, and the notebook says so beside any adoption.
- The risk-model qualifier adds to an outcome and never replaces it.
- For A1, A2, A4, B2 and C2, a sentence about the principle (inverse volatility, equal risk
  contribution, volatility targeting) is written only if the ex-tracker pool gives the same sign
  of median δ_f and of pooled D. Otherwise the notebook says the result is the tracker pairs.
- If no challenger is adopted, the notebook says that the default was not beaten at this sample
  size, and lists which challengers made no material difference, which were worse, and which the
  data could not place. It says "allocation does not matter for this book" only of the first
  group.
- The margin of ±0.10 is about twice what all measured transaction cost is worth (0.053) and a
  fifth of the baseline's level.

The package helper is not changed by this notebook. Its two known properties are defects whatever
the result; a fix is proposed afterward as a separate change. No definition, window, cap, band or
seed changes after the first run.

## Forecasts, recorded before the run

Forecasts 1–4 are informed by what the author has already seen; 5–7 are not. Forecast 1 counts
outcome 2 only; forecast 5 counts outcome 1 with or without a sign.

1. No challenger is adopted. (Under no effect the design alone makes this hold with probability
   above 0.97, so its success is not evidence of foresight.)
2. Among the nine tested challengers, each against its own comparator, the largest pooled |D|
   belongs to `vol_target` or `reuse`, the two that move capital between formations.
3. `shipped` has a pooled net Sharpe below both `equal` and `inv_vol`.
4. `reuse` is leverage, not edge: its primary test does not reach an unadjusted 0.10 in the
   positive direction.
5. The daily reset is immaterial: `frozen` comes out "no material difference".
6. `reweight` pays more measured cost than `inv_vol`, and its primary test does not reach an
   unadjusted 0.10 in the positive direction.
7. Realized volatility exceeds the formation-window σ_i in more than 60% of the 225 evaluation
   pair-folds.

## Code

- Allocators as tested functions in `pairs/stats/portfolio.py` that take σ and Σ as inputs, so
  the package gains no dependency (the Ledoit–Wolf fit is in the notebook, and scikit-learn is
  already in `environment.yml`); a leg-level book simulator in `pairs/strategies/`. Unit tests:
  weights sum to one; `erc` equalizes risk contributions on a known matrix; `leg_split` on a
  star and on disjoint pairs; the simulator reproduces `evaluate_pair_signals` on one pair under
  `daily` at m = 1.
- The notebook is built by `notebooks/build/build_pairs_allocation_notebook.py`.
- The web app is not touched.

## Numbering

`notebooks/pairs_trading_22_pairs_allocation_day_lake.ipynb`. It is a day-lake study, and by the
precedent of notebook 17 it would be inserted at 18 with the minute-lake notebooks moved to 19–22.
It is numbered 22 instead, because a third renumbering would rewrite every cross-reference for no
gain and the filename already names the data source.
