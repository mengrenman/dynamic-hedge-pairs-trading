# Pre-registration: notebook 23, the Ornstein–Uhlenbeck model, checked and then used

Written 2026-10-13, before any of the code below exists or any number from a challenger has been
seen. Everything in this file is fixed. Anything the notebook does differently must be listed in a
"Deviations" section at its end, with the reason.

## Question

The repository's pairs book assumes an Ornstein–Uhlenbeck (OU) spread without saying so: the
z-score bands, the half-life look-back and the s-score all rest on dX = κ(μ − X)dt + σ dW. This
notebook asks three things. **Does the OU model fitted on a formation window describe the trading
window that follows** (the model check)? **Do the thresholds the model says are optimal beat the
fixed bands** the book has used since notebook 01? And **does ranking survivors by speed of
reversion beat ranking them by p-value**, which is what the screen does?

The model check comes first because the selection rule ranks by the fitted speed and the
threshold rules use the fitted center and scale. Two objectives give "optimal" thresholds and
they disagree by an order of magnitude. In Bertram's (2010) return per unit of time, κ cancels,
so a threshold depends on the pair's cost per unit of scale alone and comes out narrower than one
daily move at measured costs. In Leung and Li's (2016) discounted optimal stopping, the discount
rate is a time scale, so speed matters again (faster reversion brings the entry and exit levels
closer together) and the levels come out near two stationary standard deviations from the mean,
the same order as the fixed bands. Both are tested.

## What is held fixed

Notebook 11's book as notebook 22 froze it: the `bh_dual` pool (`selections("bh_dual", f)`, 287
pair-folds over 30 live formations), the two-year formation window, the OLS hedge of P1 on P2
frozen for the trading window, every pair-fold starting flat, dividends accrued, 50 bp a year
borrow, and notebook 22's accounting: K = $200,000, equal split, full deployment, daily reset to
constant dollars, measured costs charged leg by leg. The baseline is notebook 22's baseline book:
the robust rolling z-score (look-back three half-lives clipped to 20–250 sessions, warmed up on
the formation window), entry at 2 ≤ |z| < 4, exit at |z| ≤ 0.5, stop at |z| ≥ 4, one-sided. A
decision made at the close of session t is booked on ledger row t+1 and earns the move from
close t to close t+1, so the fill is the decision close; there is no lag in the fill price.

Every book in the threshold family holds the same pairs on the same windows; only the signal
differs. The selection family changes the pool and nothing else.

Data: the day-lake caches notebooks 09–11 and 22 wrote (`day_market_bars.parquet`,
`day_rule_bh_dual.parquet`, `day_screen.parquet`, `day_scaled_by_formation.pkl`,
`alloc_folds.pkl`), notebook 14's measured costs (`cost_per_ticker_window.parquet`), and the
minute lake for cost cells notebook 14 never measured. Prices for pairs outside `alloc_folds.pkl`
and the raw closes the cost step needs are read with a pyarrow filter on the needed tickers
only; the wide frame of `day_market_bars.parquet` is never built. The one exception is the
instrument-sibling test, which reads the `close` column for every ticker with a close on every
session of one formation window, one formation at a time. All caches are read and none
rewritten. The lakes are read-only. New caches are prefixed `ou_` (smoke runs `ou_smoke_`).

## What has already been seen

Everything notebook 22 printed: the baseline's per-formation Sharpe ratios; that two positions
(BAC/CNX in 2008-07-01 and the BDX block of 2021-12-31) carry 70% of the evaluation P&L; the 36
same-underlying pair-folds; that realized volatility was below the formation-window estimate in
80% of pair-folds because volatility had already fallen inside the window (its post hoc §14.2
also put an iid benchmark for that share at 42%); 775 entries and 723 completed round trips,
with a median hold of six rows over all trades and seven over the completed ones; 118 negative
hedges; the baseline's stop share of 14.5%. Also notebook 14's 33.9 bps gross per round trip;
notebook 15's finding that the s-score's κ filter passed 99% of names; notebook 11's z look-back
of 20 sessions in 84 of 287 pair-folds (half-lives under seven sessions); the intraday half-life
of about 16.6 days (notebooks 18–20); and notebooks 04–05's result that tuning the bands is
noise. The hold-out window has been displayed in notebooks 11, 14, 17, 21 and 22.

**Design history.** Two drafts of this file were reviewed, read-only, by three and then two
agents, who computed formation-window facts on the registered pool, trading-window facts of the
baseline's own book, and data-free simulations, and nothing from any trading-window fit or
challenger. Between the drafts the author read Leung and Li (2016), *Optimal Mean Reversion
Trading*, whose Chapter 2 solves the discounted entry and exit problem under the OU model with
transaction costs and a stop-loss; its results are challenger T3 and gate G5, and nothing in the
book is data from this pool. What the reviews changed:

- Bertram's Sharpe-ratio objective at a zero rate is independent of cost and monotone in the
  threshold, so that challenger was dropped.
- The AR(1) intercept is an endpoint artifact and the AR-implied stationary s.d. is about 0.79 of
  the sample s.d., so the center and scale are the sample mean and sample s.d. of the fit rows.
- **The formation residual starts in a transient the screen selected.** Mean |X| over the
  whole-window s.d. is 3.9 at row 0, 1.5 at row 20 and 0.5–0.6 from row 100 on; the first 50 rows
  carry a median 52% of the formation sum of squares; the whole-window s.d. is about 1.45 times
  the late-regime s.d.; the half-life on the post-transient rows is 1.24 times the whole-window
  one (2.3 at the 75th percentile); and the trading window starts in the late regime. On the
  post-transient rows the AR(1) is self-consistent (AR-implied over sample s.d. 1.09 against a
  simulated 1.07); on the whole window it is not (0.79, below the null's first percentile in 205
  of 287 pair-folds). A whole-window fit would therefore have made "slower", "smaller" and a thin
  entry the expected result before any trading row was read. **The fit is made on the formation
  rows after the first 100**, and the transient is measured first (M0).
- The baseline's entries, measured on its own trading-window decision rows against the
  whole-window center and scale, sit at a median of 0.91 s.d. from the center, 39% inside the
  frozen exit band, 19% on the far side and 18% at or beyond 2 s.d., so the holding-time and
  profit checks are made on the frozen-z book's own trips and against a simulated null, and the
  baseline is checked by posterior prediction.
- Under the exact model, the share of a frozen-z book's trips held shorter than the mean
  first-passage time is already 0.61–0.76 (the mean of a skewed time, trips censored by the
  window's end), and a band exit's realized profit is at least the level distance by
  construction, so every model-check forecast is stated against the simulated null, not against
  an absolute share.
- A Monte-Carlo gate at step 1/50 session is biased by 3–25% at narrow thresholds, so the gate
  uses a Brownian-bridge crossing correction; the short-window sample s.d. of an AR(1) is biased
  low by far more than 5%, so the parameter-recovery gate tests it against its exact finite-sample
  expectation.
- Bertram's thresholds on the rolling z must be derived in the rolling scale's own units; the
  rolling robust scale is about 0.39 of the whole-window s.d. at the median (0.22 for trackers).
- Where the selection pools can differ most, 2021-12-31, the p-value pool is 20 BDX pairs by
  construction (132 of the 143 survivors contain BDX); the BH-adjusted p-value ties 97 of them, and
  the raw Engle–Granger p-value is unique within every formation.
- The distance-method pool shares 35 pair-folds with the registered pool, so smoke mode removes
  them; the HLT close of 2017-01-03 (a spin-off and reverse split the lake did not adjust) is row
  0 of six pair-folds at 7–13 s.d., so a bad-row rule is registered.

Facts from the reviews that the design now rests on, all formation-window or baseline-only:
cost per unit of whole-window scale, c / s, has median 0.0052 for the 251 ordinary pair-folds
(0.0084 in post-transient units; 10th to 90th percentile 0.0009–0.016) and 0.19–0.84 for the 36
trackers (overall range 0.0002–0.97); the whole-window φ̃ lies in [0.399, 0.994] with
Kendall-corrected half-lives of median 11.9 and 90th percentile 35.3 sessions, and 14 pair-folds
have a post-transient φ̃ outside (0, 1); Bertram's threshold at the ordinary deciles is 0.14–0.31
post-transient s.d. against one-session innovations of 0.16–0.48; the baseline's median daily
innovation is 0.33 whole-window s.d.; the simulated M1 null is near 0.52 and 1.05.

Forecasts 1, 2, 3, 5, 6 and 7 below are informed by the above; 4, 8 and 9 are not.

**Design facts counted while writing this file** (tickers, dates and the rule files only): the
BH survivors per formation, after removing same-underlying pairs and pairs that touch an
instrument flagged by notebook 11 §7's behavioral gate in that formation, number from 0 to 689;
the universe is non-empty in the 17 evaluation formations that are not tracker-only and in three
hold-out formations; seven of those 17 have more than 20 survivors (2006-06-30, 2010-07-01,
2010-12-31, 2013-07-01, 2017-12-29, 2018-06-29, 2021-12-31), and one hold-out formation does
(2023-12-29). The selection family can therefore differ from its comparator in at most seven
evaluation formations and one hold-out formation. The gate is behavioral and misses some
instruments (TVIX is in 24 of the 39 survivors of 2018-06-29); the notebook prints the gated and
ETF-list legs of every pool.

## The OU fit

On a residual series X = P1 − α_f − β_f P2 (price units, the frozen hedge) over the n rows of a
window, the fit is the OLS regression of X_t on X_{t−1} over the n − 1 pairs:

- φ̃ = φ̂ + (1 + 3φ̂)/n, the Kendall (1954) small-sample correction, is used everywhere below; φ̂
  is printed beside it;
- κ = −ln φ̃ per session and half-life = ln 2 / κ; φ̃ ≥ 1 gives an infinite half-life and
  φ̃ ≤ 0 a half-life of zero;
- **μ_f is the sample mean of X over the fit rows and s_f its sample standard deviation**
  (ddof = 0); the AR(1) intercept ratio c/(1 − φ̃) and the AR-implied σ̂_ε / √(1 − φ̃²), with
  σ̂_ε² = Σe²/(n − 3), are printed beside them and not used;
- **σ_d = s_f √(1 − φ̃²) is the model's one-session innovation s.d.**; σ̂_ε is printed beside it
  and not used;
- the OU used for first-passage times and for the simulated null has stationary s.d. s_f, speed
  κ and one-step transition X_t = μ_f + φ̃ (X_{t−1} − μ_f) + σ_d ε_t.

**Fit rows.** The **formation fit** uses the cached formation frame's rows after the first 100
(about 400 rows; that count is n in the Kendall term), so that the transient the screen selected
is left out; the whole-window fit is printed beside it. A formation row on which one leg moves
more than 40% in a session while the other moves less than 5% is dropped from the fit and counted
(six HLT pair-folds of 2018-12-31). The **trading-window fit** uses the cached trading frame
(108–127 rows in the evaluation span); a trading window with fewer than 60 rows is not fitted and
is counted (the seven 30-row pair-folds of 2025-07-01). A formation fit with φ̃ outside (0, 1)
removes the pair-fold from the model check and the threshold family and is counted; on the
registered pool that count is 14.

## The model check (measured, with forecasts; no decision)

**M0, the transient, measured before anything else.** Per pair-fold and by formation: the share
of the formation sum of squares of X in its first 50 rows, mean |X| / s_whole by row block (0–9,
10–49, 50–99, 100 on), the number of legs with a one-session log-price move above 20% inside the
formation window, and the whole-window against the post-transient half-life, s.d. and center.

**The simulated null.** One `np.random.default_rng(2301)`. For each pair-fold in
`alloc_folds.pkl["folds"]` order, ε = rng.standard_normal((200, n_i)) with n_i the cached
trading-frame length; path j is X_{j,t} = μ_f + φ̃ (X_{j,t−1} − μ_f) + σ_d ε_{j,t}, t = 1..n_i,
from X_{j,0} = the last formation residual, which is not a row of the path. The same fits and
books are run on the paths, so every measurement below has a distribution under "the formation
model is exactly right". For each draw index j the pooled statistic is computed over the
pair-folds of the period using draw j of every pair-fold; the null is the 200 values, reported by
mean and `np.percentile` (linear) at 2.5, 5, 50, 95 and 97.5, and the observed statistic is placed
against them. The null draws pair-folds independently while the real ones share dates and legs,
so its percentiles understate the sampling spread of the observed value; the notebook says so.
The cluster bootstrap of the shared index matrix gives the observed statistic its own interval and
is not applied to the null. M2′ and M3′ use 500 paths per pair-fold (the other measurements 200)
and print the Monte-Carlo s.e. of their percentiles.

- **M1, speed.** The ratio of the trading-window half-life to the formation half-life, per
  pair-fold; infinite counts as longer, zero as shorter. Reported: the share above one and the
  median ratio over fitted windows, the share infinite, the same by formation, and the null's
  distribution of the share and the median (near 0.52 and 1.05).
- **M4, scale.** The trading-window sample s.d. of X (ddof = 0) over s_f, per pair-fold, with the
  null's distribution of the share below one and of the median.
- **M5, center.** |mean of X over the trading rows − μ_f| / s_f, per pair-fold, with its null.
- **M2, holding time, on T1's completed evaluation round trips**, whose entries and exits are
  the rules the prediction describes. For a trip decided at the close t with residual X* (the
  decision row), p = +1 for a long and −1 for a short, and L = μ_f − p · 0.5 s_f the exit level
  (d = p (L − X*) > 0 by construction): the prediction is max(E[T_{X* → L′}], 1) sessions under
  the formation fit, where L′ is L moved a further 0.5826 σ_d beyond itself in the direction of
  travel (the Broadie–Glasserman–Kou correction for daily monitoring), against the rows held. A
  trip counts as shorter only if rows held < prediction strictly; ties are counted as not shorter
  and reported. Reported: the share shorter and the median of realized / predicted, against the
  same statistics on the null paths (under the exact model the share is about 0.61–0.76 because
  the prediction is a mean and completed trips are censored by the window's end), and both with
  trips ended by the stop excluded.
- **M3, profit per trip, on T1's trips, by posterior prediction.** A band exit is decided only
  once X is at or past the level, so a trip's realized profit is at least the level distance by
  construction; M3 therefore compares T1's realized gross P&L per completed trip on each
  pair-fold, in spread units, with the distribution of the same quantity over the null paths.
- **M2′ and M3′, posterior prediction for the baseline.** The baseline's own rule (rolling
  robust z, 2 / 0.5 / 4, the formation residual as warm-up) is run on the null paths of each
  pair-fold. Computed on X in spread units, identically for the realized and the simulated path:
  completed trips are `exit.sum()` of the executed position, rows held per trip are the rows with
  the position on, and a trip's gross P&L is Σ over its held rows of pos × (X_t − X_{t−1}) / s_f;
  each pair-fold's median hold and median P&L per trip are over its completed trips. A path with
  no completed trip contributes 0 to the trip count and NaN to the other two; percentiles are
  `np.nanpercentile` (linear) over the non-NaN draws; a pair-fold enters a share's denominator
  only if its realized value is defined and at least half of its draws are; "below" and "above"
  are strict. Reported: the share of pair-folds below the simulated 10th percentile and above
  the 90th, for the trip count, the median hold and the gross P&L per trip. P1, P2 and β are
  supplied from the realized pair-fold only to pass the machine's finite-price test.

Each is also reported for the hold-out pair-folds, separately.

## Validation gates, passed before any challenger is computed

- G1. The signal machine (below), run with the baseline's rules on the baseline's rolling
  z-score, reproduces notebook 11's executed positions on all 287 pair-folds exactly, and
  therefore notebook 22's baseline: Sharpe 0.514 on the evaluation span at measured cost under
  K, and $43,587.11 with 723 round trips at $10,000 a pair and flat 5 bps.
- G2. **Formulas.** In the dimensionless OU (κ = 1, s = 1) with y < b, g(w) =
  √(π/2) erfcx(−w/√2) (= e^{w²/2} ∫_{−∞}^{w} e^{−v²/2} dv), M1(y → b) = ∫_y^b g(w) dw, and
  M2(y → b) = 2 ∫_y^b e^{u²/2} ∫_{−∞}^{u} e^{−v²/2} M1(v → b) dv du. With Y = (X − μ_f) / s_f
  and κ per session, E[T] = M1 / κ and Var[T] = (M2 − M1²) / κ²; a downward passage y → b with
  y > b is the passage −y → −b. For the symmetric passage M1(−x → x) = π erfi(x/√2), the closed
  form of the mean only; the variance has no elementary closed form and `ou_first_passage`
  computes it from M2 in every case, using the erfcx single-integral form of M1. **The gate**
  integrates the explicit nested form with `scipy.integrate.quad(epsrel=1e-10, epsabs=0)` at
  every level, sharing no code with `ou_first_passage`, and requires agreement to 1e-6 relative
  at the symmetric passages x ∈ {0.25, 1, 2} and the one-sided passage −1.5 → −0.5, at half-lives
  5, 20 and 80 sessions. **Monte Carlo** (`np.random.default_rng(23)`, 200,000 paths per cell):
  κ = ln 2 / half-life, D = 1/50 session, φ_D = e^{−κD}, x₁ = φ_D x₀ + √(1 − φ_D²) ε; with the
  barrier b above both x₀ and x₁ the step counts as a crossing with probability
  exp(−(b − x₀)(b − x₁) / (κD)), one uniform per surviving path per step; the crossing time is
  step × D; no path is censored and the cell fails if any path passes 60 E[T]. The Monte Carlo
  must agree with the quadrature mean within 2% and the quadrature variance within 3% in every
  cell. E[T_{a → −a}] is the one-way passage, half of the cycle length printed in Bertram (2010).
- G3. On simulated OU series (`np.random.default_rng(2323)`, s = 1, μ = 0, X_0 ~ N(0, s²) and
  the exact transition thereafter) at half-lives 5 and 20 sessions, 504 and 126 rows, 1,000 draws
  each: (i) |mean(φ̃ − φ)| ≤ max(0.25 |mean(φ̂ − φ)|, 2e) with e = sd(φ̃ − φ)/√1000; (ii)
  |median(half-life estimate / truth) − 1| ≤ 0.15; (iii) `ou_fit` returns s_f equal to
  `np.std(X, ddof=0)` to 1e-12, and |mean(s_f² / s²) − E_n| ≤ max(0.03, 2 sd(s_f² / s²)/√1000)
  with E_n = 1 − (1/n)[1 + 2 Σ_{k=1}^{n−1} (1 − k/n) φ^k], the exact finite-sample value for a
  stationary AR(1) (0.972, 0.891, 0.891, 0.645 for half-lives 5 and 20 at 504 and 126 rows). The
  median of the AR-implied scale over s, the share of draws with φ̃ ≥ 1, and the half-life-80
  cells are printed and not gated.
- G5. `leung_li_levels` reproduces the worked example of Leung and Li (2016), Figure 2.10, to four
  decimals: with θ = 0.5388, μ = 16.6677, σ = 0.1599, r = r̂ = 0.05, c_s = c_b = 0.05 (all per
  year) and stop-loss L = 0.4834, the entry level d_L = 0.4978 and the exit level b_L = 0.5570.
  It also reproduces, to 1e-6, the no-stop levels of their Theorems 2.6 and 2.10 when L is sent
  to −∞, and the Brownian limit of their Remark 2.12, b* = c_s + σ / √(2r), when κ → 0.
- If a gate fails, no challenger is computed until the code is fixed. A change to the code is not
  a deviation; a change to a gate's target is.

**D1, the discretization (measured, not a gate).** `np.random.default_rng(2331)`; 1,000 paths of
1,000 sessions of the exact OU with s = 1 and X_0 ~ N(0, 1), φ = the median post-transient φ̃ of
the 287 formation fits (printed), c / s ∈ {0.002, 0.005, 0.02} and e* from the T2 search at that
cost. The T2 machine starts flat with the stop at 4; a decision at the close of t is filled at
that close. Profit per flip is p (X_flip − X_previous flip) − c over flip-to-flip intervals.
Reported: profit per session against (2e* − c) / E[T], flips per session against 1 / E[T], and
profit per flip against 2e* − c. Daily monitoring lets each flip overshoot the threshold in the
book's favor and lowers the flip count; the shortfall of profit per session is the
discretization effect quoted beside T2 and T2r.

## The challengers

### The signal machine

One state machine serves every book, `generate_threshold_signals` in
`pairs/strategies/signals.py`. With e the entry level, the stop at 4 and an entry block at
`block` (equal to the stop unless a rule sets it, as T3 does): long when −block < z ≤ −e, short
when e ≤ z < block; a long exits when |z| ≤ exit (`exit="band"`, notebook 11's rule) or when
z ≥ +level (`exit="level"`), a short symmetrically; the exit test precedes the stop test; the stop
is one-sided at |z| ≥ stop on the adverse side; after an exit or stop at t the earliest next entry
decision is t+1, except under the level rule, where the exit and the opposite entry are decided on
the same row and the position flips; an exit whose row has |z| ≥ block on the far side goes flat
and does not flip. A flip is one ledger row with exit and entry both set, counts as one completed
round trip and one entry, and pays |Δshares| on each leg. Decisions at the close of t are booked
on row t+1 and filled at close t; a last-row decision is never executed; a NaN or non-finite z,
price or hedge, or a non-positive price, sets the position flat with no exit or stop flag, as in
`generate_pair_signals`. The z-score and, where a rule varies them, per-row levels are supplied
by the caller; the machine returns the executed entry, exit and stop flags. G1 pins it to
notebook 11.

### T. Thresholds (same pool, same windows; only the signal differs)

| # | Name | z-score | Rules | Compared with |
|---|---|---|---|---|
| — | `baseline` | rolling robust, as notebook 11 | entry 2, band exit 0.5, stop 4 | — |
| T1 | `ou_z` | z = (X − μ_f) / s_f, frozen from the formation fit, from row 0, no warm-up | entry 2, band exit 0.5, stop 4 | baseline |
| T2 | `bertram` | the same frozen z | entry at e*, level exit at the mirror, stop 4 | T1 |
| T2r | `bertram_rolling` | the baseline's rolling robust z | entry at e*_t derived in the rolling scale's units, level exit at the mirror, stop 4 | baseline |
| T3 | `leung_li` | the same frozen z | long when z ∈ [a_L, d_L], exit long at z ≥ b_L, stop at z ≤ −4; the short side mirrored | T1 |

- **T2.** e* maximizes Bertram's (2010) expected return per unit of time for the symmetric flip
  rule, (2e − c) / E[T_{−e → e}], with E[T] in the formation fit's units and c the pair's
  round-trip cost per unit of spread, 2 × (c₁ P1 + c₂ |β| P2) / 10⁴ at the formation-close
  prices with the measured per-ticker cells, so that both e and c are in units of s_f. The search
  is `np.linspace(0.01, 3.99, 2000)` in units of s_f (an entry at |z| ≥ 4 is blocked), refined
  with `scipy.optimize.minimize_scalar(method="bounded")` between the neighbors of the best grid
  point (xatol 1e-8). If the maximum is ≤ 0 the pair-fold stays flat and is counted; on this pool
  the maximizer is interior for every pair-fold (e* ≈ (1.5 c/s)^{1/3} at small cost).
- Because c / s_f is 0.0002–0.97 on this pool, e* will be below one daily innovation for most
  ordinary pairs, so T2 is Bertram's rule taken literally at measured costs: a test of whether
  the spread reverts at the one-session horizon net of cost, evaluated on daily closes and filled
  at the decision close. D1 says how much of its result is the discretization. The notebook
  reports e* / (σ_d / s_f) for every pair-fold.
- **T2r** derives the threshold in the rolling scale's own units at each decision row:
  e*_t = min(u*(c / ŝ_t), 3.99), where ŝ_t = 1.4826 × the rolling MAD at the decision close (the
  scale the baseline's z divides by) and u*(·) is the registered search tabulated once on a
  400-point logarithmic grid of c / s ∈ [1e-4, 1] and interpolated linearly in log(c / s); entry
  at |z_t| ≥ e*_t, level exit at the mirror −sign × e*_t, stop 4 on the rolling z. The
  distribution of e*_t ŝ_t / s_f is reported.
- The cost cell in the objective is the one notebook 14 measured over the trading window, so
  the thresholds use ex-post liquidity (a limitation in cost only); the flat 5 bps books reuse
  the thresholds built on the measured cells.
- Bertram's Sharpe-ratio objective is not used: at a zero rate it reduces to √(E[T]/Var[T]),
  independent of cost and decreasing in e, so its maximizer is the grid edge.
- **T3 is Leung and Li's (2016) optimal double stopping with a stop-loss** (their Theorems 2.13
  and 2.17), in the formation fit's units: the OU has speed κ, mean μ_f and stationary s.d. s_f;
  the discount rate is **r = 0.05 / 252 per session** for entry and exit alike (2% and 10% a
  year are reported as a sensitivity, not tested); the one-way costs are c_b = c_s =
  (c₁ P1 + c₂ |β| P2) / 10⁴ at the formation-close prices; the stop is L = μ_f − 4 s_f. The
  functions F and G are the book's integrals (2.11)–(2.12), evaluated by `scipy.integrate.quad`
  to a relative tolerance of 1e-10, with derivatives by differentiating under the integral. The
  exit level b_L is the root of the book's equation (2.45) on (L*, μ_f + 10 s_f), with
  L* = (κ μ_f + r c_s) / (κ + r); the entry interval [a_L, d_L] is the pair of roots of (2.54)
  and (2.55), d_L on (L, b_L) and a_L on (L, d_L), all by `scipy.optimize.brentq` (xtol 1e-12).
  If L ≥ L* (the book's Corollary 2.14) or if no entry interval exists, the pair-fold stays flat
  and is counted. The short side is the mirror image about μ_f with the same levels. The entry
  interval is the structure notebook 11's rule already has (an entry blocked near the stop), with
  the levels now set by the model; the machine takes a_L as the entry block instead of the stop.
  A flip occurs only if b_L ≥ |d_L − μ_f| makes the opposite entry hold on the exit row.
- **Composites.** T2 and T3 are compared with T1 so that the comparison is thresholds against
  thresholds on one z-score; each is also compared with the baseline directly (pooled D with its
  99% percentile interval from the shared index matrix), which is the composite the adoption
  rule names. T1 and T2r are compared with the baseline and need none. A formation in which T1
  has no trade is dropped from T2's and T3's paired tests and counted.

### S. Selection (same signal as the baseline; only the pool differs)

The universe in formation f is the BH survivors in `day_rule_bh_dual.parquet` for f, minus
same-underlying pairs (notebook 22's sets {SPY, IVV, VOO, SPLG}, {IJH, MDY}, {GOOG, GOOGL}),
minus pairs with a leg in `day_scaled_by_formation.pkl[f]`. Both pools are built from it with
the same rule, so the comparator is not notebook 11's pool.

| # | Name | Rule |
|---|---|---|
| P | `p_rank` | the 20 smallest `eg_p_fdr`, ties broken by the raw `eg_p` of `day_screen.parquet` (ascending), then by cached row order |
| S1 | `kappa_rank` | the 20 smallest post-transient formation φ̃ (shortest half-lives); φ̃ ≤ 0 ranks first and φ̃ ≥ 1 last; ties by `eg_p` |

- The hypothesis is point-estimate speed against unit-root significance. S1's ranking uses
  formation-window facts only.
- A selected pair whose formation frame has fewer than 250 rows or whose trading frame has
  fewer than 20 rows (`fold_frames`) is dropped after the cut with no backfill, as in notebook
  11, in both pools, and counted. Overlapping pairs are allowed in both pools, as in notebook 11;
  the largest ticker multiplicity is printed. The number of pairs tied with the 20th q-value is
  printed for every formation (2021-12-31: 97). The 2021-12-31 formation, where P is 20 BDX pairs
  by construction and any 20-pair pool holds at least nine, is printed separately with each pool's
  BDX multiplicity, and the paired S test is also reported without it. As a sensitivity with no
  outcome, both pools are rebuilt with at most three pairs per ticker, the next-ranked pairs
  filling the cut.
- The S books' live sessions are those of formations with at least one surviving pair: 17
  evaluation formations and three hold-out formations, of which one (2023-12-29) can differ
  between the pools, so S1's hold-out sign is one formation and is reported, not a condition.
- Cost cells notebook 14 never measured are measured by its estimator,
  `measure_ticker_window_costs(WINDOWS, needed, MINUTE_ROOT, raw_close=…, spec=CostSpec())`,
  over the formation's trading window at the default 5-minute interval, with `raw_close` read
  for the needed tickers only, once per formation window, cached as `ou_cost_cells.parquet`, and
  checked against `cost_per_ticker_window.parquet` on the cells both hold. A cell with a
  non-empty `err` takes the median `cost_used` of the measured cells of the same formation
  across both files, excluding itself; a formation with none takes the median over all measured
  cells. The counts of each `err` type and of fallbacks are reported, and P and S1 share a cell.

## Not in scope, and why

- Re-estimating the hedge, the formation cadence, the allocation (notebook 22), market impact.
- Time-varying or regime-switching κ: the particle-filter study, if written, is separate.
- OU-based forecasts as an alpha signal: notebooks 14 and 15 showed that −(1 − φ) x has a
  positive IC on anything stationary.
- A discrete-time optimization of the thresholds (a Markov-chain approximation of the AR(1)
  with daily monitoring inside the objective): the next study if D1 shows the discretization is
  where T2's result comes from.
- Leung and Li's optimal switching (their §2.5.4, after Song and Zhang 2013), the repeated-trade
  version of T3 whose levels lie inside the single-trade ones, and their likelihood-maximizing
  hedge ratio (§2.1): both are candidates for a later study, not this one.
- The s-score with trading time (notebook 15), any change to the stop, and the hedge-ratio
  question for S1's pairs (S1 uses the OLS hedge like everything else).

## Costs

Primary: measured cost per (ticker, formation), charged leg by leg; comparison: flat 5 bps a
leg-side. Each pair's orders are charged separately.

## Periods

| Family | Period | Sessions | Formations | Use |
|---|---|---|---|---|
| T | Evaluation | 2006-07-03 to 2022-12-30 | 24 live (17 not tracker-only), 225 pair-folds | the decision |
| T | Hold-out | 2023-01-03 to 2025-08-13 | 6 live, 62 pair-folds | sign check on an exposed window |
| S | Evaluation | the same span | 17 live, at most 196 P pair-folds | the comparison |
| S | Hold-out | the same span | 3 live, at most 25 P pair-folds, one formation can differ | reported |

There is no development period because nothing is fitted or chosen beyond this file. Halves
(2016-01-01) and the span without 2008-07-01, without 2021-12-31 and without the 2022 sessions
are reported for every book, as in notebook 22.

## Statistics

Notebook 22's machinery, unchanged where it applies:

- **Primary test, paired by formation.** δ_f = SR_f(challenger) − SR_f(comparator) over the 17
  evaluation formations that are not tracker-only; |δ_f| < 1e-9 dropped; a formation in which
  the challenger's or the comparator's SR_f is undefined (no trade, zero s.d.) is dropped and
  counted; exact two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`, `method="exact"`),
  direction from the rank sums, the exact sign test beside it, tracker-only δ_f printed. An
  undefined primary p (no formation tested) is treated as 1 in Holm and contributes neither an
  adoption nor a "worse".
- **Holm** at a family-wise 5%, two-sided, across the four threshold tests (T1, T2, T2r, T3).
- **S1 is outside the family by choice**, so that its at most seven informative formations do
  not lower the threshold family's strictest step; its smallest attainable exact p is 0.0156
  (all seven signs agreeing), which a four-test family could accept only after another test had
  been rejected. It is reported with its exact p, its sign test, its pooled D and interval, and
  only the interval criteria of the outcome rules apply to it: it can be "no material
  difference", "worse" or "undetermined", never adopted.
- **Pooled D** on the evaluation span, with 95% and 99% percentile intervals from the cluster
  bootstrap over the 24 live evaluation formations (one index matrix,
  `np.random.default_rng(22)`, 10,000 draws, shared by all books; the S books' sufficient
  statistics are reindexed onto the 24 clusters with zero sessions for the empty ones, and the
  notebook says that the 99% interval's coverage was calibrated in notebook 22 for 24 live
  clusters, not 17), the leave-one-formation-out range over the 24 (S1: over its 17 live
  formations), and the hold-out value.
- **Power.** At the strictest Holm step of four (0.0125) the paired test reaches 80% power for
  an effect of about 0.9 s.d. of δ_f with 17 formations; S1 has no power at any Holm step. The
  notebook prints the s.d. of δ_f for every challenger.
- Hold-out and sub-period Sharpe ratios carry √(252 / sessions) only.

## Reported for every book, whatever the outcome

Notebook 22's list (net and gross Sharpe on live and all sessions, dollar P&L, cost, turnover,
matched-volatility drawdown and worst formation, concentration shares with dollars, effective
number of pairs, utilization, net notional), plus: completed round trips (`exit.sum()`), the
median rows held over all trades (and over completed ones beside it), the share of trips ended
by the stop, the share of positions open at the window's end, the number of flips, the
distribution of e* and of T3's a_L, d_L and b_L in units of s_f and of σ_d, the count of
pair-folds T3 leaves flat, the median of the rolling scale over s_f per pair-fold, and for the S
family the pairs of both pools by formation with their half-lives and p-values, the pairs the
pools share, the gated and ETF-list legs, and the sibling count: a pair is an **instrument
sibling** if both legs are among the 79 tickers of `notebooks/build/nb21_etf_exclusions.txt`
(lines beginning with `#` skipped), or if the duplicate test of `scaled_instrument_report` applied
to the two legs' formation closes alone (|ρ| ≥ 0.98 of daily log returns and |β − 1| < 0.15) flags
either leg with the other as partner. Share-class and ADR pairs are not classified, since no
classifier exists.

## Decision rules, fixed now

Notebook 22's ordered outcomes, with the 99% interval of pooled D on the evaluation span:

1. **No material difference**: the interval lies strictly inside ±0.10 ("consistently better"
   or "consistently worse" added when the primary test passes Holm in that direction or the
   interval excludes zero; "conflicting signals" when the two point in opposite directions).
2. **Adopted**: the primary test passes Holm in the positive direction; pooled D is positive on
   the evaluation span and in every leave-one-formation-out sample over the 24 live evaluation
   formations; and pooled D is positive in the hold-out. For T2 and T3 the composite against the
   baseline must also have a positive pooled D with a 99% interval that excludes zero.
3. **Worse**: the primary test passes Holm in the negative direction; or the interval lies below
   zero and pooled D is negative in every leave-one-formation-out sample.
4. **Undetermined**: anything else.

- For a signal change the books hold different trades, so with a baseline Sharpe whose standard
  error is 0.29 the 99% interval of D spans about ±0.75: outcome 1 is unreachable in practice,
  the reachable outcomes are adopted, worse and undetermined, and **undetermined is the expected
  outcome under no effect**. The notebook says so beside the table.
- Notebook 22's gated-pool clause is omitted on purpose: the T books hold the baseline's pairs,
  and the S universe already excludes gated pairs.
- The model check has no outcome. Beside the threshold outcomes the notebook writes "T1, T2 and
  T3 are scaled by a formation s.d. the window did not keep" if M4's median ratio is below the
  null's 2.5th percentile of that median or above its 97.5th; "centered on a mean the window
  did not keep" if M5's median is above the null's 97.5th percentile; and beside S1's outcome
  "S1 ranks by a speed the window did not keep" if M1's median ratio is above the null's 97.5th
  percentile. The percentiles are printed beside the observed medians.
- The hold-out sign is a necessary condition only; under no effect it agrees half the time.
- No definition, grid, seed or threshold changes after the first run. The fixed bands
  2 / 0.5 / 4 are not retuned.

## Forecasts, recorded before the run

Clauses marked (by construction) are expected whatever the data say and are recorded as checks
on the simulator, not as predictions.

1. **M1:** the share of pair-folds with a half-life ratio above one, and the median ratio, both
   exceed the simulated null's 95th percentile.
2. **M4:** the post-transient scale survives the window: M4's median lies inside the null's
   2.5th–97.5th percentile band.
3. **M2 and M3 on T1's trips:** the share held shorter than predicted exceeds the null's 95th
   percentile; more than 30% of pair-folds have a realized gross P&L per trip below the simulated
   10th percentile.
4. **M2′:** the baseline's realized median hold is below the simulated 10th percentile in more
   than 30% of pair-folds.
5. **T1 `ou_z`:** not adopted (by construction); it completes fewer evaluation round trips than
   the baseline's 574 (by construction); its pooled D is below −0.10.
6. **T2 `bertram`:** e* is below one daily innovation σ_d for more than half of the pair-folds
   (informed by the cost and innovation distributions seen in review); the book completes more
   than three times the baseline's evaluation round trips (by construction); its net
   evaluation-span Sharpe is below zero; D1's profit per session is within 5% of the model's, so
   the discretization is not the explanation.
7. **T2r `bertram_rolling`:** the same clauses.
8. **T3 `leung_li`:** |d_L − μ_f| and b_L − μ_f both exceed one s_f for more than half of the
   pair-folds with levels, so its bands are wider than Bertram's by an order of magnitude; it
   completes fewer evaluation round trips than the baseline (by construction); its outcome is
   "undetermined" and its pooled D is negative.
9. **S1 `kappa_rank`:** its pooled D against `p_rank` is below −0.10, and, pooled over the seven
   evaluation formations where the pools can differ, its instrument-sibling share exceeds P's;
   both shares are printed.

## Code

- `generate_threshold_signals` (the machine above) in `pairs/strategies/signals.py`, tested
  against `generate_pair_signals` under the band rule on the cached positions and on hand-built
  paths for the level rule, the flip, the stop-side exit, the entry block and the NaN rule.
- A new `pairs/models/ou.py`: `ou_fit` (the fit above with both scales, σ_d and the Kendall
  correction), `ou_first_passage` (mean and variance by the erfcx integrals), `bertram_threshold`
  (the return objective, its search and the tabulated u*) and `leung_li_levels` (F, G and the
  four roots), with tests that are the gates G2, G3 and G5 at reduced draw counts.
- The notebook is built by `notebooks/build/build_ou_notebook.py` and reuses notebook 22's
  simulator and statistics code. Development and debugging run in smoke mode (`OU_SMOKE=1`) on
  the distance-method pool with every (formation, unordered pair) of `alloc_folds.pkl["jobs"]`
  removed (745 of 780 rows remain), thinned as notebook 22's smoke selection is; the builder
  asserts that the intersection with the registered 287 is empty before any fit. In smoke mode
  the S family runs on the same reduced distance pool: P ranks by `ssd` ascending (standing in
  for the q-value), S1 by φ̃, and `day_rule_bh_dual.parquet` is read for counts only, never to
  build a pool, a fit or a cost cell. Smoke caches are prefixed `ou_smoke_`.
- The web app is not touched. Notebook 15's `ou_fit` in `avellaneda_lee.py` is not changed.

## Numbering

`notebooks/pairs_trading_23_ornstein_uhlenbeck_day_lake.ipynb`, a day-lake study numbered after
22 for the reason notebook 22 gave.

---

## Amendment 1 (2026-10-14, written after the build and before the first run on the registered pool)

The code was built and audited against the file above on the smoke pool only; nothing below
rests on a trading window of the registered pool or on any challenger. Everything in this
amendment is fixed, as the file above is; the notebook's Deviations section records anything
done differently from the file as amended.

### A1. Gate G5's target was the book's error

With the parameters printed under Leung and Li's Figure 2.10 (θ = 0.5388, μ = 16.6677,
σ = 0.1599, r = r̂ = 0.05, c_s = c_b = 0.05, L = 0.4834), equations (2.45) and (2.55) give
b_L = 0.5673 and d_L = 0.5048, not the printed 0.5570 and 0.4978. Three independent
implementations agree (the package's, the author's own quadrature written from the theorems, and
two solves that use none of the book's formulas); the printed pair is reproduced to four
decimals only at r ≈ 0.214; and at the printed cost, 1.8 stationary standard deviations, the
example's round trip loses money, so the caption cannot be what produced the figure. G5's first
clause is replaced by: `leung_li_levels` agrees to 1e-8 with an independent from-scratch
quadrature solver at the registered inputs and at two cases with a profitable entry interval
(the c / s = 0.005 case and the pair-unit case of the G5 cell); the registered printed numbers
are printed beside the registered inputs and not gated; their reproduction at r = 0.214 is
printed as a cross-check. The no-stop and Brownian-limit clauses stand.

### A2. Two counts, and the bad-row rule's effect

The bad-row rule as registered (one leg moves more than 40% in a session while the other moves
less than 5%) does not catch the HLT row it was written for: that move is −29.4% and HLT's rows
0 and 1 lie inside the skipped first 100 rows in any case, so "six HLT pair-folds of
2018-12-31" is void. Applied as written, the rule drops 67 volatility-product spike rows
(TVIX, UVXY, FAS and others, mostly 2018-02-05 to 02-08) from 32 pair-folds' post-transient
fits and moves one pair-fold (UVXY/WMT, 2018-06-29) outside (0, 1). The rule stays as
registered. The registered counts are corrected: 14 pair-folds have a post-transient φ̃
outside (0, 1) without the rule and **15 with it**; the threshold family's pool is therefore
**272** pair-folds. The notebook asserts both numbers and prints the rows the rule drops.

### A3. The threshold family's pool

A pair-fold removed under the clause above leaves the pool of every book in the threshold
family, the family's own baseline included, so that the books hold the same pairs on the same
windows and the comparisons stay like-for-like. Notebook 22's baseline on all 287 pair-folds is
reproduced by G1 and printed beside every table as `baseline_full`; the 574 evaluation round
trips and the 0.514 Sharpe are its figures, and the 272-pair-fold baseline's are printed beside
them. The primary tests and the outcomes read every threshold book against the 272-pair-fold
baseline.

### A4. T4, Palomar's threshold

Added to the threshold family, after reading Palomar (2025), *Portfolio Optimization*, Chapter
15, §15.5.2: the parametric optimal threshold for a Gaussian z-score, s* = argmax_s (1 − Φ(s)) s,
which has the unique solution **e_P = 0.7518**, with exit at the mean. On the frozen OU z:
long when −4 < z ≤ −e_P, short when e_P ≤ z < 4; a long exits when z ≥ 0 and a short when
z ≤ 0 (`exit="level"`, level 0; a flip occurs only if the opposite entry holds on the exit row);
stop 4, one-sided. Compared with T1, and with the baseline as its composite, like T2 and T3. His
nonparametric variant (an empirical trading frequency smoothed on the formation window with a
penalty λ) is not run: it is an in-sample fit with a free parameter, which notebooks 04 and 05
showed is noise on this pool. The Holm family is now five tests (T1, T2, T2r, T3, T4), strictest
step 0.01; the paired test's 80% power at that step is about 0.95 s.d. of δ_f with 17
formations. T4's trips, the distribution of its holding times and its flips are reported with
the others.

### A5. M6, the shape of the drift

Added to the model check, after the survey of nonlinear mean-reverting processes the author
was shown: whether the speed of reversion depends on the size of the deviation. On the trading
windows of the evaluation pair-folds of the threshold family's pool, every (z_t, ΔX_{t+1} / s_f)
pair with z_t = (X_t − μ_f) / s_f from the formation fit is pooled and binned by z_t with fixed
edges (−∞, −3, −2, −1.5, −1, −0.5, 0, 0.5, 1, 1.5, 2, 3, ∞). Reported per bin: the count, the
mean of ΔX_{t+1} / s_f, the mean of the linear OU prediction −(1 − φ̃) z_t, and the 2.5th–97.5th
percentiles of the bin mean over the simulated null paths (the same 200 draws). The summary
statistic is the ratio of the outer slope to the inner slope, each the OLS slope of ΔX / s_f on
z_t over the observations with |z_t| > 1.5 and with |z_t| ≤ 1 respectively, observed against
its null distribution. The same is printed for the hold-out and, separately, for the
same-underlying pair-folds. No decision rests on M6.

### A6. S1v, the speed ranking with a variance floor

Added as a sensitivity of the selection family with no outcome, after Palomar's §15.7.3 (an
optimum mean-reverting portfolio maximizes a reversion-speed proxy subject to a floor on the
spread variance). S1v ranks the universe as S1 does after removing every pair whose cost per
unit of scale, c / s_f with c = 2 × 5 bps × (P1 + |β| P2) / 10⁴ at the formation-close prices
(a flat cost, so that no cost cell is needed for ranking) and s_f the post-transient formation
s.d., exceeds **0.05**. P is unchanged. S1v's pairs, its pooled D against P with the shared
bootstrap interval, its paired δ_f, and the number of pairs the floor removes per formation are
reported beside S1's.

### A7. What has been seen since registration

The build's agents computed, on the registered pool, formation-window quantities only: the
post-transient fits (the 14 / 15 and 272 / 273 counts above, the rows the bad-row rule drops,
the median post-transient φ̃ used by D1) and the baseline book for G1. The author read Palomar's
Chapter 15 and a survey of nonlinear mean-reverting models between registration and this
amendment; neither contains data from this pool.

### A8. Forecasts added

10. **T4 `palomar`:** e_P lies between the median e* of T2 and the median |d_L − μ_f| of T3; T4
    completes more evaluation round trips than T1 (by construction) and more than the 272-pool
    baseline; its outcome is "undetermined" or "worse", not adopted, and its pooled D against
    the baseline is negative.
11. **M6:** the outer-to-inner slope ratio lies inside the null's 2.5th–97.5th percentile band
    on the evaluation span: no evidence that reversion is faster at large deviations once the
    formation transient is left out of the fit.
12. **S1v:** the floor removes more pairs from the 2010-07-01 and 2010-12-31 universes than
    from any other, and S1v's pooled D against P is above S1's.

---

## Amendment 2 (2026-10-14, written before the first run on the registered pool)

Added after reading Leung and Lu (2024), *Monte Carlo Simulation for Trading Under a Lévy-Driven
Mean-Reverting Framework* (arXiv 2309.05512v2). Nothing in it is data from this pool, and no
trading window of the registered pool has been read. Everything below is fixed.

### B1. T5, a horizon-aware threshold by simulation

Leung and Lu choose the entry level of a single trade cycle by Monte Carlo under the fitted
model, exiting at the mean, with no costs and a long horizon. T5 is that method with the pair's
measured cost, repeated trades, and this study's trading window:

- For each pair-fold of the threshold family's pool, 2,000 paths of the cached trading-window
  length n_i are simulated from the formation fit (μ_f, φ̃, σ_d, Gaussian innovations), started
  from the last formation residual, with one `np.random.default_rng(2401)` stream in cached fold
  order, separate from the model check's stream.
- On every path the registered machine is run with the frozen-z rules of T4 except the entry
  level: long when −4 < z ≤ −d, short when d ≤ z < 4, exit at the mean (`exit="level"`, level 0),
  one-sided stop at 4, fills at the decision close, and the pair's round-trip cost c in s_f units
  (the same c as T2) charged per completed round trip and per flip; a position open on the
  path's last row is marked at the last value with no exit cost, as the ledger does.
- The objective is the mean over paths of the net profit over the window, in s_f units per unit
  position, summed over the window's trades. d is searched on the grid {0.25, 0.50, …, 3.75}
  (15 points); the maximizer is refined by the vertex of the quadratic through the best grid
  point and its two neighbors, clipped to [0.25, 3.99]; at a grid edge the edge is taken and
  counted. If the objective is ≤ 0 at every grid point the pair-fold stays flat and is counted.
- d_H* is then applied to the realized trading window with the same rules. T5 is compared with
  T1, with the composite against the 272-pair-fold baseline like T2, T3 and T4. The Holm family
  is six tests (T1, T2, T2r, T3, T4, T5), strictest step 0.0083; the paired test's 80% power at
  that step is about one s.d. of δ_f with 17 formations.
- Reported: the distribution of d_H* in units of s_f and of σ_d, beside e_P and T3's |d_L − μ_f|;
  the simulated expected net profit at d_H* against T5's realized net profit on the same window,
  per pair-fold, as the share of pair-folds below the simulated 10th percentile and above the
  90th (the posterior-predictive reading of M3 applied to T5's own rule); and the count at a
  grid edge or flat.

### B2. M7, the shape of the innovations

Whether the Lévy version of the model is worth anything on this pool turns on whether the AR(1)
innovations are Gaussian. For each pair-fold: the skewness (`scipy.stats.skew`, `bias=False`)
and excess kurtosis (`scipy.stats.kurtosis`, `fisher=True`, `bias=False`) of the fit residuals
ε_t on the post-transient formation rows and on the trading-window fit rows, and the
Jarque–Bera p-value on each. Reported: the pooled distributions (median, quartiles) of both
statistics on both windows; the share of pair-folds with a Jarque–Bera p below 0.05 on each
window (5% expected under the Gaussian model); for the trading window, the share of pair-folds
whose excess kurtosis lies above the 97.5th percentile of the same statistic over the 200
Gaussian null paths of that pair-fold; for the formation window, the asymptotic Gaussian
standard errors √(6/n) and √(24/n) beside the medians. The same by formation and, separately,
for the same-underlying pair-folds. No decision rests on M7; the notebook says beside its
table whether a variance-gamma innovation model would be motivated.

### B3. Forecasts added

13. **T5 `horizon_mc`:** the median d_H* lies between e_P (0.7518) and the median |d_L − μ_f|
    of T3; T5's outcome is "undetermined", not adopted, and its pooled D is negative; its
    realized net profit falls below its own simulated 10th percentile in more than 30% of
    pair-folds.
14. **M7:** the median excess kurtosis of the post-transient formation innovations exceeds 1,
    and the trading-window kurtosis exceeds its Gaussian null's 97.5th percentile in more than
    half of the pair-folds (not informed by this pool; informed by what daily equity spreads
    usually look like).
