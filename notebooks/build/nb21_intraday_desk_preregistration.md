# Pre-registration: notebook 21, the desk at intraday frequency

Written 2026-09-29, before any of the code exists or any number has been seen. Everything here is
fixed. Anything the notebook does differently must be listed in a "Deviations" section at its end,
with the reason. It follows notebook 17's pre-registration and carries every lesson from its review.

## Question

At a breadth of 500 names and horizons from thirty minutes to a day, does any cross-sectional signal
formed from intraday price and volume data clear the measured half-spread, and how much do holding
overnight and turnover control buy? Secondary: do intraday-formed alphas add anything to the two
daily alphas that carried notebook 17, once both are judged on the same book?

## Data and universe

- Minute lake, **market layout** (`all_adjusted`, every ticker), aggregated to **30-minute bars** on
  the regular-session grid with `pairs.load_minute_bars` rules: 09:30–16:00 Eastern, 13:00 on NYSE
  early-close days, bars labeled by their start minute, forward-fill within a session only. Prices
  are split-adjusted; dividends are accrued from the day lake's factor steps for the overnight leg.
- Point-in-time universe: notebook 09's liquidity rules on the day lake, capped at the 500 most
  traded eligible names, rebuilt monthly, exactly notebook 12's `universe_at`, **minus** four
  exclusions, all operational: (i) SPY and the nine sector SPDRs used as factors; (ii) names flagged
  by `pairs.market_data.instruments.detect_scaled_instruments` in the formation window (leveraged
  and inverse funds); (iii) names whose formation-window daily-return R² against SPY or any sector
  SPDR is at least 0.90 (index trackers); (iv) the 79 tickers in `nb21_etf_exclusions.txt`, frozen
  today from notebook 17 §7.4. The count excluded under each rule is printed.
- Sessions 2010-01-04 to 2025-08-13. The lakes are read-only; the notebook writes only caches
  prefixed `intra_` under `notebooks/cache/`.

## Periods

| Period | Span | Use |
|---|---|---|
| Development | 2010-01-04 to 2018-12-31 | every fit, every selection, every look |
| Test | 2019-01-02 to 2022-12-30 | reported after development is frozen |
| Hold-out | 2023-01-03 to 2025-08-13 | notebooks 11 and 17's window; computed once, at the end |

## Books and horizons (fixed)

| Book | Formation | Target window | Position rule |
|---|---|---|---|
| **B, primary** | 15:30 close of the bar | 15:30 to the next session's 15:30 | partial adjustment toward the target once per session, φ = 0.25 primary, {1, 0.5, 0.1} reported; held overnight; 50 bp/yr borrow on shorts |
| A, intraday | 10:30 | 10:30 to the 16:00 close | full trade to target at 10:30, flat at the close, every session |
| C, thirty-minute | every bar close | the next 30-minute bar | full trade to target every bar, flat at the close |

Book B is the decision instrument. Books A and C are reported in full and never selected on; C
exists to put the fundamental-law arithmetic and the cost cliff on the page at the desk's own
frequency. Timing is strict: a signal at bar close t uses data through t; the target starts at t.

## Target

The hedged return over the book's window, with **no in-sample intercept**: the name's return minus
its factor betas times the factor returns over the same window, where the factor returns are the
ETFs' own bar returns and the PCA factors' portfolio returns over that window. Betas come from the
risk model below.

## The alpha list (fixed)

Standardized cross-sectionally at every formation (median and MAD, winsorized at ±3), then
multiplied by the pre-registered sign. The univariate IC test uses that sign; the blend is free to
disagree, and where the median ridge coefficient does, the notebook says so.

| # | Name | Definition, data through the formation bar | Sign | Source |
|---|---|---|---|---|
| 1 | `perio` | mean return over the same clock window as the target on the prior 20 sessions | + | Heston, Korajczyk & Sadka 2010 |
| 2 | `irev30` | residual return over the formation bar (the last 30 minutes) | − | Heston, Korajczyk & Sadka 2010; Lehmann 1990 |
| 3 | `ovn` | overnight return, prior close to today's 09:30 open | − | Lou, Polk & Skouras 2019 |
| 4 | `iopen` | intraday return from the 09:30 open to the formation bar, residual | − | Lou, Polk & Skouras 2019; Bogousslavsky 2021 |
| 5 | `ivol_abn` | log of the formation bar's volume over the mean volume of the same clock bar on the prior 20 sessions | + | Gervais, Kaniel & Mingelgrin 2001 |
| 6 | `irange` | log of today's high-low range so far over its mean at the same clock time on the prior 20 sessions | − | Ang, Hodrick, Xing & Zhang 2006 |
| 7 | `rev5` | residual return over the five sessions ending at the prior close (notebook 17's) | − | notebook 17 |
| 8 | `mom12_1` | return over sessions t−251..t−21 at the prior close (notebook 17's) | + | notebook 17 |

Alphas 7 and 8 are the daily controls: they carried notebook 17's equal-weight book, and the
secondary question is whether 1–6 add to them. Nothing is added after the first run; an alpha that
cannot be computed as written is dropped and the drop is logged.

## Risk model and neutralization

Notebook 12's monthly PCA loadings from daily returns (k = 15), plus the nine sector SPDRs, plus
size (log dollar volume). Target weights are projected orthogonal to the loadings **and a constant
column jointly** (notebook 17's corrected `neutralize`), then scaled to notebook 12's gross. The
book's realized exposures to SPY, the SPDRs, size and momentum are reported from returns aligned to
the realization window. Momentum is not neutralized; it is an alpha under test.

## Blends (both fixed)

1. **Equal-weight**: the mean of the eight signed z-scores. **The decision blend.**
2. **Ridge**: walk-forward, trailing 36 months, refit monthly, penalty from {0.1, 1, 10, 100} times
   the mean diagonal by leave-one-year-out cross-validation inside the training window, history
   admitted only once its target has realized. Reported with the share of refits per alpha whose
   coefficient is negative and the sign of the median coefficient.

## Costs and metrics

- **Primary cost basis** (notebook 18's): one basis point commission plus half the Roll spread per
  name per side, the Roll spread estimated per name and calendar year from that year's one-minute
  closes with `pairs.stats.microstructure.roll_spread`; borrow 50 bp/yr on overnight shorts.
  Unmeasured (name, year) cells take that year's median.
- **Secondary cost basis**: notebook 14's day-lake measured costs, so Book B can be set beside
  notebook 17's equal-weight book on the same footing.
- Per alpha, per horizon, per period: Pearson IC on the hedged target with the standard error from
  non-overlapping probe sessions (every fifth session; for Book C, every bar of every fifth
  session), the pairwise-complete count, and Benjamini–Hochberg at q = 0.10 across the eight alphas
  in development on Book B's window.
- Per book: gross and net Sharpe with s.e. √(252 / sessions) on all sessions and on sessions with
  a position, turnover per session, break-even cost in bps, P&L by year and by period, five-year
  blocks, exposures, and the five tradeable-edge clauses of notebook 14 §9 for Book B.
- Fundamental law for every book: realized blend IC, bets per year from breadth and turnover, IC ×
  √bets against the realized gross Sharpe, with the IC's standard error.
- Placebo, turnover-matched: each alpha's name-to-signal map is permuted **once per calendar
  month** and held fixed within it, so a placebo book turns over like the real one; 40 draws for the
  equal-weight Book B; the ridge placebo runs under a 30-minute budget and logs its draw count.
  Gross and net percentiles are both reported; the gross one is the comparison the null supports.

## Decision rules, fixed now

- The intraday desk **works** if Book B's equal-weight, φ = 0.25, net-of-primary-cost Sharpe on the
  test period is positive with t > 2 and the hold-out has the same sign. Otherwise it **does not**.
- Intraday alphas **add to the daily ones** if the Book B equal-weight blend of all eight has a
  higher development-period IC than the blend of alphas 7 and 8 alone by more than twice the
  difference's standard error, and the same sign of difference in the test period. Reported, not
  decided, if only one holds.
- Like-for-like against notebook 17's equal-weight book (0.596 net of day-lake measured costs, s.e.
  0.23) is reported on the secondary cost basis only, with the difference in combined standard
  errors; against notebook 11's 0.49 likewise.
- No formation time, window, threshold, universe rule, cost rule or alpha definition changes after
  the first development-period run. Sensitivities listed above are reported in full, never chosen
  from.

## Compute plan (informational, not a rule)

The 30-minute panel for the union of universe names is built once from the market layout by reading
each day file whole and filtering, and cached as `intra_bars30.parquet`; the per-name-year Roll
spreads are computed in the same pass and cached as `intra_roll.parquet`. The cold run may take up
to two hours; a warm run should be under ten minutes. If the ridge placebo does not finish 40 draws
in its budget, the shortfall is a logged deviation, never a change to anything else.

## Numbering

The notebook is a minute-lake study and follows 18–20, so it is notebook 21:
`pairs_trading_21_intraday_desk_alphas.ipynb`, built by `build_intraday_desk_notebook.py`.

## Amendment 1 (2026-09-29, written before the amended run)

The first full run (commit `e012840`) left about seventeen ETF-like names a month in the universe
because rules (ii)–(v) judge instruments by behavior. This amendment adds a sixth exclusion, by
**instrument type**, and changes nothing else:

- (vi) The Polygon security master (`~/local/parquet_lake/refdata/all/security_master.parquet`,
  read-only) gives each (ticker, holder id) a `type`. A name is kept only if its type is `CS`
  (common stock) or `ADRC` (ADR common); every other known type (`ETF`, `ETN`, `ETV`, `ETS`,
  `FUND`, `INDEX`, `PFD`, `WARRANT`, `UNIT`, `SP`, `RIGHT`, `ADRP`) is excluded. The lookup is by
  the (ticker, id) pair the day lake resolved, never by ticker alone, because reused tickers carry a
  different type in a different era (`FB`, `CA`, `EMC`, `GENZ`). A name whose pair has no type in
  the master keeps only the behavioral rules (ii)–(v); the count of such names is printed.
- Rules (i)–(v) remain in force. The exclusion table prints rule (vi) beside them.
- The alphas, books, periods, target, risk model, blends, costs, placebo and both decision rules are
  unchanged. Both rules are re-evaluated on the amended universe and printed; the first run's
  headline figures are quoted in code spans for comparison and remain in the repository's history.
