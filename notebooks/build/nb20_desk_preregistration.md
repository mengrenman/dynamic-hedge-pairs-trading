# Pre-registration: notebook 20, a statistical-arbitrage desk in miniature

Written 2026-09-28, before any of the code below exists or any number has been seen. Everything in
this file is fixed. Anything the notebook does differently must be listed in a "Deviations" section
at its end, with the reason.

## Question

Does combining several weak daily alphas, neutralizing them against a risk model, and controlling
turnover lift the day-lake cross-section above zero **net of measured costs** after 2015, where
notebook 12's single reversal signal died? Secondary: does the resulting book qualify as a
like-for-like second running example for notebook 14?

## Data and universe

- Day lake, market layout (`all_adjusted`), split-adjusted open/high/low/close and volume, dividends
  accrued from the factor steps as in notebook 09. The lakes are read-only; nb20 writes only caches
  prefixed `desk_` under `notebooks/cache/`.
- Point-in-time universe: notebook 09's liquidity rules, capped at the 500 most traded eligible
  names, rebuilt monthly (exactly notebook 12's `universe_at`).
- Sessions 2004-01-02 to 2025-08-13. The first two years are warm-up only.

## Periods

| Period | Span | Use |
|---|---|---|
| Development | 2006-01-04 to 2015-12-31 | every fit, every selection, every look |
| Test | 2016-01-04 to 2022-12-30 | reported after development is frozen |
| Hold-out | 2023-01-03 to 2025-08-13 | notebook 11's hold-out window; computed once, at the end |

## Target

Forward return over **h = 5** sessions (primary), hedged against the risk model with **no in-sample
intercept** (the notebook 12 fix). h = 1 and h = 21 are reported for every alpha but nothing is
selected on them. Timing is notebook 12's: signals use data through the close of session t and the
position is held from that close.

## The alpha list (fixed)

Each alpha is cross-sectionally standardized every day within the universe (median and MAD,
winsorized at ±3), then multiplied by its pre-registered sign. The univariate IC test uses that sign;
the blend is free to disagree, and where it does the notebook says so.

| # | Name | Definition (data through session t) | Sign | Source |
|---|---|---|---|---|
| 1 | `rev1` | residual return over session t | − | Lehmann 1990; Lo & MacKinlay 1990 |
| 2 | `rev5` | residual return over sessions t−4..t (notebook 12's signal) | − | notebook 12 |
| 3 | `ind_rev5` | 5-session return minus the sector-ETF-implied return (notebook 15's ETF residual) | − | notebook 15; Avellaneda & Lee 2010 |
| 4 | `sscore` | notebook 15's OU s-score on the ETF residual, 60-session window | − | Avellaneda & Lee 2010 |
| 5 | `mom1m` | return over sessions t−20..t | − | Jegadeesh 1990 |
| 6 | `mom12_1` | return over sessions t−251..t−21 | + | Jegadeesh & Titman 1993 |
| 7 | `vol_abn` | log(volume_t / median volume over t−19..t) | + | Gervais, Kaniel & Mingelgrin 2001 |
| 8 | `ovn21` | sum over t−20..t of log(open_s / close_{s−1}) | + | Lou, Polk & Skouras 2019 |
| 9 | `pvol21` | mean over t−20..t of log(high_s / low_s) (Parkinson range) | − | Ang, Hodrick, Xing & Zhang 2006 |
| 10 | `high52` | close_t / max(high over t−251..t) | + | George & Hwang 2004 |
| 11 | `max21` | max single-session return over t−20..t | − | Bali, Cakici & Whitelaw 2011 |

Residual returns in 1–2 are relative to the risk model below. Nothing is added to this list after
the first run. An alpha that cannot be computed as written is dropped and the drop is logged.

## Risk model and neutralization

Notebook 12's monthly PCA loadings (same k), plus the nine sector ETFs of notebook 15 as explicit
columns, plus one style column, log dollar volume (size). Positions are projected orthogonal to all
of these and made dollar-neutral with notebook 12's `neutralize`. Momentum is **not** neutralized,
because it is an alpha under test; its factor exposure is reported instead.

## Blends (both fixed)

1. **Equal-weight**: the mean of the eleven signed z-scores. The no-fit baseline.
2. **Ridge**: forward 5-session hedged return regressed on the eleven z-scores, fit on the trailing
   36 months, refit monthly, walk-forward. The ridge penalty is chosen by leave-one-year-out
   cross-validation inside the training window only, from the grid {0.1, 1, 10, 100} in units of
   the design's mean diagonal.

## Turnover control

Target weights come from the blend. Held weights move a fraction φ of the gap to the target each
session (Gârleanu & Pedersen 2013 partial adjustment). **φ = 0.25 is primary.** φ ∈ {1, 0.5, 0.1}
are reported as a sensitivity and nothing is selected on them.

## Costs and metrics

- Costs: notebook 14 §9's measured per-(ticker, year) costs, with flat 5 bps a leg-side as the
  comparison basis and 50 bp/yr borrow on shorts, exactly as notebook 11 charges.
- Per alpha, per period: Pearson IC on the hedged target with the standard error from non-overlapping
  probe days (notebook 12's estimator), the pairwise-complete day count, and the Benjamini–Hochberg
  verdict at q = 0.10 across the eleven alphas in development.
- Per book (two blends × primary φ): gross and net Sharpe with standard error √(252 / sessions),
  turnover per session, break-even cost in bps, exposure to each neutralized and non-neutralized
  factor, and the five tradeable-edge clauses of notebook 14 §9.
- Placebo: the same pipeline with each alpha's values permuted across names within each session,
  40 draws; the ridge blend's net Sharpe is reported as a percentile of that null.

## Decision rules, fixed now

- The daily desk **works** if the ridge book's net-of-measured-cost Sharpe on the test period is
  positive with t > 2 and the hold-out Sharpe has the same sign. Otherwise it **does not**, and the
  notebook says the daily-bar desk on liquid US names is out of reach with these alphas.
- It becomes a candidate second example for notebook 14 only if it is like-for-like with notebook
  11's book: same lake, full span, net of measured costs, and a Sharpe above 0.49 by more than the
  combined standard error. This is reported, not decided, in the notebook.
- No parameter, window, threshold, universe cap or alpha definition changes after the first
  development-period run. Sensitivities listed above are reported in full, never chosen from.

## Numbering

The notebook is a day-lake study and belongs with 08–16. When it lands, the minute-lake notebooks
17–19 move to 18–20 with `renumber.py` in the same commit, and this notebook becomes 17.
