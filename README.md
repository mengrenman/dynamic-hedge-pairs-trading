# dynamic-hedge-pairs-trading

A **pairs trading** research toolkit built as a Python package, featuring a
production-style pipeline from cointegration screening through walk-forward
validation and capacity analysis.

- Cointegration screening (Engle–Granger + Johansen dual-gate) with **Benjamini-Hochberg FDR correction**
- Kalman-filter dynamic hedge ratio `beta_t`, intercept `alpha_t`, residual `epsilon_t`
- Stationarity diagnostics (ADF, KPSS, half-life) + composite pair scoring
- Signal generation (z-score thresholds, stops, cooldowns)
- Evaluation & PnL with a flexible cost model incl. **square-root market impact**
- **Walk-forward validation** — rolling OOS folds, no look-ahead; also **drives pair selection** by cross-fold Sharpe stability (the composite score is only a pre-filter)
- **Circuit breaker** — post-processor that flattens positions on z-score blow-outs or rolling drawdown breach, with configurable cooldown and re-entry guard
- **Portfolio analytics** — cross-pair spread-return correlation matrix, diversification score, inverse-variance position weights
- **Market-network filters** — TMFG (Massara–Di Matteo–Aste) and PMFG (Tumminello et al.) graph filters, and the Pozzi–Di Matteo–Aste $X+Y$ centrality/peripherality index (`pairs.stats.network`)
- **Hedge ratio stability tests** — CUSUM level-shift test + rolling β-drift detection; flags structurally shifted pairs
- **Universe-wide cointegration visualisation** — multiple-testing audit, p-value heatmap, network graph with hubs and communities, static-spread half-life diagnostics ([`pairs_trading_06_cointegration_network_yahoo.ipynb`](notebooks/pairs_trading_06_cointegration_network_yahoo.ipynb))
- **Walk-forward hyperparameter tuning** — Kalman noise and signal thresholds tuned on training folds only, with selection-bias checks and a default-vs-tuned hold-out comparison ([`pairs_trading_04_hyperparameter_tuning_yahoo.ipynb`](notebooks/pairs_trading_04_hyperparameter_tuning_yahoo.ipynb))
- **Time-varying cointegration tests** — is the dynamic hedge spurious? Eroğlu–Miller–Yiğit (2021) state-space tests with bootstrap inference, applied to the Kalman hedge ([`pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb`](notebooks/pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb))
- **Two sources, compared** — the OpenBB/Yahoo daily closes notebooks 01–05 use, held against the day lake: price and return agreement, why adjusted prices drift between snapshots, coverage and survivorship, and the data-quality traps each feed exposes or hides ([`08`](notebooks/pairs_trading_08_yahoo_vs_day_lake.ipynb) does the switch to the lake actually matter)
- **Day lake, done properly** — loader for an adjusted **Polygon.io** (now **Massive.com**) day lake (both layouts, ticker-reuse resolution, explicit price basis, dividends recovered from the adjustment factors) and a point-in-time liquidity screen; a three-notebook series rebuilds the daily study without survivorship bias, without dividend look-ahead and as a portfolio ([`09`](notebooks/pairs_trading_09_daily_lake.ipynb) universe & data quality, [`10`](notebooks/pairs_trading_10_daily_cointegration.ipynb) does cointegration exist, [`11`](notebooks/pairs_trading_11_daily_portfolio.ipynb) twenty-year portfolio backtest)
- **Minute bars** — loader for an adjusted **Polygon.io** (now **Massive.com**) minute lake (two layouts, sidecar-pruned reads, regular-session grid with rule-based early closes), microstructure diagnostics (Roll spread, signature plot, Epps effect), session rules and session-counted walk-forward folds; a three-notebook series takes the pipeline intraday ([`15`](notebooks/pairs_trading_15_minute_data.ipynb) data & microstructure, [`16`](notebooks/pairs_trading_16_intraday_backtest.ipynb) intraday walk-forward, [`17`](notebooks/pairs_trading_17_intraday_portfolio.ipynb) hold-out, latency, costs, capacity)
- **Cross-sectional statistical arbitrage** — the Avellaneda–Lee design on the same day lake: monthly PCA eigenportfolios as risk factors, a residual signal on every eligible name, factor-neutral dollar-neutral weights, and information coefficients measured before any portfolio is built ([`12`](notebooks/pairs_trading_12_daily_cross_sectional.ipynb) breadth instead of pair selection — and why it is not enough)
- **Instrument-type screening** — leveraged, inverse and same-underlying funds detected by behaviour (a near-exact magnified multiple of another listed instrument) rather than from a curated ticker list, so the universe screen can finally ask what a ticker *is* (`pairs.stats`/`market_data.instruments`)
- **Market-network methods** — TMFG and PMFG graph filters and the Pozzi $X+Y$ centrality index, validated against the published worked example of Grande & Borondo (2025), then used to test their "peripheral pairs beat central ones" claim on the day lake ([`pairs_trading_13_market_networks_day_lake.ipynb`](notebooks/pairs_trading_13_market_networks_day_lake.ipynb))
- **What "alpha" actually means** — the four things the word names, separated and then measured on the day lake: Jensen's intercept, Grinold's forecast alpha, an alpha *signal*, and P&L ([`pairs_trading_14_alpha_concepts_day_lake.ipynb`](notebooks/pairs_trading_14_alpha_concepts_day_lake.ipynb))
- **Interactive explorer** — a FastAPI + HTMX app over the same package: tune thresholds and costs live (~250 ms),
  refit the hedge on demand (~1.8 s), browse the 1.7M cached cointegration tests, and run notebook 11's
  twenty-year portfolio backtest as a cancellable background job. Figures are server-rendered by
  `pairs.plotting`, so the app and the notebooks cannot diverge; the walk-forward fold distribution is shown
  beside every headline Sharpe and the hold-out counts how many times you have looked at it
  ([`webapp/`](webapp/README.md))
- Plotting of trades over price legs, optionally with the spread's z-score as a third x-aligned panel (`show_zscore=True`) so an entry, the trades it caused and the reversion that closed it line up vertically

<p align="center">
    <img src="figures/signals.png" alt="BKNG/MA traded out of sample in 2026: trades on each leg and the spread z-score with its entry, exit and stop bands" width="94%">
</p>
<p align="center">
    <img src="figures/backtest.png" alt="Out-of-sample evaluation: Sharpe 2.64, net P&L $664, 14.8% annualised return, 2.3% max drawdown, 75% hit rate, 12 trades" width="94%">
</p>
<p align="center"><em>One pair, out of sample. The pair, the hedge, the thresholds and the z-score window were
all fixed on 2020–2025 before 2026 was touched — see <a href="notebooks/pairs_trading_01_yahoo.ipynb">notebook 01</a>.
Regenerate with <code>python figures/make_readme_figures.py</code>, which refuses to write a figure whose numbers
disagree with the notebook.</em></p>

---

## Notebooks

All notebooks are committed with their outputs and open directly on GitHub — click the name.
`pairs_trading_04`–`17` are
generated by the scripts in [`notebooks/build/`](notebooks/build/README.md), which also holds a runner that
executes any notebook in place. Notebooks 08–14 read a local day-bar lake, and notebooks 15–17 a
minute-bar lake, rather than the network (see the build README). Both lakes are built from
**Polygon.io** data — the company now trades as **Massive.com**.

Every notebook names its data source in its filename, and the numbering follows the source: the
seven ending in `_yahoo` — **notebooks 01–07** — take daily closes through OpenBB, whose default
provider is **yfinance** (Yahoo Finance); **08–14** read the local **day lake** (08 is the bridge,
comparing the two sources directly); and **15–17** read the **minute lake**. Both lakes are local
parquet builds of **Polygon.io** market data — Polygon.io is now **Massive.com** — and neither
touches the network at run time.

| Notebook | Purpose |
|----------|---------|
| [`pairs_trading_01_yahoo.ipynb`](notebooks/pairs_trading_01_yahoo.ipynb) | Original walkthrough — cointegration → Kalman → signals → IS/OOS evaluation, plus FDR demo, walk-forward, parameter sensitivity, and regime analysis sections |
| [`pairs_trading_02_yahoo.ipynb`](notebooks/pairs_trading_02_yahoo.ipynb) | **Revised clean workflow** — the conceptual spine, end-to-end: cointegration screen → Kalman hedge → **walk-forward pair selection** (§3.5; composite score is only a pre-filter) → in-sample **and OOS** evaluation with trade plots → honest limitations. Advanced layers are kept out of the spine: circuit breaker, portfolio analytics, hedge-ratio stability and capacity/market-impact live in the package API (with test coverage), while parameter sensitivity and regime analysis are also demonstrated in `pairs_trading_01_yahoo.ipynb` |
| [`pairs_trading_03_yahoo.ipynb`](notebooks/pairs_trading_03_yahoo.ipynb) | **No-BH ablation** — the *same* full pipeline as nb02 (screen → Kalman → walk-forward pair selection → in-sample & OOS backtests with signals and performance metrics → limitations) with **one deliberate change: no Benjamini-Hochberg correction** (`fdr_method="none"`, raw p-values). Isolates the effect of dropping multiple-testing control on selection and performance |
| [`pairs_trading_04_hyperparameter_tuning_yahoo.ipynb`](notebooks/pairs_trading_04_hyperparameter_tuning_yahoo.ipynb) | **Hyperparameter tuning** — nb02's pipeline with a new §3.6 that tunes the Kalman noise (`q`, `em_iters`) and signal parameters (`z_method`, `z_entry`, `z_exit`, `z_stop`, `max_hold_bars`, `cooldown_bars`) on walk-forward folds of the training window only (2,304 configurations, objective = pooled out-of-fold Sharpe), with marginal-effect plots, a split-half rank-stability check, a transfer check on runner-up pairs and a pooled out-of-fold equity curve; §4/§4b then evaluate default vs tuned side by side on the same 2026 hold-out |
| [`pairs_trading_05_tuning_revisited_yahoo.ipynb`](notebooks/pairs_trading_05_tuning_revisited_yahoo.ipynb) | **Why tuned configurations lose out of sample, and what actually helps** — a nested study over a 40-pair portfolio: tune on 2022–2024 out-of-fold bars, compare procedures (defaults, per-pair tuning, pooled tuning, robust objectives) on 2025, then touch 2026 once. Levers tested: Kalman noise and EM, static and rolling OLS hedges, the z-score look-back rule, training-window length, hub vs non-hub pairs. Finds the winner's curse dominates, no fitting change beats sensible defaults detectably, and a 174-bar hold-out cannot rank procedures at all |
| [`pairs_trading_06_cointegration_network_yahoo.ipynb`](notebooks/pairs_trading_06_cointegration_network_yahoo.ipynb) | **Universe-wide visualisation** — dual-gate screen with BH FDR on the Nasdaq-100 *and* the combined S&P 500 + Nasdaq-100 universe, with a multiple-testing audit (the raw Nasdaq-100 network collapses to nothing under BH); for the combined universe: heatmap, Kamada-Kawai network with community colouring and hub detection, degree distribution and the network with its hubs removed, clustered heatmap, static-spread half-life and stationarity diagnostics (with the Kalman spread for comparison), a gallery of the most tradeable spreads, and a per-ticker partner query |
| [`pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb`](notebooks/pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb) | **Is the dynamic hedge spurious?** — explores Eroğlu, Miller & Yiğit (2021), *Time-varying cointegration and the Kalman filter*: shows the package's Kalman hedge manufactures a stationary residual for independent random walks, implements the paper's state-space model with a persistent error state and its bootstrap $t$-tests (no / fixed / time-varying cointegration), replicates a slice of its Monte Carlo, and applies the tests to real pairs |
| [`pairs_trading_08_yahoo_vs_day_lake.ipynb`](notebooks/pairs_trading_08_yahoo_vs_day_lake.ipynb) | **Two sources for the same prices.** The Yahoo frame notebooks 01–05 ran on, loaded beside the day lake for the same 505 tickers and span. Returns agree to ~8 decimal places on 99.8% of name-days and every exception is a corporate action — but 4.6% of *price* cells differ by more than 1%, because an adjusted series is a function of when it was downloaded and carries no way to reconstruct its own basis. Fifteen names differ by a constant factor with only three explained by anything visible in the data — the largest, BKNG's 1/25, is shown to be a basis difference rather than a corrupt feed by the split-invariance of dollar volume, then confirmed against the company's SEC 8-K. Coverage is the real argument for the lake: only **39%** of 2006's 500 most traded names are in today's index list. Closes by running the same backtest on both sources — Sharpe 0.51 against 0.70, because `q` and `r` are absolute variances and the Kalman hedge is therefore not scale-invariant |
| [`pairs_trading_09_daily_lake.ipynb`](notebooks/pairs_trading_09_daily_lake.ipynb) | **Day lake I — universe, adjustment and survivorship.** (Lake built from Polygon.io / Massive.com data.) The two lake layouts and the per-instrument event summary; the three price series and the look-ahead in `close_tr` (its factor embeds every dividend paid *after* the date), with the dividend stream recovered from the factor steps so it can be accrued at the ex-date instead; ticker reuse across different companies (10% of the lake); the three hazards a curated index list hides — exchange test symbols, corporate actions the adjustment missed, and the split-adjustment price trap that makes 2008 Apple look like a $5 stock and 2008 Sirius like a $21 one; the point-in-time universe and its rules; why a volatility floor is needed (money-market ETFs pass cointegration tests against anything); and survivorship bias measured on an equal-weight buy-and-hold |
| [`pairs_trading_10_daily_cointegration.ipynb`](notebooks/pairs_trading_10_daily_cointegration.ipynb) | **Day lake II — does cointegration exist?** A standing screen: 39 semi-annual formations from 2006 to 2025, the 300 most traded eligible names, all 44,850 pairs through the Engle–Granger/Johansen dual gate, ~1.7M tests in total. Keeps the whole p-value distribution rather than the winners, and reads it against the uniform null with Storey's $\hat\pi_0$; Benjamini–Hochberg discoveries era by era; formation-window power at one, two and three years; the identity of what actually passes; and the Gatev et al. distance method as a control. Emits the three selection rules notebook 11 trades |
| [`pairs_trading_11_daily_portfolio.ipynb`](notebooks/pairs_trading_11_daily_portfolio.ipynb) | **Day lake III — twenty years, three selection rules.** Rolling formation/trading folds over 2006–2025 on the point-in-time universe: BH dual-gate, raw-$p$ dual-gate and distance selections traded side by side with a frozen per-fold hedge, robust z-score, 5 bps per leg-side, borrow, and dividends accrued on the ex-date (long legs receive, short legs pay). ~4,900 trading days give a Sharpe standard error near 0.23 — enough to conclude. Plus a worked single pair-fold in the style of notebooks 01–03 — trades on each leg, the z-score with its entry/exit/stop bands, the equity curve and the full performance summary — a Kalman-hedge variant, a cost sweep, the decay of the trade over time, and the same backtest re-run on today's index members to price the survivorship bias. Two sections then cut the headline down. **§7 asks what the instruments are**: `pairs.market_data.instruments` flags, by behaviour rather than from a list, anything whose returns are a near-exact *magnified* multiple of its closest relative — 28% of the BH selections touch one, and gating them takes the rule from **+0.40 to +0.31** and $43.6k to $32.5k. **§8 holds back 2023–2025**, the first window this series ever reserved: the gated BH rule returns **−0.91** there against +0.41 in development, and all six rule-by-gate combinations decline out of sample. At a hold-out standard error near 0.62 no single cell is decisive, but the +0.40 was never tested before and does not repeat |
| [`pairs_trading_12_daily_cross_sectional.ipynb`](notebooks/pairs_trading_12_daily_cross_sectional.ipynb) | **Day lake IV — leaving pairs behind.** Notebook 10's discovery step is the breadth bottleneck, so this one removes it: hold every eligible name every day, sized by its residual against a 15-factor monthly PCA model fitted on trailing data only. Information coefficients are measured *before* any portfolio is built — plain five-day reversal forecasts next-day residual returns at IC 0.033, $t$ = 8.6 over 986 probe days, while the Avellaneda–Lee s-score is dominated and its $\kappa$ filter passes 99% of names. The book holds ~500 positions on \$1M, but turnover of 15–65% a session leaves a break-even cost of only 0.7–5.7 bps a side: best net Sharpe at 5 bps is 0.05. A skip-a-day test rules out bid-ask bounce (−4% of gross Sharpe), and a decade split shows the edge is real and gone — gross Sharpe 0.77 in 2006–2015 against 0.02 in 2016–2025 |
| [`pairs_trading_13_market_networks_day_lake.ipynb`](notebooks/pairs_trading_13_market_networks_day_lake.ipynb) | **Do peripheral pairs really trade better?** A replication of Grande & Borondo (2025), *Embedding pairs trading in market networks* (Humanities & Social Sciences Communications 12:1477), which reports on 472 Binance tokens that pairs drawn from the **periphery** of the filtered cointegration network beat both central pairs and the classic top-cointegrated benchmark. `pairs.stats.network` implements the TMFG, the PMFG and the Pozzi $X+Y$ index and reproduces all 21 values of the paper's published Fig. 2 exactly (their tie rule turns out to truncate midranks). Run on 39 semi-annual formations of the day lake with notebook 11's engine, only the *direction* survives: peripheral beats central on both filters but significantly on only one (PMFG $+0.087$ Sharpe, $p=0.008$; TMFG $+0.006$, $p=0.81$), between portfolios that score $+0.04$ and $-0.04$ against a per-draw s.d. of 0.14 — and the plain BH cointegration benchmark returns **$+0.370$**, eight times the best network portfolio. Gating checks: a node's role survives to the next formation only 1.14–1.17× more often than chance, $X+Y$ shares 29% of its rank variance with raw degree (so the repo's hub finding is the same phenomenon, not a measurement artefact), and central edges clear the dual gate 3.6× more often than peripheral ones — so on a $-\log_{10}p$ network "prefer the periphery" largely means "prefer weaker cointegration evidence" |
| [`pairs_trading_14_alpha_concepts_day_lake.ipynb`](notebooks/pairs_trading_14_alpha_concepts_day_lake.ipynb) | **What "alpha" means, and which meaning this repository measures.** The word names four objects and only two have definitions: Jensen's regression intercept (given a factor set), Grinold's conditional expected residual return (given a risk model and an IC), an alpha *signal* (a hypothesis with no units), and P&L. Each is then measured on notebook 11's pairs book, with an equal-weight long-only book on the same tickers as a foil — the foil has the **higher** Sharpe (0.476 vs 0.402) and *negative* alpha. Findings: the book's market beta is 0.052 and its alpha 1.59%/yr at $t=1.39$, i.e. unconfirmed after twenty years; alpha moves more when the *sample* changes than when three factors are added; the signal's headline IC (0.197 per fold, $t=14$) is reproduced by **random, uncointegrated pairs** (0.205), so only the pooled IC separates them; the Fundamental Law predicts an IR of 1.1–3.4 against a realised 0.40; and against a matched placebo the same evidence puts the book at the **97th** percentile of the null by median per-fold Sharpe and the **2nd** by mean — screening widens the outcome distribution rather than shifting it |
| [`pairs_trading_15_minute_data.ipynb`](notebooks/pairs_trading_15_minute_data.ipynb) | **Minute bars I — the lake and the microstructure of the spread.** (Lake built from Polygon.io / Massive.com data.) The two lake layouts and their sidecar (and where the two builds disagree), the regular-session grid (UTC→Eastern, early closes, forward-fill within session), a universe pass at session resolution with a liquidity screen (traded-minute share, dollar volume, Roll spread), the dual-gate screen on 2022–2024 session closes (10 BH passes), and the intraday microstructure of the candidate spreads: signature plot, Epps effect, bounce autocorrelation, half-life by sampling interval (≈ 20 sessions at every interval), intraday seasonality, overnight share of variance (≈ 40%), and what a 2σ reversion is worth against the round-trip cost |
| [`pairs_trading_16_intraday_backtest.ipynb`](notebooks/pairs_trading_16_intraday_backtest.ipynb) | **Minute bars II — which intraday design survives out of fold.** Session-counted walk-forward (250 sessions fit / 20 trade, positions carried across refits) on the 2022–2024 training span, pooled out-of-fold Sharpe as the single objective: hedge estimation (static vs daily-cadence Kalman vs frequency-scaled intraday Kalman vs the naive port of the daily settings — every faster hedge whitens the spread and churns), sampling frequency (1 min to session closes), session rule (overnight vs flat by the close), and a look-back × entry grid read with the winner's curse in mind. Carries forward the untuned rule on a static hedge at 30-minute bars |
| [`pairs_trading_17_intraday_portfolio.ipynb`](notebooks/pairs_trading_17_intraday_portfolio.ipynb) | **Minute bars III — the hold-out, execution realism and the portfolio.** 2025 touched once: the chosen design, the grid's "tuned" cell, and the same pairs on daily bars (same hedge, and notebook 02's daily Kalman) — all within one standard error of zero; latency (1–3 bars), cost sweep to 10 bps (break-even above 10 bps out of fold; the hold-out is flat even at zero cost), capacity (participation and square-root impact at $10k–$1M per pair), overnight-vs-intraday and time-of-day attribution, and the effective number of independent bets in a hub-dominated candidate set |

---

## Package layout

```text
repo-root/
│
├─ environment.yml
├─ README.md
│
├─ pairs/
│  ├─ __init__.py
│  ├─ market_data/           # load_prices(), load_universe() (data adapters)
│  │  ├─ openbb_history.py
│  │  ├─ polygon_lake.py
│  │  ├─ minute_bars.py      # load_minute_bars(): adjusted Polygon.io/Massive.com minute lake → session grid
│  │  └─ daily_bars.py       # load_daily_bars(), liquidity_screen(), recover_dividends(): Polygon.io/Massive.com day lake
│  │
│  ├─ universes/             # ticker list files
│  │
│  ├─ stats/
│  │  ├─ cointegration.py    # find_cointegrated_pairs_dualgate() + benjamini_hochberg_fdr()
│  │  ├─ transforms.py
│  │  ├─ stationarity.py     # ADF/KPSS, half-life, summary
│  │  ├─ portfolio.py        # pair_return_correlations(), portfolio_diversification_score(), suggest_position_weights()
│  │  ├─ stability.py        # cusum_beta_stability(), rolling_beta_drift(), summarize_hedge_ratio_stability()
│  │  └─ microstructure.py   # roll_spread(), realized_variance_signature(), epps_correlation(), autocorr_by_interval()
│  │
│  ├─ models/
│  │  └─ kalman.py           # fit_kalman_hedge(), filter_kf_on_new(), continue_kalman_*()
│  │
│  ├─ strategies/
│  │  ├─ signals.py          # zscore_from_spread(), generate_pair_signals(), session_masks()
│  │  ├─ evaluate.py         # evaluate_pair_signals(), market_impact_bps()
│  │  └─ circuit_breaker.py  # apply_circuit_breaker(), CircuitBreakerConfig
│  │
│  ├─ validation/
│  │  └─ walk_forward.py     # walk_forward_splits(), walk_forward_session_splits(), walk_forward_backtest(), summarize_walk_forward()
│  │
│  └─ plotting/
│     └─ pair_trades.py      # plot_pair_legs_with_trades() — legs, and optionally the z-score panel
│
├─ notebooks/
│  ├─ pairs_trading_01_yahoo.ipynb                   # 01–05: daily closes via OpenBB/yfinance
│  ├─ pairs_trading_02_yahoo.ipynb
│  ├─ pairs_trading_03_yahoo.ipynb
│  ├─ pairs_trading_04_hyperparameter_tuning_yahoo.ipynb   # nb02 + walk-forward hyperparameter tuning
│  ├─ pairs_trading_05_tuning_revisited_yahoo.ipynb        # nested portfolio study: why tuning loses OOS, which levers help
│  ├─ pairs_trading_06_cointegration_network_yahoo.ipynb   # universe-wide screening & network visualisation
│  ├─ pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb        # time-varying cointegration tests (Eroğlu–Miller–Yiğit 2021)
│  ├─ pairs_trading_14_alpha_concepts_day_lake.ipynb              # the four meanings of "alpha", each measured on the day-lake pairs book
│  ├─ pairs_trading_13_market_networks_day_lake.ipynb               # TMFG/PMFG + Pozzi X+Y: does the periphery really trade better?
│  ├─ pairs_trading_08_yahoo_vs_day_lake.ipynb      # OpenBB/Yahoo vs the day lake: agreement, adjustment basis, coverage
│  ├─ pairs_trading_09_daily_lake.ipynb             # day lake: price basis, dividends, data quality, point-in-time universe, survivorship
│  ├─ pairs_trading_10_daily_cointegration.ipynb    # 20 years × 1.7M cointegration tests against the uniform null
│  ├─ pairs_trading_11_daily_portfolio.ipynb        # 20-year portfolio backtest: three selection rules, costs, regimes, survivorship
│  ├─ pairs_trading_12_daily_cross_sectional.ipynb  # PCA eigenportfolios, residual reversal on the whole cross-section, cost sweep
│  ├─ pairs_trading_15_minute_data.ipynb            # minute lake, session grid, liquidity, intraday microstructure of the spread
│  ├─ pairs_trading_16_intraday_backtest.ipynb      # intraday walk-forward: hedge cadence, sampling frequency, session rule, thresholds
│  ├─ pairs_trading_17_intraday_portfolio.ipynb     # 2025 hold-out once, latency, costs, capacity, P&L attribution, portfolio
│  └─ build/                               # scripts that generate nb04–nb17, the alpha, tv and visualisation notebooks + execute.py runner
│
├─ cache/                               # auto-created; gitignored
│  ├─ viz_prices_<universe>.parquet     # cached price data (visualisation nb)
│  ├─ viz_screen_<universe>.parquet     # cached screening results
│  ├─ viz_kalman_<universe>.pkl         # cached Kalman states
│  ├─ min_*.parquet / min_*.pkl         # minute-bar notebooks: session frame, screen, candidates, fitted fold states
│  ├─ day_*.parquet                     # day-lake notebooks: market bars, universe, screen, selection rules
│  └─ xs_*.parquet / xs_*.pkl           # cross-sectional notebook: IC panel, daily target weights
│
└─ tests/                    # 356 passing, 3 xfail (documented defects)
   ├─ test_accounting_invariants.py  # conservation laws: ledger/trade-log, split-filter, impact units
   ├─ test_plotting.py        # the z-score panel, and that the 2-panel default never moves
   ├─ test_cointegration.py
   ├─ test_evaluate.py
   ├─ test_fdr.py
   ├─ test_market_impact.py
   ├─ test_walk_forward.py
   ├─ test_circuit_breaker.py # circuit breaker (45 tests)
   ├─ test_portfolio.py       # pair correlation & weights (26 tests)
   ├─ test_stability.py       # hedge ratio stability (32 tests)
   ├─ test_minute_bars.py     # minute lake loader, both layouts, session grid
   ├─ test_daily_bars.py      # day lake loader, price basis, dividends, ticker reuse, liquidity gates
   ├─ test_microstructure.py  # Roll spread, signature plot, Epps effect
   └─ ...
```

---

## Installation

**1) Create the Conda env (recommended):**
```bash
conda env create -f environment.yml
conda activate stat-arb
```

**2) Install the package (editable for development):**
```bash
pip install -e .
```

Optional extras (pick what you need):
```bash
pip install -e ".[notebooks]"   # OpenBB data loader, matplotlib, JupyterLab — required for the notebooks
pip install -e ".[polygon]"     # pyarrow for the Polygon.io (now Massive.com) parquet-lake loader
pip install -e ".[test]"        # pytest
```

**3) Run the tests:**
```bash
pytest
```

In Jupyter, enable auto-reload during development:
```python
%load_ext autoreload
%autoreload 2
```

---

## Expected data format

Most functions expect a **long-form** price table.

- **Index:** `MultiIndex` with levels `('ticker', 'datetime')`
- **Columns:** must include `'close'`
- **Sorting:** sorted by `('ticker', 'datetime')`

```python
df_prices.index.names = ["ticker", "datetime"]
df_prices = df_prices.sort_index(level=["ticker", "datetime"])
```

```python
prices_wide = df_prices.pivot_table(
    index=df_prices.index.get_level_values("datetime"),
    columns=df_prices.index.get_level_values("ticker"),
    values="close",
    aggfunc="last",
)
```

---

## Quickstart (end-to-end)

```python
from pairs import (
    # screening
    find_cointegrated_pairs_dualgate,
    benjamini_hochberg_fdr,
    # modeling
    fit_kalman_hedge, filter_kf_on_new,
    # stats & selection
    summarize_spread_stationarity_joblib,
    # portfolio analytics
    pair_return_correlations, portfolio_diversification_score,
    suggest_position_weights,
    # hedge ratio stability
    cusum_beta_stability, rolling_beta_drift,
    summarize_hedge_ratio_stability,
    # signals & evaluation
    generate_pair_signals, evaluate_pair_signals,
    market_impact_bps,
    # risk management
    apply_circuit_breaker, CircuitBreakerConfig,
    # walk-forward validation
    walk_forward_backtest,
    # plotting
    plot_pair_legs_with_trades,
)
```

```python
cut = "2022-12-31"
df_train = df_prices.loc[pd.IndexSlice[:, :cut], :]
df_test  = df_prices.loc[pd.IndexSlice[:, cut:], :]
```

**1) Cointegration screening with BH FDR correction**
```python
screen = find_cointegrated_pairs_dualgate(
    df_train, alpha_eg=0.05, alpha_joh=0.05,
    fdr_method="bh",   # Benjamini-Hochberg (default)
    only_pass=True,
)
candidates = list(screen.index)   # list of (ticker1, ticker2)
```

**2) Fit Kalman on train**
```python
states_tr, params_tr = fit_kalman_hedge(
    df_train, pairs=candidates,
    mode="filter", em_iters=5, q=1e-5,   # causal; never "smooth" for anything a signal reads
    return_params=True,
)
```

**3) Stationarity scoring → candidate shortlist → walk-forward selection**

The composite stationarity score is a *pre-filter* that yields a candidate shortlist; the
traded pair is then chosen by **walk-forward cross-fold Sharpe stability** over the training
span (see `pairs_trading_02_yahoo.ipynb` §3.5), never by in-sample ranking alone. The OOS/test
window is never consulted during selection. Minimal self-contained version:
```python
summary_tr = summarize_spread_stationarity_joblib(states_tr, alpha=0.05)
shortlist  = list(summary_tr.sort_values(["verdict", "adf_p"]).index[:15])  # pre-filter only
# Recommended: rank `shortlist` by walk-forward cross-fold Sharpe (notebook §3.5),
# then take the most stable pair instead of the top in-sample row:
pair = shortlist[0]
t1, t2 = pair
```

**4) Continue Kalman on OOS (causal — no look-ahead)**
```python
frozen     = {k: params_tr[pair][k] for k in ("F", "Q", "R")}
last_state = (params_tr[pair]["last_state_mean"], params_tr[pair]["last_state_cov"])
states_te, _ = filter_kf_on_new(
    df_test.loc[(t1,), "close"],
    df_test.loc[(t2,), "close"],
    frozen=frozen, last_state=last_state, mode="filter",
)
```

**5) Build pair frame, generate signals**
```python
prices_wide = df_test.pivot_table(
    index=df_test.index.get_level_values("datetime"),
    columns=df_test.index.get_level_values("ticker"),
    values="close",
)
df_pair = states_te.join(
    prices_wide[[t1, t2]].rename(columns={t1: "P1", t2: "P2"}), how="inner"
)
signals = generate_pair_signals(
    df_pair, z_method="robust",
    z_entry=2.0, z_exit=0.5, z_stop=4.0,
    capital_per_pair=10_000,
)
```

**6) Evaluate with realistic costs**
```python
daily, trades, summary = evaluate_pair_signals(
    df_pair, signals,
    cost_bps=2.0,
    borrow_bps_per_year=50,
    avg_daily_volume_1=2_000_000,   # market impact (optional)
    avg_daily_volume_2=2_000_000,
    impact_eta=0.14,
)
print(f"Sharpe: {summary['sharpe']:.2f}  |  Impact cost: ${summary['impact_cost_total']:,.0f}")
```

**7) Walk-forward validation**
```python
wf_results = walk_forward_backtest(
    df_pair[["P1", "P2"]],
    train_bars=504, test_bars=126, step_bars=63,
    fit_fn=my_fit_fn,
    signal_fn=my_signal_fn,
    eval_fn=my_eval_fn,
)
# wf_results is a DataFrame with one row per fold
print(wf_results[["sharpe", "ann_return", "n_trades"]])
```

---

## Methodology

```
Universe (S&P 500 + NASDAQ-100, ~517 tickers)
      │
      ▼
Dual-gate cointegration screen
  Engle-Granger  ──┐
                   ├─► both must pass  ──► BH FDR correction (controls false discovery rate)
  Johansen       ──┘
      │
      ▼
Kalman filter (EM-fitted Q, R)
  time-varying beta_t, alpha_t  ──► spread residual epsilon_t
      │
      ├─► Hedge ratio stability tests
      │     CUSUM level-shift test + rolling β-drift  ──► flag / exclude unstable pairs
      │
      ▼
Stationarity scoring (ADF, KPSS, half-life, sigma)
  Composite z-score ranking ──► candidate shortlist (pre-filter only)
      │
      ▼
Walk-forward pair selection
  rolling train/test folds over the training span
  ──► rank by cross-fold Sharpe stability ──► traded pair chosen
  (test window never consulted for selection)
      │
      ▼
Portfolio analytics (multi-pair)
  Spread-return correlation matrix ──► diversification score
  Inverse-variance weights ──► capital allocation per pair
      │
      ▼
Signal generation
  z-score (rolling / robust) ──► entry / exit / stop thresholds
      │
      ▼
Circuit breaker (post-processor)
  z-score blow-out  ──┐
                      ├─► force flat + cooldown ──► patched signals
  Rolling drawdown  ──┘
      │
      ▼
Evaluation
  flat costs (bps, per-share, borrow) + square-root market impact
      │
      ├─► In-sample metrics (Sharpe, Ann. Return, Max Drawdown, ...)
      ├─► OOS evaluation (causal Kalman continuation, no look-ahead)
      ├─► Walk-forward validation (rolling folds, OOS Sharpe distribution)
      ├─► Parameter sensitivity heatmap (z_entry × z_exit)
      ├─► Regime-conditional analysis (COVID / Rate shock / AI bull)
      └─► Capacity analysis (Sharpe vs position size)
```

### Universe-wide visualisation pipeline (`pairs_trading_06_cointegration_network_yahoo.ipynb`)

```
Nasdaq-100 (~100) and S&P 500 + Nasdaq-100 (~500)  →  load_universe() + load_prices()   [viz_prices_<universe>.parquet]
      │
      ▼
find_cointegrated_pairs_dualgate(fdr_method="bh") on both universes          [viz_screen_<universe>.parquet]
  ├─► FDR audit table: pairs tested, expected false positives, raw EG, BH EG, Johansen, dual-gate
  └─► Nasdaq-100 raw-p network vs BH network side by side (the raw network is noise; BH keeps nothing)
      │
      ▼  combined universe, dual-gate pairs, strength = −log10(eg_p_fdr)
  ├─► Heatmap over the active tickers (alphabetical, upper triangle)
  ├─► Network graph (Kamada-Kawai, greedy-modularity communities, hubs ≥ 90th-pct degree, hub table)
  ├─► Degree distribution + the network with the mega-hubs removed
  └─► Clustered heatmap (hierarchical linkage)
      │
      ▼
static (OLS) spread and fit_kalman_hedge() spread on every dual-gate pair → summarize_spread_stationarity_joblib()   [viz_kalman_<universe>.pkl]
  ├─► Static-spread half-life histogram (tradeable 5–30 bar band highlighted)
  ├─► ADF p-value vs cointegration strength scatter, coloured by spread verdict
  ├─► Static vs Kalman half-life scatter (how much the smoothed EM filter compresses half-lives)
  ├─► Gallery of standardised static spreads for the most tradeable pairs
  └─► partners(ticker, universe, only_pass) — screen statistics + spread diagnostics per partner
```

---

## Top-level API

Import directly from `pairs` (lazy-loaded, startup fast):

### Stats
| Function | Returns |
|----------|---------|
| `find_cointegrated_pairs_dualgate(data, *, fdr_method="bh", ...)` | DataFrame of pair metrics; `eg_p_fdr` column for BH-adjusted p-values |
| `benjamini_hochberg_fdr(pvalues, alpha=0.05)` | `(reject: bool array, pvalues_adj: float array)` |
| `summarize_spread_stationarity_joblib(states, ...)` | DataFrame with `adf_p`, `kpss_p`, `halflife`, `resid_sigma`, `verdict` |
| `estimate_halflife(series)` | `float` |
| `test_spread_stationarity(series, ...)` | `dict` |
| `pair_return_correlations(kf_results, *, method, min_overlap)` | Symmetric N×N DataFrame of cross-pair Δresid correlations |
| `portfolio_diversification_score(corr_matrix)` | `float` — diversification ratio (1 / mean\|ρ_off-diag\|); >3 = well diversified |
| `suggest_position_weights(kf_results, corr_matrix, *, method, max_weight)` | DataFrame with `pair`, `resid_var`, `weight`, `suggested_capital_pct` |
| `cusum_beta_stability(beta, *, alpha)` | `dict` with `cusum_stat`, `critical_val`, `is_stable`, `cusum_series` |
| `rolling_beta_drift(beta, *, window, threshold_sigma)` | `dict` with `max_roll_std_ratio`, `is_stable`, `flagged_dates`, `roll_std_series` |
| `summarize_hedge_ratio_stability(kf_results, ...)` | DataFrame indexed by `(ticker1, ticker2)` with `overall_stable` column |
| `roll_spread(prices, *, as_bps)` | `float` — Roll (1984) effective spread from the serial covariance of price changes |
| `realized_variance_signature(prices, intervals, *, session)` | `pd.Series` — per-bar realized variance by sampling interval (signature plot) |
| `epps_correlation(p1, p2, intervals, *, session)` | `pd.Series` — return correlation by sampling interval (Epps effect) |
| `autocorr_by_interval(prices, intervals, *, session)` | `pd.Series` — first-order autocorrelation of changes by interval |

### Market data
| Function | Returns |
|----------|---------|
| `load_prices(source, ...)` / `load_polygon_lake(...)` / `download_openbb(...)` | long `(ticker, datetime)` frame with `close` |
| `load_daily_bars(tickers, start, end, root, *, price, with_dividends, layout)` | `(ticker, datetime)` frame of daily bars with `close` (basis: `split`, `tr` or `raw`), `raw_close`, `volume`, `dollar_volume`, `id` and optionally `dividend`; resolves ticker reuse to one instrument and reports the clashes |
| `liquidity_screen(bars, *, min_price, min_dollar_volume, min_ann_vol, max_abs_return, exclude, top_n)` | per-ticker frame with `eligible` — a point-in-time universe; the price gate reads the unadjusted close |
| `recover_dividends(close_split, tr_price_factor)` | `pd.Series` — per-share cash dividends implied by the steps in the total-return factor |
| `load_minute_bars(tickers, start, end, root, *, freq, price, layout)` | `(ticker, datetime)` frame of regular-session minute (or k-minute / session) bars with `close`, `volume`, `n_traded`; reads the ticker-per-day or the day-per-file lake layout (sidecar-pruned) |
| `detect_lake_layout(root)` / `detect_day_lake_layout(root)` / `nyse_early_closes(start, end)` / `summarize_sessions(frame)` | lake layout name / 13:00 early-close dates / per-ticker liquidity summary |

### Models
| Function | Returns |
|----------|---------|
| `fit_kalman_hedge(data, pairs, *, mode, em_iters, return_params)` | `(states_dict, params_dict)` |
| `filter_kf_on_new(P1_new, P2_new, *, frozen, last_state, mode)` | `(states_df, last_state_dict)` |
| `continue_kalman_on_window(data, k1, k2, params, ...)` | `(states_df, params_dict)` |

### Strategies
| Function | Returns |
|----------|---------|
| `generate_pair_signals(df_pair, *, z_entry, z_exit, z_stop, ...)` | signals DataFrame with `n1`, `n2`, `pos` |
| `evaluate_pair_signals(df_pair, signals, *, cost_bps, avg_daily_volume_1, ...)` | `(daily_df, trades_df, summary_dict)` |
| `market_impact_bps(shares_traded, price, avg_daily_volume, ann_vol_bps, eta)` | `float` or array — **whole-order** dollar cost (concession × shares; scales as \|Δq\|^1.5) |
| `zscore_from_spread(spread, method="robust", ...)` | `pd.Series` |
| `session_masks(index, *, exec_lag, flatten, no_entry_after)` | `(force_flat, block_entry)` boolean Series — session rules for intraday bars, fed to `generate_pair_signals` |
| `apply_circuit_breaker(signals, df_pair, *, z_halt, cb_cooldown_bars, z_reentry, max_drawdown_pct, ...)` | `(signals_cb, audit_df)` — patched signals + halt window log |
| `CircuitBreakerConfig(z_halt, cb_cooldown_bars, z_reentry, max_drawdown_pct, ...)` | Convenience dataclass wrapping all circuit breaker parameters |

### Validation
| Function | Returns |
|----------|---------|
| `walk_forward_backtest(df, *, train_bars, test_bars, step_bars, fit_fn, signal_fn, eval_fn)` | DataFrame — one row per fold |
| `walk_forward_splits(index, *, train_bars, test_bars, step_bars)` | `List[(train_idx, test_idx)]` |
| `walk_forward_session_splits(index, *, train_sessions, test_sessions, step_sessions)` | `List[(train_idx, test_idx)]` counted in sessions, for intraday bars |
| `summarize_walk_forward(results, metric_cols)` | DataFrame — mean/std/median/min/max per metric |

### Plotting
| Function | Returns |
|----------|---------|
| `plot_pair_legs_with_trades(df_pair, signals, *, show_zscore=False, z_entry=None, z_exit=None, z_stop=None, ...)` | `(Figure, (Axes, Axes))`, or `(Figure, (Axes, Axes, Axes))` with `show_zscore=True` |

---

## Representative results (walk-forward-selected pair: BKNG / MA)

The walk-forward selector (§3.5 of `pairs_trading_02_yahoo.ipynb`) chose **BKNG / MA** as the most
stable candidate — **mean fold Sharpe 1.09, median 1.17 across 16 rolling folds**, 13 of them
positive. Single-window metrics for that pair:

| Metric | In-sample (2020–2025) | OOS (H1 2026) |
|--------|----------------------|----------------|
| Sharpe ratio | 0.52 | 3.50 |
| Ann. return | 2.9% | 18.2% |
| Max drawdown | 8.4% | 1.3% |
| Trades | 85 | 13 |

> **Read the fold distribution, not either single window.** The OOS Sharpe rests on 13 trades in
> half a year, where the standard error of an annualised Sharpe is about 1.5; the in-sample
> figure covers one filter fit over a window containing the COVID break. The honest headline is
> the **walk-forward fold distribution (mean 1.09, median 1.17, 13/16 positive)**, which is what
> actually drove selection.
>
> **Selection is causal.** The pair is chosen by walk-forward cross-fold stability on the training
> span with H1 2026 held out, and — since the look-ahead fix described below — the stationarity
> diagnostics that build the candidate shortlist are computed from causally filtered states too.
> Re-running may surface a different pair as data is extended.
>
> **A look-ahead was removed here.** Until it was caught, notebooks 01–04 fitted the Kalman hedge
> with `mode="smooth"`. The RTS smoother conditions the state at every bar on the entire sample,
> so the residual the signal read, the stationarity scores that ranked candidates, and the
> in-sample P&L all used future data. The in-sample Sharpe this table used to report was **2.45**;
> the causal figure is 0.52. Removing it also changed which pair is selected — the contaminated
> score favoured pairs whose *smoothed* residual looked stationary — and the replacement scores
> better out of fold. See the note on `mode` in **Notes & gotchas**.

---

## Limitations

| Gap | Notes |
|-----|-------|
| **Selection bias** | Candidates screened from a large universe; even with walk-forward-based selection (§3.5), picking the best pair on validation folds inflates expectations. The H1 2026 test window is held out, but validation-set selection bias remains |
| **Single-pair OOS** | only 13 OOS trades in H1 2026; need ≥50 for statistical power — the single-window OOS Sharpe is essentially noise |
| **Portfolio weights are heuristic** | Inverse-variance ignores off-diagonal covariance; a minimum-variance optimizer would be more precise |
| **Circuit breaker is back-tested** | Thresholds calibrated in-sample may over-fit; validate OOS before deploying |
| **Market impact is estimated** | Square-root model calibrated to median US equities; illiquid names need higher η |
| **Regime dependence** | Strategy performs differently across COVID crash / recovery / rate shock / AI bull regimes |
| **Borrow availability** | Short borrow on hard-to-borrow names can spike to 500 bps/year |

---

## Notes & gotchas

- **Index hygiene:** Keep index names as `('ticker', 'datetime')` and ensure data are sorted.
- **Never fit with `mode="smooth"` for anything a signal reads.** `fit_kalman_hedge` and `kalman_dynamic_hedge_joblib` default to `mode="filter"`, the causal forward recursion. The RTS smoother conditions the state at bar *t* on the whole sample including bars after *t*, which collapses the residual half-life several-fold and manufactures stationarity — so it must not feed a trading signal, a backtest, a stationarity test or a pair-selection score. It is for describing a hedge ratio after the fact. **The default was `"smooth"` until this was found**, and notebooks 01–04 passed it explicitly; the in-sample Sharpe in `pairs_trading_02_yahoo.ipynb` fell from 2.45 to 0.52 when it was corrected. `return_params` is unaffected by the choice: `F`, `Q`, `R` come from the EM fit and the exported end state is taken from the filtered recursion in both modes, so an OOS continuation started from them is causal either way.
- **No look-ahead in OOS:** `filter_kf_on_new(..., mode="filter")` uses only the causal filter — no smoother — so no future information leaks into OOS states.
- **`em_iters > 0` silently overrides `init_cov`.** pykalman's `kf.em()` defaults to re-estimating `initial_state_mean` and `initial_state_covariance` along with `Q` and `R`, so a deliberately diffuse `init_cov=1e6` comes back at around 1e-4 and the prior the function validates has no effect. The behaviour is not worse — EM also learns a sensible starting beta where the diffuse prior starts at zero — but it is not what the signature implies. Pass `em_vars=("transition_covariance", "observation_covariance")` to learn only `Q` and `R` and keep the prior.
- **Never read cointegration off a Kalman residual.** A time-varying-coefficient filter manufactures a stationary residual for *any* two I(1) series: on two independent random walks at price-like levels, this package's own Kalman-residual stationarity verdict returns "stationary" **90% of the time** ([`pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb`](notebooks/pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb) §1), while Engle–Granger on levels stays near its nominal size. The half-life and `resid_sigma` computed on Kalman states describe the fit, not a long-run relation. `pairs.recommend_hedge` is the test that *does* answer the question — it returns `"dynamic"` only where the coefficient is shown to move, `"static"` for a fixed relation and `"none"` where there is none — and it belongs upstream of the Kalman step, beside the Engle–Granger and Johansen gates.
- **BH FDR is the default:** `find_cointegrated_pairs_dualgate` applies Benjamini-Hochberg correction by default (`fdr_method="bh"`). Pass `fdr_method="none"` to revert to raw p-values — [`pairs_trading_03_yahoo.ipynb`](notebooks/pairs_trading_03_yahoo.ipynb) runs the full pipeline that way as an ablation, isolating the effect of dropping multiple-testing control. Use `"none"` only for such experiments, never to expand a tradeable candidate set.
- **EM on train only:** Freeze `F, Q, R` from training; never re-fit EM on OOS data.
- **The dynamic hedge is unwarranted for almost every pair this repository screens.** Running `pairs.recommend_hedge` over all 2,151 BH dual-gate survivors (`analysis/gate_screen_survivors.py`): 55.4% show no cointegration the test can confirm, 37.3% a *fixed* coefficient, and only 7.3% a moving one — of which 73% rest on a negative θ̂ (an error alternating sign each bar) and 30% on a |θ̂| > 1 (outside the stationary region). Requiring θ̂ ∈ (0, 1) leaves **1.8%**. Both series were run over notebook 11's 287 traded pair-folds and they agree on the rate while disagreeing on the pairs: daily levels give 2.8% dynamic, weekly logs 2.1% after the θ̂ filter, but the two forms agree on only 52% of individual folds and exactly one fold is dynamic under both. Notebook 11 applies a Kalman hedge to all 287. Read the 55% `none` with care in the other direction: 77% of those have θ̂ < 0.9 yet cannot reject θ = 1, so that column is mostly "underpowered", not "absent".
- **The Sharpe column is spelled `shapre`.** `summarize_spread_stationarity_joblib` emits it that way and notebooks 01–05 depend on the spelling, so it is kept. It has already cost this repository once: the composite score read `final_df.get("sharpe_train", pd.Series(0.0, ...))`, which never matched, so a documented five-metric score only ever had four live terms. **Never read a results column with `.get(name, default)`** — index it directly, so a rename fails loudly instead of scoring every candidate zero.
- **Execution lag:** `generate_pair_signals` uses next-bar execution (backtest-safe by construction). On minute bars that fill is optimistic — the decision bar's own close — so `pairs_trading_17_intraday_portfolio.ipynb` also reports `exec_lag=2` and `3`.
- **Session rules:** `force_flat` / `block_entry` masks (built by `session_masks`) close positions before the session end and block late entries; the flat mask must cover the last `exec_lag` bars of a session for nothing to be carried overnight.
- **Day lake price basis** (Polygon.io / Massive.com)**:** `close_tr` is back-adjusted so its factor is 1.0 on the lake's *last* date, which means the factor at any historical date embeds dividends paid after it. Never trade on it. `load_daily_bars` defaults to `price="split"` and can hand back the dividend stream (`with_dividends=True`) to accrue at the ex-date instead.
- **Price gates go on the unadjusted close:** split adjustment is backward-looking, so a "$5 minimum price" applied to the adjusted series drops stocks that later split (2008 Apple shows as $4.86) and admits ones that later reverse-split (2008 Sirius shows as $21.30). `liquidity_screen` gates on `raw_close` by default.
- **A raw market lake is not a curated list:** it contains exchange test symbols (`ZVZZT` and siblings), tickers reused by different companies over time (10% of the lake), and corporate actions the adjustment missed. `load_daily_bars` resolves the reuse; `liquidity_screen` takes `exclude` and `max_abs_return` for the other two.
- **Minute lake** (Polygon.io / Massive.com)**:** `load_minute_bars` treats lake timestamps as UTC (tz-naive), keeps regular hours 09:30–16:00 Eastern (13:00 on rule-based early-close days), forward-fills within a session only, and reads either lake layout — the market layout via the `.idx.parquet` sidecar so one symbol touches only its row groups. The two builds of the lake differ in their dividend/split factors for a few names; never mix layouts in one analysis.
- **`cost_bps` is charged per transaction on both legs' notional.** `evaluate_pair_signals` applies it to `|Δn1|·P1 + |Δn2|·P2` — per transaction, per leg-dollar, not per round trip and not per pair-notional. Two consequences. A Roll effective spread must be **halved** before it is comparable, since Roll estimates the full bid-ask spread and a marketable order crosses half of it. And a pair's effective rate is its two legs' rates weighted by the notional each *turns over*: `generate_pair_signals` sizes a trade as `capital/(P1 + |beta|·P2)`, so the legs are equal only when `P1 = |beta|·P2`, which holds for barely a third of this repository's pair-folds. A plain mean of the two over-weights the smaller leg and overstates cost by about 30%. `pairs.pair_fold_costs(..., weights=)` takes leg one's share.
- **Measured execution cost is about 2 bps a transaction, not the 5 bps the notebooks assume.** Measured on the minute lake over the windows the strategies actually hold their names (`pairs.measure_ticker_window_costs`). Reported Sharpes are therefore conservative. The measurement is single-estimator and floor-assisted — a third of cells fall below the minimum tick and are replaced by it — so treat the direction and order of magnitude as settled and the decimal as not.
- **Market impact is a whole-order cost:** `market_impact_bps` returns the square-root price concession *multiplied by the share count*, so total cost scales as |Δq|^1.5 and bps-of-capital rises as √capital. It was returning only the per-share concession until 2026-09; `notebooks/pairs_trading_17_intraday_portfolio.ipynb` §5 is the only place it is exercised, and its capacity conclusion changed. `avg_daily_volume_1/2=None` (default) disables impact modelling; all other cost parameters remain active.
- **Two accounting defects are known and tracked, not fixed.** Both are pinned by `xfail(strict=True)` tests in `tests/test_accounting_invariants.py`, so fixing either forces the marker to be removed.
  1. *The per-trade log omits the exit bar's cost* (`evaluate.py`, the round-trip loop slices `sl.iloc[:-1]`). `n_trades`, `hit_rate`, `avg_win`, `avg_loss`, `profit_factor` and hold times are therefore slightly too favourable — about 2% on profit factor at the 1 bp costs these notebooks use. **Every `sharpe`, `ann_return`, `max_drawdown` and P&L figure is computed from the daily ledger and is unaffected.** A reversal additionally double-counts its bar and emits a zero-length trade, but `generate_pair_signals` cannot flip side within a bar, so that path is unreachable from this repo's own signals.
  2. *The Kalman continuation skips the boundary predict step* (`filter_kf_on_new` hands pykalman a posterior where it expects a prior, so `P ← FPFᵀ + Q` is never applied). Splitting a series therefore does not reproduce an uninterrupted filter; the prior is over-confident by `Q` for one bar at each fold boundary. Median error ≈ 0.7 z against a 2.0 entry threshold, decaying over 2–4 bars. Fixing it re-dates every out-of-sample continuation and invalidates the pickled state caches under `cache/min_wf_*.pkl`.

- **Walk-forward callbacks:** `fit_fn` returns an artefact, `signal_fn(df_test, artefact)` generates signals, `eval_fn(df_test, signals)` returns a flat metrics dict. Any fold where a callback raises is skipped with a warning, never aborts the run.
- **Walk-forward-driven selection:** In `pairs_trading_02_yahoo.ipynb` (§3.5) the composite stationarity score is only a *pre-filter* yielding a candidate shortlist; the traded pair is selected by **walk-forward cross-fold Sharpe stability** over the training span, with the OOS/test window never consulted during selection. The selection loop is parallelised across candidates with joblib (BLAS threads pinned to 1 to avoid oversubscription).
- **Visualisation caching:** `pairs_trading_06_cointegration_network_yahoo.ipynb` writes `cache/viz_prices_<universe>.parquet`, `cache/viz_screen_<universe>.parquet` and `cache/viz_kalman_<universe>.pkl` on first run. Subsequent runs load from cache and are near-instant. Delete the relevant file to force a fresh computation. The `cache/` directory is gitignored — these files are large and data-source specific.
- **Hub-node caveat:** High-degree nodes in the cointegration network (coloured red, degree ≥ 90th percentile) are often driven by a common latent factor rather than genuine pair cointegration. Treat them with extra scepticism and verify OOS behaviour before trading.
- **Circuit breaker calibration:** `z_halt` should be set at or above the `z_stop` used in signal generation. `max_drawdown_pct` thresholds that fire frequently in-sample indicate over-fitting — validate on held-out folds before deploying.
- **Hedge ratio stability:** Run `summarize_hedge_ratio_stability(states_tr)` after fitting the Kalman filter and **before** selecting pairs for live trading. Pairs flagged as `overall_stable=False` should be inspected (plot the CUSUM path) and excluded if β drift is persistent throughout the OOS window.
- **The instrument gate applies only inside notebook 11 §7–8.** The behavioural leveraged/inverse/volatility-fund detector (`pairs.detect_scaled_instruments`) is a trade-time filter in that notebook alone. Notebooks 10, 12, 13 and 14 all describe the **ungated** universe, and only notebook 11 §8 reports gated *and* held-out figures.
- **Cluster on the unit that actually varied.** Trades inside one semi-annual formation share a hedge, a universe and a market, so they are not independent observations. Treating notebook 11's 723 round trips as independent gives `t = 3.89`; clustering them on the 30 formations gives **2.16**. Applied to the hold-out, the wrong choice turned a 32% decline into an apparent 90% collapse.
- **Notebooks 01–05 are pinned, and the rule is narrower than "never re-execute".** Their prices live in `notebooks/cache/nb0N_train_*.parquet` and the notebooks read them when present, so re-executing *with the cache in place* is offline and reproducible. What must never happen is **deleting the cache**: a fresh OpenBB/Yahoo download re-bases adjusted prices for any name that has since split and moves every number. If you do re-execute, snapshot the outputs first and diff cell by cell.
- **Prose drifts from outputs, and the drift inverts conclusions.** Re-executing a notebook without rewriting its narrative has three times left this repository asserting the opposite of its own output — a section once claimed a tuned configuration traded *more* when it traded a third less, and that two of three pairs improved when none did. `tests/test_notebook_prose.py` asserts every number in prose was printed by some notebook (rounding-aware, comma-grouped integers in scope, both halves of a comparison checked against that notebook's own outputs). It does **not** catch a reversed direction with plausible digits, a claim containing no number, or a bare uncomma'd integer. After any re-execution, re-read the prose for direction words — more/less, improves/worsens — against the new output.
- **Two builder traps.** A triple-quoted docstring inside `code(r"""...""")` closes the outer string and the builder dies with a confusing `SyntaxError` — use `#` comments inside builder code cells. And `notebooks/build/renumber.py` scans `notebooks/build/*.{py,md}`, `pairs/**/*.py`, `tests/*.py` and the notebooks, but **not** `webapp/` or `figures/`; grep those by hand after `--apply`. It is also not idempotent: apply exactly once per mapping.
- **Seed any illustrative `.sample()`.** An unseeded display sample makes a notebook produce a spurious diff on every re-execution, which is noise in exactly the notebooks whose pinning exists to keep re-execution quiet.
- **Portfolio weights:** `suggest_position_weights` uses `states_tr` (the full multi-pair Kalman dict) as input and is best re-computed periodically (e.g., monthly) since correlations between pair spreads can shift with market regime.

---

## References

- Engle, R. F., & Granger, C. W. J. (1987). *Cointegration and Error Correction: Representation, Estimation, and Testing.*
- Johansen, S. (1991). *Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.*
- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems.*
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.*
- Almgren, R., & Chriss, N. (2001). *Optimal Execution of Portfolio Transactions.* — square-root market impact model.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). *Techniques for Testing the Constancy of Regression Relationships over Time.* — CUSUM test for structural change.
- Eroğlu, B. A., Miller, J. I., & Yiğit, T. (2021). [*Time-Varying Cointegration and the Kalman Filter.*](https://doi.org/10.1080/07474938.2020.1861776) Econometric Reviews 40(4). — a Kalman filter with a time-varying coefficient forces a stationary residual out of two independent I(1) series, so stationarity of the Kalman spread is **not** evidence of cointegration; the remedy is to move the error into the state and test its persistence. Implemented and applied in [notebook 07](notebooks/pairs_trading_07_tv_cointegration_kalman_yahoo.ipynb), which measures a 100% false-positive rate at this repository's own price scale and default `q`.
- Grande, M., & Borondo, J. (2025). [*Embedding Pairs Trading in Market Networks: A Network Science Approach to Portfolio Construction.*](https://doi.org/10.1057/s41599-025-05661-7) Humanities & Social Sciences Communications 12:1477. — filter the cointegration network to a PMFG/TMFG, rank assets by the Pozzi $X+Y$ index, and trade pairs from the **periphery**; reports better risk metrics than the top-cointegrated benchmark on 472 crypto tokens. Replicated on the day lake in [notebook 13](notebooks/pairs_trading_13_market_networks_day_lake.ipynb), where only the direction survives and the plain benchmark wins by eight times.
- Pozzi, F., Di Matteo, T., & Aste, T. (2013). *Spread of Risk Across Financial Markets: Better to Invest in the Peripheries.* Scientific Reports 3:1665. — the $X$/$Y$ centrality indices notebook 13 uses.
- Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N. (2005). *A Tool for Filtering Information in Complex Systems.* PNAS 102(30); Massara, G. P., Di Matteo, T., & Aste, T. (2016). *Network Filtering for Big Data: Triangulated Maximally Filtered Graph.* J. Complex Networks 5(2). — the PMFG and TMFG filters in `pairs.stats.network`.
