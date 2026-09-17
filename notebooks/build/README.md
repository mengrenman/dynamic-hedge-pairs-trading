# Notebook build scripts

Several notebooks are generated from Python rather than edited by hand, so their code and prose can be
diffed and reviewed like source:

| script | builds | from |
|---|---|---|
| `build_tv_notebook.py` | `../tv_cointegration_kalman_yahoo.ipynb` | scratch |
| `build_tuning_notebook.py` | `../pairs_trading_04_hyperparameter_tuning_yahoo.ipynb` | `../pairs_trading_02_yahoo.ipynb` (inherits every cell it does not rewrite) |
| `build_viz_notebook.py` | `../visualize_cointegrated_pairs_yahoo.ipynb` | scratch |
| `build_tuning_study_notebook.py` | `../pairs_trading_05_tuning_revisited_yahoo.ipynb` | scratch (cold run ≈ 8 min; caches the out-of-fold study under `cache/tuning_study_oof.pkl`) |
| `build_source_comparison_notebook.py` | `../pairs_trading_06_yahoo_vs_day_lake.ipynb` | scratch (needs the local **day** lake and `cache/viz_prices_spx_ndx_combined.parquet`; no network; runs in ≈ 20 s) |
| `build_daily_lake_notebook.py` | `../pairs_trading_07_daily_lake.ipynb` | scratch (needs the local **day** lake; cold run ≈ 3 min; caches `cache/day_market_bars.parquet`, `day_universe.parquet`, `day_ticker_collisions.parquet`) |
| `build_daily_screen_notebook.py` | `../pairs_trading_08_daily_cointegration.ipynb` | scratch (cold run ≈ 1 h — 39 formations × 44,850 pair-tests; caches `cache/day_screen.parquet`, `day_screen_power.parquet`, `day_distance.parquet` and `day_rule_*.parquet`) |
| `build_daily_portfolio_notebook.py` | `../pairs_trading_09_daily_portfolio.ipynb` | scratch (needs notebook 08's `day_rule_*.parquet`; cold run ≈ 5 min) |
| `build_minute_data_notebook.py` | `../pairs_trading_11_minute_data.ipynb` | scratch (needs the local minute lake; cold run ≈ 3 min; caches `cache/min_sessions.parquet`, `min_screen.parquet`, `min_candidates.parquet`, `min_candidates_1m.parquet`) |
| `build_intraday_backtest_notebook.py` | `../pairs_trading_12_intraday_backtest.ipynb` | scratch (reads notebook 11's caches or rebuilds them; cold run ≈ 10 min; caches fitted fold states as `cache/min_wf_<hedge>_<freq>.pkl` and the chosen design as `cache/min_design.json`) |
| `build_intraday_portfolio_notebook.py` | `../pairs_trading_13_intraday_portfolio.ipynb` | scratch (reads notebook 12's design and caches; cold run ≈ 5 min; caches `cache/min_holdout_<hedge>_<freq>.pkl`) |
| `build_cross_sectional_notebook.py` | `../pairs_trading_10_daily_cross_sectional.ipynb` | scratch (needs the local **day** lake; cold run ≈ 11 min on 16 cores; caches `cache/xs_ic_panel.parquet` and `cache/xs_targets.pkl`) |

Regenerate a notebook (cells only, outputs cleared) and then execute it:

```bash
python notebooks/build/build_tv_notebook.py
python notebooks/build/execute.py notebooks/tv_cointegration_kalman_yahoo.ipynb

python notebooks/build/build_tuning_notebook.py
python notebooks/build/execute.py notebooks/pairs_trading_04_hyperparameter_tuning_yahoo.ipynb
```

Every builder writes its target notebook in place; pass `--out PATH` to write elsewhere (for example to
diff against the committed version). `execute.py` runs with `notebooks/` as the working directory,
stores the outputs, and exits non-zero if a cell raised.

Notes

- Executing needs the package installed with the `notebooks` extra and network access for OpenBB; prices
  and the screening table are cached under `notebooks/cache/` (gitignored) after the first run.
- On a 16-core machine the time-varying cointegration notebook takes about 3 minutes and the tuning
  notebook about 8 minutes from a cold cache (2 minutes warm).
- Because `build_tuning_notebook.py` starts from `pairs_trading_02_yahoo.ipynb`, a change to nb02's pipeline
  cells propagates to nb04 by re-running the builder; the cells it rewrites are asserted on content, so
  the script fails loudly if nb02 drifts.
- The lake notebooks read local Polygon-derived parquet lakes, not the network, and treat them as
  read-only. The day-lake notebooks (06–08) use `~/local/parquet_lake/day_adj` unless `DAY_LAKE` says
  otherwise; the minute-bar notebooks (09–11) use `~/local/parquet_lake/minute_adj` unless `MINUTE_LAKE`
  says otherwise.
- Run each series in order on a cold cache. Notebook 09 needs notebook 08's `day_rule_*.parquet` and
  raises if they are missing; notebooks 12 and 13 rebuild notebook 11's caches if missing, but 11 also
  wants 10's `min_design.json` (it falls back to the design recorded in its own text).
