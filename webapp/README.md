# Explorer — FastAPI + HTMX

```bash
pip install -e ".[web]"
python -m webapp                 # http://127.0.0.1:8000
# or: uvicorn webapp.main:app --reload
```

Reads the notebooks' caches under `notebooks/cache/` and never writes to them, so the app cannot
corrupt a notebook's state. Pages degrade rather than crash when a cache is absent.

## Three pages, three compute tiers

The architecture follows what things actually cost, measured on one pair of 1,508 daily bars:

| tier | what changes | cost | how the UI treats it |
|---|---|---|---|
| **1** | thresholds, z-method, costs, capital | signals + evaluate + render ≈ **250–430 ms** | live; submit on change |
| **2** | hedge model, `q`, `em_iters`, the pair | `fit_kalman_hedge` ≈ **1.8 s** (static/rolling ≈ 2 ms) | explicit run, memoised per spec |
| **3** | the portfolio backtest, the screen | **minutes to an hour** | background job, polled every 2 s, cancellable |

Grouping the knobs by what they cost is deliberate. Hiding a two-second refit behind the same
slider as a 250 ms one makes a UI feel broken.

## Figures

Server-rendered by `pairs.plotting` — the same `plot_pair_legs_with_trades` the notebooks call —
into a small LRU, and swapped in as `<img src="/fig/{token}">`. There is no client-side charting
and no build step; HTMX is one script tag. The cost is no hover or free zoom; the benefit is that
the app and the notebooks cannot draw different pictures of the same numbers.

## Why it argues with you

A GUI with threshold sliders and a live Sharpe readout is a machine for overfitting, and this
repository's own results say so: notebook 05 measured a tune-to-validation rank correlation of 0.36
for the Kalman model and −0.03 for a static hedge, and notebook 14 showed a 174-bar hold-out cannot
rank configurations at all. So the explorer is built to show the instability rather than hide it:

- **a single-window Sharpe is never displayed alone** — the walk-forward fold distribution sits
  above it, so the spread of outcomes is as prominent as the headline;
- **the hold-out is a separate, deliberate action** with a visible counter of how many times it has
  been looked at this session;
- **the pair menu says when it is hub-dominated** — 250 of the 279 Yahoo survivors involve NCLH or
  CCL, which notebook 05 §6 identifies as where the apparent edge lives;
- **the portfolio page has a switch for open issue #6**, so the leveraged-ETF question can be
  answered rather than argued about.

## Fidelity

`tests/test_webapp.py` asserts the explorer reproduces the notebooks, not merely that it renders:
the BKNG/MA hold-out must come out at Sharpe 2.64203 on 12 trades with net P&L $664.30, which is
what notebook 01 §4b prints. The portfolio job reproduces notebook 11 — Sharpe 0.402, $43,587,
287 pair-folds, 723 trades.

## Layout

```
webapp/
  state.py      lazy, process-wide, read-only registry over notebooks/cache
  compute.py    the three tiers, thin wrappers over `pairs`
  figures.py    matplotlib -> PNG bytes, LRU by content hash
  jobs.py       tier-3 lifecycle: submit, progress, cancel, error
  main.py       FastAPI routes
  templates/    Jinja2 + HTMX fragments
```
