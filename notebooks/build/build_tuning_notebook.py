"""Build notebooks/pairs_trading_04_hyperparameter_tuning_yahoo.ipynb from pairs_trading_02_yahoo.ipynb (cells only).

    python notebooks/build/build_tuning_notebook.py [--out PATH]

The result inherits every cell of nb02 except the ones rewritten below, so a change to nb02's
pipeline propagates by re-running this script; outputs are produced by execute.py.
"""
import argparse
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]               # notebooks/build/ -> notebooks/
src = nbf.read(ROOT / "pairs_trading_02_yahoo.ipynb", as_version=4)
cells = list(src.cells)
for c in cells:                                   # start from clean source
    if c.cell_type == "code":
        c.outputs = []; c.execution_count = None

def get(i, must_contain):
    s = "".join(cells[i].source)
    assert must_contain in s, f"cell {i} does not contain {must_contain!r}"
    return cells[i]

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s: nbf.v4.new_code_cell(s.strip("\n"))

# ───────────────────────── title ─────────────────────────
get(0, "Dynamic Hedge Pairs Trading").source = r"""
# Dynamic Hedge Pairs Trading — Hyperparameter Tuning

## `pairs_trading_04_hyperparameter_tuning_yahoo.ipynb`

A copy of `pairs_trading_02_yahoo.ipynb` — the clean end-to-end spine (cointegration screen → Kalman hedge →
walk-forward pair selection → in-sample and OOS evaluation) — with one addition: **§3.6 tunes the
hyperparameters that nb02 hard-codes**, using walk-forward folds on the training window only, and
**§4/§4b evaluate the default and the tuned configuration side by side on the same 2026 hold-out.**

What nb02 fixes by hand and what is tuned here:

| group | hyperparameter | nb02 default | note |
|---|---|---|---|
| Kalman | `q` (transition noise) | 1e-5 | with `em_iters>0` it is only the EM starting value |
| Kalman | `em_iters` | 5 | EM re-estimates Q and R on each training window |
| signals | `z_method` | robust | robust / rolling |
| signals | `z_entry`, `z_exit`, `z_stop` | 2.0 / 0.5 / 4.0 | thresholds on \|z\| |
| signals | `max_hold_bars`, `cooldown_bars` | None / 0 | holding cap, re-entry cooldown |

Everything up to §3.5 is identical to nb02 (the pair is still selected with the default parameters, so
tuning never sees the 2026 data), except that downloaded prices and the screening table are cached
under `notebooks/cache/` (gitignored) to make re-runs cheap.

Re-run from top to bottom; self-contained.
"""

# ───────────────────────── caching for prices & screen ─────────────────────────
get(6, "load_prices('openbb', tickers").source = r"""
# ── Load training prices (cached under notebooks/cache/, gitignored) ──────────
start_date = "2020-01-01"
end_date   = "2025-12-31"

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
_px_cache = CACHE / "nb04_train_prices.parquet"
if _px_cache.exists():
    df_train, failed = pd.read_parquet(_px_cache), []
    print(f"Loaded cached prices from {_px_cache}")
else:
    df_train, failed = load_prices('openbb', tickers, start_date, end_date, return_failed=True)
    df_train.to_parquet(_px_cache)

print(f"Loaded {len(df_train):,} rows for {df_train.index.get_level_values('ticker').nunique()} tickers")
if failed:
    print("No data for:", failed)
"""

get(9, "find_cointegrated_pairs_dualgate(").source = r"""
from pairs import find_cointegrated_pairs_dualgate

# fdr_method="bh" is the default — BH correction is applied automatically.
# The screen is the slowest step (~120k pairs); cache the result table.
_screen_cache = CACHE / "nb04_screen.parquet"
if _screen_cache.exists():
    summary_dual = pd.read_parquet(_screen_cache)
    print(f"Loaded cached screen from {_screen_cache}")
else:
    summary_dual = find_cointegrated_pairs_dualgate(
        df_filtered,
        alpha_eg=0.05, eg_trend="c",
        alpha_joh=0.05, joh_det_order=0, joh_k_ar_diff=1, joh_stat="trace",
        fdr_method="bh",
        n_workers=16, chunksize=8, show_progress=True,
    )
    summary_dual.to_parquet(_screen_cache)

print(f"\nTotal pairs screened : {len(summary_dual):,}")
print(f"Dual-gate pass (verdict=='pass'): {(summary_dual['verdict']=='pass').sum()}")
print("\nColumn eg_p_fdr contains BH-adjusted p-values:")
summary_dual.head(5)[["eg_t","eg_p","eg_p_fdr","eg_pass","joh_stat","joh_pass","verdict"]]
"""

# ───────────────────────── §3.6 tuning (inserted after §3.5.1, cell 28) ─────────────────────────
get(28, "summarize_walk_forward")
tuning_cells = [
md(r"""
## 3.6 Hyperparameter tuning — walk-forward on the training window only

nb02 picks the pair with a walk-forward driver but runs it with hand-set parameters. Here the same
folds (train 504 / test 126 / step 63 bars, 2020–2025) score a grid of configurations for the selected
pair. The objective is the **pooled out-of-fold Sharpe**: the out-of-fold daily net returns of all folds
are concatenated into one series and a single annualised Sharpe is computed on it, with a fixed $10k
capital base so returns are comparable across folds. Because the 126-bar test windows overlap (step 63),
each fold contributes only the 63 bars before the next refit (the last fold contributes all of its bars),
so every bar from bar 505 onward has exactly one out-of-fold return, from the most recently refitted
model — the way a live rolling re-estimation would trade. A configuration that sits flat most of the time
is not rewarded for it: flat days enter the series at zero. nb02's own selection metric, the *median* of
the per-fold Sharpes, is kept as a secondary column, and configurations with fewer than
`MIN_TOTAL_TRADES` trades across the folds are discarded.

The search is structured so that the expensive part is done once:

1. For each Kalman setting `(q, em_iters)` the filter is fitted on every training fold and run
   *causally* on the matching test fold (exactly `_sel_fit_fn` / `_sel_signal_fn` from §3.5). That gives
   per-fold frames with `P1, P2, beta, resid`. As in nb02's driver, the z-score look-back and its warm-up
   history come from the training fold's residual, never from the test fold.
2. Every signal configuration is then evaluated on those frames, which costs milliseconds per fold.

Nothing here touches 2026 data; the hold-out test in §4b is the only place the tuned configuration is
compared with the default on unseen prices.
"""),
code(r"""
from itertools import product
from pairs.validation import walk_forward_splits

TRAIN_BARS, TEST_BARS, STEP_BARS = 504, 126, 63     # same folds as the §3.5 selection driver

HP_DEFAULT = dict(q=1e-5, em_iters=5,                                    # nb02's hard-coded values
                  z_method="robust", z_entry=2.0, z_exit=0.5, z_stop=4.0,
                  max_hold_bars=None, cooldown_bars=0)
KALMAN_KEYS = ("q", "em_iters")
SIGNAL_KEYS = ("z_method", "z_entry", "z_exit", "z_stop", "max_hold_bars", "cooldown_bars")

KALMAN_GRID = [dict(q=q, em_iters=em) for q in (1e-6, 1e-5, 1e-4, 1e-3) for em in (0, 5)]
SIGNAL_GRID = [dict(z_method=m, z_entry=e, z_exit=x, z_stop=s, max_hold_bars=h, cooldown_bars=c)
               for m in ("robust", "rolling")
               for e in (1.5, 2.0, 2.5, 3.0)
               for x in (0.0, 0.5, 1.0)
               for s in (3.5, 4.0, 5.0)
               for h in (None, 60)
               for c in (0, 5)]
assert any({**k, **s} == HP_DEFAULT for k in KALMAN_GRID for s in SIGNAL_GRID), "default must be on the grid"
print(f"{len(KALMAN_GRID)} Kalman settings × {len(SIGNAL_GRID)} signal settings = "
      f"{len(KALMAN_GRID) * len(SIGNAL_GRID):,} configurations")

df_sel = prices_wide_tr[[ticker1, ticker2]].rename(columns={ticker1: "P1", ticker2: "P2"}).dropna()
splits = walk_forward_splits(df_sel.index, train_bars=TRAIN_BARS, test_bars=TEST_BARS, step_bars=STEP_BARS)
print(f"{ticker1}/{ticker2}: {len(df_sel):,} training bars → {len(splits)} walk-forward folds")
"""),
code(r"""
# ── Step 1: Kalman fits per fold, per (q, em_iters) — done once ─────────────────
def _kalman_key(q, em_iters):
    return f"q={q:g}, em={em_iters}"

def _fold_frame(df_train_f, df_test_f, q, em_iters):
    # Same mechanics as _sel_fit_fn + _sel_signal_fn in §3.5: EM fit on the train fold (filtered
    # states, so the residual history is causal), then continue the filter causally on the test
    # fold. The z-score look-back and its warm-up history come from the TRAIN fold only.
    _, _, states, params = _kalman_dynamic_hedge("P1", "P2", df_train_f, q=q, em_iters=em_iters,
                                                 mode="filter", return_params=True)
    if states is None or params is None:
        return None
    frozen = {"F": params["F"], "Q": params["Q"], "R": params["R"]}
    states_te, _ = filter_kf_on_new(df_test_f["P1"], df_test_f["P2"], frozen=frozen,
                                    last_state=(params["last_state_mean"], params["last_state_cov"]),
                                    mode="filter")
    return {"frame": df_test_f[["P1", "P2"]].join(states_te[["beta", "resid"]], how="inner"),
            "z_window": estimate_halflife_window(states["resid"]),
            "z_history": states["resid"]}

def _fold_frames_for(df_pair_full, splits_, q, em_iters):
    warnings.filterwarnings("ignore")
    return [_fold_frame(df_pair_full.loc[tr], df_pair_full.loc[te], q, em_iters) for tr, te in splits_]

import time
t0 = time.time()
_frames = Parallel(n_jobs=N_JOBS)(delayed(_fold_frames_for)(df_sel, splits, **k) for k in KALMAN_GRID)
fold_frames = {_kalman_key(**k): f for k, f in zip(KALMAN_GRID, _frames)}
print(f"Kalman fold fits: {len(KALMAN_GRID)} settings × {len(splits)} folds in {time.time() - t0:.0f}s")
"""),
code(r"""
# ── Step 2: evaluate every signal configuration on the precomputed folds ─────────
EVAL_KW = dict(cost_bps=1, borrow_bps_per_year=50, days_per_year=252, bars_per_year=252,
               capital_base=10_000)      # fixed base: out-of-fold returns are comparable across folds

def _pooled_sharpe(n, s1, s2, periods=252):
    # Annualised Sharpe of a return series known only through its count, sum and sum of squares
    # (mean / population std). Lets any subset of folds be pooled without keeping the returns.
    n = np.asarray(n, float); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        var = s2 / n - mean ** 2
        out = mean / np.sqrt(var) * np.sqrt(periods)
    return np.where((n >= 2) & (var > 0), out, np.nan)

def _oof_slices(frames, scfg):
    # Per-fold out-of-fold daily results, keeping only the bars before the next refit
    # (the last fold keeps all of them) so each bar appears exactly once.
    out = []
    for k, f in enumerate(frames):
        fr = None if f is None else f["frame"]
        if fr is None or len(fr) < 30:
            out.append(None); continue
        sig = generate_pair_signals(fr, capital_per_pair=10_000,
                                    z_window=f["z_window"], z_history=f["z_history"], **scfg)
        daily, _, summ = evaluate_pair_signals(fr[["P1", "P2"]], sig, **EVAL_KW)
        if k < len(frames) - 1:
            daily = daily.iloc[:STEP_BARS]
        out.append((daily, summ))
    return out

def _eval_config(kcfg, scfg, frames):
    warnings.filterwarnings("ignore")
    n_f = len(frames)
    sharpes, trades = [], 0
    fold_n = np.zeros(n_f); fold_s1 = np.zeros(n_f); fold_s2 = np.zeros(n_f)
    for k, item in enumerate(_oof_slices(frames, scfg)):
        if item is None:
            sharpes.append(np.nan); continue
        daily, summ = item
        sharpes.append(summ["sharpe"]); trades += int(summ["n_trades"])   # per-fold stats on the full 126-bar window
        r = daily["ret_net"].to_numpy(float); r = r[np.isfinite(r)]
        fold_n[k], fold_s1[k], fold_s2[k] = len(r), r.sum(), (r ** 2).sum()
    s = np.asarray(sharpes, float)
    return {**kcfg, **scfg,
            "oof_sharpe":       float(_pooled_sharpe(fold_n.sum(), fold_s1.sum(), fold_s2.sum())),
            "oof_ann_return":   float(fold_s1.sum() / max(fold_n.sum(), 1) * 252),    # arithmetic, on the $10k base
            "wf_sharpe_median": np.nanmedian(s) if np.isfinite(s).any() else np.nan,
            "wf_sharpe_mean":   np.nanmean(s)   if np.isfinite(s).any() else np.nan,
            "pct_pos_folds":    float(np.mean(s > 0)),
            "wf_trades_total":  trades,
            "n_folds":          int(np.isfinite(s).sum()),
            "fold_sharpes":     s, "fold_n": fold_n, "fold_s1": fold_s1, "fold_s2": fold_s2}

t0 = time.time()
tuning = pd.DataFrame(Parallel(n_jobs=N_JOBS, batch_size=32)(
    delayed(_eval_config)(k, s, fold_frames[_kalman_key(**k)]) for k in KALMAN_GRID for s in SIGNAL_GRID))
print(f"{len(tuning):,} configurations × {len(splits)} folds evaluated in {time.time() - t0:.0f}s")

def _match(df, cfg):
    # None (max_hold_bars) becomes NaN inside a DataFrame column, so compare with isna() there.
    m = np.ones(len(df), dtype=bool)
    for k, v in cfg.items():
        m &= (df[k].isna() if v is None else (df[k] == v)).to_numpy()
    return m

default_row = tuning[_match(tuning, HP_DEFAULT)].iloc[0]
OBJECTIVE = "oof_sharpe"
valid = (tuning[tuning["wf_trades_total"] >= MIN_TOTAL_TRADES]
         .dropna(subset=[OBJECTIVE])
         .sort_values(OBJECTIVE, ascending=False, kind="mergesort"))   # stable: ties keep grid order (simplest first)
print(f"{len(valid):,} configurations with ≥ {MIN_TOTAL_TRADES} total trades\n")

show_cols = list(KALMAN_KEYS + SIGNAL_KEYS) + ["oof_sharpe", "oof_ann_return", "wf_sharpe_median", "pct_pos_folds", "wf_trades_total"]
print("Top 10 configurations by pooled out-of-fold Sharpe:")
print(valid[show_cols].head(10).to_string(index=False, float_format="{:.3f}".format))
print("\nnb02 default configuration:")
print(default_row[show_cols].to_frame().T.to_string(index=False, float_format="{:.3f}".format))
print(f"\nDefault ranks {int((valid[OBJECTIVE] > default_row[OBJECTIVE]).sum()) + 1} "
      f"of {len(valid)} valid configurations.")
"""),
md(r"""
### 3.6.1 What matters and what does not

Marginal effect of each hyperparameter: the distribution of pooled out-of-fold Sharpe across all
configurations sharing a value (box = interquartile range across the other seven dimensions). A flat
row means the parameter does not matter much on this pair; a steep one is where the tuning gain comes
from.
"""),
code(r"""
def _value_labels(series):
    # Human-readable group labels in a sensible order: 'None' first, then numeric ascending, or strings sorted.
    if pd.api.types.is_numeric_dtype(series):
        labels = series.map(lambda v: "None" if pd.isna(v) else f"{v:g}")
        order = (["None"] if series.isna().any() else []) + [f"{v:g}" for v in sorted(series.dropna().unique())]
    else:                                        # string-valued (pandas 3 'str' dtype or object)
        labels = series.astype(str); order = sorted(labels.unique())
    return labels, order

params_to_plot = list(KALMAN_KEYS + SIGNAL_KEYS)
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, p in zip(axes.ravel(), params_to_plot):
    labels, order = _value_labels(valid[p])
    groups = valid.groupby(labels, sort=False)[OBJECTIVE]
    data = [groups.get_group(g).values for g in order if g in groups.groups]
    ticks = [g for g in order if g in groups.groups]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(ticks) + 1), ticks)      # works on matplotlib < 3.9 too
    ax.axhline(default_row[OBJECTIVE], color="red", ls="--", lw=1, label="nb02 default")
    ax.set_title(p); ax.grid(alpha=0.3)
    if p == params_to_plot[0]: ax.legend(fontsize=8)
fig.suptitle(f"Pooled out-of-fold Sharpe by hyperparameter value — {ticker1}/{ticker2} (configs with ≥ {MIN_TOTAL_TRADES} trades)")
plt.tight_layout(); plt.show()
"""),
md(r"""
### 3.6.2 How much of the improvement is selection bias?

Picking the maximum over a couple of thousand configurations guarantees a flattering in-sample number
even when nothing is really better. Two cheap checks before believing the winner:

* **Where the default sits** in the distribution of all configurations, and how far the top of the
  distribution is from its bulk (a lone outlier at the top is more suspicious than a plateau of similar
  configurations).
* **Split-half rank stability**: rank every configuration by its pooled Sharpe on the odd folds and
  again on the even folds. If tuning is finding structure rather than noise, the two rankings agree
  (Spearman correlation well above zero) and the winner on one half is near the top on the other.

The decisive check is the 2026 hold-out in §4b, which none of this has touched.
"""),
code(r"""
from scipy.stats import spearmanr

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ax.hist(valid[OBJECTIVE], bins=50, color="steelblue", alpha=0.7)
ax.axvline(default_row[OBJECTIVE], color="red", ls="--", label=f"nb02 default = {default_row[OBJECTIVE]:.2f}")
ax.axvline(valid[OBJECTIVE].iloc[0], color="green", ls="--", label=f"best = {valid[OBJECTIVE].iloc[0]:.2f}")
ax.set_xlabel("pooled out-of-fold Sharpe"); ax.set_ylabel("# configurations"); ax.legend()
ax.set_title("Distribution over the grid")

# split-half rank stability: pooled Sharpe over the odd folds vs over the even folds
N_, S1_, S2_ = (np.vstack(valid[c].values) for c in ("fold_n", "fold_s1", "fold_s2"))   # configs × folds
odd  = _pooled_sharpe(N_[:, 0::2].sum(1), S1_[:, 0::2].sum(1), S2_[:, 0::2].sum(1))
even = _pooled_sharpe(N_[:, 1::2].sum(1), S1_[:, 1::2].sum(1), S2_[:, 1::2].sum(1))
ok = np.isfinite(odd) & np.isfinite(even)
rho = spearmanr(odd[ok], even[ok]).correlation
ax = axes[1]
ax.scatter(odd[ok], even[ok], s=6, alpha=0.4)
ax.scatter(odd[0], even[0], color="green", s=60, label="best on all folds", zorder=3)
if default_row.name in valid.index:            # the default may have too few trades to be 'valid'
    d_i = int(np.where(valid.index == default_row.name)[0][0])
    ax.scatter(odd[d_i], even[d_i], color="red", s=60, label="nb02 default", zorder=3)
ax.set_xlabel("pooled OOF Sharpe, odd folds"); ax.set_ylabel("pooled OOF Sharpe, even folds")
ax.set_title(f"Split-half stability across configurations: Spearman ρ = {rho:.2f}"); ax.legend()
plt.tight_layout(); plt.show()

top_k = 20
rank_even = pd.Series(even, index=valid.index).rank(ascending=False)
print(f"Top-{top_k} configs by odd-fold Sharpe have median even-fold rank "
      f"{rank_even.loc[valid.index[np.argsort(-odd)[:top_k]]].median():.0f} of {ok.sum()}  "
      f"(pure noise would give ≈ {ok.sum() / 2:.0f}).")
"""),
md(r"""
### 3.6.3 Choose the tuned configuration, and check it transfers

The winner is the best pooled out-of-fold Sharpe among configurations with enough trades. As a guard
against a configuration that is idiosyncratic to the selected pair, the same configuration is
re-scored on the next three candidates from the §3.5 shortlist (each with its own walk-forward folds):
a genuine improvement in the *strategy* should not evaporate on neighbouring pairs.
"""),
code(r"""
TUNED = {k: valid.iloc[0][k] for k in KALMAN_KEYS + SIGNAL_KEYS}
TUNED["q"] = float(TUNED["q"]); TUNED["em_iters"] = int(TUNED["em_iters"])
TUNED["max_hold_bars"] = None if pd.isna(TUNED["max_hold_bars"]) else int(TUNED["max_hold_bars"])
TUNED["cooldown_bars"] = int(TUNED["cooldown_bars"])
for k in ("z_entry", "z_exit", "z_stop"): TUNED[k] = float(TUNED[k])

print("Tuned configuration:")
for k in KALMAN_KEYS + SIGNAL_KEYS:
    flag = "" if TUNED[k] == HP_DEFAULT[k] else "   ← changed"
    print(f"  {k:14s} {str(HP_DEFAULT[k]):>8s} → {str(TUNED[k]):<8s}{flag}")

def _wf_score(pair, cfg):
    t1, t2 = pair
    dfp = prices_wide_tr[[t1, t2]].rename(columns={t1: "P1", t2: "P2"}).dropna()
    sp = walk_forward_splits(dfp.index, train_bars=TRAIN_BARS, test_bars=TEST_BARS, step_bars=STEP_BARS)
    frames = _fold_frames_for(dfp, sp, cfg["q"], cfg["em_iters"])
    return _eval_config({k: cfg[k] for k in KALMAN_KEYS}, {k: cfg[k] for k in SIGNAL_KEYS}, frames)

others = [p for p in wf_selection.index if p != (ticker1, ticker2)][:3]
jobs = [(p, name, cfg) for p in [(ticker1, ticker2)] + others for name, cfg in (("default", HP_DEFAULT), ("tuned", TUNED))]
res = Parallel(n_jobs=N_JOBS)(delayed(_wf_score)(p, cfg) for p, name, cfg in jobs)
transfer = pd.DataFrame([{"pair": f"{p[0]}/{p[1]}", "config": name,
                          "oof_sharpe": r["oof_sharpe"], "wf_sharpe_median": r["wf_sharpe_median"],
                          "wf_trades_total": r["wf_trades_total"]} for (p, name, cfg), r in zip(jobs, res)])
transfer = transfer.pivot(index="pair", columns="config", values=["oof_sharpe", "wf_sharpe_median", "wf_trades_total"])
transfer = transfer.reindex([f"{p[0]}/{p[1]}" for p in [(ticker1, ticker2)] + others])
print("\nPooled out-of-fold Sharpe of the default vs tuned configuration (selected pair first, then runner-ups):")
print(transfer.to_string(float_format="{:.3f}".format))
"""),
md(r"""
### 3.6.4 Pooled out-of-fold equity: the honest in-sample view

The curves in §4 are in-sample in the ordinary sense: the Kalman filter is fitted by EM on the whole
training window and then run causally through it, so each bar's state uses only its own past but the
noise parameters were chosen with the window in hand. The curve below is stricter, and is what the tuning
actually scored: each fold's filter is fitted only on the bars before that fold. Where the two disagree,
this one is the yardstick.
"""),
code(r"""
def _oof_daily(frames, cfg):
    parts = [d for d, _ in (x for x in _oof_slices(frames, {k: cfg[k] for k in SIGNAL_KEYS}) if x is not None)]
    return pd.concat(parts).sort_index()

oof = {name: _oof_daily(fold_frames[_kalman_key(cfg["q"], cfg["em_iters"])], cfg)
       for name, cfg in (("default", HP_DEFAULT), ("tuned", TUNED))}

fig, ax = plt.subplots(figsize=(13, 4.5))
for name, color in (("default", "steelblue"), ("tuned", "darkorange")):
    d = oof[name]
    sh = float(_pooled_sharpe(len(d), d["ret_net"].sum(), (d["ret_net"] ** 2).sum()))
    ax.plot(d.index, d["pnl_net"].cumsum(), color=color, lw=1.5,
            label=f"{name}: pooled OOF Sharpe {sh:.2f}, {int(d['in_pos'].sum())} bars in position")
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Pooled out-of-fold equity — {ticker1}/{ticker2}, each bar from the latest refit (base $10k)")
ax.set_ylabel("Equity ($)"); ax.legend(); plt.tight_layout(); plt.show()
print(f"Out-of-fold bars: {len(oof['default'])}  ({oof['default'].index[0].date()} → {oof['default'].index[-1].date()})")
"""),
]

# ───────────────────────── §4: default + tuned side by side ─────────────────────────
get(29, "# 4. In-Sample Signal Generation").source = r"""
# 4. In-Sample Signal Generation & Evaluation (2020–2025) — default vs tuned

Same steps as nb02, run twice: once with nb02's defaults and once with the tuned configuration.
The tuned Kalman setting needs its own fit on the full training window (nb02's `states_tr`
were fitted with `q=1e-5, em_iters=5`).
"""
get(30, "df_kf = states_tr[(ticker1, ticker2)]").source = r"""
prices = df_filtered.pivot_table(
    index=df_filtered.index.get_level_values('datetime'),
    columns=df_filtered.index.get_level_values('ticker'),
    values='close',
)
pair = (ticker1, ticker2)

# default Kalman states: from §3 (q=1e-5, em_iters=5)
df_pair = pd.DataFrame({'P1': prices[ticker1], 'P2': prices[ticker2]}).join(states_tr[pair][['beta', 'resid']])

# tuned Kalman states: refit on the full training window with the tuned (q, em_iters)
states_tuned, params_tuned = fit_kalman_hedge(
    df_train, pairs=[pair], mode="filter",   # causal, as in nb02
    q=TUNED["q"], em_iters=TUNED["em_iters"], show_progress=False, return_params=True,
)
df_pair_tuned = pd.DataFrame({'P1': prices[ticker1], 'P2': prices[ticker2]}).join(states_tuned[pair][['beta', 'resid']])

print(f"Training pair frame: {df_pair.index[0].date()} → {df_pair.index[-1].date()}, {len(df_pair):,} bars")
print(f"Kalman (default) Q diag: {np.diag(params_tr[pair]['Q'])},  R: {params_tr[pair]['R'].ravel()}")
print(f"Kalman (tuned)   Q diag: {np.diag(params_tuned[pair]['Q'])},  R: {params_tuned[pair]['R'].ravel()}")
df_pair_tuned.tail(3)
"""
get(32, "signals = generate_pair_signals(").source = r"""
from pairs.strategies import generate_pair_signals

SIG_DEFAULT = {k: HP_DEFAULT[k] for k in SIGNAL_KEYS}
SIG_TUNED   = {k: TUNED[k] for k in SIGNAL_KEYS}

signals       = generate_pair_signals(df_pair,       capital_per_pair=10_000, **SIG_DEFAULT)
signals_tuned = generate_pair_signals(df_pair_tuned, capital_per_pair=10_000, **SIG_TUNED)
print(f"in-sample trades — default: {int(signals['entry'].sum())}, tuned: {int(signals_tuned['entry'].sum())}")
signals_tuned.tail(5)
"""
get(33, "4.3 Visualise in-sample trades").source = "## 4.3 Visualise in-sample trades (tuned configuration)"
get(34, "plot_pair_legs_with_trades(").source = r"""
from pairs import plot_pair_legs_with_trades

plot_pair_legs_with_trades(
    df_pair_tuned, signals_tuned,
    label1=ticker1, label2=ticker2,
    normalize=False,
    shade_positions=True,
    size_scale=0.004,
    min_marker=20, max_marker=220,
)
"""
get(36, "daily_tr, trades_tr, summary_tr = evaluate_pair_signals(").source = r"""
from pairs import evaluate_pair_signals

COSTS = dict(cost_bps=1, fee_per_share_1=0.0, fee_per_share_2=0.0,
             borrow_bps_per_year=50, days_per_year=252, bars_per_year=252)

daily_tr,   trades_tr,   summary_tr   = evaluate_pair_signals(df_pair,       signals,       **COSTS)
daily_tr_t, trades_tr_t, summary_tr_t = evaluate_pair_signals(df_pair_tuned, signals_tuned, **COSTS)

print("=== In-Sample Performance (2020-01-02 → 2025-12-31) ===")
print(pd.DataFrame({"default": pd.Series(summary_tr), "tuned": pd.Series(summary_tr_t)}).to_string())
"""
get(37, "Equity curve with drawdown shading").source = r"""
# Equity curves: default vs tuned (in-sample — the tuned curve is optimistic by construction)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(daily_tr.index,   daily_tr["equity"],   color="steelblue", lw=1.5, label="default (nb02)")
ax1.plot(daily_tr_t.index, daily_tr_t["equity"], color="darkorange", lw=1.5, label="tuned (§3.6)")
ax1.set_ylabel("Equity ($)"); ax1.set_title(f"In-Sample Equity Curve — {ticker1}/{ticker2}"); ax1.legend()
ax2.fill_between(daily_tr.index,   daily_tr["drawdown_pct"] * 100,   0, alpha=0.4, color="steelblue", label="default")
ax2.fill_between(daily_tr_t.index, daily_tr_t["drawdown_pct"] * 100, 0, alpha=0.4, color="darkorange", label="tuned")
ax2.set_ylabel("Drawdown (%)"); ax2.set_xlabel("Date"); ax2.legend()
plt.tight_layout(); plt.show()
"""

# ───────────────────────── §4b: OOS default + tuned ─────────────────────────
get(38, "# 4b. Out-of-Sample Evaluation").source = r"""
# 4b. Out-of-Sample Evaluation (2026-01-01 → 2026-06-25) — the only honest comparison

The tuned configuration was chosen on 2020–2025 folds; the default was chosen by hand. Both are now run
on the same 2026 window, which neither has seen. The Kalman filter is continued causally from each
configuration's own end-of-training state.
"""
get(39, "df_test, failed = load_prices('openbb', [ticker1, ticker2]").source = r"""
start_test = "2026-01-01"
end_test   = "2026-06-25"

_oos_cache = CACHE / f"nb04_oos_{ticker1}_{ticker2}.parquet"
if _oos_cache.exists():
    df_test, failed = pd.read_parquet(_oos_cache), []
else:
    df_test, failed = load_prices('openbb', [ticker1, ticker2], start_test, end_test, return_failed=True)
    df_test.to_parquet(_oos_cache)
print(f"OOS data: {len(df_test):,} rows, {df_test.index.get_level_values('ticker').nunique()} tickers")
if failed:
    print("No data for:", failed)
"""
get(41, "states_te, _ = filter_kf_on_new(").source = r"""
from pairs import filter_kf_on_new

P1_new = df_test.loc[(ticker1,), 'close']
P2_new = df_test.loc[(ticker2,), 'close']
prices_te = df_test.pivot_table(
    index=df_test.index.get_level_values("datetime"),
    columns=df_test.index.get_level_values("ticker"),
    values="close",
)

def _continue(params):
    frozen = {k: params[k] for k in ("F", "Q", "R")}
    last_state = (params["last_state_mean"], params["last_state_cov"])
    states_te, _ = filter_kf_on_new(P1_new, P2_new, frozen=frozen, last_state=last_state, init_cov=1e6, mode="filter")
    df_te = states_te.join(prices_te[[ticker1, ticker2]].rename(columns={ticker1: "P1", ticker2: "P2"}), how="inner")
    # Causal residual over the TRAINING window under the same frozen parameters: it sets the z-score
    # look-back and warms it up, so the OOS window never informs its own z-score (nb02's fix).
    states_hist, _ = filter_kf_on_new(df_pair["P1"], df_pair["P2"], frozen=frozen, last_state=None, init_cov=1e6, mode="filter")
    return df_te, states_hist["resid"]

df_pair_te,   z_hist_default = _continue(params_tr[pair])       # default Kalman, continued
df_pair_te_t, z_hist_tuned   = _continue(params_tuned[pair])    # tuned Kalman, continued
print(f"OOS pair frame: {df_pair_te.index[0].date()} → {df_pair_te.index[-1].date()}, {len(df_pair_te):,} bars")
print(f"z-score windows from training residual — default: {estimate_halflife_window(z_hist_default)} bars, "
      f"tuned: {estimate_halflife_window(z_hist_tuned)} bars")
"""
get(42, "signals_te = generate_pair_signals(").source = r"""
signals_te   = generate_pair_signals(df_pair_te,   capital_per_pair=10_000, z_history=z_hist_default,
                                     z_window=estimate_halflife_window(z_hist_default), **SIG_DEFAULT)
signals_te_t = generate_pair_signals(df_pair_te_t, capital_per_pair=10_000, z_history=z_hist_tuned,
                                     z_window=estimate_halflife_window(z_hist_tuned), **SIG_TUNED)

daily_te,   trades_te,   summary_te   = evaluate_pair_signals(df_pair_te[["P1", "P2"]],   signals_te,   **COSTS)
daily_te_t, trades_te_t, summary_te_t = evaluate_pair_signals(df_pair_te_t[["P1", "P2"]], signals_te_t, **COSTS)

print("=== Out-of-Sample Performance (2026-01-01 → 2026-06-25) ===")
print(pd.DataFrame({"default": pd.Series(summary_te), "tuned": pd.Series(summary_te_t)}).to_string())
"""
get(43, "4b.3 Visualise OOS trades").source = "## 4b.3 Visualise OOS trades (tuned configuration)"
get(44, "plot_pair_legs_with_trades(").source = r"""
from pairs import plot_pair_legs_with_trades

plot_pair_legs_with_trades(
    df_pair_te_t, signals_te_t,
    label1=ticker1, label2=ticker2,
    normalize=False,
    shade_positions=True,
    size_scale=0.004,
    min_marker=20, max_marker=220,
)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(daily_te.index,   daily_te["equity"],   color="steelblue",  lw=1.5, label="default (nb02)")
ax.plot(daily_te_t.index, daily_te_t["equity"], color="darkorange", lw=1.5, label="tuned (§3.6)")
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Out-of-sample equity — {ticker1}/{ticker2}, 2026"); ax.set_ylabel("Equity ($)"); ax.legend()
plt.tight_layout(); plt.show()
"""
get(45, "4b.4 In-sample vs OOS comparison").source = r"""
## 4b.4 In-sample vs OOS, default vs tuned

Two things to keep in mind when reading the table:

* The **in-sample** columns use nb02's causally-filtered Kalman states, with the EM noise parameters fitted
  on the whole training window. No bar's signal sees its own future, but the parameters were chosen with
  the window in hand, so these columns still flatter both configurations and can rank them differently
  from the walk-forward folds. The pooled out-of-fold Sharpe and equity curve in §3.6 are the right
  in-sample yardstick; the IS columns are here for continuity with nb02.
* The **OOS** columns are the evidence, but half a year yields a handful of trades: they can show a tuned
  configuration failing, they cannot show it succeeding with any confidence.
"""
get(46, "comparison = pd.DataFrame(").source = r"""
rows = ["sharpe", "ann_return", "max_drawdown_pct", "n_trades", "hit_rate", "profit_factor"]
comparison = pd.DataFrame({
    "IS default": pd.Series(summary_tr),   "IS tuned": pd.Series(summary_tr_t),
    "OOS default": pd.Series(summary_te),  "OOS tuned": pd.Series(summary_te_t),
}).loc[rows]
comparison.loc["ann_return"] = comparison.loc["ann_return"] * 100
comparison.index = ["Sharpe", "Ann. Return (%)", "Max Drawdown (%)", "Trades", "Hit Rate", "Profit Factor"]
print(comparison.to_string(float_format="{:.3f}".format))
"""

# ───────────────────────── §4b.5 what the run found (inserted before §8) ─────────────────────────
results_cell = md(r"""
## 4b.5 What this tuning run actually found

Numbers refer to the run stored in this notebook (deterministic given the cached prices; a fresh
download shifts them slightly). The objective is the pooled out-of-fold Sharpe of §3.6; nb02's per-fold
median is reported alongside.

* **The grid finds "better" configurations easily.** The best pooled out-of-fold Sharpe is 1.38 against
  0.71 for nb02's defaults, which rank around 300th of ~1,950 valid configurations — and the defaults sit
  above the median of every marginal group in §3.6.1, so by the grid's own standards they are a good
  configuration. The winner is a *stiffer and quieter* strategy: `q=1e-6` with no EM (a nearly static
  hedge ratio), rolling rather than robust z-scores, entry at 2.5 and exit at 1.0 — in position for 104
  of the 1,004 out-of-fold bars against 218 for the default, and about a third as many trades (45 vs 149
  on the folds, 26 vs 86 in the full in-sample run). It is the same configuration family that wins under
  nb02's per-fold median (2.02 vs 0.96), so the two objectives agree on the winner.
* **The pooled equity curve (§3.6.4) is the honest in-sample comparison**, and it is far less dramatic
  than the smoothed §4 curves: both configurations make money out of fold, roughly +$3,500 and +$4,400 on
  a $10k base over 2022–2025, the tuned one with fewer, longer holding periods and without the default's
  2025 drawdown.
* **The ranking is not reproducible across fold halves.** Under the pooled objective, configurations
  ranked on the odd folds have a median rank of ~1,190 of ~1,930 on the even folds — no better than chance
  (~965) — and the Spearman correlation between the two rankings is −0.18. The winner itself scores well on
  both halves (about 1.35 and 1.45), but the ordering of the rest of the grid is noise: with two to ten
  trades per 63-bar segment, a pooled Sharpe over eight segments is dominated by a few large days. nb02's
  per-fold median gave a more stable ranking (ρ ≈ 0.44), at the price of ignoring how much of the time a
  configuration is invested.
* **It transfers poorly.** On the runner-up pairs the tuned configuration helps NCLH/TEL (0.20 → 0.60) and
  hurts CCL/EXPE (0.41 → 0.15) and NCLH/SPG (0.61 → −0.01) (§3.6.3).
* **Out of sample it did not help.** On the 2026 hold-out the default made seven trades, six of them
  winners (Sharpe 1.4), while the tuned configuration made two, one of them a loser (Sharpe −0.4). With
  two to seven trades neither number is statistically meaningful, but the direction is the one selection
  bias predicts.

The honest reading: the winner is a genuine description of CCL/STT's 2020–2025 behaviour — a static
hedge, wide entry, early exit — that both objectives pick out, but it is not evidence about the *strategy*:
it does not transfer to neighbouring pairs, it loses on the hold-out, and the rest of the grid's ranking
is noise. What the exercise does establish is *which* knobs matter on this pair (§3.6.1): the exit
threshold dominates (exiting only at the mean, `z_exit=0`, is clearly worse than 0.5 or 1.0), the entry
threshold has a mild optimum near 2.5, rolling z-scores edge out robust ones, a holding cap or a cooldown
costs a little, and the Kalman noise, `em_iters` and `z_stop` barely register. A sensible next step is a
much smaller search over `z_exit` and `z_entry` only, pooled across the shortlist rather than fitted to
one pair.
""")

# ───────────────────────── §8 limitations: add tuning caveats ─────────────────────────
lim = get(47, "Limitations")
lim.source = lim.source.rstrip() + r"""

### Tuning-specific caveats (this notebook)

* **One pair, one search.** The grid was scored on the selected pair's folds. The transfer check in
  §3.6.3 is a sanity check, not a portfolio-level validation; a configuration tuned across the whole
  shortlist (pooled objective) would be the next step.
* **Selection bias is not removed, only exposed.** Thousands of configurations are compared on the same
  16 folds; §3.6.2 shows how the winner sits relative to the bulk and whether rankings agree across
  fold halves, but only the 2026 hold-out is an unbiased read — and half a year is a short one.
* **The pair was selected with the default parameters.** Selection and tuning interact; tuning first
  and selecting afterwards (or jointly) changes which pair is traded.
* **Fixed costs, fixed folds.** Cost assumptions (1 bp, 50 bp/yr borrow) and the fold geometry
  (504/126/63) are not tuned and the results are conditional on them.
"""

# assemble: original cells with §3.6 inserted after cell 28 and the results reading before §8 (cell 47)
new_cells = cells[:29] + tuning_cells + cells[29:47] + [results_cell] + cells[47:]
nb = nbf.v4.new_notebook(cells=new_cells)
nb.metadata = src.metadata
parser = argparse.ArgumentParser(description="Build pairs_trading_04_hyperparameter_tuning_yahoo.ipynb from nb02 (cells only).")
parser.add_argument("--out", type=Path, default=ROOT / "pairs_trading_04_hyperparameter_tuning_yahoo.ipynb",
                    help="output path (default: the notebook under notebooks/)")
args = parser.parse_args()
nbf.write(nb, args.out)
print("wrote", args.out, "cells:", len(new_cells))
