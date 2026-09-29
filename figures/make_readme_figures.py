"""Regenerate the two figures at the top of the README.

    python figures/make_readme_figures.py

Both come from notebook 01's **out-of-sample** window: the pair its walk-forward driver selected on
2020-2025 training data, traded through 2026 with everything frozen. The pipeline here mirrors that
notebook cell for cell, and the script asserts the resulting summary matches the numbers the
notebook prints -- so the shop window cannot drift away from the work behind it.

The notebook's own inline figure is not reused directly: at the size the README displays it, a
1280x1040 three-panel plot is unreadable and its legends sit on top of the price lines. This renders
the same data wider, with the legends moved clear of the data and larger type.

Reads two gitignored caches written by the notebook:
    cache/nb01_train_2020-01-01_2025-12-31.parquet
    cache/nb01_oos_BKNG_MA_2026-01-01_2026-06-25.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pairs import (fit_kalman_hedge, filter_kf_on_new, generate_pair_signals,   # noqa: E402
                   evaluate_pair_signals, estimate_halflife_window,
                   plot_pair_legs_with_trades)

CACHE = ROOT / "notebooks" / "cache"
OUT = ROOT / "figures"
T1, T2 = "BKNG", "MA"

# notebook 01's settings, copied verbatim
Q, EM_ITERS = 1e-5, 5
Z_ENTRY, Z_EXIT, Z_STOP, CAP = 2.0, 0.5, 4.0, 10_000
COST_BPS, BORROW_BPS = 1, 50

# what notebook 01 prints for this window; the script refuses to write a figure that disagrees
EXPECT = {"bars": 120, "n_trades": 12, "sharpe": 2.64203, "net_pnl": 664.302195}


def _prices(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.pivot_table(index=df.index.get_level_values("datetime"),
                          columns=df.index.get_level_values("ticker"), values="close")


def build():
    tr = _prices(CACHE / "nb01_train_2020-01-01_2025-12-31.parquet")
    te = _prices(CACHE / f"nb01_oos_{T1}_{T2}_2026-01-01_2026-06-25.parquet")

    df_train = (pd.concat({t: tr[t] for t in (T1, T2)}, names=["ticker", "datetime"])
                  .rename("close").to_frame())
    states_tr, params_tr = fit_kalman_hedge(df_train, pairs=[(T1, T2)], mode="filter",
                                            em_iters=EM_ITERS, q=Q, show_progress=False,
                                            return_params=True)
    p = params_tr[(T1, T2)]
    frozen = {k: p[k] for k in ("F", "Q", "R")}

    # continue the filter onto the hold-out from the last training state (notebook 01 §4b)
    states_te, _ = filter_kf_on_new(te[T1], te[T2], frozen=frozen,
                                    last_state=(p["last_state_mean"], p["last_state_cov"]),
                                    init_cov=1e6, mode="filter")
    df_pair_te = states_te.join(te[[T1, T2]].rename(columns={T1: "P1", T2: "P2"}), how="inner")

    # warm the robust z-score up on the training residual under the same frozen filter, so the
    # hold-out never informs its own z-score and does not waste half its bars warming up
    hist, _ = filter_kf_on_new(tr[T1], tr[T2], frozen=frozen, last_state=None,
                               init_cov=1e6, mode="filter")
    z_window = estimate_halflife_window(hist["resid"])

    sig = generate_pair_signals(df_pair_te, z_method="robust", z_window=z_window,
                                z_history=hist["resid"], z_entry=Z_ENTRY, z_exit=Z_EXIT,
                                z_stop=Z_STOP, capital_per_pair=CAP)
    _, _, summary = evaluate_pair_signals(df_pair_te, sig, cost_bps=COST_BPS,
                                          borrow_bps_per_year=BORROW_BPS,
                                          days_per_year=252, bars_per_year=252)
    for k, want in EXPECT.items():
        got = float(summary[k])
        if abs(got - want) > max(1e-4, abs(want) * 1e-5):
            raise SystemExit(f"summary[{k}] = {got} but notebook 01 prints {want}. "
                             f"The figure would not match the notebook; refusing to write.")
    return df_pair_te, sig, summary, z_window


def figure_signals(df_pair_te, sig, z_window):
    """The hero image. Sized and typed for a README rendering it about 900 px wide."""
    fig, axes = plot_pair_legs_with_trades(
        df_pair_te, sig, label1=T1, label2=T2, normalize=False, shade_positions=True,
        size_scale=0.004, min_marker=26, max_marker=240,
        show_zscore=True, z_entry=Z_ENTRY, z_exit=Z_EXIT, z_stop=Z_STOP,
    )
    fig.set_size_inches(13.4, 7.8)
    for ax in axes:
        ax.tick_params(labelsize=13)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        for ln in ax.get_lines():
            if ln.get_linestyle() == "-" and ln.get_linewidth() < 2:
                ln.set_linewidth(1.9)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(1.004, 1.0), fontsize=11.5,
                  frameon=False, ncol=1, handlelength=1.6, labelspacing=0.45)
    axes[0].set_title(f"{T1} / {T2} — out of sample, 2026    "
                      "(shaded = in a position)", fontsize=16, pad=12)
    axes[2].set_title("Spread z-score: entries when it leaves the band, exits as it reverts",
                      fontsize=13.5, pad=8)
    for t in axes[2].texts:
        t.set_fontsize(10.5)
    # bbox_inches="tight" trims the canvas to the content and then adds pad_inches, so the clear
    # margin on the right is set by the padding alone -- shrinking the axes with `right` only
    # squeezes the plots and moves the legends with them. The z panel's "open short spread" is the
    # widest label in the figure and therefore the one that decides how tight the edge looks, so
    # the padding is generous rather than cosmetic.
    fig.subplots_adjust(left=0.062, right=0.79, top=0.935, bottom=0.075, hspace=0.34)
    fig.savefig(OUT / "signals.png", dpi=135, facecolor="white", bbox_inches="tight",
                pad_inches=0.62)
    plt.close(fig)


def figure_evaluation(summary, z_window):
    """A wide strip of headline hold-out numbers, to sit under the plot at the same width."""
    s = summary
    # every literal $ must be escaped or matplotlib reads the span between two of them as mathtext
    tiles = [("Sharpe",        f"{s['sharpe']:.2f}",                   True),
             ("Net P&L",       f"\\${s['net_pnl']:,.0f}",                True),
             ("Ann. return",   f"{s['ann_return']*100:.1f}%",          False),
             ("Ann. vol",      f"{s['ann_vol']*100:.1f}%",             False),
             ("Max drawdown",  f"{s['max_drawdown_pct']*100:.1f}%",    False),
             ("Hit rate",      f"{s['hit_rate']*100:.0f}%",            False),
             ("Profit factor", f"{s['profit_factor']:.2f}",            False),
             ("Trades",        f"{int(s['n_trades'])}",                False)]

    fig, ax = plt.subplots(figsize=(13.4, 2.78))
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.955, "Out-of-sample evaluation", fontsize=16, weight="bold", va="top")
    ax.text(0.0, 0.79,
            f"{T1} / {T2}   ·   {pd.Timestamp(s['start']):%d %b %Y} → "
            f"{pd.Timestamp(s['end']):%d %b %Y}   ·   {int(s['bars'])} bars   ·   "
            f"\\${s['capital_base']:,.0f} capital",
            fontsize=12.5, color="0.35", va="top")

    for i, (lab, val, strong) in enumerate(tiles):
        x = i / len(tiles) + 0.5 / len(tiles)
        ax.text(x, 0.535, val, fontsize=25 if strong else 21, ha="center", va="center",
                weight="bold", color="#14532d" if strong else "0.12")
        ax.text(x, 0.29, lab, fontsize=11.5, ha="center", va="center", color="0.42")
    for i in range(1, len(tiles)):
        ax.axvline(i / len(tiles), 0.22, 0.67, color="0.88", lw=1)

    ax.text(0.0, 0.135,
            f"Pair chosen by walk-forward stability on 2020–2025; hedge, thresholds and the "
            f"{z_window}-bar z-window frozen before 2026 is touched. Costs 1 bp a leg-side plus "
            "50 bp/yr borrow.\n"
            f"{int(s['n_trades'])} trades is a small sample — notebook 14 is about why a window "
            "this short settles little.",
            fontsize=10.5, color="0.5", va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.99, bottom=0.01)
    fig.savefig(OUT / "backtest.png", dpi=135, facecolor="white", bbox_inches="tight",
                pad_inches=0.14)
    plt.close(fig)


if __name__ == "__main__":
    df_pair_te, sig, summary, z_window = build()
    print(f"{T1}/{T2} out of sample: {int(summary['bars'])} bars, "
          f"{int(summary['n_trades'])} trades, Sharpe {summary['sharpe']:.3f}, "
          f"net P&L ${summary['net_pnl']:,.0f}  (matches notebook 01)")
    figure_signals(df_pair_te, sig, z_window)
    figure_evaluation(summary, z_window)
    for f in ("signals.png", "backtest.png"):
        print(f"  wrote figures/{f}  ({(OUT / f).stat().st_size // 1024} KB)")
