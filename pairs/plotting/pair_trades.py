# pairs/plotting/pair_trades.py
"""
Visualize trades overlaid on each leg's price series for a selected pair.

Exports
-------
- plot_pair_legs_with_trades(df_pair, signals, ...)

Expected inputs
---------------
df_pair : DataFrame indexed by datetime with columns
    - 'P1', 'P2' : aligned leg prices
    (optionally other fields; not required for plotting)

signals : DataFrame indexed by the same datetime with columns
    - 'n1', 'n2' : target share changes (or levels; function diffs to get trades)
    - 'pos'      : current position state (+1 / 0 / -1), optional but enables shading
    - 'entry','exit','stop' : optional bool flags (not explicitly drawn; shading shows holds)

Returns
-------
matplotlib.figure.Figure, (Axes, Axes) -- or (Axes, Axes, Axes) when
``show_zscore=True`` adds the spread's z-score as a third, x-aligned panel.
"""

from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ["plot_pair_legs_with_trades"]


def _spans_from_bool(mask: pd.Series) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (start, end) timestamps where mask is True (contiguous runs)."""
    if mask.empty:
        return iter(())
    m = mask.astype(bool).astype(int)
    dm = m.diff().fillna(0)
    starts = list(mask.index[dm == 1])
    ends   = list(mask.index[dm == -1])
    # If we start inside a True run, prepend the first index
    if m.iloc[0] == 1:
        starts = [mask.index[0]] + starts
    # If we end inside a True run, append the last index
    if len(ends) < len(starts):
        ends = ends + [mask.index[-1]]
    return zip(starts, ends)


def plot_pair_legs_with_trades(
    df_pair: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    label1: str = "P1",
    label2: str = "P2",
    normalize: bool = True,
    base_value: float = 100.0,
    shade_positions: bool = True,
    show_zscore: bool = False,
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    shade_color: str = "0.85",
    shade_alpha: float = 0.25,  # robust, light shading
    size_scale: float = 0.002,
    min_marker: float = 20.0,
    max_marker: float = 220.0,
    buy_color: str = "tab:green",
    sell_color: str = "tab:red",
):
    """
    Plot leg prices with buy/sell markers and (optionally) shaded position spans.

    Notes
    -----
    - Marker area is proportional to trade notional at each event.
    - If `normalize=True`, each leg starts at `base_value` for visual comparability.
    - Shading spans are drawn where `signals['pos'] != 0` (if present).

    Returns
    -------
    fig, (ax1, ax2)
    """
    # --- Basic checks ---
    for col in ("P1", "P2"):
        if col not in df_pair.columns:
            raise ValueError(f"df_pair must contain '{col}' column.")
    # Align on index and bring in any available signal fields
    extra_cols = [c for c in ["pos", "n1", "n2", "entry", "exit", "stop", "z"] if c in signals.columns]
    df = pd.concat([df_pair[["P1", "P2"]], signals[extra_cols]], axis=1).dropna(subset=["P1", "P2"]).copy()

    # --- Compute trade deltas (Δshares) on the plotting index ---
    s = signals.reindex(df.index).copy()
    # If signals contain levels for n1/n2, diff() turns them into trade impulses;
    # if they are already impulses, diff will mostly be 0 except at events, which is fine.
    df["trade1"] = s["n1"].diff().fillna(s["n1"]).where(lambda x: x.abs() > 1e-9, 0.0) if "n1" in s else 0.0
    df["trade2"] = s["n2"].diff().fillna(s["n2"]).where(lambda x: x.abs() > 1e-9, 0.0) if "n2" in s else 0.0

    # --- Normalize prices for display (optional) ---
    if normalize:
        p1_0, p2_0 = float(df["P1"].iloc[0]), float(df["P2"].iloc[0])
        df["P1_plot"] = df["P1"] / p1_0 * base_value
        df["P2_plot"] = df["P2"] / p2_0 * base_value
        ylab = f"Price (base={base_value:.0f})"
    else:
        df["P1_plot"] = df["P1"]
        df["P2_plot"] = df["P2"]
        ylab = "Price"

    # --- Marker sizing by trade notional ---
    df["notional1"] = (pd.Series(df["trade1"]).abs() * df["P1"]).fillna(0.0)
    df["notional2"] = (pd.Series(df["trade2"]).abs() * df["P2"]).fillna(0.0)

    buy1_idx  = df.index[df.get("trade1", pd.Series(index=df.index, dtype=float)) > 0]
    sell1_idx = df.index[df.get("trade1", pd.Series(index=df.index, dtype=float)) < 0]
    buy2_idx  = df.index[df.get("trade2", pd.Series(index=df.index, dtype=float)) > 0]
    sell2_idx = df.index[df.get("trade2", pd.Series(index=df.index, dtype=float)) < 0]

    s1 = np.clip((df.loc[buy1_idx.union(sell1_idx), "notional1"] * size_scale).to_numpy(), min_marker, max_marker)
    s2 = np.clip((df.loc[buy2_idx.union(sell2_idx), "notional2"] * size_scale).to_numpy(), min_marker, max_marker)
    s1_series = pd.Series(s1, index=buy1_idx.union(sell1_idx))
    s2_series = pd.Series(s2, index=buy2_idx.union(sell2_idx))

    # --- Figure & axes ---
    if show_zscore:
        if "z" not in df.columns:
            raise KeyError("show_zscore=True requires a 'z' column in `signals`.")
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(13, 10.5), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 2.0, 1.7]},
        )
        panels = (ax1, ax2, ax3)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        ax3, panels = None, (ax1, ax2)

    # --- Shading of position spans on every axis ---
    if shade_positions and "pos" in df.columns:
        pos_mask = df["pos"].astype(float).fillna(0.0).ne(0)
        for a in panels:
            for start, end in _spans_from_bool(pos_mask):
                a.axvspan(start, end, color=shade_color, alpha=shade_alpha, zorder=0)

    # --- Leg 1 ---
    ax1.plot(df.index, df["P1_plot"], linewidth=1.2, label=label1, zorder=2)
    ax1.scatter(buy1_idx,  df.loc[buy1_idx,  "P1_plot"], marker="^",
                s=s1_series.reindex(buy1_idx).fillna(min_marker), label="Buy n1",
                color=buy_color, zorder=3)
    ax1.scatter(sell1_idx, df.loc[sell1_idx, "P1_plot"], marker="v",
                s=s1_series.reindex(sell1_idx).fillna(min_marker), label="Sell n1",
                color=sell_color, zorder=3)
    ax1.set_ylabel(ylab)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # --- Leg 2 ---
    ax2.plot(df.index, df["P2_plot"], linewidth=1.2, label=label2, linestyle="--", zorder=2)
    ax2.scatter(buy2_idx,  df.loc[buy2_idx,  "P2_plot"], marker="^",
                s=s2_series.reindex(buy2_idx).fillna(min_marker), label="Buy n2",
                color=buy_color, zorder=3)
    ax2.scatter(sell2_idx, df.loc[sell2_idx, "P2_plot"], marker="v",
                s=s2_series.reindex(sell2_idx).fillna(min_marker), label="Sell n2",
                color=sell_color, zorder=3)
    ax2.set_ylabel(ylab)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # --- Spread z-score: the signal the trades above are a consequence of -------
    if ax3 is not None:
        z = df["z"].astype(float)
        ax3.plot(z.index, z, linewidth=1.2, color="tab:blue", label="z of spread", zorder=2)
        ax3.axhline(0.0, color="k", linewidth=0.9, zorder=1)
        for lv, colour, name in ((z_entry, sell_color, "entry"),
                                 (z_exit, buy_color, "exit"),
                                 (z_stop, "0.35", "stop")):
            if lv is None:
                continue
            ax3.axhline(lv, color=colour, linestyle="--", linewidth=1.0,
                        label=f"{name} ±{lv:g}", zorder=1)
            ax3.axhline(-abs(lv), color=colour, linestyle="--", linewidth=1.0, zorder=1)

        # mark where the position opens and closes, so one trade can be followed
        # down the three panels: z leaves the band -> legs are traded -> z reverts
        if "pos" in df.columns:
            pos  = df["pos"].astype(float).fillna(0.0)
            prev = pos.shift(1).fillna(0.0)
            opens  = df.index[(pos.ne(0)) & (prev.eq(0))]
            closes = df.index[(pos.eq(0)) & (prev.ne(0))]
            if len(opens):
                long_spread = opens[pos.reindex(opens) > 0]
                short_spread = opens[pos.reindex(opens) < 0]
                ax3.scatter(long_spread, z.reindex(long_spread), marker="^", s=55,
                            color=buy_color, zorder=4, label="open long spread")
                ax3.scatter(short_spread, z.reindex(short_spread), marker="v", s=55,
                            color=sell_color, zorder=4, label="open short spread")
            if len(closes):
                ax3.scatter(closes, z.reindex(closes), marker="x", s=55, linewidths=1.6,
                            color="0.25", zorder=4, label="close")

        ax3.set_ylabel("z")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper left", ncol=3, fontsize=8)
        ax3.set_title("Spread z-score — entries when it leaves the band, exits as it reverts")

    # with two panels the caption sits under the pair (unchanged); with three it
    # belongs on top, because the z-score panel carries its own title
    legs_title = f"Trades superimposed on each leg • {label1} (top), {label2} (bottom)"
    (ax1 if ax3 is not None else ax2).set_title(legs_title)
    plt.tight_layout()
    return (fig, (ax1, ax2, ax3)) if ax3 is not None else (fig, (ax1, ax2))
