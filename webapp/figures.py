"""Server-rendered figures, held briefly so an <img> tag can fetch them.

HTMX swaps an HTML fragment containing ``<img src="/fig/{token}">``; the browser then fetches that
token. Rendering into a small LRU keyed by a hash of the request keeps the swapped HTML tiny and
means an unchanged figure is not re-rendered when only the metrics around it move.

Everything reuses ``pairs.plotting`` rather than reimplementing it, so what the app shows and what
the notebooks show cannot drift.
"""
from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from typing import Callable, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pairs import plot_pair_legs_with_trades

_LOCK = threading.Lock()
_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_MAX = 64


def _put(token: str, data: bytes) -> str:
    with _LOCK:
        _CACHE[token] = data
        _CACHE.move_to_end(token)
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return token


def get(token: str) -> Optional[bytes]:
    with _LOCK:
        data = _CACHE.get(token)
        if data is not None:
            _CACHE.move_to_end(token)
        return data


def _token(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _render(fig, dpi: int = 100) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def cached(key: str, make: Callable[[], bytes]) -> str:
    """Render only if this exact figure is not already in the LRU."""
    token = _token(key)
    if get(token) is None:
        _put(token, make())
    return token


# ── the three-panel signal figure ────────────────────────────────────────────

def signals(states: pd.DataFrame, sig_df: pd.DataFrame, t1: str, t2: str,
            z_entry: float, z_exit: float, z_stop: float, key: str) -> str:
    def make() -> bytes:
        fig, axes = plot_pair_legs_with_trades(
            states, sig_df, label1=t1, label2=t2, normalize=False, shade_positions=True,
            size_scale=0.004, min_marker=18, max_marker=190,
            show_zscore=True, z_entry=z_entry, z_exit=z_exit, z_stop=z_stop,
        )
        fig.set_size_inches(11.6, 7.4)
        for ax in axes:
            ax.tick_params(labelsize=9)
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        for ax, n in zip(axes, (1, 1, 2)):
            ax.legend(loc="upper left", bbox_to_anchor=(1.004, 1.0), fontsize=8.5,
                      frameon=False, ncol=n, handlelength=1.5)
        fig.subplots_adjust(left=0.06, right=0.83, top=0.95, bottom=0.06, hspace=0.30)
        return _render(fig, dpi=104)
    return cached(key, make)


# ── equity curve ─────────────────────────────────────────────────────────────

def equity(daily: pd.DataFrame, key: str, title: str = "Equity on the traded window") -> str:
    def make() -> bytes:
        fig, ax = plt.subplots(figsize=(11.6, 2.5))
        ax.plot(daily.index, daily["equity"], lw=1.6, color="steelblue")
        ax.fill_between(daily.index, daily["equity"], 0,
                        where=daily["equity"] >= 0, color="steelblue", alpha=0.12)
        ax.fill_between(daily.index, daily["equity"], 0,
                        where=daily["equity"] < 0, color="indianred", alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("cumulative net P&L ($)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        return _render(fig)
    return cached(key, make)


# ── walk-forward fold distribution: the antidote to a single Sharpe ──────────

def folds(wf: pd.DataFrame, headline: float, key: str) -> str:
    def make() -> bytes:
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 2.7),
                                 gridspec_kw={"width_ratios": [2.1, 1]})
        ax = axes[0]
        colour = ["seagreen" if v > 0 else "indianred" for v in wf["sharpe"]]
        ax.bar(range(len(wf)), wf["sharpe"], color=colour, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(headline, color="steelblue", ls="--", lw=1.5,
                   label=f"whole window = {headline:+.2f}")
        ax.set_xticks(range(len(wf)))
        ax.set_xticklabels([d.strftime("%y-%m") for d in wf["start"]], fontsize=7.5, rotation=45)
        ax.set_ylabel("fold Sharpe", fontsize=9)
        ax.set_title("The same configuration, fold by fold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

        ax = axes[1]
        ax.hist(wf["sharpe"], bins=min(12, max(4, len(wf) // 2)),
                color="slategrey", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(headline, color="steelblue", ls="--", lw=1.5)
        med, pos = wf["sharpe"].median(), (wf["sharpe"] > 0).mean() * 100
        ax.set_title(f"median {med:+.2f} · {pos:.0f}% positive · {len(wf)} folds", fontsize=10)
        ax.tick_params(labelsize=8); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        return _render(fig)
    return cached(key, make)


# ── portfolio equity for the tier-3 page ─────────────────────────────────────

def portfolio(pnl: pd.Series, active: pd.Series, key: str, title: str) -> str:
    def make() -> bytes:
        fig, axes = plt.subplots(2, 1, figsize=(11.6, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.6, 1]})
        axes[0].plot(pnl.index, pnl.cumsum(), lw=1.6, color="steelblue")
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_ylabel("cumulative net P&L ($)", fontsize=9)
        axes[0].set_title(title, fontsize=11)
        axes[0].grid(alpha=0.3); axes[0].tick_params(labelsize=9)
        axes[1].fill_between(active.index, active, 0, color="slategrey", alpha=0.5, step="mid")
        axes[1].set_ylabel("pairs held", fontsize=9)
        axes[1].grid(alpha=0.3); axes[1].tick_params(labelsize=9)
        fig.tight_layout()
        return _render(fig)
    return cached(key, make)
