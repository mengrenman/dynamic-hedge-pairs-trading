# tests/test_plotting.py
"""
Tests for plot_pair_legs_with_trades, including the optional z-score panel.

The panel is opt-in on purpose: notebooks 01-03 are hand-written, network-bound
and ship with stored outputs, so the two-panel default must keep returning
exactly two axes with the caption in the same place.  These tests pin that.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pairs import plot_pair_legs_with_trades


@pytest.fixture
def pair_and_signals():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    p2 = 100 + rng.normal(0, 1, n).cumsum()
    p1 = 1.5 * p2 + rng.normal(0, 2, n)
    z = pd.Series(np.sin(np.linspace(0, 12, n)) * 2.6, index=idx)
    # long the spread below -2, short it above +2, hold until flat is forced
    raw = pd.Series(np.where(z > 2, -1.0, np.where(z < -2, 1.0, np.nan)), index=idx)
    pos = raw.ffill().fillna(0.0)
    sig = pd.DataFrame({"n1": pos * 100.0, "n2": -pos * 200.0, "pos": pos, "z": z}, index=idx)
    df = pd.DataFrame({"P1": p1, "P2": p2}, index=idx)
    return df, sig


def test_default_is_unchanged_two_panels(pair_and_signals):
    """The two-panel default must not move: notebooks 01-03 depend on it."""
    df, sig = pair_and_signals
    fig, axes = plot_pair_legs_with_trades(df, sig, label1="AAA", label2="BBB")
    assert len(axes) == 2
    # caption sits under the pair, on the lower axis, as it always has
    assert axes[0].get_title() == ""
    assert "Trades superimposed on each leg" in axes[1].get_title()


def test_zscore_panel_adds_a_third_axis(pair_and_signals):
    df, sig = pair_and_signals
    fig, axes = plot_pair_legs_with_trades(
        df, sig, label1="AAA", label2="BBB",
        show_zscore=True, z_entry=2.0, z_exit=0.5, z_stop=4.0,
    )
    assert len(axes) == 3
    ax3 = axes[2]
    assert ax3.get_ylabel() == "z"
    # with three panels the caption moves to the top and the z panel titles itself
    assert "Trades superimposed on each leg" in axes[0].get_title()
    assert "z-score" in ax3.get_title()


def test_zscore_panel_shares_the_x_axis_with_the_legs(pair_and_signals):
    """The whole point of the panel is that a trade lines up across all three."""
    df, sig = pair_and_signals
    _, (ax1, ax2, ax3) = plot_pair_legs_with_trades(
        df, sig, show_zscore=True, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    assert ax3.get_shared_x_axes().joined(ax1, ax3)
    assert ax3.get_shared_x_axes().joined(ax2, ax3)


def test_bands_are_drawn_symmetrically_for_each_threshold(pair_and_signals):
    df, sig = pair_and_signals
    _, (_, _, ax3) = plot_pair_legs_with_trades(
        df, sig, show_zscore=True, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    ys = sorted({round(float(ln.get_ydata()[0]), 6) for ln in ax3.get_lines()
                 if len(set(ln.get_ydata())) == 1})
    for lv in (2.0, -2.0, 0.5, -0.5, 4.0, -4.0, 0.0):
        assert lv in ys, f"missing the {lv:+g} line; drew {ys}"


def test_thresholds_are_optional(pair_and_signals):
    """Omitting a threshold simply omits its band — it must not raise."""
    df, sig = pair_and_signals
    _, (_, _, ax3) = plot_pair_legs_with_trades(df, sig, show_zscore=True, z_entry=2.0)
    ys = {round(float(ln.get_ydata()[0]), 6) for ln in ax3.get_lines()
          if len(set(ln.get_ydata())) == 1}
    assert 2.0 in ys and -2.0 in ys
    assert 4.0 not in ys and 0.5 not in ys


def test_zscore_panel_requires_a_z_column(pair_and_signals):
    df, sig = pair_and_signals
    with pytest.raises(KeyError, match="z"):
        plot_pair_legs_with_trades(df, sig.drop(columns=["z"]), show_zscore=True)


def test_open_and_close_markers_match_the_position_path(pair_and_signals):
    """Markers must sit on real position transitions, not on every trade bar."""
    df, sig = pair_and_signals
    _, (_, _, ax3) = plot_pair_legs_with_trades(
        df, sig, show_zscore=True, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    pos = sig["pos"]
    prev = pos.shift(1).fillna(0.0)
    n_open = int(((pos != 0) & (prev == 0)).sum())
    n_close = int(((pos == 0) & (prev != 0)).sum())
    plotted = sum(len(c.get_offsets()) for c in ax3.collections)
    assert plotted == n_open + n_close, (
        f"z panel drew {plotted} markers for {n_open} opens and {n_close} closes"
    )
