# tests/test_book.py
"""Tests for pairs.strategies.book — the leg-level book simulator behind the allocation study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs.strategies.book import simulate_pair_book
from pairs.strategies.evaluate import evaluate_pair_signals
from pairs.strategies.signals import generate_pair_signals
from pairs.stats.stationarity import estimate_halflife

CACHE = Path(__file__).resolve().parents[1] / "notebooks" / "cache"
CAP = 10_000.0
DOLLAR_COLS = ["pnl_price", "cost", "borrow", "dividends", "pnl", "traded_notional_1",
               "traded_notional_2", "notional_1", "notional_2", "gross_notional", "net_notional",
               "n1", "n2", "dn1", "dn2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic(seed: int = 0, n: int = 400, beta: float = 1.3):
    """Prices with a mean-reverting spread (so the signal trades), dividends, states, executed signals."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    P2 = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.85 * e[t - 1] + rng.normal(0, 0.8)
    P1 = np.maximum(20 + beta * P2 + e, 1.0)
    px = pd.DataFrame({"P1": P1, "P2": P2}, index=idx)
    div = pd.DataFrame({"D1": 0.0, "D2": 0.0}, index=idx)
    div.iloc[::63, 0] = 0.3
    div.iloc[10::63, 1] = 0.2
    states = px.assign(beta=beta, resid=px["P1"] - 20 - beta * px["P2"])
    sig = generate_pair_signals(states, z_method="rolling", z_window=30, z_entry=1.5, z_exit=0.3,
                                z_stop=4.0, capital_per_pair=CAP)
    return px, div, sig


def _evaluate(px, sig, cost_bps=5.0, borrow=50.0):
    daily, trades, _ = evaluate_pair_signals(px, sig, cost_bps=cost_bps, borrow_bps_per_year=borrow,
                                             days_per_year=252, bars_per_year=252, capital_base=CAP)
    return daily, trades


def _entry_decision_rows(pos: pd.Series) -> np.ndarray:
    """Positions of the decision closes t at which a trade is opened (executed on row t + 1)."""
    p = pos.to_numpy()
    return np.array([t for t in range(len(p) - 1) if p[t + 1] != 0 and p[t] != p[t + 1]])


# ---------------------------------------------------------------------------
# Reproduces generate_pair_signals + evaluate_pair_signals
# ---------------------------------------------------------------------------

class TestMatchesEvaluate:

    @pytest.mark.parametrize("seed,beta", [(0, 1.3), (1, -0.7), (2, 0.4), (3, 2.5)])
    def test_daily_reproduces_evaluate_pnl(self, seed, beta):
        px, div, sig = _synthetic(seed, beta=beta)
        daily, trades = _evaluate(px, sig)
        assert len(trades) >= 5                       # the comparison is not vacuous
        book = simulate_pair_book(px, sig["pos"], beta, CAP, cost_bps=5.0, borrow_bps_per_year=50.0,
                                  dividends=div)
        div_line = sig["n1"] * div["D1"] + sig["n2"] * div["D2"]
        assert div_line.abs().sum() > 0
        np.testing.assert_allclose(book["pnl"] - book["dividends"], daily["pnl_net"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["dividends"], div_line, atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["pnl_price"], daily["pnl_gross"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["cost"] + book["borrow"], daily["cost"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["n1"], sig["n1"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["n2"], sig["n2"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["gross_notional"], daily["gross_exposure"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(book["net_notional"], daily["net_exposure"], atol=1e-9, rtol=0)

    def test_round_trips_and_open_position_match_evaluate(self):
        for seed in range(4):
            px, div, sig = _synthetic(seed)
            daily, trades = _evaluate(px, sig)
            book = simulate_pair_book(px, sig["pos"], 1.3, CAP)
            assert int(book["exit"].sum()) == len(trades)
            assert int(book["entry"].sum()) == len(trades) + int(sig["pos"].iloc[-1] != 0)

    def test_scalar_and_equal_pair_cost_agree(self):
        px, div, sig = _synthetic(0)
        a = simulate_pair_book(px, sig["pos"], 1.3, CAP, cost_bps=3.0)
        b = simulate_pair_book(px, sig["pos"], 1.3, CAP, cost_bps=(3.0, 3.0))
        pd.testing.assert_frame_equal(a, b)

    def test_rows_with_unusable_prices_are_dropped_like_evaluate(self):
        px, div, sig = _synthetic(1)
        px = px.copy()
        px.iloc[[150, 151, 260], 1] = [np.nan, np.nan, -1.0]
        states = px.assign(beta=1.3, resid=px["P1"] - 20 - 1.3 * px["P2"])
        sig = generate_pair_signals(states, z_method="rolling", z_window=30, z_entry=1.5, z_exit=0.3,
                                    z_stop=4.0, capital_per_pair=CAP)
        daily, _ = _evaluate(px, sig)
        book = simulate_pair_book(px, sig["pos"], 1.3, CAP, cost_bps=5.0, borrow_bps_per_year=50.0)
        assert len(book) == len(daily) == len(px) - 3
        np.testing.assert_allclose(book["pnl"], daily["pnl_net"], atol=1e-9, rtol=0)

    def test_daily_reproduces_evaluate_on_real_pair_folds(self):
        rules_f, bars_f = CACHE / "day_rule_distance_top20.parquet", CACHE / "day_market_bars.parquet"
        if not (rules_f.exists() and bars_f.exists()):
            pytest.skip("the day-lake caches are not built (notebooks 09-11)")
        rules = pd.read_parquet(rules_f)
        forms = sorted(rules["formation"].unique())
        n_trades = n_folds = 0
        for k in (8, 20, 33):
            f, nxt = forms[k], forms[k + 1]
            g = rules[rules["formation"] == f].nsmallest(4, "ssd")
            tick = sorted(set(g["ticker1"]) | set(g["ticker2"]))
            bars = pd.read_parquet(bars_f, columns=["close", "dividend"], filters=[("ticker", "in", tick)])
            PX = bars["close"].unstack("ticker")
            DV = bars["dividend"].unstack("ticker").reindex_like(PX).fillna(0.0)
            for a, b in zip(g["ticker1"], g["ticker2"]):
                form = pd.DataFrame({"P1": PX[a], "P2": PX[b]}).loc[
                    (PX.index > f - pd.DateOffset(years=2)) & (PX.index <= f)].dropna()
                trade = pd.DataFrame({"P1": PX[a], "P2": PX[b], "D1": DV[a], "D2": DV[b]}).loc[
                    (PX.index > f) & (PX.index <= nxt)].dropna()
                X = np.column_stack([np.ones(len(form)), form["P2"].to_numpy()])
                (al, be), *_ = np.linalg.lstsq(X, form["P1"].to_numpy(), rcond=None)
                rf = form["P1"] - al - be * form["P2"]
                states = trade[["P1", "P2"]].assign(beta=be, resid=trade["P1"] - al - be * trade["P2"])
                hl = estimate_halflife(rf.dropna())
                win = int(np.clip(3 * hl, 20, 250)) if np.isfinite(hl) else 60
                sig = generate_pair_signals(states, z_method="robust", z_window=win,
                                            z_history=rf.dropna(), z_entry=2.0, z_exit=0.5, z_stop=4.0,
                                            capital_per_pair=CAP)
                daily, trades = _evaluate(states[["P1", "P2"]], sig)
                book = simulate_pair_book(states[["P1", "P2"]], sig["pos"], be, CAP, cost_bps=5.0,
                                          borrow_bps_per_year=50.0, dividends=trade[["D1", "D2"]])
                div_line = sig["n1"] * trade.loc[daily.index, "D1"] + sig["n2"] * trade.loc[daily.index, "D2"]
                np.testing.assert_allclose(book["pnl"], daily["pnl_net"] + div_line, atol=1e-9, rtol=0)
                assert int(book["exit"].sum()) == len(trades)
                n_trades += len(trades)
                n_folds += 1
        assert n_folds == 12 and n_trades >= 10


# ---------------------------------------------------------------------------
# Hand-computed example
# ---------------------------------------------------------------------------

class TestHandComputed:

    def setup_method(self):
        self.idx = pd.bdate_range("2021-03-01", periods=6)
        self.px = pd.DataFrame({"P1": [100.0, 102.0, 101.0, 103.0, 104.0, 100.0],
                                "P2": [50.0, 51.0, 50.0, 52.0, 51.0, 49.0]}, index=self.idx)
        self.pos = pd.Series([0, 1, 1, 0, -1, -1], index=self.idx)
        self.div = pd.DataFrame({"D1": 0.0, "D2": 0.0}, index=self.idx)
        self.div.iloc[2, 1] = 0.5                     # leg two pays 0.5 on row 2
        self.book = simulate_pair_book(self.px, self.pos, 2.0, 1000.0, cost_bps=(10.0, 20.0),
                                       borrow_bps_per_year=252.0, dividends=self.div)

    def test_shares_are_set_at_the_decision_close(self):
        b = self.book
        # row 1: decided at close 0, T / (P1 + |beta| P2) = 1000 / 200 = 5 shares of leg one
        assert b["n1"].iloc[0] == 0.0
        assert b["n1"].iloc[1] == pytest.approx(5.0)
        assert b["n2"].iloc[1] == pytest.approx(-10.0)
        # row 2: reset to the target at close 1: 1000 / (102 + 2 * 51) = 1000 / 204
        assert b["n1"].iloc[2] == pytest.approx(1000 / 204)
        assert b["n2"].iloc[2] == pytest.approx(-2000 / 204)
        # row 3 flat; row 4 short the spread, decided at close 3: 1000 / (103 + 2 * 52) = 1000 / 207
        assert b["n1"].iloc[3] == 0.0 and b["n2"].iloc[3] == 0.0
        assert b["n1"].iloc[4] == pytest.approx(-1000 / 207)
        assert b["n2"].iloc[4] == pytest.approx(2000 / 207)
        # row 5: reset at close 4: 1000 / (104 + 2 * 51) = 1000 / 206
        assert b["n1"].iloc[5] == pytest.approx(-1000 / 206)

    def test_pnl_price_earns_the_move_from_the_decision_close(self):
        b = self.book
        assert b["pnl_price"].iloc[1] == pytest.approx(5.0 * 2.0 + (-10.0) * 1.0)
        assert b["pnl_price"].iloc[2] == pytest.approx((1000 / 204) * (-1.0) + (-2000 / 204) * (-1.0))
        assert b["pnl_price"].iloc[3] == 0.0          # flat on the exit row: nothing held
        assert b["pnl_price"].iloc[5] == pytest.approx((-1000 / 206) * (-4.0) + (2000 / 206) * (-2.0))

    def test_cost_is_charged_leg_by_leg_on_the_booking_row_close(self):
        b = self.book
        # entry on row 1: 5 shares at 102 and 10 shares at 51, 10 bps and 20 bps
        assert b["cost"].iloc[1] == pytest.approx(0.001 * 5 * 102 + 0.002 * 10 * 51)
        # exit on row 3: |dn| x close of row 3
        assert b["cost"].iloc[3] == pytest.approx(0.001 * (1000 / 204) * 103 + 0.002 * (2000 / 204) * 52)
        # row 2: a resize of both legs
        dn1, dn2 = 1000 / 204 - 5.0, -2000 / 204 + 10.0
        assert b["cost"].iloc[2] == pytest.approx(0.001 * abs(dn1) * 101 + 0.002 * abs(dn2) * 50)
        assert b["traded_notional_1"].iloc[2] == pytest.approx(abs(dn1) * 101)

    def test_borrow_accrues_on_short_shares_at_the_row_close(self):
        b = self.book
        rate = 252.0 / 1e4 / 252.0
        assert b["borrow"].iloc[1] == pytest.approx(10.0 * 51 * rate)           # leg two short
        assert b["borrow"].iloc[4] == pytest.approx((1000 / 207) * 104 * rate)  # leg one short
        assert b["borrow"].iloc[0] == 0.0 and b["borrow"].iloc[3] == 0.0

    def test_dividends_are_shares_times_dividend_on_the_ex_row(self):
        b = self.book
        assert b["dividends"].iloc[2] == pytest.approx((-2000 / 204) * 0.5)     # short leg two pays
        assert b["dividends"].drop(self.idx[2]).abs().sum() == 0.0
        np.testing.assert_allclose(b["pnl"], b["pnl_price"] - b["cost"] - b["borrow"] + b["dividends"])

    def test_event_flags_and_trade_ids(self):
        b = self.book
        assert b["entry"].tolist() == [False, True, False, False, True, False]
        assert b["exit"].tolist() == [False, False, False, True, False, False]
        assert b["resize"].tolist() == [False, False, True, False, False, True]
        assert b["trade_id"].tolist() == [0, 1, 1, 0, 2, 2]

    def test_position_open_on_the_last_row_is_never_closed(self):
        assert int(self.book["exit"].sum()) == 1
        assert self.book["pos"].iloc[-1] == -1 and self.book["n1"].iloc[-1] != 0

    def test_exposures(self):
        b = self.book
        assert b["gross_notional"].iloc[1] == pytest.approx(5 * 102 + 10 * 51)
        assert b["net_notional"].iloc[1] == pytest.approx(5 * 102 - 10 * 51)
        np.testing.assert_allclose(b["gross_notional"], b["notional_1"] + b["notional_2"])

    def test_signed_share_changes_reconcile_with_holdings(self):
        b = self.book
        np.testing.assert_allclose(b["dn1"].cumsum(), b["n1"], atol=1e-12)
        np.testing.assert_allclose(b["dn2"].cumsum(), b["n2"], atol=1e-12)

    def test_reversal_is_an_exit_and_an_entry_on_the_same_row(self):
        pos = pd.Series([0, 1, -1, 0, 0, 0], index=self.idx)
        b = simulate_pair_book(self.px, pos, 2.0, 1000.0, cost_bps=10.0)
        assert b["entry"].tolist() == [False, True, True, False, False, False]
        assert b["exit"].tolist() == [False, False, True, True, False, False]
        assert b["trade_id"].tolist() == [0, 1, 2, 0, 0, 0]
        # the reversal row trades out of one position and into the other: |n_new - n_old|
        n_old, n_new = b["n1"].iloc[1], b["n1"].iloc[2]
        assert n_old > 0 > n_new
        assert b["traded_notional_1"].iloc[2] == pytest.approx(abs(n_new - n_old) * 101.0)


# ---------------------------------------------------------------------------
# Rebalancing rules
# ---------------------------------------------------------------------------

class TestFrozen:

    def test_shares_stay_constant_inside_a_trade_and_only_the_ends_are_charged(self):
        px, _, sig = _synthetic(0)
        b = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="frozen", cost_bps=5.0)
        held = b[b["trade_id"] > 0]
        assert held.groupby("trade_id")["n1"].nunique().max() == 1
        assert held.groupby("trade_id")["n2"].nunique().max() == 1
        assert int(b["resize"].sum()) == 0
        assert (b.loc[~(b["entry"] | b["exit"]), "cost"] == 0.0).all()
        assert (b.loc[b["entry"] | b["exit"], "cost"] > 0.0).all()

    def test_entry_is_sized_at_the_target_of_the_entry_decision_close(self):
        px, _, sig = _synthetic(0)
        b = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="frozen")
        rows = _entry_decision_rows(sig["pos"])
        assert len(rows) > 3
        for t in rows:
            side = sig["pos"].iloc[t + 1]
            assert b["n1"].iloc[t + 1] == pytest.approx(side * CAP / (px["P1"].iloc[t] + 1.3 * px["P2"].iloc[t]))

    def test_a_target_that_changes_while_held_is_ignored(self):
        px, _, sig = _synthetic(2)
        rng = np.random.default_rng(5)
        T1 = pd.Series(CAP * (1 + 0.5 * rng.random(len(px))), index=px.index)
        T2 = T1.copy()
        entry_rows = _entry_decision_rows(sig["pos"])
        others = np.setdiff1d(np.arange(len(px)), entry_rows)
        T2.iloc[others] = CAP * (1 + 0.5 * rng.random(len(others)))
        a = simulate_pair_book(px, sig["pos"], 1.3, T1, reset="frozen", cost_bps=5.0)
        b = simulate_pair_book(px, sig["pos"], 1.3, T2, reset="frozen", cost_bps=5.0)
        pd.testing.assert_frame_equal(a, b)
        # under daily the same change moves the shares
        c = simulate_pair_book(px, sig["pos"], 1.3, T1, reset="daily", cost_bps=5.0)
        d = simulate_pair_book(px, sig["pos"], 1.3, T2, reset="daily", cost_bps=5.0)
        assert not np.allclose(c["n1"], d["n1"])

    def test_differs_from_daily_only_by_the_resizes(self):
        px, _, sig = _synthetic(1)
        d = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="daily", cost_bps=5.0)
        f = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="frozen", cost_bps=5.0)
        assert int(d["resize"].sum()) > 0 and int(f["resize"].sum()) == 0
        assert (d["pos"] == f["pos"]).all()
        assert (d["entry"] == f["entry"]).all() and (d["exit"] == f["exit"]).all()
        assert f["cost"].sum() < d["cost"].sum()


class TestBand:

    def test_zero_band_equals_daily(self):
        px, div, sig = _synthetic(0)
        d = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="daily", cost_bps=5.0, dividends=div)
        z = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="band", band=0.0, cost_bps=5.0, dividends=div)
        pd.testing.assert_frame_equal(d[DOLLAR_COLS], z[DOLLAR_COLS], atol=1e-9, rtol=1e-12)

    def test_huge_band_equals_frozen(self):
        px, div, sig = _synthetic(0)
        f = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="frozen", cost_bps=5.0, dividends=div)
        h = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="band", band=1e6, cost_bps=5.0, dividends=div)
        pd.testing.assert_frame_equal(f, h)

    def test_resets_exactly_when_the_held_notional_leaves_the_band(self):
        px, _, sig = _synthetic(0)
        band = 0.02
        b = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset="band", band=band)
        gross = b["gross_notional"].to_numpy()
        pos = sig["pos"].to_numpy()
        n_reset = n_kept = 0
        for i in range(2, len(px)):
            if pos[i] == 0 or pos[i - 1] != pos[i]:
                continue
            dev = abs(gross[i - 1] - CAP) / CAP           # held shares at close i - 1, against the target
            if b["resize"].iloc[i]:
                assert dev > band
                n_reset += 1
            else:
                assert dev <= band
                n_kept += 1
        assert n_reset > 0 and n_kept > 0                 # the band bites in both directions

    def test_intermediate_band_lies_between_frozen_and_daily_on_turnover(self):
        px, _, sig = _synthetic(3)
        turn = {}
        for name, kw in {"daily": dict(reset="daily"), "band": dict(reset="band", band=0.02),
                         "frozen": dict(reset="frozen")}.items():
            b = simulate_pair_book(px, sig["pos"], 1.3, CAP, **kw)
            turn[name] = (b["traded_notional_1"] + b["traded_notional_2"]).sum()
        assert turn["frozen"] < turn["band"] < turn["daily"]


# ---------------------------------------------------------------------------
# Targets, scaling, costs
# ---------------------------------------------------------------------------

class TestTargets:

    def test_scaling_the_target_scales_every_dollar_column(self):
        px, div, sig = _synthetic(0)
        for reset in ("daily", "frozen", "band"):
            a = simulate_pair_book(px, sig["pos"], 1.3, CAP, reset=reset, cost_bps=(4.0, 7.0), dividends=div)
            b = simulate_pair_book(px, sig["pos"], 1.3, 2.5 * CAP, reset=reset, cost_bps=(4.0, 7.0), dividends=div)
            pd.testing.assert_frame_equal(2.5 * a[DOLLAR_COLS], b[DOLLAR_COLS], rtol=1e-10, atol=1e-9)
            for c in ("pos", "entry", "exit", "resize", "trade_id"):
                assert (a[c] == b[c]).all()

    def test_a_per_row_target_sizes_the_next_row_at_the_decision_close(self):
        px, _, sig = _synthetic(0)
        T = pd.Series(np.linspace(5_000, 20_000, len(px)), index=px.index)
        b = simulate_pair_book(px, sig["pos"], 1.3, T, reset="daily")
        pos = sig["pos"].to_numpy()
        checked = 0
        for i in range(1, len(px)):
            if pos[i] != 0:
                t = i - 1
                assert b["gross_notional"].iloc[i] / px["P1"].iloc[i] > 0
                assert b["n1"].iloc[i] == pytest.approx(pos[i] * T.iloc[t] / (px["P1"].iloc[t] + 1.3 * px["P2"].iloc[t]))
                assert b["n2"].iloc[i] == pytest.approx(-1.3 * b["n1"].iloc[i])
                checked += 1
        assert checked > 20

    def test_gross_notional_at_the_sizing_close_equals_the_target(self):
        px, _, sig = _synthetic(1)
        for beta in (1.3, -0.6):
            b = simulate_pair_book(px, sig["pos"], beta, 12_345.0)
            pos = sig["pos"].to_numpy()
            for i in np.flatnonzero(pos != 0):
                t = i - 1
                gross_at_decision = abs(b["n1"].iloc[i]) * px["P1"].iloc[t] + abs(b["n2"].iloc[i]) * px["P2"].iloc[t]
                assert gross_at_decision == pytest.approx(12_345.0)

    def test_negative_beta_holds_both_legs_on_the_same_side(self):
        px, _, sig = _synthetic(1, beta=-0.7)
        b = simulate_pair_book(px, sig["pos"], -0.7, CAP)
        held = b[b["pos"] != 0]
        assert (np.sign(held["n1"]) == np.sign(held["n2"])).all()

    def test_a_zero_target_holds_nothing_and_costs_nothing(self):
        px, _, sig = _synthetic(0)
        b = simulate_pair_book(px, sig["pos"], 1.3, 0.0, cost_bps=5.0)
        assert (b[DOLLAR_COLS] == 0.0).all().all()


class TestCosts:

    def test_per_leg_cost_rates(self):
        px, _, sig = _synthetic(0)
        b = simulate_pair_book(px, sig["pos"], 1.3, CAP, cost_bps=(2.0, 9.0))
        expected = (2.0 * b["traded_notional_1"] + 9.0 * b["traded_notional_2"]) / 1e4
        np.testing.assert_allclose(b["cost"], expected, atol=1e-12)
        only1 = simulate_pair_book(px, sig["pos"], 1.3, CAP, cost_bps=(2.0, 0.0))
        only2 = simulate_pair_book(px, sig["pos"], 1.3, CAP, cost_bps=(0.0, 9.0))
        np.testing.assert_allclose(only1["cost"] + only2["cost"], b["cost"], atol=1e-12)

    def test_cost_is_zero_by_default_and_borrow_is_not(self):
        px, _, sig = _synthetic(0)
        b = simulate_pair_book(px, sig["pos"], 1.3, CAP)
        assert b["cost"].sum() == 0.0 and b["borrow"].sum() > 0.0
        nb = simulate_pair_book(px, sig["pos"], 1.3, CAP, borrow_bps_per_year=0.0)
        assert nb["borrow"].sum() == 0.0

    def test_no_dividends_means_a_zero_dividend_column(self):
        px, _, sig = _synthetic(0)
        assert (simulate_pair_book(px, sig["pos"], 1.3, CAP)["dividends"] == 0.0).all()

    def test_dividends_may_cover_more_rows_than_the_ledger(self):
        px, div, sig = _synthetic(0)
        extra = pd.DataFrame({"D1": 9.0, "D2": 9.0}, index=pd.bdate_range(px.index[-1] + pd.offsets.BDay(1), periods=5))
        wide = simulate_pair_book(px, sig["pos"], 1.3, CAP, dividends=pd.concat([div, extra]))
        pd.testing.assert_frame_equal(wide, simulate_pair_book(px, sig["pos"], 1.3, CAP, dividends=div))


# ---------------------------------------------------------------------------
# Validation and edge cases
# ---------------------------------------------------------------------------

class TestValidation:

    def setup_method(self):
        self.px, self.div, self.sig = _synthetic(0, n=120)
        self.pos = self.sig["pos"]

    def test_unknown_reset_raises(self):
        with pytest.raises(ValueError, match="reset"):
            simulate_pair_book(self.px, self.pos, 1.3, CAP, reset="weekly")

    @pytest.mark.parametrize("band", [-0.1, np.nan, np.inf])
    def test_bad_band_raises(self, band):
        with pytest.raises(ValueError, match="band"):
            simulate_pair_book(self.px, self.pos, 1.3, CAP, reset="band", band=band)

    @pytest.mark.parametrize("cost", [-1.0, (1.0, 2.0, 3.0), (np.nan, 1.0)])
    def test_bad_cost_raises(self, cost):
        with pytest.raises(ValueError, match="cost_bps"):
            simulate_pair_book(self.px, self.pos, 1.3, CAP, cost_bps=cost)

    def test_first_row_must_be_flat(self):
        pos = self.pos.copy()
        pos.iloc[0] = 1
        with pytest.raises(ValueError, match="pos"):
            simulate_pair_book(self.px, pos, 1.3, CAP)

    def test_pos_outside_the_three_states_raises(self):
        pos = self.pos.copy().astype(float)
        pos.iloc[10] = 2.0
        with pytest.raises(ValueError, match="pos"):
            simulate_pair_book(self.px, pos, 1.3, CAP)

    def test_misaligned_series_raise(self):
        with pytest.raises(ValueError, match="pos"):
            simulate_pair_book(self.px, self.pos.iloc[:-1], 1.3, CAP)
        with pytest.raises(ValueError, match="target_notional"):
            simulate_pair_book(self.px, self.pos, 1.3, pd.Series(CAP, index=self.px.index[1:]))
        with pytest.raises(ValueError, match="beta"):
            simulate_pair_book(self.px, self.pos, np.ones(len(self.px) - 2), CAP)

    def test_missing_price_columns_raise(self):
        with pytest.raises(ValueError, match="P1"):
            simulate_pair_book(self.px[["P1"]], self.pos, 1.3, CAP)

    def test_unsorted_or_duplicated_index_raises(self):
        with pytest.raises(ValueError, match="index"):
            simulate_pair_book(self.px.iloc[::-1], self.pos.iloc[::-1], 1.3, CAP)

    def test_target_missing_on_a_decision_close_of_a_held_row_raises(self):
        T = pd.Series(CAP, index=self.px.index)
        t = int(np.flatnonzero(self.pos.to_numpy() != 0)[0]) - 1
        T.iloc[t] = np.nan
        with pytest.raises(ValueError, match="cannot size"):
            simulate_pair_book(self.px, self.pos, 1.3, T)

    def test_a_missing_target_where_nothing_is_held_is_fine(self):
        T = pd.Series(np.nan, index=self.px.index)
        held_next = np.flatnonzero(self.pos.to_numpy() != 0) - 1
        T.iloc[held_next] = CAP
        b = simulate_pair_book(self.px, self.pos, 1.3, T, reset="daily")
        ref = simulate_pair_book(self.px, self.pos, 1.3, CAP, reset="daily")
        pd.testing.assert_frame_equal(b, ref)

    def test_dividends_must_cover_the_ledger(self):
        with pytest.raises(ValueError, match="dividends"):
            simulate_pair_book(self.px, self.pos, 1.3, CAP, dividends=self.div.iloc[5:])
        with pytest.raises(ValueError, match="dividends"):
            simulate_pair_book(self.px, self.pos, 1.3, CAP, dividends=self.div[["D1"]])

    def test_empty_ledger_returns_an_empty_frame_with_the_columns(self):
        b = simulate_pair_book(self.px.iloc[:0], self.pos.iloc[:0], 1.3, CAP)
        assert b.empty and "pnl" in b.columns

    def test_a_flat_path_produces_nothing(self):
        b = simulate_pair_book(self.px, pd.Series(0, index=self.px.index), 1.3, CAP, cost_bps=5.0)
        assert (b[DOLLAR_COLS] == 0.0).all().all() and not b["entry"].any() and not b["exit"].any()
