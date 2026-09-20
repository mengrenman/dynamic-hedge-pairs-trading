"""Tests for the measured-cost pipeline.

These run on a synthetic minute lake rather than the real one, so they check the arithmetic and
the edges rather than any particular number: that the full Roll spread is halved on the way out,
that an estimate below the minimum tick is floored, that an unmeasurable cell is reported instead
of silently dropped, and that a pair-fold cost is the mean of its two legs.
"""
import numpy as np
import pandas as pd
import pytest

from pairs.market_data.execution_costs import (CostSpec, measure_ticker_window_costs,
                                               pair_fold_costs)

FORM = pd.Timestamp("2020-01-01")
LO, HI = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")
SPEC = CostSpec(every=5, min_obs=100, min_bars=500)


def _bars(spreads, *, sessions=40, bars=390, price=100.0):
    """A minute lake in which each ticker has a known constant half-spread, in bps."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2020-01-02", periods=sessions)
    idx = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=bars, freq="min")
         for d in days]))
    frames = []
    for t, half_bps in spreads.items():
        mid = price * np.exp(np.cumsum(rng.normal(0, 2e-4, len(idx))))
        px = mid * (1 + half_bps / 1e4 * rng.choice([-1.0, 1.0], len(idx)))
        frames.append(pd.DataFrame({"close": px},
                                   index=pd.MultiIndex.from_product([[t], idx],
                                                                    names=["ticker", "datetime"])))
    return pd.concat(frames)


def _loader(frame):
    def load(tickers, lo, hi, root, **kw):
        have = [t for t in tickers if t in set(frame.index.get_level_values("ticker"))]
        return frame.loc[have] if have else frame.iloc[:0]
    return load


def _raw(tickers, price=100.0):
    idx = pd.bdate_range("2020-01-02", periods=40)
    return pd.DataFrame({t: price for t in tickers}, index=idx, dtype=float)


class TestMeasurement:
    def test_recovers_a_known_half_spread_and_does_not_double_it(self):
        cells = measure_ticker_window_costs(
            {FORM: (LO, HI)}, {FORM: ["WIDE", "TIGHT"]}, "/nowhere",
            raw_close=_raw(["WIDE", "TIGHT"]), spec=SPEC, n_jobs=1,
            loader=_loader(_bars({"WIDE": 6.0, "TIGHT": 2.0})))
        got = cells.set_index("ticker")["cost_bps"]
        # the input is the *half* spread, which is what a transaction pays, so the function must
        # return it directly rather than the doubled Roll number
        assert abs(got["WIDE"] - 6.0) < 0.6
        assert abs(got["TIGHT"] - 2.0) < 0.6
        assert cells["err"].eq("").all()

    def test_an_estimate_below_the_minimum_tick_is_floored(self):
        """A half-spread of 0.5 bps on a $5 stock is below half a cent: impossible, not cheap."""
        cells = measure_ticker_window_costs(
            {FORM: (LO, HI)}, {FORM: ["CHEAP"]}, "/nowhere",
            raw_close=_raw(["CHEAP"], price=5.0), spec=SPEC, n_jobs=1,
            loader=_loader(_bars({"CHEAP": 0.5}, price=5.0)))
        r = cells.iloc[0]
        assert abs(r["tick_floor"] - 10.0) < 1e-9        # half a cent on $5
        assert r["cost_bps"] < r["tick_floor"]
        assert r["cost_used"] == pytest.approx(r["tick_floor"])

    def test_without_raw_prices_there_is_no_floor_and_the_estimate_stands(self):
        cells = measure_ticker_window_costs(
            {FORM: (LO, HI)}, {FORM: ["A"]}, "/nowhere", spec=SPEC, n_jobs=1,
            loader=_loader(_bars({"A": 3.0})))
        r = cells.iloc[0]
        assert np.isnan(r["tick_floor"]) and r["cost_used"] == pytest.approx(r["cost_bps"])

    @pytest.mark.parametrize("why, frame", [
        ("absent from the minute lake", _bars({"OTHER": 3.0})),
        ("only", _bars({"A": 3.0}, sessions=1, bars=100)),
    ])
    def test_an_unmeasurable_cell_is_reported_not_dropped(self, why, frame):
        cells = measure_ticker_window_costs(
            {FORM: (LO, HI)}, {FORM: ["A"]}, "/nowhere", spec=SPEC, n_jobs=1,
            loader=_loader(frame))
        assert len(cells) == 1 and why in cells.iloc[0]["err"]
        assert pd.isna(cells.iloc[0]["cost_used"])

    def test_a_loader_failure_becomes_a_row_rather_than_a_crash(self):
        def boom(*a, **k):
            raise OSError("the lake is on fire")
        cells = measure_ticker_window_costs({FORM: (LO, HI)}, {FORM: ["A", "B"]}, "/nowhere",
                                            spec=SPEC, n_jobs=1, loader=boom)
        assert len(cells) == 2 and cells["err"].str.contains("OSError").all()

    def test_an_empty_request_returns_the_schema_not_an_error(self):
        cells = measure_ticker_window_costs({}, {}, "/nowhere", spec=SPEC, n_jobs=1,
                                            loader=_loader(_bars({"A": 3.0})))
        assert list(cells.columns) == ["formation", "ticker", "raw_px", "tick_floor", "bars",
                                       "cost_bps", "cost_used", "err"]
        assert pair_fold_costs(cells, {}).empty


class TestPairFolds:
    @pytest.fixture
    def cells(self):
        return pd.DataFrame({
            "formation": [FORM] * 3, "ticker": ["A", "B", "C"],
            "cost_used": [2.0, 4.0, np.nan], "err": ["", "", "absent from the minute lake"]})

    def test_a_pair_pays_the_mean_of_its_two_legs(self, cells):
        pf = pair_fold_costs(cells, {FORM: [("A", "B")]})
        assert pf.iloc[0]["cost_bps"] == pytest.approx(3.0)

    def test_one_measurable_leg_is_better_than_none(self, cells):
        pf = pair_fold_costs(cells, {FORM: [("A", "C")]}).iloc[0]
        assert pf["cost_bps"] == pytest.approx(2.0) and np.isnan(pf["c2"])

    def test_neither_leg_measurable_stays_nan_unless_a_fallback_is_asked_for(self, cells):
        sel = {FORM: [("C", "D")]}
        assert np.isnan(pair_fold_costs(cells, sel).iloc[0]["cost_bps"])
        assert pair_fold_costs(cells, sel, fallback=5.0).iloc[0]["cost_bps"] == 5.0
