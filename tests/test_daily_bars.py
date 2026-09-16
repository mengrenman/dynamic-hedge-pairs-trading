"""Unit tests for pairs.market_data.daily_bars against a small synthetic day lake in both layouts."""
import numpy as np
import pandas as pd
import pytest

pq = pytest.importorskip("pyarrow.parquet")
import pyarrow as pa

from pairs.market_data.daily_bars import (
    detect_day_lake_layout,
    liquidity_screen,
    load_daily_bars,
    recover_dividends,
)

# A quarter of sessions spanning the DST boundary (2024-11-03), so the UTC offset changes mid-sample
# and there are enough observations to estimate a volatility.
SESSIONS = pd.bdate_range("2024-09-02", "2024-11-29")


def _rows(ticker: str, figi: str, sessions, price0: float, vol: float, div_on=(), div=0.0, vol_ann=0.30):
    """Lake rows for one instrument: local midnight expressed in UTC, split/tr variants and factors."""
    rng = np.random.default_rng(abs(hash(ticker)) % 1000)
    n = len(sessions)
    px = price0 * np.exp(np.cumsum(rng.normal(0, vol_ann / np.sqrt(252), n)))
    # total-return factor: steps up on each ex-date, ends at 1.0
    f = np.ones(n)
    for d in div_on:
        i = list(sessions).index(pd.Timestamp(d))
        f[:i] *= 1.0 - div / px[i - 1]
    utc = pd.DatetimeIndex(sessions).tz_localize("US/Eastern").tz_convert("UTC").tz_localize(None)
    return pd.DataFrame({
        "datetime": utc, "ticker": ticker, "id": figi,
        "close": px * 2.0,                       # raw = twice the split-adjusted price (a 2:1 split)
        "volume": (vol / 2).astype(int) if isinstance(vol, np.ndarray) else int(vol / 2),
        "close_split": px, "volume_split": float(vol), "close_tr": px * f,
        "split_price_factor": 0.5, "tr_price_factor": f,
        "open_split": px, "high_split": px, "low_split": px,
        "open_tr": px * f, "high_tr": px * f, "low_tr": px * f,
    })


@pytest.fixture(scope="module")
def lakes(tmp_path_factory):
    root = tmp_path_factory.mktemp("daylake")
    ticker_root, market_root = root / "ticker", root / "market"
    frames = [
        _rows("AAA", "BBG_AAA", SESSIONS, 100.0, 5e6, div_on=("2024-11-01",), div=1.0),
        _rows("BBB", "BBG_BBB", SESSIONS, 50.0, 5e3),                       # illiquid: ~$250k a day
        _rows("CASH", "BBG_CASH", SESSIONS, 100.0, 9e6, vol_ann=0.01),      # cash-like: fails the vol floor
        _rows("DUP", "BBG_REAL", SESSIONS, 40.0, 3e6),                      # the real DUP …
        _rows("DUP", "NOFIGI_OLD", SESSIONS[:3], 9.0, 400),                 # … and a near-untraded namesake
    ]
    for f in frames:
        t = f["ticker"].iloc[0]
        for (y, m), g in f.groupby([f["datetime"].dt.year, f["datetime"].dt.month]):
            p = ticker_root / t / f"{y:04d}"
            p.mkdir(parents=True, exist_ok=True)
            if t == "DUP" and f["id"].iloc[0] != "BBG_REAL":
                continue                                                     # index lakes carry one instrument
            pq.write_table(pa.Table.from_pandas(g, preserve_index=False), p / f"{m:02d}.parquet")
    market = pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"], kind="mergesort")
    for (y, m), g in market.groupby([market["datetime"].dt.year, market["datetime"].dt.month]):
        p = market_root / f"{y:04d}"
        p.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g, preserve_index=False), p / f"{m:02d}.parquet")
    return ticker_root, market_root


class TestLayout:
    def test_detect(self, lakes, tmp_path):
        ticker_root, market_root = lakes
        assert detect_day_lake_layout(ticker_root) == "ticker"
        assert detect_day_lake_layout(market_root) == "market"
        with pytest.raises(FileNotFoundError):
            detect_day_lake_layout(tmp_path / "missing")

    def test_sessions_are_plain_dates_across_dst(self, lakes):
        _, market_root = lakes
        b = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root)
        got = b.loc["AAA"].index
        assert list(got) == list(SESSIONS)                    # both EDT and EST rows land on their own date
        assert got.tz is None
        assert pd.Timestamp("2024-11-04") in got              # the first session after the clocks change

    def test_date_bounds_and_min_rows(self, lakes):
        _, market_root = lakes
        b = load_daily_bars(["AAA"], "2024-11-01", "2024-11-05", market_root)
        assert b.index.get_level_values("datetime").min() == pd.Timestamp("2024-11-01")
        assert b.index.get_level_values("datetime").max() == pd.Timestamp("2024-11-05")
        few = load_daily_bars(["AAA", "BBB"], "2024-09-02", "2024-11-29", market_root, min_rows=100)
        assert few.empty
        with pytest.raises(ValueError):
            load_daily_bars(["AAA"], "2024-11-08", "2024-11-01", market_root)
        with pytest.raises(ValueError):
            load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root, price="close")

    def test_whole_lake_only_for_market_layout(self, lakes):
        ticker_root, market_root = lakes
        everything = load_daily_bars(None, "2024-09-02", "2024-11-29", market_root)
        assert set(everything.index.get_level_values("ticker")) == {"AAA", "BBB", "CASH", "DUP"}
        with pytest.raises(ValueError):
            load_daily_bars(None, "2024-09-02", "2024-11-29", ticker_root)


class TestPriceBasis:
    def test_split_tr_and_raw(self, lakes):
        _, market_root = lakes
        sp = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root, price="split")["close"]
        tr = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root, price="tr")["close"]
        raw = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root, price="raw")["close"]
        assert np.allclose(raw, sp * 2.0)                       # raw is pre-split
        # the total-return series is scaled DOWN before the ex-date and equal after it: look-ahead
        ex = pd.Timestamp("2024-11-01")
        before = tr.index.get_level_values("datetime") < ex
        assert (tr[before] < sp[before]).all()
        assert np.allclose(tr[~before], sp[~before])

    def test_dollar_volume_uses_the_split_adjusted_price(self, lakes):
        _, market_root = lakes
        for basis in ("split", "tr", "raw"):
            b = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root, price=basis)
            sp = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root)["close"]
            assert np.allclose(b["dollar_volume"], sp * b["volume"])


class TestDividends:
    def test_recover_from_factor_steps(self):
        px = pd.Series([100.0, 100.0, 99.0, 99.0])
        f = pd.Series([0.99, 0.99, 1.0, 1.0])                   # one ex-date on the third bar
        d = recover_dividends(px, f)
        assert np.allclose(d.to_numpy(), [0.0, 0.0, 1.0, 0.0])

    def test_no_dividends_gives_zeros_and_negatives_are_clipped(self):
        px = pd.Series([10.0, 11.0, 12.0])
        assert (recover_dividends(px, pd.Series([1.0, 1.0, 1.0])) == 0).all()
        assert (recover_dividends(px, pd.Series([1.0, 0.9, 0.9])) >= 0).all()
        with pytest.raises(ValueError):
            recover_dividends(px, pd.Series([1.0, 1.0]))

    def test_loader_recovers_the_declared_dividend(self, lakes):
        _, market_root = lakes
        b = load_daily_bars(["AAA", "BBB"], "2024-09-02", "2024-11-29", market_root, with_dividends=True)
        a = b.loc["AAA"]
        paid = a[a["dividend"] > 0]
        assert len(paid) == 1 and paid.index[0] == pd.Timestamp("2024-11-01")
        assert abs(float(paid["dividend"].iloc[0]) - 1.0) < 1e-6
        assert (b.loc["BBB", "dividend"] == 0).all()            # recovery is per ticker, not across the frame


class TestTickerReuse:
    def test_dominant_instrument_is_kept_and_the_clash_reported(self, lakes):
        _, market_root = lakes
        b = load_daily_bars(["DUP"], "2024-09-02", "2024-11-29", market_root)
        assert (b["id"] == "BBG_REAL").all()
        assert (b["close"] > 20).all()                          # the $9 namesake is gone
        clash = pd.DataFrame(b.attrs["ticker_collisions"])
        assert list(clash["ticker"]) == ["DUP"] and list(clash["dropped"]) == ["NOFIGI_OLD"]
        assert clash["dropped_dollar_volume"].iloc[0] < 1e5

    def test_no_collisions_reported_when_there_are_none(self, lakes):
        _, market_root = lakes
        b = load_daily_bars(["AAA"], "2024-09-02", "2024-11-29", market_root)
        assert b.attrs["ticker_collisions"] == []

    def test_result_survives_a_parquet_round_trip(self, lakes, tmp_path):
        _, market_root = lakes
        b = load_daily_bars(["AAA", "DUP"], "2024-09-02", "2024-11-29", market_root, with_dividends=True)
        f = tmp_path / "bars.parquet"
        b.to_parquet(f)                                     # attrs are serialised as JSON: must not be a frame
        pd.testing.assert_frame_equal(pd.read_parquet(f), b)

    def test_layouts_agree_once_reuse_is_resolved(self, lakes):
        ticker_root, market_root = lakes
        names = ["AAA", "BBB", "CASH", "DUP"]
        a = load_daily_bars(names, "2024-09-02", "2024-11-29", ticker_root, with_dividends=True)
        b = load_daily_bars(names, "2024-09-02", "2024-11-29", market_root, with_dividends=True)
        pd.testing.assert_frame_equal(a, b)


class TestLiquidityScreen:
    def test_flags_and_volatility_floor(self, lakes):
        _, market_root = lakes
        bars = load_daily_bars(None, "2024-09-02", "2024-11-29", market_root)
        st = liquidity_screen(bars, min_price=5.0, min_dollar_volume=1e6, min_ann_vol=0.15)
        assert bool(st.loc["AAA", "eligible"])
        assert not bool(st.loc["BBB", "eligible"])              # too little dollar volume
        assert not bool(st.loc["CASH", "eligible"])             # liquid but barely moves
        assert st.loc["CASH", "median_dollar_volume"] > st.loc["AAA", "median_dollar_volume"]
        assert st.loc["CASH", "ann_vol"] < 0.15 < st.loc["AAA", "ann_vol"]
        assert st["median_dollar_volume"].is_monotonic_decreasing

    def test_return_cap_rejects_an_unadjusted_corporate_action(self, lakes):
        _, market_root = lakes
        bars = load_daily_bars(None, "2024-09-02", "2024-11-29", market_root)
        broken = bars.copy()
        idx = ("AAA", bars.loc["AAA"].index[30])
        broken.loc[idx, "close"] = float(broken.loc[idx, "close"]) * 40      # a missed reverse split
        st = liquidity_screen(broken, min_price=5.0, min_dollar_volume=1e6, min_ann_vol=0.15)
        assert not bool(st.loc["AAA", "eligible"])
        assert st.loc["AAA", "max_abs_return"] > 1.0
        loose = liquidity_screen(broken, min_price=5.0, min_dollar_volume=1e6, min_ann_vol=0.15,
                                 max_abs_return=None)
        assert bool(loose.loc["AAA", "eligible"])                            # the gate is what rejects it

    def test_price_gate_uses_the_unadjusted_close(self, lakes):
        # AAA's raw close is twice its split-adjusted close (a 2:1 split), so a $150 floor is met on
        # the raw price and not on the adjusted one — the gate must read the raw column.
        _, market_root = lakes
        bars = load_daily_bars(None, "2024-09-02", "2024-11-29", market_root)
        assert np.allclose(bars.loc["AAA", "raw_close"], bars.loc["AAA", "close"] * 2.0)
        raw_gate = liquidity_screen(bars, min_price=150.0, min_dollar_volume=1e6, min_ann_vol=0.15)
        adj_gate = liquidity_screen(bars, min_price=150.0, min_dollar_volume=1e6, min_ann_vol=0.15,
                                    price_column="close")
        assert bool(raw_gate.loc["AAA", "eligible"])
        assert not bool(adj_gate.loc["AAA", "eligible"])

    def test_explicit_exclusions(self, lakes):
        _, market_root = lakes
        bars = load_daily_bars(None, "2024-09-02", "2024-11-29", market_root)
        st = liquidity_screen(bars, min_price=5.0, min_dollar_volume=1e6, min_ann_vol=0.15,
                              exclude=["AAA"])
        assert not bool(st.loc["AAA", "eligible"])
        assert bool(st.loc["DUP", "eligible"])

    def test_top_n_narrows_the_eligible_set(self, lakes):
        _, market_root = lakes
        bars = load_daily_bars(None, "2024-09-02", "2024-11-29", market_root)
        st = liquidity_screen(bars, min_price=5.0, min_dollar_volume=1e5, min_ann_vol=0.15, top_n=1)
        assert int(st["eligible"].sum()) == 1
        assert st[st["eligible"]].index[0] == "AAA"             # the most traded of those that qualify
