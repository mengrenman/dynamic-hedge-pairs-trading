"""Unit tests for pairs.market_data.minute_bars against a small synthetic lake in both layouts."""
import numpy as np
import pandas as pd
import pytest

pq = pytest.importorskip("pyarrow.parquet")
import pyarrow as pa

from pairs.market_data.minute_bars import (
    detect_lake_layout,
    load_minute_bars,
    nyse_early_closes,
    summarize_sessions,
    _read_market_day,
)

# Four sessions: an EDT day, an EST day, an early-close day (13:00) and another EST day.
DAYS = ["2024-07-01", "2024-11-27", "2024-11-29", "2024-12-02"]
UTC_OFFSET = {"2024-07-01": 4, "2024-11-27": 5, "2024-11-29": 5, "2024-12-02": 5}   # hours behind UTC
N_BARS = {"2024-07-01": 390, "2024-11-27": 390, "2024-11-29": 210, "2024-12-02": 390}


def _ticker_day(ticker: str, day: str, seed: int) -> pd.DataFrame:
    """Raw lake rows for one ticker-day, in UTC-naive timestamps, with pre/post-market bars."""
    rng = np.random.default_rng(seed)
    off = UTC_OFFSET[day]
    open_utc = pd.Timestamp(day) + pd.Timedelta(hours=9 + off, minutes=30)
    minutes = pd.date_range(open_utc, periods=N_BARS[day], freq="min")
    px = 100.0 + np.cumsum(rng.normal(0, 0.05, len(minutes)))
    rows = pd.DataFrame({"datetime": minutes, "close_tr": px, "close": px * 1.1, "volume": 1000})
    if ticker == "AAA":
        # gaps: no trades in minutes 3-5 of every session; on 2024-12-02 the first trade is at 09:32
        rows = rows.drop(index=[3, 4, 5])
        if day == "2024-12-02":
            rows = rows.iloc[2:]
    pre = pd.DataFrame({"datetime": [open_utc - pd.Timedelta(hours=1)], "close_tr": [1.0], "close": [1.0], "volume": [5]})
    post = pd.DataFrame({"datetime": [minutes[-1] + pd.Timedelta(minutes=30)], "close_tr": [1.0], "close": [1.0], "volume": [5]})
    out = pd.concat([pre, rows, post], ignore_index=True)
    out.insert(1, "ticker", ticker)
    return out


@pytest.fixture(scope="module")
def lakes(tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    ticker_root, market_root = root / "ticker", root / "market"
    tickers = ["AAA", "BBB"]
    for day in DAYS:
        y, m, d = day.split("-")
        frames = []
        for i, t in enumerate(tickers):
            if t == "BBB" and day == "2024-11-29":
                continue                                    # BBB has no rows at all on the early-close day
            f = _ticker_day(t, day, seed=hash((t, day)) % 1000)
            frames.append(f)
            p = ticker_root / t / y / m
            p.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(f, preserve_index=False), p / f"{d}.parquet")
        market = pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"], kind="mergesort").reset_index(drop=True)
        p = market_root / y / m
        p.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(market, preserve_index=False), p / f"{d}.parquet", row_group_size=64)
        starts = market.groupby("ticker").apply(lambda g: g.index.min())
        ends = market.groupby("ticker").apply(lambda g: g.index.max())
        idx = pd.DataFrame({"ticker": starts.index, "n_rows": (ends - starts + 1).to_numpy(),
                            "row_start": starts.to_numpy(), "row_end": ends.to_numpy()})
        pq.write_table(pa.Table.from_pandas(idx, preserve_index=False), p / f"{d}.idx.parquet")
    return ticker_root, market_root


class TestLayoutAndCalendar:
    def test_detect_layout(self, lakes, tmp_path):
        ticker_root, market_root = lakes
        assert detect_lake_layout(ticker_root) == "ticker"
        assert detect_lake_layout(market_root) == "market"
        with pytest.raises(FileNotFoundError):
            detect_lake_layout(tmp_path / "nope")

    def test_early_closes_rule(self):
        got = [d.date().isoformat() for d in nyse_early_closes("2022-01-01", "2025-12-31")]
        assert got == ["2022-11-25", "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29",
                       "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"]
        # 2021: Dec 24 was a Friday (observed holiday) -> not an early close; July 3 was a Saturday
        assert [d.date().isoformat() for d in nyse_early_closes("2021-01-01", "2021-12-31")] == ["2021-11-26"]


class TestMinuteGrid:
    def test_sessions_timezone_and_regular_hours(self, lakes):
        ticker_root, _ = lakes
        df = load_minute_bars(["AAA", "BBB"], "2024-07-01", "2024-12-02", ticker_root, n_jobs=1)
        assert list(df.columns) == ["close", "volume", "n_traded"]
        a = df.loc["AAA"]
        sizes = a.groupby(a.index.normalize()).size()
        assert {d.date().isoformat(): int(n) for d, n in sizes.items()} == N_BARS
        firsts = a.groupby(a.index.normalize()).apply(lambda g: g.index[0].time())
        assert all(t == pd.Timestamp("09:30").time() for t in firsts)             # EDT and EST alike
        assert a.loc["2024-11-29"].index[-1] == pd.Timestamp("2024-11-29 12:59")   # early close
        assert a.loc["2024-12-02"].index[-1] == pd.Timestamp("2024-12-02 15:59")
        assert (df.loc["BBB", "close"] > 50).all()                                 # pre/post rows (price 1.0) dropped

    def test_gaps_forward_filled_within_session_only(self, lakes):
        ticker_root, _ = lakes
        a = load_minute_bars(["AAA"], "2024-11-27", "2024-12-02", ticker_root, n_jobs=1).loc["AAA"]
        d = a.loc["2024-11-27"]
        assert d["n_traded"].iloc[3:6].tolist() == [0, 0, 0] and d["volume"].iloc[3:6].tolist() == [0, 0, 0]
        assert (d["close"].iloc[3:6] == d["close"].iloc[2]).all()                  # carried forward
        e = a.loc["2024-12-02"]
        assert e["close"].iloc[:2].isna().all() and e["n_traded"].iloc[:2].tolist() == [0, 0]
        assert e["close"].iloc[2:].notna().all()                                   # never inherits the previous close

    def test_ticker_session_without_trades_is_absent(self, lakes):
        ticker_root, _ = lakes
        df = load_minute_bars(["AAA", "BBB"], "2024-11-27", "2024-12-02", ticker_root, n_jobs=1)
        assert pd.Timestamp("2024-11-29") not in df.loc["BBB"].index.normalize()
        assert pd.Timestamp("2024-11-29") in df.loc["AAA"].index.normalize()
        sess = load_minute_bars(["AAA", "BBB"], "2024-11-27", "2024-12-02", ticker_root, freq="session", n_jobs=1)
        assert ("BBB", pd.Timestamp("2024-11-29")) not in sess.index
        assert ("AAA", pd.Timestamp("2024-11-29")) in sess.index

    def test_five_minute_and_session_aggregation(self, lakes):
        ticker_root, _ = lakes
        m1 = load_minute_bars(["BBB"], "2024-11-27", "2024-12-02", ticker_root, n_jobs=1).loc["BBB"]
        m5 = load_minute_bars(["BBB"], "2024-11-27", "2024-12-02", ticker_root, freq="5min", n_jobs=1).loc["BBB"]
        sizes = m5.groupby(m5.index.normalize()).size()
        assert sizes.tolist() == [78, 78]                          # BBB has no 2024-11-29 rows
        chk = m1.groupby(m1.index.floor("5min")).agg(close=("close", "last"), volume=("volume", "sum"), n_traded=("n_traded", "sum"))
        pd.testing.assert_frame_equal(chk, m5, check_names=False, check_freq=False)
        sess = load_minute_bars(["AAA", "BBB"], "2024-11-27", "2024-12-02", ticker_root, freq="session", n_jobs=1)
        assert list(sess.columns) == ["open", "close", "volume", "dollar_volume", "n_traded", "n_bars", "roll_spread_bps"]
        row = sess.loc[("AAA", pd.Timestamp("2024-11-27"))]
        a = load_minute_bars(["AAA"], "2024-11-27", "2024-11-27", ticker_root, n_jobs=1).loc["AAA"]
        assert row["n_bars"] == 390 and row["n_traded"] == 387
        assert row["close"] == a["close"].iloc[-1] and row["open"] == a["close"].iloc[0]
        assert row["volume"] == 387 * 1000
        assert np.isclose(row["dollar_volume"], (a["close"] * a["volume"]).sum())

    def test_price_column_and_date_bounds(self, lakes):
        ticker_root, _ = lakes
        tr = load_minute_bars(["BBB"], "2024-11-27", "2024-11-27", ticker_root, n_jobs=1)
        raw = load_minute_bars(["BBB"], "2024-11-27", "2024-11-27", ticker_root, price="close", n_jobs=1)
        assert np.allclose(raw["close"], tr["close"] * 1.1)
        assert tr.index.get_level_values("datetime").normalize().unique().tolist() == [pd.Timestamp("2024-11-27")]
        with pytest.raises(ValueError):
            load_minute_bars(["BBB"], "2024-11-27", "2024-11-27", ticker_root, freq="90s", n_jobs=1)


class TestMarketLayout:
    def test_market_layout_matches_ticker_layout(self, lakes):
        ticker_root, market_root = lakes
        for freq in ("1min", "5min", "session"):
            a = load_minute_bars(["AAA", "BBB"], "2024-07-01", "2024-12-02", ticker_root, freq=freq, n_jobs=1)
            b = load_minute_bars(["AAA", "BBB"], "2024-07-01", "2024-12-02", market_root, freq=freq, n_jobs=1)
            pd.testing.assert_frame_equal(a, b)

    def test_sidecar_prunes_row_groups(self, lakes, monkeypatch):
        _, market_root = lakes
        path = market_root / "2024" / "11" / "27.parquet"
        pf = pq.ParquetFile(path)
        assert pf.metadata.num_row_groups > 2            # small row groups so pruning is observable
        read = []
        orig = pq.ParquetFile.read_row_group

        def spy(self, i, *a, **k):
            read.append(i)
            return orig(self, i, *a, **k)

        monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy)
        df = _read_market_day(path, ["BBB"], ["datetime", "ticker", "close_tr", "volume"])
        assert set(df["ticker"]) == {"BBB"} and len(df) == 392                  # 390 bars + pre + post
        assert 0 < len(read) < pf.metadata.num_row_groups                          # only BBB's groups touched

    def test_missing_sidecar_falls_back_to_full_read(self, lakes, tmp_path):
        _, market_root = lakes
        src = market_root / "2024" / "11" / "27.parquet"
        dst = tmp_path / "2024" / "11"
        dst.mkdir(parents=True)
        (dst / "27.parquet").write_bytes(src.read_bytes())
        df = load_minute_bars(["AAA"], "2024-11-27", "2024-11-27", tmp_path, layout="market", n_jobs=1)
        assert len(df.loc["AAA"]) == 390


def test_summarize_sessions(lakes):
    ticker_root, _ = lakes
    sess = load_minute_bars(["AAA", "BBB"], "2024-07-01", "2024-12-02", ticker_root, freq="session", n_jobs=1)
    summ = summarize_sessions(sess)
    assert summ.loc["AAA", "n_sessions"] == 4 and summ.loc["BBB", "n_sessions"] == 3
    assert summ.loc["BBB", "traded_share"] == 1.0
    assert 0.98 < summ.loc["AAA", "traded_share"] < 1.0
    assert summ.loc["AAA", "first"] == pd.Timestamp("2024-07-01") and summ.loc["BBB", "last"] == pd.Timestamp("2024-12-02")


def test_empty_request_returns_a_well_formed_frame(tmp_path):
    """A request the lake cannot satisfy must return an empty frame, not raise.

    The empty path used to call rename_axis(["ticker", "datetime"]) on a frame carrying a
    plain RangeIndex, so "no data" surfaced as a ValueError about index name lengths.
    """
    (tmp_path / "NOSUCH").mkdir()
    out = load_minute_bars(["NOSUCH"], "2024-01-02", "2024-01-05", tmp_path, layout="ticker")
    assert out.empty
    assert list(out.index.names) == ["ticker", "datetime"]
    assert list(out.columns) == ["close", "volume", "n_traded"]
