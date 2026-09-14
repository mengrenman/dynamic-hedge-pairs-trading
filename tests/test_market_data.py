"""Unit tests for pairs.market_data (OpenBB adapter and Polygon parquet lake)."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from pairs.market_data.openbb_history import download_history_openbb, normalize_symbol
from pairs.market_data.polygon_lake import load_polygonio_lake, select_lake_files


# ── OpenBB: symbol normalisation ─────────────────────────────────────────────

class TestNormalizeSymbol:
    def test_yfinance_share_class_uses_dash(self):
        assert normalize_symbol("BRK.B", "yfinance") == "BRK-B"
        assert normalize_symbol("bf.b", "yfinance") == "BF-B"

    def test_plain_ticker_unchanged(self):
        assert normalize_symbol("AAPL", "yfinance") == "AAPL"

    def test_other_providers_untouched(self):
        assert normalize_symbol("BRK.B", "polygon") == "BRK.B"


# ── OpenBB: loader against a stub `openbb` module ────────────────────────────

def _frame(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", name="date")
    return pd.DataFrame({"close": 100.0 + np.arange(n, dtype=float)}, index=idx)


def _install_fake_openbb(monkeypatch, frames: dict, calls: list):
    """Stub `openbb.obb` so obb.equity.price.historical(symbol=...) returns frames[symbol]."""

    def historical(symbol, start_date, end_date, provider):
        calls.append(symbol)
        if symbol not in frames:
            raise RuntimeError(f"no data for {symbol}")
        return frames[symbol]

    obb = types.SimpleNamespace(
        user=types.SimpleNamespace(preferences=types.SimpleNamespace(output_type=None)),
        equity=types.SimpleNamespace(price=types.SimpleNamespace(historical=historical)),
    )
    monkeypatch.setitem(sys.modules, "openbb", types.SimpleNamespace(obb=obb))
    return obb


class TestDownloadHistoryOpenbb:
    def test_openbb_is_imported_lazily(self, monkeypatch):
        # Importing the module must not require openbb; only the call does.
        monkeypatch.setitem(sys.modules, "openbb", None)  # makes `import openbb` fail
        with pytest.raises(ImportError, match="OpenBB"):
            download_history_openbb(["AAPL"], "2024-01-01", "2024-02-01", show_progress=False)

    def test_stitches_multiindex_and_keeps_universe_ticker(self, monkeypatch):
        calls: list = []
        _install_fake_openbb(monkeypatch, {"AAPL": _frame(), "BRK-B": _frame()}, calls)

        df, failed = download_history_openbb(
            ["AAPL", "BRK.B", "ZZZZ"], "2024-01-01", "2024-02-01",
            show_progress=False, return_failed=True,
        )

        # Verbatim request first; the dash spelling only as a retry; no retry when it would be identical
        assert calls == ["AAPL", "BRK.B", "BRK-B", "ZZZZ"]
        assert failed == ["ZZZZ"]
        assert list(df.index.names) == ["ticker", "datetime"]
        assert set(df.index.get_level_values("ticker")) == {"AAPL", "BRK.B"}   # universe spelling kept
        assert len(df) == 10
        assert isinstance(df.index.get_level_values("datetime"), pd.DatetimeIndex)

    def test_dotted_symbol_that_resolves_is_not_rewritten(self, monkeypatch):
        # Exchange suffixes (VOD.L, RY.TO) must keep their dot: the verbatim request succeeds, no retry.
        calls: list = []
        _install_fake_openbb(monkeypatch, {"VOD.L": _frame()}, calls)
        df = download_history_openbb(["VOD.L"], "2024-01-01", "2024-02-01", show_progress=False)
        assert calls == ["VOD.L"]
        assert set(df.index.get_level_values("ticker")) == {"VOD.L"}

    def test_no_retry_for_other_providers(self, monkeypatch):
        calls: list = []
        _install_fake_openbb(monkeypatch, {}, calls)
        _, failed = download_history_openbb(["BRK.B"], "2024-01-01", "2024-02-01",
                                            provider="polygon", show_progress=False, return_failed=True)
        assert calls == ["BRK.B"]
        assert failed == ["BRK.B"]

    def test_requests_dataframe_output(self, monkeypatch):
        obb = _install_fake_openbb(monkeypatch, {"AAPL": _frame()}, [])
        download_history_openbb(["AAPL"], "2024-01-01", "2024-02-01", show_progress=False)
        assert obb.user.preferences.output_type == "dataframe"

    def test_all_failed_returns_empty_multiindex_frame(self, monkeypatch):
        _install_fake_openbb(monkeypatch, {}, [])
        df, failed = download_history_openbb(
            ["ZZZZ"], "2024-01-01", "2024-02-01", show_progress=False, return_failed=True
        )
        assert df.empty
        assert failed == ["ZZZZ"]
        assert list(df.index.names) == ["ticker", "datetime"]

    def test_deduplicates_and_drops_nan_tickers(self, monkeypatch):
        calls: list = []
        _install_fake_openbb(monkeypatch, {"AAPL": _frame()}, calls)
        df = download_history_openbb(
            ["AAPL", "AAPL", None], "2024-01-01", "2024-02-01", show_progress=False
        )
        assert calls == ["AAPL"]
        assert len(df) == 5


# ── Polygon parquet lake ─────────────────────────────────────────────────────

pyarrow = pytest.importorskip("pyarrow")


def _write_day_lake(root, ticker: str, months, *, tz="US/Eastern", n_days: int = 3):
    """Write <root>/<TICKER>/<YYYY>/<MM>.parquet files with n_days bars each."""
    for yyyy, mm in months:
        days = pd.date_range(f"{yyyy}-{mm:02d}-01", periods=n_days, freq="B", tz=tz)
        df = pd.DataFrame({
            "datetime": days,
            "ticker": ticker,
            "close": np.arange(n_days, dtype=float) + 1.0,
        })
        p = root / ticker / f"{yyyy:04d}" / f"{mm:02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)


class TestPolygonLake:
    def test_select_files_walks_month_layout(self, tmp_path):
        _write_day_lake(tmp_path, "AAPL", [(2024, 1), (2024, 2), (2024, 3)])
        files = select_lake_files(["aapl"], "2024-02-01", "2024-02-28", tmp_path)
        assert [f.name for f in files] == ["02.parquet"]

    def test_load_filters_range_and_builds_multiindex(self, tmp_path):
        _write_day_lake(tmp_path, "AAPL", [(2024, 1), (2024, 2)])
        _write_day_lake(tmp_path, "MSFT", [(2024, 1)])

        df = load_polygonio_lake(
            ["AAPL", "MSFT"], "2024-01-01", "2024-01-31", tmp_path,
            set_index=True, index_multi=True,
        )

        assert list(df.index.names) == ["ticker", "datetime"]
        assert set(df.index.get_level_values("ticker")) == {"AAPL", "MSFT"}
        assert len(df) == 6  # 3 bars each, February excluded
        end = pd.Timestamp("2024-01-31 23:59:59", tz="US/Eastern")
        assert df.index.get_level_values("datetime").max() <= end

    def test_tz_naive_parquet_is_localised(self, tmp_path):
        _write_day_lake(tmp_path, "AAPL", [(2024, 1)], tz=None)
        out = load_polygonio_lake(["AAPL"], "2024-01-01", "2024-01-31", tmp_path,
                                  source_tz="US/Eastern")
        assert isinstance(out["datetime"].dtype, pd.DatetimeTZDtype)
        assert str(out["datetime"].dt.tz) == "US/Eastern"
        assert len(out) == 3

    def test_to_timezone_converts(self, tmp_path):
        _write_day_lake(tmp_path, "AAPL", [(2024, 1)])
        out = load_polygonio_lake(["AAPL"], "2024-01-01", "2024-01-31", tmp_path,
                                  to_timezone="UTC")
        assert str(out["datetime"].dt.tz) == "UTC"

    def test_missing_ticker_returns_empty(self, tmp_path):
        out = load_polygonio_lake(["NOPE"], "2024-01-01", "2024-01-31", tmp_path)
        assert out.empty
        assert {"datetime", "ticker"} <= set(out.columns)
