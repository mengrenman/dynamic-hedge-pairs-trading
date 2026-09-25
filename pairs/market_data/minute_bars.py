# pairs/market_data/minute_bars.py
"""
Minute bars from the adjusted Polygon minute lake, on a regular trading-session grid.

Lake layouts
------------
Two on-disk layouts are understood and auto-detected from ``root``:

``ticker`` layout — one file per ticker-day::

    <root>/<TICKER>/<YYYY>/<MM>/<DD>.parquet

``market`` layout — one file per day holding every ticker, sorted ticker-major in
fixed-size row groups, plus a sidecar that records each ticker's row range::

    <root>/<YYYY>/<MM>/<DD>.parquet          columns: datetime, ticker, close, close_tr, ...
    <root>/<YYYY>/<MM>/<DD>.idx.parquet      columns: ticker, n_rows, row_start, row_end (inclusive)

Because the day file is sorted by ticker, a symbol's rows sit in a handful of consecutive row
groups; the loader maps ``row_start``/``row_end`` from the sidecar onto row-group offsets and reads
only those groups, so a single symbol costs milliseconds even from a 1.5-million-row day file.

Conventions
-----------
* Lake timestamps are UTC without a timezone marker and label the *start* of the minute.
  They are converted to ``tz`` (default US/Eastern) and returned timezone-naive.
* Prices come from ``price`` (default ``close_tr``: split- and dividend-adjusted).
* Regular trading hours are 09:30–16:00 Eastern (09:30–13:00 on NYSE early-close days, see
  :func:`nyse_early_closes`). Pre- and post-market bars are dropped.
* Every ticker-session with at least one trade is put on the full minute grid: minutes without
  a trade carry the last traded price forward (within the session only — a session never
  inherits the previous close), ``volume`` 0 and ``n_traded`` 0. Minutes before a session's
  first trade stay NaN. A ticker-session with no trades at all is absent from the output.
* ``freq`` aggregates the minute grid per session: ``"5min"`` etc. (bars labeled by their
  start, last bar of the session may be shorter) or ``"session"`` (one row per session with
  open/close/volume/dollar volume, traded-minute count and a Roll effective-spread estimate).

Public API
----------
- detect_lake_layout(root)
- nyse_early_closes(start, end)
- load_minute_bars(tickers, start, end, root, freq=..., price=..., ...)
- summarize_sessions(session_frame)
"""
from __future__ import annotations

import datetime as _dt
import glob
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "detect_lake_layout",
    "nyse_early_closes",
    "load_minute_bars",
    "summarize_sessions",
    "RTH_OPEN_MINUTE",
    "RTH_CLOSE_MINUTE",
    "EARLY_CLOSE_MINUTE",
]

RTH_OPEN_MINUTE = 9 * 60 + 30      # 09:30
RTH_CLOSE_MINUTE = 16 * 60         # 16:00 (exclusive)
EARLY_CLOSE_MINUTE = 13 * 60       # 13:00 (exclusive) on early-close days


# ───────────────────────── layout & calendar helpers ─────────────────────────

def detect_lake_layout(root: str | Path) -> str:
    """Return ``"ticker"`` or ``"market"`` for a lake root (raises if neither is recognized)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"lake root does not exist: {root}")
    names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not names:
        raise FileNotFoundError(f"lake root is empty: {root}")
    if all(n.isdigit() and len(n) == 4 for n in names):
        return "market"
    if any(n.isdigit() and len(n) == 4 for n in names):
        raise ValueError(f"cannot tell the layout of {root}: mixes year folders and ticker folders")
    return "ticker"


def nyse_early_closes(start, end) -> pd.DatetimeIndex:
    """
    NYSE 13:00 early-close dates in ``[start, end]`` by rule: the day after Thanksgiving, July 3
    when it falls on Monday–Thursday, and December 24 when it falls on Monday–Thursday.
    (When July 3 or December 24 is a Friday the exchange is closed for the observed holiday.)
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out = []
    for year in range(start.year, end.year + 1):
        thanksgiving = pd.Timestamp(year, 11, 1) + pd.offsets.WeekOfMonth(week=3, weekday=3)  # 4th Thursday
        out.append(thanksgiving + pd.Timedelta(days=1))
        for month, day in ((7, 3), (12, 24)):
            d = pd.Timestamp(year, month, day)
            if d.weekday() <= 3:
                out.append(d)
    idx = pd.DatetimeIndex(sorted(out))
    return idx[(idx >= start) & (idx <= end)]


def _session_close_minute(dates: pd.DatetimeIndex, early: pd.DatetimeIndex) -> np.ndarray:
    return np.where(dates.isin(early), EARLY_CLOSE_MINUTE, RTH_CLOSE_MINUTE)


# ───────────────────────── file selection & reading ──────────────────────────

def _days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.normalize(), end.normalize(), freq="D")


def _ticker_files(root: Path, ticker: str, days: pd.DatetimeIndex) -> List[Path]:
    files = []
    months = sorted({(d.year, d.month) for d in days})
    for y, m in months:
        ddir = root / ticker / f"{y:04d}" / f"{m:02d}"
        if not ddir.is_dir():
            continue
        wanted = {f"{d.day:02d}.parquet" for d in days if (d.year, d.month) == (y, m)}
        files.extend(sorted(p for p in ddir.glob("*.parquet") if p.name in wanted))
    return files


def _market_files(root: Path, days: pd.DatetimeIndex) -> List[Path]:
    files = []
    for d in days:
        p = root / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}.parquet"
        if p.exists():
            files.append(p)
    return files


def _read_ticker_file(path: Path, columns: List[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pandas()


def _read_market_day(path: Path, tickers: Optional[Sequence[str]], columns: List[str]) -> pd.DataFrame:
    """Read one market-layout day file, touching only the row groups that hold ``tickers``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    if tickers is None:
        return pf.read(columns=columns).to_pandas()

    sidecar = path.with_suffix("").with_suffix(".idx.parquet")  # DD.parquet -> DD.idx.parquet
    if not sidecar.exists():                       # no sidecar: read the whole day and filter
        df = pf.read(columns=columns).to_pandas()
        return df[df["ticker"].isin(set(tickers))]

    idx = pq.read_table(sidecar, columns=["ticker", "row_start", "row_end"]).to_pandas()
    idx = idx[idx["ticker"].isin(set(tickers))]
    if idx.empty:
        return pd.DataFrame(columns=columns)

    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    offsets = np.cumsum([0] + sizes)                       # row-group start offsets
    first = np.searchsorted(offsets, idx["row_start"].to_numpy(), side="right") - 1
    last = np.searchsorted(offsets, idx["row_end"].to_numpy(), side="right") - 1
    groups = sorted({g for a, b in zip(first, last) for g in range(int(a), int(b) + 1)})
    tab = pa.concat_tables([pf.read_row_group(g, columns=columns) for g in groups])
    df = tab.to_pandas()
    return df[df["ticker"].isin(set(idx["ticker"]))]


# ───────────────────────── session grid & aggregation ────────────────────────

def _to_local_naive(dt: pd.Series, tz: str) -> pd.Series:
    return dt.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)


def _freq_minutes(freq: str) -> Optional[int]:
    if freq == "session":
        return None
    td = pd.Timedelta(freq)
    mins = td / pd.Timedelta(minutes=1)
    if mins != int(mins) or mins < 1:
        raise ValueError(f"freq must be a whole number of minutes or 'session', got {freq!r}")
    return int(mins)


def _process(raw: pd.DataFrame, *, price: str, freq: str, tz: str, early: pd.DatetimeIndex) -> pd.DataFrame:
    """
    raw: columns ticker, datetime (UTC-naive), <price>, volume — any number of tickers/days.
    Returns a (ticker, datetime) frame on the regular session grid aggregated to ``freq``.
    """
    if raw.empty:
        # a plain RangeIndex cannot take two names; build the empty MultiIndex explicitly so that
        # "no rows in the lake for this request" surfaces as an empty frame of the right shape
        # rather than a ValueError from rename_axis.
        idx = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["ticker", "datetime"])
        return pd.DataFrame(columns=["close", "volume", "n_traded"], index=idx, dtype=float)

    df = pd.DataFrame({
        "ticker": raw["ticker"].astype(str).to_numpy(),
        "dt": _to_local_naive(pd.to_datetime(raw["datetime"]), tz).to_numpy(),
        "close": raw[price].astype(float).to_numpy(),
        "volume": raw["volume"].astype(float).to_numpy(),
    })
    dt = pd.DatetimeIndex(df["dt"])
    date = dt.normalize()
    mod = dt.hour * 60 + dt.minute
    close_minute = _session_close_minute(date, early)
    rth = (mod >= RTH_OPEN_MINUTE) & (mod < close_minute)
    df = df.loc[rth]
    if df.empty:
        return pd.DataFrame(columns=["close", "volume", "n_traded"]).rename_axis(["ticker", "datetime"])

    # ── the grid: every (ticker, session minute) for the sessions present in the data ──
    tickers = np.sort(df["ticker"].unique())
    sessions = pd.DatetimeIndex(np.sort(pd.DatetimeIndex(df["dt"]).normalize().unique()))
    n_bars = _session_close_minute(sessions, early) - RTH_OPEN_MINUTE
    minutes = np.concatenate([
        (s + pd.Timedelta(minutes=RTH_OPEN_MINUTE)).to_datetime64() + np.arange(n, dtype="timedelta64[m]")
        for s, n in zip(sessions, n_bars)
    ]).astype("datetime64[ns]")
    grid = pd.MultiIndex.from_product([tickers, minutes], names=["ticker", "datetime"])

    bars = (df.set_index(["ticker", "dt"])[["close", "volume"]]
              .rename_axis(["ticker", "datetime"]))
    bars = bars[~bars.index.duplicated(keep="last")].reindex(grid)

    traded = (bars["volume"].fillna(0.0) > 0) & bars["close"].notna()
    volume = bars["volume"].fillna(0.0)
    # forward-fill within a session only: blank everything before the session's first trade
    n_tick, n_min = len(tickers), len(minutes)
    sess_id_min = np.repeat(np.arange(len(sessions)), n_bars)                  # per grid minute
    seg = (np.repeat(np.arange(n_tick), n_min) * len(sessions) + np.tile(sess_id_min, n_tick))
    valid = bars["close"].notna().to_numpy()
    by_seg = pd.Series(valid).groupby(seg)
    seen = by_seg.cummax().to_numpy()
    any_trade = by_seg.transform("any").to_numpy()        # ticker-sessions without a single trade are dropped
    close = bars["close"].ffill().where(seen)

    out = pd.DataFrame({"close": close.to_numpy(), "volume": volume.to_numpy(),
                        "n_traded": traded.to_numpy().astype(np.int64)}, index=grid)
    out = out[any_trade]
    bars = bars[any_trade]
    k = _freq_minutes(freq)
    if k == 1:
        return out

    lvl_dt = out.index.get_level_values("datetime")
    minute_in_session = ((lvl_dt.hour * 60 + lvl_dt.minute) - RTH_OPEN_MINUTE).to_numpy()
    session = lvl_dt.normalize()
    tick = out.index.get_level_values("ticker")

    if k is not None:  # k-minute bars, labeled by bar start, bins anchored at 09:30
        label = pd.DatetimeIndex(session + pd.to_timedelta(RTH_OPEN_MINUTE + (minute_in_session // k) * k, unit="m"))
        g = out.groupby([tick, label], sort=True, observed=True)
        agg = g.agg(close=("close", "last"), volume=("volume", "sum"), n_traded=("n_traded", "sum"))
        agg.index.names = ["ticker", "datetime"]
        return agg

    # ── session bars ──
    px_raw = bars["close"]                                    # NaN on untraded minutes
    dollar = (px_raw * volume).fillna(0.0)
    g = out.assign(dollar=dollar.to_numpy(), px_raw=px_raw.to_numpy()).groupby([tick, session], sort=True, observed=True)
    agg = g.agg(open=("px_raw", "first"), close=("close", "last"), volume=("volume", "sum"),
                dollar_volume=("dollar", "sum"), n_traded=("n_traded", "sum"), n_bars=("close", "size"))
    agg.index.names = ["ticker", "datetime"]
    agg["roll_spread_bps"] = _roll_by_session(px_raw, tick, session)
    return agg[agg["close"].notna()]


def _roll_by_session(px_raw: pd.Series, tick: pd.Index, session: pd.DatetimeIndex) -> pd.Series:
    """Roll (1984) effective spread per (ticker, session) from consecutive traded-minute closes, in bps."""
    s = pd.Series(px_raw.to_numpy(), index=pd.MultiIndex.from_arrays([tick, session], names=["ticker", "session"]))
    key = s.index
    p = s.dropna()
    grp = p.groupby(level=[0, 1], sort=False)
    d1 = grp.diff()
    d0 = d1.groupby(level=[0, 1], sort=False).shift(1)
    prod = (d1 * d0)
    gp = prod.groupby(level=[0, 1], sort=True)
    cov = gp.mean() - d1.groupby(level=[0, 1], sort=True).mean() * d0.groupby(level=[0, 1], sort=True).mean()
    n = gp.count()
    mean_px = p.groupby(level=[0, 1], sort=True).mean()
    spread = 2.0 * np.sqrt((-cov).clip(lower=0.0))
    roll = (spread / mean_px * 1e4).where((cov < 0) & (n >= 30))
    full = pd.MultiIndex.from_arrays([tick, session]).unique()
    return roll.reindex(full).to_numpy()


# ───────────────────────── public loader ─────────────────────────────────────

def load_minute_bars(
    tickers: Iterable[str],
    start,
    end,
    root: str | Path,
    *,
    freq: str = "1min",
    price: str = "close_tr",
    tz: str = "US/Eastern",
    layout: Optional[str] = None,
    early_closes: Optional[pd.DatetimeIndex] = None,
    n_jobs: int = -1,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Load regular-session minute bars for ``tickers`` between ``start`` and ``end`` (inclusive dates).

    Parameters
    ----------
    tickers : iterable of str
    start, end : date-like (interpreted as calendar dates in ``tz``)
    root : lake root (ticker or market layout, auto-detected unless ``layout`` is given)
    freq : ``"1min"`` (default), any whole number of minutes such as ``"5min"``, or ``"session"``
    price : lake price column to use as ``close`` (``close_tr``, ``close_split`` or ``close``)
    tz : exchange timezone used for the session grid
    early_closes : override the rule-based early-close calendar (``nyse_early_closes``)
    n_jobs : joblib workers (-1 = all cores); one task per ticker (ticker layout) or per day (market layout)

    Returns
    -------
    DataFrame indexed by (ticker, datetime) — timezone-naive ``tz`` timestamps labeling bar
    starts — with columns ``close``, ``volume``, ``n_traded`` (minutes in the bar with a trade);
    for ``freq="session"`` the index holds session dates and the columns are ``open``, ``close``,
    ``volume``, ``dollar_volume``, ``n_traded``, ``n_bars``, ``roll_spread_bps``.
    """
    from joblib import Parallel, delayed

    root = Path(root)
    layout = layout or detect_lake_layout(root)
    if layout not in ("ticker", "market"):
        raise ValueError(f"layout must be 'ticker' or 'market', got {layout!r}")
    _freq_minutes(freq)  # validate early
    tickers = [str(t).upper() for t in tickers]
    if not tickers:
        raise ValueError("tickers is empty")
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e < s:
        raise ValueError("end is before start")
    # UTC files can hold the tail of the previous local day and the head of the next; widen by a day.
    days = _days(s - pd.Timedelta(days=1), e + pd.Timedelta(days=1))
    early = nyse_early_closes(s, e) if early_closes is None else pd.DatetimeIndex(early_closes)
    cols = ["datetime", "ticker", price, "volume"]

    if layout == "ticker":
        def task(t):
            files = _ticker_files(root, t, days)
            if not files:
                return None
            raw = pd.concat([_read_ticker_file(f, cols) for f in files], ignore_index=True)
            return _process(raw, price=price, freq=freq, tz=tz, early=early)
        items = tickers
    else:
        files = _market_files(root, days)
        def task(f):
            raw = _read_market_day(f, tickers, cols)
            return _process(raw, price=price, freq=freq, tz=tz, early=early)
        items = files

    if show_progress:
        from tqdm import tqdm
        items = tqdm(list(items), desc=f"minute bars ({layout}, {freq})", unit="task", leave=False)
    parts = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(task)(it) for it in items)
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return _process(pd.DataFrame(columns=cols), price=price, freq=freq, tz=tz, early=early)

    out = pd.concat(parts).sort_index()
    lo = s.normalize()
    hi = e.normalize() + pd.Timedelta(days=1)
    lvl = out.index.get_level_values("datetime")
    return out[(lvl >= lo) & (lvl < hi)]


def summarize_sessions(session_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker liquidity summary of a ``freq="session"`` frame: number of sessions, first/last
    session, median traded-minute share, median session dollar volume and median Roll spread.
    """
    g = session_frame.groupby(level="ticker", observed=True)
    dt = session_frame.index.get_level_values("datetime")
    return pd.DataFrame({
        "n_sessions": g.size(),
        "first": pd.Series(dt, index=session_frame.index).groupby(level="ticker", observed=True).min(),
        "last": pd.Series(dt, index=session_frame.index).groupby(level="ticker", observed=True).max(),
        "traded_share": (session_frame["n_traded"] / session_frame["n_bars"]).groupby(level="ticker", observed=True).median(),
        "median_dollar_volume": g["dollar_volume"].median(),
        "median_roll_bps": g["roll_spread_bps"].median(),
    })
