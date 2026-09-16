# pairs/market_data/daily_bars.py
"""
Daily bars from the adjusted Polygon day lake, with an explicit choice of price basis.

Lake layouts
------------
Two on-disk layouts are understood and auto-detected from ``root``:

``ticker`` layout — one file per ticker-month::

    <root>/<TICKER>/<YYYY>/<MM>.parquet

``market`` layout — one file per month holding every ticker, sorted ticker-major::

    <root>/<YYYY>/<MM>.parquet          columns: datetime, ticker, close, close_split, close_tr, ...

The market layout is the one to use for universe work: it carries every symbol that traded,
including those later delisted, so a universe built from it is free of survivorship bias. The
ticker layout covers only the index members the lake was built for.

Price basis (the ``price`` argument)
------------------------------------
Each row carries three price variants and the factors that relate them:

``"split"`` (default)
    ``close_split`` — adjusted for splits only. Splits are mechanical and known on the day, so
    this is the series a trader would actually have seen, rebased. **Use this for backtests.**
``"tr"``
    ``close_tr`` — adjusted for splits *and* dividends, back-adjusted so the factor is 1.0 on the
    lake's last date. The factor at any historical date therefore embeds every dividend paid
    **after** that date, which is information the trader did not have. Convenient for computing
    total returns over a completed sample; **look-ahead if used as a tradeable price.**
``"raw"``
    ``close`` — unadjusted; discontinuous across splits.

Ticker reuse
------------
The market layout holds every symbol that ever traded, and tickers are reused: "TROW" in
December 2003 is both T. Rowe Price and a different, near-untraded security, and "DOC" is both
Physicians Realty and a company that last traded in 2007. Each row carries a ``id`` (a FIGI, or
a ``NOFIGI__``/``CIK__`` placeholder), which is the real instrument key. By default the loader
resolves each ticker to the single instrument with the most dollar volume in the requested
window and records what it dropped in ``frame.attrs["ticker_collisions"]``; a universe built
without this step silently splices two companies into one price series.

Dividends
---------
``with_dividends=True`` recovers the per-share cash dividend from the steps in
``tr_price_factor``: on an ex-dividend date the factor steps by ``1 - D / P_prev``, so

    D_t = (1 - f_{t-1} / f_t) * close_split_{t-1}

This gives a dividend stream that can be accrued as a cash flow in the backtest, keeping the
total-return economics without importing the look-ahead of ``close_tr``.

Public API
----------
- detect_day_lake_layout(root)
- load_daily_bars(tickers, start, end, root, price=..., with_dividends=...)
- recover_dividends(close_split, tr_price_factor)
- liquidity_screen(bars, ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "detect_day_lake_layout",
    "load_daily_bars",
    "recover_dividends",
    "liquidity_screen",
]

_PRICE_COLUMN = {"split": "close_split", "tr": "close_tr", "raw": "close"}


# ───────────────────────── layout detection & file selection ─────────────────

def detect_day_lake_layout(root: str | Path) -> str:
    """Return ``"ticker"`` or ``"market"`` for a day-lake root (raises if neither is recognised)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"day lake root does not exist: {root}")
    names = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not names:
        raise FileNotFoundError(f"day lake root is empty: {root}")
    years = [n for n in names if n.isdigit() and len(n) == 4]
    if len(years) == len(names):
        return "market"
    if years:
        raise ValueError(f"cannot tell the layout of {root}: mixes year folders and ticker folders")
    return "ticker"


def _months(start: pd.Timestamp, end: pd.Timestamp) -> List[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _files(root: Path, layout: str, tickers: Optional[Sequence[str]], months) -> List[Path]:
    files: List[Path] = []
    if layout == "market":
        for y, m in months:
            p = root / f"{y:04d}" / f"{m:02d}.parquet"
            if p.exists():
                files.append(p)
    else:
        for t in tickers or []:
            for y, m in months:
                p = root / t / f"{y:04d}" / f"{m:02d}.parquet"
                if p.exists():
                    files.append(p)
    return files


# ───────────────────────── dividends ─────────────────────────────────────────

def recover_dividends(close_split: pd.Series, tr_price_factor: pd.Series) -> pd.Series:
    """
    Per-share cash dividends implied by the steps in the total-return price factor.

    On an ex-dividend date the back-adjustment factor steps by ``1 - D / P_prev``, so
    ``D_t = (1 - f_{t-1}/f_t) * close_split_{t-1}``. Dates that are not ex-dividend dates get 0.
    Negative results (factor steps the wrong way, which happens around bad or restated events)
    are clipped to 0.

    Both inputs must be sorted by date and aligned; the result is aligned to them.
    """
    px = pd.Series(close_split, dtype=float)
    f = pd.Series(tr_price_factor, dtype=float)
    if len(px) != len(f):
        raise ValueError("close_split and tr_price_factor must have the same length")
    ratio = f.shift(1) / f
    div = (1.0 - ratio) * px.shift(1)
    return div.where(np.isfinite(div), 0.0).clip(lower=0.0).fillna(0.0)


# ───────────────────────── public loader ─────────────────────────────────────

def load_daily_bars(
    tickers: Optional[Iterable[str]],
    start,
    end,
    root: str | Path,
    *,
    price: str = "split",
    with_dividends: bool = False,
    layout: Optional[str] = None,
    tz: str = "US/Eastern",
    min_rows: int = 1,
) -> pd.DataFrame:
    """
    Load daily bars for ``tickers`` between ``start`` and ``end`` (inclusive dates).

    Parameters
    ----------
    tickers : iterable of str, or None for *every* ticker in the lake (market layout only)
    start, end : date-like
    root : lake root (ticker or market layout, auto-detected unless ``layout`` is given)
    price : ``"split"`` (default, tradeable), ``"tr"`` (total return, has look-ahead) or ``"raw"``
    with_dividends : add a ``dividend`` column recovered from ``tr_price_factor`` (see
        :func:`recover_dividends`). Requires the lake's factor column.
    tz : exchange timezone; lake timestamps are midnight local expressed in UTC and are
        converted back to a plain session date
    min_rows : drop tickers with fewer than this many rows in the window

    Returns
    -------
    DataFrame indexed by ``(ticker, datetime)`` — timezone-naive session dates — with columns
    ``close`` (the chosen basis), ``raw_close`` (unadjusted, for liquidity gates), ``volume``,
    ``dollar_volume``, ``id`` (the resolved instrument)
    and, optionally, ``dividend``. ``frame.attrs["ticker_collisions"]`` is a list of dicts naming the
    tickers that mapped to more than one instrument in the window, the instrument kept and those
    dropped; wrap it in ``pd.DataFrame`` to inspect it. (A list, not a frame, so that the result can be
    written straight to parquet — pandas serialises ``attrs`` as JSON.)
    """
    if price not in _PRICE_COLUMN:
        raise ValueError(f"price must be one of {sorted(_PRICE_COLUMN)}; got {price!r}")
    root = Path(root)
    layout = layout or detect_day_lake_layout(root)
    if layout not in ("ticker", "market"):
        raise ValueError(f"layout must be 'ticker' or 'market', got {layout!r}")
    if tickers is None and layout != "market":
        raise ValueError("tickers=None (the whole lake) is only supported for the market layout")
    tickers = None if tickers is None else [str(t).upper() for t in tickers]
    if tickers is not None and not tickers:
        raise ValueError("tickers is empty")

    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if e < s:
        raise ValueError("end is before start")

    px_col = _PRICE_COLUMN[price]
    cols = ["datetime", "ticker", "id", px_col, "volume_split"]
    for extra in ("close_split", "close"):            # dollar volume is split-adjusted; the gate is raw
        if extra not in cols:
            cols.append(extra)
    if with_dividends:
        cols.append("tr_price_factor")

    import pyarrow.parquet as pq

    frames = []
    for f in _files(root, layout, tickers, _months(s, e)):
        frames.append(pq.read_table(f, columns=cols).to_pandas())
    empty_cols = ["close", "raw_close", "volume", "dollar_volume", "id"] + (["dividend"] if with_dividends else [])
    if not frames:
        out = pd.DataFrame(columns=empty_cols)
        out.index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "datetime"])
        out.attrs["ticker_collisions"] = []
        return out

    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["id"] = df["id"].astype(str)
    if tickers is not None:
        df = df[df["ticker"].isin(set(tickers))]
    # lake timestamps are local midnight expressed in UTC → recover the plain session date
    df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    df = df[(df["datetime"] >= s) & (df["datetime"] <= e)]
    base = (df["close_split"] if "close_split" in df.columns else df[px_col]).astype(float)
    df = df.assign(_dv=base.to_numpy() * df["volume_split"].astype(float).to_numpy(), _base=base.to_numpy())

    # ── one instrument per ticker: keep the id with the most dollar volume in the window ──
    by_id = df.groupby(["ticker", "id"], sort=False)["_dv"].sum()
    dominant = by_id.groupby(level="ticker").idxmax().map(lambda k: k[1])
    n_ids = by_id.groupby(level="ticker").size()
    clashing = n_ids[n_ids > 1].index
    rows = []
    for t in clashing:
        kept = dominant[t]
        for other, dv_sum in by_id.loc[t].items():
            if other != kept:
                rows.append({"ticker": t, "kept": kept, "dropped": other, "dropped_dollar_volume": float(dv_sum)})
    collisions = rows
    if len(clashing):
        df = df[df["id"].to_numpy() == df["ticker"].map(dominant).to_numpy()]

    df = df.sort_values(["ticker", "datetime"], kind="mergesort")
    df = df[~df.duplicated(["ticker", "datetime"], keep="last")]

    out = pd.DataFrame({
        "close": df[px_col].astype(float).to_numpy(),
        "raw_close": df["close"].astype(float).to_numpy(),
        "volume": df["volume_split"].astype(float).to_numpy(),
        "dollar_volume": df["_dv"].to_numpy(),
        "id": df["id"].to_numpy(),
    }, index=pd.MultiIndex.from_arrays([df["ticker"].to_numpy(), df["datetime"].to_numpy()],
                                       names=["ticker", "datetime"]))
    if with_dividends:
        # vectorised equivalent of recover_dividends applied within each ticker
        tmp = pd.DataFrame({"t": df["ticker"].to_numpy(), "px": df["_base"].to_numpy(),
                            "f": df["tr_price_factor"].astype(float).to_numpy()})
        g = tmp.groupby("t", sort=False)
        div = (1.0 - g["f"].shift(1) / tmp["f"]) * g["px"].shift(1)
        out["dividend"] = div.where(np.isfinite(div), 0.0).clip(lower=0.0).fillna(0.0).to_numpy()
    if min_rows > 1:
        keep = out.groupby(level="ticker").size() >= min_rows
        out = out[out.index.get_level_values("ticker").isin(keep[keep].index)]
    out.attrs["ticker_collisions"] = collisions
    return out


# ───────────────────────── universe helper ───────────────────────────────────

def liquidity_screen(
    bars: pd.DataFrame,
    *,
    min_price: float = 5.0,
    min_dollar_volume: float = 20e6,
    min_ann_vol: float = 0.15,
    min_coverage: float = 0.99,
    max_abs_return: Optional[float] = 1.0,
    exclude: Optional[Iterable[str]] = None,
    price_column: str = "raw_close",
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Per-ticker statistics over the window plus the eligibility flags a point-in-time universe needs.

    A ticker is eligible when it traded on at least ``min_coverage`` of the window's sessions, its
    median close is at least ``min_price``, its median dollar volume is at least
    ``min_dollar_volume``, its annualised volatility is at least ``min_ann_vol``, its largest
    absolute one-day return is at most ``max_abs_return``, and it is not in ``exclude``.

    Two of those gates exist because a raw market lake is not a curated index list:

    * the **volatility floor** — ranking a liquid universe by dollar volume alone puts money-market
      and ultra-short bond ETFs (BIL, SGOV, SHV) at the top, and because their prices barely move
      they pass cointegration tests against almost anything;
    * the **return cap** — a one-day move of several hundred percent is a corporate action the
      lake's adjustment factors missed (usually a reverse split), not a price. Passing such a series
      to a cointegration test manufactures a structural break. Set it to ``None`` to disable.

    The ``min_price`` gate is applied to ``price_column``, which defaults to the **unadjusted**
    ``raw_close`` and falls back to ``close`` when that column is absent. This matters more than it
    sounds: split adjustment is backward-looking, so a stock that later reverse-split appears far
    more expensive than it was. Sirius XM traded near \$3 in 2008 but its 1-for-10 reverse split in
    2024 leaves a split-adjusted 2008 price near \$30, and Citigroup's 2011 1-for-10 reverse split
    does the same to 2009. Gating on the adjusted price admits exactly the distressed penny stocks a
    liquidity filter is meant to exclude, and they go on to dominate a cointegration screen.

    Exchange test symbols (NASDAQ's ``ZVZZT`` and its siblings, which quote and "trade" like real
    securities) are not detectable from the numbers and should be passed in ``exclude``.

    Returns a frame indexed by ticker with ``median_price``, ``median_dollar_volume``, ``ann_vol``,
    ``coverage``, ``max_abs_return`` and ``eligible`` (``median_price`` is on ``price_column``);
    with ``top_n`` only the ``top_n`` eligible
    tickers by median dollar volume are marked eligible.
    """
    px = bars["close"].unstack("ticker")
    gate_px = bars[price_column].unstack("ticker") if price_column in bars.columns else px
    dv = bars["dollar_volume"].unstack("ticker")
    n_sessions = len(px.index)
    ret = px / px.shift(1) - 1.0
    stats = pd.DataFrame({
        "median_price": gate_px.median(),
        "median_dollar_volume": dv.median(),
        "ann_vol": np.log(px).diff().std() * np.sqrt(252),
        "coverage": px.notna().sum() / max(n_sessions, 1),
        "max_abs_return": ret.abs().max(),
    })
    eligible = (
        (stats["coverage"] >= min_coverage)
        & (stats["median_price"] >= min_price)
        & (stats["median_dollar_volume"] >= min_dollar_volume)
        & (stats["ann_vol"] >= min_ann_vol)
    )
    if max_abs_return is not None:
        eligible &= stats["max_abs_return"].fillna(np.inf) <= max_abs_return
    if exclude is not None:
        eligible &= ~stats.index.isin(set(exclude))
    stats["eligible"] = eligible
    if top_n is not None:
        keep = stats[stats["eligible"]].nlargest(top_n, "median_dollar_volume").index
        stats["eligible"] = stats.index.isin(keep)
    return stats.sort_values("median_dollar_volume", ascending=False)
