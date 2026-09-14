from __future__ import annotations
import logging
from typing import Iterable, List, Tuple, Union

import pandas as pd

__all__ = ["download_history_openbb", "normalize_symbol"]

# Providers that spell share classes with a dash (BRK-B) where index constituent
# lists use a dot (BRK.B). Other providers receive the ticker unchanged.
_DASH_SHARE_CLASS_PROVIDERS = {"yfinance"}


def normalize_symbol(ticker: str, provider: str) -> str:
    """
    Alternate spelling of a ticker to *retry* with when the provider returns nothing.

    S&P / Nasdaq constituent lists write share classes as ``BRK.B`` / ``BF.B``;
    yfinance only recognises ``BRK-B`` / ``BF-B``. But yfinance also uses dots for
    exchange suffixes (``VOD.L``, ``RY.TO``) that must keep the dot, so the spelling
    cannot be decided up front: the loader requests the ticker exactly as given
    first and falls back to this spelling only if that returns no data. The caller
    keeps the original ticker as the key either way.
    """
    t = str(ticker).strip().upper()
    if provider in _DASH_SHARE_CLASS_PROVIDERS:
        return t.replace(".", "-")
    return t


def _fetch(obb, symbol, start_date, end_date, provider):
    """One provider request; None on error or empty result."""
    try:
        df = obb.equity.price.historical(
            symbol=symbol, start_date=start_date, end_date=end_date, provider=provider,
        )
    except Exception:
        return None
    return None if (df is None or len(df) == 0) else df


def download_history_openbb(
    tickers: Union[Iterable[str], pd.Series, pd.Index],
    start_date: Union[str, pd.Timestamp],
    end_date:   Union[str, pd.Timestamp],
    *,
    provider: str = "yfinance",
    show_progress: bool = True,
    silence_logs: bool = True,
    return_failed: bool = False,
):
    """
    Download historical OHLCV for multiple tickers via OpenBB and stitch into one DataFrame.

    OpenBB is imported lazily here (not at module import) so that the rest of
    ``pairs.market_data`` — e.g. the Polygon lake loader — works without it.

    Returns:
        df                      (pd.DataFrame): concatenated results with MultiIndex (ticker, datetime).
        (optionally) failed     (List[str])   : tickers that returned no data or errored.
    """
    from tqdm import tqdm
    try:
        from openbb import obb  # OpenBB v4
    except ImportError as e:
        raise ImportError(
            "OpenBB not available. Install it with `pip install openbb` (v4) "
            "or `pip install -e '.[notebooks]'`."
        ) from e

    # Return plain DataFrames rather than OBBject wrappers
    obb.user.preferences.output_type = "dataframe"

    # Silence noisy logs if requested
    if silence_logs:
        for lg in ["openbb_core", "openbb_yfinance", "openbb", "yfinance"]:
            logging.getLogger(lg).setLevel(logging.CRITICAL)

    # Clean ticker list
    tickers = (
        pd.Index(pd.Series(list(tickers), dtype="object"))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    frames: List[Tuple[str, pd.DataFrame]] = []
    failed: List[str] = []

    iterator = tqdm(tickers, desc="Download", disable=not show_progress)
    for tkr in iterator:
        if show_progress:
            iterator.set_description(tkr)

        # Request the ticker verbatim; retry with the provider's share-class spelling
        # (BRK.B -> BRK-B) only when that differs and the first request came back empty.
        df_new = _fetch(obb, tkr, start_date, end_date, provider)
        if df_new is None:
            alt = normalize_symbol(tkr, provider)
            if alt != tkr:
                df_new = _fetch(obb, alt, start_date, end_date, provider)
        if df_new is None:
            failed.append(tkr)
            continue

        # Ensure a DatetimeIndex named consistently ('date' on some providers)
        df_new = df_new.copy()
        if not isinstance(df_new.index, pd.DatetimeIndex):
            df_new.index = pd.to_datetime(df_new.index, errors="coerce")
        df_new.index = df_new.index.rename("datetime")

        frames.append((tkr, df_new))

    if frames:
        # Concatenate using keys to form MultiIndex with ticker first, datetime second
        keys = [k for k, _ in frames]
        objs = [df for _, df in frames]
        df = pd.concat(objs, keys=keys, names=["ticker", "datetime"])
        df = df.sort_index(level=["ticker", "datetime"])
    else:
        # Return an empty frame with the correct MultiIndex shape and common OHLCV columns
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "datetime"])
        df = pd.DataFrame(index=empty_index, columns=["open", "high", "low", "close", "adj_close", "volume"])

    if return_failed:
        return df, failed
    return df
