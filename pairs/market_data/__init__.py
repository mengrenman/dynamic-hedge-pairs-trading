# pairs/market_data/__init__.py
from __future__ import annotations
from typing import Any

__all__ = [
    "load_prices", "load_polygon_lake", "download_openbb",
    "load_minute_bars", "detect_lake_layout", "nyse_early_closes", "summarize_sessions",
    "load_daily_bars", "detect_day_lake_layout", "recover_dividends", "liquidity_screen",
    "CostSpec", "measure_ticker_window_costs", "pair_fold_costs",
]

# --- polygon lake: ---
from .polygon_lake import load_polygonio_lake as load_polygon_lake

# --- adjusted minute lake (ticker or market layout), regular-session grid: ---
from .minute_bars import load_minute_bars, detect_lake_layout, nyse_early_closes, summarize_sessions

# --- adjusted day lake (ticker or market layout), explicit price basis: ---
from .daily_bars import load_daily_bars, detect_day_lake_layout, recover_dividends, liquidity_screen

# --- execution costs measured on the minute lake: ---
from .execution_costs import CostSpec, measure_ticker_window_costs, pair_fold_costs

# --- openbb: ---
from .openbb_history import download_history_openbb as download_openbb


def load_prices(source: str, *args: Any, **kwargs: Any):
    src = source.lower()
    if src == "polygon":
        return load_polygon_lake(*args, **kwargs)
    if src == "openbb":
        return download_openbb(*args, **kwargs)
    raise ValueError(f"Unknown source: {source!r}. Expected 'polygon' or 'openbb'.")
