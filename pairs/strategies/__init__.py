# pairs/strategies/__init__.py
from .signals import estimate_halflife_window, zscore_from_spread, generate_pair_signals, session_masks
from .evaluate import evaluate_pair_signals, market_impact_bps
from .spread_accounting import decompose_spread_pnl, turnover_and_fees
from .avellaneda_lee import (
    assign_sector_etf,
    trading_time_factor,
    etf_residuals,
    ou_fit,
    s_score,
    bang_bang_update,
    positions_from_state,
)

__all__ = [
    "estimate_halflife_window",
    "zscore_from_spread",
    "generate_pair_signals",
    "session_masks",
    "evaluate_pair_signals",
    "market_impact_bps",
    "decompose_spread_pnl",
    "turnover_and_fees",
    "assign_sector_etf",
    "trading_time_factor",
    "etf_residuals",
    "ou_fit",
    "s_score",
    "bang_bang_update",
    "positions_from_state",
]
