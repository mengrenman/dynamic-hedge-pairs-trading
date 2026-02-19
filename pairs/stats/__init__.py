# pairs/stats/__init__.py
"""
Statistics utilities for the pairs package:
- Cointegration screening (EG/Johansen)
- Stationarity diagnostics (ADF/KPSS, half-life)
- Basic transforms (z-score utilities)
- Portfolio analytics (cross-pair correlations, diversification, weights)
- Hedge ratio stability tests (CUSUM, rolling drift)
"""

from .cointegration import (
    benjamini_hochberg_fdr,
    find_cointegrated_pairs_executor,
    find_cointegrated_pairs_dualgate,
)
from .stationarity import (
    estimate_halflife,
    test_spread_stationarity,
    summarize_spread_stationarity_joblib,
)
from .transforms import zscore, Z, rolling_zscore
from .portfolio import (
    pair_return_correlations,
    portfolio_diversification_score,
    suggest_position_weights,
)
from .stability import (
    cusum_beta_stability,
    rolling_beta_drift,
    summarize_hedge_ratio_stability,
)

__all__ = [
    "benjamini_hochberg_fdr",
    "find_cointegrated_pairs_executor",
    "find_cointegrated_pairs_dualgate",
    "estimate_halflife",
    "test_spread_stationarity",
    "summarize_spread_stationarity_joblib",
    "zscore",
    "Z",
    "rolling_zscore",
    "pair_return_correlations",
    "portfolio_diversification_score",
    "suggest_position_weights",
    "cusum_beta_stability",
    "rolling_beta_drift",
    "summarize_hedge_ratio_stability",
]
