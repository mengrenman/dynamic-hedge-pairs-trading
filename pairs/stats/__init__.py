# pairs/stats/__init__.py
"""
Statistics utilities for the pairs package:
- Cointegration screening (EG/Johansen)
- Stationarity diagnostics (ADF/KPSS, half-life)
- Basic transforms (z-score utilities)
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
]
