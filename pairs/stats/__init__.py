# pairs/stats/__init__.py
"""
Statistics utilities for the pairs package:
- Cointegration screening (EG/Johansen)
- Stationarity diagnostics (ADF/KPSS, half-life)
- Basic transforms (z-score utilities)
- Portfolio analytics (cross-pair correlations, diversification, weights)
- Hedge ratio stability tests (CUSUM, rolling drift)
- Intraday microstructure diagnostics (Roll spread, signature plot, Epps effect)
- Time-varying cointegration (Eroglu-Miller-Yigit) and the dynamic-hedge gate
"""

from .tv_cointegration import (
    TVCointModel,
    fit_tvssm,
    bootstrap_test,
    classify_cointegration,
    recommend_hedge,
    HedgeVerdict,
)
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
from .microstructure import (
    roll_spread,
    realized_variance_signature,
    epps_correlation,
    autocorr_by_interval,
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
    "roll_spread",
    "TVCointModel",
    "fit_tvssm",
    "bootstrap_test",
    "classify_cointegration",
    "recommend_hedge",
    "HedgeVerdict",
    "realized_variance_signature",
    "epps_correlation",
    "autocorr_by_interval",
]
