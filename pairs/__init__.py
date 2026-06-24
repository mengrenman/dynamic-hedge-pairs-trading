# pairs/__init__.py
"""
pairs: Utilities for market data access, universes, plotting, statistics, strategies, and models.

Public entry points (lazy-loaded):
- Data:         load_prices(), load_polygon_lake(), download_openbb()
- Universe:     load_universe(), list_universes()
- Plotting:     plot_pair_legs_with_trades()
- Statistics:   find_cointegrated_pairs_executor(), find_cointegrated_pairs_dualgate(),
                benjamini_hochberg_fdr(),
                estimate_halflife(), test_spread_stationarity(),
                summarize_spread_stationarity_joblib(),
                pair_return_correlations(), portfolio_diversification_score(),
                suggest_position_weights(),
                cusum_beta_stability(), rolling_beta_drift(),
                summarize_hedge_ratio_stability()
- Strategies:   estimate_halflife_window(), zscore_from_spread(),
                generate_pair_signals(), evaluate_pair_signals(),
                market_impact_bps(),
                CircuitBreakerConfig, apply_circuit_breaker()
- Validation:   walk_forward_splits(), walk_forward_backtest()
- Models (opt): fit_kalman_hedge(), filter_kf_on_new(),
                continue_kalman_on_window(), continue_kalman_for_pairs_joblib()
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# ----------------------- Public API names -----------------------
__all__ = [
    "__version__",
    # data
    "load_prices",
    "load_polygon_lake",
    "download_openbb",
    # universes
    "load_universe",
    "list_universes",
    # plotting
    "plot_pair_legs_with_trades",
    # statistics (cointegration + FDR)
    "benjamini_hochberg_fdr",
    "find_cointegrated_pairs_executor",
    "find_cointegrated_pairs_dualgate",
    # statistics (stationarity)
    "estimate_halflife",
    "test_spread_stationarity",
    "summarize_spread_stationarity_joblib",
    # statistics (portfolio & stability)
    "pair_return_correlations",
    "portfolio_diversification_score",
    "suggest_position_weights",
    "cusum_beta_stability",
    "rolling_beta_drift",
    "summarize_hedge_ratio_stability",
    # strategies (signals & evaluation)
    "estimate_halflife_window",
    "zscore_from_spread",
    "generate_pair_signals",
    "evaluate_pair_signals",
    "market_impact_bps",
    # strategies (circuit breaker)
    "CircuitBreakerConfig",
    "apply_circuit_breaker",
    # validation
    "walk_forward_splits",
    "walk_forward_backtest",
    # models (Kalman) appended conditionally below
]

# ---- version (works in editable installs too) ----
try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError  # py3.8+
except Exception:  # pragma: no cover
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # fallback

try:
    __version__ = _pkg_version("pairs")
except PackageNotFoundError:  # running from source without installed metadata
    __version__ = "0+unknown"

# ----------------------- Lazy attribute loader -----------------------
# Keep imports cheap and centralized.
_LAZY_MAP = {
    # market data (re-exported from pairs.market_data)
    "load_prices": ("pairs.market_data", "load_prices"),
    "load_polygon_lake": ("pairs.market_data", "load_polygon_lake"),
    "download_openbb": ("pairs.market_data", "download_openbb"),
    # universes
    "load_universe": ("pairs.universes", "load_universe"),
    "list_universes": ("pairs.universes", "list_universes"),
    # plotting (modules may be added later; lazy import avoids hard dependency)
    "plot_pair_legs_with_trades": ("pairs.plotting.pair_trades", "plot_pair_legs_with_trades"),
    # statistics (cointegration + FDR)
    "benjamini_hochberg_fdr": ("pairs.stats.cointegration", "benjamini_hochberg_fdr"),
    "find_cointegrated_pairs_executor": ("pairs.stats.cointegration", "find_cointegrated_pairs_executor"),
    "find_cointegrated_pairs_dualgate": ("pairs.stats.cointegration", "find_cointegrated_pairs_dualgate"),
    # statistics (stationarity)
    "estimate_halflife": ("pairs.stats.stationarity", "estimate_halflife"),
    "test_spread_stationarity": ("pairs.stats.stationarity", "test_spread_stationarity"),
    "summarize_spread_stationarity_joblib": ("pairs.stats.stationarity", "summarize_spread_stationarity_joblib"),
    # statistics (portfolio & stability)
    "pair_return_correlations": ("pairs.stats.portfolio", "pair_return_correlations"),
    "portfolio_diversification_score": ("pairs.stats.portfolio", "portfolio_diversification_score"),
    "suggest_position_weights": ("pairs.stats.portfolio", "suggest_position_weights"),
    "cusum_beta_stability": ("pairs.stats.stability", "cusum_beta_stability"),
    "rolling_beta_drift": ("pairs.stats.stability", "rolling_beta_drift"),
    "summarize_hedge_ratio_stability": ("pairs.stats.stability", "summarize_hedge_ratio_stability"),
    # strategies (signals & evaluation)
    "estimate_halflife_window": ("pairs.strategies.signals", "estimate_halflife_window"),
    "zscore_from_spread": ("pairs.strategies.signals", "zscore_from_spread"),
    "generate_pair_signals": ("pairs.strategies.signals", "generate_pair_signals"),
    "evaluate_pair_signals": ("pairs.strategies.evaluate", "evaluate_pair_signals"),
    "market_impact_bps": ("pairs.strategies.evaluate", "market_impact_bps"),
    # strategies (circuit breaker)
    "CircuitBreakerConfig": ("pairs.strategies.circuit_breaker", "CircuitBreakerConfig"),
    "apply_circuit_breaker": ("pairs.strategies.circuit_breaker", "apply_circuit_breaker"),
    # validation
    "walk_forward_splits": ("pairs.validation.walk_forward", "walk_forward_splits"),
    "walk_forward_backtest": ("pairs.validation.walk_forward", "walk_forward_backtest"),
}

# Optionally expose models if present without importing now.
# This keeps import time light but provides a clean top-level API when available.
try:
    import importlib.util as _ilu
    if _ilu.find_spec("pairs.models.kalman") is not None:
        _LAZY_MAP.update({
            "fit_kalman_hedge": ("pairs.models.kalman", "fit_kalman_hedge"),
            "filter_kf_on_new": ("pairs.models.kalman", "filter_kf_on_new"),
            "continue_kalman_on_window": ("pairs.models.kalman", "continue_kalman_on_window"),
            "continue_kalman_for_pairs_joblib": ("pairs.models.kalman", "continue_kalman_for_pairs_joblib"),
        })
        __all__ += [
            "fit_kalman_hedge",
            "filter_kf_on_new",
            "continue_kalman_on_window",
            "continue_kalman_for_pairs_joblib",
        ]
except Exception:  # pragma: no cover
    pass


def __getattr__(name: str):
    """Lazy re-exports for public API symbols."""
    target = _LAZY_MAP.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    import importlib
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    globals()[name] = obj  # cache for future accesses
    return obj


# ---- Static imports for type checkers (no runtime cost) ----
if TYPE_CHECKING:  # pragma: no cover
    from .market_data import load_prices, load_polygon_lake, download_openbb
    from .universes import load_universe, list_universes
    from .plotting.pair_trades import plot_pair_legs_with_trades
    from .stats.cointegration import (
        benjamini_hochberg_fdr,
        find_cointegrated_pairs_executor,
        find_cointegrated_pairs_dualgate,
    )
    from .stats.stationarity import (
        estimate_halflife,
        test_spread_stationarity,
        summarize_spread_stationarity_joblib,
    )
    from .stats.portfolio import (
        pair_return_correlations,
        portfolio_diversification_score,
        suggest_position_weights,
    )
    from .stats.stability import (
        cusum_beta_stability,
        rolling_beta_drift,
        summarize_hedge_ratio_stability,
    )
    from .strategies.signals import (
        estimate_halflife_window,
        zscore_from_spread,
        generate_pair_signals,
    )
    from .strategies.evaluate import evaluate_pair_signals, market_impact_bps
    from .strategies.circuit_breaker import CircuitBreakerConfig, apply_circuit_breaker
    from .validation.walk_forward import walk_forward_splits, walk_forward_backtest
    try:
        from .models.kalman import (
            fit_kalman_hedge,
            filter_kf_on_new,
            continue_kalman_on_window,
            continue_kalman_for_pairs_joblib,
        )
    except Exception:
        pass
