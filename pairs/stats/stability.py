# pairs/stats/stability.py
"""
Hedge ratio (beta) stability diagnostics for pairs trading.

Exports:
- cusum_beta_stability(beta, ...) : CUSUM test for structural level shifts in β
- rolling_beta_drift(beta, ...) : rolling volatility test for β drift
- summarize_hedge_ratio_stability(kf_results, ...) : run both tests for all pairs
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "cusum_beta_stability",
    "rolling_beta_drift",
    "summarize_hedge_ratio_stability",
]

# Critical value lookup for CUSUM test (Brown-Durbin-Evans bounds)
_CUSUM_CRIT = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}


def cusum_beta_stability(
    beta: pd.Series,
    *,
    alpha: float = 0.05,
) -> dict:
    """
    CUSUM test for a structural level shift in the hedge ratio (β) series.

    Algorithm (mean-shift model, no regression regressors):
        S_t = Σ_{i=1}^{t} (β_i − β̄) / σ_β,   t = 1 … T
        stat = max|S_t| / sqrt(T)

    Critical values (Brown-Durbin-Evans):
        α = 0.10 → 1.22,   α = 0.05 → 1.36,   α = 0.01 → 1.63

    Parameters
    ----------
    beta : pd.Series — time-series of Kalman hedge ratios
    alpha : significance level; must be one of {0.10, 0.05, 0.01}

    Returns
    -------
    dict with keys:
        cusum_stat   : float  — max|S| / sqrt(T)
        critical_val : float  — critical value at given alpha
        is_stable    : bool   — True if cusum_stat < critical_val
        alpha        : float  — significance level used
        cusum_series : pd.Series — full CUSUM path (same index as non-NaN beta)
    """
    if alpha not in _CUSUM_CRIT:
        raise ValueError(f"alpha must be one of {sorted(_CUSUM_CRIT)}; got {alpha}")

    crit = _CUSUM_CRIT[alpha]
    beta_clean = pd.Series(beta, dtype=float).dropna()

    if len(beta_clean) < 2:
        nan_series = pd.Series(dtype=float)
        return {
            "cusum_stat": float("nan"),
            "critical_val": crit,
            "is_stable": True,   # insufficient data → don't flag
            "alpha": alpha,
            "cusum_series": nan_series,
        }

    arr = beta_clean.values
    mu = arr.mean()
    sigma = arr.std(ddof=1)

    if sigma == 0.0 or not np.isfinite(sigma):
        # Constant beta — perfectly stable; S is identically 0
        cusum_path = np.zeros(len(arr))
        stat = 0.0
    else:
        cusum_path = np.cumsum(arr - mu) / sigma
        stat = float(np.max(np.abs(cusum_path)) / np.sqrt(len(arr)))

    return {
        "cusum_stat": stat,
        "critical_val": crit,
        "is_stable": bool(stat < crit),
        "alpha": alpha,
        "cusum_series": pd.Series(cusum_path, index=beta_clean.index, name="cusum"),
    }


def rolling_beta_drift(
    beta: pd.Series,
    *,
    window: int = 63,
    threshold_sigma: float = 2.0,
) -> dict:
    """
    Rolling volatility test for β drift.

    Computes the rolling standard deviation of β over `window` bars, normalized
    by the global (full-sample) standard deviation.

    Flags bars where rolling_std / global_std > threshold_sigma.

    Parameters
    ----------
    beta : pd.Series — time-series of Kalman hedge ratios
    window : rolling window length in bars (default 63 ≈ one quarter)
    threshold_sigma : ratio threshold above which drift is flagged (default 2.0)

    Returns
    -------
    dict with keys:
        max_roll_std_ratio : float          — max(rolling_std / global_std)
        threshold_sigma    : float          — threshold used
        is_stable          : bool           — True if max_roll_std_ratio <= threshold_sigma
        flagged_dates      : pd.DatetimeIndex — bars where ratio exceeded threshold
        roll_std_series    : pd.Series      — full rolling std ratio path (same index as beta)
    """
    beta_s = pd.Series(beta, dtype=float)
    global_std = float(beta_s.dropna().std(ddof=1))

    if global_std == 0.0 or not np.isfinite(global_std) or beta_s.dropna().empty:
        empty_idx = pd.DatetimeIndex([]) if isinstance(beta_s.index, pd.DatetimeIndex) else pd.Index([])
        return {
            "max_roll_std_ratio": 0.0,
            "threshold_sigma": threshold_sigma,
            "is_stable": True,
            "flagged_dates": empty_idx,
            "roll_std_series": pd.Series(np.zeros(len(beta_s)), index=beta_s.index, name="roll_std_ratio"),
        }

    roll_std = beta_s.rolling(window, min_periods=max(2, window // 2)).std(ddof=1)
    ratio = roll_std / global_std

    max_ratio = float(ratio.max()) if ratio.notna().any() else 0.0
    is_stable = max_ratio <= threshold_sigma

    flagged_mask = ratio > threshold_sigma
    if isinstance(beta_s.index, pd.DatetimeIndex):
        flagged_dates = beta_s.index[flagged_mask.fillna(False)]
    else:
        flagged_dates = pd.Index(beta_s.index[flagged_mask.fillna(False)])

    return {
        "max_roll_std_ratio": max_ratio,
        "threshold_sigma": threshold_sigma,
        "is_stable": is_stable,
        "flagged_dates": flagged_dates,
        "roll_std_series": ratio.rename("roll_std_ratio"),
    }


def summarize_hedge_ratio_stability(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    *,
    cusum_alpha: float = 0.05,
    roll_window: int = 63,
    roll_threshold_sigma: float = 2.0,
    min_obs: int = 30,
) -> pd.DataFrame:
    """
    Run both stability tests for every pair in kf_results.

    Parameters
    ----------
    kf_results : dict mapping (ticker1, ticker2) → DataFrame with 'beta' column
    cusum_alpha : significance level for CUSUM test (default 0.05)
    roll_window : rolling window for drift test in bars (default 63)
    roll_threshold_sigma : ratio threshold for drift flagging (default 2.0)
    min_obs : minimum non-NaN observations required to run tests (default 30)

    Returns
    -------
    DataFrame indexed by (ticker1, ticker2) with columns:
        ["cusum_stat","cusum_critical","cusum_stable",
         "roll_max_ratio","roll_stable","n_flagged_bars","overall_stable"]

    Sorted ascending by overall_stable (unstable pairs first), then by cusum_stat descending.
    """
    records = []
    for (k1, k2), df in kf_results.items():
        if "beta" not in df.columns:
            continue

        beta = df["beta"].dropna()
        if len(beta) < min_obs:
            records.append({
                "ticker1": k1,
                "ticker2": k2,
                "cusum_stat": float("nan"),
                "cusum_critical": _CUSUM_CRIT.get(cusum_alpha, 1.36),
                "cusum_stable": True,
                "roll_max_ratio": float("nan"),
                "roll_stable": True,
                "n_flagged_bars": 0,
                "overall_stable": True,
            })
            continue

        c = cusum_beta_stability(beta, alpha=cusum_alpha)
        r = rolling_beta_drift(beta, window=roll_window, threshold_sigma=roll_threshold_sigma)

        overall = bool(c["is_stable"]) and bool(r["is_stable"])
        records.append({
            "ticker1": k1,
            "ticker2": k2,
            "cusum_stat": c["cusum_stat"],
            "cusum_critical": c["critical_val"],
            "cusum_stable": c["is_stable"],
            "roll_max_ratio": r["max_roll_std_ratio"],
            "roll_stable": r["is_stable"],
            "n_flagged_bars": len(r["flagged_dates"]),
            "overall_stable": overall,
        })

    if not records:
        return pd.DataFrame(columns=[
            "cusum_stat", "cusum_critical", "cusum_stable",
            "roll_max_ratio", "roll_stable", "n_flagged_bars", "overall_stable",
        ])

    out = pd.DataFrame(records)
    out = out.set_index(["ticker1", "ticker2"])
    out = out.sort_values(["overall_stable", "cusum_stat"], ascending=[True, False])
    return out
