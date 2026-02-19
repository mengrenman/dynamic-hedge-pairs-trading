# pairs/stats/portfolio.py
"""
Portfolio-level analytics for multi-pair trading strategies.

Exports:
- pair_return_correlations(kf_results, ...) : N×N cross-pair spread-return correlation matrix
- portfolio_diversification_score(corr_matrix) : scalar diversification ratio
- suggest_position_weights(kf_results, corr_matrix, ...) : capital allocation weights per pair
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "pair_return_correlations",
    "portfolio_diversification_score",
    "suggest_position_weights",
]


def pair_return_correlations(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    *,
    min_overlap: int = 30,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlations of spread daily returns (Δresid) across pairs.

    Parameters
    ----------
    kf_results : dict mapping (ticker1, ticker2) → DataFrame with a 'resid' column
    min_overlap : minimum overlapping bars required; pairs with fewer overlap get NaN
    method : 'pearson' (default) or 'spearman'

    Returns
    -------
    Symmetric N×N DataFrame, index and columns labelled "T1/T2".
    Diagonal entries are 1.0 (or NaN if a pair's series is all NaN).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")

    # Build spread-return series dict: label → pd.Series of Δresid
    returns: dict[str, pd.Series] = {}
    for (k1, k2), df in kf_results.items():
        label = f"{k1}/{k2}"
        if "resid" not in df.columns:
            continue
        ret = df["resid"].diff().dropna()
        if not ret.empty:
            returns[label] = ret

    if len(returns) == 0:
        return pd.DataFrame()

    labels = list(returns.keys())

    # Align all series on a common DatetimeIndex
    combined = pd.concat(returns, axis=1)
    combined.columns = labels

    # Compute correlation matrix
    corr = combined.corr(method=method)

    # Zero out (NaN) entries where pairwise overlap < min_overlap
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                continue
            overlap = combined[[li, lj]].dropna().shape[0]
            if overlap < min_overlap:
                corr.loc[li, lj] = np.nan
                corr.loc[lj, li] = np.nan

    return corr


def portfolio_diversification_score(
    corr_matrix: pd.DataFrame,
) -> float:
    """
    Portfolio diversification ratio.

    Defined as 1 / mean(|off-diagonal correlations|).

    Range: 1 (all perfectly correlated) → ∞ (all uncorrelated).
    A score > 3 indicates good diversification across pairs.
    Returns np.nan if fewer than 2 pairs (no off-diagonal entries exist).

    Parameters
    ----------
    corr_matrix : square symmetric correlation DataFrame (output of pair_return_correlations)
    """
    n = len(corr_matrix)
    if n < 2:
        return float("nan")

    # Extract off-diagonal elements only
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr_matrix.values[mask]

    # Drop NaNs from incomplete pair overlaps
    off_diag = off_diag[~np.isnan(off_diag)]

    if len(off_diag) == 0:
        return float("nan")

    mean_abs_corr = float(np.mean(np.abs(off_diag)))
    if mean_abs_corr == 0.0:
        return float("inf")

    return 1.0 / mean_abs_corr


def suggest_position_weights(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    corr_matrix: pd.DataFrame,
    *,
    method: str = "inv_var",
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """
    Suggest capital allocation weights across pairs.

    Parameters
    ----------
    kf_results : dict mapping (ticker1, ticker2) → DataFrame with 'resid' column
    corr_matrix : output of pair_return_correlations (used to keep labels consistent)
    method : 'inv_var' (default) — weight ∝ 1/Var(Δresid), or 'equal'
    max_weight : maximum weight for any single pair (default 0.40); renormalized after clipping

    Returns
    -------
    DataFrame with columns:
      ["pair", "resid_var", "inv_var_weight", "weight", "suggested_capital_pct"]
    Sorted by weight descending.
    """
    if method not in ("inv_var", "equal"):
        raise ValueError(f"method must be 'inv_var' or 'equal'; got {method!r}")
    if not (0.0 < max_weight <= 1.0):
        raise ValueError(f"max_weight must be in (0, 1]; got {max_weight}")

    rows = []
    for (k1, k2), df in kf_results.items():
        label = f"{k1}/{k2}"
        if "resid" not in df.columns:
            var = float("nan")
        else:
            ret = df["resid"].diff().dropna()
            var = float(ret.var(ddof=1)) if len(ret) >= 2 else float("nan")
        rows.append({"pair": label, "resid_var": var})

    if not rows:
        return pd.DataFrame(columns=["pair", "resid_var", "inv_var_weight", "weight", "suggested_capital_pct"])

    result = pd.DataFrame(rows)

    # Compute raw weights
    valid = result["resid_var"].notna() & (result["resid_var"] > 0)

    if method == "inv_var":
        raw = np.where(valid, 1.0 / result["resid_var"].values, 0.0)
    else:  # equal
        raw = np.where(valid, 1.0, 0.0)

    total_raw = raw.sum()
    inv_var_weight = raw / total_raw if total_raw > 0 else raw

    # Clip to max_weight and renormalize iteratively (one-pass: clip then renorm)
    clipped = np.clip(inv_var_weight, 0.0, max_weight)
    clipped_sum = clipped.sum()
    weight = clipped / clipped_sum if clipped_sum > 0 else clipped

    result["inv_var_weight"] = inv_var_weight
    result["weight"] = weight
    result["suggested_capital_pct"] = weight * 100.0

    return result.sort_values("weight", ascending=False).reset_index(drop=True)
