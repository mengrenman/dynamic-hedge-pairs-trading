# pairs/stats/transforms.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

__all__ = ["zscore", "Z", "rolling_zscore", "robust_z"]

def zscore(s: pd.Series, ddof: int = 0) -> pd.Series:
    """
    Standardize a series to zero mean and unit variance (population by default).
    Returns 0.0 where the std is 0 or not finite.
    """
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean()
    sd = s.std(ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd

# Back-compat alias matching your notebook’s name
def Z(s: pd.Series) -> pd.Series:
    return zscore(s, ddof=0)

def rolling_zscore(
    s: pd.Series, window: int, *, min_periods: Optional[int] = None, ddof: int = 0
) -> pd.Series:
    """
    Rolling z-score with windowed mean/std.
    Fills non-finite values (NaN/inf or zero-std windows) with 0.0 to match zscore() behavior.
    """
    s = pd.to_numeric(s, errors="coerce")
    r = s.rolling(window=window, min_periods=min_periods or window)
    m = r.mean()
    sd = r.std(ddof=ddof)
    out = (s - m) / sd
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def robust_z(s: pd.Series) -> pd.Series:
    """
    Robust z-score using median and MAD.

    z_robust = 0.6745 * (x - median) / MAD
    (Equivalent to (x - median) / (1.4826 * MAD); the factor makes it N(0,1) under normality.)

    If MAD is 0 or non-finite, returns 0.0 for all entries.
    """
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad
