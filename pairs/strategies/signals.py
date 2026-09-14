# pairs/strategies/signals.py
"""
Signal generation utilities for pairs trading.

Exports:
- estimate_halflife_window(...): convert spread half-life → a sensible rolling window
- zscore_from_spread(...): build z-scores (rolling / EWM / robust)
- generate_pair_signals(...): entry/exit/stop and position sizing

Notes
-----
`generate_pair_signals` expects a DataFrame with at least:
- 'resid' : spread (residual) series
- 'beta'  : hedge ratio at each bar (from Kalman states)
- 'P1','P2' : leg prices aligned to the same index (used for sizing)

Typical workflow:
states_df  = Kalman states with ['alpha','beta','y_hat','resid']
df_pair    = states_df.join({"P1": P1_series, "P2": P2_series})
signals    = generate_pair_signals(df_pair, ...)
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# Use the canonical half-life estimator from stats module
from pairs.stats.stationarity import estimate_halflife as _hl_float

__all__ = [
    "estimate_halflife_window",
    "zscore_from_spread",
    "generate_pair_signals",
]

# ---- 1) Half-life → window helper -------------------------------------------
def estimate_halflife_window(spread: pd.Series, min_win: int = 30, max_win: int = 252) -> int:
    """
    Map spread half-life (float) to a practical rolling window length (int).
    Uses 3 × half-life, clipped to [min_win, max_win].
    """
    s = pd.Series(spread).dropna()
    if s.empty:
        return int(min_win)
    hl = float(_hl_float(s))
    if not np.isfinite(hl) or hl <= 0:
        return int(min_win)
    win = int(np.clip(3.0 * hl, min_win, max_win))
    return int(win)

# ---- 2) Z-score builders (rolling / EWM / robust) ---------------------------
def zscore_from_spread(
    spread: pd.Series,
    method: str = "rolling",              # "rolling" | "ewm" | "robust"
    window: Optional[int] = None,
    halflife: Optional[float] = None,
) -> pd.Series:
    """
    Compute a z-score for a spread series using different estimators of mean/vol.

    - method="rolling": simple rolling mean/std with window (defaults to 3×HL)
    - method="ewm":     exponentially-weighted mean/std with half-life
                        (defaults to max(10, HL))
    - method="robust":  rolling median + MAD (scaled to sigma) over 'window'
                        (defaults to 3×HL)

    Returns a Series aligned to the input index.
    """
    if method not in ("rolling", "ewm", "robust"):
        raise ValueError(f"method must be one of 'rolling', 'ewm', 'robust'; got {method!r}")

    s = pd.Series(spread, dtype=float)

    if method == "ewm":
        if halflife is None:
            hl = _hl_float(s)
            halflife = float(hl) if np.isfinite(hl) and hl > 0 else 10.0
        mu = s.ewm(halflife=halflife, adjust=False).mean()
        sd = s.ewm(halflife=halflife, adjust=False).std(bias=False)

    elif method == "robust":
        if window is None:
            window = estimate_halflife_window(s)
        med = s.rolling(window, min_periods=window).median()
        mad = (s - med).abs().rolling(window, min_periods=window).median()
        sd  = 1.4826 * mad   # ≈ robust sigma
        mu  = med

    else:  # "rolling"
        if window is None:
            window = estimate_halflife_window(s)
        mu = s.rolling(window, min_periods=window).mean()
        sd = s.rolling(window, min_periods=window).std(ddof=0)

    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)

# ---- 3) Signal generator ----------------------------------------------------
def generate_pair_signals(
    df_pair: pd.DataFrame,
    *,
    z_method: str = "rolling",
    z_window: int | None = None,
    z_halflife: float | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    capital_per_pair: float = 10_000.0,
    max_hold_bars: int | None = None,
    cooldown_bars: int = 0,
    exec_lag: int = 1,              # execute on the next bar by default (no look-ahead)
) -> pd.DataFrame:
    """
    Generate entry/exit/stop signals and target sizes for a pair.

    Parameters
    ----------
    df_pair : DataFrame with index as timestamps and columns:
        - 'resid'  : spread (residual)
        - 'beta'   : hedge ratio at each bar
        - 'P1','P2': aligned prices for the two legs
    z_method : {'rolling','ewm','robust'}
    z_window : optional int window for 'rolling'/'robust' (defaults to 3×HL)
    z_halflife : optional float half-life for 'ewm' (defaults to max(10, HL))
    z_entry, z_exit, z_stop : thresholds on |z|
    capital_per_pair : notional used to size N1/N2 dollar-neutral targets
    max_hold_bars : optional cap on holding period
    cooldown_bars : bars to wait after a flatting event before re-entry
    exec_lag : shift signals forward by this many bars to emulate next-bar execution

    Returns
    -------
    DataFrame with columns (aligned to execution bar):
      ['z','pos','n1','n2','entry','exit','stop']
    """
    required = {"resid", "beta", "P1", "P2"}
    missing = required - set(df_pair.columns)
    if missing:
        raise ValueError(f"df_pair must contain columns {sorted(required)}; missing: {sorted(missing)}")

    df = df_pair.copy()
    df["z"] = zscore_from_spread(df["resid"], method=z_method, window=z_window, halflife=z_halflife)

    n = len(df)
    pos_dec  = np.zeros(n, dtype=int)     # decision at time t (pre-execution)
    n1_dec   = np.zeros(n, dtype=float)
    n2_dec   = np.zeros(n, dtype=float)
    ent_dec  = np.zeros(n, dtype=bool)
    exit_dec = np.zeros(n, dtype=bool)
    stop_dec = np.zeros(n, dtype=bool)

    pos = 0
    hold = 0
    cooldown = 0

    for t in range(n):
        z = df["z"].iloc[t]
        P1 = df["P1"].iloc[t]
        P2 = df["P2"].iloc[t]
        beta_t = df["beta"].iloc[t]

        # Not enough info: stay flat
        if (np.isnan(z) or np.isnan(P1) or np.isnan(P2) or np.isnan(beta_t) or P1 <= 0 or P2 <= 0):
            pos = 0; hold = 0
            pos_dec[t] = 0; n1_dec[t] = 0.0; n2_dec[t] = 0.0
            continue

        if cooldown > 0:
            cooldown -= 1
        can_enter = (cooldown == 0)

        # Decide using info up to and including t
        if pos == 0 and can_enter:
            if z <= -z_entry:
                pos = +1; hold = 1; ent_dec[t] = True
            elif z >= +z_entry:
                pos = -1; hold = 1; ent_dec[t] = True
        elif pos == +1:
            hold += 1
            if abs(z) <= z_exit:
                pos = 0; exit_dec[t] = True; hold = 0; cooldown = cooldown_bars
            elif z <= -z_stop:
                pos = 0; stop_dec[t] = True; hold = 0; cooldown = cooldown_bars
            elif max_hold_bars and hold >= max_hold_bars:
                pos = 0; exit_dec[t] = True; hold = 0; cooldown = cooldown_bars
        elif pos == -1:
            hold += 1
            if abs(z) <= z_exit:
                pos = 0; exit_dec[t] = True; hold = 0; cooldown = cooldown_bars
            elif z >= +z_stop:
                pos = 0; stop_dec[t] = True; hold = 0; cooldown = cooldown_bars
            elif max_hold_bars and hold >= max_hold_bars:
                pos = 0; exit_dec[t] = True; hold = 0; cooldown = cooldown_bars

        pos_dec[t] = pos

        # Size targets with info at t (dollar-neutral)
        if pos != 0:
            scale = capital_per_pair / (P1 + abs(beta_t) * P2)
            n1_dec[t] = pos * scale
            n2_dec[t] = -pos * scale * beta_t
        else:
            n1_dec[t] = 0.0
            n2_dec[t] = 0.0

    # --- Execution alignment ---
    if exec_lag < 0:
        raise ValueError("exec_lag must be >= 0")
    pos_exe  = pd.Series(pos_dec).shift(exec_lag, fill_value=0).astype(int).values
    n1_exe   = pd.Series(n1_dec).shift(exec_lag, fill_value=0.0).values
    n2_exe   = pd.Series(n2_dec).shift(exec_lag, fill_value=0.0).values
    entry_ex = pd.Series(ent_dec).shift(exec_lag, fill_value=False).astype(bool).values
    exit_ex  = pd.Series(exit_dec).shift(exec_lag, fill_value=False).astype(bool).values
    stop_ex  = pd.Series(stop_dec).shift(exec_lag, fill_value=False).astype(bool).values

    out = pd.DataFrame(
        {
            "z": df["z"].values,   # contemporaneous z (not shifted)
            "pos": pos_exe,
            "n1": n1_exe,
            "n2": n2_exe,
            "entry": entry_ex,
            "exit": exit_ex,
            "stop": stop_ex,
        },
        index=df.index,
    )
    return out
