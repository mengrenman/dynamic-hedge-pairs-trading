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
    "session_masks",
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
    *,
    history: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute a z-score for a spread series using different estimators of mean/vol.

    - method="rolling": simple rolling mean/std with window (defaults to 3×HL)
    - method="ewm":     exponentially-weighted mean/std with half-life
                        (defaults to max(10, HL))
    - method="robust":  rolling median + MAD (scaled to sigma) over 'window'
                        (defaults to 3×HL)

    history : optional Series of *past* spread values (e.g. the training-window
        residual) that precede `spread` in time. When given, (i) the window /
        half-life is estimated from `history` instead of from `spread` itself —
        scoring a test window with a look-back chosen from that same window is a
        look-ahead — and (ii) `history` warms up the rolling statistics so the first
        bars of `spread` have a full look-back instead of NaN. Only the entries
        aligned with `spread` are returned. Standard rolling semantics apply across
        the join: a NaN within the last `window` bars of `history` leaves the
        z-score undefined for up to `window` bars into `spread`. Note that the
        "robust" estimator chains two rolling medians (the level, then the MAD of the
        deviations from it), so it needs **2 × window − 1** bars of history for the
        first bar of `spread` to be defined; "rolling" and "ewm" need `window` (or a
        few half-lives). Pass a generous history rather than exactly `window` bars.

    Returns a Series aligned to the input index.
    """
    if method not in ("rolling", "ewm", "robust"):
        raise ValueError(f"method must be one of 'rolling', 'ewm', 'robust'; got {method!r}")

    s = pd.Series(spread, dtype=float)
    if history is not None:
        h = pd.Series(history, dtype=float)
        ref  = h                                      # look-back chosen from the past only
        full = pd.concat([h, s], ignore_index=True)   # positional; indices may overlap
    else:
        ref, full = s, s

    if method == "ewm":
        if halflife is None:
            hl = _hl_float(ref)
            halflife = float(hl) if np.isfinite(hl) and hl > 0 else 10.0
        mu = full.ewm(halflife=halflife, adjust=False).mean()
        sd = full.ewm(halflife=halflife, adjust=False).std(bias=False)

    elif method == "robust":
        if window is None:
            window = estimate_halflife_window(ref)
        med = full.rolling(window, min_periods=window).median()
        mad = (full - med).abs().rolling(window, min_periods=window).median()
        sd  = 1.4826 * mad   # ≈ robust sigma
        mu  = med

    else:  # "rolling"
        if window is None:
            window = estimate_halflife_window(ref)
        mu = full.rolling(window, min_periods=window).mean()
        sd = full.rolling(window, min_periods=window).std(ddof=0)

    z = ((full - mu) / sd).replace([np.inf, -np.inf], np.nan)
    if history is not None:
        z = z.iloc[len(full) - len(s):]
        z.index = s.index
    return z

# ---- 3) Signal generator ----------------------------------------------------
def generate_pair_signals(
    df_pair: pd.DataFrame,
    *,
    z_method: str = "rolling",
    z_window: int | None = None,
    z_halflife: float | None = None,
    z_history: pd.Series | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    capital_per_pair: float = 10_000.0,
    max_hold_bars: int | None = None,
    cooldown_bars: int = 0,
    exec_lag: int = 1,              # execute on the next bar by default (no look-ahead)
    force_flat: pd.Series | np.ndarray | None = None,
    block_entry: pd.Series | np.ndarray | None = None,
    initial_position: tuple[int, float, float] | None = None,
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
    z_history : optional Series of past spread values (e.g. the training-window
        residual) preceding df_pair in time; the z-score look-back is then chosen
        from this history and warmed up on it, so an out-of-sample window never
        informs its own z-score (see zscore_from_spread). Strongly recommended
        for walk-forward / OOS evaluation.
    z_entry, z_exit, z_stop : thresholds on |z|. Entries are taken only while
        ``z_entry <= |z| < z_stop``; at or beyond the stop level no position is opened
        (it would be stopped out at once), so after a stop the spread has to come back
        inside the band before the pair is traded again.
    capital_per_pair : notional used to size N1/N2 dollar-neutral targets
    max_hold_bars : optional cap on holding period
    cooldown_bars : bars to wait after a flatting event before re-entry
    exec_lag : shift signals forward by this many bars to emulate next-bar execution
    force_flat : optional boolean mask aligned to df_pair. Where True the decision is "flat":
        an open position is closed (logged as an exit) and no entry is taken. Session rules
        for intraday bars are built from it, e.g. the last ``exec_lag`` bars of each session so
        that nothing is carried overnight after the execution lag is applied.
    block_entry : optional boolean mask aligned to df_pair. Where True no new position is
        opened; an open position is managed as usual (e.g. no entries late in the session).
    initial_position : optional ``(pos, n1, n2)`` — the executed holdings when the window
        begins, e.g. the last row of the previous walk-forward fold's signals. The state
        machine starts in that position (managed from the first bar: exit, stop and stop-loss
        rules apply, and the sizes are refreshed to the window's own hedge ratio), and the
        first ``exec_lag`` bars execute those holdings instead of being flat, so consecutive
        folds can be stitched into one continuous position path.

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
    df["z"] = zscore_from_spread(df["resid"], method=z_method, window=z_window, halflife=z_halflife,
                                 history=z_history)

    n = len(df)

    def _mask(m, name):
        if m is None:
            return np.zeros(n, dtype=bool)
        if isinstance(m, pd.Series):
            m = m.reindex(df.index).fillna(False)
        m = np.asarray(m, dtype=bool)
        if m.shape != (n,):
            raise ValueError(f"{name} must have one entry per row of df_pair ({n}), got shape {m.shape}")
        return m
    flat_mask  = _mask(force_flat, "force_flat")
    block_mask = _mask(block_entry, "block_entry")

    pos_dec  = np.zeros(n, dtype=int)     # decision at time t (pre-execution)
    n1_dec   = np.zeros(n, dtype=float)
    n2_dec   = np.zeros(n, dtype=float)
    ent_dec  = np.zeros(n, dtype=bool)
    exit_dec = np.zeros(n, dtype=bool)
    stop_dec = np.zeros(n, dtype=bool)

    pos0, n1_0, n2_0 = (0, 0.0, 0.0) if initial_position is None else initial_position
    pos0 = int(np.sign(pos0))
    if pos0 == 0:
        n1_0, n2_0 = 0.0, 0.0
    pos = pos0
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
        can_enter = (cooldown == 0) and not block_mask[t] and not flat_mask[t]

        # Decide using info up to and including t
        if flat_mask[t] and pos != 0:
            pos = 0; exit_dec[t] = True; hold = 0; cooldown = cooldown_bars
        elif pos == 0 and can_enter:
            # enter only inside the band [z_entry, z_stop): a position opened at or beyond the stop
            # level would be stopped out on the next bar, and re-opened, for as long as the
            # excursion lasts — paying costs every bar
            if -z_stop < z <= -z_entry:
                pos = +1; hold = 1; ent_dec[t] = True
            elif z_entry <= z < z_stop:
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
    pos_exe  = pd.Series(pos_dec).shift(exec_lag, fill_value=pos0).astype(int).values
    n1_exe   = pd.Series(n1_dec).shift(exec_lag, fill_value=float(n1_0)).values
    n2_exe   = pd.Series(n2_dec).shift(exec_lag, fill_value=float(n2_0)).values
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


# ---- 4) Session rules for intraday bars ------------------------------------
def session_masks(
    index: pd.DatetimeIndex,
    *,
    exec_lag: int = 1,
    flatten: bool = True,
    no_entry_after: Optional[str] = None,
    session: Optional[np.ndarray] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Build the ``force_flat`` / ``block_entry`` masks for :func:`generate_pair_signals` from
    session rules on intraday bars.

    - ``flatten``: the last ``exec_lag`` bars of every session decide "flat", so that after the
      execution lag the position is closed on the session's final bar and nothing is carried
      overnight. (With ``exec_lag=1`` only the last bar is flagged; its flat decision executes
      on the next bar, i.e. the first bar of the next session — which is why the last bar's own
      P&L is still earned. The final bar's price is the last minute's close, not the closing
      auction.)
    - ``no_entry_after``: a time of day such as ``"15:30"``; bars at or after it never open a
      position (open positions are managed as usual).

    Sessions default to the calendar date of each bar. Returns ``(force_flat, block_entry)``
    boolean Series aligned to ``index``.
    """
    if exec_lag < 0:
        raise ValueError("exec_lag must be >= 0")
    idx = pd.DatetimeIndex(index)
    keys = np.asarray(idx.normalize() if session is None else session)
    if len(keys) != len(idx):
        raise ValueError("session must have one entry per bar")
    pos_from_end = pd.Series(np.arange(len(idx))).groupby(keys, sort=False).cumcount(ascending=False).to_numpy()
    flat = np.zeros(len(idx), dtype=bool)
    if flatten and exec_lag > 0:
        flat = pos_from_end < exec_lag
    block = flat.copy()
    if no_entry_after is not None:
        t = pd.Timestamp(f"2000-01-01 {no_entry_after}")
        cutoff = t.hour * 60 + t.minute
        block |= (idx.hour * 60 + idx.minute) >= cutoff
    return pd.Series(flat, index=idx, name="force_flat"), pd.Series(block, index=idx, name="block_entry")
