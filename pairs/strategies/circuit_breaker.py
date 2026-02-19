# pairs/strategies/circuit_breaker.py
"""
Circuit breaker post-processor for pairs trading signals.

Exports
-------
- CircuitBreakerConfig : dataclass for configuring trigger thresholds
- apply_circuit_breaker(signals, df_pair, *, ...) -> (signals_cb, audit_df)

Trigger logic
-------------
1. **Z-score blow-out + mandatory cooldown**
   Detects bars where |z| > z_halt, forces flat on that bar and the next
   ``cb_cooldown_bars`` bars.  Optionally, re-entry is only permitted once
   |z| has returned below ``z_reentry`` (even after the cooldown has elapsed).

2. **Strategy rolling drawdown halt**
   Runs a "shadow" evaluation pass on the *original* signals to obtain the
   equity curve.  Identifies bars where the rolling drawdown over the past
   ``drawdown_window_bars`` bars exceeds ``max_drawdown_pct`` of
   ``capital_base``.  Zeros out positions for the next ``halt_bars`` after
   the trigger bar.

Halt windows from both triggers are union-ed bar-by-bar.  The caller
receives a modified signals DataFrame (same columns, same index) plus an
audit DataFrame showing each contiguous halt window and its trigger type.

Notes
-----
- This function is a pure post-processor: it does NOT call data APIs.
- Setting ``z_halt=None`` disables the z-score trigger entirely.
- Setting ``max_drawdown_pct=None`` disables the drawdown trigger entirely.
- If both triggers are disabled the function is a no-op (returns
  ``signals.copy()`` plus an empty audit DataFrame).
- The caller is responsible for calling ``evaluate_pair_signals`` on the
  returned ``signals_cb``; this function deliberately does NOT return a
  daily_df to keep the interface composable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CircuitBreakerConfig",
    "apply_circuit_breaker",
]

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreakerConfig:
    """
    Configuration for :func:`apply_circuit_breaker`.

    Parameters
    ----------
    z_halt : float or None
        |z| threshold that fires the z-score trigger.  Must be > 0.
        ``None`` disables this trigger entirely.  Typical value: 3.5–5.0.
        Should be >= the ``z_stop`` used in ``generate_pair_signals`` so the
        circuit breaker acts as a second, harder gate.
    cb_cooldown_bars : int
        Bars to stay flat after a z-halt trigger fires (inclusive of the
        trigger bar itself).  Must be >= 0.  Default: 10.
    z_reentry : float or None
        After the mandatory cooldown, keep flat while |z| >= ``z_reentry``.
        ``None`` means no additional z condition is required.
        Default: ``None``.
    max_drawdown_pct : float or None
        Fractional rolling drawdown threshold in (0, 1] that fires the
        drawdown trigger.  E.g., 0.10 = 10% rolling drawdown relative to
        ``capital_base``.  ``None`` disables this trigger.
    drawdown_window_bars : int
        Look-back in bars for the rolling equity peak.
        ``dd_pct(t) = (equity(t) - equity.rolling(W).max()) / capital_base``.
        Default: 63 (approx. one quarter of daily bars).
    halt_bars : int
        Bars to stay flat after a drawdown trigger fires (inclusive of the
        trigger bar).  Must be >= 1.  Default: 20.
    capital_base : float or None
        Denominator for drawdown % normalisation.  ``None`` → auto-detected
        from the shadow evaluation (median in-position gross exposure, same
        logic as ``evaluate_pair_signals``).
    evaluate_kwargs : dict
        Extra keyword arguments forwarded to ``evaluate_pair_signals`` in the
        shadow pass (e.g., ``cost_bps``, ``borrow_bps_per_year``).
        Default: ``{}`` (zero costs; drawdown slightly overstated → safer).
    """

    z_halt: Optional[float] = None
    cb_cooldown_bars: int = 10
    z_reentry: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    drawdown_window_bars: int = 63
    halt_bars: int = 20
    capital_base: Optional[float] = None
    evaluate_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    signals: pd.DataFrame,
    df_pair,
    z_halt: Optional[float],
    max_drawdown_pct: Optional[float],
    halt_bars: int,
    cb_cooldown_bars: int,
) -> None:
    """Raise ``ValueError`` with descriptive messages for bad parameters."""
    required_sig_cols = {"pos", "n1", "n2"}
    missing = required_sig_cols - set(signals.columns)
    if missing:
        raise ValueError(
            f"signals is missing required columns: {sorted(missing)}"
        )

    if z_halt is not None and z_halt <= 0:
        raise ValueError(f"z_halt must be positive, got {z_halt!r}")

    if max_drawdown_pct is not None:
        if not (0 < max_drawdown_pct <= 1.0):
            raise ValueError(
                f"max_drawdown_pct must be in (0, 1], got {max_drawdown_pct!r}"
            )
        if df_pair is None:
            raise ValueError(
                "df_pair must be provided (not None) when max_drawdown_pct is set."
            )
        required_pair_cols = {"P1", "P2"}
        missing_pair = required_pair_cols - set(df_pair.columns)
        if missing_pair:
            raise ValueError(
                f"df_pair is missing required columns for drawdown trigger: "
                f"{sorted(missing_pair)}"
            )

    if halt_bars < 1:
        raise ValueError(f"halt_bars must be >= 1, got {halt_bars!r}")

    if cb_cooldown_bars < 0:
        raise ValueError(f"cb_cooldown_bars must be >= 0, got {cb_cooldown_bars!r}")


def _find_z_halt_windows(
    signals: pd.DataFrame,
    z_halt: float,
    cb_cooldown_bars: int,
    z_reentry: Optional[float],
) -> pd.Series:
    """
    Return a boolean Series (True = bar is inside a z-halt window).

    Algorithm (O(n)):
    1. Build a ``halt_end`` array initialised to -1.
    2. At each trigger bar *t* (|z[t]| > z_halt): set
       ``halt_end[t] = max(halt_end[t], t + cb_cooldown_bars)``.
    3. Forward-fill: ``halt_end[i] = max(halt_end[i], halt_end[i-1])``
       so overlapping windows merge naturally.
    4. Mark bar *i* as halted when ``i <= halt_end[i]``.
    5. Optional z_reentry extension: scan forward past each halt end;
       extend bar-by-bar while |z| >= z_reentry.

    Parameters
    ----------
    signals : DataFrame with a ``'z'`` column.
    z_halt : positive float threshold on |z|.
    cb_cooldown_bars : int >= 0; inclusive cooldown bars after trigger bar.
    z_reentry : float or None; post-cooldown re-entry condition.

    Returns
    -------
    pd.Series[bool], same index as *signals*.
    """
    n = len(signals)
    z_arr = signals["z"].to_numpy(dtype=float, copy=True)

    # --- Step 1 & 2: compute halt_end array ---------------------------------
    halt_end = np.full(n, -1, dtype=np.intp)
    for t in range(n):
        z_t = z_arr[t]
        if np.isnan(z_t):
            pass  # NaN never triggers
        elif abs(z_t) > z_halt:
            new_end = min(t + cb_cooldown_bars, n - 1)
            if new_end > halt_end[t]:
                halt_end[t] = new_end
        # Step 3: carry forward the maximum seen so far
        if t > 0 and halt_end[t - 1] > halt_end[t]:
            halt_end[t] = halt_end[t - 1]

    # --- Step 4: build boolean mask ------------------------------------------
    halted = np.zeros(n, dtype=bool)
    for i in range(n):
        if i <= halt_end[i]:
            halted[i] = True

    # --- Step 5: z_reentry extension -----------------------------------------
    if z_reentry is not None:
        i = 0
        while i < n:
            if not halted[i]:
                i += 1
                continue
            # skip to end of this contiguous window
            j = i
            while j < n and halted[j]:
                j += 1
            # j is now the first non-halted bar after this window
            # extend while |z[j]| >= z_reentry
            while j < n and not np.isnan(z_arr[j]) and abs(z_arr[j]) >= z_reentry:
                halted[j] = True
                j += 1
            i = j

    return pd.Series(halted, index=signals.index, name="cb_z_halted")


def _find_drawdown_halt_windows(
    signals: pd.DataFrame,
    df_pair: pd.DataFrame,
    max_drawdown_pct: float,
    drawdown_window_bars: int,
    halt_bars: int,
    capital_base: Optional[float],
    evaluate_kwargs: dict,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Shadow-evaluate signals, then detect rolling drawdown triggers.

    Rolling drawdown formula
    ------------------------
    ``equity(t)``         = cumulative net P&L (from shadow evaluation)
    ``rolling_peak(t)``   = ``equity.rolling(W, min_periods=1).max()``
    ``dd_pct(t)``         = ``(equity(t) - rolling_peak(t)) / capital_base``

    Trigger fires when ``dd_pct(t) < -max_drawdown_pct``.
    Halt window: bars ``[t, t + halt_bars - 1]`` (clamped to series length).

    Parameters
    ----------
    signals, df_pair : inputs to the shadow evaluation.
    max_drawdown_pct : float in (0, 1].
    drawdown_window_bars : int >= 1.
    halt_bars : int >= 1.
    capital_base : float or None (auto-detected when None).
    evaluate_kwargs : dict forwarded to evaluate_pair_signals.

    Returns
    -------
    halted_series : pd.Series[bool], same index as *signals*.
    daily_shadow  : pd.DataFrame from the shadow evaluation pass.
    """
    from pairs.strategies.evaluate import evaluate_pair_signals  # local import

    eval_kw = dict(evaluate_kwargs)
    if capital_base is not None:
        eval_kw["capital_base"] = capital_base

    daily_shadow, _, summary = evaluate_pair_signals(df_pair, signals, **eval_kw)

    # Recover capital_base: portfolio = capital_base + equity (constant offset)
    if capital_base is not None:
        cb = float(capital_base)
    else:
        cb = float(summary.get("capital_base", 1.0))
        if cb == 0:
            cb = 1.0

    equity = daily_shadow["equity"]

    # Rolling drawdown
    W = max(1, drawdown_window_bars)
    rolling_peak = equity.rolling(window=W, min_periods=1).max()
    dd_pct = (equity - rolling_peak) / cb  # always <= 0

    # Reindex to signals index (fills missing bars with NaN)
    dd_pct_aligned = dd_pct.reindex(signals.index)

    n = len(signals)
    halt_end = np.full(n, -1, dtype=np.intp)

    for t in range(n):
        dd_t = dd_pct_aligned.iloc[t]
        if np.isnan(dd_t):
            pass
        elif dd_t < -max_drawdown_pct:
            new_end = min(t + halt_bars - 1, n - 1)
            if new_end > halt_end[t]:
                halt_end[t] = new_end
        if t > 0 and halt_end[t - 1] > halt_end[t]:
            halt_end[t] = halt_end[t - 1]

    halted = np.zeros(n, dtype=bool)
    for i in range(n):
        if i <= halt_end[i]:
            halted[i] = True

    halted_series = pd.Series(halted, index=signals.index, name="cb_dd_halted")
    return halted_series, daily_shadow


def _build_audit_df(
    signals: pd.DataFrame,
    z_halted: Optional[pd.Series],
    dd_halted: Optional[pd.Series],
    combined_halted: pd.Series,
    daily_shadow: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Build an audit DataFrame with one row per contiguous halt window.

    Columns
    -------
    trigger_type      : str  — ``'z_halt'`` | ``'drawdown'`` | ``'both'``
    z_at_trigger      : float — |z| at the first bar of the window
                        (NaN for drawdown-only windows)
    dd_pct_at_trigger : float — rolling dd% at the first bar of the window
                        (NaN for z-only windows)
    halt_start        : Timestamp — first halted bar
    halt_end          : Timestamp — last halted bar
    n_halted_bars     : int — length of the contiguous window
    """
    COLS = [
        "trigger_type",
        "z_at_trigger",
        "dd_pct_at_trigger",
        "halt_start",
        "halt_end",
        "n_halted_bars",
    ]

    idx = signals.index
    n = len(signals)

    z_arr = (
        signals["z"].to_numpy(dtype=float)
        if "z" in signals.columns
        else np.full(n, np.nan)
    )
    zh_arr = (
        z_halted.to_numpy(dtype=bool)
        if z_halted is not None
        else np.zeros(n, dtype=bool)
    )
    dh_arr = (
        dd_halted.to_numpy(dtype=bool)
        if dd_halted is not None
        else np.zeros(n, dtype=bool)
    )

    # Align shadow drawdown_pct to signals index
    if daily_shadow is not None and "drawdown_pct" in daily_shadow.columns:
        dd_arr = daily_shadow["drawdown_pct"].reindex(idx).to_numpy(dtype=float)
    else:
        dd_arr = np.full(n, np.nan)

    cb_arr = combined_halted.to_numpy(dtype=bool)

    rows = []
    i = 0
    while i < n:
        if not cb_arr[i]:
            i += 1
            continue

        start_i = i
        while i < n and cb_arr[i]:
            i += 1
        end_i = i - 1  # inclusive

        is_z  = bool(zh_arr[start_i])
        is_dd = bool(dh_arr[start_i])

        if is_z and is_dd:
            ttype = "both"
        elif is_z:
            ttype = "z_halt"
        elif is_dd:
            ttype = "drawdown"
        else:
            # Window extension from prior trigger (e.g. z_reentry); label conservatively
            ttype = "z_halt"

        z_at   = float(abs(z_arr[start_i])) if not np.isnan(z_arr[start_i]) else np.nan
        dd_at  = float(dd_arr[start_i])     if not np.isnan(dd_arr[start_i]) else np.nan

        rows.append({
            "trigger_type":       ttype,
            "z_at_trigger":       z_at  if ttype in ("z_halt", "both") else np.nan,
            "dd_pct_at_trigger":  dd_at if ttype in ("drawdown", "both") else np.nan,
            "halt_start":         idx[start_i],
            "halt_end":           idx[end_i],
            "n_halted_bars":      end_i - start_i + 1,
        })

    return pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_circuit_breaker(
    signals: pd.DataFrame,
    df_pair: pd.DataFrame,
    *,
    # Z-score trigger
    z_halt: Optional[float] = None,
    cb_cooldown_bars: int = 10,
    z_reentry: Optional[float] = None,
    # Drawdown trigger
    max_drawdown_pct: Optional[float] = None,
    drawdown_window_bars: int = 63,
    halt_bars: int = 20,
    capital_base: Optional[float] = None,
    # Shadow evaluation
    evaluate_kwargs: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply circuit breaker logic to a signals DataFrame (post-processor).

    This function is a standalone post-processor and does NOT call
    ``generate_pair_signals`` or any data-fetching API.

    Parameters
    ----------
    signals : pd.DataFrame
        Output of :func:`generate_pair_signals`.  Must contain columns
        ``['pos', 'n1', 'n2']``.  Should also contain ``'z'`` when
        ``z_halt`` is specified.  Index must be a DatetimeIndex aligned
        with *df_pair*.
    df_pair : pd.DataFrame or None
        The same ``df_pair`` used in signal generation.  Required (not None)
        when ``max_drawdown_pct`` is set; used for the shadow evaluation pass.
        Must contain ``['P1', 'P2']``.  Ignored when only the z trigger is used.
    z_halt : float or None
        |z| threshold for the z-score trigger.  ``None`` disables it.
    cb_cooldown_bars : int
        Bars to remain flat after a z-halt trigger, inclusive of trigger bar.
        Default: 10.
    z_reentry : float or None
        Post-cooldown condition: keep flat while |z| >= ``z_reentry``.
        ``None`` means no z condition after the cooldown expires.
    max_drawdown_pct : float or None
        Fractional drawdown threshold in (0, 1].  E.g., 0.10 = 10%.
        ``None`` disables the drawdown trigger.
    drawdown_window_bars : int
        Rolling window in bars for the equity peak calculation.  Default: 63.
    halt_bars : int
        Bars to stay flat after a drawdown trigger fires.  Default: 20.
    capital_base : float or None
        Capital base for normalising drawdown %.  ``None`` → auto-detected.
    evaluate_kwargs : dict or None
        Keyword arguments forwarded to ``evaluate_pair_signals`` in the
        shadow pass (e.g., ``cost_bps``, ``borrow_bps_per_year``).
        Default: ``None`` (equivalent to ``{}``).

    Returns
    -------
    signals_cb : pd.DataFrame
        Modified copy of *signals* with the same columns and index.
        Halted bars have ``pos=0``, ``n1=0.0``, ``n2=0.0``.
        The ``entry`` / ``exit`` / ``stop`` flags are preserved to maintain
        an audit trail of the original decisions.
    audit_df : pd.DataFrame
        One row per contiguous halt window, with columns:

        ==================  =================================================
        trigger_type        ``'z_halt'`` | ``'drawdown'`` | ``'both'``
        z_at_trigger        |z| on the trigger bar (NaN for drawdown-only)
        dd_pct_at_trigger   rolling dd% on trigger bar (NaN for z-only)
        halt_start          Timestamp — first halted bar
        halt_end            Timestamp — last halted bar
        n_halted_bars       int — number of consecutive halted bars
        ==================  =================================================

    Raises
    ------
    ValueError
        If ``z_halt`` is provided but not positive.
        If ``max_drawdown_pct`` is provided but not in (0, 1].
        If ``halt_bars < 1`` or ``cb_cooldown_bars < 0``.
        If required columns are missing from *signals* or *df_pair*.
        If ``max_drawdown_pct`` is set but ``df_pair`` is ``None``.

    Notes
    -----
    **No-op fast paths**

    * Both triggers disabled → returns ``signals.copy()`` + empty audit.
    * All signals already flat (``signals['pos'].abs().sum() == 0``) → same.

    **Last-bar trigger**
        Halt window is clamped to the array bounds; no exception raised.

    **Overlapping windows**
        Windows from multiple trigger events are merged into a single
        contiguous entry in *audit_df*.

    **Double evaluation**
        The caller is responsible for calling ``evaluate_pair_signals`` on
        ``signals_cb``; this function does not do so.

    Examples
    --------
    >>> signals_cb, audit = apply_circuit_breaker(
    ...     signals, df_pair,
    ...     z_halt=4.0,           # halt on z blow-out (matches z_stop)
    ...     cb_cooldown_bars=10,  # 2-week cooldown
    ...     z_reentry=2.0,        # only re-enter once z comes back
    ...     max_drawdown_pct=0.05, # halt on 5 % rolling drawdown
    ...     drawdown_window_bars=63,
    ...     halt_bars=20,
    ...     capital_base=10_000,
    ...     evaluate_kwargs=dict(cost_bps=1, borrow_bps_per_year=50),
    ... )
    >>> daily, trades, summary = evaluate_pair_signals(df_pair, signals_cb, ...)
    """
    if evaluate_kwargs is None:
        evaluate_kwargs = {}

    # --- Input validation ----------------------------------------------------
    _validate_inputs(signals, df_pair, z_halt, max_drawdown_pct, halt_bars, cb_cooldown_bars)

    # --- No-op fast paths ----------------------------------------------------
    _EMPTY_AUDIT_COLS = [
        "trigger_type", "z_at_trigger", "dd_pct_at_trigger",
        "halt_start", "halt_end", "n_halted_bars",
    ]
    both_disabled = (z_halt is None) and (max_drawdown_pct is None)
    all_flat = (signals["pos"].abs().sum() == 0)
    if both_disabled or all_flat:
        return signals.copy(), pd.DataFrame(columns=_EMPTY_AUDIT_COLS)

    # --- Trigger 1: Z-score blow-out -----------------------------------------
    z_halted: Optional[pd.Series] = None
    if z_halt is not None:
        if "z" not in signals.columns:
            raise ValueError(
                "signals must contain a 'z' column when z_halt is specified."
            )
        z_halted = _find_z_halt_windows(signals, z_halt, cb_cooldown_bars, z_reentry)

    # --- Trigger 2: Rolling drawdown -----------------------------------------
    dd_halted: Optional[pd.Series] = None
    daily_shadow: Optional[pd.DataFrame] = None
    if max_drawdown_pct is not None:
        dd_halted, daily_shadow = _find_drawdown_halt_windows(
            signals,
            df_pair,
            max_drawdown_pct,
            drawdown_window_bars,
            halt_bars,
            capital_base,
            evaluate_kwargs,
        )

    # --- Union of halt windows -----------------------------------------------
    n = len(signals)
    combined = np.zeros(n, dtype=bool)
    if z_halted is not None:
        combined |= z_halted.to_numpy(dtype=bool)
    if dd_halted is not None:
        combined |= dd_halted.to_numpy(dtype=bool)
    combined_series = pd.Series(combined, index=signals.index, name="cb_halted")

    # --- Patch signals -------------------------------------------------------
    signals_cb = signals.copy()
    mask = combined_series.to_numpy(dtype=bool)
    signals_cb.loc[combined_series[mask].index, "pos"] = 0
    signals_cb.loc[combined_series[mask].index, "n1"]  = 0.0
    signals_cb.loc[combined_series[mask].index, "n2"]  = 0.0
    # entry/exit/stop intentionally preserved (original decision audit trail)

    # --- Build audit ---------------------------------------------------------
    audit_df = _build_audit_df(
        signals, z_halted, dd_halted, combined_series, daily_shadow
    )

    return signals_cb, audit_df
