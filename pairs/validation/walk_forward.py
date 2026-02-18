# pairs/validation/walk_forward.py
"""
Walk-forward (rolling window) validation for pairs trading strategies.

Design philosophy
-----------------
* **No look-ahead**: every test fold uses parameters (Kalman Q/R, signal
  thresholds) estimated *only* on data preceding that fold.
* **Rolling window**: the train window has a fixed length (``train_bars``),
  slides forward by ``step_bars`` each fold, so the filter never sees future
  prices.
* **Lightweight**: the module is intentionally free of hard Kalman/signal
  dependencies — it accepts callables so users can swap implementations.

Typical usage
-------------
>>> from pairs.validation import walk_forward_backtest
>>> fold_results = walk_forward_backtest(
...     df_pair,                # DataFrame with P1, P2, (optionally beta/resid)
...     train_bars=504,         # ~2 years of daily bars
...     test_bars=126,          # ~6 months
...     step_bars=63,           # advance by ~1 quarter each fold
...     fit_fn=my_kalman_fit,   # callable(df_train) -> fitted artefact
...     signal_fn=my_signals,   # callable(df_test, artefact) -> signals
...     eval_fn=my_evaluate,    # callable(df_test, signals) -> summary dict
... )
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd


__all__ = [
    "walk_forward_splits",
    "walk_forward_backtest",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: Optional[int] = None,
    min_test_bars: int = 1,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate (train_index, test_index) slices for walk-forward validation.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Sorted datetime index of the full sample (no duplicates).
    train_bars : int
        Number of bars in each training window.
    test_bars : int
        Number of bars in each test (out-of-sample) window.
    step_bars : int or None
        How many bars to advance the window each fold.
        Defaults to ``test_bars`` (non-overlapping test folds).
    min_test_bars : int
        Skip the final fold if it would contain fewer than this many test bars.

    Returns
    -------
    List of (train_index, test_index) tuples.

    Raises
    ------
    ValueError
        If ``train_bars`` or ``test_bars`` are non-positive, or if
        ``train_bars + test_bars > len(index)``.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2020-01-01", periods=300, freq="B")
    >>> splits = walk_forward_splits(idx, train_bars=200, test_bars=50)
    >>> len(splits)
    2
    """
    if train_bars <= 0:
        raise ValueError(f"train_bars must be positive, got {train_bars}")
    if test_bars <= 0:
        raise ValueError(f"test_bars must be positive, got {test_bars}")
    if step_bars is None:
        step_bars = test_bars
    if step_bars <= 0:
        raise ValueError(f"step_bars must be positive, got {step_bars}")

    n = len(index)
    if train_bars + test_bars > n:
        # Not enough data for even one fold — return empty gracefully.
        return []

    splits = []
    start = 0
    while True:
        train_end = start + train_bars
        test_end  = train_end + test_bars
        if train_end > n:
            break
        actual_test_end = min(test_end, n)
        actual_test_len = actual_test_end - train_end
        if actual_test_len < min_test_bars:
            break
        splits.append((index[start:train_end], index[train_end:actual_test_end]))
        start += step_bars

    return splits


# ── main walk-forward engine ──────────────────────────────────────────────────

def walk_forward_backtest(
    df: pd.DataFrame,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: Optional[int] = None,
    fit_fn: Callable[[pd.DataFrame], Any],
    signal_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    eval_fn: Callable[[pd.DataFrame, pd.DataFrame], Dict[str, Any]],
    min_test_bars: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a walk-forward back-test on a pair DataFrame.

    For each fold:
    1. Slice the *training* window from ``df``.
    2. Call ``fit_fn(df_train)`` to fit a model / calibrate parameters.
    3. Slice the *test* window from ``df``.
    4. Call ``signal_fn(df_test, artefact)`` to generate trading signals.
    5. Call ``eval_fn(df_test, signals)`` to compute performance metrics.

    All fold results are collected into a summary DataFrame indexed by fold.

    Parameters
    ----------
    df : pd.DataFrame
        Pair-level DataFrame with a ``DatetimeIndex`` and at minimum P1, P2
        columns.  Must not contain duplicate dates.
    train_bars : int
        Length of each training window (bars).
    test_bars : int
        Length of each test window (bars).
    step_bars : int or None
        How many bars to slide the window forward each fold.
        Defaults to ``test_bars`` (non-overlapping test windows).
    fit_fn : callable
        Signature: ``fit_fn(df_train: pd.DataFrame) -> artefact``
        Fits/calibrates the model on the training slice.  The returned
        artefact is passed to ``signal_fn``.
    signal_fn : callable
        Signature: ``signal_fn(df_test: pd.DataFrame, artefact) -> signals``
        Generates trading signals for the test slice.
    eval_fn : callable
        Signature: ``eval_fn(df_test: pd.DataFrame, signals: pd.DataFrame) -> dict``
        Evaluates performance; returns a flat dict of metrics.
    min_test_bars : int
        Folds whose test window is shorter than this are skipped.
    verbose : bool
        If True, print a one-line summary per fold.

    Returns
    -------
    pd.DataFrame
        One row per fold.  Columns include ``fold``, ``train_start``,
        ``train_end``, ``test_start``, ``test_end``, ``n_train``,
        ``n_test``, plus every key returned by ``eval_fn``.

    Notes
    -----
    * ``fit_fn`` is allowed to return ``None`` when the training slice is
      degenerate (too short, all NaN, etc.).  In that case, the fold is
      skipped with a warning.
    * Exceptions raised inside ``fit_fn`` or ``signal_fn`` are caught and
      logged as a warning so that a single bad fold does not abort the run.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "df must have a DatetimeIndex. "
            "Use df.set_index('datetime') or df.index = pd.to_datetime(df.index)."
        )
    if df.index.duplicated().any():
        raise ValueError("df.index contains duplicate dates. De-duplicate before calling.")

    splits = walk_forward_splits(
        df.index,
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        min_test_bars=min_test_bars,
    )

    if len(splits) == 0:
        warnings.warn(
            "walk_forward_backtest: no valid folds found with "
            f"train_bars={train_bars}, test_bars={test_bars}, "
            f"total_bars={len(df)}.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        df_train = df.loc[train_idx]
        df_test  = df.loc[test_idx]

        meta = {
            "fold":        fold_num,
            "train_start": train_idx[0],
            "train_end":   train_idx[-1],
            "test_start":  test_idx[0],
            "test_end":    test_idx[-1],
            "n_train":     len(df_train),
            "n_test":      len(df_test),
        }

        # ── Fit ────────────────────────────────────────────────────────────
        try:
            artefact = fit_fn(df_train)
        except Exception as e:
            warnings.warn(
                f"Fold {fold_num}: fit_fn raised {type(e).__name__}: {e}. Skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        if artefact is None:
            warnings.warn(
                f"Fold {fold_num}: fit_fn returned None (degenerate training data). Skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        # ── Signal ─────────────────────────────────────────────────────────
        try:
            signals = signal_fn(df_test, artefact)
        except Exception as e:
            warnings.warn(
                f"Fold {fold_num}: signal_fn raised {type(e).__name__}: {e}. Skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        # ── Evaluate ───────────────────────────────────────────────────────
        try:
            metrics = eval_fn(df_test, signals)
        except Exception as e:
            warnings.warn(
                f"Fold {fold_num}: eval_fn raised {type(e).__name__}: {e}. Skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        row = {**meta, **metrics}
        rows.append(row)

        if verbose:
            sharpe = metrics.get("sharpe", float("nan"))
            ann_ret = metrics.get("ann_return", float("nan"))
            n_tr = metrics.get("n_trades", "?")
            print(
                f"  Fold {fold_num:>2d} | "
                f"test {meta['test_start'].date()} → {meta['test_end'].date()} | "
                f"Sharpe={sharpe:+.2f}  AnnRet={ann_ret:+.1%}  Trades={n_tr}"
            )

    if not rows:
        warnings.warn(
            "walk_forward_backtest: all folds were skipped (fit/signal/eval errors).",
            UserWarning, stacklevel=2,
        )
        return pd.DataFrame()

    results = pd.DataFrame(rows).set_index("fold")
    return results


# ── convenience summary ───────────────────────────────────────────────────────

def summarize_walk_forward(
    results: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute mean ± std and median across walk-forward folds for key metrics.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`walk_forward_backtest`.
    metric_cols : list of str or None
        Columns to summarize.  Defaults to a standard set:
        ['sharpe', 'ann_return', 'max_drawdown_pct', 'n_trades', 'hit_rate'].

    Returns
    -------
    pd.DataFrame
        Index: metric names.  Columns: mean, std, median, min, max.
    """
    if metric_cols is None:
        metric_cols = [
            "sharpe", "ann_return", "max_drawdown_pct",
            "n_trades", "hit_rate", "profit_factor",
        ]
    available = [c for c in metric_cols if c in results.columns]
    if not available:
        raise ValueError(
            f"None of the requested metric_cols {metric_cols} found in results. "
            f"Available columns: {list(results.columns)}"
        )

    subset = results[available].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame({
        "mean":   subset.mean(),
        "std":    subset.std(),
        "median": subset.median(),
        "min":    subset.min(),
        "max":    subset.max(),
    })
    return summary
