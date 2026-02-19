# pairs/stats/cointegration.py
"""
Fast parallel cointegration screening (EG or Johansen) and a dual-gate
combinator that requires both tests to pass. Expects a MultiIndex (ticker, datetime)
DataFrame with a 'close' column.

Multiple-testing note
---------------------
Screening N*(N-1)/2 pairs simultaneously inflates false discoveries.
Use ``fdr_method="bh"`` in :func:`find_cointegrated_pairs_dualgate` to apply the
Benjamini-Hochberg (BH) correction, which controls the *expected* fraction of
false positives among all declared cointegrated pairs at level ``fdr_alpha``.
"""
from __future__ import annotations
from typing import Literal, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from functools import partial
import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from pairs.utils.splits import normalize_multiindex

__all__ = [
    "benjamini_hochberg_fdr",
    "find_cointegrated_pairs_executor",
    "find_cointegrated_pairs_dualgate",
]

# ── Multiple-testing correction ───────────────────────────────────────────────

def benjamini_hochberg_fdr(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg (1995) step-up procedure for False Discovery Rate control.

    Controls the *expected proportion of false positives* among all rejected
    hypotheses (declared cointegrated pairs) rather than the family-wise error
    rate used by Bonferroni.  BH is far less conservative when there are many
    true positives (which is typical in equity pair screening).

    Parameters
    ----------
    pvalues : array-like of float, shape (m,)
        Raw p-values from m simultaneous hypothesis tests.  NaN entries are
        treated as non-significant (reject=False) and are excluded from the
        adjustment.
    alpha : float, default 0.05
        Target FDR level.  At most ``alpha * 100 %`` of all discoveries are
        expected to be false positives.

    Returns
    -------
    reject : ndarray of bool, shape (m,)
        True where the adjusted p-value satisfies p_adj ≤ alpha.
    pvalues_adj : ndarray of float, shape (m,)
        BH-adjusted p-values (Simes upper bound), clipped to [0, 1].

    Notes
    -----
    The procedure ranks p-values p_(1) ≤ … ≤ p_(m) and rejects all
    hypotheses up to the largest rank k for which p_(k) ≤ (k/m)*alpha.
    Adjusted p-values are computed as p_adj_(k) = min over j≥k of (m/j)*p_(j),
    which is the Simes correction.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    pvalues_adj = np.ones(m)  # default: 1.0 (not rejected)
    reject = np.zeros(m, dtype=bool)

    if m == 0:
        return reject, pvalues_adj

    # Only process non-NaN entries
    valid_mask = ~np.isnan(pvalues)
    valid_idx = np.where(valid_mask)[0]
    pv = pvalues[valid_idx]
    n = len(pv)

    if n == 0:
        return reject, pvalues_adj

    # Sort p-values ascending; track original positions
    order = np.argsort(pv, kind="stable")
    ranks = np.arange(1, n + 1)  # 1-based ranks

    # BH adjusted p-values: p_adj_i = min over j>=i of (n / rank_j) * p_j
    # Compute from largest rank downward (cumulative minimum from right)
    scaled = (n / ranks) * pv[order]        # (m / k) * p_(k)
    adj = np.minimum.accumulate(scaled[::-1])[::-1]  # right-to-left cummin
    adj = np.clip(adj, 0.0, 1.0)

    # Map back to original valid positions
    adj_unsorted = np.empty(n)
    adj_unsorted[order] = adj
    pvalues_adj[valid_idx] = adj_unsorted

    # Reject where adjusted p ≤ alpha
    reject[valid_idx] = adj_unsorted <= alpha

    return reject, pvalues_adj


# ---- worker globals (populated by _init_worker) ----
_SERIES_MAP = None
_KEYS = None
_FULL_START = None
_FULL_END = None

def _init_worker(series_map, keys, full_start, full_end):
    """Runs once per worker process. Store heavy objects in globals and cap BLAS threads."""
    global _SERIES_MAP, _KEYS, _FULL_START, _FULL_END
    _SERIES_MAP = series_map
    _KEYS = keys
    _FULL_START = full_start
    _FULL_END = full_end

    # Cap BLAS threads inside each process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Best-effort: expand CPU affinity to all CPUs (harmless if already full)
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            p.cpu_affinity(list(range(os.cpu_count() or 1)))
    except Exception:
        pass

def _align_dropna(S1: pd.Series, S2: pd.Series) -> pd.DataFrame:
    return pd.concat([S1, S2], axis=1, keys=["S1", "S2"]).dropna()

def _alpha_to_col(alpha: float) -> int:
    if alpha <= 0.01: return 2
    if alpha <= 0.05: return 1
    if alpha <= 0.10: return 0
    return 1

def _task_eg(idx_pair, *, alpha: float, trend: str, maxlag: Optional[int], autolag: str):
    i, j = idx_pair
    k1, k2 = _KEYS[i], _KEYS[j]
    S1, S2 = _SERIES_MAP[k1], _SERIES_MAP[k2]

    # Require full-span coverage to avoid biased tests
    if not (S1.index[0] == _FULL_START and S1.index[-1] == _FULL_END and
            S2.index[0] == _FULL_START and S2.index[-1] == _FULL_END):
        return i, j, 0.0, np.nan, False

    df = _align_dropna(S1, S2)
    if len(df) < 50:
        return i, j, 0.0, np.nan, False

    with threadpool_limits(limits=1):
        tstat, pval, _ = coint(df["S1"], df["S2"], trend=trend, maxlag=maxlag, autolag=autolag)
    return i, j, float(tstat), float(pval), bool(pval < alpha)

def _task_johansen(idx_pair, *, alpha: float, det_order: int, k_ar_diff: int, stat: Literal["trace","maxeig"]):
    i, j = idx_pair
    k1, k2 = _KEYS[i], _KEYS[j]
    S1, S2 = _SERIES_MAP[k1], _SERIES_MAP[k2]

    if not (S1.index[0] == _FULL_START and S1.index[-1] == _FULL_END and
            S2.index[0] == _FULL_START and S2.index[-1] == _FULL_END):
        return i, j, 0.0, np.nan, False

    df = _align_dropna(S1, S2)
    if len(df) < 50:
        return i, j, 0.0, np.nan, False

    X = df[["S1", "S2"]].values
    with threadpool_limits(limits=1):
        res = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)

    alpha_col = _alpha_to_col(alpha)
    if stat == "trace":
        test_stat, crit_val = float(res.lr1[0]), float(res.cvt[0, alpha_col])
    else:
        test_stat, crit_val = float(res.lr2[0]), float(res.cvm[0, alpha_col])
    return i, j, test_stat, np.nan, bool(test_stat > crit_val)


def _task_dualgate(
    idx_pair,
    *,
    alpha_eg: float,
    eg_trend: str,
    eg_maxlag: Optional[int],
    eg_autolag: str,
    alpha_joh: float,
    joh_det_order: int,
    joh_k_ar_diff: int,
    joh_stat: str,
):
    """
    Fused EG + Johansen worker: run both tests in a single subprocess call.

    Returns
    -------
    (i, j, eg_t, eg_p, eg_pass, joh_stat_val, joh_pass)
    """
    i, j = idx_pair
    k1, k2 = _KEYS[i], _KEYS[j]
    S1, S2 = _SERIES_MAP[k1], _SERIES_MAP[k2]

    # Require full-span coverage
    if not (S1.index[0] == _FULL_START and S1.index[-1] == _FULL_END and
            S2.index[0] == _FULL_START and S2.index[-1] == _FULL_END):
        return i, j, 0.0, np.nan, False, 0.0, False

    df = _align_dropna(S1, S2)
    if len(df) < 50:
        return i, j, 0.0, np.nan, False, 0.0, False

    X = df[["S1", "S2"]].values

    with threadpool_limits(limits=1):
        # EG
        eg_t, eg_p, _ = coint(df["S1"], df["S2"],
                               trend=eg_trend, maxlag=eg_maxlag, autolag=eg_autolag)
        # Johansen
        res = coint_johansen(X, det_order=joh_det_order, k_ar_diff=joh_k_ar_diff)

    eg_pass = bool(float(eg_p) < alpha_eg)

    alpha_col = _alpha_to_col(alpha_joh)
    if joh_stat == "trace":
        joh_val, joh_crit = float(res.lr1[0]), float(res.cvt[0, alpha_col])
    else:
        joh_val, joh_crit = float(res.lr2[0]), float(res.cvm[0, alpha_col])
    joh_pass = bool(joh_val > joh_crit)

    return i, j, float(eg_t), float(eg_p), eg_pass, joh_val, joh_pass

def find_cointegrated_pairs_executor(
    data: pd.DataFrame,
    *,
    alpha: float = 0.05,
    method: Literal["eg", "johansen"] = "eg",
    # EG
    eg_trend: str = "c",
    eg_maxlag: Optional[int] = None,
    eg_autolag: str = "aic",
    # Johansen
    joh_det_order: int = 0,
    joh_k_ar_diff: int = 1,
    joh_stat: Literal["trace", "maxeig"] = "trace",
    # parallel
    n_workers: Optional[int] = None,      # use physical cores on TR: 64
    chunksize: int = 4,                   # small chunks to keep all workers busy
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Expects `data` with a MultiIndex whose first level is ticker and second is datetime,
    and with a 'close' column.
    Returns:
        score_matrix (n x n)
        pvalue_matrix (n x n)  [np.nan for Johansen]
        pairs (List[Tuple[str, str]])  # native Python strings
    """
    # ---- Validate and normalize index ----
    if "close" not in data.columns:
        raise ValueError("data must have column 'close' and a MultiIndex (ticker, datetime).")
    data = normalize_multiindex(data)

    # ---- Build per-ticker close series with DatetimeIndex ----
    series_map = {}
    for k, g in data.groupby(level="ticker"):
        s = g["close"].copy()
        s.index = s.index.droplevel("ticker")          # leave only datetime in index
        if not isinstance(s.index, pd.DatetimeIndex):  # ensure dtype
            s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.sort_index()
        series_map[str(k)] = s                         # ensure keys are native str

    # Native Python str keys (avoid np.str_)
    keys: List[str] = [str(k) for k in series_map.keys()]
    n = len(keys)

    score_matrix  = np.zeros((n, n))
    pvalue_matrix = np.full((n, n), np.nan)
    pairs: List[Tuple[str, str]] = []

    # Global span from the datetime level
    dt_index = data.index.get_level_values("datetime")
    full_start, full_end = dt_index.min(), dt_index.max()

    combos = list(combinations(range(n), 2))

    # Warn about multiple testing inflation
    n_tests = len(combos)
    expected_false_positives = n_tests * alpha
    if n_tests > 100:
        warnings.warn(
            f"Multiple testing: running {n_tests} pair tests at alpha={alpha}. "
            f"Expected false positives by chance: ~{expected_false_positives:.0f}. "
            "Consider applying a correction (e.g. Bonferroni: alpha/{n_tests}, "
            "or FDR/Benjamini-Hochberg) to control the false discovery rate.",
            UserWarning,
            stacklevel=2,
        )

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // 2)  # ~physical cores

    # Choose picklable task (partials are picklable; functions are top-level)
    if method == "eg":
        task = partial(_task_eg, alpha=alpha, trend=eg_trend, maxlag=eg_maxlag, autolag=eg_autolag)
        desc = "Engle–Granger cointegration tests"
    elif method == "johansen":
        task = partial(_task_johansen, alpha=alpha, det_order=joh_det_order,
                       k_ar_diff=joh_k_ar_diff, stat=joh_stat)
        desc = "Johansen cointegration tests"
    else:
        raise ValueError("method must be 'eg' or 'johansen'")

    # Run
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(series_map, keys, full_start, full_end),
    ) as ex:
        it = ex.map(task, combos, chunksize=chunksize)
        if show_progress:
            it = tqdm(it, total=len(combos), desc=desc, leave=False)

        for i, j, score, pval, is_coint in it:
            k1, k2 = keys[i], keys[j]           # plain str
            score_matrix[i, j]  = score
            pvalue_matrix[i, j] = pval
            if is_coint:
                pairs.append((k1, k2))          # tuples of str

    return score_matrix, pvalue_matrix, pairs

def find_cointegrated_pairs_dualgate(
    data: pd.DataFrame,
    *,
    # EG params
    alpha_eg: float = 0.05,
    eg_trend: str = "c",
    eg_maxlag: Optional[int] = None,
    eg_autolag: str = "aic",
    # Johansen params
    alpha_joh: float = 0.05,
    joh_det_order: int = 0,
    joh_k_ar_diff: int = 1,
    joh_stat: Literal["trace", "maxeig"] = "trace",
    # FDR correction
    fdr_method: Literal["bh", "none"] = "bh",
    fdr_alpha: Optional[float] = None,   # defaults to alpha_eg if None
    # parallel
    n_workers: Optional[int] = None,
    chunksize: int = 32,
    show_progress: bool = True,
    # output options
    only_pass: bool = False,         # return only pairs that pass both tests
    sort_by: Optional[str] = "eg_p"  # e.g. "eg_p", "eg_t", "joh_stat" or None
) -> pd.DataFrame:
    """
    Run EG and Johansen in a **single fused parallel pass**, keep pairs that
    pass BOTH (dual-gate), and return a summary DataFrame.

    Compared with running two separate executor calls, this uses one process
    pool spawn instead of two — typically saving 15–30 s on macOS/Linux where
    ``multiprocessing`` defaults to the ``spawn`` start method.

    Multiple-testing correction
    ---------------------------
    With ``fdr_method="bh"`` (default), Benjamini-Hochberg FDR correction is
    applied to the raw EG p-values before determining ``eg_pass``.  The
    corrected p-value is stored in the ``eg_p_fdr`` column.  Setting
    ``fdr_method="none"`` reverts to the uncorrected raw p-value gate used in
    earlier versions (reproduces the original behaviour).

    Parameters
    ----------
    data : pd.DataFrame
        MultiIndex (ticker, datetime) with a 'close' column.
    alpha_eg : float
        Raw EG significance threshold (also the FDR target level unless
        ``fdr_alpha`` overrides it).
    fdr_method : {"bh", "none"}
        Multiple-testing correction applied to EG p-values.
        "bh" = Benjamini-Hochberg (default); "none" = no correction.
    fdr_alpha : float or None
        FDR target level.  Defaults to ``alpha_eg``.
    n_workers : int or None
        Number of parallel worker processes.  Defaults to half the logical
        CPU count (≈ physical cores).
    chunksize : int
        Number of pair tasks per IPC batch.  Default: 32.  Larger values
        reduce IPC overhead; smaller values give more even load balancing.
        For ~5 000 pairs and 8 workers, 32–64 is a good range.
    only_pass : bool
        If True, return only pairs where verdict == "pass".
    sort_by : str or None
        Column to sort by.  "eg_p" sorts by corrected p-value when BH is
        active, otherwise by raw p-value.

    Returns
    -------
    pd.DataFrame
        Index: MultiIndex (ticker1, ticker2)
        Columns: eg_t, eg_p, eg_p_fdr, eg_pass, joh_stat, joh_pass, verdict
    """
    if fdr_alpha is None:
        fdr_alpha = alpha_eg

    if fdr_method not in ("bh", "none"):
        raise ValueError(f"fdr_method must be 'bh' or 'none', got {fdr_method!r}")

    # ---- Validate and normalise index ----------------------------------------
    if "close" not in data.columns:
        raise ValueError("data must have column 'close' and a MultiIndex (ticker, datetime).")
    data = normalize_multiindex(data)

    # ---- Build per-ticker series map -----------------------------------------
    series_map: dict = {}
    for k, g in data.groupby(level="ticker"):
        s = g["close"].copy()
        s.index = s.index.droplevel("ticker")
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors="coerce")
        series_map[str(k)] = s.sort_index()

    tickers: List[str] = [str(k) for k in series_map]
    n = len(tickers)

    dt_index = data.index.get_level_values("datetime")
    full_start, full_end = dt_index.min(), dt_index.max()

    combos = list(combinations(range(n), 2))
    n_tests = len(combos)

    # Warn about multiple testing
    if n_tests > 100:
        warnings.warn(
            f"Multiple testing: running {n_tests} pair tests at alpha_eg={alpha_eg}. "
            f"Expected false positives by chance: ~{n_tests * alpha_eg:.0f}. "
            "BH FDR correction is applied by default (fdr_method='bh').",
            UserWarning,
            stacklevel=2,
        )

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // 2)

    # ---- Single fused pool: one spawn, both tests per worker call ------------
    task = partial(
        _task_dualgate,
        alpha_eg=alpha_eg,
        eg_trend=eg_trend,
        eg_maxlag=eg_maxlag,
        eg_autolag=eg_autolag,
        alpha_joh=alpha_joh,
        joh_det_order=joh_det_order,
        joh_k_ar_diff=joh_k_ar_diff,
        joh_stat=joh_stat,
    )

    # Preallocate result arrays
    eg_t_arr   = np.zeros(n_tests)
    eg_p_arr   = np.full(n_tests, np.nan)
    eg_pass_arr  = np.zeros(n_tests, dtype=bool)
    joh_val_arr  = np.zeros(n_tests)
    joh_pass_arr = np.zeros(n_tests, dtype=bool)
    flat_idx_map: dict = {(i, j): k for k, (i, j) in enumerate(combos)}

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(series_map, tickers, full_start, full_end),
    ) as ex:
        it = ex.map(task, combos, chunksize=chunksize)
        if show_progress:
            it = tqdm(it, total=n_tests, desc="Dual-gate cointegration (EG + Johansen)", leave=False)

        for i, j, eg_t, eg_p, eg_pass, joh_val, joh_pass in it:
            k = flat_idx_map[(i, j)]
            eg_t_arr[k]    = eg_t
            eg_p_arr[k]    = eg_p
            eg_pass_arr[k] = eg_pass
            joh_val_arr[k] = joh_val
            joh_pass_arr[k] = joh_pass

    # ---- Apply BH FDR correction to raw EG p-values -------------------------
    if fdr_method == "bh":
        reject_bh, pvals_adj = benjamini_hochberg_fdr(eg_p_arr, alpha=fdr_alpha)
        n_raw_sig = int(np.sum(eg_p_arr <= alpha_eg))
        n_fdr_sig = int(np.sum(reject_bh))
        n_removed = n_raw_sig - n_fdr_sig
        if n_removed > 0:
            warnings.warn(
                f"BH FDR correction (alpha={fdr_alpha}) removed {n_removed} of "
                f"{n_raw_sig} raw EG-significant pairs "
                f"({n_fdr_sig} remain after correction across {n_tests:,} tests).",
                UserWarning,
                stacklevel=2,
            )
        eg_pass_final = reject_bh
    else:
        pvals_adj     = eg_p_arr.copy()
        eg_pass_final = eg_pass_arr

    # ---- Build output DataFrame ----------------------------------------------
    rows = []
    for k, (i, j) in enumerate(combos):
        k1, k2 = tickers[i], tickers[j]
        eg_pass  = bool(eg_pass_final[k])
        joh_pass = bool(joh_pass_arr[k])
        verdict  = "pass" if (eg_pass and joh_pass) else "fail"
        rows.append((
            k1, k2,
            float(eg_t_arr[k]),
            float(eg_p_arr[k]),
            float(pvals_adj[k]),
            eg_pass,
            float(joh_val_arr[k]),
            joh_pass,
            verdict,
        ))

    df_out = pd.DataFrame(
        rows,
        columns=[
            "ticker1", "ticker2",
            "eg_t", "eg_p", "eg_p_fdr",
            "eg_pass", "joh_stat", "joh_pass", "verdict",
        ],
    ).set_index(["ticker1", "ticker2"])

    if only_pass:
        df_out = df_out[df_out["verdict"] == "pass"]

    # Sort by FDR-corrected p-value when BH is active, raw p otherwise
    sort_col = sort_by
    if sort_by == "eg_p" and fdr_method == "bh":
        sort_col = "eg_p_fdr"

    if sort_col is not None and sort_col in df_out.columns:
        if sort_col in ("eg_p", "eg_p_fdr"):
            df_out = df_out.sort_values(by=["verdict", sort_col], ascending=[False, True])
        elif sort_col in ("eg_t", "joh_stat"):
            df_out = df_out.sort_values(by=["verdict", sort_col], ascending=[False, False])
        else:
            df_out = df_out.sort_values(by=sort_col, ascending=True)

    return df_out
