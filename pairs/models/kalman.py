# pairs/models/kalman.py
"""
Kalman-based dynamic hedge ratio fitting (joblib-parallel).
Designed for a MultiIndex (ticker, datetime) DataFrame with a 'close' column.

Public API:
- kalman_dynamic_hedge_joblib(...)
- fit_kalman_hedge(...)  # alias to kalman_dynamic_hedge_joblib
"""

from __future__ import annotations
from itertools import combinations
from typing import Optional, Literal, Dict, Tuple, List
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from pykalman import KalmanFilter

from pairs.utils.progress import tqdm_joblib
from pairs.utils.splits import normalize_multiindex

__all__ = [
    "kalman_dynamic_hedge_joblib",
    "fit_kalman_hedge",
]

# ---------- helpers ----------
def _align_pair_multiindex(data: pd.DataFrame, k1: str, k2: str) -> pd.DataFrame:
    """
    Align two close-price series from a MultiIndex (ticker, datetime) DataFrame.
    Returns a float64 DataFrame with columns ['P1','P2'] on the datetime index.
    """
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels < 2:
        raise ValueError("data must have a MultiIndex with (ticker, datetime).")
    if "close" not in data.columns:
        raise ValueError("data must include a 'close' column.")

    s1 = data.loc[(k1,), "close"]
    s2 = data.loc[(k2,), "close"]
    if isinstance(s1.index, pd.MultiIndex):
        s1 = s1.droplevel(0)
    if isinstance(s2.index, pd.MultiIndex):
        s2 = s2.droplevel(0)

    if not isinstance(s1.index, pd.DatetimeIndex):
        s1.index = pd.to_datetime(s1.index, errors="coerce")
    if not isinstance(s2.index, pd.DatetimeIndex):
        s2.index = pd.to_datetime(s2.index, errors="coerce")
    s1 = s1.sort_index()
    s2 = s2.sort_index()

    df = pd.concat([s1.rename("P1"), s2.rename("P2")], axis=1).dropna()
    return df.astype(np.float64)

def _kalman_dynamic_hedge(
    k1: str,
    k2: str,
    df: pd.DataFrame,
    *,
    q: float = 1e-5,
    r: float = 1.0,
    init_cov: float = 1e6,
    mode: Literal["smooth", "filter"] = "smooth",
    em_iters: int = 0,
    return_params: bool = False,
):
    if df is None or len(df) < 5:
        return k1, k2, None, None

    if q <= 0 or not np.isfinite(q):
        raise ValueError(f"q (transition noise) must be positive and finite, got {q}.")
    if r <= 0 or not np.isfinite(r):
        raise ValueError(f"r (observation noise) must be positive and finite, got {r}.")
    if init_cov <= 0 or not np.isfinite(init_cov):
        raise ValueError(f"init_cov must be positive and finite, got {init_cov}.")

    y = df["P1"].values.reshape(-1, 1)
    x = df["P2"].values
    n = len(df)

    # H_t = [[x_t, 1]], state = [beta_t, alpha_t]
    H = np.zeros((n, 1, 2), dtype=np.float64)
    H[:, 0, 0] = x
    H[:, 0, 1] = 1.0
    F = np.eye(2, dtype=np.float64)

    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        initial_state_mean=np.array([0.0, 0.0], dtype=np.float64),   # [beta0, alpha0]
        initial_state_covariance=np.eye(2, dtype=np.float64) * init_cov,
        transition_covariance=np.eye(2, dtype=np.float64) * q,
        observation_covariance=np.array([[r]], dtype=np.float64),
    )

    if em_iters and em_iters > 0:
        kf = kf.em(y, n_iter=em_iters)

    if mode == "smooth":
        state_means, state_covs = kf.smooth(y)
        f_means, f_covs = kf.filter(y) if return_params else (None, None)
        last_mean = None if f_means is None else f_means[-1].copy()
        last_cov  = None if f_covs  is None else f_covs[-1].copy()
    else:
        state_means, state_covs = kf.filter(y)
        last_mean, last_cov = state_means[-1].copy(), state_covs[-1].copy()

    beta_t  = state_means[:, 0]
    alpha_t = state_means[:, 1]
    y_hat   = alpha_t + beta_t * x
    resid   = df["P1"].values - y_hat

    states_df = pd.DataFrame(
        {"alpha": alpha_t, "beta": beta_t, "y_hat": y_hat, "resid": resid},
        index=df.index,
    )

    params = None
    if return_params:
        params = {
            "F": F.copy(),
            "Q": kf.transition_covariance.copy(),
            "R": kf.observation_covariance.copy(),
            "last_state_mean": last_mean,
            "last_state_cov": last_cov,
            "mode": mode,
            "em_iters": int(em_iters),
        }

    return k1, k2, states_df, params

# ---------- main API ----------
def kalman_dynamic_hedge_joblib(
    data: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
    *,
    q: float = 1e-5,
    r: float = 1.0,
    init_cov: float = 1e6,
    mode: Literal["smooth", "filter"] = "smooth",
    em_iters: int = 0,
    require_full_span: bool = False,
    n_workers: Optional[int] = None,
    chunksize: int = 100,
    show_progress: bool = True,
    return_params: bool = False,
):
    """
    Returns:
      - if return_params=False: dict[(k1,k2)] -> states DataFrame (index=datetime)
      - if return_params=True : (states_dict, params_dict)
    """
    # Avoid BLAS oversubscription
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS",      "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

    if "close" not in data.columns:
        raise ValueError("data must include a 'close' column.")
    data = normalize_multiindex(data)

    # Native Python strings for tickers
    tickers: List[str] = [str(k) for k in data.index.get_level_values("ticker").unique().tolist()]

    # Build pair list
    if pairs is None:
        idx_pairs: List[Tuple[str, str]] = list(combinations(tickers, 2))
    else:
        idx_pairs = [(str(a), str(b)) for (a, b) in pairs]

    # Optional: full-span coverage
    if require_full_span:
        dt = data.index.get_level_values("datetime")
        full_start, full_end = dt.min(), dt.max()

        def _full_span_ok(k: str) -> bool:
            s = data.loc[(k,), "close"]
            if isinstance(s.index, pd.MultiIndex):
                s = s.droplevel(0)
            return (len(s) > 0) and (s.index.min() == full_start) and (s.index.max() == full_end)

        idx_pairs = [(a, b) for (a, b) in idx_pairs if _full_span_ok(a) and _full_span_ok(b)]

    n_workers = n_workers or os.cpu_count() or 1

    def worker(k1: str, k2: str):
        df_ab = _align_pair_multiindex(data, k1, k2)
        return _kalman_dynamic_hedge(
            k1, k2, df_ab,
            q=q, r=r, init_cov=init_cov,
            mode=mode, em_iters=em_iters,
            return_params=return_params,
        )

    iterator = (delayed(worker)(k1, k2) for (k1, k2) in idx_pairs)

    if show_progress:
        with tqdm_joblib(tqdm(total=len(idx_pairs), desc="Kalman (α_t, β_t)", leave=False)):
            results = Parallel(n_jobs=n_workers, prefer="processes", batch_size=chunksize)(iterator)
    else:
        results = Parallel(n_jobs=n_workers, prefer="processes", batch_size=chunksize)(iterator)

    states_out: Dict[Tuple[str, str], pd.DataFrame] = {}
    params_out: Dict[Tuple[str, str], dict] = {}

    for k1, k2, df_res, params in results:
        if df_res is not None:
            states_out[(k1, k2)] = df_res
            if return_params and params is not None:
                params_out[(k1, k2)] = params

    if return_params:
        return states_out, params_out
    return states_out

# --------- Alias for top-level convenience ---------
def fit_kalman_hedge(*args, **kwargs):
    """Alias to kalman_dynamic_hedge_joblib for backward/forward compatibility."""
    return kalman_dynamic_hedge_joblib(*args, **kwargs)


# --- Kalman continuation on new windows (validation/test) --------------------
from typing import Optional, Dict, Tuple, Any
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# Extend public API
try:
    __all__
except NameError:
    __all__ = []
__all__ += ["filter_kf_on_new", "continue_kalman_on_window", "continue_kalman_for_pairs_joblib"]

def _align_two(series1: pd.Series, series2: pd.Series, name1="P1", name2="P2") -> pd.DataFrame:
    if not isinstance(series1.index, pd.DatetimeIndex):
        series1 = series1.copy(); series1.index = pd.to_datetime(series1.index, errors="coerce")
    if not isinstance(series2.index, pd.DatetimeIndex):
        series2 = series2.copy(); series2.index = pd.to_datetime(series2.index, errors="coerce")
    series1 = series1.sort_index()
    series2 = series2.sort_index()
    df = pd.concat([series1.rename(name1), series2.rename(name2)], axis=1, join="inner").dropna()
    return df.astype(np.float64)

def _coerce_last_state(
    last_state: Any,
    init_cov: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts:
      - dict with keys 'mean' & 'cov' (or 'last_state_mean', 'last_state_cov')
      - tuple/list of (mean, cov)
      - np.ndarray of shape (2,) for mean (cov = init_cov*I)
      - np.ndarray of shape (2,2) for cov (mean = zeros)
      - None  -> (zeros, init_cov*I)
    Returns (mean(2,), cov(2,2)) as float64.
    """
    I = np.eye(2, dtype=np.float64)
    if last_state is None:
        return np.zeros(2, dtype=np.float64), I * float(init_cov)

    # dict
    if isinstance(last_state, dict):
        mean = last_state.get("mean", last_state.get("last_state_mean", None))
        cov  = last_state.get("cov",  last_state.get("last_state_cov",  None))
        if mean is None and cov is None:
            raise ValueError("last_state dict must contain 'mean' and/or 'cov'.")
        if mean is None:
            mean = np.zeros(2, dtype=np.float64)
        if cov is None:
            cov = I * float(init_cov)
        return np.asarray(mean, dtype=np.float64).reshape(2,), np.asarray(cov, dtype=np.float64).reshape(2,2)

    # tuple/list
    if isinstance(last_state, (tuple, list)):
        if len(last_state) != 2:
            raise ValueError("last_state tuple/list must be (mean, cov).")
        mean, cov = last_state
        if mean is None:
            mean = np.zeros(2, dtype=np.float64)
        if cov is None:
            cov = I * float(init_cov)
        return np.asarray(mean, dtype=np.float64).reshape(2,), np.asarray(cov, dtype=np.float64).reshape(2,2)

    # ndarray only
    arr = np.asarray(last_state)
    if arr.shape == (2,):
        return arr.astype(np.float64), I * float(init_cov)
    if arr.shape == (2, 2):
        return np.zeros(2, dtype=np.float64), arr.astype(np.float64)

    raise TypeError("Unsupported last_state type/shape.")

def filter_kf_on_new(
    P1_new: pd.Series,
    P2_new: pd.Series,
    *,
    frozen: Dict[str, np.ndarray],          # {"F":2x2, "Q":2x2, "R":1x1}
    last_state: Optional[Any] = None,       # dict | (mean,cov) | ndarray | None
    init_cov: float = 1e6,
    mode: str = "filter",                    # or "smooth"
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Continue a trained Kalman filter on aligned Series P1_new/P2_new.

    Returns:
      - states DataFrame indexed by datetime with ['alpha','beta','y_hat','resid']
      - last_state dict: {'mean': (2,), 'cov': (2,2)} at the end of this window
    """
    # 1) Align
    df = _align_two(P1_new, P2_new, "P1", "P2")
    if len(df) < 5:
        raise ValueError("Not enough overlapping observations after alignment.")

    # 2) Inputs
    y = df["P1"].values.reshape(-1, 1)
    x = df["P2"].values
    n = len(df)

    H = np.zeros((n, 1, 2), dtype=np.float64)
    H[:, 0, 0] = x
    H[:, 0, 1] = 1.0

    F = np.asarray(frozen["F"], dtype=np.float64)
    Q = np.asarray(frozen["Q"], dtype=np.float64)
    R = np.asarray(frozen["R"], dtype=np.float64)

    init_mean, init_covm = _coerce_last_state(last_state, init_cov)

    # 3) Run KF
    kf = KalmanFilter(
        transition_matrices=F,
        observation_matrices=H,
        initial_state_mean=init_mean,
        initial_state_covariance=init_covm,
        transition_covariance=Q,
        observation_covariance=R,
    )

    if mode == "smooth":
        state_means, state_covs = kf.smooth(y)
        f_means, f_covs = kf.filter(y)  # to export filtered terminal state
        last_mean, last_cov = f_means[-1].copy(), f_covs[-1].copy()
    else:
        state_means, state_covs = kf.filter(y)
        last_mean, last_cov = state_means[-1].copy(), state_covs[-1].copy()

    beta_t  = state_means[:, 0]
    alpha_t = state_means[:, 1]
    y_hat   = alpha_t + beta_t * x
    resid   = df["P1"].values - y_hat

    out = pd.DataFrame(
        {"alpha": alpha_t, "beta": beta_t, "y_hat": y_hat, "resid": resid},
        index=df.index,
    )

    # Standardize the returned state format
    last_state_out = {"mean": last_mean, "cov": last_cov}
    return out, last_state_out

# --------- Convenience wrappers for MultiIndex DataFrames & params dict ---------

def continue_kalman_on_window(
    data: pd.DataFrame,
    k1: str,
    k2: str,
    params: Dict[str, np.ndarray],          # from training: {'F','Q','R','last_state_mean','last_state_cov', ...}
    *,
    mode: str = "filter",
    init_cov: float = 1e6,
    return_params: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, np.ndarray]]]:
    """
    Continue for a single pair on a new window held in `data` (MultiIndex: ticker, datetime).

    Returns:
      states_df, and (optionally) an updated params dict with advanced last state.
    """
    # Extract aligned Series from MultiIndex window
    s1 = data.loc[(k1,), "close"];  s2 = data.loc[(k2,), "close"]
    if isinstance(s1.index, pd.MultiIndex): s1 = s1.droplevel(0)
    if isinstance(s2.index, pd.MultiIndex): s2 = s2.droplevel(0)

    frozen = {"F": params["F"], "Q": params["Q"], "R": params["R"]}
    last_state = (params.get("last_state_mean"), params.get("last_state_cov"))

    df_states, last_state_out = filter_kf_on_new(
        s1, s2,
        frozen=frozen,
        last_state=last_state,
        init_cov=init_cov,
        mode=mode,
    )

    if not return_params:
        return df_states, None

    params_out = {
        "F": params["F"].copy(),
        "Q": params["Q"].copy(),
        "R": params["R"].copy(),
        "last_state_mean": last_state_out["mean"],
        "last_state_cov": last_state_out["cov"],
        "mode": mode,
        "em_iters": int(params.get("em_iters", 0)),
    }
    return df_states, params_out

def continue_kalman_for_pairs_joblib(
    data: pd.DataFrame,
    params_dict: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    pairs: Optional[list[Tuple[str, str]]] = None,
    *,
    mode: str = "filter",
    init_cov: float = 1e6,
    n_workers: Optional[int] = None,
    chunksize: int = 100,
    show_progress: bool = True,
    return_params: bool = False,
) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], Optional[Dict[Tuple[str, str], Dict[str, np.ndarray]]]]:
    """
    Parallel continuation for many pairs on a new window `data` (MultiIndex).
    """
    # Normalize index like in the fitter
    if "close" not in data.columns:
        raise ValueError("data must include a 'close' column.")
    data = normalize_multiindex(data)

    # Worklist
    work_pairs = list(params_dict.keys()) if pairs is None else [(str(a), str(b)) for (a, b) in pairs if (a, b) in params_dict]

    # Parallel
    n_jobs = n_workers or os.cpu_count() or 1
    parallel_kwargs = dict(n_jobs=n_jobs, prefer="processes", batch_size=chunksize)
    iterator = (
        delayed(continue_kalman_on_window)(
            data, k1, k2, params_dict[(k1, k2)],
            mode=mode, init_cov=init_cov, return_params=return_params
        )
        for (k1, k2) in work_pairs
    )
    if show_progress:
        with tqdm_joblib(tqdm(total=len(work_pairs), desc="Kalman OOS", leave=False)):
            results = Parallel(**parallel_kwargs)(iterator)
    else:
        results = Parallel(**parallel_kwargs)(iterator)

    states_out: Dict[Tuple[str, str], pd.DataFrame] = {}
    params_out: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for (k1, k2), (df_states, P_out) in zip(work_pairs, results):
        if df_states is not None and not df_states.empty:
            states_out[(k1, k2)] = df_states
            if return_params and P_out is not None:
                params_out[(k1, k2)] = P_out

    return (states_out, params_out) if return_params else (states_out, None)

