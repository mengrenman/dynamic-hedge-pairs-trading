# pairs/stats/stationarity.py
"""
Stationarity diagnostics for Kalman (or any) spreads:
- estimate_halflife(...)
- test_spread_stationarity(...)
- summarize_spread_stationarity_joblib(...)
- compute_hedged_sharpe(...)

These accept generic residual/spread series or the dict of DataFrames returned
by kalman_dynamic_hedge_joblib(...).

Notes:
- The summary function now returns a column named 'shapre' (intentional name),
  which is:
    * by default: Sharpe of residual changes Δε_t (fast proxy);
    * if `prices` is provided: hedged-return Sharpe r^H_t = r1_t - β_{t-1} r2_t,
      falling back to the proxy where needed.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Literal
import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from pairs.utils.progress import tqdm_joblib
from pairs.utils.splits import normalize_multiindex

__all__ = [
    "estimate_halflife",
    "test_spread_stationarity",
    "summarize_spread_stationarity_joblib",
    "compute_hedged_sharpe",
]

# ----------------- Half-life estimation -----------------
def estimate_halflife(resid: pd.Series) -> float:
    """
    Estimate half-life of mean reversion from residual series.

    Assumes an AR(1) process: x_t = a + b*x_{t-1} + e_t.
    Half-life = -log(2) / log(b).

    Returns np.nan if:
    - fewer than 20 observations after dropping NaNs
    - AR(1) coefficient b <= 0 (no mean reversion) or b >= 1 (unit root or explosive)
    - series has drift or trend that biases the AR(1) estimate

    Note: This estimator can be biased for series with drift or structural breaks.
    """
    r = pd.Series(resid).dropna().astype(float)
    if len(r) < 20:
        return np.nan
    r_lag = r.shift(1).dropna()
    y = r.loc[r_lag.index]
    x = r_lag
    X = np.column_stack([np.ones(len(x)), x.values])   # OLS y = a + b x
    beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
    b = float(beta[1])
    if b <= 0 or b >= 1 or not np.isfinite(b):
        return np.nan
    return -np.log(2.0) / np.log(b)

# ----------------- Stationarity tests -------------------
def test_spread_stationarity(spread: pd.Series, alpha: float = 0.05, regression: str = "c") -> dict:
    s = pd.Series(spread).astype(float).dropna()
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, *_ = adfuller(s, autolag="AIC", regression=regression)
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
        s, regression="c" if regression == "c" else "ct", nlags="auto"
    )
    adf_reject  = adf_p < alpha
    kpss_reject = kpss_p < alpha
    # ADF H0: unit-root (non-stationary). KPSS H0: stationary.
    # Stationary:     ADF rejects (evidence against unit-root) AND KPSS does not reject (consistent with stationarity).
    # Non-stationary: ADF does not reject AND KPSS rejects (both agree series is non-stationary).
    # Inconclusive:   Both reject or neither rejects (tests contradict or are both ambiguous).
    if adf_reject and not kpss_reject:
        verdict = "stationary"
    elif (not adf_reject) and kpss_reject:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive"
    return {
        "adf_stat": adf_stat, "adf_p": adf_p,
        "adf_lags": adf_lags, "adf_nobs": adf_nobs, "adf_crit": adf_crit,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p,
        "kpss_lags": kpss_lags, "kpss_crit": kpss_crit,
        "verdict": verdict
    }

# ----------------- Hedged-return Sharpe (selection metric) -------------------
def compute_hedged_sharpe(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    prices: pd.DataFrame,
    *,
    periods_per_year: float = 252.0,
    use_log_returns: bool = True,
    lag_beta: int = 1,
    min_obs: int = 10,
) -> pd.Series:
    """
    Annualized Sharpe ratio of the dollar-neutral hedged return for each pair.

    Hedged return:  r^H_t = r1_t - beta_{t-1} * r2_t
      - Uses beta from the states DataFrame in kf_results[(k1,k2)]['beta'].
      - `lag_beta` avoids look-ahead (default 1 step).
      - `prices` must be a MultiIndex DataFrame (ticker, datetime) with column 'close'.
    """
    # Normalize and validate prices index
    prices = normalize_multiindex(prices)

    out: Dict[Tuple[str, str], float] = {}

    for (k1, k2), st in kf_results.items():
        if st is None or st.empty or "beta" not in st.columns:
            out[(k1, k2)] = np.nan
            continue

        try:
            s1 = prices.loc[(k1,), "close"]
            s2 = prices.loc[(k2,), "close"]
        except KeyError:
            out[(k1, k2)] = np.nan
            continue

        # Drop ticker level to align on datetime
        if isinstance(s1.index, pd.MultiIndex): s1 = s1.droplevel(0)
        if isinstance(s2.index, pd.MultiIndex): s2 = s2.droplevel(0)

        df = (
            st[["beta"]].copy()
              .join(s1.rename("P1"))
              .join(s2.rename("P2"))
              .dropna()
              .sort_index()
        )
        if df.empty:
            out[(k1, k2)] = np.nan
            continue

        # Returns
        if use_log_returns:
            r1 = np.log(df["P1"]).diff()
            r2 = np.log(df["P2"]).diff()
        else:
            r1 = df["P1"].pct_change()
            r2 = df["P2"].pct_change()

        beta_lag = df["beta"].shift(lag_beta)
        hedged = (r1 - beta_lag * r2).dropna()

        if len(hedged) < min_obs:
            out[(k1, k2)] = np.nan
            continue

        vol = float(hedged.std(ddof=1))
        if not np.isfinite(vol) or vol == 0.0:
            out[(k1, k2)] = np.nan
            continue

        mu = float(hedged.mean())
        sharpe = mu / vol * np.sqrt(float(periods_per_year))
        out[(k1, k2)] = sharpe

    return pd.Series(out)

# ----------------- Parallel summary over many pairs --------------------------
def summarize_spread_stationarity_joblib(
    kf_results: Dict[Tuple[str, str], pd.DataFrame],
    *,
    alpha: float = 0.05,
    regression: str = "c",
    n_workers: Optional[int] = None,
    chunksize: int = 256,
    show_progress: bool = True,
    suppress_warnings: bool = True,
    # --- shapre controls ---
    prices: Optional[pd.DataFrame] = None,
    shapre_method: Literal["auto", "hedged", "resid"] = "auto",
    shapre_periods_per_year: float = 252.0,
    shapre_use_log_returns: bool = True,
    shapre_lag_beta: int = 1,
    shapre_min_obs: int = 10,
) -> pd.DataFrame:
    """
    Parallel summary of ADF+KPSS stationarity tests for 'resid'
    plus half-life estimation, residual sigma, and a 'shapre' column.

    shapre:
      - If `prices` provided and shapre_method in {'auto','hedged'}:
          compute hedged-return Sharpe (annualized) per pair using β_{t-1}.
          If unavailable for a pair, fallback to residual-change proxy.
      - Else (no prices or shapre_method='resid'):
          compute Sharpe of Δresid_t (annualized proxy).

    Returns an index (ticker1, ticker2) DataFrame with columns:
      ["adf_stat","adf_p","kpss_stat","kpss_p","halflife","resid_sigma","shapre","verdict"]
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS",      "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

    n_workers = n_workers or os.cpu_count()

    tasks = []
    for (k1, k2), df in kf_results.items():
        if "resid" not in df.columns:
            continue
        resid_values = df["resid"].to_numpy(dtype=float, copy=False)
        tasks.append((k1, k2, resid_values))

    def worker_with_sigma(k1, k2, resid_values):
        s = pd.Series(resid_values)
        if s.dropna().empty:
            return None
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                res = test_spread_stationarity(s, alpha=alpha, regression=regression)
        else:
            res = test_spread_stationarity(s, alpha=alpha, regression=regression)

        # Reuse canonical half-life estimator (AR(1) OLS, avoids duplication)
        halflife = estimate_halflife(s)

        resid_sigma = float(s.std())

        # Residual-change Sharpe proxy (annualized)
        shapre_proxy = np.nan
        try:
            ret = s.diff().dropna()
            if len(ret) >= shapre_min_obs:
                mu = float(ret.mean())
                vol = float(ret.std(ddof=1))
                if np.isfinite(vol) and vol > 0:
                    shapre_proxy = mu / vol * np.sqrt(float(shapre_periods_per_year))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass

        return {
            "ticker1": k1, "ticker2": k2,
            "adf_stat": res["adf_stat"], "adf_p": res["adf_p"],
            "kpss_stat": res["kpss_stat"], "kpss_p": res["kpss_p"],
            "halflife": halflife,
            "resid_sigma": resid_sigma,
            "shapre": shapre_proxy,   # may be replaced by hedged Sharpe later
            "verdict": res["verdict"],
        }

    iterator = (delayed(worker_with_sigma)(k1, k2, resid_values) for (k1, k2, resid_values) in tasks)

    if show_progress:
        with tqdm_joblib(tqdm(total=len(tasks), desc="Testing stationarity", leave=False)):
            results = Parallel(n_jobs=n_workers, prefer="processes", batch_size=chunksize)(iterator)
    else:
        results = Parallel(n_jobs=n_workers, prefer="processes", batch_size=chunksize)(iterator)

    rows = [r for r in results if r is not None]
    if not rows:
        return pd.DataFrame(
            columns=["adf_stat","adf_p","kpss_stat","kpss_p","halflife","resid_sigma","shapre","verdict"],
            index=pd.MultiIndex.from_tuples([], names=["ticker1","ticker2"])
        )

    out = pd.DataFrame(rows).set_index(["ticker1", "ticker2"])
    col_order = ["adf_stat", "adf_p", "kpss_stat", "kpss_p", "halflife", "resid_sigma", "shapre", "verdict"]
    out = out[col_order]

    # If prices provided and hedged Sharpe requested, compute and overlay
    if prices is not None and shapre_method in {"auto", "hedged"}:
        try:
            hedged = compute_hedged_sharpe(
                kf_results, prices,
                periods_per_year=shapre_periods_per_year,
                use_log_returns=shapre_use_log_returns,
                lag_beta=shapre_lag_beta,
                min_obs=shapre_min_obs,
            )
            hedged = hedged.reindex(out.index)
            if shapre_method == "hedged":
                out["shapre"] = hedged
            else:  # auto: prefer hedged where available, else keep proxy
                out["shapre"] = hedged.fillna(out["shapre"])
        except (ValueError, KeyError, TypeError) as e:
            warnings.warn(
                f"compute_hedged_sharpe failed ({type(e).__name__}: {e}); "
                "falling back to residual-change Sharpe proxy.",
                RuntimeWarning,
                stacklevel=2,
            )

    return out.sort_values(["verdict", "adf_p"])
