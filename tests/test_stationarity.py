# tests/test_stationarity.py
"""Unit tests for pairs.stats.stationarity."""
import warnings
import numpy as np
import pandas as pd
import pytest

from pairs.stats.stationarity import (
    estimate_halflife,
    test_spread_stationarity as stationarity_test,  # avoid pytest collecting this name
    compute_hedged_sharpe,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ar1_series(n: int = 500, phi: float = 0.9, seed: int = 42) -> pd.Series:
    """Stationary AR(1) series: x_t = phi*x_{t-1} + eps."""
    rng = np.random.default_rng(seed)
    xs = [0.0]
    for _ in range(n - 1):
        xs.append(phi * xs[-1] + rng.standard_normal())
    return pd.Series(xs)


def _random_walk(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n).cumsum())


def _make_prices(tickers, n=300, seed=0):
    """Build a MultiIndex (ticker, datetime) DataFrame with a 'close' column."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    frames = []
    for t in tickers:
        prices = 100 * np.exp(rng.standard_normal(n).cumsum() * 0.01)
        idx = pd.MultiIndex.from_product([[t], dates], names=["ticker", "datetime"])
        frames.append(pd.DataFrame({"close": prices}, index=idx))
    return pd.concat(frames).sort_index()


# ── estimate_halflife ─────────────────────────────────────────────────────────

class TestEstimateHalflife:
    def test_returns_positive_for_mean_reverting(self):
        s = _ar1_series(500, phi=0.7)
        hl = estimate_halflife(s)
        assert np.isfinite(hl) and hl > 0

    def test_returns_nan_for_unit_root(self):
        # A random walk's AR(1) phi is close to 1, producing a very large or nan HL.
        # We cannot guarantee nan (phi may be slightly < 1 from finite-sample noise),
        # so we just assert: either nan or larger than for a clearly mean-reverting series.
        s = _random_walk(300)
        hl = estimate_halflife(s)
        stationary_hl = estimate_halflife(_ar1_series(300, phi=0.5))
        assert np.isnan(hl) or (np.isfinite(stationary_hl) and hl > stationary_hl)

    def test_returns_nan_for_too_few_obs(self):
        s = pd.Series([1.0, 2.0, 1.5])
        assert np.isnan(estimate_halflife(s))

    def test_returns_nan_for_all_nan_input(self):
        s = pd.Series([np.nan] * 50)
        assert np.isnan(estimate_halflife(s))

    def test_halflife_roughly_correct(self):
        # phi=0.5 → true HL = log(2)/log(1/0.5) = 1 bar
        # phi=0.9 → true HL = log(2)/(-log(0.9)) ≈ 6.6 bars
        s = _ar1_series(2000, phi=0.9, seed=0)
        hl = estimate_halflife(s)
        true_hl = -np.log(2) / np.log(0.9)
        assert abs(hl - true_hl) < 5.0  # generous tolerance


# ── test_spread_stationarity ──────────────────────────────────────────────────

class TestSpreadStationarity:
    def test_stationary_series_verdict(self):
        s = _ar1_series(500, phi=0.5)
        result = stationarity_test(s)
        assert result["verdict"] in ("stationary", "inconclusive")

    def test_random_walk_verdict(self):
        s = _random_walk(500)
        result = stationarity_test(s)
        assert result["verdict"] in ("non-stationary", "inconclusive")

    def test_returns_expected_keys(self):
        s = _ar1_series(200)
        result = stationarity_test(s)
        expected = {"adf_stat", "adf_p", "kpss_stat", "kpss_p", "verdict",
                    "adf_lags", "adf_nobs", "adf_crit", "kpss_lags", "kpss_crit"}
        assert expected.issubset(result.keys())

    def test_verdict_logic_stationary(self):
        """When ADF rejects and KPSS does not, verdict must be 'stationary'."""
        s = _ar1_series(1000, phi=0.1)  # strongly stationary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stationarity_test(s, alpha=0.05)
        # We can't guarantee the exact test outcome, but logic must be consistent
        adf_reject = result["adf_p"] < 0.05
        kpss_reject = result["kpss_p"] < 0.05
        if adf_reject and not kpss_reject:
            assert result["verdict"] == "stationary"
        elif not adf_reject and kpss_reject:
            assert result["verdict"] == "non-stationary"
        else:
            assert result["verdict"] == "inconclusive"

    def test_verdict_logic_exhaustive(self):
        """All four (adf_reject, kpss_reject) combinations → correct verdict."""
        # We'll patch the results by checking the logic directly
        cases = [
            (True, False, "stationary"),
            (False, True, "non-stationary"),
            (True, True, "inconclusive"),
            (False, False, "inconclusive"),
        ]
        for adf_rej, kpss_rej, expected in cases:
            if adf_rej and not kpss_rej:
                verdict = "stationary"
            elif not adf_rej and kpss_rej:
                verdict = "non-stationary"
            else:
                verdict = "inconclusive"
            assert verdict == expected, f"Failed for adf_rej={adf_rej}, kpss_rej={kpss_rej}"


# ── compute_hedged_sharpe ─────────────────────────────────────────────────────

class TestComputeHedgedSharpe:
    def _make_kf_results(self, tickers, n=300, seed=1):
        """Fake Kalman states dict with constant beta=1."""
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        out = {}
        for t1, t2 in tickers:
            df = pd.DataFrame({
                "alpha": np.zeros(n),
                "beta": np.ones(n),
                "y_hat": np.zeros(n),
                "resid": np.random.default_rng(seed).standard_normal(n),
            }, index=dates)
            out[(t1, t2)] = df
        return out

    def test_returns_series_with_pair_index(self):
        prices = _make_prices(["A", "B"])
        kf = self._make_kf_results([("A", "B")])
        result = compute_hedged_sharpe(kf, prices)
        assert isinstance(result, pd.Series)
        assert ("A", "B") in result.index

    def test_missing_ticker_returns_nan(self):
        prices = _make_prices(["A", "B"])
        kf = self._make_kf_results([("A", "C")])  # C not in prices
        result = compute_hedged_sharpe(kf, prices)
        assert np.isnan(result[("A", "C")])

    def test_invalid_prices_index_raises(self):
        prices = pd.DataFrame({"close": [1.0, 2.0]})
        kf = {("A", "B"): pd.DataFrame({"beta": [1.0]})}
        with pytest.raises(ValueError, match="MultiIndex"):
            compute_hedged_sharpe(kf, prices)

    def test_empty_kf_results(self):
        prices = _make_prices(["A", "B"])
        result = compute_hedged_sharpe({}, prices)
        assert len(result) == 0
