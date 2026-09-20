"""Unit tests for pairs.stats.microstructure on simulated intraday prices."""
import numpy as np
import pandas as pd
import pytest

from pairs.stats.microstructure import (
    autocorr_by_interval,
    epps_correlation,
    realized_variance_signature,
    roll_spread,
)

N_SESSIONS, N_BARS = 60, 390


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2024-01-01", periods=N_SESSIONS)
    idx = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=N_BARS, freq="min") for d in days]))
    n = len(idx)
    efficient = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))          # 2 bp per-minute vol
    bounce = pd.Series(efficient + 0.02 * rng.choice([-1, 1], n), index=idx)   # half-spread 2 bp -> Roll 4 bp
    common = np.cumsum(rng.normal(0, 2e-4, n))
    a = common + np.cumsum(rng.normal(0, 1e-4, n))
    b = common + np.cumsum(rng.normal(0, 1e-4, n))

    def stale(x, k):   # each minute shows the price from a random 0..k minutes earlier
        lag = rng.integers(0, k + 1, len(x))
        return x[np.maximum(np.arange(len(x)) - lag, 0)]

    return {
        "idx": idx,
        "efficient": pd.Series(efficient, index=idx),
        "bounce": bounce,
        "a": pd.Series(100 * np.exp(a), index=idx), "b": pd.Series(100 * np.exp(b), index=idx),
        "a_stale": pd.Series(100 * np.exp(stale(a, 3)), index=idx), "b_stale": pd.Series(100 * np.exp(stale(b, 3)), index=idx),
    }


class TestRollSpread:
    def test_recovers_simulated_spread(self, sim):
        assert abs(roll_spread(sim["bounce"]) - 4.0) < 0.4

    def test_nan_without_negative_serial_covariance(self):
        trend = pd.Series(np.arange(100, dtype=float))         # all changes +1 -> covariance 0
        assert np.isnan(roll_spread(trend))
        assert np.isnan(roll_spread(pd.Series([1.0, 2.0, 1.0] * 5)))   # fewer than min_obs changes

    def test_price_units(self, sim):
        bps, dollars = roll_spread(sim["bounce"]), roll_spread(sim["bounce"], as_bps=False)
        assert np.isclose(dollars / sim["bounce"].mean() * 1e4, bps)

    def test_signed_keeps_the_cells_that_would_otherwise_be_dropped(self):
        """Discarding non-negative covariances keeps only the draws that looked expensive.

        A pure random walk has no spread at all. Averaging the cells that survive the NaN filter
        returns a firmly positive number; averaging the signed estimates returns roughly zero,
        which is the truth.
        """
        rng = np.random.default_rng(3)
        walks = [pd.Series(100 + np.cumsum(rng.normal(0, 0.05, 400))) for _ in range(300)]
        dropped = np.array([roll_spread(w) for w in walks])
        signed = np.array([roll_spread(w, signed=True) for w in walks])
        assert np.isnan(dropped).mean() > 0.3, "the filter should be discarding a lot"
        # 1.83 bps of spread on a series that has none, against 0.23 for the signed mean
        assert np.nanmean(dropped) > 1.5, "what survives the filter looks expensive"
        assert abs(np.mean(signed)) < 0.5, "the signed mean should sit near zero"

    def test_session_grouping_excludes_the_overnight_gap(self, sim):
        """An overnight jump is not a bid-ask bounce and must not enter the covariance.

        At one-minute sampling the guard barely matters -- 60 gaps among 23,400 changes cannot
        move a covariance far. It matters once the sampling coarsens, because each gap then sits
        among far fewer ordinary changes: at five minutes the ungrouped estimate is inflated by
        half. Five minutes is exactly where this repository measures its costs.
        """
        day = sim["bounce"].index.normalize()
        jump = pd.Series(np.cumsum(np.where(np.arange(N_SESSIONS) % 2, 5.0, -5.0)),
                         index=pd.DatetimeIndex(sorted(day.unique())))
        gapped = sim["bounce"] + day.map(jump).to_numpy()
        for every in (1, 5, 15):
            assert abs(roll_spread(gapped, session=day, every=every)
                       - roll_spread(sim["bounce"], session=day, every=every)) < 0.2
        assert abs(roll_spread(gapped.iloc[::1]) - 4.0) < 0.4          # ungrouped, still fine
        assert roll_spread(gapped.iloc[::5]) > 5.5                      # ungrouped, inflated 50%

    def test_every_pools_across_sessions_so_coarse_sampling_stays_usable(self, sim):
        day = sim["bounce"].index.normalize()
        # sampling every fifth bar leaves the bounce at the sampled endpoints untouched, so the
        # estimate should not move; what changes is how many observations are left to see it
        assert abs(roll_spread(sim["bounce"], session=day, every=5, min_obs=200) - 4.0) < 0.4
        one = sim["bounce"].iloc[:N_BARS]
        assert np.isnan(roll_spread(one, session=day[:N_BARS], every=5, min_obs=200)), \
            "76 pairs in a single session is not enough; 4,560 pooled across 60 is"


class TestSignature:
    def test_random_walk_is_flat_and_noise_inflates_fine_sampling(self, sim):
        flat = realized_variance_signature(sim["efficient"], (1, 5, 30))
        assert 0.8 < flat[1] / flat[30] < 1.25
        assert abs(flat[1] * 1e8 - 4.0) < 0.5                      # per-bar variance (2e-4)^2
        noisy = realized_variance_signature(sim["bounce"], (1, 5, 30))
        assert noisy[1] / noisy[30] > 2.0
        assert noisy[1] > flat[1] and abs(noisy[30] - flat[30]) / flat[30] < 0.3

    def test_intervals_never_span_sessions(self, sim):
        # a huge overnight jump must not enter any interval's variance
        s = sim["efficient"].copy()
        s.iloc[N_BARS:] *= 3.0
        rv = realized_variance_signature(s, (1, 30))
        assert rv[30] < 1e-5                                       # a 3x jump would add ~1.2 to the sum


class TestEppsAndAutocorr:
    def test_epps_effect_from_stale_prices(self, sim):
        stale = epps_correlation(sim["a_stale"], sim["b_stale"], (1, 5, 30))
        fresh = epps_correlation(sim["a"], sim["b"], (1, 5, 30))
        assert stale[1] < stale[30] - 0.3
        assert abs(fresh[1] - fresh[30]) < 0.1
        assert abs(fresh[30] - 0.8) < 0.05

    def test_bounce_gives_negative_first_order_autocorrelation(self, sim):
        acf = autocorr_by_interval(sim["bounce"], (1, 30))
        assert acf[1] < -0.2 and abs(acf[30]) < 0.1
        assert abs(autocorr_by_interval(sim["efficient"], (1,))[1]) < 0.05
