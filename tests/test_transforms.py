# tests/test_transforms.py
"""Unit tests for pairs.stats.transforms."""
import numpy as np
import pandas as pd
import pytest

from pairs.stats.transforms import zscore, rolling_zscore, robust_z, Z


class TestZscore:
    def test_mean_zero_std_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std(ddof=0) - 1.0) < 1e-10

    def test_constant_series_returns_zeros(self):
        s = pd.Series([3.0] * 10)
        z = zscore(s)
        assert (z == 0.0).all()

    def test_alias_z_matches_zscore(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert (Z(s) == zscore(s)).all()

    def test_handles_nan_input(self):
        s = pd.Series([1.0, np.nan, 3.0])
        z = zscore(s)
        assert z.isna().sum() == 1  # NaN propagates

    def test_preserves_index(self):
        idx = pd.date_range("2020-01-01", periods=5)
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        z = zscore(s)
        assert (z.index == idx).all()


class TestRollingZscore:
    def test_output_length_matches_input(self):
        s = pd.Series(np.random.randn(100))
        z = rolling_zscore(s, window=20)
        assert len(z) == len(s)

    def test_no_inf_values(self):
        s = pd.Series(np.random.randn(50))
        z = rolling_zscore(s, window=10)
        assert not np.isinf(z).any()

    def test_constant_window_returns_zero(self):
        # A window of constant values → std=0 → z should be 0
        s = pd.Series([5.0] * 30 + [1.0] * 10)
        z = rolling_zscore(s, window=20)
        # First 20 bars are all 5.0 → std=0 → z=0 (not nan/inf)
        assert z.iloc[19] == 0.0

    def test_accepts_optional_min_periods(self):
        s = pd.Series(np.random.randn(50))
        z = rolling_zscore(s, window=20, min_periods=10)
        assert len(z) == 50


class TestRobustZ:
    def test_output_length_matches_input(self):
        s = pd.Series(np.random.randn(100))
        z = robust_z(s)
        assert len(z) == len(s)

    def test_constant_series_returns_zeros(self):
        s = pd.Series([7.0] * 20)
        z = robust_z(s)
        assert (z == 0.0).all()

    def test_no_inf_values(self):
        s = pd.Series(np.random.randn(200))
        z = robust_z(s)
        assert not np.isinf(z).any()

    def test_median_centered(self):
        # The robust z of the median should be 0
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = robust_z(s)
        assert abs(z.iloc[2]) < 1e-10  # median is 3.0 → z=0
