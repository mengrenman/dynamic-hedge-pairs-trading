# tests/test_utils.py
"""Unit tests for pairs.utils (splits, normalize_multiindex)."""
import numpy as np
import pandas as pd
import pytest

from pairs.utils.splits import normalize_multiindex, split_by_date, split_train_val_test


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_multiindex_df(tickers=("A", "B"), n=50) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    frames = []
    for t in tickers:
        idx = pd.MultiIndex.from_product([[t], dates], names=["ticker", "datetime"])
        frames.append(pd.DataFrame({"close": np.ones(n)}, index=idx))
    return pd.concat(frames).sort_index()


# ── normalize_multiindex ───────────────────────────────────────────────────────

class TestNormalizeMultiindex:
    def test_already_correct_names_unchanged(self):
        df = _make_multiindex_df()
        out = normalize_multiindex(df)
        assert list(out.index.names) == ["ticker", "datetime"]

    def test_renames_levels(self):
        df = _make_multiindex_df()
        df.index = df.index.set_names(["sym", "date"])
        out = normalize_multiindex(df)
        assert list(out.index.names) == ["ticker", "datetime"]

    def test_raises_for_flat_index(self):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="MultiIndex"):
            normalize_multiindex(df)

    def test_raises_for_single_level_multiindex(self):
        idx = pd.MultiIndex.from_tuples([("A",), ("B",)], names=["ticker"])
        df = pd.DataFrame({"close": [1.0, 1.0]}, index=idx)
        with pytest.raises(ValueError, match="MultiIndex"):
            normalize_multiindex(df)

    def test_output_is_sorted(self):
        df = _make_multiindex_df()
        # Shuffle
        df = df.sample(frac=1.0, random_state=42)
        out = normalize_multiindex(df)
        assert out.index.is_monotonic_increasing

    def test_does_not_mutate_original(self):
        df = _make_multiindex_df()
        df.index = df.index.set_names(["sym", "date"])
        original_names = list(df.index.names)
        normalize_multiindex(df)
        assert list(df.index.names) == original_names  # unchanged


# ── split_by_date ─────────────────────────────────────────────────────────────

class TestSplitByDate:
    def test_no_overlap_between_splits(self):
        df = _make_multiindex_df(n=100)
        dates = df.index.get_level_values("datetime")
        train_end = dates[49]
        val_end = dates[99]
        train, val = split_by_date(df, train_end=train_end, val_end=val_end)
        t_dates = train.index.get_level_values("datetime")
        v_dates = val.index.get_level_values("datetime")
        assert len(set(t_dates) & set(v_dates)) == 0

    def test_all_rows_accounted_for(self):
        df = _make_multiindex_df(n=100)
        dates = df.index.get_level_values("datetime")
        train, val = split_by_date(df, train_end=dates[49], val_end=dates[99])
        assert len(train) + len(val) == len(df)

    def test_train_end_inclusive(self):
        df = _make_multiindex_df(n=50)
        dates = sorted(df.index.get_level_values("datetime").unique())
        train, _ = split_by_date(df, train_end=dates[24], val_end=dates[49])
        t_dates = train.index.get_level_values("datetime")
        assert pd.Timestamp(dates[24]) in t_dates


# ── split_train_val_test ───────────────────────────────────────────────────────

class TestSplitTrainValTest:
    def test_three_way_no_overlap(self):
        df = _make_multiindex_df(n=150)
        dates = df.index.get_level_values("datetime")
        train, val, test = split_train_val_test(
            df, train_end=dates[49], val_end=dates[99]
        )
        td = set(train.index.get_level_values("datetime"))
        vd = set(val.index.get_level_values("datetime"))
        xd = set(test.index.get_level_values("datetime"))
        assert len(td & vd) == 0
        assert len(td & xd) == 0
        assert len(vd & xd) == 0

    def test_all_rows_covered(self):
        df = _make_multiindex_df(n=150)
        dates = df.index.get_level_values("datetime")
        train, val, test = split_train_val_test(
            df, train_end=dates[49], val_end=dates[99]
        )
        assert len(train) + len(val) + len(test) == len(df)
