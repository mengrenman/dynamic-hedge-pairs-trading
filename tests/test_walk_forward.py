# tests/test_walk_forward.py
"""Unit tests for pairs.validation.walk_forward."""
import numpy as np
import pandas as pd
import pytest

from pairs.validation.walk_forward import walk_forward_splits, walk_forward_backtest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pair_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Minimal pair DataFrame with DatetimeIndex and P1, P2, resid columns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = 0.8 * resid[i - 1] + rng.standard_normal()
    return pd.DataFrame({
        "P1":    100 + rng.standard_normal(n).cumsum(),
        "P2":    100 + rng.standard_normal(n).cumsum(),
        "resid": resid,
        "beta":  np.ones(n),
    }, index=idx)


def _trivial_fit(df_train):
    """Fit function: return last residual value as 'model'."""
    return {"mu": float(df_train["resid"].mean()), "sigma": float(df_train["resid"].std())}


def _trivial_signals(df_test, artefact):
    """Signal function: all flat."""
    return pd.DataFrame({
        "pos": 0, "n1": 0.0, "n2": 0.0,
        "z": 0.0, "entry": False, "exit": False, "stop": False,
    }, index=df_test.index)


def _trivial_eval(df_test, signals):
    """Eval function: return a fixed-shape metrics dict."""
    return {
        "sharpe": 0.0,
        "ann_return": 0.0,
        "n_trades": 0,
        "hit_rate": float("nan"),
        "max_drawdown_pct": 0.0,
        "profit_factor": float("nan"),
    }


# ── walk_forward_splits ───────────────────────────────────────────────────────

class TestWalkForwardSplits:
    def test_basic_split_count(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        splits = walk_forward_splits(idx, train_bars=200, test_bars=50)
        # starts: 0; trains until 200; tests 200-250; step=50 → start=50
        # starts: 50; trains until 250; tests 250-300 → 1 more
        assert len(splits) == 2

    def test_no_overlap_between_train_and_test(self):
        idx = pd.date_range("2020-01-01", periods=400, freq="B")
        for train_idx, test_idx in walk_forward_splits(idx, train_bars=200, test_bars=100):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_train_length_is_fixed(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        splits = walk_forward_splits(idx, train_bars=200, test_bars=100, step_bars=50)
        for train_idx, _ in splits:
            assert len(train_idx) == 200

    def test_test_length_le_test_bars(self):
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        splits = walk_forward_splits(idx, train_bars=200, test_bars=100, step_bars=50)
        for _, test_idx in splits:
            assert len(test_idx) <= 100

    def test_test_indices_contiguous_and_ordered(self):
        idx = pd.date_range("2020-01-01", periods=600, freq="B")
        splits = walk_forward_splits(idx, train_bars=300, test_bars=100)
        for _, test_idx in splits:
            assert test_idx.is_monotonic_increasing

    def test_folds_advance_in_time(self):
        idx = pd.date_range("2020-01-01", periods=600, freq="B")
        splits = walk_forward_splits(idx, train_bars=200, test_bars=100, step_bars=100)
        for i in range(1, len(splits)):
            assert splits[i][1][0] > splits[i - 1][1][0]

    def test_raises_on_zero_train_bars(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        with pytest.raises(ValueError, match="train_bars"):
            walk_forward_splits(idx, train_bars=0, test_bars=50)

    def test_raises_on_zero_test_bars(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        with pytest.raises(ValueError, match="test_bars"):
            walk_forward_splits(idx, train_bars=100, test_bars=0)

    def test_empty_when_train_plus_test_exceeds_length(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        splits = walk_forward_splits(idx, train_bars=40, test_bars=20)
        assert splits == []

    def test_single_fold_when_barely_fits(self):
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        splits = walk_forward_splits(idx, train_bars=200, test_bars=100)
        assert len(splits) == 1

    def test_empty_list_when_nothing_fits(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        splits = walk_forward_splits(idx, train_bars=8, test_bars=5, min_test_bars=5)
        assert len(splits) == 0


# ── walk_forward_backtest ─────────────────────────────────────────────────────

class TestWalkForwardBacktest:
    def test_returns_dataframe(self):
        df = _make_pair_df(500)
        results = walk_forward_backtest(
            df, train_bars=252, test_bars=63, step_bars=63,
            fit_fn=_trivial_fit, signal_fn=_trivial_signals,
            eval_fn=_trivial_eval, verbose=False,
        )
        assert isinstance(results, pd.DataFrame)

    def test_correct_fold_count(self):
        n = 500
        df = _make_pair_df(n)
        train_bars, test_bars, step_bars = 252, 63, 63
        results = walk_forward_backtest(
            df, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars,
            fit_fn=_trivial_fit, signal_fn=_trivial_signals,
            eval_fn=_trivial_eval, verbose=False,
        )
        expected_splits = walk_forward_splits(
            df.index, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars
        )
        assert len(results) == len(expected_splits)

    def test_result_has_meta_columns(self):
        df = _make_pair_df(400)
        results = walk_forward_backtest(
            df, train_bars=250, test_bars=75, step_bars=75,
            fit_fn=_trivial_fit, signal_fn=_trivial_signals,
            eval_fn=_trivial_eval, verbose=False,
        )
        for col in ["train_start", "train_end", "test_start", "test_end",
                    "n_train", "n_test"]:
            assert col in results.columns

    def test_result_has_eval_metric_columns(self):
        df = _make_pair_df(400)
        results = walk_forward_backtest(
            df, train_bars=250, test_bars=75, step_bars=75,
            fit_fn=_trivial_fit, signal_fn=_trivial_signals,
            eval_fn=_trivial_eval, verbose=False,
        )
        for col in ["sharpe", "ann_return", "n_trades"]:
            assert col in results.columns

    def test_n_train_equals_train_bars(self):
        df = _make_pair_df(400)
        results = walk_forward_backtest(
            df, train_bars=250, test_bars=75, step_bars=75,
            fit_fn=_trivial_fit, signal_fn=_trivial_signals,
            eval_fn=_trivial_eval, verbose=False,
        )
        assert (results["n_train"] == 250).all()

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"P1": [1.0, 2.0], "P2": [1.0, 2.0]})
        with pytest.raises(TypeError, match="DatetimeIndex"):
            walk_forward_backtest(
                df, train_bars=1, test_bars=1,
                fit_fn=_trivial_fit, signal_fn=_trivial_signals,
                eval_fn=_trivial_eval, verbose=False,
            )

    def test_duplicate_dates_raises(self):
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"P1": [1.0, 2.0, 3.0], "P2": [1.0, 2.0, 3.0]}, index=idx)
        with pytest.raises(ValueError, match="duplicate"):
            walk_forward_backtest(
                df, train_bars=1, test_bars=1,
                fit_fn=_trivial_fit, signal_fn=_trivial_signals,
                eval_fn=_trivial_eval, verbose=False,
            )

    def test_fit_fn_exception_skips_fold_with_warning(self):
        df = _make_pair_df(400)
        call_count = [0]

        def bad_fit(df_train):
            call_count[0] += 1
            raise RuntimeError("fit failed")

        with pytest.warns(UserWarning, match="fit_fn raised"):
            results = walk_forward_backtest(
                df, train_bars=250, test_bars=75, step_bars=75,
                fit_fn=bad_fit, signal_fn=_trivial_signals,
                eval_fn=_trivial_eval, verbose=False,
            )
        assert len(results) == 0

    def test_fit_fn_returning_none_skips_fold(self):
        df = _make_pair_df(400)

        def none_fit(df_train):
            return None

        with pytest.warns(UserWarning, match="fit_fn returned None"):
            results = walk_forward_backtest(
                df, train_bars=250, test_bars=75, step_bars=75,
                fit_fn=none_fit, signal_fn=_trivial_signals,
                eval_fn=_trivial_eval, verbose=False,
            )
        assert len(results) == 0

    def test_empty_df_warns_and_returns_empty(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({"P1": np.ones(10), "P2": np.ones(10)}, index=idx)
        with pytest.warns(UserWarning, match="no valid folds"):
            results = walk_forward_backtest(
                df, train_bars=8, test_bars=10,
                fit_fn=_trivial_fit, signal_fn=_trivial_signals,
                eval_fn=_trivial_eval, verbose=False,
            )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

    def test_index_is_fold_number(self):
        df = _make_pair_df(500)
        results = walk_forward_backtest(
            df, train_bars=252, test_bars=63, step_bars=63,
            fit_fn=_trivial_fit, signal_fn=_trivial_signals,
            eval_fn=_trivial_eval, verbose=False,
        )
        assert list(results.index) == list(range(1, len(results) + 1))
