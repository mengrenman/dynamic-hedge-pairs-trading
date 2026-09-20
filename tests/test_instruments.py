"""Tests for the leveraged/inverse instrument detector (open issue #6)."""
import numpy as np
import pandas as pd
import pytest

from pairs.market_data.instruments import (Thresholds, detect_scaled_instruments,
                                           scaled_instrument_report)


@pytest.fixture
def synthetic():
    """An index, two ordinary stocks, a 3x fund, a -2x fund and a same-index twin."""
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    mkt = rng.normal(0, 0.01, n)
    sector = rng.normal(0, 0.012, n)
    # Funds carry tracking error, which is what makes each one closest to its *underlying*
    # rather than to a sibling fund: a fund-to-fund correlation compounds two tracking errors.
    # Exact multiples would tie every correlation at 1 and make argmax arbitrary.
    frames = {
        "IDX":  mkt,
        "TWIN": mkt + rng.normal(0, 0.0004, n),                 # another fund on the same index
        "BULL3": 3.0 * mkt + rng.normal(0, 0.0006, n),          # 3x leveraged
        "BEAR2": -2.0 * mkt + rng.normal(0, 0.0006, n),         # -2x inverse
        "SECT": sector,
        "AAA":  0.9 * mkt + rng.normal(0, 0.013, n),            # ordinary stock
        "BBB":  0.4 * sector + rng.normal(0, 0.015, n),
    }
    return pd.DataFrame({k: 100 * np.exp(np.cumsum(v)) for k, v in frames.items()}, index=idx)


def test_finds_the_leveraged_and_inverse_funds(synthetic):
    rep = scaled_instrument_report(synthetic)
    assert rep.loc["BULL3", "kind"] == "scaled"
    assert rep.loc["BEAR2", "kind"] == "scaled"
    # `beta` is against the closest relative (BEAR2's is BULL3); `scaled_beta` is against the
    # thing it magnifies, which is what the verdict turns on.
    assert abs(rep.loc["BULL3", "scaled_beta"]) == pytest.approx(3.0, abs=0.1)
    assert abs(rep.loc["BEAR2", "scaled_beta"]) == pytest.approx(2.0, abs=0.1)


def test_ordinary_stocks_are_not_flagged(synthetic):
    rep = scaled_instrument_report(synthetic)
    assert rep.loc["AAA", "kind"] == "ordinary"
    assert rep.loc["BBB", "kind"] == "ordinary"


def test_same_index_twins_are_duplicates_not_scaled(synthetic):
    """A twin is mechanical too, but it is a different question and gated separately."""
    rep = scaled_instrument_report(synthetic)
    assert rep.loc["TWIN", "kind"] == "duplicate"
    assert abs(rep.loc["TWIN", "beta"] - 1.0) < 0.15


def test_default_gate_takes_scaled_only(synthetic):
    assert detect_scaled_instruments(synthetic) == {"BULL3", "BEAR2"}
    both = detect_scaled_instruments(synthetic, include_duplicates=True)
    assert {"BULL3", "BEAR2"} < both and "TWIN" in both
    assert "AAA" not in both


def test_no_benchmark_is_needed(synthetic):
    """The gold-miner funds have R^2 ~ 0.005 against the S&P, so a single-benchmark test misses
    them. Removing the market entirely must not stop the sector-levered fund being found."""
    rng = np.random.default_rng(1)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    gold = rng.normal(0, 0.015, n)
    df = pd.DataFrame({
        "GDX": 100 * np.exp(np.cumsum(gold)),
        "NUGT3": 100 * np.exp(np.cumsum(3 * gold)),
        "SPY": 100 * np.exp(np.cumsum(rng.normal(0, 0.009, n))),
    }, index=idx)
    assert detect_scaled_instruments(df) == {"NUGT3"}, "the fund, not its underlying"


def test_the_underlying_is_never_gated_only_the_magnified_side(synthetic):
    """IDX is one-third of BULL3, but gating index funds out because a 3x sibling exists would
    remove far more of the universe than the leveraged funds themselves."""
    rep = scaled_instrument_report(synthetic)
    assert rep.loc["IDX", "kind"] != "scaled"
    assert rep.loc["BULL3", "scaled_vs"] == "IDX"
    assert abs(rep.loc["BULL3", "scaled_beta"]) == pytest.approx(3.0, abs=0.1)


def test_thresholds_are_honoured(synthetic):
    strict = Thresholds(rho_min=0.999999, beta_tol=0.15)
    assert detect_scaled_instruments(synthetic, thresholds=strict) <= {"BULL3", "BEAR2"}
    loose = Thresholds(rho_min=0.95, beta_tol=5.0)      # nothing is 5x away from unit beta here
    assert detect_scaled_instruments(synthetic, thresholds=loose) == set()


def test_incomplete_series_are_skipped_not_guessed(synthetic):
    df = synthetic.copy()
    df.loc[df.index[:50], "BULL3"] = np.nan
    rep = scaled_instrument_report(df)
    assert "BULL3" not in rep.index
    assert "BEAR2" in rep.index


def test_too_short_a_window_returns_empty_rather_than_noise(synthetic):
    rep = scaled_instrument_report(synthetic.iloc[:40])
    assert rep.empty and list(rep.columns) == ["partner", "rho", "beta", "ann_vol", "kind"]


def test_a_constant_series_cannot_break_it(synthetic):
    df = synthetic.assign(FLAT=100.0)
    rep = scaled_instrument_report(df)
    assert "FLAT" not in rep.index
    assert rep.loc["BULL3", "kind"] == "scaled"
