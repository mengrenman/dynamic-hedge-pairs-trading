# tests/test_spread_accounting.py
"""
Tests for pairs.strategies.spread_accounting.

Covers: the book == tradable + revaluation identity (and that the
returned revaluation matches the independent bracket formula); revaluation
collapsing to zero for a fixed-weight spread; the sign/correlation pattern
of the revaluation term on a simulated cointegrated pair traded with a
causal (one-step-ahead) basic Kalman hedge and a contrarian signal; the fee
helper being zero when the position never changes; and the relation between
`decompose_spread_pnl`'s tradable line and `evaluate_pair_signals`'s gross
P&L for a fixed-weight spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.strategies.spread_accounting import decompose_spread_pnl, turnover_and_fees
from pairs.strategies.evaluate import evaluate_pair_signals


# ---------------------------------------------------------------------------
# 1. Identity: book == tradable + revaluation, and revaluation matches the
#    independently-computed bracket formula.
# ---------------------------------------------------------------------------

def test_book_equals_tradable_plus_revaluation_identity():
    rng = np.random.default_rng(0)
    T = 200

    y = np.cumsum(rng.normal(scale=0.01, size=(T, 2)), axis=0) + np.array([4.6, 3.9])
    w = rng.normal(size=(T, 2))
    c = rng.normal(size=T)
    signal = rng.choice([-1.0, 0.0, 1.0], size=T)

    out = decompose_spread_pnl(y, w, c, signal)
    book, tradable, revaluation = out["book"], out["tradable"], out["revaluation"]

    assert np.allclose(book, tradable + revaluation, atol=1e-12)

    # t=0 has no prior bar: all three diff-based lines are 0 there.
    assert book[0] == 0.0
    assert tradable[0] == 0.0
    assert revaluation[0] == 0.0

    # Independent bracket formula:
    #   revaluation_t = signal[t-1] * ((w[t]-w[t-1]) . y[t] - (c[t]-c[t-1]))
    bracket = np.zeros(T)
    for t in range(1, T):
        bracket[t] = signal[t - 1] * (
            (w[t] - w[t - 1]) @ y[t] - (c[t] - c[t - 1])
        )
    assert np.allclose(revaluation, bracket, atol=1e-10)


def test_input_shape_validation():
    y = np.zeros((10, 2))
    w = np.zeros((10, 2))
    c = np.zeros(10)
    signal = np.zeros(10)
    with pytest.raises(ValueError):
        decompose_spread_pnl(y, w[:-1], c, signal)
    with pytest.raises(ValueError):
        decompose_spread_pnl(y, w, c[:-1], signal)
    with pytest.raises(ValueError):
        decompose_spread_pnl(y, w, c, signal[:-1])
    with pytest.raises(ValueError):
        decompose_spread_pnl(y[:, :1], w, c, signal)


# ---------------------------------------------------------------------------
# 2. Revaluation is identically zero when w and c are constant.
# ---------------------------------------------------------------------------

def test_revaluation_zero_when_weights_and_centering_constant():
    rng = np.random.default_rng(1)
    T = 150

    y = np.cumsum(rng.normal(scale=0.01, size=(T, 2)), axis=0) + np.array([4.0, 3.0])
    w = np.tile(np.array([1.0, -0.7]), (T, 1))
    c = np.full(T, 0.3)
    signal = rng.choice([-1.0, 0.0, 1.0], size=T)

    out = decompose_spread_pnl(y, w, c, signal)

    assert np.allclose(out["revaluation"], 0.0, atol=1e-10)
    assert np.allclose(out["book"], out["tradable"], atol=1e-10)


# ---------------------------------------------------------------------------
# 3. Simulated cointegrated pair, basic (causal) Kalman hedge, contrarian
#    signal: revaluation is positive in expectation and its correlation with
#    the signed lagged spread is > 0.9.
# ---------------------------------------------------------------------------

def _basic_kalman_filter(y1, y2, var_eps, var_y2, alpha, x0, V0):
    """
    Minimal causal (one-step-ahead) local-level Kalman filter for the book's
    "basic Kalman" spec: state x=(mu, gamma), Z_t=[1, y2_t], transition B=I
    (random walk, U=0), Q=diag(alpha*var_eps, alpha*var_eps/var_y2), R=var_eps.

    Returns the *predicted* (one-step-ahead) state at each bar, i.e. x[t|t-1]
    -- the state used to form that bar's spread, built only from information
    through bar t-1. This mirrors the book's MARSS xtt1 (tinitx=0) causal
    convention; it is intentionally NOT the smoothed state.
    """
    T = len(y1)
    Q = np.diag([alpha * var_eps, alpha * var_eps / var_y2])
    R = var_eps
    x = np.asarray(x0, dtype=float).copy()
    V = np.asarray(V0, dtype=float).copy()
    mu = np.zeros(T)
    gamma = np.zeros(T)
    for t in range(T):
        x_pred = x
        V_pred = V if t == 0 else V + Q
        mu[t], gamma[t] = x_pred
        Z = np.array([1.0, y2[t]])
        e = y1[t] - Z @ x_pred
        S = Z @ V_pred @ Z + R
        K = (V_pred @ Z) / S
        x = x_pred + K * e
        V = V_pred - np.outer(K, Z) @ V_pred
    return mu, gamma


def test_revaluation_positive_and_correlated_with_signed_lagged_spread():
    rng = np.random.default_rng(1)
    T = 1000

    # A cointegrated pair: y2 a random walk, y1 = mu_true + gamma_true*y2 + stationary noise.
    y2 = np.cumsum(rng.normal(scale=0.01, size=T)) + 4.0
    mu_true, gamma_true = 0.5, 0.8
    eps = rng.normal(scale=0.02, size=T)
    y1 = mu_true + gamma_true * y2 + eps

    var_eps = float(np.var(eps))
    var_y2 = float(np.var(y2))

    mu_t, gamma_t = _basic_kalman_filter(
        y1, y2, var_eps=var_eps, var_y2=var_y2, alpha=1e-5,
        x0=[mu_true, gamma_true],
        V0=np.diag([var_eps / T, var_eps / var_y2 / T]),
    )

    w = np.column_stack([np.ones(T) / (1 + gamma_t), -gamma_t / (1 + gamma_t)])
    c = mu_t / (1 + gamma_t)
    y = np.column_stack([y1, y2])

    spread_pre = (w * y).sum(axis=1) - c  # same as decompose_spread_pnl's 'spread'
    signal = -np.sign(spread_pre)          # contrarian: short when spread rich, long when cheap

    out = decompose_spread_pnl(y, w, c, signal)
    spread, revaluation = out["spread"], out["revaluation"]
    assert np.allclose(spread, spread_pre)

    signal_lag = np.zeros(T)
    signal_lag[1:] = signal[:-1]
    spread_lag = np.zeros(T)
    spread_lag[1:] = spread[:-1]
    # "signed lagged spread": the lagged spread signed by the position that was
    # betting against it -- positive when the contrarian trade is set up to
    # profit from mean reversion of that spread level.
    signed_lagged_spread = -signal_lag * spread_lag

    r = revaluation[1:]
    assert r.mean() > 0
    corr = np.corrcoef(revaluation[1:], signed_lagged_spread[1:])[0, 1]
    assert corr > 0.9


# ---------------------------------------------------------------------------
# 4. Fees are zero when the position never changes.
# ---------------------------------------------------------------------------

def test_fees_zero_when_position_constant():
    T = 60
    signal = np.ones(T)
    w = np.tile(np.array([1.0, -0.5]), (T, 1))
    cost_bps = np.full((T, 2), 5.0)

    dollars_traded, fees = turnover_and_fees(signal, w, cost_bps)

    assert np.allclose(dollars_traded, 0.0)
    assert np.allclose(fees, 0.0)


def test_fees_nonzero_when_position_changes():
    T = 5
    signal = np.array([0.0, 1.0, 1.0, -1.0, -1.0])
    w = np.tile(np.array([1.0, -0.5]), (T, 1))
    cost_bps = np.full((T, 2), 10.0)

    dollars_traded, fees = turnover_and_fees(signal, w, cost_bps)

    position = signal[:, None] * w
    expected_dollars = np.zeros((T, 2))
    expected_dollars[1:] = np.abs(position[1:] - position[:-1])
    expected_fees = (expected_dollars * (cost_bps / 1e4)).sum(axis=1)

    assert dollars_traded[0].sum() == 0.0
    assert np.allclose(dollars_traded, expected_dollars)
    assert np.allclose(fees, expected_fees)
    assert fees[2] == 0.0   # position unchanged bar 1->2 (holding +1)
    assert fees[1] > 0.0    # entering at bar 1
    assert fees[3] > 0.0    # reversing at bar 3


# ---------------------------------------------------------------------------
# 5. Pin: decompose_spread_pnl's tradable line vs evaluate_pair_signals's
#    gross P&L, for a fixed-weight spread.
#
# evaluate_pair_signals books pnl_gross_t = n1_t * dP1_t + n2_t * dP2_t on
# *executed* (already next-bar-timed) share holdings n1, n2, using raw price
# differences (not log returns, no dollar/vol scaling of the diff itself).
# decompose_spread_pnl's tradable_t = signal[t-1] * w[t-1] . (y_t - y_{t-1}).
# These coincide exactly when: (a) y is the same raw price basis as P1/P2
# (not log prices -- evaluate_pair_signals differences raw prices), and
# (b) the executed holdings equal signal[t-1] * w[t-1], i.e. n_i,t is the
# previous bar's decision, sized directly in the weight's own units (no
# per-dollar rescaling). Under those conditions the two lines are identical
# bar-for-bar, not just in aggregate.
# ---------------------------------------------------------------------------

def test_tradable_line_matches_evaluate_pair_signals_gross_pnl():
    rng = np.random.default_rng(2)
    T = 250
    idx = pd.bdate_range("2020-01-02", periods=T)

    P1 = 100.0 * np.exp(np.cumsum(rng.normal(scale=0.01, size=T)))
    P2 = 50.0 * np.exp(np.cumsum(rng.normal(scale=0.012, size=T)))
    beta = 0.6  # fixed hedge ratio

    # Decision signal at close of t (undelayed): a few holding blocks.
    signal_dec = np.zeros(T)
    signal_dec[20:80] = 1.0
    signal_dec[80:90] = 0.0
    signal_dec[90:160] = -1.0
    signal_dec[160:170] = 0.0
    signal_dec[170:230] = 1.0

    w1, w2 = 1.0, -beta
    n1 = pd.Series(signal_dec, index=idx).shift(1, fill_value=0.0) * w1
    n2 = pd.Series(signal_dec, index=idx).shift(1, fill_value=0.0) * w2

    df_pair = pd.DataFrame({"P1": P1, "P2": P2}, index=idx)
    signals = pd.DataFrame({"n1": n1.values, "n2": n2.values}, index=idx)

    daily, _trades, _summary = evaluate_pair_signals(
        df_pair, signals,
        cost_bps=0.0, fee_per_share_1=0.0, fee_per_share_2=0.0,
        borrow_bps_per_year=0.0,
    )

    assert len(daily) == T  # no rows dropped (all prices finite & positive)

    y = np.column_stack([P1, P2])
    w = np.tile(np.array([w1, w2]), (T, 1))
    c = np.zeros(T)
    out = decompose_spread_pnl(y, w, c, signal_dec)

    assert np.allclose(daily["pnl_gross"].to_numpy(), out["tradable"], atol=1e-9)
