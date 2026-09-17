# tests/test_accounting_invariants.py
"""
Conservation laws for the backtest plumbing.

The rest of the suite checks shapes, signs and monotonicity: that a function
runs, returns the documented keys, and moves the right way (more cost → less
P&L, more shares → more impact).  Every defect that preserves those properties
survives it, and three did.

The tests here assert *equalities that must hold by construction* instead:

  * the per-trade log must sum to the daily ledger,
  * filtering a series in two pieces must equal filtering it whole,
  * an order's impact cost must be the per-share concession times the shares,
  * a signals frame that does not line up with the prices must be refused.

Two of these are marked ``xfail(strict=True)`` because they describe known,
documented defects that have not been fixed yet.  Strict means the suite fails
if they ever start passing — so whoever fixes the defect is forced to come here
and remove the marker rather than leaving a stale exemption behind.
"""
import numpy as np
import pandas as pd
import pytest

from pairs.models.kalman import filter_kf_on_new
from pairs.strategies.evaluate import evaluate_pair_signals, market_impact_bps


# ── helpers ───────────────────────────────────────────────────────────────────

def _flat_tape(positions, *, p1=100.0, p2=50.0):
    """
    A tape with constant prices, so gross P&L is identically zero and every
    dollar of net P&L is a cost.  That makes cost attribution errors visible
    as an exact arithmetic discrepancy rather than a plausible-looking number.

    `positions` is a sequence of -1 / 0 / +1 spread positions.
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({"P1": [p1] * n, "P2": [p2] * n}, index=idx)
    # dollar-neutral: 100 shares of P1 ($10k) against 200 shares of P2 ($10k)
    sig = pd.DataFrame(
        {"n1": pos * 100.0, "n2": -pos * 200.0, "z": 0.0,
         "entry": False, "exit": False, "stop": False},
        index=idx,
    )
    return df, sig


def _synthetic_pair(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 100.0
    y = 1.5 * x + 2.0 + rng.normal(0, 0.5, n)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


# ── the trade log must reconcile with the daily ledger ───────────────────────

@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT: the per-trade loop drops the exit bar (`sl.iloc[:-1]`), "
           "so a closed trade is never charged its closing print. Trade-log cost "
           "and pnl_net are systematically too favourable. Affects hit_rate, "
           "avg_win, avg_loss and profit_factor; the daily ledger is correct.",
)
def test_trade_log_reconciles_with_daily_ledger():
    """Closed trades must account for every dollar the ledger charged."""
    df, sig = _flat_tape([0, 1, 1, 1, 0, 0])
    daily, trades, _ = evaluate_pair_signals(df, sig, cost_bps=10.0, capital_base=10_000.0)

    assert len(trades) == 1, "one entry and one exit should make exactly one trade"
    assert float(daily["pnl_gross"].sum()) == pytest.approx(0.0), "flat prices ⇒ no gross P&L"

    # entry costs $20 and the exit costs $20; both belong to this trade
    assert float(trades["pnl_net"].sum()) == pytest.approx(float(daily["pnl_net"].sum()))
    assert float(trades["cost"].sum()) == pytest.approx(float(daily["cost"].sum()))


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT: a reversal bar is added to both `entries` and `exits`, so "
           "`exits[exits >= start]` matches the entry bar itself. That emits a "
           "zero-duration trade, double-counts the reversal bar's cost and gross "
           "P&L, and loses the real trade that follows. Unreachable via "
           "generate_pair_signals (it cannot flip side in one bar) but live for any "
           "other signal source.",
)
def test_reversal_does_not_double_count_or_drop_trades():
    """A long → short reversal is two trades, and costs are charged once each."""
    df, sig = _flat_tape([0, 1, 1, -1, -1, 0])
    daily, trades, _ = evaluate_pair_signals(df, sig, cost_bps=10.0, capital_base=10_000.0)

    assert (trades["entry"] != trades["exit"]).all(), "no zero-duration trades"
    assert len(trades) == 2, "one long trade and one short trade"
    assert float(trades["cost"].sum()) == pytest.approx(float(daily["cost"].sum()))
    assert float(trades["pnl_net"].sum()) == pytest.approx(float(daily["pnl_net"].sum()))


def test_open_position_at_end_is_not_silently_dropped():
    """
    A position still open on the last bar has no exit. Whatever the convention,
    the ledger must still carry its entry cost — this guards the ledger half of
    the accounting, which is what every Sharpe in the repo is built on.
    """
    df, sig = _flat_tape([0, 1, 1, 1, 1])
    daily, _, summary = evaluate_pair_signals(df, sig, cost_bps=10.0, capital_base=10_000.0)
    assert float(daily["cost"].sum()) == pytest.approx(20.0), "entry print must be charged"
    assert float(summary["net_pnl"]) == pytest.approx(-20.0)


# ── splitting a filter must not change it ────────────────────────────────────

@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT: filter_kf_on_new passes the previous window's POSTERIOR as "
           "pykalman's initial_state_mean/covariance, which pykalman uses verbatim as "
           "the PRIOR for bar 0. The boundary predict (m←Fm, P←FPFᵀ+Q) is skipped, so "
           "the prior is over-confident by Q and the first bars of every continuation "
           "differ from an uninterrupted run. Fix: advance last_state by one predict "
           "step inside filter_kf_on_new, guarded on last_state is not None.",
)
def test_kalman_continuation_equals_uninterrupted_filter():
    """Filtering [0:n] must equal filtering [0:k] then continuing on [k:n]."""
    P1, P2 = _synthetic_pair()
    frozen = {"F": np.eye(2), "Q": np.diag([1e-5, 1e-5]), "R": np.array([[0.25]])}
    start = {"mean": np.array([1.0, 0.0]), "cov": np.eye(2)}
    k = len(P1) // 2

    whole, _ = filter_kf_on_new(P1, P2, frozen=frozen, last_state=start)
    first, mid = filter_kf_on_new(P1[:k], P2[:k], frozen=frozen, last_state=start)
    second, _ = filter_kf_on_new(P1[k:], P2[k:], frozen=frozen, last_state=mid)

    np.testing.assert_allclose(
        second["beta"].to_numpy(), whole["beta"].iloc[k:].to_numpy(), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        second["resid"].to_numpy(), whole["resid"].iloc[k:].to_numpy(), rtol=0, atol=1e-12
    )


def test_kalman_continuation_is_exact_once_the_predict_is_applied():
    """
    The companion to the xfail above: applying the missing predict step by hand
    makes the continuation bit-for-bit identical. This pins the *mechanism*, so
    if the defect is ever fixed inside filter_kf_on_new this test still holds and
    documents why.
    """
    P1, P2 = _synthetic_pair()
    F, Q = np.eye(2), np.diag([1e-5, 1e-5])
    frozen = {"F": F, "Q": Q, "R": np.array([[0.25]])}
    start = {"mean": np.array([1.0, 0.0]), "cov": np.eye(2)}
    k = len(P1) // 2

    whole, _ = filter_kf_on_new(P1, P2, frozen=frozen, last_state=start)
    _, mid = filter_kf_on_new(P1[:k], P2[:k], frozen=frozen, last_state=start)
    advanced = {"mean": F @ mid["mean"], "cov": F @ mid["cov"] @ F.T + Q}
    second, _ = filter_kf_on_new(P1[k:], P2[k:], frozen=frozen, last_state=advanced)

    np.testing.assert_allclose(
        second["beta"].to_numpy(), whole["beta"].iloc[k:].to_numpy(), rtol=0, atol=1e-12
    )


# ── market impact is a whole-order cost ──────────────────────────────────────

def test_impact_is_concession_times_shares():
    """The returned dollars must be the per-share concession paid on every share."""
    shares, price, adv, vol, eta = 1_000, 50.0, 500_000, 3000, 0.14
    per_share = eta * (vol / 1e4) * price * np.sqrt(shares / adv)
    got = float(market_impact_bps(shares, price, adv, vol, eta=eta))
    assert got == pytest.approx(per_share * shares)
    assert got == pytest.approx(93.91, abs=0.01)


def test_impact_cost_grows_superlinearly_with_size():
    """
    Total impact must scale as |Δq|^{3/2}. If it scales as √|Δq| the model has
    stopped penalising size — each dollar traded gets *cheaper* as the order
    grows, which inverts the only effect the model exists to capture.
    """
    base = float(market_impact_bps(1_000, 50.0, 500_000, 3000, eta=0.14))
    for mult in (2, 4, 10, 100):
        got = float(market_impact_bps(1_000 * mult, 50.0, 500_000, 3000, eta=0.14))
        assert got / base == pytest.approx(mult ** 1.5, rel=1e-9)

    # and in bps of notional, cost per dollar traded must RISE with size
    bps = lambda q: float(market_impact_bps(q, 50.0, 500_000, 3000, eta=0.14)) / (q * 50.0)
    assert bps(100) < bps(1_000) < bps(10_000)


def test_impact_flows_through_to_the_ledger_with_the_shares_factor():
    """
    The end-to-end path must carry the same |Δq| factor the helper does.
    Needs a tape with non-zero volatility: σ is estimated from rolling
    log-returns, so a flat tape switches the impact model off entirely.
    """
    rng = np.random.default_rng(3)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"P1": 100.0 + rng.normal(0, 1, n).cumsum(),
         "P2": 50.0 + rng.normal(0, 0.5, n).cumsum()},
        index=idx,
    )
    pos = np.zeros(n); pos[20:40] = 1.0          # one round trip, 100/200 shares
    sig = pd.DataFrame(
        {"n1": pos * 100.0, "n2": -pos * 200.0, "z": 0.0,
         "entry": False, "exit": False, "stop": False},
        index=idx,
    )
    _, _, summary = evaluate_pair_signals(
        df, sig, cost_bps=0.0, capital_base=10_000.0,
        avg_daily_volume_1=500_000, avg_daily_volume_2=500_000,
    )
    total = float(summary["impact_cost_total"])
    assert total > 1.0, (
        f"impact_cost_total={total:.4f}; a sub-dollar total on 600 shares traded "
        "means the per-share concession was booked as the whole order's cost"
    )

    # halving ADV must raise impact by exactly √2 (participation doubles)
    _, _, thin = evaluate_pair_signals(
        df, sig, cost_bps=0.0, capital_base=10_000.0,
        avg_daily_volume_1=250_000, avg_daily_volume_2=250_000,
    )
    assert float(thin["impact_cost_total"]) / total == pytest.approx(np.sqrt(2.0), rel=1e-9)


# ── misaligned inputs must be refused, not silently truncated ────────────────

def test_misaligned_signals_raises():
    """
    pd.concat(axis=1) unions the indexes and the NaN-price filter then drops
    whatever signals contributed, so a misaligned frame would evaluate a
    shorter, mostly-flat book and report a clean-looking Sharpe.
    """
    df, sig = _flat_tape([0, 1, 1, 0])
    shifted = sig.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=3)   # e.g. a timezone slip
    with pytest.raises(ValueError, match="not aligned"):
        evaluate_pair_signals(df, shifted, cost_bps=1.0, capital_base=10_000.0)


def test_signals_covering_a_subset_of_prices_is_allowed():
    """Legitimate case: signals for a sub-window of the price history."""
    df, sig = _flat_tape([0, 1, 1, 1, 0, 0])
    daily, _, _ = evaluate_pair_signals(
        df, sig.iloc[1:], cost_bps=10.0, capital_base=10_000.0
    )
    assert len(daily) > 0


def test_duplicate_timestamps_raise():
    """Duplicate index entries make pd.concat(axis=1) explode combinatorially."""
    df, sig = _flat_tape([0, 1, 1, 0])
    dup = pd.concat([sig, sig.iloc[[1]]]).sort_index()
    with pytest.raises(ValueError, match="duplicated"):
        evaluate_pair_signals(df, dup, cost_bps=1.0, capital_base=10_000.0)
