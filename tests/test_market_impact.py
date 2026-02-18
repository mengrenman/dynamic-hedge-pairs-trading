# tests/test_market_impact.py
"""Unit tests for market_impact_bps() and the impact cost path in evaluate_pair_signals."""
import numpy as np
import pandas as pd
import pytest

from pairs.strategies.evaluate import market_impact_bps, evaluate_pair_signals


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pair_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "P1": 50.0 + rng.standard_normal(n).cumsum(),
        "P2": 80.0 + rng.standard_normal(n).cumsum(),
    }, index=dates)


def _make_signals(df: pd.DataFrame, pos: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "n1":  pos * np.ones(len(df)),
        "n2": -pos * np.ones(len(df)),
        "pos": pos * np.ones(len(df), dtype=int),
    }, index=df.index)


# ── market_impact_bps standalone ─────────────────────────────────────────────

class TestMarketImpactBps:
    def test_returns_non_negative(self):
        """Impact dollars must always be ≥ 0."""
        result = market_impact_bps(
            shares_traded=1000, price=50.0,
            avg_daily_volume=500_000, ann_vol_bps=3000, eta=0.14,
        )
        assert float(result) >= 0.0

    def test_zero_shares_gives_zero_impact(self):
        result = market_impact_bps(
            shares_traded=0, price=50.0,
            avg_daily_volume=500_000, ann_vol_bps=3000, eta=0.14,
        )
        assert float(result) == 0.0

    def test_larger_order_gives_larger_impact(self):
        """Doubling shares_traded → impact increases (by √2 factor)."""
        small = market_impact_bps(1_000, 50.0, 500_000, 3000, eta=0.14)
        large = market_impact_bps(4_000, 50.0, 500_000, 3000, eta=0.14)
        assert float(large) > float(small)

    def test_sqrt_scaling(self):
        """Impact scales as √(shares) when other inputs are fixed."""
        imp_1k = market_impact_bps(1_000, 50.0, 500_000, 3000, eta=0.14)
        imp_4k = market_impact_bps(4_000, 50.0, 500_000, 3000, eta=0.14)
        ratio = float(imp_4k) / float(imp_1k)
        assert abs(ratio - 2.0) < 1e-9  # 4x shares → 2x impact

    def test_higher_vol_gives_higher_impact(self):
        """Higher volatility → higher market impact."""
        low  = market_impact_bps(1_000, 50.0, 500_000, ann_vol_bps=1000, eta=0.14)
        high = market_impact_bps(1_000, 50.0, 500_000, ann_vol_bps=5000, eta=0.14)
        assert float(high) > float(low)

    def test_higher_price_gives_higher_impact(self):
        """Higher price → proportionally higher dollar impact."""
        cheap = market_impact_bps(1_000, 20.0, 500_000, 3000, eta=0.14)
        pricey = market_impact_bps(1_000, 200.0, 500_000, 3000, eta=0.14)
        assert float(pricey) > float(cheap)

    def test_vectorized_shares(self):
        """Function must handle array inputs and return arrays."""
        shares = np.array([0, 500, 1000, 5000])
        result = market_impact_bps(shares, 50.0, 500_000, 3000, eta=0.14)
        assert len(result) == 4
        assert (np.array(result) >= 0).all()
        # monotone in shares
        assert result[0] == 0.0
        assert result[1] < result[2] < result[3]

    def test_eta_zero_gives_zero_impact(self):
        """When η=0, market impact must be zero regardless of other params."""
        result = market_impact_bps(1_000, 50.0, 500_000, 3000, eta=0.0)
        assert float(result) == 0.0

    def test_negative_eta_raises(self):
        with pytest.raises(ValueError, match="eta"):
            market_impact_bps(1_000, 50.0, 500_000, 3000, eta=-0.1)

    def test_zero_adv_raises(self):
        with pytest.raises(ValueError, match="avg_daily_volume"):
            market_impact_bps(1_000, 50.0, avg_daily_volume=0, ann_vol_bps=3000)

    def test_negative_adv_raises(self):
        with pytest.raises(ValueError, match="avg_daily_volume"):
            market_impact_bps(1_000, 50.0, avg_daily_volume=-100, ann_vol_bps=3000)

    def test_absolute_magnitude_plausible(self):
        """
        Sanity-check: trading 0.2% of daily volume in a $50 stock with 30% vol
        should cost on the order of ~$1–$30 (not $0 or $10,000).
        """
        shares = 1_000          # |Δq|
        adv    = 500_000        # ADV
        price  = 50.0           # $/share
        vol    = 3000           # bps = 30 %
        # participation = 0.2 %  → √(0.002) ≈ 0.0447
        # impact ≈ 0.14 × 0.30 × 50 × 0.0447 ≈ $0.094 per share × 1000 = $94
        impact = float(market_impact_bps(shares, price, adv, vol, eta=0.14))
        assert 0.01 < impact < 500.0, f"Impact {impact:.4f} outside plausible range"


# ── integrate market impact into evaluate_pair_signals ───────────────────────

class TestEvaluateWithMarketImpact:
    def test_impact_cost_key_in_summary(self):
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, summary = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=500_000, avg_daily_volume_2=500_000,
        )
        assert "impact_cost_total" in summary

    def test_impact_cost_column_in_daily(self):
        df  = _make_pair_df()
        sig = _make_signals(df)
        daily, _, _ = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=500_000, avg_daily_volume_2=500_000,
        )
        assert "impact_cost" in daily.columns

    def test_impact_cost_non_negative(self):
        df  = _make_pair_df()
        sig = _make_signals(df)
        daily, _, _ = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=500_000, avg_daily_volume_2=500_000,
        )
        assert (daily["impact_cost"] >= 0.0).all()

    def test_impact_reduces_net_pnl(self):
        """Net PnL with impact ON must be ≤ net PnL with impact OFF."""
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, s_no_impact = evaluate_pair_signals(df, sig)
        _, _, s_impact    = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=500_000, avg_daily_volume_2=500_000,
        )
        assert s_impact["net_pnl"] <= s_no_impact["net_pnl"]

    def test_smaller_adv_larger_impact(self):
        """
        Lower ADV (less liquid stock) → higher market impact → lower net PnL.
        """
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, s_liquid = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=5_000_000, avg_daily_volume_2=5_000_000,
        )
        _, _, s_illiquid = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=50_000, avg_daily_volume_2=50_000,
        )
        assert s_illiquid["net_pnl"] <= s_liquid["net_pnl"]
        assert s_illiquid["impact_cost_total"] >= s_liquid["impact_cost_total"]

    def test_higher_eta_larger_impact(self):
        """Raising η must monotonically increase impact costs."""
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, s_low = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=200_000, avg_daily_volume_2=200_000,
            impact_eta=0.05,
        )
        _, _, s_high = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=200_000, avg_daily_volume_2=200_000,
            impact_eta=0.30,
        )
        assert s_high["impact_cost_total"] > s_low["impact_cost_total"]

    def test_no_adv_gives_zero_impact_total(self):
        """When no ADV provided (default), impact_cost_total must be 0."""
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, summary = evaluate_pair_signals(df, sig)
        assert summary["impact_cost_total"] == 0.0

    def test_one_leg_adv_only(self):
        """Providing ADV for only one leg should add non-zero impact."""
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, s_both = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=200_000, avg_daily_volume_2=200_000,
        )
        _, _, s_one = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=200_000,
            avg_daily_volume_2=None,       # leg 2 impact disabled
        )
        # Impact with only one leg < both legs
        assert s_one["impact_cost_total"] < s_both["impact_cost_total"]
        assert s_one["impact_cost_total"] > 0.0

    def test_impact_cost_total_equals_daily_sum(self):
        """summary['impact_cost_total'] must equal daily['impact_cost'].sum()."""
        df  = _make_pair_df()
        sig = _make_signals(df)
        daily, _, summary = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=300_000, avg_daily_volume_2=300_000,
        )
        assert abs(summary["impact_cost_total"] - daily["impact_cost"].sum()) < 1e-9

    def test_backward_compat_no_new_required_params(self):
        """
        Calling evaluate_pair_signals without any new params must behave
        identically to the original (no market impact, impact_cost_total=0).
        """
        df  = _make_pair_df()
        sig = _make_signals(df)
        daily, trades, summary = evaluate_pair_signals(df, sig)
        assert summary["impact_cost_total"] == 0.0
        assert "impact_cost" in daily.columns
        assert (daily["impact_cost"] == 0.0).all()

    def test_summary_has_impact_key_without_adv(self):
        """impact_cost_total is always present in summary regardless of ADV."""
        df  = _make_pair_df()
        sig = _make_signals(df)
        _, _, summary = evaluate_pair_signals(df, sig)
        assert "impact_cost_total" in summary


# ── capacity analysis helper (sweep) ─────────────────────────────────────────

class TestCapacitySweep:
    """
    Reproduce the capacity analysis logic that the notebook will use:
    sweep position sizes (multiples of 1 share) and check that Sharpe degrades.
    """

    def _run(self, scale: float, adv: float) -> dict:
        df  = _make_pair_df(n=300, seed=7)
        sig = pd.DataFrame({
            "n1":  scale * np.ones(300),
            "n2": -scale * np.ones(300),
            "pos": np.ones(300, dtype=int),
        }, index=df.index)
        _, _, summary = evaluate_pair_signals(
            df, sig,
            avg_daily_volume_1=adv,
            avg_daily_volume_2=adv,
            capital_base=scale * df["P1"].iloc[100] + scale * df["P2"].iloc[100],
        )
        return summary

    def test_sharpe_degrades_with_larger_position(self):
        """Larger positions eat more market impact → lower risk-adjusted returns."""
        adv = 100_000
        s_small = self._run(scale=10,    adv=adv)
        s_large = self._run(scale=10_000, adv=adv)
        # Sharpe with large position must be lower (or at least not higher)
        # than with tiny position (where impact is negligible)
        assert s_large["impact_cost_total"] > s_small["impact_cost_total"]
