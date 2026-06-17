"""
Probe that MtM double-count bug fix in short_term_speculator_v2.run_v2
produces trade-log vs equity reconciliation within 1-2x (was 7x-47x).

Synthetic: create 3 assets with known price paths. Force one known trade.
Compute sum(trade_log.pnl) vs sum(pnl). Ratio should be 1:1 (clean, MtM-only).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "strategy"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import numpy as np

# Minimal cost model — fixed per-side cost
class _FixedCost:
    def per_side(self, asset, notional_usd=1000.0, median_adv_usd=50e6,
                 asset_ann_vol=0.6, stress=False):
        return 0.001  # 10 bps per side
    def round_trip(self, asset, notional_usd=1000.0, hold_bars=10,
                   bar_hours=24.0, median_adv_usd=50e6, asset_ann_vol=0.6,
                   stress=False):
        return 0.002  # 20 bps round-trip (residual=0 for clean test)


# Minimal engine — fires on bar 5 for ASSETA
class _OneShotEngine:
    class _Cfg:
        name = "probe_engine"
    def __init__(self):
        self.cfg = _OneShotEngine._Cfg()
    def compute_signals(self, asset_data, t):
        if t == 5:
            return {"ASSETA": 1.0}
        return {}


def make_asset_data(n_bars=30):
    """3 assets. ASSETA has a known 5% cumulative return between bars 5-10."""
    names = ["ASSETA", "ASSETB", "BTCUSDT"]
    asset_data = {}
    # ASSETA: flat before bar 5, then 1% per bar for 5 bars, then flat
    prices = np.ones(n_bars)
    prices[5] = 1.0
    for t in range(6, 11):
        prices[t] = prices[t-1] * 1.01  # 1% per bar, 5 bars = 5.1%
    for t in range(11, n_bars):
        prices[t] = prices[10]
    asset_data["ASSETA"] = {
        "close": prices.copy(),
        "high": prices.copy(),
        "low": prices.copy(),
        "returns": np.concatenate([[0.0], np.diff(prices) / prices[:-1]]),
    }
    # ASSETB and BTCUSDT flat
    for a in ["ASSETB", "BTCUSDT"]:
        asset_data[a] = {
            "close": np.ones(n_bars),
            "high": np.ones(n_bars),
            "low": np.ones(n_bars),
            "returns": np.zeros(n_bars),
        }
    # ret_matrix: rows=bars, cols=assets
    ret_matrix = np.stack([asset_data[a]["returns"] for a in names], axis=1)
    return asset_data, ret_matrix, names


def run_probe():
    from short_term_speculator_v2 import run_v2, ShortTermConfig

    asset_data, ret_matrix, names = make_asset_data()
    # Config: hold 5 bars, no stops triggered by 1%/bar rise
    cfg = ShortTermConfig(
        max_hold_bars=5, stop_loss_pct=0.20, trailing_stop_pct=0.20,
        cooldown_bars=2, max_concurrent=3, cash_floor=0.0,
        signal_threshold_and=0.15, signal_threshold_single=0.30,
        per_bucket_scale={"BLUE": 1.0, "STEADY": 1.0, "VOLATILE": 1.0, "DEGEN": 1.0,
                          "UNKNOWN": 1.0},
        bear_btc_30d_thresh=-0.99,  # always allow
    )

    engines = [_OneShotEngine()]
    cost_model = _FixedCost()

    res = run_v2(asset_data, ret_matrix, names,
                   warmup=0, ws=0, we=30,
                   engines=engines, cfg=cfg, cost_model=cost_model)

    trade_log = res["trade_log"]
    pnl_stream = res["pnl"]

    print(f"=== PROBE: MtM double-count fix (speculator_v2) ===")
    print(f"Asset path ASSETA: 1.0 -> {asset_data['ASSETA']['close'][10]:.4f} "
          f"(+{(asset_data['ASSETA']['close'][10]/1.0 - 1)*100:.2f}%)")
    print(f"Trade log ({len(trade_log)} trades):")
    sum_trade_pnl = 0.0
    for tr in trade_log:
        print(f"  asset={tr['asset']} entry_t={tr['entry_t']} exit_t={tr['t_exit']} "
              f"hold={tr['hold_bars']} ret={tr['ret']*100:.3f}% "
              f"cost={tr['cost']*100:.3f}% pnl={tr['pnl']*100:.4f}% "
              f"weight={tr['weight']:.3f} reason={tr['reason']}")
        sum_trade_pnl += float(tr["pnl"])
    print(f"Sum of trade_log.pnl: {sum_trade_pnl*100:.4f}%")

    equity_raw = float(np.prod(1 + pnl_stream)) - 1
    print(f"Sum of pnl stream:    {float(np.sum(pnl_stream))*100:.4f}%")
    print(f"Compounded equity:    {equity_raw*100:.4f}%")

    # Reconciliation ratio
    if abs(sum_trade_pnl) > 1e-10:
        ratio_sum = float(np.sum(pnl_stream)) / sum_trade_pnl
        ratio_eq = equity_raw / sum_trade_pnl
        print(f"Reconciliation sum_pnl/trade_pnl:    {ratio_sum:.3f}x "
              f"(expect ~1.0 after fix; was ~2.0 pre-fix for 5-bar hold)")
        print(f"Reconciliation equity/trade_pnl:     {ratio_eq:.3f}x")

    # Clean test: expected trade_pnl = weight * (0.05 - 0.002)
    expected_weight = (1.0 - 0.0) / 3 * 1.0  # cash_floor=0, max_concurrent=3, scale=1
    realized_ret = asset_data["ASSETA"]["close"][10] / asset_data["ASSETA"]["close"][5] - 1
    expected_trade_pnl = expected_weight * (realized_ret - 0.002)
    print(f"Expected trade pnl (analytical): {expected_trade_pnl*100:.4f}% "
          f"(weight={expected_weight:.3f}, ret={realized_ret*100:.3f}%, "
          f"cost=0.200%)")
    print(f"Match: {abs(sum_trade_pnl - expected_trade_pnl) < 1e-4}")

    # The REAL test — sum_pnl should match trade_pnl within 0.1% absolute
    # (was 2x for 5-bar hold under buggy code)
    delta = abs(float(np.sum(pnl_stream)) - sum_trade_pnl)
    verdict = "PASS" if delta < 1e-3 else "FAIL"
    print(f"\n=== VERDICT: {verdict} (delta = {delta*100:.4f}%) ===")
    return verdict == "PASS"


if __name__ == "__main__":
    ok = run_probe()
    sys.exit(0 if ok else 1)
