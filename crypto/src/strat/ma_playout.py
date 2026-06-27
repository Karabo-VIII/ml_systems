"""MA STRATEGY PLAY-OUT — the engine in action over OOS-2024, with PROD selection.

Closes the fidelity gap (DEPLOYMENT_NOTES.md): instead of naive EW, this plays the
strategy with the ACTUAL production selection/sizing:
  - holds from the audited V8 slice engine (rolling, prod cost, tier config)
  - per-bar allocation via firm/portfolio.allocate(): inverse-vol risk parity,
    vol-targeted (2%/period), 15% per-name cap, gross<=1.0, long-only
  - MtM replay: lagged weights x next-bar returns - turnover cost
  - V1 BTC-SMA100 regime gate (cash in bear)

Outputs:
  1. A QuantStats HTML tearsheet (same style as crypto/report.html) — equity curve,
     drawdown, monthly returns, full stats vs buy&hold benchmark.
  2. A "strategy in action" event log: per-rebalance snapshot of how many names held,
     which got the biggest weight, turnover, regime state — so you SEE it progress.

Single candidate at a time (default SMA|4h, the sweet spot). 2025 SEALED.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import (
    held_rolling, _cached_band_held, btc_regime, OOS_WALL_MS, OOS_START, OOS_2024_END,
)
from strat.canonical_exits import COST_PROD
from firm.portfolio import allocate, Position, PortfolioLimits

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "reports" / "MA_playout"

TIER_CFG = {
    "1d": {"lookback": 14, "slice": 14, "min_hold": 0, "cooldown": 0},
    "4h": {"lookback": 14, "slice": 7,  "min_hold": 0, "cooldown": 0},
    "2h": {"lookback": 7,  "slice": 7,  "min_hold": 3, "cooldown": 3},
}


def playout(ma_type="SMA", tf="4h", vol_target=0.02, max_per_name=0.15):
    cfg = TIER_CFG[tf]
    bpd = BARS_PER_DAY[tf]
    lookback_bars = cfg["lookback"] * bpd
    slice_bars = cfg["slice"] * bpd
    wall_ms = OOS_WALL_MS
    win_s = pd.Timestamp(OOS_START); win_e = pd.Timestamp(OOS_2024_END)
    regime = btc_regime(wall_ms)

    # Build holds + close panel
    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        bh = _cached_band_held(ma_type, tf, sym, o, h, l, c, wall_ms)
        held = held_rolling(c, bh, lookback_bars, slice_bars,
                            min_hold_bars=cfg["min_hold"], cooldown_bars=cfg["cooldown"])
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)

    close_panel = pd.DataFrame(close_cols).sort_index()
    held_panel = pd.DataFrame(held_cols).reindex(close_panel.index).fillna(0).astype(np.int8)
    ret_panel = close_panel.pct_change()
    # trailing vol per asset (for inverse-vol sizing) — causal rolling std
    vol_panel = ret_panel.rolling(max(20, bpd * 5), min_periods=10).std()

    assets = list(close_panel.columns)
    dates = close_panel.index

    # regime per date
    reg_map = {}
    if regime is not None:
        ridx = regime.index
        for dt in dates:
            d1 = pd.Timestamp(dt).normalize()
            prior = ridx[ridx <= d1]
            reg_map[dt] = int(regime[prior[-1]]) if len(prior) else 0

    plims = PortfolioLimits(vol_target=vol_target, max_gross=1.0,
                            max_per_name=max_per_name, long_only=True)

    # PER-BAR ALLOCATION (the real selection criteria)
    W = pd.DataFrame(0.0, index=dates, columns=assets)
    action_log = []
    for dti, date in enumerate(dates):
        if dti < 31:
            continue
        reg = reg_map.get(date, 1)
        if reg == 0:
            continue  # regime gate: cash in bear
        ha = held_panel.loc[date]; vr = vol_panel.loc[date]
        positions = []
        for a in assets:
            v = vr.get(a)
            if not ha.get(a) or v is None or not np.isfinite(v) or v <= 0:
                continue
            positions.append(Position(asset=a, raw_fraction=1.0 / v, vol=float(v)))
        if positions:
            al = allocate(positions, corr=None, limits=plims)
            for a, wv in al.weights.items():
                W.at[date, a] = wv
            # log a snapshot at each rebalance within the OOS window
            if win_s <= date < win_e:
                wsorted = sorted(al.weights.items(), key=lambda kv: -kv[1])
                action_log.append({
                    "date": str(date)[:16], "regime": "BULL",
                    "n_signals": len(positions), "n_held": sum(1 for _, w in al.weights.items() if w > 1e-6),
                    "gross": round(al.gross, 3), "top1": wsorted[0][0] if wsorted else "",
                    "top1_w": round(wsorted[0][1], 3) if wsorted else 0,
                    "top3": ", ".join(f"{a}:{w:.0%}" for a, w in wsorted[:3]),
                })

    # MtM replay
    Wl = W.shift(1).fillna(0.0)
    gross_ret = (Wl * ret_panel.fillna(0.0)).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net_all = gross_ret - turnover * (COST_PROD / 2)

    # benchmark: equal-weight buy&hold of u100 (always long, regime-gated for fairness shown separately)
    bench = ret_panel.fillna(0.0).mean(axis=1)

    # slice to OOS window
    mask = (dates >= win_s) & (dates < win_e)
    strat_oos = net_all[mask]
    bench_oos = bench[mask]
    oos_dates = dates[mask]

    return {
        "ma_type": ma_type, "tf": tf, "strat": strat_oos, "bench": bench_oos,
        "dates": oos_dates, "action_log": action_log, "bpd": bpd,
        "turnover": turnover[mask], "weights": W[mask],
        "n_assets": len(assets), "vol_target": vol_target, "max_per_name": max_per_name,
    }


def make_tearsheet(res, outpath):
    """Generate a QuantStats HTML tearsheet from the daily strategy returns."""
    import quantstats as qs
    # quantstats wants a daily-indexed return series; resample bar returns to daily
    s = pd.Series(res["strat"].to_numpy(), index=pd.DatetimeIndex(res["dates"]))
    b = pd.Series(res["bench"].to_numpy(), index=pd.DatetimeIndex(res["dates"]))
    # collapse intra-day bars to daily compounded returns
    s_daily = (1 + s).groupby(s.index.normalize()).prod() - 1
    b_daily = (1 + b).groupby(b.index.normalize()).prod() - 1
    s_daily.index = pd.DatetimeIndex(s_daily.index)
    b_daily.index = pd.DatetimeIndex(b_daily.index)
    title = f"MA {res['ma_type']}|{res['tf']} — PROD selection (inv-vol, {res['vol_target']:.0%} vol-tgt, {res['max_per_name']:.0%} cap) — OOS-2024"
    qs.reports.html(s_daily, benchmark=b_daily, output=str(outpath), title=title)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="SMA")
    ap.add_argument("--tf", default="4h")
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTDIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Playing out {a.type}|{a.tf} over OOS-2024 with PROD selection...")
    res = playout(a.type, a.tf)

    # tearsheet
    tear = run_dir / f"tearsheet_{a.type}_{a.tf}.html"
    try:
        make_tearsheet(res, tear)
        print(f"  Tearsheet: {tear}")
    except Exception as e:
        print(f"  Tearsheet FAILED: {e}")

    # action log
    log = res["action_log"]
    log_path = run_dir / f"action_log_{a.type}_{a.tf}.json"
    with open(log_path, "w") as f:
        json.dump({"ma_type": a.type, "tf": a.tf, "n_rebalances": len(log),
                   "action_log": log}, f, indent=2, default=str)

    # summary stats
    strat = res["strat"].to_numpy()
    comp = float(np.prod(1 + strat) - 1) * 100
    bpd = res["bpd"]
    lin_wk = float(np.mean(strat)) * bpd * 7 * 100
    cum = np.cumprod(1 + strat)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100
    sharpe = (float(np.mean(strat)) * bpd * 365) / (float(np.std(strat)) * np.sqrt(bpd * 365)) if np.std(strat) > 1e-12 else 0
    avg_names = np.mean([a["n_held"] for a in log]) if log else 0
    avg_gross = np.mean([a["gross"] for a in log]) if log else 0

    print(f"\n  === {a.type}|{a.tf} PROD-selection play-out (OOS-2024) ===")
    print(f"  Compound: {comp:+.1f}% | Linear/wk: {lin_wk:+.2f}% | Sharpe: {sharpe:.2f} | MaxDD: {maxdd:.1f}%")
    print(f"  Avg names held: {avg_names:.1f} | Avg gross: {avg_gross:.1%} | rebalances logged: {len(log)}")
    print(f"\n  First 8 action snapshots:")
    for e in log[:8]:
        print(f"   {e['date']} {e['regime']} | {e['n_held']} names, gross {e['gross']:.0%} | top: {e['top3']}")

    with open(run_dir / f"summary_{a.type}_{a.tf}.json", "w") as f:
        json.dump({"ma_type": a.type, "tf": a.tf, "compound_pct": round(comp, 1),
                   "lin_wk_pct": round(lin_wk, 3), "sharpe": round(sharpe, 2),
                   "maxdd_pct": round(maxdd, 1), "avg_names_held": round(avg_names, 1),
                   "avg_gross": round(avg_gross, 3), "n_rebalances": len(log),
                   "selection": "inv-vol risk-parity + vol-target + per-name-cap (PROD allocate)",
                   "run_dated": ts}, f, indent=2, default=str)
    print(f"\n  Run folder: {run_dir}")


if __name__ == "__main__":
    main()
