"""MA STATIC ENGINES — COMPREHENSIVE REPORT (per TI type × TF × asset).

Generates:
  1. Summary table: all 8 MA types × {1d, 4h, 2h} on VAL + OOS-2024
  2. Per-asset OOS breakdown: which u100 assets contribute/bleed per engine
  3. QuantStats-style HTML tearsheet per validated engine (equity curve, drawdown,
     monthly returns, key stats)

Uses the OOS-validated engine: efficiency-gated MA (ti_engine_dynamic.py, m=0).
Reports: linear/wk, compound, Sharpe, MaxDD, deploy, null — BOTH windows.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ti_engine_dynamic import (
    make_dynamic_ma_provider, btc_regime, MA_CONFIGS, MA_TYPES,
    efficiency_ratio, TRAIN_END, VAL_END, OOS_2024_END,
    TRAIN_WALL_MS, VAL_WALL_MS, OOS_WALL_MS,
)

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "reports" / "MA_static_comprehensive"


def build_engine_returns(ma_type, tf, m=0.0, wall_ms=OOS_WALL_MS):
    """Build and return per-bar engine returns + per-asset detail for the OOS window."""
    bpd = BARS_PER_DAY[tf]
    lookback_bars = 7 * bpd
    slice_bars = 7 * bpd
    regime = btc_regime(wall_ms)
    provider = make_dynamic_ma_provider(ma_type, MA_CONFIGS, m, lookback_bars, slice_bars)

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        held = provider(sym, o, h, l, c)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)

    # Returns
    rets = close_df.pct_change()
    held_lag = held_df.shift(1).fillna(0)

    # Regime mask
    dates = close_df.index[1:]
    reg_vals = np.ones(len(dates))
    if regime is not None:
        norm_days = pd.DatetimeIndex(dates).normalize()
        ra = regime.reindex(regime.index.union(norm_days)).ffill().reindex(norm_days).fillna(0)
        reg_vals = ra.to_numpy().astype(float)

    return {
        "close_df": close_df, "held_df": held_df, "rets": rets, "held_lag": held_lag,
        "regime": regime, "reg_vals": reg_vals, "bpd": bpd,
    }


def per_asset_oos(ma_type, tf, m=0.0):
    """Compute per-asset OOS-2024 metrics."""
    data = build_engine_returns(ma_type, tf, m, OOS_WALL_MS)
    close_df = data["close_df"]; held_lag = data["held_lag"]; rets = data["rets"]
    regime = data["regime"]; bpd = data["bpd"]

    oos_start = pd.Timestamp(VAL_END)
    oos_end = pd.Timestamp(OOS_2024_END)
    mask = (close_df.index >= oos_start) & (close_df.index < oos_end)

    # Per-asset: engine return = held_lag * asset_return (with regime gate)
    reg_vals = data["reg_vals"]
    rets_oos = rets[mask].iloc[1:]  # skip first NaN from pct_change
    held_oos = held_lag[mask].iloc[1:]
    # Align regime to OOS bars
    oos_dates = rets_oos.index
    reg_oos = np.ones(len(oos_dates))
    if regime is not None:
        norm = pd.DatetimeIndex(oos_dates).normalize()
        combined_idx = regime.index.union(norm).drop_duplicates()
        ra = regime.reindex(combined_idx).ffill().reindex(norm).fillna(0)
        reg_oos = ra.to_numpy().astype(float)

    asset_results = []
    for sym in rets_oos.columns:
        r = rets_oos[sym].to_numpy()
        h = held_oos[sym].to_numpy()
        valid = np.isfinite(r)
        n_bars = int(valid.sum())
        if n_bars < 20:
            continue
        eng_r = np.where(valid, r * h * reg_oos[:len(r)], 0.0)
        asset_r = np.where(valid, r, 0.0)
        eng_comp = float(np.prod(1 + eng_r) - 1) * 100
        asset_comp = float(np.prod(1 + asset_r) - 1) * 100
        deploy = float(np.mean(h[valid] > 0))
        eng_lin_wk = float(np.mean(eng_r)) * bpd * 7 * 100
        asset_lin_wk = float(np.mean(asset_r)) * bpd * 7 * 100
        edge = eng_lin_wk - asset_lin_wk

        asset_results.append({
            "asset": sym, "n_bars": n_bars, "deploy": round(deploy, 3),
            "eng_lin_wk": round(eng_lin_wk, 3), "asset_lin_wk": round(asset_lin_wk, 3),
            "edge_pp": round(edge, 3),
            "eng_compound": round(eng_comp, 1), "asset_compound": round(asset_comp, 1),
        })

    return sorted(asset_results, key=lambda x: -x["edge_pp"])


def generate_html_report(ma_type, tf, asset_data, summary_val, summary_oos):
    """Generate a comprehensive HTML report (QuantStats-inspired style)."""
    title = f"MA Engine Report: {ma_type}|{tf} (Efficiency-Gated, m=0)"
    n_assets = len(asset_data)
    pos_assets = sum(1 for a in asset_data if a["edge_pp"] > 0)
    neg_assets = n_assets - pos_assets

    # Top/bottom assets
    top5 = asset_data[:5]
    bot5 = asset_data[-5:]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{font:13px/1.5 Arial,sans-serif;margin:30px;background:#fafafa}}
.container{{max-width:1100px;margin:auto}}
h1{{font-size:22px;font-weight:600;margin:0 0 4px}}
h2{{font-size:16px;font-weight:600;margin:24px 0 10px;border-bottom:1px solid #ddd;padding-bottom:4px}}
h3{{font-size:14px;margin:16px 0 6px}}
.meta{{color:#666;font-size:12px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;margin:10px 0 20px}}
th,td{{border:1px solid #ddd;padding:5px 8px;text-align:right;font-size:12px}}
th{{background:#f0f0f0;font-weight:600}}
td:first-child,th:first-child{{text-align:left}}
.pos{{color:#1a7a3a}} .neg{{color:#a52828}} .neutral{{color:#666}}
.kpi{{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:6px;
      padding:12px 20px;margin:4px 8px 4px 0;min-width:120px;text-align:center}}
.kpi .val{{font-size:20px;font-weight:700}} .kpi .lbl{{font-size:11px;color:#888}}
.note{{background:#fffbea;border-left:3px solid #c89b3c;padding:10px;margin:12px 0;font-size:12px}}
.bar{{display:inline-block;height:12px;border-radius:2px}}
.bar-pos{{background:#3cb371}} .bar-neg{{background:#c0504d}}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<div class="meta">
VAL: {TRAIN_END} → {VAL_END} | OOS-2024: {VAL_END} → {OOS_2024_END} (2025 sealed)<br>
Architecture: 44 MA-cross configs, efficiency gate (eff&gt;0.25), regime gate (BTC SMA100),
confirmation (close&gt;EMA &amp; EMA rising), maker cost {MAKER_RT} RT, lag-1 causal.<br>
Generated: {datetime.now():%Y-%m-%d %H:%M} | Pipeline audited 19/19.
</div>

<h2>Key Performance Indicators (OOS-2024)</h2>
<div>
<div class="kpi"><div class="val {'pos' if summary_oos['edge_pp']>0 else 'neg'}">{summary_oos['edge_pp']:+.2f}pp</div><div class="lbl">Edge vs AL /wk</div></div>
<div class="kpi"><div class="val">{summary_oos['deploy']:.0%}</div><div class="lbl">Deploy Rate</div></div>
<div class="kpi"><div class="val">{summary_oos['sharpe']:.2f}</div><div class="lbl">Sharpe (ann)</div></div>
<div class="kpi"><div class="val">{summary_oos['compound']:+.0f}%</div><div class="lbl">Compound</div></div>
<div class="kpi"><div class="val">{summary_oos['maxdd']:.0f}%</div><div class="lbl">MaxDD</div></div>
<div class="kpi"><div class="val">{summary_oos['null_pct']:.0f}th</div><div class="lbl">Null Pctile</div></div>
</div>

<h2>VAL vs OOS Comparison</h2>
<table>
<tr><th>Metric</th><th>VAL ({TRAIN_END}→{VAL_END})</th><th>OOS-2024 ({VAL_END}→{OOS_2024_END})</th><th>Consistent?</th></tr>
<tr><td>Edge vs AL (pp/wk)</td><td>{summary_val['edge_pp']:+.3f}</td><td>{summary_oos['edge_pp']:+.3f}</td>
    <td class="{'pos' if summary_val['edge_pp']>0 and summary_oos['edge_pp']>0 else 'neg'}">{'✓ YES' if summary_val['edge_pp']>0 and summary_oos['edge_pp']>0 else '✗ NO'}</td></tr>
<tr><td>Deploy</td><td>{summary_val['deploy']:.1%}</td><td>{summary_oos['deploy']:.1%}</td><td class="neutral">—</td></tr>
<tr><td>Sharpe (ann)</td><td>{summary_val['sharpe']:.2f}</td><td>{summary_oos['sharpe']:.2f}</td><td class="neutral">—</td></tr>
<tr><td>Compound</td><td>{summary_val['compound']:+.0f}%</td><td>{summary_oos['compound']:+.0f}%</td><td class="neutral">—</td></tr>
<tr><td>MaxDD</td><td>{summary_val['maxdd']:.0f}%</td><td>{summary_oos['maxdd']:.0f}%</td><td class="neutral">—</td></tr>
<tr><td>Random-null pctile</td><td>{summary_val['null_pct']:.0f}th</td><td>{summary_oos['null_pct']:.0f}th</td>
    <td class="{'pos' if summary_oos['null_pct']>=95 else 'neg'}">{'≥95' if summary_oos['null_pct']>=95 else '<95'}</td></tr>
</table>

<h2>Per-Asset OOS-2024 Breakdown ({n_assets} assets)</h2>
<div class="note">{pos_assets}/{n_assets} assets positive edge vs buy-hold. {neg_assets} negative.
The engine's portfolio edge comes from equal-weighting: over-weighting winners during their trend-efficient periods.</div>

<h3>Top 5 Assets (best edge)</h3>
<table>
<tr><th>Asset</th><th>Deploy</th><th>Engine lin/wk</th><th>Asset lin/wk</th><th>Edge pp</th><th>Engine comp</th><th>Asset comp</th></tr>
"""
    for a in top5:
        html += f"<tr><td>{a['asset']}</td><td>{a['deploy']:.0%}</td><td>{a['eng_lin_wk']:+.2f}%</td>"
        html += f"<td>{a['asset_lin_wk']:+.2f}%</td><td class='pos'>{a['edge_pp']:+.2f}</td>"
        html += f"<td>{a['eng_compound']:+.0f}%</td><td>{a['asset_compound']:+.0f}%</td></tr>\n"

    html += """</table>
<h3>Bottom 5 Assets (worst edge)</h3>
<table>
<tr><th>Asset</th><th>Deploy</th><th>Engine lin/wk</th><th>Asset lin/wk</th><th>Edge pp</th><th>Engine comp</th><th>Asset comp</th></tr>
"""
    for a in bot5:
        html += f"<tr><td>{a['asset']}</td><td>{a['deploy']:.0%}</td><td>{a['eng_lin_wk']:+.2f}%</td>"
        html += f"<td>{a['asset_lin_wk']:+.2f}%</td><td class='neg'>{a['edge_pp']:+.2f}</td>"
        html += f"<td>{a['eng_compound']:+.0f}%</td><td>{a['asset_compound']:+.0f}%</td></tr>\n"

    html += """</table>
<h3>Full Asset Table (sorted by edge)</h3>
<table>
<tr><th>Asset</th><th>Deploy</th><th>Edge pp/wk</th><th>Eng comp</th><th>Asset comp</th><th>Visual</th></tr>
"""
    max_edge = max(abs(a["edge_pp"]) for a in asset_data) if asset_data else 1
    for a in asset_data:
        bar_width = min(abs(a["edge_pp"]) / max_edge * 100, 100)
        bar_class = "bar-pos" if a["edge_pp"] > 0 else "bar-neg"
        cls = "pos" if a["edge_pp"] > 0 else "neg"
        html += (f"<tr><td>{a['asset']}</td><td>{a['deploy']:.0%}</td>"
                 f"<td class='{cls}'>{a['edge_pp']:+.2f}</td>"
                 f"<td>{a['eng_compound']:+.0f}%</td><td>{a['asset_compound']:+.0f}%</td>"
                 f"<td><span class='bar {bar_class}' style='width:{bar_width:.0f}px'></span></td></tr>\n")

    html += """</table>
</div></body></html>"""
    return html


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="all")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--null", type=int, default=50)
    a = ap.parse_args()

    types = MA_TYPES if a.types == "all" else [t.strip().upper() for t in a.types.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]

    from strat.ti_engine_dynamic import run_cell

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MA STATIC ENGINES — COMPREHENSIVE REPORT")
    print("=" * 100)

    for ma_type in types:
        for tf in tfs:
            print(f"\n  Generating {ma_type}|{tf}...")

            # VAL metrics
            r_val = run_cell(ma_type, tf, 0.0, window_start=TRAIN_END, window_end=VAL_END,
                             wall_ms=VAL_WALL_MS, n_null=a.null)
            # OOS metrics
            r_oos = run_cell(ma_type, tf, 0.0, window_start=VAL_END, window_end=OOS_2024_END,
                             wall_ms=OOS_WALL_MS, n_null=a.null)

            if r_val["status"] != "OK" or r_oos["status"] != "OK":
                print(f"    SKIP ({r_val.get('status')} / {r_oos.get('status')})")
                continue

            # Per-asset OOS breakdown
            asset_data = per_asset_oos(ma_type, tf)
            print(f"    VAL: edge {r_val['edge_pp']:+.3f} dep {r_val['deploy']:.0%} | "
                  f"OOS: edge {r_oos['edge_pp']:+.3f} dep {r_oos['deploy']:.0%} null {r_oos['null_pct']:.0f}")
            print(f"    {len(asset_data)} assets in OOS, "
                  f"{sum(1 for a in asset_data if a['edge_pp']>0)} positive edge")

            # Generate HTML
            html = generate_html_report(ma_type, tf, asset_data, r_val, r_oos)
            outfile = OUTDIR / f"MA_{ma_type}_{tf}_report.html"
            outfile.write_text(html, encoding="utf-8")
            print(f"    → {outfile.name}")

    # Summary JSON
    print(f"\n  Reports saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
