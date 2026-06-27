"""MA PROD ENGINES — comprehensive report (NO shortcuts).

Uses the PRODUCTION V2 architecture from oos_engine_test.py:
  - select_band_trainval (curated band, FROZEN from TRAIN+VAL, OOS-excluded)
  - make_hold_provider (canonical fixed-EW ensemble)
  - decoupled lookback (14d) + 7d slice
  - V1 BTC-SMA100 regime gate (Combined V1+V2)
  - maker cost, lag-1 causal

Generates per TI type × TF:
  1. Summary: engine / always-long / combined (compound + linear/wk + slices + MaxDD)
  2. Per-asset OOS-2024 breakdown
  3. HTML report
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
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider
from strat.ma_type_upgrade import _ema, MA_TYPES
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"          # loads 2024 only (2025 sealed)
OOS_2024_END = "2024-12-31"
VAL_START = "2023-07-01"
VAL_END = "2024-05-15"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
VAL_WALL_MS = int(pd.Timestamp(VAL_END).value // 10**6)
OUTDIR = CRYPTO / "reports" / "MA_PROD_comprehensive"


def _btc_regime(wall_ms):
    rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not rec:
        return None
    d, _, _, _, cl = rec
    sma = pd.Series(cl, index=d).rolling(100).mean()
    s = pd.Series(0, index=d, dtype=int); st = 0
    for i in range(100, len(d)):
        if np.isnan(sma.iloc[i]):
            continue
        if st == 0 and cl[i] > sma.iloc[i] * 1.02:
            st = 1
        elif st == 1 and cl[i] < sma.iloc[i] * 0.98:
            st = 0
        s.iloc[i] = st
    # DE-LEAK (V1.1 align_htf_on bug class): regime[D] is the day-D 1d CLOSE (~23:59).
    # The per-bar prior[-1] broadcast on a day-D bar must read the last CLOSED 1d bar =
    # regime[D-1]; shift forward one 1d bar. See runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json.
    return s.shift(1).fillna(0).astype(int)


def _compute_held(sym_close, band_held, lookback_bars, slice_bars):
    """PROD decoupled confirmation + slice-hold (exactly oos_engine_test logic)."""
    fin = np.asarray(sym_close, dtype=float)
    m = np.isfinite(fin)
    held = np.zeros(len(fin), dtype=np.int8)
    if m.sum() < max(5, lookback_bars // 4):
        return held
    cseg = fin[m]
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[True], ema_lb[1:] >= ema_lb[:-1]])
    bh_seg = np.asarray(band_held, dtype=bool)[m]
    confirmed = bh_seg & above & rising
    held_seg = np.zeros(len(cseg), dtype=np.int8)
    i = 0; L = len(cseg)
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)
            j = i
            while j < end and confirmed[j]:
                held_seg[j] = 1
                j += 1
            i = j
        else:
            i += 1
    held[np.where(m)[0]] = held_seg
    return held


def run_prod_cell(ma_type, tf, lookback_days=14, slice_days=7, window="OOS"):
    """Run the PROD V2 engine. window in {VAL, OOS}. Returns metrics + per-asset."""
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    wall_ms = OOS_WALL_MS if window == "OOS" else VAL_WALL_MS
    win_start = pd.Timestamp(OOS_START if window == "OOS" else VAL_START)
    win_end = pd.Timestamp(OOS_2024_END if window == "OOS" else VAL_END)

    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    provider = make_hold_provider(sub_band, ma_type)
    regime = _btc_regime(wall_ms)

    close_dict, held_dict = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        band_held = provider(sym, o, h, l, c)
        held = _compute_held(c, band_held, lookback_bars, slice_bars)
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)

    if len(close_dict) < 10:
        return None

    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    win_dates = [d for d in all_dates if win_start <= d < win_end]
    if len(win_dates) < 10:
        return None

    # Per-asset accumulation
    asset_eng = {s: [] for s in close_dict}
    asset_al = {s: [] for s in close_dict}
    asset_deploy = {s: 0 for s in close_dict}
    asset_nbars = {s: 0 for s in close_dict}

    comb_rets, al_rets, eng_rets = [], [], []
    for i in range(1, len(win_dates)):
        dp, dc = win_dates[i-1], win_dates[i]
        # regime
        regime_val = 1
        if regime is not None:
            d1 = pd.Timestamp(dp).normalize()
            prior = regime.index[regime.index <= d1]
            regime_val = int(regime[prior[-1]]) if len(prior) else 0

        eng_r, all_r = [], []
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]; cc = close_dict[sym][dc]
                if cp > 0:
                    ret = (cc / cp) - 1.0
                    all_r.append(ret)
                    asset_al[sym].append(ret)
                    asset_nbars[sym] += 1
                    held_v = held_dict[sym][dp] if dp in held_dict[sym].index else 0
                    if held_v == 1:
                        eng_r.append(ret)
                        asset_deploy[sym] += 1
                        # combined: only if regime bull
                        asset_eng[sym].append(ret if regime_val == 1 else 0.0)
                    else:
                        asset_eng[sym].append(0.0)
        eng_rets.append(float(np.mean(eng_r)) if eng_r else 0.0)
        al_rets.append(float(np.mean(all_r)) if all_r else 0.0)
        comb_rets.append(eng_rets[-1] if regime_val == 1 else 0.0)

    comb = np.array(comb_rets); al = np.array(al_rets); eng = np.array(eng_rets)
    n_bars = len(comb)

    def metrics(rets):
        slice_pnls = []
        i = 0
        while i + slice_bars <= n_bars:
            slice_pnls.append(float(np.prod(1 + rets[i:i+slice_bars]) - 1) * 100)
            i += slice_bars
        comp = float(np.prod(1 + rets) - 1) * 100
        lin_wk = float(np.mean(rets)) * bpd * 7 * 100
        cum = np.cumprod(1 + rets)
        maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0
        sharpe = (float(np.mean(rets)) * bpd * 365) / (float(np.std(rets)) * np.sqrt(bpd * 365)) if np.std(rets) > 1e-12 else 0
        pct_prof = sum(1 for s in slice_pnls if s > 0) / len(slice_pnls) if slice_pnls else 0
        return {"compound": round(comp, 1), "lin_wk": round(lin_wk, 3), "maxdd": round(maxdd, 1),
                "sharpe": round(sharpe, 2), "pct_slices_prof": round(pct_prof, 3),
                "median_slice": round(float(np.median(slice_pnls)), 2) if slice_pnls else 0,
                "n_slices": len(slice_pnls)}

    m_comb = metrics(comb); m_al = metrics(al); m_eng = metrics(eng)
    deploy = float(np.mean(comb != 0)) if len(comb) else 0

    # ---- TRADE-LEVEL BREAKDOWN (per-asset discrete trades) ----
    # A "trade" = a contiguous run of held==1 for one asset. Its return is the
    # compounded gated return over that run. Classify win/loss.
    trades = []
    for sym in close_dict:
        if sym not in held_dict:
            continue
        h_ser = held_dict[sym]
        c_ser = close_dict[sym]
        # restrict to window
        in_win = (h_ser.index >= win_start) & (h_ser.index < win_end)
        h_w = h_ser[in_win].to_numpy()
        idx_w = h_ser.index[in_win]
        if len(h_w) < 3:
            continue
        c_w = c_ser.reindex(idx_w).to_numpy()
        # find contiguous held runs
        i = 0; L = len(h_w)
        while i < L:
            if h_w[i] == 1:
                j = i
                while j < L and h_w[j] == 1:
                    j += 1
                # trade from entry i to exit j (exclusive); return = c[j-1]/c[i]-1 approx
                # use bar returns within the run (entry executed at i, exit at min(j,L-1))
                entry_idx = i
                exit_idx = min(j, L - 1)
                if exit_idx > entry_idx and np.isfinite(c_w[entry_idx]) and np.isfinite(c_w[exit_idx]) and c_w[entry_idx] > 0:
                    tret = (c_w[exit_idx] / c_w[entry_idx] - 1.0) * 100
                    # subtract round-trip maker cost
                    tret -= MAKER_RT * 100
                    trades.append({
                        "asset": sym, "entry": str(idx_w[entry_idx])[:10],
                        "exit": str(idx_w[exit_idx])[:10],
                        "bars_held": exit_idx - entry_idx, "ret_pct": round(tret, 2),
                    })
                i = j
            else:
                i += 1

    n_trades = len(trades)
    wins = [t for t in trades if t["ret_pct"] > 0]
    losses = [t for t in trades if t["ret_pct"] <= 0]
    trade_stats = {
        "n_trades": n_trades,
        "n_wins": len(wins), "n_losses": len(losses),
        "win_rate": round(len(wins) / n_trades, 3) if n_trades else 0,
        "avg_win_pct": round(float(np.mean([t["ret_pct"] for t in wins])), 2) if wins else 0,
        "avg_loss_pct": round(float(np.mean([t["ret_pct"] for t in losses])), 2) if losses else 0,
        "avg_trade_pct": round(float(np.mean([t["ret_pct"] for t in trades])), 2) if trades else 0,
        "best_trade": max(trades, key=lambda t: t["ret_pct"]) if trades else None,
        "worst_trade": min(trades, key=lambda t: t["ret_pct"]) if trades else None,
        "avg_bars_held": round(float(np.mean([t["bars_held"] for t in trades])), 1) if trades else 0,
        "gross_win_pct": round(float(np.sum([t["ret_pct"] for t in wins])), 1) if wins else 0,
        "gross_loss_pct": round(float(np.sum([t["ret_pct"] for t in losses])), 1) if losses else 0,
        "profit_factor": round(abs(float(np.sum([t["ret_pct"] for t in wins])) /
                                   float(np.sum([t["ret_pct"] for t in losses]))), 2)
                         if losses and np.sum([t["ret_pct"] for t in losses]) != 0 else None,
        "top10_winners": sorted(wins, key=lambda t: -t["ret_pct"])[:10],
        "top10_losers": sorted(losses, key=lambda t: t["ret_pct"])[:10],
    }

    # per-asset
    asset_rows = []
    for sym in close_dict:
        if asset_nbars[sym] < 20:
            continue
        e = np.array(asset_eng[sym]); a = np.array(asset_al[sym])
        if len(e) < 20:
            continue
        eng_comp = float(np.prod(1 + e) - 1) * 100
        al_comp = float(np.prod(1 + a) - 1) * 100
        dep = asset_deploy[sym] / asset_nbars[sym] if asset_nbars[sym] else 0
        eng_lin = float(np.mean(e)) * bpd * 7 * 100
        al_lin = float(np.mean(a)) * bpd * 7 * 100
        asset_rows.append({
            "asset": sym, "deploy": round(dep, 3), "n_bars": asset_nbars[sym],
            "eng_lin_wk": round(eng_lin, 3), "al_lin_wk": round(al_lin, 3),
            "edge_pp": round(eng_lin - al_lin, 3),
            "eng_compound": round(eng_comp, 1), "al_compound": round(al_comp, 1),
        })
    asset_rows.sort(key=lambda x: -x["edge_pp"])

    return {
        "ma_type": ma_type, "tf": tf, "lookback_days": lookback_days, "slice_days": slice_days,
        "window": window, "n_assets": len(close_dict), "n_bars": n_bars,
        "n_band_configs": len(sub_band),
        "combined": m_comb, "always_long": m_al, "engine_nogate": m_eng,
        "deploy": round(deploy, 3),
        "edge_vs_al_lin_wk": round(m_comb["lin_wk"] - m_al["lin_wk"], 3),
        "assets": asset_rows,
        "trade_stats": trade_stats,
    }


def gen_html(ma_type, tf, val, oos):
    """Generate HTML report from VAL + OOS PROD results."""
    a = oos["assets"]
    n = len(a); pos = sum(1 for x in a if x["edge_pp"] > 0)
    title = f"PROD MA Engine: {ma_type}|{tf} (V2 band + 14d lookback + 7d slice + regime gate)"
    cv, co = val["combined"], oos["combined"]
    al_o = oos["always_long"]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:13px/1.5 Arial,sans-serif;margin:30px;background:#fafafa}}
.container{{max-width:1100px;margin:auto}}
h1{{font-size:21px;margin:0 0 4px}} h2{{font-size:16px;margin:22px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
h3{{font-size:14px;margin:14px 0 6px}}
.meta{{color:#666;font-size:12px;margin-bottom:14px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}
th,td{{border:1px solid #ddd;padding:5px 8px;text-align:right;font-size:12px}}
th{{background:#f0f0f0;font-weight:600}} td:first-child,th:first-child{{text-align:left}}
.pos{{color:#1a7a3a;font-weight:600}} .neg{{color:#a52828;font-weight:600}} .neutral{{color:#666}}
.kpi{{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 18px;margin:4px 8px 4px 0;min-width:110px;text-align:center}}
.kpi .val{{font-size:19px;font-weight:700}} .kpi .lbl{{font-size:11px;color:#888}}
.note{{background:#eef7ee;border-left:3px solid #3cb371;padding:10px;margin:12px 0;font-size:12px}}
.bar{{display:inline-block;height:11px;border-radius:2px}} .bp{{background:#3cb371}} .bn{{background:#c0504d}}
</style></head><body><div class="container">
<h1>{title}</h1>
<div class="meta">VAL {VAL_START}→{VAL_END} | OOS-2024 {OOS_START}→{OOS_2024_END} (2025 sealed) |
{oos['n_band_configs']} curated band configs (frozen TRAIN+VAL) | {oos['n_assets']} assets |
maker {MAKER_RT} RT | Generated {datetime.now():%Y-%m-%d %H:%M}</div>

<div class="note"><b>PROD architecture (no shortcuts):</b> select_band_trainval curated band +
make_hold_provider ensemble + 14d decoupled lookback + 7d slice-hold + V1 BTC-SMA100 regime gate.
This is the canonical V2 engine, not a simplified approximation.</div>

<h2>OOS-2024 KPIs (Combined V1+V2, regime-gated)</h2>
<div>
<div class="kpi"><div class="val {'pos' if oos['edge_vs_al_lin_wk']>0 else 'neg'}">{oos['edge_vs_al_lin_wk']:+.2f}pp</div><div class="lbl">Edge vs AL /wk</div></div>
<div class="kpi"><div class="val pos">{co['lin_wk']:+.2f}%</div><div class="lbl">Linear /wk</div></div>
<div class="kpi"><div class="val">{co['compound']:+.0f}%</div><div class="lbl">Compound</div></div>
<div class="kpi"><div class="val">{oos['deploy']:.0%}</div><div class="lbl">Deploy</div></div>
<div class="kpi"><div class="val">{co['sharpe']:.2f}</div><div class="lbl">Sharpe</div></div>
<div class="kpi"><div class="val">{co['maxdd']:.0f}%</div><div class="lbl">MaxDD</div></div>
<div class="kpi"><div class="val">{co['pct_slices_prof']:.0%}</div><div class="lbl">Slices Profitable</div></div>
</div>

<h2>VAL vs OOS (Combined V1+V2)</h2>
<table>
<tr><th>Metric</th><th>VAL</th><th>OOS-2024</th><th>Always-Long (OOS)</th></tr>
<tr><td>Linear /wk</td><td>{cv['lin_wk']:+.3f}%</td><td class="pos">{co['lin_wk']:+.3f}%</td><td>{al_o['lin_wk']:+.3f}%</td></tr>
<tr><td>Compound</td><td>{cv['compound']:+.0f}%</td><td>{co['compound']:+.0f}%</td><td>{al_o['compound']:+.0f}%</td></tr>
<tr><td>Sharpe (ann)</td><td>{cv['sharpe']:.2f}</td><td>{co['sharpe']:.2f}</td><td>{al_o['sharpe']:.2f}</td></tr>
<tr><td>MaxDD</td><td>{cv['maxdd']:.0f}%</td><td>{co['maxdd']:.0f}%</td><td>{al_o['maxdd']:.0f}%</td></tr>
<tr><td>Slices profitable</td><td>{cv['pct_slices_prof']:.0%}</td><td>{co['pct_slices_prof']:.0%}</td><td>{al_o['pct_slices_prof']:.0%}</td></tr>
<tr><td>Median slice</td><td>{cv['median_slice']:+.2f}%</td><td>{co['median_slice']:+.2f}%</td><td>{al_o['median_slice']:+.2f}%</td></tr>
</table>
"""
    # ---- Trade-level breakdown (what worked vs what didn't) ----
    ts = oos.get("trade_stats", {})
    if ts and ts.get("n_trades"):
        pf = ts.get("profit_factor")
        pf_str = f"{pf:.2f}" if pf is not None else "n/a"
        html += f"""<h2>Trade-Level Breakdown — what worked vs what didn't</h2>
<div class="note">A "trade" = one contiguous held run for one asset (entry→exit), net of {MAKER_RT*100:.2f}% round-trip maker cost.
{ts['n_trades']} discrete trades in OOS-2024.</div>
<div>
<div class="kpi"><div class="val">{ts['n_trades']}</div><div class="lbl">Total Trades</div></div>
<div class="kpi"><div class="val {'pos' if ts['win_rate']>=0.5 else 'neutral'}">{ts['win_rate']:.0%}</div><div class="lbl">Win Rate</div></div>
<div class="kpi"><div class="val pos">{ts['avg_win_pct']:+.2f}%</div><div class="lbl">Avg Win</div></div>
<div class="kpi"><div class="val neg">{ts['avg_loss_pct']:+.2f}%</div><div class="lbl">Avg Loss</div></div>
<div class="kpi"><div class="val {'pos' if ts['avg_trade_pct']>0 else 'neg'}">{ts['avg_trade_pct']:+.2f}%</div><div class="lbl">Avg Trade</div></div>
<div class="kpi"><div class="val">{pf_str}</div><div class="lbl">Profit Factor</div></div>
<div class="kpi"><div class="val">{ts['avg_bars_held']:.0f}</div><div class="lbl">Avg Bars Held</div></div>
</div>
<table>
<tr><th>Bucket</th><th>Count</th><th>Gross P&amp;L</th><th>Avg/trade</th></tr>
<tr><td class="pos">Winning trades</td><td>{ts['n_wins']}</td><td class="pos">{ts['gross_win_pct']:+.0f}%</td><td>{ts['avg_win_pct']:+.2f}%</td></tr>
<tr><td class="neg">Losing trades</td><td>{ts['n_losses']}</td><td class="neg">{ts['gross_loss_pct']:+.0f}%</td><td>{ts['avg_loss_pct']:+.2f}%</td></tr>
</table>
<h3>Top 10 Winning Trades</h3>
<table><tr><th>Asset</th><th>Entry</th><th>Exit</th><th>Bars</th><th>Return</th></tr>
"""
        for t in ts.get("top10_winners", []):
            html += (f"<tr><td>{t['asset']}</td><td>{t['entry']}</td><td>{t['exit']}</td>"
                     f"<td>{t['bars_held']}</td><td class='pos'>{t['ret_pct']:+.2f}%</td></tr>\n")
        html += """</table>
<h3>Top 10 Losing Trades</h3>
<table><tr><th>Asset</th><th>Entry</th><th>Exit</th><th>Bars</th><th>Return</th></tr>
"""
        for t in ts.get("top10_losers", []):
            html += (f"<tr><td>{t['asset']}</td><td>{t['entry']}</td><td>{t['exit']}</td>"
                     f"<td>{t['bars_held']}</td><td class='neg'>{t['ret_pct']:+.2f}%</td></tr>\n")
        html += "</table>\n"

    html += f"""<h2>Per-Asset OOS-2024 ({n} assets, {pos} positive edge)</h2>
<table><tr><th>Asset</th><th>Deploy</th><th>Eng lin/wk</th><th>AL lin/wk</th><th>Edge pp</th><th>Eng comp</th><th>AL comp</th><th></th></tr>
"""
    max_e = max((abs(x["edge_pp"]) for x in a), default=1)
    for x in a:
        cls = "pos" if x["edge_pp"] > 0 else "neg"
        bw = min(abs(x["edge_pp"]) / max_e * 90, 90)
        bc = "bp" if x["edge_pp"] > 0 else "bn"
        html += (f"<tr><td>{x['asset']}</td><td>{x['deploy']:.0%}</td><td>{x['eng_lin_wk']:+.2f}%</td>"
                 f"<td>{x['al_lin_wk']:+.2f}%</td><td class='{cls}'>{x['edge_pp']:+.2f}</td>"
                 f"<td>{x['eng_compound']:+.0f}%</td><td>{x['al_compound']:+.0f}%</td>"
                 f"<td><span class='bar {bc}' style='width:{bw:.0f}px'></span></td></tr>\n")
    html += "</table></div></body></html>"
    return html


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="all")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--lookback", type=int, default=14)
    a = ap.parse_args()
    types = MA_TYPES if a.types == "all" else [t.strip().upper() for t in a.types.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"MA PROD ENGINES — comprehensive report (lookback={a.lookback}d, slice=7d)")
    print("=" * 100)
    print(f"  {'type':<6} {'tf':<4} | {'VAL/wk':>8} {'OOS/wk':>8} {'vsAL':>7} {'dep':>5} "
          f"{'Sh':>5} {'%prof':>6} | assets+")
    print("  " + "-" * 78)

    summary = []
    for ma in types:
        for tf in tfs:
            val = run_prod_cell(ma, tf, a.lookback, 7, window="VAL")
            oos = run_prod_cell(ma, tf, a.lookback, 7, window="OOS")
            if val is None or oos is None:
                print(f"  {ma:<6} {tf:<4} | SKIP")
                continue
            pos = sum(1 for x in oos["assets"] if x["edge_pp"] > 0)
            tstat = oos.get("trade_stats", {})
            print(f"  {ma:<6} {tf:<4} | {val['combined']['lin_wk']:>+7.3f} "
                  f"{oos['combined']['lin_wk']:>+7.3f} {oos['edge_vs_al_lin_wk']:>+6.2f} "
                  f"{oos['deploy']:>4.0%} {oos['combined']['sharpe']:>5.2f} "
                  f"{oos['combined']['pct_slices_prof']:>5.0%} | {pos}/{len(oos['assets'])} | "
                  f"trades {tstat.get('n_trades',0)} WR {tstat.get('win_rate',0):.0%} "
                  f"PF {tstat.get('profit_factor') or 0:.2f}")
            html = gen_html(ma, tf, val, oos)
            (OUTDIR / f"PROD_MA_{ma}_{tf}.html").write_text(html, encoding="utf-8")
            summary.append({"ma": ma, "tf": tf, "val": val["combined"], "oos": oos["combined"],
                            "edge": oos["edge_vs_al_lin_wk"], "deploy": oos["deploy"],
                            "n_pos_assets": pos, "n_assets": len(oos["assets"]),
                            "trade_stats": oos.get("trade_stats")})

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"prod_summary_{ts}.json", "w") as f:
        json.dump({"summary": summary, "lookback": a.lookback, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Reports + summary saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
