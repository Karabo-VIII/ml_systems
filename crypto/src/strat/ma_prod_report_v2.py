"""MA PROD REPORT v2 — built on the AUDITED V8 slice engine (19/19 pass).

Reproduces per-candidate PROD reports for the deployable tiers (1d/4h/2h), with:
  - ROLLING slices, PROD cost (0.0028 RT taker+slip)
  - Tier config: 1d=14d/14d, 4h=14d/7d, 2h=7d/7d+whipsaw(minhold3/cooldown3)
  - EXPLICIT DATE SLICING shown: every window prints its wall date AND the actual
    data min/max bar that loaded (so you can SEE the slicing, not infer it)
  - Per-asset OOS breakdown + trade-level breakdown (win/loss, profit factor)
  - VAL vs OOS side-by-side

This supersedes ma_prod_report.py (which used a non-audited reimplementation).
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
    _confirmed_signal, held_rolling, _cached_band_held, btc_regime,
    VAL_START, VAL_END, OOS_START, OOS_2024_END, VAL_WALL_MS, OOS_WALL_MS,
)
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "reports" / "MA_PROD_v2"

# Tier config (lookback_days, slice_days, min_hold_bars, cooldown_bars) — from V8 study
TIER_CFG = {
    "1d": {"lookback": 14, "slice": 14, "min_hold": 0, "cooldown": 0, "tier": "LONG"},
    "4h": {"lookback": 14, "slice": 7,  "min_hold": 0, "cooldown": 0, "tier": "MEDIUM"},
    "2h": {"lookback": 7,  "slice": 7,  "min_hold": 3, "cooldown": 3, "tier": "MEDIUM"},
}
CANDIDATES = {  # the OOS-validated types per TF (from V8 study, prod rolling)
    "1d": ["VIDYA", "SMA", "EMA", "DEMA"],
    "4h": ["SMA", "VIDYA", "WMA", "EMA", "KAMA"],
    "2h": ["SMA", "VIDYA", "EMA"],
}


def run_candidate(ma_type, tf, window):
    """Run one candidate via the audited engine, returning metrics + per-asset + trades
    + EXPLICIT date provenance for the window."""
    cfg = TIER_CFG[tf]
    bpd = BARS_PER_DAY[tf]
    lookback_bars = cfg["lookback"] * bpd
    slice_bars = cfg["slice"] * bpd
    wall_ms = OOS_WALL_MS if window == "OOS" else VAL_WALL_MS
    win_s = pd.Timestamp(OOS_START if window == "OOS" else VAL_START)
    win_e = pd.Timestamp(OOS_2024_END if window == "OOS" else VAL_END)
    wall_date = "2025-01-01" if window == "OOS" else VAL_END

    regime = btc_regime(wall_ms)
    close_cols, held_cols = {}, {}
    data_min, data_max = None, None
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
        dmin, dmax = dates.min(), dates.max()
        data_min = dmin if data_min is None else min(data_min, dmin)
        data_max = dmax if data_max is None else max(data_max, dmax)

    if len(close_cols) < 10:
        return None

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)

    rets = close_df.pct_change().to_numpy()[1:]
    held_lag = held_df.shift(1).fillna(0).to_numpy()[1:]
    turn = held_df.diff().abs().fillna(0).to_numpy()[1:]
    dates = close_df.index[1:]
    reg_vals = np.ones(len(dates))
    if regime is not None:
        norm = pd.DatetimeIndex(dates).normalize()
        idx = regime.index.union(norm).drop_duplicates()
        ra = regime.reindex(idx).ffill().reindex(norm).fillna(0)
        reg_vals = ra.to_numpy().astype(float)

    mask = (dates >= win_s) & (dates < win_e)
    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    eng_gross = np.where(n_held > 0, (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (COST_PROD / 2.0) / np.maximum(n_held, 1), 0.0)
    eng = (eng_gross - cost) * reg_vals
    al = np.where(valid.sum(axis=1) > 0, (rets_f * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1), 0.0)

    eng_w = eng[mask]; al_w = al[mask]
    win_dates = dates[mask]
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0
    lin_wk = float(np.mean(eng_w)) * bpd * 7 * 100
    al_lin_wk = float(np.mean(al_w)) * bpd * 7 * 100
    comp = float(np.prod(1 + eng_w) - 1) * 100
    cum = np.cumprod(1 + eng_w)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0
    sharpe = (float(np.mean(eng_w)) * bpd * 365) / (float(np.std(eng_w)) * np.sqrt(bpd * 365)) if np.std(eng_w) > 1e-12 else 0
    slice_pnls = []
    i = 0
    while i + slice_bars <= len(eng_w):
        slice_pnls.append(float(np.prod(1 + eng_w[i:i+slice_bars]) - 1) * 100)
        i += slice_bars
    pct_prof = sum(1 for s in slice_pnls if s > 0) / len(slice_pnls) if slice_pnls else 0

    # per-asset
    asset_rows = []
    sub_close = close_df[mask if len(mask) == len(close_df) else (close_df.index >= win_s) & (close_df.index < win_e)]
    cmask = (close_df.index >= win_s) & (close_df.index < win_e)
    for sym in close_df.columns:
        cs = close_df[sym][cmask]
        hs = held_df[sym].shift(1).fillna(0)[cmask]
        r = cs.pct_change().to_numpy()[1:]
        hh = hs.to_numpy()[1:]
        v = np.isfinite(r)
        if v.sum() < 20:
            continue
        rg = reg_vals[(dates >= win_s) & (dates < win_e)][:len(r)] if regime is not None else np.ones(len(r))
        eng_r = np.where(v, r * hh * rg[:len(r)], 0.0)
        eng_comp = float(np.prod(1 + eng_r) - 1) * 100
        bh_comp = float(np.prod(1 + np.where(v, r, 0.0)) - 1) * 100
        dep = float(np.mean(hh[v] > 0))
        asset_rows.append({"asset": sym, "deploy": round(dep, 3),
                           "eng_compound": round(eng_comp, 1), "bh_compound": round(bh_comp, 1),
                           "edge": round(eng_comp - bh_comp, 1)})
    asset_rows.sort(key=lambda x: -x["edge"])

    # trades
    trades = []
    for sym in close_df.columns:
        hser = held_df[sym]
        cser = close_df[sym]
        inw = (hser.index >= win_s) & (hser.index < win_e)
        hw = hser[inw].to_numpy(); idxw = hser.index[inw]
        cw = cser.reindex(idxw).to_numpy()
        i = 0; L = len(hw)
        while i < L:
            if hw[i] == 1:
                j = i
                while j < L and hw[j] == 1:
                    j += 1
                ei = min(j, L - 1)
                if ei > i and np.isfinite(cw[i]) and np.isfinite(cw[ei]) and cw[i] > 0:
                    tret = (cw[ei] / cw[i] - 1.0) * 100 - COST_PROD * 100
                    trades.append({"asset": sym, "entry": str(idxw[i])[:10], "exit": str(idxw[ei])[:10],
                                   "bars": ei - i, "ret": round(tret, 2)})
                i = j
            else:
                i += 1
    wins = [t for t in trades if t["ret"] > 0]; losses = [t for t in trades if t["ret"] <= 0]
    gw = sum(t["ret"] for t in wins); gl = sum(t["ret"] for t in losses)
    trade_stats = {
        "n": len(trades), "win_rate": round(len(wins) / len(trades), 3) if trades else 0,
        "avg_win": round(np.mean([t["ret"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t["ret"] for t in losses]), 2) if losses else 0,
        "profit_factor": round(abs(gw / gl), 2) if gl != 0 else None,
        "avg_bars": round(np.mean([t["bars"] for t in trades]), 1) if trades else 0,
        "top_win": sorted(wins, key=lambda t: -t["ret"])[:8],
        "top_loss": sorted(losses, key=lambda t: t["ret"])[:8],
    }

    return {
        "ma_type": ma_type, "tf": tf, "tier": cfg["tier"], "window": window,
        "lookback_days": cfg["lookback"], "slice_days": cfg["slice"],
        "min_hold_bars": cfg["min_hold"], "cooldown_bars": cfg["cooldown"],
        "cost_rt": COST_PROD, "n_assets": len(close_cols),
        # EXPLICIT DATE PROVENANCE (the user wants to SEE the slicing)
        "window_label": window,
        "window_start": str(win_s)[:10], "window_end": str(win_e)[:10],
        "data_wall": wall_date,
        "loaded_data_min": str(data_min)[:16], "loaded_data_max": str(data_max)[:16],
        "n_window_bars": int(mask.sum()),
        "first_window_bar": str(win_dates.min())[:16] if len(win_dates) else "n/a",
        "last_window_bar": str(win_dates.max())[:16] if len(win_dates) else "n/a",
        "deploy": round(deploy, 4), "lin_wk": round(lin_wk, 4), "al_lin_wk": round(al_lin_wk, 4),
        "edge_pp": round(lin_wk - al_lin_wk, 4), "compound": round(comp, 1),
        "maxdd": round(maxdd, 1), "sharpe": round(sharpe, 3),
        "n_7d_slices": len(slice_pnls), "pct_profitable_7d_slices": round(pct_prof, 3),
        "assets": asset_rows, "trade_stats": trade_stats,
    }


def gen_html(ma_type, tf, val, oos):
    cfg = TIER_CFG[tf]
    a = oos["assets"]; n = len(a); pos = sum(1 for x in a if x["edge"] > 0)
    ts = oos["trade_stats"]
    title = f"PROD v2 — {ma_type}|{tf} ({oos['tier']})"
    pf = ts.get("profit_factor"); pf_s = f"{pf:.2f}" if pf else "n/a"
    wk = "whipsaw(min-hold {0}/cooldown {1})".format(cfg["min_hold"], cfg["cooldown"]) if cfg["min_hold"] else "none"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:13px/1.5 Arial,sans-serif;margin:28px;background:#fafafa}}.container{{max-width:1080px;margin:auto}}
h1{{font-size:20px;margin:0 0 4px}}h2{{font-size:15px;margin:20px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
h3{{font-size:13px;margin:12px 0 6px}}.meta{{color:#555;font-size:12px;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 16px}}th,td{{border:1px solid #ddd;padding:5px 8px;text-align:right;font-size:12px}}
th{{background:#f0f0f0;font-weight:600}}td:first-child,th:first-child{{text-align:left}}
.pos{{color:#1a7a3a;font-weight:600}}.neg{{color:#a52828;font-weight:600}}
.kpi{{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 16px;margin:4px 6px 4px 0;min-width:100px;text-align:center}}
.kpi .val{{font-size:18px;font-weight:700}}.kpi .lbl{{font-size:10px;color:#888}}
.dates{{background:#eef3fb;border-left:3px solid #4a78c0;padding:10px 14px;margin:10px 0;font-size:12px;font-family:monospace}}
.note{{background:#eef7ee;border-left:3px solid #3cb371;padding:9px;margin:10px 0;font-size:12px}}
.bar{{display:inline-block;height:11px;border-radius:2px}}.bp{{background:#3cb371}}.bn{{background:#c0504d}}
</style></head><body><div class="container">
<h1>{title}</h1>
<div class="meta">Audited V8 slice engine (19/19 pass) · ROLLING · PROD cost {oos['cost_rt']} RT (taker+slip) ·
lookback {cfg['lookback']}d / hold {cfg['slice']}d · whipsaw {wk} · regime-gated · Generated {datetime.now():%Y-%m-%d %H:%M}</div>

<h2>Date Slicing (explicit provenance)</h2>
<div class="dates">
<b>VAL window</b>&nbsp; {val['window_start']} → {val['window_end']} &nbsp;|&nbsp; data wall {val['data_wall']}<br>
&nbsp; loaded bars: {val['loaded_data_min']} … {val['loaded_data_max']} &nbsp;|&nbsp; in-window bars: {val['n_window_bars']}
({val['first_window_bar']} … {val['last_window_bar']})<br>
<b>OOS-2024 window</b>&nbsp; {oos['window_start']} → {oos['window_end']} &nbsp;|&nbsp; data wall {oos['data_wall']} &nbsp;<b>(2025 SEALED)</b><br>
&nbsp; loaded bars: {oos['loaded_data_min']} … {oos['loaded_data_max']} &nbsp;|&nbsp; in-window bars: {oos['n_window_bars']}
({oos['first_window_bar']} … {oos['last_window_bar']})
</div>

<h2>OOS-2024 KPIs</h2>
<div>
<div class="kpi"><div class="val {'pos' if oos['edge_pp']>0 else 'neg'}">{oos['edge_pp']:+.2f}pp</div><div class="lbl">Edge vs AL/wk</div></div>
<div class="kpi"><div class="val pos">{oos['lin_wk']:+.2f}%</div><div class="lbl">Linear/wk</div></div>
<div class="kpi"><div class="val">{oos['compound']:+.0f}%</div><div class="lbl">Compound</div></div>
<div class="kpi"><div class="val">{oos['deploy']:.0%}</div><div class="lbl">Deploy</div></div>
<div class="kpi"><div class="val">{oos['sharpe']:.2f}</div><div class="lbl">Sharpe</div></div>
<div class="kpi"><div class="val">{oos['maxdd']:.0f}%</div><div class="lbl">MaxDD</div></div>
<div class="kpi"><div class="val">{oos['pct_profitable_7d_slices']:.0%}</div><div class="lbl">7d slices prof</div></div>
</div>

<h2>VAL vs OOS</h2>
<table><tr><th>Metric</th><th>VAL</th><th>OOS-2024</th></tr>
<tr><td>Linear/wk</td><td>{val['lin_wk']:+.3f}%</td><td class="pos">{oos['lin_wk']:+.3f}%</td></tr>
<tr><td>Edge vs AL/wk</td><td>{val['edge_pp']:+.3f}</td><td>{oos['edge_pp']:+.3f}</td></tr>
<tr><td>Compound</td><td>{val['compound']:+.0f}%</td><td>{oos['compound']:+.0f}%</td></tr>
<tr><td>Sharpe</td><td>{val['sharpe']:.2f}</td><td>{oos['sharpe']:.2f}</td></tr>
<tr><td>MaxDD</td><td>{val['maxdd']:.0f}%</td><td>{oos['maxdd']:.0f}%</td></tr>
<tr><td>Deploy</td><td>{val['deploy']:.0%}</td><td>{oos['deploy']:.0%}</td></tr>
<tr><td>7d slices profitable</td><td>{val['pct_profitable_7d_slices']:.0%}</td><td>{oos['pct_profitable_7d_slices']:.0%}</td></tr>
</table>

<h2>Trade-Level (OOS-2024)</h2>
<div class="note">Trade = contiguous held run, net of {oos['cost_rt']*100:.2f}% RT. {ts['n']} trades.</div>
<div>
<div class="kpi"><div class="val">{ts['n']}</div><div class="lbl">Trades</div></div>
<div class="kpi"><div class="val">{ts['win_rate']:.0%}</div><div class="lbl">Win Rate</div></div>
<div class="kpi"><div class="val pos">{ts['avg_win']:+.2f}%</div><div class="lbl">Avg Win</div></div>
<div class="kpi"><div class="val neg">{ts['avg_loss']:+.2f}%</div><div class="lbl">Avg Loss</div></div>
<div class="kpi"><div class="val">{pf_s}</div><div class="lbl">Profit Factor</div></div>
<div class="kpi"><div class="val">{ts['avg_bars']:.0f}</div><div class="lbl">Avg Bars Held</div></div>
</div>
<h3>Top Winning Trades</h3>
<table><tr><th>Asset</th><th>Entry</th><th>Exit</th><th>Bars</th><th>Return</th></tr>
"""
    for t in ts["top_win"]:
        html += f"<tr><td>{t['asset']}</td><td>{t['entry']}</td><td>{t['exit']}</td><td>{t['bars']}</td><td class='pos'>{t['ret']:+.2f}%</td></tr>\n"
    html += "</table><h3>Top Losing Trades</h3><table><tr><th>Asset</th><th>Entry</th><th>Exit</th><th>Bars</th><th>Return</th></tr>\n"
    for t in ts["top_loss"]:
        html += f"<tr><td>{t['asset']}</td><td>{t['entry']}</td><td>{t['exit']}</td><td>{t['bars']}</td><td class='neg'>{t['ret']:+.2f}%</td></tr>\n"

    html += f"""</table>
<h2>Per-Asset OOS-2024 ({n} assets, {pos} positive edge)</h2>
<table><tr><th>Asset</th><th>Deploy</th><th>Eng comp</th><th>Buy&amp;Hold comp</th><th>Edge</th><th></th></tr>
"""
    mx = max((abs(x["edge"]) for x in a), default=1)
    for x in a:
        cls = "pos" if x["edge"] > 0 else "neg"; bw = min(abs(x["edge"]) / mx * 90, 90); bc = "bp" if x["edge"] > 0 else "bn"
        html += (f"<tr><td>{x['asset']}</td><td>{x['deploy']:.0%}</td><td>{x['eng_compound']:+.0f}%</td>"
                 f"<td>{x['bh_compound']:+.0f}%</td><td class='{cls}'>{x['edge']:+.0f}</td>"
                 f"<td><span class='bar {bc}' style='width:{bw:.0f}px'></span></td></tr>\n")
    html += "</table></div></body></html>"
    return html


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="1d,4h,2h")
    a = ap.parse_args()
    tfs = [t.strip() for t in a.tfs.split(",")]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 96)
    print(f"MA PROD REPORT v2 (audited engine) — run-dated {ts}")
    print("=" * 96)
    print(f"  {'type':<6}{'tf':<4}{'tier':<7}| {'VAL/wk':>8}{'OOS/wk':>8}{'edge':>7}{'dep':>5}"
          f"{'Sh':>5}{'7dprof':>7}{'trades':>7}{'PF':>5}")
    print("  " + "-" * 80)

    summary = []
    for tf in tfs:
        for ma in CANDIDATES[tf]:
            val = run_candidate(ma, tf, "VAL")
            oos = run_candidate(ma, tf, "OOS")
            if val is None or oos is None:
                print(f"  {ma:<6}{tf:<4}| SKIP"); continue
            ts_o = oos["trade_stats"]
            print(f"  {ma:<6}{tf:<4}{oos['tier']:<7}| {val['lin_wk']:>+7.2f}{oos['lin_wk']:>+8.2f}"
                  f"{oos['edge_pp']:>+7.2f}{oos['deploy']:>5.0%}{oos['sharpe']:>5.2f}"
                  f"{oos['pct_profitable_7d_slices']:>6.0%}{ts_o['n']:>7}"
                  f"{(ts_o['profit_factor'] or 0):>5.2f}")
            html = gen_html(ma, tf, val, oos)
            (OUTDIR / f"PRODv2_{ma}_{tf}_{ts}.html").write_text(html, encoding="utf-8")
            summary.append({"ma": ma, "tf": tf, "val": val, "oos": oos})

    with open(OUTDIR / f"prodv2_summary_{ts}.json", "w") as f:
        json.dump({"summary": summary, "run_dated": ts,
                   "windows": {"VAL": [VAL_START, VAL_END], "OOS": [OOS_START, OOS_2024_END]}},
                  f, indent=2, default=str)
    print(f"\n  Run-dated {ts}. Reports + summary in {OUTDIR}")


if __name__ == "__main__":
    main()
