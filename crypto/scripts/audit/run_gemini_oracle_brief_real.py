"""Run Gemini's exact oracle-brief script against REAL Binance public data.

Implements the methodology specified in docs/ORACLE_EXERCISE_BRIEF_FOR_GEMINI_2026_05_18.md
section by section. Outputs tables in Gemini's exact tabular format so the user
can paste them into a Gemini follow-up for verdict-only commentary.

No look-ahead. 24bps RT cost. 4% size. NEVER touches data >= 2026-01-01.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 87 symbols from the brief
SYMBOLS = [
    "AAVE","ADA","ALGO","APT","AR","ARB","ARKM","ASTER","ATOM","AVAX","BCH","BIO",
    "BLUR","BNB","BONK","BTC","CFG","CHZ","CRV","D","DASH","DEXE","DOGE","DOT",
    "DYDX","EIGEN","ENA","ENJ","ETC","ETH","FET","FIL","FLOKI","GIGGLE","GUN",
    "HBAR","ICP","INJ","JST","KAT","LDO","LINK","LTC","MOVR","NEAR","NEIRO","NIGHT",
    "ONDO","OP","ORDI","PENGU","PEPE","PLUME","PNUT","POL","PROM","RENDER","SEI",
    "SHIB","SOL","SPK","STO","SUI","SUPER","TAO","TIA","TON","TREE","TRUMP","TRX",
    "U","UNI","VIRTUAL","W","WIF","WLD","WLFI","XAUT","XLM","XPL","XRP","XUSD",
    "ZAMA","ZBT","ZEC","ZEN","ZRO",
]

# DNA bucket mapping from §3c of the brief
BLUE = {"BTC","ETH"}
STEADY = {"BCH","BNB","ETC","SOL","TRX","XRP"}
DEGEN = {"BONK","PEPE","SHIB","WIF","WLD"}
VOLATILE_LIST = {"ADA","ALGO","ATOM","AVAX","DASH","DOGE","DOT","FET","FIL","LINK",
                  "LTC","MATIC","NEAR","RENDER","RNDR","TAO","ZEC"}
def bucket_of(sym):
    if sym in BLUE: return "BLUE"
    if sym in STEADY: return "STEADY"
    if sym in DEGEN: return "DEGEN"
    if sym in VOLATILE_LIST: return "VOLATILE"
    return "VOLATILE"

START_MS = 1704067200000  # 2024-01-01 00:00:00 UTC
END_MS   = 1767225599999  # 2025-12-31 23:59:59 UTC (was off-by-one-year in brief!)


def fetch_klines(sym):
    """Fetch 1d klines from Binance with fallback to 1000{sym} prefix for high-supply meme tokens."""
    url = "https://api.binance.com/api/v3/klines"
    for try_sym in [f"{sym}USDT", f"1000{sym}USDT"]:
        params = {
            "symbol": try_sym,
            "interval": "1d",
            "startTime": START_MS,
            "endTime": END_MS,
            "limit": 1000,
        }
        try:
            r = requests.get(url, params=params, timeout=20)
        except Exception as e:
            return None, f"net-error: {e}"
        if r.status_code == 200 and r.json():
            rows = []
            for k in r.json():
                rows.append({
                    "asset": sym,
                    "openTime": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "trades": int(k[8]),
                })
            return pd.DataFrame(rows), try_sym
    return None, f"no-data ({sym}USDT and 1000{sym}USDT both empty/404)"


def main():
    print("=" * 78)
    print("ORACLE EXERCISE — running Gemini-brief methodology against REAL Binance data")
    print(f"Window: 2024-01-01 -> 2025-12-31 (UNSEEN >= 2026-01-01 NOT TOUCHED)")
    print("=" * 78)

    # ---- Data acquisition -----------------------------------------------------
    frames, skipped, sym_map = [], [], {}
    for s in SYMBOLS:
        df, info = fetch_klines(s)
        if df is None or len(df) < 30:
            skipped.append((s, info))
            print(f"  SKIP {s}: {info}")
        else:
            frames.append(df)
            sym_map[s] = info
            print(f"  OK   {s} ({info}): {len(df)} rows")
        time.sleep(0.06)  # polite delay

    print()
    print(f"PULLED: {len(frames)} usable / {len(SYMBOLS)} requested ({len(skipped)} skipped)")
    print(f"Skipped: {[s for s,_ in skipped]}")

    panel = pd.concat(frames)
    panel["date"] = pd.to_datetime(panel["openTime"], unit="ms").dt.date
    panel = panel.sort_values(["asset","date"]).reset_index(drop=True)

    # Forward-return columns (per-asset!)
    panel["ret_1d"]    = panel.groupby("asset")["close"].pct_change(1)
    panel["close_t1"]  = panel.groupby("asset")["close"].shift(-1)
    panel["close_t2"]  = panel.groupby("asset")["close"].shift(-2)
    panel["close_t4"]  = panel.groupby("asset")["close"].shift(-4)
    panel["close_t6"]  = panel.groupby("asset")["close"].shift(-6)
    panel["close_t8"]  = panel.groupby("asset")["close"].shift(-8)
    panel["fwd_1d_from_t1"] = panel["close_t2"] / panel["close_t1"] - 1
    panel["fwd_3d_from_t1"] = panel["close_t4"] / panel["close_t1"] - 1
    panel["fwd_5d_from_t1"] = panel["close_t6"] / panel["close_t1"] - 1
    panel["fwd_7d_from_t1"] = panel["close_t8"] / panel["close_t1"] - 1

    # Restrict to 8Q WF (UNSEEN exclusion)
    panel = panel[(panel["date"] >= pd.Timestamp("2024-01-01").date()) &
                  (panel["date"] <= pd.Timestamp("2025-12-31").date())].copy()

    # BTC regime
    btc = panel[panel["asset"] == "BTC"][["date","close"]].sort_values("date").copy()
    btc["close_30d_ago"] = btc["close"].shift(30)
    btc["btc_30d"]      = btc["close"] / btc["close_30d_ago"] - 1
    btc["regime"]       = btc["btc_30d"].apply(
        lambda x: "crash" if (pd.notna(x) and x <= -0.15) else
                  ("bear"  if (pd.notna(x) and x <= -0.05) else
                  ("bull"  if (pd.notna(x) and x >=  0.05) else "chop"))
    )
    panel = panel.merge(btc[["date","btc_30d","regime"]], on="date", how="left")
    panel["bucket"] = panel["asset"].apply(bucket_of)
    panel["quarter"] = panel["date"].apply(
        lambda d: f"{(d.year - 2024) * 4 + (d.month-1)//3 + 1}"
    )
    qmap = {"1":"24Q1","2":"24Q2","3":"24Q3","4":"24Q4",
            "5":"25Q1","6":"25Q2","7":"25Q3","8":"25Q4"}
    panel["quarter"] = panel["quarter"].map(qmap)

    print()
    print(f"Panel rows (24Q1-25Q4): {len(panel)}, assets={panel['asset'].nunique()}, dates={panel['date'].nunique()}")

    out_lines = []
    def echo(line=""):
        print(line); out_lines.append(line)

    # ---- §4c Distribution tables ---------------------------------------------
    echo()
    echo("## 4. Distribution Tables (REAL DATA from Binance)")
    echo()
    for thresh in [0.05, 0.10, 0.15, 0.25]:
        ev = panel[panel["ret_1d"] >= thresh].dropna(
            subset=["fwd_1d_from_t1","fwd_3d_from_t1","fwd_5d_from_t1","fwd_7d_from_t1"])
        echo(f"### Trigger: ret_1d >= +{thresh*100:.0f}% (n={len(ev)})")
        echo()
        echo("| Horizon | n | Mean | Median | Std | P10 | P25 | P75 | P90 | Pos % | Asymmetry |")
        echo("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for N, col in [(1,"fwd_1d_from_t1"),(3,"fwd_3d_from_t1"),(5,"fwd_5d_from_t1"),(7,"fwd_7d_from_t1")]:
            s = ev[col]
            asym = s.quantile(0.90) / abs(s.quantile(0.10)) if s.quantile(0.10) != 0 else float("nan")
            echo(f"| **{N}d (t+{N+1})** | {len(s)} | {s.mean():+.4%} | {s.median():+.4%} | {s.std():.4%} | "
                 f"{s.quantile(0.10):+.4%} | {s.quantile(0.25):+.4%} | {s.quantile(0.75):+.4%} | "
                 f"{s.quantile(0.90):+.4%} | {(s>0).mean()*100:.1f}% | {asym:.2f} |")
        echo()

    # ---- §4d Per-quarter regime breakdown (using +15%) -----------------------
    echo("## 5. Per-Quarter Regime Breakdown (Threshold = +15%)")
    echo()
    echo("| Quarter | Regime Dominance | n events | Mean Fwd 3d | Mean Fwd 5d | BTC 30d Midpoint |")
    echo("|---|---|---:|---:|---:|---:|")
    ev15 = panel[panel["ret_1d"] >= 0.15].dropna(subset=["fwd_3d_from_t1","fwd_5d_from_t1"])
    for q in ["24Q1","24Q2","24Q3","24Q4","25Q1","25Q2","25Q3","25Q4"]:
        sub = ev15[ev15["quarter"] == q]
        if len(sub):
            # quarter regime: most common regime label across events in that quarter
            reg = sub["regime"].value_counts().index[0] if len(sub["regime"]) else "?"
            btc_mid = sub["btc_30d"].median()
            echo(f"| **{q}** | {reg.capitalize()} | {len(sub)} | "
                 f"{sub['fwd_3d_from_t1'].mean():+.4%} | {sub['fwd_5d_from_t1'].mean():+.4%} | "
                 f"{btc_mid:+.2%} |")
        else:
            echo(f"| **{q}** | n/a | 0 | - | - | - |")
    echo()

    # ---- §4e Per-asset breakdown (top 15 by event count, threshold=+15%) -----
    echo("## 6. Per-Asset Breakdown (Top 15 by Count, Threshold = +15%)")
    echo()
    echo("| Asset | Bucket | n events | Mean Trigger | Mean Fwd 3d | Mean Fwd 5d |")
    echo("|---|---|---:|---:|---:|---:|")
    by_asset = (ev15.groupby("asset").agg(
        n=("ret_1d","size"),
        mean_trigger=("ret_1d","mean"),
        mean_fwd_3d=("fwd_3d_from_t1","mean"),
        mean_fwd_5d=("fwd_5d_from_t1","mean"),
    ).reset_index().sort_values("n", ascending=False).head(15))
    for _, r in by_asset.iterrows():
        b = bucket_of(r["asset"])
        echo(f"| **{r['asset']}** | {b} | {int(r['n'])} | "
             f"{r['mean_trigger']:+.4%} | {r['mean_fwd_3d']:+.4%} | {r['mean_fwd_5d']:+.4%} |")
    echo()

    # ---- §4f Per-bucket breakdown (threshold=+15%) ---------------------------
    echo("## 7. Per-Bucket Breakdown (Threshold = +15%)")
    echo()
    echo("| Bucket | n events | Mean Fwd 3d | Median Fwd 3d | Mean Fwd 5d |")
    echo("|---|---:|---:|---:|---:|")
    for b in ["BLUE","STEADY","DEGEN","VOLATILE"]:
        sub = ev15[ev15["bucket"] == b]
        if len(sub):
            echo(f"| **{b}** | {len(sub)} | {sub['fwd_3d_from_t1'].mean():+.4%} | "
                 f"{sub['fwd_3d_from_t1'].median():+.4%} | {sub['fwd_5d_from_t1'].mean():+.4%} |")
        else:
            echo(f"| **{b}** | 0 | - | - | - |")
    echo()

    # ---- §4g Naive strategy projection ---------------------------------------
    echo("## 8. Naive Strategy Projection (24bps RT, 4% NAV per entry)")
    echo()
    echo("| Threshold | Horizon | Trades | Gross sum | Net sum (post-cost) | Mean net / trade | NAV @4% (8Q) |")
    echo("|---|---:|---:|---:|---:|---:|---:|")
    COST_RT = 0.0024
    SIZE = 0.04
    for thresh in [0.05, 0.10, 0.15, 0.25]:
        ev = panel[panel["ret_1d"] >= thresh].dropna(
            subset=["fwd_1d_from_t1","fwd_3d_from_t1","fwd_5d_from_t1"])
        for N, col in [(1,"fwd_1d_from_t1"),(3,"fwd_3d_from_t1"),(5,"fwd_5d_from_t1")]:
            gross = ev[col].sum()
            net = gross - COST_RT * len(ev)
            nav_pct = SIZE * net * 100
            mean_net = net / len(ev) * 100 if len(ev) else 0
            echo(f"| +{thresh*100:.0f}% | {N}d | {len(ev)} | "
                 f"{gross*100:+.2f}% | {net*100:+.2f}% | "
                 f"{mean_net:+.4f}% | **{nav_pct:+.2f}%** |")
    echo()

    # ---- §4h Regime-gated projection (threshold=+15%, 3d) --------------------
    echo("## 9. Regime-Gated Projection (Threshold = +15%, 3d Horizon)")
    echo()
    echo("| Regime Gate | Trades | Gross sum | Net sum | NAV @4% (8Q) | Delta vs All-Events |")
    echo("|---|---:|---:|---:|---:|---:|")
    all_ev = panel[panel["ret_1d"] >= 0.15].dropna(subset=["fwd_3d_from_t1"])
    base_nav = SIZE * (all_ev["fwd_3d_from_t1"].sum() - COST_RT * len(all_ev)) * 100
    echo(f"| **All events** | {len(all_ev)} | "
         f"{all_ev['fwd_3d_from_t1'].sum()*100:+.2f}% | "
         f"{(all_ev['fwd_3d_from_t1'].sum() - COST_RT * len(all_ev))*100:+.2f}% | "
         f"{base_nav:+.2f}% | - |")
    for gate_name, gate in [("Bull only", {"bull"}),
                             ("Bull + Chop", {"bull","chop"}),
                             ("Chop only", {"chop"}),
                             ("Bear + Crash", {"bear","crash"}),
                             ("Bull + Chop - Crash", {"bull","chop","bear"})]:
        sub = all_ev[all_ev["regime"].isin(gate)]
        gross = sub["fwd_3d_from_t1"].sum()
        net = gross - COST_RT * len(sub)
        nav = SIZE * net * 100
        echo(f"| **{gate_name}** | {len(sub)} | {gross*100:+.2f}% | {net*100:+.2f}% | {nav:+.2f}% | {nav - base_nav:+.2f}% |")
    echo()

    # ---- §4i Sensitivity / stress -------------------------------------------
    echo("## 10. Sensitivity / Stress (Threshold = +25%, 5d horizon, Bull-only gate)")
    echo()
    ev25 = panel[(panel["ret_1d"] >= 0.25) & (panel["regime"] == "bull")].dropna(
        subset=["fwd_5d_from_t1"])
    echo(f"Bull-only +25% / 5d-hold base: n={len(ev25)} events")
    echo()
    echo("| Stress | Cost RT (bps) | Size | NAV @8Q | Delta |")
    echo("|---|---:|---:|---:|---:|")
    for cost_bps, size_pct, label in [
            (24,  0.04, "Baseline"),
            (50,  0.04, "Slippage (50bps)"),
            (100, 0.04, "Adverse (100bps)"),
            (24,  0.02, "Half size (2%)"),
            (24,  0.08, "Double size (8%)"),
        ]:
        c = cost_bps / 10000
        gross = ev25["fwd_5d_from_t1"].sum()
        net = gross - c * len(ev25)
        nav = size_pct * net * 100
        echo(f"| {label} | {cost_bps} | {size_pct:.0%} | {nav:+.2f}% | {nav - (0.04 * (gross - 0.0024*len(ev25)) * 100):+.2f}% |")
    echo()

    # Fill-rate stress
    echo("**Fill-rate stress** (assumes 25% of triggers actually fill at t+1):")
    fill_rate = 0.25
    gross = ev25["fwd_5d_from_t1"].sum() * fill_rate
    net = gross - 0.0024 * len(ev25) * fill_rate
    nav = 0.04 * net * 100
    echo(f"- 25% fill: effective n={int(len(ev25)*fill_rate)}, NAV @4% = {nav:+.2f}%")
    echo()

    # Bootstrap subsample
    echo("**Subsample stability** (bootstrap 1000x over 8 quarters of +25% trigger 5d-hold events):")
    rng = np.random.default_rng(42)
    all25 = panel[panel["ret_1d"] >= 0.25].dropna(subset=["fwd_5d_from_t1"])
    n = len(all25)
    boot_navs = []
    for _ in range(1000):
        sample = all25.iloc[rng.integers(0, n, size=n)]
        gross = sample["fwd_5d_from_t1"].sum()
        net = gross - 0.0024 * n
        boot_navs.append(0.04 * net * 100)
    boot_navs = np.array(boot_navs)
    echo(f"- n_events={n}, bootstrap NAV percentiles: P5={np.percentile(boot_navs,5):+.2f}%, "
         f"P50={np.percentile(boot_navs,50):+.2f}%, P95={np.percentile(boot_navs,95):+.2f}%")
    echo()

    echo("---")
    echo()
    echo("## Data Acquisition Notes (vs Gemini's synthetic claim)")
    echo(f"- Symbols requested: {len(SYMBOLS)}")
    echo(f"- Symbols pulled successfully: {len(frames)}")
    echo(f"- Symbols skipped: {[s for s,_ in skipped]}")
    echo(f"- Panel rows (24Q1-25Q4 only): {len(panel)}")
    echo(f"- Universe coverage: {panel['asset'].nunique()} unique assets, {panel['date'].nunique()} unique dates")
    echo()
    echo("Data source: api.binance.com/api/v3/klines (public, no auth). REAL measurement, NOT synthetic.")

    out_path = OUT_DIR / "ORACLE_EXERCISE_BINANCE_REAL_RESULTS_2026_05_18.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Also dump the panel for downstream verification
    panel_path = OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"
    panel.to_parquet(panel_path)
    print(f"Wrote panel to {panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
