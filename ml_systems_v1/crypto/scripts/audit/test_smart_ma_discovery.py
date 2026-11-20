"""Smart-discovery MA framework: Fibonacci + log-spaced + decorrelated candidates.

User insight (2026-05-20): SMA(28,29) brute-force winner has +0.91 signal correlation
with neighbors. The grid is finding ONE signal measured 20 ways, not 20 signals.
The "lead" is noise. Test semantically-meaningful + decorrelated candidates instead.

Compares:
  - Fibonacci pairs: (5,8), (5,13), (8,13), (8,21), (13,21), (13,34), (21,34), (21,55), (34,55), (34,89), (55,89)
  - Log-spaced (golden ratio): pairs where slow/fast ≈ 1.618 or 2.618
  - Decorrelated grid: starting from SMA(13,34), greedily add only signals with corr<0.6
  - Compare to brute-force winner SMA(28,29)
Under SAME multi-exit menu as V2 (incl. setup-toxicity).

================================================================================
UPPER_BOUND_NOT_DEPLOY_ESTIMATE -- READ THIS BEFORE CITING ANY NAV NUMBER
================================================================================
2026-05-20 oracle audit (per docs/ORACLE_CORRECTIONS_2026_05_20.md):

The `nav_4pct_upper_bound_arithmetic` field below is the ARITHMETIC SUM of
per-event PnL × 4% notional sizing. It assumes:
  - Every event entered at full 4% sizing
  - Zero capacity constraint (5,243 events / ~330 days = 16 entries/day = 64% NAV/d)
  - No concurrent-position cap; no capital lockup; no compound accounting

This is the SAME failure mode as the 5 simulators deprecated by commit f60365d
(improve_metrics_v2 etc.) -- just expressed as notional-sizing-without-cap rather
than forward-return-as-K-selector. Both are upper bounds, not deploy estimates.

WHAT TO USE INSTEAD:
  - Per-event mean / median / hit-rate / Sharpe = HONEST per-event characterization
  - For portfolio NAV: feed the per-event ledger through `honest_v2_simulator.py`
    or v3 paper_trade_replay (canonical truth, yaml inflates 1.7-7.6x)
  - RELATIVE comparisons (setup-toxic > 7d-hold, Fibonacci > brute-force) ARE valid
    when both arms share the same notional aggregator -- but the absolute NAV is not

DO NOT cite "+1,311% NAV @ 4%" or similar without explicit `arithmetic upper bound`
qualification. The realistic deploy estimate for any 17-setup ensemble is the
honest_v2 OOS signal-K result (+33% / 10mo / annualized +40%; commit 7df4b7b).
================================================================================
"""
from __future__ import annotations
import json, numpy as np, pandas as pd, polars as pl
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
COST = 0.0024
SIZE = 0.04

FIB_PERIODS = [5, 8, 13, 21, 34, 55, 89]
FIB_PAIRS = [(f, s) for f in FIB_PERIODS for s in FIB_PERIODS if f < s]
# Log-spaced golden-ratio family
GOLDEN_PAIRS = []
for f in range(5, 60):
    for ratio in (1.618, 2.0, 2.618, 3.236):
        s = int(round(f * ratio))
        if 5 <= s <= 100 and f < s:
            GOLDEN_PAIRS.append((f, s))
GOLDEN_PAIRS = sorted(set(GOLDEN_PAIRS))
# Brute-force "winners" from the existing CSV (for comparison)
BF_PAIRS = [(28, 29), (27, 30), (25, 26), (13, 34), (8, 21), (5, 8), (5, 13), (8, 13), (10, 30), (20, 50)]


def compute_ma_flip_day(chimera_idx, asset, ev_date, fast, slow, ma_type="SMA", max_fwd=14):
    sub = chimera_idx.get(asset)
    if sub is None: return None
    row = sub[sub["date"] == ev_date]
    if row.empty: return None
    idx = row.index[0]
    closes = sub["close"].values
    start = max(0, idx - max(slow, 50))
    seg = closes[start:idx + max_fwd + 1]
    if ma_type == "SMA":
        fa = pd.Series(seg).rolling(fast).mean().values
        sl = pd.Series(seg).rolling(slow).mean().values
    else:
        fa = pd.Series(seg).ewm(span=fast, adjust=False).mean().values
        sl = pd.Series(seg).ewm(span=slow, adjust=False).mean().values
    entry_pos = idx - start
    for k in range(1, min(max_fwd, len(seg) - entry_pos)):
        if entry_pos + k < len(seg):
            if pd.notna(fa[entry_pos + k]) and pd.notna(sl[entry_pos + k]) and fa[entry_pos + k] <= sl[entry_pos + k]:
                return k
    return None


def load_chimera_with_atr():
    panel_rows = []
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        df = pl.read_parquet(f, columns=["timestamp","high","low","close"]).to_pandas()
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df["asset"] = sym
        df["high_low"] = df["high"] - df["low"]
        df["high_pc"] = (df["high"] - df["close"].shift(1)).abs()
        df["low_pc"] = (df["low"] - df["close"].shift(1)).abs()
        df["tr"] = df[["high_low","high_pc","low_pc"]].max(axis=1)
        df["atr30_pct"] = df["tr"].rolling(30).mean() / df["close"]
        df["atr50_pct"] = df["tr"].rolling(50).mean() / df["close"]
        panel_rows.append(df[["asset","date","close","atr30_pct","atr50_pct"]])
    return pd.concat(panel_rows, ignore_index=True)


def apply_exits(entry_close, fwd_closes, atrs, ma_flip_day):
    results = {}
    if not entry_close or entry_close <= 0: return results
    def pct(c): return c / entry_close - 1 - COST
    if len(fwd_closes) >= 7 and fwd_closes[6] is not None: results["D_7d"] = pct(fwd_closes[6])
    if ma_flip_day is not None and ma_flip_day <= 14 and ma_flip_day < len(fwd_closes) and fwd_closes[ma_flip_day] is not None:
        results["S_setup_toxic"] = pct(fwd_closes[ma_flip_day])
    elif len(fwd_closes) >= 14 and fwd_closes[13] is not None:
        results["S_setup_toxic"] = pct(fwd_closes[13])
    # Trail 5%/3%
    peak = entry_close; exit_g = None; armed = False
    for c in fwd_closes[:14]:
        if c is None: continue
        if c > peak: peak = c
        ret = c/entry_close - 1
        if not armed and ret >= 0.05: armed = True
        if armed and c <= peak * 0.97: exit_g = c; break
    if exit_g is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None: exit_g = fwd_closes[13]
    if exit_g is not None: results["G_trail_5_3"] = pct(exit_g)
    # ATR(50) K=4 + TP30 (best ATR variant from V2)
    atr = atrs.get("atr50")
    if atr and atr > 0:
        exit_f = None
        for c in fwd_closes[:14]:
            if c is None: continue
            ret = c/entry_close - 1
            if ret <= -4*atr: exit_f = c; break
            if ret >= 0.30: exit_f = c; break
        if exit_f is None and len(fwd_closes) >= 14 and fwd_closes[13] is not None: exit_f = fwd_closes[13]
        if exit_f is not None: results["F_atr50_K4_TP30"] = pct(exit_f)
    return results


def fwd_path(chimera_idx, asset, ev_date, n=15):
    sub = chimera_idx.get(asset)
    if sub is None: return None
    row = sub[sub["date"] == ev_date]
    if row.empty: return None
    idx = row.index[0]
    if idx + 1 >= len(sub): return None
    entry_close = float(sub.iloc[idx]["close"])
    atrs = {"atr30": float(sub.iloc[idx].get("atr30_pct", np.nan)) if pd.notna(sub.iloc[idx].get("atr30_pct", np.nan)) else None,
            "atr50": float(sub.iloc[idx].get("atr50_pct", np.nan)) if pd.notna(sub.iloc[idx].get("atr50_pct", np.nan)) else None}
    fwd = []
    for k in range(1, n+1):
        if idx + k < len(sub):
            fwd.append(float(sub.iloc[idx + k]["close"]))
        else:
            fwd.append(None)
    return entry_close, atrs, fwd


def evaluate_pair_long_events(events_pd, chimera_idx, ma_type, fast, slow):
    fast_col, slow_col = f"{ma_type}_{fast}", f"{ma_type}_{slow}"
    if fast_col not in events_pd.columns or slow_col not in events_pd.columns:
        return None
    ev = events_pd[events_pd[fast_col].notna() & events_pd[slow_col].notna()].copy()
    ev = ev[ev[fast_col] > ev[slow_col]]
    if len(ev) < 100:
        return None
    strat_pnls = {}
    for _, row in ev.iterrows():
        asset = row["asset"]
        ev_date = pd.to_datetime(row["date"]).date() if not isinstance(row["date"], date) else row["date"]
        path = fwd_path(chimera_idx, asset, ev_date)
        if path is None: continue
        entry_close, atrs, fwd = path
        ma_flip = compute_ma_flip_day(chimera_idx, asset, ev_date, fast, slow, ma_type)
        exits = apply_exits(entry_close, fwd, atrs, ma_flip)
        for k, v in exits.items():
            strat_pnls.setdefault(k, []).append(v)
    summary = {}
    for s, pnls in strat_pnls.items():
        arr = np.array(pnls)
        # Per-event stats are HONEST. nav_4pct_upper_bound_arithmetic is NOT a
        # deploy estimate -- see top-of-file UPPER_BOUND_NOT_DEPLOY_ESTIMATE notice.
        summary[s] = {"n": len(arr), "mean_pct": arr.mean()*100, "hit_rate": (arr>0).mean()*100,
                       "sharpe": arr.mean()/(arr.std()+1e-9),
                       "nav_4pct_upper_bound_arithmetic": arr.sum()*SIZE*100}
    return summary


def signal_correlation(events_pd, ma, f1, s1, f2, s2):
    if f"{ma}_{f1}" not in events_pd.columns: return None
    sig1 = (events_pd[f"{ma}_{f1}"] > events_pd[f"{ma}_{s1}"]).astype(int) - (events_pd[f"{ma}_{f1}"] < events_pd[f"{ma}_{s1}"]).astype(int)
    sig2 = (events_pd[f"{ma}_{f2}"] > events_pd[f"{ma}_{s2}"]).astype(int) - (events_pd[f"{ma}_{f2}"] < events_pd[f"{ma}_{s2}"]).astype(int)
    return np.corrcoef(sig1, sig2)[0,1]


def main():
    print("Loading panels...")
    chimera = load_chimera_with_atr()
    chimera_idx = {a: sub.sort_values("date").reset_index(drop=True) for a, sub in chimera.groupby("asset")}
    events = pl.read_parquet(ROOT/"runs/oracle_layer3/ma_ema_permutation/event_ma_snapshot.parquet").to_pandas()
    events = events[events["side"] == "long"]

    # === Test FIBONACCI candidates ===
    print(f"\n=== FIBONACCI pairs (n={len(FIB_PAIRS)} candidates, semantically meaningful) ===")
    results = {}
    for f, s in FIB_PAIRS:
        sum_ = evaluate_pair_long_events(events, chimera_idx, "SMA", f, s)
        if sum_ is None: continue
        results[f"SMA({f},{s})"] = sum_
    # LOUD upper-bound warning per oracle-corrections (2026-05-20)
    print("\n  [WARN] NAV columns below are ARITHMETIC UPPER BOUNDS at notional 4% sizing.")
    print("         They are NOT deploy estimates. For deploy: use honest_v2_simulator.py")
    print("         or v3 paper_trade_replay. RELATIVE rank is valid; absolute % is not.")

    # Print top 5 by setup-toxic NAV
    ranked = [(k, v["S_setup_toxic"]["nav_4pct_upper_bound_arithmetic"], v["S_setup_toxic"]["mean_pct"], v["S_setup_toxic"]["hit_rate"], v["S_setup_toxic"]["n"])
              for k, v in results.items() if "S_setup_toxic" in v]
    ranked.sort(key=lambda x: -x[1])
    print(f'  {"pair":<16}{"NAV_UB_arith":>18}{"mean":>10}{"hit%":>8}{"n":>8}')
    for r in ranked[:10]:
        print(f'  {r[0]:<16}{r[1]:>+17.2f}%{r[2]:>+9.3f}%{r[3]:>7.1f}%{r[4]:>8d}')

    # === Test brute-force "winners" + golden-ratio + decorrelated ===
    print(f"\n=== BRUTE-FORCE WINNERS (high signal-correlation with each other) ===")
    bf_results = {}
    for f, s in BF_PAIRS:
        sum_ = evaluate_pair_long_events(events, chimera_idx, "SMA", f, s)
        if sum_ is None: continue
        bf_results[f"SMA({f},{s})"] = sum_
    ranked_bf = [(k, v["S_setup_toxic"]["nav_4pct_upper_bound_arithmetic"], v["S_setup_toxic"]["mean_pct"], v["S_setup_toxic"]["hit_rate"], v["S_setup_toxic"]["n"])
              for k, v in bf_results.items() if "S_setup_toxic" in v]
    ranked_bf.sort(key=lambda x: -x[1])
    print(f'  {"pair":<16}{"NAV_UB_arith":>18}{"mean":>10}{"hit%":>8}{"n":>8}')
    for r in ranked_bf[:10]:
        print(f'  {r[0]:<16}{r[1]:>+17.2f}%{r[2]:>+9.3f}%{r[3]:>7.1f}%{r[4]:>8d}')

    # === Signal-correlation matrix among Fibonacci candidates ===
    print(f"\n=== Signal correlation matrix (Fibonacci pairs) — independence diagnostic ===")
    selected = [(13,34), (21,34), (8,21), (5,13), (34,55), (21,55), (8,13)]
    print(f'  {"":>10}', end="")
    for f1, s1 in selected:
        print(f'  ({f1},{s1})', end="")
    print()
    for f1, s1 in selected:
        print(f'  ({f1:>2},{s1:>2})', end="  ")
        for f2, s2 in selected:
            c = signal_correlation(events, "SMA", f1, s1, f2, s2)
            print(f'  {c:>+.3f}', end="")
        print()

    out = {"fibonacci": results, "brute_force": bf_results}
    (OUT_DIR / "smart_ma_discovery.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_DIR}/smart_ma_discovery.json")

if __name__ == "__main__":
    main()
