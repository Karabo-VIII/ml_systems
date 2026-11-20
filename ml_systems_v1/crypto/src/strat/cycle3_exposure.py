"""src/strat/cycle3_exposure.py -- META-FOLD CYCLE 3: exposure rules for mom14-K5-r3 engine.

Tests four rules targeting the 2025 chop bleed vs BTC-SMA200 gate:
  (a) base: BTC-SMA200 gate (forces cash when BTC below its SMA200)
  (b) dual: BTC-SMA200 AND breadth-scale (exposure *= fraction_above_SMA50)
  (c) vol-target: BTC-SMA200 gate + portfolio-level vol-target overlay (sigma_t -> target/sigma)
  (d) dist: distance-from-SMA partial de-risk (scale by BTC price/SMA200 clip to [0,1])

Reports per-year compound 2020-2025 + full-cycle + maxDD + avg_expo.
Honest: no in-sample tuning of parameters beyond what is described.

RWYB: python -m strat.cycle3_exposure
No emoji (cp1252).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml

COST = ml.COST
YEARS = list(range(2020, 2026))


# ---- primitives from cycle2_referee (re-derived inline) ----

def book_returns(W, ind):
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    return bret, pos


def year_comp(bret, y):
    s = pd.Timestamp(f"{y}-01-01"); e = pd.Timestamp(f"{y+1}-01-01")
    m = (bret.index >= s) & (bret.index < e)
    x = bret[m].to_numpy()
    return round((np.prod(1 + x) - 1) * 100, 1) if m.sum() > 2 else None


def maxdd_full(bret):
    x = bret.to_numpy()
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100), 1)


def full_comp(bret):
    x = bret.to_numpy()
    return round((np.prod(1 + x) - 1) * 100, 1)


def summarize(W, ind, name):
    bret, pos = book_returns(W, ind)
    row = {"name": name}
    for y in YEARS:
        row[str(y)] = year_comp(bret, y)
    row["FULL"] = full_comp(bret)
    row["maxDD"] = maxdd_full(bret)
    row["avg_expo"] = round(float(pos.sum(axis=1).mean()), 2)
    return row, bret


# ---- base engine: mom14-K5-r3 with BTC-SMA200 market gate ----

def make_base_weights(ind, K=5, rebal=3):
    """Identical to ml.topk_weight but uses ind['gate'] which already encodes per-asset SMA200."""
    return ml.topk_weight(ind["mom14"], ind, K=K, gate=True, rebal=rebal)


# ---- Rule (a): base BTC-market gate -- when BTC < SMA200, whole book to cash ----
# The gate in mover_lab is per-asset (C > SMA200). For the BTC-market gate we override:
# when BTC itself is below its SMA200, ALL positions are zeroed.

def apply_btc_market_gate(W, ind):
    """Zero all positions when BTC is below its own SMA200 (daily)."""
    btc_col = "BTC" if "BTC" in ind["C"].columns else (
        [c for c in ind["C"].columns if "BTC" in c.upper()][0]
        if any("BTC" in c.upper() for c in ind["C"].columns) else None
    )
    if btc_col is None:
        print("[WARN] No BTC column found -- base gate is pass-through")
        return W.copy()
    btc_in_trend = (ind["C"][btc_col] > ind["sma200"][btc_col]).reindex(W.index).fillna(False)
    Wg = W.copy()
    Wg.loc[~btc_in_trend, :] = 0.0
    return Wg


# ---- Rule (b): DUAL gate: BTC-SMA200 AND breadth-scale ----
# When BTC > SMA200, scale exposure by fraction of universe above their own SMA50.
# When BTC < SMA200, 0 (same as base gate).

def make_dual_gate(W_ungated_base, ind):
    """
    W_ungated_base: raw topk weights (full EW to K picks, no BTC gate applied yet).
    Returns: breadth-scaled weights, then zeroed if BTC < SMA200.
    """
    # Breadth = fraction of assets above SMA50 on each day
    above_sma50 = (ind["C"] > ind["sma50"]).fillna(False)
    breadth = above_sma50.mean(axis=1)  # [0..1]

    # scale rows by breadth
    W_scaled = W_ungated_base.multiply(breadth, axis=0)

    # then apply BTC market gate
    W_gated = apply_btc_market_gate(W_scaled, ind)
    return W_gated


# ---- Rule (c): vol-target overlay on top of BTC gate ----
# After BTC gate (book goes to cash when bear), apply a vol-target on the remaining equity.
# Target: annualised vol = 0.25 (25%). Scale = min(1, target / realized_vol_20d).

def make_vol_target(W_base_gated, ind, target_vol=0.25, lookback=20):
    """
    Compute realised portfolio vol over lookback days, then scale position so that
    expected vol = target_vol. Cap scale at 1.0 (no leverage).
    """
    R = ind["R"].reindex(index=W_base_gated.index, columns=W_base_gated.columns).fillna(0.0)
    pos = W_base_gated.shift(1).fillna(0.0)
    port_ret = (pos * R).sum(axis=1)

    # Realised vol (daily std -> annualise)
    real_vol = port_ret.rolling(lookback, min_periods=5).std() * np.sqrt(365)
    real_vol = real_vol.fillna(method="bfill").fillna(target_vol)

    scale = (target_vol / real_vol).clip(upper=1.0)

    W_vt = W_base_gated.multiply(scale, axis=0)
    return W_vt


# ---- Rule (d): distance-from-SMA partial de-risk ----
# When BTC is above SMA200, scale exposure by a clipped distance signal:
#   dist_ratio = BTC / SMA200 (1.0 = at par; 1.2 = 20% above)
#   If dist_ratio > 1.1 (far above) -> full exposure (stable bull)
#   If dist_ratio in [1.0, 1.1] -> ramp from 0.5 to 1.0 (chop zone just above)
#   If dist_ratio < 1.0 -> 0 (below SMA200, same as base gate)
# This cuts exposure during the choppy "BTC hovering just above SMA200" regime.

def make_dist_scale(W_raw, ind):
    """
    Partial de-risk based on BTC's distance from its own SMA200.
    """
    btc_col = "BTC" if "BTC" in ind["C"].columns else (
        [c for c in ind["C"].columns if "BTC" in c.upper()][0]
        if any("BTC" in c.upper() for c in ind["C"].columns) else None
    )
    if btc_col is None:
        return W_raw.copy()

    btc_c = ind["C"][btc_col].reindex(W_raw.index)
    btc_sma200 = ind["sma200"][btc_col].reindex(W_raw.index)

    # dist_ratio: how far above (or below) SMA200
    dist_ratio = (btc_c / btc_sma200.replace(0, np.nan)).fillna(1.0)

    # scale: 0 below 1.0; ramp 0.5->1.0 in [1.0, 1.1]; 1.0 above 1.1
    def dist_to_scale(r):
        if r < 1.0:
            return 0.0
        elif r < 1.1:
            # linear ramp from 0.5 to 1.0
            return 0.5 + 0.5 * (r - 1.0) / 0.1
        else:
            return 1.0

    scale = dist_ratio.apply(dist_to_scale)

    W_scaled = W_raw.multiply(scale, axis=0)
    return W_scaled


# ---- formatting ----

def fmt_pct(v):
    if v is None:
        return "   N/A  "
    return f"{v:+8.1f}%"


def print_table(rows):
    hdrs = ["Rule"] + [str(y) for y in YEARS] + ["FULL", "maxDD", "expo"]
    col_w = [24] + [9] * len(YEARS) + [10, 9, 6]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr_line = "|" + "|".join(f"{h:^{w}}" for h, w in zip(hdrs, col_w)) + "|"
    print(sep)
    print(hdr_line)
    print(sep)
    for r in rows:
        cols = [r["name"]]
        for y in YEARS:
            cols.append(fmt_pct(r.get(str(y))))
        cols.append(fmt_pct(r.get("FULL")))
        cols.append(fmt_pct(r.get("maxDD")))
        cols.append(f"{r.get('avg_expo', 0.0):5.2f}")
        print("|" + "|".join(f"{c:^{w}}" for c, w in zip(cols, col_w)) + "|")
    print(sep)


def main():
    print("[cycle3] loading mover_lab 2020-01..2026-05 ...")
    ind = ml.load("2020-01-01", "2026-06-01")

    assets = list(ind["C"].columns)
    print(f"[cycle3] {len(assets)} assets: {assets}")

    # ---- raw topk weights (K=5, rebal=3, per-asset SMA200 gate already baked in via ind['gate']) ----
    W_raw = ml.topk_weight(ind["mom14"], ind, K=5, gate=True, rebal=3)

    # ---- (a) base = raw + BTC market-level gate ----
    W_a = apply_btc_market_gate(W_raw, ind)

    # ---- (b) dual = breadth-scale + BTC gate ----
    # Start from W_raw (per-asset gate already in), then breadth scale, then BTC gate
    W_b = make_dual_gate(W_raw, ind)

    # ---- (c) vol-target on top of base (a) ----
    W_c = make_vol_target(W_a, ind, target_vol=0.25, lookback=20)

    # ---- (d) dist-from-SMA scale (replaces BTC-binary gate with a ramp) ----
    W_d = make_dist_scale(W_raw, ind)

    # ---- also include the pure per-asset gated version (no BTC market gate) for reference ----
    W_ref_nobtcgate = W_raw.copy()

    # ---- BTC buy-and-hold for context ----
    btc_col = "BTC" if "BTC" in ind["C"].columns else (
        [c for c in ind["C"].columns if "BTC" in c.upper()][0]
        if any("BTC" in c.upper() for c in ind["C"].columns) else None
    )

    rows = []
    brets = {}

    configs = [
        ("(ref) no-BTC-mktgate", W_ref_nobtcgate),
        ("(a) BTC-SMA200 gate",  W_a),
        ("(b) dual: gate+breadth", W_b),
        ("(c) gate+vol-target25", W_c),
        ("(d) dist-from-SMA ramp", W_d),
    ]

    for name, W in configs:
        row, bret = summarize(W, ind, name)
        rows.append(row)
        brets[name] = bret
        print(f"  computed: {name}")

    # ---- EW buy-and-hold for context ----
    W_bh = pd.DataFrame(1.0 / len(assets), index=ind["C"].index, columns=ind["C"].columns)
    row_bh, bret_bh = summarize(W_bh, ind, "(bh) EW-buy-hold")
    rows.append(row_bh)
    brets["(bh) EW-buy-hold"] = bret_bh

    print("\n" + "=" * 80)
    print("CYCLE 3: EXPOSURE RULES -- mom14-K5-r3 ENGINE (2020-2025, taker, long-only)")
    print("=" * 80)
    print_table(rows)

    # ---- delta vs base (a) ----
    print("\n--- DELTA vs (a) BTC-SMA200 gate (pp difference per year) ---")
    base_row = rows[1]  # (a)
    delta_hdrs = ["Rule"] + [str(y) for y in YEARS] + ["FULL", "maxDD"]
    col_w2 = [24] + [9] * len(YEARS) + [10, 9]
    sep2 = "+" + "+".join("-" * w for w in col_w2) + "+"
    print(sep2)
    print("|" + "|".join(f"{h:^{w}}" for h, w in zip(delta_hdrs, col_w2)) + "|")
    print(sep2)
    for r in rows:
        if r["name"] == base_row["name"]:
            continue
        delta_cols = [r["name"]]
        for y in YEARS:
            bv = base_row.get(str(y))
            rv = r.get(str(y))
            if bv is None or rv is None:
                delta_cols.append("   N/A  ")
            else:
                d = rv - bv
                delta_cols.append(f"{d:+8.1f} ")
        bv_f = base_row.get("FULL"); rv_f = r.get("FULL")
        delta_cols.append(fmt_pct((rv_f - bv_f) if bv_f is not None and rv_f is not None else None))
        bv_d = base_row.get("maxDD"); rv_d = r.get("maxDD")
        # positive delta on maxDD = WORSE (more negative); show as-is
        delta_cols.append(fmt_pct((rv_d - bv_d) if bv_d is not None and rv_d is not None else None))
        print("|" + "|".join(f"{c:^{w}}" for c, w in zip(delta_cols, col_w2)) + "|")
    print(sep2)

    # ---- 2025-specific deep-dive (chop problem) ----
    print("\n--- 2025 QUARTERLY BREAKDOWN (the chop problem) ---")
    quarters = [
        ("2025 Q1", "2025-01-01", "2025-04-01"),
        ("2025 Q2", "2025-04-01", "2025-07-01"),
    ]
    q_names = [q[0] for q in quarters]
    col_w3 = [24] + [12] * len(quarters)
    sep3 = "+" + "+".join("-" * w for w in col_w3) + "+"
    print(sep3)
    print("|" + "|".join(f"{h:^{w}}" for h, w in zip(["Rule"] + q_names, col_w3)) + "|")
    print(sep3)
    for name, bret in brets.items():
        qcols = [name]
        for _, qs, qe in quarters:
            m = (bret.index >= pd.Timestamp(qs)) & (bret.index < pd.Timestamp(qe))
            x = bret[m].to_numpy()
            if m.sum() > 2:
                c = round((np.prod(1 + x) - 1) * 100, 1)
                qcols.append(f"{c:+8.1f}%  ")
            else:
                qcols.append("   N/A      ")
        print("|" + "|".join(f"{c:^{w}}" for c, w in zip(qcols, col_w3)) + "|")
    print(sep3)

    # ---- verdict ----
    print("\n=== VERDICT ===")
    # auto-derive verdict based on numbers
    base = rows[1]
    base_2022 = base.get("2022")
    base_2025 = base.get("2025")
    base_full = base.get("FULL")
    base_dd = base.get("maxDD")

    findings = []
    for r in rows[2:5]:  # (b), (c), (d)
        y25 = r.get("2025"); y22 = r.get("2022"); full = r.get("FULL"); dd = r.get("maxDD")
        improves_2025 = (y25 is not None and base_2025 is not None and y25 > base_2025 + 0.5)
        holds_2022 = (y22 is not None and base_2022 is not None and abs(y22 - base_2022) < 3.0)
        net_positive = (full is not None and base_full is not None and full >= base_full - 200)
        dd_ok = (dd is not None and base_dd is not None and dd <= base_dd + 5.0)
        verdict = "IMPROVE" if (improves_2025 and holds_2022) else "NO-IMPROVE"
        findings.append((r["name"], verdict, y25, base_2025, y22, base_2022, full, base_full, dd, base_dd))
        print(f"  {r['name']}")
        print(f"    2025: {fmt_pct(y25)} vs base {fmt_pct(base_2025)}  => {'BETTER' if improves_2025 else 'NOT-BETTER'}")
        print(f"    2022: {fmt_pct(y22)} vs base {fmt_pct(base_2022)}  => {'HOLDS' if holds_2022 else 'REGRESSES'}")
        print(f"    FULL: {fmt_pct(full)} vs base {fmt_pct(base_full)}  maxDD: {fmt_pct(dd)} vs {fmt_pct(base_dd)}")
        print(f"    => {verdict}")
        print()

    any_winner = any(f[1] == "IMPROVE" for f in findings)
    if any_winner:
        winners = [f[0] for f in findings if f[1] == "IMPROVE"]
        print(f"ANSWER: YES -- rule(s) that improve 2025 without regressing 2022: {winners}")
    else:
        print("ANSWER: NO -- no tested rule improves 2025 AND holds 2022. The 2025 chop bleed is")
        print("  structurally hard to fix with a passive exposure overlay on this engine.")
        print("  The BTC-SMA200 gate is already near-optimal for the binary bull/bear distinction;")
        print("  the 2025 drawdown arises from a chop regime that exposure scaling does not resolve.")

    # ---- save JSON ----
    out_path = Path(__file__).resolve().parents[2] / "runs" / "strat" / "cycle3_exposure_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "rows": rows,
        "findings": [{"name": f[0], "verdict": f[1], "2025": f[2], "base_2025": f[3],
                      "2022": f[4], "base_2022": f[5], "full": f[6], "base_full": f[7],
                      "maxDD": f[8], "base_maxDD": f[9]} for f in findings],
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[cycle3] results saved to {out_path}")


if __name__ == "__main__":
    main()
