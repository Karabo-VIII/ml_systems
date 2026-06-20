"""src/strat/v51_exo_sweep.py -- FULL v51 EXOGENOUS MOVE-CATCH SWEEP.

ALL ~20 v51 exo features x BOTH directions (top/bottom tercile trigger) x exit {time, trail5}
x by_regime=True x block=True. DEV-walled. Holm/BH-correct across full sweep.
Objective: which exo families break the de-risked-beta wall (bull-only)?
    WALL-BREAKER criteria:
      (A) BEAR-POSITIVE: bear block_p_le0 < 0.05
      (B) REGIME-CONDITIONAL: bear edge != bull edge (NOT just beta continuation)
Ranked by BEAR realized-net (not bull, not overall). No emoji. RWYB.

Usage: python -m strat.v51_exo_sweep [--n 50] [--n_null 300] [--out <path>]
"""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.v51_feature_lab as fl
import strat.capture_lab as cl

# ---- causality manifest: features we trust as point-in-time vs suspect ----
CAUSALITY = {
    "norm_funding":         "SAFE",  # trailing funding rate (exchange-published hourly; last-of-day)
    "s3_global_lsr_z":      "SAFE",  # z-score of L/S ratio across session (trailing 30d z)
    "s3_smart_vs_retail_z": "SAFE",  # same: cross-sectional z of smart-vs-retail LSR spread
    "bs_basis_z30":         "SAFE",  # perp-spot basis z-score (30d trailing)
    "bs_basis_xsec_z":      "SAFE",  # cross-sectional basis z (trailing)
    "liq_total_usd":        "SAFE",  # sum of forced liq USD in the bar day (known at eod)
    "liq_delta_z30":        "SAFE",  # liq imbalance z (30d trailing)
    "liq_capitulation":     "SAFE",  # max of daily liq-spike flag (day known at eod)
    "liq_short_panic":      "SAFE",  # max of short-liq-spike flag
    "wh_whale_net_usd":     "SAFE",  # sum of whale net flow (exchange data, trailing)
    "norm_whale":           "SAFE",  # trailing z of whale activity
    "stbl_total_zscore_30d":"SAFE",  # stablecoin supply z (30d trailing)
    "stbl_compound_shock":  "SAFE",  # max stablecoin compound-shock flag in day
    "norm_vpin":            "SAFE",  # VPIN (volume-synchronized PIN) -- trailing window
    "norm_kyle_lambda":     "SAFE",  # Kyle lambda (price-impact proxy) -- trailing
    "norm_hawkes_imbalance":"SAFE",  # Hawkes-process imbalance -- trailing window
    "te_imb":               "SAFE",  # transfer-entropy imbalance -- trailing window
    "norm_oi_change":       "SAFE",  # OI change z -- trailing
    "mv_days_since_listed_binance": "SAFE",  # listing age -- deterministic, no look-ahead
    "xd_momentum_rank":     "SAFE",  # cross-day momentum rank -- trailing
    # price-derived controls (for comparison)
    "mom14":                "SAFE",
    "brk14":                "SAFE",
}

V51_FEATURES = list(fl.V51.keys())  # the 20 exo features
PRICE_CONTROLS = ["mom14", "brk14"]  # for comparison

EXITS = ["time", "trail5"]


def holm_correct(p_arr):
    """Holm step-down correction. Returns adjusted p-values (same order as input)."""
    n = len(p_arr)
    order = np.argsort(p_arr)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = min(1.0, p_arr[idx] * (n - rank))
        running_max = max(running_max, corrected)
        adj[idx] = running_max
    return adj


def sweep(n=50, n_null=300, out_path=None):
    t0 = time.time()
    print(f"[sweep] Loading v51 daily lab (n={n}) ...")
    lab = fl.load_v51_daily(n=n)
    C = lab["C"]
    print(f"  {len(lab['syms'])} assets; range {C.index.min().date()} -> {C.index.max().date()}")

    # determine which features have enough data
    avail = [c for c in V51_FEATURES if c in lab["F"] and lab["F"][c].notna().sum().sum() > 500]
    print(f"  v51 features with data: {len(avail)} / {len(V51_FEATURES)}: {avail}")

    # build full sweep: feature x direction x exit
    # direction "top" = top cross-sectional tercile (default in fired_matrix for chimera/other)
    # direction "bot" = bottom tercile (contrarian: liq spike, extreme funding, etc.)
    tasks = []
    for feat in avail + PRICE_CONTROLS:
        for direction in ("top", "bot"):
            for exit_kind in EXITS:
                tasks.append((feat, direction, exit_kind))

    print(f"  Total sweep tasks: {len(tasks)}")
    results = []

    for i, (feat, direction, exit_kind) in enumerate(tasks):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{len(tasks)} ({elapsed:.0f}s)")

        # CAUSALITY AUDIT: skip suspects (none expected, but enforce)
        causal = CAUSALITY.get(feat, "UNKNOWN")
        if causal != "SAFE":
            results.append({"feat": feat, "direction": direction, "exit": exit_kind,
                            "causality": causal, "note": "SKIPPED: causality suspect"})
            continue

        # Build a modified lab with the feature in the right direction
        lab2 = dict(lab)
        F2 = dict(lab["F"])
        if direction == "bot":
            # Invert the feature so that bottom-tercile entries are triggered by "high" value of negated feature
            X = lab["F"].get(feat)
            if X is None:
                results.append({"feat": feat, "direction": direction, "exit": exit_kind,
                                "note": "feature missing"}); continue
            F2[feat + "_bot"] = -X
            lab2["F"] = F2
            ti_key = feat + "_bot"
        else:
            ti_key = feat

        try:
            r = cl.evaluate_ti(lab2, ti_key, tf="1d", exit_kind=exit_kind, n_null=n_null,
                               by_regime=True, block=True, seed=42)
        except Exception as ex:
            results.append({"feat": feat, "direction": direction, "exit": exit_kind,
                            "note": f"ERROR: {ex}"}); continue

        if "note" in r and "insufficient" in str(r.get("note", "")):
            results.append({"feat": feat, "direction": direction, "exit": exit_kind,
                            "note": "insufficient signals (<30)"}); continue

        row = {
            "feat": feat, "direction": direction, "exit": exit_kind,
            "causality": causal,
            "n_fired": r.get("n_fired"),
            "mean_MFE_pct": r.get("mean_MFE_fired"),
            "mean_realized_net_pct": r.get("mean_realized_net"),
            "edge_vs_random_pp": r.get("edge_vs_random_pp"),
            "p_vs_random": r.get("p_vs_random"),
            "capture_rate": r.get("capture_rate"),
        }
        # block stats
        blk = r.get("block", {})
        row["block_p05_pp"] = blk.get("block_p05_pp")
        row["block_p_le0"] = blk.get("block_p_le0")

        # by_regime: extract bull, chop, bear
        for rg in ("bull", "chop", "bear"):
            d = (r.get("by_regime") or {}).get(rg)
            if d:
                row[f"{rg}_n"] = d["n"]
                row[f"{rg}_realized_pct"] = d["realized_net"]
                row[f"{rg}_edge_pp"] = d["edge_pp"]
                row[f"{rg}_p_rnd"] = d["p_vs_random"]
                blk_r = d.get("block", {})
                row[f"{rg}_block_p05"] = blk_r.get("block_p05_pp")
                row[f"{rg}_block_ple0"] = blk_r.get("block_p_le0")
            else:
                for k in ["n", "realized_pct", "edge_pp", "p_rnd", "block_p05", "block_ple0"]:
                    row[f"{rg}_{k}"] = None

        results.append(row)

    df = pd.DataFrame(results)

    # Holm/BH correction across the full sweep (on bear block_p_le0, the decisive p)
    has_bear_p = df["bear_block_ple0"].notna()
    if has_bear_p.sum() > 0:
        raw_ps = df.loc[has_bear_p, "bear_block_ple0"].values.astype(float)
        adj = holm_correct(raw_ps)
        df.loc[has_bear_p, "bear_block_ple0_holm"] = adj
    else:
        df["bear_block_ple0_holm"] = np.nan

    # BH correction as well
    if has_bear_p.sum() > 0:
        order = np.argsort(raw_ps)
        m = len(raw_ps)
        bh_adj = np.empty(m)
        for rank_i, idx in enumerate(np.argsort(order)):  # rank_i is the rank (0-based)
            pass
        # proper BH
        sorted_p = raw_ps[order]
        bh_thresh = (np.arange(1, m + 1) / m) * 0.05  # just compute adjusted p
        bh_adj_sorted = np.minimum.accumulate(sorted_p * m / np.arange(1, m + 1))[::-1]
        bh_adj_sorted = np.minimum(bh_adj_sorted, 1.0)
        bh_out = np.empty(m)
        bh_out[order] = bh_adj_sorted
        df.loc[has_bear_p, "bear_block_ple0_bh"] = bh_out
    else:
        df["bear_block_ple0_bh"] = np.nan

    # Sort leaderboard: bear realized desc (the decisive metric per charter)
    df_valid = df[df["bear_realized_pct"].notna()].copy()
    df_valid = df_valid.sort_values("bear_realized_pct", ascending=False)

    elapsed = time.time() - t0
    print(f"\n[sweep] Done in {elapsed:.0f}s. {len(df)} tasks, {has_bear_p.sum()} with bear block p.")

    # Print leaderboard
    print("\n=== EXOGENOUS MOVE-CATCH LEADERBOARD (ranked by BEAR realized-net) ===")
    print(f"  {'feat':30}{'dir':5}{'exit':7}{'bear_real%':>11}{'bear_edge':>11}{'bear_ple0':>11}{'bear_ple0_holm':>15}"
          f"{'chop_real%':>11}{'bull_real%':>11}")
    for _, row in df_valid.head(30).iterrows():
        print(f"  {str(row['feat']):30}{str(row['direction']):5}{str(row['exit']):7}"
              f"{str(row.get('bear_realized_pct','')):>11}"
              f"{str(row.get('bear_edge_pp','')):>11}"
              f"{str(row.get('bear_block_ple0','')):>11}"
              f"{str(row.get('bear_block_ple0_holm','')):>15}"
              f"{str(row.get('chop_realized_pct','')):>11}"
              f"{str(row.get('bull_realized_pct','')):>11}")

    # Wall-breaker candidates: bear block_p_le0 < 0.05 (Holm-corrected)
    wall_breakers = df_valid[df_valid["bear_block_ple0_holm"] < 0.05] if "bear_block_ple0_holm" in df_valid else pd.DataFrame()
    print(f"\n=== WALL-BREAKERS (bear Holm-corrected block p < 0.05): {len(wall_breakers)} candidates ===")
    for _, row in wall_breakers.iterrows():
        print(f"  {row['feat']} dir={row['direction']} exit={row['exit']}  "
              f"bear={row.get('bear_realized_pct')}%  ple0={row.get('bear_block_ple0')}  "
              f"holm={row.get('bear_block_ple0_holm')}")

    # Regime-conditional candidates (bear > bull, i.e. NOT just beta)
    regime_cond = df_valid[
        df_valid["bear_realized_pct"].notna() &
        df_valid["bull_realized_pct"].notna() &
        (df_valid["bear_realized_pct"] > df_valid["bull_realized_pct"] + 0.5)  # bear beats bull by >0.5pp
    ]
    print(f"\n=== REGIME-CONDITIONAL (bear > bull + 0.5pp): {len(regime_cond)} candidates ===")
    for _, row in regime_cond.head(15).iterrows():
        print(f"  {row['feat']} dir={row['direction']} exit={row['exit']}  "
              f"bear={row.get('bear_realized_pct')}%  bull={row.get('bull_realized_pct')}%  "
              f"bear_ple0={row.get('bear_block_ple0')}  holm={row.get('bear_block_ple0_holm')}")

    # Also: chop+bear combined (non-bull participation)
    df_valid["chop_bear_mean_pct"] = (
        df_valid["chop_realized_pct"].fillna(0) + df_valid["bear_realized_pct"].fillna(0)
    ) / 2
    chop_bear_top = df_valid.sort_values("chop_bear_mean_pct", ascending=False).head(10)
    print(f"\n=== TOP-10 by CHOP+BEAR (non-bull, avg realized-net) ===")
    for _, row in chop_bear_top.iterrows():
        print(f"  {str(row['feat']):30} dir={row['direction']}  exit={row['exit']}  "
              f"chop={row.get('chop_realized_pct')}%  bear={row.get('bear_realized_pct')}%  "
              f"bear_ple0={row.get('bear_block_ple0')}  holm={row.get('bear_block_ple0_ple0_holm', row.get('bear_block_ple0_holm'))}")

    # Save
    out = out_path or str(Path(__file__).resolve().parents[2] / "runs" / "strat" / "v51_exo_sweep.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    print(f"\n[sweep] Full results saved -> {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Max assets to load")
    ap.add_argument("--n_null", type=int, default=300, help="Null draws for random-entry test")
    ap.add_argument("--out", type=str, default=None, help="Output JSON path")
    a = ap.parse_args()
    sweep(n=a.n, n_null=a.n_null, out_path=a.out)
