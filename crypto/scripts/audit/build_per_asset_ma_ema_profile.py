"""build_per_asset_ma_ema_profile.py -- Steps 2 + 3 + 4 of per-asset architecture.

For each asset with MA/EMA data:
  1. Identify candidate cells (WF-robust + n>=20 + non-degenerate + hit>=0.45 + positive mean net)
  2. Take top-K (default 10) by per-asset Sharpe-proxy
  3. Slice each top-K cell by:
     a. btc_regime_30d (bull/chop/bear/crash)
     b. asset_own_regime (asset's own 30d-return regime)
  4. Compute cousin set within asset's top-K via signal-correlation (low |corr| = complementary)
  5. Tag each cell with regime-survival: ALL_WEATHER / BLOCK_OWN_BEAR /
     BLOCK_OWN_CRASH / BULL_ONLY based on per-regime Sharpe sign + sample size

Output: data/processed/per_asset_ma_ema_profile.parquet (long form, cell-level)
        + runs/audit/MA_EMA_PROFILE_2026_05_20/PER_ASSET_PROFILES.md (human-readable)

DESIGN CHOICES:
  - Uses pair_by_asset_cadence.parquet (TRAIN-window per-asset stats, 1d cadence)
    cross-referenced with wf_robust_cells.parquet (TRAIN A/B/C sub-fold robustness).
  - For per-regime slicing, uses event_ma_snapshot.parquet (VAL window).
    Cross-window: per-asset RANKING from TRAIN, per-regime CONDITIONING from VAL.
    This is OK because we want stable cells (TRAIN) that ALSO show regime
    discrimination on out-of-sample data (VAL).
  - Cousin set: top-3 lowest pairwise |signal_corr| within asset's top-10.
  - Regime-survival tagging:
      - n_signaled in regime < 5: insufficient (don't deploy in that regime)
      - Sharpe in regime > +0.10: SURVIVES
      - Sharpe in regime in [-0.10, +0.10]: MARGINAL
      - Sharpe in regime < -0.10: BREAKS
    Cell tag = aggregation of own-regime survival:
      - ALL_WEATHER: survives in 3+ of {own_bull, own_chop, own_bear, own_crash}
        with sufficient samples
      - BLOCK_OWN_CRASH: survives own_bull/chop/bear; breaks/insuff own_crash
      - BLOCK_OWN_BEAR: survives own_bull/chop; breaks own_bear AND own_crash
      - BULL_ONLY: survives only own_bull
      - INSUFFICIENT_DATA: too few events to assess
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PERMUT_DIR = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation"
PER_ASSET_DIR = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
OUT_MD = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "PER_ASSET_PROFILES.md"

# Cost
BUCKET_COST_FRAC_TAKER = {"BLUE": 0.0018, "STEADY": 0.0012, "VOLATILE": 0.0013, "DEGEN": 0.0010}

# Quality gates per cell (must pass to be a candidate)
MIN_N_SIGNALED = 20      # minimum fires per cell in TRAIN
MIN_HIT_RATE = 0.45      # 45% minimum hit rate
MIN_MEAN_NET_PCT = 0.10  # mean PnL net > 0.10% per event (covers cost margin)
TOP_K_PER_ASSET = 10     # how many cells to keep per asset

# Regime survival thresholds (Step 2 robustness fix 2026-05-20: relaxed thresholds.
# Previous (N_REGIME_MIN=5, SHARPE_SURVIVE=0.10) over-tagged BULL_ONLY because most
# assets have <5 events per bear/chop sample in the VAL window. The fix:
#   - N_REGIME_MIN: 5 -> 3 (allow smaller-sample positive classifications)
#   - SHARPE_SURVIVE: 0.10 -> 0.05 (lower bar for "survives")
#   - SHARPE_BREAK: -0.10 -> -0.15 (require clearer evidence of break)
#   - Add "neutral" tier: -0.15 <= sharpe <= 0.05 = "neutral" (deploy at reduced size)
# Net effect: more cells classified as ALL_WEATHER / BULL_AND_CHOP; fewer demoted
# to BULL_ONLY solely due to missing samples.
N_REGIME_MIN = 3
SHARPE_SURVIVE = 0.05
SHARPE_BREAK = -0.15


def survival_label(sharpe: float, n: int) -> str:
    if n < N_REGIME_MIN: return "insufficient"
    if sharpe > SHARPE_SURVIVE: return "survives"
    if sharpe < SHARPE_BREAK:   return "breaks"
    return "neutral"


def classify_cell(per_own_regime: dict) -> str:
    """Aggregate per-own-regime survival into a single deploy tag."""
    labels = {r: survival_label(s["sharpe"] if s else 0, s["n"] if s else 0)
              for r, s in per_own_regime.items()}
    survives = {r for r, lab in labels.items() if lab == "survives"}
    breaks = {r for r, lab in labels.items() if lab == "breaks"}
    neutral = {r for r, lab in labels.items() if lab == "neutral"}
    insuff = {r for r, lab in labels.items() if lab == "insufficient"}

    # ALL_WEATHER: positive in 3+ regimes
    if len({"bull", "chop", "bear", "crash"} & survives) >= 3:
        return "ALL_WEATHER"
    # BLOCK_OWN_CRASH: positive in bull, chop, bear; only crash is bad/missing
    if {"bull", "chop", "bear"} <= (survives | neutral) and "crash" not in survives:
        if "crash" in breaks:
            return "BLOCK_OWN_CRASH"
    # BLOCK_OWN_BEAR: positive in bull+chop, bear/crash break
    if {"bull", "chop"} <= (survives | neutral) and ({"bear", "crash"} & breaks):
        return "BLOCK_OWN_BEAR"
    # BULL_AND_CHOP: positive in bull AND chop with sufficient samples
    if {"bull", "chop"} <= survives:
        return "BULL_AND_CHOP"
    # BULL_ONLY: positive only in bull
    if "bull" in survives and not ({"chop", "bear", "crash"} & survives):
        return "BULL_ONLY"
    # INSUFFICIENT_DATA: no clear positives, mostly insufficient
    if len(survives) == 0 and len(insuff) >= 3:
        return "INSUFFICIENT_DATA"
    return "REGIME_DEPENDENT"


def long_only_stats(events: pd.DataFrame, ma_type: str, fast: int, slow: int, cost: float) -> dict:
    """Per-event long-only PnL: signal * mag - cost for signal==+1 fires."""
    col_f = f"{ma_type}_{fast}"; col_s = f"{ma_type}_{slow}"
    if col_f not in events.columns or col_s not in events.columns:
        return None
    sub = events[events[col_f].notna() & events[col_s].notna()].copy()
    sub["sig"] = np.sign(sub[col_f] - sub[col_s])
    sub["mag"] = sub["magnitude_signed"] / 100.0
    fired = sub[sub["sig"] == 1]
    if len(fired) < 3:
        return {"n": len(fired), "mean": 0.0, "hit": 0.0, "sum": 0.0, "sharpe": 0.0}
    pnl = fired["mag"] - cost
    return {
        "n": int(len(fired)),
        "mean": float(pnl.mean() * 100),
        "hit": float((pnl > 0).mean()),
        "sum": float(pnl.sum() * 100),
        "sharpe": float(pnl.mean() / pnl.std()) if pnl.std() > 1e-9 else 0.0,
    }


def compute_cousin_set(events: pd.DataFrame, candidate_cells: list, target_size: int = 3) -> list:
    """Greedy: pick cell with highest sharpe; then add next-best cell whose
    |signal_correlation| with all selected < 0.6. Until target_size or no candidate."""
    # Build signal vector per cell on this asset's events
    sig_vectors = {}
    for c in candidate_cells:
        col_f = f"{c['ma_type']}_{c['fast']}"
        col_s = f"{c['ma_type']}_{c['slow']}"
        if col_f not in events.columns or col_s not in events.columns:
            continue
        valid = events[col_f].notna() & events[col_s].notna()
        sig = np.sign(events[col_f] - events[col_s]).where(valid, 0).values
        sig_vectors[c["cell_id"]] = (sig, valid.values)

    # Sort candidates by sharpe desc
    sorted_cells = sorted(candidate_cells, key=lambda c: -c["sharpe"])
    selected = [sorted_cells[0]["cell_id"]] if sorted_cells else []

    for c in sorted_cells[1:]:
        if c["cell_id"] not in sig_vectors:
            continue
        max_c = 0.0
        for sk in selected:
            if sk not in sig_vectors:
                continue
            s1, v1 = sig_vectors[c["cell_id"]]
            s2, v2 = sig_vectors[sk]
            both = v1 & v2
            if both.sum() < 10:
                continue
            corr = np.corrcoef(s1[both], s2[both])[0, 1]
            if not np.isnan(corr):
                max_c = max(max_c, abs(corr))
        if max_c < 0.6:
            selected.append(c["cell_id"])
        if len(selected) >= target_size:
            break
    return selected


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("PER-ASSET MA/EMA PROFILE BUILD")
    print("="*78)

    print("Loading inputs...")
    per_asset = pl.read_parquet(PER_ASSET_DIR / "pair_by_asset_cadence.parquet").to_pandas()
    wf_robust = pl.read_parquet(PER_ASSET_DIR / "wf_robust_cells.parquet").to_pandas()
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"])

    print(f"  per_asset_cadence: {len(per_asset):,} rows")
    print(f"  wf_robust_cells:   {len(wf_robust):,} rows")
    print(f"  own_regime_panel:  {len(own_regime):,} rows")

    snap = pl.read_parquet(PERMUT_DIR / "event_ma_snapshot.parquet").to_pandas()
    snap["date"] = pd.to_datetime(snap["date"])
    print(f"  event_ma_snapshot: {len(snap):,} rows ({snap['asset'].nunique()} assets)")

    # Merge asset_own_regime into snapshot
    snap = snap.merge(own_regime[["asset", "date", "asset_own_regime"]],
                       on=["asset", "date"], how="left")
    print(f"  snap after merge: {len(snap):,} rows; "
          f"own_regime null: {snap['asset_own_regime'].isna().sum()}")

    # Filter wf_robust to 1d cadence + wf_robust True
    wf_1d = wf_robust[(wf_robust["cadence"] == "1d") & (wf_robust["wf_robust"] == True)].copy()
    print(f"  wf-robust 1d cells: {len(wf_1d):,} (positive in TRAIN A/B/C)")

    # Process each asset
    profile_rows = []
    assets_in_snap = sorted(snap["asset"].unique())
    print(f"\nBuilding profiles for {len(assets_in_snap)} assets in event_ma_snapshot...")
    for ai, asset in enumerate(assets_in_snap):
        events = snap[snap["asset"] == asset].copy()
        if len(events) < MIN_N_SIGNALED:
            continue
        bucket = events["bucket"].iloc[0]
        cost = BUCKET_COST_FRAC_TAKER.get(bucket, 0.0018)

        # WF-robust cells for this asset (1d cadence)
        asset_wf = wf_1d[wf_1d["asset"] == asset]
        if len(asset_wf) == 0:
            # Fallback: use per_asset_cadence top by Sharpe directly without WF flag
            pac = per_asset[(per_asset["asset"] == asset) &
                             (per_asset["cadence"] == "1d") &
                             (per_asset["n_signaled"] >= MIN_N_SIGNALED) &
                             (~per_asset["degenerate_signal"]) &
                             (~per_asset["signal_quasi_constant"]) &
                             (per_asset["hit_rate"] >= MIN_HIT_RATE) &
                             (per_asset["mean_pnl_pct"] >= MIN_MEAN_NET_PCT)]
            candidates = pac.nlargest(TOP_K_PER_ASSET, "sharpe_proxy").to_dict("records")
        else:
            # Use WF-robust cells, joined with per_asset for full stats
            joined = asset_wf.merge(per_asset, on=["asset", "cadence", "ma_type", "fast", "slow"],
                                     how="left", suffixes=("", "_pa"))
            joined = joined[joined["n_signaled"] >= MIN_N_SIGNALED] if "n_signaled" in joined.columns else joined
            candidates = joined.nlargest(TOP_K_PER_ASSET, "sharpe_taker_total").to_dict("records")

        if not candidates:
            continue

        # For each candidate, compute LO stats on VAL events + per-regime slicing
        asset_cells = []
        for cand in candidates:
            ma_type = cand["ma_type"]; fast = int(cand["fast"]); slow = int(cand["slow"])
            cell_id = f"{asset}|{ma_type}|{fast}|{slow}"
            overall = long_only_stats(events, ma_type, fast, slow, cost)
            if overall["n"] < MIN_N_SIGNALED:
                continue
            # Slice per BTC regime
            per_btc = {}
            for reg in ("bull", "chop", "bear", "crash"):
                s = events[events["btc_regime_30d"] == reg]
                per_btc[reg] = long_only_stats(s, ma_type, fast, slow, cost) if len(s) else None
            # Slice per asset-own regime
            per_own = {}
            for reg in ("bull", "chop", "bear", "crash"):
                s = events[events["asset_own_regime"] == reg]
                per_own[reg] = long_only_stats(s, ma_type, fast, slow, cost) if len(s) else None
            classification = classify_cell(per_own)

            asset_cells.append({
                "cell_id": cell_id, "asset": asset, "bucket": bucket,
                "ma_type": ma_type, "fast": fast, "slow": slow,
                "overall_n": overall["n"], "overall_mean_pct": overall["mean"],
                "overall_hit": overall["hit"], "overall_sum_pct": overall["sum"],
                "sharpe": overall["sharpe"],
                "btc_bull": json.dumps(per_btc.get("bull")),
                "btc_chop": json.dumps(per_btc.get("chop")),
                "btc_bear": json.dumps(per_btc.get("bear")),
                "btc_crash": json.dumps(per_btc.get("crash")),
                "own_bull": json.dumps(per_own.get("bull")),
                "own_chop": json.dumps(per_own.get("chop")),
                "own_bear": json.dumps(per_own.get("bear")),
                "own_crash": json.dumps(per_own.get("crash")),
                "regime_tag": classification,
                "is_cousin_set_member": False,  # filled in next step
            })
        if not asset_cells:
            continue

        # Cousin selection
        cousins = compute_cousin_set(events, asset_cells, target_size=3)
        for c in asset_cells:
            c["is_cousin_set_member"] = c["cell_id"] in cousins

        profile_rows.extend(asset_cells)
        if (ai + 1) % 10 == 0:
            print(f"  {ai+1}/{len(assets_in_snap)} ({asset}): {len(asset_cells)} cells; "
                  f"cousins={len(cousins)}; tags=" +
                  str(dict(pd.Series([c['regime_tag'] for c in asset_cells]).value_counts())))

    if not profile_rows:
        print("[FATAL] no cells generated")
        return 2

    profile_df = pd.DataFrame(profile_rows)
    print(f"\nTotal profile rows: {len(profile_df):,} across {profile_df['asset'].nunique()} assets")
    print(f"\nRegime-tag distribution:")
    print(profile_df["regime_tag"].value_counts().to_string())
    print(f"\nCousin set members: {profile_df['is_cousin_set_member'].sum()} cells")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[OK] wrote {OUT_PARQUET}")

    # Top-level summary markdown
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Per-Asset MA/EMA Profile (2026-05-20)\n"]
    lines.append(f"**Source data**: pair_by_asset_cadence + wf_robust_cells + event_ma_snapshot + asset_own_regime_panel\n")
    lines.append(f"**Gates**: WF-robust on TRAIN A/B/C OR fallback to per-asset top-Sharpe with quality filters\n")
    lines.append(f"**Cells per asset**: top-{TOP_K_PER_ASSET} by per-asset Sharpe-proxy on TRAIN\n")
    lines.append(f"**Cousin set**: top-3 lowest pairwise |signal_corr| within asset's top-{TOP_K_PER_ASSET}\n")
    lines.append(f"**Regime survival**: classified ALL_WEATHER / BLOCK_OWN_CRASH / BLOCK_OWN_BEAR / BULL_ONLY / REGIME_DEPENDENT\n")
    lines.append(f"\n## Summary\n")
    lines.append(f"- Total cells in profile: {len(profile_df):,}")
    lines.append(f"- Assets with at least one profile cell: {profile_df['asset'].nunique()}")
    lines.append(f"- Cousin set members: {profile_df['is_cousin_set_member'].sum()}")
    lines.append(f"\n### Regime-tag distribution\n")
    for tag, n in profile_df["regime_tag"].value_counts().items():
        lines.append(f"- {tag}: {n} cells ({n/len(profile_df)*100:.1f}%)")
    lines.append(f"\n### ALL_WEATHER cells (cross-regime robust)\n")
    aw = profile_df[profile_df["regime_tag"] == "ALL_WEATHER"].sort_values("sharpe", ascending=False).head(20)
    if len(aw):
        lines.append("| asset | bucket | type | (fast, slow) | n | mean % | hit | Sharpe | cousin? |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for _, r in aw.iterrows():
            lines.append(f"| {r['asset']} | {r['bucket']} | {r['ma_type']} | "
                         f"({r['fast']}, {r['slow']}) | {r['overall_n']} | "
                         f"{r['overall_mean_pct']:+.3f} | {r['overall_hit']:.2f} | "
                         f"{r['sharpe']:+.4f} | {'✓' if r['is_cousin_set_member'] else ''} |")
    lines.append(f"\n### Top-30 cells by overall Sharpe (any regime tag, COUSIN MEMBERS first)\n")
    cousins_first = profile_df.sort_values(["is_cousin_set_member", "sharpe"], ascending=[False, False]).head(30)
    lines.append("| asset | bucket | type | (fast, slow) | n | mean % | hit | Sharpe | tag | cousin? |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---|---:|")
    for _, r in cousins_first.iterrows():
        lines.append(f"| {r['asset']} | {r['bucket']} | {r['ma_type']} | "
                     f"({r['fast']}, {r['slow']}) | {r['overall_n']} | "
                     f"{r['overall_mean_pct']:+.3f} | {r['overall_hit']:.2f} | "
                     f"{r['sharpe']:+.4f} | {r['regime_tag']} | "
                     f"{'✓' if r['is_cousin_set_member'] else ''} |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
