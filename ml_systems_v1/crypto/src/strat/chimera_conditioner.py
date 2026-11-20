"""src/strat/chimera_conditioner.py -- Chimera exogenous conditioner on the C1 ROUTER.

NL-CYCLE 2 MANDATE: Use chimera as a REGIME/EXPOSURE conditioner on the C1 adaptive_meta_engine
router. Pre-registered rules (no sweep). Evaluated on referee_harness canonical mechanics
(7d-slice, N=5000, 3 seeds, OOS 2022-01-01+, 1-bar-lag, taker, BH=fixed-EW, date-block perm).

PRE-REGISTERED RULES (designed before running, no post-hoc tuning):
  R1 FUND_CROWDED   : cross-sectional median fund_rate_z30 > 2.0 -> scale all positions to 30%
                      (crowded longs at extreme -> distribution risk, reduce but don't go zero)
  R2 BASIS_PANIC    : cross-sectional median bs_basis_z30 < -2.0 -> CASH (0%)
                      (severe backwardation = futures sellers panic = spot weakness follows)
  R3 STBL_INFLOW    : stbl_stable_shock OR stbl_compound_shock on ANY asset (global signal)
                      -> scale UP weights by 1.3x (capped at 1.0 row-sum)
                      (stablecoin inflow = dry powder entering the market)
  R4 ETF_MEGA       : etf_btc_etf_mega_inflow == 1 (BTC-only, post-2024)
                      -> override router: hold BTC at 100% weight
                      (ETF mega inflow = institutional demand, concentrated into BTC)
  R5 LIQ_CAP_FILTER : per-asset liq_capitulation flag -> EXCLUDE that asset for the day
                      (long liquidation capitulation = exhausted longs, more downside likely)

CAUSAL: chimera features at bar d are used at d (signal), acted at d+1 via 1-bar lag in
book_daily_returns (consistent with referee_harness convention).

HONEST ABOUT COVERAGE:
  - ETF data: ~54% of OOS bars (starts Jan 2024). R4 only fires post-2024.
  - fund_rate_z30, bs_basis_z30, stbl_*: 99-100% OOS coverage. R1/R2/R3 fire throughout OOS.
  - liq_capitulation: 100% coverage but rare (~5 events per asset over OOS).

No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.adaptive_meta_engine as ame
from strat.referee_harness import (
    book_daily_returns, bh_ew_weights, slice_stats, bh_slice_stats
)

COST = lab.COST
CHIMERA_DIR = ROOT.parent / "data" / "processed" / "chimera" / "1d"

# ============================================================
# RULE THRESHOLDS (pre-registered, no post-hoc tuning)
# ============================================================
R1_FUND_Z30_THR   = 2.0     # crowd-long extreme
R2_BASIS_Z30_THR  = -2.0    # panic backwardation
R3_STBL_SCALE     = 1.3     # stablecoin inflow scale-up factor
R4_ETF_BTC_WEIGHT = 1.0     # ETF mega inflow: go 100% BTC
R5_LIQ_CAP_EXCL   = True    # exclude liq_capitulation assets


# ============================================================
# CHIMERA LOADER (per-asset, lazy)
# ============================================================
def load_chimera_panel(syms: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load chimera parquet for each symbol, return dict of sym -> pd.DataFrame indexed by date."""
    out = {}
    for sym in syms:
        key = sym.lower().replace("usdt", "") + "usdt"
        # find the file
        matches = list(CHIMERA_DIR.glob(f"{key}_v51_chimera_1d_*.parquet"))
        if not matches:
            # try exact
            matches = list(CHIMERA_DIR.glob(f"{sym.lower()}_v51_chimera_1d_*.parquet"))
        if not matches:
            continue
        fpath = sorted(matches)[-1]  # newest
        df = pl.read_parquet(fpath).filter(
            (pl.col("date") >= pl.lit(start).str.to_date()) &
            (pl.col("date") < pl.lit(end).str.to_date())
        )
        # skip xex_ poisoned columns
        cols_to_keep = [c for c in df.columns if not c.startswith("xex_")]
        df = df.select(cols_to_keep)
        pdf = df.to_pandas().set_index("date")
        pdf.index = pd.to_datetime(pdf.index)
        out[sym] = pdf
    return out


def build_chimera_signals(chimera: dict[str, pd.DataFrame], date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a date-indexed DataFrame of conditioner signals from chimera.

    Columns:
      cs_fund_z30_med   : cross-sectional median of fund_rate_z30 (rule R1)
      cs_basis_z30_med  : cross-sectional median of bs_basis_z30 (rule R2)
      any_stbl_shock    : 1 if any asset fires stbl_stable_shock OR stbl_compound_shock (R3)
      btc_etf_mega      : 1 if BTC fires etf_btc_etf_mega_inflow (R4; NaN pre-2024)
      liq_cap_<SYM>     : 1 if asset SYM fires liq_capitulation (R5; per-asset)
    """
    rows = []
    syms = list(chimera.keys())

    for d in date_index:
        row = {"date": d}
        fund_vals, basis_vals, stbl_any, btc_etf_mega = [], [], 0, np.nan
        per_asset_liq = {}

        for sym in syms:
            cdf = chimera.get(sym)
            if cdf is None or d not in cdf.index:
                continue
            r = cdf.loc[d]

            # R1: funding z30 per asset
            fz = r.get("fund_rate_z30")
            if pd.notna(fz):
                fund_vals.append(float(fz))

            # R2: basis z30 per asset
            bz = r.get("bs_basis_z30")
            if pd.notna(bz):
                basis_vals.append(float(bz))

            # R3: stablecoin shock (global signal, same across assets -- use BTC as ref)
            if sym == "BTCUSDT":
                ss = r.get("stbl_stable_shock", 0) or 0
                cs = r.get("stbl_compound_shock", 0) or 0
                stbl_any = int(bool(ss) or bool(cs))

            # R4: ETF mega inflow (BTC-specific)
            if sym == "BTCUSDT":
                em = r.get("etf_btc_etf_mega_inflow")
                btc_etf_mega = float(em) if pd.notna(em) else np.nan

            # R5: per-asset liquidation capitulation
            lc = r.get("liq_capitulation", 0) or 0
            per_asset_liq[sym] = int(bool(lc))

        row["cs_fund_z30_med"] = float(np.median(fund_vals)) if fund_vals else np.nan
        row["cs_basis_z30_med"] = float(np.median(basis_vals)) if basis_vals else np.nan
        row["any_stbl_shock"] = stbl_any
        row["btc_etf_mega"] = btc_etf_mega
        for sym in syms:
            row[f"liq_cap_{sym}"] = per_asset_liq.get(sym, 0)
        rows.append(row)

    sig = pd.DataFrame(rows).set_index("date")
    sig.index = pd.to_datetime(sig.index)
    return sig


# ============================================================
# CONDITIONER: apply rules to the router's W matrix
# ============================================================
def apply_chimera_conditioner(
    W_router: pd.DataFrame,
    signals: pd.DataFrame,
    syms: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply pre-registered chimera rules to the router weight matrix.
    Returns (W_conditioned, firing_stats).
    """
    W = W_router.copy()
    stats = {
        "r1_fire_days": 0, "r2_fire_days": 0, "r3_fire_days": 0,
        "r4_fire_days": 0, "r5_fire_total": 0,
    }

    for d in W.index:
        if d not in signals.index:
            continue
        sig = signals.loc[d]
        row = W.loc[d].copy()
        modified = False

        # R5 first: exclude liq_capitulation assets (before scaling)
        for sym in syms:
            cap_col = f"liq_cap_{sym}"
            if cap_col in sig.index and sig[cap_col] == 1 and sym in row.index and row[sym] > 0:
                row[sym] = 0.0
                stats["r5_fire_total"] += 1
                modified = True
        # Renormalize after R5 if needed
        if modified and row.sum() > 0:
            row = row / row.sum()

        # R4: ETF mega inflow -> 100% BTC (overrides everything, post-2024 only)
        etf_mega = sig.get("btc_etf_mega", np.nan)
        if pd.notna(etf_mega) and etf_mega == 1.0:
            row[:] = 0.0
            if "BTCUSDT" in row.index:
                row["BTCUSDT"] = R4_ETF_BTC_WEIGHT
            stats["r4_fire_days"] += 1
            W.loc[d] = row
            continue  # R4 overrides all other rules for this day

        # R1: crowded longs -> scale down to 30%
        fz = sig.get("cs_fund_z30_med", np.nan)
        if pd.notna(fz) and fz > R1_FUND_Z30_THR:
            row = row * 0.30
            stats["r1_fire_days"] += 1

        # R2: basis panic -> CASH
        bz = sig.get("cs_basis_z30_med", np.nan)
        if pd.notna(bz) and bz < R2_BASIS_Z30_THR:
            row[:] = 0.0
            stats["r2_fire_days"] += 1

        # R3: stablecoin inflow -> scale up (cap at 1.0 row-sum)
        stbl = sig.get("any_stbl_shock", 0)
        if stbl == 1 and row.sum() > 1e-9:
            scaled = row * R3_STBL_SCALE
            if scaled.sum() > 1.0:
                scaled = scaled / scaled.sum()
            row = scaled
            stats["r3_fire_days"] += 1

        W.loc[d] = row

    return W, stats


# ============================================================
# MAIN EVALUATION
# ============================================================
def main():
    t0 = time.time()
    OOS_START = "2022-01-01"
    OOS_END   = "2026-06-01"
    N         = 5000
    SEEDS     = [11, 23, 42]

    print("=" * 76)
    print("CHIMERA CONDITIONER ON C1 ROUTER -- NL-CYCLE 2")
    print(f"OOS: {OOS_START} -> {OOS_END} | n_slices={N} | seeds={SEEDS}")
    print(f"Rules (pre-registered): R1=fund_z30>{R1_FUND_Z30_THR} scale30% | "
          f"R2=basis_z30<{R2_BASIS_Z30_THR} CASH | R3=stbl_shock scale{R3_STBL_SCALE}x | "
          f"R4=ETF_mega 100%BTC | R5=liq_cap exclude")
    print("=" * 76)

    # ---- Load price data ----
    print("\n[1] Loading price indicators...")
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    syms = list(C.columns)

    # ---- Load chimera ----
    print(f"[2] Loading chimera for {len(syms)} assets from {OOS_START}...")
    chimera = load_chimera_panel(syms, "2019-01-01", OOS_END)
    print(f"    Loaded chimera for: {sorted(chimera.keys())}")
    missing = set(syms) - set(chimera.keys())
    if missing:
        print(f"    WARNING: no chimera for {missing}")

    # ---- Build chimera signals ----
    print("[3] Building chimera signals panel...")
    signals = build_chimera_signals(chimera, C.index)
    # coverage report
    print(f"    Signals rows: {len(signals)}")
    for col in ["cs_fund_z30_med", "cs_basis_z30_med", "any_stbl_shock", "btc_etf_mega"]:
        oos_mask = signals.index >= pd.Timestamp(OOS_START)
        oos_sig = signals[oos_mask][col]
        nn = oos_sig.notna().sum()
        print(f"    {col:<30} OOS coverage {nn}/{len(oos_sig)} = {100*nn/len(oos_sig):.0f}%")
    # Rule firing rates in OOS
    oos_sig = signals[signals.index >= pd.Timestamp(OOS_START)]
    fz_fires = (oos_sig["cs_fund_z30_med"] > R1_FUND_Z30_THR).sum()
    bz_fires = (oos_sig["cs_basis_z30_med"] < R2_BASIS_Z30_THR).sum()
    stbl_fires = (oos_sig["any_stbl_shock"] == 1).sum()
    etf_fires = (oos_sig["btc_etf_mega"] == 1).sum()
    print(f"    OOS rule fire rates: R1={fz_fires}d R2={bz_fires}d R3={stbl_fires}d R4={etf_fires}d")

    # ---- Build BH baseline ----
    print("[4] Building BH baseline...")
    bh_W = bh_ew_weights(ind)
    bh_b = book_daily_returns(bh_W, ind)
    bh_stats = {s: bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS}
    bh_pr = [bh_stats[s]["pos_rate"] for s in SEEDS]
    bh_mn = [bh_stats[s]["mean_pct"] for s in SEEDS]
    bh_p5 = [bh_stats[s]["p05_pct"] for s in SEEDS]
    print(f"    BH pos_rate={round(float(np.mean(bh_pr)),1)}% mean={round(float(np.mean(bh_mn)),2)}% "
          f"p05={round(float(np.mean(bh_p5)),2)}%")

    # ---- Build router (C1 baseline) ----
    print("[5] Building C1 router weight matrix...")
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    W_router = ame.build_weight_matrix(ind, vthr)
    rb = book_daily_returns(W_router, ind)
    rprs = [slice_stats(rb, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    rpr = [x["pos_rate"] for x in rprs]; rmn = [x["mean_pct"] for x in rprs]
    rp5 = [x["p05_pct"] for x in rprs]; rdn = [x["down_wk_eng_mean"] for x in rprs]
    print(f"    Router pos_rate={round(float(np.mean(rpr)),1)}% "
          f"mean={round(float(np.mean(rmn)),2)}% "
          f"p05={round(float(np.mean(rp5)),2)}% "
          f"down_wk_mean={round(float(np.mean(rdn)),2)}%")

    # ---- Apply chimera conditioner ----
    print("[6] Applying chimera conditioner to router...")
    W_cond, fire_stats = apply_chimera_conditioner(W_router, signals, syms)
    print(f"    Rule firing: {fire_stats}")

    # ---- Evaluate conditioned engine ----
    cb = book_daily_returns(W_cond, ind)
    cprs_raw = [slice_stats(cb, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    cpr = [x["pos_rate"] for x in cprs_raw]
    cmn = [x["mean_pct"] for x in cprs_raw]
    cp5 = [x["p05_pct"] for x in cprs_raw]
    cdn = [x["down_wk_eng_mean"] for x in cprs_raw]
    cbw = [x["beat_bh_pct"] for x in cprs_raw]
    print(f"    Conditioned pos_rate={round(float(np.mean(cpr)),1)}% "
          f"mean={round(float(np.mean(cmn)),2)}% "
          f"p05={round(float(np.mean(cp5)),2)}% "
          f"down_wk_mean={round(float(np.mean(cdn)),2)}% "
          f"beat_bh={round(float(np.mean(cbw)),1)}%")

    # ---- Delta table ----
    print("\n" + "=" * 76)
    print("RESULTS TABLE (N=5000 slices, 3 seeds, OOS 2022-01-01+)")
    print("=" * 76)
    print(f"{'Engine':<28} {'pos_rate%':>10} {'mean%':>8} {'p05%':>8} {'down_wk%':>10} {'beat_bh%':>10}")
    print("-" * 76)
    print(f"{'BH (EW fixed)':<28} {round(float(np.mean(bh_pr)),1):>10} "
          f"{round(float(np.mean(bh_mn)),2):>8} "
          f"{round(float(np.mean(bh_p5)),2):>8}  {'N/A':>8}  {'N/A':>8}")
    print(f"{'C1 Router (price-only)':<28} {round(float(np.mean(rpr)),1):>10} "
          f"{round(float(np.mean(rmn)),2):>8} "
          f"{round(float(np.mean(rp5)),2):>8} "
          f"{round(float(np.mean(rdn)),2):>10} "
          f"{round(float(np.mean([x['beat_bh_pct'] for x in rprs])),1):>10}")
    print(f"{'C2 Chimera-Conditioned':<28} {round(float(np.mean(cpr)),1):>10} "
          f"{round(float(np.mean(cmn)),2):>8} "
          f"{round(float(np.mean(cp5)),2):>8} "
          f"{round(float(np.mean(cdn)),2):>10} "
          f"{round(float(np.mean(cbw)),1):>10}")
    print("-" * 76)
    # Deltas
    delta_pr  = round(float(np.mean(cpr)) - float(np.mean(rpr)), 1)
    delta_mn  = round(float(np.mean(cmn)) - float(np.mean(rmn)), 2)
    delta_p5  = round(float(np.mean(cp5)) - float(np.mean(rp5)), 2)
    delta_dn  = round(float(np.mean(cdn)) - float(np.mean(rdn)), 2)
    print(f"{'Delta (C2 - C1 Router)':<28} {delta_pr:>+10.1f} {delta_mn:>+8.2f} "
          f"{delta_p5:>+8.2f} {delta_dn:>+10.2f}  {'---':>10}")

    # ---- Individual rule ablations ----
    print("\n[7] Rule ablations (one rule at a time on top of base router)...")
    ablation_results = {}
    rules = {
        "R1_fund_crowded": {"r1": True, "r2": False, "r3": False, "r4": False, "r5": False},
        "R2_basis_panic":  {"r1": False, "r2": True,  "r3": False, "r4": False, "r5": False},
        "R3_stbl_inflow":  {"r1": False, "r2": False, "r3": True,  "r4": False, "r5": False},
        "R4_etf_mega":     {"r1": False, "r2": False, "r3": False, "r4": True,  "r5": False},
        "R5_liq_cap":      {"r1": False, "r2": False, "r3": False, "r4": False, "r5": True},
    }
    for rname, flags in rules.items():
        Wa, fst = _apply_rules_selective(W_router, signals, syms, flags)
        ba = book_daily_returns(Wa, ind)
        aprs = [slice_stats(ba, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        apr = round(float(np.mean([x["pos_rate"] for x in aprs])), 1)
        amn = round(float(np.mean([x["mean_pct"] for x in aprs])), 2)
        ap5 = round(float(np.mean([x["p05_pct"] for x in aprs])), 2)
        adn = round(float(np.mean([x["down_wk_eng_mean"] for x in aprs])), 2)
        ablation_results[rname] = {"pos_rate": apr, "mean": amn, "p05": ap5, "down_wk": adn, "fires": fst}
        dpr = round(apr - float(np.mean(rpr)), 1)
        dmn = round(amn - float(np.mean(rmn)), 2)
        dp5 = round(ap5 - float(np.mean(rp5)), 2)
        print(f"  {rname:<22} pos_rate={apr}% ({dpr:+.1f}) mean={amn}% ({dmn:+.2f}) "
              f"p05={ap5}% ({dp5:+.2f}) down_wk={adn}%  fires={fst}")

    # ---- Date-block permutation test on C2 vs C1 ----
    print("\n[8] Date-block permutation test: does C2 beat C1 router reliably?")
    perm_p = _date_block_perm(cb, rb, OOS_START, OOS_END, n_perm=2000, seed=99)
    print(f"    C2 mean vs C1 mean: {round(float(cb[cb.index >= OOS_START].mean())*100,4)}% vs "
          f"{round(float(rb[rb.index >= OOS_START].mean())*100,4)}%")
    print(f"    Date-block permutation p-value (C2 > C1 daily mean): {perm_p:.4f}")

    runtime = round(time.time() - t0, 1)
    out = {
        "oos": [OOS_START, OOS_END], "n_slices": N, "seeds": SEEDS,
        "bh": {"pos_rate": round(float(np.mean(bh_pr)),1), "mean_pct": round(float(np.mean(bh_mn)),2),
               "p05_pct": round(float(np.mean(bh_p5)),2)},
        "c1_router": {"pos_rate": round(float(np.mean(rpr)),1), "mean_pct": round(float(np.mean(rmn)),2),
                      "p05_pct": round(float(np.mean(rp5)),2),
                      "down_wk_mean": round(float(np.mean(rdn)),2),
                      "beat_bh": round(float(np.mean([x['beat_bh_pct'] for x in rprs])),1),
                      "seeds_pr": rpr, "seeds_mn": rmn},
        "c2_chimera": {"pos_rate": round(float(np.mean(cpr)),1), "mean_pct": round(float(np.mean(cmn)),2),
                       "p05_pct": round(float(np.mean(cp5)),2),
                       "down_wk_mean": round(float(np.mean(cdn)),2),
                       "beat_bh": round(float(np.mean(cbw)),1),
                       "seeds_pr": cpr, "seeds_mn": cmn,
                       "fire_stats": fire_stats},
        "delta_c2_minus_c1": {"pos_rate": delta_pr, "mean_pct": delta_mn, "p05_pct": delta_p5,
                               "down_wk_mean": delta_dn},
        "ablations": ablation_results,
        "perm_p_c2_gt_c1": perm_p,
        "runtime_s": runtime,
    }
    outp = ROOT.parent / "runs" / "strat" / "chimera_conditioner_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({runtime}s)")
    return out


# ============================================================
# SELECTIVE RULE APPLICATION (for ablations)
# ============================================================
def _apply_rules_selective(
    W_router: pd.DataFrame,
    signals: pd.DataFrame,
    syms: list[str],
    flags: dict,
) -> tuple[pd.DataFrame, dict]:
    """Same as apply_chimera_conditioner but only applies flagged rules."""
    W = W_router.copy()
    stats = {"r1": 0, "r2": 0, "r3": 0, "r4": 0, "r5": 0}

    for d in W.index:
        if d not in signals.index:
            continue
        sig = signals.loc[d]
        row = W.loc[d].copy()

        if flags.get("r5", False):
            modified = False
            for sym in syms:
                cap_col = f"liq_cap_{sym}"
                if cap_col in sig.index and sig[cap_col] == 1 and sym in row.index and row[sym] > 0:
                    row[sym] = 0.0
                    stats["r5"] += 1
                    modified = True
            if modified and row.sum() > 0:
                row = row / row.sum()

        if flags.get("r4", False):
            etf_mega = sig.get("btc_etf_mega", np.nan)
            if pd.notna(etf_mega) and etf_mega == 1.0:
                row[:] = 0.0
                if "BTCUSDT" in row.index:
                    row["BTCUSDT"] = R4_ETF_BTC_WEIGHT
                stats["r4"] += 1
                W.loc[d] = row
                continue

        if flags.get("r1", False):
            fz = sig.get("cs_fund_z30_med", np.nan)
            if pd.notna(fz) and fz > R1_FUND_Z30_THR:
                row = row * 0.30
                stats["r1"] += 1

        if flags.get("r2", False):
            bz = sig.get("cs_basis_z30_med", np.nan)
            if pd.notna(bz) and bz < R2_BASIS_Z30_THR:
                row[:] = 0.0
                stats["r2"] += 1

        if flags.get("r3", False):
            stbl = sig.get("any_stbl_shock", 0)
            if stbl == 1 and row.sum() > 1e-9:
                scaled = row * R3_STBL_SCALE
                if scaled.sum() > 1.0:
                    scaled = scaled / scaled.sum()
                row = scaled
                stats["r3"] += 1

        W.loc[d] = row

    return W, stats


# ============================================================
# DATE-BLOCK PERMUTATION TEST
# ============================================================
def _date_block_perm(
    eng_ret: pd.Series,
    base_ret: pd.Series,
    oos_start: str,
    oos_end: str,
    n_perm: int = 2000,
    block_size: int = 21,
    seed: int = 99,
) -> float:
    """Test: C2 daily mean > C1 daily mean, using block-permutation of (eng - base) differences."""
    mask = (eng_ret.index >= pd.Timestamp(oos_start)) & (eng_ret.index < pd.Timestamp(oos_end))
    diff = (eng_ret - base_ret)[mask].fillna(0.0).values
    obs_mean = float(diff.mean())

    rng = np.random.default_rng(seed)
    n = len(diff)
    # split into blocks
    n_blocks = n // block_size
    blocks = [diff[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    leftover = diff[n_blocks * block_size:]

    null_means = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(blocks))
        perm = np.concatenate([b * s for b, s in zip(blocks, signs)])
        if len(leftover) > 0:
            perm = np.concatenate([perm, leftover * rng.choice([-1.0, 1.0])])
        null_means.append(float(perm.mean()))

    p_val = float(np.mean(np.array(null_means) >= obs_mean))
    return round(p_val, 4)


if __name__ == "__main__":
    main()
