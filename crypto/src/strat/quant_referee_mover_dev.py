"""quant_referee_mover_dev.py -- INDEPENDENT referee re-derivation (DEV-walled).

Quant-expert lane. Re-derives, from scratch, the decisive mover-SELECTION-alpha numbers on
fleet_lab DEV (<= 2024-05-15). Trusts NO prior lane script. Cross-checks the 4 lane verdicts.

THE CLAIM: ungated mover engine (rank by composite, top-K, market circuit-breaker on TOTAL
exposure) beats random-SAME-EXPOSURE by z=5-6 = cross-sectional SELECTION alpha (not timing).

THE DECISIVE TEST (per-bar PAIRED same-exposure, exposure cancels exactly):
  excess(di) = roi(real top-K at exposure e) - mean_seed roi(random K at SAME exposure e)
  -> stratify by regime; report (a) pooled-overlapping z (optimistic), (b) HONEST non-overlapping
     z (stride=HOLD, autocorr-free), (c) moving-block bootstrap p05 + frac>0 (fat-tail/autocorr aware).

K=3 INDEPENDENT DERIVATIONS of the decisive chop/bear question:
  D1: paired non-overlapping honest z per regime.
  D2: moving-block bootstrap (block=4 non-ov units) p05 + frac>0 per regime.
  D3: real vs EW-of-ALL-valid at the SAME exposure (zero-concentration control) per regime.

RWYB: python -m strat.quant_referee_mover_dev
No emoji (cp1252). DEV-walled. No git commit. Does NOT touch OOS.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl

COST = fl.COST          # 0.0024 taker RT
HOLD = 7
WARMUP = 210
N_CTRL = 30             # random-pick control seeds per bar
BLOCK = 4               # moving-block bootstrap block length (in non-overlapping units = 28 calendar days)
N_BOOT = 5000


# ---------- causal panels ----------
def panels(lab):
    C = lab["C"]; F = lab["F"]; R = lab["R"]
    return {
        "C": C, "R": R,
        "sma200": C.rolling(200, min_periods=200).mean(),
        "sma50": C.rolling(50, min_periods=50).mean(),
        "vol20": R.rolling(20, min_periods=20).std(),
        "mom14": F["mom14"], "rangepos": F["rangepos"], "volexp": F["volexp"], "accel": F["accel"],
    }


def _z(row, cols):
    r = row[cols]; mu = r.mean(); sd = r.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=cols)
    return ((r - mu) / (sd + 1e-12)).fillna(0.0)


def mover_comp(pan, di):
    C = pan["C"]; rc = C.iloc[di]
    valid = [s for s in C.columns
             if pd.notna(rc[s]) and rc[s] > 0
             and pd.notna(pan["mom14"].iloc[di][s]) and pd.notna(pan["accel"].iloc[di][s])
             and pd.notna(pan["volexp"].iloc[di][s]) and pd.notna(pan["rangepos"].iloc[di][s])]
    if len(valid) < 8:
        return None, valid
    comp = (0.35 * _z(pan["mom14"].iloc[di], valid) + 0.25 * _z(pan["rangepos"].iloc[di], valid)
            + 0.20 * _z(pan["volexp"].iloc[di], valid) + 0.20 * _z(pan["accel"].iloc[di], valid))
    return comp, valid


def exposure_at(pan, di, vol_hi):
    C = pan["C"]; rc = C.iloc[di]; rs = pan["sma50"].iloc[di]
    btc = rc.get("BTCUSDT", np.nan); s200 = pan["sma200"].iloc[di].get("BTCUSDT", np.nan)
    btc_up = pd.notna(s200) and pd.notna(btc) and btc > s200
    pres = rc.notna() & rs.notna()
    breadth = float((rc[pres] > rs[pres]).mean()) if pres.sum() else 0.5
    bv = pan["vol20"].iloc[di].get("BTCUSDT", np.nan)
    hi_vol = pd.notna(bv) and bv >= vol_hi
    if not btc_up:
        return 0.20
    if breadth >= 0.50 and not hi_vol:
        return 1.00
    if breadth >= 0.30:
        return 0.70
    return 0.40


def regime_at(pan, di):
    C = pan["C"]; rc = C.iloc[di]; rs = pan["sma50"].iloc[di]
    btc = rc.get("BTCUSDT", np.nan); s200 = pan["sma200"].iloc[di].get("BTCUSDT", np.nan)
    if pd.isna(btc) or pd.isna(s200):
        return "chop"
    btc_up = btc > s200
    pres = rc.notna() & rs.notna()
    breadth = float((rc[pres] > rs[pres]).mean()) if pres.sum() else 0.5
    if di >= 30:
        p0 = C.iloc[di - 30]; p1 = rc
        rr = [p1[s] / p0[s] - 1 for s in C.columns if pd.notna(p0.get(s)) and pd.notna(p1.get(s)) and p0.get(s, 0) > 0]
        uni = float(np.median(rr)) if rr else 0.0
    else:
        uni = 0.0
    if btc_up and breadth >= 0.50 and uni > 0:
        return "bull"
    if (not btc_up) and breadth < 0.40:
        return "bear"
    return "chop"


def roi(pan, di, picks, expo):
    C = pan["C"]
    if not picks:
        return 0.0
    rr = [C.iloc[di + HOLD].get(s) / C.iloc[di].get(s) - 1
          for s in picks if pd.notna(C.iloc[di].get(s)) and pd.notna(C.iloc[di + HOLD].get(s)) and C.iloc[di].get(s, 0) > 0]
    if not rr:
        return 0.0
    return expo * float(np.mean(rr)) - expo * COST


def block_boot(x, block, n_boot, rng):
    """Moving-block bootstrap of the mean. Returns (p05, frac>0)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < block + 1:
        return None, None
    nb = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.choice(starts_pool, size=nb, replace=True)
        samp = np.concatenate([x[s:s + block] for s in st])[:n]
        means[b] = samp.mean()
    return float(np.percentile(means, 5)), float((means > 0).mean())


def main():
    t0 = time.time()
    print("=" * 88)
    print("QUANT REFEREE -- independent re-derivation of mover-SELECTION alpha (DEV <= %s)" % fl.DEV_END)
    print("=" * 88)
    lab = fl.load_wide(n=50)
    pan = panels(lab)
    C = pan["C"]
    print(f"DEV: {len(lab['syms'])} assets | {C.index.min().date()} -> {C.index.max().date()} ({len(C.index)} bars)")
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"
    max_date = str(C.index.max().date())

    half = len(C.index) // 2
    bve = pan["vol20"]["BTCUSDT"].iloc[:half].dropna()
    vol_hi = float(bve.quantile(0.80)) if len(bve) else 0.05
    print(f"vol_hi (first-half DEV q80): {vol_hi:.4f}\n")

    ctrl_rng = np.random.default_rng(20240515)
    boot_rng = np.random.default_rng(7)
    out = {"dev_end": fl.DEV_END, "max_date": max_date, "n_assets": len(lab["syms"]),
           "vol_hi": round(vol_hi, 5), "hold": HOLD, "by_K": {}}

    valid_dis = list(range(WARMUP, len(C.index) - HOLD - 1))

    for K in [3, 5]:
        rows = []
        for di in valid_dis:
            comp, valid = mover_comp(pan, di)
            if comp is None or len(valid) < K + 2:
                continue
            e = exposure_at(pan, di, vol_hi)
            reg = regime_at(pan, di)
            real_picks = list(comp.sort_values(ascending=False).index[:K])
            r_real = roi(pan, di, real_picks, e)
            # D1/D2 control: random K, SAME exposure, same valid set
            cr = [roi(pan, di, list(ctrl_rng.choice(valid, size=K, replace=False)), e) for _ in range(N_CTRL)]
            r_ctrl = float(np.mean(cr))
            # D3 control: EW of ALL valid, SAME exposure (zero concentration / pure market portfolio)
            r_ew = roi(pan, di, valid, e)
            rows.append({"di": di, "regime": reg, "expo": e,
                         "real": r_real, "ctrl": r_ctrl, "ew": r_ew,
                         "exc_ctrl": r_real - r_ctrl, "exc_ew": r_real - r_ew})
        df = pd.DataFrame(rows)
        # honest non-overlapping subsample
        df_no = df.iloc[::HOLD].reset_index(drop=True)
        print("-" * 88)
        print(f"[K={K}] n_bars={len(df)}  n_nonoverlap={len(df_no)}")

        def stat_block(sub_all, sub_no, col):
            ex = sub_all[col].to_numpy()
            exh = sub_no[col].to_numpy()
            n = len(ex); nh = len(exh)
            mean_ex = float(ex.mean())
            z_pool = mean_ex / (ex.std(ddof=1) / np.sqrt(n)) if n > 2 and ex.std(ddof=1) > 1e-12 else 0.0
            z_hon = float(exh.mean() / (exh.std(ddof=1) / np.sqrt(nh))) if nh > 2 and exh.std(ddof=1) > 1e-12 else 0.0
            p05, frac = block_boot(exh, BLOCK, N_BOOT, boot_rng) if nh >= BLOCK + 1 else (None, None)
            return {"n": n, "n_hon": nh, "mean_pp": round(100 * mean_ex, 3),
                    "z_pool": round(z_pool, 2), "z_honest": round(z_hon, 2),
                    "boot_p05_pp": round(100 * p05, 3) if p05 is not None else None,
                    "boot_frac_pos": round(frac, 3) if frac is not None else None}

        kd = {"overall": {}, "per_regime": {}}
        # D1 (vs random) + D2 (block boot) overall
        kd["overall"]["vs_random"] = stat_block(df, df_no, "exc_ctrl")
        kd["overall"]["vs_EWall"] = stat_block(df, df_no, "exc_ew")
        o = kd["overall"]["vs_random"]
        print(f"  OVERALL vs-random : mean={o['mean_pp']}pp  z_pool={o['z_pool']}  "
              f"z_HONEST={o['z_honest']}  boot_p05={o['boot_p05_pp']}pp  frac>0={o['boot_frac_pos']}")
        print(f"  {'regime':7}{'n_no':>6}{'realE':>8}{'-rand':>9}{'z_hon':>8}{'p05_pp':>9}{'frac>0':>8}  ||  {'-EWall':>9}{'z_hon':>8}{'p05_pp':>9}")
        for reg in ["bull", "chop", "bear"]:
            sa = df[df["regime"] == reg]; sn = df_no[df_no["regime"] == reg]
            if len(sn) < 5:
                print(f"  {reg:7}{len(sn):>6}   (insufficient non-overlap n)")
                continue
            vr = stat_block(sa, sn, "exc_ctrl")
            ve = stat_block(sa, sn, "exc_ew")
            kd["per_regime"][reg] = {"vs_random": vr, "vs_EWall": ve,
                                     "n_bars": len(sa), "avg_expo": round(float(sa["expo"].mean()), 3),
                                     "real_mean_pp": round(100 * float(sa["real"].mean()), 3)}
            print(f"  {reg:7}{vr['n_hon']:>6}{100*sa['real'].mean():>8.2f}{vr['mean_pp']:>9.2f}"
                  f"{vr['z_honest']:>8}{str(vr['boot_p05_pp']):>9}{str(vr['boot_frac_pos']):>8}"
                  f"  ||  {ve['mean_pp']:>9.2f}{ve['z_honest']:>8}{str(ve['boot_p05_pp']):>9}")
        out["by_K"][f"K{K}"] = kd

    # ---------- VERDICT ----------
    print("\n" + "=" * 88)
    print("REFEREE VERDICT (gate on HONEST non-overlapping z>=2 AND block-boot p05>0, in EACH regime)")
    print("=" * 88)
    verdict = {}
    for K in [3, 5]:
        pr = out["by_K"][f"K{K}"]["per_regime"]
        lines = []
        survive_nonbull = []
        for reg in ["bull", "chop", "bear"]:
            if reg not in pr:
                continue
            vr = pr[reg]["vs_random"]
            lines.append(f"{reg}: z_hon={vr['z_honest']} p05={vr['boot_p05_pp']}pp frac>0={vr['boot_frac_pos']} ex={vr['mean_pp']}pp")
            if reg in ("chop", "bear") and vr["z_honest"] is not None and vr["z_honest"] >= 2.0 \
                    and vr["boot_p05_pp"] is not None and vr["boot_p05_pp"] > 0:
                survive_nonbull.append(reg)
        bull = pr.get("bull", {}).get("vs_random", {})
        bull_real = bull.get("z_honest", 0) >= 2.0 and (bull.get("boot_p05_pp") or -1) > 0
        bull_marg = bull.get("z_honest", 0) >= 1.5
        if survive_nonbull:
            v = "REAL REGIME-ROBUST SELECTION ALPHA -- survives non-bull: " + ",".join(survive_nonbull)
        elif bull_real:
            v = "BULL-ONLY -- real in bull, absent chop/bear"
        elif bull_marg:
            v = "BULL-ONLY MARGINAL -- bull borderline after autocorr-correction, absent chop/bear"
        else:
            v = "NO SELECTION ALPHA on DEV in any regime once autocorr-corrected"
        verdict[f"K{K}"] = v
        print(f"  K={K}: {v}")
        print("        " + " | ".join(lines))
    out["verdict"] = verdict
    out["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"quant_referee_mover_dev_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
