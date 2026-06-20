"""capture_chop_battery.py -- THE DECISIVE ADVERSARIAL BATTERY on the chop move-CATCH lead.

Quant-referee lane. The lead (capture_lab selftest): mom14/brk14 beat a churn-immune random-ENTRY null
INTO a move-window IN CHOP (mom14 chop +0.98pp/1d, p~0 IID), bull positive, BEAR negative. The p~0 is
IID-resampled = OVERCONFIDENT: entries OVERLAP heavily (7d holds, 50 assets -> serial + cross-sectional
dependence; effective N << nominal). Subject CHOP+BEAR edges for mom14/brk14/rsi14 (time-exit, 1d+4h) to:

  (1) DATE-BLOCK moving-block bootstrap -- resample contiguous CALENDAR-DATE blocks of entries (NOT iid).
      Honest p and p05 on the edge. This is the make-or-break (kills the overlap inflation).
  (2) REVERSE-SCORE -- invert the trigger (WORST-momentum / below-threshold). Real directional edge =>
      anti-momentum LOSES in chop. If anti-momentum ALSO wins => the 'edge' is concentration/selection of
      windows-with-moves, not direction.
  (3) REGIME-LABEL SHUFFLE -- block-shuffle the bull/chop/bear labels, re-measure chop edge. Real
      regime-conditional edge DIES; regime-independent churn SURVIVES.
  (4) CALENDAR-INVARIANCE -- edge per CALENDAR-day across {1d,4h,1h}. Real economic edge ~stable per day;
      an artifact that grows with bar-count (more overlapping draws) is suspect.

DEV-walled via fleet_lab.load_wide (<= 2024-05-15). Long-only spot, taker cost, causal. No emoji (cp1252).
RWYB: python -m... run from crypto/src with this file's path. Holm/BH note in the verdict.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl
import strat.capture_lab as cl

COST = fl.COST


# ---------------------------------------------------------------------------
# Shared entry extraction: build the per-entry table (one row per (date,asset))
# with realized net return, MFE, regime, entry-DATE, and the raw TI score.
# This is the substrate ALL four cross-checks operate on, derived ONCE per TI/tf.
# ---------------------------------------------------------------------------
def build_entries(lab, ti, tf="1d", min_move=0.03, warm=40):
    """Return a DataFrame of EVERY valid (di, asset) pool entry with time-exit realized net, MFE, regime,
    entry_date, ti_value, and a boolean 'fired' (TI move-onset trigger). Time-exit only (vectorized, exact)."""
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]; hold = 7 * bpd
    MFE = cl.mfe_matrix(C, hold); TIME = cl.time_return_matrix(C, hold)
    reg = cl.regime_series(lab, tf)
    fired = cl.fired_matrix(lab, ti)
    X = lab["F"][ti]                                   # raw TI value (for reverse-score by rank)
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    Ca = C.to_numpy(); MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy()
    valid &= np.isfinite(Ca) & np.isfinite(MFEa) & np.isfinite(TIMEa)
    poolmat = valid & (MFEa > min_move)                # the pool = windows WITH an available up-move to catch
    rows_i, cols_j = np.where(poolmat)
    rega = reg.to_numpy()
    Xa = X.to_numpy()
    df = pd.DataFrame({
        "di": rows_i,
        "asset": cols_j,
        "date": C.index.to_numpy()[rows_i],
        "realized": TIMEa[rows_i, cols_j],
        "mfe": MFEa[rows_i, cols_j],
        "regime": rega[rows_i],
        "fired": fired.to_numpy()[rows_i, cols_j],
        "tival": Xa[rows_i, cols_j],
    })
    df["date"] = pd.to_datetime(df["date"])
    return df, hold


# ---------------------------------------------------------------------------
# (1) DATE-BLOCK MOVING-BLOCK BOOTSTRAP
# Resample contiguous blocks of CALENDAR DATES (the dependence unit). For each
# bootstrap rep we draw a set of date-blocks covering ~ the original span, take
# ALL entries on those dates, and recompute  edge = mean(realized|fired) - mean(realized|pool)
# within a regime. The block carries BOTH cross-asset (same date) and serial
# (adjacent dates) dependence, so the bootstrap dist is the HONEST null+sampling dist.
# ---------------------------------------------------------------------------
def date_block_bootstrap(df, regime, block_days=21, n_boot=2000, seed=0, tf="1d"):
    """Honest p and p05 for the fired-vs-pool edge within `regime`, via date-block resampling.
    Returns dict with point edge, bootstrap mean, p05 (5th pct of edge dist), and p(edge<=0)."""
    bpd = fl.BARS_PER_DAY[tf]
    block_bars = block_days * bpd
    sub = df[df["regime"] == regime]
    if len(sub) < 50 or sub["fired"].sum() < 20:
        return {"regime": regime, "note": "insufficient", "n": int(len(sub)), "n_fired": int(sub["fired"].sum())}
    # point estimate
    pt = sub.loc[sub["fired"], "realized"].mean() - sub["realized"].mean()
    # unique ordered dates present for THIS regime's entries, grouped into contiguous blocks
    all_dates = pd.Index(sorted(df["date"].unique()))           # full date axis (regime-agnostic blocks)
    date_pos = {d: i for i, d in enumerate(all_dates)}
    # index entries by their date-position for fast block membership
    sub = sub.copy(); full = df.copy()
    sub["dpos"] = sub["date"].map(date_pos)
    full["dpos"] = full["date"].map(date_pos)
    max_pos = len(all_dates)
    n_blocks_needed = max(1, int(np.ceil(max_pos / block_bars)))
    rng = np.random.default_rng(seed)
    # pre-bucket entries by block id for speed
    sub_block = (sub["dpos"] // block_bars).to_numpy()
    full_block = (full["dpos"] // block_bars).to_numpy()
    sub_real = sub["realized"].to_numpy(); sub_fired = sub["fired"].to_numpy()
    full_real = full["realized"].to_numpy()
    full_reg = (full["regime"].to_numpy() == regime)
    n_block_ids = int(max(sub_block.max(), full_block.max())) + 1
    # group indices per block
    sub_by_block = [np.where(sub_block == b)[0] for b in range(n_block_ids)]
    full_by_block = [np.where((full_block == b) & full_reg)[0] for b in range(n_block_ids)]
    edges = np.empty(n_boot)
    for r in range(n_boot):
        chosen = rng.integers(0, n_block_ids, size=n_blocks_needed)
        si = np.concatenate([sub_by_block[b] for b in chosen]) if len(chosen) else np.array([], int)
        fi = np.concatenate([full_by_block[b] for b in chosen]) if len(chosen) else np.array([], int)
        if len(si) == 0 or sub_fired[si].sum() < 5 or len(fi) < 5:
            edges[r] = np.nan; continue
        fired_mean = sub_real[si][sub_fired[si]].mean()
        pool_mean = full_real[fi].mean()
        edges[r] = fired_mean - pool_mean
    edges = edges[np.isfinite(edges)]
    return {
        "regime": regime, "n": int(len(sub)), "n_fired": int(sub_fired.sum()),
        "point_edge_pp": round(100 * pt, 3),
        "boot_mean_pp": round(100 * float(edges.mean()), 3),
        "boot_p05_pp": round(100 * float(np.percentile(edges, 5)), 3),
        "boot_p50_pp": round(100 * float(np.percentile(edges, 50)), 3),
        "p_edge_le_0": round(float(np.mean(edges <= 0)), 4),      # honest one-sided p (block-respecting)
        "block_days": block_days, "n_boot": len(edges),
    }


# ---------------------------------------------------------------------------
# (2) REVERSE-SCORE: invert the trigger. For momentum/breakout/rsi the 'fired'
# set is HIGH score. The reverse set is the LOW score (anti-momentum / below band).
# We hold the POOL fixed (windows with a move) and ask: does the anti-trigger set
# also beat the pool in chop? Real directional edge => reverse LOSES (negative edge).
# Concentration artifact (any selection of move-windows wins) => reverse ALSO wins.
# ---------------------------------------------------------------------------
def reverse_score(lab, ti, df, regime, tf="1d"):
    """Edge of the INVERTED trigger vs pool, within regime. Inverted = mirror of fired_matrix threshold."""
    F = lab["F"]; X = F[ti]
    # mirror the fired_matrix logic with an inverted comparison
    if ti in ("mom7", "mom14", "mom30", "accel", "brk14"):
        rev = X < 0.0
    elif ti == "rsi14":
        rev = X < 45                                   # symmetric mirror of >55
    elif ti == "rangepos":
        rev = X < 0.3
    elif ti == "volexp":
        rev = X < 0.8
    else:
        rev = X.lt(X.quantile(0.34, axis=1), axis=0)
    reva = rev.to_numpy()
    rev_flag = reva[df["di"].to_numpy(), df["asset"].to_numpy()]
    sub = df[df["regime"] == regime]
    sub_rev = rev_flag[(df["regime"] == regime).to_numpy()]
    pool_mean = sub["realized"].mean()
    if sub_rev.sum() < 20:
        return {"regime": regime, "note": "insufficient reverse signals", "n_rev": int(sub_rev.sum())}
    rev_mean = sub.loc[sub_rev, "realized"].mean()
    fired_mean = sub.loc[sub["fired"], "realized"].mean()
    return {
        "regime": regime, "n_rev": int(sub_rev.sum()), "n_fired": int(sub["fired"].sum()),
        "fired_edge_pp": round(100 * (fired_mean - pool_mean), 3),
        "reverse_edge_pp": round(100 * (rev_mean - pool_mean), 3),
        "directional": bool((fired_mean - pool_mean) > 0 and (rev_mean - pool_mean) < 0),
    }


# ---------------------------------------------------------------------------
# (3) REGIME-LABEL SHUFFLE: destroy the regime->date mapping by CIRCULARLY
# ROTATING the regime label series (preserves the label autocorrelation / run
# structure, destroys alignment to the actual market). Re-measure the chop edge
# under many rotations. Real regime-conditional edge => shuffled chop edge ~ 0
# (the true edge falls OUTSIDE the shuffled dist). Regime-independent churn =>
# the real edge sits INSIDE the shuffled dist (it survives label destruction).
# ---------------------------------------------------------------------------
def regime_label_shuffle(lab, df, regime, tf="1d", n_shuf=500, seed=0):
    """Distribution of the fired-vs-pool edge in `regime` under circular rotations of the regime labels."""
    reg = cl.regime_series(lab, tf)
    reg_arr = reg.to_numpy()
    di = df["di"].to_numpy(); real = df["realized"].to_numpy(); fired = df["fired"].to_numpy()
    # real edge (true labels)
    true_mask = df["regime"].to_numpy() == regime
    real_edge = real[true_mask & fired].mean() - real[true_mask].mean()
    rng = np.random.default_rng(seed)
    n = len(reg_arr)
    edges = np.empty(n_shuf)
    for s in range(n_shuf):
        shift = rng.integers(50, n - 50)
        rot = np.roll(reg_arr, shift)
        lbl = rot[di]                                  # rotated regime label at each entry's date
        m = lbl == regime
        if m.sum() < 30 or (m & fired).sum() < 10:
            edges[s] = np.nan; continue
        edges[s] = real[m & fired].mean() - real[m].mean()
    edges = edges[np.isfinite(edges)]
    # p = fraction of shuffled edges >= the real edge (if small, real edge is regime-conditional / special)
    p_shuf = float(np.mean(edges >= real_edge))
    return {
        "regime": regime, "real_edge_pp": round(100 * float(real_edge), 3),
        "shuf_mean_pp": round(100 * float(edges.mean()), 3),
        "shuf_p95_pp": round(100 * float(np.percentile(edges, 95)), 3),
        "p_shuf_ge_real": round(p_shuf, 4), "n_shuf": len(edges),
        "regime_conditional": bool(p_shuf < 0.05),     # real edge beats 95% of label-destroyed edges
    }


# ---------------------------------------------------------------------------
# (4) CALENDAR-INVARIANCE: edge per CALENDAR-DAY across timeframes. The time-exit
# holds 7 CALENDAR days at every tf (hold = 7*bpd bars), so the realized return is
# already per-7-calendar-day. We just report the chop/bear edge at 1d/4h/1h. A real
# economic per-day edge is ~stable; an overlap artifact inflates with bar density.
# ---------------------------------------------------------------------------
def calendar_invariance(ti, regime, tfs=("1d", "4h", "1h"), min_move=0.03):
    out = {}
    for tf in tfs:
        lab = fl.load_wide(n=50, tf=tf, min_bars=(200 * fl.BARS_PER_DAY[tf] if tf != "1d" else 400))
        df, hold = build_entries(lab, ti, tf=tf, min_move=min_move)
        sub = df[df["regime"] == regime]
        if len(sub) < 50 or sub["fired"].sum() < 20:
            out[tf] = {"note": "insufficient", "n": int(len(sub))}; continue
        edge = sub.loc[sub["fired"], "realized"].mean() - sub["realized"].mean()
        out[tf] = {
            "n": int(len(sub)), "n_fired": int(sub["fired"].sum()),
            "fired_net_pct": round(100 * float(sub.loc[sub["fired"], "realized"].mean()), 3),
            "pool_net_pct": round(100 * float(sub["realized"].mean()), 3),
            "edge_per_7cal_day_pp": round(100 * float(edge), 3),
        }
    return out


# ---------------------------------------------------------------------------
def run_full_battery(tis=("mom14", "brk14", "rsi14"), tfs=("1d", "4h"), regimes=("chop", "bear")):
    report = {"meta": {"dev_wall": fl.DEV_END, "cost": COST, "hold": "7 calendar days", "min_move": 0.03}}
    labs = {tf: fl.load_wide(n=50, tf=tf, min_bars=(200 * fl.BARS_PER_DAY[tf] if tf != "1d" else 400)) for tf in tfs}
    for ti in tis:
        report[ti] = {}
        for tf in tfs:
            lab = labs[tf]
            df, hold = build_entries(lab, ti, tf=tf)
            tfres = {}
            for rg in regimes:
                bb = date_block_bootstrap(df, rg, block_days=21, n_boot=2000, tf=tf)
                rv = reverse_score(lab, ti, df, rg, tf=tf)
                sh = regime_label_shuffle(lab, df, rg, tf=tf, n_shuf=500)
                tfres[rg] = {"block_bootstrap": bb, "reverse_score": rv, "regime_shuffle": sh}
            report[ti][tf] = tfres
    # (4) calendar-invariance across 1d/4h/1h (separate, loads 1h)
    report["calendar_invariance"] = {}
    for ti in tis:
        report["calendar_invariance"][ti] = {}
        for rg in regimes:
            report["calendar_invariance"][ti][rg] = calendar_invariance(ti, rg, tfs=("1d", "4h", "1h"))
    return report


def verdict_line(ti, tf, rg, node):
    bb = node["block_bootstrap"]; rv = node["reverse_score"]; sh = node["regime_shuffle"]
    if "note" in bb:
        return f"  {ti:6} {tf:3} {rg:4}: {bb['note']}"
    # REAL requires: honest block p < .05 AND p05 > 0 AND directional (reverse loses) AND regime-conditional
    real = (bb["p_edge_le_0"] < 0.05 and bb["boot_p05_pp"] > 0
            and rv.get("directional", False) and sh.get("regime_conditional", False))
    amb = (bb["p_edge_le_0"] < 0.10 and bb["boot_p05_pp"] > -0.1)
    v = "REAL" if real else ("AMBIGUOUS" if amb else "ARTIFACT")
    return (f"  {ti:6} {tf:3} {rg:4}: {v:9} | pt {bb['point_edge_pp']:>6}pp  block-p {bb['p_edge_le_0']:.3f}  "
            f"p05 {bb['boot_p05_pp']:>6}pp | rev {rv.get('reverse_edge_pp','--'):>6}pp dir={rv.get('directional')} | "
            f"shuf-p {sh['p_shuf_ge_real']:.3f} cond={sh['regime_conditional']}")


if __name__ == "__main__":
    print("=" * 110)
    print("DECISIVE BATTERY -- chop/bear move-CATCH lead, DEV-walled <= 2024-05-15, time-exit, taker cost")
    print("=" * 110)
    rep = run_full_battery()
    print("\n### (1)-(3) BLOCK-BOOTSTRAP + REVERSE-SCORE + REGIME-SHUFFLE  (per TI x TF x regime) ###\n")
    for ti in ("mom14", "brk14", "rsi14"):
        for tf in ("1d", "4h"):
            for rg in ("chop", "bear"):
                print(verdict_line(ti, tf, rg, rep[ti][tf][rg]))
    print("\n### (4) CALENDAR-INVARIANCE -- edge per 7-calendar-day across 1d/4h/1h ###\n")
    for ti in ("mom14", "brk14", "rsi14"):
        for rg in ("chop", "bear"):
            ci = rep["calendar_invariance"][ti][rg]
            cells = "  ".join(f"{tf}={ci[tf].get('edge_per_7cal_day_pp', ci[tf].get('note'))}pp(n{ci[tf].get('n_fired','-')})"
                              for tf in ("1d", "4h", "1h"))
            print(f"  {ti:6} {rg:4}: {cells}")
    out = Path(__file__).resolve().parent / "capture_chop_battery_results.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n[written] {out}")
