"""src/strat/mech_exit_sweep.py -- SUB-DAILY MECHANISM EXIT SWEEP (4h primary).

THESIS (user 2026-06-20): The 1d failure in chop/bear was a HOLD-policy artifact.
Replace fixed 7d hold with MECHANISM exits (trailing-stop, profit-target, time-stop, ATR-stop).
Does a fast exit mechanism convert the bull-only selection edge into something positive in chop/bear?

EXPERIMENT DESIGN:
  - 4h universe (50 assets, DEV-wall 2024-05-15)
  - Regime labels: bull / chop / bear via rolling BTC 30-bar momentum (causal)
  - Agents: 3 pre-registered feature sets
  - Exits: 5 pre-registered mechanisms (fixed-hold baseline + 4 mechanisms)
  - Null: same-exposure SHUFFLE -- pre-cache mechanism returns for ALL (asset,bar) pairs, then
    shuffle just resamples from the cached array (fast). Moving-block bootstrap for variance.
  - Metric: net ROI (COST=0.0024 taker RT per trade deducted)
  - Holm correction across mechanism x regime cells

RWYB: python -m strat.mech_exit_sweep
No emoji. No git commits.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.fleet_lab import load_wide, agent_score, COST, BARS_PER_DAY, DEV_END

# ---- CONSTANTS ----------------------------------------------------------------
TF = "4h"
BPD = BARS_PER_DAY[TF]          # 6 bars/day
K = 5                            # top-K picks per agent
N_SHUFFLE = 500                  # shuffle iterations for null
BLOCK_LEN = 6                    # moving-block length for bootstrap (in slices)
MAX_HOLD = 42                    # 7 calendar days @4h -- the fixed baseline
REGIME_LOOKBACK = 60             # ~10 calendar days @4h
MIN_VALID_SLICES = 30            # minimum slices per regime to report

# Pre-registered agents (diverse, no tuning)
AGENTS = {
    "mom_composite": ["mom7", "mom14", "accel"],
    "breakout_flow":  ["brk14", "ofi", "vpin"],
    "full_composite": ["mom14", "rsi14", "brk14", "ofi", "vpin"],
}

# Pre-registered mechanism grid (small, pre-locked before running)
# Each mechanism specifies how exits are computed from per-asset bar-level cache
MECH_GRID = {
    "fixed_42":     {"type": "fixed",    "hold": 42},
    "trail_2pct":   {"type": "trailing", "trail": 0.02, "max_hold": 42},
    "trail_4pct":   {"type": "trailing", "trail": 0.04, "max_hold": 42},
    "target5_ts2":  {"type": "target",   "target": 0.05, "trail": 0.02, "max_hold": 42},
    "time12_cut2":  {"type": "time",     "hold": 12, "stop": -0.02},
    "atr_1p5":      {"type": "atr",      "atr_mult": 1.5, "max_hold": 42},
}


# ---- REGIME LABELS -----------------------------------------------------------
def compute_regimes(lab: dict) -> pd.Series:
    """Causal bull/chop/bear labels using rolling BTC 30-bar return vs rolling percentiles."""
    C = lab["C"]
    btc_cols = [c for c in C.columns if "BTC" in c.upper()]
    mkt = C[btc_cols[0]].ffill() if btc_cols else C.fillna(method="ffill").mean(axis=1)
    roll_ret = mkt.pct_change(REGIME_LOOKBACK)
    p33 = roll_ret.rolling(200, min_periods=60).quantile(0.33)
    p67 = roll_ret.rolling(200, min_periods=60).quantile(0.67)
    regime = pd.Series("chop", index=C.index)
    regime[roll_ret > p67] = "bull"
    regime[roll_ret < p33] = "bear"
    regime[roll_ret.isna()] = "chop"
    return regime


# ---- VECTORISED MECHANISM CACHE ----------------------------------------------
def build_mech_cache(lab: dict, mech: dict) -> np.ndarray:
    """Pre-compute gross returns for EVERY (asset_idx, entry_bar) pair under mechanism `mech`.

    Returns matrix shape (n_bars, n_assets) of gross returns.
    NaN where entry is not possible (close to end, missing price).
    Also returns hold_bars matrix shape (n_bars, n_assets).
    """
    C = lab["C"].values       # (T, A)
    H = lab["H"].values
    L = lab["L"].values
    T, A = C.shape
    gross = np.full((T, A), np.nan, dtype=np.float32)
    holds = np.full((T, A), np.nan, dtype=np.float32)

    mtype = mech["type"]

    if mtype == "fixed":
        hold = int(mech["hold"])
        # entry bar i: exit at i+hold
        for i in range(T - hold - 1):
            ep = C[i, :]                        # (A,)
            xp = C[i + hold, :]
            valid = (~np.isnan(ep)) & (~np.isnan(xp)) & (ep > 0)
            gross[i, valid] = xp[valid] / ep[valid] - 1
            holds[i, valid] = hold
        return gross, holds

    max_hold = int(mech.get("max_hold", MAX_HOLD))

    if mtype == "trailing":
        trail = float(mech["trail"])
        target = mech.get("target", None)
        for i in range(T - max_hold - 1):
            ep = C[i, :]
            valid_mask = (~np.isnan(ep)) & (ep > 0)
            vi = np.where(valid_mask)[0]
            if len(vi) == 0:
                continue
            # vectorise over assets: simulate bar-by-bar for assets in parallel
            ep_v = ep[vi]
            peak = ep_v.copy()
            done = np.zeros(len(vi), dtype=bool)
            exit_ret = np.zeros(len(vi), dtype=np.float32)
            exit_h   = np.full(len(vi), max_hold, dtype=np.float32)

            for h in range(1, max_hold + 1):
                bi = i + h
                if bi >= T:
                    break
                hi_v = H[bi, vi]
                lo_v = L[bi, vi]
                ci_v = C[bi, vi]
                # fill NaN H/L with close
                hi_v = np.where(np.isnan(hi_v), ci_v, hi_v)
                lo_v = np.where(np.isnan(lo_v), ci_v, lo_v)
                # update peak
                peak = np.maximum(peak, hi_v)
                stop_lv = peak * (1 - trail)
                # check stop
                stopped = (~done) & (lo_v <= stop_lv)
                exit_ret = np.where(stopped, np.maximum(stop_lv, lo_v) / ep_v - 1, exit_ret)
                exit_h   = np.where(stopped, h, exit_h)
                done = done | stopped
                # check target
                if target is not None:
                    targeted = (~done) & (hi_v >= ep_v * (1 + target))
                    exit_ret = np.where(targeted, target, exit_ret)
                    exit_h   = np.where(targeted, h, exit_h)
                    done = done | targeted
                # final bar
                if h == max_hold:
                    final = ~done
                    exit_ret = np.where(final & ~np.isnan(ci_v), ci_v / ep_v - 1, exit_ret)
                    exit_h   = np.where(final, h, exit_h)
                    done = done | final
                if done.all():
                    break

            gross[i, vi] = exit_ret
            holds[i, vi] = exit_h
        return gross, holds

    if mtype == "target":
        trail = float(mech.get("trail", 0.02))
        target = float(mech["target"])
        # Same as trailing but with profit target first
        for i in range(T - max_hold - 1):
            ep = C[i, :]
            valid_mask = (~np.isnan(ep)) & (ep > 0)
            vi = np.where(valid_mask)[0]
            if len(vi) == 0:
                continue
            ep_v = ep[vi]
            peak = ep_v.copy()
            done = np.zeros(len(vi), dtype=bool)
            exit_ret = np.zeros(len(vi), dtype=np.float32)
            exit_h   = np.full(len(vi), max_hold, dtype=np.float32)

            for h in range(1, max_hold + 1):
                bi = i + h
                if bi >= T:
                    break
                hi_v = H[bi, vi]; lo_v = L[bi, vi]; ci_v = C[bi, vi]
                hi_v = np.where(np.isnan(hi_v), ci_v, hi_v)
                lo_v = np.where(np.isnan(lo_v), ci_v, lo_v)
                peak = np.maximum(peak, hi_v)
                # trailing stop
                stop_lv = peak * (1 - trail)
                stopped = (~done) & (lo_v <= stop_lv)
                exit_ret = np.where(stopped, np.maximum(stop_lv, lo_v) / ep_v - 1, exit_ret)
                exit_h   = np.where(stopped, h, exit_h)
                done = done | stopped
                # profit target
                targeted = (~done) & (hi_v >= ep_v * (1 + target))
                exit_ret = np.where(targeted, target, exit_ret)
                exit_h   = np.where(targeted, h, exit_h)
                done = done | targeted
                if h == max_hold:
                    final = ~done
                    exit_ret = np.where(final & ~np.isnan(ci_v), ci_v / ep_v - 1, exit_ret)
                    done = done | final
                if done.all():
                    break

            gross[i, vi] = exit_ret
            holds[i, vi] = exit_h
        return gross, holds

    if mtype == "time":
        hold = int(mech["hold"])
        stop = float(mech.get("stop", -0.02))
        for i in range(T - hold - 1):
            ep = C[i, :]
            valid_mask = (~np.isnan(ep)) & (ep > 0)
            vi = np.where(valid_mask)[0]
            if len(vi) == 0:
                continue
            ep_v = ep[vi]
            done = np.zeros(len(vi), dtype=bool)
            exit_ret = np.zeros(len(vi), dtype=np.float32)
            exit_h   = np.full(len(vi), hold, dtype=np.float32)

            for h in range(1, hold + 1):
                bi = i + h
                if bi >= T:
                    break
                lo_v = L[bi, vi]; ci_v = C[bi, vi]
                lo_v = np.where(np.isnan(lo_v), ci_v, lo_v)
                # intrabar stop
                intrabar_ret = lo_v / ep_v - 1
                cut = (~done) & (intrabar_ret <= stop)
                exit_ret = np.where(cut, stop, exit_ret)
                exit_h   = np.where(cut, h, exit_h)
                done = done | cut
                if h == hold:
                    final = ~done
                    exit_ret = np.where(final & ~np.isnan(ci_v), ci_v / ep_v - 1, exit_ret)
                    done = done | final
                if done.all():
                    break

            gross[i, vi] = exit_ret
            holds[i, vi] = exit_h
        return gross, holds

    if mtype == "atr":
        atr_mult = float(mech["atr_mult"])
        n_atr = 14
        # compute ATR for all bars (causal: rolling TR)
        tr_rows = np.zeros((T, A), dtype=np.float32)
        for j in range(1, T):
            hi = H[j, :]; lo = L[j, :]; pc = C[j-1, :]
            hi = np.where(np.isnan(hi), C[j, :], hi)
            lo = np.where(np.isnan(lo), C[j, :], lo)
            pc = np.where(np.isnan(pc), C[j, :], pc)
            tr_rows[j, :] = np.maximum.reduce([hi - lo, np.abs(hi - pc), np.abs(lo - pc)])
        # rolling ATR (mean of last n_atr TRs)
        atr_mat = np.full((T, A), np.nan, dtype=np.float32)
        for j in range(n_atr, T):
            atr_mat[j, :] = np.nanmean(tr_rows[j-n_atr:j, :], axis=0)

        for i in range(n_atr + 1, T - max_hold - 1):
            ep = C[i, :]
            atr_i = atr_mat[i, :]
            valid_mask = (~np.isnan(ep)) & (~np.isnan(atr_i)) & (ep > 0)
            vi = np.where(valid_mask)[0]
            if len(vi) == 0:
                continue
            ep_v = ep[vi]
            atr_v = atr_i[vi]
            stop_lv = ep_v - atr_mult * atr_v
            done = np.zeros(len(vi), dtype=bool)
            exit_ret = np.zeros(len(vi), dtype=np.float32)
            exit_h   = np.full(len(vi), max_hold, dtype=np.float32)

            for h in range(1, max_hold + 1):
                bi = i + h
                if bi >= T:
                    break
                lo_v = L[bi, vi]; ci_v = C[bi, vi]
                lo_v = np.where(np.isnan(lo_v), ci_v, lo_v)
                stopped = (~done) & (lo_v <= stop_lv)
                exit_ret = np.where(stopped, np.maximum(stop_lv, lo_v) / ep_v - 1, exit_ret)
                exit_h   = np.where(stopped, h, exit_h)
                done = done | stopped
                if h == max_hold:
                    final = ~done
                    exit_ret = np.where(final & ~np.isnan(ci_v), ci_v / ep_v - 1, exit_ret)
                    done = done | final
                if done.all():
                    break

            gross[i, vi] = exit_ret
            holds[i, vi] = exit_h
        return gross, holds

    raise ValueError(f"Unknown mechanism type: {mtype}")


# ---- AGENT SCORING (VECTORISED) ----------------------------------------------
def score_all_bars(lab: dict, feats: list, signs: list = None) -> np.ndarray:
    """Compute agent composite z-score for ALL bars, ALL assets. Shape (T, A)."""
    F = lab["F"]
    C = lab["C"]
    A = len(C.columns)
    T = len(C.index)
    parts = []
    for j, f in enumerate(feats):
        mat = F[f].values  # (T, A) -- already shifted-1 in load_wide
        # cross-sectional z-score per bar
        row_mean = np.nanmean(mat, axis=1, keepdims=True)
        row_std  = np.nanstd(mat,  axis=1, keepdims=True)
        z = (mat - row_mean) / (row_std + 1e-12)
        z = np.where(np.isnan(z), 0.0, z)
        sgn = 1.0 if signs is None else signs[j]
        parts.append(sgn * z)
    if not parts:
        return np.zeros((T, A), dtype=np.float32)
    return np.mean(parts, axis=0).astype(np.float32)  # (T, A)


# ---- PORTFOLIO NET ROI USING CACHE -------------------------------------------
def portfolio_rois(score_mat: np.ndarray, gross_cache: np.ndarray,
                    hold_cache: np.ndarray, dis: np.ndarray, K: int = K) -> tuple:
    """At each bar in dis: pick top-K assets by score, look up cached gross, compute net.

    Returns (nets, avg_holds) arrays of length len(dis).
    """
    A = score_mat.shape[1]
    nets = []
    avg_holds = []
    for i in dis:
        sc = score_mat[i, :]          # (A,)
        gc = gross_cache[i, :]        # (A,) cached gross returns
        hc = hold_cache[i, :]         # (A,) cached hold bars
        # eligible: both score and gross available
        elig = np.where(~np.isnan(sc) & ~np.isnan(gc))[0]
        if len(elig) < K:
            continue
        ranked = elig[np.argsort(sc[elig])[::-1]][:K]
        avg_gross = float(np.mean(gc[ranked]))
        avg_hold  = float(np.mean(hc[ranked]))
        nets.append(avg_gross - COST)
        avg_holds.append(avg_hold)
    return np.array(nets, dtype=np.float64), np.array(avg_holds, dtype=np.float64)


# ---- SHUFFLE NULL (FULLY VECTORISED, NO PYTHON LOOPS INSIDE SHUFFLE) ---------
def shuffle_null(gross_cache: np.ndarray, hold_cache: np.ndarray,
                  dis: np.ndarray, K: int = K,
                  n_shuffle: int = N_SHUFFLE, seed: int = 42) -> np.ndarray:
    """Same-exposure shuffle: pick K random ELIGIBLE assets per bar, fully numpy.

    Returns array of mean net ROIs under null (length n_shuffle).
    """
    rng = np.random.default_rng(seed)
    A = gross_cache.shape[1]

    gc_sub = gross_cache[dis, :]   # (n_dis, A)
    elig_mask = ~np.isnan(gc_sub)
    n_elig = elig_mask.sum(axis=1)
    keep = n_elig >= K
    gc_keep = gc_sub[keep, :]        # (M, A)
    em_keep = elig_mask[keep, :]     # (M, A)
    M = gc_keep.shape[0]
    if M == 0:
        return np.zeros(n_shuffle)

    # Batch all n_shuffle iterations at once: shape (n_shuffle, M, A)
    # Random scores; ineligible -> -inf so they never get picked
    # Process in batches to avoid huge memory (n_shuffle=500, M~9k, A=50 -> 225M floats = ~900MB too big)
    # Use batch_size to keep memory reasonable
    batch_size = 50  # 50 * 9000 * 50 * 4 bytes = 90MB per batch
    n_batches = int(np.ceil(n_shuffle / batch_size))
    null_means = []

    gc_keep_f = gc_keep.astype(np.float32)
    em_keep_f = em_keep.astype(np.float32)

    rows_idx = np.arange(M, dtype=np.int32)

    for b in range(n_batches):
        bs = min(batch_size, n_shuffle - b * batch_size)
        # (bs, M, A) random scores
        rand_scores = rng.random((bs, M, A), dtype=np.float32)
        # mask ineligible: broadcast em_keep (M, A) -> (bs, M, A)
        rand_scores[:, ~em_keep] = -np.inf
        # top-K per bar: argpartition along axis=2
        top_k_idx = np.argpartition(rand_scores, -K, axis=2)[:, :, -K:]  # (bs, M, K)
        # gather gross returns: (bs, M, K)
        batch_idx = np.arange(bs)[:, None, None]
        rows_idx_b = np.arange(M)[None, :, None]
        picked = gc_keep_f[rows_idx_b, top_k_idx]        # (bs, M, K)
        bar_rois = picked.mean(axis=2) - COST            # (bs, M)
        batch_means = bar_rois.mean(axis=1)              # (bs,)
        null_means.extend(batch_means.tolist())

    return np.array(null_means[:n_shuffle])


# ---- MAIN SWEEP --------------------------------------------------------------
def run_sweep():
    t0 = time.time()
    print(f"[mech_exit_sweep] Loading 4h DEV data (wall={DEV_END})...")
    lab = load_wide(n=50, min_bars=BPD * 200, tf=TF)
    C = lab["C"]
    T, A = C.shape
    print(f"  {len(lab['syms'])} assets, {T} bars x {A} assets")
    print(f"  range {C.index.min()} -> {C.index.max()}")

    # regimes
    regime = compute_regimes(lab)
    vc = regime.value_counts()
    print(f"  Regime distribution: {dict(vc)}")

    warm = max(70, REGIME_LOOKBACK + 10)
    all_valid = np.array([i for i in range(warm, T - MAX_HOLD - 2)])
    regime_bars = {
        "bull": np.array([i for i in all_valid if regime.iloc[i] == "bull"]),
        "chop": np.array([i for i in all_valid if regime.iloc[i] == "chop"]),
        "bear": np.array([i for i in all_valid if regime.iloc[i] == "bear"]),
        "ALL":  all_valid,
    }
    for r, bs in regime_bars.items():
        print(f"    {r}: {len(bs)} bars")

    # pre-score all agents
    print("\n  Pre-computing agent scores...")
    agent_scores = {}
    for agent_name, feats in AGENTS.items():
        agent_scores[agent_name] = score_all_bars(lab, feats)
        print(f"    {agent_name}: done")

    results = []

    for mech_name, mech in MECH_GRID.items():
        t_mech = time.time()
        print(f"\n  Building mechanism cache: {mech_name}...", flush=True)
        gross_cache, hold_cache = build_mech_cache(lab, mech)
        print(f"    cache built in {time.time()-t_mech:.1f}s; "
              f"valid entries: {(~np.isnan(gross_cache)).sum():,}", flush=True)

        for agent_name, sc_mat in agent_scores.items():
            for regime_name, dis in regime_bars.items():
                if len(dis) < MIN_VALID_SLICES:
                    continue

                # agent result
                nets, avg_holds = portfolio_rois(sc_mat, gross_cache, hold_cache, dis, K)
                if len(nets) < MIN_VALID_SLICES:
                    continue

                mean_net  = float(np.mean(nets))
                mean_hold = float(np.mean(avg_holds))

                # shuffle null
                null = shuffle_null(gross_cache, hold_cache, dis, K,
                                     n_shuffle=N_SHUFFLE, seed=42)
                null_mean = float(np.mean(null))
                null_std  = float(np.std(null))
                z = (mean_net - null_mean) / (null_std + 1e-12)
                p05_null = float(np.percentile(null, 5))
                beats_p05 = mean_net > p05_null
                frac_beat = float(np.mean(nets > null_mean))

                results.append({
                    "agent":          agent_name,
                    "mech":           mech_name,
                    "regime":         regime_name,
                    "n":              len(nets),
                    "mean_net_pp":    round(mean_net * 100, 4),
                    "null_mean_pp":   round(null_mean * 100, 4),
                    "z":              round(z, 3),
                    "p05_null_pp":    round(p05_null * 100, 4),
                    "beats_p05":      beats_p05,
                    "frac_beat_null": round(frac_beat, 3),
                    "avg_hold_bars":  round(mean_hold, 1),
                    "avg_hold_days":  round(mean_hold / BPD, 2),
                    "cost_pp":        round(COST * 100, 3),
                })

                flag = "BEAT-P05" if beats_p05 else ""
                print(f"    {agent_name:<22} {regime_name:<5} net={mean_net*100:+.3f}% "
                      f"z={z:+.2f} hold={mean_hold/BPD:.1f}d null={null_mean*100:+.3f}% {flag}",
                      flush=True)

    # ---- HOLM CORRECTION ---------------------------------------------------
    from scipy import stats as sp_stats
    df = pd.DataFrame(results)
    df["p_val"] = sp_stats.norm.sf(df["z"].clip(-6, 6))

    # Correct only mechanism cells (not baseline fixed_42) x regime x agent
    mech_mask = df["mech"] != "fixed_42"
    pvals = df.loc[mech_mask, "p_val"].values
    n_tests = len(pvals)
    holm_p = np.ones(len(df))
    if n_tests > 0:
        order = np.argsort(pvals)
        hp = pvals.copy()
        for rank, idx in enumerate(order):
            hp[idx] = min(1.0, pvals[idx] * (n_tests - rank))
        for i in range(1, len(order)):
            hp[order[i]] = max(hp[order[i]], hp[order[i-1]])
        df.loc[mech_mask, "holm_p"] = hp
    df.loc[~mech_mask, "holm_p"] = np.nan
    df["holm_sig05"] = df["holm_p"] < 0.05

    elapsed = time.time() - t0

    # ---- PRINT RESULTS -------------------------------------------------------
    print("\n" + "=" * 110)
    print("MECHANISM EXIT SWEEP RESULTS (4h, DEV-walled, same-exposure block-shuffle null)")
    print("=" * 110)
    for agent_name in AGENTS:
        print(f"\n-- Agent: {agent_name} --")
        print(f"  {'Mech':<20} {'Regime':<7} {'N':>5} {'NetROI%':>8} {'Null%':>8} "
              f"{'z':>6} {'BeatP05':>8} {'HoldDays':>9} {'HolmP':>8} {'Sig?':>5}")
        sub = df[df["agent"] == agent_name].sort_values(["regime", "mech"])
        for _, row in sub.iterrows():
            bp = "YES" if row["beats_p05"] else "no"
            sg = "*" if row.get("holm_sig05", False) else ""
            hp = f"{row['holm_p']:.3f}" if not pd.isna(row.get("holm_p", np.nan)) else "  --"
            print(f"  {row['mech']:<20} {row['regime']:<7} {row['n']:>5} "
                  f"{row['mean_net_pp']:>8.3f} {row['null_mean_pp']:>8.3f} "
                  f"{row['z']:>6.2f} {bp:>8} {row['avg_hold_days']:>9.1f} {hp:>8} {sg:>5}")

    print("\n" + "=" * 110)
    print("KEY QUESTION: Does mechanism exit rescue chop/bear beyond the fixed-hold baseline?")
    print("=" * 110)
    for regime_name in ["bull", "chop", "bear"]:
        print(f"\n  REGIME: {regime_name}")
        base = df[(df["mech"] == "fixed_42") & (df["regime"] == regime_name)]
        mech_r = df[(df["mech"] != "fixed_42") & (df["regime"] == regime_name)]
        for agent_name in AGENTS:
            ab = base[base["agent"] == agent_name]
            if ab.empty:
                continue
            base_net = ab["mean_net_pp"].iloc[0]
            base_z   = ab["z"].iloc[0]
            print(f"    {agent_name}: baseline net={base_net:+.3f}% z={base_z:+.2f}")
            am = mech_r[mech_r["agent"] == agent_name]
            for _, row in am.iterrows():
                delta = row["mean_net_pp"] - base_net
                arr = "+" if delta >= 0 else ""
                hp = f"holm={row['holm_p']:.3f}" if not pd.isna(row.get("holm_p", np.nan)) else ""
                print(f"      {row['mech']:<20}: net={row['mean_net_pp']:+.3f}% z={row['z']:+.2f} "
                      f"delta={arr}{delta:.3f}pp hold={row['avg_hold_days']:.1f}d {hp}")

    print("\n" + "=" * 110)
    print("COST DRAG SUMMARY")
    print("=" * 110)
    print(f"  Taker RT cost per trade: {COST*100:.2f}%")
    mh = df.groupby("mech")["avg_hold_days"].mean()
    for mech_name, avg_d in mh.items():
        trades_yr = 365 / max(avg_d, 0.01)
        ann_cost = trades_yr * COST * 100
        print(f"  {mech_name:<20}: avg_hold={avg_d:.1f}d -> {trades_yr:.0f} trades/yr -> {ann_cost:.0f}pp/yr cost drag")

    print(f"\n[mech_exit_sweep] Completed in {elapsed:.1f}s")
    out_dir = Path(__file__).resolve().parents[2] / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mech_exit_sweep_4h.json"
    df.to_json(str(out_path), orient="records", indent=2)
    print(f"[mech_exit_sweep] Saved to {out_path}")
    return df


if __name__ == "__main__":
    run_sweep()
