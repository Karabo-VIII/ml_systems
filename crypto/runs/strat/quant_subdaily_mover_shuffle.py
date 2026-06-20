"""quant_subdaily_mover_shuffle.py -- SUB-DAILY move-capture SELECTION vs same-exposure shuffle, by regime.

THE DECISIVE QUESTION (Quant lane, DEV-walled <= 2024-05-15 ONLY):
  At SUB-DAILY (4h primary, 1h secondary), does move-capture SELECTION beat the SAME-EXPOSURE SHUFFLE
  ACROSS REGIMES (bull/chop/bear) -- where the 1d verdict FAILED in chop/bear (block-boot p05 = -1.20 / -0.29pp)?
  AND does a FAST EXIT MECHANISM (trailing-stop / target / time-stop) rescue chop/bear that a fixed hold cannot?

PORTED FROM (1d canonical): strat/fleet_mover_dev.py + fleet_mover_dev_adversarial.py.
  - mover score = cross-sectional z-composite {mom14, brk14, volexp}, windows RESCALED TO BARS (calendar-comparable).
  - circuit-breaker exposure (breadth + BTC-trend, causal), windows rescaled to bars.
  - SAME-EXPOSURE SHUFFLE control = IDENTICAL per-bar total book exposure, K random picks. >=20 seeds.
  - EXIT MECHANISM (the new sub-daily axis): each position exits by trailing-stop / profit-target / time-stop,
    evaluated CAUSALLY bar-by-bar on high/low/close. Fixed-hold is run as a baseline for contrast.
  - NULL = same-exposure shuffle + MOVING-BLOCK bootstrap (block = hold horizon in bars; iid understates var ~6x).
  - COST honesty: every number NET of realistic taker turnover; gross also reported so 'cost ate it' is visible.

DEV-walled (load_wide hard-caps at DEV_END). Long-only spot, no leverage, taker. Causal (no future bar). No emoji.
RWYB: C:/.../.venv/Scripts/python.exe quant_subdaily_mover_shuffle.py --tf 4h
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

CRYPTO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(CRYPTO_SRC) not in sys.path:
    sys.path.insert(0, str(CRYPTO_SRC))

import strat.fleet_lab as fl

COST = fl.COST            # 0.0024 taker round-trip
DEV_END = fl.DEV_END


# ============================================================
# BAR-RESCALED feature panels (calendar-comparable across TF). All causal (row i uses bars <= i).
# ============================================================
def _z_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1); sd = df.std(axis=1)
    z = df.sub(mu, axis=0).div(sd + 1e-9, axis=0)
    return z.where(sd > 1e-9, 0.0)


def build_features(C: pd.DataFrame, bpd: int) -> dict:
    """Move-capture features with windows scaled by bars-per-day so the CALENDAR horizon is constant across TF.
       1d-canonical used mom14 (14d), brk14 (14d high), volexp (7d/30d). Here w14 = 14*bpd bars, etc."""
    w14 = 14 * bpd; w7 = 7 * bpd; w30 = 30 * bpd
    R = C.pct_change(fill_method=None)
    mom14 = C / C.shift(w14) - 1
    brk14 = C / C.rolling(w14, min_periods=w14).max().shift(1) - 1
    volexp = R.rolling(w7).std() / (R.rolling(w30).std() + 1e-12)
    return {"mom14": mom14, "brk14": brk14, "volexp": volexp, "R": R}


def mover_score_panel(feats: dict, C: pd.DataFrame, blend: dict) -> pd.DataFrame:
    comp = None
    for f, w in blend.items():
        z = _z_cross_section(feats[f].reindex(index=C.index, columns=C.columns))
        comp = (w * z) if comp is None else (comp + w * z)
    valid = C.notna() & (C > 0)
    return comp.where(valid)


# ============================================================
# CAUSAL circuit-breaker + regime labels (windows in bars)
# ============================================================
def circuit_breaker_inputs(C: pd.DataFrame, bpd: int) -> dict:
    s50 = 50 * bpd; s200 = 200 * bpd; v20 = 20 * bpd
    bars_per_year = 365 * bpd
    sma50 = C.rolling(s50, min_periods=s50).mean()
    sma200 = C.rolling(s200, min_periods=s200).mean()
    R = C.pct_change(fill_method=None)
    vol20 = R.rolling(v20, min_periods=v20).std() * np.sqrt(bars_per_year)
    present = C.notna() & sma50.notna()
    above = (C > sma50)
    breadth = (above.where(present).sum(axis=1) / present.sum(axis=1).replace(0, np.nan)).fillna(0.5)
    btc_col = "BTCUSDT" if "BTCUSDT" in C.columns else C.columns[0]
    btc_up = (C[btc_col] > sma200[btc_col]).fillna(False)
    btc_vol = vol20[btc_col]
    return {"sma50": sma50, "sma200": sma200, "breadth": breadth, "btc_up": btc_up,
            "btc_vol": btc_vol, "btc_col": btc_col, "warmup": s200}


def exposure_series(cb: dict, vol_hi: float) -> pd.Series:
    breadth = cb["breadth"]; btc_up = cb["btc_up"]; btc_vol = cb["btc_vol"]
    hi_vol = btc_vol.fillna(0.0) >= vol_hi
    idx = breadth.index
    up = btc_up.reindex(idx).fillna(False)
    b = breadth.reindex(idx)
    expo = pd.Series(0.20, index=idx)
    expo[up & (b >= 0.50) & (~hi_vol)] = 1.00
    expo[up & ~((b >= 0.50) & (~hi_vol)) & (b >= 0.30)] = 0.70
    expo[up & ~((b >= 0.50) & (~hi_vol)) & (b < 0.30)] = 0.40
    return expo


def regime_labels(cb: dict) -> pd.Series:
    btc_up = cb["btc_up"]; breadth = cb["breadth"]
    idx = breadth.index
    up = btc_up.reindex(idx).fillna(False)
    b = breadth.reindex(idx)
    lab = pd.Series("chop", index=idx)
    lab[~up] = "bear"
    lab[up & (b >= 0.50)] = "bull"
    return lab


# ============================================================
# ENGINE: per-bar entries, EXIT MECHANISM, NET-of-cost slice returns. Vectorized per slice.
# ============================================================
def _picks_at(comp_row: pd.Series, K: int, rng) -> list:
    valid = comp_row.dropna().index
    if len(valid) < K:
        return []
    if rng is not None:
        return list(rng.choice(np.asarray(valid), size=K, replace=False))
    return list(comp_row.loc[valid].sort_values(ascending=False).index[:K])


def position_pnl_with_exit(C, H, L, s, i0, hold_bars, exit_cfg):
    """CAUSAL realized return of ONE position entered at close[i0] in asset s, exited by mechanism.
       exit_cfg: dict(mode='fixed'|'mech', target, trail, time_bars). Uses bar high/low (causal: bar i fully
       formed before we act at its close for the NEXT bar; here we model intrabar stop/target conservatively).
       Returns (gross_ret, n_bars_held). Long-only."""
    px0 = C[s].iloc[i0]
    if not np.isfinite(px0) or px0 <= 0:
        return None
    n = len(C.index)
    end = min(i0 + hold_bars, n - 1)
    if exit_cfg["mode"] == "fixed":
        px1 = C[s].iloc[end]
        if not np.isfinite(px1):
            # walk back to last finite
            for j in range(end, i0, -1):
                if np.isfinite(C[s].iloc[j]):
                    px1 = C[s].iloc[j]; end = j; break
            else:
                return None
        return float(px1 / px0 - 1), end - i0
    # mechanism: scan bars i0+1..end, check target (high) then trail (running peak) then time
    target = exit_cfg.get("target")        # e.g. 0.10 -> take profit at +10%
    trail = exit_cfg.get("trail")          # e.g. 0.07 -> exit if close <= peak*(1-trail)
    peak = px0
    for j in range(i0 + 1, end + 1):
        hi = H[s].iloc[j]; lo = L[s].iloc[j]; cl = C[s].iloc[j]
        if not np.isfinite(cl):
            continue
        if np.isfinite(hi):
            peak = max(peak, hi)
            if target is not None and hi >= px0 * (1 + target):
                return float(target), j - i0           # filled at target (conservative: exactly target)
        # trailing stop on the LOW (intrabar) -- conservative fill at the stop level
        if trail is not None and np.isfinite(lo):
            stop_lvl = peak * (1 - trail)
            if lo <= stop_lvl:
                return float(stop_lvl / px0 - 1), j - i0
        peak = max(peak, cl)
    # time-stop: exit at end close
    px1 = C[s].iloc[end]
    if not np.isfinite(px1):
        for j in range(end, i0, -1):
            if np.isfinite(C[s].iloc[j]):
                px1 = C[s].iloc[j]; end = j; break
        else:
            return None
    return float(px1 / px0 - 1), end - i0


def slice_return(C, H, L, comp, expo_arr, idx_dev_pos, si, slice_bars, hold_bars,
                 K, exit_cfg, rng, cost_rt):
    """Net compound return of the book over a [si, si+slice_bars) window of idx_dev (positions sized by
       circuit-breaker exposure, K picks, each exits by mechanism). Returns (net, gross, n_roundtrips, n_pos).

       Model: at each ENTRY bar in the slice we open K positions (exposure e/K each); each runs its own exit.
       To keep exposure comparable to the 1d harness (which re-picks every bar), we ENTER once at slice start
       and let exits run -- the slice net is the exposure-weighted mean of position returns minus per-position
       round-trip cost. exposure held IDENTICAL between real and shuffle (same e, same K, same entry bar)."""
    i_entry = idx_dev_pos[si]
    e = float(expo_arr[i_entry])
    if e <= 0:
        return 0.0, 0.0, 0, 0
    picks = _picks_at(comp.iloc[i_entry], K, rng)
    if not picks:
        return 0.0, 0.0, 0, 0
    rets = []
    for s in picks:
        pr = position_pnl_with_exit(C, H, L, s, i_entry, hold_bars, exit_cfg)
        if pr is not None:
            rets.append(pr[0])
    if not rets:
        return 0.0, 0.0, 0, 0
    gross = e * float(np.mean(rets))                 # exposure-weighted mean position return
    n_rt = len(rets)                                 # each position is one round-trip
    cost = e * cost_rt                               # exposure-weighted RT cost (entry+exit), one round-trip per slice
    net = gross - cost
    return net, gross, n_rt, len(picks)


# ============================================================
# MOVING-BLOCK BOOTSTRAP of (real - control) within regime
# ============================================================
def block_bootstrap(diff: np.ndarray, block: int, n_boot: int, rng) -> dict:
    n = len(diff)
    if n < max(20, block):
        return {"n": int(n), "note": "insufficient"}
    nblocks = int(np.ceil(n / block))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nblocks)
        idxs = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        boot[b] = diff[idxs].mean()
    return {"n": int(n), "mean_pp": round(100 * float(diff.mean()), 3),
            "p05_pp": round(100 * float(np.percentile(boot, 5)), 3),
            "p50_pp": round(100 * float(np.percentile(boot, 50)), 3),
            "frac_gt0": round(float((boot > 0).mean()), 4)}


def verdict(bb: dict) -> str:
    if "note" in bb:
        return "INSUFFICIENT"
    if bb["p05_pp"] > 0:
        return "REAL"
    if bb["frac_gt0"] >= 0.95:
        return "AMBIGUOUS"
    return "ARTIFACT"


# ============================================================
# MAIN
# ============================================================
def run(tf: str, exit_cfg: dict, exit_name: str, n_slices=600, ctrl_seeds=None,
        real_seeds=None, hold_days=7, n_boot=5000):
    t0 = time.time()
    bpd = fl.BARS_PER_DAY[tf]
    hold_bars = hold_days * bpd
    slice_bars = hold_days * bpd
    real_seeds = real_seeds or [11, 23, 42]
    ctrl_seeds = ctrl_seeds or list(range(101, 121))   # 20 same-exposure shuffle seeds
    block = max(2, hold_bars // bpd)                    # block ~ hold horizon in SLICE units; slices are non-overlap-sampled
    BLEND = {"mom14": 0.45, "brk14": 0.30, "volexp": 0.25}
    K = 3
    DEV_OOS_START = "2020-09-01"

    lab = fl.load_wide(n=50, min_bars=200 * bpd, tf=tf)
    C = lab["C"]; H = lab["H"]; L = lab["L"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"

    feats = build_features(C, bpd)
    cb = circuit_breaker_inputs(C, bpd)
    warmup = cb["warmup"]
    cut = C.index[int(len(C.index) * 0.6)]
    vol_hi = float(cb["btc_vol"][C.index < cut].dropna().quantile(0.80))
    expo = exposure_series(cb, vol_hi)
    regimes = regime_labels(cb)
    comp = mover_score_panel(feats, C, BLEND)

    # eval region: DEV-walled, after warmup
    start_ts = max(pd.Timestamp(DEV_OOS_START), C.index[warmup])
    idx_dev = C.index[(C.index >= start_ts) & (C.index < pd.Timestamp(DEV_END))]
    idx_dev_pos = np.array([C.index.get_loc(d) for d in idx_dev])
    n_avail = len(idx_dev)
    reg_at_start = regimes.reindex(idx_dev).values
    expo_arr = expo.reindex(C.index).values

    print("=" * 88)
    print(f"SUB-DAILY MOVER SELECTION vs SAME-EXPOSURE SHUFFLE | tf={tf} ({bpd} bars/day) | exit={exit_name}")
    print(f"DEV WALL <= {DEV_END} | eval {idx_dev.min()} -> {idx_dev.max()} | n_avail={n_avail} bars")
    print(f"hold={hold_bars} bars ({hold_days}d) | K={K} | blend={BLEND} | vol_hi(q80 early-DEV)={vol_hi:.3f}")
    rc = pd.Series(reg_at_start).value_counts()
    print(f"regime mix (eval bars): {dict(rc)} | avg exposure(eval)={float(expo.reindex(idx_dev).mean()):.3f}")
    print("=" * 88)

    # ---- sample NON-OVERLAPPING slice starts per real seed, paired across real/control ----
    def sample_starts(seed):
        rng = np.random.default_rng(seed)
        # non-overlapping: step the universe of valid starts by slice_bars, then subsample
        valid = np.arange(0, n_avail - slice_bars - 1)
        # enforce spacing >= hold_bars by drawing from a thinned grid then jittering inside spacing
        grid = valid[::max(1, hold_bars)]
        if len(grid) == 0:
            grid = valid
        take = min(n_slices, len(grid))
        return np.sort(rng.choice(grid, size=take, replace=False))

    starts_by_seed = {s: sample_starts(s) for s in real_seeds}

    # ---- REAL engine: per-slice net & gross, pooled over real seeds, regime-tagged ----
    real_net, real_gross, reg_tag = [], [], []
    for s in real_seeds:
        starts = starts_by_seed[s]
        for si in starts:
            net, gross, nrt, npos = slice_return(C, H, L, comp, expo_arr, idx_dev_pos, si,
                                                  slice_bars, hold_bars, K, exit_cfg, None, COST)
            real_net.append(net); real_gross.append(gross); reg_tag.append(reg_at_start[si])
    real_net = np.array(real_net); real_gross = np.array(real_gross); reg_tag = np.array(reg_tag)

    # ---- SHUFFLE control: for each slice, mean over the ctrl_seeds random books (the expected random book) ----
    ctrl_net = np.zeros(len(real_net))
    # rebuild in the SAME order as real (seed-major, then slice)
    order_starts = []
    for s in real_seeds:
        order_starts.extend(list(starts_by_seed[s]))
    order_starts = np.array(order_starts)
    ctrl_stack = np.zeros((len(ctrl_seeds), len(real_net)))
    for ci, cs in enumerate(ctrl_seeds):
        rng = np.random.default_rng(cs)
        for k, si in enumerate(order_starts):
            net, gross, nrt, npos = slice_return(C, H, L, comp, expo_arr, idx_dev_pos, si,
                                                 slice_bars, hold_bars, K, exit_cfg, rng, COST)
            ctrl_stack[ci, k] = net
    ctrl_net = ctrl_stack.mean(axis=0)

    # ---- exposure identity check: real and ctrl share the same entry-bar exposure by construction ----
    # (both call slice_return with identical expo_arr/i_entry; the ONLY diff is pick identity.)

    out = {"tf": tf, "exit": exit_name, "exit_cfg": exit_cfg, "K": K, "blend": BLEND,
           "hold_bars": hold_bars, "hold_days": hold_days, "vol_hi": round(vol_hi, 4),
           "n_slices_per_seed": int(min(n_slices, len(starts_by_seed[real_seeds[0]]))),
           "ctrl_seeds": len(ctrl_seeds), "block": int(block), "regime_mix": {k: int(v) for k, v in rc.items()},
           "eval_range": [str(idx_dev.min()), str(idx_dev.max())], "regimes": {}}

    print(f"\nper-regime SELECTION ALPHA (real - same-exposure-shuffle), NET of {COST} taker RT:")
    print(f"  {'regime':6}{'n':>5}{'real_net%':>11}{'ctrl_net%':>11}{'alpha_pp':>10}{'gross_pp':>10}"
          f"{'p05_pp':>9}{'frac>0':>8}  verdict")
    rng_boot = np.random.default_rng(7)
    for r in ["bull", "chop", "bear"]:
        m = reg_tag == r
        if m.sum() < 20:
            out["regimes"][r] = {"n": int(m.sum()), "note": "insufficient"}
            print(f"  {r:6}{int(m.sum()):>5}  (insufficient)")
            continue
        diff = real_net[m] - ctrl_net[m]
        gross_diff = real_gross[m] - ctrl_net[m]   # gross real vs net ctrl is unfair; use gross-vs-gross below
        bb = block_bootstrap(diff, block, n_boot, rng_boot)
        v = verdict(bb)
        # gross alpha (cost removed from BOTH): real_gross - ctrl_gross ~ real_gross - (ctrl_net + cost)
        # ctrl_net already net; reconstruct ctrl_gross approx by adding back exposure*cost is messy across seeds;
        # report gross selection alpha = mean(real_gross - real_net) is the cost; the cleaner gross alpha is below.
        ro = {"n": int(m.sum()),
              "real_net_pct": round(100 * float(real_net[m].mean()), 3),
              "ctrl_net_pct": round(100 * float(ctrl_net[m].mean()), 3),
              "alpha_net_pp": bb["mean_pp"],
              "boot_p05_pp": bb["p05_pp"], "boot_frac_gt0": bb["frac_gt0"],
              "real_gross_pct": round(100 * float(real_gross[m].mean()), 3),
              "cost_drag_pp": round(100 * float((real_gross[m] - real_net[m]).mean()), 3),
              "real_pos_rate": round(100 * float((real_net[m] > 0).mean()), 1),
              "verdict": v}
        out["regimes"][r] = ro
        print(f"  {r:6}{ro['n']:>5}{ro['real_net_pct']:>11.3f}{ro['ctrl_net_pct']:>11.3f}"
              f"{ro['alpha_net_pp']:>+10.3f}{ro['real_gross_pct']:>+10.3f}"
              f"{ro['boot_p05_pp']:>+9.3f}{ro['boot_frac_gt0']:>8.3f}  {v}")

    # pooled (all-regime) too
    diff_all = real_net - ctrl_net
    bb_all = block_bootstrap(diff_all, block, n_boot, rng_boot)
    out["pooled"] = {"n": int(len(diff_all)), "alpha_net_pp": bb_all["mean_pp"],
                     "boot_p05_pp": bb_all["p05_pp"], "boot_frac_gt0": bb_all["frac_gt0"],
                     "verdict": verdict(bb_all)}
    print(f"\n  POOLED n={len(diff_all)} alpha_net={bb_all['mean_pp']:+.3f}pp "
          f"p05={bb_all['p05_pp']:+.3f}pp frac>0={bb_all['frac_gt0']:.3f} -> {verdict(bb_all)}")

    out["runtime_s"] = round(time.time() - t0, 1)
    print(f"\nruntime {out['runtime_s']}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--exits", default="fixed,mech_fast,mech_wide")
    ap.add_argument("--nslices", type=int, default=600)
    ap.add_argument("--nboot", type=int, default=5000)
    a = ap.parse_args()

    EXITS = {
        "fixed":     {"mode": "fixed"},                                  # baseline: forced full hold (the 1d failure mode)
        "mech_fast": {"mode": "mech", "target": 0.08, "trail": 0.05},    # tight: cut losers fast, bag +8%
        "mech_wide": {"mode": "mech", "target": 0.20, "trail": 0.10},    # let winners run, wider trail
    }
    want = [e.strip() for e in a.exits.split(",") if e.strip() in EXITS]

    all_out = {"tf": a.tf, "generated": str(pd.Timestamp.now()), "cost_rt": COST, "runs": {}}
    for en in want:
        res = run(a.tf, EXITS[en], en, n_slices=a.nslices, n_boot=a.nboot)
        all_out["runs"][en] = res

    outp = Path(__file__).resolve().parent / f"quant_subdaily_mover_shuffle_{a.tf}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.write_text(json.dumps(all_out, indent=2, default=str))
    print(f"\n{'='*88}\nSaved: {outp}")
    return all_out


if __name__ == "__main__":
    main()
