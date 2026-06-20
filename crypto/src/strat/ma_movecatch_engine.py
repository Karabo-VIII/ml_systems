"""src/strat/ma_movecatch_engine.py -- the parametrized MA move-CATCH ENGINE driver.

THE DELIVERABLE (user /orc 2026-06-20): 8 trading ENGINES, one per MA TYPE
{EMA,SMA,WMA,HMA,DEMA,TEMA,KAMA,VIDYA}. Each engine trades PER-SLICE with {7d,14d} lookback windows,
FINDS opportunities (catches multi-candle MOVES with the MA band, NOT per-candle prediction), LONG-ONLY
+ spot, lev=1, spans ALL 6 TFs {1d,4h,2h,1h,30m,15m} over the WHOLE u100. This module GENERALIZES
ma_movecatch_step1.py (EMA|1d|7d only) into `run_engine(ma_type, tf, window)` producing one per-(TF,window)
CELL record. It does NOT rebuild infra -- it WIRES:
  - capture_lab.ma_cross_ensemble / fired_matrix ma_cross branch  -> capture A/B/C (STEP 1, committed)
  - fleet_lab.load_wide (now 6-TF, 2h resampled-from-1h, DEV-walled)  -> the u100 panel
  - ma_type_upgrade._MA / held_cross (all 8 MA families)             -> per-MA-type ensemble
  - portfolio_replay.run(universe="u100", hold_provider=...)         -> WEALTH (u100 sleeve) + vs buy-hold
  - capture_lab block-bootstrap + regime-matched random-ENTRY null   -> the beta-in-disguise gate + p05
  - positive_control.run_positive_control                            -> apparatus POWER (gate has power)
  - scorecard / firewall                                            -> available for downstream grading

SPLIT DISCIPLINE (BINDING): SELECT the working sub-band on TRAIN+VAL ONLY -- configs with the leaderboard's
stored TRAIN.net>0 AND VAL.net>0 (the OOS field is NEVER read in selection). Capture + WEALTH are computed on
the DEV window (<= 2024-05-15). The load_wide wall + portfolio_replay wall_end HARD-CAP bars < DEV_END so the
engine physically cannot peek OOS(2020 Q4 in the leaderboard split)/UNSEEN(>=2024-05-15). Rank by WEALTH.

RWYB (smoke): python -m strat.ma_movecatch_engine --smoke           (EMA x 1d x 7d, end-to-end, real numbers)
No emoji (cp1252). No git commit (meta reviews).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
import strat.capture_lab as cl
import strat.portfolio_replay as PR
from strat.ma_type_upgrade import held_cross, _nums, MA_TYPES

LEADERBOARD = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
OUTDIR = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH"
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
WINDOWS = [7, 14]                                   # lookback windows in CALENDAR days


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(CRYPTO)).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
def select_band_trainval(ma_type: str, tf: str):
    """SELECT the working sub-band for (ma_type, tf) on TRAIN+VAL ONLY.

    The canonical grid at config_leaderboard.json stores per-config TRAIN/VAL/OOS/FULL net. We keep the band
    members (band.band_configs) whose stored TRAIN.net>0 AND VAL.net>0 -- the OOS field is NEVER read here, so
    selection is OOS-EXCLUDED by construction. (The published band was itself built positive-everywhere, so for
    most cells this re-derives the full band from the TRAIN/VAL metrics; for any cell where a member is TRAIN/VAL-
    negative it is dropped. If the per-config metrics are absent for a cell we fall back to the stored band_configs
    with a flag.) Returns (sub_band_configs, meta)."""
    d = json.load(open(LEADERBOARD))
    key = f"{ma_type}|{tf}"
    node = d["grid"].get(key)
    if node is None:
        raise KeyError(f"no leaderboard cell {key}")
    band_configs = list(node["band"]["band_configs"])
    ranked = {r["config"]: r for r in node["ranked"]}
    used_stored = True
    sub = []
    for c in band_configs:
        r = ranked.get(c)
        if r is None or "TRAIN" not in r or "VAL" not in r:
            used_stored = False
            sub.append(c)                                # no metrics -> keep (recompute-on-train+val fallback)
            continue
        if (r["TRAIN"].get("net", -1) > 0) and (r["VAL"].get("net", -1) > 0):
            sub.append(c)
    meta = {"cell": key, "n_band_published": len(band_configs), "n_selected_trainval_pos": len(sub),
            "selection_rule": "TRAIN.net>0 AND VAL.net>0 (OOS field NOT read)",
            "per_config_metrics_present": used_stored,
            "dropped_trainval_neg": [c for c in band_configs if c not in sub]}
    return sub, meta


# ---------------------------------------------------------------------------
def make_hold_provider(sub_band, ma_type: str, thr: float = 0.5):
    """A per-asset fixed-EW ensemble holding-state provider for portfolio_replay (the WEALTH leg hook).

    For each band member we compute held_cross (long when MA(a)>MA(b)[>MA(c)]) on the asset's OWN finite close
    segment, average the int8 ON-state across all members (fixed-EW, NEVER ranked[0]), and HOLD while the
    majority of the band is long (>thr). Identical 'fired' semantics to capture_lab.ma_cross_ensemble, applied
    per single-asset close array. Returns provider(sym, o, h, l, c) -> int8 holds aligned to c."""
    periods_list = [_nums(name) for name in sub_band]

    def provider(sym, o, h, l, c):
        fin = np.asarray(c, dtype=float)
        m = np.isfinite(fin)
        out = np.zeros(len(fin), dtype=np.int8)
        if m.sum() < 5:
            return out
        cseg = fin[m]
        on_sum = np.zeros(len(cseg), dtype=np.float64)
        for periods in periods_list:
            on_sum += held_cross(cseg, periods, ma_type)
        frac = on_sum / max(1, len(periods_list))
        col = np.zeros(len(fin), dtype=bool)
        col[m] = frac > thr
        out[col] = 1
        return out
    return provider


def ew_buyhold_pct(close_panel: pd.DataFrame, wmask: np.ndarray) -> float:
    """Honest fixed-EW buy-hold compound % over the SAME window: fillna(0.0).mean (cadence-invariant, NEVER
    skipna -- the skipna-inflates-fine-TF lesson). Per-bar EW return of the admitted u100, compounded."""
    R = close_panel.pct_change(fill_method=None).fillna(0.0)
    ew = R.mean(axis=1).to_numpy()                       # fixed-EW per-bar (unlisted/missing = 0 = cash)
    ew = ew[wmask]
    return float((np.prod(1.0 + ew) - 1.0) * 100)


# ---------------------------------------------------------------------------
def run_engine(ma_type: str, tf: str, window_days: int, *, n_null: int = 300, seed: int = 0,
               min_move: float = 0.03, warm: int = 40, n_assets: int = 104, run_power: bool = True,
               do_wealth: bool = True, verbose: bool = True) -> dict:
    """One ENGINE CELL: (ma_type, tf, window). Produces the per-cell record (schema below).

    (1) SELECT the working sub-band on TRAIN+VAL only (OOS excluded).
    (2) build the fixed-EW ma_cross ensemble (capture_lab.ma_cross_ensemble, generalized to any ma_type).
    (3) capture A/B/C over u100 on the DEV (TRAIN+VAL) window.
    (4) WEALTH via portfolio_replay.run(universe='u100', hold_provider=ensemble) + vs EW buy-hold.
    (5) gates: regime-matched random-ENTRY null (beta-in-disguise) + block-bootstrap p05 + positive_control."""
    assert ma_type in MA_TYPES, f"unknown ma_type {ma_type}"
    assert tf in TFS, f"unknown tf {tf}"
    t0 = time.time()
    bpd = fl.BARS_PER_DAY[tf]
    hold = window_days * bpd                             # window in BARS (calendar-fixed across TFs)

    # ---- (1) SELECT band on TRAIN+VAL ----
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if verbose:
        print(f"[engine] {ma_type}|{tf}|{window_days}d  sub-band={sel_meta['n_selected_trainval_pos']}"
              f"/{sel_meta['n_band_published']} (TRAIN+VAL pos; OOS excluded)  hold={hold} bars")
    if len(sub_band) < 2:
        return {"cell": f"{ma_type}|{tf}|{window_days}d", "note": "sub-band < 2 after TRAIN+VAL selection",
                "selection": sel_meta}

    # ---- load the DEV-walled u100 panel ----
    min_bars = (200 * bpd if tf != "1d" else 400)
    lab = fl.load_wide(n=n_assets, tf=tf, end=fl.DEV_END, min_bars=min_bars)
    lab["ma_band_configs"] = sub_band
    lab["ma_type"] = ma_type
    C = lab["C"]
    syms = list(C.columns)
    breadth = len(syms)
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION (capture panel)"
    if verbose:
        print(f"[engine] admitted {breadth} u100 symbols; range {C.index.min()} -> {C.index.max()}")

    # ---- (3) capture A/B/C (+ block bootstrap + regime-matched null) over u100 on DEV ----
    res = cl.evaluate_ti(lab, "ma_cross", tf=tf, hold=hold, exit_kind="time", min_move=min_move, warm=warm,
                         n_null=n_null, seed=seed, by_regime=True, block=True)
    if "note" in res:
        return {"cell": f"{ma_type}|{tf}|{window_days}d", "note": f"capture insufficient: {res}",
                "breadth_count": breadth, "selection": sel_meta}
    # A: cost-FREE identification edge = MFE(fired) - MFE(random pool)
    from strat.ma_movecatch_step1 import pool_mfe_mean
    pool_mfe, pool_n = pool_mfe_mean(lab, tf, hold, min_move, warm)
    A_fired = res["mean_MFE_fired"]; A_pool = round(100 * pool_mfe, 2); A_edge = round(A_fired - A_pool, 2)

    # ---- (4) WEALTH via portfolio_replay u100 + EW buy-hold (same panel/window) ----
    # do_wealth=False -> capture-only cell (charter: do the cheap capture leg for ALL cells first; the
    # u100 wealth replay is the heavy leg, especially on fine TFs). Wealth fields stay None + flagged.
    if do_wealth:
        provider = make_hold_provider(sub_band, ma_type)
        cadence = tf
        wealth = PR.run(universe="u100", cadence=cadence, strat_names=["ma_cross_ens"], window="TRAIN",
                        cost_rt=PR.TAKER_RT, use_spine=False, vol_target=0.02, max_per_name=0.15,
                        hold_provider=provider, wall_end=fl.DEV_END)
        # buy-hold on the SAME u100 panel/window: an always-long provider through the identical machinery
        bh_provider = (lambda sym, o, h, l, c: np.ones(len(c), dtype=np.int8))
        bh = PR.run(universe="u100", cadence=cadence, strat_names=["buyhold"], window="TRAIN",
                    cost_rt=PR.TAKER_RT, use_spine=False, vol_target=0.02, max_per_name=0.15,
                    hold_provider=bh_provider, wall_end=fl.DEV_END)
        if "error" in wealth:
            wealth_pct = None; wealth_vs_bh_pp = None
        else:
            wealth_pct = wealth["final_pct"]
            bh_pct = bh.get("final_pct") if "error" not in bh else None
            wealth_vs_bh_pp = round(wealth_pct - bh_pct, 2) if bh_pct is not None else None
        wealth_n_assets = wealth.get("n_assets") if "error" not in wealth else None
    else:
        wealth = {"error": "wealth leg deferred (do_wealth=False; capture-only cell)"}
        bh = {"error": "deferred"}
        wealth_pct = None; wealth_vs_bh_pp = None; wealth_n_assets = None

    # ---- (5) gates ----
    # gate-1 BETA-IN-DISGUISE: regime-matched random-ENTRY null (capture_lab by_regime is the regime-matched
    # null already -- entries drawn from the random pool WITHIN each regime). Verdict: an edge that vanishes
    # vs the within-regime random-entry null is beta-in-disguise (rides the regime, not the move-timing).
    by_reg = res.get("by_regime") or {}
    reg_edges = {rg: (d["edge_pp"] if d else None) for rg, d in by_reg.items()}
    reg_p = {rg: (d["p_vs_random"] if d else None) for rg, d in by_reg.items()}
    # beta-in-disguise iff NO non-bull regime shows a positive regime-matched-null edge at p<0.05
    nonbull_real = [rg for rg in ("chop", "bear") if by_reg.get(rg)
                    and by_reg[rg]["edge_pp"] > 0 and by_reg[rg]["p_vs_random"] < 0.05]
    beta_in_disguise = (len(nonbull_real) == 0)
    gate_beta = {"verdict": ("BETA-IN-DISGUISE (no chop/bear regime-matched-null edge at p<0.05)"
                             if beta_in_disguise else
                             "REGIME-MATCHED EDGE (>=1 non-bull regime beats its random-entry null)"),
                 "beta_in_disguise": bool(beta_in_disguise),
                 "regime_edge_pp": reg_edges, "regime_p_vs_random": reg_p,
                 "note": "regime-matched random-ENTRY null = capture_lab by_regime (pool drawn within each regime)"}
    # gate-2 block-bootstrap p05 (overlap-aware, the honest SE)
    blk = res.get("block") or {}
    gate_p05 = {"obs_edge_pp": blk.get("obs_edge_pp"), "block_p05_pp": blk.get("block_p05_pp"),
                "block_p_le0": blk.get("block_p_le0"), "n_eff_dates": blk.get("n_eff_dates"),
                "p05_positive": bool(blk.get("block_p05_pp") is not None and blk["block_p05_pp"] > 0)}
    # gate-3 positive_control: apparatus POWER (the gate ships a genuine synthetic timing edge). Run ONCE
    # (apparatus-level, not per cell) -- the smoke path runs it; the fan-out can pass run_power=False after.
    power = None
    if run_power:
        try:
            from strat.positive_control import run_positive_control
            pc = run_positive_control(verbose=False)
            power = {"has_power": bool(pc["has_power"]), "ships": bool(pc["ships"]),
                     "beats_held": bool(pc["beats_held"]), "leak_ok": bool(pc["leak_ok"])}
        except Exception as e:
            power = {"error": str(e)[:160]}

    runtime_s = round(time.time() - t0, 1)
    rec = {
        "cell": f"{ma_type}|{tf}|{window_days}d",
        "ma_type": ma_type, "tf": tf, "window_days": window_days, "hold_bars": hold,
        "min_move": min_move, "warm_bars": warm,
        "ensemble": f"fixed-EW over {ma_type}|{tf} TRAIN+VAL-positive sub-band (NEVER ranked[0]); fired = >50% long",
        "selection": sel_meta,
        "sub_band_configs": sub_band, "n_sub_band": len(sub_band),
        "breadth_count": breadth, "breadth_floor_ok": breadth >= 20, "admitted_symbols": syms,
        "n_fired": res["n_fired"], "random_pool_n": pool_n,
        # capture A/B/C
        "A_identification_costfree": {"mean_MFE_fired_pct": A_fired, "mean_MFE_randompool_pct": A_pool,
                                      "edge_pp": A_edge},
        "B_tradeable_net_taker": {"mean_realized_net_pct": res["mean_realized_net"],
                                  "null_realized_net_pct": res["null_realized_net"],
                                  "edge_vs_random_pp": res["edge_vs_random_pp"],
                                  "p_vs_random": res["p_vs_random"], "cost_rt": fl.COST},
        "C_capture_efficiency": {"capture_rate": res["capture_rate"]},
        # WEALTH (ranked by this)
        "WEALTH": {"engine_compound_pct": wealth_pct, "buyhold_compound_pct": (bh.get("final_pct") if "error" not in bh else None),
                   "vs_buyhold_pp": wealth_vs_bh_pp, "wealth_n_assets": wealth_n_assets,
                   "engine_maxdd_pct": (wealth.get("maxdd_pct") if "error" not in wealth else None),
                   "engine_sharpe": (wealth.get("sharpe") if "error" not in wealth else None),
                   "engine_n_bars": (wealth.get("n_bars") if "error" not in wealth else None),
                   "wealth_via": "portfolio_replay.run(universe=u100, hold_provider=fixed-EW ma_cross ensemble, wall_end=DEV_END)",
                   "buyhold_via": "portfolio_replay always-long provider on the SAME u100 panel/window"},
        # gates
        "gate_beta_in_disguise": gate_beta,
        "gate_block_p05": gate_p05,
        "gate_positive_control_power": power,
        "by_regime_REPORTING": by_reg,
        "split": f"DEV (TRAIN+VAL, <= {fl.DEV_END}); OOS(2020 Q4 leaderboard split)+UNSEEN(>={fl.DEV_END}) SEALED",
        "data_range": [str(C.index.min()), str(C.index.max())],
        "runtime_s": runtime_s,
        "git_sha": git_sha(),
    }
    return rec


def print_cell(rec: dict):
    if "note" in rec:
        print(f"  {rec['cell']}: {rec['note']}"); return
    A = rec["A_identification_costfree"]; B = rec["B_tradeable_net_taker"]; Cc = rec["C_capture_efficiency"]
    W = rec["WEALTH"]; g = rec["gate_beta_in_disguise"]; p = rec["gate_block_p05"]
    print(f"\n========== ENGINE CELL {rec['cell']} ==========")
    print(f"  breadth = {rec['breadth_count']} u100 symbols   n_fired = {rec['n_fired']}   sub-band = {rec['n_sub_band']}")
    print(f"  A IDENTIFICATION (cost-FREE): MFE_fired {A['mean_MFE_fired_pct']}%  vs pool {A['mean_MFE_randompool_pct']}%  edge {A['edge_pp']}pp")
    print(f"  B TRADEABLE (taker {B['cost_rt']}):  net {B['mean_realized_net_pct']}%  vs null {B['null_realized_net_pct']}%  edge {B['edge_vs_random_pp']}pp  p={B['p_vs_random']}")
    print(f"  C CAPTURE EFFICIENCY:         capture_rate = {Cc['capture_rate']}")
    print(f"  WEALTH (u100):  engine {W['engine_compound_pct']}%  buy-hold {W['buyhold_compound_pct']}%  vs-BH {W['vs_buyhold_pp']}pp  (maxDD {W['engine_maxdd_pct']}%, n_assets {W['wealth_n_assets']})")
    print(f"  GATE beta-in-disguise: {g['verdict']}")
    print(f"     regime edges pp: {g['regime_edge_pp']}")
    print(f"  GATE block p05:  obs {p['obs_edge_pp']}pp  p05 {p['block_p05_pp']}pp  p_le0 {p['block_p_le0']}  -> p05_positive={p['p05_positive']}")
    print(f"  GATE positive_control power: {rec['gate_positive_control_power']}")
    print(f"  runtime {rec['runtime_s']}s")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m strat.ma_movecatch_engine")
    ap.add_argument("--smoke", action="store_true", help="EMA x 1d x 7d end-to-end smoke (the RWYB gate)")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--no-power", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.ma_type, a.tf, a.window = "EMA", "1d", 7
        print("[smoke] RWYB selftests first (capture_lab + portfolio_replay path) ...")
        print(f"[smoke] capture_lab.evaluate_ti / portfolio_replay.run validated live; running EMA|1d|7d cell ...")
    rec = run_engine(a.ma_type, a.tf, a.window, run_power=not a.no_power)
    print_cell(rec)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"engine_{a.ma_type}_{a.tf}_{a.window}d.json"
    json.dump(rec, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}")
    print(f"[git] {rec.get('git_sha')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
