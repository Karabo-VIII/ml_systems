"""
experiments/adaptive_ma/oracle_dna_1d_u20_runner.py

1d ORACLE-DNA on a u20 subsample (first 20 of config/universes/u50.yaml).

REUSES (does NOT re-implement):
  * experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py :: run(), load_asset
      -> per-asset held-out MA-DNA AUC (1/2/3-MA distance/slope/gap/cross/ribbon features live in the
         chimera norm_/xd_ columns), SHUFFLED-LABEL control, positive control, regime firewall.
  * runs/research/oracle_ceiling_builder.py :: oracle_high_capture
      -> perfect-foresight high-capture oracle (hold<=7d, next-open entry, net taker 0.0024) for the
         ORACLE HOLD DISTRIBUTION (median hold in bars) per asset.

OUTPUT: experiments/adaptive_ma/oracle_dna_1d_u20.json
  per-asset + u20 aggregate: oracle hold distribution, held-out AUC (real vs shuffled p95),
  capture-rate (realized/available), realizable ceiling, and the apparatus/DNA verdicts.

RWYB: .venv/Scripts/python.exe experiments/adaptive_ma/oracle_dna_1d_u20_runner.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))
sys.path.insert(0, str(ROOT / "experiments" / "adaptive_ma" / "sol"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from oracle_ceiling_builder import oracle_high_capture, summarize, COST_RT  # noqa: E402
import oracle_dna_shuffled_falsifier as fal  # noqa: E402

CADENCE = "1d"
# first 20 of u50 (declarative u20 subsample; no u20.yaml exists)
U20 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ZEC", "TRX", "PEPE", "ADA",
       "LINK", "SUI", "AVAX", "TAO", "FET", "ENJ", "ORDI", "NEAR", "WLD", "ENA"]

N_SHUFFLE = 30      # shuffled-label control seeds (kept modest for <5min budget over 20 assets)
N_BOOKS = 200       # random-entry null draws for the regime firewall


def _dist(a):
    a = np.asarray([x for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def per_asset(asset):
    t0 = time.time()
    # ---- oracle hold distribution (reuse audited DP) ----
    ts, op, hi, cl, X, feats = fal.load_asset(asset, CADENCE)
    f_dp, trades = oracle_high_capture(ts, op, hi)
    if not trades:
        return {"asset": asset, "error": "no oracle trades"}
    holds = np.array([j - i for i, j in trades], dtype=int)          # hold in BARS (1d bars)
    s = summarize(ts, trades, op, hi)                                # oracle compound / per-move net

    # ---- MA-DNA decomposition + shuffled-label control + firewall (reuse falsifier run) ----
    dna = fal.run(asset=asset, cadence=CADENCE, n_shuffle=N_SHUFFLE, n_books=N_BOOKS, verbose=False)

    ho = dna["real"]["HELD_OUT"]
    capture_skill_ho = ho["capture_plain"]["skill"]                  # realized/available skill in [0..1]
    out = {
        "asset": asset, "n_bars": int(len(op)),
        # ORACLE HOLD DISTRIBUTION (bars)
        "oracle_n_trades": int(len(trades)),
        "oracle_hold_bars": {"median": float(np.median(holds)), "mean": float(holds.mean()),
                             "p25": float(np.percentile(holds, 25)), "p75": float(np.percentile(holds, 75)),
                             "min": int(holds.min()), "max": int(holds.max())},
        "oracle_compound_pct": s.get("total_capturable_compound_pct"),
        "oracle_median_hold_hours": s.get("median_hold_hours"),
        "oracle_base_rate": dna["oracle_base_rate"],
        "exit_H_bars": dna["exit_H_bars"],                          # = median hold used as exit horizon
        # MA-DNA held-out skill
        "held_out_auc_real": dna["real_held_out_auc"],
        "held_out_auc_shuffled_p95": dna["shuffled_control"]["auc"]["p95"],
        "held_out_auc_shuffled_mean": dna["shuffled_control"]["auc"]["mean"],
        "held_out_ic_fwd": dna["real_held_out_ic_fwd"],
        # CAPTURE-RATE + REALIZABLE CEILING (held-out)
        "capture_skill_heldout": capture_skill_ho,                  # 0=chance .. 1=perfect selection
        "dna_capture_compound_pct": ho["capture_plain"]["dna_compound_pct"],
        "best_capture_compound_pct": None,  # filled below from per-move ceiling on held-out
        "dna_mean_net_pct": ho["capture_plain"].get("dna_mean_net_pct"),
        "best_mean_net_pct": ho["capture_plain"].get("best_mean_net_pct"),
        "chance_mean_net_pct": ho["capture_plain"].get("chance_mean_net_pct"),
        # realizable ceiling = realizable mean-net per move if you select perfectly (best_mean), and the
        # realized fraction the DNA actually achieves of the (best - chance) gap = capture_skill.
        "realizable_ceiling_mean_net_pct": ho["capture_plain"].get("best_mean_net_pct"),
        "verdict": dna["VERDICT"],
        "secs": round(time.time() - t0, 1),
    }
    return out


def main():
    t_start = time.time()
    print(f"[1d ORACLE-DNA u20] assets={len(U20)}  n_shuffle={N_SHUFFLE}  n_books={N_BOOKS}  cost_rt={COST_RT}")
    results = []
    for a in U20:
        try:
            r = per_asset(a)
        except Exception as e:
            r = {"asset": a, "error": repr(e)[:200]}
        results.append(r)
        if "error" in r:
            print(f"  {a:6} ERROR: {r['error']}")
        else:
            print(f"  {a:6} bars={r['n_bars']:5} oracle_trades={r['oracle_n_trades']:4} "
                  f"hold_med={r['oracle_hold_bars']['median']:.1f}b "
                  f"AUC real={r['held_out_auc_real']:.3f} shufp95={r['held_out_auc_shuffled_p95']:.3f} "
                  f"cap_skill={r['capture_skill_heldout']:+.3f} "
                  f"DNA_genuine={r['verdict']['DNA_GENUINE_SIGNAL']} ({r['secs']}s)")

    ok = [r for r in results if "error" not in r]
    # ---- u20 aggregates ----
    agg = {
        "n_assets_ok": len(ok), "n_assets_err": len(results) - len(ok),
        "oracle_hold_median_bars": _dist([r["oracle_hold_bars"]["median"] for r in ok]),
        "oracle_base_rate": _dist([r["oracle_base_rate"] for r in ok]),
        "held_out_auc_real": _dist([r["held_out_auc_real"] for r in ok]),
        "held_out_auc_shuffled_p95": _dist([r["held_out_auc_shuffled_p95"] for r in ok]),
        "held_out_ic_fwd": _dist([r["held_out_ic_fwd"] for r in ok]),
        "capture_skill_heldout": _dist([r["capture_skill_heldout"] for r in ok]),
        "realizable_ceiling_mean_net_pct": _dist([r["realizable_ceiling_mean_net_pct"] for r in ok if r["realizable_ceiling_mean_net_pct"] is not None]),
        "dna_capture_compound_pct": _dist([r["dna_capture_compound_pct"] for r in ok]),
        # how many assets show GENUINE DNA (AUC beats shuffled AND capture beats firewall)
        "n_apparatus_sound": int(sum(1 for r in ok if r["verdict"]["APPARATUS_SOUND"])),
        "n_dna_genuine": int(sum(1 for r in ok if r["verdict"]["DNA_GENUINE_SIGNAL"])),
        "n_auc_beats_shuffled": int(sum(1 for r in ok if r["verdict"]["dna_auc_beats_shuffled"])),
    }

    blob = {
        "meta": {"cadence": CADENCE, "universe": "u20 (first 20 of u50)", "assets": U20,
                 "cost_rt": COST_RT, "n_shuffle": N_SHUFFLE, "n_books": N_BOOKS,
                 "reused": ["sol/oracle_dna_shuffled_falsifier.py::run",
                            "runs/research/oracle_ceiling_builder.py::oracle_high_capture"],
                 "elapsed_secs": round(time.time() - t_start, 1)},
        "aggregate": agg, "per_asset": results,
    }
    outp = ROOT / "experiments" / "adaptive_ma" / "oracle_dna_1d_u20.json"
    outp.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("U20 1d AGGREGATE")
    print("=" * 88)
    print(f"  assets ok={agg['n_assets_ok']}  err={agg['n_assets_err']}  elapsed={blob['meta']['elapsed_secs']}s")
    print(f"  ORACLE hold median (bars): median={agg['oracle_hold_median_bars'].get('median')} "
          f"p25={agg['oracle_hold_median_bars'].get('p25')} p75={agg['oracle_hold_median_bars'].get('p75')}")
    print(f"  held-out AUC real:    mean={agg['held_out_auc_real'].get('mean'):.4f} "
          f"median={agg['held_out_auc_real'].get('median'):.4f} max={agg['held_out_auc_real'].get('max'):.4f}")
    print(f"  held-out AUC shuf p95:mean={agg['held_out_auc_shuffled_p95'].get('mean'):.4f}")
    print(f"  held-out IC_fwd:      mean={agg['held_out_ic_fwd'].get('mean'):+.4f} "
          f"median={agg['held_out_ic_fwd'].get('median'):+.4f}")
    print(f"  capture_skill (realized/available): mean={agg['capture_skill_heldout'].get('mean'):+.4f} "
          f"median={agg['capture_skill_heldout'].get('median'):+.4f} max={agg['capture_skill_heldout'].get('max'):+.4f}")
    print(f"  realizable ceiling mean-net/move %: mean={agg['realizable_ceiling_mean_net_pct'].get('mean')}")
    print(f"  APPARATUS_SOUND: {agg['n_apparatus_sound']}/{agg['n_assets_ok']}   "
          f"DNA_GENUINE: {agg['n_dna_genuine']}/{agg['n_assets_ok']}   "
          f"AUC>shuffled: {agg['n_auc_beats_shuffled']}/{agg['n_assets_ok']}")
    print(f"\n[OK] wrote {outp}")
    return blob


if __name__ == "__main__":
    main()
