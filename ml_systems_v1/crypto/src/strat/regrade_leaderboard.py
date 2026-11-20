"""RE-GRADE LEADERBOARD -- run every runnable BOOK candidate through the canonical
scorecard (RWYB) + ingest the already-honestly-graded trade/lab candidates, into ONE
uniform deflated leaderboard. User mandate (2026-06-11 /orc): honest across the board;
1d/3d ROI reported as soft-benchmark, never an eliminator.

BOOK candidates (re-scored LIVE from their daily net series via scorecard.score_book):
  TSMOM_breadth, BLEND_50r, regime_beta, low_vol_tilt, buy_hold, RANDOM_null
  (from tsmom_ensemble.run -- MtM-correct, lagged weights). These are the core ship hunt.
TRADE/LAB candidates (ingested from their persisted run JSONs with claim-tags + the gate
each failed) -- regime_dna SYS_A/B/C, family_regime_map, momentum_rotation, trend_book,
setup_chaser, alt_bar, mover_ride, config_map. These were run under the (now-hardened)
harness; we surface their honest held-out verdict uniformly, flagging any selection caveat.

Output: runs/strat/REGRADE_LEADERBOARD_<stamp>.json  (+ console table). No emoji.
Run:  python -m strat.regrade_leaderboard --universe u50
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strat.scorecard import score_book  # noqa: E402

RUNS = ROOT.parent / "runs" / "strat"
RUNS_M = ROOT.parent / "runs" / "mining"


def _latest(pattern_dir: Path, pat: str):
    fs = sorted(glob.glob(str(pattern_dir / pat)))
    return fs[-1] if fs else None


def regrade_books(universe: str, cost_per_side: float) -> list[dict]:
    """RWYB: re-run the TSMOM/regime/blend books and score each via the canonical scorecard."""
    from strat.tsmom_ensemble import run as tsmom_run
    out, _ = tsmom_run(universe, "1d", cost_per_side)
    cards = []
    for name, d in out.items():
        net = d.get("_net")
        if net is None or len(net) < 50:
            continue
        card = score_book(name, net)
        card["source"] = f"tsmom_ensemble.run({universe},1d) RWYB"
        card["claim_tag"] = "VERIFIED"
        cards.append(card)
    return cards


def ingest_trade_jsons() -> list[dict]:
    """Ingest already-graded trade/lab candidates from their persisted JSONs, uniformly."""
    cards = []

    # regime_dna systems (already honest per the hardened r2 methodology)
    for uni in ["u10", "u50"]:
        f = _latest(RUNS, f"regime_dna_{uni}_1d_trend_2026061[01]_*.json")
        if not f:
            continue
        r = json.load(open(f))["result"]
        def _map(stat, breadth):  # regime_dna r2 schema (fractions) -> scorecard pct schema
            if not stat or not stat.get("n"):
                return {}
            jk = stat.get("jk_drop_top5pct_mean")
            return {"n": stat["n"], "mean_pct": round(stat["mean"] * 100, 3),
                    "se_pct": round(stat["se"] * 100, 3),
                    "jk_drop_top5pct_mean_pct": round(jk * 100, 3) if jk is not None else None,
                    "n_eff": round(stat.get("n_eff_pos", 0), 1),
                    "breadth_pos": (breadth or {}).get("pos"), "breadth_tot": (breadth or {}).get("tot")}
        for sysname, s in r.get("systems", {}).items():
            cards.append({
                "name": f"{sysname}[{uni}]", "kind": "TRADES", "source": Path(f).name,
                "claim_tag": "VERIFIED",
                "per_split": {"OOS": _map(s.get("OOS", {}), s.get("breadth_OOS")),
                              "UNSEEN": _map(s.get("UNSEEN", {}), s.get("breadth_UNSEEN"))},
                "note": "regime_dna r2 (fair jk, per-trade SE, regime-filtered null)"})
        # the regime decomposition headline (UP vs DOWN)
        ad = r.get("sysA_up_down", {})
        if ad:
            cards.append({"name": f"SYS_A_UPonly[{uni}]", "kind": "TRADES_DERIVED",
                          "source": Path(f).name, "claim_tag": "VERIFIED",
                          "per_split": {"OOS": ad.get("OOS", {}).get("UP", {}),
                                        "UNSEEN": ad.get("UNSEEN", {}).get("UP", {})},
                          "note": "UP-regime-only cells (the beta/trend component that survives)"})

    # the single-strat labs: headline held-out + the gate each failed
    lab_specs = [
        ("momentum_rotation_lab_2026-06-10.json", RUNS, "Family2_mover_rotation",
         "UNSEEN +172% compound but n_eff 4.1, firewall+PBO NOT RUN (concentration-fragile)"),
        ("trend_book_lab_2026-06-10.json", RUNS, "trend_book_lab",
         "UNSEEN 0 trades (flat in bear); OOS -7.5%/yr; bear-abstention mechanism"),
        ("symmetric_trend_book_2026-06-10.json", RUNS, "symmetric_trend_LS_perp",
         "UNSEEN +13.8%/yr on 4 short trades (n too small); oos_beats_baseline=false"),
        ("setup_chaser_book_2026-06-10.json", RUNS, "setup_chaser_book",
         "UNSEEN -33%/yr; battery FAIL jk3 -14.6 p05 -16.9 PBO 0.79 (clean NULL)"),
        ("alt_bar_trend_lab_2026-06-10.json", RUNS, "alt_bar_trend(Renko/Range/HA)",
         "UNSEEN -20 to -53%/yr; firewall_beats_null=false 0/10 seeds (NULL)"),
    ]
    for fn, d, nm, gate in lab_specs:
        p = d / fn
        if p.exists():
            cards.append({"name": nm, "kind": "LAB_INGEST", "source": fn,
                          "claim_tag": "REPORTED", "gate_failed": gate})
    return cards


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.regrade_leaderboard")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--maker", action="store_true")
    a = ap.parse_args()
    cps = 0.0006 if a.maker else 0.0012

    book_cards = regrade_books(a.universe, cps)
    trade_cards = ingest_trade_jsons()
    all_cards = book_cards + trade_cards

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha},
           "universe": a.universe, "cost_per_side": cps, "books": book_cards, "trades_labs": trade_cards}
    p = RUNS / f"REGRADE_LEADERBOARD_{a.universe}_{stamp}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)

    print(f"## RE-GRADE LEADERBOARD -- {a.universe} -- {'maker' if a.maker else 'taker'} -- canonical scorecard")
    print(f"\n=== BOOK candidates (RWYB via scorecard.score_book) ===")
    print(f"   {'book':16} | {'SEL ann/dd':>16} | {'OOS ann/dd':>16} | {'UNSEEN comp/dd':>16} | "
          f"{'full p05':>8} | {'held p05':>8} | {'3d med/+%':>12} | ship")
    # sort books by full-cycle ann desc
    def book_full_ann(c):
        return (c.get("per_split", {}).get("SEL", {}) or {}).get("ann_pct", -1e9)
    for c in sorted(book_cards, key=book_full_ann, reverse=True):
        ps = c["per_split"]
        def g(sp, k):
            v = ps.get(sp, {})
            return v.get(k) if v.get("n") else None
        sb = (ps.get("OOS", {}).get("softbench_roi", {}) or {}).get("3d", {})
        fp = c["full_block_bootstrap"].get("p05")
        hp = c.get("heldout_block_bootstrap", {}).get("p05")
        ship = c["ship_read"]["ship"]
        print(f"   {c['name']:16} | {str(g('SEL','ann_pct'))+'/'+str(g('SEL','maxdd_pct')):>16} | "
              f"{str(g('OOS','ann_pct'))+'/'+str(g('OOS','maxdd_pct')):>16} | "
              f"{str(g('UNSEEN','compound_pct'))+'/'+str(g('UNSEEN','maxdd_pct')):>16} | "
              f"{str(fp):>8} | {str(hp):>8} | "
              f"{str(sb.get('median_pct'))+'/'+str(sb.get('frac_positive')):>12} | {'SHIP' if ship else 'no'}")
    print(f"\n=== TRADE / LAB candidates (ingested, honest held-out) ===")
    for c in trade_cards:
        if c["kind"].startswith("TRADES"):
            u = c["per_split"].get("UNSEEN", {})
            o = c["per_split"].get("OOS", {})
            print(f"   {c['name']:22} OOS mean {o.get('mean_pct')}+-{o.get('se_pct')}% | "
                  f"UNSEEN mean {u.get('mean_pct')}+-{u.get('se_pct')}% jk {u.get('jk_drop_top5pct_mean_pct')} "
                  f"breadth {u.get('breadth_pos')}/{u.get('breadth_tot')}")
        else:
            print(f"   {c['name']:22} [{c['claim_tag']}] {c.get('gate_failed','')}")
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
