"""src/strat/dynamic_capture_engine.py -- the DYNAMIC TI x TIMEFRAME x ASSET move-CAPTURE engine.

CHARTER (user, 2026-06-18 -- autonomous campaign): a REPEATABLE engine that captures moves
(trend-following / continuation, NOT prediction) across all assets x TI x timeframes, ADAPTING its
config to market conditions/regimes, ranked by WEALTH (compound net), dev on 2020 / iron on 2021 /
2022 = no-touch bear. See project-dynamic-capture-engine-charter-2026-06-18 + the design map
(workflow wf_36a48ce5-d88).

PHASED BUILD (this file = the orchestrator; the per-cell sweep lives in ma_strat_builder.run_cell):
  PHASE 0 (this commit) -- the STATIC starting harness: 8 MA types x 6 TFs x u10, working-band
     ensemble, honestly gated, graded at the TAKER floor, ranked by WEALTH, with capture-rate as a
     SECONDARY diagnostic (never a selection key). Emits a reproducibility-fingerprinted artifact
     JSON + a tiered register. This is the FLOOR the dynamic layer must beat.
  PHASE 2 (next) -- the regime ROUTER: a per-TI-type two-tier regime mechanism
     (Tier-1 SMA-200 UP/DOWN position GATE; Tier-2 regime restricts a band-SUBSET, never a single
     config). LESSON: regime GATING works, regime config-SWITCHING hurts (D33; train_ma_walkforward
     refuted) -- so the dynamic layer is gate + band-subset, compared head-to-head vs this static floor.

HONEST-GRADE INVARIANTS (baked in):
  - cost = TAKER_RT (0.0024) floor; maker is an optimistic stress only.
  - SELECTION on TRAIN+VAL only; OOS is a held-out confirm (ma_strat_builder enforces this).
  - capture-rate / coverage / entry-lag are DIAGNOSTICS, never the rank key. Rank = WEALTH (net_oos).
  - p05 block-bootstrap + 2022 bear + beats_noskill reported per cell.
  - fixed-EW (fillna(0).mean) u10; sweep ALL 6 TFs.

RWYB:
  python -m strat.dynamic_capture_engine --selftest
  python -m strat.dynamic_capture_engine --tfs 1d,4h,2h,1h,30m,15m
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]    # crypto/src
CRYPTO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.ma_strat_builder as msb           # noqa: E402
from strat.ma_type_upgrade import MA_TYPES     # noqa: E402

OUT = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
OUT.mkdir(parents=True, exist_ok=True)
ALL_TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]


# ---------------------------------------------------------------------------
# TIERING (wealth + robustness; mirrors ti_candidate_register intent)
# ---------------------------------------------------------------------------
def _tier(r: dict) -> str:
    """Tier a cell by the FORWARD-TRANSLATION triple (the real all-weather test):
    A_allweather = positive 2020-OOS (bull dev) AND positive 2021-fwd (unseen mixed year) AND
                   bear-preserving 2022 (>= -5%). p05 is reported as a robustness flag, not a tier gate
                   (it is a single 2020-bull-window bootstrap; forward 2021+2022 is the stronger test).
    B_preserve   = participates + preserves on ONE forward axis. C_bull_only = neither. D_weak = 2020<=0."""
    if "error" in r:
        return "ERR"
    oos = r.get("net_oos") or 0.0
    fwd = r.get("net_2021_fwd")
    bear = r.get("net_bear_2022")
    if oos <= 0:
        return "D_weak"
    translate = (fwd is not None and fwd > 0)
    preserve = (bear is not None and bear >= -5.0)
    if translate and preserve:
        return "A_allweather"
    if translate or (bear is not None and bear >= -30.0):
        return "B_preserve"
    return "C_bull_only"


# ---------------------------------------------------------------------------
# REPRODUCIBILITY FINGERPRINT
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(CRYPTO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def _fingerprint(params: dict) -> dict:
    blob = json.dumps(params, sort_keys=True, default=str)
    return {
        "config_hash": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
        "git_sha": _git_sha(),
        "params": params,
    }


# ---------------------------------------------------------------------------
# RUN (phase 0 = static)
# ---------------------------------------------------------------------------
def run(tfs, ma_types, cost: str = "taker", static_only: bool = True,
        tag: str = "phase0", verbose: bool = True, gate: str = "none") -> dict:
    msb.COST_RT = msb.TAKER_RT if cost == "taker" else msb.MAKER_RT
    msb.REGIME_GATE = None if gate == "none" else gate
    seed_cost = {"taker": msb.TAKER_RT, "maker": msb.MAKER_RT}[cost]
    params = {
        "phase": tag, "mode": "static" if static_only else "best",
        "cost": cost, "cost_rt": seed_cost, "regime_gate": gate,
        "regime_gate_n": msb.REGIME_GATE_N if gate != "none" else None,
        "tfs": tfs, "ma_types": ma_types,
        "universe": msb.SYMS,
        "split": {"TRAIN": list(msb.TRAIN), "VAL": list(msb.VAL), "OOS": list(msb.OOS)},
        "bootstrap_seed": 0, "rank_key": "net_oos(WEALTH)",
    }
    fp = _fingerprint(params)
    print(f"## DYNAMIC CAPTURE ENGINE [{tag}]  cost={cost}({seed_cost})  "
          f"static_only={static_only}  git={fp['git_sha']}  hash={fp['config_hash']}")
    print(f"   TFs={tfs}  MA_TYPES={ma_types}  rank=WEALTH(net_oos)")

    rows = []
    for cad in tfs:
        if cad not in ALL_TFS:
            print(f"   [skip] {cad}"); continue
        print(f"\n=== TF={cad} ===")
        bh = msb.buyhold_sanity(cad)
        print(f"   buy-hold(full 2020): {bh}%")
        for mt in ma_types:
            r = msb.run_cell(mt, cad, verbose=verbose, static_only=static_only)
            r["tier"] = _tier(r)
            r["tf"] = cad
            rows.append(r)

    ok = [r for r in rows if "error" not in r]
    # WEALTH-ranked leaderboard (held-out OOS net). robustness shown, never the sort key.
    leaderboard = sorted(ok, key=lambda r: -(r.get("net_oos") or -999))

    payload = {
        "engine": "dynamic_capture_engine", "tag": tag,
        "generated": "STAMP_AFTER_RETURN",
        "fingerprint": fp,
        "rank_key": "net_oos (WEALTH, held-out OOS compound net %)",
        "metric_defs": msb.METRIC_DEFS,
        "n_cells": len(ok), "n_errors": len(rows) - len(ok),
        "leaderboard": [_clean(r) for r in leaderboard],
    }
    return payload


def _clean(r: dict) -> dict:
    keep = ["tf", "ma_type", "tier", "best_family", "best_mode", "best_cooldown",
            "best_min_hold", "best_exit", "sel_gate", "net_train", "net_val", "net_oos",
            "p05_oos_bootstrap", "maxDD_full", "hold_oos", "coverage_train",
            "capture_oos_median", "entry_lag_train_mean", "beats_noskill_fixedhold",
            "dynamic_vs_static_oosdelta", "net_2021_fwd", "net_bear_2022", "bh_net_full", "error"]
    return {k: r[k] for k in keep if k in r}


# ---------------------------------------------------------------------------
# LEADERBOARD PRINT + REGISTER
# ---------------------------------------------------------------------------
def _print_leaderboard(payload: dict):
    print("\n=== WEALTH-RANKED LEADERBOARD (held-out OOS net %; robustness shown) ===")
    print(f"  {'TF':4}{'MA':6}{'tier':13}{'fam':4}{'mode':7}{'2020%':>7}{'2021fwd':>8}"
          f"{'2022br':>7}{'p05':>7}{'maxDD':>7}{'cap':>5}{'lag':>5}{'beatNS':>7}  exit")
    for r in payload["leaderboard"]:
        def f(x, w, d=1):
            return (" " * w) if x is None else f"{x:>{w}.{d}f}"
        print(f"  {r['tf']:4}{r['ma_type']:6}{r.get('tier',''):13}"
              f"{r.get('best_family',''):4}{r.get('best_mode',''):7}"
              f"{f(r.get('net_oos'),7)}{f(r.get('net_2021_fwd'),8)}"
              f"{f(r.get('net_bear_2022'),7)}{f(r.get('p05_oos_bootstrap'),7)}"
              f"{f(r.get('maxDD_full'),7)}"
              f"{f(r.get('capture_oos_median'),5,2)}{f(r.get('entry_lag_train_mean'),5,2)}"
              f"{str(r.get('beats_noskill_fixedhold')):>7}"
              f"  {r.get('best_exit','')}")
    # tier tally
    from collections import Counter
    tally = Counter(r.get("tier") for r in payload["leaderboard"])
    print(f"\n  tiers: {dict(tally)}")


def _write(payload: dict, tag: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload["generated"] = ts
    p = OUT / f"dynamic_capture_{tag}_{ts}.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # append a register row (machine-readable, append-only)
    reg = OUT / "DYNAMIC_CAPTURE_REGISTER.jsonl"
    top = payload["leaderboard"][:1]
    line = {"ts": ts, "tag": tag, "git": payload["fingerprint"]["git_sha"],
            "hash": payload["fingerprint"]["config_hash"], "n_cells": payload["n_cells"],
            "top_cell": _clean(top[0]) if top else None,
            "tier_counts": {t: sum(1 for r in payload["leaderboard"] if r.get("tier") == t)
                            for t in ["A_allweather", "B_preserve", "C_bull_only", "D_weak"]}}
    with reg.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")
    return p


# ---------------------------------------------------------------------------
# SELFTEST (two-sided + repeatability)
# ---------------------------------------------------------------------------
def selftest() -> int:
    print("[selftest] dynamic_capture_engine -- 1d, EMA+SMA, static, taker")
    p1 = run(["1d"], ["EMA", "SMA"], cost="taker", static_only=True, tag="selftest", verbose=True)
    assert p1["n_cells"] == 2, f"expected 2 cells, got {p1['n_cells']}"
    lb = p1["leaderboard"]
    assert lb and all("net_oos" in r for r in lb), "leaderboard missing net_oos"
    assert all(r.get("tier") in ("A_allweather", "B_preserve", "C_bull_only", "D_weak") for r in lb), "bad tier"
    # wealth-sorted descending
    nets = [r["net_oos"] for r in lb]
    assert nets == sorted(nets, reverse=True), "leaderboard not wealth-sorted"
    # repeatability: same fingerprint hash on a re-run (seed/params deterministic)
    p2 = run(["1d"], ["EMA", "SMA"], cost="taker", static_only=True, tag="selftest", verbose=False)
    assert p1["fingerprint"]["config_hash"] == p2["fingerprint"]["config_hash"], "fingerprint not reproducible"
    assert [r["net_oos"] for r in p2["leaderboard"]] == nets, "results not reproducible run-to-run"
    _print_leaderboard(p1)
    print("\n[selftest] PASSED (structure + wealth-sort + reproducibility)")
    return 0


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.dynamic_capture_engine")
    ap.add_argument("--tfs", default=",".join(ALL_TFS))
    ap.add_argument("--ma_types", default=",".join(MA_TYPES))
    ap.add_argument("--cost", default="taker", choices=["taker", "maker"])
    ap.add_argument("--tag", default="phase0_static")
    ap.add_argument("--gate", default="none", choices=["none", "sma200"],
                    help="Tier-1 regime position gate (sma200 = cash when close<=SMA200)")
    ap.add_argument("--best_mode", action="store_true",
                    help="allow dynamic mode to win cells (default static_only)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]
    ma_types = [m.strip() for m in a.ma_types.split(",") if m.strip()]
    payload = run(tfs, ma_types, cost=a.cost, static_only=not a.best_mode, tag=a.tag,
                  verbose=True, gate=a.gate)
    p = _write(payload, a.tag)
    _print_leaderboard(payload)
    print(f"\n[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
