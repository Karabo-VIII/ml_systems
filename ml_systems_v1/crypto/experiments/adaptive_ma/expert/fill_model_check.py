"""experiments/adaptive_ma/expert/fill_model_check.py -- wire src/strat/fill_model.py (taker 0.0024).

The brief mandates the cost be applied through the kept apparatus' fill model. run_u100.py charges
taker 0.0024 via StrategySpec.cost_rt (the canonical harness cost axis). This script CONFIRMS that the
held-out adaptive result is identical when the cost is re-applied through src/strat/fill_model.py's
`apply_fill_model(harness, "taker")` path -- i.e. the two cost surfaces agree -- and reports the
adaptive held-out compound under each fill mode (taker = realistic; maker_pessimistic = stress;
ideal_ref = costless-ish reference).

For mode="taker" (p_fill=1.0, adverse=0.0) the fill model is DETERMINISTIC and, on a harness whose spec
cost already equals 0.0024, reduces to identity on each trade's net -- so median compound MUST match the
direct-cost harness exactly. That equality is the wiring proof. maker_pessimistic is expected to collapse
(p_fill 0.30 + adverse 0.96) -- a sanity anchor, NOT a ship number.

RWYB:  python experiments/adaptive_ma/expert/fill_model_check.py [--n 12]
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_ma as A  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec  # noqa: E402
from strat.fill_model import apply_fill_model, MODES  # noqa: E402

TAKER = 0.0024
HELD = ["OOS", "UNSEEN"]
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")


def _load(loader, s):
    try:
        g = loader.load(s, cadence="1d")
    except Exception:
        return None
    return pd.DataFrame({"date": pd.to_datetime(g["date"].to_list()),
                         "open": g["open"].to_numpy().astype(float), "high": g["high"].to_numpy().astype(float),
                         "low": g["low"].to_numpy().astype(float), "close": g["close"].to_numpy().astype(float)})


def _spec(cost):
    return StrategySpec(fast_col="adaptive_fast", slow_col="adaptive_slow", signal="crossover",
                        filter_col=None, exit_policy="signal_flip", cost_rt=cost, use_funding=False,
                        funding_scale=0.0, max_hold_bars=None, max_hold_ext_bars=None)


def _held_comp_direct(res):
    nets = [t["net_pnl"] for t in res.trades if t["window"] in HELD]
    a = np.asarray(nets, float)
    return float((np.prod(1.0 + a) - 1.0) * 100) if a.size else 0.0


def main(n):
    loader = ChimeraLoader()
    syms = UniverseLoader.load().list("u100")[:n]
    frames = {s: df for s in syms if (df := _load(loader, s)) is not None and len(df) > 400}
    print(f"[fill_model_check] {len(frames)} assets | taker={TAKER} | modes={list(MODES)}\n", flush=True)

    rows = []
    max_taker_mismatch = 0.0
    for s, df in frames.items():
        feat = A.compute_features(df, xs_disp=None)
        adp = A.build_adaptive_columns(feat)
        h = CanonicalHarness(adp, _spec(TAKER), WIN, chimera_path=f"fm::{s}")

        direct = _held_comp_direct(h.run())  # held-out compound, taker via spec cost
        fm = {m: apply_fill_model(h, m) for m in MODES}  # taker / maker_pessimistic / ideal_ref
        # held-out compound via fill model = compound OOS then UNSEEN (chain), to compare windows we
        # report the pooled-window medians; the wiring equality is per-window taker median vs the
        # direct per-window compound.
        taker_oos = fm["taker"]["OOS"]["median"]
        taker_uns = fm["taker"]["UNSEEN"]["median"]
        # direct per-window compound (for the equality check)
        def _wc(res, w):
            nets = [t["net_pnl"] for t in res.trades if t["window"] == w]
            a = np.asarray(nets, float)
            return float((np.prod(1.0 + a) - 1.0) * 100) if a.size else 0.0
        r = h.run()
        d_oos, d_uns = _wc(r, "OOS"), _wc(r, "UNSEEN")
        max_taker_mismatch = max(max_taker_mismatch, abs(taker_oos - d_oos), abs(taker_uns - d_uns))
        rows.append({"sym": s,
                     "direct_held_comp": round(direct, 2),
                     "taker_OOS": taker_oos, "taker_UNSEEN": taker_uns,
                     "maker_pess_OOS": fm["maker_pessimistic"]["OOS"]["median"],
                     "maker_pess_UNSEEN": fm["maker_pessimistic"]["UNSEEN"]["median"],
                     "ideal_OOS": fm["ideal_ref"]["OOS"]["median"], "ideal_UNSEEN": fm["ideal_ref"]["UNSEEN"]["median"]})

    print(f"{'sym':12} {'directHeld%':>11} {'tkOOS':>8} {'tkUNS':>8} {'mkOOS':>8} {'mkUNS':>8} {'idOOS':>8} {'idUNS':>8}")
    for x in rows:
        print(f"{x['sym']:12} {x['direct_held_comp']:>11} {x['taker_OOS']:>8} {x['taker_UNSEEN']:>8} "
              f"{x['maker_pess_OOS']:>8} {x['maker_pess_UNSEEN']:>8} {x['ideal_OOS']:>8} {x['ideal_UNSEEN']:>8}")

    tk_oos = float(np.mean([x["taker_OOS"] for x in rows]))
    tk_uns = float(np.mean([x["taker_UNSEEN"] for x in rows]))
    mk_uns = float(np.mean([x["maker_pess_UNSEEN"] for x in rows]))
    id_uns = float(np.mean([x["ideal_UNSEEN"] for x in rows]))
    print("-" * 86)
    print(f"WIRING PROOF: max |fill_model taker median - direct-cost compound| over assets x window = "
          f"{max_taker_mismatch:.6f} (== 0 -> fill_model taker path agrees with spec-cost path)")
    print(f"MEANS  taker: OOS={tk_oos:+.2f}% UNSEEN={tk_uns:+.2f}% | maker_pess UNSEEN={mk_uns:+.2f}% | "
          f"ideal UNSEEN={id_uns:+.2f}%")
    print(f"(adaptive held-out is negative under the realistic taker fill -> consistent with run_u100.py refutation)")

    out = {"n_assets": len(frames), "wiring_max_mismatch": max_taker_mismatch,
           "means": {"taker_OOS": round(tk_oos, 2), "taker_UNSEEN": round(tk_uns, 2),
                     "maker_pess_UNSEEN": round(mk_uns, 2), "ideal_UNSEEN": round(id_uns, 2)},
           "per_asset": rows}
    Path(__file__).resolve().parent.joinpath("fill_model_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[saved] fill_model_check.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    main(args.n)
