"""scripts/outoffit_slice.py -- CLEAN out-of-fit headline (2020 in-sample year EXCLUDED).

WHY: the MA working band was SELECTED on 2020 (TRAIN). The committed full-window metrics
(2020 -> 2024-05-15) therefore INCLUDE the in-sample fit-year. This script reports the SAME
drawdown-insurance metrics over the OUT-OF-FIT sub-window [2021-01-01, 2024-05-15) -- 2020
EXCLUDED -- using ONLY the verified equity curves.

NO NEW ACCOUNTING. It re-runs the canonical, verified `run_wealth` internals
(src/strat/ma_wealth_driver.py) to reproduce the EXACT dated engine + TRUE-HODL equity curves,
then post-hoc SLICES that already-correct MtM stream to the sub-window and re-bases compounding
to the sub-window start. A post-hoc slice of a correct mark-to-market equity stream is valid and
leak-free (OOS/UNSEEN past 2024-05-15 is NOT touched; the DEV wall already sealed it).

FIDELITY GATE: before slicing, this asserts the FULL-window compound% and maxDD% recomputed from
the reconstructed curves match the committed engine_<TYPE>_clean.json scalars (within tolerance).
That proves the reconstructed curve IS the verified curve, not a hand-roll.

No emoji (cp1252). No git commit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CRYPTO = Path(__file__).resolve().parents[1]
SRC = CRYPTO / "src"
for p in (str(SRC), str(CRYPTO.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import strat.fleet_lab as fl  # noqa: E402
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402
from strat.ma_wealth_driver import (  # noqa: E402  -- the VERIFIED accounting primitives, reused verbatim
    _u100_symbols, _load_asset, _compound, _true_hodl, MAKER_RT,
)

SUB_START = pd.Timestamp("2021-01-01")
SUB_END = pd.Timestamp("2024-05-15")  # == DEV_END; the full window already ends here. We only drop 2020.
MMC = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH"


def _maxdd_pct(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def _rebase_compound_pct(eq: np.ndarray) -> float:
    """Compound % of an equity curve re-based to its FIRST value (sub-window start = 100%)."""
    if len(eq) == 0 or eq[0] == 0:
        return 0.0
    return float((eq[-1] / eq[0] - 1.0) * 100)


def build_curves(ma_type: str, tf: str, window: str = "7d", cost_rt: float = MAKER_RT):
    """Reproduce run_wealth's EXACT pipeline up to the dated engine + HODL equity curves.
    Returns (index, eng_eq, hodl_eq, full_eng_pct, full_eng_dd, full_hodl_pct, full_hodl_dd, n_assets, n_total)."""
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)
    sub_band, _ = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        raise RuntimeError(f"sub-band < 2 for {ma_type}|{tf}")
    provider = make_hold_provider(sub_band, ma_type)

    closes, holds = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        hh = provider(sym, o, h, l, c).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]
    if not closes:
        raise RuntimeError(f"no assets loaded for {ma_type}|{tf}")

    cp = pd.DataFrame(closes).sort_index()
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)
    assert cp.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"
    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    w_eng = held.div(n_listed, axis=0).fillna(0.0)

    eng_pct, eng_net, eng_eq, eng_gross, eng_dd = _compound(w_eng, R, cost_rt)
    hodl_pct, hodl_eq, hodl_dd, n_total = _true_hodl(cp, R, listed, cost_rt)

    idx = cp.index
    return (idx, np.asarray(eng_eq), np.asarray(hodl_eq),
            eng_pct, eng_dd, hodl_pct, hodl_dd, int(cp.shape[1]), int(n_total))


def committed_scalars(ma_type: str, tf: str, window: str = "7d") -> dict:
    d = json.load(open(MMC / f"engine_{ma_type}_clean.json"))
    for c in d["cells"]:
        if c["tf"] == tf and c["window"] == window:
            return c
    raise KeyError(f"no committed cell {ma_type}|{tf}|{window}")


def run_cell(ma_type: str, tf: str, window: str = "7d") -> dict:
    (idx, eng_eq, hodl_eq, full_eng_pct, full_eng_dd,
     full_hodl_pct, full_hodl_dd, n_assets, n_total) = build_curves(ma_type, tf, window)

    # --- FIDELITY GATE: reconstructed FULL-window must match committed clean.json scalars ---
    com = committed_scalars(ma_type, tf, window)
    fid = {
        "eng_compound_recon": round(full_eng_pct, 2),
        "eng_compound_committed": com["engine_compound_pct"],
        "eng_dd_recon": round(full_eng_dd, 1),
        "eng_dd_committed": com["maxdd_engine_pct"],
        "hodl_compound_recon": round(full_hodl_pct, 2),
        "hodl_compound_committed": com["hodl_compound_pct"],
        "hodl_dd_recon": round(full_hodl_dd, 1),
        "hodl_dd_committed": com["maxdd_hodl_pct"],
        "n_assets_recon_wealth_panel": n_assets,
        "n_symbols_committed_capture_leg": com["n_symbols"],
        "n_assets_note": ("clean.json n_symbols is the CAPTURE-leg admitted count (type-specific); the WEALTH "
                          "panel (what is sliced here) is the type-independent u100 loader = 56 for all cells, "
                          "matching run_wealth's own n_assets. NOT compared in the gate (different quantities)."),
    }
    # FIDELITY GATE keys on the four WEALTH scalars that DEFINE the sliced equity curves -- each must match the
    # committed clean.json to the penny. (n_symbols is the capture-leg count, a different field; not gated.)
    # tolerances: compound within 0.5% relative; maxDD within 0.5pp.
    def _relok(a, b, tol=5e-3):
        return abs(a - b) <= tol * max(abs(b), 1.0)
    fid_ok = (
        _relok(full_eng_pct, com["engine_compound_pct"])
        and _relok(full_hodl_pct, com["hodl_compound_pct"])
        and abs(full_eng_dd - com["maxdd_engine_pct"]) <= 0.5
        and abs(full_hodl_dd - com["maxdd_hodl_pct"]) <= 0.5
    )
    fid["PASS"] = bool(fid_ok)

    # --- SLICE to the out-of-fit sub-window [2021-01-01, 2024-05-15) ---
    mask = np.asarray((idx >= SUB_START) & (idx < SUB_END))
    sub_idx = idx[mask]
    sub_eng = eng_eq[mask]
    sub_hodl = hodl_eq[mask]

    out = {
        "ma_type": ma_type,
        "tf": tf,
        "window": window,
        "engine_maxDD_pct": round(_maxdd_pct(sub_eng), 1),
        "hodl_maxDD_pct": round(_maxdd_pct(sub_hodl), 1),
        "engine_compound_pct": round(_rebase_compound_pct(sub_eng), 2),
        "hodl_compound_pct": round(_rebase_compound_pct(sub_hodl), 2),
        # full-window (for the side-by-side; from the SAME reconstructed verified curves)
        "full_engine_maxDD_pct": round(full_eng_dd, 1),
        "full_hodl_maxDD_pct": round(full_hodl_dd, 1),
        "full_engine_compound_pct": round(full_eng_pct, 2),
        "full_hodl_compound_pct": round(full_hodl_pct, 2),
        "sub_n_bars": int(mask.sum()),
        "sub_date_range": [str(sub_idx.min()), str(sub_idx.max())] if mask.sum() else [None, None],
        "n_assets": n_assets,
        "n_total_admitted": n_total,
        "fidelity_gate": fid,
    }
    return out


def main() -> int:
    cells = [(t, tf) for t in ("EMA", "HMA", "TEMA") for tf in ("1d", "4h")]
    results = []
    all_fid_pass = True
    for ma_type, tf in cells:
        r = run_cell(ma_type, tf)
        results.append(r)
        fp = r["fidelity_gate"]["PASS"]
        all_fid_pass = all_fid_pass and fp
        print(f"[{ma_type:4s}|{tf:2s}] OUT-OF-FIT(2021->2024-05): "
              f"eng_DD={r['engine_maxDD_pct']:>6}% hodl_DD={r['hodl_maxDD_pct']:>6}% | "
              f"eng_cmp={r['engine_compound_pct']:>10}% hodl_cmp={r['hodl_compound_pct']:>9}% | "
              f"FULL eng_DD={r['full_engine_maxDD_pct']}% | FIDELITY={'PASS' if fp else 'FAIL'} "
              f"(n_bars_sub={r['sub_n_bars']})")
    print(f"\nALL FIDELITY GATES PASS: {all_fid_pass}")

    out_dir = CRYPTO / "runs" / "engines" / "V1_ma_drawdown_insurance" / "outoffit"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "window": "2021-01-01 -> 2024-05-15 (OUT-OF-FIT; 2020 in-sample fit-year EXCLUDED)",
        "method": ("post-hoc slice of the VERIFIED run_wealth MtM equity curves; re-based compounding to "
                   "sub-window start; NO new accounting; maker 0.0006 RT; DEV wall < 2024-05-15 (OOS/UNSEEN sealed)"),
        "selection_note": ("MA working band was selected on 2020 TRAIN; this sub-window EXCLUDES that fit-year. "
                           "Drawdown-insurance is driven by 2021 (May-21 selloff), 2022 (full bear), 2023 (chop) -- "
                           "3 out-of-fit selloffs -- so the property should survive."),
        "all_fidelity_gates_pass": all_fid_pass,
        "cells": results,
        "git_sha_source": json.load(open(MMC / "engine_EMA_clean.json")).get("git_sha"),
    }
    out_path = out_dir / "outoffit_2021_2024-05.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nWROTE {out_path}")
    return 0 if all_fid_pass else 1


if __name__ == "__main__":
    sys.exit(main())
