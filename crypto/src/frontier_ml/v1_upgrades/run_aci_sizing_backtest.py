"""Width-aware vs flat sizing backtest -- B007 E1 downstream gate.

Reads a per-bar parquet produced by run_aci_v1_eval.py and computes:
    - flat sizing strategy: position = sign(point_pred) * |point_pred| / vol_proxy
    - width-aware sizing:   position = sign(point_pred) * (c / width) capped at 1.0
                            where width is the ACI conformal-interval width.

Both strategies are sized to comparable target risk (gross exposure normalized).

Reports per-strategy Sharpe / Sortino / DD / Calmar and the lift gates:
    delta_sortino   = sortino(width-aware) - sortino(flat)
    delta_sharpe    = sharpe(width-aware) - sharpe(flat)

Decision gate (B007 RESPONSE §10 E1, downstream half):
    width-aware lifts Sortino by >= 0.3 vs flat sizing -> ship E1 as default.

Usage:
    python -m frontier_ml.v1_upgrades.run_aci_sizing_backtest \
        --parquet logs/frontier_ml/aci_eval/aci_BTC_val_<...>.parquet

This is a *gate-check* on the VAL slice. The OOS slice is reserved for the
final pre-deploy strategy backtest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _annualization(n_bars: int, span_days: float | None = None) -> float:
    """Bars-to-year scale. Heuristic: dollar-bar BTC ~ 5K bars/day historically.
    If span_days is unknown, fall back to sqrt(n_bars / 252) which is wrong but
    deterministic; the LIFT (delta) is invariant to this scale, so don't sweat it.
    """
    if span_days and span_days > 0:
        bars_per_year = n_bars / span_days * 365.0
    else:
        bars_per_year = 252.0  # conservative fallback
    return float(np.sqrt(bars_per_year))


def _stats(returns: np.ndarray, ann_scale: float) -> dict:
    """Sharpe / Sortino / max-DD / Calmar from a per-bar return stream."""
    r = np.asarray(returns, dtype=np.float64)
    mu = r.mean()
    sd = r.std(ddof=0) + 1e-12
    sharpe = float(mu / sd * ann_scale)
    downside = r[r < 0]
    dsd = (np.sqrt((downside ** 2).mean()) if downside.size else 1e-12)
    sortino = float(mu / max(dsd, 1e-12) * ann_scale)

    # Max drawdown on cumulative log-equity
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min())  # negative
    cagr = float(mu * ann_scale ** 2)  # crude; depends on ann_scale^2 = bars/year
    calmar = float(cagr / abs(max_dd)) if max_dd < -1e-12 else float("nan")

    return {
        "n": int(len(r)),
        "mean_per_bar": float(mu),
        "std_per_bar": float(sd),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cagr_proxy": cagr,
        "calmar_proxy": calmar,
    }


def _flat_sizing(point_pred: np.ndarray, vol_proxy: np.ndarray, target_vol: float) -> np.ndarray:
    """Constant-target-vol sized position from sign + magnitude of point prediction."""
    sign = np.sign(point_pred)
    mag = np.abs(point_pred)
    raw = sign * mag / np.maximum(vol_proxy, 1e-8)
    return _clip_unit(_normalize_to_target_vol(raw, target_vol))


def _width_aware_sizing(
    point_pred: np.ndarray, width: np.ndarray, target_vol: float, c: float = 0.005
) -> np.ndarray:
    """Position scales with confidence (1/width); same direction as point_pred."""
    sign = np.sign(point_pred)
    pos = sign * (c / np.maximum(width, 1e-8))
    return _clip_unit(_normalize_to_target_vol(pos, target_vol))


def _width_gate_sizing(
    point_pred: np.ndarray, width: np.ndarray, target_vol: float, gate_q: float = 0.75
) -> np.ndarray:
    """Binary regime-gate: trade only when width <= q-th quantile, else cash.

    Inside the gate, magnitude follows |point_pred|. Risk-off when width is
    in the top (1-gate_q) quantile of the local stream.
    """
    threshold = np.quantile(width, gate_q)
    in_gate = (width <= threshold).astype(np.float64)
    raw = np.sign(point_pred) * np.abs(point_pred) * in_gate
    return _clip_unit(_normalize_to_target_vol(raw, target_vol))


def _width_quantile_sizing(
    point_pred: np.ndarray, width: np.ndarray, target_vol: float, n_buckets: int = 5
) -> np.ndarray:
    """Position magnitude is a step function of width's rolling quantile.

    Tightest 20% of width -> full position; widest 20% -> zero.
    Same direction as point_pred.
    """
    edges = np.quantile(width, np.linspace(0, 1, n_buckets + 1))
    # Higher bucket = wider = lower confidence => smaller position multiplier
    bucket = np.clip(np.searchsorted(edges, width, side="right") - 1, 0, n_buckets - 1)
    multipliers = np.linspace(1.0, 0.0, n_buckets)  # bucket 0 -> 1.0, last -> 0.0
    mult = multipliers[bucket]
    raw = np.sign(point_pred) * mult
    return _clip_unit(_normalize_to_target_vol(raw, target_vol))


def _width_modulated_sizing(
    point_pred: np.ndarray, width: np.ndarray, target_vol: float, alpha: float = 1.0
) -> np.ndarray:
    """Multiplier on the flat strategy: keeps point_pred magnitude, scales by width^-alpha.

    pos = sign(pp) * |pp| / vol * (median_width / width)^alpha

    This preserves the predicted magnitude information that flat already uses, and
    only modulates by width-relative confidence. alpha=1.0 = full inverse-width
    scaling; alpha=0 = identity (collapses to flat sizing).
    """
    sign = np.sign(point_pred)
    mag = np.abs(point_pred)
    vol_proxy = max(np.std(point_pred), 1e-8)
    median_w = float(np.median(width))
    confidence = (median_w / np.maximum(width, 1e-8)) ** alpha
    raw = sign * mag / vol_proxy * confidence
    return _clip_unit(_normalize_to_target_vol(raw, target_vol))


def _normalize_to_target_vol(pos: np.ndarray, target_vol: float) -> np.ndarray:
    """Scale a position stream to target stddev. No-op if std is zero."""
    s = pos.std(ddof=0)
    if s < 1e-12:
        return pos
    return pos * (target_vol / s)


def _clip_unit(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, -1.0, 1.0)


def _strategy_returns(pos: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Position at bar t * realized return at bar t = bar's strategy return.

    Position is set BEFORE observing y_true (pos comes from ACI prediction at
    end of window k; y_true[k] is the realized next-bar return). So no leak.
    """
    return pos * y_true


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True,
                   help="Per-bar parquet from run_aci_v1_eval.py --save-parquet")
    p.add_argument("--target-vol", type=float, default=0.005,
                   help="Target per-bar stddev for both sizing rules (apples-to-apples).")
    p.add_argument("--width-c", type=float, default=0.005,
                   help="Width-aware constant: pos = sign(pred) * c / width.")
    p.add_argument("--gate-q", type=float, default=0.75,
                   help="Width-gate quantile: trade only when width <= q-th quantile.")
    p.add_argument("--n-buckets", type=int, default=5,
                   help="Width-qbin: number of width-quantile buckets for stepwise sizing.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Width-modulated: exponent on (median_width/width). "
                        "alpha=0 collapses to flat; alpha=1 = full inverse-width scaling.")
    p.add_argument("--span-days", type=float, default=None,
                   help="Approx span of the parquet for annualization. "
                        "If omitted, uses sqrt(252) as a deterministic fallback "
                        "(LIFTS are scale-invariant).")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "logs" / "frontier_ml" / "aci_sizing"))
    args = p.parse_args()

    pq = Path(args.parquet)
    if not pq.exists():
        print(f"[size] parquet not found: {pq}", file=sys.stderr)
        sys.exit(2)
    df = pl.read_parquet(pq)
    print(f"[size] loaded {len(df):,} rows from {pq.name}")

    needed = ["y_true", "point_pred", "width", "L", "U", "regime_label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[size] parquet missing cols: {missing}", file=sys.stderr)
        sys.exit(2)

    y = df.get_column("y_true").to_numpy()
    pp = df.get_column("point_pred").to_numpy()
    w = df.get_column("width").to_numpy()
    regime = df.get_column("regime_label").to_numpy()

    # vol proxy for flat sizing: rolling std of point_pred (constant if absent).
    # For apples-to-apples, both sizings get target-vol normalized.
    vol_proxy = np.full_like(pp, max(np.std(pp), 1e-6))

    pos_flat = _flat_sizing(pp, vol_proxy, args.target_vol)
    pos_width = _width_aware_sizing(pp, w, args.target_vol, c=args.width_c)
    pos_gate = _width_gate_sizing(pp, w, args.target_vol, gate_q=args.gate_q)
    pos_qbin = _width_quantile_sizing(pp, w, args.target_vol, n_buckets=args.n_buckets)
    pos_mod = _width_modulated_sizing(pp, w, args.target_vol, alpha=args.alpha)

    r_flat = _strategy_returns(pos_flat, y)
    r_width = _strategy_returns(pos_width, y)
    r_gate = _strategy_returns(pos_gate, y)
    r_qbin = _strategy_returns(pos_qbin, y)
    r_mod = _strategy_returns(pos_mod, y)

    ann = _annualization(len(y), args.span_days)
    strat_returns = {
        "flat": r_flat,
        "width_aware": r_width,
        "width_gate": r_gate,
        "width_qbin": r_qbin,
        "width_mod": r_mod,
    }
    strat_stats = {name: _stats(r, ann) for name, r in strat_returns.items()}

    # Deltas vs flat
    deltas = {}
    for name in ("width_aware", "width_gate", "width_qbin", "width_mod"):
        deltas[name] = {
            "sortino": strat_stats[name]["sortino"] - strat_stats["flat"]["sortino"],
            "sharpe":  strat_stats[name]["sharpe"]  - strat_stats["flat"]["sharpe"],
        }

    # Per-regime Sortino under all four strategies
    per_regime = {}
    for rid, rname in [(0, "bear"), (1, "chop"), (2, "bull")]:
        mask = regime == rid
        if mask.sum() < 100:
            continue
        per_regime[rname] = {
            "n": int(mask.sum()),
            **{
                f"sortino_{name}": _stats(r[mask], ann)["sortino"]
                for name, r in strat_returns.items()
            },
        }

    # Decision-gate: any width-rule beats flat by the B007 threshold?
    any_clears_sortino = any(d["sortino"] >= 0.3 for d in deltas.values())
    any_clears_sharpe  = any(d["sharpe"]  >= 0.2 for d in deltas.values())
    best_rule = max(deltas, key=lambda k: deltas[k]["sortino"])

    summary = {
        "parquet": str(pq),
        "n_bars": int(len(df)),
        "target_vol": args.target_vol,
        "width_c": args.width_c,
        "gate_q": args.gate_q,
        "n_buckets": args.n_buckets,
        "ann_scale_used": ann,
        "stats": strat_stats,
        "deltas_vs_flat": deltas,
        "per_regime": per_regime,
        "decision_gate": {
            "any_rule_delta_sortino_ge_0p3": bool(any_clears_sortino),
            "any_rule_delta_sharpe_ge_0p2":  bool(any_clears_sharpe),
            "best_rule_by_sortino": best_rule,
            "best_delta_sortino": float(deltas[best_rule]["sortino"]),
            "best_delta_sharpe":  float(deltas[best_rule]["sharpe"]),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sizing_{pq.stem}_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    for name, st in strat_stats.items():
        print(f"[size] {name:<14s} Sharpe={st['sharpe']:+.3f}  "
              f"Sortino={st['sortino']:+.3f}  DD={st['max_dd']:+.4f}")
    print("[size] DELTAS vs flat:")
    for name, d in deltas.items():
        print(f"[size]   {name:<14s} dSortino={d['sortino']:+.3f}  dSharpe={d['sharpe']:+.3f}")
    print(f"[size] decision_gate = {summary['decision_gate']}")
    print(f"[size] per-regime sortino: {per_regime}")
    print(f"[size] summary written to {out_path}")
    return summary


if __name__ == "__main__":
    main()
