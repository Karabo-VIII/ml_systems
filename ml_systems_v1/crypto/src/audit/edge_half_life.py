"""edge_half_life.py -- G3 governance: per-pillar edge decay modeler.

Per docs/STRATEGIC_OBJECTIVES.md §7-G3.

For each pillar with multi-period IC/Sharpe observations, fits:

    IC(t) = IC_0 * exp(-λ * t)

and reports half-life = ln(2)/λ (days). Predicts IC at T+90.

When predicted IC at T+90 < Filter-tier threshold (IC < 0.015), the pillar
is auto-flagged for DECAY_WATCH (G1 transition).

INPUTS
------
- Per-pillar rolling IC/Sharpe history. Sources (in priority order):
    1. runs/v3_validation/per_pillar_history.parquet (if exists)
    2. logs/strat_audit/paper_trade_replay_v3_<PILLAR>_*.json (scan + assemble)
    3. (future) live trading IC trace

OUTPUT
------
- runs/half_life/<pillar>.parquet (raw + fitted decay)
- runs/half_life/summary.md (table + predicted decay)

USAGE
-----
    python src/audit/edge_half_life.py [--min-obs 3]

CONTRACT
--------
- Minimum 3 distinct time-windowed measurements per pillar to fit
- Half-life < 30 days = HIGH decay → auto-flag DECAY_WATCH
- Half-life > 180 days = LOW decay → pillar is robust
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V3_LOGS = ROOT / "logs" / "strat_audit"
OUT_DIR = ROOT / "runs" / "half_life"
OUT_DIR.mkdir(parents=True, exist_ok=True)


__contract__ = {
    "kind": "edge_half_life_model",
    "owner": "audit/governance",
    "outputs": "runs/half_life/<pillar>.parquet + summary.md",
    "invariants": [
        "fits IC(t) = IC_0 * exp(-lambda*t) via log-linear regression",
        "minimum 3 observations required",
        "predicted IC at T+90 below 0.015 -> auto-flag DECAY_WATCH",
    ],
}


def _scan_v3_history(blend: str) -> pd.DataFrame:
    """Assemble per-window v3 metrics for one blend.

    Returns DataFrame: [window_end_date, sharpe_ann, total_ret_pct, max_dd_pct, n_trades]
    """
    pattern = re.compile(rf"^paper_trade_replay_v3_{re.escape(blend)}_u\d+_(\d+)_(\d+)\.json$")
    rows: List[Dict] = []
    for p in V3_LOGS.glob(f"paper_trade_replay_v3_{blend}_*.json"):
        m = pattern.match(p.name)
        if not m:
            continue
        end_date = m.group(2)  # YYYYMMDD
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # R32++ Lane A: read v3 metrics via canonical schema, not magic strings.
        import sys as _sys
        from pathlib import Path as _Path
        _src = _Path(__file__).resolve().parents[1]
        if str(_src) not in _sys.path:
            _sys.path.insert(0, str(_src))
        from audit.v3_schema import get_v3_metric
        sh = get_v3_metric(data, "sharpe_ann")
        tr = get_v3_metric(data, "total_ret_pct")
        if sh is None:
            sh = float("nan")
        if tr is None:
            tr = float("nan")
        # R32+++ auditor-HIGH fix: route n_trades + max_dd_pct through v3_schema
        # too. v3 emits `n_closed_total`/`max_drawdown_pct`; bare data.get()
        # silently returned 0/NaN on v3 outputs.
        nt = get_v3_metric(data, "n_trades")
        dd = get_v3_metric(data, "max_dd_pct")
        rows.append({
            "window_end": pd.to_datetime(end_date, format="%Y%m%d"),
            "sharpe_ann": float(sh),
            "total_ret_pct": float(tr),
            "max_dd_pct": float(dd) if dd is not None else float("nan"),
            "n_trades": int(nt) if nt is not None else 0,
        })
    return pd.DataFrame(rows).sort_values("window_end").reset_index(drop=True)


def _fit_exponential_decay(t_days: np.ndarray, y: np.ndarray
                             ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Fit y(t) = y_0 * exp(-lambda * t) via log-linear regression.

    Returns (y_0, lambda, half_life_days). All None if fit fails.
    """
    mask = (y > 0) & np.isfinite(y) & np.isfinite(t_days)
    if mask.sum() < 3:
        return None, None, None
    log_y = np.log(y[mask])
    t = t_days[mask]
    # Linear regression: log_y = log(y_0) - lambda * t
    coef = np.polyfit(t, log_y, 1)  # [slope, intercept]
    slope, intercept = float(coef[0]), float(coef[1])
    y_0 = float(math.exp(intercept))
    lam = float(-slope)
    if lam <= 0:
        return y_0, lam, None  # no decay (growing/flat) → no half-life
    half_life = float(math.log(2) / lam)
    return y_0, lam, half_life


def analyze_pillar(blend: str, min_obs: int = 3) -> Optional[Dict]:
    """Fit decay model for one pillar. Returns analysis dict or None."""
    history = _scan_v3_history(blend)
    if len(history) < min_obs:
        return None
    t_days = (history["window_end"] - history["window_end"].iloc[0]).dt.days.values.astype(float)

    metrics_to_fit = ["sharpe_ann", "total_ret_pct"]
    out = {"blend": blend, "n_obs": len(history), "first": str(history["window_end"].iloc[0].date()),
           "last": str(history["window_end"].iloc[-1].date())}

    for m in metrics_to_fit:
        y = history[m].values
        # Sharpe can be negative; for decay only meaningful when consistently positive
        if (y > 0).sum() < min_obs:
            out[f"{m}_decay"] = None
            continue
        y_0, lam, hl = _fit_exponential_decay(t_days, np.where(y > 0, y, np.nan))
        out[f"{m}_y_0"] = y_0
        out[f"{m}_lambda"] = lam
        out[f"{m}_half_life_days"] = hl
        if y_0 is not None and lam is not None and lam > 0:
            # Predict at T+90
            t_predict = float(t_days[-1] + 90)
            out[f"{m}_predict_T90"] = y_0 * math.exp(-lam * t_predict)
        else:
            out[f"{m}_predict_T90"] = None

    history.to_parquet(OUT_DIR / f"{blend}.parquet", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-obs", type=int, default=3)
    args = ap.parse_args()

    # Discover blends from v3 logs
    blends = set()
    for p in V3_LOGS.glob("paper_trade_replay_v3_*.json"):
        m = re.match(r"^paper_trade_replay_v3_(.+?)_u\d+_\d+_\d+\.json$", p.name)
        if m:
            blends.add(m.group(1))

    print(f"[half_life] discovered {len(blends)} blends with v3 history")

    summary_rows = []
    for blend in sorted(blends):
        result = analyze_pillar(blend, min_obs=args.min_obs)
        if result is None:
            continue
        summary_rows.append(result)

    # Summary report
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    out_md = OUT_DIR / "summary.md"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(f"# Edge Half-Life Summary — {today}\n\n")
        fh.write(f"Pillars analyzed: {len(summary_rows)} (min {args.min_obs} obs each)\n\n")
        fh.write(f"| Blend | NObs | First | Last | Sh λ | Sh HL_days | Sh T+90 | Auto-flag |\n")
        fh.write(f"|---|---|---|---|---|---|---|---|\n")
        n_flagged = 0
        for r in summary_rows:
            lam = r.get("sharpe_ann_lambda")
            hl = r.get("sharpe_ann_half_life_days")
            t90 = r.get("sharpe_ann_predict_T90")
            flag = ""
            if hl is not None and hl < 30:
                flag = "🔴 fast decay"
                n_flagged += 1
            elif t90 is not None and t90 < 0.3:
                flag = "🟠 weak T+90"
                n_flagged += 1
            fh.write(f"| {r['blend']} | {r['n_obs']} | {r['first']} | {r['last']} | "
                      f"{lam if lam is None else round(lam, 4)} | "
                      f"{hl if hl is None else round(hl, 1)} | "
                      f"{t90 if t90 is None else round(t90, 3)} | {flag} |\n")
        fh.write(f"\n**Flagged pillars (auto-DECAY_WATCH candidates): {n_flagged}**\n")
    print(f"[half_life] summary at {out_md}; {len(summary_rows)} pillars analyzed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
