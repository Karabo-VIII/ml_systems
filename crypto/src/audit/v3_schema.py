"""v3_schema.py -- canonical V3 paper-trade-replay JSON schema registry.

PURPOSE
-------
Root cause of multiple R32 audit findings: distributed agreements without
enforcement. v3 emits `sharpe_annualized` + `total_pnl_pct` + `n_closed_total`;
G2 drift_monitor and yaml_claim_corrector and edge_half_life and
bootstrap_sharpe_ci each read these via independent magic-string aliases.
Drift between writer and reader -> silent miss (e.g., R32 RED-TEAM found
edge_half_life producing all-NaN because it read `sharpe_ann` instead of
`sharpe_annualized`).

This module is the single source of truth. Every consumer imports
V3_FIELD_MAP and reads via `get_v3_metric(data, "sharpe_ann")` instead of
hand-coding aliases.

USAGE
-----
    from audit.v3_schema import V3_FIELD_MAP, get_v3_metric, extract_v3_metrics

    metrics = extract_v3_metrics(json_data)
    # -> {"sharpe_ann": 1.23, "total_ret_pct": 5.67, ...}

    sharpe = get_v3_metric(json_data, "sharpe_ann")
    # -> reads the first matching alias

CONTRACT
--------
- Single source of truth for v3 JSON top-level metric keys
- Canonical names are the LEFT-HAND side (e.g., "sharpe_ann")
- Aliases (RIGHT side tuple) are read in order; first match wins
- When v3 paper_trade_replay_v3.py emission changes, this file is the
  single update point
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


__contract__ = {
    "kind": "schema_registry",
    "owner": "audit/governance",
    "outputs": "canonical v3 field mapping (in-memory dict)",
    "invariants": [
        "every consumer of v3 JSON reads via this module, not raw strings",
        "left-side keys = canonical names; right-side = read aliases",
        "first alias match wins (order matters)",
    ],
}


# Canonical name -> tuple of v3 JSON key aliases (read order = priority)
V3_FIELD_MAP: Dict[str, tuple] = {
    # Core metrics
    "sharpe_ann":      ("sharpe_annualized", "sharpe_ann", "sharpe"),
    "total_ret_pct":   ("total_pnl_pct", "total_ret_pct"),
    "max_dd_pct":      ("max_dd_pct",),
    "n_trades":        ("n_closed_total", "n_trades"),
    "hit_rate_pct":    ("hit_rate_pct",),    # not in v3 top-level; see derive_hit_rate
    # Window metadata
    "n_days":          ("n_days",),
    "window_start":    ("window_start",),
    "window_end":      ("window_end",),
    "blend":           ("blend",),
    "universe":        ("universe",),
    # NAV
    "nav_initial":     ("nav_initial",),
    "nav_final":       ("nav_final",),
    # Per-day arrays (for bootstrap CI + half-life)
    "per_day":         ("per_day",),
}


def get_v3_metric(data: Dict[str, Any], canonical: str) -> Optional[Any]:
    """Read a canonical metric from a v3 JSON dict. Returns first-matching
    alias value or None if not found."""
    aliases = V3_FIELD_MAP.get(canonical)
    if not aliases:
        return None
    for a in aliases:
        if a in data:
            return data[a]
    return None


def extract_v3_metrics(data: Dict[str, Any],
                         canonical_keys: Optional[Iterable[str]] = None
                         ) -> Dict[str, Any]:
    """Extract all (or specified subset of) canonical metrics from v3 JSON."""
    keys = list(canonical_keys) if canonical_keys else list(V3_FIELD_MAP.keys())
    out: Dict[str, Any] = {}
    for k in keys:
        v = get_v3_metric(data, k)
        if v is not None:
            out[k] = v
    return out


def derive_hit_rate_pct(data: Dict[str, Any]) -> Optional[float]:
    """v3 doesn't emit hit_rate_pct at top-level; derive from per_day."""
    per_day = get_v3_metric(data, "per_day")
    if not isinstance(per_day, list) or not per_day:
        return None
    n_pos = sum(1 for r in per_day
                  if isinstance(r, dict) and isinstance(r.get("day_pnl_pct"), (int, float))
                  and r["day_pnl_pct"] > 0)
    n_total = sum(1 for r in per_day
                    if isinstance(r, dict) and isinstance(r.get("day_pnl_pct"), (int, float)))
    return (100.0 * n_pos / n_total) if n_total > 0 else None


def daily_returns_from_v3(data: Dict[str, Any]):
    """Extract numpy array of daily returns from v3 per_day list.

    Returns None if insufficient data. Used by bootstrap_sharpe_ci +
    edge_half_life + risk_parity_allocator.
    """
    import numpy as np
    per_day = get_v3_metric(data, "per_day")
    if not isinstance(per_day, list) or len(per_day) < 2:
        return None
    # Prefer nav-derived; fallback to day_pnl_pct
    navs = [r.get("nav") for r in per_day
              if isinstance(r, dict) and isinstance(r.get("nav"), (int, float))
              and r.get("nav") > 0]
    if len(navs) >= 2:
        arr = np.asarray(navs, dtype=np.float64)
        return np.diff(arr) / arr[:-1]
    pnls = [r.get("day_pnl_pct") for r in per_day
              if isinstance(r, dict) and isinstance(r.get("day_pnl_pct"), (int, float))]
    if len(pnls) >= 5:
        return np.asarray([p / 100.0 for p in pnls], dtype=np.float64)
    return None


def _smoke() -> int:
    """Verify schema map + extract on a sample v3 JSON."""
    import json
    import sys
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ROOT = Path(__file__).resolve().parents[2]
    # Find any v3 JSON
    sample = next(ROOT.glob("logs/strat_audit/paper_trade_replay_v3_*.json"), None)
    if not sample:
        print("[SMOKE] no v3 JSON found")
        return 1
    data = json.loads(sample.read_text(encoding="utf-8"))
    metrics = extract_v3_metrics(data,
                                   ("sharpe_ann", "total_ret_pct", "max_dd_pct", "n_trades"))
    print(f"[SMOKE] sample: {sample.name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    hr = derive_hit_rate_pct(data)
    print(f"  hit_rate_pct (derived): {hr}")
    rets = daily_returns_from_v3(data)
    print(f"  daily_returns N: {len(rets) if rets is not None else 'None'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
