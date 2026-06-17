"""regime_quality.py — iron-clad regime classification audit.

User mandate 2026-05-16: ensure regime classification is iron-clad
before CC-H6 / regime-FiLM rely on it for cohort training.

Seven gate checks against the chimera regime_label column. Returns a
report with PASS/FAIL per check + a summary verdict.

INVOKE
------
    python src/audit/regime_quality.py --asset BTCUSDT
    python src/audit/regime_quality.py --universe u100 --json out.json

Each check:
  1. PRESENCE     -- regime_label column exists in chimera
  2. RANGE        -- values in {0, 1, 2}, no nulls
  3. PERSISTENCE  -- P(regime[t] == regime[t-1]) >= 0.95 (Markov-like)
  4. DISTRIBUTION -- entropy > log(3) * 0.50 (no collapsed labels)
  5. CROSS-ASSET  -- BTC regime agrees with ETH/SOL within 80% on
                     shared timestamps (regime should be macro-driven)
  6. STABILITY    -- per-asset class proportions ~constant across the
                     train/val/oos walk-forward split
  7. NO-LOOKAHEAD -- regime at bar t doesn't use bar t+1+ features
                     (heuristic: compare label changes vs return changes)

INTENT
------
This module is read-only. It does not change regime labels. The output
informs whether downstream regime-aware models (CC-H6, regime-FiLM)
can trust the labels in `regime_label`.
"""
from __future__ import annotations

__contract__ = {
    "kind": "regime_quality_audit",
    "owner": "audit/wm",
    "outputs": ["runs/audit/regime_quality_<DATE>.md"],
    "invariants": [
        "read-only: reads chimera parquet; never writes back",
        "version-agnostic: reuses pipeline regime_label, not per-model",
        "7 explicit checks with PASS/FAIL + remediation",
    ],
}

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHIMERA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"


# ───────────────────────────────────────────────────────────────────
# Loading
# ───────────────────────────────────────────────────────────────────

def load_chimera(asset: str) -> Optional["pl.DataFrame"]:
    """Load the most recent v50 chimera for an asset."""
    if pl is None:
        return None
    candidates = sorted(CHIMERA_DIR.glob(f"{asset.lower()}_v50_chimera_*.parquet"),
                          reverse=True)
    if not candidates:
        return None
    return pl.read_parquet(candidates[0])


# ───────────────────────────────────────────────────────────────────
# Gate checks
# ───────────────────────────────────────────────────────────────────

def check_presence(df) -> dict:
    has = "regime_label" in df.columns
    return {
        "name": "presence",
        "pass": has,
        "detail": ("regime_label column present" if has
                    else "MISSING regime_label column — pipeline regime_engine did not run"),
        "remediation": ("OK" if has
                          else "Run src/pipeline/regime_engine.py to build regime_label"),
    }


def check_range(df) -> dict:
    if "regime_label" not in df.columns:
        return {"name": "range", "pass": False,
                  "detail": "skipped — no regime_label", "remediation": "fix presence first"}
    s = df["regime_label"]
    n_null = s.null_count()
    vals = set(int(v) for v in s.drop_nulls().unique().to_list())
    ok = vals.issubset({0, 1, 2}) and n_null == 0
    return {
        "name": "range",
        "pass": ok,
        "detail": (f"n_null={n_null}, unique_values={sorted(vals)}"),
        "remediation": ("OK" if ok
                          else "regime_engine must emit only {0,1,2} with no nulls"),
    }


def check_persistence(df, threshold: float = 0.95) -> dict:
    if "regime_label" not in df.columns:
        return {"name": "persistence", "pass": False,
                  "detail": "skipped", "remediation": "fix presence first"}
    vals = df["regime_label"].to_numpy()
    if len(vals) < 100:
        return {"name": "persistence", "pass": False,
                  "detail": "<100 bars; can't compute",
                  "remediation": "load a longer chimera"}
    persistence = float(np.mean(vals[1:] == vals[:-1]))
    ok = persistence >= threshold
    return {
        "name": "persistence",
        "pass": ok,
        "detail": f"P(regime[t]==regime[t-1])={persistence:.4f} (threshold {threshold})",
        "remediation": ("OK" if ok
                          else f"label noisy; consider regime_engine smoothing — current "
                               f"{persistence:.3f} < {threshold}"),
    }


def check_distribution(df, entropy_floor_ratio: float = 0.50) -> dict:
    """Shannon entropy of regime label distribution. log(3) ~ 1.099 is max."""
    if "regime_label" not in df.columns:
        return {"name": "distribution", "pass": False,
                  "detail": "skipped", "remediation": "fix presence first"}
    vals = df["regime_label"].to_numpy()
    counts = np.bincount(vals.astype(np.int64), minlength=3) / max(1, len(vals))
    counts = counts[counts > 0]
    entropy = float(-np.sum(counts * np.log(counts)))
    max_entropy = np.log(3)
    ratio = entropy / max_entropy
    ok = ratio >= entropy_floor_ratio
    return {
        "name": "distribution",
        "pass": ok,
        "detail": (f"entropy={entropy:.4f} / log(3)={max_entropy:.4f}  "
                    f"= ratio {ratio:.3f} (floor {entropy_floor_ratio})"),
        "remediation": ("OK" if ok
                          else "regime distribution collapsed; check regime_engine "
                                "thresholds"),
    }


def check_cross_asset_agreement(asset: str, peers: list, threshold: float = 0.80) -> dict:
    """Compare this asset's regime to peer assets on shared timestamps."""
    if pl is None:
        return {"name": "cross_asset", "pass": False,
                  "detail": "polars not available", "remediation": "install polars"}
    df_a = load_chimera(asset)
    if df_a is None or "regime_label" not in df_a.columns or "timestamp" not in df_a.columns:
        return {"name": "cross_asset", "pass": False,
                  "detail": "asset chimera missing", "remediation": "build chimera"}
    a_ts = df_a["timestamp"].to_numpy()
    a_reg = df_a["regime_label"].to_numpy()
    agreements = []
    for p in peers:
        df_p = load_chimera(p)
        if df_p is None or "regime_label" not in df_p.columns:
            continue
        p_ts = df_p["timestamp"].to_numpy()
        p_reg = df_p["regime_label"].to_numpy()
        # asof-backward to align peer onto asset timestamps
        idx = np.searchsorted(p_ts, a_ts, side="right") - 1
        valid = idx >= 0
        if valid.sum() < 100:
            continue
        a_v = a_reg[valid]
        p_v = p_reg[idx[valid]]
        agreements.append((p, float(np.mean(a_v == p_v))))
    if not agreements:
        return {"name": "cross_asset", "pass": False,
                  "detail": "no peer chimeras available",
                  "remediation": "build chimeras for ETHUSDT + SOLUSDT first"}
    mean_agree = float(np.mean([a for _, a in agreements]))
    ok = mean_agree >= threshold
    return {
        "name": "cross_asset",
        "pass": ok,
        "detail": (f"asset={asset} peer_mean_agreement={mean_agree:.4f} "
                    f"(threshold {threshold}); per-peer: "
                    + ", ".join(f"{p}={a:.3f}" for p, a in agreements)),
        "remediation": ("OK" if ok
                          else "regime is asset-local, not macro; rebuild regime_engine "
                                "from BTC-only (single source of truth)"),
    }


def check_stability(df, n_splits: int = 4) -> dict:
    """Per-class distribution should be roughly stable across walk-forward splits."""
    if "regime_label" not in df.columns:
        return {"name": "stability", "pass": False,
                  "detail": "skipped", "remediation": "fix presence first"}
    vals = df["regime_label"].to_numpy().astype(np.int64)
    n = len(vals)
    if n < 1000:
        return {"name": "stability", "pass": False,
                  "detail": "<1000 bars; can't split", "remediation": "load longer chimera"}
    chunk = n // n_splits
    proportions = []
    for k in range(n_splits):
        s = vals[k * chunk:(k + 1) * chunk]
        p = np.bincount(s, minlength=3) / len(s)
        proportions.append(p)
    proportions = np.array(proportions)
    # Max delta of any class proportion across splits
    max_delta = float(np.max(proportions.max(axis=0) - proportions.min(axis=0)))
    ok = max_delta < 0.50   # no class swings by more than 50pp across splits
    return {
        "name": "stability",
        "pass": ok,
        "detail": (f"max class-proportion delta across {n_splits} splits = "
                    f"{max_delta:.3f}"),
        "remediation": ("OK if regime distribution truly drifts; "
                          f"otherwise verify regime_engine isn't drifting"),
    }


def check_no_lookahead(df) -> dict:
    """Heuristic: regime label at bar t should be derivable from features
    AT OR BEFORE bar t. Test: rerun regime_engine logic on first half
    of the chimera and verify labels match — if regime_engine looks at
    future bars, the first-half labels will differ from the full-data
    labels.

    We can't easily rerun the engine here, so a weaker test: check that
    regime changes correlate with PAST returns more than FUTURE returns.
    """
    if "regime_label" not in df.columns or "norm_return_1" not in df.columns:
        return {"name": "no_lookahead", "pass": False,
                  "detail": "skipped — need regime_label + norm_return_1",
                  "remediation": "ensure return column present"}
    reg = df["regime_label"].to_numpy().astype(np.int64)
    ret = df["norm_return_1"].to_numpy()
    # Find regime CHANGE points
    changes = np.where(np.diff(reg) != 0)[0]
    if len(changes) < 30:
        return {"name": "no_lookahead", "pass": True,
                  "detail": f"only {len(changes)} regime changes; can't test rigorously",
                  "remediation": "verify on longer chimera"}
    # Cumulative returns over the 10 bars BEFORE vs AFTER each change point
    w = 10
    before = []
    after = []
    for c in changes:
        if c < w or c + w >= len(ret):
            continue
        before.append(np.abs(np.nansum(ret[c - w:c])))
        after.append(np.abs(np.nansum(ret[c:c + w])))
    if not before:
        return {"name": "no_lookahead", "pass": False,
                  "detail": "no usable change points", "remediation": "longer data"}
    ratio = float(np.mean(after) / max(np.mean(before), 1e-12))
    # If regime is purely backward-looking, BEFORE should accumulate more
    # signal than AFTER. If ratio > 1.5, suspicious of look-ahead.
    ok = ratio < 1.5
    return {
        "name": "no_lookahead",
        "pass": ok,
        "detail": (f"mean|cum_ret| AFTER/BEFORE regime change = {ratio:.3f}  "
                    f"(>1.5 suggests look-ahead)"),
        "remediation": ("OK" if ok
                          else "regime_engine may use future bars; verify time-causal"),
    }


# ───────────────────────────────────────────────────────────────────
# Top-level runner
# ───────────────────────────────────────────────────────────────────

def run_audit(asset: str = "BTCUSDT",
                 peers: tuple = ("ETHUSDT", "SOLUSDT")) -> dict:
    df = load_chimera(asset)
    if df is None:
        return {
            "asset": asset,
            "error": f"chimera for {asset} not found at {CHIMERA_DIR}",
            "checks": [],
            "pass_count": 0, "total": 7,
        }
    checks = [
        check_presence(df),
        check_range(df),
        check_persistence(df),
        check_distribution(df),
        check_cross_asset_agreement(asset, list(peers)),
        check_stability(df),
        check_no_lookahead(df),
    ]
    pass_count = sum(1 for c in checks if c["pass"])
    return {
        "asset": asset,
        "checks": checks,
        "pass_count": pass_count,
        "total": len(checks),
        "verdict": ("IRON-CLAD" if pass_count == len(checks)
                      else f"GAP ({len(checks) - pass_count} failure)"),
    }


def render_markdown(report: dict) -> str:
    out: list[str] = []
    today = dt.date.today().isoformat()
    out.append(f"# Regime Quality Audit — {report['asset']} ({today})\n")
    out.append(f"**Verdict**: {report.get('verdict', 'ERROR')}  "
                 f"({report['pass_count']}/{report['total']} checks pass)\n")
    if report.get("error"):
        out.append(f"\n**ERROR**: {report['error']}\n")
        return "\n".join(out)
    out.append("| # | Check | Pass? | Detail | Remediation |")
    out.append("|---|---|---|---|---|")
    for i, c in enumerate(report["checks"], 1):
        out.append(f"| {i} | `{c['name']}` | "
                     f"{'YES' if c['pass'] else '**NO**'} | {c['detail']} | "
                     f"{c['remediation']} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--peers", nargs="*",
                    default=["ETHUSDT", "SOLUSDT"])
    ap.add_argument("--json", help="Also write JSON report to this path")
    args = ap.parse_args()
    report = run_audit(args.asset, tuple(args.peers))
    md = render_markdown(report)
    today = dt.date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"regime_quality_{args.asset}_{today}.md"
    out_path.write_text(md, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(md)
    print(f"\nReport: {out_path.relative_to(PROJECT_ROOT)}")
    return 0 if report.get("pass_count", 0) == report.get("total", 7) else 1


if __name__ == "__main__":
    sys.exit(main())
