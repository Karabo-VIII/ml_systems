"""rigorous_pipeline_validation.py -- universe-wide validator-grade probe.

Tests 5 adversarial axes on the data pipeline state after 2026-05-21 fixes:

  T1. PRE_TRAIN_GATE UNIVERSE SWEEP: run gate on 10 diverse assets (not just BTC).
      VERIFIED if all return rc<=1; INFERRED if same fix profile across assets.
  T2. SCHEMA COMPLETENESS: every chimera dollar + cadence has identical core schema.
      Detects partial-build / interrupted-write / missing-column states.
  T3. CROSS-CADENCE XREL CONSISTENCY: xrel values are daily-constant by design.
      Pick a sample (asset, date); xrel on dollar must equal xrel on 1d at same date.
  T4. LOOK-AHEAD AUDIT ON XREL: for each xrel feature, verify daily panel uses ONLY
      same-date cross-section. Compute xrel from raw source; compare to chimera xrel.
  T5. ADVERSARIAL PROBES: (a) v50 recovery integrity (selected v51 cols match recovered
      v50 cols within 1e-9 rel err); (b) sparse-by-design coverage (dv_/xex_/soc_
      have non-null values for the assets we EXPECT them to cover, not random).

OUTPUT:
  runs/audit/PIPELINE_RIGOROUS_VALIDATION_2026_05_21/REPORT.md
  runs/audit/PIPELINE_RIGOROUS_VALIDATION_2026_05_21/per_test.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit" / "PIPELINE_RIGOROUS_VALIDATION_2026_05_21"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHIM_DOLLAR = ROOT / "data" / "processed" / "chimera" / "dollar"
CHIM_1D = ROOT / "data" / "processed" / "chimera" / "1d"
CHIM_4H = ROOT / "data" / "processed" / "chimera" / "4h"
CHIM_LEGACY = ROOT / "data" / "processed" / "chimera_legacy" / "dollar"

# 10 diverse assets covering DNA buckets (PRIME / STEADY / DEGEN) + recovered-stale
TEST_ASSETS = ["BTC", "ETH", "SOL",        # PRIME
                "AAVE", "ADA", "BNB",       # STEADY
                "SHIB", "WIF",              # DEGEN (also recovered today)
                "GUN", "STO"]               # recently-stale-and-refreshed

# Reference schema: list of REQUIRED core columns (v51 chimera dollar)
REQUIRED_CORE_COLS = [
    "timestamp", "bar_id", "open", "high", "low", "close", "volume",
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
    "norm_return_1", "norm_hl_spread", "norm_flow_imbalance",
    "norm_kyle_lambda", "norm_vpin", "norm_hawkes_imbalance",
    "xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
]
REQUIRED_XREL_FEATURES = [
    "xrel_rv_bpv_5m_xrank", "xrel_rv_bpv_5m_xpct10", "xrel_rv_bpv_5m_xratio",
    "xrel_wh_whale_net_usd_xratio", "xrel_lob_kyle_lambda_mean_xratio",
]


# ===== T1 — Universe-wide pre_train_gate sweep =====

def t1_universe_gate_sweep() -> dict:
    """Run pre_train_gate on 10 assets; pass if all rc<=1."""
    print("\n[T1] Universe-wide pre_train_gate sweep (10 assets)...")
    results = []
    pre_train_gate = ROOT / "src" / "pipeline" / "pre_train_gate.py"
    python_exe = sys.executable
    for asset in TEST_ASSETS:
        t0 = time.time()
        proc = subprocess.run(
            [python_exe, str(pre_train_gate), "--asset", asset],
            capture_output=True, text=True, timeout=600, encoding="utf-8",
            errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rc = proc.returncode
        # Extract summary line if present
        out = proc.stdout + proc.stderr
        summary = ""
        for line in out.splitlines():
            if "Summary:" in line or "WARNS:" in line or "HARD FAIL" in line:
                summary = line.strip()
                break
        results.append({
            "asset": asset, "rc": rc, "summary": summary[:200],
            "elapsed_s": round(time.time() - t0, 1),
        })
        status_word = "OK" if rc == 0 else ("WARN" if rc == 1 else "FAIL")
        print(f"  {asset:<6} rc={rc} ({status_word})  {results[-1]['elapsed_s']}s  {summary[:100]}")
    n_ok = sum(1 for r in results if r["rc"] == 0)
    n_warn = sum(1 for r in results if r["rc"] == 1)
    n_fail = sum(1 for r in results if r["rc"] >= 2)
    return {
        "name": "T1_universe_gate_sweep",
        "n_tested": len(results),
        "n_ok": n_ok, "n_warn": n_warn, "n_fail": n_fail,
        "verdict": "PASS" if n_fail == 0 else "FAIL",
        "per_asset": results,
    }


# ===== T2 — Schema completeness =====

def t2_schema_completeness() -> dict:
    """Every chimera dollar + cadence file has REQUIRED_CORE_COLS + xrel_* + correct count."""
    print("\n[T2] Schema completeness audit (435 chimera files)...")
    issues = []
    n_files = 0
    n_clean = 0
    for cad, cdir in [("dollar", CHIM_DOLLAR), ("1d", CHIM_1D), ("4h", CHIM_4H),
                       ("1h", ROOT / "data/processed/chimera/1h"),
                       ("15m", ROOT / "data/processed/chimera/15m")]:
        if not cdir.exists():
            issues.append({"cadence": cad, "issue": "DIR_MISSING"})
            continue
        files = sorted(cdir.glob("*_v51_chimera_*.parquet"))
        for f in files:
            n_files += 1
            asset = f.name.split("_")[0].upper().replace("USDT", "")
            try:
                schema = set(pl.read_parquet_schema(f).keys())
                missing_core = [c for c in REQUIRED_CORE_COLS if c not in schema]
                missing_xrel = [c for c in REQUIRED_XREL_FEATURES if c not in schema]
                if missing_core or missing_xrel:
                    issues.append({
                        "cadence": cad, "asset": asset, "file": f.name,
                        "missing_core": missing_core[:3],
                        "missing_xrel": missing_xrel[:3],
                        "n_cols": len(schema),
                    })
                else:
                    n_clean += 1
            except Exception as e:
                issues.append({"cadence": cad, "asset": asset, "file": f.name,
                                "issue": f"READ_FAIL: {type(e).__name__}: {e}"})
    print(f"  {n_clean}/{n_files} files clean; {len(issues)} issues")
    if issues:
        for i in issues[:5]:
            print(f"    {i}")
    return {
        "name": "T2_schema_completeness",
        "n_files": n_files, "n_clean": n_clean, "n_issues": len(issues),
        "verdict": "PASS" if not issues else ("WARN" if len(issues) < 5 else "FAIL"),
        "sample_issues": issues[:10],
    }


# ===== T3 — Cross-cadence xrel consistency =====

def t3_cross_cadence_xrel_consistency() -> dict:
    """xrel features are daily-constant. dollar xrel on date D should equal cadence xrel on date D."""
    print("\n[T3] Cross-cadence xrel consistency (3 assets × xrel_rv_bpv_5m_xratio)...")
    results = []
    test_feat = "xrel_rv_bpv_5m_xratio"
    for asset in ["BTC", "ETH", "SOL"]:
        sym = asset.lower() + "usdt"
        dollar_files = sorted(CHIM_DOLLAR.glob(f"{sym}_v51_chimera_*.parquet"))
        d1_files = sorted(CHIM_1D.glob(f"{sym}_v51_chimera_1d_*.parquet"))
        if not dollar_files or not d1_files:
            results.append({"asset": asset, "issue": "MISSING_FILES"})
            continue
        try:
            # Load last 100 dates from dollar + 1d
            df_dol = pl.read_parquet(dollar_files[-1],
                                       columns=["timestamp", test_feat]).tail(50000).to_pandas()
            df_dol["date"] = pl.from_pandas(df_dol)["timestamp"].apply(
                lambda x: x).to_pandas() if False else (df_dol["timestamp"] // 86_400_000)
            df_dol = df_dol.groupby("date")[test_feat].first().reset_index()
            df_dol = df_dol.tail(30)

            df_1d = pl.read_parquet(d1_files[-1],
                                     columns=["timestamp", test_feat]).tail(30).to_pandas()
            df_1d["date"] = df_1d["timestamp"] // 86_400_000

            # Join on date
            merged = df_dol.merge(df_1d, on="date", suffixes=("_dol", "_1d"))
            if len(merged) == 0:
                results.append({"asset": asset, "issue": "NO_DATE_OVERLAP"})
                continue
            d_dol = merged[f"{test_feat}_dol"].values
            d_1d = merged[f"{test_feat}_1d"].values
            # Allow small tolerance (float precision)
            valid = ~(np.isnan(d_dol) | np.isnan(d_1d))
            if valid.sum() == 0:
                results.append({"asset": asset, "issue": "ALL_NAN"})
                continue
            diff = np.abs(d_dol[valid] - d_1d[valid])
            max_diff = float(diff.max()) if len(diff) > 0 else 0.0
            rel_diff = max_diff / (abs(d_dol[valid]).max() + 1e-9)
            ok = max_diff < 1e-6 or rel_diff < 1e-4
            results.append({
                "asset": asset, "n_pairs": int(valid.sum()),
                "max_abs_diff": max_diff, "rel_diff": rel_diff,
                "ok": ok,
            })
            print(f"  {asset}: n={valid.sum()} max_abs_diff={max_diff:.2e}  rel={rel_diff:.2e}  "
                  f"{'OK' if ok else 'DIFF'}")
        except Exception as e:
            results.append({"asset": asset, "issue": f"{type(e).__name__}: {e}"})
            print(f"  {asset}: ERROR {type(e).__name__}: {e}")
    n_ok = sum(1 for r in results if r.get("ok"))
    return {
        "name": "T3_cross_cadence_xrel_consistency",
        "n_tested": len(results), "n_ok": n_ok,
        "verdict": "PASS" if n_ok == len(results) else "WARN",
        "per_asset": results,
    }


# ===== T4 — Look-ahead audit on xrel =====

def t4_look_ahead_audit_xrel() -> dict:
    """For a sample (asset, date), recompute xratio from raw source; compare to stored.

    If the stored value uses ANY future-date data, recompute (same-date only) will differ.
    """
    print("\n[T4] Look-ahead audit on xrel (recompute xratio from same-date cross-section)...")
    test_feat_src = "rv_bpv_5m"
    test_feat_xrel = "xrel_rv_bpv_5m_xratio"
    sample_dates = []
    try:
        # Load all 87 chimera dollar files and extract date + rv_bpv_5m + xrel
        results = []
        all_files = sorted(CHIM_DOLLAR.glob("*_v51_chimera_*.parquet"))[:30]  # cap for speed
        # Build a date-indexed panel: rows = (asset, date), col = rv_bpv_5m + xrel_ratio
        panel_rows = []
        for f in all_files:
            asset = f.name.split("_")[0].upper().replace("USDT", "")
            schema = set(pl.read_parquet_schema(f).keys())
            if test_feat_src not in schema or test_feat_xrel not in schema:
                continue
            df = pl.read_parquet(f, columns=["timestamp", test_feat_src, test_feat_xrel]).tail(200_000).to_pandas()
            df["date"] = df["timestamp"] // 86_400_000
            # daily last value
            df = df.groupby("date").last().reset_index()
            df["asset"] = asset
            panel_rows.append(df)
        if not panel_rows:
            return {"name": "T4_look_ahead_audit_xrel", "verdict": "SKIP",
                    "reason": "no files with both src + xrel cols"}
        import pandas as pd
        panel = pd.concat(panel_rows, ignore_index=True)
        # Pick last 5 dates that have at least 10 assets
        date_counts = panel.groupby("date").size()
        good_dates = sorted(date_counts[date_counts >= 10].index)[-5:]
        for d in good_dates:
            slice_ = panel[panel["date"] == d]
            # Recompute xratio = value / median(|value|)
            vals = slice_[test_feat_src].astype(float).values
            abs_median = np.median(np.abs(vals[~np.isnan(vals)]))
            if abs_median < 1e-12:
                continue
            recomputed = vals / abs_median
            recomputed = np.clip(recomputed, -100.0, 100.0)
            stored = slice_[test_feat_xrel].astype(float).values
            valid = ~(np.isnan(recomputed) | np.isnan(stored))
            if valid.sum() == 0:
                continue
            diff = np.abs(recomputed[valid] - stored[valid])
            max_diff = float(diff.max())
            ok = max_diff < 1e-3
            results.append({
                "date": int(d), "n_assets": int(valid.sum()),
                "abs_median": float(abs_median),
                "max_diff_vs_recompute": max_diff, "ok": ok,
            })
            print(f"  date={d} n={valid.sum()} abs_median={abs_median:.3e} "
                  f"max_diff={max_diff:.3e} {'OK' if ok else 'LEAK?'}")
        n_ok = sum(1 for r in results if r.get("ok"))
        return {
            "name": "T4_look_ahead_audit_xrel",
            "n_dates_tested": len(results), "n_ok": n_ok,
            "verdict": "PASS" if n_ok == len(results) and results else "WARN",
            "per_date": results,
        }
    except Exception as e:
        return {"name": "T4_look_ahead_audit_xrel", "verdict": "ERROR",
                "error": f"{type(e).__name__}: {e}"}


# ===== T5 — Adversarial probes =====

def t5_adversarial_probes() -> dict:
    """(a) v50 recovery: recovered v50 norm_return_1 matches v51 norm_return_1.
       (b) sparse-by-design coverage: BTC has wiki_views; SHIB doesn't (sanity check)."""
    print("\n[T5] Adversarial probes (recovery integrity + sparse-by-design)...")
    probes = []
    # (a) Recovery integrity for BTC
    try:
        v51 = pl.read_parquet(
            CHIM_DOLLAR / "btcusdt_v51_chimera_20260519.parquet",
            columns=["timestamp", "norm_return_1", "norm_hl_spread"]).tail(1000).to_pandas()
        v50_files = sorted(CHIM_LEGACY.glob("btcusdt_v50_chimera_*.parquet"))
        if not v50_files:
            probes.append({"probe": "5a_recovery", "verdict": "SKIP", "reason": "no v50 file"})
        else:
            v50 = pl.read_parquet(
                v50_files[-1],
                columns=["timestamp", "norm_return_1", "norm_hl_spread"]).tail(1000).to_pandas()
            merged = v51.merge(v50, on="timestamp", suffixes=("_v51", "_v50"))
            diff_ret = float((merged["norm_return_1_v51"] - merged["norm_return_1_v50"]).abs().max())
            diff_spr = float((merged["norm_hl_spread_v51"] - merged["norm_hl_spread_v50"]).abs().max())
            ok = diff_ret < 1e-9 and diff_spr < 1e-9
            probes.append({
                "probe": "5a_recovery", "n_pairs": len(merged),
                "max_diff_norm_return_1": diff_ret,
                "max_diff_norm_hl_spread": diff_spr,
                "verdict": "PASS" if ok else "FAIL",
            })
            print(f"  5a: v50/v51 norm_return_1 max_diff={diff_ret:.2e} "
                  f"norm_hl_spread max_diff={diff_spr:.2e}  {'OK' if ok else 'MISMATCH'}")
    except Exception as e:
        probes.append({"probe": "5a_recovery", "verdict": "ERROR",
                        "error": f"{type(e).__name__}: {e}"})

    # (b) Sparse-by-design coverage check
    try:
        btc = pl.read_parquet(
            CHIM_DOLLAR / "btcusdt_v51_chimera_20260519.parquet",
            columns=["soc_wiki_views"] if "soc_wiki_views" in
            pl.read_parquet_schema(CHIM_DOLLAR / "btcusdt_v51_chimera_20260519.parquet")
            else []).tail(10000).to_pandas()
        shib = pl.read_parquet(
            CHIM_DOLLAR / "shibusdt_v51_chimera_20260517.parquet",
            columns=["soc_wiki_views"] if "soc_wiki_views" in
            pl.read_parquet_schema(CHIM_DOLLAR / "shibusdt_v51_chimera_20260517.parquet")
            else []).tail(10000).to_pandas()
        btc_null_pct = btc["soc_wiki_views"].isna().mean() * 100 if len(btc.columns) else 100
        shib_null_pct = shib["soc_wiki_views"].isna().mean() * 100 if len(shib.columns) else 100
        # BTC should have data (covered); SHIB should be all null (not covered)
        ok = btc_null_pct < 80 and shib_null_pct > 90
        probes.append({
            "probe": "5b_sparse_coverage",
            "btc_soc_wiki_null_pct": float(btc_null_pct),
            "shib_soc_wiki_null_pct": float(shib_null_pct),
            "expected": "BTC covered (<80% null), SHIB not covered (>90% null)",
            "verdict": "PASS" if ok else "WARN",
        })
        print(f"  5b: BTC soc_wiki null={btc_null_pct:.1f}%  SHIB null={shib_null_pct:.1f}%  "
              f"{'OK (expected pattern)' if ok else 'UNEXPECTED'}")
    except Exception as e:
        probes.append({"probe": "5b_sparse_coverage", "verdict": "ERROR",
                        "error": f"{type(e).__name__}: {e}"})

    n_pass = sum(1 for p in probes if p.get("verdict") == "PASS")
    return {
        "name": "T5_adversarial_probes",
        "n_probes": len(probes), "n_pass": n_pass,
        "verdict": "PASS" if all(p.get("verdict") in ("PASS", "SKIP") for p in probes) else "WARN",
        "probes": probes,
    }


# ===== Driver =====

def main():
    print("=" * 78)
    print("RIGOROUS PIPELINE VALIDATION — Validator-grade adversarial probe")
    print("=" * 78)
    t_start = time.time()
    results = []
    for fn in (t1_universe_gate_sweep, t2_schema_completeness,
                t3_cross_cadence_xrel_consistency, t4_look_ahead_audit_xrel,
                t5_adversarial_probes):
        try:
            r = fn()
        except Exception as e:
            r = {"name": fn.__name__, "verdict": "EXCEPTION",
                 "error": f"{type(e).__name__}: {e}"}
        results.append(r)
    elapsed = time.time() - t_start

    # Write JSON
    (OUT_DIR / "per_test.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Markdown report
    lines = [
        f"# Rigorous Pipeline Validation — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        f"**Validator lens**: adversarial multi-axis probe of post-2026-05-21-fixes state",
        f"**Wall-clock**: {elapsed:.0f}s",
        "",
        "## Verdict table",
        "",
        "| Test | Verdict | Notes |",
        "|---|:---:|---|",
    ]
    verdicts = []
    for r in results:
        v = r.get("verdict", "?")
        verdicts.append(v)
        notes = ""
        if "n_tested" in r and "n_ok" in r:
            notes = f"{r.get('n_ok', 0)}/{r.get('n_tested', 0)} OK"
            if "n_warn" in r: notes += f" / {r['n_warn']} warn / {r.get('n_fail',0)} fail"
        elif "n_files" in r:
            notes = f"{r.get('n_clean', 0)}/{r['n_files']} clean; {r.get('n_issues', 0)} issues"
        elif "n_dates_tested" in r:
            notes = f"{r.get('n_ok',0)}/{r['n_dates_tested']} dates pass"
        elif "n_probes" in r:
            notes = f"{r.get('n_pass', 0)}/{r['n_probes']} probes pass"
        emoji = "🟢" if v == "PASS" else ("🟡" if v == "WARN" else "🔴")
        lines.append(f"| {r['name']} | {emoji} {v} | {notes} |")
    lines.extend([
        "",
        "## Per-test detail",
        "",
    ])
    for r in results:
        lines.append(f"### {r['name']} — {r.get('verdict','?')}")
        lines.append("")
        if "per_asset" in r:
            lines.append("| Asset | rc | summary |")
            lines.append("|---|:---:|---|")
            for a in r["per_asset"]:
                if "rc" in a:
                    lines.append(f"| {a['asset']} | {a['rc']} | {a.get('summary','')[:120]} |")
                else:
                    lines.append(f"| {a.get('asset','?')} | - | {a.get('issue','')} |")
        if "sample_issues" in r and r["sample_issues"]:
            lines.append("\n**Sample issues:**")
            for i in r["sample_issues"][:5]:
                lines.append(f"- {i}")
        if "per_date" in r:
            lines.append("\n**Date-by-date xrel recompute check:**")
            for d in r["per_date"]:
                lines.append(f"- date={d['date']}: max_diff={d['max_diff_vs_recompute']:.2e} {'OK' if d.get('ok') else 'DIFF'}")
        if "probes" in r:
            for p in r["probes"]:
                lines.append(f"- **{p['probe']}**: {p.get('verdict','?')}")
                for k, v in p.items():
                    if k not in ("probe", "verdict"):
                        lines.append(f"    - {k}: {v}")
        lines.append("")

    # Headline
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_warn = sum(1 for v in verdicts if v == "WARN")
    n_fail = sum(1 for v in verdicts if v in ("FAIL", "EXCEPTION", "ERROR"))
    headline = "🟢 PASS" if n_fail == 0 and n_warn == 0 else (
                "🟡 WARN" if n_fail == 0 else "🔴 FAIL")
    lines.insert(2, f"**Headline**: {headline} ({n_pass} PASS, {n_warn} WARN, {n_fail} FAIL of {len(verdicts)})")
    lines.insert(3, "")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'='*78}")
    print(f"FINAL VERDICT: {headline}")
    print(f"{'='*78}")
    print(f"Output: {OUT_DIR.relative_to(ROOT)}/REPORT.md")


if __name__ == "__main__":
    main()
