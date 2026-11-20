"""Pre-train CI gate -- one-shot pipeline check before training a WM/ranker.

Composes all the upstream validators into a single fail-fast check:
  1. data_health_check         (registry coverage, freshness, schema, universe)
  2. validate_chimera           (per-asset 14-check)
  3. cross_asset_consistency   (xd_* sanity)
  4. pipeline_e2e_test         (12-stage integration)
  5. purge_split summary       (verify split sizes for the chosen cadence)

Run BEFORE any model training session. Exit 0 = clean, training can proceed.
Exit 1 = warns only, training can proceed but consider fixing.
Exit 2 = hard fails, training will likely produce broken models.

Usage:
  python src/pipeline/pre_train_gate.py                       # full check on all v51 assets
  python src/pipeline/pre_train_gate.py --asset BTC --quick   # just BTC, skip slow checks
  python src/pipeline/pre_train_gate.py --cadence 1d          # split summary at 1d
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def run(cmd: list[str], label: str, capture: bool = True) -> tuple[int, str]:
    print(f"\n[gate] === {label} ===")
    print(f"[gate]      $ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, timeout=600,
                           cwd=PROJECT_ROOT)
        out = r.stdout
        if not capture or r.returncode != 0:
            tail = "\n".join(out.splitlines()[-15:])
            print(tail)
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def _validate_chimera_legacy(asset: str) -> tuple[int, str]:
    """Lightweight check that chimera_legacy is healthy for the given asset.

    V1-V14 / V0 train on `data/processed/chimera_legacy/dollar/<sym>_v50_chimera_*.parquet`.
    The v51 validator (`validate_chimera.py`) does NOT cover this layer. This
    inline check verifies:
      1. At least one v50 parquet exists for the asset.
      2. Newest snapshot has >= 100 rows (sanity).
      3. Newest snapshot has the canonical f34 feature columns.
      4. Newest snapshot has target_return_{1,4,16,64}.

    Returns (rc, message). rc 0=PASS, 1=WARN, 2=FAIL.
    """
    from pathlib import Path as _P
    sym = asset.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    sym_l = sym.lower()
    legacy_dir = _P(__file__).resolve().parents[2] / "data" / "processed" / "chimera_legacy" / "dollar"
    files = sorted(legacy_dir.glob(f"{sym_l}_v50_chimera*.parquet"))
    if not files:
        return 2, f"  FAIL: no chimera_legacy file for {sym} in {legacy_dir}"

    newest = files[-1]   # alphabetic sort = date sort given YYYYMMDD suffix
    try:
        import polars as _pl
        schema = _pl.read_parquet_schema(newest)
        cols = set(schema.keys())
    except Exception as e:
        return 2, f"  FAIL: cannot read schema of {newest.name}: {e}"

    # Required: 13 base features + 4 targets at minimum (f13 is V1.0's smallest config)
    required_features_f13 = {
        "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
        "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
        "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
        "norm_spread_bps",
    }
    required_targets = {"target_return_1", "target_return_4",
                        "target_return_16", "target_return_64"}

    missing_features = required_features_f13 - cols
    missing_targets = required_targets - cols
    issues = []
    if missing_features:
        issues.append(f"missing f13 features: {sorted(missing_features)[:5]}")
    if missing_targets:
        issues.append(f"missing targets: {sorted(missing_targets)}")

    # Row sanity (cheap: just read one column)
    try:
        n_rows = _pl.scan_parquet(newest).select(_pl.len()).collect().item()
    except Exception:
        n_rows = -1
    if n_rows < 100:
        issues.append(f"row count {n_rows} too low")

    msg_parts = [f"  asset:    {sym}",
                 f"  newest:   {newest.name}",
                 f"  files:    {len(files)} v50 snapshots",
                 f"  rows:     {n_rows:,}" if n_rows > 0 else f"  rows:     <unread>",
                 f"  features: {len(cols)} cols, f13 minimum present"
                 if not missing_features else
                 f"  features: {len(cols)} cols, MISSING f13: {sorted(missing_features)[:3]}"]

    if issues:
        return 2, "\n".join(msg_parts + [f"  FAIL: {' | '.join(issues)}"])
    return 0, "\n".join(msg_parts + ["  OK: f13 features + 4 targets present"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC",
                    help="Asset for per-asset checks (default BTC). "
                         "Use --universe to gate across multiple assets instead.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Run per-asset checks across every asset in the named "
                         "universe. Overrides --asset when set. Returns the WORST "
                         "verdict across all assets (any rc=2 -> hard fail).")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--layer", default="both", choices=["legacy", "v51", "both"],
                    help="Which chimera layer to validate. 'legacy' = v50 chimera "
                         "(used by V1-V14 / V0); 'v51' = new v51 chimera (used by "
                         "V19 / frontier strategies / meta-learner); 'both' = both. "
                         "Default 'both' preserves prior behavior. Pass 'legacy' "
                         "when training V1-V14 to skip the v51 check (avoids "
                         "false-fail when v51 build is incomplete but v50 is fine).")
    ap.add_argument("--quick", action="store_true", help="Skip slow checks (raw scan)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="No-op stub. Pre-train gate is a read-only validator "
                         "with no cached output, but accepts --force so that "
                         "refresh.py --force can pass it uniformly to every "
                         "stage without per-stage guards.")
    args = ap.parse_args()

    # Resolve asset list: --universe overrides --asset.
    if args.universe:
        try:
            import sys as _sys
            from pathlib import Path as _P
            _sys.path.insert(0, str(_P(__file__).resolve().parent))
            from universe_loader import UniverseLoader  # noqa: E402
            assets = [s.upper().replace("USDT", "")
                      for s in UniverseLoader.load().list(args.universe)]
            print(f"[gate] Universe: {args.universe} ({len(assets)} assets)")
        except Exception as e:
            print(f"[gate] FALLBACK: universe={args.universe} load failed ({e}); "
                  f"using --asset {args.asset}")
            assets = [args.asset]
    else:
        assets = [args.asset]

    print(f"[gate] PRE-TRAIN GATE -- {datetime.now(timezone.utc).isoformat()}")
    print(f"[gate] Assets: {','.join(assets)}, Cadence: {args.cadence}, Quick: {args.quick}")

    # If multi-asset, run gate per asset and aggregate worst verdict.
    if len(assets) > 1:
        worst_rc = 0
        per_asset_results = []
        for a in assets:
            print(f"\n{'#' * 70}\n# GATE FOR ASSET: {a}\n{'#' * 70}")
            args.asset = a   # mutate to drive the per-asset block below
            args.universe = None  # prevent recursion
            sub_rc = _run_single_asset(args)
            per_asset_results.append((a, sub_rc))
            worst_rc = max(worst_rc, sub_rc)
        print(f"\n{'=' * 70}\n[gate] MULTI-ASSET VERDICT\n{'=' * 70}")
        for a, rc in per_asset_results:
            tag = "PASS" if rc == 0 else ("WARN" if rc == 1 else "FAIL")
            print(f"  {tag}  {a}  (rc={rc})")
        print(f"\n[gate] Worst rc across {len(assets)} assets: {worst_rc}")
        sys.exit(worst_rc)

    # Single-asset path (legacy behavior).
    sys.exit(_run_single_asset(args))


def _run_single_asset(args) -> int:
    """Single-asset gate body. Returns rc (0=clean, 1=warn, 2=fail).

    All `sys.exit(N)` calls in this function were converted to `return N` so
    the multi-asset wrapper can aggregate verdicts across the universe.
    """
    results = []

    # 0. Registry contract test — runs ONCE per gate invocation, before any
    # per-asset work. Catches feature_registry.yaml edits that break the
    # downstream chimera build (label-leak token, orphan sources_to_join,
    # prefix collision, missing schema_version). Added 2026-05-22 per
    # oracle pipeline-A+ closure.
    cmd = [PYTHON, "src/pipeline/registry_contract_test.py"]
    rc_reg, out_reg = run(cmd, "0. Registry contract test")
    results.append({"check": "registry_contract", "rc": rc_reg,
                    "asset": args.asset, "parsed": True})
    # CRITICAL on rc=2 (e.g. missing schema_version, reserved-name violation).
    # rc=1 is WARN-only (zero data files, missing prefix) — gate continues.

    # 1. Data health
    cmd = [PYTHON, "src/pipeline/data_health_check.py"]
    if args.quick:
        cmd.append("--quick")
    rc, out = run(cmd, "1. Data health check")
    # exit code: 0=clean, 2=hard fail. (1 means warns only — accept for gate)
    n_pass = n_warn = n_fail = 0
    parsed = False
    for line in out.splitlines():
        if "Summary:" in line and "pass" in line:
            try:
                # "[health] Summary: 35 pass, 11 warn, 0 fail"
                parts = line.replace("[health] Summary:", "").strip().split(",")
                n_pass = int(parts[0].strip().split()[0])
                n_warn = int(parts[1].strip().split()[0])
                n_fail = int(parts[2].strip().split()[0])
                parsed = True
            except Exception as e:
                # Format drift: don't silently mask. Mark as parse_error so
                # downstream verdict treats it as warn (rc still respected).
                print(f"  [gate] WARN: data_health summary parse failed: {e}", flush=True)
    results.append({"check": "data_health", "rc": rc,
                    "pass": n_pass, "warn": n_warn, "fail": n_fail,
                    "parsed": parsed})

    # 2a. Legacy chimera (v50) validator -- inline lightweight check.
    # V1-V14 + V0 train on chimera_legacy/dollar/. v51 (frontier) is a
    # separate layer for V19 and downstream strategies.
    if args.layer in ("legacy", "both"):
        rc_legacy, leg_msg = _validate_chimera_legacy(args.asset)
        print(f"\n[gate] === 2a. Chimera_legacy (v50) validation ===")
        print(leg_msg)
        results.append({"check": "chimera_legacy", "rc": rc_legacy,
                        "asset": args.asset, "msg": leg_msg.split("\n")[0],
                        "parsed": True})

    # 2b. Chimera v51 v2 validator (only when v51 layer is requested)
    if args.layer in ("v51", "both"):
        cmd = [PYTHON, "src/pipeline/validate_chimera.py", "--asset", args.asset]
        rc, out = run(cmd, "2b. Chimera v51 v2 validation")
        n_pass = n_warn = n_fail = 0
        parsed = False
        for line in out.splitlines():
            if "Summary:" in line and "clean" in line:
                try:
                    # "Summary: 0 clean, 1 warn-only, 0 fail (of 1)"
                    parts = line.replace("Summary:", "").split("(")[0].split(",")
                    n_pass = int(parts[0].strip().split()[0])
                    n_warn = int(parts[1].strip().split()[0])
                    n_fail = int(parts[2].strip().split()[0])
                    parsed = True
                except Exception as e:
                    print(f"  [gate] WARN: chimera_v51 summary parse failed: {e}", flush=True)
        results.append({"check": "chimera_v51", "rc": rc,
                        "clean": n_pass, "warn": n_warn, "fail": n_fail,
                        "parsed": parsed})

        # 2c. R32+ pipeline-audit CRIT-3 fix: FeatureRegistry column-count +
        # staleness + per-prefix null-rate gate. Catches the lob_proxy 99.2%
        # null + stale-chimera classes of bug that previously slipped past 2b.
        try:
            import datetime as _dt
            from pathlib import Path as _Path
            import polars as _pl
            chimera_dir = _Path("data/processed/chimera/dollar")
            if chimera_dir.exists():
                # Latest dated file for this asset
                files = sorted(chimera_dir.glob(f"{args.asset.lower()}usdt_v51_chimera_*.parquet"))
                if files:
                    latest = files[-1]
                    # Staleness: file mtime
                    mtime = _dt.datetime.fromtimestamp(latest.stat().st_mtime)
                    age_days = (_dt.datetime.now() - mtime).days
                    stale = age_days > 14
                    # Column count + null-rate scan.
                    # 2026-05-21 BUG FIX: previously `n_rows=1000` read the FIRST
                    # 1000 rows of the chimera. For dollar bars (millions of rows
                    # spanning years), the first 1000 rows are from the asset's
                    # earliest history when many features were not yet computed
                    # (e.g., BTC's first 1000 dollar bars are from 2017-08, before
                    # OI/funding/etc. were tracked). Result: false-positive
                    # `dead_features` reports for features that ARE active in
                    # current data. Fix: sample the LAST 10000 rows so the gate
                    # reflects current-data signal, not historical-data sparsity.
                    n_total = _pl.scan_parquet(latest).select(_pl.len()).collect().item()
                    sample_n = min(10000, n_total)
                    df_head = _pl.scan_parquet(latest).slice(
                        max(0, n_total - sample_n), sample_n).collect()
                    n_cols = len(df_head.columns)
                    # Per-prefix null fraction ACROSS ALL columns of family
                    # R32++ Lane B fix (validator HIGH): previously only checked
                    # first column; partial-fill bugs in cols 2-N silently passed.
                    prefix_nulls = {}
                    for prefix in ("lob_", "bd_", "xex_", "dv_", "soc_", "te_",
                                     "etf_", "stbl_", "wh_", "liq_", "mv_", "hbr_", "rv_"):
                        pfx_cols = [c for c in df_head.columns if c.startswith(prefix)]
                        if not pfx_cols:
                            continue
                        total_cells = len(pfx_cols) * len(df_head)
                        if total_cells == 0:
                            continue
                        total_nulls = sum(int(df_head[c].is_null().sum()) for c in pfx_cols)
                        null_frac = float(total_nulls / total_cells)
                        prefix_nulls[prefix] = round(null_frac, 3)
                    # 2026-05-21: SPARSE-BY-DESIGN prefixes — their producers
                    # only compute for a top-N subset of u100 by design:
                    #   - dv_  : Deribit DVOL (BTC/ETH only)
                    #   - xex_ : Coinbase cross-exchange (5 names)
                    #   - soc_ : Wikipedia pageviews (top-10)
                    #   - etf_ : ETF flows (BTC/ETH spot ETFs)
                    #   - te_  : Transfer-entropy panel (10 leader assets)
                    # Excluded from high_null_prefixes gate so non-covered assets
                    # don't HARD-FAIL just for lacking these limited sources.
                    SPARSE_BY_DESIGN = {"dv_", "xex_", "soc_", "etf_", "te_"}
                    high_null_prefixes = {p: f for p, f in prefix_nulls.items()
                                           if f > 0.80 and p not in SPARSE_BY_DESIGN}
                    # R32+++ pipeline-HIGH fix: per-feature std check.
                    # Catches the norm_funding=0 / dead-norm class of bug where
                    # a feature loads non-null but is constant (std~0). Such
                    # features pass null-rate gates but contribute zero signal
                    # to training. Sample only norm_* (the normalized layer
                    # post-RankGauss) since raw features may legitimately be
                    # rank-bounded.
                    dead_features: list[str] = []
                    norm_cols = [c for c in df_head.columns
                                    if c.startswith("norm_")]
                    for c in norm_cols:
                        try:
                            std_val = float(df_head[c].std())
                            if std_val < 1e-6:
                                dead_features.append(c)
                        except Exception:
                            continue
                    # R32+++ pipeline-crawler-HIGH fix: high-null prefix is
                    # CRITICAL (dead features entering training), NOT a warning.
                    # Previously rc=1 (WARN) → run_all_training.py treated as
                    # non-blocking → chimera with 99.2% null lob_ proceeded
                    # to model training. Now: rc=2 if high_null_prefixes OR
                    # dead_features, rc=1 if just stale.
                    if high_null_prefixes or dead_features:
                        _rc = 2
                    elif stale:
                        _rc = 1
                    else:
                        _rc = 0
                    results.append({
                        "check": "chimera_v51_extended",
                        "rc": _rc,
                        "asset": args.asset,
                        "latest_file": latest.name,
                        "age_days": age_days,
                        "stale": stale,
                        "n_cols": n_cols,
                        "prefix_null_rates": prefix_nulls,
                        "high_null_prefixes": high_null_prefixes,
                        "dead_features": dead_features,
                        "parsed": True,
                    })
                    if stale:
                        print(f"  [gate] WARN: chimera for {args.asset} is {age_days}d stale "
                                f"(>{14}d threshold)", flush=True)
                    if high_null_prefixes:
                        print(f"  [gate] WARN: {args.asset} has high-null prefix families: "
                                f"{high_null_prefixes}", flush=True)
                    if dead_features:
                        print(f"  [gate] CRIT: {args.asset} has {len(dead_features)} "
                                f"dead norm_ features (std<1e-6): {dead_features[:5]}"
                                f"{'...' if len(dead_features) > 5 else ''}",
                                flush=True)
        except ImportError:
            results.append({
                "check": "chimera_v51_extended", "rc": 1,
                "asset": args.asset, "error": "polars unavailable",
                "parsed": True,
            })
        except Exception as e:
            # R32+++ validator-crawler-HIGH fix: broadened from
            # (OSError, ValueError) to Exception. Polars raises ComputeError /
            # ArrowError which aren't OSError/ValueError subclasses -> the
            # tight catch silently let those crash the entire gate function
            # instead of appending a fail record. Verdict loop would see
            # zero checks instead of one fail.
            print(f"  [gate] WARN: chimera_v51_extended check failed: {e}", flush=True)
            results.append({
                "check": "chimera_v51_extended", "rc": 2,
                "asset": args.asset, "error": str(e),
                "parsed": True,
            })

    # 3. Cross-asset consistency (skip if --asset is BTC alone, else check pair)
    cmd = [PYTHON, "src/pipeline/cross_asset_consistency.py",
           "--assets", f"BTC,{args.asset}" if args.asset.upper() != "BTC" else "BTC"]
    rc, out = run(cmd, "3. Cross-asset xd_* consistency")
    n_pass = n_warn = n_fail = 0
    parsed = False
    for line in out.splitlines():
        if "Summary:" in line and "pass" in line:
            try:
                # "[xd_audit] Summary: 4 pass, 0 warn, 0 fail, 0 skip"
                parts = line.split("Summary:")[1].strip().split(",")
                n_pass = int(parts[0].strip().split()[0])
                n_warn = int(parts[1].strip().split()[0])
                n_fail = int(parts[2].strip().split()[0])
                parsed = True
            except Exception as e:
                print(f"  [gate] WARN: xd_consistency summary parse failed: {e}", flush=True)
    results.append({"check": "xd_consistency", "rc": rc,
                    "pass": n_pass, "warn": n_warn, "fail": n_fail,
                    "parsed": parsed})

    # 4. End-to-end test
    cmd = [PYTHON, "src/pipeline/pipeline_e2e_test.py", "--asset", args.asset]
    rc, out = run(cmd, "4. E2E pipeline integration")
    n_stages_pass = 0
    n_stages_total = 0
    parsed = False
    for line in out.splitlines():
        if "Summary:" in line and "stages" in line:
            try:
                # "[e2e] Summary: 12/12 stages PASS"
                parts = line.split("Summary:")[1].strip().split("/")
                n_stages_pass = int(parts[0].strip())
                n_stages_total = int(parts[1].strip().split()[0])
                parsed = True
            except Exception as e:
                print(f"  [gate] WARN: e2e summary parse failed: {e}", flush=True)
    results.append({"check": "e2e", "rc": rc,
                    "pass": n_stages_pass, "total": n_stages_total,
                    "parsed": parsed})

    # 5. Split summary
    cmd = [PYTHON, "src/pipeline/purge_split.py", "--asset", args.asset, "--cadence", args.cadence]
    rc, out = run(cmd, "5. Purge-aware split summary")
    train_rows = val_rows = oos_rows = unseen_rows = 0
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("train_rows"):
            train_rows = int(s.split()[-1].replace(",", ""))
        if s.startswith("val_rows"):
            val_rows = int(s.split()[-1].replace(",", ""))
        if s.startswith("oos_rows"):
            oos_rows = int(s.split()[-1].replace(",", ""))
        if s.startswith("unseen_rows"):
            unseen_rows = int(s.split()[-1].replace(",", ""))
    results.append({"check": "split", "rc": rc,
                    "train": train_rows, "val": val_rows, "oos": oos_rows, "unseen": unseen_rows})

    # Final verdict
    print()
    print("=" * 80)
    print("[gate] FINAL VERDICT")
    print("=" * 80)
    total_fails = 0
    total_warns = 0
    for r in results:
        marker = "OK"
        if r.get("fail", 0) > 0 or r.get("rc", 0) >= 2:
            marker = "FAIL"
            total_fails += 1
        elif r.get("warn", 0) > 0 or r.get("rc", 0) == 1:
            marker = "WARN"
            total_warns += 1
        # Format-drift guard: if a sub-validator has rc=0 but parser failed,
        # we cannot assert OK -- the Summary line format drifted and counts
        # are stale. Treat as WARN so the gate flags it for review.
        elif "parsed" in r and not r["parsed"] and r.get("rc", 0) == 0:
            marker = "WARN"
            total_warns += 1
            r["parse_drift"] = True
        print(f"  {marker:>4}  {r['check']:<20}  {r}")
    print()
    if total_fails > 0:
        print(f"[gate] HARD FAIL: {total_fails} checks failed; do NOT train until fixed.")
        return 2
    if total_warns > 0:
        print(f"[gate] WARNS: {total_warns} checks have warnings (not blocking).")
        return 1
    print("[gate] CLEAN: all checks pass. Training can proceed.")
    return 0


if __name__ == "__main__":
    main()
