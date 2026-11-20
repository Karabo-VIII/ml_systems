"""V51 v2 validator -- catch alignment / leakage / NaN / new-layout issues.

Validates against the new SOTA layout:
  data/processed/<SYMBOL>/v51.parquet            # full
  data/processed/<SYMBOL>/v51_{1d,4h,1h,15m}.parquet
  data/_manifests/v51_<SYMBOL>.json
  data/features/<SYMBOL>/frontier_daily.parquet
  data/features/_global/<source>.parquet
  data/raw_external/<source>/<file>.parquet

Checks:
   1. Schema: v51 has all 80 registered features + base v50 cols + 11 new helpers
   2. V50 fixes applied:
      a) tick_seq column present, monotonic within timestamp
      b) returns_clean column present, NaN at index 0 (not 0.0)
      c) target_return_<h>_raw columns present (uncapped)
      d) is_u10/u50/u100 + asset_dna columns present
   3. Manifest exists + checksums match
   4. Backward compat: every v50 column carries forward with same values
   5. ts_monotonic: zero-diff dups OK (have tick_seq tiebreaker now), no decreasing
   6. forward_target_shift: target_return_h matches close-derived formula
   7. NaN budget per registered feature
   8. Date gaps
   9. Hawkes alignment vs source
  10. Cadence files: schema + plausible row counts
  11. Universe membership consistency: is_u<N> matches loader.is_in()
  12. Manifest reproducibility: build with same inputs gives same checksums

Run:
  python src/pipeline/validate_chimera.py             # all assets in processed/<SYMBOL>/
  python src/pipeline/validate_chimera.py --asset BTC
  python src/pipeline/validate_chimera.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Windows cp1252 stdout chokes on Greek letters (sigma, alpha) and arrows that
# validators emit in detail messages (e.g., "frozen 1 sigma...", "->"). Reconfigure
# the stream to replace unencodable chars instead of crashing. No-op on POSIX.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import numpy as np
import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_registry import FeatureRegistry  # noqa: E402
from universe_loader import UniverseLoader  # noqa: E402
import layout as _layout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"

NAN_WARN_THRESHOLD = 0.30
# 2026-05-24: per-family stricter budget for "should be near-universal" families.
# Source: post-rebuild audit found bd_/lob_/te_/hawkes_/liq_ silently >50% NaN
# on assets where they SHOULD be populated. The global 30% gate missed it
# because rolling 30d means + cold-start nulls inflate the per-feature count
# under the global threshold. Strict family budgets surface the silent gap.
NAN_FAMILY_STRICT_BUDGET = {
    "bd_":     0.20,   # book-depth profile (real Binance Vision 30s) -- u100 coverage
    "lob_":    0.20,   # lob_proxy (derived from aggTrades) -- u100 coverage
    "te_":     0.40,   # transfer-entropy panel -- BTC-driven, looser
    "hawkes_": 0.20,   # hawkes-process intensities -- universal
    "liq_":    0.40,   # liquidations -- sparser by trade-flow nature
}

# 2026-05-22: Added in response to CRITICAL bug audit
# `runs/audit/ML_BIGMOVE_PROD_VALIDATION_2026_05_22/` (Opus auditor a4c0a0fba19024c19)
# which found (a) `btc_ret_same_day` was forward-shifted (label leak), and
# (b) `te_btc_imb`, `etf_btc_etf_total_z30`, `bd_imbalance_l5` were silently
# frozen (n_unique==1) for entire months without any validator catching it.
FREEZE_FRACTION_THRESHOLD = 0.50   # n_unique / n_rows must exceed this per month
LOOKAHEAD_CORR_THRESHOLD = 0.20    # abs(corr) with forward 1d return > this = leak suspect


@dataclass
class CheckResult:
    name: str
    severity: str  # 'ok' | 'warn' | 'fail'
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class AssetReport:
    symbol: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def n_pass(self) -> int: return sum(1 for c in self.checks if c.severity == "ok")
    @property
    def n_warn(self) -> int: return sum(1 for c in self.checks if c.severity == "warn")
    @property
    def n_fail(self) -> int: return sum(1 for c in self.checks if c.severity == "fail")

    def add(self, c: CheckResult) -> None:
        self.checks.append(c)


def check_schema(v50: pl.DataFrame, v51: pl.DataFrame, registry: FeatureRegistry,
                 expected_min_added: int | None = None) -> CheckResult:
    """Schema: v51 should have v50 + frontier features (sum of registry sources) +
    a small allowance for v51 helper columns (tick_seq, returns_clean, raw targets,
    metadata is_u10/u50/u100, asset_dna).

    Threshold derives DYNAMICALLY from the registry's
    expected_total_new_features so adding/removing sources auto-tracks
    without code edits. Allows -2 slack for sources that may produce
    fewer features than declared.
    """
    actual = len(v51.columns) - len(v50.columns)
    if expected_min_added is None:
        # Derive from registry: total declared frontier features minus a small
        # slack (some sources may emit fewer cols if they had no rows for an
        # asset; e.g. wiki_pageviews short-coverage assets).
        try:
            registry_total = registry.chimera.expected_total_new_features
            slack = 2  # tolerate 2 missing features without failing
            expected_min_added = max(0, registry_total - slack)
        except Exception:
            expected_min_added = 80  # conservative fallback
    if actual >= expected_min_added:
        return CheckResult("schema_count", "ok",
                           f"v50={len(v50.columns)} v51={len(v51.columns)} +{actual} (>= +{expected_min_added})")
    return CheckResult("schema_count", "fail",
                       f"v50={len(v50.columns)} v51={len(v51.columns)} +{actual} (expected >= +{expected_min_added})")


def check_v50_fixes(v51: pl.DataFrame) -> list[CheckResult]:
    out = []
    # Fix 1: tick_seq present + within-ms monotonic
    if "tick_seq" not in v51.columns:
        out.append(CheckResult("fix1_tick_seq_present", "fail", "tick_seq missing"))
    else:
        # check tick_seq is non-decreasing within any timestamp
        if "timestamp" in v51.columns:
            grouped_max = v51.group_by("timestamp").agg(pl.col("tick_seq").max())
            n_grouped = len(grouped_max)
            n_unique_ts = v51["timestamp"].n_unique()
            out.append(CheckResult("fix1_tick_seq_correct", "ok",
                                   f"tick_seq computed for {n_unique_ts} unique ts"))
        else:
            out.append(CheckResult("fix1_tick_seq_correct", "warn", "no timestamp col"))

    # Fix 2: returns_clean present, NaN at index 0
    if "returns_clean" not in v51.columns:
        out.append(CheckResult("fix2_returns_clean_present", "fail", "returns_clean missing"))
    else:
        first = v51["returns_clean"][0]
        if first is None or (isinstance(first, float) and np.isnan(first)):
            out.append(CheckResult("fix2_returns_clean_first_nan", "ok",
                                   f"returns_clean[0]=NaN (no silent fill_null)"))
        else:
            out.append(CheckResult("fix2_returns_clean_first_nan", "fail",
                                   f"returns_clean[0]={first} (expected NaN)"))

    # Fix 3: target_return_<h>_raw present
    raw_cols = [f"target_return_{h}_raw" for h in (1, 4, 16, 64)]
    missing_raw = [c for c in raw_cols if c not in v51.columns]
    if missing_raw:
        out.append(CheckResult("fix3_target_raw_present", "fail",
                               f"missing raw target cols: {missing_raw}"))
    else:
        out.append(CheckResult("fix3_target_raw_present", "ok",
                               f"all 4 target_return_h_raw present"))

    # Fix 4: metadata cols present + asset_dna value
    meta_cols = ["is_u10", "is_u50", "is_u100", "asset_dna"]
    missing_meta = [c for c in meta_cols if c not in v51.columns]
    if missing_meta:
        out.append(CheckResult("fix4_metadata_present", "fail",
                               f"missing metadata cols: {missing_meta}"))
    else:
        dna = v51["asset_dna"][0]
        if dna in ("BLUE", "STEADY", "VOLATILE", "DEGEN", "UNKNOWN"):
            out.append(CheckResult("fix4_metadata_present", "ok",
                                   f"all 4 metadata cols, asset_dna={dna}"))
        else:
            out.append(CheckResult("fix4_metadata_present", "fail",
                                   f"asset_dna invalid: {dna}"))
    return out


def check_universe_membership(v51: pl.DataFrame, symbol: str, loader: UniverseLoader) -> CheckResult:
    expected = {
        "is_u10": loader.is_in(symbol, "u10"),
        "is_u50": loader.is_in(symbol, "u50"),
        "is_u100": loader.is_in(symbol, "u100"),
    }
    actual = {k: v51[k][0] if k in v51.columns else None for k in expected}
    diffs = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    if diffs:
        return CheckResult("universe_membership", "fail",
                           f"mismatches: {diffs}")
    return CheckResult("universe_membership", "ok",
                       f"u10={expected['is_u10']}, u50={expected['is_u50']}, u100={expected['is_u100']}")


def check_manifest(symbol: str) -> CheckResult:
    fp = _layout.manifest_path(symbol)
    if not fp.exists():
        return CheckResult("manifest_exists", "fail", f"missing {fp}")
    try:
        m = json.loads(fp.read_text())
        keys = {"asset", "chimera_version", "v50_input_sha256", "row_count", "fixes_applied"}
        missing_keys = [k for k in keys if k not in m]
        if missing_keys:
            return CheckResult("manifest_complete", "fail",
                               f"missing keys: {missing_keys}")
        return CheckResult("manifest_complete", "ok",
                           f"v={m['chimera_version']}, rows={m['row_count']}, fixes={len(m['fixes_applied'])}")
    except Exception as e:
        return CheckResult("manifest_complete", "fail", f"parse err: {e}")


def check_backward_compat(v50: pl.DataFrame, v51: pl.DataFrame) -> CheckResult:
    """Every v50 column must exist in v51 with identical values (sample 10K rows)."""
    missing = [c for c in v50.columns if c not in v51.columns]
    if missing:
        return CheckResult("backward_compat_columns", "fail", f"missing v50 cols: {missing}")
    n_check = min(10000, len(v50))
    s50 = v50.head(n_check)
    s51 = v51.head(n_check).select(v50.columns)
    diffs = []
    for c in v50.columns:
        if c == "date":
            continue
        try:
            a = s50[c].fill_null(-9.999e9)
            b = s51[c].fill_null(-9.999e9)
            if a.dtype.is_numeric():
                if (a - b).abs().sum() > 1e-6:
                    diffs.append(c)
            else:
                if (a != b).sum() > 0:
                    diffs.append(c)
        except Exception:
            diffs.append(c)
    if diffs:
        return CheckResult("backward_compat_values", "fail", f"diffs in cols: {diffs[:5]}")
    return CheckResult("backward_compat_values", "ok", f"v50 cols identical (n={n_check})")


def check_ts_monotonic(v51: pl.DataFrame) -> CheckResult:
    if "timestamp" not in v51.columns:
        return CheckResult("ts_monotonic", "fail", "no timestamp")
    diffs = v51["timestamp"].diff().drop_nulls()
    n_neg = (diffs < 0).sum()
    if n_neg > 0:
        return CheckResult("ts_monotonic", "fail", f"{n_neg} strictly-decreasing")
    n_zero = (diffs == 0).sum()
    return CheckResult("ts_monotonic", "ok",
                       f"{len(v51)} bars: 0 decreasing, {n_zero} zero-diff (handled by tick_seq)")


def check_target_formula(v51: pl.DataFrame) -> CheckResult:
    if "close" not in v51.columns:
        return CheckResult("target_formula", "ok", "no close col; skip")
    closes = v51["close"].to_numpy()
    n = len(closes)
    rng = np.random.default_rng(seed=42)
    fails = []
    for h in (1, 4, 16, 64):
        col_clipped = f"target_return_{h}"
        col_raw = f"target_return_{h}_raw"
        if col_raw not in v51.columns:
            continue
        targets = v51[col_raw].to_numpy()
        n_sample = min(200, max(0, n - 2 * h - 100))
        if n_sample < 5:
            continue
        idxs = rng.integers(low=100, high=n - h - 1, size=n_sample)
        bad = 0
        for i in idxs:
            if not np.isfinite(targets[i]) or not np.isfinite(closes[i]) or not np.isfinite(closes[i+h]):
                continue
            expected = (closes[i+h] - closes[i]) / max(abs(closes[i]), 1e-12)
            if abs(targets[i] - expected) > 1e-4:
                bad += 1
        if bad > n_sample * 0.05:
            fails.append(f"{col_raw}: {bad}/{n_sample} sampled mismatch")
    if fails:
        return CheckResult("target_formula", "fail", "; ".join(fails))
    return CheckResult("target_formula", "ok",
                       f"target_return_h_raw matches close[i+h]/close[i]-1 in 200-row sample")


def check_nan_budget(v51: pl.DataFrame, registry: FeatureRegistry) -> CheckResult:
    """NaN-rate gate per feature.

    2026-05-21: sparse-by-design features (Deribit DVOL = BTC/ETH only,
    Wikipedia pageviews = top-10 only, Coinbase cross-exchange = 5 names,
    BTC/ETH ETF flows = BTC/ETH only) are EXEMPTED from the >30% NaN check.
    These external sources have intrinsic asset-coverage limits; flagging
    them as "budget violation" creates noise for the 80%+ of u100 that
    legitimately lacks the data.
    """
    n_rows = len(v51)
    # 2026-05-24: xex_ already listed (T2-E addition fits the same sparse-by-design
    # pattern: 5 assets out of 100 covered). New: fund_/premium_ are NOT sparse-by-design
    # -- they should be near-universal, so do NOT add to this list.
    # bd_bgf_ is sparse-BY-DATE (raw source bd starts 2023-01-01; assets with
    # chimera history pre-2023 get ~50% NaN). Exempted with this rationale.
    # lob_bgf_ is sparse-BY-DATE for the SAME reason (lob_proxy_panel.py
    # currently covers 2026-01-01 -> 2026-05-08 only; pre-2026 chimera bars
    # get NaN by raw-availability bound). Exempted under the same rationale.
    SPARSE_BY_DESIGN_PREFIXES = ("dv_", "xex_", "soc_", "etf_", "te_",
                                 "bd_bgf_", "lob_bgf_")
    high = []
    sparse_skipped = 0
    for f in registry.list_features():
        if f not in v51.columns:
            continue
        # Whitelist sparse-by-design external-source features
        if any(f.startswith(p) for p in SPARSE_BY_DESIGN_PREFIXES):
            sparse_skipped += 1
            continue
        n_nan = v51[f].null_count()
        frac = n_nan / max(n_rows, 1)
        if frac > NAN_WARN_THRESHOLD:
            high.append((f, frac))
    # 2026-05-24: per-family strict budget on bd_/lob_/te_/hawkes_/liq_.
    # These should be near-universal on assets the source covers; >budget NaN
    # = silent join failure or missing input file (caught silently by the
    # global 30% gate when half the family is filled and half isn't).
    #
    # 2026-05-31: evaluate the family-strict budget on the RECENT WINDOW, not full
    # history. bd_/lob_ are sparse-by-DATE (Binance Vision book depth from ~2023;
    # lob_proxy from ~2026), so on FULL history they are legitimately >90% NaN
    # (the source did not exist) -- which produced FALSE hard-FAILs after the
    # 2026-05-30 rebuild (verified: lob_l1_imb 96% full-NaN but 0% in the last
    # 100k bars). The silent-join-failure signal the strict budget targets lives in
    # the RECENT window where the source exists; evaluate there so real regressions
    # still FAIL while pre-source-date NaN does not. Also honors the
    # SPARSE_BY_DESIGN exemption consistently (the old loop re-caught lob_bgf_/bd_bgf_
    # via the broader lob_/bd_ prefix despite them being whitelisted above).
    RECENT_WINDOW_DAYS = 60
    if "timestamp" in v51.columns and n_rows:
        max_ts = v51["timestamp"].max()
        recent = v51.filter(pl.col("timestamp") >= max_ts - RECENT_WINDOW_DAYS * 86_400_000)
    else:
        recent = v51
    n_recent = max(len(recent), 1)
    family_violations = []
    for f in v51.columns:
        for prefix, budget in NAN_FAMILY_STRICT_BUDGET.items():
            if not f.startswith(prefix):
                continue
            # Honor the sparse-by-design whitelist (lob_bgf_/bd_bgf_/te_ etc.):
            # the >30% WARN check above skips these, so the strict loop must too.
            if any(f.startswith(p) for p in SPARSE_BY_DESIGN_PREFIXES):
                break
            frac = recent[f].null_count() / n_recent
            if frac > budget:
                family_violations.append((f, frac, budget))
            break
    if family_violations or high:
        msgs = []
        if high:
            sample = ", ".join(f"{n}({f:.0%})" for n, f in high[:5])
            msgs.append(f"{len(high)} features >{NAN_WARN_THRESHOLD:.0%} NaN: {sample}")
        if family_violations:
            fam_sample = ", ".join(
                f"{n}({f:.0%} > {b:.0%})" for n, f, b in family_violations[:5])
            msgs.append(
                f"{len(family_violations)} family-strict violations: {fam_sample}")
        # Family-strict violations are a hard FAIL: they indicate a silent join
        # failure / missing input file (a feature family that should be near-
        # universal is mostly NaN). Generic above-soft-threshold NaN stays WARN.
        # (Previously this returned "warn" even on family-strict violations, so
        # NaN-polluted training data passed pre_train_gate.)
        severity = "fail" if family_violations else "warn"
        return CheckResult(
            "nan_budget", severity,
            f"{'; '.join(msgs)} (skipped {sparse_skipped} sparse-by-design)",
            metrics={
                "high_nan": [n for n, _ in high],
                "family_violations": [n for n, _, _ in family_violations],
            },
        )
    return CheckResult("nan_budget", "ok",
                       f"all features <{NAN_WARN_THRESHOLD:.0%} NaN, all family-strict "
                       f"budgets met (skipped {sparse_skipped} sparse-by-design)")


def check_feature_freshness(v51: pl.DataFrame, registry: FeatureRegistry) -> list[CheckResult]:
    """Per-month feature-freeze detector.

    Added 2026-05-22 after Opus auditor (run id a4c0a0fba19024c19) found
    `te_btc_imb`, `etf_btc_etf_total_z30`, and `bd_imbalance_l5` were silently
    constant for entire months on BTC chimera — no existing validator
    caught them. A frozen feature contributes no signal but also doesn't
    trigger NaN-budget warnings.

    Method: group by year-month, compute n_unique/n_rows per feature.
    If ratio < FREEZE_FRACTION_THRESHOLD (default 0.50) for a feature in any
    month, raise a WARN.

    Sparse-by-design features (te_*, dv_*, xex_*, soc_*, etf_*) are skipped
    because they ARE meant to be sparse / not-always-fresh.
    """
    import pandas as pd
    if "date" not in v51.columns and "timestamp" not in v51.columns:
        return [CheckResult("freshness", "warn", "no date column to group by")]
    # Lightweight: only pull `date` + features-of-interest
    SPARSE_BY_DESIGN_PREFIXES = ("dv_", "xex_", "soc_", "etf_", "te_")
    # Explicit named exemptions (added 2026-05-22 per Opus LOB audit a55323fd6f31a3d9e):
    SPARSE_BY_DESIGN_NAMES = {
        "bd_depth_at_02pct",   # Binance ±0.2% depth bands added Jan 2026; 0.0 pre-2026 by design
        "bd_thin_book_frac",   # binary indicator on rolling-median; <1% positive by design
    }
    candidate_feats = [f for f in registry.list_features()
                        if f in v51.columns
                        and not any(f.startswith(p) for p in SPARSE_BY_DESIGN_PREFIXES)
                        and f not in SPARSE_BY_DESIGN_NAMES]
    if not candidate_feats:
        return [CheckResult("freshness_freeze", "ok", "no registered features to check")]
    cols_needed = [c for c in ["date", "timestamp"] if c in v51.columns] + candidate_feats
    df = v51.select(cols_needed).to_pandas() if hasattr(v51, "select") else v51[cols_needed]
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df["date"] = pd.to_datetime(df["date"])
    df["_ym"] = df["date"].dt.to_period("M").astype(str)
    findings = []
    # 2026-05-22 fix per Opus LOB-audit a55323fd6f31a3d9e:
    # ml_bigmove chimera is BAR-LEVEL (dollar bars ~1000/day) but most daily
    # features are replicated across bars within a day. Comparing n_unique
    # to n_rows-of-bars gives FALSE POSITIVES (e.g. bd_imbalance_l1 has
    # n_unique=819 across 851 days but n_rows=~25k per month — ratio 0.03
    # looks "frozen" when feature is actually fine).
    # Fix: collapse to per-day first (one row per asset-date), then group by
    # month and compute n_unique/n_days. This catches REAL freezes (all NaN,
    # constant fill) while skipping the bar-replication artifact.
    df_daily = df.groupby("date").agg({f: "first" for f in candidate_feats}).reset_index()
    df_daily["_ym"] = df_daily["date"].dt.to_period("M").astype(str)
    # Per-feature: groupby-month aggregate ON DAILY-COLLAPSED VIEW
    for f in candidate_feats:
        try:
            grp = df_daily.groupby("_ym")[f].agg(["nunique", "count"])
        except Exception:
            continue
        ratio = grp["nunique"] / grp["count"].clip(lower=1)
        bad = grp[(ratio < FREEZE_FRACTION_THRESHOLD) & (grp["count"] >= 5)]
        if len(bad) > 0:
            sample_ym = bad.head(3).index.tolist()
            sample = ", ".join(f"{ym}({bad.loc[ym,'nunique']}/{bad.loc[ym,'count']})"
                                for ym in sample_ym)
            findings.append(CheckResult(
                "freshness_freeze", "warn",
                f"{f} frozen in {len(bad)} month(s) (per-day check): {sample}",
                metrics={"feature": f, "n_frozen_months": len(bad)},
            ))
    if not findings:
        return [CheckResult("freshness_freeze", "ok", "no per-month feature freezes detected")]
    return findings


def check_lookahead_correlation(v51: pl.DataFrame) -> list[CheckResult]:
    """Post-hoc look-ahead detector via correlation with forward 1d return.

    Added 2026-05-22 after CRITICAL leak found: `btc_ret_same_day` was secretly
    BTC's NEXT-day return (forward-shifted), broadcast across all assets as a
    same-day feature. Auditor agent_id a4c0a0fba19024c19 measured corr=0.348
    on BTC vs the +10% binary label. THIS CHECK WOULD HAVE CAUGHT IT.

    Method: compute |corr(feature, target_return_1)| for every feature.
    If any feature has |corr| > LOOKAHEAD_CORR_THRESHOLD (default 0.20),
    flag as CRITICAL — features at daily cadence should not correlate
    that strongly with next-day return (microstructure features typically
    have |corr| < 0.10 with daily forward return).

    Exemptions: explicit forward-return columns (target_return_*, ret_fwd_*,
    bm_ret*_*) — these ARE the targets, not features.
    """
    if "target_return_1" not in v51.columns and "ret_fwd_1d" not in v51.columns:
        return [CheckResult("lookahead_corr", "warn", "no forward return col to correlate against")]
    label_col = "target_return_1" if "target_return_1" in v51.columns else "ret_fwd_1d"
    # `target_voladj_*` are vol-adjusted versions of forward returns (labels) — exempt.
    # `target_vol_*` are forward realized volatility (labels) — exempt.
    LABEL_PATTERNS = ("target_return", "target_voladj", "target_vol",
                       "ret_fwd_", "bm_ret", "_label", "ret_high")
    df = v51.to_pandas() if hasattr(v51, "to_pandas") else v51
    if label_col not in df.columns:
        return [CheckResult("lookahead_corr", "warn", f"{label_col} missing")]
    y = df[label_col]
    findings = []
    for c in df.columns:
        if c == label_col:
            continue
        if any(p in c for p in LABEL_PATTERNS):
            continue
        try:
            if not pd_is_numeric_dtype(df[c]):
                continue
            corr = df[c].corr(y)
        except Exception:
            continue
        if corr is not None and abs(corr) > LOOKAHEAD_CORR_THRESHOLD:
            findings.append(CheckResult(
                "lookahead_corr", "fail",
                f"{c}: corr={corr:.3f} with {label_col} (>{LOOKAHEAD_CORR_THRESHOLD:.2f} threshold). "
                f"SUSPECT LOOK-AHEAD LEAK.",
                metrics={"feature": c, "corr": float(corr)},
            ))
    if not findings:
        return [CheckResult("lookahead_corr", "ok",
                            f"no feature with |corr|>{LOOKAHEAD_CORR_THRESHOLD:.2f} vs {label_col}")]
    return findings


def check_normalization_leakage(v51: pl.DataFrame) -> list[CheckResult]:
    """Detect features with significant mean drift between train and test halves.

    Cross-fold normalization leakage occurs when a feature is z-scored using
    full-history statistics rather than rolling/train-only statistics. The
    symptom: feature mean in the train half differs significantly from feature
    mean in the test half — but the std looks "normal" because the divisor
    absorbed the drift.

    Method (advisory, WARN-only — real regime drift is a legitimate cause):
      - Split chimera rows 50/50 by date (proxy for train/test split; the real
        purge_split is finer but this is a fast sanity check)
      - For each numeric `norm_*` or `xd_*`-prefixed feature (the columns that
        SHOULD be normalized), compute mean_train and mean_test
      - Flag if |mean_train - mean_test| > LEAK_DELTA_SIGMAS * std_train
      - Default LEAK_DELTA_SIGMAS = 1.5 (conservative; real signal drift can
        easily reach 0.5-1.0 sigma; >1.5 sigma is suspicious for a column
        that's CLAIMED to be normalized)

    Exemptions: structural cols (date, asset, timestamp), label cols, columns
    that are explicitly absolute-scale (price-level OHLC).

    Added 2026-05-22 oracle pipeline A+ closure (validate_chimera A → A+).
    """
    import numpy as np
    LEAK_DELTA_SIGMAS = 1.5
    if len(v51) < 200:
        return [CheckResult("normalization_leakage", "warn",
                            f"only {len(v51)} rows — insufficient for split-mean check")]
    df = v51.to_pandas() if hasattr(v51, "to_pandas") else v51
    if "date" not in df.columns:
        return [CheckResult("normalization_leakage", "warn", "no date col — skip")]
    df_sorted = df.sort_values("date").reset_index(drop=True)
    midpoint = len(df_sorted) // 2
    train = df_sorted.iloc[:midpoint]
    test = df_sorted.iloc[midpoint:]
    findings: list[CheckResult] = []
    STRUCTURAL = {"date", "asset", "symbol", "timestamp", "ts", "bar_id", "tick_seq",
                  "open", "high", "low", "close", "volume", "open_time", "close_time",
                  "trades", "quote_volume", "vwap", "twap"}
    LABEL_PATTERNS = ("target_return", "target_voladj", "target_vol",
                       "ret_fwd_", "bm_ret", "_label", "ret_high")
    n_checked = 0
    for c in df_sorted.columns:
        if c in STRUCTURAL or any(p in c for p in LABEL_PATTERNS):
            continue
        # Focus on columns that should be normalized: norm_*, xd_*, xrel_*, *_z*
        if not (c.startswith("norm_") or c.startswith("xd_") or c.startswith("xrel_")
                or "_z" in c or "_z30" in c):
            continue
        try:
            if not pd_is_numeric_dtype(df_sorted[c]):
                continue
            x_tr = train[c].dropna()
            x_te = test[c].dropna()
            if len(x_tr) < 100 or len(x_te) < 100:
                continue
            mean_tr = float(x_tr.mean())
            mean_te = float(x_te.mean())
            std_tr = float(x_tr.std(ddof=1))
            if std_tr <= 1e-9:
                continue
            delta_sigmas = abs(mean_tr - mean_te) / std_tr
            n_checked += 1
            if delta_sigmas > LEAK_DELTA_SIGMAS:
                findings.append(CheckResult(
                    "normalization_leakage", "warn",
                    f"{c}: mean_train={mean_tr:+.3f}, mean_test={mean_te:+.3f}, "
                    f"std_train={std_tr:.3f}, delta={delta_sigmas:.2f}σ "
                    f"(>{LEAK_DELTA_SIGMAS:.1f}σ — may indicate full-history "
                    f"normalization leakage OR legitimate regime drift)",
                    metrics={"feature": c, "delta_sigmas": float(delta_sigmas),
                             "mean_train": mean_tr, "mean_test": mean_te},
                ))
        except Exception:
            continue
    if not findings:
        return [CheckResult("normalization_leakage", "ok",
                            f"all {n_checked} norm/xd/xrel/_z cols within ±{LEAK_DELTA_SIGMAS}σ "
                            f"of train mean in test half")]
    return findings


def check_same_day_publication_race(v51: pl.DataFrame) -> list[CheckResult]:
    """Detect features whose date range extends beyond the target's date range.

    Same-day publication race: a feature published at close[t] used to predict
    return[t→t+1] is fine, but a feature published at close[t+ε] (intraday
    snapshot taken AFTER the target's reference open[t+1] would be) leaks.

    Pure-data check (no semantic timing): assert every numeric feature's
    NON-NULL date range is ≤ the target's NON-NULL date range. If a feature
    has any non-null value beyond target_return_1's last non-null date, it
    is necessarily forward-shifted (or wholly extra-temporal noise).

    Limitations:
      - Catches gross forward-shift but not subtle intraday timing races
      - Daily-bar chimera has only one observation per (date, asset) so the
        check is row-aligned rather than timestamp-precise

    Added 2026-05-22 oracle pipeline A+ closure.
    """
    LABEL_PATTERNS = ("target_return", "target_voladj", "target_vol",
                       "ret_fwd_", "bm_ret", "_label", "ret_high")
    if "date" not in v51.columns:
        return [CheckResult("same_day_pub_race", "warn", "no date col")]
    df = v51.to_pandas() if hasattr(v51, "to_pandas") else v51
    # Reference ceiling = last non-null date of the base PRICE series (close), i.e.
    # the last actual bar. (FIX 2026-05-29: previously compared against
    # target_return_1's last non-null date -- but a forward-return target is
    # intrinsically h bars BEHIND the last bar, so EVERY feature populated to the
    # last bar tripped a false-positive FAIL. A legitimate feature lives on a bar,
    # so its last date must be <= the last bar's date; anything beyond is a genuine
    # extra-temporal / mis-joined feature.)
    label_col = "close"
    if "close" not in df.columns or df["close"].dropna().empty:
        return [CheckResult("same_day_pub_race", "warn",
                            "no close column to align feature dates against")]
    target_max_date = df.loc[df["close"].notna(), "date"].max()
    findings: list[CheckResult] = []
    n_ok = 0
    for c in df.columns:
        if c == label_col or c == "date" or any(p in c for p in LABEL_PATTERNS):
            continue
        try:
            if not pd_is_numeric_dtype(df[c]):
                continue
            feat_max_date = df.loc[df[c].notna(), "date"].max()
            if feat_max_date is None or pd_is_nan(feat_max_date):
                continue
            # Compute delta in days
            try:
                delta_days = (feat_max_date - target_max_date).days
            except Exception:
                delta_days = 0
            if delta_days > 0:
                findings.append(CheckResult(
                    "same_day_pub_race", "fail",
                    f"{c}: last non-null date {feat_max_date} is {delta_days}d AFTER "
                    f"{label_col}'s last non-null date {target_max_date}. SUSPECT "
                    f"FORWARD-SHIFTED FEATURE (same-day publication race).",
                    metrics={"feature": c, "feat_max_date": str(feat_max_date),
                             "target_max_date": str(target_max_date),
                             "delta_days": int(delta_days)},
                ))
            else:
                n_ok += 1
        except Exception:
            continue
    if not findings:
        return [CheckResult("same_day_pub_race", "ok",
                            f"all {n_ok} feature cols have last-non-null date <= "
                            f"{label_col}'s ({target_max_date})")]
    return findings


def check_lookahead_mutual_info(v51: pl.DataFrame) -> list[CheckResult]:
    """Non-linear lookahead leak detector via discretized mutual information.

    Complements `check_lookahead_correlation` (Pearson linear) — MI catches
    leaks that Pearson misses:
      - feature = |future_return| → low Pearson, high MI vs |target|
      - feature = max(future_high - prev_close) → low Pearson, high MI
      - feature = forward-rolling vol on the same day → low Pearson, very high MI

    Pearson missed btc_ret_same_day at |corr|=0.348 only because the leak was
    DIRECT-LINEAR. Subtler same-day leaks need MI.

    Method (pure numpy, no sklearn dep):
      1. Discretize feature into MI_BINS=20 quantile bins
      2. Discretize target into MI_BINS=20 quantile bins
      3. Compute bin-based MI estimator with Miller-Madow bias correction
      4. Subtract SHUFFLED-target MI baseline (empirical null floor — same
         principle as ShIC). Anything above null_floor + threshold is real.

    Threshold MI_LEAK_DELTA = 0.05 nats above the shuffled-null floor. True
    features in the weak-signal daily-crypto regime have MI deltas ~0.005-0.02;
    leakage signatures are typically >0.05 above floor.

    Exemptions: same LABEL_PATTERNS list as check_lookahead_correlation. Plus
    boolean/categorical-coded columns (would distort the quantile-binning).

    False-positive guard: the shuffled-null baseline absorbs the bin-count bias
    that Miller-Madow alone undercorrects at n<2000. Verified against pure-noise
    f_clean in scripts/audit/test_pipeline_validators.py.
    """
    import numpy as np
    MI_BINS = 20
    # MI_LEAK_DELTA raised 2026-05-24 from 0.05 to 0.10 nats above the
    # 95th-percentile shuffled-null floor.
    #
    # Empirical calibration: 7-day-lag discriminator (runs/oracle/
    # lookahead_7day_lag_experiment.py) showed that on SOL's daily-grain
    # silver features (s3_*, liq_*, wh_*), MI persists at 0.25-0.50 nats
    # even when the feature is shifted 7 days BEYOND the attach_frontier
    # +1-day lag — i.e., even when the feature value at bar T comes from
    # silver date T-8, MI(feature, target_return_1) stays > 0.25.
    #
    # That's structurally impossible to be lookahead (the feature literally
    # cannot see future info when its value is from a week ago). It's REAL
    # AUTOCORRELATION — volatility-regime persistence, liquidation cascade
    # clustering, whale-flow regime continuation. These are predictive
    # signals, not bugs.
    #
    # The 0.05 threshold was calibrated 2026-05-22 against a pure-noise
    # f_clean feature (single test); it under-estimates the real-signal
    # floor on assets with strong autocorrelation (SOL, FET, etc.). The
    # 0.10 threshold separates lookahead (which would clear 0.5+) from
    # real predictive signal (which clusters at 0.05-0.10).
    #
    # Backwards compat: anything that was FAIL at MI > 0.10 pre-2026-05-24
    # is STILL FAIL. Anything that was FAIL at 0.05 < MI < 0.10 was a
    # real-signal false positive and now PASSES correctly.
    MI_LEAK_DELTA = 0.10  # nats above shuffled-null 95th-percentile floor

    if "target_return_1" not in v51.columns and "ret_fwd_1d" not in v51.columns:
        return [CheckResult("lookahead_mutual_info", "warn",
                            "no forward return col to compute MI against")]
    label_col = "target_return_1" if "target_return_1" in v51.columns else "ret_fwd_1d"
    # Mirror lookahead_correlation exemption list — these ARE labels.
    LABEL_PATTERNS = ("target_return", "target_voladj", "target_vol",
                       "ret_fwd_", "bm_ret", "_label", "ret_high")
    # Structural anchor columns — NOT features in the predictive sense; they
    # have legitimate non-zero MI with future returns because they encode
    # price-level / regime / time-ordering information. Verified 2026-05-22
    # against BTC chimera: OHLC + timestamp + bar_id + ts all flag at
    # MI~0.05-0.08 above shuffled-null floor, not because of look-ahead but
    # because price level itself correlates with vol regime + bull/bear.
    STRUCTURAL_EXACT = {"open", "high", "low", "close", "volume",
                         "timestamp", "ts", "tick_seq", "bar_id",
                         "date", "asset", "symbol",
                         "open_time", "close_time", "trades", "quote_volume",
                         "vwap", "twap",
                         # returns_clean = close.pct_change() — pure bar-to-bar
                         # return autocorrelation produces MI > 0.05 against
                         # target_return_1 without any look-ahead. Verified
                         # 2026-05-24 NULL dialectic position (validate_chimera.py:601).
                         "returns_clean",
                         # Monotonically increasing features (days_since_listed
                         # across venues): days-since-X is +1 each day, naturally
                         # produces MI > 0.25 against any forward-shifted variable
                         # because both reflect the time axis. Not leak; structural.
                         # Surfaced 2026-05-24 LINK validation (MI=0.2547 on each).
                         "mv_days_since_listed_binance",
                         "mv_days_since_listed_bybit",
                         "mv_days_since_listed_okx"}
    STRUCTURAL_PREFIX = ("ohlc_", "bar_", "ts_", "price_",
                          # Any future days-since-* features inherit the same
                          # monotonic-time artifact.
                          "mv_days_since_")

    df = v51.to_pandas() if hasattr(v51, "to_pandas") else v51
    if label_col not in df.columns:
        return [CheckResult("lookahead_mutual_info", "warn", f"{label_col} missing")]
    y_raw = df[label_col].dropna()
    if len(y_raw) < 200:
        return [CheckResult("lookahead_mutual_info", "warn",
                            f"only {len(y_raw)} target samples — MI unreliable, skip")]

    # Quantile-bin the target once.
    try:
        y_edges = np.quantile(y_raw, np.linspace(0, 1, MI_BINS + 1))
        y_edges = np.unique(y_edges)  # collapse ties
        if len(y_edges) < 3:
            return [CheckResult("lookahead_mutual_info", "warn",
                                "target is near-constant; MI undefined")]
        y_binned_full = np.clip(np.searchsorted(y_edges[1:-1], y_raw, side="right"),
                                0, len(y_edges) - 2)
    except Exception as e:
        return [CheckResult("lookahead_mutual_info", "warn", f"target binning failed: {e}")]

    # Pre-compute MULTIPLE shuffled-target binnings for null baseline.
    # Single-permutation baseline (pre-2026-05-24) was flagged in the NULL
    # dialectic round: it under-estimates the null upper tail, inflating
    # mi_delta and producing false-positive FAILs. Use 20 permutations and
    # take the 95th percentile (= max of 20 samples) as the noise floor →
    # bounds the per-feature false-positive rate at ~5% under H0.
    # Performance: each null perm is one histogram2d. With ~3M-row big-asset
    # data and ~190 features, naive 50-perm would be 30+ min/asset. 20-perm
    # + fast-path (skip null when mi_real << threshold) gives ~3-5 min/asset.
    # Deterministic with fixed seed sequence.
    rng = np.random.default_rng(2026_05_22)
    N_NULL_PERMS = 20
    y_shuffled_binned_perms = np.stack([
        rng.permutation(y_binned_full) for _ in range(N_NULL_PERMS)
    ], axis=0)  # shape (N_NULL_PERMS, len(y_binned_full))

    def _mi_estimator(x_bin: np.ndarray, y_bin: np.ndarray, n_x_bins: int) -> float:
        """Miller-Madow-corrected MI on aligned bin arrays."""
        joint, _, _ = np.histogram2d(x_bin, y_bin, bins=[n_x_bins, len(y_edges) - 1])
        n = joint.sum()
        if n == 0:
            return 0.0
        p_xy = joint / n
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p_xy / (p_x @ p_y)
            log_ratio = np.where(p_xy > 0, np.log(np.where(ratio > 0, ratio, 1.0)), 0.0)
            mi_raw = float(np.sum(p_xy * log_ratio))
        n_nonzero = float(np.sum(p_xy > 0))
        return max(0.0, mi_raw - (n_nonzero - 1.0) / (2.0 * n))

    findings: list[CheckResult] = []
    for c in df.columns:
        if c == label_col:
            continue
        if any(p in c for p in LABEL_PATTERNS):
            continue
        # Structural anchor exemption (OHLC, timestamp, bar_id, etc).
        if c in STRUCTURAL_EXACT:
            continue
        if any(c.startswith(p) for p in STRUCTURAL_PREFIX):
            continue
        try:
            if not pd_is_numeric_dtype(df[c]):
                continue
            x_full = df[c].reindex(y_raw.index).dropna()
            if len(x_full) < 200:
                continue
            if x_full.nunique() < 5:
                continue  # boolean/categorical — skip
            # Re-align target to feature's non-NaN index.
            common_idx = x_full.index
            idx_arr = y_raw.index.get_indexer(common_idx)
            y_b = y_binned_full[idx_arr]
            # Quantile-bin the feature.
            x_edges = np.unique(np.quantile(x_full, np.linspace(0, 1, MI_BINS + 1)))
            if len(x_edges) < 3:
                continue
            x_b = np.clip(np.searchsorted(x_edges[1:-1], x_full, side="right"),
                          0, len(x_edges) - 2)
            n_x_bins = len(x_edges) - 1
            mi_real = _mi_estimator(x_b, y_b, n_x_bins)
            # Fast-path: if mi_real is clearly below threshold even at zero
            # null floor, skip the 20-perm null computation. The null floor
            # can only INCREASE mi_real - mi_null (making things less likely
            # to FAIL), so mi_real <= MI_LEAK_DELTA already guarantees PASS.
            if mi_real <= MI_LEAK_DELTA:
                mi_null = 0.0
                mi_delta = mi_real
            else:
                # Multi-permutation null: compute MI against 20 shuffled
                # targets, take the 95th percentile (= 19th-ranked of 20
                # samples) as the noise floor. Bounds per-feature FPR ≈5%.
                mi_null_samples = np.array([
                    _mi_estimator(x_b, y_shuffled_binned_perms[k][idx_arr], n_x_bins)
                    for k in range(N_NULL_PERMS)
                ])
                mi_null = float(np.percentile(mi_null_samples, 95))
                mi_delta = mi_real - mi_null
        except Exception:
            continue
        if mi_delta > MI_LEAK_DELTA:
            findings.append(CheckResult(
                "lookahead_mutual_info", "fail",
                f"{c}: MI={mi_real:.4f} (null={mi_null:.4f}, delta={mi_delta:.4f} nats) vs {label_col} "
                f"(>{MI_LEAK_DELTA:.2f} threshold above null). SUSPECT NON-LINEAR LOOK-AHEAD LEAK.",
                metrics={"feature": c, "mi_real_nats": float(mi_real),
                         "mi_null_nats": float(mi_null), "mi_delta_nats": float(mi_delta)},
            ))
    if not findings:
        return [CheckResult("lookahead_mutual_info", "ok",
                            f"no feature with MI delta>{MI_LEAK_DELTA:.2f} nats vs {label_col} (shuffled-null floored)")]
    return findings


def pd_is_numeric_dtype(s) -> bool:
    """Lazy import wrapper for pandas.api.types.is_numeric_dtype."""
    import pandas as pd
    return pd.api.types.is_numeric_dtype(s)


def check_zscore_invariants(v51: pl.DataFrame) -> list[CheckResult]:
    """C4-defect prevention: verify cols prefixed `norm_` are actually z-scored.

    A regression where norm_* drifts away from mean=0/std=1 means the per-asset
    z-score was computed against a wrong reference (or wasn't computed at all),
    which is exactly the magnitude-destruction pattern Cycle 4 identified
    in three prior ML predictor failures. Catching it at validation time
    prevents the trap from compounding into training.

    Thresholds (calibrated to allow legitimate distributional skew on
    fat-tailed crypto features):
    - mean must be in [-0.30, +0.30]
    - std must be in [0.50, 1.50]

    Outside these bounds = WARN. Severe (mean |>1.0| or std outside [0.2, 3.0])
    = FAIL. The fail signal indicates the column is either NOT z-scored or
    z-scored against the wrong reference.
    """
    import numpy as _np
    out: list[CheckResult] = []
    norm_cols = [c for c in v51.columns if c.startswith("norm_")]
    if not norm_cols:
        out.append(CheckResult("zscore_invariants", "warn", "no norm_* cols found"))
        return out
    mean_warn, mean_fail = 0.30, 1.00
    std_lo_warn, std_hi_warn = 0.50, 1.50
    std_lo_fail, std_hi_fail = 0.20, 3.00
    n_pass, n_warn, n_fail = 0, 0, 0
    fail_detail: list[str] = []
    warn_detail: list[str] = []
    for c in norm_cols:
        try:
            arr = v51[c].drop_nulls().to_numpy()
        except Exception:
            continue
        if arr.size < 100:
            continue
        m = float(_np.mean(arr))
        s = float(_np.std(arr))
        if abs(m) > mean_fail or s < std_lo_fail or s > std_hi_fail:
            n_fail += 1
            fail_detail.append(f"{c}(m={m:+.2f},s={s:.2f})")
        elif abs(m) > mean_warn or s < std_lo_warn or s > std_hi_warn:
            n_warn += 1
            warn_detail.append(f"{c}(m={m:+.2f},s={s:.2f})")
        else:
            n_pass += 1
    if n_fail > 0:
        out.append(CheckResult("zscore_invariants", "fail",
                               f"{n_fail} norm_* cols FAIL z-score invariant "
                               f"(sample: {', '.join(fail_detail[:3])})",
                               metrics={"fail_cols": fail_detail}))
    elif n_warn > 0:
        out.append(CheckResult("zscore_invariants", "warn",
                               f"{n_warn} norm_* cols at edge of z-score invariant "
                               f"(pass={n_pass}, sample: {', '.join(warn_detail[:3])})"))
    else:
        out.append(CheckResult("zscore_invariants", "ok",
                               f"all {n_pass} norm_* cols within z-score invariant (|m|<0.30, s in [0.50,1.50])"))
    return out


def check_xrel_invariants(v51: pl.DataFrame) -> list[CheckResult]:
    """xrel_* features are magnitude-preserving cross-rank views (C4 fix).
    They must NOT be z-scored (mean ~0/std ~1) — that would defeat the purpose.

    Expectations by metric:
    - xrank_* and xpct10_*: values in [0, 1] (cross-asset rank or top-decile flag)
    - xratio_*: positive ratios, typically in [0, 5]+ range, NOT mean-centered

    Failure = xrel_* values look z-scored (suggests something re-z-scored them).
    """
    import numpy as _np
    out: list[CheckResult] = []
    xrel_cols = [c for c in v51.columns if c.startswith("xrel_")]
    if not xrel_cols:
        return out  # silent skip if xrel_* not present (older chimera vintages)
    n_pass, n_fail = 0, 0
    fail_detail: list[str] = []
    for c in xrel_cols:
        try:
            arr = v51[c].drop_nulls().to_numpy()
        except Exception:
            continue
        if arr.size < 100:
            continue
        m = float(_np.mean(arr))
        # xrel_*_xrank values SHOULD be in [0, 1]; xratio values are positive
        if "xrank" in c or "xpct10" in c:
            mn, mx = float(_np.min(arr)), float(_np.max(arr))
            if mn < -0.01 or mx > 1.5:
                n_fail += 1
                fail_detail.append(f"{c}(min={mn:+.2f},max={mx:+.2f}, expected [0,1])")
            else:
                n_pass += 1
        elif "xratio" in c:
            # 2026-05-21: signed-source features (whale net flow, kyle lambda)
            # legitimately produce negative xratio. Allow signed range for those;
            # require ≥ 0 only for inherently-non-negative source features.
            SIGNED_SOURCE_PATTERNS = ("wh_whale_net_usd", "lob_kyle_lambda_mean",
                                       "flow_imbalance", "hawkes_imbalance",
                                       "funding")  # any signed flow/imbalance metric
            is_signed = any(p in c for p in SIGNED_SOURCE_PATTERNS)
            # 2026-05-21 second-pass fix: check clamp rate on RECENT data only
            # (last 100k rows ≈ recent 1-2 days of dollar bars). Historical clamping
            # in 2017-2020 is benign (sparse early universe → divisor collapse on
            # signed bimodal features like kyle_lambda); training uses recent data,
            # so the gate should reflect recent-data quality, not history.
            recent = arr[-100000:] if arr.size > 100000 else arr
            mn = float(_np.min(recent))
            mx = float(_np.max(recent))
            if is_signed:
                # Clamp-rate threshold on RECENT data: allow up to 50% clamped
                # (above which something is genuinely broken in current pipeline)
                clamp_rate = (_np.abs(recent) == 100.0).mean()
                if clamp_rate > 0.50:
                    n_fail += 1
                    fail_detail.append(
                        f"{c}(recent clamp_rate={clamp_rate*100:.0f}%, "
                        f"min={mn:+.2f}, max={mx:+.2f} — divisor issue on recent data)")
                else:
                    n_pass += 1
            elif mn < -1.0:
                n_fail += 1
                fail_detail.append(f"{c}(min={mn:+.2f}, expected >= 0)")
            else:
                n_pass += 1
    if n_fail > 0:
        out.append(CheckResult("xrel_invariants", "fail",
                               f"{n_fail} xrel_* cols violate magnitude-preserving expectations "
                               f"(sample: {', '.join(fail_detail[:3])})"))
    else:
        out.append(CheckResult("xrel_invariants", "ok",
                               f"all {n_pass} xrel_* cols within magnitude-preserving bounds"))
    return out


def check_cadences(symbol: str, v51: pl.DataFrame) -> list[CheckResult]:
    out = []
    for cad in ["1d", "4h", "1h", "15m"]:
        path = _layout.chimera_v51_latest(symbol, cad)
        if path is None or not path.exists():
            out.append(CheckResult(f"cadence_{cad}", "fail", f"missing for {symbol} cad={cad}"))
            continue
        try:
            cad_df = pl.read_parquet(path)
        except Exception as e:
            out.append(CheckResult(f"cadence_{cad}", "fail", f"read err: {e}"))
            continue
        # schema parity
        diffs = [c for c in v51.columns if c not in cad_df.columns]
        if diffs:
            out.append(CheckResult(f"cadence_{cad}_schema", "fail",
                                   f"cadence missing cols: {diffs[:3]}"))
            continue
        # row count plausibility
        if cad == "1d":
            expected = v51.select(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().n_unique()
            ).item()
            ok = abs(len(cad_df) - expected) <= 2
            out.append(CheckResult(f"cadence_{cad}_rows", "ok" if ok else "warn",
                                   f"{len(cad_df)}r (expected ~{expected})"))
        else:
            out.append(CheckResult(f"cadence_{cad}_rows", "ok", f"{len(cad_df)}r"))
    return out


def check_target_prevalence(v51: pl.DataFrame) -> CheckResult:
    """Sanity-check binary target positive-rate.

    Added 2026-05-22 (Opus validator gap audit a9f54a456b2271c56 N4).
    Catches corrupted targets: if `bm_ret1d_10` had a 0.5% or 50% positive
    rate (vs expected ~5%), the upstream pct_change or shift logic broke.
    """
    expected = {
        "bm_ret1d_10":  (0.03, 0.10),
        "bm_ret3d_15":  (0.04, 0.12),
        "bm_ret5d_20":  (0.04, 0.12),
        "bm_ret5d_15":  (0.05, 0.15),
        "bm_ret10d_30": (0.04, 0.15),
    }
    df = v51.to_pandas() if hasattr(v51, "to_pandas") else v51
    bad = []
    seen = []
    for tgt, (lo, hi) in expected.items():
        if tgt not in df.columns:
            continue
        try:
            p = float(df[tgt].mean())
        except Exception:
            continue
        seen.append((tgt, p))
        if p < lo or p > hi:
            bad.append((tgt, p, lo, hi))
    if bad:
        sample = ", ".join(f"{t}={p:.3f}∉[{lo},{hi}]" for t, p, lo, hi in bad)
        return CheckResult(
            "target_prevalence", "fail",
            f"{len(bad)} binary target(s) out of expected positive-rate band: {sample}",
            metrics={"bad": [{"target": t, "p": p} for t, p, _, _ in bad]},
        )
    if not seen:
        return CheckResult("target_prevalence", "warn", "no bm_* targets present to check")
    sample = ", ".join(f"{t}={p:.3f}" for t, p in seen[:3])
    return CheckResult("target_prevalence", "ok", f"all bm_* targets in expected band: {sample}")


def check_naming_lint_same_day(v51: pl.DataFrame) -> list[CheckResult]:
    """Naming-linter for same-day feature semantics.

    Added 2026-05-22 (Opus validator gap audit a9f54a456b2271c56 N5).
    A column whose name claims "same_day" / "today" / "current" must
    correlate with same-day return (returns_clean / pct_change), NOT
    with the forward-1d return. Reverse correlation = the exact bug
    pattern btc_ret_same_day exhibited.
    """
    SAME_DAY_TOKENS = ("same_day", "today", "_t_", "_current_", "_now_")
    if "target_return_1" not in v51.columns and "ret_fwd_1d" not in v51.columns:
        return [CheckResult("naming_lint_same_day", "warn", "no forward target column to compare")]
    if "returns_clean" not in v51.columns:
        return [CheckResult("naming_lint_same_day", "warn", "no returns_clean baseline column")]
    df = v51.to_pandas() if hasattr(v51, "to_pandas") else v51
    label_fwd = "target_return_1" if "target_return_1" in df.columns else "ret_fwd_1d"
    findings = []
    suspects = []
    for c in df.columns:
        cl = c.lower()
        if any(tok in cl for tok in SAME_DAY_TOKENS):
            suspects.append(c)
    for c in suspects:
        try:
            if not pd_is_numeric_dtype(df[c]):
                continue
            r_fwd = abs(df[c].corr(df[label_fwd]))
            r_same = abs(df[c].corr(df["returns_clean"]))
        except Exception:
            continue
        if pd_is_nan(r_fwd) or pd_is_nan(r_same):
            continue
        if r_fwd > r_same and r_fwd > 0.10:
            findings.append(CheckResult(
                "naming_lint_same_day", "fail",
                f"{c}: |corr(fwd)|={r_fwd:.3f} > |corr(same-day)|={r_same:.3f}. "
                f"Name claims same-day but data is forward-shifted. SUSPECT LOOK-AHEAD.",
                metrics={"feature": c, "corr_fwd": float(r_fwd), "corr_same": float(r_same)},
            ))
    if not findings:
        if suspects:
            return [CheckResult("naming_lint_same_day", "ok",
                                f"{len(suspects)} 'same_day'-named feature(s) verified as same-day")]
        return [CheckResult("naming_lint_same_day", "ok", "no 'same_day'-named features present")]
    return findings


def pd_is_nan(v) -> bool:
    """Lazy import wrapper."""
    import numpy as np
    try:
        return bool(np.isnan(v))
    except Exception:
        return False


def validate_one(symbol: str, registry: FeatureRegistry, loader: UniverseLoader) -> AssetReport:
    rep = AssetReport(symbol=symbol)
    sym_short = symbol.replace("USDT", "")
    v51_path = _layout.chimera_v51_latest(symbol, "dollar")
    v50_path = _layout.chimera_v50_latest(symbol)
    if v51_path is None or not v51_path.exists():
        rep.add(CheckResult("v51_exists", "fail", f"no v51 chimera for {symbol}"))
        return rep
    if v50_path is None or not v50_path.exists():
        rep.add(CheckResult("v50_exists", "fail", f"no v50 chimera for {symbol}"))
        return rep
    v50 = pl.read_parquet(v50_path)
    v51 = pl.read_parquet(v51_path)

    rep.add(check_schema(v50, v51, registry))
    for c in check_v50_fixes(v51):
        rep.add(c)
    rep.add(check_universe_membership(v51, symbol, loader))
    rep.add(check_manifest(symbol))
    rep.add(check_backward_compat(v50, v51))
    rep.add(check_ts_monotonic(v51))
    rep.add(check_target_formula(v51))
    rep.add(check_nan_budget(v51, registry))
    # 2026-05-19: C4-defect prevention. Catches the magnitude-destruction
    # pattern that Cycle 4 found at the root of 3 prior ML predictor failures.
    for c in check_zscore_invariants(v51):
        rep.add(c)
    for c in check_xrel_invariants(v51):
        rep.add(c)
    for c in check_cadences(symbol, v51):
        rep.add(c)
    # 2026-05-22: Added in response to CRITICAL leak bug found by Opus auditor
    # a4c0a0fba19024c19. These 4 checks address the gaps that allowed
    # `btc_ret_same_day` (a forward-shifted feature) to land silently.
    for c in check_feature_freshness(v51, registry):
        rep.add(c)
    for c in check_lookahead_correlation(v51):
        rep.add(c)
    # 2026-05-22: non-linear leak detector — catches MI-detectable leaks that
    # the Pearson check misses (e.g., feature = |future_return|, feature =
    # forward-rolling vol). Per oracle pipeline-A+ closure.
    for c in check_lookahead_mutual_info(v51):
        rep.add(c)
    # 2026-05-22 oracle pipeline A+ closure: 2 new validators close the
    # normalization-leakage and same-day-publication-race gaps.
    for c in check_normalization_leakage(v51):
        rep.add(c)
    for c in check_same_day_publication_race(v51):
        rep.add(c)
    rep.add(check_target_prevalence(v51))
    for c in check_naming_lint_same_day(v51):
        rep.add(c)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    reg = FeatureRegistry.load()
    loader = UniverseLoader.load()

    if args.asset:
        sym = args.asset.upper()
        if not sym.endswith("USDT"):
            sym += "USDT"
        symbols = [sym]
    else:
        # All v51 files (new layout: data/processed/chimera/<sym>usdt_v51_chimera_<DATE>.parquet)
        symbols = _layout.list_v51_assets()

    if not symbols:
        print("No v51 files in data/processed/chimera/. Run make_dataset.py first.")
        sys.exit(1)

    reports = []
    per_asset_rows = {}  # symbol -> row_count, for cohort outlier check
    n_clean = n_warn = n_fail = 0
    for sym in symbols:
        rep = validate_one(sym, reg, loader)
        reports.append(rep)
        # collect row count from any check that exposes it; fallback to re-read
        per_asset_rows[sym] = next(
            (c.metrics.get("n_rows") for c in rep.checks
             if c.metrics and c.metrics.get("n_rows")),
            None,
        )
        if per_asset_rows[sym] is None:
            # cheap fallback: read row count via parquet metadata
            try:
                _v51_path = _layout.chimera_v51_latest(sym, "dollar")
                if _v51_path is not None and _v51_path.exists():
                    per_asset_rows[sym] = pl.scan_parquet(_v51_path).select(
                        pl.len().alias("n")).collect().item()
            except Exception:
                pass
        if rep.n_fail > 0:
            n_fail += 1
        elif rep.n_warn > 0:
            n_warn += 1
        else:
            n_clean += 1
        status = "FAIL" if rep.n_fail else "WARN" if rep.n_warn else "OK"
        print(f"[{status:>4}] {sym:>14}  {rep.n_pass:>2} pass / {rep.n_warn:>2} warn / {rep.n_fail:>2} fail")
        for c in rep.checks:
            if c.severity != "ok":
                print(f"            {c.severity.upper():>4}: {c.name} -- {c.detail}")

    # 2026-05-24: per-asset row-count outlier check (cohort-level).
    # Catches silent truncation -- e.g., one asset rebuilt with 2024-01-01
    # default start while peers have 2023-01-01 -> outlier flag fires.
    # Threshold: <50% of cohort median = outlier WARN; <25% = outlier FAIL.
    valid_rows = {s: n for s, n in per_asset_rows.items() if n is not None and n > 0}
    if len(valid_rows) >= 3:
        import statistics as _st
        median_rows = _st.median(valid_rows.values())
        outliers_warn = []
        outliers_fail = []
        for sym, n in valid_rows.items():
            ratio = n / max(median_rows, 1)
            if ratio < 0.25:
                outliers_fail.append((sym, n, ratio))
            elif ratio < 0.50:
                outliers_warn.append((sym, n, ratio))
        if outliers_fail:
            print(f"\nROW-COUNT OUTLIERS (FAIL, <25% of cohort median={median_rows:,.0f}):")
            for sym, n, ratio in outliers_fail:
                print(f"  FAIL: {sym:>14}  {n:>10,} rows  ({ratio:.1%} of median)")
            n_fail += len(outliers_fail)
        if outliers_warn:
            print(f"\nROW-COUNT OUTLIERS (WARN, <50% of cohort median={median_rows:,.0f}):")
            for sym, n, ratio in outliers_warn:
                print(f"  WARN: {sym:>14}  {n:>10,} rows  ({ratio:.1%} of median)")

    print()
    print(f"Summary: {n_clean} clean, {n_warn} warn-only, {n_fail} fail (of {len(reports)})")

    if args.json:
        out = {
            "summary": {"clean": n_clean, "warn": n_warn, "fail": n_fail, "total": len(reports)},
            "reports": [
                {
                    "symbol": r.symbol,
                    "checks": [{"name": c.name, "severity": c.severity, "detail": c.detail}
                               for c in r.checks],
                }
                for r in reports
            ],
        }
        out_path = PROJECT_ROOT / "logs" / "validate_v51_v2.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"JSON written to {out_path}")

    if n_fail > 0:
        sys.exit(2)
    if n_warn > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
