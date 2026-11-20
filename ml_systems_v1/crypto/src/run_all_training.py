#!/usr/bin/env python
"""
Master Training Runner -- Feature-Driven
==========================================

Runs ALL models that support a given feature count, with all their variants.

Usage:
    python src/run_all_training.py --features 13    # All models at f13 (base only)
    python src/run_all_training.py --features 25    # All models at f25
    python src/run_all_training.py --features 34    # All models at f34
    python src/run_all_training.py --features 25 --only-base  # Skip adapter/snapshot/NCL
    python src/run_all_training.py --features 25 --model v1_0  # Single model
    python src/run_all_training.py --dry-run --features 25     # Preflight only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry: every model and its variants
# ─────────────────────────────────────────────────────────────────────────────

# Archived models — runner skips these entirely (use --include-archived to force)
# V9: GRU+MoE+RSSM memorizes across retrains. Best ShIC=0.0070 (need 0.015)
#     after 90 epochs. Logs in logs/v9/_abandoned_shic_fail_20260415/. See that
#     directory's README.md for full verdict.
ARCHIVED_MODELS = {"v9"}

# Models BLOCKED on missing infrastructure. These have a runtime gate that
# refuses to train in degraded mode (per @browser no-silent-failures).
# Listing them here is a safety net so the runner doesn't even invoke them.
# v12: cross-asset attention is dead code in single-asset path; needs a
#      MultiAssetDataset + forward_multi_asset wiring (1-2d). Set
#      V12_HEADLINE_MODE=1 once dataloader is built. (added 2026-05-07)
# V22/V23/V24 trainers wired 2026-05-07 (round 4) -- NO LONGER blocked.
BLOCKED_MODELS = {
    "v12": "requires MultiAssetDataset + V12_HEADLINE_MODE=1 (see docs/WM_HEADLINE_UPGRADE_PLAN §14)",
}


# (model_id, label, script, extra_args, supported_features)
MODELS = [
    # V1 family (have adapter/snapshot/NCL variants)
    ("v1_0", "V1.0 (Transformer+RSSM baseline)",
     "src/wm/v1/v1_0_training/train_world_model.py", [], [13, 25, 29, 34, 127, 133, 154, 161]),

    ("v1_1", "V1.1 (XD anti-memorization)",
     "src/wm/v1/v1_1_training/train_world_model.py", [], [13, 18, 25, 29, 34, 37, 41, 127, 133, 154, 161]),

    ("v1_4", "V1.4 (FeatureAttention)",
     "src/wm/v1/v1_4_training/train_world_model.py", [], [13, 18, 25, 29, 34, 127, 133, 154, 161]),

    ("v1_6", "V1.6 (KL+Gumbel+ATME+Dream)",
     "src/wm/v1/v1_6_training/train_world_model.py", [], [13, 18, 25, 29, 34, 127, 133, 154, 161]),

    # V3/V6/V9 FULL models (RSSM/JEPA, same anti-memorization mechanism as V1.6)
    # The "_clean" variants stripped RSSM and memorized catastrophically
    # (V3-clean f34 2026-04-08: IC1=0.27, ShIC=0.0002, ratio=0.0007). The FULL
    # models have the proven V1.6-class defenses: categorical RSSM latent,
    # Gumbel straight-through, free_nats KL floor, ATME that zeros h_seq AND
    # forces obs-only posterior readout. Runner uses them without --clean.
    # V3/V6/V9 live in the SOTA 41-feature taxonomy where f25 is NOT a defined
    # slice (their slices are 13/18/30/34/37/41).
    ("v3", "V3 (WaveNet-GRU + RSSM)",
     "src/wm/v3/v3_training/train_world_model.py", [], [13, 29, 34, 41, 127, 133, 154, 161]),

    ("v6", "V6 (JEPA + Adversarial)",
     "src/wm/v6/v6_training/train_world_model.py", [], [13, 29, 34, 41, 127, 133, 154, 161]),

    ("v9", "V9 (MoE + RSSM)",
     "src/wm/v9/v9_training/train_world_model.py", [], [13, 29, 34, 127, 133, 154, 161]),

    # V4 (Mamba-3 + RSSM) -- same SOTA taxonomy
    ("v4", "V4 (Mamba-3 SSM + RSSM)",
     "src/wm/v4/v4_training/train_world_model.py", [], [13, 29, 34, 41, 127, 133, 154, 161]),

    # V8 (kept, low priority) -- same SOTA taxonomy, no f25 slice defined
    ("v8", "V8 (Neural ODE, RK4)",
     "src/wm/v8/v8_training/train_world_model.py", [], [13, 18, 29, 34, 41, 127, 133, 154, 161]),

    # New architectures (base only)
    ("v11", "V11 (WaveNet+MoE+Discriminator)",
     "src/wm/v11/v11_training/train_world_model.py", [], [13, 25, 29, 34, 127, 133, 154, 161]),

    ("v12", "V12 (Cross-Asset Attention)",
     "src/wm/v12/v12_training/train_world_model.py", [], [13, 25, 29, 34, 127, 133, 154, 161]),

    ("v13", "V13 (TFT Variable Selection)",
     "src/wm/v13/v13_training/train_world_model.py", [], [13, 25, 29, 34, 41, 127, 133, 154, 161]),

    ("v14", "V14 (Diffusion Return Distribution)",
     "src/wm/v14/v14_training/train_world_model.py", [], [13, 25, 29, 34, 127, 133, 154, 161]),

    # V15+ -- 2026-04-26 model layer refresh (Option B). Architectures + smoke
    # tests committed; trainer scripts pending Job 2 (full 53-asset v51 build).
    ("v15", "V15 (PatchTST encoder, drop-in)",
     "src/wm/v15/patchtst_encoder.py", [], [121]),  # encoder-only stub; no trainer yet

    # V16/V17 RECLASSIFIED 2026-06-11: they are A1-agent backbones (DreamerV3 /
    # TD-MPC2 planners), NOT forecasters. Moved out of the forecaster registry to
    # src/agents/a1_wm_consuming/backbones/{v16_dreamerv3,v17_tdmpc2}/. See
    # docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md SS1.4 + the agents.json registry.

    # V18 (Chronos-T5 finetune) REMOVED 2026-05-07: directory does not exist;
    # registry pointer was broken. Per WM_SCORESHEET_MERGED §V18 KILL? — foundation-
    # model finetune misaligned with 8GB VRAM constraint (60M+ params at finetune
    # time). If foundation-model approach is pursued, build it as a fresh V-version
    # under a different number with explicit LoRA + quantization plan.

    # V19 ARCHIVED 2026-05-09 to backups/BKP_20260509_INCOMPLETE_VERSIONS/wm/v19/
    # (was settings.py + smoke_test.py only; no world_model.py / train_world_model.py).
    # Restore from backup if completed in future.

    # V22-V24: SOTA gaps from WM_SOTA_IRON_CLAD_AUDIT_2026_05_07.
    # Trainers wired 2026-05-07 (round 4). Each is a full V-version with
    # settings.py + world_model.py (get_loss V1.x-compatible) + train_world_model.py.
    # V22: --resume REMOVED 2026-05-21 oracle validation. The pre-2026-05-13
    # ckpts were trained under the broken per-bar supervision (IC=0.21 /
    # ShIC=0.0 memorization). Post-SOTA-sweep retrain must be FRESH so the
    # last-bar supervision + stride=1 sliding-window training reaches its
    # own equilibrium. To re-enable resume after a SOTA-sweep ckpt exists,
    # add "--resume" back to extra args.
    ("v22", "V22 (iTransformer + last-bar supervision — Timer-XL pattern)",
     "src/wm/v22/v22_training/train_world_model.py", [], [13, 18, 25, 29, 34, 37, 41, 127, 133, 154, 161]),

    ("v23", "V23 (xLSTM — extended LSTM with matrix memory)",
     "src/wm/v23/v23_training/train_world_model.py", [], [13, 18, 25, 29, 34, 37, 41, 127, 133, 154, 161]),

    ("v24", "V24 (TimesNet — FFT multi-period 2D inception)",
     "src/wm/v24/v24_training/train_world_model.py", [], [13, 18, 25, 29, 34, 37, 41, 127, 133, 154, 161]),

    # V25 Frontier — round-6 first-principles synthesis (2026-05-07).
    # Combines hard-coded crypto period embeddings + regime-conditioned
    # cross-feature attention + rate-budget VIB + tail-adaptive Huber +
    # adversarial regime training. NO single paper combines these.
    # See memory/feedback_unconstrained_default_synthesis.md for protocol.
    ("v25", "V25 (Frontier — first-principles crypto synthesis)",
     "src/wm/v25/v25_training/train_world_model.py", [], [13, 18, 25, 29, 34, 37, 41, 127, 133, 154, 161]),
]

# V15 is PLANNED but no trainer yet — encoder-only stub. V22/V23/V24 trainers
# wired 2026-05-07 (round 4) so they are NO LONGER in ARCHIVED_PENDING.
# V16/V17 RECLASSIFIED 2026-06-11 to A1 backbones (src/agents/...) — they are
# no longer forecasters, so they are removed from this registry entirely.
ARCHIVED_PENDING = {"v15"}  # v19/v20 physically archived 2026-05-09

# V1 variant scripts (adapter, snapshot, NCL) -- only for V1 family
V1_VARIANTS = {
    "v1_0": [
        # V1.0 has no adapter/snapshot/NCL variants
    ],
    "v1_1": [
        ("adapter", "src/wm/v1/v1_1_training/train_adapter.py"),
        ("snapshot", "src/wm/v1/v1_1_training/train_snapshot.py"),
        ("ncl", "src/wm/v1/v1_1_training/train_diversity.py"),
    ],
    "v1_4": [
        ("adapter", "src/wm/v1/v1_4_training/train_adapter.py"),
        ("snapshot", "src/wm/v1/v1_4_training/train_snapshot.py"),
        ("ncl", "src/wm/v1/v1_4_training/train_diversity.py"),
    ],
    "v1_6": [
        ("adapter", "src/wm/v1/v1_6_training/train_adapter.py"),
        ("snapshot", "src/wm/v1/v1_6_training/train_snapshot.py"),
    ],
}


def preflight(script_path: str, n_features: int | None = None) -> str:
    """Strengthened 2026-04-26 from py_compile-only to multi-check.

    Checks:
      1. Script file exists
      2. py_compile syntax-clean
      3. (if n_features given) settings.py for the same training dir resolves
         get_feature_config(n_features) without raising
      4. Required data input is present (chimera v50 or v51 per the script).

    Returns empty string on PASS, error message on FAIL.
    """
    path = PROJECT_ROOT / script_path
    if not path.exists():
        return f"Not found: {script_path}"
    # 2. compile
    try:
        import py_compile
        py_compile.compile(str(path), doraise=True)
    except Exception as e:
        return f"compile err: {str(e)[:80]}"
    # 3. feature config resolves
    if n_features is not None:
        # derive settings module from script path: src/<maj>/<sub>_training/train_world_model.py
        # -> src.<maj>.<sub>_training.settings
        parts = Path(script_path).parts
        if len(parts) >= 3 and parts[0] == "src":
            try:
                import importlib, sys as _sys
                _sys.path.insert(0, str(PROJECT_ROOT / "src"))
                mod_path = ".".join(parts[1:-1]) + ".settings"
                mod = importlib.import_module(mod_path)
                if hasattr(mod, "get_feature_config"):
                    mod.get_feature_config(n_features)
                # else: settings may not have feature config (V0 baselines etc.)
            except ValueError as e:
                return f"feature config: {str(e)[:80]}"
            except Exception as e:
                # Settings imports can fail for unrelated reasons; warn but don't block
                pass
    # 4. data input check (best-effort: chimera v50 + v51 presence for u10)
    # Layout: data/processed/chimera_legacy/dollar/<sym>_v50_chimera_<date>.parquet
    # (V1.x settings.DATA_DIR points to chimera_legacy/dollar/ specifically.)
    legacy_dir = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    v50_glob = list(legacy_dir.glob("*_v50_chimera*.parquet")) if legacy_dir.exists() else []
    if not v50_glob:
        # Fall back to the parent in case some legacy script writes there
        parent_dir = legacy_dir.parent
        v50_glob = list(parent_dir.glob("*_v50_chimera*.parquet")) if parent_dir.exists() else []
        if not v50_glob:
            return f"no v50 chimera files in {legacy_dir}"
    return ""


def _log_dir(script_path: str):
    """
    Derive log directory from a script path.
    Post-2026-04-29: scripts live under src/wm/{maj}/{sub}_training/...
    -> logs/{maj}/{sub}/   (e.g. src/wm/v1/v1_0_training/x.py -> logs/v1/v1_0/)
    Legacy form (src/{maj}/{sub}_training/) still supported for any
    pre-migration callers.
    """
    parts = Path(script_path).parts
    if len(parts) < 4 or parts[0] != "src":
        return None
    if parts[1] == "wm" and len(parts) >= 5:
        major = parts[2]
        sub = parts[3].replace("_training", "")
    else:
        major = parts[1]
        sub = parts[2].replace("_training", "")
    return PROJECT_ROOT / "logs" / major / sub


# Universal completion banner written by anti-fragile training framework
# at the end of every successful run. Confirmed present in V1.0 through V1.6
# and expected for V3/V6/V8/V9/V11-V14 (same anti_fragile.py framework).
_DONE_MARKER = "TRAINING COMPLETE"


def is_complete(model_id: str, script_path: str, n_features: int) -> bool:
    """
    A model is complete if its log directory contains any *train*.log file that:
      1. mentions the target feature count ("Features:     N" or f{N} in name)
      2. contains the "TRAINING COMPLETE" banner (universal done marker)

    Scans all matching train logs (not just the most recent) to tolerate
    interleaved feature-count runs in the same directory.
    """
    log_dir = _log_dir(script_path)
    if log_dir is None or not log_dir.exists():
        return False

    # Collect candidate logs: feature-aware name OR feature-blind name
    candidates = sorted(log_dir.glob("*train*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return False

    feat_marker = "f%d" % n_features          # e.g. "f34" (appears in filename or --features)
    feat_line = "Features:     %d" % n_features  # log header printed by training banner

    for log_path in candidates:
        # Fast filter: filename must either contain f{N} or be a feature-blind name
        name = log_path.name
        # If filename encodes a different feature count, skip
        # (e.g. v1_1_f25_train_*.log for feature-aware names)
        import re
        fn_feat = re.search(r"_f(\d+)_train", name)
        if fn_feat and int(fn_feat.group(1)) != n_features:
            continue

        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Must contain the completion banner
        if _DONE_MARKER not in content:
            continue

        # Must also confirm the feature count (either from filename or log body)
        if fn_feat is not None and int(fn_feat.group(1)) == n_features:
            return True
        if feat_line in content:
            return True
        if ("--features %d" % n_features) in content or ("features=%d" % n_features) in content:
            return True

    return False


def run_one(label: str, script: str, args: list, dry_run: bool) -> bool:
    path = PROJECT_ROOT / script
    cmd = [sys.executable, str(path)] + args

    print("\n" + "=" * 80)
    print("  %s" % label)
    print("  %s" % " ".join(cmd))
    print("  %s" % datetime.now().strftime("%H:%M:%S"))
    print("=" * 80)

    if dry_run:
        print("  [DRY RUN]")
        return True

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=86400)
        h = (time.time() - start) / 3600
        ok = result.returncode == 0
        print("  [%s] %.1fh" % ("OK" if ok else "FAIL:%d" % result.returncode, h))
        return ok
    except subprocess.TimeoutExpired:
        print("  [TIMEOUT]")
        return False
    except Exception as e:
        print("  [ERROR] %s" % e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Feature-Driven Training Runner")
    parser.add_argument("--features", type=int, required=True,
                        choices=[13, 18, 25, 29, 34, 37, 41, 51, 121, 127, 133, 154, 161],
                        help="Feature count -- runs ALL models supporting this count. "
                             "f41 added 2026-05-21 (feature-mining shows f41 captures 80%% of top-signal vs f29's 60%%). "
                             "f51/f121 supported by V22/V25/V23/V24 only. "
                             "f127/133/154/161 added 2026-05-25 -- 28-version SUPPORTED_FEATURE_COUNTS "
                             "extension. f127 = f121 + rv_jumps_6; f133 = f127 + te_panel_6; "
                             "f154 = f133 + T2_21 (sparse-by-date lob_bgf); f161 = f154 + XEX_NEW_7.")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model ID (e.g., v1_0, v3_clean, v11)")
    parser.add_argument("--only-base", action="store_true",
                        help="Skip adapter/snapshot/NCL variants")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-train models even if best_ema checkpoint already exists")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Isolate the base model's checkpoints/logs under a tag (forwarded to the "
                             "base trainer). e.g. --run-tag vsn_fr writes v1_1_f41_vsn_fr_wm_*.pt -- lets "
                             "A/B variants (baseline vs V1_VSN=1/V1_FORWARD_REGIME=1) run via this gated "
                             "wrapper without clobbering each other's checkpoints. Base-only (variant "
                             "scripts are not tag-forwarded; pair with --only-base).")
    parser.add_argument("--frontier", action="store_true",
                        help="Pass --frontier flag to models (use frontier components)")
    parser.add_argument("--include-archived", action="store_true",
                        help="Include models in ARCHIVED_MODELS (by default they are skipped)")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated model IDs to exclude (e.g., --exclude v8,v4)")
    parser.add_argument("--skip-gate", action="store_true",
                        help="Skip src/pipeline/pre_train_gate.py (NOT RECOMMENDED). "
                             "The gate runs 5 validators (data_health + chimera_v51 schema + "
                             "xd_consistency + e2e + split). Skipping = train on possibly-broken data.")
    parser.add_argument("--gate-asset", type=str, default="BTC",
                        help="Asset to validate in pre_train_gate (default BTC). "
                             "Gate is fast; running per-asset is unnecessary. "
                             "Use --gate-universe to gate across all u10 assets.")
    parser.add_argument("--gate-universe", type=str, default=None,
                        choices=["u10", "u50", "u100"],
                        help="Run gate across every asset in the universe; takes "
                             "precedence over --gate-asset when set.")
    parser.add_argument("--gate-layer", type=str, default="legacy",
                        choices=["legacy", "v51", "both"],
                        help="Which chimera layer the gate validates. Default 'legacy' "
                             "= v50 chimera (V1-V14 + V0 use this). 'v51' = new v51 "
                             "(V19 / frontier). 'both' = both. Set to 'both' once v51 "
                             "rebuild lands; until then, V1-V14 retrains use 'legacy' "
                             "to skip the broken-v51 false-fail.")
    parser.add_argument("--auto-refresh", action="store_true",
                        help="Before gate+preflight, run `src/pipeline/refresh.py --target "
                             "chimera_v51 --scope u50` to rebuild any stale upstream data. "
                             "Off by default to keep training cycles fast.")
    args = parser.parse_args()

    n_feat = args.features

    # Filter models by feature support
    eligible = [(mid, label, script, extra, feats)
                for mid, label, script, extra, feats in MODELS
                if n_feat in feats]

    # Filter archived (unless explicitly overridden)
    if not getattr(args, "include_archived", False):
        archived_hit = [e for e in eligible if e[0] in ARCHIVED_MODELS]
        if archived_hit:
            print("  [ARCHIVED — skipped] %s" % ", ".join(e[0] for e in archived_hit))
            print("  (to include: pass --include-archived)")
        eligible = [e for e in eligible if e[0] not in ARCHIVED_MODELS]

    # Filter BLOCKED (infrastructure-pending; cannot run in degraded mode)
    blocked_hit = [e for e in eligible if e[0] in BLOCKED_MODELS]
    if blocked_hit:
        print("  [BLOCKED — infrastructure pending]")
        for e in blocked_hit:
            print("    %s: %s" % (e[0], BLOCKED_MODELS[e[0]]))
    eligible = [e for e in eligible if e[0] not in BLOCKED_MODELS]

    # User-specified exclusions (additive with ARCHIVED_MODELS)
    if args.exclude:
        exclude_set = {x.strip().lower() for x in args.exclude.split(",") if x.strip()}
        excluded_hit = [e for e in eligible if e[0].lower() in exclude_set]
        if excluded_hit:
            print("  [EXCLUDED via --exclude] %s" % ", ".join(e[0] for e in excluded_hit))
        eligible = [e for e in eligible if e[0].lower() not in exclude_set]

    if args.model:
        eligible = [e for e in eligible if args.model.lower() in e[0].lower()]

    if not eligible:
        print("No models support f%d%s" % (n_feat,
              " matching '%s'" % args.model if args.model else ""))
        sys.exit(1)

    # Build run plan: base models + V1 variants
    plan = []
    skipped = []
    for mid, label, script, extra, _ in eligible:
        # Base model -- skip if already complete (unless --force)
        if not args.force and is_complete(mid, script, n_feat):
            skipped.append("%s base f%d" % (label, n_feat))
        else:
            base_args = ["--features", str(n_feat)] + extra
            if args.frontier:
                base_args.append("--frontier")
            if args.run_tag:
                base_args += ["--run-tag", args.run_tag]
            plan.append(("%s base f%d" % (label, n_feat), script, base_args))

        # V1 variants (if not --only-base) -- variants are not gated by base completion
        if not args.only_base and mid in V1_VARIANTS:
            for var_name, var_script in V1_VARIANTS[mid]:
                var_path = PROJECT_ROOT / var_script
                if var_path.exists():
                    var_args = ["--features", str(n_feat)] if "--features" not in str(var_script) else []
                    plan.append(("%s %s f%d" % (label, var_name, n_feat),
                                 var_script, var_args))

    print("=" * 80)
    print("  TRAINING: f%d across %d models (%d runs total)" % (
        n_feat, len(eligible), len(plan)))
    print("  %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    for i, (label, _, _) in enumerate(plan):
        print("  %2d. %s" % (i + 1, label))
    if skipped:
        print("\n  SKIPPED (already complete -- use --force to re-train):")
        for s in skipped:
            print("    - %s" % s)

    # Optional: refresh upstream data via DAG runner
    if args.auto_refresh:
        print("\n--- AUTO-REFRESH (src/pipeline/refresh.py --target chimera_v51 --scope u50) ---")
        rc = subprocess.call(
            [sys.executable, str(PROJECT_ROOT / "src" / "pipeline" / "refresh.py"),
             "--target", "chimera_v51", "--scope", "u50"],
            cwd=str(PROJECT_ROOT),
        )
        if rc != 0:
            print("  refresh.py exited %d -- aborting (use --skip-gate at your peril)" % rc)
            sys.exit(rc)

    # Pre-train gate (5 validators) -- runs before preflight so we fail fast
    # on broken data BEFORE compiling each model's import graph.
    if not args.skip_gate:
        gate_cmd = [sys.executable,
                    str(PROJECT_ROOT / "src" / "pipeline" / "pre_train_gate.py"),
                    "--layer", args.gate_layer]
        if args.gate_universe:
            gate_cmd.extend(["--universe", args.gate_universe])
            scope_msg = f"universe={args.gate_universe}"
        else:
            gate_cmd.extend(["--asset", args.gate_asset])
            scope_msg = f"asset={args.gate_asset}"
        print("\n--- PRE-TRAIN GATE (%s, layer=%s) ---" % (scope_msg, args.gate_layer))
        rc = subprocess.call(gate_cmd, cwd=str(PROJECT_ROOT))
        if rc == 2:
            print("  GATE FAILED (rc=2: hard fail). Aborting. Run with --skip-gate to override.")
            sys.exit(2)
        elif rc == 1:
            print("  GATE WARN (rc=1: warnings, training may proceed but data is suspect).")
        else:
            print("  GATE PASS")

    # Preflight (strengthened 2026-04-26)
    if not args.skip_preflight:
        print("\n--- PREFLIGHT (compile + feature_config + data presence) ---")
        ok = True
        for label, script, _ in plan:
            err = preflight(script, n_features=args.features)
            if err:
                print("  FAIL: %s -- %s" % (label, err))
                ok = False
            else:
                print("  OK: %s" % label)
        if not ok:
            sys.exit(1)

    # Execute
    results = []
    t0 = time.time()
    for label, script, run_args in plan:
        success = run_one(label, script, run_args, args.dry_run)
        results.append((label, success))

    # Summary
    total_h = (time.time() - t0) / 3600
    print("\n" + "=" * 80)
    print("  SUMMARY: f%d (%.1fh total)" % (n_feat, total_h))
    print("=" * 80)
    for label, ok in results:
        print("  [%s] %s" % ("OK" if ok else "FAIL", label))
    n_ok = sum(1 for _, ok in results if ok)
    print("\n  %d/%d passed" % (n_ok, len(results)))

    # Coverage report -- which active versions actually got trained at this f-count
    print("\n" + "=" * 80)
    print("  COVERAGE: who supports f%d?" % n_feat)
    print("=" * 80)
    for mid, label, _, _, feats in MODELS:
        if mid in ARCHIVED_MODELS and not getattr(args, "include_archived", False):
            marker = "archived"
        elif n_feat not in feats:
            marker = "n/a"
        elif any(label in r[0] for r in results if r[1]):
            marker = "TRAINED"
        elif any(label in r[0] for r in results if not r[1]):
            marker = "FAILED"
        elif any(label in s for s in skipped):
            marker = "fresh"
        else:
            marker = "missing"
        print("  %-9s %s" % (marker, label))


if __name__ == "__main__":
    main()
