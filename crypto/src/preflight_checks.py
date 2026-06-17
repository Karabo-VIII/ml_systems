"""
V4 Crypto System — Pre-Flight Check System

Lightweight but powerful checks that catch known failure modes BEFORE
training or pipeline execution. Each check is fast, deterministic, and
prints a clear PASS/FAIL result.

Usage:
    python src/preflight_checks.py --all          # Run everything
    python src/preflight_checks.py --pipeline     # Pipeline checks
    python src/preflight_checks.py --architecture # Architecture checks
    python src/preflight_checks.py --training     # Training config checks
    python src/preflight_checks.py --data         # Data integrity checks
    python src/preflight_checks.py --version 1    # Check specific version

Exit codes:
    0 = All checks passed
    1 = Warnings only (WARN)
    2 = Critical failures (FAIL)
"""

import sys
import os
import argparse
import importlib
import ast
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# ── Result tracking ──────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self):
        self.passes: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def ok(self, msg: str):
        self.passes.append(msg)
        print(f"  [OK]   {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def fail(self, msg: str):
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")

    def summary(self) -> int:
        total = len(self.passes) + len(self.warnings) + len(self.failures)
        print(f"\n{'='*60}")
        print(f"PRE-FLIGHT SUMMARY: {len(self.passes)} passed, "
              f"{len(self.warnings)} warnings, {len(self.failures)} failures "
              f"(out of {total} checks)")
        if self.failures:
            print(f"\nCRITICAL FAILURES:")
            for f in self.failures:
                print(f"  [FAIL] {f}")
            return 2
        elif self.warnings:
            print(f"\nWARNINGS:")
            for w in self.warnings:
                print(f"  [WARN] {w}")
            return 1
        else:
            print("\nAll checks passed!")
            return 0


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_pipeline(result: CheckResult):
    print("\n--- PIPELINE CHECKS ---")

    # P1: Hurst exponent computed on returns, not prices
    sota_file = SRC_DIR / "pipeline" / "sota_shared_logic_v50.py"
    if sota_file.exists():
        content = sota_file.read_text(encoding="utf-8", errors="replace")

        # Check Hurst is called on returns/diffs, not raw close prices
        # Look for the call pattern to get_rs_hurst_rolling
        hurst_calls = [line.strip() for line in content.split("\n")
                       if "get_rs_hurst_rolling" in line and not line.strip().startswith("#")]
        for call in hurst_calls:
            if "close" in call.lower() and "return" not in call.lower() and "diff" not in call.lower():
                result.fail(f"Hurst computed on prices, not returns: {call[:80]}")
            else:
                result.ok("Hurst input appears to use returns/diffs")

        # P2: Check frac_diff d value isn't hardcoded for all assets
        if "d=0.4" in content and "d=" not in content.replace("d=0.4", ""):
            result.warn("FracDiff uses fixed d=0.4 for all assets (should be adaptive per asset)")
        else:
            result.ok("FracDiff d parameter check passed")

        # P3: Check target clipping is reasonable
        if "clip(-0.15" in content or "clip(lower=-0.15" in content:
            result.warn("Target clipping uses fixed +-0.15 (should be per-asset percentile)")

        # P4: Check robust_normalize uses forward-fill for warmup (not fill_null(0.0))
        if 'fill_null(strategy="forward")' in content or "fill_null(strategy='forward')" in content:
            result.ok("robust_normalize uses forward-fill for warmup nulls")
        elif "def robust_normalize" in content:
            # Check the function body for fill_null(0.0) without forward fill
            fn_start = content.index("def robust_normalize")
            fn_end = content.index("\ndef ", fn_start + 1) if "\ndef " in content[fn_start + 1:] else fn_start + 500
            fn_body = content[fn_start:fn_end]
            if "fill_null(0.0)" in fn_body and "forward" not in fn_body:
                result.warn("robust_normalize fills warmup nulls with 0.0 (first ~200 bars are fake zeros)")
            else:
                result.ok("robust_normalize null handling appears correct")

        # P5: Check VPIN implementation
        if "raw_vpin_proxy" in content or 'volume.*abs' in content:
            result.warn("VPIN appears to be a crude proxy (volume*|return|), not real VPIN")

        # P6: Timestamp validation (check both pipeline files)
        maker_file = SRC_DIR / "pipeline" / "make_dataset_legacy.py"
        maker_content = ""
        if maker_file.exists():
            maker_content = maker_file.read_text(encoding="utf-8", errors="replace")
        combined = content + maker_content
        if "MIN_TS" in combined or "1_577_836_800_000" in combined or "1_640_000_000_000" in combined:
            result.ok("Timestamp range validation present")
        elif "1e12" in combined or "1.5e12" in combined:
            result.ok("Timestamp range check present (basic)")
        else:
            result.warn("No timestamp range validation found")

        # P7: Check target null handling — should drop ALL target nulls, not fill with 0.0
        if "drop_nulls" in content:
            # Check if auxiliary targets are in the drop list (not filled with 0.0)
            drop_section = content[content.index("drop_nulls"):][:500]
            if "target_return_50" in drop_section or "target_vol_20" in drop_section or "all_targets" in drop_section:
                result.ok("Target nulls dropped on ALL targets (no fabrication)")
            elif "fill_null(0.0)" in content:
                # Check if fill_null(0.0) is applied to auxiliary targets after drop_nulls
                drop_idx = content.index("drop_nulls")
                after_drop = content[drop_idx:]
                if 'col("target_return_50").fill_null(0.0)' in after_drop:
                    result.warn("Auxiliary targets filled with 0.0 after drop_nulls (tail fabrication)")
                else:
                    result.ok("Target null handling appears correct")

    else:
        result.fail(f"Pipeline file not found: {sota_file}")

    # P7: Check data_config.yaml exists and has all 10 assets
    config_file = CONFIG_DIR / "data_config.yaml"
    if config_file.exists():
        config_text = config_file.read_text(encoding="utf-8", errors="replace")
        expected_assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
        found = sum(1 for a in expected_assets if a in config_text)
        if found == 10:
            result.ok(f"data_config.yaml has all 10 assets")
        else:
            result.warn(f"data_config.yaml has {found}/10 expected assets")
    else:
        result.fail("data_config.yaml not found")


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CHECKS (per version)
# ══════════════════════════════════════════════════════════════════════════════

def check_architecture(result: CheckResult, versions: Optional[List[int]] = None):
    print("\n--- ARCHITECTURE CHECKS ---")

    if versions is None:
        versions = list(range(1, 10))

    for v in versions:
        vdir = SRC_DIR / f"v{v}_training"
        if not vdir.exists():
            result.warn(f"V{v} directory not found: {vdir}")
            continue

        print(f"\n  -- V{v} --")

        # Read key files
        settings_file = vdir / "settings.py"
        components_file = vdir / "components.py"
        world_model_file = vdir / "world_model.py"
        train_file = vdir / "train_world_model.py"

        settings_text = ""
        components_text = ""
        wm_text = ""
        train_text = ""

        if settings_file.exists():
            settings_text = settings_file.read_text(encoding="utf-8", errors="replace")
        if components_file.exists():
            components_text = components_file.read_text(encoding="utf-8", errors="replace")
        if world_model_file.exists():
            wm_text = world_model_file.read_text(encoding="utf-8", errors="replace")
        if train_file.exists():
            train_text = train_file.read_text(encoding="utf-8", errors="replace")

        # ── Settings Consistency ──
        # A1: Check FEATURE_LIST has 13 items
        feat_match = re.search(r'FEATURE_LIST\s*=\s*\[([^\]]+)\]', settings_text, re.DOTALL)
        if feat_match:
            # Extract only quoted strings (ignores inline comments with commas)
            feats = re.findall(r'"([^"]+)"', feat_match.group(1))
            if not feats:
                feats = re.findall(r"'([^']+)'", feat_match.group(1))
            if len(feats) == 13:
                result.ok(f"V{v}: FEATURE_LIST has 13 features")
            else:
                result.fail(f"V{v}: FEATURE_LIST has {len(feats)} features (expected 13)")

        # A2: Check REWARD_HORIZONS
        if "REWARD_HORIZONS" in settings_text:
            if "[1, 4, 16, 64]" in settings_text or "[1,4,16,64]" in settings_text:
                result.ok(f"V{v}: REWARD_HORIZONS = [1, 4, 16, 64]")
            else:
                result.warn(f"V{v}: REWARD_HORIZONS may not be [1, 4, 16, 64]")

        # A3: Check NUM_BINS = 255
        if "NUM_BINS" in settings_text:
            if "255" in settings_text:
                result.ok(f"V{v}: NUM_BINS = 255")
            else:
                result.warn(f"V{v}: NUM_BINS is not 255")

        # A4: Check NUM_WORKERS = 0 (Windows)
        if "NUM_WORKERS" in settings_text:
            if "NUM_WORKERS = 0" in settings_text or "NUM_WORKERS=0" in settings_text:
                result.ok(f"V{v}: NUM_WORKERS = 0 (Windows safe)")
            else:
                result.warn(f"V{v}: NUM_WORKERS may not be 0 (Windows requires 0)")

        # ── Causality Checks ──
        # A5: Check for bidirectional=True in components (V2 known broken)
        if "bidirectional=True" in components_text or "bidirectional=True" in wm_text:
            if v == 2:
                result.fail(f"V{v}: BiGRU uses bidirectional=True (KNOWN future leakage)")
            else:
                result.fail(f"V{v}: Found bidirectional=True (potential future leakage)")

        # A6: Check attention has causal mask (V5, V7 known missing)
        if "MultiheadAttention" in components_text or "self.attn(" in components_text:
            if "causal" in components_text.lower() or "tril" in components_text or "is_causal" in components_text:
                result.ok(f"V{v}: Attention appears to have causal masking")
            elif v in [5, 7]:
                result.fail(f"V{v}: Attention has NO causal mask (future leakage)")
            elif v == 1:
                # V1 has manual causal mask via tril
                if "tril" in components_text:
                    result.ok(f"V{v}: Causal mask via torch.tril found")
                else:
                    result.warn(f"V{v}: No causal mask detected in attention")

        # ── Numerical Stability ──
        # A7: Check log_vars clamping
        if "log_vars" in wm_text:
            if "clamp" in wm_text and "log_var" in wm_text:
                result.ok(f"V{v}: log_vars appears to be clamped")
            else:
                result.warn(f"V{v}: log_vars is UNCLAMPED (can zero any loss term)")

        # A8: Check TwoHot decode uses fp32
        if "TwoHotSymlog" in components_text:
            if ".float()" in components_text and ("decode" in components_text or "softmax" in components_text):
                result.ok(f"V{v}: TwoHot appears to use fp32 for softmax")
            else:
                result.warn(f"V{v}: TwoHot decode may not force fp32 (fp16 overflow risk)")

        # A9: Check TwoHotSymlog defaults match settings
        twohot_defaults = re.search(r'class TwoHotSymlog.*?def __init__\(self.*?num_bins.*?=\s*(\d+)',
                                     components_text, re.DOTALL)
        if twohot_defaults:
            default_bins = int(twohot_defaults.group(1))
            if default_bins != 255:
                result.warn(f"V{v}: TwoHotSymlog default num_bins={default_bins} (settings use 255)")
            else:
                result.ok(f"V{v}: TwoHotSymlog defaults match settings")

        # ── Weight Initialization ──
        # A10: Check _init_weights exists
        if "_init_weights" in wm_text:
            result.ok(f"V{v}: _init_weights() defined")
        else:
            result.warn(f"V{v}: No _init_weights() (using PyTorch defaults)")

        # ── V6-Specific ──
        if v == 6:
            # A11: Check V6 gradient penalty is actually used
            if "DISC_GRAD_PENALTY" in settings_text:
                if "grad_penalty" in wm_text or "gradient_penalty" in wm_text:
                    result.ok(f"V{v}: Gradient penalty appears implemented")
                else:
                    result.fail(f"V{v}: DISC_GRAD_PENALTY configured but NOT used in loss")

            # A12: Check for separate GradScalers
            if "retain_graph=True" in train_text:
                result.warn(f"V{v}: retain_graph=True found (doubles peak VRAM)")
                # Check for dual scalers
                scaler_count = train_text.count("GradScaler")
                if scaler_count < 2:
                    result.fail(f"V{v}: Single GradScaler for dual optimizer (should use 2)")

        # ── V8-Specific ──
        if v == 8:
            if "checkpoint" in components_text.lower() or "adjoint" in components_text.lower():
                result.ok(f"V{v}: ODE solver has gradient checkpointing/adjoint")
            else:
                result.warn(f"V{v}: ODE solver stores full graph (380 calls, high VRAM)")

        # ── V9-Specific ──
        if v == 9:
            if "balance" in wm_text.lower() or "load_balance" in wm_text.lower():
                result.ok(f"V{v}: MoE has load balancing")
            else:
                result.warn(f"V{v}: MoE router has no load balancing (expert collapse risk)")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_training(result: CheckResult, versions: Optional[List[int]] = None):
    print("\n--- TRAINING CHECKS ---")

    if versions is None:
        versions = list(range(1, 10))

    # T1: Anti-fragile config checks
    af_file = SRC_DIR / "anti_fragile.py"
    if af_file.exists():
        af_text = af_file.read_text(encoding="utf-8", errors="replace")

        # T1a: Time reversal should be disabled
        if "aug_time_reverse_prob" in af_text:
            match = re.search(r'aug_time_reverse_prob.*?=\s*([\d.]+)', af_text)
            if match:
                val = float(match.group(1))
                if val > 0:
                    result.fail(f"aug_time_reverse_prob = {val} (should be 0.0 — corrupts targets)")
                else:
                    result.ok("aug_time_reverse_prob = 0.0 (safe)")

        # T1b: Block swap should be disabled for RSSM models
        match = re.search(r'aug_block_swap_prob.*?=\s*([\d.]+)', af_text)
        if match:
            val = float(match.group(1))
            if val > 0:
                result.warn(f"aug_block_swap_prob = {val} (creates RSSM state discontinuities)")
            else:
                result.ok("aug_block_swap_prob = 0.0 (safe for RSSM)")

        # T1c: Shuffled IC seed should vary
        if "seed=42" in af_text and "seed=42 +" not in af_text:
            result.warn("Shuffled IC uses fixed seed=42 (should vary per epoch)")
        else:
            result.ok("Shuffled IC seed appears to vary")

        # T1d: Shuffle should be at sequence level
        # Check if the shuffle operates on indices then slices into windows
        if "rng.shuffle(indices)" in af_text or "rng.shuffle(fold_indices)" in af_text:
            result.warn("Shuffled IC shuffles at bar level, not sequence level (destroys input coherence)")

        # T1e: Check missing feature warning
        if "fill_null" in af_text or "np.zeros" in af_text:
            if "[WARN]" in af_text or "warning" in af_text.lower():
                result.ok("Missing feature warning present")
            else:
                result.warn("Missing features silently filled with zeros (no warning)")

    # T2: Per-version training checks
    for v in versions:
        vdir = SRC_DIR / f"v{v}_training"
        train_file = vdir / "train_world_model.py"
        settings_file = vdir / "settings.py"

        if not train_file.exists():
            continue

        train_text = train_file.read_text(encoding="utf-8", errors="replace")
        settings_text = ""
        if settings_file.exists():
            settings_text = settings_file.read_text(encoding="utf-8", errors="replace")

        # T2a: EMA state saved in checkpoint
        if v not in [2, 6]:  # JEPA models handle EMA differently
            if "ema" in train_text.lower():
                if "ema_model.state_dict()" in train_text or "ema_state" in train_text:
                    # Check if it's in the save dict
                    if re.search(r'["\']ema.*state', train_text):
                        result.ok(f"V{v}: EMA state saved in checkpoint")
                    else:
                        result.fail(f"V{v}: EMA model used but state NOT saved in checkpoint (lost on resume)")
                else:
                    result.warn(f"V{v}: EMA model present but state_dict save not detected")

        # T2b: AdamW parameter groups (weight decay on norms check)
        if "AdamW" in train_text:
            if "param_groups" in train_text or "no_decay" in train_text or "weight_decay.*0.0" in train_text:
                result.ok(f"V{v}: AdamW uses parameter groups (proper weight decay)")
            else:
                result.warn(f"V{v}: AdamW applies weight decay to ALL params (including LayerNorm/bias)")

        # T2c: NaN check before backward
        if "isnan" in train_text or "isfinite" in train_text:
            result.ok(f"V{v}: NaN check present in training loop")
        else:
            result.warn(f"V{v}: No NaN check before backward pass")

        # T2d: Check weight decay value
        wd_match = re.search(r'WM_WEIGHT_DECAY\s*=\s*([\d.e-]+)', settings_text)
        if wd_match:
            wd = float(wd_match.group(1))
            if wd > 0.1:
                result.warn(f"V{v}: WM_WEIGHT_DECAY = {wd} (very high, may impair convergence)")
            elif wd > 0:
                result.ok(f"V{v}: WM_WEIGHT_DECAY = {wd}")

        # T2e: Check for emoji in print statements
        for pyfile in vdir.glob("*.py"):
            text = pyfile.read_text(encoding="utf-8", errors="replace")
            # Check for common emoji ranges
            emoji_pattern = re.compile(
                "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF"
                "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]"
            )
            emojis = emoji_pattern.findall(text)
            if emojis:
                result.fail(f"V{v}/{pyfile.name}: Contains emoji characters (Windows cp1252 crash)")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_validation(result: CheckResult, versions: Optional[List[int]] = None):
    print("\n--- VALIDATION CHECKS ---")

    if versions is None:
        versions = list(range(1, 10))

    # V1: Check validation_utils.py
    val_file = SRC_DIR / "validation_utils.py"
    if val_file.exists():
        val_text = val_file.read_text(encoding="utf-8", errors="replace")

        # V1a: Shuffled IC in robustness gates
        if "shuffled" in val_text.lower() and "gate" in val_text.lower():
            result.ok("Shuffled IC appears in robustness gates")
        else:
            result.warn("Shuffled IC may not be in robustness gates")

    # V2: Per-version validate_world.py checks
    for v in versions:
        vdir = SRC_DIR / f"v{v}_training"
        val_script = vdir / "validate_world.py"

        if not val_script.exists():
            continue

        vw_text = val_script.read_text(encoding="utf-8", errors="replace")

        # V2a: Check if shuffled IC is in standard gate check
        if "_check_gates" in vw_text or "check_gate" in vw_text:
            # Find the full _check_gates method (up to next def or class)
            gate_start = vw_text.find("_check_gates")
            gate_end = vw_text.find("\n    def ", gate_start + 1)
            if gate_end == -1:
                gate_end = gate_start + 2000
            gate_section = vw_text[gate_start:gate_end]
            if "shuffled" in gate_section.lower() or "SHUFFLED_IC" in gate_section:
                result.ok(f"V{v}: Shuffled IC in gate checks")
            else:
                result.fail(f"V{v}: Shuffled IC NOT in standard gate checks (false positives possible)")

        # V2b: Check for train/val loss ratio gate (constant defined in settings)
        settings_file = SRC_DIR / f"v{v}_training" / "settings.py"
        if settings_file.exists():
            settings_text = settings_file.read_text(encoding="utf-8", errors="replace")
            if "GATE_LOSS_RATIO_MAX" in settings_text:
                result.ok(f"V{v}: Train/val loss ratio gate constant defined")
            else:
                result.warn(f"V{v}: No GATE_LOSS_RATIO_MAX constant (add to settings.py)")

        # V2c: Check GATE_SHUFFLED_IC_RATIO_MIN in settings
        if settings_file.exists():
            if "GATE_SHUFFLED_IC_RATIO_MIN" in settings_text:
                result.ok(f"V{v}: GATE_SHUFFLED_IC_RATIO_MIN defined in settings")
            else:
                result.fail(f"V{v}: Missing GATE_SHUFFLED_IC_RATIO_MIN in settings")

        # V2d: Check _compute_shuffled_ic method exists
        if "_compute_shuffled_ic" in vw_text:
            result.ok(f"V{v}: Shuffled IC computation method present")
        elif v != 2:  # V2 is quarantined
            result.warn(f"V{v}: No _compute_shuffled_ic method in validate_world.py")

        # V2e: Check overlapping window stride
        stride_match = re.search(r'stride\s*=\s*seq_len\s*//\s*(\d+)', vw_text)
        if stride_match:
            divisor = int(stride_match.group(1))
            if divisor > 1:
                result.warn(f"V{v}: Validation uses overlapping windows (stride=seq_len//{divisor}, inflates IC)")
            else:
                result.ok(f"V{v}: Validation uses non-overlapping windows")

        # V2d: Check for NaN handling in corrcoef
        if "isfinite" in vw_text or "isnan" in vw_text:
            result.ok(f"V{v}: NaN handling in IC computation")
        else:
            result.warn(f"V{v}: No NaN check on corrcoef result")


# ══════════════════════════════════════════════════════════════════════════════
# DATA INTEGRITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_data(result: CheckResult):
    print("\n--- DATA INTEGRITY CHECKS ---")

    processed_dir = DATA_DIR / "processed"
    if not processed_dir.exists():
        result.warn("No processed data directory found")
        return

    parquet_files = list(processed_dir.glob("*_v50_chimera*.parquet"))
    if not parquet_files:
        result.warn("No processed parquet files found")
        return

    result.ok(f"Found {len(parquet_files)} processed parquet files")

    # Try to load and validate each file
    try:
        import polars as pl

        expected_features = [
            "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
            "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
            "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
            "norm_spread_bps"
        ]
        expected_targets = [
            "target_return_1", "target_return_4", "target_return_16",
            "target_return_64", "target_return_50", "target_vol_20"
        ]

        for pf in parquet_files:
            asset_name = pf.stem.replace("_v50_chimera", "")
            # Read schema only (fast)
            schema = pl.read_parquet_schema(pf)
            cols = list(schema.keys())

            # D1: Check all 13 features present
            missing_feats = [f for f in expected_features if f not in cols]
            if missing_feats:
                result.fail(f"{asset_name}: Missing features: {missing_feats}")
            else:
                result.ok(f"{asset_name}: All 13 features present")

            # D2: Check all 6 targets present
            missing_tgts = [t for t in expected_targets if t not in cols]
            if missing_tgts:
                result.fail(f"{asset_name}: Missing targets: {missing_tgts}")
            else:
                result.ok(f"{asset_name}: All 6 targets present")

            # D3: Check timestamp column exists
            if "timestamp" not in cols:
                result.fail(f"{asset_name}: No 'timestamp' column")

            # D4: Check bar_id exists
            if "bar_id" not in cols:
                result.warn(f"{asset_name}: No 'bar_id' column")

            # D5: Spot-check data quality (first 5 and last 5 rows)
            try:
                df = pl.read_parquet(pf)
                n_rows = len(df)
                result.ok(f"{asset_name}: {n_rows:,} bars loaded")

                # Check timestamp range (13-digit milliseconds)
                if "timestamp" in cols:
                    ts_min = df["timestamp"].min()
                    ts_max = df["timestamp"].max()
                    if ts_min is not None and ts_max is not None:
                        if 1.0e12 < ts_min < 2.0e12 and 1.0e12 < ts_max < 2.0e12:
                            result.ok(f"{asset_name}: Timestamps in valid 13-digit ms range")
                        else:
                            result.fail(f"{asset_name}: Timestamp range invalid: [{ts_min}, {ts_max}]")

                # Check feature nulls
                for feat in expected_features:
                    if feat in cols:
                        null_count = df[feat].null_count()
                        null_pct = null_count / n_rows * 100
                        if null_pct > 5:
                            result.warn(f"{asset_name}: {feat} has {null_pct:.1f}% nulls")

                # Check target tail: compare tail zero-rate vs dataset-wide zero-rate
                # Original bug: fill_null(0.0) fabricated zeros ONLY at the tail
                # Detection: tail zero-rate significantly higher than baseline = fabrication
                for tgt in expected_targets:  # Check ALL targets (primary + auxiliary)
                    if tgt in cols:
                        tail = df[tgt].tail(100)
                        tail_zeros = int((tail == 0.0).sum())
                        total_zeros = int((df[tgt] == 0.0).sum())
                        baseline_rate = total_zeros / n_rows if n_rows > 0 else 0
                        expected_in_100 = baseline_rate * 100
                        # Flag if tail has >3x expected zeros AND >5 absolute (avoid noise on low-zero assets)
                        if tail_zeros > max(5, expected_in_100 * 3):
                            result.fail(f"{asset_name}: {tgt} has {tail_zeros} zeros in last 100 rows vs {expected_in_100:.1f} expected (tail corruption)")

                # Check for null targets (should be zero after drop_nulls in pipeline)
                for tgt in expected_targets:
                    if tgt in cols:
                        null_count = df[tgt].null_count()
                        if null_count > 0:
                            result.warn(f"{asset_name}: {tgt} has {null_count} nulls (should be 0 after pipeline)")

                # Check feature std approximately 1.0 (skip warmup)
                warmup = min(1000, n_rows // 5)
                df_post_warmup = df.slice(warmup)
                for feat in expected_features:
                    if feat in cols:
                        std = df_post_warmup[feat].std()
                        if std is not None:
                            if std < 0.3 or std > 3.0:
                                result.warn(f"{asset_name}: {feat} std={std:.2f} (expected ~1.0)")

            except Exception as e:
                result.warn(f"{asset_name}: Could not read full file: {e}")

    except ImportError:
        result.warn("Polars not installed, skipping detailed data checks")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VERSION CONSISTENCY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def check_consistency(result: CheckResult):
    print("\n--- CROSS-VERSION CONSISTENCY ---")

    feature_lists = {}
    reward_horizons = {}
    num_assets = {}

    for v in range(1, 10):
        settings_file = SRC_DIR / f"v{v}_training" / "settings.py"
        if not settings_file.exists():
            continue

        text = settings_file.read_text(encoding="utf-8", errors="replace")

        # Extract FEATURE_LIST (parse only quoted strings, ignore comments)
        feat_match = re.search(r'FEATURE_LIST\s*=\s*\[([^\]]+)\]', text, re.DOTALL)
        if feat_match:
            feats = tuple(re.findall(r'"([^"]+)"', feat_match.group(1)))
            if not feats:
                feats = tuple(re.findall(r"'([^']+)'", feat_match.group(1)))
            feature_lists[v] = feats

        # Extract REWARD_HORIZONS
        rh_match = re.search(r'REWARD_HORIZONS\s*=\s*\[([^\]]+)\]', text)
        if rh_match:
            reward_horizons[v] = rh_match.group(1).strip()

        # Extract NUM_ASSETS
        na_match = re.search(r'NUM_ASSETS\s*=\s*(\d+)', text)
        if na_match:
            num_assets[v] = int(na_match.group(1))

    # Check FEATURE_LIST consistency
    unique_flists = set(feature_lists.values())
    if len(unique_flists) == 1:
        result.ok(f"FEATURE_LIST consistent across all {len(feature_lists)} versions")
    elif len(unique_flists) > 1:
        result.fail(f"FEATURE_LIST INCONSISTENT across versions: {len(unique_flists)} unique lists")
        for v, fl in feature_lists.items():
            result.warn(f"  V{v}: {len(fl)} features")

    # Check REWARD_HORIZONS consistency
    unique_rh = set(reward_horizons.values())
    if len(unique_rh) == 1:
        result.ok(f"REWARD_HORIZONS consistent across all {len(reward_horizons)} versions")
    elif len(unique_rh) > 1:
        result.fail(f"REWARD_HORIZONS INCONSISTENT across versions")

    # Check NUM_ASSETS consistency
    unique_na = set(num_assets.values())
    if len(unique_na) == 1:
        result.ok(f"NUM_ASSETS = {list(unique_na)[0]} consistent across all versions")
    elif len(unique_na) > 1:
        result.fail(f"NUM_ASSETS INCONSISTENT across versions: {unique_na}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V4 Crypto System Pre-Flight Checks")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--pipeline", action="store_true", help="Pipeline checks")
    parser.add_argument("--architecture", action="store_true", help="Architecture checks")
    parser.add_argument("--training", action="store_true", help="Training config checks")
    parser.add_argument("--validation", action="store_true", help="Validation checks")
    parser.add_argument("--data", action="store_true", help="Data integrity checks")
    parser.add_argument("--consistency", action="store_true", help="Cross-version consistency")
    parser.add_argument("--version", type=int, nargs="+", help="Check specific version(s)")

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.all, args.pipeline, args.architecture, args.training,
                args.validation, args.data, args.consistency]):
        args.all = True

    versions = args.version if args.version else None

    result = CheckResult()

    print("=" * 60)
    print("V4 CRYPTO SYSTEM — PRE-FLIGHT CHECKS")
    print("=" * 60)

    if args.all or args.pipeline:
        check_pipeline(result)

    if args.all or args.architecture:
        check_architecture(result, versions)

    if args.all or args.training:
        check_training(result, versions)

    if args.all or args.validation:
        check_validation(result, versions)

    if args.all or args.data:
        check_data(result)

    if args.all or args.consistency:
        check_consistency(result)

    exit_code = result.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
