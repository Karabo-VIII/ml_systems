"""
Propagate V1 training infrastructure features to V2-V9 variant trainers.

Phase 1 (already applied):
1. Add FeatureAutopsy import to snapshot and NCL trainers
2. Add kl_raw to epoch_keys where missing (snapshot trainers)
3. Expand Kendall weight logging (w_kl, w_reg)

Phase 2 (this run):
4. Add autopsy setup + run() calls in training loops
5. Add NaN recovery mechanism (reinit model on >50% NaN batches)
6. Add checkpoint collision guards (n_features verification on resume)
7. Add n_features to checkpoint save dicts

Run: python tools/propagate_training_features.py --dry-run  (preview)
      python tools/propagate_training_features.py            (apply)
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

# RSSM versions (have KL divergence)
RSSM_VERSIONS = [3, 4, 5, 7, 8, 9]
# JEPA versions (no KL, use contrastive + VICReg)
JEPA_VERSIONS = [2, 6]
ALL_VERSIONS = sorted(RSSM_VERSIONS + JEPA_VERSIONS)

DRY_RUN = "--dry-run" in sys.argv
modified_files = []


def find_all_trainer_files(trainer_type: str) -> list:
    """Find all train_<type>.py files across V2-V9 base and variants."""
    files = []
    for major in ALL_VERSIONS:
        base_dir = SRC / f"v{major}" / f"v{major}_training"
        f = base_dir / f"train_{trainer_type}.py"
        if f.exists():
            files.append(f)
        # Variants
        for variant in range(1, 4):
            var_dir = SRC / f"v{major}" / f"v{major}_{variant}_training"
            f = var_dir / f"train_{trainer_type}.py"
            if f.exists():
                files.append(f)
    return files


def get_version_from_path(filepath: Path) -> int:
    """Extract major version number from file path."""
    for p in filepath.parts:
        m = re.match(r"v(\d+)", p)
        if m:
            return int(m.group(1))
    return -1


def is_variant(filepath: Path) -> bool:
    """Check if this is a variant trainer (_1, _2, _3) vs base."""
    for p in filepath.parts:
        if re.match(r"v\d+_\d+_training", p):
            return True
    return False


def get_version_tag(filepath: Path) -> str:
    """Get version tag like 'v3e' or 'v4_1e' from path."""
    parts = filepath.parts
    for p in parts:
        m = re.match(r"v(\d+)_(\d+)_training", p)
        if m:
            return f"v{m.group(1)}_{m.group(2)}"
        m = re.match(r"v(\d+)_training", p)
        if m:
            return f"v{m.group(1)}"
    return "vX"


# ==============================================================================
# Phase 2: Autopsy setup + run()
# ==============================================================================

def add_autopsy_setup_snapshot(content: str, filepath: Path) -> str:
    """Add autopsy initialization after ckpt_history = [] in snapshot trainers."""
    if "autopsy = FeatureAutopsy(" in content:
        return content  # Already has it

    vtag = get_version_tag(filepath)

    # Anchor: after "ckpt_history = []" and "early_stopped = False"
    anchor = "    early_stopped = False\n"
    if anchor not in content:
        # Some files might not have early_stopped
        anchor = "    ckpt_history = []\n"
        if anchor not in content:
            return content

    autopsy_block = (
        "\n"
        "    # -- Feature Autopsy (non-console diagnostics) ----------------------------\n"
        f'    autopsy_path = ENSEMBLE_MODEL_DIR / f"{vtag}e_seed_{{seed_idx}}_autopsy.jsonl"\n'
        "    autopsy = FeatureAutopsy(\n"
        "        feature_list=FEATURE_LIST, base_dim=INPUT_DIM,\n"
        "        log_path=autopsy_path, horizons=REWARD_HORIZONS, device=DEVICE,\n"
        "    )\n"
    )

    content = content.replace(anchor, anchor + autopsy_block, 1)
    return content


def add_autopsy_setup_ncl(content: str, filepath: Path) -> str:
    """Add autopsy initialization after ckpt_history = [] in NCL trainers."""
    if "autopsy = FeatureAutopsy(" in content:
        return content

    vtag = get_version_tag(filepath)

    anchor = "    ckpt_history = []\n"
    if anchor not in content:
        return content

    autopsy_block = (
        "\n"
        "    # -- Feature Autopsy (non-console diagnostics) ----------------------------\n"
        f'    autopsy_path = NCL_MODEL_DIR / f"{vtag}d_autopsy.jsonl"\n'
        "    autopsy = FeatureAutopsy(\n"
        "        feature_list=FEATURE_LIST, base_dim=INPUT_DIM,\n"
        "        log_path=autopsy_path, horizons=REWARD_HORIZONS, device=DEVICE,\n"
        "    )\n"
    )

    content = content.replace(anchor, anchor + autopsy_block, 1)
    return content


def add_autopsy_run_snapshot(content: str, filepath: Path) -> str:
    """Add autopsy.run() call in snapshot validation block."""
    if "autopsy.run(" in content:
        return content

    # Anchor: after "if not passed:" + print reason block, before early stopping
    # Pattern: the line right before "# Early stopping" or "if patience_counter >= WM_PATIENCE"
    anchor = "            # Early stopping\n            if patience_counter >= WM_PATIENCE:"
    if anchor not in content:
        # Try alternate spacing
        anchor = "            # Early stopping\n            if patience_counter >= WM_PATIENCE:"
        if anchor not in content:
            return content

    autopsy_block = (
        "            # -- Feature Autopsy (non-console diagnostics) --------------------\n"
        "            try:\n"
        "                do_ablation = (epoch + 1) % 10 == 0\n"
        "                do_raw_ic = (epoch + 1 == WM_VAL_EVERY)\n"
        "                autopsy.run(\n"
        "                    ema_model, val_loader, epoch + 1, revin=revin,\n"
        "                    do_ablation=do_ablation, do_raw_ic=do_raw_ic,\n"
        "                )\n"
        "            except Exception:\n"
        "                pass  # autopsy must never crash training\n"
        "\n"
    )

    content = content.replace(anchor, autopsy_block + anchor, 1)
    return content


def add_autopsy_run_ncl(content: str, filepath: Path) -> str:
    """Add autopsy.run() call in NCL validation block."""
    if "autopsy.run(" in content:
        return content

    # Try with comment first, then without
    anchor = "            # Early stopping\n            if patience_counter >= WM_PATIENCE:"
    if anchor not in content:
        # Some NCL trainers lack the comment
        anchor = "            if patience_counter >= WM_PATIENCE:"
        if anchor not in content:
            return content

    autopsy_block = (
        "            # -- Feature Autopsy (non-console diagnostics) --------------------\n"
        "            try:\n"
        "                do_ablation = (epoch + 1) % 10 == 0\n"
        "                do_raw_ic = (epoch + 1 == WM_VAL_EVERY)\n"
        "                autopsy.run(\n"
        "                    ema_model, val_loader, epoch + 1, revin=revin,\n"
        "                    do_ablation=do_ablation, do_raw_ic=do_raw_ic,\n"
        "                )\n"
        "            except Exception:\n"
        "                pass  # autopsy must never crash training\n"
        "\n"
    )

    content = content.replace(anchor, autopsy_block + anchor, 1)
    return content


# ==============================================================================
# Phase 2: NaN Recovery
# ==============================================================================

def add_nan_recovery_snapshot(content: str, filepath: Path) -> str:
    """Add NaN recovery mechanism before memory cleanup in snapshot trainers."""
    if "nan_recovery_count" in content:
        return content  # Already has it

    # Need to initialize nan_recovery_count near other training state vars
    # Add it after "early_stopped = False"
    init_anchor = "    early_stopped = False\n"
    if init_anchor in content and "nan_recovery_count" not in content:
        content = content.replace(
            init_anchor,
            init_anchor + "    nan_recovery_count = 0\n",
            1,
        )

    # Insert NaN recovery block before gc.collect() memory cleanup
    # The pattern is: after epoch summary print, before gc.collect()
    # Use gc.collect() as anchor since it's universal
    gc_anchor = "        if (epoch + 1) % 10 == 0:\n            gc.collect()"
    if gc_anchor not in content:
        return content

    nan_recovery_block = (
        "        # -- NaN collapse recovery ------------------------------------------------\n"
        "        nan_frac = nan_count / WM_STEPS_PER_EPOCH if WM_STEPS_PER_EPOCH > 0 else 0\n"
        "        if nan_frac > 0.5:\n"
        "            nan_recovery_count += 1\n"
        "            if nan_recovery_count > 3:\n"
        '                print(f"  [FATAL] NaN collapse after {nan_recovery_count} recovery attempts. Aborting.")\n'
        "                break\n"
        '            print(f"  [NaN RECOVERY] {nan_frac:.0%} NaN batches at epoch {epoch+1}. "\n'
        '                  f"Reinitializing (attempt {nan_recovery_count}/3)")\n'
        "            for m in model.modules():\n"
        "                if hasattr(m, 'reset_parameters'):\n"
        "                    m.reset_parameters()\n"
        "            if hasattr(model, 'log_vars'):\n"
        "                model.log_vars.data.zero_()\n"
        "            optimizer = torch.optim.AdamW(\n"
        "                list(model.parameters()), lr=current_lr,\n"
        "                weight_decay=WM_WEIGHT_DECAY,\n"
        "            )\n"
        '            scaler = torch.amp.GradScaler("cuda")\n'
        "            ema_model = copy.deepcopy(model)\n"
        '            best_val_loss = float("inf")\n'
        '            best_shuffled_ic = -float("inf")\n'
        "            patience_counter = 0\n"
        '            print(f"  [NaN RECOVERY] Model reinitialized. Continuing from epoch {epoch+2}.")\n'
        "            continue\n"
        "\n"
    )

    content = content.replace(gc_anchor, nan_recovery_block + gc_anchor, 1)
    return content


def add_nan_recovery_ncl(content: str, filepath: Path) -> str:
    """Add NaN recovery mechanism before memory cleanup in NCL trainers."""
    if "nan_recovery_count" in content:
        return content

    # Initialize nan_recovery_count after ckpt_history = []
    init_anchor = "    ckpt_history = []\n"
    if init_anchor in content:
        content = content.replace(
            init_anchor,
            init_anchor + "    nan_recovery_count = 0\n",
            1,
        )

    gc_anchor = "        if (epoch + 1) % 10 == 0:\n            gc.collect()"
    if gc_anchor not in content:
        return content

    nan_recovery_block = (
        "        # -- NaN collapse recovery ------------------------------------------------\n"
        "        nan_frac = nan_count / WM_STEPS_PER_EPOCH if WM_STEPS_PER_EPOCH > 0 else 0\n"
        "        if nan_frac > 0.5:\n"
        "            nan_recovery_count += 1\n"
        "            if nan_recovery_count > 3:\n"
        '                print(f"  [FATAL] NaN collapse after {nan_recovery_count} recovery attempts. Aborting.")\n'
        "                break\n"
        '            print(f"  [NaN RECOVERY] {nan_frac:.0%} NaN batches at epoch {epoch+1}. "\n'
        '                  f"Reinitializing (attempt {nan_recovery_count}/3)")\n'
        "            for m in model.modules():\n"
        "                if hasattr(m, 'reset_parameters'):\n"
        "                    m.reset_parameters()\n"
        "            if hasattr(model, 'log_vars'):\n"
        "                model.log_vars.data.zero_()\n"
        "            all_params = list(model.parameters())\n"
        "            if revin is not None:\n"
        "                all_params += list(revin.parameters())\n"
        "            optimizer = torch.optim.AdamW(\n"
        "                all_params, lr=current_lr,\n"
        "                weight_decay=DIVERSITY_WEIGHT_DECAY,\n"
        "            )\n"
        '            scaler = torch.amp.GradScaler("cuda")\n'
        "            ema_model = copy.deepcopy(model)\n"
        '            best_val_loss = float("inf")\n'
        '            best_shuffled_ic = -float("inf")\n'
        "            patience_counter = 0\n"
        '            print(f"  [NaN RECOVERY] Model reinitialized. Continuing from epoch {epoch+2}.")\n'
        "            continue\n"
        "\n"
    )

    content = content.replace(gc_anchor, nan_recovery_block + gc_anchor, 1)
    return content


# ==============================================================================
# Phase 2: Checkpoint Collision Guards
# ==============================================================================

def add_collision_guard_snapshot(content: str, filepath: Path) -> str:
    """Add n_features collision guard in snapshot checkpoint resume."""
    if "ckpt_n_feat" in content:
        return content  # Already has it

    # In snapshot trainers, the resume section has:
    #   gate_passed = ckpt.get("gate_passed", False)
    # or:
    #   best_ic = ckpt.get("best_ic", 0.0)
    # Insert collision guard after loading state, before the "Resumed at" print

    # Find the anchor: the print line showing "Resumed at epoch"
    anchor_pattern = re.compile(
        r'(            print\(f"    Resumed at epoch \{start_epoch\},.*?\n'
        r'(?:.*?\n)*?'
        r'.*?patience.*?\))\n',
        re.DOTALL,
    )
    # Simpler anchor: look for the specific "Resumed at epoch" print
    # and insert the guard BEFORE it
    resumed_line = '            print(f"    Resumed at epoch {start_epoch},'
    if resumed_line not in content:
        return content

    guard_block = (
        "\n"
        "            # -- Checkpoint collision guard --\n"
        '            ckpt_n_feat = ckpt.get("n_features")\n'
        "            if ckpt_n_feat is not None and ckpt_n_feat != INPUT_DIM:\n"
        "                raise RuntimeError(\n"
        '                    f"Checkpoint was trained with {ckpt_n_feat} features but "\n'
        '                    f"current settings use INPUT_DIM={INPUT_DIM}. "\n'
        '                    f"Delete {ckpt_path.name} or use matching feature config."\n'
        "                )\n"
        "\n"
    )

    content = content.replace(resumed_line, guard_block + resumed_line, 1)
    return content


def add_collision_guard_ncl(content: str, filepath: Path) -> str:
    """Add n_features collision guard in NCL checkpoint resume."""
    if "ckpt_n_feat" in content:
        return content

    resumed_line = '            print(f"    Resumed at epoch {start_epoch},'
    if resumed_line not in content:
        return content

    guard_block = (
        "\n"
        "            # -- Checkpoint collision guard --\n"
        '            ckpt_n_feat = ckpt.get("n_features")\n'
        "            if ckpt_n_feat is not None and ckpt_n_feat != INPUT_DIM:\n"
        "                raise RuntimeError(\n"
        '                    f"Checkpoint was trained with {ckpt_n_feat} features but "\n'
        '                    f"current settings use INPUT_DIM={INPUT_DIM}. "\n'
        '                    f"Delete {ckpt_path.name} or use matching feature config."\n'
        "                )\n"
        "\n"
    )

    content = content.replace(resumed_line, guard_block + resumed_line, 1)
    return content


# ==============================================================================
# Phase 2: n_features in checkpoint save
# ==============================================================================

def add_n_features_to_checkpoint_snapshot(content: str, filepath: Path) -> str:
    """Add n_features field to checkpoint save dicts in snapshot trainers."""
    if '"n_features": INPUT_DIM' in content:
        return content

    # Pattern: "version": "v3e_seed_training", or "v4_1e_seed_training",
    version_pattern = re.compile(r'("version": "v\d+(?:_\d+)?e_seed_training",)\n')
    match = version_pattern.search(content)
    if match:
        old = match.group(0)
        new = old.rstrip("\n") + '\n                "n_features": INPUT_DIM,\n'
        content = content.replace(old, new, 1)
    return content


def add_n_features_to_checkpoint_ncl(content: str, filepath: Path) -> str:
    """Add n_features field to checkpoint save dicts in NCL trainers."""
    if '"n_features": INPUT_DIM' in content:
        return content

    # Pattern: "v3d_diversity_ncl_antifragile", or "v4_1d_diversity_ncl_antifragile",
    version_pattern = re.compile(r'("version": "v\d+(?:_\d+)?d_diversity_ncl_antifragile",)\n')
    match = version_pattern.search(content)
    if match:
        old = match.group(0)
        new = old.rstrip("\n") + '\n                "n_features": INPUT_DIM,\n'
        content = content.replace(old, new, 1)
    return content


# ==============================================================================
# Phase 1 (kept for completeness, already applied)
# ==============================================================================

def add_autopsy_import(content: str, filepath: Path) -> str:
    """Add FeatureAutopsy import if not present."""
    if "feature_autopsy" in content:
        return content

    if "from log_utils import" in content:
        content = content.replace(
            "from log_utils import setup_logging\n",
            "from log_utils import setup_logging\nfrom diagnostics.feature_autopsy import FeatureAutopsy\n",
            1,
        )
        content = content.replace(
            "from log_utils import setup_logging, teardown_logging\n",
            "from log_utils import setup_logging, teardown_logging\nfrom diagnostics.feature_autopsy import FeatureAutopsy\n",
            1,
        )
    elif "from log_utils" in content:
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "from log_utils" in line:
                lines.insert(i + 1, "from diagnostics.feature_autopsy import FeatureAutopsy")
                break
        content = "\n".join(lines)
    return content


def add_kl_raw_to_epoch_keys_snapshot(content: str, filepath: Path) -> str:
    """Add kl_raw to epoch_keys in snapshot trainers if missing."""
    if '"kl_raw"' in content:
        return content

    content = content.replace(
        '"total", "rec", "kl", "regime", "regime_acc"',
        '"total", "rec", "kl", "kl_raw", "regime", "regime_acc"',
    )
    content = content.replace(
        '"total", "rec", "kl", "regime", "regime_acc", "ncl"',
        '"total", "rec", "kl", "kl_raw", "regime", "regime_acc", "ncl"',
    )
    return content


def add_kl_raw_to_validate_metrics(content: str, filepath: Path) -> str:
    """Add kl_raw to validate() metrics dict if it has kl but not kl_raw."""
    if '"kl_raw"' in content:
        return content

    content = content.replace(
        '"rec": [], "kl": [], "regime"',
        '"rec": [], "kl": [], "kl_raw": [], "regime"',
    )
    return content


def expand_kendall_logging(content: str, filepath: Path, major: int) -> str:
    """Expand Kendall weight logging to include w_kl and w_reg."""
    if major in JEPA_VERSIONS:
        return content
    if "w_kl:" in content and "w_reg:" in content:
        return content

    old_pattern = '_w_r1 = math.exp(-_s_r1)'
    if old_pattern in content and '_w_kl = math.exp' not in content:
        new_block = (
            '_w_r1 = math.exp(-_s_r1)\n'
            '            _w_kl = math.exp(-_s[1].item())\n'
            '            _regime_idx = 2 + len(REWARD_HORIZONS)\n'
            '            _s_reg = _s[_regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX).item()\n'
            '            _w_reg = math.exp(-_s_reg)'
        )
        content = content.replace(old_pattern, new_block, 1)

    old_print = 'f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f}"'
    new_print = 'f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f} w_kl:{_w_kl:.2f} w_reg:{_w_reg:.2f}"'
    if old_print in content:
        content = content.replace(old_print, new_print)

    return content


# ==============================================================================
# Main processor
# ==============================================================================

def process_file(filepath: Path, trainer_type: str) -> tuple:
    """Process a single trainer file. Returns (modified: bool, changes: list)."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    changes = []

    major = get_version_from_path(filepath)
    if major == -1:
        print(f"  [SKIP] Cannot determine version: {filepath}")
        return False, []

    # Phase 1: imports and logging (idempotent, already applied to most)
    before = content
    content = add_autopsy_import(content, filepath)
    if content != before:
        changes.append("autopsy_import")

    if trainer_type in ("snapshot", "ncl"):
        if major in RSSM_VERSIONS:
            before = content
            content = add_kl_raw_to_epoch_keys_snapshot(content, filepath)
            if content != before:
                changes.append("kl_raw_epoch_keys")

            before = content
            content = add_kl_raw_to_validate_metrics(content, filepath)
            if content != before:
                changes.append("kl_raw_validate")

        before = content
        content = expand_kendall_logging(content, filepath, major)
        if content != before:
            changes.append("kendall_expand")

    # Phase 2: autopsy setup + run, NaN recovery, collision guard, n_features
    if trainer_type == "snapshot":
        before = content
        content = add_autopsy_setup_snapshot(content, filepath)
        if content != before:
            changes.append("autopsy_setup")

        before = content
        content = add_autopsy_run_snapshot(content, filepath)
        if content != before:
            changes.append("autopsy_run")

        before = content
        content = add_nan_recovery_snapshot(content, filepath)
        if content != before:
            changes.append("nan_recovery")

        before = content
        content = add_collision_guard_snapshot(content, filepath)
        if content != before:
            changes.append("collision_guard")

        before = content
        content = add_n_features_to_checkpoint_snapshot(content, filepath)
        if content != before:
            changes.append("n_features_ckpt")

    elif trainer_type == "ncl":
        before = content
        content = add_autopsy_setup_ncl(content, filepath)
        if content != before:
            changes.append("autopsy_setup")

        before = content
        content = add_autopsy_run_ncl(content, filepath)
        if content != before:
            changes.append("autopsy_run")

        before = content
        content = add_nan_recovery_ncl(content, filepath)
        if content != before:
            changes.append("nan_recovery")

        before = content
        content = add_collision_guard_ncl(content, filepath)
        if content != before:
            changes.append("collision_guard")

        before = content
        content = add_n_features_to_checkpoint_ncl(content, filepath)
        if content != before:
            changes.append("n_features_ckpt")

    if content != original:
        if not DRY_RUN:
            filepath.write_text(content, encoding="utf-8")
        return True, changes
    return False, []


def main():
    mode = "DRY RUN" if DRY_RUN else "APPLYING"
    print(f"=== Propagate Training Features Phase 2 ({mode}) ===\n")

    total_modified = 0
    change_counts = {}

    for trainer_type in ["snapshot", "ncl", "adapter"]:
        files = find_all_trainer_files(trainer_type)
        print(f"\n--- {trainer_type.upper()} trainers ({len(files)} files) ---")

        for f in sorted(files):
            rel = f.relative_to(PROJECT_ROOT)
            modified, changes = process_file(f, trainer_type)
            if modified:
                total_modified += 1
                modified_files.append(rel)
                for c in changes:
                    change_counts[c] = change_counts.get(c, 0) + 1
                print(f"  [MOD] {rel}  ({', '.join(changes)})")
            else:
                print(f"  [OK]  {rel}")

    print(f"\n=== Summary: {total_modified} files modified ===")
    if change_counts:
        print("  Changes breakdown:")
        for k, v in sorted(change_counts.items()):
            print(f"    {k}: {v} files")
    if DRY_RUN and total_modified > 0:
        print("  Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
