"""
Add --features flag to V2-V9 base snapshot, NCL, and adapter trainers.

Changes applied per file:
1. Add 'from settings import get_feature_config' import
2. Add '--features' to argparse in __main__
3. Override settings.FEATURE_LIST/INPUT_DIM at runtime via get_feature_config()
4. Pass input_dim= to model constructor
5. Use feature_list in load_full_data calls
6. Update checkpoint collision guard to use n_features

Run: python tools/add_features_flag.py --dry-run  (preview)
      python tools/add_features_flag.py            (apply)
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

ALL_VERSIONS = [2, 3, 4, 5, 6, 7, 8, 9]
DRY_RUN = "--dry-run" in sys.argv
modified_files = []


def get_version_prefix(major: int) -> str:
    """Get version-specific model prefix."""
    return f"v{major}"


def add_features_to_snapshot(filepath: Path, major: int) -> list:
    """Add --features flag to a base snapshot trainer."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    changes = []

    if 'add_argument("--features"' in content or "add_argument(\n        \"--features\"" in content:
        return []  # Already done

    vp = get_version_prefix(major)

    # 1. Add get_feature_config import after 'from settings import *'
    if "from settings import get_feature_config" not in content:
        content = content.replace(
            "from settings import *\n",
            "from settings import *\nfrom settings import get_feature_config\n",
            1,
        )
        changes.append("import_get_feature_config")

    # 2. Add n_features parameter to train_ensemble()
    old_sig = "def train_ensemble(use_revin: bool = False):"
    new_sig = "def train_ensemble(use_revin: bool = False, n_features: int = None):"
    if old_sig in content:
        content = content.replace(old_sig, new_sig, 1)
        changes.append("train_ensemble_param")

    # 3. Add feature config resolution at start of train_ensemble()
    # Insert after af_config = AntifragileConfig()
    config_anchor = "    af_config = AntifragileConfig()\n"
    if config_anchor in content and "get_feature_config" not in content.split("def train_ensemble")[1].split("def ")[0] if "def train_ensemble" in content else True:
        feature_config_block = (
            "\n"
            "    # -- Feature config (runtime override via --features) ---------------------\n"
            "    if n_features is not None:\n"
            "        feature_list, input_dim, base_dim = get_feature_config(n_features)\n"
            "        feat_tag = f\"f{n_features}\"\n"
            "    else:\n"
            "        feature_list, input_dim, base_dim = FEATURE_LIST, INPUT_DIM, INPUT_DIM\n"
            "        n_features = INPUT_DIM\n"
            "        feat_tag = f\"f{INPUT_DIM}\"\n"
            "\n"
        )
        content = content.replace(config_anchor, config_anchor + feature_config_block, 1)
        changes.append("feature_config_block")

    # 4. Replace FEATURE_LIST in load_full_data() call
    old_load = "        DATA_DIR, FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS"
    new_load = "        DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS"
    if old_load in content:
        content = content.replace(old_load, new_load, 1)
        changes.append("load_full_data_feature_list")

    # 5. Update Features print line
    old_feat_print = '    print(f"  Features:     {INPUT_DIM}")'
    new_feat_print = '    print(f"  Features:     {n_features} ({feat_tag})")'
    if old_feat_print in content:
        content = content.replace(old_feat_print, new_feat_print, 1)
        changes.append("features_print")

    # 6. Pass input_dim to model constructor in train_single_seed
    # Find "Model().to(DEVICE)" and add input_dim= parameter
    model_patterns = [
        (r"(\w+WorldModel)\(\)\.to\(DEVICE\)", r"\1(input_dim=input_dim).to(DEVICE)"),
        (r"(\w+WorldModel)\(\)\.to\(DEVICE\)", r"\1(input_dim=input_dim).to(DEVICE)"),
    ]
    for pat, repl in model_patterns:
        content = re.sub(pat, repl, content, count=1)

    # 7. Pass n_features/feature_list/input_dim/base_dim to train_single_seed
    # Add parameters to train_single_seed signature
    old_single_sig_pattern = re.compile(
        r"def train_single_seed\(\s*\n"
        r"(\s+seed_idx: int,\s*\n"
        r"\s+seed: int,\s*\n"
        r"\s+all_segments: list,\s*\n"
        r"\s+train_segments: list,\s*\n"
        r"\s+val_segments: list,\s*\n"
        r"\s+train_loader: DataLoader,\s*\n"
        r"\s+val_loader: DataLoader,\s*\n"
        r"\s+af_config,\s*\n)"
        r"(\s+use_revin: bool = False,\s*\n)"
    )
    match = old_single_sig_pattern.search(content)
    if match and "n_features: int" not in content.split("def train_single_seed")[1].split("def ")[0] if "def train_single_seed" in content else True:
        # Add n_features params after use_revin line
        old_revin_line = match.group(2)
        new_params = (
            old_revin_line.rstrip("\n") + "\n"
            "    n_features: int = None,\n"
            "    feature_list: list = None,\n"
            "    input_dim: int = None,\n"
            "    base_dim: int = None,\n"
        )
        content = content.replace(old_revin_line, new_params, 1)
        changes.append("train_single_seed_params")

    # 8. Add defaults for n_features params at top of train_single_seed
    seed_init_anchor = "    set_seed(seed)\n"
    if seed_init_anchor in content and "feature_list = feature_list or FEATURE_LIST" not in content:
        defaults_block = (
            "    feature_list = feature_list or FEATURE_LIST\n"
            "    input_dim = input_dim or INPUT_DIM\n"
            "    base_dim = base_dim or INPUT_DIM\n"
            "    n_features = n_features or INPUT_DIM\n"
        )
        content = content.replace(seed_init_anchor, seed_init_anchor + defaults_block, 1)
        changes.append("single_seed_defaults")

    # 9. Update autopsy to use resolved feature_list/base_dim
    old_autopsy = "        feature_list=FEATURE_LIST, base_dim=INPUT_DIM,"
    new_autopsy = "        feature_list=feature_list, base_dim=base_dim,"
    if old_autopsy in content:
        content = content.replace(old_autopsy, new_autopsy, 1)
        changes.append("autopsy_feature_list")

    # 10. Update collision guard to use n_features
    old_guard = "            if ckpt_n_feat is not None and ckpt_n_feat != INPUT_DIM:"
    new_guard = "            if ckpt_n_feat is not None and ckpt_n_feat != n_features:"
    if old_guard in content:
        content = content.replace(old_guard, new_guard, 1)
        changes.append("collision_guard_n_features")

    # 11. Update n_features in checkpoint save
    old_ckpt = '"n_features": INPUT_DIM,'
    new_ckpt = '"n_features": n_features,'
    if old_ckpt in content:
        content = content.replace(old_ckpt, new_ckpt, 1)
        changes.append("ckpt_n_features")

    # 12. Pass feature params in train_single_seed() call from train_ensemble
    # Add after use_revin=use_revin in the call
    old_call_line = "            use_revin=use_revin,\n        )"
    new_call_line = (
        "            use_revin=use_revin,\n"
        "            n_features=n_features,\n"
        "            feature_list=feature_list,\n"
        "            input_dim=input_dim,\n"
        "            base_dim=base_dim,\n"
        "        )"
    )
    if old_call_line in content and "n_features=n_features,\n            feature_list" not in content:
        content = content.replace(old_call_line, new_call_line, 1)
        changes.append("pass_features_to_single_seed")

    # 13. Add --features to argparse
    if 'add_argument("--features"' not in content and "add_argument(\n        \"--features\"" not in content:
        # Handle both inline and multi-line --revin arg patterns
        if '    parser.add_argument(\n        "--revin"' in content:
            # Multi-line style (V2, V3)
            features_arg = (
                '    parser.add_argument(\n'
                '        "--features", type=int, choices=[13, 18, 30, 37], default=None,\n'
                '        help="Feature count override: 13/18/30/37 (default: settings.INPUT_DIM)"\n'
                '    )\n'
            )
            content = content.replace(
                '    parser.add_argument(\n        "--revin"',
                features_arg + '    parser.add_argument(\n        "--revin"',
                1,
            )
            changes.append("argparse_features")
        elif '    parser.add_argument("--revin"' in content:
            # Inline style (V5-V9)
            features_arg = (
                '    parser.add_argument("--features", type=int, choices=[13, 18, 30, 37], default=None,\n'
                '                        help="Feature count override: 13/18/30/37 (default: settings.INPUT_DIM)")\n'
            )
            content = content.replace(
                '    parser.add_argument("--revin"',
                features_arg + '    parser.add_argument("--revin"',
                1,
            )
            changes.append("argparse_features")

    # 14. Pass n_features to train_ensemble call
    old_call = "    success = train_ensemble(use_revin=args.revin)"
    new_call = "    success = train_ensemble(use_revin=args.revin, n_features=args.features)"
    if old_call in content:
        content = content.replace(old_call, new_call, 1)
        changes.append("main_call_features")

    if content != original:
        if not DRY_RUN:
            filepath.write_text(content, encoding="utf-8")
        return changes
    return []


def add_features_to_ncl(filepath: Path, major: int) -> list:
    """Add --features flag to a base NCL trainer."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    changes = []

    if ('add_argument("--features"' in content or 'add_argument(\n        "--features"' in content) and "get_feature_config" in content:
        return []

    vp = get_version_prefix(major)

    # 1. Add get_feature_config import
    if "from settings import get_feature_config" not in content:
        content = content.replace(
            "from settings import *\n",
            "from settings import *\nfrom settings import get_feature_config\n",
            1,
        )
        changes.append("import_get_feature_config")

    # 2. Add n_features parameter to train_diversity_model()
    # Pattern varies: some have load_backbone, freeze_backbone, use_revin
    train_fn_pattern = re.compile(
        r"(def train_diversity_model\([^)]*)(use_revin: bool = False)"
    )
    match = train_fn_pattern.search(content)
    if match and "n_features" not in match.group(0):
        old = match.group(0)
        new = match.group(1) + "use_revin: bool = False, n_features: int = None"
        content = content.replace(old, new, 1)
        changes.append("train_diversity_param")

    # 3. Feature config resolution (after af_config or augmentor setup)
    config_anchor = "    af_config = AntifragileConfig()\n"
    if config_anchor in content and "feature_list, input_dim, base_dim = get_feature_config" not in content:
        feature_config_block = (
            "\n"
            "    # -- Feature config (runtime override via --features) ---------------------\n"
            "    if n_features is not None:\n"
            "        feature_list, input_dim, base_dim = get_feature_config(n_features)\n"
            "        feat_tag = f\"f{n_features}\"\n"
            "    else:\n"
            "        feature_list, input_dim, base_dim = FEATURE_LIST, INPUT_DIM, INPUT_DIM\n"
            "        n_features = INPUT_DIM\n"
            "        feat_tag = f\"f{INPUT_DIM}\"\n"
            "\n"
        )
        content = content.replace(config_anchor, config_anchor + feature_config_block, 1)
        changes.append("feature_config_block")

    # 4. Replace FEATURE_LIST in load_full_data()
    old_load = "        DATA_DIR, FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS"
    new_load = "        DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS"
    if old_load in content:
        content = content.replace(old_load, new_load, 1)
        changes.append("load_full_data_feature_list")

    # 5. Pass input_dim to model constructor
    model_patterns = [
        (r"(Diversity\w+Model)\(\s*\)", r"\1(input_dim=input_dim)"),
        (r"(Diversity\w+Model)\((\s*\n\s+)", r"\1(input_dim=input_dim,\2"),
    ]
    for pat, repl in model_patterns:
        new_content = re.sub(pat, repl, content, count=1)
        if new_content != content:
            content = new_content
            changes.append("model_input_dim")
            break

    # 6. Update autopsy
    old_autopsy = "        feature_list=FEATURE_LIST, base_dim=INPUT_DIM,"
    new_autopsy = "        feature_list=feature_list, base_dim=base_dim,"
    if old_autopsy in content:
        content = content.replace(old_autopsy, new_autopsy, 1)
        changes.append("autopsy_feature_list")

    # 7. Update collision guard
    old_guard = "            if ckpt_n_feat is not None and ckpt_n_feat != INPUT_DIM:"
    new_guard = "            if ckpt_n_feat is not None and ckpt_n_feat != n_features:"
    if old_guard in content:
        content = content.replace(old_guard, new_guard, 1)
        changes.append("collision_guard_n_features")

    # 8. Update ckpt save
    old_ckpt = '"n_features": INPUT_DIM,'
    new_ckpt = '"n_features": n_features,'
    if old_ckpt in content:
        content = content.replace(old_ckpt, new_ckpt, 1)
        changes.append("ckpt_n_features")

    # 9. Add --features to argparse
    if 'add_argument("--features"' not in content and "add_argument(\n        \"--features\"" not in content:
        revin_anchor = '        "--revin", action="store_true",'
        if revin_anchor in content:
            features_arg = (
                '    parser.add_argument(\n'
                '        "--features", type=int, choices=[13, 18, 30, 37], default=None,\n'
                '        help="Feature count override: 13/18/30/37 (default: settings.INPUT_DIM)"\n'
                '    )\n'
            )
            content = content.replace(
                '    parser.add_argument(\n        "--revin"',
                features_arg + '    parser.add_argument(\n        "--revin"',
                1,
            )
            changes.append("argparse_features")

    # 10. Pass n_features in main call
    # Pattern varies: train_diversity_model(..., use_revin=args.revin, ...)
    old_call_pattern = re.compile(r"(use_revin=args\.revin),?\s*\n(\s+\))")
    match = old_call_pattern.search(content)
    if match and "n_features=args.features" not in content:
        old = match.group(0)
        new = match.group(1) + ",\n        n_features=args.features,\n" + match.group(2)
        content = content.replace(old, new, 1)
        changes.append("main_call_features")

    if content != original:
        if not DRY_RUN:
            filepath.write_text(content, encoding="utf-8")
        return changes
    return []


def add_features_to_adapter(filepath: Path, major: int) -> list:
    """Add --features flag to a base adapter trainer.

    Adapter trainers use main() not train_ensemble(), and load a frozen base model.
    The --features flag must match the base model's feature count.
    """
    content = filepath.read_text(encoding="utf-8")
    original = content
    changes = []

    if ('add_argument("--features"' in content or 'add_argument(\n        "--features"' in content) and "get_feature_config" in content:
        return []

    # 1. Add get_feature_config import
    if "from settings import get_feature_config" not in content:
        content = content.replace(
            "from settings import *\n",
            "from settings import *\nfrom settings import get_feature_config\n",
            1,
        )
        changes.append("import_get_feature_config")

    # 2. Add --features to argparse (find --revin as anchor)
    if 'add_argument("--features"' not in content:
        # Some adapters use parser.add_argument("--revin", others use indented form
        if '    parser.add_argument("--revin"' in content:
            features_arg = (
                '    parser.add_argument("--features", type=int, choices=[13, 18, 30, 37], default=None,\n'
                '                        help="Feature count override (must match base model)")\n'
            )
            content = content.replace(
                '    parser.add_argument("--revin"',
                features_arg + '    parser.add_argument("--revin"',
                1,
            )
            changes.append("argparse_features")

    # 3. Add feature config resolution after argparse
    # Insert after "use_revin = args.revin"
    revin_assign = "    use_revin = args.revin\n"
    if revin_assign in content and "get_feature_config" not in content.split("def main")[1] if "def main" in content else True:
        feature_config_block = (
            "\n"
            "    # -- Feature config (runtime override via --features) ---------------------\n"
            "    n_features = getattr(args, 'features', None)\n"
            "    if n_features is not None:\n"
            "        feature_list, input_dim, base_dim = get_feature_config(n_features)\n"
            "    else:\n"
            "        feature_list, input_dim, base_dim = FEATURE_LIST, INPUT_DIM, INPUT_DIM\n"
            "        n_features = INPUT_DIM\n"
            "\n"
        )
        content = content.replace(revin_assign, revin_assign + feature_config_block, 1)
        changes.append("feature_config_block")

    # 4. Replace FEATURE_LIST in load_full_data()
    # Adapters use various indentation patterns
    if "FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS" in content and "feature_list, ASSET_TO_IDX" not in content:
        content = content.replace(
            "FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS",
            "feature_list, ASSET_TO_IDX, REWARD_HORIZONS",
            1,
        )
        changes.append("load_full_data_feature_list")

    # 5. Pass input_dim to base model constructor
    model_patterns = [
        (r"(\w+WorldModel)\(\)\.to\(", r"\1(input_dim=input_dim).to("),
    ]
    for pat, repl in model_patterns:
        new_content = re.sub(pat, repl, content, count=1)
        if new_content != content:
            content = new_content
            changes.append("model_input_dim")
            break

    if content != original:
        if not DRY_RUN:
            filepath.write_text(content, encoding="utf-8")
        return changes
    return []


def main():
    mode = "DRY RUN" if DRY_RUN else "APPLYING"
    print(f"=== Add --features Flag to V2-V9 Base Trainers ({mode}) ===\n")

    total_modified = 0

    for trainer_type, add_fn in [
        ("snapshot", add_features_to_snapshot),
        ("ncl", add_features_to_ncl),
        ("adapter", add_features_to_adapter),
    ]:
        print(f"\n--- {trainer_type.upper()} base trainers ---")
        for major in ALL_VERSIONS:
            base_dir = SRC / f"v{major}" / f"v{major}_training"
            filepath = base_dir / f"train_{trainer_type}.py"
            if not filepath.exists():
                continue

            rel = filepath.relative_to(PROJECT_ROOT)
            changes = add_fn(filepath, major)
            if changes:
                total_modified += 1
                modified_files.append(rel)
                print(f"  [MOD] {rel}  ({', '.join(changes)})")
            else:
                print(f"  [OK]  {rel}")

    print(f"\n=== Summary: {total_modified} files modified ===")
    if DRY_RUN and total_modified > 0:
        print("  Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
