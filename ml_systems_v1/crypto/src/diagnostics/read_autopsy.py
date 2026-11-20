"""
Read and display Feature Autopsy JSONL logs in human-readable format.

Auto-discovers autopsy logs across all world model versions (V1-V9).

Usage:
    python src/diagnostics/read_autopsy.py                     # All models
    python src/diagnostics/read_autopsy.py --model v1.0        # Specific model
    python src/diagnostics/read_autopsy.py --model v4           # V4 base
    python src/diagnostics/read_autopsy.py --metric grad       # Filter metric
    python src/diagnostics/read_autopsy.py --epoch 10          # Specific epoch
    python src/diagnostics/read_autopsy.py --compare           # Side-by-side model comparison
    python src/diagnostics/read_autopsy.py path/to/file.jsonl  # Direct file (legacy)
"""
import json
import argparse
from pathlib import Path


# Root logs directory
LOG_ROOT = Path(__file__).resolve().parent.parent.parent / "logs"

# Map model names to (major_version_dir, sub_dir) log paths
# V1 models: logs/v1/v1_0/, logs/v1/v1_1/, etc.
# V2-V9 models: logs/v2/v2/, logs/v3/v3/, etc.
ALL_MODELS = {}

# V1 variants
for minor in range(8):
    ALL_MODELS[f"v1.{minor}"] = ("v1", f"v1_{minor}")

# V2-V9 base models
for major in range(2, 10):
    ALL_MODELS[f"v{major}"] = (f"v{major}", f"v{major}")

# V2-V9 variant models (.1/.2/.3)
for major in range(2, 10):
    for variant in range(1, 4):
        ALL_MODELS[f"v{major}.{variant}"] = (f"v{major}", f"v{major}_{variant}")


def discover_autopsy_logs(model_filter=None):
    """Find all autopsy JSONL files, optionally filtered by model.

    Returns dict: {model_name: [list of (path, records)]}
    """
    found = {}

    if model_filter:
        # Single model
        keys = [model_filter] if model_filter in ALL_MODELS else []
        if not keys:
            print(f"Unknown model: {model_filter}")
            print(f"Available: {', '.join(sorted(ALL_MODELS.keys()))}")
            return {}
    else:
        keys = sorted(ALL_MODELS.keys())

    for model_name in keys:
        major_dir, sub_dir = ALL_MODELS[model_name]
        log_dir = LOG_ROOT / major_dir / sub_dir
        if not log_dir.exists():
            continue

        autopsy_files = sorted(log_dir.glob("*_autopsy_*.jsonl"))
        if not autopsy_files:
            continue

        model_logs = []
        for f in autopsy_files:
            records = load_records(str(f))
            if records:
                model_logs.append((f, records))

        if model_logs:
            found[model_name] = model_logs

    return found


def load_records(path: str):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_header(path, records):
    """Print log file header info."""
    # Extract feature count from filename (e.g., v1_0_f13_autopsy_...)
    stem = path.stem
    feat_str = "?"
    for part in stem.split("_"):
        if part.startswith("f") and part[1:].isdigit():
            feat_str = part[1:]
            break

    print(f"\n{'='*70}")
    print(f"  Log: {path.name}")
    print(f"  Features: {feat_str}, Epochs logged: {[r['epoch'] for r in records]}")
    print(f"  Base dim: {records[0].get('base_dim', '?')}")
    print(f"{'='*70}")


def print_rec_mse_trajectory(records):
    """Show per-feature reconstruction MSE over epochs."""
    print("\n--- Reconstruction MSE (lower = learned temporal pattern) ---")
    print(f"{'Feature':<25}", end="")
    for r in records:
        print(f"  Ep{r['epoch']:>3}", end="")
    print()
    print("-" * (25 + 7 * len(records)))

    if not records or "rec_mse_per_feat" not in records[0]:
        print("  No reconstruction data available.")
        return

    features = list(records[0]["rec_mse_per_feat"].keys())
    for feat in features:
        print(f"{feat:<25}", end="")
        for r in records:
            val = r.get("rec_mse_per_feat", {}).get(feat, -1)
            if val >= 0:
                print(f"  {val:.4f}", end="")
            else:
                print(f"     N/A", end="")
        print()


def print_grad_trajectory(records):
    """Show per-feature gradient magnitude over epochs."""
    print("\n--- Gradient Norm (higher = model relies on this for return prediction) ---")
    print(f"{'Feature':<25}", end="")
    for r in records:
        print(f"  Ep{r['epoch']:>3}", end="")
    print()
    print("-" * (25 + 7 * len(records)))

    if not records or "grad_norm_per_feat" not in records[0]:
        print("  No gradient data available.")
        return

    features = list(records[0]["grad_norm_per_feat"].keys())
    for feat in features:
        print(f"{feat:<25}", end="")
        for r in records:
            val = r.get("grad_norm_per_feat", {}).get(feat, -1)
            if val >= 0:
                print(f"  {val:.4f}", end="")
            else:
                print(f"     N/A", end="")
        print()


def print_ablation(records):
    """Show feature group ablation IC results."""
    ablation_records = [r for r in records if "group_ablation" in r]
    if not ablation_records:
        print("\n--- No ablation data available ---")
        return

    for r in ablation_records:
        abl = r["group_ablation"]
        baseline = abl.get("baseline", {})
        print(f"\n--- Feature Group Ablation (Epoch {r['epoch']}) ---")
        print(f"  Baseline IC_1: {baseline.get('ic_1', 0):.4f}")
        print(f"  {'Group':<18} {'IC_1':>8} {'IC_1 Drop':>10} {'IC Avg Drop':>12}  Features")
        print(f"  {'-'*75}")
        for group_name, group_data in abl.items():
            if group_name == "baseline":
                continue
            print(f"  {group_name:<18} {group_data.get('ic_1', 0):>8.4f} "
                  f"{group_data.get('ic_1_drop', 0):>+10.4f} "
                  f"{group_data.get('ic_avg_drop', 0):>+12.4f}  "
                  f"{', '.join(group_data.get('features_zeroed', []))}")


def print_raw_ic(records):
    """Show raw feature-target IC (baseline importance)."""
    raw_records = [r for r in records if "raw_feature_ic" in r]
    if not raw_records:
        print("\n--- No raw feature IC data available ---")
        return

    r = raw_records[0]
    raw_ic = r["raw_feature_ic"]
    print(f"\n--- Raw Feature -> h=1 Target IC (Epoch {r['epoch']}, no model) ---")
    sorted_feats = sorted(raw_ic.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, ic in sorted_feats:
        bar = "#" * int(abs(ic) * 500)
        sign = "+" if ic >= 0 else "-"
        print(f"  {feat:<25} {sign}{abs(ic):.4f}  {bar}")


def print_epoch_detail(records, epoch):
    """Print all available data for a specific epoch."""
    matches = [r for r in records if r["epoch"] == epoch]
    if not matches:
        print(f"No data for epoch {epoch}")
        return
    r = matches[0]
    print(f"\n--- Full Autopsy: Epoch {epoch} ({r.get('timestamp', 'N/A')}) ---")
    print(json.dumps(r, indent=2))


def print_comparison(all_logs):
    """Side-by-side comparison of latest autopsy across models."""
    print("\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON (latest epoch per model)")
    print("=" * 70)

    # Gather latest record from each model
    summaries = []
    for model_name, logs in sorted(all_logs.items()):
        for path, records in logs:
            latest = records[-1]
            feat_count = latest.get("n_features", "?")
            epoch = latest["epoch"]
            summaries.append((model_name, path.stem, feat_count, epoch, latest))

    if not summaries:
        print("  No autopsy data to compare.")
        return

    # Collect all feature names across models
    all_feats_grad = set()
    all_feats_rec = set()
    for _, _, _, _, rec in summaries:
        all_feats_grad.update(rec.get("grad_norm_per_feat", {}).keys())
        all_feats_rec.update(rec.get("rec_mse_per_feat", {}).keys())

    # Gradient comparison
    if all_feats_grad:
        print(f"\n--- Gradient Norm Comparison ---")
        header = f"{'Feature':<25}"
        for model, stem, fc, ep, _ in summaries:
            label = f"{model}(f{fc},e{ep})"
            header += f"  {label:>16}"
        print(header)
        print("-" * (25 + 18 * len(summaries)))

        for feat in sorted(all_feats_grad):
            row = f"{feat:<25}"
            for _, _, _, _, rec in summaries:
                val = rec.get("grad_norm_per_feat", {}).get(feat, -1)
                if val >= 0:
                    row += f"  {val:>16.4f}"
                else:
                    row += f"  {'N/A':>16}"
            print(row)

    # Reconstruction MSE comparison
    if all_feats_rec:
        print(f"\n--- Reconstruction MSE Comparison ---")
        header = f"{'Feature':<25}"
        for model, stem, fc, ep, _ in summaries:
            label = f"{model}(f{fc},e{ep})"
            header += f"  {label:>16}"
        print(header)
        print("-" * (25 + 18 * len(summaries)))

        for feat in sorted(all_feats_rec):
            row = f"{feat:<25}"
            for _, _, _, _, rec in summaries:
                val = rec.get("rec_mse_per_feat", {}).get(feat, -1)
                if val >= 0:
                    row += f"  {val:>16.4f}"
                else:
                    row += f"  {'N/A':>16}"
            print(row)


def display_model(model_name, logs, metric, epoch):
    """Display autopsy data for a single model."""
    print(f"\n{'#'*70}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"  Log files: {len(logs)}")
    print(f"{'#'*70}")

    for path, records in logs:
        print_header(path, records)

        if epoch is not None:
            print_epoch_detail(records, epoch)
            continue

        if metric in ("rec", "all"):
            print_rec_mse_trajectory(records)
        if metric in ("grad", "all"):
            print_grad_trajectory(records)
        if metric in ("ablation", "all"):
            print_ablation(records)
        if metric in ("raw", "all"):
            print_raw_ic(records)


def main():
    parser = argparse.ArgumentParser(
        description="Feature Autopsy Reader -- diagnose any world model's brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/diagnostics/read_autopsy.py                  # All models with autopsy data
  python src/diagnostics/read_autopsy.py --model v1.0     # Just V1.0
  python src/diagnostics/read_autopsy.py --model v4       # Just V4 base
  python src/diagnostics/read_autopsy.py --model v9.1     # V9 variant 1
  python src/diagnostics/read_autopsy.py --compare        # Side-by-side
  python src/diagnostics/read_autopsy.py --metric grad    # Gradients only
  python src/diagnostics/read_autopsy.py path/to/file.jsonl  # Direct file
        """,
    )
    parser.add_argument("logfile", nargs="?", default=None,
                        help="Direct path to autopsy JSONL (legacy mode)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model version (v1.0-v1.7, v2-v9, v2.1-v9.3)")
    parser.add_argument("--epoch", "-e", type=int, default=None,
                        help="Show detail for specific epoch")
    parser.add_argument("--metric", choices=["rec", "grad", "ablation", "raw", "all"],
                        default="all", help="Which metric to show (default: all)")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Side-by-side comparison across models")
    args = parser.parse_args()

    # Legacy mode: direct file path
    if args.logfile:
        records = load_records(args.logfile)
        if not records:
            print("No records found.")
            return
        print(f"Loaded {len(records)} records from {args.logfile}")
        path = Path(args.logfile)
        print_header(path, records)
        if args.epoch is not None:
            print_epoch_detail(records, args.epoch)
            return
        if args.metric in ("rec", "all"):
            print_rec_mse_trajectory(records)
        if args.metric in ("grad", "all"):
            print_grad_trajectory(records)
        if args.metric in ("ablation", "all"):
            print_ablation(records)
        if args.metric in ("raw", "all"):
            print_raw_ic(records)
        return

    # Auto-discovery mode
    all_logs = discover_autopsy_logs(args.model)

    if not all_logs:
        if args.model:
            print(f"No autopsy logs found for {args.model} in {LOG_ROOT}")
        else:
            print(f"No autopsy logs found in {LOG_ROOT}")
        print("\nAutopsy logs are generated during training (every 5 validation epochs).")
        print("Start a training run to generate autopsy data.")
        return

    print(f"Found autopsy data for: {', '.join(sorted(all_logs.keys()))}")

    if args.compare:
        print_comparison(all_logs)
        return

    for model_name, logs in sorted(all_logs.items()):
        display_model(model_name, logs, args.metric, args.epoch)


if __name__ == "__main__":
    main()
