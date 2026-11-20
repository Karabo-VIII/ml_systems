#!/usr/bin/env python
"""Apply uniform --meta wiring edits to a V-trainer file.

Idempotent: safe to run twice. Only adds edits if the markers are missing.

Usage:
  python scripts/wire_meta_into_trainer.py src/wm/v23/v23_training/train_world_model.py v23
  python scripts/wire_meta_into_trainer.py src/wm/v13/v13_training/train_world_model.py v13
"""
import sys
from pathlib import Path


def wire(path: Path, version: str) -> bool:
    src = path.read_text(encoding="utf-8")
    if "from meta_runtime import" in src:
        print(f"  {version}: already wired (meta_runtime imported)")
        return False

    # Edit 1: import after data_api
    import_block = (
        "from data_api import load_full_data_for_training as load_full_data  # noqa: E402\n\n"
        "# Round-9: shared meta-learner runtime (opt-in via --meta; no-op when empty)\n"
        "_shared_path = str(Path(__file__).resolve().parent.parent.parent / \"_shared\")\n"
        "if _shared_path not in sys.path:\n"
        "    sys.path.insert(0, _shared_path)\n"
        "from meta_runtime import MetaRuntime, add_meta_args  # noqa: E402"
    )
    src = src.replace(
        "from data_api import load_full_data_for_training as load_full_data  # noqa: E402",
        import_block,
        1,
    )

    # Edit 2: meta_rt setup before shic_tracker
    setup_block = (
        f"    # Round-9: meta-learner runtime (no-op when --meta empty; default)\n"
        f"    meta_rt = MetaRuntime.from_args(args, model, MODEL_DIR,\n"
        f"                                     trunk_dim=RETURN_HEAD_DIM,\n"
        f"                                     device=DEVICE, version=\"{version}\")\n\n"
        "    shic_tracker = ShuffledICTracker(af_config)"
    )
    src = src.replace(
        "    shic_tracker = ShuffledICTracker(af_config)",
        setup_block,
        1,
    )

    # Edit 3: capture base_outputs from get_loss (replace `_` with `base_outputs`)
    src = src.replace(
        "loss, loss_dict, _ = model.get_loss(",
        "loss, loss_dict, base_outputs = model.get_loss(",
        1,
    )

    # Edit 4: add residual call after scheduler.step()
    residual_block = (
        "            update_ema(model, ema_model)\n"
        "            scheduler.step()\n\n"
        "            # Round-9: residual meta-learner step (no-op when --meta empty)\n"
        "            if meta_rt.flags:\n"
        "                meta_rt.train_step_residual(model, base_outputs, targets_gpu, step)"
    )
    src = src.replace(
        "            update_ema(model, ema_model)\n            scheduler.step()",
        residual_block,
        1,
    )

    # Edit 5: save_and_summarize before TRAINING COMPLETE
    save_block = (
        f"    # Round-9: save meta variant checkpoints + summary\n"
        f"    meta_rt.save_and_summarize(version=\"{version}\", n_features=input_dim)\n\n"
        "    print(\"\\n  TRAINING COMPLETE"
    )
    src = src.replace(
        "    print(\"\\n  TRAINING COMPLETE",
        save_block,
        1,
    )

    # Edit 6: add_meta_args(parser) — find the argparse block and append
    # Look for the last add_argument line before parse_args
    argparse_marker = "    parser.add_argument(\"--resume\", action=\"store_true\")"
    if argparse_marker in src:
        src = src.replace(
            argparse_marker,
            argparse_marker + "\n    add_meta_args(parser)",
            1,
        )
    else:
        # Fallback: insert before parse_args
        src = src.replace(
            "    args = parser.parse_args()",
            "    add_meta_args(parser)\n    args = parser.parse_args()",
            1,
        )

    path.write_text(src, encoding="utf-8")
    print(f"  {version}: wired (5 edits applied)")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wire_meta_into_trainer.py PATH VERSION")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Not found: {p}")
        sys.exit(1)
    wire(p, sys.argv[2])
