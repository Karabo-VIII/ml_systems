#!/usr/bin/env python
"""Flag-Variant Batch Runner -- /un Information-First Experimental Design.

Fires the SAME training script for a specified model version multiple times
with different frontier-ML upgrade flags, isolating each run via --run-tag
so checkpoints don't clobber each other. After the batch lands, runs the
analyzer to print a side-by-side ShIC/IC/Gap comparison.

This is the orchestrator companion to:
  - run_all_training.py     -- runs ALL models at a given features count
  - run_version_training.py -- runs all architectural variants of a version
  - run_flag_batch.py       -- runs all FLAG variants of one version (THIS FILE)

Per /un Information-First Experimental Design (CLAUDE.md skill):
    "Design the FIRST experiment to extract maximum information. Run 1
    iteration of N parallel candidates and let the batch tell you which
    paradigm is correct. 1 batch of 3 variants > 3 sequential 1-variant
    runs."

Default flag-variant registry (one per row, all baseline + 6 frontier flags):
  baseline       no flags             V1.x reference
  sam            --sam --sam-rho 0.7   B003 R1 SAMformer-validated
  mtp            --mtp                 B002 R1 sequential causal-chain
  mdn_skewt      --mdn skewed_t        B003 R3 skewed-Student-t mixture
  fraug          --fraug 0.10/0.5      B003 R2 frequency-domain mask
  label_noise    --label-noise 0.5     B007 E2 anti-memorization
  logit_clip     --logit-clip 4.0      B007 §5.2 bounded logit norm

With --include-combos:
  sam_label_noise         SAM + label-noise (anti-mem stack)
  sam_fraug               SAM + FrAug (gradient + augmentation)

Each run produces:
  models/{version}/base/{version}_f{features}_{tag}_wm_best_ema.pt
  logs/{version}/{version}_f{features}_{tag}_train_<ts>.log

Usage:
  python src/run_flag_batch.py --version v1_1 --features 29
  python src/run_flag_batch.py --version v1_1 --features 29 --variants baseline sam mtp
  python src/run_flag_batch.py --version v1_1 --features 29 --include-combos
  python src/run_flag_batch.py --version v1_1 --features 29 --dry-run
  python src/run_flag_batch.py --version v1_1 --features 29 --analyze-only
  python src/run_flag_batch.py --version v1_1 --features 29 --skip-preflight --seed 42
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Version registry: which versions support flag-variant batches
# ---------------------------------------------------------------------------

VERSION_TRAINERS = {
    "v1_0": "src/wm/v1/v1_0_training/train_world_model.py",
    "v1_1": "src/wm/v1/v1_1_training/train_world_model.py",
    "v1_4": "src/wm/v1/v1_4_training/train_world_model.py",
    "v1_6": "src/wm/v1/v1_6_training/train_world_model.py",
    "v3":   "src/wm/v3/v3_training/train_world_model.py",
    "v4":   "src/wm/v4/v4_training/train_world_model.py",
    "v6":   "src/wm/v6/v6_training/train_world_model.py",
    "v8":   "src/wm/v8/v8_training/train_world_model.py",
    "v11":  "src/wm/v11/v11_training/train_world_model.py",
    "v12":  "src/wm/v12/v12_training/train_world_model.py",
    "v13":  "src/wm/v13/v13_training/train_world_model.py",
    "v14":  "src/wm/v14/v14_training/train_world_model.py",
}

# Per-version flag wiring status. Keys NOT listed below are NOT wired to
# behaviorally consume the flag (silent no-op). Update when porting flag
# wiring across versions.
#
# "run_tag":  trainer consumes args.run_tag and isolates ckpt_prefix. Without
#             this, parallel-batch variants will CLOBBER each other.
# "label_noise" / "logit_clip": the loss-path wiring exists (not just argparse).
#
# As of 2026-05-02: V1.1 has full wiring; others have argparse via
# add_upgrade_args() but the loss-path side requires per-version porting.
VERSION_CAPABILITIES = {
    # V1.x cohort: full wiring (run_tag + label_noise loss-path + logit_clip hooks)
    "v1_0": {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    "v1_1": {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    "v1_4": {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    "v1_6": {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    # Cross-architecture: full wiring (V1 standards harmonized 2026-05-03)
    "v3":   {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    "v4":   {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    "v6":   {"run_tag": True,  "label_noise": False, "logit_clip": False},  # no build_upgrade_context
    "v12":  {"run_tag": True,  "label_noise": True,  "logit_clip": True},
    # Argparse-only versions (run_tag isolation works; loss-path wiring is V1.x-grade only)
    "v8":   {"run_tag": True,  "label_noise": False, "logit_clip": False},
    "v11":  {"run_tag": True,  "label_noise": False, "logit_clip": False},
    "v13":  {"run_tag": True,  "label_noise": False, "logit_clip": False},
    "v14":  {"run_tag": True,  "label_noise": False, "logit_clip": False},
}

# ---------------------------------------------------------------------------
# Flag-variant registry
# ---------------------------------------------------------------------------

# Each entry: (run_tag, [extra args])
DEFAULT_VARIANTS = [
    ("baseline",     []),
    ("sam",          ["--sam", "--sam-rho", "0.7"]),
    ("mtp",          ["--mtp"]),
    ("mdn_skewt",    ["--mdn", "--mdn-mode", "skewed_t", "--mdn-components", "3"]),
    ("fraug",        ["--fraug", "--fraug-mask-ratio", "0.10", "--fraug-p", "0.5"]),
    ("label_noise",  ["--label-noise", "--label-noise-ratio", "0.5",
                      "--label-noise-sigma-residual", "0.02"]),
    ("logit_clip",   ["--logit-clip", "--logit-clip-tau", "4.0"]),
]

COMBO_VARIANTS = [
    ("sam_label_noise", ["--sam", "--sam-rho", "0.7",
                         "--label-noise", "--label-noise-ratio", "0.5"]),
    ("sam_fraug",       ["--sam", "--sam-rho", "0.7",
                         "--fraug", "--fraug-mask-ratio", "0.10", "--fraug-p", "0.5"]),
]

# Variants whose ARGS aren't supported by some versions: filter them via preflight.
# (The preflight scans the trainer's --help to confirm each flag is recognized.)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight_trainer(trainer_path: Path) -> str:
    """py_compile + --help dry-run. Returns empty string on PASS."""
    if not trainer_path.exists():
        return f"trainer not found: {trainer_path}"
    try:
        import py_compile
        py_compile.compile(str(trainer_path), doraise=True)
    except Exception as e:
        return f"py_compile failed: {str(e)[:120]}"
    try:
        proc = subprocess.run(
            [sys.executable, str(trainer_path), "--help"],
            capture_output=True, text=True, timeout=60,
            env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        if proc.returncode != 0:
            return f"--help exit {proc.returncode}: {proc.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "trainer --help timed out (likely heavy import)"
    return ""


def variant_supported(trainer_help: str, args: list[str]) -> bool:
    """Check that every long-flag in `args` is mentioned in `trainer_help`."""
    for tok in args:
        if tok.startswith("--"):
            if tok not in trainer_help:
                return False
    return True


# Completion marker emitted by the trainer's final report block.
# Pre-2026-05-02 V1.x trainers all print this at end-of-run (see
# v1_1_training/train_world_model.py:1104). Resume runs that skip the
# final report do NOT emit this; they're treated as INCOMPLETE.
_COMPLETION_MARKER = "TRAINING COMPLETE (Anti-Fragile)"


def _model_family(version: str) -> str:
    """Map version_id to its family directory under models/.

    Layout:  models/{family}/{version}/base/  for V1.x and V3/V4/V6/V8.
             models/{family}/base/             for V11/V12/V13/V14.
    """
    if version.startswith("v1_"):
        return "v1"
    return version


def _resolve_ckpt_dir(version: str) -> Path:
    """Read the version's actual settings.{MODEL_DIR,BASE_MODEL_DIR}.

    V3/V4/V6/V8 use BASE_MODEL_DIR = MODEL_DIR / "base" (nested layout).
    V11/V12/V13/V14 use MODEL_DIR directly = .../{ver}/base.
    V1.x use BASE_MODEL_DIR via the trainers' module.
    Falls back to legacy heuristic only if settings.py import fails.
    """
    candidates = []
    if version.startswith("v1_"):
        candidates.append(f"src/wm/v1/{version}_training")
    else:
        candidates.append(f"src/wm/{version}/{version}_training")
    for rel in candidates:
        sett = PROJECT_ROOT / rel / "settings.py"
        if not sett.exists():
            continue
        # Avoid expensive import; parse with a regex line search.
        text = sett.read_text(errors="replace")
        # Look for BASE_MODEL_DIR first; fall back to MODEL_DIR.
        for key in ("BASE_MODEL_DIR", "MODEL_DIR"):
            for line in text.splitlines():
                if line.lstrip().startswith(f"{key} ") or line.lstrip().startswith(f"{key}\t") or line.lstrip().startswith(f"{key}="):
                    # Crude eval: replace PROJECT_ROOT and join with /
                    expr = line.split("=", 1)[1].strip()
                    if "PROJECT_ROOT" in expr:
                        # E.g. PROJECT_ROOT / "models" / "v12" / "base"
                        # or MODEL_DIR / "base"
                        parts = []
                        for p in expr.split("/"):
                            p = p.strip()
                            if p.startswith('"') or p.startswith("'"):
                                parts.append(p.strip('"').strip("'"))
                            elif p == "PROJECT_ROOT":
                                continue
                        return PROJECT_ROOT.joinpath(*parts)
                    elif expr.startswith("MODEL_DIR"):
                        # BASE_MODEL_DIR = MODEL_DIR / "base"
                        # Recurse: resolve MODEL_DIR first.
                        for line2 in text.splitlines():
                            if line2.lstrip().startswith("MODEL_DIR ") or line2.lstrip().startswith("MODEL_DIR="):
                                expr2 = line2.split("=", 1)[1].strip()
                                parts = []
                                for p in expr2.split("/"):
                                    p = p.strip()
                                    if p.startswith('"') or p.startswith("'"):
                                        parts.append(p.strip('"').strip("'"))
                                    elif p == "PROJECT_ROOT":
                                        continue
                                base = PROJECT_ROOT.joinpath(*parts)
                                return base / "base"
            if False:
                pass
    # Final fallback: V1.x layout
    family = _model_family(version)
    return PROJECT_ROOT / "models" / family / version / "base"


def variant_completion(version: str, features: int, run_tag: str,
                        in_progress_window_sec: int = 300) -> tuple[str, str]:
    """Probe (version, features, run_tag) status.

    Returns (status, reason) where status is one of:
        "complete"       -- best_ema exists AND log shows TRAINING COMPLETE
        "in_progress"    -- _wm_latest.pt mtime within in_progress_window_sec
                            (likely a live run; do NOT requeue)
        "incomplete"     -- partial (no completion marker, no recent activity)
        "missing"        -- no checkpoint, no log -- fresh slot
    """
    family = _model_family(version)
    base_dir = _resolve_ckpt_dir(version)
    feat_tag = f"f{features}"
    suffix = f"_{run_tag}" if run_tag else ""
    ema = base_dir / f"{version}_{feat_tag}{suffix}_wm_best_ema.pt"
    latest = base_dir / f"{version}_{feat_tag}{suffix}_wm_latest.pt"

    log_dir = PROJECT_ROOT / "logs" / family / version
    log_glob = list(log_dir.glob(f"{version}_{feat_tag}{suffix}_train_*.log")) if log_dir.exists() else []
    latest_log = max(log_glob, key=lambda p: p.stat().st_mtime) if log_glob else None

    # In-progress: _wm_latest.pt freshly modified
    import time as _t
    now = _t.time()
    if latest.exists() and (now - latest.stat().st_mtime) < in_progress_window_sec:
        return "in_progress", f"latest.pt updated {int(now - latest.stat().st_mtime)}s ago"

    # Complete: best_ema present + sized + log shows completion marker
    if ema.exists() and ema.stat().st_size >= 1024:
        if latest_log is not None:
            try:
                text = latest_log.read_text(errors="replace")
            except Exception as e:
                return "incomplete", f"log unreadable: {e}"
            if _COMPLETION_MARKER in text:
                return "complete", f"complete (log {latest_log.name})"
            return "incomplete", f"best_ema exists but log lacks '{_COMPLETION_MARKER}'"
        return "incomplete", "best_ema exists but no log found"

    # Otherwise: missing (no ckpt) or partial (latest.pt is stale)
    if latest.exists():
        return "incomplete", f"latest.pt stale ({int(now - latest.stat().st_mtime)}s old); no best_ema"
    return "missing", f"no checkpoint at {ema.relative_to(PROJECT_ROOT)}"


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

def run_variant(
    trainer_path: Path, n_features: int, run_tag: str, extra_args: list[str],
    seed: int | None,
) -> tuple[int, float]:
    """Spawn a training run for one variant. Returns (exit_code, wall_seconds)."""
    cmd = [sys.executable, str(trainer_path),
           "--features", str(n_features), "--run-tag", run_tag]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    cmd += extra_args
    print(f"\n{'=' * 70}")
    print(f"  [variant] {run_tag}  -- {' '.join(cmd[2:])}")
    print(f"{'=' * 70}")
    t0 = time.time()
    proc = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    elapsed = time.time() - t0
    return proc.returncode, elapsed


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

def run_analyzer(version: str, n_features: int) -> int:
    """Invoke compare_v1_variants.py with the version threaded through.

    Post-2026-05-03: analyzer is parametrized on --version, so any of the
    12 active versions (v1_0..v1_6, v3, v4, v6, v8, v11..v14) routes to
    the right logs/{family}/{version}/ directory.
    """
    cmd = [sys.executable, "src/frontier_ml/v1_upgrades/compare_v1_variants.py",
           "--version", version,
           "--features", str(n_features), "--auto"]
    print(f"\n{'=' * 70}\n  [analyze] {' '.join(cmd)}\n{'=' * 70}")
    proc = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Flag-Variant Batch Runner (info-first)")
    p.add_argument("--version", required=True, choices=sorted(VERSION_TRAINERS.keys()),
                   help="Model version to flag-batch (must have train_world_model.py).")
    p.add_argument("--features", type=int, required=True,
                   help="Feature count passed to each variant (must be supported by the trainer).")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variants to run by run_tag. Default: full default set.")
    p.add_argument("--include-combos", action="store_true",
                   help="Add COMBO_VARIANTS to the batch (e.g. sam_label_noise).")
    p.add_argument("--seed", type=int, default=42,
                   help="Per-variant seed for reproducibility (default 42).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands and per-variant flag-support without executing.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip py_compile + --help preflight (NOT RECOMMENDED).")
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip training; just run compare_v1_variants on existing logs.")
    p.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip variants whose best_ema checkpoint exists AND log shows "
                        "'TRAINING COMPLETE'. Default ON. Use --no-skip-completed to "
                        "force-resume incomplete checkpoints (the trainer's _wm_latest.pt "
                        "auto-resume kicks in regardless).")
    p.add_argument("--force", action="store_true",
                   help="Re-run variants even when complete (delete the checkpoint or "
                        "let the trainer resume). Implies --no-skip-completed.")
    args = p.parse_args()

    if args.analyze_only:
        sys.exit(run_analyzer(args.version, args.features))

    trainer_rel = VERSION_TRAINERS[args.version]
    trainer_path = PROJECT_ROOT / trainer_rel
    print(f"[batch] version={args.version}  trainer={trainer_rel}")
    print(f"[batch] features={args.features}  seed={args.seed}")

    # Preflight
    if not args.skip_preflight:
        err = preflight_trainer(trainer_path)
        if err:
            print(f"[batch] PREFLIGHT FAIL: {err}", file=sys.stderr)
            sys.exit(2)
        print(f"[batch] preflight OK")

    # Build the batch
    chosen = list(DEFAULT_VARIANTS)
    if args.include_combos:
        chosen += COMBO_VARIANTS
    if args.variants:
        wanted = set(args.variants)
        chosen = [(t, a) for (t, a) in chosen if t in wanted]
        unknown = wanted - {t for (t, _) in DEFAULT_VARIANTS + COMBO_VARIANTS}
        if unknown:
            print(f"[batch] WARN: unknown variants ignored: {unknown}", file=sys.stderr)

    # Refuse to run on a version that doesn't isolate checkpoints via --run-tag,
    # unless it's a single-variant batch (no clobber risk).
    caps = VERSION_CAPABILITIES.get(args.version, {})
    if not caps.get("run_tag", False) and len(chosen) > 1:
        print(f"[batch] HALT: version {args.version} does NOT consume --run-tag in its "
              f"trainer, so a multi-variant batch would CLOBBER checkpoints. Currently "
              f"only V1.1 has full --run-tag wiring (2026-05-02). Run a single variant "
              f"at a time, or port --run-tag wiring to {args.version} first "
              f"(reference: src/wm/v1/v1_1_training/train_world_model.py uses ckpt_prefix "
              f"with run_tag_str).", file=sys.stderr)
        sys.exit(2)

    # Filter by trainer-flag support (if preflight ran, we have the help text)
    help_text = ""
    if not args.skip_preflight:
        proc = subprocess.run(
            [sys.executable, str(trainer_path), "--help"],
            capture_output=True, text=True, timeout=60,
            env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        help_text = proc.stdout
    supported = []
    skipped = []
    for tag, extras in chosen:
        # Check argparse exposure
        if help_text and not variant_supported(help_text, extras):
            skipped.append((tag, extras, "flag not in trainer --help"))
            continue
        # Also check capability: skip variants whose loss-path is no-op on this version.
        if tag in {"label_noise", "logit_clip"} and not caps.get(tag, False):
            skipped.append((tag, extras, f"loss-path wiring missing on {args.version}"))
            continue
        if tag.startswith("sam_label_noise") and not caps.get("label_noise", False):
            skipped.append((tag, extras, f"label-noise loss-path missing on {args.version}"))
            continue
        supported.append((tag, extras))
    if skipped:
        print(f"[batch] {len(skipped)} variant(s) SKIPPED (capability):")
        for tag, extras, reason in skipped:
            print(f"           {tag:<22s}  ({reason})")

    # Triage variants by status; skip complete + in_progress unless --force.
    skip_completed = args.skip_completed and not args.force
    if skip_completed:
        finished = []     # ALREADY COMPLETE -- do not requeue
        live = []         # IN PROGRESS -- do not stomp
        queued = []       # INCOMPLETE/MISSING -- run (trainer will resume from _wm_latest.pt)
        for tag, extras in supported:
            status, reason = variant_completion(args.version, args.features, tag)
            if status == "complete":
                finished.append((tag, reason))
            elif status == "in_progress":
                live.append((tag, reason))
            else:
                queued.append((tag, extras, status, reason))
        if finished:
            print(f"[batch] {len(finished)} variant(s) COMPLETE (skipped):")
            for tag, reason in finished:
                print(f"           {tag:<22s}  {reason}")
        if live:
            print(f"[batch] {len(live)} variant(s) IN PROGRESS (skipped to avoid stomp):")
            for tag, reason in live:
                print(f"           {tag:<22s}  {reason}")
            print(f"           (use --force to override; checkpoint/path collision possible)")
        if queued:
            print(f"[batch] {len(queued)} variant(s) QUEUED:")
            for tag, _extras, status, reason in queued:
                marker = "RESUME" if status == "incomplete" else "FRESH"
                print(f"           {tag:<22s}  [{marker}]  {reason}")
        supported = [(t, a) for (t, a, _s, _r) in queued]
    elif args.force:
        print(f"[batch] --force set: re-running ALL {len(supported)} variant(s) "
              f"regardless of completion status. Note: trainer's _wm_latest.pt "
              f"auto-resume may still kick in unless you delete it first.")

    if not supported:
        print(f"[batch] no variants to run (all complete or filtered). Use --force to rerun.")
        # Still run the analyzer over existing logs.
        run_analyzer(args.version, args.features)
        sys.exit(0)

    # Print plan
    print(f"\n[batch] {len(supported)} variant(s) planned:")
    for tag, extras in supported:
        print(f"           {tag:<22s} {' '.join(extras)}")

    if args.dry_run:
        print("\n[batch] --dry-run: no execution.")
        sys.exit(0)

    # Run
    results = []
    t_start = time.time()
    for tag, extras in supported:
        rc, sec = run_variant(trainer_path, args.features, tag, extras, args.seed)
        results.append((tag, rc, sec))
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[batch] {tag} {status}  wall={sec/60:.1f}min")

    total_min = (time.time() - t_start) / 60.0
    print(f"\n{'=' * 70}\n  BATCH SUMMARY  total={total_min:.1f}min  {len(results)} runs\n{'=' * 70}")
    for tag, rc, sec in results:
        marker = "OK  " if rc == 0 else f"FAIL"
        print(f"  {marker}  {tag:<22s} {sec/60:6.1f}min  rc={rc}")

    # Analyze
    print()
    run_analyzer(args.version, args.features)


if __name__ == "__main__":
    main()
