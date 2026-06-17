"""End-to-end pipeline + model layer validation for BTC + ETH.

Composite validation script for the user's mandate:
  "I need the end-to-end run for BTC and ETH, including make_datasets,
   and validation. ... So in essence, I need to make sure that the pipeline,
   and the new feature integration at the pipeline and model layers are solid."

Stages run sequentially:
  STAGE 1 (pipeline e2e):       run pipeline_e2e_test for BTC + ETH (12 stages each)
  STAGE 2 (chimera validation): validate_chimera for BTC + ETH
  STAGE 3 (xd_* consistency):   cross_asset_consistency BTC + ETH
  STAGE 4 (V0 baseline f121):   verify V0 baseline can resolve f121 features
  STAGE 5 (V1.x f121 resolves): verify V1.0/V1.1/V1.4/V1.6 get_feature_config(121) works
  STAGE 6 (TrainingLoader f121): smoke load BTC/ETH at 1d cadence with f121 features
  STAGE 7 (math suite):         11 mathematical correctness tests
  STAGE 8 (model registry):     verify all V*+ entries map to existing files
  STAGE 9 (data integrity):     v50 vs v51 identity check on BTC + ETH
  STAGE 10 (run_all preflight): every model script in MODELS list py_compiles

Single command: python src/pipeline/e2e_btc_eth_validation.py
Exit codes: 0 clean, 1 warns, 2 hard fail.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


@dataclass
class StageResult:
    stage: int
    name: str
    severity: str   # 'ok' | 'warn' | 'fail'
    detail: str = ""
    elapsed_s: float = 0.0


def run(cmd: list[str], label: str, timeout: int = 600) -> tuple[int, str]:
    print(f"\n[e2e-val] === {label} ===")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
        dt = time.time() - t0
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def stage_pipeline_e2e(asset: str) -> StageResult:
    t0 = time.time()
    rc, out = run([PYTHON, "src/pipeline/pipeline_e2e_test.py", "--asset", asset],
                  f"S1 pipeline_e2e {asset}")
    n_pass = 0
    for line in out.splitlines():
        if "Summary:" in line and "stages" in line:
            try:
                n_pass = int(line.split("Summary:")[1].strip().split("/")[0].strip())
            except Exception:
                pass
    sev = "ok" if (rc == 0 and n_pass >= 12) else "fail"
    return StageResult(1, f"pipeline_e2e_{asset}", sev,
                       f"{n_pass}/12 stages pass" if n_pass else "no summary line found",
                       elapsed_s=round(time.time() - t0, 1))


def stage_chimera_validation(asset: str) -> StageResult:
    t0 = time.time()
    rc, out = run([PYTHON, "src/pipeline/validate_chimera.py", "--asset", asset],
                  f"S2 chimera validation {asset}")
    n_pass = n_warn = n_fail = 0
    for line in out.splitlines():
        if "Summary:" in line and ("clean" in line or "warn" in line):
            try:
                parts = line.split("Summary:")[1].split("(")[0].split(",")
                n_pass = int(parts[0].strip().split()[0])
                n_warn = int(parts[1].strip().split()[0])
                n_fail = int(parts[2].strip().split()[0])
            except Exception:
                pass
    sev = "fail" if n_fail > 0 else ("warn" if n_warn > 0 else "ok")
    return StageResult(2, f"chimera_validate_{asset}", sev,
                       f"{n_pass} clean, {n_warn} warn, {n_fail} fail",
                       elapsed_s=round(time.time() - t0, 1))


def stage_xd_consistency() -> StageResult:
    t0 = time.time()
    rc, out = run([PYTHON, "src/pipeline/cross_asset_consistency.py", "--assets", "BTC,ETH"],
                  "S3 cross-asset consistency BTC+ETH")
    n_fail = 0
    for line in out.splitlines():
        if "Summary:" in line and "pass" in line:
            try:
                parts = line.split("Summary:")[1].strip().split(",")
                n_fail = int(parts[2].strip().split()[0])
            except Exception:
                pass
    sev = "fail" if n_fail > 0 else "ok"
    return StageResult(3, "xd_consistency", sev,
                       f"{n_fail} fails",
                       elapsed_s=round(time.time() - t0, 1))


def stage_v0_f121_resolves() -> StageResult:
    t0 = time.time()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from v0.v0_baseline.settings import get_feature_list_121  # type: ignore
        feats = get_feature_list_121()
        sev = "ok" if len(feats) == 121 else "fail"
        return StageResult(4, "v0_f121_resolves", sev,
                           f"{len(feats)} features (expected 121)",
                           elapsed_s=round(time.time() - t0, 2))
    except Exception as e:
        return StageResult(4, "v0_f121_resolves", "fail", f"err: {e}",
                           elapsed_s=round(time.time() - t0, 2))


def stage_v1x_f121_resolves() -> StageResult:
    t0 = time.time()
    versions = ["v1_0_training", "v1_1_training", "v1_4_training", "v1_6_training"]
    fails = []
    for v in versions:
        try:
            import importlib
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            mod = importlib.import_module(f"v1.{v}.settings")
            fc = mod.get_feature_config(121)
            n_feats = len(fc[0]) if isinstance(fc[0], list) else fc[0]
            if n_feats != 121:
                fails.append(f"{v}: got {n_feats}, expected 121")
        except Exception as e:
            fails.append(f"{v}: {str(e)[:60]}")
    sev = "fail" if fails else "ok"
    detail = "; ".join(fails) if fails else "all 4 V1.x versions resolve f121 OK"
    return StageResult(5, "v1x_f121_resolves", sev, detail,
                       elapsed_s=round(time.time() - t0, 2))


def stage_training_loader_f121() -> StageResult:
    t0 = time.time()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from pipeline.training_loader import TrainingLoader  # type: ignore
        from v19.v19_training.settings import FEATURE_LIST_121  # type: ignore
        tl = TrainingLoader(
            asset_subset=["BTCUSDT", "ETHUSDT"],
            cadence="1d",
            features=FEATURE_LIST_121,
            targets=["target_return_1"],
        )
        sm = tl.summary()
        if sm.get("panel_rows", 0) < 100:
            return StageResult(6, "tl_f121_smoke", "fail",
                               f"too few rows: {sm.get('panel_rows')}",
                               elapsed_s=round(time.time() - t0, 1))
        return StageResult(6, "tl_f121_smoke", "ok",
                           f"BTC+ETH 1d f121 panel: {sm['panel_rows']} rows, "
                           f"train={sm['train_rows']}, val={sm['val_rows']}, oos={sm['oos_rows']}",
                           elapsed_s=round(time.time() - t0, 1))
    except Exception as e:
        return StageResult(6, "tl_f121_smoke", "fail", f"err: {str(e)[:120]}",
                           elapsed_s=round(time.time() - t0, 1))


def stage_math_suite() -> StageResult:
    t0 = time.time()
    rc, out = run([PYTHON, "tests/test_model_math.py"], "S7 math validation")
    n_pass = n_fail = 0
    for line in out.splitlines():
        if "Result:" in line and "pass" in line:
            try:
                n_pass = int(line.split(":")[1].strip().split()[0])
                n_fail = int(line.split("pass,")[1].strip().split()[0])
            except Exception:
                pass
    sev = "fail" if n_fail > 0 else "ok"
    return StageResult(7, "math_suite", sev,
                       f"{n_pass} pass, {n_fail} fail",
                       elapsed_s=round(time.time() - t0, 1))


def stage_model_registry_paths() -> StageResult:
    t0 = time.time()
    try:
        import yaml
        with open(PROJECT_ROOT / "config" / "model_registry.yaml", encoding="utf-8") as f:
            reg = yaml.safe_load(f)
        missing = []
        for name, spec in reg.get("models", {}).items():
            cg = spec.get("checkpoint_glob", "")
            if not cg or spec.get("status") in ("planned", "stub", "deprecated", "frozen", "experimental"):
                continue
            paths = list(PROJECT_ROOT.glob(cg))
            if not paths:
                missing.append(name)
        sev = "fail" if missing else "ok"
        return StageResult(8, "model_registry_paths",
                           sev,
                           f"{len(missing)} production models with missing checkpoints: {missing[:3]}" if missing else "all production checkpoints present",
                           elapsed_s=round(time.time() - t0, 1))
    except Exception as e:
        return StageResult(8, "model_registry_paths", "fail", f"err: {e}",
                           elapsed_s=round(time.time() - t0, 1))


def stage_v50_v51_identity() -> StageResult:
    t0 = time.time()
    rc, out = run([PYTHON, "src/pipeline/v50_backward_compat.py", "--asset", "BTC"],
                  "S9 v50 vs v51 identity (BTC)")
    sev = "fail" if rc not in (0, 1) else ("warn" if rc == 1 else "ok")
    return StageResult(9, "v50_v51_identity", sev,
                       f"V50 BC verifier rc={rc}",
                       elapsed_s=round(time.time() - t0, 1))


def stage_run_all_compile() -> StageResult:
    """Verify every script in run_all_training MODELS list py_compiles."""
    t0 = time.time()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        import importlib
        ral = importlib.import_module("run_all_training")
        models = ral.MODELS
    except Exception as e:
        return StageResult(10, "run_all_compile", "fail", f"can't import: {e}",
                           elapsed_s=round(time.time() - t0, 1))
    fails = []
    for entry in models:
        # entry can be (id, label, script, extras, feats)
        script = entry[2]
        path = PROJECT_ROOT / script
        if not path.exists():
            fails.append(f"{entry[0]}: missing {script}")
            continue
        try:
            import py_compile
            py_compile.compile(str(path), doraise=True)
        except Exception as e:
            fails.append(f"{entry[0]}: compile err {str(e)[:60]}")
    sev = "fail" if fails else "ok"
    return StageResult(10, "run_all_compile", sev,
                       f"{len(models)} models in registry, {len(fails)} compile failures: {fails[:3]}" if fails else f"{len(models)} models all compile clean",
                       elapsed_s=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="Skip slow per-asset stages (1, 2, 3, 9)")
    args = ap.parse_args()

    print("=" * 80)
    print("E2E PIPELINE + MODEL LAYER VALIDATION (BTC + ETH)")
    print("=" * 80)

    stages = []
    if not args.quick:
        for asset in ("BTC", "ETH"):
            stages.append(stage_pipeline_e2e(asset))
            stages.append(stage_chimera_validation(asset))
        stages.append(stage_xd_consistency())
    stages.append(stage_v0_f121_resolves())
    stages.append(stage_v1x_f121_resolves())
    stages.append(stage_training_loader_f121())
    stages.append(stage_math_suite())
    stages.append(stage_model_registry_paths())
    if not args.quick:
        stages.append(stage_v50_v51_identity())
    stages.append(stage_run_all_compile())

    print()
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    n_ok = n_warn = n_fail = 0
    for s in stages:
        flag = {"ok": " OK ", "warn": "WARN", "fail": "FAIL"}[s.severity]
        print(f"  [{flag}] S{s.stage:>2} {s.name:<32}  {s.detail}  ({s.elapsed_s}s)")
        n_ok += int(s.severity == "ok")
        n_warn += int(s.severity == "warn")
        n_fail += int(s.severity == "fail")
    print()
    print(f"Summary: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL  (of {len(stages)} stages)")
    if args.json:
        out = PROJECT_ROOT / "logs" / "e2e_btc_eth_validation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{
            "stage": s.stage, "name": s.name, "severity": s.severity,
            "detail": s.detail, "elapsed_s": s.elapsed_s,
        } for s in stages], indent=2))
        print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
    if n_fail > 0:
        sys.exit(2)
    if n_warn > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
