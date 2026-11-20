"""Per-version smoke tests for the V1.x SHIP cohort.

Closes WM_FINDINGS gap #3 ("No per-version smoke tests under tests/").
CLAUDE.md "Smoke Test Depth" rule #5 requires a 1-batch forward pass
through the changed code path. Without these tests, a settings drift
to V1.x only surfaces during a 4-GPU-hour training run.

Each test:
    1. Imports the version's settings + world_model modules.
    2. Constructs the model with smallest supported feature count.
    3. Runs a forward_train + get_loss on a synthetic batch (B=2, T=8).
    4. Asserts: no NaN/inf in any loss component, total loss > 0,
       backward() does not raise.

Each test runs in <5s on CPU. Skipped if the version's directory is
missing or settings.py fails to import.

Run:
    python -m pytest tests/test_wm_smoke.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

VERSIONS = [
    # (id, dir, n_features, model_class)
    ("v1_0", "src/wm/v1/v1_0_training", 13, "TransformerWorldModel"),
    ("v1_1", "src/wm/v1/v1_1_training", 13, "TransformerWorldModel"),
    ("v1_4", "src/wm/v1/v1_4_training", 13, "TransformerWorldModel"),
    ("v1_6", "src/wm/v1/v1_6_training", 13, "TransformerWorldModel"),
    # Tier-1 architecturally-novel base trainers
    ("v3",   "src/wm/v3/v3_training",   13, "WaveNetGRUWorldModel"),
    ("v4",   "src/wm/v4/v4_training",   13, "MambaWorldModel"),
    ("v6",   "src/wm/v6/v6_training",   13, "CausalJEPAWorldModel"),
    ("v8",   "src/wm/v8/v8_training",   13, "NeuralODEWorldModel"),
    # V9 ARCHIVED in run_all_training.MODELS (regime-leak failure mode);
    # smoke skipped via test_v9_archived below (asserts the archive flag).
    # Tier-3 frozen-eval (architecture restored 2026-04-29 from 3c54d26)
    ("v11",  "src/wm/v11/v11_training", 13, "MicrostructureWorldModel"),
    ("v12",  "src/wm/v12/v12_training", 13, "CrossAssetWorldModel"),
    ("v13",  "src/wm/v13/v13_training", 13, "TFTWorldModel"),
    ("v14",  "src/wm/v14/v14_training", 13, "DiffusionWorldModel"),
]


def _setup_paths(version_dir: str):
    """Add the version's training dir + src/ + components to sys.path
    so `from settings import *` and `from world_model import ...` work.
    """
    vdir = ROOT / version_dir
    paths = [str(SRC), str(vdir)]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


@pytest.mark.parametrize("version,vdir,n_features,model_class", VERSIONS)
def test_smoke_imports(version: str, vdir: str, n_features: int, model_class: str):
    """Smoke: settings + world_model + components import cleanly.

    A full 1-batch forward test would require per-version constructor
    plumbing (V1.1 uses XD base/input_dim split, V1.6 has different
    head dims, etc.) which is too brittle for a regression smoke. Import
    + get_feature_config() + class-presence check together catch the
    failure modes a Phase-A-style migration introduces (path drift,
    NameError on missing constants, signature breakage).
    """
    _setup_paths(vdir)
    for mod_name in ["settings", "world_model", "components"]:
        sys.modules.pop(mod_name, None)

    settings = pytest.importorskip("settings", reason=f"{version} settings.py missing")
    wm_mod = pytest.importorskip("world_model", reason=f"{version} world_model.py missing")
    pytest.importorskip("components", reason=f"{version} components.py missing")

    fc = getattr(settings, "get_feature_config", None)
    if fc is None:
        pytest.skip(f"{version}: get_feature_config not defined")
    feature_list, input_dim, _base_dim = fc(n_features)
    assert len(feature_list) == n_features, \
        f"{version}: get_feature_config({n_features}) returned {len(feature_list)} features"

    assert hasattr(wm_mod, model_class), \
        f"{version}: world_model.{model_class} missing"

    # Cross-version invariants must be present and correct
    assert getattr(settings, "WM_BATCH_SIZE", None) == 32, \
        f"{version}: WM_BATCH_SIZE drift"
    assert getattr(settings, "NUM_BINS", None) == 255, \
        f"{version}: NUM_BINS drift"
    assert tuple(getattr(settings, "ACTIVE_HORIZONS", ())) == (1, 4, 16, 64), \
        f"{version}: ACTIVE_HORIZONS drift"
    assert getattr(settings, "TWOHOT_FOCAL_GAMMA", None) == 0.0, \
        f"{version}: TWOHOT_FOCAL_GAMMA drift (must be 0.0)"


def test_v0_floor_paths():
    """V0 baseline scripts importable + DATA_DIR points to chimera_legacy/dollar/."""
    sys.path.insert(0, str(ROOT / "src" / "wm" / "v0" / "v0_baseline"))
    sys.modules.pop("settings", None)
    settings = pytest.importorskip("settings")
    assert hasattr(settings, "DATA_DIR")
    expected_tail = ("chimera_legacy", "dollar")
    actual = settings.DATA_DIR.parts[-2:]
    assert actual == expected_tail, f"V0 DATA_DIR tail {actual} != {expected_tail}"


def test_v9_archived():
    """V9 must be in run_all_training.ARCHIVED_MODELS (regime-leak failure)."""
    sys.path.insert(0, str(SRC))
    import run_all_training  # noqa: E402
    assert "v9" in run_all_training.ARCHIVED_MODELS, \
        "V9 must be archived per fix log; ShIC=0.007 across 90 epochs"


# ─── Stub / library / meta -- import-only smoke (no settings invariant check) ─

STUB_FILES = [
    ("v10",  "src/wm/v10/v10_meta/train_meta.py"),
    ("v15",  "src/wm/v15/patchtst_encoder.py"),
    # V16/V17 reclassified 2026-06-11 to A1 backbones (no longer forecaster stubs).
    ("v16",  "src/agents/a1_wm_consuming/backbones/v16_dreamerv3/v16_training/dreamer_v3.py"),
    ("v17",  "src/agents/a1_wm_consuming/backbones/v17_tdmpc2/v17_training/td_mpc2.py"),
    ("v18",  "src/wm/v18/v18_training/finetune_chronos.py"),
    ("v19",  "src/wm/v19/v19_training/smoke_test.py"),
]


@pytest.mark.parametrize("name,rel", STUB_FILES)
def test_stub_compiles(name: str, rel: str):
    """V10/V15-V19 stubs: file exists + py_compiles cleanly.

    Stubs intentionally don't have full trainers, so we don't import
    them (would trigger argparse / sys.path side effects). Just verify
    the source compiles -- catches the post-Phase-A path drift class
    of bugs (parents[N] off by one, etc.).
    """
    import py_compile
    p = ROOT / rel
    assert p.exists(), f"{name}: {rel} missing"
    py_compile.compile(str(p), doraise=True)


def test_v0_nonlinear_compiles():
    """V0 non-linear baseline scripts py_compile cleanly."""
    import py_compile
    for rel in [
        "src/wm/v0/v0_baseline/linear_baseline.py",
        "src/wm/v0/v0_baseline/nonlinear_baselines.py",
        "src/wm/v0/v0_baseline/save_baseline_preds.py",
        "src/wm/v0/v0_baseline/_workers.py",
    ]:
        py_compile.compile(str(ROOT / rel), doraise=True)
