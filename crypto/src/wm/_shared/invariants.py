"""WM-layer cross-version invariants — single canonical home.

Per CLAUDE.md "Cross-Version Training Invariants" table + "Code Change
Verification #10 Settings Constant Sync", these constants MUST be
identical across every active WM version. Drift here is the #1 source
of silent training failures.

This module is the SOURCE OF TRUTH. Every `settings.py` should either:

  1) Import directly: `from invariants import CANONICAL_INVARIANTS as I`,
     then `WM_BATCH_SIZE = I["WM_BATCH_SIZE"]` etc.

  2) Hard-code locally + add `assert_canonical(globals())` at module
     bottom to fail-fast on drift.

Either pattern eliminates the bug class where V11 sets `BIN_MIN=-1` but
V13 silently inherits a `-5` default — caught only after a 12-epoch
training run wastes 4 GPU-hours.

Per docstring sections at the bottom: how to retrofit a version + how
this composes with `config/_invariants.yaml` (CDAP commit-gate).
"""
from __future__ import annotations

__contract__ = {
    "kind": "wm_invariants_registry",
    "owner": "wm/_shared",
    "outputs": [],
    "invariants": [
        "single source of truth for cross-version WM constants",
        "no side effects at import time",
        "no torch / numpy dependency (importable from settings)",
        "values match CLAUDE.md `## Cross-Version Training Invariants`",
    ],
}

# ───────────────────────────────────────────────────────────────────
# Canonical scalar constants (CLAUDE.md Cross-Version table)
# ───────────────────────────────────────────────────────────────────

CANONICAL_INVARIANTS: dict[str, object] = {
    # Bucketing / output binning — schema-compat with all WM ckpts
    "BIN_MIN":                  -1.0,
    "BIN_MAX":                   1.0,
    "NUM_BINS":                  255,
    # Training loop
    "WM_STEPS_PER_EPOCH":        2000,
    "DIVERSITY_STEPS_PER_EPOCH": 2000,
    "WM_BATCH_SIZE":             32,
    "DIRECT_RETURN_WEIGHT":      3.0,
    "TWOHOT_FOCAL_GAMMA":        0.0,    # disabled — focal accelerates memorization
    # Target / horizon
    "target_prefix":             "target_return",
    # NOTE: AdamW betas=(0.9, 0.95) is a trainer-side invariant
    # (lives in `train_world_model.py`), enforced by CDAP yaml's
    # `adamw_betas_canonical` regex. Not included here because
    # settings.py modules don't declare optimizer hyperparameters.
}

CANONICAL_LISTS: dict[str, list] = {
    "ACTIVE_HORIZONS": [1, 4, 16, 64],
}

# ───────────────────────────────────────────────────────────────────
# Version-class-differentiated invariants
# (CLAUDE.md notes ATME drop is class-specific, not cohort-wide)
# ───────────────────────────────────────────────────────────────────

ATME_BY_CLASS: dict[str, float] = {
    "RSSM_per_sample":  0.15,   # V1.0/V1.1/V1.4/V1.6 baseline
    "RSSM_batch":       0.40,   # V3 batch-level (legacy — flagged ⚠ in iron-clad audit)
    "JEPA":             0.0,    # V6 has no h_seq to drop
    "iTransformer":     0.15,   # V22/V23/V24/V25
}

# ───────────────────────────────────────────────────────────────────
# Trainer invariants (CLAUDE.md Code Change Verification §11)
# ───────────────────────────────────────────────────────────────────

REQUIRED_TRAINER_PATTERNS: dict[str, str] = {
    "strict_false_on_load":     r"load_state_dict\([^)]*strict\s*=\s*False",
    "shic_decline_persisted":    r"shic_decline_count",
    "n_features_in_ckpt":        r"\bn_features\b",
    "adamw_betas":               r"betas\s*=\s*\(\s*0\.9\s*,\s*0\.95\s*\)",
}

# ───────────────────────────────────────────────────────────────────
# Self-check helpers
# ───────────────────────────────────────────────────────────────────


class InvariantDriftError(AssertionError):
    """Raised when a settings module's constants drift from canonical."""


def assert_canonical(settings_globals: dict, *,
                       skip: set[str] | None = None,
                       version_name: str = "<unknown>") -> None:
    """Verify the calling settings.py module against canonical constants.

    Usage at the bottom of any `settings.py`::

        from invariants import assert_canonical
        assert_canonical(globals(), version_name="v1_1")

    On drift this raises `InvariantDriftError` with a clear message
    listing every offending constant. ``skip`` lets a version exempt
    a specific constant by name (use sparingly — drift is rarely
    legitimate).
    """
    skip = skip or set()
    errors: list[str] = []
    for name, expected in CANONICAL_INVARIANTS.items():
        if name in skip:
            continue
        if name not in settings_globals:
            errors.append(f"  - {name}: MISSING (expected {expected!r})")
            continue
        actual = settings_globals[name]
        if actual != expected:
            errors.append(
                f"  - {name}: drift — expected {expected!r}, got {actual!r}")
    for name, expected in CANONICAL_LISTS.items():
        if name in skip:
            continue
        if name not in settings_globals:
            errors.append(f"  - {name}: MISSING (expected {expected!r})")
            continue
        actual = settings_globals[name]
        if list(actual) != list(expected):
            errors.append(
                f"  - {name}: drift — expected {expected!r}, got {actual!r}")
    if errors:
        msg = (f"\nInvariant drift in {version_name}:\n"
                 + "\n".join(errors)
                 + "\n\nSee src/wm/_shared/invariants.py + "
                   "config/_invariants.yaml.")
        raise InvariantDriftError(msg)


def get_canonical(name: str, default: object = None) -> object:
    """Resolve a canonical constant by name. Falls back to ``default`` if
    not registered — useful for opt-in propagation when retrofitting."""
    if name in CANONICAL_INVARIANTS:
        return CANONICAL_INVARIANTS[name]
    if name in CANONICAL_LISTS:
        return CANONICAL_LISTS[name]
    return default


# ───────────────────────────────────────────────────────────────────
# Retrofit recipe (in this docstring so it travels with the file)
# ───────────────────────────────────────────────────────────────────

__doc_retrofit__ = """\
Recipe for retrofitting a version's settings.py:

    # At the end of settings.py:
    try:
        from invariants import assert_canonical
        assert_canonical(globals(), version_name="v1_1")
    except ImportError:
        # Allowed for stand-alone smoke tests outside the project tree
        pass

Why opt-in via try/except: lets stub / library / archive versions
import settings.py without dragging in the registry. Production
trainers should always have the registry available.

Why globals() not the module: settings.py is `from settings import *`'d
into trainer modules. Checking after-the-import via the module path
needs `importlib` indirection; globals() works at the END of the module
body when all constants are defined.
"""
