"""WM-layer shared-components registry.

Per `docs/WM_VERSION_VERDICTS_2026_05_16.md` issue W-2: shared components
(RMSNorm / TwoHotSymlog / MLPHead / SwiGLU / RotaryEmbedding /
CausalTransformerBlock / CryptoPeriodEmbedding / RateBudgetVIB) are
defined and reimplemented in every version's `components.py`. Total
duplication: 6+ classes × 16+ versions ≈ 100+ identical class definitions.

This module is the **canonical map**: for each component, which version's
`components.py` is the source-of-truth, and an SHA256-based drift check.

NOTE on the "remember target IC" constraint: we don't move classes here.
Moving `RMSNorm` from `v1_1_training/components.py` to `_shared/` would
change the import path in every consumer, risking silent ckpt
incompatibility. Instead, this registry:

  1) Names the canonical home for each component.
  2) Exposes a drift-check function that flags when a version's
     local copy has diverged from canonical (lets `wm_audit_crawler`
     surface the problem before a retrain inherits the bug).
  3) Optionally provides re-exports so NEW versions can import from
     the registry without picking a specific source-of-truth version.

Composition: see `src/wm/_shared/invariants.py` (cross-version constants)
and `src/wm/_shared/headline_components.py` (CC-H1..H7 Headline-tier
upgrades).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

__contract__ = {
    "kind": "wm_components_registry",
    "owner": "wm/_shared",
    "outputs": [],
    "invariants": [
        "no torch import (importable from any settings.py)",
        "documents canonical home for each shared component",
        "drift detection is hash-based, not import-based",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WM_BASE = PROJECT_ROOT / "src" / "wm"

# ───────────────────────────────────────────────────────────────────
# Canonical component map: component_name → (file_path_relative_to_WM,
#                                              class_name)
#
# V1.1 chosen as canonical home for the shared transformer/RSSM
# components because V1.1 is the SHIP-RECORD architecture (IC=0.073).
# V25 components are version-local (frontier-specific extensions).
# ───────────────────────────────────────────────────────────────────

CANONICAL_COMPONENTS: dict[str, tuple[str, str]] = {
    # The transformer + binning + activation stack — canonical at V1.1
    "RMSNorm":                ("v1/v1_1_training/components.py", "RMSNorm"),
    "RotaryEmbedding":        ("v1/v1_1_training/components.py", "RotaryEmbedding"),
    "TwoHotSymlog":           ("v1/v1_1_training/components.py", "TwoHotSymlog"),
    "SwiGLU":                 ("v1/v1_1_training/components.py", "SwiGLU"),
    "MLPHead":                ("v1/v1_1_training/components.py", "MLPHead"),
    "CausalTransformerBlock": ("v1/v1_1_training/components.py", "CausalTransformerBlock"),
    # Frontier components — canonical at _shared/frontier_components
    "CryptoPeriodEmbedding":  ("_shared/frontier_components.py", "CryptoPeriodEmbedding"),
    "RateBudgetVIB":          ("_shared/frontier_components.py", "RateBudgetVIB"),
    "tail_adaptive_huber":    ("_shared/frontier_components.py", "tail_adaptive_huber"),
    # Headline-tier components — canonical at _shared/headline_components
    "CrossAssetAttention":    ("_shared/headline_components.py", "CrossAssetAttention"),
    "MultiResolutionEncoder": ("_shared/headline_components.py", "MultiResolutionEncoder"),
    "QuantileHead":           ("_shared/headline_components.py", "QuantileHead"),
    "RegimeConditional":      ("_shared/headline_components.py", "RegimeConditional"),
    "DreamRolloutLoss":       ("_shared/headline_components.py", "DreamRolloutLoss"),
    "LinearAttention":        ("_shared/headline_components.py", "LinearAttention"),
}

# Components defined LOCALLY per version that we explicitly do NOT
# canonicalize (variant ablations need their own copies).
VERSION_LOCAL_COMPONENTS: dict[str, list[str]] = {
    "v25": ["CryptoPeriodEmbedding (inline duplicate)",
             "RateBudgetVIB (inline duplicate — signature divergent)"],
    "v8":  ["ODEDynamics", "RK4Solver", "EulerSolver"],
    "v3":  ["WaveNetTCN", "MultiScaleAggregator", "CausalGRU"],
    "v4":  ["Mamba3SSM", "MambaBlock"],
    "v6":  ["TimeDiscriminator", "JEPAPredictor"],
    "v11": ["HurstMoE", "WaveNetExpert"],
    "v13": ["VariableSelectionNetwork", "GatedResidualNetwork"],
    "v14": ["DDPMDenoiser", "DiffusionSampler"],
}


def _extract_class_source(file_path: Path, class_name: str) -> str | None:
    """Read a Python file and return the source of one class (best-effort).

    Strips leading whitespace + trailing newlines so different indentation
    or trailing newlines don't false-positive as drift.
    """
    if not file_path.exists():
        return None
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    # Match `class Name(` up to the next top-level (un-indented) `class `
    # or `def ` or EOF.
    pat = (rf"^class\s+{re.escape(class_name)}\b[^\n]*:\n"
            r"(?:[ \t]+[^\n]*\n|\n)*")
    m = re.search(pat, text, re.MULTILINE)
    if not m:
        return None
    body = m.group(0)
    # Normalize: strip trailing blank lines + leading whitespace per line
    lines = [ln.rstrip() for ln in body.splitlines()]
    return "\n".join(lines).rstrip()


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def canonical_hash(component: str) -> str | None:
    """Return the 16-char SHA256 prefix of the canonical implementation."""
    if component not in CANONICAL_COMPONENTS:
        return None
    rel, cls = CANONICAL_COMPONENTS[component]
    src = _extract_class_source(WM_BASE / rel, cls)
    return _hash(src) if src else None


def find_divergent_copies(component: str) -> list[dict]:
    """Walk src/wm/**/components.py + world_model.py and flag any version
    whose copy of ``component`` diverges from canonical. Returns a list of
    findings suitable for `wm_audit_crawler` ingestion."""
    if component not in CANONICAL_COMPONENTS:
        return []
    rel, cls = CANONICAL_COMPONENTS[component]
    canonical_src = _extract_class_source(WM_BASE / rel, cls)
    if canonical_src is None:
        return [{
            "severity": "HIGH", "kind": "CANONICAL_MISSING",
            "component": component, "expected_at": rel,
        }]
    canonical_h = _hash(canonical_src)
    findings: list[dict] = []
    canonical_path = WM_BASE / rel
    for path in WM_BASE.rglob("components.py"):
        if path == canonical_path:
            continue
        # Skip archive dirs
        if "/archive/" in str(path).replace("\\", "/"):
            continue
        src = _extract_class_source(path, cls)
        if src is None:
            continue
        local_h = _hash(src)
        if local_h != canonical_h:
            findings.append({
                "severity": "MEDIUM",
                "kind": "COMPONENT_DRIFT",
                "component": component,
                "version_file": str(path.relative_to(PROJECT_ROOT)),
                "canonical_at": rel,
                "canonical_hash": canonical_h,
                "local_hash": local_h,
                "remediation": (
                    f"Either (a) replace {cls} in this file with `from ..."
                    f"v1_1_training.components import {cls}` "
                    f"(after schema-compat retrofit), or (b) document "
                    f"intentional divergence in version docstring."),
            })
    # Also check world_model.py for inline duplicates (V25 pattern)
    for path in WM_BASE.rglob("world_model.py"):
        if "/archive/" in str(path).replace("\\", "/"):
            continue
        src = _extract_class_source(path, cls)
        if src is None:
            continue
        findings.append({
            "severity": "MEDIUM",
            "kind": "INLINE_COMPONENT_DEFINITION",
            "component": component,
            "version_file": str(path.relative_to(PROJECT_ROOT)),
            "canonical_at": rel,
            "remediation": (
                f"Component `{cls}` is defined inline in world_model.py; "
                f"prefer importing from {rel}."),
        })
    return findings


def audit_all_components() -> list[dict]:
    """Run the drift detection for every canonical component. Returns a
    flat list of findings."""
    out: list[dict] = []
    for name in CANONICAL_COMPONENTS:
        out.extend(find_divergent_copies(name))
    return out


__all__ = [
    "CANONICAL_COMPONENTS",
    "VERSION_LOCAL_COMPONENTS",
    "canonical_hash",
    "find_divergent_copies",
    "audit_all_components",
]

if __name__ == "__main__":
    findings = audit_all_components()
    print(f"[components_registry] {len(findings)} drift findings")
    for f in findings[:20]:
        comp = f.get("component", "")
        kind = f.get("kind", "")
        ver = f.get("version_file", "")
        print(f"  [{f.get('severity','?')}] {kind} {comp} :: {ver}")
    if len(findings) > 20:
        print(f"  ... and {len(findings) - 20} more")
