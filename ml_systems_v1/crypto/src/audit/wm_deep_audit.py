"""wm_deep_audit.py -- standardized deep-audit framework for WM versions.

User mandate 2026-05-16: per-version analysis should follow a consistent
template (first principles → code → execution → gaps → verdict) so each
audit is comparable + reproducible.

This is the structured complement to:
  - wm_audit_crawler (cohort-wide 8-axis crawler; runs nightly)
  - components_registry (drift detection across shared classes)
  - WM_VERSION_VERDICTS_2026_05_16.md (manual kill/keep matrix)

INVOKE
------
    python src/audit/wm_deep_audit.py --version v3
    python src/audit/wm_deep_audit.py --version v3 --include-ablations
    python src/audit/wm_deep_audit.py --all-active
    python src/audit/wm_deep_audit.py --version v4 --out docs/

OUTPUT
------
Markdown report per version under runs/audit/deep/<version>_<DATE>.md
(or docs/ if --out docs/). Sections:
  1. First Principles
  2. Architecture (from world_model.py docstring + class scan)
  3. Code Quality (anti-mem / invariants / trainer patterns)
  4. Saving Mechanism (ckpt schema + strict=False + collision guard)
  5. Speed Profile (param-count + flop-class estimate)
  6. Gaps (Headline plan, components drift, plan coverage)
  7. Verdict (KEEP / FIX / KILL with reasoning)
"""
from __future__ import annotations

__contract__ = {
    "kind": "wm_deep_audit",
    "owner": "audit/wm",
    "outputs": ["runs/audit/deep/<version>_<DATE>.md"],
    "invariants": [
        "non-invasive: reads code + docstrings; never instantiates models",
        "version-name-driven: works on any src/wm/v*/ entry",
        "complements wm_audit_crawler (cohort) with per-version depth",
    ],
}

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WM_BASE = PROJECT_ROOT / "src" / "wm"
DOCS_DIR = PROJECT_ROOT / "docs"
OUT_DEFAULT = PROJECT_ROOT / "runs" / "audit" / "deep"


# ───────────────────────────────────────────────────────────────────
# Read helpers (non-invasive — never imports the version code)
# ───────────────────────────────────────────────────────────────────

def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def _extract_module_docstring(path: Path) -> str:
    """Return the first triple-quoted block at top of the file (best-effort)."""
    src = _read(path)
    m = re.search(r'^"""(.*?)"""', src, re.DOTALL | re.MULTILINE)
    if not m:
        m = re.search(r"^'''(.*?)'''", src, re.DOTALL | re.MULTILINE)
    return (m.group(1).strip() if m else "(no module docstring)")


def _classes(path: Path) -> list[str]:
    src = _read(path)
    return re.findall(r"^class\s+(\w+)", src, re.MULTILINE)


def _consts(path: Path, names: list[str]) -> dict[str, str]:
    src = _read(path)
    out: dict[str, str] = {}
    for n in names:
        m = re.search(rf"^{n}\s*=\s*([^\n#]+)", src, re.MULTILINE)
        if m:
            out[n] = m.group(1).strip().rstrip(",")
    return out


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))


def _has(path: Path, pattern: str) -> bool:
    return bool(re.search(pattern, _read(path)))


# ───────────────────────────────────────────────────────────────────
# Per-version helpers
# ───────────────────────────────────────────────────────────────────

def discover_subversions(top_dir: Path) -> list[dict]:
    """Find all *_training/ subdirs of a top-level vN/ dir, or return
    the top-level itself if it has settings.py directly."""
    subs: list[dict] = []
    if not top_dir.exists():
        return subs
    train_dirs = sorted(top_dir.glob("*_training"))
    if train_dirs:
        for sub in train_dirs:
            name = sub.name.replace("_training", "")
            entry: dict = {"name": name, "dir": sub}
            entry["settings"]    = sub / "settings.py"
            entry["world_model"] = sub / "world_model.py"
            entry["trainer"]     = sub / "train_world_model.py"
            entry["components"]  = sub / "components.py"
            entry["validate"]    = sub / "validate_world.py"
            subs.append(entry)
    else:
        # Top-level only (V0/V10/V15-V17/V21 pattern)
        entry = {"name": top_dir.name, "dir": top_dir}
        # Best-effort find
        for fname, key in (("settings.py", "settings"),
                            ("world_model.py", "world_model"),
                            ("train_world_model.py", "trainer"),
                            ("components.py", "components"),
                            ("validate_world.py", "validate")):
            cand = top_dir / fname
            entry[key] = cand if cand.exists() else None
            if entry[key] is None:
                # Try one level down
                deep = next(top_dir.rglob(fname), None)
                entry[key] = deep
        subs.append(entry)
    return subs


# Anti-memo mechanism signatures (each maps to typical code patterns)
ANTI_MEMO_PATTERNS: dict[str, list[str]] = {
    # Tightened to require strong-signal patterns (avoid false positives
    # like JEPA matching unrelated "EMA_DECAY" or VIB matching variable
    # names mentioning bits).
    "RSSM":          [r"\bRSSM\b", r"prior_logits|posterior_head"],
    "VIB":           [r"\bVIB\b", r"vib_mu|vib_logvar", r"VariationalInformationBottleneck"],
    "ATME":          [r"TEMPORAL_CTX_DROP", r"\bATME\b"],
    "JEPA":          [r"\bJEPA\b", r"joint_embedding", r"target_encoder"],
    "TimeDisc":      [r"TimeDiscriminator", r"adversarial.*disc"],
    "XD-split":      [r"XD_DROPOUT", r"XD_NOISE_STD"],
    "Block-mask":    [r"block_mask\s*=\s*True", r"WM_BLOCK_SIZE_RATIO"],
}

TRAINER_PATTERNS: dict[str, str] = {
    "strict_false_on_load":  r"load_state_dict\([^)]*strict\s*=\s*False",
    "shic_decline_persisted": r"shic_decline_count",
    "n_features_in_ckpt":     r"\bn_features\b",
    "adamw_betas_canonical":  r"betas\s*=\s*\(\s*0\.9\s*,\s*0\.95\s*\)",
    "purge_split_used":       r"split_four_way|get_split_dates|purge_split",
    "ema_decay":              r"EMA_DECAY|ema_decay|ema_model",
    "amp_autocast":           r"autocast|GradScaler|bf16|float16",
}

CANONICAL_INVARIANT_NAMES = [
    "BIN_MIN", "BIN_MAX", "NUM_BINS",
    "WM_BATCH_SIZE", "WM_STEPS_PER_EPOCH", "DIVERSITY_STEPS_PER_EPOCH",
    "DIRECT_RETURN_WEIGHT", "TWOHOT_FOCAL_GAMMA", "ACTIVE_HORIZONS",
    "target_prefix",
]


def estimate_params(settings_path: Path) -> Optional[int]:
    """Static param-count estimate from settings.py constants.
    Returns None if the version doesn't fit the V1.x-style spec.
    Estimate is for the Transformer+RSSM family; coarse but useful."""
    if not settings_path.exists():
        return None
    c = _consts(settings_path, [
        "WM_D_MODEL", "WM_N_HEADS", "WM_N_LAYERS", "WM_D_FF",
        "RSSM_LATENT_DIM", "RSSM_CLASSES", "NUM_BINS", "INPUT_DIM",
    ])
    try:
        d = int(c.get("WM_D_MODEL", "256"))
        n = int(c.get("WM_N_LAYERS", "3"))
        ff = int(c.get("WM_D_FF", str(4 * d)))
        rd = int(c.get("RSSM_LATENT_DIM", "0"))
        rc = int(c.get("RSSM_CLASSES", "0"))
        bins = int(c.get("NUM_BINS", "255"))
        feat = int(c.get("INPUT_DIM", "37"))
    except (ValueError, KeyError):
        return None
    # Attention: 4 * d**2 per layer (qkv + out_proj)
    # FFN: 3 * d * ff per layer (SwiGLU gate + up + down)
    # Encoder: feat * d
    # RSSM heads: d * rd * rc * 2
    # Return heads: 4 horizons * (d * bins * 2)
    p_attn = n * 4 * d * d
    p_ffn = n * 3 * d * ff
    p_enc = feat * d
    p_rssm = 2 * d * rd * rc if rd and rc else 0
    p_heads = 4 * d * bins
    return p_attn + p_ffn + p_enc + p_rssm + p_heads


# ───────────────────────────────────────────────────────────────────
# Section builders
# ───────────────────────────────────────────────────────────────────

def _section_first_principles(sub: dict) -> str:
    """Pull the architecture intent from world_model.py docstring."""
    doc = _extract_module_docstring(sub["world_model"]) if sub.get("world_model") else "(no world_model.py)"
    # Trim to ~30 lines
    lines = doc.splitlines()
    if len(lines) > 30:
        lines = lines[:30] + ["...(truncated)"]
    return "\n".join(lines)


def _section_architecture(sub: dict) -> dict:
    wm = sub.get("world_model")
    components = sub.get("components")
    out: dict = {}
    out["world_model_lines"] = _line_count(wm) if wm else 0
    out["world_model_classes"] = _classes(wm) if wm else []
    out["components_lines"] = _line_count(components) if components else 0
    out["components_classes"] = _classes(components) if components else []
    out["param_estimate"] = estimate_params(sub["settings"]) if sub.get("settings") else None
    return out


def _section_invariants(sub: dict) -> dict:
    s = sub.get("settings")
    if not s or not s.exists():
        return {"_missing": True}
    return _consts(s, CANONICAL_INVARIANT_NAMES)


def _section_anti_memo(sub: dict) -> list[str]:
    scan = ""
    for k in ("world_model", "settings"):
        p = sub.get(k)
        if p and p.exists():
            scan += "\n" + _read(p)
    present: list[str] = []
    for mech, patterns in ANTI_MEMO_PATTERNS.items():
        if any(re.search(pat, scan) for pat in patterns):
            present.append(mech)
    return present


def _section_trainer(sub: dict) -> dict:
    t = sub.get("trainer")
    if not t or not t.exists():
        return {"_missing": True}
    src = _read(t)
    out: dict = {"lines": _line_count(t)}
    for name, pat in TRAINER_PATTERNS.items():
        out[name] = bool(re.search(pat, src))
    return out


def _section_headline_plan(version_name: str) -> dict:
    plan = DOCS_DIR / "WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md"
    if not plan.exists():
        return {"_missing": True}
    text = _read(plan)
    pretty = version_name.replace("v", "V").replace("_", ".")
    plan_present = bool(
        re.search(rf"##\s+\d+\.\s+{re.escape(pretty)}\b", text, re.IGNORECASE)
    )
    return {"plan_section_exists": plan_present, "doc_path": str(plan.relative_to(PROJECT_ROOT))}


def _section_iron_clad(version_name: str) -> str:
    """Pull the iron-clad audit verdict line if present."""
    p = DOCS_DIR / "WM_SOTA_IRON_CLAD_AUDIT_2026_05_07.md"
    if not p.exists():
        return ""
    text = _read(p)
    # Crude row match in the summary matrix table
    pretty = version_name.replace("v", "V").replace("_", ".")
    m = re.search(rf"\|\s*\*?\*?{re.escape(pretty)}\*?\*?\s*\|.*", text)
    return m.group(0).strip() if m else ""


# ───────────────────────────────────────────────────────────────────
# Report renderer
# ───────────────────────────────────────────────────────────────────

def render_report(version_name: str, sub: dict) -> str:
    today = dt.date.today().isoformat()
    first = _section_first_principles(sub)
    arch = _section_architecture(sub)
    inv = _section_invariants(sub)
    anti = _section_anti_memo(sub)
    trn = _section_trainer(sub)
    hp = _section_headline_plan(version_name)
    ic_row = _section_iron_clad(version_name)

    out: list[str] = []
    out.append(f"# WM Deep Audit -- {sub['name']} ({today})\n")
    out.append(f"> Auto-generated by `src/audit/wm_deep_audit.py`. Pairs with the")
    out.append(f"> cohort-level `wm_audit_crawler` + `WM_VERSION_VERDICTS_2026_05_16.md`.")
    out.append("")

    # ── 1. First principles ──
    out.append("## 1. First Principles\n")
    out.append("```")
    out.append(first)
    out.append("```")
    if ic_row:
        out.append(f"\n**Iron-clad audit (2026-05-07) row**: {ic_row[:200]}")
    out.append("")

    # ── 2. Architecture ──
    out.append("## 2. Architecture\n")
    out.append(f"- `world_model.py`: **{arch['world_model_lines']} lines**, "
                 f"{len(arch['world_model_classes'])} classes")
    if arch['world_model_classes']:
        out.append(f"  - Classes: {', '.join(arch['world_model_classes'])}")
    out.append(f"- `components.py`: **{arch['components_lines']} lines**, "
                 f"{len(arch['components_classes'])} classes")
    if arch['components_classes']:
        out.append(f"  - Classes: {', '.join(arch['components_classes'])}")
    if arch['param_estimate']:
        est_m = arch['param_estimate'] / 1e6
        tier = ("UNDERSIZED" if est_m < 4 else
                  "SIZED" if est_m < 10 else
                  "LARGE")
        out.append(f"- Estimated trainable params (static): "
                     f"**~{est_m:.2f}M** [{tier}]")
    else:
        out.append("- Param estimate: n/a (non-V1.x-style architecture)")
    out.append("")

    # ── 3. Cross-version invariants ──
    out.append("## 3. CLAUDE.md Cross-Version Invariants\n")
    if inv.get("_missing"):
        out.append("- No settings.py found.")
    else:
        out.append("| Constant | Value in settings.py | Canonical |")
        out.append("|---|---|---|")
        canonical_map = {
            "BIN_MIN": "-1.0", "BIN_MAX": "1.0", "NUM_BINS": "255",
            "WM_BATCH_SIZE": "32", "WM_STEPS_PER_EPOCH": "2000",
            "DIVERSITY_STEPS_PER_EPOCH": "2000",
            "DIRECT_RETURN_WEIGHT": "3.0", "TWOHOT_FOCAL_GAMMA": "0.0",
            "ACTIVE_HORIZONS": "[1, 4, 16, 64]",
            "target_prefix": "\"target_return\"",
        }
        for k, expected in canonical_map.items():
            got = inv.get(k, "(missing)")
            match = "OK" if got.strip() == expected else "DRIFT" if got != "(missing)" else "MISSING"
            out.append(f"| `{k}` | `{got}` | `{expected}` | {match} |")
    out.append("")

    # ── 4. Anti-memo mechanisms ──
    out.append("## 4. Anti-Memorization Stack\n")
    if anti:
        out.append("Mechanisms detected in world_model.py + settings.py:")
        for m in anti:
            out.append(f"- {m}")
    else:
        out.append("**WARNING**: no anti-memo mechanism detected. Verify by hand.")
    out.append("")

    # ── 5. Saving mechanism / trainer invariants ──
    out.append("## 5. Saving Mechanism & Trainer Invariants\n")
    if trn.get("_missing"):
        out.append("- No `train_world_model.py` found (stub / library version).")
    else:
        out.append(f"- `train_world_model.py`: **{trn['lines']} lines**")
        out.append("| Pattern | Present? |")
        out.append("|---|---|")
        for k in ("strict_false_on_load", "shic_decline_persisted",
                   "n_features_in_ckpt", "adamw_betas_canonical",
                   "purge_split_used", "ema_decay", "amp_autocast"):
            v = trn.get(k, False)
            out.append(f"| `{k}` | {'YES' if v else '**NO**'} |")
    out.append("")

    # ── 6. Headline plan coverage ──
    out.append("## 6. Headline Plan Coverage\n")
    if hp.get("_missing"):
        out.append("- Headline plan doc missing.")
    else:
        if hp.get("plan_section_exists"):
            out.append(f"- Per-version section EXISTS in `{hp['doc_path']}`.")
        else:
            out.append(f"- **GAP**: no per-version section in "
                         f"`{hp['doc_path']}`. Per CLAUDE.md D10 rule, "
                         f"every active version needs a documented "
                         f"Headline-target ladder.")
    out.append("")

    # ── 7. Verdict (auto-template; manual finalization expected) ──
    out.append("## 7. Verdict (auto-skeleton; humanize before shipping)\n")
    out.append("**Architecture viability**:")
    tier = "UNKNOWN"
    if arch.get("param_estimate"):
        em = arch['param_estimate'] / 1e6
        tier = ("UNDERSIZED — bump capacity before training" if em < 4
                 else "SIZED — proceed to training"
                 if em < 10 else "LARGE — verify VRAM budget")
    out.append(f"- Param tier: {tier}")
    if not anti:
        out.append("- **NO ANTI-MEMO** detected — flag in iron-clad audit.")
    if trn.get("_missing"):
        out.append("- No trainer — library / stub status.")
    elif not trn.get("strict_false_on_load"):
        out.append("- **NO strict=False on load** — fix before retrain.")
    elif not trn.get("adamw_betas_canonical"):
        out.append("- AdamW betas drift — fix to (0.9, 0.95).")
    if not hp.get("plan_section_exists", True):
        out.append("- No Headline plan — author one before allocating GPU.")
    out.append("")
    out.append("**Default verdict**: see `WM_VERSION_VERDICTS_2026_05_16.md` "
                 "for the human-curated decision.")
    out.append("")
    out.append("## Cross-references\n")
    out.append("- `runs/audit/wm_audit_2026-05-16.md` (cohort 8-axis crawler)")
    out.append("- `docs/WM_VERSION_VERDICTS_2026_05_16.md` (kill/keep matrix)")
    out.append("- `docs/WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md` (Headline specs)")
    out.append("- `docs/WM_SOTA_IRON_CLAD_AUDIT_2026_05_07.md` (May-7 baseline)")
    out.append("- `src/wm/_shared/invariants.py` (canonical constants)")
    out.append("- `src/wm/_shared/components_registry.py` (drift detector)")
    return "\n".join(out)


def emit_for_version(version: str, out_dir: Path,
                      include_ablations: bool = False) -> list[Path]:
    """Find sub-versions of `version` (e.g. v3 -> v3_1, v3_2, v3_3, v3),
    emit one report per sub-version."""
    top = WM_BASE / version
    if not top.exists():
        print(f"[deep_audit] unknown version: {version}", flush=True)
        return []
    subs = discover_subversions(top)
    today = dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s in subs:
        if not include_ablations and s["name"] != version and "_" in s["name"]:
            # Ablation sub-version, e.g. v3_1; skip unless requested
            continue
        report = render_report(s["name"], s)
        p = out_dir / f"{s['name']}_deep_audit_{today}.md"
        p.write_text(report, encoding="utf-8")
        paths.append(p)
        print(f"[deep_audit] wrote {p.relative_to(PROJECT_ROOT)}")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", help="Top-level version dir name "
                                        "(v0/v1/v3/v4/v6/v8/...)")
    ap.add_argument("--include-ablations", action="store_true",
                    help="Emit reports for ablation sub-versions too "
                         "(e.g. v3_1, v3_2, v3_3 alongside v3 base)")
    ap.add_argument("--all-active", action="store_true",
                    help="Run for every top-level version dir in src/wm/")
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT),
                    help=f"Output dir (default: {OUT_DEFAULT.relative_to(PROJECT_ROOT)})")
    args = ap.parse_args()
    out_dir = Path(args.out) if Path(args.out).is_absolute() else (PROJECT_ROOT / args.out)

    if args.all_active:
        n_total = 0
        for d in sorted(WM_BASE.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            paths = emit_for_version(d.name, out_dir,
                                       include_ablations=args.include_ablations)
            n_total += len(paths)
        print(f"[deep_audit] {n_total} reports written to "
              f"{out_dir.relative_to(PROJECT_ROOT)}")
        return 0

    if not args.version:
        ap.error("Either --version or --all-active is required")
    paths = emit_for_version(args.version, out_dir,
                               include_ablations=args.include_ablations)
    return 0 if paths else 2


if __name__ == "__main__":
    sys.exit(main())
