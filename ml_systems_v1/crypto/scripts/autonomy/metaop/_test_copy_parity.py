#!/usr/bin/env python3
"""DRIFT FIREWALL for the two forked metaop engine copies (G-A interim guard, 2026-06-07).

The harness was "separated" by COPYING scripts/autonomy/metaop -> harness/metaop, and the two have since diverged
(this audit had to apply the verifier-guard fix to BOTH by hand). Until they are de-duplicated into one canonical
source (G-A, multi-hour), THIS test is the mechanical guard: it asserts both copies still expose the same
TRUST-CRITICAL surface, so a fix/feature added to one copy but not the other is caught immediately (the #1
silent-failure class: cross-version propagation). It does NOT require identical files -- only that the load-bearing
symbols exist in both. Both copies share the package name `metaop`, so each is introspected in its own subprocess.

Usage: python scripts/autonomy/metaop/_test_copy_parity.py            (compare both copies)
       python scripts/autonomy/metaop/_test_copy_parity.py <pkg_root>  (worker: dump one copy's surface as JSON)
No emoji (Windows cp1252).
"""
import importlib
import inspect
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# KNOWN, TRACKED drift (baseline): symbols a copy is CURRENTLY missing, with the closing item. The firewall WARNS
# loudly about these every run but does NOT fail on them -- it FAILS only on NEW drift (regression guard). Each entry
# must name the tracking item so it cannot be quietly forgotten.
#
# G-A CLOSED (2026-06-07): the two copies are no longer forks -- scripts/autonomy/metaop is now a set of THIN shims
# over the SINGLE canonical engine in harness/metaop (see docs/HARNESS_GAP_CLOSURE_2026_06_07.md G-A). The scripts
# brain shim re-exports OllamaBrain (brain-swap reachable from the live loop), so the only prior known gap is closed
# and KNOWN_DRIFT is now EMPTY: any missing trust-critical symbol is real (NEW) drift and FAILS.
KNOWN_DRIFT: dict = {}

# module -> the symbols that MUST exist in both copies (the trust-critical / load-bearing surface).
REQUIRED = {
    # N3 REPLANNER: _replan_reason/_merge_replan + the replan tuning knobs must exist in BOTH copies, else a copy
    # could silently lose the plan-execute recovery loop (the #1 fragility). The scripts copy re-exports them.
    "graph": ["build", "make_nodes", "_run_verify", "_screen_verify_cmd", "_VERIFY_DENY",
              "_replan_reason", "_merge_replan", "DEFAULT_REPLAN_STALL", "DEFAULT_MAX_REPLANS"],
    "brain": ["make_brain", "Brain", "MockBrain", "OllamaBrain", "CliBrain", "AnthropicBrain", "LiteLLMBrain"],
    "tools": ["Tools", "HARD_DENY"],
    "learnings": ["record", "summary_for_plan"],
    "worker": ["Worker"],
    "experts": ["available"],
    # SOTA layer (U6, 2026-06-08): the evolve / cascade / dspy / champion-install modules are now WIRED into the live
    # loop (champion install seam in manager, cascade backend + mechanical tier in graph dispatch, evolve fitness in
    # evolution_loop). They MUST be reachable from BOTH metaop copies (the scripts copy carries thin re-export shims
    # over the canonical harness engine) so a fix to one copy can't silently skip the other.
    "champion": ["apply_champion", "write_champion", "read_champion", "is_improvement"],
    "evolve": ["evolve", "evolve_planner"],
    "cascade_brain": ["CascadeBrain", "make_cascade"],
    "dspy_planner": ["compile_planner", "install_compiled_planner"],
}

# SIGNATURE parity (2026-06-12): symbol-PRESENCE is not enough -- a WRAPPED function can keep its name but DRIFT its
# signature, so a caller passing a new param the wrapper forgot to forward crashes at call time. That is exactly how
# the shim build() silently dropped `fill_window` and every `run_metaop launch` crashed with
# `TypeError: build() got an unexpected keyword argument 'fill_window'`. This guard catches that class: for each
# WRAPPED trust-critical callable, the shim copy MUST accept every param the canonical accepts, EXCEPT the ones the
# shim deliberately injects (hardcodes) and therefore hides from its own signature -- listed per-callable below.
# Rule: (canonical_params - injected) must be a SUBSET of shim_params; any miss = FAIL.
SIG_PARITY = {
    # the crypto shim build()/make_nodes() hardcode these 8 dependency params, so they're allowed to be absent:
    "graph.build":      {"workspace", "cwd", "persona_dir", "persona_aliases",
                         "framer", "recaller", "recorder", "harvester"},
    "graph.make_nodes": {"workspace", "cwd", "persona_dir", "persona_aliases",
                         "framer", "recaller", "recorder", "harvester"},
    # the brain shim make_brain() injects the crypto domain + cwd, so those two are allowed to be absent:
    "brain.make_brain": {"domain", "cwd"},
}


def _surface(pkg_root):
    """Worker: print {module: [present required symbols]} for the metaop package rooted at pkg_root."""
    sys.path.insert(0, pkg_root)
    out = {}
    for mod, names in REQUIRED.items():
        try:
            m = importlib.import_module(f"metaop.{mod}")
            out[mod] = sorted(n for n in names if hasattr(m, n))
        except Exception as e:
            out[mod] = {"_import_error": f"{type(e).__name__}: {e}"}
    # capture the parameter list of each SIG_PARITY callable (for the signature-drift guard)
    sigs = {}
    for qual in SIG_PARITY:
        mod, fn = qual.split(".")
        try:
            obj = getattr(importlib.import_module(f"metaop.{mod}"), fn, None)
            if callable(obj):
                sigs[qual] = list(inspect.signature(obj).parameters)
        except Exception:
            pass
    out["_sigs"] = sigs
    print(json.dumps(out))


def main():
    if len(sys.argv) > 1:
        _surface(sys.argv[1])
        return 0
    roots = {"scripts": os.path.join(ROOT, "scripts", "autonomy"), "harness": os.path.join(ROOT, "harness")}
    surfaces = {}
    for label, r in roots.items():
        if not os.path.isdir(os.path.join(r, "metaop")):
            print(f"[copy-parity] SKIP {label}: no metaop pkg at {r}")
            continue
        p = subprocess.run([sys.executable, os.path.abspath(__file__), r], capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            surfaces[label] = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            print(f"[copy-parity] FAIL {label}: could not introspect\n{p.stdout}\n{p.stderr}")
            return 1
    if len(surfaces) < 2:
        print("[copy-parity] only one copy present -- nothing to compare (OK)")
        return 0
    fails, known = [], []
    for mod, names in REQUIRED.items():
        for label, surf in surfaces.items():
            got = surf.get(mod)
            if isinstance(got, dict):
                fails.append(f"{label}/{mod}: import error {got.get('_import_error')}")
                continue
            baseline = set(KNOWN_DRIFT.get(f"{label}/{mod}", []))
            for n in names:
                if n in (got or []):
                    continue
                if n in baseline:
                    known.append(f"{label}/{mod}: {n} (KNOWN/tracked -> close in G-A dedup)")
                else:
                    fails.append(f"{label}/{mod}: MISSING trust-critical symbol '{n}' (NEW drift -- fix landed in "
                                 "the other copy but not this one; propagate it)")
    # SIGNATURE-drift guard: the shim must accept every canonical param except the ones it deliberately injects.
    h_sigs = surfaces.get("harness", {}).get("_sigs", {}) if isinstance(surfaces.get("harness"), dict) else {}
    s_sigs = surfaces.get("scripts", {}).get("_sigs", {}) if isinstance(surfaces.get("scripts"), dict) else {}
    for qual, injected in SIG_PARITY.items():
        H, S = set(h_sigs.get(qual, [])), set(s_sigs.get(qual, []))
        if not H or not S:
            continue  # one copy lacks the symbol entirely -> the presence check above already covers it
        missing = (H - set(injected)) - S
        if missing:
            fails.append(f"scripts/{qual}: SIGNATURE DRIFT -- shim is missing param(s) {sorted(missing)} that the "
                         "canonical accepts; a caller passing them crashes at call time (the fill_window-class bug). "
                         "Forward them in the shim wrapper (or add to SIG_PARITY's injected set if intentionally hidden).")
    if known:
        print(f"[copy-parity] KNOWN drift ({len(known)}) -- tracked, not failing:")
        for k in known:
            print("   ~", k)
    if fails:
        print(f"[copy-parity] NEW DRIFT DETECTED ({len(fails)}) -- FAIL:")
        for f in fails:
            print("   -", f)
        return 1
    print("[copy-parity] PASS -- no NEW drift across the two metaop copies (known gaps tracked above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
