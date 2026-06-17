"""src/framework/discovery_contract.py -- the DISCOVERY PREFLIGHT CONTRACT (stage-03 gate).

WHY (user mandate 2026-06-11): *"When we are doing a discovery process, I don't want models missing
such things as timeframes, or there being gaps such as strat-based exit vs mechanical exits. These
should be aspects we have moved from and are no longer busy with. The intelligence part is what
should be our focus."* + *"there was a config decomposition process that ... eliminated such
candidates as MA(28,29) given they are practically similar."*

THE GAP THIS CLOSES. Two pieces of the discovery substrate were ad-hoc / manual, so the agent kept
having to re-remember them by hand:
  1. DIMENSIONAL COVERAGE -- a discovery run could silently test only 4h and never sweep timeframes,
     or test only MECHANICAL exits (trailing/time-stop) and never STRAT exits (MA-cross/managed-RSI).
     The dimensions are catalogued (config/strategy_dimension_registry.yaml + ti_master_catalog.yaml
     + feature_registry.yaml = the stage-00 lattice), but nothing CONSUMED them to flag omissions.
  2. CONFIG CANONICALIZATION -- near-identical configs (MA(28,29) ~= MA(27,30)) inflate the search
     space + multiple-comparisons. Handled before only by hand-curated sparse grids (ratio >=2x), a
     design choice baked into constants -- never a reusable, enforced step.

This module formalizes BOTH as mechanical, reusable, single-source-of-truth-driven guarantees, so the
INTELLIGENCE is freed to hunt the edge instead of babysitting coverage/dedup.

DESIGN PRINCIPLE (important): the coverage gate does NOT force exhaustive testing. It forces CONSCIOUS
DECLARATION. A run declares which axis-members it covers and explicitly WAIVES the rest; a SILENT
omission (neither covered nor waived) is a WARN; an undeclared run is a FAIL. So "we have moved from
this" = you waive it on purpose, and the harness guarantees nothing is forgotten by accident.

Where it sits in the formalized harness: it is the STAGE-03 (strat build) PREFLIGHT -- it reads the
STAGE-00 (research/decomposition) registries and runs before candidate_gate, so every discovery
campaign is dimensionally complete + search-space-canonical by construction.

RWYB:
    python src/framework/discovery_contract.py --selftest         # no external data; exits 0 on pass
    python src/framework/discovery_contract.py --demo             # show coverage on a 4h-only/mech-only run
    python src/framework/discovery_contract.py --canon "28,29 28,30 29,30 50,200"   # dedup a grid
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a hard dep elsewhere
    yaml = None

ROOT = Path(__file__).resolve().parents[2]
REG_STRAT = ROOT / "config" / "strategy_dimension_registry.yaml"
REG_TI = ROOT / "config" / "ti_master_catalog.yaml"
REG_FEAT = ROOT / "config" / "feature_registry.yaml"


# ----------------------------------------------------------------------------------------------
# PART 1 -- config canonicalization (near-duplicate elimination)
# ----------------------------------------------------------------------------------------------
def near_dup(c1, c2, rel_tol: float = 0.15) -> bool:
    """True iff two configs are PRACTICALLY THE SAME: every paired param within rel_tol relative
    difference. rel diff = |a-b| / max(|a|,|b|). MA(28,29) vs MA(28,30): 0 and 0.033 -> near-dup;
    MA(28,29) vs MA(50,200): 0.44 and 0.855 -> distinct. Configs must be same length."""
    if len(c1) != len(c2):
        return False
    for a, b in zip(c1, c2):
        m = max(abs(a), abs(b))
        if m == 0:
            continue  # both zero -> identical on this param
        if abs(a - b) / m > rel_tol:
            return False
    return True


@dataclass
class CanonResult:
    rel_tol: float
    n_raw: int
    n_effective: int
    n_collapsed: int
    representatives: list  # the deduped grid (list of tuples)
    collapse_map: dict     # repr(rep) -> [members it absorbed]

    def summary(self) -> str:
        pct = (100.0 * self.n_collapsed / self.n_raw) if self.n_raw else 0.0
        return (f"canonicalize_grid(rel_tol={self.rel_tol}): {self.n_raw} raw -> "
                f"{self.n_effective} effective ({self.n_collapsed} near-dups collapsed, {pct:.0f}%). "
                f"Use n_effective={self.n_effective} for honest multiple-comparison accounting.")


def canonicalize_grid(configs, rel_tol: float = 0.15) -> CanonResult:
    """Collapse practically-identical configs to mutually-separated REPRESENTATIVES.

    Greedy over a deterministic sort: each config joins the first existing representative it is a
    near-dup of; otherwise it becomes a new representative. By construction NO TWO REPRESENTATIVES are
    near-dups of each other -> the surviving grid has no two practically-similar configs, and
    n_effective is the honest size of the search space (the denominator for multiple-comparison
    correction). This is the formalization of the old hand-curated 'sparse grid (ratio>=2x)' trick.

    configs: iterable of tuples/lists of numbers (e.g. MA pairs (fast, slow)). Deterministic output.
    """
    norm = [tuple(float(x) for x in c) for c in configs]
    seen, uniq = set(), []
    for c in norm:  # drop exact dups first (deterministic)
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    uniq.sort()
    reps: list = []
    collapse_map: dict = {}
    for c in uniq:
        hit = None
        for r in reps:
            if near_dup(c, r, rel_tol):
                hit = r
                break
        if hit is None:
            reps.append(c)
            collapse_map[repr(c)] = [c]
        else:
            collapse_map[repr(hit)].append(c)
    n_raw = len(uniq)
    n_eff = len(reps)
    return CanonResult(rel_tol=rel_tol, n_raw=n_raw, n_effective=n_eff,
                       n_collapsed=n_raw - n_eff, representatives=reps, collapse_map=collapse_map)


# ----------------------------------------------------------------------------------------------
# PART 2 -- dimensional coverage (never silently miss a timeframe / an exit family)
# ----------------------------------------------------------------------------------------------
def _load_yaml(p: Path):
    if yaml is None or not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def registry_axes(root: Path = ROOT) -> dict:
    """The canonical axis -> [members] map, read from the stage-00 registries (single source of
    truth). Returns the members the discovery process is EXPECTED to be aware of per axis."""
    strat = _load_yaml(root / "config" / "strategy_dimension_registry.yaml") or {}
    ti = _load_yaml(root / "config" / "ti_master_catalog.yaml") or {}
    feat = _load_yaml(root / "config" / "feature_registry.yaml") or {}

    def names(key):
        return [e.get("name") for e in strat.get(key, []) if isinstance(e, dict) and e.get("name")]

    axes = {
        "cadence": names("axis2_cadence"),
        "chart_type": names("axis1_chart_bartype"),
        "instrument": names("axis3_instrument"),
        "regime": names("axis5_regime"),
        "method": names("axis6_method"),
        "approach": names("axis7_approach_portfolio"),
        "entry_policy": names("axis8a_entry_policy"),
        "exit_mechanism": names("axis8b_exit_mechanism"),
    }
    # exit FAMILY -- the user's flagged gap (strat-based vs mechanical), read from the family tags
    fams = sorted({e.get("family") for e in strat.get("axis8b_exit_mechanism", [])
                   if isinstance(e, dict) and e.get("family")})
    axes["exit_family"] = fams or ["mechanical", "strat", "hybrid"]
    # factor families (Dimension A of the factor registry)
    if isinstance(ti, dict):
        axes["factor_family_ti"] = [k for k in ti.keys() if k != "meta" and isinstance(ti.get(k), list)]
    # frontier feature sources (Dimension C)
    if isinstance(feat, dict):
        srcs = feat.get("sources") or feat.get("feature_sources") or feat
        if isinstance(srcs, dict):
            axes["frontier_source"] = [k for k in srcs.keys() if not str(k).startswith("_")]
    return {k: v for k, v in axes.items() if v}


@dataclass
class CoverageResult:
    verdict: str                       # PASS | WARN | FAIL
    per_axis: dict = field(default_factory=dict)   # axis -> {covered, omitted, waived, n_registry}
    flags: list = field(default_factory=list)      # human-readable silent-omission flags
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"coverage verdict: {self.verdict}"]
        for ax, d in self.per_axis.items():
            cov, om, wv = len(d["covered"]), len(d["omitted"]), len(d["waived"])
            tag = "" if not om else f"  <-- {om} SILENTLY OMITTED"
            lines.append(f"  {ax:16s}: {cov}/{d['n_registry']} covered, {wv} waived{tag}")
        for f in self.flags:
            lines.append(f"  FLAG: {f}")
        return "\n".join(lines)


def coverage_report(declared: dict, waivers=None, root: Path = ROOT,
                    require_exit_both_families: bool = True) -> CoverageResult:
    """Check a discovery run's DECLARED axis coverage against the canonical registries.

    declared : {axis: [members covered]}  -- use the literal token "all" (str) to mean full coverage.
    waivers  : {axis: [members consciously skipped]}  OR axis name -> "all" to waive the whole axis.
               A waived member is NOT a silent omission. (This is how you say 'we have moved from X'.)

    Verdict: FAIL if nothing declared (undeclared coverage). WARN if any registry member is neither
    covered nor waived (a SILENT omission -- the thing the user does not want). PASS otherwise.
    Special rule (the user's exact gap): if exits are tested, BOTH the strat AND mechanical exit
    families must be covered (or waived) -- you cannot silently test only mechanical exits.
    """
    waivers = waivers or {}
    axes = registry_axes(root)
    res = CoverageResult(verdict="PASS")
    if not declared:
        res.verdict = "FAIL"
        res.flags.append("UNDECLARED COVERAGE: a discovery run must declare which axis-members it "
                         "covers (and waive the rest). Silent/empty coverage is not allowed.")
        return res

    any_silent = False
    for ax, members in axes.items():
        dec = declared.get(ax, [])
        wv = waivers.get(ax, [])
        cov_all = (dec == "all" or dec == ["all"])
        wv_all = (wv == "all" or wv == ["all"])
        covered = list(members) if cov_all else [m for m in members if m in set(dec)]
        waived = list(members) if wv_all else [m for m in members if m in set(wv)]
        omitted = [m for m in members if m not in set(covered) and m not in set(waived)]
        res.per_axis[ax] = {"covered": covered, "omitted": omitted, "waived": waived,
                            "n_registry": len(members)}
        if omitted:
            any_silent = True
            res.flags.append(f"{ax}: {len(omitted)} member(s) neither tested nor waived: "
                            f"{omitted[:6]}{' ...' if len(omitted) > 6 else ''}")

    # the flagged gap: strat-vs-mechanical exit split
    if require_exit_both_families and "exit_family" in axes:
        ef = res.per_axis["exit_family"]
        tested_or_waived = set(ef["covered"]) | set(ef["waived"])
        for fam in ("mechanical", "strat"):
            if fam in axes["exit_family"] and fam not in tested_or_waived:
                any_silent = True
                res.flags.append(f"EXIT-FAMILY GAP: '{fam}' exits neither tested nor waived -- the "
                                f"strat-vs-mechanical exit gap the user flagged. Cover or waive it.")

    if res.verdict != "FAIL":
        res.verdict = "WARN" if any_silent else "PASS"
    return res


# ----------------------------------------------------------------------------------------------
# PART 3 -- the preflight (run both; the stage-03 gate entry point)
# ----------------------------------------------------------------------------------------------
def preflight(run_decl: dict, root: Path = ROOT) -> dict:
    """Stage-03 discovery preflight. run_decl = {
        'name': str, 'axes': {axis:[members]|'all'}, 'waivers': {...}, 'configs': [tuples] (opt),
        'rel_tol': float (opt) }.
    Returns a structured verdict dict (PASS/WARN/FAIL) + the canonicalization report if configs given.
    """
    cov = coverage_report(run_decl.get("axes", {}), run_decl.get("waivers"), root)
    out = {"name": run_decl.get("name", "discovery_run"), "coverage": asdict(cov)}
    verdict = cov.verdict
    if run_decl.get("configs"):
        canon = canonicalize_grid(run_decl["configs"], run_decl.get("rel_tol", 0.15))
        out["canonicalization"] = asdict(canon)
    out["verdict"] = verdict
    return out


# ----------------------------------------------------------------------------------------------
# SELFTEST (no external data)
# ----------------------------------------------------------------------------------------------
def _selftest() -> int:
    ok = True

    # 1. canonicalize_grid collapses MA near-dups but not distinct pairs
    grid = [(28, 29), (28, 30), (29, 30), (27, 29), (50, 100), (50, 200), (20, 100)]
    c = canonicalize_grid(grid, rel_tol=0.15)
    assert c.n_raw == 7, c.n_raw
    assert c.n_effective == 4, (c.n_effective, c.representatives)  # {28x29 cluster}, (50,100),(50,200),(20,100)
    assert not near_dup((28, 29), (50, 200), 0.15)
    assert near_dup((28, 29), (28, 30), 0.15)
    print("  [1] canonicalize_grid: 7 raw -> 4 effective (MA(28,29)~MA(28,30)~MA(29,30)~MA(27,29) collapsed) OK")

    # 2. coverage: a 4h-only / mechanical-only run is WARN with the two named gaps
    cov = coverage_report({"cadence": ["4h"], "exit_mechanism": ["trailing_chandelier", "fixed_horizon_time_stop"],
                           "exit_family": ["mechanical"]})
    assert cov.verdict == "WARN", cov.verdict
    cad = cov.per_axis["cadence"]
    assert "1d" in cad["omitted"] and "15m" in cad["omitted"], cad["omitted"]
    assert any("EXIT-FAMILY GAP" in f and "strat" in f for f in cov.flags), cov.flags
    print("  [2] coverage WARN on 4h-only + mechanical-only: timeframe omissions + strat-exit gap flagged OK")

    # 3. coverage PASS when the rest is consciously waived (we have 'moved from' them)
    axes = registry_axes()
    full = {ax: "all" for ax in axes}
    cov2 = coverage_report({"cadence": ["1d"], "exit_family": ["mechanical", "strat"]},
                           waivers={**{ax: "all" for ax in axes},
                                    "cadence": [c for c in axes["cadence"] if c != "1d"]})
    assert cov2.verdict == "PASS", (cov2.verdict, cov2.flags)
    print("  [3] coverage PASS when omissions are explicitly WAIVED (conscious, not silent) OK")

    # 4. undeclared coverage FAILs
    cov3 = coverage_report({})
    assert cov3.verdict == "FAIL", cov3.verdict
    print("  [4] undeclared coverage -> FAIL OK")

    # 5. preflight integrates both
    pf = preflight({"name": "demo", "axes": {"cadence": ["4h"], "exit_family": ["mechanical"]},
                    "configs": grid})
    assert pf["verdict"] == "WARN"
    assert pf["canonicalization"]["n_effective"] == 4
    print("  [5] preflight integrates coverage + canonicalization OK")

    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def _demo() -> int:
    print("=== DEMO: a 4h-only, mechanical-exit-only discovery run (the failure mode) ===")
    pf = preflight({"name": "demo_4h_mech",
                    "axes": {"cadence": ["4h"],
                             "exit_mechanism": ["trailing_chandelier"],
                             "exit_family": ["mechanical"]},
                    "configs": [(28, 29), (28, 30), (29, 30), (50, 200), (50, 100)]})
    cov = CoverageResult(**{k: v for k, v in pf["coverage"].items()})
    print(cov.summary())
    if "canonicalization" in pf:
        print("  " + canonicalize_grid([(28, 29), (28, 30), (29, 30), (50, 200), (50, 100)]).summary())
    print(f"VERDICT: {pf['verdict']}  (WARN = omissions are SILENT; waive them to make it conscious)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Discovery preflight contract (coverage + config canonicalization).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--canon", type=str, help='dedup a grid, e.g. "28,29 28,30 50,200"')
    ap.add_argument("--rel-tol", type=float, default=0.15)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.demo:
        return _demo()
    if a.canon:
        grid = [tuple(float(x) for x in tok.split(",")) for tok in a.canon.split()]
        print(canonicalize_grid(grid, a.rel_tol).summary())
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
