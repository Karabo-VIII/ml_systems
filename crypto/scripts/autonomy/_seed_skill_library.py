#!/usr/bin/env python3
"""Seed the skill_library INDEX.json with the ACTUAL reusable assets already in the repo.

Run once (or re-run to refresh provenance SHA after new commits):
    python scripts/autonomy/_seed_skill_library.py

Idempotent: register() overwrites existing entries by name.
No emoji (Windows cp1252 safety).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))

from skill_library import register  # noqa: E402


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


SHA = _sha()

ASSETS = [
    # ------------------------------------------------------------------
    # GATES / VALIDATION
    # ------------------------------------------------------------------
    dict(
        name="candidate_gate",
        kind="gate",
        path="src/strat/candidate_gate.py",
        entrypoint="evaluate_candidate",
        signature="(harness: CanonicalHarness, family_n: int | None = None, n_books: int = 200, require_taker: bool = True) -> dict",
        summary=(
            "Single reusable validation gate: wires harness -> leak-probe -> firewall -> battery "
            "into one callable returning a consolidated verdict dict with 'verdict', 'lens_a/b/c', "
            "'cost_warning', 'leak', 'firewall', and 'battery' sub-dicts."
        ),
        tested_on="SOL 4h + BTC 1d; positive_control synthetic series (RWYB 2026-06-05)",
        provenance_sha=SHA,
        tags=["validation", "gate", "strat", "foundation", "apparatus"],
    ),
    # ------------------------------------------------------------------
    # FIREWALL (random-entry null)
    # ------------------------------------------------------------------
    dict(
        name="random_entry_null",
        kind="probe",
        path="src/strat/firewall.py",
        entrypoint="random_entry_null",
        signature="(harness: CanonicalHarness, n_books: int = 300, seed: int = 7, regime_matched: bool = False) -> dict",
        summary=(
            "Cost-matched random-entry null (LD-4 firewall). Checks whether a candidate's per-window "
            "compound beats a null of the same n_trades entered at random bars, held for durations sampled "
            "from the candidate's own holding distribution. Returns 'beats_held', 'pos_held', 'pos_all', "
            "per-window detail. Regime-matched variant available."
        ),
        tested_on="SOL 4h, BTC 1d; F2 CRITICAL bug (zero-trade window) fixed 2026-06-05 (RWYB)",
        provenance_sha=SHA,
        tags=["firewall", "null", "validation", "strat", "apparatus", "robustness"],
    ),
    # ------------------------------------------------------------------
    # BATTERY (robustness)
    # ------------------------------------------------------------------
    dict(
        name="battery_evaluate",
        kind="harness",
        path="src/strat/battery.py",
        entrypoint="evaluate",
        signature=(
            "(unseen_returns: list, comps: dict, unseen_maxdd_pct: float, "
            "n_boots: int = 2000, family_n: int | None = None) -> dict"
        ),
        summary=(
            "Robustness battery (Lens A/B/C). Lens A (strict): all-4-positive AND n>=15 AND n_eff>=15 "
            "AND jk2/jk3>0 AND p05>0 AND maxDD<30%. Lens B/C are permissive variants. Returns 'verdict' "
            "('SHIP'/'INCUBATE'/'REJECT') + full stats including jackknife, bootstrap p05, DSR."
        ),
        tested_on="SOL 4h, BTC 1d; F6 off-by-one bootstrap fixed 2026-06-05 (RWYB)",
        provenance_sha=SHA,
        tags=["battery", "robustness", "validation", "strat", "apparatus", "bootstrap"],
    ),
    # ------------------------------------------------------------------
    # POSITIVE CONTROL (apparatus soundness)
    # ------------------------------------------------------------------
    dict(
        name="positive_control",
        kind="probe",
        path="src/strat/positive_control.py",
        entrypoint="run_positive_control",
        signature="(verbose: bool = True) -> dict",
        summary=(
            "Statistical-power half of the apparatus soundness check. Builds a synthetic regime-switching "
            "price with a GENUINE past-only SMA timing edge, then confirms the full evaluate_candidate chain "
            "SHIPs it (passes leak + beats firewall + clears battery). Proves the gate has power -- not just "
            "that it rejects ghosts. Run if gate seems too strict (returns 'verdict': 'SHIP' on success)."
        ),
        tested_on="Synthetic deterministic seed=11 series; two-sided soundness verified 2026-06-05 (RWYB)",
        provenance_sha=SHA,
        tags=["positive_control", "soundness", "apparatus", "validation", "strat"],
    ),
    # ------------------------------------------------------------------
    # ORACLE DNA SHUFFLED FALSIFIER
    # ------------------------------------------------------------------
    dict(
        name="oracle_dna_shuffled_falsifier",
        kind="probe",
        path="experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py",
        entrypoint="run",
        signature=(
            "(asset: str = 'SOL', cadence: str = '4h', n_shuffle: int = 50, "
            "n_books: int = 400, seed: int = 7, verbose: bool = True, "
            "exit_h: int | None = None, min_move_net: float = 0.02) -> dict"
        ),
        summary=(
            "Two-sided DNA falsifier for the oracle-decomposition method. "
            "Three controls on held-out folds: (A) shuffled-label -- genuine pipeline must collapse to AUC~0.5; "
            "(B) positive control -- synthetic known-function label must be learnable (pipeline has power); "
            "(C) regime-matched firewall -- DNA must beat cost-matched random entries IN SAME REGIME. "
            "Returns per-control AUC, IC, capture-skill, and a 'genuine' boolean."
        ),
        tested_on="SOL 4h, BTC 1d (2026-06-06 campaign -- all controls null held-out at bar-level)",
        provenance_sha=SHA,
        tags=["oracle", "dna", "falsifier", "shuffled", "probe", "adaptive_ma", "validation"],
    ),
    # ------------------------------------------------------------------
    # ORACLE CEILING BUILDER
    # ------------------------------------------------------------------
    dict(
        name="oracle_ceiling_builder",
        kind="tool",
        path="runs/research/oracle_ceiling_builder.py",
        entrypoint="oracle_high_capture",
        signature=(
            "(ts_ms: np.ndarray, open_: np.ndarray, high: np.ndarray, "
            "cost: float = 0.0024, min_hold_hours: float = 4.0, "
            "min_move_net: float = 0.0) -> dict"
        ),
        summary=(
            "Perfect-foresight long-only per-move-capture CEILING map via backward DP (max-product longest path). "
            "Computes the maximum compound a clairvoyant single-position trader could realize "
            "(entry=open[k], exit=high[j], non-overlapping, taker cost deducted). "
            "Returns trades list, per-window compound, and move distribution. "
            "Use to establish what a perfect MA-DNA signal would attain."
        ),
        tested_on="SOL 4h, BTC 1d, ETH 4h (oracle_ceiling_builder.py --selftest passes; RWYB 2026-06-06)",
        provenance_sha=SHA,
        tags=["oracle", "ceiling", "dp", "swing", "research", "harvestability"],
    ),
    # ------------------------------------------------------------------
    # NARRATE ENGINE
    # ------------------------------------------------------------------
    dict(
        name="narrate",
        kind="engine",
        path="src/narrate/__init__.py",
        entrypoint="narrate",
        signature="(asset: str, cadence: str, start: int | None = None, end: int | None = None) -> MarketNarration",
        summary=(
            "Descriptive market-intelligence engine. Narrates state, structure, regime, flow, and notable events "
            "by decomposing ALL chimera features into human-readable family reads. Returns MarketNarration "
            "(structured + prose). DESCRIPTIVE only -- describes the WHAT, does not forecast. "
            "Per-setup (multi-candle STATE), chart-type aware."
        ),
        tested_on="BTC/SOL/ETH multiple cadences; narrate selftest (selftest_narrate.py) 2026-06-06",
        provenance_sha=SHA,
        tags=["narrate", "market_intelligence", "descriptive", "regime", "engine"],
    ),
    # ------------------------------------------------------------------
    # AUTONOMY OPS TOOLS
    # ------------------------------------------------------------------
    dict(
        name="watcher",
        kind="tool",
        path="scripts/autonomy/watcher.py",
        entrypoint="main (CLI: python scripts/autonomy/watcher.py [--lifetime-hours N])",
        signature="(standalone script; no importable API -- run as subprocess)",
        summary=(
            "Absolute 1-min liveness watcher. Every 60s: auto-discovers loop locks, checks process liveness "
            "(OS-level PID check), appends timestamped CHECK-IN to runs/autonomy/watcher.log. "
            "Exits early with reason=loop_dead if a loop dies. Does NOT auto-relaunch (relaunch belongs to orchestrator). "
            "Required by the autonomous-runner mandate."
        ),
        tested_on="Live runs 2026-06-06; self-healing stale-lock logic verified",
        provenance_sha=SHA,
        tags=["autonomy", "ops", "watcher", "liveness", "monitor"],
    ),
    dict(
        name="loops_alive",
        kind="tool",
        path="scripts/autonomy/loops_alive.py",
        entrypoint="main (CLI: python scripts/autonomy/loops_alive.py <thread1> [thread2 ...])",
        signature="(*threads: str) -> int  # exit code = number of DEAD/missing threads",
        summary=(
            "Loop liveness checker. Checks each named thread's lock PID against the OS "
            "(not just lock-file existence -- catches crashed-but-lock-leftover state). "
            "Prints '<n_alive> alive | dead: [...]'. Exit code = number of dead threads (0 = all alive). "
            "Composable with && / || in shell scripts."
        ),
        tested_on="Verified gap-fix (PID vs lock-existence) 2026-06-06",
        provenance_sha=SHA,
        tags=["autonomy", "ops", "liveness", "loops", "health"],
    ),
    dict(
        name="launch_autonomy",
        kind="tool",
        path="scripts/autonomy/launch_autonomy.py",
        entrypoint="main (CLI: python scripts/autonomy/launch_autonomy.py --objective '...' [--mode attended|unattended])",
        signature="(standalone script; orchestrator entry-point)",
        summary=(
            "Canonical entry-point to stand up THREE autonomous loops together: (1) problem-solver "
            "(expert + plain LangGraph loop), (2) meta agent (improves the solving: learnings/frontier re-rank), "
            "(3) project-wide evolution loop (fires every 3h; hardens framework). "
            "Two modes: attended (OVERSEER present) and unattended (fully autonomous)."
        ),
        tested_on="Verified 2026-06-06 against live metaop stack",
        provenance_sha=SHA,
        tags=["autonomy", "ops", "launch", "orchestrator", "loops"],
    ),
    dict(
        name="resume_all",
        kind="tool",
        path="scripts/autonomy/resume_all.py",
        entrypoint="main (CLI: python scripts/autonomy/resume_all.py [--thread T])",
        signature="(standalone script; recovery + cell-style lineage layer)",
        summary=(
            "Resume-all + loop-brief metadata: (1) lists every parked metaop loop and resumes it from "
            "durable SQLite checkpoint (state survives a dead orchestrator); (2) injects loop briefs "
            "(runs/autonomy/loop_briefs/<thread>.json) with objective + parent generation + prior-generation "
            "learnings digest for cell-style compounding. Use after crash/subscription-limit recovery."
        ),
        tested_on="Cross-process resume verified 2026-06-06 (meta_graph.py MEMORY.md)",
        provenance_sha=SHA,
        tags=["autonomy", "ops", "resume", "recovery", "loops", "lineage"],
    ),
]


def main():
    print(f"=== seeding skill_library from SHA {SHA[:12]} ===")
    for a in ASSETS:
        register(**a)
    print(f"\nseeded {len(ASSETS)} assets into {ROOT / 'runs' / 'autonomy' / 'skill_library' / 'INDEX.json'}")


if __name__ == "__main__":
    main()
