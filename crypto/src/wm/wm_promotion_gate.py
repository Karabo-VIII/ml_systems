"""src/wm/wm_promotion_gate.py -- Compound-return champion-gate for WM promotion.

PURPOSE
-------
Decide whether a newly-trained WM version should replace the current champion.

GATE RULE (STRICT MONOTONIC FLOOR)
-----------------------------------
A candidate is promoted ONLY if its held-out compound return strictly exceeds
the champion's by at least `min_improvement` (default 0.0, i.e., strictly better).
The floor ratchets UPWARD only: a rejected candidate never lowers the bar.

WHY COMPOUND, NOT IC
---------------------
IC (Information Coefficient) measures per-bar predictability -- a useful *within-WM*
diagnostic that tells you whether training converged, but it does NOT directly
answer "is this WM worth running in production?". Our objective is held-out
compound return (see MEMORY.md founding framing, PROJECT_NORTH_STAR.md). A model
with IC 0.08 that produces flat compound return loses to a model with IC 0.04 that
compounds well. The gate enforces this distinction mechanically.

IC is NOT checked here. Use GATE_IC_MIN in settings.py as a within-training
diagnostic before you even produce a candidate for this gate.

CHAMPION RECORD
---------------
`runs/wm/champion.json` stores the current champion's compound + checkpoint ID.
Written atomically (tmp -> rename) so a crash cannot corrupt the record.
The floor only moves up: a promote() call updates the record; a reject never does.

CONNECT TO wm_value_probe
-------------------------
`evaluate_candidate` accepts pre-computed probe result dicts (the dict returned
by src/strat/wm_value_probe.evaluate_asset or a portfolio summary), extracts the
compound number, and feeds it into should_promote.  This design keeps should_promote
a pure, testable function with no I/O, while evaluate_candidate is the seam that
connects to the probe harness.

Full wiring example:
    from strat.wm_value_probe import evaluate_asset
    from strat.wm_entry_producer import WMEntryProducer
    producer_cand = WMEntryProducer(n_features=41, ckpt_path=candidate_ckpt_path)
    # run probe on UNSEEN segment for each asset, average compound
    probe_result = {"compound_pct": <mean across assets>}
    decision = evaluate_candidate(probe_result, champion_ckpt_id="v1.1-f34-ep120")

USAGE
-----
    python src/wm/wm_promotion_gate.py          # run self-test
    python src/wm/wm_promotion_gate.py --selftest
    python src/wm/wm_promotion_gate.py --promote-compound 0.12 --champion-compound 0.07
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# CDAP contract
# ---------------------------------------------------------------------------
__contract__ = {
    "kind": "promotion_gate",
    "inputs": [
        "candidate_compound: float (held-out compound return, decimal, e.g. 0.10 = 10%)",
        "champion_compound: float",
    ],
    "outputs": [
        "decision: dict {promote, reason, candidate_compound, champion_compound, margin}"
    ],
    "invariants": {
        "gate_metric": "held-out compound return (NOT IC -- IC is a within-WM diagnostic only)",
        "monotonic_floor": "champion record only ever moves up; reject never lowers the bar",
        "atomic_write": "champion.json written via tmp+rename to survive crashes",
        "no_ic_gate": "IC is intentionally absent -- see PURPOSE section",
    },
}

# Default champion record path
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAMPION_PATH = _ROOT / "runs" / "wm" / "champion.json"


# ---------------------------------------------------------------------------
# Core gate: pure function, no I/O
# ---------------------------------------------------------------------------

def should_promote(
    candidate_compound: float,
    champion_compound: float,
    *,
    min_improvement: float = 0.0,
    candidate_meta: Optional[Dict[str, Any]] = None,
    champion_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide whether a candidate WM should replace the current champion.

    Gate rule (STRICT MONOTONIC FLOOR):
        promote iff candidate_compound > champion_compound + min_improvement

    This is intentionally simple and hard to game. The compound return is the
    single objective; no weighted combinations, no secondary tie-breakers.

    Args:
        candidate_compound: Held-out compound return of the candidate (decimal,
            e.g. 0.10 means +10%). Must be from UNSEEN segment only -- never OOS.
        champion_compound:  Same metric for the current champion.
        min_improvement:    Minimum required margin above champion (default 0.0
            = strictly better; raise to e.g. 0.01 to require +1pp clear margin).
        candidate_meta:     Optional dict of extra candidate info (ckpt_id, epoch,
            IC, n_assets, etc.) stored in the record but not used in the gate.
        champion_meta:      Same for champion.

    Returns:
        dict with keys:
            promote          (bool)   -- True = candidate should become champion
            reason           (str)    -- human-readable explanation
            candidate_compound (float)
            champion_compound  (float)
            margin           (float)  -- candidate - champion (positive = candidate better)
            min_improvement  (float)
            candidate_meta   (dict | None)
            champion_meta    (dict | None)
    """
    margin = candidate_compound - champion_compound
    threshold = min_improvement  # candidate must beat champion BY AT LEAST this

    if margin > threshold:
        promote = True
        reason = (
            f"PROMOTE: candidate compound {candidate_compound:+.4f} "
            f"exceeds champion {champion_compound:+.4f} "
            f"by {margin:+.4f} (min_improvement={min_improvement:+.4f})"
        )
    elif margin == threshold and threshold == 0.0 and candidate_compound == champion_compound:
        # Exact tie -- do NOT promote; keeping champion avoids pointless churn
        promote = False
        reason = (
            f"REJECT (tie): candidate compound {candidate_compound:+.4f} "
            f"equals champion {champion_compound:+.4f}; champion retained (no churn on tie)"
        )
    else:
        promote = False
        reason = (
            f"REJECT: candidate compound {candidate_compound:+.4f} "
            f"does NOT exceed champion {champion_compound:+.4f} "
            f"by required margin {min_improvement:+.4f} (actual margin {margin:+.4f})"
        )

    return {
        "promote": promote,
        "reason": reason,
        "candidate_compound": candidate_compound,
        "champion_compound": champion_compound,
        "margin": margin,
        "min_improvement": min_improvement,
        "candidate_meta": candidate_meta,
        "champion_meta": champion_meta,
    }


# ---------------------------------------------------------------------------
# Champion record I/O (atomic write; floor only moves up)
# ---------------------------------------------------------------------------

def load_champion_record(path: Path = DEFAULT_CHAMPION_PATH) -> Dict[str, Any]:
    """Load the champion record from disk.

    Returns a default zero-compound record if the file does not exist yet
    (first run: any positive candidate will be promoted).

    Returns:
        dict with keys: compound, ckpt_id, meta
    """
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        # Validate required keys
        if "compound" not in record:
            raise ValueError(
                f"champion.json at {path} is missing required key 'compound'. "
                "Delete the file to start fresh."
            )
        return record
    else:
        # No champion yet -- floor is at -infinity so the first candidate always wins
        return {
            "compound": float("-inf"),
            "ckpt_id": None,
            "meta": {},
            "note": "default record (no champion yet); first valid candidate will be promoted",
        }


def save_champion_record(
    compound: float,
    ckpt_id: Optional[str],
    meta: Optional[Dict[str, Any]] = None,
    path: Path = DEFAULT_CHAMPION_PATH,
) -> None:
    """Atomically write a new champion record.

    Uses a tmp file in the same directory + os.replace() for crash safety.
    The floor only moves UP: callers MUST check should_promote() before calling
    this and only call it when promote=True.

    Args:
        compound: New champion's held-out compound return.
        ckpt_id:  Checkpoint identifier string (path, version tag, etc.).
        meta:     Optional additional metadata stored verbatim.
        path:     Destination path (default: runs/wm/champion.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "compound": compound,
        "ckpt_id": ckpt_id,
        "meta": meta or {},
    }
    # Atomic write: write to tmp then os.replace (POSIX-atomic; on Windows also
    # atomic within the same filesystem per os.replace docs)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".champion_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp_path, str(path))
    except Exception:
        # Best-effort cleanup of the tmp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Higher-level: evaluate + conditionally promote
# ---------------------------------------------------------------------------

def evaluate_candidate(
    candidate_probe_result: Dict[str, Any],
    champion_ckpt_id: Optional[str] = None,
    *,
    candidate_ckpt_id: Optional[str] = None,
    min_improvement: float = 0.0,
    champion_path: Path = DEFAULT_CHAMPION_PATH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Evaluate a candidate probe result against the stored champion and
    conditionally update the champion record.

    This is the integration seam between wm_value_probe.py and the gate.
    It does NOT run the probe itself -- pass in the pre-computed result dict.

    Probe result dict format (from wm_value_probe.evaluate_asset or a
    portfolio-level summary you compute from multiple evaluate_asset calls):
        {"compound_pct": <float>}  -- compound in PERCENT (e.g. 10.0 = +10%)
        OR
        {"wm": {"compound_pct": <float>}}  -- nested format from evaluate_asset

    The gate converts percent -> decimal internally for consistent storage.

    Args:
        candidate_probe_result: Dict from wm_value_probe with compound_pct.
        champion_ckpt_id:       ID of current champion (overrides record on disk
                                only for the purpose of this call; record on disk
                                is the source of truth).
        candidate_ckpt_id:      ID string for the candidate (stored on promote).
        min_improvement:        Minimum required margin (decimal). Default 0.0.
        champion_path:          Path to champion.json.
        dry_run:                If True, compute the decision but do NOT write the
                                record (useful for testing).

    Returns:
        decision dict from should_promote, augmented with:
            champion_ckpt_id    (str | None)
            candidate_ckpt_id   (str | None)
            record_updated      (bool)
            dry_run             (bool)

    Connection to wm_value_probe
    ----------------------------
    Full example:
        from strat.wm_value_probe import evaluate_asset, WMEntryProducer
        producer = WMEntryProducer(n_features=41)
        per_asset = [evaluate_asset(sym, producer) for sym in assets]
        valid = [r for r in per_asset if not r.get("skip")]
        mean_compound_pct = np.mean([r["wm"]["compound_pct"] for r in valid])
        probe_summary = {"compound_pct": mean_compound_pct}
        decision = evaluate_candidate(probe_summary, candidate_ckpt_id="v1.1-f41-ep80")
    """
    # Extract compound_pct from probe result
    if "compound_pct" in candidate_probe_result:
        compound_pct = float(candidate_probe_result["compound_pct"])
    elif "wm" in candidate_probe_result and "compound_pct" in candidate_probe_result["wm"]:
        compound_pct = float(candidate_probe_result["wm"]["compound_pct"])
    else:
        raise ValueError(
            "candidate_probe_result must contain 'compound_pct' (top-level) "
            "or 'wm.compound_pct' (nested). "
            f"Got keys: {list(candidate_probe_result.keys())}"
        )

    # Convert percent -> decimal for the gate (storage uses decimal for precision)
    candidate_compound = compound_pct / 100.0

    # Load champion record
    record = load_champion_record(champion_path)
    champion_compound = float(record["compound"])
    stored_champion_ckpt_id = record.get("ckpt_id")

    # Run the pure gate
    decision = should_promote(
        candidate_compound=candidate_compound,
        champion_compound=champion_compound,
        min_improvement=min_improvement,
        candidate_meta={"ckpt_id": candidate_ckpt_id, "raw_pct": compound_pct},
        champion_meta={"ckpt_id": stored_champion_ckpt_id or champion_ckpt_id},
    )

    record_updated = False
    if decision["promote"] and not dry_run:
        save_champion_record(
            compound=candidate_compound,
            ckpt_id=candidate_ckpt_id,
            meta={"promoted_from_compound_pct": compound_pct},
            path=champion_path,
        )
        record_updated = True

    decision["champion_ckpt_id"] = stored_champion_ckpt_id or champion_ckpt_id
    decision["candidate_ckpt_id"] = candidate_ckpt_id
    decision["record_updated"] = record_updated
    decision["dry_run"] = dry_run

    return decision


# ---------------------------------------------------------------------------
# Self-test (also serves as __main__)
# ---------------------------------------------------------------------------

def _run_selftest(champion_path: Path) -> None:
    """Verify promote/reject/ratchet behavior.

    Test cases:
    1. Better candidate (0.10 vs 0.05) -> PROMOTE, record updates to 0.10
    2. Equal candidate (0.10 vs 0.10)  -> REJECT (tie), record stays at 0.10
    3. Worse candidate (0.03 vs 0.10)  -> REJECT, record stays at 0.10 (not lowered)
    4. min_improvement=0.02: margin 0.01 -> REJECT even though candidate > champion
    5. min_improvement=0.02: margin 0.03 -> PROMOTE
    """
    import shutil

    # Use a temp copy so we don't clobber the real champion.json
    tmp_dir = Path(tempfile.mkdtemp(prefix="wm_gate_selftest_"))
    test_champion_path = tmp_dir / "champion.json"

    try:
        print("\n" + "=" * 60)
        print("  WM_PROMOTION_GATE SELF-TEST")
        print("  Gate metric: held-out compound return (NOT IC)")
        print("=" * 60)

        failures = []

        # ---- Test 1: Better candidate (no existing record -> default -inf champion) ----
        print("\n[T1] Better candidate: compound=0.10, champion=-inf (no record yet)")
        decision = evaluate_candidate(
            {"compound_pct": 10.0},
            candidate_ckpt_id="v1.1-test-T1",
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  reason={decision['reason']}")
        assert decision["promote"], "T1 FAIL: expected PROMOTE"
        assert decision["record_updated"], "T1 FAIL: expected record_updated=True"
        assert abs(decision["candidate_compound"] - 0.10) < 1e-9, "T1 FAIL: compound mismatch"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.10) < 1e-9, "T1 FAIL: champion record not updated"
        print("     T1 PASS: promoted to 0.10, record updated")

        # ---- Test 2: Equal candidate -> reject (no churn on tie) ----
        print("\n[T2] Equal candidate: compound=0.10, champion=0.10")
        decision = evaluate_candidate(
            {"compound_pct": 10.0},
            candidate_ckpt_id="v1.1-test-T2",
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  reason={decision['reason']}")
        assert not decision["promote"], "T2 FAIL: expected REJECT on tie"
        assert not decision["record_updated"], "T2 FAIL: expected record_updated=False"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.10) < 1e-9, "T2 FAIL: record should be unchanged"
        print("     T2 PASS: rejected (tie), floor stays at 0.10")

        # ---- Test 3: Worse candidate -> reject, floor does NOT lower ----
        print("\n[T3] Worse candidate: compound=0.03, champion=0.10")
        decision = evaluate_candidate(
            {"compound_pct": 3.0},
            candidate_ckpt_id="v1.1-test-T3",
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  reason={decision['reason']}")
        assert not decision["promote"], "T3 FAIL: expected REJECT"
        assert not decision["record_updated"], "T3 FAIL: expected record_updated=False"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.10) < 1e-9, "T3 FAIL: floor must NOT drop"
        print("     T3 PASS: rejected, floor stays at 0.10 (not lowered to 0.03)")

        # ---- Test 4: min_improvement=0.02, margin 0.01 -> reject ----
        print("\n[T4] min_improvement=0.02: candidate=0.11 (+0.01 over 0.10) -> expect REJECT")
        decision = evaluate_candidate(
            {"compound_pct": 11.0},
            candidate_ckpt_id="v1.1-test-T4",
            min_improvement=0.02,
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  reason={decision['reason']}")
        assert not decision["promote"], "T4 FAIL: expected REJECT (margin 0.01 < min_improvement 0.02)"
        assert not decision["record_updated"], "T4 FAIL: record should not update"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.10) < 1e-9, "T4 FAIL: floor must stay at 0.10"
        print("     T4 PASS: rejected (margin 0.01 below min_improvement 0.02)")

        # ---- Test 5: min_improvement=0.02, margin 0.03 -> promote ----
        print("\n[T5] min_improvement=0.02: candidate=0.13 (+0.03 over 0.10) -> expect PROMOTE")
        decision = evaluate_candidate(
            {"compound_pct": 13.0},
            candidate_ckpt_id="v1.1-test-T5",
            min_improvement=0.02,
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  reason={decision['reason']}")
        assert decision["promote"], "T5 FAIL: expected PROMOTE"
        assert decision["record_updated"], "T5 FAIL: expected record_updated=True"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.13) < 1e-9, "T5 FAIL: champion should be 0.13"
        print("     T5 PASS: promoted to 0.13, floor ratcheted up")

        # ---- Test 6: Ratchet confirmed -- champion is now 0.13, old 0.10 can't reclaim ----
        print("\n[T6] Ratchet: candidate=0.10, champion=0.13 -> expect REJECT")
        decision = evaluate_candidate(
            {"compound_pct": 10.0},
            candidate_ckpt_id="v1.1-test-T6",
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  reason={decision['reason']}")
        assert not decision["promote"], "T6 FAIL: expected REJECT (old champion can't reclaim)"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.13) < 1e-9, "T6 FAIL: floor must stay at 0.13"
        print("     T6 PASS: ratchet confirmed -- 0.13 floor holds, old 0.10 rejected")

        # ---- Test 7: dry_run -- should compute decision but NOT write ----
        print("\n[T7] dry_run=True: candidate=0.20, champion=0.13 -> PROMOTE decision but no write")
        decision = evaluate_candidate(
            {"compound_pct": 20.0},
            candidate_ckpt_id="v1.1-test-T7",
            champion_path=test_champion_path,
            dry_run=True,
        )
        print(f"     promote={decision['promote']}  record_updated={decision['record_updated']}")
        assert decision["promote"], "T7 FAIL: dry_run should still compute promote=True"
        assert not decision["record_updated"], "T7 FAIL: dry_run must not write record"
        record = load_champion_record(test_champion_path)
        assert abs(record["compound"] - 0.13) < 1e-9, "T7 FAIL: dry_run must not change record"
        print("     T7 PASS: dry_run computed PROMOTE but did not write")

        # ---- Test 8: nested probe format (wm.compound_pct) ----
        print("\n[T8] Nested probe format {wm: {compound_pct: 20.0}}")
        decision = evaluate_candidate(
            {"wm": {"compound_pct": 20.0}},
            candidate_ckpt_id="v1.1-test-T8",
            champion_path=test_champion_path,
        )
        print(f"     promote={decision['promote']}  candidate_compound={decision['candidate_compound']:.4f}")
        assert decision["promote"], "T8 FAIL: nested format should extract compound correctly"
        assert abs(decision["candidate_compound"] - 0.20) < 1e-9, "T8 FAIL: compound extraction wrong"
        print("     T8 PASS: nested probe format parsed correctly, promoted to 0.20")

        print("\n" + "=" * 60)
        print("  ALL SELF-TESTS PASSED (8/8)")
        print("  promote/reject/ratchet all correct")
        print("  Gate metric: compound return (IC is within-WM diagnostic only)")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WM compound-return promotion gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--selftest", action="store_true", default=False,
                        help="Run self-test (default when called with no args)")
    parser.add_argument("--promote-compound", type=float, default=None,
                        help="Candidate compound (decimal, e.g. 0.10 for +10%%)")
    parser.add_argument("--champion-compound", type=float, default=None,
                        help="Champion compound (decimal). If omitted, loads from champion.json")
    parser.add_argument("--min-improvement", type=float, default=0.0,
                        help="Minimum required margin above champion (default 0.0)")
    parser.add_argument("--champion-path", type=Path,
                        default=DEFAULT_CHAMPION_PATH,
                        help="Path to champion.json")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Compute decision without writing champion record")
    args = parser.parse_args()

    # Default: run self-test
    if args.selftest or (args.promote_compound is None and args.champion_compound is None):
        _run_selftest(args.champion_path)
        return

    if args.promote_compound is None:
        parser.error("--promote-compound is required when not running --selftest")

    # Direct CLI usage: evaluate a specific candidate vs champion
    if args.champion_compound is not None:
        decision = should_promote(
            candidate_compound=args.promote_compound,
            champion_compound=args.champion_compound,
            min_improvement=args.min_improvement,
        )
    else:
        decision = evaluate_candidate(
            {"compound_pct": args.promote_compound * 100.0},
            min_improvement=args.min_improvement,
            champion_path=args.champion_path,
            dry_run=args.dry_run,
        )

    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
