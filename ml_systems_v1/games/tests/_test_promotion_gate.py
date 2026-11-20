"""Unit test for the monotonic promotion gate (train_robust.candidate_beats_champion).

This is the single most load-bearing training-correctness function in the engine: it decides
whether a candidate net REPLACES the champion. Its H1/M1 monotonicity contract (2026-06-07) was
written to fix a real ratcheting-DOWN bug -- a below-champion candidate (0.75 vs champ 0.80 with
tol=0.10) used to be PROMOTED on a strong classical score and then recorded its LOWER wr_random as
the new floor, so "the playable net never gets weaker" silently broke. The function had ZERO test
coverage (audit 2026-06-10), so that exact regression could come back unnoticed. This file locks
the contract into the auto-discovered test gate (run_tests.py).

Run:  .venv/Scripts/python.exe -m az._test_promotion_gate
Exit: 0 if every gate case holds; nonzero otherwise. Fast (<5s), CPU. No emoji (Windows cp1252).
"""
from __future__ import annotations

from az.train_robust import candidate_beats_champion, Champion

CASES: list[tuple[str, bool]] = []


def _ck(name: str, cond: bool) -> None:
    CASES.append((name, bool(cond)))


def _champ(wr_random: float, score_classical: float = 0.0, loss: float = 1.0) -> Champion:
    return Champion(
        iter=5,
        winrate_vs_random=wr_random,
        winrate_vs_classical=0.0,
        loss=loss,
        score_vs_classical=score_classical,
    )


def run() -> int:
    # 1. STRICT FLOOR (H1) -- THE canonical bug. A candidate BELOW the champion on wr_random is
    #    REJECTED unconditionally, even with a crushing classical score. (0.75 vs 0.80 @ tol=0.10
    #    used to PROMOTE and ratchet the floor DOWN.)
    promote, reason = candidate_beats_champion(
        cand_wr_random=0.75, cand_wr_classical=0.99, cand_loss=0.01,
        champ=_champ(0.80), tol=0.10, cand_score_classical=0.99,
    )
    _ck("below-floor candidate REJECTED even w/ strong classical (H1)", promote is False)

    # 2. STRICT WIN -- strictly above champion (beyond tol) on the primary yardstick -> PROMOTE.
    promote, _ = candidate_beats_champion(
        cand_wr_random=0.95, cand_wr_classical=0.0, cand_loss=1.0,
        champ=_champ(0.80), tol=0.0,
    )
    _ck("clearly-stronger candidate PROMOTED", promote is True)

    # 3. CLIMB tie-break PROMOTE -- wr_random tied (within tol); higher draw-aware classical score
    #    (losing -> drawing progress) -> PROMOTE.
    promote, reason = candidate_beats_champion(
        cand_wr_random=0.80, cand_wr_classical=0.0, cand_loss=1.0,
        champ=_champ(0.80, score_classical=0.25), tol=0.05, cand_score_classical=0.50,
    )
    _ck("tied wr_random + higher score_vs_classical PROMOTES (CLIMB)",
        promote is True and "CLIMB" in reason)

    # 4. CLIMB tie-break HOLD -- wr_random tied; LOWER classical score = classical regression -> HOLD.
    promote, _ = candidate_beats_champion(
        cand_wr_random=0.80, cand_wr_classical=0.0, cand_loss=0.01,
        champ=_champ(0.80, score_classical=0.25), tol=0.05, cand_score_classical=0.10,
    )
    _ck("tied wr_random + lower score_vs_classical HOLDS champion", promote is False)

    # 5. SATURATED-TIE HOLD (M1) -- both wr_random AND score_vs_classical tied, no h2h gate. Train
    #    loss is NOT a strength signal, so a lower loss MUST NOT promote (else gate -> loss-chasing).
    promote, _ = candidate_beats_champion(
        cand_wr_random=0.80, cand_wr_classical=0.0, cand_loss=0.01,
        champ=_champ(0.80, score_classical=0.25, loss=1.0), tol=0.05, cand_score_classical=0.25,
    )
    _ck("saturated tie + lower loss HOLDS (loss is not strength, M1)", promote is False)

    # 6. h2h VETO -- a candidate that does not out-play the champion head-to-head is NEVER promoted,
    #    regardless of a dominant vs-random number.
    promote, reason = candidate_beats_champion(
        cand_wr_random=0.95, cand_wr_classical=0.0, cand_loss=1.0,
        champ=_champ(0.80), tol=0.0, h2h_winrate=0.40, h2h_threshold=0.55,
    )
    _ck("h2h below threshold VETOES an otherwise-winning candidate",
        promote is False and "h2h" in reason)

    # 7. Saturated tie, h2h PASSED, lower loss -> loss MAY break the tie -> PROMOTE.
    promote, _ = candidate_beats_champion(
        cand_wr_random=0.80, cand_wr_classical=0.0, cand_loss=0.10,
        champ=_champ(0.80, score_classical=0.25, loss=1.0), tol=0.05,
        cand_score_classical=0.25, h2h_winrate=0.60, h2h_threshold=0.55,
    )
    _ck("saturated tie + h2h passed + lower loss PROMOTES", promote is True)

    # 8. STRICT >= at tol=0.0 -- exactly-equal wr_random + tied score + no h2h reaches the saturated
    #    tie and HOLDS (no improvement is not an improvement).
    promote, _ = candidate_beats_champion(
        cand_wr_random=0.80, cand_wr_classical=0.0, cand_loss=1.0,
        champ=_champ(0.80, score_classical=0.0, loss=1.0), tol=0.0, cand_score_classical=0.0,
    )
    _ck("exact tie at tol=0.0 with no improvement HOLDS", promote is False)

    failed = [n for n, ok in CASES if not ok]
    print(f"=== promotion-gate unit test: {len(CASES)-len(failed)}/{len(CASES)} passed ===")
    for n, ok in CASES:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    if failed:
        print(f"\nFAILED: {failed}")
        return 1
    print("[ok] candidate_beats_champion monotonicity contract holds (H1 floor + M1 saturated-tie)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
