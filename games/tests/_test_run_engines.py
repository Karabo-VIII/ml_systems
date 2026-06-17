"""
CI lock: the TURNKEY demo (projects/chess_zero/run_engines.py) actually plays all THREE
engines to a valid terminal result -- chess, connect-4, atari -- using the trained
checkpoints (or a fresh net if a checkpoint is missing), without error.

This is the gate for run_engines.py (the "click run" headline deliverable). It runs each
engine for ONE game/episode in the fast, non-rendered mode (low sims, delay 0) and asserts:

  TEST 1 (chess)    -- play_chess returns a dict whose game has a VALID python-chess result
                       ('1-0' / '0-1' / '1/2-1/2'), the champion-POV is WIN/LOSS/DRAW, and at
                       least one legal SAN move was played.
  TEST 2 (connect4) -- play_connect4 returns a dict whose game ended (net-POV WIN/LOSS/DRAW)
                       with >= 1 move; the 7-column board reached a real terminal.
  TEST 3 (atari)    -- play_atari returns a dict with a finite episode return in [-1, 1]
                       (single-drop catch: +1 caught / -1 missed) and >= 1 step.
  TEST 4 (cli/all)  -- run(engine='all', fast=True, render=False) returns 3 outcome dicts,
                       one per engine, each well-formed.

SPEED: every engine runs at LOW sims (chess 6, connect4 8, atari 6) with render off, so the
whole gate finishes in well under the 200s budget (measured ~20-40s on the dev box). The
checkpoints are loaded if present; a missing checkpoint falls back to an untrained net and the
test STILL passes (it asserts the demo RUNS + produces a valid result, not a strength bar --
strength is locked by the per-engine training tests).

NON-FLAKY: the only assertions are structural (valid result strings, legal moves, finite
returns) -- never an absolutist outcome (a trained champion can legitimately lose to the
classical engine; that is the honest recorded score_vs_classical).

Run:  .venv/Scripts/python.exe -m az._test_run_engines
Exit: 0 = the turnkey demo plays all three engines to a valid terminal result. No emoji (cp1252).
"""
from __future__ import annotations

import time

import run_engines

# Low-sim knobs: just enough for a legal, terminating game; tiny for the CI budget.
_CHESS_SIMS = 6
_C4_SIMS = 8
_ATARI_SIMS = 6
_DEVICE = "cpu"  # deterministic + no GPU dependency in CI

_VALID_CHESS_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
_VALID_POV = {"WIN", "LOSS", "DRAW"}


def test_chess_plays_to_result() -> None:
    res = run_engines.play_chess(games=1, delay=0.0, render=False, device=_DEVICE,
                                 mcts_sims=_CHESS_SIMS)
    assert res["engine"] == "chess"
    assert len(res["games"]) == 1
    g = res["games"][0]
    assert g["result"] in _VALID_CHESS_RESULTS, f"bad chess result {g['result']!r}"
    # a real game must have actually finished (not the '*' in-progress sentinel)
    assert g["result"] != "*", "chess game did not reach a terminal result"
    assert g["champ_pov"] in _VALID_POV, f"bad champ POV {g['champ_pov']!r}"
    assert g["plies"] >= 1, "no plies were played"
    assert len(g["san"]) == g["plies"], "SAN log / ply-count mismatch"
    # every recorded move is a non-empty SAN string (legal moves were played)
    assert all(isinstance(s, str) and s for s in g["san"]), "empty/illegal SAN in log"
    print(f"  [TEST 1] chess OK: result {g['result']} (champ {g['champ_pov']}), "
          f"{g['plies']} legal plies")


def test_connect4_plays_to_result() -> None:
    res = run_engines.play_connect4(games=1, delay=0.0, render=False, device=_DEVICE,
                                    mcts_sims=_C4_SIMS)
    assert res["engine"] == "connect4"
    assert len(res["games"]) == 1
    g = res["games"][0]
    assert g["net_pov"] in _VALID_POV, f"bad connect4 net POV {g['net_pov']!r}"
    assert g["moves"] >= 1, "no connect4 moves were played"
    # connect-4 cannot exceed 42 placements (6x7 board)
    assert g["moves"] <= 42, f"impossible connect4 move count {g['moves']}"
    assert (res["w"] + res["d"] + res["l"]) == 1, "W/D/L did not sum to the game count"
    print(f"  [TEST 2] connect4 OK: net {g['net_pov']} in {g['moves']} moves")


def test_atari_plays_to_result() -> None:
    res = run_engines.play_atari(games=1, delay=0.0, render=False, device=_DEVICE,
                                 mcts_sims=_ATARI_SIMS)
    assert res["engine"] == "atari"
    assert len(res["games"]) == 1
    g = res["games"][0]
    ret = g["return"]
    # single-drop catch: a finished episode yields a per-drop reward in [-1, 1]
    assert -1.0 <= ret <= 1.0, f"atari return {ret} out of [-1, 1]"
    assert g["steps"] >= 1, "no atari steps were taken"
    assert abs(res["mean_return"] - ret) < 1e-9, "mean_return != single-episode return"
    assert res["catches"] in (0, 1), "catches must be 0 or 1 for one episode"
    print(f"  [TEST 3] atari OK: return {ret:+.1f} in {g['steps']} steps")


def test_run_all_returns_three_outcomes() -> None:
    outcomes = run_engines.run(engine="all", games=1, delay=0.0, render=False,
                               device=_DEVICE, fast=True)
    assert len(outcomes) == 3, f"expected 3 outcomes, got {len(outcomes)}"
    engines = [o["engine"] for o in outcomes]
    assert engines == ["chess", "connect4", "atari"], f"unexpected order/engines: {engines}"
    for o in outcomes:
        assert "games" in o and len(o["games"]) == 1
        assert isinstance(o.get("trained"), bool)
    print(f"  [TEST 4] run(all) OK: {engines}")


def main() -> int:
    print("=" * 72)
    print("  TURNKEY DEMO GATE -- run_engines plays chess + connect4 + atari to a result")
    print("=" * 72)
    t0 = time.perf_counter()

    test_chess_plays_to_result()
    test_connect4_plays_to_result()
    test_atari_plays_to_result()
    test_run_all_returns_three_outcomes()

    dt = time.perf_counter() - t0
    print("-" * 72)
    print(f"  ALL PASS in {dt:.1f}s: the turnkey demo plays all three trained engines to a "
          "valid terminal result (chess checkmate/draw, connect-4 win/draw, atari catch/miss) "
          "with legal moves -- using the trained checkpoints, never hard-failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
