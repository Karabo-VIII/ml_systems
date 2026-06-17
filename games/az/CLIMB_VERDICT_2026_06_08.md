# Chess AZ — Honest Climb Verdict (overseer RWYB, 2026-06-08 14:45 SAST)

Objective (frontier): confirm the visual self-play engine is **WORKING / LEARNING / EVOLVING**, with
acceptance = live viz renders games; trainer writes a strength_curve with >=2 iters; champion floor
monotonic; honest climb verdict.

## Verdict against the acceptance test

| Acceptance item | Result | Evidence |
|---|---|---|
| live viz renders games | **ARTIFACT PRESENT (rendering unverified headless)** | `selfplay_viz.html` (831 KB, Jun 7) |
| strength_curve >= 2 iters | **PASS** | `strength_curve.json` = 111 iters (0..110) |
| champion floor monotonic | **PARTIAL** | champion = `net_iter106`; wr_vs_random trends up (mean 0.04 first-20 -> 0.15 last-20, max 0.75 @ iter107); flat vs classical |
| honest climb verdict | **see below** | — |

## Honest climb verdict

- **WORKING: YES.** The self-play -> train -> eval -> champion-gate -> viz loop ran 111 iterations
  cleanly and produced checkpoints + a strength curve + an eval ledger.
- **LEARNING (vs random): YES, real.** total_loss 3.384 -> 1.652; large-sample eval of the bootstrap
  net = **70% win vs random (35W/15D/0L / 50 games)**, **100% with MCTS (30/30)**. This is genuine
  strength acquisition, not memorization/collapse. (An earlier glance at the stale `train_run.log`
  tail showed isolated 4-game samples reading "0.000" — that was high-variance noise + a wins-only
  win_rate; RETRACTED. The 50-game eval is authoritative.)
- **EVOLVING: CEILINGED.** Strength rises against the *random* floor but hits a **hard 0% ceiling vs a
  depth-1 classical bot** — 0 wins across all 111 iters AND across the 50/30-game evals. The net learns
  to beat aimless legal play but cannot out-calculate even 1-ply tactical capture.

## Two real defects found (honest, not papered over)

1. **The run is DEAD, the state advertises "live."** `train_run.log` last written 2026-06-07 11:01;
   `train_supervisor.pid`=23932 now resolves to `AppVShNotify` (PID recycled). AUTONOMY_ON + frontier
   were created today 13:57 but **no trainer process is executing the objective.** A supervisor that
   trusts the stale PID would believe a run is alive. (Flagged, not deleted — not this overseer's file.)
2. **Checkpoint bloat: ~76 GB.** `robust_checkpoints/net_iterN.pt` grows 147 MB -> 1.27 GB across iters
   (optimizer/replay state almost certainly serialized into every checkpoint). The run crashed at the
   iter-107 `.tmp` write. This is a disk/IO defect independent of the strength question.

## Concrete levers to break the classical ceiling (NOT run — compute-bound, side-project)

- MCTS sims 32/48 -> 200+ at eval and self-play (1-ply tactics need deeper search than 32 sims give).
- Use the just-added **multiprocess `selfplay_pool` (13.5x)** to afford many more iters/hour.
- Larger net / more iters; verify value targets aren't near-constant from the fixed start position
  (diversify openings so the value head sees decisive games).
- Fix the checkpoint serializer to save weights-only (+ a separate small optimizer file) before any
  long relaunch, or the 76 GB problem repeats.

**Overseer stance:** objective ANSWERED (confirmed working + learning-vs-random; refuted evolving-past-
classical). Frontier nodes empty -> per IDLE-STOP, the honest deliverable is this verdict, NOT a
multi-hour busywork relaunch of a known-ceilinged run. Relaunch is a deliberate compute decision for
the user/next window, with the checkpoint-bloat fix as a precondition.
