"""
chess_zero.az.train_robust -- a HARDENED AlphaZero self-play -> train -> eval loop
built to run UNATTENDED for HOURS on a single RTX 4060 (8GB) without crashing or
hanging, that demonstrably LEARNS, and that exposes a STRENGTH CURVE so a human
can SEE the agent improve.

This is the production-grade sibling of train_demo.py. It reuses the SAME az/
modules (encoding/net/mcts/selfplay) and the SAME AlphaZero loss; it adds the
robustness machinery a multi-hour unattended run needs:

  1. BIGGER NET than the 5-min demo (default C=80 / 8 blocks vs demo C=32/4),
     sized to fit 8GB with headroom. Actual peak VRAM is MEASURED via
     torch.cuda.max_memory_allocated and reported; --vram-cap-gb backs off if a
     forward/backward probe lands too close to the limit.
  2. CHECKPOINT EVERY ITERATION (net + optimizer + iter + RNG states) to
     az/robust_checkpoints/, atomically, plus a `latest.pt` pointer. RESUME from
     the latest checkpoint automatically on restart -- a crash loses <= one iter.
  3. CUDA-OOM GUARD: self-play and train are wrapped for torch.cuda.OutOfMemoryError
     (and generic RuntimeError "out of memory"); on OOM we empty_cache, shrink the
     batch / sims, and CONTINUE -- a transient OOM never kills the run.
  4. NEVER HANG: a per-move MCTS sim cap (MCTS.n_simulations), a per-game ply cap,
     AND a per-game wall-clock guard. A game exceeding its budget is adjudicated
     (material count) / aborted and the loop moves on.
  5. STRENGTH EVAL each iter -> appends a real row to az/strength_curve.json:
     {iter, total_loss, policy_loss, value_loss, winrate_vs_random,
      winrate_vs_classical_d1, n_eval_games, wall_s, ckpt, ...}. The classical
     opponent is engine.py Engine(depth=1). Games are REAL (actually played);
     draws are reported honestly.
  6. CLI: --iters N (ADDITIVE on resume -- runs N MORE iters from the resume point,
     so `--iters 2` is always 2 fresh iters, never a no-op), --max-hours H (clean stop
     when wall-clock exceeds H -- the 6h envelope), --resume (default ON), plus
     self-play/sim/eval knobs.
  7. SUPERVISOR: `--supervise` runs the loop in an in-process auto-restart guard
     that relaunches from the latest checkpoint if a worker iteration dies, so the
     unattended robustness is real end-to-end.

HONEST SCOPE: this is a LEARNING-DYNAMICS build. A 6h/4060 run yields a
weak-but-improving player, NOT a master. The PROOF is the strength curve trending
upward (win-rate vs random should climb fast; vs the classical depth-1 engine
slowly), the loss trending down, and the run surviving hours unattended.

Run:
    # short smoke (proves the machinery):
    .venv\\Scripts\\python.exe -m az.train_robust --iters 2 \\
        --games-per-iter 2 --selfplay-sims 12 --eval-games 4 --eval-sims 8 \\
        --channels 48 --n-blocks 4

    # the real 6h run (supervised, auto-restart):
    .venv\\Scripts\\python.exe -m az.train_robust \\
        --iters 1000 --max-hours 6 --supervise
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess
import torch

from .encoding import N_INPUT_PLANES, N_POLICY, board_to_planes, move_to_index
from .net import AlphaZeroNet, count_params
from .mcts import MCTS
from .selfplay import Sample, train_step
from .openings import sample_opening_board

# engine.py lives one package up (engine). Import it
# relatively so this works under `python -m az.train_robust`.
from chess_engine.engine import Engine
# Pluggable teacher opponent: classical Engine by DEFAULT (zero setup), optional
# real-world UCI engine (e.g. Stockfish) when --engine-path is given.
from .uci_engine import make_opponent


HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RobustConfig:
    # net (BIGGER than the demo's C=32/4; sized for 8GB with headroom)
    channels: int = 80
    n_blocks: int = 8
    # self-play
    iterations: int = 1000
    games_per_iter: int = 8
    selfplay_workers: int = 1         # THROUGHPUT (the BIG lever): N CPU worker PROCESSES generate self-play
                                      # games in parallel (self-play is CPU-bound -> ~Nx faster; measured 13.5x at
                                      # N=16 on 20 cores). Applies to opponent=='self'. 1 = single-process (default).
                                      # Set to ~cores-2 for max throughput. See selfplay_pool.py.
    parallel_games: int = 1           # THROUGHPUT KNOB: when >1 AND opponent=='self', self-play
                                      # games are generated in GPU-BATCHED groups of this size
                                      # (one net forward does up to parallel_games leaf evals ->
                                      # ~Nx fewer GPU round-trips, more games/sec). teacher/mix
                                      # stay sequential (an external engine is not batchable).
                                      # 1 = the unchanged sequential path. See batched_selfplay.py.
    selfplay_sims: int = 64           # per-move MCTS sim cap (NEVER-HANG guard #1)
    temp_moves: int = 20              # plies sampled w/ temperature=1 (exploration)
    max_plies: int = 160              # per-game ply cap (NEVER-HANG guard #2)
    game_wall_s: float = 90.0         # per-game wall-clock cap (NEVER-HANG guard #3)
    # OPENING DIVERSITY (the "vary the starting conditions" lever, 2026-06-09).
    # Every self-play game used to start from the IDENTICAL startpos, so on a peaked
    # imitation net the games funnelled into one rote line and the value head rarely
    # saw diverse decisive positions -> "plays the same way" / bad learned habits.
    # opening_mode starts each SELF-PLAY game from a distinct, sound opening (the
    # opening plies are NOT training samples). EVAL is unaffected (always startpos)
    # so the strength curve stays a comparable yardstick.
    #   "startpos" = old behaviour; "book" = curated sound lines; "random" = guarded
    #   random plies; "mixed" = book + guarded random jitter. See openings.py.
    opening_mode: str = "mixed"       # DEFAULT ON: diverse starts (book + jitter)
    opening_plies: int = 4            # random plies for "random"; jitter plies for "mixed"
    # training
    train_steps_per_iter: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    l2: float = 1e-4
    buffer_size: int = 60000
    # NUMERICAL SAFETY (2026-06-09 pre-restart audit S1/S2/S3). The prod loop previously had
    # NO NaN guard, NO gradient clipping, and equal value-weight -- all silent-failure vectors.
    value_loss_weight: float = 0.5    # <=1 so the large early value-MSE doesn't drown the policy head (E3)
    grad_clip: float = 5.0            # global-norm clip; one outlier batch can else corrupt weights for many steps
    nan_abort: int = 8                # consecutive/accumulated non-finite losses in an iter before RAISE (supervisor restarts)
    # evaluation
    eval_games: int = 30              # split across random + classical (raised 20->30 for less
                                      # binomial noise; the strict monotonic floor is noise-sensitive)
    # EVAL TRUST (2026-06-09 audit S5): the champion's monotonic FLOOR is set once at seed time.
    # A lucky 15-game seed read (95% Wilson CI ~ +/-0.25 at n=15) could lock the floor far above
    # the net's true strength -> every later candidate REJECTs forever. Seed from a LARGER eval so
    # the floor is set on a tight estimate. One-time cost at startup. Per-iter CI is also logged.
    seed_eval_games: int = 120        # games for the ONE-TIME champion-seed eval (split rand/classical)
    # FORGETTING / DRIFT detector (2026-06-09 audit S5b/F11/E1): periodically play the current net
    # vs the FROZEN SEED (net_iter0 / the bootstrap). The net should beat its own starting point
    # (>0.5) and stay there; a SUSTAINED drop below 0.5 means catastrophic forgetting / drift that
    # the vs-random + vs-classical-d1 axes cannot see (E6 -- co-adaptation hides absolute decline).
    forgetting_eval_interval: int = 10  # eval vs the frozen seed every N iters (0 = off)
    forgetting_eval_games: int = 0      # games for it (0 = use eval_games)
    eval_sims: int = 32               # low-sim MCTS for the net at eval time
    eval_max_plies: int = 160
    eval_game_wall_s: float = 60.0
    classical_depth: int = 1          # Engine(depth=1) -- the classical opponent
    # robustness
    max_hours: float = 6.0            # wall-clock envelope; stop cleanly past it
    vram_cap_gb: float = 7.4          # back the net off if a probe exceeds this
    min_sims: int = 8                 # OOM-guard floor on sims
    min_batch: int = 16               # OOM-guard floor on batch
    floor_oom_retries: int = 10       # bounded retries at the OOM floor (min_sims/min_batch)
                                      # before RAISING -- so a PERSISTENT OOM at the floor
                                      # can't livelock the loop (the supervisor only catches
                                      # exceptions; an infinite floor-retry was invisible to
                                      # it). On raise, --supervise restarts the worker.
    # champion gate (TASK 2A: monotonic, never-regress promotion)
    champion_gate: bool = True         # only promote a candidate that does NOT regress
    champion_tol: float = 0.0          # UPSIDE tie-band only; BELOW-champion always rejected
    champion_h2h: bool = False         # if True, ALSO require cand>55% head-to-head vs champion
    champion_h2h_games: int = 20       # games for the optional head-to-head match
    # self-play opponent (TASK 2B: refine vs the TEACHER / anchor so the curve can CLIMB)
    selfplay_opponent: str = "self"    # {"self","teacher","mix"}; teacher = classical Engine
    selfplay_teacher_depth: int = 1    # Engine(depth) for teacher/mix self-play games
    engine_path: str = ""              # OPTIONAL UCI engine binary (e.g. Stockfish) for the
                                       # teacher; "" => our in-repo classical Engine (default)
    uci_movetime_ms: int = 50          # per-move budget for a UCI teacher engine (ms)
    anchor_kl: float = 0.0             # KL(bootstrap||candidate) penalty weight (0 = off)
    teacher_distill: bool = True       # DENSE TEACHER GRADING: in teacher/mix games, ALSO label
                                       # the teacher's chosen move as a one-hot policy target so the
                                       # net imitates the teacher at EVERY teacher move (online
                                       # distillation), not just the sparse game outcome. Default ON
                                       # (strictly more learning signal where the teacher plays;
                                       # self-only games are unaffected). --no-teacher-distill to off.
    # CURRICULUM (the moving target): when ON, bump selfplay_teacher_depth by 1 the FIRST
    # time the candidate's draw-aware score vs the engine AT THE CURRENT teacher depth
    # crosses curriculum_threshold, so the teacher gets harder as the net masters the
    # current depth. Capped at curriculum_max_depth. Default OFF -> behaviour unchanged.
    #
    # CORRECTNESS (3-bug fix 2026-06-08):
    #   (a) LATCH: curriculum_last_bumped_depth records the depth we LAST bumped AT, so the
    #       bump fires ONCE per depth crossing -- not every iter once the threshold is
    #       crossed (the champion's score only ratchets up, so the un-latched gate re-fired
    #       forever and ran the depth straight to the cap in consecutive iters).
    #   (b) RIGHT MEASUREMENT: the advance is gated on a DEDICATED eval of the candidate vs
    #       Engine(depth=selfplay_teacher_depth) -- the CURRENT teacher -- not the fixed
    #       classical_depth=1 eval (which measured depth-1 mastery and advanced on the wrong
    #       signal).
    #   (c) PERSISTED: selfplay_teacher_depth + curriculum_last_bumped_depth ride in the
    #       checkpoint config (load_checkpoint restores them) AND supervise() forwards the
    #       live depth to the restarted child, so curriculum progress survives both a
    #       crash-resume and a --supervise restart.
    curriculum: bool = False
    curriculum_threshold: float = 0.6  # candidate score vs the CURRENT teacher depth to advance
    curriculum_max_depth: int = 4      # cap teacher depth bumps
    # LATCH state (b)/(a): the teacher depth we have ALREADY bumped at (-1 = none yet). The
    # bump only fires when this is < the current selfplay_teacher_depth, so each depth
    # crossing advances exactly once. Persisted in the checkpoint so resume keeps the latch.
    curriculum_last_bumped_depth: int = -1
    # AUTO-BALANCE (2026-06-08): let the system self-determine the THROUGHPUT knobs
    # (selfplay_workers / games_per_iter / train_steps_per_iter) from hardware + the max_hours
    # budget, balanced per unit time, instead of hand-tuning them -- WITHOUT touching the
    # learning-contract knobs above (champion gate / anchor_kl / curriculum / lr / opponent),
    # which stay principled. At startup workers <- cores-2 (capped); each iter the online
    # controller (auto_balance.rebalance) nudges games/steps toward a target iter-time =
    # max_hours/auto_iters_in_budget while holding the replay ratio in band. The floor-OOM
    # backoff still guards VRAM (GPU blow-up); this guards CPU/iter-time bloat. Default OFF
    # (no-op; the explicit CLI params are used as-is). See auto_balance.py.
    auto_balance: bool = False
    auto_iters_in_budget: int = 24     # >= this many champion-gate ATTEMPTS across the window
    auto_replay_ratio: float = 2.0     # gradient-samples / new-samples target (sane band 1-4)
    # io
    ckpt_dir: str = "robust_checkpoints"
    curve_path: str = "strength_curve.json"
    seed: int = 0


# --------------------------------------------------------------------------- #
# CHAMPION (TASK 2A): the best net so far on a fixed yardstick. The self-play
# actor + the playable latest.pt/net_iterN.pt always track the champion, so the
# strength curve for the champion is MONOTONIC (never regresses below bootstrap).
# --------------------------------------------------------------------------- #
@dataclass
class Champion:
    iter: int                          # iter whose net is the current champion
    winrate_vs_random: float
    winrate_vs_classical: float
    loss: float
    # CLIMB axis (2026-06-07): draw-aware score (wins + 0.5*draws)/games vs the
    # classical engine. Progress vs the engine appears as DRAWS first (losing -> drawing),
    # which leaves winrate_vs_classical flat but raises this. The gate's tie-break uses
    # this so draw-progress is promotable. Defaults 0.0 for pre-climb checkpoints.
    score_vs_classical: float = 0.0
    state_dict: dict = field(default_factory=dict, repr=False)  # champion's weights

    def to_payload(self) -> dict:
        return {
            "iter": self.iter,
            "winrate_vs_random": self.winrate_vs_random,
            "winrate_vs_classical": self.winrate_vs_classical,
            "score_vs_classical": self.score_vs_classical,
            "loss": self.loss,
            "state_dict": self.state_dict,
        }

    @classmethod
    def from_payload(cls, d: dict) -> "Champion":
        return cls(
            iter=int(d["iter"]),
            winrate_vs_random=float(d["winrate_vs_random"]),
            winrate_vs_classical=float(d.get("winrate_vs_classical", 0.0)),
            score_vs_classical=float(d.get("score_vs_classical", 0.0)),
            loss=float(d.get("loss", float("nan"))),
            state_dict=d.get("state_dict", {}),
        )


def candidate_beats_champion(cand_wr_random: float, cand_wr_classical: float,
                             cand_loss: float, champ: Champion, tol: float,
                             cand_score_classical: float = 0.0,
                             h2h_winrate: Optional[float] = None,
                             h2h_threshold: float = 0.55) -> Tuple[bool, str]:
    """Decide whether a candidate should be PROMOTED over the champion. Primary
    yardstick is winrate_vs_random (the never-regress contract). Returns
    (promote, reason). If h2h_winrate is given (optional head-to-head match vs the
    champion), the candidate ALSO must clear the published-AZ-style >threshold gate.

    MONOTONICITY CONTRACT (H1/M1 fix, 2026-06-07):
      * STRICT FLOOR: a candidate BELOW the champion on wr_random
        (cand_wr_random < champ.winrate_vs_random - 1e-9) is REJECTED unconditionally --
        it can NEVER be promoted via the classical/loss tie-break. This is what enforces
        "the playable net can never get weaker". (Before the fix, a within-tol-BELOW
        candidate fell through to the tie-break and could be PROMOTED on a strong
        classical score, then recorded its LOWER wr_random as the new floor -- ratcheting
        monotonicity DOWN. e.g. 0.75 vs champ 0.80 with tol=0.10 used to promote.)
      * tol is NOT a regression band. It only widens the UPSIDE tie band: a candidate
        ABOVE the champion but within +tol counts as a "tie" (decided by the classical /
        loss tie-break) rather than a strict win, so we don't promote on a within-noise
        wr_random *increase*. A candidate must still be >= the champion to reach the
        tie-break at all. With default tol=0.0 the gate is strict ">= champion".

    CLIMB axis (2026-06-07): in the wr_random tie band the classical tie-break now uses
    the DRAW-AWARE SCORE (wins + 0.5*draws)/games, NOT win_rate. Empirically the net
    stalls at winrate_vs_classical == 0.0 for 100+ iters because progress vs the stronger
    engine first appears as DRAWS (losing -> drawing), which win_rate cannot see -- so the
    old win_rate tie-break never promoted and the net never climbed. score_vs_classical
    rising (e.g. 0.0 -> 0.25 -> 0.5 as losses become draws) is REAL, measurable progress
    and now triggers a PROMOTE. This REPLACES the old wr_classical(win-rate) tie-break.
    The strict wr_random floor is untouched: the playable net still never regresses vs
    random. A LOWER score_vs_classical in the tie band -> HOLD (classical regression).

      * SATURATED-TIE HOLD (M1): when wr_random AND score_vs_classical are BOTH tied,
        train loss is NOT a strength signal -- so we HOLD the champion (return False)
        unless the optional head-to-head gate is enabled AND already passed (h2h_winrate
        not None and >= threshold, vetted above). Loss is a tie-break ONLY in conjunction
        with a passed h2h gate; it can never promote on its own."""
    # head-to-head veto first (if enabled): a candidate that does not out-play the
    # champion head-to-head is never promoted, regardless of vs-random numbers.
    if h2h_winrate is not None and h2h_winrate < h2h_threshold:
        return False, (f"h2h {h2h_winrate:.3f} < {h2h_threshold:.2f} vs champion")
    # MONOTONIC FLOOR (H1): the tie-break (classical / loss) branch is reached ONLY when
    # the candidate is effectively a TRUE TIE or ABOVE on wr_random, i.e.
    # cand_wr_random >= champ.winrate_vs_random - eps. A candidate that is BELOW the
    # champion -- even if it is within `tol` of it -- is REJECTED here and can NEVER be
    # promoted via the classical/loss tie-break (which is what used to ratchet the floor
    # DOWN: a 0.75 vs 0.80 candidate with a strong classical score was promoted and then
    # recorded 0.75 as the new floor). `tol` widens the +/-band that counts as a "tie"
    # for the purpose of NOT promoting on a within-noise wr_random *increase*, but it is
    # NOT a regression band: a below-champion candidate is rejected regardless of tol.
    eps = 1e-9
    if cand_wr_random < champ.winrate_vs_random - eps:
        return False, (f"wr_random {cand_wr_random:.3f} < champ {champ.winrate_vs_random:.3f} "
                       f"-- REJECT (below monotonic floor)")
    # strictly above the champion (beyond tol) on the primary yardstick -> promote.
    if cand_wr_random > champ.winrate_vs_random + tol + eps:
        return True, (f"wr_random {cand_wr_random:.3f} > champ {champ.winrate_vs_random:.3f}")
    # CLIMB TIE-BREAK: cand is >= champ (never below) but within +tol of it on wr_random.
    # The DRAW-AWARE SCORE vs the classical engine decides. A higher score (more wins OR
    # more draws -- i.e. fewer losses) is real strength progress -> promote. A lower score
    # is a classical regression -> hold. (This replaces the old win_rate tie-break, which
    # was blind to losing->drawing progress and left the curve stuck at wr_classical=0.)
    if cand_score_classical > champ.score_vs_classical + eps:
        return True, (f"wr_random tied ({cand_wr_random:.3f}); score_vs_classical "
                      f"{cand_score_classical:.3f} > {champ.score_vs_classical:.3f} (CLIMB)")
    if cand_score_classical < champ.score_vs_classical - eps:
        return False, (f"wr_random tied; score_vs_classical {cand_score_classical:.3f} < "
                       f"{champ.score_vs_classical:.3f} -- HOLD champion")
    # SATURATED TIE (M1): both wr_random and score_vs_classical tied. Train loss is NOT a
    # strength signal, so promoting on lower loss alone would degenerate the gate to
    # loss-chasing. HOLD the champion -- UNLESS the optional h2h gate is enabled AND the
    # candidate passed it (h2h_winrate is not None here means h2h ran and cleared thr).
    if h2h_winrate is None:
        return False, (f"wr_random+score_vs_classical tied ({cand_wr_random:.3f}/"
                       f"{cand_score_classical:.3f}); loss is NOT a strength signal "
                       f"-- HOLD champion (enable --champion-h2h to break ties)")
    # h2h is ON and already passed the threshold (vetoed above) -> loss may break the tie.
    champ_loss = champ.loss
    if not (champ_loss == champ_loss):  # champ loss is NaN
        return True, f"wr tied; h2h {h2h_winrate:.3f} passed; champ loss NaN -> accept"
    if cand_loss < champ_loss - 1e-9:
        return True, (f"wr tied; h2h {h2h_winrate:.3f} passed; loss {cand_loss:.4f} "
                      f"< champ {champ_loss:.4f}")
    return False, (f"wr tied; h2h {h2h_winrate:.3f} passed but loss {cand_loss:.4f} "
                   f">= champ {champ_loss:.4f} -- no improvement")


# --------------------------------------------------------------------------- #
# RNG state capture / restore (for bit-resumable checkpoints)
# --------------------------------------------------------------------------- #
def _capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["torch_cuda"]])
    except Exception as e:  # never let RNG-restore failure crash a resume
        print(f"[resume] WARN: RNG restore failed ({e}); continuing with fresh RNG")


def _as_byte_tensor(x) -> torch.Tensor:
    """torch.get_rng_state() returns a ByteTensor; torch.load may give it back as
    a plain tensor -- coerce to the dtype set_rng_state expects."""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    return t.to(torch.uint8).cpu()


# --------------------------------------------------------------------------- #
# Checkpoint I/O (atomic write + latest pointer + auto-resume)
# --------------------------------------------------------------------------- #
def _atomic_torch_save(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on Windows + POSIX


def _prune_old_checkpoints(ckpt_dir: str, keep_last: int = 3) -> None:
    """Bound disk over a long run. Post-2026-06-08 each net_iterN.pt is WEIGHTS-ONLY (~42MB);
    the big ephemeral state (optimizer + replay buffer, which used to bloat every file to
    ~700MB-and-growing) now lives in ONE rolling `train_state.pt` sidecar. Pruning still keeps
    net_iter0 (the bootstrap seed) + the last `keep_last` net_iterN.pt; champion.pt + latest.pt
    + train_state.pt preserve the champion + warm-resume target, so older net_iterN.pt are safe
    to delete. Best-effort: never crashes a save."""
    import glob
    import re
    try:
        pairs = []
        for p in glob.glob(os.path.join(ckpt_dir, "net_iter*.pt")):
            m = re.search(r"net_iter(\d+)\.pt$", os.path.basename(p))
            if m:
                pairs.append((int(m.group(1)), p))
        pairs.sort(key=lambda x: x[0])
        if len(pairs) <= keep_last + 1:
            return
        keep = {pairs[0][1]} | {p for _, p in pairs[-keep_last:]}  # net_iter0 + last keep_last
        for _, p in pairs:
            if p not in keep:
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception:
        pass


def _sweep_stale_tmp(ckpt_dir: str) -> int:
    """Remove orphan `*.pt.tmp` files left by an atomic write that crashed mid-flush (e.g. the
    observed net_iter107.pt.tmp from a hard kill during a ~1GB checkpoint write). These are never
    valid (a clean write os.replace()s the .tmp to its final name), so any surviving .tmp is dead
    weight. Best-effort; never crashes a run. Returns the count removed."""
    import glob
    n = 0
    try:
        for p in glob.glob(os.path.join(ckpt_dir, "*.tmp")):
            try:
                os.remove(p)
                n += 1
            except OSError:
                pass
    except Exception:
        pass
    return n


def _save_train_state(ckpt_dir: str, it: int, optimizer, buffer: List[Sample]) -> str:
    """Rolling sidecar (`train_state.pt`) for the BIG ephemeral resume state: optimizer +
    replay buffer. This is the bloat fix (2026-06-08): these two used to be re-pickled into
    EVERY net_iterN.pt, so each file grew to ~700MB (buffer ~1KB/sample x up to 60k samples)
    and, with keep_last=3, a multi-hour run carried multiple GB of duplicated buffer on disk
    + a slow ~1GB write every iter + a crash-risk mid-write .tmp. ONE file, overwritten
    atomically each iter, fully decouples per-iter checkpoint size from the buffer. The buffer
    is REGENERABLE self-play data, so a crash that loses it just re-warms via self-play -- the
    net weights (the precious state) live in the small net_iterN.pt + champion.pt."""
    payload = {
        "iter": it,
        "optimizer": optimizer.state_dict(),
        # plain arrays (~1.4KB/sample of planes+pi); the only large item, now isolated here
        "buffer": [(s.planes, s.pi, s.player, s.z) for s in buffer],
    }
    path = os.path.join(ckpt_dir, "train_state.pt")
    _atomic_torch_save(payload, path)
    return path


def save_checkpoint(ckpt_dir: str, it: int, net, optimizer, cfg: RobustConfig,
                    buffer: List[Sample],
                    champion: Optional["Champion"] = None,
                    weights_state_dict: Optional[dict] = None,
                    promoted: Optional[bool] = None) -> str:
    """Persist everything needed to resume at iteration it+1. Post-2026-06-08 this is SPLIT:
    `net_iter{it}.pt` carries the SMALL, precious, per-iter state (weights + iter + RNG +
    config + champion metadata, ~42MB), while the BIG ephemeral state (optimizer + replay
    buffer) goes to ONE rolling `train_state.pt` via `_save_train_state` (the bloat fix).
    train_state.pt is written FIRST so its iter is always >= the net's (load trusts the
    sidecar only when ts.iter >= net.iter; else it falls back to inline keys).

    CHAMPION GATE (TASK 2A): `weights_state_dict`, if given, is the state_dict that
    becomes net_iter{it}.pt's PLAYABLE weights (the champion's weights on a REJECT,
    the candidate's on a PROMOTE). This keeps net_iter{it}.pt + the latest.pt pointer
    MONOTONIC -- a rejected (degraded) candidate is never written as the playable net.
    The in-training `net`'s weights are NOT used for the file when weights_state_dict
    is supplied. `champion` metadata is persisted inline (with the champion's own
    state_dict) so a resume can always reconstruct the self-play actor = champion,
    independent of which net_iterN.pt files survive on disk."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = weights_state_dict if weights_state_dict is not None else net.state_dict()
    # (1) big ephemeral state -> rolling sidecar FIRST (so ts.iter >= net.iter on a clean write)
    _save_train_state(ckpt_dir, it, optimizer, buffer)
    # (2) small per-iter net checkpoint (NO optimizer, NO buffer -> ~42MB, un-bloated)
    payload = {
        "iter": it,
        "channels": cfg.channels,
        "n_blocks": cfg.n_blocks,
        "state_dict": state_dict,
        "rng": _capture_rng(),
        "config": asdict(cfg),
        "promoted": promoted,
    }
    if champion is not None:
        payload["champion"] = champion.to_payload()
    path = os.path.join(ckpt_dir, f"net_iter{it}.pt")
    _atomic_torch_save(payload, path)
    # update the latest-pointer atomically too
    _atomic_torch_save({"iter": it, "path": os.path.basename(path)},
                       os.path.join(ckpt_dir, "latest.json.tmp.pt"))
    os.replace(os.path.join(ckpt_dir, "latest.json.tmp.pt"),
               os.path.join(ckpt_dir, "latest.pt"))
    _prune_old_checkpoints(ckpt_dir, keep_last=3)  # bound disk (champion.pt + train_state.pt + latest.pt preserve resume)
    return path


def save_champion_sidecar(ckpt_dir: str, champion: "Champion") -> str:
    """Persist the champion to a standalone sidecar file (`champion.pt`), atomically.

    M3 FIX (2026-06-07): the seeded champion used to live only in memory until the
    first PROMOTE wrote it inline into a net_iterN.pt. A crash/restart BEFORE that
    first promote re-seeded the champion from a fresh, noisy winrate_vs_random eval --
    so the monotonic floor could silently drift between runs. Writing the champion to
    its own sidecar the instant it is seeded means a resume restores the SAME floor.
    The sidecar carries the net size so the resume can reconstruct champion_net."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = champion.to_payload()
    _atomic_torch_save(payload, os.path.join(ckpt_dir, "champion.pt"))
    return os.path.join(ckpt_dir, "champion.pt")


def load_champion_sidecar(ckpt_dir: str) -> Optional["Champion"]:
    """Restore the champion from `champion.pt` if present, else None."""
    path = os.path.join(ckpt_dir, "champion.pt")
    if not os.path.exists(path):
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return Champion.from_payload(d)
    except Exception as e:
        print(f"[champion] WARN: sidecar restore failed ({e}); will re-seed")
        return None


# --------------------------------------------------------------------------- #
# INSTANCE LOCK (FIX 4): refuse to start a 2nd trainer on the same ckpt-dir. Two
# trainers sharing a dir race on strength_curve.json + champion.pt + latest.pt and
# silently corrupt each other's state (lost-update on the curve, champion divergence).
# An EXCLUSIVE O_CREAT|O_EXCL lockfile carrying the owner pid makes the 2nd launch a
# clean, loud refusal. A STALE lock (owner pid dead -- e.g. after a hard crash) is
# reclaimed so a crashed run never wedges the dir forever.
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    """True if a process with this pid is currently alive (cross-platform)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        # Windows: OpenProcess via ctypes; a 0 handle => not alive (or access denied,
        # which for our own-spawned trainers won't happen). Fall back to tasklist-free
        # ctypes probe so we don't shell out.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        # distinguish "alive" from "zombie/exited but handle still openable": check exit code
        STILL_ACTIVE = 259
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        if not ok:
            return True  # couldn't query -> assume alive (conservative: don't reclaim)
        return exit_code.value == STILL_ACTIVE
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but not ours -> alive


def acquire_instance_lock(ckpt_dir: str) -> str:
    """Acquire an EXCLUSIVE lock on ckpt_dir via `<ckpt_dir>/.train.lock` (O_CREAT|O_EXCL).
    Writes the owner pid. If the lock already exists AND its owner pid is ALIVE, print a
    clear error and EXIT (refuse to start a 2nd trainer on the same dir). If the owner pid
    is DEAD (stale lock from a hard crash), reclaim it. Returns the lock path; release with
    release_instance_lock() on clean exit."""
    os.makedirs(ckpt_dir, exist_ok=True)
    lock_path = os.path.join(ckpt_dir, ".train.lock")
    for _ in range(2):  # at most: try, reclaim-stale, try once more
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            print(f"[lock] acquired instance lock {os.path.basename(lock_path)} "
                  f"(pid {os.getpid()})")
            return lock_path
        except FileExistsError:
            # someone holds it -- is the owner alive or is this a stale lock?
            owner_pid = -1
            try:
                with open(lock_path, "r") as f:
                    owner_pid = int((f.read() or "-1").strip() or "-1")
            except Exception:
                owner_pid = -1
            if _pid_alive(owner_pid):
                print(f"[lock] ERROR: another trainer (pid {owner_pid}) already holds "
                      f"{lock_path}.\n"
                      f"[lock] Refusing to start a 2nd trainer on the same ckpt-dir "
                      f"(would corrupt strength_curve.json + champion.pt). "
                      f"Use a different --ckpt-dir or stop the running trainer.")
                sys.exit(3)
            # stale: owner dead -> reclaim
            print(f"[lock] stale lock from dead pid {owner_pid} -- reclaiming "
                  f"{os.path.basename(lock_path)}")
            try:
                os.remove(lock_path)
            except OSError:
                pass
            # loop and retry the O_EXCL create
    # extremely unlikely: a race re-created the lock between remove and retry
    print(f"[lock] ERROR: could not acquire {lock_path} after reclaim attempt -- aborting.")
    sys.exit(3)


def release_instance_lock(lock_path: Optional[str]) -> None:
    """Release (unlink) the instance lock if WE own it. Best-effort; never raises."""
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            owner_pid = -1
            try:
                with open(lock_path, "r") as f:
                    owner_pid = int((f.read() or "-1").strip() or "-1")
            except Exception:
                owner_pid = -1
            if owner_pid == os.getpid() or owner_pid == -1:
                os.remove(lock_path)
                print(f"[lock] released instance lock {os.path.basename(lock_path)}")
    except Exception:
        pass


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the path to the newest valid iter checkpoint, or None."""
    latest_ptr = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ptr):
        try:
            ptr = torch.load(latest_ptr, map_location="cpu", weights_only=False)
            cand = os.path.join(ckpt_dir, ptr["path"])
            if os.path.exists(cand):
                return cand
        except Exception:
            pass  # fall through to glob scan
    # fall back: highest iter number among net_iter*.pt
    cands = glob.glob(os.path.join(ckpt_dir, "net_iter*.pt"))
    if not cands:
        return None
    def _iter_of(p):
        try:
            return int(os.path.basename(p).split("net_iter")[1].split(".pt")[0])
        except Exception:
            return -1
    return max(cands, key=_iter_of)


def load_checkpoint(path: str, net, optimizer, device,
                    cfg: Optional["RobustConfig"] = None
                    ) -> Tuple[int, List[Sample], Optional[Champion]]:
    """Restore net + optimizer + RNG + buffer (+ champion) from a checkpoint.
    Returns (last_completed_iter, buffer, champion). strict=False on the net for
    schema tolerance. champion is None for pre-gate checkpoints (back-compat); the
    caller seeds the champion from this checkpoint in that case.

    CURRICULUM PERSISTENCE (bug-c fix 2026-06-08): when `cfg` is given AND curriculum
    is ON, restore the CURRICULUM PROGRESS (selfplay_teacher_depth +
    curriculum_last_bumped_depth) from the checkpoint's saved config so a resume (and,
    via supervise() forwarding, a --supervise restart) keeps the climbed teacher depth
    instead of resetting to the CLI default. Only these two fields are pulled back --
    everything else stays as the live cfg (so CLI overrides on resume still apply)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if cfg is not None and getattr(cfg, "curriculum", False):
        saved = ck.get("config") or {}
        # back-compat: pre-fix checkpoints lack curriculum_last_bumped_depth -> default -1.
        saved_depth = saved.get("selfplay_teacher_depth")
        saved_latch = saved.get("curriculum_last_bumped_depth", -1)
        if saved_depth is not None:
            cfg.selfplay_teacher_depth = int(saved_depth)
            cfg.curriculum_last_bumped_depth = int(saved_latch)
            print(f"[curriculum] resumed teacher depth {cfg.selfplay_teacher_depth} "
                  f"(latched at {cfg.curriculum_last_bumped_depth}) from checkpoint config")
    net.load_state_dict(ck["state_dict"], strict=False)
    # --- optimizer + buffer: prefer the rolling train_state.pt sidecar (new split format);
    #     fall back to the INLINE keys for old-format net_iterN.pt (pre-2026-06-08, written by
    #     a still-running old-code trainer) AND for bootstrap_supervised.py checkpoints (which
    #     carry an empty/absent buffer + no train_state). Trust the sidecar only when its iter
    #     is >= the net's, so a partial write never pairs a stale buffer with a newer net. ---
    opt_state = None
    buf_raw = None
    ts_path = os.path.join(os.path.dirname(path), "train_state.pt")
    if os.path.exists(ts_path):
        try:
            ts = torch.load(ts_path, map_location=device, weights_only=False)
            if int(ts.get("iter", -1)) >= int(ck["iter"]):
                opt_state = ts.get("optimizer")
                buf_raw = ts.get("buffer")
            else:
                print(f"[resume] train_state.pt iter {ts.get('iter')} < net iter {ck['iter']} "
                      f"-- using inline fallback")
        except Exception as e:
            print(f"[resume] WARN: train_state.pt load failed ({e}); using inline fallback")
    if opt_state is None:
        opt_state = ck.get("optimizer")           # old-format / bootstrap inline optimizer
    if buf_raw is None:
        buf_raw = ck.get("buffer", [])            # old-format inline buffer (empty for bootstrap)
    if optimizer is not None and opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"[resume] WARN: optimizer state restore failed ({e}); using fresh optimizer")
    if "rng" in ck:
        _restore_rng(ck["rng"])
    buffer: List[Sample] = []
    for planes, pi, player, z in (buf_raw or []):
        buffer.append(Sample(planes=np.asarray(planes, dtype=np.float32),
                             pi=np.asarray(pi, dtype=np.float32),
                             player=bool(player), z=float(z)))
    champion: Optional[Champion] = None
    if isinstance(ck.get("champion"), dict):
        try:
            champion = Champion.from_payload(ck["champion"])
        except Exception as e:
            print(f"[resume] WARN: champion restore failed ({e}); will re-seed champion")
    return int(ck["iter"]), buffer, champion


# --------------------------------------------------------------------------- #
# Self-play with the never-hang guards (sim cap + ply cap + wall-clock cap)
# --------------------------------------------------------------------------- #
def _material_balance(board: chess.Board) -> int:
    """Cheap centipawn material balance from White's perspective (for adjudication
    of a game that hit its wall-clock/ply cap)."""
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    bal = 0
    for sq, piece in board.piece_map().items():
        v = vals[piece.piece_type]
        bal += v if piece.color == chess.WHITE else -v
    return bal


def _adjudicate(board: chess.Board) -> Optional[bool]:
    """Adjudicate an unfinished game by material. Returns winner colour
    (chess.WHITE/BLACK) or None for a 'too close to call' -> draw."""
    bal = _material_balance(board)
    if bal >= 150:       # ~1.5 pawns up = call it for White
        return chess.WHITE
    if bal <= -150:
        return chess.BLACK
    return None


def _move_from_visits(visits: Dict[chess.Move, int], temperature: float) -> chess.Move:
    """Pick the played move from an ALREADY-computed MCTS visit-count dict, using the
    exact same rule as MCTS.best_move (argmax for temperature<=1e-6, else sample
    proportional to N^(1/temperature)). M2 FIX (2026-06-07): self-play used to call
    mcts.run() for the pi target and then mcts.best_move(), which runs mcts.run() a
    SECOND time -- doubling the per-move search cost. Sampling the played move from the
    same `visits` halves self-play cost with identical semantics (same distribution,
    same sampling distribution on the temperature>0 path). It is ALSO strictly more
    correct: the played move now comes from the EXACT search whose visit counts became
    the pi training target, instead of a fresh re-search with new Dirichlet noise that
    could disagree with the labelled pi."""
    moves = list(visits.keys())
    counts = np.array([visits[m] for m in moves], dtype=np.float64)
    if len(moves) == 0 or counts.sum() == 0:
        return moves[0] if moves else None
    if temperature <= 1e-6:
        return moves[int(counts.argmax())]
    probs = counts ** (1.0 / temperature)
    probs = probs / probs.sum()
    return moves[int(np.random.choice(len(moves), p=probs))]


def generate_selfplay_game_guarded(net, n_simulations: int, temp_moves: int,
                                   max_plies: int, game_wall_s: float,
                                   device, opponent: str = "self",
                                   teacher_depth: int = 1,
                                   net_is_white: bool = True,
                                   engine_path: str = "",
                                   uci_movetime_ms: int = 50,
                                   distill_teacher: bool = True,
                                   opening_mode: str = "startpos",
                                   opening_plies: int = 4,
                                   opening_seed: Optional[int] = None) -> List[Sample]:
    """Like selfplay.generate_selfplay_game but with a PER-GAME WALL-CLOCK guard
    on top of the per-move sim cap and the per-game ply cap. If the game exceeds
    its wall-clock budget OR ply cap unfinished, it is adjudicated by material so
    z is still a real, signed training target. NEVER hangs on a single game.

    TASK 2B -- opponent in {'self','teacher'}:
      'self'    : the net (MCTS) plays BOTH sides (classic AlphaZero self-play).
      'teacher' : the net (MCTS) plays one colour; the TEACHER opponent plays the
                  other -- so the training targets come from STRONG games (learning
                  from the teacher) instead of weak net-vs-net drift. The teacher is
                  built by make_opponent(engine_path, classical_depth=teacher_depth):
                  our in-repo classical Engine by DEFAULT (engine_path=""), or an
                  OPTIONAL real-world UCI engine (e.g. Stockfish) when engine_path is
                  given. A bad UCI path WARNs and falls back to classical -- never
                  crashes the game loop.
    For 'teacher'/'mix' games we emit training Samples on the NET's moves (the pi targets
    are the net's own MCTS distribution) AND -- when distill_teacher=True (default) -- on the
    TEACHER's moves too, labelled with a ONE-HOT of the teacher's chosen move. That is DENSE
    TEACHER GRADING (online distillation): the net imitates the teacher's policy at every
    teacher move, not just learning from the sparse game outcome. It is the same supervised
    signal the bootstrap uses, applied ONLINE during refinement so teacher knowledge keeps
    flowing in (the curriculum then raises the teacher depth; the champion gate guards
    regression; anchor-kl keeps it near the strong base). z is the real game outcome for every
    sample, so the value head learns from the full strong game regardless."""
    mcts = MCTS(net, n_simulations=n_simulations, device=device)
    # Teacher opponent (only for 'teacher' games): classical Engine by default,
    # optional UCI engine when engine_path is given. make_opponent never raises --
    # a bad UCI path WARNs and falls back to classical.
    teacher = (make_opponent(engine_path=engine_path, classical_depth=teacher_depth,
                             uci_movetime_ms=uci_movetime_ms)
               if opponent == "teacher" else None)
    # OPENING DIVERSITY: start from a distinct sound opening (book/random/mixed) so
    # self-play / teacher games don't all begin from the same startpos. The opening
    # plies are NOT recorded as samples (the loop only appends from here on). EVAL
    # paths never call this with a non-startpos mode, so the yardstick is unaffected.
    if opening_mode != "startpos":
        board = sample_opening_board(np.random.default_rng(opening_seed),
                                     mode=opening_mode, random_plies=opening_plies)
    else:
        board = chess.Board()
    samples: List[Sample] = []
    ply = 0
    t0 = time.time()
    aborted = False

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if time.time() - t0 > game_wall_s:
            aborted = True
            break
        net_to_move = (opponent == "self") or ((board.turn == chess.WHITE) == net_is_white)
        temperature = 1.0 if ply < temp_moves else 0.0
        if net_to_move:
            visits = mcts.run(board, add_noise=(temperature > 0))
            # inline _visits_to_pi (kept identical to selfplay) so we don't import a private
            pi = np.zeros(N_POLICY, dtype=np.float32)
            total = sum(visits.values())
            if total > 0:
                for mv, n in visits.items():
                    idx = move_to_index(board, mv)
                    if idx is not None:
                        pi[idx] = n / total
            samples.append(Sample(planes=board_to_planes(board), pi=pi, player=board.turn))
            # M2: reuse the SAME `visits` already computed for the pi target -- do NOT
            # call mcts.best_move() (which would run a SECOND mcts.run search).
            move = _move_from_visits(visits, temperature)
        else:  # teacher (classical/UCI engine) move
            move = teacher.select_move(board)
            if move is None or move not in board.legal_moves:
                move = next(iter(board.legal_moves))
            if distill_teacher:
                # DENSE TEACHER GRADING (online distillation): label THIS position with the
                # teacher's chosen move as a one-hot policy target so the net imitates the
                # teacher at every teacher move (not just the sparse game outcome). z is filled
                # in below for every sample, so this also feeds the value head.
                t_idx = move_to_index(board, move)
                if t_idx is not None:
                    t_pi = np.zeros(N_POLICY, dtype=np.float32)
                    t_pi[t_idx] = 1.0
                    samples.append(Sample(planes=board_to_planes(board), pi=t_pi,
                                          player=board.turn))
        board.push(move)
        ply += 1

    # Determine winner: real result if finished, else adjudicate by material.
    if board.is_game_over(claim_draw=True) and not aborted:
        result = board.result(claim_draw=True)
        winner = (chess.WHITE if result == "1-0"
                  else chess.BLACK if result == "0-1" else None)
    else:
        winner = _adjudicate(board)  # ply/wall-clock cap hit

    for s in samples:
        s.z = 0.0 if winner is None else (1.0 if s.player == winner else -1.0)
    # Release the teacher engine (no-op for classical; engine.quit() for UCI).
    if teacher is not None:
        teacher.close()
    return samples


# --------------------------------------------------------------------------- #
# Evaluation vs RANDOM and vs the CLASSICAL engine (Engine(depth=1))
# --------------------------------------------------------------------------- #
def _random_move(board: chess.Board, rng: np.random.Generator) -> chess.Move:
    moves = list(board.legal_moves)
    return moves[int(rng.integers(len(moves)))]


def _play_match(net, opponent: str, n_games: int, eval_sims: int, max_plies: int,
                game_wall_s: float, classical_depth: int, device,
                rng: np.random.Generator, n_workers: int = 1) -> Dict[str, float]:
    """Play n_games of (net, greedy MCTS) vs an opponent. opponent in
    {'random','classical'}. Net alternates colour each game. Returns
    {win, draw, loss, games, win_rate, score} from the NET's perspective. Honest draws.

    PARALLEL EVAL (2026-06-08): n_workers>1 splits the games across CPU worker processes (each runs
    THIS same function with n_workers=1 on its chunk) and aggregates -- eval was the 2nd-biggest
    per-iter cost (~150s/8 games). Only kicks in for n_games>=4 (so spawn overhead is amortized);
    any pool failure falls back to the sequential path below.

    DRAW-AWARE SCORE (CLIMB axis, 2026-06-07): `score = (wins + 0.5*draws) / games`
    is the standard chess score. Progress vs a STRONGER opponent (the classical
    engine) first shows up as losing -> DRAWING it, which leaves win_rate flat at 0.0
    but raises score from 0.0 toward 0.5. The champion gate's climb tie-break uses
    `score` (not win_rate) so this draw-progress is MEASURABLE + promotable. win_rate
    is kept unchanged (the strict monotonic floor vs random still uses it).

    A game that hits its ply/wall-clock cap unfinished is adjudicated by material
    (so eval also NEVER hangs)."""
    if n_workers > 1 and n_games >= 4:
        try:
            from .selfplay_pool import play_match_parallel
            return play_match_parallel(net, opponent, n_games, eval_sims, max_plies, game_wall_s,
                                       classical_depth, n_workers=n_workers,
                                       channels=getattr(net, "channels", None),
                                       n_blocks=getattr(net, "n_blocks", None))
        except Exception as e:
            print(f"[eval-pool] FAILED ({type(e).__name__}: {e}) -> sequential eval this call")
    mcts = MCTS(net, n_simulations=eval_sims, device=device)
    engine = Engine(depth=classical_depth) if opponent == "classical" else None
    wins = draws = losses = 0
    net.eval()
    for g in range(n_games):
        net_is_white = (g % 2 == 0)
        board = chess.Board()
        ply = 0
        t0 = time.time()
        aborted = False
        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            if time.time() - t0 > game_wall_s:
                aborted = True
                break
            net_to_move = (board.turn == chess.WHITE) == net_is_white
            if net_to_move:
                move = mcts.best_move(board, temperature=0.0)  # greedy
            elif opponent == "random":
                move = _random_move(board, rng)
            else:  # classical
                res = engine.search(board)
                move = res.move if res.move is not None else _random_move(board, rng)
            board.push(move)
            ply += 1

        if board.is_game_over(claim_draw=True) and not aborted:
            result = board.result(claim_draw=True)
            winner = (chess.WHITE if result == "1-0"
                      else chess.BLACK if result == "0-1" else None)
        else:
            winner = _adjudicate(board)
        if winner is None:
            draws += 1
        elif (winner == chess.WHITE) == net_is_white:
            wins += 1
        else:
            losses += 1
    return {"win": wins, "draw": draws, "loss": losses, "games": n_games,
            "win_rate": wins / n_games if n_games else 0.0,
            "score": (wins + 0.5 * draws) / n_games if n_games else 0.0}


def _play_head_to_head(net_a, net_b, n_games: int, eval_sims: int, max_plies: int,
                       game_wall_s: float, device) -> Dict[str, float]:
    """Play n_games of net_a (greedy MCTS) vs net_b (greedy MCTS), alternating
    colours. Returns {win,draw,loss,games,win_rate} from net_a's perspective. Used
    for the optional published-AZ-style >55% head-to-head champion gate (TASK 2A)."""
    mcts_a = MCTS(net_a, n_simulations=eval_sims, device=device)
    mcts_b = MCTS(net_b, n_simulations=eval_sims, device=device)
    wins = draws = losses = 0
    net_a.eval(); net_b.eval()
    for g in range(n_games):
        a_is_white = (g % 2 == 0)
        board = chess.Board()
        ply = 0
        t0 = time.time()
        aborted = False
        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            if time.time() - t0 > game_wall_s:
                aborted = True
                break
            a_to_move = (board.turn == chess.WHITE) == a_is_white
            mcts = mcts_a if a_to_move else mcts_b
            board.push(mcts.best_move(board, temperature=0.0))
            ply += 1
        if board.is_game_over(claim_draw=True) and not aborted:
            result = board.result(claim_draw=True)
            winner = (chess.WHITE if result == "1-0"
                      else chess.BLACK if result == "0-1" else None)
        else:
            winner = _adjudicate(board)
        if winner is None:
            draws += 1
        elif (winner == chess.WHITE) == a_is_white:
            wins += 1
        else:
            losses += 1
    return {"win": wins, "draw": draws, "loss": losses, "games": n_games,
            "win_rate": wins / n_games if n_games else 0.0}


# --------------------------------------------------------------------------- #
# VRAM probe: measure a forward+backward at full batch; back the net off if it
# exceeds the cap. Returns (channels, n_blocks, peak_gb).
# --------------------------------------------------------------------------- #
def probe_and_size_net(cfg: RobustConfig, device) -> Tuple[int, int, float]:
    if device.type != "cuda":
        return cfg.channels, cfg.n_blocks, 0.0
    channels, n_blocks = cfg.channels, cfg.n_blocks
    for attempt in range(6):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
            x = torch.randn(cfg.batch_size, N_INPUT_PLANES, 8, 8, device=device)
            logits, value = net(x)
            loss = logits.float().mean() + value.float().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            del net, opt, x, logits, value, loss
            torch.cuda.empty_cache()
            if peak_gb <= cfg.vram_cap_gb:
                return channels, n_blocks, peak_gb
            # too big: back off (drop a block, then channels)
            print(f"[vram] probe C={channels}/B={n_blocks} peak={peak_gb:.2f}GB "
                  f"> cap {cfg.vram_cap_gb}GB -- backing off")
            if n_blocks > 4:
                n_blocks -= 1
            else:
                channels = max(32, channels - 16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[vram] OOM at C={channels}/B={n_blocks} -- backing off")
            if n_blocks > 4:
                n_blocks -= 1
            else:
                channels = max(32, channels - 16)
    return channels, n_blocks, float("nan")


# --------------------------------------------------------------------------- #
# OOM-guarded helpers around self-play and training
# --------------------------------------------------------------------------- #
def _is_oom(err: Exception) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def train_step_anchored(net, optimizer, batch: List[Sample], device, l2: float,
                        anchor_net, anchor_kl: float,
                        value_loss_weight: float = 1.0, grad_clip: float = 5.0):
    """Local variant of selfplay.train_step that ADDS a KL anchor toward a frozen
    reference policy (the bootstrap/champion): + anchor_kl * KL(anchor || candidate).
    This is the TASK 2B regularizer that keeps the policy from drifting off the strong
    base during self-play refinement. Returns (total_loss, policy_loss, value_loss, grad_norm).
    Kept INSIDE train_robust.py so selfplay.py is untouched; falls back to the plain
    loss when anchor_kl <= 0 or anchor_net is None.

    NUMERICAL SAFETY (2026-06-09): identical guard to selfplay.train_step -- never backprop a
    non-finite loss (silent weight corruption); clip the global grad norm; return the pre-clip
    norm for monitoring; value_loss_weight (<=1) protects the policy head from value-MSE dominance."""
    import torch.nn.functional as F

    net.train()
    planes = torch.as_tensor(np.stack([s.planes for s in batch]),
                             dtype=torch.float32, device=device)
    target_pi = torch.as_tensor(np.stack([s.pi for s in batch]),
                                dtype=torch.float32, device=device)
    target_z = torch.as_tensor(np.array([s.z for s in batch], dtype=np.float32),
                               device=device).unsqueeze(1)
    logits, value = net(planes)
    log_p = F.log_softmax(logits, dim=1)
    policy_loss = -(target_pi * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_z)
    loss = policy_loss + value_loss_weight * value_loss
    if anchor_net is not None and anchor_kl > 0.0:
        with torch.no_grad():
            a_logits, _ = anchor_net(planes)
            a_log_p = F.log_softmax(a_logits, dim=1)
            a_p = a_log_p.exp()
        # KL(anchor || candidate) = sum a_p * (a_log_p - log_p)
        kl = (a_p * (a_log_p - log_p)).sum(dim=1).mean()
        loss = loss + anchor_kl * kl
    # GUARD: never backprop a non-finite loss (silent weight corruption). Skip the step.
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return float("nan"), float("nan"), float("nan"), float("nan")
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip))
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item()), grad_norm


# --------------------------------------------------------------------------- #
# strength_curve.json append (the PROOF-OF-LEARNING artifact)
# --------------------------------------------------------------------------- #
def wilson_ci95(wins: int, n: int) -> Tuple[float, float, float]:
    """95% Wilson score interval for a binomial win-rate. Returns (lo, hi, half_width).
    Wilson is correct at the small n the per-iter gate runs at (normal-approx CIs are
    garbage at n<30). half_width is the +/- a human should read on any win_rate headline:
    at n=15 it is ~0.25 (a 0.50 net reads anywhere 0.25-0.70), at n=60 ~0.13, n=120 ~0.09."""
    if n <= 0:
        return 0.0, 1.0, 1.0
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half), half


def write_heartbeat(ckpt_dir: str, it: int, phase: str) -> None:
    """Atomic heartbeat for an EXTERNAL watchdog (S6). The in-process supervise() only catches
    EXCEPTIONS; a deadlock / CUDA stall / segfault / SIGKILL leaves the process 'alive' with GPU
    at 0% and NO exception -- the in-process supervisor never fires. A separate watchdog process
    (watchdog.py) polls this file and restarts the trainer when (now - t) exceeds a stall bound.
    Best-effort: heartbeat I/O must never crash training."""
    try:
        payload = {"pid": os.getpid(), "iter": it, "phase": phase, "t": time.time()}
        path = os.path.join(ckpt_dir, "heartbeat.json")
        tmp = path + ".hb.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        pass


def append_strength_row(curve_path: str, row: dict) -> None:
    """Append one row to the strength curve, atomically. The file is a JSON list."""
    rows: List[dict] = []
    if os.path.exists(curve_path):
        try:
            with open(curve_path) as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                rows = []
        except Exception:
            rows = []
    rows.append(row)
    tmp = curve_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=2)
    os.replace(tmp, curve_path)


# --------------------------------------------------------------------------- #
# The hardened loop (one process; the supervisor relaunches it on death)
# --------------------------------------------------------------------------- #
def run_robust(cfg: RobustConfig, resume: bool = True) -> dict:
    """Thin wrapper that acquires the EXCLUSIVE per-ckpt-dir instance lock (FIX 4),
    runs the hardened loop, and releases the lock on clean exit (try/finally). Refuses
    to start a 2nd trainer on the same ckpt-dir (the lock acquire sys.exit(3)s on a live
    owner). The supervisor relaunches run_robust on a worker death; each relaunch
    re-acquires the lock -- and since the dead worker's lock is stale (its pid is gone),
    the reclaim path lets the restart proceed cleanly."""
    ckpt_dir = os.path.join(HERE, cfg.ckpt_dir)
    lock_path = acquire_instance_lock(ckpt_dir)
    try:
        return _run_robust_inner(cfg, resume=resume)
    finally:
        release_instance_lock(lock_path)


def _run_robust_inner(cfg: RobustConfig, resume: bool = True) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # seed everything (a resume will overwrite RNG from the checkpoint)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    ckpt_dir = os.path.join(HERE, cfg.ckpt_dir)
    curve_path = os.path.join(HERE, cfg.curve_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    _swept = _sweep_stale_tmp(ckpt_dir)  # clear orphan *.pt.tmp from any prior mid-write crash
    if _swept:
        print(f"[cleanup] removed {_swept} orphan *.tmp from a prior interrupted write")

    # ---- size the net (VRAM probe + back-off) ----
    channels, n_blocks, probe_gb = probe_and_size_net(cfg, device)
    if (channels, n_blocks) != (cfg.channels, cfg.n_blocks):
        print(f"[vram] net backed off to C={channels}/B={n_blocks} "
              f"(requested C={cfg.channels}/B={cfg.n_blocks})")
    cfg.channels, cfg.n_blocks = channels, n_blocks

    net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    n_params = count_params(net)
    print(f"[device] {device} ({dev_name})")
    print(f"[net] channels={channels} blocks={n_blocks} params={n_params:,} "
          f"probe_peak_vram={probe_gb:.2f}GB  cap={cfg.vram_cap_gb}GB")

    # ---- resume ----
    buffer: List[Sample] = []
    start_iter = 0
    champion: Optional[Champion] = None
    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            # the checkpoint's net size wins (so the loaded weights fit)
            try:
                head = torch.load(latest, map_location="cpu", weights_only=False)
                if (head.get("channels"), head.get("n_blocks")) != (channels, n_blocks):
                    channels, n_blocks = head["channels"], head["n_blocks"]
                    cfg.channels, cfg.n_blocks = channels, n_blocks
                    net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                                                 weight_decay=cfg.l2)
                    print(f"[resume] adopting checkpoint net size C={channels}/B={n_blocks}")
            except Exception:
                pass
            last_iter, buffer, champion = load_checkpoint(latest, net, optimizer, device, cfg)
            # M3: if the latest checkpoint carried no inline champion (e.g. it was saved
            # before the first promote), fall back to the champion sidecar so a crash-
            # resume restores the SAME floor instead of re-seeding from a noisy eval.
            if champion is None:
                champion = load_champion_sidecar(ckpt_dir)
                if champion is not None:
                    print(f"[champion] restored champion from sidecar "
                          f"(iter {champion.iter} wr_random={champion.winrate_vs_random:.3f})")
            start_iter = last_iter + 1
            print(f"[resume] loaded {os.path.basename(latest)} "
                  f"(last completed iter={last_iter}) -> continue at iter {start_iter}; "
                  f"buffer={len(buffer)}")
            if champion is not None:
                print(f"[champion] resumed champion = iter {champion.iter} "
                      f"(wr_random={champion.winrate_vs_random:.3f} "
                      f"wr_classical={champion.winrate_vs_classical:.3f})")
        else:
            print("[resume] no checkpoint found -- cold start at iter 0")

    rng = np.random.default_rng(cfg.seed + start_iter)  # eval RNG

    # ---- champion gate setup (TASK 2A) ----
    # The champion is the best-so-far net on the fixed yardstick (winrate_vs_random).
    # A separate `champion_net` holds the champion's weights and is the SELF-PLAY ACTOR
    # (so games are generated by the STRONGEST net, never a degraded candidate). Seed it
    # from the resumed champion, or -- on a pre-gate / cold checkpoint -- from the
    # currently-loaded weights (the bootstrap seed). When the gate is OFF, self-play
    # uses the in-training `net` (legacy behaviour) and every candidate is "promoted".
    champion_net: Optional[AlphaZeroNet] = None
    bootstrap_anchor_net: Optional[AlphaZeroNet] = None
    if cfg.champion_gate:
        if champion is None:
            # seed champion from the just-loaded weights; eval it once so the gate has
            # a real yardstick (the bootstrap's own winrate_vs_random) to defend.
            net.eval()
            # S5: seed the monotonic floor from a LARGER eval (not the noisy per-iter 15-game
            # match) so a lucky read can't lock the floor too high -> permanent REJECT.
            half0 = max(1, max(cfg.seed_eval_games, cfg.eval_games) // 2)
            # S6 FIX: the seed eval runs BEFORE the training loop and can take minutes (120 games);
            # write a heartbeat so an external watchdog sees the warmup is ALIVE (else it false-kills
            # the trainer mid-seed-eval and it never reaches iter 1). S5 FIX: parallelize across the
            # self-play workers so the tight-floor seed eval isn't a single-process block.
            write_heartbeat(ckpt_dir, max(start_iter - 1, 0), "seed_eval")
            seed_workers = max(1, min(cfg.selfplay_workers, half0))
            print(f"[champion] seeding floor from {2*half0} games "
                  f"(seed_eval_games={cfg.seed_eval_games}, {seed_workers} workers) for a tight estimate...")
            seed_rand = _play_match(net, "random", half0, cfg.eval_sims,
                                    cfg.eval_max_plies, cfg.eval_game_wall_s,
                                    cfg.classical_depth, device, rng, n_workers=seed_workers)
            write_heartbeat(ckpt_dir, max(start_iter - 1, 0), "seed_eval")
            seed_clas = _play_match(net, "classical", half0, cfg.eval_sims,
                                    cfg.eval_max_plies, cfg.eval_game_wall_s,
                                    cfg.classical_depth, device, rng, n_workers=seed_workers)
            champion = Champion(
                iter=max(start_iter - 1, 0),
                winrate_vs_random=seed_rand["win_rate"],
                winrate_vs_classical=seed_clas["win_rate"],
                score_vs_classical=seed_clas["score"],
                loss=float("nan"),
                state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            )
            print(f"[champion] seeded champion from the bootstrap/base net: "
                  f"wr_random={champion.winrate_vs_random:.3f} "
                  f"wr_classical={champion.winrate_vs_classical:.3f} "
                  f"score_classical={champion.score_vs_classical:.3f}")
            # M3: persist the seeded champion to its sidecar IMMEDIATELY so a crash
            # before the first promote does not re-roll a fresh (noisy) champion floor.
            sc_path = save_champion_sidecar(ckpt_dir, champion)
            print(f"[champion] persisted seeded champion -> {os.path.basename(sc_path)}")
        champion_net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
        champion_net.load_state_dict(champion.state_dict, strict=False)
        champion_net.eval()
        # anchor net for the optional KL penalty toward the bootstrap policy (TASK 2B)
        if cfg.anchor_kl > 0.0:
            bootstrap_anchor_net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
            bootstrap_anchor_net.load_state_dict(champion.state_dict, strict=False)
            bootstrap_anchor_net.eval()

    # ---- FORGETTING/DRIFT baseline (S5b): a FROZEN copy of the seed net (net_iter0 / bootstrap).
    # The current net is periodically played head-to-head vs this; a sustained <0.5 means the net
    # can't beat its own starting point = catastrophic forgetting. On resume we reload net_iter0
    # (the ORIGINAL seed), not the resumed weights, so the axis is always "vs the bootstrap". ----
    baseline_net: Optional[AlphaZeroNet] = None
    if cfg.forgetting_eval_interval > 0:
        baseline_net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
        seed_path = os.path.join(ckpt_dir, "net_iter0.pt")
        loaded_seed = False
        if os.path.exists(seed_path):
            try:
                _bd = torch.load(seed_path, map_location=device, weights_only=False)
                baseline_net.load_state_dict(_bd["state_dict"], strict=False)
                loaded_seed = True
            except Exception as e:
                print(f"[forgetting] WARN: could not load net_iter0 as baseline ({e}); "
                      f"using the current weights")
        if not loaded_seed:
            baseline_net.load_state_dict(net.state_dict(), strict=False)
        baseline_net.eval()
        print(f"[forgetting] drift baseline = {'net_iter0 (frozen seed)' if loaded_seed else 'current bootstrap weights'} "
              f"-- net plays it every {cfg.forgetting_eval_interval} iters (<0.5 sustained = FORGETTING)")

    wall_start = time.time()
    cur_sims = cfg.selfplay_sims
    cur_batch = cfg.batch_size

    # ---- AUTO-BALANCE startup: derive workers from hardware; seed games/steps from the budget ----
    # (online rebalance from the first measured iter onward; learning-contract knobs untouched).
    auto_targets = None
    auto_target_iter_s = 0.0
    if cfg.auto_balance:
        from .auto_balance import HW, BalanceTargets, derive_workers, initial_plan, target_iter_seconds
        auto_targets = BalanceTargets(iters_in_budget=cfg.auto_iters_in_budget,
                                      replay_ratio=cfg.auto_replay_ratio)
        cores = os.cpu_count() or 4
        cfg.selfplay_workers = derive_workers(cores, cfg.games_per_iter, auto_targets)
        auto_target_iter_s = target_iter_seconds(cfg.max_hours, auto_targets)
        print(f"[auto-balance] ON: cores={cores} -> selfplay_workers={cfg.selfplay_workers}; "
              f"target iter-time={auto_target_iter_s:.0f}s (budget {cfg.max_hours}h / "
              f"{cfg.auto_iters_in_budget} iters); replay-ratio target {cfg.auto_replay_ratio:.1f}; "
              f"start games_per_iter={cfg.games_per_iter} train_steps={cfg.train_steps_per_iter} "
              f"(throughput auto-tuned online; gate/anchor-kl/curriculum/lr stay fixed)")

    # H2 FIX (2026-06-07): --iters is ADDITIVE, not an absolute ceiling. On a resume
    # at start_iter, run cfg.iterations MORE iters (range [start_iter, start_iter+N)).
    # Previously `range(start_iter, cfg.iterations)` made --iters an absolute cap, so a
    # resume where start_iter >= cfg.iterations produced an EMPTY loop: no net, no curve
    # row, silent exit 0 (e.g. the docstring's `--iters 2` smoke did nothing on a dir
    # already at iter>=2). Additive semantics match the user's `learn` loop intuition.
    stop_iter = start_iter + cfg.iterations
    for it in range(start_iter, stop_iter):
        # ---- wall-clock envelope (clean stop) ----
        elapsed_h = (time.time() - wall_start) / 3600.0
        if elapsed_h >= cfg.max_hours:
            print(f"[stop] wall-clock {elapsed_h:.2f}h >= max_hours {cfg.max_hours}h "
                  f"-- stopping cleanly before iter {it}")
            break

        it_t0 = time.time()

        # ---- candidate starts the iter from the CHAMPION's weights (gate ON) ----
        # so every candidate refines the STRONGEST net, never a degraded predecessor.
        if cfg.champion_gate and champion_net is not None:
            net.load_state_dict(champion.state_dict, strict=False)

        # ---- (a) SELF-PLAY (OOM-guarded) ----
        # Actor = the CHAMPION net (gate ON) so games come from the strongest net, not
        # a degraded candidate. Opponent per cfg.selfplay_opponent (self/teacher/mix).
        actor = champion_net if (cfg.champion_gate and champion_net is not None) else net
        write_heartbeat(ckpt_dir, it, "selfplay")  # S6: phase heartbeat for the external watchdog
        sp_t0 = time.time()
        new_examples = 0
        # DIVERSITY GAUGE (2026-06-09): collect this iter's per-game sample lists so we can
        # MEASURE self-play diversity -- distinct starting positions + decisive-game fraction.
        # This is the instrument that makes a dead-diversity run (every game from the same
        # start -> the net reinforces one rote line) VISIBLE instead of silently failing: the
        # prior 111-iter run had no such gauge, so collapsed openings looked identical to a
        # healthy run on loss/win-rate. Cheap: hashes the first sample's planes per game.
        games_this_iter: List[List[Sample]] = []
        actor.eval()
        g = 0
        floor_oom = 0  # consecutive OOM retries WHILE ALREADY AT min_sims (livelock guard)
        # THROUGHPUT KNOB (--parallel-games): batched self-play applies ONLY to pure 'self'
        # opponent (the batchable case). teacher/mix interleave an external engine that can't be
        # batched, so they keep the sequential path -- noted once so the ignore is never silent.
        use_batched = (cfg.parallel_games > 1 and cfg.selfplay_opponent == "self")
        if cfg.parallel_games > 1 and not use_batched:
            print(f"[parallel] --parallel-games {cfg.parallel_games} applies to opponent='self' "
                  f"only; opponent='{cfg.selfplay_opponent}' uses the sequential path")
        # MULTIPROCESS SELF-PLAY (--selfplay-workers N): the real throughput lever. Self-play is CPU/Python-bound
        # (MCTS + move-gen), so N CPU worker PROCESSES generate this iter's games in parallel (~Nx faster, measured
        # 13.5x at N=16 on a 20-core box). Now covers ALL opponents: 'self' (speed), 'teacher' (QUALITY -- learn from
        # the classical bot, which cures the pure-self-play degradation), and 'mix' (half self + half teacher = dual
        # learning at multiprocess speed). A pool failure falls back to the sequential loop so an iter is never lost.
        use_workers = (cfg.selfplay_workers > 1 and cfg.selfplay_opponent in ("self", "teacher", "mix"))
        if use_workers:
            try:
                from .selfplay_pool import generate_games_parallel, generate_teacher_games_parallel
                wt0 = time.time()
                W, N, seedb = cfg.selfplay_workers, cfg.games_per_iter, cfg.seed + it * 100003
                common = dict(sims=cur_sims, temp_moves=cfg.temp_moves, max_plies=cfg.max_plies,
                              game_wall_s=cfg.game_wall_s, channels=cfg.channels, n_blocks=cfg.n_blocks,
                              opening_mode=cfg.opening_mode, opening_plies=cfg.opening_plies)
                if cfg.selfplay_opponent == "self":
                    group = generate_games_parallel(actor, n_games=N, n_workers=W, seed_base=seedb, **common)
                elif cfg.selfplay_opponent == "teacher":
                    group = generate_teacher_games_parallel(actor, n_games=N, n_workers=W,
                                                            teacher_depth=cfg.selfplay_teacher_depth,
                                                            distill_teacher=cfg.teacher_distill,
                                                            seed_base=seedb, **common)
                else:  # mix: half self (diversity) + half teacher (high-quality data) = dual learning
                    half = N // 2
                    group = generate_games_parallel(actor, n_games=half, n_workers=W, seed_base=seedb, **common)
                    group += generate_teacher_games_parallel(actor, n_games=N - half, n_workers=W,
                                                             teacher_depth=cfg.selfplay_teacher_depth,
                                                             distill_teacher=cfg.teacher_distill,
                                                             seed_base=seedb + 7, **common)
                for samples in group:
                    buffer.extend(samples)
                    new_examples += len(samples)
                    games_this_iter.append(samples)
                g = cfg.games_per_iter  # done -> skip the sequential loop below
                print(f"[selfplay-pool] {len(group)} games ({cfg.selfplay_opponent}) via {W} workers in "
                      f"{time.time() - wt0:.1f}s ({new_examples} samples)")
            except Exception as e:
                print(f"[selfplay-pool] FAILED ({type(e).__name__}: {e}) -> sequential fallback this iter")
                use_workers = False
                g = 0
                new_examples = 0
        while g < cfg.games_per_iter:
            try:
                if use_batched:
                    # generate a GPU-batched group of up to parallel_games self-play games in one
                    # lockstep run (leaf evals batched across games). seed varies per iter+offset
                    # so games are fresh each iteration. An OOM here is caught below: cur_sims is
                    # halved (less per-eval memory) and the group retried; floor-bound still applies.
                    from .batched_selfplay import generate_selfplay_games_batched
                    n_this = min(cfg.parallel_games, cfg.games_per_iter - g)
                    group = generate_selfplay_games_batched(
                        actor, n_games=n_this, n_simulations=cur_sims,
                        temp_moves=cfg.temp_moves, max_plies=cfg.max_plies,
                        game_wall_s=cfg.game_wall_s, device=device,
                        seed=cfg.seed + it * 100003 + g,
                        opening_mode=cfg.opening_mode, opening_plies=cfg.opening_plies)
                    for samples in group:
                        buffer.extend(samples)
                        new_examples += len(samples)
                        games_this_iter.append(samples)
                    g += n_this
                    floor_oom = 0
                    continue
                # opponent for THIS game (mix alternates self/teacher game-by-game)
                if cfg.selfplay_opponent == "mix":
                    game_opp = "teacher" if (g % 2 == 0) else "self"
                else:
                    game_opp = cfg.selfplay_opponent
                samples = generate_selfplay_game_guarded(
                    actor, n_simulations=cur_sims, temp_moves=cfg.temp_moves,
                    max_plies=cfg.max_plies, game_wall_s=cfg.game_wall_s,
                    device=device, opponent=game_opp,
                    teacher_depth=cfg.selfplay_teacher_depth,
                    net_is_white=(g % 2 == 0),
                    engine_path=cfg.engine_path,
                    uci_movetime_ms=cfg.uci_movetime_ms,
                    distill_teacher=cfg.teacher_distill,
                    opening_mode=cfg.opening_mode, opening_plies=cfg.opening_plies,
                    opening_seed=cfg.seed + it * 100003 + g)
                buffer.extend(samples)
                new_examples += len(samples)
                games_this_iter.append(samples)
                g += 1
                floor_oom = 0  # progress made -> reset the floor-retry budget
            except Exception as e:
                if _is_oom(e):
                    torch.cuda.empty_cache()
                    at_floor = (cur_sims <= cfg.min_sims)
                    cur_sims = max(cfg.min_sims, cur_sims // 2)
                    if at_floor:
                        # already at the floor: halving cannot free more memory. Bound the
                        # retries so a PERSISTENT floor-OOM RAISES (supervisor restarts)
                        # instead of livelocking the loop forever (invisible to the supervisor).
                        floor_oom += 1
                        print(f"[oom] self-play OOM AT FLOOR (sims={cur_sims}) "
                              f"retry {floor_oom}/{cfg.floor_oom_retries}, game {g}")
                        if floor_oom >= cfg.floor_oom_retries:
                            raise RuntimeError(
                                f"persistent self-play OOM at floor sims={cur_sims} after "
                                f"{floor_oom} retries -- raising so --supervise can restart "
                                f"(was a silent livelock)") from e
                    else:
                        print(f"[oom] self-play OOM -> empty_cache, sims->{cur_sims}, retry game {g}")
                    continue  # retry same game at lower sims
                raise  # non-OOM: let the supervisor catch + restart
        if len(buffer) > cfg.buffer_size:
            buffer = buffer[-cfg.buffer_size:]
        sp_dt = time.time() - sp_t0

        # ---- DIVERSITY GAUGE: distinct starting positions + decisive-game fraction ----
        # The instrument that catches a collapsed-self-play run (the failure the user spotted
        # by eye: "it plays the same way"). distinct_starts counts unique opening positions
        # (hash of the first sample's planes per game); decisive_frac is the share of games
        # that ended decisively (someone won -> nonzero z) rather than as a draw. A healthy
        # diverse run has distinct_starts ~ n_games and a non-trivial decisive_frac; a dead
        # run pins distinct_starts at 1. We WARN loudly when diversity is dead so it can never
        # again look identical to a healthy run on loss/win-rate alone.
        n_games_iter = len(games_this_iter)
        distinct_starts = len({hash(gm[0].planes.tobytes()) for gm in games_this_iter if gm})
        decisive_games = sum(1 for gm in games_this_iter if gm and any(s.z != 0.0 for s in gm))
        decisive_frac = (decisive_games / n_games_iter) if n_games_iter else 0.0
        if (cfg.opening_mode != "startpos" and n_games_iter > 1 and distinct_starts <= 1):
            print(f"[diversity] WARN: opening_mode='{cfg.opening_mode}' but {n_games_iter} games "
                  f"shared {distinct_starts} starting position -- self-play diversity is DEAD "
                  f"(the net will reinforce one rote line). Check openings.sample_opening_board.")

        # ---- (b) TRAIN (OOM-guarded) ----
        write_heartbeat(ckpt_dir, it, "train")  # S6
        tr_t0 = time.time()
        step_losses: List[Tuple[float, float, float]] = []
        grad_norms: List[float] = []
        nan_count = 0  # non-finite losses this iter (NaN-guard skips the step; abort if it runs away)
        if len(buffer) >= 2:  # small-buffer-safe: train with whatever we have (sample-with-replacement) -> NEVER skip-to-nan
            net.train()
            s = 0
            floor_oom = 0  # consecutive OOM retries WHILE ALREADY AT min_batch (livelock guard)
            while s < cfg.train_steps_per_iter:
                try:
                    idx = rng.integers(0, len(buffer), size=min(cur_batch, len(buffer)))
                    batch = [buffer[i] for i in idx]
                    if cfg.anchor_kl > 0.0 and bootstrap_anchor_net is not None:
                        loss, pl, vl, gnorm = train_step_anchored(
                            net, optimizer, batch, device, cfg.l2,
                            bootstrap_anchor_net, cfg.anchor_kl,
                            value_loss_weight=cfg.value_loss_weight, grad_clip=cfg.grad_clip)
                    else:
                        loss, pl, vl, gnorm = train_step(
                            net, optimizer, batch, device, l2=cfg.l2,
                            value_loss_weight=cfg.value_loss_weight, grad_clip=cfg.grad_clip)
                    # NaN GUARD (S1): train_step returns loss=NaN when it SKIPPED a non-finite-loss
                    # batch (no backward/step). Count it, don't record it, and ABORT a runaway so
                    # --supervise restarts from the last good checkpoint instead of silently churning.
                    if not (loss == loss):  # NaN
                        nan_count += 1
                        print(f"[nan] non-finite loss -> step skipped "
                              f"(nan_count={nan_count}/{cfg.nan_abort}, step {s})")
                        if nan_count >= cfg.nan_abort:
                            raise RuntimeError(
                                f"{nan_count} non-finite losses this iter (>= nan_abort "
                                f"{cfg.nan_abort}) -- raising so --supervise restarts from the "
                                f"last good checkpoint (silent weight-corruption guard)")
                        s += 1  # advance so a persistently-bad batch source can't livelock
                        continue
                    step_losses.append((loss, pl, vl))
                    grad_norms.append(gnorm)
                    s += 1
                    floor_oom = 0  # progress made -> reset the floor-retry budget
                except Exception as e:
                    if _is_oom(e):
                        torch.cuda.empty_cache()
                        at_floor = (cur_batch <= cfg.min_batch)
                        cur_batch = max(cfg.min_batch, cur_batch // 2)
                        if at_floor:
                            # already at the floor: halving cannot free more memory. Bound
                            # the retries so a PERSISTENT floor-OOM RAISES (supervisor restarts)
                            # instead of livelocking forever (invisible to the supervisor).
                            floor_oom += 1
                            print(f"[oom] train OOM AT FLOOR (batch={cur_batch}) "
                                  f"retry {floor_oom}/{cfg.floor_oom_retries}, step {s}")
                            if floor_oom >= cfg.floor_oom_retries:
                                raise RuntimeError(
                                    f"persistent train OOM at floor batch={cur_batch} after "
                                    f"{floor_oom} retries -- raising so --supervise can restart "
                                    f"(was a silent livelock)") from e
                        else:
                            print(f"[oom] train OOM -> empty_cache, batch->{cur_batch}, retry step {s}")
                        continue
                    raise
        tr_dt = time.time() - tr_t0

        if step_losses:
            tail = step_losses[-min(20, len(step_losses)):]
            total_loss = float(np.mean([s[0] for s in tail]))
            policy_loss = float(np.mean([s[1] for s in tail]))
            value_loss = float(np.mean([s[2] for s in tail]))
        else:
            total_loss = policy_loss = value_loss = float("nan")

        # ---- (c) STRENGTH EVAL: vs random + vs classical depth-1 ----
        write_heartbeat(ckpt_dir, it, "eval")  # S6
        ev_t0 = time.time()
        net.eval()
        half = max(1, cfg.eval_games // 2)
        # PARALLEL EVAL: split each match across CPU workers (eval was the 2nd-biggest iter cost).
        # Capped at the self-play worker count; _play_match no-ops the pool for <4 games.
        ev_workers = max(1, min(cfg.selfplay_workers, half))
        ev_rand = _play_match(net, "random", half, cfg.eval_sims,
                              cfg.eval_max_plies, cfg.eval_game_wall_s,
                              cfg.classical_depth, device, rng, n_workers=ev_workers)
        ev_clas = _play_match(net, "classical", half, cfg.eval_sims,
                              cfg.eval_max_plies, cfg.eval_game_wall_s,
                              cfg.classical_depth, device, rng, n_workers=ev_workers)
        # EVAL TRUST (S5): attach the 95% Wilson CI so every win_rate is read WITH its
        # uncertainty. The champion gate decides on ev_rand's win_rate; if its CI is wide the
        # promote/reject is being made on noise -- make that LOUD instead of silent.
        _rl, _rh, rand_ci = wilson_ci95(ev_rand["win"], ev_rand["games"])
        _cl, _ch, clas_ci = wilson_ci95(ev_clas["win"], ev_clas["games"])
        if cfg.champion_gate and rand_ci > 0.12:
            print(f"[eval] WARN: gate win_rate_vs_random={ev_rand['win_rate']:.3f} "
                  f"+/-{rand_ci:.3f} (n={ev_rand['games']}) -- CI too wide for a reliable gate "
                  f"decision; raise --eval-games (n>=60 -> +/-~0.13, n>=120 -> +/-~0.09).")

        # ---- FORGETTING / DRIFT axis (S5b): current net vs the FROZEN SEED, every N iters ----
        wr_vs_baseline = None
        baseline_ci = None
        if (baseline_net is not None and cfg.forgetting_eval_interval > 0
                and (it % cfg.forgetting_eval_interval == 0 or it == stop_iter - 1)):
            fg_games = cfg.forgetting_eval_games or cfg.eval_games
            h2h_base = _play_head_to_head(net, baseline_net, fg_games, cfg.eval_sims,
                                          cfg.eval_max_plies, cfg.eval_game_wall_s, device)
            wr_vs_baseline = h2h_base["win_rate"]
            _bl, _bh, baseline_ci = wilson_ci95(h2h_base["win"], h2h_base["games"])
            print(f"[forgetting] vs frozen seed: win_rate={wr_vs_baseline:.3f} +/-{baseline_ci:.3f} "
                  f"(n={fg_games})  (should be >0.5 and rising; <0.5 sustained = forgetting)")
            if wr_vs_baseline < 0.45 and baseline_ci < 0.15:
                print(f"[forgetting] WARN: net is LOSING to its own frozen seed "
                      f"({wr_vs_baseline:.3f} +/-{baseline_ci:.3f}) -- catastrophic forgetting / "
                      f"drift away from the bootstrap. The vs-random axis cannot see this.")
        ev_dt = time.time() - ev_t0

        peak_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if device.type == "cuda" else 0.0)

        # ---- (d) CHAMPION GATE (TASK 2A): promote only if no regression ----
        cand_wr_random = ev_rand["win_rate"]
        cand_wr_classical = ev_clas["win_rate"]
        cand_score_classical = ev_clas["score"]  # draw-aware CLIMB axis
        promoted = True
        gate_reason = "gate disabled -- candidate accepted"
        h2h_winrate = None
        if cfg.champion_gate and champion is not None:
            # optional published-AZ-style head-to-head match candidate-vs-champion
            if cfg.champion_h2h:
                h2h = _play_head_to_head(net, champion_net, cfg.champion_h2h_games,
                                         cfg.eval_sims, cfg.eval_max_plies,
                                         cfg.eval_game_wall_s, device)
                h2h_winrate = h2h["win_rate"]
            promoted, gate_reason = candidate_beats_champion(
                cand_wr_random, cand_wr_classical, total_loss, champion,
                cfg.champion_tol, cand_score_classical=cand_score_classical,
                h2h_winrate=h2h_winrate)

        if promoted:
            # candidate becomes the new champion; persist ITS weights as the playable net
            cand_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            champion = Champion(iter=it, winrate_vs_random=cand_wr_random,
                                winrate_vs_classical=cand_wr_classical,
                                score_vs_classical=cand_score_classical,
                                loss=total_loss, state_dict=cand_sd)
            if champion_net is not None:
                champion_net.load_state_dict(cand_sd, strict=False)
                champion_net.eval()
            weights_for_file = None  # save_checkpoint uses net.state_dict() (the candidate)
            # M3: keep the champion sidecar in lock-step with the in-memory champion so a
            # crash-resume always restores the CURRENT floor, even if the inline-champion
            # checkpoint write is lost.
            if cfg.champion_gate:
                save_champion_sidecar(ckpt_dir, champion)
            print(f"[gate] PROMOTE iter {it}: {gate_reason}")
        else:
            # REJECT: write the CHAMPION's weights as net_iter{it}.pt so BOTH the
            # latest.pt pointer AND the highest-net_iterN.pt glob fallback resolve to
            # the champion -> the playable net is MONOTONIC (never regresses).
            weights_for_file = champion.state_dict
            print(f"[gate] REJECT iter {it}: {gate_reason} "
                  f"(keep champion iter {champion.iter} "
                  f"wr_random={champion.winrate_vs_random:.3f})")

        # ---- CURRICULUM (the MOVING TARGET): make the teacher harder as the net masters
        # the CURRENT teacher depth. Bug-fixed 2026-06-08 (3 bugs, see RobustConfig):
        #   (b) MEASURE THE RIGHT THING: gate on a DEDICATED eval of THIS iter's net vs
        #       Engine(depth=selfplay_teacher_depth) -- the CURRENT teacher -- not the fixed
        #       depth-1 eval above. When the teacher already IS depth 1, reuse the depth-1
        #       score we just computed (cand_score_classical) instead of re-evaluating.
        #   (a) LATCH: only bump if we have not ALREADY bumped at this depth
        #       (curriculum_last_bumped_depth < selfplay_teacher_depth) AND we are below the
        #       cap -- so the bump fires exactly ONCE per depth crossing, not every iter.
        # The harder teacher resets the climb so the net keeps improving instead of
        # saturating against a fixed depth. OFF by default (no-op).
        curriculum_bumped = False
        curriculum_score_vs_teacher = None
        if cfg.curriculum and champion is not None:
            cur_depth = cfg.selfplay_teacher_depth
            already_bumped_here = cfg.curriculum_last_bumped_depth >= cur_depth
            if cur_depth < cfg.curriculum_max_depth and not already_bumped_here:
                # (b) dedicated score vs the CURRENT teacher depth (reuse depth-1 eval when
                # the teacher is depth 1; classical_depth defaults to 1 so they coincide).
                if cur_depth == cfg.classical_depth:
                    curriculum_score_vs_teacher = cand_score_classical
                else:
                    ev_teach = _play_match(net, "classical", half, cfg.eval_sims,
                                           cfg.eval_max_plies, cfg.eval_game_wall_s,
                                           cur_depth, device, rng)
                    curriculum_score_vs_teacher = ev_teach["score"]
                if curriculum_score_vs_teacher >= cfg.curriculum_threshold - 1e-9:
                    cfg.selfplay_teacher_depth += 1
                    cfg.curriculum_last_bumped_depth = cur_depth  # (a) latch this crossing
                    curriculum_bumped = True
                    print(f"[curriculum] candidate score vs teacher depth {cur_depth} = "
                          f"{curriculum_score_vs_teacher:.3f} >= {cfg.curriculum_threshold:.2f}"
                          f" -- bumping teacher depth {cur_depth} -> "
                          f"{cfg.selfplay_teacher_depth} (cap {cfg.curriculum_max_depth}, "
                          f"latched at {cfg.curriculum_last_bumped_depth})")

        # ---- checkpoint EVERY iteration (atomic + latest pointer) ----
        ckpt_path = save_checkpoint(ckpt_dir, it, net, optimizer, cfg, buffer,
                                    champion=champion,
                                    weights_state_dict=weights_for_file,
                                    promoted=promoted)

        it_dt = time.time() - it_t0
        wall_s = round(time.time() - wall_start, 1)

        # ---- append the strength-curve row (the PROOF artifact) ----
        row = {
            "iter": it,
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "winrate_vs_random": ev_rand["win_rate"],
            "winrate_vs_classical_d1": ev_clas["win_rate"],
            # EVAL TRUST (S5): 95% Wilson CI half-width on each win_rate -- the +/- a human must
            # read on the headline. A gate decision with a wide CI here was made on noise.
            "winrate_vs_random_ci95": round(rand_ci, 4),
            "winrate_vs_classical_d1_ci95": round(clas_ci, 4),
            # FORGETTING/DRIFT axis (S5b): None on iters we don't run it. <0.5 sustained = the net
            # can't beat its own frozen seed = catastrophic forgetting (invisible to vs-random).
            "winrate_vs_baseline": (round(wr_vs_baseline, 4) if wr_vs_baseline is not None else None),
            "winrate_vs_baseline_ci95": (round(baseline_ci, 4) if baseline_ci is not None else None),
            # CLIMB axis (2026-06-07): draw-aware score vs the classical engine. This is
            # the field that MOVES while winrate_vs_classical_d1 sits at 0 (losing -> drawing).
            "score_vs_classical_d1": ev_clas["score"],
            "n_eval_games": ev_rand["games"] + ev_clas["games"],
            "vs_random": ev_rand,
            "vs_classical_d1": ev_clas,
            # champion-gate audit fields (TASK 2A): candidate eval is ALWAYS recorded;
            # `promoted` says whether it became the playable champion; champion_* is the
            # MONOTONIC curve (what latest.pt/the player actually is after this iter).
            "promoted": promoted,
            "gate_reason": gate_reason,
            "h2h_winrate_vs_champion": h2h_winrate,
            "champion_iter": (champion.iter if champion is not None else None),
            "champion_winrate_vs_random": (champion.winrate_vs_random
                                           if champion is not None else None),
            "champion_winrate_vs_classical": (champion.winrate_vs_classical
                                              if champion is not None else None),
            "champion_score_vs_classical": (champion.score_vs_classical
                                            if champion is not None else None),
            "selfplay_opponent": cfg.selfplay_opponent,
            "selfplay_teacher_depth": cfg.selfplay_teacher_depth,
            "curriculum_bumped": curriculum_bumped,
            # curriculum audit (2026-06-08): the score that was MEASURED against the CURRENT
            # teacher depth (None when curriculum off / already latched / at cap), and the
            # latch (last depth we bumped at) so the once-per-crossing behaviour is visible.
            "curriculum_score_vs_teacher": curriculum_score_vs_teacher,
            "curriculum_last_bumped_depth": cfg.curriculum_last_bumped_depth,
            "anchor_kl": cfg.anchor_kl,
            "wall_s": wall_s,
            "ckpt": os.path.relpath(ckpt_path, HERE),
            "buffer_size": len(buffer),
            "new_examples": new_examples,
            # DIVERSITY GAUGE (2026-06-09): the self-play health the prior run could not see.
            "opening_mode": cfg.opening_mode,
            "selfplay_games": n_games_iter,
            "selfplay_distinct_starts": distinct_starts,
            "selfplay_decisive_frac": round(decisive_frac, 3),
            "train_steps": len(step_losses),
            # NUMERICAL-SAFETY telemetry (2026-06-09): grad-norm EMA + NaN-skip count make
            # gradient explosions / silent weight-corruption visible per iter (H3/H1).
            "grad_norm": round(float(np.mean(grad_norms)), 4) if grad_norms else None,
            "grad_norm_max": round(float(np.max(grad_norms)), 4) if grad_norms else None,
            "nan_skipped": nan_count,
            "cur_sims": cur_sims,
            "cur_batch": cur_batch,
            "peak_vram_gb": round(peak_gb, 3),
            "channels": cfg.channels,
            "n_blocks": cfg.n_blocks,
            "timing_s": {"selfplay": round(sp_dt, 1), "train": round(tr_dt, 1),
                         "eval": round(ev_dt, 1), "iter_total": round(it_dt, 1)},
        }
        append_strength_row(curve_path, row)

        print(f"\n[iter {it}] buffer={len(buffer)} new={new_examples} "
              f"steps={len(step_losses)} sims={cur_sims} batch={cur_batch}")
        print(f"           selfplay: {n_games_iter} games, "
              f"{distinct_starts} distinct starts (mode={cfg.opening_mode}), "
              f"decisive={decisive_frac:.0%}")
        _gn_str = f" grad_norm={np.mean(grad_norms):.3f}" if grad_norms else ""
        print(f"           loss total={total_loss:.4f} "
              f"(policy={policy_loss:.4f} value={value_loss:.4f} w_v={cfg.value_loss_weight})"
              f"{_gn_str}")
        if nan_count:
            print(f"           [nan] {nan_count} non-finite-loss step(s) skipped this iter")
        print(f"           vs random:    W{ev_rand['win']} D{ev_rand['draw']} "
              f"L{ev_rand['loss']} win_rate={ev_rand['win_rate']:.3f} +/-{rand_ci:.3f} "
              f"(95% CI, n={ev_rand['games']})")
        print(f"           vs classical: W{ev_clas['win']} D{ev_clas['draw']} "
              f"L{ev_clas['loss']} win_rate={ev_clas['win_rate']:.3f} +/-{clas_ci:.3f}")
        if cfg.champion_gate and champion is not None:
            print(f"           gate: {'PROMOTE' if promoted else 'REJECT'}  "
                  f"champion=iter{champion.iter} "
                  f"wr_random={champion.winrate_vs_random:.3f} (MONOTONIC floor)")
        print(f"           peak_vram={peak_gb:.2f}GB  timing: selfplay={sp_dt:.1f}s "
              f"train={tr_dt:.1f}s eval={ev_dt:.1f}s iter={it_dt:.1f}s  wall={wall_s/60:.1f}min")

        # ---- AUTO-BALANCE online: nudge games/steps (and workers) toward the target iter-time
        #      for the NEXT iter, holding the replay ratio in band. THROUGHPUT ONLY -- the gate,
        #      anchor-kl, curriculum, lr, opponent are never touched (auto_balance has no access
        #      to them). The floor-OOM backoff still guards VRAM separately. ----
        if cfg.auto_balance and auto_targets is not None and auto_target_iter_s > 0:
            from .auto_balance import rebalance, derive_workers
            avg_plies = new_examples / max(1, cfg.games_per_iter)  # samples/game ~ plies/game
            new_g, new_s, notes = rebalance(cfg.games_per_iter, cfg.train_steps_per_iter,
                                            cfg.batch_size, avg_plies, it_dt, auto_target_iter_s,
                                            auto_targets)
            for n in notes:
                print(f"           {n}")
            cfg.games_per_iter = new_g
            cfg.train_steps_per_iter = new_s
            cfg.selfplay_workers = derive_workers(os.cpu_count() or 4, new_g, auto_targets)

    total_wall = time.time() - wall_start
    print("\n" + "=" * 64)
    print(f"[done] wall={total_wall/60:.1f} min  iters_requested={cfg.iterations} "
          f"(ran [{start_iter}, {stop_iter}))  device={device}")
    print(f"[done] strength curve -> {os.path.relpath(curve_path, HERE)}")
    print(f"[done] checkpoints    -> {os.path.relpath(ckpt_dir, HERE)}/")
    print("=" * 64)
    # ---- AUTO-EMIT the honest learning report (iters/hr, champion-floor monotonicity, curve
    #      slope, parabolic=rising+accel). Best-effort: a report failure never fails the run. ----
    report = None
    try:
        from .learning_report import report as _emit_report
        report = _emit_report(ckpt_dir)
        r = report
        print(f"[report] SPEED    : {r['iters_completed']} iters | mean {r['mean_iter_s']}s/iter | "
              f"{r['iters_per_hour']} iters/hr | self-play {r['selfplay_games_per_s']} games/s")
        print(f"[report] LEARNING : champion floor vs random = {r['champion_floor_vs_random']} "
              f"(monotonic={r['champion_floor_monotonic']})")
        print(f"[report] vs CLASSIC: {r['score_vs_classical_trend']} (draw-aware slope "
              f"{r['score_vs_classical_slope_per_iter']}/iter, "
              f"parabolic={r['score_vs_classical_parabolic(rising+accelerating)']})")
        print(f"[report] DIVERSITY: opening_mode={r.get('opening_mode')} | "
              f"distinct-starts/games={r.get('selfplay_diversity_ratio')} | "
              f"decisive={r.get('selfplay_decisive_frac')}"
              + ("  *** DIVERSITY DEAD ***" if r.get('selfplay_diversity_dead') else ""))
        print("=" * 64)
    except Exception as e:
        print(f"[report] (skipped: {type(e).__name__}: {e})")
    return {"wall_s": total_wall, "curve_path": curve_path, "ckpt_dir": ckpt_dir, "report": report}


# --------------------------------------------------------------------------- #
# In-process SUPERVISOR: auto-restart the loop from the latest checkpoint if a
# worker iteration dies (real unattended robustness). Honors the wall-clock
# envelope across restarts so it cannot loop forever.
# --------------------------------------------------------------------------- #
def supervise(cfg: RobustConfig, resume: bool, max_restarts: int = 50) -> None:
    global_start = time.time()
    restarts = 0
    while True:
        elapsed_h = (time.time() - global_start) / 3600.0
        remaining_h = cfg.max_hours - elapsed_h
        if remaining_h <= 0:
            print(f"[supervisor] wall-clock envelope {cfg.max_hours}h spent -- done.")
            return
        # the child inherits the REMAINING budget so the envelope is global. CURRICULUM
        # (bug-c fix 2026-06-08): forward the LIVE teacher depth + latch from cfg so the
        # restarted child starts at the depth the curriculum has climbed to, not the CLI
        # default. run_robust ALSO restores these from the checkpoint on resume (the durable
        # path); forwarding them here is the belt-and-suspenders in-memory path so the first
        # post-crash child is correct even before it reads the checkpoint config.
        child_cfg = RobustConfig(**{**asdict(cfg), "max_hours": remaining_h})
        try:
            print(f"[supervisor] launching loop (restart #{restarts}, "
                  f"remaining {remaining_h:.2f}h, resume={resume or restarts > 0}, "
                  f"teacher_depth={child_cfg.selfplay_teacher_depth})")
            run_robust(child_cfg, resume=resume or restarts > 0)
            print("[supervisor] loop returned cleanly (envelope reached or iters done).")
            return
        except KeyboardInterrupt:
            print("[supervisor] KeyboardInterrupt -- stopping (no restart).")
            return
        except Exception:
            restarts += 1
            print(f"[supervisor] WORKER DIED (restart {restarts}/{max_restarts}):")
            traceback.print_exc()
            if restarts >= max_restarts:
                print("[supervisor] max restarts reached -- giving up.")
                return
            # CURRICULUM: run_robust mutates child_cfg in place, so child_cfg now holds the
            # highest teacher depth + latch the dead child reached. Carry it back into cfg so
            # the NEXT child_cfg is rebuilt at that depth (in addition to the checkpoint
            # restore inside run_robust).
            if cfg.curriculum:
                cfg.selfplay_teacher_depth = child_cfg.selfplay_teacher_depth
                cfg.curriculum_last_bumped_depth = child_cfg.curriculum_last_bumped_depth
            # always resume after a crash so we pick up the latest checkpoint
            resume = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2.0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    d = RobustConfig()
    ap = argparse.ArgumentParser(
        description="Hardened AlphaZero self-play->train->eval loop (unattended-safe).")
    ap.add_argument("--iters", type=int, default=d.iterations,
                    help="number of iterations to run THIS invocation (ADDITIVE on "
                         "resume: a resume at iter K runs K..K+iters, not an absolute "
                         "ceiling). So `--iters 2` always runs 2 more iters.")
    ap.add_argument("--max-hours", type=float, default=d.max_hours,
                    help="wall-clock envelope; stop cleanly past it (6h run = 6)")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="resume from latest checkpoint (DEFAULT ON)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore checkpoints, cold-start at iter 0")
    ap.add_argument("--supervise", action="store_true",
                    help="run inside the in-process auto-restart supervisor")
    # net
    ap.add_argument("--channels", type=int, default=d.channels)
    ap.add_argument("--n-blocks", type=int, default=d.n_blocks)
    ap.add_argument("--vram-cap-gb", type=float, default=d.vram_cap_gb)
    # self-play
    ap.add_argument("--games-per-iter", type=int, default=d.games_per_iter)
    ap.add_argument("--selfplay-workers", type=int, default=d.selfplay_workers,
                    help="THROUGHPUT (the big lever): generate self-play games across N CPU worker PROCESSES in "
                         "parallel (self-play is CPU-bound; measured ~13.5x at N=16 on 20 cores). opponent='self' "
                         "only. 1 = single-process (default); set ~cores-2 to saturate the CPU.")
    ap.add_argument("--parallel-games", type=int, default=d.parallel_games,
                    help="THROUGHPUT: generate self-play games in GPU-batched groups of this size "
                         "(one net forward does up to N leaf evals -> ~Nx fewer GPU round-trips, "
                         "more games/sec). Applies to opponent='self' only; teacher/mix stay "
                         "sequential. 1 = unchanged sequential path (default). Raise it to push "
                         "more learning per unit time; watch VRAM (bigger N = bigger eval batch).")
    ap.add_argument("--selfplay-sims", type=int, default=d.selfplay_sims)
    ap.add_argument("--temp-moves", type=int, default=d.temp_moves)
    ap.add_argument("--max-plies", type=int, default=d.max_plies)
    ap.add_argument("--game-wall-s", type=float, default=d.game_wall_s)
    ap.add_argument("--opening-mode", choices=["startpos", "book", "random", "mixed"],
                    default=d.opening_mode,
                    help="OPENING DIVERSITY for SELF-PLAY (the 'vary the starting conditions' "
                         "fix): startpos=old behaviour (every game from the same start -> the "
                         "net reinforces one rote line); book=curated sound openings; "
                         "random=guarded random plies; mixed=book + guarded random jitter "
                         "(DEFAULT). EVAL always uses startpos so the strength curve is "
                         "unaffected. See openings.py.")
    ap.add_argument("--opening-plies", type=int, default=d.opening_plies,
                    help="number of random plies for --opening-mode random, or random JITTER "
                         "plies on top of the book line for mixed (guarded so a game never "
                         "starts already-lost). Ignored for startpos/book. Default 4.")
    # train
    ap.add_argument("--train-steps", type=int, default=d.train_steps_per_iter)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--buffer-size", type=int, default=d.buffer_size)
    ap.add_argument("--value-loss-weight", type=float, default=d.value_loss_weight,
                    help="weight on the value-MSE term (default 0.5; <=1 keeps the large early "
                         "value loss from drowning the policy head -- mitigates value-head overfit)")
    ap.add_argument("--grad-clip", type=float, default=d.grad_clip,
                    help="global gradient-norm clip (default 5.0; one outlier batch can else "
                         "corrupt weights for many steps). Pre-clip norm is logged as grad_norm")
    ap.add_argument("--nan-abort", type=int, default=d.nan_abort,
                    help="non-finite losses in an iter before RAISE so --supervise restarts from "
                         "the last good checkpoint (silent weight-corruption guard; default 8)")
    # eval
    ap.add_argument("--eval-games", type=int, default=d.eval_games,
                    help="games for the PER-ITER gate eval, split rand/classical (default 30 -> "
                         "15/match -> 95%% CI ~+/-0.25; raise to >=120 for +/-~0.09 reliable gating). "
                         "The CI is logged + a WARN fires when it's too wide to gate on.")
    ap.add_argument("--seed-eval-games", type=int, default=d.seed_eval_games,
                    help="games for the ONE-TIME champion-floor seed eval (default 120). A larger "
                         "seed avoids a lucky read locking the monotonic floor too high (S5).")
    ap.add_argument("--forgetting-eval-interval", type=int, default=d.forgetting_eval_interval,
                    help="play the current net vs the FROZEN SEED (net_iter0) every N iters to "
                         "detect catastrophic forgetting/drift (the vs-random axis can't see it). "
                         "0 = off. Default 10. <0.5 sustained = forgetting.")
    ap.add_argument("--forgetting-eval-games", type=int, default=d.forgetting_eval_games,
                    help="games for the forgetting eval (0 = use --eval-games)")
    ap.add_argument("--eval-sims", type=int, default=d.eval_sims)
    ap.add_argument("--classical-depth", type=int, default=d.classical_depth)
    ap.add_argument("--floor-oom-retries", type=int, default=d.floor_oom_retries,
                    help="bounded OOM retries at the sims/batch FLOOR before raising so "
                         "--supervise restarts the worker (prevents a silent livelock when "
                         "halving can no longer free memory; default 10)")
    ap.add_argument("--seed", type=int, default=d.seed)
    # champion gate (TASK 2A) -- DEFAULT ON; monotonic never-regress promotion
    ap.add_argument("--champion-gate", dest="champion_gate", action="store_true",
                    default=True,
                    help="only promote a candidate that does NOT regress vs the "
                         "champion on winrate_vs_random (DEFAULT ON; makes latest.pt "
                         "monotonic -- never worse than the bootstrap)")
    ap.add_argument("--no-champion-gate", dest="champion_gate", action="store_false",
                    help="disable the champion gate (A/B: every candidate promoted, "
                         "the OLD degrading behaviour)")
    ap.add_argument("--champion-tol", type=float, default=d.champion_tol,
                    help="UPSIDE tie-band epsilon (default 0): a candidate ABOVE the "
                         "champion but within +tol counts as a TIE (decided by the "
                         "classical/loss tie-break), not a strict win. NOT a regression "
                         "band -- a candidate BELOW the champion's wr_random is ALWAYS "
                         "rejected, so the playable net is strictly monotonic.")
    ap.add_argument("--champion-h2h", dest="champion_h2h", action="store_true",
                    default=d.champion_h2h,
                    help="ALSO require candidate >55%% head-to-head vs the champion")
    ap.add_argument("--champion-h2h-games", type=int, default=d.champion_h2h_games)
    # self-play opponent (TASK 2B) -- refine vs the teacher / anchor to CLIMB
    ap.add_argument("--selfplay-opponent", choices=["self", "teacher", "mix"],
                    default=d.selfplay_opponent,
                    help="self=net-vs-net (classic AZ); teacher=net vs classical Engine "
                         "(learn from strong games); mix=alternate (DEFAULT self)")
    ap.add_argument("--selfplay-teacher-depth", type=int, default=d.selfplay_teacher_depth,
                    help="Engine(depth) for teacher/mix self-play games")
    ap.add_argument("--engine-path", type=str, default=d.engine_path,
                    help="OPTIONAL path to a real-world UCI engine binary (e.g. "
                         "stockfish.exe) to use as the teacher opponent. Default "
                         "'' => our in-repo classical Engine (zero setup, always "
                         "works). A bad path WARNs + falls back to classical. "
                         "ASYMMETRY: this engine is used for the SELF-PLAY TEACHER "
                         "ONLY. The strength EVAL (vs random + vs classical) ALWAYS "
                         "uses the in-repo classical Engine(depth=--classical-depth), "
                         "so the strength curve stays a stable, reproducible yardstick "
                         "independent of which teacher you train against.")
    ap.add_argument("--uci-movetime", dest="uci_movetime_ms", type=int,
                    default=d.uci_movetime_ms,
                    help="per-move time budget (ms) for a UCI teacher engine "
                         "(--engine-path); ignored by the in-repo classical engine. "
                         "Default 50ms.")
    ap.add_argument("--anchor-kl", type=float, default=d.anchor_kl,
                    help="KL(bootstrap||candidate) penalty weight to anchor the policy "
                         "to the strong base (0 = off; TASK 2B regularizer)")
    ap.add_argument("--no-teacher-distill", dest="teacher_distill", action="store_false",
                    default=d.teacher_distill,
                    help="DISABLE dense teacher grading. By default (ON) teacher/mix games ALSO "
                         "label the teacher's chosen move as a one-hot policy target (online "
                         "distillation -- the net imitates the teacher at every teacher move, not "
                         "just the sparse game outcome). Pass this to revert to net-moves-only "
                         "labelling.")
    # CURRICULUM (the moving target) -- default OFF; harder teacher as the net masters depth
    ap.add_argument("--curriculum", dest="curriculum", action="store_true",
                    default=d.curriculum,
                    help="ON: bump --selfplay-teacher-depth by 1 each time the champion's "
                         "draw-aware score_vs_classical crosses --curriculum-threshold at "
                         "the current depth (capped at --curriculum-max-depth), so the "
                         "teacher gets harder as the net climbs. DEFAULT OFF (behaviour "
                         "unchanged).")
    ap.add_argument("--curriculum-threshold", type=float, default=d.curriculum_threshold,
                    help="champion score_vs_classical that advances the curriculum to the "
                         "next teacher depth (default 0.6)")
    ap.add_argument("--curriculum-max-depth", type=int, default=d.curriculum_max_depth,
                    help="cap on the curriculum teacher-depth bumps (default 4)")
    # AUTO-BALANCE (throughput self-tuning) -- default OFF; the explicit params are used as-is
    ap.add_argument("--auto-balance", dest="auto_balance", action="store_true",
                    default=d.auto_balance,
                    help="LET THE SYSTEM self-determine the THROUGHPUT knobs (selfplay-workers / "
                         "games-per-iter / train-steps) from cores + the --max-hours budget, "
                         "balanced per unit time, instead of hand-tuning them. Workers <- cores-2 "
                         "(capped); each iter games/steps are nudged toward target iter-time = "
                         "max_hours/--auto-iters-in-budget while holding the replay ratio in band. "
                         "The learning contract (gate/anchor-kl/curriculum/lr/opponent) is NEVER "
                         "auto-tuned. VRAM blow-up stays guarded by the floor-OOM backoff. "
                         "DEFAULT OFF.")
    ap.add_argument("--auto-iters-in-budget", type=int, default=d.auto_iters_in_budget,
                    help="auto-balance target: >= this many champion-gate ATTEMPTS across the "
                         "--max-hours window (sets the target per-iter wall-time). Default 24.")
    ap.add_argument("--auto-replay-ratio", type=float, default=d.auto_replay_ratio,
                    help="auto-balance target replay ratio = (train_steps*batch)/(games*plies); "
                         "sane band 1-4. Couples train-steps to games. Default 2.0.")
    ap.add_argument("--skip-invariant-check", action="store_true",
                    help="skip the pre-training invariant gate (I13: value-sign / z-perspective / "
                         "bijection / terminals / target-integrity / never-hang / NaN-guard). "
                         "The gate runs by DEFAULT before training and HALTS (exit 2) on a broken "
                         "invariant. Only skip for fast inner-loop iteration.")
    ap.add_argument("--ckpt-dir", type=str, default=d.ckpt_dir,
                    help="checkpoint subdir under az/ (default robust_checkpoints; "
                         "use a fresh dir to refine from a bootstrap seed without "
                         "clobbering an existing self-play run)")
    ap.add_argument("--curve-path", type=str, default=d.curve_path,
                    help="strength-curve JSON path, relative to az/ (default "
                         "strength_curve.json -- the SHARED curve). Point a throwaway "
                         "run at a curve INSIDE its own ckpt-dir (e.g. "
                         "'<ckpt-dir>/strength_curve.json') so it does NOT interleave "
                         "rows with a concurrent live run's shared curve.")
    return ap


def cfg_from_args(args) -> RobustConfig:
    return RobustConfig(
        channels=args.channels, n_blocks=args.n_blocks,
        iterations=args.iters, games_per_iter=args.games_per_iter,
        parallel_games=args.parallel_games, selfplay_workers=args.selfplay_workers,
        selfplay_sims=args.selfplay_sims, temp_moves=args.temp_moves,
        max_plies=args.max_plies, game_wall_s=args.game_wall_s,
        opening_mode=args.opening_mode, opening_plies=args.opening_plies,
        train_steps_per_iter=args.train_steps, batch_size=args.batch_size,
        lr=args.lr, buffer_size=args.buffer_size,
        value_loss_weight=args.value_loss_weight, grad_clip=args.grad_clip,
        nan_abort=args.nan_abort,
        eval_games=args.eval_games, eval_sims=args.eval_sims,
        seed_eval_games=args.seed_eval_games,
        forgetting_eval_interval=args.forgetting_eval_interval,
        forgetting_eval_games=args.forgetting_eval_games,
        classical_depth=args.classical_depth, max_hours=args.max_hours,
        vram_cap_gb=args.vram_cap_gb, seed=args.seed,
        floor_oom_retries=args.floor_oom_retries,
        ckpt_dir=args.ckpt_dir, curve_path=args.curve_path,
        champion_gate=args.champion_gate, champion_tol=args.champion_tol,
        champion_h2h=args.champion_h2h, champion_h2h_games=args.champion_h2h_games,
        selfplay_opponent=args.selfplay_opponent,
        selfplay_teacher_depth=args.selfplay_teacher_depth,
        engine_path=args.engine_path,
        uci_movetime_ms=args.uci_movetime_ms,
        anchor_kl=args.anchor_kl,
        teacher_distill=args.teacher_distill,
        curriculum=args.curriculum,
        curriculum_threshold=args.curriculum_threshold,
        curriculum_max_depth=args.curriculum_max_depth,
        auto_balance=args.auto_balance,
        auto_iters_in_budget=args.auto_iters_in_budget,
        auto_replay_ratio=args.auto_replay_ratio,
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = cfg_from_args(args)
    print(f"[cfg] {asdict(cfg)}")
    # PRE-TRAINING INVARIANT GATE (I13, audit S4): mechanically re-check the catastrophic
    # correctness invariants before burning a multi-hour run. Lazy import to avoid an import
    # cycle (the gate imports from this module). HALT (exit 2) on any broken invariant.
    if not args.skip_invariant_check:
        try:
            from ..run_invariants_check import main as _invariants_gate
            print("[preflight] running pre-training invariant gate (I13)...")
            rc = _invariants_gate()
            if rc != 0:
                print("[preflight] INVARIANT GATE FAILED -- aborting training (exit 2). "
                      "Fix the broken invariant or pass --skip-invariant-check to override.")
                sys.exit(2)
            print("[preflight] invariant gate PASSED -- proceeding to train.")
        except SystemExit:
            raise
        except Exception as e:
            print(f"[preflight] WARN: invariant gate could not run ({type(e).__name__}: {e}); "
                  f"proceeding (pass --skip-invariant-check to silence)")
    if args.supervise:
        supervise(cfg, resume=args.resume)
    else:
        run_robust(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
