"""
chess_zero.az.selfplay_pool -- MULTIPROCESS self-play actors (the real throughput lever).

DIAGNOSIS (2026-06-08): self-play is CPU/Python-bound -- the MCTS tree ops + python-chess move generation dominate;
the net eval is microseconds. So a single self-play process pegs ~1 core and leaves the other ~19 idle, and GPU
leaf-batching (batched_selfplay.py) only trims GPU round-trips, it cannot saturate a GPU that was never the
bottleneck. The fix is the standard distributed-AlphaZero pattern: N WORKER PROCESSES each generating games in
parallel (one Python interpreter each -> no GIL contention), feeding the learner. On a 20-core box that is up to a
~10-16x self-play throughput multiplier -> proportionally more iters/hour -> a faster-climbing strength curve.

DESIGN:
  - Workers run on CPU (torch.set_num_threads(1) each, so N workers use N cores cleanly instead of N*cores BLAS
    threads oversubscribing). The net is small (~10.5M params); CPU eval is fine and avoids N CUDA contexts OOMing an
    8GB GPU. Within a worker, generate_selfplay_games_batched still batches that worker's games' leaf evals.
  - The worker fn is MODULE-LEVEL + picklable (Windows 'spawn' re-imports it). The net is shipped as a cpu state_dict
    + (channels, n_blocks); each worker rebuilds the net. Samples (numpy planes/pi + scalars) pickle back cleanly.
  - Per-worker seeds differ so games vary.

Self-play ONLY (net plays both sides) -- the batchable/parallelizable case; teacher/mix games stay sequential.
No emoji (Windows cp1252).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import List

from .selfplay import Sample


def _selfplay_worker(payload: dict) -> List[List["Sample"]]:
    """One actor process: rebuild the net on CPU, generate `n_games` self-play games, return their Sample lists.
    Module-level + picklable for the Windows spawn start method."""
    import torch
    torch.set_num_threads(1)  # CRITICAL: 1 BLAS thread/worker so N workers map to N cores (no oversubscription)
    from .net import AlphaZeroNet
    from .batched_selfplay import generate_selfplay_games_batched
    net = AlphaZeroNet(channels=payload["channels"], n_blocks=payload["n_blocks"])
    net.load_state_dict(payload["state_dict"], strict=False)
    net.eval()
    dev = torch.device("cpu")
    return generate_selfplay_games_batched(
        net, n_games=payload["n_games"], n_simulations=payload["sims"],
        temp_moves=payload["temp_moves"], max_plies=payload["max_plies"],
        game_wall_s=payload["game_wall_s"], device=dev,
        c_puct=payload.get("c_puct", 1.5), seed=payload["seed"],
        opening_mode=payload.get("opening_mode", "startpos"),
        opening_plies=payload.get("opening_plies", 4))


def generate_games_parallel(net, n_games: int, n_workers: int, sims: int = 64, temp_moves: int = 20,
                            max_plies: int = 120, game_wall_s: float = 90.0, seed_base: int = 0,
                            channels: int | None = None, n_blocks: int | None = None,
                            opening_mode: str = "startpos", opening_plies: int = 4) -> List[List["Sample"]]:
    """Generate `n_games` SELF-play games across `n_workers` processes (CPU). Returns a flat list of per-game Sample
    lists (z filled). Falls back to a single in-process batched run when n_workers<=1. The net's current weights are
    snapshotted to CPU and shipped to each worker, so workers play with THIS net (e.g. the champion).

    OPENING DIVERSITY (opening_mode != 'startpos'): forwarded to the batched generator so every worker's games
    start from distinct, sound openings -- the per-worker seeds already differ, so the openings differ across workers
    AND across games. Worker i samples openings from its own seed (seed_base + 1009*i)."""
    channels = channels if channels is not None else getattr(net, "channels", 80)
    n_blocks = n_blocks if n_blocks is not None else getattr(net, "n_blocks", 8)
    if n_workers <= 1:
        from .batched_selfplay import generate_selfplay_games_batched
        import torch
        return generate_selfplay_games_batched(net, n_games, sims, temp_moves, max_plies, game_wall_s,
                                               next(net.parameters()).device if any(True for _ in net.parameters())
                                               else torch.device("cpu"), seed=seed_base,
                                               opening_mode=opening_mode, opening_plies=opening_plies)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    per = max(1, math.ceil(n_games / n_workers))
    payloads = [dict(state_dict=state, channels=channels, n_blocks=n_blocks, n_games=per, sims=sims,
                     temp_moves=temp_moves, max_plies=max_plies, game_wall_s=game_wall_s, seed=seed_base + 1009 * i,
                     opening_mode=opening_mode, opening_plies=opening_plies)
                for i in range(n_workers)]
    ctx = mp.get_context("spawn")
    games: List[List["Sample"]] = []
    with ctx.Pool(processes=n_workers) as pool:
        for res in pool.map(_selfplay_worker, payloads):
            if res:
                games.extend(res)
    return games[:n_games]


def _teacher_worker(payload: dict) -> List[List["Sample"]]:
    """One actor process for TEACHER games: the net (MCTS) plays the in-repo classical Engine (high-quality data --
    the net learns FROM a stronger opponent, which is what stops the pure-self-play degradation). Returns the per-game
    Sample lists. With distill_teacher (default True) BOTH the net's moves (MCTS pi) AND the teacher's moves (one-hot
    DENSE GRADING) are labelled; z is the real outcome. Module-level + picklable for spawn."""
    import torch
    torch.set_num_threads(1)
    from .net import AlphaZeroNet
    from .train_robust import generate_selfplay_game_guarded  # lazy: avoid an import cycle at module load
    net = AlphaZeroNet(channels=payload["channels"], n_blocks=payload["n_blocks"])
    net.load_state_dict(payload["state_dict"], strict=False)
    net.eval()
    dev = torch.device("cpu")
    games: List[List["Sample"]] = []
    for k in range(payload["n_games"]):
        samples = generate_selfplay_game_guarded(
            net, n_simulations=payload["sims"], temp_moves=payload["temp_moves"],
            max_plies=payload["max_plies"], game_wall_s=payload["game_wall_s"], device=dev,
            opponent="teacher", teacher_depth=payload["teacher_depth"],
            net_is_white=((payload["seed"] + k) % 2 == 0),
            distill_teacher=payload.get("distill_teacher", True),
            opening_mode=payload.get("opening_mode", "startpos"),
            opening_plies=payload.get("opening_plies", 4),
            opening_seed=payload["seed"] + 100003 * k)
        games.append(samples)
    return games


def generate_teacher_games_parallel(net, n_games: int, n_workers: int, teacher_depth: int = 2, sims: int = 64,
                                    temp_moves: int = 20, max_plies: int = 160, game_wall_s: float = 90.0,
                                    seed_base: int = 0, channels: int | None = None,
                                    n_blocks: int | None = None,
                                    distill_teacher: bool = True,
                                    opening_mode: str = "startpos", opening_plies: int = 4) -> List[List["Sample"]]:
    """Generate `n_games` NET-vs-CLASSICAL teacher games across `n_workers` CPU processes. This is the QUALITY path
    (learn from a stronger opponent) at multiprocess speed -- the cure for pure-self-play degradation. Self-play's
    `generate_games_parallel` is the SPEED-only path; a dual-learning iter mixes both. distill_teacher (default True)
    forwards DENSE TEACHER GRADING (one-hot label on the teacher's moves) into the guarded generator.

    OPENING DIVERSITY (opening_mode != 'startpos'): forwarded so teacher games ALSO start from distinct sound
    openings (each game seeds its opening from the worker seed + a per-game offset)."""
    channels = channels if channels is not None else getattr(net, "channels", 80)
    n_blocks = n_blocks if n_blocks is not None else getattr(net, "n_blocks", 8)
    if n_workers <= 1:
        from .train_robust import generate_selfplay_game_guarded
        import torch
        dev = next(net.parameters()).device if any(True for _ in net.parameters()) else torch.device("cpu")
        out = []
        for k in range(n_games):
            out.append(generate_selfplay_game_guarded(net, sims, temp_moves, max_plies, game_wall_s, dev,
                                                      opponent="teacher", teacher_depth=teacher_depth,
                                                      net_is_white=(k % 2 == 0),
                                                      distill_teacher=distill_teacher,
                                                      opening_mode=opening_mode, opening_plies=opening_plies,
                                                      opening_seed=seed_base + 100003 * k))
        return out
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    per = max(1, math.ceil(n_games / n_workers))
    payloads = [dict(state_dict=state, channels=channels, n_blocks=n_blocks, n_games=per, sims=sims,
                     temp_moves=temp_moves, max_plies=max_plies, game_wall_s=game_wall_s,
                     teacher_depth=teacher_depth, distill_teacher=distill_teacher,
                     seed=seed_base + 1009 * i,
                     opening_mode=opening_mode, opening_plies=opening_plies) for i in range(n_workers)]
    ctx = mp.get_context("spawn")
    games: List[List["Sample"]] = []
    with ctx.Pool(processes=n_workers) as pool:
        for res in pool.map(_teacher_worker, payloads):
            if res:
                games.extend(res)
    return games[:n_games]


def _eval_worker(payload: dict) -> dict:
    """One actor process for EVALUATION: plays payload['n_games'] of (net greedy MCTS) vs an opponent
    ('random'|'classical') by calling the SAME verified _play_match (n_workers=1 -> sequential in the
    worker, no recursion). Returns its {win,draw,loss,games,...} dict. Module-level + picklable for
    spawn. Reusing _play_match (not a re-implementation) keeps eval semantics identical to sequential."""
    import torch
    torch.set_num_threads(1)
    import numpy as np
    from .net import AlphaZeroNet
    from .train_robust import _play_match  # lazy: avoid an import cycle at module load
    net = AlphaZeroNet(channels=payload["channels"], n_blocks=payload["n_blocks"])
    net.load_state_dict(payload["state_dict"], strict=False)
    net.eval()
    rng = np.random.default_rng(payload["seed"])
    return _play_match(net, payload["opponent"], payload["n_games"], payload["eval_sims"],
                       payload["max_plies"], payload["game_wall_s"], payload["classical_depth"],
                       torch.device("cpu"), rng, n_workers=1)


def play_match_parallel(net, opponent: str, n_games: int, eval_sims: int, max_plies: int,
                        game_wall_s: float, classical_depth: int, n_workers: int,
                        seed_base: int = 0, channels: int | None = None,
                        n_blocks: int | None = None) -> dict:
    """Parallel EVALUATION: split n_games across n_workers CPU processes (each runs the verified
    _play_match on its chunk), then AGGREGATE the win/draw/loss counts. Eval was the 2nd-biggest
    per-iter cost (~150s/8 games sequential); this cuts it ~Nx. Colour balance is preserved because
    each chunk alternates net's colour internally and the chunk sizes are near-equal. Returns the
    SAME dict shape as _play_match."""
    channels = channels if channels is not None else getattr(net, "channels", 80)
    n_blocks = n_blocks if n_blocks is not None else getattr(net, "n_blocks", 8)
    n_workers = max(1, min(n_workers, n_games))
    base, rem = divmod(n_games, n_workers)
    chunks = [base + (1 if i < rem else 0) for i in range(n_workers)]
    chunks = [c for c in chunks if c > 0]
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    payloads = [dict(state_dict=state, channels=channels, n_blocks=n_blocks, opponent=opponent,
                     n_games=chunks[i], eval_sims=eval_sims, max_plies=max_plies,
                     game_wall_s=game_wall_s, classical_depth=classical_depth,
                     seed=seed_base + 7919 * i) for i in range(len(chunks))]
    ctx = mp.get_context("spawn")
    win = draw = loss = games = 0
    with ctx.Pool(processes=len(payloads)) as pool:
        for r in pool.map(_eval_worker, payloads):
            if r:
                win += r["win"]; draw += r["draw"]; loss += r["loss"]; games += r["games"]
    return {"win": win, "draw": draw, "loss": loss, "games": games,
            "win_rate": win / games if games else 0.0,
            "score": (win + 0.5 * draw) / games if games else 0.0}


def _bench(n_games_each: int = 8):
    """Quick throughput benchmark: 1 worker (single-process) vs N workers. Run from repo ROOT:
       .venv/Scripts/python.exe -m az.selfplay_pool"""
    import os
    import torch
    from .net import AlphaZeroNet
    cores = os.cpu_count() or 4
    n_workers = max(2, min(16, cores - 2))
    net = AlphaZeroNet(channels=80, n_blocks=8).eval()  # bootstrap-size net
    cfg = dict(sims=48, temp_moves=10, max_plies=80, game_wall_s=60.0)

    t0 = time.time()
    g1 = generate_games_parallel(net, n_games=n_games_each, n_workers=1, seed_base=1, **cfg)
    dt1 = time.time() - t0
    gps1 = len(g1) / dt1 if dt1 else 0.0

    t0 = time.time()
    gN = generate_games_parallel(net, n_games=n_games_each * n_workers, n_workers=n_workers, seed_base=2, **cfg)
    dtN = time.time() - t0
    gpsN = len(gN) / dtN if dtN else 0.0

    print(f"[bench] cores={cores}  net=80/8  sims={cfg['sims']}")
    print(f"  1 worker : {len(g1)} games in {dt1:.1f}s -> {gps1:.3f} games/s")
    print(f"  {n_workers} workers: {len(gN)} games in {dtN:.1f}s -> {gpsN:.3f} games/s")
    print(f"  SPEEDUP  : {gpsN / gps1:.1f}x" if gps1 else "  SPEEDUP  : n/a")
    # validity: every game produced >=1 sample with signed z
    ok = all(len(g) >= 1 and all(s.z in (-1.0, 0.0, 1.0) for s in g) for g in gN)
    print(f"  validity : {'OK -- all games have valid signed-z samples' if ok else 'FAIL'}")
    return gps1, gpsN


if __name__ == "__main__":
    _bench()
