"""
chess_zero.az.bootstrap_supervised -- SUPERVISED IMITATION bootstrap of the
AlphaZero net by learning from the CLASSICAL engine (engine.py) as the ORACLE.

WHY THIS EXISTS (the honest situation):
    Pure self-play-FROM-SCRATCH (train_robust.py) LEARNS its loss (3.38 -> 1.91)
    but hit a COMPUTE CEILING on a single RTX 4060: after ~100 iterations the net
    still scored 0.00 vs a random mover. RL-from-scratch needs orders of magnitude
    more self-play games to discover even basic tactics.

    Imitation learning is FAR more sample-efficient: instead of rediscovering chess
    from sparse win/loss signal, we directly TEACH the net to copy a teacher that
    ALREADY plays at a very good level (the alpha-beta classical engine, which
    crushes random 100% by checkmate). A few tens of thousands of labelled
    positions move the strength needle in MINUTES, not hours.

THE METHOD:
    1. GENERATE a labelled position set with the classical engine as oracle:
         - classical(depth=Dgen) vs classical(depth=Dgen) games   (good lines)
         - classical(depth=Dgen) vs random games                  (how to punish junk)
         - random-rollout positions                               (diversity / off-policy)
       For EACH visited position we record:
         policy target = the classical engine's chosen move -> move_to_index  (a
                         hard class label; optionally soft mass on the top-1)
         value  target = tanh(engine_score_cp / VALUE_SCALE), side-to-move POV,
                         in [-1, 1]  (the engine's own static/lookahead assessment)
       Positions are DEDUPED by FEN (first-seen wins).

    2. SUPERVISED-TRAIN AlphaZeroNet:
         loss = CrossEntropy(policy_logits, target_move_idx)
              + VALUE_WEIGHT * MSE(tanh(value), target_value)
       AMP / mixed precision, 4060-8GB-safe. The net is C=80 / B=8 to MATCH
       train_robust.py so the produced checkpoint is schema-compatible with the
       self-play loop AND play.py (so self-play can REFINE from this strong base).

    3. CHECKPOINT to az/bootstrap_checkpoints/ (NEVER clobbers robust_checkpoints/).
       The checkpoint is written in the SAME payload shape train_robust.load_checkpoint
       expects (iter / channels / n_blocks / state_dict / optimizer / rng / config /
       buffer) so `train_robust.py --resume` can pick it up and refine.

HONEST CEILING (binding): this is IMITATION. The net APPROACHES but should not
exceed its teacher. The honest ceiling = the classical engine's strength. No
master/superhuman claims. eval_bootstrap.py reports REAL win-rates.

Run (the real bootstrap, ~15-25 min on a 4060):
    .venv\\Scripts\\python.exe -m az.bootstrap_supervised \\
        --target-positions 40000 --epochs 8 --gen-depth 3

    # quick smoke (machinery only):
    .venv\\Scripts\\python.exe -m az.bootstrap_supervised \\
        --target-positions 600 --epochs 2 --gen-depth 1 --self-games 4 \\
        --vs-random-games 4 --random-rollouts 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import chess
import torch
import torch.nn.functional as F

from .encoding import N_INPUT_PLANES, N_POLICY, board_to_planes, move_to_index
from .net import AlphaZeroNet, count_params
from chess_engine.engine import Engine


HERE = os.path.dirname(os.path.abspath(__file__))

# Value squashing scale: engine scores are centipawns (side-to-move relative).
# tanh(cp / 400) maps ~+/-4 pawns to ~+/-0.76, a full rook (~5p) to ~0.85, and a
# mate-score (1e6) saturates to ~1.0. 400cp is a standard "win-prob" pawn scale.
VALUE_SCALE = 400.0


@dataclass
class BootstrapConfig:
    # net (MUST match train_robust.py defaults so the ckpt is schema-compatible)
    channels: int = 80
    n_blocks: int = 8
    # data generation
    #   The classical engine (pure Python) is the bottleneck: depth=1 no-quiescence
    #   ~17 searches/s, shallow-quiescence (qd=1) ~11/s, depth=2 ~7/s on a 4060 box.
    #   For IMITATION we want a teacher that does not blunder to a capture but is
    #   FAST -- depth=1 + shallow quiescence (qd=1) is the sweet spot (sound enough
    #   to crush random; ~10x faster than the default depth=4/qd=6 engine).
    target_positions: int = 22000     # stop generating once we have this many DEDUPED
    gen_depth: int = 1                # classical teacher search depth for LABELS
    gen_quiescence_depth: int = 1     # teacher quiescence cap (0 = off; speed/quality knob)
    self_games: int = 90              # classical-vs-classical games
    vs_random_games: int = 90         # classical-vs-random games
    random_rollouts: int = 24         # random-vs-random games (diversity positions)
    max_plies: int = 120              # per game ply cap
    opening_temp_plies: int = 6       # first N plies: teacher picks a RANDOM top-k move
    opening_topk: int = 3             # diversity at the root so games are not identical
    # training
    epochs: int = 8
    batch_size: int = 256
    lr: float = 1e-3
    l2: float = 1e-4
    value_weight: float = 1.0         # weight on the value MSE term
    val_frac: float = 0.05            # held-out fraction (sanity only; imitation)
    # io
    ckpt_dir: str = "bootstrap_checkpoints"
    data_cache: str = "bootstrap_data.npz"
    seed: int = 0


# --------------------------------------------------------------------------- #
# Data generation: the classical engine as the oracle
# --------------------------------------------------------------------------- #
def _opening_diversity_move(engine: Engine, board: chess.Board, best: chess.Move,
                            topk: int, rng: random.Random) -> chess.Move:
    """Pick a RANDOM move among the teacher's top-k for opening diversity (so the
    generated games are not all identical). Cheap: we shuffle the legal moves and
    evaluate at most a handful via a quiescence-only static look (one push + eval),
    then sample from the top-k. Used ONLY for the move PLAYED, never the LABEL."""
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return best
    # cheap scoring: static eval of the child (no recursive search) -> our POV
    rng.shuffle(moves)
    cand = moves[:max(topk * 2, 6)]  # cap the candidate set so this stays cheap
    scored = []
    for m in cand:
        board.push(m)
        try:
            sub = engine.search(board)   # depth/quiescence as configured (cheap at d=1)
            scored.append((m, -sub.score))
        finally:
            board.pop()
    scored.sort(key=lambda t: t[1], reverse=True)
    pool = [m for m, _ in scored[:max(1, topk)]]
    return rng.choice(pool) if pool else best


def _label_position(board: chess.Board, engine: Engine,
                    out: Dict[str, Tuple[np.ndarray, int, float]]
                    ) -> Optional[chess.Move]:
    """Label a single position with the teacher's best move + value, dedup by FEN.

    Stores out[fen] = (planes, policy_index, value_target) and RETURNS the teacher's
    best move (so the caller can reuse it as the move to play -- avoiding a second
    redundant search). The label move is the engine's TRUE best move at gen_depth.
    Returns None only if the position is terminal / unencodable."""
    if board.is_game_over(claim_draw=True):
        return None
    res = engine.search(board)
    if res.move is None:
        return None
    fen = board.fen()
    if fen not in out:  # dedup: first-seen label wins
        idx = move_to_index(board, res.move)
        if idx is not None:
            planes = board_to_planes(board)
            value = float(np.tanh(res.score / VALUE_SCALE))  # stm-relative, [-1,1]
            out[fen] = (planes, idx, value)
    return res.move


def _play_labelled_game(label_engine: Engine, white_kind: str, black_kind: str,
                        cfg: BootstrapConfig, rng: random.Random,
                        out: Dict[str, Tuple[np.ndarray, int, float]]) -> int:
    """Play one game (kinds in {'classical','random'}), labelling EVERY position
    with the teacher (label_engine) before the move is made. Returns #new labels.

    Efficiency: the teacher's best move from the LABEL search is REUSED as the
    classical side's played move (no second search), except in the opening plies
    where we deliberately diversify via a shallow top-k re-score."""
    board = chess.Board()
    n0 = len(out)
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < cfg.max_plies:
        # ALWAYS label the current position with the teacher (oracle); reuse its move.
        best = _label_position(board, label_engine, out)
        if best is None:
            break
        kind = white_kind if board.turn == chess.WHITE else black_kind
        if kind == "random":
            move = rng.choice(list(board.legal_moves))
        elif ply < cfg.opening_temp_plies:
            # opening diversity: random pick among the teacher's top-k (cheap)
            move = _opening_diversity_move(label_engine, board, best,
                                           cfg.opening_topk, rng)
        else:
            move = best  # REUSE the label search -- no redundant second search
        board.push(move)
        ply += 1
        if len(out) >= cfg.target_positions:
            break
    return len(out) - n0


def _build_plan(cfg: BootstrapConfig, rng: random.Random) -> List[Tuple[str, str]]:
    """The interleaved game plan (so an early target-hit still has all flavours)."""
    plan: List[Tuple[str, str]] = []
    for _ in range(cfg.self_games):
        plan.append(("classical", "classical"))
    for i in range(cfg.vs_random_games):
        plan.append(("classical", "random") if i % 2 == 0 else ("random", "classical"))
    for _ in range(cfg.random_rollouts):
        plan.append(("random", "random"))
    rng.shuffle(plan)
    return plan


def _gen_worker(args) -> Dict[str, Tuple[np.ndarray, int, float]]:
    """A process-pool worker (must be module-level so Windows 'spawn' can pickle it).
    Plays its slice of the plan with its own seed and returns its labelled dict.
    Each worker stops once it has produced its per-worker position quota."""
    cfg, sub_plan, seed, quota = args
    rng = random.Random(seed)
    label_engine = Engine(depth=cfg.gen_depth,
                          quiescence=(cfg.gen_quiescence_depth > 0),
                          q_max_depth=max(1, cfg.gen_quiescence_depth))
    out: Dict[str, Tuple[np.ndarray, int, float]] = {}
    for (wk, bk) in sub_plan:
        if len(out) >= quota:
            break
        # local target so the worker stops near its quota (cfg is shared -> override)
        local_cfg = cfg
        _play_labelled_game(label_engine, wk, bk, local_cfg, rng, out)
        if len(out) >= quota:
            break
    return out


def generate_dataset(cfg: BootstrapConfig, workers: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the labelled position set. Returns (planes, idxs, values) arrays.

    The classical engine (pure Python) is the throughput bottleneck (~4-5 pos/s
    single-process). Games are INDEPENDENT, so workers>1 fans generation across a
    process pool and dedups in the parent -- the key lever to land in budget."""
    t0 = time.time()
    rng = random.Random(cfg.seed)
    plan = _build_plan(cfg, rng)

    if workers <= 1:
        # single-process path (also the fallback)
        label_engine = Engine(depth=cfg.gen_depth,
                              quiescence=(cfg.gen_quiescence_depth > 0),
                              q_max_depth=max(1, cfg.gen_quiescence_depth))
        out: Dict[str, Tuple[np.ndarray, int, float]] = {}
        games_played = 0
        for (wk, bk) in plan:
            if len(out) >= cfg.target_positions:
                break
            _play_labelled_game(label_engine, wk, bk, cfg, rng, out)
            games_played += 1
            if games_played % 10 == 0:
                dt = time.time() - t0
                print(f"[gen] games={games_played}/{len(plan)} positions={len(out)} "
                      f"({len(out)/max(1e-9,dt):.0f} pos/s, {dt:.0f}s)", flush=True)
    else:
        # parallel path: split the plan across workers; each gets a per-worker quota
        import multiprocessing as mp
        per_worker_quota = max(1, cfg.target_positions // workers + cfg.target_positions // (4 * workers))
        chunks: List[List[Tuple[str, str]]] = [plan[i::workers] for i in range(workers)]
        jobs = [(cfg, chunks[i], cfg.seed + 1000 * (i + 1), per_worker_quota)
                for i in range(workers)]
        print(f"[gen] parallel: {workers} workers, ~{per_worker_quota} pos/worker "
              f"(target {cfg.target_positions})", flush=True)
        out = {}
        ctx = mp.get_context("spawn")  # Windows-safe
        with ctx.Pool(processes=workers) as pool:
            for wi, sub in enumerate(pool.imap_unordered(_gen_worker, jobs)):
                out.update(sub)  # dedup across workers (later writes overwrite; fine)
                dt = time.time() - t0
                print(f"[gen] worker {wi+1}/{workers} returned {len(sub)} -> "
                      f"total {len(out)} positions ({len(out)/max(1e-9,dt):.0f} pos/s, "
                      f"{dt:.0f}s)", flush=True)

    planes = np.stack([v[0] for v in out.values()]).astype(np.float32)
    idxs = np.array([v[1] for v in out.values()], dtype=np.int64)
    values = np.array([v[2] for v in out.values()], dtype=np.float32)
    dt = time.time() - t0
    print(f"[gen] DONE: {len(out)} unique positions in {dt:.0f}s "
          f"({len(out)/max(1e-9,dt):.0f} pos/s)", flush=True)
    print(f"[gen] value target stats: mean={values.mean():.3f} std={values.std():.3f} "
          f"min={values.min():.3f} max={values.max():.3f}", flush=True)
    return planes, idxs, values


# --------------------------------------------------------------------------- #
# Supervised training (CE policy + MSE value, AMP)
# --------------------------------------------------------------------------- #
def train_supervised(cfg: BootstrapConfig, planes: np.ndarray, idxs: np.ndarray,
                     values: np.ndarray, device) -> Tuple[AlphaZeroNet, dict]:
    n = planes.shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    planes_t = torch.from_numpy(planes)          # CPU; batched to GPU per step
    idxs_t = torch.from_numpy(idxs)
    values_t = torch.from_numpy(values)

    net = AlphaZeroNet(channels=cfg.channels, n_blocks=cfg.n_blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"[train] net C={cfg.channels}/B={cfg.n_blocks} params={count_params(net):,} "
          f"train={len(tr_idx)} val={len(val_idx)} amp={use_amp}", flush=True)

    def run_eval() -> Tuple[float, float, float]:
        net.eval()
        ce_sum = mse_sum = top1 = 0.0
        with torch.no_grad():
            for s in range(0, len(val_idx), cfg.batch_size):
                bidx = val_idx[s:s + cfg.batch_size]
                xb = planes_t[bidx].to(device, non_blocking=True)
                yb = idxs_t[bidx].to(device, non_blocking=True)
                vb = values_t[bidx].to(device, non_blocking=True).unsqueeze(1)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, value = net(xb)
                    ce = F.cross_entropy(logits, yb, reduction="sum")
                    mse = F.mse_loss(value, vb, reduction="sum")
                ce_sum += float(ce.item())
                mse_sum += float(mse.item())
                top1 += float((logits.argmax(1) == yb).sum().item())
        m = max(1, len(val_idx))
        return ce_sum / m, mse_sum / m, top1 / m

    history = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        net.train()
        ep_perm = rng.permutation(len(tr_idx))
        tr_shuf = tr_idx[ep_perm]
        run_ce = run_mse = seen = 0.0
        steps = 0
        for s in range(0, len(tr_shuf), cfg.batch_size):
            bidx = tr_shuf[s:s + cfg.batch_size]
            xb = planes_t[bidx].to(device, non_blocking=True)
            yb = idxs_t[bidx].to(device, non_blocking=True)
            vb = values_t[bidx].to(device, non_blocking=True).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, value = net(xb)
                ce = F.cross_entropy(logits, yb)
                mse = F.mse_loss(value, vb)
                loss = ce + cfg.value_weight * mse
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = len(bidx)
            run_ce += float(ce.item()) * bs
            run_mse += float(mse.item()) * bs
            seen += bs
            steps += 1
        v_ce, v_mse, v_top1 = run_eval()
        row = {"epoch": ep, "train_ce": run_ce / max(1, seen),
               "train_mse": run_mse / max(1, seen), "val_ce": v_ce,
               "val_mse": v_mse, "val_top1_move_acc": v_top1,
               "wall_s": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[train] epoch {ep}: train_ce={row['train_ce']:.4f} "
              f"train_mse={row['train_mse']:.4f} | val_ce={v_ce:.4f} "
              f"val_mse={v_mse:.4f} val_top1_move_acc={v_top1:.3f} "
              f"({row['wall_s']:.0f}s)", flush=True)

    return net, {"history": history, "n_positions": int(n),
                 "n_train": int(len(tr_idx)), "n_val": int(len(val_idx))}


# --------------------------------------------------------------------------- #
# Checkpoint: write in the shape train_robust.load_checkpoint() expects, so
# `train_robust.py --resume` can pick it up and REFINE from this strong base.
# --------------------------------------------------------------------------- #
def _atomic_torch_save(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_bootstrap_checkpoint(ckpt_dir: str, net: AlphaZeroNet, opt,
                              cfg: BootstrapConfig, meta: dict) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    # iter=-1 so that train_robust resumes at iter 0 (start_iter = last_iter + 1).
    payload = {
        "iter": -1,
        "channels": cfg.channels,
        "n_blocks": cfg.n_blocks,
        "state_dict": net.state_dict(),
        "optimizer": opt.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "config": asdict(cfg),
        "buffer": [],   # bootstrap leaves the self-play replay buffer empty
        "bootstrap_meta": meta,
        "source": "bootstrap_supervised",
    }
    path = os.path.join(ckpt_dir, "net_bootstrap.pt")
    _atomic_torch_save(payload, path)
    # latest pointer in train_robust's format (so resume's find_latest works here too)
    _atomic_torch_save({"iter": -1, "path": os.path.basename(path)},
                       os.path.join(ckpt_dir, "latest.json.tmp.pt"))
    os.replace(os.path.join(ckpt_dir, "latest.json.tmp.pt"),
               os.path.join(ckpt_dir, "latest.pt"))
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    d = BootstrapConfig()
    ap = argparse.ArgumentParser(
        description="Supervised imitation bootstrap of the AZ net from the classical engine.")
    ap.add_argument("--channels", type=int, default=d.channels)
    ap.add_argument("--n-blocks", type=int, default=d.n_blocks)
    ap.add_argument("--target-positions", type=int, default=d.target_positions)
    ap.add_argument("--gen-depth", type=int, default=d.gen_depth)
    ap.add_argument("--gen-quiescence-depth", type=int, default=d.gen_quiescence_depth,
                    help="teacher quiescence cap (0=off; speed/quality knob)")
    ap.add_argument("--self-games", type=int, default=d.self_games)
    ap.add_argument("--vs-random-games", type=int, default=d.vs_random_games)
    ap.add_argument("--random-rollouts", type=int, default=d.random_rollouts)
    ap.add_argument("--max-plies", type=int, default=d.max_plies)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--value-weight", type=float, default=d.value_weight)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--workers", type=int, default=0,
                    help="data-gen process workers (0 = auto = cpu_count-1; 1 = serial)")
    ap.add_argument("--reuse-data", action="store_true",
                    help="reuse cached dataset (bootstrap_data.npz) if present")
    ap.add_argument("--ckpt-dir", type=str, default=d.ckpt_dir,
                    help="output dir under az/ (default bootstrap_checkpoints). Use a NEW dir (e.g. bootstrap_d4) to "
                         "build a STRONGER-TEACHER bootstrap WITHOUT clobbering the existing one -- the ceiling lever: "
                         "imitate classical --gen-depth 3/4 (a much stronger teacher) so the net starts far stronger.")
    return ap


def cfg_from_args(args) -> BootstrapConfig:
    return BootstrapConfig(
        channels=args.channels, n_blocks=args.n_blocks,
        target_positions=args.target_positions, gen_depth=args.gen_depth,
        gen_quiescence_depth=args.gen_quiescence_depth,
        self_games=args.self_games, vs_random_games=args.vs_random_games,
        random_rollouts=args.random_rollouts, max_plies=args.max_plies,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        value_weight=args.value_weight, seed=args.seed, ckpt_dir=args.ckpt_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = cfg_from_args(args)
    print(f"[cfg] {asdict(cfg)}", flush=True)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})",
          flush=True)

    ckpt_dir = os.path.join(HERE, cfg.ckpt_dir)
    cache_path = os.path.join(ckpt_dir, cfg.data_cache)

    # ---- data ----
    if args.reuse_data and os.path.exists(cache_path):
        print(f"[gen] reusing cached dataset {cache_path}", flush=True)
        z = np.load(cache_path)
        planes, idxs, values = z["planes"], z["idxs"], z["values"]
        print(f"[gen] cached: {planes.shape[0]} positions", flush=True)
    else:
        workers = args.workers
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 2) - 1)
        planes, idxs, values = generate_dataset(cfg, workers=workers)
        os.makedirs(ckpt_dir, exist_ok=True)
        np.savez_compressed(cache_path, planes=planes, idxs=idxs, values=values)
        print(f"[gen] cached dataset -> {cache_path}", flush=True)

    # ---- train ----
    net, meta = train_supervised(cfg, planes, idxs, values, device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

    # ---- checkpoint (train_robust-resumable) ----
    path = save_bootstrap_checkpoint(ckpt_dir, net, opt, cfg, meta)
    print(f"[ckpt] bootstrap checkpoint -> {path}", flush=True)
    # also write a small json summary next to it
    summary = {"config": asdict(cfg), "meta": meta, "ckpt": path,
               "value_scale": VALUE_SCALE}
    with open(os.path.join(ckpt_dir, "bootstrap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] supervised bootstrap complete. "
          f"final val_top1_move_acc="
          f"{meta['history'][-1]['val_top1_move_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
