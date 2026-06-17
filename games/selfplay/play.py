"""
chess_zero.play -- the SINGLE front door. Play manually, watch AI-vs-AI, or run AI self-play WHILE LEARNING.

The 3 modes you most likely want:
    human   : YOU play vs the engine, interactively (type moves, see the board).
    watch   : the AI plays ITSELF (classical engine vs engine) to a finished game.
    learn   : AI SELF-PLAY WHILE LEARNING -- the AlphaZero loop (self-play -> train the net -> eval); RESUMES
              from the supervised-bootstrap checkpoint if present, so self-play REFINES a strong base.
    learn-watch : the INTEGRATED "watch it learn" experience -- kick off DUAL learning (self-play AND vs the
              classical engine) in the BACKGROUND and WATCH live games where the LATEST (successively-stronger)
              net plays the classical engine (default), itself, or random -- rendered on a LIVE VISUAL BOARD --
              with the strength curve advancing after each game. ONE command: launch -> watch the agent play a
              real opponent -> see it improve. Ctrl-C tears down.

The LEARNED-NET modes (use the supervised-bootstrap checkpoint az/bootstrap_checkpoints/net_bootstrap.pt):
    net-play  : YOU play vs the LEARNED net (net-only argmax policy, or --mcts for net+search).
    net-watch : WATCH the learned net play the classical engine (or random) to a finished game.

Plus the engine diagnostics:
    vs-random : the litmus (engine must crush a random mover).
    vs-self   : engine-vs-engine self-play (same as `watch`).
    position  : feed a FEN, get the engine's move.
    bench     : nodes/time per move sanity.

Every game runs to a real termination (checkmate / stalemate / insufficient material / 75-move / fivefold) --
python-chess decides; we never invent a result.

Usage:
    python play.py human  [--engine-white] [--depth 4] [--ascii]      # play manually
    python play.py watch  [--depth 4]                                  # AI vs AI
    python play.py learn  [--bootstrap]                               # AI self-play + learning (AlphaZero)
    python play.py learn-watch [--ckpt-dir _demo_test] [--watch classical|self|random] \
                               [--train-opponent self|mix|teacher] [--engine-depth 2] [--engine-path PATH] \
                               [--mcts] [--move-delay 0.3] [--no-viz] [--board] [--no-board] \
                               [--max-games N] [--iters N] [--games-per-iter N] [--eval-games N] \
                               [--selfplay-sims N] [--max-hours H] [--no-train]   # INTEGRATED: watch it learn
                               # --viz (DEFAULT ON): TRUE browser board (real pieces, live) at az/<ckpt-dir>/live.html
    python play.py net-play  [--ckpt <path>] [--mcts] [--mcts-sims 64] [--net-white] [--ascii]
    python play.py net-watch [--ckpt <path>] [--mcts] [--opponent classical|random] [--depth 4] \
                             [--live] [--move-delay 0.3] [--board]
    python play.py vs-random --games 20 [--depth 4] [--seed 0] [--print-game]
    python play.py position  --fen "<FEN>" [--depth 4]
    python play.py bench      [--depth 4]
"""
from __future__ import annotations

import argparse
import os
import random
import signal
import subprocess
import sys
import time

# --- repo-root bootstrap -------------------------------------------------------
# selfplay/play.py lives one level under the games-engine root. Put that root on
# sys.path so az/ and chess_engine/ import as top-level packages -- whether run as
# 'python selfplay/play.py', '-m selfplay.play', or spawned by run_engine.py.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # games-engine root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess

from chess_engine.engine import Engine, best_move, MATE_THRESHOLD

_HERE = _ROOT                  # games-engine root (where az/ lives; was the chess_zero dir)
_REPO_ROOT = _ROOT             # subprocess cwd, so a spawned 'python -m az.X' resolves
_BOOTSTRAP_CKPT = os.path.join(_HERE, "az", "bootstrap_checkpoints", "net_bootstrap.pt")
# Self-play refinement of the bootstrap uses its OWN dir so it never clobbers (or
# inherits) the old from-scratch robust_checkpoints/ run.
_REFINE_DIR_NAME = "robust_from_bootstrap"
_REFINE_CKPT_DIR = os.path.join(_HERE, "az", _REFINE_DIR_NAME)


def _run_learn(from_bootstrap: bool = True) -> None:
    """Launch the HARDENED AlphaZero self-play->train->eval loop (train_robust.py).

    With a supervised-bootstrap checkpoint present (default), we SEED a DEDICATED
    refinement dir (az/robust_from_bootstrap/) with the bootstrap weights as
    net_iter0, then run train_robust pointed at that dir (--ckpt-dir). Self-play
    auto-resumes from the strong imitation base and REFINES it -- instead of
    cold-starting from random (which hit the compute ceiling at 0.00 vs random).
    The old from-scratch robust_checkpoints/ run is left untouched.
    """
    print("Launching AlphaZero SELF-PLAY + LEARNING (real self-play -> train -> eval vs random + classical).")
    ckpt_dir_arg = "robust_checkpoints"
    if from_bootstrap and os.path.exists(_BOOTSTRAP_CKPT):
        _seed_refine_from_bootstrap(_REFINE_CKPT_DIR)
        ckpt_dir_arg = _REFINE_DIR_NAME
        print(f"Refining the SUPERVISED-BOOTSTRAP base ({_BOOTSTRAP_CKPT})")
        print(f"  -> self-play checkpoints go to az/{_REFINE_DIR_NAME}/ (resumes the strong base).\n")
    elif from_bootstrap:
        print("No bootstrap checkpoint found -- cold-starting self-play from random weights.")
        print("TIP: run  python -m az.bootstrap_supervised  first for a strong base.\n")
    else:
        print("--no-bootstrap: resuming the from-scratch robust_checkpoints/ run.\n")
    # FIX (2026-06-08): isolate the strength curve INSIDE the ckpt-dir, exactly like
    # learn-watch does. Without --curve-path, train_robust defaults to az/strength_curve.json
    # -- the SHARED curve that already holds 111 dead from-scratch rows (schema clash); the
    # bootstrap-refine run was appending its rows there and polluting it. A per-ckpt-dir
    # curve keeps this run's progress self-contained.
    curve_rel = os.path.join(ckpt_dir_arg, "strength_curve.json")
    subprocess.run([sys.executable, "-m", "az.train_robust",
                    "--ckpt-dir", ckpt_dir_arg,
                    "--curve-path", curve_rel],
                   cwd=_REPO_ROOT)  # repo root so the `projects.*` package import resolves


def _seed_refine_from_bootstrap(target_dir: str = _REFINE_CKPT_DIR) -> bool:
    """Seed `target_dir` with the bootstrap weights as net_iter0 + latest pointer,
    IFF that dir has no later checkpoint already. Returns True if it seeded.
    Idempotent + non-clobbering: an in-progress refinement run's progress wins (we
    never overwrite a higher iter)."""
    import glob
    import torch
    os.makedirs(target_dir, exist_ok=True)
    existing = glob.glob(os.path.join(target_dir, "net_iter*.pt"))
    if existing:
        print(f"(ckpt dir {os.path.basename(target_dir)}/ already has progress -- resuming it, not re-seeding)")
        return False
    ck = torch.load(_BOOTSTRAP_CKPT, map_location="cpu", weights_only=False)
    ck = dict(ck)
    ck["iter"] = 0  # so train_robust resumes at iter 1 from the bootstrap weights
    dst = os.path.join(target_dir, "net_iter0.pt")
    tmp = dst + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, dst)
    ptr_tmp = os.path.join(target_dir, "latest.json.tmp.pt")
    torch.save({"iter": 0, "path": "net_iter0.pt"}, ptr_tmp)
    os.replace(ptr_tmp, os.path.join(target_dir, "latest.pt"))
    return True


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
class RandomPlayer:
    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def move(self, board: chess.Board) -> chess.Move:
        return self.rng.choice(list(board.legal_moves))


class EnginePlayer:
    def __init__(self, depth: int = 4, time_limit=None):
        self.engine = Engine(depth=depth, time_limit=time_limit)
        self.depth = depth
        self.name = f"engine(d={depth})"
        self.total_nodes = 0
        self.total_time = 0.0
        self.moves_made = 0

    def move(self, board: chess.Board) -> chess.Move:
        res = self.engine.search(board)
        self.total_nodes += res.nodes
        self.total_time += res.time_s
        self.moves_made += 1
        return res.move


class NetPlayer:
    """The LEARNED AlphaZero net as a player. Two modes:
        net-only : argmax over the legal-masked policy head (NO search).
        net+MCTS : greedy MCTS (--mcts-sims simulations) guided by the net.
    Loads a checkpoint written by bootstrap_supervised.py / train_robust.py
    (schema-tolerant: reads channels/n_blocks from the payload; strict=False)."""

    def __init__(self, ckpt_path: str, use_mcts: bool = False, mcts_sims: int = 64,
                 device=None, temperature: float = 0.0):
        import torch
        # az modules use package-relative imports; importable as `az.*` because az/
        # has an __init__.py and the chess_zero dir is on sys.path when play.py runs.
        from az.net import AlphaZeroNet
        from az.mcts import MCTS
        self._torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        channels = ck.get("channels", 80)
        n_blocks = ck.get("n_blocks", 8)
        self.net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(self.device)
        self.net.load_state_dict(ck["state_dict"], strict=False)
        self.net.eval()
        self.use_mcts = use_mcts
        self.temperature = float(temperature)
        import numpy as _np
        self._rng = _np.random.default_rng()
        self.mcts = MCTS(self.net, n_simulations=mcts_sims, device=self.device) if use_mcts else None
        mode = f"net+MCTS({mcts_sims})" if use_mcts else "net-only(argmax)"
        if self.temperature > 0:
            mode += f" T={self.temperature:g}"
        self.name = f"net[{mode}] C={channels}/B={n_blocks} src={ck.get('source','?')}"
        print(f"[net] loaded {os.path.basename(ckpt_path)}: {self.name}")

    def move(self, board: chess.Board) -> chess.Move:
        if self.use_mcts:
            return self.mcts.best_move(board, temperature=self.temperature)
        # net-only: argmax (temperature 0) or temperature-SAMPLED (>0 -> varied games) over the legal-masked policy
        from az.encoding import board_to_planes, legal_policy_mask
        planes = board_to_planes(board)
        mask, idx_to_move = legal_policy_mask(board)
        probs, _ = self.net.predict(planes, legal_mask=mask, device=self.device)
        legal = list(idx_to_move.keys())
        if not legal:
            return next(iter(board.legal_moves))
        if self.temperature > 0:
            p = self._temp_probs(probs, legal)
            return idx_to_move[legal[int(self._rng.choice(len(legal), p=p))]]
        best_idx, best_p = None, -1.0
        for idx in legal:
            if probs[idx] > best_p:
                best_p, best_idx = probs[idx], idx
        return idx_to_move[best_idx]

    def _temp_probs(self, probs, legal):
        """Temperature-scaled, normalized distribution over the legal policy indices (for varied self-play)."""
        import numpy as _np
        p = _np.array([max(float(probs[i]), 1e-12) for i in legal], dtype=_np.float64)
        p = p ** (1.0 / self.temperature)
        s = p.sum()
        return (_np.ones(len(legal)) / len(legal)) if (s <= 0 or not _np.isfinite(s)) else (p / s)


# --------------------------------------------------------------------------- #
# Game loop
# --------------------------------------------------------------------------- #
def play_game(white, black, max_moves: int = 300, record_san: bool = False):
    """
    Play one full game. Returns (result_str, reason_str, san_moves).
    result_str is python-chess's board.result() ('1-0', '0-1', '1/2-1/2').
    """
    board = chess.Board()
    san_moves = []
    players = {chess.WHITE: white, chess.BLACK: black}

    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        player = players[board.turn]
        move = player.move(board)
        if move not in board.legal_moves:
            # Should never happen (python-chess generates legal moves), but be safe.
            raise RuntimeError(f"{player.name} produced an ILLEGAL move {move} "
                               f"in {board.fen()}")
        if record_san:
            san_moves.append(board.san(move))
        board.push(move)

    result = board.result(claim_draw=True)
    reason = _termination_reason(board)
    return result, reason, san_moves


def _termination_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "75-move rule"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.can_claim_fifty_moves():
        return "50-move rule (claimed)"
    if board.can_claim_threefold_repetition():
        return "threefold repetition (claimed)"
    return "max-moves cap"


def format_san_game(san_moves) -> str:
    """Render a SAN move list as '1. e4 e5 2. Nf3 ...'."""
    out = []
    for i, mv in enumerate(san_moves):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.")
        out.append(mv)
    return " ".join(out)


def play_game_live(white, black, max_moves: int = 300, move_delay: float = 0.0,
                   show_board: bool = False, use_unicode: bool = True,
                   on_move=None):
    """Play one full game, rendering MOVE-BY-MOVE LIVE as it happens.

    Prints each move (SAN, with a move-number prefix) the instant it is chosen,
    optionally an ASCII/unicode board after each move, and sleeps `move_delay`
    seconds between moves so a human can watch. Returns (result, reason, san_moves).
    Identical termination logic to play_game (python-chess decides).

    on_move : optional callback `on_move(board, san_moves)` invoked AFTER each move
              is pushed (board reflects the new position; san_moves is the running
              SAN list). Used by learn-watch to drive the LIVE browser visualizer.
              Defaults to None so existing callers are unaffected. The callback is
              best-effort: an exception in it never breaks the game loop."""
    board = chess.Board()
    san_moves = []
    players = {chess.WHITE: white, chess.BLACK: black}

    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        player = players[board.turn]
        move = player.move(board)
        if move not in board.legal_moves:
            raise RuntimeError(f"{player.name} produced an ILLEGAL move {move} "
                               f"in {board.fen()}")
        san = board.san(move)
        san_moves.append(san)
        # Live line: "  12. Nf3   (White)" style.
        mover = "White" if board.turn == chess.WHITE else "Black"
        movenum = board.fullmove_number
        prefix = f"{movenum}." if board.turn == chess.WHITE else f"{movenum}..."
        print(f"  {prefix:<6} {san:<8} ({mover})", flush=True)
        board.push(move)
        if show_board:
            try:
                print(board.unicode(borders=False, invert_color=True)
                      if use_unicode else str(board), flush=True)
            except Exception:
                print(board, flush=True)
        if on_move is not None:
            try:
                on_move(board, san_moves)
            except Exception:
                pass  # the browser viz is best-effort; never break the game
        if move_delay > 0:
            time.sleep(move_delay)

    result = board.result(claim_draw=True)
    reason = _termination_reason(board)
    return result, reason, san_moves


# --------------------------------------------------------------------------- #
# Mode: engine vs random
# --------------------------------------------------------------------------- #
def run_vs_random(games: int, depth: int, seed: int, print_game: bool,
                  time_limit=None):
    tl = f", time<={time_limit}s/move" if time_limit else ""
    print(f"=== engine(depth={depth}) vs RANDOM | {games} games | seed={seed}{tl} ===",
          flush=True)
    rng = random.Random(seed)
    wins = draws = losses = 0  # from the ENGINE's perspective
    engine_white_count = 0
    first_game_san = None
    first_game_meta = None

    t0 = time.perf_counter()
    eng_nodes_total = 0
    eng_time_total = 0.0
    eng_moves_total = 0

    for g in range(games):
        engine_player = EnginePlayer(depth=depth, time_limit=time_limit)
        random_player = RandomPlayer(rng)
        engine_is_white = (g % 2 == 0)  # alternate colours for fairness
        if engine_is_white:
            white, black = engine_player, random_player
            engine_white_count += 1
        else:
            white, black = random_player, engine_player

        record = print_game and g == 0
        result, reason, san = play_game(white, black, record_san=record)

        # Map result to engine perspective.
        if result == "1-0":
            outcome = "win" if engine_is_white else "loss"
        elif result == "0-1":
            outcome = "loss" if engine_is_white else "win"
        else:
            outcome = "draw"

        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1

        eng_nodes_total += engine_player.total_nodes
        eng_time_total += engine_player.total_time
        eng_moves_total += engine_player.moves_made

        if record:
            first_game_san = san
            first_game_meta = (engine_is_white, result, reason)

        print(f"  game {g+1:>2}: engine={'White' if engine_is_white else 'Black':<5} "
              f"result={result:<7} ({reason:<22}) -> engine {outcome.upper()}",
              flush=True)

    elapsed = time.perf_counter() - t0
    print("\n--- TALLY (engine perspective) ---")
    print(f"  W/D/L = {wins}/{draws}/{losses}  out of {games}")
    score_pts = wins + 0.5 * draws
    print(f"  score = {score_pts}/{games}  ({100*score_pts/games:.1f}%)")
    print(f"  engine played White in {engine_white_count}/{games} games")
    if eng_moves_total:
        print(f"  engine: {eng_moves_total} moves, "
              f"{eng_nodes_total:,} nodes, {eng_time_total:.1f}s "
              f"(avg {eng_nodes_total//eng_moves_total:,} nodes/move, "
              f"{eng_time_total/eng_moves_total:.2f}s/move, "
              f"{int(eng_nodes_total/eng_time_total) if eng_time_total else 0:,} nps)")
    print(f"  wall time: {elapsed:.1f}s")

    if first_game_san is not None:
        ew, res, rsn = first_game_meta
        print(f"\n--- FULL GAME 1 (engine = {'White' if ew else 'Black'}, "
              f"result {res}, {rsn}) ---")
        print(format_san_game(first_game_san))

    return wins, draws, losses


# --------------------------------------------------------------------------- #
# Mode: engine vs engine
# --------------------------------------------------------------------------- #
def run_vs_self(depth: int, print_game: bool):
    print(f"=== engine(depth={depth}) vs engine(depth={depth}) self-play ===")
    white = EnginePlayer(depth=depth)
    black = EnginePlayer(depth=depth)
    result, reason, san = play_game(white, black, record_san=True)
    print(f"result = {result}  ({reason})  in {len(san)} plies")
    nodes = white.total_nodes + black.total_nodes
    tt = white.total_time + black.total_time
    mm = white.moves_made + black.moves_made
    if mm:
        print(f"engine: {mm} moves, {nodes:,} nodes, {tt:.1f}s "
              f"(avg {nodes//mm:,} nodes/move)")
    if print_game:
        print("\n--- FULL GAME (SAN) ---")
        print(format_san_game(san))
    return result, reason


# --------------------------------------------------------------------------- #
# Mode: net-watch -- WATCH the learned net play (vs classical engine or random)
# --------------------------------------------------------------------------- #
def run_net_watch(ckpt: str, use_mcts: bool, mcts_sims: int, opponent: str,
                  depth: int, seed: int, live: bool = False,
                  move_delay: float = 0.3, show_board: bool = False) -> None:
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        print("Run  python -m az.bootstrap_supervised  to create it.")
        return
    net_player = NetPlayer(ckpt, use_mcts=use_mcts, mcts_sims=mcts_sims)
    if opponent == "random":
        opp = RandomPlayer(random.Random(seed))
    else:
        opp = EnginePlayer(depth=depth)
    print(f"=== WATCH: {net_player.name}  vs  {opp.name} (net plays White) ===\n")
    if live:
        result, reason, san = play_game_live(net_player, opp, move_delay=move_delay,
                                             show_board=show_board)
        print(f"\nresult = {result}  ({reason})  in {len(san)} plies  [net was White]")
    else:
        result, reason, san = play_game(net_player, opp, record_san=True)
        print(f"result = {result}  ({reason})  in {len(san)} plies  [net was White]")
        print("\n--- FULL GAME (SAN) ---")
        print(format_san_game(san))
    return result, reason


# --------------------------------------------------------------------------- #
# Mode: learn-watch -- the INTEGRATED experience.
#   kick off self-play+training in the BACKGROUND -> WATCH live net-vs-net games
#   from the LATEST (successively-stronger) checkpoint -> SEE the strength curve
#   advance. One single command delivers: launch -> watch -> see it improve.
# --------------------------------------------------------------------------- #
def _load_net_player_safe(ckpt_path: str, use_mcts: bool, mcts_sims: int,
                          quiet: bool = True, temperature: float = 0.0):
    """Load a NetPlayer from a checkpoint that a BACKGROUND trainer may be
    mid-writing. The trainer writes atomically (tmp + os.replace), so a torch.load
    failure here means we caught a transient state -- we retry a few times, then
    signal failure (None) so the caller can fall back to the previous checkpoint.
    Never raises."""
    import io
    import contextlib
    for attempt in range(4):
        try:
            sink = io.StringIO()
            if quiet:
                with contextlib.redirect_stdout(sink):
                    return NetPlayer(ckpt_path, use_mcts=use_mcts, mcts_sims=mcts_sims, temperature=temperature)
            return NetPlayer(ckpt_path, use_mcts=use_mcts, mcts_sims=mcts_sims, temperature=temperature)
        except Exception as e:  # mid-write / corrupt-read -> retry, then give up
            if attempt < 3:
                time.sleep(0.4)
                continue
            print(f"  [warn] could not load {os.path.basename(ckpt_path)} ({e}); "
                  f"using previous checkpoint")
            return None
    return None


def _read_latest_curve_point(curve_path: str):
    """Return the last row of a train_robust strength_curve.json, or None. Tolerant
    of a mid-write file (the trainer writes it atomically; a transient read just
    returns None and we try again next game)."""
    if not os.path.exists(curve_path):
        return None
    try:
        import json
        with open(curve_path) as f:
            rows = json.load(f)
        if isinstance(rows, list) and rows:
            return rows[-1]
    except Exception:
        return None
    return None


def _read_all_curve_points(curve_path: str):
    """Return the FULL list of rows from a train_robust strength_curve.json, or [].

    Used by the live browser visualizer to draw the strength-curve sparkline over
    ALL iters (the terminal readout uses only the last point). Tolerant of a
    mid-write file (the trainer writes atomically; a transient read returns [])."""
    if not os.path.exists(curve_path):
        return []
    try:
        import json
        with open(curve_path) as f:
            rows = json.load(f)
        if isinstance(rows, list):
            return rows
    except Exception:
        return []
    return []


def _find_latest_ckpt_in(ckpt_dir: str):
    """Newest valid iter checkpoint path + its iter number in `ckpt_dir`, via the
    latest.pt pointer (fall back to highest net_iterN.pt). Returns (path, iter) or
    (None, -1). Mirrors train_robust.find_latest_checkpoint but standalone so play.py
    needs no az import for this."""
    import glob
    latest_ptr = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ptr):
        try:
            import torch
            ptr = torch.load(latest_ptr, map_location="cpu", weights_only=False)
            cand = os.path.join(ckpt_dir, ptr["path"])
            if os.path.exists(cand):
                return cand, int(ptr.get("iter", -1))
        except Exception:
            pass
    cands = glob.glob(os.path.join(ckpt_dir, "net_iter*.pt"))
    if not cands:
        return None, -1

    def _iter_of(p):
        try:
            return int(os.path.basename(p).split("net_iter")[1].split(".pt")[0])
        except Exception:
            return -1
    best = max(cands, key=_iter_of)
    return best, _iter_of(best)


def run_learn_watch(ckpt_dir_name: str, use_mcts: bool, mcts_sims: int,
                    move_delay: float, show_board: bool, max_games: int,
                    max_hours: float, iters: int, games_per_iter: int,
                    eval_games: int, selfplay_sims: int, no_train: bool,
                    train_steps: int = 200, game_pause: float = 1.0,
                    temperature: float = 0.0, watch: str = "classical",
                    train_opponent: str = "mix", engine_depth: int = 2,
                    engine_path: str = "", watch_max_plies: int = 200,
                    viz: bool = True, uci_movetime_ms: int = 50,
                    parallel_games: int = 1, curriculum: bool = False,
                    selfplay_workers: int = 1, anchor_kl: float = 0.0,
                    auto_balance: bool = False, opening_mode: str = "mixed",
                    opening_plies: int = 4) -> None:
    """The integrated 'kick off -> WATCH the agent play -> see it improve' experience.

    1. Seed a (fresh-or-resumed) ckpt-dir under az/ from the bootstrap net.
    2. Launch train_robust as a BACKGROUND subprocess doing DUAL learning -- self-play
       AND vs the classical engine (--selfplay-opponent <train_opponent>, default mix)
       -- with its strength curve written INSIDE the ckpt-dir (so a throwaway demo
       never pollutes the shared az/strength_curve.json).
    3. FOREGROUND loop: load the LATEST checkpoint (robust to mid-write), play ONE
       game on a LIVE VISUAL BOARD where the net plays the chosen opponent
       (--watch classical|self|random; default classical so you SEE the agent play
       the strong engine), then print the latest strength-curve point. The net side
       ALTERNATES colour across games so the agent plays both White and Black. As
       training writes newer checkpoints, successive games come from a stronger net.
       Ctrl-C stops watching and tears the trainer down.

    Note: net-vs-classical games are slower (the engine searches) and the agent will
    often LOSE to the engine -- that is honest and the point. A per-game ply cap
    (watch_max_plies) guarantees a watched game can never hang forever.
    """
    import torch  # noqa: F401  (import early so a missing torch fails fast + clearly)

    ckpt_dir = os.path.join(_HERE, "az", ckpt_dir_name)
    # curve lives INSIDE the ckpt-dir; path passed to train_robust is relative to az/.
    curve_rel = os.path.join(ckpt_dir_name, "strength_curve.json")
    curve_abs = os.path.join(_HERE, "az", curve_rel)

    if not os.path.exists(_BOOTSTRAP_CKPT):
        print(f"ERROR: bootstrap checkpoint not found: {_BOOTSTRAP_CKPT}")
        print("Run  python -m az.bootstrap_supervised  first.")
        return

    # the watched matchup label: net plays classical engine / itself / random.
    _watch_opp_label = {"classical": f"the CLASSICAL engine(d={engine_depth})",
                        "self": "ITSELF (net-vs-net)",
                        "random": "a RANDOM mover"}.get(watch, watch)
    print("=" * 68)
    print("  LEARN-WATCH -- DUAL learning, WATCH the agent play & improve")
    print("=" * 68)
    print(f"  ckpt-dir : az/{ckpt_dir_name}/   (seeded from the bootstrap net)")
    print(f"  curve    : az/{curve_rel}")
    mode = (f"net+MCTS({mcts_sims})" if use_mcts
            else (f"net-only(sampled T={temperature:g})" if temperature > 0 else "net-only(argmax)"))
    print(f"  learn    : self-play AND vs-engine  (--selfplay-opponent {train_opponent}, "
          f"teacher-depth {engine_depth})")
    print(f"  watch    : the net vs {_watch_opp_label}, {mode}, "
          f"board={'on' if show_board else 'off'}, move_delay={move_delay}s")
    print()

    # --- (1) seed the ckpt-dir from the bootstrap (idempotent / non-clobbering) ---
    _seed_refine_from_bootstrap(ckpt_dir)

    # --- (2) launch train_robust in the BACKGROUND ---
    proc = None
    logf = None  # L3 FIX: only the train branch opens a log file; init to None so the
                 # finally-block close is a no-op (not a NameError) on --no-train.
    if no_train:
        print("[train] --no-train: NOT launching the trainer; watching the seeded "
              "net only (games will NOT improve).\n")
    else:
        cmd = [sys.executable, "-u", "-m", "az.train_robust",
               "--ckpt-dir", ckpt_dir_name,
               "--curve-path", curve_rel,
               "--max-hours", str(max_hours),
               "--iters", str(iters),
               "--games-per-iter", str(games_per_iter),
               "--parallel-games", str(parallel_games),
               "--selfplay-workers", str(selfplay_workers),
               "--anchor-kl", str(anchor_kl),
               "--eval-games", str(eval_games),
               "--selfplay-sims", str(selfplay_sims),
               "--train-steps", str(train_steps),
               # FIX 2 (2026-06-07): run the BACKGROUND trainer under its own in-process
               # auto-restart SUPERVISOR. The learn-watch session runs for HOURS; without
               # --supervise a single worker-iteration crash (transient CUDA fault, etc.)
               # killed the trainer permanently and the watch silently stopped improving.
               # --supervise relaunches the worker from the latest checkpoint (resume) so
               # the unattended watch survives crashes end-to-end.
               "--supervise",
               # DUAL learning: the agent learns from self-play AND from games vs the
               # classical engine (--selfplay-opponent mix alternates both).
               "--selfplay-opponent", train_opponent,
               "--selfplay-teacher-depth", str(engine_depth),
               # OPENING DIVERSITY: vary the starting position each self-play game so the
               # net stops reinforcing one rote line (the "vary the starting conditions" fix).
               "--opening-mode", opening_mode,
               "--opening-plies", str(opening_plies)]
        # optional UCI/Stockfish engine path (the other worker adds --engine-path to
        # train_robust). Pass it through ONLY when non-empty, so the common empty
        # default never trips an "unrecognized argument" if that flag has not landed.
        if engine_path:
            cmd += ["--engine-path", engine_path]
            # FIX 3 (2026-06-07): pass the UCI per-move time budget through to the trainer
            # ONLY when a UCI engine is actually in use (paired with --engine-path), so the
            # common classical-engine path never references an unrecognized flag.
            cmd += ["--uci-movetime", str(uci_movetime_ms)]
        if curriculum:
            # EVOLVING (the moving teacher): bump the self-play teacher depth as the net masters
            # the current depth, so the bar keeps rising instead of saturating against a fixed
            # opponent. Persisted across the --supervise restarts (load_checkpoint restores it).
            cmd += ["--curriculum"]
        if auto_balance:
            # SELF-TUNING throughput: the trainer derives workers + nudges games/steps toward a
            # target iter-time from the --max-hours budget (the explicit --games-per-iter /
            # --train-steps become the STARTING point; the learning contract stays fixed).
            cmd += ["--auto-balance"]
        train_log = os.path.join(ckpt_dir, "train.log")
        print(f"[train] launching background trainer -> log: az/{ckpt_dir_name}/train.log")
        print(f"        {' '.join(cmd)}\n")
        logf = open(train_log, "w", encoding="utf-8")
        # IMPORTANT (Windows): put the trainer in its OWN process group so a Ctrl-C
        # (CTRL_BREAK_EVENT) delivered to THIS process's group does NOT propagate
        # to the trainer and fracture its process tree before we can tree-kill it
        # cleanly in the finally block. Without this, Ctrl-C orphans the trainer's
        # `python -m` grandchild.
        popen_kwargs = dict(cwd=_REPO_ROOT, stdout=logf, stderr=subprocess.STDOUT)
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs["start_new_session"] = True  # own POSIX session/pgid
        proc = subprocess.Popen(cmd, **popen_kwargs)

    # Make a headless CTRL_BREAK (SIGBREAK) behave like an interactive Ctrl-C
    # (SIGINT): both should raise KeyboardInterrupt so the finally-block teardown
    # runs. Without this, on Windows the MKL/Fortran runtime intercepts CTRL_BREAK
    # and HARD-ABORTS the process before Python's handler runs -> orphaned trainer.
    def _to_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt()
    _prev_break = None
    if hasattr(signal, "SIGBREAK"):
        try:
            _prev_break = signal.signal(signal.SIGBREAK, _to_keyboard_interrupt)
        except Exception:
            _prev_break = None

    # --- (3) FOREGROUND watch loop ---
    # Build the WATCHED opponent the net plays against (the agent-vs-engine view):
    #   classical -> the classical search Engine (you SEE the agent play a strong foe)
    #   random    -> a random mover (sanity / fast)
    #   self      -> the net plays itself (classic AZ self-play view; no extra opponent)
    # The classical Engine is constructed ONCE and reused; the net is reloaded as
    # newer checkpoints land. RandomPlayer is reseeded per game so games vary.
    watch = watch if watch in ("classical", "self", "random") else "classical"
    watch_opp = None
    if watch == "classical":
        watch_opp = EnginePlayer(depth=engine_depth)

    def _watch_matchup_label(net_is_white: bool) -> str:
        """Human-readable matchup string for the game banner, oriented by colour."""
        if watch == "self":
            return "net-vs-net self-play"
        opp = (f"classical(d={engine_depth})" if watch == "classical" else "random")
        return (f"net(White) vs {opp}(Black)" if net_is_white
                else f"{opp}(White) vs net(Black)")

    # --- TRUE browser visualizer (real piece graphics, live-updating, no server) ---
    # The board now lives in the BROWSER (real SVG pieces), not as ASCII in the
    # terminal. We still print the SAN line per move (cheap). Best-effort: if the
    # viz import/start fails for any reason, we degrade to terminal-only.
    live_viz = None
    if viz:
        try:
            from az.live_viz import LiveViz
            live_viz = LiveViz(ckpt_dir)
            live_viz.start(title="learn-watch starting -- waiting for the first game...")
            print(f"  [viz] LIVE browser board -> file://{live_viz.html_path}")
            print( "        (auto-opens once; auto-refreshes every 1s as the agent plays)\n")
        except Exception as e:
            print(f"  [viz] could not start the browser visualizer ({e}); "
                  f"terminal-only.\n")
            live_viz = None

    last_ckpt_path = None
    net_player = None
    games_played = 0
    try:
        while True:
            # has the trainer exited? (only if we started one)
            trainer_done = (proc is not None and proc.poll() is not None)

            # pick the newest checkpoint; fall back to bootstrap if none yet.
            cand_path, cand_iter = _find_latest_ckpt_in(ckpt_dir)
            if cand_path is None:
                cand_path, cand_iter = _BOOTSTRAP_CKPT, 0

            # (re)load the net ONLY when the checkpoint path changed (cheap loop).
            if cand_path != last_ckpt_path or net_player is None:
                loaded = _load_net_player_safe(cand_path, use_mcts, mcts_sims, temperature=temperature)
                if loaded is not None:
                    net_player = loaded
                    last_ckpt_path = cand_path
                elif net_player is None:
                    # very first load hit a mid-write; wait + retry the loop.
                    time.sleep(0.6)
                    continue

            games_played += 1
            cp = _read_latest_curve_point(curve_abs)
            cur_iter = cp["iter"] if cp else cand_iter
            src = os.path.basename(last_ckpt_path)

            # ALTERNATE the net's colour across games so the agent plays BOTH sides
            # over successive games (game 1 net=White, game 2 net=Black, ...).
            net_is_white = (games_played % 2 == 1)
            if watch == "self":
                white_player, black_player = net_player, net_player
            else:
                # a freshly-seeded random mover per game when watch == random.
                opp = (watch_opp if watch == "classical"
                       else RandomPlayer(random.Random(1000 + games_played)))
                white_player = net_player if net_is_white else opp
                black_player = opp if net_is_white else net_player

            print("-" * 68)
            print(f"  GAME {games_played}  |  net from iter {cur_iter}  ({src})  "
                  f"|  {_watch_matchup_label(net_is_white)}")
            print("-" * 68)

            # Build the live-viz header + per-move callback for THIS game. The
            # callback rewrites the browser page after each move (best-effort).
            _viz_header = (f"GAME {games_played} | net from iter {cur_iter} ({src}) | "
                           f"{_watch_matchup_label(net_is_white)}")
            _on_move = None
            if live_viz is not None:
                def _on_move(board, sans, _hdr=_viz_header):
                    live_viz.update(board, sans, _hdr,
                                    _read_all_curve_points(curve_abs))

            # max_moves is a FULLMOVE cap; watch_max_plies is in plies (half-moves).
            # Convert so a net-vs-classical game can never hang forever.
            _max_moves = max(1, watch_max_plies // 2)
            result, reason, san = play_game_live(
                white_player, black_player, move_delay=move_delay,
                show_board=show_board, max_moves=_max_moves, on_move=_on_move)
            # Report the outcome from the NET's perspective so a loss is honest+clear.
            if watch == "self":
                outcome = "self-play"
            elif result == "1/2-1/2":
                outcome = "net DREW"
            elif result in ("1-0", "0-1"):
                net_won = (result == "1-0") == net_is_white
                outcome = "net WON" if net_won else "net LOST"
            else:  # '*' -- truncated by the ply cap, no decisive result yet
                outcome = "unfinished (ply cap)"
            net_color = "White" if net_is_white else "Black"
            print(f"\n  result = {result}  ({reason})  in {len(san)} plies  "
                  f"[net was {net_color} -- {outcome}]")

            # strength-curve readout (the PROOF the net is improving across games)
            cp = _read_latest_curve_point(curve_abs)
            if cp is None:
                print("  [strength] iter 0 (no eval yet -- trainer still on its "
                      "first iteration; the curve fills in as iters complete)")
            else:
                print(f"  [strength] iter {cp['iter']}: "
                      f"winrate_vs_random={cp.get('winrate_vs_random', float('nan')):.3f}  "
                      f"winrate_vs_classical={cp.get('winrate_vs_classical_d1', float('nan')):.3f}  "
                      f"loss={cp.get('total_loss', float('nan')):.3f}")
            print()

            if trainer_done:
                print("[train] background trainer has EXITED -- playing one final game "
                      "from the last checkpoint, then stopping.")
                break
            if max_games > 0 and games_played >= max_games:
                # FIX (2026-06-08): only claim a background trainer was left running when one
                # was actually launched. Under --no-train, proc is None -- the old message
                # ("trainer left running in background...") was a lie.
                tail = (" (trainer left running in background if still alive)"
                        if proc is not None else "")
                print(f"[watch] reached --max-games {max_games} -- stopping the watch "
                      f"loop.{tail}")
                break
            if game_pause > 0:
                # PACING: a small pause between watched games throttles the loop so it
                # does not burn CPU re-playing from the same checkpoint faster than the
                # background trainer can write a newer (stronger) one -- and so a human
                # watching has time to read each game. (Independent of move policy; it is
                # NOT a determinism assumption -- temperature>0 / --mcts make games vary.)
                time.sleep(game_pause)
    except KeyboardInterrupt:
        print("\n\n[watch] Ctrl-C -- stopping the watch loop.")
    finally:
        _teardown_trainer(proc)
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass
        if _prev_break is not None and hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _prev_break)
            except Exception:
                pass
    print("\n[learn-watch] done.")


def _win_descendant_pids(root_pid: int):
    """Return [root_pid] + ALL descendant PIDs (children, grandchildren, ...) by
    walking Win32_Process ParentProcessId. We do this OURSELVES rather than rely on
    `taskkill /T`, whose tree-walk is not atomic: killing the `python -m` launcher
    first re-parents its grandchild before /T enumerates it, orphaning the trainer."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | "
             "Select-Object ProcessId,ParentProcessId | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except Exception:
        return [root_pid]
    children = {}
    for line in out.splitlines()[1:]:  # skip header
        parts = line.replace('"', '').split(',')
        if len(parts) < 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    # BFS from root over the parent->children map.
    seen, stack, order = set(), [root_pid], []
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        order.append(p)
        stack.extend(children.get(p, []))
    return order


def _teardown_trainer(proc) -> None:
    """Cleanly stop the background trainer if we started one and it is still alive.

    IMPORTANT (Windows): `python -m projects...` re-execs into a GRANDCHILD that
    does the real work, and `taskkill /T` can orphan that grandchild (non-atomic
    tree walk). So we ENUMERATE the full descendant PID set ourselves and kill each
    one explicitly -- no orphan GPU job survives. POSIX kills the whole session
    process group (trainer is its own session leader)."""
    if proc is None:
        return
    if proc.poll() is not None:
        print(f"[train] background trainer already exited (code {proc.returncode}).")
        return
    print("[train] stopping the background trainer (kill full process tree)...")
    if os.name == "nt":
        pids = _win_descendant_pids(proc.pid)
        # kill leaves-first (reverse BFS order) so a parent can't re-spawn/re-parent
        for pid in reversed(pids):
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            except Exception:
                pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    try:
        proc.wait(timeout=15)
        print(f"[train] background trainer tree stopped (code {proc.returncode}).")
        return
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=10)
        print("[train] background trainer killed.")
    except Exception as e:
        print(f"[train] WARN: could not confirm trainer teardown ({e}); "
              f"check for an orphan PID {getattr(proc, 'pid', '?')}.")


# --------------------------------------------------------------------------- #
# Mode: net-play -- YOU play vs the learned net (interactive)
# --------------------------------------------------------------------------- #
def run_net_play(ckpt: str, use_mcts: bool, mcts_sims: int, net_white: bool,
                 use_unicode: bool) -> None:
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        print("Run  python -m az.bootstrap_supervised  to create it.")
        return
    from chess_engine.play_human import render_board, parse_human_move, announce_result, HELP_TEXT
    net_player = NetPlayer(ckpt, use_mcts=use_mcts, mcts_sims=mcts_sims)
    board = chess.Board()
    human_color = chess.BLACK if net_white else chess.WHITE
    print("=== chess_zero -- human vs LEARNED NET ===")
    print(f"you are {'White' if human_color == chess.WHITE else 'Black'}; "
          f"opponent = {net_player.name}")
    print(HELP_TEXT)
    print()
    print(render_board(board, use_unicode))
    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            sys.stdout.write(f"\nyour move ({'White' if board.turn else 'Black'}): ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                print("\n(eof -- ending session)")
                return
            text = line.strip()
            if text == "":
                continue
            low = text.lower()
            if low in ("quit", "exit", "resign"):
                print(f"\nyou {low}.")
                return
            if low == "help":
                print(HELP_TEXT); continue
            if low == "board":
                print(render_board(board, use_unicode)); continue
            if low == "fen":
                print(board.fen()); continue
            if low == "moves":
                print(" ".join(m.uci() for m in board.legal_moves)); continue
            mv = parse_human_move(board, text)
            if mv is None:
                print(f"  illegal/unparsable move: '{text}'. Try UCI (e2e4) or SAN (Nf3).")
                continue
            board.push(mv)
            print(render_board(board, use_unicode))
        else:
            mv = net_player.move(board)
            print(f"\nnet plays: {board.san(mv)}  ({mv.uci()})")
            board.push(mv)
            print(render_board(board, use_unicode))
    announce_result(board)


# --------------------------------------------------------------------------- #
# Mode: play a position (simple CLI)
# --------------------------------------------------------------------------- #
def run_position(fen: str, depth: int):
    board = chess.Board(fen)
    print(f"position: {board.fen()}")
    print(board)
    mv, info = best_move(board, depth=depth)
    san = board.san(mv)
    mate = ""
    if abs(info["score_cp"]) >= MATE_THRESHOLD:
        mate = "  (forced mate)"
    print(f"\nbest move: {san}  ({board.uci(mv)})  "
          f"eval={info['score_cp']}cp{mate}")
    print(f"depth={info['depth']}  nodes={info['nodes']:,}  "
          f"time={info['time_s']:.2f}s  nps={info['nps']:,}")
    if info["pv_san"]:
        print(f"pv: {' '.join(info['pv_san'])}")
    return mv, info


# --------------------------------------------------------------------------- #
# Mode: bench (nodes/time per move at a few positions)
# --------------------------------------------------------------------------- #
BENCH_FENS = [
    ("startpos", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame", "8/2k5/3p4/p2P1p2/P2P1P2/8/8/4K3 w - - 0 1"),
]


def run_bench(depth: int):
    print(f"=== bench (depth {depth}) ===")
    for name, fen in BENCH_FENS:
        board = chess.Board(fen)
        mv, info = best_move(board, depth=depth)
        print(f"  {name:<10} best={board.san(mv):<7} eval={info['score_cp']:>6}cp  "
              f"nodes={info['nodes']:>9,}  time={info['time_s']:.2f}s  "
              f"nps={info['nps']:>8,}  depth={info['depth']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="chess_zero -- play manually, watch AI-vs-AI, or self-play WHILE LEARNING")
    sub = ap.add_subparsers(dest="mode", required=True)

    # --- the 3 user-facing modes ---
    p_h = sub.add_parser("human", help="PLAY MANUALLY vs the engine (interactive)")
    p_h.add_argument("--engine-white", action="store_true", help="engine moves first (you play Black)")
    p_h.add_argument("--depth", type=int, default=4)
    p_h.add_argument("--movetime", type=float, default=None, help="engine seconds/move")
    p_h.add_argument("--ascii", action="store_true", help="ASCII board (default: unicode)")

    p_w = sub.add_parser("watch", help="WATCH AI vs AI (classical engine plays itself)")
    p_w.add_argument("--depth", type=int, default=4)

    p_l = sub.add_parser("learn", help="AI SELF-PLAY WHILE LEARNING (AlphaZero: self-play -> train -> eval)")
    p_l.add_argument("--no-bootstrap", dest="bootstrap", action="store_false", default=True,
                     help="cold-start self-play from random weights (default: resume the supervised-bootstrap base)")

    # Default to the COMMITTED champion (the strong net you actually want to play); fall back to
    # the (gitignored, maybe-absent) supervised-bootstrap only if the champion isn't present.
    _champ_ckpt = os.path.join(_HERE, "az", "robust_from_bootstrap", "champion.pt")
    _boot_ckpt = os.path.join(_HERE, "az", "bootstrap_checkpoints", "net_bootstrap.pt")
    _dflt_ckpt = _champ_ckpt if os.path.exists(_champ_ckpt) else _boot_ckpt
    p_np = sub.add_parser("net-play", help="YOU play vs the LEARNED net")
    p_np.add_argument("--ckpt", type=str, default=_dflt_ckpt)
    p_np.add_argument("--mcts", action="store_true", help="net+MCTS (default: net-only argmax policy)")
    p_np.add_argument("--mcts-sims", type=int, default=64)
    p_np.add_argument("--net-white", action="store_true", help="net plays White (you play Black)")
    p_np.add_argument("--ascii", action="store_true")

    p_nw = sub.add_parser("net-watch", help="WATCH the learned net play (vs classical or random)")
    p_nw.add_argument("--ckpt", type=str, default=_dflt_ckpt)
    p_nw.add_argument("--mcts", action="store_true", help="net+MCTS (default: net-only argmax policy)")
    p_nw.add_argument("--mcts-sims", type=int, default=64)
    p_nw.add_argument("--opponent", choices=["classical", "random"], default="classical")
    p_nw.add_argument("--depth", type=int, default=4, help="classical opponent depth")
    p_nw.add_argument("--seed", type=int, default=0)
    p_nw.add_argument("--live", action="store_true",
                      help="render the game LIVE move-by-move (default: dump SAN at the end)")
    p_nw.add_argument("--move-delay", type=float, default=0.3,
                      help="seconds between moves in --live mode (default 0.3)")
    p_nw.add_argument("--board", action="store_true",
                      help="show the board after each move in --live mode")

    # --- the INTEGRATED experience: kick off DUAL learning -> WATCH the agent play -> see it improve ---
    p_lw = sub.add_parser("learn-watch",
                          help="KICK OFF DUAL learning (self-play AND vs the engine) in the "
                               "background AND WATCH the agent play live on a VISUAL BOARD "
                               "from successively-stronger checkpoints")
    p_lw.add_argument("--ckpt-dir", type=str, default=_REFINE_DIR_NAME,
                      help="ckpt subdir under az/ (default robust_from_bootstrap; use a "
                           "FRESH name like _demo_test for a throwaway run)")
    p_lw.add_argument("--watch", choices=["classical", "self", "random"], default="classical",
                      help="WHO the watched net plays: classical=the agent vs the classical "
                           "engine (DEFAULT -- watch it play a strong opponent); self=net-vs-net "
                           "self-play; random=net vs a random mover (fast sanity)")
    p_lw.add_argument("--train-opponent", choices=["self", "mix", "teacher"], default="mix",
                      help="DUAL-learning opponent for the BACKGROUND trainer (passed as "
                           "--selfplay-opponent): mix=self-play AND vs-engine (DEFAULT); "
                           "self=pure self-play; teacher=pure vs-engine")
    p_lw.add_argument("--engine-depth", type=int, default=2,
                      help="classical engine search depth -- used for BOTH the watched "
                           "classical opponent AND the trainer's teacher games "
                           "(--selfplay-teacher-depth; default 2)")
    p_lw.add_argument("--engine-path", type=str, default="",
                      help="optional UCI/Stockfish engine path passed through to the trainer "
                           "(--engine-path); empty = use the built-in classical engine. "
                           "ASYMMETRY: this routes the UCI engine to the TRAINING TEACHER "
                           "ONLY (the opponent in self-play games). The WATCHED opponent and "
                           "the trainer's strength EVAL both stay on the in-repo classical "
                           "engine regardless -- so the curve's 'vs classical' axis is a "
                           "stable yardstick even when training against Stockfish.")
    p_lw.add_argument("--uci-movetime", dest="uci_movetime_ms", type=int, default=50,
                      help="per-move time budget (ms) for a UCI teacher engine -- passed "
                           "through to the trainer's --uci-movetime ONLY when --engine-path "
                           "is set (default 50)")
    p_lw.add_argument("--mcts", action="store_true",
                      help="net+MCTS for the net side (default: net-only argmax -- cheap)")
    p_lw.add_argument("--mcts-sims", type=int, default=48)
    p_lw.add_argument("--temperature", type=float, default=0.0,
                      help="net move SAMPLING temperature: 0=argmax (deterministic); "
                           ">0 (e.g. 1.0) = sampled -> each game VARIES")
    p_lw.add_argument("--move-delay", type=float, default=0.3,
                      help="seconds between moves so you can watch (default 0.3)")
    # --viz (DEFAULT ON): the TRUE browser visualizer -- real piece graphics on a
    # live-updating, self-contained HTML page (az/<ckpt-dir>/live.html). --no-viz
    # falls back to terminal-only. When --viz is on, the ASCII board defaults OFF
    # (the board lives in the browser; we avoid double-rendering).
    p_lw.add_argument("--viz", dest="viz", action="store_true", default=True,
                      help="TRUE browser visualizer: real piece graphics on a live "
                           "self-contained HTML board (DEFAULT ON)")
    p_lw.add_argument("--no-viz", dest="viz", action="store_false",
                      help="disable the browser visualizer; terminal-only")
    # --board defaults to None so we can tell 'user said nothing' from an explicit
    # choice: with --viz ON and no explicit --board/--no-board, the ASCII board is
    # OFF (no double-render); --board forces it on, --no-board forces it off.
    p_lw.add_argument("--board", dest="board", action="store_true", default=None,
                      help="ALSO show the ASCII board in the terminal (default: OFF "
                           "when --viz is on, since the board is in the browser)")
    p_lw.add_argument("--no-board", dest="board", action="store_false",
                      help="DISABLE the terminal ASCII board")
    p_lw.add_argument("--watch-max-plies", type=int, default=200,
                      help="hard ply cap per watched game so a net-vs-engine game can never "
                           "hang forever (default 200)")
    p_lw.add_argument("--max-games", type=int, default=0,
                      help="stop after N watched games (0 = until trainer exits / Ctrl-C)")
    p_lw.add_argument("--no-train", action="store_true",
                      help="do NOT launch the trainer; just watch the seeded net (no improvement)")
    # passthrough to the background train_robust
    p_lw.add_argument("--max-hours", type=float, default=6.0, help="trainer wall-clock envelope")
    p_lw.add_argument("--iters", type=int, default=1000, help="trainer max iterations")
    p_lw.add_argument("--games-per-iter", type=int, default=8, help="trainer self-play games/iter")
    p_lw.add_argument("--parallel-games", type=int, default=1,
                      help="THROUGHPUT: generate self-play games in GPU-batched groups of this "
                           "size (more games/sec via ~Nx fewer GPU round-trips). This is the "
                           "'control the rate of parallel instances' knob. Takes effect with "
                           "--train-opponent self (the batchable case); ignored for mix/teacher. "
                           "1 = sequential (default); raise it + watch VRAM.")
    p_lw.add_argument("--curriculum", action="store_true",
                      help="EVOLVING (moving teacher): bump the self-play teacher depth as the net "
                           "masters the current depth (latched once per crossing, persisted across "
                           "--supervise restarts), so the agent keeps facing a harder opponent "
                           "instead of saturating. Off by default.")
    p_lw.add_argument("--selfplay-workers", type=int, default=1,
                      help="SPEED (the big lever): generate self-play games across N CPU worker "
                           "PROCESSES in parallel (self-play is CPU-bound; ~13.5x at N=16 on 20 "
                           "cores -> the curve climbs that much faster). Use with --train-opponent "
                           "self. 1 = single-process (default); set ~cores-2 to saturate the CPU.")
    p_lw.add_argument("--anchor-kl", type=float, default=0.0,
                      help="QUALITY/anti-drift: KL(bootstrap||candidate) penalty weight. Pure self-play "
                           "DRIFTS off the strong imitation base and DEGRADES; a positive anchor (e.g. 1.0) "
                           "keeps the net near the bootstrap so it improves FROM strength instead of "
                           "forgetting it. 0 = off (default).")
    p_lw.add_argument("--auto-balance", action="store_true", default=False,
                      help="SELF-TUNING throughput: let the trainer derive selfplay-workers + nudge "
                           "games/steps toward a target iter-time from the --max-hours budget (balanced "
                           "per unit time, no CPU/iter-time bloat) instead of you hand-tuning them. "
                           "--games-per-iter / --train-steps become the STARTING point. The learning "
                           "contract (gate/anchor-kl/curriculum/lr) stays fixed. Off by default.")
    p_lw.add_argument("--opening-mode", choices=["startpos", "book", "random", "mixed"],
                      default="mixed",
                      help="OPENING DIVERSITY for the trainer's SELF-PLAY (the 'vary the starting "
                           "conditions' fix): each game starts from a distinct sound opening so the "
                           "net stops reinforcing one rote line / bad learned habits. startpos=old "
                           "behaviour; book=curated sound openings; random=guarded random plies; "
                           "mixed=book + jitter (DEFAULT). Eval stays on startpos (curve unaffected).")
    p_lw.add_argument("--opening-plies", type=int, default=4,
                      help="random plies for --opening-mode random, or jitter plies on top of the "
                           "book line for mixed (default 4; ignored for startpos/book)")
    p_lw.add_argument("--eval-games", type=int, default=20, help="trainer eval games/iter")
    p_lw.add_argument("--selfplay-sims", type=int, default=64, help="trainer self-play MCTS sims")
    p_lw.add_argument("--train-steps", type=int, default=200, help="trainer gradient steps/iter")
    p_lw.add_argument("--game-pause", type=float, default=1.0,
                      help="seconds to pause between watched games -- PACING only: throttles "
                           "the watch so it does not re-play from the same checkpoint faster "
                           "than the trainer writes a newer one, and gives a human time to "
                           "read each game (default 1.0)")

    p_vr = sub.add_parser("vs-random", help="engine vs random mover")
    p_vr.add_argument("--games", type=int, default=20)
    p_vr.add_argument("--depth", type=int, default=4)
    p_vr.add_argument("--seed", type=int, default=0)
    p_vr.add_argument("--time", type=float, default=None,
                      help="per-move time limit in seconds (depth stays the cap)")
    p_vr.add_argument("--print-game", action="store_true")

    p_vs = sub.add_parser("vs-self", help="engine vs engine self-play")
    p_vs.add_argument("--depth", type=int, default=4)
    p_vs.add_argument("--print-game", action="store_true")

    p_pos = sub.add_parser("position", help="best move for a FEN")
    p_pos.add_argument("--fen", type=str, default=chess.STARTING_FEN)
    p_pos.add_argument("--depth", type=int, default=4)

    p_b = sub.add_parser("bench", help="nodes/time per move sanity")
    p_b.add_argument("--depth", type=int, default=4)

    args = ap.parse_args()
    if args.mode == "human":
        from chess_engine.play_human import play  # same dir; runs in-process so stdin/stdout stay interactive
        play(engine_white=args.engine_white, depth=args.depth, movetime=args.movetime,
             use_unicode=not args.ascii)
    elif args.mode == "watch":
        run_vs_self(args.depth, print_game=True)
    elif args.mode == "learn":
        _run_learn(from_bootstrap=args.bootstrap)
    elif args.mode == "net-play":
        run_net_play(args.ckpt, use_mcts=args.mcts, mcts_sims=args.mcts_sims,
                     net_white=args.net_white, use_unicode=not args.ascii)
    elif args.mode == "net-watch":
        run_net_watch(args.ckpt, use_mcts=args.mcts, mcts_sims=args.mcts_sims,
                      opponent=args.opponent, depth=args.depth, seed=args.seed,
                      live=args.live, move_delay=args.move_delay, show_board=args.board)
    elif args.mode == "learn-watch":
        # Resolve the ASCII-board default: with the browser viz ON, the terminal
        # board is OFF unless the user explicitly asked for it (avoid double-render).
        # With --no-viz, keep the historical default (ASCII board ON).
        show_board = args.board
        if show_board is None:
            show_board = (not args.viz)
        run_learn_watch(args.ckpt_dir, use_mcts=args.mcts, mcts_sims=args.mcts_sims,
                        move_delay=args.move_delay, show_board=show_board,
                        max_games=args.max_games, max_hours=args.max_hours,
                        iters=args.iters, games_per_iter=args.games_per_iter,
                        eval_games=args.eval_games, selfplay_sims=args.selfplay_sims,
                        no_train=args.no_train, train_steps=args.train_steps,
                        game_pause=args.game_pause, temperature=args.temperature,
                        watch=args.watch, train_opponent=args.train_opponent,
                        engine_depth=args.engine_depth, engine_path=args.engine_path,
                        watch_max_plies=args.watch_max_plies, viz=args.viz,
                        uci_movetime_ms=args.uci_movetime_ms,
                        parallel_games=args.parallel_games, curriculum=args.curriculum,
                        selfplay_workers=args.selfplay_workers, anchor_kl=args.anchor_kl,
                        auto_balance=args.auto_balance, opening_mode=args.opening_mode,
                        opening_plies=args.opening_plies)
    elif args.mode == "vs-random":
        run_vs_random(args.games, args.depth, args.seed, args.print_game,
                      time_limit=args.time)
    elif args.mode == "vs-self":
        run_vs_self(args.depth, args.print_game)
    elif args.mode == "position":
        run_position(args.fen, args.depth)
    elif args.mode == "bench":
        run_bench(args.depth)


if __name__ == "__main__":
    main()
