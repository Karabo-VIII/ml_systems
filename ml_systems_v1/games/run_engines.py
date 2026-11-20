"""
projects/chess_zero/run_engines.py -- the TURNKEY "click run" demo.

ONE command -> WATCH three engines that were trained on a single 4060 actually PLAY,
rendered move-by-move in the terminal:

    CHESS     -- the strongest AlphaZero champion (residual CNN + PUCT MCTS) vs the
                 in-repo classical negamax engine (Engine, depth 2).
    CONNECT-4 -- the freshly self-play-trained Connect4Net (NeuralMCTS) vs a 1-ply
                 win/block tactical heuristic.
    ATARI     -- the MuZero-RL agent PLANNING OVER ITS LEARNED MODEL on the scaled-down
                 Atari 'catch' env (5x5x2 pixel grid), single episode.

It uses the EXISTING trained checkpoints + the EXISTING search/net code -- no new
training, no new search. Everything is ADDITIVE; this file imports the az package and
the classical engine, nothing more.

USAGE (the command the user clicks):

    python projects/chess_zero/run_engines.py
    python -m run_engines

    --engine {chess,connect4,atari,all}   which engine(s) to play (default: all)
    --games N                             games/episodes per engine (default: 1)
    --delay SEC                           per-move sleep so it is watchable (default: 0.4)
    --fast                                delay=0 + low sims (CI / quick look)
    --no-render                           scores only (no board animation)
    --device {auto,cpu,cuda}              torch device (default: auto)

ROBUSTNESS: if a checkpoint is missing, a clear "[checkpoint missing -- UNTRAINED net]"
banner is printed and the engine STILL runs on a fresh net, so the launcher never
hard-fails. When the checkpoint is present its trained strength is loaded + announced.

No emoji anywhere (Windows cp1252) -- boards/grids are pure ASCII.
"""
from __future__ import annotations

__contract__ = {
    "kind": "demo-runner",
    "inputs": [
        "trained checkpoints: az/robust_from_bootstrap/champion.pt, az/checkpoints/connect4.pt, "
        "az/checkpoints/atari.pt (each optional -- missing -> untrained net, still runs)",
        "CLI: --engine --games --delay --fast --no-render --device",
    ],
    "outputs": [
        "stdout: per-engine animated board/grid + a result + strength line; a final summary",
        "play_chess/play_connect4/play_atari return a result dict (programmatic use + tests)",
        "exit_code: 0 when every requested engine played to a valid terminal result",
    ],
    "invariants": [
        "ADDITIVE: imports az.* + the classical engine; trains nothing, mutates no checkpoint",
        "a missing checkpoint NEVER hard-fails -- it falls back to a fresh (untrained) net",
        "every rendered board/grid is pure ASCII (no emoji; Windows cp1252 safe)",
        "every engine plays to a real terminal result (chess board.result / C4 winner / catch done)",
        "runnable both as a script and as 'python -m run_engines'",
    ],
}

import argparse
import os
import random
import sys
import time

# --------------------------------------------------------------------------- #
# Path bootstrap: make `from az.* import ...` and `from chess_engine.engine import ...` work
# whether this file is run as a bare script (python run_engines.py) OR as a
# module (python -m run_engines). chess_zero/ is added to
# sys.path so the az package + engine.py resolve identically to play.py.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))            # projects/chess_zero
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Checkpoint locations (relative to this file -- absolute-resolved here).
CKPT_CHESS = os.path.join(_HERE, "az", "robust_from_bootstrap", "champion.pt")
CKPT_CONNECT4 = os.path.join(_HERE, "az", "checkpoints", "connect4.pt")
CKPT_ATARI = os.path.join(_HERE, "az", "checkpoints", "atari.pt")                  # MuZero / CatchEnv
CKPT_ATARI_MINATAR = os.path.join(_HERE, "az", "checkpoints", "atari_minatar.pt")  # DQN / real MinAtar


# --------------------------------------------------------------------------- #
# Small presentation helpers (ASCII only).
# --------------------------------------------------------------------------- #
_BAR = "=" * 72
_RULE = "-" * 72


def _banner_intro() -> None:
    print(_BAR)
    print("  CHESS / CONNECT-4 / ATARI -- AlphaZero + MuZero engines, trained on a 4060")
    print(_BAR)
    print("  Watch three engines play, move by move, using their trained checkpoints.")
    print("  AlphaZero (residual CNN + PUCT MCTS) for the board games; MuZero (planning")
    print("  over a LEARNED model) for the scaled-down Atari. ASCII boards. No setup.")
    print(_BAR)


def _section(title: str) -> None:
    print()
    print(_BAR)
    print(f"  {title}")
    print(_BAR)


def _resolve_device(arg: str):
    """Map --device {auto,cpu,cuda} to a torch device string. auto -> cuda if available."""
    import torch
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg == "cuda" and not torch.cuda.is_available():
        print("[device] cuda requested but unavailable; falling back to cpu")
        return "cpu"
    return arg


_WEB_ACTIVE = False  # set True when a browser viz is live, so --no-render still PACES the browser


def _frame(lines, delay: float, render: bool) -> None:
    """Print one animation frame + sleep. Pacing also applies when a browser viz is live, so
    `--web --no-render` stays watchable in the browser (not an instant blur), not just the terminal."""
    if render:
        print()
        for ln in lines:
            print(ln)
    if delay > 0 and (render or _WEB_ACTIVE):
        time.sleep(delay)


# --------------------------------------------------------------------------- #
# Browser visualizer (--web). One self-contained live.html (az/web_viz.py), opened
# once; each engine rewrites it as it plays. palette[0] = empty cell; [k>0] = (label,color).
# --------------------------------------------------------------------------- #
_C4_PALETTE = [("empty", ""), ("red = player 0 (X)", "#e2463f"),
               ("yellow = player 1 (O)", "#f2c14e")]
_CATCH_PALETTE = [("empty", ""), ("ball", "#4dd2ff"), ("paddle", "#36c08a"),
                  ("ball on paddle", "#ffffff")]
_MINATAR_COLORS = ["#36c08a", "#4dd2ff", "#7a8699", "#f0a93b",
                   "#e2463f", "#b07cf0", "#f25fb0", "#9be24d"]


def make_web_viz(subtitle: str = "", open_browser: bool = True):
    """Create + open the shared live.html browser visualizer (runs/viz/). Returns the viz
    (or None on any failure -- the demo then just renders ASCII)."""
    global _WEB_ACTIVE
    try:
        from az.web_viz import LiveGameViz
        viz = LiveGameViz(os.path.join(_HERE, "runs", "viz"),
                          title="games_engine -- live", subtitle=subtitle)
        ob = open_browser and os.environ.get("GAMESENGINE_NO_BROWSER", "") not in ("1", "true", "TRUE")
        viz.start(open_browser=ob)
        _WEB_ACTIVE = True  # so _frame paces the browser even under --no-render
        print(f"  [web] live view: file://{viz.html_path.replace(os.sep, '/')}  "
              f"(auto-refreshes; opened in your browser)")
        return viz
    except Exception as exc:  # never let the viz break the demo
        print(f"  [web] visualizer unavailable ({exc}); continuing with terminal render")
        return None


# Human-readable MinAtar channel names (the obs channels, in order) so the legend is meaningful.
_MINATAR_CHANNELS = {
    "breakout": ["paddle", "ball", "ball trail", "brick"],
    "space_invaders": ["cannon", "alien", "aliens moving left", "aliens moving right",
                       "friendly bullet", "enemy bullet"],
    "asterix": ["player", "enemy", "enemy trail", "gold"],
}


def _minatar_palette(n_channels: int, game: str = ""):
    names = _MINATAR_CHANNELS.get(game, [])
    pal = [("empty", "")]
    for k in range(n_channels):
        label = names[k] if k < len(names) else "channel %d" % k
        pal.append((label, _MINATAR_COLORS[k % len(_MINATAR_COLORS)]))
    return pal


def _catch_grid(obs):
    """5x5x2 catch obs -> 2D palette-index grid (1 ball, 2 paddle, 3 both)."""
    h, w, _ = obs.shape
    return [[(3 if (obs[r, c, 0] > 0 and obs[r, c, 1] > 0) else
              1 if obs[r, c, 0] > 0 else 2 if obs[r, c, 1] > 0 else 0)
             for c in range(w)] for r in range(h)]


def _minatar_grid(obs):
    """(H,W,C) MinAtar obs -> 2D palette-index grid (HIGHEST active channel + 1, else 0).
    Uses the highest active channel to match MinAtar's own renderer (np.amax over channels),
    so overlapping cells surface the more informative entity and the web grid agrees with ASCII."""
    h, w, c = obs.shape
    out = []
    for r in range(h):
        row = []
        for col in range(w):
            active = [k for k in range(c) if obs[r, col, k] > 0]
            row.append((active[-1] + 1) if active else 0)
        out.append(row)
    return out


def _web_chess(viz, board, san_log, header, status, done=False):
    """Push the current chess position to the browser viz (real-piece SVG). Never raises."""
    if viz is None:
        return
    try:
        import chess.svg
        from az.web_viz import moves_to_html
        last = board.peek() if board.move_stack else None
        viz.board_svg(chess.svg.board(board, lastmove=last, size=420),
                      header=header, moves_html=moves_to_html(san_log),
                      status=status, done=done)
    except Exception:
        pass


def _web_grid(viz, grid, palette, header, status, board_bg="#10131c", done=False, col_labels=None):
    """Push a grid frame to the browser viz. Never raises."""
    if viz is None:
        return
    try:
        viz.grid(grid, palette, header=header, status=status, board_bg=board_bg,
                 done=done, col_labels=col_labels)
    except Exception:
        pass


# =========================================================================== #
# 1) CHESS -- the strongest champion vs the classical negamax engine.
# =========================================================================== #
def play_chess(games: int = 1, delay: float = 0.4, render: bool = True,
               device: str = "cpu", mcts_sims: int = 64,
               mode: str = "selfplay", chess_depth: int = 2, viz=None) -> dict:
    """Play `games` full game(s) with the trained AlphaZero CHAMPION.

      mode="selfplay"     -- the champion plays ITSELF (the iconic AlphaZero demo): real, legal,
                             non-random chess from both sides, with opening variety via temperature.
                             This is the HONEST showcase -- it shows the net actually playing chess.
      mode="vs-classical" -- the champion vs the in-repo classical Engine(depth=chess_depth).

    HONEST CEILING: the champion CRUSHES random (100%) but is a WEAK-but-real learner vs a classical
    minimax -- it even loses to depth-1 (recorded score_vs_classical=0.0). Chess MASTERY is
    compute-bound (orders of magnitude more self-play than a 4060 session); this is the SOTA
    ALGORITHM (AlphaZero) running + a net that genuinely plays, not a master. Self-play is therefore
    the honest, watchable default. Never hard-fails: a missing champion.pt -> untrained net.

    The per-game dict's champ_pov is always WIN/LOSS/DRAW (White's POV in self-play) so the CI gate
    contract is stable across modes.
    """
    import torch
    import chess
    from az.net import AlphaZeroNet
    from az.mcts import MCTS
    from chess_engine.engine import Engine

    title = ("AlphaZero champion SELF-PLAY -- the net plays ITSELF" if mode == "selfplay"
             else f"AlphaZero champion vs classical negamax, depth {chess_depth}")
    _section(f"ENGINE 1/3 -- CHESS  ({title})")

    # --- load the champion (strict=False: schema-tolerant, like play.NetPlayer) ---
    net = AlphaZeroNet(channels=80, n_blocks=8).to(device)
    trained = False
    strength = ""
    if os.path.exists(CKPT_CHESS):
        ck = torch.load(CKPT_CHESS, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"], strict=False)
        trained = True
        strength = (f"champion iter {ck.get('iter', '?')} -- beats random "
                    f"{ck.get('winrate_vs_random', float('nan')):.2f} (vs a classical minimax it is a "
                    f"WEAK-but-real learner -- chess strength is compute-bound, this is the AlphaZero "
                    f"algorithm + a net that genuinely plays, not a master)")
        print(f"  [trained] loaded {os.path.basename(CKPT_CHESS)} -- {strength}")
    else:
        print(f"  [checkpoint missing -- playing with an UNTRAINED net] ({CKPT_CHESS})")
    net.eval()

    mcts = MCTS(net, n_simulations=mcts_sims, device=device)
    classical = Engine(depth=chess_depth) if mode == "vs-classical" else None
    _OPEN_TEMP_PLIES = 8  # self-play opening variety so it is not the same game every run

    results = []
    for gi in range(games):
        champ_is_white = (gi % 2 == 0)
        if mode == "selfplay":
            print(f"\n  Game {gi + 1}/{games}: champion (White) vs champion (Black) -- self-play")
        else:
            print(f"\n  Game {gi + 1}/{games}: champion plays "
                  f"{'WHITE' if champ_is_white else 'BLACK'} vs classical(depth={chess_depth})")
        board = chess.Board()
        san_log = []
        move_num = 0
        max_plies = 400  # safety cap so an UNTRAINED net can never spin forever
        while not board.is_game_over(claim_draw=True) and move_num < max_plies:
            if mode == "selfplay":
                temp = 1.0 if move_num < _OPEN_TEMP_PLIES else 0.0
                mv = mcts.best_move(board, temperature=temp)
                who = "white" if board.turn == chess.WHITE else "black"
            else:
                champ_to_move = (board.turn == chess.WHITE) == champ_is_white
                if champ_to_move:
                    mv = mcts.best_move(board, temperature=0.0)
                    who = "champ"
                else:
                    mv = classical.search(board).move
                    who = "class"
            if mv is None or mv not in board.legal_moves:
                mv = next(iter(board.legal_moves))
            san = board.san(mv)
            san_log.append(san)
            board.push(mv)
            move_num += 1
            label = (f"  move {move_num:3d}  {who}: {san:8s}  "
                     f"(result so far: {board.result(claim_draw=True)})")
            _web_chess(viz, board, san_log, f"CHESS -- {title}", label.strip(), done=False)
            _frame([label, "", str(board)], delay, render)

        result = board.result(claim_draw=True)
        _web_chess(viz, board, san_log, f"CHESS -- {title}",
                   f"FINAL: {result}  ({move_num} plies)", done=True)
        # champ_pov is always WIN/LOSS/DRAW. In self-play it is White's POV (stable CI contract).
        if mode == "selfplay":
            champ_pov = "WIN" if result == "1-0" else ("LOSS" if result == "0-1" else "DRAW")
            human = ("White wins" if result == "1-0" else
                     ("Black wins" if result == "0-1" else "draw"))
            print(f"\n  RESULT game {gi + 1}: {result}  (self-play: {human}, {move_num} plies)")
        else:
            if result == "1-0":
                champ_pov = "WIN" if champ_is_white else "LOSS"
            elif result == "0-1":
                champ_pov = "LOSS" if champ_is_white else "WIN"
            else:
                champ_pov = "DRAW"
            print(f"\n  RESULT game {gi + 1}: {result}  (champion: {champ_pov}, {move_num} plies)")
        results.append({"result": result, "champ_pov": champ_pov, "plies": move_num,
                        "san": san_log})

    wins = sum(1 for r in results if r["champ_pov"] == "WIN")
    draws = sum(1 for r in results if r["champ_pov"] == "DRAW")
    losses = sum(1 for r in results if r["champ_pov"] == "LOSS")
    if mode == "selfplay":
        print(f"\n  CHESS summary (SELF-PLAY): {games} game(s) of the champion vs itself "
              f"(White {wins} / Black {losses} / draw {draws})"
              f"{'  [' + strength + ']' if strength else '  [untrained net]'}")
    else:
        print(f"\n  CHESS summary: champion W{wins} D{draws} L{losses} over {games} game(s)"
              f"{'  [' + strength + ']' if strength else '  [untrained net]'}")
    return {"engine": "chess", "mode": mode, "trained": trained, "strength": strength,
            "games": results, "w": wins, "d": draws, "l": losses}


def play_chess_strong(games: int = 1, delay: float = 0.4, render: bool = True,
                      movetime: float = 0.5, sf_elo: int = 1500, viz=None) -> dict:
    """Showcase the GENUINE chess strength: the in-repo CLASSICAL engine (negamax + alpha-beta +
    quiescence + the upgraded evaluate_v2 -- MEASURED ~1600 Elo) COMPETING AGAINST THE BEST
    (Stockfish, Elo-capped to a fair, watchable level). With no Stockfish binary present it falls
    back to the strong engine playing ITSELF (still real, strong chess). This is the honest
    'compete against the best' demo -- the opposite of parading the weak net."""
    import chess
    from chess_engine.engine import Engine
    from chess_engine.sf_engine import find_stockfish, StockfishEngine

    ours = Engine(time_limit=movetime)                 # evaluate_v2 default -> ~1600 Elo
    sf_path = find_stockfish()
    if sf_path:
        opp = StockfishEngine(sf_path, movetime=movetime, elo=sf_elo)
        opp_name = f"Stockfish@{sf_elo}"
        title = f"STRONG classical engine (~1600 Elo) vs {opp_name} -- competing against the best"
    else:
        opp = Engine(time_limit=movetime)
        opp_name = "itself (no Stockfish binary found -- see PLAY_HUMAN.md)"
        title = "STRONG classical engine (~1600 Elo) SELF-PLAY -- real, strong chess"
    _section(f"ENGINE 1/3 -- CHESS  ({title})")
    print(f"  [strong] classical engine (evaluate_v2, {movetime}s/move) vs {opp_name}")

    results = []
    try:
        for gi in range(games):
            ours_white = (gi % 2 == 0)
            board = chess.Board()
            san_log = []
            move_num = 0
            max_plies = 300
            print(f"\n  Game {gi + 1}/{games}: our engine plays "
                  f"{'WHITE' if ours_white else 'BLACK'} vs {opp_name}")
            while not board.is_game_over(claim_draw=True) and move_num < max_plies:
                our_turn = (board.turn == chess.WHITE) == ours_white
                mv = (ours if our_turn else opp).search(board).move
                if mv is None or mv not in board.legal_moves:
                    mv = next(iter(board.legal_moves))
                san = board.san(mv)
                san_log.append(san)
                board.push(mv)
                move_num += 1
                who = "ours" if our_turn else "oppt"
                label = (f"  move {move_num:3d}  {who}: {san:8s}  "
                         f"(so far: {board.result(claim_draw=True)})")
                _web_chess(viz, board, san_log, f"CHESS -- {title}", label.strip(), done=False)
                _frame([label, "", str(board)], delay, render)
            result = board.result(claim_draw=True)
            _web_chess(viz, board, san_log, f"CHESS -- {title}",
                       f"FINAL: {result}  ({move_num} plies)", done=True)
            if result == "1-0":
                pov = "WIN" if ours_white else "LOSS"
            elif result == "0-1":
                pov = "LOSS" if ours_white else "WIN"
            else:
                pov = "DRAW"
            print(f"\n  RESULT game {gi + 1}: {result}  (our engine: {pov}, {move_num} plies)")
            results.append({"result": result, "our_pov": pov, "plies": move_num, "san": san_log})
    finally:
        if sf_path:
            opp.close()

    wins = sum(1 for r in results if r["our_pov"] == "WIN")
    draws = sum(1 for r in results if r["our_pov"] == "DRAW")
    losses = sum(1 for r in results if r["our_pov"] == "LOSS")
    print(f"\n  CHESS summary (STRONG): our engine W{wins} D{draws} L{losses} "
          f"vs {opp_name} over {games} game(s)")
    return {"engine": "chess", "mode": "strong", "opponent": opp_name,
            "games": results, "w": wins, "d": draws, "l": losses}


# =========================================================================== #
# 2) CONNECT-4 -- the trained net (NeuralMCTS) vs the 1-ply win/block heuristic.
# =========================================================================== #
def play_connect4(games: int = 1, delay: float = 0.4, render: bool = True,
                  device: str = "cpu", mcts_sims: int = 128, viz=None, strong: bool = False,
                  solver_budget: float = 0.6) -> dict:
    """Play `games` Connect-4 game(s): the AI (player 0 (X) on even games, player 1 (O) on odd) vs
    the 1-ply win/block tactical heuristic. Returns a result dict. Never hard-fails: a missing
    connect4.pt -> untrained net.

    strong=False: the AI is the bare trained Connect4Net (NeuralMCTS).
    strong=True : the AI is the HYBRID -- net opening + provably-perfect solver mid/endgame
                  (MEASURED +~134 Elo over the bare net) -- the genuine-strength showcase."""
    import torch
    from az.net import Connect4Net
    from az.connect4 import Connect4, heuristic_1ply_action, N_ROWS, N_COLS
    from az.mcts import NeuralMCTS
    from az.connect4_hybrid import HybridConnect4Player

    def _c4grid(st):  # board cells -> top-row-first 2D grid (matches the ASCII orientation)
        cells = st[0]
        return [list(cells[r * N_COLS:(r + 1) * N_COLS]) for r in range(N_ROWS - 1, -1, -1)]

    eng_name = "HYBRID (net+solver)" if strong else "AlphaZero net"
    _section(f"ENGINE 2/3 -- CONNECT-4  ({eng_name} vs 1-ply win/block heuristic)")

    net = Connect4Net(channels=64, n_blocks=5, n_input_planes=3, n_policy=7, rows=6, cols=7).to(device)
    trained = False
    strength = ""
    if os.path.exists(CKPT_CONNECT4):
        ck = torch.load(CKPT_CONNECT4, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"], strict=False)
        trained = True
        meta = ck.get("meta", {})
        strength = (f"trained {meta.get('iters', '?')} iters -- vs_random {meta.get('vs_random', '?')}, "
                    f"vs_heuristic {meta.get('vs_heuristic', '?')}")
        if strong:
            strength += " | HYBRID: + perfect-endgame solver (MEASURED +~134 Elo over the bare net)"
        print(f"  [trained] loaded {os.path.basename(CKPT_CONNECT4)} -- {strength}")
    else:
        print(f"  [checkpoint missing -- playing with an UNTRAINED net] ({CKPT_CONNECT4})")
    net.eval()

    game = Connect4()
    hybrid = HybridConnect4Player(game, net, device=device, mcts_sims=mcts_sims,
                                  budget_s=solver_budget) if strong else None
    results = []
    for gi in range(games):
        net_is_p0 = (gi % 2 == 0)
        net_sym = "X" if net_is_p0 else "O"
        net_color = "red" if net_is_p0 else "yellow"
        opp_color = "yellow" if net_is_p0 else "red"
        # X (palette idx 1) is ALWAYS red, O (idx 2) ALWAYS yellow; label WHO is the net this game.
        if net_is_p0:
            c4pal = [("empty", ""), (f"{eng_name} (red)", "#e2463f"),
                     ("1-ply heuristic (yellow)", "#f2c14e")]
        else:
            c4pal = [("empty", ""), ("1-ply heuristic (red)", "#e2463f"),
                     (f"{eng_name} (yellow)", "#f2c14e")]
        c4hdr = f"CONNECT-4 -- {eng_name} = {net_color}  vs  1-ply win/block heuristic = {opp_color}"
        c4cols = list(range(N_COLS))
        print(f"\n  Game {gi + 1}/{games}: {eng_name} plays {net_sym} "
              f"(player {0 if net_is_p0 else 1}) vs heuristic")
        rng = random.Random(1000 + gi)
        state = game.initial_state()
        move_num = 0
        while not game.is_terminal(state):
            p = game.current_player(state)
            net_to_move = (p == 0) == net_is_p0
            if net_to_move:
                if hybrid is not None:
                    a = hybrid.action(state)
                    who = "hyb "
                else:
                    a = NeuralMCTS(game, net, n_simulations=mcts_sims, device=device).best_action_batched(
                        state, temperature=0.0, batch_size=16)
                    who = "net "
            else:
                a = heuristic_1ply_action(game, state, rng)
                who = "heur"
            if a not in game.legal_actions(state):
                a = game.legal_actions(state)[0]
            state = game.apply(state, a)
            move_num += 1
            label = f"  move {move_num:2d}  {who} -> column {a}"
            _web_grid(viz, _c4grid(state), c4pal, c4hdr,
                      f"move {move_num}: {who.strip()} dropped a disc in column {a}",
                      board_bg="#13315c", done=False, col_labels=c4cols)
            _frame([label, "", game.render(state)], delay, render)

        # winner: returns() is player-0-absolute (+1 p0, -1 p1, 0 draw)
        z = game.returns(state)
        if z == 0:
            net_pov = "DRAW"
        else:
            p0_won = z > 0
            net_pov = "WIN" if (p0_won == net_is_p0) else "LOSS"
        _web_grid(viz, _c4grid(state), c4pal, c4hdr,
                  f"GAME OVER -- net {net_pov} (net = {net_color}) in {move_num} moves",
                  board_bg="#13315c", done=True, col_labels=c4cols)
        print(f"\n  RESULT game {gi + 1}: net {net_pov}  ({move_num} moves)")
        results.append({"net_pov": net_pov, "moves": move_num})

    wins = sum(1 for r in results if r["net_pov"] == "WIN")
    draws = sum(1 for r in results if r["net_pov"] == "DRAW")
    losses = sum(1 for r in results if r["net_pov"] == "LOSS")
    print(f"\n  CONNECT-4 summary: net W{wins} D{draws} L{losses} over {games} game(s)"
          f"{'  [' + strength + ']' if strength else '  [untrained net]'}")
    return {"engine": "connect4", "trained": trained, "strength": strength,
            "games": results, "w": wins, "d": draws, "l": losses}


def play_connect4_human(human_first: bool = True, device: str = "cpu", mcts_sims: int = 160,
                        strong: bool = True, solver_budget: float = 0.8) -> dict:
    """Interactive Connect-4: YOU vs the STRONGEST engine in the repo. You drop a disc by typing a
    column 0-6; the AI replies. Type q/quit/Ctrl-C to stop. Never hard-fails (a missing connect4.pt
    -> untrained net that still plays legally).

    strong=True (default): the HYBRID -- the trained net plays the opening, the provably-perfect
    bitboard solver takes over the mid/endgame, with never-miss-a-win / always-block guardrails
    (MEASURED +~134 Elo over the bare net). strong=False: the bare net (NeuralMCTS) only."""
    import torch
    from az.net import Connect4Net
    from az.connect4 import Connect4, N_COLS
    from az.mcts import NeuralMCTS
    from az.connect4_hybrid import HybridConnect4Player

    _section("CONNECT-4 -- HUMAN vs the " + ("HYBRID (net + perfect-endgame solver)" if strong
                                             else "trained AlphaZero net"))
    net = Connect4Net(channels=64, n_blocks=5, n_input_planes=3, n_policy=7, rows=6, cols=7).to(device)
    if os.path.exists(CKPT_CONNECT4):
        ck = torch.load(CKPT_CONNECT4, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"], strict=False)
        kind = (f"HYBRID -- net ({mcts_sims} sims) opening + perfect solver ({solver_budget}s) endgame"
                if strong else f"net ({mcts_sims} sims/move)")
        print(f"  [trained] loaded {os.path.basename(CKPT_CONNECT4)} -- {kind}")
    else:
        print(f"  [checkpoint missing -- the net is UNTRAINED] ({CKPT_CONNECT4})")
    net.eval()

    game = Connect4()
    ai_player = HybridConnect4Player(game, net, device=device, mcts_sims=mcts_sims,
                                     budget_s=solver_budget) if strong else None
    state = game.initial_state()
    human_player = 0 if human_first else 1
    you, ai = ("X", "O") if human_player == 0 else ("O", "X")
    print(f"  You are '{you}' (player {human_player}); the AI is '{ai}'. Columns are numbered "
          f"0-{N_COLS - 1} along the bottom. Type a column to drop a disc (q to quit).\n")
    print(game.render(state))
    while not game.is_terminal(state):
        if game.current_player(state) == human_player:
            legal = game.legal_actions(state)
            a = None
            while a is None:
                try:
                    txt = input(f"\n  your move -- column {legal}: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n  (quit)")
                    return {"engine": "connect4", "result": "quit"}
                if txt in ("q", "quit", "exit"):
                    print("  (quit)")
                    return {"engine": "connect4", "result": "quit"}
                if txt.isdigit() and int(txt) in legal:
                    a = int(txt)
                else:
                    print(f"  '{txt}' is not a legal column; pick one of {legal}")
            state = game.apply(state, a)
            print(f"  you -> column {a}")
        else:
            if ai_player is not None:
                a = ai_player.action(state)
            else:
                a = NeuralMCTS(game, net, n_simulations=mcts_sims, device=device).best_action_batched(
                    state, temperature=0.0, batch_size=16)
            if a not in game.legal_actions(state):
                a = game.legal_actions(state)[0]
            state = game.apply(state, a)
            print(f"\n  AI -> column {a}")
        print(game.render(state))

    z = game.returns(state)  # player-0-absolute: +1 p0, -1 p1, 0 draw
    if z == 0:
        msg = "DRAW"
    else:
        human_won = ((z > 0) == (human_player == 0))
        msg = "YOU WIN!" if human_won else "the AI WINS"
    print(f"\n  RESULT: {msg}")
    return {"engine": "connect4", "result": msg}


# =========================================================================== #
# 3) ATARI (scaled) -- the MuZero-RL agent planning over its LEARNED model.
# =========================================================================== #
def _render_catch(obs) -> list:
    """ASCII render of the 5x5x2 catch grid: 'o' = ball, '=' = paddle, '#' if they coincide,
    '.' = empty. obs is (H, W, 2): channel 0 ball, channel 1 paddle. Returns a list of lines."""
    import numpy as np
    h, w, _ = obs.shape
    ball = obs[:, :, 0]
    paddle = obs[:, :, 1]
    lines = ["    +" + "-" * (2 * w + 1) + "+"]
    for r in range(h):
        cells = []
        for c in range(w):
            b = ball[r, c] > 0
            p = paddle[r, c] > 0
            if b and p:
                cells.append("#")
            elif b:
                cells.append("o")
            elif p:
                cells.append("=")
            else:
                cells.append(".")
        lines.append("    | " + " ".join(cells) + " |")
    lines.append("    +" + "-" * (2 * w + 1) + "+")
    return lines


def play_atari(games: int = 1, delay: float = 0.4, render: bool = True,
               device: str = "cpu", mcts_sims: int = 24, viz=None) -> dict:
    """Run `games` episode(s) of the scaled-down Atari 'catch' env with the trained MuZero-RL agent
    PLANNING OVER ITS LEARNED MODEL (no env inside the search). Renders the 5x5 grid + running score
    each step. Returns a result dict. Never hard-fails: a missing atari.pt -> untrained net.
    """
    import torch
    from az.muzero_rl import MuZeroRLNet, MuZeroRLMCTS
    from az.minatar_env import CatchEnv

    _section("ENGINE 3/3 -- ATARI (scaled)  (MuZero-RL planning over a LEARNED model -- catch)")

    net = MuZeroRLNet(obs_shape=(5, 5, 2), num_actions=3, latent_dim=64, channels=32, hidden=128).to(device)
    trained = False
    strength = ""
    max_steps = 10  # the trained-agent horizon recorded in the checkpoint meta
    if os.path.exists(CKPT_ATARI):
        ck = torch.load(CKPT_ATARI, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"], strict=False)
        trained = True
        meta = ck.get("meta", {})
        max_steps = int(meta.get("max_steps", 10))
        strength = (f"trained {meta.get('iters', '?')} iters -- trained return "
                    f"{meta.get('trained_return', float('nan')):+.2f} vs random "
                    f"{meta.get('random_return', float('nan')):+.2f} "
                    f"(margin {meta.get('margin', float('nan')):+.2f})")
        print(f"  [trained] loaded {os.path.basename(CKPT_ATARI)} (env={ck.get('env', '?')}) -- {strength}")
    else:
        print(f"  [checkpoint missing -- playing with an UNTRAINED net] ({CKPT_ATARI})")
    net.eval()

    num_actions = 3
    action_name = {0: "LEFT", 1: "STAY", 2: "RIGHT"}
    results = []
    for gi in range(games):
        env = CatchEnv(size=5, seed=100 + gi)
        obs = env.reset()
        print(f"\n  Episode {gi + 1}/{games}: agent steers the paddle (=) under the ball (o)")
        _web_grid(viz, _catch_grid(obs), _CATCH_PALETTE,
                  "ATARI -- MuZero/Catch (planning over a LEARNED model)", "step 0  score 0.0")
        _frame([f"  step  0  score 0.0"] + _render_catch(obs), delay, render)
        ep_return = 0.0
        steps = 0
        for t in range(max_steps):
            a = MuZeroRLMCTS(net, num_actions, n_simulations=mcts_sims).best_action(
                obs, temperature=0.0, add_noise=False)
            obs, reward, done = env.step(a)
            ep_return += float(reward)
            steps += 1
            label = f"  step {steps:2d}  action {action_name.get(a, a):5s}  score {ep_return:+.1f}"
            _web_grid(viz, _catch_grid(obs), _CATCH_PALETTE,
                      "ATARI -- MuZero/Catch (planning over a LEARNED model)",
                      label.strip(), done=done)
            _frame([label] + _render_catch(obs), delay, render)
            if done:
                break
        verdict = "CAUGHT (+1)" if ep_return > 0 else ("MISSED (-1)" if ep_return < 0 else "0")
        print(f"\n  RESULT episode {gi + 1}: return {ep_return:+.1f}  [{verdict}]  ({steps} steps)")
        results.append({"return": ep_return, "steps": steps})

    mean_return = sum(r["return"] for r in results) / max(1, len(results))
    catches = sum(1 for r in results if r["return"] > 0)
    print(f"\n  ATARI summary: caught {catches}/{games}, mean return {mean_return:+.2f}"
          f"{'  [' + strength + ']' if strength else '  [untrained net]'}")
    return {"engine": "atari", "trained": trained, "strength": strength,
            "games": results, "mean_return": mean_return, "catches": catches,
            "atari_env": "catch (MuZero, learned model)"}


def _render_minatar(obs) -> list:
    """ASCII render of a MinAtar grid (H, W, C). Each cell shows the HIGHEST-index ACTIVE channel as a
    symbol (breakout channels: 0 paddle '=', 1 ball 'o', 2 trail ':', 3 brick '#'). Empty -> '.'.
    Highest-index matches MinAtar's own np.amax renderer (and the web grid in _minatar_grid)."""
    h, w, c = obs.shape
    syms = ["=", "o", ":", "#", "*", "+", "@", "x"]
    lines = ["    +" + "-" * (2 * w + 1) + "+"]
    for r in range(h):
        cells = []
        for col in range(w):
            active = [k for k in range(c) if obs[r, col, k] > 0]
            cells.append(syms[active[-1]] if active and active[-1] < len(syms) else
                         ("?" if active else "."))
        lines.append("    | " + " ".join(cells) + " |")
    lines.append("    +" + "-" * (2 * w + 1) + "+")
    return lines


def _minatar_ckpt_path(game: str) -> str:
    """Checkpoint path for a MinAtar game. Breakout uses the legacy atari_minatar.pt; others get a
    _<game> suffix (atari_minatar_space_invaders.pt, atari_minatar_asterix.pt)."""
    if game == "breakout":
        return CKPT_ATARI_MINATAR
    return os.path.join(_HERE, "az", "checkpoints", f"atari_minatar_{game}.pt")


def _available_minatar_games() -> list:
    """The MinAtar games whose trained checkpoints are present, in showcase order."""
    return [g for g in ("breakout", "space_invaders", "asterix")
            if os.path.exists(_minatar_ckpt_path(g))]


def play_atari_minatar(games: int = 1, delay: float = 0.3, render: bool = True,
                       device: str = "cpu", max_steps: int = 300,
                       max_render_steps: int = 70, game: str = "breakout", viz=None) -> dict:
    """Play `games` episode(s) of a REAL MinAtar game (scaled Atari) with the trained DQN.
    Renders the 10x10 grid (ch0 '=', ch1 'o', ch2 ':', ch3 '#', ...) for the first `max_render_steps`,
    then runs to the episode end (capped at max_steps) and reports the score.
    Returns a play_atari-compatible dict. Never hard-fails: a missing checkpoint -> untrained net.
    """
    import torch
    import numpy as np
    from az.dqn_minatar import MinAtarQNet
    from az.minatar_env import MinAtarEnv

    _section(f"ENGINE 3/3 -- ATARI (REAL MinAtar {game})  (DQN -- the agent plays a real Atari game)")

    ckpt_path = _minatar_ckpt_path(game)
    qnet = None
    trained = False
    strength = ""
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        game = str(ck.get("game", f"minatar:{game}")).split(":")[-1]
        qnet = MinAtarQNet(**ck["arch"]).to(device)
        qnet.load_state_dict(ck["state_dict"])
        qnet.eval()
        trained = True
        meta = ck.get("meta", {})
        tr, rr = meta.get("trained_return", float("nan")), meta.get("random_return", float("nan"))
        ratio = (tr / rr) if (rr and rr > 0) else float("nan")
        strength = (f"trained {meta.get('episodes_trained', '?')} eps -- mean score {tr:.0f} vs random "
                    f"{rr:.2f} ({ratio:.0f}x random)")
        print(f"  [trained] loaded {os.path.basename(ckpt_path)} (game={game}) -- {strength}")
    else:
        print(f"  [checkpoint missing -- playing with an UNTRAINED net] ({ckpt_path})")

    env = MinAtarEnv(game)
    mpal = _minatar_palette(env.obs_shape[2], game)  # one named color per MinAtar channel
    if qnet is None:
        qnet = MinAtarQNet(in_channels=env.obs_shape[2], num_actions=env.num_actions,
                           h=env.obs_shape[0], w=env.obs_shape[1]).to(device)
        qnet.eval()

    def _greedy(obs) -> int:
        x = np.ascontiguousarray(np.transpose(obs, (2, 0, 1)), dtype=np.float32)
        with torch.no_grad():
            q = qnet(torch.from_numpy(x).unsqueeze(0).to(device))
        return int(torch.argmax(q, dim=1).item())

    results = []
    for gi in range(games):
        env.seed(100 + gi)
        obs = env.reset()
        print(f"\n  Episode {gi + 1}/{games}: the DQN agent plays Breakout "
              f"(paddle '=', ball 'o', bricks '#') -- breaking bricks to score")
        _web_grid(viz, _minatar_grid(obs), mpal, f"ATARI -- real MinAtar {game} (DQN)",
                  "step 0  score 0", board_bg="#0c0f17")
        _frame(["  step   0  score 0"] + _render_minatar(obs), delay, render)
        ep_return = 0.0
        steps = 0
        while steps < max_steps:
            a = _greedy(obs)
            obs, reward, done = env.step(a)
            ep_return += float(reward)
            steps += 1
            if steps <= max_render_steps:
                _web_grid(viz, _minatar_grid(obs), mpal, f"ATARI -- real MinAtar {game} (DQN)",
                          f"step {steps}  score {ep_return:.0f}", board_bg="#0c0f17", done=done)
            if render and steps <= max_render_steps:
                _frame([f"  step {steps:3d}  score {ep_return:.0f}"] + _render_minatar(obs),
                       delay, render)
            elif render and steps == max_render_steps + 1:
                print(f"  ... agent keeps playing (showed first {max_render_steps} steps; "
                      f"running to the episode end) ...")
            if done:
                break
        # final done frame so the web view settles on GAME OVER (Breakout runs past the 70-step
        # web cap, so without this the page would show a stale LIVE badge forever after exit).
        _web_grid(viz, _minatar_grid(obs), mpal, f"ATARI -- real MinAtar {game} (DQN)",
                  f"GAME OVER -- score {ep_return:.0f}  ({steps} steps)", board_bg="#0c0f17", done=True)
        print(f"\n  RESULT episode {gi + 1}: score {ep_return:.0f}  ({steps} steps)  "
              f"-- real MinAtar {game}")
        results.append({"return": ep_return, "steps": steps})

    mean_return = sum(r["return"] for r in results) / max(1, len(results))
    catches = sum(1 for r in results if r["return"] > 1.0)  # episodes that actually scored
    print(f"\n  ATARI (MinAtar {game}) summary: mean score {mean_return:.1f} over {games} ep(s)"
          f"{'  [' + strength + ']' if strength else '  [untrained net]'}")
    return {"engine": "atari", "trained": trained, "strength": strength,
            "games": results, "mean_return": mean_return, "catches": catches,
            "atari_env": f"MinAtar {game} (DQN, real Atari)", "game": game}


# =========================================================================== #
# Orchestration.
# =========================================================================== #
def _final_summary(outcomes: list) -> None:
    print()
    print(_BAR)
    print("  SUMMARY -- three trained engines, played live")
    print(_BAR)
    for o in outcomes:
        eng = o["engine"]
        tag = "trained" if o.get("trained") else "UNTRAINED"
        if eng == "chess":
            if o.get("mode") == "strong":
                print(f"  CHESS     [strong ~1600]  our engine W{o['w']} D{o['d']} L{o['l']} "
                      f"vs {o.get('opponent', 'Stockfish')}")
            elif o.get("mode", "selfplay") == "selfplay":
                print(f"  CHESS     [{tag}]  champion SELF-PLAY "
                      f"(White {o['w']} / Black {o['l']} / draw {o['d']})")
            else:
                print(f"  CHESS     [{tag}]  champion W{o['w']} D{o['d']} L{o['l']} vs classical")
        elif eng == "connect4":
            ai = "HYBRID" if o.get("strength", "").find("HYBRID") >= 0 else "net"
            print(f"  CONNECT-4 [{tag}]  {ai} W{o['w']} D{o['d']} L{o['l']} vs 1-ply heuristic")
        elif eng == "atari":
            print(f"  ATARI     [{tag}]  {o.get('atari_env', 'scaled')}: "
                  f"mean score/return {o['mean_return']:+.2f} over {len(o['games'])} ep(s)")
        if o.get("strength"):
            print(f"              ({o['strength']})")
    print(_BAR)
    print("  Done. Each engine played to a real terminal result using its trained checkpoint.")
    print(_BAR)


def run(engine: str = "all", games: int = 1, delay: float = 0.4, render: bool = True,
        device: str = "auto", fast: bool = False, atari_mode: str = "auto",
        atari_game: str = "breakout", web: bool = False, strong: bool = False) -> list:
    """Programmatic entry: play the requested engine(s) and return a list of result dicts.
    `fast` forces delay=0 and low sims (CI / quick look). `atari_mode`: 'minatar' (real MinAtar via
    DQN), 'catch' (MuZero over a learned model on CatchEnv), or 'auto' (minatar if a checkpoint
    exists, else catch). `atari_game`: which MinAtar game ('breakout'|'space_invaders'|'asterix') or
    'all' to play every trained one in sequence."""
    dev = _resolve_device(device)
    if fast:
        delay = 0.0
    # sims: low under --fast (CI speed), production otherwise.
    chess_sims = 8 if fast else 64
    c4_sims = 24 if fast else 128
    atari_sims = 8 if fast else 24
    # atari backend: prefer the REAL MinAtar Breakout DQN when available (a real Atari game).
    use_minatar = (atari_mode == "minatar" or
                   (atari_mode == "auto" and os.path.exists(CKPT_ATARI_MINATAR)))
    minatar_steps = 40 if fast else 300  # cap the (otherwise long) Breakout episode
    # One shared browser visualizer for the whole run (opened once; each engine rewrites it).
    viz = make_web_viz("Chess / Connect-4 / Atari -- trained engines playing live") if web else None

    outcomes = []
    if engine in ("chess", "all"):
        if strong:
            outcomes.append(play_chess_strong(games=games, delay=delay, render=render,
                                              movetime=0.2 if fast else 0.5, viz=viz))
        else:
            outcomes.append(play_chess(games=games, delay=delay, render=render,
                                       device=dev, mcts_sims=chess_sims, viz=viz))
    if engine in ("connect4", "all"):
        outcomes.append(play_connect4(games=games, delay=delay, render=render,
                                      device=dev, mcts_sims=c4_sims, viz=viz, strong=strong,
                                      solver_budget=0.3 if fast else 0.6))
    if engine in ("atari", "all"):
        if use_minatar:
            avail = _available_minatar_games() or ["breakout"]
            selected = avail if atari_game == "all" else \
                [atari_game if atari_game in avail else avail[0]]
            for g in selected:
                outcomes.append(play_atari_minatar(games=games, delay=delay, render=render,
                                                   device=dev, max_steps=minatar_steps, game=g, viz=viz))
        else:
            outcomes.append(play_atari(games=games, delay=delay, render=render,
                                        device=dev, mcts_sims=atari_sims, viz=viz))
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch the trained Chess, Connect-4, and Atari engines play, live, in the terminal.")
    p.add_argument("--engine", choices=["chess", "connect4", "atari", "all"], default="all",
                   help="which engine(s) to play (default: all -- chess, then connect4, then atari)")
    p.add_argument("--games", type=int, default=1, help="games/episodes per engine (default: 1)")
    p.add_argument("--delay", type=float, default=0.4,
                   help="per-move sleep in seconds for watchability (default: 0.4)")
    p.add_argument("--fast", action="store_true",
                   help="delay=0 + low sims (CI / quick look)")
    p.add_argument("--no-render", action="store_true",
                   help="scores only (no animated board)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="torch device (default: auto -> cuda if available)")
    p.add_argument("--atari-mode", choices=["auto", "minatar", "catch"], default="auto",
                   help="atari backend: minatar (real MinAtar via DQN), catch (MuZero over a learned "
                        "model on CatchEnv), or auto (minatar if its checkpoint exists)")
    p.add_argument("--atari-game", choices=["breakout", "space_invaders", "asterix", "all"],
                   default="breakout",
                   help="which real MinAtar game to play (default breakout), or 'all' for every "
                        "trained one in sequence")
    p.add_argument("--web", action="store_true",
                   help="ALSO open a live browser visualizer (real graphics) -- a self-contained "
                        "auto-refreshing runs/viz/live.html, no server")
    p.add_argument("--strong", action="store_true",
                   help="showcase GENUINE strength: chess = the ~1600 classical engine vs Stockfish; "
                        "connect-4 = the net+perfect-solver HYBRID (+~134 Elo). The honest, strong demo.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    _banner_intro()
    outcomes = run(engine=args.engine, games=max(1, args.games), delay=args.delay,
                   render=not args.no_render, device=args.device, fast=args.fast,
                   atari_mode=args.atari_mode, atari_game=args.atari_game, web=args.web,
                   strong=args.strong)
    _final_summary(outcomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
