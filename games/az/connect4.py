"""
chess_zero.az.connect4 -- a Connect-4 GameAdapter + self-play training, on the EXISTING AlphaZero
neural search (mcts.NeuralMCTS) and the EXISTING net contract (net.Connect4Net).

THE POINT (and the whole reason this file is short): we do NOT build a new search. Connect-4 is
"solved at local scale" by IMPLEMENTING THE SAME ~7-method GameAdapter contract that TicTacToe and
chess already satisfy (game_adapter.GameAdapter), and feeding it to the proven, game-agnostic PUCT
search NeuralMCTS. The self-play train loop here is modelled 1:1 on _test_neural_adapter._train_ttt_net
(NeuralMCTS self-play -> train policy to MCTS visit-counts + value to game outcome).

Game: 7 columns x 6 rows. A move drops a piece into a column (<= 7 legal actions; a full column is
illegal). Win = 4-in-a-row horizontally, vertically, or on either diagonal. Draw on a full board.
Two-player zero-sum; returns() is PLAYER-0-ABSOLUTE (+1 p0 wins, -1 p1 wins, 0 draw) -- the SAME
convention as TicTacToe, so NeuralMCTS's player-0-absolute -> side-to-move conversion works unchanged.

STATE: ((cells: tuple of 42 in {0=empty, 1=p0, 2=p1}, in ROW-MAJOR order with row 0 = BOTTOM),
         player: 0|1). Row 0 is the bottom of the board so "drop" = fill the lowest empty row of a column.

encode(state) -> (3, 6, 7) float32 planes from the SIDE-TO-MOVE POV (AlphaZero canonical orientation,
so the net is colour-symmetric): plane 0 = 'my' pieces, plane 1 = 'their' pieces, plane 2 = side-to-move
flag (all-ones if p0 to move, else all-zeros). 6 rows x 7 cols.

HONEST CEILING: Connect-4 is a first-player-win SOLVED game (Allis 1988); perfect play is NOT reached
by a small net at a local sim budget, and we do not claim it. The bar this file proves is STRONG-AND-
LEARNED: clearly beats random by a large margin and holds its own / wins vs a 1-ply win-or-block
heuristic. See _test_connect4.py for the RWYB numbers + the locked, non-flaky CI bar.

No emoji (Windows cp1252).
"""
from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple

from .game_adapter import GameAdapter

N_COLS = 7
N_ROWS = 6
N_CELLS = N_ROWS * N_COLS  # 42


def _idx(row: int, col: int) -> int:
    """Row-major flat index; row 0 = bottom row."""
    return row * N_COLS + col


# Pre-computed list of all 4-in-a-row line index-quads (horizontal, vertical, both diagonals).
# Built once at import; pure index arithmetic, no per-call cost.
def _build_win_lines() -> Tuple[Tuple[int, int, int, int], ...]:
    lines: List[Tuple[int, int, int, int]] = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            # horizontal -> (r, c..c+3)
            if c + 3 < N_COLS:
                lines.append((_idx(r, c), _idx(r, c + 1), _idx(r, c + 2), _idx(r, c + 3)))
            # vertical -> (r..r+3, c)
            if r + 3 < N_ROWS:
                lines.append((_idx(r, c), _idx(r + 1, c), _idx(r + 2, c), _idx(r + 3, c)))
            # diagonal up-right -> (r+i, c+i)
            if r + 3 < N_ROWS and c + 3 < N_COLS:
                lines.append((_idx(r, c), _idx(r + 1, c + 1), _idx(r + 2, c + 2), _idx(r + 3, c + 3)))
            # diagonal up-left -> (r+i, c-i)
            if r + 3 < N_ROWS and c - 3 >= 0:
                lines.append((_idx(r, c), _idx(r + 1, c - 1), _idx(r + 2, c - 2), _idx(r + 3, c - 3)))
    return tuple(lines)


_WIN_LINES = _build_win_lines()


class Connect4(GameAdapter):
    """Connect-4 over the engine-agnostic GameAdapter contract. State =
    ((cells: tuple[42] in {0,1,2}, row-major, row0=bottom), player: 0|1)."""

    name = "connect4"
    n_cols = N_COLS
    n_rows = N_ROWS

    @property
    def num_actions(self) -> int:
        return N_COLS  # the policy head width: one logit per column

    def initial_state(self):
        return ((0,) * N_CELLS, 0)

    def current_player(self, state) -> int:
        return state[1]

    def _drop_row(self, cells, col: int) -> int:
        """The lowest empty row in `col`, or -1 if the column is full."""
        for r in range(N_ROWS):
            if cells[_idx(r, col)] == 0:
                return r
        return -1

    def legal_actions(self, state) -> List[int]:
        cells, _ = state
        # a column is legal iff its TOP cell (row N_ROWS-1) is empty
        return [c for c in range(N_COLS) if cells[_idx(N_ROWS - 1, c)] == 0]

    def apply(self, state, action: int):
        cells, player = state
        r = self._drop_row(cells, action)
        if r < 0:
            raise ValueError(f"illegal action {action} (column full) on board {cells}")
        nb = list(cells)
        nb[_idx(r, action)] = player + 1   # 1 for p0, 2 for p1
        return (tuple(nb), 1 - player)

    def _winner(self, cells) -> int:
        """0 = no winner; 1 = p0 has 4-in-a-row; 2 = p1 has 4-in-a-row."""
        for a, b, c, d in _WIN_LINES:
            v = cells[a]
            if v != 0 and cells[b] == v and cells[c] == v and cells[d] == v:
                return v
        return 0

    def is_terminal(self, state) -> bool:
        cells, _ = state
        if self._winner(cells) != 0:
            return True
        return all(v != 0 for v in cells)  # full board -> draw

    def returns(self, state) -> float:
        """PLAYER-0-ABSOLUTE terminal value in [-1, 1] (same convention as TicTacToe)."""
        w = self._winner(state[0])
        if w == 0:
            return 0.0
        return 1.0 if w == 1 else -1.0

    def encode(self, state):
        """state -> (3, 6, 7) float32 planes from the SIDE-TO-MOVE POV.
        plane 0 = my pieces, plane 1 = their pieces, plane 2 = side-to-move flag (1.0 if p0 to move)."""
        import numpy as np
        cells, player = state
        me = player + 1
        them = 2 - player
        planes = np.zeros((3, N_ROWS, N_COLS), dtype=np.float32)
        for i in range(N_CELLS):
            r, c = divmod(i, N_COLS)
            if cells[i] == me:
                planes[0, r, c] = 1.0
            elif cells[i] == them:
                planes[1, r, c] = 1.0
        planes[2, :, :] = 1.0 if player == 0 else 0.0
        return planes

    def render(self, state) -> str:
        cells, _ = state
        sym = {0: ".", 1: "X", 2: "O"}
        # print TOP row first (row N_ROWS-1) down to bottom (row 0), like a real board
        rows = []
        for r in range(N_ROWS - 1, -1, -1):
            rows.append(" ".join(sym[cells[_idx(r, c)]] for c in range(N_COLS)))
        rows.append(" ".join(str(c) for c in range(N_COLS)))  # column labels
        return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Baseline opponents (for the honest strength checks; NOT used in training).
# --------------------------------------------------------------------------- #
def random_action(game: Connect4, state, rng: random.Random) -> int:
    return rng.choice(game.legal_actions(state))


def heuristic_1ply_action(game: Connect4, state, rng: random.Random) -> int:
    """1-ply tactical baseline: (1) if any legal move WINS NOW, play it; (2) else if the opponent
    has an immediate winning move, BLOCK it; (3) else play a random legal move (center-biased by a
    simple preference only as a tie-break among randoms). This is the honest 'did it learn real
    Connect-4 tactics, not just beat a flailer' opponent."""
    legal = game.legal_actions(state)
    me = game.current_player(state)
    # (1) immediate win
    for a in legal:
        ns = game.apply(state, a)
        if game.is_terminal(ns):
            w = game._winner(ns[0])
            # winner is in {1,2}; me in {0,1} -> me wins iff w == me+1
            if w == me + 1:
                return a
    # (2) block opponent's immediate win: simulate giving the opponent each of THEIR moves
    #     from the position AFTER we pass (i.e. check each of our legal columns -- if NOT taking
    #     it lets the opponent win there next, we must take it). Concretely: for each legal col,
    #     check whether the opponent dropping there RIGHT NOW would win; if so, we take that col.
    opp = 1 - me
    for a in legal:
        # hypothetically let the opponent move into column a from the current cells
        opp_state = (state[0], opp)
        ns = game.apply(opp_state, a)
        if game.is_terminal(ns) and game._winner(ns[0]) == opp + 1:
            return a  # block it
    # (3) otherwise random (center-preferring tie-break for a slightly stronger baseline)
    center_order = sorted(legal, key=lambda c: abs(c - N_COLS // 2))
    # keep it stochastic: pick uniformly among the legal moves, but the heuristic's tactical
    # layer above is what makes it a real test; the fallback is intentionally simple.
    return rng.choice(legal) if rng.random() < 0.5 else center_order[0]


# --------------------------------------------------------------------------- #
# SELF-PLAY TRAINING -- modelled on _test_neural_adapter._train_ttt_net.
# NeuralMCTS self-play games -> train policy head to the MCTS visit distribution
# and the value head to the game outcome (from each recorded position's mover POV).
# CPU-friendly. Returns (net, history_of_iter_metrics) so the caller can show a curve.
# --------------------------------------------------------------------------- #
def _selfplay_one_game(game, net, sims, batch_size, opening_plies, np, py_rng, device=None):
    """Play ONE self-play game with NeuralMCTS; return its (planes, pi, mover) samples (pi = the
    normalized MCTS visit distribution -- the improved policy AlphaZero trains the policy head to).
    Early plies are sampled from the visit distribution (temperature 1) for opening diversity; later
    plies are greedy (argmax) -- the standard AlphaZero self-play temperature schedule.

    device: torch device for the net's leaf evals (threaded into NeuralMCTS -> net.predict_many).
            None -> the net's own device (CPU fallback)."""
    from .mcts import NeuralMCTS
    mcts = NeuralMCTS(game, net, n_simulations=sims, device=device)
    state = game.initial_state()
    history = []
    move_num = 0
    while not game.is_terminal(state):
        visits = mcts.run_batched(state, add_noise=True, batch_size=batch_size)
        tot = sum(visits.values())
        if tot <= 0:
            state = game.apply(state, py_rng.choice(game.legal_actions(state)))
            continue
        pi = np.zeros(game.num_actions, dtype=np.float32)
        for a, n in visits.items():
            pi[a] = n / tot
        history.append((game.encode(state), pi, game.current_player(state)))
        actions = list(visits.keys())
        counts = np.array([visits[a] for a in actions], dtype=np.float64)
        if move_num < opening_plies:
            probs = counts / counts.sum()
            a = actions[int(np.random.choice(len(actions), p=probs))]
        else:
            a = actions[int(counts.argmax())]
        state = game.apply(state, a)
        move_num += 1
    if not history:
        return []
    z_p0 = game.returns(state)  # player-0 absolute outcome
    # value target = outcome from each recorded position's MOVER perspective
    return [(h[0], h[1], (z_p0 if h[2] == 0 else -z_p0)) for h in history]


# --------------------------------------------------------------------------- #
# GPU-EXPLOITING PARALLEL SELF-PLAY: play N games in LOCKSTEP and batch the
# per-simulation leaf evals across ALL games into ONE net.predict_many forward.
#
# WHY: the single-game run_batched only pools the <=7-action tree of ONE game, so
# even at batch_size=256 it can only fill a handful of leaves -> the GPU sits idle
# (measured: GPU per-forward latency ~2.5ms is ~constant in batch size, so a 32ch
# conv at batch 16 is wildly underused; at batch 256 it does 17x the work for the
# same wall-cost). Pooling N concurrent games' leaves into one forward is the
# standard leaf-parallel-across-games trick (mirrors batched_selfplay.py for chess)
# and is what actually turns the 4060 into throughput. It is engine-agnostic: it
# uses ONLY the GameAdapter contract (encode / legal_policy_mask / apply /
# is_terminal / returns / current_player) + net.predict_many. Correctness is the
# SAME PUCT as NeuralMCTS (identical Q + c*P*sqrt(sumN)/(1+N), per-ply value
# negation, Dirichlet root); the only change is the ORDER leaves are gathered.
# --------------------------------------------------------------------------- #
class _PNode:
    __slots__ = ("prior", "to_play", "children", "visit_count", "value_sum", "is_expanded")

    def __init__(self, prior, to_play):
        self.prior = prior
        self.to_play = to_play
        self.children = {}              # action -> _PNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def _p_select_child(node, c_puct):
    sqrt_total = (max(1, node.visit_count)) ** 0.5
    best_score, best_a, best_c = -float("inf"), None, None
    for a, ch in node.children.items():
        q = -ch.value()                # child value is child-POV -> negate
        u = c_puct * ch.prior * sqrt_total / (1 + ch.visit_count)
        s = q + u
        if s > best_score:
            best_score, best_a, best_c = s, a, ch
    return best_a, best_c


def selfplay_games_parallel(game, net, n_games, sims, opening_plies, device=None,
                            c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25,
                            np=None, seed=0):
    """Play n_games self-play games in LOCKSTEP, batching all games' per-sim leaf evals into ONE
    net.predict_many. Returns a flat list of (planes, pi, mover-POV outcome) training samples (the
    SAME sample contract as _selfplay_one_game). Engine-agnostic over the GameAdapter contract."""
    import numpy as _np
    if np is None:
        np = _np
    rng = _np.random.default_rng(seed)

    class _G:
        __slots__ = ("state", "root", "history", "move_num", "done")

        def __init__(self):
            self.state = game.initial_state()
            self.root = None
            self.history = []          # (planes, pi, mover)
            self.move_num = 0
            self.done = False

    games = [_G() for _ in range(n_games)]

    def _batched_eval(items):
        """items: list of (game_idx, state). ONE forward over all. Returns {gi: (priors, value)}."""
        if not items:
            return {}
        planes_list, masks_list, metas = [], [], []
        for gi, st in items:
            mask, idx_to_action = game.legal_policy_mask(st)
            planes_list.append(game.encode(st))
            masks_list.append(mask)
            metas.append((gi, idx_to_action))
        out = net.predict_many(planes_list, masks_list, device=device)
        res = {}
        for (gi, idx_to_action), (probs, value) in zip(metas, out):
            priors = {idx_to_action[idx]: float(probs[idx]) for idx in idx_to_action}
            tot = sum(priors.values())
            if tot > 0:
                priors = {a: p / tot for a, p in priors.items()}
            else:
                n = max(1, len(idx_to_action))
                priors = {a: 1.0 / n for a in idx_to_action}
            res[gi] = (priors, float(value))
        return res

    def _terminal_stm(st):
        if not game.is_terminal(st):
            return None
        z_p0 = game.returns(st)
        return z_p0 if game.current_player(st) == 0 else -z_p0

    while not all(g.done for g in games):
        active = [i for i, g in enumerate(games) if not g.done]
        if not active:
            break

        # 1) fresh root per active game; EXPAND all roots in ONE batch
        for i in active:
            games[i].root = _PNode(prior=1.0, to_play=game.current_player(games[i].state))
        root_eval = _batched_eval([(i, games[i].state) for i in active])
        for i in active:
            priors, _value = root_eval[i]
            root = games[i].root
            ctp = 1 - root.to_play
            for a, p in priors.items():
                root.children[a] = _PNode(prior=p, to_play=ctp)
            root.is_expanded = True
            # Dirichlet root noise (exploration) on every self-play root
            if root.children:
                acts = list(root.children.keys())
                noise = rng.dirichlet([dirichlet_alpha] * len(acts))
                for a, nz in zip(acts, noise):
                    ch = root.children[a]
                    ch.prior = (1 - dirichlet_eps) * ch.prior + dirichlet_eps * float(nz)

        # 2) run `sims` simulations; each sim: SELECT one leaf per active game, batch-EVAL, backup
        for _ in range(sims):
            leaves = []      # [gi, leaf_node, leaf_state, path, term_or_None]
            to_eval = []     # (gi, leaf_state) for non-terminal leaves
            for i in active:
                node = games[i].root
                st = games[i].state
                path = [node]
                while node.is_expanded and node.children:
                    a, ch = _p_select_child(node, c_puct)
                    if a is None:
                        break
                    st = game.apply(st, a)
                    node = ch
                    path.append(node)
                term = _terminal_stm(st)
                leaves.append([i, node, st, path, term])
                if term is None:
                    to_eval.append((i, st))
            ev = _batched_eval(to_eval)   # a game contributes <=1 leaf/sim -> gi unique
            for i, node, st, path, term in leaves:
                if term is not None:
                    value = term
                else:
                    priors, value = ev[i]
                    if not node.is_expanded:
                        ctp = 1 - node.to_play
                        for a, p in priors.items():
                            node.children[a] = _PNode(prior=p, to_play=ctp)
                        node.is_expanded = True
                # backup, negating per ply (side-to-move relative)
                v = value
                for nd in reversed(path):
                    nd.visit_count += 1
                    nd.value_sum += v
                    v = -v

        # 3) each active game: record sample, pick + play a move, advance/terminate
        for i in active:
            g = games[i]
            visits = {a: ch.visit_count for a, ch in g.root.children.items()}
            tot = sum(visits.values())
            if tot <= 0:
                g.state = game.apply(g.state, rng.choice(game.legal_actions(g.state)))
            else:
                pi = np.zeros(game.num_actions, dtype=np.float32)
                for a, n in visits.items():
                    pi[a] = n / tot
                g.history.append((game.encode(g.state), pi, game.current_player(g.state)))
                actions = list(visits.keys())
                counts = np.array([visits[a] for a in actions], dtype=np.float64)
                if g.move_num < opening_plies:
                    probs = counts / counts.sum()
                    a = actions[int(rng.choice(len(actions), p=probs))]
                else:
                    a = actions[int(counts.argmax())]
                g.state = game.apply(g.state, a)
                g.move_num += 1
            if game.is_terminal(g.state):
                g.done = True

    # assign z (player-0 absolute) per game; sample value = mover-POV outcome
    samples = []
    for g in games:
        if not g.history:
            continue
        z_p0 = game.returns(g.state)
        for (planes, pi, mover) in g.history:
            samples.append((planes, pi, (z_p0 if mover == 0 else -z_p0)))
    return samples


def train_connect4(net, n_iters: int = 6, games_per_iter: int = 40, sims: int = 50,
                   lr: float = 1e-2, weight_decay: float = 1e-4, seed: int = 0,
                   eval_games: int = 0, batch_size: int = 16, opening_plies: int = 8,
                   train_epochs: int = 4, train_minibatch: int = 64, buffer_iters: int = 3,
                   verbose: bool = True, device=None, eval_sims: Optional[int] = None,
                   parallel_games: int = 0):
    """Self-play train `net` on Connect-4 with NeuralMCTS -- the standard AlphaZero loop:
    self-play games -> append (planes, MCTS-visit-policy, outcome) samples to a SLIDING REPLAY
    BUFFER -> train the net for several minibatch epochs over the buffer. The replay buffer +
    multi-epoch minibatch training (vs the naive one-step-per-game) is what lets a small net learn
    actual TACTICS (win/block patterns) under the CI compute budget, not just beat random flailers.

    Loss = cross-entropy(policy, MCTS visits) + MSE(value, mover-POV outcome) -- exactly AlphaZero's.

    SPEED: search uses NeuralMCTS.run_batched (the EXISTING tree-parallel virtual-loss path) which
    batches the per-simulation leaf net-evals into one forward -- several x faster than sequential
    run() on CPU at batch_size=16; that headroom is what buys the harder training inside 240s. The
    converged visit-count policy is unchanged; virtual loss only reorders how leaves are gathered.

    buffer_iters: how many recent iterations of samples the replay buffer retains (a sliding window;
                  the newest, strongest-net games dominate while older games regularize).

    Returns (net, metrics): a per-iteration list of
      {iter, avg_loss, avg_ploss, avg_vloss, n_samples, buffer_size, [winrate_vs_random]}.
    All RNG (random, numpy, torch) is seeded so the CURVE SHAPE reproduces; torch CPU training is
    not bit-exact run-to-run, so the CI test locks a MARGIN, not exact numbers (the TTT lesson)."""
    import numpy as np
    import torch
    from collections import deque

    # --- device resolution (GPU when available; CPU fallback) ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    net.to(device)                      # net params + buffers live on `device`
    if eval_sims is None:
        eval_sims = sims

    game = Connect4()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    torch.manual_seed(seed)
    np.random.seed(seed)
    py_rng = random.Random(seed)

    # sliding replay buffer of per-iteration sample lists
    iter_samples: "deque" = deque(maxlen=max(1, buffer_iters))

    metrics = []
    for it in range(n_iters):
        # --- self-play: generate this iteration's games ---
        new_samples = []
        if parallel_games and parallel_games > 1:
            # GPU-exploiting path: play games in lockstep batches, pooling leaf evals across games.
            remaining = games_per_iter
            while remaining > 0:
                nb = min(parallel_games, remaining)
                new_samples.extend(
                    selfplay_games_parallel(game, net, nb, sims, opening_plies, device=device,
                                            np=np, seed=seed * 100003 + it * 9973 + remaining)
                )
                remaining -= nb
        else:
            for _ in range(games_per_iter):
                new_samples.extend(
                    _selfplay_one_game(game, net, sims, batch_size, opening_plies, np, py_rng,
                                       device=device)
                )
        iter_samples.append(new_samples)
        buffer = [s for chunk in iter_samples for s in chunk]

        # --- train: several minibatch epochs over the replay buffer ---
        # Whole replay buffer pinned on `device` once per iter (small: a few k positions x 3x6x7).
        planes_all = torch.as_tensor(np.asarray([s[0] for s in buffer]), dtype=torch.float32,
                                     device=device)
        pis_all = torch.as_tensor(np.asarray([s[1] for s in buffer]), dtype=torch.float32,
                                  device=device)
        zs_all = torch.as_tensor(np.asarray([s[2] for s in buffer]), dtype=torch.float32,
                                 device=device)
        n = planes_all.shape[0]
        losses, plosses, vlosses = [], [], []
        net.train()
        for _ep in range(train_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, train_minibatch):
                idx = perm[start:start + train_minibatch]
                opt.zero_grad()
                logits, value = net.forward(planes_all[idx])
                logp = torch.log_softmax(logits, dim=-1)
                ploss = -(pis_all[idx] * logp).sum(dim=-1).mean()
                vloss = ((value.squeeze(-1) - zs_all[idx]) ** 2).mean()
                loss = ploss + vloss
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                plosses.append(float(ploss.item()))
                vlosses.append(float(vloss.item()))

        rec = {
            "iter": it,
            "avg_loss": float(np.mean(losses)) if losses else float("nan"),
            "avg_ploss": float(np.mean(plosses)) if plosses else float("nan"),
            "avg_vloss": float(np.mean(vlosses)) if vlosses else float("nan"),
            "n_samples": len(new_samples),
            "buffer_size": n,
        }
        if eval_games > 0:
            wr = eval_vs_random(net, n_games=eval_games, sims=eval_sims, seed=1000 + it,
                                device=device)
            rec["winrate_vs_random"] = wr
        metrics.append(rec)
        if verbose:
            extra = f"  winrate_vs_random={rec.get('winrate_vs_random'):.3f}" if eval_games > 0 else ""
            print(f"[connect4] iter {it}: avg_loss={rec['avg_loss']:.4f} "
                  f"(p={rec['avg_ploss']:.4f} v={rec['avg_vloss']:.4f})  "
                  f"new_samples={len(new_samples)} buffer={n}{extra}")
    return net, metrics


# --------------------------------------------------------------------------- #
# Strength evaluation helpers (honest W/D/L; legality asserted every move).
# --------------------------------------------------------------------------- #
def _play_one(game: Connect4, net, sims: int, neural_is_p0: bool,
              opponent_fn, rng: random.Random, batch_size: int = 16, device=None) -> float:
    """One full game: NeuralMCTS (greedy) vs an opponent_fn(game, state, rng). Returns the
    player-0-absolute outcome. Asserts every NEURAL move is legal. Uses the batched search
    (best_action_batched) for speed -- same argmax-over-visits selection as best_action.
    device threads into NeuralMCTS (-> net.predict_many) so eval runs on the same device as train."""
    from .mcts import NeuralMCTS
    s = game.initial_state()
    while not game.is_terminal(s):
        p = game.current_player(s)
        neural_to_move = (p == 0) == neural_is_p0
        if neural_to_move:
            a = NeuralMCTS(game, net, n_simulations=sims, device=device).best_action_batched(
                s, temperature=0.0, batch_size=batch_size)
            assert a in game.legal_actions(s), (
                f"NEURAL search returned ILLEGAL action {a} at\n{game.render(s)}"
            )
        else:
            a = opponent_fn(game, s, rng)
        s = game.apply(s, a)
    return game.returns(s)


def _wdl_vs(net, opponent_fn, n_games: int, sims: int, seed: int, device=None) -> Tuple[int, int, int]:
    """Play n_games (neural alternates colour) vs opponent_fn. Returns (W, D, L) from the
    NEURAL agent's perspective. Seeds python RNG for stable eval. device threads into the search."""
    game = Connect4()
    rng = random.Random(seed)
    w = d = l = 0
    for gi in range(n_games):
        neural_is_p0 = (gi % 2 == 0)
        z = _play_one(game, net, sims, neural_is_p0, opponent_fn, rng, device=device)
        nz = z if neural_is_p0 else -z
        if nz > 0:
            w += 1
        elif nz == 0:
            d += 1
        else:
            l += 1
    return w, d, l


def eval_vs_random(net, n_games: int = 24, sims: int = 50, seed: int = 0, device=None) -> float:
    """Non-loss rate (= (W + D) / N) vs a random opponent -- the learning-curve scalar."""
    w, d, l = _wdl_vs(net, lambda g, s, r: random_action(g, s, r), n_games, sims, seed, device=device)
    return (w + d) / max(1, n_games)


def eval_wdl_vs_random(net, n_games: int = 24, sims: int = 50, seed: int = 0,
                       device=None) -> Tuple[int, int, int]:
    return _wdl_vs(net, lambda g, s, r: random_action(g, s, r), n_games, sims, seed, device=device)


def eval_wdl_vs_heuristic(net, n_games: int = 24, sims: int = 50, seed: int = 0,
                          device=None) -> Tuple[int, int, int]:
    return _wdl_vs(net, lambda g, s, r: heuristic_1ply_action(g, s, r), n_games, sims, seed,
                   device=device)


if __name__ == "__main__":
    # Quick manual RWYB driver (NOT the CI test -- that is _test_connect4.py).
    from .net import Connect4Net
    print("=" * 72)
    print("  Connect-4 on the EXISTING AlphaZero NeuralMCTS -- self-play train + strength")
    print("=" * 72)
    net = Connect4Net()
    net, metrics = train_connect4(net, n_iters=6, games_per_iter=40, sims=50,
                                  seed=0, eval_games=20)
    w, d, l = eval_wdl_vs_random(net, n_games=24, sims=50, seed=0)
    print(f"[connect4] vs RANDOM   (24 games, 50 sims): W{w} D{d} L{l}")
    w2, d2, l2 = eval_wdl_vs_heuristic(net, n_games=24, sims=50, seed=0)
    print(f"[connect4] vs HEURISTIC(24 games, 50 sims): W{w2} D{d2} L{l2}")
