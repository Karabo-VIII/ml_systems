"""
chess_zero.az.minatar_env -- the SCALED-DOWN ATARI environment for single-agent MuZero.

HONEST FRAMING (do not overclaim): full pixel-Atari to human level is COMPUTE-BOUND and out of
local CPU reach. The resourceful, academically-standard scaled-down Atari is MinAtar (Young & Tian
2019, arXiv:1903.03176): a miniaturized Atari suite -- a small multi-channel pixel grid (10x10xC),
single agent, discrete actions, sparse/per-step reward. It captures Atari's CORE challenge (pixel-grid
observation, temporal dynamics, delayed/per-step reward, planning) yet trains on CPU in minutes. That
is exactly the right credible local-scale target. The ceiling we claim is: "scaled-down Atari, an
agent that LEARNS to play clearly better than random" -- NOT human-level real Atari.

This module exposes ONE uniform SINGLE-AGENT env API (deliberately gym-like but dependency-free):
    reset()        -> obs            (np.float32 grid, shape (H, W, C))
    step(action)   -> (obs, reward, done)
    num_actions    : int            (discrete action set the agent chooses among)
    obs_shape      : (H, W, C)
    seed(s)        : reproducible RNG

TWO BACKENDS, auto-selected (prefer MinAtar -- it is the credible standard benchmark):
  (a) MinAtarEnv  : wraps `minatar.Environment`. We use the game's MINIMAL action set (the standard
                    MinAtar practice -- learning over only the playable actions is much faster) and
                    disable sticky actions + difficulty ramping for reproducibility. Default game is
                    Breakout: DENSE reward (a paddle/ball/brick game -> reward most lives), so a tiny
                    model learns above random fastest on a CPU CI budget.
  (b) CatchEnv    : a SELF-CONTAINED, ZERO-DEPENDENCY MinAtar-STYLE fallback (a 10x10x2 pixel grid:
                    one falling ball channel + one paddle channel; 3 actions left/stay/right; +1 on a
                    catch, -1 on a miss). Fully reproducible, pure numpy. Used ONLY if MinAtar import
                    fails -- but it is a perfectly acceptable scaled-Atari env in its own right.

`make_env(prefer_minatar=True, game="breakout")` returns the best available backend and a string tag
naming which one was chosen (so the RWYB output can state the env honestly).

No GPU. No emoji (Windows cp1252). ADDITIVE: this file is new; it touches no existing module.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__contract__ = {
    "kind": "rl-environment",
    "inputs": ["a discrete action (int)"],
    "outputs": [
        "reset() -> obs grid (H,W,C) float32",
        "step(action) -> (obs, reward, done)",
        "num_actions, obs_shape, seed()",
    ],
    "invariants": [
        "SINGLE-AGENT, reward-based (no opponent, no negamax) -- this is RL, not a 2-player game",
        "obs is a multi-channel pixel grid (scaled-Atari), not a flat feature vector",
        "deterministic given a seed (sticky actions + difficulty ramping disabled in MinAtar)",
        "no emoji in any print (Windows cp1252)",
    ],
}


# --------------------------------------------------------------------------- #
# (a) MinAtar backend -- the credible standard scaled-Atari benchmark.
# --------------------------------------------------------------------------- #
class MinAtarEnv:
    """Single-agent wrapper over `minatar.Environment`.

    Uses the game's MINIMAL action set: the agent chooses an index 0..num_actions-1 which maps to a
    real MinAtar action via `minimal_action_set()`. This is standard MinAtar practice and is what
    makes a tiny model learn above random inside a CPU CI budget (Breakout's full set is 6 actions but
    only 3 are meaningful). Sticky actions + difficulty ramping are OFF for reproducibility."""

    def __init__(self, game: str = "breakout", seed: Optional[int] = None):
        from minatar import Environment  # imported lazily so the fallback path needs no minatar
        self._env = Environment(game, sticky_action_prob=0.0, difficulty_ramping=False)
        self.game = game
        # restrict the agent to the meaningful (minimal) action set
        self._action_map = list(self._env.minimal_action_set())
        self.num_actions = len(self._action_map)
        shp = self._env.state_shape()  # (H, W, C)
        self.obs_shape: Tuple[int, int, int] = (int(shp[0]), int(shp[1]), int(shp[2]))
        if seed is not None:
            self.seed(seed)
        self._env.reset()

    def seed(self, s: int) -> None:
        self._env.seed(int(s))

    def reset(self) -> np.ndarray:
        self._env.reset()
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        real_a = self._action_map[int(action)]
        reward, done = self._env.act(real_a)
        return self._obs(), float(reward), bool(done)

    def _obs(self) -> np.ndarray:
        return np.asarray(self._env.state(), dtype=np.float32)


# --------------------------------------------------------------------------- #
# (b) CatchEnv -- self-contained, zero-dependency MinAtar-STYLE fallback.
# A (size,size,2) pixel grid: channel 0 = falling ball, channel 1 = paddle (bottom
# row). 3 actions: 0 = left, 1 = stay, 2 = right. The ball falls one row per step
# from a random top column; +1 if the paddle is under it when it reaches the
# bottom, -1 on a miss.
#
# TWO MODES:
#   multi_drop=True  : CONTINUING -- the ball RESPAWNS after each catch/miss and the
#                      episode runs ep_steps steps, so there are MANY reward events
#                      per episode (a Breakout-like per-event reward structure).
#   multi_drop=False : SINGLE-DROP -- one drop, then the episode ends (terminal). The
#                      reward sits at a FIXED, search-reachable horizon and the value
#                      target is a clean +-1, so credit assignment is fast and the
#                      learns-above-random margin is wide + low-variance. This is the
#                      DEFAULT (chosen for a robust, cheap CPU-CI demonstration).
# Random play scores badly (a random paddle rarely ends under the ball); a
# ball-tracking policy scores ~+1 per drop. Pure numpy, reproducible given a seed.
# --------------------------------------------------------------------------- #
class CatchEnv:
    """Self-contained scaled-Atari 'catch': a (size,size,2) pixel grid, single agent, 3 actions.

    The ZERO-DEPENDENCY fallback if MinAtar is unavailable -- but a legitimate scaled-Atari task in
    its own right: a real pixel-grid, temporal-dynamics, delayed-reward planning problem (steer the
    paddle under a ball it sees fall), exercising the exact same MuZero machinery. Default is the
    single-drop mode (clean +-1 terminal reward at a fixed horizon) -- the cheapest, lowest-variance
    way to demonstrate learning clearly above random on a CPU CI budget."""

    def __init__(self, size: int = 5, seed: Optional[int] = None, ep_steps: int = 20,
                 multi_drop: bool = False):
        self.size = size
        self.num_actions = 3
        self.obs_shape: Tuple[int, int, int] = (size, size, 2)
        self.ep_steps = ep_steps
        self.multi_drop = multi_drop
        self._rng = np.random.RandomState(seed if seed is not None else 0)
        self._ball_row = 0
        self._ball_col = 0
        self._paddle = size // 2
        self._t = 0
        self.reset()

    def seed(self, s: int) -> None:
        self._rng = np.random.RandomState(int(s))

    def _spawn_ball(self) -> None:
        self._ball_row = 0
        self._ball_col = int(self._rng.randint(self.size))

    def reset(self) -> np.ndarray:
        self._spawn_ball()
        # start the paddle at a RANDOM column so the agent must actively STEER (not just sit still
        # and get lucky); this is what forces a real tracking policy and makes random play score badly.
        # The grid is sized so the drop time (size-1 steps) always allows the paddle to reach the ball
        # column from any start -- the task is fully solvable by a tracking policy (greedy -> +1).
        self._paddle = int(self._rng.randint(self.size))
        self._t = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # move paddle (0 left, 1 stay, 2 right), clamped to the grid
        if action == 0:
            self._paddle = max(0, self._paddle - 1)
        elif action == 2:
            self._paddle = min(self.size - 1, self._paddle + 1)
        # ball falls one row
        self._ball_row += 1
        reward = 0.0
        landed = self._ball_row >= self.size - 1
        if landed:  # ball reached the paddle row -> score
            reward = 1.0 if self._paddle == self._ball_col else -1.0
            if self.multi_drop:
                self._spawn_ball()  # respawn (continuing); keep the paddle where it is
        self._t += 1
        if self.multi_drop:
            done = self._t >= self.ep_steps
        else:
            done = landed
        return self._obs(), float(reward), bool(done)

    def _obs(self) -> np.ndarray:
        g = np.zeros((self.size, self.size, 2), dtype=np.float32)
        r = min(self._ball_row, self.size - 1)
        g[r, self._ball_col, 0] = 1.0
        g[self.size - 1, self._paddle, 1] = 1.0
        return g


# --------------------------------------------------------------------------- #
# Factory: prefer MinAtar (the credible standard); fall back to CatchEnv.
# --------------------------------------------------------------------------- #
def make_env(prefer_minatar: bool = True, game: str = "breakout",
             seed: Optional[int] = None) -> Tuple[object, str]:
    """Return (env, backend_tag). Tries MinAtar first (the academic standard scaled-Atari); on any
    import/construction failure, returns the zero-dependency CatchEnv fallback. backend_tag names the
    chosen backend so callers can state the env honestly in their output."""
    if prefer_minatar:
        try:
            env = MinAtarEnv(game=game, seed=seed)
            return env, f"minatar:{game}"
        except Exception as exc:  # minatar missing or broke -> fall back, never crash
            print(f"[minatar_env] MinAtar unavailable ({type(exc).__name__}: {exc}); "
                  f"using self-contained CatchEnv fallback")
    env = CatchEnv(seed=seed)
    return env, "catch_fallback"
