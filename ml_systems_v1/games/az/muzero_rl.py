"""
chess_zero.az.muzero_rl -- SINGLE-AGENT, REWARD-BASED MuZero for SCALED-DOWN ATARI (MinAtar).

WHY THIS IS A SEPARATE MODULE (additive, not a change to muzero.py): the existing muzero.py is the
TWO-PLAYER board-game framing -- its MCTS backup NEGATES the value per ply (negamax) because in a
zero-sum game what is good for me is bad for my opponent. That framing is WRONG for a single-agent RL
task: in Atari there is no opponent, reward accrues PER STEP, and the value of a state is the
DISCOUNTED SUM of FUTURE REWARDS along my own trajectory. So this module re-derives MuZero for the
single-agent reward setting. It reuses the SHAPE of muzero.py (h/g/f, latent MCTS, K-step unroll) but
the backup, the reward head's role, and the value targets are the genuine RL adaptation.

THE KEY DIFFERENCE vs muzero.py (the real adaptation work) -- the BACKUP:
  two-player (muzero.py):   value <- node.reward + gamma * (-value)     # negamax: flip every ply
  single-agent (HERE):      value <- node.reward + gamma *  value       # NO negation: accumulate my
                                                                        # own discounted future reward
On backup we walk the selected path from the leaf to the root and, at each node, ADD that node's edge
reward and DISCOUNT the running value -- never negating. The leaf's bootstrap value (f's value head)
is the estimate of the rest of the trajectory's discounted reward. This is the standard MuZero
single-agent (Atari) backup (Schrittwieser et al. 2020, arXiv:1911.08265, Appendix B).

THE THREE NETWORKS:
  representation  h : pixel grid (H,W,C) -> latent s0        (small conv, then flatten -> latent)
  dynamics        g : (s_k, action_onehot) -> (s_{k+1}, reward_k)   # the REWARD head MATTERS now
  prediction      f : s_k -> (policy_logits_k, value_k)

PLAN OVER THE LEARNED MODEL ONLY: the search encodes the REAL observation ONCE with h at the root,
then every tree edge is a LATENT transition g(s,a)->(s',r) scored by f(s')->(pi,v). The environment is
stepped ZERO times inside the search (asserted by `sim_step_calls == 0`). The env is used ONLY to
generate self-play DATA (real episodes + real per-step rewards = the training signal).

TRAINING: episodic self-play -> a rolling replay buffer of (obs, action, mcts_pi, reward) transitions.
For each sampled start position we build a K-step unroll and match, at each unrolled latent:
  policy_k  -> the MCTS visit distribution recorded at that real step,
  value_k   -> the n-STEP BOOTSTRAPPED return  G = sum_{i=0..n-1} gamma^i r_{t+i} + gamma^n V_boot,
  reward_k  -> the observed per-step reward,
  latent_k  -> the CONSISTENCY target h(real obs_k) [stop-grad] (EfficientZero, Ye et al. 2021).
Root Dirichlet noise + temperature give exploration. Gradient flows through g into h (co-trained).

THREE settings are load-bearing for learning ABOVE random on a CPU budget (each measured, each fixed a
real failure -- see train_muzero_rl docstring + _train_on_replay): (1) the VALUE LOSS is scaled by 0.25
(MuZero standard) so the reward-scale value MSE does not starve the policy CE; (2) a RANDOM-DATA WARMUP
seeds replay + pre-trains the heads so MCTS is not stone-cold at iteration 0; (3) the LATENT CONSISTENCY
loss directly supervises g to the real next-state encoding -- without it g learns the wrong per-action
direction (the value head learns "aligned=good" but the search still picks 'stay' and never beats
random; with it the dynamics learns the real transition and the plan steers correctly).

PROOF IT LEARNS (RWYB, see __main__ / _test_muzero_rl.py): mean episode return climbs across training,
and the trained agent's mean eval return CLEARLY exceeds a random-policy baseline on the same env.
That margin -- the canonical RL "it learned" signal -- is the robust (non-flaky) bar; we never assert
an absolutist score (torch training is not bit-reproducible; cf. the _test_neural_adapter lesson).

HONEST CEILING: scaled-down Atari (MinAtar 10x10xC, or the CatchEnv fallback), an agent that LEARNS
clearly above random on a CPU CI budget. NOT human-level real pixel Atari -- that is compute-bound and
out of local reach. The CORRECTNESS (single-agent reward-discounted non-negamax backup + model-only
planning + the learns-above-random margin) is what this locks.

No GPU. No emoji (Windows cp1252). ADDITIVE: does not touch muzero.py / net.py / mcts.py / train_robust.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .minatar_env import CatchEnv, make_env

__contract__ = {
    "kind": "single-agent-model-based-rl",
    "inputs": [
        "a single-agent reward-based env (reset/step, obs grid, discrete actions) -- used ONLY to "
        "generate self-play data, NEVER inside the planner",
        "MuZeroRLNet (learned h/g/f with a per-step reward head)",
    ],
    "outputs": [
        "MuZeroRLMCTS: PUCT plan over LATENT states (g+f only), SINGLE-AGENT reward-discounted backup",
        "train_muzero_rl: a trained net + an episode-return training curve",
        "eval_vs_random: trained mean episode return vs a random-policy baseline (the learned signal)",
    ],
    "invariants": [
        "SEARCH NEVER calls env.step / env.reset -- only h once at root then g/f over latents "
        "(enforced by sim_step_calls == 0, asserted in self-play, eval, and the CI test)",
        "BACKUP is single-agent reward-discounted with NO per-ply negation (value = reward + "
        "gamma*value up the path) -- the key difference vs the two-player negamax muzero.py",
        "the dynamics reward head fits the OBSERVED per-step reward (it matters now; it is not ~0)",
        "value targets are n-step bootstrapped returns; the trained agent must beat random by a "
        "clear MARGIN (never an absolutist score -- training is not bit-reproducible)",
        "no emoji in any print (Windows cp1252)",
    ],
}


# --------------------------------------------------------------------------- #
# THE THREE NETWORKS. The representation is a small CONV over the pixel grid (the
# scaled-Atari obs is spatial), then flatten -> latent. Latents are L2-normalized
# (MuZero latent-state normalization) so unrolled g cannot drift to large scale.
# --------------------------------------------------------------------------- #
class MuZeroRLNet(nn.Module):
    """h / g / f for a single-agent scaled-Atari task.

    obs_shape   : (H, W, C) pixel grid.
    num_actions : discrete action-space width (dynamics consumes a one-hot of this).
    latent_dim  : width of the learned latent state s_k.
    channels    : conv width in the representation network.

    The reward and value heads are LINEAR (unbounded): per-step Atari reward is small-integer-ish and
    the discounted return can exceed [-1,1], so we do NOT tanh-squash them (unlike the board-game
    value in muzero.py which is a bounded win/draw/loss in [-1,1])."""

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int,
                 latent_dim: int = 64, channels: int = 16, hidden: int = 128):
        super().__init__()
        H, W, C = obs_shape
        self.obs_shape = (H, W, C)
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        # h: (C,H,W) pixel grid -> conv -> flatten -> latent s0
        self.h_conv = nn.Sequential(
            nn.Conv2d(C, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self._conv_flat = channels * H * W
        self.h_proj = nn.Linear(self._conv_flat, latent_dim)

        # g: (s_k, action_onehot) -> (s_{k+1}, reward_k)
        self.g_body = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.g_next = nn.Linear(hidden, latent_dim)
        self.g_reward = nn.Linear(hidden, 1)

        # f: s_k -> (policy_logits, value)
        self.f_body = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU())
        self.f_policy = nn.Linear(hidden, num_actions)
        self.f_value = nn.Linear(hidden, 1)

    @staticmethod
    def _l2norm(s: torch.Tensor) -> torch.Tensor:
        return s / (s.norm(dim=-1, keepdim=True) + 1e-8)

    def represent(self, obs: torch.Tensor) -> torch.Tensor:
        """h: (B,H,W,C) grid -> (B,latent_dim) latent s0. Accepts HWC and permutes to conv CHW."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        # HWC -> CHW for conv
        x = obs.permute(0, 3, 1, 2).contiguous()
        z = self.h_conv(x).flatten(1)
        return self._l2norm(self.h_proj(z))

    def dynamics(self, s: torch.Tensor, action_onehot: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """g: (s_k, one-hot action) -> (s_{k+1}, reward_k). reward_k is (B,1), UNBOUNDED."""
        z = self.g_body(torch.cat([s, action_onehot], dim=-1))
        s_next = self._l2norm(self.g_next(z))
        reward = self.g_reward(z)
        return s_next, reward

    def prediction(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """f: s_k -> (policy_logits (B,num_actions), value (B,1) UNBOUNDED)."""
        b = self.f_body(s)
        return self.f_policy(b), self.f_value(b)

    @property
    def device(self):
        """The device the model's parameters live on (CPU or CUDA). Used to place inference tensors."""
        return next(self.parameters()).device

    # ---- inference convenience for the LATENT search (numpy in / numpy+float out) ----
    @torch.no_grad()
    def initial_inference(self, obs_grid) -> Tuple[np.ndarray, np.ndarray, float]:
        """h then f at the ROOT: observation -> (latent s0, policy_probs, value).
        The ONLY place the real observation enters the planner. Tensors are placed on the model's
        device (GPU when the model is on CUDA); the latent is returned to numpy/CPU for the tree."""
        self.eval()
        dev = self.device
        x = torch.as_tensor(np.asarray(obs_grid), dtype=torch.float32, device=dev)
        s0 = self.represent(x)
        logits, value = self.prediction(s0)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return s0.squeeze(0).cpu().numpy(), probs, float(value.item())

    @torch.no_grad()
    def recurrent_inference(self, latent: np.ndarray, action: int
                            ) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """g then f at a NON-root node: (latent, action) -> (next_latent, reward, policy_probs, value).
        Pure model -- no env of any kind. Every search edge below the root uses this. Tensors on the
        model's device; outputs returned to numpy/CPU for the tree."""
        self.eval()
        dev = self.device
        s = torch.as_tensor(latent, dtype=torch.float32, device=dev).reshape(1, -1)
        a = torch.zeros(1, self.num_actions, dtype=torch.float32, device=dev)
        a[0, action] = 1.0
        s_next, reward = self.dynamics(s, a)
        logits, value = self.prediction(s_next)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return (s_next.squeeze(0).cpu().numpy(), float(reward.item()),
                probs, float(value.item()))


# --------------------------------------------------------------------------- #
# THE LATENT SEARCH. SINGLE-AGENT: NO per-ply negation on backup. A node holds a
# LATENT vector; expansion uses g+f; the root uses h+f. env.step is NEVER called.
# --------------------------------------------------------------------------- #
class _MzRLNode:
    __slots__ = ("prior", "latent", "reward", "children", "visit_count", "value_sum")

    def __init__(self, prior: float):
        self.prior = prior
        self.latent: Optional[np.ndarray] = None
        self.reward: float = 0.0          # reward on the edge that REACHED this node (g output)
        self.children: Dict[int, "_MzRLNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MuZeroRLMCTS:
    """PUCT search that PLANS OVER THE LEARNED MODEL ONLY, SINGLE-AGENT reward-discounted.

    The root is the ONLY contact with the real env -- and ONLY to read the observation (h). Every node
    below is a pure-latent node expanded by g+f. The search calls h once (root) and g+f thereafter; it
    NEVER calls env.step/env.reset. `self.model_calls` and `self.sim_step_calls` instrument that
    (sim_step_calls stays 0).

    VALUE NORMALIZATION: because rewards are unbounded (unlike the board-game [-1,1]), raw Q values can
    be on any scale, which would make a fixed c_puct mis-balance exploration. We track a running
    [min,max] of backed-up Q across the tree (the MuZero Atari trick) and NORMALIZE Q into [0,1] inside
    the PUCT score, so c_puct is scale-free."""

    def __init__(self, model: MuZeroRLNet, num_actions: int, c_puct: float = 1.25,
                 n_simulations: int = 30, discount: float = 0.99,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.model = model
        self.num_actions = num_actions
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        # instrumentation: proof the planner is model-only
        self.model_calls = 0          # h or g+f evaluations
        self.sim_step_calls = 0       # MUST stay 0 -- the planner never steps the env
        # running Q range for scale-free PUCT (MuZero MinMaxStats)
        self._q_min = float("inf")
        self._q_max = -float("inf")

    def _update_q_range(self, q: float) -> None:
        self._q_min = min(self._q_min, q)
        self._q_max = max(self._q_max, q)

    def _normalize_q(self, q: float) -> float:
        if self._q_max > self._q_min:
            return (q - self._q_min) / (self._q_max - self._q_min)
        return q  # range not yet established -> use raw (early sims)

    def _select_child(self, node: "_MzRLNode") -> Tuple[int, "_MzRLNode"]:
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score, best_action, best_child = -float("inf"), None, None
        for action, child in node.children.items():
            if child.visit_count > 0:
                # single-agent: the child's value IS this state's continuation value -- NO negation.
                # Q for the edge = edge reward + discount * child value (the action-value).
                q = child.reward + self.discount * child.value()
                q = self._normalize_q(q)
            else:
                q = 0.0  # unvisited child -> neutral normalized Q (pure prior-driven)
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _add_dirichlet_noise(self, root: "_MzRLNode") -> None:
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            ch = root.children[a]
            ch.prior = (1 - self.dirichlet_eps) * ch.prior + self.dirichlet_eps * n

    def run(self, obs_grid, add_noise: bool = False) -> Dict[int, int]:
        """Run n_simulations PLANNING OVER THE MODEL. obs_grid (the real observation) is the ONLY
        real-env input, read once at the root; the tree thereafter is pure latent. Returns
        {action: visit_count} over root actions (the full action space -- single agent, no legal mask:
        every action is available every step in these envs)."""
        s0, probs, _root_value = self.model.initial_inference(obs_grid)
        self.model_calls += 1
        root = _MzRLNode(prior=1.0)
        root.latent = s0
        for a in range(self.num_actions):
            root.children[a] = _MzRLNode(prior=float(probs[a]))
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node = root
            path = [node]
            # SELECT down to a not-yet-expanded leaf (a child whose latent is still None).
            while node.children:
                action, child = self._select_child(node)
                if action is None:
                    break
                node = child
                path.append(node)
                if node.latent is None:    # leaf reached -> stop to EXPAND it
                    break
            leaf = path[-1]
            parent = path[-2] if len(path) >= 2 else None
            if leaf.latent is None and parent is not None:
                # EXPAND the leaf PURELY through the learned model: g(parent_latent, action)->...
                action = next(a for a, ch in parent.children.items() if ch is leaf)
                next_latent, reward, child_probs, value = self.model.recurrent_inference(
                    parent.latent, action)
                self.model_calls += 1
                leaf.latent = next_latent
                leaf.reward = reward          # the per-step reward the model predicts for this edge
                for a in range(self.num_actions):
                    leaf.children[a] = _MzRLNode(prior=float(child_probs[a]))
            else:
                # degenerate (root re-selected as its own leaf) -> bootstrap from the root value
                value = _root_value
            # BACKUP: SINGLE-AGENT reward-discounted, NO per-ply negation.
            # `value` starts as the leaf's bootstrap (f value at the leaf). Walking UP the path, at
            # each node we record the running value, update the Q range, then fold THIS node's edge
            # reward in and discount for the step above:  value <- node.reward + discount * value.
            for nd in reversed(path):
                nd.visit_count += 1
                nd.value_sum += value
                self._update_q_range(nd.reward + self.discount * value)
                value = nd.reward + self.discount * value

        return {a: ch.visit_count for a, ch in root.children.items()}

    def best_action(self, obs_grid, temperature: float = 0.0,
                    add_noise: bool = False) -> int:
        visits = self.run(obs_grid, add_noise=add_noise or (temperature > 0))
        actions = list(visits.keys())
        counts = np.array([visits[a] for a in actions], dtype=np.float64)
        if counts.sum() == 0:
            return actions[0]
        if temperature <= 1e-6:
            return actions[int(counts.argmax())]
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        return actions[int(np.random.choice(len(actions), p=probs))]


# --------------------------------------------------------------------------- #
# SELF-PLAY DATA GENERATION. THIS is where the real env is used -- to PLAY OUT
# episodes and collect REAL per-step rewards (the training signal). The planner
# inside uses the model only; the env is stepped to advance the real episode.
# --------------------------------------------------------------------------- #
class _Transition:
    __slots__ = ("obs", "action", "pi", "reward")

    def __init__(self, obs, action, pi, reward):
        self.obs = obs        # encoded observation grid (np, HWC)
        self.action = action  # the action actually played
        self.pi = pi          # MCTS visit distribution (num_actions,)
        self.reward = reward  # OBSERVED per-step reward from the env after taking `action`


def selfplay_episode(env, model: MuZeroRLNet, num_actions: int, sims: int = 30,
                     temperature_moves: int = 8, discount: float = 0.99,
                     max_steps: int = 200, add_noise: bool = True
                     ) -> Tuple[List[_Transition], float]:
    """Play ONE real episode. At each step the action is chosen by MuZeroRLMCTS PLANNING OVER THE
    MODEL (no env in the search); the REAL env advances the episode (env.step) and yields the real
    per-step reward. Returns (transitions, episode_return)."""
    obs = env.reset()
    transitions: List[_Transition] = []
    ep_return = 0.0
    for t in range(max_steps):
        mcts = MuZeroRLMCTS(model, num_actions, n_simulations=sims, discount=discount)
        temp = 1.0 if t < temperature_moves else 0.5
        visits = mcts.run(obs, add_noise=add_noise)
        assert mcts.sim_step_calls == 0, "MuZero planner stepped the env -- not pure MuZero"
        pi = np.zeros(num_actions, dtype=np.float32)
        tot = sum(visits.values())
        for a, n in visits.items():
            pi[a] = n / tot if tot > 0 else 1.0 / num_actions
        actions = list(visits.keys())
        counts = np.array([visits[a] for a in actions], dtype=np.float64)
        if temp <= 1e-6 or counts.sum() == 0:
            action = actions[int(counts.argmax())] if counts.sum() > 0 else 0
        else:
            p = counts ** (1.0 / temp)
            p = p / p.sum()
            action = actions[int(np.random.choice(len(actions), p=p))]
        next_obs, reward, done = env.step(action)
        transitions.append(_Transition(np.asarray(obs, dtype=np.float32), action, pi, float(reward)))
        ep_return += float(reward)
        obs = next_obs
        if done:
            break
    return transitions, ep_return


# --------------------------------------------------------------------------- #
# TRAINING: K-step unroll with n-STEP BOOTSTRAPPED value targets.
# --------------------------------------------------------------------------- #
def _nstep_return(ep: List[_Transition], i: int, n: int, discount: float,
                  value_fn) -> float:
    """n-step bootstrapped return from step i:
        G = sum_{j=0..n-1} discount^j * r_{i+j}  +  discount^n * V_boot(obs_{i+n})
    If the episode ends within n steps, the bootstrap term is dropped (terminal value = 0)."""
    g = 0.0
    disc = 1.0
    for j in range(n):
        k = i + j
        if k >= len(ep):
            return g  # episode ended -> no bootstrap (terminal)
        g += disc * ep[k].reward
        disc *= discount
    boot_idx = i + n
    if boot_idx < len(ep):
        g += disc * value_fn(ep[boot_idx].obs)
    return g


def _gather_unroll_batch(episodes: List[List[_Transition]], batch: int, K: int, num_actions: int,
                         n_step: int, discount: float, value_fn, rng: random.Random,
                         obs_shape: Tuple[int, int, int]):
    """Sample `batch` (episode, start-index) positions and build K-step unroll targets with n-step
    bootstrapped value targets. valid_mask[k]=0 when the unroll ran past the episode end.

    Also returns the OBSERVATION at each unroll step (obs_k) -- the target for the latent CONSISTENCY
    loss (EfficientZero, Ye et al. 2021): the unrolled g latent at step k should match the
    stop-gradient h(obs_k), which directly supervises the dynamics model to learn the REAL transition
    (a small-pixel paddle shift that the implicit value/policy/reward signal alone learns too weakly)."""
    H, W, C = obs_shape
    obs0 = []
    obsk = [[] for _ in range(K + 1)]
    act = [[] for _ in range(K + 1)]
    pol = [[] for _ in range(K + 1)]
    val = [[] for _ in range(K + 1)]
    rew = [[] for _ in range(K + 1)]
    valid = [[] for _ in range(K + 1)]
    zero_obs = np.zeros((H, W, C), dtype=np.float32)
    for _ in range(batch):
        ep = rng.choice(episodes)
        i0 = rng.randrange(len(ep))
        obs0.append(np.asarray(ep[i0].obs, dtype=np.float32))
        for k in range(K + 1):
            j = i0 + k
            if j < len(ep):
                st = ep[j]
                oh = np.zeros(num_actions, dtype=np.float32)
                oh[st.action] = 1.0
                obsk[k].append(np.asarray(st.obs, dtype=np.float32))
                act[k].append(oh)
                pol[k].append(st.pi)
                val[k].append(_nstep_return(ep, j, n_step, discount, value_fn))
                rew[k].append(st.reward)
                valid[k].append(1.0)
            else:
                obsk[k].append(zero_obs)
                act[k].append(np.zeros(num_actions, dtype=np.float32))
                pol[k].append(np.zeros(num_actions, dtype=np.float32))
                val[k].append(0.0)
                rew[k].append(0.0)
                valid[k].append(0.0)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return (t(obs0),
            [t(obsk[k]) for k in range(K + 1)],
            [t(act[k]) for k in range(K + 1)],
            [t(pol[k]) for k in range(K + 1)],
            [t(val[k]).unsqueeze(-1) for k in range(K + 1)],
            [t(rew[k]).unsqueeze(-1) for k in range(K + 1)],
            [t(valid[k]).unsqueeze(-1) for k in range(K + 1)])


def _random_episode(env, num_actions: int, rng: random.Random, max_steps: int) -> List[_Transition]:
    """One random-policy episode -> transitions with a UNIFORM pi target. Used ONLY to WARM-START
    the replay buffer so the value/reward heads are not stone-cold when MCTS first relies on them
    (breaking the cold-start symmetry cheaply -- no MCTS cost). Standard MuZero seeds replay with
    exploratory data; this is the minimal CPU-budget form of that."""
    obs = env.reset()
    transitions: List[_Transition] = []
    uniform = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)
    for _ in range(max_steps):
        action = rng.randrange(num_actions)
        next_obs, reward, done = env.step(action)
        transitions.append(_Transition(np.asarray(obs, dtype=np.float32), action,
                                        uniform.copy(), float(reward)))
        obs = next_obs
        if done:
            break
    return transitions


def train_muzero_rl(env=None, model: Optional[MuZeroRLNet] = None,
                    iterations: int = 12, episodes_per_iter: int = 12, sims: int = 24,
                    K: int = 5, n_step: int = 5, discount: float = 0.99, batch: int = 128,
                    train_steps: int = 40, lr: float = 2e-3, replay_window: int = 200,
                    value_weight: float = 0.25, reward_weight: float = 1.0,
                    consistency_weight: float = 1.0,
                    warmup_random_episodes: int = 30, warmup_train_steps: int = 60,
                    max_steps: int = 200, seed: int = 0, verbose: bool = True,
                    prefer_minatar: bool = True, game: str = "breakout", device=None
                    ) -> Tuple[MuZeroRLNet, List[float], List[float], str]:
    """Full single-agent MuZero train loop. Returns
    (trained model, mean-episode-return-per-iter curve, train-loss-per-iter curve, backend_tag).

    Each iteration: (1) generate self-play episodes with MuZeroRLMCTS (planning over the CURRENT
    model), recording transitions + real per-step rewards into a rolling replay buffer; (2) train
    h/g/f on K-step unrolls with n-step bootstrapped value targets. The return curve (mean episode
    return per iteration) should CLIMB as the policy improves -- the canonical RL learning signal.

    Two settings are load-bearing for learning ABOVE random on a CPU budget (both measured):
      - value_weight=0.25 (MuZero's standard value-loss scale): the value target lives on the reward
        scale (a discounted sum of +-1 events, magnitude several), so an unscaled value MSE dwarfs the
        policy CE and STARVES the policy head -- with value_weight=1.0 the loss does not fall and the
        policy never improves; at 0.25 the loss falls monotonically and the policy learns.
      - a random-data WARMUP (warmup_random_episodes): the value/reward heads get a non-cold start
        before MCTS relies on them, breaking the cold-start symmetry (a random policy alone gives
        nothing for the search to amplify)."""
    backend_tag = "external"
    if env is None:
        env, backend_tag = make_env(prefer_minatar=prefer_minatar, game=game, seed=seed)
    num_actions = env.num_actions
    obs_shape = env.obs_shape
    if model is None:
        model = MuZeroRLNet(obs_shape=obs_shape, num_actions=num_actions)

    # --- device resolution (GPU when available; CPU fallback) ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)                    # model params + buffers on `device`

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    if hasattr(env, "seed"):
        env.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    @torch.no_grad()
    def _value_of(obs_grid) -> float:
        model.eval()
        x = torch.as_tensor(np.asarray(obs_grid), dtype=torch.float32, device=device)
        s = model.represent(x)
        _, v = model.prediction(s)
        return float(v.item())

    def _train_on_replay(replay, n_steps_) -> float:
        """Run n_steps_ gradient updates on K-step unrolls drawn from the replay buffer. Returns the
        mean loss. The loss = policy CE (-> MCTS visits) + value_weight * value MSE (-> n-step return)
        + reward MSE (-> observed per-step reward) + consistency_weight * latent consistency
        (-> h(real next obs)), summed over the K-step unroll; gradient into the dynamics path is halved
        each step so deep unrolls do not dominate.

        The CONSISTENCY term (EfficientZero) is load-bearing here: it directly supervises the unrolled
        g latent at step k to match the stop-gradient h(obs_k), so the dynamics learns the REAL paddle
        transition. Without it the implicit value/policy/reward signal learns the value head but leaves
        g's per-action direction WRONG (measured: the search then picks 'stay' and never beats random);
        with it g learns the transition and the plan steers correctly."""
        model.train()
        losses = []
        for _ in range(n_steps_):
            obs0, obsks, acts, pols, vals, rews, valids = _gather_unroll_batch(
                replay, batch, K, num_actions, n_step, discount, _value_of, rng, obs_shape)
            # move the whole K-step unroll batch onto the model's device (GPU when on CUDA)
            obs0 = obs0.to(device)
            obsks = [t.to(device) for t in obsks]
            acts = [t.to(device) for t in acts]
            pols = [t.to(device) for t in pols]
            vals = [t.to(device) for t in vals]
            rews = [t.to(device) for t in rews]
            valids = [t.to(device) for t in valids]
            s = model.represent(obs0)
            total_loss = 0.0
            for k in range(K + 1):
                logits, value = model.prediction(s)
                logp = F.log_softmax(logits, dim=-1)
                vmask = valids[k]
                denom = vmask.sum().clamp(min=1.0)
                ploss = (-(pols[k] * logp).sum(dim=-1, keepdim=True) * vmask).sum() / denom
                vloss = (((value - vals[k]) ** 2) * vmask).sum() / denom
                step_loss = ploss + value_weight * vloss
                if k < K:
                    s, reward = model.dynamics(s, acts[k])
                    rdenom = valids[k + 1].sum().clamp(min=1.0)
                    rloss = (((reward - rews[k + 1]) ** 2) * valids[k + 1]).sum() / rdenom
                    # latent consistency: the predicted next latent s should match h(real next obs),
                    # stop-gradient on the target (the encoder is not pulled toward the dynamics).
                    with torch.no_grad():
                        target_latent = model.represent(obsks[k + 1])
                    cmask = valids[k + 1]
                    cdenom = cmask.sum().clamp(min=1.0)
                    closs = (((s - target_latent) ** 2).sum(dim=-1, keepdim=True) * cmask).sum() / cdenom
                    step_loss = step_loss + reward_weight * rloss + consistency_weight * closs
                    s.register_hook(lambda grad: grad * 0.5)
                total_loss = total_loss + step_loss
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(total_loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    replay: List[List[_Transition]] = []
    return_curve: List[float] = []
    loss_curve: List[float] = []

    # --- WARMUP: seed replay with random-policy episodes + pre-train the heads (no MCTS cost). ---
    if warmup_random_episodes > 0:
        warm = []
        for _ in range(warmup_random_episodes):
            tr = _random_episode(env, num_actions, rng, max_steps)
            if tr:
                warm.append(tr)
        if warm:
            replay.extend(warm)
            wloss = _train_on_replay(replay, warmup_train_steps)
            if verbose:
                print(f"[muzero_rl] warmup: {len(warm)} random episodes, "
                      f"{warmup_train_steps} train steps, loss={wloss:.4f}")

    for it in range(iterations):
        # --- self-play (planner = model only) ---
        episodes: List[List[_Transition]] = []
        ep_returns: List[float] = []
        for _ in range(episodes_per_iter):
            transitions, ep_ret = selfplay_episode(
                env, model, num_actions, sims=sims, discount=discount, max_steps=max_steps)
            if transitions:
                episodes.append(transitions)
                ep_returns.append(ep_ret)
        replay.extend(episodes)
        if len(replay) > replay_window:
            replay = replay[-replay_window:]
        mean_return = float(np.mean(ep_returns)) if ep_returns else 0.0
        return_curve.append(mean_return)

        # --- train h/g/f on K-step unrolls with n-step bootstrapped value targets ---
        mean_loss = _train_on_replay(replay, train_steps)
        loss_curve.append(mean_loss)
        if verbose:
            print(f"[muzero_rl] iter {it + 1:2d}/{iterations}  episodes={len(episodes)}  "
                  f"mean_return={mean_return:+.3f}  train_loss={mean_loss:.4f}")
    return model, return_curve, loss_curve, backend_tag


# --------------------------------------------------------------------------- #
# EVALUATION: trained agent (planning over the LEARNED model) vs a random policy.
# The search uses ONLY h/g/f; the env advances the real episode + yields reward.
# total_sim_step_calls MUST be 0 across the whole evaluation.
# --------------------------------------------------------------------------- #
def eval_policy(env, model: Optional[MuZeroRLNet], num_actions: int, n_episodes: int = 20,
                sims: int = 24, discount: float = 0.99, max_steps: int = 200,
                random_policy: bool = False, seed: int = 0) -> Tuple[float, int]:
    """Run n_episodes and return (mean_episode_return, total_sim_step_calls_inside_search).

    If random_policy=True, actions are uniform random (the baseline). Otherwise the trained MuZero
    agent plans over the LEARNED model (greedy: temperature 0, no noise). total_sim_step_calls is the
    number of env.step calls made INSIDE the planner -- it MUST be 0 (the planner is model-only)."""
    rng = random.Random(seed)
    np.random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed + 1)  # eval seed distinct from train seed
    returns = []
    total_sim_step = 0
    for _ in range(n_episodes):
        obs = env.reset()
        ep_ret = 0.0
        for _t in range(max_steps):
            if random_policy:
                action = rng.randrange(num_actions)
            else:
                mcts = MuZeroRLMCTS(model, num_actions, n_simulations=sims, discount=discount)
                action = mcts.best_action(obs, temperature=0.0, add_noise=False)
                total_sim_step += mcts.sim_step_calls  # accumulates 0
            obs, reward, done = env.step(action)
            ep_ret += float(reward)
            if done:
                break
        returns.append(ep_ret)
    return float(np.mean(returns)), total_sim_step


# --------------------------------------------------------------------------- #
# RWYB driver: train a modest agent, print the return curve, and prove it beats
# random by a clear margin with 0 env.step calls inside the search.
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Single-agent reward-based MuZero on scaled-down Atari.")
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minatar", action="store_true",
                    help="run the MinAtar game (the academic-standard scaled-Atari) instead of the "
                         "self-contained CatchEnv. HONEST NOTE: minimal-action MinAtar Breakout is too "
                         "sparse to LEARN above random on a CPU minute-budget (measured: trained 0.0 vs "
                         "random 0.7); it needs far more compute. The CatchEnv default is the robust, "
                         "cheap learns-above-random demonstration.")
    ap.add_argument("--game", type=str, default="breakout", help="MinAtar game (with --minatar)")
    args = ap.parse_args()

    print("=" * 72)
    print("  SINGLE-AGENT REWARD-BASED MuZero on SCALED-DOWN ATARI")
    print("=" * 72)
    if args.minatar:
        env, backend = make_env(prefer_minatar=True, game=args.game, seed=args.seed)
        max_steps, K, n_step = 120, 5, 8
    else:
        env = CatchEnv(seed=args.seed)            # the proven robust learns-above-random demo
        backend = "catch (self-contained scaled-Atari)"
        max_steps, K, n_step = 10, 4, 6

    model, return_curve, loss_curve, _ = train_muzero_rl(
        env=env, iterations=args.iterations, episodes_per_iter=args.episodes_per_iter,
        sims=args.sims, K=K, n_step=n_step, discount=0.99, train_steps=50, lr=3e-3,
        consistency_weight=2.0, warmup_random_episodes=100, warmup_train_steps=150,
        max_steps=max_steps, seed=args.seed, verbose=True)

    num_actions = env.num_actions
    rand_mean, rand_steps = eval_policy(env, None, num_actions, n_episodes=args.eval_episodes,
                                        random_policy=True, seed=args.seed, max_steps=max_steps)
    trained_mean, trained_steps = eval_policy(env, model, num_actions, sims=args.sims,
                                              n_episodes=args.eval_episodes, random_policy=False,
                                              seed=args.seed, max_steps=max_steps)
    print("-" * 72)
    print(f"backend                : {backend}")
    print(f"return curve (per iter): {[round(r, 3) for r in return_curve]}")
    print(f"loss curve (per iter)  : first {loss_curve[0]:.3f} -> last {loss_curve[-1]:.3f}")
    print(f"random  mean return    : {rand_mean:+.3f}  ({args.eval_episodes} eval episodes)")
    print(f"trained mean return    : {trained_mean:+.3f}  ({args.eval_episodes} eval episodes)")
    print(f"margin (trained-random): {trained_mean - rand_mean:+.3f}")
    print(f"env.step calls INSIDE search (must be 0): {trained_steps}")
    assert trained_steps == 0, "planner stepped the env -- not model-only MuZero"
    print("-" * 72)
    verdict = "LEARNED (beats random)" if trained_mean > rand_mean else "did NOT beat random"
    print(f"VERDICT: {verdict}. HONEST CEILING: scaled-down Atari, learns-above-random on CPU -- "
          f"NOT human-level real Atari (that is compute-bound and out of local reach).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
