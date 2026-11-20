"""
chess_zero.az.train_minatar -- the GPU training driver for the MinAtar DQN agent.

Trains a Double-DQN (az.dqn_minatar) on a MinAtar game until it CLEARLY beats the random-policy
baseline, with a periodic eval-vs-random and a champion-save (best eval return so far). Writes a
learning-curve JSON for inspection and a `play_episode` entry so the overseer can run / render one
episode for the demo.

USAGE (RWYB):
    python -m az.train_minatar --game breakout --frames 400000 --eval-episodes 30
    python -m az.train_minatar --verify-only            # reload the saved ckpt and re-eval
    python -m az.train_minatar --play                   # run + render one episode from the ckpt

CHECKPOINT (canonical contract):
    {"state_dict", "arch": {ctor kwargs}, "game": "minatar:<name>",
     "meta": {"trained_return", "random_return", "n_eval", "episodes_trained"}}
saved to  projects/chess_zero/az/checkpoints/atari_minatar.pt

GPU is SHARED (a chess trainer + a src/wm process are running): we use a small batch (32) and a tiny
net, and a separate CUDA context is fine. No emoji (Windows cp1252). ADDITIVE: new file only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

# Allow `python az/train_minatar.py` as well as `python -m az.train_minatar`.
try:
    from az.dqn_minatar import DQNAgent, ReplayBuffer, load_qnet, MinAtarQNet
    from az.minatar_env import make_env
except ImportError:  # pragma: no cover - direct-script invocation
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from az.dqn_minatar import DQNAgent, ReplayBuffer, load_qnet, MinAtarQNet
    from az.minatar_env import make_env

__contract__ = {
    "kind": "rl-training-driver",
    "inputs": ["a MinAtar game name", "frame budget + eval cadence"],
    "outputs": [
        "a champion checkpoint at az/checkpoints/atari_minatar.pt",
        "a learning-curve JSON (eval return vs frames)",
        "RWYB stdout: random baseline + trained + reloaded eval returns",
    ],
    "invariants": [
        "eval is GREEDY (eps=0) over >= n_eval episodes; random baseline uses the SAME episode count",
        "champion-save keeps the best eval return seen",
        "the saved checkpoint is reload-verified before the run is declared a success",
        "no emoji in any print (Windows cp1252)",
    ],
}

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "atari_minatar.pt")
CURVE_PATH = os.path.join(_HERE, "minatar_dqn_curve.json")


def ckpt_path_for(game: str) -> str:
    """Per-game champion checkpoint path.

    Breakout keeps the legacy filename `atari_minatar.pt` (the original single-game checkpoint the
    demo already loads); every OTHER game gets `atari_minatar_<game>.pt`. This makes the trainer
    multi-game without disturbing the existing breakout artifact."""
    if game == "breakout":
        return os.path.join(CKPT_DIR, "atari_minatar.pt")
    return os.path.join(CKPT_DIR, f"atari_minatar_{game}.pt")


def curve_path_for(game: str) -> str:
    """Per-game learning-curve JSON path (breakout keeps the legacy filename)."""
    if game == "breakout":
        return os.path.join(_HERE, "minatar_dqn_curve.json")
    return os.path.join(_HERE, f"minatar_dqn_curve_{game}.json")


# --------------------------------------------------------------------------- #
# Evaluation helpers (greedy net policy + random policy share one harness).
# --------------------------------------------------------------------------- #
def eval_random(game: str, n_episodes: int, seed: int = 999, max_steps: int = 5000) -> float:
    """Mean return of a uniform-random policy over n_episodes (the honest baseline)."""
    env, _ = make_env(prefer_minatar=True, game=game, seed=seed)
    env.seed(seed)
    rng = np.random.RandomState(seed)
    rets = []
    for _ in range(n_episodes):
        env.reset()
        done = False
        R = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = rng.randint(env.num_actions)
            _, r, done = env.step(a)
            R += r
            steps += 1
        rets.append(R)
    return float(np.mean(rets))


def eval_policy(act_fn, game: str, n_episodes: int, seed: int = 777,
                max_steps: int = 5000) -> tuple:
    """Mean return of a deterministic policy `act_fn(obs)->action` over n_episodes.

    Returns (mean, std, all_returns)."""
    env, _ = make_env(prefer_minatar=True, game=game, seed=seed)
    env.seed(seed)
    rets = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        R = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = act_fn(obs)
            obs, r, done = env.step(a)
            R += r
            steps += 1
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets)), rets


# --------------------------------------------------------------------------- #
# Training loop.
# --------------------------------------------------------------------------- #
def train(game: str = "breakout", frames: int = 400_000, eval_episodes: int = 30,
          eval_every: int = 20_000, batch_size: int = 32, buffer_cap: int = 100_000,
          warmup: int = 5_000, target_sync: int = 1_000, eps_start: float = 1.0,
          eps_end: float = 0.05, eps_decay_frames: int = 100_000, gamma: float = 0.99,
          lr: float = 2.5e-4, seed: int = 0, time_budget_s: Optional[float] = None,
          train_freq: int = 1, device_str: Optional[str] = None,
          dueling: bool = True, n_step: int = 3) -> dict:
    """Train the DQN; return a result dict. Saves the champion checkpoint + curve JSON.

    train_freq: do one learn() step every `train_freq` frames (default 1 = learn every frame, the
        original behavior). A value of 4 is the standard DQN setting: it collects 4x more experience
        per learn step, which on a small net is much faster wall-clock (the learn backward dominates)
        with negligible loss of sample-efficiency -- the right lever when the GPU is contended.
    device_str: force 'cpu' or 'cuda' (default: auto-pick cuda if available). On a SATURATED shared
        GPU the batch-1 act() + small batch-32 learn() are kernel-launch-latency bound and CPU is
        actually faster; this lets us run on CPU to avoid queuing behind other CUDA processes."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = ckpt_path_for(game)
    curve_path = curve_path_for(game)
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_freq = max(1, int(train_freq))
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, tag = make_env(prefer_minatar=True, game=game, seed=seed)
    env.seed(seed)
    if not tag.startswith("minatar:"):
        raise RuntimeError(f"MinAtar backend unavailable (got tag={tag}); refusing to train on fallback")

    print(f"[train_minatar] game={tag} obs={env.obs_shape} actions={env.num_actions} device={device}")

    # honest random baseline FIRST
    random_return = eval_random(game, eval_episodes)
    print(f"[train_minatar] random-policy baseline over {eval_episodes} eps: mean={random_return:.3f}")

    agent = DQNAgent(env.obs_shape, env.num_actions, device=device, lr=lr, gamma=gamma,
                     dueling=dueling, n_step=n_step)
    buffer = ReplayBuffer(buffer_cap, env.obs_shape)
    print(f"[train_minatar] Rainbow upgrades: dueling={dueling} n_step={n_step} "
          f"(Double-DQN already on)")

    # --- n-step return accumulation (Rainbow): emit (obs0, a0, R_n, obs_n, done_n) ----------- #
    _nbuf = deque()

    def _emit_nstep():
        obs0, a0 = _nbuf[0][0], _nbuf[0][1]
        ret = 0.0
        boot_obs, boot_done = _nbuf[-1][3], _nbuf[-1][4]
        for k, (_, _, rr, no, dn) in enumerate(_nbuf):
            ret += (gamma ** k) * rr
            if dn or k == agent.n_step - 1:
                boot_obs, boot_done = no, dn
                break
        buffer.push(obs0, a0, ret, boot_obs, boot_done)
        _nbuf.popleft()

    def push_transition(o, a_, r_, no_, dn_):
        _nbuf.append((o, a_, r_, no_, dn_))
        if len(_nbuf) >= agent.n_step:
            _emit_nstep()
        if dn_:
            while _nbuf:
                _emit_nstep()

    def eps_at(frame: int) -> float:
        if frame >= eps_decay_frames:
            return eps_end
        frac = frame / max(1, eps_decay_frames)
        return eps_start + frac * (eps_end - eps_start)

    curve = []  # list of {frame, eval_mean, eval_std, eps, best}
    best_return = -1e9
    best_meta = {}
    # CHAMPION GATE: never overwrite a STRONGER existing champion (protects the committed weights;
    # a short/experimental run cannot clobber a strong one). The run must BEAT the existing score.
    if os.path.exists(ckpt_path):
        try:
            _prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _pb = float(_prev.get("meta", {}).get("trained_return", -1e9))
            if _pb > -1e8:
                best_return = _pb
                print(f"[train_minatar] champion gate: existing champion = {_pb:.3f}; "
                      f"this run overwrites ONLY if it beats that")
        except Exception:
            pass
    obs = env.reset()
    ep_return = 0.0
    ep_returns = []
    losses = []
    t0 = time.time()
    last_eval_frame = 0
    episodes_done = 0

    for frame in range(1, frames + 1):
        eps = eps_at(frame)
        a = agent.act(obs, eps)
        next_obs, r, done = env.step(a)
        push_transition(obs, a, r, next_obs, done)
        ep_return += r
        obs = next_obs
        if done:
            ep_returns.append(ep_return)
            ep_return = 0.0
            episodes_done += 1
            obs = env.reset()

        if len(buffer) >= warmup and (frame % train_freq == 0):
            loss = agent.learn(buffer, batch_size)
            losses.append(loss)
        if frame % target_sync == 0:
            agent.sync_target()

        # periodic eval vs random
        if frame - last_eval_frame >= eval_every:
            last_eval_frame = frame
            mean, std, _ = eval_policy(agent.act_greedy, game, eval_episodes)
            recent_train = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            recent_loss = float(np.mean(losses[-500:])) if losses else 0.0
            elapsed = time.time() - t0
            is_best = mean > best_return
            if is_best:
                best_return = mean
                best_meta = {
                    "trained_return": round(mean, 4),
                    "random_return": round(random_return, 4),
                    "n_eval": eval_episodes,
                    "episodes_trained": episodes_done,
                    "frames_trained": frame,
                }
                agent.save(ckpt_path, game=tag, meta=best_meta)
            curve.append({"frame": frame, "eval_mean": round(mean, 4), "eval_std": round(std, 4),
                          "eps": round(eps, 4), "best": round(best_return, 4),
                          "train_recent": round(recent_train, 4)})
            print(f"[train_minatar] frame={frame:>7d} eps={eps:.3f} eval_mean={mean:6.3f} "
                  f"(std={std:5.3f}) best={best_return:6.3f} train50={recent_train:6.3f} "
                  f"loss={recent_loss:.4f} t={elapsed:5.0f}s {'  <-NEW BEST*' if is_best else ''}")
            with open(curve_path, "w") as f:
                json.dump({"game": tag, "random_return": random_return, "curve": curve}, f, indent=2)

        if time_budget_s is not None and (time.time() - t0) > time_budget_s:
            print(f"[train_minatar] time budget {time_budget_s:.0f}s reached at frame {frame}; stopping")
            break

    # final eval of the LAST net (the champion on disk may be from an earlier best)
    final_mean, final_std, _ = eval_policy(agent.act_greedy, game, eval_episodes)
    print(f"[train_minatar] FINAL net eval: mean={final_mean:.3f} std={final_std:.3f}")
    if final_mean > best_return:
        best_return = final_mean
        best_meta = {
            "trained_return": round(final_mean, 4),
            "random_return": round(random_return, 4),
            "n_eval": eval_episodes,
            "episodes_trained": episodes_done,
            "frames_trained": frame,
        }
        agent.save(ckpt_path, game=tag, meta=best_meta)
        print(f"[train_minatar] FINAL net is the champion (mean={final_mean:.3f}); saved")

    if best_meta:
        print(f"[train_minatar] CHAMPION saved to {ckpt_path}: {best_meta}")
    else:
        print(f"[train_minatar] champion UNCHANGED -- this run's best ({best_return:.3f}) did not "
              f"beat the existing champion; nothing was written (gate held)")
    return {
        "game": tag,
        "random_return": random_return,
        "best_return": best_return,
        "best_meta": best_meta,
        "curve": curve,
        "ckpt_path": ckpt_path,
    }


# --------------------------------------------------------------------------- #
# Verify (reload contract) + play (single-episode demo entry).
# --------------------------------------------------------------------------- #
def verify(eval_episodes: Optional[int] = None, game: str = "breakout") -> dict:
    """Reload the saved checkpoint into a FRESH net and re-run the eval. This is the reload contract:
    the SAVED net (built only from arch kwargs + state_dict) must reproduce the trained return."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path_for(game)
    net, ckpt = load_qnet(ckpt_path, device=device)
    game_tag = ckpt["game"]
    game = game_tag.split(":", 1)[1] if ":" in game_tag else game_tag
    n_eval = eval_episodes or int(ckpt["meta"].get("n_eval", 30))

    @torch.no_grad()
    def greedy(obs):
        chw = np.ascontiguousarray(np.transpose(obs, (2, 0, 1)), dtype=np.float32)
        x = torch.from_numpy(chw).unsqueeze(0).to(device)
        return int(torch.argmax(net(x), dim=1).item())

    mean, std, rets = eval_policy(greedy, game, n_eval)
    rand = eval_random(game, n_eval)
    print(f"[verify] reloaded {ckpt_path}")
    print(f"[verify] game={game_tag}  saved-meta={ckpt['meta']}")
    print(f"[verify] RELOADED net eval over {n_eval} eps: mean={mean:.3f} std={std:.3f} "
          f"(min={min(rets):.1f} max={max(rets):.1f})")
    print(f"[verify] random baseline over {n_eval} eps:   mean={rand:.3f}")
    print(f"[verify] margin (reloaded - random) = {mean - rand:.3f}")
    return {"game": game_tag, "reloaded_return": mean, "reloaded_std": std,
            "random_return": rand, "n_eval": n_eval, "saved_meta": ckpt["meta"]}


def play_episode(render: bool = False, seed: int = 12345, game: str = "breakout") -> float:
    """Run ONE greedy episode from the saved checkpoint; optionally print the grid each step.

    Returns the episode return. The overseer wires this into the demo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_qnet(ckpt_path_for(game), device=device)
    game_tag = ckpt["game"]
    game = game_tag.split(":", 1)[1] if ":" in game_tag else game_tag
    env, _ = make_env(prefer_minatar=True, game=game, seed=seed)
    env.seed(seed)

    @torch.no_grad()
    def greedy(obs):
        chw = np.ascontiguousarray(np.transpose(obs, (2, 0, 1)), dtype=np.float32)
        x = torch.from_numpy(chw).unsqueeze(0).to(device)
        return int(torch.argmax(net(x), dim=1).item())

    obs = env.reset()
    done = False
    R = 0.0
    steps = 0
    while not done and steps < 5000:
        if render:
            grid = np.asarray(obs)
            # collapse channels to a single ascii map: '.' empty, digit = which channel is on
            flat = np.zeros(grid.shape[:2], dtype=int) - 1
            for c in range(grid.shape[2]):
                flat[grid[:, :, c] > 0] = c
            rows = ["".join("." if v < 0 else str(v) for v in row) for row in flat]
            print(f"--- step {steps} (return {R:.1f}) ---")
            print("\n".join(rows))
        a = greedy(obs)
        obs, r, done = env.step(a)
        R += r
        steps += 1
    print(f"[play] game={game_tag} episode return={R:.1f} over {steps} steps")
    return R


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train/verify/play a DQN on MinAtar (scaled Atari).")
    ap.add_argument("--game", default="breakout",
                    help="MinAtar game: breakout/asterix/freeway/space_invaders/seaquest")
    ap.add_argument("--frames", type=int, default=400_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="wall-clock cap in minutes (honest-stop)")
    ap.add_argument("--train-freq", type=int, default=1,
                    help="learn() once every N frames (default 1; 4 = faster wall-clock, std DQN)")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                    help="force device (default: auto cuda-if-available; cpu avoids a saturated GPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dueling", dest="dueling", action="store_true", default=True,
                    help="dueling Q-network -- split V(s)+A(s,a) streams (default ON; Rainbow)")
    ap.add_argument("--no-dueling", dest="dueling", action="store_false",
                    help="the original plain conv->fc Q-network")
    ap.add_argument("--n-step", type=int, default=3,
                    help="n-step return bootstrapping (default 3; 1 = original 1-step)")
    ap.add_argument("--verify-only", action="store_true", help="reload the saved ckpt and re-eval")
    ap.add_argument("--play", action="store_true", help="run + render one episode from the ckpt")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.eval_episodes, game=args.game)
        return
    if args.play:
        play_episode(render=True, game=args.game)
        return

    budget = args.time_budget_min * 60.0 if args.time_budget_min else None
    train(game=args.game, frames=args.frames, eval_episodes=args.eval_episodes,
          eval_every=args.eval_every, batch_size=args.batch_size, seed=args.seed,
          time_budget_s=budget, train_freq=args.train_freq, device_str=args.device,
          dueling=args.dueling, n_step=args.n_step)
    print("\n[main] === RELOAD VERIFICATION ===")
    verify(args.eval_episodes, game=args.game)


if __name__ == "__main__":
    main()
