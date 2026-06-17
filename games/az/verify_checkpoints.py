"""
chess_zero.az.verify_checkpoints -- RWYB reload check for the demo checkpoints.

Loads each saved checkpoint FROM DISK into a FRESH net (the exact loader contract a demo would use)
and re-runs the strength eval, proving the SAVED weights reproduce the trained strength (a checkpoint
that does not reload is useless for the "click run and watch it play" demo).

Reload contract (both files):
    connect4.pt:  net = Connect4Net(**ckpt["arch"]); net.load_state_dict(ckpt["state_dict"])
    atari.pt:     net = MuZeroRLNet(**ckpt["arch"]); net.load_state_dict(ckpt["state_dict"])

Run:  .venv/Scripts/python.exe -m az.verify_checkpoints
No emoji (Windows cp1252).
"""
from __future__ import annotations

import os

import torch

from .connect4 import eval_wdl_vs_random, eval_wdl_vs_heuristic
from .net import Connect4Net
from .minatar_env import CatchEnv, make_env, MinAtarEnv
from .muzero_rl import MuZeroRLNet, eval_policy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")


def verify_connect4(device, eval_games: int = 40, eval_sims: int = 128, seed: int = 4242):
    path = os.path.join(_CKPT_DIR, "connect4.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # FRESH net from the stored arch -> load weights -> move to device
    net = Connect4Net(**ckpt["arch"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    rw, rd, rl = eval_wdl_vs_random(net, n_games=eval_games, sims=eval_sims, seed=seed, device=device)
    hw, hd, hl = eval_wdl_vs_heuristic(net, n_games=eval_games, sims=eval_sims,
                                       seed=seed + 1, device=device)
    print("=" * 64)
    print(f"CONNECT-4 reloaded from {path}")
    print(f"  arch          : {ckpt['arch']}")
    print(f"  meta(saved)   : vs_random={ckpt['meta']['vs_random']} "
          f"vs_heuristic={ckpt['meta']['vs_heuristic']}")
    print(f"  RELOADED vs RANDOM    ({eval_games} games, {eval_sims} sims): "
          f"W{rw} D{rd} L{rl}  (non-loss {(rw + rd) / eval_games:.3f})")
    print(f"  RELOADED vs HEURISTIC ({eval_games} games, {eval_sims} sims): "
          f"W{hw} D{hd} L{hl}  (non-loss {(hw + hd) / eval_games:.3f})")
    ok = (rw + rd) / eval_games >= 0.90 and (hw + hd) / eval_games >= 0.55 and hw > hl
    print(f"  VERDICT: {'PASS' if ok else 'WEAK'} -- reloaded net crushes random and beats the "
          f"1-ply heuristic." if ok else
          f"  VERDICT: WEAK -- reloaded numbers below the demo bar.")
    return ok


def verify_atari(device, n_eval: int = 30, seed: int = 1):
    path = os.path.join(_CKPT_DIR, "atari.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = MuZeroRLNet(**ckpt["arch"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    env_tag = ckpt.get("env", "catch")
    if env_tag.startswith("minatar"):
        game = env_tag.split(":", 1)[1]
        env, backend = make_env(prefer_minatar=True, game=game, seed=seed)
        max_steps = ckpt["meta"].get("max_steps", 200)
    else:
        env = CatchEnv(seed=seed)
        max_steps = ckpt["meta"].get("max_steps", 10)
    sims = ckpt["meta"].get("sims", 24)
    num_actions = env.num_actions
    # multi-seed eval: a single-seed +-1 mean is high-variance; average for a stable RWYB number.
    eval_seeds = [seed + 6007 * k for k in range(5)]
    import numpy as np
    rand_vals, tr_vals, steps = [], [], 0
    for s in eval_seeds:
        rm, _ = eval_policy(env, None, num_actions, n_episodes=n_eval,
                            random_policy=True, seed=s, max_steps=max_steps)
        tm, steps = eval_policy(env, net, num_actions, sims=sims, n_episodes=n_eval,
                                random_policy=False, seed=s, max_steps=max_steps)
        rand_vals.append(rm)
        tr_vals.append(tm)
    rand_mean = float(np.mean(rand_vals))
    trained_mean = float(np.mean(tr_vals))
    trained_min = float(np.min(tr_vals))
    margin = trained_mean - rand_mean
    print("=" * 64)
    print(f"ATARI ({env_tag}) reloaded from {path}")
    print(f"  arch          : {ckpt['arch']}")
    print(f"  meta(saved)   : trained_return={ckpt['meta']['trained_return']} "
          f"random_return={ckpt['meta']['random_return']} margin={ckpt['meta']['margin']}")
    print(f"  RELOADED trained mean return ({n_eval} eps x {len(eval_seeds)} seeds, {sims} sims): "
          f"{trained_mean:+.3f}  (min-seed {trained_min:+.3f})")
    print(f"  RELOADED random  mean return ({n_eval} eps x {len(eval_seeds)} seeds)            : "
          f"{rand_mean:+.3f}")
    print(f"  margin (trained-random): {margin:+.3f}   env.step-in-search: {steps} (must be 0)")
    assert steps == 0, "planner stepped the env -- not model-only MuZero"
    ok = (trained_mean >= 0.5 and margin > 0.4) if not env_tag.startswith("minatar") else (margin > 0)
    print(f"  VERDICT: {'PASS' if ok else 'WEAK'} -- reloaded net "
          f"{'clears the competence bar' if ok else 'is below the demo bar'}.")
    return ok


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"verify device: {device}")
    c4 = verify_connect4(device)
    at = verify_atari(device)
    print("=" * 64)
    print(f"RELOAD VERIFICATION: connect4 {'PASS' if c4 else 'WEAK'}  |  atari {'PASS' if at else 'WEAK'}")
    return 0 if (c4 and at) else 1


if __name__ == "__main__":
    raise SystemExit(main())
