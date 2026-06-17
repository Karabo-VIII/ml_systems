"""
chess_zero.az.train_atari_gpu -- GPU MuZero-RL training to SOLID competence on a scaled-Atari env,
with a turnkey CHAMPION checkpoint.

Two targets (run both; keep the more impressive one that clears its bar):
  --env catch            : the robust, low-variance CatchEnv (single-drop +-1 terminal). Optimal
                           return ~+1.0, random ~-0.6. Target: trained mean return >= +0.5 (a big
                           margin) over 30 eval episodes.
  --env minatar:<game>   : a REAL MinAtar game (Young & Tian 2019). Freeway has the densest reward,
                           so try it first. Target: trained clearly beats random over 30 episodes.

CHAMPION-GATE: eval every chunk; keep + save the BEST trained-mean-return net seen (a late noisy
iter cannot regress the saved artifact).

Saved checkpoint (projects/chess_zero/az/checkpoints/atari.pt):
    torch.save({"state_dict": net.state_dict(),
                "arch": {<MuZeroRLNet ctor kwargs>},
                "env": "catch" | "minatar:Freeway",
                "meta": {"trained_return": X, "random_return": Y, "margin": Z, "n_eval": 30, ...}}, p)
Loader contract:  net = MuZeroRLNet(**ckpt["arch"]); net.load_state_dict(ckpt["state_dict"]).

Run:  .venv/Scripts/python.exe -m az.train_atari_gpu --env catch [...]
No emoji (Windows cp1252). ADDITIVE (new file; touches no existing module).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .minatar_env import CatchEnv, MinAtarEnv, make_env
from .muzero_rl import MuZeroRLNet, train_muzero_rl, eval_policy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
_CKPT_PATH = os.path.join(_CKPT_DIR, "atari.pt")


def _log(env_tag: str, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}][{env_tag}] {msg}"
    print(line, flush=True)
    logp = os.path.join(_HERE, f"atari_gpu_train_{env_tag.replace(':', '_')}.log")
    with open(logp, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _build_env(env_arg: str, seed: int):
    """Return (env, env_tag, max_steps, K, n_step). env_arg: 'catch' or 'minatar:<game>'."""
    if env_arg.startswith("minatar"):
        game = env_arg.split(":", 1)[1] if ":" in env_arg else "freeway"
        env, backend = make_env(prefer_minatar=True, game=game, seed=seed)
        if not isinstance(env, MinAtarEnv):
            raise RuntimeError(f"requested MinAtar:{game} but got fallback ({backend})")
        return env, f"minatar:{game}", 200, 5, 10
    env = CatchEnv(seed=seed)
    return env, "catch", 10, 4, 6


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU MuZero-RL -> champion checkpoint on scaled-Atari.")
    ap.add_argument("--env", type=str, default="catch",
                    help="'catch' or 'minatar:<game>' (e.g. minatar:freeway)")
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--episodes-per-iter", type=int, default=24)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-steps", type=int, default=50)
    ap.add_argument("--consistency-weight", type=float, default=2.0)
    ap.add_argument("--warmup-episodes", type=int, default=100)
    ap.add_argument("--warmup-train-steps", type=int, default=150)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--eval-seeds", type=int, default=3,
                    help="number of distinct eval seeds to average the gate eval over (variance "
                         "reduction; a single-seed +-1 mean is noisy).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", action="store_true",
                    help="save the champion to atari.pt (only pass for the env you want to ship).")
    ap.add_argument("--target-return", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="override the per-episode step cap (0 = env default). Capping MinAtar "
                         "episodes makes self-play chunks far cheaper (batch-1 latent search).")
    args = ap.parse_args()

    os.makedirs(_CKPT_DIR, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    env, env_tag, max_steps, K, n_step = _build_env(args.env, args.seed)
    if args.max_steps > 0:
        max_steps = args.max_steps
    num_actions = env.num_actions
    obs_shape = env.obs_shape
    open(os.path.join(_HERE, f"atari_gpu_train_{env_tag.replace(':', '_')}.log"),
         "w", encoding="utf-8").close()

    arch = dict(obs_shape=tuple(int(x) for x in obs_shape), num_actions=int(num_actions),
                latent_dim=args.latent_dim, channels=args.channels, hidden=args.hidden)
    model = MuZeroRLNet(**arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(env_tag, f"device={device}  obs_shape={obs_shape}  num_actions={num_actions}  "
                  f"MuZeroRLNet params={n_params:,}  max_steps={max_steps} K={K} n_step={n_step}")

    # MULTI-SEED gate eval: a single-seed 40-ep mean of +-1 outcomes is high-variance (the SAME net
    # measured +0.48 vs +0.85 across seeds), which makes the champion-gate noisy. We average the eval
    # over several seeds so the saved-champion decision tracks the net's REAL strength, not one lucky
    # seed's draw of ball/paddle starts.
    eval_seeds = [args.seed + 6007 * k for k in range(args.eval_seeds)]

    def _eval_mean(m, random_policy):
        vals = []
        last_steps = 0
        for s in eval_seeds:
            mean_r, last_steps = eval_policy(env, m, num_actions, sims=args.sims,
                                             n_episodes=args.n_eval, random_policy=random_policy,
                                             seed=s, max_steps=max_steps)
            vals.append(mean_r)
        return float(np.mean(vals)), float(np.min(vals)), last_steps

    # random baseline (fixed; the bar the trained agent must clear), multi-seed mean
    rand_mean, _rmin, _ = _eval_mean(None, random_policy=True)
    _log(env_tag, f"random baseline mean return ({args.n_eval} eps x {len(eval_seeds)} seeds): "
                  f"{rand_mean:+.3f}")

    best_return = -1e9
    best_meta = None
    curve = []
    t_start = time.perf_counter()
    iters_done = 0
    warm = args.warmup_episodes
    warm_steps = args.warmup_train_steps
    while iters_done < args.iters:
        chunk = min(args.chunk, args.iters - iters_done)
        model, rc, lc, _tag = train_muzero_rl(
            env=env, model=model, iterations=chunk, episodes_per_iter=args.episodes_per_iter,
            sims=args.sims, K=K, n_step=n_step, discount=0.99, train_steps=args.train_steps,
            lr=args.lr, consistency_weight=args.consistency_weight,
            warmup_random_episodes=warm, warmup_train_steps=warm_steps,
            max_steps=max_steps, seed=args.seed + iters_done, verbose=False, device=device)
        warm, warm_steps = 0, 0  # warmup only the FIRST chunk
        iters_done += chunk

        trained_mean, trained_min, steps = _eval_mean(model, random_policy=False)
        assert steps == 0, "planner stepped the env -- not model-only MuZero"
        margin = trained_mean - rand_mean
        elapsed = time.perf_counter() - t_start
        _log(env_tag, f"iter {iters_done:2d}/{args.iters}  trained {trained_mean:+.3f} "
                      f"(min-seed {trained_min:+.3f})  random {rand_mean:+.3f}  margin {margin:+.3f}  "
                      f"selfplay_return~{rc[-1]:+.3f}  loss~{lc[-1]:.3f}  [{elapsed:.0f}s]")
        curve.append({"iter": iters_done, "trained_return": round(trained_mean, 3),
                      "trained_min_seed": round(trained_min, 3), "margin": round(margin, 3)})

        if trained_mean > best_return:
            best_return = trained_mean
            best_meta = {"trained_return": round(float(trained_mean), 4),
                         "trained_return_min_seed": round(float(trained_min), 4),
                         "random_return": round(float(rand_mean), 4),
                         "margin": round(float(margin), 4), "n_eval": args.n_eval,
                         "eval_seeds": len(eval_seeds),
                         "env": env_tag, "iters": iters_done, "sims": args.sims,
                         "max_steps": max_steps, "arch": arch, "device": str(device)}
            if args.save:
                torch.save({"state_dict": model.state_dict(), "arch": arch,
                            "env": env_tag, "meta": best_meta}, _CKPT_PATH)
                _log(env_tag, f"  -> NEW CHAMPION saved (return {trained_mean:+.3f}): {_CKPT_PATH}")
            else:
                _log(env_tag, f"  -> new best (return {trained_mean:+.3f}); --save not set, not writing")

    total = time.perf_counter() - t_start
    _log(env_tag, "=" * 60)
    _log(env_tag, f"DONE in {total:.0f}s. BEST trained return {best_return:+.3f} "
                  f"(random {rand_mean:+.3f}, margin {best_return - rand_mean:+.3f}) "
                  f"at iter {best_meta['iters']}.")
    verdict = ("CLEARS target" if best_return >= args.target_return
               else "beats random" if best_return > rand_mean else "did NOT beat random")
    _log(env_tag, f"VERDICT: {verdict} (target >= {args.target_return}).")
    with open(os.path.join(_HERE, f"atari_gpu_curve_{env_tag.replace(':', '_')}.json"),
              "w", encoding="utf-8") as f:
        json.dump({"curve": curve, "champion": best_meta, "random_return": rand_mean}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
