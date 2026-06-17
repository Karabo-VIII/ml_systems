"""experiments/adaptive_ma/expert/falsifiers.py -- robustness / -k falsifiers for the adaptive-MA finding.

Confirms the main result (run_u100.py) is NOT an artifact of one config choice:
  F-A. EXIT-POLICY sensitivity: re-run the firewall with a TIME-STOP exit (max_hold) instead of
       opposite-cross -> does a different uniform exit hide a timing edge? (brief: "is the exit the same?")
  F-B. ALTERNATE fixed baselines: adaptive vs each CONSTITUENT config of the map (not just 10/30),
       pooled held-out -> is 10/30 a cherry-picked-hard baseline, or does adaptive fail vs all?
  F-C. BETA benchmark (src/strat/benchmark.py): does adaptive beat a beta-matched costless hold on
       held-out? (defense-in-depth vs the firewall's random-entry null.)
  F-D. COST sensitivity: firewall is cost-MATCHED already, but report adaptive held-out compound at
       taker(0.0024) vs ideal(0.0010) to show cost is not the sole killer of the underlying signal.

Runs on a representative subset (default 20 assets) for speed; the firewall verdict was already 0/69 on
the full universe, so this is confirmation, not the primary evidence. RWYB:
  python experiments/adaptive_ma/expert/falsifiers.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_ma as A  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec  # noqa: E402
from strat.firewall import random_entry_null  # noqa: E402
from strat.benchmark import benchmark_excess  # noqa: E402

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
TAKER = 0.0024
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")


def _load(loader, s):
    try:
        g = loader.load(s, cadence="1d")
    except Exception:
        return None
    return pd.DataFrame({"date": pd.to_datetime(g["date"].to_list()),
                         "open": g["open"].to_numpy().astype(float), "high": g["high"].to_numpy().astype(float),
                         "low": g["low"].to_numpy().astype(float), "close": g["close"].to_numpy().astype(float)})


def _spec(fast, slow, exit_policy, cost=TAKER, max_hold=None, ma_type="sma"):
    return StrategySpec(fast_col=fast, slow_col=slow, signal="crossover", filter_col=None,
                        exit_policy=exit_policy, cost_rt=cost, use_funding=False, funding_scale=0.0,
                        max_hold_bars=max_hold, max_hold_ext_bars=max_hold)


def _comp(nets):
    a = np.asarray(nets, float)
    return float((np.prod(1.0 + a) - 1.0) * 100) if a.size else 0.0


def _held_nets(res):
    return [t["net_pnl"] for t in res.trades if t["window"] in HELD]


def main(n):
    loader = ChimeraLoader()
    syms = UniverseLoader.load().list("u100")[:n]
    frames = {s: df for s in syms if (df := _load(loader, s)) is not None and len(df) > 400}
    print(f"[falsifiers] {len(frames)} assets\n")

    # F-A: exit-policy sensitivity (time-stop) firewall, F-C beta, F-D cost, F-B constituents
    fa_beat = 0; fa_eval = 0
    fc_beat = 0; fc_eval = 0
    # constituent configs (unique) for F-B
    constit = sorted(set(A.all_configs()))
    fb_pool_adapt = []
    fb_pool_by_cfg = {c: [] for c in constit}
    fd_taker = []; fd_ideal = []

    for s, df in frames.items():
        feat = A.compute_features(df, xs_disp=None)
        adp = A.build_adaptive_columns(feat)

        # F-A: adaptive with TIME-STOP exit (max_hold=20) -> firewall
        h_ts = CanonicalHarness(adp, _spec("adaptive_fast", "adaptive_slow", "max_hold_only", max_hold=20), WIN,
                                chimera_path=f"ts::{s}")
        try:
            fw = random_entry_null(h_ts, n_books=150, seed=7)
            fa_eval += 1
            fa_beat += 1 if fw["beats_held"] else 0
        except Exception:
            pass

        # F-C: beta-matched benchmark on opposite-cross adaptive
        h_oc = CanonicalHarness(adp, _spec("adaptive_fast", "adaptive_slow", "signal_flip"), WIN, chimera_path=f"oc::{s}")
        try:
            b = benchmark_excess(h_oc)
            fc_eval += 1
            fc_beat += 1 if b["beats_beta_held"] else 0
        except Exception:
            pass

        # F-D: adaptive held-out compound at taker vs ideal cost
        r_taker = h_oc.run()
        h_ideal = CanonicalHarness(adp, _spec("adaptive_fast", "adaptive_slow", "signal_flip", cost=0.0010), WIN,
                                   chimera_path=f"ideal::{s}")
        r_ideal = h_ideal.run()
        fd_taker.append(_comp(_held_nets(r_taker)))
        fd_ideal.append(_comp(_held_nets(r_ideal)))
        fb_pool_adapt += _held_nets(r_taker)

        # F-B: each constituent fixed config, pooled held-out nets
        for (fl, sl, mt) in constit:
            d2 = A.build_fixed_columns(adp, fl, sl, mt)
            h_c = CanonicalHarness(d2, _spec("fix_fast", "fix_slow", "signal_flip", ma_type=mt), WIN,
                                   chimera_path=f"c::{s}")
            fb_pool_by_cfg[(fl, sl, mt)] += _held_nets(h_c.run())

    print("=" * 74)
    print("F-A  EXIT-POLICY SENSITIVITY (time-stop max_hold=20) -- firewall beats-null on held-out:")
    print(f"     {fa_beat}/{fa_eval} assets  (opposite-cross was 0/69 on full u100)")
    print("-" * 74)
    print("F-C  BETA BENCHMARK (beats costless beta-matched hold on OOS+UNSEEN):")
    print(f"     {fc_beat}/{fc_eval} assets")
    print("-" * 74)
    print("F-D  COST SENSITIVITY -- adaptive held-out (OOS+UNSEEN pooled) compound, mean across assets:")
    print(f"     taker 0.0024: {np.mean(fd_taker):+.2f}%   ideal 0.0010: {np.mean(fd_ideal):+.2f}%   "
          f"(median taker {np.median(fd_taker):+.2f}% / ideal {np.median(fd_ideal):+.2f}%)")
    print("-" * 74)
    print("F-B  ADAPTIVE vs CONSTITUENT FIXED CONFIGS -- pooled held-out per-trade expectancy (%):")
    exp_adapt = float(np.mean(fb_pool_adapt) * 100) if fb_pool_adapt else 0.0
    print(f"     ADAPTIVE (switches among them): exp={exp_adapt:+.4f}%  n={len(fb_pool_adapt)}")
    for c in constit:
        pool = fb_pool_by_cfg[c]
        e = float(np.mean(pool) * 100) if pool else 0.0
        wr = float((np.asarray(pool) > 0).mean()) if pool else 0.0
        flag = "  <- adaptive WORSE" if e > exp_adapt else ""
        print(f"     fixed {str(c):>18}: exp={e:+.4f}%  wr={wr:.3f}  n={len(pool)}{flag}")
    print("=" * 74)

    out = {"FA_timestop_firewall": {"beat": fa_beat, "eval": fa_eval},
           "FC_beta_benchmark": {"beat": fc_beat, "eval": fc_eval},
           "FD_cost": {"taker_mean": round(float(np.mean(fd_taker)), 2), "ideal_mean": round(float(np.mean(fd_ideal)), 2),
                       "taker_median": round(float(np.median(fd_taker)), 2), "ideal_median": round(float(np.median(fd_ideal)), 2)},
           "FB_constituents": {"adaptive_exp_pct": round(exp_adapt, 4),
                               "by_cfg": {str(c): round(float(np.mean(fb_pool_by_cfg[c]) * 100), 4) for c in constit}}}
    Path(__file__).resolve().parent.joinpath("falsifiers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[saved] falsifiers.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    main(args.n)
