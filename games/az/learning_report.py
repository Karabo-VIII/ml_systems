"""
chess_zero.az.learning_report -- quantify "is it learning, how fast, and is the curve parabolic?"

Reads a run's strength_curve.json + train.log and reports the things the user actually asked about:
  - SPEED: iters completed, mean iter wall-time, iters/hour, mean self-play games/sec (from the [selfplay-pool]/
    timing lines) -- so "faster learning" is a number, not a vibe.
  - LEARNING: the champion floor (winrate vs random) -- monotonic non-decreasing is the gate guarantee.
  - PARABOLIC: the slope of the watchable strength axes over iters. We fit the draw-aware score-vs-classical and
    winrate-vs-random vs iter and report the trend (rising / flat / falling) + whether the SECOND difference is
    positive (accelerating = parabolic) on the axis that is moving. Honest: with a flat axis we say FLAT, not parabolic.

Pure stdlib (no numpy needed for the small fit). No emoji (Windows cp1252).
Run:  .venv/Scripts/python.exe -m az.learning_report --ckpt-dir robust_fast
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_curve(curve_path: str) -> list[dict]:
    if not os.path.exists(curve_path):
        return []
    try:
        d = json.loads(Path(curve_path).read_text(encoding="utf-8"))
        return d if isinstance(d, list) else [d]
    except Exception:
        return []


def _parse_timing(train_log: str) -> dict:
    """Pull per-iter timing + selfplay-pool throughput from the trainer log."""
    iters, selfplay_s, iter_s, pool_games, pool_s = [], [], [], [], []
    if os.path.exists(train_log):
        txt = Path(train_log).read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"selfplay=([\d.]+)s train=[\d.]+s eval=[\d.]+s iter=([\d.]+)s", txt):
            selfplay_s.append(float(m.group(1)))
            iter_s.append(float(m.group(2)))
        for m in re.finditer(r"\[selfplay-pool\] (\d+) games via \d+ workers in ([\d.]+)s", txt):
            pool_games.append(int(m.group(1)))
            pool_s.append(float(m.group(2)))
    return {"selfplay_s": selfplay_s, "iter_s": iter_s, "pool_games": pool_games, "pool_s": pool_s}


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs xs (per unit x). 0 if <2 points or zero variance."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _trend(slope: float, eps: float = 1e-3) -> str:
    return "RISING" if slope > eps else "FALLING" if slope < -eps else "FLAT"


def _parabolic(xs: list[float], ys: list[float]) -> bool:
    """PARABOLIC = genuinely improving with acceleration: the overall trend must be RISING (slope>0) AND the mean
    second-difference > 0 (bending up). A FALLING curve that happens to have a positive second difference is NOT
    parabolic -- it is decelerating decline; reporting it as parabolic would be a lie."""
    if len(ys) < 3:
        return False
    if _slope(xs, ys) <= 1e-3:          # not even rising -> not parabolic, full stop
        return False
    second = [ys[i + 1] - 2 * ys[i] + ys[i - 1] for i in range(1, len(ys) - 1)]
    return (sum(second) / len(second)) > 1e-4


def report(ckpt_dir: str) -> dict:
    base = ckpt_dir if os.path.isabs(ckpt_dir) else str(HERE / ckpt_dir)
    curve = _load_curve(os.path.join(base, "strength_curve.json"))
    timing = _parse_timing(os.path.join(base, "train.log"))

    iters = [r.get("iter") for r in curve if r.get("iter") is not None]
    vr = [r.get("winrate_vs_random") for r in curve]
    vs = [r.get("score_vs_classical_d1") for r in curve]
    champ = [r.get("champion_winrate_vs_random") for r in curve]

    def _pairs(ys):
        xs2, ys2 = [], []
        for i, y in zip(iters, ys):
            if y is not None:
                xs2.append(float(i)); ys2.append(float(y))
        return xs2, ys2

    xr, yr = _pairs(vr)
    xs_, ys_ = _pairs(vs)

    mean_iter = sum(timing["iter_s"]) / len(timing["iter_s"]) if timing["iter_s"] else 0.0
    iters_per_hr = 3600.0 / mean_iter if mean_iter else 0.0
    gps = (sum(timing["pool_games"]) / sum(timing["pool_s"])) if timing["pool_s"] else 0.0

    # DIVERSITY (2026-06-09): the self-play health the prior 111-iter run was BLIND to.
    # A collapsed-openings run (every game from the same start -> the net reinforces one
    # rote line) used to look identical to a healthy run on loss/win-rate; this surfaces it.
    div_ratio = []  # distinct_starts / games per iter (1.0 = every game a fresh opening)
    decisive = []
    for r in curve:
        ng = r.get("selfplay_games")
        ds = r.get("selfplay_distinct_starts")
        if ng:
            div_ratio.append((ds or 0) / ng)
        if r.get("selfplay_decisive_frac") is not None:
            decisive.append(float(r["selfplay_decisive_frac"]))
    mean_div = (sum(div_ratio) / len(div_ratio)) if div_ratio else None
    mean_decisive = (sum(decisive) / len(decisive)) if decisive else None
    opening_mode = curve[-1].get("opening_mode") if curve else None
    # DEAD if openings were supposed to vary but most games collapsed onto the same start
    # (mean distinct/games < 0.5 = more than half the games duplicate another's opening).
    diversity_dead = (mean_div is not None and mean_div < 0.5 and
                      opening_mode not in (None, "startpos"))

    out = {
        "iters_completed": len(iters),
        "mean_iter_s": round(mean_iter, 1),
        "iters_per_hour": round(iters_per_hr, 1),
        "selfplay_games_per_s": round(gps, 3),
        "champion_floor_vs_random": champ[-1] if champ else None,
        "champion_floor_monotonic": all((champ[i] or 0) <= (champ[i + 1] or 0) for i in range(len(champ) - 1)) if champ else None,
        "vs_random_trend": _trend(_slope(xr, yr)), "vs_random_slope_per_iter": round(_slope(xr, yr), 4),
        "score_vs_classical_trend": _trend(_slope(xs_, ys_)), "score_vs_classical_slope_per_iter": round(_slope(xs_, ys_), 4),
        "score_vs_classical_parabolic(rising+accelerating)": _parabolic(xs_, ys_),
        "opening_mode": opening_mode,
        "selfplay_diversity_ratio": round(mean_div, 3) if mean_div is not None else None,
        "selfplay_decisive_frac": round(mean_decisive, 3) if mean_decisive is not None else None,
        "selfplay_diversity_dead": diversity_dead,
        "latest": curve[-1] if curve else None,
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="robust_fast", help="run dir (under az/ or absolute) with strength_curve.json + train.log")
    a = ap.parse_args(argv)
    r = report(a.ckpt_dir)
    print("=" * 78)
    print(f"LEARNING REPORT -- {a.ckpt_dir}")
    print("=" * 78)
    print(f"  SPEED     : {r['iters_completed']} iters | mean {r['mean_iter_s']}s/iter | "
          f"{r['iters_per_hour']} iters/hr | self-play {r['selfplay_games_per_s']} games/s")
    print(f"  LEARNING  : champion floor vs random = {r['champion_floor_vs_random']} "
          f"(monotonic={r['champion_floor_monotonic']})")
    print(f"  vs RANDOM : {r['vs_random_trend']} (slope {r['vs_random_slope_per_iter']}/iter)")
    print(f"  vs CLASSIC: {r['score_vs_classical_trend']} (draw-aware slope "
          f"{r['score_vs_classical_slope_per_iter']}/iter, parabolic(rising+accel)="
          f"{r['score_vs_classical_parabolic(rising+accelerating)']})")
    div = r['selfplay_diversity_ratio']
    print(f"  DIVERSITY : opening_mode={r['opening_mode']} | distinct-starts/games="
          f"{div if div is not None else 'n/a'} | decisive={r['selfplay_decisive_frac']}"
          + ("  *** DIVERSITY DEAD -- self-play collapsed onto one opening ***"
             if r['selfplay_diversity_dead'] else ""))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
