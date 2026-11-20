"""scripts/probe_rank_persistence_decompose.py -- SETTLE the rank-persistence question.

The fair-test found config-rank Spearman PERSISTS month1->month2 (book_rho up to 0.91, rising with
cadence-fineness). The worker ASSERTED (did not prove) this is "just the slow-MA cost ordering persisting".

This probe DECOMPOSES it. For each (asset, config) it recomputes:
  - m1 = month-1 net compound (%)            (the predictor)
  - m2 = month-2 net compound (%)            (the held-out outcome)
  - turn1 = month-1 trade count (turnover)   (the cost/frequency axis = the confound)
  - maspeed = slowest MA period in the entry spec (a structural turnover proxy, no replay needed)

Then it answers, per cadence x pair:
  A. RAW  rank-rho(m1, m2)          -- reproduce the finding.
  B. SPEED rank-rho(maspeed, m1)    -- is the m1 ranking itself just the MA-speed ordering?
     and rank-rho(maspeed, m2)      -- and does MA-speed alone predict m2?
  C. PARTIAL rank-rho(m1, m2 | turn1)  -- AFTER partialling out month-1 turnover, does m1 rank still
     predict m2 rank? (partial Spearman = corr of the residuals of rank(m1)~rank(turn1) and
     rank(m2)~rank(turn1)). If ~0 => persistence IS the turnover/cost ordering (worker right).
     If still positive => there is regime-specific residual information in the ordering (worker WRONG).
  D. PARTIAL rank-rho(m1, m2 | maspeed)  -- same, controlling for the structural MA-speed axis.

Pooled across assets (the BOOK view, which is the rho the finding quotes) and reported per-asset.

RWYB: python scripts/probe_rank_persistence_decompose.py
No git commit. No emoji.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import strat.config_selector_jan2feb as CS  # noqa: E402
from strat.portfolio_replay import TAKER_RT  # noqa: E402

PAIRS = {
    "bull2024": (("2024-02-01", "2024-03-01"), ("2024-03-01", "2024-04-01")),
    "range2023": (("2023-09-01", "2023-10-01"), ("2023-10-01", "2023-11-01")),
}
CADENCES = ["4h", "1h", "30m", "15m"]


def ma_speed(entry_name):
    """Slowest MA period in an ema_a_b[_c] entry spec. Slower = trades rarely = survives cost."""
    nums = [int(x) for x in entry_name.split("_")[1:]]
    return max(nums)


def rank(x):
    return pd.Series(np.asarray(x, float)).rank().to_numpy()


def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial_spear(x, y, z):
    """Partial Spearman of x,y controlling z = Pearson corr of residuals of rank(x)~rank(z), rank(y)~rank(z)."""
    rx, ry, rz = rank(x), rank(y), rank(z)
    if len(set(rz)) < 2 or np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None

    def resid(r, on):
        on = np.column_stack([np.ones_like(on), on])
        beta, *_ = np.linalg.lstsq(on, r, rcond=None)
        return r - on @ beta
    ex, ey = resid(rx, rz), resid(ry, rz)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return None
    return float(np.corrcoef(ex, ey)[0, 1])


def run_cadence(cadence, syms, entry_specs, configs, train_win, test_win):
    cost = TAKER_RT
    js = pd.Timestamp(train_win[0]).value // 10**6
    je = pd.Timestamp(train_win[1]).value // 10**6
    fs = pd.Timestamp(test_win[0]).value // 10**6
    fe = pd.Timestamp(test_win[1]).value // 10**6

    # per (asset, config): m1, m2 compound + m1 trade count
    rows = []  # (asset, cfg, m1, m2, turn1, maspeed)
    panels = {}
    for sym in syms:
        try:
            o, h, l, c, ms = CS._panel(sym, cadence)
        except Exception:
            continue
        keep = ms < fe
        o, h, l, c, ms = o[keep], h[keep], l[keep], c[keep], ms[keep]
        if ((ms >= js) & (ms < je)).sum() < 5:
            continue
        panels[sym] = (o, h, l, c, ms)
    if len(panels) < 3:
        return None

    for sym, (o, h, l, c, ms) in panels.items():
        for (en, ex) in configs:
            m1c, _, m1n = CS.config_perf(o, h, l, c, ms, en, ex, js, je, cost)
            m2c, _, _ = CS.config_perf(o, h, l, c, ms, en, ex, fs, fe, cost)
            rows.append((sym[:-4], (en, ex), m1c * 100, m2c * 100, m1n, ma_speed(en)))

    df = pd.DataFrame(rows, columns=["asset", "cfg", "m1", "m2", "turn1", "maspeed"])

    # ---- BOOK view: mean per-config compound across assets (the rho the finding quotes) ----
    g = df.groupby("cfg").agg(m1=("m1", "mean"), m2=("m2", "mean"),
                              turn1=("turn1", "mean"), maspeed=("maspeed", "first"))
    raw = spear(g["m1"], g["m2"])
    speed_m1 = spear(g["maspeed"], g["m1"])
    speed_m2 = spear(g["maspeed"], g["m2"])
    turn_m1 = spear(g["turn1"], g["m1"])           # turnover vs m1 (negative expected: more trades = worse)
    part_turn = partial_spear(g["m1"], g["m2"], g["turn1"])
    part_speed = partial_spear(g["m1"], g["m2"], g["maspeed"])

    # ---- are the persistently-TOP configs ALWAYS the slowest MAs? ----
    top5_m1 = list(g.sort_values("m1", ascending=False).head(5).index)
    top5_m2 = list(g.sort_values("m2", ascending=False).head(5).index)
    slowest_specs = set(s for s in entry_specs if ma_speed(s) >= sorted([ma_speed(x) for x in entry_specs])[-2])
    top5_m1_slow_frac = np.mean([k[0] in slowest_specs for k in top5_m1])
    top5_m2_slow_frac = np.mean([k[0] in slowest_specs for k in top5_m2])

    # ---- per-asset partials (does residual survive within-asset?) ----
    per_asset_raw, per_asset_part = [], []
    for a, sub in df.groupby("asset"):
        r = spear(sub["m1"], sub["m2"])
        pt = partial_spear(sub["m1"], sub["m2"], sub["turn1"])
        if r is not None:
            per_asset_raw.append(r)
        if pt is not None:
            per_asset_part.append(pt)

    return {
        "cadence": cadence, "n_assets": len(panels),
        "book_raw_rho": rd(raw),
        "speed_vs_m1_rho": rd(speed_m1),     # is m1 ranking == MA-speed ordering?
        "speed_vs_m2_rho": rd(speed_m2),     # does MA-speed alone predict m2?
        "turnover_vs_m1_rho": rd(turn_m1),
        "partial_rho_given_turnover": rd(part_turn),   # THE settling number
        "partial_rho_given_maspeed": rd(part_speed),
        "mean_per_asset_raw_rho": rd(float(np.mean(per_asset_raw)) if per_asset_raw else None),
        "mean_per_asset_partial_given_turnover": rd(float(np.mean(per_asset_part)) if per_asset_part else None),
        "top5_m1_slowMA_frac": rd(top5_m1_slow_frac), "top5_m2_slowMA_frac": rd(top5_m2_slow_frac),
        "top3_m1_cfgs": [f"{k[0]}|{k[1]}" for k in top5_m1[:3]],
        "top3_m2_cfgs": [f"{k[0]}|{k[1]}" for k in top5_m2[:3]],
    }


def rd(x):
    return round(x, 3) if isinstance(x, float) else x


def main():
    entry_specs, configs = CS.build_config_space(4)
    import yaml
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / "u10.yaml"))
    syms = [x["symbol"] for x in spec["assets"]]

    out = {}
    for pair, (tr, te) in PAIRS.items():
        print(f"\n{'='*92}\n## PAIR {pair}  train {tr}  test {te}\n{'='*92}")
        print(f"{'cad':4} {'rawRho':>7} {'spd~m1':>7} {'spd~m2':>7} {'turn~m1':>8} "
              f"{'PART|turn':>10} {'PART|spd':>9} {'t5m1slow':>9} {'t5m2slow':>9}")
        out[pair] = {}
        for cad in CADENCES:
            r = run_cadence(cad, syms, entry_specs, configs, tr, te)
            if r is None:
                print(f"{cad:4} INSUFFICIENT")
                continue
            out[pair][cad] = r
            print(f"{cad:4} {r['book_raw_rho']:>7} {r['speed_vs_m1_rho']:>7} {r['speed_vs_m2_rho']:>7} "
                  f"{r['turnover_vs_m1_rho']:>8} {r['partial_rho_given_turnover']:>10} "
                  f"{r['partial_rho_given_maspeed']:>9} {r['top5_m1_slowMA_frac']:>9} {r['top5_m2_slowMA_frac']:>9}")
            print(f"     top3 m1: {r['top3_m1_cfgs']}")
            print(f"     top3 m2: {r['top3_m2_cfgs']}")

    p = ROOT / "runs" / "strat" / "rank_persistence_decompose.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
