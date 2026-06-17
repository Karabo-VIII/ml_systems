"""Adversarial probe: reconstruct SYS_A DOWN-regime cells (u10 1d trend) and report
per-cell OOS/UNSEEN sums, n, and the same-direction-bounce (D58) exposure.
Mirrors regime_dna_lab.run() selection exactly. No emoji (cp1252)."""
import os, sys
os.chdir(r"c:/Users/karab/Documents/coding/v4_crypto_stystem")
sys.path.insert(0, "src")
import numpy as np, yaml
from strat.regime_dna_lab import (
    detect_regime, simulate_tagged, split_of, _cfgs, MIN_TRADES,
)
from mining.family_regime_map import _norm_sym
from pipeline.chimera_loader import ChimeraLoader

universe, cadence, regime_mode = "u10", "1d", "trend"
spec = yaml.safe_load(open(f"config/universes/{universe}.yaml"))
syms = [a["symbol"] for a in spec["assets"]]

from strat.regime_dna_lab import OOS_END_MS

book = {}
for sym in syms:
    try:
        df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence,
                                  features=["open", "high", "low", "close"])
    except Exception as e:
        print("SKIP", sym, e); continue
    ts = df["timestamp"].to_numpy()
    if (ts < OOS_END_MS).sum() < 250:
        continue
    o = df["open"].to_numpy().astype(float)
    h = df["high"].to_numpy().astype(float)
    l = df["low"].to_numpy().astype(float)
    c = df["close"].to_numpy().astype(float)
    reg = detect_regime(c, regime_mode)
    split_arr = np.array([split_of(int(t)) for t in ts], dtype=object)
    book[sym] = {cfg: simulate_tagged(cfg[0], cfg[1], o, h, l, c, reg, split_arr)
                 for cfg in _cfgs()}

assets = list(book.keys())
regimes = sorted({tr["regime"] for a in assets for cfg in book[a]
                  for tr in book[a][cfg] if tr["regime"] is not None})

def trnet(sym, cfg, split, regime=None):
    return [tr["net"] for tr in book[sym][cfg]
            if tr["split"] == split and (regime is None or tr["regime"] == regime)]

# SYS_A selection (TRAIN only), exactly as in run()
regime_cfg = {}
for s in assets:
    for r in regimes:
        cand = [(c, np.mean(trnet(s, c, "TRAIN", r))) for c in _cfgs()
                if len(trnet(s, c, "TRAIN", r)) >= MIN_TRADES]
        if cand:
            regime_cfg[(s, r)] = max(cand, key=lambda x: x[1])[0]

print("regimes:", regimes, " n_regime_cells:", len(regime_cfg))
print()

def cfg_str(c):
    return f"{c[0]}/{'stop' if c[1] else 'sig'}"

# Report per-cell, separated by regime
for target_reg in ["DOWN", "UP"]:
    print("=" * 70)
    print(f"{target_reg}-REGIME CELLS")
    print(f"{'asset':10} {'config':18} | {'TRAIN n/sum':>14} | {'OOS n/sum':>14} | {'UNSEEN n/sum':>14}")
    oos_pos = oos_neg = uns_pos = uns_neg = 0
    n_cells = 0
    for (s, r), cfg in sorted(regime_cfg.items()):
        if r != target_reg:
            continue
        n_cells += 1
        tr = trnet(s, cfg, "TRAIN", r)
        oo = trnet(s, cfg, "OOS", r)
        un = trnet(s, cfg, "UNSEEN", r)
        oos_sum = sum(oo); uns_sum = sum(un)
        if len(oo) >= 3:
            if oos_sum > 0:
                oos_pos += 1
            else:
                oos_neg += 1
        if len(un) >= 3:
            if uns_sum > 0:
                uns_pos += 1
            else:
                uns_neg += 1
        print(f"{s:10} {cfg_str(cfg):18} | {len(tr):3} {sum(tr):+8.3f} | "
              f"{len(oo):3} {oos_sum:+8.3f} | {len(un):3} {uns_sum:+8.3f}")
    print(f"  cells={n_cells}  OOS(n>=3): pos={oos_pos} neg={oos_neg}  "
          f"UNSEEN(n>=3): pos={uns_pos} neg={uns_neg}")
    print()
