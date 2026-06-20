"""strat/volbreak_sweep.py -- Volatility Breakout strategy sweep.

STYLE: VOLATILITY BREAKOUT
  1. Range-compression: vol20 < squeeze_thresh * median_vol20  AND  recent range narrow
  2. Entry trigger: C >= hh14  (price breaks to 14d high after a quiet period)
  3. Position management: ATR-trail stop -- once in, exit when C drops k*atr14 below
     the running peak since entry.  Winners run; losers cut fast.

Compare:
  - squeeze thresholds: 0.6, 0.75, 0.9  (how compressed before we look for the break)
  - ATR trail multipliers: 1.5, 2.5, 3.5  (how tight the trail)
  - K (top-K selection by breakout strength): 2, 3, 5

RWYB: python -m strat.volbreak_sweep
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab


# ---------------------------------------------------------------------------
# Core builder: volatility breakout with ATR trail
# ---------------------------------------------------------------------------

def _median_vol(vol20: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling median of vol20 (60d) -- represents the 'normal' vol level."""
    return vol20.rolling(window, min_periods=20).median()


def build_volbreak(
    ind: dict,
    squeeze_thresh: float = 0.75,  # vol20 must be < thresh * rolling_median_vol
    atr_mult: float = 2.5,          # trail = peak_since_entry - k * atr14
    K: int = 3,                     # top-K to hold at any time
    gate: bool = True,              # filter by C > sma200
    rebal: int = 1,                 # check/update every bar
    squeeze_range_pct: float = 0.08, # additional: (hh14-ll14)/ll14 < this for 'narrow'
) -> pd.DataFrame:
    """
    Build weight matrix W for volatility breakout strategy.

    Entry conditions (evaluated on bar d, position takes effect bar d+1 via harness lag):
      1. vol20[d] < squeeze_thresh * rolling_median_vol[d]   (compressed volatility)
      2. (hh14[d] - ll14[d]) / ll14[d] < squeeze_range_pct  (narrow 14d range)  [optional tight]
      3. C[d] >= hh14[d]                                     (breakout trigger)
      4. gate[d] = True  (if gate=True)

    Position management:
      - Track per-asset peak since entry
      - Exit when C[d] < peak - atr_mult * atr14[d]  (ATR trail violated)
      - Rank active + new entries by breakout strength = (C - hh14.shift(1)) / atr14
        (how much above the high, normalized by ATR)
      - Hold top-K of eligible
    """
    C      = ind["C"]
    vol20  = ind["vol20"]
    atr14  = ind["atr14"]
    hh14   = ind["hh14"]
    ll14   = ind["ll14"]
    g      = ind["gate"] if gate else pd.DataFrame(True, index=C.index, columns=C.columns)

    med_vol = _median_vol(vol20)

    syms  = list(C.columns)
    dates = list(C.index)
    N     = len(dates)
    S     = len(syms)
    sym_idx = {s: i for i, s in enumerate(syms)}

    # Convert to numpy for speed
    C_np    = C.to_numpy()
    vol_np  = vol20.to_numpy()
    med_np  = med_vol.to_numpy()
    atr_np  = atr14.to_numpy()
    hh_np   = hh14.to_numpy()
    ll_np   = ll14.to_numpy()
    g_np    = g.to_numpy().astype(bool)

    W_np = np.zeros((N, S), dtype=float)

    # State tracking: peak price since entry, per asset
    peak      = np.full(S, np.nan)   # peak C since position opened
    in_pos    = np.zeros(S, dtype=bool)

    prev_hh = np.full(S, np.nan)  # hh14 from previous bar (for breakout above yesterday's high)

    for i in range(1, N):
        c_row   = C_np[i]
        v_row   = vol_np[i]
        m_row   = med_np[i]
        atr_row = atr_np[i]
        hh_row  = hh_np[i]
        ll_row  = ll_np[i]
        g_row   = g_np[i]
        c_prev  = C_np[i - 1]
        hh_prev = hh_np[i - 1]

        # --- Update trailing stop for existing positions ---
        for j in range(S):
            if in_pos[j]:
                # Update peak
                if not np.isnan(c_row[j]) and (np.isnan(peak[j]) or c_row[j] > peak[j]):
                    peak[j] = c_row[j]
                # Check trail: exit if C < peak - k*ATR
                if not np.isnan(atr_row[j]) and not np.isnan(peak[j]):
                    trail_stop = peak[j] - atr_mult * atr_row[j]
                    if c_row[j] < trail_stop:
                        in_pos[j] = False
                        peak[j] = np.nan
                # Also exit if gate lost (price below sma200)
                if gate and not g_row[j]:
                    in_pos[j] = False
                    peak[j] = np.nan

        # --- Check for new breakout entries ---
        # Squeeze condition: vol compressed vs its own recent median
        squeeze_ok = np.zeros(S, dtype=bool)
        for j in range(S):
            v = v_row[j]; m = m_row[j]
            h = hh_row[j]; l = ll_row[j]
            if np.isnan(v) or np.isnan(m) or np.isnan(h) or np.isnan(l) or l <= 0:
                continue
            vol_squeezed   = v < squeeze_thresh * m
            # range tightness (optional -- use if squeeze_range_pct < 1.0)
            range_narrow   = (h - l) / l < squeeze_range_pct
            squeeze_ok[j]  = vol_squeezed and range_narrow

        # Breakout: C[i] >= hh14[i-1]  (closing above yesterday's 14d high)
        # Use hh_prev because hh14 at bar i includes bar i itself -> use shift(1)
        for j in range(S):
            if squeeze_ok[j] and g_row[j] and not in_pos[j]:
                hp = hh_prev[j]
                if not np.isnan(hp) and not np.isnan(c_row[j]) and c_row[j] >= hp:
                    in_pos[j] = True
                    peak[j] = c_row[j]

        # --- Score entries by breakout strength = (C - hh14_prev) / ATR14 ---
        # Among those in_pos, pick top-K
        scores = []
        for j in range(S):
            if in_pos[j]:
                atr_j = atr_row[j]
                hp_j  = hh_prev[j]
                if np.isnan(atr_j) or atr_j <= 0 or np.isnan(hp_j):
                    scores.append((j, 0.0))
                else:
                    bo_strength = (c_row[j] - hp_j) / atr_j
                    scores.append((j, bo_strength))

        scores.sort(key=lambda x: -x[1])
        top_k = [j for j, _ in scores[:K]]

        if top_k:
            w = 1.0 / len(top_k)
            for j in top_k:
                W_np[i, j] = w

    W = pd.DataFrame(W_np, index=dates, columns=syms)
    return W


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------

def run_sweep(ind: dict) -> list[dict]:
    results = []

    # Parameter grid
    squeeze_threshs = [0.60, 0.75, 0.90]
    atr_mults       = [1.5, 2.5, 3.5]
    Ks              = [2, 3, 5]
    # range filter: tight (8%) vs loose (15%)
    range_pcts      = [0.08, 0.15]

    configs = []
    # Core 3x3 grid with fixed K=3, range=0.08
    for sq in squeeze_threshs:
        for am in atr_mults:
            configs.append(dict(squeeze_thresh=sq, atr_mult=am, K=3, squeeze_range_pct=0.08,
                                label=f"sq{sq}_atr{am}_K3_rng8"))

    # Vary K at best grid center
    for k in Ks:
        configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=k, squeeze_range_pct=0.08,
                            label=f"sq0.75_atr2.5_K{k}_rng8"))

    # Vary range filter
    for rp in range_pcts:
        configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=3, squeeze_range_pct=rp,
                            label=f"sq0.75_atr2.5_K3_rng{int(rp*100)}"))

    # Looser squeeze for comparison (no range filter -- very wide)
    configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=3, squeeze_range_pct=0.25,
                        label="sq0.75_atr2.5_K3_rng25_loose"))
    # No range filter at all (pure vol-compression-only squeeze)
    configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=3, squeeze_range_pct=1.0,
                        label="sq0.75_atr2.5_K3_novol_rngOFF"))
    configs.append(dict(squeeze_thresh=0.60, atr_mult=2.5, K=3, squeeze_range_pct=1.0,
                        label="sq0.60_atr2.5_K3_rngOFF"))
    configs.append(dict(squeeze_thresh=0.90, atr_mult=2.5, K=3, squeeze_range_pct=1.0,
                        label="sq0.90_atr2.5_K3_rngOFF"))
    # vary ATR with range off
    configs.append(dict(squeeze_thresh=0.75, atr_mult=1.5, K=3, squeeze_range_pct=1.0,
                        label="sq0.75_atr1.5_K3_rngOFF"))
    configs.append(dict(squeeze_thresh=0.75, atr_mult=3.5, K=3, squeeze_range_pct=1.0,
                        label="sq0.75_atr3.5_K3_rngOFF"))
    # vary K with range off
    configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=2, squeeze_range_pct=1.0,
                        label="sq0.75_atr2.5_K2_rngOFF"))
    configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=5, squeeze_range_pct=1.0,
                        label="sq0.75_atr2.5_K5_rngOFF"))

    # No-gate version of best area
    configs.append(dict(squeeze_thresh=0.75, atr_mult=2.5, K=3, squeeze_range_pct=0.08,
                        label="sq0.75_atr2.5_K3_nogate"))

    # Deduplicate labels
    seen = set()
    unique_configs = []
    for c in configs:
        if c["label"] not in seen:
            seen.add(c["label"])
            unique_configs.append(c)

    for cfg in unique_configs:
        lbl = cfg["label"]
        gate = "nogate" not in lbl
        print(f"  running {lbl} ...", flush=True)
        W = build_volbreak(
            ind,
            squeeze_thresh=cfg["squeeze_thresh"],
            atr_mult=cfg["atr_mult"],
            K=cfg["K"],
            gate=gate,
            squeeze_range_pct=cfg["squeeze_range_pct"],
        )
        m = lab.evaluate(W, ind, H=3, label=lbl)
        results.append(m)

    return results


def print_table(results: list[dict]) -> None:
    hdr = (f"{'Config':<40} | {'2020':>7} | {'2021':>7} | {'2022':>7} | {'Full':>7} "
           f"| {'maxDD':>6} | {'gr21':>5} | {'grAll':>5} | {'expo':>5}")
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        print(
            f"{m['label']:<40} | {str(m.get('comp_2020','?')):>7} | {str(m.get('comp_2021','?')):>7}"
            f" | {str(m.get('comp_2022','?')):>7} | {str(m.get('comp_full','?')):>7}"
            f" | {str(m.get('maxDD','?')):>6} | {str(m.get('green_2021','?')):>5}"
            f" | {str(m.get('green_all','?')):>5} | {str(m.get('avg_expo','?')):>5}"
        )


def diagnose_squeeze(ind: dict) -> None:
    """Print how often squeeze conditions fire across the grid."""
    C = ind["C"]
    vol20 = ind["vol20"]
    hh14 = ind["hh14"]
    ll14 = ind["ll14"]
    med_vol = _median_vol(vol20)

    for sq in [0.60, 0.75, 0.90]:
        for rng in [0.08, 0.15, 0.25]:
            vol_sq = (vol20 < sq * med_vol)
            rng_ok = ((hh14 - ll14) / ll14.replace(0, np.nan) < rng)
            both = (vol_sq & rng_ok).fillna(False)
            # breakout on top
            hh_prev = hh14.shift(1)
            bo = (C >= hh_prev)
            trigger = both & bo
            rate = trigger.stack().mean() * 100
            days_with_any = (trigger.any(axis=1)).mean() * 100
            print(f"  sq={sq} rng={rng}: squeeze_rate={rate:.2f}%  days_with_any_trigger={days_with_any:.1f}%")


def main():
    print("Loading data ...")
    ind = lab.load()
    print(f"  assets: {list(ind['C'].columns)}")
    print(f"  dates:  {ind['C'].index[0].date()} -> {ind['C'].index[-1].date()}")
    print()

    print("DIAGNOSTIC -- squeeze fire rates:")
    diagnose_squeeze(ind)
    print()

    # Reference: gated-beta
    print("Running reference (gated-beta) ...")
    beta_W = ind["gate"].astype(float).div(
        ind["gate"].sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)
    ref = lab.evaluate(beta_W, ind, H=3, label="[REF] gated-beta-EW")

    print()
    print("Running volatility breakout sweep ...")
    results = run_sweep(ind)

    print()
    print("=" * 110)
    print("RESULTS TABLE -- Volatility Breakout Sweep (H=3 checkpoint blocks)")
    print("=" * 110)
    all_rows = [ref] + results
    print_table(all_rows)

    # Find best by comp_full and best by green_all
    res_only = [r for r in results if r.get("comp_full") is not None]
    best_comp = max(res_only, key=lambda r: r["comp_full"])
    best_green = max(res_only, key=lambda r: (r.get("green_all") or 0))
    best_2021  = max(res_only, key=lambda r: (r.get("comp_2021") or -9999))

    print()
    print("BEST by comp_full   :", best_comp["label"],
          f"  full={best_comp['comp_full']}%  2021={best_comp.get('comp_2021')}%  maxDD={best_comp['maxDD']}%  green21={best_comp.get('green_2021')}%")
    print("BEST by green_all   :", best_green["label"],
          f"  green_all={best_green.get('green_all')}%  full={best_green['comp_full']}%  maxDD={best_green['maxDD']}%")
    print("BEST by comp_2021   :", best_2021["label"],
          f"  2021={best_2021.get('comp_2021')}%  full={best_2021['comp_full']}%  maxDD={best_2021['maxDD']}%")

    # Greedy = highest exposure * highest comp_2021 (most market time during bull)
    greedy = max(res_only, key=lambda r: (r.get("avg_expo") or 0) * max(r.get("comp_2021") or 0, 0))
    print("GREEDIEST config    :", greedy["label"],
          f"  expo={greedy['avg_expo']}  2021={greedy.get('comp_2021')}%  full={greedy['comp_full']}%")

    print()
    print("REGIME BREAKDOWN:")
    print(f"  Reference (gated-beta): 2020={ref['comp_2020']}% 2021={ref['comp_2021']}% 2022={ref['comp_2022']}% full={ref['comp_full']}%")
    for r in res_only:
        wins = []
        if (r.get("comp_2021") or -9999) > (ref.get("comp_2021") or 0): wins.append("BEATS-beta-2021")
        if (r.get("comp_2022") or -9999) > (ref.get("comp_2022") or 0): wins.append("BEATS-beta-2022")
        if (r.get("green_all") or 0) > 55: wins.append("green>55%")
        if wins:
            print(f"  {r['label']:<42}: {', '.join(wins)}")


if __name__ == "__main__":
    main()
