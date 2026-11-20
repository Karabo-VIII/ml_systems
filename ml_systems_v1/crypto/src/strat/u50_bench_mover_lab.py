"""u50_bench_mover_lab.py -- META-FOLD CYCLE 2: widen the bench to u50 and re-test Cycle-1 survivors.

DELIVERABLE: does a deeper bench (u50 vs u10) improve the momentum edge?

APPROACH:
  - Build a u50 loader that mirrors mover_lab.load() but loads each sym via _panel (ChimeraLoader).
  - Reuse mover_lab.evaluate() and mover_lab.topk_weight() unchanged.
  - Test the Cycle-1 survivors on u50:
      (A) mom14 top-K for K in {5, 10}
      (B) breakout top-K for K in {5, 10}
  - Report per-year 2020/2021/2022/2023/2024/2025 + full.
  - Coverage manifest: which u50 names have daily data and how many bars per year.

CAUSAL: W.loc[d] uses only ind[...].loc[d] or earlier.
COST  : mover_lab taker cost on |dpos|.
RWYB  : python -m strat.u50_bench_mover_lab
No emoji (cp1252). No git commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml
import strat.ma_strat_builder as msb
from strat.ma_per_instrument import _panel   # ChimeraLoader-backed, canonical

# ---------------------------------------------------------------------------
# U50 candidate universe (from u50_subdaily plot directory + u10 base)
# ---------------------------------------------------------------------------
U50_SYMS = [
    # u10 base (always present)
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
    "LTCUSDT", "DOTUSDT", "BCHUSDT", "UNIUSDT",
    # extended u50 candidates (from plots directory)
    "AAVEUSDT", "ALGOUSDT", "APTUSDT", "ARBUSDT", "BLURUSDT",
    "BNBUSDT",  # already above -- deduplicated below
    "BONKUSDT", "CRVUSDT", "DASHUSDT", "ENAUSDT", "ENJUSDT",
    "ETCUSDT", "FETUSDT", "FILUSDT", "HBARUSDT", "ICPUSDT",
    "JSTUSDT", "LDOUSDT", "NEARUSDT", "OPUSDT", "ORDIUSDT",
    "PENGUUSDT", "PEPEUSDT", "RENDERUSDT", "SEIUSDT", "SHIBUSDT",
    "SUIUSDT", "SUPERUSDT", "TAOUSDT", "TONUSDT", "TREEUSDT",
    "TRUMPUSDT", "WIFUSDT", "WLDUSDT", "ZECUSDT",
]
# deduplicate preserving order
_seen = set()
U50_SYMS = [s for s in U50_SYMS if not (s in _seen or _seen.add(s))]

START = "2020-01-01"
END   = "2026-06-01"


def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def load_u50(start=START, end=END):
    """Build ind dict for all u50 symbols with available daily data.

    Each symbol is loaded via _panel (ChimeraLoader 1d), aligned to a common
    date index, then indicators computed.  Missing bars filled 0 (=cash).
    Returns (ind, coverage) where coverage is a DataFrame showing bars/year.
    """
    panels = {}
    coverage_rows = []
    for sym in U50_SYMS:
        try:
            o, h, l, c, ms = _panel(sym, "1d")
        except Exception as ex:
            print(f"  SKIP {sym}: {ex}")
            continue
        s_ms = int(pd.Timestamp(start).value // 10**6)
        e_ms = int(pd.Timestamp(end).value   // 10**6)
        idx  = np.searchsorted(ms, s_ms)
        eidx = np.searchsorted(ms, e_ms)
        if eidx - idx < 30:
            print(f"  SKIP {sym}: only {eidx - idx} bars in window")
            continue
        dates = pd.to_datetime(ms[idx:eidx], unit="ms").normalize()
        c_s = pd.Series(c[idx:eidx], index=dates, name=sym)
        # deduplicate (intraday artefacts)
        c_s = c_s[~c_s.index.duplicated(keep="last")]
        panels[sym] = c_s
        # coverage
        yr = dates.year.value_counts().sort_index()
        row = {"sym": sym, "total_bars": len(c_s)}
        for y in [2020, 2021, 2022, 2023, 2024, 2025]:
            row[f"bars_{y}"] = int(yr.get(y, 0))
        coverage_rows.append(row)

    if not panels:
        raise RuntimeError("No symbols loaded -- check data paths")

    all_dates = pd.date_range(start=start, end=pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    C = pd.DataFrame({s: panels[s].reindex(all_dates) for s in panels})
    # forward-fill intraweek gaps (weekends etc.) up to 3 bars, then leave NaN
    C = C.fillna(method="ffill", limit=3)
    R = C.pct_change()

    tr_h = C.rolling(14, min_periods=14).max()  # proxy for h
    tr_l = C.rolling(14, min_periods=14).min()  # proxy for l
    ind = {
        "C": C, "R": R,
        "sma200": C.rolling(200, min_periods=200).mean(),
        "sma50":  C.rolling(50, min_periods=50).mean(),
        "mom14":  C / C.shift(14) - 1,
        "mom7":   C / C.shift(7) - 1,
        "mom30":  C / C.shift(30) - 1,
        "rsi14":  C.apply(_rsi),
        "ret1":   R,
        "hh14":   tr_h,
        "ll14":   tr_l,
        "vol20":  R.rolling(20, min_periods=10).std() * np.sqrt(365),
        "atr14":  (tr_h - tr_l).rolling(14, min_periods=14).mean(),
    }
    ind["gate"] = (C > ind["sma200"]).fillna(False)

    cov = pd.DataFrame(coverage_rows).set_index("sym")
    return ind, cov


# ---------------------------------------------------------------------------
# Breakout signal (mirrors Cycle-1 definition: 14d high breakout)
# ---------------------------------------------------------------------------
def breakout_score(ind):
    """Score = (C - hh14_prev) / hh14_prev; positive = breaking out above prior 14d high."""
    hh_prev = ind["hh14"].shift(1)
    return (ind["C"] - hh_prev) / (hh_prev.replace(0, np.nan))


# ---------------------------------------------------------------------------
# EVALUATE wrapper that adds per-year columns
# ---------------------------------------------------------------------------
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def evaluate_full(W, ind, label=""):
    """Evaluate + add per-year compound columns (mover_lab.evaluate covers 2020-2022 only)."""
    base = ml.evaluate(W, ind, H=3, label=label)
    # extend with 2023/2024/2025
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (ml.COST / 2.0)
    x = bret.to_numpy()
    def comp(s, e):
        mask = np.asarray((bret.index >= s) & (bret.index < e))
        xs = x[mask]
        return round((np.prod(1 + xs) - 1) * 100, 1) if mask.sum() > 2 else None
    for y in [2023, 2024, 2025]:
        base[f"comp_{y}"] = comp(f"{y}-01-01", f"{y+1}-01-01")
    return base


# ---------------------------------------------------------------------------
# SHUFFLE CONTROL: permute ROWS of score to assess luck
# ---------------------------------------------------------------------------
def shuffle_control(score, ind, K, n_shuffles=200, rebal=3, gate=True, seed=42):
    rng = np.random.default_rng(seed)
    comps = []
    sc = score.copy()
    for _ in range(n_shuffles):
        sc_shuf = pd.DataFrame(
            rng.permutation(sc.to_numpy()),
            index=sc.index, columns=sc.columns,
        )
        W = ml.topk_weight(sc_shuf, ind, K=K, rebal=rebal, gate=gate)
        res = evaluate_full(W, ind, label="shuf")
        comps.append(res["comp_full"])
    return np.array(comps)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("META-FOLD CYCLE 2: u50 bench-widening test")
    print(f"Candidate universe: {len(U50_SYMS)} symbols")
    print(f"Window: {START} -> {END}")
    print("=" * 70)

    print("\n[1] Loading u50 daily data ...")
    ind, cov = load_u50()
    loaded_syms = list(ind["C"].columns)
    print(f"    Loaded {len(loaded_syms)} symbols: {loaded_syms}")

    print("\n[2] Coverage manifest (bars per year):")
    print(cov.to_string())

    print("\n[3] Running strategies ...")
    results = []

    # ---- mom14 K5 rebal3 (Cycle-1 winner) ----
    for K in [5, 10]:
        W = ml.topk_weight(ind["mom14"], ind, K=K, rebal=3, gate=True)
        res = evaluate_full(W, ind, label=f"mom14_K{K}_r3")
        results.append(res)
        print(f"    mom14 K={K} r=3: {res['comp_2020']}% / {res['comp_2021']}% / {res['comp_2022']}% / "
              f"{res.get('comp_2023')}% / {res.get('comp_2024')}% / {res.get('comp_2025')}% "
              f"| full={res['comp_full']}% maxDD={res['maxDD']}%")

    # ---- mom14 K5 rebal7 (Cycle-1 variant) ----
    for K in [5, 10]:
        W = ml.topk_weight(ind["mom14"], ind, K=K, rebal=7, gate=True)
        res = evaluate_full(W, ind, label=f"mom14_K{K}_r7")
        results.append(res)
        print(f"    mom14 K={K} r=7: {res['comp_2020']}% / {res['comp_2021']}% / {res['comp_2022']}% / "
              f"{res.get('comp_2023')}% / {res.get('comp_2024')}% / {res.get('comp_2025')}% "
              f"| full={res['comp_full']}% maxDD={res['maxDD']}%")

    # ---- breakout top-K (Cycle-1 winner) ----
    brk = breakout_score(ind)
    for K in [5, 10]:
        W = ml.topk_weight(brk, ind, K=K, rebal=5, gate=True)
        res = evaluate_full(W, ind, label=f"breakout_K{K}_r5")
        results.append(res)
        print(f"    breakout K={K} r=5: {res['comp_2020']}% / {res['comp_2021']}% / {res['comp_2022']}% / "
              f"{res.get('comp_2023')}% / {res.get('comp_2024')}% / {res.get('comp_2025')}% "
              f"| full={res['comp_full']}% maxDD={res['maxDD']}%")

    # ---- gated-beta baseline (EW of gated) ----
    beta_w = ind["gate"].astype(float)
    beta_w = beta_w.div(beta_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    res_beta = evaluate_full(beta_w, ind, label="gated_beta_EW")
    results.append(res_beta)
    print(f"    gated-beta EW: {res_beta['comp_2020']}% / {res_beta['comp_2021']}% / {res_beta['comp_2022']}% / "
          f"{res_beta.get('comp_2023')}% / {res_beta.get('comp_2024')}% / {res_beta.get('comp_2025')}% "
          f"| full={res_beta['comp_full']}% maxDD={res_beta['maxDD']}%")

    # ---- shuffle controls for mom14 K5 r3 ----
    print("\n[4] Shuffle controls for mom14 K5 r3 (n=200) ...")
    null_dist = shuffle_control(ind["mom14"], ind, K=5, n_shuffles=200, rebal=3, seed=42)
    live_comp = [r["comp_full"] for r in results if r["label"] == "mom14_K5_r3"][0]
    p_val = float(np.mean(null_dist >= live_comp))
    print(f"    mom14 K5 r3 live={live_comp}%  null_mean={null_dist.mean():.1f}%  "
          f"null_p05={np.percentile(null_dist,5):.1f}%  p_live={p_val:.3f}")

    # ---- u10 reference (original mover_lab.load) for comparison ----
    print("\n[5] u10 reference (mover_lab.load) for apples-to-apples ...")
    ind_u10 = ml.load(start=START, end=END)
    for K in [5, 10]:
        W10 = ml.topk_weight(ind_u10["mom14"], ind_u10, K=K, rebal=3, gate=True)
        r10 = evaluate_full(W10, ind_u10, label=f"u10_mom14_K{K}_r3")
        results.append(r10)
        print(f"    [u10] mom14 K={K} r=3: {r10['comp_2020']}% / {r10['comp_2021']}% / {r10['comp_2022']}% / "
              f"{r10.get('comp_2023')}% / {r10.get('comp_2024')}% / {r10.get('comp_2025')}% "
              f"| full={r10['comp_full']}% maxDD={r10['maxDD']}%")

    # ---- MARKDOWN TABLE ----
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    cols = ["label", "comp_2020", "comp_2021", "comp_2022", "comp_2023", "comp_2024", "comp_2025",
            "comp_full", "maxDD", "avg_expo", "avg_turnover"]
    header = "| Strategy | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Full | maxDD | expo | turn |"
    sep    = "|" + "|".join(["---"] * (len(cols))) + "|"
    print(header)
    print(sep)
    for r in results:
        row = "| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |"
        print(row)

    print("\n[6] Verdict ...")
    mom14_u50_full = [r["comp_full"] for r in results if r["label"] == "mom14_K5_r3"]
    mom14_u10_full = [r["comp_full"] for r in results if r["label"] == "u10_mom14_K5_r3"]
    if mom14_u50_full and mom14_u10_full:
        delta = mom14_u50_full[0] - mom14_u10_full[0]
        print(f"    u50 mom14 K5 vs u10 mom14 K5 (full): {delta:+.1f}pp")
        if delta > 10:
            print("    VERDICT: wider bench IMPROVES momentum (more dispersion helps)")
        elif delta > 0:
            print("    VERDICT: wider bench modestly helps -- marginal improvement")
        else:
            print("    VERDICT: wider bench does NOT improve momentum -- u10 sufficient or oversaturation")
    print(f"\n    Shuffle p-val (mom14 K5 r3 vs random-gated-5): {p_val:.3f}")
    if p_val < 0.05:
        print("    GENUINE: momentum on u50 is statistically real (p < 0.05)")
    elif p_val < 0.10:
        print("    MARGINAL: p < 0.10, borderline (noise cannot be excluded)")
    else:
        print("    NOT SIGNIFICANT: mom14 u50 K5 is consistent with luck at this horizon")

    print("\nDone.")


if __name__ == "__main__":
    main()
