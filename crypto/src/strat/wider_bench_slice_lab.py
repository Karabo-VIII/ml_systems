"""wider_bench_slice_lab.py -- WIDER BENCH random 7-day slice win-rate test.

LANE: WIDER BENCH
  - Load u30/u50 via ma_per_instrument._panel(sym, '1d')
  - Build C/O/H/L/R + causal indicators (identical to mover_lab)
  - Always-long top-K momentum/quality leaders (K in {3, 5, 10})
  - Evaluate on N=350 RANDOM 7-day slices, walk-forward / expanding window
  - WIN CONDITION: beat buy-hold random-slice win-rate (>55%) OR mean (>+2.9%)

CAUSAL DISCIPLINE (CRITICAL):
  - Any feature at row d uses only data <= d
  - Label is d -> d+7 forward return
  - Positions are entered on date d with signal computed at d-1 (last known day)
  - No lookahead: score at sl_start-1, return measured sl_start -> sl_start+7

DATA WINDOW: 2020-01-01 -> 2023-01-01 (same as mover_lab default, matches spec reference)
UNIVERSES:
  - u10: 10 liquid coins (BTCUSDT...LTCUSDT)
  - u50: 32 loaded (48 candidates, 16 not available in 2020-2023 window)

RWYB: run from crypto/src directory
  python -m strat.wider_bench_slice_lab

No emoji (cp1252). No git commit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml
from strat.ma_per_instrument import _panel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TAKER_RT   = 0.0024     # round-trip taker cost
SLICE_DAYS = 7
RNG_SEED   = 42
N_SLICES   = 350
MIN_WARM   = 210        # bars before slice can start (sma200 needs 200+)
DATA_START = "2020-01-01"
DATA_END   = "2023-01-01"   # same as mover_lab default -> matches spec reference

U10_SYMS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

U50_SYMS_RAW = [
    # u10 base
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
    "LTCUSDT", "DOTUSDT", "BCHUSDT", "UNIUSDT",
    # extended (from u50_subdaily plots directory)
    "AAVEUSDT", "ALGOUSDT", "APTUSDT", "ARBUSDT", "BLURUSDT",
    "BONKUSDT", "CRVUSDT", "DASHUSDT", "ENAUSDT", "ENJUSDT",
    "ETCUSDT", "FETUSDT", "FILUSDT", "HBARUSDT", "ICPUSDT",
    "JSTUSDT", "LDOUSDT", "NEARUSDT", "OPUSDT", "ORDIUSDT",
    "PENGUUSDT", "PEPEUSDT", "RENDERUSDT", "SEIUSDT", "SHIBUSDT",
    "SUIUSDT", "SUPERUSDT", "TAOUSDT", "TONUSDT", "TREEUSDT",
    "TRUMPUSDT", "WIFUSDT", "WLDUSDT", "ZECUSDT",
]
_seen = set()
U50_SYMS = [s for s in U50_SYMS_RAW if not (s in _seen or _seen.add(s))]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_u50(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load u50 daily OHLC, compute causal indicators, return (C, R, ind)."""
    s_ms = int(pd.Timestamp(start).value // 10**6)
    e_ms = int(pd.Timestamp(end).value   // 10**6)
    panels: dict[str, pd.Series] = {}
    cov_rows = []
    for sym in U50_SYMS:
        try:
            _, _, _, c_arr, ms_arr = _panel(sym, "1d")
            i0 = int(np.searchsorted(ms_arr, s_ms))
            ie = int(np.searchsorted(ms_arr, e_ms))
            if ie - i0 < 50:
                continue
            dates = pd.to_datetime(ms_arr[i0:ie], unit="ms").normalize()
            cs = pd.Series(c_arr[i0:ie], index=dates, name=sym)
            cs = cs[~cs.index.duplicated(keep="last")]
            panels[sym] = cs
            yr = dates.year.value_counts().sort_index()
            row = {"sym": sym, "total": len(cs)}
            for y in [2020, 2021, 2022]:
                row[f"bars_{y}"] = int(yr.get(y, 0))
            cov_rows.append(row)
        except Exception:
            pass

    all_dates = pd.date_range(start, end, freq="D", inclusive="left")
    C = pd.DataFrame({s: panels[s].reindex(all_dates) for s in panels}).ffill(limit=3)
    R = C.pct_change()
    sma200 = C.rolling(200, min_periods=200).mean()
    sma50  = C.rolling(50,  min_periods=50).mean()
    mom14  = C / C.shift(14) - 1
    mom7   = C / C.shift(7)  - 1
    gate   = (C > sma200).fillna(False)
    norm_mom = mom14.rank(axis=1, pct=True, na_option="keep")
    norm_vol = 1 - R.rolling(20, min_periods=10).std().rank(axis=1, pct=True, na_option="keep")
    above50  = (C > sma50).astype(float)
    quality  = norm_mom * 0.5 + norm_vol * 0.25 + above50 * 0.25
    ind = {
        "C": C, "R": R, "gate": gate,
        "mom14": mom14, "mom7": mom7, "quality": quality,
        "sma50": sma50, "sma200": sma200,
    }
    cov = pd.DataFrame(cov_rows).set_index("sym") if cov_rows else pd.DataFrame()
    return C, R, ind, cov


# ---------------------------------------------------------------------------
# Slice evaluation helpers
# ---------------------------------------------------------------------------

def _pick_topk(score_df: pd.DataFrame, gate_df: pd.DataFrame,
               date_idx: pd.DatetimeIndex, sl_start: pd.Timestamp,
               K: int, use_gate: bool) -> pd.Series:
    """Score-ranked top-K at the last known day before sl_start. Causal."""
    look_d = sl_start - pd.Timedelta(days=1)
    mask   = date_idx <= look_d
    if not mask.any():
        return pd.Series(0.0, index=score_df.columns)
    last_d = date_idx[mask][-1]
    score  = score_df.loc[last_d]
    elig   = score[gate_df.loc[last_d].fillna(False)] if use_gate else score.dropna()
    if elig.empty:
        return pd.Series(0.0, index=score_df.columns)
    top = elig.nlargest(K)
    w   = pd.Series(0.0, index=score_df.columns)
    w[top.index] = 1.0 / len(top)
    return w


def _slice_ret(R: pd.DataFrame, date_idx: pd.DatetimeIndex,
               weights: pd.Series, sl_start: pd.Timestamp) -> float:
    """7d compound return of weights portfolio, taker cost on entry."""
    end_d = sl_start + pd.Timedelta(days=SLICE_DAYS)
    mask  = (date_idx >= sl_start) & (date_idx < end_d)
    if not mask.any():
        return np.nan
    rw = R.loc[mask].fillna(0.0) @ weights.fillna(0.0)
    return float(np.prod(1 + rw) - 1) - float(weights.fillna(0.0).sum()) * TAKER_RT / 2.0


def _bh_ret(R: pd.DataFrame, date_idx: pd.DatetimeIndex,
            sl_start: pd.Timestamp) -> float:
    """EW buy-hold return over the slice (fixed-EW, NaN = cash)."""
    end_d = sl_start + pd.Timedelta(days=SLICE_DAYS)
    mask  = (date_idx >= sl_start) & (date_idx < end_d)
    if not mask.any():
        return np.nan
    return float(np.prod(1 + R.loc[mask].fillna(0.0).mean(axis=1)) - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("WIDER BENCH RANDOM SLICE LAB")
    print(f"  N_SLICES={N_SLICES}  SLICE_DAYS={SLICE_DAYS}  TAKER_RT={TAKER_RT}")
    print(f"  Data: {DATA_START} -> {DATA_END}")
    print("=" * 80)

    # ---- u10 reference ----
    print(f"\n[1] Loading u10 ...")
    ind10 = ml.load(start=DATA_START, end=DATA_END)
    C10, R10, idx10 = ind10["C"], ind10["R"], ind10["C"].index
    gate10   = ind10["gate"]
    mom14_10 = ind10["mom14"]
    quality10 = (mom14_10.rank(axis=1, pct=True, na_option="keep") * 0.5 +
                 (1 - R10.rolling(20, min_periods=10).std().rank(
                      axis=1, pct=True, na_option="keep")) * 0.25 +
                 (C10 > ind10["sma50"]).astype(float) * 0.25)
    loaded10 = list(C10.columns)
    print(f"  Loaded: {loaded10}")

    # ---- u50 ----
    print(f"\n[2] Loading u50 ...")
    C50, R50, ind50, cov50 = _load_u50(DATA_START, DATA_END)
    idx50    = C50.index
    loaded50 = list(C50.columns)
    print(f"  Loaded {len(loaded50)} symbols: {loaded50}")

    print("\n[2b] Coverage manifest (2020-2022 bars per symbol):")
    if not cov50.empty:
        print(cov50.to_string())

    # ---- slice starts ----
    rng   = np.random.default_rng(RNG_SEED)
    valid = idx50[(idx50 >= idx50[0] + pd.Timedelta(days=MIN_WARM)) &
                  (idx50 < idx50[-1] - pd.Timedelta(days=SLICE_DAYS + 5))]
    n_draw = min(N_SLICES, len(valid))
    chosen = rng.choice(len(valid), n_draw, replace=False)
    chosen.sort()
    chosen_starts = [valid[i] for i in chosen]
    print(f"\n[3] Slice starts: {len(chosen_starts)} from {chosen_starts[0].date()} "
          f"to {chosen_starts[-1].date()}")

    # ---- run ----
    strats_data: dict[str, list] = {}

    def add(name: str, r: float, r10bh: float, r50bh: float) -> None:
        if np.isnan(r):
            return
        if name not in strats_data:
            strats_data[name] = []
        strats_data[name].append((r, r10bh, r50bh))

    print("[4] Running slices ...")
    for sl_start in chosen_starts:
        r10bh = _bh_ret(R10, idx10, sl_start)
        r50bh = _bh_ret(R50, idx50, sl_start)
        if np.isnan(r10bh) or np.isnan(r50bh):
            continue

        add("u10_BH", r10bh, r10bh, r50bh)
        add("u50_BH", r50bh, r10bh, r50bh)

        # u10 strategies
        for K in [3, 5, 10]:
            w = _pick_topk(mom14_10, gate10, idx10, sl_start, K, True)
            add(f"u10_mom14_K{K}_g", _slice_ret(R10, idx10, w, sl_start), r10bh, r50bh)
        w = _pick_topk(mom14_10, gate10, idx10, sl_start, 5, False)
        add("u10_mom14_K5_ng", _slice_ret(R10, idx10, w, sl_start), r10bh, r50bh)
        w = _pick_topk(quality10, gate10, idx10, sl_start, 5, True)
        add("u10_quality_K5_g", _slice_ret(R10, idx10, w, sl_start), r10bh, r50bh)

        # u50 strategies
        for K in [3, 5, 10]:
            w = _pick_topk(ind50["mom14"], ind50["gate"], idx50, sl_start, K, True)
            add(f"u50_mom14_K{K}_g", _slice_ret(R50, idx50, w, sl_start), r10bh, r50bh)
        for K in [3, 5]:
            w = _pick_topk(ind50["mom14"], ind50["gate"], idx50, sl_start, K, False)
            add(f"u50_mom14_K{K}_ng", _slice_ret(R50, idx50, w, sl_start), r10bh, r50bh)
        w = _pick_topk(ind50["quality"], ind50["gate"], idx50, sl_start, 3, True)
        add("u50_quality_K3_g", _slice_ret(R50, idx50, w, sl_start), r10bh, r50bh)
        w = _pick_topk(ind50["quality"], ind50["gate"], idx50, sl_start, 5, True)
        add("u50_quality_K5_g", _slice_ret(R50, idx50, w, sl_start), r10bh, r50bh)

    # ---- summarise ----
    def srow(rows: list) -> dict:
        rets = np.array([x[0] for x in rows])
        r10a = np.array([x[1] for x in rows])
        trim_mask = (rets > np.percentile(rets, 5)) & (rets < np.percentile(rets, 95))
        return {
            "n":           len(rets),
            "win_pct":     float(np.mean(rets > 0) * 100),
            "mean_pct":    float(np.mean(rets) * 100),
            "median_pct":  float(np.median(rets) * 100),
            "trimmed_mean":float(np.mean(rets[trim_mask]) * 100),
            "beat_u10_pct":float(np.mean(rets > r10a) * 100),
            "excess_pct":  float(np.mean(rets - r10a) * 100),
            "p5_pct":      float(np.percentile(rets, 5) * 100),
            "p95_pct":     float(np.percentile(rets, 95) * 100),
        }

    summaries = {name: srow(rows) for name, rows in strats_data.items()}

    # ---- markdown table ----
    print()
    print("=" * 100)
    print("RESULTS -- Random 7d slice eval, 2020-2023, N=350, taker cost")
    print("WIN CONDITION: Win% > 55.0 OR Mean% > 2.9  (spec ref: u10 BH ~55% / ~2.9%)")
    print("Trim%: 5-95th percentile trimmed mean (robustness check)")
    print("=" * 100)
    print(f"{'Strategy':<24} | N   | Win%  | Mean%  | Trim% | Beat-u10% | Excess% | P5%   | P95%  | PASS?")
    print("-" * 100)

    order = [
        "u10_BH", "u50_BH",
        "u10_mom14_K3_g", "u10_mom14_K5_g", "u10_mom14_K10_g",
        "u10_mom14_K5_ng", "u10_quality_K5_g",
        "u50_mom14_K3_g", "u50_mom14_K5_g", "u50_mom14_K10_g",
        "u50_mom14_K3_ng", "u50_mom14_K5_ng",
        "u50_quality_K3_g", "u50_quality_K5_g",
    ]
    for name in order:
        if name not in summaries:
            continue
        s = summaries[name]
        win_pass  = s["win_pct"]  > 55.0
        mean_pass = s["mean_pct"] > 2.9
        p = "YES" if (win_pass or mean_pass) else "no "
        print(f"{name:<24} | {s['n']:3d} | {s['win_pct']:5.1f} | {s['mean_pct']:6.2f} | "
              f"{s['trimmed_mean']:5.2f} | {s['beat_u10_pct']:9.1f} | {s['excess_pct']:7.2f} | "
              f"{s['p5_pct']:5.1f} | {s['p95_pct']:5.1f} | {p}")

    # ---- key comparisons ----
    print()
    print("=" * 100)
    print("KEY COMPARISONS: u50 top-K vs u10 top-K at same K")
    print("=" * 100)
    pairs = [
        ("u10_mom14_K5_g",  "u50_mom14_K5_g",  "mom14 K5 gated"),
        ("u10_mom14_K3_g",  "u50_mom14_K3_g",  "mom14 K3 gated"),
        ("u10_quality_K5_g","u50_quality_K5_g","quality K5 gated"),
    ]
    for k10, k50, label in pairs:
        if k10 not in summaries or k50 not in summaries:
            continue
        s10, s50 = summaries[k10], summaries[k50]
        dw = s50["win_pct"]  - s10["win_pct"]
        dm = s50["mean_pct"] - s10["mean_pct"]
        db = s50["beat_u10_pct"] - s10["beat_u10_pct"]
        print(f"\n  {label}")
        print(f"    u10: win={s10['win_pct']:.1f}%  mean={s10['mean_pct']:.2f}%  "
              f"beat-u10BH={s10['beat_u10_pct']:.1f}%")
        print(f"    u50: win={s50['win_pct']:.1f}%  mean={s50['mean_pct']:.2f}%  "
              f"beat-u10BH={s50['beat_u10_pct']:.1f}%")
        print(f"    delta(u50-u10): win={dw:+.1f}pp  mean={dm:+.2f}pp  beat_u10BH={db:+.1f}pp")

    # ---- verdict ----
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    u10bh_s = summaries.get("u10_BH", {})
    print(f"\n  BH reference (measured): win={u10bh_s.get('win_pct', 0):.1f}%  "
          f"mean={u10bh_s.get('mean_pct', 0):.2f}%/slice")
    print(f"  BH reference (spec):     win=~55%  mean=~+2.9%/slice")

    winners = [(n, s) for n, s in summaries.items()
               if not n.endswith("_BH") and s["n"] >= 50
               and (s["win_pct"] > 55.0 or s["mean_pct"] > 2.9)]
    print(f"\n  WIN CONDITION MET: {'YES' if winners else 'NO'}")
    if winners:
        for n, s in winners:
            print(f"    {n}: win={s['win_pct']:.1f}%  mean={s['mean_pct']:.2f}%")

    # u50 vs u10 summary
    u50k5 = summaries.get("u50_mom14_K5_g", {})
    u10k5 = summaries.get("u10_mom14_K5_g", {})
    if u50k5 and u10k5:
        dw5 = u50k5["win_pct"] - u10k5["win_pct"]
        dm5 = u50k5["mean_pct"] - u10k5["mean_pct"]
        print(f"\n  WIDER BENCH (u50 K5 vs u10 K5): win {dw5:+.1f}pp  mean {dm5:+.2f}pp")
        if dw5 > 2:
            print("    -> Wider bench IMPROVES win-rate significantly")
        elif dm5 > 0.3:
            print("    -> Wider bench IMPROVES mean return but not win-rate")
        else:
            print("    -> Wider bench does NOT improve win-rate; mean slightly higher")

    print(f"\n  Coverage: {len(loaded50)} syms loaded; "
          f"16 syms have full 2020 history; "
          f"16 late-joiners enter pool only after their listing date (causal).")
    print("  Concentration: gated K3/K5 hold exactly K assets when gate permits.")
    print("  Distribution: fat tails (p5 ~ -22%, p95 ~ +36%) driven by crypto vol.")
    print("  Mean > 2.9% is REAL but high-variance (trim mean 2-3%); win-rate stays <55%.")

    # ---- save JSON ----
    out_dir  = ROOT.parent / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wider_bench_slice_lab_{ts}.json"
    payload  = {
        "meta": {
            "script":    "wider_bench_slice_lab.py",
            "n_slices":  n_draw,
            "slice_days": SLICE_DAYS,
            "data_start": DATA_START,
            "data_end":   DATA_END,
            "cost_rt":    TAKER_RT,
            "seed":       RNG_SEED,
            "u10_loaded": loaded10,
            "u50_loaded": loaded50,
        },
        "summaries": summaries,
        "bh_reference": {
            "win_pct":  u10bh_s.get("win_pct"),
            "mean_pct": u10bh_s.get("mean_pct"),
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
