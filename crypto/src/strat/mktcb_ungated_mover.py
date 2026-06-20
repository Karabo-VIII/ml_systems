"""src/strat/mktcb_ungated_mover.py -- MARKET-LEVEL CIRCUIT-BREAKER for ungated mover capture.

OBJECTIVE (corrected cycle): move-capture across a 7d window with 14d lookback.
Be IN the assets that MOVE and ride them. NO per-asset SMA200 gate (this was the
proven failure: top-3 movers gated out 38-39% of the time).

FIX: a MARKET-LEVEL circuit-breaker scales TOTAL BOOK EXPOSURE by a causal market signal
WITHOUT excluding any individual mover candidate.

FOUR RULES TESTED (all causal, long-only, taker cost):
  A. BTC_SMA200 scale  -- scale to 1.0 if BTC > SMA200 else BEAR_SCALE (0.0..0.3)
  B. BREADTH scale     -- scale proportional to % universe above SMA50
  C. VOL_TARGET        -- inverse-vol sizing: scale = vol_target / book_realized_vol
  D. DD_DERISK         -- drawdown-responsive: scale down exponentially as DD deepens

Ungated baseline: top-3 by 14d momentum, rebal every 3d, EW, NO gate.
Gated baseline: the per-asset SMA200 router (adaptive_meta_engine.build_weight_matrix).

JUDGE METRICS:
  (a) CAPTURE-RATE = realized / available-move vs in-window oracle per year
  (b) random-7d-slice profitability: pos_rate, mean%, p05%, beat_bh%
  (c) bear-survival: 2022 comp%, maxDD over bear period, and down-week behavior
  (d) full-cycle compound + maxDD vs gated baseline + EW BH

Output: exposure-rule x regime matrix table (bear/chop/bull) + capture-in-bull vs
survival-in-bear decomposition.

RWYB: python -m strat.mktcb_ungated_mover
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as rh
import strat.adaptive_meta_engine as ame

# ============================================================
# CONSTANTS
# ============================================================
COST = lab.COST          # taker round-trip
TOP_K = 3                # top movers to hold
REBAL = 3                # rebalance every 3 days
DATA_START = "2020-01-01"
DATA_END   = "2026-06-01"

OOS_START  = "2022-01-01"   # bear begins
OOS_END    = DATA_END

N_SLICES   = 500
SEEDS      = [11, 23, 42]
SLICE_DAYS = 7

# CB rule parameters
BEAR_SCALE    = 0.10    # BTC_SMA200 rule: exposure in downtrend
BREADTH_MIN   = 0.20    # breadth rule: 0 exposure at or below this
BREADTH_MAX   = 0.80    # breadth rule: full exposure at or above this
VOL_TARGET    = 0.60    # vol_target rule: annualised target vol
DD_START      = 0.05    # DD rule: start scaling down at 5% DD
DD_FULL       = 0.25    # DD rule: full cash at 25% DD


# ============================================================
# UNGATED TOP-K BUILDER (the base mover book, no asset gate)
# ============================================================
def ungated_topk(ind: dict, K: int = TOP_K, rebal: int = REBAL) -> pd.DataFrame:
    """Hold top-K by 14d momentum, EW, rebal every `rebal` days. NO per-asset gate."""
    C = ind["C"]
    mom14 = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last = -9999
    cur_w: dict = {}
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            row = mom14.iloc[i]
            valid = row.dropna()
            if len(valid) >= K:
                picks = valid.nlargest(K).index.tolist()
                cur_w = {s: 1.0 / K for s in picks}
            elif len(valid) > 0:
                cur_w = {s: 1.0 / len(valid) for s in valid.index.tolist()}
            last = i
        for s, w in cur_w.items():
            W.loc[d, s] = w
    return W


# ============================================================
# MARKET-LEVEL CIRCUIT BREAKER: apply scale to a weight matrix
# ============================================================
def _btc_sma200_scale(ind: dict, bear_scale: float = BEAR_SCALE) -> pd.Series:
    """Causal: 1.0 if BTC > SMA200 on that day, else bear_scale."""
    C = ind["C"]; s200 = ind["sma200"]
    btc_c = C["BTCUSDT"]; btc_s = s200["BTCUSDT"]
    scale = pd.Series(1.0, index=C.index)
    for d in C.index:
        bc = btc_c.loc[d]; bs = btc_s.loc[d]
        if pd.notna(bc) and pd.notna(bs):
            if bc <= bs:
                scale.loc[d] = bear_scale
    return scale


def _breadth_scale(ind: dict, bmin: float = BREADTH_MIN, bmax: float = BREADTH_MAX) -> pd.Series:
    """Causal: exposure = clamp((breadth - bmin) / (bmax - bmin), 0, 1)."""
    C = ind["C"]; sma50 = ind["sma50"]
    scale = pd.Series(np.nan, index=C.index)
    for i, d in enumerate(C.index):
        row_c = C.iloc[i]; row_s = sma50.iloc[i]
        above = 0; total = 0
        for sym in C.columns:
            cv = row_c[sym]; sv = row_s[sym]
            if pd.notna(cv) and pd.notna(sv):
                above += int(cv > sv); total += 1
        breadth = above / total if total > 0 else 0.5
        scale.loc[d] = float(np.clip((breadth - bmin) / max(bmax - bmin, 1e-9), 0.0, 1.0))
    return scale


def _voltarget_scale(bret_ungated: pd.Series, vol_target: float = VOL_TARGET,
                     window: int = 20) -> pd.Series:
    """Causal: realized vol from last `window` days; scale = vol_target / realvol, capped 1.0."""
    realvol = bret_ungated.rolling(window, min_periods=5).std() * np.sqrt(365)
    scale = (vol_target / realvol.clip(lower=1e-6)).clip(upper=1.0).fillna(1.0)
    return scale


def _dd_scale(bret_ungated: pd.Series,
              dd_start: float = DD_START, dd_full: float = DD_FULL) -> pd.Series:
    """Causal drawdown-responsive: scale = 1 - clamp((dd - dd_start)/(dd_full-dd_start), 0, 1)."""
    eq = (1 + bret_ungated).cumprod()
    pk = eq.expanding().max()
    dd = ((eq - pk) / pk).abs()     # drawdown as positive fraction
    frac = ((dd - dd_start) / max(dd_full - dd_start, 1e-9)).clip(0.0, 1.0)
    scale = (1.0 - frac).fillna(1.0)
    return scale


def apply_exposure_scale(W: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Multiply each row of W by the scalar exposure, keeping row-sum <= 1."""
    aligned = scale.reindex(W.index).fillna(1.0)
    scaled = W.multiply(aligned, axis=0)
    return scaled


# ============================================================
# ORACLE CAPTURE RATE (leak-ok: oracle only used for the denominator)
# ============================================================
def oracle_capture_rate(W: pd.DataFrame, ind: dict, year_start: str, year_end: str) -> dict:
    """For each bar in [year_start, year_end), compute:
      - realised return of the book (already cost-net via book_daily_returns)
      - 'available' = best 7d fwd return among any asset (oracle, biased but denominator only)
    Return mean(realized/available) capped to [0,1] per period.
    """
    C = ind["C"]
    bret = rh.book_daily_returns(W, ind)
    mask = (C.index >= pd.Timestamp(year_start)) & (C.index < pd.Timestamp(year_end))
    bret_p = bret[mask]
    # available-move oracle: max across assets of 7d fwd compound (causal violation is intentional --
    # this is the DENOMINATOR benchmark only; never used for position selection)
    C_p = C[mask]
    # roll 7d fwd compound per asset
    fwd7 = C.shift(-7) / C - 1   # fwd 7d from d
    fwd7_p = fwd7[mask]
    available = fwd7_p.max(axis=1).clip(lower=0.0)   # best bull move available

    # realized = 7d compound of book returns, rolling (centre at d with 7d window)
    # We approximate as: realized_over_7d at d = prod(1+bret[d:d+7])
    # Use expanding window of 7 for the trailing realized
    realized_trailing = bret_p.rolling(7, min_periods=1).apply(lambda x: (1 + x).prod() - 1, raw=True)
    avail_trailing = available.rolling(7, min_periods=1).apply(lambda x: (1 + x).prod() - 1, raw=True).clip(lower=1e-6)

    cr = (realized_trailing / avail_trailing).clip(0.0, 1.0)
    return {
        "capture_mean": round(float(cr.mean()), 3),
        "capture_median": round(float(cr.median()), 3),
        "n": int(cr.notna().sum()),
    }


# ============================================================
# REGIME LABELING for the matrix table (causal)
# ============================================================
def regime_label_series(ind: dict) -> pd.Series:
    """Causal daily regime: 'bear'=BTC below SMA200, 'bull'=breadth>50%, 'chop' otherwise."""
    C = ind["C"]; s200 = ind["sma200"]; sma50 = ind["sma50"]
    btc_c = C["BTCUSDT"]; btc_s = s200["BTCUSDT"]
    labels = {}
    for i, d in enumerate(C.index):
        bc = btc_c.loc[d]; bs = btc_s.loc[d]
        if pd.isna(bc) or pd.isna(bs) or bc <= bs:
            labels[d] = "bear"
            continue
        row_c = C.iloc[i]; row_s = sma50.iloc[i]
        above = 0; total = 0
        for sym in C.columns:
            cv = row_c[sym]; sv = row_s[sym]
            if pd.notna(cv) and pd.notna(sv):
                above += int(cv > sv); total += 1
        breadth = above / total if total > 0 else 0.5
        labels[d] = "bull" if breadth >= 0.50 else "chop"
    return pd.Series(labels)


def regime_slice_stats(bret: pd.Series, bh: pd.Series,
                       regime_ser: pd.Series, regime: str,
                       oos_start: str) -> dict:
    """Slice stats restricted to bars in a given regime (OOS only)."""
    mask = (regime_ser.reindex(bret.index) == regime) & (bret.index >= pd.Timestamp(oos_start))
    idx = bret.index[mask]
    if len(idx) < SLICE_DAYS + 5:
        return {"pos_rate": None, "mean_pct": None, "n_bars": len(idx)}
    rng = np.random.default_rng(99)
    max_start = len(idx) - SLICE_DAYS
    eng_r, bh_r = [], []
    for _ in range(300):
        si = rng.integers(0, max_start)
        sl = idx[si: si + SLICE_DAYS]
        eng_r.append(float((1 + bret.loc[sl]).prod() - 1))
        bh_r.append(float((1 + bh.loc[sl]).prod() - 1))
    eng = np.array(eng_r); bhr = np.array(bh_r)
    return {
        "pos_rate": round(100 * float((eng > 0).mean()), 1),
        "mean_pct": round(100 * float(eng.mean()), 2),
        "beat_bh_pct": round(100 * float((eng > bhr).mean()), 1),
        "n_bars": len(idx),
    }


# ============================================================
# FULL EVALUATION OF ONE STRATEGY
# ============================================================
def eval_strategy(W: pd.DataFrame, ind: dict, bh_b: pd.Series, bh_W: pd.DataFrame,
                  label: str, regime_ser: pd.Series) -> dict:
    bret = rh.book_daily_returns(W, ind)
    # Full-period via lab.evaluate
    full = lab.evaluate(W, ind, H=7, label=label)
    # Bear period compound (2022 specifically)
    C = ind["C"]
    bear22 = (C.index >= "2022-01-01") & (C.index < "2023-01-01")
    eq_bear = float((1 + bret[bear22]).prod() - 1) * 100
    # 2024 drawdown H1
    h1_2024 = (C.index >= "2024-01-01") & (C.index < "2024-07-01")
    eq_2024h1 = float((1 + bret[h1_2024]).prod() - 1) * 100
    # 2025 drawdown
    d2025 = (C.index >= "2025-01-01") & (C.index < "2026-01-01")
    eq_2025 = float((1 + bret[d2025]).prod() - 1) * 100
    # OOS maxDD
    oos_mask = C.index >= OOS_START
    bret_oos = bret[oos_mask]
    eq_oos = (1 + bret_oos).cumprod()
    pk_oos = eq_oos.expanding().max()
    maxdd_oos = round(float(((eq_oos - pk_oos) / pk_oos).min() * 100), 1)

    # Random-slice stats
    slice_agg = defaultdict(list)
    for s in SEEDS:
        st = rh.slice_stats(bret, bh_b, OOS_START, OOS_END, N_SLICES, SLICE_DAYS, s)
        for k, v in st.items():
            if v is not None:
                slice_agg[k].append(v)
    ss = {k: round(float(np.mean(v)), 1) for k, v in slice_agg.items()}

    # Regime-breakdown matrix
    rgm = {}
    for r in ("bear", "chop", "bull"):
        rs = regime_slice_stats(bret, bh_b, regime_ser, r, OOS_START)
        rgm[r] = rs

    # Capture rate per year
    cap = {}
    for yr_s, yr_e in [("2022-01-01","2023-01-01"),("2023-01-01","2024-01-01"),
                        ("2024-01-01","2025-01-01"),("2025-01-01","2026-01-01")]:
        k = yr_s[:4]
        cap[k] = oracle_capture_rate(W, ind, yr_s, yr_e)

    # Average exposure OOS
    avg_expo = round(float(W.sum(axis=1)[oos_mask].mean()), 3)

    return {
        "label": label,
        "comp_2020": full["comp_2020"],
        "comp_2021": full["comp_2021"],
        "comp_2022": full["comp_2022"],
        "comp_full": full["comp_full"],
        "maxDD_full": full["maxDD"],
        "maxDD_oos": maxdd_oos,
        "bear_22_pct": round(eq_bear, 1),
        "h1_2024_pct": round(eq_2024h1, 1),
        "yr_2025_pct": round(eq_2025, 1),
        "avg_expo_oos": avg_expo,
        "green_all": full["green_all"],
        "slice_pos_rate": ss.get("pos_rate"),
        "slice_mean_pct": ss.get("mean_pct"),
        "slice_p05_pct": ss.get("p05_pct"),
        "slice_beat_bh": ss.get("beat_bh_pct"),
        "down_wk_eng_mean": ss.get("down_wk_eng_mean"),
        "regime_bear": rgm["bear"],
        "regime_chop": rgm["chop"],
        "regime_bull": rgm["bull"],
        "capture_by_year": cap,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 76)
    print("MARKET-LEVEL CIRCUIT-BREAKER -- UNGATED MOVER CAPTURE DECOMPOSITION")
    print(f"OOS: {OOS_START} -> {OOS_END}  |  n_slices={N_SLICES}  |  seeds={SEEDS}")
    print("=" * 76)

    # ---- load ----
    ind = lab.load(DATA_START, DATA_END)
    bh_W = rh.bh_ew_weights(ind)
    bh_b = rh.book_daily_returns(bh_W, ind)
    C = ind["C"]

    # ---- regime labels ----
    print("\nBuilding regime labels...")
    regime_ser = regime_label_series(ind)

    # ---- base weight matrices ----
    print("Building weight matrices...")

    # 1. EW buy-hold baseline
    W_bh = bh_W.copy()

    # 2. Gated router (old baseline -- per-asset SMA200)
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    W_gated = ame.build_weight_matrix(ind, vthr)

    # 3. UNGATED top-3 by mom14 (the raw mover book, no circuit breaker)
    W_raw = ungated_topk(ind, K=TOP_K, rebal=REBAL)

    # ---- circuit breaker scales (computed causal from DATA_START) ----
    print("Computing circuit breaker scales...")
    # Pre-compute ungated daily returns for vol/DD-based scalers
    bret_raw = rh.book_daily_returns(W_raw, ind)

    scale_btc   = _btc_sma200_scale(ind, bear_scale=BEAR_SCALE)
    scale_brd   = _breadth_scale(ind)
    scale_volt  = _voltarget_scale(bret_raw, vol_target=VOL_TARGET)
    scale_dd    = _dd_scale(bret_raw)

    # Combined: BTC + vol-target (additive conservative)
    scale_combo = (scale_btc * scale_volt).clip(0.0, 1.0)

    # Apply scales
    W_btc_cb  = apply_exposure_scale(W_raw, scale_btc)
    W_brd_cb  = apply_exposure_scale(W_raw, scale_brd)
    W_volt_cb = apply_exposure_scale(W_raw, scale_volt)
    W_dd_cb   = apply_exposure_scale(W_raw, scale_dd)
    W_combo   = apply_exposure_scale(W_raw, scale_combo)

    strategies = [
        ("EW_BH",            W_bh),
        ("Gated_Router",     W_gated),
        ("Ungated_Raw",      W_raw),
        ("CB_BTC_SMA200",    W_btc_cb),
        ("CB_Breadth",       W_brd_cb),
        ("CB_VolTarget",     W_volt_cb),
        ("CB_DrawdownRisk",  W_dd_cb),
        ("CB_BTC+VolTarget", W_combo),
    ]

    # ---- evaluate all ----
    print(f"\nEvaluating {len(strategies)} strategies...")
    results = {}
    for name, W in strategies:
        print(f"  {name}...", end=" ", flush=True)
        r = eval_strategy(W, ind, bh_b, bh_W, name, regime_ser)
        results[name] = r
        print(f"full={r['comp_full']}%  bear22={r['bear_22_pct']}%  "
              f"slice_pos={r['slice_pos_rate']}%  DD={r['maxDD_oos']}%")

    # ---- PRINT TABLE ----
    print("\n" + "=" * 110)
    print("EXPOSURE-RULE x REGIME MATRIX + OVERALL METRICS")
    print("=" * 110)
    hdr = (f"{'Strategy':<22} | {'Full%':>7} {'Bear22%':>8} {'2024H1%':>8} {'2025%':>7}"
           f" | {'MaxDD_OOS':>9} | {'Slice_PR%':>9} {'Mn%':>6} {'P05%':>6} {'BtBH%':>6}"
           f" | {'DwnWk%':>7} | {'Expo':>5}")
    print(hdr)
    print("-" * 110)
    for name, r in results.items():
        dwm = r.get("down_wk_eng_mean") or 0.0
        print(f"{name:<22} | {r['comp_full']:>7.1f} {r['bear_22_pct']:>8.1f} "
              f"{r['h1_2024_pct']:>8.1f} {r['yr_2025_pct']:>7.1f}"
              f" | {r['maxDD_oos']:>9.1f}"
              f" | {r['slice_pos_rate']:>9.1f} {r['slice_mean_pct']:>6.2f} "
              f"{r['slice_p05_pct']:>6.2f} {r['slice_beat_bh']:>6.1f}"
              f" | {dwm:>7.2f}"
              f" | {r['avg_expo_oos']:>5.2f}")

    # ---- REGIME BREAKDOWN TABLE ----
    print("\n" + "=" * 90)
    print("REGIME SLICE STATS (OOS 2022+): bear / chop / bull (pos_rate% | mean% | beat_bh%)")
    print("=" * 90)
    for name, r in results.items():
        bear = r["regime_bear"]; chop = r["regime_chop"]; bull = r["regime_bull"]
        def fmt(d):
            if d.get("pos_rate") is None:
                return "  N/A "
            return f"{d['pos_rate']:4.0f}%|{d['mean_pct']:5.1f}%|{d['beat_bh_pct']:4.0f}%"
        print(f"  {name:<22} | bear={fmt(bear)} | chop={fmt(chop)} | bull={fmt(bull)} "
              f"| n_bear={bear.get('n_bars',0)} chop={chop.get('n_bars',0)} bull={bull.get('n_bars',0)}")

    # ---- CAPTURE RATE TABLE ----
    print("\n" + "=" * 90)
    print("ORACLE CAPTURE RATE by YEAR (mean realized/available, trailing 7d, capped [0,1])")
    print("=" * 90)
    print(f"  {'Strategy':<22}  {'2022':>8}  {'2023':>8}  {'2024':>8}  {'2025':>8}")
    print("-" * 70)
    for name, r in results.items():
        cap = r["capture_by_year"]
        def fc(yr): c = cap.get(yr, {}); return f"{c.get('capture_mean',0):.3f}" if c else "  N/A"
        print(f"  {name:<22}  {fc('2022'):>8}  {fc('2023'):>8}  {fc('2024'):>8}  {fc('2025'):>8}")

    # ---- VERDICT ----
    print("\n" + "=" * 76)
    print("VERDICT DECOMPOSITION")
    print("=" * 76)
    raw = results.get("Ungated_Raw", {})
    btc = results.get("CB_BTC_SMA200", {})
    brd = results.get("CB_Breadth", {})
    volt = results.get("CB_VolTarget", {})
    dd = results.get("CB_DrawdownRisk", {})
    combo = results.get("CB_BTC+VolTarget", {})
    gated = results.get("Gated_Router", {})
    bh_r = results.get("EW_BH", {})

    def delta(a, b, key): return round((a.get(key) or 0) - (b.get(key) or 0), 1)

    print(f"\nUngated_Raw vs Gated_Router:")
    print(f"  bear22 delta:   {delta(raw, gated, 'bear_22_pct'):+.1f}pp  "
          f"(raw={raw.get('bear_22_pct')}%  gated={gated.get('bear_22_pct')}%)")
    print(f"  slice_pos_rate: {delta(raw, gated, 'slice_pos_rate'):+.1f}pp  "
          f"(raw={raw.get('slice_pos_rate')}%  gated={gated.get('slice_pos_rate')}%)")
    print(f"  full compound:  {delta(raw, gated, 'comp_full'):+.1f}pp")

    print(f"\nBest CB rules vs Ungated_Raw (bear22 / maxDD_oos / slice_pos_rate):")
    for nm, rs in [("CB_BTC_SMA200", btc), ("CB_Breadth", brd),
                   ("CB_VolTarget", volt), ("CB_DrawdownRisk", dd), ("CB_BTC+VolTarget", combo)]:
        print(f"  {nm:<22}  bear22={delta(rs,raw,'bear_22_pct'):+.1f}pp  "
              f"DD={delta(rs,raw,'maxDD_oos'):+.1f}pp  "
              f"pos_rate={delta(rs,raw,'slice_pos_rate'):+.1f}pp  "
              f"full={delta(rs,raw,'comp_full'):+.1f}pp")

    print(f"\nBest CB vs Gated_Router (the question: does market-CB beat per-asset-gate?):")
    best_name = max(
        ["CB_BTC_SMA200","CB_Breadth","CB_VolTarget","CB_DrawdownRisk","CB_BTC+VolTarget"],
        key=lambda n: (results[n].get("slice_pos_rate") or 0)
    )
    best_r = results[best_name]
    print(f"  Best by pos_rate: {best_name}")
    print(f"  vs Gated_Router:  bear22={delta(best_r,gated,'bear_22_pct'):+.1f}pp  "
          f"DD={delta(best_r,gated,'maxDD_oos'):+.1f}pp  "
          f"pos_rate={delta(best_r,gated,'slice_pos_rate'):+.1f}pp  "
          f"full={delta(best_r,gated,'comp_full'):+.1f}pp")

    # ---- save ----
    out = {
        "oos": [OOS_START, OOS_END], "n_slices": N_SLICES, "seeds": SEEDS,
        "params": {"TOP_K": TOP_K, "REBAL": REBAL, "BEAR_SCALE": BEAR_SCALE,
                   "BREADTH_MIN": BREADTH_MIN, "BREADTH_MAX": BREADTH_MAX,
                   "VOL_TARGET": VOL_TARGET, "DD_START": DD_START, "DD_FULL": DD_FULL},
        "results": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    outp = ROOT.parent / "runs" / "strat" / "mktcb_ungated_mover_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
