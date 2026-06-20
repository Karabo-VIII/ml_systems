"""u50_wider_bench_engine.py -- WIDER BENCH ENGINE TOURNAMENT

LANE: Does more dispersion (u30/u50 vs u10) raise the fraction of up-movers held
      -> higher 7-day positive-rate, walk-forward OOS, honest on coverage?

ENGINE: Always-long top-K momentum/quality leaders across u30 or u50.
Signals: mom14, quality composite (mom14+low-vol+above-sma50), breakout (above 14d high).
Walk-forward: training window EXPANDS from MIN_WARM to slice_start; score computed at
last known bar before each slice (causal).

WIN CONDITION (from spec):
  - Beat u10 buy-hold positive-rate (measured ~53.7%) AND/OR mean (+2.97%)
  - As an ACTIVE engine (top-K selection, not buy-hold)
  - OOS / walk-forward, leak-free, N>=300 slices

EXTRA REQUIREMENT: per-slice, report down-week vs up-week behavior:
  - "down week" = u10 BH return < 0 on that slice
  - does the engine approximate CASH on down weeks? (the desired never-negative behavior)

UNIVERSES:
  u10  -- 10 liquid (mover_lab default)
  u30  -- u10 + 20 well-known alts with reasonable 2020 coverage
  u50  -- u30 + 18 more coins (many post-2021 listings; causal = enters pool at listing)

CAUSAL DISCIPLINE (referee will check):
  - Gate = C > sma200 at row d (computed from data[:d] only)
  - Score at row d uses only data[:d]
  - Slice return measured [sl_start, sl_start+7d)
  - Position set from signal at (sl_start - 1 bar), entered at sl_start open (proxied = close)
  - No global scaler: all ranks computed within training window up to sl_start-1

RWYB: python -m strat.u50_wider_bench_engine
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
# Config
# ---------------------------------------------------------------------------
TAKER_RT   = 0.0024    # round-trip taker cost (mover_lab constant)
SLICE_DAYS = 7
N_SLICES   = 350
RNG_SEED   = 42
MIN_WARM   = 210       # sma200 needs 200 bars; add 10 buffer
DATA_START = "2020-01-01"
DATA_END   = "2023-01-01"

# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------
U10_SYMS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

U30_EXTRA = [
    "DOTUSDT", "BCHUSDT", "UNIUSDT", "TRXUSDT",
    "AAVEUSDT", "ALGOUSDT", "ETCUSDT", "FILUSDT",
    "ICPUSDT", "NEARUSDT", "SHIBUSDT", "ZECUSDT",
    "DASHUSDT", "HBARUSDT", "ENJUSDT", "CRVUSDT",
    "JSTUSDT", "LDOUSDT", "FETUSDT", "COMPUSDT",
]

U50_EXTRA = [
    "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
    "SEIUSDT", "WLDUSDT", "BLURUSDT", "BONKUSDT",
    "PEPEUSDT", "WIFUSDT", "RENDERUSDT", "TAOUSDT",
    "TONUSDT", "SUPERUSDT", "ORDIUSDT", "ENAUSDT",
    "PENGUUSDT", "TRUMPUSDT",
]

def _dedup(lst):
    seen = set(); return [x for x in lst if not (x in seen or seen.add(x))]

U30_SYMS = _dedup(U10_SYMS + U30_EXTRA)
U50_SYMS = _dedup(U30_SYMS + U50_EXTRA)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_universe(syms: list[str], start: str, end: str):
    """Load daily close for each symbol, build causal indicators.

    Returns (ind, loaded_syms, coverage_df).
    Missing bars filled with NaN -> treated as cash in slices.
    """
    s_ms = int(pd.Timestamp(start).value // 10**6)
    e_ms = int(pd.Timestamp(end).value   // 10**6)
    panels: dict[str, pd.Series] = {}
    cov_rows = []

    for sym in syms:
        try:
            _, _, _, c_arr, ms_arr = _panel(sym, "1d")
        except Exception:
            continue
        i0 = int(np.searchsorted(ms_arr, s_ms))
        ie = int(np.searchsorted(ms_arr, e_ms))
        if ie - i0 < 50:
            continue
        dates = pd.to_datetime(ms_arr[i0:ie], unit="ms").normalize()
        cs = pd.Series(c_arr[i0:ie], index=dates, name=sym)
        cs = cs[~cs.index.duplicated(keep="last")]
        panels[sym] = cs
        yr = dates.year.value_counts().sort_index()
        row = {"sym": sym, "total_bars": len(cs),
               "first_bar": str(dates[0].date())}
        for y in [2020, 2021, 2022]:
            row[f"bars_{y}"] = int(yr.get(y, 0))
        cov_rows.append(row)

    if not panels:
        raise RuntimeError(f"No symbols loaded for universe {syms[:3]}...")

    all_dates = pd.date_range(start, end, freq="D", inclusive="left")
    C = pd.DataFrame({s: panels[s].reindex(all_dates) for s in panels})
    # Forward-fill short weekend/holiday gaps (<=3 bars); keep NaN for pre-listing
    C = C.ffill(limit=3)
    R = C.pct_change()

    sma200 = C.rolling(200, min_periods=200).mean()
    sma50  = C.rolling(50,  min_periods=50).mean()
    mom14  = C / C.shift(14) - 1
    mom7   = C / C.shift(7)  - 1
    mom30  = C / C.shift(30) - 1
    gate   = (C > sma200).fillna(False)

    # Quality composite (cross-sectional rank to be causal-per-slice -- see note)
    # NOTE: rank is computed over the full history for the indicator panels, but
    # the POSITION decision at each slice only uses the value at (sl_start - 1 bar),
    # which is causally valid as long as the ranking uses only past data.
    # We compute the raw scores here; the ranking happens at selection time using
    # only the trailing window.
    vol20      = R.rolling(20, min_periods=10).std()

    ind = {
        "C":      C,
        "R":      R,
        "gate":   gate,
        "mom14":  mom14,
        "mom7":   mom7,
        "mom30":  mom30,
        "sma200": sma200,
        "sma50":  sma50,
        "vol20":  vol20,
    }
    cov = pd.DataFrame(cov_rows).set_index("sym") if cov_rows else pd.DataFrame()
    return ind, list(panels.keys()), cov


# ---------------------------------------------------------------------------
# Selection helpers (causal)
# ---------------------------------------------------------------------------

def _quality_score_at(ind: dict, date: pd.Timestamp) -> pd.Series:
    """Composite quality at a single date (causal: only uses data up to `date`).

    quality = 0.5 * mom14_rank + 0.25 * (1-vol_rank) + 0.25 * above_sma50
    All ranks are computed over AVAILABLE history up to `date`.
    """
    C, R, sma50, mom14, vol20 = (ind["C"], ind["R"], ind["sma50"],
                                  ind["mom14"], ind["vol20"])
    # slice to data known at `date`
    c_d    = C.loc[:date].iloc[-1]
    s50_d  = sma50.loc[:date].iloc[-1]
    m14_d  = mom14.loc[:date].iloc[-1]
    vol_d  = vol20.loc[:date].iloc[-1]

    # cross-sectional rank (only on non-NaN assets)
    valid  = m14_d.dropna().index.intersection(vol_d.dropna().index)
    if len(valid) < 2:
        return pd.Series(np.nan, index=C.columns)

    m14_r  = m14_d[valid].rank(pct=True)
    vol_r  = vol_d[valid].rank(pct=True)
    above  = (c_d[valid] > s50_d[valid]).astype(float).fillna(0.0)
    q      = m14_r * 0.5 + (1 - vol_r) * 0.25 + above * 0.25
    return q.reindex(C.columns)  # NaN for unlisted/missing


def _pick_topk(ind: dict, signal_key: str, date: pd.Timestamp,
               K: int, use_gate: bool, quality: bool = False) -> pd.Series:
    """Select top-K assets at `date`, return EW weight vector.

    CAUSAL: only uses ind[...].loc[:date].iloc[-1].
    """
    gate = ind["gate"]
    g_d  = gate.loc[:date].iloc[-1]  # gate at date

    if quality:
        score = _quality_score_at(ind, date)
    else:
        score_df = ind[signal_key]
        score = score_df.loc[:date].iloc[-1]

    if use_gate:
        elig_mask = g_d.fillna(False)
        score = score.where(elig_mask, other=np.nan)

    score = score.dropna()
    if score.empty:
        return pd.Series(0.0, index=ind["C"].columns)

    top = score.nlargest(min(K, len(score)))
    w   = pd.Series(0.0, index=ind["C"].columns)
    w[top.index] = 1.0 / len(top)
    return w


# ---------------------------------------------------------------------------
# Slice evaluation
# ---------------------------------------------------------------------------

def _slice_ret(ind: dict, weights: pd.Series,
               sl_start: pd.Timestamp) -> float:
    """7-day compound return of weights portfolio from sl_start."""
    R   = ind["R"]
    idx = R.index
    end_d = sl_start + pd.Timedelta(days=SLICE_DAYS)
    mask  = (idx >= sl_start) & (idx < end_d)
    if not mask.any():
        return np.nan
    w_arr = weights.reindex(R.columns).fillna(0.0).to_numpy()
    rmat  = R.loc[mask].fillna(0.0).to_numpy()
    port  = rmat @ w_arr
    exposure = w_arr.sum()
    return float(np.prod(1 + port) - 1) - exposure * TAKER_RT / 2.0


def _bh_ret(ind: dict, sl_start: pd.Timestamp) -> float:
    """EW buy-hold return over the slice (fixed-EW over all loaded assets)."""
    R   = ind["R"]
    idx = R.index
    end_d = sl_start + pd.Timedelta(days=SLICE_DAYS)
    mask  = (idx >= sl_start) & (idx < end_d)
    if not mask.any():
        return np.nan
    # fixed-EW: fillna(0) = cash for missing assets
    ew = R.loc[mask].fillna(0.0).mean(axis=1)
    return float(np.prod(1 + ew) - 1)


# ---------------------------------------------------------------------------
# Down-week analysis helper
# ---------------------------------------------------------------------------

def _down_week_stats(slice_rets: list[float], bh_rets: list[float]) -> dict:
    """Split returns into up-week and down-week (by reference BH), report stats."""
    rets = np.array(slice_rets)
    bh   = np.array(bh_rets)
    up   = rets[bh >= 0]
    dn   = rets[bh <  0]
    return {
        "n_up":        int((bh >= 0).sum()),
        "n_dn":        int((bh <  0).sum()),
        "win_up":      float(np.mean(up > 0) * 100) if len(up) else np.nan,
        "win_dn":      float(np.mean(dn > 0) * 100) if len(dn) else np.nan,
        "mean_up":     float(np.mean(up) * 100)     if len(up) else np.nan,
        "mean_dn":     float(np.mean(dn) * 100)     if len(dn) else np.nan,
        "cash_rate_dn":float(np.mean(dn == 0) * 100) if len(dn) else np.nan,
    }


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _summary(slice_rets: list[float]) -> dict:
    rets = np.array(slice_rets)
    mask = (rets > np.percentile(rets, 5)) & (rets < np.percentile(rets, 95))
    return {
        "n":          len(rets),
        "win_pct":    float(np.mean(rets > 0) * 100),
        "mean_pct":   float(np.mean(rets) * 100),
        "median_pct": float(np.median(rets) * 100),
        "trim_mean":  float(np.mean(rets[mask]) * 100),
        "p5_pct":     float(np.percentile(rets, 5) * 100),
        "p95_pct":    float(np.percentile(rets, 95) * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("WIDER BENCH ENGINE TOURNAMENT: u10 / u30 / u50")
    print(f"  N_SLICES={N_SLICES}  SLICE_DAYS={SLICE_DAYS}  TAKER_RT={TAKER_RT}")
    print(f"  Data: {DATA_START} -> {DATA_END}")
    print(f"  WIN CONDITION: win% > 55 OR mean% > 2.97 (u10 BH reference)")
    print("=" * 80)

    # ---- Load universes ----
    print(f"\n[1] Loading u10 ...")
    ind10 = ml.load(start=DATA_START, end=DATA_END)
    loaded10 = list(ind10["C"].columns)
    # add vol20 and sma50 to ind10 for quality scoring
    C10 = ind10["C"]; R10 = ind10["R"]
    ind10["vol20"] = R10.rolling(20, min_periods=10).std()
    ind10["sma50"] = C10.rolling(50, min_periods=50).mean()
    print(f"  Loaded {len(loaded10)}: {loaded10}")

    print(f"\n[2] Loading u30 ...")
    ind30, loaded30, cov30 = _load_universe(U30_SYMS, DATA_START, DATA_END)
    print(f"  Loaded {len(loaded30)} symbols.")

    print(f"\n[3] Loading u50 ...")
    ind50, loaded50, cov50 = _load_universe(U50_SYMS, DATA_START, DATA_END)
    print(f"  Loaded {len(loaded50)} symbols.")

    print(f"\n[4] Coverage manifest (bars 2020-2022):")
    print(cov50.to_string())

    # ---- Slice starts (common for all universes) ----
    idx_ref = ind10["C"].index
    valid   = idx_ref[
        (idx_ref >= idx_ref[0] + pd.Timedelta(days=MIN_WARM)) &
        (idx_ref <  idx_ref[-1] - pd.Timedelta(days=SLICE_DAYS + 5))
    ]
    rng     = np.random.default_rng(RNG_SEED)
    chosen  = rng.choice(len(valid), min(N_SLICES, len(valid)), replace=False)
    chosen.sort()
    sl_starts = [valid[i] for i in chosen]
    print(f"\n[5] Slices: N={len(sl_starts)}, from {sl_starts[0].date()} to {sl_starts[-1].date()}")

    # ---- Strategy names & accumulators ----
    strats: dict[str, dict] = {
        "u10_BH":           {"ind": ind10, "rets": [], "bh": []},
        "u30_BH":           {"ind": ind30, "rets": [], "bh": []},
        "u50_BH":           {"ind": ind50, "rets": [], "bh": []},
        # u10 active
        "u10_mom14_K5_g":   {"ind": ind10, "rets": [], "bh": []},
        "u10_mom14_K3_ng":  {"ind": ind10, "rets": [], "bh": []},
        "u10_quality_K5_g": {"ind": ind10, "rets": [], "bh": [], "quality": True},
        # u30 active
        "u30_mom14_K5_g":   {"ind": ind30, "rets": [], "bh": []},
        "u30_mom14_K10_g":  {"ind": ind30, "rets": [], "bh": []},
        "u30_mom14_K5_ng":  {"ind": ind30, "rets": [], "bh": []},
        "u30_quality_K5_g": {"ind": ind30, "rets": [], "bh": [], "quality": True},
        "u30_quality_K3_g": {"ind": ind30, "rets": [], "bh": [], "quality": True},
        # u50 active
        "u50_mom14_K5_g":   {"ind": ind50, "rets": [], "bh": []},
        "u50_mom14_K10_g":  {"ind": ind50, "rets": [], "bh": []},
        "u50_mom14_K5_ng":  {"ind": ind50, "rets": [], "bh": []},
        "u50_quality_K5_g": {"ind": ind50, "rets": [], "bh": [], "quality": True},
        "u50_quality_K3_g": {"ind": ind50, "rets": [], "bh": [], "quality": True},
    }

    # Parse strategy names into params
    def _parse(name: str) -> tuple | None:
        """Return (universe, signal, K, gate, quality) or None for BH."""
        if name.endswith("_BH"):
            return None
        parts = name.split("_")
        uni   = parts[0]   # u10 / u30 / u50
        sig   = parts[1]   # mom14 / quality
        Kstr  = parts[2]   # K3 / K5 / K10
        K     = int(Kstr[1:])
        gated = parts[3] == "g" if len(parts) > 3 else True
        qual  = (sig == "quality")
        signal_key = "mom14" if not qual else "mom14"  # quality uses helper
        return (uni, signal_key, K, gated, qual)

    print("\n[6] Running slices ...")
    for i, sl_start in enumerate(sl_starts):
        if i % 50 == 0:
            print(f"  slice {i}/{len(sl_starts)} ({sl_start.date()})")

        # BH references
        bh10 = _bh_ret(ind10, sl_start)
        bh30 = _bh_ret(ind30, sl_start)
        bh50 = _bh_ret(ind50, sl_start)
        if np.isnan(bh10) or np.isnan(bh30) or np.isnan(bh50):
            continue

        strats["u10_BH"]["rets"].append(bh10); strats["u10_BH"]["bh"].append(bh10)
        strats["u30_BH"]["rets"].append(bh30); strats["u30_BH"]["bh"].append(bh10)
        strats["u50_BH"]["rets"].append(bh50); strats["u50_BH"]["bh"].append(bh10)

        # signal date = last known bar before sl_start
        look_d = sl_start - pd.Timedelta(days=1)
        idx10  = ind10["C"].index
        mask10 = idx10 <= look_d
        if not mask10.any():
            continue
        sig_date = idx10[mask10][-1]

        for name, cfg in strats.items():
            if name.endswith("_BH"):
                continue
            params = _parse(name)
            if params is None:
                continue
            uni, signal_key, K, gated, qual = params
            ind_u = cfg["ind"]
            w = _pick_topk(ind_u, signal_key, sig_date, K, gated, quality=qual)
            r = _slice_ret(ind_u, w, sl_start)
            if not np.isnan(r):
                cfg["rets"].append(r)
                cfg["bh"].append(bh10)  # compare to u10 BH throughout

    # ---- Summarize ----
    print("\n[7] Summarizing ...")
    summaries = {}
    down_week = {}
    for name, cfg in strats.items():
        if not cfg["rets"]:
            continue
        summaries[name] = _summary(cfg["rets"])
        summaries[name]["beat_u10bh_pct"] = float(
            np.mean(np.array(cfg["rets"]) > np.array(cfg["bh"])) * 100
        )
        down_week[name] = _down_week_stats(cfg["rets"], cfg["bh"])

    # ---- Markdown results table ----
    print()
    print("=" * 110)
    print("RESULTS -- Random 7d slice eval, walk-forward OOS, N=350, taker cost")
    print(f"WIN CONDITION: win% > 55.0 OR mean% > 2.97 (u10 BH reference)")
    print("=" * 110)
    hdr = (f"{'Strategy':<26} | {'N':>3} | {'Win%':>5} | {'Mean%':>6} | "
           f"{'Trim%':>5} | {'Median%':>7} | {'Beat-BH%':>8} | {'P5%':>5} | {'P95%':>5} | PASS?")
    print(hdr)
    print("-" * 110)

    order = [
        "u10_BH", "u30_BH", "u50_BH",
        "u10_mom14_K5_g", "u10_mom14_K3_ng", "u10_quality_K5_g",
        "u30_mom14_K5_g", "u30_mom14_K10_g", "u30_mom14_K5_ng",
        "u30_quality_K5_g", "u30_quality_K3_g",
        "u50_mom14_K5_g", "u50_mom14_K10_g", "u50_mom14_K5_ng",
        "u50_quality_K5_g", "u50_quality_K3_g",
    ]
    for name in order:
        if name not in summaries:
            continue
        s = summaries[name]
        win_pass  = s["win_pct"]  > 55.0
        mean_pass = s["mean_pct"] > 2.97
        passed    = "YES" if (win_pass or mean_pass) else "no "
        is_bh     = name.endswith("_BH")
        passed    = "---" if is_bh else passed
        print(f"{name:<26} | {s['n']:>3} | {s['win_pct']:>5.1f} | {s['mean_pct']:>6.2f} | "
              f"{s['trim_mean']:>5.2f} | {s['median_pct']:>7.2f} | {s['beat_u10bh_pct']:>8.1f} | "
              f"{s['p5_pct']:>5.1f} | {s['p95_pct']:>5.1f} | {passed}")

    # ---- Down-week behavior table ----
    print()
    print("=" * 110)
    print("DOWN-WEEK BEHAVIOR (u10 BH < 0 = down-week; ideal engine = cash in down weeks)")
    print("win_dn = % of down-week slices where engine returned > 0 (want LOW, engine should be flat/cash)")
    print("mean_dn = avg return on down-week slices (want ~0 = cash, not -BH)")
    print("=" * 110)
    hdr2 = (f"{'Strategy':<26} | {'N_up':>4} | {'N_dn':>4} | {'Win%_up':>7} | "
            f"{'Win%_dn':>7} | {'Mean%_up':>8} | {'Mean%_dn':>8} | Cash-dn%")
    print(hdr2)
    print("-" * 110)
    for name in order:
        if name not in down_week:
            continue
        dw = down_week[name]
        print(f"{name:<26} | {dw['n_up']:>4} | {dw['n_dn']:>4} | "
              f"{dw['win_up']:>7.1f} | {dw['win_dn']:>7.1f} | "
              f"{dw['mean_up']:>8.2f} | {dw['mean_dn']:>8.2f} | "
              f"{dw['cash_rate_dn']:>7.1f}%")

    # ---- Universe width comparison ----
    print()
    print("=" * 110)
    print("UNIVERSE WIDTH COMPARISON: does more dispersion help?")
    print("=" * 110)
    pairs = [
        ("u10_mom14_K5_g",  "u30_mom14_K5_g",  "u50_mom14_K5_g",  "mom14 K5 gated"),
        ("u10_mom14_K3_ng", "u30_mom14_K5_ng",  "u50_mom14_K5_ng", "mom14 K5 no-gate"),
        ("u10_quality_K5_g","u30_quality_K5_g", "u50_quality_K5_g","quality K5 gated"),
    ]
    for k10, k30, k50, label in pairs:
        if not all(k in summaries for k in [k10, k30, k50]):
            continue
        s10, s30, s50 = summaries[k10], summaries[k30], summaries[k50]
        print(f"\n  {label}")
        print(f"    u10: win={s10['win_pct']:.1f}%  mean={s10['mean_pct']:.2f}%  "
              f"trim={s10['trim_mean']:.2f}%  beat-BH={s10['beat_u10bh_pct']:.1f}%")
        print(f"    u30: win={s30['win_pct']:.1f}%  mean={s30['mean_pct']:.2f}%  "
              f"trim={s30['trim_mean']:.2f}%  beat-BH={s30['beat_u10bh_pct']:.1f}%")
        print(f"    u50: win={s50['win_pct']:.1f}%  mean={s50['mean_pct']:.2f}%  "
              f"trim={s50['trim_mean']:.2f}%  beat-BH={s50['beat_u10bh_pct']:.1f}%")
        dw_u10u30 = s30['win_pct'] - s10['win_pct']
        dw_u10u50 = s50['win_pct'] - s10['win_pct']
        dm_u10u30 = s30['mean_pct'] - s10['mean_pct']
        dm_u10u50 = s50['mean_pct'] - s10['mean_pct']
        print(f"    delta u30-u10: win={dw_u10u30:+.1f}pp  mean={dm_u10u30:+.2f}pp")
        print(f"    delta u50-u10: win={dw_u10u50:+.1f}pp  mean={dm_u10u50:+.2f}pp")

    # ---- Verdict ----
    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)
    u10bh = summaries.get("u10_BH", {})
    print(f"\n  BH reference (measured): win={u10bh.get('win_pct',0):.1f}%  "
          f"mean={u10bh.get('mean_pct',0):.2f}%/slice  N={u10bh.get('n',0)}")
    print(f"  BH reference (spec):     win=~55%  mean=~+2.97%/slice")

    winners = [(n, s) for n, s in summaries.items()
               if not n.endswith("_BH") and s["n"] >= 50
               and (s["win_pct"] > 55.0 or s["mean_pct"] > 2.97)]
    if winners:
        print(f"\n  WIN CONDITION MET by {len(winners)} engine(s):")
        for n, s in sorted(winners, key=lambda x: -x[1]["win_pct"]):
            print(f"    {n}: win={s['win_pct']:.1f}%  mean={s['mean_pct']:.2f}%")
    else:
        print("\n  WIN CONDITION NOT MET by any engine (all < 55% win / < 2.97% mean)")

    # Wider-bench summary
    u50k5_g = summaries.get("u50_mom14_K5_g", {})
    u10k5_g = summaries.get("u10_mom14_K5_g", {})
    if u50k5_g and u10k5_g:
        dw = u50k5_g["win_pct"] - u10k5_g["win_pct"]
        dm = u50k5_g["mean_pct"] - u10k5_g["mean_pct"]
        print(f"\n  WIDER BENCH DELTA (u50 K5 vs u10 K5):")
        print(f"    win-rate: {dw:+.1f}pp  |  mean-return: {dm:+.2f}pp")
        if dw > 3:
            print("    -> Wider bench MEANINGFULLY RAISES win-rate")
        elif dw > 1:
            print("    -> Wider bench modestly raises win-rate")
        else:
            print("    -> Wider bench does NOT raise win-rate")
        if dm > 0.5:
            print("    -> Wider bench RAISES mean return materially")
        elif dm > 0:
            print("    -> Wider bench marginally raises mean return")
        else:
            print("    -> Wider bench LOWERS mean return (dilution / late-listing assets)")

    # Down-week cashout summary
    dw_u10 = down_week.get("u10_BH", {})
    best_dn_win = None
    best_dn_nm  = None
    for n, dw_d in down_week.items():
        if n.endswith("_BH"):
            continue
        w_dn = dw_d.get("win_dn", 100)
        if best_dn_win is None or w_dn < best_dn_win:
            best_dn_win = w_dn
            best_dn_nm  = n
    print(f"\n  DOWN-WEEK ANALYSIS:")
    print(f"    u10 BH mean on down weeks: {dw_u10.get('mean_dn', np.nan):.2f}%  "
          f"(N={dw_u10.get('n_dn',0)})")
    print(f"    Best cash-in-down-week engine: {best_dn_nm} "
          f"(win_dn={best_dn_win:.1f}%)")
    gated_engines = [(n, d) for n, d in down_week.items()
                     if not n.endswith("_BH") and "_g" in n]
    print(f"    Gated engines (gate=C>sma200) down-week wins: "
          + "  ".join(f"{n}={d['win_dn']:.0f}%" for n, d in gated_engines[:6]))

    print(f"\n  Universes loaded: u10={len(loaded10)}  u30={len(loaded30)}  u50={len(loaded50)}")
    print(f"  Runtime: {time.time()-t0:.1f}s")

    # ---- Save JSON ----
    out_dir  = ROOT.parent / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"u50_wider_bench_engine_{ts}.json"
    payload  = {
        "meta": {
            "script":      "u50_wider_bench_engine.py",
            "n_slices":    len(sl_starts),
            "slice_days":  SLICE_DAYS,
            "data_start":  DATA_START,
            "data_end":    DATA_END,
            "cost_rt":     TAKER_RT,
            "seed":        RNG_SEED,
            "loaded_u10":  loaded10,
            "loaded_u30":  loaded30,
            "loaded_u50":  loaded50,
        },
        "summaries":  summaries,
        "down_week":  down_week,
        "bh_ref":     u10bh,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
