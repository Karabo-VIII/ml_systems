"""JANUARY 7-DAY-SLICE runner for the MA ORACLE (ma_oracle_engine.py).

WHAT THIS IS (and is NOT)
-------------------------
A THIN DRIVER over the original `ma_oracle_engine.MAOracleEngine`. It does NOT invent a new metric --
it runs the SAME oracle (rank the top-N performers, then for each one find the best-capturing MA-family
config + capture_rate -- the "oracle MA") but evaluated on each 7-DAY CALENDAR SLICE of January:

    Jan (1-7), (8-14), (15-21), (22-28)  -- the whole month in equal 7-day slices,

for EACH year and EACH timeframe {15m,30m,1h,4h,1d}. It reuses the engine's exact MA grid
(SMA+EMA, fast(5,10,20) x slow(20,50,100)), its causal cross logic, and its capture definition --
verified bit-equal to MAOracleEngine.best_ma_capture on daily (see --reconcile).

The original engine is DAILY-native (keys off the calendar `date` column). This driver adds the only
missing piece -- cadence-aware loading keyed off `timestamp` for intraday (the `date` column repeats
intraday and would collapse to daily) -- without modifying the original.

OUTPUT (per cadence,year,slice): the oracle table (top-N performers + best_ti + capture_rate) and the
winning-config DNA (the modal best MA config = "the oracle MA" for that slice). Plots per timeframe show
the oracle MA across the 4 slices of each January.

HINDSIGHT UPPER BOUND -- descriptive, not a tradeable signal. (Config CHOICE is hindsight; each config's
MA + cross signal is causal, past-only.)

CLI:
    python src/oracle/jan_slice_oracle.py                 # full: all cadences, 2020-2026, top-25
    python src/oracle/jan_slice_oracle.py --reconcile     # prove parity vs ma_oracle_engine on daily
    python src/oracle/jan_slice_oracle.py --cadences 1h,1d --years 2024,2025 --top-n 25
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(SRC / "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# reuse the ORIGINAL engine's grid + cross logic + capture primitives (do NOT reimplement them)
from oracle.ma_oracle_engine import (  # noqa: E402
    MAOracleEngine, _build_ma_grid, _sma, _ema, _crosses,
    DEFAULT_FAST, DEFAULT_SLOW, _last_idx_le, _to_date,
)
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

HINDSIGHT_LABEL = "HINDSIGHT UPPER BOUND -- descriptive, not a tradeable signal."
ALL_CADENCES = ["15m", "30m", "1h", "4h", "1d"]
# 7 days expressed in bars per cadence (= the slice length AND the perfect-entry window, per the engine).
BARS_PER_DAY = {"15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
SLICES = [("W1", 1, 7), ("W2", 8, 14), ("W3", 15, 21), ("W4", 22, 28)]
GRID = _build_ma_grid(DEFAULT_FAST, DEFAULT_SLOW)   # SMA+EMA, fast(5,10,20) x slow(20,50,100), fast<slow


def _ema_vec(x: np.ndarray, span: int) -> np.ndarray:
    """Vectorized EMA that REPRODUCES ma_oracle_engine._ema (SMA-seeded at index span-1) bit-for-bit,
    via pandas ewm(adjust=False) with the seed forced into position 0. ~10x faster on intraday."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < span:
        return out
    alpha = 2.0 / (span + 1.0)
    tail = x[span - 1:].astype(float).copy()
    tail[0] = float(np.mean(x[:span]))
    out[span - 1:] = pd.Series(tail).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return out


def _load_series(loader, sym, cadence):
    """(datetime64 array, close float array) sorted; timestamp-keyed for intraday, date for daily."""
    try:
        df = loader.load(sym, cadence=cadence, features=["close", "date", "timestamp"])
    except Exception:
        return None
    cols = df.columns
    if "close" not in cols:
        return None
    if cadence == "1d" or "timestamp" not in cols:
        if "date" not in cols:
            return None
        df = df.select(["date", "close"]).drop_nulls().unique(subset=["date"], keep="last").sort("date")
        dt = pd.to_datetime(df["date"].to_list())
    else:
        df = df.select(["timestamp", "close"]).drop_nulls().unique(subset=["timestamp"], keep="last").sort("timestamp")
        dt = pd.to_datetime(np.asarray(df["timestamp"].to_list(), dtype="int64"), unit="ms")
    close = df["close"].to_numpy().astype(float)
    if len(close) < max(DEFAULT_SLOW) + 2:
        return None
    return np.asarray(dt), close


def _best_ma_capture_windowed(closes, ma_cache, ws, q):
    """EXACT port of MAOracleEngine.best_ma_capture, but for an explicit bar window [ws, q] instead of a
    date+lookback. Returns (best_ti, fast, slow, captured, perfect, capture_rate, in_position).
    MAs are read from ma_cache (computed on the FULL past series -> warmup lookback before the slice is
    allowed; only the config CHOICE is hindsight, the signal is causal). q = slice-end bar (query bar)."""
    cq = closes[q]
    window_closes = closes[ws:q + 1]
    c_min = float(np.min(window_closes))
    perfect = (cq - c_min) / c_min if c_min > 0 else 0.0
    best = None  # (captured, fam, f, s, entry_idx)
    for (fam, f, s) in GRID:
        ma_f, ma_s = ma_cache[(fam, f)], ma_cache[(fam, s)]
        spread = ma_f[:q + 1] - ma_s[:q + 1]          # nothing past q is read
        golden, death = _crosses(spread)
        if not golden:
            continue
        last_golden = golden[-1]
        if [d for d in death if d > last_golden]:
            continue                                   # closed before/at q -> flat
        c_entry = closes[last_golden]
        if c_entry <= 0 or np.isnan(c_entry):
            continue
        captured = (cq - c_entry) / c_entry
        if best is None or captured > best[0]:
            best = (captured, fam, f, s, last_golden)
    if best is None:
        return None, None, None, 0.0, perfect, 0.0, False
    captured, fam, f, s, _ = best
    cap_rate = max(0.0, min(1.0, captured / perfect)) if perfect > 0 else 0.0
    return f"{fam}({f},{s})", f, s, float(captured), float(perfect), float(cap_rate), True


def run(cadences, years, universe="u100", top_n=25):
    loader = ChimeraLoader()
    syms = loader.universes.list(universe)
    rows = []
    for cad in cadences:
        print(f"  [{cad}] scanning {len(syms)} assets ...", flush=True)
        bpd = BARS_PER_DAY[cad]
        # cache per-asset: datetime, close, MA grid (computed ONCE on the full series)
        data = {}
        for sym in syms:
            loaded = _load_series(loader, sym, cad)
            if loaded is None:
                continue
            dt, close = loaded
            ma_cache = {}
            for w in set(DEFAULT_FAST) | set(DEFAULT_SLOW):
                ma_cache[("SMA", w)] = _sma(close, w)
                ma_cache[("EMA", w)] = _ema_vec(close, w)
            yrs = dt.astype("datetime64[Y]").astype(int) + 1970
            mon = dt.astype("datetime64[M]").astype(int) % 12 + 1
            day = (dt.astype("datetime64[D]") - dt.astype("datetime64[M]")).astype(int) + 1
            data[sym] = (close, ma_cache, yrs, mon, day)
        # per (year, slice): rank top-N performers over the slice, then oracle MA per performer
        for year in years:
            for (wk, dlo, dhi) in SLICES:
                # gather each asset's slice [ws,q] + trailing perf
                perf = []
                slice_idx = {}
                for sym, (close, ma_cache, yrs, mon, day) in data.items():
                    idx = np.where((yrs == year) & (mon == 1) & (day >= dlo) & (day <= dhi))[0]
                    if idx.size < 2:
                        continue
                    ws, q = int(idx[0]), int(idx[-1])
                    if close[ws] <= 0 or np.isnan(close[ws]) or np.isnan(close[q]):
                        continue
                    perf.append((close[q] / close[ws] - 1.0, sym))
                    slice_idx[sym] = (ws, q)
                if not perf:
                    continue
                perf.sort(reverse=True)
                top = perf[:top_n]
                per_perf = []
                for trail, sym in top:
                    close, ma_cache, *_ = data[sym]
                    ws, q = slice_idx[sym]
                    bt, f, s, cap, perfect, cr, inpos = _best_ma_capture_windowed(close, ma_cache, ws, q)
                    if inpos:
                        per_perf.append({"sym": sym, "trailing_perf": round(trail, 4), "best_ti": bt,
                                         "fast": f, "slow": s, "captured_return": round(cap, 4),
                                         "perfect_return": round(perfect, 4), "capture_rate": round(cr, 4)})
                if not per_perf:
                    continue
                cfgs = [p["best_ti"] for p in per_perf]
                cnt = Counter(cfgs)
                modal = cnt.most_common(1)[0][0]
                mf, ms = next((p["fast"], p["slow"]) for p in per_perf if p["best_ti"] == modal)
                rows.append({
                    "cadence": cad, "year": year, "slice": wk, "days": f"{dlo}-{dhi}",
                    "n_top_in_position": len(per_perf), "n_ranked": len(top),
                    "oracle_ma_modal": modal, "modal_fast": mf, "modal_slow": ms,
                    "modal_count": cnt[modal],
                    "mean_capture_rate": round(float(np.mean([p["capture_rate"] for p in per_perf])), 4),
                    "mean_captured_return": round(float(np.mean([p["captured_return"] for p in per_perf])), 4),
                    "dna_top5": "; ".join(f"{k}:{v}" for k, v in cnt.most_common(5)),
                    "performers": per_perf,
                })
        print(f"  [{cad}] slices with an oracle MA: {sum(1 for r in rows if r['cadence']==cad)}", flush=True)
    return rows


def _reconcile():
    """Prove the windowed capture == MAOracleEngine.best_ma_capture on DAILY for a real (sym, date)."""
    eng = MAOracleEngine()
    loader = ChimeraLoader()
    sym = "BTCUSDT"
    loaded = _load_series(loader, sym, "1d")
    dt, close = loaded
    ma_cache = {}
    for w in set(DEFAULT_FAST) | set(DEFAULT_SLOW):
        ma_cache[("SMA", w)] = _sma(close, w)
        ma_cache[("EMA", w)] = _ema_vec(close, w)
    # pick a query day with >= 30 bars of history; lookback 7
    q = 400
    ws = q - 7
    qdate = str(pd.Timestamp(dt[q]).date())
    bt, f, s, cap, perfect, cr, inpos = _best_ma_capture_windowed(close, ma_cache, ws, q)
    ref = eng.best_ma_capture(sym, qdate, lookback_days=7)
    ok = (bt == ref["best_ti"]) and abs(cap - ref["captured_return"]) < 1e-6 and abs(cr - ref["capture_rate"]) < 1e-6
    print("=== RECONCILE vs MAOracleEngine.best_ma_capture (BTC 1d, q=Jan-day, lookback 7) ===")
    print(f"  query date     : {qdate}")
    print(f"  driver  best_ti={bt}  captured={cap:.6f}  capture_rate={cr:.6f}")
    print(f"  engine  best_ti={ref['best_ti']}  captured={ref['captured_return']:.6f}  capture_rate={ref['capture_rate']:.6f}")
    print(f"  PARITY: {'OK' if ok else 'MISMATCH'}")
    return ok


def _plot(rows, cadences, out_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame([{k: r[k] for k in r if k != "performers"} for r in rows])
    years = sorted(df["year"].unique())
    colors = {"modal_fast": "#1f77b4", "modal_slow": "#d62728"}
    xlab = [f"{w}\n(d{lo}-{hi})" for (w, lo, hi) in SLICES]
    bydir = out_root / "by_january"
    bydir.mkdir(parents=True, exist_ok=True)
    for year in years:
        sub_y = df[df["year"] == year]
        cads = [c for c in cadences if c in set(sub_y["cadence"])]
        if not cads:
            continue
        fig, axes = plt.subplots(len(cads), 1, figsize=(7.2, 1.9 * len(cads)), squeeze=False, sharex=True)
        for ax, cad in zip(axes[:, 0], cads):
            s = (sub_y[sub_y["cadence"] == cad].set_index("slice").reindex([w for w, _, _ in SLICES]))
            xs = list(range(len(SLICES)))
            ax.plot(xs, s["modal_fast"].to_numpy(dtype=float), marker="o", ms=7, lw=1.6,
                    color=colors["modal_fast"], label="fast")
            ax.plot(xs, s["modal_slow"].to_numpy(dtype=float), marker="o", ms=7, lw=1.6,
                    color=colors["modal_slow"], label="slow")
            # annotate capture-rate as a faint bar in the background (right axis)
            ax2 = ax.twinx()
            ax2.bar(xs, s["mean_capture_rate"].to_numpy(dtype=float), width=0.5, alpha=0.12, color="green")
            ax2.set_ylim(0, 1); ax2.set_ylabel("capt", fontsize=6); ax2.tick_params(labelsize=6)
            ax.set_ylabel(f"{cad}\nMA win"); ax.set_yticks([5, 10, 20, 50, 100]); ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.25); ax.legend(loc="upper left", fontsize=7, ncol=2)
        axes[-1, 0].set_xticks(xs); axes[-1, 0].set_xticklabels(xlab, fontsize=8)
        fig.suptitle(f"ORACLE MA per 7-day slice -- JANUARY {year} (top-{int(sub_y['n_ranked'].max())} movers)\n"
                     "modal best-capturing MA config per slice, all timeframes (faint=capture rate) -- "
                     "HINDSIGHT", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(bydir / f"oracle_ma_Jan{year}.png", dpi=110)
        plt.close(fig)
        print(f"    wrote {bydir}/oracle_ma_Jan{year}.png")


def main():
    ap = argparse.ArgumentParser(description=HINDSIGHT_LABEL)
    ap.add_argument("--cadences", default=None)
    ap.add_argument("--years", default=None)
    ap.add_argument("--universe", default="u100")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--reconcile", action="store_true")
    args = ap.parse_args()
    if args.reconcile:
        sys.exit(0 if _reconcile() else 1)

    cadences = args.cadences.split(",") if args.cadences else ALL_CADENCES
    years = [int(y) for y in args.years.split(",")] if args.years else list(range(2020, 2027))
    print("=" * 80)
    print("JANUARY 7-DAY-SLICE MA ORACLE (thin driver over ma_oracle_engine) -- " + HINDSIGHT_LABEL)
    print(f"cadences={cadences} years={years} universe={args.universe} top_n={args.top_n}")
    print(f"slices={[f'{lo}-{hi}' for _, lo, hi in SLICES]}  grid={len(GRID)} (SMA+EMA, fast{list(DEFAULT_FAST)} x slow{list(DEFAULT_SLOW)})")
    print("=" * 80)
    if not _reconcile():
        print("ABORT: driver does not reconcile with ma_oracle_engine.")
        sys.exit(2)
    print()
    rows = run(cadences, years, args.universe, args.top_n)
    if not rows:
        print("(no slices produced)")
        return
    out_root = PROJECT_ROOT / "runs" / "oracle" / "jan_slices"
    out_root.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame([{k: r[k] for k in r if k != "performers"} for r in rows])
    flat.sort_values(["cadence", "year", "slice"]).to_csv(out_root / "oracle_ma_jan_slices.csv", index=False)
    # full per-performer detail (long form)
    detail = []
    for r in rows:
        for p in r["performers"]:
            detail.append({"cadence": r["cadence"], "year": r["year"], "slice": r["slice"], **p})
    pd.DataFrame(detail).to_csv(out_root / "oracle_ma_jan_slices_performers.csv", index=False)
    print(f"\nwrote {out_root/'oracle_ma_jan_slices.csv'} ({len(flat)} slices) + performers detail")
    _plot(rows, cadences, out_root)
    print("\n--- digest: modal oracle MA per cadence (across all Jan slices) ---")
    for cad in cadences:
        s = flat[flat["cadence"] == cad]["oracle_ma_modal"]
        if len(s):
            print(f"  {cad:>3}: {s.mode().iloc[0]}  (mean capture_rate "
                  f"{flat[flat['cadence']==cad]['mean_capture_rate'].mean():.2f})")
    print("\nDONE.")


if __name__ == "__main__":
    main()
