"""
experiments/adaptive_ma/oracle_holdtime_prefilter.py

CHEAP PRE-FILTER (no DNA fit): oracle HOLD-TIME distribution per cadence x u20.

WHY: an MA of window W lags the price by ~(W-1)/2 bars. A move that the perfect-foresight
oracle holds for only ~2 bars is OVER before even a fast MA can confirm it (the prior overseer
found 4h oracle median hold ~2 bars => MA can't time it). So before spending DNA-fit compute on a
(cadence, asset), rank cadences by oracle MEDIAN HOLD IN BARS: MA has a chance only where
median_hold_bars >> MA_lag. This falsifies the premise that a cadence offers MA-capturable moves.

REUSES the audited perfect-foresight high-capture DP (entry=open[k], exit=high[j]>k, non-overlap,
1h<=hold<7d, net taker 0.0024) from runs/research/oracle_ceiling_builder.py. We compute ONLY the
hold distribution (median/quartiles in BARS and in HOURS, bars-per-day) -- NOT AUC/IC/capture.

DATA: time bars (15m/30m/1h/4h/1d) via ChimeraLoader; event bars (range/dib/runs_volume/
adaptive_vol) via raw OHLC under data/processed/bars/<cad>/ (chimera lacks full-u20 for these).
dollar is handled as a documented special case (O(n*H) DP infeasible at 2.7M bars x 48k/7d window;
project docs already flag dollar as granularity-degenerate) -- BTC-only spot check, capped.

RWYB: .venv/Scripts/python.exe experiments/adaptive_ma/oracle_holdtime_prefilter.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from oracle_ceiling_builder import oracle_high_capture, COST_RT, MS_PER_HOUR  # noqa: E402

U20 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ZEC", "TRX", "PEPE", "ADA",
       "LINK", "SUI", "AVAX", "TAO", "FET", "ENJ", "ORDI", "NEAR", "WLD", "ENA"]

TIME_CADENCES = ["15m", "30m", "1h", "4h", "1d"]
EVENT_CADENCES = ["range", "dib", "runs_volume", "adaptive_vol"]
# rough MA-lag reference: a responsive MA(W=10) lags ~4.5 bars; SMA-cross needs the move to
# outlast BOTH the slow MA lag and the confirm bar. Use 4.5 bars as the "fast-MA lag" yardstick.
MA_FAST_LAG_BARS = 4.5


def _load_time(asset, cad):
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset + "USDT", cadence=cad, features=["open", "high"])
    ts = g["timestamp"].to_numpy().astype(np.int64)
    op = g["open"].to_numpy().astype(np.float64)
    hi = g["high"].to_numpy().astype(np.float64)
    return ts, op, hi


def _raw_files(asset, cad):
    a = asset + "USDT"
    if cad == "adaptive_vol":
        return sorted(glob.glob(str(ROOT / f"data/processed/bars/adaptive_vol/{a}_adaptive_vol.parquet")))
    if cad == "runs_volume":
        return sorted(glob.glob(str(ROOT / f"data/processed/bars/runs_volume/{a}_vol_runs_*.parquet")))
    if cad == "range":
        return sorted(glob.glob(str(ROOT / f"data/processed/bars/range/{a}_range_*.parquet")))
    if cad == "dib":
        return sorted(glob.glob(str(ROOT / f"data/processed/bars/dib/{a}_dib_*.parquet")))
    return []


def _load_event(asset, cad):
    fs = _raw_files(asset, cad)
    if not fs:
        raise FileNotFoundError(f"no raw bars for {asset} {cad}")
    df = pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")
    tcol = "bar_end_ts" if "bar_end_ts" in df.columns else "timestamp"
    ts = df[tcol].to_numpy().astype(np.int64)
    op = df["open"].to_numpy().astype(np.float64)
    hi = df["high"].to_numpy().astype(np.float64)
    return ts, op, hi


def _clean(ts, op, hi):
    # strictly-increasing ts required by searchsorted in the DP; dedupe-sort if needed
    if not np.all(np.diff(ts) > 0):
        order = np.argsort(ts, kind="stable")
        ts, op, hi = ts[order], op[order], hi[order]
        keep = np.concatenate([[True], np.diff(ts) > 0])
        ts, op, hi = ts[keep], op[keep], hi[keep]
    return ts, op, hi


def per_asset(asset, cad):
    t0 = time.time()
    if cad in TIME_CADENCES:
        ts, op, hi = _load_time(asset, cad)
    else:
        ts, op, hi = _load_event(asset, cad)
    ts, op, hi = _clean(ts, op, hi)
    n = len(op)
    span_days = float((ts[-1] - ts[0]) / (24 * 3600 * 1000)) if n > 1 else 0.0
    bars_per_day = float(n / span_days) if span_days > 0 else float("nan")

    out = {"asset": asset, "n_bars": n, "span_days": round(span_days, 1),
           "bars_per_day": round(bars_per_day, 2), "secs": None}
    # FLOORED (audited default 1h min-hold = excludes sub-hour scalps, parity with prior 4h work)
    # vs UNFLOORED (min-hold=0 = genuine move structure, strips the floor x bar-density artifact).
    for tag, minh in (("floored", 1.0), ("unfloored", 0.0)):
        f, trades = oracle_high_capture(ts, op, hi, min_hold_hours=minh)
        if not trades:
            out[tag] = {"n_trades": 0, "note": "no oracle trades"}
            continue
        hb = np.array([j - i for i, j in trades], dtype=np.int64)
        hh = np.array([(ts[j] - ts[i]) / MS_PER_HOUR for i, j in trades], dtype=np.float64)
        out[tag] = {
            "n_trades": int(len(trades)),
            "hold_bars": {"median": float(np.median(hb)), "mean": float(hb.mean()),
                          "p25": float(np.percentile(hb, 25)), "p75": float(np.percentile(hb, 75)),
                          "min": int(hb.min()), "max": int(hb.max())},
            "hold_hours": {"median": float(np.median(hh)), "p25": float(np.percentile(hh, 25)),
                           "p75": float(np.percentile(hh, 75))},
            "pct_holds_at_1bar": float(100.0 * np.mean(hb <= 1)),
        }
    out["secs"] = round(time.time() - t0, 1)
    return out


def _agg_med(vals):
    a = np.asarray([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "median": float(np.median(a)), "mean": float(a.mean()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def main():
    t_start = time.time()
    cadences = TIME_CADENCES + EVENT_CADENCES
    print(f"[ORACLE HOLD-TIME PRE-FILTER] cadences={cadences}  assets={len(U20)}  cost_rt={COST_RT}")
    print(f"  metric = oracle median HOLD IN BARS (MA-fast-lag yardstick = {MA_FAST_LAG_BARS} bars)\n")
    per_cad = {}
    for cad in cadences:
        rows = []
        for a in U20:
            try:
                r = per_asset(a, cad)
            except Exception as e:
                r = {"asset": a, "error": repr(e)[:160]}
            rows.append(r)
        ok = [r for r in rows if "hold_bars" in r]
        asset_med_bars = [r["hold_bars"]["median"] for r in ok]
        asset_med_hrs = [r["hold_hours"]["median"] for r in ok]
        agg = {
            "n_ok": len(ok), "n_err": len(rows) - len(ok),
            "median_hold_bars_acrossassets": _agg_med(asset_med_bars),
            "median_hold_hours_acrossassets": _agg_med(asset_med_hrs),
            "bars_per_day": _agg_med([r["bars_per_day"] for r in ok]),
            "n_trades": _agg_med([r["n_trades"] for r in ok]),
            "n_bars": _agg_med([r["n_bars"] for r in ok]),
        }
        per_cad[cad] = {"aggregate": agg, "per_asset": rows}
        mb = agg["median_hold_bars_acrossassets"]
        mh = agg["median_hold_hours_acrossassets"]
        print(f"  {cad:13} ok={len(ok):2}/20  median_hold_bars={mb.get('median'):6.1f} "
              f"[p25={mb.get('p25'):5.1f} p75={mb.get('p75'):6.1f}]  "
              f"median_hold_hrs={mh.get('median'):7.1f}  bars/day={agg['bars_per_day'].get('median'):7.1f}")

    # ---- RANK cadences by median-of-asset-median hold in BARS (the MA-capturability proxy) ----
    rank = sorted(per_cad.items(), key=lambda kv: -(kv[1]["aggregate"]["median_hold_bars_acrossassets"].get("median") or 0))
    print("\n" + "=" * 92)
    print("CADENCE RANK by oracle median HOLD-IN-BARS  (higher = more room for an MA to time the move)")
    print("=" * 92)
    print(f"  {'rank':4} {'cadence':13} {'med_hold_bars':>13} {'bars/MA_lag':>11} {'med_hold_hrs':>12} "
          f"{'MA_can_time?':>12}")
    ranked_out = []
    for k, (cad, d) in enumerate(rank, 1):
        mb = d["aggregate"]["median_hold_bars_acrossassets"].get("median") or 0.0
        mh = d["aggregate"]["median_hold_hours_acrossassets"].get("median") or 0.0
        ratio = mb / MA_FAST_LAG_BARS
        verdict = "YES" if ratio >= 2.0 else ("MARGINAL" if ratio >= 1.0 else "NO (too sharp)")
        print(f"  {k:>4} {cad:13} {mb:>13.1f} {ratio:>11.2f} {mh:>12.1f} {verdict:>12}")
        ranked_out.append({"cadence": cad, "median_hold_bars": mb, "bars_per_MA_lag": round(ratio, 2),
                           "median_hold_hours": mh, "MA_can_time": verdict})

    blob = {
        "meta": {"assets": U20, "cadences": cadences, "cost_rt": COST_RT,
                 "ma_fast_lag_bars": MA_FAST_LAG_BARS,
                 "metric": "oracle perfect-foresight high-capture hold distribution (bars & hours)",
                 "reused": "runs/research/oracle_ceiling_builder.py::oracle_high_capture (audited DP)",
                 "dollar_note": ("excluded from exact DP: 2.7M bars x ~48k-bar/7d window = ~130B ops/asset "
                                 "infeasible; project docs flag dollar as granularity-degenerate for high-capture"),
                 "elapsed_secs": round(time.time() - t_start, 1)},
        "rank_by_hold_bars": ranked_out,
        "per_cadence": per_cad,
    }
    outp = ROOT / "experiments" / "adaptive_ma" / "oracle_holdtime_prefilter.json"
    outp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}  (elapsed {blob['meta']['elapsed_secs']}s)")
    return blob


if __name__ == "__main__":
    main()
