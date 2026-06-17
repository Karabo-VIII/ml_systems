"""Random-7d-slice PRICE-ORACLE vs TI-ORACLE comparison, per timeframe.

User ask (2026-06-10): "run the decomposition/MA oracle framework that gives us the
price oracle vs TI oracle comparison; slice it for a random 7d period per time frame."

Thin runner over the existing anchor framework (`src/strat/ti_oracle_anchor.py`,
imported READ-ONLY -- that file is owned by a concurrent instance): for each cadence
{1d, 4h, 1h, 30m, 15m} draw ONE seeded-random 7-day window and report, inside it:
  PRICE ORACLE : best available long round-trip = lowest low -> highest high AFTER
                 that low within the window (same definition as the anchor spec;
                 the [2,10]% band is REPORTED, not used to discard the slice).
  TI ORACLE    : best-in-hindsight MA/EMA config from the anchor grid
                 (SMA/EMA x fast {5,10,20,50} x slow {20,50,100,200}), each config
                 simulated CAUSALLY (full-history MA warmup, golden/death cross,
                 next-bar-open fills, taker 0.24% RT) -- only the config CHOICE is
                 hindsight. Window-start already-long state enters at first open
                 (trend-in-progress capture), per the anchor's own semantics.
  CAPTURE      : ti_roi / price_roi + the winning config (DNA) + top-3 configs.

Windows are drawn INDEPENDENTLY per cadence (the ask: "a random 7d period per
time frame"), seeded for reproducibility; each window requires full MA warmup
history before it. Descriptive ANCHOR semantics inherited: hindsight by design,
no pass/fail verdict. No emoji (cp1252).

Run:
  python -m mining.oracle_slice7d --asset BTC --seed 11
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from strat.ti_oracle_anchor import (  # noqa: E402  (read-only import of the anchor)
    MoveEvent, precompute_configs, causal_ma_long_return, TAKER_RT,
    PRICE_ROI_LO, PRICE_ROI_HI,
)
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research_anchor",
    "inputs": {"chimera_ohlc": "ChimeraLoader (canonical access)"},
    "outputs": {"json": "runs/mining/oracle_slice7d_<asset>_<stamp>.json"},
    "invariants": {
        "anchor_semantics": "price/TI oracle definitions imported from ti_oracle_anchor (read-only)",
        "causal_ti": "MA signals causal; next-bar fills; only config choice is hindsight",
        "seeded_windows": "random 7d window per cadence drawn from seeded RNG, independent per cadence",
        "descriptive": "ANCHOR not a gate; no pass/fail verdict",
    },
}

CADENCE_7D_BARS = {"1d": 7, "4h": 42, "1h": 168, "30m": 336, "15m": 672}
WARMUP_BARS = 220  # slow-200 MA fully warm before any window


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def _ts_iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def slice_cadence(asset: str, cadence: str, nbars: int, rng: np.random.Generator) -> dict:
    df = ChimeraLoader().load(asset, cadence=cadence,
                              features=["open", "high", "low", "close"])
    cols = df.columns
    ts = df["timestamp"].to_numpy() if "timestamp" in cols else None
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    lo = df["low"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    n = len(c)
    if n < WARMUP_BARS + nbars + 2:
        return {"cadence": cadence, "error": f"series too short ({n} bars)"}

    s = int(rng.integers(WARMUP_BARS, n - nbars))
    e = s + nbars  # window [s, e)

    # ---- price oracle on the slice (anchor definition, band reported not enforced)
    li = int(np.argmin(lo[s:e]))
    lo_val = float(lo[s + li])
    if li + 1 >= nbars:
        price_roi, hi_idx = 0.0, s + li
    else:
        hj = int(np.argmax(h[s + li + 1:e]))
        hi_idx = s + li + 1 + hj
        price_roi = (float(h[hi_idx]) - lo_val) / lo_val

    # ---- TI oracle on the same window (full grid, causal sim, hindsight choice)
    configs = precompute_configs(c)
    per_cfg = {}
    for label, (f_arr, s_arr) in configs.items():
        per_cfg[label] = causal_ma_long_return(f_arr, s_arr, o, s, e)
    ranked = sorted(per_cfg.items(), key=lambda kv: kv[1], reverse=True)
    best_cfg, best_roi = ranked[0]
    n_traded = sum(1 for v in per_cfg.values() if abs(v) > 1e-12)

    window_ret = float(c[e - 1] / o[s] - 1.0)  # buy-window-open hold-to-end reference
    out = {
        "cadence": cadence, "bars": nbars, "win_start_idx": s,
        "window_utc": ([_ts_iso(int(ts[s])), _ts_iso(int(ts[e - 1]))] if ts is not None else None),
        "price_oracle_roi": float(price_roi),
        "price_oracle_in_2_10_band": bool(PRICE_ROI_LO <= price_roi <= PRICE_ROI_HI),
        "price_low_utc": (_ts_iso(int(ts[s + li])) if ts is not None else None),
        "price_high_utc": (_ts_iso(int(ts[hi_idx])) if ts is not None else None),
        "ti_oracle_roi": float(best_roi),
        "ti_winning_config": best_cfg,
        "ti_top3": [{"cfg": k, "roi": float(v)} for k, v in ranked[:3]],
        "ti_worst": {"cfg": ranked[-1][0], "roi": float(ranked[-1][1])},
        "n_configs_traded": int(n_traded), "n_configs": len(per_cfg),
        "capture_ratio": (float(best_roi / price_roi) if price_roi > 0 else None),
        "buy_and_hold_window": window_ret,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="random 7d slice: price vs TI oracle per cadence")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    asset = _norm_sym(args.asset)
    rng = np.random.default_rng(args.seed)

    rows = []
    for cad, nbars in CADENCE_7D_BARS.items():
        try:
            rows.append(slice_cadence(asset, cad, nbars, rng))
        except Exception as ex:
            rows.append({"cadence": cad, "error": f"{type(ex).__name__}: {ex}"})

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"tool": "oracle_slice7d", "asset": asset, "seed": args.seed,
               "taker_rt": TAKER_RT, "slices": rows,
               "note": "descriptive anchor; windows drawn independently per cadence"}
    out_path = OUT / f"oracle_slice7d_{asset}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("")
    print("=" * 100)
    print(f"RANDOM-7d SLICE: PRICE-ORACLE vs TI-ORACLE -- {asset} (seed {args.seed}; "
          f"taker {TAKER_RT*1e4:.0f}bps RT; hindsight config choice, causal signals)")
    print("=" * 100)
    print(f"{'cadence':>8} | {'window (UTC)':>33} | {'price orc':>9} | {'TI orc':>8} | "
          f"{'capture':>7} | {'B&H':>7} | winning cfg")
    print("-" * 100)
    for r in rows:
        if "error" in r:
            print(f"{r['cadence']:>8} | ERROR: {r['error']}")
            continue
        w = f"{r['window_utc'][0]} -> {r['window_utc'][1]}" if r["window_utc"] else "?"
        cap = f"{r['capture_ratio']:.2f}" if r["capture_ratio"] is not None else "n/a"
        print(f"{r['cadence']:>8} | {w:>33} | {r['price_oracle_roi']*100:8.2f}% | "
              f"{r['ti_oracle_roi']*100:7.2f}% | {cap:>7} | "
              f"{r['buy_and_hold_window']*100:6.2f}% | {r['ti_winning_config']}"
              f"{'' if r['price_oracle_in_2_10_band'] else '  [out-of-2-10-band]'}")
    print("-" * 100)
    print("per-slice TI DNA (top-3 configs by captured ROI):")
    for r in rows:
        if "error" in r:
            continue
        t3 = ", ".join(f"{d['cfg']}:{d['roi']*100:+.2f}%" for d in r["ti_top3"])
        print(f"{r['cadence']:>8} : {t3}   (configs trading: {r['n_configs_traded']}/{r['n_configs']})")
    print("=" * 100)
    print("ANCHOR semantics: TI < price oracle is EXPECTED; capture ratio is the read. "
          "One random window per cadence -- a SLICE VIEW, not a population estimate "
          "(population stats: runs/strat/ti_oracle_anchor_BTC_15m.json).")
    print(f"JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
