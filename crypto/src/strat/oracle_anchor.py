"""src/strat/oracle_anchor.py -- the PRICE-ORACLE vs TI-ORACLE anchoring (user spec, 2026-06-10).

THE TASK (user): for an asset, find all 2-10% MOVES (price oracle) in a 7-14 day window, across all timeframes,
genuinely anywhere on the chart; then with HINDSIGHT ask "what EMA/MA config would have maximally captured most of
that oracle ROI?" (the TI oracle). Report PRICE ORACLE vs TI ORACLE. TI oracle < price oracle, but it ANCHORS the
discovery process (price ceiling vs best-realizable-TI ceiling). Descriptive hindsight -- NO predictability claim yet.

DEFINITIONS (per asset, per cadence, per window [calendar days]):
  PRICE ORACLE  = compound of every zigzag UP-leg whose magnitude is in [lo, hi] (default 2-10%) within the window
                  -- the perfect swing-trader capturing exactly the move-sized swings (LO-harvestable). GROSS.
  TI ORACLE     = max over an EMA/MA CONFIG GRID of that config's GROSS captured ROI within the window (long when the
                  MA-signal is bullish, lagged 1 bar = past-only-vs-fill). The config is a real causal rule; only the
                  SELECTION of the best config per window uses hindsight (that is the "oracle" part). Reports the best
                  config too (the modal winner = the DNA seed for the next step).
  CAPTURE       = TI_oracle / price_oracle in [0,1] -- how much of the move-ROI the best MA config banks.

MA is computed on the FULL series (warmed up) so even a 100-bar MA is valid inside a 14-day window; capture is measured
ONLY within the window. No emoji (cp1252). Run:
  python src/strat/oracle_anchor.py --assets BTC,ETH,SOL --cadences 1d,4h,1h --window-days 14
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from strat.entry_signal_lab import load_ohlc, _zigzag_pivots   # reuse the FIXED zigzag + loader


# --- PRICE ORACLE: compound of 2-10% up-legs within a window -------------------------------------------------
def price_oracle_roi(close: np.ndarray, lo=0.02, hi=0.10) -> tuple[float, int]:
    """Compound ROI of every zigzag UP-leg with magnitude in [lo, hi]. Returns (roi_pct, n_moves)."""
    if len(close) < 3: return 0.0, 0
    piv = _zigzag_pivots(close, thr=lo)
    comp = 1.0; n = 0
    for k in range(len(piv) - 1):
        v0, v1 = piv[k][1], piv[k + 1][1]
        if v1 > v0:
            r = v1 / v0 - 1.0
            if lo <= r <= hi:
                comp *= (1 + r); n += 1
            elif r > hi:                       # a bigger move: count only the [lo,hi] portion (a "2-10% move" target)
                comp *= (1 + hi); n += 1
    return float((comp - 1.0) * 100), n


# --- TI ORACLE: best EMA/MA config's gross captured ROI in the window ----------------------------------------
def _ema(c, n): return pd.Series(c).ewm(span=n, adjust=False).mean().to_numpy()
def _sma(c, n): return pd.Series(c).rolling(n, min_periods=max(2, n // 2)).mean().to_numpy()

def _config_grid():
    cfgs = []
    for n in (5, 8, 10, 13, 20, 34, 50, 100):
        cfgs.append(("sma", (n,))); cfgs.append(("ema", (n,)))
    for f, s in [(5, 20), (8, 21), (10, 30), (13, 34), (20, 50), (50, 100), (9, 26)]:
        cfgs.append(("ema_cross", (f, s))); cfgs.append(("sma_cross", (f, s)))
    return cfgs

def _config_position(close, kind, params):
    """Past-only long/flat position array for a config (1 = long, 0 = flat)."""
    if kind in ("sma", "ema"):
        ma = (_sma if kind == "sma" else _ema)(close, params[0])
        return (close > ma).astype(float)
    f, s = params
    maf = (_ema if "ema" in kind else _sma)(close, f); mas = (_ema if "ema" in kind else _sma)(close, s)
    return (maf > mas).astype(float)

def ti_oracle_roi(close: np.ndarray, lo_idx: int, hi_idx: int) -> tuple[float, str]:
    """max over the config grid of GROSS captured ROI within [lo_idx, hi_idx]. Returns (roi_pct, best_config_label)."""
    ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1.0
    best = -1e9; best_lbl = "none"
    for kind, params in _config_grid():
        pos = _config_position(close, kind, params)
        pl = np.roll(pos, 1); pl[0] = 0.0                       # lag 1 bar (signal at close -> fill next bar)
        seg = (1.0 + pl[lo_idx:hi_idx + 1] * ret[lo_idx:hi_idx + 1])
        roi = float(np.prod(seg) - 1.0) * 100
        if roi > best:
            best = roi; best_lbl = f"{kind}{params}"
    return best, best_lbl


# --- the anchoring report ------------------------------------------------------------------------------------
def anchor_report(assets, cadences, window_days=14, lo=0.02, hi=0.10, max_windows=400):
    from collections import Counter
    rows = []
    for sym in assets:
        for cad in cadences:
            df = load_ohlc(sym if sym.endswith("USDT") else sym + "USDT", cad)
            if df is None or len(df) < 60: continue
            dates = pd.to_datetime(df["date"]).to_numpy(); close = df["close"].to_numpy(float)
            t0 = dates[0]; tend = dates[-1]; win = np.timedelta64(window_days, "D")
            caps, pos_caps, prices, tis, n_moves, best_cfgs = [], [], [], [], [], []
            wstart = t0; guard = 0
            while wstart < tend and guard < max_windows:
                wend = wstart + win
                idx = np.where((dates >= wstart) & (dates < wend))[0]
                wstart = wend; guard += 1
                if idx.size < 5: continue
                lo_i, hi_i = idx[0], idx[-1]
                p_roi, nm = price_oracle_roi(close[lo_i:hi_i + 1], lo, hi)
                if p_roi <= 0.5 or nm == 0: continue            # only windows that HAD a 2-10% move (the events)
                ti_roi, cfg = ti_oracle_roi(close, lo_i, hi_i)
                prices.append(p_roi); tis.append(ti_roi); n_moves.append(nm); best_cfgs.append(cfg)
                caps.append(ti_roi / p_roi if p_roi > 1e-9 else 0.0)
            if not prices: continue
            modal = Counter(best_cfgs).most_common(3)
            rows.append({"asset": sym, "cad": cad, "n_windows": len(prices), "moves_per_win": float(np.mean(n_moves)),
                         "price_oracle_med": float(np.median(prices)), "ti_oracle_med": float(np.median(tis)),
                         "capture_med": float(np.median(caps)), "capture_p25": float(np.percentile(caps, 25)),
                         "capture_p75": float(np.percentile(caps, 75)), "modal_cfg": modal})
    return rows


def _print(rows, window_days):
    print(f"## PRICE-ORACLE vs TI-ORACLE anchoring -- {window_days}d windows -- 2-10% up-moves -- best EMA/MA config (hindsight, GROSS)")
    print(f"   {'asset':6} {'cad':4} {'#win':>5} {'mv/win':>6} {'PRICE_orc%':>10} {'TI_orc%':>8} {'capture':>8} {'[p25-p75]':>12}  modal best config")
    for r in rows:
        modal = ", ".join(f"{c}:{n}" for c, n in r["modal_cfg"])
        print(f"   {r['asset']:6} {r['cad']:4} {r['n_windows']:>5} {r['moves_per_win']:6.1f} {r['price_oracle_med']:10.1f} "
              f"{r['ti_oracle_med']:8.1f} {r['capture_med']:8.2f} [{r['capture_p25']:.2f}-{r['capture_p75']:.2f}]  {modal}")


# =============================================================================
# DECOMPOSITION (user 2026-06-10): the TI oracle is a MULTITUDE -- decompose across STRUCTURES (price>MA, cross+flip,
# cross+mechanical-exit, price>fast>slow stack) x MA TYPES (SMA/EMA/WMA/Hull/DEMA/KAMA) x EXITS. Report which FORMS win
# when (heterogeneity), not one answer. GROSS captured ROI of the 2-10% moves, per window.
# =============================================================================
import pandas_ta as _ta
MA_TYPES = ["sma", "ema", "wma", "hma", "dema", "kama"]   # simple/exp/weighted/hull/double-exp/adaptive

def _ma(close: np.ndarray, kind: str, n: int) -> np.ndarray:
    try:
        out = getattr(_ta, kind)(pd.Series(close), length=n)
        return out.to_numpy(float) if out is not None else np.full(len(close), np.nan)
    except Exception:
        return np.full(len(close), np.nan)

def _mech_mask(entry: np.ndarray, close, high, atr, exit_kind: str, param) -> np.ndarray:
    """In-position mask: enter on entry rising-edge, exit by the mechanical rule. exit_kind: atr|time|tp."""
    n = len(close); pos = np.zeros(n); i = 1
    while i < n:
        if entry[i] and not entry[i - 1]:
            ep = close[i]; hwm = close[i]; j = i
            while j < n:
                pos[j] = 1.0
                if exit_kind == "time" and (j - i) >= param: break
                if exit_kind == "tp" and close[j] >= ep * (1 + param): break
                if exit_kind == "atr" and np.isfinite(atr[j - 1]) and close[j] <= hwm - param * atr[j - 1]: break
                hwm = max(hwm, close[j]); j += 1
            i = j + 1
        else:
            i += 1
    return pos

def _structures(close, high, atr):
    """Yield (structure, ma_type, label, position_mask). Positions are pre-lag (caller lags 1 bar)."""
    singles = (5, 8, 13, 20, 34, 50); crosses = ((5, 20), (8, 21), (10, 30), (20, 50))
    for mt in MA_TYPES:
        mas = {n: _ma(close, mt, n) for n in set(singles + tuple(x for c in crosses for x in c))}
        for n in singles:                                            # STRUCTURE 1: price > MA
            yield ("price>MA", mt, f"{mt}{n}", (close > mas[n]).astype(float))
        for f, s in crosses:
            cross = (mas[f] > mas[s]).astype(float)
            yield ("cross+flip", mt, f"{mt}{f}/{s}", cross)           # STRUCTURE 2: cross, signal-flip exit
            entry = cross
            yield ("cross+atr", mt, f"{mt}{f}/{s}", _mech_mask(entry, close, high, atr, "atr", 3.0))   # 3: +ATR-trail
            yield ("cross+time", mt, f"{mt}{f}/{s}", _mech_mask(entry, close, high, atr, "time", 10))   # 4: +time-stop
            yield ("cross+tp", mt, f"{mt}{f}/{s}", _mech_mask(entry, close, high, atr, "tp", 0.05))     # 5: +take-profit
            yield ("price>f>s", mt, f"{mt}{f}/{s}", ((close > mas[f]) & (mas[f] > mas[s])).astype(float))  # 6: stack

def decompose_report(assets, cadences, window_days=14, lo=0.02, hi=0.10, max_windows=300):
    from collections import Counter, defaultdict
    out = []
    for sym in assets:
        for cad in cadences:
            df = load_ohlc(sym if sym.endswith("USDT") else sym + "USDT", cad)
            if df is None or len(df) < 80: continue
            dates = pd.to_datetime(df["date"]).to_numpy(); close = df["close"].to_numpy(float)
            high = df["high"].to_numpy(float); atr = df["atr14"].to_numpy(float)
            ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1.0
            configs = list(_structures(close, high, atr))             # (struct, matype, label, mask)
            laps = [(s, m, l, np.roll(p, 1)) for (s, m, l, p) in configs]   # lag 1 bar
            for s, m, l, p in laps: p[0] = 0.0
            win = np.timedelta64(window_days, "D"); wstart = dates[0]; guard = 0
            cap_struct = defaultdict(list); cap_matype = defaultdict(list)
            best_struct_win = Counter(); best_matype_win = Counter(); overall_caps = []; n_eval = 0
            while wstart < dates[-1] and guard < max_windows:
                wend = wstart + win
                idx = np.where((dates >= wstart) & (dates < wend))[0]; wstart = wend; guard += 1
                if idx.size < 5: continue
                a, b = idx[0], idx[-1]
                p_roi, nm = price_oracle_roi(close[a:b + 1], lo, hi)
                if p_roi <= 0.5 or nm == 0: continue
                n_eval += 1
                best_by_struct = defaultdict(lambda: -1e9); best_by_matype = defaultdict(lambda: -1e9)
                overall = -1e9; overall_key = None
                for s, m, l, pl in laps:
                    roi = float(np.prod(1.0 + pl[a:b + 1] * ret[a:b + 1]) - 1.0) * 100
                    cap = roi / p_roi
                    if cap > best_by_struct[s]: best_by_struct[s] = cap
                    if cap > best_by_matype[m]: best_by_matype[m] = cap
                    if cap > overall: overall = cap; overall_key = (s, m)
                for s, v in best_by_struct.items(): cap_struct[s].append(v)
                for m, v in best_by_matype.items(): cap_matype[m].append(v)
                best_struct_win[overall_key[0]] += 1; best_matype_win[overall_key[1]] += 1
                overall_caps.append(overall)
            if not overall_caps: continue
            out.append({"asset": sym, "cad": cad, "n_win": n_eval, "overall_capture": float(np.median(overall_caps)),
                        "by_struct": {s: round(float(np.median(v)), 2) for s, v in cap_struct.items()},
                        "by_matype": {m: round(float(np.median(v)), 2) for m, v in cap_matype.items()},
                        "struct_wins": best_struct_win.most_common(), "matype_wins": best_matype_win.most_common()})
    return out


def _print_decompose(rows):
    structs = ["price>MA", "cross+flip", "cross+atr", "cross+time", "cross+tp", "price>f>s"]
    print(f"## TI-ORACLE DECOMPOSITION -- median capture (best config in class / price-oracle), per 2-10% move window")
    print(f"   {'asset':5} {'cad':4} {'#win':>4} {'OVERALL':>7} | " + " ".join(f"{s:>10}" for s in structs))
    for r in rows:
        bs = r["by_struct"]
        print(f"   {r['asset']:5} {r['cad']:4} {r['n_win']:>4} {r['overall_capture']:>7.2f} | " +
              " ".join(f"{bs.get(s, 0):>10.2f}" for s in structs))
    print(f"\n   -- MA-TYPE median capture (best config of each type) --")
    print(f"   {'asset':5} {'cad':4} | " + " ".join(f"{m:>6}" for m in MA_TYPES))
    for r in rows:
        bm = r["by_matype"]; print(f"   {r['asset']:5} {r['cad']:4} | " + " ".join(f"{bm.get(m, 0):>6.2f}" for m in MA_TYPES))
    print(f"\n   -- THE MULTITUDE: which STRUCTURE wins each window (overall argmax) --")
    for r in rows:
        tot = sum(n for _, n in r["struct_wins"]); ws = ", ".join(f"{s} {100*n//max(tot,1)}%" for s, n in r["struct_wins"])
        print(f"   {r['asset']:5} {r['cad']:4}: {ws}")
    print(f"   -- which MA-TYPE wins each window --")
    for r in rows:
        tot = sum(n for _, n in r["matype_wins"]); wm = ", ".join(f"{m} {100*n//max(tot,1)}%" for m, n in r["matype_wins"])
        print(f"   {r['asset']:5} {r['cad']:4}: {wm}")


def _parse_periods(label):
    """'hma5' -> (5,); 'ema5/20' -> (5,20)."""
    import re
    return tuple(int(x) for x in re.findall(r"\d+", label))

def plot_window(asset, cadence, start, window_days=14, lo=0.02, hi=0.10, outdir=None):
    """Render ONE 2-week slice: price + price-oracle swings (2-10% moves) + the BEST TI-oracle config (MA + long-shade)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sym = asset if asset.endswith("USDT") else asset + "USDT"
    df = load_ohlc(sym, cadence)
    if df is None: print("no data"); return None
    dates = pd.to_datetime(df["date"]).to_numpy(); close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float); atr = df["atr14"].to_numpy(float)
    ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1.0
    wstart = np.datetime64(pd.Timestamp(start)); wend = wstart + np.timedelta64(window_days, "D")
    idx = np.where((dates >= wstart) & (dates < wend))[0]
    if idx.size < 4: print(f"window {start} has only {idx.size} bars"); return None
    a, b = idx[0], idx[-1]
    p_roi, nm = price_oracle_roi(close[a:b + 1], lo, hi)
    # full-swing oracle (uncapped up-legs >=2%) -- the realizable ceiling when a window is one big trend
    _piv = _zigzag_pivots(close[a:b + 1], thr=lo); _c = 1.0
    for _k in range(len(_piv)-1):
        _v0,_v1 = _piv[_k][1], _piv[_k+1][1]
        if _v1>_v0 and (_v1/_v0-1.0)>=lo: _c *= (_v1/_v0)
    full_swing = (_c-1.0)*100
    best = -1e9; bsel = None
    for s, m, l, p in _structures(close, high, atr):
        pl = np.roll(p, 1); pl[0] = 0.0
        roi = float(np.prod(1.0 + pl[a:b + 1] * ret[a:b + 1]) - 1.0) * 100
        if roi > best: best = roi; bsel = (s, m, l, pl)
    struct, matype, label, mask = bsel
    periods = _parse_periods(label)
    xd = pd.to_datetime(dates[a:b + 1])
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(xd, close[a:b + 1], color="black", lw=1.4, label="close")
    piv = _zigzag_pivots(close[a:b + 1], thr=lo)
    px = [a + i for i, _ in piv]; py = [v for _, v in piv]
    ax.plot(pd.to_datetime(dates[px]), py, color="gray", ls="--", lw=0.8, alpha=0.7, zorder=2)
    first=True
    for k in range(len(piv) - 1):
        (i0, v0), (i1, v1) = piv[k], piv[k + 1]
        if v1 > v0 and lo <= (v1 / v0 - 1.0):
            ax.plot([pd.to_datetime(dates[a + i0]), pd.to_datetime(dates[a + i1])], [v0, v1],
                    color="green", lw=2.6, alpha=0.55, zorder=3, label="price-oracle move (>=2%)" if first else None); first=False
    for n in periods:
        ma = _ma(close, matype, n)
        ax.plot(xd, ma[a:b + 1], lw=1.1, alpha=0.9, label=f"{matype.upper()}({n})")
    on = mask[a:b + 1] > 0.5
    y0,y1=ax.get_ylim()
    ax.fill_between(xd, y0, y1, where=on, color="tab:blue", alpha=0.10, step="mid", label="TI long (captured)")
    cap = best / full_swing if full_swing > 1e-9 else 0.0
    ax.set_title(f"{asset} {cadence}  [{pd.Timestamp(start).date()} +{window_days}d]   "
                 f"price-oracle(2-10% moves) {p_roi:.1f}%  |  full-swing {full_swing:.1f}%  |  "
                 f"TI-oracle {best:.1f}% ({struct},{label})  |  capture(vs full) {cap:.0%}", fontsize=10)
    ax.legend(fontsize=8, loc="best"); ax.tick_params(labelsize=8); fig.autofmt_xdate()
    outdir = Path(outdir) if outdir else (ROOT.parent / "plots" / "oracle_anchor")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{asset}_{cadence}_{pd.Timestamp(start).date()}_w{window_days}.png"
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[PLOT] {path}  | price-oracle(2-10%) {p_roi:.1f}%  full-swing {full_swing:.1f}%  TI {best:.1f}% ({struct},{label})  capture(vs full) {cap:.0%}")
    return {"asset": asset, "cad": cadence, "n_bars": int(b-a+1), "price_oracle": round(p_roi,1),
            "full_swing": round(full_swing,1), "ti_oracle": round(best,1), "capture_vs_full": round(cap,3),
            "struct": struct, "config": label, "path": str(path)}


def grid_plot(assets, cadences, start, window_days=14, lo=0.02, hi=0.10):
    """Render asset x cadence plots for ONE fixed window + a summary table (the oracle numbers DIFFER per cell)."""
    rows=[]
    for sym in assets:
        for cad in cadences:
            try: r=plot_window(sym, cad, start, window_days, lo, hi)
            except Exception as e: print(f"  [skip] {sym} {cad}: {e}"); r=None
            if r: rows.append(r)
    print(f"\n## ORACLE GRID -- window {start} +{window_days}d -- price-oracle(2-10%) / full-swing / TI-oracle / capture / best config")
    print(f"   {'asset':6} {'cad':4} {'#bars':>5} {'priceOrc%':>9} {'fullSwing%':>10} {'TIorc%':>7} {'capt':>5}  best config")
    for r in rows:
        print(f"   {r['asset']:6} {r['cad']:4} {r['n_bars']:>5} {r['price_oracle']:>9} {r['full_swing']:>10} "
              f"{r['ti_oracle']:>7} {r['capture_vs_full']:>5.0%}  {r['struct']},{r['config']}")
    return rows


def _dump_json(mode, args, rows):
    """Persist results + a repro block (exact command + git SHA + data range) -> runs/mining/ (tracked)."""
    import json, subprocess, sys as _sys
    try: sha = subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
    except Exception: sha = "unknown"
    cmd = "python " + " ".join(_sys.argv)
    outdir = ROOT.parent / "runs" / "mining"; outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_{args.assets.replace(',','-')}_{args.cadences.replace(',','-')}_w{args.window_days}"
    path = outdir / f"ti_oracle_{tag}.json"
    payload = {"repro": {"command": cmd, "git_sha": sha, "script": "src/strat/oracle_anchor.py",
                         "note": "deterministic (no RNG); re-run the command to regenerate bit-identically"},
               "params": {"mode": mode, "assets": args.assets, "cadences": args.cadences,
                          "window_days": args.window_days, "lo": args.lo, "hi": args.hi},
               "results": rows}
    with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2, default=str)
    print(f"[persisted] {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.oracle_anchor")
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--cadences", default="1d,4h,1h")
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--lo", type=float, default=0.02); ap.add_argument("--hi", type=float, default=0.10)
    ap.add_argument("--decompose", action="store_true", help="decompose the TI oracle across structures x MA-types x exits")
    ap.add_argument("--grid", action="store_true", help="batch: render assets x cadences for one window + summary table")
    ap.add_argument("--plot", action="store_true", help="render a 2-week slice chart (price + oracle moves + best TI config)")
    ap.add_argument("--start", default=None, help="window start date YYYY-MM-DD for --plot")
    ap.add_argument("--json", action="store_true", help="persist results + repro block to runs/mining/ (tracked artifact)")
    a = ap.parse_args(argv)
    if a.grid:
        if not a.start: print('--grid needs --start YYYY-MM-DD'); return 1
        rows=grid_plot(a.assets.split(','), a.cadences.split(','), a.start, a.window_days, a.lo, a.hi)
        if a.json: _dump_json('grid', a, rows)
        return 0
    if a.plot:
        if not a.start: print('--plot needs --start YYYY-MM-DD'); return 1
        plot_window(a.assets.split(',')[0], a.cadences.split(',')[0], a.start, a.window_days, a.lo, a.hi); return 0
    if a.decompose:
        rows = decompose_report(a.assets.split(","), a.cadences.split(","), a.window_days, a.lo, a.hi)
        _print_decompose(rows)
        if a.json: _dump_json("decompose", a, rows)
        return 0
    rows = anchor_report(a.assets.split(","), a.cadences.split(","), a.window_days, a.lo, a.hi)
    _print(rows, a.window_days)
    if a.json: _dump_json("anchor", a, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
