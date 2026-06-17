"""TI HARNESS -- the CONFIGURED technical-indicator lens of the decomposition harness (rolling-window reader).

WHY this is a SEPARATE lens (the load-bearing design point):
  decompose/narrate read chimera features that ALREADY EXIST (pre-computed, fixed); econometric_signature computes
  CANONICAL estimators (no choice). A technical indicator is DIFFERENT -- it does not exist until you CONFIGURE it:
  you must choose the FAMILY (rsi/ma/macd/bollinger/adx/atr) AND its PARAMETERS (period, fast/slow/signal, std,
  thresholds) before any number exists. So a TI read is always RELATIVE TO A CONFIG -- the same window gives a
  different read under RSI-7 vs RSI-21. This harness makes the config FIRST-CLASS: it is an explicit input, echoed in
  every output, so a read is never config-ambiguous.

DESCRIPTIVE only (reads what the configured TIs SAY over the window) -- NOT signal-mining, NOT a strategy.

Backend: pandas_ta (same library the oracle INDICATOR_REGISTRY uses, src/oracle/indicators_ta.py) -- the canonical
indicator math is shared, this is just a window READER rather than the oracle's capture-analyzer.

Rolling-window + WARMUP: a 7-day window at 4h is ~42 bars, but an MA-50 needs 50 bars of history. The harness loads
max(period)*3 bars BEFORE `start` to warm the indicators, computes over [start-warmup, end], and reports STATE within
[start, end] (crosses are only counted inside the window).

Run:
  python -m mining.ti_harness --asset BTC --cadence 4h --start 2025-01-01 --end 2025-01-08
  python -m mining.ti_harness --asset SOL --cadence 4h --start 2024-07-09 --end 2024-07-16 --config fast --json
  python -m mining.ti_harness --selftest
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# pandas_ta 0.3.14b0 references the removed numpy.NaN on numpy>=2; shim BEFORE import (same fix as indicators_ta.py).
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]
import pandas as pd          # noqa: E402
import pandas_ta as ta       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "30m": 48, "15m": 96, "dollar": 6}

# --------------------------------------------------------------------------- the CONFIG (first-class)
# A config is a list of indicator specs: {"family", "params", "thresholds"}. PRESETS are named configs; --config
# takes a preset NAME or a path to a JSON file with the same shape. The DEFAULT is the canonical "TA dashboard".
PRESETS = {
    "default": [
        {"family": "rsi", "params": {"length": 14}, "thresholds": {"oversold": 30, "overbought": 70}},
        {"family": "ma", "params": {"kind": "ema", "fast": 20, "slow": 50}, "thresholds": {}},
        {"family": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}, "thresholds": {}},
        {"family": "bollinger", "params": {"length": 20, "std": 2.0}, "thresholds": {}},
        {"family": "adx", "params": {"length": 14}, "thresholds": {"trend": 25}},
        {"family": "atr", "params": {"length": 14}, "thresholds": {}},
    ],
    "fast": [
        {"family": "rsi", "params": {"length": 7}, "thresholds": {"oversold": 25, "overbought": 75}},
        {"family": "ma", "params": {"kind": "ema", "fast": 9, "slow": 21}, "thresholds": {}},
        {"family": "macd", "params": {"fast": 6, "slow": 13, "signal": 5}, "thresholds": {}},
        {"family": "bollinger", "params": {"length": 10, "std": 2.0}, "thresholds": {}},
        {"family": "adx", "params": {"length": 7}, "thresholds": {"trend": 25}},
    ],
    "slow": [
        {"family": "rsi", "params": {"length": 21}, "thresholds": {"oversold": 35, "overbought": 65}},
        {"family": "ma", "params": {"kind": "sma", "fast": 50, "slow": 200}, "thresholds": {}},
        {"family": "macd", "params": {"fast": 19, "slow": 39, "signal": 9}, "thresholds": {}},
        {"family": "bollinger", "params": {"length": 50, "std": 2.5}, "thresholds": {}},
        {"family": "adx", "params": {"length": 21}, "thresholds": {"trend": 25}},
    ],
}


def _max_period(config: list) -> int:
    m = 1
    for spec in config:
        p = spec.get("params", {})
        m = max(m, p.get("length", 1), p.get("slow", 1), p.get("fast", 1))
    return int(m)


def _crosses(a: np.ndarray, b: np.ndarray, win_mask: np.ndarray):
    """Indices where a crosses ABOVE b (up) and BELOW b (down), counted only inside win_mask."""
    d = np.sign(a - b)
    up, dn = [], []
    for i in range(1, len(d)):
        if not win_mask[i] or np.isnan(d[i]) or np.isnan(d[i - 1]):
            continue
        if d[i - 1] <= 0 and d[i] > 0:
            up.append(i)
        elif d[i - 1] >= 0 and d[i] < 0:
            dn.append(i)
    return up, dn


# --------------------------------------------------------------------------- per-indicator readers
def _read_rsi(close, dates, win, spec):
    length = spec["params"]["length"]; th = spec["thresholds"]
    rsi = ta.rsi(pd.Series(close), length=length)
    rsi = rsi.to_numpy() if rsi is not None else np.full(len(close), np.nan)
    wv = rsi[win]
    cur = float(wv[-1]) if len(wv) and np.isfinite(wv[-1]) else None
    ob, os_ = th["overbought"], th["oversold"]
    zone = "n/a" if cur is None else ("OVERBOUGHT" if cur >= ob else ("OVERSOLD" if cur <= os_ else "neutral"))
    valid = wv[np.isfinite(wv)]
    pct_ob = round(float(np.mean(valid >= ob)), 3) if len(valid) else None
    pct_os = round(float(np.mean(valid <= os_)), 3) if len(valid) else None
    # crosses of the thresholds inside the window
    cob, _ = _crosses(rsi, np.full(len(rsi), ob), win)
    _, cos = _crosses(rsi, np.full(len(rsi), os_), win)
    return {"name": f"RSI-{length}", "value": round(cur, 1) if cur is not None else None, "zone": zone,
            "thresholds": [os_, ob], "frac_overbought": pct_ob, "frac_oversold": pct_os,
            "crossed_into_overbought": len(cob), "crossed_into_oversold": len(cos),
            "summary": f"RSI-{length} {round(cur,1) if cur is not None else 'n/a'} ({zone})"}


def _read_ma(close, dates, win, spec):
    kind = spec["params"].get("kind", "ema"); fast = spec["params"]["fast"]; slow = spec["params"]["slow"]
    fn = ta.ema if kind == "ema" else ta.sma
    f = fn(pd.Series(close), length=fast); s = fn(pd.Series(close), length=slow)
    f = f.to_numpy() if f is not None else np.full(len(close), np.nan)
    s = s.to_numpy() if s is not None else np.full(len(close), np.nan)
    up, dn = _crosses(f, s, win)  # golden (up) / death (down) crosses in-window
    wf, ws = f[win], s[win]
    state, dist = "n/a", None
    if len(wf) and np.isfinite(wf[-1]) and np.isfinite(ws[-1]) and ws[-1] != 0:
        state = "ABOVE (golden)" if wf[-1] > ws[-1] else "BELOW (death)"
        dist = round(100.0 * (wf[-1] / ws[-1] - 1.0), 2)
    gd = [dates[i][:10] for i in up]; dd = [dates[i][:10] for i in dn]
    return {"name": f"{kind.upper()}-{fast}/{slow}", "fast_vs_slow": state, "fast_minus_slow_pct": dist,
            "golden_crosses": gd, "death_crosses": dd,
            "summary": f"{kind.upper()}{fast}/{slow} {state}"
                       f"{(' +'+str(len(gd))+'gold') if gd else ''}{(' +'+str(len(dd))+'death') if dd else ''}"}


def _read_macd(close, dates, win, spec):
    f, sl, sig = spec["params"]["fast"], spec["params"]["slow"], spec["params"]["signal"]
    df = ta.macd(pd.Series(close), fast=f, slow=sl, signal=sig)
    if df is None or df.shape[1] < 3:
        return {"name": f"MACD-{f}/{sl}/{sig}", "summary": "MACD n/a"}
    macd = df.iloc[:, 0].to_numpy(); hist = df.iloc[:, 1].to_numpy(); signl = df.iloc[:, 2].to_numpy()
    up, dn = _crosses(macd, signl, win)
    wh = hist[win]
    hcur = float(wh[-1]) if len(wh) and np.isfinite(wh[-1]) else None
    histsign = "n/a" if hcur is None else ("positive (bullish)" if hcur > 0 else "negative (bearish)")
    return {"name": f"MACD-{f}/{sl}/{sig}", "histogram": round(hcur, 5) if hcur is not None else None,
            "histogram_sign": histsign, "bull_crosses": [dates[i][:10] for i in up],
            "bear_crosses": [dates[i][:10] for i in dn],
            "summary": f"MACD hist {histsign}"
                       f"{(' +'+str(len(up))+'bull') if up else ''}{(' +'+str(len(dn))+'bear') if dn else ''}"}


def _read_bollinger(close, dates, win, spec):
    length = spec["params"]["length"]; std = spec["params"]["std"]
    bb = ta.bbands(pd.Series(close), length=length, std=std)
    if bb is None or bb.shape[1] < 3:
        return {"name": f"BB-{length}/{std}", "summary": "BB n/a"}
    lower = bb.iloc[:, 0].to_numpy(); mid = bb.iloc[:, 1].to_numpy(); upper = bb.iloc[:, 2].to_numpy()
    rng = upper - lower
    pctb = np.where(rng > 0, (close - lower) / rng, np.nan)   # %B: 0=lower band, 1=upper band
    wpb = pctb[win]
    cur = float(wpb[-1]) if len(wpb) and np.isfinite(wpb[-1]) else None
    valid = wpb[np.isfinite(wpb)]
    # band-walk: fraction of the window outside the bands (riding/breaking a band)
    rode_upper = round(float(np.mean(valid > 1.0)), 3) if len(valid) else None
    rode_lower = round(float(np.mean(valid < 0.0)), 3) if len(valid) else None
    # bandwidth (squeeze vs expansion): mean (upper-lower)/mid over the window
    wbw = (rng / np.where(mid != 0, mid, np.nan))[win]
    bw = round(float(np.nanmean(wbw)), 4) if np.isfinite(np.nanmean(wbw)) else None
    pos = "n/a" if cur is None else ("ABOVE upper" if cur > 1 else ("BELOW lower" if cur < 0
                                     else ("upper-half" if cur > 0.5 else "lower-half")))
    return {"name": f"BB-{length}/{std}sd", "pct_b": round(cur, 3) if cur is not None else None, "position": pos,
            "frac_rode_upper": rode_upper, "frac_rode_lower": rode_lower, "mean_bandwidth": bw,
            "summary": f"BB %B {round(cur,2) if cur is not None else 'n/a'} ({pos})"}


def _read_adx(close, high, low, dates, win, spec):
    length = spec["params"]["length"]; trend = spec["thresholds"]["trend"]
    adf = ta.adx(pd.Series(high), pd.Series(low), pd.Series(close), length=length)
    if adf is None or adf.shape[1] < 3:
        return {"name": f"ADX-{length}", "summary": "ADX n/a"}
    adx = adf.iloc[:, 0].to_numpy(); dmp = adf.iloc[:, 1].to_numpy(); dmn = adf.iloc[:, 2].to_numpy()
    wa = adx[win]
    cur = float(wa[-1]) if len(wa) and np.isfinite(wa[-1]) else None
    di_dir = "n/a"
    if len(adx) and np.isfinite(dmp[-1]) and np.isfinite(dmn[-1]):
        di_dir = "+DI>-DI (up)" if dmp[-1] > dmn[-1] else "-DI>+DI (down)"
    verdict = "n/a" if cur is None else ("TRENDING" if cur >= trend else ("ranging" if cur < 20 else "weak/building"))
    return {"name": f"ADX-{length}", "value": round(cur, 1) if cur is not None else None, "verdict": verdict,
            "di_direction": di_dir, "trend_threshold": trend,
            "summary": f"ADX-{length} {round(cur,1) if cur is not None else 'n/a'} ({verdict}, {di_dir})"}


def _read_atr(close, high, low, dates, win, spec):
    length = spec["params"]["length"]
    atr = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=length)
    atr = atr.to_numpy() if atr is not None else np.full(len(close), np.nan)
    wa = atr[win]; wc = close[win]
    cur = float(wa[-1]) if len(wa) and np.isfinite(wa[-1]) else None
    pct = round(100.0 * cur / wc[-1], 2) if (cur is not None and len(wc) and wc[-1]) else None
    return {"name": f"ATR-{length}", "atr": round(cur, 6) if cur is not None else None, "atr_pct_of_price": pct,
            "summary": f"ATR-{length} {pct}% of price"}


_READERS = {"rsi": _read_rsi, "ma": _read_ma, "macd": _read_macd, "bollinger": _read_bollinger}


def compute_ti_window(asset, cadence, start, end, config) -> dict:
    sym = asset.upper(); sym = sym if sym.endswith("USDT") else sym + "USDT"
    df = ChimeraLoader().load(sym, cadence=cadence).sort("date")
    dates = df["date"].cast(str).to_numpy()
    close = df["close"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float) if "high" in df.columns else close
    low = df["low"].to_numpy().astype(float) if "low" in df.columns else close
    n = len(close)
    # window mask + warmup: compute on [start-warmup, end], report state within [start, end]
    in_win = np.ones(n, bool)
    if start:
        in_win &= (dates >= start)
    if end:
        in_win &= (dates <= end)
    if not in_win.any():
        return {"error": f"no bars in [{start},{end}] for {sym} {cadence}"}
    warm = _max_period(config) * 3
    first = int(np.argmax(in_win))                      # first in-window index
    lo = max(0, first - warm)
    sl = slice(lo, n)
    c, h, l, d = close[sl], high[sl], low[sl], dates[sl]
    win = in_win[sl]
    out = {"asset": sym, "cadence": cadence, "start": start, "end": end,
           "n_window_bars": int(win.sum()), "warmup_bars_loaded": int(first - lo),
           "config_preset_or_path": config_label(config), "config": config, "indicators": []}
    move = round(100.0 * (c[win][-1] / c[win][0] - 1.0), 2) if win.sum() >= 2 else None
    out["window_price_move_pct"] = move
    for spec in config:
        fam = spec["family"]
        if fam in _READERS:
            r = _READERS[fam](c, d, win, spec)
        elif fam == "adx":
            r = _read_adx(c, h, l, d, win, spec)
        elif fam == "atr":
            r = _read_atr(c, h, l, d, win, spec)
        else:
            r = {"name": fam, "summary": f"unknown family '{fam}'"}
        r["family"] = fam
        out["indicators"].append(r)
    out["ti_consensus"] = "  |  ".join(i.get("summary", "") for i in out["indicators"])
    return out


_CONFIG_LABEL = {"label": "default"}


def config_label(_config) -> str:
    return _CONFIG_LABEL["label"]


def render_text(o: dict) -> str:
    if "error" in o:
        return str(o["error"])
    L = [f"## TI HARNESS -- {o['asset']} -- {o['cadence']} -- {o['start']} -> {o['end']}  "
         f"({o['n_window_bars']} bars; warmup {o['warmup_bars_loaded']})",
         f"config: {o['config_preset_or_path']}   window price move {o['window_price_move_pct']}%",
         "(a TI read is RELATIVE TO THIS CONFIG -- a different config gives a different read)", ""]
    for i in o["indicators"]:
        L.append(f"[{i['family']}] {i.get('name','')}")
        for k, v in i.items():
            if k in ("family", "name", "summary"):
                continue
            L.append(f"    {k}: {v}")
    L.append("")
    L.append(f"TI CONSENSUS: {o['ti_consensus']}")
    return "\n".join(L)


def _load_config(arg: str):
    if arg in PRESETS:
        _CONFIG_LABEL["label"] = f"preset:{arg}"
        return PRESETS[arg]
    p = Path(arg)
    if p.exists():
        _CONFIG_LABEL["label"] = f"file:{p.name}"
        return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit(f"--config '{arg}' is neither a known preset {list(PRESETS)} nor a JSON file path")


def selftest() -> int:
    """Data-free gate: synthetic uptrend -> EMA golden + RSI not-oversold + ADX trending; downtrend -> the inverse."""
    import numpy as _np
    fails = []
    n = 400
    up = _np.cumsum(_np.full(n, 0.01) + _np.random.RandomState(1).randn(n) * 0.002) + 4.6
    up = _np.exp(up)  # smooth uptrend in price
    dn = up[::-1].copy()
    cfg = PRESETS["default"]
    win = _np.zeros(n, bool); win[-60:] = True
    d = [f"2025-01-{i:02d}" for i in range(1, n + 1)]
    ma_up = _read_ma(up, d, win, cfg[1]); ma_dn = _read_ma(dn, d, win, cfg[1])
    print(f"  [{'PASS' if 'ABOVE' in ma_up['fast_vs_slow'] else 'FAIL'}] uptrend -> EMA golden: {ma_up['fast_vs_slow']}")
    if "ABOVE" not in ma_up["fast_vs_slow"]:
        fails.append("ma_up")
    print(f"  [{'PASS' if 'BELOW' in ma_dn['fast_vs_slow'] else 'FAIL'}] downtrend -> EMA death: {ma_dn['fast_vs_slow']}")
    if "BELOW" not in ma_dn["fast_vs_slow"]:
        fails.append("ma_dn")
    adx_up = _read_adx(up, up * 1.001, up * 0.999, d, win, cfg[4])
    print(f"  [{'PASS' if adx_up['verdict']=='TRENDING' else 'FAIL'}] smooth uptrend -> ADX TRENDING: {adx_up['value']}")
    if adx_up["verdict"] != "TRENDING":
        fails.append("adx")
    rsi_up = _read_rsi(up, d, win, cfg[0])
    print(f"  [{'PASS' if rsi_up['zone']!='OVERSOLD' else 'FAIL'}] uptrend -> RSI not oversold: {rsi_up['value']}")
    if rsi_up["zone"] == "OVERSOLD":
        fails.append("rsi")
    print(f"\nSELFTEST: {'ALL PASS' if not fails else 'FAILED: '+','.join(fails)}")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mining.ti_harness",
                                 description="Configured technical-indicator lens over a rolling window.")
    ap.add_argument("--asset")
    ap.add_argument("--cadence", default="4h")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--config", default="default", help="preset name (default|fast|slow) or path to a config JSON")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.asset:
        ap.error("--asset is required (or use --selftest)")
    config = _load_config(a.config)
    o = compute_ti_window(a.asset, a.cadence, a.start, a.end, config)
    tag = f"{a.asset.upper().replace('USDT','')}_{a.cadence}_{a.start}_{a.end}"
    (OUT / f"ti_{tag}.json").write_text(json.dumps(o, indent=2, default=str), encoding="utf-8")
    print(json.dumps(o, indent=2, default=str) if a.json else render_text(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
