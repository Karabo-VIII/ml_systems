"""PORTFOLIO REPLAY ENGINE -- fit TI strategies, replay them as ONE risk-budgeted portfolio
forward (paper-trade style) -> equity curve + metrics + per-strategy attribution.

User /orc (2026-06-11): "a strat replay engine where we can fit TI(s), then have them replay
a portfolio, like paper trading." The building blocks existed but were not wired into one
replay loop -- this engine wires them:
  - signal generation : TI families (MA cross / Donchian / ROC / RSI / Bollinger / vol-exp)
                        -> a per-(asset, strategy) HOLDING state (long while in-position).
  - sizing            : firm/decision_spine.decide() (optional, --spine) gates+sizes each bet
                        by a trailing forecast; default = inverse-vol risk parity.
  - portfolio         : firm/portfolio.allocate() -> vol-targeted, gross- + per-name-capped,
                        correlation-aware weights across all currently-held bets.
  - replay            : MtM-correct (lagged weights x next-bar returns, turnover cost) -> the
                        portfolio equity curve over ANY window (TRAIN/OOS/UNSEEN/ALL = paper).
DECLARATIVE: a SPEC = {strategies, universe, cadence, risk policy, window}; swap TIs freely.
Honest: strictly causal (all forecasts trailing; weights lagged 1 bar), taker/maker cost,
respects the MtM-no-double-count invariant. UNSEEN is a VALID window (that IS forward paper-
trading on sealed data). No emoji (cp1252).

Run:
  python -m strat.portfolio_replay --universe u10 --window ALL \
      --strategies "ema_50_100,donch20,rsi_30_50" --spine
  python -m strat.portfolio_replay --universe u50 --window UNSEEN --strategies ema_50_100
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

from mining.family_regime_map import sma, ema, atr14, rsi14, _norm_sym  # noqa: E402
from firm.portfolio import allocate, Position, PortfolioLimits          # noqa: E402
from firm.decision_spine import decide, Forecast, RiskLimits            # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                       # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

TAKER_RT, MAKER_RT = 0.0024, 0.0006
ANN = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
# window splits (match the project's WIN spec)
WIN = {"TRAIN": (None, "2024-05-15"), "VAL": ("2024-05-15", "2025-03-15"),
       "OOS": ("2025-03-15", "2025-12-31"), "UNSEEN": ("2025-12-31", "2026-06-01"),
       "ALL": (None, None)}

# ---- the TI strategy library: name -> (family, params). Holding = long while in-position. ----
STRATS = {
    "ema_50_100": ("2MA", dict(type="EMA", fast=50, slow=100)),
    "ema_50_200": ("2MA", dict(type="EMA", fast=50, slow=200)),
    "ema_20_100": ("2MA", dict(type="EMA", fast=20, slow=100)),
    "sma_50_200": ("2MA", dict(type="SMA", fast=50, slow=200)),
    "ema_10_50_100": ("3MA", dict(type="EMA", fast=10, mid=50, slow=100)),   # MA(x,y,z)
    "ema_20_50_200": ("3MA", dict(type="EMA", fast=20, mid=50, slow=200)),
    "sma_10_50_200": ("3MA", dict(type="SMA", fast=10, mid=50, slow=200)),
    "donch20":    ("DONCH", dict(n=20, exit_n=10)),
    "roc20":      ("ROC", dict(n=20)),
    "rsi_30_50":  ("RSI", dict(lo=30, hi=50)),
    "boll20":     ("BOLL", dict(n=20, k=2)),
    "trendgate":  ("TREND", dict(n=100)),
}


def _ma(c, n, kind):
    return sma(c, n) if kind == "SMA" else ema(c, n)


def apply_trail_stop(held, close, trail):
    """RISK-CONTROL overlay: a high-water trailing stop on a per-asset holding series.
    Once long, track the high-water close; if close drops > trail below it, FORCE flat
    (stopped out) and stay out until the underlying signal re-arms (held goes 0 then 1).
    Returns (stopped_held, stop_exit_idx_set). Causal (uses close at bar i only)."""
    import numpy as _np
    out = _np.asarray(held, dtype=_np.int8).copy()
    inpos = False
    hw = 0.0
    stopped = False                     # stopped this signal-episode -> wait for re-arm
    stop_idx = set()
    for i in range(len(held)):
        if not held[i]:                 # signal off -> reset episode
            inpos, stopped = False, False
            out[i] = 0
            continue
        if stopped:                     # signal still on but we were stopped this episode
            out[i] = 0
            continue
        if not inpos:                   # fresh entry
            inpos, hw = True, float(close[i])
        hw = max(hw, float(close[i]))
        if float(close[i]) < hw * (1.0 - trail):
            out[i] = 0                  # stop hit -> flat from here this episode
            inpos, stopped = False, True
            stop_idx.add(i)
        else:
            out[i] = 1
    return out, stop_idx


def holding_state(name, o, h, l, c):
    """Binary array: 1 while the strategy holds LONG (causal: state at close of bar t)."""
    fam, p = STRATS[name]
    n = len(c)
    hold = np.zeros(n, dtype=np.int8)
    if fam == "2MA":
        a = _ma(c, p["fast"], p["type"]) > _ma(c, p["slow"], p["type"])
        return a.astype(np.int8)                       # long while fast>slow
    if fam == "3MA":                                   # MA(x,y,z): long while fast>mid>slow
        f, m, s = _ma(c, p["fast"], p["type"]), _ma(c, p["mid"], p["type"]), _ma(c, p["slow"], p["type"])
        return ((f > m) & (m > s)).astype(np.int8)
    if fam == "TREND":
        s = sma(c, p["n"]); return (np.isfinite(s) & (c > s)).astype(np.int8)
    if fam == "DONCH":
        hh = np.full(n, np.nan); ll = np.full(n, np.nan)
        for i in range(p["n"], n): hh[i] = np.max(h[i - p["n"]:i])
        for i in range(p["exit_n"], n): ll[i] = np.min(l[i - p["exit_n"]:i])
        inpos = False
        for i in range(n):
            if not inpos and np.isfinite(hh[i]) and c[i] > hh[i]: inpos = True
            elif inpos and np.isfinite(ll[i]) and c[i] < ll[i]: inpos = False
            hold[i] = 1 if inpos else 0
        return hold
    if fam == "ROC":
        roc = np.full(n, np.nan); roc[p["n"]:] = c[p["n"]:] / c[:-p["n"]] - 1
        return (roc > 0).astype(np.int8)
    if fam == "RSI":
        r = rsi14(c); inpos = False
        for i in range(n):
            if not inpos and r[i] < p["lo"]: inpos = True
            elif inpos and r[i] > p["hi"]: inpos = False
            hold[i] = 1 if inpos else 0
        return hold
    if fam == "BOLL":
        m = sma(c, p["n"]); sd = np.full(n, np.nan)
        for i in range(p["n"] - 1, n): sd[i] = np.std(c[i - p["n"] + 1:i + 1])
        inpos = False
        for i in range(n):
            if not inpos and np.isfinite(sd[i]) and c[i] < m[i] - p["k"] * sd[i]: inpos = True
            elif inpos and np.isfinite(m[i]) and c[i] >= m[i]: inpos = False
            hold[i] = 1 if inpos else 0
        return hold
    raise ValueError(name)


def _mask(ms, window):
    lo, hi = WIN[window]
    m = np.ones(len(ms), dtype=bool)
    if lo: m &= ms >= int(dt.datetime.fromisoformat(lo).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    if hi: m &= ms < int(dt.datetime.fromisoformat(hi).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    return m


def run(universe, cadence, strat_names, window, cost_rt, use_spine, vol_target, max_per_name,
        trail_stop=0.0):
    import pandas as pd
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    if "assets" in spec:
        syms = [a["symbol"] for a in spec["assets"]]
    else:
        u50 = yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u50.yaml"))
        syms = [a["symbol"] for a in u50["assets"]] + [a["symbol"] for a in spec.get("extra_assets", [])]
        syms = [s for s in dict.fromkeys(syms) if s not in set(spec.get("excluded_assets") or [])]

    # DATE-ALIGNED panels: floor each ts to the cadence boundary + dedup so ALL assets share
    # ONE index (the union-grid drift bug otherwise explodes the index ~Nx).
    freq = {"1d": "D", "4h": "4h", "1h": "h", "30m": "30min", "15m": "15min"}.get(cadence, "D")
    closes, rets, vols, mus = {}, {}, {}, {}
    holds = {nm: {} for nm in strat_names}
    for sym in syms:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence, features=["open", "high", "low", "close"])
        except Exception:
            continue
        dates = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor(freq)
        o = df["open"].to_numpy().astype(float); h = df["high"].to_numpy().astype(float)
        l = df["low"].to_numpy().astype(float); c = df["close"].to_numpy().astype(float)
        hser = {nm: holding_state(nm, o, h, l, c).astype(float) for nm in strat_names}
        sub = pd.DataFrame({"c": c, **{f"h_{nm}": hser[nm] for nm in strat_names}}, index=dates)
        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
        sc = sub["c"]
        closes[sym] = sc
        rets[sym] = sc.pct_change()
        vols[sym] = sc.pct_change().rolling(30, min_periods=15).std()
        mus[sym] = sc.pct_change().rolling(30, min_periods=15).mean()
        for nm in strat_names:
            holds[nm][sym] = sub[f"h_{nm}"]
    close_panel = pd.DataFrame(closes).sort_index()
    # HARDEN(audit): fail loud if the floor/dedup alignment regresses (a dropped floor
    # explodes the union index ~Nx and silently produces a plausible-looking -99% curve).
    _max_rows = max((len(s) for s in closes.values()), default=0)
    assert len(close_panel) <= 1.5 * _max_rows, (
        f"date-alignment regression: panel {len(close_panel)} rows >> max per-asset "
        f"{_max_rows} (floor/dedup broke -> equity would be wrong)")
    ret_panel = pd.DataFrame(rets).reindex(close_panel.index)
    vol_panel = pd.DataFrame(vols).reindex(close_panel.index)
    mu_panel = pd.DataFrame(mus).reindex(close_panel.index)
    hold_panels = {nm: pd.DataFrame(holds[nm]).reindex(close_panel.index).fillna(0.0) for nm in strat_names}
    held_any = (sum(hold_panels.values()) > 0).astype(int)   # dates x assets (any strategy long)
    n_stops = 0
    if trail_stop and trail_stop > 0:             # RISK CONTROL: trailing-stop overlay per asset
        for a in held_any.columns:
            stopped, sidx = apply_trail_stop(held_any[a].to_numpy(), close_panel[a].to_numpy(), trail_stop)
            held_any[a] = stopped
            n_stops += len(sidx)
    held_any = held_any > 0
    assets = list(close_panel.columns)
    dates = close_panel.index

    plims = PortfolioLimits(vol_target=vol_target, max_gross=1.0, max_per_name=max_per_name, long_only=True)
    rlims = RiskLimits(base_kelly_fraction=0.25, max_fraction=max_per_name, confidence_floor=0.52)

    # per-DATE portfolio weights via firm/portfolio.allocate (+ optional firm/decision_spine gate)
    import numpy as _np
    W = pd.DataFrame(0.0, index=dates, columns=assets)
    for dti, date in enumerate(dates):
        if dti < 31:
            continue
        positions = []
        ha = held_any.loc[date]; vr = vol_panel.loc[date]
        for a in assets:
            v = vr.get(a)
            if not ha.get(a) or v is None or not _np.isfinite(v) or v <= 0:
                continue
            raw = 1.0 / v
            if use_spine:
                mu = mu_panel.loc[date, a]
                if not _np.isfinite(mu):
                    continue
                d = decide(Forecast(mean_net_return=float(mu), std=float(v)),
                           round_trip_cost=cost_rt, bankroll=1.0, limits=rlims, regime_posterior=1.0)
                if d.action != "BET":
                    continue
                raw = d.fraction
            positions.append(Position(asset=a, raw_fraction=float(raw), vol=float(v)))
        if positions:
            for a, wv in allocate(positions, corr=None, limits=plims).weights.items():
                W.at[date, a] = wv

    # MtM replay (vectorized, lagged weights x next-bar returns - turnover cost) over the window
    Wl = W.shift(1).fillna(0.0)
    gross_ret = (Wl * ret_panel.fillna(0.0)).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net_all = gross_ret - turnover * (cost_rt / 2)
    wmask = _mask(np.array([int(t.value // 10**6) for t in dates]), window)
    net = net_all[wmask]
    if len(net) < 5:
        return {"error": f"window {window} too short ({len(net)} bars)"}
    eqarr = (1 + net).cumprod().to_numpy()
    d = net.to_numpy()
    weights_t = [W.loc[dt].to_dict() for dt in dates]
    turn_series = turnover[wmask].to_numpy()
    per_strat_hold_bars = {nm: int(hold_panels[nm].reindex(dates)[wmask].sum().sum()) for nm in strat_names}
    dd = float(((eqarr - np.maximum.accumulate(eqarr)) / np.maximum.accumulate(eqarr)).min() * 100)
    nyr = len(d) / ANN.get(cadence, 365)
    annr = float((eqarr[-1] ** (1 / nyr) - 1) * 100) if eqarr[-1] > 0 else -100.0
    sharpe = float(d.mean() / (d.std() + 1e-12) * np.sqrt(ANN.get(cadence, 365)))
    # rolling 3d ROI (the charter soft-benchmark)
    h3 = ANN.get(cadence, 365) and {"1d": 3, "4h": 18, "1h": 72, "30m": 144, "15m": 288}.get(cadence, 3)
    roll3 = np.array([eqarr[i] / eqarr[i - h3] - 1 for i in range(h3, len(eqarr))]) if len(eqarr) > h3 else np.array([])
    return {
        "window": window, "n_bars": len(d), "n_assets": len(assets),
        "final_equity": float(eqarr[-1]), "final_pct": float((eqarr[-1] - 1) * 100),
        "ann_pct": round(annr, 1), "maxdd_pct": round(dd, 1), "sharpe": round(sharpe, 2),
        "avg_gross": round(float(np.mean([sum(abs(v) for v in w.values()) for w in weights_t if w])), 3),
        "avg_daily_turnover": round(float(np.mean(turn_series)), 4),
        "trail_stop": trail_stop, "n_stop_exits": n_stops,
        "win_bar_rate": round(float((d > 0).mean()), 3),
        "roll3d_pos_rate": round(float((roll3 > 0).mean()), 3) if len(roll3) else None,
        "roll3d_median_pct": round(float(np.median(roll3) * 100), 3) if len(roll3) else None,
        "per_strat_hold_bars": per_strat_hold_bars,
        "equity_curve_tail": [round(x, 4) for x in eqarr[-10:].tolist()],
        # full series for validation/plotting (not persisted by the CLI; programmatic use).
        # FULL (unmasked) W + ret + mask so an independent MtM recon can lag correctly across
        # a window boundary (the masked weights would lose the pre-window lagged weight).
        "_dates": [str(t) for t in net.index],
        "_equity": eqarr.tolist(),
        "_net": d.tolist(),                                    # windowed net (equity basis)
        "_turnover": turn_series.tolist(),                     # windowed turnover
        "_W_full": W, "_ret_full": ret_panel.fillna(0.0),      # FULL weights + returns
        "_wmask": wmask.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(prog="python -m strat.portfolio_replay")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--strategies", default="ema_50_100,donch20,rsi_30_50")
    ap.add_argument("--window", default="ALL")
    ap.add_argument("--start", default=None, help="custom window start YYYY-MM-DD (overrides --window)")
    ap.add_argument("--end", default=None, help="custom window end YYYY-MM-DD (exclusive)")
    ap.add_argument("--maker", action="store_true")
    ap.add_argument("--spine", action="store_true", help="gate+size bets via firm/decision_spine (default: inverse-vol)")
    ap.add_argument("--vol-target", type=float, default=0.02)
    ap.add_argument("--max-per-name", type=float, default=0.15)
    ap.add_argument("--trail-stop", type=float, default=0.0, help="high-water trailing-stop fraction (e.g. 0.05 = 5%); 0 = signal-flip exit only")
    a = ap.parse_args()
    strat_names = [s.strip() for s in a.strategies.split(",") if s.strip() in STRATS]
    if not strat_names:
        print(f"no valid strategies; choose from: {list(STRATS)}"); return 2
    if a.start or a.end:                               # custom window overrides --window
        WIN["CUSTOM"] = (a.start, a.end); a.window = "CUSTOM"
    if a.window not in WIN:
        print(f"unknown window {a.window}; choose from {list(WIN)} or pass --start/--end"); return 2
    cost = MAKER_RT if a.maker else TAKER_RT
    r = run(a.universe, a.cadence, strat_names, a.window, cost, a.spine, a.vol_target, a.max_per_name,
            trail_stop=a.trail_stop)
    print(f"\n## PORTFOLIO REPLAY -- {a.universe} {a.cadence} -- window={a.window} -- "
          f"{'maker' if a.maker else 'taker'} -- sizing={'firm-spine' if a.spine else 'inverse-vol'}")
    print(f"   strategies: {strat_names}")
    if "error" in r:
        print(f"   {r['error']}"); return 0
    print(f"   final equity $1 -> ${r['final_equity']:.3f} ({r['final_pct']:+.1f}%) over {r['n_bars']} bars, {r['n_assets']} assets")
    print(f"   ann {r['ann_pct']:+.1f}% | maxDD {r['maxdd_pct']:.1f}% | Sharpe {r['sharpe']} | "
          f"avg gross {r['avg_gross']} | turnover {r['avg_daily_turnover']}")
    print(f"   win-bar {r['win_bar_rate']*100:.0f}% | 3d-ROI positive {r['roll3d_pos_rate']} "
          f"median {r['roll3d_median_pct']}%")
    print(f"   attribution (hold-bars/strat): {r['per_strat_hold_bars']}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"portfolio_replay_{a.universe}_{a.window}_{stamp}.json"
    r_persist = {k: v for k, v in r.items() if not k.startswith("_")}   # drop heavy/non-serializable series
    r = r_persist
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha},
               "spec": {"strategies": strat_names, "universe": a.universe, "cadence": a.cadence,
                        "window": a.window, "sizing": "firm-spine" if a.spine else "inverse-vol",
                        "vol_target": a.vol_target, "max_per_name": a.max_per_name, "cost_rt": cost},
               "result": r}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"   [persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
