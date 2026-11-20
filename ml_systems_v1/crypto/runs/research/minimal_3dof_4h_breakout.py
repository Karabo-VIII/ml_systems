"""runs/research/minimal_3dof_4h_breakout.py -- the MINIMAL 3-DOF candidate (researcher prescription).

Per the adaptive-MA researcher REDIRECT (experiments/adaptive_ma/OVERSEER_LOG.md check-in 6):
ER as a HARD GATE (not a switcher), 4h cadence, ATR-trail exit (not opposite-cross), breakout-confirm
entry, MINIMAL degrees of freedom. This is the falsifiable test of that prescription.

CANDIDATE (all constants PRE-REGISTERED below, fixed BEFORE looking at any held-out number):
  cadence   : 4h
  MA        : single FIXED config -- EMA fast=8 / slow=21 (the adaptive_ma _BASE[2] 'trend' config)
  ER gate   : Kaufman efficiency ratio over ER_WIN bars, HARD GATE er > ER_GATE (skip when below)
  entry     : breakout-confirm = close > prior-N-bar HIGH  AND  fast>slow  AND  er>ER_GATE  (at close)
  exit      : ATR-trailing stop (mult * ATR) + time-stop (<=7d = 42 4h bars). NO take-profit.

All entry features are STRICTLY PAST-ONLY (.shift(1) / rolling(...).shift(1)); fill = next-bar open
(SetupHarness contract). Score = held-out per-trade expectancy + compound on the UNSEEN split.
Decisive soundness test = cost-matched random-ENTRY firewall (src/strat/firewall.py): a breakout-in-trend
setup must BEAT random entries (same count / durations / cost) on held-out, else it is beta-in-disguise.

RWYB:  python runs/research/minimal_3dof_4h_breakout.py [--quick] [--assets BTCUSDT,ETHUSDT]
No emoji (cp1252). Reuses the kept apparatus (SetupHarness + firewall + battery). Does NOT commit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader            # noqa: E402
from pipeline.universe_loader import UniverseLoader          # noqa: E402
from wealth_bot.harness import WindowSpec, ema_past_only     # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy     # noqa: E402
from strat.firewall import random_entry_null                 # noqa: E402
from strat.battery import evaluate as battery_evaluate       # noqa: E402

# ---- PRE-REGISTERED CONSTANTS (fixed before seeing held-out) --------------------------------------
CADENCE   = "4h"
TAKER     = 0.0024
MA_FAST, MA_SLOW = 8, 21          # single fixed EMA config
ER_WIN    = 20                    # Kaufman ER lookback (bars)  ~ 3.3 days on 4h
ER_GATE   = 0.40                  # HARD GATE: enter only when ER > 0.40 (clean directional move)
BREAK_N   = 20                    # breakout lookback: close > max(prior N-bar HIGH)
ATR_WIN   = 14                    # ATR lookback (bars)
ATR_MULT  = 3.0                   # ATR-trailing stop width
TIME_STOP = 42                    # 7 days = 42 * 4h bars
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]


def _load_ohlc(loader: ChimeraLoader, sym: str) -> pd.DataFrame | None:
    try:
        g = loader.load(sym, cadence=CADENCE)
    except Exception:
        return None
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    return pd.DataFrame({"date": dt,
                         "open": np.asarray(d["open"], float), "high": np.asarray(d["high"], float),
                         "low": np.asarray(d["low"], float), "close": np.asarray(d["close"], float)})


def _kaufman_er(close: pd.Series, win: int) -> pd.Series:
    change = (close - close.shift(win)).abs()
    vol_path = close.diff().abs().rolling(win, min_periods=win // 2).sum()
    return (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0)


def build_entry(df: pd.DataFrame) -> pd.DataFrame:
    """Add the past-only ER / MA / breakout / ATR columns and the boolean entry. Past-only by construction."""
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    high, low = out["high"].astype(float), out["low"].astype(float)

    fast = ema_past_only(close, length=MA_FAST, shift=0)       # past-only EMA (uses <= t)
    slow = ema_past_only(close, length=MA_SLOW, shift=0)
    er = _kaufman_er(close, ER_WIN).shift(1)                   # shift(1): strictly past at decision bar
    prior_n_high = high.rolling(BREAK_N).max().shift(1)        # max of bars [t-N .. t-1] -> strictly prior

    # ATR (Wilder true range, simple rolling mean). NOT shifted here: SetupHarness reads atr[j-1] (prior bar).
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_WIN, min_periods=ATR_WIN // 2).mean()

    # ENTRY (confirmed at close of bar t): breakout AND trend-up AND ER-gate
    out["entry"] = ((close > prior_n_high) & (fast > slow) & (er > ER_GATE)).fillna(False).astype(int)
    out["_er"], out["_fast"], out["_slow"], out["_priorhigh"] = er, fast, slow, prior_n_high
    return out


def run_asset(df: pd.DataFrame):
    d = build_entry(df)
    policy = ExitPolicy(atr_trail_mult=ATR_MULT, atr_col="atr", max_hold_bars=TIME_STOP)
    h = SetupHarness(d, "entry", policy, WIN, cost_rt=TAKER, regime_match_on_entry=True)
    res = h.run()
    return h, res


def _per_trade_exp(trades, w):
    nets = [t["net_pnl"] for t in trades if t["window"] == w]
    return (float(np.mean(nets) * 100) if nets else 0.0), len(nets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 25 assets")
    ap.add_argument("--assets", type=str, default=None, help="comma-list, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--firewall", action="store_true", default=True)
    ap.add_argument("--no-firewall", dest="firewall", action="store_false")
    args = ap.parse_args()

    loader = ChimeraLoader()
    if args.assets:
        syms = [s.strip().upper() for s in args.assets.split(",")]
    else:
        syms = UniverseLoader.load().list("u100")
        if args.quick:
            syms = syms[:25]
    print(f"[3DOF-4h] cadence={CADENCE} taker={TAKER} | MA EMA {MA_FAST}/{MA_SLOW} | ER>{ER_GATE} "
          f"(win {ER_WIN}) | breakout N={BREAK_N} | ATR {ATR_MULT}x(win {ATR_WIN}) | time-stop {TIME_STOP} bars "
          f"(7d) | assets={len(syms)}", flush=True)

    per_asset, pool = {}, {w: [] for w in WINDOWS}
    fw_held, n_loaded = [], 0
    for k, s in enumerate(syms, 1):
        df = _load_ohlc(loader, s)
        if df is None or len(df) < 600:
            continue
        n_loaded += 1
        h, res = run_asset(df)
        rec = {"n_setups": int(df.pipe(build_entry)["entry"].sum())}
        for w in WINDOWS:
            ws = res.window_stats[w]
            exp, n = _per_trade_exp(res.trades, w)
            rec[w] = {"comp": round(ws.compound_pct, 2), "n": n, "wr": round(ws.win_rate, 3),
                      "exp_pct": round(exp, 4), "dd": round(ws.max_dd_pct, 2)}
            pool[w] += [t["net_pnl"] for t in res.trades if t["window"] == w]
        if args.firewall:
            try:
                fw = random_entry_null(h, n_books=200, seed=7, regime_matched=True)   # within setup-ON timing
                fwp = random_entry_null(h, n_books=200, seed=7, regime_matched=False)  # plain: selection edge
                rec["fw_regime"] = {w: {"real": fw["per_window"][w]["real"],
                                        "p95": fw["per_window"][w]["null_p95"],
                                        "beats": fw["per_window"][w]["beats_null"]} for w in WINDOWS}
                rec["fw_plain"] = {w: {"real": fwp["per_window"][w]["real"],
                                       "p95": fwp["per_window"][w]["null_p95"],
                                       "beats": fwp["per_window"][w]["beats_null"]} for w in WINDOWS}
                rec["fw_beats_held"] = bool(fw["beats_held"])
                rec["fw_plain_beats_held"] = bool(fwp["beats_held"])
                fw_held.append((bool(fw["beats_held"]), bool(fwp["beats_held"])))
            except Exception as e:  # noqa: BLE001
                rec["fw_error"] = repr(e)
        per_asset[s] = rec
        if k % 10 == 0:
            print(f"[run] {k}/{len(syms)} ({n_loaded} loaded)", flush=True)

    # ---- aggregate ----
    agg = {}
    for w in WINDOWS:
        comps = np.array([per_asset[s][w]["comp"] for s in per_asset], float)
        nets = np.asarray(pool[w], float)
        agg[w] = {
            "pooled_trade_exp_pct": (round(float(nets.mean() * 100), 4) if nets.size else None),
            "pooled_winrate": (round(float((nets > 0).mean()), 3) if nets.size else None),
            "pooled_n_trades": int(nets.size),
            "mean_comp": round(float(comps.mean()), 2), "median_comp": round(float(np.median(comps)), 2),
            "n_assets_pos_comp": int((comps > 0).sum()), "n_assets": int(comps.size),
        }
    if fw_held:
        agg["firewall_regime_matched"] = {"n_beat_null_held": int(sum(a for a, _ in fw_held)),
                                          "n_assets": len(fw_held)}
        agg["firewall_plain"] = {"n_beat_null_held": int(sum(b for _, b in fw_held)),
                                 "n_assets": len(fw_held)}

    out = {"config": {"cadence": CADENCE, "taker": TAKER, "ma": [MA_FAST, MA_SLOW, "ema"],
                      "er_win": ER_WIN, "er_gate": ER_GATE, "breakout_n": BREAK_N,
                      "atr_win": ATR_WIN, "atr_mult": ATR_MULT, "time_stop_bars": TIME_STOP,
                      "windows": WIN.__dict__, "n_assets_loaded": n_loaded},
           "aggregate": agg, "per_asset": per_asset}

    # ---- report ----
    print("\n" + "=" * 86)
    print(f"MINIMAL 3-DOF 4h BREAKOUT (ER-gate + ATR-trail)  |  {n_loaded} assets  |  taker {TAKER}")
    print("=" * 86)
    print(f"{'window':8} {'pool exp%':>9} {'pool wr':>8} {'pool n':>7} {'mean comp':>10} "
          f"{'med comp':>9} {'pos/assets':>11}")
    for w in WINDOWS:
        a = agg[w]
        print(f"{w:8} {str(a['pooled_trade_exp_pct']):>9} {str(a['pooled_winrate']):>8} "
              f"{a['pooled_n_trades']:>7} {a['mean_comp']:>10} {a['median_comp']:>9} "
              f"{str(a['n_assets_pos_comp'])+'/'+str(a['n_assets']):>11}")
    if fw_held:
        fwh = agg["firewall_regime_matched"]; fwp = agg["firewall_plain"]
        print("-" * 86)
        print(f"FIREWALL regime-matched (within setup-ON timing) beats null on OOS+UNSEEN: "
              f"{fwh['n_beat_null_held']}/{fwh['n_assets']} assets")
        print(f"FIREWALL plain (selection vs random-anywhere)   beats null on OOS+UNSEEN: "
              f"{fwp['n_beat_null_held']}/{fwp['n_assets']} assets")
    print("=" * 86, flush=True)

    outp = ROOT / "runs" / "research" / ("minimal_3dof_4h_result_quick.json" if args.quick
                                         else "minimal_3dof_4h_result.json")
    if args.assets:
        outp = ROOT / "runs" / "research" / "minimal_3dof_4h_result_assets.json"
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outp}", flush=True)
    return out


if __name__ == "__main__":
    main()
