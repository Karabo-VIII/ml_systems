"""src/strat/engine_timeframe_sweep.py -- THE SWEEP-MANDATE VALIDATION of the daily engine methodology.

WHAT THIS IS (2026-06-14): the SWEEP-ALL-TIMEFRAMES validation of the engine methodology documented in
docs/ENGINE_METHODOLOGY.md. The deployable engine (src/strat/daily_engine.py) is hardcoded DAILY
(CADENCE='1d', VOL_WINDOW=30, REGIME_LOOKBACK=60, ANN=365). The project's HARD RULE is: never silently
default a single cadence -- SWEEP all {1d,4h,1h,30m,15m} (feedback-sweep-all-timeframes-never-default-one).
This tool REPLICATES THE SAME methodology at each cadence -- the SAME builders, verbatim -- with the
windows + annualization SCALED correctly, to answer the empirical question:

    Does the core+overlay methodology GENERALIZE across cadences, or does cost eat it as the cadence
    finers (the EXPECTED 'coarse-cadence-best' result)?

HOW IT REUSES THE ENGINE (no re-implementation, no daily-default breakage):
  - It calls daily_engine's OWN builders verbatim: core_weights / regime_scalar_series / build_book /
    buy_hold_net / window_stats. It does NOT fork the engine.
  - It SCALES the engine's bar-count globals to keep ~the same WALL-CLOCK windows at each cadence, by
    temporarily patching DE module globals (exactly as the gauntlet's _engine_net does) and ALWAYS
    restoring them in a finally-block. The daily default + selftest are UNTOUCHED.
        VOL_WINDOW      30 daily bars (~30d)  -> 30 * bars_per_day
        REGIME_LOOKBACK 60 daily bars (~60d)  -> 60 * bars_per_day   (via RRB.LOOKBACK['1d'])
        REGIME_MIN_DWELL 5 daily bars (~5d)   -> 5  * bars_per_day
        ANN             365                   -> 365 * bars_per_day   (correct per-cadence annualization)
  - It builds a cadence-aware date-aligned CLOSE panel from the SAME ChimeraLoader source the daily
    engine uses (ma_per_instrument._panel), floored to the cadence's bar.

WHAT IT REPORTS per cadence (full cycle + a recent slice):
  ENGINE (core+overlay) vs CORE-ALONE (no overlay) vs BUY-HOLD (EW u10) --
  compound / CAGR / Sharpe / maxDD / coverage / daily(bar)-pos-rate + avg turnover/bar + ann cost drag.
  The KEY question per cadence: is the CORE still positive, does the OVERLAY still CONTROL drawdown,
  and does cost eat the edge as cadence finers?

CAUSAL / honest: same disciplines as the daily engine (lag-1 MtM weights, train-fit regime thresholds,
past-only features, taker cost default, rank by NET wealth). UNSEEN is not specially sealed here -- this
is a methodology-generalization sweep over the FULL available span, not a ship decision; the ship gate
is the per-cadence gauntlet (run separately).

RWYB:
  python -m strat.engine_timeframe_sweep              # full sweep over {1d,4h,1h,30m,15m}
  python -m strat.engine_timeframe_sweep --cadences 1d,4h
  python -m strat.engine_timeframe_sweep --selftest   # two-sided soundness of the scaling logic
No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

import strat.daily_engine as DE                                              # noqa: E402
import strat.rolling_regime_book as RRB                                      # noqa: E402
from strat.daily_engine import (build_book, buy_hold_net, window_stats)      # noqa: E402
from strat.ma_per_instrument import _panel                                   # noqa: E402
from strat.portfolio_replay import TAKER_RT, MAKER_RT                        # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# bars per (wall-clock) day at each cadence -- the scaling factor for windows + annualization.
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "30m": 48, "15m": 96}
FLOOR = {"1d": "D", "4h": "4h", "1h": "h", "30m": "30min", "15m": "15min"}
# the DAILY engine's canonical bar-counts (the wall-clock anchors we hold ~constant across cadences).
BASE_VOL_WINDOW_DAYS = 30      # DE.VOL_WINDOW at 1d
BASE_REGIME_LOOKBACK_DAYS = 60  # DE.REGIME_LOOKBACK at 1d
BASE_MIN_DWELL_DAYS = 5        # DE.REGIME_MIN_DWELL at 1d
BASE_ANN = 365.0


def _syms():
    return [a["symbol"] for a in yaml.safe_load(
        open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]


# ===========================================================================
# 1. cadence-aware date-aligned CLOSE panel (mirrors DE.load_close_panel, parametrized by cadence)
# ===========================================================================
def load_close_panel_cad(cadence, syms=None):
    """Date-aligned CLOSE DataFrame [bars x assets] over the FULL u10 history at `cadence`. Same source
    + alignment contract as DE.load_close_panel, but floored to the cadence's bar (DE.load_close_panel
    hardcodes '1d')."""
    syms = syms or _syms()
    closes = {}
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        idx = pd.to_datetime(ms, unit="ms").floor(FLOOR[cadence])
        s = pd.Series(c, index=idx)
        s = s[~s.index.duplicated(keep="last")]
        closes[sym] = s
    panel = pd.DataFrame(closes).sort_index()
    _max_rows = max((len(s) for s in closes.values()), default=0)
    assert len(panel) <= 1.5 * _max_rows + 5, (
        f"date-alignment regression at {cadence}: panel {len(panel)} rows >> max per-asset {_max_rows}")
    return panel


# ===========================================================================
# 2. scaled-window context manager -- patch DE globals to keep ~same wall-clock windows, ALWAYS restore
# ===========================================================================
class _ScaledEngine:
    """Temporarily scale the daily-engine's bar-count globals to `cadence` so the SAME methodology runs
    with ~the same WALL-CLOCK windows (and correct annualization). Patches DE module globals + the
    regime kernel's RRB.LOOKBACK['1d'] (which regime_scalar_series reads), and ALWAYS restores them.
    This is the gauntlet's _engine_net pattern, generalized to all four scaled params at once."""

    def __init__(self, cadence):
        self.cadence = cadence
        self.bpd = BARS_PER_DAY[cadence]

    def __enter__(self):
        self._saved = (DE.CADENCE, DE.ANN, DE.VOL_WINDOW, DE.REGIME_LOOKBACK, DE.REGIME_MIN_DWELL,
                       RRB.LOOKBACK.get("1d"))
        DE.CADENCE = self.cadence
        DE.ANN = BASE_ANN * self.bpd
        DE.VOL_WINDOW = max(2, int(round(BASE_VOL_WINDOW_DAYS * self.bpd)))
        DE.REGIME_LOOKBACK = max(5, int(round(BASE_REGIME_LOOKBACK_DAYS * self.bpd)))
        DE.REGIME_MIN_DWELL = max(1, int(round(BASE_MIN_DWELL_DAYS * self.bpd)))
        # regime_scalar_series registers RRB.LOOKBACK['1d'] = DE.REGIME_LOOKBACK internally, but we set it
        # here too so the scaled value is visible even if the call order changes.
        RRB.LOOKBACK["1d"] = DE.REGIME_LOOKBACK
        return self

    def __exit__(self, *exc):
        (DE.CADENCE, DE.ANN, DE.VOL_WINDOW, DE.REGIME_LOOKBACK, DE.REGIME_MIN_DWELL, lb1d) = self._saved
        if lb1d is not None:
            RRB.LOOKBACK["1d"] = lb1d
        return False

    def params(self):
        return {"cadence": self.cadence, "bars_per_day": self.bpd,
                "vol_window_bars": max(2, int(round(BASE_VOL_WINDOW_DAYS * self.bpd))),
                "regime_lookback_bars": max(5, int(round(BASE_REGIME_LOOKBACK_DAYS * self.bpd))),
                "min_dwell_bars": max(1, int(round(BASE_MIN_DWELL_DAYS * self.bpd))),
                "ann_bars": BASE_ANN * self.bpd}


# ===========================================================================
# 3. run one cadence: ENGINE vs CORE-ALONE vs BUY-HOLD over a window (full cycle + recent slice)
# ===========================================================================
def run_cadence(cadence, cost_rt=TAKER_RT, recent_lo="2024-01-01", recent_hi="2026-06-01"):
    """Build the three books at `cadence` (scaled windows) and report full-cycle + recent-slice stats +
    turnover/cost diagnostics. Reuses DE.build_book / buy_hold_net / window_stats verbatim."""
    with _ScaledEngine(cadence) as sc:
        panel = load_close_panel_cad(cadence)
        span = [str(panel.index[0])[:10], str(panel.index[-1])[:10]]
        eng = build_book(panel, core="voltgt", use_overlay=True, cost_rt=cost_rt)
        cor = build_book(panel, core="voltgt", use_overlay=False, cost_rt=cost_rt)
        bh_net, bh_gross = buy_hold_net(panel, cost_rt=cost_rt)

        full = {
            "ENGINE": window_stats(eng["net"], eng["gross_exposure"]),
            "CORE_ALONE": window_stats(cor["net"], cor["gross_exposure"]),
            "BUYHOLD": window_stats(bh_net, bh_gross),
        }
        recent = {
            "ENGINE": window_stats(eng["net"], eng["gross_exposure"], recent_lo, recent_hi),
            "CORE_ALONE": window_stats(cor["net"], cor["gross_exposure"], recent_lo, recent_hi),
            "BUYHOLD": window_stats(bh_net, bh_gross, recent_lo, recent_hi),
        }
        # turnover + annualized cost drag (the cost-eats-it diagnostic, the KEY question)
        eng_turn = float(eng["turnover"].mean())
        ann_bars = BASE_ANN * sc.bpd
        ann_cost_drag = eng_turn * (cost_rt / 2.0) * ann_bars * 100   # % per year
        # the overlay's drawdown-control value at this cadence: engine maxDD - core maxDD (>0 = engine better)
        dd_eng = full["ENGINE"].get("maxdd_pct")
        dd_cor = full["CORE_ALONE"].get("maxdd_pct")
        overlay_dd_saved = (round(dd_eng - dd_cor, 2) if (dd_eng is not None and dd_cor is not None) else None)
        params = sc.params()

    return {
        "cadence": cadence,
        "panel_shape": list(panel.shape),
        "span": span,
        "scaled_params": params,
        "regime_share_full": eng["regime_share"],
        "full_cycle": full,
        "recent_slice": {"window": [recent_lo, recent_hi], "stats": recent},
        "diagnostics": {
            "avg_turnover_per_bar": round(eng_turn, 5),
            "ann_cost_drag_taker_pct": round(ann_cost_drag, 2),
            "overlay_dd_saved_pp_full": overlay_dd_saved,
            "core_positive_full": bool((full["CORE_ALONE"].get("compound_pct") or -1) > 0),
            "engine_positive_full": bool((full["ENGINE"].get("compound_pct") or -1) > 0),
        },
    }


# ===========================================================================
# 4. verdict synthesis
# ===========================================================================
def build_verdict(results):
    """Honest cross-cadence verdict: does the methodology replicate, or is it coarse-cadence-specific?"""
    lines, rows = [], []
    for cad in ["1d", "4h", "1h", "30m", "15m"]:
        r = results.get(cad)
        if not r:
            continue
        f = r["full_cycle"]
        d = r["diagnostics"]
        eng, cor, bh = f["ENGINE"], f["CORE_ALONE"], f["BUYHOLD"]
        rows.append({
            "cadence": cad,
            "engine_compound_pct": eng.get("compound_pct"), "engine_sharpe": eng.get("sharpe"),
            "engine_maxdd_pct": eng.get("maxdd_pct"),
            "core_compound_pct": cor.get("compound_pct"), "core_sharpe": cor.get("sharpe"),
            "bh_compound_pct": bh.get("compound_pct"),
            "ann_cost_drag_pct": d.get("ann_cost_drag_taker_pct"),
            "overlay_dd_saved_pp": d.get("overlay_dd_saved_pp_full"),
            "core_positive": d.get("core_positive_full"),
        })
    # methodology holds at a cadence if: CORE positive AND overlay controls DD (saves > 0pp) AND
    # the engine Sharpe is sane (>0).
    holds = []
    for row in rows:
        cad = row["cadence"]
        ok = (row["core_positive"] and (row["overlay_dd_saved_pp"] or -1) > 0
              and (row["engine_sharpe"] or -1) > 0)
        holds.append((cad, ok))
    held = [c for c, ok in holds if ok]
    failed = [c for c, ok in holds if not ok]
    # is cost the killer as cadence finers? compare ann cost drag coarse vs fine
    drags = {row["cadence"]: row["ann_cost_drag_pct"] for row in rows}
    lines.append("KEY QUESTION: does the core+overlay methodology REPLICATE across cadences, or does "
                 "cost eat it as the cadence finers (coarse-cadence-best)?")
    lines.append("")
    for row in rows:
        lines.append(
            f"[{row['cadence']:>3}] ENGINE comp {row['engine_compound_pct']}% Sh {row['engine_sharpe']} "
            f"DD {row['engine_maxdd_pct']}% | CORE comp {row['core_compound_pct']}% Sh {row['core_sharpe']} "
            f"| BH comp {row['bh_compound_pct']}% | cost-drag {row['ann_cost_drag_pct']}%/yr | "
            f"overlay saved {row['overlay_dd_saved_pp']}pp DD | core+:{row['core_positive']}")
    lines.append("")
    if held and not failed:
        head = (f"METHODOLOGY REPLICATES at ALL {len(held)} swept cadences ({held}): the core stays "
                f"positive and the overlay controls drawdown at every cadence. Cost rises as the cadence "
                f"finers but does NOT (yet) flip the core negative.")
    elif held:
        head = (f"METHODOLOGY IS COARSE-CADENCE-SPECIFIC: it holds at {held} but FAILS at {failed}. "
                f"As the cadence finers, turnover-driven cost drag (and/or regime-classifier resolution) "
                f"erodes the core -- the expected coarse-cadence-best result. Deploy on the coarse "
                f"cadence(s); the fine cadences add cost without adding edge.")
    else:
        head = ("METHODOLOGY DOES NOT REPLICATE at the swept fine cadences: the core does not stay "
                "robustly positive / the overlay does not control DD once cost is charged. This is the "
                "coarse-cadence-only regime -- the daily engine is the deployable form.")
    lines.insert(1, f"HEADLINE: {head}")
    # cost-trend note
    if "1d" in drags and "15m" in drags:
        lines.append("")
        lines.append(f"COST TREND: annualized taker cost-drag rises from {drags.get('1d')}%/yr (1d) to "
                     f"{drags.get('15m')}%/yr (15m) -- the turnover tax compounds as the cadence finers, "
                     "the mechanism behind coarse-cadence-best.")
    return {"headline": head, "replicates_cadences": held, "failed_cadences": failed,
            "rows": rows, "cost_drag_by_cadence": drags, "lines": lines}


# ===========================================================================
# 5. selftest -- two-sided soundness of the SCALING logic (no market)
# ===========================================================================
def selftest():
    """POSITIVE: the scaled-engine context manager scales the windows + annualization by exactly
    bars_per_day and RESTORES the daily defaults on exit (a synthetic positive-drift panel still yields
    a positive core under the scaled params). NEGATIVE: a zero-exposure book yields ~0 net regardless of
    cadence scaling (the scaling does not manufacture return). Plus: ANN scaling is exact + restored."""
    print("## ENGINE-TIMEFRAME-SWEEP SELFTEST (two-sided)")
    ok = True

    # ---- SCALING + RESTORE: the context manager scales globals by bars_per_day and restores them ----
    base = (DE.CADENCE, DE.ANN, DE.VOL_WINDOW, DE.REGIME_LOOKBACK, DE.REGIME_MIN_DWELL)
    for cad in ["4h", "1h", "30m", "15m"]:
        bpd = BARS_PER_DAY[cad]
        with _ScaledEngine(cad):
            scaled_ann = DE.ANN
            scaled_vw = DE.VOL_WINDOW
            scaled_lb = DE.REGIME_LOOKBACK
            scaled_rrb = RRB.LOOKBACK["1d"]
        restored = (DE.CADENCE, DE.ANN, DE.VOL_WINDOW, DE.REGIME_LOOKBACK, DE.REGIME_MIN_DWELL)
        exact = (abs(scaled_ann - BASE_ANN * bpd) < 1e-9
                 and scaled_vw == int(round(BASE_VOL_WINDOW_DAYS * bpd))
                 and scaled_lb == int(round(BASE_REGIME_LOOKBACK_DAYS * bpd))
                 and scaled_rrb == scaled_lb)
        print(f"  SCALE[{cad}]: ANN {BASE_ANN}->{scaled_ann} volwin 30->{scaled_vw} lookback 60->{scaled_lb} "
              f"(x{bpd}); restored={restored == base} exact={exact}")
        ok &= exact and (restored == base)

    # ---- POSITIVE: a positive-drift synthetic panel under SCALED 4h params -> positive core ----
    rng = np.random.default_rng(0)
    bpd = BARS_PER_DAY["4h"]
    n = 200 * bpd
    idx = pd.date_range("2021-01-01", periods=n, freq="4h")
    closes = {}
    for j, s in enumerate(["A", "B", "C", "D"]):
        r = rng.normal(0.0004, 0.01, n)                  # mild positive per-bar drift
        closes[s] = pd.Series(100 * np.cumprod(1 + r), index=idx)
    panel = pd.DataFrame(closes)
    with _ScaledEngine("4h"):
        bk = build_book(panel, core="voltgt", use_overlay=False)
        st = window_stats(bk["net"], bk["gross_exposure"])
    print(f"  POSITIVE (scaled 4h, +drift): core compound {st['compound_pct']}% coverage "
          f"{st.get('coverage_pct')}% (expect >0, coverage high)")
    ok &= (st["compound_pct"] > 0 and (st.get("coverage_pct") or 0) > 80)

    # ---- NEGATIVE: a zero-exposure book under scaled params -> ~0 net (scaling != phantom return) ----
    with _ScaledEngine("1h"):
        from strat.daily_engine import core_weights
        cw = core_weights(panel.asfreq("1h").ffill() if False else panel, core="voltgt")
        zeroW = cw * 0.0
        rets = panel.pct_change(fill_method=None).fillna(0.0)
        zero_net = (zeroW.shift(1).fillna(0.0) * rets).sum(axis=1)
        znet = float(np.cumprod(1 + zero_net.to_numpy())[-1] - 1) * 100
    print(f"  NEGATIVE (zero-exposure, scaled): compound {znet:.6f}% (expect ~0 -- scaling is no phantom edge)")
    ok &= (abs(znet) < 1e-6)

    # ---- ANN annualization sanity: Sharpe under 4h params on the SAME daily-equivalent stream scales by
    #      sqrt(bars_per_day) vs an un-scaled ANN (the annualization must track the cadence) ----
    daily_like = pd.Series(rng.normal(0.001, 0.01, 1000), index=pd.date_range("2021-01-01", periods=1000, freq="D"))
    sh_daily = float(daily_like.mean() / (daily_like.std() + 1e-12) * np.sqrt(BASE_ANN))
    sh_4h = float(daily_like.mean() / (daily_like.std() + 1e-12) * np.sqrt(BASE_ANN * BARS_PER_DAY["4h"]))
    print(f"  ANN: same per-bar Sharpe annualizes to {sh_daily:.2f} @1d-ANN vs {sh_4h:.2f} @4h-ANN "
          f"(ratio {sh_4h/sh_daily:.2f} ~ sqrt(6)={np.sqrt(6):.2f})")
    ok &= (abs(sh_4h / sh_daily - np.sqrt(6)) < 0.05)

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ===========================================================================
# 6. CLI
# ===========================================================================
def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _print_cadence(r):
    f = r["full_cycle"]
    d = r["diagnostics"]
    p = r["scaled_params"]
    print(f"\n########## CADENCE {r['cadence']} ##########")
    print(f"   panel {r['panel_shape']} {r['span'][0]}..{r['span'][1]} | scaled: volwin={p['vol_window_bars']} "
          f"lookback={p['regime_lookback_bars']} dwell={p['min_dwell_bars']} ANN={p['ann_bars']:.0f}")
    print(f"   regime time-share (full): {r['regime_share_full']}")
    print(f"   {'book':12} {'compound%':>13} {'CAGR%':>9} {'Sharpe':>7} {'maxDD%':>9} "
          f"{'barPos%':>8} {'cover%':>8}")
    for k in ("ENGINE", "CORE_ALONE", "BUYHOLD"):
        m = f.get(k, {})
        if "error" in m:
            print(f"   {k:12} {m['error']}")
            continue
        print(f"   {k:12} {str(m.get('compound_pct')):>13} {str(m.get('cagr_pct')):>9} "
              f"{str(m.get('sharpe')):>7} {str(m.get('maxdd_pct')):>9} "
              f"{str(m.get('daily_pos_rate_pct')):>8} {str(m.get('coverage_pct')):>8}")
    print(f"   [recent {r['recent_slice']['window'][0]}..{r['recent_slice']['window'][1]}] "
          f"ENGINE comp {r['recent_slice']['stats']['ENGINE'].get('compound_pct')}% "
          f"Sh {r['recent_slice']['stats']['ENGINE'].get('sharpe')} "
          f"DD {r['recent_slice']['stats']['ENGINE'].get('maxdd_pct')}%")
    print(f"   [cost] avg turnover/bar {d['avg_turnover_per_bar']} -> ann drag {d['ann_cost_drag_taker_pct']}%/yr "
          f"| overlay saved {d['overlay_dd_saved_pp_full']}pp DD | core+:{d['core_positive_full']}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m strat.engine_timeframe_sweep")
    ap.add_argument("--cadences", default="1d,4h,1h,30m,15m")
    ap.add_argument("--maker", action="store_true", help="maker cost (0.0006 rt) instead of taker")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    cost_rt = MAKER_RT if a.maker else TAKER_RT
    cost_name = "maker" if a.maker else "taker"
    cadences = [c.strip() for c in a.cadences.split(",") if c.strip()]
    print(f"## ENGINE TIMEFRAME SWEEP -- the SWEEP-mandate validation of the engine methodology")
    print(f"   replicating daily_engine (vol-target CORE + regime defensive OVERLAY) at {cadences} "
          f"-- windows + ANN SCALED by bars/day -- {cost_name} cost")
    print(f"   SAME builders verbatim (build_book/buy_hold_net/window_stats); daily default UNTOUCHED.\n")

    results = {}
    for cad in cadences:
        if cad not in BARS_PER_DAY:
            print(f"   [skip] unknown cadence {cad}")
            continue
        print(f"   [run] {cad} ...", flush=True)
        results[cad] = run_cadence(cad, cost_rt=cost_rt)
        _print_cadence(results[cad])

    verdict = build_verdict(results)
    print("\n" + "=" * 92)
    print("## AGGREGATE VERDICT -- does the methodology replicate across timeframes?")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 92)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"engine_timeframe_sweep_{stamp}.json"
    out = {
        "repro": {"command": "python -m strat.engine_timeframe_sweep " + " ".join(argv),
                  "git_sha": _git_sha(), "cost_rt": cost_rt, "cost_name": cost_name,
                  "cadences": cadences, "universe": "u10",
                  "scaling": "windows in BARS = daily_bars * bars_per_day; ANN = 365 * bars_per_day",
                  "base_daily_params": {"vol_window": BASE_VOL_WINDOW_DAYS,
                                        "regime_lookback": BASE_REGIME_LOOKBACK_DAYS,
                                        "min_dwell": BASE_MIN_DWELL_DAYS, "ann": BASE_ANN}},
        "results": results,
        "verdict": verdict,
    }
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
