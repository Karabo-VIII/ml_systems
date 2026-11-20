"""src/strat/complete_stack.py -- the COMPLETE stack: full keeper + flicker-fixed BTC market gate.

WHY (user /orc 2026-06-12): the regime search found the MARKET gate (BTC > BTC.SMA100) is the only
door-opener (4h bear -6.0 -> -2.7, no rally/bull cost) but it FLICKERS at 1h in a choppy bull (2024 bull
21.4 -> 10.1). This assembles the full stack (FIXED 2MA-slow + TRAIL 10% + min_hold 12 + MAKER) and tests
flicker-fixed gate forms to keep the bear benefit WITHOUT the bull tax:

  GATE_NONE      no gate (the prior full-stack winner)
  GATE_BTC100    BTC > BTC.SMA100                       (raw -- flickers at 1h)
  GATE_BTC100_H  BTC vs SMA100 with 3% HYSTERESIS band  (allow-long latches; kills flicker)
  GATE_BTC200    BTC > BTC.SMA200                       (slower market SMA -- smoother)

Per (gate, cadence in {4h,1h,30m,15m}, period): book ROI% / maxDD%. MAKER cost. Equal-weight u10 book,
causal MtM, all TRAIN-era. Goal: pick the gate form that softens the bear at every cadence with no bull
flicker. RWYB: python -m strat.complete_stack. No emoji (cp1252).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold

PERIODS = {
    "Jan2020_rally": ("2020-01-07", "2020-02-07", "rally"),
    "Feb2020_revsl": ("2020-02-07", "2020-03-07", "reversal"),
    "JanFeb_comb":   ("2020-01-07", "2020-03-07", "comb"),
    "Jun2022_bear":  ("2022-06-01", "2022-07-01", "bear"),
    "Feb2024_bull":  ("2024-02-01", "2024-03-01", "bull"),
}
CADENCES = ["4h", "1h", "30m", "15m"]
WARMUP = 600
GATES = ["GATE_NONE", "GATE_BTC100", "GATE_BTC100_H", "GATE_BTC200"]


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


def _sma(c, n):
    if len(c) < n:
        return np.full(len(c), np.nan)
    cs = np.cumsum(np.insert(c, 0, 0.0))
    out = np.full(len(c), np.nan); out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _hysteresis(c, n, band):
    """latched regime: ON when close > SMA*(1+band), OFF when close < SMA*(1-band), else hold."""
    s = _sma(c, n)
    up = c > s * (1 + band)
    dn = c < s * (1 - band)
    state = np.zeros(len(c), dtype=np.int8)
    cur = 0
    for i in range(len(c)):
        if np.isnan(s[i]):
            cur = 0
        elif up[i]:
            cur = 1
        elif dn[i]:
            cur = 0
        state[i] = cur
    return state.astype(bool)


_BTC = {}


def _btc_gate(cadence, kind):
    key = (cadence, kind)
    if key in _BTC:
        return _BTC[key]
    o, h, l, c, ms = _cached_panel("BTCUSDT", cadence)
    if kind == "GATE_BTC100":
        reg = np.nan_to_num(c > _sma(c, 100)).astype(bool)
    elif kind == "GATE_BTC100_H":
        reg = _hysteresis(c, 100, 0.03)
    elif kind == "GATE_BTC200":
        reg = np.nan_to_num(c > _sma(c, 200)).astype(bool)
    else:
        reg = np.ones(len(c), bool)
    _BTC[key] = (ms, reg)
    return _BTC[key]


def _weight(name, o, c, ms, cadence, gate):
    h = holding_state(name, o, c, c, c).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    h = min_hold(h, 12).astype(np.float64)          # full keeper stack
    if gate == "GATE_NONE":
        return h
    btc_ms, btc_reg = _btc_gate(cadence, gate)
    idx = np.clip(np.searchsorted(btc_ms, ms, side="right") - 1, 0, len(btc_reg) - 1)
    return h * btc_reg[idx].astype(np.float64)


def book_net(config_set, cadence, start, end, gate):
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell = []
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, cadence)
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o, c, ms = o[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c) < 20:
            continue
        wm = ms >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        for name in config_set:
            w = _weight(name, o, c, ms, cadence, gate)
            pos = np.zeros(len(c)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = pos * ret - flips * (MAKER_RT / 2.0)
            per_cell.append(net[wm])
    if not per_cell:
        return None
    m = min(len(x) for x in per_cell)
    return np.mean([x[:m] for x in per_cell], axis=0)


def metrics(book):
    if book is None or len(book) < 5:
        return {}
    eq = np.cumprod(1 + book); peak = np.maximum.accumulate(eq)
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(float(((eq - peak) / peak).min() * 100), 1)}


def main() -> int:
    allcfg = {}
    for fam in ("2MA", "3MA"):
        allcfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(allcfg)
    slow = [n for n in allcfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    print(f"COMPLETE stack = FIXED({len(slow)} cfg) + TRAIL10 + HOLD12 + MAKER; gates: {GATES}\n")

    results = {}
    for cad in CADENCES:
        print(f"########## CADENCE {cad} -- gate x period (ROI% / maxDD%), MAKER ##########")
        print(f"   {'gate':16}" + "".join(f"{PERIODS[p][2][:9]:>15}" for p in PERIODS))
        for gate in GATES:
            row = f"   {gate:16}"
            for plabel, (s, e, _n) in PERIODS.items():
                mt = metrics(book_net(slow, cad, s, e, gate))
                results[(cad, gate, plabel)] = mt
                row += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))):>15}" if mt else f"{'--':>15}"
            print(row)
        print()

    # which gate keeps the bear benefit AND the bull (no flicker), per cadence
    print("[DECISION] per cadence: d_bear and d_bull vs GATE_NONE (want d_bear>0 AND d_bull>=~0)")
    print(f"   {'cadence':8} {'gate':16} {'d_bear':>8} {'d_bull':>8} {'verdict':>12}")
    for cad in CADENCES:
        bn_bear = results[(cad, "GATE_NONE", "Jun2022_bear")].get("roi", np.nan)
        bn_bull = results[(cad, "GATE_NONE", "Feb2024_bull")].get("roi", np.nan)
        for gate in GATES[1:]:
            db = results[(cad, gate, "Jun2022_bear")].get("roi", np.nan) - bn_bear
            dl = results[(cad, gate, "Feb2024_bull")].get("roi", np.nan) - bn_bull
            verdict = "KEEP" if (db > 0.3 and dl > -1.5) else ("flicker" if dl < -3 else "weak")
            print(f"   {cad:8} {gate:16} {db:>+8.1f} {dl:>+8.1f} {verdict:>12}")

    # chart: bear ROI and bull ROI by gate per cadence
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(CADENCES)); w = 0.2
    for i, gate in enumerate(GATES):
        ax[0].bar(x + (i - 1.5) * w, [results[(c, gate, "Jun2022_bear")].get("roi", np.nan) for c in CADENCES], w, label=gate)
        ax[1].bar(x + (i - 1.5) * w, [results[(c, gate, "Feb2024_bull")].get("roi", np.nan) for c in CADENCES], w, label=gate)
    for a, t in ((ax[0], "BEAR (Jun2022) -- want gate to LIFT (less negative)"), (ax[1], "BULL (Feb2024) -- want gate to NOT cut (no flicker)")):
        a.set_xticks(x); a.set_xticklabels(CADENCES); a.axhline(0, color="k", lw=0.7); a.legend(fontsize=7); a.set_title(t); a.set_ylabel("ROI %")
    fig.tight_layout()
    out = ROOT.parent / "runs" / "periods" / "TRAIN" / "_CROSS" / "charts" / "complete_stack.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    json.dump({f"{c}|{g}|{p}": m for (c, g, p), m in results.items()},
              open(out.parent.parent / "analysis" / "complete_stack.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
