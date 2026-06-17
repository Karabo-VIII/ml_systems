"""src/strat/regime_gate_search.py -- can a BETTER regime gate open the long-only sit-out door?

WHY (user /orc 2026-06-12, the open door): the cumulative ladder showed a raw SMA200 sit-out is HARMFUL
(too laggy at 4h -> sat out the whole young rally; throws away all upside via full-cash). But the bear
stays NEGATIVE for a long-only book, and a WORKING regime gate is the only lever (short out of scope).
So we SEARCH the regime-instrument space on the keeper stack (FIXED 2MA-slow + TRAIL 10%), asking the
two-sided question: does the gate cut the BEAR/REVERSAL damage WITHOUT killing the RALLY/BULL?

GATES (applied to the held series of the keeper stack):
  G0_NONE      no gate (baseline = FIXED+TRAIL)
  G_SMA200     long only when close > SMA200            (the known failure -- reference)
  G_SMA100     long only when close > SMA100            (faster self-regime)
  G_SMA50      long only when close > SMA50             (much faster)
  G_SLOPE100   long only when SMA100 is RISING (10-bar) (trend filter, not level)
  G_HALF200    HALF size (0.5w) below SMA200            (de-risk, not full sit-out)
  G_BTC100     long only when BTC > BTC.SMA100          (MARKET-wide regime -- the ML 'trade-vs-sit-out' door)

Per (gate, cadence in {4h,1h}, period): book ROI%, maxDD%. TAKER throughout (isolate the gate, no maker
confound). Decision: GOOD gate = improves bear/reversal AND does not materially hurt rally/bull.
Equal-weight u10 book, causal MtM, all TRAIN-era. RWYB: python -m strat.regime_gate_search. No emoji.
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
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel

PERIODS = {
    "Jan2020_rally": ("2020-01-07", "2020-02-07", "rally"),
    "Feb2020_revsl": ("2020-02-07", "2020-03-07", "reversal"),
    "JanFeb_comb":   ("2020-01-07", "2020-03-07", "rally->reversal"),
    "Jun2022_bear":  ("2022-06-01", "2022-07-01", "bear"),
    "Feb2024_bull":  ("2024-02-01", "2024-03-01", "bull"),
}
CADENCES = ["4h", "1h"]
ANN = {"4h": 365 * 6, "1h": 365 * 24}
WARMUP = 600
GATES = ["G0_NONE", "G_SMA200", "G_SMA100", "G_SMA50", "G_SLOPE100", "G_HALF200", "G_BTC100"]


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


def _sma(c, n):
    if len(c) < n:
        return np.full(len(c), np.nan)
    cs = np.cumsum(np.insert(c, 0, 0.0))
    out = np.full(len(c), np.nan); out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


_BTC_CACHE = {}


def _btc_regime(cadence):
    """BTC market regime on BTC's full grid: (btc_ms, close>SMA100 bool). Cached per cadence."""
    if cadence in _BTC_CACHE:
        return _BTC_CACHE[cadence]
    o, h, l, c, ms = _cached_panel("BTCUSDT", cadence)
    reg = c > _sma(c, 100)
    reg = np.nan_to_num(reg).astype(bool)
    _BTC_CACHE[cadence] = (ms, reg)
    return _BTC_CACHE[cadence]


def _weight(name, o, c, ms, cadence, gate):
    """float weight (0..1) for the keeper stack (FIXED+TRAIL10) under a regime gate."""
    h = holding_state(name, o, c, c, c).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.float64)   # keeper stack base
    if gate == "G0_NONE":
        return h
    if gate == "G_SMA200":
        return h * (c > _sma(c, 200)).astype(np.float64)
    if gate == "G_SMA100":
        return h * (c > _sma(c, 100)).astype(np.float64)
    if gate == "G_SMA50":
        return h * (c > _sma(c, 50)).astype(np.float64)
    if gate == "G_SLOPE100":
        s = _sma(c, 100); rising = np.zeros(len(c), bool); rising[10:] = s[10:] > s[:-10]
        return h * rising.astype(np.float64)
    if gate == "G_HALF200":
        above = (c > _sma(c, 200))
        return h * np.where(np.nan_to_num(above).astype(bool), 1.0, 0.5)
    if gate == "G_BTC100":
        btc_ms, btc_reg = _btc_regime(cadence)
        idx = np.clip(np.searchsorted(btc_ms, ms, side="right") - 1, 0, len(btc_reg) - 1)
        return h * btc_reg[idx].astype(np.float64)
    return h


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
            net = pos * ret - flips * (TAKER_RT / 2.0)
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
    print(f"FIXED-slow family: {len(slow)} configs; gates: {GATES}\n")

    results = {}
    for cad in CADENCES:
        print(f"########## CADENCE {cad} -- gate x period (ROI% / maxDD%), keeper stack + gate, TAKER ##########")
        print(f"   {'gate':12}" + "".join(f"{PERIODS[p][2][:11]:>16}" for p in PERIODS))
        for gate in GATES:
            row = f"   {gate:12}"
            for plabel, (s, e, _n) in PERIODS.items():
                mt = metrics(book_net(slow, cad, s, e, gate))
                results[(cad, gate, plabel)] = mt
                row += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))):>16}" if mt else f"{'--':>16}"
            print(row)
        print()

    # decision view: rally-cost vs bear-benefit vs the no-gate baseline (4h + 1h pooled)
    print("[DECISION] vs G0_NONE: rally/bull cost (want ~0) vs bear/reversal benefit (want +), pooled 4h+1h")
    print(f"   {'gate':12} {'d_rally':>9} {'d_bull':>9} {'d_bear':>9} {'d_revsl':>9} {'verdict':>10}")
    base = {p: np.mean([results[(c, 'G0_NONE', p)].get('roi', np.nan) for c in CADENCES]) for p in PERIODS}
    for gate in GATES[1:]:
        d = {p: np.mean([results[(c, gate, p)].get('roi', np.nan) for c in CADENCES]) - base[p] for p in PERIODS}
        helps = d["Jun2022_bear"] > 0.5 or d["Feb2020_revsl"] > 0.5
        hurts = d["Jan2020_rally"] < -1.0 or d["Feb2024_bull"] < -1.0
        verdict = "OPENS" if (helps and not hurts) else ("HARMFUL" if hurts else "neutral")
        print(f"   {gate:12} {d['Jan2020_rally']:>+9.1f} {d['Feb2024_bull']:>+9.1f} {d['Jun2022_bear']:>+9.1f} {d['Feb2020_revsl']:>+9.1f} {verdict:>10}")

    # chart: bear-benefit vs rally-cost scatter (the two-sided test), 4h+1h pooled
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for gate in GATES:
        rally = np.mean([results[(c, gate, "Jan2020_rally")].get("roi", np.nan) for c in CADENCES])
        bear = np.mean([results[(c, gate, "Jun2022_bear")].get("roi", np.nan) for c in CADENCES])
        ax[0].scatter(rally, bear, s=80); ax[0].annotate(gate, (rally, bear), fontsize=8)
    ax[0].set_xlabel("rally ROI % (want high -- gate must not kill it)")
    ax[0].set_ylabel("bear ROI % (want less negative -- the open door)")
    ax[0].set_title("Regime gate two-sided test (4h+1h pooled): top-right is the goal")
    ax[0].axhline(base["Jun2022_bear"], color="grey", ls=":", lw=0.8)
    ax[0].axvline(base["Jan2020_rally"], color="grey", ls=":", lw=0.8)
    # per-gate combined ROI bars
    x = np.arange(len(PERIODS)); w = 0.12
    for i, gate in enumerate(GATES):
        vals = [np.mean([results[(c, gate, p)].get("roi", np.nan) for c in CADENCES]) for p in PERIODS]
        ax[1].bar(x + (i - 3) * w, vals, w, label=gate)
    ax[1].set_xticks(x); ax[1].set_xticklabels([PERIODS[p][2][:9] for p in PERIODS], fontsize=8, rotation=10)
    ax[1].axhline(0, color="k", lw=0.7); ax[1].legend(fontsize=7); ax[1].set_ylabel("ROI % (4h+1h mean)")
    ax[1].set_title("Regime gate x period (keeper stack + gate)")
    fig.tight_layout()
    out = ROOT.parent / "runs" / "periods" / "TRAIN" / "_CROSS" / "charts" / "regime_gate_search.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    json.dump({f"{c}|{g}|{p}": m for (c, g, p), m in results.items()},
              open(out.parent.parent / "analysis" / "regime_gate_search.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
