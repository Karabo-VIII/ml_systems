"""src/strat/deep2020_ma_remedy.py -- the LEAST-WEAKNESS MA x Timeframe book per MA kind (2020).

User /orc 2026-06-13/14: "if MA x Timeframe were your ONLY approach, what is the best build with the LEAST
weaknesses?" Combine (a) the brute-force 2020 MA x TF grid we already ran with (b) the literature remedies
synthesized in LITERATURE_CROSSCHECK.md, and REMEDY the specific weaknesses to develop a robust set of MA
strats PER MA KIND (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA). Stay in 2020 (VAL Jul-Sep, OOS Oct-Dec). Side-by-side.

THE WEAKNESSES (from the 2020 deep-dive) -> the LITERATURE REMEDY (knob measured here):
  W1 whipsaw/overtrading at fine TF (cost destroys net; Block A)         -> R1 CONFIRM BAND (hysteresis no-trade
                                                                              buffer around the cross; Zakamulin)
  W2 selection risk / data-snooping (in-sample #1 fails OOS; STW 1999)   -> R2 SELECT-ON-VAL + CLUSTER ENSEMBLE
                                                                              (robust center, not the lucky #1)
  W3/W4 under-participation + over-exit (Blocks A/B)                     -> base FULL stack = trail10 + min_hold12
                                                                              (loose exit; tighter was refuted)
  W7 bear vulnerability / no regime (full-cycle; HYZ/Faber)             -> R3 REGIME GATE (long only when slow MA
                                                                              slope>0; ~neutral IN 2020 bull, kept
                                                                              for robustness -> shows its 2020 cost)
  W8 vol-clustering DD spikes (Faber variance-drain; AQR)               -> R4 VOL-TARGET sizing (scale exposure to
                                                                              a vol target, cap [0,1]; no leverage)

HONEST CONSTRUCTION (this is itself the W2 remedy): the winning CLUSTER and every remedy knob are chosen on the
VAL window (Jul-Sep) ONLY; ALL reported numbers are the held-out OOS (Oct-Dec). We never pick on OOS. LONG-ONLY,
spot, lev<=1 (vol-target only ever REDUCES exposure). Maker cost (MAKER_RT). UNSEEN (2025-2026) untouched.

Side-by-side per (MA kind, TF): RAW-best-on-VAL (single config) vs the remedy ladder (ENS -> +CONFIRM ->
+VOLTGT -> +REGIME = FULL-REMEDY) vs BUYHOLD vs VOLTGT_BH, with a WEAKNESS PROFILE (net/Sharpe/maxDD/time-in/
turnover/VAL->OOS drift). The "least-weakness" winner = best OOS net s.t. maxDD and drift bounded.
RWYB: python -m strat.deep2020_ma_remedy --cadences 1d,4h,2h,1h. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.ma_2020_breakdown import _panel

WIN = ("2020-07-01", "2021-01-01"); SPLIT = "2020-10-01"     # VAL Jul-Sep | OOS Oct-Dec
WARMUP = 400
CADENCES = ["1d", "4h", "2h", "1h"]                          # 30m/15m excluded: Block A shows cost destroys them
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
VOLWIN = {"1d": 14, "4h": 84, "2h": 168, "1h": 336}          # ~2-week realized-vol lookback per cadence
BAND = 0.005                                                 # R1 confirm-band half-width (0.5% hysteresis)


def _speed(periods):
    m = max(periods)
    return "fast<20" if m < 20 else "mid20-60" if m < 60 else "slow60-150" if m < 150 else "vslow150+"


def _held_band(mas, band):
    """Causal cross with a hysteresis no-trade buffer (R1). band=0 -> raw cross.
    2MA: enter MA0 > MA1*(1+band); exit MA0 < MA1*(1-band). 3MA: also require the MA1>MA2 alignment."""
    a, b = mas[0], mas[1]
    if band == 0.0:                                              # fast path: raw cross, no hysteresis loop
        h = (a > b) if len(mas) == 2 else ((a > b) & (b > mas[2]))
        return np.nan_to_num(h).astype(np.int8)
    enter = a > b * (1 + band)
    exitc = a < b * (1 - band)
    if len(mas) == 3:
        cc = mas[2]
        enter = enter & (b > cc * (1 + band))
        exitc = exitc | (b < cc * (1 - band))
    enter = np.nan_to_num(enter); exitc = np.nan_to_num(exitc, nan=1.0)
    h = np.zeros(len(a), np.int8); cur = 0
    for i in range(len(a)):
        if cur == 0 and enter[i]:
            cur = 1
        elif cur == 1 and exitc[i]:
            cur = 0
        h[i] = cur
    return h


def _load_assets(cad):
    """Per cadence, load each asset's (close, ms, ret, win-mask, date-idx, realized-vol-shifted) ONCE
    (MA-independent), plus the global vol target = median realized vol. Reused across all 8 MA kinds."""
    s_ms = pd.Timestamp(WIN[0]).value // 10**6; e_ms = pd.Timestamp(WIN[1]).value // 10**6
    split_ms = pd.Timestamp(SPLIT).value // 10**6                # vol TARGET must be VAL-only (no OOS snoop)
    vw = VOLWIN[cad]; assets = []; rv_meds = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()  # past-only
        idx = pd.to_datetime(ms2[win], unit="ms")
        assets.append({"c": c2, "ret": ret, "win": win, "idx": idx, "rv": rv})
        val_mask = (ms2 >= s_ms) & (ms2 < split_ms)              # VAL leg only (Jul-Sep) -- target never sees OOS
        rv_meds.append(np.nanmedian(rv[val_mask]) if val_mask.sum() > 5 else np.nan)
    vt = float(np.nanmedian(rv_meds)) if rv_meds and not np.all(np.isnan(rv_meds)) else None  # VAL-only vol target
    return assets, vt


def _book(assets, caches, cfgs, mt, band, regime, vt):
    """Daily book return series (equal-weight assets) for an ENSEMBLE of cfgs (len 1 = single) under the
    base FULL stack + optional remedies. Causal throughout (position lagged 1 bar; vol-target uses past vol).
    caches[i] = precomputed {period: MA-array} for asset i (built once per (cad, mt) to avoid recompute)."""
    per_asset = []
    for A, cache in zip(assets, caches):
        c2, ret, win, idx, rv = A["c"], A["ret"], A["win"], A["idx"], A["rv"]
        poss = []
        for name in cfgs:
            pp = _nums(name); mas = [cache[p] for p in pp]
            held = _held_band(mas, band)                                  # R1 (band=0 -> raw)
            held = min_hold(apply_trail_stop(held.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)  # base stack
            if regime:                                                    # R3 regime gate
                slow = mas[-1]; slope = (np.concatenate([[0.0], np.diff(slow)]) > 0).astype(float)
                held = held * slope
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]                  # lag 1 bar
            poss.append(pos)
        pos = np.mean(poss, axis=0)                                       # R2 ensemble
        if vt is not None:                                               # R4 vol-target (cap 1.0 = no leverage)
            mult = np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
            pos = pos * mult
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
        per_asset.append(pd.Series(net, index=idx))
        # stash position for time-in / turnover (OOS only computed later from the book)
    # FIXED equal-weight of the full universe (a not-yet-listed asset = cash = 0 return). fillna(0) over the
    # union index makes the book CADENCE-INVARIANT; the prior mean(skipna=True) reweighted to EW-of-available
    # on partial rows -> a listing-date alignment artifact that inflated finer cadences (verify audit, 2026-06-14).
    b = pd.concat(per_asset, axis=1).fillna(0.0).mean(axis=1)
    return b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def _expo(assets, caches, cfgs, mt, band, regime, vt):
    """Mean OOS exposure (time-in) + turnover proxy across assets, for the weakness profile."""
    tin, turn = [], []
    for A, cache in zip(assets, caches):
        c2, win, idx, rv = A["c"], A["win"], A["idx"], A["rv"]
        poss = []
        for name in cfgs:
            pp = _nums(name); mas = [cache[p] for p in pp]
            held = _held_band(mas, band)
            held = min_hold(apply_trail_stop(held.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
            if regime:
                slow = mas[-1]; slope = (np.concatenate([[0.0], np.diff(slow)]) > 0).astype(float)
                held = held * slope
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            poss.append(pos)
        pos = np.mean(poss, axis=0)
        if vt is not None:
            pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        ser = pd.Series(pos[win], index=idx)
        oos = ser[ser.index >= pd.Timestamp(SPLIT)]
        if len(oos) > 1:
            tin.append(float(oos.mean()))
            turn.append(float(np.abs(np.diff(oos.to_numpy())).sum()))
    return (round(float(np.mean(tin)), 2) if tin else None,
            round(float(np.mean(turn)), 1) if turn else None)


def _metrics(d):
    val = d[d.index < pd.Timestamp(SPLIT)].to_numpy()
    oos = d[d.index >= pd.Timestamp(SPLIT)].to_numpy()
    if len(oos) < 5:
        return None
    eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    val_net = float((np.prod(1 + val) - 1) * 100) if len(val) else 0.0
    oos_net = float((eq[-1] - 1) * 100)
    return {"val_net": round(val_net, 1), "net": round(oos_net, 1),
            "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(dd, 1), "drift": round(oos_net - val_net, 1)}


def _bh(assets, vt=None):
    """Buy-hold (vt None) or vol-targeted buy-hold book daily series."""
    per_asset = []
    for A in assets:
        ret, win, idx, rv = A["ret"], A["win"], A["idx"], A["rv"]
        pos = np.ones(len(ret))
        if vt is not None:
            pos = np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        net = (pos * ret)[win]
        per_asset.append(pd.Series(net, index=idx))
    # FIXED equal-weight of the full universe (a not-yet-listed asset = cash = 0 return). fillna(0) over the
    # union index makes the book CADENCE-INVARIANT; the prior mean(skipna=True) reweighted to EW-of-available
    # on partial rows -> a listing-date alignment artifact that inflated finer cadences (verify audit, 2026-06-14).
    b = pd.concat(per_asset, axis=1).fillna(0.0).mean(axis=1)
    return b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    allcfg = list(ma_cfg)
    allp = sorted({p for n in allcfg for p in _nums(n)})          # union of all grid periods (cache once per cad,mt)
    print(f"MA-REMEDY: {len(allcfg)} configs x {len(MA_TYPES)} MA x {len(CADENCES)} TF | VAL-select, OOS-report, 2020\n")

    export = {}
    for cad in CADENCES:
        assets, vt = _load_assets(cad)
        if not assets:
            print(f"## {cad}: no assets"); continue
        bh = _metrics_bh(_bh(assets)); vbh = _metrics_bh(_bh(assets, vt))
        export[f"{cad}|_BENCH"] = {"buyhold": bh, "voltgt_bh": vbh}
        print(f"########## {cad}  (BUYHOLD net {bh['net']}% Sh {bh['sharpe']} DD {bh['maxdd']}% | "
              f"VOLTGT_BH net {vbh['net']}% Sh {vbh['sharpe']} DD {vbh['maxdd']}%) ##########")
        print(f"   {'MA':5} {'variant':12} {'VALnet':>7} {'OOSnet':>7} {'Sharpe':>7} {'maxDD':>7} "
              f"{'time-in':>8} {'turn':>6} {'drift':>7}")
        for mt in MA_TYPES:
            caches = [{p: _MA[mt](A["c"], p) for p in allp} for A in assets]   # build once per (cad, mt)
            # single-config books (FULL stack, no extra remedy) -> VAL metrics for clustering + RAW-best-on-VAL
            singles = {}
            for name in allcfg:
                d = _book(assets, caches, [name], mt, 0.0, False, None)
                m = _metrics(d)
                if m:
                    singles[name] = m
            if not singles:
                continue
            # cluster by struct x speed; winning cluster = best mean VAL Sharpe (SELECT ON VAL only -> W2 remedy)
            # VAL Sharpe needs the VAL leg; recompute compactly from each single book's VAL via stored drift+net
            clusters = {}
            for name, m in singles.items():
                nums = _nums(name); cl = f"{'2MA' if len(nums)==2 else '3MA'}-{_speed(nums)}"
                clusters.setdefault(cl, []).append(name)
            # pick winning cluster by mean VAL net (proxy for VAL strength; net is the wealth objective)
            best_cl = max(clusters, key=lambda cl: np.mean([singles[n]["val_net"] for n in clusters[cl]]))
            members = clusters[best_cl]
            raw_best = max(members, key=lambda n: singles[n]["val_net"])   # single best-on-VAL in the winning cluster

            variants = {
                "RAW-best": ([raw_best], 0.0, False, None),                # single config, FULL stack
                "ENS":      (members, 0.0, False, None),                   # +R2 cluster ensemble
                "+CONFIRM": (members, BAND, False, None),                  # +R1 confirm band
                "+VOLTGT":  (members, BAND, False, vt),                    # +R4 vol-target
                "FULL-REM": (members, BAND, True, vt),                     # +R3 regime gate (= all remedies)
            }
            rows = {}
            for vname, (cfgs, band, regime, vtv) in variants.items():
                d = _book(assets, caches, cfgs, mt, band, regime, vtv)
                m = _metrics(d)
                if not m:
                    continue
                tin, turn = _expo(assets, caches, cfgs, mt, band, regime, vtv)
                m["time_in"] = tin; m["turnover"] = turn
                rows[vname] = m
                print(f"   {mt:5} {vname:12} {m['val_net']:>7} {m['net']:>7} {m['sharpe']:>7} {m['maxdd']:>7} "
                      f"{str(tin):>8} {str(turn):>6} {m['drift']:>7}")
            export[f"{cad}|{mt}"] = {"winning_cluster": best_cl, "n_members": len(members),
                                     "raw_best_cfg": f"{mt}({','.join(map(str,_nums(raw_best)))})", "variants": rows}
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump(export, open(op / f"ma_remedy_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'ma_remedy_{jt}.json'}  -- least-weakness MA book per MA kind x TF (VAL-select, OOS-report)")
    return 0


def _metrics_bh(d):
    oos = d[d.index >= pd.Timestamp(SPLIT)].to_numpy()
    if len(oos) < 5:
        return {"net": None, "sharpe": None, "maxdd": None}
    eq = np.cumprod(1 + oos); pk = np.maximum.accumulate(eq); dd = float(((eq - pk) / pk).min() * 100)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(oos) / (np.std(oos) + 1e-12) * np.sqrt(365)), 2), "maxdd": round(dd, 1)}


if __name__ == "__main__":
    sys.exit(main())
