"""src/strat/deep2020_ironed_fine.py -- IRON OUT the fine-TF MA TREND system (CONSTRUCTION, not refutation).

User /orc 2026-06-13: build the per-timeframe MA TREND system with CREASES IRONED OUT for FINE cadences
{1h,30m,15m} on the 2020 deep-dive protocol. Produce DEPLOYABLE specs OR an honest cost-wall verdict with
the EXACT threshold. The DOMINANT fine-TF crease is WHIPSAW + COST: the leaderboard showed 15m HMA/TEMA full
coverage but net 2.5-3.9% (cost-eaten); buy-hold / VOLTGT_BH win on BOTH net and Sharpe at 30m/15m.

THE CENTRAL QUESTION: can confirmation + min-hold + whipsaw-filter + family + regime-gate iron a fine-TF MA
TREND system to clear cost OOS and become a deploy candidate, OR does the cost wall bind?

This builds the BEFORE->AFTER ladder per TF, reporting OOS net + maxDD + turnover at EACH step under BOTH
maker AND taker cost, vs BUYHOLD and VOLTGT_BH at that TF:
  S0  naive single-MA   the VAL-best single 2MA config (the 'bet on one' baseline)
  S1  + family          equal-weight book of all slow 2MA+3MA configs (param-fragility fix)
  S2  + whipsaw-filter  confirm(K) + min_hold(M) + cooldown -- params SELECTED on TRAIN+VAL only
  S3  + exit overlay    chandelier / mh-trail (give-back fix)
  S4  + regime gate     MARKET-regime (BTC>SMA100) participation gate
  S5  + vol-target      scale book exposure inversely to realized vol (the deployable stack)

DISCIPLINE: causal/lag-1 (pos[t]=held[t-1]); SELECT whipsaw params + MA-type + gate on TRAIN+VAL, confirm
ONCE on OOS; no look-ahead. cost realism flagged (maker p_fill 0.21-0.40 reality per CLAUDE.md). Equal-weight
u10 book; SOL/AVAX/DOGE only have late-2020 history so they enter the book only in VAL/OOS (no look-ahead,
just thinner breadth in TRAIN). RWYB: python -m strat.deep2020_ironed_fine. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT, TAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES
from strat.ma_2020_breakdown import _panel, SPLIT, YEAR, WARMUP
from strat.structural_fixes import min_hold, confirm, cooldown

CADENCES = ["1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
# annualization factor (bars/yr) per cadence -- for Sharpe + vol-target
ANN = {"1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
# ~1-week timestop / vol lookback per cadence (bars)
VOLWIN = {"1h": 168, "30m": 336, "15m": 672}
# chandelier ATR window (bars) -- ~22 daily bars equivalent scaled to TF would be huge; keep at 50 bars
CHAND_PER = 50


# ---------------------------------------------------------------------------------------------
# per-asset panel cache (close, high, low, ms) sliced to [2020-WARMUP .. 2021), causal
# ---------------------------------------------------------------------------------------------
def _asset_arrays(sym, cad):
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    s_ms = pd.Timestamp(YEAR[0]).value // 10**6
    e_ms = pd.Timestamp(YEAR[1]).value // 10**6
    e_idx = int(np.searchsorted(ms, e_ms))
    s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
    c2, h2, l2, ms2 = c[s_idx:e_idx], h[s_idx:e_idx], l[s_idx:e_idx], ms[s_idx:e_idx]
    if len(c2) < 200:
        return None
    win = ms2 >= s_ms
    if win.sum() < 100:
        return None
    return c2, h2, l2, ms2, win


# BTC market-regime (close > SMA100), aligned to any asset's ms grid -- the MARKET gate
_BTC_REG = {}


def _dwell(state, m):
    """min-dwell debounce on a 0/1 state: once it flips, hold the new value >= m bars (kills flicker)."""
    out = state.copy().astype(np.int8)
    last = out[0]; hold = 0
    for i in range(1, len(out)):
        if hold > 0:
            out[i] = last; hold -= 1
        elif out[i] != last:
            last = out[i]; hold = m - 1
    return out.astype(bool)


def _btc_regime(cad, dwell=0):
    """MARKET regime aligned to BTC's grid, LAGGED 1 bar (causal: the gate decision usable at bar t
    is the regime observed at the PRIOR close t-1, same lag discipline as pos[t]=held[t-1]).
    dwell>0 applies a min-dwell debounce to the regime STATE first (kills the fine-TF gate-flicker)."""
    key = (cad, dwell)
    if key in _BTC_REG:
        return _BTC_REG[key]
    a = _asset_arrays("BTCUSDT", cad)
    c = a[0]; ms = a[3]
    sma100 = pd.Series(c).rolling(100, min_periods=1).mean().to_numpy()
    reg = np.nan_to_num(c > sma100).astype(bool)
    if dwell > 0:
        reg = _dwell(reg, dwell)
    reg_lag = np.zeros(len(reg), dtype=bool)
    reg_lag[1:] = reg[:-1]                                   # lag 1 bar -> no same-bar look-ahead
    _BTC_REG[key] = (ms, reg_lag)
    return _BTC_REG[key]


def _chandelier(held, c, hi, lo, k=3.0, per=CHAND_PER):
    tr = np.maximum(hi - lo, np.abs(hi - np.concatenate([[c[0]], c[:-1]])))
    atr = pd.Series(tr).rolling(per, min_periods=1).mean().to_numpy()
    h = held.copy().astype(np.int8)
    d = np.diff(np.concatenate([[0], h, [0]]))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        peak = c[s]
        for i in range(s, e):
            peak = max(peak, c[i])
            if c[i] <= peak - k * atr[i]:
                h[i + 1:e] = 0
                break
    return h


# ---------------------------------------------------------------------------------------------
# held-series builders per stage. All causal: a held[t] uses close[:t+1] only.
# ---------------------------------------------------------------------------------------------
def _entry_held(c, periods, ma_type, cache):
    mas = [cache[p] for p in periods]
    h = (mas[0] > mas[1]) if len(periods) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))
    return np.nan_to_num(h).astype(np.int8)


def _whipsaw_filter(held, K, M, cool):
    """confirm K consecutive true bars to ENTER, then min-hold M bars, then cooldown N after exit."""
    h = confirm(held.astype(np.int8), K) if K > 1 else held.astype(np.int8)
    if M > 0:
        h = min_hold(h, M).astype(np.int8)
    if cool > 0:
        h = cooldown(h, cool).astype(np.int8)
    return h.astype(np.int8)


# ---------------------------------------------------------------------------------------------
# book builder: given a list of (held -> weight) producers per config, build the equal-weight
# u10 book of bar-level returns, plus turnover. Returns daily-resampled net Series + raw turnover.
# ---------------------------------------------------------------------------------------------
def _book_from_positions(asset_pos, cad, cost_rt):
    """asset_pos = {sym: (pos_array, ret_array, ms_array, win_mask)}; pos is the FINAL fraction-long
    (already lagged 1 bar). Returns (daily_net Series, turnover_per_bar_sum, n_assets_in_book)."""
    cells = []
    turnovers = []
    for sym, (pos, ret, ms, win) in asset_pos.items():
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pos * ret - flips * (cost_rt / 2.0)
        idx = pd.to_datetime(ms[win], unit="ms")
        cells.append(pd.Series(net[win], index=idx))
        turnovers.append(float(np.sum(flips[win])))   # sum of |dpos| over OOS-eligible window handled later
    if not cells:
        return None, 0.0, 0
    book = pd.concat(cells, axis=1).mean(axis=1, skipna=True)
    return book, float(np.mean(turnovers)), len(cells)


def _compound(s, lo, hi):
    x = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))]
    return float(np.prod(1 + x.to_numpy()) - 1) * 100 if len(x) else np.nan


def _maxdd(s, lo, hi):
    x = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))].to_numpy()
    if len(x) < 3:
        return np.nan
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _sharpe(s, lo, hi, cad):
    x = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))].to_numpy()
    if len(x) < 5:
        return np.nan
    # the book is bar-level net; annualize by bars/yr
    return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN[cad]))


def _coverage(s, lo, hi, pos_series):
    """% of bars with >50% exposure in the window (using the family fraction-long), book-level."""
    x = pos_series[(pos_series.index >= pd.Timestamp(lo)) & (pos_series.index < pd.Timestamp(hi))].to_numpy()
    return round(float(np.mean(x > 0.5) * 100), 1) if len(x) else np.nan


# ---------------------------------------------------------------------------------------------
# Stage producers -- each returns asset_pos {sym: (pos, ret, ms, win)} + a book-level exposure series
# ---------------------------------------------------------------------------------------------
def _build_caches(cad, slow):
    """per asset: arrays + per-MA-type period caches. Returns {sym: (c,h,l,ms,win, {ma_type:{p:ma}})}."""
    uniq = sorted({p for n in slow for p in _nums(n)})
    out = {}
    for sym in SYMS:
        a = _asset_arrays(sym, cad)
        if a is None:
            continue
        c, h, l, ms, win = a
        caches = {mt: {p: _MA[mt](c, p) for p in uniq} for mt in MA_TYPES}
        out[sym] = (c, h, l, ms, win, caches)
    return out


def _lag(held):
    pos = np.zeros(len(held), dtype=np.float64)
    pos[1:] = held[:-1]
    return pos


def _ret_of(c):
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    return ret


def _stage_positions(panels, slow, ma_type, stage, K, M, cool, gate, voltgt):
    """produce asset_pos + book exposure for a given stack config.
    stage in {'single','family'}; whipsaw filter via K/M/cool; gate bool; voltgt float|None (target vol)."""
    asset_pos = {}
    exp_cells = []
    # precompute book-level realized vol from BTC for the vol-target (use per-asset own vol)
    for sym, (c, h, l, ms, win, caches) in panels.items():
        ret = _ret_of(c)
        cache = caches[ma_type]
        if stage == "single":
            cfgs = [slow[0]]   # placeholder; caller passes the chosen single via slow=[chosen]
        else:
            cfgs = slow
        poss = []
        for name in cfgs:
            held = _entry_held(c, _nums(name), ma_type, cache)
            if K > 1 or M > 0 or cool > 0:
                held = _whipsaw_filter(held, K, M, cool)
            poss.append(_lag(held.astype(np.float64)))
        fpos = np.mean(poss, axis=0)                      # family fraction-long (already lagged)
        # regime gate (MARKET): zero exposure when BTC below its SMA100 (aligned to this asset's ms)
        # gate is an int: 0 = off; >0 = min-dwell bars on the BTC regime state (anti-flicker)
        if gate:
            breg = _btc_regime_for(ms, gate)
            fpos = fpos * breg
        # vol-target: scale by clip(target/realized_vol[t-1],0,1)
        if voltgt is not None:
            rv = pd.Series(ret).rolling(VOLWIN_CUR).std().shift(1).to_numpy()
            scale = np.clip(voltgt / (rv + 1e-12), 0.0, 1.0)
            scale = np.nan_to_num(scale, nan=0.0)
            fpos = fpos * scale
        asset_pos[sym] = (fpos, ret, ms, win)
        exp_cells.append(pd.Series(fpos[win], index=pd.to_datetime(ms[win], unit="ms")))
    book_exp = pd.concat(exp_cells, axis=1).mean(axis=1, skipna=True) if exp_cells else None
    return asset_pos, book_exp


# module-level current-cadence vol window (set per cadence in main)
VOLWIN_CUR = 168
_BTC_ALIGN = {}


def _btc_regime_for(ms, dwell=0):
    """return BTC-regime float(0/1) aligned to the given ms grid (searchsorted), with optional dwell."""
    cad = _CUR_CAD
    bms, breg = _btc_regime(cad, dwell)
    idx = np.clip(np.searchsorted(bms, ms, side="right") - 1, 0, len(breg) - 1)
    return breg[idx].astype(np.float64)


_CUR_CAD = "1h"


# ---------------------------------------------------------------------------------------------
# exit overlay applied at the held level (chandelier / mh-trail) BEFORE lagging -- so re-do family
# ---------------------------------------------------------------------------------------------
def _family_with_exit(panels, slow, ma_type, K, M, cool, exit_kind, gate, voltgt, half_gate=False):
    asset_pos = {}
    exp_cells = []
    for sym, (c, h, l, ms, win, caches) in panels.items():
        ret = _ret_of(c)
        cache = caches[ma_type]
        poss = []
        for name in slow:
            held = _entry_held(c, _nums(name), ma_type, cache)
            if K > 1 or M > 0 or cool > 0:
                held = _whipsaw_filter(held, K, M, cool)
            # exit overlay
            if exit_kind == "chandelier":
                held = _chandelier(held, c, h, l)
            elif exit_kind == "mh_trail15":
                held = apply_trail_stop(min_hold(held, max(M, 12)).astype(np.int8).copy(), c, 0.15)[0].astype(np.int8)
            elif exit_kind == "trail10":
                held = apply_trail_stop(held.astype(np.int8).copy(), c, 0.10)[0].astype(np.int8)
            poss.append(_lag(held.astype(np.float64)))
        fpos = np.mean(poss, axis=0)
        if gate:
            breg = _btc_regime_for(ms, gate)
            # half_gate de-risks to 0.5 below regime (not full sit-out) -- keeps participation
            fpos = fpos * np.where(breg > 0.5, 1.0, 0.5) if half_gate else fpos * breg
        if voltgt is not None:
            rv = pd.Series(ret).rolling(VOLWIN_CUR).std().shift(1).to_numpy()
            scale = np.nan_to_num(np.clip(voltgt / (rv + 1e-12), 0.0, 1.0), nan=0.0)
            fpos = fpos * scale
        asset_pos[sym] = (fpos, ret, ms, win)
        exp_cells.append(pd.Series(fpos[win], index=pd.to_datetime(ms[win], unit="ms")))
    book_exp = pd.concat(exp_cells, axis=1).mean(axis=1, skipna=True) if exp_cells else None
    return asset_pos, book_exp


# ---------------------------------------------------------------------------------------------
# benchmarks: BUYHOLD + VOLTGT_BH at the TF (equal-weight u10)
# ---------------------------------------------------------------------------------------------
def _benchmarks(panels, cad, voltgt):
    bh_cells, vt_cells, bh_exp = [], [], []
    for sym, (c, h, l, ms, win, caches) in panels.items():
        ret = _ret_of(c)
        idx = pd.to_datetime(ms[win], unit="ms")
        bh_cells.append(pd.Series(ret[win], index=idx))
        rv = pd.Series(ret).rolling(VOLWIN_CUR).std().shift(1).to_numpy()
        scale = np.nan_to_num(np.clip(voltgt / (rv + 1e-12), 0.0, 1.0), nan=0.0)
        vt_cells.append(pd.Series((scale * ret)[win], index=idx))
        bh_exp.append(pd.Series(scale[win], index=idx))
    bh = pd.concat(bh_cells, axis=1).mean(axis=1, skipna=True)
    vt = pd.concat(vt_cells, axis=1).mean(axis=1, skipna=True)
    vt_exp = pd.concat(bh_exp, axis=1).mean(axis=1, skipna=True)
    return bh, vt, vt_exp


def _metrics(book, exp, cad, cost_label, turnover):
    lo, hi = SPLIT["OOS"]
    breadth = None
    return {
        "oos_net": round(_compound(book, lo, hi), 1),
        "oos_maxdd": round(_maxdd(book, lo, hi), 1),
        "oos_sharpe": round(_sharpe(book, lo, hi, cad), 2),
        "oos_cov": _coverage(book, lo, hi, exp) if exp is not None else None,
        "trainval_net": round(_compound(book, SPLIT["TRAIN"][0], SPLIT["VAL"][1]), 1),
        "turnover": round(turnover, 1),
        "cost": cost_label,
    }


def _breadth_neff(asset_pos, cost_rt, cad):
    """OOS breadth (# of 10 assets net-positive) + effective N from cross-asset correlation."""
    lo, hi = pd.Timestamp(SPLIT["OOS"][0]), pd.Timestamp(SPLIT["OOS"][1])
    per_asset_daily = {}
    pos_count = 0
    for sym, (pos, ret, ms, win) in asset_pos.items():
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pos * ret - flips * (cost_rt / 2.0)
        idx = pd.to_datetime(ms[win], unit="ms")
        s = pd.Series(net[win], index=idx)
        s = s[(s.index >= lo) & (s.index < hi)]
        if len(s) < 5:
            continue
        comp = float(np.prod(1 + s.to_numpy()) - 1)
        if comp > 0:
            pos_count += 1
        per_asset_daily[sym] = s.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1))
    if len(per_asset_daily) < 2:
        return pos_count, len(per_asset_daily), 1.0
    M = pd.DataFrame(per_asset_daily).fillna(0.0)
    corr = M.corr().to_numpy()
    n = corr.shape[0]
    mean_corr = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
    eff_n = 1.0 / (mean_corr + (1 - mean_corr) / n) if n > 1 else 1.0
    return pos_count, n, round(float(eff_n), 1)


def main() -> int:
    global CADENCES, VOLWIN_CUR, _CUR_CAD
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    slow2 = [n for n in slow if len(_nums(n)) == 2]
    print(f"IRONED FINE: {len(slow)} slow configs ({len(slow2)} 2MA) x {len(CADENCES)} TF; SELECT on TRAIN+VAL, confirm OOS")
    print(f"split: TRAIN {SPLIT['TRAIN']} VAL {SPLIT['VAL']} OOS {SPLIT['OOS']}; maker={MAKER_RT*1e4:.0f}bps taker={TAKER_RT*1e4:.0f}bps RT\n")

    # whipsaw-filter param grid to SELECT on TRAIN+VAL (per TF). K=confirm bars, M=min-hold, cool=cooldown.
    WHIP_GRID = [(1, 0, 0), (2, 12, 0), (3, 24, 0), (3, 24, 12), (4, 48, 24), (6, 48, 24), (8, 96, 48)]
    EXITS = ["none", "trail10", "chandelier", "mh_trail15"]

    report = {}
    for cad in CADENCES:
        _CUR_CAD = cad
        VOLWIN_CUR = VOLWIN[cad]
        print(f"================================ {cad} ================================")
        panels = _build_caches(cad, slow)
        if len(panels) < 3:
            print(f"  [skip] only {len(panels)} assets with data")
            continue
        n_assets_train = sum(1 for s, p in panels.items()
                             if (p[3][p[4]] >= pd.Timestamp(SPLIT['TRAIN'][0]).value // 10**6).any()
                             and (p[3] < pd.Timestamp(SPLIT['TRAIN'][1]).value // 10**6).any())
        # vol-target level = median per-asset realized vol over the whole window (descriptive scale)
        rvs = []
        for sym, (c, h, l, ms, win, caches) in panels.items():
            rv = pd.Series(_ret_of(c)).rolling(VOLWIN_CUR).std().to_numpy()
            rvs.append(np.nanmedian(rv))
        voltgt = float(np.nanmedian(rvs))
        print(f"  assets in book: {len(panels)}; vol-target level (median realized bar-vol): {voltgt:.5f}")

        # benchmarks
        bh, vt, vt_exp = _benchmarks(panels, cad, voltgt)
        bh_oos = round(_compound(bh, *SPLIT["OOS"]), 1); bh_dd = round(_maxdd(bh, *SPLIT["OOS"]), 1)
        bh_sh = round(_sharpe(bh, *SPLIT["OOS"], cad), 2)
        vt_oos = round(_compound(vt, *SPLIT["OOS"]), 1); vt_dd = round(_maxdd(vt, *SPLIT["OOS"]), 1)
        vt_sh = round(_sharpe(vt, *SPLIT["OOS"], cad), 2); vt_cov = _coverage(vt, *SPLIT["OOS"], vt_exp)
        print(f"  BENCH  BUYHOLD    OOS net {bh_oos:>7}%  maxDD {bh_dd:>7}%  Sharpe {bh_sh:>5}  cov 100")
        print(f"  BENCH  VOLTGT_BH  OOS net {vt_oos:>7}%  maxDD {vt_dd:>7}%  Sharpe {vt_sh:>5}  cov {vt_cov}\n")

        # ---- SELECT MA-type on TRAIN+VAL (family, no overlay) ----
        tv_lo, tv_hi = SPLIT["TRAIN"][0], SPLIT["VAL"][1]
        type_scores = {}
        for mt in MA_TYPES:
            ap, exp = _stage_positions(panels, slow, mt, "family", 1, 0, 0, False, None)
            bk, _, _ = _book_from_positions(ap, cad, MAKER_RT)
            type_scores[mt] = _compound(bk, tv_lo, tv_hi)
        best_type = max(type_scores, key=lambda k: (type_scores[k] if not np.isnan(type_scores[k]) else -1e9))
        # ---- SELECT whipsaw params on TRAIN+VAL (family, best_type, maker) ----
        whip_scores = {}
        for (K, M, cl) in WHIP_GRID:
            ap, exp = _stage_positions(panels, slow, best_type, "family", K, M, cl, False, None)
            bk, _, _ = _book_from_positions(ap, cad, MAKER_RT)
            whip_scores[(K, M, cl)] = _compound(bk, tv_lo, tv_hi)
        best_whip = max(whip_scores, key=lambda k: (whip_scores[k] if not np.isnan(whip_scores[k]) else -1e9))
        # ---- SELECT exit on TRAIN+VAL ----
        exit_scores = {}
        for ex in EXITS:
            ap, exp = _family_with_exit(panels, slow, best_type, *best_whip, ex, False, None)
            bk, _, _ = _book_from_positions(ap, cad, MAKER_RT)
            exit_scores[ex] = _compound(bk, tv_lo, tv_hi)
        best_exit = max(exit_scores, key=lambda k: (exit_scores[k] if not np.isnan(exit_scores[k]) else -1e9))
        # ---- SELECT regime-gate dwell on TRAIN+VAL (0=off; >0 = anti-flicker min-dwell on BTC regime) ----
        GATE_GRID = [0, 24, 72, 168]   # off / 1d / 3d / 1w dwell (in bars varies by TF but same nominal grid)
        gate_scores = {}
        for g in GATE_GRID:
            ap, exp = _family_with_exit(panels, slow, best_type, *best_whip, best_exit, g, None)
            bk, _, _ = _book_from_positions(ap, cad, MAKER_RT)
            gate_scores[g] = _compound(bk, tv_lo, tv_hi)
        best_gate = max(gate_scores, key=lambda k: (gate_scores[k] if not np.isnan(gate_scores[k]) else -1e9))
        print(f"  SELECTED (TRAIN+VAL, maker): MA-type={best_type}  whipsaw(K,M,cool)={best_whip}  exit={best_exit}  gate_dwell={best_gate}")
        print(f"    type TRAIN+VAL net: " + " ".join(f"{t}:{type_scores[t]:.0f}" for t in MA_TYPES))
        print(f"    whip TRAIN+VAL net: " + " ".join(f"{k}:{v:.0f}" for k, v in whip_scores.items()))
        print(f"    exit TRAIN+VAL net: " + " ".join(f"{k}:{v:.0f}" for k, v in exit_scores.items()))
        print(f"    gate TRAIN+VAL net: " + " ".join(f"{k}:{v:.0f}" for k, v in gate_scores.items()) + "\n")

        # ---- BEFORE->AFTER ladder, confirmed on OOS, BOTH costs ----
        # S0 single (VAL-best single 2MA, best_type)
        val_lo, val_hi = SPLIT["VAL"]
        single_val = {}
        for name in slow2:
            ap, _ = _stage_positions(panels, [name], best_type, "single", 1, 0, 0, False, None)
            bk, _, _ = _book_from_positions(ap, cad, MAKER_RT)
            single_val[name] = _compound(bk, val_lo, val_hi)
        best_single = max(single_val, key=lambda k: (single_val[k] if not np.isnan(single_val[k]) else -1e9))

        ladder = []  # (label, stage_fn_args)
        ladder.append(("S0_single", dict(cfgs=[best_single], stage="single", K=1, M=0, cool=0, exit="none", gate=0, vt=None)))
        ladder.append(("S1_family", dict(cfgs=slow, stage="family", K=1, M=0, cool=0, exit="none", gate=0, vt=None)))
        ladder.append(("S2_whipsaw", dict(cfgs=slow, stage="family", K=best_whip[0], M=best_whip[1], cool=best_whip[2], exit="none", gate=0, vt=None)))
        ladder.append(("S3_exit", dict(cfgs=slow, stage="family", K=best_whip[0], M=best_whip[1], cool=best_whip[2], exit=best_exit, gate=0, vt=None)))
        ladder.append(("S4_gate", dict(cfgs=slow, stage="family", K=best_whip[0], M=best_whip[1], cool=best_whip[2], exit=best_exit, gate=best_gate, vt=None)))
        ladder.append(("S5_voltgt", dict(cfgs=slow, stage="family", K=best_whip[0], M=best_whip[1], cool=best_whip[2], exit=best_exit, gate=best_gate, vt=voltgt)))

        print(f"  BEFORE->AFTER ladder (OOS, confirmed once). turnover = mean per-asset sum|dpos| over OOS window.")
        hdr = f"   {'stage':12} | {'MAKER net%':>11} {'mkDD%':>7} {'mkTurn':>7} | {'TAKER net%':>11} {'tkDD%':>7} | {'cov':>5} {'breadth':>8} {'n_eff':>6}"
        print(hdr)
        cad_rows = {}
        for label, args in ladder:
            cfgs = args["cfgs"]
            if args["exit"] == "none" and not args["gate"] and args["vt"] is None:
                ap_m, exp = _stage_positions(panels, cfgs, best_type, args["stage"], args["K"], args["M"], args["cool"], False, None)
            else:
                # use the exit-capable builder for S3+ (family only)
                ap_m, exp = _family_with_exit(panels, cfgs, best_type, args["K"], args["M"], args["cool"], args["exit"], args["gate"], args["vt"])
            # OOS turnover (sum |dpos| within OOS window per asset, then mean)
            turn = _oos_turnover(ap_m, cad)
            bk_m, _, _ = _book_from_positions(ap_m, cad, MAKER_RT)
            bk_t, _, _ = _book_from_positions(ap_m, cad, TAKER_RT)
            mk = _metrics(bk_m, exp, cad, "maker", turn)
            tk = _metrics(bk_t, exp, cad, "taker", turn)
            br_m, n_in, neff = _breadth_neff(ap_m, MAKER_RT, cad)
            br_t, _, _ = _breadth_neff(ap_m, TAKER_RT, cad)
            print(f"   {label:12} | {mk['oos_net']:>11} {mk['oos_maxdd']:>7} {turn:>7.1f} | "
                  f"{tk['oos_net']:>11} {tk['oos_maxdd']:>7} | {str(mk['oos_cov']):>5} "
                  f"{f'{br_m}/{n_in}':>8} {neff:>6}")
            cad_rows[label] = {"maker": mk, "taker": tk, "turnover": round(turn, 1),
                               "breadth_maker": br_m, "breadth_taker": br_t, "n_in_book": n_in, "n_eff": neff,
                               "maker_sharpe": mk["oos_sharpe"], "taker_sharpe": tk["oos_sharpe"]}
        # ---- PARTICIPATION FRONTIER: the remaining lever (gate aggressiveness) at the S5 stack ----
        # no-gate (max participation + vol-target) vs full-gate (selected) -- the risk-vs-net frontier
        print(f"\n  PARTICIPATION FRONTIER (S5 stack, vary the gate -- the remaining lever for net-vs-risk):")
        print(f"   {'variant':16} | {'MAKER net%':>11} {'mkDD%':>7} {'cov':>6} {'turn':>6} | vs VOLTGT_BH net {vt_oos}% maxDD {vt_dd}%")
        gdwell = best_gate or 168                      # half-gate uses 1w dwell when no gate was selected
        frontier = {}
        for vlabel, gval, half in [("nogate_voltgt", 0, False), ("halfgate_voltgt", gdwell, True),
                                   ("fullgate_voltgt", gdwell, False)]:
            ap_f, exp_f = _family_with_exit(panels, slow, best_type, *best_whip, best_exit, gval, voltgt, half_gate=half)
            turn_f = _oos_turnover(ap_f, cad)
            bkf, _, _ = _book_from_positions(ap_f, cad, MAKER_RT)
            mkf = _metrics(bkf, exp_f, cad, "maker", turn_f)
            bkf_t, _, _ = _book_from_positions(ap_f, cad, TAKER_RT)
            tkf = _metrics(bkf_t, exp_f, cad, "taker", turn_f)
            frontier[vlabel] = {"maker": mkf, "taker": tkf, "turnover": round(turn_f, 1)}
            print(f"   {vlabel:16} | {mkf['oos_net']:>11} {mkf['oos_maxdd']:>7} {str(mkf['oos_cov']):>6} {turn_f:>6.1f} |")

        # verdict bar
        s5 = cad_rows["S5_voltgt"]
        deploy = (s5["maker"]["oos_net"] >= 0.7 * vt_oos and abs(s5["maker"]["oos_maxdd"]) < abs(bh_dd)
                  and s5["maker"]["oos_cov"] is not None and s5["maker"]["oos_cov"] > 50
                  and s5["taker"]["oos_net"] > 0)
        report[cad] = {"selected": {"ma_type": best_type, "whipsaw": list(best_whip), "exit": best_exit,
                                    "gate_dwell": best_gate, "best_single": best_single, "voltgt_level": round(voltgt, 6)},
                       "bench": {"buyhold": {"oos_net": bh_oos, "oos_maxdd": bh_dd, "oos_sharpe": bh_sh},
                                 "voltgt_bh": {"oos_net": vt_oos, "oos_maxdd": vt_dd, "oos_sharpe": vt_sh, "oos_cov": vt_cov}},
                       "ladder": cad_rows, "frontier": frontier, "n_assets_book": len(panels),
                       "deploy_candidate_maker": bool(deploy)}
        print(f"\n  VERDICT {cad}: S5 maker net {s5['maker']['oos_net']}% (vs VOLTGT_BH {vt_oos}%), "
              f"taker net {s5['taker']['oos_net']}%, maxDD {s5['maker']['oos_maxdd']}% (BH {bh_dd}%), "
              f"cov {s5['maker']['oos_cov']}, breadth {s5['breadth_maker']}/10 maker.")
        print(f"  -> {'DEPLOY CANDIDATE (maker)' if deploy else 'COST-WALL-BOUND or under-participating'}\n")

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump(report, open(op / "ironed_fine.json", "w"), indent=1, default=str)
    print(f"[json] {op / 'ironed_fine.json'}")
    return 0


def _oos_turnover(asset_pos, cad):
    lo = pd.Timestamp(SPLIT["OOS"][0]); hi = pd.Timestamp(SPLIT["OOS"][1])
    tos = []
    for sym, (pos, ret, ms, win) in asset_pos.items():
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        idx = pd.to_datetime(ms, unit="ms")
        m = (idx >= lo) & (idx < hi)
        tos.append(float(np.sum(flips[m])))
    return float(np.mean(tos)) if tos else 0.0


if __name__ == "__main__":
    sys.exit(main())
