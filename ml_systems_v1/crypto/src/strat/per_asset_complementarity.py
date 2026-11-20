"""src/strat/per_asset_complementarity.py -- FINAL fresh-axis stage: PER-ASSET complementarity.

THE NEW AXIS (vs the prior 8 stages, which tailored by TF/MA-type but ran a UNIFORM sleeve across all 10
assets): does tailoring the SLEEVE FAMILY per asset (trend-adaptive-MA vs mean-reversion-oscillator vs a
blend) beat one-size-fits-all? And is "across 10 it hits somewhere" (cross-asset breadth) empirically the
diversifier -- or is it the per-asset tailoring?

HONEST PRIOR (do NOT re-mine blindly): per-asset CONFIG DNA was NOISE in prior work (D62 + regime_dna; the
per-asset config selection did NOT transfer OOS). This tool tests a strictly COARSER cut -- the SLEEVE FAMILY
(trend vs MR vs blend), not the fine config -- and an asset-ARCHETYPE cut (trend-preferring vs MR-preferring),
to see whether a coarse per-asset choice transfers OOS or is ALSO noise (converging D62).

WHAT IT DOES (2020 BAND ONLY; u10; maker; causal):
  SPLIT: TRAIN+VAL = 2020-07-01..2020-10-01 (select), OOS = 2020-10-01..2021-01-01 (confirm). Selection NEVER
         sees OOS. The window is fenced; data outside 2020 is loaded but never scored.
  1. PER-ASSET CHARACTERIZATION: Hurst(VR), efficiency-ratio (ER), realized vol -- per asset over TRAIN+VAL;
     and which SLEEVE FAMILY wins per asset on TRAIN+VAL. Do assets SEPARATE into trend/MR archetypes or are
     they one BTC-beta cluster?
  2. PER-ASSET SLEEVE SELECTION: a book that picks the best sleeve family per asset on TRAIN+VAL, confirmed
     OOS, vs the UNIFORM books (all-trend / all-MR / all-50_50). MANDATORY CONTROL: vs RANDOM per-asset
     assignment of the SAME composition (does the SELECTION add skill, or would random asset->sleeve do as
     well -- the test that exposes per-asset DNA as noise).
  3. CROSS-ASSET COMPLEMENTARITY: cross-asset return correlation, n_eff (1/sum w_i^2 on equal weights via the
     corr matrix), breadth (fraction of assets engaged/winning per day), and the "across 10 it hits somewhere"
     gap-fill rate. Is breadth the real diversifier vs the per-asset tailoring?
  4. VERDICT (two-sided): per-asset family-tailoring beats uniform AND beats random-assignment OOS (real
     signal) -- OR it is noise (converging D62). Quantify whether cross-asset breadth is the actual diversifier.

CONSTRAINTS: 2020 band only; TRAIN+VAL-select / OOS-confirm; RANDOM-assignment control MANDATORY; claim-tag;
no emoji (cp1252); RWYB; bounded/efficient. Do NOT git commit.

RWYB: python -m strat.per_asset_complementarity --cadence 1d
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import MAKER_RT, apply_trail_stop
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel
from strat.data_expansion import block_bootstrap_distribution
from strat.deep2020_osc import _grid, _val, _mr_held

WIN = ("2020-07-01", "2021-01-01")     # 2020 BAND ONLY -- data outside is loaded but never scored
SEL_END = "2020-10-01"                  # TRAIN+VAL = WIN[0]..SEL_END ; OOS = SEL_END..WIN[1]
WARMUP = 400
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"


# -----------------------------------------------------------------------------------------------------
# PER-ASSET sleeve daily-net builders (the refactor: NOT u10-averaged -- one daily series per asset)
# -----------------------------------------------------------------------------------------------------
def _daily(net_bar: pd.Series) -> pd.Series:
    return net_bar.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def _trend_net_per_asset(cad):
    """{sym: daily net series} for the EMA slow-family TREND sleeve (equal-weight over slow configs)."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    s_ms = pd.Timestamp(WIN[0]).value // 10**6
    e_ms = pd.Timestamp(WIN[1]).value // 10**6
    out = {}
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
        uniq = sorted({p for n in slow for p in _nums(n)})
        cache = {p: _MA["EMA"](c2, p) for p in uniq}
        nets = []
        for name in slow:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
        idx = pd.to_datetime(ms2[win], unit="ms")
        out[sym] = _daily(pd.Series(np.mean(nets, axis=0), index=idx))
    return out


def _mr_net_per_asset(cad):
    """{sym: daily net series} for the equal-weight MR oscillator family sleeve."""
    grid = _grid()
    s_ms = pd.Timestamp(WIN[0]).value // 10**6
    e_ms = pd.Timestamp(WIN[1]).value // 10**6
    out = {}
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o2, h2, l2, c2, ms2 = o[s0:e], h[s0:e], l[s0:e], c[s0:e], ms[s0:e]
        if len(c2) < 40:
            continue
        win = ms2 >= s_ms
        if win.sum() < 30:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        idx = pd.to_datetime(ms2[win], unit="ms")
        nets = []
        for g in grid:
            kind, n, lo, hi = g
            v = _val(kind, c2, h2, l2, n)
            held = min_hold(_mr_held(v, lo, hi), 6).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
        out[sym] = _daily(pd.Series(np.mean(nets, axis=0), index=idx))
    return out


# -----------------------------------------------------------------------------------------------------
# Per-asset characterization (Hurst-VR + ER) on TRAIN+VAL price returns
# -----------------------------------------------------------------------------------------------------
def _hurst_vr(r, k=10):
    r = r[np.isfinite(r)]
    if len(r) < k * 4:
        return None
    v1 = np.var(r)
    rk = np.add.reduceat(r, np.arange(0, len(r) - len(r) % k, k))
    vk = np.var(rk)
    if v1 <= 0 or vk <= 0:
        return None
    return float(0.5 + 0.5 * np.log(vk / (k * v1)) / np.log(k))   # >0.5 trending, <0.5 MR


def _efficiency_ratio(c):
    """Kaufman ER over the whole TRAIN+VAL price path: |net move| / sum(|bar moves|). High = trendy."""
    if len(c) < 5:
        return None
    direction = abs(c[-1] - c[0])
    volatility = float(np.sum(np.abs(np.diff(c))))
    return float(direction / (volatility + 1e-12))


def _characterize(cad):
    """per asset over TRAIN+VAL price path: Hurst-VR (daily-bar returns), ER, ann-vol."""
    s_ms = pd.Timestamp(WIN[0]).value // 10**6
    sel_ms = pd.Timestamp(SEL_END).value // 10**6
    out = {}
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        m = (ms >= s_ms) & (ms < sel_ms)
        cc = c[m]
        if len(cc) < 40:
            out[sym] = None
            continue
        r = cc[1:] / cc[:-1] - 1.0
        # daily-bar Hurst on the cadence's own bars
        hv = _hurst_vr(r, k=max(2, min(10, len(r) // 8)))
        er = _efficiency_ratio(cc)
        vol = float(np.std(r))
        out[sym] = {"hurst_vr": round(hv, 3) if hv is not None else None,
                    "efficiency_ratio": round(er, 3) if er is not None else None,
                    "vol_per_bar": round(vol, 4), "n_bars": int(len(cc))}
    return out


# -----------------------------------------------------------------------------------------------------
# Perf helpers
# -----------------------------------------------------------------------------------------------------
def _perf(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    if len(x) < 3:
        return {"net": None, "sharpe": None, "maxdd": None, "n": len(x)}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1), "n": int(len(x))}


def _p05(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 5:
        return float("nan")
    bb = block_bootstrap_distribution(x, n_boot=600, block=5, seed=13)
    return round(bb["p05"] * 100, 1)


def _slice(s: pd.Series, lo, hi) -> pd.Series:
    return s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))]


def _book_from_assignment(assign, trend, mr, lo, hi):
    """Build a u10 equal-weight daily book given a {sym: 'trend'|'mr'|'blend'} assignment, over [lo,hi).
    Each asset contributes its assigned sleeve's daily net; book = cross-asset equal-weight mean per day."""
    cols = []
    for sym, fam in assign.items():
        if fam == "trend":
            s = trend.get(sym)
        elif fam == "mr":
            s = mr.get(sym)
        else:  # blend 50/50
            ts, ms_ = trend.get(sym), mr.get(sym)
            if ts is None or ms_ is None:
                s = ts if ts is not None else ms_
            else:
                j = pd.concat([ts.rename("t"), ms_.rename("m")], axis=1).dropna()
                s = pd.Series(0.5 * j["t"].to_numpy() + 0.5 * j["m"].to_numpy(), index=j.index)
        if s is None:
            continue
        cols.append(_slice(s, lo, hi).rename(sym))
    if not cols:
        return None
    df = pd.concat(cols, axis=1)
    return df.mean(axis=1, skipna=True), df


def main() -> int:
    cad = "1d"
    if "--cadence" in sys.argv:
        cad = sys.argv[sys.argv.index("--cadence") + 1]
    CHARTS.mkdir(parents=True, exist_ok=True)
    print(f"\n########## PER-ASSET COMPLEMENTARITY -- cadence={cad} (2020 band; u10; maker; causal) ##########")
    print(f"   SELECT on TRAIN+VAL [{WIN[0]}..{SEL_END}) ; CONFIRM on OOS [{SEL_END}..{WIN[1]})")

    trend = _trend_net_per_asset(cad)
    mr = _mr_net_per_asset(cad)
    chars = _characterize(cad)
    syms = [s for s in SYMS if s in trend and s in mr]
    print(f"   assets with both sleeves: {len(syms)} -> {syms}")

    # ---------- 1. PER-ASSET CHARACTERIZATION + which family wins on TRAIN+VAL ----------
    per_asset = {}
    for sym in syms:
        t_sel = _slice(trend[sym], WIN[0], SEL_END).to_numpy()
        m_sel = _slice(mr[sym], WIN[0], SEL_END).to_numpy()
        pt, pm = _perf(t_sel), _perf(m_sel)
        # winner by Sharpe on TRAIN+VAL, with a small margin so near-ties go to BLEND
        sh_t = pt["sharpe"] if pt["sharpe"] is not None else -9
        sh_m = pm["sharpe"] if pm["sharpe"] is not None else -9
        if abs(sh_t - sh_m) < 0.30:
            sel_family = "blend"
        else:
            sel_family = "trend" if sh_t > sh_m else "mr"
        ch = chars.get(sym) or {}
        per_asset[sym] = {
            "char": ch,
            "trainval_trend": {"net": pt["net"], "sharpe": pt["sharpe"]},
            "trainval_mr": {"net": pm["net"], "sharpe": pm["sharpe"]},
            "selected_family": sel_family,
        }
        print(f"   {sym:9} H={ch.get('hurst_vr')!s:>6} ER={ch.get('efficiency_ratio')!s:>6}  "
              f"TV trend Sh={sh_t:+.2f} / MR Sh={sh_m:+.2f}  -> pick {sel_family.upper()}")

    # archetype split: do assets separate? (count trend vs MR vs blend selections)
    fam_counts = {f: sum(1 for s in syms if per_asset[s]["selected_family"] == f) for f in ("trend", "mr", "blend")}
    hursts = [per_asset[s]["char"].get("hurst_vr") for s in syms if per_asset[s]["char"].get("hurst_vr") is not None]
    hurst_spread = round(float(np.std(hursts)), 3) if hursts else None
    print(f"\n   family selection counts (TRAIN+VAL): {fam_counts}  |  Hurst spread (std across assets) = {hurst_spread}")

    # ---------- 2. PER-ASSET SELECTION vs UNIFORM vs RANDOM ----------
    sel_assign = {s: per_asset[s]["selected_family"] for s in syms}
    uni_trend = {s: "trend" for s in syms}
    uni_mr = {s: "mr" for s in syms}
    uni_blend = {s: "blend" for s in syms}

    def book_perf(assign, tag):
        sel_book, _ = _book_from_assignment(assign, trend, mr, SEL_END, WIN[1])     # OOS
        if sel_book is None or len(sel_book) < 5:
            return None
        x = sel_book.to_numpy()
        p = _perf(x); p["p05"] = _p05(x)
        return p, sel_book

    res = {}
    for tag, assign in [("per_asset_selected", sel_assign), ("uniform_trend", uni_trend),
                        ("uniform_mr", uni_mr), ("uniform_blend", uni_blend)]:
        bp = book_perf(assign, tag)
        if bp:
            res[tag] = bp[0]
            res[tag + "__series"] = bp[1]

    # RANDOM-assignment control: SAME composition as the per-asset selection (preserve the count of each
    # family) but SHUFFLED across assets. If random does as well as the selection, the per-asset choice is noise.
    sel_families = [sel_assign[s] for s in syms]
    rng = np.random.default_rng(20200601)
    rand_nets = []
    N_RAND = 200
    for _ in range(N_RAND):
        perm = list(sel_families)
        rng.shuffle(perm)
        rassign = {s: perm[i] for i, s in enumerate(syms)}
        rb, _ = _book_from_assignment(rassign, trend, mr, SEL_END, WIN[1])
        if rb is not None and len(rb) >= 5:
            x = rb.to_numpy()
            eq = np.cumprod(1 + x)
            rand_nets.append(float((eq[-1] - 1) * 100))
    rand_nets = np.array(rand_nets)
    sel_net = res["per_asset_selected"]["net"]
    rand_mean = round(float(np.mean(rand_nets)), 1) if len(rand_nets) else None
    rand_p50 = round(float(np.median(rand_nets)), 1) if len(rand_nets) else None
    rand_p95 = round(float(np.percentile(rand_nets, 95)), 1) if len(rand_nets) else None
    # percentile of the actual selection within the random-assignment distribution
    sel_pctile = round(float(np.mean(rand_nets <= sel_net)) * 100, 1) if len(rand_nets) else None

    print(f"\n   OOS net%:  per-asset-SELECTED = {sel_net}  | uniform-trend = {res.get('uniform_trend',{}).get('net')}  "
          f"| uniform-MR = {res.get('uniform_mr',{}).get('net')}  | uniform-blend = {res.get('uniform_blend',{}).get('net')}")
    print(f"   RANDOM-assignment control (same composition, {len(rand_nets)} shuffles): mean={rand_mean} "
          f"p50={rand_p50} p95={rand_p95}  -> selection sits at {sel_pctile}th pctile of random")

    # ---------- 3. CROSS-ASSET COMPLEMENTARITY (breadth = the candidate real diversifier) ----------
    # Use the per-asset SELECTED sleeve's OOS daily series per asset (the deployed book composition).
    _, sel_panel = _book_from_assignment(sel_assign, trend, mr, SEL_END, WIN[1])
    panel = sel_panel.dropna(how="all")
    corr = panel.corr()
    # average pairwise correlation (off-diagonal)
    cm = corr.to_numpy()
    iu = np.triu_indices_from(cm, k=1)
    avg_pair_corr = round(float(np.nanmean(cm[iu])), 3)
    # n_eff via equal-weight portfolio variance vs avg single-asset variance:
    # n_eff = 1 / (avg_corr + (1-avg_corr)/N) is the classic eff-N for equal corr; report both that and the
    # eigenvalue participation ratio (PR = (sum lambda)^2 / sum lambda^2).
    N = panel.shape[1]
    n_eff_corr = round(float(1.0 / (avg_pair_corr + (1 - avg_pair_corr) / N)), 2) if N else None
    try:
        ev = np.linalg.eigvalsh(np.nan_to_num(cm, nan=0.0))
        ev = ev[ev > 0]
        pr = round(float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)), 2) if len(ev) else None
    except Exception:
        pr = None
    # breadth per day: fraction of assets with a POSITIVE return that day (among engaged)
    arr = panel.to_numpy()
    daily_n_engaged = np.sum(np.abs(np.nan_to_num(arr)) > 1e-9, axis=1)       # assets nonzero that day
    daily_n_winning = np.sum(np.nan_to_num(arr) > 0, axis=1)
    # "across 10 it hits somewhere": on days the BOOK is down, is at least one asset up?
    book_daily = np.nanmean(arr, axis=1)
    down_days = book_daily < 0
    hits_on_down = np.mean((daily_n_winning[down_days] >= 1)) if down_days.sum() else None
    breadth_mean = round(float(np.mean(daily_n_winning / np.maximum(daily_n_engaged, 1))), 3)
    print(f"\n   CROSS-ASSET: avg pairwise corr = {avg_pair_corr}  | n_eff(corr) = {n_eff_corr}  | "
          f"participation-ratio = {pr}  (N={N})")
    print(f"   breadth: mean fraction of engaged assets winning/day = {breadth_mean}  | "
          f"on book-DOWN days, >=1 asset up: {round(float(hits_on_down),3) if hits_on_down is not None else None}")

    # Is breadth the real diversifier? Compare the 10-asset book Sharpe vs the mean single-asset Sharpe.
    single_sharpes = []
    for sym in syms:
        s = _slice(trend[sym] if sel_assign[sym] == "trend" else mr[sym] if sel_assign[sym] == "mr"
                   else None, SEL_END, WIN[1])
        if sel_assign[sym] == "blend":
            j = pd.concat([_slice(trend[sym], SEL_END, WIN[1]).rename("t"),
                           _slice(mr[sym], SEL_END, WIN[1]).rename("m")], axis=1).dropna()
            s = pd.Series(0.5 * j["t"].to_numpy() + 0.5 * j["m"].to_numpy(), index=j.index)
        if s is not None and len(s) >= 5:
            single_sharpes.append(_perf(s.to_numpy())["sharpe"])
    mean_single_sharpe = round(float(np.nanmean(single_sharpes)), 2) if single_sharpes else None
    book_sharpe = res["per_asset_selected"]["sharpe"]
    diversification_gain = round(float(book_sharpe - mean_single_sharpe), 2) if (book_sharpe is not None and mean_single_sharpe is not None) else None
    print(f"   10-asset book Sharpe = {book_sharpe}  vs mean single-asset Sharpe = {mean_single_sharpe}  "
          f"-> breadth Sharpe-gain = {diversification_gain}")

    # ---------- 4. VERDICT ----------
    beats_uniform = all(sel_net is not None and res.get(u, {}).get("net") is not None and sel_net >= res[u]["net"]
                        for u in ("uniform_trend", "uniform_mr", "uniform_blend"))
    beats_best_uniform = sel_net is not None and sel_net >= max(
        [res.get(u, {}).get("net", -1e9) for u in ("uniform_trend", "uniform_mr", "uniform_blend")])
    beats_random = sel_pctile is not None and sel_pctile >= 90.0    # selection in top decile of random
    per_asset_real = bool(beats_best_uniform and beats_random)
    if per_asset_real:
        verdict = ("PER-ASSET family-tailoring is REAL: beats best uniform AND sits >=90th pctile of "
                   "random-assignment OOS")
    elif beats_best_uniform and not beats_random:
        verdict = ("per-asset tailoring beats uniform numerically BUT NOT random-assignment -> the SELECTION "
                   "is NOISE (random asset->family does as well); converges D62")
    else:
        verdict = ("per-asset tailoring does NOT beat uniform OOS -> NULL; the uniform book (one sleeve on all) "
                   "is as good or better")
    breadth_is_diversifier = (diversification_gain is not None and diversification_gain > 0.2
                              and avg_pair_corr < 0.85)
    print(f"\n   VERDICT: {verdict}")
    print(f"   breadth-is-the-real-diversifier: {breadth_is_diversifier} "
          f"(Sharpe-gain {diversification_gain} at avg corr {avg_pair_corr})")

    export = {
        "_meta": {"cadence": cad, "window": WIN, "select_window": [WIN[0], SEL_END], "oos_window": [SEL_END, WIN[1]],
                  "universe": "u10", "cost": "maker", "constraint": "2020 BAND ONLY",
                  "claim_tag": "2020-OOS bull-only; coarse SLEEVE-FAMILY cut (not fine config, D62); "
                               "TRAIN+VAL-select/OOS-confirm; random-assignment control MANDATORY",
                  "n_assets": len(syms), "syms": syms},
        "1_characterization": {s: per_asset[s] for s in syms},
        "archetype_split": {"family_selection_counts": fam_counts, "hurst_spread_across_assets": hurst_spread},
        "2_selection_vs_uniform": {
            "per_asset_selected": res.get("per_asset_selected"),
            "uniform_trend": res.get("uniform_trend"),
            "uniform_mr": res.get("uniform_mr"),
            "uniform_blend": res.get("uniform_blend"),
            "random_assignment_control": {"n_shuffles": int(len(rand_nets)), "mean_net": rand_mean,
                                          "p50_net": rand_p50, "p95_net": rand_p95,
                                          "selection_net": sel_net, "selection_pctile_in_random": sel_pctile},
            "beats_best_uniform": bool(beats_best_uniform), "beats_random": bool(beats_random),
        },
        "3_cross_asset_complementarity": {
            "avg_pairwise_corr": avg_pair_corr, "n_eff_corr": n_eff_corr, "participation_ratio": pr, "N": int(N),
            "breadth_mean_winning_frac": breadth_mean,
            "on_down_days_at_least_one_up": round(float(hits_on_down), 3) if hits_on_down is not None else None,
            "book_sharpe": book_sharpe, "mean_single_asset_sharpe": mean_single_sharpe,
            "breadth_sharpe_gain": diversification_gain,
        },
        "4_verdict": {"per_asset_tailoring_real": per_asset_real, "verdict": verdict,
                      "breadth_is_real_diversifier": bool(breadth_is_diversifier)},
    }
    jpath = OUT / "per_asset_complementarity.json"
    json.dump(export, open(jpath, "w"), indent=1, default=str)
    print(f"\n[json] {jpath}")

    # ---------- CHARTS ----------
    _chart_archetype(per_asset, syms, cad)
    _chart_selection(res, rand_nets, sel_net, cad)
    return 0


# -----------------------------------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------------------------------
def _chart_archetype(per_asset, syms, cad):
    """each asset's trend-vs-chop tendency (Hurst on x, ER on y) colored by which sleeve wins TRAIN+VAL."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = {"trend": "#1f77b4", "mr": "#ff7f0e", "blend": "#2ca02c"}
    for sym in syms:
        ch = per_asset[sym]["char"] or {}
        hv, er = ch.get("hurst_vr"), ch.get("efficiency_ratio")
        fam = per_asset[sym]["selected_family"]
        if hv is None or er is None:
            continue
        ax1.scatter(hv, er, color=cmap[fam], s=90, edgecolor="k", zorder=3)
        ax1.annotate(sym.replace("USDT", ""), (hv, er), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax1.axvline(0.5, color="k", ls="--", lw=0.8, label="Hurst 0.5 (trend|MR boundary)")
    ax1.set_xlabel("Hurst (VR)  >0.5 trending / <0.5 mean-reverting")
    ax1.set_ylabel("efficiency ratio  (high = trendy)")
    ax1.set_title(f"Per-asset archetype ({cad}, TRAIN+VAL 2020)\ncolor = sleeve that WINS TRAIN+VAL")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=cmap[k], label=f"{k} wins") for k in cmap]
               + [plt.Line2D([0], [0], color="k", ls="--", label="Hurst 0.5")], fontsize=8, loc="best")
    ax1.grid(alpha=0.25)
    # right: per-asset trend vs MR Sharpe (TRAIN+VAL) -- do they separate?
    x = np.arange(len(syms))
    sh_t = [per_asset[s]["trainval_trend"]["sharpe"] or 0 for s in syms]
    sh_m = [per_asset[s]["trainval_mr"]["sharpe"] or 0 for s in syms]
    w = 0.38
    ax2.bar(x - w / 2, sh_t, w, color="#1f77b4", label="TREND Sharpe (TV)")
    ax2.bar(x + w / 2, sh_m, w, color="#ff7f0e", label="MR Sharpe (TV)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels([s.replace("USDT", "") for s in syms], rotation=40, fontsize=8)
    ax2.set_ylabel("Sharpe (TRAIN+VAL)"); ax2.set_title("Per-asset TREND vs MR Sharpe (the selection signal)")
    ax2.legend(fontsize=8)
    fig.suptitle("PER-ASSET ARCHETYPE -- do assets separate into trend vs MR? (2020 band, u10)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = CHARTS / "per_asset_archetype.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def _chart_selection(res, rand_nets, sel_net, cad):
    """per-asset-tailored book vs uniform vs random-assignment (OOS net%)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # left: OOS equity of the books
    colors = {"per_asset_selected": "#2ca02c", "uniform_trend": "#1f77b4", "uniform_mr": "#ff7f0e",
              "uniform_blend": "#9467bd"}
    for tag, col in colors.items():
        s = res.get(tag + "__series")
        if s is None:
            continue
        eq = (np.cumprod(1 + s.to_numpy()) - 1) * 100
        ax1.plot(s.index, eq, color=col, lw=(2.4 if tag == "per_asset_selected" else 1.4),
                 label=f"{tag} ({res[tag]['net']}%)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("OOS compound %"); ax1.set_title(f"Per-asset selected vs uniform books ({cad}, 2020 OOS)")
    ax1.legend(fontsize=8); ax1.tick_params(axis="x", labelrotation=30, labelsize=7)
    # right: random-assignment distribution with the selection marked
    if len(rand_nets):
        ax2.hist(rand_nets, bins=30, color="#bbbbbb", edgecolor="#888", label="random asset->family (same composition)")
        ax2.axvline(sel_net, color="#2ca02c", lw=2.5, label=f"per-asset SELECTED ({sel_net}%)")
        ax2.axvline(float(np.median(rand_nets)), color="k", ls="--", lw=1.0, label=f"random median ({np.median(rand_nets):.1f}%)")
        pct = float(np.mean(rand_nets <= sel_net)) * 100
        ax2.annotate(f"selection at {pct:.0f}th pctile\nof random-assignment\n(need >=90th to be REAL)",
                     xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=9,
                     bbox=dict(boxstyle="round", fc="#fffbe6", ec="#999"))
    ax2.set_xlabel("OOS net %"); ax2.set_ylabel("count")
    ax2.set_title("MANDATORY control: selection vs RANDOM assignment\n(in-distribution => per-asset DNA is noise)")
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("PER-ASSET TAILORING vs UNIFORM vs RANDOM (the noise test) -- 2020 OOS, u10, maker", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = CHARTS / "per_asset_vs_uniform_vs_random.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


if __name__ == "__main__":
    sys.exit(main())
