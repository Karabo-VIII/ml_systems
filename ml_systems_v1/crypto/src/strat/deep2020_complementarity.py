"""src/strat/deep2020_complementarity.py -- PHASE 1b: the TREND<->MR COMPLEMENTARITY foundation.

User CORE vision (verbatim): "if one is out one day, the other is on, and is able to capture the missing
gaps for the other, ideally per timeframe." This tool extends the orthogonal MR oscillator family
(OSCILLATORS.md found RSI/Stoch/BB%b/CCI orthogonal to the MA trend book, corr +0.28-0.31) across ALL
finer TFs {1d,4h,2h,1h,30m,15m} and MEASURES trend-vs-MR complementarity + GAP-FILLING per timeframe.

WHAT IT PRODUCES (per TF, OOS Oct-Dec 2020, u10, maker, causal):
  - the MR oscillator family sleeve (equal-weight RSI/Stoch/BB%b/CCI MR configs) -- reuses deep2020_osc.
  - the TREND MA sleeve (EMA slow family) -- reuses deep2020_osc._ma_book mechanics, but instrumented to
    also emit per-day EXPOSURE (was the sleeve engaged?), which the original return-only book did not.
  - COMPLEMENTARITY metrics:
      (1) corr(trend, MR) on daily net returns + within-family corr (the orthogonality baseline).
      (2) THE GAP-FILLING metric (the user's exact ask): on days the TREND sleeve is FLAT (exposure ~0)
          or DOWN (net<0), the MR sleeve's hit-rate + mean return -- and vice-versa. The conditional
          "gap-fill rate" = P(other sleeve POSITIVE | this sleeve is out/down), per TF.
      (3) coverage union: fraction of OOS days with >=1 sleeve engaged.
      (4) the optimal static blend: 50/50 + min-variance + risk-parity weights per TF.
  - THE COMBINED COMPLEMENTARY BOOK per TF vs the best single sleeve: net, Sharpe, maxDD, coverage,
    p05 (block-bootstrap). Does combining improve risk-adjusted return at EVERY TF or only some?
  - THE COMPLEMENTARITY SCORE per TF (a single number): (1 - corr) weighted by the symmetric gap-fill rate.

CONSTRAINTS (user mandate): 2020 BAND ONLY (window-fenced to 2020-07-01..2021-01-01, OOS = 2020-10-01).
DO NOT touch 2026/other data. Synthetic only for nulls (not used here). Charts via matplotlib (Agg). No
emoji (cp1252). RWYB. Do NOT git commit.

HONEST: MR standalone is WEAKER than trend (crypto MR is hard, dead-list D37) -- it is a DIVERSIFIER not a
primary edge; the value is the COMBINATION. Two-sided: TFs where complementarity does NOT help are reported.
2020-bull-only limitation flagged (synthetic regime-stress is a later phase).

RWYB: python -m strat.deep2020_complementarity --cadences 1d,4h,2h,1h,30m,15m
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

from strat.portfolio_replay import MAKER_RT, apply_trail_stop
from strat.replay_distinct_grid import distinct_specs
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import _MA, _nums
from strat.ma_2020_breakdown import _panel
from strat.data_expansion import block_bootstrap_distribution
import strat.portfolio_replay as PR
from strat.deep2020_osc import _grid, _val, _mr_held

WIN = ("2020-07-01", "2021-01-01")          # 2020 BAND ONLY -- data outside is loaded but never scored
SPLIT = "2020-10-01"                         # OOS = Oct-Dec 2020
WARMUP = 400
ALL_CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"


# -----------------------------------------------------------------------------------------------------
# Sleeve builders -- return BOTH a daily net-return book AND a daily EXPOSURE book (engaged fraction).
# Exposure = per-day mean over u10 of (was-this-asset-in-a-position-this-bar), resampled to daily MAX
# (engaged at all that day). The original return-only books could not answer "is the sleeve out today?".
# -----------------------------------------------------------------------------------------------------
def _daily_compound(net_series: pd.Series) -> pd.Series:
    return net_series.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def _daily_exposure(exp_series: pd.Series) -> pd.Series:
    # a day "engaged" if the sleeve held any position during it -> take the daily MAX of bar exposure
    return exp_series.resample("1D").max().dropna()


def _mr_sleeve(cad):
    """equal-weight MR oscillator family sleeve. Returns (daily_net, daily_exposure) over OOS window."""
    s_ms = pd.Timestamp(WIN[0]).value // 10**6
    e_ms = pd.Timestamp(WIN[1]).value // 10**6
    grid = _grid()
    per_sym_net = []          # per asset: bar net averaged over all configs
    per_sym_exp = []          # per asset: bar exposure (held) averaged over all configs
    idx_ref = None
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
        cfg_nets, cfg_exps = [], []
        for g in grid:
            kind, n, lo, hi = g
            v = _val(kind, c2, h2, l2, n)
            held = min_hold(_mr_held(v, lo, hi), 6).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            cfg_nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
            cfg_exps.append(pos[win])                       # was this config in a position this bar?
        per_sym_net.append(pd.Series(np.mean(cfg_nets, axis=0), index=idx))
        per_sym_exp.append(pd.Series(np.mean(cfg_exps, axis=0), index=idx))
    if not per_sym_net:
        return None, None
    net_bar = pd.concat(per_sym_net, axis=1).mean(axis=1, skipna=True)
    exp_bar = pd.concat(per_sym_exp, axis=1).mean(axis=1, skipna=True)   # u10 average engaged fraction
    return _daily_compound(net_bar), _daily_exposure(exp_bar)


def _trend_sleeve(cad):
    """EMA slow-family trend sleeve (the reference book). Returns (daily_net, daily_exposure) over OOS."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=40))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    s_ms = pd.Timestamp(WIN[0]).value // 10**6
    e_ms = pd.Timestamp(WIN[1]).value // 10**6
    per_sym_net, per_sym_exp = [], []
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
        cfg_nets, cfg_exps = [], []
        for name in slow:
            pp = _nums(name); mas = [cache[p] for p in pp]
            h0 = np.nan_to_num((mas[0] > mas[1]) if len(pp) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))).astype(np.int8)
            h0 = min_hold(apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8), 12).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = h0[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            cfg_nets.append((pos * ret - flips * (MAKER_RT / 2.0))[win])
            cfg_exps.append(pos[win])
        idx = pd.to_datetime(ms2[win], unit="ms")
        per_sym_net.append(pd.Series(np.mean(cfg_nets, axis=0), index=idx))
        per_sym_exp.append(pd.Series(np.mean(cfg_exps, axis=0), index=idx))
    if not per_sym_net:
        return None, None
    net_bar = pd.concat(per_sym_net, axis=1).mean(axis=1, skipna=True)
    exp_bar = pd.concat(per_sym_exp, axis=1).mean(axis=1, skipna=True)
    return _daily_compound(net_bar), _daily_exposure(exp_bar)


# -----------------------------------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------------------------------
def _oos(s: pd.Series) -> pd.Series:
    return s[s.index >= pd.Timestamp(SPLIT)]


def _perf(x: np.ndarray) -> dict:
    if len(x) < 3:
        return {"net": None, "sharpe": None, "maxdd": None}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(365)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1)}


def _p05(x: np.ndarray) -> float:
    if len(x) < 5:
        return float("nan")
    bb = block_bootstrap_distribution(x, n_boot=600, block=5, seed=13)
    return round(bb["p05"] * 100, 1)


def _gap_fill(this_net, this_exp, other_net, EXP_FLAT=0.10):
    """THE GAP-FILLING metric. 'this sleeve is OUT/DOWN' = exposure<=EXP_FLAT (flat) OR net<0 (down).
    On those days, what does the OTHER sleeve do? Returns hit-rate (P(other>0)) + mean other return."""
    df = pd.concat([this_net.rename("tn"), this_exp.rename("te"), other_net.rename("on")], axis=1).dropna()
    if len(df) < 5:
        return None
    out_mask = (df["te"].to_numpy() <= EXP_FLAT) | (df["tn"].to_numpy() < 0)   # this sleeve is out OR down
    n_out = int(out_mask.sum())
    if n_out == 0:
        return {"n_gap_days": 0, "gap_fill_rate": None, "gap_fill_mean_ret_pct": None,
                "gap_frac_of_oos": 0.0}
    other_on_gap = df["on"].to_numpy()[out_mask]
    fill_rate = float(np.mean(other_on_gap > 0))             # P(other sleeve POSITIVE on this sleeve's gap days)
    fill_mean = float(np.mean(other_on_gap) * 100)
    return {"n_gap_days": n_out, "gap_fill_rate": round(fill_rate, 3),
            "gap_fill_mean_ret_pct": round(fill_mean, 3),
            "gap_frac_of_oos": round(n_out / len(df), 3)}


def _coverage_union(trend_exp, mr_exp, EXP_FLAT=0.10):
    df = pd.concat([trend_exp.rename("t"), mr_exp.rename("m")], axis=1).dropna()
    if len(df) == 0:
        return None
    engaged = (df["t"].to_numpy() > EXP_FLAT) | (df["m"].to_numpy() > EXP_FLAT)
    only_t = (df["t"].to_numpy() > EXP_FLAT) & ~(df["m"].to_numpy() > EXP_FLAT)
    only_m = (df["m"].to_numpy() > EXP_FLAT) & ~(df["t"].to_numpy() > EXP_FLAT)
    both = (df["t"].to_numpy() > EXP_FLAT) & (df["m"].to_numpy() > EXP_FLAT)
    none = ~engaged
    n = len(df)
    return {"coverage_union": round(float(engaged.mean()), 3),
            "only_trend": round(float(only_t.mean()), 3), "only_mr": round(float(only_m.mean()), 3),
            "both": round(float(both.mean()), 3), "neither": round(float(none.mean()), 3), "n_days": n}


def _blends(trend_net, mr_net):
    """static blend weights: 50/50, min-variance, risk-parity (inverse-vol). Returns blended daily series."""
    df = pd.concat([trend_net.rename("t"), mr_net.rename("m")], axis=1).dropna()
    t = df["t"].to_numpy(); m = df["m"].to_numpy()
    blends = {"50_50": 0.5}
    # min-variance long-only weight on trend (closed form, 2-asset): w_t = (v_m - cov)/(v_t+v_m-2cov), clipped
    vt, vm = np.var(t), np.var(m); cov = np.cov(t, m)[0, 1]
    denom = vt + vm - 2 * cov
    w_mv = (vm - cov) / denom if abs(denom) > 1e-18 else 0.5
    w_mv = float(np.clip(w_mv, 0.0, 1.0))
    # risk-parity (inverse-vol)
    st, sm = np.std(t), np.std(m)
    w_rp = float(sm / (st + sm)) if (st + sm) > 1e-18 else 0.5   # inverse-vol weight to trend
    weights = {"50_50": 0.5, "min_var": w_mv, "risk_parity": w_rp}
    series = {name: pd.Series(w * t + (1 - w) * m, index=df.index) for name, w in weights.items()}
    return weights, series, df


def main() -> int:
    cadences = ALL_CADENCES
    if "--cadences" in sys.argv:
        cadences = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    CHARTS.mkdir(parents=True, exist_ok=True)

    export = {"_meta": {"window": WIN, "oos_split": SPLIT, "universe": "u10", "cost": "maker",
                        "constraint": "2020 BAND ONLY", "exp_flat_threshold": 0.10,
                        "claim_tag": "in-sample 2020-OOS, bull-only; MR standalone WEAK (D37) -- value is the COMBINATION"}}
    equity_data = {}      # cad -> {trend, mr, combined daily oos cumulative}
    corr_data = {}        # cad -> {trend_mr corr}

    for cad in cadences:
        print(f"\n########## {cad} -- TREND<->MR COMPLEMENTARITY (OOS Oct-Dec 2020, u10, maker) ##########")
        tnet, texp = _trend_sleeve(cad)
        mnet, mexp = _mr_sleeve(cad)
        if tnet is None or mnet is None:
            print(f"   [skip] {cad}: insufficient data")
            continue
        tnet, texp = _oos(tnet), _oos(texp)
        mnet, mexp = _oos(mnet), _oos(mexp)

        # align
        df = pd.concat([tnet.rename("t"), mnet.rename("m")], axis=1).dropna()
        if len(df) < 10:
            print(f"   [skip] {cad}: <10 aligned OOS days")
            continue
        t_arr, m_arr = df["t"].to_numpy(), df["m"].to_numpy()

        # (1) correlation
        corr = float(df["t"].corr(df["m"]))
        corr_data[cad] = corr

        # (2) gap-filling (BOTH directions)
        gf_mr_fills_trend = _gap_fill(tnet, texp, mnet)   # trend out/down -> does MR fill?
        gf_trend_fills_mr = _gap_fill(mnet, mexp, tnet)   # MR out/down -> does trend fill?

        # (3) coverage union
        cov = _coverage_union(texp, mexp)

        # (4) blends
        weights, blend_series, _ = _blends(tnet, mnet)

        # sleeve + combined perf
        p_trend = _perf(t_arr); p_mr = _perf(m_arr)
        perf_blends = {}
        for name, s in blend_series.items():
            x = s.to_numpy()
            pm = _perf(x); pm["p05"] = _p05(x)
            perf_blends[name] = pm
        p_trend["p05"] = _p05(t_arr); p_mr["p05"] = _p05(m_arr)

        # best single sleeve (by Sharpe)
        best_single = "trend" if (p_trend["sharpe"] or -9) >= (p_mr["sharpe"] or -9) else "mr"
        best_single_perf = p_trend if best_single == "trend" else p_mr
        # best blend (by Sharpe)
        best_blend_name = max(perf_blends, key=lambda k: perf_blends[k]["sharpe"] or -9)
        best_blend_perf = perf_blends[best_blend_name]

        # (5) COMPLEMENTARITY SCORE: (1-corr) weighted by the symmetric gap-fill rate
        gfr_a = (gf_mr_fills_trend or {}).get("gap_fill_rate") or 0.0
        gfr_b = (gf_trend_fills_mr or {}).get("gap_fill_rate") or 0.0
        sym_gap_fill = 0.5 * (gfr_a + gfr_b)
        comp_score = round((1.0 - corr) * sym_gap_fill, 3)

        # does combining help risk-adjusted return vs the BEST single sleeve?
        helps = (best_blend_perf["sharpe"] is not None and best_single_perf["sharpe"] is not None
                 and best_blend_perf["sharpe"] > best_single_perf["sharpe"]
                 and best_blend_perf["maxdd"] >= best_single_perf["maxdd"])   # higher Sharpe AND not-worse DD
        helps_dd_only = (best_blend_perf["maxdd"] > best_single_perf["maxdd"]
                         and best_blend_perf["sharpe"] is not None
                         and best_blend_perf["sharpe"] >= best_single_perf["sharpe"] - 0.15)

        # ---- print ----
        print(f"   {'sleeve':16} {'net%':>7} {'Sharpe':>7} {'maxDD%':>7} {'p05%':>7}")
        print(f"   {'TREND (MA)':16} {p_trend['net']:>7} {p_trend['sharpe']:>7} {p_trend['maxdd']:>7} {p_trend['p05']:>7}")
        print(f"   {'MR (osc)':16} {p_mr['net']:>7} {p_mr['sharpe']:>7} {p_mr['maxdd']:>7} {p_mr['p05']:>7}")
        for name in ("50_50", "min_var", "risk_parity"):
            pm = perf_blends[name]
            print(f"   {('blend '+name):16} {pm['net']:>7} {pm['sharpe']:>7} {pm['maxdd']:>7} {pm['p05']:>7}  (w_trend={weights[name]:.2f})")
        print(f"   corr(trend, MR) = {corr:+.2f}  ({'ORTHOGONAL' if corr < 0.4 else 'correlated'})")
        if gf_mr_fills_trend:
            print(f"   GAP-FILL: trend OUT/DOWN {gf_mr_fills_trend['n_gap_days']}d ({gf_mr_fills_trend['gap_frac_of_oos']:.0%} of OOS) "
                  f"-> MR fills: hit {gf_mr_fills_trend['gap_fill_rate']:.0%}, mean {gf_mr_fills_trend['gap_fill_mean_ret_pct']:+.2f}%/d")
        if gf_trend_fills_mr:
            print(f"   GAP-FILL: MR OUT/DOWN {gf_trend_fills_mr['n_gap_days']}d ({gf_trend_fills_mr['gap_frac_of_oos']:.0%} of OOS) "
                  f"-> trend fills: hit {gf_trend_fills_mr['gap_fill_rate']:.0%}, mean {gf_trend_fills_mr['gap_fill_mean_ret_pct']:+.2f}%/d")
        if cov:
            print(f"   coverage union = {cov['coverage_union']:.0%}  (only-trend {cov['only_trend']:.0%}, only-MR {cov['only_mr']:.0%}, "
                  f"both {cov['both']:.0%}, neither {cov['neither']:.0%})")
        print(f"   COMPLEMENTARITY SCORE = {comp_score}  [(1-corr)*sym_gap_fill, sym_gap_fill={sym_gap_fill:.2f}]")
        verdict = ("COMBINING HELPS (Sharpe up, DD not worse)" if helps
                   else "DD-only improvement" if helps_dd_only
                   else "combining does NOT improve risk-adjusted return")
        print(f"   best single = {best_single} (Sh {best_single_perf['sharpe']}); best blend = {best_blend_name} "
              f"(Sh {best_blend_perf['sharpe']}, DD {best_blend_perf['maxdd']}) -> {verdict}")

        export[cad] = {
            "corr_trend_mr": round(corr, 3),
            "trend": p_trend, "mr": p_mr, "blends": perf_blends, "blend_weights": {k: round(v, 3) for k, v in weights.items()},
            "gap_fill_mr_fills_trend": gf_mr_fills_trend, "gap_fill_trend_fills_mr": gf_trend_fills_mr,
            "coverage": cov, "complementarity_score": comp_score, "sym_gap_fill_rate": round(sym_gap_fill, 3),
            "best_single": best_single, "best_blend": best_blend_name,
            "combining_helps": bool(helps), "combining_helps_dd_only": bool(helps_dd_only), "verdict": verdict,
            "n_oos_days": len(df),
        }
        # stash equity for charts (50/50 = the canonical combined book)
        equity_data[cad] = {
            "trend": np.cumprod(1 + t_arr), "mr": np.cumprod(1 + m_arr),
            "combined": np.cumprod(1 + blend_series["50_50"].to_numpy()),
            "dates": df.index, "trend_exp": texp.reindex(df.index).to_numpy(),
            "mr_exp": mexp.reindex(df.index).to_numpy(),
            "trend_net": t_arr, "mr_net": m_arr,
        }

    # rank TFs by complementarity score
    ranked = sorted([c for c in export if c != "_meta"], key=lambda c: -(export[c]["complementarity_score"] or -9))
    export["_ranking"] = {"by_complementarity_score": [(c, export[c]["complementarity_score"]) for c in ranked]}
    print("\n## TF ranking by COMPLEMENTARITY SCORE (best pair-filling first)")
    for c in ranked:
        e = export[c]
        print(f"   {c:5} score={e['complementarity_score']:>5}  corr={e['corr_trend_mr']:+.2f}  "
              f"sym_gap_fill={e['sym_gap_fill_rate']:.2f}  verdict={e['verdict']}")

    # ---- CHARTS ----
    _chart_corr_heatmap(export, cadences)
    if ranked:
        _chart_gap_timeline(equity_data.get(ranked[0]), ranked[0])
    _chart_combined_vs_sleeves(equity_data, [c for c in cadences if c in equity_data])

    # ---- JSON ----
    jpath = OUT / "complementarity_matrix.json"
    json.dump(export, open(jpath, "w"), indent=1, default=str)
    print(f"\n[json] {jpath}")
    return 0


# -----------------------------------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------------------------------
def _chart_corr_heatmap(export, cadences):
    """corr(trend,MR) across TFs + a reference band for within-family corr (~0.85-0.94 trend-trend)."""
    cs = [c for c in cadences if c in export]
    if not cs:
        return
    corrs = [export[c]["corr_trend_mr"] for c in cs]
    scores = [export[c]["complementarity_score"] for c in cs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: corr(trend,MR) bar with within-family reference band
    x = np.arange(len(cs))
    bars = ax1.bar(x, corrs, color="#1f77b4", label="corr(trend, MR) cross-family")
    ax1.axhspan(0.85, 0.94, color="#d62728", alpha=0.18, label="within-trend-family corr (no diversification)")
    ax1.axhline(0.40, color="k", ls="--", lw=0.8, label="orthogonality line (0.40)")
    for xi, v in zip(x, corrs):
        ax1.annotate(f"{v:+.2f}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(cs); ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("correlation"); ax1.set_title("Cross-family corr(trend, MR) per TF\n(low = orthogonal = diversifies)")
    ax1.legend(fontsize=8, loc="upper right")
    # right: complementarity score
    bars2 = ax2.bar(x, scores, color="#2ca02c")
    for xi, v in zip(x, scores):
        ax2.annotate(f"{v}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(cs)
    ax2.set_ylabel("complementarity score  (1-corr) x sym_gap_fill")
    ax2.set_title("COMPLEMENTARITY SCORE per TF\n(higher = pair fills each other's gaps better)")
    fig.suptitle("TREND<->MR complementarity across timeframes (2020 OOS, u10, maker) -- 2020-bull-only", fontsize=11)
    fig.tight_layout()
    p = CHARTS / "complementarity_corr_heatmap.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def _chart_gap_timeline(ed, cad):
    """for the best-complementarity TF: trend vs MR exposure over OOS, shading the gap-fill days."""
    if ed is None:
        return
    dates = ed["dates"]
    tn, mn = ed["trend_net"], ed["mr_net"]
    te, me = ed["trend_exp"], ed["mr_exp"]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    # top: exposure of each sleeve
    axA.fill_between(dates, 0, te, color="#1f77b4", alpha=0.45, label="TREND exposure")
    axA.fill_between(dates, 0, -me, color="#ff7f0e", alpha=0.45, label="MR exposure")
    axA.axhline(0, color="k", lw=0.6)
    axA.set_ylabel("sleeve exposure\n(trend up / MR down)")
    axA.set_title(f"GAP-FILLING timeline -- best-complementarity TF = {cad} (2020 OOS)\n"
                  "shaded = a day one sleeve is OUT/DOWN and the OTHER captures (the user's vision, visualized)")
    axA.legend(fontsize=8, loc="upper left")
    # shade gap-fill days: trend out/down AND MR positive (green); MR out/down AND trend positive (purple)
    trend_out = (te <= 0.10) | (tn < 0)
    mr_out = (me <= 0.10) | (mn < 0)
    mr_fills = trend_out & (mn > 0)
    trend_fills = mr_out & (tn > 0)
    for i, d in enumerate(dates):
        if mr_fills[i]:
            axA.axvspan(d, d, color="#2ca02c", alpha=0.0)   # placeholder; real shading below on axB
    # bottom: daily net of each sleeve + the combined, with gap-fill markers
    axB.bar(dates, tn * 100, width=0.8, color="#1f77b4", alpha=0.6, label="trend daily net %")
    axB.bar(dates, mn * 100, width=0.8, color="#ff7f0e", alpha=0.6, label="MR daily net %")
    # markers where the other sleeve fills the gap
    yb = (np.nanmin(np.concatenate([tn, mn])) * 100) - 0.3
    axB.scatter(dates[mr_fills], np.full(int(mr_fills.sum()), yb), marker="^", color="#2ca02c", s=22,
                label="MR fills trend gap", zorder=5)
    axB.scatter(dates[trend_fills], np.full(int(trend_fills.sum()), yb - 0.4), marker="v", color="#9467bd", s=22,
                label="trend fills MR gap", zorder=5)
    axB.axhline(0, color="k", lw=0.6)
    axB.set_ylabel("daily net %"); axB.legend(fontsize=8, loc="upper left", ncol=2)
    n_mr_fills, n_trend_fills = int(mr_fills.sum()), int(trend_fills.sum())
    axB.annotate(f"MR filled {n_mr_fills} trend-gap days; trend filled {n_trend_fills} MR-gap days",
                 xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="#fffbe6", ec="#999"))
    fig.tight_layout()
    p = CHARTS / "gap_filling_timeline.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def _chart_combined_vs_sleeves(equity_data, cs):
    """per TF: combined (50/50) equity vs trend-alone vs MR-alone."""
    if not cs:
        return
    n = len(cs); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False)
    for k, cad in enumerate(cs):
        ax = axes[k // ncol][k % ncol]
        ed = equity_data[cad]; d = ed["dates"]
        ax.plot(d, (ed["trend"] - 1) * 100, color="#1f77b4", label="trend alone")
        ax.plot(d, (ed["mr"] - 1) * 100, color="#ff7f0e", label="MR alone")
        ax.plot(d, (ed["combined"] - 1) * 100, color="#2ca02c", lw=2.2, label="combined 50/50")
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{cad}", fontsize=10)
        ax.set_ylabel("OOS compound %"); ax.legend(fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Combined complementary book (50/50) vs trend-alone vs MR-alone, per TF (2020 OOS, u10, maker)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = CHARTS / "combined_vs_sleeves_equity.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


if __name__ == "__main__":
    sys.exit(main())
