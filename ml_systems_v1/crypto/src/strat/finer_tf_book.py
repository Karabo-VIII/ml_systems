"""src/strat/finer_tf_book.py -- THE DEPLOYABLE BOOK (the runnable capstone of the finer-TF discovery engine).

This is the SHIP-TODAY artifact: it turns the discovery engine's CONVERGED recommendation
(DISCOVERY_ENGINE_FINDINGS.md / DISCOVERY_ENGINE_README.md) into ONE runnable command that produces, on the
REAL 2020 OOS band, the deployable long-only book's equity curve + honest stats vs its benchmarks.

THE BOOK (long-only + spot, the deployable subset -- NO short, NO dynamic timer):
  - TREND   : ADAPTIVE-MA (VIDYA/KAMA, the PHASE-1a per-TF winner from ma_type_tf_research.json) trend sleeve
              + the base trail-stop(0.10) + the per-TF confirm/exit overlay. The robust participating-beta core.
  - MR      : the STATIC mean-reversion oscillator complement (chop DD-dampening, from deep2020_complementarity).
              A long-only complement DAMPENS chop DD; it CANNOT fill a bear gap (the engine's honest finding).
  - VOLTGT_DEF: a vol-target DEFENSIVE overlay on the trend sleeve (scale DOWN in high vol; never lever up).
              The best within-long-only risk-reducer (from complementary_sleeve_search).

It is BETA WITH RISK CONTROL -- NOT alpha, NOT a dynamic regime-timer. On the 2020 OOS (a clean bull) the
book's NET is BELOW buy-hold (the participation tax is EXPECTED + honest); its value is risk-adjusted
(Sharpe / maxDD / p05) + the whole-cycle DD protection the bull cannot show. We report it as such.

PRE-REGISTERED WEIGHTS (no OOS fit): inverse-vol (equal-risk) computed ONCE on the TRAIN+VAL real slice
(everything BEFORE the OOS split, 2020-07-01..2020-10-01 in the scored window), then FROZEN and applied to
the OOS. This is causal -- the weights never see the OOS days they are graded on.

OPTIONAL --longshort-insurance (OFF by default, LO-EXCEPTION-GATED): when flipped on, adds the PHASE-6
LONGSHORT_MA bear-insurance sleeve (the symmetric long-short adaptive-MA engine, maker + modelled
short-borrow). This sleeve VIOLATES the standing long-only+spot constraint -> it is RESEARCH; turning it on
requires the user's explicit long-only-exception sign-off. The book is fully deployable WITHOUT it.

BENCHMARKS (all on the same OOS days):
  - TREND_ALONE : the adaptive-MA trend sleeve by itself (the book must justify the MR+voltgt additions).
  - VOLTGT_BH   : equal-weight u10 buy-hold scaled by the SAME past-only vol-target overlay (defensive beta).
  - BUYHOLD     : equal-weight u10 buy-hold (raw beta -- the bull yardstick the book pays a participation tax to).

HONEST / BINDING CONSTRAINTS (user mandate): 2020 BAND ONLY (never 2026/other data); maker cost 0.0006;
causal lag-1 MtM (no MtM double-count); charts PNG; no emoji (cp1252); RWYB; do NOT git commit.

RWYB:
  python -m strat.finer_tf_book                              # the deployable book on 1d + 4h (the robust coarse TFs)
  python -m strat.finer_tf_book --cadences 1d,4h,1h          # add a finer TF
  python -m strat.finer_tf_book --longshort-insurance        # FLIP ON the bear-insurance sleeve (RESEARCH, LO-exception)
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

# Reuse the EXACT deployable sleeve builders (do NOT reinvent the mechanics).
import strat.deep2020_complementarity as COMP                          # noqa: E402  (WIN/SPLIT/SYMS/WARMUP)
import strat.complementary_sleeve_search as CSS                        # noqa: E402  (MR / VOLTGT_DEF / _panel_window / _daily)
import strat.longshort_ma_engine as LS                                 # noqa: E402  (adaptive-MA trend + longshort + PHASE1A winners)
from strat.portfolio_replay import MAKER_RT, TAKER_RT                  # noqa: E402
from strat.data_expansion import block_bootstrap_distribution          # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
SYMS = COMP.SYMS
SPLIT = pd.Timestamp(COMP.SPLIT)        # OOS = days >= 2020-10-01 (the within-2020 OOS, a clean bull)
ANN = 365                               # daily Sharpe annualization (the books are daily-compounded)

# The deployable cadences to REPORT (the robust coarse cadences). 1d + 4h are the deploy-grade pair; the
# finer TFs are higher-OOS-net but higher-turnover/cost-fragile (a finer-TF bull artifact -- we keep the
# coarse pair as the deployable headline and note the best finer TF).
DEPLOY_CADENCES = ["1d", "4h"]

# PRE-REGISTERED weight policy: inverse-vol (equal-RISK) on the TRAIN+VAL real slice (pre-OOS), then FROZEN.
# This is the ONLY place weights are set, and it never sees the OOS. The long-only book = {TREND, MR, VOLTGT_DEF}.
LONG_ONLY_SLEEVES = ["TREND", "MR", "VOLTGT_DEF"]
# When --longshort-insurance is ON (LO-exception), the longshort sleeve joins at its OWN pre-registered weight.
INSURANCE_SLEEVE = "LONGSHORT"

__contract__ = {
    "kind": "finer_tf_deployable_book",
    "inputs": {
        "trend": "ADAPTIVE-MA (VIDYA/KAMA) per-TF winner trend sleeve (LS._trend_book; PHASE-1a winners from "
                 "ma_type_tf_research.json) -- the participating-beta core, REAL 2020 band",
        "mr": "the static MR oscillator complement (CSS._mr_long_book -> COMP._mr_sleeve) -- chop DD-dampening",
        "voltgt_def": "a past-only vol-target DEFENSIVE overlay on the trend sleeve (CSS._voltgt_def_book)",
        "optional_insurance": "the PHASE-6 LONGSHORT_MA sleeve (LS._longshort_book, maker + modelled short-"
                              "borrow) -- OFF by default; LO-exception-gated (RESEARCH, deploy needs sign-off)",
        "weights": "PRE-REGISTERED inverse-vol (equal-risk) on the TRAIN+VAL real slice (pre-OOS), FROZEN for OOS",
    },
    "outputs": {
        "book_equity_oos": "the deployable book's 2020-OOS daily equity curve (cum %) + honest stats",
        "stats": "net / Sharpe / maxDD / p05(block-bootstrap) / coverage / turnover, vs VOLTGT_BH + BUYHOLD + "
                 "TREND_ALONE -- honestly reported (it is BETA: net < buy-hold in the 2020 bull is EXPECTED)",
        "best_deployable_tf": "the best of the reported deployable cadences by risk-adjusted (Sharpe) OOS",
    },
    "invariants": {
        "band_2020_only": "REAL 2020 band ONLY (WIN=2020-07-01..2021-01-01, OOS>=2020-10-01); never 2026/other",
        "causal_lag1_maker": "the sleeve builders lag the position 1 bar + charge maker half-spread per flip + "
                             "no MtM double-count (the deployable cost convention from the simulator invariants)",
        "pre_registered_weights": "inverse-vol weights fit on TRAIN+VAL (pre-OOS) ONLY, then FROZEN -- never "
                                  "fit on the OOS days they are graded on (no look-ahead)",
        "long_only_default": "the default book is long-only+spot {TREND, MR, VOLTGT_DEF}; the LONGSHORT "
                             "insurance sleeve is OFF unless --longshort-insurance (LO-exception sign-off)",
        "honest_beta_reporting": "the book is PARTICIPATING BETA with risk control, NOT alpha -- net < buy-hold "
                                 "in the 2020 bull is EXPECTED and stated; the value is risk-adjusted",
        "reuse_not_reinvent": "every sleeve is the EXACT deployable builder from the engine modules; this file "
                              "only ASSEMBLES + grades + charts -- no new signal mechanics",
    },
}


# =====================================================================================================
# 1. SLEEVE ASSEMBLY (REAL 2020 band -- no synthetic context, so CSS/LS builders read the real panels)
# =====================================================================================================
def build_long_only_sleeves(cad):
    """Build the 3 deployable long-only sleeve daily books on the REAL 2020 band for one cadence. Returns
    {name -> daily pd.Series over the WIN window} (the caller slices to OOS). NO synthetic context here -> the
    CSS/LS panel loaders read the real 2020 panels. TREND uses the adaptive per-TF winner; MR + VOLTGT_DEF are
    CSS's deployable long-only sleeves."""
    trend = LS._trend_book(cad)                       # adaptive-MA (VIDYA/KAMA) per-TF winner, long-only
    return {
        "TREND": trend,
        "MR": CSS._mr_long_book(cad),                 # static oscillator complement (COMP._mr_sleeve)
        "VOLTGT_DEF": CSS._voltgt_def_book(cad, trend),  # past-only vol-target defensive overlay on trend
    }


def build_insurance_sleeve(cad, borrow_bps=LS.BORROW_BASE):
    """Build the OPTIONAL longshort bear-insurance sleeve (PHASE-6 engine) on the REAL 2020 band. RESEARCH:
    violates long-only+spot; only added when --longshort-insurance (LO-exception sign-off). maker + modelled
    short-borrow."""
    return LS._longshort_book(cad, borrow_bps=borrow_bps)


# =====================================================================================================
# 2. PRE-REGISTERED WEIGHTS -- inverse-vol on the TRAIN+VAL (pre-OOS) real slice, then FROZEN.
# =====================================================================================================
def pre_registered_weights(sleeves, names):
    """Inverse-vol (equal-RISK) weights computed on the PRE-OOS slice (days < SPLIT) of each sleeve, then
    FROZEN. This is causal -- the weights are set on the TRAIN+VAL window and never see the OOS days they are
    graded on. Sleeves with degenerate/no pre-OOS vol get an equal fallback. Returns {name -> weight} (sums 1)."""
    vols = {}
    for n in names:
        s = sleeves.get(n)
        if s is None:
            vols[n] = None
            continue
        pre = s[s.index < SPLIT].dropna()
        vols[n] = float(pre.std()) if (len(pre) >= 5 and float(pre.std()) > 1e-9) else None
    have = [n for n in names if vols[n] is not None]
    if not have:
        return {n: 1.0 / len(names) for n in names}
    inv = {n: 1.0 / vols[n] for n in have}
    tot = sum(inv.values())
    w = {n: inv[n] / tot for n in have}
    for n in names:
        w.setdefault(n, 0.0)
    return w


def blend_book(sleeves, weights):
    """Blend the sleeve daily books at the FIXED pre-registered weights on the common daily index (over the
    full WIN window; the caller slices to OOS). Returns (blended daily Series, per-day active-sleeve count)."""
    cols = {n: s for n, s in sleeves.items() if s is not None and weights.get(n, 0.0) > 0}
    if not cols:
        return None, None
    df = pd.concat([s.rename(n) for n, s in cols.items()], axis=1).dropna()
    if len(df) < 8:
        return None, None
    wsum = sum(weights.get(n, 0.0) for n in df.columns)
    blended = np.zeros(len(df))
    for n in df.columns:
        blended += weights.get(n, 0.0) * df[n].to_numpy()
    if wsum > 1e-9:
        blended = blended / wsum                      # renormalize over present sleeves
    active = (np.abs(df.to_numpy()) > 1e-9).sum(axis=1)
    return pd.Series(blended, index=df.index), active


# =====================================================================================================
# 3. BENCHMARKS -- BUYHOLD (raw beta) + VOLTGT_BH (defensive beta), on the same OOS days.
# =====================================================================================================
def buyhold_daily(cad):
    """Equal-weight u10 buy-hold daily return over the WIN window (the raw-beta yardstick). Reuses CSS's
    real-panel equal-weight daily substrate."""
    return CSS._bh_eqw_daily(cad)


def voltgt_bh_daily(cad, lookback=10, target_vol=0.02, max_scale=1.0):
    """Equal-weight u10 buy-hold scaled by the SAME past-only vol-target overlay VOLTGT_DEF uses (defensive
    beta benchmark). scale = min(max_scale, target_vol / trailing_realized_vol), shifted 1 day (causal),
    capped at max_scale (defend, never lever). Mirrors CSS._voltgt_def_book's overlay but on buy-hold."""
    bh = CSS._bh_eqw_daily(cad)
    if bh is None:
        return None
    rv = bh.rolling(lookback, min_periods=lookback).std().shift(1)
    scale = np.clip(target_vol / (rv.to_numpy() + 1e-9), 0.0, max_scale)
    scale = np.where(np.isfinite(scale), scale, max_scale)
    return pd.Series(bh.to_numpy() * scale, index=bh.index)


# =====================================================================================================
# 4. METRICS (OOS only) -- net / Sharpe / maxDD / p05 / coverage / turnover.
# =====================================================================================================
def _oos(s):
    return s[s.index >= SPLIT] if s is not None else None


def perf(x):
    """net / Sharpe / maxDD / p05 (block-bootstrap) on a daily return array."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"net": None, "sharpe": None, "maxdd": None, "p05": None, "n_days": int(len(x))}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    bb = block_bootstrap_distribution(x, n_boot=600, block=5, seed=13)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
            "p05": round(float(bb["p05"]) * 100, 1),
            "n_days": int(len(x))}


def equity_curve(s):
    """cum-% equity curve + dates for an OOS daily Series (for the chart + JSON)."""
    if s is None or len(s) == 0:
        return {"dates": [], "cum_pct": []}
    cum = (np.cumprod(1 + s.to_numpy()) - 1.0) * 100
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index], "cum_pct": [round(float(v), 2) for v in cum]}


# =====================================================================================================
# 5. THE BOOK (one cadence) -- assemble, weight, grade, vs benchmarks.
# =====================================================================================================
def run_cadence(cad, longshort_insurance=False, borrow_bps=LS.BORROW_BASE):
    """Build + grade the deployable book on the real 2020 OOS for one cadence. Returns the full result dict
    (stats + equity + weights + sleeve OOS stats + benchmarks) or None if the panels are unavailable."""
    sleeves = build_long_only_sleeves(cad)
    if sleeves.get("TREND") is None:
        return None
    names = list(LONG_ONLY_SLEEVES)
    if longshort_insurance:
        ins = build_insurance_sleeve(cad, borrow_bps=borrow_bps)
        if ins is not None:
            sleeves[INSURANCE_SLEEVE] = ins
            names = names + [INSURANCE_SLEEVE]

    # PRE-REGISTERED weights on the pre-OOS slice, then FROZEN
    weights = pre_registered_weights(sleeves, names)

    # the BOOK over the full WIN, then slice to OOS
    book_full, active_full = blend_book(sleeves, weights)
    if book_full is None:
        return None
    book_oos = _oos(book_full)
    active_oos = active_full[book_full.index >= SPLIT] if active_full is not None else None

    # coverage (frac of OOS days with >=1 engaged sleeve) + turnover (mean |day-over-day book return change|
    # proxy is not meaningful; we report book-level turnover as the trend sleeve's flip-driven turnover proxy
    # via |diff of sign(book)| is misleading -- instead report COVERAGE + the trend sleeve's turnover analogue
    # as the deployable engagement signal). We compute coverage from active sleeves + a turnover proxy from
    # the book's nonzero-day fraction transitions.
    coverage = float(np.mean(active_oos > 0)) if active_oos is not None and len(active_oos) else None
    bk = book_oos.to_numpy()
    # turnover proxy: number of activity on/off transitions per day (book engaged <-> flat), annualized-free
    engaged = (np.abs(bk) > 1e-9).astype(int)
    turnover = round(float(np.mean(np.abs(np.diff(np.concatenate([[0], engaged]))))), 3) if len(engaged) else None

    book_perf = perf(bk)
    book_perf["coverage"] = round(coverage, 3) if coverage is not None else None
    book_perf["turnover_proxy"] = turnover

    # benchmarks on the same OOS days
    trend_oos = _oos(sleeves["TREND"])
    bh_oos = _oos(buyhold_daily(cad))
    voltgt_bh_oos = _oos(voltgt_bh_daily(cad))

    benchmarks = {
        "TREND_ALONE": perf(trend_oos.to_numpy()) if trend_oos is not None else None,
        "VOLTGT_BH": perf(voltgt_bh_oos.to_numpy()) if voltgt_bh_oos is not None else None,
        "BUYHOLD": perf(bh_oos.to_numpy()) if bh_oos is not None else None,
    }

    # per-sleeve OOS stats (transparency on what each leg contributes)
    sleeve_oos = {}
    for n in names:
        s = _oos(sleeves.get(n))
        sleeve_oos[n] = perf(s.to_numpy()) if s is not None else None

    return {
        "cadence": cad,
        "ma_type": LS.PHASE1A_WINNERS.get(cad, {}).get("ma_type"),
        "longshort_insurance_on": bool(longshort_insurance),
        "weights": {n: round(float(weights.get(n, 0.0)), 3) for n in names},
        "book_oos": book_perf,
        "benchmarks": benchmarks,
        "sleeve_oos": sleeve_oos,
        "equity": {
            "BOOK": equity_curve(book_oos),
            "TREND_ALONE": equity_curve(trend_oos),
            "VOLTGT_BH": equity_curve(voltgt_bh_oos),
            "BUYHOLD": equity_curve(bh_oos),
        },
    }


# =====================================================================================================
# 6. CHART -- the deployable book vs trend-alone / VOLTGT_BH / BUYHOLD, stats annotated.
# =====================================================================================================
C = {"BOOK": "#1b7837", "TREND_ALONE": "#2166ac", "VOLTGT_BH": "#e08214", "BUYHOLD": "#999999"}


def chart_book(results, longshort_insurance):
    """One figure: per reported cadence, the deployable book's OOS equity vs trend-alone / VOLTGT_BH /
    BUYHOLD, with the honest stats annotated. Saves charts/finer_tf_book_equity.png."""
    cads = [c for c in results if results[c] is not None]
    if not cads:
        print("[chart] no cadences to plot")
        return
    n = len(cads)
    fig, axes = plt.subplots(1, n, figsize=(8.2 * n, 6.2), squeeze=False)
    title_extra = "  +LONGSHORT insurance (RESEARCH, LO-exception)" if longshort_insurance else ""
    for ci, cad in enumerate(cads):
        ax = axes[0][ci]
        r = results[cad]
        eq = r["equity"]
        order = ["BUYHOLD", "VOLTGT_BH", "TREND_ALONE", "BOOK"]
        for name in order:
            e = eq.get(name)
            if not e or not e["cum_pct"]:
                continue
            dates = pd.to_datetime(e["dates"])
            lw = 2.6 if name == "BOOK" else 1.6
            z = 5 if name == "BOOK" else 2
            ax.plot(dates, e["cum_pct"], label=name.replace("_", " "), color=C[name], lw=lw, zorder=z)
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_title(f"{cad}: deployable book OOS equity (2020 Oct-Dec, real)", fontsize=11, fontweight="bold")
        ax.set_ylabel("cumulative net % (OOS)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        # annotate the honest stats block
        bp = r["book_oos"]; bm = r["benchmarks"]
        ta = bm.get("TREND_ALONE") or {}; vb = bm.get("VOLTGT_BH") or {}; bh = bm.get("BUYHOLD") or {}
        wtxt = ", ".join(f"{k}={v:.2f}" for k, v in r["weights"].items())
        lines = [
            f"BOOK ({r['ma_type']}-MA trend + MR + voltgt_def)",
            f"  net {bp['net']}%  Sharpe {bp['sharpe']}  maxDD {bp['maxdd']}%",
            f"  p05 {bp['p05']}%  coverage {bp.get('coverage')}  turnover {bp.get('turnover_proxy')}",
            f"  weights: {wtxt}",
            "",
            f"TREND-ALONE net {ta.get('net')}% (Sh {ta.get('sharpe')}, DD {ta.get('maxdd')}%)",
            f"VOLTGT-BH   net {vb.get('net')}% (Sh {vb.get('sharpe')}, DD {vb.get('maxdd')}%)",
            f"BUYHOLD     net {bh.get('net')}% (Sh {bh.get('sharpe')}, DD {bh.get('maxdd')}%)",
            "",
            "HONEST: BETA with risk control, NOT alpha.",
            "Net < buy-hold in a bull = the participation tax",
            "(EXPECTED). Value = risk-adjusted + DD control.",
        ]
        ax.text(0.985, 0.02, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.6, family="monospace",
                bbox=dict(boxstyle="round", fc="#f4faf4", ec="#1b7837", alpha=0.95))
    fig.suptitle(
        "THE DEPLOYABLE FINER-TF BOOK -- 2020 OOS equity vs trend-alone / VOLTGT-BH / BUYHOLD" + title_extra
        + "\nadaptive-MA (VIDYA/KAMA) trend + static MR complement + VOLTGT_DEF overlay  |  long-only+spot  |  "
          "maker 0.0006  |  causal lag-1 MtM  |  PRE-REGISTERED inverse-vol weights (pre-OOS, frozen)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = CHARTS / "finer_tf_book_equity.png"
    fig.savefig(p, dpi=130, facecolor="white"); plt.close(fig)
    print(f"[figure] {p}  ({p.stat().st_size} bytes)")


# =====================================================================================================
# 7. VERDICT (honest, two-sided) + the best deployable TF.
# =====================================================================================================
def build_verdict(results, longshort_insurance):
    lines = []
    cads = [c for c in results if results[c] is not None]
    # best deployable TF by risk-adjusted (Sharpe) OOS
    best_tf, best_sharpe = None, -1e9
    for c in cads:
        sh = results[c]["book_oos"].get("sharpe")
        if sh is not None and sh > best_sharpe:
            best_sharpe, best_tf = sh, c

    lines.append("THE DEPLOYABLE FINER-TF BOOK -- 2020 OOS (Oct-Dec, real, a clean bull) honest stats:")
    for c in cads:
        r = results[c]; bp = r["book_oos"]; bm = r["benchmarks"]
        bh = bm.get("BUYHOLD") or {}; vb = bm.get("VOLTGT_BH") or {}; ta = bm.get("TREND_ALONE") or {}
        beats_bh = (bp["net"] or 0) > (bh.get("net") or 0)
        lines.append(
            f"  {c:3}: BOOK net {bp['net']}% / Sharpe {bp['sharpe']} / maxDD {bp['maxdd']}% / p05 {bp['p05']}% "
            f"/ coverage {bp.get('coverage')} / turnover {bp.get('turnover_proxy')}")
        lines.append(
            f"       vs BUYHOLD net {bh.get('net')}% (Sh {bh.get('sharpe')}) | VOLTGT-BH net {vb.get('net')}% "
            f"(Sh {vb.get('sharpe')}) | TREND-ALONE net {ta.get('net')}% (Sh {ta.get('sharpe')})")
        # honest beta read
        if not beats_bh:
            lines.append(f"       -> BETA: book net < buy-hold (participation tax, EXPECTED in a bull); "
                         f"value is risk-adjusted (Sharpe {bp['sharpe']} vs BH {bh.get('sharpe')}, "
                         f"maxDD {bp['maxdd']}% vs BH {bh.get('maxdd')}%)")
        else:
            lines.append(f"       -> book net >= buy-hold this OOS (a bull where the trend sleeve participated "
                         f"strongly); still BETA, not alpha -- whole-cycle DD protection is the durable value")

    lines.append("")
    lines.append(f"-> BEST DEPLOYABLE TF (by risk-adjusted OOS Sharpe): {best_tf} (Sharpe {best_sharpe})")
    if longshort_insurance:
        lines.append("")
        lines.append("LONGSHORT INSURANCE is ON (--longshort-insurance): the book now carries the PHASE-6 "
                     "long-short bear-insurance sleeve. THIS IS RESEARCH -- it violates long-only+spot; "
                     "deploying it requires the user's explicit LO-exception sign-off. The 2020 OOS is a bull, "
                     "so the insurance sleeve DRAGS here (it shorts a rising market); its value is bear-DD "
                     "protection, which a bull OOS cannot show (see longshort_book.json for the full-cycle "
                     "synthetic stress where it lowers bear maxDD).")
    else:
        lines.append("")
        lines.append("LONGSHORT INSURANCE is OFF (default, deployable today). The bear from a liability into "
                     "near-flat unlock needs the LO-exception sign-off (--longshort-insurance to preview; "
                     "RESEARCH). Within long-only+spot, VOLTGT_DEF is the best risk-reducer (dampens, never rescues).")

    lines.append("")
    lines.append("CAVEATS (binding): (1) 2020 OOS is a CLEAN BULL (~0% bear) -- the book is participating beta, "
                 "net < buy-hold is EXPECTED; the DD-protection value is what generalizes, untested here on a real "
                 "bear. (2) REAL 2020 band ONLY; no 2026/other data touched. (3) PRE-REGISTERED inverse-vol weights "
                 "on the pre-OOS (TRAIN+VAL) slice, FROZEN -- no OOS fit. (4) maker 0.0006, causal lag-1 MtM, no "
                 "double-count. (5) the finer TFs post higher OOS net but at higher turnover/cost-fragility (a "
                 "finer-TF bull artifact); 1d+4h are the deployable coarse headline. (6) LONGSHORT insurance = "
                 "RESEARCH (short violates long-only+spot); OFF by default.")
    return {"best_deployable_tf": best_tf, "best_deployable_sharpe": best_sharpe,
            "longshort_insurance_on": bool(longshort_insurance), "lines": lines}


# =====================================================================================================
# 8. MAIN
# =====================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.finer_tf_book")
    ap.add_argument("--cadences", default=",".join(DEPLOY_CADENCES),
                    help="comma-separated cadences to report (default: 1d,4h -- the robust coarse pair)")
    ap.add_argument("--longshort-insurance", action="store_true",
                    help="FLIP ON the PHASE-6 longshort bear-insurance sleeve (RESEARCH; LO-exception sign-off)")
    ap.add_argument("--borrow-bps", type=float, default=LS.BORROW_BASE,
                    help="short-borrow bps/yr for the insurance sleeve (only used with --longshort-insurance)")
    a = ap.parse_args(argv)

    CHARTS.mkdir(parents=True, exist_ok=True)
    cadences = [c.strip() for c in a.cadences.split(",") if c.strip()]
    print("## THE DEPLOYABLE FINER-TF BOOK -- the ship-today long-only book (the discovery-engine recommendation)")
    print(f"   cadences={cadences}  long-only+spot={'NO (insurance ON, RESEARCH)' if a.longshort_insurance else 'YES'}  "
          f"maker={MAKER_RT}  band=2020 OOS (Oct-Dec, real)")
    if a.longshort_insurance:
        print("   >> --longshort-insurance ON: the longshort sleeve VIOLATES long-only+spot -> RESEARCH "
              "(deploy needs the user's explicit LO-exception sign-off).")

    results = {}
    for cad in cadences:
        print(f"\n## building + grading the book at {cad} ...")
        results[cad] = run_cadence(cad, longshort_insurance=a.longshort_insurance, borrow_bps=a.borrow_bps)
        r = results[cad]
        if r is None:
            print(f"   {cad}: panels unavailable -- SKIPPED")
            continue
        bp = r["book_oos"]; bm = r["benchmarks"]; bh = bm.get("BUYHOLD") or {}
        print(f"   {cad}: BOOK net {bp['net']}% / Sharpe {bp['sharpe']} / maxDD {bp['maxdd']}% / p05 {bp['p05']}% "
              f"(weights {r['weights']})")
        print(f"        vs BUYHOLD net {bh.get('net')}% (Sh {bh.get('sharpe')}, DD {bh.get('maxdd')}%) "
              f"-- net<BH is the participation tax (BETA, expected)")

    # verdict
    verdict = build_verdict(results, a.longshort_insurance)
    print("\n" + "=" * 100)
    print("## HONEST VERDICT")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 100)

    # chart
    chart_book(results, a.longshort_insurance)

    # persist
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export = {
        "repro": {"command": "python -m strat.finer_tf_book " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "cost_taker": TAKER_RT,
                  "win": COMP.WIN, "split": COMP.SPLIT, "oos": "2020-10-01..2021-01-01 (real, clean bull)",
                  "cadences": cadences, "long_only_sleeves": LONG_ONLY_SLEEVES,
                  "longshort_insurance_on": bool(a.longshort_insurance),
                  "phase1a_winners": LS.PHASE1A_WINNERS, "weight_policy": "inverse_vol_preOOS_frozen",
                  "borrow_bps": a.borrow_bps if a.longshort_insurance else None,
                  "constraint": "REAL 2020 BAND ONLY; never 2026/other; maker cost; causal lag-1 MtM; "
                                "PRE-REGISTERED weights (pre-OOS, frozen); LONGSHORT insurance = RESEARCH "
                                "(deploy needs LO-exception sign-off)"},
        "results": results,
        "verdict": verdict,
    }
    p = OUT / "finer_tf_book.json"
    json.dump(export, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
