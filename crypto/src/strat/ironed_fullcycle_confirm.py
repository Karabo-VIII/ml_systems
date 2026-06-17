"""src/strat/ironed_fullcycle_confirm.py -- THE HONEST NEXT GATE: confirm/refute the IRONED MA trend
systems + combined book on a BEAR-INCLUSIVE FULL-CYCLE window, using the canonical scorecard.

PROVENANCE (user /orc 2026-06-13): two per-TF ironed MA trend systems + a combined book were built and
verified ONLY on the 2020 bull (0% bear in the 2020 OOS). Their de-risk creases (regime gate, vol-target)
and drawdown-protection value are ASSERTED-but-UNSHOWN. This run CONFIRMS or REFUTES them on the full
2021-2026 u10 panel via the standard strat-layer loader, graded by src/strat/scorecard.py canonical splits
(SEL 2018..2025-03-15 / OOS 2025-03-15..12-31 / UNSEEN 2025-12-31..2026-06-01 -- includes the 2022 bear,
2025 chop, and the UNSEEN test-once tail). This is a GENERALIZATION/ROBUSTNESS test, two-sided.

DISCIPLINE (binding):
  - PRE-REGISTERED SPECS, used VERBATIM (commit 3822c29 / ironed_coarse.py + ironed_combined.py):
      1d : FAMILY-ONLY  -- dict(family=True, exit_="none", conf_k=0,  gate=False, voltgt=False)
      4h : FULL-IRONED  -- dict(family=True, exit_="none", conf_k=2,  gate=True,  voltgt=True)
      COMBINED: {1d, 4h} equal-weight trend CORE + funding-dispersion CARRY satellite, 70/30 capital,
                3x satellite leverage cap (per core_satellite_book). (15m optional; coarse-only here is
                the strict DEPLOY pair the user named -- 15m is a separate fine builder, reported as N/A.)
  - NO RE-TUNING on the full window. The ONLY thing changed vs the 2020 builder is (a) the data WINDOW
    (full 2020-01-07..2026-05-28) and (b) the regime-gate breadth-tercile FIT window, which is moved to a
    causal SEL-PREFIX (2020-01-07..2023-01-01) so it does NOT look ahead into OOS/UNSEEN. The gate's
    policy map (bull->hold, neutral->1.0x, bear->flat), the family, the confirm-K, the vol-target formula,
    the costs -- all VERBATIM. Re-fitting the breadth terciles on the early SEL prefix is the SAME
    procedure the 2020 builder ran (fit-on-TRAIN), just on the full-cycle TRAIN prefix; it is NOT a
    re-tune of the spec.
  - Causal / lag-1 / maker cost (MAKER_RT), same as the builders.
  - Graded by score_book on the daily net stream. UNSEEN is TEST-ONCE (read once, never re-touched).
  - Every number is VERIFIED iff produced by this run (RWYB).

RWYB: python -m strat.ironed_fullcycle_confirm
JSON: runs/strat/ironed_fullcycle_confirm.json ; MD hand-written: runs/strat/IRONED_FULLCYCLE_CONFIRM.md.
No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))

import strat.portfolio_replay as PR                                       # noqa: E402
from strat.portfolio_replay import MAKER_RT                              # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.ma_type_upgrade import _nums                                  # noqa: E402
from strat.battery import block_bootstrap_p05_p95                        # noqa: E402
import strat.ironed_coarse as IC                                         # noqa: E402
from strat.scorecard import score_book, SPLITS                           # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)
ANN_DAILY = 365.0

# the full-cycle window (the canonical scorecard windows live INSIDE this; data starts 2020-01-07)
FULL_YEAR = ("2020-01-01", "2026-06-01")
# the regime-gate breadth-tercile FIT window = a causal SEL prefix (covers the 2022 bear + recovery),
# strictly BEFORE OOS (2025-03-15) and UNSEEN. This is the full-cycle analogue of the 2020 fit-on-TRAIN.
REGIME_FIT = ("2020-01-01", "2023-01-01")

# PRE-REGISTERED spec kwargs (VERBATIM from ironed_combined.coarse_sleeve calls; commit 3822c29)
SPEC = {
    "1d": dict(family=True, exit_="none", conf_k=0, gate=False, voltgt=False),   # 1d FAMILY-ONLY
    "4h": dict(family=True, exit_="none", conf_k=2, gate=True, voltgt=True),     # 4h FULL-IRONED
}

# satellite sizing discipline (per core_satellite_book) -- pre-registered deployment constants
SAT_MAX_LEVERAGE = 3.0
RECO_CORE_FRAC = 0.70

# a quick bear sub-window for the de-risk-value leg (2022 bear -- the deepest sustained crypto bear in-window)
BEAR_2022 = ("2021-11-10", "2022-12-31")   # ATH-to-bottom-ish sustained bear


def _slow_family():
    """the slow EMA family (60<=max_len<150), VERBATIM from ironed_coarse.run_cadence."""
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    slow2 = [n for n in slow if len(_nums(n)) == 2]
    fam_set = slow2 if len(slow2) >= 5 else slow
    return fam_set


def _to_daily(book):
    """resample a bar-level book net Series to a daily compound net Series over the full window."""
    s = book.dropna()
    daily = s.resample("1D").apply(lambda v: float(np.prod(1 + v) - 1)).dropna()
    daily.index = pd.to_datetime(daily.index).normalize()
    daily = daily[~daily.index.duplicated(keep="last")]
    return daily


def coarse_sleeve_fullcycle(cad):
    """Reconstruct the PRE-REGISTERED coarse sleeve on the FULL window. Returns (daily net, expo daily,
    regime series, thresholds, bar-level book). Overrides IC.YEAR so IC._closes loads the full history;
    fits the regime gate on the causal SEL prefix only (no look-ahead)."""
    fam_set = _slow_family()
    # override the builder's window constant so _closes spans the full cycle (no spec re-tune)
    IC.YEAR = FULL_YEAR
    closes = IC._closes(cad)
    panel_df = IC._book_close_panel(closes, cad)
    regime, th = IC.market_regime(panel_df, cad, REGIME_FIT[0], REGIME_FIT[1])
    book, expo = IC.build_stack(closes, panel_df, regime, cad, slow=fam_set, **SPEC[cad])
    daily = _to_daily(book)
    expo_daily = expo.resample("1D").mean().dropna() if expo is not None else None
    if expo_daily is not None:
        expo_daily.index = pd.to_datetime(expo_daily.index).normalize()
    return daily, expo_daily, regime, th, book, closes


def buyhold_benchmark(cad):
    """equal-weight long-only BUYHOLD daily net stream on the full window (the de-risk comparison bar)."""
    IC.YEAR = FULL_YEAR
    closes = IC._closes(cad)
    bm = IC.benchmarks(closes, cad)
    bh = _to_daily(bm["BUYHOLD"])
    vtg = _to_daily(bm["VOLTGT_BH"])
    return bh, vtg


def carry_satellite():
    """funding-dispersion DEPLOYABLE daily net stream (the orthogonal carry), full window."""
    from strat.funding_satellite_assessment import satellite_net_stream
    s = satellite_net_stream(universe="u50", k=5)
    s.index = pd.to_datetime(s.index).normalize()
    return s[~s.index.duplicated(keep="last")].dropna()


# ---------------------------------------------------------------------------
# bear-leg de-risk metrics (Q3a)
# ---------------------------------------------------------------------------
def _maxdd(daily):
    d = np.asarray(daily, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan")
    eq = np.cumprod(1 + d); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _compound(daily):
    d = np.asarray(daily, float); d = d[np.isfinite(d)]
    return float((np.prod(1 + d) - 1) * 100) if len(d) else float("nan")


def _sl(s, lo, hi):
    return s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))].dropna()


def bear_leg(daily, bh, expo_daily, regime, cad):
    """Q3a: on the 2022 bear, is the system maxDD MATERIALLY below buy-hold? Does the gate de-risk?"""
    lo, hi = BEAR_2022
    sys_b = _sl(daily, lo, hi)
    bh_b = _sl(bh, lo, hi)
    out = {"window": BEAR_2022,
           "system_compound_pct": round(_compound(sys_b), 1), "system_maxdd_pct": round(_maxdd(sys_b), 1),
           "buyhold_compound_pct": round(_compound(bh_b), 1), "buyhold_maxdd_pct": round(_maxdd(bh_b), 1),
           "n_days": int(len(sys_b))}
    out["maxdd_protection_pp"] = round(out["system_maxdd_pct"] - out["buyhold_maxdd_pct"], 1)  # +ve = less DD
    # gate exposure in the bear: avg book exposure + regime label share over the bear window
    if expo_daily is not None:
        e = _sl(expo_daily, lo, hi)
        out["avg_book_exposure_in_bear"] = round(float(e.mean()), 3) if len(e) else None
    if regime is not None:
        rb = regime[(regime.index >= pd.Timestamp(lo)) & (regime.index < pd.Timestamp(hi))]
        if len(rb):
            out["regime_share_in_bear"] = {k: round(float(np.mean(rb.to_numpy() == k)), 3)
                                           for k in ("bull", "neutral", "bear")}
    return out


def random_entry_null(daily, n_iter=500, seed=11):
    """cost-matched random-entry null: same #in-market days, same per-day net magnitudes, randomly
    permuted onto the calendar (destroys timing/regime skill, preserves cost+vol). Returns p-value that
    the real compound beats the null compound distribution."""
    d = daily.dropna().to_numpy()
    if len(d) < 30:
        return None
    real = float(np.prod(1 + d) - 1)
    rng = np.random.default_rng(seed)
    null = np.empty(n_iter)
    for i in range(n_iter):
        perm = rng.permutation(d)
        null[i] = float(np.prod(1 + perm) - 1)
    # permutation of i.i.d. returns leaves compound ~invariant -> this null tests timing only weakly;
    # the SHARPE/maxDD null is the right one. We report both compound-beat and a maxDD comparison.
    p_compound = float(np.mean(null >= real))
    # maxDD null: does the real path have a SHALLOWER maxDD than random orderings (the de-risk timing test)?
    def mdd(x):
        eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq); return float(((eq - pk) / pk).min())
    real_mdd = mdd(d)
    null_mdd = np.array([mdd(rng.permutation(d)) for _ in range(n_iter)])
    p_mdd = float(np.mean(null_mdd <= real_mdd))   # frac of random orderings with maxDD as shallow as real
    return {"real_compound": round(real * 100, 1), "null_compound_median": round(float(np.median(null)) * 100, 1),
            "p_compound_beat": round(p_compound, 3),
            "real_maxdd_pct": round(real_mdd * 100, 1), "null_maxdd_median_pct": round(float(np.median(null_mdd)) * 100, 1),
            "p_maxdd_shallower": round(p_mdd, 3)}


# ---------------------------------------------------------------------------
# combined book (Q1..4 of the combined layer) -- 70/30 core+carry, 3x sat cap
# ---------------------------------------------------------------------------
def equal_weight_core(sleeves):
    df = pd.DataFrame(sleeves).sort_index()
    return df.mean(axis=1, skipna=True)


def blend_core_sat(core_daily, sat_daily, core_frac, sat_lev):
    L = min(float(sat_lev), SAT_MAX_LEVERAGE)
    df = pd.concat({"core": core_daily, "sat": sat_daily}, axis=1).dropna()
    if len(df) < 10:
        return None, None
    combined = core_frac * df["core"] + (1.0 - core_frac) * (L * df["sat"])
    return combined, df


def main(argv=None):
    print("## IRONED FULL-CYCLE CONFIRM -- de-risk-value generalization test (PRE-REGISTERED specs, no re-tune)")
    print(f"   full window {FULL_YEAR}; regime-gate fit (causal SEL prefix) {REGIME_FIT}; cost maker {MAKER_RT}")
    print(f"   canonical splits {SPLITS}")
    print(f"   1d spec {SPEC['1d']} | 4h spec {SPEC['4h']}")
    sys.stdout.flush()

    results = {}

    # ---- per-TF sleeves on the full cycle ----
    sleeves = {}
    sleeve_meta = {}
    bh_by_cad = {}
    for cad in ("1d", "4h"):
        print(f"\n========== SLEEVE {cad} (spec {SPEC[cad]}) ==========")
        sys.stdout.flush()
        daily, expo_daily, regime, th, book, closes = coarse_sleeve_fullcycle(cad)
        bh, vtg = buyhold_benchmark(cad)
        bh_by_cad[cad] = bh
        sleeves[cad] = daily
        # canonical scorecard
        card = score_book(f"ironed_{cad}", daily)
        # bear leg
        bear = bear_leg(daily, bh, expo_daily, regime, cad)
        # nulls (full-cycle and held-out)
        held = _sl(daily, SPLITS["OOS"][0], SPLITS["UNSEEN"][1])
        null_full = random_entry_null(daily)
        bh_card = score_book(f"buyhold_{cad}", bh)
        sleeve_meta[cad] = {"thresholds": th, "regime_fit_window": REGIME_FIT}
        results[f"sleeve_{cad}"] = {
            "spec": SPEC[cad], "regime_thresholds": th, "scorecard": card,
            "buyhold_scorecard": bh_card, "bear_leg_2022": bear, "random_entry_null_full": null_full,
        }
        # console summary
        ps = card["per_split"]
        def g(sp, k):
            return ps.get(sp, {}).get(k)
        print(f"   [scorecard] SEL comp {g('SEL','compound_pct')}% (DD {g('SEL','maxdd_pct')}) | "
              f"OOS comp {g('OOS','compound_pct')}% (DD {g('OOS','maxdd_pct')}) | "
              f"UNSEEN comp {g('UNSEEN','compound_pct')}% (DD {g('UNSEEN','maxdd_pct')})")
        print(f"   [full p05] {card['full_block_bootstrap'].get('p05')} | held-out p05 "
              f"{card['heldout_block_bootstrap'].get('p05')} | ship_read {card['ship_read']}")
        print(f"   [BUYHOLD]  SEL comp {bh_card['per_split'].get('SEL',{}).get('compound_pct')}% | "
              f"OOS {bh_card['per_split'].get('OOS',{}).get('compound_pct')}% | "
              f"UNSEEN {bh_card['per_split'].get('UNSEEN',{}).get('compound_pct')}%")
        print(f"   [BEAR 2022] system maxDD {bear['system_maxdd_pct']} vs BH {bear['buyhold_maxdd_pct']} "
              f"(protection {bear['maxdd_protection_pp']:+}pp) | sys comp {bear['system_compound_pct']}% "
              f"vs BH {bear['buyhold_compound_pct']}% | regime-share {bear.get('regime_share_in_bear')} | "
              f"avg-exp {bear.get('avg_book_exposure_in_bear')}")
        print(f"   [NULL] real comp {null_full['real_compound']}% vs random-median {null_full['null_compound_median']}% "
              f"(p_beat {null_full['p_compound_beat']}); real maxDD {null_full['real_maxdd_pct']} vs "
              f"random-median {null_full['null_maxdd_median_pct']} (p_shallower {null_full['p_maxdd_shallower']})")
        sys.stdout.flush()

    # ---- COMBINED book: {1d,4h} equal-weight trend CORE + carry satellite (70/30, 3x cap) ----
    print(f"\n========== COMBINED BOOK: trend CORE {{1d,4h}} + carry SATELLITE (70/30, 3x cap) ==========")
    sys.stdout.flush()
    core = equal_weight_core({"1d": sleeves["1d"], "4h": sleeves["4h"]})
    core_card = score_book("trend_core_1d_4h", core)
    sat = carry_satellite()
    sat_card = score_book("carry_satellite", sat)
    combined, df = blend_core_sat(core, sat, RECO_CORE_FRAC, SAT_MAX_LEVERAGE)
    comb_card = score_book("combined_core_satellite", combined) if combined is not None else None
    # core-alone on the SAME overlap as the blend (apples-to-apples diversification read)
    core_overlap = df["core"] if df is not None else core
    core_overlap_card = score_book("core_on_blend_overlap", core_overlap) if df is not None else core_card
    corr = float(np.corrcoef(df["core"].to_numpy(), df["sat"].to_numpy())[0, 1]) if df is not None and len(df) > 2 else None
    # cross-TF correlation
    cdf = pd.DataFrame({"1d": sleeves["1d"], "4h": sleeves["4h"]}).dropna()
    xtf_corr = float(cdf.corr().iloc[0, 1]) if len(cdf) > 10 else None

    results["combined"] = {
        "core_def": "equal-weight {1d_family_only, 4h_full_ironed} trend core (15m fine sleeve N/A here -- coarse-only pair)",
        "core_scorecard": core_card,
        "satellite_scorecard": sat_card,
        "combined_scorecard": comb_card,
        "core_on_overlap_scorecard": core_overlap_card,
        "core_sat_pearson": round(corr, 4) if corr is not None else None,
        "cross_tf_corr_1d_4h": round(xtf_corr, 4) if xtf_corr is not None else None,
        "blend": {"core_frac": RECO_CORE_FRAC, "sat_leverage": SAT_MAX_LEVERAGE, "overlap_days": int(len(df)) if df is not None else 0},
    }
    cs = core_card["per_split"]; ss = sat_card["per_split"]
    print(f"   CORE {{1d,4h}}: SEL {cs.get('SEL',{}).get('compound_pct')}% | OOS {cs.get('OOS',{}).get('compound_pct')}% | "
          f"UNSEEN {cs.get('UNSEEN',{}).get('compound_pct')}% | full p05 {core_card['full_block_bootstrap'].get('p05')} | "
          f"x-TF corr {xtf_corr}")
    print(f"   SAT (carry): SEL {ss.get('SEL',{}).get('compound_pct')}% | OOS {ss.get('OOS',{}).get('compound_pct')}% | "
          f"UNSEEN {ss.get('UNSEEN',{}).get('compound_pct')}% | core-sat corr {corr}")
    if comb_card is not None:
        cc = comb_card["per_split"]; co = core_overlap_card["per_split"]
        print(f"   COMBINED 70/30: SEL {cc.get('SEL',{}).get('compound_pct')}% | OOS {cc.get('OOS',{}).get('compound_pct')}% | "
              f"UNSEEN {cc.get('UNSEEN',{}).get('compound_pct')}% | full p05 {comb_card['full_block_bootstrap'].get('p05')}")
        print(f"   (core on same overlap: SEL {co.get('SEL',{}).get('compound_pct')}% | OOS {co.get('OOS',{}).get('compound_pct')}% | "
              f"UNSEEN {co.get('UNSEEN',{}).get('compound_pct')}%)")
    sys.stdout.flush()

    # ---- persist ----
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=str(ROOT.parent)).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "repro": {"command": "python -m strat.ironed_fullcycle_confirm", "git_sha": sha,
                  "cost_maker": MAKER_RT, "full_window": FULL_YEAR, "regime_fit_window": REGIME_FIT,
                  "canonical_splits": SPLITS, "specs": SPEC, "sat_max_leverage": SAT_MAX_LEVERAGE,
                  "reco_core_frac": RECO_CORE_FRAC, "bear_window": BEAR_2022, "generated": stamp,
                  "discipline": "PRE-REGISTERED specs verbatim; only data window + causal SEL-prefix "
                                "regime-fit changed; no re-tune; UNSEEN test-once; maker cost; causal lag-1"},
        "results": results,
    }
    p = OUT / "ironed_fullcycle_confirm.json"
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
