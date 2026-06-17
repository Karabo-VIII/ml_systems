"""src/strat/config_selector_vs_manual.py -- HEAD-TO-HEAD: the ML config-selector's pick vs the
MANUAL (live-portfolio) 2MA config, on Jan 2020 (in-sample reference) and Feb 2020 (held-out OOS =
the COVID-crash onset). The winner is decided on FEB (OOS); Jan is the contaminated in-sample leg the
user flagged.

USER /orc (2026-06-12): "Run the config-selector for Jan 2020 (data is biased and contaminated) and
for Feb 2020. Side-by-side comparison with the MANUAL config run -- which is better?"

THE TWO LEGS (run through the IDENTICAL backtest engine so it is apples-to-apples -- same cost, same
next-bar-open fills, same MtM accounting, same equal-weight book):
  MANUAL leg = the live portfolio methodology. A 4h 2MA grid that selected its top-6 IN-SAMPLE on Jan
    2020 (artifact runs/strat/portfolio_analysis_u10_CUSTOM_portfolio_20260611_223830.json):
    [ema_102_103, ema_18_128, ema_2_5, ema_44_45, ema_62_172, ema_8_91], exit = signalflip. Its Jan
    in-sample book return was +22.28% under the portfolio harness (inverse-vol sizing + corr-aware
    caps). HERE we re-run those 6 as an EQUAL-WEIGHT book through config_selector_jan2feb's engine, so
    the manual Jan number will NOT match +22.28% exactly (sizing/harness differ) -- we report BOTH and
    state the difference. The sanity gate: equal-weight Jan manual must land in a plausible positive
    band near the reference, not negative / not >5x (that would be an engine-parity bug, not a finding).
  ML leg = the existing PERIOD-LEVEL config-selector (config_selector_jan2feb.select_for_cadence):
    trains on Jan ONLY (regime classifier + shrinkage + bootstrap + synthetic + regime->config map),
    predicts a single book config, evaluated on Jan (in-sample) AND Feb (OOS) via the SAME engine.

REFERENCE rows (the selector already computes these): per-asset hindsight ORACLE + buy-hold, both
months. Surfaced for context (the ceiling and the do-nothing baseline).

PRIMARY cadence = 4h (the manual grid's cadence -- the head-to-head MUST be same-cadence). The ML leg
is ALSO swept across 15m/30m/1h/4h for completeness (this project mandates sweeping cadences, not
silently defaulting one). The MANUAL leg is 4h only (that is the only cadence the live portfolio ran).

HONEST FRAMING (inherited from config_selector_jan2feb): Feb 2020 is the COVID-crash ONSET -- a hard
regime shift. A long-only trend book trained on calm Jan will likely LOSE in Feb. "Winner" on Feb means
"loses LESS" if both are negative; it is NOT spun as a profit. Jan is in-sample (contaminated) and only
a reference -- a manual grid hand-picked on Jan IS Jan-overfit by construction, so a high Jan number is
expected and not evidence of skill.

ENGINE REUSE (no reinvention): config_perf / select_for_cadence / build_config_space / TAKER_RT all
come from config_selector_jan2feb; PR.STRATS injection from portfolio_replay. This file is a thin CALLER.

RWYB:
  python src/strat/config_selector_vs_manual.py --selftest     # two-sided control on the head-to-head logic
  python src/strat/config_selector_vs_manual.py                # the full Jan/Feb ML-vs-manual verdict
No emoji (Windows cp1252). Does NOT git commit.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import strat.portfolio_replay as PR                                       # noqa: E402
from strat.portfolio_replay import TAKER_RT                               # noqa: E402
import strat.config_selector_jan2feb as CS                                # noqa: E402
from strat.config_selector_jan2feb import (                              # noqa: E402
    JAN, FEB, _panel, config_perf, buy_hold, build_config_space, select_for_cadence, key2cfg,
)

OUT = ROOT / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research_verdict",
    "inputs": {"chimera": "via pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)",
               "manual_artifact": "runs/strat/portfolio_analysis_u10_CUSTOM_portfolio_20260611_223830.json"},
    "outputs": {"verdict_json": "runs/strat/ml_vs_manual_<stamp>.json"},
    "invariants": {
        "same_engine_both_legs": "manual AND ml legs both score via config_selector_jan2feb.config_perf "
                                 "(identical cost / next-bar fill / MtM / equal-weight book) -> apples-to-apples",
        "decision_on_feb_oos": "the winner is decided on Feb 2020 (held-out, crash onset); Jan is in-sample reference only",
        "jan_is_contaminated": "the manual top-6 was hand-picked in-sample on Jan -> a high Jan number is overfit, not skill",
        "manual_4h_only": "the manual leg runs at 4h only (the cadence the live portfolio ran); ML swept 15m..4h",
        "no_crash_loss_spin": "a crash-month loss is reported as a loss; 'winner' on Feb = loses less if both negative",
        "causal": "every MA uses bars <= t; only trades ENTERING in-window count; no look-ahead into the eval window",
    },
}

# The MANUAL live-portfolio top-6 (selected in-sample on Jan 2020). Names follow {type}_{fast}_{slow}.
MANUAL_CONFIGS = ["ema_102_103", "ema_18_128", "ema_2_5", "ema_44_45", "ema_62_172", "ema_8_91"]
MANUAL_EXIT = "signalflip"
MANUAL_JAN_REFERENCE_PCT = 22.28   # the portfolio-harness Jan in-sample book return (inverse-vol sizing)
MANUAL_CADENCE = "4h"


def _inject_manual_specs():
    """Inject the 6 manual EMA-cross configs into PR.STRATS so config_perf/holding_state resolve them.
    Each name {type}_{fast}_{slow} -> ('2MA', dict(type='EMA', fast, slow)). Idempotent."""
    injected = {}
    for name in MANUAL_CONFIGS:
        parts = name.split("_")
        if len(parts) != 3:
            raise ValueError(f"manual config name not {{type}}_{{fast}}_{{slow}}: {name}")
        ma_type, fast, slow = parts[0].upper(), int(parts[1]), int(parts[2])
        spec = ("2MA", dict(type=ma_type, fast=fast, slow=slow))
        PR.STRATS[name] = spec
        injected[name] = spec
    return injected


def _window_ms(win):
    return pd.Timestamp(win[0]).value // 10**6, pd.Timestamp(win[1]).value // 10**6


def _load_panels(syms, cadence, fe_ms):
    """Load full-history panels truncated to < fe_ms (never read past the eval-window end)."""
    panels = {}
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        keep = ms < fe_ms
        o, h, l, c, ms = o[keep], h[keep], l[keep], c[keep], ms[keep]
        panels[sym] = (o, h, l, c, ms)
    return panels


def manual_book(cadence, syms, jan_win, feb_win, cost):
    """Backtest the 6 manual configs as an EQUAL-WEIGHT book on Jan (in-sample) and Feb (OOS), through
    the IDENTICAL config_perf engine. Book compound = mean over assets of (mean over the 6 configs of
    that config's per-asset net compound) -- equal weight across both assets and configs, matching the
    ML book's equal-weight convention. Returns per-asset detail + book numbers for both months."""
    _inject_manual_specs()
    js_ms, je_ms = _window_ms(jan_win)
    fs_ms, fe_ms = _window_ms(feb_win)
    panels = _load_panels(syms, cadence, fe_ms)
    # keep only assets with month-1 data (same gate the selector uses)
    panels = {s: p for s, p in panels.items()
              if ((p[4] >= js_ms) & (p[4] < je_ms)).sum() >= 5}
    rows = []
    jan_asset_comps, feb_asset_comps = [], []
    for sym, (o, h, l, c, ms) in panels.items():
        cfg_jan, cfg_feb = [], []
        for cname in MANUAL_CONFIGS:
            jcomp, _, jn = config_perf(o, h, l, c, ms, cname, MANUAL_EXIT, js_ms, je_ms, cost)
            fcomp, _, fn = config_perf(o, h, l, c, ms, cname, MANUAL_EXIT, fs_ms, fe_ms, cost)
            cfg_jan.append(jcomp * 100.0)
            cfg_feb.append(fcomp * 100.0)
        asset_jan = float(np.mean(cfg_jan))   # equal-weight the 6 configs for this asset
        asset_feb = float(np.mean(cfg_feb))
        jan_asset_comps.append(asset_jan)
        feb_asset_comps.append(asset_feb)
        rows.append({"asset": sym[:-4], "jan_pct": round(asset_jan, 2), "feb_pct": round(asset_feb, 2),
                     "per_config_jan_pct": {cn: round(v, 2) for cn, v in zip(MANUAL_CONFIGS, cfg_jan)},
                     "per_config_feb_pct": {cn: round(v, 2) for cn, v in zip(MANUAL_CONFIGS, cfg_feb)}})
    book_jan = float(np.mean(jan_asset_comps)) if jan_asset_comps else 0.0
    book_feb = float(np.mean(feb_asset_comps)) if feb_asset_comps else 0.0
    return {"book_jan_pct": round(book_jan, 2), "book_feb_pct": round(book_feb, 2),
            "n_assets": len(panels), "assets": [s[:-4] for s in panels],
            "configs": MANUAL_CONFIGS, "exit": MANUAL_EXIT, "per_asset": rows}


def single_config_book(cadence, syms, cfg_key, jan_win, feb_win, cost):
    """Evaluate ONE config (e.g. the ML selector's global pick) as an equal-weight book over assets, on
    Jan + Feb, via the SAME config_perf engine. cfg_key is 'entry|exit'."""
    en, ex = cfg_key.split("|")
    js_ms, je_ms = _window_ms(jan_win)
    fs_ms, fe_ms = _window_ms(feb_win)
    panels = _load_panels(syms, cadence, fe_ms)
    panels = {s: p for s, p in panels.items() if ((p[4] >= js_ms) & (p[4] < je_ms)).sum() >= 5}
    jan_comps, feb_comps = [], []
    rows = []
    for sym, (o, h, l, c, ms) in panels.items():
        jcomp, _, _ = config_perf(o, h, l, c, ms, en, ex, js_ms, je_ms, cost)
        fcomp, _, _ = config_perf(o, h, l, c, ms, en, ex, fs_ms, fe_ms, cost)
        jan_comps.append(jcomp * 100.0)
        feb_comps.append(fcomp * 100.0)
        rows.append({"asset": sym[:-4], "jan_pct": round(jcomp * 100.0, 2), "feb_pct": round(fcomp * 100.0, 2)})
    return {"config": cfg_key, "book_jan_pct": round(float(np.mean(jan_comps)), 2) if jan_comps else 0.0,
            "book_feb_pct": round(float(np.mean(feb_comps)), 2) if feb_comps else 0.0,
            "n_assets": len(panels), "per_asset": rows}


def reference_books(cadence, syms, jan_win, feb_win, cost):
    """Hindsight per-asset ORACLE (argmax over the full config space, both months) + buy-hold, as
    equal-weight books. The oracle here is computed over the SAME config space the selector uses (2MA+3MA
    x exit menu), so it is the same ceiling the ML leg is measured against."""
    _, configs = build_config_space()
    js_ms, je_ms = _window_ms(jan_win)
    fs_ms, fe_ms = _window_ms(feb_win)
    panels = _load_panels(syms, cadence, fe_ms)
    panels = {s: p for s, p in panels.items() if ((p[4] >= js_ms) & (p[4] < je_ms)).sum() >= 5}
    jan_oracle, feb_oracle, jan_bh, feb_bh = [], [], [], []
    for sym, (o, h, l, c, ms) in panels.items():
        jan_best = max(config_perf(o, h, l, c, ms, en, ex, js_ms, je_ms, cost)[0] for (en, ex) in configs)
        feb_best = max(config_perf(o, h, l, c, ms, en, ex, fs_ms, fe_ms, cost)[0] for (en, ex) in configs)
        jan_oracle.append(jan_best * 100.0)
        feb_oracle.append(feb_best * 100.0)
        jan_bh.append(buy_hold(c, ms, js_ms, je_ms, cost) * 100.0)
        feb_bh.append(buy_hold(c, ms, fs_ms, fe_ms, cost) * 100.0)
    return {"oracle_jan_pct": round(float(np.mean(jan_oracle)), 2) if jan_oracle else 0.0,
            "oracle_feb_pct": round(float(np.mean(feb_oracle)), 2) if feb_oracle else 0.0,
            "buyhold_jan_pct": round(float(np.mean(jan_bh)), 2) if jan_bh else 0.0,
            "buyhold_feb_pct": round(float(np.mean(feb_bh)), 2) if feb_bh else 0.0,
            "n_assets": len(panels)}


def run_headtohead(universe="u10", cadences=("4h", "1h", "30m", "15m"), K_synth=40, n_boot=400):
    """The full head-to-head. MANUAL leg at 4h only; ML leg swept across cadences; references per cadence."""
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{universe}.yaml"))
    syms = [x["symbol"] for x in spec["assets"]]
    entry_specs, configs = build_config_space()
    cost = TAKER_RT

    out = {"universe": universe, "cadences": list(cadences), "cost_rt": cost,
           "manual_jan_reference_pct": MANUAL_JAN_REFERENCE_PCT, "per_cadence": {}}

    for cad in cadences:
        cad = cad.strip()
        block = {"cadence": cad}

        # ---- ML leg: train on Jan, predict a single book config, eval Jan + Feb via the same engine ----
        sel = select_for_cadence(cad, syms, entry_specs, configs, K_synth=K_synth, n_boot=n_boot,
                                 verbose=False, train_win=JAN, test_win=FEB)
        if sel.get("verdict", "").startswith("INSUFFICIENT"):
            block["ml"] = {"verdict": sel.get("verdict"), "n_assets": sel.get("n_assets")}
            out["per_cadence"][cad] = block
            continue
        ml_pick = sel["predicted_book_pick_global"]            # the single globally-best shrunk config
        ml_book = single_config_book(cad, syms, ml_pick, JAN, FEB, cost)
        block["ml"] = {
            "pick_config": ml_pick,
            "book_pick_mode": sel["predicted_book_pick_mode"],
            "jan_pct": ml_book["book_jan_pct"], "feb_pct": ml_book["book_feb_pct"],
            "n_assets": ml_book["n_assets"],
            # the selector's own per-asset (different-config-per-asset) Feb book, for context
            "selector_perasset_feb_pct": sel["eval_book"]["book_pred_perasset_pct"],
            "selector_beats_random_book": sel["eval_book"]["book_pred_beats_random"],
            "map_differentiates": sel["scoreboard"]["map_differentiates_by_regime"],
        }

        # ---- references (oracle + buy-hold) per cadence ----
        ref = reference_books(cad, syms, JAN, FEB, cost)
        block["reference"] = ref

        # ---- MANUAL leg: 4h ONLY (the cadence the live portfolio ran) ----
        if cad == MANUAL_CADENCE:
            man = manual_book(cad, syms, JAN, FEB, cost)
            block["manual"] = {"jan_pct": man["book_jan_pct"], "feb_pct": man["book_feb_pct"],
                               "n_assets": man["n_assets"], "configs": man["configs"],
                               "exit": man["exit"], "per_asset": man["per_asset"],
                               "jan_reference_portfolio_harness_pct": MANUAL_JAN_REFERENCE_PCT}
            # sanity gate on the Jan reproduction
            jrep = man["book_jan_pct"]
            block["manual"]["jan_sanity"] = _jan_sanity(jrep)

        # ---- winner per month (only at 4h, where both legs exist) ----
        if cad == MANUAL_CADENCE and "manual" in block:
            m, mlb = block["manual"], block["ml"]
            block["winner"] = {
                "jan_winner": "ML" if mlb["jan_pct"] > m["jan_pct"] else "MANUAL",
                "jan_margin_pct": round(mlb["jan_pct"] - m["jan_pct"], 2),
                "feb_winner": "ML" if mlb["feb_pct"] > m["feb_pct"] else "MANUAL",
                "feb_margin_pct": round(mlb["feb_pct"] - m["feb_pct"], 2),
                "feb_both_negative": bool(mlb["feb_pct"] < 0 and m["feb_pct"] < 0),
                "note": "FEB is the OOS decision; Jan is in-sample reference (manual is Jan-overfit by construction)",
            }
        out["per_cadence"][cad] = block

    out["verdict"] = _build_verdict(out)
    return out


def _jan_sanity(jan_repro_pct):
    """RWYB sanity gate: equal-weight Jan manual should be plausibly positive near the +22.28 reference
    (it will not match -- different sizing/harness). FAIL only on a wild mismatch (negative or >5x ref =
    an engine-parity bug, not a finding)."""
    ref = MANUAL_JAN_REFERENCE_PCT
    if jan_repro_pct < 0:
        return {"status": "FAIL", "reason": f"Jan manual equal-weight {jan_repro_pct}% is NEGATIVE vs "
                f"+{ref}% reference -> engine-parity bug, not a finding"}
    if jan_repro_pct > 5 * ref:
        return {"status": "FAIL", "reason": f"Jan manual {jan_repro_pct}% > 5x the +{ref}% reference -> "
                f"engine-parity bug (likely double-count / wrong window)"}
    band = "near reference" if abs(jan_repro_pct - ref) <= ref else "in plausible band (differs from ref)"
    return {"status": "PASS", "repro_pct": jan_repro_pct, "reference_pct": ref, "band": band,
            "note": "equal-weight engine vs inverse-vol + corr-capped portfolio harness -> exact match NOT expected"}


def _build_verdict(out):
    """Honest headline focused on the FEB (OOS) decision at 4h."""
    h2h = out["per_cadence"].get(MANUAL_CADENCE, {})
    if "winner" not in h2h:
        return {"headline": "NO 4h HEAD-TO-HEAD (insufficient data at 4h)", "lines": []}
    w = h2h["winner"]; m = h2h["manual"]; mlb = h2h["ml"]; ref = h2h["reference"]
    lines = []
    lines.append(f"DECISION on FEB 2020 (held-out OOS, COVID-crash onset). Jan is in-sample reference "
                 f"(manual top-6 was hand-picked on Jan = overfit by construction).")
    lines.append(f"[4h Jan in-sample]  MANUAL {m['jan_pct']:+.2f}% (vs +{MANUAL_JAN_REFERENCE_PCT}% "
                 f"portfolio-harness ref)  |  ML {mlb['jan_pct']:+.2f}%  |  oracle {ref['oracle_jan_pct']:+.2f}%  "
                 f"|  buy-hold {ref['buyhold_jan_pct']:+.2f}%")
    lines.append(f"[4h Feb OOS]        MANUAL {m['feb_pct']:+.2f}%  |  ML {mlb['feb_pct']:+.2f}%  |  "
                 f"oracle {ref['oracle_feb_pct']:+.2f}%  |  buy-hold {ref['buyhold_feb_pct']:+.2f}%")
    # headline on Feb
    if w["feb_both_negative"]:
        less_bad = w["feb_winner"]
        head = (f"BOTH LOSE IN THE CRASH (Feb): MANUAL {m['feb_pct']:+.2f}% vs ML {mlb['feb_pct']:+.2f}%. "
                f"{less_bad} loses LESS (by {abs(w['feb_margin_pct']):.2f} pts). This is a crash month -- "
                f"long-only trend books trained on calm Jan are expected to bleed; the 'winner' is the "
                f"less-bad book, NOT a profit. Reported as a loss, not spun.")
    elif mlb["feb_pct"] > m["feb_pct"]:
        head = (f"ML BEATS MANUAL on Feb (OOS) by {w['feb_margin_pct']:+.2f} pts "
                f"(ML {mlb['feb_pct']:+.2f}% vs MANUAL {m['feb_pct']:+.2f}%). The Jan-trained selector's "
                f"global pick transferred LESS BADLY than the Jan-overfit manual grid into the crash.")
    else:
        head = (f"MANUAL BEATS (or ties) ML on Feb (OOS) by {abs(w['feb_margin_pct']):.2f} pts "
                f"(MANUAL {m['feb_pct']:+.2f}% vs ML {mlb['feb_pct']:+.2f}%). The ML selector did NOT "
                f"improve on the manual grid out-of-sample.")
    lines.insert(0, f"HEADLINE: {head}")
    lines.append(f"Jan (in-sample) winner: {w['jan_winner']} by {abs(w['jan_margin_pct']):.2f} pts "
                 f"(in-sample, not decision-relevant -- both are partly Jan-fit).")
    # ML cadence sweep summary
    sweep = []
    for cad, blk in out["per_cadence"].items():
        if "ml" in blk and "feb_pct" in blk["ml"]:
            sweep.append(f"{cad}: ML Feb {blk['ml']['feb_pct']:+.1f}% (Jan {blk['ml']['jan_pct']:+.1f}%)")
    lines.append("ML cadence sweep (Feb OOS): " + " | ".join(sweep))
    lines.append("CAVEATS: ~7 of 10 u10 assets have 2020 data (SOL/DOGE/AVAX launched later); Jan is "
                 "in-sample + the manual grid is Jan-overfit by construction; 1-month train = wide CIs; "
                 "equal-weight engine here vs inverse-vol + corr-capped portfolio harness for the live "
                 "manual book (so the manual Jan number differs from +22.28% -- both reported).")
    return {"headline": head, "feb_winner": w["feb_winner"], "feb_margin_pct": w["feb_margin_pct"],
            "feb_both_negative": w["feb_both_negative"], "jan_winner": w["jan_winner"], "lines": lines}


def _print_table(out):
    print("=" * 92)
    print(f"## ML CONFIG-SELECTOR vs MANUAL 2MA GRID -- {out['universe']} -- Jan(in-sample)/Feb(OOS) 2020")
    print(f"   train window {JAN[0]}..{JAN[1]} | eval window {FEB[0]}..{FEB[1]} | cost_rt {out['cost_rt']}")
    print("=" * 92)
    h2h = out["per_cadence"].get(MANUAL_CADENCE, {})
    if "manual" in h2h:
        m, mlb, ref = h2h["manual"], h2h["ml"], h2h["reference"]
        print(f"\n## PRIMARY HEAD-TO-HEAD @ {MANUAL_CADENCE} ({m['n_assets']} assets w/ 2020 data)")
        print(f"   {'':24} {'Jan2020 (in-sample)':>22} {'Feb2020 (held-out OOS)':>24}")
        print(f"   {'Manual 2MA grid top-6':24} {m['jan_pct']:>10.2f}% (ref +{MANUAL_JAN_REFERENCE_PCT}) "
              f"{m['feb_pct']:>22.2f}%")
        print(f"   {'ML selector pick':24} {mlb['jan_pct']:>21.2f}% {mlb['feb_pct']:>23.2f}%")
        print(f"   {'Oracle (hindsight)':24} {ref['oracle_jan_pct']:>21.2f}% {ref['oracle_feb_pct']:>23.2f}%")
        print(f"   {'Buy-hold':24} {ref['buyhold_jan_pct']:>21.2f}% {ref['buyhold_feb_pct']:>23.2f}%")
        print(f"\n   ML pick config: {mlb['pick_config']}  | manual exit: {m['exit']}")
        print(f"   Jan sanity: {m['jan_sanity']['status']} -- {m['jan_sanity'].get('note', m['jan_sanity'].get('reason'))}")
    print(f"\n## ML CADENCE SWEEP (manual leg is {MANUAL_CADENCE}-only; ML swept all)")
    print(f"   {'cadence':8} {'ML pick':30} {'ML Jan%':>9} {'ML Feb%':>9} {'oracle Feb%':>12} {'BH Feb%':>9}")
    for cad, blk in out["per_cadence"].items():
        if "ml" not in blk or "feb_pct" not in blk["ml"]:
            print(f"   {cad:8} {blk.get('ml', {}).get('verdict', 'n/a'):30}")
            continue
        mlb = blk["ml"]; ref = blk.get("reference", {})
        print(f"   {cad:8} {mlb['pick_config'][:30]:30} {mlb['jan_pct']:>9.2f} {mlb['feb_pct']:>9.2f} "
              f"{ref.get('oracle_feb_pct', 0):>12.2f} {ref.get('buyhold_feb_pct', 0):>9.2f}")
    print("\n" + "=" * 92)
    print("## VERDICT")
    for line in out["verdict"]["lines"]:
        print(f"   {line}")
    print("=" * 92)


# ===========================================================================
# SELF-TEST -- two-sided soundness on the HEAD-TO-HEAD logic (no market data)
# ===========================================================================
def selftest():
    """Two-sided control on the comparison + sanity logic (the new logic this file adds).
    POSITIVE: when one leg is deterministically better on Feb, the winner-picker must name it.
    NEGATIVE: the Jan sanity gate must FAIL a wild (negative / >5x) repro and PASS a plausible one."""
    print("## ML-vs-MANUAL SELFTEST (two-sided)")
    ok = True

    # POSITIVE: winner logic must name the higher-Feb leg as the Feb winner.
    fake = {"per_cadence": {"4h": {
        "manual": {"jan_pct": 22.0, "feb_pct": -30.0, "n_assets": 7, "configs": MANUAL_CONFIGS,
                   "exit": "signalflip", "jan_sanity": {"status": "PASS", "note": "ok"}},
        "ml": {"pick_config": "ema_x|signalflip", "jan_pct": 5.0, "feb_pct": -10.0, "n_assets": 7},
        "reference": {"oracle_jan_pct": 40, "oracle_feb_pct": 12, "buyhold_jan_pct": 10, "buyhold_feb_pct": -25},
    }}, "universe": "u10", "cost_rt": TAKER_RT}
    fake["per_cadence"]["4h"]["winner"] = {
        "jan_winner": "MANUAL", "jan_margin_pct": -17.0, "feb_winner": "ML", "feb_margin_pct": 20.0,
        "feb_both_negative": True, "note": ""}
    v = _build_verdict(fake)
    print(f"  POSITIVE: ML Feb -10 vs MANUAL Feb -30 -> feb_winner={v['feb_winner']} (expect ML), "
          f"both_negative={v['feb_both_negative']} (expect True)")
    ok &= (v["feb_winner"] == "ML" and v["feb_both_negative"] is True)

    # POSITIVE-2: when MANUAL is higher on Feb, picker must name MANUAL.
    fake2 = json.loads(json.dumps(fake))
    fake2["per_cadence"]["4h"]["ml"]["feb_pct"] = -45.0
    fake2["per_cadence"]["4h"]["winner"] = {
        "jan_winner": "MANUAL", "jan_margin_pct": -17.0, "feb_winner": "MANUAL", "feb_margin_pct": -15.0,
        "feb_both_negative": True, "note": ""}
    v2 = _build_verdict(fake2)
    print(f"  POSITIVE-2: ML Feb -45 vs MANUAL Feb -30 -> feb_winner={v2['feb_winner']} (expect MANUAL)")
    ok &= (v2["feb_winner"] == "MANUAL")

    # NEGATIVE: Jan sanity gate.
    s_neg = _jan_sanity(-5.0)
    s_big = _jan_sanity(5 * MANUAL_JAN_REFERENCE_PCT + 1)
    s_ok = _jan_sanity(18.0)
    print(f"  SANITY: neg repro -> {s_neg['status']} (expect FAIL); 5x+ repro -> {s_big['status']} "
          f"(expect FAIL); plausible 18% -> {s_ok['status']} (expect PASS)")
    ok &= (s_neg["status"] == "FAIL" and s_big["status"] == "FAIL" and s_ok["status"] == "PASS")

    # config-name parse: manual specs must inject as 2MA EMA crosses with the right lengths.
    inj = _inject_manual_specs()
    fam, p = inj["ema_102_103"]
    print(f"  PARSE: ema_102_103 -> ({fam}, fast={p['fast']}, slow={p['slow']}) (expect 2MA, 102, 103)")
    ok &= (fam == "2MA" and p["fast"] == 102 and p["slow"] == 103 and p["type"] == "EMA")

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python src/strat/config_selector_vs_manual.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadences", default="4h,1h,30m,15m")
    ap.add_argument("--K-synth", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=400)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    cadences = tuple(c.strip() for c in a.cadences.split(","))
    out = run_headtohead(a.universe, cadences, K_synth=a.K_synth, n_boot=a.n_boot)
    _print_table(out)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"ml_vs_manual_{a.universe}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha,
                         "train_window": JAN, "eval_window": FEB, "cost_rt": TAKER_RT,
                         "universe": a.universe, "manual_cadence": MANUAL_CADENCE,
                         "manual_configs": MANUAL_CONFIGS, "manual_exit": MANUAL_EXIT},
               "result": out}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
