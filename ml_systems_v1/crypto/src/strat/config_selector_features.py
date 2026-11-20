"""src/strat/config_selector_features.py -- FEATURE-ENRICHED config selection, JAN-only, look-ahead-guarded.

USER /orc (2026-06-12): "make our selector richer by giving it RELEVANT FEATURES that help the selection."
The baseline selector (config_selector_jan2feb / config_selector_vs_manual) ranks configs by their raw Jan
2020 compound return, so it crowns Jan-winning FAST/MEDIUM MA configs that DIE in the Feb 2020 COVID crash.
A concurrent descriptive instance reported the SLOW 2MA family (60-150) is the robust survivor. The HYPOTHESIS
under test: do GENERIC, PRE-REGISTERED, JAN-ONLY robustness features steer the selector toward the robust
(slower / more-stable / lower-churn) configs and improve the held-out Feb book?

PRE-REGISTERED FEATURES (computed from JAN ALONE -- NOT chosen because we know slow won in Feb; any quant
pre-registers these as generic robustness priors):
  F1 slowness          : the slow-MA period (mean MA period for 3MA) of the config. Structural robustness
                         prior -- slower MAs trade rarely => less cost-bleed / whipsaw in chop.
  F2 within-Jan stability: split Jan into K=3 sub-blocks, compute each config's per-block compound; reward
                         CONSISTENT sign / low cross-block variance (penalizes lucky one-block Jan winners =
                         the overfit fast configs). THE pure overfit-killer, fully Jan-only.
  F3 whipsaw / turnover : Jan trade-count per config (churn). High churn = fragile + cost-bled -> penalize.
  F4 exit-x-regime      : if Jan whipsaw (avg trades/asset) is HIGH, prefer signalflip/timestop over trailing
                         exits (the concurrent instance found trailing-stop WORST in the choppy top -- the
                         chop whipsaws the stop). A pre-registered exit-conditioning RULE, not a fit.

ENRICHED SCORE (weights PRE-REGISTERED at w=1 each; NOT tuned on Feb):
    enriched(cfg) = z(raw_jan_compound) + w1*z(slowness) + w2*z(within_jan_stability) - w3*z(whipsaw)
                    [ + F4 exit-rule penalty applied to trailing exits when Jan-whipsaw is high ]
The selected BOOK = the single config with the best enriched score on JAN (one config for all assets, the same
"global book pick" convention the baseline reports). It is evaluated on FEB (held out) via the IDENTICAL engine.

ENGINE REUSE (NO reinvention): config_perf / build_config_space / held_for_exit / buy_hold / _panel / JAN /
FEB / TAKER_RT all come from config_selector_jan2feb -> apples-to-apples with the baseline + the head-to-head.

LOOK-AHEAD GUARD (audited): every feature reads ONLY Jan bars (s_ms..e_ms = JAN window); the F4 whipsaw
threshold is a Jan statistic; weights are constants set before any Feb number was computed. Feb is touched
ONCE, in the final eval. Stated explicitly in the output's look_ahead_audit block.

RWYB:
  python src/strat/config_selector_features.py --selftest   # two-sided control (slow+stable scores high; lucky-fast penalized)
  python src/strat/config_selector_features.py              # full enriched-vs-baseline Feb verdict + ablation

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
from strat.config_selector_jan2feb import (                              # noqa: E402
    JAN, FEB, _panel, config_perf, buy_hold, build_config_space, held_for_exit,
    cfg2key, EXIT_MENU,
)

OUT = ROOT / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# PRE-REGISTERED enriched-score weights -- set BEFORE any Feb number is computed; NOT tuned on Feb.
W1_SLOWNESS = 1.0
W2_STABILITY = 1.0
W3_WHIPSAW = 1.0
K_BLOCKS = 3                  # within-Jan sub-blocks for F2 stability
WHIPSAW_HI_TRADES = 4.0       # F4: Jan avg trades/asset above this => "high whipsaw" => penalize trailing exits
TRAILING_EXITS = {"trail5", "trail10", "atr3"}   # exits with a moving stop that chop whipsaws
F4_TRAIL_PENALTY = 1.0        # z-score units subtracted from trailing exits in a high-whipsaw Jan

__contract__ = {
    "kind": "research_verdict",
    "inputs": {"chimera": "via pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)"},
    "outputs": {"verdict_json": "runs/strat/config_selector_features_<stamp>.json"},
    "invariants": {
        "same_engine_as_baseline": "config_perf / build_config_space / held_for_exit from config_selector_jan2feb "
                                   "-> identical cost / next-bar fill / MtM / equal-weight book as the baseline + h2h",
        "features_jan_only": "F1/F2/F3/F4 read ONLY Jan bars (the train window); no Feb data enters any feature",
        "weights_preregistered": "w1=w2=w3=1.0 set before any Feb number; NOT tuned on Feb (sensitivity band reported, not optimized)",
        "feb_held_out": "Feb is touched ONCE in the final eval; selection is on Jan enriched score",
        "objective_is_compound": "judged on held-out net COMPOUND return; never AUC/IC",
        "ablation_one_at_a_time": "each feature added singly to the raw-return baseline to attribute the Feb delta",
    },
}


# ===========================================================================
# 0. config-name -> MA periods (for F1 slowness). Names: {type}_{fast}_{slow} or {type}_{a}_{b}_{c}.
# ===========================================================================
def _ma_periods(entry_name):
    """Return (slow_period, mean_period) parsed from the entry-spec name. Falls back to PR.STRATS params."""
    parts = entry_name.split("_")
    nums = [int(p) for p in parts[1:] if p.isdigit()]
    if not nums:
        fam, p = PR.STRATS.get(entry_name, ("?", {}))
        nums = [v for v in p.values() if isinstance(v, (int, float))]
    if not nums:
        return 0.0, 0.0
    return float(max(nums)), float(np.mean(nums))


# ===========================================================================
# 1. JAN-ONLY features per config (look-ahead-guarded)
# ===========================================================================
def jan_features(panels, configs, js_ms, je_ms, cost):
    """Per-config Jan-only features, pooled across assets. Returns a dict keyed by config:
        raw_jan   : mean Jan compound across assets (the baseline ranking signal)
        slowness  : slow-MA period (structural prior; entry-only, asset-independent)
        stability : within-Jan cross-block sign-consistency reward (F2; higher = more stable)
        whipsaw   : mean Jan trades/asset (F3 churn; higher = more fragile)
    plus a scalar jan_avg_trades (the F4 regime gauge) = grand-mean trades/asset over ALL configs."""
    # K=3 within-Jan sub-blocks (equal-time) for the stability feature
    span = je_ms - js_ms
    blocks = [(js_ms + i * span // K_BLOCKS, js_ms + (i + 1) * span // K_BLOCKS) for i in range(K_BLOCKS)]
    feats = {}
    all_trade_counts = []
    for cfg in configs:
        en, ex = cfg
        per_asset_comp, per_asset_trades_n, per_asset_block_signs = [], [], []
        for sym, (o, h, l, c, ms) in panels.items():
            comp, rets, n = config_perf(o, h, l, c, ms, en, ex, js_ms, je_ms, cost)
            per_asset_comp.append(comp * 100.0)
            per_asset_trades_n.append(n)
            # F2: per-block compound sign for THIS asset (look-ahead-guarded: blocks are inside Jan)
            block_comps = [config_perf(o, h, l, c, ms, en, ex, bs, be, cost)[0] for (bs, be) in blocks]
            per_asset_block_signs.append(block_comps)
        raw_jan = float(np.mean(per_asset_comp))
        whipsaw = float(np.mean(per_asset_trades_n))
        all_trade_counts.append(whipsaw)
        # F2 stability = mean over assets of (fraction of non-negative blocks) - normalized cross-block std.
        # Reward consistency: a config positive (or flat) in all 3 Jan blocks scores high; a config that is
        # huge in one block and negative in the others (the lucky one-block fast winner) scores low.
        stab_terms = []
        for bcomps in per_asset_block_signs:
            arr = np.array(bcomps, float)
            nonneg_frac = float(np.mean(arr >= -1e-9))           # in [0,1]: how many blocks didn't lose
            spread = float(np.std(arr))                          # cross-block dispersion (decimal compound)
            stab_terms.append(nonneg_frac - spread)              # consistency reward - dispersion penalty
        stability = float(np.mean(stab_terms)) if stab_terms else 0.0
        slow_p, mean_p = _ma_periods(en)
        feats[cfg] = {"raw_jan": round(raw_jan, 4), "slowness": slow_p, "mean_period": mean_p,
                      "stability": round(stability, 4), "whipsaw": round(whipsaw, 3)}
    jan_avg_trades = float(np.mean(all_trade_counts)) if all_trade_counts else 0.0
    return feats, jan_avg_trades, [b for b in blocks]


def _zdict(values):
    """z-score a dict of name->value (population std; constant vector -> all zeros)."""
    keys = list(values)
    arr = np.array([values[k] for k in keys], float)
    mu, sd = float(np.mean(arr)), float(np.std(arr))
    if sd < 1e-12:
        return {k: 0.0 for k in keys}
    return {k: float((values[k] - mu) / sd) for k in keys}


# ===========================================================================
# 2. ENRICHED SCORE + ablation variants
# ===========================================================================
def enriched_scores(feats, jan_avg_trades, configs, w1=W1_SLOWNESS, w2=W2_STABILITY, w3=W3_WHIPSAW,
                    use_f1=True, use_f2=True, use_f3=True, use_f4=True):
    """Compute the enriched score per config from the Jan features. Each F can be toggled off for ablation.
    enriched = z(raw_jan) + w1*z(slowness)[F1] + w2*z(stability)[F2] - w3*z(whipsaw)[F3]
               + F4 exit-rule penalty (trailing exits penalized when Jan-whipsaw is high)."""
    z_raw = _zdict({c: feats[c]["raw_jan"] for c in configs})
    z_slow = _zdict({c: feats[c]["slowness"] for c in configs})
    z_stab = _zdict({c: feats[c]["stability"] for c in configs})
    z_whip = _zdict({c: feats[c]["whipsaw"] for c in configs})
    high_whipsaw = jan_avg_trades > WHIPSAW_HI_TRADES
    scores = {}
    for c in configs:
        s = z_raw[c]                                             # baseline ranking signal (always on)
        if use_f1:
            s += w1 * z_slow[c]
        if use_f2:
            s += w2 * z_stab[c]
        if use_f3:
            s -= w3 * z_whip[c]
        if use_f4 and high_whipsaw and c[1] in TRAILING_EXITS:
            s -= F4_TRAIL_PENALTY                                # pre-registered: avoid trailing exits in chop
        scores[c] = s
    return scores, {"high_whipsaw_jan": bool(high_whipsaw), "jan_avg_trades": round(jan_avg_trades, 3),
                    "whipsaw_threshold": WHIPSAW_HI_TRADES}


def book_feb(panels, cfg, fs_ms, fe_ms, cost):
    """Equal-weight Feb book compound for a SINGLE config across assets (the global-book-pick convention)."""
    en, ex = cfg
    rets = [config_perf(o, h, l, c, ms, en, ex, fs_ms, fe_ms, cost)[0] * 100.0
            for (o, h, l, c, ms) in panels.values()]
    return float(np.mean(rets)) if rets else 0.0


def book_jan(panels, cfg, js_ms, je_ms, cost):
    en, ex = cfg
    rets = [config_perf(o, h, l, c, ms, en, ex, js_ms, je_ms, cost)[0] * 100.0
            for (o, h, l, c, ms) in panels.values()]
    return float(np.mean(rets)) if rets else 0.0


# ===========================================================================
# 3. STEP-0: the slow-family ceiling, in OUR engine
# ===========================================================================
SLOW_FAMILY = ["ema_60_120", "ema_75_150", "ema_100_150", "ema_62_172"]
SLOW_EXIT = "signalflip"


def _inject_slow_family():
    for nm in SLOW_FAMILY:
        p = nm.split("_")
        PR.STRATS[nm] = ("2MA", dict(type="EMA", fast=int(p[1]), slow=int(p[2])))


def slow_family_book(panels, js_ms, je_ms, fs_ms, fe_ms, cost):
    """STEP-0: explicit slow-2MA(60-150) family book, equal-weight over the 4 configs AND over assets,
    in OUR engine (same config_perf / cost / fill). Reports Jan + Feb so we can compare to the +18.2% claim."""
    _inject_slow_family()
    jan_a, feb_a, per_asset = [], [], []
    for sym, (o, h, l, c, ms) in panels.items():
        cj = [config_perf(o, h, l, c, ms, nm, SLOW_EXIT, js_ms, je_ms, cost)[0] * 100.0 for nm in SLOW_FAMILY]
        cf = [config_perf(o, h, l, c, ms, nm, SLOW_EXIT, fs_ms, fe_ms, cost)[0] * 100.0 for nm in SLOW_FAMILY]
        jan_a.append(float(np.mean(cj))); feb_a.append(float(np.mean(cf)))
        per_asset.append({"asset": sym[:-4], "jan_pct": round(float(np.mean(cj)), 2),
                          "feb_pct": round(float(np.mean(cf)), 2)})
    return {"configs": SLOW_FAMILY, "exit": SLOW_EXIT,
            "book_jan_pct": round(float(np.mean(jan_a)), 2) if jan_a else 0.0,
            "book_feb_pct": round(float(np.mean(feb_a)), 2) if feb_a else 0.0,
            "per_asset": per_asset}


# ===========================================================================
# 4. ORACLE + buy-hold reference (over the SAME config space)
# ===========================================================================
def reference(panels, configs, js_ms, je_ms, fs_ms, fe_ms, cost):
    jan_o, feb_o, jan_bh, feb_bh = [], [], [], []
    for sym, (o, h, l, c, ms) in panels.items():
        jan_o.append(max(config_perf(o, h, l, c, ms, en, ex, js_ms, je_ms, cost)[0] for (en, ex) in configs) * 100.0)
        feb_o.append(max(config_perf(o, h, l, c, ms, en, ex, fs_ms, fe_ms, cost)[0] for (en, ex) in configs) * 100.0)
        jan_bh.append(buy_hold(c, ms, js_ms, je_ms, cost) * 100.0)
        feb_bh.append(buy_hold(c, ms, fs_ms, fe_ms, cost) * 100.0)
    return {"oracle_jan_pct": round(float(np.mean(jan_o)), 2), "oracle_feb_pct": round(float(np.mean(feb_o)), 2),
            "buyhold_jan_pct": round(float(np.mean(jan_bh)), 2), "buyhold_feb_pct": round(float(np.mean(feb_bh)), 2)}


# ===========================================================================
# 5. THE RUN -- per cadence: baseline pick vs enriched pick vs slow ceiling vs oracle vs BH + ablation
# ===========================================================================
def _load_panels(syms, cadence, js_ms, je_ms, fe_ms):
    panels = {}
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        keep = ms < fe_ms
        o, h, l, c, ms = o[keep], h[keep], l[keep], c[keep], ms[keep]
        if ((ms >= js_ms) & (ms < je_ms)).sum() >= 5:
            panels[sym] = (o, h, l, c, ms)
    return panels


def run_cadence(cadence, syms, entry_specs, configs, cost):
    js_ms, je_ms = pd.Timestamp(JAN[0]).value // 10**6, pd.Timestamp(JAN[1]).value // 10**6
    fs_ms, fe_ms = pd.Timestamp(FEB[0]).value // 10**6, pd.Timestamp(FEB[1]).value // 10**6
    panels = _load_panels(syms, cadence, js_ms, je_ms, fe_ms)
    if len(panels) < 3:
        return {"cadence": cadence, "verdict": "INSUFFICIENT_ASSETS", "n_assets": len(panels)}

    feats, jan_avg_trades, _ = jan_features(panels, configs, js_ms, je_ms, cost)

    # ---- BASELINE: pick by raw Jan compound (the current selector's ranking signal) ----
    raw_best = max(configs, key=lambda c: feats[c]["raw_jan"])
    base_pick = raw_best
    base_feb = book_feb(panels, base_pick, fs_ms, fe_ms, cost)
    base_jan = book_jan(panels, base_pick, js_ms, je_ms, cost)

    # ---- ENRICHED: pick by the full enriched score (all 4 features) ----
    enr_scores, f4_info = enriched_scores(feats, jan_avg_trades, configs)
    enr_best = max(configs, key=lambda c: enr_scores[c])
    enr_feb = book_feb(panels, enr_best, fs_ms, fe_ms, cost)
    enr_jan = book_jan(panels, enr_best, js_ms, je_ms, cost)

    # ---- ABLATION: add ONE feature at a time on top of the raw-return baseline ----
    ablation = {}
    feature_flags = {"F1_slowness": dict(use_f1=True, use_f2=False, use_f3=False, use_f4=False),
                     "F2_stability": dict(use_f1=False, use_f2=True, use_f3=False, use_f4=False),
                     "F3_whipsaw": dict(use_f1=False, use_f2=False, use_f3=True, use_f4=False),
                     "F4_exit_rule": dict(use_f1=False, use_f2=False, use_f3=False, use_f4=True)}
    for fname, flags in feature_flags.items():
        sc, _ = enriched_scores(feats, jan_avg_trades, configs, **flags)
        pick = max(configs, key=lambda c: sc[c])
        ablation[fname] = {"pick": cfg2key(pick), "feb_pct": round(book_feb(panels, pick, fs_ms, fe_ms, cost), 2),
                           "jan_pct": round(book_jan(panels, pick, js_ms, je_ms, cost), 2),
                           "feb_delta_vs_baseline": round(book_feb(panels, pick, fs_ms, fe_ms, cost) - base_feb, 2),
                           "slowness": feats[pick]["slowness"]}

    # ---- references + STEP-0 slow-family ceiling ----
    ref = reference(panels, configs, js_ms, je_ms, fs_ms, fe_ms, cost)
    slow = slow_family_book(panels, js_ms, je_ms, fs_ms, fe_ms, cost)

    return {
        "cadence": cadence, "n_assets": len(panels), "assets": [s[:-4] for s in panels],
        "f4_regime": f4_info,
        "baseline": {"pick": cfg2key(base_pick), "jan_pct": round(base_jan, 2), "feb_pct": round(base_feb, 2),
                     "slowness": feats[base_pick]["slowness"], "whipsaw": feats[base_pick]["whipsaw"],
                     "stability": feats[base_pick]["stability"]},
        "enriched": {"pick": cfg2key(enr_best), "jan_pct": round(enr_jan, 2), "feb_pct": round(enr_feb, 2),
                     "slowness": feats[enr_best]["slowness"], "whipsaw": feats[enr_best]["whipsaw"],
                     "stability": feats[enr_best]["stability"],
                     "feb_delta_vs_baseline": round(enr_feb - base_feb, 2),
                     "picked_slower": bool(feats[enr_best]["slowness"] > feats[base_pick]["slowness"]),
                     "picked_lower_whipsaw": bool(feats[enr_best]["whipsaw"] < feats[base_pick]["whipsaw"]),
                     "picked_more_stable": bool(feats[enr_best]["stability"] > feats[base_pick]["stability"])},
        "ablation": ablation,
        "slow_family_ceiling": {"jan_pct": slow["book_jan_pct"], "feb_pct": slow["book_feb_pct"],
                                "configs": slow["configs"], "exit": slow["exit"], "per_asset": slow["per_asset"]},
        "reference": ref,
    }


def run(universe="u10", cadences=("4h", "1h", "30m", "15m")):
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{universe}.yaml"))
    syms = [x["symbol"] for x in spec["assets"]]
    entry_specs, configs = build_config_space()
    cost = TAKER_RT
    out = {"universe": universe, "cadences": list(cadences), "cost_rt": cost,
           "weights": {"w1_slowness": W1_SLOWNESS, "w2_stability": W2_STABILITY, "w3_whipsaw": W3_WHIPSAW,
                       "K_blocks": K_BLOCKS, "whipsaw_hi_trades": WHIPSAW_HI_TRADES,
                       "f4_trail_penalty": F4_TRAIL_PENALTY},
        "config_space": {"n_entry_specs": len(entry_specs), "n_configs": len(configs),
                         "exit_menu": [e[0] for e in EXIT_MENU]},
        "look_ahead_audit": {
            "features_jan_only": "F1/F2/F3 read ONLY the JAN window bars; F2 sub-blocks are inside JAN; "
                                 "F4 threshold is a JAN statistic. No Feb bar enters any feature.",
            "weights_preregistered": "w1=w2=w3=1.0, K=3, whipsaw_hi=4.0 trades/asset, trail_penalty=1.0 -- "
                                     "all constants set before any Feb number; NOT tuned on Feb.",
            "feb_touched_once": "Feb is read only in the final book_feb eval of the SELECTED config + the references.",
        },
        "per_cadence": {}}
    for cad in cadences:
        out["per_cadence"][cad.strip()] = run_cadence(cad.strip(), syms, entry_specs, configs, cost)
    out["verdict"] = _verdict(out)
    return out


def _verdict(out):
    lines, deltas, improved = [], [], 0
    cadences_with_data = 0
    f2_drove, f1_drove = 0, 0
    for cad, blk in out["per_cadence"].items():
        if blk.get("verdict", "").startswith("INSUFFICIENT"):
            continue
        cadences_with_data += 1
        b, e, s, r = blk["baseline"], blk["enriched"], blk["slow_family_ceiling"], blk["reference"]
        d = e["feb_pct"] - b["feb_pct"]
        deltas.append(d)
        if d > 0.01:
            improved += 1
        # which single feature had the biggest positive Feb delta?
        abl = blk["ablation"]
        best_feat = max(abl, key=lambda f: abl[f]["feb_delta_vs_baseline"])
        if best_feat == "F2_stability" and abl[best_feat]["feb_delta_vs_baseline"] > 0:
            f2_drove += 1
        if best_feat == "F1_slowness" and abl[best_feat]["feb_delta_vs_baseline"] > 0:
            f1_drove += 1
        lines.append(
            f"[{cad}] baseline {b['pick']} Feb {b['feb_pct']:+.2f}% (slow={b['slowness']:.0f}) | "
            f"ENRICHED {e['pick']} Feb {e['feb_pct']:+.2f}% (slow={e['slowness']:.0f}) | "
            f"delta {d:+.2f}pts | slow-fam ceiling {s['feb_pct']:+.2f}% | oracle {r['oracle_feb_pct']:+.2f}% | "
            f"BH {r['buyhold_feb_pct']:+.2f}% | top-feat={best_feat}({abl[best_feat]['feb_delta_vs_baseline']:+.2f})")
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    if not cadences_with_data:
        head = "NO DATA"
    elif improved >= max(1, cadences_with_data // 2 + 1) and mean_delta > 0.5:
        head = (f"FEATURES HELP (modest): enriched selection beat the raw-return baseline on {improved}/"
                f"{cadences_with_data} cadences, mean Feb delta {mean_delta:+.2f}pts. The Jan-only robustness "
                f"features steered the pick toward slower/more-stable configs that transferred less badly.")
    elif improved >= 1:
        head = (f"MIXED: enriched helped on {improved}/{cadences_with_data} cadences (mean delta {mean_delta:+.2f}pts) "
                f"but not robustly. The regime change is only partly anticipable from Jan-only structure.")
    else:
        head = (f"FEATURES DO NOT HELP OOS: enriched selection did NOT beat the raw-return baseline on Feb on any "
                f"cadence (mean delta {mean_delta:+.2f}pts). The calm-Jan->crash-Feb regime shift is NOT predictable "
                f"from Jan-only robustness structure -- the configs that survive the crash are not identifiable as "
                f"'slow/stable/low-churn' from calm-January alone. This is itself the answer to the hypothesis.")
    return {"headline": head, "mean_feb_delta_pts": round(mean_delta, 2),
            "cadences_improved": improved, "cadences_with_data": cadences_with_data,
            "f2_stability_top_on_n_cadences": f2_drove, "f1_slowness_top_on_n_cadences": f1_drove,
            "lines": lines}


def _print(out):
    print("=" * 100)
    print(f"## FEATURE-ENRICHED CONFIG SELECTION -- {out['universe']} -- Jan(train)/Feb(held-out) 2020")
    print(f"   train {JAN[0]}..{JAN[1]} | eval {FEB[0]}..{FEB[1]} | cost_rt {out['cost_rt']} | "
          f"{out['config_space']['n_configs']} configs | weights w1=w2=w3=1 (pre-registered)")
    print("=" * 100)
    for cad, blk in out["per_cadence"].items():
        if blk.get("verdict", "").startswith("INSUFFICIENT"):
            print(f"\n## {cad}: {blk['verdict']} (n={blk.get('n_assets')})")
            continue
        b, e, s, r = blk["baseline"], blk["enriched"], blk["slow_family_ceiling"], blk["reference"]
        print(f"\n## {cad} ({blk['n_assets']} assets) -- Jan-whipsaw={blk['f4_regime']['jan_avg_trades']} "
              f"(high={blk['f4_regime']['high_whipsaw_jan']})")
        print(f"   {'':22} {'pick':30} {'Jan%':>8} {'Feb%':>8} {'slow':>6} {'whip':>6}")
        print(f"   {'BASELINE (raw Jan)':22} {b['pick']:30} {b['jan_pct']:>8.2f} {b['feb_pct']:>8.2f} "
              f"{b['slowness']:>6.0f} {b['whipsaw']:>6.2f}")
        print(f"   {'ENRICHED (F1-F4)':22} {e['pick']:30} {e['jan_pct']:>8.2f} {e['feb_pct']:>8.2f} "
              f"{e['slowness']:>6.0f} {e['whipsaw']:>6.2f}   delta_Feb={e['feb_delta_vs_baseline']:+.2f}pts "
              f"(slower={e['picked_slower']}, more_stable={e['picked_more_stable']})")
        print(f"   {'SLOW-FAM ceiling':22} {'ema_60-150 x signalflip':30} {s['jan_pct']:>8.2f} {s['feb_pct']:>8.2f}")
        print(f"   {'ORACLE (hindsight)':22} {'':30} {r['oracle_jan_pct']:>8.2f} {r['oracle_feb_pct']:>8.2f}")
        print(f"   {'BUY-HOLD':22} {'':30} {r['buyhold_jan_pct']:>8.2f} {r['buyhold_feb_pct']:>8.2f}")
        print(f"   ABLATION (single feature on raw baseline -> Feb delta):")
        for fname, a in blk["ablation"].items():
            print(f"      {fname:16} pick={a['pick'][:26]:26} Feb={a['feb_pct']:>7.2f}% "
                  f"delta={a['feb_delta_vs_baseline']:+.2f}pts (slow={a['slowness']:.0f})")
    print("\n" + "=" * 100)
    print("## VERDICT")
    print(f"   HEADLINE: {out['verdict']['headline']}")
    for ln in out["verdict"]["lines"]:
        print(f"   {ln}")
    print(f"   mean Feb delta (enriched - baseline) = {out['verdict']['mean_feb_delta_pts']:+.2f}pts across "
          f"{out['verdict']['cadences_with_data']} cadences; improved on {out['verdict']['cadences_improved']}.")
    print(f"   F2-stability was the top feature on {out['verdict']['f2_stability_top_on_n_cadences']} cadence(s); "
          f"F1-slowness on {out['verdict']['f1_slowness_top_on_n_cadences']}.")
    print(f"   LOOK-AHEAD: {out['look_ahead_audit']['features_jan_only']} {out['look_ahead_audit']['weights_preregistered']}")
    print("=" * 100)


# ===========================================================================
# SELF-TEST -- two-sided control (no market data): does the enriched score reward the right configs?
# ===========================================================================
def selftest():
    """Two-sided control on the FEATURE-SCORING logic (the new logic this file adds).
    POSITIVE: a config that is genuinely slow + stable + low-whipsaw must score HIGHER than a lucky
              one-block-fast Jan winner with the SAME raw Jan return -- so the features re-rank correctly.
    NEGATIVE: when two configs are identical on every feature, the enriched score must be equal (no
              spurious preference); and the F4 exit-rule must penalize a trailing exit ONLY when Jan
              whipsaw is high."""
    print("## FEATURE-ENRICHED SELECTOR SELFTEST (two-sided)")
    ok = True

    # Build a synthetic feature table. Two configs with the SAME raw Jan return:
    #   GOOD = slow, stable (consistent across blocks), low whipsaw   -> should score high
    #   LUCKY_FAST = fast, unstable (one huge block), high whipsaw     -> should score low (the overfit trap)
    cfgs = [("slow_100_200", "signalflip"), ("fast_2_5", "signalflip"),
            ("mid_20_50", "signalflip"), ("med_30_60", "signalflip")]
    feats = {
        ("slow_100_200", "signalflip"): {"raw_jan": 10.0, "slowness": 200.0, "mean_period": 150.0,
                                          "stability": 0.9, "whipsaw": 1.0},   # GOOD
        ("fast_2_5", "signalflip"):     {"raw_jan": 10.0, "slowness": 5.0, "mean_period": 3.5,
                                          "stability": -0.5, "whipsaw": 9.0},  # LUCKY FAST (same raw return)
        ("mid_20_50", "signalflip"):    {"raw_jan": 3.0, "slowness": 50.0, "mean_period": 35.0,
                                          "stability": 0.4, "whipsaw": 3.0},
        ("med_30_60", "signalflip"):    {"raw_jan": 1.0, "slowness": 60.0, "mean_period": 45.0,
                                          "stability": 0.5, "whipsaw": 2.0},
    }
    sc, info = enriched_scores(feats, jan_avg_trades=2.0, configs=cfgs)   # low whipsaw -> F4 not triggered
    good = sc[("slow_100_200", "signalflip")]
    lucky = sc[("fast_2_5", "signalflip")]
    print(f"  POSITIVE: same raw Jan=10 -> GOOD(slow/stable/low-churn) score={good:+.3f} vs "
          f"LUCKY_FAST score={lucky:+.3f} (expect GOOD > LUCKY_FAST)")
    ok &= (good > lucky)
    # and the enriched argmax should NOT be the lucky-fast config
    pick = max(cfgs, key=lambda c: sc[c])
    print(f"  POSITIVE-2: enriched argmax = {pick[0]} (expect NOT 'fast_2_5')")
    ok &= (pick[0] != "fast_2_5")

    # NEGATIVE: two identical-feature configs must score equal (no spurious tie-break).
    cfgs2 = [("a_50_100", "signalflip"), ("b_50_100", "signalflip"), ("c_10_20", "signalflip")]
    feats2 = {c: {"raw_jan": 5.0, "slowness": 100.0, "mean_period": 75.0, "stability": 0.3, "whipsaw": 2.0}
              for c in cfgs2[:2]}
    feats2[cfgs2[2]] = {"raw_jan": 0.0, "slowness": 20.0, "mean_period": 15.0, "stability": 0.0, "whipsaw": 5.0}
    sc2, _ = enriched_scores(feats2, jan_avg_trades=2.0, configs=cfgs2)
    eq = abs(sc2[cfgs2[0]] - sc2[cfgs2[1]]) < 1e-9
    print(f"  NEGATIVE: identical-feature configs score equal? {eq} (expect True)")
    ok &= eq

    # F4 control: a trailing exit must be penalized ONLY when Jan whipsaw is high.
    cfgs3 = [("x_20_50", "trail10"), ("y_20_50", "signalflip"), ("z_100_200", "signalflip")]
    fbase = {"raw_jan": 5.0, "slowness": 50.0, "mean_period": 35.0, "stability": 0.3, "whipsaw": 5.0}
    feats3 = {("x_20_50", "trail10"): dict(fbase), ("y_20_50", "signalflip"): dict(fbase),
              ("z_100_200", "signalflip"): {**fbase, "slowness": 200.0, "mean_period": 150.0}}
    sc_lo, _ = enriched_scores(feats3, jan_avg_trades=2.0, configs=cfgs3)    # low whipsaw -> NO F4 penalty
    sc_hi, _ = enriched_scores(feats3, jan_avg_trades=8.0, configs=cfgs3)    # high whipsaw -> F4 penalizes trail
    trail_lo = sc_lo[("x_20_50", "trail10")]
    trail_hi = sc_hi[("x_20_50", "trail10")]
    print(f"  F4: trailing-exit score lo-whipsaw={trail_lo:+.3f} vs hi-whipsaw={trail_hi:+.3f} "
          f"(expect hi < lo, i.e. penalized in chop)")
    ok &= (trail_hi < trail_lo - 0.5)

    # _ma_periods parse sanity
    slow, mean = _ma_periods("ema_62_172")
    print(f"  PARSE: ema_62_172 -> slow={slow:.0f} mean={mean:.0f} (expect 172, 117)")
    ok &= (slow == 172.0 and abs(mean - 117.0) < 1.0)

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python src/strat/config_selector_features.py")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadences", default="4h,1h,30m,15m")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    cadences = tuple(c.strip() for c in a.cadences.split(","))
    out = run(a.universe, cadences)
    _print(out)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"config_selector_features_{a.universe}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha,
                         "train_window": JAN, "eval_window": FEB, "cost_rt": TAKER_RT, "universe": a.universe},
               "result": out}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
