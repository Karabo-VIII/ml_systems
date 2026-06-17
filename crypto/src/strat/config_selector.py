"""src/strat/config_selector.py -- the GENERALIZED period-level regime->config SELECTOR (the FAIR TEST).

THE QUESTION (vs config_selector_jan2feb.py, which was hardcoded to the Jan->Feb 2020 COVID CRASH onset):
on a CALM, PERSISTENT-REGIME month-pair -- where month-1 AND month-2 are the SAME regime (a sustained
trend, or a sustained range) -- does the config-selector learn a genuine regime->config SUITABILITY map,
or does it STILL only find cost-avoidance (collapse to one slow config) like the crash test did? This
settles whether the prior NULL was the CRASH CONFOUND or a REAL LIMIT of period-level config-selection.

THIS FILE IS A THIN CALLER. It reuses ALL the logic in config_selector_jan2feb.py (the selector + its
6-technique data-expansion pipeline + the eval + the config-rank-persistence metric + the two-sided
selftest) and src/strat/data_expansion.py (the canonical toolkit). The ONLY new things here are:
  (1) parameterized (train_start, train_end, test_start, test_end) windows (config_selector_jan2feb stays
      working -- select_for_cadence now takes optional train_win/test_win, defaulting to JAN/FEB).
  (2) MEASURED regime-persistence across the universe for the chosen pair (the crux: report how same the
      two months' regimes actually are, per asset, before believing any "fair" claim).
  (3) the FAIR-TEST verdict (CONFIRMED iff config-rank persists AND the map differentiates AND the book
      approaches the oracle; REFUTED otherwise -- both are high-value, neither is spun).

THE 4 KEY NEW QUESTIONS (answered explicitly per cadence, vs the crash test):
  1. Does the regime->config map DIFFERENTIATE across assets/regime-buckets, or COLLAPSE to one config
     (the cost-avoidance signature)?  -> scoreboard.map_differentiates_by_regime + n_distinct_configs_picked
  2. Does the model APPROACH the oracle (small gap), not just beat random?  -> book gap_to_oracle + ratio
  3. Is any 'win' REGIME INTELLIGENCE (trend configs in a persistent trend) or STILL cost-avoidance (slow
     configs trade rarely)?  -> we inspect WHICH configs win + their hold/turnover signature.
  4. Does month-1's best config STAY best in month-2?  -> config_rank_persistence.book_spearman_rho (THE
     cleanest single number for 'does regime->config transfer'). Reported per-asset + book.

TWO-SIDED CONTROLS (mandated): the synthetic-edge positive control must SHIP and the random / noise
negative control must FAIL -- both are in config_selector_jan2feb.selftest() (and data_expansion._selftest),
re-run here so the machinery's soundness travels with the fair test.

PERSISTENT-REGIME PAIRS (verified before running; AVOIDING any pair spanning a known shock -- 2020-03 COVID,
2021-05 crash, 2022-06 collapse):
  PRIMARY  : 2024-02->03  SUSTAINED BULL. u10 M1 +29%/100%-up, M2 +22.5%/80%-up; regime persist 6/10.
  SECOND   : 2023-09->10  CALM RANGE->mild-up (low vol). u10 M1 +5.4%, M2 +18.4%; worst DD ~-16%; persist 6/10.
  (Both MEASURED in the probe; the persistence number is RE-MEASURED + reported by this script at run time.)

RWYB:
  python src/strat/config_selector.py --selftest                       # two-sided controls (no market)
  python src/strat/config_selector.py --pair bull2024                  # the PRIMARY fair test
  python src/strat/config_selector.py --pair range2023                 # the SECOND (robustness) fair test
  python src/strat/config_selector.py --pair both                      # both, with a combined verdict
  python src/strat/config_selector.py --train 2024-02-01:2024-03-01 --test 2024-03-01:2024-04-01

No emoji (Windows cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# REUSE the existing selector wholesale -- do NOT reinvent.
import strat.config_selector_jan2feb as CS                              # noqa: E402
from strat.config_selector_jan2feb import (build_config_space, select_for_cadence,  # noqa: E402
                                           regime_label, EXIT_MENU, FLOOR, key2cfg, selftest)
from strat.portfolio_replay import TAKER_RT                            # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                       # noqa: E402
from mining.family_regime_map import _norm_sym                          # noqa: E402

OUT = ROOT / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research_verdict",
    "inputs": {"chimera": "via pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)"},
    "outputs": {"verdict_json": "runs/strat/config_selector_fairtest_<pair>_<stamp>.json"},
    "invariants": {
        "thin_caller_reuses_jan2feb": "ALL selector + data-expansion + eval logic is config_selector_jan2feb; "
                                      "this file only parameterizes the windows + measures regime persistence + "
                                      "renders the fair-test verdict",
        "persistent_regime_required": "the test-month regime must ~= the train-month regime; persistence is "
                                      "MEASURED across the universe and reported (the crux of 'fair')",
        "no_shock_in_pair": "the chosen pairs avoid 2020-03 / 2021-05 / 2022-06 known shocks",
        "two_sided_controls": "synthetic-edge positive control must SHIP; random/noise negative must FAIL "
                              "(re-run via selftest)",
        "config_rank_persistence_is_the_crux": "month1->month2 per-config Spearman rho is the single cleanest "
                                               "'does regime->config transfer' number",
        "objective_is_compound": "judged on held-out net COMPOUND return; never AUC/IC",
        "no_spin": "CONFIRMED requires rank-persists AND map-differentiates AND approaches-oracle; a fair-case "
                   "refutation is reported as a refutation, not spun",
    },
}

# Verified persistent-regime pairs (regime/return probe 2026-06-12; persistence RE-MEASURED at run time).
PAIRS = {
    "bull2024":  {"train": ("2024-02-01", "2024-03-01"), "test": ("2024-03-01", "2024-04-01"),
                  "label": "SUSTAINED BULL 2024-02->03",
                  "probe": "u10 M1 +29.2%/100%-up, M2 +22.5%/80%-up; worst-asset-DD M2 -29% (normal vol, "
                           "no market-wide shock -- book stayed +22.5%); regime persist 6/10"},
    "range2023": {"train": ("2023-09-01", "2023-10-01"), "test": ("2023-10-01", "2023-11-01"),
                  "label": "CALM RANGE->mild-up 2023-09->10",
                  "probe": "u10 M1 +5.4%/70%-up (low vol), M2 +18.4%/100%-up; worst-asset-DD ~-16%; "
                           "regime persist 6/10 -- a CALMER, different regime for robustness"},
}


# ===========================================================================
# MEASURE regime persistence across the universe for a chosen pair (the crux)
# ===========================================================================
def measure_regime_persistence(syms, train_win, test_win, cadence="1d"):
    """For each asset: the month-1 END regime vs the month-2 END regime (the SAME regime_label the selector
    uses). Report per-asset (train_regime, test_regime, same?) + the universe persistence fraction + the
    regime mix of each month. This is what makes 'fair' an EMPIRICAL claim, not an assumption."""
    ts_ms = lambda x: pd.Timestamp(x).value // 10**6
    t1s, t1e = ts_ms(train_win[0]), ts_ms(train_win[1])
    t2s, t2e = ts_ms(test_win[0]), ts_ms(test_win[1])
    rows = []
    for sym in syms:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence, features=["close"])
        except Exception:
            continue
        idx = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor(FLOOR[cadence])
        c = df["close"].to_numpy().astype(float)
        ms = (idx.asi8 // 10**6).astype("int64")
        if ((ms >= t1s) & (ms < t1e)).sum() < 5 or ((ms >= t2s) & (ms < t2e)).sum() < 5:
            continue
        tb1, _, f1 = regime_label(c, ms, t1s, t1e)     # regime_label returns (trend, vol_bucket=None, feats)
        tb2, _, f2 = regime_label(c, ms, t2s, t2e)
        rows.append({"asset": sym[:-4], "train_regime": tb1, "test_regime": tb2,
                     "same": tb1 == tb2, "train_ret": f1.get("window_ret"), "test_ret": f2.get("window_ret")})
    n = len(rows)
    persist = sum(1 for r in rows if r["same"])
    mix1 = dict(Counter(r["train_regime"] for r in rows))
    mix2 = dict(Counter(r["test_regime"] for r in rows))
    return {"cadence_for_regime": cadence, "n_assets": n, "n_persistent": persist,
            "persistence_frac": round(persist / n, 3) if n else None,
            "train_regime_mix": mix1, "test_regime_mix": mix2, "per_asset": rows,
            "is_persistent": (persist / n >= 0.5) if n else False}


# ===========================================================================
# CONFIG-NATURE inspection (Key Question 3): is a winning config a TREND config or a COST-AVOIDANCE config?
# ===========================================================================
def classify_config(cfg_key):
    """Heuristic label of a (entry|exit) config's NATURE, for the 'regime intelligence vs cost-avoidance'
    question. entry name encodes the MA spans; a fast-MA / short-span entry = responsive (trades often);
    a very slow-MA entry = rare-trading (cost-surviving). exit signalflip = let-it-ride (trend-following);
    tight trail / short timestop = defensive. We surface the entry's slowest span + the exit kind so the
    verdict can say WHICH kind of config won and WHY."""
    en, ex = cfg_key.split("|")
    nums = [int(t) for t in en.replace("ema_", "").replace("sma_", "").split("_") if t.isdigit()]
    slowest = max(nums) if nums else None
    fast = min(nums) if nums else None
    ma_kind = "EMA" if "ema" in en else ("SMA" if "sma" in en else "?")
    # exit nature
    exit_nature = {"signalflip": "ride", "trail5": "tight-defensive", "trail10": "loose-defensive",
                   "atr3": "vol-defensive", "time24": "time-capped", "tp8": "take-profit"}.get(ex, ex)
    # speed nature from the slowest span (rough): >=100 = slow/rare; <=50 = responsive
    if slowest is None:
        speed = "?"
    elif slowest >= 100:
        speed = "slow-rare(cost-surviving)"
    elif slowest <= 50:
        speed = "responsive(trend-reactive)"
    else:
        speed = "medium"
    return {"entry": en, "exit": ex, "ma_kind": ma_kind, "fast_span": fast, "slow_span": slowest,
            "entry_speed": speed, "exit_nature": exit_nature}


# ===========================================================================
# RUN one pair across cadences
# ===========================================================================
def run_pair(pair_key, train_win, test_win, label, probe, cadences, syms, entry_specs, configs,
             K_synth=40, n_boot=400):
    print("=" * 90)
    print(f"## FAIR TEST -- PAIR '{pair_key}': {label}")
    print(f"   train(month-1) {train_win[0]}..{train_win[1]}  ->  test(month-2) {test_win[0]}..{test_win[1]}")
    print(f"   probe (1d, u10): {probe}")
    print("=" * 90)

    # ---- THE CRUX: measure regime persistence across u10 (at the 1d regime cadence) ----
    persistence = measure_regime_persistence(syms, train_win, test_win, cadence="1d")
    print(f"\n## REGIME PERSISTENCE (1d, the selector's regime cadence):")
    print(f"   train regime mix {persistence['train_regime_mix']}  ->  test regime mix {persistence['test_regime_mix']}")
    print(f"   PERSISTENCE = {persistence['n_persistent']}/{persistence['n_assets']} assets keep the same "
          f"regime ({persistence['persistence_frac']}) -> is_persistent={persistence['is_persistent']}")
    for r in persistence["per_asset"]:
        flag = "" if r["same"] else "  <-- regime SHIFTED"
        print(f"     {r['asset']:6} {str(r['train_regime']):5} -> {str(r['test_regime']):5} "
              f"(train_ret {r['train_ret']}, test_ret {r['test_ret']}){flag}")

    results = {}
    for cad in cadences:
        r = select_for_cadence(cad, syms, entry_specs, configs, K_synth=K_synth, n_boot=n_boot,
                               train_win=train_win, test_win=test_win)
        results[cad] = r
        if r.get("verdict", "").startswith("INSUFFICIENT"):
            print(f"\n## {cad}: {r['verdict']} (n_assets={r.get('n_assets')})")
            continue
        sb = r["scoreboard"]; bk = r["eval_book"]; ex = r["data_expansion"]; crp = r["config_rank_persistence"]
        print(f"\n## {cad}: {r['n_assets']} assets ({', '.join(r['assets'])})")
        print(f"   DATA-EXPANSION: cross-sec={ex['1_cross_sectional']['n_assets']} | "
              f"sub-period={ex['2_sub_period']['n_subperiod_asset_config_samples']} (OVERLAP) | "
              f"bootstrap={ex['3_block_bootstrap']['n_boot']}/cfg | "
              f"synthetic={ex['4_synthetic']['K_paths']}x{ex['4_synthetic'].get('n_assets_simulated','?')} | "
              f"shrink B={ex['5_shrinkage']['shrinkage_B']}")
        # Q1: map differentiates?
        print(f"   Q1 map DIFFERENTIATES? {sb['map_differentiates_by_regime']} "
              f"({sb['n_distinct_configs_picked']} distinct config(s) across {len(ex['6_regime_to_config']['buckets'])} buckets) "
              f"-- {'genuine regime->config skill' if sb['map_differentiates_by_regime'] else 'COLLAPSES to single config (cost-avoidance signature)'}")
        # Q4: config-rank persistence (the crux number)
        print(f"   Q4 CONFIG-RANK PERSISTENCE (month1->month2 per-config Spearman rho): "
              f"book={crp['book_spearman_rho']}  mean-per-asset={crp['mean_per_asset_spearman_rho']}")
        # Q2: approaches oracle?
        gap = bk["book_gap_to_oracle_pct"]; orc = bk["book_oracle_pct"]; prd = bk["book_pred_perasset_pct"]
        ratio = round(prd / orc, 2) if orc not in (0, None) and abs(orc) > 1e-6 else None
        print(f"   Q2 APPROACHES ORACLE? book pred {prd}% vs oracle {orc}% (gap {gap}%, pred/oracle ratio {ratio}) "
              f"vs random {bk['book_random_mean_pct']}% vs BH {bk['book_buyhold_pct']}%")
        # Q3: which configs win + their nature
        regime_map = r["regime_config_map"]
        picks = {b: regime_map[b]["best_config"] for b in regime_map}
        print(f"   Q3 WHICH configs picked per bucket (nature):")
        for b, cfgk in picks.items():
            nat = classify_config(cfgk)
            print(f"      {b:10} -> {cfgk:30} [{nat['entry_speed']}, exit={nat['exit_nature']}]")
        # per-asset eval table
        print(f"\n   {'asset':6} {'predReg':8} {'pred_cfg':26} {'pred%':>8} {'oracle%':>8} "
              f"{'gap':>7} {'rand%':>7} {'BH%':>7} {'>rand':>6}")
        for row in r["eval_per_asset"]:
            print(f"   {row['asset']:6} {row['pred_regime']:8} {row['pred_config'][:26]:26} "
                  f"{row['pred_feb_pct']:>8.2f} {row['feb_oracle_pct']:>8.2f} {row['gap_to_oracle_pct']:>7.2f} "
                  f"{row['random_mean_pct']:>7.2f} {row['buy_hold_pct']:>7.2f} {str(row['beats_random']):>6}")
        print(f"   SCOREBOARD: {sb['n_beats_random']}/{sb['n_assets']} beat random | "
              f"{sb['n_beats_buyhold']}/{sb['n_assets']} beat buy-hold | "
              f"book {'BEATS' if sb['book_beats_random'] else 'LOSES TO'} random ({sb['book_pctile_vs_random']}th pctile)")

    return {"pair": pair_key, "label": label, "probe": probe, "train_win": train_win, "test_win": test_win,
            "regime_persistence": persistence, "results": results}


# ===========================================================================
# FAIR-TEST VERDICT (per pair, across cadences)
# ===========================================================================
def fairtest_verdict(pair_result):
    """CONFIRMED iff, in this PERSISTENT regime: config-rank PERSISTS (book rho > +0.3 on a majority of
    cadences) AND the map DIFFERENTIATES (>1 distinct config on a majority) AND the book APPROACHES the
    oracle (pred/oracle ratio > ~0.5 on a majority, i.e. captures over half the hindsight-best). REFUTED if
    rank doesn't persist OR the map collapses to one config (cost-avoidance) OR it stays far from the oracle.
    No spin: a fair-case refutation = 'config-selection does NOT transfer even when the regime persists'."""
    valid = {c: r for c, r in pair_result["results"].items() if not r.get("verdict", "").startswith("INSUFFICIENT")}
    persistence = pair_result["regime_persistence"]
    lines = []
    if not valid:
        return {"headline": "NO VALID CADENCE", "lines": ["all cadences had insufficient data."], "verdict": "INCONCLUSIVE"}
    n_cad = len(valid)
    # gather the three signals per cadence
    rho_book = {c: r["config_rank_persistence"]["book_spearman_rho"] for c, r in valid.items()}
    differ = {c: bool(r["scoreboard"]["map_differentiates_by_regime"]) for c, r in valid.items()}
    ratio = {}
    for c, r in valid.items():
        bk = r["eval_book"]; orc = bk["book_oracle_pct"]; prd = bk["book_pred_perasset_pct"]
        ratio[c] = (prd / orc) if (orc and abs(orc) > 1e-6 and orc > 0) else (1.0 if prd >= orc else 0.0)
    n_persist = sum(1 for c in valid if rho_book[c] is not None and rho_book[c] > 0.30)
    n_differ = sum(1 for c in valid if differ[c])
    n_approach = sum(1 for c in valid if ratio[c] is not None and ratio[c] > 0.5)
    mean_rho = float(np.mean([rho_book[c] for c in valid if rho_book[c] is not None])) if any(
        rho_book[c] is not None for c in valid) else None

    maj = n_cad // 2 + 1
    persists_maj = n_persist >= maj
    differ_maj = n_differ >= maj
    approach_maj = n_approach >= maj
    confirmed = persists_maj and differ_maj and approach_maj
    # DEFENSIVE axis (distinct from oracle-capture): does the book BEAT passive buy-hold? This is the value a
    # selector adds when HOLD loses (a down/chop regime) -- a strict-thesis REFUTED can still preserve capital.
    # Without this axis the verdict over-generalizes a bear run (selector beats BH 10/10) into "a REAL LIMIT".
    beats_bh = {c: r["eval_book"]["book_pred_perasset_pct"] > r["eval_book"].get("book_buyhold_pct", -1e9)
                for c, r in valid.items()}
    n_beats_bh = sum(1 for c in valid if beats_bh[c])
    beats_bh_maj = n_beats_bh >= maj

    lines.append(f"PERSISTENT-REGIME FAIR TEST -- {pair_result['label']}.")
    lines.append(f"Regime persistence MEASURED: {persistence['n_persistent']}/{persistence['n_assets']} "
                 f"assets keep the same regime month1->month2 ({persistence['persistence_frac']}); "
                 f"train mix {persistence['train_regime_mix']} -> test mix {persistence['test_regime_mix']}.")
    lines.append("")
    for c in valid:
        r = valid[c]; bk = r["eval_book"]; sb = r["scoreboard"]
        lines.append(f"[{c}] config-rank rho(book)={rho_book[c]} | map-differentiates={differ[c]} "
                     f"({sb['n_distinct_configs_picked']} cfgs) | pred {bk['book_pred_perasset_pct']}% / "
                     f"oracle {bk['book_oracle_pct']}% (ratio {round(ratio[c],2) if ratio[c] is not None else None}) | "
                     f"book {'BEATS' if sb['book_beats_random'] else 'LOSES'} random")
    lines.append("")
    lines.append(f"Q1 map differentiates on {n_differ}/{n_cad} cadences (majority={differ_maj}).")
    lines.append(f"Q2 approaches oracle (>50% capture) on {n_approach}/{n_cad} (majority={approach_maj}).")
    lines.append(f"Q3 DEFENSIVE -- book beats passive buy-hold on {n_beats_bh}/{n_cad} (majority={beats_bh_maj}): "
                 f"capital-preservation value vs hold, DISTINCT from oracle capture (matters when hold loses).")
    lines.append(f"Q4 config-rank persists (book rho>+0.30) on {n_persist}/{n_cad} (majority={persists_maj}); "
                 f"mean book rho={round(mean_rho,3) if mean_rho is not None else None}.")
    lines.append("")

    if confirmed:
        verdict = "CONFIRMED"
        head = ("CONFIRMED (FAIR CASE): in this persistent regime the config-selector shows genuine "
                "regime->config TRANSFER -- month-1's config ranking PERSISTS into month-2 (rho>+0.3), the "
                "map DIFFERENTIATES by regime (not one collapsed config), AND the book captures >50% of the "
                "hindsight oracle, all on a majority of cadences. This is the FIRST real evidence the thesis "
                "works: the prior Jan->Feb NULL was the CRASH CONFOUND, not a limit of period-level config-"
                "selection. (Still 1-month-trained = wide CIs; needs UNSEEN + multi-pair confirmation to ship.)")
    else:
        verdict = "REFUTED"
        # name the dominant failure mode
        fails = []
        if not persists_maj:
            fails.append(f"config-rank does NOT persist (book rho>+0.3 on only {n_persist}/{n_cad}; "
                         f"mean rho {round(mean_rho,3) if mean_rho is not None else None}) -- last month's "
                         f"best config is uninformative for next month EVEN in the same regime")
        if not differ_maj:
            fails.append(f"the map COLLAPSES to one config on a majority ({n_differ}/{n_cad} differentiate) "
                         f"-- the cost-avoidance signature, NOT learned regime navigation")
        if not approach_maj:
            fails.append(f"it stays FAR from the oracle (>50% capture on only {n_approach}/{n_cad}) -- beating "
                         f"random is not the same as approaching the hindsight best")
        head = ("REFUTED (STRICT THESIS): even with the regime PERSISTING month1->month2, " + "; ".join(fails) +
                ". The strict regime->config-TRANSFER thesis does NOT hold even in the fair case -- the prior NULL "
                "was not merely the crash confound.")
        if beats_bh_maj:
            head += (f" SCOPE NOTE: HOWEVER the book BEATS passive buy-hold on a majority of cadences "
                     f"({n_beats_bh}/{n_cad}) -- in this regime the selector DOES add DEFENSIVE "
                     "(cost-avoidance / capital-preservation) value over hold. The refutation is specifically "
                     "about CAPTURING the hindsight oracle via a regime->config map, NOT about beating buy-hold. "
                     "Do NOT cite this as 'config-selection adds no value'.")
        else:
            head += (" And the book does not beat passive buy-hold here either, so in THIS regime there is no "
                     "defensive value to salvage. (Honest, not spun.)")
    lines.insert(2, f"HEADLINE: {head}")
    return {"headline": head, "verdict": verdict, "confirmed": confirmed,
            "n_cadences": n_cad, "n_map_differentiates": n_differ, "n_approaches_oracle": n_approach,
            "n_beats_buyhold_book": n_beats_bh, "beats_buyhold_majority": beats_bh_maj,
            "n_config_rank_persists": n_persist, "mean_book_rho": round(mean_rho, 3) if mean_rho is not None else None,
            "rho_book_by_cadence": rho_book, "pred_oracle_ratio_by_cadence": {c: round(v, 3) if v is not None else None for c, v in ratio.items()},
            "lines": lines}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python src/strat/config_selector.py")
    ap.add_argument("--selftest", action="store_true", help="two-sided controls (reuses jan2feb selftest)")
    ap.add_argument("--pair", default="both", choices=["bull2024", "range2023", "both"],
                    help="which verified persistent-regime pair to run")
    ap.add_argument("--train", default=None, help="custom train window START:END (e.g. 2024-02-01:2024-03-01)")
    ap.add_argument("--test", default=None, help="custom test window START:END")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadences", default="4h,1h,30m,15m")
    ap.add_argument("--max-entry-each", type=int, default=4)
    ap.add_argument("--K-synth", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=400)
    a = ap.parse_args(argv)

    if a.selftest:
        print("## CONFIG-SELECTOR (generalized) -- two-sided controls (reusing jan2feb + data_expansion selftests)\n")
        rc1 = selftest()
        import strat.data_expansion as DX
        print()
        rc2 = DX._selftest()
        rc = 0 if (rc1 == 0 and rc2 == 0) else 1
        print(f"\n## COMBINED SELFTEST {'PASS' if rc == 0 else 'FAIL'}")
        return rc

    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{a.universe}.yaml"))
    syms = [x["symbol"] for x in spec["assets"]]
    entry_specs, configs = build_config_space(a.max_entry_each)
    cadences = [c.strip() for c in a.cadences.split(",")]

    # custom windows override the pair menu
    if a.train and a.test:
        ts, te = a.train.split(":"); ss, se = a.test.split(":")
        run_specs = [("custom", (ts, te), (ss, se), f"CUSTOM {ts}..{te} -> {ss}..{se}", "custom (no probe)")]
    else:
        keys = ["bull2024", "range2023"] if a.pair == "both" else [a.pair]
        run_specs = [(k, PAIRS[k]["train"], PAIRS[k]["test"], PAIRS[k]["label"], PAIRS[k]["probe"]) for k in keys]

    print(f"## CONFIG-SELECTOR FAIR TEST -- {a.universe} -- config space: {len(entry_specs)} entry x "
          f"{len(EXIT_MENU)} exits = {len(configs)} configs -- cadences {cadences}\n")

    all_pairs = []
    for (pk, tw, sw, lab, pr) in run_specs:
        pres = run_pair(pk, tw, sw, lab, pr, cadences, syms, entry_specs, configs,
                        K_synth=a.K_synth, n_boot=a.n_boot)
        verdict = fairtest_verdict(pres)
        pres["verdict"] = verdict
        all_pairs.append(pres)
        print("\n" + "-" * 90)
        print(f"## FAIR-TEST VERDICT -- {pk}: {verdict['verdict']}")
        for line in verdict["lines"]:
            print(f"   {line}")
        print("-" * 90)

    # ---- combined verdict ----
    print("\n" + "=" * 90)
    print("## COMBINED FAIR-TEST VERDICT (across pairs)")
    verdicts = [p["verdict"]["verdict"] for p in all_pairs]
    n_conf = sum(1 for v in verdicts if v == "CONFIRMED")
    if n_conf == len(verdicts) and verdicts:
        combined = ("CONFIRMED across all tested persistent-regime pairs -- the thesis holds in the fair case; "
                    "the prior crash-test null was the confound. Promote to UNSEEN + multi-pair validation.")
    elif n_conf >= 1:
        combined = (f"MIXED -- CONFIRMED on {n_conf}/{len(verdicts)} pairs. Regime->config transfer appears in "
                    "SOME persistent regimes but not robustly; treat as suggestive, not established.")
    else:
        any_defensive = any(p["verdict"].get("beats_buyhold_majority") for p in all_pairs)
        if any_defensive:
            combined = ("REFUTED (STRICT THESIS) across all tested pairs -- a month-1 regime->config map does NOT "
                        "transfer to approach the hindsight oracle in any tested regime (map collapses to a slow "
                        "config or stays far below 50% oracle-capture). BUT this is SCOPE-LIMITED, not universal: "
                        "on >=1 pair the selector BEATS passive buy-hold (defensive cost-avoidance value is real, "
                        "esp. in down-regimes where hold loses ~25pp). The refutation is about oracle CAPTURE via a "
                        "regime map, NOT about beating hold. Do NOT cite as 'config-selection is worthless'.")
        else:
            combined = ("REFUTED across all tested persistent-regime pairs -- period-level config-selection does NOT "
                        "transfer even when the regime persists, AND does not beat passive buy-hold in these "
                        "regimes. The prior null was a REAL LIMIT, not just the crash confound. Honest result.")
    print(f"   {combined}")
    print(f"   per-pair: " + ", ".join(f"{p['pair']}={p['verdict']['verdict']}" for p in all_pairs))
    print("=" * 90)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = a.pair if not (a.train and a.test) else "custom"
    p = OUT / f"config_selector_fairtest_{tag}_{stamp}.json"
    json.dump({
        "repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha, "cost_rt": TAKER_RT,
                  "universe": a.universe, "cadences": cadences},
        "combined_verdict": combined, "per_pair_verdict": {p_["pair"]: p_["verdict"]["verdict"] for p_ in all_pairs},
        "pairs": all_pairs,
    }, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
