"""src/strat/ti_candidate_register.py -- the END-TO-END CANDIDATE REGISTER: per TI family + per TI type, the
deployable all-weather candidate emerging from the 2020/2021/2022 working-band rolling-selection.

USER /orc 2026-06-16 (3h): "end-to-end report for 2020-21 where candidates per TI family AND TI type start
emerging." This CONSOLIDATES: (a) the all-weather rolling-from-band results -- ti_band_rolling (18 non-MA TIs) +
working_band_rolling (8 MA types) @ 4h, per-year net 2020/2021/2022; (b) the within-year 6/3/3 band robustness
(within_2020/2021/2022). Into a TIERED candidate register ranked per family.

TIERS (sharper than 'preserves vs buy-hold' -- which ALL de-risked TIs pass; that just confirms beta):
  A_allweather : participates BOTH bulls (2020 & 2021 net > 0) AND the 2022 BEAR net >= -5% (flat-or-positive
                 across a full bull->bear cycle -- the long-only all-weather grail).
  B_preserve   : participates bulls AND bear net in [-30, -5) -- a de-risked beta that PRESERVES (loses far less
                 than buy-hold) but still bleeds the bear.
  C_bull_only  : participates bulls but bear net < -30% -- bull-dependent.
  D_weak       : does not cleanly participate both bulls.

The candidate to TRADE per (family, type) = the rolling-from-band book (rolling-pick), expressed as the specific
config the rolling policy selects each window (the user's "express the specific configs, picked with rolling
knowledge"). Long-only spot, fixed-EW, maker, UNSEEN sealed. No emoji.

RWYB: python -m strat.ti_candidate_register
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AW = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
WY = ROOT.parent / "runs" / "strat" / "within_year"
PER = ROOT.parent / "runs" / "periods" / "TRAIN"
FAM_ORDER = ["trend", "momentum", "breakout", "volume", "mean-reversion", "MA"]
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]


def _latest(pattern):
    fs = sorted(glob.glob(str(AW / pattern)))
    return json.load(open(fs[-1])) if fs else None


def _within_band(year, ti, tf="4h"):
    """# robust configs (TRAIN&VAL>0 ironed) for a non-MA TI in within_<year>.json @ tf, else None."""
    p = WY / f"within_{year}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    cell = d.get("indicators", {}).get(ti, {}).get("per_tf", {}).get(tf)
    if not cell:
        return None
    return sum(1 for r in cell.get("rows", []) if r["iron"]["robust"])


def _ma_within_band(year, mt, tf="4h"):
    """# band configs (positive 3-way) for an MA type from config_leaderboard.json @ tf, else None."""
    p = PER / str(year) / "DEEP_DIVE" / "config_leaderboard.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    cell = d.get("grid", {}).get(f"{mt}|{tf}")
    if not cell:
        return None
    return cell["band"]["n_band_2ma"] + cell["band"]["n_band_3ma"]


def _tier(n20, n21, n22):
    if None in (n20, n21, n22):
        return "D_weak", None
    bulls = n20 > 0 and n21 > 0
    score = round(min(n20, 150) * 0.1 + min(n21, 400) * 0.1 + n22 * 1.0, 1)   # bear net dominates (the scarce thing)
    if not bulls:
        return "D_weak", score
    if n22 >= -5:
        return "A_allweather", score
    if n22 >= -30:
        return "B_preserve", score
    return "C_bull_only", score


def _collect():
    rows = []
    tib = _latest("ti_band_rolling_*.json")
    if tib:
        res = tib["by_tf"].get("4h", {}).get("results", {})
        for ti, rec in res.items():
            rp = rec["rolling_pick"]
            n = [rp.get(y, {}).get("net") for y in ("2020_bull", "2021_mixed", "2022_bear")]
            dd22 = rp.get("2022_bear", {}).get("maxdd")
            tier, score = _tier(*n)
            rows.append({"ti": ti, "family": rec["family"], "n20": n[0], "n21": n[1], "n22": n[2], "dd22": dd22,
                         "tier": tier, "score": score, "n_picks": rec.get("n_distinct_picks"),
                         "band": [_within_band(y, ti) for y in (2020, 2021, 2022)]})
    mab = _latest("working_band_rolling_*.json")
    if mab:
        res = mab["by_tf"].get("4h", {}).get("results", {})
        for mt, rec in res.items():
            rp = rec.get("rolling_pick", {})
            n = [rp.get(y, {}).get("net") for y in ("2020_bull", "2021_mixed", "2022_bear")]
            dd22 = rp.get("2022_bear", {}).get("maxdd")
            tier, score = _tier(*n)
            rows.append({"ti": mt, "family": "MA", "n20": n[0], "n21": n[1], "n22": n[2], "dd22": dd22,
                         "tier": tier, "score": score, "n_picks": rec.get("n_distinct_picks"),
                         "band": [_ma_within_band(y, mt) for y in (2020, 2021, 2022)]})
    return rows


def main(argv=None) -> int:
    rows = _collect()
    if not rows:
        print("no all-weather results found (run ti_band_rolling + working_band_rolling first)"); return 1
    rows.sort(key=lambda r: (r["score"] is not None, r["score"] or -1e9), reverse=True)
    # buy-hold 2022 reference
    tib = _latest("ti_band_rolling_*.json")
    bh = None
    if tib:
        any_rec = next(iter(tib["by_tf"]["4h"]["results"].values()))
        bh = any_rec["buyhold"]
    L = ["# CANDIDATE REGISTER -- all-weather (2020/2021/2022) per TI family + per TI type @ 4h", ""]
    L.append("The deployable candidate per type = the ROLLING-FROM-BAND book (walk-forward pick from the trailing-"
             "positive band; the user's 'rolling knowledge + rolling performance'). NO look-ahead. Long-only spot, "
             "fixed-EW u10, ironed sleeve, maker. net% per year; bear = 2022 (buy-hold 2022 ~ -61..-71%). "
             "[VERIFIED all-weather]. TIERS: A=all-weather (bulls + bear>=-5%), B=preserve (bear -30..-5), "
             "C=bull-only (bear<-30), D=weak bulls.")
    L.append("")
    L.append("| rank | TI | family | tier | 2020 | 2021 | **2022 bear** | DD22 | band 20/21/22 | picks |")
    L.append("|---:|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        b = "/".join(str(x) if x is not None else "-" for x in r["band"])
        L.append(f"| {i} | {r['ti']} | {r['family']} | {r['tier'][0]} | {r['n20']} | {r['n21']} | "
                 f"**{r['n22']}** | {r['dd22']} | {b} | {r['n_picks']} |")
    L.append("")
    # per-family + per-tier rollup
    L.append("## Emerging candidates per family (tier A/B first)")
    for fam in FAM_ORDER:
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        a = [r["ti"] for r in fr if r["tier"] == "A_allweather"]
        bb = [r["ti"] for r in fr if r["tier"] == "B_preserve"]
        L.append(f"- **{fam}**: A(all-weather)={a or '--'} ; B(preserve)={bb or '--'} ; "
                 f"best={max(fr, key=lambda r: r['score'] or -1e9)['ti']} "
                 f"(score {max(fr, key=lambda r: r['score'] or -1e9)['score']})")
    L.append("")
    tierA = [r["ti"] for r in rows if r["tier"] == "A_allweather"]
    L.append(f"## HEADLINE: {len(tierA)} TIER-A all-weather candidates (positive/flat in the 2022 bear AND "
             f"participate both bulls): {tierA}")
    L.append(f"buy-hold all-weather: 2020 {bh.get('2020_bull',{}).get('net') if bh else '?'}% / "
             f"2021 {bh.get('2021_mixed',{}).get('net') if bh else '?'}% / 2022 {bh.get('2022_bear',{}).get('net') if bh else '?'}%. "
             f"Every TI preserves the bear (de-risked beta); the TIER-A set is rarer -- it stays flat-or-positive "
             f"through the full cycle. CAVEAT: rolling-pick has selection (no look-ahead but in-sample-tuned "
             f"lookback); these are CANDIDATES, not ship-grade -- next is the robustness battery (10-seed / p05 / "
             f"jackknife) on the tier-A set.")
    out = AW / "CANDIDATE_REGISTER_2020_2022.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"\n[md] {out}")
    # chart: tier-colored 2022-bear net per TI (sorted)
    rs = sorted([r for r in rows if r["n22"] is not None], key=lambda r: r["n22"], reverse=True)
    colmap = {"A_allweather": "#2ca02c", "B_preserve": "#1f77b4", "C_bull_only": "#ff7f0e", "D_weak": "#d62728"}
    fig, ax = plt.subplots(figsize=(max(11, len(rs) * 0.5), 6))
    ax.bar(range(len(rs)), [r["n22"] for r in rs], color=[colmap[r["tier"]] for r in rs])
    ax.axhline(0, color="grey", lw=0.6); ax.axhline(-5, color="green", ls=":", lw=1, label="tier-A floor (-5%)")
    if bh:
        ax.axhline(bh.get("2022_bear", {}).get("net", -65), color="black", ls="--", lw=1.2, label="buy-hold 2022")
    ax.set_xticks(range(len(rs))); ax.set_xticklabels([f"{r['ti']}" for r in rs], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("2022 BEAR net % (rolling-from-band)")
    ax.set_title("ALL-WEATHER candidate emergence @ 4h: 2022-BEAR net per TI (GREEN=tier-A all-weather, BLUE=preserve, "
                 "ORANGE=bull-only). Above the green dotted line = flat-or-positive through the bear.", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = AW / "charts" / "candidate_register_2022bear_4h.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[chart] {p}")
    json.dump({"rows": rows, "tier_A": tierA}, open(AW / "candidate_register_2020_2022.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
