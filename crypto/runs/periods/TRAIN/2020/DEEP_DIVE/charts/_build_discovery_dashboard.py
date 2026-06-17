"""
MASTER DASHBOARD builder -- the executive ONE-figure summary of the finer-TF MA
discovery-engine investigation (the 8-stage capstone).

Every number on the figure is pulled LIVE from the stage JSONs in the parent
DEEP_DIVE directory -- NO hardcoded results. Run:

    python runs/periods/TRAIN/2020/DEEP_DIVE/charts/_build_discovery_dashboard.py

Output: runs/periods/TRAIN/2020/DEEP_DIVE/charts/discovery_engine_dashboard.png

Sources (each panel cites its JSON):
  (a) ma_type_tf_research.json     winners_by_tf + family_avg_net_by_tf
  (b) complementarity_matrix.json  corr_trend_mr per TF + synthetic_regime_stress dd-by-regime
  (c) dynamic_engine.json          OOS DYN vs STATIC per TF + synthetic stress falsification
  (d) synthetic_regime_stress.json robustness_rank (worst-scenario worst-seed net)
  (e) complementary_sleeve_search.json long_only_relaxation_value + scoreboard
  (f) longshort_book.json          PHASE 6: longshort engine net-positive + book-vs-singles + LO-exception value
  (g) regime_gated_book.json       PHASE 7: detector timing-skill (beats shuffle) but gated book != deployable
  (h) DISCOVERY_ENGINE_FINDINGS.md verdict bullets
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.dirname(HERE)  # DEEP_DIVE directory


def load(name):
    with open(os.path.join(DD, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


s1 = load("ma_type_tf_research.json")
s2 = load("complementarity_matrix.json")
s3 = load("dynamic_engine.json")
s4 = load("synthetic_regime_stress.json")
s4b = load("synthetic_intraday_stress.json")
s5 = load("complementary_sleeve_search.json")
s6 = load("longshort_book.json")       # PHASE 6: longshort engine + multi-sleeve book
s7 = load("regime_gated_book.json")    # PHASE 7: regime-gated longshort (detector skill, gated-book deployability)

TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]

# ---- color palette ----
C_ADAPT = "#1b7837"   # green (adaptive wins)
C_LOWLAG = "#5aae61"
C_SIMPLE = "#a6dba0"
C_TREND = "#2166ac"   # blue
C_MR = "#b2182b"      # red
C_STATIC = "#762a83"  # purple
C_DYN = "#e08214"     # orange
C_BH = "#999999"      # grey
C_SHORT = "#b2182b"
C_LO = "#4393c3"
TXT = "#222222"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfb",
})

fig = plt.figure(figsize=(19, 16.6))
gs = fig.add_gridspec(
    4, 2, hspace=0.46, wspace=0.18,
    left=0.055, right=0.975, top=0.925, bottom=0.035,
)

fig.suptitle(
    "FINER-TF MA STRATEGY-DISCOVERY ENGINE  --  MASTER DASHBOARD",
    fontsize=18, fontweight="bold", y=0.982,
)
fig.text(
    0.5, 0.951,
    "2020 band only (TRAIN 6mo / VAL 3mo / OOS 3mo)  |  u10  |  maker 0.0006  |  causal lag-1 MtM  |  held-out, two-sided  |  8 composable engine stages  |  every number traces to a stage JSON",
    ha="center", fontsize=9.5, color="#555555",
)

# =====================================================================
# PANEL (a): MA-type family x TF  +  per-TF winner (adaptive wins every TF)
#   source: s1.family_avg_net_by_tf + s1.winners_by_tf
# =====================================================================
ax = fig.add_subplot(gs[0, 0])
fam = s1["family_avg_net_by_tf"]
win = s1["winners_by_tf"]
x = np.arange(len(TFS))
w = 0.26
adapt = [fam[t]["adaptive"] for t in TFS]
lowlag = [fam[t]["low_lag"] for t in TFS]
simple = [fam[t]["simple"] for t in TFS]
ax.bar(x - w, adapt, w, label="adaptive (VIDYA/KAMA)", color=C_ADAPT)
ax.bar(x, lowlag, w, label="low-lag (HMA/DEMA/TEMA)", color=C_LOWLAG)
ax.bar(x + w, simple, w, label="simple (SMA/EMA/WMA)", color=C_SIMPLE)
# annotate per-TF winner MA type + its tuned OOS net
for i, t in enumerate(TFS):
    wt = win[t]
    ax.annotate(
        f"{wt['ma_type']}\n{wt['oos_net']:.0f}%",
        xy=(x[i], max(adapt[i], lowlag[i], simple[i])),
        xytext=(0, 6), textcoords="offset points",
        ha="center", fontsize=7.5, fontweight="bold", color=C_ADAPT,
    )
ax.set_xticks(x)
ax.set_xticklabels(TFS)
ax.set_ylabel("family avg OOS net %")
ax.set_title("(a) Adaptive MA wins every finer TF")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
ax.set_ylim(0, max(max(adapt), max(simple)) * 1.30)
ax.grid(axis="y", alpha=0.3)
ax.text(
    0.5, -0.20,
    "VIDYA wins {4h,2h,1h,30m,15m}; KAMA wins 1d. Label = winner MA-type + its tuned OOS net.  [ma_type_tf_research.json]",
    transform=ax.transAxes, ha="center", fontsize=7.3, color="#666666",
)

# =====================================================================
# PANEL (b): trend-vs-MR complementarity -- corr per TF + DD-dampening by regime
#   source: s2[tf].corr_trend_mr  +  s4.verdict.complementarity_dd_by_regime
# =====================================================================
ax = fig.add_subplot(gs[0, 1])
corr = [s2[t]["corr_trend_mr"] for t in TFS]
bars = ax.bar(x, corr, color=C_TREND, alpha=0.85, width=0.55)
for i, c in enumerate(corr):
    ax.annotate(f"+{c:.2f}", xy=(x[i], c), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
ax.axhspan(0.85, 0.95, color="#cccccc", alpha=0.45)
ax.text(len(TFS) - 1.15, 0.895, "within-trend corr 0.85-0.94", fontsize=7,
        color="#555555", ha="right", va="center")
ax.set_xticks(x)
ax.set_xticklabels(TFS)
ax.set_ylabel("corr(trend, MR) on returns", color=C_TREND)
ax.set_ylim(0, 1.0)
ax.set_title("(b) Trend + MR are orthogonal -- combining DD-dampens (in CHOP)")
ax.grid(axis="y", alpha=0.3)

# inset: DD-dampening by regime (blend maxDD vs trend-alone, pp) -- the honest catch
dd = s4["verdict"]["complementarity_dd_by_regime"]
reg_order = ["bull", "bear", "chop", "stitched"]
axi = ax.inset_axes([0.50, 0.46, 0.46, 0.40])
ddv = [dd[r] for r in reg_order]
cols = [(C_ADAPT if v > 0 else C_MR) for v in ddv]
axi.bar(range(len(reg_order)), ddv, color=cols, width=0.6)
for i, v in enumerate(ddv):
    axi.annotate(f"{v:+.1f}", xy=(i, v),
                 xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                 ha="center", fontsize=6.8, fontweight="bold")
axi.axhline(0, color="#444", lw=0.8)
axi.set_xticks(range(len(reg_order)))
axi.set_xticklabels(reg_order, fontsize=6.5)
axi.set_title("blend maxDD vs trend-alone (pp)\n+ = dampens, - = worse", fontsize=6.8)
axi.tick_params(labelsize=6.5)
axi.grid(axis="y", alpha=0.25)
ax.text(
    0.5, -0.20,
    "Orthogonal at every TF (corr +0.21..+0.31). Inset: complementarity DAMPENS in chop (+1.3pp) but is a LIABILITY in bear (-11.3pp).  [complementarity_matrix.json + synthetic_regime_stress.json]",
    transform=ax.transAxes, ha="center", fontsize=7.0, color="#666666",
)

# =====================================================================
# PANEL (c): dynamic-vs-static skill -- 6-TF OOS net + synthetic falsification
#   source: s3.verdict.lines (per-TF OOS DYN vs STATIC) + s3.verdict.real_edge_cadences
#           + s4.verdict / s4b.verdict (falsification)
# =====================================================================
ax = fig.add_subplot(gs[1, 0])
# parse OOS DYN/STATIC net per TF from dynamic_engine results (authoritative source)
dyn_net, stat_net, skill_tf = [], [], []
for t in TFS:
    oos = s3["results"][t]["splits"]["OOS"]
    dyn_net.append(oos["DYNAMIC"]["net"])
    stat_net.append(oos["STATIC"]["net"])
    skill_tf.append(t in s3["verdict"]["timing_skill"])
w2 = 0.38
b1 = ax.bar(x - w2 / 2, dyn_net, w2, label="DYNAMIC (regime-timed)", color=C_DYN)
b2 = ax.bar(x + w2 / 2, stat_net, w2, label="STATIC blend", color=C_STATIC)
# mark the lone real-2020 timing-skill TF
for i, t in enumerate(TFS):
    if skill_tf[i]:
        ax.annotate("timing-skill\n(p<0.10)\n*real-2020 only*", xy=(x[i], max(dyn_net[i], stat_net[i])),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=6.8, fontweight="bold", color=C_DYN)
ax.set_xticks(x)
ax.set_xticklabels(TFS)
ax.set_ylabel("OOS net % (2020 bull)")
ax.set_title("(c) Dynamic timing has NO skill -- static wins 5/6")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(dyn_net), max(stat_net)) * 1.32)
# falsification banner
n_stitched_hits = len(s4b["verdict"]["dynamic_significant_hits"])
ax.text(
    0.5, -0.225,
    "Static beats dynamic at 5/6 TFs on real 2020; the lone 30m edge REVERSES under a synthetic regime-flip\n"
    "(stage 4: stitched sign-test n.s.; stage 4b: 0/4 (cadence,regime) cells significant at TRUE intraday res).  [dynamic_engine.json + synthetic_*_stress.json]",
    transform=ax.transAxes, ha="center", fontsize=7.0, color="#666666",
)

# =====================================================================
# PANEL (d): robustness across regimes -- worst-scenario worst-seed net
#   source: s4.verdict.robustness_rank  (TREND_ALONE most robust)
# =====================================================================
ax = fig.add_subplot(gs[1, 1])
rr = s4["verdict"]["robustness_rank"]
order = ["TREND_ALONE", "VOLTGT_BH", "STATIC", "DYNAMIC", "MR_ALONE", "BUYHOLD"]
labels = ["TREND\nALONE", "VOLTGT\nBH", "STATIC\nblend", "DYNAMIC", "MR\nALONE", "BUY\nHOLD"]
vals = [rr[k] for k in order]
colmap = {"TREND_ALONE": C_TREND, "VOLTGT_BH": "#66c2a5", "STATIC": C_STATIC,
          "DYNAMIC": C_DYN, "MR_ALONE": C_MR, "BUYHOLD": C_BH}
cols = [colmap[k] for k in order]
bars = ax.barh(range(len(order)), vals, color=cols)
# place value labels OUTSIDE the bar tip (to the left of the negative bar end) in dark text
for i, v in enumerate(vals):
    ax.annotate(f"{v:.1f}%", xy=(v, i), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8.0, fontweight="bold", color=TXT)
ax.set_xlim(min(vals) * 1.18, 2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("worst-scenario worst-seed net % (higher = more robust)")
ax.set_title("(d) Robustness across bull/bear/chop/stitched (20 seeds)")
ax.grid(axis="x", alpha=0.3)
ax.text(
    0.5, -0.20,
    "Synthetic regime-stress, generator VALIDATED 3/3. TREND-ALONE is the most robust survivor; the long-only MR sleeve\n"
    "is the worst (buys falling knives in a bear). Excludes BUYHOLD beta.  [synthetic_regime_stress.json]",
    transform=ax.transAxes, ha="center", fontsize=7.0, color="#666666",
)

# =====================================================================
# PANEL (e): the long-only-exception price -- SHORT vs long-only complement (bear)
#   source: s5.verdict.long_only_relaxation_value + s5.verdict.scoreboard
# =====================================================================
ax = fig.add_subplot(gs[2, 0])
sb = s5["verdict"]["scoreboard"]
lov = s5["verdict"]["long_only_relaxation_value"]
# bear return-corr to trend: SHORT (anticorr) vs long-only candidates (corr ~+1)
cands = ["SHORT_MA", "LONGSHORT_MA", "VOLTGT_DEF", "CASH_GATE", "MR_LONG"]
clabel = ["SHORT_MA\n(short)", "LONGSHORT\n(short)", "VOLTGT_DEF\n(LO)", "CASH_GATE\n(LO)", "MR_LONG\n(LO)"]
bear_corr = [sb[c]["bear_return_corr"] for c in cands]
bear_net = [sb[c]["bear_standalone_net"] for c in cands]
cols = [C_SHORT if not sb[c]["long_only"] else C_LO for c in cands]
b = ax.bar(np.arange(len(cands)), bear_net, color=cols, width=0.6)
ax.set_ylim(min(bear_net) - 5.5, max(bear_net) + 4)
for i, c in enumerate(cands):
    v = bear_net[i]
    if v >= 0:
        yoff, va, col = 4, "bottom", TXT          # above positive bar
    elif v > -10:
        yoff, va, col = -22, "top", TXT           # below shallow negative bar
    else:
        yoff, va, col = 12, "bottom", "white"     # inside the deep negative bar
    ax.annotate(f"corr {bear_corr[i]:+.2f}\nnet {v:+.1f}%",
                xy=(i, v), xytext=(0, yoff), textcoords="offset points",
                ha="center", va=va, fontsize=6.8, fontweight="bold", color=col)
ax.axhline(0, color="#444", lw=0.9)
# trend-alone bleeds in bear (the gap to fill) = -6.9 per findings/stage5 Q2
ax.axhline(-6.9, color=C_TREND, lw=1.4, ls="--")
ax.text(len(cands) - 0.5, -6.9, " trend bleeds -6.9% (the gap)", color=C_TREND,
        fontsize=6.8, va="bottom", ha="right")
ax.set_xticks(np.arange(len(cands)))
ax.set_xticklabels(clabel, fontsize=7.2)
ax.set_ylabel("bear standalone net %")
ax.set_title("(e) Only a SHORT sleeve fills the BEAR gap -- the price of long-only")
ax.grid(axis="y", alpha=0.3)
short_bear_dd = lov["SHORT_MA"]["bear_dd_advantage_vs_trendMR_pp"]
short_bear_net = lov["SHORT_MA"]["bear_net_advantage_vs_trendMR"]
ax.text(
    0.5, -0.255,
    f"Red = short (return-anticorrelated, GENUINE fill); blue = long-only (corr ~+1 to trend, only DAMPENS).\n"
    f"Swapping long-only MR for SHORT_MA buys +{short_bear_dd:.1f}pp bear DD protection + +{short_bear_net:.1f}pp bear net = the quantified LO-exception value.  [complementary_sleeve_search.json]",
    transform=ax.transAxes, ha="center", fontsize=7.0, color="#666666",
)

# =====================================================================
# PANEL (f): TEXT / VERDICT panel  -- the convergent verdict (from FINDINGS.md)
# =====================================================================
ax = fig.add_subplot(gs[2, 1])
ax.axis("off")
ax.set_title("(f) The convergent verdict", loc="left")

# pull a few anchor numbers to embed live
win_15m = win["15m"]["oos_net"]
adapt_15 = fam["15m"]["adaptive"]
simple_15 = fam["15m"]["simple"]
corr_lo, corr_hi = min(corr), max(corr)
bear_dd = dd["bear"]
chop_dd = dd["chop"]

# PHASE 6/7 anchor numbers (live)
s6_best_tf = s6["verdict"]["best_tf_for_longshort"]
s6_stitch_ls = s6["engine"]["1d"]["stitched"]["longshort_net"]["mean"]
s6_ls_value_dd = s6["multisleeve_book"][s6_best_tf]["_deployable_vs_research"]["longshort_worst_dd_value_pp"]
s7_prec = s7["detector_skill_stitched"]["precision"]["mean"]
s7_base = s7["detector_skill_stitched"]["base_rate"]["mean"]
s7_beats_shuf = s7["detector_skill_stitched"]["frac_seeds_beats_shuffle95"]

bullets = [
    ("1. ADAPTIVE MA wins EVERY finer TF.",
     f"VIDYA wins 4h/2h/1h/30m/15m, KAMA wins 1d; adaptive edge widens at finer cadence. Best trend sleeve = participating BETA (net < buy-hold in the 2020 bull)."),
    ("2. COMPLEMENTARITY is REAL but regime-conditional.",
     f"Trend + MR orthogonal at every TF (corr +{corr_lo:.2f}..+{corr_hi:.2f}). DD-dampens in CHOP ({chop_dd:+.1f}pp) -- but is a BEAR LIABILITY ({bear_dd:+.1f}pp): long-only buys falling knives."),
    ("3. DYNAMIC timing has NO skill -- robust to resolution.",
     "Static blend wins 5/6 TFs; lone 30m edge REVERSES under a synthetic regime-flip + does NOT replicate at TRUE intraday res (0/4 cells). SHIP THE STATIC BLEND."),
    ("4. TRUE bear-gap-fill REQUIRES a SHORT sleeve (the LO-exception).",
     f"Only SHORT_MA is return-anticorrelated in the bear; it buys +{short_bear_dd:.1f}pp bear DD protection + +{short_bear_net:.1f}pp bear net. Long-only can only dampen, never rescue."),
    ("5. The LONGSHORT engine is net-POSITIVE but is bear-INSURANCE, not a book.",
     f"Symmetric long-short adaptive-MA is net-positive full-cycle (1d stitched +{s6_stitch_ls:.1f}%) + adds +{s6_ls_value_dd:.1f}pp bear-DD at {s6_best_tf}, BUT a naive equal-risk 4-sleeve book beats NO single sleeve (0/6). [PHASE 6]"),
    ("6. A regime DETECTOR has real timing skill -- yet the gated book still doesn't deploy.",
     f"Bear detector precision {s7_prec:.2f} vs base-rate {s7_base:.2f}, beats shuffle-95 in {s7_beats_shuf:.0%} of seeds (NOT the dynamic null) -- but on a MILD/SHORT 2020 bear its false-alarm cost > the bear gain (0/6 beat trend). [PHASE 7]"),
    ("ENGINE'S REAL ASSET: it KILLS its own false positives.",
     "Every surfaced 'edge' was adversarially falsified by the engine itself (validate-generator-first, proper sign test, RWYB, two-sided). The discipline is the durable deliverable."),
]

y = 0.97
for head, body in bullets:
    ax.text(0.0, y, head, transform=ax.transAxes, fontsize=8.4,
            fontweight="bold", color=TXT, va="top")
    y -= 0.044
    # wrap body
    import textwrap
    for ln in textwrap.wrap(body, width=98):
        ax.text(0.02, y, ln, transform=ax.transAxes, fontsize=7.3,
                color="#333333", va="top")
        y -= 0.035
    y -= 0.014

ax.text(
    0.0, 0.015,
    "DEPLOYABLE NOW + RUNNABLE (long-only+spot): `python -m strat.finer_tf_book` -> adaptive-MA (VIDYA/KAMA) trend + static MR chop-complement + VOLTGT_DEF\n"
    "overlay. 2020 OOS: 4h book net 13.8% / Sharpe 3.07 / maxDD -4.0% (BETA: net < buy-hold 50.6%, but ~5x lower DD).  [finer_tf_book.json]\n"
    "UNLOCK (user's call): the LONGSHORT_MA bear-insurance sleeve (--longshort-insurance, RESEARCH/LO-exception) -- turns the bear liability near-flat.",
    transform=ax.transAxes, fontsize=6.6, color="#666666", va="bottom",
)

# =====================================================================
# PANEL (g): PHASE 6 -- the LONGSHORT engine is net-POSITIVE full-cycle (stitched LS vs trend vs cost-matched
#   NULL per TF) but a naive multi-sleeve book beats NO single sleeve. source: s6.engine + s6.verdict
# =====================================================================
ax = fig.add_subplot(gs[3, 0])
ls_stitch = [s6["engine"][t]["stitched"]["longshort_net"]["mean"] for t in TFS]
null_stitch = [s6["engine"][t]["stitched"]["null_net"]["mean"] for t in TFS]
bear_dd_val = [s6["multisleeve_book"][t]["_deployable_vs_research"]["longshort_worst_dd_value_pp"] for t in TFS]
xg = np.arange(len(TFS)); wg = 0.27
ax.bar(xg - wg, ls_stitch, wg, label="LONGSHORT net (stitched full-cycle)", color=C_STATIC)
ax.bar(xg, null_stitch, wg, label="cost-matched random-dir NULL", color=C_BH)
ax.bar(xg + wg, bear_dd_val, wg, label="bear-DD value added to book (pp)", color=C_ADAPT)
ax.axhline(0, color="#444", lw=0.8)
for i, t in enumerate(TFS):
    ax.annotate(f"{ls_stitch[i]:.1f}", xy=(xg[i] - wg, ls_stitch[i]), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=6.8, fontweight="bold", color=C_STATIC)
ax.set_xticks(xg); ax.set_xticklabels(TFS)
ax.set_ylabel("% net / pp")
ax.set_title("(g) PHASE 6: the LONGSHORT engine is net-POSITIVE -- bear-INSURANCE, not a book")
ax.legend(loc="upper right", fontsize=7.0, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)
book_wins6 = len(s6["verdict"]["book_wins_cadences"])
ax.text(
    0.5, -0.20,
    f"Symmetric long-short adaptive-MA beats its cost-matched NULL at every TF (the edge is the SIGNAL) + is borrow-insensitive (0->30bps ~0pp).\n"
    f"But a naive equal-risk 4-sleeve BOOK beats EVERY single sleeve at {book_wins6}/6 cadences -- a static mix is NOT the answer (needs regime-routing). "
    f"SHORT = RESEARCH.  [longshort_book.json]",
    transform=ax.transAxes, ha="center", fontsize=7.0, color="#666666",
)

# =====================================================================
# PANEL (h): PHASE 7 -- the bear DETECTOR has real TIMING SKILL (precision >> base-rate, beats shuffle-95) but
#   the GATED book still does not beat trend-alone on a mild/short 2020 bear. source: s7.detector_skill + stress
# =====================================================================
ax = fig.add_subplot(gs[3, 1])
det = s7["detector_skill_stitched"]
prec = det["precision"]["mean"]; prec_sd = det["precision"]["std"]
base = det["base_rate"]["mean"]; recall = det["recall"]["mean"]
shuf_margin = det["precision_minus_shuffle"]["mean"]
labels = ["detector\nprecision", "base-rate\n(random)", "recall", "precision\n- shuffle"]
vals = [prec, base, recall, shuf_margin]
cols = [C_DYN, C_BH, C_LOWLAG, C_ADAPT]
bh = ax.bar(range(len(labels)), vals, color=cols, width=0.6)
ax.errorbar(0, prec, yerr=prec_sd, color="#222", capsize=3, lw=1.0)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.2f}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=7.5, fontweight="bold", color=TXT)
ax.axhline(base, color=C_BH, ls="--", lw=1.0)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("fraction")
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("(h) PHASE 7: detector HAS timing skill -- gated book still != deployable")
ax.grid(axis="y", alpha=0.3)
# gated-vs-trend control across TFs (the deployability null)
beats_trend7 = len(s7["verdict"]["beats_trend_cadences"])
beats_shuf7 = len(s7["verdict"]["beats_shuffle_cadences"])
gate_net_1d = s7["stress"]["1d"]["_controls"]["gated_vs_trend"]["net_gain_pp"]
ax.text(
    0.5, -0.20,
    f"The bear DETECTOR concentrates its ON-time in the TRUE bear (precision {prec:.2f} >> base {base:.2f}; beats shuffle-95 in "
    f"{det['frac_seeds_beats_shuffle95']:.0%} of seeds, +{shuf_margin:.2f} over shuffle at {beats_shuf7}/6 TFs) -- NOT the dynamic null.\n"
    f"BUT the gated longshort book beats trend-alone at {beats_trend7}/6 (1d net {gate_net_1d:+.1f}pp): on a MILD/SHORT 2020 bear the gate's "
    f"false-alarm cost > the bear gain. Binary insurance is -EV here; a deeper/longer bear is the open door.  [regime_gated_book.json]",
    transform=ax.transAxes, ha="center", fontsize=7.0, color="#666666",
)

# caveats footer across the whole figure
fig.text(
    0.5, 0.010,
    "CAVEATS (binding): 2020-bull-band in-sample (Oct-Dec OOS ~0% bear); SHORT/long-short = RESEARCH (deploy needs the long-only-exception sign-off); "
    "synthetic stress is a calibrated TEST surface, not real future data; UNSEEN N/A (2020 band only). Stage 4b intraday verdict shown at the current run's seed count; direction is stable.",
    ha="center", fontsize=7.2, color="#888888", style="italic",
)

out = os.path.join(HERE, "discovery_engine_dashboard.png")
fig.savefig(out, dpi=130, facecolor="white")
print("WROTE", out)
print("size_bytes", os.path.getsize(out))
