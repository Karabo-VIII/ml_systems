"""MOVER-TIMING -- decomposed mover problem #2 (TIMING / the fizzle-filter).

THE QUESTION (two parts; the 2nd is the lever):
  (1) BURST-ONSET DETECTION: at 1m, can we DETECT that a magnitude-BURST is STARTING
      in the next K minutes from the PRE-burst micro-state (vol compression, volume
      build/accel, liq, funding, OI, recent path shape)? Held-out OOS AUC. And -- can
      we act on it: how much LEAD TIME does the fire give vs how much of the burst is
      already gone?
  (2) THE FIZZLE-FILTER (the lever): take the lane-#4 trigger set (first minute whose
      CLOSE >= day_open*(1+1.5%)). Only ~31% of triggers become >=5% intraday movers;
      the rest fizzle, and lane-#4 proved NO exit rescues a coin-flip entry. Build a
      CAUSAL classifier AT the trigger minute (data <= trigger ONLY, incl. the
      magnitude-continuation features) predicting "this trigger becomes a >=5% mover
      (not a fizzle)". Report OOS AUC AND -- crucially -- the CONVERSION HIT-RATE in the
      top-quantile (does filtering to top-30% / top-10% raise the 31% mover-rate to
      50%+?). Then re-run the lane-#4 ride-to-close net%/event ON THE FILTERED triggers:
      does a better ENTRY make capture net-POSITIVE after 24bps OOS?

GROUNDING (trusted, not re-derived):
  - The daily move is a single-hour BURST (51% of the net move in one hour).
  - Lane #4 (mover_ride): unconditional ride bleeds because the ENTRY is a coin-flip
    (~31% trigger->mover conversion); no exit can rescue it.
  - Lane #3 (mover_continuation): DIRECTIONAL continuation is information-bound dead
    (OOS AUC 0.52) but MAGNITUDE-continuation is robustly predictable (OOS AUC 0.73).
    The magnitude-continuation feature family is the candidate filter content.
  - Prior vol-expansion work was killed as a DIRECTIONAL edge (6h-session C1: magnitude
    not direction, drift+concentrated+maker-fragile). Here it is reused as a MAGNITUDE
    burst-timing / fizzle-filter, a DIFFERENT use -- the burst label is sign-agnostic
    (Part 1) and the fizzle label is "becomes a big-runup mover" (Part 2, magnitude not
    direction-of-day). The filter does NOT claim to predict direction.

HONEST HELD-OUT (mandatory, two-sided):
  TRAIN < 2024-01-01 fits everything (models, z-stats, thresholds, quantile cutoffs).
  OOS 2024-01-01..2025-07-01 scored ONCE. UNSEEN >= 2025-07-01 NEVER touched.
  Every feature uses data <= the decision minute; forward labels from next-minute open.
  SHUFFLED-LABEL null per model is the false-positive floor: a filter that is TRAIN-only
  or does not beat its shuffled-null is a NULL and is reported as such. The decisive test
  for the lever is whether the FILTERED-cohort ride net beats the ALL-trigger ride net
  OUT OF SAMPLE -- not in TRAIN, not on the conversion rate alone.

Reuses src/mining/cascade_oracle.load_panel (1m grid + OI/funding/liq) and the lane-#4
trigger + trail-exit mechanics (re-implemented here to keep the script standalone).

Run:
  python -m mining.mover_burst_timing --universe u10
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mining.cascade_oracle import load_panel  # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"liq_subbar_1m": "via cascade_oracle.load_panel (1m + OI/funding/liq)"},
    "outputs": {"study_json": "runs/mining/mover_burst_timing_<tag>_<stamp>.json"},
    "invariants": {
        "train_only_fit": "models, z-stats, quantile thresholds ALL from TRAIN only",
        "causal_features": "every feature computed from data <= the decision minute",
        "forward_label_next_open": "burst + ride labels measured from next-minute open",
        "unseen_untouched": "no UNSEEN row enters features, labels, or reports",
        "shuffled_null": "each model reports a shuffled-label AUC null as the FP floor",
        "lever_test_oos": "FILTERED ride net must beat ALL-trigger ride net OUT OF SAMPLE",
    },
}

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY_MS = 86_400_000

# ---- Part-1 burst-onset config ----
BURST_X = 0.02           # a "burst" = an absolute price move of >= 2% ...
BURST_K = 30             # ... that completes within the next 30 minutes
BURST_STEP = 10          # stride for the burst-detection scan (every 10th minute, dense enough)

# ---- Part-2 fizzle-filter config (lane-#4 inheritance) ----
T_TRIG = 0.015           # lane-#4 onset: first close >= day_open*(1+1.5%)
MOVER_RUNUP = 0.05       # "becomes a mover" = day runup-from-open >= 5%
MIN_MINUTES_LEFT = 60    # trigger must leave >=60m of day to ride/measure
TRAIL_K = 0.02           # lane-#4 trail (2% high-water) for the ride re-run
RIDE_COST_RT = 0.0024    # 24bps round-trip canonical gate
TOP_QUANTILES = [0.30, 0.10]  # report conversion hit-rate in top-30% and top-10%

SEED = 7


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


# ===================================================================== shared panel

def _panel_arrays(sym: str) -> dict:
    df = load_panel(sym)
    return {
        "ms": df["minute_ts"].to_numpy(),
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": df["close"].to_numpy(),
        "vol": df["vol_usd"].fill_null(0.0).to_numpy(),
        "buy": df["buy_aggr_usd"].fill_null(0.0).to_numpy(),
        "sell": df["sell_aggr_usd"].fill_null(0.0).to_numpy(),
        "liq_long": df["liq_long_usd"].fill_null(0.0).to_numpy(),
        "liq_short": df["liq_short_usd"].fill_null(0.0).to_numpy(),
        "ret1m": df["ret_1m"].to_numpy(),
        "pre_vol": df["pre_vol"].to_numpy(),
        "liq_ratio": df["liq_ratio"].to_numpy(),
        "oi1": df["oi_d1h"].to_numpy(),
        "oi4": df["oi_d4h"].to_numpy(),
        "oi24": df["oi_d24h"].to_numpy(),
        "fund": df["funding"].to_numpy(),
        "regime": df["regime_above_sma200"].to_numpy(),
        "real": df["is_real"].to_numpy().astype(bool),
    }


def _roll(a, w, fn="sum"):
    """Causal trailing rolling stat over window w (NaN until w-1)."""
    out = np.full(len(a), np.nan)
    if fn == "sum":
        c = np.cumsum(np.insert(np.nan_to_num(a), 0, 0.0))
        out[w - 1:] = c[w:] - c[:-w]
    elif fn == "std":
        # rolling std via cumulative sums (population)
        x = np.nan_to_num(a)
        c1 = np.cumsum(np.insert(x, 0, 0.0))
        c2 = np.cumsum(np.insert(x * x, 0, 0.0))
        s1 = c1[w:] - c1[:-w]
        s2 = c2[w:] - c2[:-w]
        var = np.maximum(s2 / w - (s1 / w) ** 2, 0.0)
        out[w - 1:] = np.sqrt(var)
    return out


def _micro_features(P: dict, m: int, day_start: int) -> dict:
    """Strictly-causal pre-decision micro-state at minute m (uses data <= m only).

    Shared by Part 1 (burst onset) and Part 2 (fizzle filter); Part 2 adds day-anchored
    and continuation features on top. Window stats end at m inclusive."""
    close = P["close"]
    vol = P["vol"]
    buy = P["buy"]
    sell = P["sell"]
    ret1m = P["ret1m"]
    f = {}

    # --- realized-vol over recent windows + COMPRESSION (short/long ratio) ---
    def rvol(w):
        lo = m - w + 1
        if lo < day_start - 1440:   # bounded to ~1d of history
            lo = max(0, m - w + 1)
        seg = ret1m[max(0, m - w + 1):m + 1]
        seg = seg[np.isfinite(seg)]
        return float(np.std(seg)) if len(seg) >= max(5, w // 3) else np.nan
    rv5, rv15, rv30, rv60, rv240 = rvol(5), rvol(15), rvol(30), rvol(60), rvol(240)
    f["rv15"] = rv15
    f["rv60"] = rv60
    # compression: recent short-window vol vs the slower 240m vol (low => coiled)
    f["vol_compression"] = (rv30 / rv240) if (np.isfinite(rv30) and np.isfinite(rv240) and rv240 > 0) else np.nan
    f["vol_ratio_5_60"] = (rv5 / rv60) if (np.isfinite(rv5) and np.isfinite(rv60) and rv60 > 0) else np.nan

    # --- volume build + acceleration ---
    v5 = float(np.sum(vol[max(0, m - 4):m + 1]))
    v30 = float(np.sum(vol[max(0, m - 29):m + 1]))
    v240 = float(np.sum(vol[max(0, m - 239):m + 1]))
    vbar30 = v240 / 240.0 * 30.0 if v240 > 0 else np.nan          # expected 30m volume
    f["vol_build"] = (v30 / vbar30) if (np.isfinite(vbar30) and vbar30 > 0) else np.nan
    f["vol_accel"] = ((v5 / 5.0) / (v30 / 30.0)) if v30 > 0 else np.nan   # last-5m pace vs 30m pace

    # --- aggressor imbalance + acceleration ---
    b30 = float(np.sum(buy[max(0, m - 29):m + 1]))
    s30 = float(np.sum(sell[max(0, m - 29):m + 1]))
    b5 = float(np.sum(buy[max(0, m - 4):m + 1]))
    s5 = float(np.sum(sell[max(0, m - 4):m + 1]))
    f["aggr_imb30"] = ((b30 - s30) / (b30 + s30)) if (b30 + s30) > 0 else np.nan
    imb5 = ((b5 - s5) / (b5 + s5)) if (b5 + s5) > 0 else np.nan
    f["aggr_accel"] = (imb5 - f["aggr_imb30"]) if (np.isfinite(imb5) and np.isfinite(f["aggr_imb30"])) else np.nan

    # --- liquidation flow (forced activity precedes/accompanies bursts) ---
    ll30 = float(np.sum(P["liq_long"][max(0, m - 29):m + 1]))
    ls30 = float(np.sum(P["liq_short"][max(0, m - 29):m + 1]))
    f["liq_tot30_z"] = np.log1p(ll30 + ls30)          # magnitude (z'd per-asset later)
    f["liq_ls_asym"] = ((ll30 - ls30) / (ll30 + ls30)) if (ll30 + ls30) > 0 else 0.0
    f["liq_ratio"] = float(P["liq_ratio"][m]) if np.isfinite(P["liq_ratio"][m]) else np.nan

    # --- OI / funding (positioning fuel) ---
    f["oi_d1h"] = float(P["oi1"][m]) if np.isfinite(P["oi1"][m]) else np.nan
    f["oi_d4h"] = float(P["oi4"][m]) if np.isfinite(P["oi4"][m]) else np.nan
    f["funding"] = float(P["fund"][m]) if (P["fund"][m] is not None and np.isfinite(P["fund"][m])) else np.nan

    # --- recent path SHAPE ---
    r15 = float(close[m] / close[max(0, m - 15)] - 1.0)
    r60 = float(close[m] / close[max(0, m - 60)] - 1.0)
    f["ret15m"] = r15
    f["ret60m"] = r60
    # straight-line vs path-length (efficiency: 1 = straight trend, ~0 = chop)
    seg = close[max(0, m - 60):m + 1]
    step = np.abs(np.diff(seg))
    pathlen = float(np.sum(step))
    f["path_efficiency"] = (abs(seg[-1] - seg[0]) / pathlen) if pathlen > 0 else np.nan
    f["regime"] = (1.0 if P["regime"][m] else 0.0) if P["regime"][m] is not None else np.nan
    f["hour_utc"] = float((P["ms"][m] % DAY_MS) / 3_600_000)
    return f


# ===================================================================== PART 1: burst onset

BURST_FEATS = ["vol_compression", "vol_ratio_5_60", "rv15", "rv60", "vol_build",
               "vol_accel", "aggr_imb30", "aggr_accel", "liq_tot30_z", "liq_ls_asym",
               "liq_ratio", "oi_d1h", "oi_d4h", "funding", "ret15m", "ret60m",
               "path_efficiency", "regime", "hour_utc"]


def burst_events(sym: str) -> list[dict]:
    """Sample minutes on a stride; label 'a >=BURST_X move STARTS within BURST_K min',
    measured forward from the next-minute open (causal). Features at the decision close.
    A 'starting' burst means the move is NOT already underway: the recent 15m has not
    itself been a burst (so the label is onset, not mid-burst)."""
    P = _panel_arrays(sym)
    ms, close, opens, real = P["ms"], P["close"], P["open"], P["real"]
    n = len(ms)
    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    # map each minute to its day-start (for intraday window bounding)
    ds_of = np.zeros(n, dtype=np.int64)
    cur = 0
    bounds = np.append(day_starts, n)
    for di in range(len(day_starts)):
        ds_of[bounds[di]:bounds[di + 1]] = day_starts[di]

    out = []
    lo = 1500          # need history for the slow windows
    hi = n - (BURST_K + 5)
    for m in range(lo, hi, BURST_STEP):
        sp = split_of(int(ms[m]))
        if sp == "UNSEEN":
            continue
        if not real[m] or not real[m + 1]:
            continue
        if real[max(0, m - 240):m + 1].mean() < 0.90:
            continue
        entry = opens[m + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        # forward path from next-minute open over K minutes
        fwd = close[m + 1:m + 1 + BURST_K]
        if len(fwd) < BURST_K or not np.all(np.isfinite(fwd)) or real[m + 1:m + 1 + BURST_K].mean() < 0.90:
            continue
        fwd_rel = fwd / entry - 1.0
        burst_mag = float(np.max(np.abs(fwd_rel)))
        is_burst = int(burst_mag >= BURST_X)
        # onset guard: the last 15m must NOT already be a >=BURST_X move (avoid labeling
        # mid-burst as onset; we want the START)
        already = abs(close[m] / close[max(0, m - 15)] - 1.0) >= BURST_X
        if already:
            continue
        # lead-time: first minute index (1..K) at which the |move| first reaches BURST_X
        lead = None
        if is_burst:
            reach = np.flatnonzero(np.abs(fwd_rel) >= BURST_X)
            lead = int(reach[0]) + 1 if len(reach) else None
        f = _micro_features(P, m, int(ds_of[m]))
        out.append({"sym": sym, "ms": int(ms[m]), "split": sp,
                    "is_burst": is_burst, "burst_mag": burst_mag, "lead_min": lead,
                    "feats": f})
    return out


# ===================================================================== PART 2: fizzle filter

# continuation feature family (lane #3, magnitude-AUC-0.73 content) + lane-#4 trigger context
FIZZLE_FEATS = BURST_FEATS + ["overshoot", "t2trig", "day_run_pace", "run_accel",
                              "dayvol_ratio"]


def trail_exit(opens, closes, f, end, k):
    """Lane-#4 ride: from fill row f to day end with a k% high-water trail. Returns gross
    return. (Re-implemented from mover_ride.trail_exit, identical mechanics.)"""
    entry = opens[f]
    c = closes[f:end + 1]
    runmax = np.maximum.accumulate(np.maximum(c, entry))
    breach = c < runmax * (1.0 - k)
    idx = int(np.argmax(breach))
    if breach[idx] and (f + idx + 1) <= end:
        return float(opens[f + idx + 1] / entry - 1.0)
    return float(closes[end] / entry - 1.0)


def trigger_events(sym: str) -> list[dict]:
    """Lane-#4 trigger set: first minute whose CLOSE >= day_open*(1+T_TRIG). For each
    trigger: causal features at the trigger close, the fizzle label (day runup-from-open
    >= MOVER_RUNUP), and the lane-#4 ride-to-close gross (for the lever re-run)."""
    P = _panel_arrays(sym)
    ms, opens, closes, highs, real = P["ms"], P["open"], P["close"], P["high"], P["real"]
    n = len(ms)
    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    day_ends = np.append(day_starts[1:] - 1, n - 1)

    out = []
    for ds, de in zip(day_starts, day_ends):
        if de - ds < 1200 or real[ds:de + 1].mean() < 0.90:
            continue
        d_open = opens[ds]
        if not np.isfinite(d_open) or d_open <= 0:
            continue
        sp = split_of(int(ms[ds]))
        if sp == "UNSEEN":
            continue
        rel = closes[ds:de + 1] / d_open - 1.0
        hit = np.flatnonzero(rel >= T_TRIG)
        if len(hit) == 0:
            continue
        m = ds + int(hit[0])        # trigger minute
        f = m + 1                   # fill (next-minute open)
        if de - f < MIN_MINUTES_LEFT or not (real[m] and real[f]):
            continue
        entry = opens[f]
        if not np.isfinite(entry) or entry <= 0:
            continue
        # --- LABEL: does the day become a >=5% mover (runup-from-open)? (magnitude, causal future) ---
        day_runup = float(np.max(highs[ds:de + 1]) / d_open - 1.0)
        is_mover = int(day_runup >= MOVER_RUNUP)
        # --- lane-#4 ride gross (to-close trail) for the lever re-run ---
        ride_gross = trail_exit(opens, closes, f, de, TRAIL_K)
        # --- features: micro-state at trigger close + day-anchored continuation context ---
        feats = _micro_features(P, m, int(ds))
        feats["overshoot"] = float(rel[m - ds] - T_TRIG)
        feats["t2trig"] = float(m - ds)        # minutes from day-open to trigger
        # day-so-far run pace: rel at trigger / minutes elapsed (impulsive vs grinding)
        feats["day_run_pace"] = float(rel[m - ds] / max(m - ds, 1))
        # run acceleration: last-30m pace vs day-so-far average pace
        ret30 = float(closes[m] / closes[max(ds, m - 30)] - 1.0)
        feats["run_accel"] = (ret30 / 30.0) / feats["day_run_pace"] if feats["day_run_pace"] != 0 else np.nan
        # day vol-so-far ratio vs pre-day vol
        dvol = float(np.nanstd(P["ret1m"][ds:m + 1])) if m - ds >= 30 else np.nan
        pv = P["pre_vol"][m]
        feats["dayvol_ratio"] = (dvol / pv) if (np.isfinite(pv) and pv > 0 and np.isfinite(dvol)) else np.nan
        out.append({"sym": sym, "ms": int(ms[m]), "split": sp,
                    "is_mover": is_mover, "day_runup": day_runup,
                    "ride_gross": float(ride_gross), "feats": feats})
    return out


# ===================================================================== modeling

def _zscore_per_asset(evs_tr, evs_oo, feats):
    """Per-(asset,feature) z from TRAIN stats only; applied to TRAIN+OOS. NaN preserved."""
    def mat(evs):
        return np.array([[e["feats"].get(fn, np.nan) for fn in feats] for e in evs], dtype=float)
    Xtr, Xoo = mat(evs_tr), mat(evs_oo)
    sym_tr = np.array([e["sym"] for e in evs_tr])
    sym_oo = np.array([e["sym"] for e in evs_oo])
    Ztr, Zoo = Xtr.copy(), Xoo.copy()
    for s in set(sym_tr):
        mtr, moo = sym_tr == s, sym_oo == s
        for j in range(len(feats)):
            col = Xtr[mtr, j]
            mu, sd = np.nanmean(col), np.nanstd(col)
            if np.isfinite(sd) and sd > 0:
                Ztr[mtr, j] = (col - mu) / sd
                if moo.any():
                    Zoo[moo, j] = (Xoo[moo, j] - mu) / sd
    return Ztr, Zoo


def fit_score(Ztr, ytr, Zoo, yoo, feats, seed):
    """HGB (early-stopped) + logit sanity arm; shuffled-null OOS AUC floor; perm-importance.
    Returns AUCs + the HGB OOS probability scores."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    out = {}
    if len(np.unique(ytr)) < 2 or len(np.unique(yoo)) < 2:
        return {"degenerate": True}, None

    def new_hgb(rs):
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=rs)

    hgb = new_hgb(seed)
    hgb.fit(Ztr, ytr)
    p_tr = hgb.predict_proba(Ztr)[:, 1]
    p_oo = hgb.predict_proba(Zoo)[:, 1]
    out["hgb_train"] = float(roc_auc_score(ytr, p_tr))
    out["hgb_oos"] = float(roc_auc_score(yoo, p_oo))

    med = np.nanmedian(Ztr, axis=0)
    Ztr_i = np.where(np.isnan(Ztr), med, Ztr)
    Zoo_i = np.where(np.isnan(Zoo), med, Zoo)
    logit = LogisticRegression(max_iter=2000, C=1.0)
    logit.fit(Ztr_i, ytr)
    out["logit_train"] = float(roc_auc_score(ytr, logit.predict_proba(Ztr_i)[:, 1]))
    out["logit_oos"] = float(roc_auc_score(yoo, logit.predict_proba(Zoo_i)[:, 1]))

    # shuffled-label null (permute TRAIN labels, refit, OOS AUC against TRUE yoo)
    rng = np.random.default_rng(seed + 99)
    nulls = []
    ysh = ytr.copy()
    for k in range(5):
        rng.shuffle(ysh)
        if len(np.unique(ysh)) < 2:
            continue
        hsh = new_hgb(seed + k)
        hsh.fit(Ztr, ysh)
        nulls.append(float(roc_auc_score(yoo, hsh.predict_proba(Zoo)[:, 1])))
    out["shuffled_oos_auc_mean"] = float(np.mean(nulls)) if nulls else None
    out["shuffled_oos_auc_max"] = float(np.max(nulls)) if nulls else None

    base = out["hgb_oos"]
    imp = {}
    rng2 = np.random.default_rng(seed + 7)
    for j, fn in enumerate(feats):
        Zp = Zoo.copy()
        col = Zp[:, j].copy()
        rng2.shuffle(col)
        Zp[:, j] = col
        imp[fn] = float(base - roc_auc_score(yoo, hgb.predict_proba(Zp)[:, 1]))
    out["perm_importance_oos"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:10])
    return out, p_oo


# ===================================================================== Part-1 analysis

def analyze_burst(events: list[dict], seed: int) -> dict:
    train = [e for e in events if e["split"] == "TRAIN"]
    oos = [e for e in events if e["split"] == "OOS"]
    ytr = np.array([e["is_burst"] for e in train], dtype=int)
    yoo = np.array([e["is_burst"] for e in oos], dtype=int)
    Ztr, Zoo = _zscore_per_asset(train, oos, BURST_FEATS)
    res, p_oo = fit_score(Ztr, ytr, Zoo, yoo, BURST_FEATS, seed)
    res["n_train"] = len(train)
    res["n_oos"] = len(oos)
    res["base_rate_train"] = float(ytr.mean())
    res["base_rate_oos"] = float(yoo.mean())

    # --- LEAD-TIME analysis: among OOS bursts the top-decile model SCORE flags, how much
    #     of the burst is still ahead when the signal fires? We report the realized lead
    #     (minutes to burst completion) for the positive (true-burst) OOS events that the
    #     model ranks in its top-30% score band -- the actionable subset.
    if not res.get("degenerate") and p_oo is not None:
        order = np.argsort(-p_oo)
        k = max(1, int(0.30 * len(oos)))
        top = set(order[:k].tolist())
        leads_top = [oos[i]["lead_min"] for i in range(len(oos))
                     if i in top and oos[i]["is_burst"] and oos[i]["lead_min"] is not None]
        leads_all = [e["lead_min"] for e in oos if e["is_burst"] and e["lead_min"] is not None]
        # precision of the top band (fraction that are true bursts)
        prec_top = float(np.mean([oos[i]["is_burst"] for i in top])) if top else None
        res["lead_time"] = {
            "median_lead_min_top30_truebursts": (float(np.median(leads_top)) if leads_top else None),
            "median_lead_min_all_truebursts": (float(np.median(leads_all)) if leads_all else None),
            "precision_top30": prec_top,
            "base_rate_oos": float(yoo.mean()),
            "n_truebursts_top30": len(leads_top),
            "burst_window_K": BURST_K,
            "note": "lead = minutes from the next-minute fill to first reaching the burst "
                    "magnitude; small lead = most of the burst already gone when actionable",
        }
    return res


# ===================================================================== Part-2 analysis (the lever)

def analyze_fizzle(events: list[dict], seed: int) -> dict:
    train = [e for e in events if e["split"] == "TRAIN"]
    oos = [e for e in events if e["split"] == "OOS"]
    ytr = np.array([e["is_mover"] for e in train], dtype=int)
    yoo = np.array([e["is_mover"] for e in oos], dtype=int)
    Ztr, Zoo = _zscore_per_asset(train, oos, FIZZLE_FEATS)
    res, p_oo = fit_score(Ztr, ytr, Zoo, yoo, FIZZLE_FEATS, seed)
    res["n_train"] = len(train)
    res["n_oos"] = len(oos)
    res["conversion_rate_train"] = float(ytr.mean())   # the ~31% base rate, on TRAIN
    res["conversion_rate_oos"] = float(yoo.mean())

    if res.get("degenerate") or p_oo is None:
        return res

    yoo_mover = yoo.astype(bool)
    ride_oo = np.array([e["ride_gross"] for e in oos])
    net_all = ride_oo - RIDE_COST_RT       # lane-#4 all-trigger ride net @24bps, OOS

    # ---- THE CONVERSION HIT-RATE in the top quantiles of the filter (OOS) ----
    order = np.argsort(-p_oo)
    quant = {}
    for q in TOP_QUANTILES:
        k = max(1, int(round(q * len(oos))))
        sel = order[:k]
        conv = float(yoo_mover[sel].mean())          # mover-rate among filtered triggers
        ride_net_sel = float(net_all[sel].mean())     # lane-#4 ride net on the FILTERED cohort
        ride_gross_sel = float(ride_oo[sel].mean())
        quant[f"top_{int(q*100)}pct"] = {
            "k_selected": k,
            "conversion_hit_rate": conv,             # vs base ~0.31
            "lift_vs_base": conv / float(yoo.mean()) if yoo.mean() > 0 else None,
            "ride_net_24bps_mean": ride_net_sel,
            "ride_gross_mean": ride_gross_sel,
        }

    res["filter_quantiles_oos"] = quant
    res["lever"] = {
        "all_trigger_ride_net_24bps_mean": float(net_all.mean()),
        "all_trigger_ride_gross_mean": float(ride_oo.mean()),
        "all_trigger_conversion": float(yoo.mean()),
        "note": "lever PASSES iff a top-quantile FILTERED ride net beats the all-trigger "
                "ride net AND is net-positive OUT OF SAMPLE",
    }
    # decisive verdict flags (OOS, two-sided)
    beats_null = (res.get("hgb_oos") is not None and res.get("shuffled_oos_auc_max") is not None
                  and res["hgb_oos"] > res["shuffled_oos_auc_max"] + 0.005)
    top30 = quant.get("top_30pct", {})
    top10 = quant.get("top_10pct", {})
    res["verdict"] = {
        "auc_beats_shuffled_null": bool(beats_null),
        "top30_conversion": top30.get("conversion_hit_rate"),
        "top10_conversion": top10.get("conversion_hit_rate"),
        "top30_ride_net_positive_oos": bool((top30.get("ride_net_24bps_mean") or -1) > 0),
        "top10_ride_net_positive_oos": bool((top10.get("ride_net_24bps_mean") or -1) > 0),
        "top30_beats_all_trigger_oos": bool((top30.get("ride_net_24bps_mean") or -1)
                                            > float(net_all.mean())),
        "top10_beats_all_trigger_oos": bool((top10.get("ride_net_24bps_mean") or -1)
                                            > float(net_all.mean())),
    }
    res["verdict"]["LEVER_PASSES_OOS"] = bool(
        beats_null and (
            (res["verdict"]["top30_ride_net_positive_oos"] and res["verdict"]["top30_beats_all_trigger_oos"])
            or (res["verdict"]["top10_ride_net_positive_oos"] and res["verdict"]["top10_beats_all_trigger_oos"])
        ))
    return res


# ===================================================================== driver

def main() -> int:
    ap = argparse.ArgumentParser(description="Mover TIMING: burst-onset + fizzle-filter (held-out)")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-burst", action="store_true", help="run only Part 2 (the lever)")
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
        tag = args.tag or "_".join(s[:-4] for s in syms[:4]).lower()
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        tag = args.tag or args.universe
    else:
        ap.error("provide --assets or --universe")

    t0 = time.time()

    # ---- gather Part-2 trigger events (always) and Part-1 burst events (unless skipped) ----
    all_trig, all_burst = [], []
    for sym in syms:
        try:
            tev = trigger_events(sym)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        n_tr = sum(1 for e in tev if e["split"] == "TRAIN")
        n_oo = sum(1 for e in tev if e["split"] == "OOS")
        all_trig.extend(tev)
        bmsg = ""
        if not args.skip_burst:
            bev = burst_events(sym)
            all_burst.extend(bev)
            bn_tr = sum(1 for e in bev if e["split"] == "TRAIN")
            bn_oo = sum(1 for e in bev if e["split"] == "OOS")
            bmsg = f" | burst-samples T/O={bn_tr}/{bn_oo}"
        print(f"[{sym}] triggers T/O={n_tr}/{n_oo}{bmsg} ({time.time()-t0:.0f}s)")

    n_assets = len({e["sym"] for e in all_trig}) or 1

    burst_res = None
    if not args.skip_burst and all_burst:
        # per-asset z of the magnitude features needs the liq_tot30_z handled by z-scoring;
        # the per-asset z in _zscore_per_asset already normalizes each feature per asset.
        print("\n[Part 1] fitting burst-onset detector ...")
        burst_res = analyze_burst(all_burst, args.seed)

    print("\n[Part 2] fitting fizzle-filter (the lever) ...")
    fizzle_res = analyze_fizzle(all_trig, args.seed)

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "mover_burst_timing", "git_sha": sha, "seed": args.seed,
        "params": {
            "burst_x": BURST_X, "burst_k": BURST_K, "burst_step": BURST_STEP,
            "t_trig": T_TRIG, "mover_runup": MOVER_RUNUP, "trail_k": TRAIL_K,
            "ride_cost_rt": RIDE_COST_RT, "top_quantiles": TOP_QUANTILES,
            "burst_feats": BURST_FEATS, "fizzle_feats": FIZZLE_FEATS,
        },
        "n_assets": n_assets,
        "part1_burst_onset": burst_res,
        "part2_fizzle_filter": fizzle_res,
        "caveats": [
            "u10 = CURRENT membership (survivorship on absolute levels); per-asset z'd",
            "shuffled-null is the false-positive floor; >0.005 over its max = real signal",
            "the fizzle label is MAGNITUDE (day-runup>=5%), NOT direction -- the filter "
            "does not claim to predict the sign of the move",
            "ride re-run uses lane-#4 trail-to-close mechanics; 24bps RT is the gate; real "
            "p_fill 0.21-0.40 (D43) would degrade live further",
            "UNSEEN: never entered features/labels/reports",
        ],
    }
    out_path = OUT / f"mover_burst_timing_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ---------------------------------------------------------------- STORY
    print("\n" + "=" * 82)
    print("MOVER TIMING -- (1) BURST-ONSET DETECTION + (2) FIZZLE-FILTER (the lever) -- STORY")
    print("=" * 82)

    if burst_res and not burst_res.get("degenerate"):
        b = burst_res
        print(f"\n[PART 1] BURST-ONSET (>= {BURST_X*100:.0f}% move starting within {BURST_K}m)")
        print(f"  samples TRAIN/OOS = {b['n_train']}/{b['n_oos']}; "
              f"burst base-rate OOS = {b['base_rate_oos']*100:.1f}%")
        print(f"  HGB  AUC train {b['hgb_train']:.3f}  OOS {b['hgb_oos']:.3f}  "
              f"(logit OOS {b['logit_oos']:.3f}; shuffled-null max {b.get('shuffled_oos_auc_max',0):.3f})")
        lt = b.get("lead_time", {})
        if lt:
            print(f"  ACTIONABILITY: top-30% score band precision {(lt.get('precision_top30') or 0)*100:.1f}% "
                  f"(base {lt['base_rate_oos']*100:.1f}%); median lead {lt.get('median_lead_min_top30_truebursts')}m "
                  f"of the {BURST_K}m window -> "
                  f"{'most burst AHEAD' if (lt.get('median_lead_min_top30_truebursts') or 0) <= BURST_K*0.4 else 'much burst GONE'}")
        print("  top burst features (perm-imp OOS): " +
              ", ".join(f"{k}{v:+.3f}" for k, v in list(b.get('perm_importance_oos', {}).items())[:5]))
    elif not args.skip_burst:
        print("\n[PART 1] burst-onset: DEGENERATE or no data")

    f = fizzle_res
    print(f"\n[PART 2] FIZZLE-FILTER (does this +1.5% trigger become a >= {MOVER_RUNUP*100:.0f}% mover?)")
    if f.get("degenerate"):
        print("  DEGENERATE label")
    else:
        print(f"  triggers TRAIN/OOS = {f['n_train']}/{f['n_oos']}; "
              f"conversion base-rate OOS = {f['conversion_rate_oos']*100:.1f}% (the ~31% to beat)")
        print(f"  HGB  AUC train {f['hgb_train']:.3f}  OOS {f['hgb_oos']:.3f}  "
              f"(logit OOS {f['logit_oos']:.3f}; shuffled-null max {f.get('shuffled_oos_auc_max',0):.3f})")
        lev = f.get("lever", {})
        print(f"  ALL-trigger ride (lane #4): gross {lev.get('all_trigger_ride_gross_mean',0)*100:+.3f}%  "
              f"NET@24bps {lev.get('all_trigger_ride_net_24bps_mean',0)*100:+.3f}% (OOS)")
        for q in TOP_QUANTILES:
            qk = f"top_{int(q*100)}pct"
            d = f.get("filter_quantiles_oos", {}).get(qk, {})
            if d:
                print(f"  FILTER {qk:<10} (n={d['k_selected']}): conversion {d['conversion_hit_rate']*100:.1f}% "
                      f"(lift {d.get('lift_vs_base') or 0:.2f}x)  ride NET@24bps {d['ride_net_24bps_mean']*100:+.3f}% "
                      f"(gross {d['ride_gross_mean']*100:+.3f}%)")
        v = f.get("verdict", {})
        print(f"\n  VERDICT (OOS, two-sided):")
        print(f"    AUC beats shuffled-null? {v.get('auc_beats_shuffled_null')}")
        print(f"    top-30% conversion {v.get('top30_conversion') and v['top30_conversion']*100:.1f}% "
              f"| net-positive? {v.get('top30_ride_net_positive_oos')} "
              f"| beats all-trigger? {v.get('top30_beats_all_trigger_oos')}")
        print(f"    top-10% conversion {v.get('top10_conversion') and v['top10_conversion']*100:.1f}% "
              f"| net-positive? {v.get('top10_ride_net_positive_oos')} "
              f"| beats all-trigger? {v.get('top10_beats_all_trigger_oos')}")
        print(f"    ==> LEVER PASSES OOS? {v.get('LEVER_PASSES_OOS')}")
        print("  top fizzle features (perm-imp OOS): " +
              ", ".join(f"{k}{vv:+.3f}" for k, vv in list(f.get('perm_importance_oos', {}).items())[:6]))

    print(f"\n({time.time()-t0:.0f}s)  JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
