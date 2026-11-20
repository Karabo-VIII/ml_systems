"""MOVER-CONTINUATION DISCRIMINATION -- decomposed mover problem #3 (CONTINUATION).

QUESTION (the deployable spec): once an asset is moving (onset at +1.5% intraday from
day-open), can we DISCRIMINATE continuers from reversers with a HELD-OUT signal whose
OOS AUC > 0.58, predicting continuation-given-onset?

GROUNDING (trusted, not re-derived):
  D72 (mover_metalabel) got internal-data continuation AUC = 0.52 (HGB OOS 0.521 /
  logit OOS 0.510) with 16 pooled causal features, and concluded the constraint is
  INFORMATION, not method -- needing AUC > 0.58. Its LABEL, though, was "the trail3
  trade pays net@24bps" -- a trade-economics label dominated by mover-day membership,
  NOT pure directional continuation. A just-completed characterization found mover
  STATUS is sticky (vol clusters, 1.45x) but DIRECTION is not (1.08x); post-move is
  REVERSAL-in-median / CONTINUATION-in-fat-tail.

WHY THIS IS NOT JUST RE-RUNNING D72 -- three fresh angles, each a different LABEL:
  (A) DIRECTIONAL continuation: from the onset fill, does price advance FURTHER
      (forward return over H minutes > 0, and a stricter "reaches +X% beyond onset").
      The D72 question, but on the pure forward-return label (mover-membership removed).
  (B) MAGNITUDE continuation: does the move KEEP BEING BIG regardless of sign
      (|forward move| over H in the top half)? Volatility is persistent (1.45x), so
      this should be MORE predictable than direction. If it clears 0.58 it is a
      both-sided (straddle / Lane-4) opportunity, not a directional bet.
  (C) THE FAT TAIL: predict the TOP-DECILE forward continuers (the big runners that
      carry the positive mean). Reframed as precision@top-k, not AUC-over-all.

THE MECHANISM FEATURES (the substantive new content vs D72's pooled bag) -- isolate the
POSITIONING signature of the onset, all strictly causal at/<= the onset close:
  OI-BUILDING (continuation fuel): oi_d1h, oi_d4h, oi_d24h (new positioning entering).
  LIQUIDATION-WICK (forced/exhaustion = reversal): long-liq usd, short-liq usd, the
      LONG-MINUS-SHORT liq asymmetry, liq_ratio -- D72 used only liq_ratio (long).
  FUNDING direction (crowded-long => squeeze risk): funding sign + level.
  AGGRESSOR FLOW: day-so-far buy/sell imbalance AND its ACCELERATION (last-30m
      imbalance minus first-half imbalance -- is the buying intensifying or fading?).
  ONSET SHAPE: run slope acceleration (ret over last 30m to onset vs the average
      per-minute pace from day-open) -- accelerating (impulsive) vs grinding/fading.
  Plus the D72 controls (overshoot, t2trig, pre_vol, dayvol_ratio, vol_surge, regime,
      BTC rel/r24h, hour).
  NOT AVAILABLE causally at 1m without a fragile chimera as-of join: whale-net-usd and
  spot-perp basis (chimera bs_basis_*/wh_whale_net_usd are range/dollar-bar clocked).
  Their ABSENCE is itself part of the verdict: if internal 1m flow can't crack 0.58,
  the residual is precisely the external-data (whale/basis/orderbook) question.

HONEST HELD-OUT (mandatory):
  TRAIN < 2024-01-01 fits everything (model, z-stats, k-thresholds). OOS
  2024-01-01..2025-07-01 scored ONCE. UNSEEN >= 2025-07-01 NEVER touched (no row enters
  features/labels/reports). Every feature uses data <= onset close (next-minute fill for
  forward labels). Two-sided: SHUFFLED-LABEL null per angle -- a positive that doesn't
  beat its shuffled null, or is TRAIN-only, is a NULL and is reported as such.
  HGB uses early_stopping (validation_fraction) so we do NOT repeat D72's 0.9998 TRAIN
  memorization; logit is the linear sanity arm.

Run:
  python -m mining.mover_continuation --universe u10
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
    "outputs": {"study_json": "runs/mining/mover_continuation_<tag>_<stamp>.json"},
    "invariants": {
        "train_only_fit": "model, z-stats, k-thresholds ALL from TRAIN only",
        "causal_features": "every feature computed from data <= onset close",
        "forward_label_next_open": "forward returns measured from next-minute open (no same-bar)",
        "unseen_untouched": "no UNSEEN row enters features, labels, or reports",
        "shuffled_null": "each angle reports a shuffled-label AUC null as the false-positive floor",
    },
}

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY_MS = 86_400_000
T_TRIG = 0.015            # onset: +1.5% from UTC day open (D72 trigger, unchanged)
FWD_H = 240              # forward horizon for continuation labels (minutes)
MIN_MINUTES_LEFT = FWD_H + 5   # onset must leave a full forward horizon in the day
CONT_FURTHER = 0.015     # "reaches +1.5% beyond onset" stricter directional label
DECILE_TOPK = 0.10       # tail = top 10% of forward continuation

# mechanism + control features (all causal at/<= onset close)
FEATURES = [
    # --- OI-building (continuation fuel) ---
    "oi_d1h", "oi_d4h", "oi_d24h",
    # --- liquidation-wick (forced/exhaustion -> reversal) ---
    "liq_ratio", "liq_long_z", "liq_short_z", "liq_ls_asym",
    # --- funding (crowded-long squeeze risk) ---
    "funding",
    # --- aggressor flow + acceleration ---
    "aggr_imb", "aggr_accel",
    # --- onset shape ---
    "run_accel", "overshoot", "t2trig",
    # --- controls ---
    "pre_vol", "dayvol_ratio", "vol_surge", "regime",
    "btc_rel", "btc_r24h", "hour_utc",
]
MECH_FEATURES = ["oi_d1h", "oi_d4h", "oi_d24h", "liq_ratio", "liq_long_z",
                 "liq_short_z", "liq_ls_asym", "funding", "aggr_imb", "aggr_accel",
                 "run_accel"]


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def btc_context() -> dict:
    df = load_panel("BTCUSDT")
    ms = df["minute_ts"].to_numpy()
    closes = df["close"].to_numpy()
    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    d_open = np.repeat(closes[day_starts], np.diff(np.append(day_starts, len(ms))))
    rel = closes / d_open - 1.0
    r24h = np.full(len(closes), np.nan)
    r24h[1440:] = closes[1440:] / closes[:-1440] - 1.0
    return {"ms0": int(ms[0]), "rel": rel, "r24h": r24h}


def extract_events(sym: str, btc_ctx: dict | None) -> list[dict]:
    """One onset event per asset-day (first close crossing +T from day open). Mechanism
    features at the onset close; forward continuation labels from the next-minute open."""
    df = load_panel(sym)
    ms = df["minute_ts"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    vol = df["vol_usd"].to_numpy()
    buy = df["buy_aggr_usd"].fill_null(0.0).to_numpy()
    sell = df["sell_aggr_usd"].fill_null(0.0).to_numpy()
    liq_long = df["liq_long_usd"].to_numpy()
    liq_short = df["liq_short_usd"].to_numpy()
    pre_vol = df["pre_vol"].to_numpy()
    ret1m = df["ret_1m"].to_numpy()
    oi1 = df["oi_d1h"].to_numpy()
    oi4 = df["oi_d4h"].to_numpy()
    oi24 = df["oi_d24h"].to_numpy()
    fund = df["funding"].to_numpy()
    liqr = df["liq_ratio"].to_numpy()
    regime = df["regime_above_sma200"].to_numpy()
    real = df["is_real"].to_numpy().astype(bool)
    n = len(df)

    # 30m rolling liq sums for z-scoring the wick magnitude at onset (causal)
    def roll_sum(a, w):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full(len(a), np.nan)
        out[w - 1:] = c[w:] - c[:-w]
        return out
    ll30 = roll_sum(liq_long, 30)
    ls30 = roll_sum(liq_short, 30)

    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    day_ends = np.append(day_starts[1:] - 1, n - 1)

    day_tot = np.array([vol[ds:de + 1].sum() for ds, de in zip(day_starts, day_ends)])
    day_close_arr = np.array([closes[de] for de in day_ends])
    avg30 = np.full(len(day_tot), np.nan)
    for i in range(len(day_tot)):
        lo = max(0, i - 30)
        if i - lo >= 10:
            avg30[i] = day_tot[lo:i].mean()

    events = []
    for di, (ds, de) in enumerate(zip(day_starts, day_ends)):
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
        m = ds + int(hit[0])            # onset minute (close crosses +T)
        f = m + 1                       # fill minute (next-minute open)
        if de - f < MIN_MINUTES_LEFT or not (real[m] and real[f]):
            continue
        entry = opens[f]
        if not np.isfinite(entry) or entry <= 0:
            continue

        # ---- forward continuation outcomes (from entry, over FWD_H) ----
        fwd_end = f + FWD_H
        seg = closes[f:fwd_end + 1]
        if not np.all(np.isfinite(seg)) or real[f:fwd_end + 1].mean() < 0.90:
            continue
        fwd_ret = float(closes[fwd_end] / entry - 1.0)             # directional continuation
        fwd_path = closes[f:fwd_end + 1] / entry - 1.0
        fwd_max = float(np.max(fwd_path))                          # best continuation reached
        fwd_min = float(np.min(fwd_path))
        fwd_absmove = float(max(abs(fwd_max), abs(fwd_min)))       # magnitude continuation

        # ---- mechanism features (strictly causal at close of m) ----
        elapsed = (m - ds + 1) / 1440.0
        vsum = float(vol[ds:m + 1].sum())
        bsum = float(buy[ds:m + 1].sum())
        ssum = float(sell[ds:m + 1].sum())
        # aggressor acceleration: last-30m imbalance minus day-so-far imbalance
        b30 = float(buy[max(ds, m - 29):m + 1].sum())
        s30 = float(sell[max(ds, m - 29):m + 1].sum())
        imb_day = (bsum - ssum) / (bsum + ssum) if bsum + ssum > 0 else np.nan
        imb_30 = (b30 - s30) / (b30 + s30) if b30 + s30 > 0 else np.nan
        aggr_accel = (imb_30 - imb_day) if (np.isfinite(imb_30) and np.isfinite(imb_day)) else np.nan
        # run acceleration: last-30m return pace vs average per-minute pace day-so-far
        ret_30 = float(closes[m] / closes[max(ds, m - 30)] - 1.0)
        avg_pace = rel[m - ds] / max(m - ds, 1)        # mean per-minute since day open
        run_accel = (ret_30 / 30.0) / avg_pace if avg_pace != 0 and np.isfinite(avg_pace) else np.nan
        # liq-wick magnitudes at onset (z vs day-so-far liq distribution)
        ll = ll30[m] if np.isfinite(ll30[m]) else 0.0
        ls = ls30[m] if np.isfinite(ls30[m]) else 0.0
        # z relative to that asset's trailing day liq sums (computed downstream per-asset)
        dvol_sofar = float(np.nanstd(ret1m[ds:m + 1])) if m - ds >= 30 else np.nan

        btc_rel = btc_r24h = np.nan
        if btc_ctx is not None:
            bi = int((ms[m] - btc_ctx["ms0"]) // 60_000)
            if 0 <= bi < len(btc_ctx["rel"]):
                btc_rel = btc_ctx["rel"][bi]
                btc_r24h = btc_ctx["r24h"][bi]

        events.append({
            "sym": sym, "day_ms": int(ms[ds]), "t0_ms": int(ms[m]), "split": sp,
            "fwd_ret": fwd_ret, "fwd_max": fwd_max, "fwd_min": fwd_min,
            "fwd_absmove": fwd_absmove,
            "feats": {
                "oi_d1h": float(oi1[m]) if np.isfinite(oi1[m]) else np.nan,
                "oi_d4h": float(oi4[m]) if np.isfinite(oi4[m]) else np.nan,
                "oi_d24h": float(oi24[m]) if np.isfinite(oi24[m]) else np.nan,
                "liq_ratio": float(liqr[m]) if np.isfinite(liqr[m]) else np.nan,
                "liq_long_raw": float(ll), "liq_short_raw": float(ls),
                "funding": float(fund[m]) if fund[m] is not None and np.isfinite(fund[m]) else np.nan,
                "aggr_imb": float(imb_day) if np.isfinite(imb_day) else np.nan,
                "aggr_accel": float(aggr_accel) if np.isfinite(aggr_accel) else np.nan,
                "run_accel": float(run_accel) if np.isfinite(run_accel) else np.nan,
                "overshoot": float(rel[m - ds] - T_TRIG),
                "t2trig": float(m - ds),
                "pre_vol": float(pre_vol[m]) if np.isfinite(pre_vol[m]) else np.nan,
                "dayvol_ratio": (dvol_sofar / pre_vol[m]
                                 if np.isfinite(pre_vol[m]) and pre_vol[m] > 0
                                 and np.isfinite(dvol_sofar) else np.nan),
                "vol_surge": (vsum / (avg30[di] * elapsed)
                              if np.isfinite(avg30[di]) and avg30[di] > 0 else np.nan),
                "regime": (1.0 if regime[m] else 0.0) if regime[m] is not None else np.nan,
                "btc_rel": btc_rel, "btc_r24h": btc_r24h,
                "hour_utc": float((ms[m] % DAY_MS) / 3_600_000),
            },
        })
    return events


def _finalize_feats(events: list[dict]) -> None:
    """Derive per-asset liq z-scores + the long/short asymmetry from TRAIN-distribution
    stats. z-stats are fit on TRAIN rows only (no leakage), applied to all splits."""
    by_sym: dict[str, list[dict]] = {}
    for e in events:
        by_sym.setdefault(e["sym"], []).append(e)
    for sym, evs in by_sym.items():
        tr = [e for e in evs if e["split"] == "TRAIN"]
        ll_tr = np.array([e["feats"]["liq_long_raw"] for e in tr], dtype=float)
        ls_tr = np.array([e["feats"]["liq_short_raw"] for e in tr], dtype=float)
        ll_mu, ll_sd = np.nanmean(ll_tr), np.nanstd(ll_tr)
        ls_mu, ls_sd = np.nanmean(ls_tr), np.nanstd(ls_tr)
        for e in evs:
            ll = e["feats"].pop("liq_long_raw")
            ls = e["feats"].pop("liq_short_raw")
            e["feats"]["liq_long_z"] = float((ll - ll_mu) / ll_sd) if ll_sd > 0 else 0.0
            e["feats"]["liq_short_z"] = float((ls - ls_mu) / ls_sd) if ls_sd > 0 else 0.0
            # long-minus-short asymmetry, normalized (forced-long wick > forced-short => exhaustion top)
            denom = ll + ls
            e["feats"]["liq_ls_asym"] = float((ll - ls) / denom) if denom > 0 else 0.0


def _mat(evs, feats):
    X = np.array([[e["feats"][f] for f in feats] for e in evs], dtype=float)
    return X


def _zscore_per_asset(evs_tr, evs_oo, feats):
    """Per-(asset,feature) z from TRAIN stats only; applied to TRAIN+OOS."""
    Xtr = _mat(evs_tr, feats)
    Xoo = _mat(evs_oo, feats)
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


def fit_score(Ztr, ytr, Zoo, yoo, seed, feats):
    """HGB (early-stopped, no D72 memorization) + logit sanity. Returns AUCs + the HGB
    OOS scores for tail analysis + a permutation-importance proxy."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    out = {}
    if len(np.unique(ytr)) < 2 or len(np.unique(yoo)) < 2:
        return {"degenerate": True}, None
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=20, random_state=seed)
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

    # shuffled-label null (TRAIN labels permuted; refit HGB; OOS AUC against TRUE yoo)
    rng = np.random.default_rng(seed + 99)
    ysh = ytr.copy()
    rng.shuffle(ysh)
    nulls = []
    for k in range(5):
        rng.shuffle(ysh)
        hsh = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2, n_iter_no_change=20,
            random_state=seed + k)
        if len(np.unique(ysh)) < 2:
            continue
        hsh.fit(Ztr, ysh)
        nulls.append(float(roc_auc_score(yoo, hsh.predict_proba(Zoo)[:, 1])))
    out["shuffled_oos_auc_mean"] = float(np.mean(nulls)) if nulls else None
    out["shuffled_oos_auc_max"] = float(np.max(nulls)) if nulls else None

    # permutation importance on OOS (HGB), mechanism-feature attribution
    base = out["hgb_oos"]
    imp = {}
    rng2 = np.random.default_rng(seed + 7)
    for j, fn in enumerate(feats):
        Zp = Zoo.copy()
        col = Zp[:, j].copy()
        rng2.shuffle(col)
        Zp[:, j] = col
        imp[fn] = float(base - roc_auc_score(yoo, hgb.predict_proba(Zp)[:, 1]))
    out["perm_importance_oos"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    return out, p_oo


def precision_at_k(p_oo, fwd_vals, evs_oo, train_thr, frac=DECILE_TOPK):
    """Tail angle: among the top-`frac` OOS events by model score, what fraction are
    genuine top-decile continuers (fwd >= TRAIN top-decile threshold)? Compare to base
    rate (= frac by construction of the label) and to random selection."""
    n = len(p_oo)
    k = max(1, int(round(frac * n)))
    order = np.argsort(-np.asarray(p_oo))
    top_idx = order[:k]
    is_tail = np.asarray(fwd_vals) >= train_thr   # label fit on TRAIN threshold
    base_rate = float(is_tail.mean())
    prec_topk = float(is_tail[top_idx].mean())
    # lift over base
    lift = prec_topk / base_rate if base_rate > 0 else None
    # mean realized forward continuation of the selected top-k (economic read)
    mean_fwd_topk = float(np.mean(np.asarray(fwd_vals)[top_idx]))
    mean_fwd_all = float(np.mean(fwd_vals))
    return {"k": k, "n": n, "base_rate_tail": base_rate, "precision_at_topk": prec_topk,
            "lift": lift, "mean_fwd_topk": mean_fwd_topk, "mean_fwd_all": mean_fwd_all}


def run_angle(name, label_fn, train, oos, feats, seed, tail=False, fwd_key=None):
    ytr = np.array([label_fn(e) for e in train], dtype=int)
    yoo = np.array([label_fn(e) for e in oos], dtype=int)
    Ztr, Zoo = _zscore_per_asset(train, oos, feats)
    res, p_oo = fit_score(Ztr, ytr, Zoo, yoo, seed, feats)
    res["label"] = name
    res["train_pos_rate"] = float(ytr.mean())
    res["oos_pos_rate"] = float(yoo.mean())
    res["n_train"] = int(len(ytr))
    res["n_oos"] = int(len(yoo))
    if not res.get("degenerate") and tail and fwd_key and p_oo is not None:
        fwd_tr = np.array([e[fwd_key] for e in train])
        fwd_oo = np.array([e[fwd_key] for e in oos])
        thr = float(np.quantile(fwd_tr, 1 - DECILE_TOPK))   # TRAIN top-decile threshold
        res["tail"] = precision_at_k(p_oo, fwd_oo, oos, thr)
        res["tail"]["train_topdecile_thr"] = thr
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Mover continuation-discrimination (held-out)")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None)
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
    print("loading BTC context...")
    bctx = btc_context()
    all_events = []
    for sym in syms:
        try:
            evs = extract_events(sym, None if sym == "BTCUSDT" else bctx)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        n_tr = sum(1 for e in evs if e["split"] == "TRAIN")
        print(f"[{sym}] onset events TRAIN/OOS = {n_tr}/{len(evs)-n_tr}")
        all_events.extend(evs)
    _finalize_feats(all_events)
    n_assets = len({e["sym"] for e in all_events}) or 1

    train = [e for e in all_events if e["split"] == "TRAIN"]
    oos = [e for e in all_events if e["split"] == "OOS"]
    print(f"\ntotal onset events: TRAIN {len(train)} / OOS {len(oos)} over {n_assets} assets")

    # forward-return descriptive read (is the median a reverser or continuer?)
    fwd_tr = np.array([e["fwd_ret"] for e in train])
    fwd_oo = np.array([e["fwd_ret"] for e in oos])
    desc = {
        "fwd_ret_median_train": float(np.median(fwd_tr)),
        "fwd_ret_median_oos": float(np.median(fwd_oo)),
        "fwd_ret_mean_train": float(np.mean(fwd_tr)),
        "fwd_ret_mean_oos": float(np.mean(fwd_oo)),
        "p_fwd_up_train": float((fwd_tr > 0).mean()),
        "p_fwd_up_oos": float((fwd_oo > 0).mean()),
        "fwd_h_min": FWD_H,
    }

    # ---- ANGLE A: directional continuation ----
    A1 = run_angle("A_dir_fwd_up", lambda e: int(e["fwd_ret"] > 0), train, oos, FEATURES, args.seed)
    A2 = run_angle("A_dir_reaches_+1.5pct", lambda e: int(e["fwd_max"] >= CONT_FURTHER),
                   train, oos, FEATURES, args.seed)
    # mechanism-only variant of A1 (does the positioning signature alone carry it?)
    A1m = run_angle("A_dir_fwd_up_MECH_ONLY", lambda e: int(e["fwd_ret"] > 0),
                    train, oos, MECH_FEATURES, args.seed)

    # ---- ANGLE B: magnitude continuation (move keeps being big, sign-agnostic) ----
    # label fit on TRAIN median of |forward move| (no OOS leakage)
    absmove_tr = np.array([e["fwd_absmove"] for e in train])
    abs_med = float(np.median(absmove_tr))
    B1 = run_angle("B_mag_absmove_gt_trainmed",
                   lambda e: int(e["fwd_absmove"] > abs_med), train, oos, FEATURES, args.seed)
    B1m = run_angle("B_mag_absmove_MECH_ONLY",
                    lambda e: int(e["fwd_absmove"] > abs_med), train, oos, MECH_FEATURES, args.seed)

    # ---- ANGLE C: the fat tail (top-decile directional continuers) ----
    C1 = run_angle("C_tail_topdecile_fwd_ret",
                   lambda e: int(e["fwd_ret"] >= np.quantile(fwd_tr, 1 - DECILE_TOPK)),
                   train, oos, FEATURES, args.seed, tail=True, fwd_key="fwd_ret")

    angles = {"A1_dir_up": A1, "A2_dir_reaches": A2, "A1m_dir_up_mech": A1m,
              "B1_mag": B1, "B1m_mag_mech": B1m, "C1_tail": C1}

    # verdict logic
    def auc_of(a):
        return a.get("hgb_oos") if not a.get("degenerate") else None

    def beats_null(a):
        h = a.get("hgb_oos"); s = a.get("shuffled_oos_auc_max")
        return (h is not None and s is not None and h > s + 0.005)

    best = max(((k, auc_of(v)) for k, v in angles.items() if auc_of(v) is not None),
               key=lambda kv: kv[1], default=(None, None))
    any_clears = {k: bool((auc_of(v) or 0) > 0.58 and beats_null(v)) for k, v in angles.items()}
    verdict = {
        "best_angle": best[0], "best_oos_auc": best[1],
        "any_clears_0_58_held_out": any(any_clears.values()),
        "clears_per_angle": any_clears,
        "tail_lift_oos": C1.get("tail", {}).get("lift"),
        "d72_baseline_auc": 0.521,
        "spec_threshold": 0.58,
    }

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "mover_continuation", "git_sha": sha, "seed": args.seed,
        "params": {"T": T_TRIG, "fwd_h_min": FWD_H, "cont_further": CONT_FURTHER,
                   "decile_topk": DECILE_TOPK, "features": FEATURES,
                   "mech_features": MECH_FEATURES},
        "n_assets": n_assets, "n_train": len(train), "n_oos": len(oos),
        "descriptive": desc, "angles": angles, "verdict": verdict,
        "caveats": [
            "u10 current membership (survivorship on absolute levels)",
            "whale-net-usd / spot-perp basis NOT available causally at 1m (chimera is "
            "range/dollar-bar clocked) -- their absence bounds the internal-data test",
            "shuffled-null is the false-positive floor; >0.005 over its max = real signal",
        ],
    }
    out_path = OUT / f"mover_continuation_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ---------------------------------------------------------------- STORY
    print("\n" + "=" * 80)
    print(f"MOVER CONTINUATION DISCRIMINATION (onset +1.5pct, fwd {FWD_H}m) -- STORY")
    print("=" * 80)
    print(f"onset events: TRAIN {len(train)} / OOS {len(oos)} over {n_assets} assets")
    print(f"DESCRIPTIVE: median fwd-ret OOS {desc['fwd_ret_median_oos']*100:+.3f}%  "
          f"mean {desc['fwd_ret_mean_oos']*100:+.3f}%  P(up) {desc['p_fwd_up_oos']*100:.1f}%  "
          f"=> {'REVERSAL-in-median' if desc['fwd_ret_median_oos']<0 else 'continuation-in-median'}")
    print("-" * 80)
    print(f"{'ANGLE':<28}{'AUC tr':>8}{'AUC OOS':>9}{'shuf max':>9}{'beats?':>8}{'clears.58':>10}")
    for k, v in angles.items():
        if v.get("degenerate"):
            print(f"{k:<28}  DEGENERATE (single-class label)")
            continue
        bn = "yes" if beats_null(v) else "no"
        cl = "YES" if any_clears[k] else "-"
        print(f"{k:<28}{v['hgb_train']:>8.3f}{v['hgb_oos']:>9.3f}"
              f"{(v.get('shuffled_oos_auc_max') or 0):>9.3f}{bn:>8}{cl:>10}")
    print("-" * 80)
    t = C1.get("tail")
    if t:
        print(f"TAIL (top-{int(DECILE_TOPK*100)}% by model score): precision {t['precision_at_topk']*100:.1f}%  "
              f"base {t['base_rate_tail']*100:.1f}%  lift {t['lift']:.2f}x  "
              f"mean-fwd selected {t['mean_fwd_topk']*100:+.2f}% vs all {t['mean_fwd_all']*100:+.2f}%")
    print("-" * 80)
    print("TOP MECHANISM FEATURES (perm-importance, directional angle A1):")
    pi = A1.get("perm_importance_oos", {})
    for fn, iv in list(pi.items())[:8]:
        tag_m = " [MECH]" if fn in MECH_FEATURES else ""
        print(f"  {fn:<16}{iv:+.4f}{tag_m}")
    print("-" * 80)
    print(f"VERDICT: best={verdict['best_angle']} OOS AUC={verdict['best_oos_auc']:.3f} | "
          f"any clears 0.58 held-out? {verdict['any_clears_0_58_held_out']} | "
          f"D72 baseline 0.521")
    print(f"({time.time()-t0:.0f}s)  JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
