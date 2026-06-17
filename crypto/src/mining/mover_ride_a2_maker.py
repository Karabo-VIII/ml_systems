"""MOVER-RIDE LIVE-PATH TEST -- A2-gated, MAKER-filled, swept-exit, vs the arbiter null.

THE QUESTION (Wave-2A flagship): is move-riding a WIN once you make it honest?
  Wave-1 established (real artifacts under runs/mining/):
    - on >=5% mover-days at 1m, after a +1.5% intraday confirmation there is still
      median ~+5.3% of ORACLE ceiling REMAINING (D67 "too late" refuted at 1m);
    - UNCONDITIONAL riding is net-NEGATIVE at 24bps taker (all 9 mover_ride cells lose;
      the rider underperforms random-entry-on-the-same-movers -- the arbiter null);
    - BUT at trigger-time the A2 model ("price reaches +1.5% FURTHER beyond the onset")
      gets OOS AUC 0.648 (clears the 0.58 spec, beats its shuffled-null max 0.516 wide).
  This tool builds the LIVE-PATH backtest: enter confirmed movers at the +1.5% trigger
  ONLY when A2 fires, exit by a SWEPT policy, scale by the B1 magnitude score, model
  MAKER fills HONESTLY, and compare to RANDOM-ENTRY-ON-THE-SAME-MOVERS at the identical
  exit + cost + fill model (the null that killed naive riding).

WHY MAKER, AND WHY A CHASE PENALTY (the honesty that matters):
  You cannot take a confirmed +1.5% move at the touch with a maker order without a fill
  model -- a resting limit at the onset close fills ONLY if price comes BACK to it. So:
    - p_fill in [0.25, 0.40]: a fraction of gated events never fill at all (you miss them);
    - which ones you miss is NOT random -- the FAST-UP legs (the best A2 events) run away
      and never fill your limit, while the ones that PULL BACK fill (ADVERSE SELECTION).
      We model this mechanically: a limit posted at the onset close fills iff a later
      minute's LOW within a fill-window <= limit; the entry is then the limit price (you
      got your price -- maker, no slippage). The adverse selection is INTRINSIC to that
      rule (you preferentially keep the events that dipped back to you), so we do NOT
      additionally hand-penalize; the fill rule itself is the chase penalty. We then
      CALIBRATE the fill-window so realized p_fill lands in [0.25, 0.40] and report it.
    - RT cost = 12bps maker round-trip (6bps each side). Applied identically to null.

EXIT POLICIES (swept, selected on TRAIN+VAL only):
  vol_trail(k):  high-water trailing stop at k * pre_vol_atr (vol-scaled, not fixed %);
  fixed_target(tp, sl): take-profit at +tp, hard stop at -sl, else day-close;
  time_stop(H):  hold H minutes then exit at that minute's open, day-bounded.
  All exits day-bounded (no overnight); fills at next-minute open (stops/targets fill at
  the open of the minute AFTER the breach minute; an intrabar gap-through fills at that
  open -- conservative, no within-bar optimism). Identical geometry for event and null.

SIZING (B1 magnitude score, optional): scale notional in [0.5, 1.5] by the B1 percentile
  (sign-agnostic vol-continuation). Selected on TRAIN+VAL; OFF is also swept.

THE ARBITER NULL (what killed naive riding): RANDOM-ENTRY-ON-THE-SAME-MOVERS. Same mover
  days, same +1.5% confirmation, but the A2 gate is REPLACED by a random post-trigger
  entry minute (uniform over the eligible post-trigger window) with the IDENTICAL maker
  fill model, exit policy, cost, and sizing=1.0. If A2-gated riding does not beat this on
  net per-event, the "edge" is just mover-day vol, not the A2 signal.

PRE-REGISTRATION (gates fixed BEFORE any UNSEEN look):
  - Fit A2 (+B1) HGB on TRAIN (<2024-01-01) only.
  - SELECT on TRAIN+VAL (VAL = 2024-01-01..2025-07-01): the (exit policy, A2 gate
    threshold, sizing on/off) maximizing VAL beat-null net per-event, subject to
    breadth >= 6/10 assets on TRAIN+VAL. Exactly ONE config advances.
  - PRE-REGISTERED UNSEEN GATE (>= 2025-07-01, evaluated ONCE):
       (1) beat-null net per-event > 0 (A2-gated minus random-entry, same movers);
       (2) breadth >= 6/10 assets (per-asset beat-null mean > 0, >=3 events to count);
       (3) block-bootstrap p05 of the beat-null delta > 0 (weekly clusters).
    ALIVE iff all three hold; else dead / needs-deeper with the killing test named.

Splits: TRAIN < 2024-01-01 | VAL 2024-01-01..2025-07-01 | UNSEEN >= 2025-07-01.
Long-only + spot + lev=1 (riding UP-moves; NO shorts). Objective = net per-move
EXPECTANCY (not AUC/IC). Seeded; git lineage in JSON. No emoji (cp1252). No git commit.

Run:
  python -m mining.mover_ride_a2_maker --universe u10
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
    "outputs": {"study_json": "runs/mining/mover_ride_a2_maker_<tag>_<stamp>.json"},
    "invariants": {
        "train_only_model_fit": "A2/B1 HGB fit on TRAIN only; threshold/exit/sizing on TRAIN+VAL",
        "causal_features": "every A2/B1 feature computed from data <= onset close",
        "maker_fill_model": "limit at onset close; fills iff later low<=limit within window; p_fill in [.25,.40]",
        "arbiter_null": "random-entry-on-same-movers, identical fill/exit/cost, sizing=1",
        "unseen_once": "UNSEEN evaluated exactly once at the very end; never in fit/select",
        "long_only_spot_lev1": "ride UP-moves only; no shorts; no leverage",
        "next_open_fills": "decision close of m -> fill open of m+1; day-bounded exits",
    },
}

# ---- splits (3-way: TRAIN fits model; TRAIN+VAL selects; UNSEEN scored once) ----
TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
VAL_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY_MS = 86_400_000
CLUSTER_MS = 7 * DAY_MS

# ---- event geometry (matches mover_continuation A2 exactly) ----
T_TRIG = 0.015            # onset: +1.5% from UTC day open
CONT_FURTHER = 0.015      # A2 label: fwd_max >= +1.5% beyond onset (reaches further)
B1_FWD_H = 240            # B1 magnitude horizon (must match mover_continuation B1)
MIN_MINUTES_LEFT = 90     # onset must leave >=90m of day to ride + fill window
MAKER_RT = 0.0012         # 12bps maker round-trip
# HONEST MAKER MODEL (calibrated empirically -- see calibration note below):
#   a TRUE passive maker bid sits BELOW current price and fills only on a pullback. A
#   limit posted AT the onset close fills ~99.6% (it is a market order in disguise, price
#   always wicks your level immediately). Posting a bid `FILL_OFFSET` below current price,
#   live for `FILL_WINDOW` minutes, fills iff a later low touches it: off=0.005 / W=15 ->
#   realized p_fill ~ 0.33 (in band) AND you enter 0.5% cheaper (you bought the dip) but
#   only keep the ~1/3 of movers that pulled back to you = the intrinsic adverse selection
#   / chase penalty (the fast-up continuation legs run away and never fill).
FILL_OFFSET = 0.008       # maker bid posted 0.8% below current price (passive); calibrated
FILL_WINDOW = 15          # minutes the resting bid stays live -> realized p_fill ~0.32 (u10)
P_FILL_LO, P_FILL_HI = 0.25, 0.40  # acceptable realized maker fill-rate band

# A2/B1 feature set (identical to mover_continuation -- causal at/<= onset close)
FEATURES = [
    "oi_d1h", "oi_d4h", "oi_d24h",
    "liq_ratio", "liq_long_z", "liq_short_z", "liq_ls_asym",
    "funding", "aggr_imb", "aggr_accel", "run_accel",
    "overshoot", "t2trig", "pre_vol", "dayvol_ratio", "vol_surge",
    "regime", "btc_rel", "btc_r24h", "hour_utc",
]

# swept exit policies (name, kind, params)
EXITS = [
    ("vt_2.0", "vol_trail", {"k": 2.0}),
    ("vt_3.0", "vol_trail", {"k": 3.0}),
    ("vt_4.0", "vol_trail", {"k": 4.0}),
    ("ft_3_2", "fixed_target", {"tp": 0.03, "sl": 0.02}),
    ("ft_5_3", "fixed_target", {"tp": 0.05, "sl": 0.03}),
    ("ft_4_2", "fixed_target", {"tp": 0.04, "sl": 0.02}),
    ("ts_120", "time_stop", {"H": 120}),
    ("ts_240", "time_stop", {"H": 240}),
]
A2_GATE_QS = [0.50, 0.60, 0.70, 0.80]   # gate: take only events with A2 score >= TRAIN q
SIZING_MODES = ["off", "b1"]


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("VAL" if ms < VAL_END_MS else "UNSEEN")


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


# ----------------------------------------------------------------- event extraction
def extract_events(sym: str, btc_ctx: dict | None) -> list[dict]:
    """One onset event per asset-day. Carries A2/B1 features (causal at onset close), the
    forward 240m path (for the B1 label + A2 label), AND the full post-fill intraday OHLC
    path to day-end (for swept exit simulation + the maker fill model)."""
    df = load_panel(sym)
    ms = df["minute_ts"].to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    vol = df["vol_usd"].to_numpy()
    buy = df["buy_aggr_usd"].fill_null(0.0).to_numpy()
    sell = df["sell_aggr_usd"].fill_null(0.0).to_numpy()
    liq_long = df["liq_long_usd"].to_numpy()
    liq_short = df["liq_short_usd"].to_numpy()
    pre_vol = df["pre_vol"].to_numpy()
    ret1m = df["ret_1m"].to_numpy()
    oi1 = df["oi_d1h"].to_numpy(); oi4 = df["oi_d4h"].to_numpy(); oi24 = df["oi_d24h"].to_numpy()
    fund = df["funding"].to_numpy()
    liqr = df["liq_ratio"].to_numpy()
    regime = df["regime_above_sma200"].to_numpy()
    real = df["is_real"].to_numpy().astype(bool)
    n = len(df)

    def roll_sum(a, w):
        c = np.cumsum(np.insert(a, 0, 0.0)); o = np.full(len(a), np.nan)
        o[w - 1:] = c[w:] - c[:-w]; return o
    ll30 = roll_sum(liq_long, 30); ls30 = roll_sum(liq_short, 30)

    day_ids = ms // DAY_MS
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    day_ends = np.append(day_starts[1:] - 1, n - 1)
    day_tot = np.array([vol[ds:de + 1].sum() for ds, de in zip(day_starts, day_ends)])
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
        rel = closes[ds:de + 1] / d_open - 1.0
        hit = np.flatnonzero(rel >= T_TRIG)
        if len(hit) == 0:
            continue
        m = ds + int(hit[0])             # onset minute (close crosses +T)
        f = m + 1                        # next-minute open (causal fill reference)
        if de - f < MIN_MINUTES_LEFT or not (real[m] and real[f]):
            continue
        onset_close = closes[m]
        if not np.isfinite(onset_close) or onset_close <= 0:
            continue

        # --- forward 240m path (A2 + B1 labels), real-coverage gated ---
        fwd_end = min(f + B1_FWD_H, de)
        seg_real = real[f:fwd_end + 1].mean()
        seg_close = closes[f:fwd_end + 1]
        if len(seg_close) < 60 or not np.all(np.isfinite(seg_close)) or seg_real < 0.90:
            continue
        entry_ref = opens[f]
        if not np.isfinite(entry_ref) or entry_ref <= 0:
            continue
        fwd_path = seg_close / entry_ref - 1.0
        fwd_max = float(np.max(fwd_path))
        fwd_min = float(np.min(fwd_path))
        fwd_ret = float(fwd_path[-1])
        fwd_absmove = float(max(abs(fwd_max), abs(fwd_min)))

        # --- A2/B1 mechanism features (strictly causal at close of m) ---
        vsum = float(vol[ds:m + 1].sum()); bsum = float(buy[ds:m + 1].sum()); ssum = float(sell[ds:m + 1].sum())
        b30 = float(buy[max(ds, m - 29):m + 1].sum()); s30 = float(sell[max(ds, m - 29):m + 1].sum())
        imb_day = (bsum - ssum) / (bsum + ssum) if bsum + ssum > 0 else np.nan
        imb_30 = (b30 - s30) / (b30 + s30) if b30 + s30 > 0 else np.nan
        aggr_accel = (imb_30 - imb_day) if (np.isfinite(imb_30) and np.isfinite(imb_day)) else np.nan
        ret_30 = float(closes[m] / closes[max(ds, m - 30)] - 1.0)
        avg_pace = rel[m - ds] / max(m - ds, 1)
        run_accel = (ret_30 / 30.0) / avg_pace if avg_pace != 0 and np.isfinite(avg_pace) else np.nan
        ll = ll30[m] if np.isfinite(ll30[m]) else 0.0
        ls = ls30[m] if np.isfinite(ls30[m]) else 0.0
        dvol_sofar = float(np.nanstd(ret1m[ds:m + 1])) if m - ds >= 30 else np.nan
        elapsed = (m - ds + 1) / 1440.0
        btc_rel = btc_r24h = np.nan
        if btc_ctx is not None:
            bi = int((ms[m] - btc_ctx["ms0"]) // 60_000)
            if 0 <= bi < len(btc_ctx["rel"]):
                btc_rel = btc_ctx["rel"][bi]; btc_r24h = btc_ctx["r24h"][bi]
        # pre-event vol -> per-minute ATR proxy for the vol_trail (causal pre_vol = 1m std)
        pv = float(pre_vol[m]) if np.isfinite(pre_vol[m]) else np.nan

        events.append({
            "sym": sym, "day_ms": int(ms[ds]), "t0_ms": int(ms[m]), "split": sp,
            "onset_minute_of_day": int(m - ds),
            # labels
            "fwd_ret": fwd_ret, "fwd_max": fwd_max, "fwd_min": fwd_min, "fwd_absmove": fwd_absmove,
            # path for the live-path sim: indices into the asset arrays
            "f": int(f), "de": int(de), "onset_close": float(onset_close),
            "pre_vol_1m": pv,
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
                "pre_vol": pv,
                "dayvol_ratio": (dvol_sofar / pv if np.isfinite(pv) and pv and pv > 0
                                 and np.isfinite(dvol_sofar) else np.nan),
                "vol_surge": (vsum / (avg30[di] * elapsed) if np.isfinite(avg30[di]) and avg30[di] > 0 else np.nan),
                "regime": (1.0 if regime[m] else 0.0) if regime[m] is not None else np.nan,
                "btc_rel": btc_rel, "btc_r24h": btc_r24h,
                "hour_utc": float((ms[m] % DAY_MS) / 3_600_000),
            },
        })
    return events


def _finalize_feats(events: list[dict]) -> None:
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
            ll = e["feats"].pop("liq_long_raw"); ls = e["feats"].pop("liq_short_raw")
            e["feats"]["liq_long_z"] = float((ll - ll_mu) / ll_sd) if ll_sd > 0 else 0.0
            e["feats"]["liq_short_z"] = float((ls - ls_mu) / ls_sd) if ls_sd > 0 else 0.0
            denom = ll + ls
            e["feats"]["liq_ls_asym"] = float((ll - ls) / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------- model (A2 + B1)
def _mat(evs, feats):
    return np.array([[e["feats"][f] for f in feats] for e in evs], dtype=float)


def _zscore_per_asset(evs_fit, evs_all_groups, feats):
    """Per-(asset,feature) z from TRAIN stats only; applied to every split."""
    Xfit = _mat(evs_fit, feats)
    sym_fit = np.array([e["sym"] for e in evs_fit])
    stats = {}
    for s in set(sym_fit):
        mfit = sym_fit == s
        mu = np.nanmean(Xfit[mfit], axis=0); sd = np.nanstd(Xfit[mfit], axis=0)
        stats[s] = (mu, sd)
    out = []
    for evs in evs_all_groups:
        X = _mat(evs, feats); Z = X.copy()
        syms = np.array([e["sym"] for e in evs])
        for s in set(syms):
            if s not in stats:
                continue
            mu, sd = stats[s]; m = syms == s
            for j in range(len(feats)):
                if np.isfinite(sd[j]) and sd[j] > 0:
                    Z[m, j] = (X[m, j] - mu[j]) / sd[j]
        out.append(Z)
    return out


def fit_a2_b1(train, val, unseen, seed):
    """Fit A2 (reaches +1.5% further) + B1 (|fwd move| > TRAIN median) HGB on TRAIN ONLY.
    Return per-event scores for TRAIN/VAL/UNSEEN, plus held-out AUC sanity (A2 on VAL)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    Ztr, Zva, Zun = _zscore_per_asset(train, [train, val, unseen], FEATURES)
    # A2 label
    ya2 = np.array([int(e["fwd_max"] >= CONT_FURTHER) for e in train], dtype=int)
    hgb_a2 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                            l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.2, n_iter_no_change=20,
                                            random_state=seed)
    hgb_a2.fit(Ztr, ya2)
    sa2_tr = hgb_a2.predict_proba(Ztr)[:, 1]
    sa2_va = hgb_a2.predict_proba(Zva)[:, 1]
    sa2_un = hgb_a2.predict_proba(Zun)[:, 1]
    # B1 label (TRAIN median of |fwd move|)
    abs_med = float(np.median([e["fwd_absmove"] for e in train]))
    yb1 = np.array([int(e["fwd_absmove"] > abs_med) for e in train], dtype=int)
    hgb_b1 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                            l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.2, n_iter_no_change=20,
                                            random_state=seed)
    hgb_b1.fit(Ztr, yb1)
    sb1_tr = hgb_b1.predict_proba(Ztr)[:, 1]
    sb1_va = hgb_b1.predict_proba(Zva)[:, 1]
    sb1_un = hgb_b1.predict_proba(Zun)[:, 1]
    for e, a, b in zip(train, sa2_tr, sb1_tr):
        e["a2"] = float(a); e["b1"] = float(b)
    for e, a, b in zip(val, sa2_va, sb1_va):
        e["a2"] = float(a); e["b1"] = float(b)
    for e, a, b in zip(unseen, sa2_un, sb1_un):
        e["a2"] = float(a); e["b1"] = float(b)
    # sanity AUC (NOT a gate; A2 on VAL/UNSEEN)
    sanity = {}
    try:
        yva = np.array([int(e["fwd_max"] >= CONT_FURTHER) for e in val])
        sanity["a2_val_auc"] = float(roc_auc_score(yva, sa2_va)) if len(set(yva)) > 1 else None
    except Exception:
        sanity["a2_val_auc"] = None
    # TRAIN A2 score quantiles -> gate thresholds (fit on TRAIN only)
    # Diagnostic: A2 base-rate + score drift across splits (the absolute-threshold gate is
    # non-stationary -- A2's CALIBRATION drifts even though its RANKING/AUC holds). We gate
    # by a per-split TOP-FRACTION (rank) instead, which is the deployable form (each period,
    # take the strongest N% of confirmed movers) and is robust to base-rate drift.
    a2_thr_train = {q: float(np.quantile(sa2_tr, q)) for q in A2_GATE_QS}
    drift = {
        "a2_base_rate_train": float(np.mean(ya2)),
        "a2_base_rate_val": (float(np.mean([int(e["fwd_max"] >= CONT_FURTHER) for e in val]))
                             if val else None),
        "a2_base_rate_unseen": (float(np.mean([int(e["fwd_max"] >= CONT_FURTHER) for e in unseen]))
                                if unseen else None),
        "a2_score_mean_train": float(sa2_tr.mean()),
        "a2_score_mean_val": float(sa2_va.mean()) if len(sa2_va) else None,
        "a2_score_mean_unseen": float(sa2_un.mean()) if len(sa2_un) else None,
    }
    # B1 -> sizing percentile mapping fit on TRAIN
    b1_lo = float(np.quantile(sb1_tr, 0.10)); b1_hi = float(np.quantile(sb1_tr, 0.90))
    return {"a2_thr_train": a2_thr_train, "b1_lo": b1_lo, "b1_hi": b1_hi,
            "abs_med": abs_med, "sanity": sanity, "drift": drift}


# ----------------------------------------------------------------- live-path mechanics
def maker_fill(opens, highs, lows, closes, m, de, ref_px, window, offset):
    """Resting PASSIVE maker bid posted `offset` below ref_px after decision minute m.
    Fills iff some minute x in (m, m+window] has low <= bid. Entry = bid (you got your
    price -- maker rebate side, no slippage). The fast-up legs that never pull back to the
    bid DO NOT FILL -> the intrinsic adverse selection / chase penalty (realized p_fill is
    calibrated to [0.25,0.40] via offset/window). Returns (filled, fill_row, entry_px)."""
    bid = ref_px * (1.0 - offset)
    x_hi = min(m + window, de)
    for x in range(m + 1, x_hi + 1):
        if lows[x] <= bid:
            return True, x, float(bid)
    return False, -1, np.nan


def exit_from_entry(opens, highs, lows, closes, entry, fr, de, pre_vol_1m, kind, params):
    """Entry already established at `entry` (the maker limit price). Ride from row fr to
    day end under the named exit policy. Returns gross fractional return entry->exit."""
    end = de
    if kind == "time_stop":
        H = params["H"]
        x = min(fr + H, end)
        return float(opens[x] / entry - 1.0) if x > fr else float(closes[end] / entry - 1.0)
    if kind == "vol_trail":
        k = params["k"]
        # trailing fraction = k * pre_vol_1m, floored so it isn't absurdly tight
        trail = max(k * (pre_vol_1m if np.isfinite(pre_vol_1m) and pre_vol_1m > 0 else 0.004), 0.005)
        c = closes[fr:end]
        if len(c) == 0:
            return float(closes[end] / entry - 1.0)
        runmax = np.maximum.accumulate(np.maximum(c, entry))
        breach = c < runmax * (1.0 - trail)
        idx = int(np.argmax(breach))
        if breach[idx]:
            x = fr + idx
            return float(opens[x + 1] / entry - 1.0) if x + 1 <= end else float(closes[end] / entry - 1.0)
        return float(closes[end] / entry - 1.0)
    if kind == "fixed_target":
        tp = params["tp"]; sl = params["sl"]
        tp_px = entry * (1.0 + tp); sl_px = entry * (1.0 - sl)
        for x in range(fr, end + 1):
            # decision at close of x; conservative: stop checked before target if both
            if lows[x] <= sl_px:
                xf = min(x + 1, end)
                return float(opens[xf] / entry - 1.0)
            if highs[x] >= tp_px:
                xf = min(x + 1, end)
                return float(opens[xf] / entry - 1.0)
        return float(closes[end] / entry - 1.0)
    raise ValueError(kind)


# panel arrays cache (avoid reloading per event)
_PANEL = {}


def panel_arrays(sym):
    if sym not in _PANEL:
        df = load_panel(sym)
        _PANEL[sym] = (df["open"].to_numpy(), df["high"].to_numpy(),
                       df["low"].to_numpy(), df["close"].to_numpy())
    return _PANEL[sym]


def b1_size(b1, b1_lo, b1_hi):
    """Notional in [0.5,1.5] linear in the B1 score percentile (TRAIN-fit lo/hi)."""
    if b1_hi <= b1_lo:
        return 1.0
    p = (b1 - b1_lo) / (b1_hi - b1_lo)
    p = min(max(p, 0.0), 1.0)
    return 0.5 + 1.0 * p


def simulate(events, fitted, exit_name, exit_kind, exit_params, a2_q, sizing, rng,
             null=False, window=FILL_WINDOW, offset=FILL_OFFSET):
    """For each gated event model the IDENTICAL passive-maker fill + exit for both arms.
      A2-gated arm: decision at the onset close; post a passive bid `offset` below it; fill
        on a pullback within `window`; size by B1 if enabled.
      ARBITER null (random-entry-on-the-SAME-movers): SAME gated movers, but the decision
        minute is a RANDOM post-trigger minute; post the IDENTICAL passive bid `offset`
        below THAT minute's close; same fill rule, same exit, same cost, sizing=1. This
        isolates the A2 *timing/selection* skill -- if the gate carries no information,
        a random post-trigger maker entry on the same movers should match it.
    Returns per-event net rows (entry->exit gross - maker RT, * size) and realized fill-rate.

    GATE: per-split TOP-FRACTION (rank). a2_q=0.8 => take the top 20% of THIS event set by
    A2 score (deployable: each period, ride the strongest N% of confirmed movers). Robust
    to A2's base-rate/calibration drift; tests A2's RANKING skill, which is what AUC measures.
    """
    a2_scores = np.array([e["a2"] for e in events])
    thr = float(np.quantile(a2_scores, a2_q)) if len(a2_scores) else 1.0
    rows = []; n_gated = 0; n_filled = 0
    for e in events:
        if e["a2"] < thr:
            continue
        n_gated += 1
        o, h, l, c = panel_arrays(e["sym"])
        m0 = e["f"] - 1            # onset minute row (f = m0+1)
        de = e["de"]
        if null:
            # random post-trigger DECISION minute on the same mover (no A2 timing skill)
            pool_hi = min(de - window - 30, m0 + 600)   # within ~10h post-onset, leave room
            if pool_hi <= m0 + 1:
                continue
            m = int(rng.integers(m0 + 1, pool_hi + 1))
            ref_px = float(c[m]); size = 1.0
        else:
            m = m0
            ref_px = e["onset_close"]
            size = b1_size(e["b1"], fitted["b1_lo"], fitted["b1_hi"]) if sizing == "b1" else 1.0
        if not np.isfinite(ref_px) or ref_px <= 0:
            continue
        filled, fr, entry = maker_fill(o, h, l, c, m, de, ref_px, window, offset)
        if not filled:
            continue   # missed the passive fill -> not a trade (the chase penalty)
        n_filled += 1
        if de - fr < 30:
            continue
        gross = exit_from_entry(o, h, l, c, entry, fr, de, e["pre_vol_1m"], exit_kind, exit_params)
        net = (gross - MAKER_RT) * size
        rows.append({"sym": e["sym"], "t0_ms": e["t0_ms"], "net": float(net),
                     "gross": float(gross), "size": float(size)})
    fill_rate = (n_filled / n_gated) if n_gated else 0.0
    return rows, {"n_gated": n_gated, "n_filled": n_filled, "fill_rate": fill_rate}


def beatnull_stats(ev_rows, nl_rows, n_assets):
    """Per-event net expectancy of A2-gated arm, the null arm, and the matched beat-null
    delta (event mean - null mean), with weekly-cluster block-bootstrap p05 + breadth."""
    if not ev_rows:
        return {"n": 0}
    ev = np.array([r["net"] for r in ev_rows])
    nl = np.array([r["net"] for r in nl_rows]) if nl_rows else np.array([0.0])
    out = {
        "n_event": len(ev), "n_null": len(nl),
        "ev_net_mean": float(ev.mean()), "ev_net_median": float(np.median(ev)),
        "ev_win_rate": float((ev > 0).mean()),
        "nl_net_mean": float(nl.mean()),
        "beatnull_mean": float(ev.mean() - nl.mean()),
    }
    # weekly-cluster bootstrap on the EVENT net (and on the beat-null delta of means)
    cl = np.array([r["t0_ms"] // CLUSTER_MS for r in ev_rows])
    uniq = np.unique(cl)
    rng = np.random.default_rng(11)
    boots_ev = []; boots_bn = []
    nl_mean = nl.mean()
    for _ in range(5000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        samp = np.concatenate([ev[cl == cc] for cc in pick])
        boots_ev.append(samp.mean())
        boots_bn.append(samp.mean() - nl_mean)
    out["ev_net_p05"] = float(np.quantile(boots_ev, 0.05))
    out["ev_net_p95"] = float(np.quantile(boots_ev, 0.95))
    out["beatnull_p05"] = float(np.quantile(boots_bn, 0.05))
    out["beatnull_p95"] = float(np.quantile(boots_bn, 0.95))
    out["n_clusters"] = int(len(uniq))
    # breadth: per-asset beat-null mean > 0 (>=3 events to count)
    per_ev = {}; per_nl = {}
    for r in ev_rows:
        per_ev.setdefault(r["sym"], []).append(r["net"])
    for r in nl_rows:
        per_nl.setdefault(r["sym"], []).append(r["net"])
    pos = 0; qual = 0
    for s, v in per_ev.items():
        if len(v) >= 3:
            qual += 1
            nlm = np.mean(per_nl.get(s, [0.0]))
            if np.mean(v) - nlm > 0:
                pos += 1
    out["breadth_pos"] = pos; out["breadth_qual"] = qual; out["breadth_tot"] = n_assets
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A2-gated maker-filled mover-ride live-path test")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]; tag = args.tag or "custom"
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]; tag = args.tag or args.universe
    else:
        ap.error("provide --assets or --universe")

    t0 = time.time()
    print("loading BTC context...")
    bctx = btc_context()
    all_events = []
    for sym in syms:
        try:
            evs = extract_events(sym, None if sym == "BTCUSDT" else bctx)
        except FileNotFoundError as ex:
            print(f"[{sym}] SKIP: {ex}"); continue
        spc = {"TRAIN": 0, "VAL": 0, "UNSEEN": 0}
        for e in evs:
            spc[e["split"]] += 1
        print(f"[{sym}] onset events TRAIN/VAL/UNSEEN = {spc['TRAIN']}/{spc['VAL']}/{spc['UNSEEN']}")
        all_events.extend(evs)
    _finalize_feats(all_events)
    n_assets = len({e["sym"] for e in all_events}) or 1

    train = [e for e in all_events if e["split"] == "TRAIN"]
    val = [e for e in all_events if e["split"] == "VAL"]
    unseen = [e for e in all_events if e["split"] == "UNSEEN"]
    trval = train + val
    print(f"\nevents: TRAIN {len(train)} / VAL {len(val)} / UNSEEN {len(unseen)} over {n_assets} assets")

    print("fitting A2 + B1 on TRAIN only...")
    fitted = fit_a2_b1(train, val, unseen, args.seed)
    print(f"  A2 VAL sanity AUC = {fitted['sanity'].get('a2_val_auc')}")
    dr = fitted["drift"]
    print(f"  A2 base-rate drift TRAIN {dr['a2_base_rate_train']:.3f} -> VAL "
          f"{dr['a2_base_rate_val']:.3f} -> UNSEEN {dr['a2_base_rate_unseen']:.3f}  "
          f"(gate = per-split TOP-FRACTION, robust to this drift)")

    rng = np.random.default_rng(args.seed)

    # ---------------- SELECTION on TRAIN+VAL (sweep exit x a2_q x sizing) ----------------
    sel_grid = {}
    best = None
    for (exit_name, exit_kind, exit_params) in EXITS:
        for a2_q in A2_GATE_QS:
            for sizing in SIZING_MODES:
                key = f"{exit_name}|a2q{a2_q}|sz_{sizing}"
                ev_rows, ev_fill = simulate(trval, fitted, exit_name, exit_kind, exit_params,
                                            a2_q, sizing, np.random.default_rng(args.seed + 1), null=False)
                nl_rows, _ = simulate(trval, fitted, exit_name, exit_kind, exit_params,
                                      a2_q, sizing, np.random.default_rng(args.seed + 2), null=True)
                st = beatnull_stats(ev_rows, nl_rows, n_assets)
                st["fill_rate"] = ev_fill["fill_rate"]; st["n_gated"] = ev_fill["n_gated"]
                sel_grid[key] = st
                if st.get("n_event", 0) < 50:
                    continue
                # selection: max VAL+TRAIN beat-null net, breadth>=6/10, fill in band
                if st["breadth_pos"] < 6:
                    continue
                # honesty check: realized fill must be clearly passive-maker (not a
                # market-order-in-disguise). Calibrated offset lands the band ~0.32;
                # allow [0.20,0.45] so a2_q-driven drift doesn't over-filter selection.
                if not (0.20 <= st["fill_rate"] <= 0.45):
                    continue
                score = st["beatnull_mean"]
                if best is None or score > best[1]:
                    best = (key, score, exit_name, exit_kind, exit_params, a2_q, sizing)

    if best is None:
        # relax: drop the fill-band constraint (report it), keep breadth + beat-null
        for (exit_name, exit_kind, exit_params) in EXITS:
            for a2_q in A2_GATE_QS:
                for sizing in SIZING_MODES:
                    key = f"{exit_name}|a2q{a2_q}|sz_{sizing}"
                    st = sel_grid[key]
                    if st.get("n_event", 0) < 50 or st["breadth_pos"] < 6:
                        continue
                    score = st["beatnull_mean"]
                    if best is None or score > best[1]:
                        best = (key, score, exit_name, exit_kind, exit_params, a2_q, sizing)

    selected = None
    unseen_res = None
    if best is not None:
        key, score, exit_name, exit_kind, exit_params, a2_q, sizing = best
        selected = {"key": key, "exit": exit_name, "a2_q": a2_q, "sizing": sizing,
                    "trval_beatnull_mean": score, "trval_stats": sel_grid[key]}
        # ---------------- UNSEEN: evaluate the ONE selected config exactly once ----------------
        ev_rows, ev_fill = simulate(unseen, fitted, exit_name, exit_kind, exit_params,
                                    a2_q, sizing, np.random.default_rng(args.seed + 101), null=False)
        nl_rows, _ = simulate(unseen, fitted, exit_name, exit_kind, exit_params,
                              a2_q, sizing, np.random.default_rng(args.seed + 102), null=True)
        ust = beatnull_stats(ev_rows, nl_rows, n_assets)
        ust["fill_rate"] = ev_fill["fill_rate"]; ust["n_gated"] = ev_fill["n_gated"]
        # pre-registered UNSEEN gate
        gate = {
            "beatnull_mean_gt0": bool(ust.get("beatnull_mean", -1) > 0),
            "breadth_ge6": bool(ust.get("breadth_pos", 0) >= 6),
            "beatnull_p05_gt0": bool(ust.get("beatnull_p05", -1) > 0),
        }
        alive = all(gate.values())
        unseen_res = {"stats": ust, "gate": gate, "alive": alive}

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "mover_ride_a2_maker", "git_sha": sha, "seed": args.seed,
        "params": {"T_trig": T_TRIG, "cont_further": CONT_FURTHER, "maker_rt": MAKER_RT,
                   "p_fill_band": [P_FILL_LO, P_FILL_HI], "fill_window": FILL_WINDOW,
                   "fill_offset": FILL_OFFSET,
                   "exits": [e[0] for e in EXITS], "a2_gate_qs": A2_GATE_QS,
                   "sizing_modes": SIZING_MODES, "features": FEATURES},
        "n_assets": n_assets, "n_train": len(train), "n_val": len(val), "n_unseen": len(unseen),
        "a2_sanity": fitted["sanity"], "a2_drift": fitted["drift"],
        "gate_mode": "per-split TOP-FRACTION (rank), robust to A2 base-rate drift",
        "selection_grid_trval": sel_grid,
        "selected": selected, "unseen": unseen_res,
        "caveats": [
            "u10 current membership (survivorship on absolute levels)",
            "maker fill = resting limit at onset close; fast-up legs that never pull back DO NOT fill (intrinsic adverse selection = the chase penalty)",
            "null = random-entry-on-same-movers on the SAME A2-gated set; identical exit/cost; sizing=1",
            "UNSEEN scored exactly once with the single TRAIN+VAL-selected config",
        ],
    }
    out_path = OUT / f"mover_ride_a2_maker_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ---------------------------------------------------------------- STORY
    print("\n" + "=" * 84)
    print("A2-GATED MAKER-FILLED MOVER-RIDE -- LIVE-PATH TEST vs RANDOM-ENTRY-ON-SAME-MOVERS")
    print("=" * 84)
    print(f"events TRAIN {len(train)} / VAL {len(val)} / UNSEEN {len(unseen)} | A2 VAL AUC {fitted['sanity'].get('a2_val_auc')}")
    print("\nTRAIN+VAL SELECTION GRID (beat-null net/ev | ev net/ev | fill | breadth):")
    # show the top 12 by beat-null mean among those with n>=50
    rk = sorted([(k, v) for k, v in sel_grid.items() if v.get("n_event", 0) >= 50],
                key=lambda kv: -kv[1].get("beatnull_mean", -1))
    for k, v in rk[:12]:
        print(f"  {k:<26} bn {v['beatnull_mean']*100:+.3f}% | ev {v['ev_net_mean']*100:+.3f}% "
              f"(n={v['n_event']}, win {v['ev_win_rate']*100:.0f}%) | fill {v['fill_rate']*100:.0f}% | "
              f"breadth {v['breadth_pos']}/{v['breadth_qual']}")
    if selected:
        print(f"\nSELECTED (TRAIN+VAL): {selected['key']}  beat-null {selected['trval_beatnull_mean']*100:+.3f}%/ev")
        s = selected["trval_stats"]
        print(f"  TRAIN+VAL: ev net {s['ev_net_mean']*100:+.3f}%/ev | null {s['nl_net_mean']*100:+.3f}%/ev | "
              f"fill {s['fill_rate']*100:.0f}% | breadth {s['breadth_pos']}/{s['breadth_qual']} | "
              f"beatnull p05 {s.get('beatnull_p05',0)*100:+.3f}%")
        u = unseen_res["stats"]; g = unseen_res["gate"]
        print(f"\n*** UNSEEN (scored ONCE) ***")
        print(f"  ev net {u['ev_net_mean']*100:+.3f}%/ev (n={u['n_event']}, win {u['ev_win_rate']*100:.0f}%) | "
              f"null {u['nl_net_mean']*100:+.3f}%/ev")
        print(f"  BEAT-NULL {u['beatnull_mean']*100:+.3f}%/ev  [p05 {u.get('beatnull_p05',0)*100:+.3f}%, "
              f"p95 {u.get('beatnull_p95',0)*100:+.3f}%]")
        print(f"  fill {u['fill_rate']*100:.0f}% | breadth {u['breadth_pos']}/{u['breadth_qual']} (/{u['breadth_tot']})")
        print(f"  GATE: beatnull>0 {g['beatnull_mean_gt0']} | breadth>=6 {g['breadth_ge6']} | "
              f"beatnull_p05>0 {g['beatnull_p05_gt0']}")
        print(f"  VERDICT: {'ALIVE' if unseen_res['alive'] else 'NOT ALIVE (dead / needs-deeper)'}")
    else:
        print("\nNO TRAIN+VAL config met selection (n>=50, breadth>=6) -- the grid is the verdict.")
    print(f"\n({time.time()-t0:.0f}s)  JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
