"""MOVER-RIDE WAVE-4A: HYBRID PASSIVE+MARKETABLE-LIMIT FILL vs ARBITER NULL.

CONTEXT (Wave-2A result, artifact mover_ride_a2_maker_u10_20260618_002718.json):
  The A2-gated passive-maker backtest is NET DEAD on UNSEEN (beat-null -0.46%/ev;
  gate fails 3/3). The diagnosed kill mechanism:
    1. A passive bid 0.8% below price (FILL_WINDOW=15m) has realized p_fill ~0.34
       in TRAIN/VAL but drops to ~0.21 in UNSEEN (choppier 2024+ regime).
    2. The p_fill ADVERSE SELECTS against the fast-up continuation legs A2 picks:
       the runners never pull back to the bid, so you only keep ~1/3 of movers --
       exactly the weak-continuing third.
    3. A2's RANKING signal IS durable: UNSEEN gross rises monotonically in gate
       tightness. The problem is CAPTURE, not discrimination.

THE LEVER: a HYBRID passive+marketable-limit fill. Place the passive bid first.
  If it fills within the first PASSIVE_WIN minutes -> maker fill (entry = bid,
  maker cost). If STILL UNFILLED after PASSIVE_WIN minutes -> send a marketable
  LIMIT (cross the spread, pay taker + SLIP_BPS slippage) to catch the runners.
  This catches the fast-up continuation legs (which A2 selects FOR), at the cost of
  paying taker+slip on exactly those best fills. The question: does catching the
  runners (at hybrid slippage) flip beat-null > 0 on UNSEEN -- or does the added
  slippage on the best legs just reintroduce the taker hurdle?

DESIGN:
  - Sweep PASSIVE_WIN (minutes before fallback): [3, 5, 10, 15]
  - Sweep SLIP_BPS (extra slippage on the marketable fallback): [10, 15, 20, 30]
  - Taker cost on fallback fills: TAKER_RT (24bps round-trip, vs 12bps maker).
    Specifically: passive fill pays MAKER_RT; marketable fill pays TAKER_RT + SLIP_BPS.
  - The null uses the IDENTICAL hybrid fill (same PASSIVE_WIN, same SLIP_BPS) --
    so the cost differential between arms is zero and the gate tests A2 timing ONLY.
  - Entry price on marketable fallback: onset_close * (1 + SLIP_BPS) -- worst-case
    marketable entry at the bid+slip (conservative; you are chasing a runner).
  - Same swept exits as Wave-2A; same A2/B1 model; same TRAIN < 2024-01-01 fit;
    same TRAIN+VAL selection; UNSEEN scored ONCE.

PRE-REGISTERED UNSEEN GATE (>= 2025-07-01, evaluated ONCE):
  (1) beat-null net per-event > 0;
  (2) breadth >= 6/10 assets (per-asset beat-null mean > 0, >=3 events);
  (3) block-bootstrap p05 of beat-null delta > 0 (weekly clusters).
  ALIVE iff all three hold; else CLOSED with the decisive number named.

Splits: TRAIN < 2024-01-01 | VAL 2024-01-01..2025-07-01 | UNSEEN >= 2025-07-01.
Long-only + spot + lev=1. No shorts. No external data. No git commit.

Run:
  python -m mining.mover_ride_hybrid_fill --universe u10
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
    "outputs": {"study_json": "runs/mining/mover_ride_hybrid_fill_<tag>_<stamp>.json"},
    "invariants": {
        "train_only_model_fit": "A2/B1 HGB fit on TRAIN only; threshold/exit/sizing on TRAIN+VAL",
        "causal_features": "every A2/B1 feature computed from data <= onset close",
        "hybrid_fill_model": (
            "passive bid (FILL_OFFSET below price, live PASSIVE_WIN minutes) -> if filled: "
            "maker cost; if unfilled: marketable limit at onset_close*(1+SLIP_BPS) -> taker+slip cost. "
            "Entry = bid (passive) or onset_close*(1+SLIP_BPS) (marketable). "
            "Null uses IDENTICAL hybrid fill on same gated movers."
        ),
        "arbiter_null": "random-entry-on-same-movers, identical hybrid fill/exit/cost, sizing=1",
        "unseen_once": "UNSEEN evaluated exactly once at the very end; never in fit/select",
        "long_only_spot_lev1": "ride UP-moves only; no shorts; no leverage",
        "next_open_fills": "passive fill: at the bid row; marketable fill: onset_close*(1+slip); exits next-open",
    },
}

# ---- splits ----
TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
VAL_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
DAY_MS = 86_400_000
CLUSTER_MS = 7 * DAY_MS

# ---- event geometry (identical to Wave-2A) ----
T_TRIG = 0.015
CONT_FURTHER = 0.015
B1_FWD_H = 240
MIN_MINUTES_LEFT = 90
MAKER_RT = 0.0012       # 12 bps round-trip (passive fills)
TAKER_RT = 0.0024       # 24 bps round-trip (marketable fallback fills)

# ---- passive bid parameters (same as Wave-2A; calibrated to p_fill ~0.32 in TRAIN/VAL) ----
FILL_OFFSET = 0.008     # passive bid 0.8% below price
# PASSIVE_WIN is swept (below); the full fill window in Wave-2A was 15m

# ---- sweep space: (PASSIVE_WIN_minutes, SLIP_BPS) ----
# PASSIVE_WIN: minutes of passive waiting before marketable fallback fires
# SLIP_BPS: extra slippage on the marketable leg (basis points, e.g. 15 -> 0.0015)
PASSIVE_WINS = [3, 5, 10, 15]
SLIP_BPS_VALS = [10, 15, 20, 30]       # bps of extra slippage on the runner-catch leg

# ---- A2/B1 features (identical to Wave-2A) ----
FEATURES = [
    "oi_d1h", "oi_d4h", "oi_d24h",
    "liq_ratio", "liq_long_z", "liq_short_z", "liq_ls_asym",
    "funding", "aggr_imb", "aggr_accel", "run_accel",
    "overshoot", "t2trig", "pre_vol", "dayvol_ratio", "vol_surge",
    "regime", "btc_rel", "btc_r24h", "hour_utc",
]

# ---- exit policies (identical to Wave-2A) ----
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
A2_GATE_QS = [0.50, 0.60, 0.70, 0.80]
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


# ---- event extraction (identical to Wave-2A) ----
def extract_events(sym: str, btc_ctx: dict | None) -> list[dict]:
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
        m = ds + int(hit[0])
        f = m + 1
        if de - f < MIN_MINUTES_LEFT or not (real[m] and real[f]):
            continue
        onset_close = closes[m]
        if not np.isfinite(onset_close) or onset_close <= 0:
            continue
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
        pv = float(pre_vol[m]) if np.isfinite(pre_vol[m]) else np.nan

        events.append({
            "sym": sym, "day_ms": int(ms[ds]), "t0_ms": int(ms[m]), "split": sp,
            "onset_minute_of_day": int(m - ds),
            "fwd_ret": fwd_ret, "fwd_max": fwd_max, "fwd_min": fwd_min, "fwd_absmove": fwd_absmove,
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


# ---- model (A2 + B1) -- identical to Wave-2A ----
def _mat(evs, feats):
    return np.array([[e["feats"][f] for f in feats] for e in evs], dtype=float)


def _zscore_per_asset(evs_fit, evs_all_groups, feats):
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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    Ztr, Zva, Zun = _zscore_per_asset(train, [train, val, unseen], FEATURES)
    ya2 = np.array([int(e["fwd_max"] >= CONT_FURTHER) for e in train], dtype=int)
    hgb_a2 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                            l2_regularization=1.0, early_stopping=True,
                                            validation_fraction=0.2, n_iter_no_change=20,
                                            random_state=seed)
    hgb_a2.fit(Ztr, ya2)
    sa2_tr = hgb_a2.predict_proba(Ztr)[:, 1]
    sa2_va = hgb_a2.predict_proba(Zva)[:, 1]
    sa2_un = hgb_a2.predict_proba(Zun)[:, 1]
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
    sanity = {}
    try:
        yva = np.array([int(e["fwd_max"] >= CONT_FURTHER) for e in val])
        sanity["a2_val_auc"] = float(roc_auc_score(yva, sa2_va)) if len(set(yva)) > 1 else None
    except Exception:
        sanity["a2_val_auc"] = None
    a2_thr_train = {q: float(np.quantile(sa2_tr, q)) for q in A2_GATE_QS}
    drift = {
        "a2_base_rate_train": float(np.mean(ya2)),
        "a2_base_rate_val": (float(np.mean([int(e["fwd_max"] >= CONT_FURTHER) for e in val]))
                             if val else None),
        "a2_base_rate_unseen": (float(np.mean([int(e["fwd_max"] >= CONT_FURTHER) for e in unseen]))
                                if unseen else None),
    }
    b1_lo = float(np.quantile(sb1_tr, 0.10)); b1_hi = float(np.quantile(sb1_tr, 0.90))
    return {"a2_thr_train": a2_thr_train, "b1_lo": b1_lo, "b1_hi": b1_hi,
            "abs_med": abs_med, "sanity": sanity, "drift": drift}


# ---- HYBRID FILL MODEL ----
def hybrid_fill(opens, highs, lows, closes, m, de,
                ref_px, passive_win, slip_frac):
    """Two-phase fill:
    Phase 1 (passive): resting bid FILL_OFFSET below ref_px for PASSIVE_WIN minutes.
      If low of any minute in (m, m+passive_win] <= bid -> filled passively.
      Entry = bid; cost = MAKER_RT.
    Phase 2 (marketable fallback): if passive missed, send marketable limit at
      ref_px * (1 + slip_frac) AT minute m+passive_win (the next available minute).
      This always fills (we are crossing the spread + paying slip).
      Entry = ref_px * (1 + slip_frac); cost = TAKER_RT.
      Conservative: slip_frac > 0 means we pay above the current price to ensure fill
      (we are chasing the runner, so we pay up).

    Returns: (fill_row, entry_px, fill_kind)
      fill_row: the minute index of the fill (for the exit engine to start from)
      entry_px: the realized entry price
      fill_kind: "passive" or "marketable"
    Always fills (unless day end is too close), never returns unfilled.
    """
    bid = ref_px * (1.0 - FILL_OFFSET)
    # Phase 1: passive window
    x_end = min(m + passive_win, de)
    for x in range(m + 1, x_end + 1):
        if lows[x] <= bid:
            return x, float(bid), "passive"
    # Phase 2: marketable fallback - fire at the first minute after passive_win
    fallback_row = min(m + passive_win + 1, de)
    if fallback_row > de or de - fallback_row < 30:
        # too close to day end: use passive bid price at last minute (accept it)
        return x_end, float(bid), "passive_forced"
    # Entry at onset_close * (1 + slip) = conservative chase price
    entry_px = ref_px * (1.0 + slip_frac)
    return fallback_row, float(entry_px), "marketable"


# ---- exit engine (identical to Wave-2A) ----
def exit_from_entry(opens, highs, lows, closes, entry, fr, de, pre_vol_1m, kind, params):
    end = de
    if kind == "time_stop":
        H = params["H"]
        x = min(fr + H, end)
        return float(opens[x] / entry - 1.0) if x > fr else float(closes[end] / entry - 1.0)
    if kind == "vol_trail":
        k = params["k"]
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
            if lows[x] <= sl_px:
                xf = min(x + 1, end)
                return float(opens[xf] / entry - 1.0)
            if highs[x] >= tp_px:
                xf = min(x + 1, end)
                return float(opens[xf] / entry - 1.0)
        return float(closes[end] / entry - 1.0)
    raise ValueError(kind)


_PANEL: dict = {}


def panel_arrays(sym):
    if sym not in _PANEL:
        df = load_panel(sym)
        _PANEL[sym] = (df["open"].to_numpy(), df["high"].to_numpy(),
                       df["low"].to_numpy(), df["close"].to_numpy())
    return _PANEL[sym]


def b1_size(b1, b1_lo, b1_hi):
    if b1_hi <= b1_lo:
        return 1.0
    p = (b1 - b1_lo) / (b1_hi - b1_lo)
    p = min(max(p, 0.0), 1.0)
    return 0.5 + 1.0 * p


def simulate(events, fitted, exit_name, exit_kind, exit_params,
             a2_q, sizing, rng,
             passive_win, slip_frac,
             null=False):
    """Hybrid-fill simulation for A2-gated arm OR the arbiter null.

    A2-gated arm: decision at onset close; attempt passive bid; fallback to
      marketable at onset_close*(1+slip_frac) after passive_win minutes.
      Cost = MAKER_RT on passive fills, TAKER_RT on marketable fills.
    Null: same gated movers, random post-trigger decision minute, identical
      hybrid fill, identical cost logic, sizing=1.
    Returns per-event net rows and fill-kind stats.
    """
    a2_scores = np.array([e["a2"] for e in events])
    thr = float(np.quantile(a2_scores, a2_q)) if len(a2_scores) else 1.0
    rows = []
    n_gated = 0; n_passive = 0; n_marketable = 0; n_forced = 0; n_tooshort = 0
    for e in events:
        if e["a2"] < thr:
            continue
        n_gated += 1
        o, h, l, c = panel_arrays(e["sym"])
        m0 = e["f"] - 1   # onset minute
        de = e["de"]
        if null:
            pool_hi = min(de - passive_win - 30, m0 + 600)
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
        # hybrid fill
        fr, entry, fkind = hybrid_fill(o, h, l, c, m, de, ref_px, passive_win, slip_frac)
        if fkind == "passive":
            cost = MAKER_RT; n_passive += 1
        elif fkind == "marketable":
            cost = TAKER_RT; n_marketable += 1
        else:  # passive_forced
            cost = MAKER_RT; n_forced += 1
        if de - fr < 30:
            n_tooshort += 1; continue
        if not np.isfinite(entry) or entry <= 0:
            continue
        gross = exit_from_entry(o, h, l, c, entry, fr, de, e["pre_vol_1m"], exit_kind, exit_params)
        net = (gross - cost) * size
        rows.append({
            "sym": e["sym"], "t0_ms": e["t0_ms"], "net": float(net),
            "gross": float(gross), "size": float(size),
            "fill_kind": fkind, "cost": float(cost),
        })
    fill_stats = {
        "n_gated": n_gated,
        "n_filled": n_passive + n_marketable + n_forced,
        "n_passive": n_passive, "n_marketable": n_marketable,
        "n_passive_forced": n_forced,
        "pct_passive": (n_passive / (n_gated or 1)),
        "pct_marketable": ((n_marketable + n_forced) / (n_gated or 1)),
        "n_tooshort": n_tooshort,
    }
    return rows, fill_stats


def beatnull_stats(ev_rows, nl_rows, n_assets, run_bootstrap=False):
    """Compute beat-null stats. Bootstrap is expensive (5000 iters x n_clusters);
    set run_bootstrap=True only for the final selected config (UNSEEN + best TRVAL).
    Selection loop uses run_bootstrap=False for speed (1024 configs)."""
    if not ev_rows:
        return {"n_event": 0}
    ev = np.array([r["net"] for r in ev_rows])
    nl = np.array([r["net"] for r in nl_rows]) if nl_rows else np.array([0.0])
    out = {
        "n_event": len(ev), "n_null": len(nl),
        "ev_net_mean": float(ev.mean()), "ev_net_median": float(np.median(ev)),
        "ev_win_rate": float((ev > 0).mean()),
        "nl_net_mean": float(nl.mean()),
        "beatnull_mean": float(ev.mean() - nl.mean()),
        "pct_passive": float(np.mean([r["fill_kind"] == "passive" for r in ev_rows])),
        "pct_marketable": float(np.mean([r["fill_kind"] in ("marketable", "passive_forced")
                                         for r in ev_rows])),
    }
    # breadth (cheap, always run)
    per_ev: dict = {}; per_nl: dict = {}
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

    if run_bootstrap:
        # weekly-cluster block-bootstrap on beat-null delta (only for final eval)
        cl = np.array([r["t0_ms"] // CLUSTER_MS for r in ev_rows])
        uniq = np.unique(cl)
        rng_b = np.random.default_rng(11)
        boots_ev = []; boots_bn = []
        nl_mean = nl.mean()
        for _ in range(5000):
            pick = rng_b.choice(uniq, size=len(uniq), replace=True)
            samp = np.concatenate([ev[cl == cc] for cc in pick])
            boots_ev.append(samp.mean())
            boots_bn.append(samp.mean() - nl_mean)
        out["ev_net_p05"] = float(np.quantile(boots_ev, 0.05))
        out["ev_net_p95"] = float(np.quantile(boots_ev, 0.95))
        out["beatnull_p05"] = float(np.quantile(boots_bn, 0.05))
        out["beatnull_p95"] = float(np.quantile(boots_bn, 0.95))
        out["n_clusters"] = int(len(uniq))
    else:
        out["ev_net_p05"] = None; out["ev_net_p95"] = None
        out["beatnull_p05"] = None; out["beatnull_p95"] = None
        out["n_clusters"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="A2-gated hybrid-fill mover-ride Wave-4A")
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
        print(f"[{sym}] TRAIN/VAL/UNSEEN = {spc['TRAIN']}/{spc['VAL']}/{spc['UNSEEN']}")
        all_events.extend(evs)
    _finalize_feats(all_events)
    n_assets = len({e["sym"] for e in all_events}) or 1

    train = [e for e in all_events if e["split"] == "TRAIN"]
    val = [e for e in all_events if e["split"] == "VAL"]
    unseen = [e for e in all_events if e["split"] == "UNSEEN"]
    trval = train + val
    print(f"\nevents: TRAIN {len(train)} / VAL {len(val)} / UNSEEN {len(unseen)} / assets {n_assets}")

    print("fitting A2 + B1 on TRAIN only...")
    fitted = fit_a2_b1(train, val, unseen, args.seed)
    print(f"  A2 VAL AUC = {fitted['sanity'].get('a2_val_auc')}")
    dr = fitted["drift"]
    print(f"  A2 base-rate TRAIN {dr['a2_base_rate_train']:.3f} -> VAL {dr['a2_base_rate_val']:.3f} "
          f"-> UNSEEN {dr['a2_base_rate_unseen']:.3f}")

    rng_base = np.random.default_rng(args.seed)

    # ---- SELECTION: sweep all (passive_win, slip_bps, exit, a2_q, sizing) on TRAIN+VAL ----
    # The selection grid is the full cross-product.
    sel_grid: dict = {}
    best = None
    n_combos = len(PASSIVE_WINS) * len(SLIP_BPS_VALS) * len(EXITS) * len(A2_GATE_QS) * len(SIZING_MODES)
    print(f"\nselection sweep: {n_combos} configs on TRAIN+VAL...")

    for pw in PASSIVE_WINS:
        for slip_bps in SLIP_BPS_VALS:
            slip_frac = slip_bps / 10_000.0
            fill_key = f"pw{pw}_sl{slip_bps}"
            for (exit_name, exit_kind, exit_params) in EXITS:
                for a2_q in A2_GATE_QS:
                    for sizing in SIZING_MODES:
                        key = f"{fill_key}|{exit_name}|a2q{a2_q}|sz_{sizing}"
                        ev_rows, ev_fill = simulate(
                            trval, fitted, exit_name, exit_kind, exit_params,
                            a2_q, sizing, np.random.default_rng(args.seed + 1),
                            passive_win=pw, slip_frac=slip_frac, null=False)
                        nl_rows, _ = simulate(
                            trval, fitted, exit_name, exit_kind, exit_params,
                            a2_q, sizing, np.random.default_rng(args.seed + 2),
                            passive_win=pw, slip_frac=slip_frac, null=True)
                        # no bootstrap in selection loop (speed: 1024 configs)
                        st = beatnull_stats(ev_rows, nl_rows, n_assets, run_bootstrap=False)
                        st.update({"fill_stats_ev": ev_fill,
                                   "passive_win": pw, "slip_bps": slip_bps})
                        sel_grid[key] = st
                        if st.get("n_event", 0) < 50:
                            continue
                        if st.get("breadth_pos", 0) < 6:
                            continue
                        score = st["beatnull_mean"]
                        if best is None or score > best[1]:
                            best = (key, score, exit_name, exit_kind, exit_params,
                                    a2_q, sizing, pw, slip_frac)

    if best is None:
        print("  [no config met n>=50 + breadth>=6 on TRAIN+VAL; relaxing breadth to >=4]")
        for pw in PASSIVE_WINS:
            for slip_bps in SLIP_BPS_VALS:
                slip_frac = slip_bps / 10_000.0
                fill_key = f"pw{pw}_sl{slip_bps}"
                for (exit_name, exit_kind, exit_params) in EXITS:
                    for a2_q in A2_GATE_QS:
                        for sizing in SIZING_MODES:
                            key = f"{fill_key}|{exit_name}|a2q{a2_q}|sz_{sizing}"
                            st = sel_grid[key]
                            if st.get("n_event", 0) < 50 or st.get("breadth_pos", 0) < 4:
                                continue
                            score = st["beatnull_mean"]
                            if best is None or score > best[1]:
                                best = (key, score, exit_name, exit_kind, exit_params,
                                        a2_q, sizing, pw, slip_frac)

    selected = None; unseen_res = None
    if best is not None:
        (key, score, exit_name, exit_kind, exit_params, a2_q, sizing, pw, slip_frac) = best
        # Re-run the selected config with bootstrap for proper TRVAL reporting
        ev_rows_tv, ev_fill_tv = simulate(
            trval, fitted, exit_name, exit_kind, exit_params,
            a2_q, sizing, np.random.default_rng(args.seed + 1),
            passive_win=pw, slip_frac=slip_frac, null=False)
        nl_rows_tv, _ = simulate(
            trval, fitted, exit_name, exit_kind, exit_params,
            a2_q, sizing, np.random.default_rng(args.seed + 2),
            passive_win=pw, slip_frac=slip_frac, null=True)
        trval_stats_full = beatnull_stats(ev_rows_tv, nl_rows_tv, n_assets, run_bootstrap=True)
        trval_stats_full["fill_stats_ev"] = ev_fill_tv

        selected = {
            "key": key, "exit": exit_name, "a2_q": a2_q, "sizing": sizing,
            "passive_win": pw, "slip_bps": int(slip_frac * 10_000),
            "trval_beatnull_mean": score,
            "trval_stats": trval_stats_full,
        }
        print(f"\nSELECTED: {key}  beat-null {score*100:+.4f}%/ev")

        # ---- UNSEEN: evaluate the ONE selected config exactly ONCE ----
        print("evaluating UNSEEN (once)...")
        ev_rows, ev_fill = simulate(
            unseen, fitted, exit_name, exit_kind, exit_params,
            a2_q, sizing, np.random.default_rng(args.seed + 101),
            passive_win=pw, slip_frac=slip_frac, null=False)
        nl_rows, _ = simulate(
            unseen, fitted, exit_name, exit_kind, exit_params,
            a2_q, sizing, np.random.default_rng(args.seed + 102),
            passive_win=pw, slip_frac=slip_frac, null=True)
        # bootstrap for the final UNSEEN eval (the one gate that matters)
        ust = beatnull_stats(ev_rows, nl_rows, n_assets, run_bootstrap=True)
        ust["fill_stats_ev"] = ev_fill
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
        "tool": "mover_ride_hybrid_fill", "version": "wave4a", "git_sha": sha, "seed": args.seed,
        "params": {
            "T_trig": T_TRIG, "cont_further": CONT_FURTHER,
            "maker_rt": MAKER_RT, "taker_rt": TAKER_RT,
            "fill_offset": FILL_OFFSET,
            "passive_wins_swept": PASSIVE_WINS,
            "slip_bps_swept": SLIP_BPS_VALS,
            "exits": [e[0] for e in EXITS], "a2_gate_qs": A2_GATE_QS,
            "sizing_modes": SIZING_MODES, "features": FEATURES,
        },
        "design": (
            "Hybrid fill: passive bid (FILL_OFFSET below price, live PASSIVE_WIN min) -> "
            "if filled: maker cost (12bps RT); if unfilled: marketable limit at "
            "onset_close*(1+SLIP_BPS) -> taker cost (24bps RT). "
            "Catches the fast-up runners at the cost of taker+slip on those fills. "
            "Null uses identical hybrid fill on same A2-gated movers; random decision minute."
        ),
        "n_assets": n_assets, "n_train": len(train), "n_val": len(val), "n_unseen": len(unseen),
        "a2_sanity": fitted["sanity"], "a2_drift": fitted["drift"],
        "n_selection_configs": n_combos,
        "selected": selected, "unseen": unseen_res,
        "caveats": [
            "u10 current membership (survivorship on absolute levels)",
            "marketable fallback entry = onset_close*(1+slip_frac): conservative (worst-case chase price)",
            "null = random-entry-on-same-movers; identical hybrid fill; sizing=1",
            "UNSEEN scored exactly once with the single TRAIN+VAL-selected config",
            "cost is fill-kind-specific: passive->12bps RT, marketable->24bps RT",
        ],
    }
    out_path = OUT / f"mover_ride_hybrid_fill_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ---- STORY ----
    print("\n" + "=" * 88)
    print("WAVE-4A: A2-GATED HYBRID-FILL MOVER-RIDE vs RANDOM-ENTRY-ON-SAME-MOVERS (ARBITER)")
    print("=" * 88)
    print(f"events TRAIN {len(train)} / VAL {len(val)} / UNSEEN {len(unseen)} / assets {n_assets}")
    print(f"A2 VAL AUC = {fitted['sanity'].get('a2_val_auc')}")
    print(f"selection grid: {n_combos} configs")

    print("\nTOP-15 TRAIN+VAL configs by beat-null (n>=50, any breadth):")
    rk = sorted(
        [(k, v) for k, v in sel_grid.items() if v.get("n_event", 0) >= 50],
        key=lambda kv: -kv[1].get("beatnull_mean", -1)
    )
    for k, v in rk[:15]:
        fs = v.get("fill_stats_ev", {})
        pm = fs.get("pct_marketable", 0)
        print(f"  {k:<40} bn {v['beatnull_mean']*100:+.3f}% | ev {v['ev_net_mean']*100:+.3f}%"
              f" (n={v['n_event']}, win {v['ev_win_rate']*100:.0f}%)"
              f" | mkt% {pm*100:.0f}% | br {v['breadth_pos']}/{v.get('breadth_qual','?')}")

    if selected:
        s = selected["trval_stats"]
        fs = s.get("fill_stats_ev", {})
        print(f"\nSELECTED: {selected['key']}")
        print(f"  passive_win={selected['passive_win']}m  slip={selected['slip_bps']}bps")
        print(f"  TRAIN+VAL: beat-null {selected['trval_beatnull_mean']*100:+.4f}%/ev")
        print(f"    ev net {s['ev_net_mean']*100:+.3f}%/ev | null {s['nl_net_mean']*100:+.3f}%/ev "
              f"| beatnull p05 {s.get('beatnull_p05',0)*100:+.3f}%")
        print(f"    fill: {fs.get('pct_passive',0)*100:.0f}% passive / "
              f"{fs.get('pct_marketable',0)*100:.0f}% marketable "
              f"(n_gated={fs.get('n_gated',0)})")
        print(f"    breadth {s['breadth_pos']}/{s.get('breadth_qual','?')} assets positive")

        u = unseen_res["stats"]; g = unseen_res["gate"]
        ufs = u.get("fill_stats_ev", {})
        print(f"\n*** UNSEEN (scored ONCE, never re-selected) ***")
        print(f"  n_event={u['n_event']} | win {u['ev_win_rate']*100:.0f}%")
        print(f"  ev net  {u['ev_net_mean']*100:+.3f}%/ev  |  null {u['nl_net_mean']*100:+.3f}%/ev")
        print(f"  BEAT-NULL: {u['beatnull_mean']*100:+.4f}%/ev  "
              f"[p05 {u.get('beatnull_p05',0)*100:+.4f}%  p95 {u.get('beatnull_p95',0)*100:+.4f}%]")
        print(f"  fill mix: {u.get('pct_passive',0)*100:.0f}% passive / "
              f"{u.get('pct_marketable',0)*100:.0f}% marketable")
        print(f"  breadth: {u['breadth_pos']}/{u.get('breadth_qual','?')} (need 6/{n_assets})")
        print(f"  n_clusters={u.get('n_clusters',0)}")
        print(f"  GATE: beatnull>0 {g['beatnull_mean_gt0']} | "
              f"breadth>=6 {g['breadth_ge6']} | beatnull_p05>0 {g['beatnull_p05_gt0']}")
        print(f"\n  VERDICT: {'ALIVE -- move-riding has a realizable internal-data form' if unseen_res['alive'] else 'NOT ALIVE -- internal-data long-only move-riding is CLOSED'}")
    else:
        print("\nNO config met selection criteria -- grid is the verdict.")

    print(f"\n({time.time()-t0:.0f}s)  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
