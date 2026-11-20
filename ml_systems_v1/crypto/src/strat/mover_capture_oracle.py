"""src/strat/mover_capture_oracle.py -- MOVE-CAPTURE ORACLE DECOMPOSITION

OBJECTIVE (2026-06-20):
  Across a 7d forward window (14d lookback for signals), identify the ORACLE
  top-mover picks and measure what fraction a CAUSAL ungated engine captures.

KEY PROBLEM BEING SOLVED:
  Prior engine's SMA200 per-asset gate structurally deletes movers:
    - Top-1 forward-7d mover is below its OWN SMA200 on 39% of days
    - 38% of top-3 movers excluded by gate
    - BTC is the top mover only 9% of days
  Fix = MARKET-LEVEL circuit-breaker (BTC-trend / breadth / vol) that
  risk-offs the TOTAL BOOK, NOT individual assets.

JUDGE ON:
  (a) CAPTURE-RATE = realized_net / oracle_available_move after taker cost
  (b) Random-7d-slice profitability vs buy-hold AND vs gated router
  (c) Bear-survival via market-level circuit-breaker (2022 + 2024/2025 DDs)

ORACLE DEFINITION:
  For each 7-day forward window starting at date d (using 14d lookback for
  signal construction), the oracle picks the assets that delivered the
  highest realized 7-day forward return (clairvoyant; the CEILING).
  Oracle-K: hold top-K assets by realized-fwd-7d-return (EW), entered at
  close of bar d (lagged 1 bar = bar d+1 fill), exited at bar d+7.

CAUSAL ENGINE (ungated):
  Signal at day d: rank all assets by 14-day momentum (past-only).
  No per-asset SMA200 gate. Rebalance every 7 days.
  Market circuit-breaker: scale TOTAL book exposure by a MARKET signal
  (BTC vs SMA200 * breadth * vol) -> 100%/50%/20% exposure tiers.

LEAK-FREE:
  - 14d lookback only uses prices[d-14..d] (strictly causal)
  - Oracle labels only used for ORACLE weight matrix (ceiling, never trained on)
  - Date-block permutation for any AUC test
  - Walk-forward: no peeking beyond the rebalance date

RWYB: python -m strat.mover_capture_oracle
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as ref

COST = lab.COST          # taker round-trip (~0.24% of position)
TAKER_RT = COST          # alias
N_SLICES = 500
SEEDS = [11, 23, 42]
OOS_START = "2022-01-01"
OOS_END = "2026-06-01"
DATA_START = "2020-01-01"


# ===========================================================================
# SECTION 1: ORACLE CONSTRUCTION
# The 7-day ahead oracle -- the ceiling. For each date d, the oracle knows
# which assets will move the most over the next 7 days (perfect foresight).
# ===========================================================================

def build_oracle_weights(ind: dict, K: int = 3) -> pd.DataFrame:
    """Oracle top-K by realized 7-day forward return. EW over top-K.
    CEILING ONLY -- uses future data. Never used for training or selection.
    Entered at close of day d (acted at d+1 by lag convention), exited at d+7.
    """
    C = ind["C"]
    fwd7 = C.shift(-7) / C - 1     # forward 7-day return (future leak = oracle by design)
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for i in range(len(C) - 7):
        d = C.index[i]
        row = fwd7.iloc[i]
        valid = row.dropna().sort_values(ascending=False)
        if len(valid) == 0:
            continue
        picks = valid.head(K)
        w = 1.0 / len(picks)
        for s in picks.index:
            W.loc[d, s] = w
    return W


def oracle_available_move_series(ind: dict, K: int = 3) -> pd.Series:
    """For each date d, the average realized 7d forward return of the top-K movers.
    This is the DENOMINATOR for capture-rate. Annualised: /7*365 but we report raw 7d.
    """
    C = ind["C"]
    fwd7 = C.shift(-7) / C - 1
    result = {}
    for i in range(len(C) - 7):
        d = C.index[i]
        row = fwd7.iloc[i].dropna().sort_values(ascending=False)
        if len(row) == 0:
            result[d] = np.nan
            continue
        result[d] = float(row.head(K).mean())
    return pd.Series(result)


# ===========================================================================
# SECTION 2: MOVER GATE DIAGNOSIS
# Quantify exactly HOW MUCH the SMA200 gate excludes movers (the proven failure).
# ===========================================================================

def diagnose_sma200_gate(ind: dict) -> dict:
    """Diagnose per-asset SMA200 gate: what fraction of top-K movers are gated out?"""
    C = ind["C"]
    gate = ind["gate"]     # C > SMA200 (bool DataFrame)
    fwd7 = C.shift(-7) / C - 1

    # Only look at OOS region
    oos = C.index >= pd.Timestamp(OOS_START)
    C_oos = C[oos]; gate_oos = gate[oos]; fwd7_oos = fwd7[oos]

    # Per day: identify top-1, top-3 mover, check if gated
    top1_excluded = []
    top3_excluded_frac = []
    top1_is_btc = []

    for i in range(len(C_oos) - 7):
        d = C_oos.index[i]
        row_fwd = fwd7_oos.iloc[i].dropna()
        if len(row_fwd) == 0:
            continue
        sorted_fwd = row_fwd.sort_values(ascending=False)
        top1 = sorted_fwd.index[0]
        top3 = sorted_fwd.index[:3].tolist()

        # Is top-1 below its own SMA200?
        g_top1 = bool(gate_oos.loc[d, top1]) if top1 in gate_oos.columns else True
        top1_excluded.append(not g_top1)

        # Fraction of top-3 excluded
        n_excl = sum(1 for s in top3 if s in gate_oos.columns and not bool(gate_oos.loc[d, s]))
        top3_excluded_frac.append(n_excl / len(top3))

        # Is top mover BTC?
        top1_is_btc.append(top1 == "BTCUSDT")

    return {
        "top1_excluded_pct": round(100 * np.mean(top1_excluded), 1),
        "top3_excluded_frac_mean_pct": round(100 * np.mean(top3_excluded_frac), 1),
        "top1_is_btc_pct": round(100 * np.mean(top1_is_btc), 1),
        "n_days_oos": len(top1_excluded),
        "diagnosis": (
            f"Top-1 mover excluded by SMA200 gate: {round(100*np.mean(top1_excluded),1)}% of OOS days | "
            f"Top-3 exclusion rate: {round(100*np.mean(top3_excluded_frac),1)}% | "
            f"BTC is top mover: {round(100*np.mean(top1_is_btc),1)}%"
        )
    }


# ===========================================================================
# SECTION 3: MARKET-LEVEL CIRCUIT BREAKER (replaces per-asset gate)
# Scale TOTAL book exposure by market health.
# ===========================================================================

def market_exposure_scalar(ind: dict) -> pd.Series:
    """Causal market-level exposure scalar: 3 tiers based on BTC-trend x breadth x vol.

    FULL (1.0):    BTC > SMA200 AND breadth > 50%
    HALF (0.5):    BTC > SMA200 AND breadth 30-50%, OR BTC < SMA200 AND vol < vol75
    DEFENSIVE (0.2): BTC < SMA200 AND vol > vol75 (full bear with high vol)

    Pre-computed from rolling past data only. vol threshold from rolling 180d quantile.
    """
    C = ind["C"]
    sma200 = ind["sma200"]
    sma50 = ind["sma50"]
    vol20 = ind["vol20"]

    # Rolling vol threshold: 75th pctile of BTC vol over past 180 days (causal)
    btc_vol = vol20["BTCUSDT"]
    vol75 = btc_vol.rolling(180, min_periods=60).quantile(0.75)

    scalar = pd.Series(1.0, index=C.index)

    for i in range(len(C)):
        d = C.index[i]
        btc_price = C.loc[d, "BTCUSDT"]
        btc_s200 = sma200.loc[d, "BTCUSDT"]

        if pd.isna(btc_s200):   # warmup
            scalar.iloc[i] = 0.5
            continue

        btc_above_200 = btc_price > btc_s200

        # breadth: fraction of assets above their own SMA50
        above50 = 0; total = 0
        for sym in C.columns:
            cv = C.loc[d, sym]; s50 = sma50.loc[d, sym]
            if pd.notna(cv) and pd.notna(s50):
                above50 += int(cv > s50); total += 1
        breadth = above50 / total if total > 0 else 0.5

        bv = btc_vol.iloc[i] if not pd.isna(btc_vol.iloc[i]) else 0.5
        v75 = vol75.iloc[i] if not pd.isna(vol75.iloc[i]) else 9999.0
        hi_vol = bv > v75

        if btc_above_200 and breadth > 0.50:
            scalar.iloc[i] = 1.0           # FULL: clean bull
        elif btc_above_200 and breadth > 0.30 and not hi_vol:
            scalar.iloc[i] = 0.7           # PARTIAL: recovering
        elif btc_above_200 and (breadth <= 0.30 or hi_vol):
            scalar.iloc[i] = 0.5           # CAUTIOUS: chop or vol spike
        elif (not btc_above_200) and not hi_vol:
            scalar.iloc[i] = 0.3           # BEAR-LOW-VOL: mild defensive
        else:
            scalar.iloc[i] = 0.15          # BEAR-HIGH-VOL: max defensive

    return scalar


# ===========================================================================
# SECTION 4: CAUSAL UNGATED MOVER ENGINE
# Rank by 14d momentum (past-only), hold top-K, NO per-asset gate.
# Market circuit-breaker scales total exposure.
# ===========================================================================

def build_ungated_mover_weights(ind: dict, K: int = 3, rebal: int = 7,
                                 use_circuit_breaker: bool = True) -> pd.DataFrame:
    """Causal ungated mover engine.
    Signal: 14d momentum rank (past-only, causal).
    No per-asset SMA200 gate.
    Rebalance every `rebal` days.
    Market circuit-breaker: scale TOTAL exposure by market_exposure_scalar().
    """
    C = ind["C"]
    mom14 = ind["mom14"]   # C / C.shift(14) - 1  (past-only)

    if use_circuit_breaker:
        cb_scalar = market_exposure_scalar(ind)
    else:
        cb_scalar = pd.Series(1.0, index=C.index)

    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last_rebal = -999
    cur_w = {}

    for i, d in enumerate(C.index):
        # Rebalance every `rebal` bars
        if i - last_rebal >= rebal:
            row_mom = mom14.iloc[i]
            valid = row_mom.dropna().sort_values(ascending=False)
            if len(valid) > 0:
                picks = valid.head(K)
                w_raw = 1.0 / len(picks)
                cur_w = {s: w_raw for s in picks.index}
            else:
                cur_w = {}
            last_rebal = i

        # Apply circuit-breaker scaling to total exposure
        scale = float(cb_scalar.iloc[i])
        row = {col: cur_w.get(col, 0.0) * scale for col in C.columns}
        W.iloc[i] = row

    return W


def build_ungated_mover_weights_with_carry(ind: dict, K: int = 3, rebal: int = 7) -> pd.DataFrame:
    """Variant: no gate, but EXIT a position early if it drops > 7% from entry (a soft stop).
    Winners are HELD and carried forward even past the rebal date if they're still up.
    This explores the asymmetric exit (let winners run, cut losers).
    """
    C = ind["C"]
    mom14 = ind["mom14"]
    cb_scalar = market_exposure_scalar(ind)

    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last_rebal = -999
    cur_w = {}
    entry_prices = {}  # track entry price per asset for exit logic

    for i, d in enumerate(C.index):
        scale = float(cb_scalar.iloc[i])

        # Check if any current positions should be exited (stop-loss at -7%)
        stops_hit = set()
        for sym, ew in cur_w.items():
            if sym in entry_prices and sym in C.columns:
                cur_price = C.iloc[i].get(sym, np.nan)
                ep = entry_prices[sym]
                if pd.notna(cur_price) and pd.notna(ep) and ep > 0:
                    dd = (cur_price / ep) - 1.0
                    if dd < -0.07:   # 7% drawdown from entry -> exit
                        stops_hit.add(sym)
        if stops_hit:
            cur_w = {k: v for k, v in cur_w.items() if k not in stops_hit}
            # rebalance survivors to equal weight
            if cur_w:
                ew_new = 1.0 / len(cur_w)
                cur_w = {k: ew_new for k in cur_w}

        # Rebalance every `rebal` bars
        if i - last_rebal >= rebal:
            row_mom = mom14.iloc[i]
            valid = row_mom.dropna().sort_values(ascending=False)
            if len(valid) > 0:
                picks = valid.head(K)
                ew = 1.0 / len(picks)
                new_w = {s: ew for s in picks.index}
                # Record entry prices for new positions
                for sym in new_w:
                    if sym not in cur_w:   # new entry
                        ep = C.iloc[i].get(sym, np.nan)
                        if pd.notna(ep):
                            entry_prices[sym] = ep
                cur_w = new_w
            last_rebal = i

        row = {col: cur_w.get(col, 0.0) * scale for col in C.columns}
        W.iloc[i] = row

    return W


# ===========================================================================
# SECTION 5: CAPTURE RATE COMPUTATION
# For each 7-day window, measure realized / oracle_available_move.
# ===========================================================================

def compute_capture_rate(W_engine: pd.DataFrame, W_oracle: pd.DataFrame,
                         ind: dict, oos_start: str, oos_end: str) -> dict:
    """Compute capture rate: realized_engine_net / oracle_available_gross.

    For each non-overlapping 7-day window in [oos_start, oos_end]:
      - oracle_return: compound return of oracle top-K (gross, perfect foresight ceiling)
      - engine_return: compound return of engine (net of cost)
      - capture_rate: engine_return / oracle_return

    Pooled capture-rate = sum(engine_net) / sum(oracle_gross) over all windows.
    """
    bret_engine = ref.book_daily_returns(W_engine, ind)
    bret_oracle = ref.book_daily_returns(W_oracle, ind)   # oracle also costs (lag+cost for fairness)

    idx = bret_engine.index
    oos_mask = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))
    oos_idx = idx[oos_mask]

    window_size = 7
    windows = []
    i = 0
    while i + window_size <= len(oos_idx):
        sl = oos_idx[i: i + window_size]
        e_ret = float((1 + bret_engine.loc[sl]).prod() - 1)
        o_ret = float((1 + bret_oracle.loc[sl]).prod() - 1)
        windows.append({
            "start": str(sl[0].date()),
            "engine_7d_net": round(e_ret * 100, 2),
            "oracle_7d_gross": round(o_ret * 100, 2),
            "capture": round(e_ret / o_ret, 4) if abs(o_ret) > 1e-6 else None,
        })
        i += window_size  # non-overlapping

    eng_rets = np.array([w["engine_7d_net"] for w in windows]) / 100
    orc_rets = np.array([w["oracle_7d_gross"] for w in windows]) / 100

    # Pooled capture: sum(engine) / sum(oracle) over positive-oracle windows
    pos_orc_mask = orc_rets > 0
    pooled_capture = (
        float(eng_rets[pos_orc_mask].sum() / orc_rets[pos_orc_mask].sum())
        if pos_orc_mask.sum() > 0 else 0.0
    )

    # Also: per-window capture distribution
    valid_caps = [w["capture"] for w in windows if w["capture"] is not None]

    return {
        "n_windows": len(windows),
        "n_pos_oracle_windows": int(pos_orc_mask.sum()),
        "pooled_capture_rate": round(pooled_capture, 4),
        "median_capture_rate": round(float(np.median(valid_caps)), 4) if valid_caps else None,
        "mean_capture_rate": round(float(np.mean(valid_caps)), 4) if valid_caps else None,
        "oracle_mean_7d": round(float(orc_rets.mean() * 100), 2),
        "oracle_median_7d": round(float(np.median(orc_rets) * 100), 2),
        "engine_mean_7d": round(float(eng_rets.mean() * 100), 2),
        "engine_median_7d": round(float(np.median(eng_rets) * 100), 2),
        "windows": windows,
    }


# ===========================================================================
# SECTION 6: REGIME-STRATIFIED CAPTURE RATE
# Break capture rate by regime: bull / bear / chop
# ===========================================================================

def assign_regime(ind: dict) -> pd.Series:
    """Causal daily regime labels: BULL / BEAR / CHOP based on BTC vs SMA200 + breadth."""
    C = ind["C"]
    sma200 = ind["sma200"]
    sma50 = ind["sma50"]
    regimes = {}

    for i, d in enumerate(C.index):
        btc_price = C.loc[d, "BTCUSDT"]
        btc_s200 = sma200.loc[d, "BTCUSDT"]
        if pd.isna(btc_s200):
            regimes[d] = "WARMUP"
            continue
        btc_up = btc_price > btc_s200
        above50 = sum(1 for sym in C.columns
                      if pd.notna(C.loc[d, sym]) and pd.notna(sma50.loc[d, sym])
                      and C.loc[d, sym] > sma50.loc[d, sym])
        total = sum(1 for sym in C.columns if pd.notna(C.loc[d, sym]) and pd.notna(sma50.loc[d, sym]))
        breadth = above50 / total if total > 0 else 0.5

        if btc_up and breadth >= 0.5:
            regimes[d] = "BULL"
        elif not btc_up:
            regimes[d] = "BEAR"
        else:
            regimes[d] = "CHOP"

    return pd.Series(regimes)


def regime_stratified_capture(W_engine: pd.DataFrame, W_oracle: pd.DataFrame,
                               ind: dict, oos_start: str, oos_end: str) -> dict:
    """Break capture rate by regime (BULL/BEAR/CHOP) for 7-day windows."""
    regimes = assign_regime(ind)
    bret_engine = ref.book_daily_returns(W_engine, ind)
    bret_oracle = ref.book_daily_returns(W_oracle, ind)

    idx = bret_engine.index
    oos_mask = (idx >= pd.Timestamp(oos_start)) & (idx < pd.Timestamp(oos_end))
    oos_idx = idx[oos_mask]

    window_size = 7
    by_regime: dict[str, list] = {"BULL": [], "BEAR": [], "CHOP": []}

    i = 0
    while i + window_size <= len(oos_idx):
        sl = oos_idx[i: i + window_size]
        e_ret = float((1 + bret_engine.loc[sl]).prod() - 1)
        o_ret = float((1 + bret_oracle.loc[sl]).prod() - 1)
        # Regime = most common regime in the window
        window_regimes = [regimes.get(d, "CHOP") for d in sl]
        from collections import Counter
        regime = Counter(window_regimes).most_common(1)[0][0]
        if regime in by_regime:
            cap = e_ret / o_ret if abs(o_ret) > 1e-6 else None
            by_regime[regime].append({
                "e": e_ret, "o": o_ret, "cap": cap,
                "start": str(sl[0].date())
            })
        i += window_size

    out = {}
    for r, rows in by_regime.items():
        if not rows:
            out[r] = {"n": 0}
            continue
        e_arr = np.array([x["e"] for x in rows])
        o_arr = np.array([x["o"] for x in rows])
        caps = [x["cap"] for x in rows if x["cap"] is not None]
        pos_o = o_arr > 0
        pooled = float(e_arr[pos_o].sum() / o_arr[pos_o].sum()) if pos_o.sum() > 0 else 0.0
        out[r] = {
            "n": len(rows),
            "pooled_capture": round(pooled, 4),
            "median_capture": round(float(np.median(caps)), 4) if caps else None,
            "engine_mean_7d_pct": round(float(e_arr.mean() * 100), 2),
            "oracle_mean_7d_pct": round(float(o_arr.mean() * 100), 2),
            "engine_pos_rate": round(float((e_arr > 0).mean() * 100), 1),
        }
    return out


# ===========================================================================
# SECTION 7: CAUSAL SIGNAL REVERSE-ENGINEERING
# What causal signal best approximates the oracle's pick?
# Test: mom-7d, mom-14d, RSI14, vol-breakout, momentum-rank as predictors of
# being in the oracle's top-K selection.
# ===========================================================================

def reverse_engineer_oracle_signal(ind: dict, K: int = 3) -> dict:
    """Find the causal signal that best predicts oracle top-K membership.
    Returns AUC per signal on OOS region (date-block permutation for null check).
    """
    from sklearn.metrics import roc_auc_score

    C = ind["C"]
    fwd7 = C.shift(-7) / C - 1

    # OOS region only, exclude warmup
    oos_mask = C.index >= pd.Timestamp(OOS_START)
    oos_dates = C.index[oos_mask]

    # Build label: binary (1 = in oracle top-K, 0 = not)
    labels = []
    signals = {
        "mom7_rank": [],
        "mom14_rank": [],
        "mom30_rank": [],
        "rsi14_rank": [],
        "mom14": [],
        "mom7": [],
        "range_pos": [],   # position within 14d range
    }

    for d in oos_dates[:-8]:   # exclude last 7 bars (no forward label)
        row_fwd = fwd7.loc[d].dropna()
        if len(row_fwd) < K:
            continue
        top_k_set = set(row_fwd.nlargest(K).index)

        for sym in C.columns:
            if sym not in row_fwd.index:
                continue
            lbl = 1 if sym in top_k_set else 0

            # Causal features
            m7 = ind["mom7"].loc[d, sym]
            m14 = ind["mom14"].loc[d, sym]
            m30 = ind["mom30"].loc[d, sym]
            rsi = ind["rsi14"].loc[d, sym]
            hh14 = ind["hh14"].loc[d, sym]
            ll14 = ind["ll14"].loc[d, sym]
            cp = C.loc[d, sym]
            range_p = (cp - ll14) / (hh14 - ll14 + 1e-8) if pd.notna(hh14) and pd.notna(ll14) else np.nan

            labels.append(lbl)
            # Rank signals cross-sectionally (rank within the day)
            signals["mom7_rank"].append(float(pd.notna(m7)) * (m7 if pd.notna(m7) else 0))
            signals["mom14_rank"].append(m14 if pd.notna(m14) else 0)
            signals["mom30_rank"].append(m30 if pd.notna(m30) else 0)
            signals["rsi14_rank"].append(rsi if pd.notna(rsi) else 50)
            signals["mom14"].append(m14 if pd.notna(m14) else 0)
            signals["mom7"].append(m7 if pd.notna(m7) else 0)
            signals["range_pos"].append(range_p if pd.notna(range_p) else 0.5)

    labels = np.array(labels)
    results = {}
    for sig_name, sig_vals in signals.items():
        sv = np.array(sig_vals)
        try:
            auc = roc_auc_score(labels, sv)
        except Exception:
            auc = np.nan
        results[sig_name] = round(float(auc), 4)

    # Date-block permutation null: shuffle dates (7-day blocks), recompute AUC
    # Use mom14 as the representative signal
    null_aucs = []
    rng = np.random.default_rng(42)
    sig14 = np.array(signals["mom14"])
    n = len(labels)
    block_size = len(C.columns)   # per-day block
    n_blocks = n // block_size
    # Permute day blocks (not individual rows -- preserves cross-sectional structure)
    n_perms = 200
    for _ in range(n_perms):
        perm = rng.permutation(n_blocks)
        perm_sig = np.concatenate([sig14[p * block_size: (p+1) * block_size] for p in perm])
        perm_sig = perm_sig[:n]
        try:
            auc = roc_auc_score(labels[:len(perm_sig)], perm_sig)
        except Exception:
            auc = 0.5
        null_aucs.append(auc)

    p_value = float(np.mean(np.array(null_aucs) >= results["mom14"]))

    return {
        "signal_aucs": results,
        "n_obs": int(len(labels)),
        "n_pos": int(labels.sum()),
        "mom14_p_vs_date_block_null": round(p_value, 4),
        "null_auc_mean": round(float(np.mean(null_aucs)), 4),
        "null_auc_p95": round(float(np.percentile(null_aucs, 95)), 4),
        "best_signal": max(results, key=results.get),
        "best_auc": max(results.values()),
    }


# ===========================================================================
# SECTION 8: BEAR SURVIVAL ANALYSIS
# Show how the circuit-breaker preserves capital in the 2022 bear and later DDs.
# ===========================================================================

def bear_survival_analysis(W_engines: dict, W_oracle: pd.DataFrame,
                            ind: dict) -> dict:
    """Compound return over known drawdown periods for each engine variant."""
    C = ind["C"]

    # Key drawdown periods
    periods = {
        "bear_2022": ("2022-01-01", "2022-12-31"),
        "bull_2023": ("2023-01-01", "2023-12-31"),
        "correction_2024q1": ("2024-03-01", "2024-07-01"),
        "bull_2024q4": ("2024-10-01", "2025-01-01"),
        "correction_2025": ("2025-01-01", "2025-06-01"),
    }

    # EW buy-hold baseline
    bh_W = ref.bh_ew_weights(ind)
    bh_b = ref.book_daily_returns(bh_W, ind)

    results = {}
    for pname, (ps, pe) in periods.items():
        mask = (C.index >= pd.Timestamp(ps)) & (C.index < pd.Timestamp(pe))
        if mask.sum() < 5:
            continue
        bh_ret = float((1 + bh_b[mask]).prod() - 1) * 100
        row = {"bh": round(bh_ret, 1)}
        for eng_name, W in W_engines.items():
            b = ref.book_daily_returns(W, ind)
            ret = float((1 + b[mask]).prod() - 1) * 100
            row[eng_name] = round(ret, 1)
        results[pname] = row

    return results


# ===========================================================================
# MAIN: BUILD + RUN EVERYTHING
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("MOVE-CAPTURE ORACLE DECOMPOSITION")
    print(f"OOS: {OOS_START} -> {OOS_END} | n_slices={N_SLICES} | seeds={SEEDS}")
    print("Objective: CAPTURE-RATE = realized_net / oracle_available_move")
    print("=" * 80)

    print("\n[1] Loading data...")
    ind = lab.load(DATA_START, OOS_END)
    C = ind["C"]
    print(f"  Assets: {list(C.columns)}")
    print(f"  Date range: {C.index[0].date()} -> {C.index[-1].date()} ({len(C)} days)")

    # ---------------------------------------------------
    # Section 1: Diagnose the SMA200 gate failure
    # ---------------------------------------------------
    print("\n[2] Diagnosing SMA200 per-asset gate failure...")
    gate_diag = diagnose_sma200_gate(ind)
    print(f"  {gate_diag['diagnosis']}")

    # ---------------------------------------------------
    # Section 2: Build Oracle (ceiling) + causal engines
    # ---------------------------------------------------
    print("\n[3] Building oracle (top-3 fwd-7d movers -- CEILING, uses future data)...")
    W_oracle_k3 = build_oracle_weights(ind, K=3)
    W_oracle_k1 = build_oracle_weights(ind, K=1)

    print("\n[4] Building causal ungated mover engines...")
    # Variant A: ungated, NO circuit-breaker (to isolate mover benefit)
    W_ungated_bare = build_ungated_mover_weights(ind, K=3, rebal=7, use_circuit_breaker=False)
    # Variant B: ungated WITH market circuit-breaker (the proposed fix)
    W_ungated_cb = build_ungated_mover_weights(ind, K=3, rebal=7, use_circuit_breaker=True)
    # Variant C: ungated top-1 (maximum concentration)
    W_ungated_top1 = build_ungated_mover_weights(ind, K=1, rebal=7, use_circuit_breaker=True)
    # Variant D: ungated top-5 (diversified)
    W_ungated_top5 = build_ungated_mover_weights(ind, K=5, rebal=7, use_circuit_breaker=True)
    # Variant E: carry-with-stop (asymmetric exit)
    W_carry_stop = build_ungated_mover_weights_with_carry(ind, K=3, rebal=7)
    # Gated router (the baseline being replaced)
    import strat.adaptive_meta_engine as ame
    train_mask = C.index < pd.Timestamp(OOS_START)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    W_router = ame.build_weight_matrix(ind, vthr)

    engines = {
        "oracle_top3": W_oracle_k3,
        "oracle_top1": W_oracle_k1,
        "ungated_bare_top3": W_ungated_bare,
        "ungated_cb_top3": W_ungated_cb,
        "ungated_cb_top1": W_ungated_top1,
        "ungated_cb_top5": W_ungated_top5,
        "carry_stop_top3": W_carry_stop,
        "gated_router": W_router,
    }

    # ---------------------------------------------------
    # Section 3: Capture rates
    # ---------------------------------------------------
    print("\n[5] Computing capture rates (non-overlapping 7d windows in OOS)...")
    bh_W = ref.bh_ew_weights(ind)

    capture_results = {}
    for name, W in engines.items():
        cap = compute_capture_rate(W, W_oracle_k3, ind, OOS_START, OOS_END)
        capture_results[name] = cap
        print(f"  {name:25}: pooled_capture={cap['pooled_capture_rate']:.4f} "
              f"median={cap['median_capture_rate']} "
              f"eng_mean_7d={cap['engine_mean_7d']}% "
              f"oracle_mean_7d={cap['oracle_mean_7d']}%")

    # ---------------------------------------------------
    # Section 4: Random-slice profitability (referee harness)
    # ---------------------------------------------------
    print("\n[6] Random-slice profitability (referee harness, n=500 per seed)...")
    bh_b = ref.book_daily_returns(bh_W, ind)
    slice_results = {}
    for name in ["oracle_top3", "ungated_bare_top3", "ungated_cb_top3", "ungated_cb_top1",
                  "ungated_cb_top5", "carry_stop_top3", "gated_router"]:
        W = engines[name]
        b = ref.book_daily_returns(W, ind)
        prs, mns, bws = [], [], []
        for seed in SEEDS:
            s = ref.slice_stats(b, bh_b, OOS_START, OOS_END, N_SLICES, 7, seed)
            prs.append(s["pos_rate"]); mns.append(s["mean_pct"]); bws.append(s["beat_bh_pct"])
        slice_results[name] = {
            "pos_rate": round(float(np.mean(prs)), 1),
            "pos_rate_seeds": prs,
            "mean_pct": round(float(np.mean(mns)), 2),
            "beat_bh": round(float(np.mean(bws)), 1),
        }
        print(f"  {name:25}: pos_rate={slice_results[name]['pos_rate']}% "
              f"mean={slice_results[name]['mean_pct']}% "
              f"beat_bh={slice_results[name]['beat_bh']}%")

    # BH baseline
    bh_prs = [ref.bh_slice_stats(bh_b, OOS_START, OOS_END, N_SLICES, 7, s)["pos_rate"] for s in SEEDS]
    bh_mns = [ref.bh_slice_stats(bh_b, OOS_START, OOS_END, N_SLICES, 7, s)["mean_pct"] for s in SEEDS]
    bh_stats = {"pos_rate": round(float(np.mean(bh_prs)), 1), "mean_pct": round(float(np.mean(bh_mns)), 2)}
    print(f"  {'EW_BH':25}: pos_rate={bh_stats['pos_rate']}% mean={bh_stats['mean_pct']}%")

    # ---------------------------------------------------
    # Section 5: Regime-stratified capture
    # ---------------------------------------------------
    print("\n[7] Regime-stratified capture rates (BULL/BEAR/CHOP)...")
    regime_cap = {}
    for name in ["ungated_bare_top3", "ungated_cb_top3", "gated_router"]:
        rc = regime_stratified_capture(engines[name], W_oracle_k3, ind, OOS_START, OOS_END)
        regime_cap[name] = rc
        print(f"  {name}:")
        for r, rv in rc.items():
            if rv.get("n", 0) == 0:
                continue
            print(f"    {r:8}: n={rv['n']:3} pooled_capture={rv.get('pooled_capture',0):.3f} "
                  f"eng_mean={rv.get('engine_mean_7d_pct',0):.2f}% oracle_mean={rv.get('oracle_mean_7d_pct',0):.2f}% "
                  f"eng_pos_rate={rv.get('engine_pos_rate',0):.1f}%")

    # ---------------------------------------------------
    # Section 6: Reverse-engineer oracle signal
    # ---------------------------------------------------
    print("\n[8] Reverse-engineering oracle: which causal signal best predicts top-3 membership?")
    rev_eng = reverse_engineer_oracle_signal(ind, K=3)
    print(f"  n_obs={rev_eng['n_obs']} n_pos={rev_eng['n_pos']} (base_rate={100*rev_eng['n_pos']/rev_eng['n_obs']:.1f}%)")
    print(f"  Signal AUCs (OOS, predicting oracle top-3 membership):")
    for sig, auc in sorted(rev_eng["signal_aucs"].items(), key=lambda x: -x[1]):
        print(f"    {sig:20}: AUC={auc:.4f}")
    print(f"  Best signal: {rev_eng['best_signal']} AUC={rev_eng['best_auc']:.4f}")
    print(f"  mom14 AUC vs date-block null: p={rev_eng['mom14_p_vs_date_block_null']:.4f} "
          f"(null_mean={rev_eng['null_auc_mean']:.4f}, null_p95={rev_eng['null_auc_p95']:.4f})")

    # ---------------------------------------------------
    # Section 7: Bear survival
    # ---------------------------------------------------
    print("\n[9] Bear survival analysis (compound returns by period)...")
    survival = bear_survival_analysis(
        {k: engines[k] for k in ["ungated_bare_top3", "ungated_cb_top3", "gated_router"]},
        W_oracle_k3, ind
    )
    header = f"  {'Period':28} | {'BH':>7} | {'ungated_bare':>12} | {'ungated_cb':>10} | {'gated_rtr':>9}"
    print(header); print("  " + "-" * 78)
    for pname, row in survival.items():
        bh_v = row.get("bh", 0)
        ub = row.get("ungated_bare_top3", 0)
        uc = row.get("ungated_cb_top3", 0)
        gr = row.get("gated_router", 0)
        print(f"  {pname:28} | {bh_v:>7.1f}% | {ub:>12.1f}% | {uc:>10.1f}% | {gr:>9.1f}%")

    # ---------------------------------------------------
    # VERDICT TABLE
    # ---------------------------------------------------
    print("\n" + "=" * 80)
    print("VERDICT TABLE")
    print("=" * 80)
    print(f"\n  Oracle (top-3 ceiling): pooled_capture = 1.0 by definition")
    print(f"  (Oracle captures 100% of its own available move -- the ceiling)")
    print()
    print(f"  {'Engine':25} | {'Capture%':>8} | {'Pos-rate%':>10} | {'Mean-7d%':>9} | {'Beat-BH%':>9}")
    print(f"  {'-'*75}")
    for name in ["oracle_top3", "ungated_bare_top3", "ungated_cb_top3",
                  "ungated_cb_top1", "ungated_cb_top5", "carry_stop_top3", "gated_router"]:
        cap = capture_results[name]["pooled_capture_rate"]
        if name in slice_results:
            pr = slice_results[name]["pos_rate"]
            mn = slice_results[name]["mean_pct"]
            bw = slice_results[name]["beat_bh"]
        else:
            pr = mn = bw = "-"
        print(f"  {name:25} | {cap:>8.3f} | {pr:>10} | {mn:>9} | {bw:>9}")

    print(f"\n  EW_BH baseline: pos_rate={bh_stats['pos_rate']}% mean={bh_stats['mean_pct']}%")

    # ---------------------------------------------------
    # CAPTURE GAP: what fraction of oracle does causal engine achieve?
    # ---------------------------------------------------
    oracle_mean_7d = capture_results["oracle_top3"]["oracle_mean_7d"]
    ungated_cb_mean_7d = capture_results["ungated_cb_top3"]["engine_mean_7d"]
    gap_pct = round((oracle_mean_7d - ungated_cb_mean_7d) / (abs(oracle_mean_7d) + 1e-6) * 100, 1)
    abs_capture = capture_results["ungated_cb_top3"]["pooled_capture_rate"]

    print(f"\n  CAPTURE GAP (honest ceiling):")
    print(f"    Oracle top-3 average 7d return: {oracle_mean_7d}%")
    print(f"    Causal ungated+CB top-3 avg 7d: {ungated_cb_mean_7d}%")
    print(f"    Pooled capture (engine/oracle): {abs_capture:.1%}")
    print(f"    Gap: {gap_pct}% of oracle unavailable to causal engine")

    # ---------------------------------------------------
    # Save artifact
    # ---------------------------------------------------
    t1 = time.time()
    out = {
        "meta": {
            "script": "mover_capture_oracle.py",
            "oos": [OOS_START, OOS_END],
            "n_slices": N_SLICES,
            "seeds": SEEDS,
            "runtime_s": round(t1 - t0, 1),
        },
        "gate_diagnosis": gate_diag,
        "capture_rates": {k: {kk: vv for kk, vv in v.items() if kk != "windows"}
                          for k, v in capture_results.items()},
        "slice_stats": slice_results,
        "bh_baseline": bh_stats,
        "regime_capture": regime_cap,
        "reverse_engineer": rev_eng,
        "bear_survival": survival,
        "verdict": {
            "oracle_mean_7d_pct": oracle_mean_7d,
            "best_causal_mean_7d_pct": ungated_cb_mean_7d,
            "pooled_capture_rate_causal_cb": abs_capture,
            "gap_pct_of_oracle": gap_pct,
            "best_signal_for_oracle": rev_eng["best_signal"],
            "best_signal_auc": rev_eng["best_auc"],
            "mom14_auc": rev_eng["signal_aucs"].get("mom14"),
            "mom14_significant": rev_eng["mom14_p_vs_date_block_null"] < 0.05,
        },
    }

    out_dir = ROOT.parent / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mover_capture_oracle_decomp.json"
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    tmp.replace(out_path)
    print(f"\n  Artifact: {out_path}")
    print(f"  Runtime: {round(t1-t0,1)}s")
    return out


if __name__ == "__main__":
    main()
