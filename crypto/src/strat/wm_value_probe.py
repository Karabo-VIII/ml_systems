"""src/strat/wm_value_probe.py -- Does a forecaster (WM) add held-out value? + GATE-A.

TWO LAYERS IN THIS FILE
=======================
(L1) THE PROBE (original): does the already-trained V1.1 WM add held-out COMPOUND-return
     value as an INPUT to a trading decision vs a no-WM control? YES/NO with real numbers.
     This is clause-2's compute engine and needs the trained V1.1 checkpoint (GPU).
       Run:  python src/strat/wm_value_probe.py [--assets ...] [--seeds N]

(L2) GATE-A (docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S2.4): the exact bar a FORECASTER
     must clear before ANY A1 (WM-consuming agent) is wired onto it. Five hard, pre-registered,
     mechanically-checked clauses; FAIL ANY clause => A1 forbidden (the WM may still serve as a
     lower-stakes sizer/filter). All numbers measured on OOS/UNSEEN, never train.
       Run:  python src/strat/wm_value_probe.py --gate-a            (RWYB two-sided, CPU/synthetic)
             python src/strat/wm_value_probe.py --gate-a --forecaster-id V1.1   (real, when V1.1 lands)

THE 5 GATE-A CLAUSES (S2.4)
---------------------------
  1. GENUINE LEARNING (anti-memorization): ShIC(h=1) > 0.015 AND ShIC/contig-IC > 0.3,
     and -- tightened for A1 -- genuine at EVERY horizon the planner queries, h in {1,4,16}.
     Rejects ic1=+0.21/ShIC=0.000 memorizers and the voladj IC=0.10/raw=0.017 shortcut.
  2. HELD-OUT COMPOUND over the right null: the WM-driven predict-then-rule policy beats
     BOTH buy-and-hold AND a regime-matched + cost-matched RANDOM-ENTRY null, >=10 seeds,
     positive margin. (Reuses strat.firewall.random_entry_null + the L1 probe engine.)
  3. SEED-ROBUSTNESS: >=8/10 seeds clear clauses 1-2.
  4. REGIME-COVERAGE: clauses 1-2 hold separately in trending / mean-reverting / high-vol,
     not just pooled (a bull-only-genuine WM is a beta proxy a planner will over-trade).
  5. COST-HONESTY (REPORTING-ONLY NOTE, not a discriminating clause): TAKER 0.0024 round-trip is
     THE gate value; maker 0.0010 is a labeled sensitivity, never the headline (D43: real p_fill
     0.21-0.40). The cost discipline is ENFORCED inside clause 2 (which runs at taker); clause 5
     only DOCUMENTS the cost basis (cost_basis_is_taker) + surfaces the maker sensitivity. It does
     not emit PASS/FAIL and does not gate the verdict.

VERDICT RULE: the four DISCRIMINATING clauses (1-4) ALL PASS => "ELIGIBLE A1 SUBSTRATE" (one-way
ratchet, monotonic floor). Any FAIL => "A1 FORBIDDEN". Clause 5 is a reporting-only cost NOTE and
is NOT part of the verdict set. The verdict is WRITTEN to runs/registry/forecasters.json.

IC / ShIC are used here ONLY as the anti-memorization DIAGNOSTIC (clause 1) -- IC remains
BANNED as a primary/objective metric. The MONEY axis (clauses 2-4) is the compound return of
a SETUP across a multi-candle MOVE; per-bar predictability is never the objective.

RWYB (two-sided, CPU, no GPU, no V1.1 checkpoint): a genuinely-predictive SYNTHETIC forecaster
PASSES Gate-A; a SHUFFLED / zero-signal one FAILS. This verifies the gate LOGIC + its two-
sidedness; the real V1.1 run plugs the trained producer in when it lands (--forecaster-id V1.1).

UNSEEN SEGMENT (L1): 90% + 400-bar purge_gap to end of data. NEVER touched in training.

HARD CONSTRAINTS (inherited): LONG-ONLY, SPOT, LEVERAGE=1, TAKER 0.24% round-trip honest cost,
walk-forward purge, no look-ahead, objective = WEALTH (compound %), SETUP/MOVE not per-bar.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_V11 = _ROOT / "wm" / "v1" / "v1_1_training"
if str(_V11) not in sys.path:
    sys.path.insert(0, str(_V11))

from strat.setup_harness import SetupHarness, ExitPolicy
from strat.firewall import random_entry_null
from wealth_bot.harness import WindowSpec

__contract__ = {
    "kind": "forecaster_gate_a",
    "version": "1.0",
    "spec_source": "docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S2.4",
    "inputs": [
        "L1: trained V1.1 WMEntryProducer + chimera UNSEEN slice (real, GPU)",
        "L2 Gate-A: a Forecaster implementing the 3-method ShIC contract -- predict(df)->{h:pred}, "
        "features()->[n,d] (the matrix predict consumed), predict_from_features(feats,h)->[m] "
        "(re-runnable on shuffled rows). make_v11_forecaster_factory is the WORKING V1.1 adapter "
        "over WMEntryProducer.run_inference (runs CPU with device='cpu'); _SyntheticForecaster is "
        "the reference adapter for the two-sided RWYB.",
    ],
    "outputs": [
        "per-clause PASS/FAIL (5 clauses) + overall {ELIGIBLE A1 SUBSTRATE | A1 FORBIDDEN}",
        "written to runs/registry/forecasters.json (TRUTH axis -- never ranked by compound)",
    ],
    "invariants": [
        "IC/ShIC used ONLY as the clause-1 anti-memorization DIAGNOSTIC; IC BANNED as objective",
        "clause-1 ShIC genuine at EVERY planner horizon h in {1,4,16}, not just h=1",
        "clause-2 beats BOTH buy-hold AND regime+cost-matched random-entry null by a PRE-REGISTERED "
        "ECONOMIC margin (fixed pp floor + a fraction of the null's seed dispersion), not a bare +epsilon",
        "clause-3 >=8/10 seeds clear clauses 1-2",
        "clause-4 clauses 1-2 hold PER-REGIME (trending/mean-reverting/high-vol), not just pooled",
        "clause-5 is REPORTING-ONLY (cost NOTE): asserts cost_basis_is_taker + surfaces maker as a "
        "labeled sensitivity; it emits NO pass/fail and is NOT in the verdict set (taker is enforced "
        "inside clause 2)",
        "the FOUR discriminating clauses 1-4 ALL PASS => eligible A1 substrate (monotonic ratchet); "
        "any FAIL => A1 forbidden (clause 5 is not a gate)",
        "MONEY axis = compound of SETUP across a MOVE (multi-candle); never per-bar predictability",
        "walk-forward purge; entry fill = next-bar open (Pattern-T safe); no look-ahead",
        "two-sided RWYB: genuine synthetic forecaster PASSES, shuffled/zero-signal FAILS (CPU/synthetic)",
    ],
}

# Assets to evaluate
DEFAULT_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# Standard 50/20/20/10 split anchors -- MUST match training split
# These date anchors are used for WINDOW LABELLING ONLY (to confirm UNSEEN coverage).
# The actual UNSEEN slice is taken at index 90%+purge_gap (the training split convention).
WIN = WindowSpec(
    train_end="2024-05-15",
    val_end="2025-03-15",
    oos_end="2025-12-31",
    unseen_end="2026-06-01",
)

# Cost model
TAKER_RT = 0.0024  # 0.24% round-trip taker

# Purge gap (must match training: hurst/zscore window = 400 bars)
PURGE_GAP_BARS = 400


# ---------------------------------------------------------------------------
# Data loading: chimera -> polars + pandas OHLC slice
# ---------------------------------------------------------------------------

def load_unseen_data(sym: str):
    """Load chimera data, slice to UNSEEN segment, return (df_polars, df_pandas, window, unseen_start_idx).

    df_pandas is needed by SetupHarness (needs date/open/high/low/close + atr14).
    df_polars is the full chimera (for WMEntryProducer feature extraction).

    CRITICAL: WindowSpec date boundaries are derived from the ACTUAL index-based training split
    (50/20/20/10 with 400-bar purge gap). This ensures the SetupHarness labels bars as "UNSEEN"
    in exact alignment with what the training script never touched. The static WIN constant above
    is NOT used for per-asset evaluation -- date boundaries vary per-asset because dollar bars
    arrive at different rates.
    """
    import datetime
    import polars as pl
    from pipeline.chimera_loader import ChimeraLoader
    from pipeline.data_integrity import selective_drop_nulls
    from settings import FEATURE_LIST, REWARD_HORIZONS

    loader = ChimeraLoader()
    df_full = loader.load(sym, cadence="dollar")
    df_full = selective_drop_nulls(df_full, FEATURE_LIST, REWARD_HORIZONS, sym)

    n = len(df_full)
    ts_col = "timestamp" if "timestamp" in df_full.columns else "date"

    # Index-based training split boundaries (must match train_world_model.py)
    train_end_idx = int(n * 0.50)
    val_end_idx = int(n * 0.70)
    oos_end_idx = int(n * 0.90)
    unseen_start_idx = oos_end_idx + PURGE_GAP_BARS

    # Derive date strings from actual timestamps at split boundaries
    def ts_to_date(ts_ms: int) -> str:
        return datetime.datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")

    train_end_str = ts_to_date(int(df_full[ts_col][train_end_idx]))
    val_end_str = ts_to_date(int(df_full[ts_col][val_end_idx]))
    oos_end_str = ts_to_date(int(df_full[ts_col][oos_end_idx]))

    window = WindowSpec(
        train_end=train_end_str,
        val_end=val_end_str,
        oos_end=oos_end_str,
        unseen_end="2030-12-31",  # far future cap
    )

    df_unseen_pl = df_full.slice(unseen_start_idx, n - unseen_start_idx)
    n_unseen = len(df_unseen_pl)
    print(f"  {sym}: total={n:,}, unseen_start={unseen_start_idx}, n_unseen={n_unseen:,} "
          f"(oos_end={oos_end_str})")

    # Build pandas OHLC for SetupHarness
    df_pd = df_unseen_pl.select([ts_col, "open", "high", "low", "close"]).to_pandas()
    df_pd = df_pd.rename(columns={ts_col: "date"})
    df_pd["date"] = pd.to_datetime(df_pd["date"], unit="ms")
    for c in ("open", "high", "low", "close"):
        df_pd[c] = df_pd[c].astype(float)

    # Add past-only ATR14 for chandelier exit
    h, l, pc = df_pd["high"], df_pd["low"], df_pd["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df_pd["atr14"] = tr.rolling(14, min_periods=7).mean()
    df_pd = df_pd.reset_index(drop=True)

    return df_unseen_pl, df_pd, window, unseen_start_idx


# ---------------------------------------------------------------------------
# Buy-and-hold compound on the UNSEEN slice
# ---------------------------------------------------------------------------

def buy_hold_compound(df_pd: pd.DataFrame) -> float:
    """Log-sum of bar returns (fully invested, no cost)."""
    c = df_pd["close"].to_numpy(float)
    bar_rets = np.diff(c) / c[:-1]
    compound = (1.0 + bar_rets).prod() - 1.0
    return float(compound) * 100.0  # in %


# ---------------------------------------------------------------------------
# Run harness for one entry column
# ---------------------------------------------------------------------------

def run_harness(df_pd: pd.DataFrame, entry: np.ndarray, label: str, window=None) -> dict:
    """Run SetupHarness on entry array and return UNSEEN compound + trade stats.

    window: WindowSpec with date-aligned split boundaries (derived from index-based training
            split per asset). If None, falls back to the global WIN constant (for quick tests only).

    EXIT POLICY (dollar-bar calibrated):
    Dollar bars are ~0.09% ATR each (BTC at $117k). An ATR-based trail needs ~22x multiplier
    to achieve 2% trail width. Instead we use a PERCENTAGE trail (2%) with 5% hard stop and
    a 500-bar time cap (~8-16 hours at typical dollar-bar cadence). This is the correct
    policy for dollar bars -- ATR-3x was designed for daily/4h bars.
    """
    assert len(entry) == len(df_pd), f"entry length {len(entry)} != df length {len(df_pd)}"
    assert entry.dtype == bool or set(np.unique(entry)).issubset({0, 1, True, False}), \
        "entry must be boolean"

    df_w = df_pd.copy()
    df_w["entry"] = entry.astype(int)

    policy = ExitPolicy(
        trail_pct=0.02,      # 2% trailing stop (calibrated for dollar bars)
        sl_pct=0.05,         # 5% hard backstop
        max_hold_bars=500,   # ~8-16 hours at dollar-bar cadence
    )

    win = window if window is not None else WIN
    harness = SetupHarness(
        df=df_w,
        entry_col="entry",
        policy=policy,
        windows=win,
        cost_rt=TAKER_RT,
    )
    results = harness.run()

    stats = results.window_stats.get("UNSEEN")
    if stats is None:
        avail = list(results.window_stats.keys())
        if avail:
            stats = results.window_stats[avail[-1]]
        else:
            return {"label": label, "compound_pct": 0.0, "n_trades": 0,
                    "max_dd_pct": 0.0, "win_rate": 0.0, "coverage": 0.0}

    n_entry_bars = int(entry.sum())
    return {
        "label": label,
        "compound_pct": float(stats.compound_pct),
        "n_trades": int(stats.n_trades),
        "max_dd_pct": float(stats.max_dd_pct),
        "win_rate": float(stats.win_rate),
        "coverage": n_entry_bars / max(len(entry), 1),
    }


# ---------------------------------------------------------------------------
# Random-entry null (regime-matched, 10 seeds)
# ---------------------------------------------------------------------------

def run_random_null(df_pd: pd.DataFrame, wm_entry: np.ndarray, n_seeds: int = 10, window=None) -> dict:
    """Random entries with same density + same exit policy as WM (regime-matched).

    Draws random LONG entries from ALL bars in the UNSEEN window, matching WM entry COUNT.
    This is the unbiased null: if WM regime gate provides value, it will show up as WM
    outperforming this random null even when the null can enter anywhere.

    Note: we use the SAME ENTRY COUNT as WM entries to control for entry density.
    """
    n = len(df_pd)
    n_wm_entries = int(wm_entry.sum())

    if n_wm_entries == 0:
        return {"label": "random_null_mean", "compound_pct": 0.0, "n_trades": 0,
                "max_dd_pct": 0.0, "win_rate": 0.0, "n_seeds": n_seeds, "compound_std": 0.0}

    # Eligible bars: skip first 100 (ATR warmup) and last 501 (need room for exit)
    eligible = np.arange(100, n - 501)

    compounds = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # Sample same number of entries from eligible bars (with replacement is OK for high-freq)
        n_sample = min(n_wm_entries, len(eligible))
        chosen = rng.choice(eligible, size=n_sample, replace=False)
        rand_entry = np.zeros(n, dtype=bool)
        rand_entry[chosen] = True

        res = run_harness(df_pd, rand_entry, label=f"random_s{seed}", window=window)
        compounds.append(res["compound_pct"])

    mean_c = float(np.mean(compounds))
    std_c = float(np.std(compounds))
    return {
        "label": "random_null_mean",
        "compound_pct": mean_c,
        "compound_std": std_c,
        "compound_all": compounds,
        "n_seeds": n_seeds,
        "n_wm_entries": n_wm_entries,
    }


# ===========================================================================
# GATE-A (docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S2.4)
# The exact bar a FORECASTER clears before ANY A1 is wired onto it.
# ===========================================================================
#
# A "Forecaster" here is the minimal abstraction Gate-A needs: a callable that, given an
# OHLC frame, returns a dict {h: per-bar predicted forward return at horizon h} for the
# planner horizons h in {1, 4, 16}. The trained V1.1 WMEntryProducer is ONE such forecaster
# (see make_v11_forecaster); a synthetic genuine/null forecaster is another (RWYB). Gate-A is
# agnostic to which -- it only consumes predicted-return arrays + realized returns.
#
# Pre-registered thresholds (frozen; never tuned on UNSEEN):
GATE_A = {
    "shic_h1_min": 0.015,          # clause 1: ShIC(h=1) floor
    "shic_over_contig_min": 0.30,  # clause 1: ShIC / contiguous-IC ratio floor (anti-memorization)
    "planner_horizons": [1, 4, 16],# clause 1: genuine at EVERY horizon the planner queries
    "n_seeds": 10,                 # clauses 2-3
    "seed_pass_min": 8,            # clause 3: >=8/10 seeds clear clauses 1-2
    # clause 2 economic-margin floor (pre-registered; never tuned on UNSEEN). The WM-policy
    # held-out compound must beat EACH null by MORE than noise -- not merely a strictly-positive
    # margin (which a coin-flip clears half the time). The bar is the MAX of:
    #   (a) a fixed floor compound_margin_floor_pp (pp), AND
    #   (b) compound_margin_null_disp_frac * (the random-null's own seed-to-seed dispersion, pp)
    #       -- i.e. the margin must exceed a fraction of the null's noise band, so a within-noise
    #       edge does NOT pass. (a) guards the buy-hold side (which has no seed dispersion); (b)
    #       scales the random-null side by the actual sampling noise that null exhibits.
    "compound_margin_floor_pp": 2.0,        # fixed pp floor vs BOTH nulls (~2pp held-out compound)
    "compound_margin_null_disp_frac": 0.50, # + must beat the random null by >=0.5 * its seed-std
    "taker_cost_rt": 0.0024,       # clause 5: THE gate value
    "maker_cost_rt": 0.0010,       # clause 5: labeled sensitivity ONLY (never the headline)
    "min_regime_bars": 40,         # clause 4: a regime slice needs enough bars to score
}


# --- clause-1 primitives: IC / ShIC (the ANTI-MEMORIZATION diagnostic ONLY) ----------------

def _ic(pred: np.ndarray, realized: np.ndarray) -> float:
    """Spearman rank IC between predicted and realized forward return. NaN-safe.

    IC is used here PURELY as the clause-1 anti-memorization diagnostic (is the learning
    genuine, or did the model memorize?). It is NEVER a primary/objective metric -- the MONEY
    axis is the compound return of the SETUP/MOVE (clauses 2-4)."""
    p, r = np.asarray(pred, float), np.asarray(realized, float)
    m = np.isfinite(p) & np.isfinite(r)
    if m.sum() < 8:
        return 0.0
    pr = pd.Series(p[m]).rank().to_numpy()
    rr = pd.Series(r[m]).rank().to_numpy()
    if pr.std() < 1e-12 or rr.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(pr, rr)[0, 1])


def _global_shuffled_ic(forecaster, h: int, sel_idx: np.ndarray, realized_h: np.ndarray,
                        n_seeds: int = 5) -> float:
    """Global shuffled IC at horizon h -- the project's CANONICAL anti-memorization metric
    (faithful to src/wm/v1/v1_1_training/validate_world.py::_compute_global_shuffled_ic).

    Semantics: GLOBALLY shuffle the bar indices, then RE-RUN the forecaster on the shuffled
    FEATURE rows, pairing each prediction with the matching shuffled target. Because the
    (feature, target) pairing is preserved by the shuffle, a forecaster that learned a genuine
    feature->return mapping RETAINS a positive IC (ShIC > 0); a forecaster that memorized
    TEMPORAL position collapses to ~0 (de-temporalization destroys its only signal). Reported as
    the SIGNED mean over seeds (NOT abs -- a noisy abs-mean has a positive floor that would falsely
    'pass' a memorizer). The forecaster must expose features() and predict_from_features(feats, h).

    sel_idx    : held-out bar indices (into the forecaster's feature matrix) with a valid target.
    realized_h : full-length realized forward-return array (NaN where no future)."""
    feats = forecaster.features()                         # [n, d] forecaster's input features
    ics = []
    for s in range(n_seeds):
        rng = np.random.default_rng(42 + s * 1000)
        perm = sel_idx.copy()
        rng.shuffle(perm)                                 # global shuffle of held-out indices
        shuf_feats = feats[perm]                          # features re-ordered...
        shuf_real = realized_h[perm]                      # ...target re-ordered the SAME way
        pred = forecaster.predict_from_features(shuf_feats, h)   # RE-RUN on shuffled features
        m = np.isfinite(pred) & np.isfinite(shuf_real)
        if m.sum() < 16:
            continue
        ics.append(_ic(pred[m], shuf_real[m]))
    return float(np.mean(ics)) if ics else 0.0


def _realized_fwd_return(closes: np.ndarray, h: int) -> np.ndarray:
    """Forward h-bar simple return, past-only-aligned: realized[t] = close[t+h]/close[t]-1.
    Trailing h entries (no future) are NaN and dropped by the IC mask. No look-ahead in the
    PREDICTION -- the prediction is the model's; this is just the realized target it is scored
    against, which legitimately uses the future (that is what 'realized' means)."""
    c = np.asarray(closes, float)
    out = np.full(c.shape, np.nan)
    if h < len(c):
        out[:-h] = c[h:] / c[:-h] - 1.0
    return out


def clause1_genuine_learning(forecaster, df_pd: pd.DataFrame, held_mask: np.ndarray,
                             seed: int = 0) -> dict:
    """CLAUSE 1 -- genuine learning at EVERY planner horizon h in {1,4,16}.

    PASS iff, at every h: ShIC > 0.015 AND ShIC/contig-IC > 0.30 (the project's signed-ShIC
    convention), measured on the held-out (OOS+UNSEEN) bars only. The ratio is the anti-
    memorization test: a memorizer has high contig-IC but ShIC ~ 0 -> ratio ~ 0 -> FAIL; a genuine
    feature->return learner retains positive IC under the global feature shuffle -> ratio ~ 1.
    We additionally require contig-IC > 0 (a forecaster with negative held-out IC has not learned
    a usable mapping, so a high ShIC/IC ratio off a tiny/negative IC is not 'genuine')."""
    closes = df_pd["close"].to_numpy(float)
    preds = forecaster.predict(df_pd)  # {h: per-bar predicted forward return}
    per_h = {}
    all_pass = True
    for h in GATE_A["planner_horizons"]:
        pred_h = np.asarray(preds[h], float)
        realized_h = _realized_fwd_return(closes, h)
        # held-out scope: held bars with a valid target AND a finite prediction
        sel = held_mask & np.isfinite(realized_h) & np.isfinite(pred_h)
        sel_idx = np.where(sel)[0]
        contig_ic = _ic(pred_h[sel], realized_h[sel])
        shic = _global_shuffled_ic(forecaster, h, sel_idx, realized_h)
        ratio = shic / contig_ic if abs(contig_ic) > 1e-9 else 0.0
        h_pass = bool(contig_ic > 0 and shic > GATE_A["shic_h1_min"]
                      and ratio > GATE_A["shic_over_contig_min"])
        per_h[h] = {"contig_ic": round(contig_ic, 4), "shic": round(shic, 4),
                    "shic_over_contig": round(ratio, 3), "pass": h_pass}
        all_pass = all_pass and h_pass
    return {"pass": all_pass, "per_horizon": per_h}


# --- clause-4 primitive: regime labeller (past-only) ---------------------------------------

def label_regimes(df_pd: pd.DataFrame) -> np.ndarray:
    """Past-only regime label per bar: 'trending' / 'mean_reverting' / 'high_vol'.

    Definitions (all from STRICTLY PRIOR bars -- no look-ahead):
      - high_vol       : trailing-50 realized vol in the top tercile of its own past distribution
      - trending       : |trailing-50 SMA slope| large relative to trailing vol (directional drift)
      - mean_reverting : everything else (choppy / range)
    A bar with insufficient history is labelled '' (excluded from per-regime scoring)."""
    c = df_pd["close"].to_numpy(float)
    n = len(c)
    lab = np.array([""] * n, dtype=object)
    w = 50
    rets = np.diff(c, prepend=c[0]) / np.maximum(np.roll(c, 1), 1e-12)
    rets[0] = 0.0
    vol = pd.Series(rets).rolling(w, min_periods=w).std().to_numpy()
    sma = pd.Series(c).rolling(w, min_periods=w).mean().to_numpy()
    slope = (sma - np.roll(sma, w)) / np.maximum(np.abs(np.roll(sma, w)), 1e-12)
    slope[:w] = np.nan
    # past-only vol tercile cut: expanding 80th pct of vol up to (not incl) bar t
    for t in range(n):
        if not np.isfinite(vol[t]) or not np.isfinite(slope[t]):
            continue
        past_vol = vol[max(0, t - 400):t]
        past_vol = past_vol[np.isfinite(past_vol)]
        hi_cut = np.percentile(past_vol, 80) if past_vol.size >= 20 else np.inf
        if vol[t] >= hi_cut:
            lab[t] = "high_vol"
        elif abs(slope[t]) > 1.5 * vol[t] * np.sqrt(w):
            lab[t] = "trending"
        else:
            lab[t] = "mean_reverting"
    return lab


# --- clause-2 primitive: predict-then-rule policy compound vs BOTH nulls --------------------

def _predict_then_rule_entry(forecaster, df_pd: pd.DataFrame, h: int = 4) -> np.ndarray:
    """The B0 predict-then-rule entry: LONG when E[r_h] - cost > 0 (a past-only boolean setup).
    This is the simplest WM-driven rule -- exactly the B0 baseline Gate-A clause 2 scores. The
    setup is confirmed at the CLOSE of bar t; SetupHarness fills at opens[t+1] (Pattern-T safe)."""
    preds = forecaster.predict(df_pd)
    pred_h = np.asarray(preds[h], float)
    entry = np.zeros(len(df_pd), dtype=bool)
    valid = np.isfinite(pred_h)
    entry[valid] = pred_h[valid] > GATE_A["taker_cost_rt"]
    return entry


def clause2_compound_over_nulls(forecaster, df_pd: pd.DataFrame, window: WindowSpec,
                                cost_rt: float, n_seeds: int = 10, seed_base: int = 0) -> dict:
    """CLAUSE 2 -- held-out compound beats BOTH buy-hold AND regime+cost-matched random null.

    The WM-driven predict-then-rule policy is scored on the SETUP/MOVE harness (multi-candle,
    Pattern-T-safe). 'Held-out' compound = OOS+UNSEEN. PASS iff held-out compound > buy-hold AND
    > the random-entry null mean, by >= margin, across >= n_seeds null seeds (positive margin).
    Reuses strat.firewall.random_entry_null (regime-matched on the setup-ON bars) + the L1 engine."""
    entry = _predict_then_rule_entry(forecaster, df_pd, h=4)
    df_w = df_pd.copy()
    df_w["entry"] = entry.astype(int)
    policy = ExitPolicy(tp_pct=0.08, sl_pct=0.05, max_hold_bars=16)  # a multi-candle MOVE policy
    h = SetupHarness(df_w, "entry", policy, window, cost_rt=cost_rt)
    res = h.run()
    held = ["OOS", "UNSEEN"]
    wm_held = float(sum(res.window_stats[w].compound_pct for w in held))

    # buy-hold over the held-out span (fully invested, no cost) for parity reference
    wlab = np.array([h._window_label(pd.Timestamp(t)) for t in df_pd["date"]])
    c = df_pd["close"].to_numpy(float)
    bh_held = 0.0
    for w in held:
        idx = np.where(wlab == w)[0]
        if idx.size > 1:
            seg = c[idx]
            bh_held += float((np.prod(1.0 + np.diff(seg) / seg[:-1]) - 1.0) * 100.0)

    # regime+cost-matched random-entry null (multi-seed), reusing the firewall verbatim
    null_held = []
    for s in range(n_seeds):
        fw = random_entry_null(h, n_books=120, seed=seed_base + s + 1, regime_matched=True)
        null_held.append(sum((fw["per_window"][w]["null_p50"] or 0.0) for w in held))
    null_mean = float(np.mean(null_held))
    null_disp = float(np.std(null_held)) if len(null_held) > 1 else 0.0  # null seed-to-seed noise (pp)

    # PRE-REGISTERED economic-margin floor (NOT strictly-positive): the WM-policy held-out
    # compound must beat EACH null by MORE than noise. Fixed floor guards the buy-hold side;
    # the null side additionally must clear a fraction of the null's own seed dispersion.
    floor_pp = GATE_A["compound_margin_floor_pp"]
    null_margin_req = max(floor_pp, GATE_A["compound_margin_null_disp_frac"] * null_disp)
    beats_bh = bool(wm_held > bh_held + floor_pp)
    beats_null = bool(wm_held > null_mean + null_margin_req)
    return {
        "pass": bool(beats_bh and beats_null),
        "wm_held_compound_pp": round(wm_held, 2),
        "buy_hold_held_pp": round(bh_held, 2),
        "random_null_held_mean_pp": round(null_mean, 2),
        "random_null_held_disp_pp": round(null_disp, 2),
        "beats_buy_hold": beats_bh,
        "beats_random_null": beats_null,
        "margin_vs_bh_pp": round(wm_held - bh_held, 2),
        "margin_vs_null_pp": round(wm_held - null_mean, 2),
        "bh_margin_floor_pp": round(floor_pp, 2),
        "null_margin_required_pp": round(null_margin_req, 2),
        "n_trades_held": int(sum(res.window_stats[w].n_trades for w in held)),
    }


def _clauses_12_for_seed(forecaster, df_pd, window, cost_rt, seed) -> dict:
    """Clauses 1+2 for ONE seed (a fresh forecaster instance per seed lets the synthetic null
    re-randomize; a real trained WM is deterministic so its per-seed variation is only the null's).

    The inner clause-2 null uses the CONFIGURED n_seeds (GATE_A['n_seeds'], 10) -- NOT a reduced
    count -- so each seed's clause-2 verdict carries the SAME null-sampling fidelity as the primary
    pooled clause-2; a smaller inner count made the per-seed verdicts noisier than the primary."""
    fc = forecaster(seed) if callable(forecaster) else forecaster
    held_mask = np.array([window_label_for(df_pd, i, window) in ("OOS", "UNSEEN")
                          for i in range(len(df_pd))])
    c1 = clause1_genuine_learning(fc, df_pd, held_mask, seed=seed)
    c2 = clause2_compound_over_nulls(fc, df_pd, window, cost_rt,
                                     n_seeds=GATE_A["n_seeds"], seed_base=seed * 10)
    return {"seed": seed, "clause1": c1, "clause2": c2,
            "pass": bool(c1["pass"] and c2["pass"])}


def window_label_for(df_pd: pd.DataFrame, i: int, window: WindowSpec) -> str:
    ts = pd.Timestamp(df_pd["date"].iloc[i])
    if ts < pd.Timestamp(window.train_end):
        return "TRAIN"
    if ts < pd.Timestamp(window.val_end):
        return "VAL"
    if ts < pd.Timestamp(window.oos_end):
        return "OOS"
    return "UNSEEN"


def run_gate_a(forecaster_factory, df_pd: pd.DataFrame, window: WindowSpec,
               forecaster_id: str, n_seeds: int = 10, verbose: bool = True) -> dict:
    """Run all 5 Gate-A clauses on a forecaster (factory: seed -> forecaster instance).

    Returns a per-clause PASS/FAIL dict + overall verdict. PURE LOGIC -- the same code path runs
    for a synthetic forecaster (RWYB) and the real V1.1 producer (when it lands)."""
    cost_rt = GATE_A["taker_cost_rt"]  # clause 5: THE gate value
    base_fc = forecaster_factory(0)

    # --- CLAUSE 1 + 2 (pooled, primary seed) ---
    held_mask = np.array([window_label_for(df_pd, i, window) in ("OOS", "UNSEEN")
                          for i in range(len(df_pd))])
    c1 = clause1_genuine_learning(base_fc, df_pd, held_mask, seed=0)
    c2 = clause2_compound_over_nulls(base_fc, df_pd, window, cost_rt, n_seeds=n_seeds, seed_base=0)

    # --- CLAUSE 3: >=8/10 seeds clear clauses 1-2 ---
    seed_results = [_clauses_12_for_seed(forecaster_factory, df_pd, window, cost_rt, s)
                    for s in range(n_seeds)]
    n_seed_pass = sum(1 for r in seed_results if r["pass"])
    c3_pass = bool(n_seed_pass >= GATE_A["seed_pass_min"])

    # --- CLAUSE 4: clauses 1-2 hold PER-REGIME (not just pooled) ---
    regimes = label_regimes(df_pd)
    per_regime = {}
    c4_pass = True
    for rg in ("trending", "mean_reverting", "high_vol"):
        rg_mask = held_mask & (regimes == rg)
        n_rg = int(rg_mask.sum())
        if n_rg < GATE_A["min_regime_bars"]:
            per_regime[rg] = {"n_bars": n_rg, "skipped": True,
                              "reason": "insufficient_held_out_bars", "pass": False}
            c4_pass = False
            continue
        # clause-1 restricted to this regime's held-out bars
        c1_rg = clause1_genuine_learning(base_fc, df_pd, rg_mask, seed=0)
        # clause-2 proxy per regime: predict-then-rule entries that fall in this regime must
        # out-compound a regime-restricted random draw (sign test on within-regime trade mean)
        c2_rg = _clause2_regime(base_fc, df_pd, rg_mask, cost_rt)
        rg_pass = bool(c1_rg["pass"] and c2_rg["pass"])
        per_regime[rg] = {"n_bars": n_rg, "clause1_pass": c1_rg["pass"],
                          "clause2_pass": c2_rg["pass"], "clause2_detail": c2_rg, "pass": rg_pass}
        c4_pass = c4_pass and rg_pass

    # --- CLAUSE 5: cost honesty -- taker is the gate; maker is a labeled sensitivity ---
    c2_maker = clause2_compound_over_nulls(base_fc, df_pd, window, GATE_A["maker_cost_rt"],
                                           n_seeds=n_seeds, seed_base=0)
    c5 = {
        "gate_cost_rt": GATE_A["taker_cost_rt"],
        "gate_cost_label": "TAKER (THE gate value)",
        "sensitivity_cost_rt": GATE_A["maker_cost_rt"],
        "sensitivity_cost_label": "MAKER (labeled sensitivity ONLY -- never the headline; D43 p_fill 0.21-0.40)",
        "taker_clause2_pass": c2["pass"],
        "maker_clause2_pass": c2_maker["pass"],
        "taker_wm_held_pp": c2["wm_held_compound_pp"],
        "maker_wm_held_pp": c2_maker["wm_held_compound_pp"],
        # CLAUSE 5 IS REPORTING-ONLY, NOT A DISCRIMINATOR. It does not return PASS/FAIL and does
        # NOT gate the overall verdict. Its job is the cost-honesty NOTE: assert the HEADLINE used
        # for the verdict was TAKER (cost_basis_is_taker) and surface MAKER purely as a labeled
        # sensitivity. The cost discipline is ENFORCED inside clause 2 (which runs at taker); a
        # separate pass=True here would have been a vacuous always-pass clause padding the verdict.
        "reporting_only": True,
        "cost_basis_is_taker": bool(cost_rt == GATE_A["taker_cost_rt"]),
    }

    # CLAUSE 5 is reporting-only (cost NOTE) -- it is NOT in the verdict set. The verdict is the
    # AND of the four DISCRIMINATING clauses (1-4); clause 5 only documents the cost basis.
    clauses = {1: c1["pass"], 2: c2["pass"], 3: c3_pass, 4: c4_pass}
    overall = all(clauses.values())
    verdict = "ELIGIBLE A1 SUBSTRATE" if overall else "A1 FORBIDDEN"

    result = {
        "forecaster_id": forecaster_id,
        "gate": "GATE_A",
        "spec": "docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S2.4",
        "cost_rt_gate_value": cost_rt,
        "clause1_genuine_learning": c1,
        "clause2_compound_over_nulls": c2,
        "clause3_seed_robustness": {"pass": c3_pass, "n_seed_pass": n_seed_pass,
                                    "n_seeds": n_seeds, "threshold": GATE_A["seed_pass_min"],
                                    "per_seed_pass": [r["pass"] for r in seed_results]},
        "clause4_regime_coverage": {"pass": c4_pass, "per_regime": per_regime},
        "clause5_cost_honesty": c5,
        "clauses_pass": clauses,
        "overall_pass": overall,
        "verdict": verdict,
        "thresholds": dict(GATE_A),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if verbose:
        _print_gate_a(result)
    return result


def _clause2_regime(forecaster, df_pd: pd.DataFrame, rg_mask: np.ndarray, cost_rt: float) -> dict:
    """Per-regime clause-2 proxy: within this regime's held-out bars, do the predict-then-rule
    LONG entries earn a positive net forward return that beats a regime-restricted RANDOM entry
    of the same count? (A pooled compound can hide a regime where the WM is a coin flip -- this
    catches the bull-only-genuine beta-proxy clause 4 exists to reject.)"""
    closes = df_pd["close"].to_numpy(float)
    realized = _realized_fwd_return(closes, 4)
    entry = _predict_then_rule_entry(forecaster, df_pd, h=4)
    sel = rg_mask & entry & np.isfinite(realized)
    n_wm = int(sel.sum())
    if n_wm < 5:
        return {"pass": False, "n_wm_entries": n_wm, "reason": "too_few_regime_entries"}
    wm_mean = float(np.mean(realized[sel] - cost_rt))
    # regime-restricted random null: same count, drawn from this regime's held-out bars
    pool = np.where(rg_mask & np.isfinite(realized))[0]
    rng = np.random.default_rng(7)
    null_means = []
    for _ in range(200):
        ch = rng.choice(pool, size=min(n_wm, pool.size), replace=False)
        null_means.append(float(np.mean(realized[ch] - cost_rt)))
    null_p50 = float(np.percentile(null_means, 50))
    return {"pass": bool(wm_mean > 0 and wm_mean > null_p50), "n_wm_entries": n_wm,
            "wm_mean_net": round(wm_mean, 5), "random_null_p50_net": round(null_p50, 5)}


def _print_gate_a(r: dict) -> None:
    print("\n" + "=" * 74)
    print(f"  GATE-A  forecaster={r['forecaster_id']}   ({r['spec']})")
    print("=" * 74)
    c1 = r["clause1_genuine_learning"]
    print(f"  [1] GENUINE LEARNING (ShIC>{GATE_A['shic_h1_min']} & ShIC/IC>{GATE_A['shic_over_contig_min']} "
          f"@ h in {GATE_A['planner_horizons']}): {'PASS' if c1['pass'] else 'FAIL'}")
    for h, d in c1["per_horizon"].items():
        print(f"        h={h:<3} contig_IC={d['contig_ic']:+.4f}  ShIC={d['shic']:.4f}  "
              f"ShIC/IC={d['shic_over_contig']:.3f}  -> {'pass' if d['pass'] else 'FAIL'}")
    c2 = r["clause2_compound_over_nulls"]
    print(f"  [2] HELD-OUT COMPOUND > buy-hold AND random null: {'PASS' if c2['pass'] else 'FAIL'}")
    print(f"        WM_held={c2['wm_held_compound_pp']:+.2f}%  B&H={c2['buy_hold_held_pp']:+.2f}%  "
          f"null_mean={c2['random_null_held_mean_pp']:+.2f}%  "
          f"(vs B&H {c2['margin_vs_bh_pp']:+.2f}pp, vs null {c2['margin_vs_null_pp']:+.2f}pp)")
    c3 = r["clause3_seed_robustness"]
    print(f"  [3] SEED-ROBUSTNESS (>={c3['threshold']}/{c3['n_seeds']} seeds clear 1+2): "
          f"{'PASS' if c3['pass'] else 'FAIL'}  ({c3['n_seed_pass']}/{c3['n_seeds']} passed)")
    c4 = r["clause4_regime_coverage"]
    print(f"  [4] REGIME-COVERAGE (1+2 hold per-regime, not just pooled): "
          f"{'PASS' if c4['pass'] else 'FAIL'}")
    for rg, d in c4["per_regime"].items():
        if d.get("skipped"):
            print(f"        {rg:<15} n={d['n_bars']:<5} SKIPPED ({d['reason']}) -> FAIL")
        else:
            print(f"        {rg:<15} n={d['n_bars']:<5} clause1={d['clause1_pass']} "
                  f"clause2={d['clause2_pass']} -> {'pass' if d['pass'] else 'FAIL'}")
    c5 = r["clause5_cost_honesty"]
    print(f"  [5] COST-HONESTY (NOTE, reporting-only -- NOT a verdict clause): "
          f"taker {c5['gate_cost_rt']} is the gate value; maker {c5['sensitivity_cost_rt']} "
          f"is a labeled sensitivity  [cost_basis_is_taker={c5['cost_basis_is_taker']}]")
    print(f"        taker held={c5['taker_wm_held_pp']:+.2f}%  [maker sensitivity held="
          f"{c5['maker_wm_held_pp']:+.2f}%]")
    print("-" * 74)
    print(f"  OVERALL: {r['verdict']}   "
          f"(clauses pass: {sorted(k for k, v in r['clauses_pass'].items() if v)} / "
          f"fail: {sorted(k for k, v in r['clauses_pass'].items() if not v)})")
    print("=" * 74)


# --- registry writer (TRUTH axis -- forecasters.json, never ranked by compound) -------------

def write_to_forecasters_registry(result: dict, registry_path: Path | None = None) -> Path:
    """Write the Gate-A verdict into runs/registry/forecasters.json (the F TRUTH leaderboard).

    Per docs S1.5: forecasters.json is keyed by held-out IC + ShIC (TRUTH -- did it learn or
    memorize?), NEVER by compound. We append/update a champion record with the Gate-A summary +
    the per-horizon ShIC diagnostics; agents.json (the MONEY leaderboard) is left untouched so
    IC-as-objective cannot re-enter through a shared board."""
    if registry_path is None:
        registry_path = _ROOT.parent / "runs" / "registry" / "forecasters.json"
    registry_path = Path(registry_path)
    reg = {"champions": []}
    if registry_path.exists():
        try:
            reg = json.loads(registry_path.read_text())
        except Exception:
            reg = {"champions": []}
    reg.setdefault("champions", [])
    c1 = result["clause1_genuine_learning"]
    shic_h1 = c1["per_horizon"].get(1, {}).get("shic")
    ic_h1 = c1["per_horizon"].get(1, {}).get("contig_ic")
    ratio_h1 = c1["per_horizon"].get(1, {}).get("shic_over_contig")
    record = {
        "forecaster_id": result["forecaster_id"],
        "gate": "GATE_A",
        "held_out_ic_h1": ic_h1,
        "shic_h1": shic_h1,
        "shic_over_contig_ic_h1": ratio_h1,
        "shic_per_horizon": {str(h): d["shic"] for h, d in c1["per_horizon"].items()},
        "passed_gate_a": result["overall_pass"],
        "verdict": result["verdict"],
        "clauses_pass": {str(k): v for k, v in result["clauses_pass"].items()},
        "cost_rt_gate_value": result["cost_rt_gate_value"],
        "eligible_a1_substrate": result["overall_pass"],
        "ts": result["ts"],
    }
    # replace any prior record for the same forecaster_id (monotonic: keep the latest verdict)
    reg["champions"] = [c for c in reg["champions"]
                        if c.get("forecaster_id") != result["forecaster_id"]]
    reg["champions"].append(record)
    reg["_note"] = ("F (forecaster) champions ONLY. Keyed by held-out IC + ShIC (the TRUTH axis -- "
                    "did it learn or memorize?). NEVER ranked by compound. Physically separate from "
                    "agents.json so IC-as-objective cannot re-enter through a shared leaderboard. "
                    "passed_gate_a / eligible_a1_substrate are set by src/strat/wm_value_probe.py "
                    "--gate-a (docs/AGENT_LAYER_ARCHITECTURE_2026_06_11.md S2.4).")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write (tmp + replace) -- the project's G-AUDIT-020 contract
    tmp = registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    tmp.replace(registry_path)
    return registry_path


# ===========================================================================
# SYNTHETIC FORECASTERS for the two-sided RWYB (CPU, no GPU, no V1.1 checkpoint)
# ===========================================================================
# A forecaster exposes ONE method: predict(df_pd) -> {h: per-bar predicted forward return}.
# The trained V1.1 producer will be wrapped to expose the same surface (make_v11_forecaster).

class _SyntheticForecaster:
    """Builds a regime-switching synthetic market with a GENUINE predictive FEATURE, plus a
    forecaster over it. The market design is the key to a HONEST two-sided test:

    MARKET: a hidden per-bar 'signal feature' phi[t] drives the NEXT-bar drift. There are three
    regime types -- bull (phi>0 -> up-drift), BEAR (phi<0 -> DOWN-drift), high-vol (noisy, small
    drift). The bear segments mean buy-hold is NOT a free win: a forecaster that reads phi and goes
    flat in bear can BEAT buy-hold (clause 2's hard bar). All three regimes recur in every window.

    FORECASTER (the only difference between the two arms):
      genuine=True  : pred[t] = f(phi[t]) -- reads the real feature. Under the project's GLOBAL
                      FEATURE SHUFFLE (re-run on shuffled feature rows, target shuffled the same
                      way), the feature->return mapping is PRESERVED -> ShIC>0, ratio~1. The
                      predict-then-rule policy avoids bear -> beats B&H + random in every regime.
                      Should PASS Gate-A.
      genuine=False : pred[t] = g(t) -- a function of TEMPORAL POSITION only (a memorizer),
                      uncorrelated with phi. Contiguous IC can look OK by luck on a slice, but the
                      global feature shuffle DESTROYS it (position is gone) -> ShIC~0, ratio~0.
                      The policy has no real edge -> fails clause 2/4. Should FAIL Gate-A.

    This mirrors the real WM anti-memorization test exactly (validate_world.py): genuine = learned
    feature->return signal survives de-temporalization; memorizer = collapses. Two-sidedness proven."""

    def __init__(self, genuine: bool, seed: int = 0, n: int = 2600):
        self.genuine = genuine
        self.seed = seed
        rng = np.random.default_rng(1234)  # market shared across ALL seeds (deterministic substrate)
        regime_len = 30
        n_reg = n // regime_len + 1
        kinds = np.array([["bull", "bear", "hivol"][k % 3] for k in range(n_reg)])
        # hidden signal feature phi[t]: sign+magnitude of the regime, + per-bar noise (still readable)
        phi = np.zeros(n)
        vol = np.zeros(n)
        for k in range(n_reg):
            sl = slice(k * regime_len, min((k + 1) * regime_len, n))
            if kinds[k] == "bull":
                phi[sl], vol[sl] = 0.012, 0.010
            elif kinds[k] == "bear":
                phi[sl], vol[sl] = -0.022, 0.010   # DEEP genuine DOWN-drift -> sitting out bear is a
                                                   # large, stable edge random entry cannot replicate
            else:  # hivol: genuine up-drift, larger noise. A positive control must be genuinely
                   # predictive in EVERY regime (incl. high-vol) -- SNR kept > 1 so the feature leads.
                phi[sl], vol[sl] = 0.014, 0.016
        phi = phi + rng.normal(0.0, 0.002, n)      # per-bar feature noise (still strongly predictive)
        # NEXT-bar return is driven by phi (the feature genuinely leads the move) + noise
        eps = rng.normal(0.0, 1.0, n) * vol
        rets = np.zeros(n)
        rets[1:] = phi[:-1] + eps[1:]              # ret[t] = phi[t-1] + noise  (phi LEADS by 1 bar)
        close = 100.0 * np.cumprod(1.0 + rets)
        open_ = np.concatenate([[100.0], close[:-1]])
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.002, n)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.002, n)))
        dates = pd.date_range("2019-01-01", periods=n, freq="D")
        self.df = pd.DataFrame({"date": dates, "open": open_, "high": high,
                                "low": low, "close": close})
        # The forecaster's INPUT FEATURE matrix: the genuine leading feature phi shifted to be
        # PAST-ONLY at bar t (phi[t-1] is known at the close of bar t and predicts ret[t+...]).
        phi_pastonly = np.concatenate([[0.0], phi[:-1]])   # feature available at bar t
        self._features = phi_pastonly.reshape(-1, 1)       # [n, 1]
        self._n = n

    # --- the Forecaster surface Gate-A consumes ---
    def features(self) -> np.ndarray:
        return self._features

    def predict_from_features(self, feats: np.ndarray, h: int) -> np.ndarray:
        """Map a feature matrix -> per-bar predicted forward return at horizon h. This is the
        re-runnable inference path the GLOBAL FEATURE SHUFFLE calls. genuine reads the feature;
        the memorizer ignores it (its prediction is positional, supplied via predict())."""
        x = np.asarray(feats, float).reshape(len(feats), -1)[:, 0]
        if self.genuine:
            return x * h                       # genuine: prediction IS a function of the feature
        # memorizer: prediction does NOT depend on the feature -> a constant-ish positional echo.
        # Under the shuffle this carries no feature signal -> ShIC ~ 0 (exactly the intent).
        return np.zeros(len(feats))

    def predict(self, df_pd: pd.DataFrame) -> dict:
        """Return {h: per-bar predicted forward return} aligned to this forecaster's own market."""
        n = self._n
        out = {}
        phi_past = self._features[:, 0]
        for h in GATE_A["planner_horizons"]:
            if self.genuine:
                # genuine: read the leading feature (past-only) + small measurement noise
                noise = np.random.default_rng(99 + self.seed + h).normal(0, 0.001, n)
                out[h] = phi_past * h + noise
            else:
                # memorizer: prediction is a function of TEMPORAL POSITION, uncorrelated with phi.
                # On a contiguous slice it may correlate with realized return by luck, but it
                # encodes NO feature signal -> the global feature shuffle exposes it (ShIC~0).
                t = np.arange(n)
                pos = np.sin(t / 11.0 + self.seed) * 0.01   # positional pattern (re-randomized/seed)
                out[h] = pos * h
        return out


def make_synthetic_forecaster_factory(genuine: bool):
    """Returns factory(seed) -> _SyntheticForecaster, and the shared OHLC frame (seed 0)."""
    def factory(seed: int):
        return _SyntheticForecaster(genuine=genuine, seed=seed)
    return factory, factory(0).df


# ---------------------------------------------------------------------------
# THE FORECASTER ADAPTER CONTRACT (what Gate-A consumes; what V1.1 must implement)
# ---------------------------------------------------------------------------
# A "Forecaster" is ANY object exposing EXACTLY these three methods. Gate-A is agnostic to
# the implementation (synthetic, V1.1, V12, ...). The ShIC clause (clause 1) REQUIRES all three
# -- features() + predict_from_features() are what make the global-feature-shuffle re-runnable:
#
#   predict(df_pd)                  -> {h: per-bar predicted forward return}, aligned to df_pd.
#                                      MUST cache the feature matrix it used (so features() can
#                                      return it). PAST-ONLY (the model's causal shift).
#   features()                      -> [n, d] the EXACT feature matrix predict() consumed, in the
#                                      SAME row order. (Call predict() first; it populates the cache.)
#   predict_from_features(feats, h) -> [m] re-runnable inference on ARBITRARY feature rows at
#                                      horizon h. This is the contract the global feature shuffle
#                                      calls with SHUFFLED rows -- so a genuine feature->return
#                                      learner retains IC (ShIC>0) and a temporal memorizer collapses.
#
# The reference adapter below (_ChimeraSurfaceForecaster) implements this contract on the chimera
# loader + WMEntryProducer.run_inference surface, so --gate-a --forecaster-id V1.1 RUNS (CPU-capable)
# on the real trained WM the moment its checkpoint lands -- it does NOT crash with AttributeError.
# ---------------------------------------------------------------------------

def make_v11_forecaster_factory(n_features: int = 41, device: str | None = None):
    """Wrap the trained V1.1 WMEntryProducer to the Forecaster surface (predict / features /
    predict_from_features). Used by --gate-a --forecaster-id V1.1.

    This is a WORKING adapter -- it implements the full ShIC contract over the producer's
    run_inference({h: pred}) accessor (the producer already returns RAW per-horizon predicted
    returns; Gate-A consumes those directly, NOT the thresholded entry). It loads the real
    checkpoint on construction (CPU if device='cpu', else the settings DEVICE). It does NOT need
    the GPU to be CORRECT -- run with device='cpu' to exercise the full gate path on the trained
    WM without GPU contention; device defaults to the WM's own setting for the production run.

    The factory returns factory(seed) -> forecaster. The trained WM is DETERMINISTIC, so per-seed
    variation flows only through the null sampling in clauses 2-3 (the synthetic null re-randomizes
    per seed; the real WM does not -- that is the correct, honest behaviour for a fixed model)."""
    from strat.wm_entry_producer import WMEntryProducer  # local import: avoids torch at module load

    class _V11Forecaster:
        """The WORKING V1.1 forecaster adapter (the exact ShIC contract).

        features(): the chimera feature matrix the producer consumed (cached by predict()).
        predict_from_features(feats, h): re-runs the producer's inference on arbitrary feature
            rows for one horizon -- the re-runnable path the global feature shuffle requires.
        predict(df_pd): {h: per-bar predicted forward return}, caches the feature matrix."""

        def __init__(self, seed: int = 0):
            self.seed = seed
            self._p = WMEntryProducer(n_features=n_features, device=device)
            self._features_cache = None      # [n, input_dim] last feature matrix predict() used
            self._asset_idx = None           # asset index for the cached frame
            self._n = None                   # row count of the cached frame

        # --- map a df_pd (OHLC+chimera) to the producer's [n, input_dim] feature matrix ---
        def _extract_feats(self, df_pd: pd.DataFrame):
            """Build the producer's feature matrix for df_pd. Accepts a frame that already
            carries the chimera feature columns (the real run supplies the chimera slice); the
            producer's selective_drop_nulls + extract_features_targets do the canonical extraction."""
            import polars as pl
            from pipeline.data_integrity import selective_drop_nulls, extract_features_targets
            df_pl = pl.from_pandas(df_pd) if not isinstance(df_pd, pl.DataFrame) else df_pd
            sym = self._infer_symbol(df_pd)
            df_clean = selective_drop_nulls(df_pl, self._p.feature_list,
                                            self._p.reward_horizons, sym)
            feats, _ = extract_features_targets(df_clean, self._p.feature_list,
                                                self._p.reward_horizons, sym)
            return np.asarray(feats, dtype=np.float32), self._p.asset_to_idx[sym]

        @staticmethod
        def _infer_symbol(df_pd: pd.DataFrame) -> str:
            # the real run threads the asset through; default to BTCUSDT for a single-asset slice.
            sym = getattr(df_pd, "_gate_a_symbol", None) or "BTCUSDT"
            return sym if sym.endswith("USDT") else sym + "USDT"

        def predict(self, df_pd: pd.DataFrame) -> dict:
            """{h: per-bar predicted forward return} aligned to df_pd; caches the feature matrix."""
            feats, asset_idx = self._extract_feats(df_pd)
            self._features_cache = feats
            self._asset_idx = asset_idx
            self._n = len(feats)
            preds_all = self._p.run_inference(feats, asset_idx)   # {h_all: [n]} raw predicted returns
            # surface ONLY the planner horizons Gate-A queries (a subset of REWARD_HORIZONS)
            return {h: np.asarray(preds_all[h], dtype=float) for h in GATE_A["planner_horizons"]}

        def features(self) -> np.ndarray:
            """The [n, input_dim] feature matrix predict() consumed (call predict() first)."""
            if self._features_cache is None:
                raise RuntimeError("call predict(df_pd) before features() -- it populates the cache")
            return self._features_cache

        def predict_from_features(self, feats: np.ndarray, h: int) -> np.ndarray:
            """Re-runnable inference on ARBITRARY feature rows at horizon h (the global-shuffle path).
            Runs the SAME producer inference on the supplied rows and returns horizon-h preds."""
            feats = np.asarray(feats, dtype=np.float32)
            asset_idx = self._asset_idx if self._asset_idx is not None else 0
            preds_all = self._p.run_inference(feats, asset_idx)
            return np.asarray(preds_all[h], dtype=float)

    def factory(seed: int):
        return _V11Forecaster(seed)
    return factory


# ---------------------------------------------------------------------------
# GATE-A RWYB: two-sided demonstration (CPU, synthetic, no GPU)
# ---------------------------------------------------------------------------

def gate_a_rwyb(n_seeds: int = 10) -> dict:
    """Two-sided RWYB: a GENUINE synthetic forecaster PASSES Gate-A; a SHUFFLED/zero-signal one
    FAILS. Verifies the gate LOGIC + its two-sidedness on CPU -- NOT GPU perf. The real V1.1 run
    plugs the trained producer into the SAME code path when it lands."""
    win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15",
                     oos_end="2025-09-30", unseen_end="2030-12-31")
    print("\n" + "#" * 74)
    print("#  GATE-A RWYB -- two-sided gate-logic verification (CPU, synthetic, no GPU)")
    print("#  (genuine forecaster must PASS; shuffled/zero-signal must FAIL)")
    print("#" * 74)

    gen_factory, gen_df = make_synthetic_forecaster_factory(genuine=True)
    gen = run_gate_a(gen_factory, gen_df, win, forecaster_id="SYNTH_GENUINE", n_seeds=n_seeds)

    null_factory, null_df = make_synthetic_forecaster_factory(genuine=False)
    nul = run_gate_a(null_factory, null_df, win, forecaster_id="SYNTH_SHUFFLED_NULL", n_seeds=n_seeds)

    print("\n" + "#" * 74)
    print("#  TWO-SIDED RESULT")
    print("#" * 74)
    print(f"  GENUINE forecaster  -> {gen['verdict']}   (overall_pass={gen['overall_pass']})")
    print(f"  SHUFFLED/null       -> {nul['verdict']}   (overall_pass={nul['overall_pass']})")
    two_sided_ok = bool(gen["overall_pass"] and not nul["overall_pass"])
    print(f"\n  TWO-SIDED SOUNDNESS: {'PASS' if two_sided_ok else 'CHECK'} -- "
          + ("the gate ACCEPTS a genuine forecaster AND REJECTS a shuffled/zero-signal one "
             "(it is not an accept-everything or reject-everything sieve)."
             if two_sided_ok else
             "the gate did NOT cleanly separate genuine from null -- inspect the per-clause flags above."))
    # write BOTH verdicts to the registry (the genuine one is the meaningful champion record)
    p = write_to_forecasters_registry(gen)
    write_to_forecasters_registry(nul)
    print(f"\n  registry written: {p}")
    return {"genuine": gen, "null": nul, "two_sided_ok": two_sided_ok, "registry": str(p)}


# ===========================================================================
# Main evaluation loop (L1 -- the original real-data WM value probe; needs V1.1 GPU)
# ===========================================================================

def evaluate_asset(sym: str, producer, n_seeds: int = 10) -> dict:
    print(f"\n{'='*60}")
    print(f"  {sym} -- UNSEEN evaluation")
    print(f"{'='*60}")

    df_unseen_pl, df_pd, window, _ = load_unseen_data(sym)
    n_unseen = len(df_pd)

    if n_unseen < 500:
        print(f"  [SKIP] {sym}: only {n_unseen} UNSEEN bars (< 500 minimum)")
        return {"asset": sym, "skip": True, "reason": "insufficient_unseen"}

    # --- WM entry signal ---
    print(f"  Running WM inference on {n_unseen:,} UNSEEN bars...")
    wm_entry = producer.produce(df_unseen_pl, sym, mode="h4_roll_rgm16")
    n_wm = int(wm_entry.sum())
    print(f"  WM entries: {n_wm} ({n_wm/n_unseen*100:.1f}% of bars)")

    # --- WM backtest ---
    print(f"  Backtesting WM signal...")
    wm_res = run_harness(df_pd, wm_entry, "wm_h4_roll_rgm16", window=window)
    print(f"  WM compound: {wm_res['compound_pct']:+.2f}%  "
          f"trades={wm_res['n_trades']}  maxDD={wm_res['max_dd_pct']:.1f}%  "
          f"winrate={wm_res['win_rate']:.1%}")

    # --- Buy-and-hold ---
    bh_compound = buy_hold_compound(df_pd)
    print(f"  Buy&Hold: {bh_compound:+.2f}%")

    # --- Random null ---
    print(f"  Running random-entry null ({n_seeds} seeds, regime-matched)...")
    rand_res = run_random_null(df_pd, wm_entry, n_seeds=n_seeds, window=window)
    print(f"  Random null: mean={rand_res['compound_pct']:+.2f}% "
          f"(+/-{rand_res.get('compound_std',0):.2f}%)  "
          f"entries_matched={rand_res.get('n_wm_entries',0)}")

    # --- Verdict for this asset ---
    beats_bh = wm_res["compound_pct"] > bh_compound
    beats_rand = wm_res["compound_pct"] > rand_res["compound_pct"]
    margin_vs_rand = wm_res["compound_pct"] - rand_res["compound_pct"]
    margin_vs_bh = wm_res["compound_pct"] - bh_compound

    verdict = "PASS" if (beats_bh and beats_rand) else \
              "PARTIAL" if (beats_rand and not beats_bh) else "FAIL"
    print(f"  VERDICT: {verdict}  (vs B&H: {margin_vs_bh:+.2f}pp, vs random: {margin_vs_rand:+.2f}pp)")

    return {
        "asset": sym,
        "n_unseen_bars": n_unseen,
        "wm": wm_res,
        "buy_hold_pct": bh_compound,
        "random_null": rand_res,
        "beats_bh": beats_bh,
        "beats_rand": beats_rand,
        "margin_vs_rand_pp": margin_vs_rand,
        "margin_vs_bh_pp": margin_vs_bh,
        "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="WM forecaster value probe + Gate-A")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-features", type=int, default=41)
    parser.add_argument("--gate-a", action="store_true",
                        help="Run GATE-A (S2.4): the 5-clause forecaster->A1 eligibility gate. "
                             "Default = CPU/synthetic two-sided RWYB (no GPU). With --forecaster-id "
                             "V1.1 it runs the gate on the real trained V1.1 forecaster (needs the "
                             "landed checkpoint).")
    parser.add_argument("--forecaster-id", type=str, default=None,
                        help="Gate-A target forecaster id (e.g. V1.1). Omit for the synthetic RWYB.")
    parser.add_argument("--device", type=str, default=None,
                        help="Gate-A real-forecaster device override (e.g. cpu). Default = WM setting. "
                             "Use cpu to run the V1.1 gate path without GPU contention.")
    args = parser.parse_args()

    # ---- GATE-A branch (S2.4): forecaster -> A1 eligibility ----
    if args.gate_a:
        if args.forecaster_id and args.forecaster_id.upper() not in ("SYNTH", "SYNTHETIC"):
            # REAL forecaster run -- only valid once the named WM (e.g. V1.1) has landed.
            print(f"\n[Gate-A] REAL forecaster run requested: {args.forecaster_id} "
                  f"(device={args.device or 'WM-default'})")
            factory = make_v11_forecaster_factory(n_features=args.n_features, device=args.device)
            # build the real OHLC frame from the first asset's UNSEEN+held-out slice
            sym0 = args.assets[0]
            _, df_pd, window, _ = load_unseen_data(sym0)
            df_pd._gate_a_symbol = sym0  # thread the asset through to the forecaster adapter
            res = run_gate_a(factory, df_pd, window, forecaster_id=args.forecaster_id,
                             n_seeds=args.seeds)
            p = write_to_forecasters_registry(res)
            print(f"\n  registry written: {p}")
            return res
        # default: two-sided CPU/synthetic RWYB (verifies gate LOGIC + two-sidedness, not GPU perf)
        return gate_a_rwyb(n_seeds=args.seeds)

    print("\n" + "=" * 70)
    print("  WM V1.1 HELD-OUT VALUE PROBE")
    print("  Question: Does V1.1 WM add compound-return value on UNSEEN?")
    print("  Signal: h4_roll_rgm16 (sweep-validated, rebal_bars=48, regime_window=500)")
    print("  Exit: 2% pct trail + 5% hard stop + 500-bar time cap (dollar-bar calibrated)")
    print("  Cost: TAKER 0.0024 RT")
    print("  Controls: (a) buy-and-hold, (b) random-entry null (10 seeds, regime-matched)")
    print("=" * 70)

    from strat.wm_entry_producer import WMEntryProducer
    producer = WMEntryProducer(n_features=args.n_features)

    all_results = []
    for sym in args.assets:
        try:
            res = evaluate_asset(sym, producer, n_seeds=args.seeds)
            all_results.append(res)
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"asset": sym, "skip": True, "reason": str(e)})

    # --- Portfolio summary ---
    valid = [r for r in all_results if not r.get("skip")]
    print("\n" + "=" * 70)
    print("  PORTFOLIO SUMMARY (UNSEEN segment, per-asset compound %)")
    print("=" * 70)
    print(f"  {'Asset':<12} {'WM':>10} {'B&H':>10} {'Rand_mean':>12} {'vs_rand':>10} {'vs_B&H':>10} {'VERDICT':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for r in valid:
        print(f"  {r['asset']:<12} "
              f"{r['wm']['compound_pct']:>+10.2f}% "
              f"{r['buy_hold_pct']:>+10.2f}% "
              f"{r['random_null']['compound_pct']:>+12.2f}% "
              f"{r['margin_vs_rand_pp']:>+10.2f}pp "
              f"{r['margin_vs_bh_pp']:>+10.2f}pp "
              f"{r['verdict']:>10}")

    if valid:
        wm_mean = np.mean([r["wm"]["compound_pct"] for r in valid])
        bh_mean = np.mean([r["buy_hold_pct"] for r in valid])
        rand_mean = np.mean([r["random_null"]["compound_pct"] for r in valid])
        pass_count = sum(1 for r in valid if r["verdict"] == "PASS")
        partial_count = sum(1 for r in valid if r["verdict"] == "PARTIAL")
        fail_count = sum(1 for r in valid if r["verdict"] == "FAIL")

        print(f"\n  {'PORTFOLIO AVG':<12} "
              f"{wm_mean:>+10.2f}% "
              f"{bh_mean:>+10.2f}% "
              f"{rand_mean:>+12.2f}%")
        print(f"\n  PASS: {pass_count}/{len(valid)}  PARTIAL: {partial_count}/{len(valid)}  FAIL: {fail_count}/{len(valid)}")

        # Final verdict
        print("\n" + "=" * 70)
        overall_beats_bh = wm_mean > bh_mean
        overall_beats_rand = wm_mean > rand_mean
        if overall_beats_bh and overall_beats_rand and pass_count >= len(valid) * 0.5:
            print("  FINAL VERDICT: YES -- WM adds held-out compound value.")
            print(f"    Portfolio WM {wm_mean:+.2f}% beats B&H {bh_mean:+.2f}% "
                  f"AND random {rand_mean:+.2f}% (margin: {wm_mean - rand_mean:+.2f}pp)")
        elif overall_beats_rand and not overall_beats_bh:
            print("  FINAL VERDICT: PARTIAL -- WM beats random control but not buy-hold.")
            print(f"    Portfolio WM {wm_mean:+.2f}% vs B&H {bh_mean:+.2f}%  "
                  f"vs random {rand_mean:+.2f}%")
            print("    WM provides ENTRY TIMING value but the bull-market beta floor is higher.")
        else:
            print("  FINAL VERDICT: NO -- WM does NOT add held-out compound value.")
            print(f"    Portfolio WM {wm_mean:+.2f}% vs B&H {bh_mean:+.2f}%  "
                  f"vs random {rand_mean:+.2f}%")
            print("    More GPU on WM upgrades is NOT justified by this metric.")
        print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
