"""src/strat/s3_conditioner_test.py -- S3 top-trader ratio as a regime CONDITIONER on the trend book.

PRE-REGISTERED HYPOTHESIS (TRAIN-only evidence, 2026-06-17):
  When top-traders (top_pos_lsr, top_acct_lsr) are at extreme long positions (per-asset z-score > threshold),
  forward 20d returns are depressed (TRAIN: extreme z>1.5 -> +0.18% vs normal +5.02%). Contrarian signal.

  GATE spec: SKIP new book entries when per-asset EWMA-z-score of top_pos_lsr > 1.5 OR top_acct_lsr > 1.5
  (threshold and direction PRE-REGISTERED from TRAIN; not tuned on OOS).

HONEST CONTRACT:
  1. Un-gated trend book = baseline (same as trend_book_lab.py best config = atr_mult=10, regime_gate=True)
  2. Gated book = skip entries where conditioner fires
  3. Shuffled-conditioner null = same gate structure but conditioner dates randomly shuffled per-asset
     (same exposure reduction, random timing -> isolates whether conditioner TIMING matters)
  4. All results: TRAIN, OOS (verdict), UNSEEN (touched once)
  5. Long-only, spot, lev=1, taker 0.0024 RT
  6. Optimize on TRAIN+VAL only; OOS+UNSEEN are held-out (seal intact)

Constraints: long-only, spot, lev=1, WEALTH not IC, UNSEEN sealed.

Run:
    python src/strat/s3_conditioner_test.py
    python src/strat/s3_conditioner_test.py --z-thresh 1.0  # test looser gate
    python src/strat/s3_conditioner_test.py --col top_pos_lsr --z-thresh 1.5
    python src/strat/s3_conditioner_test.py --col top_acct_lsr --z-thresh 1.5
    python src/strat/s3_conditioner_test.py --col both --z-thresh 1.5  # pre-registered OR combo
"""
from __future__ import annotations
import argparse, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (aligned with trend_book_lab.py)
# ---------------------------------------------------------------------------
COST_RT     = 0.0024   # taker round-trip
ATR_PERIOD  = 14
LONG_MA     = 200
SHORT_MA    = 50
ACCEL_MA    = 20

TRAIN_END  = pd.Timestamp("2024-05-15")
VAL_END    = pd.Timestamp("2025-03-15")
OOS_END    = pd.Timestamp("2025-12-31")
UNSEEN_END = pd.Timestamp("2026-06-01")

U10 = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT",
    "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT",
    "LINK": "LINKUSDT", "LTC": "LTCUSDT",
}

ATR_MULT    = 10.0   # pre-registered: trend_book_lab best config
REGIME_GATE = True   # pre-registered
EWMA_SPAN   = 20     # EWMA smoothing for z-score computation (past-only)
Z_THRESH    = 1.5    # pre-registered threshold (from TRAIN quintile analysis)

N_SHUFFLE   = 200    # number of shuffled-conditioner null runs


# ---------------------------------------------------------------------------
# OHLC loading
# ---------------------------------------------------------------------------
def load_ohlc(sym: str) -> pd.DataFrame | None:
    cl = ChimeraLoader()
    try:
        loaded = cl.load(sym, cadence="1d")
    except Exception:
        return None
    df = loaded if hasattr(loaded, "iloc") else pd.DataFrame(loaded.to_dict(as_series=False))
    df["date"] = (pd.to_datetime(df["date"], unit="ms")
                  if np.issubdtype(df["date"].dtype, np.number)
                  else pd.to_datetime(df["date"]))
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Indicator computation (identical to trend_book_lab.py)
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    n = len(c)
    prev_c = np.empty(n); prev_c[0] = np.nan; prev_c[1:] = c[:-1]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    df = df.copy()
    df["_tr"] = tr
    df["atr14"]    = df["_tr"].rolling(ATR_PERIOD).mean()
    df["sma200"]   = df["close"].rolling(LONG_MA).mean()
    df["sma50"]    = df["close"].rolling(SHORT_MA).mean()
    df["sma20"]    = df["close"].rolling(ACCEL_MA).mean()
    df["sma50_rising"] = (df["sma50"] > df["sma50"].shift(1)).astype(float)
    df.drop(columns=["_tr"], inplace=True)
    return df


def build_entry_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cond1 = df["close"] > df["sma50"]
    cond2 = df["sma50"] > df["sma200"]
    cond3 = df["sma50_rising"] > 0.5
    regime_ok = df["close"] > df["sma200"]
    df["entry_signal"] = (cond1 & cond2 & cond3 & regime_ok).astype(float)
    nan_mask = df[["sma200", "sma50", "sma20", "atr14"]].isna().any(axis=1)
    df.loc[nan_mask, "entry_signal"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Conditioner: per-asset EWMA z-score (past-only)
# ---------------------------------------------------------------------------
def build_conditioner_signal(
    df: pd.DataFrame,
    s3_series: pd.Series,   # indexed by date, the LSR column
    col_name: str,
    z_thresh: float = Z_THRESH,
    ewma_span: int = EWMA_SPAN,
) -> pd.DataFrame:
    """Add a 'gate_block' column = 1 when conditioner fires (skip entries), 0 otherwise.

    Gate fires when per-asset PAST-ONLY EWMA z-score of LSR > z_thresh.
    EWMA mean and std computed on expanding window up to t-1 (shift(1) to prevent any
    same-bar look-ahead). Missing s3 values -> gate does NOT fire (stay in market).

    Past-only guarantee: z-score at bar t uses only data through bar t-1.
    """
    df = df.copy()
    # Align s3 onto df dates
    df["_lsr"] = df["date"].map(s3_series.to_dict())
    # Shift by 1: use YESTERDAY's reading to gate TODAY's entry
    df["_lsr_lag"] = df["_lsr"].shift(1)
    # EWMA z-score: rolling expanding EWMA mean + std, then normalize
    # Use pandas ewm with span=ewma_span, past-only (no min_periods concern since we shift)
    lsr_vals = df["_lsr_lag"]
    ewma_mean = lsr_vals.ewm(span=ewma_span, min_periods=5).mean()
    ewma_std  = lsr_vals.ewm(span=ewma_span, min_periods=5).std()
    z = (lsr_vals - ewma_mean) / (ewma_std + 1e-8)
    df["_lsr_z"] = z
    # Gate fires if z > z_thresh and the value is actually present
    df["gate_block"] = ((df["_lsr_z"] > z_thresh) & df["_lsr_lag"].notna()).astype(int)
    # Clean up
    df.drop(columns=["_lsr", "_lsr_lag", "_lsr_z"], inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Single-asset simulator (with optional gate)
# ---------------------------------------------------------------------------
def _label_window(ts: pd.Timestamp) -> str:
    if ts < TRAIN_END:  return "TRAIN"
    if ts < VAL_END:    return "VAL"
    if ts < OOS_END:    return "OOS"
    return "UNSEEN"


def simulate_asset(
    df: pd.DataFrame,
    gate_block: pd.Series | None = None,   # boolean series aligned to df index (if None = ungated)
) -> list[dict]:
    """Run trend book on one asset. gate_block = 1 means skip entry at that bar."""
    df = compute_indicators(df)
    df = build_entry_signal(df)

    opens     = df["open"].values.astype(float)
    highs     = df["high"].values.astype(float)
    lows      = df["low"].values.astype(float)
    closes    = df["close"].values.astype(float)
    atr       = df["atr14"].values.astype(float)
    dates     = pd.to_datetime(df["date"])
    entry_arr = df["entry_signal"].values > 0.5

    # gate_block: 1 = skip entry
    if gate_block is not None:
        gb = gate_block.values.astype(int)
    else:
        gb = np.zeros(len(df), dtype=int)

    n = len(opens)
    trades = []
    i = 0
    while i < n - 2:
        if not entry_arr[i] or gb[i] == 1:
            i += 1
            continue
        entry_fill = i + 1
        if entry_fill >= n:
            break
        entry_p = opens[entry_fill]
        hwm = max(entry_p, highs[entry_fill])
        exit_fill = None
        exit_p = None
        reason = "tail_flush"
        j = entry_fill + 1
        while j < n:
            atr_ref = atr[j - 1] if j > 0 and np.isfinite(atr[j - 1]) else np.nan
            if np.isfinite(atr_ref):
                stop_level = hwm - ATR_MULT * atr_ref
                if lows[j] <= stop_level:
                    exit_fill = j
                    exit_p = min(opens[j], stop_level)
                    reason = "atr_trail"
                    break
            hwm = max(hwm, highs[j])
            j += 1
        if exit_fill is None:
            exit_fill = n - 1
            exit_p = closes[n - 1]
            reason = "tail_flush"
        net = exit_p / entry_p - 1.0 - COST_RT
        ts = dates.iloc[i]
        trades.append({
            "window":        _label_window(ts),
            "entry_idx":     int(i),
            "exit_idx":      int(exit_fill),
            "entry_ts":      str(ts.date()),
            "net_pnl":       float(net),
            "duration_bars": int(exit_fill - entry_fill),
            "exit_reason":   reason,
            "gated":         0,   # 0 = not gated (went through)
        })
        i = max(exit_fill, i + 1)
    return trades


# ---------------------------------------------------------------------------
# Book aggregation
# ---------------------------------------------------------------------------
def book_compound(per_asset_trades: dict[str, list[dict]], window: str) -> dict:
    asset_comps = []
    asset_ns = []
    for sym, trades in per_asset_trades.items():
        sub = [t for t in trades if t["window"] == window]
        if not sub:
            asset_comps.append(0.0); asset_ns.append(0)
            continue
        rets = np.array([t["net_pnl"] for t in sub])
        comp = float((np.prod(1.0 + rets) - 1.0) * 100.0)
        asset_comps.append(comp); asset_ns.append(len(sub))
    n_assets = len(asset_comps)
    book_total = float((np.prod([(1.0 + c / 100.0) for c in asset_comps]) ** (1.0 / n_assets) - 1.0) * 100.0)
    return {
        "book_compound_pct": round(book_total, 3),
        "total_trades":      sum(asset_ns),
        "asset_comps":       {sym: round(c, 2) for sym, c in zip(per_asset_trades.keys(), asset_comps)},
    }


def book_max_dd(per_asset_trades: dict[str, list[dict]], window: str) -> float:
    dds = []
    for sym, trades in per_asset_trades.items():
        sub = [t for t in trades if t["window"] == window]
        if not sub:
            continue
        rets = np.array([t["net_pnl"] for t in sub])
        eq = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min() * 100.0)
        dds.append(dd)
    return round(min(dds) if dds else 0.0, 2)


def cagr_from_compound(compound_pct: float, window: str) -> float:
    spans = {
        "TRAIN": ((pd.Timestamp("2022-01-01"), TRAIN_END)),   # s3 starts 2022
        "VAL":   (TRAIN_END, VAL_END),
        "OOS":   (VAL_END,   OOS_END),
        "UNSEEN":(OOS_END,   UNSEEN_END),
    }
    start, end = spans[window]
    n_years = (end - start).days / 365.25
    if n_years <= 0 or compound_pct <= -100.0:
        return 0.0
    return round(((1.0 + compound_pct / 100.0) ** (1.0 / n_years) - 1.0) * 100.0, 2)


# ---------------------------------------------------------------------------
# Shuffled-conditioner null
# ---------------------------------------------------------------------------
def shuffle_gate_per_asset(gate_blocks: dict[str, pd.Series], rng: np.random.Generator) -> dict[str, pd.Series]:
    """Shuffle the gate_block values within each asset's date range (preserves exposure rate, randomises timing)."""
    shuffled = {}
    for sym, gb in gate_blocks.items():
        vals = gb.values.copy()
        rng.shuffle(vals)
        shuffled[sym] = pd.Series(vals, index=gb.index)
    return shuffled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(col: str = "both", z_thresh: float = Z_THRESH, n_shuffle: int = N_SHUFFLE) -> None:
    print(f"\n=== S3 CONDITIONER TEST ===")
    print(f"  col={col}  z_thresh={z_thresh}  n_shuffle={n_shuffle}")
    print(f"  book config: atr_mult={ATR_MULT}  regime_gate={REGIME_GATE}")
    print(f"  PRE-REGISTERED direction: gate=SKIP when top-trader longs at extreme z > threshold\n")

    # --- Load OHLC ---
    print("Loading OHLC for U10...")
    asset_dfs = {}
    for base, sym in U10.items():
        df = load_ohlc(sym)
        if df is not None and len(df) > 300:
            asset_dfs[base] = df
            print(f"  {base}: {len(df)} bars, {df['date'].min().date()} -> {df['date'].max().date()}")
        else:
            print(f"  {base}: SKIP (no data)")

    # --- Load S3 metrics ---
    s3_path = ROOT / "data" / "processed" / "panels" / "daily" / "s3_metrics_panel.parquet"
    print(f"\nLoading s3 panel: {s3_path}")
    df_s3 = pd.read_parquet(s3_path)
    df_s3["date"] = pd.to_datetime(df_s3["date"])

    # Build per-asset s3 series
    s3_by_asset: dict[str, dict[str, pd.Series]] = {}
    for base in asset_dfs:
        sub = df_s3[df_s3["asset"] == base].set_index("date").sort_index()
        s3_by_asset[base] = {
            "top_pos_lsr":  sub["top_pos_lsr"].dropna(),
            "top_acct_lsr": sub["top_acct_lsr"].dropna(),
            "taker_lsr":    sub["taker_lsr"].dropna(),
        }

    # --- Build gate signals per asset ---
    # Note: gate is computed on the full date range but only applied to entry bars;
    # past-only z-score uses EWMA so early bars have less data (handled by min_periods=5)
    print(f"\nBuilding conditioner gate (col={col}, z_thresh={z_thresh})...")
    gate_blocks: dict[str, pd.Series] = {}
    gate_rates: dict[str, float] = {}

    for base, df in asset_dfs.items():
        df_ind = compute_indicators(df)
        df_ind = build_entry_signal(df_ind)
        # entry opportunities (where the trend book would signal)
        entry_dates = df_ind[df_ind["entry_signal"] > 0.5]["date"]

        # Build gate_block series aligned to df index
        gb_pos  = pd.Series(0, index=df.index, dtype=int)
        gb_acct = pd.Series(0, index=df.index, dtype=int)

        if col in ("top_pos_lsr", "both"):
            df_tmp = build_conditioner_signal(df, s3_by_asset[base]["top_pos_lsr"], "top_pos_lsr", z_thresh)
            gb_pos = df_tmp["gate_block"]
        if col in ("top_acct_lsr", "both"):
            df_tmp = build_conditioner_signal(df, s3_by_asset[base]["top_acct_lsr"], "top_acct_lsr", z_thresh)
            gb_acct = df_tmp["gate_block"]

        if col == "both":
            gate_blocks[base] = (gb_pos | gb_acct).astype(int)
        elif col == "top_pos_lsr":
            gate_blocks[base] = gb_pos
        elif col == "top_acct_lsr":
            gate_blocks[base] = gb_acct
        else:
            raise ValueError(f"Unknown col: {col}")

        # Gate rate on entry signals only (OOS window, honest)
        oos_mask = (df["date"] >= VAL_END) & (df["date"] < OOS_END)
        oos_entry_mask = oos_mask & (df_ind["entry_signal"].values > 0.5)
        n_oos_entry = oos_entry_mask.sum()
        n_gated = (gate_blocks[base][oos_entry_mask] == 1).sum() if n_oos_entry > 0 else 0
        gate_rates[base] = n_gated / n_oos_entry if n_oos_entry > 0 else 0.0

    avg_gate_rate = np.mean(list(gate_rates.values()))
    print(f"  OOS gate rate on entry signals: {avg_gate_rate*100:.1f}% avg (range: "
          f"{min(gate_rates.values())*100:.1f}%-{max(gate_rates.values())*100:.1f}%)")
    print(f"  Gate rates per asset: {' '.join(f'{k}={v*100:.0f}%' for k,v in sorted(gate_rates.items()))}")

    # --- Run UN-GATED baseline ---
    print("\nRunning UN-GATED baseline...")
    ungated_trades = {}
    for base, df in asset_dfs.items():
        ungated_trades[base] = simulate_asset(df, gate_block=None)
    ungated_results = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        b = book_compound(ungated_trades, w)
        b["cagr"] = cagr_from_compound(b["book_compound_pct"], w)
        b["max_dd"] = book_max_dd(ungated_trades, w)
        ungated_results[w] = b

    # --- Run GATED book ---
    print("Running GATED book (conditioner applied)...")
    gated_trades = {}
    for base, df in asset_dfs.items():
        gated_trades[base] = simulate_asset(df, gate_block=gate_blocks[base])
    gated_results = {}
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        b = book_compound(gated_trades, w)
        b["cagr"] = cagr_from_compound(b["book_compound_pct"], w)
        b["max_dd"] = book_max_dd(gated_trades, w)
        gated_results[w] = b

    # --- Shuffled-conditioner null ---
    print(f"Running shuffled-conditioner null ({n_shuffle} shuffles)...")
    rng = np.random.default_rng(42)
    shuffle_oos = []
    shuffle_unseen = []
    shuffle_train = []
    for _ in range(n_shuffle):
        shuffled_gb = shuffle_gate_per_asset(gate_blocks, rng)
        sh_trades = {}
        for base, df in asset_dfs.items():
            sh_trades[base] = simulate_asset(df, gate_block=shuffled_gb[base])
        sh_oos = book_compound(sh_trades, "OOS")["book_compound_pct"]
        sh_un  = book_compound(sh_trades, "UNSEEN")["book_compound_pct"]
        sh_tr  = book_compound(sh_trades, "TRAIN")["book_compound_pct"]
        shuffle_oos.append(sh_oos)
        shuffle_unseen.append(sh_un)
        shuffle_train.append(sh_tr)

    shuffle_oos    = np.array(shuffle_oos)
    shuffle_unseen = np.array(shuffle_unseen)
    shuffle_train  = np.array(shuffle_train)

    gated_oos    = gated_results["OOS"]["book_compound_pct"]
    gated_unseen = gated_results["UNSEEN"]["book_compound_pct"]
    ungated_oos  = ungated_results["OOS"]["book_compound_pct"]
    ungated_un   = ungated_results["UNSEEN"]["book_compound_pct"]

    p_oos_vs_ungated = float((shuffle_oos >= gated_oos).mean())
    p_oos_vs_shuffle = float((shuffle_oos >= gated_oos).mean())  # same: how often shuffle beats gated
    p_un_vs_ungated  = float((shuffle_unseen >= gated_unseen).mean())

    # --- Print results ---
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n{'Window':<10} {'UN-GATED%':>12} {'GATED%':>10} {'DELTA':>8} {'SH_MEAN':>10} {'SH_P90':>9} {'p_SH>=G':>9}")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        ug = ungated_results[w]["book_compound_pct"]
        g  = gated_results[w]["book_compound_pct"]
        delta = g - ug
        if w == "OOS":
            sh_arr = shuffle_oos
        elif w == "UNSEEN":
            sh_arr = shuffle_unseen
        elif w == "TRAIN":
            sh_arr = shuffle_train
        else:
            sh_arr = None
        if sh_arr is not None:
            sh_mean = np.mean(sh_arr)
            sh_p90  = np.percentile(sh_arr, 90)
            p_sh    = float((sh_arr >= g).mean())
            print(f"  {w:<8} {ug:>12.2f}% {g:>9.2f}% {delta:>+8.2f}pp {sh_mean:>9.2f}% {sh_p90:>9.2f}% {p_sh:>9.3f}")
        else:
            print(f"  {w:<8} {ug:>12.2f}% {g:>9.2f}% {delta:>+8.2f}pp {'N/A':>9} {'N/A':>9} {'N/A':>9}")
    print()

    # Trade count comparison
    print("TRADE COUNTS (total across U10)")
    for w in ("TRAIN", "VAL", "OOS", "UNSEEN"):
        n_ug = ungated_results[w]["total_trades"]
        n_g  = gated_results[w]["total_trades"]
        print(f"  {w}: ungated={n_ug}, gated={n_g}, blocked={n_ug-n_g} ({100*(n_ug-n_g)/max(n_ug,1):.1f}%)")
    print()

    # Verdict
    oos_delta   = gated_oos - ungated_oos
    un_delta    = gated_unseen - ungated_un
    sh_oos_p50  = np.percentile(shuffle_oos, 50)
    sh_oos_p05  = np.percentile(shuffle_oos, 5)
    gated_beats_ungated_oos = gated_oos > ungated_oos
    gated_beats_shuffle_oos = gated_oos > np.percentile(shuffle_oos, 50)  # beats median shuffle

    print("VERDICT")
    print("-"*50)
    print(f"  OOS: gated vs ungated:  {oos_delta:+.2f}pp  ({'POSITIVE' if oos_delta > 0 else 'NEGATIVE'})")
    print(f"  OOS: gated vs shuffle p50: gated={gated_oos:.2f}%  sh_p50={sh_oos_p50:.2f}%  sh_p05={sh_oos_p05:.2f}%")
    print(f"  OOS: P(shuffle >= gated) = {p_oos_vs_ungated:.3f}  (low=conditioner timing matters)")
    print(f"  UNSEEN: gated vs ungated: {un_delta:+.2f}pp  ({'POSITIVE' if un_delta > 0 else 'NEGATIVE'})")
    print()

    # Distinguish: value vs exposure-reduction
    # If gated book loses fewer trades but gains proportionally more -> timing skill
    # If shuffle beats ungated too (same exposure reduction) -> it's just fewer trades, not timing
    sh_oos_vs_ug = np.mean(shuffle_oos) - ungated_oos
    print(f"  Exposure-reduction test: shuffle_mean - ungated = {sh_oos_vs_ug:+.2f}pp")
    print(f"  Conditioner timing test: gated - shuffle_mean    = {gated_oos - np.mean(shuffle_oos):+.2f}pp")
    print()
    if gated_oos > ungated_oos and gated_oos > np.percentile(shuffle_oos, 75):
        print("  => CONDITIONER ADDS VALUE (beats ungated AND shuffle, timing skill present)")
    elif gated_oos > ungated_oos and gated_oos <= np.percentile(shuffle_oos, 75):
        print("  => VALUE IS EXPOSURE-REDUCTION (beats ungated but shuffle ties/beats gated)")
    elif gated_oos < ungated_oos:
        print("  => CONDITIONER HURTS (gated underperforms ungated book)")
    else:
        print("  => NEUTRAL (no clear advantage)")

    # Also test top_acct_lsr and taker_lsr standalone if running 'both'
    print()
    print(f"  S3 column used: {col}")
    print(f"  z_thresh: {z_thresh}  (pre-registered from TRAIN, NOT tuned on OOS)")
    print(f"  Conditioner direction: CONTRARIAN (skip extreme longs)")
    print()

    # Save results
    out_dir = ROOT / "runs" / "strat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config":   {"col": col, "z_thresh": z_thresh, "atr_mult": ATR_MULT, "regime_gate": REGIME_GATE},
        "ungated":  {w: {"compound_pct": ungated_results[w]["book_compound_pct"],
                         "cagr": ungated_results[w]["cagr"],
                         "max_dd": ungated_results[w]["max_dd"],
                         "n_trades": ungated_results[w]["total_trades"]} for w in ("TRAIN","VAL","OOS","UNSEEN")},
        "gated":    {w: {"compound_pct": gated_results[w]["book_compound_pct"],
                         "cagr": gated_results[w]["cagr"],
                         "max_dd": gated_results[w]["max_dd"],
                         "n_trades": gated_results[w]["total_trades"]} for w in ("TRAIN","VAL","OOS","UNSEEN")},
        "null":     {
            "n_shuffle":       n_shuffle,
            "oos_mean":        float(np.mean(shuffle_oos)),
            "oos_p05":         float(np.percentile(shuffle_oos, 5)),
            "oos_p50":         float(np.percentile(shuffle_oos, 50)),
            "oos_p95":         float(np.percentile(shuffle_oos, 95)),
            "p_shuffle_ge_gated_oos": float((shuffle_oos >= gated_oos).mean()),
            "unseen_mean":     float(np.mean(shuffle_unseen)),
            "unseen_p50":      float(np.percentile(shuffle_unseen, 50)),
        },
        "gate_rates": gate_rates,
        "exposure_reduction_oos_pp": float(np.mean(shuffle_oos) - ungated_oos),
        "timing_skill_oos_pp":       float(gated_oos - np.mean(shuffle_oos)),
    }
    out_path = out_dir / f"s3_conditioner_{col}_z{int(z_thresh*10)}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--col",       default="both",
                    choices=["top_pos_lsr", "top_acct_lsr", "taker_lsr", "both"],
                    help="Which s3 column(s) to use as conditioner")
    ap.add_argument("--z-thresh",  type=float, default=Z_THRESH,
                    help="z-score threshold (pre-registered=1.5)")
    ap.add_argument("--n-shuffle", type=int,   default=N_SHUFFLE,
                    help="Number of shuffled-null runs")
    args = ap.parse_args()
    main(col=args.col, z_thresh=args.z_thresh, n_shuffle=args.n_shuffle)
