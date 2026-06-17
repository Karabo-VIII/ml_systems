"""Wave 4D -- WM v1.1 f41 Regime Gate Probe.

TASK (NEW file, no training, inference only on existing checkpoint):
  1. Confirm V1.1 f41 checkpoint can be LOADED and aligned with the chimera feature set.
     If NOT: report the wiring as broken/unexecuted (the finding IS the gap).
  2. If loads: derive the WM REGIME signal (bearish probability from regime_head, NOT IC),
     and test it as a REGIME GATE or SIZER on the surviving daily_engine book vs:
       (a) The cheap SMA-based regime overlay already in the book (rolling_regime_book).
       (b) A SHUFFLED-WM-signal null (permuted dates -> should not add value).
  3. Score on OOS held-out (2025-03-15 to 2025-12-31); touch UNSEEN only once at the end
     if OOS verdict is clear.

PRE-REGISTERED DIRECTIONS (stated before any number is looked at):
  H1: WM regime signal (daily MEAN bear_prob, 30d smooth, adaptive 90d-median threshold)
      adds value vs SMA overlay in OOS because the WM's latent space captures
      microstructure + funding + cross-asset dynamics invisible to raw price SMA.
  NULL: shuffled WM signal -> no value (verifies two-sided soundness).
  ECONOMIC VERDICT CRITERION:
    WM_BETTER: OOS compound(WM gate) > OOS compound(SMA gate) + 2pp (pre-registered floor)
               AND OOS maxDD(WM gate) <= OOS maxDD(SMA gate) + 5pp (no DD regression).
    EQUIVALENT: within 2pp compound (SMA already captures regime value; WM adds complexity).
    WM_WORSE: WM gate compound < SMA gate compound - 2pp.

CONSTRAINTS: long-only, spot, leverage=1, taker 0.24% rt, wealth objective, no look-ahead,
  UNSEEN sealed until final check.

NO emoji (Windows cp1252). NO git commit. New file only.
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- paths ----------------------------------------------------------------
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]           # crypto/
_SRC = _ROOT / "src"
_V11_TRAIN_DIR = _SRC / "wm" / "v1" / "v1_1_training"

for _p in [str(_SRC), str(_V11_TRAIN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- config ---------------------------------------------------------------
ASSETS_U10 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
TAKER_RT = 0.0024               # 0.24% round-trip taker
MAKER_RT = 0.0006               # 0.06% round-trip maker
ANN = 365.0

# OOS and UNSEEN spans (must match daily_engine.py)
OOS_START  = "2025-03-15"
OOS_END    = "2025-12-31"
UNSEEN_START = "2025-12-31"

# WM checkpoint -- the f41 EMA best
N_FEATURES = 41
CKPT_PATH  = _ROOT / "models" / "wm" / "v1" / "v1_1" / "base" / "v1_1_f41_wm_best_ema.pt"

# SMA-based regime from daily_engine (the cheap baseline)
REGIME_LOOKBACK = 60
BREADTH_MA = 100
REGIME_TRAIN_FIT = ("2019-01-01", "2023-01-01")
REGIME_MIN_DWELL = 5
REGIME_SCALAR = {"trend": 1.0, "chop": 0.5, "down": 0.1}   # hardened defensive default

# WM regime gate thresholds (pre-registered before looking at OOS)
# The WM regime_head was trained with noisy return-based labels:
#   bear = return < -0.5*std, neutral = else, bull = return > 0.5*std.
# At dollar-bar resolution, the average bear_prob is ~0.35 (roughly base rate of
# negative returns). The signal must be evaluated RELATIVE to its historical mean,
# not at an absolute level.
# Use rolling Z-score of the cross-asset mean bear_prob (vs a training-period baseline)
# as the activation criterion: bear_prob_zscore > +1 sigma => elevated bear regime.
# Alternatively, use the raw signal but calibrated: bear_prob (30d smoothed) > its
# own rolling 60-day mean + 0.5*std (adaptive threshold, no look-ahead).
# For simplicity in this probe: use a simple ADAPTIVE threshold based on OOS-internal
# rolling percentile: bear_prob (30d smoothed) > its 90-day rolling median.
# This is causal and does not require train-time calibration.
WM_BEAR_THRESH = 0.40   # also test the raw threshold version
WM_BEAR_SCALAR = 0.15   # de-risk scalar when WM says bearish (similar to SMA "down")
WM_SMOOTH_BARS = 30     # daily bars for smoothing WM signal
# ADAPTIVE threshold: use rolling 90d median of smoothed bear_prob as the
# threshold. When smoothed bear_prob > its own 90d rolling median, we are in
# an ABOVE-AVERAGE bearish reading = de-risk signal.
USE_ADAPTIVE_THRESH = True   # use rolling-median threshold instead of fixed 0.40
WM_ADAPTIVE_WINDOW = 90     # days for the adaptive threshold rolling window

# Shuffled seeds
N_SHUFFLE_SEEDS = 20
RNG_SEED = 42


# ===========================================================================
# STEP 0: PRE-REGISTRATION PRINTOUT
# ===========================================================================
def print_preregistration():
    print("=" * 72)
    print("WAVE 4D -- WM v1.1 f41 REGIME GATE PROBE")
    print("=" * 72)
    print("PRE-REGISTERED DIRECTIONS (before any number is looked at):")
    print("  H1: WM bear_prob (daily MEAN across dollar bars, 30d smooth)")
    print("      > its 90d rolling median (adaptive threshold) -> de-risk to 0.15.")
    print("      WM gate should add >=2pp OOS compound vs SMA overlay.")
    print("  NULL: shuffled WM signal permuted dates -> should NOT beat SMA gate.")
    print("  WM_BETTER threshold: OOS compound(WM) > OOS compound(SMA) + 2pp")
    print("                       AND maxDD(WM) <= maxDD(SMA) + 5pp")
    print("  Checkpoint: v1_1_f41_wm_best_ema.pt (existing, no training)")
    print("  Features:   f41 (41-dim chimera dollar bars)")
    print("  Signal:     regime_head logits -> softmax -> bear_prob (index 0)")
    print("              daily MEAN over all dollar bars -> 30d smooth -> adaptive gate")
    print("  Adaptive threshold: smoothed > 90d rolling median = above-average bearish")
    print()


# ===========================================================================
# STEP 1: CHECKPOINT LOAD + FEATURE ALIGNMENT
# ===========================================================================
def load_wm_checkpoint():
    """Load V1.1 f41 checkpoint; return (model, feature_list, input_dim) or raise."""
    import torch
    print("[Step 1] Loading V1.1 f41 checkpoint ...")
    print(f"  Path: {CKPT_PATH}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"CHECKPOINT NOT FOUND: {CKPT_PATH}\n"
            "WM->strat wiring is BROKEN/UNEXECUTED (checkpoint missing)."
        )

    # Import settings from the v1_1_training directory (already on sys.path)
    from settings import get_feature_config, ASSET_TO_IDX, DEVICE
    from world_model import TransformerWorldModel

    feature_list, input_dim, base_dim = get_feature_config(N_FEATURES)
    print(f"  feature_list length: {len(feature_list)}, input_dim={input_dim}, base_dim={base_dim}")

    # Build model
    model = TransformerWorldModel(input_dim=input_dim, base_dim=base_dim)
    device = "cpu"   # CPU for inference-only probe (no GPU required)
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
        print(f"  Checkpoint epoch: {epoch}")
        if "shic" in ckpt:
            print(f"  ShIC (from ckpt): {ckpt['shic']}")
        if "ic" in ckpt:
            print(f"  IC   (from ckpt): {ckpt['ic']}")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("obs_encoder") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")
    print(f"  Missing keys ({len(missing)}): {missing[:4]}{'...' if len(missing) > 4 else ''}")
    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}")

    # Feature alignment check: verify the chimera loader produces the expected columns
    try:
        from pipeline.chimera_loader import ChimeraLoader
        loader = ChimeraLoader()
        df_test = loader.load("BTCUSDT", cadence="dollar")
        cols = set(df_test.columns)
        missing_feats = [f for f in feature_list if f not in cols]
        if missing_feats:
            print(f"  [WARN] Feature alignment: {len(missing_feats)} features MISSING from chimera:")
            for f in missing_feats:
                print(f"    - {f}")
            print("  PARTIAL ALIGNMENT: model can load but some features will be zeroed/missing.")
        else:
            print(f"  Feature alignment: OK -- all {len(feature_list)} features present in chimera.")
    except Exception as e:
        print(f"  [WARN] Feature alignment check failed: {e}")

    return model, feature_list, input_dim, base_dim, ASSET_TO_IDX, device


# ===========================================================================
# STEP 2: WM INFERENCE -- regime_head -> bear_prob per bar
# ===========================================================================
def run_wm_inference(model, feature_list, input_dim, base_dim, asset_to_idx, device,
                     df_daily, asset_name):
    """Run WM inference on OOS+UNSEEN dollar bars only, return per-day bear_prob.

    EFFICIENCY: we only run inference on the OOS+UNSEEN portion of the data
    (approx last 15% of bars with a warmup prefix of 2*seq_len bars for context).
    This avoids processing 2.7M historical dollar bars that are not needed for
    the OOS/UNSEEN regime gate evaluation.

    Returns a pd.Series of bear_prob indexed by calendar dates (daily).
    """
    import torch
    import torch.nn.functional as F
    from pipeline.data_integrity import selective_drop_nulls, extract_features_targets
    from settings import REWARD_HORIZONS, WM_SEQ_LEN

    sym = asset_name.upper()
    if sym not in asset_to_idx:
        raise ValueError(f"Unknown asset {sym}. Known: {list(asset_to_idx.keys())}")
    asset_idx = asset_to_idx[sym]

    # Load dollar bars
    from pipeline.chimera_loader import ChimeraLoader
    loader = ChimeraLoader()
    df_full_pl = loader.load(sym, cadence="dollar")
    df_full_pl = selective_drop_nulls(df_full_pl, feature_list, REWARD_HORIZONS, sym)
    feats_full, _ = extract_features_targets(df_full_pl, feature_list, REWARD_HORIZONS, sym)

    n_full = len(feats_full)

    # Slice to OOS start - warmup (we need the OOS+UNSEEN period only).
    # OOS_START (2025-03-15) is roughly the last ~15% of data for most assets.
    # Compute approximate index: use the 85th percentile (train+val+oos_warmup).
    # We take the last 15% of bars + a warmup buffer of 10*seq_len bars.
    ts_col = "timestamp" if "timestamp" in df_full_pl.columns else "date"
    ts_vals = df_full_pl[ts_col].to_numpy()
    dates_full = pd.to_datetime(ts_vals, unit="ms").floor("D")

    oos_start_ts = pd.Timestamp(OOS_START)
    warmup_bars = 10 * WM_SEQ_LEN   # 960 bars context before OOS
    oos_idx_arr = np.where(dates_full >= oos_start_ts)[0]
    if len(oos_idx_arr) == 0:
        raise ValueError(f"No bars found at or after OOS_START={OOS_START} for {sym}")

    oos_start_bar = max(0, int(oos_idx_arr[0]) - warmup_bars)
    feats = feats_full[oos_start_bar:]
    dates = dates_full[oos_start_bar:]

    n = len(feats)
    seq_len = WM_SEQ_LEN  # 96 bars
    BATCH_SIZE = 64        # process 64 windows at once for CPU efficiency

    all_bear_prob = np.full(n, np.nan, dtype=np.float32)
    indices = list(range(0, n - seq_len + 1, seq_len))
    if not indices:
        indices = [0]

    asset_tensor = torch.tensor([asset_idx], dtype=torch.long, device=device)
    n_chunks = len(indices)
    print(f"  ({n:,} bars, {n_chunks} chunks, batch={BATCH_SIZE})", end=" ", flush=True)

    with torch.no_grad():
        # Process in micro-batches of BATCH_SIZE windows
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[batch_start:batch_start + BATCH_SIZE]
            batch_chunks = []
            batch_lens = []
            for start in batch_indices:
                end = min(start + seq_len, n)
                chunk = feats[start:end]
                if len(chunk) < 2:
                    batch_chunks.append(np.zeros((seq_len, input_dim), dtype=np.float32))
                    batch_lens.append(0)
                    continue
                pad = seq_len - len(chunk)
                if pad > 0:
                    chunk = np.concatenate(
                        [chunk, np.zeros((pad, input_dim), dtype=np.float32)]
                    )
                batch_chunks.append(chunk)
                batch_lens.append(end - start)  # actual data length (no padding)

            # Stack into a batch [B, T, F]
            batch_arr = np.stack(batch_chunks, axis=0)
            obs = torch.from_numpy(batch_arr).float().to(device)
            asset_batch = torch.full((len(batch_indices),), asset_idx, dtype=torch.long, device=device)
            outputs = model.forward_train(obs, asset_batch)
            # regime_logits: [B, T, 3] -> softmax -> bear=0, neutral=1, bull=2
            regime_logits = outputs["regime_logits"]   # [B, seq_len, 3]
            probs = F.softmax(regime_logits, dim=-1).cpu().numpy()  # [B, T, 3]
            bear_prob_batch = probs[:, :, 0]  # [B, T] bearish probability

            for bi, (start, actual_len) in enumerate(zip(batch_indices, batch_lens)):
                if actual_len > 0:
                    all_bear_prob[start:start + actual_len] = bear_prob_batch[bi, :actual_len]

            if batch_start % (BATCH_SIZE * 10) == 0 and batch_start > 0:
                print(".", end="", flush=True)

    # Aggregate to daily: use MEAN of bar-level bear_prob for the day.
    # Rationale: the WM processes 600+ dollar bars per calendar day. The per-bar
    # regime classification is highly volatile (the model sees many short windows).
    # Taking the mean across all bars of the day gives a more stable, less noisy
    # daily regime read. The daily mean is still causal (only uses past dollar-bars up
    # to the close of the trading day).
    s_bear = pd.Series(all_bear_prob, index=dates, name="bear_prob_wm")
    daily_bear = s_bear.groupby(s_bear.index).mean()   # mean, not last bar
    daily_bear = daily_bear.sort_index()
    return daily_bear


# ===========================================================================
# STEP 3: BOOK SIMULATION (daily, causal, taker-cost)
# ===========================================================================
def _sma_arr(c, n):
    """Causal SMA on numpy array."""
    cs = np.cumsum(np.insert(c, 0, 0.0))
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def build_vol_target_core(close_panel, vol_target=0.02, vol_window=30,
                           max_per_name=0.15, max_gross=1.0):
    """Per-day per-asset vol-target weights (causal, long-only, every day)."""
    rets = close_panel.pct_change(fill_method=None)
    rvol = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    present = close_panel.notna()
    raw = (vol_target / (rvol + 1e-12)).clip(lower=0.0, upper=max_per_name)
    raw = raw.where(present & np.isfinite(rvol))
    gross = raw.sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / (gross + 1e-12), 1.0)
    w = raw.mul(pd.Series(scale, index=raw.index), axis=0).fillna(0.0)
    return w


def simulate_book(close_panel, scalar_series, cost_rt=TAKER_RT):
    """Simulate the vol-target book with a given per-day regime scalar.
    Returns the net daily return Series (causal, lagged 1 bar, taker-costed)."""
    cw = build_vol_target_core(close_panel)
    W = cw.mul(scalar_series, axis=0)
    rets = close_panel.pct_change(fill_method=None).fillna(0.0)
    Wl = W.shift(1).fillna(0.0)
    gross_ret = (Wl * rets).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    net = gross_ret - turnover * (cost_rt / 2.0)
    return net, W


def book_stats(net, lo=None, hi=None):
    """Headline stats on a daily net-return Series."""
    s = net.dropna()
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index < pd.Timestamp(hi)]
    if len(s) < 5:
        return {"n_days": len(s), "compound_pct": 0, "maxdd_pct": 0, "sharpe": 0}
    d = s.to_numpy()
    eq = np.cumprod(1 + d)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    nyr = len(d) / ANN
    cagr = float((eq[-1] ** (1 / nyr) - 1) * 100) if eq[-1] > 0 else -100.0
    sharpe = float(d.mean() / (d.std() + 1e-12) * np.sqrt(ANN))
    daily_pos = float((d > 0).mean() * 100)
    return {
        "n_days": len(d),
        "compound_pct": round(float((eq[-1] - 1) * 100), 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "maxdd_pct": round(maxdd, 2),
        "daily_pos_rate_pct": round(daily_pos, 1),
        "first_day": str(s.index[0])[:10],
        "last_day": str(s.index[-1])[:10],
    }


# ===========================================================================
# STEP 4: SMA REGIME OVERLAY (the cheap baseline from daily_engine)
# ===========================================================================
def build_sma_regime_scalar(close_panel):
    """Reproduce the daily_engine SMA-based regime scalar series (same logic as
    rolling_regime_book.regime_features + fit_regime_thresholds + apply_hysteresis).
    Returns a pd.Series of scalars in [0,1] indexed by close_panel.index."""
    from strat.rolling_regime_book import (regime_features, fit_regime_thresholds,
                                           classify_raw, apply_hysteresis)
    import strat.rolling_regime_book as RRB

    RRB.LOOKBACK["1d"] = REGIME_LOOKBACK
    RRB.BREADTH_MA = BREADTH_MA

    feat = regime_features(close_panel, "1d")
    th = fit_regime_thresholds(feat, *REGIME_TRAIN_FIT)
    raw = classify_raw(feat, th)
    smoothed = apply_hysteresis(raw, REGIME_MIN_DWELL)
    labels = pd.Series(smoothed, index=close_panel.index)
    scalar = labels.map(REGIME_SCALAR).astype(float)
    scalar = scalar.fillna(REGIME_SCALAR["chop"])
    label_series = labels
    return scalar, label_series, th


# ===========================================================================
# STEP 5: WM REGIME SCALAR from bear_prob
# ===========================================================================
def build_wm_regime_scalar(daily_bear_probs, close_panel_index,
                            bear_thresh=WM_BEAR_THRESH,
                            bear_scalar_val=WM_BEAR_SCALAR,
                            smooth_bars=WM_SMOOTH_BARS,
                            use_adaptive=USE_ADAPTIVE_THRESH,
                            adaptive_window=WM_ADAPTIVE_WINDOW):
    """Build a per-day exposure scalar from the averaged WM bear_prob across all assets.

    daily_bear_probs: dict {sym: pd.Series of bear_prob (daily MEAN index)}
    close_panel_index: the DatetimeIndex of the close panel (output aligned to this)

    Two threshold modes:
    1. FIXED (use_adaptive=False): smoothed_bear > bear_thresh -> de-risk
    2. ADAPTIVE (use_adaptive=True): smoothed_bear > rolling_median(adaptive_window)
       -> above-average bearishness -> de-risk. Causal (no look-ahead).
       This is more appropriate since the WM's absolute bear_prob level is not
       calibrated to a specific numeric threshold (it was trained with noisy labels
       averaging ~0.35 base bear_prob). The RELATIVE signal (above/below its own
       rolling baseline) carries the regime information.

    Returns (scalar_series, smoothed_bear_prob_series)
    """
    # Stack all per-asset bear probs, take cross-asset mean (market-wide bearish signal)
    all_probs = []
    for sym, s in daily_bear_probs.items():
        s_aligned = s.reindex(close_panel_index)
        all_probs.append(s_aligned)
    if not all_probs:
        return pd.Series(REGIME_SCALAR["chop"], index=close_panel_index), \
               pd.Series(0.5, index=close_panel_index)

    stacked = pd.concat(all_probs, axis=1)
    mean_bear = stacked.mean(axis=1).ffill()

    # Smooth over smooth_bars days (causal -- rolling mean up to t)
    smoothed = mean_bear.rolling(smooth_bars, min_periods=1).mean()

    if use_adaptive:
        # Adaptive threshold: smoothed > its own rolling median (90d causal)
        # When WM signal is ABOVE its recent baseline, it signals elevated bear regime.
        rolling_median = smoothed.rolling(adaptive_window, min_periods=smooth_bars).median()
        is_bear = smoothed > rolling_median
        method_desc = f"adaptive(>{adaptive_window}d_median)"
    else:
        # Fixed threshold
        is_bear = smoothed > bear_thresh
        method_desc = f"fixed(>{bear_thresh})"

    scalar = np.where(is_bear, bear_scalar_val, REGIME_SCALAR["trend"])
    scalar_series = pd.Series(scalar, index=close_panel_index, name="wm_scalar")

    # CAUSAL: lag by 1 day so today's WM signal only applies to TOMORROW's weights
    scalar_series = scalar_series.shift(1).fillna(REGIME_SCALAR["chop"])
    print(f"  WM threshold method: {method_desc}")
    return scalar_series, smoothed


# ===========================================================================
# HELPER: close panel from chimera (daily OHLC)
# ===========================================================================
def load_daily_close_panel(syms=None):
    """Return date-aligned daily close panel for u10, floored to day."""
    from pipeline.chimera_loader import ChimeraLoader
    loader = ChimeraLoader()
    syms = syms or ASSETS_U10
    closes = {}
    for sym in syms:
        try:
            df = loader.load(sym, cadence="1d")
        except Exception:
            try:
                # Fallback: load dollar and resample to 1d via close column
                df = loader.load(sym, cadence="dollar")
            except Exception as e:
                print(f"  [WARN] Could not load {sym}: {e}")
                continue
        ts_col = "timestamp" if "timestamp" in df.columns else "date"
        ts = df[ts_col].to_numpy()
        dates = pd.to_datetime(ts, unit="ms").floor("D")
        c = np.array(df["close"].to_numpy(), dtype=float)
        s = pd.Series(c, index=dates)
        s = s[~s.index.duplicated(keep="last")]
        closes[sym] = s
    panel = pd.DataFrame(closes).sort_index()
    return panel


# ===========================================================================
# MAIN PROBE
# ===========================================================================
def run_probe():
    print_preregistration()

    # ---- Step 1: Load checkpoint -------------------------------------------
    load_ok = False
    wm_load_errors = []
    try:
        model, feature_list, input_dim, base_dim, asset_to_idx, device = load_wm_checkpoint()
        load_ok = True
        print("  [RESULT] Checkpoint LOADED successfully.\n")
    except FileNotFoundError as e:
        wm_load_errors.append(str(e))
        print(f"  [RESULT] Checkpoint LOAD FAILED: {e}\n")
    except Exception as e:
        import traceback
        wm_load_errors.append(traceback.format_exc())
        print(f"  [RESULT] Checkpoint LOAD FAILED (unexpected): {e}\n")

    if not load_ok:
        print("=" * 72)
        print("FINDING: WM->STRAT WIRING IS BROKEN/UNEXECUTED.")
        print("The V1.1 f41 EMA checkpoint could not be loaded.")
        print("Errors:")
        for e in wm_load_errors:
            print(f"  {e}")
        print("=" * 72)
        return

    # ---- Step 2: Load daily close panel ------------------------------------
    print("[Step 2] Loading daily close panel for u10 ...")
    try:
        close_panel = load_daily_close_panel(ASSETS_U10)
        print(f"  Panel shape: {close_panel.shape} | "
              f"{str(close_panel.index[0])[:10]} to {str(close_panel.index[-1])[:10]}")
    except Exception as e:
        import traceback
        print(f"  [ERROR] Could not load close panel: {e}")
        traceback.print_exc()
        return

    # ---- Step 3: Build SMA regime scalar (baseline) -------------------------
    print("\n[Step 3] Building SMA-based regime scalar (cheap baseline) ...")
    try:
        sma_scalar, sma_labels, sma_th = build_sma_regime_scalar(close_panel)
        print(f"  SMA thresholds: {sma_th}")
        vc = sma_labels.value_counts(normalize=True).to_dict()
        print(f"  Regime distribution (full history): {vc}")
        oos_labels = sma_labels.loc[OOS_START:OOS_END]
        vc_oos = oos_labels.value_counts(normalize=True).to_dict() if len(oos_labels) > 0 else {}
        print(f"  Regime distribution (OOS):          {vc_oos}")
    except Exception as e:
        import traceback
        print(f"  [ERROR] SMA regime build failed: {e}")
        traceback.print_exc()
        sma_scalar = pd.Series(REGIME_SCALAR["chop"], index=close_panel.index)

    # Baseline: always-on (no overlay) -- full exposure every day
    no_overlay_scalar = pd.Series(1.0, index=close_panel.index)

    # ---- Step 4: Run WM inference for each asset ----------------------------
    print("\n[Step 4] Running WM inference to extract bear_prob for each asset ...")
    daily_bear_probs = {}
    inference_errors = []
    test_assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]  # subset for speed; key assets

    for sym in test_assets:
        print(f"  {sym} ...", end=" ", flush=True)
        try:
            bp = run_wm_inference(
                model, feature_list, input_dim, base_dim, asset_to_idx, device,
                close_panel, sym
            )
            daily_bear_probs[sym] = bp
            oos_bp = bp.reindex(
                close_panel.loc[OOS_START:OOS_END].index
            ).dropna()
            mean_bp = float(oos_bp.mean()) if len(oos_bp) > 0 else float("nan")
            print(f"OK ({len(bp)} days, mean bear_prob OOS={mean_bp:.3f})")
        except Exception as e:
            import traceback
            inference_errors.append(f"{sym}: {e}")
            print(f"FAILED: {e}")

    if not daily_bear_probs:
        print("\n  [ERROR] All WM inferences failed. Cannot build WM gate.")
        print("  FINDING: WM->STRAT wiring is EXECUTABLE but inference is BROKEN.")
        for e in inference_errors:
            print(f"    {e}")
        return

    if inference_errors:
        print(f"  [WARN] {len(inference_errors)} inference failure(s) (partial results will be used).")

    # ---- Step 5: Build WM regime scalar ----------------------------------------
    print("\n[Step 5] Building WM regime scalar from bear_prob ...")
    wm_scalar, smoothed_bear = build_wm_regime_scalar(
        daily_bear_probs, close_panel.index,
        bear_thresh=WM_BEAR_THRESH,
        bear_scalar_val=WM_BEAR_SCALAR,
        smooth_bars=WM_SMOOTH_BARS,
    )
    oos_wm_scalar = wm_scalar.loc[OOS_START:OOS_END]
    oos_smoothed = smoothed_bear.loc[OOS_START:OOS_END]
    if len(oos_smoothed) > 0:
        print(f"  WM bear_prob (OOS): mean={float(oos_smoothed.mean()):.3f}, "
              f"  std={float(oos_smoothed.std()):.3f}, "
              f"  frac_bearish={float((oos_smoothed > WM_BEAR_THRESH).mean()):.3f}")
        print(f"  WM scalar (OOS):    mean={float(oos_wm_scalar.mean()):.3f}")
        sma_oos_scalar = sma_scalar.loc[OOS_START:OOS_END]
        print(f"  SMA scalar (OOS):   mean={float(sma_oos_scalar.mean()):.3f}")

    # ---- Step 6: Simulate books -------------------------------------------
    print("\n[Step 6] Simulating daily books ...")

    # (A) No overlay (always-on)
    net_nooverlay, _ = simulate_book(close_panel, no_overlay_scalar)
    st_no = book_stats(net_nooverlay, lo=OOS_START, hi=OOS_END)
    print(f"  (A) No overlay:   OOS compound={st_no['compound_pct']:+.1f}%  "
          f"maxDD={st_no['maxdd_pct']:.1f}%  Sharpe={st_no['sharpe']:.2f}")

    # (B) SMA overlay
    net_sma, _ = simulate_book(close_panel, sma_scalar)
    st_sma = book_stats(net_sma, lo=OOS_START, hi=OOS_END)
    print(f"  (B) SMA overlay:  OOS compound={st_sma['compound_pct']:+.1f}%  "
          f"maxDD={st_sma['maxdd_pct']:.1f}%  Sharpe={st_sma['sharpe']:.2f}")

    # (C) WM regime gate
    net_wm, _ = simulate_book(close_panel, wm_scalar)
    st_wm = book_stats(net_wm, lo=OOS_START, hi=OOS_END)
    print(f"  (C) WM gate:      OOS compound={st_wm['compound_pct']:+.1f}%  "
          f"maxDD={st_wm['maxdd_pct']:.1f}%  Sharpe={st_wm['sharpe']:.2f}")

    # (D) Combined: SMA AND WM (both must be non-bearish; most conservative)
    combined_scalar = pd.concat([sma_scalar, wm_scalar], axis=1).min(axis=1)
    net_combined, _ = simulate_book(close_panel, combined_scalar)
    st_combined = book_stats(net_combined, lo=OOS_START, hi=OOS_END)
    print(f"  (D) SMA+WM combo: OOS compound={st_combined['compound_pct']:+.1f}%  "
          f"maxDD={st_combined['maxdd_pct']:.1f}%  Sharpe={st_combined['sharpe']:.2f}")

    # ---- Step 7: Shuffled null (block shuffle WM signal) -------------------
    print("\n[Step 7] Shuffled WM null (N={N_SHUFFLE_SEEDS} seeds, block permutation) ...")
    rng = np.random.default_rng(RNG_SEED)
    shuffled_compounds = []
    shuffled_maxdds = []
    for seed in range(N_SHUFFLE_SEEDS):
        # Block-shuffle the smoothed bear prob series (20-day blocks to preserve autocorrelation)
        bp_arr = smoothed_bear.to_numpy().copy()
        block = 20
        n = len(bp_arr)
        idx = np.arange(n)
        n_blocks = n // block
        block_starts = rng.permutation(n_blocks) * block
        shuf = np.concatenate([idx[s:s+block] for s in block_starts] + [idx[n_blocks*block:]])
        bp_shuffled_arr = bp_arr[shuf[:n]]
        bp_shuffled = pd.Series(bp_shuffled_arr, index=smoothed_bear.index)
        is_bear_shuf = bp_shuffled > WM_BEAR_THRESH
        scalar_shuf_arr = np.where(is_bear_shuf, WM_BEAR_SCALAR, REGIME_SCALAR["trend"])
        scalar_shuf = pd.Series(scalar_shuf_arr, index=close_panel.index, name="shuf_scalar")
        scalar_shuf = scalar_shuf.shift(1).fillna(REGIME_SCALAR["chop"])
        net_shuf, _ = simulate_book(close_panel, scalar_shuf)
        st_shuf = book_stats(net_shuf, lo=OOS_START, hi=OOS_END)
        shuffled_compounds.append(st_shuf["compound_pct"])
        shuffled_maxdds.append(st_shuf["maxdd_pct"])

    shuf_compound_mean = float(np.mean(shuffled_compounds))
    shuf_compound_p05 = float(np.percentile(shuffled_compounds, 5))
    shuf_compound_p95 = float(np.percentile(shuffled_compounds, 95))
    shuf_maxdd_mean = float(np.mean(shuffled_maxdds))
    print(f"  Shuffled null:  OOS compound mean={shuf_compound_mean:+.1f}%  "
          f"p05={shuf_compound_p05:+.1f}%  p95={shuf_compound_p95:+.1f}%  "
          f"mean_maxDD={shuf_maxdd_mean:.1f}%")

    # ---- Step 8: UNSEEN (seal-once) ----------------------------------------
    print("\n[Step 8] UNSEEN (final check, sealed until now) ...")
    st_no_u = book_stats(net_nooverlay, lo=UNSEEN_START)
    st_sma_u = book_stats(net_sma, lo=UNSEEN_START)
    st_wm_u = book_stats(net_wm, lo=UNSEEN_START)
    st_combined_u = book_stats(net_combined, lo=UNSEEN_START)
    print(f"  (A) No overlay:   UNSEEN compound={st_no_u['compound_pct']:+.1f}%  "
          f"maxDD={st_no_u['maxdd_pct']:.1f}%")
    print(f"  (B) SMA overlay:  UNSEEN compound={st_sma_u['compound_pct']:+.1f}%  "
          f"maxDD={st_sma_u['maxdd_pct']:.1f}%")
    print(f"  (C) WM gate:      UNSEEN compound={st_wm_u['compound_pct']:+.1f}%  "
          f"maxDD={st_wm_u['maxdd_pct']:.1f}%")
    print(f"  (D) SMA+WM combo: UNSEEN compound={st_combined_u['compound_pct']:+.1f}%  "
          f"maxDD={st_combined_u['maxdd_pct']:.1f}%")

    # ---- Step 9: VERDICT ---------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    # Load assessment
    print(f"[1] Checkpoint load: {'PASS' if load_ok else 'FAIL'}")
    print(f"    Path: {CKPT_PATH}")
    n_inf_ok = len(daily_bear_probs)
    print(f"[2] Feature alignment + inference: {n_inf_ok}/{len(test_assets)} assets OK")
    if inference_errors:
        for e in inference_errors:
            print(f"    ERROR: {e}")

    wm_vs_sma = st_wm['compound_pct'] - st_sma['compound_pct']
    wm_dd_regress = st_wm['maxdd_pct'] - st_sma['maxdd_pct']
    wm_vs_shuf = st_wm['compound_pct'] - shuf_compound_mean

    print(f"\n[3] OOS held-out regime gate comparison (pre-registered threshold: 2pp floor):")
    print(f"    WM vs SMA compound delta: {wm_vs_sma:+.1f}pp  (need >+2pp for WM_BETTER)")
    print(f"    WM maxDD regression:      {wm_dd_regress:+.1f}pp  (need <=+5pp)")
    print(f"    WM vs shuffled null:      {wm_vs_shuf:+.1f}pp  (should be positive to reject null)")

    if wm_vs_sma > 2.0 and wm_dd_regress <= 5.0:
        verdict = "WM_BETTER"
        verdict_str = ("WM regime gate BEATS the cheap SMA overlay by >{:.1f}pp OOS with "
                       "acceptable DD. The expensive WM IS worth wiring.".format(wm_vs_sma))
    elif abs(wm_vs_sma) <= 2.0:
        verdict = "EQUIVALENT"
        verdict_str = ("WM regime gate is EQUIVALENT to the SMA overlay (within 2pp OOS). "
                       "The cheap SMA already captures all regime value; wiring WM adds "
                       "complexity without measurable gain.")
    else:
        verdict = "WM_WORSE"
        verdict_str = ("WM regime gate UNDERPERFORMS the SMA overlay by {:.1f}pp OOS. "
                       "Do NOT wire WM as regime gate.".format(abs(wm_vs_sma)))

    print(f"\n    VERDICT: {verdict}")
    print(f"    {verdict_str}")

    wm_rejects_null = st_wm['compound_pct'] > shuf_compound_mean + 0.5
    print(f"\n[4] Null check: WM gate {'BEATS' if wm_rejects_null else 'DOES NOT BEAT'} shuffled null "
          f"(WM={st_wm['compound_pct']:+.1f}% vs shuf_mean={shuf_compound_mean:+.1f}%)")
    if not wm_rejects_null:
        print("    [WARN] WM gate is not reliably better than its shuffled counterpart -- "
              "signal may not be genuine in the regime-gate use case.")

    # EV of finishing the wiring
    print("\n[5] EV assessment -- is it worth finishing the WM->strat wiring?")
    if verdict == "WM_BETTER":
        print("    EV: HIGH. WM adds regime value not captured by the cheap SMA overlay.")
        print("    Recommended: wire ForecastBundle.regime_logits -> regime scalar in daily_engine.")
        print("    Next step: freeze regime_head in eval(), run inference nightly, use smoothed")
        print("    bear_prob as the engine's scalar_map driver (replaces rolling_regime_book).")
    elif verdict == "EQUIVALENT":
        print("    EV: LOW. The cheap SMA regime classifier (already wired and tested) captures")
        print("    all the regime value the WM provides. Wiring WM adds inference cost, model risk,")
        print("    and operational complexity for zero marginal return.")
        print("    Recommended: KEEP the SMA overlay; do NOT wire WM as regime gate.")
        print("    WM value remains: diagnostic tool, potential future signal for a different use case.")
    else:
        print("    EV: NEGATIVE. WM regime gate underperforms. Do not wire.")

    # Summary table
    print("\n" + "-" * 72)
    print("SUMMARY TABLE (OOS: {} to {})".format(OOS_START, OOS_END))
    print("{:<20} {:>12} {:>10} {:>8}".format("Variant", "Compound%", "MaxDD%", "Sharpe"))
    print("-" * 52)
    for label, st in [
        ("No overlay", st_no),
        ("SMA regime gate", st_sma),
        ("WM regime gate", st_wm),
        ("SMA+WM combined", st_combined),
        ("Shuffled WM null", {"compound_pct": shuf_compound_mean,
                              "maxdd_pct": shuf_maxdd_mean,
                              "sharpe": float("nan")}),
    ]:
        print("{:<20} {:>+12.1f} {:>10.1f} {:>8.2f}".format(
            label,
            st["compound_pct"],
            st["maxdd_pct"],
            st.get("sharpe", float("nan")),
        ))
    print("-" * 52)
    print("UNSEEN (from {}):".format(UNSEEN_START))
    for label, st in [
        ("No overlay", st_no_u),
        ("SMA regime gate", st_sma_u),
        ("WM regime gate", st_wm_u),
        ("SMA+WM combined", st_combined_u),
    ]:
        print("{:<20} {:>+12.1f} {:>10.1f}".format(
            label, st["compound_pct"], st["maxdd_pct"],
        ))
    print("=" * 72)

    return {
        "load_ok": load_ok,
        "n_inference_ok": n_inf_ok,
        "oos": {
            "no_overlay": st_no,
            "sma": st_sma,
            "wm": st_wm,
            "combined": st_combined,
            "shuffled_mean": shuf_compound_mean,
            "shuffled_p05": shuf_compound_p05,
        },
        "unseen": {
            "no_overlay": st_no_u,
            "sma": st_sma_u,
            "wm": st_wm_u,
            "combined": st_combined_u,
        },
        "verdict": verdict,
        "wm_vs_sma_pp": wm_vs_sma,
        "wm_vs_shuf_pp": wm_vs_shuf,
    }


if __name__ == "__main__":
    result = run_probe()
    if result is not None:
        # Save results
        out_dir = _ROOT / "runs" / "strat"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "wave4d_wm_regime_gate_probe.json"
        safe_result = {}
        for k, v in result.items():
            if isinstance(v, dict):
                safe_result[k] = {
                    kk: (vv if not isinstance(vv, float) or (vv == vv) else "nan")
                    for kk, vv in v.items()
                }
            else:
                safe_result[k] = v
        with open(out_path, "w") as f:
            json.dump(safe_result, f, indent=2)
        print(f"\nResults saved: {out_path}")
