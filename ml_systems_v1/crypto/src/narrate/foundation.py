"""src/narrate/foundation.py -- the DOWNLOADED time-series foundation-model descriptive layer.

Wires the MOMENT-1 time-series foundation model (AutonLab, CMU -- arXiv:2402.03885) into the narrate
engine to ADD a model-grounded descriptive read on top of our hand-curated family reads. MOMENT is used
for DESCRIPTION, not forecasting:

  - ANOMALY: how unusual the period's return-structure is vs the asset's own history. Computed from
    MOMENT's pretrained masked-reconstruction error per rolling 512-bar return window, expressed as a
    percentile (0-100) of the history distribution. Higher = more unusual structure.
  - ANALOG: the historical window whose MOMENT embedding is most cosine-similar to the period window --
    "this period most resembles <date>".
  - VALIDATION (MANDATORY): a self-test proving MOMENT reproduces ground truth we already trust. We
    correlate the per-window MOMENT anomaly score against (a) realized volatility (norm_yz_volatility),
    (b) the presence of known liquidation events (liq_capitulation / liq_short_panic / liq_*_spike), and
    we check whether embedding-nearest windows tend to share regime_label. A PASS/WEAK/FAIL verdict is
    emitted. A downloaded model is only trusted if it agrees with our labels; WEAK/FAIL is a real finding.

DESIGN STANCE: descriptive only; never forecasts; per-window (multi-candle) structure, not per-candle.
On ANY failure (download blocked, OOM, API mismatch) returns {available: False, reason: "..."} -- the
caller already wraps this in try/except, but this layer must ALSO degrade gracefully on its own.

Series encoded: CLOSE-to-close log returns (descriptive structure, not price levels). The model is cached
at module level so repeated narrate() calls reuse one load. History is subsampled so a BTC 4h run stays
well under ~90s after the one-time model download/cache.

__contract__ = {
    "kind": "descriptive_foundation_layer",
    "inputs": ["df shim (df[col].to_numpy())", "period_mask bool[n]", "ref_mask bool[n]"],
    "outputs": ["dict: anomaly_pctile, anomaly_read, analog, validation, available"],
    "invariants": [
        "never raises -- returns {available: False, reason} on any failure",
        "descriptive only -- no forecasting, no look-ahead used for any TRADING claim",
        "validation block always present when available -- proves MOMENT agrees with our labels",
    ],
}
"""
from __future__ import annotations

import warnings

import numpy as np

# --- module-level config -----------------------------------------------------
_SEQ_LEN = 512            # MOMENT-1 native context length
_MODEL_NAME = "AutonLab/MOMENT-1-base"
_MAX_WINDOWS = 240        # cap on encoded history windows (subsample if more) -> keeps wall-clock bounded
_BATCH = 32               # forward-pass batch size (CPU-friendly)

# cached singletons (loaded once per process)
_EMBED_MODEL = None
_RECON_MODEL = None
_LOAD_ERROR = None


def _get_models():
    """Lazy-load + cache the MOMENT embedding and reconstruction pipelines. Returns (embed, recon) or raises."""
    global _EMBED_MODEL, _RECON_MODEL, _LOAD_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    if _EMBED_MODEL is not None and _RECON_MODEL is not None:
        return _EMBED_MODEL, _RECON_MODEL
    try:
        import torch  # noqa: F401
        from momentfm import MOMENTPipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            em = MOMENTPipeline.from_pretrained(_MODEL_NAME, model_kwargs={"task_name": "embedding"})
            em.init()
            em.eval()
            rc = MOMENTPipeline.from_pretrained(_MODEL_NAME, model_kwargs={"task_name": "reconstruction"})
            rc.init()
            rc.eval()
        _EMBED_MODEL, _RECON_MODEL = em, rc
        return em, rc
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = f"MOMENT load failed: {type(e).__name__}: {str(e)[:120]}"
        raise RuntimeError(_LOAD_ERROR) from e


def _log_returns(close: np.ndarray) -> np.ndarray:
    """Close-to-close log returns, length n-1, finite-cleaned."""
    c = np.asarray(close, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(np.where(c > 0, c, np.nan)))
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _build_windows(ret: np.ndarray, end_indices: np.ndarray) -> np.ndarray:
    """For each end index e (into ret), build the 512-length window ret[e-511 .. e] with left zero-pad.

    Returns array [n_windows, 512] float32.
    """
    out = np.zeros((len(end_indices), _SEQ_LEN), dtype=np.float32)
    for k, e in enumerate(end_indices):
        s = e - _SEQ_LEN + 1
        if s < 0:
            seg = ret[0:e + 1]
            out[k, _SEQ_LEN - len(seg):] = seg  # left-pad with zeros
        else:
            out[k] = ret[s:e + 1]
    return out


def _encode(model, windows: np.ndarray) -> np.ndarray:
    """MOMENT embedding for each [512] window. Returns [n, 768] float32."""
    import torch
    embs = []
    with torch.no_grad():
        for i in range(0, len(windows), _BATCH):
            chunk = windows[i:i + _BATCH]
            x = torch.from_numpy(chunk).float().unsqueeze(1)  # [b, 1, 512]
            out = model(x_enc=x)
            embs.append(out.embeddings.detach().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def _recon_error(model, windows: np.ndarray) -> np.ndarray:
    """MOMENT masked-reconstruction MSE for each window. Returns [n] float (higher = more anomalous)."""
    import torch
    errs = []
    with torch.no_grad():
        for i in range(0, len(windows), _BATCH):
            chunk = windows[i:i + _BATCH]
            x = torch.from_numpy(chunk).float().unsqueeze(1)            # [b, 1, 512]
            input_mask = torch.ones((x.shape[0], _SEQ_LEN))
            out = model(x_enc=x, input_mask=input_mask)
            rec = out.reconstruction.squeeze(1).detach().cpu().numpy()  # [b, 512]
            mse = np.mean((rec - chunk) ** 2, axis=1)
            errs.append(mse)
    return np.concatenate(errs, axis=0).astype(np.float64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN-safe; returns 0.0 if degenerate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return 0.0
    a, b = a[m], b[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _date_str(ts_ms) -> str:
    import pandas as pd
    return str(pd.to_datetime(int(ts_ms), unit="ms").date())


def foundation_layer(df, period_mask, ref_mask) -> dict:
    """MOMENT descriptive layer. Returns the dict documented at module top. NEVER raises."""
    try:
        cols = set(df.columns)
        if "close" not in cols:
            return {"available": False, "reason": "no close column"}

        close = np.asarray(df["close"].to_numpy(), dtype=np.float64)
        ts = np.asarray(df["timestamp"].to_numpy(), dtype=np.int64) if "timestamp" in cols else np.arange(len(close))
        period_mask = np.asarray(period_mask, dtype=bool)
        ref_mask = np.asarray(ref_mask, dtype=bool)
        n = len(close)
        if n < 64 or period_mask.sum() == 0:
            return {"available": False, "reason": f"too few bars (n={n}, period={int(period_mask.sum())})"}

        # returns are length n-1; index i of ret corresponds to bar i+1's close-to-close move.
        ret = _log_returns(close)              # len n-1
        n_ret = len(ret)

        # --- choose window END indices (into ret) over the reference history ---------
        ref_idx = np.where(ref_mask)[0]
        ref_idx = ref_idx[ref_idx >= 1]        # need >=1 prior bar for a return
        ref_ret_idx = ref_idx - 1              # map bar index -> ret index
        ref_ret_idx = ref_ret_idx[(ref_ret_idx >= 0) & (ref_ret_idx < n_ret)]
        if len(ref_ret_idx) < 8:
            return {"available": False, "reason": "insufficient history windows"}

        # subsample to <= _MAX_WINDOWS, evenly spaced (keep chronological coverage)
        if len(ref_ret_idx) > _MAX_WINDOWS:
            sel = np.linspace(0, len(ref_ret_idx) - 1, _MAX_WINDOWS).round().astype(int)
            hist_end = ref_ret_idx[np.unique(sel)]
        else:
            hist_end = ref_ret_idx
        # bar index that each hist window ends on (for dating + label lookups)
        hist_bar_end = hist_end + 1

        # --- the PERIOD window: ends on the last bar of the period -------------------
        period_bars = np.where(period_mask)[0]
        period_end_bar = int(period_bars[-1])
        period_ret_end = min(period_end_bar - 1, n_ret - 1)
        if period_ret_end < 0:
            return {"available": False, "reason": "period has no return bar"}

        em, rc = _get_models()

        # build + encode history windows + the period window in one pass
        all_end = np.append(hist_end, period_ret_end)
        windows = _build_windows(ret, all_end)          # [W+1, 512]
        embs = _encode(em, windows)                     # [W+1, 768]
        recon = _recon_error(rc, windows)               # [W+1]

        hist_emb, period_emb = embs[:-1], embs[-1]
        hist_recon, period_recon = recon[:-1], recon[-1]

        # --- ANOMALY: period recon error as a percentile of history recon errors ----
        anomaly_pctile = float((hist_recon < period_recon).mean() * 100.0)
        if anomaly_pctile >= 90:
            anomaly_read = "unusually structured (top-decile-novel vs this asset's history)"
        elif anomaly_pctile >= 70:
            anomaly_read = "somewhat unusual structure"
        elif anomaly_pctile <= 15:
            anomaly_read = "calm / very typical structure"
        else:
            anomaly_read = "typical structure"

        # --- ANALOG: nearest historical window by embedding cosine similarity -------
        hn = hist_emb / (np.linalg.norm(hist_emb, axis=1, keepdims=True) + 1e-9)
        pn = period_emb / (np.linalg.norm(period_emb) + 1e-9)
        cos = hn @ pn                                   # [W]
        # ANALOG must be a PAST, non-overlapping window (descriptive of history, never the future):
        # keep only windows that END at or before the period window's start (period_end_bar - _SEQ_LEN).
        past_nonoverlap = hist_bar_end <= (period_end_bar - _SEQ_LEN)
        cos_masked = np.where(past_nonoverlap, cos, -np.inf)
        if np.all(~np.isfinite(cos_masked)):
            # no clean past window (short history) -> allow any non-future window
            cos_masked = np.where(hist_bar_end < period_end_bar, cos, -np.inf)
        if np.all(~np.isfinite(cos_masked)):
            cos_masked = cos                            # last-resort fall back to full set
        a_i = int(np.argmax(cos_masked))
        analog_bar = int(hist_bar_end[a_i])
        analog_date = _date_str(ts[analog_bar]) if analog_bar < n else "n/a"
        analog = f"this period most resembles {analog_date} (cos={float(cos[a_i]):.3f})"

        # --- VALIDATION: does MOMENT reproduce what we already know? -----------------
        validation = _validate(df, cols, hist_bar_end, hist_recon, hist_emb, ts)

        return {
            "available": True,
            "model": _MODEL_NAME,
            "n_history_windows": int(len(hist_end)),
            "anomaly_pctile": round(anomaly_pctile, 1),
            "anomaly_read": anomaly_read,
            "period_recon_mse": round(float(period_recon), 6),
            "analog": analog,
            "analog_date": analog_date,
            "validation": validation,
        }
    except Exception as e:  # noqa: BLE001 -- graceful: never raise to the caller
        return {"available": False, "reason": f"{type(e).__name__}: {str(e)[:140]}"}


def _validate(df, cols, hist_bar_end, hist_recon, hist_emb, ts) -> dict:
    """Prove MOMENT agrees with our ground-truth labels. Returns numbers + a PASS/WEAK/FAIL verdict."""
    out = {}
    bars = hist_bar_end.astype(int)

    # window-level realized vol: mean norm_yz_volatility over the 512-bar window ending at each hist bar.
    def _window_mean(arr):
        v = np.asarray(arr, dtype=np.float64)
        res = np.full(len(bars), np.nan)
        for k, e in enumerate(bars):
            s = max(0, e - _SEQ_LEN + 1)
            seg = v[s:e + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                res[k] = seg.mean()
        return res

    # (a) anomaly score vs realized volatility
    corr_vol = None
    if "norm_yz_volatility" in cols:
        wvol = _window_mean(df["norm_yz_volatility"].to_numpy())
        corr_vol = _safe_corr(hist_recon, wvol)
        out["corr_anomaly_vs_realized_vol"] = round(corr_vol, 3)

    # (b) anomaly score vs presence of known liquidation events in the window
    liq_cols = [c for c in ("liq_capitulation", "liq_short_panic", "liq_short_spike", "liq_long_spike") if c in cols]
    corr_liq = None
    if liq_cols:
        # total liq-event count within the window ending at each hist bar
        stacked = np.zeros(len(ts), dtype=np.float64)
        for c in liq_cols:
            stacked = stacked + np.nan_to_num(np.asarray(df[c].to_numpy(), dtype=np.float64), nan=0.0)
        win_has_liq = np.zeros(len(bars))
        win_liq_count = np.zeros(len(bars))
        for k, e in enumerate(bars):
            s = max(0, e - _SEQ_LEN + 1)
            seg = stacked[s:e + 1]
            win_liq_count[k] = float(seg.sum())
            win_has_liq[k] = 1.0 if seg.sum() > 0 else 0.0
        # use liq COUNT density vs anomaly (richer than binary when most windows contain >=1 sparse event)
        corr_liq = _safe_corr(hist_recon, win_liq_count)
        out["corr_anomaly_vs_liq_count"] = round(corr_liq, 3)
        # also report the mean-anomaly lift for windows with above-median liq count
        med = np.median(win_liq_count)
        hi = hist_recon[win_liq_count > med]
        lo = hist_recon[win_liq_count <= med]
        if len(hi) and len(lo) and np.mean(lo) != 0:
            out["anomaly_lift_high_liq_windows"] = round(float(np.mean(hi) / (np.mean(lo) + 1e-12)), 3)

    # (c) do embedding-nearest windows share regime_label?
    regime_agree = None
    if "regime_label" in cols and len(hist_emb) >= 6:
        reg = np.asarray(df["regime_label"].to_numpy())
        reg_at = np.array([reg[b] if b < len(reg) else -99 for b in bars])
        hn = hist_emb / (np.linalg.norm(hist_emb, axis=1, keepdims=True) + 1e-9)
        sim = hn @ hn.T
        np.fill_diagonal(sim, -np.inf)
        nn = np.argmax(sim, axis=1)
        valid = (reg_at != -99) & (reg_at[nn] != -99)
        if valid.sum() >= 5:
            regime_agree = float((reg_at[valid] == reg_at[nn][valid]).mean())
            # baseline = chance agreement given the label marginal distribution
            vals, counts = np.unique(reg_at[valid], return_counts=True)
            p = counts / counts.sum()
            chance = float((p ** 2).sum())
            out["nn_regime_agreement"] = round(regime_agree, 3)
            out["nn_regime_chance_baseline"] = round(chance, 3)
            out["nn_regime_lift_over_chance"] = round(regime_agree - chance, 3)

    # --- verdict: MOMENT must AGREE with our labels to be trusted ----------------
    # Score the three checks. Anomaly should rise with vol and with liq density; nearest
    # embeddings should share regime above chance.
    passes = 0
    checks = 0
    notes = []
    if corr_vol is not None:
        checks += 1
        if corr_vol >= 0.30:
            passes += 1; notes.append(f"vol corr {corr_vol:+.2f} (strong)")
        elif corr_vol >= 0.12:
            passes += 0.5; notes.append(f"vol corr {corr_vol:+.2f} (weak-positive)")
        else:
            notes.append(f"vol corr {corr_vol:+.2f} (no agreement)")
    if corr_liq is not None:
        checks += 1
        if corr_liq >= 0.20:
            passes += 1; notes.append(f"liq corr {corr_liq:+.2f} (strong)")
        elif corr_liq >= 0.08:
            passes += 0.5; notes.append(f"liq corr {corr_liq:+.2f} (weak-positive)")
        else:
            notes.append(f"liq corr {corr_liq:+.2f} (no agreement)")
    if regime_agree is not None:
        checks += 1
        lift = out.get("nn_regime_lift_over_chance", 0.0)
        if lift >= 0.15:
            passes += 1; notes.append(f"regime NN lift {lift:+.2f} (strong)")
        elif lift >= 0.05:
            passes += 0.5; notes.append(f"regime NN lift {lift:+.2f} (weak)")
        else:
            notes.append(f"regime NN lift {lift:+.2f} (at/below chance)")

    if checks == 0:
        verdict = "FAIL"
        notes.append("no ground-truth labels available to validate against")
    else:
        frac = passes / checks
        if frac >= 0.67:
            verdict = "PASS"
        elif frac >= 0.34:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

    out["verdict"] = verdict
    out["summary"] = f"MOMENT vs our labels: {verdict} -- " + "; ".join(notes)
    out["interpretation"] = (
        "PASS = MOMENT's unsupervised anomaly/embedding structure reproduces our hand-built vol/liq/regime "
        "labels (trust the model's descriptive read). WEAK = partial agreement (use as a soft prior). "
        "FAIL = MOMENT does not reproduce our ground truth on this asset/cadence (do NOT trust its read here)."
    )
    return out
