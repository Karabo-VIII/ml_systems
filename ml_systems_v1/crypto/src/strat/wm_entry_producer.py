"""src/strat/wm_entry_producer.py -- V1.1 World Model -> past-only boolean LONG entry.

WHAT THIS DOES
--------------
Loads a trained V1.1 WM checkpoint, runs inference (PAST-ONLY: bar t uses only
observations [0..t-1] via the causal transformer), and extracts the per-bar predicted
return at horizon h. That prediction is thresholded to produce a boolean LONG entry
column compatible with SetupHarness / entry_signal_lab.

SIGNAL LOGIC (replicating the sweep-validated signal_engine.py approach)
--------------------------------------------------------------------------
The canonical best cross-asset strategy from signal_engine.py uses:
  * h=4 predicted return rolling mean over 2d rebal window (h4_roll_rgm16)
  * h=16 predicted return as regime gate (rolling 500-bar mean > 0 = bullish regime)
  * Entry = regime_ok AND rolling_mean(h4_pred, 2d) > 0

We expose two signal modes:
  "h4_roll_rgm16"  : the sweep-validated best (rolling h4 + h16 regime gate)
  "h4_roll_simple" : rolling h4 mean only (no regime gate, for ablation)
  "h16_sign"       : raw h16 prediction > 0 (simplest possible threshold)

All are PAST-ONLY: the WM sees obs[0..t-1] at bar t (causal shift built into
forward_train), so pred[t] is a function of strictly past observations.

LOOK-AHEAD CONTROLS
--------------------
1. Model forward uses the CAUSAL shift (obs_emb_shifted in world_model.py line 262):
   prediction at position t is computed from the transformer output for position t,
   which was built from obs[:, :t, :] (shifted-by-one encoding). Structurally past-only.
2. We pass observations in non-overlapping windows of SEQ_LEN=96. The first SEQ_LEN
   predictions of each window are warmed-up from scratch (h_seq initialized to zero),
   so the first few bars of each chunk are conservative estimates -- not inflated.
3. The entry_col is booleanized BEFORE being passed to SetupHarness, which fills at
   NEXT-BAR-OPEN (Pattern-T safe). No same-bar fill.

UNSEEN SEGMENT HANDLING
------------------------
The caller is responsible for slicing to the UNSEEN segment (90%+purge_gap to end).
This function is agnostic: pass any df_unseen slice and it will run inference on it.
The 50/20/20/10 split is enforced by the evaluation script (wm_value_probe.py).

USAGE
-----
    from strat.wm_entry_producer import WMEntryProducer
    producer = WMEntryProducer()  # loads best EMA checkpoint
    entry_col = producer.produce(df_unseen, 'BTCUSDT', mode='h4_roll_rgm16')
    # entry_col is a boolean np.ndarray of shape (len(df_unseen),)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# Path setup: support running from project root or src/
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_V11_TRAIN_DIR = _PROJECT_ROOT / "src" / "wm" / "v1" / "v1_1_training"
for _p in [str(_PROJECT_ROOT / "src"), str(_V11_TRAIN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

__contract__ = {
    "kind": "wm_entry_producer",
    "version": "1.0",
    "inputs": ["df(polars or pandas with chimera feature columns)", "asset_name str", "mode str"],
    "outputs": ["entry_col: np.ndarray bool (True = setup confirmed at close of this bar)"],
    "invariants": [
        "CAUSAL: prediction at bar t uses only obs[0..t-1] (causal shift in world_model.py)",
        "NO look-ahead: inference window-start re-initializes h_seq from zeros (conservative warm-up)",
        "entry_col is boolean -- SetupHarness fills at NEXT-BAR-OPEN (Pattern-T safe)",
        "UNSEEN segment determination is the CALLER's responsibility (this function is segment-agnostic)",
    ],
}

_SignalMode = Literal["h4_roll_rgm16", "h4_roll_simple", "h16_sign"]

# Rebal window in bars -- 2d rebal at dollar bars (BTC ~ 2-4 bars/hour ~ 48-96/day)
# We use a fixed 48-bar window matching ~1 day at typical BTC dollar-bar cadence.
_DEFAULT_REBAL_BARS = 48
_DEFAULT_REGIME_WINDOW = 500  # bars for h16 regime rolling mean


class WMEntryProducer:
    """Loads V1.1 checkpoint, runs inference, returns past-only boolean LONG entry."""

    def __init__(self, checkpoint_path: str | Path | None = None, n_features: int = 41,
                 device: str | None = None):
        from settings import get_feature_config, BASE_MODEL_DIR, DEVICE
        from world_model import TransformerWorldModel

        self.n_features = n_features
        feature_list, input_dim, base_dim = get_feature_config(n_features)
        self.feature_list = feature_list
        self.input_dim = input_dim
        self.base_dim = base_dim

        # Use provided device or settings default
        self.device = device or DEVICE

        # Resolve checkpoint
        if checkpoint_path is None:
            checkpoint_path = BASE_MODEL_DIR / f"v1_1_f{n_features}_wm_best_ema.pt"
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"[WMEntryProducer] Checkpoint not found: {self.checkpoint_path}")

        # Build + load model
        self.model = TransformerWorldModel(input_dim=input_dim, base_dim=base_dim).to(self.device)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith("obs_encoder") for k in ckpt):
            state_dict = ckpt
        else:
            state_dict = ckpt
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        if missing:
            print(f"  [WMEntryProducer] {len(missing)} new keys (random init)")
        if unexpected:
            print(f"  [WMEntryProducer] {len(unexpected)} unexpected keys (ignored)")

        from settings import REWARD_HORIZONS, WM_SEQ_LEN, ASSET_TO_IDX
        self.reward_horizons = REWARD_HORIZONS
        self.seq_len = WM_SEQ_LEN
        self.asset_to_idx = ASSET_TO_IDX
        print(f"  [WMEntryProducer] Loaded {self.checkpoint_path.name} on {self.device} | "
              f"f{n_features} input_dim={input_dim} base_dim={base_dim}")

    # ------------------------------------------------------------------
    # Core inference: extract per-bar predicted return for a given horizon
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_inference(self, feats: np.ndarray, asset_idx: int) -> dict[int, np.ndarray]:
        """Run inference on feature array, return {horizon: pred_array} for all horizons.

        Args:
            feats: [N, input_dim] float32 array (UNSEEN segment features)
            asset_idx: integer asset index (from ASSET_TO_IDX)

        Returns:
            dict {1: [N], 4: [N], 16: [N], 64: [N]} -- predicted returns per bar.
            Bars in the first seq_len of each non-overlapping chunk get conservative
            estimates from the zero-initialized h_seq. This is correct and conservative.
        """
        from settings import WM_SEQ_LEN

        n = len(feats)
        preds = {h: np.full(n, np.nan, dtype=np.float32) for h in self.reward_horizons}

        # Process in non-overlapping windows to avoid context contamination
        # The WM's causal shift means pred[t] uses obs[0..t-1] within the window.
        # Cross-window continuity: no RSSM state is carried across windows (conservative).
        seq_len = WM_SEQ_LEN
        indices = list(range(0, n - seq_len + 1, seq_len))
        if not indices:
            # If fewer than seq_len bars, run on whatever we have (padded)
            indices = [0]

        asset_tensor = torch.tensor([asset_idx], dtype=torch.long, device=self.device)

        for start in indices:
            end = min(start + seq_len, n)
            chunk = feats[start:end]
            if len(chunk) < 2:
                continue
            # Pad if needed (last partial window)
            pad = seq_len - len(chunk)
            if pad > 0:
                chunk = np.concatenate([chunk, np.zeros((pad, self.input_dim), dtype=np.float32)])

            obs = torch.from_numpy(chunk).unsqueeze(0).float().to(self.device)
            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                outputs = self.model.forward_train(obs, asset_tensor)

            for h in self.reward_horizons:
                logits_h = outputs["return_logits"][h]
                pred_h = self.model.bucketer.decode(logits_h).cpu().numpy().flatten()
                actual_len = min(seq_len - pad, end - start)
                preds[h][start:start + actual_len] = pred_h[:actual_len]

        return preds

    # ------------------------------------------------------------------
    # Signal conversion: predictions -> past-only boolean entry
    # ------------------------------------------------------------------
    def produce(
        self,
        df,
        asset_name: str,
        mode: _SignalMode = "h4_roll_rgm16",
        rebal_bars: int = _DEFAULT_REBAL_BARS,
        regime_window: int = _DEFAULT_REGIME_WINDOW,
    ) -> np.ndarray:
        """Produce a past-only boolean LONG entry column from WM predictions.

        Args:
            df: polars or pandas DataFrame with chimera feature columns
            asset_name: e.g. 'BTCUSDT'
            mode: signal generation mode
            rebal_bars: rolling window for signal (default ~1 day of dollar bars)
            regime_window: rolling window for h16 regime gate (default 500 bars)

        Returns:
            bool np.ndarray of shape (len(df),) -- True = "LONG setup confirmed at close of bar t"
        """
        from pipeline.data_integrity import selective_drop_nulls, extract_features_targets

        # Normalize asset name
        sym = asset_name.upper()
        if not sym.endswith("USDT"):
            sym = sym + "USDT"
        if sym not in self.asset_to_idx:
            raise ValueError(f"[WMEntryProducer] Unknown asset: {sym}. Known: {list(self.asset_to_idx.keys())}")
        asset_idx = self.asset_to_idx[sym]

        # Convert to polars if needed
        import polars as pl
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            df_pl = pl.from_pandas(df)
        else:
            df_pl = df

        # Extract features (handles null-dropping / filling)
        df_clean = selective_drop_nulls(df_pl, self.feature_list, self.reward_horizons, sym)
        feats, _ = extract_features_targets(df_clean, self.feature_list, self.reward_horizons, sym)

        n = len(feats)
        # Run inference
        preds = self.run_inference(feats, asset_idx)

        # Convert predictions to boolean entry signal
        entry = self._preds_to_entry(preds, mode=mode, rebal_bars=rebal_bars,
                                     regime_window=regime_window, n=n)
        return entry

    def _preds_to_entry(
        self, preds: dict, mode: _SignalMode, rebal_bars: int, regime_window: int, n: int
    ) -> np.ndarray:
        """Convert raw per-horizon prediction arrays to a past-only boolean entry."""
        entry = np.zeros(n, dtype=bool)

        if mode == "h4_roll_rgm16":
            # Sweep-validated: rolling mean of h4 preds + h16 regime gate
            p4 = np.nan_to_num(preds[4], 0.0)
            p16 = np.nan_to_num(preds[16], 0.0)
            cs4 = np.cumsum(np.insert(p4, 0, 0.0))
            cs16 = np.cumsum(np.insert(p16, 0, 0.0))
            rb = rebal_bars
            rw = min(regime_window, n - 1)
            for i in range(max(rb, rw), n):
                # Rolling mean of h4 signal over rebal_bars
                roll4 = (cs4[i + 1] - cs4[i + 1 - rb]) / rb
                # Regime gate: rolling mean of h16 over regime_window
                regime_avg = (cs16[i + 1] - cs16[i + 1 - rw]) / rw
                regime_ok = regime_avg > 0
                entry[i] = regime_ok and roll4 > 0

        elif mode == "h4_roll_simple":
            # No regime gate: just rolling mean of h4
            p4 = np.nan_to_num(preds[4], 0.0)
            cs4 = np.cumsum(np.insert(p4, 0, 0.0))
            rb = rebal_bars
            for i in range(rb, n):
                roll4 = (cs4[i + 1] - cs4[i + 1 - rb]) / rb
                entry[i] = roll4 > 0

        elif mode == "h16_sign":
            # Simplest: predict h16 return > 0
            p16 = np.nan_to_num(preds[16], 0.0)
            entry = p16 > 0

        else:
            raise ValueError(f"[WMEntryProducer] Unknown mode: {mode}. "
                             f"Use one of: h4_roll_rgm16, h4_roll_simple, h16_sign")

        return entry
