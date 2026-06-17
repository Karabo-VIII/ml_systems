"""
V3.x Adaptive Residual Adapter

Lightweight FiLM-based modulation layer that sits on top of a frozen V3 world model.
Provides regime-adaptive prediction corrections using:
  - Compressed V3 representations (feat bottleneck)
  - Context signals (rolling IC, prediction bias, regime, volatility)
  - Per-horizon scalar scale + shift (bounded, ~15K params)

Design principle: V3 provides deep structural knowledge (slow brain).
V3.x provides current regime adjustments (fast brain). V3.x can only
MODULATE predictions, never override them.

Classes:
  - AdaptiveResidualAdapter: Core FiLM modulation network
  - AdaptedWorldModel: Composite wrapper (frozen V3 + trainable adapter)
  - ContextComputer: Computes 12-dim context from rolling performance
  - DriftMonitor: Tracks IC degradation, triggers retraining
  - RegimeReplayBuffer: Regime-balanced sample storage
"""
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path

try:
    from settings import (
        REWARD_HORIZONS, ADAPTER_FEAT_DIM, ADAPTER_CONTEXT_DIM,
        ADAPTER_BOTTLENECK, ADAPTER_FILM_HIDDEN,
        ADAPTER_MAX_SCALE_RANGE, ADAPTER_MAX_SHIFT_INIT,
        RETURN_HEAD_DIM, NUM_BINS,
        DRIFT_WINDOW_SIZE, DRIFT_WARN_RATIO, DRIFT_RETRAIN_RATIO,
        REPLAY_BUFFER_SIZE,
    )
except ImportError:
    from v3_training.settings import (
        REWARD_HORIZONS, ADAPTER_FEAT_DIM, ADAPTER_CONTEXT_DIM,
        ADAPTER_BOTTLENECK, ADAPTER_FILM_HIDDEN,
        ADAPTER_MAX_SCALE_RANGE, ADAPTER_MAX_SHIFT_INIT,
        RETURN_HEAD_DIM, NUM_BINS,
        DRIFT_WINDOW_SIZE, DRIFT_WARN_RATIO, DRIFT_RETRAIN_RATIO,
        REPLAY_BUFFER_SIZE,
    )


# =============================================================================
# ADAPTIVE RESIDUAL ADAPTER (FiLM modulation)
# =============================================================================

class AdaptiveResidualAdapter(nn.Module):
    """
    FiLM-style modulation of V3's return trunk output.

    Takes V3's combined representation (feat) and a context vector,
    produces per-horizon scale and shift to modulate the return trunk.

    Architecture:
      feat [B,T,832] -> Linear(832, bottleneck) -> [B,T,16]
      context [B,12] -> Linear(12, 16) -> [B,16] -> expand to [B,T,16]
      cat -> [B,T,32] -> Linear(32, film_hidden) -> SiLU
      -> per-horizon: Linear(film_hidden, 2) -> (raw_scale, raw_shift)
      scale = sigmoid(raw_scale) * 2*range + (1 - range)  # [0.7, 1.3]
      shift = tanh(raw_shift) * max_shift

    ~15K parameters. Structurally cannot overfit.
    """

    def __init__(
        self,
        feat_dim: int = ADAPTER_FEAT_DIM,
        context_dim: int = ADAPTER_CONTEXT_DIM,
        bottleneck: int = ADAPTER_BOTTLENECK,
        film_hidden: int = ADAPTER_FILM_HIDDEN,
        scale_range: float = ADAPTER_MAX_SCALE_RANGE,
        shift_init: float = ADAPTER_MAX_SHIFT_INIT,
        horizons: list = None,
    ):
        super().__init__()
        self.horizons = horizons or REWARD_HORIZONS
        self.scale_range = scale_range

        # Compress V3's 832-dim representation to bottleneck
        self.feat_compress = nn.Linear(feat_dim, bottleneck, bias=False)

        # Encode context signals
        self.ctx_encoder = nn.Linear(context_dim, bottleneck, bias=False)

        # Shared FiLM generator
        self.film_net = nn.Sequential(
            nn.Linear(bottleneck * 2, film_hidden),
            nn.SiLU(),
        )

        # Per-horizon scale + shift heads (output 2 scalars each)
        self.film_heads = nn.ModuleDict({
            str(h): nn.Linear(film_hidden, 2)
            for h in self.horizons
        })

        # Learnable max shift magnitude (starts conservative, hard-clamped)
        self.max_shift = nn.Parameter(torch.tensor(shift_init))
        self.max_shift_ceiling = 0.05  # Hard upper bound to prevent runaway

        self._init_weights()

    def _init_weights(self):
        """Initialize so adapter starts as near-identity (scale~1, shift~0)."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'max_shift' not in name:
                nn.init.zeros_(p)

        # Bias film_heads so sigmoid(0) * 2*range + (1-range) = 1.0
        # sigmoid(0) = 0.5, so 0.5 * 2*0.3 + 0.7 = 1.0 (perfect)
        # shift: tanh(0) = 0. No bias needed.
        for h in self.horizons:
            nn.init.zeros_(self.film_heads[str(h)].bias)

    def forward(
        self,
        feat: torch.Tensor,
        context: torch.Tensor,
    ) -> dict:
        """
        Compute per-horizon scale and shift for trunk modulation.

        Args:
            feat:    [B, T, feat_dim] V3's cat(h_seq, z_post)
            context: [B, context_dim] rolling performance signals

        Returns:
            dict of {horizon: (scale [B,T,1], shift [B,T,1])}
        """
        B, T, _ = feat.shape

        # Compress feat and context
        f = self.feat_compress(feat)                          # [B, T, bottleneck]
        c = self.ctx_encoder(context)                         # [B, bottleneck]
        c = c.unsqueeze(1).expand(-1, T, -1)                 # [B, T, bottleneck]

        # FiLM generation
        combined = torch.cat([f, c], dim=-1)                  # [B, T, 2*bottleneck]
        h = self.film_net(combined)                            # [B, T, film_hidden]

        # Per-horizon scale and shift
        modulations = {}
        for horizon in self.horizons:
            raw = self.film_heads[str(horizon)](h)             # [B, T, 2]
            raw_scale = raw[..., 0:1]                          # [B, T, 1]
            raw_shift = raw[..., 1:2]                          # [B, T, 1]

            # Bounded scale: [1-range, 1+range]
            scale = torch.sigmoid(raw_scale) * (2 * self.scale_range) + (1 - self.scale_range)
            # Bounded shift: [-max_shift, +max_shift] with hard ceiling
            effective_shift = torch.clamp(
                F.softplus(self.max_shift), max=self.max_shift_ceiling
            )
            shift = torch.tanh(raw_shift) * effective_shift

            modulations[horizon] = (scale, shift)

        return modulations


# =============================================================================
# ADAPTED WORLD MODEL (composite: frozen V3 + adapter)
# =============================================================================

class AdaptedWorldModel(nn.Module):
    """
    Composite model: frozen V3 base + trainable adapter.

    Drop-in replacement for WaveNetGRUWorldModel in validation/inference.
    Runs V3 forward (no gradients), applies adapter modulation to the
    return trunk, then re-runs V3's frozen return heads on modulated trunk.
    """

    def __init__(self, base_model: nn.Module, adapter: AdaptiveResidualAdapter):
        super().__init__()
        self.base = base_model
        self.adapter = adapter

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        # Expose base model attributes needed by external code
        self.bucketer = self.base.bucketer

    def forward_train(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        context: torch.Tensor = None,
        masked_obs_seq: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass with adapter modulation.

        Args:
            obs_seq:     [B, T, INPUT_DIM]
            asset_id:    [B]
            context:     [B, ADAPTER_CONTEXT_DIM] or None (uses zeros)
            masked_obs_seq: [B, T, INPUT_DIM] optional (passed to base)

        Returns:
            Same output dict as base model, with adapted return_logits
        """
        # Run frozen base model
        with torch.no_grad():
            base_out = self.base.forward_train(obs_seq, asset_id, masked_obs_seq)

        # Default context: zeros (adapter acts as identity)
        if context is None:
            context = torch.zeros(
                obs_seq.size(0), ADAPTER_CONTEXT_DIM,
                device=obs_seq.device, dtype=obs_seq.dtype,
            )

        # Get feat for adapter input
        feat = torch.cat([base_out["h_seq"], base_out["z_post"]], dim=-1)

        # Compute modulations
        modulations = self.adapter(feat.detach(), context)

        # Apply modulation to trunk and re-run frozen return heads.
        # NOTE: Don't use torch.no_grad() here — return head params are already
        # frozen (requires_grad=False), but we need gradients to flow through
        # the heads back to the adapter via modulated_trunk.
        ret_trunk = base_out["ret_trunk"].detach()
        adapted_logits = {}
        for h in self.adapter.horizons:
            scale, shift = modulations[h]
            modulated_trunk = ret_trunk * scale + shift          # [B, T, RETURN_HEAD_DIM]
            adapted_logits[h] = self.base.return_heads[str(h)](modulated_trunk)

        # Return adapted outputs (everything else unchanged from base)
        return {
            "recon": base_out["recon"],
            "return_logits": adapted_logits,
            "regime_logits": base_out["regime_logits"],
            "prior_logits": base_out["prior_logits"],
            "post_logits": base_out["post_logits"],
            "h_seq": base_out["h_seq"],
            "z_post": base_out["z_post"],
            "ret_trunk": ret_trunk,
            # Adapter diagnostics
            "adapter_modulations": modulations,
            "base_return_logits": base_out["return_logits"],
        }

    def encode_sequence(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        context: torch.Tensor = None,
    ):
        """
        Encode sequence with adapted predictions.
        Compatible with validate_world.py and make_predict_fn.
        """
        outputs = self.forward_train(obs_seq, asset_id, context)
        return_preds = {}
        for h in REWARD_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
        context: torch.Tensor = None,
        mask_ratio: float = 0.0,
        block_mask: bool = False,
    ):
        """
        Compute adapter loss using TwoHot cross-entropy on adapted logits.

        Uses the same loss function as V3 (bucketer.compute_loss) for strong
        O(1) gradients through the logit space, rather than decoded scalars.
        Also adds shift regularization to prevent max_shift runaway.
        """
        outputs = self.forward_train(obs_seq, asset_id, context)

        total_loss = torch.tensor(0.0, device=obs_seq.device)
        loss_dict = {}

        for h in REWARD_HORIZONS:
            logits = outputs["return_logits"][h]  # [B, T, NUM_BINS]
            target = target_returns[h]             # [B, T]

            # TwoHot cross-entropy loss (O(1) gradients through logit space)
            l_h = self.bucketer.compute_loss(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            loss_dict[f"ret_{h}"] = l_h.item()
            total_loss = total_loss + l_h

            # Also compute base model loss for comparison
            with torch.no_grad():
                base_logits = outputs["base_return_logits"][h]
                base_l_h = self.bucketer.compute_loss(
                    base_logits.reshape(-1, base_logits.size(-1)),
                    target.reshape(-1),
                )
                loss_dict[f"base_ret_{h}"] = base_l_h.item()

        # Shift regularization: penalize large max_shift to prevent runaway
        shift_reg = 0.1 * self.adapter.max_shift.abs()
        total_loss = total_loss + shift_reg
        loss_dict["shift_reg"] = shift_reg.item()

        loss_dict["total"] = total_loss.item()

        # Adapter diagnostics
        modulations = outputs["adapter_modulations"]
        for h in REWARD_HORIZONS:
            scale, shift = modulations[h]
            loss_dict[f"scale_{h}"] = scale.mean().item()
            loss_dict[f"shift_{h}"] = shift.mean().item()

        return total_loss, loss_dict, outputs


# =============================================================================
# CONTEXT COMPUTER
# =============================================================================

class ContextComputer:
    """
    Computes the 12-dim context vector from rolling performance data.

    Context layout:
      [0:4]  Rolling IC per horizon (r1, r4, r16, r64)
      [4:8]  Rolling prediction bias per horizon
      [8:11] Regime probabilities (bear, neutral, bull)
      [11]   Volatility level (mean norm_deviation)

    All values standardized to ~[-1, 1] range.
    """

    def __init__(self, lookback: int = 2000, horizons: list = None):
        self.lookback = lookback
        self.horizons = horizons or REWARD_HORIZONS
        self.reset()

    def reset(self):
        """Clear all rolling buffers."""
        self._preds = {h: deque(maxlen=self.lookback) for h in self.horizons}
        self._actuals = {h: deque(maxlen=self.lookback) for h in self.horizons}
        self._regime_probs = deque(maxlen=self.lookback)  # [3] per entry
        self._volatility = deque(maxlen=self.lookback)

    def update(
        self,
        predictions: dict,
        actuals: dict,
        regime_probs: np.ndarray = None,
        volatility: float = None,
    ):
        """
        Add new data point to rolling buffers.

        Args:
            predictions: {horizon: float or array}
            actuals: {horizon: float or array}
            regime_probs: [3] softmax probabilities (bear, neutral, bull)
            volatility: scalar norm_deviation value
        """
        for h in self.horizons:
            if h in predictions and h in actuals:
                p = np.atleast_1d(predictions[h]).flatten()
                a = np.atleast_1d(actuals[h]).flatten()
                for pi, ai in zip(p, a):
                    if np.isfinite(pi) and np.isfinite(ai):
                        self._preds[h].append(float(pi))
                        self._actuals[h].append(float(ai))

        if regime_probs is not None:
            self._regime_probs.append(np.asarray(regime_probs, dtype=np.float32))

        if volatility is not None and np.isfinite(volatility):
            self._volatility.append(float(volatility))

    def get_context(self) -> np.ndarray:
        """
        Compute 12-dim context vector from rolling buffers.

        Returns:
            np.ndarray [12] with values in ~[-1, 1]
        """
        ctx = np.zeros(ADAPTER_CONTEXT_DIM, dtype=np.float32)

        # [0:4] Rolling IC per horizon
        for i, h in enumerate(self.horizons):
            p = np.array(list(self._preds[h]))
            a = np.array(list(self._actuals[h]))
            if len(p) > 30 and np.std(p) > 1e-10 and np.std(a) > 1e-10:
                ic = float(np.corrcoef(p, a)[0, 1])
                ctx[i] = np.clip(ic * 10.0, -1.0, 1.0)  # Scale IC (~0.03) to ~[-1,1]

        # [4:8] Rolling prediction bias per horizon
        for i, h in enumerate(self.horizons):
            p = np.array(list(self._preds[h]))
            a = np.array(list(self._actuals[h]))
            if len(p) > 30:
                bias = float(np.mean(p - a))
                ctx[4 + i] = np.clip(bias * 100.0, -1.0, 1.0)  # Scale bias to ~[-1,1]

        # [8:11] Regime probabilities
        if self._regime_probs:
            regime = np.mean(list(self._regime_probs), axis=0)
            ctx[8:11] = regime * 2.0 - 1.0  # [0,1] -> [-1,1]

        # [11] Volatility level
        if self._volatility:
            vol = float(np.mean(list(self._volatility)))
            ctx[11] = np.clip(vol, -1.0, 1.0)  # Already normalized

        return ctx

    @property
    def n_samples(self) -> int:
        """Number of samples in the shortest buffer."""
        counts = [len(self._preds[h]) for h in self.horizons]
        return min(counts) if counts else 0


# =============================================================================
# DRIFT MONITOR
# =============================================================================

class DriftMonitor:
    """
    Tracks rolling IC per horizon over time.
    Detects performance degradation and triggers adapter retraining.

    Status levels:
      OK:      IC within normal range
      WARN:    IC < warn_ratio * baseline for 3+ consecutive checks
      RETRAIN: IC < retrain_ratio * baseline
      ALERT:   IC sign flip (predicting backwards)
    """

    def __init__(
        self,
        window_size: int = DRIFT_WINDOW_SIZE,
        warn_ratio: float = DRIFT_WARN_RATIO,
        retrain_ratio: float = DRIFT_RETRAIN_RATIO,
        horizons: list = None,
    ):
        self.window_size = window_size
        self.warn_ratio = warn_ratio
        self.retrain_ratio = retrain_ratio
        self.horizons = horizons or REWARD_HORIZONS

        self.baseline_ic = {h: None for h in self.horizons}
        self.history = []  # list of (timestamp, {h: ic})
        self.warn_count = 0

    def set_baseline(self, ic_values: dict):
        """Set baseline IC from initial V3 validation."""
        for h in self.horizons:
            if h in ic_values:
                self.baseline_ic[h] = float(ic_values[h])

    def update(self, ic_values: dict, timestamp: float = None):
        """Record new IC measurement."""
        ts = timestamp or time.time()
        entry = {h: float(ic_values.get(h, 0)) for h in self.horizons}
        self.history.append((ts, entry))

    def check(self) -> tuple:
        """
        Check current drift status.

        Returns:
            (status: str, message: str)
            status is one of: 'OK', 'WARN', 'RETRAIN', 'ALERT'
        """
        if not self.history or not any(v is not None for v in self.baseline_ic.values()):
            return "OK", "Insufficient data for drift detection"

        # Get recent IC (last entry)
        _, recent_ic = self.history[-1]

        # Check for sign flip (ALERT)
        for h in self.horizons:
            baseline = self.baseline_ic.get(h)
            if baseline is not None and baseline > 0 and recent_ic.get(h, 0) < 0:
                self.warn_count = 0
                return "ALERT", f"IC sign flip at horizon {h}: baseline={baseline:.4f}, current={recent_ic[h]:.4f}"

        # Compute mean IC ratio
        ratios = []
        for h in self.horizons:
            baseline = self.baseline_ic.get(h)
            if baseline is not None and baseline > 0:
                ratios.append(recent_ic.get(h, 0) / baseline)

        if not ratios:
            return "OK", "No baseline set"

        mean_ratio = float(np.mean(ratios))

        if mean_ratio < self.retrain_ratio:
            self.warn_count = 0
            return "RETRAIN", f"IC at {mean_ratio:.1%} of baseline (threshold: {self.retrain_ratio:.0%})"

        if mean_ratio < self.warn_ratio:
            self.warn_count += 1
            if self.warn_count >= 3:
                return "RETRAIN", f"IC below {self.warn_ratio:.0%} for {self.warn_count} consecutive checks"
            return "WARN", f"IC at {mean_ratio:.1%} of baseline (warn #{self.warn_count})"

        self.warn_count = 0
        return "OK", f"IC at {mean_ratio:.1%} of baseline"

    def save_history(self, path: Path):
        """Save drift history to JSON."""
        data = {
            "baseline_ic": {str(h): v for h, v in self.baseline_ic.items() if v is not None},
            "history": [
                {"timestamp": ts, "ic": {str(h): v for h, v in ic.items()}}
                for ts, ic in self.history
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_history(self, path: Path):
        """Load drift history from JSON."""
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.baseline_ic = {int(h): v for h, v in data.get("baseline_ic", {}).items()}
        self.history = [
            (entry["timestamp"], {int(h): v for h, v in entry["ic"].items()})
            for entry in data.get("history", [])
        ]


# =============================================================================
# REGIME REPLAY BUFFER
# =============================================================================

class RegimeReplayBuffer:
    """
    Fixed-size buffer balanced across market regimes.

    Stores (feat, context, target_returns) tuples.
    When full, drops oldest samples from the majority regime.
    During adapter training, mix replay samples with recent data
    to prevent forgetting rare regimes (e.g., crashes during calm markets).
    """

    N_REGIMES = 3  # bear=0, neutral=1, bull=2

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        self.max_size = max_size
        per_regime = max_size // self.N_REGIMES
        self.buffers = {r: deque(maxlen=per_regime) for r in range(self.N_REGIMES)}

    def add(self, sample: dict, regime_label: int):
        """
        Add a sample to the appropriate regime buffer.

        Args:
            sample: dict with 'feat', 'context', 'targets' tensors
            regime_label: 0=bear, 1=neutral, 2=bull
        """
        regime_label = max(0, min(regime_label, self.N_REGIMES - 1))
        self.buffers[regime_label].append(sample)

    def sample(self, n: int, balanced: bool = True) -> list:
        """
        Sample from buffer, optionally balanced across regimes.

        Args:
            n: number of samples to return
            balanced: if True, equal samples per regime

        Returns:
            list of sample dicts
        """
        if balanced:
            per_regime = max(1, n // self.N_REGIMES)
            samples = []
            for r in range(self.N_REGIMES):
                buf = list(self.buffers[r])
                if buf:
                    indices = np.random.choice(len(buf), min(per_regime, len(buf)), replace=True)
                    samples.extend([buf[i] for i in indices])
            np.random.shuffle(samples)
            return samples[:n]
        else:
            all_samples = []
            for r in range(self.N_REGIMES):
                all_samples.extend(list(self.buffers[r]))
            if not all_samples:
                return []
            indices = np.random.choice(len(all_samples), min(n, len(all_samples)), replace=True)
            return [all_samples[i] for i in indices]

    def __len__(self):
        return sum(len(buf) for buf in self.buffers.values())

    def regime_counts(self) -> dict:
        """Return count per regime."""
        return {r: len(buf) for r, buf in self.buffers.items()}

    def save(self, path: Path):
        """Save buffer to disk (numpy arrays)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for r in range(self.N_REGIMES):
            data[f"regime_{r}"] = list(self.buffers[r])
        np.save(path, data, allow_pickle=True)

    def load(self, path: Path):
        """Load buffer from disk."""
        path = Path(path)
        if not path.exists():
            return
        data = np.load(path, allow_pickle=True).item()
        for r in range(self.N_REGIMES):
            key = f"regime_{r}"
            if key in data:
                for sample in data[key]:
                    self.buffers[r].append(sample)
