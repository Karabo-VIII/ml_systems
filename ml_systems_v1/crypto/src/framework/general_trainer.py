"""src/framework/general_trainer.py -- Layer-B REAL forecasting trainer.

This closes the audit's #1 overclaim: Layer B ("general WM") used to route a problem to a
LABEL STRING with NO model and NO trainer -- a stub. This module makes Layer B a GENUINE
forecasting layer: a minimal-but-real trainer that consumes GeneralAdapter output, splits it
walk-forward, trains a small CPU-friendly sequence model, and validates on a TRUE held-out
segment with shuffled-IC memorization detection.

WHAT IT IS (honest scope):
  - A MINIMAL baseline forecaster (a small GRU OR a linear/MLP-over-window head), NOT a SOTA WM.
  - Domain-AGNOSTIC: it consumes the `segments: List[dict]` contract directly (the same contract
    GeneralAdapter.to_segments() emits). No crypto coupling, no chimera import, no yaml.
  - Reuses the domain-general training spine from src/anti_fragile.py:
      WalkForwardSplitter.split_four_way  (4-way 50/20/20/10 + 400-bar purge; raises clearly on
                                           too-small data), AntifragileDataset (windowing),
      ShuffledICTracker semantics          (we compute a row-permuted shuffled IC inline so the
                                           predict_fn signature stays simple + self-contained).

WHAT IT IS NOT:
  - Not a replacement for the per-version WM zoo (src/wm/v1..v25). Those are richer (TwoHot heads,
    RSSM/JEPA latents, NCL ensembles). This is the floor: "Layer B can genuinely LEARN + VALIDATE
    on an arbitrary time-series", with a two-sided positive/negative control proving it.

THE OBJECTIVE (per CLAUDE.md + MEMORY.md):
  - The MODEL is trained to predict target_return_<h> (regression, Huber/MSE).
  - The held-out DIAGNOSTIC is IC at h (rank/pearson corr of pred vs realized on UNSEEN). Per the
    post-reset framing, per-bar IC is BANNED as a *primary trading objective* -- here it is used
    ONLY as a within-trainer learning diagnostic / control gate (exactly the >0.015 sanity role
    the project still permits). A trainer that cannot move held-out IC on a PLANTED signal is
    broken; a trainer that finds IC in PURE NOISE is broken. Those two controls are the deliverable.

No emoji (Windows cp1252). torch + numpy. Imports anti_fragile for the shared spine.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# Shared domain-general spine (splitter + dataset live here; this is the "reuse" the brief wants)
from anti_fragile import (  # noqa: E402
    WalkForwardSplitter,
    AntifragileConfig,
    AntifragileDataset,
)

# ---------------------------------------------------------------------------
# CDAP contract
# ---------------------------------------------------------------------------
__contract__ = {
    "kind": "trainer",
    "module": "Layer-B general forecasting trainer (train_layer_b)",
    "inputs": [
        "segments: List[dict] (GeneralAdapter.to_segments() output: features [N,C], "
        "target_return_<h>, asset_idx, timestamp)",
    ],
    "outputs": [
        "result dict: {held_out_ic, val_loss, shuffled_ic, n_params, train_loss, "
        "epochs_trained, horizon, model_kind, n_train_windows, n_unseen_bars, ...}",
    ],
    "invariants": {
        "domain_agnostic": "no crypto/chimera/yaml import; consumes the segments contract only",
        "walk_forward": "uses WalkForwardSplitter.split_four_way (4-way + 400-bar purge)",
        "held_out_truth": "held_out_ic is measured on the UNSEEN split, never train/val",
        "no_lookahead": "feature standardisation stats are fit on TRAIN ONLY then applied to all "
                        "splits (G-AUDIT-011: no full-history standardisation)",
        "small_model": "minimal baseline (GRU or MLP), CPU-friendly by default",
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GeneralTrainerConfig:
    """Hyperparameters for the minimal Layer-B forecaster.

    Defaults are tuned to be CPU-friendly and small-data-robust (the controls run in seconds).
    """
    seq_len: int = 32                 # window length fed to the model
    horizon: int = 1                  # which target_return_<h> to learn/evaluate
    model_kind: str = "gru"           # "gru" | "mlp" | "linear"
    hidden: int = 32                  # GRU hidden / MLP width
    n_layers: int = 1                 # GRU layers
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 8                 # early-stop patience on val IC (epochs)
    huber_delta: float = 1.0          # Huber transition point (in std-target units)
    loss: str = "huber"              # "huber" | "mse"
    stride: int = 1                   # window stride (1 = dense, max sample count on small data)
    seed: int = 0
    device: str = "cpu"               # "cpu" | "cuda" -- CPU default (small model, reproducible)
    # purge gap for the 4-way split. The crypto path uses 400 (cascading-norm window). For a
    # GENERIC / small time-series there is no 200-bar rolling z-score, so a large purge would
    # demand >2400 bars just to form non-empty splits. We expose it; the controls use a smaller
    # purge appropriate to a feed-forward/short-window model with no long-memory normalization.
    purge_gap_bars: int = 16
    verbose: bool = True


# ---------------------------------------------------------------------------
# The minimal model (small, CPU-friendly, domain-agnostic)
# ---------------------------------------------------------------------------

class _GRURegressor(nn.Module):
    """1-2 layer GRU over the window -> scalar return prediction at the last step.

    Tiny by design: input_dim -> hidden GRU -> Linear(hidden, 1).
    Predicts the target at the LAST bar of the window (causal: only past+present in the window).
    """
    def __init__(self, input_dim: int, hidden: int = 32, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, C]
        out, _ = self.gru(x)          # [B, T, H]
        last = out[:, -1, :]          # [B, H] -- prediction anchored at last bar of the window
        return self.head(last).squeeze(-1)  # [B]


class _MLPRegressor(nn.Module):
    """Flatten-the-window MLP baseline (no recurrence). input_dim*T -> hidden -> 1."""
    def __init__(self, input_dim: int, seq_len: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, C]
        return self.net(x).squeeze(-1)


class _LinearRegressor(nn.Module):
    """Pure linear over the flattened window (the simplest possible baseline)."""
    def __init__(self, input_dim: int, seq_len: int):
        super().__init__()
        self.lin = nn.Linear(input_dim * seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, C]
        B = x.shape[0]
        return self.lin(x.reshape(B, -1)).squeeze(-1)


def _build_model(kind: str, input_dim: int, cfg: GeneralTrainerConfig) -> nn.Module:
    kind = kind.lower()
    if kind == "gru":
        return _GRURegressor(input_dim, cfg.hidden, cfg.n_layers, cfg.dropout)
    if kind == "mlp":
        return _MLPRegressor(input_dim, cfg.seq_len, cfg.hidden, cfg.dropout)
    if kind == "linear":
        return _LinearRegressor(input_dim, cfg.seq_len)
    raise ValueError(f"Unknown model_kind={kind!r}; use 'gru' | 'mlp' | 'linear'.")


# ---------------------------------------------------------------------------
# Helpers: train-only standardisation (no look-ahead) + windowed batches
# ---------------------------------------------------------------------------

def _fit_feature_scaler(train_segments: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std over the TRAIN split only (G-AUDIT-011: no full-history stats).

    Returns (mean[C], std[C]) as float32. std floored at 1e-6.
    """
    feats = np.concatenate([s["features"] for s in train_segments], axis=0)
    mean = feats.mean(axis=0).astype(np.float32)
    std = feats.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _fit_target_scaler(train_segments: List[dict], horizon: int) -> float:
    """Std of the TRAIN target -- used to scale the regression target so Huber delta is meaningful.

    Returns a single float (target std), floored at 1e-8.
    """
    key = f"target_return_{horizon}"
    tgt = np.concatenate([s[key] for s in train_segments], axis=0)
    s = float(tgt.std())
    return s if s > 1e-8 else 1.0


def _apply_feature_scaler(segments: List[dict], mean: np.ndarray, std: np.ndarray) -> List[dict]:
    """Return NEW segments with features standardised by (mean,std). Targets untouched."""
    out = []
    for s in segments:
        ns = dict(s)
        ns["features"] = ((s["features"] - mean) / std).astype(np.float32)
        out.append(ns)
    return out


def _last_bar_windows(
    segments: List[dict],
    seq_len: int,
    horizon: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Materialise (X, y) where X=[M, T, C] windows and y=[M] = target at the window's LAST bar.

    Anchoring the label at the last bar makes the model causal: the window [start, start+T) only
    contains data up to and including the prediction bar. (target_return_<h> is itself the FORWARD
    return over the next h bars, computed upstream by the adapter -- so no future *features* leak.)
    """
    key = f"target_return_{horizon}"
    X_parts, y_parts = [], []
    for seg in segments:
        feats = seg["features"]
        tgt = seg[key]
        n = len(feats)
        if n < seq_len:
            continue
        # last valid window start so that (start+seq_len-1) indexes a real bar
        starts = np.arange(0, n - seq_len + 1, stride, dtype=np.int64)
        for st in starts:
            end = st + seq_len
            X_parts.append(feats[st:end])
            y_parts.append(tgt[end - 1])  # label at the LAST bar of the window
    if not X_parts:
        return np.zeros((0, seq_len, segments[0]["features"].shape[1]), np.float32), np.zeros((0,), np.float32)
    X = np.stack(X_parts).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.float32)
    return X, y


def _last_bar_timestamps(
    segments: List[dict],
    seq_len: int,
    stride: int,
) -> np.ndarray:
    """Timestamps at each window's LAST bar, aligned 1:1 with _last_bar_windows() output.

    Mirrors _last_bar_windows' iteration EXACTLY (same seg order, same start grid) so the
    returned ts[i] corresponds to (X[i], y[i]). Used for calendar grouping in the spine.
    """
    # NOTE: the index-based WalkForwardSplitter.split_four_way() does NOT carry 'timestamp'
    # into split segments (only features/asset_idx/target_return_*/regime_label). When the
    # key is absent we return an EMPTY array -- the caller (executor) then honestly marks the
    # calendar (Lens C monthly) gate as N/A rather than fabricating timestamps.
    ts_parts: List[np.ndarray] = []
    for seg in segments:
        if "timestamp" not in seg:
            return np.zeros((0,), dtype=np.int64)
        n = len(seg["features"])
        if n < seq_len:
            continue
        ts = np.asarray(seg["timestamp"])
        starts = np.arange(0, n - seq_len + 1, stride, dtype=np.int64)
        for st in starts:
            end = st + seq_len
            ts_parts.append(ts[end - 1])  # timestamp at the LAST bar of the window
    if not ts_parts:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(ts_parts, dtype=np.int64)


def _ic(pred: np.ndarray, real: np.ndarray) -> float:
    """Pearson IC of pred vs real, with finiteness + non-degenerate guards. NaN-safe -> returns 0."""
    mask = np.isfinite(pred) & np.isfinite(real)
    p, r = pred[mask], real[mask]
    if len(p) < 10 or np.std(p) < 1e-10 or np.std(r) < 1e-10:
        return 0.0
    ic = float(np.corrcoef(p, r)[0, 1])
    return ic if np.isfinite(ic) else 0.0


@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Batched forward over X=[M,T,C] -> preds [M] (numpy)."""
    model.eval()
    if len(X) == 0:
        return np.zeros((0,), np.float32)
    preds = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        preds[i:i + batch_size] = model(xb).detach().cpu().numpy().astype(np.float32)
    return preds


# ---------------------------------------------------------------------------
# THE TRAINER
# ---------------------------------------------------------------------------

def train_layer_b(
    segments: List[dict],
    model: Optional[nn.Module] = None,
    config: Optional[GeneralTrainerConfig] = None,
    return_arrays: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Train a minimal Layer-B forecaster on GeneralAdapter segments and validate on UNSEEN.

    Parameters
    ----------
    segments : List[dict]
        The GeneralAdapter.to_segments() contract: each dict has 'features' [N,C], 'asset_idx',
        'timestamp', and 'target_return_<h>' arrays.
    model : nn.Module, optional
        Provide your own model (must map [B,T,C] -> [B]). If None, a small model is built per
        config.model_kind.
    config : GeneralTrainerConfig, optional
        Hyperparameters. If None, defaults are used.
    return_arrays : bool, default False
        If True, the result dict additionally carries the raw held-out arrays under
        'unseen_pred' (model predictions on UNSEEN windows) and 'unseen_real'
        (realized target_return_<h> at each window's last bar) plus the UNSEEN window
        timestamps under 'unseen_ts'. The executor (framework.execute) uses these to run
        the returns-oriented robustness spine (src/strat/battery.py) on a sign-of-forecast
        directional strategy. ADDITIVE + backward-compatible (default off keeps the old
        scalar-only contract).
    **overrides :
        Convenience keyword overrides applied on top of config (e.g. horizon=4, max_epochs=30,
        model_kind="mlp", device="cuda"). Unknown keys raise.

    Returns
    -------
    dict with keys:
        held_out_ic   : IC on the UNSEEN split (the make-or-break number)
        val_loss      : best validation loss (scaled-target Huber/MSE)
        val_ic        : IC on the VAL split at the selected epoch
        shuffled_ic   : row-permuted shuffled IC on UNSEEN (memorization probe; ~0 expected)
        train_loss    : final train loss
        n_params      : model parameter count
        epochs_trained: epochs actually run before early-stop
        horizon, model_kind, n_train_windows, n_val_windows, n_unseen_bars, input_dim,
        seed, device, wall_time_s
        (if return_arrays=True) unseen_pred, unseen_real, unseen_ts
    """
    # ---- config resolution -------------------------------------------------
    cfg = config or GeneralTrainerConfig()
    if overrides:
        valid = set(GeneralTrainerConfig.__dataclass_fields__.keys())
        bad = set(overrides) - valid
        if bad:
            raise TypeError(f"train_layer_b got unknown override(s): {sorted(bad)}. "
                            f"Valid: {sorted(valid)}")
        cfg = GeneralTrainerConfig(**{**asdict(cfg), **overrides})

    t0 = time.time()

    # ---- reproducibility ---------------------------------------------------
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # ---- validate the segments contract (defence-in-depth at the boundary) -
    if not segments:
        raise ValueError("train_layer_b: empty segments list.")
    key = f"target_return_{cfg.horizon}"
    for i, seg in enumerate(segments):
        for req in ("features", "asset_idx", key):
            if req not in seg:
                raise KeyError(f"segment[{i}] missing required key {req!r} "
                               f"(have: {sorted(seg.keys())})")
        if seg["features"].ndim != 2:
            raise ValueError(f"segment[{i}].features must be [N,C], got ndim={seg['features'].ndim}")

    input_dim = segments[0]["features"].shape[1]

    # ---- 4-way walk-forward split (the shared spine) -----------------------
    af_cfg = AntifragileConfig()
    af_cfg.purge_gap_bars = cfg.purge_gap_bars
    splitter = WalkForwardSplitter(af_cfg)
    # split_four_way raises a CLEAR ValueError if a segment is too small (the brief's "small-data
    # guard now raises clearly"). We let that propagate -- it's the honest failure mode.
    train_segs, val_segs, oos_segs, unseen_segs = splitter.split_four_way(segments)

    if not train_segs or not val_segs or not unseen_segs:
        raise ValueError(
            f"train_layer_b: 4-way split produced empty split(s) "
            f"(train={len(train_segs)}, val={len(val_segs)}, oos={len(oos_segs)}, "
            f"unseen={len(unseen_segs)}). Provide more bars or reduce purge_gap_bars "
            f"(current={cfg.purge_gap_bars})."
        )

    # ---- train-only standardisation (no look-ahead) ------------------------
    f_mean, f_std = _fit_feature_scaler(train_segs)
    t_std = _fit_target_scaler(train_segs, cfg.horizon)

    train_n = _apply_feature_scaler(train_segs, f_mean, f_std)
    val_n = _apply_feature_scaler(val_segs, f_mean, f_std)
    unseen_n = _apply_feature_scaler(unseen_segs, f_mean, f_std)

    # ---- materialise windows ----------------------------------------------
    Xtr, ytr = _last_bar_windows(train_n, cfg.seq_len, cfg.horizon, cfg.stride)
    Xva, yva = _last_bar_windows(val_n, cfg.seq_len, cfg.horizon, cfg.stride)
    Xun, yun = _last_bar_windows(unseen_n, cfg.seq_len, cfg.horizon, cfg.stride)

    if len(Xtr) == 0 or len(Xva) == 0 or len(Xun) == 0:
        raise ValueError(
            f"train_layer_b: not enough windows after slicing "
            f"(train={len(Xtr)}, val={len(Xva)}, unseen={len(Xun)}; seq_len={cfg.seq_len}). "
            f"Reduce seq_len or provide more data."
        )

    # ---- model -------------------------------------------------------------
    if model is None:
        model = _build_model(cfg.model_kind, input_dim, cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.loss.lower() == "huber":
        loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    elif cfg.loss.lower() == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss={cfg.loss!r}; use 'huber' | 'mse'.")

    # tensors (scale targets by train std so Huber delta is in std units)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr / t_std).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_scaled = yva / t_std

    n_train = len(Xtr_t)
    idx_all = np.arange(n_train)

    best_val_ic = -np.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_left = cfg.patience
    final_train_loss = np.nan

    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.max_epochs):
        # -- train epoch (mini-batch SGD over shuffled windows) --
        model.train()
        rng.shuffle(idx_all)
        epoch_losses = []
        for bs in range(0, n_train, cfg.batch_size):
            bidx = idx_all[bs:bs + cfg.batch_size]
            xb = Xtr_t[bidx]
            yb = ytr_t[bidx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan

        # -- validation (loss + IC on contiguous val split) --
        with torch.no_grad():
            model.eval()
            vpred_scaled = model(Xva_t).detach().cpu().numpy().astype(np.float32)
            vloss = float(loss_fn(torch.from_numpy(vpred_scaled),
                                  torch.from_numpy(yva_scaled.astype(np.float32))).item())
        val_ic = _ic(vpred_scaled, yva)  # IC is scale-invariant -> compare scaled pred vs raw y

        improved = val_ic > best_val_ic + 1e-5
        if improved:
            best_val_ic = val_ic
            best_val_loss = vloss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1

        if cfg.verbose and (epoch % 5 == 0 or improved):
            print(f"    epoch {epoch:3d} | train_loss={final_train_loss:.5f} "
                  f"val_loss={vloss:.5f} val_IC={val_ic:+.4f} "
                  f"{'*' if improved else ''}")

        if patience_left <= 0:
            if cfg.verbose:
                print(f"    early-stop at epoch {epoch} (best val_IC={best_val_ic:+.4f} "
                      f"@ epoch {best_epoch})")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- held-out UNSEEN evaluation (the make-or-break number) -------------
    un_pred = _predict(model, Xun, device)
    held_out_ic = _ic(un_pred, yun)

    # UNSEEN window last-bar timestamps (aligned 1:1 with un_pred / yun). Used by the
    # returns-oriented spine (battery Lens C monthly grouping) when return_arrays=True.
    un_ts = _last_bar_timestamps(unseen_segs, cfg.seq_len, cfg.stride) if return_arrays else None

    # ---- shuffled IC on UNSEEN (memorization probe) ------------------------
    # Permute the UNSEEN targets relative to predictions: if the trainer "learned" anything that
    # survives a target permutation, the model is fitting an artefact. A correctly-behaving model
    # has shuffled_ic ~ 0 (any real signal is destroyed by the permutation).
    shuf_rng = np.random.default_rng(cfg.seed + 777)
    perm = shuf_rng.permutation(len(yun))
    shuffled_ic = _ic(un_pred, yun[perm])

    wall = time.time() - t0

    result: Dict[str, Any] = {
        "held_out_ic": held_out_ic,
        "val_ic": best_val_ic if np.isfinite(best_val_ic) else 0.0,
        "val_loss": best_val_loss if np.isfinite(best_val_loss) else float("nan"),
        "shuffled_ic": shuffled_ic,
        "train_loss": final_train_loss,
        "n_params": int(n_params),
        "epochs_trained": int(best_epoch + 1),
        "horizon": cfg.horizon,
        "model_kind": cfg.model_kind if model is None else type(model).__name__,
        "n_train_windows": int(len(Xtr)),
        "n_val_windows": int(len(Xva)),
        "n_unseen_windows": int(len(Xun)),
        "n_unseen_bars": int(sum(len(s["features"]) for s in unseen_segs)),
        "input_dim": int(input_dim),
        "seed": cfg.seed,
        "device": str(device),
        "wall_time_s": round(wall, 2),
    }
    if return_arrays:
        # Raw held-out arrays for the returns-oriented spine (battery.py). These let the
        # executor form a sign-of-forecast directional strategy: trade_ret = sign(pred)*real.
        result["unseen_pred"] = un_pred.astype(np.float32)
        result["unseen_real"] = yun.astype(np.float32)
        result["unseen_ts"] = (un_ts if un_ts is not None
                               else np.zeros((0,), dtype=np.int64))
    return result


# ---------------------------------------------------------------------------
# Synthetic data generators for the positive/negative controls
# ---------------------------------------------------------------------------

def _make_segment_from_arrays(
    features: np.ndarray,
    target: np.ndarray,
    horizons: Tuple[int, ...] = (1, 4, 16, 64),
    asset_idx: int = 0,
    asset_name: str = "synthetic",
) -> dict:
    """Pack raw arrays into the segment contract (so the trainer's boundary is exercised directly).

    The provided `target` is written to target_return_1; other horizons get zeros (the trainer
    only reads the requested horizon, default 1).
    """
    n = len(features)
    base_ms = 1_600_000_000_000
    ts = (base_ms + np.arange(n, dtype=np.int64) * 3_600_000).astype(np.int64)
    seg = {
        "asset_idx": asset_idx,
        "asset_name": asset_name,
        "timestamp": ts,
        "features": features.astype(np.float32),
    }
    for h in horizons:
        seg[f"target_return_{h}"] = (target.astype(np.float32) if h == 1
                                     else np.zeros(n, dtype=np.float32))
    return seg


def make_positive_control(
    n: int = 6000,
    n_features: int = 4,
    signal_feature: int = 0,
    lag: int = 1,
    noise_std: float = 1.0,
    seed: int = 0,
) -> List[dict]:
    """Synthesize a series where the target IS a (lagged, noisy) function of a feature.

    target[t] = beta * feature[signal_feature][t - lag] + noise[t]

    So a window ending at bar t (which CONTAINS feature[t-lag]) carries genuine information about
    target[t]. The other features are pure noise (distractors). The signal-to-noise is set so a
    small model can clearly recover it: held-out IC should land well above 0 (typically 0.4-0.7).

    Returns a one-element segments list (single instrument).
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, n_features)).astype(np.float32)
    driver = feats[:, signal_feature]
    # lagged driver -> target. beta chosen with noise_std=1 for a moderate (recoverable) SNR.
    beta = 1.0
    target = np.zeros(n, dtype=np.float32)
    target[lag:] = beta * driver[:-lag]
    target += rng.standard_normal(n).astype(np.float32) * noise_std
    return [_make_segment_from_arrays(feats, target, asset_name="positive_control")]


def make_negative_control(
    n: int = 6000,
    n_features: int = 4,
    seed: int = 0,
) -> List[dict]:
    """Synthesize PURE NOISE: target is independent of every feature.

    features ~ N(0,1) i.i.d., target ~ N(0,1) i.i.d. and INDEPENDENT. A correctly-behaving trainer
    cannot find signal here -> held-out IC must be ~0 (|IC| small). If it reports a clearly
    positive IC, the trainer is HALLUCINATING signal == BROKEN.

    Returns a one-element segments list.
    """
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((n, n_features)).astype(np.float32)
    target = rng.standard_normal(n).astype(np.float32)  # independent of feats
    return [_make_segment_from_arrays(feats, target, asset_name="negative_control")]


# ---------------------------------------------------------------------------
# RWYB: the two-sided soundness proof (positive + negative control + boundary)
# ---------------------------------------------------------------------------

# Acceptance thresholds for the controls (documented + asserted)
_POS_IC_MIN = 0.15      # positive control MUST clear this (planted signal recovered)
_NEG_IC_ABS_MAX = 0.10  # negative control MUST stay within this band of 0 (no hallucinated signal)


def run_controls(verbose: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Run the positive control, the negative control, and the GeneralAdapter boundary test.

    Returns a dict with both results + a PASS/FAIL verdict per the two-sided soundness rule:
        positive control IC > _POS_IC_MIN  AND  |negative control IC| < _NEG_IC_ABS_MAX.
    """
    out: Dict[str, Any] = {}

    # -- POSITIVE control ----------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("  [1] POSITIVE CONTROL: target = lagged(feature_0) + noise  (signal EXISTS)")
        print("=" * 70)
    pos_segs = make_positive_control(seed=seed)
    pos_cfg = GeneralTrainerConfig(seed=seed, verbose=verbose, model_kind="gru",
                                   seq_len=16, max_epochs=60, purge_gap_bars=16)
    pos = train_layer_b(pos_segs, config=pos_cfg)
    out["positive"] = pos
    if verbose:
        print(f"  -> held_out_IC = {pos['held_out_ic']:+.4f}  (must be > {_POS_IC_MIN})  "
              f"| shuffled_IC = {pos['shuffled_ic']:+.4f}  | n_params={pos['n_params']}")

    # -- NEGATIVE control ----------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("  [2] NEGATIVE CONTROL: target INDEPENDENT of all features  (PURE NOISE)")
        print("=" * 70)
    neg_segs = make_negative_control(seed=seed)
    neg_cfg = GeneralTrainerConfig(seed=seed, verbose=verbose, model_kind="gru",
                                   seq_len=16, max_epochs=60, purge_gap_bars=16)
    neg = train_layer_b(neg_segs, config=neg_cfg)
    out["negative"] = neg
    if verbose:
        print(f"  -> held_out_IC = {neg['held_out_ic']:+.4f}  (must be |IC| < {_NEG_IC_ABS_MAX})  "
              f"| shuffled_IC = {neg['shuffled_ic']:+.4f}  | n_params={neg['n_params']}")

    # -- BOUNDARY test: GeneralAdapter DataFrame -> segments -> trainer -------
    if verbose:
        print("\n" + "=" * 70)
        print("  [3] BOUNDARY: synthetic DataFrame -> GeneralAdapter.to_segments() -> train_layer_b")
        print("=" * 70)
    boundary = _boundary_test(verbose=verbose, seed=seed)
    out["boundary"] = boundary

    # -- VERDICT -------------------------------------------------------------
    pos_ok = pos["held_out_ic"] > _POS_IC_MIN
    neg_ok = abs(neg["held_out_ic"]) < _NEG_IC_ABS_MAX
    boundary_ok = bool(boundary.get("ran_ok", False))
    verdict = pos_ok and neg_ok and boundary_ok

    out["verdict"] = {
        "positive_control_pass": bool(pos_ok),
        "negative_control_pass": bool(neg_ok),
        "boundary_pass": bool(boundary_ok),
        "overall_pass": bool(verdict),
        "thresholds": {"pos_ic_min": _POS_IC_MIN, "neg_ic_abs_max": _NEG_IC_ABS_MAX},
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  VERDICT (two-sided soundness)")
        print("=" * 70)
        print(f"    positive control  : IC={pos['held_out_ic']:+.4f}  "
              f"{'PASS' if pos_ok else 'FAIL'} (learns planted signal, > {_POS_IC_MIN})")
        print(f"    negative control  : IC={neg['held_out_ic']:+.4f}  "
              f"{'PASS' if neg_ok else 'FAIL'} (no hallucinated signal, |IC| < {_NEG_IC_ABS_MAX})")
        print(f"    boundary test     : {'PASS' if boundary_ok else 'FAIL'} "
              f"(adapter segments -> trainer; held_out_IC={boundary.get('held_out_ic', float('nan')):+.4f})")
        print(f"    OVERALL           : {'PASS' if verdict else 'FAIL'}")
        if not verdict:
            print("    !! A trainer that passes positive but fails negative FINDS SIGNAL IN NOISE "
                  "== BROKEN. Report honestly.")
    return out


def _boundary_test(verbose: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Build a synthetic pandas DataFrame, run GeneralAdapter.to_segments(), feed train_layer_b.

    Proves the adapter -> trainer boundary works end-to-end (the brief's RWYB step 3). Uses a
    PLANTED signal (lagged feature) so we also confirm a positive IC flows through the real
    adapter path (not just the in-module synthetic segment packer).
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        return {"ran_ok": False, "error": f"pandas unavailable: {exc}"}

    from framework.general_adapter import GeneralAdapter

    n = 6000
    rng = np.random.default_rng(seed + 5)
    f0 = rng.standard_normal(n).astype(np.float32)   # the driver
    f1 = rng.standard_normal(n).astype(np.float32)   # distractor
    f2 = rng.standard_normal(n).astype(np.float32)   # distractor
    lag = 1
    y = np.zeros(n, dtype=np.float32)
    y[lag:] = f0[:-lag]
    y += rng.standard_normal(n).astype(np.float32) * 1.0

    dates = pd.date_range("2021-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "feat_driver": f0,
        "feat_noise1": f1,
        "feat_noise2": f2,
        "y": y,
    })

    adapter = GeneralAdapter(data_source=df, target_col="y",
                             instrument="boundary_series", cadence="1h")
    segs = adapter.to_segments()
    # validate the contract explicitly (the adapter's own guarantee)
    GeneralAdapter.validate_segment(segs[0], raise_on_fail=True)

    cfg = GeneralTrainerConfig(seed=seed, verbose=verbose, model_kind="gru",
                               seq_len=16, max_epochs=50, purge_gap_bars=16)
    res = train_layer_b(segs, config=cfg)
    res["ran_ok"] = True
    res["n_segments_from_adapter"] = len(segs)
    res["adapter_input_dim"] = int(segs[0]["features"].shape[1])
    if verbose:
        print(f"  -> adapter produced {len(segs)} segment(s), input_dim="
              f"{segs[0]['features'].shape[1]}; trainer held_out_IC={res['held_out_ic']:+.4f}")
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="framework.general_trainer",
        description="Layer-B REAL forecasting trainer + positive/negative control proof.",
    )
    ap.add_argument("--controls", action="store_true",
                    help="Run the positive + negative control + boundary test and print the verdict.")
    ap.add_argument("--json-out", action="store_true", help="Emit the controls result as JSON.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="Suppress per-epoch logging.")
    args = ap.parse_args()

    if args.controls or True:  # default action = run controls (this module's whole point is the proof)
        res = run_controls(verbose=not args.quiet, seed=args.seed)
        if args.json_out:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        sys.exit(0 if res["verdict"]["overall_pass"] else 1)
