"""Teacher inference cache.

Runs every teacher (V1.x family + foundation) on a fixed sample of windows
ONCE, caches their TwoHot logits + h_seq mean-pool to disk. Subsequent
distillation training loads cached logits instead of re-running teachers
per step (~9x faster).

Output layout:
    data/_caches/distillation/
        windows.npz                 # the fixed window set: (N, S, F) features +
                                       (N,) asset_ids + (N,) target_h{1,4,16,64}
                                       + (N,) start_idx + (N,) timestamps
        teacher_<name>_logits.npz   # per-teacher: (N, H, NUM_BINS) ensemble logits

Usage:
    # 1. Fix the window set (50K windows recommended for distillation)
    python -m src.frontier_ml.distillation.teacher_inference \
        --build-windows --universe u100 --n-windows 50000

    # 2. Cache foundation logits
    python -m src.frontier_ml.distillation.teacher_inference \
        --teacher foundation --ckpt models/frontier_ml/foundation/latest.pt

    # 3. Cache V1.x teacher logits
    python -m src.frontier_ml.distillation.teacher_inference \
        --teacher v1_1 --ckpt models/v1/v1_1/best_ema.pt

The cache provides a frozen training set for the distillation loop.

NOTE: This is the SCAFFOLD. The V1.x teacher loaders are stubbed -- they need
the V1.x model class shape to match. We'll wire them when the V1 trainings
land fresh checkpoints (per user's parallel-track plan).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from frontier_ml.foundation.harmony import apply_harmony  # noqa: E402
apply_harmony(verbose=False)

from frontier_ml.foundation.backbone import FoundationBackbone, DEFAULT_CONFIG  # noqa: E402
from frontier_ml.foundation.data_loader import FoundationDataset  # noqa: E402

CACHE_DIR = _PROJECT_ROOT / "data" / "_caches" / "distillation"
HORIZONS = (1, 4, 16, 64)


# ---------------------------------------------------------------------------
# Window set builder
# ---------------------------------------------------------------------------

def build_window_set(
    universe: str = "u100",
    seq_len: int = 512,
    n_windows: int = 50_000,
    seed: int = 11,
    segment: str = "train",
) -> Path:
    """Sample N windows from the dataset's TRAIN segment and persist them.

    Saves features + targets + asset_ids + start_idx so teachers can later
    forward in batches and cache logits aligned by row-index.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / "windows.npz"

    print(f"[distill-cache] building window set: universe={universe} N={n_windows} seq={seq_len}",
          flush=True)
    ds = FoundationDataset(universe=universe, seq_len=seq_len, seed=seed,
                            horizons=HORIZONS)
    F = ds.n_features

    rng = np.random.default_rng(seed)
    feat_arr = np.empty((n_windows, seq_len, F), dtype=np.float16)
    asset_ids = np.empty(n_windows, dtype=np.int64)
    start_idx = np.empty(n_windows, dtype=np.int64)
    targets = np.empty((n_windows, len(HORIZONS)), dtype=np.float32)
    n_assets = len(ds.asset_ids)
    max_h = max(HORIZONS)

    for i in range(n_windows):
        # Choose asset with available train segment
        for _ in range(20):
            a = int(rng.integers(0, n_assets))
            if segment == "train":
                lo, hi = 0, ds.n_train_bars[a] - seq_len - max_h
            else:
                lo = ds.n_train_bars[a]
                hi = ds.n_bars[a] - seq_len - max_h
            if hi - lo > 100:
                break
        if hi - lo <= 100:
            a = 0
            lo, hi = 0, ds.n_train_bars[0] - seq_len - max_h
        s = int(rng.integers(lo, hi))
        feat_arr[i] = ds.features_arr[a][s:s+seq_len]
        asset_ids[i] = a
        start_idx[i] = s
        for hi_, h_ in enumerate(HORIZONS):
            targets[i, hi_] = ds.targets_arr[a][s + seq_len - 1, hi_]
        if (i + 1) % 5000 == 0:
            print(f"  built {i+1}/{n_windows}", flush=True)

    asset_names = np.array(ds.asset_ids, dtype=object)

    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp,
              features=feat_arr,
              asset_ids=asset_ids,
              start_idx=start_idx,
              targets=targets,
              asset_names=asset_names,
              feature_names=np.array(ds.features, dtype=object),
              horizons=np.array(HORIZONS, dtype=np.int64),
              universe=universe,
              seq_len=seq_len)
    if out.exists():
        out.unlink()
    tmp.rename(out)
    print(f"[distill-cache] saved {out}: {n_windows:,} windows  "
          f"feat {feat_arr.nbytes/1e9:.2f} GB", flush=True)
    return out


# ---------------------------------------------------------------------------
# Teacher inference -- foundation
# ---------------------------------------------------------------------------

def cache_foundation(ckpt_path: Path, batch_size: int = 16) -> Path:
    """Cache TwoHot logits for the foundation backbone."""
    win_path = CACHE_DIR / "windows.npz"
    if not win_path.exists():
        raise FileNotFoundError(f"window set not built: {win_path} (run --build-windows)")
    data = np.load(win_path, allow_pickle=True)
    features = data["features"]              # (N, S, F) fp16
    asset_ids = data["asset_ids"]
    n_features = features.shape[2]
    N = features.shape[0]

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", DEFAULT_CONFIG)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FoundationBackbone(n_features=n_features, config=cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"[distill-cache] foundation params={model.num_params():,}  step={state.get('step','?')}",
          flush=True)

    NUM_BINS = cfg["num_bins"]
    logits_out = np.empty((N, len(HORIZONS), NUM_BINS), dtype=np.float16)
    h_pool_out = np.empty((N, cfg["d_model"]), dtype=np.float16)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(features[i:i+batch_size].astype(np.float32)).to(device)
            ab = torch.from_numpy(asset_ids[i:i+batch_size]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(xb, asset_ids=ab)
            for hi_, h_ in enumerate(HORIZONS):
                logits_out[i:i+batch_size, hi_] = out["return_logits"][f"h{h_}"].float().cpu().numpy().astype(np.float16)
            h_pool_out[i:i+batch_size] = out["h_seq"].mean(dim=1).float().cpu().numpy().astype(np.float16)
            if (i // batch_size) % 50 == 0:
                rate = (i + batch_size) / max(0.1, time.time() - t0)
                print(f"  foundation cache {i+batch_size}/{N}  rate {rate:.0f} win/s",
                      flush=True)
    elapsed = time.time() - t0

    out_path = CACHE_DIR / "teacher_foundation_logits.npz"
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp,
              logits=logits_out,
              h_pool=h_pool_out,
              horizons=np.array(HORIZONS, dtype=np.int64),
              ckpt_step=int(state.get("step", -1)),
              n_windows=N)
    if out_path.exists():
        out_path.unlink()
    tmp.rename(out_path)
    print(f"[distill-cache] foundation -> {out_path}  ({elapsed:.0f}s, "
          f"logits {logits_out.nbytes/1e9:.2f} GB)", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# V1.x teacher loaders -- STUB (wire when fresh V1 ckpts arrive)
# ---------------------------------------------------------------------------

def cache_v1x_teacher(name: str, ckpt_path: Path) -> Path:
    """Stub for V1.x family teachers.

    Real wiring requires importing each version's WorldModel class and
    matching its forward signature to produce TwoHot logits at h={1,4,16,64}.
    Per the parallel-track plan, this is filled in once the V1 trainings
    finish and we have fresh ckpts under models/v1/v1_*/.

    For now: load model, run forward, expect predict_returns() to exist
    OR fall back to running the model and pulling its return_logits dict.
    """
    print(f"[distill-cache] STUB: V1.x teacher {name} cache not yet wired.")
    print(f"[distill-cache]       Wire this when V1.x retrains complete.")
    raise NotImplementedError(
        f"V1.x teacher {name} cache stub. Wire in a per-version loader once "
        "V1.x trainings finish and produce fresh checkpoints."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-windows", action="store_true",
                    help="Build the fixed window set (run once)")
    ap.add_argument("--universe", default="u100")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=50_000)
    ap.add_argument("--teacher", default=None,
                    help="One of: foundation, v1_0, v1_1, v1_4, v1_6, v3, v4, v6, v11, v12, v14")
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint path for the teacher")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.build_windows:
        build_window_set(
            universe=args.universe,
            seq_len=args.seq_len,
            n_windows=args.n_windows,
        )

    if args.teacher:
        if args.teacher == "foundation":
            ckpt = Path(args.ckpt) if args.ckpt else (
                _PROJECT_ROOT / "models" / "frontier_ml" / "foundation" / "latest.pt"
            )
            cache_foundation(ckpt, batch_size=args.batch_size)
        else:
            ckpt = Path(args.ckpt) if args.ckpt else None
            if ckpt is None:
                raise SystemExit(f"--teacher {args.teacher} requires --ckpt")
            cache_v1x_teacher(args.teacher, ckpt)


if __name__ == "__main__":
    main()
