"""V4 mini-probe: 4 options × ~400 steps on real BTC chimera data.

Compares loss trajectories + train IC for:
  A. Baseline V4 (TEMPORAL_CTX_DROP=0.15, no forecast head)
  B. ATME bumped (TEMPORAL_CTX_DROP=0.20, no forecast head)
  C. Forecast head wired (TEMPORAL_CTX_DROP=0.15, forecast MSE auxiliary loss)
  D. Both (B + C combined)

Forecast head adds: predict obs[t+h] from h_seq[t] for h in [1, 4, 16, 64].
Loss = MSE(forecast_logits[h], obs_seq.roll(-h, dim=1)). Mamba h_seq is the
state that should predict the next bar — this anchors it explicitly.

Probe is read-only re: V4 source files; it constructs the forecast heads
externally and wires the loss in the probe loop.
"""
import sys
import os
import time
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl

# === Setup paths to load V4 settings + world_model in isolation ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = PROJECT_ROOT / "src" / "wm" / "v4" / "v4_training"
SRC_DIR = PROJECT_ROOT / "src"
SHARED_DIR = PROJECT_ROOT / "src" / "wm" / "_shared"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(SHARED_DIR))
os.environ["PYTHONIOENCODING"] = "utf-8"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


settings = _load("settings", str(V4_DIR / "settings.py"))
wm_mod = _load("world_model", str(V4_DIR / "world_model.py"))
MambaWorldModel = wm_mod.MambaWorldModel


# === Load real BTC chimera data ===
def load_real_data(n_assets=3, n_seqs=200, seq_len=96):
    """Load real chimera data, return (obs_seq, asset_id, targets) tensors."""
    chimera_dir = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    assets = ["btcusdt", "ethusdt", "solusdt"][:n_assets]
    feat_list = settings.FEATURE_LIST_29
    horizons = [1, 4, 16, 64]

    all_obs, all_asset_id, all_tgt = [], [], {h: [] for h in horizons}
    for aidx, sym in enumerate(assets):
        files = sorted((chimera_dir).glob(f"{sym}_v50_chimera_*.parquet"))
        if not files:
            continue
        df = pl.read_parquet(files[-1])
        # Ensure feature columns present
        feats = df.select(feat_list).to_numpy()
        tgts = {h: df[f"target_return_{h}"].to_numpy() for h in horizons}

        # Use the first 50% (training segment) to draw probe sequences
        train_n = int(len(feats) * 0.50)
        feats = feats[:train_n]
        for h in horizons:
            tgts[h] = tgts[h][:train_n]

        n_take = min(n_seqs // n_assets, max(0, train_n - seq_len))
        if n_take <= 0:
            continue
        starts = np.linspace(0, train_n - seq_len - 1, n_take).astype(int)
        for s in starts:
            all_obs.append(feats[s:s + seq_len])
            all_asset_id.append(aidx)
            for h in horizons:
                all_tgt[h].append(tgts[h][s:s + seq_len])

    obs = torch.tensor(np.stack(all_obs), dtype=torch.float32)
    asset_id = torch.tensor(all_asset_id, dtype=torch.long)
    targets = {h: torch.tensor(np.stack(all_tgt[h]), dtype=torch.float32) for h in horizons}
    print(f"  Loaded {obs.shape[0]} sequences x {obs.shape[1]} bars x {obs.shape[2]} feats from {n_assets} assets")
    return obs, asset_id, targets


def make_forecast_heads(d_model, n_features, horizons):
    """One Linear head per horizon: predicts obs[t+h] from h_seq[t]."""
    return nn.ModuleDict({
        str(h): nn.Linear(d_model, n_features) for h in horizons
    })


def compute_train_ic(model, obs, asset_id, targets, horizons, batch_size=8):
    """Train-set IC at h=1 (just for loss-curve sanity, not held-out)."""
    model.eval()
    preds_h1, tgts_h1 = [], []
    with torch.no_grad():
        for i in range(0, len(obs), batch_size):
            x = obs[i:i+batch_size]
            a = asset_id[i:i+batch_size]
            out = model.forward_train(x, a, x, temporal_ctx_drop=0.0)
            ret_logits = out["return_logits"][1]  # [B, T, NUM_BINS]
            decoded = model.bucketer.decode(ret_logits.reshape(-1, ret_logits.shape[-1]))
            preds_h1.append(decoded.reshape(x.shape[0], x.shape[1]))
            tgts_h1.append(targets[1][i:i+batch_size])
    preds_h1 = torch.cat(preds_h1, dim=0).flatten().cpu().numpy()
    tgts_h1 = torch.cat(tgts_h1, dim=0).flatten().cpu().numpy()
    nz = np.abs(tgts_h1) > 1e-7
    if nz.sum() < 30:
        return 0.0
    return float(np.corrcoef(preds_h1[nz], tgts_h1[nz])[0, 1])


def probe_one(option_name, atme_prob, use_forecast_head, forecast_weight,
              obs, asset_id, targets, n_steps=400, batch_size=8, seed=42):
    """One probe configuration."""
    print(f"\n=== {option_name} | atme={atme_prob} | forecast={use_forecast_head} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use MUCH smaller V4 for fast probe (probe is comparing TRAJECTORIES, not absolute values)
    model = MambaWorldModel(
        input_dim=29, d_model=128, n_layers=2,  # smaller for speed
        latent_dim=24, classes=24, num_assets=3,
    )
    # Bucketer was initialized to settings.DEVICE ("cuda"); rebuild on CPU for probe
    from components import TwoHotSymlog
    model.bucketer = TwoHotSymlog(settings.NUM_BINS, settings.BIN_MIN, settings.BIN_MAX, "cpu")
    model.train()
    fc_heads = None
    if use_forecast_head:
        fc_heads = make_forecast_heads(d_model=128, n_features=29, horizons=[1, 4, 16, 64])
        fc_heads.train()

    params = list(model.parameters())
    if fc_heads is not None:
        params += list(fc_heads.parameters())
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=5e-2)

    n_seqs = obs.shape[0]
    losses, recons, fc_losses, ret_terms = [], [], [], []
    t0 = time.time()
    for step in range(n_steps):
        idx = np.random.choice(n_seqs, batch_size, replace=False)
        x = obs[idx]
        a = asset_id[idx]
        tgt = {h: targets[h][idx] for h in [1, 4, 16, 64]}

        out = model.forward_train(x, a, x, temporal_ctx_drop=atme_prob)
        # Reconstruction
        recon_loss = F.mse_loss(out["recon"], x)
        # Per-horizon return loss (Huber on decoded)
        ret_loss_total = torch.tensor(0.0)
        for h in [1, 4, 16, 64]:
            logits = out["return_logits"][h].reshape(-1, out["return_logits"][h].shape[-1])
            t_flat = tgt[h].reshape(-1)
            # Use bucketer's CE loss
            ret_loss_total = ret_loss_total + model.bucketer.compute_loss(logits, t_flat)

        # Forecast head loss
        fc_loss = torch.tensor(0.0)
        if fc_heads is not None:
            h_seq = out["h_seq"]                 # [B, T, D]
            for h in [1, 4, 16, 64]:
                # Predict obs[t+h] from h_seq[t]; only predict where t+h is in-bounds
                fc_pred = fc_heads[str(h)](h_seq[:, :-h, :])     # [B, T-h, F]
                fc_tgt = x[:, h:, :]                             # [B, T-h, F]
                fc_loss = fc_loss + F.mse_loss(fc_pred, fc_tgt)
            fc_loss = fc_loss / 4.0

        total = ret_loss_total + 0.1 * recon_loss + forecast_weight * fc_loss
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        losses.append(total.item())
        recons.append(recon_loss.item())
        ret_terms.append(ret_loss_total.item())
        fc_losses.append(fc_loss.item() if fc_heads else 0.0)

    elapsed = time.time() - t0
    # Train IC at end
    train_ic = compute_train_ic(model, obs[:48], asset_id[:48], {h: targets[h][:48] for h in [1, 4, 16, 64]}, [1])
    final_total = np.mean(losses[-30:])
    final_ret = np.mean(ret_terms[-30:])
    final_recon = np.mean(recons[-30:])
    final_fc = np.mean(fc_losses[-30:])
    init_total = np.mean(losses[:30])
    drop_pct = 100 * (1 - final_total / init_total)
    print(f"  total {init_total:.3f} -> {final_total:.3f} ({drop_pct:.1f}% drop)")
    print(f"  ret_loss={final_ret:.3f}  recon={final_recon:.4f}  forecast_mse={final_fc:.4f}")
    print(f"  train_IC@h1 = {train_ic:.4f}   ({elapsed:.0f}s, {n_steps} steps)")
    return {
        "name": option_name,
        "init_total": init_total,
        "final_total": final_total,
        "drop_pct": drop_pct,
        "final_ret": final_ret,
        "final_recon": final_recon,
        "final_fc": final_fc,
        "train_ic": train_ic,
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 70)
    print("V4 OPTIONS PROBE — real BTC/ETH/SOL chimera data, 400 steps each")
    print("=" * 70)

    obs, asset_id, targets = load_real_data(n_assets=3, n_seqs=240, seq_len=96)

    results = []
    results.append(probe_one("A_baseline",       atme_prob=0.15, use_forecast_head=False, forecast_weight=0.0,
                             obs=obs, asset_id=asset_id, targets=targets))
    results.append(probe_one("B_atme_0.20",      atme_prob=0.20, use_forecast_head=False, forecast_weight=0.0,
                             obs=obs, asset_id=asset_id, targets=targets))
    results.append(probe_one("C_forecast_head",  atme_prob=0.15, use_forecast_head=True,  forecast_weight=0.5,
                             obs=obs, asset_id=asset_id, targets=targets))
    results.append(probe_one("D_B_plus_C",       atme_prob=0.20, use_forecast_head=True,  forecast_weight=0.5,
                             obs=obs, asset_id=asset_id, targets=targets))

    print("\n" + "=" * 70)
    print(f"{'Option':<22} {'init_loss':>10} {'final_loss':>11} {'drop%':>7} {'ret':>8} {'recon':>8} {'forecast':>10} {'train_IC':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<22} {r['init_total']:>10.3f} {r['final_total']:>11.3f} {r['drop_pct']:>7.1f} "
              f"{r['final_ret']:>8.3f} {r['final_recon']:>8.4f} {r['final_fc']:>10.4f} {r['train_ic']:>10.4f}")
    print("=" * 70)
    # Pick winner by train_IC (with tie-break on drop%)
    best = max(results, key=lambda r: (r["train_ic"], r["drop_pct"]))
    print(f"\n[WINNER by train_IC + drop%] {best['name']}")


if __name__ == "__main__":
    main()
