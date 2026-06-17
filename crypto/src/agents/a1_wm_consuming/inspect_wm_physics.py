"""
World Model Physics Inspection
===============================
Diagnose world model prediction quality across regimes and horizons.

Generates 6 diagnostic charts:
  1. Direction & IC overview (per horizon, IS vs OOS)
  2. Magnitude calibration (predicted vs actual scatter)
  3. Distribution comparison (predicted vs actual histograms)
  4. Regime analysis (IC/direction per regime, confusion matrix)
  5. Asset IC heatmap
  6. Scenario traces (bull/bear/chop with WM predictions)

Usage:
    python src/agents/a1_wm_consuming/inspect_wm_physics.py --ensemble
    python src/agents/a1_wm_consuming/inspect_wm_physics.py --world-model v1_0
    python src/agents/a1_wm_consuming/inspect_wm_physics.py --ensemble --max-windows 500
"""

import argparse
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DEVICE, NUM_ASSETS, REWARD_HORIZONS, ASSET_LIST, ASSET_TO_IDX,
    WARMUP_BARS, PURGE_GAP_BARS, VAL_FRACTION, BARS_PER_DAY,
)
from train_agent import (
    load_world_model, load_ensemble_model, load_data, PROJECT_ROOT,
)

HORIZONS = REWARD_HORIZONS  # [1, 4, 16, 64]
HORIZON_NAMES = {1: "h=1", 4: "h=4", 16: "h=16", 64: "h=64"}
REGIME_NAMES = {0: "Bear", 1: "Neutral", 2: "Bull"}
SEQ_LEN = WARMUP_BARS  # 96 bars per window
from datetime import date as _date
_DATE_SUBDIR = _date.today().isoformat()
PLOT_DIR = PROJECT_ROOT / "plots" / "agent" / "wm_physics" / _DATE_SUBDIR


def get_bucketer(model):
    """Get TwoHot bucketer from model or ensemble."""
    if hasattr(model, "models"):  # CrossModelEnsemble
        return model.models[0].bucketer
    return model.bucketer


# ==========================================================================
# Phase 1: Data Collection
# ==========================================================================

def collect_predictions(model, segments, mode, max_windows_per_asset=200, seed=42):
    """Run model forward pass on train/val data, collect predictions vs actuals.

    Args:
        model: World model or ensemble (frozen, eval)
        segments: Data segments from load_data()
        mode: "train" or "val"
        max_windows_per_asset: Cap on windows per asset (controls runtime)

    Returns dict with arrays of predictions, actuals, regime info.
    """
    bucketer = get_bucketer(model)
    rng = np.random.default_rng(seed)

    all_preds = {h: [] for h in HORIZONS}
    all_actuals = {h: [] for h in HORIZONS}
    all_regime_probs = []   # [N, 3] predicted regime probabilities
    all_regime_labels = []  # [N] actual regime labels (0/1/2)
    all_asset_ids = []      # [N] asset index
    total_windows = 0

    for seg in segments:
        asset_idx = seg["asset_idx"]
        features = seg["features"]
        n_bars = len(features)

        # Train/val split (matches environment.py)
        split_point = int(n_bars * (1.0 - VAL_FRACTION))
        if mode == "train":
            start, end = 0, split_point
        else:
            start = split_point + PURGE_GAP_BARS
            end = n_bars
            if start >= end:
                start = split_point

        feats = features[start:end]
        n_available = len(feats)
        if n_available < SEQ_LEN:
            continue

        # Generate non-overlapping window indices
        window_starts = list(range(0, n_available - SEQ_LEN + 1, SEQ_LEN))
        if len(window_starts) > max_windows_per_asset:
            window_starts = sorted(rng.choice(
                window_starts, size=max_windows_per_asset, replace=False
            ))

        # Batch windows for efficiency
        batch_size = 32
        for batch_start in range(0, len(window_starts), batch_size):
            batch_indices = window_starts[batch_start:batch_start + batch_size]
            obs_list = []
            for ws in batch_indices:
                obs_list.append(feats[ws:ws + SEQ_LEN])

            obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
            aid = torch.full((len(obs_list),), asset_idx, dtype=torch.long,
                             device=DEVICE)

            with torch.no_grad():
                outputs = model.forward_train(obs, aid)

            # Decode return predictions
            for h in HORIZONS:
                logits = outputs["return_logits"][h]  # [B, T, 255]
                preds_h = bucketer.decode(logits).cpu().numpy()  # [B, T]

                # Collect actuals
                tgt_key = f"target_return_{h}"
                for bi, ws in enumerate(batch_indices):
                    abs_start = start + ws
                    actual_h = seg[tgt_key][abs_start:abs_start + SEQ_LEN]
                    pred_h = preds_h[bi]
                    min_len = min(len(actual_h), len(pred_h))
                    all_preds[h].append(pred_h[:min_len])
                    all_actuals[h].append(actual_h[:min_len])

            # Regime predictions
            regime_logits = outputs.get("regime_logits")
            if regime_logits is not None:
                regime_probs = torch.softmax(regime_logits, dim=-1).cpu().numpy()
            else:
                regime_probs = np.full((len(obs_list), SEQ_LEN, 3), 1 / 3)

            for bi, ws in enumerate(batch_indices):
                abs_start = start + ws
                n_bars_window = min(SEQ_LEN, end - abs_start)
                all_regime_probs.append(regime_probs[bi, :n_bars_window])
                all_asset_ids.extend([asset_idx] * n_bars_window)

                if "regime_label" in seg:
                    rl = seg["regime_label"][abs_start:abs_start + n_bars_window]
                    all_regime_labels.append(rl)
                else:
                    all_regime_labels.append(np.ones(n_bars_window, dtype=np.int64))

            total_windows += len(batch_indices)

    # Concatenate
    result = {"n_windows": total_windows}
    for h in HORIZONS:
        result[f"preds_{h}"] = np.concatenate(all_preds[h])
        result[f"actuals_{h}"] = np.concatenate(all_actuals[h])
    result["regime_probs"] = np.concatenate(all_regime_probs, axis=0)
    result["regime_labels"] = np.concatenate(all_regime_labels)
    result["asset_ids"] = np.array(all_asset_ids)
    return result


# ==========================================================================
# Phase 2: Metrics
# ==========================================================================

def spearman_ic(pred, actual):
    """Rank correlation."""
    n = len(pred)
    if n < 30:
        return 0.0
    rp = np.argsort(np.argsort(pred)).astype(float)
    ra = np.argsort(np.argsort(actual)).astype(float)
    d = rp - ra
    return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def compute_metrics(data):
    """Compute comprehensive metrics from collected predictions."""
    metrics = {}

    for h in HORIZONS:
        p = data[f"preds_{h}"]
        a = data[f"actuals_{h}"]
        n = len(p)

        # Direction accuracy
        correct = ((p > 0) & (a > 0)) | ((p < 0) & (a < 0))
        dir_acc = float(correct.mean()) if n > 0 else 0.0

        # Spearman IC
        ic = spearman_ic(p, a)

        # Magnitude stats
        pred_std = float(np.std(p))
        actual_std = float(np.std(a))
        compression = pred_std / (actual_std + 1e-12)

        # Calibration: binned predicted vs actual
        if n > 100:
            n_bins = 20
            sorted_idx = np.argsort(p)
            bin_size = n // n_bins
            cal_pred, cal_actual = [], []
            for b in range(n_bins):
                idx = sorted_idx[b * bin_size:(b + 1) * bin_size]
                cal_pred.append(float(p[idx].mean()))
                cal_actual.append(float(a[idx].mean()))
            cal_slope = np.polyfit(cal_pred, cal_actual, 1)[0] if len(cal_pred) > 2 else 0.0
        else:
            cal_pred, cal_actual, cal_slope = [], [], 0.0

        # Conditional direction (top quartile predictions only)
        thresh = np.percentile(np.abs(p), 75) if n > 100 else 0
        strong_mask = np.abs(p) >= thresh
        strong_dir = float(correct[strong_mask].mean()) if strong_mask.sum() > 30 else 0.0

        metrics[h] = {
            "ic": ic, "dir_acc": dir_acc, "strong_dir_acc": strong_dir,
            "pred_std": pred_std, "actual_std": actual_std,
            "compression": compression, "cal_slope": cal_slope,
            "cal_pred": cal_pred, "cal_actual": cal_actual, "n": n,
        }

    # Per-regime metrics
    regime_labels = data["regime_labels"]
    for regime in [0, 1, 2]:
        mask = regime_labels == regime
        if mask.sum() < 100:
            metrics[f"regime_{regime}"] = {"n": int(mask.sum())}
            continue

        regime_m = {"n": int(mask.sum())}
        for h in HORIZONS:
            p = data[f"preds_{h}"][mask]
            a = data[f"actuals_{h}"][mask]
            regime_m[f"ic_{h}"] = spearman_ic(p, a)
            correct = ((p > 0) & (a > 0)) | ((p < 0) & (a < 0))
            regime_m[f"dir_{h}"] = float(correct.mean())
        metrics[f"regime_{regime}"] = regime_m

    # Per-asset IC
    asset_ids = data["asset_ids"]
    for a_idx in range(NUM_ASSETS):
        mask = asset_ids == a_idx
        if mask.sum() < 100:
            continue
        for h in HORIZONS:
            p = data[f"preds_{h}"][mask]
            act = data[f"actuals_{h}"][mask]
            metrics.setdefault("asset_ic", {})[f"{ASSET_LIST[a_idx]}_{h}"] = spearman_ic(p, act)

    # Regime head accuracy
    pred_regime = np.argmax(data["regime_probs"], axis=-1)
    actual_regime = data["regime_labels"]
    valid = actual_regime >= 0
    if valid.sum() > 100:
        metrics["regime_accuracy"] = float((pred_regime[valid] == actual_regime[valid]).mean())
        # Per-class accuracy
        for r in [0, 1, 2]:
            r_mask = actual_regime == r
            if r_mask.sum() > 10:
                metrics[f"regime_acc_{r}"] = float((pred_regime[r_mask] == r).mean())
    else:
        metrics["regime_accuracy"] = 0.0

    return metrics


# ==========================================================================
# Phase 3: Charts
# ==========================================================================

def plot_direction_ic(is_m, oos_m, save_path):
    """Chart 1: Direction accuracy + IC per horizon (IS vs OOS)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("World Model Signal Quality: Direction & IC", fontsize=14, fontweight="bold")

    # (0,0) Direction accuracy per horizon
    ax = axes[0, 0]
    x = np.arange(len(HORIZONS))
    w = 0.35
    is_dir = [is_m[h]["dir_acc"] for h in HORIZONS]
    oos_dir = [oos_m[h]["dir_acc"] for h in HORIZONS]
    ax.bar(x - w / 2, is_dir, w, label="IS", color="#4C72B0")
    ax.bar(x + w / 2, oos_dir, w, label="OOS", color="#DD8452")
    ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Random (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
    ax.set_ylabel("Direction Accuracy")
    ax.set_title("Direction Accuracy (all predictions)")
    ax.legend()
    ax.set_ylim(0.4, 0.6)

    # (0,1) IC per horizon
    ax = axes[0, 1]
    is_ic = [is_m[h]["ic"] for h in HORIZONS]
    oos_ic = [oos_m[h]["ic"] for h in HORIZONS]
    ax.bar(x - w / 2, is_ic, w, label="IS", color="#4C72B0")
    ax.bar(x + w / 2, oos_ic, w, label="OOS", color="#DD8452")
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
    ax.set_ylabel("Spearman IC")
    ax.set_title("Spearman IC (rank correlation)")
    ax.legend()

    # (1,0) Conditional direction accuracy (top 25% strongest predictions)
    ax = axes[1, 0]
    is_sdir = [is_m[h]["strong_dir_acc"] for h in HORIZONS]
    oos_sdir = [oos_m[h]["strong_dir_acc"] for h in HORIZONS]
    ax.bar(x - w / 2, is_sdir, w, label="IS (top 25%)", color="#4C72B0")
    ax.bar(x + w / 2, oos_sdir, w, label="OOS (top 25%)", color="#DD8452")
    ax.axhline(0.5, color="red", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
    ax.set_ylabel("Direction Accuracy")
    ax.set_title("Conditional Direction (strongest 25% predictions)")
    ax.legend()
    ax.set_ylim(0.4, 0.65)

    # (1,1) Magnitude compression ratio
    ax = axes[1, 1]
    is_comp = [is_m[h]["compression"] for h in HORIZONS]
    oos_comp = [oos_m[h]["compression"] for h in HORIZONS]
    ax.bar(x - w / 2, is_comp, w, label="IS", color="#4C72B0")
    ax.bar(x + w / 2, oos_comp, w, label="OOS", color="#DD8452")
    ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="Ideal (1:1)")
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
    ax.set_ylabel("Std(pred) / Std(actual)")
    ax.set_title("Magnitude Compression Ratio (1.0 = perfect)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_calibration(data, metrics, label, save_path):
    """Chart 2: Magnitude calibration scatter + binned calibration."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Magnitude Calibration: {label}", fontsize=14, fontweight="bold")

    for i, h in enumerate(HORIZONS):
        p = data[f"preds_{h}"]
        a = data[f"actuals_{h}"]
        m = metrics[h]

        # Top row: scatter (subsample for visibility)
        ax = axes[0, i]
        n_plot = min(5000, len(p))
        idx = np.random.choice(len(p), n_plot, replace=False)
        ax.scatter(p[idx], a[idx], alpha=0.05, s=1, c="#4C72B0")
        # Regression line
        if len(p) > 100:
            z = np.polyfit(p, a, 1)
            x_range = np.linspace(p.min(), p.max(), 100)
            ax.plot(x_range, z[0] * x_range + z[1], "r-", lw=2,
                    label=f"slope={z[0]:.1f}")
            ax.plot(x_range, x_range, "k--", alpha=0.3, label="ideal")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{HORIZON_NAMES[h]} | IC={m['ic']:+.4f}")
        ax.legend(fontsize=8)

        # Bottom row: binned calibration curve
        ax = axes[1, i]
        cp = m["cal_pred"]
        ca = m["cal_actual"]
        if cp:
            ax.plot(cp, ca, "o-", color="#4C72B0", markersize=4, label="Binned actual")
            min_v = min(min(cp), min(ca))
            max_v = max(max(cp), max(ca))
            ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.3, label="Perfect")
            ax.set_xlabel("Mean predicted (binned)")
            ax.set_ylabel("Mean actual (binned)")
            ax.set_title(f"Calibration | slope={m['cal_slope']:.1f}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_distributions(data, label, save_path):
    """Chart 3: Predicted vs actual return distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Distribution Comparison: {label}", fontsize=14, fontweight="bold")

    for i, h in enumerate(HORIZONS):
        ax = axes[i // 2, i % 2]
        p = data[f"preds_{h}"]
        a = data[f"actuals_{h}"]

        # Clip for visibility (99th percentile)
        clip_val = max(np.percentile(np.abs(a), 99), 1e-6)
        a_clip = np.clip(a, -clip_val, clip_val)

        # Actual distribution
        bins = np.linspace(-clip_val, clip_val, 80)
        ax.hist(a_clip, bins=bins, alpha=0.5, density=True, color="#DD8452",
                label=f"Actual (std={np.std(a):.6f})")

        # Prediction distribution (may need different range)
        p_clip = np.clip(p, -clip_val, clip_val)
        ax.hist(p_clip, bins=bins, alpha=0.5, density=True, color="#4C72B0",
                label=f"Predicted (std={np.std(p):.6f})")

        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.set_title(f"{HORIZON_NAMES[h]} | Compression = {np.std(p) / (np.std(a) + 1e-12):.4f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_regime_analysis(data, metrics, label, save_path):
    """Chart 4: IC and direction accuracy per regime + regime confusion."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Regime Analysis: {label}", fontsize=14, fontweight="bold")
    regime_colors = {0: "#C44E52", 1: "#8C8C8C", 2: "#55A868"}

    # (0,0) IC per regime per horizon
    ax = axes[0, 0]
    x = np.arange(len(HORIZONS))
    w = 0.25
    for ri, regime in enumerate([0, 1, 2]):
        rm = metrics.get(f"regime_{regime}", {})
        ics = [rm.get(f"ic_{h}", 0) for h in HORIZONS]
        ax.bar(x + ri * w - w, ics, w, label=REGIME_NAMES[regime],
               color=regime_colors[regime])
    ax.axhline(0, color="gray", ls="-", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
    ax.set_ylabel("Spearman IC")
    ax.set_title("IC by Market Regime")
    ax.legend()

    # (0,1) Direction accuracy per regime per horizon
    ax = axes[0, 1]
    for ri, regime in enumerate([0, 1, 2]):
        rm = metrics.get(f"regime_{regime}", {})
        dirs = [rm.get(f"dir_{h}", 0.5) for h in HORIZONS]
        ax.bar(x + ri * w - w, dirs, w, label=REGIME_NAMES[regime],
               color=regime_colors[regime])
    ax.axhline(0.5, color="red", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
    ax.set_ylabel("Direction Accuracy")
    ax.set_title("Direction by Market Regime")
    ax.legend()
    ax.set_ylim(0.35, 0.65)

    # (0,2) Regime distribution
    ax = axes[0, 2]
    regime_labels = data["regime_labels"]
    counts = [int((regime_labels == r).sum()) for r in [0, 1, 2]]
    total = sum(counts)
    bars = ax.bar([0, 1, 2], [c / total for c in counts],
                  color=[regime_colors[r] for r in [0, 1, 2]])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([REGIME_NAMES[r] for r in [0, 1, 2]])
    ax.set_ylabel("Fraction")
    ax.set_title(f"Regime Distribution (n={total:,})")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{c:,}", ha="center", fontsize=9)

    # (1,0) Regime confusion matrix
    ax = axes[1, 0]
    pred_regime = np.argmax(data["regime_probs"], axis=-1)
    confusion = np.zeros((3, 3), dtype=np.float64)
    for true_r in range(3):
        mask = regime_labels == true_r
        if mask.sum() > 0:
            for pred_r in range(3):
                confusion[true_r, pred_r] = (pred_regime[mask] == pred_r).mean()

    im = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    for (i, j), val in np.ndenumerate(confusion):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if val > 0.5 else "black", fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Bear", "Neutral", "Bull"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Bear", "Neutral", "Bull"])
    ax.set_xlabel("Predicted Regime")
    ax.set_ylabel("Actual Regime")
    ax.set_title(f"Regime Confusion (acc={metrics.get('regime_accuracy', 0):.1%})")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (1,1) Regime probability distributions
    ax = axes[1, 1]
    rp = data["regime_probs"]
    for ri, rname in enumerate(["Bear", "Neutral", "Bull"]):
        ax.hist(rp[:, ri], bins=50, alpha=0.5, density=True,
                color=regime_colors[ri], label=rname)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Regime Probability Distributions")
    ax.legend()

    # (1,2) Per-regime mean prediction vs actual for h=1
    ax = axes[1, 2]
    p1 = data["preds_1"]
    a1 = data["actuals_1"]
    for regime in [0, 1, 2]:
        mask = regime_labels == regime
        if mask.sum() > 100:
            mean_p = float(p1[mask].mean())
            mean_a = float(a1[mask].mean())
            std_a = float(a1[mask].std())
            ax.bar(regime * 3, mean_a * 1e4, color=regime_colors[regime],
                   alpha=0.6, label=f"Actual ({REGIME_NAMES[regime]})")
            ax.bar(regime * 3 + 1, mean_p * 1e4, color=regime_colors[regime],
                   alpha=1.0, edgecolor="black", linewidth=1.5)
            ax.errorbar(regime * 3, mean_a * 1e4, yerr=std_a * 1e4 / 5,
                        color="black", capsize=3)
    ax.set_xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(["Bear", "Neutral", "Bull"])
    ax.set_ylabel("Mean Return (x1e4)")
    ax.set_title("h=1: Mean Pred (solid edge) vs Actual per Regime")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_asset_heatmap(is_m, oos_m, save_path):
    """Chart 5: Asset x Horizon IC heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Asset x Horizon IC Heatmap", fontsize=14, fontweight="bold")

    for ax, (m, title) in zip(axes, [(is_m, "In-Sample"), (oos_m, "Out-of-Sample")]):
        asset_ic = m.get("asset_ic", {})
        matrix = np.zeros((NUM_ASSETS, len(HORIZONS)))
        for ai in range(NUM_ASSETS):
            for hi, h in enumerate(HORIZONS):
                key = f"{ASSET_LIST[ai]}_{h}"
                matrix[ai, hi] = asset_ic.get(key, 0.0)

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.10, vmax=0.15, aspect="auto")
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7, color="black")
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([HORIZON_NAMES[h] for h in HORIZONS])
        ax.set_yticks(range(NUM_ASSETS))
        ax.set_yticklabels([a.replace("USDT", "") for a in ASSET_LIST])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


def plot_scenario_traces(data, segments, mode, model, save_path):
    """Chart 6: Bull/Bear/Chop scenario traces with WM predictions."""
    bucketer = get_bucketer(model)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle("Scenario Traces: WM Predictions vs Reality",
                 fontsize=14, fontweight="bold")

    regime_targets = {0: "Bear", 1: "Neutral/Chop", 2: "Bull"}
    regime_colors = {0: "#C44E52", 1: "#8C8C8C", 2: "#55A868"}

    for row, target_regime in enumerate([2, 0, 1]):  # Bull, Bear, Chop
        # Find a segment with long consecutive run of this regime
        best_seg = None
        best_start = 0
        best_len = 0

        for seg in segments:
            if "regime_label" not in seg:
                continue
            rl = seg["regime_label"]
            n = len(rl)
            # Find longest consecutive run of target_regime
            run_start = 0
            run_len = 0
            for i in range(n):
                if rl[i] == target_regime:
                    run_len += 1
                    if run_len > best_len and run_len >= SEQ_LEN:
                        best_len = run_len
                        best_start = run_start
                        best_seg = seg
                else:
                    run_start = i + 1
                    run_len = 0

        if best_seg is None or best_len < SEQ_LEN:
            axes[row, 0].text(0.5, 0.5, f"No {regime_targets[target_regime]} segment found",
                              transform=axes[row, 0].transAxes, ha="center")
            axes[row, 1].text(0.5, 0.5, f"No {regime_targets[target_regime]} segment found",
                              transform=axes[row, 1].transAxes, ha="center")
            continue

        # Extract window
        ws = best_start
        window_feats = best_seg["features"][ws:ws + SEQ_LEN]
        actual_r1 = best_seg["target_return_1"][ws:ws + SEQ_LEN]
        asset_idx = best_seg["asset_idx"]
        asset_name = ASSET_LIST[asset_idx]

        # Get predictions
        obs = torch.from_numpy(window_feats[np.newaxis]).float().to(DEVICE)
        aid = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            outputs = model.forward_train(obs, aid)
        pred_r1 = bucketer.decode(outputs["return_logits"][1]).cpu().numpy()[0]

        min_len = min(len(actual_r1), len(pred_r1))
        actual_r1 = actual_r1[:min_len]
        pred_r1 = pred_r1[:min_len]
        t = np.arange(min_len)

        # Left: bar-by-bar returns
        ax = axes[row, 0]
        ax.bar(t, actual_r1 * 1e4, alpha=0.4, color=regime_colors[target_regime],
               label="Actual")
        ax.plot(t, pred_r1 * 1e4, color="#4C72B0", lw=1.5, alpha=0.8,
                label="Predicted")
        ax.set_ylabel("Return (x1e4)")
        ax.set_title(f"{regime_targets[target_regime]} ({asset_name}) - Bar Returns")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.5)

        # Right: cumulative returns
        ax = axes[row, 1]
        cum_actual = np.cumsum(actual_r1) * 1e4
        cum_pred = np.cumsum(pred_r1) * 1e4
        ax.plot(t, cum_actual, color=regime_colors[target_regime], lw=2,
                label="Actual cumulative")
        ax.plot(t, cum_pred, color="#4C72B0", lw=2, ls="--",
                label="Predicted cumulative")
        ax.fill_between(t, cum_actual, cum_pred, alpha=0.1, color="gray")
        ax.set_ylabel("Cumulative Return (x1e4)")
        ax.set_title(f"{regime_targets[target_regime]} - Cumulative (gap = tracking error)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("Bar")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {save_path.name}")


# ==========================================================================
# Phase 4: Console Report
# ==========================================================================

def print_report(is_m, oos_m, label):
    """Print formatted physics report."""
    print(f"\n  {'=' * 75}")
    print(f"  WORLD MODEL PHYSICS REPORT: {label}")
    print(f"  {'=' * 75}")

    print(f"\n  {'Horizon':<8s} {'IS Dir':>7s} {'OOS Dir':>8s} {'IS IC':>8s} "
          f"{'OOS IC':>8s} {'IS Compr':>9s} {'OOS Compr':>10s} {'Cal.Slope':>10s} {'Verdict':>8s}")
    print(f"  {'-' * 75}")

    for h in HORIZONS:
        im = is_m[h]
        om = oos_m[h]
        # Verdict: good physics if IC > 0 OOS and direction > 48% OOS
        if om["ic"] > 0.02 and om["dir_acc"] > 0.48:
            verdict = "GOOD"
        elif om["ic"] > 0 and om["dir_acc"] > 0.47:
            verdict = "WEAK"
        elif om["ic"] <= 0:
            verdict = "FAIL"
        else:
            verdict = "POOR"

        print(f"  {HORIZON_NAMES[h]:<8s} {im['dir_acc']:>6.1%} {om['dir_acc']:>7.1%} "
              f"{im['ic']:>+7.4f} {om['ic']:>+7.4f} "
              f"{im['compression']:>8.4f} {om['compression']:>9.4f} "
              f"{om['cal_slope']:>9.1f} [{verdict}]")

    # Regime head
    print(f"\n  Regime Head Performance:")
    print(f"    Overall accuracy: IS={is_m.get('regime_accuracy', 0):.1%}, "
          f"OOS={oos_m.get('regime_accuracy', 0):.1%}")
    for r in [0, 1, 2]:
        is_acc = is_m.get(f"regime_acc_{r}", 0)
        oos_acc = oos_m.get(f"regime_acc_{r}", 0)
        print(f"    {REGIME_NAMES[r]:<10s}: IS={is_acc:.1%}, OOS={oos_acc:.1%}")

    # Best/worst assets OOS
    print(f"\n  OOS Asset IC (h=1, sorted):")
    asset_ic = oos_m.get("asset_ic", {})
    h1_ics = [(a.replace("_1", ""), ic) for a, ic in asset_ic.items() if a.endswith("_1")]
    for name, ic in sorted(h1_ics, key=lambda x: -x[1]):
        bar = "+" * max(0, int(ic * 300))
        print(f"    {name:<12s} IC={ic:>+.4f} {bar}")

    # Magnitude compression summary
    print(f"\n  Magnitude Compression (std_pred / std_actual):")
    for h in HORIZONS:
        im = is_m[h]
        print(f"    {HORIZON_NAMES[h]}: pred_std={im['pred_std']:.6f}, "
              f"actual_std={im['actual_std']:.6f}, ratio={im['compression']:.4f}")

    # Physics verdict
    print(f"\n  {'=' * 75}")
    print(f"  PHYSICS VERDICT:")
    oos_h1_ic = oos_m[1]["ic"]
    oos_h1_dir = oos_m[1]["dir_acc"]
    compression = is_m[1]["compression"]

    verdicts = []
    if oos_h1_ic > 0.03:
        verdicts.append("[PASS] h=1 signal is genuine (OOS IC > 0.03)")
    else:
        verdicts.append(f"[WARN] h=1 OOS IC = {oos_h1_ic:.4f} (marginal)")

    if oos_m[64]["ic"] < 0:
        verdicts.append("[FAIL] h=64 IC reverses OOS (memorized temporal patterns)")
    else:
        verdicts.append("[PASS] h=64 IC positive OOS")

    if compression < 0.01:
        verdicts.append(f"[FAIL] Magnitude compression {compression:.4f} (predictions ~{1/compression:.0f}x too small)")
    elif compression < 0.1:
        verdicts.append(f"[WARN] Magnitude compression {compression:.4f} (predictions ~{1/compression:.0f}x too small)")
    else:
        verdicts.append(f"[PASS] Magnitude ratio {compression:.4f}")

    regime_acc = oos_m.get("regime_accuracy", 0)
    if regime_acc > 0.5:
        verdicts.append(f"[PASS] Regime head accuracy {regime_acc:.1%}")
    elif regime_acc > 0.35:
        verdicts.append(f"[WARN] Regime head accuracy {regime_acc:.1%} (near random)")
    else:
        verdicts.append(f"[FAIL] Regime head accuracy {regime_acc:.1%}")

    for v in verdicts:
        print(f"    {v}")

    print(f"  {'=' * 75}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="World Model Physics Inspection"
    )
    parser.add_argument("--world-model", type=str, default="v1_0")
    parser.add_argument("--features", type=int, choices=[13, 18, 19], default=13)
    parser.add_argument("--revin", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--ensemble-models", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=300,
                        help="Max windows per asset per mode (controls runtime)")
    args = parser.parse_args()

    model_label = "V1.E ensemble" if args.ensemble else args.world_model

    print("=" * 70)
    print("  WORLD MODEL PHYSICS INSPECTION")
    print("=" * 70)
    print(f"  Model: {model_label}")
    print(f"  Max windows/asset: {args.max_windows}")
    print(f"  Device: {DEVICE}")
    print()

    t0 = time.time()
    np.random.seed(args.seed)

    # Load model
    print("  Loading model...")
    if args.ensemble:
        ensemble_keys = None
        if args.ensemble_models:
            ensemble_keys = [k.strip() for k in args.ensemble_models.split(",")]
        model, revin, feature_list, load_data_fn = load_ensemble_model(ensemble_keys)
    else:
        model, revin, feature_list, load_data_fn = load_world_model(
            args.world_model, args.features, args.revin
        )

    segments = load_data(load_data_fn, feature_list)

    # Collect predictions
    print("\n  Collecting IS predictions...")
    is_data = collect_predictions(model, segments, "train",
                                  max_windows_per_asset=args.max_windows,
                                  seed=args.seed)
    print(f"  IS: {is_data['n_windows']} windows, {len(is_data['preds_1']):,} predictions")

    print("  Collecting OOS predictions...")
    oos_data = collect_predictions(model, segments, "val",
                                   max_windows_per_asset=args.max_windows,
                                   seed=args.seed + 1000)
    print(f"  OOS: {oos_data['n_windows']} windows, {len(oos_data['preds_1']):,} predictions")

    # Compute metrics
    print("\n  Computing metrics...")
    is_metrics = compute_metrics(is_data)
    oos_metrics = compute_metrics(oos_data)

    # Print report
    print_report(is_metrics, oos_metrics, model_label)

    # Generate charts
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "ensemble" if args.ensemble else args.world_model
    print(f"\n  Generating charts -> {PLOT_DIR}/")

    plot_direction_ic(
        is_metrics, oos_metrics,
        PLOT_DIR / f"1_direction_ic_{tag}.png"
    )
    plot_calibration(
        oos_data, oos_metrics, f"OOS ({model_label})",
        PLOT_DIR / f"2_calibration_oos_{tag}.png"
    )
    plot_distributions(
        oos_data, f"OOS ({model_label})",
        PLOT_DIR / f"3_distributions_oos_{tag}.png"
    )
    plot_regime_analysis(
        oos_data, oos_metrics, f"OOS ({model_label})",
        PLOT_DIR / f"4_regime_analysis_oos_{tag}.png"
    )
    plot_asset_heatmap(
        is_metrics, oos_metrics,
        PLOT_DIR / f"5_asset_ic_heatmap_{tag}.png"
    )
    plot_scenario_traces(
        oos_data, segments, "val", model,
        PLOT_DIR / f"6_scenario_traces_{tag}.png"
    )

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
