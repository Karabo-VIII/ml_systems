"""walk_forward -- N-seed audit + bootstrap CI summarizer.

Wraps signal_picker + upgrades into a single multi-seed validation loop.
This is the discipline that yesterday's LSTM/DQN attempts skipped (single-seed claims
of +44% / +40% were debunked to median -7% / -34% at 10-seed audit).

__contract__:
  inputs: BotConfig, prepared data, list of seeds
  outputs: per-seed results table + summary (mean/median/std/p05/p95)
  invariants:
    - every seed runs independently (no shared state)
    - seed is EXPLICIT in every randomness consumer (LGBM bagging+feature_fraction)
    - bootstrap CIs use real trade returns (no synthetic)
"""
from __future__ import annotations

__contract__ = {
    "kind": "walk_forward_validator",
    "owner": "wealth_bot/framework/walk_forward",
    "purpose": "Multi-seed N=10 audit + bootstrap CIs",
    "invariants": [
        "every seed independent",
        "no synthetic data in validation",
        "trade-level returns retained for bootstrap",
    ],
}

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BotConfig
from .signal_picker import PickerOutput, train_picker, evaluate_actions
from .upgrades import train_ensemble, calibrate_threshold, apply_threshold


# ML-path n_eff entry gate (BINDING 2026-05-25, trainer+oracle+architect 3-way consensus):
# below this many in-segment signal fires across TRAIN+VAL+OOS+UNSEEN, LGBM has too few
# fit/eval samples per refit to extract genuine signal beyond initialization noise.
# At n_eff_total < 30, the static-rule path beats LGBM by a wide margin (10-seed audit).
# The framework should hard-block ML promotion below this floor and force a static-only
# bifurcation downstream.
ML_PATH_N_EFF_MIN = 30


def _count_signal_fires(signals: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, int]:
    """Count rows where ANY strategy fires, restricted per segment."""
    any_fire = (signals.sum(axis=1) > 0)
    return {seg: int((any_fire & m).sum()) for seg, m in masks.items()}


def n_seed_audit(
    cfg: BotConfig,
    df_lag: pd.DataFrame,
    signals: np.ndarray,
    fwd_ret: np.ndarray,
    masks: dict[str, np.ndarray],
    seeds: list[int] | None = None,
    use_ensemble: bool = False,
    use_threshold_calibration: bool = False,
    verbose: bool = True,
    save_preds: bool = False,
    enforce_ml_n_eff_gate: bool = True,
) -> dict:
    """Run N-seed audit; return per-seed + summary.

    If use_ensemble=True, also computes ensemble-averaged result as a single output.
    If use_threshold_calibration=True, calibrates threshold on VAL per seed (or once on ensemble preds).
    If save_preds=True, raw preds arrays are retained under the '_preds' key
    of the return dict for downstream .npz serialization by save_audit_json.
    The '_preds' key is stripped before JSON serialization (handled in save_audit_json).

    BINDING v1.2 (2026-05-25): ML-path n_eff gate. If the total in-segment signal
    fires across TRAIN+VAL+OOS+UNSEEN is below ML_PATH_N_EFF_MIN (=30), the audit
    returns early with an `_ml_n_eff_gate_failed` flag. Caller MUST bifurcate to
    static-rule path; LGBM has no statistical power here. Set enforce_ml_n_eff_gate=False
    only when running synthetic/diagnostic seeds.
    """
    if seeds is None:
        seeds = list(range(cfg.n_seeds))

    # ── ML-path n_eff entry gate ──
    fires_per_seg = _count_signal_fires(signals, masks)
    total_fires = sum(fires_per_seg.values())
    ml_gate = {
        "fires_per_segment": fires_per_seg,
        "total_signal_fires": total_fires,
        "ml_path_min_required": ML_PATH_N_EFF_MIN,
        "passes": total_fires >= ML_PATH_N_EFF_MIN,
        "rationale": (
            f"ML path requires >= {ML_PATH_N_EFF_MIN} total signal fires across "
            "TRAIN+VAL+OOS+UNSEEN for LGBM refits to have statistical power. "
            "Below this, static rules beat LGBM by audited margin."
        ),
    }
    if enforce_ml_n_eff_gate and not ml_gate["passes"]:
        if verbose:
            print(f"[n_seed_audit] ML_PATH_N_EFF_GATE FAILED: "
                  f"total_signal_fires={total_fires} < {ML_PATH_N_EFF_MIN}. "
                  f"Per-seg fires: {fires_per_seg}. "
                  "Returning early — caller must bifurcate to static-rule path.",
                  flush=True)
        return {
            "n_seeds": 0,
            "per_seed": [],
            "summary": {},
            "ensemble": None,
            "_ml_n_eff_gate": ml_gate,
            "_ml_n_eff_gate_failed": True,
        }

    per_seed_results = []
    per_seed_outputs: list[PickerOutput] = []
    for s in seeds:
        if verbose:
            print(f"[n_seed_audit] seed {s+1}/{len(seeds)}...", flush=True)
        out = train_picker(df_lag, signals, fwd_ret, cfg.chimera_features,
                            cfg.fwd_bars, cfg.model, seed=s, threshold=0.0)
        per_seed_outputs.append(out)

        # If threshold calibration enabled, calibrate per seed on VAL
        if use_threshold_calibration:
            best_thr, thr_scores = calibrate_threshold(
                out.preds, signals, fwd_ret, masks, cfg.fwd_bars,
                cfg.upgrades.u2_threshold_grid, metric="compound_pct", seg="VAL",
            )
            actions, chosen = apply_threshold(out.preds, signals, cfg.fwd_bars, best_thr)
        else:
            best_thr = 0.0
            thr_scores = {}
            actions, chosen = out.actions, out.chosen

        results = evaluate_actions(actions, fwd_ret, masks, cfg.fwd_bars)
        per_seed_results.append({
            "seed": s,
            "best_threshold": best_thr,
            "threshold_scores": thr_scores,
            **{seg: results[seg] for seg in masks},
        })

    summary = _summarize(per_seed_results, segments=list(masks.keys()))

    # ── BINDING 2026-05-25 (trust-stack item #1): per_seed_oos_gate ──
    # The ensemble result masks per-seed OOS instability. A live deploy uses
    # ONE seed, not the ensemble. If only 30% of seeds are OOS-positive,
    # the live deployment has a 70% chance of being a loser. Gate the
    # ensemble report behind per-seed OOS robustness.
    oos_positive_seeds_pct = float(summary.get("OOS", {}).get("positive_seeds_pct", 0.0))
    PER_SEED_OOS_GATE_PCT = 70.0  # >=70% of seeds must be OOS-positive
    seed_gate_passed = oos_positive_seeds_pct >= PER_SEED_OOS_GATE_PCT
    if not seed_gate_passed:
        # Don't raise — log + add to summary so caller can react. This is the
        # claim-contract enforcement point: if seed_gate_passed=False, the
        # candidate is NOT ship-tier regardless of ensemble headline.
        print(f"[walk_forward] PER_SEED_OOS_GATE FAILED: "
              f"{oos_positive_seeds_pct:.1f}% < {PER_SEED_OOS_GATE_PCT:.1f}% "
              f"(positive_seeds={summary['OOS']['positive_seeds']}/{cfg.n_seeds})",
              flush=True)
    summary["_per_seed_oos_gate"] = {
        "gate_threshold_pct": PER_SEED_OOS_GATE_PCT,
        "observed_pct": oos_positive_seeds_pct,
        "passed": seed_gate_passed,
        "rationale": ("70% seeds must be OOS-positive because LIVE deploys one seed; "
                      "ensemble headline masks per-seed instability"),
    }

    # Ensemble pathway: average preds across seeds, single output
    ensemble_result = None
    if use_ensemble and len(per_seed_outputs) >= 2:
        stacked = np.stack([o.preds for o in per_seed_outputs], axis=0)
        ens_preds = np.nanmean(stacked, axis=0)
        if use_threshold_calibration:
            best_thr_ens, ens_thr_scores = calibrate_threshold(
                ens_preds, signals, fwd_ret, masks, cfg.fwd_bars,
                cfg.upgrades.u2_threshold_grid, metric="compound_pct", seg="VAL",
            )
        else:
            best_thr_ens = 0.0
            ens_thr_scores = {}
        ens_actions, ens_chosen = apply_threshold(ens_preds, signals, cfg.fwd_bars, best_thr_ens)
        ens_results = evaluate_actions(ens_actions, fwd_ret, masks, cfg.fwd_bars)
        ensemble_result = {
            "best_threshold": best_thr_ens,
            "threshold_scores": ens_thr_scores,
            **{seg: ens_results[seg] for seg in masks},
        }

    result = {
        "n_seeds": len(seeds),
        "per_seed": per_seed_results,
        "summary": summary,
        "ensemble": ensemble_result,
        "_ml_n_eff_gate": ml_gate,
    }

    if save_preds:
        # Stash raw preds arrays (per-seed + ensemble if computed) for
        # downstream .npz serialization by save_audit_json. Stripped before JSON.
        preds_payload = {
            "per_seed_preds": np.stack([o.preds for o in per_seed_outputs], axis=0),
            "per_seed_actions": np.stack([o.actions for o in per_seed_outputs], axis=0),
            "per_seed_chosen": np.stack([o.chosen for o in per_seed_outputs], axis=0),
        }
        if use_ensemble and len(per_seed_outputs) >= 2:
            stacked = np.stack([o.preds for o in per_seed_outputs], axis=0)
            preds_payload["ensemble_preds"] = np.nanmean(stacked, axis=0)
            preds_payload["ensemble_best_threshold"] = np.array(
                ensemble_result["best_threshold"] if ensemble_result else 0.0
            )
        result["_preds"] = preds_payload

    return result


def _summarize(per_seed_results: list[dict], segments: list[str]) -> dict:
    """Compute per-segment mean/median/std/p05/p95 across seeds."""
    out = {}
    for seg in segments:
        comps = np.array([r[seg]["compound_pct"] for r in per_seed_results])
        n_trades = np.array([r[seg]["n_trades"] for r in per_seed_results])
        wrs = np.array([r[seg]["win_rate"] for r in per_seed_results])
        dds = np.array([r[seg]["max_dd_pct"] for r in per_seed_results])
        positive_seeds = int((comps > 0).sum())
        out[seg] = {
            "compound_mean": float(comps.mean()),
            "compound_median": float(np.median(comps)),
            "compound_std": float(comps.std()),
            "compound_p05": float(np.percentile(comps, 5)),
            "compound_p95": float(np.percentile(comps, 95)),
            "compound_min": float(comps.min()),
            "compound_max": float(comps.max()),
            "positive_seeds_pct": positive_seeds * 100.0 / len(comps),
            "positive_seeds": positive_seeds,
            "mean_trades": float(n_trades.mean()),
            "mean_wr": float(wrs.mean()),
            "median_max_dd": float(np.median(dds)),
            "worst_max_dd": float(dds.min()),
        }
    return out


def bootstrap_trade_returns(
    trade_returns: np.ndarray,
    n_boot: int = 1000,
    confidence: list[float] = [5, 50, 95],
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap a list of per-trade returns to get compound percentiles."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(trade_returns)
    if n == 0:
        return {f"p{c}": 0.0 for c in confidence}
    compounds = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(trade_returns, size=n, replace=True)
        compounds[b] = (np.prod(1 + sample) - 1) * 100
    return {f"p{c}": float(np.percentile(compounds, c)) for c in confidence}


def save_audit_json(audit_result: dict, out_path: str | Path) -> None:
    """Save audit results to JSON (atomic via tmp+rename).

    If audit_result contains a '_preds' key (populated when n_seed_audit was
    called with save_preds=True), the raw arrays are written to a sibling
    .npz file (<out_path stem>_preds.npz) BEFORE the JSON is finalized, and
    the '_preds' key is stripped from the JSON payload so the file remains
    text-only.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pop preds payload (if any) for sidecar .npz; never written into JSON.
    preds_payload = audit_result.pop("_preds", None)
    if preds_payload is not None:
        npz_path = out_path.with_name(out_path.stem + "_preds.npz")
        np.savez_compressed(npz_path, **preds_payload)
        audit_result["_preds_npz"] = str(npz_path.name)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as fp:
        json.dump(audit_result, fp, indent=2, default=str)
    tmp.replace(out_path)
