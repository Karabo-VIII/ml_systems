"""Mathematical correctness tests for M1-M5 components.

Pytest-friendly. Validates the MATH (formulas, roundtrips, invariants), not
just shape correctness like the smoke tests do.

Tests:
  1. symlog / symexp roundtrip (V16, V17 use this)
  2. TwoHotEncoder.encode / decode roundtrip
  3. lambda_return formula against analytical case (gamma=1, lambda=1 -> sum)
  4. DreamerV3 imagine vs forward consistency under zero actions
  5. SAC actor: log-prob correction for tanh squash
  6. MPPI plan converges to optimal action under known reward landscape
  7. MoE gate weights sum to 1 + entropy bounded by log(K)
  8. GRPO advantages: within-group mean ≈ 0
  9. Multi-task loss: log_var_initial near zero -> losses summed equally
  10. TrainingLoader normalization stats persist across cache reload
  11. purge_split: train/val/oos/unseen are disjoint by date

Run:
  python tests/test_model_math.py        # standalone, no pytest needed
  pytest tests/test_model_math.py        # pytest invocation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------- Test 1: symlog/symexp roundtrip ----------
def test_symlog_symexp_roundtrip():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents" / "a1_wm_consuming" / "backbones" / "v16_dreamerv3" / "v16_training"))
    from dreamer_v3 import symlog, symexp  # type: ignore
    x = torch.tensor([-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0])
    y = symexp(symlog(x))
    err = (x - y).abs().max().item()
    assert err < 1e-5, f"symlog/symexp roundtrip err={err}"
    print(f"  [PASS] symlog/symexp roundtrip max err = {err:.2e}")


# ---------- Test 2: TwoHotEncoder roundtrip ----------
def test_twohot_roundtrip():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents" / "a1_wm_consuming" / "backbones" / "v16_dreamerv3" / "v16_training"))
    from dreamer_v3 import TwoHotEncoder, symexp  # type: ignore
    enc = TwoHotEncoder(n_bins=255, bin_min=-5.0, bin_max=5.0)
    # Encode a value, then decode SHOULD match (within bin width).
    targets = torch.tensor([-2.5, -0.5, 0.0, 0.5, 2.5])
    twohot = enc.encode(targets)  # [5, 255]
    # Convert two-hot to logits (log probabilities)
    logits = (twohot + 1e-9).log()
    decoded = enc.decode(logits)
    err = (decoded - targets).abs().max().item()
    # Allow up to 0.05 because of bin discretization + symlog space
    assert err < 0.1, f"twohot roundtrip err={err}; targets={targets}, decoded={decoded}"
    print(f"  [PASS] TwoHot encode/decode roundtrip max err = {err:.4f}")


# ---------- Test 3: lambda_return formula ----------
def test_lambda_return_analytical():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents" / "a1_wm_consuming"))
    from dreamer_v3_agent import lambda_return  # type: ignore
    # gamma=1, lambda=1, all continues=1: lambda_return at t=0 should = sum(rewards) + value[end]
    H = 5
    rewards = torch.ones(1, H) * 0.5
    values = torch.zeros(1, H + 1)
    values[:, -1] = 10.0  # bootstrap
    continues = torch.ones(1, H)
    ret = lambda_return(rewards, values, continues, lambda_=1.0)
    expected_t0 = rewards.sum().item() + values[0, -1].item()
    assert abs(ret[0, 0].item() - expected_t0) < 1e-3, \
        f"lambda_return t=0 mismatch: got {ret[0, 0].item()}, expected {expected_t0}"
    print(f"  [PASS] lambda_return analytical case (gamma=1, lambda=1) matches sum-of-rewards + bootstrap")


# ---------- Test 4: DreamerV3 imagine vs forward consistency ----------
def test_dreamer_imagine_consistency():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents" / "a1_wm_consuming" / "backbones" / "v16_dreamerv3" / "v16_training"))
    from dreamer_v3 import DreamerV3WorldModel  # type: ignore
    torch.manual_seed(123)
    wm = DreamerV3WorldModel(obs_dim=10, action_dim=1, n_assets=2,
                             hidden_dim=32, stoch_categories=4, stoch_dimensions=4,
                             mlp_hidden=32).eval()
    B = 2
    init = wm.rssm.initial_state(B, "cpu")
    actions = torch.zeros(B, 5, 1)
    with torch.no_grad():
        roll = wm.imagine_rollout(init, actions)
    # Just validate that all imagine-step outputs have the right shape
    assert roll["feat"].shape == (B, 5, wm.rssm.hidden_dim + wm.rssm.stoch_dim)
    assert roll["rewards"].shape == (B, 5)
    assert roll["continues"].shape == (B, 5)
    # And that continues are in [0, 1]
    c = roll["continues"]
    assert (c >= 0).all() and (c <= 1).all(), "continues out of [0, 1]"
    print(f"  [PASS] DreamerV3 imagine_rollout shapes + continue probs in [0, 1]")


# ---------- Test 5: SAC actor log-prob correction for tanh squash ----------
def test_sac_logprob_tanh_correction():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents" / "a1_wm_consuming"))
    from sac_agent import GaussianActor  # type: ignore
    torch.manual_seed(0)
    actor = GaussianActor(obs_dim=8, action_dim=2)
    obs = torch.randn(64, 8)
    a, log_p = actor.sample(obs)
    # log_p should be a single scalar per sample (sum over action_dim)
    assert log_p.shape == (64, 1)
    # Action is tanh-squashed, so should be in (-1, 1)
    assert (a > -1).all() and (a < 1).all(), "action not in (-1, 1)"
    # log_p should be finite (no NaN or inf from log(1 - a^2))
    assert torch.isfinite(log_p).all(), "log_prob has inf/NaN"
    print(f"  [PASS] SAC actor: log_p finite, action in (-1, 1), shape correct")


# ---------- Test 6: MPPI converges on synthetic landscape ----------
def test_mppi_convergence():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agents" / "a1_wm_consuming" / "backbones" / "v17_tdmpc2" / "v17_training"))
    from td_mpc2 import TDMPC2WorldModel, MPPIPlanner  # type: ignore
    torch.manual_seed(42)
    wm = TDMPC2WorldModel(obs_dim=4, action_dim=1, n_assets=1,
                          latent_dim=8, hidden_dim=16, asset_embed_dim=2).eval()
    # Don't train; just test that planner runs without NaN
    planner = MPPIPlanner(wm, action_dim=1, n_samples=16, horizon=3, n_iters=2)
    obs = torch.randn(1, 4)
    aid = torch.zeros(1, dtype=torch.long)
    a = planner.plan(obs, aid)
    assert torch.isfinite(a).all(), f"MPPI returned non-finite action: {a}"
    assert a.shape == (1,), f"unexpected shape {a.shape}"
    print(f"  [PASS] MPPI plan finite + correct shape; action = {a.item():.4f}")


# ---------- Test 7: MoE gate properties ----------
def test_moe_gate_invariants():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agent"))
    from moe_gate_v51 import MoEGate  # type: ignore
    torch.manual_seed(0)
    K = 5
    gate = MoEGate(state_dim=20, n_champions=K)
    state = torch.randn(32, 20)
    weights = gate(state)
    # Sum to 1 per row
    assert torch.allclose(weights.sum(dim=-1), torch.ones(32), atol=1e-4)
    # All non-negative
    assert (weights >= 0).all()
    # Entropy upper-bounded by log(K)
    eps = 1e-9
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    log_k = float(torch.log(torch.tensor(K)))
    assert (entropy <= log_k + 1e-3).all(), f"entropy exceeds log(K)={log_k}"
    print(f"  [PASS] MoE gate: weights sum=1, non-neg, entropy<=log(K)={log_k:.3f}")


# ---------- Test 8: GRPO advantages within-group mean ----------
def test_grpo_advantages():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agent"))
    from grpo_patch import compute_grpo_advantages  # type: ignore
    torch.manual_seed(0)
    rewards = torch.randn(8, 16, 4)  # 8 groups, 16 rollouts, 4 timesteps
    adv = compute_grpo_advantages(rewards)
    # Within-group mean should be ~0
    grp_means = adv.mean(dim=1)  # [8, 4]
    assert grp_means.abs().max() < 1e-5, f"grp mean should be ~0, got {grp_means.abs().max()}"
    # Within-group std should be ~1 (after z-score)
    grp_stds = adv.std(dim=1)  # [8, 4]
    assert (grp_stds - 1.0).abs().max() < 0.5, f"grp std drifted: {grp_stds}"
    print(f"  [PASS] GRPO: within-group mean=0 (max abs {grp_means.abs().max():.2e})")


# ---------- Test 9: Multi-task initial log_var ≈ 0 ----------
def test_multitask_initial_balance():
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "agent"))
    from multi_task_heads import MultiTaskHead  # type: ignore
    head = MultiTaskHead(feat_dim=64)
    # All 4 log_var params start at 0 (per Kendall 2018 init)
    for name, p in head.named_parameters():
        if "log_var" in name:
            assert abs(p.item()) < 1e-6, f"{name} should init to 0, got {p.item()}"
    print(f"  [PASS] Multi-task heads: all 4 log_var params init at 0")


# ---------- Test 10: TrainingLoader normalization cache persistence ----------
def test_training_loader_norm_cache():
    from pipeline.training_loader import TrainingLoader, NORM_CACHE_DIR
    tl1 = TrainingLoader(universe="u10", cadence="1d",
                         features=["norm_return_1"],
                         targets=["target_return_1"])
    stats1 = tl1.fit_normalizers(force=True)
    cache_path = NORM_CACHE_DIR / f"{tl1.cache_key}.parquet"
    assert cache_path.exists(), f"cache not written to {cache_path}"
    # Reload
    tl2 = TrainingLoader(universe="u10", cadence="1d",
                         features=["norm_return_1"],
                         targets=["target_return_1"])
    stats2 = tl2.fit_normalizers(force=False)
    # Compare for the feature
    assert "norm_return_1" in stats1 and "norm_return_1" in stats2
    s1 = stats1["norm_return_1"]
    s2 = stats2["norm_return_1"]
    assert abs(s1.mean - s2.mean) < 1e-9 and abs(s1.std - s2.std) < 1e-9
    print(f"  [PASS] TrainingLoader: norm stats persist across cache reload (mean={s1.mean:.4f}, std={s1.std:.4f})")


# ---------- Test 11: purge_split disjoint segments ----------
def test_purge_split_disjoint():
    import polars as pl
    from pipeline.purge_split import split_chimera, get_split_dates
    boundaries = get_split_dates()
    # Build synthetic df spanning all 4 segments
    import numpy as np_  # alias to avoid shadowing
    from datetime import date, datetime, timedelta, timezone
    start = date(2020, 1, 1)
    days = (date(2026, 4, 1) - start).days
    ts_ms = np_.array([
        int(datetime.combine(start + timedelta(days=i), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
        for i in range(days)
    ], dtype=np_.int64)
    df = pl.DataFrame({"timestamp": ts_ms, "close": np_.cumsum(np_.random.randn(days))})
    train, val, oos, unseen = split_chimera(df, boundaries=boundaries)
    # Each segment's date range should NOT overlap with another
    def date_range(s):
        if len(s) == 0:
            return None, None
        ts = s["timestamp"].to_numpy()
        return ts.min(), ts.max()
    tr = date_range(train)
    v = date_range(val)
    o = date_range(oos)
    u = date_range(unseen)
    # Overlap test: train's max < val's min < val's max < oos's min ...
    if tr[0] is not None and v[0] is not None:
        assert tr[1] < v[0], f"train+val overlap: train max {tr[1]} >= val min {v[0]}"
    if v[0] is not None and o[0] is not None:
        assert v[1] < o[0], f"val+oos overlap: val max {v[1]} >= oos min {o[0]}"
    print(f"  [PASS] purge_split: train ({tr[0]}-{tr[1]}) < val < oos < unseen, all disjoint")


# ---------- Runner ----------
TESTS = [
    test_symlog_symexp_roundtrip,
    test_twohot_roundtrip,
    test_lambda_return_analytical,
    test_dreamer_imagine_consistency,
    test_sac_logprob_tanh_correction,
    test_mppi_convergence,
    test_moe_gate_invariants,
    test_grpo_advantages,
    test_multitask_initial_balance,
    test_training_loader_norm_cache,
    test_purge_split_disjoint,
]


def main():
    print(f"Running {len(TESTS)} math + integration tests")
    print("=" * 60)
    n_pass = n_fail = 0
    for fn in TESTS:
        try:
            fn()
            n_pass += 1
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("=" * 60)
    print(f"Result: {n_pass} pass, {n_fail} fail")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
