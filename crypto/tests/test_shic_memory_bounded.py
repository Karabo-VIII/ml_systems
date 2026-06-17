"""Regression guard: compute_shuffled_ic must stay MEMORY-BOUNDED.

Provenance (2026-06-11): a V1.1 training run OOM'd ~1.5h in at compute_shuffled_ic
because it ran the model on the FULL multi-million-bar asset per fold (a fancy-index
copy feats[fold_indices] ~ len*41*4 bytes + a full-length forward pass + np.corrcoef's
float64 doubling), exhausting host RAM. Fixed by capping bars/asset
(AntifragileConfig.shuffled_ic_max_bars = 90000) + a memory-light float32 Pearson.

This test FAILS if that cap is removed or regressed -- i.e. if compute_shuffled_ic
ever forwards more than the cap's worth of bars in a single call again.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from anti_fragile import AntifragileConfig, ShuffledICTracker  # noqa: E402


def test_shic_cap_enforced():
    cfg = AntifragileConfig()
    cap = getattr(cfg, "shuffled_ic_max_bars", 0)
    assert cap and cap > 0, "shuffled_ic_max_bars must be set (>0) -- the OOM cap is the fix"

    class _Dummy:
        def eval(self):
            return self

    # A 300k-bar asset. UNCAPPED, the per-fold fancy-index copy + forward pass over
    # ~1/3 of this (and on the real chimera, millions of bars) is what OOM'd.
    segs = [
        {
            "features": np.random.randn(300_000, 41).astype(np.float32),
            "target_return_1": np.random.randn(300_000).astype(np.float32),
            "asset_idx": 0,
        }
    ]

    seen_max = {"n": 0}

    def _mock_predict(model, feats, asset_idx, horizon):
        seen_max["n"] = max(seen_max["n"], len(feats))
        # THE GUARD: the forward pass must NEVER see more than the cap (per fold <= cap/folds).
        assert len(feats) <= cap, "CAP VIOLATED: forward saw %d bars > cap %d" % (len(feats), cap)
        return (feats[:, 0] * 0.05 + np.random.randn(len(feats)) * 0.5).astype(np.float32)

    try:
        tracker = ShuffledICTracker(cfg)
    except TypeError:
        tracker = ShuffledICTracker()

    ic = tracker.compute_shuffled_ic(_Dummy(), segs, _mock_predict, horizon=1, seed=42)

    assert np.isfinite(ic), "ShIC must be finite"
    assert seen_max["n"] <= cap, "forward saw %d bars > cap %d (memory unbounded)" % (seen_max["n"], cap)
    print("PASS: cap=%d enforced (max forward %d bars), ShIC=%.4f, no OOM" % (cap, seen_max["n"], ic))


if __name__ == "__main__":
    test_shic_cap_enforced()
    print("test_shic_memory_bounded: ALL PASS")
