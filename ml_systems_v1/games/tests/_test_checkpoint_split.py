"""RWYB probe for the 2026-06-08 checkpoint-bloat fix: the optimizer + replay buffer were
moved OUT of every net_iterN.pt into ONE rolling train_state.pt sidecar.

Verifies:
  1. net_iterN.pt is SMALL (no 'buffer'/'optimizer' keys) and train_state.pt holds them.
  2. A new-format save round-trips: weights + optimizer-state + buffer + champion all restore.
  3. BACKWARD-COMPAT: an old-format net_iterN.pt (inline buffer+optimizer, no sidecar) still loads.
  4. STALE-SIDECAR GUARD: a train_state.pt OLDER than the net falls back to the inline buffer.

Run from repo root:  python -m az._test_checkpoint_split
No GPU needed (CPU; tiny net). No emoji (Windows cp1252).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from az.train_robust import (
    RobustConfig, save_checkpoint, load_checkpoint, find_latest_checkpoint,
    Champion, AlphaZeroNet, _atomic_torch_save, _capture_rng,
)
from az.selfplay import Sample


def _mk_net():
    return AlphaZeroNet(channels=8, n_blocks=1)


def _populate_opt_state(net, opt):
    """Give Adam non-empty state without a real forward (shape-agnostic): fake grads + a step."""
    for p in net.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    opt.zero_grad()


def _mk_buffer(n, n_planes=19, n_policy=256):
    return [Sample(planes=np.random.randn(n_planes, 8, 8).astype(np.float32),
                   pi=np.random.randn(n_policy).astype(np.float32),
                   player=bool(i % 2), z=float((i % 3) - 1)) for i in range(n)]


def _sz(p):
    return os.path.getsize(p) if os.path.exists(p) else -1


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = RobustConfig(channels=8, n_blocks=1, curriculum=False)

    # ---------- (1)+(2) new-format save round-trip ----------
    with tempfile.TemporaryDirectory() as d:
        net = _mk_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        _populate_opt_state(net, opt)
        buf = _mk_buffer(200)
        champ = Champion(iter=5, winrate_vs_random=1.0, winrate_vs_classical=0.0,
                         loss=1.5, score_vs_classical=0.25,
                         state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        net_path = save_checkpoint(d, 5, net, opt, cfg, buf, champion=champ)

        ts_path = os.path.join(d, "train_state.pt")
        assert os.path.exists(net_path), "net_iter5.pt missing"
        assert os.path.exists(ts_path), "train_state.pt missing"
        # net file must NOT carry the big keys; sidecar must
        net_payload = torch.load(net_path, map_location="cpu", weights_only=False)
        assert "buffer" not in net_payload, "BLOAT: buffer still inline in net_iterN.pt"
        assert "optimizer" not in net_payload, "BLOAT: optimizer still inline in net_iterN.pt"
        ts_payload = torch.load(ts_path, map_location="cpu", weights_only=False)
        assert "buffer" in ts_payload and "optimizer" in ts_payload, "sidecar missing big keys"
        assert _sz(net_path) < _sz(ts_path), f"net {_sz(net_path)} !< sidecar {_sz(ts_path)} (buffer not isolated)"
        print(f"[1] net_iter5.pt={_sz(net_path)//1024}KB  train_state.pt={_sz(ts_path)//1024}KB  "
              f"(buffer+opt isolated to sidecar) OK")

        # round-trip into fresh net/opt
        net2 = _mk_net()
        opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
        latest = find_latest_checkpoint(d)
        it2, buf2, champ2 = load_checkpoint(latest, net2, opt2, "cpu", cfg)
        assert it2 == 5, f"iter {it2} != 5"
        assert len(buf2) == len(buf), f"buffer len {len(buf2)} != {len(buf)}"
        assert champ2 is not None and champ2.iter == 5 and abs(champ2.score_vs_classical - 0.25) < 1e-9, "champion not restored"
        # weights restored bit-for-bit
        for (k, a), (_, b) in zip(net.state_dict().items(), net2.state_dict().items()):
            assert torch.allclose(a.cpu(), b.cpu()), f"weight {k} mismatch after resume"
        assert len(opt2.state_dict()["state"]) > 0, "optimizer state not restored"
        # sample content survived
        assert np.allclose(buf2[0].planes, buf[0].planes) and abs(buf2[7].z - buf[7].z) < 1e-9, "buffer content corrupted"
        print(f"[2] round-trip OK: iter={it2} buffer={len(buf2)} champion=iter{champ2.iter} "
              f"weights+optimizer+samples restored")

        # ---------- (4) STALE-SIDECAR GUARD (sidecar iter < net iter -> inline fallback) ----------
        # train_state.pt is at iter 5; write a NEWER old-format net_iter9.pt with inline buffer.
        old9 = {"iter": 9, "channels": 8, "n_blocks": 1,
                "state_dict": net.state_dict(), "optimizer": opt.state_dict(),
                "rng": _capture_rng(), "config": {},
                "buffer": [(s.planes, s.pi, s.player, s.z) for s in _mk_buffer(37)]}
        _atomic_torch_save(old9, os.path.join(d, "net_iter9.pt"))
        _atomic_torch_save({"iter": 9, "path": "net_iter9.pt"}, os.path.join(d, "latest.pt"))
        net4 = _mk_net(); opt4 = torch.optim.Adam(net4.parameters(), lr=1e-3)
        it4, buf4, _ = load_checkpoint(find_latest_checkpoint(d), net4, opt4, "cpu", cfg)
        assert it4 == 9 and len(buf4) == 37, f"stale-guard FAIL: iter={it4} buf={len(buf4)} (want 9/37 from inline)"
        print(f"[4] stale-sidecar guard OK: net iter9 > sidecar iter5 -> used inline buffer ({len(buf4)})")

    # ---------- (3) BACKWARD-COMPAT: old-format checkpoint, NO sidecar at all ----------
    with tempfile.TemporaryDirectory() as d2:
        net = _mk_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        _populate_opt_state(net, opt)
        old = {"iter": 0, "channels": 8, "n_blocks": 1,
               "state_dict": net.state_dict(), "optimizer": opt.state_dict(),
               "rng": _capture_rng(), "config": {},
               "buffer": [(s.planes, s.pi, s.player, s.z) for s in _mk_buffer(64)]}
        _atomic_torch_save(old, os.path.join(d2, "net_iter0.pt"))
        _atomic_torch_save({"iter": 0, "path": "net_iter0.pt"}, os.path.join(d2, "latest.pt"))
        assert not os.path.exists(os.path.join(d2, "train_state.pt")), "test setup: sidecar should be absent"
        net3 = _mk_net(); opt3 = torch.optim.Adam(net3.parameters(), lr=1e-3)
        it3, buf3, _ = load_checkpoint(find_latest_checkpoint(d2), net3, opt3, "cpu", cfg)
        assert it3 == 0 and len(buf3) == 64, f"backcompat FAIL: iter={it3} buf={len(buf3)} (want 0/64 inline)"
        assert len(opt3.state_dict()["state"]) > 0, "backcompat: optimizer not restored from inline"
        print(f"[3] backward-compat OK: old-format net_iter0 (no sidecar) -> inline buffer ({len(buf3)}) + optimizer restored")

    print("\nALL CHECKPOINT-SPLIT CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
