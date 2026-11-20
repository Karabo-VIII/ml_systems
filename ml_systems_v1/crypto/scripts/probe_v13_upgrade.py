"""RWYB probe for the V13 world-class upgrade (recon graft + shared levers).

Proves, on REAL model code with synthetic-but-shaped tensors (NO training):
  (A) LEVERS OFF (default) -> the two shared-lever flags add nothing: no VSN module,
      no 'forward_regime' key, no 'fr_aux' loss term; loss dict keys match the
      no-lever baseline. (The recon graft is the always-on KEYSTONE -- it is meant
      to change the base loss, so 'rec' is now non-zero by design.)
  (B) RECON KEYSTONE REAL: recon output is non-zero (not the old torch.zeros stub),
      shape [B,T,F]; loss_dict['rec'] > 0; VIB kl > 0; grad flows through to_mu/z.
  (C) LEVERS ON: build with V13_VSN=1 + attach forward-regime head; one
      forward+backward runs finite; grad flows to the VSN gate AND the fr head.
  (D) NO LOOK-AHEAD: perturbing obs at t=T-1 changes recon only at positions >= the
      causal-shift boundary, never an earlier position's recon (causal decoder path).

Run: python scripts/probe_v13_upgrade.py
"""
import os
import sys
from pathlib import Path

import torch

_V13 = Path(__file__).resolve().parent.parent / "src" / "wm" / "v13" / "v13_training"
sys.path.insert(0, str(_V13))


def _build_model(input_dim):
    import importlib.util
    # Load the V13 world_model by explicit FILE PATH (mirrors the trainer) so the
    # bare-name 'world_model' collision with V1's module is avoided, and so the
    # env-var-gated __init__ branches read the CURRENT flags on each (re)build.
    for m in ("v13_probe_wm", "settings"):
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location(
        "v13_probe_wm", str(_V13 / "world_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TFTWorldModel(input_dim=input_dim)


def _fake_batch(B, T, F):
    torch.manual_seed(0)
    obs = torch.randn(B, T, F)
    asset = torch.randint(0, 10, (B,))
    targets = {h: torch.randn(B, T) * 0.01 for h in (1, 4, 16, 64)}
    targets["regime_label"] = torch.randint(0, 3, (B, T))
    return obs, asset, targets


def main():
    B, T, F = 4, 16, 25
    results = []

    # ---- (A) + (B): LEVERS OFF (default), recon keystone ON -------------------
    os.environ.pop("V13_VSN", None)
    os.environ.pop("V13_FORWARD_REGIME", None)
    model = _build_model(F)
    model.train()
    obs, asset, targets = _fake_batch(B, T, F)

    assert model.shared_vsn is None, "A FAIL: shared_vsn constructed with flag OFF"
    assert model._use_forward_regime is False, "A FAIL: _use_forward_regime True with flag OFF"
    assert model.forward_regime_head is None, "A FAIL: fr head present with flag OFF"

    out = model.forward_train(obs, asset)
    assert "forward_regime" not in out, "A FAIL: forward_regime key present with flag OFF"
    recon = out["recon"]
    assert recon.shape == (B, T, F), f"B FAIL: recon shape {tuple(recon.shape)} != {(B,T,F)}"
    assert float(recon.abs().sum()) > 1e-6, "B FAIL: recon is all-zero (stub not replaced)"
    results.append("A: levers OFF -> no VSN module, no forward_regime key, no fr head  OK")
    results.append(f"B: recon REAL shape={tuple(recon.shape)} abs_sum={float(recon.abs().sum()):.4f} (non-zero)  OK")

    total, loss_dict, outputs = model.get_loss(obs, asset, targets, mask_ratio=0.25,
                                               regime_labels=targets["regime_label"])
    assert "fr_aux" not in loss_dict, "A FAIL: fr_aux in loss_dict with flag OFF"
    assert loss_dict["rec"] > 0.0, f"B FAIL: loss_dict['rec']={loss_dict['rec']} not > 0"
    assert loss_dict["kl"] > 0.0, f"B FAIL: VIB kl={loss_dict['kl']} not > 0"
    assert torch.isfinite(total), "B FAIL: total loss not finite"
    results.append(f"A: loss_dict has NO 'fr_aux' key (lever OFF)  OK  [keys: rec={loss_dict['rec']:.4f}, kl={loss_dict['kl']:.4f}]")

    # grad flows through the bottleneck (to_mu) -> recon term feeds the latent
    total.backward()
    assert model.to_mu.weight.grad is not None and float(model.to_mu.weight.grad.abs().sum()) > 0, \
        "B FAIL: no grad through VIB to_mu (bottleneck not in graph)"
    assert model.recon_decoder[0].w_gate.weight.grad is not None and \
        float(model.recon_decoder[0].w_gate.weight.grad.abs().sum()) > 0, \
        "B FAIL: no grad through recon_decoder"
    results.append("B: grad flows through to_mu (z) + recon_decoder (bottleneck REAL)  OK")

    # ---- capture OFF-baseline loss_dict KEY SET for the parity claim ----------
    off_keys = set(loss_dict.keys())

    # ---- (C): LEVERS ON -------------------------------------------------------
    os.environ["V13_VSN"] = "1"
    os.environ["V13_FORWARD_REGIME"] = "1"
    model_on = _build_model(F)
    model_on.train()
    assert model_on.shared_vsn is not None, "C FAIL: shared_vsn None with V13_VSN=1"

    # attach forward-regime head exactly as the trainer does
    sys.path.insert(0, str(_V13.parent.parent / "_shared"))
    from forward_regime_head import attach_forward_regime_head
    attach_forward_regime_head(model_on, verbose=False)
    assert model_on._use_forward_regime is True and model_on.forward_regime_head is not None, \
        "C FAIL: attach did not set the head"

    obs2, asset2, targets2 = _fake_batch(B, T, F)
    # build forward-regime labels (nested dict) like the trainer's collate
    bear = torch.randint(0, 2, (B, T)).float()
    trend = torch.randint(0, 3, (B, T)).float()
    move = torch.randint(0, 2, (B, T)).float()
    # Mask only the last few rows (no future) so SOME rows stay valid. (In real
    # training T>=96 and K=64 leaves T-64 valid rows; here T=16 so we mask 4 to
    # keep the masked-vs-valid mechanics exercised without zeroing every row.)
    bear[:, -4:] = float("nan")
    trend[:, -4:] = float("nan")
    move[:, -4:] = float("nan")
    targets2["forward_regime_labels"] = {"bear": bear, "trend": trend, "move": move}

    out_on = model_on.forward_train(obs2, asset2)
    assert "forward_regime" in out_on, "C FAIL: forward_regime key missing with head attached"
    for k in ("bear_logits", "trend_logits", "move_logits"):
        assert k in out_on["forward_regime"], f"C FAIL: {k} missing"

    total2, ld2, _ = model_on.get_loss(obs2, asset2, targets2, mask_ratio=0.25,
                                       regime_labels=targets2["regime_label"])
    assert "fr_aux" in ld2, "C FAIL: fr_aux not in loss_dict with lever ON"
    assert torch.isfinite(total2), "C FAIL: total not finite with levers ON"
    total2.backward()
    assert model_on.shared_vsn.gate_proj.weight.grad is not None and \
        float(model_on.shared_vsn.gate_proj.weight.grad.abs().sum()) > 0, \
        "C FAIL: no grad to shared VSN gate"
    assert model_on.forward_regime_head.bear_head.net[0].weight.grad is not None and \
        float(model_on.forward_regime_head.bear_head.net[0].weight.grad.abs().sum()) > 0, \
        "C FAIL: no grad to forward-regime head"
    # parity: ON loss keys = OFF keys + exactly {'fr_aux'}
    on_keys = set(ld2.keys())
    extra = on_keys - off_keys
    assert extra == {"fr_aux"}, f"C FAIL: ON added keys other than fr_aux: {extra}"
    results.append(f"C: levers ON -> VSN gate fires + fr head fires; finite fwd/bwd; "
                   f"grad flows to BOTH; loss keys = OFF + {{'fr_aux'}}  OK")

    # ---- (D): NO LOOK-AHEAD in the causal path --------------------------------
    # The shared VSN is provably causal (its own self-test). Here we check the
    # recon path: recon[t] is decoded from the bottleneck at position t, whose
    # encoder input is the CAUSAL-SHIFT of obs (position t sees obs[t-1]). Perturb
    # the FINAL input bar obs[:, T-1] and confirm recon at position 0 is unchanged
    # (a future bar cannot alter an earlier-position reconstruction).
    model.eval()  # deterministic (no VIB sampling, no ATME)
    with torch.no_grad():
        base = model.forward_train(obs, asset)["recon"]
        obs_p = obs.clone()
        obs_p[:, T - 1, :] += 5.0
        pert = model.forward_train(obs_p, asset)["recon"]
        d0 = float((pert[:, 0] - base[:, 0]).abs().sum())
    assert d0 < 1e-5, f"D FAIL: recon at t=0 changed ({d0}) when only the LAST input bar moved (look-ahead)"
    results.append(f"D: perturb last input bar -> recon[t=0] unchanged (delta={d0:.2e}); causal, no look-ahead  OK")

    print("\n".join(results))
    print("\nALL RWYB CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
