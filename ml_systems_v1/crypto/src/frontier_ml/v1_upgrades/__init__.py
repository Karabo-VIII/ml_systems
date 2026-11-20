"""V1.x upgrade modules — drop-in replacements / wrappers per browser dialog.

Each module is independent and feature-flag-gated in `train_world_model.py`
so we can A/B-test upgrades cleanly without breaking the V1.x baseline.

Modules:
    sam.py         — Sharpness-Aware Minimization (B003 R1)
                     Wraps any base optimizer; expects two forward+backward
                     passes per step (closure or manual first_step/second_step).
                     Source: github.com/davda54/sam (canonical implementation).
    pcgrad.py      — PCGrad gradient surgery for multi-horizon (B003 4.6)
                     Resolves gradient conflicts between the 4 horizon heads
                     before the optimizer step. Yu 2020 arxiv 2001.06782.
    mtp_head.py    — Multi-Token Prediction sequential head (B002 R1)
                     Replaces independent {h1,h4,h16,h64} heads with a causal
                     chain h1 -> h4 -> h16 -> h64 sharing intermediate states.
                     DeepSeek-V3 style.
    fraug.py       — FrAug frequency-domain augmentation (B003 R2)
                     FFT/IFFT mask augmentation on input feature sequences.
                     ~0 marginal training cost; lifts ShIC by promoting
                     spectral invariance. arxiv 2302.09292.

All four target the existing V1.x architecture (TransformerWorldModel +
RSSM + multi-horizon TwoHot heads). They are designed to be tested
independently then composed if all probe-positive.

Decision rules per upgrade are documented in
src/frontier_ml/browser_dialog/RESPONSE_B003_v0plus_envelope_push.md and
RESPONSE_B002_frontier_lab_overlay.md.
"""
