# `src/frontier_ml/` — Frontier-tier WM approaches for the daily-bar regime

> **Status (2026-05-01):** PLANNING + LITERATURE REVIEW only. No code yet.
> The user-mandated build sequence is: pipeline first, then docs + poke,
> then SOTA-validate, then build.

## Goal

Push the WM signal from current SHIP-tier (IC ≈ 0.06 / ShIC ≈ 0.03) into
the **Headline tier** (IC > 0.10 / ShIC > 0.05) where the WM signal IS
the alpha — not just a position-sizing input. The full ladder per
CLAUDE.md INDISPUTABLE OPERATING LENS:

```
Filter      IC > 0.015, ShIC > 0.015
Sizer       IC > 0.030, ShIC > 0.020
Trader      IC > 0.050, ShIC > 0.030     [V1.x is here]
Headline    IC > 0.10,  ShIC > 0.05      [PRIMARY TARGET]
Ambitious   IC > 0.13,  ShIC > 0.065
Capacity    IC > 0.20,  ShIC > 0.10      [requires V20 tick-level]
```

## Hardware constraint (HARD)

- **GPU:** 1× RTX 4060 (8 GB VRAM)
- **CPU:** Intel i9 (~16-24 cores)
- **RAM:** assumed 32-64 GB (Windows host)

This rules out:
- ❌ 100M+ parameter foundation models (8 GB can't hold weights + grads + Adam states + activations at usable batch size)
- ❌ Frontier-LLM-scale pretraining (10K H100 territory)
- ❌ Tick-level multi-million-token context windows on the GPU

This *enables*:
- ✅ 5-30M parameter models comfortably (V1.x = 2M, V11-V14 = 2-7M)
- ✅ ~50M params with gradient checkpointing + small batch
- ✅ Smart self-supervision at modest scale (architecture > parameters)
- ✅ Distillation from an ensemble of medium models into a deployable single
- ✅ CPU-side data work (slim caches, polars, multi-threaded ingest)

## Files

- `README.md` — this orientation
- `PLAN.md` — actual approach with explicit holes poked
- `LITERATURE.md` — relevant SOTA papers, what they tell us, what they don't
- `(future)` — code stubs once plan is validated

## Build order

1. **Pipeline complete** (in flight, 2026-05-01)
2. **Document + poke** ← THIS phase
3. **SOTA-validate against literature** — every load-bearing claim must
   trace to a paper or a documented experiment
4. **Build** — one clear bet at a time, never multiple parallel
   architectures (we have ONE GPU)

## Provenance

User mandate 2026-05-01: build a frontier-tier WM for crypto
within hardware constraints. "Compute is replaceable, intelligence is
what we're building." Document + poke first; pipeline must land first.
