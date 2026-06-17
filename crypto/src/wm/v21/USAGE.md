# V21 — Mamba-3 + Latent Neural-ODE Hybrid (Library Backbone)

> **Role in cohort**: Continuous-time dynamics conditioned on bar duration
> (addresses dollar-bar irregularity). **Library-only**; no trainer yet.
>
> **Verdict (2026-05-16)**: BUILD-CONDITIONAL — capacity bump needed (1.98M
> → 4M+); architectural novelty is real (bar_duration conditioning addresses
> a known V1-V14 gap). Plausible after V4 SOTA-2026 trains and validates the
> Mamba family.

## Purpose

Dollar bars trigger at irregular wall-clock intervals: a quiet day might
have 5 bars, a volatile day 500. V1-V14 treat each bar as having unit
duration — which is **architecturally wrong** for time-dependent processes
(funding cycles, mean reversion timescales).

V21 addresses this via:
- **Mamba-3 backbone** (V4's pattern; linear-time sequence scaling)
- **Latent Neural ODE residual** at end of stack: integrates dynamics
  `dx/dt = f(x)` over per-bar duration `dt`
- **Bar-duration conditioning**: each bar carries its `norm_bar_duration`
  in chimera; V21 reads it explicitly

The bet: per-bar duration conditioning + continuous-time integration give
V21 a structural advantage on irregular dollar bars that pure Transformer
or Mamba (with implicit unit-time assumption) lacks.

## Architecture (current backbone)

```
Obs (B, T, F=34) + bar_durations (B, T)
  └── input_proj → d_model=256
       └── 4× Mamba3Block (reuses V4's components)
            └── mamba_norm → h_seq [B, T, 256]
                 └── LatentNODE block:
                      ├── x = h_seq.norm()
                      ├── For step in 0..n_steps=4:
                      │    └── x = x + sub_dt * f(x)        # Euler
                      └── h_seq + (x - h_seq_norm)         # residual
                            └── norm_out → h_seq [B, T, 256]
                                 ├── h_pool = h_seq[:, -1, :]  (last bar only!)
                                 └── return_heads × {1,4,16,64}(h_pool)
```

## Smoke (2026-05-16 verified)

```
[v21-mamba-node] params: 1,979,548 (1.98M)
[v21-mamba-node] return_logits + bar-dur conditioning OK
[v21-mamba-node] backward OK
[v21-mamba-node] PASS smoke
```

## Status: BUILD-CONDITIONAL

| Axis | Assessment |
|---|---|
| Architecture novelty | ✅ ONLY cohort member with explicit bar_duration conditioning |
| Code quality | ✅ Clean; uses V4's Mamba3Block (consistent); smoke + backward |
| Capacity | ❌ 1.98M = 2x BELOW iron-clad floor. Needs bump |
| Anti-memo | ❌ No RSSM / VIB / ATME in current backbone; only Mamba's natural state-space limits |
| Speed | ✅ Mamba linear-time; NODE adds ~4 fwd passes per bar (modest) |
| Trainer | ❌ NOT BUILT |
| Cohort fit | ⚠ Pools to LAST bar only (`h_pool = h_seq[:, -1, :]`) — drops per-bar prediction signal that V1-V14 use |
| Bar-duration ready? | ✅ Yes — `norm_bar_duration` already in chimera_legacy per V21 docstring |

## To convert V21 from LIBRARY to PRODUCTION WM

Required work (~1.5-2 weeks):

1. **Per-bar prediction** (~2 hours): replace `h_pool = h_seq[:, -1, :]` with
   per-bar prediction (drop pooling). Aligns with V1-V14 multi-horizon
   per-bar TwoHot loss.

2. **Capacity bump** (~1 hour): bump `n_mamba_layers` 4 → 6, `d_model`
   256 → 320 → ~4M+ params.

3. **Add anti-memo stack** (~1 day): RSSM bottleneck + ATME + VIB
   (mirror V4's pattern).

4. **Settings.py** (~2 hours): cohort canonical invariants + bar_duration
   feature index lookup + Headline flags (CC-H5/H6/FiLM).

5. **Trainer** (~3-4 days): adapt V4's trainer. Critical addition:
   pass `bar_durations` from chimera into forward(). Verify
   norm_bar_duration column exists in chimera_legacy.

6. **Validate world** (~1 day): per-horizon IC measurement; compare to V4.

7. **First SOTA training** (~3-4 GPU-d): measure if bar-duration conditioning
   actually lifts IC vs V4 (the natural baseline since both are Mamba-3).

## Headline projection

If V21 gets the full SOTA-2026 treatment:
- Expected IC: 0.060-0.085 (Mamba-3 family lift + bar-duration novelty)
- Expected ShIC: 0.030-0.045
- Verdict: **plausible Trader-tier**; uncertain if it crosses Headline (0.10)
  — needs first training to know

## V21 Achilles heel

The bar_duration conditioning is interesting BUT chimera_legacy's
`norm_bar_duration` distribution needs to be checked: if it's heavily
concentrated near 1.0 (most bars trigger near unit time), the NODE residual
is mostly a no-op. Verify with:

```python
import polars as pl
df = pl.read_parquet("data/processed/chimera_legacy/dollar/btcusdt_v50_chimera_*.parquet")
print(df["norm_bar_duration"].describe())
# If std < 0.1 and 90th percentile near 1.0, the bar-duration signal
# is too weak for V21's NODE to help
```

If norm_bar_duration is degenerate (low variance), V21's architectural
novelty evaporates and it reduces to "Mamba with extra MLP layer". In that
case, archive V21.

## Files

```
src/wm/v21/
├── __init__.py
└── v21_mamba_node.py     # V21Backbone + LatentNODEBlock + smoke()
                          # No v21_training/ dir yet
```

## Cross-references

- B005 §3 + MODE arXiv 2601.00920 — continuous-time forecasting paper
- V4's `components.py` (`Mamba3Block`) — reused
- `docs/WM_VERSION_INVENTORY_2026_04_29.md` — V21 listed
- V16/V17 (sibling library stubs) — similar build-status, different verdicts
