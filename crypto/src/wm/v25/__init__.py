"""V25 — Frontier Crypto WM (first-principles synthesis).

A single architecture combining five first-principles innovations, none of
which appear together in any published time-series WM:

  1. Channel-tokenized cross-feature attention (iTransformer base) — no
     timestamp-sync requirement for cross-asset interaction
  2. HARD-CODED crypto period embeddings (8h funding / 24h UTC / 7d weekly /
     30d monthly) — known a priori, not discovery problems
  3. Hurst-regime conditioned FFN (per-bar bull/sideways/bear gating) —
     bull/bear/sideways have qualitatively different statistical structure
  4. Rate-budget VIB (auto-tuned β to hit bits-per-timestep target) —
     information-theoretic instead of cargo-cult β tuning
  5. Tail-adaptive Huber loss (upweight |target| > 2σ) + adversarial regime
     training (upweight worst-quintile regime per batch) — heavy-tailed
     crypto reality + anti-fragile by construction

Designed for our anti-fragile crypto regime, not for paper benchmarks.
Iron-clad by mechanism, not by citation.
"""
