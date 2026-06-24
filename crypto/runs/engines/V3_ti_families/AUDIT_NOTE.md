# V3 TI Families — AUDIT NOTE (2026-06-24)

## Numbers Require Validation

The compound numbers (+47K-122K%) are REAL but primarily driven by the
REGIME GATE (bear avoidance), not the TI signals themselves.

### What's proven:
- The regime gate (V1 BTC SMA100) avoids the 2022 bear
- Being long (ANY selection) under the gate in 2020-2024 compounds enormously
- The architecture (ensemble + gate + cost) is correct

### What's NOT yet proven for non-MA families:
- Whether RSI/MACD/Bollinger/Supertrend/ADX signals add value BEYOND random
  selection under the same gate
- The random-null test (which the MA family passed at 100th percentile) has
  NOT been run for any non-MA family

### Known issues:
- Funding engine: BROKEN (proxy signal always-on at high freq → +358B% artifact)
- OI-Divergence: uses price-based proxy, not actual chimera features
- AL baseline varies by TF (rebalancing premium at higher frequency)

### What's needed before claiming these work:
1. Random-null test: same gate, random ensemble, same deploy rate → compare
2. Funding: replace proxy with actual chimera `norm_funding` feature
3. OI-Divergence: replace proxy with chimera `norm_oi_price_divergence`

### The MA family remains the gold standard:
- Random-null: 100th percentile (genuine timing, not just gate)
- OOS validated: +193.5% compound on held-out 2024 data
- Per-slice: 59% profitable, +4.11%/week, 35% deployment
