# Deploy decision — the one honest candidate + the risk-point choice (2026-06-11)

The re-grade ([STRATEGY_LEADERBOARD_2026_06_11.md](STRATEGY_LEADERBOARD_2026_06_11.md)) leaves exactly
one honest, robust candidate: **the regime-gated trend book** (0.5·regime-beta SMA + 0.5·TSMOM
ensemble, equal-risk, vol-scaled, LO+spot+lev=1, cash in bear). This is the charter's Fork-A core. It
is not a strict-UNSEEN-positive ship; it is a robust **full-cycle wealth + capital-preservation**
book whose honest behavior is published below. The only open decision is the **risk point**, which is
a user call (it trades expected return against bear-drawdown).

## The risk-point frontier (u50, 1d, taker; RWYB)
| Book | Full-cycle ann% | Full-cycle DD | full p05 | UNSEEN (2026 bear) | character |
|---|---|---|---|---|---|
| regime_beta | +73% | −55% | +106 | −3.5% | aggressive (pure regime-gated trend) |
| BLEND_75r | +69% | −52% | +101 | −3.1% | aggressive-balanced |
| **BLEND_50r** | +64% | −49% | +88 | −2.8% | **balanced (charter default)** |
| BLEND_25r | +59% | −49% | +73 | −2.4% | defensive |
| TSMOM_breadth | +54% | −49% | +59 | −2.1% | most defensive |
| (buy & hold) | +70% | −82% | −69 | −18.3% | the benchmark it beats on risk |

All variants: full-cycle block-bootstrap p05 strongly positive, decisively beat buy&hold AND a random
null, and preserve capital in the bear (UNSEEN −2 to −3.5% vs market −18%).

## Honest expectations — PUBLISHED BEFORE any deploy (the trust contract)
- **It is beta+regime-gate, not market-neutral alpha.** It makes money by riding trends in up-regimes
  and sitting in cash in down-regimes. In a sustained long-only bear it **preserves** (loses a little),
  it does **not earn**. Do not expect positive returns every quarter.
- **Win/lumpiness:** per-window (1d/3d) returns are a coin-flip (~50% positive, median ≈ 0%); the
  wealth comes from the trend tail over the full cycle, not from per-window positivity. Expect long
  flat/underwater stretches inside a positive full-cycle.
- **Drawdown:** budget for ~the full-cycle DD of the chosen variant (−49% to −55%) and assume the
  realized DD can exceed the backtest. Size accordingly (the charter envelope: vol-target, heat cap,
  daily kill-switch, tiered DD ladder).
- **What it is NOT:** it does not capture the daily movers and does not earn in bears — those need
  shorts/perps (LO-exception sign-off) or external leading data (Coinglass/on-chain, parked).

## The decision (user)
1. **Risk point:** which variant — defensive (BLEND_25r / TSMOM, shallower bear) vs balanced
   (BLEND_50r) vs aggressive (regime_beta, highest full-cycle, deepest DD)? Default = BLEND_50r.
2. **Paper-trade greenlight:** wire the chosen variant to stage-04 paper trading (no real capital) to
   confirm live fills match the model before any capital question.
3. **The two BUILD gaps remain user-gated** (not buildable within LO+spot+internal data, which the
   re-grade confirmed is exhausted): perp bear-short sign-off, and the external-data discriminator.

Repro: `python -m strat.tsmom_ensemble --universe u50 --cadence 1d` (the books) ·
`python -m strat.regrade_leaderboard --universe u50` (the scorecard grade).
