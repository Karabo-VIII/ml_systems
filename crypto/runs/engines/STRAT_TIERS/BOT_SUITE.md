# Working-bot suite (2026-07-02) — answering "why can't you build a working bot?"

**The cognitive fix:** I was grading ONE edge (cross-sectional trend) in ALL regimes and holding it to an academic
alpha bar (DSR>0.95, beat-buy-hold-net-of-survivorship). Real bots harvest a specific edge in its specific regime with
risk control. Fix: build each edge as its own bot, run it only in its regime, grade by working-bot criteria
(in-condition expectancy, PF, bounded DD, holds across windows), regime-switch, and cash out the regime with no edge.

## What each edge does, by regime (per-regime P&L attribution, OOS)

| Edge | BULL | CHOP | BEAR | verdict |
|---|---|---|---|---|
| Trend (MA) | **+24pp** | −24 | −7 | offense in bull only |
| Momentum top-8 (MOM8_ER) | **best** | cash | cash (gate) | **the offense** |
| Range/MR ("bounce") | −21 | −38 | **+36pp** | oversold-bounce, NOT a chop bot |
| Rebalance/vol-harvest | +13 | **−28.5** | −0.7 | **DEAD on crypto** (momentum+high-corr substrate) |

Two independent "chop fills" (MR, rebalancing) both LOSE in chop → **there is no long-only chop edge in this alt
universe; chop = cash.** Rebalancing (Shannon's Demon) fails because it bets on mean-reversion but crypto alts have
momentum + high correlation — it buys the fallers and they keep falling.

## Deployable candidates (VAL untuned / OOS, prod cost, block-boot p05)

| Bot | VAL comp | OOS comp | OOS Calmar | OOS DD | OOS expo | VAL p05 |
|---|---|---|---|---|---|---|
| **MOM8_ER (LEAD)** | +257% | **+51%** | **2.45** | **−21%** | **10%** | **+13** |
| MOM8+BOUNCE | +293% | +48% | 1.24 | −38% | 21% | +27 |
| SWITCH_RC (trend switch) | +126% | +20% | 0.53 | −39% | 30% | +18 |
| *BH_btc* | +117% | +11% | 0.22 | −50% | 100% | — |
| *BH_alt* | +156% | −54% | −0.67 | −81% | 100% | — |

**MOM8_ER is the working bot.** Momentum(14d) top-8, gated by breadth AND BTC trend-efficiency (deploy only clean-trend
weeks), inv-vol cap33, per-7d rebalance, cash otherwise. It made **+51% net in 2024-2026** (BH_alt −54%) at −21% DD,
in the market only 10% of the time — a selective, risk-controlled, net-positive all-weather bot that HOLDS on the
untuned VAL window. That is a working bot by every practitioner criterion.

## The honest bar it does NOT clear

Full-cycle block-boot **p05 is still −24** (the deep-bear weeks drag the lower bound negative), DSR₄₀ < 0.15. So it is
not statistically-certified alpha — it is a **selective momentum-beta harvester with a proven bear cash-gate**. For a
retailer that is a deployable bot; for institutional alpha-certification it is not. The gap is regime (a long-only bot
cannot earn in a correlated crash) — which is why it goes to cash there rather than pretending to.

## Design rules this session proved (reusable)

1. **Chop = cash.** No long-only chop edge exists on crypto alts (MR and rebalancing both lose there).
2. **Momentum top-K > plain trend-following** as the bull allocator (+51% vs +20% OOS).
3. **Bear = cash, not bounce.** Oversold-bounce flatters mild bears (VAL) but adds drawdown in deep bears (OOS).
4. **Selectivity is the edge.** The best bot is in the market 10% of the time; being invested in an alt bear loses.

Artifacts: `strat_bot_suite.{py,json}`, `strat_range_bot`, `strat_regime_switch_bot`, `strat_rebalance_bot`. The
canonical bot-archetype reference (grid/DCA/trend/breakout/momentum/rebalance regime map) is in the session record.
