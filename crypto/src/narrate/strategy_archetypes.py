"""src/narrate/strategy_archetypes.py -- the MASTER MAP of trading-strategy archetypes.

User mandate (2026-06-06): "iron out the different market strategies (scalping, htf, trending), and write out their
characteristics fully (have this as a master map) and then we will select the relevant one for our problem so we
don't repeat the same mistakes."

WHY THIS EXISTS -- the per-candle trap. Prior instances (and this one's scalp-oracle) optimized PER CANDLE: capture
every 2-bar wiggle. That is the SCALPING archetype, and it is a category error for our mandate (2-5%+ net per MOVE,
hold hours-to-7d, long-only spot, daily/4h). The unit of trading is a SETUP across a MULTI-CANDLE MODE. This map
pins down each archetype's TIMESCALE, edge source, entry-signal nature, and -- critically -- whether it trades
per-candle or per-setup, so we pick the right MODE before hunting signals.

`OUR_MANDATE` encodes the project constraints; `select_for(mandate)` scores each archetype's fit and names the
match + the explicit MISFITS (the traps).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Archetype:
    key: str
    name: str
    timescale: str                 # entry-to-exit horizon
    unit: str                      # per-candle | per-setup (multi-candle) | per-event | continuous
    edge_source: str               # where the alpha comes from
    entry_signal: str              # what fires an entry (the thing we'd hunt)
    suitable_regime: str           # trend | range | high-vol | any
    instruments: str               # typical indicators/features
    crypto_fit: str                # how it manifests in crypto specifically
    failure_modes: str
    capital_profile: str           # leverage/turnover/cost sensitivity
    our_fit: int = 0               # -2 strong misfit ... +2 strong fit (vs OUR_MANDATE)
    fit_reason: str = ""


ARCHETYPES: list[Archetype] = [
    Archetype(
        "hft_mm", "HFT / market-making", "microseconds-seconds", "continuous",
        "Latency + queue priority + spread capture + inventory management; rebate harvesting.",
        "Continuous quoting; entry is a resting order filled by adverse flow, not a 'signal'.",
        "any (thrives on volume)", "L2 book dynamics, queue position, microprice, OFI.",
        "Crypto's fragmented venues + maker rebates make MM viable but co-location/infra-gated; toxic flow (VPIN) is the enemy.",
        "Adverse selection, latency arms race, inventory blowups in cascades.",
        "Massive turnover, infra-bound, cost edge IS the strategy.",
        -2, "Infra/latency-gated; not a signal-hunt; impossible without co-location. Out of scope."),
    Archetype(
        "scalping", "Scalping", "seconds-minutes (1-a few bars)", "per-candle",
        "Tiny repeated edges on micro-moves; capture the next 1-2 bars repeatedly.",
        "Micro-pullback / order-flow flip / 1-2 bar reversal -- a per-CANDLE trigger.",
        "range / high-liquidity", "1m/tick bars, order flow, micro-MAs, VWAP bands.",
        "Crypto 24/7 + thin books make scalping costly (taker fees + slippage eat the tiny edge); funding bleeds positions.",
        "Costs/slippage dominate the tiny edge; per-candle noise-fitting; over-trading.",
        "Very high turnover; cost-fragile; needs maker fills.",
        -2, "THE TRAP we fell into (scalp oracle = 2-bar wiggles). Per-candle, cost-fragile, mismatched to 2-5%/move."),
    Archetype(
        "intraday_momentum", "Intraday momentum / day trading", "minutes-hours (flat by ~1 day)", "per-setup",
        "Ride an intraday impulse/trend within the day; exit before overnight.",
        "Intraday breakout / momentum ignition / VWAP reclaim -- a multi-bar setup within a session.",
        "trend / high-vol", "5m-1h bars, intraday MAs, VWAP, RVOL, opening-range.",
        "Crypto has no 'session', so 'intraday' = a chosen window; funding accrues; BTC-led impulses dominate.",
        "Chop whipsaws; no session structure to anchor; overnight gap risk is absent but vol is continuous.",
        "High turnover; moderate cost sensitivity.",
        +1, "Multi-candle setup, but horizon shorter than our hours-to-7d band; viable on the fast end (4h)."),
    Archetype(
        "swing", "Swing trading", "hours-to-days (1-7d)", "per-setup (multi-candle)",
        "Capture a discrete multi-day MOVE/leg between structure points.",
        "Pullback-into-trend, breakout-and-hold, support reclaim, momentum-with-confirmation -- a SETUP that forms "
        "over several bars and resolves over days.",
        "trend / transitional", "4h-1d bars, swing MAs, structure (HH/HL), momentum + vol-expansion, positioning.",
        "Crypto swings are large (5-15%+ legs) and BTC-led; funding/liquidation context filters entries; weekends thin.",
        "Regime flips mid-swing; beta (BTC) overrides the idiosyncratic thesis.",
        "Low-moderate turnover; cost-robust; the sweet spot for taker fees.",
        +2, "EXACT match: 2-5%+ net per MOVE, hold hours-to-7d, long-only spot, daily/4h. Our primary mode."),
    Archetype(
        "position_trend", "Position / trend-following (HTF)", "days-weeks-months", "per-setup (slow)",
        "Ride large secular trends; let winners run; few large trades.",
        "Trend-regime confirmation (MA stack, breakout of higher timeframe range) -- slow MA-based entry.",
        "strong trend", "1d-1w bars, long MAs, Donchian/Turtle breakouts, trend filters.",
        "Crypto's power-law trends make this lucrative but drawdowns are violent; funding cost over long holds matters.",
        "Whipsaw in ranges; huge give-back at trend ends; long flat periods.",
        "Very low turnover; cost-insensitive; drawdown-tolerant capital needed.",
        +1, "Adjacent: hold can exceed 7d, but the ENTRY logic (trend confirmation) overlaps our swing mode. MAs live here, as a MODE filter -- not a per-candle trigger."),
    Archetype(
        "mean_reversion", "Mean reversion", "hours-days", "per-setup",
        "Fade statistical extremes back to a mean; sell strength / buy weakness in a range.",
        "Stretch beyond bands (z-score/Bollinger), exhaustion, RSI extreme + reversal confirmation.",
        "range / low-trend", "Bollinger/z-score, RSI, VWAP reversion, vol-of-vol.",
        "Crypto mean-reverts hard intraday but TRENDS on the daily; fading a crypto breakout is dangerous (reflexivity).",
        "Trends destroy it (the gap risk is unbounded long-only); 'catching a falling knife' in cascades.",
        "Moderate turnover; regime-fragile.",
        0, "Conditional fit: works in ranging regimes only. Useful as a SUB-mode the narrator can flag, not the primary."),
    Archetype(
        "breakout", "Breakout / volatility expansion", "hours-days", "per-setup",
        "Enter as price exits a coil/range with volume; capture the expansion leg.",
        "Range-break + volatility expansion + volume/flow confirmation -- a setup that triggers on the break.",
        "transitional (coil->expansion)", "Donchian/range, ATR/vol-squeeze, RVOL, OFI confirmation.",
        "Crypto's vol-clustering makes coils->expansions clean; liquidation cascades often IGNITE the break.",
        "False breakouts (fakeouts) in chop; needs volume/flow confirmation to avoid traps.",
        "Low-moderate turnover; cost-robust.",
        +2, "Strong fit + composes with swing: breakout is a primary ENTRY-SIGNAL family for our multi-day moves."),
    Archetype(
        "funding_carry", "Funding/basis carry (delta-neutral)", "days-weeks", "continuous (held)",
        "Harvest perp funding / futures basis while hedged delta-neutral; market-agnostic yield.",
        "Funding/basis exceeds a threshold -> put on the carry; not a directional entry.",
        "any (neutral)", "funding rate, basis, OI -- crypto-native.",
        "Purely crypto: there is no equity analogue. This is BETA+YIELD, the project's known-robust sleeve.",
        "Funding flips; basis collapse / deleveraging events; exchange/counterparty risk.",
        "Low directional risk; capital-intensive; yield-like.",
        -1, "Out of THIS scope (we hunt directional long entries), but it IS the project's known beta+yield sleeve -- note it."),
    Archetype(
        "event_driven", "Event-driven (liquidations/unlocks/listings)", "minutes-days", "per-event",
        "Trade discrete catalysts: liquidation cascades, token unlocks, listings, funding resets.",
        "A specific EVENT fires (liq cascade > threshold, listing, unlock date) -> conditional entry.",
        "any (catalyst-gated)", "liquidation flags, calendar (unlocks), exchange events.",
        "Crypto-native and rich: liquidation cascades, exchange listings, unlock cliffs, ETF flows.",
        "Events are rare (small n), crowded, and reflexive; hard to backtest with significance.",
        "Bursty turnover; high variance.",
        +1, "Composes as an ENTRY-CONDITION layer on swing (e.g. enter the swing only after a capitulation event)."),
    Archetype(
        "stat_arb_rv", "Statistical arbitrage / relative value", "hours-days", "per-setup (cross-sectional)",
        "Exploit cross-sectional mispricings: pairs, baskets, lead-lag, cross-exchange.",
        "Cross-sectional z-score / rank divergence / lead-lag signal -> long-short or relative entry.",
        "any (market-neutral)", "cross-asset rank (xrel_*), pairs spreads, lead-lag.",
        "Crypto lead-lag (BTC->alts) is real but strongest intraday; cross-exchange + pairs are crowded.",
        "Crowded; relationships break; daily lead-lag is weak (we tested: null held-out).",
        "Moderate turnover; needs short leg (we are long-only -> limited).",
        -1, "Limited by long-only-spot; cross-sectional RANK is still a useful narrator CONTEXT, not a standalone strategy here."),
]

ARCHETYPES_BY_KEY = {a.key: a for a in ARCHETYPES}


# ---------------------------------------------------------------------------
OUR_MANDATE = {
    "per_move_net_target": "2-5%+ net",
    "hold_band": "hours to <7 days",
    "constraints": "LONG-ONLY, SPOT, leverage=1",
    "primary_resolution": "4h / 1d (cost-clearing favors these)",
    "unit": "per-setup (multi-candle MOVE), NEVER per-candle",
    "exit": "OUT OF SCOPE here -- a separate decomposable domain (trailing/fixed/volatility)",
    "objective": "robust held-out compound return (wealth), not Sharpe, not per-bar IC",
}


def select_for(mandate: dict = None) -> dict:
    """Rank archetypes by fit to the mandate; name the primary mode, the composable entry-signal families, and the
    explicit MISFITS (the traps to avoid). Deterministic -- this is the 'pick the mode' decision support."""
    ranked = sorted(ARCHETYPES, key=lambda a: a.our_fit, reverse=True)
    primary = [a for a in ranked if a.our_fit >= 2]
    compose = [a for a in ranked if a.our_fit == 1]
    misfit = [a for a in ranked if a.our_fit <= -2]
    return {
        "mandate": mandate or OUR_MANDATE,
        "primary_mode": [a.key for a in primary],
        "composable_entry_layers": [a.key for a in compose],
        "conditional": [a.key for a in ranked if a.our_fit == 0],
        "misfit_traps": [a.key for a in misfit],
        "verdict": ("PRIMARY MODE = swing (multi-day MOVE) + breakout as the entry-signal engine; compose "
                    "intraday-momentum (fast end), position-trend (MAs as a MODE filter, not a per-candle trigger), "
                    "and event-driven (liquidation-gated entries). AVOID scalping + HFT (per-candle / infra-gated = "
                    "the trap). Mean-reversion only in confirmed ranges. Funding-carry is the separate beta+yield sleeve."),
    }


def to_markdown() -> str:
    """Render the master map as a doc."""
    L = ["# Market Strategy Archetypes -- MASTER MAP",
         "",
         "> Generated from `src/narrate/strategy_archetypes.py`. The map exists to **select the right MODE for our "
         "mandate** and stop trading per-candle. Edit the structured source, not this file.",
         "",
         "## Our mandate (the selection constraints)",
         ""]
    for k, v in OUR_MANDATE.items():
        L.append(f"- **{k}**: {v}")
    sel = select_for()
    L += ["", "## Selection verdict", "", sel["verdict"], "",
          f"- **Primary mode:** {', '.join(sel['primary_mode'])}",
          f"- **Composable entry layers:** {', '.join(sel['composable_entry_layers'])}",
          f"- **Conditional (regime-gated):** {', '.join(sel['conditional'])}",
          f"- **Misfit traps (AVOID):** {', '.join(sel['misfit_traps'])}", "",
          "## The archetypes (full characteristics)", ""]
    fit_label = {-2: "AVOID (trap)", -1: "out of scope", 0: "conditional", 1: "composable", 2: "PRIMARY"}
    for a in sorted(ARCHETYPES, key=lambda x: x.our_fit, reverse=True):
        L += [f"### {a.name}  --  *{fit_label[a.our_fit]}*",
              f"- **Timescale:** {a.timescale}  |  **Unit:** {a.unit}  |  **Suitable regime:** {a.suitable_regime}",
              f"- **Edge source:** {a.edge_source}",
              f"- **Entry signal (what we'd hunt):** {a.entry_signal}",
              f"- **Instruments:** {a.instruments}",
              f"- **Crypto fit:** {a.crypto_fit}",
              f"- **Failure modes:** {a.failure_modes}",
              f"- **Capital profile:** {a.capital_profile}",
              f"- **Our fit:** {a.fit_reason}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    if "--write-doc" in sys.argv:
        from pathlib import Path
        out = Path(__file__).resolve().parents[2] / "docs" / "MARKET_STRATEGY_ARCHETYPES.md"
        out.write_text(to_markdown(), encoding="utf-8")
        print(f"[OK] wrote {out}")
    else:
        import json
        print(json.dumps(select_for(), indent=2))
