# src/narrate/ -- Package Overview

> Generated 2026-06-06. Canonical reference for the narrate package.
> See also: `docs/MARKET_STRATEGY_ARCHETYPES.md` (the strategy-mode master map rendered from `src/narrate/strategy_archetypes.py`).

## Purpose

`src/narrate/` answers a single question: **given an asset, a period, and a chart type -- what is the market DOING?**

It does NOT predict. It does NOT size. It does NOT suggest exits. It describes.

The output -- a `MarketNarration` (structured + prose) -- is the intelligence layer that the
`discover` skill mines for edges and that the `trader` skill consults for context during a live
sleeve. It sits at the TOP of the value chain:

```
narrate -> discover -> trader
  (what)    (edge)     (act)
```

## Design stance (binding)

- **DESCRIPTIVE, not predictive.** Every read is "what is" not "what will be."
- **ENTRY-SIGNAL framing only.** Conditions that precede setups. Exits are out of scope here.
- **PER-SETUP, not per-candle.** A narration describes a multi-candle STATE; it never fires on
  a single bar.
- **Chart-type aware.** The same calendar period on a daily time bar vs a dollar bar vs a DIB
  tells a structurally different story. The engine can narrate each and compare what they see.

## Module map

| Module | What it does |
|---|---|
| `__init__.py` | Package entry; re-exports `narrate`, `MarketNarration`. |
| `feature_map.py` | Canonical decomposition of all chimera columns into 12 families (`FAMILIES`). Curated `Feature` records (~120+ columns) with direction polarity, human title, crypto-specific flag. `classify(col)` maps any column to a family. `coverage_report(cols)` measures curated vs tradeable coverage. |
| `crypto_context.py` | Crypto-specific caveats surfaced per active family. Ensures every narration acknowledges 24/7 continuity, perp funding/liquidation reflexivity, BTC-beta dominance, and whale-flow observability. |
| `strategy_archetypes.py` | Master map of 9 trading archetypes (`ARCHETYPES`). Encodes `OUR_MANDATE` (per-setup, 2-5%/move, long-only spot, 4h/1d). `select_for()` scores each archetype's fit deterministically: PRIMARY = swing + breakout; composable = intraday_momentum + position_trend + event_driven; conditional = mean_reversion; AVOID = scalping + hft_mm. The narration's `mode_hint` calls this. |
| `state.py` | `FamilyRead` dataclass (direction, score, intensity_pctile, salient columns). `compute_state(df, period_mask, ref_mask)` scores each family using the window's signal columns vs the asset's full history. Handles missing columns gracefully. |
| `narrator.py` | The combined engine. `narrate(asset, cadence, start, end, ...)` loads chimera via `ChimeraLoader`, slices the window, calls `compute_state`, synthesizes the regime (`_regime_synthesis`), maps to mode hint (`_mode_hint`), layers crypto caveats, and assembles `MarketNarration`. `render_prose()` converts to plain text. |
| `charts.py` | `narrate_across_charts(asset, start, end, cadences)` calls `narrate()` for each cadence and builds a `CrossChartResult` with structured divergence analysis (`_build_comparison`) + prose (`_render_prose`). `CHART_PROFILES` describes what each chart type sees and its blind spots. |
| `foundation.py` | Layer 3: MOMENT-1 foundation model (AutonLab / CMU, arXiv:2402.03885). Computes anomaly percentile, historical analog, and a mandatory validation verdict (PASS/WEAK/FAIL) correlating MOMENT scores against our own labels. Degrades gracefully -- never raises, returns `{available: False, reason: ...}` on any failure. Opt-in via `with_foundation=True`. |
| `artifacts.py` | Layer 2: our own trained models (LightGBM boosters, etc.) surfaced as descriptive reads. One bad model does not break the layer. Opt-in via `with_artifacts=True`. |

## The 3 intelligence layers

```
Layer 1 -- Chimera family decomposition     (always active)
  feature_map.py + state.py
  All chimera columns decomposed into 12 families.
  Per-family FamilyRead: direction, score, intensity_pctile, salient columns.
  Asset-relative scoring (vs the asset's own history, not cross-sectional).

Layer 2 -- Trained artifacts                (opt-in: with_artifacts=True)
  artifacts.py
  Our own LightGBM / WM descriptors as one-line characterizations per model.
  Per-model isolation: one failure does not block others.

Layer 3 -- MOMENT foundation model          (opt-in: with_foundation=True)
  foundation.py
  MOMENT-1 anomaly percentile + historical analog + validation verdict.
  Only trusted if validation returns PASS (report WEAK/FAIL explicitly).
```

## The 12 families

| Family key | Title | Key question |
|---|---|---|
| `structure` | Price structure & trend | Where is price vs its own MAs? Trending or ranging? |
| `momentum` | Momentum & returns | Which direction and how hard? Accelerating or fading? |
| `volatility` | Volatility & activity | Compressed (coil) or expanded? Vol clustering? |
| `orderflow` | Order flow & microstructure | Buyers or sellers in control? VPIN, Hawkes, OFI, Kyle-lambda. |
| `liquidity` | Liquidity & order book | Deep or fragile? Cross-exchange spread stress? |
| `derivatives` | Funding, OI & basis | Crowded long or short? Carry stressed? Funding flipped? |
| `liquidation` | Liquidations & forced flow | Cascades, capitulation, short squeezes? |
| `positioning` | Long/short ratio, smart vs retail | Global LSR, top-trader posture, smart/retail divergence. |
| `whale` | Whale & large-trade flow | Net large-print buying or selling? |
| `cross_asset` | Cross-asset & relative context | BTC beta, universe rank, ETF flows, stablecoin supply, transfer entropy. |
| `social` | Attention & social | Retail attention rising? |
| `regime` | Regime labels (precomputed) | Pipeline regime tag, Hurst, asset DNA, universe flags. |

Crypto-native families (`derivatives`, `liquidation`, `positioning`, `whale`) have no equity
analogue. Read them on their own terms. `crypto_context.py` surfaces per-family caveats in
every narration.

## Entry-only / per-setup framing

The narration is valid for entry-signal characterization only:

- The `mode_hint` field names the **strategy MODE** the described state suits (e.g. `swing`,
  `breakout`). It does NOT select a strategy or suggest an entry trigger.
- Exit logic (trailing stops, fixed targets, volatility exits) is a SEPARATE decomposable
  domain and is explicitly out of scope. Never appear in any narration output.
- The `mode_hint.avoid` list names the traps -- `scalping` and `hft_mm` -- the per-candle
  architectures that the prior project history proves are the wrong frame.

## Crypto-as-its-own-market framing

Every narration acknowledges:

1. **24/7 continuous** -- no session gaps; "intraday" is a chosen window, not a fixed structure.
2. **Perp funding + liquidation reflexivity** -- `norm_funding`, `fund_rate_z30`, and
   `liq_capitulation` / `liq_short_panic` flags are the dominant positioning signals. A crowded
   long means forced selling will amplify the next down-move nonlinearly.
3. **BTC-beta dominance** -- `xd_btc_return` + `xd_cross_return_mean` give the market backdrop.
   Idiosyncratic alt moves are significant; BTC-led moves are context.
4. **Whale flow observable** -- `wh_whale_net_usd` and the `wh_*` family are live signals.
5. **Funding carry is a SEPARATE sleeve** -- the known-robust beta+yield sleeve is OUT OF SCOPE
   for the directional narration.

## Example usage

### Python

```python
from src.narrate import narrate

# single cadence
nr = narrate("BTC", cadence="4h", start="2025-09-01", end="2025-11-01")
print(nr.to_text())        # prose
d = nr.to_dict()           # structured

# with all layers
nr = narrate("SOL", cadence="dollar", start="2025-10-01", end="2025-11-01",
             with_foundation=True, with_artifacts=True)
print(nr.to_text())

# multi-chart comparison
from src.narrate.charts import narrate_across_charts
result = narrate_across_charts("ETH", start="2025-09-01", end="2025-11-01",
                                cadences=("1d", "4h", "dollar", "dib", "range"))
print(result.prose)
d = result.to_dict()       # per_chart, divergences, consensus
```

### CLI (charts module demo)

```
python -m src.narrate.charts --asset BTC --start 2025-09-01 --end 2025-11-01
```

### Archetype map

```python
from src.narrate.strategy_archetypes import select_for
print(select_for())        # mandate-fit verdict for all archetypes
```

Or render the human doc:

```
python src/narrate/strategy_archetypes.py --write-doc
# writes docs/MARKET_STRATEGY_ARCHETYPES.md
```

## Where narrate sits in the value chain

```
narrate (this package)
  Describe the WHAT.
  Output: MarketNarration / CrossChartResult.
  Feeds: discover (the conditioner-search context) + trader (regime context for live sleeves).

discover skill
  Hunt a TRADEABLE EDGE conditioned on the described state.
  Discrimination -> harvestability -> robustness battery.
  The narration tells discover WHAT FAMILIES to probe and WHICH ARCHETYPE is in play.

trader skill
  Sizing / lifecycle / live ops on what discover ships.
  Consults narrate for ongoing regime monitoring (regime shifts trigger risk-playbook responses).
```

The chain is one-directional: narrate DOES NOT size positions. It DOES NOT validate edges.
Those are `trader` and `discover` responsibilities.

## Links

- Archetype master map: `docs/MARKET_STRATEGY_ARCHETYPES.md`
- Skill: `.claude/skills/narrate/SKILL.md`
- Discover skill: `.claude/skills/discover/SKILL.md`
- Trader skill: `.claude/skills/trader/SKILL.md`
- Apparatus audit: `docs/APPARATUS_AUDIT_2026_06_05.md`
