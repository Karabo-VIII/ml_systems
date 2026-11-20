"""Generate docs/CHIMERA_FEATURE_DICTIONARY.md from the source-of-truth narrate.feature_map (every feature listed
with its meaning) + the hand-synthesized ASSET-CONDITIONAL interpretation per family (from the 2026-06-09 research
scouts). RWYB: the per-feature tables are generated from feature_map, not hand-transcribed. No emoji."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from narrate import feature_map as fm  # noqa: E402

# ASSET-CONDITIONAL read per family: how this family's signals read differently across archetypes.
# Synthesized from the 2026-06-09 research scouts (derivatives mechanics + asset archetypes).
ASSET_COND = {
    "structure": "Math is archetype-invariant, but RELIABILITY scales with depth: trend/Hurst/MA-distance are clean on "
                 "BTC/ETH (deep, continuous) and gappy/whipsawy on memes (thin, reflexive). On stablecoins all "
                 "price-structure features are near-zero noise around the peg (ignore).",
    "momentum": "BTC momentum is institutional + slow (ETF/treasury flows); large-alt momentum is narrative-reflexive "
                "(ETF/staking rumors); MEME momentum is pure attention reflexivity -- violent, self-fulfilling, and the "
                "exit is when attention peaks. Same return value, opposite durability.",
    "volatility": "DVOL/implied-vol features exist ONLY for BTC/ETH (Deribit) -- null elsewhere. Realized-vol/jump "
                  "features are universal but the BASELINE differs ~4x by tier (BTC ~50-60% ann -> meme >200%); a "
                  "'high vol' z-score must be read against the asset's own regime, not a global bar.",
    "orderflow": "Hawkes/VPIN/flow-imbalance/tick features are GENUINE microstructure on BTC (real two-sided flow) but "
                 "MANIPULABLE on thin meme/micro books (wash trading, spoof-driven taker prints). Trust order-flow "
                 "signals in proportion to book depth; on memes treat them as manipulation surface, not information.",
    "liquidity": "LOB depth / spread / venue-count are a real liquidity gauge on BTC/ETH (multi-layer books) and "
                 "largely SYNTHETIC on memes (spoofed walls, 10-30% spreads in thin hours -- a '$500k wall' may be the "
                 "entire visible book). Depth deterioration on a stablecoin's book is a depeg warning.",
    "derivatives": "THE most archetype-dependent family. FUNDING: on BTC = crowded institutional leverage with real "
                   "mean-reversion at extremes (contrarian); on a MEME = often one whale dominating a thin perp, or the "
                   "insider short-leg of a pump-and-dump. BASIS/premium: reliable leverage gauge on BTC (CME+perp), "
                   "noise on memes (perp-only). OI: a true leverage thermometer on BTC; on memes it can be one actor and "
                   "vanish overnight. On stablecoins, any funding/basis deviation = a depeg/credit signal, not direction.",
    "liquidation": "On BTC/ETH liquidation cascades are causal-but-BOUNDED (deep spot support) -> the post-cascade "
                   "bounce is a contrarian-long regularity (capitulation). On thin alts/memes a liquidation is often a "
                   "TERMINAL event (50%+ in minutes), NOT a bounce setup. Cross-collateral DeFi adds an on-chain "
                   "liquidation layer for DeFi tokens.",
    "positioning": "Top-trader LSR / smart-vs-retail divergence is meaningful on BTC (deep, real institutional "
                   "participants). On memes 'top traders' may be a few coordinated wallets -> the smart-money read "
                   "breaks down; taker imbalance on thin books is easily manufactured by a single whale testing "
                   "liquidity.",
    "whale": "On BTC whale flow is institutional + trackable but gameable; ~15-40% of alerts are internal/custodial "
             "transfers (noise). On MEMES whale flow is ADVERSARIAL -- the whale is typically the deployer/insider "
             "executing an exit; 'whale accumulation' is frequently the pump leg before the dump. DeFi: TVL-flow whales "
             "carry genuine fundamental signal (protocol inflow = accumulation).",
    "cross_asset": "These encode BTC-beta + cross-sectional RANK. For BTC they are self-referential (~identity). For "
                   "large-alts they capture the ~65-70% BTC-led component. For memes BTC explains <30% of variance -> "
                   "cross-asset features under-describe them (idiosyncratic reflexivity dominates). Cross-sectional rank "
                   "(xrel_*) is where the 1-week reversal + dispersion edge lives.",
    "social": "Attention INVERTS by archetype: on BTC social/attention is coincident-to-lagging noise (a sentiment "
              "gauge, not a trigger). On MEMES attention IS the fundamental -- the only asset -- and social velocity "
              "leads price with a short lag; the catch: peak social = insider exit timing. DeFi: a partnership/audit "
              "post can carry real content.",
    "regime": "Regime labels (SMA200 / Hurst / DNA) are COHORT-WIDE, BTC-driven -- a regime call is really a market-"
              "state call. regime_label encodes price-vs-trend position (above/below MA). Same label means the same "
              "thing across assets BECAUSE it's the market regime, not an asset property; but its TRADING implication "
              "differs (a bull regime in a meme is far more fragile than in BTC).",
}

# Plain-language 'what it tells us on its own' per family (the standalone read).
STANDALONE = {
    "structure": "Where price sits relative to its own trend/structure + how trendy vs choppy the tape is.",
    "momentum": "Direction + speed of recent price/flow drift across horizons.",
    "volatility": "How big moves are + are getting (realized, jumps, implied) -- the predictable channel.",
    "orderflow": "Who is pressing -- aggressive buy vs sell flow, trade intensity, informed-flow toxicity.",
    "liquidity": "How deep/tight/fragile the book is -- the cost-of-execution + slippage gauge.",
    "derivatives": "Leverage + positioning + carry -- funding, basis, OI, premium (the reflexivity fuel).",
    "liquidation": "Forced flow -- where leverage is breaking and which way it overshoots.",
    "positioning": "Who is on which side -- long/short ratios, smart vs retail, taker imbalance.",
    "whale": "Large-actor flow -- accumulation vs distribution (institutional on BTC, insider on memes).",
    "cross_asset": "How the asset sits vs BTC + the rest of the universe -- beta + cross-sectional rank.",
    "social": "Attention/narrative intensity -- reflexive fuel (THE signal on memes, noise on BTC).",
    "regime": "The precomputed market-state label (trend/Hurst/DNA) the asset is in.",
}

HEADER = """# Chimera Feature Dictionary (2026-06-09)

**Purpose.** Understand every chimera feature individually -- what it MEANS, what it TELLS US on its own, AND how its
interpretation CHANGES when pegged to a particular asset archetype (does funding rate mean the same on BTC as on a
meme? -- no). Per-feature meanings are generated from the source-of-truth `src/narrate/feature_map.py` (218 curated
features, 100% column coverage, 12 families); the asset-conditional layer is synthesized from the 2026-06-09 research
scouts (see [CRYPTO_MARKET_UNDERSTANDING.md](CRYPTO_MARKET_UNDERSTANDING.md) for citations).

**How to read.** Each family below gives: (a) what it tells us on its own, (b) the ASSET-CONDITIONAL read (BTC vs
large-alt vs mid/DeFi vs meme vs stablecoin), (c) every feature with its meaning + polarity (+1 high=bullish,
-1 high=bearish, 0 contextual). For a single (asset, period, timeframe) VIEW of these features, use the decomposer:
`python -m mining.decompose --asset <SYM> --cadence <TF> --start <d> --end <d> --plots`.

## The headline: the SIGNAL x ARCHETYPE interpretation matrix
The same reading means different things by archetype (synthesized from the derivatives-mechanics + archetype scouts):

| Signal | BTC (beta/reserve) | ETH/large-cap L1 | Mid-cap / DeFi | MEME | Stablecoin |
|---|---|---|---|---|---|
| **Funding rate** | crowded institutional leverage; strong contrarian at extremes | same, noisier; narrative-driven spikes revert faster | thin perp -> a 'high funding' may be ONE whale, not crowding | often manipulated; may be the insider short-leg of a pump | N/A -- deviation = depeg/credit signal, not direction |
| **Basis / premium** | reliable leverage gauge (CME+perp) | reliable, faster-moving | perp-only, noisy | meaningless multi-day | deviation = depeg risk |
| **Open interest** | true leverage thermometer ($60-70B) | reliable, ~1/3 of BTC | fragile (single-actor) | unreliable; vanishes on rugs | N/A |
| **Liquidations** | causal + BOUNDED -> contrarian bounce | larger cascades (higher beta) | violent/fast (thin) | usually TERMINAL, not a bounce | protocol-specific only |
| **Whale flow** | institutional, trackable, gameable | protocol/staking flows | TVL-flow = real; spot ambiguous | ADVERSARIAL (insider exit) | treasury/peg arb |
| **Social / attention** | coincident-lagging noise | leads 1-3d on narratives | partnership/audit = content | IS the signal (peak = exit) | depeg-rumor tail only |
| **Order-book depth** | real liquidity gauge | moderate | thin, spoofable | synthetic/spoofed | deep; deterioration = depeg |
| **Implied vol (DVOL)** | high (Deribit) | high (Deribit) | N/A | N/A | N/A |

**Cross-cutting rule:** trust order-flow / depth / whale / positioning signals IN PROPORTION TO LIQUIDITY DEPTH. They
are information on BTC and a manipulation surface on memes. Stablecoins invert all return-based logic (peg process).

---
"""


def main():
    lines = [HEADER]
    fam_titles = {k: fm.FAMILIES[k].title for k in fm.FAMILIES}
    fam_q = {k: fm.FAMILIES[k].question for k in fm.FAMILIES}
    by_fam = {}
    for col, f in fm.FEATURES.items():
        by_fam.setdefault(f.family, []).append(f)
    for fam in fm.FAMILY_ORDER:
        feats = by_fam.get(fam, [])
        lines.append(f"## {fam_titles.get(fam, fam)}  (`{fam}`, {len(feats)} features)")
        lines.append(f"*Question it answers:* {fam_q.get(fam,'')}")
        lines.append(f"\n**On its own:** {STANDALONE.get(fam,'')}")
        lines.append(f"\n**Asset-conditional:** {ASSET_COND.get(fam,'')}\n")
        lines.append("| feature | meaning | polarity |")
        lines.append("|---|---|---|")
        for f in feats:
            pol = {1: "+1 bullish-high", -1: "-1 bearish-high", 0: "0 contextual"}.get(f.polarity, str(f.polarity))
            cr = " *(crypto-specific)*" if f.crypto_specific else ""
            desc = (f.desc or "").replace("|", "/").replace("\n", " ")
            lines.append(f"| `{f.col}` -- {f.title}{cr} | {desc} | {pol} |")
        lines.append("")
    out = ROOT / "docs" / "CHIMERA_FEATURE_DICTIONARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[written] {out}  ({len(lines)} lines, {sum(len(v) for v in by_fam.values())} features)")


if __name__ == "__main__":
    main()
