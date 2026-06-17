# Three user questions answered with RWYB (2026-06-11)

## Q1 — "Did you test MA entry + MA exit, or just MA exit?"
You caught a real gap. The intraday_oracle used a **breakout** entry + **ATR-trail/time-stop** exit
— NOT your "normal human" **MA-cross-in + MA-cross-out**. So I ran exactly that at intraday cadences
(`family_regime_map`, EMA_20_100 / EMA_50_200 crosses: golden-cross-in, death-cross-out, ± ATR
trail; u50; runs/mining/family_regime_map_u50_20260611_141252.json):

| Cadence | MA-in + MA-out (TREND class) OOS | verdict |
|---|---|---|
| **1h** | n=98,091, win 33%, PF **0.91** (sum −10,372%) | **NET NEGATIVE** — the MA cross flips on every 1h wiggle (98k trades) → whipsaw destroys it |
| **4h** | n=24,024, win 36%, PF **1.23** (sum +11,763%) | positive in sum = the trend-premium beta, but firewall-fragile (= Family2: OOS-real, UNSEEN-within-random) |

So your exact approach hits the **same wall, harder** — MA-cross-in/out at intraday is *worse* than
breakout+ATR because the cross whipsaws more. It only "works" at 4h+, where it's the slow-book beta
we already have. (MR class — RSI/Boll — is PF 0.84–0.91, net-negative everywhere.)

## Q2 — "What does Coinglass get us, and is there a free proxy before we buy?"
**Coinglass gets:** the forward liquidation **heatmap** (where leveraged-position liquidation levels
cluster *ahead* of price — the "magnets") + aggregated cross-exchange OI/funding/liq. The thesis: the
magnet shows which way price gets pulled = the continuation discriminator (the only lock on the
+6%/event intraday prize; needs OOS AUC ≥ 0.58, internal data = 0.52).

**The free proxy test** (`liq_proxy_probe.py`, u10, the D72 continuation label): I built a proxy of
the heatmap's information from on-disk data — LSR crowding, OI-fuel z-score, squeeze-direction
(oi_z/LSR), downside-liq-risk (oi_z·LSR), and a VWAP-node magnet-proximity — and added it to the
discriminator:

| feature set | OOS AUC |
|---|---|
| baseline (D72 feats) | 0.491 |
| **+ magnet-proxy feats** | **0.491** (Δ +0.000) |
| proxy-only (positioning) | 0.501 (coin-flip) |

**DECISION: SKIP Coinglass.** A reasonable free proxy of its positioning/magnet information adds
*zero* discriminative lift (stays at coin-flip). Coinglass is a cleaner/aggregated version of the
*same* positioning signal — to move 0.49 → 0.58 its precision increment would have to be enormous,
which the proxy gives no reason to expect. **Don't spend the $29/mo on this thesis.**
Honest caveat: the proxy is rough (no per-position entry prices; LSR is Binance-global; magnet is a
VWAP proxy not the true liq-level map) — a null proxy is *suggestive, not decisive* — but it gives no
green light, and the rule was "don't buy an unproven thesis." If revisited, demand a stronger *free*
signal first. (Genuinely orthogonal info — news/catalyst/social, NOT positioning — is the only
untested discriminator class left; positioning is now proxy-tested null.)

## Q3 — "If there are moves, solving for them is the way."
Agreed, and the moves are real (oracle +5.4–6.9%/event). But the "solve" = the continuation
discriminator, and we've now tested it three ways: internal price/vol/flow features (AUC 0.52),
MA-cross mechanism (whipsaw-walled), and the free positioning-magnet proxy (AUC 0.49). **The signal
that says which move continues is not in price or positioning data — free or paid.** The honest
remaining lever is either (a) the magnitude/vol channel (bet move SIZE not direction — untested), or
(b) genuinely orthogonal external info (news/catalyst), or (c) accept the slow regime-gated book.

Repro: `python -m mining.family_regime_map --universe u50 --cadences 1h,4h` ;
`python -m mining.liq_proxy_probe --universe u10`.
