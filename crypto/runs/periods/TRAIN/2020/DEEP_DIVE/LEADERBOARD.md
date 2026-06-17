# 2020 emergent strategies -- RANKED per timeframe + COVERAGE + pitfalls (`deep2020_leaderboard.py`)

User /orc 2026-06-13: "best emergent strategies, ranked, several per timeframe; pitfalls; and COVERAGE --
are we in the market every day?" Candidate causal long-only strategies graded on the 2020 OOS (Oct-Dec):
per-MA-type FAMILY (the slow-config book of each MA type) + BUYHOLD + VOLTGT_BH. avg_exp = average exposure
0..1 (the real time-in-market); coverage = % of days with >50% exposure.

## THE KEY TENSION: ranking metric changes the winner (Sharpe rewards UNDER-participation)
- **By NET (the North Star = WEALTH):** BUYHOLD and VOLTGT_BH win at EVERY timeframe. The MA families ALWAYS
  have lower net (they sit out part of the bull). E.g. 1d: buy-hold 47% vs best MA-family 31%; 15m: buy-hold
  154% vs best MA-family 55%.
- **By SHARPE:** at COARSE (1d/2h) the low-exposure MA families edge out buy-hold -- but ONLY because they are
  barely invested (low exposure -> low vol -> high Sharpe). That is UNDER-PARTICIPATION, not skill (VIDYA@4h
  Sharpe 3.42 but net 18% at 22% exposure vs buy-hold net 50%). At FINE (30m/15m) buy-hold/vol-target win on
  BOTH Sharpe AND net (the cost-bearing MA families degrade: 15m HMA/TEMA net 2.5-3.9%, Sharpe ~0.5).

## Ranked tables (OOS Sharpe; net% / maxDD% / avg_exp / coverage)
**1d:** 1.EMA-fam (Sh2.77 net19 exp0.22 cov0%) 2.HMA-fam (2.66/31/0.48/43%) 3.VOLTGT_BH (2.65/45/0.84/everyday)
... 8.BUYHOLD (2.34/47/1.0/everyday). [MA families flat MOST days at 1d]
**4h:** 1.VIDYA-fam (3.42/18/0.22/6%) 2.EMA-fam (3.01/28/0.40/46%) 3.VOLTGT_BH (2.75/49/everyday) 8.BUYHOLD (2.46/51).
**1h:** 1.VIDYA-fam (4.51/47/0.41/54%) 2.TEMA (4.0/50/75%) ... 9.VOLTGT_BH (3.7/74/everyday) 10.BUYHOLD (3.45/89).
**2h:** 1.VIDYA (3.27/35/43%) 2.VOLTGT_BH (1.89/56/everyday) 4.BUYHOLD (1.82/59).
**30m:** 1.VOLTGT_BH (4.27/96/everyday) 2.BUYHOLD (4.09/126) 3.VIDYA (3.96/48/70%) ... 10.HMA (2.53/29/95%).
**15m:** 1.VOLTGT_BH (4.81/115/everyday) 2.BUYHOLD (4.55/154) 3.VIDYA (4.05/55/77%) ... 10.HMA (0.44/2.5/everyday).

## COVERAGE -- are we in the market every day?
- **BUYHOLD + VOLTGT_BH: YES, every day** (avg exposure 1.0 / 0.84-0.89).
- **MA families: NO at coarse, increasingly YES at fine.** Coverage (% of days >50% exposed): 1d 0-43% (FLAT
  most days -- big gaps); 4h 6-59%; 2h 43-85%; 1h 54-90%; 30m 68-95%; 15m 77-96%. So a daily MA strat is
  out of the market most days; only at 15m is it in nearly every day -- BUT 15m cost then destroys the net.
- The coverage-vs-cost tradeoff: coarse MA = sparse coverage + low cost; fine MA = dense coverage + high cost.
  A FAMILY (many configs) raises coverage vs a single config (which is flat far more), but the family net is
  still below buy-hold.

## THE HONEST "BEST" (per the North Star = wealth, with risk + coverage)
**VOL-TARGETED BUY-HOLD** is the single best emergent strategy: highest or near-highest net at every TF
(best at 30m/15m: +96%/+115%), best Sharpe at fine TFs, lower maxDD than buy-hold (-18 to -20 vs -25 to -27),
and EVERY-DAY coverage. It is the one strategy that wins on net AND risk AND coverage. The MA families are
de-risked variants -- lower net, sparser coverage, "better" only by a Sharpe that rewards sitting out.

## PITFALLS (ranked by how badly they mislead)
1. **Sharpe rewards UNDER-PARTICIPATION in a bull.** Ranking by Sharpe selects the strategy that is barely
   invested (VIDYA@4h: top Sharpe, 22% exposure, net 18% vs buy-hold 50%). For WEALTH, rank by NET, not Sharpe.
2. **Config/MA-type ranking is mostly NOISE** (eff N ~1.2; the families cluster within ~0.3-0.5 Sharpe and the
   order FLIPS between Sharpe and net). "Several ranked configs per TF" exist but the ranking does not transfer
   -- picking the VAL-best is selection risk (the family + XS findings).
3. **Coverage gaps = the participation tax.** Coarse MA strategies are FLAT most days (1d: 0-43% coverage),
   forfeiting the drift -> that is why their net < buy-hold.
4. **Cost destroys fine-cadence MA.** 15m HMA/TEMA full coverage but net 2.5-3.9% (cost-eaten) -- the dense
   coverage is paid to the exchange.
5. **Bull-specificity.** The MA families' only real value (lower maxDD) pays ONLY in a bear (untested); in the
   2020 bull they are strictly worse than buy-hold on net. Do not read these ranks as regime-general.
6. **In-sample / no UNSEEN.** Ranks are OOS-within-2020 (adjacent same-regime), not UNSEEN-robust.

json: leaderboard_*.json. RWYB: python -m strat.deep2020_leaderboard --cadences <tf>.
