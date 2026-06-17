# ER-gated fixed-MA (4h, 3 DOF) -- held-out results (QUICK)

- **3 DOF**: ER hard-gate > 0.4 | fixed SMA(10)/30 | ATR-trail 3.0xATR14 + time-stop 42 bars (7d)
- **Entry**: `ER>0.4 AND close>prior_20_bar_high AND fast>slow` (all past-only; SetupHarness fills at next-bar open)
- **Cost**: taker 0.0024 round-trip (src/strat/fill_model). **Held-out = OOS+UNSEEN.** NET = after-cost per-trade return.
- **Pooled held-out per-trade NET expectancy**: **-1.4231%** (n=258, winrate=0.283, median=-2.2571%, p05=-9.2804%, p95=8.8404%)
- **Pooled UNSEEN-only per-trade NET expectancy**: **-0.6191%** (n=83, winrate=0.337)
- **Assets with positive held-out expectancy**: 1/15 (median asset held exp = -1.1598%)

| asset | HELD n | HELD exp% | HELD wr | OOS n | OOS exp% | UNSEEN n | UNSEEN exp% |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUIUSDT | 11 | +1.129 | 0.545 | 7 | +1.002 | 4 | +1.351 |
| DOGEUSDT | 18 | -0.097 | 0.444 | 13 | +0.338 | 5 | -1.228 |
| ETHUSDT | 22 | -0.205 | 0.318 | 16 | -0.119 | 6 | -0.436 |
| TRXUSDT | 19 | -0.229 | 0.421 | 12 | -0.088 | 7 | -0.471 |
| BNBUSDT | 16 | -0.277 | 0.312 | 10 | +0.733 | 6 | -1.962 |
| AVAXUSDT | 15 | -0.679 | 0.267 | 11 | -0.676 | 4 | -0.686 |
| ZECUSDT | 26 | -0.998 | 0.269 | 19 | -3.395 | 7 | +5.510 |
| BTCUSDT | 19 | -1.160 | 0.158 | 13 | -1.237 | 6 | -0.993 |
| SOLUSDT | 20 | -1.418 | 0.25 | 15 | -2.447 | 5 | +1.671 |
| TAOUSDT | 18 | -1.510 | 0.333 | 11 | -2.950 | 7 | +0.755 |
| PEPEUSDT | 15 | -1.669 | 0.4 | 11 | -0.624 | 4 | -4.545 |
| XRPUSDT | 13 | -2.957 | 0.231 | 8 | -3.133 | 5 | -2.676 |
| ADAUSDT | 14 | -3.047 | 0.214 | 11 | -3.520 | 3 | -1.312 |
| LINKUSDT | 14 | -3.497 | 0.071 | 8 | -4.709 | 6 | -1.880 |
| FETUSDT | 18 | -5.318 | 0.056 | 10 | -6.603 | 8 | -3.712 |
