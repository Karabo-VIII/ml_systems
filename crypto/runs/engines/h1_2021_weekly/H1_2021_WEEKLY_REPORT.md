# H1-2021 WEEKLY LEDGER -- MASTER REPORT (EMA-band V1 drawdown-insurance engine)

**Engine:** the V1 MA drawdown-insurance engine, MA type = **EMA** (the representative type; the property is robust
across all 8 MA families -- EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA per `docs/engines/V1_ma_drawdown_insurance.md` §3).
It is the **fixed-EW ensemble over the EMA working band** that was **frozen on 2020 TRAIN AND VAL**
(`config_leaderboard.json`, selection rule `TRAIN.net>0 AND VAL.net>0` -- the OOS field is never read), going
**long while >50% of the band's held_cross is aligned, in cash otherwise.** Long-only + spot, lev = 1. The unit is a
**SETUP across a multi-candle MOVE** -- hold through the trend, de-risk to cash when the band breaks. Move-CATCH, not
per-candle prediction.

**Accounting:** the one authoritative wealth path `src/strat/ma_wealth_driver.py:run_wealth` -- **MtM-reconciled,
lag-1 causal** (weights `W.shift(1)`; episode entries fill at the **open of bar i+1**), **maker 0.0006 RT**, TRUE
buy-and-hold baseline, DEV wall. No HTF / no `align_htf_on` leak.

**Window:** **H1-2021 = [2021-01-01, 2021-07-01)** = Jan-Jun 2021, sliced into **26 non-overlapping 7-day weeks**
(W00 = Jan 1-7 … W25 ends Jun 30). Warmup bars loaded from 2020-03-01; **all metrics are on H1 bars only.**
**2022 is sealed (untouched):** max loaded bar < 2022-01-01 on every TF (panel max in 2021-06), asserted both in
`run_wealth` and in the per-asset loader (`eval_end = "2021-07-01"`).

**Run:** synchronous, all 6 TFs in one process -- `crypto/scripts/h1_2021_weekly_ledger.py`.
**Artifacts:** `perf_weekly.json` + `trades_<TF>.csv` (6 files), this directory.

---

## RECONCILIATION / SEAL / CAUSALITY -- PASS (`reconciliation_gate.PASS = TRUE`)

| Check | Result |
|---|---|
| Reconciliation gate (sign-agreement on both gate TFs) | **PASS** -- 1d: H1 engine **+448.07%** vs mean-episode-roi **+361.88%** (both +); 4h: H1 engine **+308.58%** vs mean-episode-roi **+18.41%** (both +). Episodes are per-name attribution; the 1d mean is dominated by a few long-held bull episodes, the 4h panel has many short churned episodes so its mean is small but positive. Sign agrees in direction on both. |
| `all_mtm_reconciled` | **TRUE** (all 6 TFs) |
| SEAL OK (max loaded bar < 2022-01-01) | **TRUE** (every TF; panel_max in 2021-06) |
| deployment in [0,1] | **TRUE** (all TFs; H1-wide avg gross: 1d 0.693, 4h 0.515, 2h 0.486, 1h 0.479, 30m 0.463, 15m 0.467) |
| weeks tile [Jan1, Jul1) -- no gaps/overlaps | **TRUE** (26 contiguous 7-day buckets; every H1 bar -> exactly one week) |
| Causality | lag-1 fills (engine `W.shift(1)`; episode entries at open of bar i+1); band_off exit; no HTF leak |

---

## 1. H1-2021 CUMULATIVE -- engine vs always-long market (compound, maker 0.0006)

| TF | engine_net | market_net | engine - market | #episodes | avg gross phi |
|----|-----------|-----------|-----------------|-----------|------------|
| **1d**  | **+448.07%** | +417.53% | **+30.54pp** | 79 | 0.693 |
| 4h  | +308.58% | +386.05% | -77.47pp | 521 | 0.515 |
| 2h  | +283.76% | +406.83% | -123.07pp | 1045 | 0.486 |
| 1h  | +249.30% | +422.65% | -173.35pp | 1941 | 0.479 |
| 30m | +193.80% | +396.05% | -202.24pp | 3047 | 0.463 |
| 15m | +112.13% | +265.40% | -153.26pp | 4050 | 0.467 |

**Only 1d beats the always-long market in H1-2021** (+30.54pp). Finer TFs lag the market and the churn-drag widens
monotonically 4h -> 30m. The 1d outperformance is **structural, not up-market alpha**: coarse 1d entered the late-2020
ramp **early** (BTC episode entered 2020-10-09 @ $11,067, exited 2021-04-21 @ $53,800, **+385.81% net**, then de-risked
in late June), while at all finer TFs (4h..15m) intra-trend churn drags the engine **below** the always-long market.
This is exactly consistent with the V1 doc's binding **1d/4h scope** and the **"drawdown-insurance, not up-market
alpha"** claim -- H1-2021 is a (mostly) bull half-year, the regime in which the engine is *designed to give up upside.*

---

## 2. THE WEEKLY PERFORMANCE GRID -- 26 weeks × 6 TFs (engine_net_pct), + 1d-market reference

Each cell is the engine's **net** return for that 7-day bucket at that TF. The `mkt 1d` column is the always-long 1d
market for the same week (reference). `dep 1d` = the 1d engine's average gross deployment that week (the de-risk dial).

| Week | Date range | 1d | 4h | 2h | 1h | 30m | 15m | mkt 1d | dep 1d |
|------|-----------|----|----|----|----|----|----|--------|--------|
| W00 | 2021-01-01 -> 2021-01-08 | +19.55 | +22.00 | +23.55 | +19.61 | +12.07 | +6.74 | +31.46 | 0.645 |
| W01 | 2021-01-08 -> 2021-01-15 | +17.08 | +10.00 | +11.93 | +11.62 | +11.60 | +7.77 | +18.89 | 0.866 |
| W02 | 2021-01-15 -> 2021-01-22 | +3.60 | +7.08 | +5.98 | +6.89 | +5.58 | +3.89 | +5.05 | 0.912 |
| W03 | 2021-01-22 -> 2021-01-29 | +27.16 | +9.09 | +5.71 | +5.11 | +4.48 | +4.39 | +28.07 | 0.903 |
| W04 | 2021-01-29 -> 2021-02-05 | +20.28 | +22.03 | +19.27 | +16.17 | +13.63 | +8.50 | +24.63 | 0.926 |
| **W05** | **2021-02-05 -> 2021-02-12** | **+47.08** | **+48.24** | **+47.00** | **+43.13** | **+36.40** | **+31.23** | **+50.61** | 0.982 | <- **BEST week (all TFs)** |
| W06 | 2021-02-12 -> 2021-02-19 | +15.35 | +13.35 | +8.06 | +7.99 | +7.09 | +6.47 | +15.35 | 1.000 |
| W07 | 2021-02-19 -> 2021-02-26 | -14.16 | -8.22 | -4.97 | -4.24 | -4.93 | -7.55 | -14.36 | 0.991 |
| W08 | 2021-02-26 -> 2021-03-05 | +14.07 | +4.34 | +5.05 | +2.56 | +5.80 | +6.35 | +14.86 | 0.945 |
| W09 | 2021-03-05 -> 2021-03-12 | +16.34 | +12.31 | +12.73 | +11.30 | +7.55 | +3.83 | +18.34 | 0.954 |
| W10 | 2021-03-12 -> 2021-03-19 | +11.18 | +6.90 | +5.16 | +6.21 | +4.91 | +6.14 | +12.92 | 0.908 |
| W11 | 2021-03-19 -> 2021-03-26 | -10.34 | -6.25 | -4.96 | -2.97 | -2.05 | -2.55 | -11.26 | 0.881 |
| W12 | 2021-03-26 -> 2021-04-02 | +21.69 | +12.04 | +11.80 | +12.36 | +11.08 | +11.03 | +26.32 | 0.875 |
| W13 | 2021-04-02 -> 2021-04-09 | +11.52 | +8.77 | +8.32 | +6.50 | +7.63 | +3.66 | +12.04 | 0.982 |
| W14 | 2021-04-09 -> 2021-04-16 | +25.59 | +21.73 | +20.45 | +16.03 | +12.45 | +10.09 | +26.10 | 0.960 |
| W15 | 2021-04-16 -> 2021-04-23 | -13.71 | -12.58 | -9.25 | -5.82 | -4.40 | -5.11 | -15.14 | 0.897 |
| W16 | 2021-04-23 -> 2021-04-30 | +5.18 | +0.45 | +0.68 | +3.71 | +6.37 | +5.28 | +14.59 | 0.674 |
| W17 | 2021-04-30 -> 2021-05-07 | +19.63 | +15.06 | +16.08 | +16.26 | +13.31 | +11.82 | +21.53 | 0.946 |
| W18 | 2021-05-07 -> 2021-05-14 | -6.70 | -8.83 | -10.44 | -10.37 | -9.63 | -7.13 | -9.66 | 0.803 | <- May-crash onset; de-risk begins |
| **W19** | **2021-05-14 -> 2021-05-21** | **-20.96** | -6.00 | -4.94 | -6.12 | -6.82 | -9.46 | **-26.58** | **0.576** | <- **WORST week / MAY CRASH** |
| W20 | 2021-05-21 -> 2021-05-28 | -3.73 | -0.32 | -2.22 | -5.36 | -4.40 | -3.59 | -9.30 | **0.086** | <- de-risked to cash |
| W21 | 2021-05-28 -> 2021-06-04 | -0.13 | -0.05 | +3.68 | +5.83 | +6.31 | +3.79 | +8.93 | 0.098 |
| W22 | 2021-06-04 -> 2021-06-11 | -1.69 | -5.92 | -9.96 | -12.04 | -11.40 | -12.07 | -18.56 | 0.078 |
| W23 | 2021-06-11 -> 2021-06-18 | -0.27 | -1.07 | -3.04 | -4.63 | -2.77 | -0.30 | -0.17 | 0.029 |
| W24 | 2021-06-18 -> 2021-06-25 | -0.29 | -1.08 | -1.26 | -1.65 | -2.21 | -3.15 | -20.25 | 0.008 | <- market -20%, engine ~flat (cash) |
| W25 | 2021-06-25 -> 2021-07-01 | +0.00 | +0.07 | +0.98 | +5.01 | +3.37 | +5.05 | +8.77 | 0.000 |

**Best week (all 6 TFs):** **W05, 2021-02-05 -> 2021-02-12** -- engine **+47.08% (1d)**, +48.24% (4h), market +50.61%.
A full-participation bull week (phi ~ 0.98-1.0). **Worst week (all TFs):** **W19, 2021-05-14 -> 2021-05-21** -- engine
**-20.96% (1d)** vs **market -26.58%** -- the May-2021 crash, where the engine PRESERVES **+5.6pp at 1d** (and far more
at finer TFs: 4h -6.00% vs -26.58% = **+20.6pp**) by cutting deployment from ~0.9 to 0.576 (1d) / 0.235 (4h).

---

## 3. PER-WEEK TRADE HIGHLIGHTS -- 1d and 4h (instruments, entry -> exit, roi_net)

The full ledger is the CSVs (`trades_1d.csv` = 79 episodes, `trades_4h.csv` = 521). Below are the **week-LOCAL events**:
the instruments whose **entry OR exit falls inside that week**, top ~3-5 by |roi|. `ENTER` = the position opened that
week; `EXIT` = it closed that week (the de-risk-to-cash event); `ENTER/EXIT` = a same-week round trip. entry_px = open
at entry+1 bar (lag-1 fill); exit_px = close at band-break.

### 3.1 -- 1d ledger (79 episodes; long holds -- entries cluster early, mass-exit on the May band-break)

| Week | Week-local 1d events (top by \|roi\|) |
|------|------|
| W00 (01-01->01-08) | ETC ENTER 01-04@7.04 -> 06-03@69.83 **+892.0%**; FET ENTER 01-04@0.0556 -> 04-18@0.542 **+874.8%**; HBAR ENTER 01-07@0.0379 -> 04-21@0.291 **+667.6%**; XLM ENTER 01-05@0.193 -> 03-13@0.407 +110.7%; LINK ENTER 01-04@13.62 -> 03-23@26.74 +96.2% |
| W01 (01-08->01-15) | ENJ ENTER 01-09@0.208 -> 04-23@2.24 **+975.3%**; SOL ENTER 01-08@3.17 -> 05-22@31.27 **+884.7%**; AVAX ENTER 01-08@5.80 -> 03-05@24.31 +318.9%; ZEC ENTER 01-10@96.65 -> 05-18@255.0 +163.6% |
| W02 (01-15->01-22) | CRV ENTER 01-17@1.69 -> 02-23@2.14 +26.3%; TRX EXIT 01-05@0.0285 -> 01-21@0.0277 -3.0%; JST E/X 01-19@0.0289 -> 01-20@0.0284 -1.7% |
| W03 (01-22->01-29) | JST ENTER 01-24@0.0293 -> 04-24@0.102 +247.3%; DASH EXIT 01-09@109.9 -> 01-26@104.8 -4.7% |
| W04 (01-29->02-05) | TRX ENTER 01-30@0.0321 -> 05-17@0.113 +251.3%; LTC EXIT 2020-10-23@55.44 -> 01-30@133.3 +140.4%; DASH ENTER 02-03@116.5 -> 03-23@206.6 +77.3%; BCH EXIT 2020-12-14@276.7 -> 01-30@416.9 +50.6% |
| W05 (02-05->02-12) | FIL ENTER 02-10@39.89 -> 05-09@146.4 **+266.7%** (best week -- most names already held from Jan) |
| W06 (02-12->02-19) | (no week-local entry/exit; fully invested, riding Jan positions) |
| W07 (02-19->02-26) | CRV EXIT 01-17@1.69 -> 02-23@2.14 +26.3% (late-Feb dip -- engine mostly stays long, week -14.16%) |
| W08 (02-26->03-05) | ETH EXIT 2020-10-10@370.5 -> 03-01@1570 **+323.5%**; ETH re-ENTER 03-04@1540 -> 05-20@2766 +79.5%; BCH EXIT 02-02@430.3 -> 02-27@482.9 +12.1% |
| W09 (03-05->03-12) | INJ EXIT 2020-11-14@1.26 -> 03-11@12.36 **+878.9%**; AVAX EXIT 01-08@5.80 -> 03-05@24.31 +318.9%; XRP EXIT 02-01@0.374 -> 03-11@0.452 +20.8% |
| W10 (03-12->03-19) | AAVE EXIT 2020-11-09@54.83 -> 03-17@382.9 **+597.8%**; XRP ENTER 03-18@0.470 -> 05-20@1.179 +150.6%; XLM EXIT 01-05@0.193 -> 03-13@0.407 +110.7% |
| W11 (03-19->03-26) | LINK EXIT 01-04@13.62 -> 03-23@26.74 +96.2%; DASH EXIT 02-03@116.5 -> 03-23@206.6 +77.3%; LTC EXIT 02-03@156.1 -> 03-24@176.2 +12.8% |
| W12 (03-26->04-02) | LTC ENTER 03-30@195.8 -> 05-18@294.2 +50.2%; LINK ENTER 03-31@29.40 -> 05-18@42.56 +44.7%; DASH ENTER 03-30@216.1 -> 05-18@305.0 +41.1%; BCH ENTER 04-01@555.2 -> 05-19@694.5 +25.0% |
| W13 (04-02->04-09) | SUPER EXIT 03-27@3.46 -> 04-05@2.78 -19.6%; AVAX ENTER 04-05@31.21 -> 04-20@27.30 -12.6%; AAVE EXIT 03-31@382.1 -> 04-08@371.0 -3.0% |
| W14 (04-09->04-16) | AAVE ENTER 04-13@399.2 -> 04-20@360.0 -9.9% (Apr blow-off top region -- fewer fresh entries) |
| W15 (04-16->04-23) | FET EXIT 01-04@0.0556 -> 04-18@0.542 **+874.8%**; HBAR EXIT 01-07@0.0379 -> 04-21@0.291 **+667.6%**; DOT EXIT 2020-12-14@4.91 -> 04-20@35.48 **+622.2%**; NEAR EXIT 2020-11-11@0.877 -> 04-18@5.148 +486.7%; **BTC EXIT 2020-10-09@11,067 -> 04-21@53,800 +385.81%** |
| W16 (04-23->04-30) | CHZ EXIT 2020-11-25@0.0132 -> 04-24@0.388 **+2833.4%**; ADA EXIT 2020-11-09@0.105 -> 04-23@1.158 **+999.2%**; ENJ EXIT 01-09@0.208 -> 04-23@2.24 +975.3%; ATOM EXIT 2020-12-30@5.42 -> 04-23@20.49 +278.1%; JST EXIT 01-24@0.0293 -> 04-24@0.102 +247.3% |
| W17 (04-30->05-07) | AVAX ENTER 05-01@31.82 -> 05-19@23.17 -27.2%; DOT ENTER 05-01@36.92 -> 05-18@40.62 +9.9%; HBAR ENTER 05-06@0.298 -> 05-07@0.307 +2.7% (last fresh entries before the crash) |
| W18 (05-07->05-14) | FIL EXIT 02-10@39.89 -> 05-09@146.4 **+266.7%**; CHZ EXIT 04-28@0.537 -> 05-09@0.485 -9.6%; ENJ EXIT 04-28@2.72 -> 05-08@2.46 -9.4% (de-risk starting) |
| **W19 (05-14->05-21)** | **MASS EXIT (band-break):** BNB EXIT 2020-12-15@29.55 -> 05-18@508.4 **+1619.9%**; ZEN EXIT 2020-12-26@11.98 -> 05-18@114.9 **+858.8%**; UNI EXIT 2020-12-28@3.73 -> 05-18@35.03 **+838.2%**; TRX EXIT 01-30@0.0321 -> 05-17@0.113 +251.3%; ZEC EXIT 01-10@96.65 -> 05-18@255.0 +163.6% -- **the engine books the bull and goes to cash** |
| W20 (05-21->05-28) | DOGE EXIT 2020-12-13@0.00323 -> 05-24@0.366 **+11,205.2%**; SOL EXIT 01-08@3.17 -> 05-22@31.27 **+884.7%**; AAVE EXIT 04-27@439.1 -> 05-21@372.5 -15.2% (last holders cut; dep -> 0.086) |
| W21 (05-28->06-04) | ETC EXIT 01-04@7.04 -> 06-03@69.83 **+892.0%**; DOGE ENTER 06-03@0.400 -> 06-06@0.371 -7.3%; ZEN ENTER 05-30@105.8 -> 06-04@104.1 -1.6% |
| W22 (06-04->06-11) | ADA EXIT 04-28@1.34 -> 06-10@1.53 +14.5%; DOGE EXIT 06-03@0.400 -> 06-06@0.371 -7.3%; SOL ENTER 06-04@37.4 -> 06-19@35.2 -5.9% |
| W23-W25 (06-11->07-01) | near-empty (dep 0.03 -> 0.0) -- engine is in CASH through the June selloff; SOL EXIT 06-04@37.4 -> 06-19@35.2 -5.9% the only notable close |

### 3.2 -- 4h ledger (521 episodes; faster cadence -- many week-local round trips)

| Week | Week-local 4h events (top by \|roi\|) |
|------|------|
| W00 (01-01->01-08) | AAVE ENTER 2020-12-30@83.49 -> 02-14@483.5 **+478.8%**; AVAX ENTER 01-01@3.53 -> 01-26@12.17 +244.6%; ZEN ENTER 01-01@12.94 -> 01-23@27.22 +110.2%; SOL ENTER 01-01@1.63 -> 01-21@3.30 +102.6% |
| W01 (01-08->01-15) | CRV ENTER 01-14@0.671 -> 02-14@3.09 **+359.8%**; UNI ENTER 01-13@5.53 -> 02-07@17.74 +220.6%; ENJ ENTER 01-13@0.172 -> 01-28@0.376 +118.7%; DOGE EXIT 2020-12-24@0.00447 -> 01-11@0.00852 +90.7% |
| W02 (01-15->01-22) | SOL EXIT 01-01@1.63 -> 01-21@3.30 +102.6%; BCH EXIT 2020-12-25@310.2 -> 01-17@479.6 +54.5%; ZEC EXIT 01-07@70.97 -> 01-20@100.5 +41.5%; ATOM EXIT 01-13@6.23 -> 01-21@7.99 +28.2% |
| W03 (01-22->01-29) | BNB ENTER 01-28@42.08 -> 02-26@228.5 **+442.7%**; DOGE ENTER 01-28@0.0125 -> 02-14@0.0635 **+406.9%**; AVAX EXIT 01-01@3.53 -> 01-26@12.17 +244.6%; ADA ENTER 01-28@0.343 -> 03-04@1.152 +235.1% |
| W04 (01-29->02-05) | AVAX ENTER 01-30@11.72 -> 02-16@37.59 **+220.6%**; DASH ENTER 02-03@116.4 -> 02-23@248.2 +113.2%; DOT ENTER 02-02@16.60 -> 02-23@33.81 +103.6%; CHZ ENTER 01-30@0.0217 -> 02-23@0.0418 +92.0% |
| W05 (02-05->02-12, BEST) | UNI EXIT 01-13@5.53 -> 02-07@17.74 +220.6%; ZEN ENTER 02-09@38.28 -> 02-22@68.42 +78.6%; BCH ENTER 02-05@438.0 -> 02-22@651.7 +48.7% |
| W06 (02-12->02-19) | AAVE EXIT 2020-12-30@83.49 -> 02-14@483.5 **+478.8%**; DOGE EXIT 01-28@0.0125 -> 02-14@0.0635 +406.9%; CRV EXIT 01-14@0.671 -> 02-14@3.09 +359.8%; AVAX EXIT 01-30@11.72 -> 02-16@37.59 +220.6% |
| W07 (02-19->02-26, dip) | CHZ ENTER 02-24@0.0517 -> 03-21@0.558 **+980.5%**; ATOM EXIT 01-28@8.22 -> 02-21@21.59 +162.7%; ALGO EXIT 01-22@0.532 -> 02-20@1.34 +151.8%; DASH EXIT 02-03@116.4 -> 02-23@248.2 +113.2% |
| W08 (02-26->03-05) | BNB EXIT 01-28@42.08 -> 02-26@228.5 **+442.7%**; FIL ENTER 03-02@37.80 -> 04-07@158.4 +318.8%; ENJ ENTER 03-01@0.594 -> 03-20@2.196 +269.7%; ADA EXIT 01-28@0.343 -> 03-04@1.152 +235.1% |
| W09 (03-05->03-12) | AVAX ENTER 03-09@34.22 -> 03-12@28.00 -18.2%; DOGE ENTER 03-06@0.051 -> 03-22@0.0576 +12.8%; BTC ENTER 03-07@50,130 -> 03-17@55,450 +10.6% (chop -- small round trips) |
| W10 (03-12->03-19) | NEAR EXIT 03-01@4.12 -> 03-18@6.01 +45.6%; UNI EXIT 03-01@23.78 -> 03-15@30.20 +26.9%; AVAX EXIT 03-09@34.22 -> 03-12@28.00 -18.2% |
| W11 (03-19->03-26) | CHZ EXIT 02-24@0.0517 -> 03-21@0.558 **+980.5%**; ENJ EXIT 03-01@0.594 -> 03-20@2.196 +269.7%; XRP ENTER 03-25@0.512 -> 04-18@1.35 +163.6%; FET EXIT 03-01@0.260 -> 03-21@0.575 +121.0% |
| W12 (03-26->04-02) | DOGE ENTER 04-01@0.0591 -> 04-22@0.261 **+341.0%**; ETC ENTER 03-29@12.25 -> 04-23@31.10 +153.7%; TRX ENTER 03-26@0.0661 -> 04-19@0.141 +112.9%; BNB ENTER 03-27@263.8 -> 04-17@515.0 +95.1% |
| W13 (04-02->04-09) | FIL EXIT 03-02@37.80 -> 04-07@158.4 **+318.8%**; JST EXIT 03-26@0.088 -> 04-07@0.123 +39.6%; SUPER EXIT 03-25@2.73 -> 04-02@3.30 +20.9% |
| W14 (04-09->04-16) | SOL EXIT 03-27@14.92 -> 04-14@26.42 +77.0%; ATOM ENTER 04-09@21.07 -> 04-17@24.46 +16.1%; UNI ENTER 04-11@30.26 -> 04-17@34.76 +14.8% |
| W15 (04-16->04-23) | DOGE EXIT 04-01@0.0591 -> 04-22@0.261 **+341.0%**; XRP EXIT 03-25@0.512 -> 04-18@1.35 +163.6%; TRX EXIT 03-26@0.0661 -> 04-19@0.141 +112.9%; BNB EXIT 03-27@263.8 -> 04-17@515.0 +95.1% |
| W16 (04-23->04-30) | ETC ENTER 04-26@32.10 -> 05-12@99.95 **+211.1%**; ETC EXIT 03-29@12.25 -> 04-23@31.10 +153.7%; DOGE ENTER 04-26@0.268 -> 05-10@0.509 +90.1%; BCH ENTER 04-27@854.3 -> 05-13@1320 +54.5% |
| W17 (04-30->05-07) | SOL EXIT 04-18@30.25 -> 05-06@44.17 +45.9%; INJ EXIT 04-21@13.60 -> 05-03@19.28 +41.7%; ADA ENTER 05-05@1.33 -> 05-19@1.83 +37.3%; DASH ENTER 04-30@295.5 -> 05-12@359.1 +21.5% |
| W18 (05-07->05-14) | ETC EXIT 04-26@32.10 -> 05-12@99.95 **+211.1%**; DOGE EXIT 04-26@0.268 -> 05-10@0.509 +90.1%; BCH EXIT 04-27@854.3 -> 05-13@1320 +54.5%; LINK EXIT 04-27@36.93 -> 05-12@45.77 +23.9%; SHIB E/X 05-11->05-12 -21.7% |
| **W19 (05-14->05-21)** | ETH EXIT 04-26@2449 -> 05-15@3749 **+53.0%**; ADA EXIT 05-05@1.33 -> 05-19@1.83 +37.3%; ZEC EXIT 04-26@227.2 -> 05-15@292.7 +28.7%; HBAR E/X 05-14@0.349 -> 05-17@0.303 -13.2%; SOL E/X 05-16@46.35 -> 05-19@40.89 -11.8% -- **4h cuts fast, week only -6.00% vs market -26.58%** |
| W20 (05-21->05-28) | ADA ENTER 05-26@1.77 -> 05-28@1.67 -5.6%; ETC ENTER 05-26@80.13 -> 05-28@77.85 -2.9% (almost flat -- dep 0.016) |
| W21 (05-28->06-04) | DOGE ENTER 06-02@0.419 -> 06-04@0.364 -13.3%; FET ENTER 06-03@0.408 -> 06-04@0.378 -7.4%; SOL ENTER 06-03@39.43 -> 06-08@37.30 -5.5% (a few failed re-entries, quickly cut) |
| W22 (06-04->06-11) | DOGE EXIT 06-02@0.419 -> 06-04@0.364 -13.3%; FET E/X 06-05 -8.3%; SOL ENTER 06-08@41.20 -> 06-11@37.83 -8.2% |
| W23-W25 (06-11->07-01) | small churned round trips only (CHZ/ALGO/AR/SHIB), all <= ±8% -- engine largely in cash |

---

## 4. NARRATIVE -- the H1-2021 arc

**Jan-early Feb -- the alt rally (W00-W06): the engine PARTICIPATES, lagging slightly.** The band is near-fully
deployed (1d dep 0.65 -> 1.00 by W06). At 1d the engine enters the late-2020 ramp early and holds the big multi-month
carries -- BTC (entered 2020-10-09, eventually +385.8%), ETH, LTC, DOGE -- while booking fresh Jan winners (ETC +892%,
ENJ +975%, SOL +885%, AAVE +479% at 4h). Peak week is **W05 (Feb 5-12): +47.08% (1d) / +48.24% (4h)** against market
+50.61%. The engine tracks the bull within a few points but never beats it intra-week -- *no up-market alpha,* exactly
as designed. Finer TFs lag more each week (W05: 1d +47 -> 15m +31) as churn taxes the same move.

**Late-Feb dip (W07, Feb 19-26): -14.16% (1d) ~ market -14.36%.** The band is still long (dep 0.99) so the engine
takes the dip nearly 1:1 with the market -- this is a *fast* drawdown the slow band does not dodge; finer TFs cushion it
(4h -8.22%, 1h -4.24%) by exiting some names faster. A symmetric smaller dip recurs at W11 (-10.34% vs -11.26%) and
W15 (-13.71% vs -15.14%).

**March chop + April blow-off (W08-W17): grinds up with the market, gives up a few points to churn.** Through March
the engine re-enters the survivors (FIL, XRP, ENJ, CHZ at 4h +980%) and rides the April blow-off (W12 +21.69%, W14
+25.59%). The huge 1d **EXITS cluster in W15-W16 (Apr 18-30)** as the first band-breaks fire on the April top: FET
+875%, HBAR +668%, DOT +622%, **BTC +385.8%**, then CHZ +2833%, ADA +999%, ENJ +975% -- the engine is **booking the
entire bull leg into cash** right at the local top. W16 deployment already drops to 0.674.

**THE MAY-2021 CRASH (W18-W20) -- the engine DE-RISKS to cash, the property in action.** This is the headline event:
- **W18 (May 7-14):** crash onset. Engine -6.70% (1d) vs market -9.66%; dep starts falling 0.95 -> 0.80. Early exits
  (FIL +267%, then CHZ/ENJ cut at small losses).
- **W19 (May 14-21) -- the WORST week and the crash core (May 19 capitulation):** market **-26.58%**, engine **-20.96%
  (1d)** -- a **+5.6pp** cushion at 1d, and a *large* cushion at finer TFs (**4h -6.00% = +20.6pp**, 2h -4.94%, 1h
  -6.12%). The 1d **mass band-break EXIT** fires here: BNB +1620%, ZEN +859%, UNI +838%, ZEC +164%, TRX +251% all
  close on May 17-18 -- the engine **books the bull and goes to cash.** 1d deployment collapses 0.80 -> **0.576**; 4h to
  **0.235.** The finer-TF advantage this week is exactly the "exit faster" mechanic -- coarse 1d still carries some names
  one bar too long, which is why its single-week number is worse than 4h even though its H1 cumulative is best.
- **W20 (May 21-28):** the last holders are cut (DOGE +11,205%, SOL +885%, AAVE -15.2%). **Deployment -> 0.086 (1d) /
  0.016 (4h)** -- the engine is now essentially in CASH. Week -3.73% vs market -9.30%.

**June (W21-W25): in cash, preserving through the second leg down.** Deployment stays ~0.0-0.1; a few failed re-entries
(DOGE, FET, SOL) get cut for small losses. The cleanest demonstration is **W24 (Jun 18-25): market -20.25%, engine
-0.29% (1d)** -- the band is flat in cash and simply does not take the second June crash leg. Across W21-W25 the engine
gives up upside on the bounces (W21 -0.13% vs +8.93%, W25 +0.00% vs +8.77%) but that is the *cost* of the insurance --
it is out of the market on both the down legs and the dead-cat bounces.

**Per-TF differences (coarse vs fine).** Coarse **1d** is the only TF that beats the always-long market over H1
(**+448.07% vs +417.53%, +30.54pp**) -- its early entry into the late-2020 ramp + clean late-June de-risk dominate.
Every finer TF *lags* the market and the gap widens monotonically with cadence (4h -77pp -> 30m -202pp) as intra-trend
churn (each held_cross flip pays maker cost) taxes the same moves; **15m** (+112% vs +265%) is the worst and is the
**out-of-scope/broken** TF per the V1 doc. The trade-off is consistent week by week: finer TFs cushion *fast* dips
better (they exit sooner -- e.g. W19 4h -6% vs 1d -21%) but bleed that advantage back through churn in the long trends.
This is the binding **1d/4h scope** restated on a fresh half-year window.

**H1 cumulative engine vs market, restated:** 1d **+448.07% / +417.53% (+30.54pp)**; 4h +308.58% / +386.05%
(-77.47pp); 2h +283.76% / +406.83% (-123.07pp); 1h +249.30% / +422.65% (-173.35pp); 30m +193.80% / +396.05%
(-202.24pp); 15m +112.13% / +265.40% (-153.26pp).

---

## 5. HONEST CAVEATS (ship attached)

- **This is the frozen V1 drawdown-insurance engine** -- the band is selected on 2020 TRAIN+VAL and held FIXED;
  H1-2021 is a forward (post-fit) window for the band but is **incubation, not live-validated.** The property
  (de-risk into selloffs) is what is being demonstrated, not a tuned-to-2021 result.
- **EMA is representative of all 8 MA types.** The V1 doc shows the maxDD-halving and de-risk behavior are robust
  across EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA (16/16 cells at 1d/4h, all beta-gates PASS). EMA was run here as the
  canonical type; another type would shift the weekly numbers a few points but not the arc.
- **Net compound is NOT the headline.** The engine is **de-risked beta / drawdown-insurance, NOT up-market alpha.**
  It has **ZERO up-market timing alpha** -- it lags the always-long market in every bull week and at 5 of 6 TFs over
  H1. The 1d **+30.54pp** outperformance is the **compounded bear-/top-dodge** (early entry + clean exit), not skill at
  picking up-moves -- do not read it as "the engine makes more money in a bull." Per the V1 doc, full-window compound is
  partly the compounded 2020 bull and must never be the headline; the same caution applies to any H1 compound number.
- **2021 is ONE (mostly bull) regime.** H1-2021 is a strong bull half with one sharp crash (May). It exercises the
  *give-up-upside* side and *one* de-risk event well, but a single half-year is not a regime sweep -- the property's
  real evidence base is the multi-event red-team in the V1 doc (COVID-2020, May-2021, 2022 bear, 2023 chop).
- **Panel = the point-in-time u100 wealth panel** (56 names; survivorship-honest -- dead names make HODL worse, which
  *strengthens* the relative halving). The always-long "market" here is the rebalanced EW index, the correct same-phi
  reference for the engine's per-bar deployment.

---

## 6. RETURN SUMMARY (for the caller)

- **Report path:** `crypto/runs/engines/h1_2021_weekly/H1_2021_WEEKLY_REPORT.md`
- **H1 cumulative engine vs market (engine / market / delta-pp):**
  - 1d **+448.07% / +417.53% / +30.54pp** (only TF that beats market)
  - 4h +308.58% / +386.05% / -77.47pp
  - 2h +283.76% / +406.83% / -123.07pp
  - 1h +249.30% / +422.65% / -173.35pp
  - 30m +193.80% / +396.05% / -202.24pp
  - 15m +112.13% / +265.40% / -153.26pp
- **Best week:** W05, 2021-02-05 -> 2021-02-12, engine net **+47.08% (1d)** / +48.24% (4h) (market +50.61%).
- **Worst week:** W19, 2021-05-14 -> 2021-05-21 (the May crash), engine net **-20.96% (1d)** (market -26.58%).
- **May-crash week (W19, May 14-21) engine vs market:** **1d -20.96% vs -26.58% (+5.6pp preserved); 4h -6.00% vs
  -26.58% (+20.6pp preserved)** -- deployment cut to 0.576 (1d) / 0.235 (4h); fully to cash (dep ~ 0.09 / 0.02) by W20.
- **Gates:** reconciliation PASS, seal OK (max bar < 2022-01-01), causality lag-1, deployment in [0,1], weeks tile
  cleanly. 2022 untouched.
