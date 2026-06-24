# ENGINE_BUILDS — master index + provenance for the move-capture engines

This is the **home that indexes ALL engine versions** (V1 now; V2+ later) **and preserves the JOURNEY** so a fresh
instance never loses track of how we got here. Two things live here and nowhere else:

1. **The version index** — which engines exist, their status, scope, doc, and the commit that landed each.
2. **The provenance** — the commit trail from the start of this work to the 1st verified success, so the recipe and
   the lessons are not re-paid for.

Source of binding intent: [`../../CHARTER.md`](../../CHARTER.md). Live state: [`../../CURRENT.md`](../../CURRENT.md).
Dead veins (never re-mine): [`../MARKET_FRAMEWORK/01_DEAD_LIST.md`](../MARKET_FRAMEWORK/01_DEAD_LIST.md).

---

## 1. Index — engine versions

| Version | Name | Doc (detail home) | Status | Scope | Landing commit |
|---|---|---|---|---|---|
| **V1** | `V1_ma_drawdown_insurance` | [`V1_ma_drawdown_insurance.md`](V1_ma_drawdown_insurance.md) (supersedes the old [`../MA_DRAWDOWN_INSURANCE_FINDINGS.md`](../MA_DRAWDOWN_INSURANCE_FINDINGS.md), now a redirect) | **1st VERIFIED SUCCESS** (red-teamed GO; DEV-window incubation, not live-validated) | **1d / 4h** over **u100** (56 of 100 point-in-time names; >=20 floor met). Fine TFs (2h/1h/30m/15m) NOT in scope — 15m BROKEN. | `efb23a4b` |
| **V1.1** | `V1.1_finetf_exits` | [`V1.1_finetf_exits.md`](V1.1_finetf_exits.md) | **REFUTED as a fine-TF wealth edge** (the htf_entry_gate "fix" was a LOOK-AHEAD leak, caught + FIXED) — earned null (D81); reusable exit_harness + run_wealth eval-window KEPT | **2h/1h/30m/15m** over **u100** (16 cells; de-leaked). De-leaked: dominated by the coarse engine on net 0/16; drawdown-insurance only. | `5c960179` |
| **V1.2** | `V1.2_combinatrix` | [`V1.2_combinatrix.md`](V1.2_combinatrix.md) + [`V1.2_engine_library.md`](V1.2_engine_library.md) | **engine library (30 de-risked betas) + combinatrix = disciplined NULL** (D82): no combo beats buy-hold/best-single OOS+IRON after deflation (PBO 0.71). Reusable `combinatrix.py` (CSCV/PBO/deflated-Sharpe/RC) KEPT. | families **1/2/3** TIs, all 6 TFs, **27** point-in-time u100 names; 275 combos. Only conditioning/regime-gate survives = drawdown-insurance. | `07a88474` + (this commit) |
| **V2** | `V2_subtf_decoupled` | [`../../runs/engines/V2_subtf_decoupled/README.md`](../../runs/engines/V2_subtf_decoupled/README.md) | **VERIFIED (OOS): MA family deployable** — WMA\|4h +193.5% compound / +4.11%/wk OOS, 6/8 types pass, 100th-pct random-null. Gold standard. | 8 MA types × {1d,4h,2h,1h,30m}; u100; 7d-lookback/7d-slice decoupled + V1 regime gate. **4h is the sweet spot.** | `35414e51` / `ca597508` |
| **V3** | `V3_ti_families` | [`../../runs/engines/V3_ti_families/README.md`](../../runs/engines/V3_ti_families/README.md) | **8 non-MA TI families built + OOS-validated. 2 NEW validated static engines: Bollinger\|1d (+2.38pp), RSI\|4h (+0.81pp). ADX\|4h borderline; MACD\|1h OOS-fit REJECTED; Stochastic/Supertrend/Funding/OI NULL.** Headline DEV compounds were inflated ~9.5x (forensic audit). | 8 families (RSI/MACD/Bollinger/Stochastic/ADX/Supertrend/Funding/OI) × {1d,4h,2h,1h,30m}; u100; ensemble + V1 gate + decoupled confirmation; Funding/OI use real chimera features. | `e9d79e55` |
| _V4+ (template)_ | `V<N>_<short_name>` | `V<N>_<short_name>.md` | _PROPOSED / BUILDING / VERIFIED / REFUTED_ | _universe + the {1d,4h,2h,1h,30m,15m} cadences touched + lookback window_ | _commit SHA once landed_ |

**V1 one-line claim** (cite verbatim, never the full-window compound): at 1d/4h, an EW u100 sleeve that is long while
the MA band holds and in cash otherwise roughly **HALVES portfolio max-drawdown (~-35% vs buy-hold ~-82%)** via
**genuine same-exposure timing skill** (+6 to +15.5pp of drawdown avoided beyond a phi-matched churn-immune dumb book).
It is **capital-preservation within long-only spot — NOT alpha, NOT a money-maker.**

**V1 artifacts:** canonical engine home `runs/engines/V1_ma_drawdown_insurance/` (manifest: its `README.md`).
- VERIFIED: `engines/engine_<TYPE>_clean.json` (8 types), `verification/_skeptic{1,2,3}_*.json` (the 3-axis red-team),
  `step1/step1_ema_1d_7d.json` (capture STEP-1), `slices/` (forensic slices: `forensic_slice_bull_2021-01.csv`,
  `forensic_slice_oos_val_2024-03-25_2024-03-31.{csv,json}`, `slices_within_window.{csv,json}`),
  `outoffit/outoffit_2021_2024-05.json`, `charts/` (equity + per-trade PNGs + `slices/`).
- Open / un-verified (verify before use): `open_unverified/` — `val_forward_<TYPE>.json` (held-out forward windows),
  `decoupled_7slice_14lb_<TYPE>{,_4h}.json` (7d-slice / 14d-lookback), `_decoupled_compare_{1d,4h}.json`.
- Drivers (code): `src/strat/ma_wealth_driver.py` (wealth, authoritative), `src/strat/ma_exposure_control.py`
  (exposure-matched skill control), `src/strat/ma_movecatch_engine.py` (band selection + holds),
  `src/strat/capture_lab.py` (STEP-1 capture).

---

## 2. Build methodology — the repeatable recipe for the NEXT engine

This is the **gated process V1 followed**, distilled so V2 reuses it instead of re-deriving it. Each step earned its
place by catching a real defect (see the journey, section 3).

**(a) Canonical configs — the band grid, ensembled, never hindsight-ranked.**
Configs come from the BAND grid leaderboard at `runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json`
(+ `robust_ma_runners.json`), selected via `src/strat/{ma_2020_config_leaderboard,robust_ma_runners,working_band_rolling}.py`.
Trade the **fixed-EW ENSEMBLE** of `band.band_configs` per **(type, TF)** — **NEVER `ranked[0]`** (that is hindsight
config-picking). Selection reads only the TRAIN/\VAL net field; the OOS field is never read in development.

**(b) Capture STEP first, then WEALTH via the ONE verified driver.**
Run the **capture STEP** first (`capture_lab` — `held_cross` alignment, capture-rate), THEN compute **WEALTH** through
the single authoritative driver `src/strat/ma_wealth_driver.py`. **Never hand-roll wealth in a per-worker backtest** —
that is exactly how the two fabrications below entered. Wealth must run the universe via the verified accounting path
(`portfolio_replay` / `ma_wealth_driver`), not a u10-hardwired sleeve (a breadth trap).

**(c) The gate battery (all must pass before any "success" word):**
- **No-skill fixed-hold baseline** — a TRUE buy@listing HODL, so the engine is measured against the dumb default.
- **Exposure-matched, churn-immune control** — CFAL (constant-fraction always-long) and **REHTM** (random-entry
  hold-to-maturity, many seeds, hold-length matched to the engine's run length). Use **CFAL/REHTM, NOT the broken A9
  same-exposure SHUFFLE** (the shuffle is churn-penalized = a rigged null; see the BROKEN-NULL lesson in section 4).
- **Lag-1 causal** — signals act on the next bar; a lag-0 positive-control proves the test has power (V1's lag-0 cheat
  explodes compound to 2.6M%, confirming the shipped numbers are causal, not leak-flattered).
- **Block-bootstrap** — honest date-block resampling for the p05, not IID shuffling.
- **Select on TRAIN/VAL, never OOS/UNSEEN** — DEV wall < 2024-05-15 sealed.
- **Rank by WEALTH** (compound net), never Sharpe; track capture-rate alongside. IC is banned as a primary metric.

**(d) Aggressive adversarial red-team on >=3 axes BEFORE any success claim.**
V1's three: (1) timing-skill-not-low-exposure (the make-or-break: beat the exposure-matched dumb book at the same phi),
(2) leak-free + survivorship-honest + cost-robust (lag-0 power test, DEV wall, dead-name injection, taker cost),
(3) robust-not-curve-fit (drawdown cut in ALL stress events, not just 2022). A claim ships only on GO across all axes.

**(e) Synchronous, in-process runs.**
Run the build SYNCHRONOUSLY in-process. Detached fan-out in this campaign orphaned workers that survived a restart
(produced ghost results no one was reading) — in-process keeps accounting and provenance under one verified path.

---

## 3. Provenance / the journey — the commit trail

The arc from the start of this work to the V1 success runs along **two strands**. All SHAs below are real
(`git log --oneline`, verified this session); subjects are the exact commit subjects.

### Strand A — the OPERATING MODEL that made constructive building possible
Before V1 could be built constructively, the project needed an operating model that stops a fresh instance from
collapsing the breadth surface and quitting at a verdict. That system is the foundation V1 stands on.

| Commit | Subject | What it gave us |
|---|---|---|
| `b0e14c5f` | operating-model: 4-pillar anti-collapse system (gate + state spine + knowledge retrieval + memory shard) | The 4-pillar anti-collapse system: give-up gate + CHARTER/CURRENT state spine + knowledge inject + memory shard. |
| `f66cae53` | charter: META-AGENT operating mode is the default | META-AGENT / OVERSEER as the default operating mode (coordinate + dispatch + judge + enforce). |
| `c7e7e34c` | knowledge: point map + CURRENT at the MA config store (closes self-location gap) | Pointed the knowledge map + CURRENT at the MA config store, so a worker can FIND the canonical band grid. |
| `a4184bd3` | wall-clock: bind globally (all instances + all sub-projects) + inject into the operating model | Wall-clock grounding bound globally + injected into the operating model (no felt time). |

### Strand B — the MA ENGINE campaign (to the 1st verified success)
| Commit | Subject | What happened |
|---|---|---|
| `19e291a2` | ma-engine STEP 1: ma_cross band-ensemble capture on u100 1d/7d | STEP-1 capture: `ma_cross` band-ensemble on u100, 1d / 7d lookback (`capture_lab`). |
| `34be2be5` | ma-engine: 2h resample-from-1h into the capture loader (TF fix) | 2h TF fix — resample 2h from 1h into the capture loader. |
| _(in-flight, pre-`efb23a4b`)_ | — | **The 8 engines built**, one per MA TYPE (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA), each per (type x TF). |
| _(the wealth-driver saga)_ | — | **TWO fabrications caught by the gates en route:** (1) a **+5828% MtM-look** (a mark-to-market double-count / look artifact), and (2) a **+4854pp mislabeled-baseline** (gain measured against the wrong baseline). Both were caught BECAUSE wealth was forced through the single verified driver + the gate battery — not a per-worker hand-roll. This is why methodology step (b) exists. |
| _(the 3-axis red-team)_ | — | Aggressive adversarial red-team on the 3 axes (exposure-skill / leak+survivorship+cost / robustness) — GO on all three. |
| `efb23a4b` | SUCCESS (1st verified): MA drawdown-insurance sleeve at coarse TFs (1d/4h), u100 | **The 1st VERIFIED SUCCESS.** Doc: `V1_ma_drawdown_insurance.md`; artifacts: `runs/engines/V1_ma_drawdown_insurance/`. |
| `957b756d` | state: note the 1st verified success (drawdown-insurance sleeve) in CURRENT + clean scratch | State note in CURRENT + scratch clean. |

**This session (post-`957b756d`, un-committed — overseer commits):** forensic **within-slice** examination —
the bull-week slice (`forensic_slice_bull_2021-01.csv`) and the random **Mar-2024 OOS-for-band VAL slice**
(`forensic_slice_oos_val_2024-03-25_2024-03-31.{csv,json}`), to inspect engine behaviour candle-by-candle within a
single move rather than only in aggregate.

> **Note on the BROKEN A9 same-exposure SHUFFLE null.** The same-exposure random-K SHUFFLE was found to be a *broken*
> (churn-penalized) null and is **not** a valid control — it is tracked in the memory shard
> (`broken-same-exposure-shuffle-null`), not by a dedicated commit SHA, so no SHA is cited for it here. The decisive
> preservation control is **CFAL/REHTM**, not the shuffle.

---

## 4. Lessons that bind the next build

1. **Adversarial-verify beats confident self-diagnosis.** Two confident, internally-plausible wealth numbers (+5828%
   MtM-look, +4854pp mislabeled-baseline) were both WRONG and both shipped past self-review — only the gate battery +
   adversarial red-team caught them. Presume every number is broken until an adversary fails to break it. A "success"
   word is earned only after GO on >=3 independent attack axes.
2. **Shared verified infra > per-worker improvisation.** Both fabrications entered via per-worker hand-rolled wealth.
   Forcing wealth through the ONE authoritative driver (`ma_wealth_driver`) — and capture through the ONE capture lab —
   is what made the defects visible and reproducible. Build on the shared verified path; do not re-implement accounting
   per worker.
3. **The decisive preservation test = beat a same-exposure, churn-immune control.** A drawdown improvement is only
   skill if it survives a control at the *same exposure (phi)* that cannot churn its way to the result. That control is
   **CFAL/REHTM**, NOT the broken A9 same-exposure shuffle. "We avoided drawdown" without an exposure-matched
   churn-immune comparison is a low-exposure artifact, not timing skill.

---

_Maintenance: when a new engine version lands, add its row to the section-1 index (name -> doc -> status -> scope ->
commit), append its build/verify commits to the section-3 journey, and fold any new hard-won lesson into section 4.
Keep this file the single home for the version index + the journey._
