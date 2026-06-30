# Deployable candidates — single MA engine + regime gate (2026-06-30, /orc autonomous)

Built from the quant arc: all 8 MA types are statistically **one engine** (cross-type weekly corr 0.97, n_eff≈1), so the
deployable is **one MA type** (SMA — simplest, cheapest, no adaptive params) + a regime gate. Full band ≈ minimal 6-cross
(n_eff≈2.6). The only layer carrying real, persistent, decoupled signal is the **regime gate**; the type, periods, and
config-speed are the same trend factor (config×regime specialization = noise, 12/24 sign-persistence = chance).

**Engine (shared by both candidates):** SMA full band, per-asset MA-cross signal, inv-vol weighted, cap 33%/name,
14d-lookback / 7d-slice rolling, next-open fills, prod cost 0.0028 RT, u100, 1d. Long-only spot, lev=1.

## The two candidates (graded OOS-2024 bull / OOS-2025 bear / OOS-ALL, vs BTC buy-hold)

| Candidate | Gate | OOS-ALL comp | maxDD | Calmar | OOS-2025 (bear) | Role |
|---|---|---|---|---|---|---|
| **A — BIN_breadth** | breadth>gate (binary) | **+26%** | −34% | **0.78** | −19% | **wealth-max** |
| **B — BIN_x_ER** | breadth AND trend-efficiency (ER>0.35) | +11% | **−21%** | 0.55 | **≈ flat** | **capital-preservation** |
| *(BTC buy-hold)* | — | +11% | −50% | 0.22 | −22% | benchmark |

Both **pass the deployable screen** (positive full-cycle, maxDD<35%, beat BTC-BH on Calmar). The choice is a risk-profile dial:

- **Candidate A (current breadth engine)** — more return (+26% over the 2-yr cycle), more drawdown (−34%). Best for a wealth
  objective. It's the engine the project already has; this campaign confirms it's the best point on the gate frontier for return.
- **Candidate B (breadth + trend-efficiency)** — the genuinely *new* finding. Adding "deploy only when BTC is in a clean trend
  (efficiency-ratio > 0.35)" on top of the breadth gate **neutralizes the bear** (OOS-2025 ≈ flat vs −19%) and **halves the
  drawdown** (−21% vs −34%), at the cost of bull participation (+13% vs +56% in OOS-2024). Most **consistent** candidate by
  sub-period (positive 3 of 4 OOS half-years, smallest losses). Best for capital preservation / low-drawdown.
  - ⚠️ **Threshold sensitivity (red-team correction):** only the **bear-window sign** is stable across ER thr 0.25–0.45
    (OOS-2025 ∈ [−8, +5]%) — and that stability is mostly "the gate is in cash ~90% of the time." The **OOS-ALL return
    magnitude is threshold-sensitive** (+0.3% to +28.8%, with a near-zero *hole* at thr=0.40 right next to the 0.35 operating
    point). The robust, real deliverable is the **drawdown halving**, not a specific return number.

## Candidate C (conditional gate) — REFUTED, do not deploy

Tested a 3-state "best of both" gate (full deploy when breadth>0.60, require ER only when marginal). It is a **bull-overfit
trap**: +70% OOS-2024 (beats A) but **−45% / −51% DD in the bear** (deploying full on high breadth catches bear *rallies*),
OOS-ALL −7%, positive in only 1 of 4 sub-periods. Confirms you cannot get A's return AND B's drawdown — it's a frontier.

## Honest profile (binding — no false victory; red-team-verified)

Adversarially audited (expert-auditor): **look-ahead CLEAN** (ER + breadth + SMA path all causal, perturbation-tested; band
selection OOS-excluded). But the verdict is **de-risked beta / capital preservation, NOT alpha** — and that distinction is binding:
- **Alpha is refuted:** NW(L3) HAC p > 0.05 (B = 0.345), block-bootstrap p05 < 0 (B = −38%), deployed-week win-rate 0.42
  (fewer than half of deployed weeks positive), on a small deployed sample (~12 weeks). No statistically robust positive return.
- **Capital preservation is real and causal:** sit in cash ~88% of the time, deploy only on broad + clean-trend weeks → the
  drawdown halving (B: −21% vs BTC −50%; OOS-ALL Calmar 0.55 vs BTC 0.22) is the genuine, mechanical deliverable.
- ⚠️ **Survivorship:** the candidate book uses today's u100 with no point-in-time listing mask → **headline returns
  (+11% / +26%) are overstated**; the BTC-BH benchmark is clean. **Size off the drawdown/preservation property, never the
  backtest return.** Returns are lumpy (carried by a few >7% trend weeks).

## Live state (2026-05-28): **both CASH** — breadth 0.38 (gate closed). Do not deploy this week.

## Deploy spec

- **Sizing:** treat as de-risked **beta**, not alpha. Cap real-capital exposure conservatively (e.g. ≤ a fixed % of the
  crypto sleeve per deployed week); do not size off the backtest compound (lumpy + survivorship-overstated).
- **Cadence:** emit the weekly book each Monday from the frozen spec; deploy inv-vol cap33 across the crossed names when the
  gate is open, else cash. Candidate B additionally requires market-ER > 0.35.
- **Pre-registered live-forward gate** (to upgrade from "de-risked beta" to "ship more capital"): ≥ 20 deployed weeks; over
  them, compound beats cash AND beats BTC-BH on Calmar; maxDD within −35% (A) / −25% (B). No re-tuning of any parameter
  during tracking (re-tuning voids it).
- **Kill-switch:** maxDD breaches −40% (A) / −30% (B); or deployed-week median ≤ 0 over ≥ 20 deployed weeks; or gate degenerate
  (<5% deploy over 52 weeks). On kill: return to cash, fresh pre-registration to resume.

## Recommendation

Deploy **Candidate B (breadth + trend-efficiency)** as the lead — it does what the whole arc says the MA family is actually
*good at* (mechanical de-risking) better than the current engine: ≈flat bear, half the drawdown, robustly. Keep **Candidate A**
as the higher-return / higher-DD alternative for a more aggressive risk budget. Artifacts: `strat_deployable_candidates.{py,json}`,
`strat_candidates_verify.{py,json}`.
