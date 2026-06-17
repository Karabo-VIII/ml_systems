# Econometric Signature — the math/econ lens of the decomposition harness

**Purpose (2026-06-09).** Put COMPUTED NUMBERS to the time-series theory that
[`CRYPTO_MARKET_UNDERSTANDING.md`](CRYPTO_MARKET_UNDERSTANDING.md) §II only *documented*. This is the **agnostic,
whole-series** lens — "what *kind of stochastic process* is this asset's return stream" — the complement to the
per-window chimera viewer ([`CHIMERA_DECOMPOSER.md`](CHIMERA_DECOMPOSER.md)). Descriptive characterization, NOT
signal-mining.

## The two-lens pivot
- **Agnostic / "what process is this?"** → `econometric_signature` (this doc): whole-series GARCH / tails / Hurst /
  stationarity / jumps. Cadence-and-asset scoped; n≥500.
- **Exactness / narrative at a period** → `decompose` (the feature table) + `narrate` (the prose story): what every
  chimera feature did over a specific `(asset, window, cadence)` slice.

Read order for any slice: **narrate (story) → econometric signature (the process backdrop) → interpretation on top.**

## Tool: `src/mining/econometric_signature.py`
```
python -m mining.econometric_signature --asset BTC --cadence 4h            # whole-series (default)
python -m mining.econometric_signature --asset ETH --cadence 4h --json     # JSON to runs/mining/
python -m mining.econometric_signature --asset SOL --cadence 4h --start 2025-01-01 --end 2025-06-01   # sub-window
```
A sub-window with **n < 500** prints a loud warning and tags GARCH/tail rows `[SMALL-N]` (these are full-sample
properties; a 7-day window cannot estimate them). Output: grouped text table + `runs/mining/econ_signature_<sym>_<cad>.json`.

**Sections (canonical estimators):** (1) distribution & tails — moments, annualized vol, **Hill tail-index** L/R,
cubic-law check, Jarque-Bera; (2) dependence & memory — Ljung-Box on ret / |ret| / ret², AC1, **Hurst (R/S + DFA)** on
ret and |ret|; (3) stationarity — ADF + KPSS; (4) volatility process — **GARCH(1,1)-t** persistence / half-life / ν +
**GJR** leverage γ; (5) jumps — **Barndorff-Nielsen-Shephard** RV-vs-BV fraction + threshold count. Each row carries a
**3-way reconciliation: our estimate | §II literature | our chimera proxy** (does `norm_vol_cluster` track fitted GARCH
persistence? does `hurst_regime` track real Hurst?) with an AGREE / DISAGREE flag.

**Estimators sanity-verified** (overseer-independent, hand-built known-property series): iid-Normal → Hurst≈0.5;
Student-t(3) → Hill α≈2.7; a manually-simulated GARCH(1,1) with true persistence 0.98 → recovered **0.9806**; jumps
isolated (smooth 0.0 vs spiky 0.60). Re-run the tracked gate: `python -m mining.econometric_signature --selftest`.

## Measured signature — u10 majors (4h, whole series; 2026-06-09)
| Metric | BTC | ETH | SOL | DOGE | §II reconciliation |
|---|---|---|---|---|---|
| EXCESS kurtosis | 19.2 | 12.6 | 8.8 | **108** | 6–26 fat tails → AGREE (DOGE = extreme meme tail) |
| Hill α (L / R) | 2.67 / 2.79 | 2.77 / 3.21 | 2.94 / 2.98 | 2.45 / **1.96** | cubic band 2–3.5 → AGREE (DOGE right tail <2 = near-infinite variance) |
| Ljung-Box ret p | 0 | 0 | 0 | 0 | efficient → AGREE-IN-SPIRIT (rejects only on negligible bid-ask AC1) |
| Ljung-Box \|ret\| p | 0 | 0 | 0 | 0 | vol clustering → AGREE |
| AC1(ret) | −0.023 | −0.023 | −0.032 | +0.006 | micro-negative (bid-ask bounce) → AGREE |
| Hurst(ret) DFA | 0.52 | 0.51 | 0.54 | 0.55 | ~0.5 random walk → AGREE |
| Hurst(\|ret\|) DFA | 0.84 | 0.84 | 0.80 | 0.82 | >0.5 vol long-memory → AGREE |
| ADF p / KPSS p | ~0 / 0.10 | ~0 / 0.07 | ~0 / 0.10 | ~0 / 0.08 | returns stationary → AGREE |
| GARCH persistence | 1.0 | 1.0 | 1.0 | 1.0 | ~1.0 near-integrated → AGREE (half-life → ∞) |
| GARCH ν (t-dof) | 3.04 | 3.33 | 4.05 | 3.32 | Student-t ν~3–4 → AGREE (≈ Hill α, as a t implies) |
| GJR γ (leverage) | +0.017 | +0.009 | +0.010 | **−0.049** | inverted/absent in BTC `[UNCERTAIN]` → MIXED (majors mild-equity, DOGE inverted) |
| BNS jump fraction | 0.165 | 0.123 | 0.069 | 0.075 | jumps present → chimera `rv_jump_frac` tracks it (AGREE) |

**What it establishes:** all four are fat-tailed (cubic α≈2–3, ν≈3–4), volatility-clustered with near-integrated
GARCH (persistence ≈ 1.0, shocks never decay) and strong vol long-memory (Hurst|ret| 0.80–0.84), **direction
≈ random-walk** (Hurst-ret ≈ 0.5, only bid-ask-bounce AC1), and **stationary in returns**. The archetype split from
[`CRYPTO_MARKET_UNDERSTANDING.md`](CRYPTO_MARKET_UNDERSTANDING.md) §IV is now *quantified*: BTC most equity-like
(kurtosis 19, mild +leverage), **DOGE the meme extreme** (kurtosis 108, skew +3.4, right-tail α 1.96, inverted leverage).

**Reconciliation finding (the new knowledge):** our engineered chimera proxies are **z-scored** (`norm_*`), so they
read as *sign/co-movement* indicators near 0, not 1:1 magnitudes — they AGREE directionally with the canonical
estimators (vol-cluster↔persistence, jump_frac↔BNS, kurtosis↔Hill) but are NOT calibrated to them. That gap (proxy
vs canonical estimator) is itself the actionable result: the chimera feature space *captures the shape* of these
stylized facts but not their *magnitude*.

## Canonical audit & basis (overseer independent rerun, 2026-06-09)

Re-run independently — `--selftest` 6/6 PASS, and the full BTC + DOGE output re-read directly (not the build
agent's table). The numbers hold up, and three INTERNAL-CONSISTENCY cross-checks — which a fabricated or buggy
result would fail — pass:
- **GARCH ν ≈ Hill α** (BTC ν 3.04 / α 2.73; DOGE ν 3.32 / α 2.20): the fitted Student-t innovation dof matches the
  order-statistic tail index, as a t-distribution requires (the unconditional tail is *slightly* heavier than ν
  because GARCH clustering thickens it — the correct direction, not a contradiction).
- **Hurst(|ret|) ≫ Hurst(ret)** (BTC 0.84 vs 0.52; DOGE 0.82 vs 0.55), confirmed by BOTH R/S and DFA: long memory
  in volatility, none in direction — the "vol predictable, direction not" thesis, measured two independent ways.
- **skew sign ↔ leverage sign**: BTC −0.76 skew + GJR +0.017 (equity sign = crashes-bigger); DOGE +3.36 skew +
  GJR −0.049 (inverted = pumps-bigger). Two unrelated estimators tell the SAME per-asset story → strong evidence
  the numbers are real, not artifacts.

**Honest caveats (audit, not bugs):**
- GARCH persistence prints **exactly 1.0** (BTC 0.0512+0.9488; DOGE 0.171+0.829 = 1.000) — this is the **IGARCH
  boundary** (non-stationary vol on-sample, half-life > data). Read it as "persistence ≳ 0.99, near-integrated",
  NOT a precise 1.0. Risk meaning below.
- The chimera `rv_jump_frac` proxy **matches** the canonical BNS fraction for DOGE (0.0748 vs 0.0751) but reads
  **~half** for BTC (0.078 vs 0.165) — the engineered feature is not calibrated to the BNS definition; trust the
  BNS number for jump magnitude.
- The `norm_*` chimera proxy means sit near 0 (z-scored) — they confirm DIRECTION/co-movement, not magnitude. The
  AGREE flags compare OUR canonical estimator to the §II literature (correct), with the proxy shown for context.

### The canonical basis — what each fact MEANS for real money (LO + spot + lev=1)
| Econometric fact (measured) | Canonical implication |
|---|---|
| Fat tails, cubic α≈2–3 (DOGE right-tail **1.96 < 2**) | Gaussian VaR / Sharpe UNDERSTATE ruin risk; size on EVT / fractional-Kelly, not σ. α<2 ⇒ variance may not exist → the memecoin **1/16-Kelly cap is a first-principles necessity**, not caution. |
| GARCH persistence ≈ 1 (near-integrated) | No vol mean-reversion to lean on; **a vol spike is the new regime** until proven otherwise. Size assuming vol clusters and persists — do not bet on it decaying. |
| Hurst(ret) ≈ 0.5; only bid-ask-bounce AC1 | **No directional edge from price/return history** — the project's core finding, now measured per asset. A directional claim must come from a NON-price feature (funding / flow / on-chain) and clear `firewall` + `positive_control`. |
| Hurst(\|ret\|) 0.80–0.84 (vol long memory) | Volatility IS forecastable — but at LO+spot+lev=1 it is only harvestable via options/convexity (DEFERRED). **Observable, not bankable here.** |
| Returns stationary (ADF reject, KPSS cannot) | The series is well-behaved/estimable; it is the *predictability* that is absent, not the stationarity. |
| BTC neg-skew/equity-leverage vs DOGE pos-skew/inverted-leverage/α<2 | The §IV archetype split is now *quantified*: BTC ≈ a fat-tailed equity; DOGE ≈ a positive-skew lottery. Per-archetype sizing (Tier-1 std-Kelly → memecoin 1/16-Kelly) is justified by the **math**, not convention. |

Descriptive characterization only — **no edge is claimed.** It states what the *risk* is, which is the precondition
for any later strategy work. Reproduce: `python -m mining.econometric_signature --selftest` + `--asset <SYM> --cadence 4h`.

## Where this sits in the framework
Tracked in the canonical store ([`workspaces/crypto/_market/manifest.yaml`](../workspaces/crypto/_market/manifest.yaml)):
the **tool** under `01_mining` (the decompose layer) and **this doc + the measured numbers** under `00_research`
(the research layer) — so the math/econ lens, like the chimera lens, now lives in BOTH layers. No emoji (cp1252).
