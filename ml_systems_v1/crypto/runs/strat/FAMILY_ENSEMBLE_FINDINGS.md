# FAMILY-ENSEMBLE FINDINGS -- the capstone of the 6h translation-solution build

**The end-to-end honest story of "can the 2020 long-only edge translate to 2021+ and pay full-cycle?"**
Provenance: 6h /orc build, three phases. This doc closes the arc with the decisive cheapest falsifier.
All numbers carry a claim-tag. Real capital; long-only + spot, short OFF, fixed-EW, PIT survivorship-clean,
maker cost. UNSEEN (2025-12-31 -> 2026-06-01) was read EXACTLY ONCE at the frozen LIGHT level.

Repro:
- Phase 1: `python -m strat.family_ensemble_book --years 2020,2021,2022,2023,2024,2025` (commit 82c4e22)
- Phase 2: `python -m strat.family_ensemble_unseen` (commit 6a4c1ff)
- Phase 3 (this falsifier): `python -m strat.family_free_control --years 2020,2021,2022,2023,2024,2025`
  -> `runs/strat/family_free_control_20260616_013650.json` + `runs/strat/family_free_vs_book.png`

---

## 0. The question (verbatim spirit)

A long-only book of 2020-selected signal families (trend / breakout / momentum / MA band-ensembles), with
volume + mean-reversion dropped, under a light vol-target de-risk overlay -- does it TRANSLATE forward
(2021+), and does it PAY across a full bull->bear cycle? And the killer control: **is any of the
family-ensemble apparatus actually necessary, or is the whole result just "buy-hold with a vol-brake"?**

---

## 1. The arc, phase by phase (all VERIFIED via the canonical scorecard / PIT engine)

### Phase 1 -- the family-ensemble book preserves crashes, does NOT capture bulls [VERIFIED-FULL-CYCLE]

Per-year, book at LIGHT de-risk vs EW buy-hold (PIT), 2020-2025:

| year | regime | EW buy-hold net / maxDD | book net / maxDD | book time-in |
|------|--------|-------------------------|------------------|--------------|
| 2020 | bull   | +26.7% / -19.4%         | +17.8% / -4.4%   | 0.37 |
| 2021 | MEGA-bull | +208.4% / -49.5%     | +23.1% / -11.5%  | 0.31 |
| 2022 | BEAR   | -71.9% / -73.4%         | -14.0% / -19.8%  | 0.28 |
| 2023 | bull   | +116.8% / -33.9%        | +20.2% / -8.2%   | 0.38 |
| 2024 | bull   | +65.8% / -45.9%         | +28.3% / -7.7%   | 0.33 |
| 2025 | bear   | -55.5% / -61.1%         | -18.0% / -22.5%  | 0.29 |

- Full-cycle 2020-2025 compound [VERIFIED]: **book +57.7% (maxDD -25.7%, Calmar 2.25) vs EW buy-hold
  +75.5% (maxDD -78.9%, Calmar 0.96)**. (The originally-cited 3-year 2020-2022 cut was book +24.7% vs
  BH +9.7%; over the full 6 years buy-hold's later bull years pull it ahead on raw compound.)
- The signature: the book massively UNDER-PARTICIPATES in bulls (captures ~11% of the 2021 mega-bull) but
  loses far less in bears. Calmar 2.25 vs 0.96 -- it wins on RISK-ADJUSTED terms, not on raw wealth.
- **Phase-1 read: REAL_WITH_CAVEAT -- drawdown-preserving INSURANCE, not bull-beating alpha.** The book
  is in cash ~70% of the time (time-in ~0.3); the participation cost is the price of the preservation.

### Phase 2 -- the preserve signature HOLDS out-of-sample, but ship = FALSE [VERIFIED-UNSEEN-ONCE]

- Sealed UNSEEN (2025-12-31 -> 2026-06-01, a CHOP/SIDEWAYS regime, BH -8.1% / -40.2%): **book net -0.9%,
  maxDD -2.3% -- a 37.9pp drawdown margin over buy-hold.** The lose-less signature replicated OOS. [VERIFIED]
- BUT the canonical scorecard fails the ship gate: **held-out (OOS+UNSEEN) block-bootstrap p05 = -38.8%;
  OOS-2025 compound -19.4%.** A drawdown-preserving book is NOT a positive-held-out-wealth book. [VERIFIED]
- **The critical catch:** the DROPPED families (volume + mean-reversion) preserve the 2022 bear EQUALLY or
  BETTER (dropped maxDD -4.8% vs translating -19.8%) and the Sharpe-NULL family-pick preserves too (-10.7%).
  -> **bear-preservation is NOT a translating-family edge; it is FAMILY-AGNOSTIC.** Verdict: AMBIGUOUS /
  "the edge is a family-agnostic vol-brake / go-to-cash property." [VERIFIED]

### Phase 3 -- THE DECISIVE FALSIFIER: is it just "buy-hold + a vol-brake"? [VERIFIED-FALSIFIER]

Phase 2 said the families don't matter; the natural killer is: **strip the signal ENTIRELY.** Take plain
EW-beta (equal-weight buy-hold of the PIT-active assets -- NO families, NO MA/TI selection, NO
frozen-2020 selection) and apply the SAME LIGHT vol-target overlay the book uses. If that one-liner
reproduces the book's preservation AND full-cycle, the entire apparatus is dead.

Three streams, LIGHT de-risk, 2020-2025 [VERIFIED, K=3 independent derivations, 3/3 agree]:

| stream | full-cycle net | full-cycle maxDD | Calmar | 2022 bear maxDD | UNSEEN net / maxDD |
|--------|----------------|------------------|--------|-----------------|--------------------|
| family-ensemble BOOK   | +57.7% | **-25.7%** | **2.25** | **-19.8%** | -0.9% / **-2.3%** |
| FAMILY-FREE vol-gate   | +57.7% | -72.9% | 0.79 | -66.2% | -3.3% / -35.5% |
| RAW EW-beta (no brake) | +75.5% | -78.9% | 0.96 | -73.4% | -8.1% / -40.2% |

**The decisive numbers (the falsifier did NOT fire as "apparatus dead"):**
- The family-free LIGHT vol-gate matches the book's per-year maxDD in **0 of 6 years**.
- 2022 bear: the book preserves **53.6pp** vs raw beta; the family-free vol-gate preserves only **7.2pp**
  (its maxDD -66.2% sits within 7.2pp of un-braked beta's -73.4%). The vol-brake captures ~13% of the
  drawdown protection a signal delivers.
- UNSEEN: the book preserves (-2.3%); the family-free vol-gate does NOT (-35.5%, only 4.7pp better than
  raw beta's -40.2%).
- The full-cycle COMPOUND coincidentally matches to **0.0pp** (book +57.7% == family-free +57.7%) -- but
  this is path-incidental (a product-of-years coincidence), with utterly different drawdowns (Calmar 2.25
  vs 0.79). Matching the endpoint while losing 73% along the way is not "the same book."

**Mechanism (probed, VERIFIED):** the LIGHT vol-target keeps **~83-88% average exposure** every year --
including ~0.88 exposure THROUGH the 2022 bear. It shaves volatility off the hottest names; it does NOT go
to cash in a downtrend (vol-target is symmetric to volatility, not to drawdown). The book preserves because
the trend/breakout/momentum/MA SIGNALS turn OFF in a downtrend -> ~30% time-in -> in cash 70% of the time.
**Preservation = a long-only timing-signal going to CASH, which is family-agnostic among signals (Phase 2),
but is NOT reproducible by a continuous vol-brake on always-on beta (Phase 3).**

Verdict: **APPARATUS_ADDS_VALUE on the RISK axis** -- but the value-add is "being a long-only timing signal
that de-risks to cash," NOT "the family selection" and NOT "the vol-brake." The de-risk OVERLAY is the weak
component; the SIGNAL (any de-risking signal) is the load-bearing one.

---

## 2. The decisive verdict -- is there a deployable wealth edge?

**NO deployable WEALTH (held-out positive-compound) edge.** Every construction in this arc fails the
scorecard ship gate (held-out p05 < 0):
- the family-ensemble book: held-out p05 -38.8% [VERIFIED, Phase 2]
- the family-free vol-gate: held-out p05 -78.1%, full p05 -86.1% [VERIFIED, Phase 3 scorecard]

The ONLY translating, OOS-replicated result is a **family-agnostic drawdown-INSURANCE property** delivered
by a long-only timing signal de-risking to cash. That is a **RISK tool, not alpha**:
- it does NOT beat buy-hold on raw wealth full-cycle (+57.7% vs +75.5%);
- it wins ONLY on risk-adjusted terms (Calmar 2.25 vs 0.96, maxDD -25.7% vs -78.9%);
- its held-out compound is not provably positive (p05 < 0) -- so even the risk tool is not ship-grade as a
  standalone wealth sleeve.

And the cheapest possible version of that risk tool -- "buy-hold + a LIGHT vol-brake" -- does NOT work: the
vol-brake captures ~13% of the protection and costs ~18pp of bull participation for almost no crash relief.

---

## 3. The honest deployable -- what to actually deploy (if anything)

**As a standalone wealth sleeve: nothing here ships** (held-out p05 < 0 across the board).

**As a small DEFENSIVE/insurance allocation inside a larger book (the only defensible use):**
- the **signal-gated de-risk-to-cash book** (the family-ensemble book, OR equivalently any single
  translating family -- the family choice is immaterial per Phase 2), NOT the vol-braked-beta one-liner.
- Expected behaviour by regime [VERIFIED on 2020-2025 + UNSEEN]:
  - **bull:** under-participates hard (captures ~11-25% of buy-hold) -- a drag in mega-bulls.
  - **bear:** loses ~3-4x less than buy-hold (2022: -14% vs -72%; UNSEEN chop: -1% vs -8%).
  - **chop:** roughly flat, small negative -- it sits in cash and waits.
- Sizing it as INSURANCE: it is a hedge you pay a carry for (the bull drag). It only earns its keep if you
  believe a deep bear is coming and you cannot/will not simply hold less beta. A plain "hold 30% of your
  beta in cash" achieves a similar exposure profile more cheaply -- the signal's only edge over static
  cash is that it RE-ENTERS on trend resumption (the 2023/2024 recoveries), which static cash does not.

**What it is NOT:** it is NOT alpha; it is NOT bull-beating; it is NOT a positive-held-out-wealth sleeve;
and it is NOT reproducible by a vol-brake (Phase 3 killed that shortcut). The vol-target OVERLAY specifically
is the WEAK link -- it should be dropped or replaced by the signal's own cash-going, which does the work.

---

## 4. The single most important open question (within long-only)

**Is there ANY long-only construction that participates in bulls AND preserves bears -- or is that the
fundamental long-only tradeoff?**

Every result in this arc is a point on ONE frontier: more participation <-> less preservation. Raw beta is
the max-participation/max-crash corner; the signal-gated book is the max-preservation/min-participation
corner; the vol-brake is a poor interior point. Nothing escaped the frontier -- the book preserved by being
in CASH, which is definitionally non-participation. The open question is whether the frontier can be BENT:
- a long-only construction that stays IN during bear RALLIES (bears have +30-50% counter-trend bounces) and
  only de-risks the terminal legs -- capturing bull-like upside inside a bear -- would bend it; nothing here
  tried that (the signals are all trend-following, so they sit out the whole bear including its rallies).
- OR the honest answer is: **long-only cannot have both** (you cannot preserve a -73% bear without going to
  cash, and cash cannot participate), and the ONLY way to participate-AND-preserve is to add a SHORT leg
  (out of scope here, short OFF) or an orthogonal carry sleeve (e.g. the funding-dispersion neutral carry,
  a separate finding) whose return does not depend on beta direction.

That is the next falsifier: **build the bear-rally-participating long-only book and test whether it bends
the participation/preservation frontier, or whether it just re-discovers that long-only is one-sided.** It
is the cheapest remaining shot at a long-only edge that is BOTH; if it fails, the long-only participate-AND-
preserve dream is closed and the only doors left are SHORT (off) or beta-orthogonal CARRY.

---

## 5. Claim-tags / caveats (binding)

- All per-year, full-cycle, UNSEEN, and 2022-bear numbers above: **[VERIFIED]** -- re-derived this session
  via `family_free_control.py` + K=3 independent derivations (3/3 agree), PIT engine, fixed-EW, maker cost.
- Held-out p05 figures: **[VERIFIED]** via the canonical `scorecard.score_book` block-bootstrap.
- **Survivorship residual (UNFIXABLE from our data):** coins that traded 2021-2022 but delisted before 2026
  (LUNA, FTT, ...) were never collected into chimera -> cannot be included. This biases the buy-hold
  benchmark UPWARD (the bear was worse than shown for a real 2021 holder) and makes the book's relative
  preservation a CONSERVATIVE lower bound. Flagged, not silently ignored.
- **Cost realism:** maker round-trip on signal flips; the family-free control charges NO flip cost on its
  continuous vol-re-size (shown best-case, the right bias for a falsifier). MakerCostModel p_fill 0.25-0.50
  means live equity ~50-75% of these fixed-backtest figures -- a haircut, applied to ALL three streams alike,
  does not change the relative verdict.
- **The match-tolerances are pre-registered** (DD <=5pp, compound <=10pp = the book verdict's own material
  margin); the verdict is two-sided (it would have said APPARATUS_DEAD had the vol-brake matched).
