# FAMILY-ENSEMBLE BOOK -- PHASE 2: SEALED UNSEEN TEST-ONCE + ROBUSTNESS BATTERY

**Verdict: AMBIGUOUS** -- the drawdown-preserving signature is GENUINE out-of-sample, but the book is
NOT a deployable wealth strategy on its own terms (held-out compound p05 < 0), and the bear-preservation
is NOT a family-selection edge (it is the de-risk/go-to-cash overlay, which the *dropped* families do
even better). The one real translating-family differentiator is full-cycle COMPOUND (more bull
participation while still preserving), not crash insurance.

- Harness: `src/strat/family_ensemble_unseen.py` (imports the PHASE-1 book from `family_ensemble_book.py`;
  no re-implementation -- bit-exact reproduction confirmed, 0.0 daily diff on the 2022 book).
- Run (TEST-ONCE on UNSEEN): `PYTHONPATH=src python -m strat.family_ensemble_unseen` (from repo root).
- Artifacts: `runs/strat/family_ensemble_unseen_20260616_012216.json`,
  `runs/strat/family_ensemble_unseen_equity.png`, `runs/strat/family_vs_null_robustness.png`.
- Claim tag: **[VERIFIED-UNSEEN-ONCE]** (the UNSEEN window 2025-12-31 -> 2026-06-01 was computed exactly
  once; no tuning/iterating/peeking; the LIGHT de-risk level is the FROZEN PHASE-1 pick, not re-tuned).

---

## PRE-REGISTRATION (stated BEFORE the UNSEEN run, persisted verbatim in the JSON)

- **H0**: the book does NOT show the preserve signature OOS (UNSEEN) AND/OR fails scorecard p05 AND/OR the
  dropped families preserve EQUALLY (no family edge).
- **H1**: the preserve signature holds on UNSEEN (book maxDD < BH maxDD) AND held-out p05 > 0 AND the
  translating families add something over the dropped/random families.
- **Asymmetric loss**: false-ship a non-preserving book (real capital into a crash it does NOT cushion)
  >> false-skip. UNSEEN is test-once.

---

## 1. THE SEALED UNSEEN TEST-ONCE (2025-12-31 -> 2026-06-01, LIGHT de-risk) [VERIFIED-UNSEEN-ONCE]

| metric | EW buy-hold (PIT) | family-ensemble book (light) |
|---|---|---|
| net | **-8.1%** | **-0.9%** |
| maxDD | **-40.2%** | **-2.3%** |
| Sharpe | -- | -0.83 |
| n_days | 149 | 149 |

**UNSEEN regime**: CHOP/SIDEWAYS with a deep mid-window crash. EW buy-hold was +18% by mid-Jan, crashed
to -28% by early Feb (maxDD -40.2%), recovered to ~+4% in May, and ended -8.1%. A whipsaw, not a clean
trend (`family_ensemble_unseen_equity.png`).

**THE TEST (preserve signature OOS): PRESENT.** Book maxDD -2.3% vs BH maxDD -40.2% -> **margin 37.9pp**.
The book behaved exactly like the drawdown-preserving book it claims to be: it cut exposure to near-cash
through the Feb crash and lost only -0.9% over the whole window vs BH's -8.1%.

**HONEST MECHANISM CAVEAT (the chart tells what the number hides):** the book's equity curve is nearly
FLAT for the entire window -- it preserved by going almost entirely to CASH, not by skillfully timing the
crash. It captured essentially nothing on the way down AND nothing on the recovery. This is *insurance via
non-participation*, not a timing edge. "Loses less" is true; "makes money OOS" is false (net -0.9%).

**K=3 independent derivations of the load-bearing UNSEEN numbers (answer-frequency 3/3):**
- D1 (harness): book maxDD -2.31%, BH maxDD -40.17%, margin 37.9pp.
- D2 (raw parquet, no harness, independent PIT EW buy-hold): BH maxDD **-40.6%** (the preserve-signature
  denominator replicates within 0.4pp; net differs because D2 omits the activation/liquidity filter --
  but the maxDD, the claim under test, is invariant).
- D3 (independent rebuild + by-hand maxDD): book -2.31% / BH -40.17% / **margin 37.9pp** (exact match).

---

## 2. CANONICAL SCORECARD (full daily-net stream, SEL/OOS/UNSEEN + block-bootstrap p05)

`scorecard.score_book` on the book's full 2020-01-01 -> 2026-06-01 daily-net stream (n_days = 2338):

| split | compound | maxDD | Sharpe | n |
|---|---|---|---|---|
| SEL (pre-2025-03-15) | +90.2% | -25.7% | 0.98 | 1898 |
| OOS (2025-03-15..12-31) | **-19.4%** | -25.8% | -1.25 | 291 |
| UNSEEN (2025-12-31..2026-06-01) | **-0.9%** | -2.3% | -0.83 | 149 |

- **FULL block-bootstrap p05 = -17.9%** (NEGATIVE).
- **HELD-OUT (OOS+UNSEEN) block-bootstrap p05 = -38.8%** (NEGATIVE). [K=3 seeds {0,7,42}: -33 to -39%; SIGN invariant.]
- **scorecard `ship_read`: ship = FALSE** (unseen_compound_pos False, full_p05_pos False, heldout_p05_pos False).

**The decisive scorecard statistic: held-out p05 = -38.8% (< 0).** The book LOSES money out-of-selection
(OOS compound -19.4%) and the held-out compound's 5th percentile is deeply negative. As a standalone
held-out WEALTH strategy it FAILS the project ship gate. The +90% SEL compound is a bull-window artifact
(2020-2024 was net-up); the book does not generalize to a positive held-out compound.

---

## 3. ROBUSTNESS BATTERY (the referee pass)

### 3a. FAMILY-vs-NULL -- is it the FAMILIES, or just the de-risk/cash-going? (`family_vs_null_robustness.png`)

| family-set | 2022 bear book maxDD (vs BH -73.4%) | full-cycle 2020-2022 net (vs BH +9.7%) | UNSEEN net | UNSEEN maxDD |
|---|---|---|---|---|
| **translating** {trend,breakout,momentum,MA} | -19.8% (margin **53.6pp**) | **+24.7%** | -0.9% | -2.3% |
| **dropped/random** {volume,mean-reversion} | -4.8% (margin **68.6pp**) | +12.8% | -0.0% | -0.0% |
| **Sharpe-NULL pick** {volume,MR,MA,breakout} | -10.7% (margin **62.7pp**) | +16.4% | -1.0% | -1.7% |

**THE KEY DISTINCTION (this is the crux the task demanded):**
- **Bear preservation is NOT a translating-family edge.** The DROPPED families (volume+MR) preserve the
  2022 bear *strictly BETTER* (-4.8% maxDD, margin 68.6pp) than the translating families (-19.8%, 53.6pp).
  The Sharpe-NULL pick also preserves better (-10.7%). **All three family-sets crush BH's -73.4%.** The
  preserver is the **de-risk/go-to-cash overlay**, which is family-AGNOSTIC -- the dropped families, which
  are mostly flat/low-time-in, go to cash even harder and preserve even more. `family_edge` on the
  bear-2022 dimension = **FALSE**.
- **The ONE genuine translating-family differentiator is full-cycle COMPOUND** (+24.7% vs dropped +12.8%
  vs sharpe-null +16.4%). Translating participates MORE in the 2020/2021 bull while still preserving, so
  it out-compounds the cash-heavy dropped families over the cycle. `family_edge` on full-cycle = TRUE.
- **OUT-OF-SAMPLE (UNSEEN) all three are flat-to-negative.** Translating -0.9%, dropped -0.0%, sharpe-null
  -1.0%. Nobody made money OOS; the "edge" that survives is only the *relative* lose-less, which is
  strongest for the flattest (dropped) book.

**Honest correction to the harness label:** the harness `dropped_preserves_equally` flag reads FALSE only
because dropped preserves *better* than translating (margin gap +15pp), not because it preserves less.
The plain-English read is: **the dropped families preserve at LEAST as well as the translating ones** ->
the "family selection" is NOT what buys the crash insurance.

### 3b. JACKKNIFE (drop-one-family + concentration)

Dropping any single translating family barely moves the 2022 preservation (margin 51.5-55.7pp, all near
the 53.6pp baseline) or the UNSEEN result (net -0.4% to -1.3%, maxDD -1.5% to -3.1%):

| dropped family | kept | 2022 book maxDD (margin) | UNSEEN net / maxDD |
|---|---|---|---|
| trend | breakout,momentum,MA | -19.9% (53.5pp) | -1.3% / -2.3% |
| breakout | trend,momentum,MA | -19.7% (53.7pp) | -0.7% / -1.5% |
| momentum | trend,breakout,MA | -17.7% (55.7pp) | -1.2% / -3.1% |
| MA | trend,breakout,momentum | -21.9% (51.5pp) | -0.4% / -2.3% |

**No single family is load-bearing** -- the result is NOT concentration-driven within the translating set
(n_eff over 4 families is effectively the full 4; dropping any one preserves the signature). This is
expected and consistent with the family-vs-null finding: the preservation is the *shared* de-risk overlay,
so removing one family cannot break it.

### 3c. DE-RISK OVERLAY vs RAW signal (is preservation the SIGNAL or the OVERLAY?)

| level | 2022 book maxDD (margin) | 2022 net | UNSEEN net / maxDD (margin) |
|---|---|---|---|
| RAW (none) | -22.4% (51.0pp) | -15.9% | -1.4% / -2.7% (37.5pp) |
| LIGHT overlay | -19.8% (53.6pp) | -14.0% | -0.9% / -2.3% (37.9pp) |

**The RAW signal already preserves OOS** (UNSEEN margin 37.5pp with NO vol-target overlay; 2022 margin
51.0pp). The light overlay adds a marginal +0.4-2.6pp of preservation. PHASE 1's finding -- that the
preservation lives in the signal stack (trail-stop + min-hold + long-only-can-go-flat), not the
vol-target overlay -- **CONFIRMS out-of-sample.** The de-risk knob is a minor tuner, not the source.

---

## 4. VERDICT: AMBIGUOUS

**Gates:** preserve_oos = TRUE | held-out p05 > 0 = **FALSE** | family_edge (any dimension) = TRUE
(full-cycle only; bear-preservation family_edge = FALSE).

| H1 clause | result |
|---|---|
| preserve signature holds on UNSEEN (book maxDD < BH) | **PASS** (margin 37.9pp, K=3 confirmed) |
| held-out compound p05 > 0 | **FAIL** (p05 = -38.8%; OOS compound -19.4%) |
| translating families add something over dropped/random | **PARTIAL** (full-cycle compound YES; bear preservation NO -- dropped preserves better) |

H1 requires all three -> **NOT fully met -> AMBIGUOUS** (the preserve signature is real OOS, but the
held-out compound p05 fails and the family-selection story is only half-true). It is not ARTIFACT (the
headline preserve-signature genuinely replicated OOS) and not REAL (it is not a positive-p05 deployable,
and the crash insurance is not a family edge).

**Decisive statistic:** held-out (OOS+UNSEEN) block-bootstrap **p05 = -38.8%** with **OOS compound
-19.4%** -- the book does not earn a positive held-out return; it only loses less than buy-hold by sitting
in cash. Pair with the family-vs-null number: dropped/MR families preserve the 2022 bear BETTER (-4.8%
maxDD) than the translating families (-19.8%), so the crash insurance is the cash overlay, not the family
choice.

**The single cheapest falsifier:** run the SAME de-risk/trail-stop/long-only-go-flat overlay on a
PURE-CASH-OR-EW-BETA toggle (no signal families at all -- just "exposure on in calm rv, off in high rv").
If that family-FREE de-risk rule preserves the 2022 bear and the UNSEEN crash to within a few pp of the
translating book (which the dropped-family result already strongly implies it will), then the entire
"family-ensemble" framing adds nothing over a one-line vol-gate on EW beta -> the book collapses to "EW
beta with a volatility brake." That one control would settle whether ANY of the family machinery earns
its complexity. (Cost: ~5 minutes; it reuses `build_buyhold` + the vol-target clip already in the file.)

---

## 5. HONEST DEPLOYABLE RECOMMENDATION

**Do NOT deploy this as a standalone wealth strategy.** Held-out p05 < 0 and OOS compound -19.4% -- it
does not make money out-of-selection; it loses less than buy-hold by de-risking to cash. The +90% SEL
compound is a bull-window artifact.

**What it actually IS (and could be used for):** a drawdown-preserving DEFENSIVE OVERLAY / insurance
sleeve. The genuine, OOS-confirmed property is: *in a crash, this book is near-flat instead of -40 to
-73%.* That is real and replicated three ways. But:
1. The insurance is bought by the **de-risk/go-to-cash overlay**, which is **family-agnostic** -- the
   dropped families (volume+MR) deliver the SAME (better) crash protection. So if the goal is crash
   insurance, **the simplest version (a vol-gate on EW beta, or the flattest family set) is preferred** --
   do not pay for the 4-family ensemble complexity.
2. The translating-family book's ONE edge over the cash-heavy alternatives is **full-cycle compound**
   (more bull participation: +24.7% vs +12.8%) -- it is the best of the *preserving* books at also
   catching upside. So if you want preservation *plus* some bull participation across a full cycle, the
   translating book is the right pick among the de-risked variants -- at the **LIGHT** de-risk level (raw
   already preserves; light adds a marginal cushion; medium/heavy would over-de-risk and kill the bull
   participation that is its only differentiator).

**Expected behaviour by regime (OOS-grounded):**
- **Bull**: heavy UNDER-participation (~11% of buy-hold's bull net in 2021; it is not a bull strategy).
- **Bear/crash**: near-flat (UNSEEN crash: -2.3% maxDD vs BH -40.2%; 2022: -19.8% vs -73.4%). The value.
- **Chop/sideways (like UNSEEN)**: roughly flat, slightly negative (-0.9%) -- preserves capital, earns
  ~nothing.

**Net recommendation:** treat as a **0-to-small-allocation defensive diversifier**, NOT a core wealth
sleeve, and ONLY if a family-free vol-gate control (the cheapest falsifier above) fails to match it. Live
fills (p_fill 0.25-0.50 per CLAUDE.md) further haircut the already-negative held-out returns. The honest
status is: a real, replicated *risk* property, attached to a *return* profile that does not clear the
ship bar.

---

## Appendix: discipline checklist (pre-delivery self-audit)

- [x] **UNSEEN test-once**: the sealed window was computed exactly once; LIGHT de-risk is the FROZEN
      PHASE-1 pick, not re-tuned on UNSEEN. The selftest path provably never touches the UNSEEN window.
- [x] **Long-only spot**: positions in [0,1]; zero short logic (inherited from `_candidate_net_series_capped`).
- [x] **Fixed-EW**: `fillna(0.0).mean` everywhere (NEVER skipna); buy-hold cadence-invariant. Selftest (4) enforces.
- [x] **Survivorship-clean PIT**: data-derived listing dates; UNSEEN listing cutoff = window START
      (admit iff listed by 2025-12-31). RESIDUAL caveat: coins that traded pre-2026 but delisted are not
      in chimera (cannot fix from our data) -- but this is the UNSEEN window (2026), so the survivorship
      gap is minimal here vs the 2021 study.
- [x] **Frozen, no re-fit**: 2020-selected bands; UNSEEN is pure forward. The Sharpe-NULL family-pick is
      computed on 2020-only Sharpe (no UNSEEN peek).
- [x] **Causal / MtM-no-double-count**: positions lagged 1 bar; rolling rv shift(1); maker cost on flips
      (inherited verbatim from the PHASE-1 stack; 0.0 daily-diff reproduction proves no drift).
- [x] **Block-bootstrap (not iid)** on the autocorrelated daily stream (block=5); held-out p05 the gate.
- [x] **Multiple-comparisons honesty**: the family_edge "YES" came from ONE of two dimensions (full-cycle),
      not the bear dimension -- flagged explicitly; not reported as a blanket family edge.
- [x] **K=3 derivations** on the two load-bearing claims (UNSEEN preserve margin; held-out p05 sign);
      answer-frequency 3/3 on both.
- [x] **No emoji** (cp1252-safe). **Did NOT git commit.**
