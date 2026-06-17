# Apparatus Red-Audit + Hardening (2026-06-05)

> **What this is:** the −k "is the method sound?" check on the foundation's central promise — a
> TRUSTWORTHY validation apparatus. An adversarial auditor attacked every apparatus tool; each finding
> was then **verified against the actual code** (per CLAUDE.md rule 13 — agents hallucinate even
> adversarially) and triaged CONFIRMED / REFUTED. Confirmed fixes were applied to the consolidated
> `src/strat/` layer (and one isolated canonical-harness fix), each RWYB-validated. This is FOUNDATION
> work (hardening the apparatus), **not solving** (no edge was sought; conclusions held OPEN).

## Outcome

The apparatus was **not yet trustworthy** as audited. Seven verified holes were fixed and RWYB-proven;
two auditor "criticals" were **refuted by code verification** (the auditor applied the wrong invariant);
four are canonical-harness items documented for review. The apparatus now passes its known-result
sanity check (BTC 1d taker reproduces FOUNDATION §15: TRAIN ~82% not 135%, OOS negative) and correctly
NOT-SHIPs the known PEPE-provisional candidate while surfacing leak/concentration flags it previously
could not.

## Verified triage

| ID | Sev | Finding | Verdict | Action |
|---|---|---|---|---|
| **F6** | HIGH | block-bootstrap start-index off-by-one (`sp=a.size-block`) never sampled the last observation in any resample | **CONFIRMED** | FIXED `src/strat/battery.py` (`sp=a.size-block+1`) |
| **F12** | MED | adverse selection `gross*(1-adverse)` shrank the MAGNITUDE of losses (made losers *less* bad — backwards) | **CONFIRMED** | FIXED `src/strat/fill_model.py` (penalty `gross - adverse*\|gross\| - cost`) |
| **F2** | CRIT→HIGH | firewall `beats_held` used `... if x is not None`, silently dropping a zero-trade window from `all()` | **CONFIRMED** (mitigated: `pos_held`/`all_4_positive` already blocked a false SHIP in both call sites) | FIXED `src/strat/firewall.py` (`is True`; zero-trade ⇒ FAIL) + gate uses the hardened flag |
| **F11** | MED | integrated gate used the ADVISORY, cadence-sensitive `shift_sensitivity_test` as a HARD gate | **CONFIRMED** | FIXED `src/strat/candidate_gate.py` (now `relative_leak_test` vs an auto-built clean twin) |
| **F9** | MED | `StrategySpec` default `cost_rt=0.0010` (maker) + `from_r12_defaults` default maker — naive callers get optimistic cost | **CONFIRMED** | ENFORCED at gate: `evaluate_candidate` warns loudly (`cost_warning`) below taker 0.0024; canonical default documented (below) |
| **F5** | MED | `discriminate` forward label `close.shift(-H)` at a window's last bars uses closes in the NEXT window (boundary-crossing label) | **CONFIRMED** | FIXED `src/strat/discover.py` (keep only rows where `w == w_fwd`) |
| **F7** | MED | Lens A `(p05 or -1) > 0` truthiness pitfall | **CONFIRMED (functionally harmless)** | CLARIFIED `src/strat/battery.py` (`p05 is not None and p05 > 0`) |
| **F8** | HIGH | `signal_flip` exit = `closes[j]>closes[j-1] and not signal[j]` — up-close condition contradicted "flip only"; held through down-bars | **CONFIRMED (latent: no apparatus path uses this exit_policy)** | FIXED `src/wealth_bot/harness.py` (pure `not signal[j]`) — isolated, reversible, zero current-result impact, **uncommitted/flagged** |
| **F1** | CRIT | "no MtM reconciliation gate" | **REFUTED** | This engine is **trade-level** (`exit_p/entry_p-1-cost`, single-position, `i=max(exit_fill_bar,…)`) — the (2N-1)/N bar-level-MtM double-count class structurally cannot occur. Auditor applied the wrong invariant. No action (a no-overlap assertion could be added as a bonus, low priority). |
| **F4** | HIGH | "random-entry null exit can drift cross-window" | **REFUTED** | INTENTIONAL parity: the REAL harness also holds trades across window boundaries. Clipping only the null would break the apples-to-apples comparison and is *less* conservative. Documented in `firewall.py`, not changed. |
| **F3** | CRIT→LOW | trade window label uses the SIGNAL bar (`entry_i`) not the FILL bar (`entry_fill_bar=entry_i+1`) — boundary trades misattributed | **CONFIRMED but QUANTIFIED IMMATERIAL** — `runs/staging/probe_f3_window_label_2026_06_05.py` on the PEPE candidate (152 trades): **0 boundary-straddlers, delta_pp = 0.00 every window**. The misattribution requires a signal to fire on a window's exact last bar, which did not occur. | DEFERRED (now quantified-safe) — recommended one-liner below stays review-optional; re-run the probe per candidate before relying on it |
| **F13** | LOW | `WindowSpec.train_start` stored but unused by `_window_label` | **CONFIRMED** | DOCUMENTED — no apparatus path relies on `train_start`; recommended guard below, review-required |
| **F14** | LOW | battery `n`/`n_eff` are UNSEEN-only counts | **CONFIRMED (intentional)** | DOCUMENTED in `battery.py` (held-out strictness, not a bug) |

## Canonical-harness items deferred for review (NOT silently changed)

The harness is the trust-critical engine; per the staging directive ("stage trust-critical for review")
these are documented with the exact recommended fix rather than mutated mid-autonomous-run:

- **F3** — change `src/wealth_bot/harness.py:663` `ts = df["date"].iloc[entry_i]` →
  `ts = df["date"].iloc[entry_fill_bar]` (label by the bar where capital deploys). Requires matching the
  firewall's `eligible` window assignment to the fill bar (`wlab[i+1]`) for parity. Impact: re-attributes
  ≤1 boundary trade per window split — small but removes a held-out-purity leak vector.
- **F9** — change `StrategySpec.cost_rt` default and `from_r12_defaults(cost_rt=…)` default from `0.0010`
  (maker) to `0.0024` (taker) so the conservative cost is the default. Ripples to every `from_r12_defaults`
  self-test (numbers get *more honest*, e.g. TRAIN 135%→82%). Handled at the gate layer this run via the
  `cost_warning`; the canonical default change is the cleaner fix.
- **F13** — add `if self._train_s is not None and ts < self._train_s: return "PRE"` (or drop pre-start
  rows) so `train_start` is honored.

## RWYB evidence (all self-tests pass — `python src/strat/<module>.py`)

```
battery.py        GOOD->SHIP-TIER(LensA, n_eff 25, p05 8.0); GHOST->FAIL(n_eff 1.3, conc_flag); CHASER->gate_pass
fill_model.py     taker TRAIN +81.9% / OOS -5.7% / UNSEEN +6.1%  (reproduces FOUNDATION §15 known-result)
                  maker_pessimistic collapses (TRAIN -86.6%, OOS -19.4%)  (F12: losses deepen correctly)
firewall.py       BTC R12 beats_null=False all windows; beats_held=False -> BETA-IN-DISGUISE  (F2 flag honest)
candidate_gate.py PEPE whale-gated -> NOT-SHIP; leak ratio 0.965 = PAST_ONLY_OK (F11 relative twin works);
                  firewall_held=False; concentration_flag=true (n_eff 6.9); cost_warning=null (taker)
discover.py       discriminate: liq_delta_z30 = the one same-sign-4-window + beats-null-on-UNSEEN gate
                  (SURVIVES the F5 boundary-label fix); scan 8-cell proof -> 0 SHIP-tier, LEAK_SUSPECT now
                  surfaced on some whale-gated cells (F11 sensitivity the old gate lacked)
positive_control.py  POWER check (the false-NEGATIVE half of soundness): a synthetic GENUINE past-only
                  timing edge -> firewall_beats_held=True, all-4-positive (TRAIN +1104% / UNSEEN +44%),
                  leak-clean, battery-recognized -> HAS_POWER=True. (A whipsaw variant -> FAIL, correctly
                  rejected.) SHIP-TIER not reached: n=8 < 15 on the ~143-day UNSEEN window -- sample-size
                  discipline, reserving Lens A for a higher-frequency substrate, NOT a power gap.
```

## LD-3 DSR/Holm gate — the last broken lock-down item, now FIXED

Separate from the F-series (which audited the new `src/strat/` tools), the `src/audit/check_dsr_holm.py`
CDAP gate (the family-wise multiple-testing correction) was a **no-op**, confirmed by code:
- failing Holm was *always* `severity="warn"` (line 181) → never halted a commit, even for a genuine
  ship-CLAIM (the docstring promised "flag as CRIT exit 2" — never delivered);
- `n_trials = len(candidates)` = count of *written* ship-claim JSONs → undercounts the true family-N
  (NULL/REFUTED rounds + aggregation DoF that were tested but never written) → Bailey threshold too lenient.

**FIXED (staged uncommitted, RWYB `python src/audit/check_dsr_holm.py --selftest`):**
- a candidate that CLAIMS ship-tier AND fails Holm → `severity="critical"` → exit 2 (HALT). A NULL/REFUTED
  round failing Holm stays `warn` (informational). `_is_ship_claim()` positively identifies a ship verdict;
  ambiguous JSONs warn (conservative).
- family-N = `max(written, manifest _sweep_manifest.json n_variants_tested, per-JSON declared)`; Holm divisor
  and Bailey SR_threshold use this honest family size.
- selftest: strong-ship@N=1 → exit 0; weak-ship@N=200 → **exit 2 (HALT)**; weak-NULL-round@N=200 → exit 1.
- Zero blast radius: `runs/audit/` is empty post-reset, so the live gate scans nothing (exit 0) — the change
  only bites a *future* sweep that writes a ship-claim JSON. Integration: `check_invariants.py` re-maps the
  finding severity, so a critical correctly propagates to commit-halt (verified the wiring + a clean run).

## Foundation completeness-critic follow-up (2026-06-05 PM)

A second independent agent red-teamed the WHOLE foundation for what's MISSING (avenue / apparatus /
methodology / framework / reproducibility gaps). 12 gaps returned; verified against code and triaged:

| Gap | Sev | Verdict | Action |
|---|---|---|---|
| G-1 STEP-5 benchmark-excess had no callable (gate chain incomplete) | CRIT | CONFIRMED | FIXED — built `src/strat/benchmark.py:benchmark_excess` (candidate-net vs beta-matched costless static hold, per regime incl. bear); wired as a hard ship gate in `evaluate_candidate` (`beats_beta_held`). RWYB: genuine edge beats beta every window; PEPE candidate beats_beta on held-out but still NOT-SHIP (fails firewall) — the multi-gate chain surfaces "beta-excess is regime luck, not timing". |
| G-4 StrategySpec default cost still maker 0.0010 | HIGH | CONFIRMED | FIXED — applied F9 canonical: `StrategySpec.cost_rt` + `from_r12_defaults` default → 0.0024 taker (+ docstring example G-12). RWYB: firewall BTC-R12 TRAIN now +81.9% (honest taker) not +134.9%. |
| G-5 "10-seed p05>0" is a misnomer (battery uses block-bootstrap) | HIGH | CONFIRMED | FIXED — clarified in README/`__init__`/AVENUE_SPECS: block-bootstrap p05 is the seed-equivalent for a deterministic rule strategy; the 10-seed bar applies to stochastic/ML candidates (add a seed loop). |
| G-6 README still says DSR "KNOWN-BROKEN" | MED | CONFIRMED | FIXED — README gate chain updated to "FIXED 2026-06-05; ship-fails-Holm → exit 2". |
| G-7 window dates hardcoded in 3 modules; AVENUE_SPECS UNSEEN-date typo | MED | CONFIRMED | FIXED — added `strat.DEFAULT_WINDOWS` (single source); corrected the UNSEEN boundary in AVENUE_SPECS to [2025-12-31, 2026-05-22]. |
| G-3 "everything uncommitted" | HIGH | REJECTED (not a gap) | INTENTIONAL: staging-uncommitted is the user's directive ("stage trust-critical for review") + file-checkpointing/lineage makes it retrievable. Committing trust-critical code unauthorized would VIOLATE the directive. The user decides when to commit. |
| G-2 firewall tiered-cost ladder + regime-matched-null variant | HIGH | CONFIRMED | PARTLY FIXED — regime-matched-null DONE (`random_entry_null(..., regime_matched=True)` draws the null from gate-ON bars only; RWYB on BTC-R12 both modes → BETA-IN-DISGUISE). Tiered-cost ladder DEFERRED (needs $-vol ranking data; realism-tuning, solving-phase). |
| G-8 generalized Phase-1 base-book tool | MED | CONFIRMED | DEFERRED — building a multi-asset base book edges into solving; honest-base v1/v2 already explored (failed the floor) in staging. Marked OPEN PRE-SOLVING. |
| G-9/10/11/12 (survivorship not quantified; "VERIFIED" data-paths transient; leak vacuous for filter-less; harness docstring) | MED/LOW | CONFIRMED | G-12 FIXED (docstring); G-9/G-10/G-11 = documentation notes for the solving phase (transient-data caveat, survivorship haircut procedure, filter-less leak limitation). |

Net: the two CRITICAL/HIGH apparatus-completeness gaps (G-1 missing STEP 5; G-4 optimistic default) are CLOSED + RWYB-proven. The gate chain is now end-to-end built. Remaining items are solving-phase enhancements (G-2/G-8) or doc notes (G-9/10/11) — none block the foundation. G-3 is intentional, not a gap.

## Two-sided soundness (the key result)

A gate that rejects everything is useless; so is one that accepts everything. This apparatus is proven on
BOTH sides: it **REJECTS** (concentration ghost -> FAIL; BTC-R12 beta -> BETA-IN-DISGUISE; whipsaw crossover
-> FAIL; 8-cell scan -> 0 SHIP) **AND ACCEPTS** (positive_control: a hand-crafted genuine timing edge beats
the firewall + is recognized). It is calibrated, not a sieve in either direction. The one nuance surfaced:
**SHIP-TIER (Lens A) is sample-size-gated** (n>=15 / n_eff>=15) -- a low-frequency genuine edge tops out at
PRAGMATIC/PROVISIONAL on a short held-out window. That is intentional (real-capital sample discipline) and it
tells the solving phase that a clean SHIP-TIER result wants a higher-frequency substrate (finer bars / longer
held-out), consistent with AVENUE-1's priority.

## What this changes for the foundation

The apparatus promise ("a trustworthy gate any avenue flows through") is now **earned, not nominal**:
the gate was adversarially attacked, the real holes fixed, the false alarms refuted with code evidence,
and the whole chain RWYB-proven on real data. The `src/strat/` layer (turnkey, importable) is the home;
`src/strat/README.md` is its contract. **Conclusions remain HELD OPEN** — this run hardened the *measuring
instrument*, it did not measure any avenue for an edge.
