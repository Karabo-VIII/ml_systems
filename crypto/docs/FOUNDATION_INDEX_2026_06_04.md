# FOUNDATION — Index & Entry-Point (2026-06-04/05)

> **What this is:** the single entry-point to the project foundation laid 2026-06-04/05. The GOAL of
> this phase was to **SET THE FOUNDATION** (frame the problem, map the avenues, build a trustworthy
> apparatus, fix the methodology + operating framework) — **NOT to solve whether an edge exists.**
> Conclusions are HELD OPEN; the 4h/daily-LO corner is the KNOWN-HARD starting premise, not a verdict.

## 1. The document map
| Doc | Purpose |
|---|---|
| [`FOUNDATION_2026_06_04.md`](FOUNDATION_2026_06_04.md) | The full record (§1–6 problem framing & architecture; §7 research; §8 framework-eval; §9–24 the 4h/daily-LO exploration + apparatus — read with the re-frame banner). |
| [`AVENUE_MAP_2026_06_04.md`](AVENUE_MAP_2026_06_04.md) | **The core deliverable** — the 8-axis exploration space, per-avenue status (EXPLORED/PARTIAL/UNEXPLORED/KNOWN-HARD), the apparatus to test each, and the EV-ranked queue for the solving phase. |
| [`AVENUE_SPECS_2026_06_05.md`](AVENUE_SPECS_2026_06_05.md) | **Executable per-avenue specs** — for each of the 8 avenues: mechanism hypothesis, pre-registered falsifier, apparatus tools (in order), data needed, EV rationale, expected failure mode. The recipes the solving phase runs. (paths agent-verified; conclusions held OPEN) |
| [`APPARATUS_LOCKDOWN_SPEC_2026_06_04.md`](APPARATUS_LOCKDOWN_SPEC_2026_06_04.md) | The trustworthy-measurement contract (cost+fill realism, leak probe, family-N gate, random-entry-null firewall, bear-inclusive holdout). |
| [`APPARATUS_AUDIT_2026_06_05.md`](APPARATUS_AUDIT_2026_06_05.md) | **The −k "is the method sound?" check** — adversarial red-audit of the apparatus + verified triage (7 holes fixed + RWYB-proven, 2 auditor-criticals refuted by code, 4 canonical items deferred for review). The proof the gate is trustworthy, not just nominal. |
| [`RETEST_PLAN_2026_06_04.md`](RETEST_PLAN_2026_06_04.md) | The methodology + phasing (Phase 0 apparatus → Phase 1 base → the forks). |
| [`src/strat/README.md`](../src/strat/README.md) | **The apparatus contract** — the turnkey importable home (`battery / firewall / fill_model / candidate_gate / discover`), the gate chain, the run-the-self-tests block. |
| [`runs/staging/MANIFEST_2026_06_04.md`](../runs/staging/MANIFEST_2026_06_04.md) | Every staged script + finding + integration note. |

## 2. The apparatus — CONSOLIDATED into `src/strat/` (turnkey, RWYB-validated; staged uncommitted for review)
The scattered `runs/staging/*_2026_06_04.py` scripts were ported 2026-06-05 into one coherent importable
package + hardened against the [apparatus red-audit](APPARATUS_AUDIT_2026_06_05.md). Contract: [`src/strat/README.md`](../src/strat/README.md).

| Module | Role | Status |
|---|---|---|
| `src/strat/battery.py` | the reusable Lens A/B/C robustness gate | PORTED + F6/F7 hardened + self-tested (ship/ghost/chaser) |
| `src/strat/firewall.py` | LD-4 cost-matched random-entry null (PRIMARY gate) | PORTED + F2 hardened (zero-trade window ⇒ FAIL); RWYB flags beta-in-disguise |
| `src/strat/fill_model.py` | LD-1 cost+fill realism (taker/maker/ideal) | PORTED + F12 hardened (adverse-sign); taker reproduces FOUNDATION §15 |
| `src/strat/benchmark.py` | **STEP 5** benchmark-excess (candidate-net vs beta-matched costless static, per regime incl. bear) | BUILT 2026-06-05 (closes completeness-gap G-1); wired as a hard ship gate; RWYB |
| `src/strat/candidate_gate.py` | **the INTEGRATED gate** (cost→leak→firewall→battery→benchmark→verdict) | PORTED + F11 (relative-leak twin) + F9 (taker enforce) + STEP 5; RWYB on PEPE → NOT-SHIP |
| `src/strat/discover.py` | discovery front-end (`discriminate`) + scan loop | PORTED + F5 (boundary-label) hardened; `liq_delta_z30` survives the fix |
| `src/strat/positive_control.py` + `selftest_all.py` | gate POWER proof + one-shot data-free regression runner | RWYB: HAS_POWER=True; `selftest_all` 4/4 PASS (battery+dsr+power+benchmark) |
| `src/audit/check_dsr_holm.py` | LD-3 DSR/Holm family-N gate | FIXED 2026-06-05 (ship-fails-Holm → exit 2; honest family-N); `--selftest` passes |
| `src/wealth_bot/leak_probe.py` | LD-2 look-ahead probe | `relative_leak_test` cadence-robust (legit→OK ratio ~1.0, leaked→LEAK ratio ~2.6); now wired into the gate |
| `src/wealth_bot/harness.py` | the kept canonical engine | F8 (signal_flip) fixed uncommitted/flagged; F3/F9/F13 documented for review (see audit) |
| `src/pipeline/make_chimera_bars.py` (existing) | builds info-driven-bar chimeras (the unexplored substrate) | PROOF-built (PEPE runs_tick, 1 ok/0 fail) → substrate is buildable |
**Apparatus invariant:** every future candidate runs `evaluate_candidate` = cost-lens → relative-leak → firewall → battery → (solving-phase: bear-holdout → DSR@family-N).
**RWYB:** `python src/strat/selftest_all.py` (one-shot data-free regression → 4/4 PASS) + the per-module + data-dependent smokes (evidence in the audit doc).

## 3. The operating framework (live)
AUTONOMOUS_RUNNER (§6 consensus-for-whiplash + claim-integrity guards) · STANDARDS rule 13 (non-linear self-improving, inherited by all skills) · the global autonomous-mode hook (`.claude/autonomous_mode.json` + `scripts/autonomous_mode_check.py`, checked every turn) · bypass-permissions + file-checkpointing · the discover/trader skill banners (post-reset caveats).

## 4. Status — conclusions HELD OPEN
- **KNOWN-HARD (premise):** 4h/daily-LO active alpha (re-confirmed this run; that re-confirmation was the over-reach, not the deliverable).
- **The genuinely-UNEXPLORED frontier (the foundation's job to equip, the solving-phase's job to test):** info-driven bars whose chimeras are unbuilt (dib/runs/adaptive_vol × universe); the 184 features as GATES (esp. s3-smart-money, hawkes, liq, macro); ML-meta-label on a proven gate; self-improving decay-rotation; WM-as-trainer; the full TI×ASSET×REGIME×TIMEFRAME grid; setup-capture rotation.

## 5. The next phase (SOLVING — a separate mandate, NOT this one; needs user authorization)
Per the AVENUE_MAP EV queue: (1) build the missing info-driven-bar chimeras → mine selective capture; (2) the gate-space grid (features-as-gates); (3) ML-meta-label on survivors; (4) cross-sectional/breadth pooled; (5) self-improving/WM after a validated library exists. Each goes through the §2 apparatus. **The foundation is set; finding an edge is the next phase.**

### START-HERE for the first solving session (when authorized) — operational, not run here
1. **Confirm the apparatus:** `python src/strat/selftest_all.py` (must be 4/4 PASS) — and the BTC-1d known-result sanity check (`python src/strat/fill_model.py` → taker TRAIN ≈ +82%, OOS negative). Gate: do not mine until both green.
2. **Cheapest first avenue = AVENUE-2 (gate space), zero build cost** (1d chimera materialized): `strat.discover.discriminate(sym, '1d')` across BTC/ETH/PEPE/DOGE on the full `GATE_FEATS` family. The standing lead is **`liq_delta_z30`** (the one same-sign-4-window + beats-shuffle-null gate; SURVIVES the F5 fix). DISCRIMINATION ≠ HARVESTABILITY — it must still clear the gate.
3. **Pre-register family-N** BEFORE opening any cell: write `n_variants_tested` into a `_sweep_manifest.json`; the DSR gate (`src/audit/check_dsr_holm.py`, now fixed) reads it — a ship-claim that fails Holm at the true family-N now HALTS (exit 2). Touch UNSEEN (`strat.DEFAULT_WINDOWS`, [2025-12-31, 2026-05-22]) ONCE per finalized candidate.
4. **Run each cell through `strat.scan` → `strat.evaluate_candidate`** (cost→leak→firewall→battery→benchmark). SHIP-TIER requires Lens A + firewall-beats-held + beats-beta-held + leak-clean + taker cost. Honor the §AVENUE_SPECS solving-phase caveats (survivorship haircut, re-verify data paths, filter-less leak needs a bespoke reference).
5. **Build chimeras only if a 1d gate survives** (AVENUE-1, compute-heavy): `src/pipeline/make_chimera_bars.py --bar-types dib range runs_tick adaptive_vol` (proof-built). Higher-frequency substrate is also where SHIP-TIER's n≥15 becomes reachable (per the positive-control finding).
