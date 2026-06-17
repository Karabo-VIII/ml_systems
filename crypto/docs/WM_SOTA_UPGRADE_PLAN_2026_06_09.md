# WM SOTA Upgrade Plan — make the V0–V25 cohort a usable INPUT

> 2026-06-09. Focused mandate (user, after halting the broad router/cross-pollination/TI work): **upgrade the existing
> WM models V0–V25 to SOTA so they can act as an INPUT** for the crypto project. TI + the method-router + the
> cross-pollination bus are DEFERRED. Grounded in a deep, file-cited audit (the "A2 diagnosis") + recon (data present;
> V1.1 already trained at `models/wm/v1/v1_1/base/v1_1_f41_wm_best_ema.pt`).

## The honest diagnosis (don't chase a wall)

The best **validated** WM is V1.1: **IC h=1 ≈ 0.067, ShIC ≈ 0.033** (the rest of V3–V25 are mostly **unmeasured** —
"?" in every scoresheet D1 cell). The weakness is two distinct things:

1. **Architecture gaps — real, closeable, BOUNDED (~+0.015–0.020 IC → ceiling ≈ 0.08–0.087):**
   - **Unmeasured modern architectures.** V22 iTransformer (ICLR'24), V23 xLSTM (NeurIPS'24), V24 TimesNet (ICLR'23),
     V25 frontier-synthesis were **built but never trained/validated at f-full**. We literally don't know if one is better.
   - **Memorization ratio.** ShIC/IC ≈ 0.49 → ~half the measured IC is temporal fingerprinting, not OOS signal.
     `V1_HEADLINE_MODE` (tighter XD-dropout / KL free-nats / ATME) is coded but never run.
   - **Cross-asset.** V12's `forward_multi_asset` is dead code; V1.x processes each asset in isolation (median pairwise
     corr 0.55 — real contemporaneous structure left on the table).
   - **Sequence length.** V4/Mamba (linear-time, built for 512+ bars) runs at seq_len 96 like everything else.
2. **A signal wall + the WRONG TARGET — the bigger finding:**
   - **Signal wall:** IC>0.10 on daily/dollar bars is likely a structural ceiling (dead-list: direction AUC ≈ 0.51 on
     100+ assets; D44 needs IC≈0.60 for 1%/day; Kronos-12B TSFM scored 0.029 < our 0.067). The scoresheet itself says
     "Headline (IC>0.10) is the ceiling for daily/dollar-bar architectures; past it the bottleneck is tick-level
     representation (V20)." **So grinding IC past ~0.087 at this resolution is wall-chasing.**
   - **Wrong target:** the WM optimizes per-bar return (TwoHot h=1) — the exact lens the project's founding framing
     **BANNED** ("the unit is the SETUP across a MOVE; IC/per-bar predictability is the wrong lens"). The entry-signal
     lab found per-bar timing is **fungible** inside a move. A better architecture on the wrong target = a
     better-trained wrong model.

**Therefore "SOTA WM as input" is redefined (the elevation):** not "max IC" (a wall + a banned metric), but **"the
model that produces the most useful INPUT for the downstream decision layer, validated on held-out COMPOUND return."**

## Definition of done (verifiable, honest)

- The **cohort is measured** (the unmeasured modern architectures trained+validated; we know the best).
- The **cheap real gaps are closed** (memorization fix run; cross-asset wired; seq-len exercised) with honest before/after.
- The WM's value **as an input is PROVEN or REFUTED on held-out compound** (not IC) under the trading harness.
- The training **toolchain is RWYB-verified** (no silent training failure) before any multi-day run.
- **No false victory:** "good WM" is named as a multi-GPU-day program; IC>0.10 is stated as a signal ceiling, not promised.

## The program (EV-ranked; architecture-facing vs signal-facing tagged)

| # | Move | Type | Effort | Why it's not wall-chasing | Validate on |
|---|------|------|--------|---------------------------|-------------|
| 1 | **Proof-of-value: does V1.1 (already trained) add held-out COMPOUND as an input?** | measurement | LOW (no retrain) | The definitively-missing test; grounds the whole program; directly answers "can it act as an input" | held-out compound under the harness, vs a no-WM control |
| 2 | **Toolchain RWYB** (pre-train gate + 1-epoch smoke) | infra | LOW | Gate against silent training failure before any GPU-day run | gate exit 0 + a real loss step |
| 3 | **Measure the unmeasured modern architectures** (V22/V23/V24/V25) | architecture | HIGH (GPU-days each) | We built SOTA 2023–24 architectures and never checked them; pure information | IC/ShIC + then compound (move 1's harness) |
| 4 | **HEADLINE_MODE memorization fix** on V1.1 | training | MED (~2.5 GPU-days) | Strips the ~half-IC fingerprint; raises the GENUINE generalizing component (ShIC) | ShIC/IC ratio across 3 walk-forward windows |
| 5 | **Cross-asset wiring** (V12 `forward_multi_asset` + synced loader) | architecture | MED (~3 GPU-days) | The one structural gap V1.x literally cannot exploit (~+0.005–0.012 IC) | IC + cross-corr<0.85 → V10 ensemble |
| 6 | **Retarget pilot: setup/move-onset target** (not per-bar return) | objective-framing | MED | The audit's strongest finding; aligns with the founding framing + what the strategy actually uses | held-out compound (capture-rate), NOT IC |
| 7 | V4/Mamba @ seq_len 512 | architecture | MED | Exercise the one architecture built for long context (regime/funding cycles) | IC at regime transitions specifically |

**Compute reality (honest):** training is GPU-days *each* on the RTX 4060; a 12h window cannot produce a fully-trained
SOTA cohort. The window delivers: this plan + the toolchain RWYB + the **proof-of-value (move 1)** + the **highest-ROI
training kicked off in background** (moves 3/4) + honest verdicts. The full cohort is a multi-day program this plan
sequences. **The bottleneck is compute + the target framing, not the harness.**

## Cross-layer lessons folded in (from the games audit, deferred-but-relevant)

- **Monotonic champion gate** (games `train_robust.py`) → a WM **promotion gate**: a new WM version advances only if it
  strictly beats the current best on the PRIMARY metric (here: held-out compound, not IC) — no regression promotions.
- **Setup-not-candle / terminal-not-per-step** (games reward = game outcome, not per-move eval) → move 6 (retarget).
- **Anti-self-deception** (Wilson CI, forgetting-axis, no self-referential eval) → validate WMs on UNSEEN compound only,
  never on the training/IC leaderboard.

## First actions (this window)

1. **RWYB the toolchain** (move 2) — gate the cohort is trainable without silent failure.
2. **Proof-of-value** (move 1) — build the WM→signal bridge, measure V1.1's compound contribution as an input. This is
   the decision that shapes everything: if positive, the WMs are more useful than IC says and the upgrades are worth the
   GPU-days; if negative, we confront it honestly before burning compute.
3. **Kick off** the highest-ROI training (move 4 HEADLINE_MODE and/or move 3 a modern architecture) in background so the
   GPU produces a better model while the compute-light work proceeds.

---

## EXECUTION LOG — 2026-06-10 (code-hardening pass; training DEFERRED by the user)

> User clarified mid-pass: **harden + improve the architectures in CODE; leave the multi-day training for a dedicated
> compute window.** So this pass did NOT train; it removed the blockers + added the capability so a later training run
> produces good models. Proof-of-value used the *already-trained* V1.1 (no training).

**DONE + committed:**
- **`8d36695` — cohort-wide silent OOM fixed (the highest-impact find).** `AntifragileDataset` built a Python list of
  ~30M `(seg,start)` tuples → `MemoryError` on the full 30.2M-bar data → **no architecture could train at full scale**.
  Numpy index (~10× less RAM, behaviour-preserving). RWYB-confirmed: V22 loads all 30.2M bars + reaches Epoch-1 step.
- **`8280db8` — proof-of-value + reusable tools.** V1.1 as a long-only per-bar input LOSES to buy-hold in a bear
  (−72% vs −42%) but BEATS random entry +20pp (4/4 assets); regime gate ~coin-flip (49%). → **WM value as input =
  regime/bear detection, not IC.** Tools: `src/strat/wm_entry_producer.py`, `wm_value_probe.py`.
- **`bc9ef6a` / `fdf2167` / `62ac0b4` — hardening.** cp1252 crash-prints fixed (V1.1/V12/V25/_shared); `assert_canonical`
  drift-gate added to V3/4/6/8 (16 files, no drift); V24/TimesNet within-window look-ahead flagged at source. All
  cross-version constants + train-loop invariants verified CLEAN; V22–V25 already NaN-guard loss.
- **`<this pass>` — improvement scaffold (the value lever).** `src/wm/_shared/regime_targets.py` (forward bear-onset /
  forward-trend / move-onset label builders; no-look-ahead PROVEN) + `forward_regime_head.py` (OFF-by-default head +
  masked aux loss). Base training byte-for-byte unchanged; wiring is spec'd + deliberately unapplied; validated later on
  held-out COMPOUND (not IC).

**Verdict:** the V0–V25 cohort is now **SOTA-ready in code** — it can build + train the full dataset, invariants are
clean, the modern architectures are trainable, and the regime/move-onset improvement (the proof-of-value's lever) is
scaffolded and ready to wire. The bottleneck is now **the user's training compute window + the target reframe**, not
the harness.

**REMAINING WM work (code/scope, still no training):**
1. **V12 cross-asset unblock** — the one structural gap V1.x can't address (`forward_multi_asset` dead code +
   `MultiAssetDataset`). ~1–2d build; highest-value remaining architecture improvement.
2. **HEADLINE_MODE memorization config** — verify `V1_HEADLINE_MODE` wiring (tighter XD-dropout/KL/ATME) so the ShIC
   fix is a clean switch at train time.
3. **Compound-not-IC promotion gate** — wire `wm_value_probe` into a monotonic WM-version promotion gate (the games
   champion-gate lesson): a new version advances only if it raises held-out compound, never IC.
4. **Wire the regime/move-onset head** (the spec in `forward_regime_head.py`) when training resumes; A/B vs per-bar
   target on held-out compound.
5. Minor: `xd_funding_spread` (idx 36) is a constant-zero dead feature in the f41 list (data-source gap; low impact).
