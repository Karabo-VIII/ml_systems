# Strategy-Decision Agent -- Design (2026-06-11)

> How to tackle the strategy problem: an agent that learns LINEAR + NON-LINEAR relationships between technical signals (MA, chimera) and GOOD SETUPS, decides enter, lets the move play out, exits by policy. The CONDITIONAL setup-edge framing (NOT per-candle IC). From a 4-lens design workflow (audit-prior / Kronos / design / honest-ceiling).

---

# SYNTHESIS (decision-grade)

All four load-bearing claims verified against actual code: the next-bar-open fill contract (setup_harness.py:245-246, "Pattern T banned"), the `regime_matched` flag isolating within-gate timing (firewall.py:52-60), the `membership_matched` flag isolating trigger-timing from move-selection (firewall.py:62-72), and both broken/correct prior-attempt artifacts. Writing the synthesis now.

---

# THE STRATEGY-DECISION AGENT: A Conditional Setup-Edge Meta-Labeler over Technical Signals

**Decision-grade synthesis. 2026-06-11. Grounded in code (file:line cited), tagged MEASURED / VERIFIED-elsewhere / REPORTED / PROJECTED.**

---

## 1. THE REFRAME — stated crisply, and why the dead-list does NOT refute it

**What you want:** an agent that learns the LINEAR and NON-LINEAR relationships between classic technical signals and good setups — "the MA says (x,y), chimera feature 21 says this → I buy, let the move play out, exit by policy." Named precisely, this is a **conditional meta-labeler / setup-selector**, not a forecaster.

**The load-bearing distinction (this is the whole argument):**

- **What the dead-list killed** is the UNCONDITIONAL, PER-CANDLE object: `E[r_{t+h} | x_t]` has ~0 predictability. That is the IC lens — per-candle prediction, even multi-period — and MEMORY.md bans it as a primary metric.
- **What you are proposing** is a DIFFERENT object: `P(this multi-candle MOVE pays net cost | a classic trigger fired AND context features C hold)`. A trigger (MA cross, breakout) with **zero unconditional edge** can, in principle, have **positive edge on the sub-population where C holds**.

Conditioning is not prediction. A coin that is fair overall can be biased on the subset "landed on a red table." The dead-list refuted the unconditional trigger (D17/D44/D45/D55 — direction HARD-null across 100+ assets × 5 TF) and banned per-candle IC (D13). It did **not** test "does the trigger have edge CONDITIONAL on context." That is the surviving question, and it is explicitly the **one ML use the project did NOT kill: D16 — "ML-as-alpha dead; meta-labeler survives."**

So: **the reframe is sound. "The dead-list killed this" is FALSE — it killed a sibling, not this object.**

**The honest counterweight, up front:** the conditional hypothesis has its own ceiling. It can fail two ways that are NOT the IC failure: (a) the context features genuinely don't discriminate setup-quality — the conditional edge is *also* ~0; (b) the features *appear* to discriminate in-sample only because the meta-labeler searched a large config space and selected the path-fit winner. And the most important empirical fact in this entire analysis: **this exact object was already built once, carefully, and returned a null** (see §6). Treat "conditioning produces a robust edge" as **[PROJECTED]** until the gauntlet passes — not pessimism, the only honest prior, because conditional-edge meta-labeling is the single most overfit-prone construct in quant.

---

## 2. WHAT THE PRIOR ATTEMPT GOT WRONG — salvage vs redo (file:line grounded)

Five prior ML attempts split cleanly into **two eras**. The OLD pair is exactly the careless work you distrust; the NEW trio already embodies the right methodology (and honestly returns NULL). MEASURED from code reading + the two NEW artifacts' committed run-JSONs.

### The OLD pair — REDO (un-runnable today, wrong goal)

**`src/training/train_rank_model.py` — THE CLEAREST WRONG GOAL (the IC trap, literally).**
- GOAL: cross-sectional LambdaRank — "which of today's assets outperforms tomorrow."
- LABEL: `fwd_ret = diff(daily_close)/close`, then within-day **rank of the 1-bar return** (L109-110, L151). **This is per-candle prediction — the exact banned IC lens.** No entry, no hold, no exit, no setup.
- VALIDATION: top-1 hit-rate (L250-265). **No cost model, no compound, no seeds, no walk-forward, no shuffled control.** Accuracy on a per-candle rank, never converted to wealth.
- VERDICT: **REDO entirely.** Salvage only the chimera-feature plumbing (`FEATURE_NAMES`, the polars daily-panel loader).

**`src/training/train_meta_labeler.py` — RIGHT SHAPE, BROKEN EXECUTION** (the prototype of what you want, done carelessly — VERIFIED against code this session):
- GOAL (correct in spirit): López-de-Prado meta-labeling — fire → P(win|context) → size. This IS the conditional-edge idea.
- **LABEL BUG (VERIFIED L109): `win = 1 if ret_trade > 2*cost/2 else 0`.** `2*cost/2 == cost` — the win threshold is barely-above-breakeven, making the label a near-coin-flip by construction, throwing away the move magnitude that drives wealth.
- **SPLIT BUG:** records appended **engine-outer/asset-inner** (build loop), then sliced `int(0.70*n)` — so "first 70%" is the first ~70% of (engine,asset) pairs, **NOT chronological**. Silent leakage masquerading as a time split. Plus an imported-but-unused `train_test_split` (the tell of unfinished work).
- WRONG OBJECTIVE: trains/reports **AUC + Brier only** (L234-239). Accuracy is orthogonal to compound (right on many tiny winners, wrong on the few large losers).
- **REGIME-AS-THE-ONLY-SIGNAL, unguarded:** features include asset/regime/bucket one-hots (L198). With no firewall and no per-regime reporting, the model learns "bull-regime ALT trades win" = **beta, not conditional timing edge.**
- **DEAD ON ARRIVAL:** imports `from cost_model import SPOT_COST` etc. via `sys.path.insert(.../src/strategy)` (L38, L41-46). `src/strategy/` was deleted at the 2026-06-04 reset; `triple_barrier_exit.py` is gone. **It cannot run today and persisted no artifact** (`models/meta_labeler/` holds only an unrelated `v8_catboost.pkl`).
- VERDICT: **REDO label + split + validation. SALVAGE the meta-labeler ARCHITECTURE** (fire→P(win|context)). **Do NOT patch — rebuild on the NEW template, delete the dead `src/strategy/` import path.**

### The NEW trio — SALVAGE (methodologically correct, empirically NULL)

**`src/mining/mover_metalabel.py` — THE TEMPLATE, DONE RIGHT, HONESTLY NULL** (VERIFIED: contract block L67-78, pre-registered constants L80-92):
- Meta-labeler on a PROVEN, mechanism-chosen trigger (`ml_as_metalabeler_only`, L72; cites D16/D17 to forbid signal-generation). 16 causal-at-trigger features, z-scored from **TRAIN stats only**. Label = net-of-cost outcome of a multi-candle trailing ride (setup-level, cost-aware — the label train_meta_labeler should have had). Pre-registered tau (67th TRAIN pctile, L89). Four OOS ALIVE gates (portfolio net>0, breadth≥6, drop-top-3 jackknife>0, AUC>0.52). UNSEEN sealed. Git SHA + seed + lineage in JSON. Fixed-span annualization.
- **EMPIRICAL (MEASURED):** HGB **train AUC 0.99984 vs OOS AUC 0.5210** — textbook memorization; `oos_gates.alive = False`. **Honest NULL — the discipline worked, the gate caught it.**
- VERDICT: **SALVAGE WHOLESALE as the canonical template.** One weakness to fix: it does not wire in `shuffled_market_control.py` (the policy-overfit gate); add it.

**`src/strat/vol_config_ml.py` — SOPHISTICATED, HONESTLY NULL** (MEASURED: margin-trap / not-learnable on real BTC/ETH/SOL):
- The standout instrument: it **separates LEARNABILITY from PROFITABILITY** ("the margin trap" — vol can predict the config-class above base-rate while *acting* on it fails to beat fixed). Ships a TWO-SIDED selftest (positive control must return YES, negative must return NULL). These are the exact instruments that stop the IC-trap self-deception.
- VERDICT: **SALVAGE the learnability⊥profitability split and the two-sided selftest.** Extend coverage (walk-forward + seeds + UNSEEN) only if a future variant shows non-null.

**`src/wealth_bot/framework/signal_picker.py` — WIRED, but a REGRESSOR (mild IC-trap risk):**
- Per-strategy LGBM **forward-return regressor**; picks the highest predicted-fwd_ret among firing strategies. Genuine walk-forward, non-overlapping execution, seed-decorrelated sub-streams, per-refit importance logging — **production-grade plumbing.** But the objective is still "predict the return," and its `+49.8% UNSEEN on PEPE/EMA` headline is from an **ARCHIVED dossier** (REPORTED, not currently reproducible — do not cite as live; re-verify under the hardened apparatus, like the debunked REGIME_ROUTER +20.25%).
- VERDICT: **SALVAGE the WF/non-overlap/seed-decorrelation machinery. REFRAME the head** from fwd_ret regressor → setup-level net-of-cost **classifier** (converge onto mover_metalabel's framing).

### Salvage/Redo summary

| File | Era | Goal right? | Runs today? | Action |
|---|---|---|---|---|
| `train_rank_model.py` | OLD | NO (per-candle rank-IC) | NO (dead path) | **REDO** (keep feature plumbing only) |
| `train_meta_labeler.py` | OLD | YES (shape) / NO (exec) | NO (dead path) | **REDO** label+split+val; **SALVAGE** architecture; **TOMBSTONE** the file |
| `mover_metalabel.py` | NEW | YES | YES (null) | **SALVAGE as template** (+ shuffled control) |
| `vol_config_ml.py` | NEW | YES | YES (null) | **SALVAGE the rig** (learnability⊥profitability) |
| `signal_picker.py` | NEW | PARTLY (regressor) | YES (wired) | **SALVAGE plumbing; REFRAME head → classifier** |

---

## 3. THE KRONOS VERDICT — borrow what, reject what

Kronos is a decoder-only foundation model over **tokenized OHLCV candles**: a Binary Spherical Quantization tokenizer turns each bar into hierarchical discrete tokens, an AR transformer samples future candle tokens, decoded back to a price path.

**REJECT (high confidence):**
1. **Kronos-as-forecaster and its backtest pattern.** All three of its backtest harnesses are pure **predict-then-threshold** (`pred_return > thresh → buy`, `run_backtest_kronos.py:146-151`). That IS the per-candle IC trap. **It already LOST to our own V1.1 zero-shot: pooled IC h=1 = 0.029 < our 0.067 record** (VERIFIED-elsewhere from the logged eval JSON; its own pre-registered rule landed on "doesn't help"). A 12B-K-line, 45-exchange model scored below our 8GB model on our data. **Re-evaluating it as a forecaster is the over-mining trap — settled null.**
2. **Its accounting.** Naive all-in flip simulator, no per-trade cost inside the trade, no MtM-double-count guard, no fill model — violates our Backtest Simulator + MakerCostModel invariants. Use **none** of it.
3. **The "pretrained foundation model = automatic edge" framing.** The empirical fact is the opposite here.

**BORROW (narrow, conditional, [PROJECTED] until measured):**
1. **The frozen tokenizer/encoder as ONE candidate representation — and ONLY via the A2 H1 warm-start hook**, scored on COMPOUND through the leak-guarded gate, never on IC. **BUT it is dominated by warm-starting from our OWN ShIC-certified V1.1 encoder** (anti-memorization-anchored on our exact data). Kronos's rep is at best a **secondary ablation arm**, and its instance-z-normalization (MEASURED: `kronos.py:544`) discards absolute-level and cross-asset information our chimera f41 + cross-asset features already encode — so its marginal information is plausibly small. **Net: a cheap optional ablation, NOT a roadmap item.** Only run it if a multi-scale-feature probe first shows a representation gap.
2. **Finetune-on-CSV ergonomics (the idea, not the code)** — config-driven, resumable, skip-existing UX. But its data handling (ffill on missing, 15% test split with NO purge gap) violates our invariants. **Borrow the UX, copy none of the data code.**

**Bottom line:** as a forecaster, **DEAD END**. As a feature source, a **low-EV optional ablation arm** behind the same gates. For your actual goal (conditional edge of classic TIs), **Kronos contributes nothing directly** — it has no conditional-edge mechanism, no setup unit, no context-gating.

---

## 4. THE DESIGN — the conditional meta-labeler / setup-selector, mapped to the built apparatus

**What this class IS:** a supervised **meta-labeler** — learn `P(good MOVE | trigger fired, context)`, trade only the high-P subset. **What it is NOT:** a forecaster (IC-banned), predict-then-threshold (the Kronos trap), or offline-RL (the A2 machinery is unnecessary complexity for a binary enter/abstain decision over a fixed trigger). The A2 idea, reduced to its supervised core for the ENTRY decision, IS this meta-labeler.

### 4.1 The UNIT = a triggered classic setup across a multi-candle MOVE — ALREADY BUILT

`src/strat/setup_harness.py` is precisely this primitive; no new sim needed. It scores an arbitrary past-only boolean ENTRY column chased to a declarative `ExitPolicy`, per window, with UNSEEN as the verdict surface, and **never computes IC** (`__contract__`: "IC-INDEPENDENT"). Leak discipline is structural and **VERIFIED this session: `entry_p = opens[i+1]` (next-bar open, "Pattern T banned", L245-246)**; HWM seeded incl. the fill bar (L249); pessimistic stop-before-target intrabar; honest gap-through fills. It already duck-types `CanonicalHarness` → plugs into firewall + battery with zero glue.

**Candidate triggers (each a past-only boolean column):** MA cross / MA stack / price-reclaim-above-MA; N-bar Donchian/close breakout; pullback-to-rising-MA; RSI turn-from-oversold (D51: the TURN is the mechanism, not the depth); vol-contraction-then-expansion (vol is an entry-eligibility/sizing signal per D41/D65, not direction). **Sweep ALL cadences {15m,30m,1h,4h,1d} + alt-bar-types — do NOT default 4h** (standing HARD RULE; cadence materially changes the answer).

### 4.2 The LABEL = the SETUP OUTCOME (triple-barrier / meta-label), produced BY the harness

For each firing, run the move through `setup_harness` with a **pre-registered** `ExitPolicy` and label `y = (net_pnl > 0)` where `net_pnl` already nets the 0.0024 taker round-trip. This is the triple-barrier outcome (TP/SL/time) over the holding window — **NOT the next candle's return.** Pre-register the exit BEFORE reading any result.
**Caveat (D09):** triple-barrier labels can *strip* signal (Val AUC 0.472 in one bull config). Mitigation: report multiple label definitions (TP-hit vs net>0 vs net>cost-band) and verify the meta-labeler's lift is **stable across them** — a label that only works at one barrier set is overfit.

### 4.3 The FEATURES = causal context AT the trigger (this is where the EV lives)

All features from data `≤ trigger close`; per-(asset,feature) z-scores from **TRAIN stats only**:
- **MA geometry:** fast/slow slopes, spread, price-to-MA distance, stack ordering, time-since-cross.
- **Chimera f41:** regime, flow, OI, funding, liq_ratio, micro-imbalance — read via `pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)` (the mandated access path).
- **Regime:** SMA-200 side + past-only vol tercile.
- **Cross-asset:** BTC-relative position + 24h return at the trigger.
- **The WM signal as ONE feature among many:** decode the FROZEN V1.1 forecaster's twohot DISTRIBUTION moments `[E[r], Std[r], P(r<0)]` via `src/wm/forecast_bundle.py` (detached, eval). It enters as a **BELIEF feature, NEVER as the label or a predicted-reward** — using the decoded return as label/reward trips CDAP `no_predicted_return_as_realized_reward` and re-acquires forecaster-GIGO. This is the legitimate WM transfer.

### 4.4 The MODEL = interpretable-first escalation (the linear AND non-linear conditional edge)

Climb only if the lower rung shows held-out lift:
1. **Logistic / elastic-net** — the LINEAR conditional edge; coefficients directly answer "which feature relationships matter."
2. **HistGradientBoostingClassifier** — the NON-LINEAR conditional edge; native NaN, seed-fixed, `early_stopping=False`. The workhorse.
3. **Small attention/NN** — ONLY if GBM wins held-out AND a sequence-of-context hypothesis is specifically motivated. Biggest silent-overfit surface; gate hardest.

The object is `f(context) → P(good setup | trigger, context)`, used as a **binary ENTRY GATE** (trade the top-tau subset), **NOT a position-size multiplier** (the train_meta_labeler defect — it conflates the entry decision with sizing). Operating point = a pre-registered TRAIN percentile. Fit model, z-stats, imputation medians, AND tau **on TRAIN only**.

**Folding in our experience:** model class is NOT the lever — D17/D55 proved logistic/GBM/attention all converge to AUC~0.5 when the information isn't in the bars (and Kronos's 12B model lost to our 8GB one). The chess engine teaches the right transfer: the **champion-gate** (monotonic promotion — a candidate ships only if it beats the incumbent on held-out, the same gate that kept our chess net safe while experimenting) and **model-based-RL discipline** (plan over a learned belief, never consume a forecast as truth). The WM experience teaches the ShIC anti-memorization stance, now realized for the *policy* as the shuffled-market control.

### 4.5 The EXIT = a SEPARATE policy; isolate the ENTRY edge first

Entry and exit are different dimensions. Start with a **robust declarative exit** (fixed-horizon / ATR-trail / TP-SL) held FIXED while proving the entry-gate's conditional edge. The exit-as-skill axis is separately measurable via `src/strat/exit_capture_proxy.py` — but **D61: smart exit-timing was NULL on daily breakouts** (the "beats-null" was a hold-length artifact the fair test caught). A LEARNED exit is a Phase-2 fast-follow, gated by `exit_capture_proxy`'s fair test, **not part of v1.**

---

## 5. THE RUTHLESS VALIDATION GAUNTLET + the right goal/spec

**THE RIGHT GOAL (the cardinal fix vs the prior attempt):** optimize **HELD-OUT COMPOUND on the SETUP-across-the-MOVE stream**, net of taker 0.0024 — **never AUC / accuracy / Brier / IC.** AUC>0.55 survives ONLY as a within-model discrimination DIAGNOSTIC/gate, never as the thing maximized. The meta-labeler is a GATE on a fixed causal trigger — never a signal generator.

**PRE-REGISTRATION (the discipline the prior attempt skipped):** before reading any result, fix the trigger mechanism (by mechanism, not grid result), the context feature set (ALL causal), the exit, the label (net>0), the operating point (tau as a TRAIN quantile), and the **asymmetric loss** (a false LONG costs more than a missed winner — fixed, never tuned on UNSEEN).

**THE GAUNTLET — run in order on the entry-gate's SELECTED trade stream. Each is a KILL gate. All already built.**

1. **OOS AUC / lift floor (>0.55).** Discrimination must exist + a monotone TRAIN decile lift that survives OOS. (mover_metalabel used 0.52 — its *failure* threshold; tighten to 0.55.) Diagnostic only.
2. **Random-entry firewall — PRIMARY (`src/strat/firewall.py`).** Selected-subset compound must beat a cost-matched random-entry null on ALL held-out windows. **Use `regime_matched=True`** (VERIFIED L52-60: draw nulls only from gate-ON bars → isolates *within-trigger selection* from trigger/regime selection) **AND `membership_matched=True`** (VERIFIED L62-72: draw each null from WITHIN the same multi-candle MOVE the real setup fires in → isolates *trigger timing* from *move-selection*). **This is the single most important gate** — it is the direct antidote to "beta in disguise" and "move-selection masquerading as timing," the two failure modes the prior attempt's top feature (regime one-hot) literally exhibited.
3. **Shuffled-market control (`src/agents/_shared/shuffled_market_control.py`).** Re-score the SAME policy on surrogate markets (perm + block — predictability-destroying, marginal-preserving). GENUINE iff edge collapses to ~0 on surrogates (overfit_fraction ≤0.20). This is the ONLY gate that sees an invisible path-memorizer. (phase/iaaft are mechanism diagnostics only — a hard ValueError fires if you put them in the verdict.)
4. **PBO via CSCV (`src/strat/pbo_cscv.py`, ship PBO<0.10) — MANDATORY, not optional.** The meta-labeler searches trigger × feature × threshold × barrier × hold × cadence — the in-sample-best is by construction the most path-fit. PBO answers the orthogonal-to-DSR question: does the SELECTION PROCESS produce OOS under-performers? PBO~0.5 = skill-less selection. **The config search specifically requires this gate.**
5. **Walk-forward + seed robustness (`src/wealth_bot/framework/walk_forward.py`).** 50/20/20/10 + 400-bar purge; **≥8/10 seeds OOS-positive** (the gauntlet floor — tighter than the module's 70% harness default; the canonical incident is single-seed +44% → median −7% at 10-seed audit, the project's worst overfit). Bootstrap p05>0.
6. **Battery Lens A/B/C (`src/strat/battery.py`).** All-4-windows-positive, n_eff floors, jackknife jk2/jk3>0, p05>0, maxDD<30% (20% binding downstream). Jackknife guards concentration (a few big trades carrying the subset).
7. **No-skill control + positive control (`src/strat/positive_control.py`).** A RANDOM and a DUMB (regime-only) meta-labeler must **FAIL** the chain; a known synthetic edge must **SHIP**. Two-sided soundness — a gate that rejects everything is as useless as one that ships everything. (`exit_capture_proxy` already caught a hold-length artifact via exactly this no-skill control.)
8. **Per-regime reporting (never aggregate).** +X% that is +40% bull / −30% bear is a regime BET, not a conditional edge.

UNSEEN touched ONCE. LONG-only, spot, lev=1, taker 0.0024 (not optimistic maker fills — D43's p_fill 0.21-0.40 wall + D60's 1h cost-wall can kill the subset even with genuine AUC).

---

## 6. THE HONEST CEILING — fresh tractable question, but the most overfit-prone test in quant

**This is a genuinely FRESH question, NOT the IC ceiling** (§1). Conditional ≠ unconditional; the dead-list does not refute it. **But it has its own ceiling, and the prior is brutal:**

> **This exact object was already built, carefully, on a PROVEN trigger, and returned a null.** `src/mining/mover_metalabel.py` (D72) is a correctly-built trigger-time meta-labeler on a 1m mover with *confirmed* +5.3-5.7% median meat — 16 causal features, HGB+logit, 7054 TRAIN events, OOS-once on 3369 — and got **OOS AUC 0.521/0.510: NO held-out continuation information.** The selected-third was indistinguishable from the rejected-third. D73 is a 5th independent kill of factor→config mapping. D45/D62/D63 HARD-null bar-level entry-timing at daily/4h; D55 HARD-nulls direction across 100+ assets × 5 TF.

**Implication — the highest-EV design variable is NOT the model class, it is the FEATURE SET.** Logistic/GBM/attention all converge to AUC~0.5 when the information isn't in the bars (D17/D55 proved this; Kronos confirmed it). The open thread (D72 spec) is a trigger-time signal with **OOS AUC > ~0.58 from EXTERNAL/leading data** (Coinglass liq-heatmap proximity, on-chain netflow, news/social — none in-tree yet) and possibly **RESOLUTION** (sub-15m, a DATA decision orthogonal to the agent).

**Therefore build the agent so the feature set is a swappable input, and make the FIRST deliverable a VERDICT, not a bot:** for a given (trigger, cadence, feature-set), does the selected subset beat the unconditional trigger AND the regime+membership-matched firewall on UNSEEN with AUC>0.55? **A clean "NO with internal features at 4h" is a high-value, cheap re-confirmation that the constraint is INFORMATION, not methodology — and that null is the product, not a failure** (report-first mandate). A "YES" with a new external feature is the first genuine alpha.

**The failure modes, named (and where each is caught):** multiple-comparisons over the config grid → PBO<0.10. Label leakage via barrier/non-causal-features/whole-series-norm → causal pre-registration + relative leak probe. Regime-as-only-signal (beta) → regime_matched firewall + per-regime reporting. Move-selection-as-timing → membership_matched firewall. Concentration → jackknife. Invisible path-memorization → shuffled-market control. Seed lottery → ≥8/10 seeds. AUC-as-objective → objective is compound. Coin-flip passing → no-skill control must fail.

**Treat "the conditional meta-labeler finds edge" as [PROJECTED], never measured, until UNSEEN says otherwise. The conditional reframe is worth testing precisely BECAUSE it is a genuinely different hypothesis from the IC null — but it is the most overfit-prone test in the book, so the gauntlet is the whole point.**

---

## 7. THE CONCRETE FIRST BUILD — smallest end-to-end slice that honestly answers "is there a conditional edge?"

**Deliverable: a VERDICT artifact, not a bot.** Effort: ~3-5 focused days (reuses the entire sim + gate apparatus; supervised, not RL).

**New file — `src/strat/setup_meta_labeler.py`** (the generalization of `mover_metalabel.py` from 1m-movers to arbitrary classic triggers). Carries `__contract__` (`ml_as_metalabeler_only` / `train_only_fit` / `causal_features` / `unseen_untouched`), a pre-registration block, git+seed lineage.

**Pre-register (fix before any result):**
- **Trigger:** ONE family — `EMA(fast) crosses EMA(slow)`. (First run `src/oracle/run.py` to pick the trigger family with the highest hindsight-oracle ceiling — condition on a trigger that *has* meat, per the D67/D72 lesson: meat must EXIST before discrimination matters.)
- **Cadences:** sweep {15m, 30m, 1h, 4h, 1d} — do NOT default 4h.
- **Universe:** u10 (flag survivorship — current-membership inflates absolute-level features).
- **Exit:** ONE fixed `ExitPolicy` (ATR-trail + time-stop), held constant.
- **Label:** `y = (net_pnl > 0)` from `setup_harness`, taker 0.0024 — reported also at net>cost-band and TP-hit (D09 stability check).
- **Features:** the §4.3 set, ALL causal ≤ trigger close, z-scored TRAIN-only.
- **Model:** logistic → HistGBM escalation. tau = 67th TRAIN pctile.
- **Asymmetric loss:** false-LONG penalized > missed-winner (ratio pre-registered).

**Pipeline:** ChimeraLoader → enumerate trigger firings → harness produces label + causal features per firing → split 50/20/20/10 + 400-bar purge → fit model/z-stats/imputation/tau on TRAIN only → apply tau → emit the selected-subset trade stream.

**Run the FULL gauntlet (§5) on the selected stream:** AUC>0.55 floor → regime+membership-matched firewall → shuffled-market control → PBO<0.10 → ≥8/10-seed walk-forward → battery A/B/C → no-skill + positive control → per-regime report. **UNSEEN touched once.**

**VERDICT (the deliverable):** for each (cadence), report — does the selected subset beat (a) the unconditional trigger, (b) the regime+membership-matched random-entry firewall, on UNSEEN, with AUC>0.55, PBO<0.10, ≥8/10 seeds? 
- **If NO (the honest prior):** the constraint is INFORMATION — internal features at this cadence carry no conditional continuation edge. High-value, cheap, re-confirms D72 on a new trigger; the next move is a **DATA decision** (external/leading features OR sub-15m resolution), not a modeling one.
- **If YES:** the first measured conditional edge — promote through the champion-gate, then (and only then) build the bot.

**Tombstone `src/training/train_meta_labeler.py` + `train_rank_model.py`** (dead `src/strategy/` import path + wrong objective) in the same change; port their one good idea (meta-labeling) into the new template.

---

**One-line bottom line:** The reframe is correct and the dead-list does not refute it — but build the conditional meta-labeler **feature-set-first, verdict-first, on the new-era discipline** (`mover_metalabel`'s gate stack + `vol_config_ml`'s margin-trap instrument + `signal_picker`'s WF/seed plumbing + `setup_harness`'s leak-guarded unit), judge on **held-out compound** through the full firewall/PBO/shuffle/seed/jackknife gauntlet, and expect an honest INTERNAL-feature null at 4h to be the *first* high-value result. Success is [PROJECTED] until UNSEEN says otherwise; the gauntlet is the whole point.

---

# APPENDIX -- the 4 lenses

## LENS: AUDIT THE PRIOR IN-PROJECT ML DECISION ATTEMPT — what was the goal/features/label/model/validation of each, why the user is right that they were not thorough/careful/right-goal, and what to SALVAGE vs REDO.

**Summary:** Five prior ML attempts split into two eras. The TWO OLD ones (src/training/train_meta_labeler.py, train_rank_model.py) are exactly what the user means by "not done right": train_rank_model predicts the per-candle next-day return RANK (the banned IC lens, wrong unit), validates on top-1 hit-rate with ZERO cost / ZERO seeds / ZERO compound / no shuffled-market control; train_meta_labeler has the right SHAPE (triple-barrier meta-label on a fired signal) but a broken WIN label (ret>cost, a coin-flip threshold), a non-chronological "time" split, AUC-only validation, and both import a src/strategy/ path that was deleted in the 2026-06-04 reset — they CANNOT RUN today and never shipped an artifact. The THREE NEW ones (mover_metalabel.py, vol_config_ml.py, signal_picker.py) are the methodologically-correct template the user actually wants — meta-labeler-on-a-proven-trigger, setup-level capture target, causal pre-registered gates, train-only fit, OOS-once — and they are HONESTLY NULL (mover_metalabel HGB train AUC 0.9998 vs OOS 0.521 → gate alive=False; vol_config_ml → margin-trap / not-learnable on real data). SALVAGE the new-era discipline (it is the A2 strategy-agent spec in miniature); REDO the old-era goal/label/validation entirely; do NOT resurrect the dead src/strategy/ files.

## The two eras (provenance grounds the user's complaint)

`git log` dates the OLD pair to the V7/"Items 1-8" framework (commits 622674a, fa2a7a0) — pre-reset. The NEW trio postdate the 2026-06-04 reset (mover_metalabel.py, vol_config_ml.py modified 2026-06-10; signal_picker.py 2026-05-25). The reset archived `src/strategy/` to `archive/restart_2026_06_04/`. This split is the whole story: the user's "not thorough/careful/right-goal" complaint lands squarely on the OLD pair; the NEW trio is already the corrected methodology (just empirically null).

---

## A. `src/training/train_rank_model.py` — THE CLEAREST "WRONG GOAL" (the IC trap, literally)

- **GOAL:** cross-sectional LambdaRank — "among today's 10-24 assets, which will outperform tomorrow" (docstring L8-11).
- **FEATURES:** 34 chimera features aggregated to **daily mean** per asset (`load_asset_daily_panel`, L73-111). Mean-over-day destroys the intrabar structure; one value per feature per day.
- **LABEL (the core defect):** `fwd_ret[:-1] = np.diff(daily_close)/daily_close[:-1]` (L109-110) → **next-DAY per-asset return**, then `ranks = fwds.argsort().argsort()` (L151) = the within-day rank of that 1-bar return. **This is per-candle prediction — the exact IC lens MEMORY.md bans.** The unit is one daily candle, not a setup-across-a-move. There is no entry, no hold, no exit, no triple barrier.
- **MODEL:** LightGBM LambdaRank, NDCG objective (L223-235).
- **VALIDATION (insufficient on every axis):** time_split 70/15/15 (L165) — correctly chronological, the one thing it got right. But the "test" metric is **top-1 hit rate** (L250-265): fraction of days where the predicted top asset = actual top asset. **No cost model (SPOT_COST never imported), no compound return, no per-seed robustness, no walk-forward refit (single static fit), no shuffled-market control, no block-bootstrap, no jackknife.** Top-1 hit-rate is an accuracy proxy on a per-candle rank — it answers a question the project has banned and never converts to held-out wealth.
- **VERDICT:** REDO entirely. Wrong unit (candle), wrong objective (rank-IC), wrong validation (accuracy not compound). The only salvage is the chimera-feature plumbing (`FEATURE_NAMES` L57-70, the polars daily-panel loader pattern).

---

## B. `src/training/train_meta_labeler.py` — RIGHT SHAPE, BROKEN EXECUTION (this is the prototype of what the user wants, done carelessly)

- **GOAL (correct in spirit):** López-de-Prado meta-labeling — engine fires, classifier estimates P(win|context), size = signal × P(win) (docstring L9-19). This IS the conditional-edge / meta-labeler-on-a-trigger idea the user is now asking for.
- **FEATURES:** per-bar context — vol_30, ret_5d, ret_30d, signed_vol, plus norm_funding/oi/whale/efficiency if present (`extract_context` L52-80). Thin and hand-picked; no chimera breadth.
- **LABEL (broken):** `win = 1 if ret_trade > 2*cost/2 else 0` (L109). `2*cost/2` == `cost` — so the win threshold is barely-above-breakeven, making the label a near-coin-flip (win-rate ≈ 0.5 by construction). The OUTCOME comes from a triple-barrier exit (L99-119) — the unit IS a setup-across-a-move, which is RIGHT. But the label collapses a multi-candle PnL into a binary at a trivial threshold, throwing away the magnitude that actually matters for wealth.
- **MODEL:** LightGBM binary classifier (L218-230).
- **VALIDATION (three concrete defects):**
  1. `from sklearn.model_selection import train_test_split` imported (L189) but **never used** — dead import, a tell of unfinished work.
  2. The split is `split_idx = int(0.70*n); X[:split_idx]` (L206-207) over `records` — but records are appended in **engine-outer, asset-inner order** (`build_labels` L89-91 loops `for eng: for asset:`), so "first 70% of records" is **NOT chronological** — it is the first ~70% of (engine,asset) pairs. Trades from late calendar dates for early engines sit in "train"; trades from early dates for late engines sit in "val". This is a silent look-ahead/leakage path masquerading as a time split.
  3. Validation = **AUC + Brier only** (L234-239). No seeds, no compound, no held-out replay of the sizing rule, no shuffled-market control, no walk-forward. AUC 0.5x on a coin-flip label tells you nothing about wealth.
- **DEAD ON ARRIVAL:** imports `from universe import ...`, `from cost_model import SPOT_COST`, `from triple_barrier_exit import ...` via `sys.path.insert(.../src/strategy)` (L38, 41-46). I confirmed `src/strategy/` **does not exist** (deleted at reset) and `triple_barrier_exit.py` is **gone from the tree**. `models/meta_labeler/` holds only unrelated `v8_catboost.pkl` — **no meta_labeler_u*.txt artifact** was ever persisted. So this file cannot run and left no validated output.
- **VERDICT:** REDO the label (use net-of-cost setup return magnitude or a meaningful barrier-touch label, not ret>cost), REDO the split (true chronological by entry_t, which IS recorded at L116), REDO validation (compound + seeds + shuffled control). SALVAGE the meta-labeler ARCHITECTURE (fire→P(win|context)→size) — it is correct and is exactly the conditional-edge framing.

---

## C. `src/mining/mover_metalabel.py` — THE NEW-ERA TEMPLATE, DONE RIGHT, HONESTLY NULL (salvage wholesale)

- **GOAL:** meta-labeler on a PROVEN, MECHANISM-CHOSEN trigger (+1.5% from day-open at 1m), gating not generating (`__contract__` "ml_as_metalabeler_only", L67-78). Explicitly cites D16/D17 to forbid signal-generation. This is the framework-endorsed ML use.
- **FEATURES:** 16, ALL causal-at-trigger-close (L90-92, docstring L24-29), per-(asset,feature) z-scored from **TRAIN stats only** (L294-303).
- **LABEL (correct):** `y = (trail3_gross - 24bps) > 0` (L283) — net-of-cost outcome of a multi-candle trailing-exit ride. Setup-level, cost-aware. This is the label train_meta_labeler should have had.
- **MODEL:** HistGradientBoosting (primary, seed 7) + LogisticRegression (sanity) (L305-316).
- **VALIDATION (this is the gold standard the old pair lacked):** PRE-REGISTERED operating point (tau=67th pctile of TRAIN p, L323); calendar-time TRAIN/OOS/UNSEEN split (L80-101); four required OOS ALIVE gates — portfolio net/yr>0, breadth≥6, drop-top-3 jackknife>0, OOS AUC>0.52 (L348-355); UNSEEN sealed; git SHA + seed + lineage in JSON (L357-370); fixed-span annualization to kill the inflation bug.
- **EMPIRICAL RESULT (I ran the artifact):** HGB **train AUC 0.99984 vs OOS AUC 0.5210** — textbook overfit, the gate caught it. `oos_gates.alive = False` (net_pos=False, breadth=False, jk3=False). OOS_selected n=381, net24 = -0.29%/event, -0.075%/yr. **Honest NULL — the discipline worked.**
- **VERDICT:** SALVAGE as the canonical template for the strategy-decision agent. The one weakness to note: HGB's 0.9998 train AUC on 16 features = it is memorizing; a stronger version needs heavier regularization / fewer leaves and a shuffled-market control (it has jackknife + breadth but not the policy-overfit shuffle from src/agents/_shared/shuffled_market_control.py — which postdates none of this and SHOULD be wired in).

---

## D. `src/strat/vol_config_ml.py` — SOPHISTICATED, TWO-SIDED SELFTEST, HONESTLY NULL (salvage the rig)

- **GOAL:** can a SIMPLE nonlinear model map pre-move VOLATILITY → the winning config-CLASS (formulation×MA-type), then replay it CAUSALLY OOS (docstring L1-8).
- **LABEL:** the config-class that MAXIMIZED capture on each move (L298-303) — setup-level, capture-based. Right unit.
- **THE STANDOUT FEATURE — it separates LEARNABILITY from PROFITABILITY (L34-42, verdict_line L595-623).** It explicitly names and tests for "the margin trap": vol can predict the class above base-rate (1) while acting on it fails to beat fixed (2). This is precisely the self-deception that sank older attempts (winning-DNA correlation ≠ capture improvement). It also ships a TWO-SIDED selftest (positive control where vol DETERMINES config must come back YES; negative control must come back NULL — L767-820, 916-964).
- **VALIDATION:** causal pre-move features (strictly before ev.start, L191-232 with an explicit look-ahead assert at L508), TRAIN-only fit, OOS-once, compares ml vs fixed vs rolled vs ceiling vs random (L539-570).
- **EMPIRICAL RESULT (I ran the artifact):** full-target → "LEARNABLE-BUT-UNPROFITABLE (MARGIN TRAP)"; formulation-target → "GENUINELY-NOT-LEARNABLE (NULL)". **Honest NULL on real BTC/ETH/SOL.**
- **TWO WEAKNESSES to flag (not fatal, but the user's "carefully" bar):** (1) single TRAIN/TEST split, not walk-forward / not multi-seed (the RF is seeded internally but there is no seed-robustness sweep across data splits); (2) one asset-set, no UNSEEN hold-back beyond the single OOS — so a chance positive here would not yet clear the project's 10-seed + block-bootstrap bar. The rig is excellent; the statistical coverage is one notch below ship-grade.
- **VERDICT:** SALVAGE the learnability-vs-profitability split and the two-sided selftest — these are the exact instruments that stop the IC-trap self-deception. REDO nothing in the method; only extend coverage (walk-forward, seeds, UNSEEN) IF a future variant shows a non-null signal worth promoting.

---

## E. `src/wealth_bot/framework/signal_picker.py` — WIRED, CAUSAL, BUT A REGRESSOR (the IC-trap risk lives here)

- **GOAL:** per-strategy LGBM forward-return REGRESSOR; pick the strategy with the highest predicted fwd_ret among those firing (`train_picker`, L59-173). It is wired into the live bot (`src/wealth_bot/bot/runner.py`).
- **LABEL:** `fwd_ret` — forward return over fwd_bars (continuous). **This is the one new-era file that still leans on per-bar return prediction** (predict fwd_ret > threshold → fire, L165). It is a milder IC-trap than train_rank_model because the unit is a held position over fwd_bars and it only chooses AMONG already-firing strategies (a selector, not a generator) — but the objective is still "predict the return," which is the lens the user is moving away from.
- **VALIDATION DISCIPLINE (good):** genuine walk-forward (train window ends at cur, L105-107), non-overlapping execution (skip fwd_bars after fire, L170), explicit-seed requirement with decorrelated LGBM sub-streams (L101-103), per-refit feature-importance logging (L136-143). The provenance (+49.8% UNSEEN on PEPE/EMA, 10-seed audit, docstring L4-6) is the strongest of any file — but that was a (TI,ASSET) gold-standard dossier now ARCHIVED per MEMORY.md, so the number is not current.
- **VERDICT:** SALVAGE the walk-forward + non-overlapping + seed-decorrelation machinery (it is production-grade). REFRAME the objective: a meta-labeler (P(setup pays | context), classifier on a setup-level net-of-cost label) is the right head, not a fwd_ret regressor — i.e. converge signal_picker's plumbing onto mover_metalabel's framing.

---

## Bottom line: SALVAGE vs REDO

| File | Era | Goal right? | Label | Validation | Runs today? | Action |
|---|---|---|---|---|---|---|
| train_rank_model.py | OLD | NO (per-candle rank-IC) | next-day return rank | top-1 hit-rate, 0 cost/seed/compound | NO (dead src/strategy) | **REDO** (keep only feature plumbing) |
| train_meta_labeler.py | OLD | YES (meta-label) | broken (ret>cost coinflip) | AUC only, non-chrono split, unused split import | NO (dead src/strategy) | **REDO label+split+val; SALVAGE architecture** |
| mover_metalabel.py | NEW | YES | net-of-cost ride (setup) | pre-registered 4 gates, UNSEEN sealed, lineage | YES (null) | **SALVAGE as template** (add shuffled control) |
| vol_config_ml.py | NEW | YES | best config-class (capture) | learnability⊥profitability, 2-sided selftest | YES (null) | **SALVAGE the rig** (extend to WF+seeds) |
| signal_picker.py | NEW | PARTLY (fwd_ret regressor) | forward return (mild IC) | WF + non-overlap + seed-decorrelated | YES (wired) | **SALVAGE plumbing; REFRAME head to classifier** |

**The user is correct, and precisely so:** the OLD pair is the careless work — wrong unit (candle), wrong/trivial labels, accuracy-not-compound validation, no robustness battery, and now literally un-runnable. The NEW trio already embodies the right methodology (setup-unit, cost-net labels, causal pre-registered gates, learnability-vs-profitability separation, sealed UNSEEN) — they just return honest NULLs on daily/4h/1m long-only crypto, which is consistent with the project's burned history. The strategy-decision agent (A2-on-technical-signals) should be BUILT ON the new-era discipline (mover_metalabel's gate stack + vol_config_ml's margin-trap instrument + signal_picker's WF/seed plumbing), with the conditional-edge framing made explicit: classic trigger has no unconditional edge → test for edge CONDITIONAL on chimera/regime/context, label at the SETUP level net-of-cost, judge on held-out COMPOUND with the full robustness battery (seeds + shuffled-market control + block-bootstrap + jackknife + UNSEEN). Do NOT resurrect train_meta_labeler.py / train_rank_model.py — port their one good idea (meta-labeling) into the new template and delete the dead src/strategy import path.

**Failure modes:** (1) The two OLD files import a deleted src/strategy/ path — they cannot run; any claim about their "results" is unverifiable because no artifact was ever persisted (models/meta_labeler/ has only unrelated v8_catboost files, models/rank_v1/ does not exist). I tagged their defects from CODE READING, not from re-running. (2) The NEW two files' NULL verdicts I report (mover_metalabel gate alive=False, AUC train 0.9998/OOS 0.521; vol_config_ml margin-trap/null) are MEASURED — I loaded and parsed the committed run JSONs (runs/mining/mover_metalabel_u10_20260610_195259.json, runs/strat/vol_config_ml.json). (3) signal_picker's +49.8% UNSEEN provenance is from an ARCHIVED dossier (MEMORY.md says all dossiers archived 2026-06-04) — REPORTED, not currently reproducible; do not cite the number as live. (4) The non-chronological-split claim for train_meta_labeler is a code-structure inference (records appended engine-outer/asset-inner, then sliced 70%) — high-confidence from the loop order at L89-91 + slice at L206, but I did not instrument the actual record order at runtime since the file can't run. (5) I did NOT audit train_meta_labeler's TripleBarrier outcome logic itself (the dependency is deleted) — the label-threshold defect is independent of it.

**Open questions:** (1) Were the OLD files ever actually run pre-reset and what did they produce? The run logs/CSVs (logs/meta_labeler_trades_u*.csv per L170) would confirm whether they ever executed — worth a glob of the archive before fully writing them off. (2) Should the new strategy-agent label be binary (meta-label, mover_metalabel-style) or magnitude-aware (expected net capture)? mover_metalabel's binary >0 label discards magnitude; for WEALTH optimization a calibrated expected-net-capture head may be better — open design question. (3) mover_metalabel's HGB train AUC 0.9998 means it memorizes 16 features on the TRAIN events — is the OOS null because there's no conditional edge, or because the model is too flexible and a regularized linear/shallow model would generalize? The logit OOS AUC 0.5096 suggests truly no linear conditional edge either, but a middle-complexity model was not swept. (4) None of the new tools wire in src/agents/_shared/shuffled_market_control.py (the policy-overfit gate) — adding it is the missing robustness layer before any positive result is believed. (5) vol_config_ml is single-split / single-seed across data — does a walk-forward + 10-seed version change the null? Cheap to test, would harden the verdict.

## LENS: THE KRONOS / EXTERNAL LESSON — what a candle-TSFM (tokenized-OHLCV autoregressive + finetune-on-CSV + predict-then-threshold backtest) teaches our LEARNED conditional-edge strategy-decision agent: what to REJECT (the predict-then-threshold = IC trap) and what, if anything, to BORROW (tokenizer-as-representation, finetune ergonomics).

**Summary:** Kronos is a decoder-only foundation model over TOKENIZED OHLCV candles: a BSQ tokenizer quantizes each bar into hierarchical discrete tokens, an autoregressive transformer samples FUTURE candle tokens, and they get decoded back to an OHLCV price path. Every one of its three backtest harnesses is pure predict-then-threshold (`pred_return > thresh -> buy`) — which IS the per-candle IC trap the user is explicitly moving away from, and exactly the framing that already LOST to our own V1.1 zero-shot (verified pooled IC 0.029 < 0.067, `eval_kronos.py`). REJECT: Kronos-as-forecaster and its backtest pattern wholesale; do not re-evaluate it as a standalone predictor (over-mining a settled null). BORROW (narrow, conditional, [PROJECTED]): ONLY the frozen TOKENIZER/encoder as ONE candidate representation feeding the decision agent's H1 warm-start hook — and even that is dominated by warm-starting from our OWN ShIC-certified V1.1 encoder, so Kronos's rep is at best a cheap ablation arm, not a path. Its CSV-finetune ergonomics are clean but we already have a hardened harness; copy the UX idea, not the code. Honest bottom line: as a FEATURE SOURCE it is a low-EV, optional ablation; as a forecaster it is a DEAD END for our purpose. Do not overclaim it as a representation win — that is unmeasured.

## What Kronos actually is (code-grounded)

**Two-stage tokenized-candle TSFM** (`external/Kronos/model/`):
1. **Tokenizer** (`module.py:225` `BSQuantizer`, `kronos.py:13` `KronosTokenizer`): encodes a window of 6-dim OHLCVA bars through transformer encoder blocks (`module.py:465` `TransformerBlock`, RMSNorm + RoPE), then **Binary Spherical Quantization** (`module.py:39`, paper 2406.07548) into HIERARCHICAL discrete tokens — `s1_bits` coarse + `s2_bits` fine (`HierarchicalEmbedding`, `module.py:400`). So each candle becomes a (pre, post) token pair. This is a learned VQ-style codebook for candlesticks.
2. **Predictor** (`kronos.py:180` `Kronos`): decoder-only AR transformer over those tokens with a `DualHead` (`module.py:486`) predicting s1 then s2 conditioned on s1 (`DependencyAwareLayer` cross-attn, `module.py:446`). `auto_regressive_inference` (`kronos.py:389`) autoregressively SAMPLES future tokens (temperature/top-p, `sample_count` paths averaged) and `tokenizer.decode`s them back to an OHLCV path (`KronosPredictor.predict`, `kronos.py:519`).

**Normalization is per-window instance z-score** (`kronos.py:544` `x_mean,x_std = mean/std(x,axis=0)`; same in `qlib_test.py:85` and the CSV dataset `finetune_base_model.py:125`). The model sees only WITHIN-WINDOW shape; absolute price level and cross-asset scale are destroyed and re-applied at the end. It is a univariate, single-series, shape-of-the-next-candles model.

## The approach Kronos prescribes for trading — and why it is the trap

All THREE backtest paths in the repo are predict-then-threshold:
- `examples/run_backtest_kronos.py:146-151`: `combined['pred_return'] = predicted.pct_change(); signal = +1 if pred_return > threshold (0.02) else -1 if < -threshold` — literally the IC/per-candle rule.
- `finetune/qlib_test.py:277-282`: signal = `preds[:,:,close]` minus `last_day_close` (last/mean/max/min variants) fed to `TopkDropoutStrategy` — rank-by-predicted-change, the cross-sectional version of the same trap.
- README §"Step 4"/"From Demo to Production" (lines 289, 304): "generate prediction signals (e.g., forecasted price change)" then "raw signals … fed into a portfolio optimization model." Even the authors flag (line 221) it is "a simplified example and not a production-ready quantitative trading system."

This is precisely the lens MEMORY.md bans: it asks "what is next bar's return?" and trades the point estimate. It is per-candle (even if pred_len>1, the decision collapses to a thresholded scalar). It has no notion of a SETUP, no conditional-edge learning, no multi-candle MOVE as the unit, no cost-matched firewall. It is the Kronos-shaped instance of the exact thing the user is reframing AWAY from.

**And we already paid for the null.** `src/frontier_ml/kronos_baseline/eval_kronos.py` ran Kronos-small zero-shot on chimera_legacy u10 OOS, predicted next-bar close → return → Spearman IC vs `target_return_1`. Result (`logs/frontier_ml/kronos_baseline/kronos_small_zero_shot_*.json`, VERIFIED in `TSFM_WAVELET_WM_SURVEY_2026_06_09.md:127`): **pooled IC h=1 = 0.029**, below our V1.1 record 0.067 and far below the 0.10 headline. The pre-registered rule (`eval_kronos.py:253`, ≥0.060→pivot/finetune; <0.030→doesn't help) landed on **"doesn't help."** A 12B-K-line, 45-exchange foundation model scored below our 8GB-trained model, zero-shot, on our data.

## What to REJECT (high confidence)
1. **Kronos-as-forecaster + its predict-then-threshold backtest.** This is the IC trap with a fancy tokenizer. It is settled-null for our objective. Re-running it as a standalone predictor (zero-shot OR finetune-for-IC) is the over-mining trap (`TSFM_WAVELET_WM_SURVEY` rec #3: "LOW EV / DO NOT").
2. **Its accounting.** `run_backtest_kronos.py:158-228` is a naive all-in buy/hold-flip simulator with no per-trade cost charged inside the trade, no MtM-double-count guard, no fill model. It violates our Backtest Simulator Invariants and MakerCostModel invariants. Use NONE of it — our `wealth_bot/harness.py` (taker 0.0024 inside the trade, MtM-no-double-count) is the only sim A2 trains against (`A2_BUILD_SPEC.md` B.5/C.5).
3. **"Pretrained foundation model = automatic edge" framing.** The empirical fact is the opposite here.

## What to BORROW (narrow, conditional, all [PROJECTED] until measured)
1. **The TOKENIZER / encoder as ONE candidate representation — and only via A2's H1 warm-start hook.** The legitimately interesting part of Kronos is the BSQ candle tokenizer (`module.py:225`): a self-supervised, reconstruction-trained discrete representation of bars. The A2 spec already defines the exact, GIGO-safe slot for an external representation: `A2_BUILD_SPEC.md:151-164` (H1) + `C.1:306-313` — "import the genuinely-learnt REPRESENTATION … WITHOUT consuming its PREDICTION." A frozen Kronos encoder producing per-window embeddings → `tsfm_*` feature columns (PCA'd) into the setup/oracle/capture-rate apparatus, scored on the COMPOUND objective and the leak-guarded gate, NOT on IC, is a defensible experiment. **BUT:** (a) H1's PRIMARY mitigation is to warm-start from our OWN ShIC>0-certified forecaster encoder (V1.1/V12/…), which is strictly preferable because it was anti-memorization-anchored on our exact data; Kronos's rep is at best a SECONDARY ablation arm ("does an external candle-rep beat our own encoder?"); (b) the zero-shot 0.029 is weak evidence its rep carries structure our features miss; (c) its instance-z-norm throws away the absolute-level and cross-asset information our chimera f41 + cross-asset features already encode, so the marginal information it adds is plausibly small. Net: **a cheap ablation, not a roadmap item.** Only run it if/after a multi-scale-feature test shows there is structure a pretrained encoder might capture better (`TSFM_WAVELET_WM_SURVEY` rec #2 logic).
2. **Finetune-on-CSV ERGONOMICS (the idea, not the code).** `finetune_csv/` is clean: one YAML (data path, lookback/predict window), `train_sequential.py` with `--skip-existing/--skip-tokenizer/--skip-basemodel`, time-ordered 70/15/15 split (`finetune_base_model.py:75`). The reusable LESSON is config-driven, resumable, skip-existing training UX — which our framework/pipeline mostly has. Note the WARNINGS in their own code: ffill on missing values (`:69`) and a 15% test split with NO purge gap — both violate our invariants (400-bar purge, no silent fill). So borrow the UX pattern, copy none of the data handling.
3. **Architecture micro-lessons (already independently present):** RoPE + RMSNorm + SwiGLU FFN decoder (`module.py`), hierarchical coarse/fine token factorization, sample-and-average for probabilistic paths. Nothing here we don't already have in `src/wm/`; not a reason to adopt Kronos.

## The honest verdict on "feature source vs dead end"
- **As a forecaster for our decision problem: DEAD END.** It is the IC trap incarnate and it empirically lost to V1.1. Settled.
- **As a feature/representation source: a LOW-EV OPTIONAL ABLATION**, structurally slotted into A2-H1, dominated by warm-starting from our own ShIC-certified encoders. Worth a half-day ablation IF the A2/oracle apparatus first shows a representation gap; not worth a build otherwise. Treat any "Kronos rep helps" claim as [PROJECTED], never measured — the only Kronos number we have measured (0.029) points the other way.
- **For the user's actual goal** (learn the CONDITIONAL edge of classic TIs: trigger × chimera/regime context → enter setup → let it play out → exit policy), Kronos contributes NOTHING directly: it has no conditional-edge mechanism, no setup unit, no context-gating. The right substrate is the A2 DT-inner/evolutionary-outer policy over our f41 + TI features against the hardened harness + the shuffled-market/firewall/PBO gate stack — Kronos sits, at most, as one frozen-encoder ablation arm inside that, behind the same gates.

**Failure modes:** FAILURE MODES NAMED (and how this answer guards them): (1) IC-anchoring — Kronos's entire trading framing is per-candle predict-then-threshold; rejecting it is the whole point. (2) Over-mining a settled null — the 0.029<0.067 result is pre-registered and VERIFIED; re-evaluating Kronos as a forecaster is explicitly the over-mining trap; I do NOT recommend it. (3) Foundation-model halo / overclaiming a representation win — I tag the tokenizer-as-feature path [PROJECTED], note the zero-shot weak evidence, note instance-z-norm discards level/cross-asset info, and rank it BELOW our own ShIC-certified encoder. (4) Look-ahead/leak import — Kronos's own pipeline ffills and uses a no-purge 15% test split and a cost-free naive sim; I flag all three as violations of our invariants (400-bar purge, no silent fill, cost-inside-trade) and say borrow UX not code. (5) Single-seed/concentration — not directly raised by Kronos, but any feature ablation must run through battery/PBO/N-seed/firewall (A2 B.5), which I route it to. CLAIM TAGS: Kronos arch/backtest mechanics = MEASURED (read from external/Kronos code, file:line cited). Kronos 0.029 vs V1.1 0.067 = VERIFIED-elsewhere (read from TSFM survey + eval_kronos.py decision rule; I did not re-run the GPU eval this session — REPORTED from the logged JSON the survey cites). "Tokenizer rep might add info" = PROJECTED. "Instance-z-norm discards cross-asset/level info" = MEASURED (kronos.py:544 + qlib_test.py:85).

**Open questions:** OPEN QUESTIONS / decision support: (1) Is there ANY structure a pretrained candle-encoder captures that our f41+cross-asset features miss? This is the only thing that would make the Kronos-rep ablation worth running — and it should be answered by a cheaper multi-scale-feature probe on the oracle capture-rate FIRST (per TSFM survey rec #2), not by spinning up Kronos. (2) Does warm-starting A2's encoder from our OWN V1.1 (the H1 PRIMARY) materially beat from-scratch? That ablation is higher-EV and must run before any external-encoder arm. (3) Would a Kronos FINETUNE (not zero-shot) on crypto change the rep quality enough to matter? Unknown, but gated behind (1)+(2) — and finetuning-for-forecasting re-enters the IC trap, so a finetune would have to target representation quality measured on compound, which the repo gives no harness for (we'd build it). NOT RECOMMENDED to pursue (3) until (1)+(2) land positive. (4) The user's core build is A2 (DT-inner/evolutionary-outer conditional-edge policy); Kronos's only honest role there is one optional frozen-encoder ablation arm — confirm with the user that the A2 substrate, not a TSFM, is where effort goes.

## LENS: THE RIGHT DESIGN for the strategy-decision agent: the CONDITIONAL meta-labeler / setup-selector class -- learn f(context at a triggered classic technical setup) -> P(this multi-candle MOVE pays net cost), reverse-engineered from the hindsight setup-selector ORACLE, mapped concretely onto the already-built apparatus (setup_harness, oracle, firewall, battery, pbo_cscv, shuffled_market_control). This is the A2 idea (raw-data self-evolving) made concrete on TECHNICAL-SIGNAL features and reduced to its supervised core: it is NOT offline-RL, it is the meta-labeling layer the dead-list says is the ONE surviving ML use (D16). The deliverable is a VERDICT first, a bot only if the gate passes.

**Summary:** Build the strategy-decision agent as a CONDITIONAL meta-labeler on classic technical triggers, NOT a forecaster and NOT an RL policy. The architecture is a clean 6-component pipeline: (1) UNIT = a triggered classic setup (MA cross/stack, breakout, pullback-to-MA, RSI-turn) chased to a policy exit -- the setup_harness.py primitive already implements this exactly (next-bar-open fills, intrabar leak guards, duck-types firewall/battery). (2) LABEL = the triple-barrier/meta-label OUTCOME of entering at the trigger (did the MOVE pay net 24bps taker over the hold), produced BY the harness, never the next candle's return. (3) FEATURES = causal context AT the trigger (MA configs/slopes, chimera f41, regime, vol, cross-asset, and the V1.1 WM signal as ONE feature among many). (4) MODEL = interpretable-first escalation logistic/elastic-net -> GBM (HistGBM, native NaN) -> small attention only if the cheaper rungs win; the object learned is P(good setup | trigger, context), the CONDITIONAL edge the unconditional trigger lacks. (5) EXIT = a SEPARATE, robust-first policy (fixed-horizon/ATR-trail/TP), entry-edge isolated first. (6) TRANSFER = WM-as-feature (frozen V1.1 belief moments via forecast_bundle, never its prediction-as-reward), the chess champion-gate as the monotonic promotion rule, and the oracle-decomposition methodology as the whole frame. CRITICAL HONEST CONSTRAINT, grounded in the dead-list: this EXACT object was already built once (mover_metalabel.py, D72) and on a proven 1m trigger returned OOS AUC 0.52 -- NO held-out conditional information; D73 is a 5th independent kill of factor->config mapping. The constraint at daily/4h with internal data is INFORMATION (D45/D55/D62/D63 HARD-null), not methodology. So this agent's first deliverable is a VERDICT (does the meta-labeler's selected-subset beat the unconditional trigger AND the random-entry firewall on UNSEEN, with OOS AUC>0.55+), and the highest-EV variable is the FEATURE SET (external/leading data) and possibly RESOLUTION, not the model class. The prior train_meta_labeler.py is genuinely inadequate (dead src/strategy/ import path, position-size-multiplier framing not a clean entry gate, no firewall/PBO/battery, confused win-label) and should be rebuilt, not patched; mover_metalabel.py is the correct template to generalize from 1m-movers to classic-technical-triggers.

# The Strategy-Decision Agent -- concrete build spec (the conditional meta-labeler / setup-selector class)

## 0. What this class IS and IS NOT (name it precisely so we don't re-acquire a dead ceiling)

The user's reframe ("trade good old technical signals, but in a BETTER way -- learn the CONDITIONAL edge") is, named precisely, a **meta-labeling / setup-selector**: a classic trigger has no unconditional edge (the dead-list refuted that exhaustively), so we learn `P(good MOVE | trigger fired, context)` and trade only the high-P subset. This is explicitly the **one ML use the project did NOT kill**: D16 (`docs/MARKET_FRAMEWORK/01_DEAD_LIST.md:31`) -- "ML-as-alpha dead; **meta-labeler survives**". It is NOT a forecaster (IC-banned, D13), NOT predict-then-threshold (the Kronos/IC trap), NOT offline-RL (that is the separate A2 class in `docs/A2_BUILD_SPEC.md`). The A2 spec's own framing (A2 = "the A2 class made concrete on technical-signal features") collapses, for the ENTRY decision, to exactly this supervised meta-labeler -- the offline-RL machinery is unnecessary complexity for a binary enter/abstain decision over a fixed trigger.

## 1. THE UNIT = a triggered classic setup across a multi-candle MOVE -- ALREADY BUILT

`src/strat/setup_harness.py` is precisely this primitive and needs no new sim. It scores an arbitrary past-only boolean ENTRY column chased to a declarative `ExitPolicy`, per window, with UNSEEN as the verdict surface, and it **never computes IC** (`setup_harness.py:65-81` `__contract__`: "IC-INDEPENDENT"). Its leak discipline is structural: `entry_p = opens[i+1]` (next-bar open, Pattern T banned, `setup_harness.py:245-246`); TP/SL/trail breach via `highs[j]/lows[j]` only; pessimistic stop-before-target intrabar; prior-bar high-water-mark for trails; gap-through honest fills. It already duck-types `CanonicalHarness` so it plugs into the firewall + battery with zero glue (`setup_harness.py:20-24`).

**Candidate triggers to enumerate (each a past-only boolean column the harness consumes):**
- MA cross (fast EMA/SMA/WMA over slow), MA stack (price>MA1>MA2>MA3), price-reclaim-above-MA
- N-bar Donchian/close breakout (the RWYB demo, `setup_harness.py:531-532`) and pullback-to-rising-MA
- RSI turn-from-oversold (the dead-list says the TURN is the mechanism, not the depth -- D51)
- Volatility-contraction-then-expansion (vol is a SIZING/entry-eligibility signal per D41/D65, not direction)

Each trigger is the SETUP DEFINITION; the meta-labeler is what makes it conditional. Sweep ALL cadences {15m,30m,1h,4h,1d} + alt-bar-types per the standing rule -- do NOT default 4h.

## 2. THE LABEL = the SETUP OUTCOME (triple-barrier / meta-label), produced BY the harness

For each trigger firing, run the move through `setup_harness` with a pre-registered `ExitPolicy` and label `y = (net_pnl > 0)` where `net_pnl` already nets the 0.0024 taker round-trip (`setup_harness.py:301`). This is the triple-barrier outcome (TP/SL/time, `ExitPolicy` at `setup_harness.py:85-120`), NOT the next candle's return. The label is the MOVE result over the holding window -- exactly the founding unit. Pre-register the exit BEFORE reading any result (the mover_metalabel discipline, `src/mining/mover_metalabel.py:16-38`).

CAVEAT from the dead-list: D09 (`01_DEAD_LIST.md:24`) found triple-barrier labels can strip signal (Val AUC 0.472 in one bull config). Mitigation: report multiple label definitions (TP-hit vs net>0 vs net>cost-band) and verify the meta-labeler's lift is stable across them; a label that only works at one barrier set is overfit.

## 3. THE FEATURES = causal context AT the trigger (this is where the EV lives)

All features computed from data `<= trigger close` (the `causal_features` invariant, `mover_metalabel.py:75`). Per-(asset,feature) z-scores from TRAIN stats ONLY (no leak, `mover_metalabel.py:294-303`):
- **MA geometry**: fast/slow MA slopes, spread, price-to-MA distance, MA stack ordering, time-since-cross
- **Chimera f41**: the curated feature dictionary (regime, flow, OI, funding, liq_ratio, micro-imbalance) -- read via `pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)` (the mandated access path)
- **Regime**: SMA-200 above/below + past-only vol tercile (`within_window_capture_proxy.past_only_regime_bins`)
- **Cross-asset**: BTC-relative position and 24h return at the trigger (`mover_metalabel.py:btc_context`)
- **The WM signal as ONE feature among many**: decode the FROZEN V1.1 forecaster's twohot DISTRIBUTION moments [E[r], Std[r], P(r<0)] via `src/wm/forecast_bundle.py` (detached, eval, `genuine_learning` provenance). It enters as a BELIEF feature, NEVER as the label or a predicted-reward (CDAP `no_predicted_return_as_realized_reward`). This is the legitimate WM transfer: representation/belief, not prediction-consumption.

## 4. THE MODEL = interpretable-first escalation, learning the LINEAR and NON-LINEAR conditional edge

Escalation order (cheapest/most-interpretable first; only climb if the lower rung shows held-out lift):
1. **Logistic regression / elastic-net** -- the LINEAR conditional edge; coefficients answer the user's "which feature relationships matter" directly (TRAIN-median imputation, `mover_metalabel.py:310-316`).
2. **HistGradientBoostingClassifier** -- the NON-LINEAR conditional edge; native NaN handling, seed-fixed, early_stopping=False (`mover_metalabel.py:305-307`). This is the workhorse.
3. **Small attention/NN** -- ONLY if GBM wins held-out AND a sequence-of-context hypothesis is specifically motivated. Biggest silent-overfit surface; gate hardest.

The object is `f(context) -> P(good setup | trigger, context)`, used as a **binary ENTRY GATE** (trade the top-tau predicted subset), NOT a position-size multiplier. Operating point = a pre-registered TRAIN percentile (e.g. tau = 67th, `mover_metalabel.py:89,323`). Fit on TRAIN only; tau/z-stats/imputation-medians ALL from TRAIN.

**This is the precise fix vs the inadequate prior `src/training/train_meta_labeler.py`:** it (a) imports from the DEAD `src/strategy/` path (`train_meta_labeler.py:38`, archived at the 2026-06-04 reset), (b) frames the output as a position-size confidence MULTIPLIER (`train_meta_labeler.py:16-20`) rather than a clean enter/abstain gate, (c) has a confused win-label `ret_trade > 2*cost/2` (`train_meta_labeler.py:109`), (d) does a naive 70/30 chronological split with NO firewall / PBO / battery / UNSEEN-once / per-seed gate. Rebuild on the `mover_metalabel.py` template (which is correct), do not patch.

## 5. THE EXIT = a SEPARATE policy; isolate the ENTRY edge first

Entry and exit are different dimensions. Start with a robust declarative exit (`ExitPolicy`: fixed-horizon, ATR-trail, TP/SL) and hold it FIXED while proving the entry-gate's conditional edge. The exit-as-skill axis is separately measurable via `src/strat/exit_capture_proxy.py` -- but note D61 (`01_DEAD_LIST.md:76`): smart exit-timing was NULL on daily breakouts (the "beats-null" was a hold-length artifact the fair `timing_skill_vs_baseline` test caught). So a LEARNED exit is a Phase-2 fast-follow, gated by `exit_capture_proxy`'s fair test, not part of v1.

## 6. THE GATE CHAIN (all built; the meta-labeler's selected-subset is the candidate stream)

Run in this order on the entry-gate's selected trades (every gate is policy-/stream-agnostic and already exists):
1. **OOS AUC / lift floor** -- the discrimination must exist: OOS AUC > 0.55 (mover_metalabel used 0.52; tighten -- 0.52 is the failure threshold, see below) + monotone TRAIN decile lift that survives OOS.
2. **Random-entry firewall** (`src/strat/firewall.py`, PRIMARY) -- the selected-subset's per-window compound must beat a cost-matched random-entry null. Use `regime_matched=True` (draw the null from trigger-ON bars) to isolate WITHIN-trigger selection from trigger/regime selection, and `membership_matched=True` to isolate trigger-timing from move-selection (`firewall.py:52-72`). Else: BETA-IN-DISGUISE.
3. **Battery Lens A/B/C** (`src/strat/battery.py:evaluate`) -- all-4-positive, n_eff>=15/8, jk2/jk3>0, block-bootstrap p05>0, maxDD<30% (project floor 20%).
4. **PBO via CSCV** (`src/strat/pbo_cscv.py`, ship PBO<0.10) -- MANDATORY because trigger x cadence x feature-set x tau is a SELECTION SEARCH; the in-sample-best is by construction the most noise-fit.
5. **N-seed / per-seed-OOS** (`src/wealth_bot/framework/walk_forward.py`, >=70% seeds OOS-positive) -- meta-labelers are seed-sensitive; single-seed +44% to median -7% is the canonical incident in that module's docstring.
6. **Positive control** (`src/strat/positive_control.py`) -- confirm the chain SHIPS a known genuine synthetic edge (two-sided soundness; a gate that rejects everything is useless).
7. **shuffled_market_control** (`src/agents/_shared/shuffled_market_control.py`) is available if you ever wrap the gate as a policy stream, but for a supervised meta-labeler the firewall+PBO+per-seed triad is the load-bearing overfit defense; the AUC>0.55 floor is the discrimination test.

UNSEEN touched ONCE. 50/20/20/10 + 400-bar purge. LONG-only, spot, lev=1, taker 0.0024.

## 7. The honest prior (brutal; this is the user's core problem so it must be named)

**This exact object was already built and it returned an INFORMATION null, not a methodology null.** `src/mining/mover_metalabel.py` (D72, `01_DEAD_LIST.md:88`) is a correctly-built trigger-time meta-labeler on a PROVEN trigger (1m mover with confirmed +5.3-5.7% median meat) with 16 causal features, HGB+logit, TRAIN-fit on 7054 events, OOS-once on 3369 -- and got **OOS AUC 0.521/0.510**: NO held-out continuation information; the selected-third was indistinguishable from the rejected-third. D73 is a 5th independent kill of factor->config mapping at the per-move bar. D45/D62/D63 HARD-null bar-level entry-timing at daily/4h; D55 HARD-nulls direction across 100+ assets x 5 TF. **The conditional edge the user hypothesizes is real in PRINCIPLE but has repeatedly measured ~0 with INTERNAL data at daily/4h/1m.**

Therefore the highest-EV design variable is NOT the model class (logistic vs GBM vs attention all converge to the same AUC~0.5 when the information isn't there -- D17/D55 proved this) but the **FEATURE SET**: the open thread (D72 spec) is a trigger-time signal with OOS AUC > ~0.58 from EXTERNAL/leading data (Coinglass liq-heatmap proximity, on-chain netflow, news/social), and possibly RESOLUTION (sub-15m, a DATA decision orthogonal to the agent). Build the agent so the feature set is a swappable input and the FIRST deliverable is a VERDICT on a given (trigger, cadence, feature-set): does the selected subset beat the unconditional trigger AND the firewall on UNSEEN with AUC>0.55? A clean "NO with internal features at 4h" is a high-value, cheap re-confirmation that the constraint is information; a "YES" with a new external feature is the first genuine alpha. Treat "the conditional meta-labeler finds edge" as [PROJECTED], never measured, until UNSEEN says otherwise.

## 8. Concrete component map (what to build, where)

- `src/strat/setup_meta_labeler.py` (NEW) -- the generalization of `mover_metalabel.py` from 1m-movers to ARBITRARY classic-technical triggers: takes a trigger-column generator + an `ExitPolicy`, emits per-trigger causal features + harness labels, fits the interpretable->GBM escalation TRAIN-only, applies tau, returns the selected-subset trade stream. Carries `__contract__` (ml_as_metalabeler_only / train_only_fit / causal_features / unseen_untouched), pre-registration block, git+seed lineage.
- REUSE unchanged: `setup_harness.py` (unit+label+leak guard), `firewall.py` + `battery.py` + `pbo_cscv.py` + `positive_control.py` (gate chain), `walk_forward.py` (seeds), `forecast_bundle.py` (WM-as-feature), `within_window_capture_proxy.py` / `exit_capture_proxy.py` (entry/exit axis decomposition), the oracle (`src/oracle/run.py`) for the hindsight setup-selector ceiling the proxy targets.
- DEPRECATE: `src/training/train_meta_labeler.py` (tombstone -- dead import path + wrong framing), `src/training/train_rank_model.py` (cross-sectional LambdaRank = D17 HARD-null), `src/strat/vol_config_ml.py` (D73 territory -- factor->config, 5th-killed). `signal_picker.py` survives as the per-strategy regressor substrate but its +49.8% PEPE headline is pre-reset and apparatus-suspect; re-verify before citing.

Effort: ~3-5 focused days for the v1 (one trigger family x sweep of cadences, internal features, the full gate chain producing the VERDICT). The agent is far cheaper than A1/A2 because it reuses the entire sim + gate apparatus and is supervised, not RL.

**Failure modes:** (1) INFORMATION CEILING mistaken for a methodology gap -- the #1 risk: D72 already showed this exact agent gets OOS AUC 0.52 with internal features; swapping logistic->GBM->attention will NOT move it (D17/D55: model class is irrelevant when the info isn't in the bars). The agent must be built feature-set-first, and a null at 4h-internal is the EXPECTED first result, to be reported honestly, not tuned-through. (2) AUC-as-the-objective regression -- AUC is a within-meta-labeler discrimination DIAGNOSTIC (floor >0.55), NOT the ship metric; the ship metric is held-out COMPOUND of the selected subset beating the unconditional trigger AND the firewall. Reporting AUC alone re-imports an IC-like per-decision lens. (3) Selection bias from the trigger x cadence x feature x tau search -- the in-sample-best config is by construction the most noise-fit; without PBO<0.10 this is the canonical evolutionary-search trap (the whole reason pbo_cscv exists). (4) Label leakage / barrier overfit -- D09 showed triple-barrier can strip signal; a meta-labeler that only lifts at one barrier set is fitting the label, not the move. (5) Position-size-multiplier framing (the prior train_meta_labeler.py defect) -- conflates the entry-gate decision with sizing; keep entry binary and isolate it first. (6) WM-as-prediction leak -- using the frozen forecaster's decoded return as the label or reward (not just a belief feature) trips CDAP no_predicted_return_as_realized_reward and re-acquires forecaster-GIGO. (7) Concentration -- a few big trades carrying the selected subset (the recurring lesson #6; D71 DOGE +27pp/event); n_eff and jackknife jk2/jk3 in the battery guard this. (8) Single-seed overfit -- the +44%->-7% incident; the >=70% per-seed-OOS gate is more binding for a searched meta-labeler than for a forecaster. (9) Survivorship -- current-membership universe inflates absolute-level features; point-in-time universe or at minimum the caveat logged (mover_metalabel flags it).

**Open questions:** (1) What EXTERNAL/leading feature pushes trigger-time discrimination from AUC~0.52 to >0.58? The D72 spec names Coinglass liq-heatmap proximity, on-chain netflow, news/social -- none are in-tree yet; this is the actual frontier and may be a data-acquisition project, not a modeling one. (2) Does the conditional edge live BELOW the bar -- i.e. is RESOLUTION (sub-15m) the real lever rather than features, given D45/D60 cost-walls and D72's 1m meat? (3) Which trigger family has the highest oracle ceiling to condition on -- run the oracle (src/oracle/run.py) to find which classic setups precede the best moves before building the meta-labeler on them, so we condition on a trigger with meat (the D67/D72 lesson: meat must EXIST before discrimination matters). (4) Is the right label a binary outcome or a magnitude-bucketed one (does conditioning on "big move vs small move" carry more held-out info than "win vs loss")? (5) Should the agent ever route ACROSS trigger families by regime (a within-asset ensemble) -- or is that D33/D73 territory (routing dilutes)? (6) Cost realism: at finer cadences the maker p_fill 0.21-0.40 wall (D43) and the 1h cost-wall (D60) may kill the selected subset even with genuine AUC -- the gate must use taker 0.0024 not optimistic fills. (7) Is signal_picker.py's pre-reset +49.8% PEPE result reproducible under the current hardened apparatus, or apparatus-inflated like the archived REGIME_ROUTER +20.25%?

## LENS: THE HONEST CEILING + THE RUTHLESS VALIDATION — the project's conscience. I judge "does conditioning a classic trigger on context features produce a robust held-out edge?" against the single most overfit-prone thing in quant (a conditional-edge meta-labeler searches a huge config space), with the project's own already-built gauntlet as the bar. Every "it works" is [PROJECTED] until the gauntlet passes; I name the specific mechanisms by which a backtest artifact masquerades as a conditional edge.

**Summary:** The reframe is SOUND and the dead-list does NOT refute it: IC~0 is per-candle UNCONDITIONAL prediction; the hypothesis here is per-SETUP CONDITIONAL edge (a trigger with no unconditional edge may have edge when context features hold). These are different objects — but conditional-edge meta-labeling is the highest path-overfit-risk activity in quant, so the hypothesis STILL has a ceiling (the context may not predict setup-quality either) and "it works" must be treated as [PROJECTED] until a ruthless gauntlet passes. The prior attempt (src/training/train_meta_labeler.py) is the careless version the user correctly distrusts: it optimizes binary win/loss AUC (the wrong objective — accuracy is not held-out compound), uses a leaky in-file 70/30 chronological split (NOT the 50/20/20/10+400-bar-purge invariant), touches OOS/UNSEEN, has a label-construction bug (win = ret_trade > 2*cost/2, i.e. > cost — a label-leakage-adjacent definition that bakes the cost threshold into the label), runs NO firewall / NO PBO / NO shuffled-market control / NO seed-robustness / NO per-regime split, and its strongest "signal" (regime + asset one-hots top the feature importance) is exactly the regime-as-the-only-signal failure mode (it learns "bull markets win," i.e. beta, not conditional timing edge). The careful template ALREADY EXISTS in-tree: src/mining/mover_metalabel.py is the right way (pre-registered thresholds, TRAIN-only fit of model+z-stats+imputation+tau, causal features, OOS gates incl AUC>0.52 + breadth + jackknife, UNSEEN sealed) — and even IT, done carefully, found continuation-given-onset has NO internal edge (OOS AUC 0.52, D72). The RIGHT spec: optimize HELD-OUT COMPOUND on the SETUP (never AUC/accuracy/IC); the gauntlet is src/strat/{firewall,pbo_cscv,battery,positive_control}.py + src/agents/_shared/shuffled_market_control.py + walk_forward 50/20/20/10+400-purge, ≥8/10 seeds, PBO<0.10 (mandatory — the config search makes path-overfit risk HIGH), shuffled-market GENUINE, taker 0.0024, beats a regime+cost-matched random-entry null AND buy-hold per-regime, and a no-skill control (random/dumb meta-labeler) must FAIL. Honest prior: most likely first result at daily/4h is the conditional edge does NOT survive the firewall — and that null is the high-value product, not a failure.

## The reframe is correct AND it still has a ceiling — both true

**Why the dead-list does NOT refute this (the load-bearing distinction).** The dead-list's IC~0 / per-candle-predictability findings, and the founding-framing ban on IC (MEMORY.md), are about the UNCONDITIONAL, PER-CANDLE object: E[r_{t+h} | x_t] has ~0 predictability. The user's hypothesis is a DIFFERENT object: P(setup pays across the MOVE | a classic trigger fired AND context features C hold) — a CONDITIONAL, PER-SETUP object. A trigger (MA cross, breakout) with zero unconditional edge can in principle have positive edge on the sub-population where C holds. Conditioning is not prediction; the meta-labeler is the textbook frame for it (López de Prado AFML Ch.3, cited in src/training/train_meta_labeler.py:21). So "the dead-list killed this" is FALSE — it killed a sibling, not this.

**But the ceiling is real and must be stated up front.** The conditional hypothesis can fail two ways that are NOT the IC failure: (1) the context features C genuinely don't discriminate setup-quality (the conditional edge is also ~0 — this is what D72 found at 1m: continuation-given-onset OOS AUC 0.52, i.e. the meta-labeler is a coin flip; project-d72 memory); (2) C *appears* to discriminate in-sample purely because the meta-labeler searched a large config space and selected the path-fit winner. Treat "conditioning produces a robust edge" as **[PROJECTED]** until the gauntlet passes. This is not pessimism — it is the only honest prior, because conditional-edge meta-labeling is the single most overfit-prone construct in quant: it multiplies the search space (triggers × context features × thresholds × barriers × holds × regimes) and then reports the best cell.

## Audit of the prior CARELESS attempt (why the user's distrust is justified) — src/training/train_meta_labeler.py

I read the file. Concrete, file:line-grounded defects — each is a reason the prior result is uninterpretable, not a small flaw:

1. **WRONG OBJECTIVE (the cardinal sin).** It trains a LightGBM binary classifier on win/loss and reports Val AUC + Brier (train_meta_labeler.py:234-239). AUC/accuracy is NOT the objective. A meta-labeler can have AUC 0.55 and NEGATIVE held-out compound (it can be "right" on many tiny winners and "wrong" on the few large losers — asymmetric payoff makes accuracy orthogonal to wealth). The objective must be held-out COMPOUND on the SETUP stream, per CLAUDE.md's wealth-not-Sharpe mandate. AUC is at best a within-model diagnostic, never the gate.

2. **LEAKY / NON-CANONICAL SPLIT.** It uses an in-file `split_idx = int(0.70*n)` chronological 70/30 split of the *trade records* (train_meta_labeler.py:206-208), NOT the project invariant 50/20/20/10 with a 400-bar purge gap (CLAUDE.md "Data split"). Splitting on trade-record index (not bar time, with purge) means a trade whose entry is in "train" and exit in "val" straddles the boundary → normalization/label leakage. There is no UNSEEN segment held back at all.

3. **OOS/UNSEEN TOUCHED.** `build_labels(...)` scans `warmup..splits.val_end` (train_meta_labeler.py:163) — it stops at val_end so it doesn't read OOS here, but there is no sealed UNSEEN reserve and no walk-forward; the model is fit and "validated" on the same TRAIN+VAL pool with an internal split. The careful version (mover_metalabel.py:80-101, split_of + UNSEEN-untouched) is the contrast.

4. **LABEL-DEFINITION SMELL.** `win = 1 if ret_trade > 2*cost/2 else 0` (train_meta_labeler.py:109) — i.e. win iff gross > cost. Baking the cost threshold into the binary label, then optimizing classification of that label, is not the same as optimizing net compound; it also makes the label sensitive to the exact cost constant, and triple-barrier labels are themselves a known leakage surface (the barrier outcome is determined by future bars — fine for a label, but the FEATURES must be strictly causal at entry, which is not asserted/tested here).

5. **REGIME-AS-THE-ONLY-SIGNAL, UNGUARDED.** Features include asset one-hot, regime one-hot, bucket one-hot (train_meta_labeler.py:198). With no per-regime evaluation and no random-entry firewall, the model will rank `regime_id`/`asset_id` high (it learns "ALT bull-regime trades win") — that is BETA, not conditional timing edge. There is no firewall to strip it. This is THE failure mode the user's own apparatus (firewall.py regime_matched + membership_matched, lines 52-118) was built to catch, and the prior attempt used none of it.

6. **NO MULTIPLE-COMPARISONS / SELECTION-BIAS CONTROL.** The script is parameterized over --universe, --k-up, --k-down, --max-hold (train_meta_labeler.py:140-143); running it across that grid and keeping the best is an un-deflated multiple-comparisons search. No PBO (pbo_cscv.py), no DSR-at-family-N. The in-sample-best config is by construction the most noise-fit (pbo_cscv.py:7-9).

7. **NO SEED ROBUSTNESS, NO POSITIVE/NEGATIVE CONTROL.** Single LightGBM fit (seed implicit). No ≥8/10-seed audit (walk_forward PER_SEED_OOS_GATE), no no-skill control proving a dumb meta-labeler FAILS, no positive control proving the chain can SHIP a known edge.

The careless attempt is therefore **uninterpretable** — even if it printed a positive backtest, none of the artifacts that separate a real conditional edge from a path artifact were run. The user's instinct ("not done thoroughly/carefully/with the right goal") is exactly right and is now grounded in code.

## The RIGHT goal + spec (what the prior attempt lacked)

**(a) Objective.** Optimize HELD-OUT COMPOUND on the SETUP-across-the-MOVE stream, net of taker 0.0024, never AUC/accuracy/Brier/IC. The unit is the per-setup realized net return (the SETUP/MOVE-level ShIC analog, shuffled_market_control.py:65, 122). AUC>0.52 survives ONLY as a within-model diagnostic GATE (as in mover_metalabel.py:352), never as the thing being maximized. The meta-labeler is a GATE/SIZER on a fixed causal trigger — never a signal generator (mover_metalabel.py:67, the "ml_as_metalabeler_only" invariant; this is the framework-endorsed ML use, cf the dead-list D16/D17 ban on ML signal generators).

**(b) Pre-registration (the discipline the prior attempt skipped).** Before reading any result, fix: the trigger mechanism (by mechanism, not by grid result — mover_metalabel.py:18-21), the context feature set (ALL causal at the trigger close — mover_metalabel.py:25-29), the exit policy, the label (net>0), the operating point (tau as a TRAIN quantile — mover_metalabel.py:32, 323), and the asymmetric loss (a false LONG costs more than a missed winner — A2_BUILD_SPEC B.7, pre-registered never tuned on UNSEEN). TRAIN-ONLY fit of the model, the z-score stats, the imputation medians, AND tau (mover_metalabel.py:73-77, 294-323) — the careless version fit everything on the pooled set.

## The RUTHLESS validation gauntlet (the apparatus the prior attempt skipped — all already built)

Run in this order; each is a KILL gate, not advisory. (This IS the A2_BUILD_SPEC B.5 chain repointed at the meta-labeler's selected-setup stream — A2_BUILD_SPEC.md:247-262.)

1. **Random-entry firewall — PRIMARY (src/strat/firewall.py).** The selected-setup compound must beat a null of the SAME trade count entered at RANDOM bars, held for durations from the candidate's OWN holding distribution, at the SAME cost, on ALL held-out windows. CRUCIALLY use **regime_matched** (draw nulls only from gate-ON bars — firewall.py:52, 97-100) AND **membership_matched** (draw nulls from WITHIN the same multi-candle MOVE the real setup fires in — firewall.py:104-118): membership_matched isolates TRIGGER TIMING from MOVE-SELECTION, the exact discriminator for "is the conditional edge real timing or just being-present-in-good-moves." If it doesn't beat the regime+membership-matched null → BETA/MOVE-SELECTION IN DISGUISE.

2. **Shuffled-market control — the policy-overfit / ShIC analog (src/agents/_shared/shuffled_market_control.py).** Re-score the SAME meta-labeled policy on surrogate markets (perm + block — predictability-destroying, marginal preserved; verdict driven ONLY by these, shuffled_market_control.py:236, 277, 337-341). GENUINE iff it collapses to ~0 on the surrogate (overfit_fraction ≤ 0.20, beats p95) — its edge came from real temporal structure. OVERFIT (≥0.35) → it memorized the path → KILL. This catches the invisible memorizer no inspection can (H3, A2_BUILD_SPEC.md:173-176). phase/iaaft are mechanism diagnostics only (a hard ValueError fires if you put them in verdict_kinds — shuffled_market_control.py:337).

3. **PBO via CSCV — MANDATORY here, not optional (src/strat/pbo_cscv.py, ship PBO<0.10).** The meta-labeler searches a large config space (triggers × features × thresholds × barriers × holds); the in-sample-best is by construction the most path-fit (pbo_cscv.py:7-19). PBO answers the orthogonal-to-DSR question: does the SELECTION PROCESS itself produce OOS under-performers? PBO~0.5 = skill-less selection. This is the gate the config search specifically requires.

4. **Walk-forward + seed robustness (src/wealth_bot/framework/walk_forward.py).** 50/20/20/10 + 400-bar purge (CLAUDE.md invariant); ≥8/10 seeds OOS-positive (the gauntlet floor; the module's own PER_SEED_OOS_GATE_PCT=70 is the looser harness default — A2_BUILD_SPEC.md:116-119 carries the canonical cautionary tale: single-seed +44%/+40% LSTM/DQN debunked to median −7%/−34% at 10-seed audit, the project's worst overfit incident). Bootstrap p05>0.

5. **Battery Lens A/B/C (src/strat/battery.py evaluate).** all-4-windows-positive, n_eff floor, jackknife jk2/jk3>0, p05>0, maxDD<30% (20% binding downstream).

6. **No-skill control (the prior attempt's missing conscience).** A RANDOM meta-labeler and a DUMB one (e.g. label = current regime only) must FAIL the chain. If a coin-flip "conditioner" passes, the gate is broken — this is the two-sided-soundness rule (positive_control.py proves the chain SHIPs a known genuine edge; the no-skill control proves it KILLS a fake one). The exit_capture_proxy already caught a hold-length artifact via exactly this no-skill control (project-exit-axis-null memory) — the same artifact lurks here.

7. **Per-regime reporting (never aggregate).** Report compound per trending/mean-reverting/high-vol regime separately (A2_BUILD_SPEC.md:196-201). An aggregate +X% that is +40% bull / −30% bear is a regime BET, not a conditional edge. This is the direct antidote to defect #5 above.

8. **Leak probe (src/wealth_bot/leak_probe.py relative_leak_test).** Use the RELATIVE test against a same-cadence past-only reference (absolute over-triggers on coarse bars — A2_BUILD_SPEC.md:110-114). The triple-barrier label + any feature touching the move's future = the leakage surface to probe.

## The specific ways this fails (name them, watch for them)

- **Multiple-comparisons / selection bias.** The config grid → the best cell is noise-fit. CAUGHT BY: PBO<0.10. UNGUARDED in the prior attempt.
- **Label leakage via the barrier.** Triple-barrier outcome uses future bars (fine for the label) but a feature computed from the same future window, or normalization fit on the whole series, leaks the answer. CAUGHT BY: causal-feature pre-registration (every feature ≤ trigger close, mover_metalabel.py:25-29) + relative_leak_test. The prior attempt's extract_context (train_meta_labeler.py:52-80) reads r[t-30:t] (causal) but never asserts/tests it, and its z-stats are not provably TRAIN-only.
- **Regime-as-the-only-signal (beta in disguise).** The model learns "bull-regime/ALT trades win." CAUGHT BY: regime_matched firewall + per-regime reporting. This was the prior attempt's TOP feature (defect #5).
- **Move-selection masquerading as timing.** The "edge" is being present in good moves, not the trigger timing. CAUGHT BY: membership_matched firewall (firewall.py:104-118).
- **Concentration.** Compound carried by 1-3 trades/assets. CAUGHT BY: jackknife drop-top-K>0 (battery + mover_metalabel.py:238-241). The wealth-bot trust stack (CLAUDE.md) requires top_3_pct_of_compound discipline.
- **Path-memorization invisible as a curve.** CAUGHT BY: shuffled-market control (the only gate that sees it).
- **Seed lottery.** One lucky seed. CAUGHT BY: ≥8/10-seed audit.

## Honest prior (state it before any result)

The careful version of this exact construct, run at 1m on movers (mover_metalabel.py / D72), already found continuation-given-onset has NO internal edge (OOS AUC 0.52). At daily/4h LO the project's repeated history (D17/D44/D45, the A/B/C fork) is that signal is weak. The MOST LIKELY first result is that the conditional edge does NOT beat the regime+membership-matched random-entry firewall on UNSEEN — and per A2_BUILD_SPEC.md:279-282 + the report-first mandate, that NULL is the high-value deliverable (it bounds the ceiling cheaply with gates that already exist), NOT a failure. Ship only if the FULL gauntlet passes; tie/marginal ⇒ no edge to ship. The conditional reframe is worth testing precisely BECAUSE it is a genuinely different hypothesis from the IC null — but it is the most overfit-prone test in the book, so the gauntlet is the whole point.

**Failure modes:** (1) Multiple-comparisons over the trigger×feature×threshold×barrier×hold grid → in-sample-best is noise-fit (guard: PBO<0.10, pbo_cscv.py). (2) Label leakage via the triple-barrier / non-causal features / whole-series normalization (guard: causal-feature pre-registration ≤ trigger close + relative_leak_test). (3) Regime-as-the-only-signal = beta in disguise (guard: regime_matched firewall + per-regime reporting; this was the prior attempt's TOP feature). (4) Move-selection masquerading as trigger-timing (guard: membership_matched firewall). (5) Concentration in 1-3 trades/assets (guard: jackknife drop-top-K>0). (6) Invisible path-memorization shown only as a profitable curve (guard: shuffled_market_control — the policy-side ShIC). (7) Seed lottery (guard: ≥8/10-seed walk-forward; the +44%→−7% incident). (8) Optimizing AUC/accuracy instead of compound — the cardinal prior-attempt sin (guard: objective = held-out compound; AUC>0.52 is a diagnostic gate only). (9) No two-sided soundness — a coin-flip meta-labeler passing (guard: no-skill control must FAIL + positive_control must SHIP).

**Open questions:** (1) Does any classic trigger + causal-context cell beat the regime+membership-matched random-entry firewall on UNSEEN at ANY cadence — and is it the same one D72 already nulled at 1m, or a genuinely untested cadence/trigger? (2) Asymmetric loss is mandated (false-LONG > missed-winner) but the exact ratio is a pre-registration choice — what ratio, and how to fix it without tuning on UNSEEN? (3) Per-regime sample sizes: high-vol/bear regimes may have n_eff below the floor, making the per-regime gate unpassable — does the test then honestly say "insufficient data to conclude" rather than pooling? (4) Warm-starting context-feature representation from a Gate-A forecaster encoder (H1, representation-only, no prediction) — does it help the meta-labeler, and does it stay clear of the no_predicted_return_as_realized_reward CDAP line? (5) Is the trigger universe point-in-time (no survivorship) — mover_metalabel.py flags u10 current-membership survivorship on absolute levels as an open caveat.

