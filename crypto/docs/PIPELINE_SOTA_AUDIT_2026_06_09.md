# Solutioning Pipeline — SOTA audit + fortification (2026-06-09)

The [Solutioning Pipeline](SOLUTIONING_PIPELINE.md) was red-teamed (opus, verified against live code) and benchmarked
against world-class **quant research-to-production**, **MLOps/data-science lifecycle**, and **autonomous-discovery**
frameworks (3 cited web-research passes). This doc records the audit, the SOTA checklist we measured against, what was
fortified, and the honest remaining gaps. The verdict on v1 was fair: *a clean honest stage-LEDGER, but not a
reproducible pipeline — gates were prose not predicates, nothing was bound to code/data/seed, and it named but never
invoked the real apparatus.* v2 fixes the top-5 EV gaps.

## What the audit found + what was fixed (top-5 EV)
| ID | Finding (verified vs live code) | Fix (committed) | Status |
|---|---|---|---|
| **C1** | Gates were prose assertions (`--passed --evidence "<text>"`) — nothing prevented a false pass | **Machine-checked `gate_spec` per stage**: 02_engine RUNS `check_invariants.py` (CDAP), 03_strat passes iff a SHIP run exists in `runs.jsonl`; `--manual-override` stamps `passed_by=human` (auditable) | FIXED |
| **C2** | Zero reproducibility lineage — no result bound to git SHA / data / seed / env | **`lineage` block** auto-captured at every record/gate/run: `{git_sha, git_dirty, artifact_sha256, python, seed, data_ref, ts}` | FIXED |
| **H3** | No run/experiment-level tracking — stage-03 candidates had nowhere structured to live (re-scatter) | **Run registry** `runs.jsonl` + `run` cmd: `{run_id, params, metrics, status, lineage}` (the MLflow-shaped layer); the strat gate selects the SHIP run from it | FIXED |
| **H4** | Framework only NAMED `candidate_gate`/CDAP, never invoked them | gate_spec now **invokes** CDAP (02) and the SHIP-run predicate (03); strat candidates flow through `candidate_gate` then `run --status SHIP` | FIXED (gate wiring) |
| **H5** | `status` and `gate.passed` decoupled; registry counted stale `done` | **`status` derived from `gate.passed`** (single source); registry counts gate-passed stages; un-pass resets status | FIXED |
| **H6** | Non-atomic writes, no lock — violated the project's own atomic-write contract; concurrent writers corrupt | **atomic tmp+os.replace** + **fail-open per-workspace lock** (commit-lease pattern); read-modify-write under lock | FIXED |
| M7 | doctor/registry crashed on a malformed manifest | per-manifest try/except + reported as `MALFORMED`, counted toward exit | FIXED |
| M8 | `ref` artifacts auto-passed doctor (dangling cross-workspace refs invisible) | doctor now **verifies ref targets** (the referenced workspace manifest must exist) | FIXED |
| M9 | exact-dict dedup let duplicate artifacts accumulate (verified: DEAD_LIST recorded twice) | **dedup on PATH** (update note/lineage in place) | FIXED |
| M10 | `gate`/`advance` didn't validate stage names (KeyError) | shared `_check_stage` guard | FIXED |
| M12 | `CostModel.round_trip(maker:bool)` baked in a crypto maker/taker model | generalized to `round_trip(symbol, side, notional, venue)` — each market models its own cost | FIXED |
| L11 | stage-06 "monitoring/decay/kill-switch" is a label, not a mechanism | doc reframed as "records that monitoring is armed"; the live monitor (drift/champion-challenger/rollback) is **deferred** (see gaps) | NOTED |

## The SOTA checklist (what world-class frameworks have) — our coverage
Synthesized from the research; ✅ = satisfied by v2, ◻ = deferred/open (tracked below).
- ✅ **Held-out / UNSEEN-once discipline + positive/negative controls** — `candidate_gate` (firewall null + benchmark + battery + positive_control); the dead-list is the refuted-hypothesis registry.
- ✅ **Reproducibility = code+data+seed+env per result** — the `lineage` block (matches MLflow/W&B/PROV practice).
- ✅ **Run → experiment → registry hierarchy** — `runs.jsonl` (runs) + manifest stages (experiments) + SHIP-run promotion gate.
- ✅ **Machine-checked gates / exit-code contracts** — gate_spec runs CDAP / SHIP-predicate; CDAP itself is the unskippable pre-commit gate.
- ✅ **Atomic, resumable, idempotent store** — atomic writes + lock + migrate-on-load; `advance` is a strict gated state machine.
- ✅ **Knowledge accumulation (episodic/semantic/procedural)** — memory/ + MARKET_FRAMEWORK dead-list + the skill library.
- ✅ **Anti-reward-hacking: evaluator independent of generator** — `candidate_gate` is a separate, code-fixed evaluator; the dead-list + 4-bounds + UNSEEN-once guard self-deception (the project's whole methodology, doc 03).
- ✅ **Cost realism as the binding constraint** — taker 0.24% baseline + calibrated p_fill (the audit's #4 ad-hoc failure mode is closed).
- ◻ **Champion/challenger + shadow deployment** (stage-06) — not built (no live capital yet; deferred).
- ◻ **Drift/decay detectors (PSI/KS, concept-drift, alpha-decay monitor)** (stage-06) — deferred until a strat is live.
- ◻ **Feature store / data-version snapshots (DVC/lakeFS-style)** — `data_ref` field exists; a content-addressed data snapshot id is not yet wired (crypto uses chimera manifests `data/manifests/v51_*.json`).
- ◻ **Full DAG orchestration (Snakemake/Dagster-style) with step caching** — the pipeline is a gated linear state machine; a DAG with caching is a future upgrade if stages branch.
- ◻ **Schema-validated stage I/O contracts (Great Expectations/TFDV)** — manifests are validated for shape; per-artifact data-schema validation is deferred.

## The remaining gaps (honest, EV-ranked)
**WIRED 2026-06-09 (the "wire remaining" pass):**
- ✅ **CryptoAdapter(MarketAdapter)** — `src/framework/crypto_adapter.py` wraps the chimera loader / universe yamls /
  taker cost / feature_map / data manifests; `isinstance(CryptoAdapter(), MarketAdapter)==True` (RWYB). The
  agnosticism loop is closed; it's the template a stocks adapter copies.
- ✅ **Data-snapshot lineage** — `record/run --data-ref` is now populated from `CryptoAdapter.data_snapshot_id()`
  (a content hash of `data/manifests/v51_*.json`); a result is bit-reproducible against the exact data version.
- ✅ **CDAP coverage** — `check_invariants.check_framework_store()` runs the framework `doctor` every commit
  (warn-level); the store can no longer drift to broken paths/refs unnoticed (RWYB: catches a planted missing artifact).

**Still deferred (need a live strat, or low EV now):**
1. **Stage-06 monitoring** (champion/challenger + drift + alpha-decay + auto-rollback/kill-switch) — the biggest
   deferred block; only needed once a candidate clears stage-03/05 and takes real capital.
2. **Difficulty-adaptive evaluator depth** — self-consistency K-sampling on the highest-stakes gates (the
   autonomous-discovery anti-Goodhart lever) — low EV until stage-03 strat-search is running.

## Sources (researched frameworks)
Quant lifecycle: W&B "Architecting Alpha", NautilusTrader (research=backtest=live single codepath), QuantConnect LEAN,
VertoxQuant strategy-decay, the champion/challenger + shadow-deployment patterns. MLOps: CRISP-DM, Microsoft TDSP,
Google MLOps maturity + Rules of ML, the **ML Test Score (Breck et al. 2017)**, MLflow/W&B run-experiment-registry,
Feast/Tecton feature stores, DVC/lakeFS data versioning, Great Expectations/TFDV. Autonomous discovery: AlphaEvolve
(program-DB / island model), Sakana AI-Scientist (FSM + tree search), METR reward-hacking findings (evaluator outside
the write perimeter), W3C PROV + FAIR + RO-Crate provenance, the three-memory model (episodic/semantic/procedural).
(Full URL list in the 2026-06-09 research-scout outputs; key anchors: research.google ML Test Score; arXiv 2506.13131
AlphaEvolve; arXiv 2502.14297 AI-Scientist eval; metr.org reward-hacking.)
