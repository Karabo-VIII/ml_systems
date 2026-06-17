# The Solutioning Pipeline — the repeatable, market-agnostic working model

**Mandate (user, 2026-06-09):** *"Establish a framework for working through these problems: research/decomposition →
mining → engine build → strat build → bot build → execution → deployment. Something that lets me take a market + an
instrument, put it through this working model, and get the result I want — for stocks or crypto. Store the information
accurately (not disparately). I want something repeatable."*

This is the **project's operating model**: a single, repeatable process that takes a `(market, instrument)` from "we
know nothing" to "deployed bot," with **one coherent store per workspace** so information is never scattered. The
**stages, gates, and storage are GENERIC**; each market supplies the per-stage *implementation*. The crypto
implementation already exists (it's the apparatus this project has built); a stocks workspace would supply stock
equivalents of the same contracts.

> Realized in code: [`src/framework/pipeline.py`](../src/framework/pipeline.py) (the workspace/manifest/registry tool)
> + [`workspaces/`](../workspaces/) (the store) + [`workspaces/REGISTRY.md`](../workspaces/REGISTRY.md) (the global
> index). Meta-layer: [`src/firm/`](../src/firm/) (PM/CIO). Autonomy layer: `docs/AUTONOMY_FRAMEWORK.md` (runs it).

## The 7 stages (each advances only when its GATE passes)
Stages **00–02 are MARKET-scoped** (decompose/mine/engineer once per market, shared by all its instruments → stored
under the `_market` workspace). Stages **03–06 are INSTRUMENT-scoped** (per BTCUSDT, AAPL, …).

| # | Stage | Purpose | Output (the artifact) | ADVANCE GATE | Crypto implementation | (Stocks would be) |
|---|---|---|---|---|---|---|
| 00 | **research / decomposition** | decompose the market to its fundamental constituents (the 8-axis lattice) | market model + feature dictionary + dead-list + decomposition lattice | `research_complete`: market decomposed; dead-list exists | `docs/MARKET_FRAMEWORK/`, `CRYPTO_MARKET_UNDERSTANDING`, `CHIMERA_FEATURE_DICTIONARY` | equities microstructure model + factor dictionary + dead-list |
| 01 | **mining** | mine the data for STRUCTURE (regime / cluster / trend / predictability) | mining findings + a decomposer/viewer | `mining_complete`: structure characterized, RWYB | `src/mining/` (chimera_mine/analyze/deep_mine/predictability/decompose) + `CHIMERA_MINING_FINDINGS` | the same engine on an equities feature panel |
| 02 | **engine build** | build the reusable ENGINES + the validation APPARATUS | oracle/decomposer engines + the gate chain | `engine_ready`: gate chain selftests + accepts a positive control | `src/oracle/`, `src/strat/candidate_gate`+`battery`+`firewall`+`benchmark`, `src/firm/` | reuse the SAME gate apparatus (market-agnostic); swap the data adapter |
| 03 | **strat build** | build + GATE-validate strategy candidates | gated strategy candidate(s) | `strat_gated`: clears `candidate_gate` net-of-cost, beats passive, 10-seed, UNSEEN-once | `src/strat/` candidates run through `candidate_gate` | same gate; equities cost/borrow model |
| 04 | **bot build** | wrap a validated strat into a BOT (sizing / risk / lifecycle) | bot spec + code | `bot_built`: position-sizer + risk controls + lifecycle wired | `src/wealth_bot/bot/`, `src/firm/` (risk/portfolio/decision_spine) | same bot skeleton; equities risk limits |
| 05 | **execution** | execution model (fills, costs) + paper-trade | execution config + paper-trade result | `execution_validated`: realistic-cost paper-trade matches backtest within tolerance | maker/taker cost model (`config/maker_cost_calibration.yaml`) + paper_trade | broker fill model + paper-trade |
| 06 | **deployment** | deploy to live (monitoring / decay / kill-switch) | deploy spec + live monitor | `deployed`: monitoring + decay-detection + kill-switch armed | `deployment_ranking.yaml`, `risk_manager` kill-switches | broker API + the same monitor |

**Over the stages sits the META-LAYER** (the PM/CIO — `src/firm/`: decision-spine, market-state, portfolio, risk,
trader-mindset) that judges and allocates, and the **AUTONOMY layer** (the agent that *runs* the pipeline —
`docs/AUTONOMY_FRAMEWORK.md`, the `/orc` 3-loop model). The pipeline is the WHAT; the firm-harness is the WHO-decides;
the autonomy loop is the HOW-it-runs.

## The storage model — one coherent store, never disparate
Every `(market, instrument)` is a **workspace** with a single source of truth (`manifest.yaml`) + 7 stage dirs:
```
workspaces/
  REGISTRY.md                      # auto-rendered global index of all workspaces + their stage
  <market>/
    _market/                       # the MARKET-scoped workspace (stages 00-02, shared)
      manifest.yaml                # THE store: current_stage, per-stage status+gate+artifacts
      00_research/ 01_mining/ 02_engine/ 03_strat/ 04_bot/ 05_execution/ 06_deployment/
    <INSTRUMENT>/                  # e.g. BTCUSDT, AAPL -- INSTRUMENT-scoped (stages 03-06)
      manifest.yaml                # 00-02 reference _market; 03-06 are this instrument's
      00_research/ ... 06_deployment/
```
The **manifest is the anti-disparity mechanism**: instead of findings scattered across `docs/`, `runs/`, `memory/`,
each artifact is *recorded against its stage* in the workspace it belongs to, with a kind + note. You always know, for
any market/instrument, exactly what stage it's at, what's been produced, and whether each gate passed — from one file.

### The repeatable loop (the CLI)
```
python -m framework.pipeline init     <market> <instrument>          # scaffold (idempotent, atomic)
python -m framework.pipeline record   <market> <instrument> <stage> --path P --kind doc|code|data|run|ref --note "..." [--seed N --data-ref R]
python -m framework.pipeline run      <market> <instrument> <stage> --run-id ID --status SHIP --params '{}' --metrics '{}'   # the experiment-run registry (runs.jsonl)
python -m framework.pipeline gate     <market> <instrument> <stage>                       # RUNS the stage's gate_spec (machine-checked)
python -m framework.pipeline gate     <market> <instrument> <stage> --manual-override --evidence "..."   # human-attested (passed_by=human, auditable)
python -m framework.pipeline advance  <market> <instrument>          # next stage (only if current gate passed)
python -m framework.pipeline status   <market> <instrument>          # render one workspace (with lineage tags)
python -m framework.pipeline registry                               # render + write the global index
python -m framework.pipeline doctor                                 # store-accuracy gate: every path + ref resolves, manifests well-formed
```
**Gates are MACHINE-CHECKED, not asserted** (the SOTA fortification — see [PIPELINE_SOTA_AUDIT_2026_06_09.md](PIPELINE_SOTA_AUDIT_2026_06_09.md)):
stage 02's gate RUNS `check_invariants.py` (CDAP); stage 03's gate passes iff a SHIP run exists in `runs.jsonl` (which
came from `candidate_gate`); the others are human-attested with `--manual-override` (stamped `passed_by=human` +
evidence + lineage). **Every record/gate/run captures a `lineage` block** (`git_sha + dirty + artifact_sha256 + python +
seed + data_ref + ts`) so any result is reproducible/attributable. Writes are atomic (tmp+os.replace) + lock-guarded.
(Runs from the repo root via the `src` source-root `.pth`; or `python src/framework/pipeline.py ...`.)

### The discovery preflight (stage-03 entry gate — coverage + config canonicalization)
Stage 03 (strat build) runs a **discovery preflight** ([`src/framework/discovery_contract.py`](../src/framework/discovery_contract.py))
*before* `candidate_gate`, so every discovery campaign is **dimensionally complete + search-space-canonical by
construction** — the agent never silently misses a timeframe or tests only mechanical (not strat-based) exits, and
never wastes budget on near-duplicate configs. It *consumes the stage-00 registries* (the decomposition lattice):
- `coverage_report(declared, waivers)` — checks the run's declared axis coverage against
  [`config/strategy_dimension_registry.yaml`](../config/strategy_dimension_registry.yaml) (+ the TI/factor catalogs);
  a registry member that is **neither tested nor waived** is a WARN (silent omission), an undeclared run FAILs. It
  forces *conscious declaration*, not exhaustive testing — you waive what "we've moved from", so nothing is forgotten.
- `canonicalize_grid(configs, rel_tol)` — collapses near-duplicate configs (MA(28,29) ≈ MA(27,30)) to
  mutually-separated representatives + reports the honest effective-N for multiple-comparison accounting.

This is the mechanization of the user mandate (2026-06-11): *"I don't want models missing such things as timeframes,
or gaps such as strat-based exit vs mechanical exits — these should be aspects we have moved from … the intelligence
part is what should be our focus."* RWYB: `python src/framework/discovery_contract.py --selftest` (5/5).

## Market-agnosticism (the contract)
The pipeline is generic because each stage is a **contract**, not a crypto-specific step. The ONE thing a new market
must supply is the **[`MarketAdapter`](../src/framework/adapter.py)** contract (data loader + cost model + universe +
cadences + feature-families); everything else is shared:
- **Data adapter** (per market): produces the market's feature panel. Crypto = chimera (`pipeline.chimera_loader`);
  stocks = an equities OHLCV+fundamentals+microstructure loader implementing the same `load(symbol, cadence)` shape.
- **Decomposition lattice** (shared): the 8 constituent axes (chart-type, cadence, instrument, signal, regime, method,
  portfolio, entry/exit + actor/sector lenses) apply to any market.
- **Validation apparatus** (shared, market-agnostic): `candidate_gate` (random-entry firewall null, beat-passive,
  block-bootstrap p05, jackknife, UNSEEN-once, positive control) works on *any* market's trade book — only the cost
  model differs.
- **Meta-layer + autonomy** (shared): the firm-harness + the `/orc` loop run any market's pipeline.
To onboard a new market: `init <market> _market`, plug a data adapter, run stages 00→06. The dead-list, gates, and
storage carry over; only the data + cost model are new.

## Where crypto is right now (honest current state)
Per the registry: **crypto/_market is at stage 03_strat (3/7 done)** — research (00), mining (01), and engine (02) are
COMPLETE and gate-passed; strat/bot/execution/deployment are NOT started (deliberately — we are building the research/
ingredient layer, no strategy yet). `crypto/BTCUSDT` inherits 00–02 from `_market` and sits at 03. `stocks/_market` is
a fresh stub at 00, demonstrating the model is generic. The A/B/C fork (accept beta+yield ceiling vs chase the
sub-bar/info-bar frontier — see `docs/MARKET_FRAMEWORK/05_OPEN_THREADS.md`) is the decision that unblocks stage 03.

## Why this is the answer to the three asks
1. **Repeatable:** the same 7-stage gated loop runs for every market/instrument; the CLI mechanizes it; gates are
   machine-checked (a false pass is no longer free).
2. **Not disparate:** the manifest + run-registry + global registry are the single store — every artifact/run is
   recorded against its stage; `doctor` proves the store accurate (paths + refs resolve, manifests well-formed).
3. **Market-agnostic:** stages/gates/storage/apparatus are generic; only the `MarketAdapter` (data + cost) is
   per-market — so crypto, stocks, or any market flow through the identical working model.
4. **Reproducible (SOTA):** every record/gate/run binds the result to its `lineage` (code SHA + artifact hash + env +
   seed + data_ref + time); the store is atomic + crash-safe. See [PIPELINE_SOTA_AUDIT_2026_06_09.md](PIPELINE_SOTA_AUDIT_2026_06_09.md)
   for the audit vs world-class quant/MLOps/autonomous-discovery frameworks + the honest remaining gaps.
