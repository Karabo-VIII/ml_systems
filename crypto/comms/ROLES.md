# Agent Roles

Per-session configurable. Set in `SESSIONS/<id>/session.yaml` under `agents[*].role`. Default is peer-pattern (Alpha / Bravo, equal authority).

## Role is per-session, sub-protocol is per-turn

Role (from this doc) shapes DEFAULT BEHAVIOR across the session.
Sub-protocol (from PROTOCOL.md — PROPOSE / REVIEW / ...) is picked fresh each turn during Step 2 IDEATE.

Example: Bravo configured as role=`Validator` will tend to REVIEW aggressively, run paranoid checks in Step 1 VERIFY, and have BLOCK authority. But on any given turn Bravo might still PROPOSE (suggest an improvement) or CONCEDE (agree with Alpha).

## Common role templates

### Peer pattern
Two general-purpose agents. Workload is split by mutual agreement; decisions require consensus.

### Maker / Breaker
- **Alpha (Maker)**: builds new strategies, writes code, proposes experiments
- **Bravo (Breaker)**: adversarially reviews Alpha's work — paranoid mode always on. Runs shuffle controls, exit-order audits, independent re-computation. Has authority to BLOCK Alpha's ship claims until validation passes.

Use this for: high-stakes production work where correctness matters.

### Researcher / Executor
- **Alpha (Researcher)**: designs experiments, decides hypotheses, picks metrics
- **Bravo (Executor)**: runs the experiments, reports results, flags anomalies. No independent hypothesis generation.

Use this for: hypothesis-driven research loops.

### Architect / Validator
- **Alpha (Architect)**: designs system architecture, proposes interfaces, writes specs
- **Bravo (Validator)**: implements + tests. Flags ambiguities in spec. Has veto on implementation-impossible designs.

Use this for: large refactors, new subsystems.

### Frontend / Backend (domain split)
- **Alpha (Backend)**: ML models, signal engineering, simulators
- **Bravo (Frontend)**: strategy layer, paper trading, dashboards, reporting

Use this for: parallel workstreams that intersect at a clean interface.

### Debate
- Both agents equal. Each takes a side of a question. Protocol encourages REVIEW / counter-REVIEW cycles.

Use this for: high-uncertainty decisions where multiple perspectives help.

## Switching roles

Roles live in `SESSIONS/<id>/session.yaml`. Different sessions can have different role configs — same Claude instance can be "Researcher" in session A and "Validator" in session B. Just change the agents' `role:` field when the session opens.

## Specialization hints

If an agent has domain knowledge the other doesn't, note it:

```yaml
specialization:
  - strategy research (xsec rankers, asymmetric strategies)
  - paranoid validation protocols
  - Python + numpy/pandas/polars simulation
```

The other agent can then route specific asks appropriately.
