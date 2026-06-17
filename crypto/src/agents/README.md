# `src/agents/` -- the ACTING-model layer (A1 / A2 / A1H)

The 3-class model taxonomy (machine-enforced; see the 4 CDAP invariants in
`config/_invariants.yaml` under `agent_taxonomy`):

| Class | Lives in | Consumes | KPI |
|---|---|---|---|
| **F** (forecaster) | `src/wm/v*/` (NOT here) | raw 41-dim chimera features | genuine learning (ShIC>0.015 + held-out IC>0 diagnostic) |
| **A1** (WM-consuming) | `a1_wm_consuming/` | a FROZEN forecaster's `ForecastBundle` | held-out **compound** over a FROZEN forecaster |
| **A2** (raw-data) | `a2_raw_data/` | raw price/bars/TIs directly | held-out **compound** from raw data |
| **A1H** (hybrid) | `a1h_hybrid/` | frozen bundle **AND** raw bars | held-out compound **AND must beat its own A2-ablation** |

Cross-cutting:
- `_shared/` -- genuinely class-agnostic helpers (policy/reward utils) shared
  across classes. Class-specific code stays in its class dir.
- `registry.py` (optional helper) -- class-aware writes to `runs/registry/`.

Hard rules (CDAP-enforced):
1. **F is FROZEN inside any agent** -- no `F.train()`, no F-params in an
   optimizer, no missing `.detach()` on a bundle tensor (`forecaster_frozen_in_agents`).
2. **A critic/reward target NEVER traces to a decoded `return_logits`** -- it
   must trace to a realized `target_return_*` (the GIGO firewall;
   `no_predicted_return_as_realized_reward`).
3. **Every agent-logic module declares `__class_tag__ in {A1, A2, A1H}`**
   (`agent_class_declared`).
4. V16/V17 are A1 backbones, NOT forecasters -- they live under
   `a1_wm_consuming/backbones/`, never `src/wm/` (`v16_v17_not_in_wm`).

The F->A1 contract is `src/wm/forecast_bundle.py` (the ONLY thing an A1 may
import from a forecaster -- frozen, detached, in `eval()`).
