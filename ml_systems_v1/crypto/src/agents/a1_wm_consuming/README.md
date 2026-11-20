# A1 -- WM-consuming agents (`__class_tag__ = "A1"`)

**Class:** an A1 plans/acts over a FROZEN forecaster's outputs. It does NOT
build or train the world model -- it consumes one someone else froze, via the
`ForecastBundle` contract (`src/wm/forecast_bundle.py`).

**KPI:** held-out **compound** return over a FROZEN forecaster. Ship ONLY if it
beats the predict-then-rule baseline (B0) on UNSEEN compound by the
pre-registered `should_promote` margin, with >=8/10 seeds positive + bootstrap
p05>0 + maxDD<30%.

**GIGO exposure: HIGH** -- it inherits the forecaster's defects. The firewall:
- the forecaster is FROZEN (no `F.train()`, no F-params in the optimizer, all
  bundle tensors `.detach()`'d) -- CDAP `forecaster_frozen_in_agents`;
- the critic regresses to REALIZED return (`target_return_*`), NEVER the
  forecaster's decoded `return_logits` -- CDAP `no_predicted_return_as_realized_reward`.

**Contents:** the half-built DreamerV3 A1 (`dreamer_v3_agent.py`, `environment.py`,
`policy.py`, `ppo.py`, `sac_agent.py`, `config.py`, `rewards.py`, ...) moved from
`src/agent/`; `backbones/` holds the A1 backbones V16 (DreamerV3) and V17
(TD-MPC2), reclassified out of the forecaster zoo.
