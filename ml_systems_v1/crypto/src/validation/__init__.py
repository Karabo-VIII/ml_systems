"""Validation gates -- run AT layer boundaries, not just at end of pipeline.

Three gate kinds:
  pipeline_gates: post-build per-asset asset validation (schema, NaN budget,
                  freshness, registry consistency).
  contract_gates: enforce data_api contracts (e.g. v51 must carry te_*, xd_*
                  to be safe for FEATURE_LIST_127+ training).
  model_gates:    post-train (loss curves, IC, ShIC, hallucination check).
                  TODO: extract from anti_fragile / validation_utils.
  strategy_gates: post-backtest (DSR, PBO, cost realism). TODO.

Layer-boundary discipline: each gate runs INSIDE the producer (fail-fast on
the build) AND when the consumer opens the data (defense-in-depth). This
catches the silent-NaN class of failures we hit on te_panel.
"""
