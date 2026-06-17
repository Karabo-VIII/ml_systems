# A2 -- raw-data self-evolving agents (`__class_tag__ = "A2"`)

**Class:** ingests raw price/bars/TIs directly and learns representation + policy
end-to-end. It is its own implicit world model -- it does NOT consume a frozen
forecaster's `ForecastBundle`.

**KPI:** held-out **compound** return from raw data. Same gate as A1 MINUS the
source-forecaster clause (there is no F), PLUS a policy-overfit control: the
shuffled/surrogate-market test (a genuine policy -> ~0 on phase-randomized
returns; a memorizer still "profits" -> that residual is the policy-side ShIC=0
signature).

**GIGO exposure on the WM axis: NONE** -- A2 removes the forecaster amplifier.
It cannot manufacture signal that is not in the bars (Ceiling 2 still binds).

**Status this phase: EMPTY SCAFFOLD** -- no A2 code is moved or written in
Phase 0. The `src/agent/` cluster is entirely WM-coupled (A1); nothing belongs
here yet.
