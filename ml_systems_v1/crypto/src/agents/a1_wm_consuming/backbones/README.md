# A1 backbones

World-model architectures an A1 planner imagines/plans over. Reclassified here
from `src/wm/` (the forecaster zoo) -- they are A1 substrates, not forecasters,
and were never registered in `wm_tournament.py`.

- `v16_dreamerv3/` -- DreamerV3 RSSM (was `src/wm/v16`).
- `v17_tdmpc2/`    -- TD-MPC2 decoupled WM + MPPI (was `src/wm/v17`).

`src/wm/v16` and `src/wm/v17` now contain only a `MOVED.md` tombstone; CDAP
`v16_v17_not_in_wm` hard-fails if model code reappears there.
