# TRAIN / 2020 / January — MA building-block exercise

**Period:** 2020-01-07 → 2020-02-07 (the oldest month we have; data starts 2020-01-07).
**Split:** TRAIN (< 2024-05-15). **Universe:** u10 (7 assets live this early).
**Exercise:** full-funnel MA decomposition — all 306 distinct 2MA + 1,466 distinct 3MA through each
asset × 4 cadences (4h/1h/30m/15m) × 3 exits (signal-flip/trail-5/trail-10) = 137,766 cells.

## Contents
- `analysis/MA_BUILDING_BLOCK_2026_06_12.md` — the what-worked/didn't writeup (also in `docs/`).
- `charts/` — the figures:
  - `ma_building_block_analysis.png` (6-panel), `ma_killers.png` (killer map + whipsaw share)
  - `ma_mechanics.png` (taker), `ma_mechanics_maker.png` (maker lever)
  - `ma_moves_<ASSET>.png` ×7 (best 2MA+3MA per timeframe on price), `ma_equity_<ASSET>.png` ×7 (price+equity+exit overlay)
- `raw/` — the run JSONs (gitignored, regenerable): `ma_per_instrument_*`, `ma_mechanics*`, `ma_best_config_per_tf`.

## Headline findings (this period)
- Best config per tf is 7/7-positive, converges to a ~1-week wall-clock hold (4h ema_8_28 +23.9%, 1h ema_22_108 +24.7%, 30m ema_37_203 +24.5%, 15m ema_186_208_233 +21.7%).
- One dominant killer: over-trading via MA-speed/cadence MISMATCH → cost drag (gross flat ~10-13%, cost drag 0.8→10.5% taker; 15m net −2.3%).
- Maker flips it: all 4 cadences net-positive at maker (15m −2.3% → +5.4%).
- Whipsaw = ~10-16% of cost drag (the cooldown target); the rest is sheer trade count.

## Reproduce
`python -m strat.ma_per_instrument --max-configs 1500` → `ma_mechanics [--maker]` → `ma_killers` →
`ma_analysis_plots` / `ma_move_grid` / `ma_equity_grid`. All take `--start 2020-01-07 --end 2020-02-07`.
