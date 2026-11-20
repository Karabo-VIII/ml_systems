# wealth_bot — PEPE 4h EMA/MA Bot Framework

Two parallel bot families share the framework:

- **ML-gated**: LGBM signal-picker reads chimera features, gates the raw MA-cross signal.
- **Pure-static**: MA-cross + whale/depth filter, no ML.

## Development methodology — 2-phase (binding)

**Phase 1 — Robust Discovery**: pick (Instrument, Indicator, Approach), audit (10-seed, all-4-windows positive, p05>0, max_dd<30%), ship verified baseline.

**Phase 2 — Oracle-Augmented Refinement**: mine trade-decision context (before/during/after each fire), surface refinement hypotheses, validate HONESTLY on prior windows only, test UNSEEN as final holdout. Cross-window persistence test is the discipline that prevents shipping small-sample noise.

**Full spec**: [docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md](../../docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md). Binding for any new wealth-bot.

## Current deploy candidate (2026-05-25, post-rebuild chimera)

**ML primary**: [`pepe_ema_bot_static_1strat.yaml`](configs/pepe_ema_bot_static_1strat.yaml)
- Strategy: EMA_cross(7, 15) + whale_net>0
- Gate: LGBM (10-seed ensemble, threshold 0)
- Sizing: quarter-Kelly on rolling equity, max position 1.0
- Risk: DD trip 25%, consecutive-losses trip 10
- Window split: TRAIN 2023-05-06 → 2024-05-15, VAL → 2025-03-15, OOS → 2025-12-31, UNSEEN 2026-01-01 → 2026-05-22
- UNSEEN compound: **+51.86% (ensemble) / +51.86% median / +50.66% p05 / 10/10 seeds positive**
- Engine-equity vs Kelly-managed paper-trade journal: +14.23% (quarter-Kelly on $5K)

**Alternate (higher upside)**: [`pepe_ema_bot_static_2strat_ortho.yaml`](configs/pepe_ema_bot_static_2strat_ortho.yaml)
- Two strategies: EMA(12,26)+whale, EMA(20,30)+bd_imb>med (book-imbalance > 30d rolling median)
- LGBM-gated picker chooses the higher predicted-return strategy per bar
- UNSEEN compound: +55.45% ensemble / +31.95% p05
- OOS more robust (10/10 positive) than 1-strat (4/10), wider seed dispersion

## Canonical paths

| Layer | File |
|---|---|
| Config schema | [`framework/config.py`](framework/config.py) |
| Data loader (chimera + filters) | [`framework/data_loader.py`](framework/data_loader.py) |
| LGBM picker | [`framework/signal_picker.py`](framework/signal_picker.py) |
| Ensemble + threshold | [`framework/upgrades.py`](framework/upgrades.py) |
| Multi-seed audit | [`framework/walk_forward.py`](framework/walk_forward.py) |
| Paper-trade bot | [`bot/runner.py`](bot/runner.py) |

## Entry points

| Task | Script |
|---|---|
| Train + audit | `scripts/wealth_bot/train_and_audit.py --config <yaml> --ablation full` |
| Paper-trade replay | `scripts/wealth_bot/run_paper_trade.py --audit-json <path> --segment UNSEEN` |
| Pure-static control | `scripts/wealth_bot/pure_static_baseline.py` |
| Deploy-layer breakdown | `scripts/wealth_bot/deploy_layer_breakdown.py` |
| Firing + coverage | `scripts/wealth_bot/firing_coverage_analysis.py` |
| Allocation methods | `scripts/wealth_bot/allocation_method_ranking.py` |
| Monthly breakdown | `scripts/wealth_bot/monthly_breakdown.py` |

## Doctrine

Optimize for **wealth** (compound return), not Sharpe. Robust = (a) 10/10 seeds positive on UNSEEN, (b) block-bootstrap p05 > 0, (c) max DD < 30%.

Full record: [PROJECT_NORTH_STAR.md §3.1](../../PROJECT_NORTH_STAR.md).
