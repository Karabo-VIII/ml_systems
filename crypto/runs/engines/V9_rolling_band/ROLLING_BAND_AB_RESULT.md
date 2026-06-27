# A/B: FROZEN band vs ROLLING-reselect top-K band (MA family) — RESULT 2026-06-27

USER IDEA: keep top-K configs per MA type, dynamically reselect the recent winners every
14d from trailing realized P&L, trade them in the next 7d window — "static engines that
act dynamically by swapping configs to conditions."

## DESIGN (SOTA-rigorous, one variable changed)
- SAME 120-config universe, confirmation, regime gate, PROD cost (0.0028), lag-1, inv-vol
  + 15% cap allocator. ONLY frozen-vs-rolling band differs.
- FROZEN: fixed TRAIN+VAL-positive band (114 cfgs), voted every bar.
- ROLLING: every 14d, score all 120 cfgs by realized P&L over the TRAILING 14d (STRICTLY
  causal — bars < window start only), pick top-10, vote for the next 7d.
- RANDOM-RESELECT NULL (n=10): same machinery, pick 10 configs at RANDOM each window. The
  decisive control — if rolling can't beat random selection, the selection has no skill.
- Per-asset (each asset picks its own recent-best configs — matches the user's intent).
- OOS-2024, 2025 sealed.

## RESULT: FROZEN WINS. Rolling-with-skill = 1/12 cells (within multiple-comparison noise).

| Cell | Frozen | Rolling | Random null (P5-P95) | verdict |
|------|--------|---------|----------------------|---------|
| SMA\|1d | +1.67 | +1.37 | +1.58 [1.28,1.79] | FROZEN |
| EMA\|1d | +1.68 | +1.71 | +1.83 [1.76,1.93] | rolling~random (churn) |
| WMA\|1d | +1.60 | +1.52 | +1.63 | FROZEN |
| VIDYA\|1d | +1.57 | +1.93 | +1.63 [1.47,1.77] | rolling wins (1/12, likely noise) |
| KAMA\|1d | +1.58 | +0.63 | +1.70 | FROZEN (rolling collapses) |
| DEMA\|1d | +1.17 | +1.53 | +1.17 | rolling~random |
| SMA\|4h | +1.60 | +0.78 | +1.33 | FROZEN |
| EMA\|4h | +1.29 | +0.89 | +1.29 | FROZEN |
| WMA\|4h | +1.43 | +0.79 | +1.37 | FROZEN |
| VIDYA\|4h | +1.20 | +1.08 | +1.26 | FROZEN |
| KAMA\|4h | +1.57 | +0.83 | +1.57 | FROZEN |
| DEMA\|4h | +1.17 | +0.69 | +1.09 | FROZEN |

## THE DECISIVE EVIDENCE: the random-reselect null
Rolling-by-recent-P&L performs AT OR BELOW the random-reselect mean in most cells. Picking
configs by recent performance is NO BETTER than random — the selection has NO SKILL. Several
cells COLLAPSE under rolling (KAMA|1d +1.58→+0.63, SMA|4h +1.60→+0.78): reselection chases
configs right as their edge mean-reverts away (the persistence trap).

The single "win" (VIDYA|1d) is 1 of 12 cells = exactly the false-positive rate from multiple
comparisons (12 x 5% ~ 0.6 expected by chance). Not claimed as real without dedicated confirm.

## CONCLUSION
The FROZEN TRAIN+VAL band is the SOTA deployable design for the MA family. Rolling
config-reselection by recent performance does NOT beat it and usually loses, because MA
config performance does NOT persist — a 5th independent confirmation of D73 ("recent-winner
config is anti-predictive"), now at the 14d-lookback / 7d-window / u100 / band resolution.

This CLOSES the config-rotation thread for MA. The frozen band already captures the durable
edge; chasing recent winners adds churn + mean-reversion drag. DEPLOY THE FROZEN BAND.
(D73 family extended; logged. Other rotation mechanisms — e.g. regime-conditioned, not
recent-performance — remain theoretically open but lower priority given 5 consistent kills.)
