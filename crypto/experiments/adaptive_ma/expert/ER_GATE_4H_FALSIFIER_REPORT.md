# ER-gated fixed-MA @ 4h vs REGIME-MATCHED null — FALSIFIER REPORT

**Auditor RED-team task:** prove the ER-gated fixed-MA @ 4h (ATR-trail exit) BEATS a regime-matched null
(random entries drawn ONLY from inside the same ER>0.4 windows, same hold-time distribution, same cost)
AND passes a positive-control power-check @4h. If it does not beat the gated-random null, the edge is
regime/beta, not timing.

**Verdict: REFUTED.** On held-out data the ER-gated fixed-MA @ 4h beats the regime-matched null on
**0 of 77** u100 assets — and across **6 MA/threshold configs** the gated-RANDOM null is *positive* on
OOS while the real MA-gated entry is *negative*. The MA entry timing does not add value; it subtracts it.
The return that exists is **regime/beta** (being long inside ER>0.4 windows), not entry timing.

## Strategy under test (minimal honest config, fixed up-front per RESEARCHER_REPORT_1 redirect)
- Cadence **4h**; fixed MA **8/21 EMA** ("trend" pair); ER gate **Kaufman ER(20) > 0.4**.
- Entry (primary, "state"): LONG when `fast>slow AND er>0.4`, confirmed at close, filled next-bar-open.
- Exit: **ATR-trail ×3.0** (ATR-14) + 42-bar (7-day) time cap. ONE policy. Taker **0.0024** round-trip.
- All past-only (MA/ER close-of-bar vs next-open fill; ATR read as `atr[j-1]`).

## Regime-matched null (the falsifier)
`strat.firewall.random_entry_null(regime_matched=True)` with `harness.spec.filter_col='er', op='gt',
val=0.4` → the null draws random entries **only from bars where ER>0.4** (the SAME regime the strategy
gates on), holds for durations sampled from the strategy's OWN trade distribution, at the SAME cost. This
isolates WITHIN-regime entry TIMING from gate/regime SELECTION. The strategy's entries are a strict subset
of the null's eligible bars.

## Results (all RWYB-reproduced 2026-06-05, real chimera 4h)

**Positive-control power-check @4h** (`positive_control_4h.py`, synthetic, no market claim) — runs the
EXACT falsifier apparatus on a price with a GENUINE within-ER>0.4 long-only timing edge (clean UP & DOWN
trends both pass the gate; long-only MA captures only UP; random ER>0.4 entries catch both):
- OOS real **+431.25%** vs null_p95 **+26.97** → beats_null **True**
- UNSEEN real **+123.20%** vs null_p95 **+8.35** → beats_null **True**; gated-random null *negative* there.
- → **HAS POWER**: the regime-matched firewall DETECTS a real 4h within-regime timing edge. A null on
  real data is therefore a real refutation, not a dead firewall. **PASS.**

**Main falsifier @4h, u100, state-based** (`er_gate_4h.py --entry state`, 77 assets):
- **0/77** assets beat the regime-matched null AND positive on held-out (OOS+UNSEEN).
- beats-null flag (clears null p95 on both held-out windows): **0/77**.
- OOS real compound: mean −8.92%, **median −12.99%**. UNSEEN: mean +6.85% (outlier-skewed), median −0.7%.
- BTC example: OOS real **+1.31%** is *below* the regime-matched null **p50 +4.43%** — random ER>0.4
  entries out-earn the MA-timed entry.

**Robustness sweep** (6 configs × 25 assets, regime-matched null):

| MA / ER thresh | beat_null & pos_held | OOS real median | OOS gated-null p50 median |
|---|:--:|--:|--:|
| 8/21, ER>0.4  | 0/25 | −9.75%  | **+11.22%** |
| 5/20, ER>0.4  | 0/25 | −9.75%  | **+9.02%**  |
| 10/30, ER>0.4 | 0/25 | −8.11%  | **+8.17%**  |
| 8/21, ER>0.3  | 0/25 | −12.68% | **+4.27%**  |
| 8/21, ER>0.5  | 0/25 | −5.08%  | **+6.70%**  |
| 20/50, ER>0.4 | 0/25 | −4.02%  | **+7.72%**  |

In **every** config the gated-RANDOM null median is **positive** while the real MA-gated median is
**negative** → the MA entry timing actively underperforms random selection within the same regime (it
enters late / into reversals).

**Leak check** (`SetupHarness.leak_guard`, BTC/ETH/SOL): INSUFFICIENT_EDGE (negative held-out base — no
positive edge to leak) → the negative results are genuine, not a look-ahead artifact; the structural
next-bar-open fill guarantees past-only.

**Cross-EVENT variant** (faithful "trade the cross only when ER>0.4"): too sparse to power (2 UNSEEN
trades across 13 assets — a fresh cross follows chop, where ER<0.4), but still 0/13 beat. State-based is
the powered test.

## Conclusion
The redirect (ER-gate + 4h + ATR-trail) was tested on its OWN recommended terms and is **REFUTED**: the
ER-gated fixed-MA @ 4h does not beat a regime-matched null on any of 77 assets or any of 6 configs. The
edge is **regime/beta**, not timing — confirmed two-sidedly because the positive control proves the same
firewall WOULD detect a genuine 4h within-regime timing edge.

## RWYB reproduction
```
python src/strat/selftest_all.py                                              # apparatus 4/4 PASS
python experiments/adaptive_ma/expert/positive_control_4h.py                  # firewall HAS POWER @4h (PASS)
python experiments/adaptive_ma/expert/er_gate_4h.py --entry state             # full u100: 0/77 beat regime-null
python experiments/adaptive_ma/expert/er_gate_4h.py --probe BTCUSDT --entry state   # single-asset detail
```
Artifacts: `er_gate_4h.py`, `positive_control_4h.py`, `er_gate_4h_u100.json`, `er_gate_4h_quick.json`,
`positive_control_4h.json`.
```
SAFETY: no commit/push/deploy/capital — analysis + JSON only, under experiments/.
```
