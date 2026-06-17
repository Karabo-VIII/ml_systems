# FINDING — the chart-type axis (imbalance bars) + a NEW order-flow directional micro-edge (2026-06-13)

> Triggered by the user: "did you check the whole data landscape -- chimera, chart types?" Honest answer:
> NO -- the 4-lane mover sprint ran on 1m TIME bars only. The project HAS chimera on alternative bars
> (dollar=full 104 assets; DIB=Dollar-Imbalance + range = 3-asset pilot; runs_tick/volume = NOT built).
> So "movers dead" was scoped to time bars. Tested the most-promising un-swept axis: DIB imbalance bars.
> Scripts: `src/mining/dib_mover_axis_v2.py` (honest; v1 quarantined -- had an outcome-conditioning artifact).

## The DIB / imbalance-bar re-test (3-asset pilot: BTC, ETH, PEPE), held-out
- **Test 1 -- "DIB triggers LEAD the move": REFUTED.** DIB inter-bar time genuinely compresses in bursts
  (p10 = 1-3s) but the flow signals do NOT lead: |flow_imbalance|/|vpin| OOS AUC 0.47-0.52 (coin-flip);
  the TIME-bar MOMENTUM baseline (0.58-0.65) leads BETTER. A same-sign flow run fires on 75.6% of ALL
  bars / 100% of days = zero specificity. Coincident, not leading -- same as the 1m time-bar sprint. (A
  v1 "+0.90 lead, 3/3" result was an ARTIFACT: definitional-lead + hindsight mover-day conditioning;
  quarantined.)
- **Test 2 -- continuation on DIB: PARTIAL (2/3).** Order-flow state discriminates continuation OOS
  (BTC 0.58, ETH 0.65, PEPE 0.58, all p<0.001 vs shuffled) -- beats the time-bar 0.52, clears 0.58 on 2/3,
  but modest/not clean across all 3.
- **Test 3 -- CAPTURE on DIB: the genuine survivor (3/3 OOS, NOT just momentum).** Trading the order-flow
  imbalance DIRECTION (|flow|>=p90, hold ~16 bars, cost-honest) is OOS net-positive 3/3 (BTC +0.18%,
  ETH +0.21%, PEPE +0.59%/trade). Survives the hardest adversarial cuts: threshold-robust + MONOTONIC
  (edge rises p50->p95), and MOMENTUM-ORTHOGONAL (trade the residual-flow after regressing out momentum:
  edge ~unchanged; corr(flow,mom)=0.12-0.22; on the ~20% flow-vs-momentum DISAGREE cases flow still wins).
  **The order flow carries DIRECTIONAL info orthogonal to price momentum -- the first directional edge of
  the session.**

## Verdict + honest caveats
- The user's LEADING-timing hypothesis is refuted; do NOT rebuild full-universe DIB to chase it.
- BUT the axis surfaced a REAL momentum-orthogonal order-flow directional capture edge (3/3 OOS,
  threshold-robust). It lives in the FLOW FEATURES (`norm_flow_imbalance`, `norm_vpin`), NOT the bar-type
  sampling -- so it should be reproducible on the FULL-universe TIME-bar chimera (which carries those cols),
  no DIB rebuild needed. The DIB sampling is not what creates the edge.
- COST-FRAGILE: BTC/ETH die at ~0.25-0.30% RT (on the maker cost cliff, p_fill 0.21-0.40); only PEPE to
  0.40%. n=3 (tiny). The per-day +2-6% sums are overlap-naive (6-9 simultaneous trades) -- NOT a
  compoundable curve; do not headline.

## THE DECISIVE GO/NO-GO RESULT -- DEAD (breadth + real cost killed it)
Ran it: `src/mining/flow_direction_breadth_probe.py`, u10, 4h, hold=16, momentum-orthogonal residual-flow,
REAL cost via the canonical `src/strat/fill_model.py` (taker cost_rt 24bps; maker p_fill 1.0/0.40/0.21
adverse 0.96). **VERDICT: DEAD. The order-flow-direction axis is closed.**
- The directional EXCESS over a cost-matched random-direction null is REAL but tiny: ~**+9.3 bps/trade
  gross** -- SMALLER than the 24bps taker round-trip. Net:
  **OOS mean -9.7 bps (4/10 positive) / UNSEEN mean -14.3 bps (5/10 positive)**.
- Up-rate (directional hit-rate) clusters at **0.47-0.56 = coin-flip**. The only net-positives are
  **2 of 10 names** (ETH +67 / DOGE +84 bps UNSEEN) = CONCENTRATION (the campaign's own firewall lesson,
  cf the Family2 +172% concentration catch). Big losers on the other side (XRP -106, SOL -77, BNB -71).
- **The pilot's headline DIRECTLY CONTRADICTED:** BTC residual OOS = **-42 bps at up-rate 0.468**, vs the
  pilot's "BTC +0.18%". The 3/3 DIB pilot was bar-sampling small-sample luck; it did NOT generalize.
- Robustness cuts all confirm dead: hold=4 net **-33.8 bps** (beats-null only 3/10 -- weaker than random
  OOS at short horizon); signed-VPIN UNSEEN **-4.5 bps** (fails the null); residual ~= raw (the
  momentum-orthogonality framing is NOT the issue -- the signal is simply sub-cost). Maker leg is
  catastrophic by construction (adverse 0.96 on a multi-candle hold = the D43/D70 "maker can't hold a
  directional multi-bar position" wall), so the honest realizable read is the taker leg, and it is
  negative. The canonical scorecard/battery (p05/jackknife/DSR) was correctly NOT run -- gated on
  surviving breadth+cost, which it did not; dressing up a net-negative 2-name signal would only manufacture
  a false positive.
- Recorded as **dead-list D75**. The MAGNITUDE/vol content of VPIN/flow may still serve as a SIZING input
  (cf the fizzle-filter AUC 0.70 + magnitude-continuation AUC 0.73 from the 4-problem sprint) -- never as a
  standalone DIRECTION signal. runs_tick/runs_volume bars remain genuinely UNBUILT (un-tested axis), but
  any edge there would live in flow features that ARE present on existing bars and were just falsified.
- Files: probe `src/mining/flow_direction_breadth_probe.py`; results
  `runs/mining/flow_direction_breadth_u10_4h_hold{16,4}.json` (+ a dollar-cadence corroboration run was
  launched as background robustness, not the crux -- the 4h falsification stands regardless).

## The honest landscape-coverage update (for the record)
The mover "DIRECTION dead" conclusion was scoped to TIME bars + the standard feature use. The chart-type
sweep is now: imbalance/range bars tested (3-asset pilot) -> leading-timing refuted, order-flow-DIRECTION
edge surfaced (cost-fragile). runs_tick/runs_volume bars are NOT built (un-tested). The full-universe
alternative-bar sweep would require a pipeline build; the order-flow edge does NOT need it (it's in the
features). Relates to [[project-mover-4problem-decomposition-2026-06-13]].
