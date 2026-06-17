# A1H -- hybrid agents (`__class_tag__ = "A1H"`)

**Class:** an A1 that ALSO peeks at raw data -- consumes a frozen forecaster's
`ForecastBundle` AND raw bars (a side-channel). The forecaster stays frozen.

**KPI:** held-out **compound** -- A1's full gate PLUS it MUST beat its own
A2-ablation (same agent with the forecaster channel removed) by a pre-registered
margin. **Non-shippable without the ablation, by rule.**

**Why A1H is a trap-detector, not just a class:** A1H-vs-A2-ablation is the
cleanest empirical test of whether the forecaster adds ANY value. If
A1H <= A2-ablation, the WM is proven useless on this task. An un-ablated A1H is
the most seductive overfit in the taxonomy -- it can always fall back to raw
data and credit the WM.

**Status this phase: EMPTY SCAFFOLD.**
