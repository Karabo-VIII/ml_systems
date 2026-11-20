# Calibrated Uncertainty Protocol

> Every numerical / factual claim in user-facing output must carry a
> provenance tag. Extends and HARDENS the existing
> `memory/feedback_search_reliability_protocol.md`.

## The three tags

| Tag | Meaning | Example |
|---|---|---|
| **VERIFIED** | I just confirmed this number by reading the source. Reproducible right now. | "BTC 2026 starts $88,848 (read from `btcusdt_v51_chimera_1d_*.parquet`)" |
| **REPORTED** | Cited from a project memo/doc/git commit. Was true when written; may be stale. | "v6_frontier Sh 4.28 [REPORTED, per `frontier_item_etf_flow_ship_2026_04_23.md`]" |
| **INFERRED** | My estimate from first-principles / model knowledge / triangulation. Not measured. | "Per-event 1-2% EV per signal [INFERRED, based on similar-class published research]" |

## When to tag

**Every number, ROI, Sharpe, hit-rate, t-stat, threshold, file count, or factual
claim about the codebase in user-facing output requires a tag.**

Exceptions where tags can be omitted:
- General math operations ("if 5 sleeves each contribute 0.2%/day, blend is +1%/day")
- Standard programming facts ("py_compile checks syntax")
- Anything the user just said in the current turn

## Failure mode this prevents

Today's `META_ROI_SYNTHESIS_2026_05_13.md` mixed measured numbers
(`CONSERVATIVE Sh 11.5-13.1`) with aspirational numbers (`+5-10%/yr lift over
naive carry`) without distinguishing. A reader (including future-me) can't
tell which to trust. Tags fix this.

## Required output discipline

In bullet points / tables:
> "FIL 4h SMA(22,27) taker 0.81% / sharpe 0.11 [VERIFIED, wf_robust_v2 run 2026-05-13]"

In prose:
> "The xsec K=5+5 ranker shows Sharpe 3.36 walk-forward [REPORTED] but live
> capacity is INFERRED to be ~$2-5M based on typical TIER_B ranker scaling."

In tables — add a "src" column or footnote:
| Sleeve | Sharpe | Src |
|---|---|---|
| xsec K=5+5 | 3.36 [V] | `xgb_ndcg_correction_2026_04_22` |
| frontier v6 | 4.28 [R] | `frontier_item_etf_flow_ship_2026_04_23` |
| MEV bundle timing | 2-4%/d [I] | research_scout 2026-05-13 |

## Stale-REPORTED handling

A REPORTED number is only as fresh as its memo. If the memo is > 30 days old
AND the claim is load-bearing for a decision, RE-VERIFY before citing. Today's
synthesis claimed `v6_frontier ORPHANED` based on a 2-day-old memo; the rewire
happened 1 day after the memo. **Always re-verify load-bearing REPORTED claims.**

## INFERRED honesty

When INFERRED is the only available source, say so explicitly:
> "MEV bundle timing 2-4% daily alpha [INFERRED — no project measurement;
> based on Jito Labs reported retail RFQ flow]"

Never tag an INFERRED number with VERIFIED.

## Compatibility with existing protocols

This protocol extends `memory/feedback_search_reliability_protocol.md`. If
that file says "tag numerical claims," this file says "ALWAYS tag, here's the
3-letter shorthand."

## Self-check before submitting any user-facing report

1. Grep my own draft for digits-followed-by-percent or "Sh " or "+/-"
2. Confirm every match has a tag
3. Confirm every REPORTED tag's memo is < 30 days old or has been re-verified
4. Confirm every INFERRED tag is honest about not being measured

If any check fails — fix before submitting.
