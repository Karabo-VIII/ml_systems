# Sonnet scout hallucinated xsec ranker orphan

**Date observed**: 2026-05-13
**Severity**: high (Sonnet output drove an incorrect strategic claim)
**Frequency**: one-off but consistent with known Sonnet failure mode

## Context

During /meta synthesis, I delegated to a Sonnet Explore scout to inventory
the alpha-bearing pillars. The scout reported:
> "xsec K=5+5 FULL ranker — Sh 3.70, +161%/yr — Model trained at
> `models/xsec_ranker/xsec_ranker_v1.pkl`; NOT wired into live paper_trader_v2"

I accepted this claim and propagated it into `META_ROI_SYNTHESIS_2026_05_13.md`
as a high-priority orphan.

## What went wrong

The claim was wrong. The production stack uses
`xgb_ndcg_v1_u4h_v0_base_48feat` (in `models/xsec_ranker/`) which IS the K=5
ranker, and it's wired into CONSERVATIVE / PRIME / AGGRESSIVE / V6_FRONTIER /
multiple other blends. `xsec_ranker_v1.pkl` is an OLDER standalone file, not
the production K=5+5 model.

Sonnet conflated two different `.pkl` files in `models/xsec_ranker/` and
treated the older one as authoritative.

## Root cause

I trusted the Sonnet scout's claim without verifying. The scout's report
listed file paths and model names but I didn't:
1. Open the actual `production_blends.yaml` and grep for "xsec" — would have
   shown `4h_K5_h32_sleeve` (the production K=5)
2. Compare the file names — the scout said `xsec_ranker_v1.pkl` but the
   production rankers are named `xgb_ndcg_v1_*.pkl`
3. Run `BlendComposer` to validate which models are actually consumed

## How to apply

1. After any Sonnet scout return, run the verification protocol in
   `agent_protocols/sonnet_integration_safety.md`:
   - Numerical/path claims: re-read source
   - "Done/orphan" verdicts: run the actual validator
   - File-name claims: `ls` to confirm
2. Treat Sonnet output as HYPOTHESIS not FACT
3. When in doubt about <4 file scope, skip Sonnet entirely

## Specific Sonnet failure mode catalogued

Sonnet conflated two similar-named .pkl files. This is the "lexical similarity
masks semantic difference" failure mode. Watch for this pattern:
- File X and file Y have similar names → Sonnet may not distinguish them
- Different model versions with similar names → check explicit version tags

## Related

- `memory/agent_protocols/sonnet_integration_safety.md` — verification protocol
- `memory/research_delegation_protocol.md` — 4-agent delegation
- Failure mode catalogued in INDEX of sonnet_integration_safety.md as
  "Sonnet conflated xsec_ranker_v1.pkl with xgb_ndcg_v1_u4h_v0_base_48feat.pkl"
