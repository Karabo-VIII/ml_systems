"""Kronos baseline -- E1 from FRONTIER_RESEARCH_RESPONSE_2026_05_02.md.

Kronos (Shi et al., AAAI 2026, arxiv:2508.02739) is a finance-specialized
foundation model: decoder-only autoregressive on K-line (OHLCV) sequences,
pretrained on 12 billion K-line records from 45 exchanges. MIT-licensed,
HuggingFace-hosted under NeoQuasar/Kronos-{mini,small,base,large}.

This module's job: run Kronos-small zero-shot on our chimera_legacy 10-asset
OOS segments and measure IC vs V1.1's 0.067 baseline. The result determines
whether we PIVOT Prong 1 (foundation) from "scratch pretrain on 4060" to
"finetune Kronos on our 10-asset corpus" -- saving ~30 GPU-hours.

Files:
    install_check.py  -- verify kronos package available + smoke-tokenize 1 bar
    eval_kronos.py    -- run zero-shot on OOS, compute IC vs V1.x baseline
"""
