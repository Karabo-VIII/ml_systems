"""Prong 3 -- multi-modal alignment for the foundation backbone.

Adds a cross-attention adapter on top of the FROZEN foundation backbone
that conditions on side channels:
    - funding rate (chimera norm_funding)
    - open-interest delta (chimera norm_oi_change)
    - macro: DXY / S&P / BTC dominance (daily, prev-close lagged)
    - on-chain: stable mints, exchange netflow (DefiLlama, daily)
    - news embeddings: pre-computed sentence-BERT vectors (lagged)

Per LITERATURE.md Hole 6: every channel carries an explicit lag (default
1 bar = 5 min) so the model never sees same-bar-close information. Walk-
forward purge gap is configured to >= longest lookback used.

Workflow:
    channels.py    -- ingest + align side-channel data per timestamp; lagged
    adapter.py     -- cross-attention adapter (~2M params) on frozen backbone
    finetune.py    -- training loop (foundation frozen, adapter trains only)
"""
