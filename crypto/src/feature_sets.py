"""Centralized feature-set contract for the entire model layer (V0-V19).

Single source of truth for FEATURE_LIST_* and get_feature_config(). Every
version's settings.py should re-export from here rather than duplicate the
definitions. This kills the drift problem where each version maintained its
own copy and they diverged subtly over time.

Naming convention:
    fN  = the N-feature subset, where N is the column count consumed by
          the model encoder.

Cumulative families (V50 schema, columns 0-40):

    f13   legacy core              (idx 0..12)
    f18   f13 + extended base      (+ ma_distance, whale, efficiency, return_4, return_16; idx 13..17)
    f21   f18 + Tier 1             (+ return_kurtosis, bar_duration, funding_momentum; idx 18..20)
    f25   f21 + Hawkes             (+ 4 hawkes_*; idx 21..24)
    f30   f25 + IC-boost           (+ 5 dynamics features; idx 25..29)
    f34   f30 + SOTA Tier 3        (+ yz_volatility, cs_spread, perm_entropy, kyle_lambda; idx 30..33)
    f29   f34 minus 5 dead feats   (Pattern P, 2026-04-14: drop indices [5, 9, 20, 26, 32])
    f37   f30 + 7 xd_*             (legacy; SKIPS SOTA Tier 3 — backward-compat checkpoint shape)
    f41   f34 + 7 xd_*             (full v50 — current default for V19)

V51 frontier additions (columns 41-120, eleven families):

    +5    hbr_* (Hawkes branching ratio)         -> f46
    +14   s3_*  (top-trader long/short)
    +9    bs_*  (basis)
    +13   liq_* (liquidations)
    +5    wh_*  (whale activity)
    +1    soc_* (Wikipedia attention)
    +3    xex_* (cross-exchange spreads)
    +3    dv_*  (Deribit DVOL)
    +13   stbl_* (stable mints)
    +13   etf_* (ETF flows)
    +1    fp_*  (funding panel)
    -----
    +80 total -> f121

Segmentation cuts (rationale: tiered ablation between f41 and f121):

    f46   f41 + Hawkes branching                          (microstructure: SHIPS in memory)
    f60   f46 + S3 (14)                                    (top-trader signals)
    f73   f60 + liquidations (13)                          (event-driven, regime-dep)
    f78   f73 + whale (5)                                  (institutional flow: SHIPS)
    f81   f78 + xex (3)                                    (BTC/ETH cross-exchange)
    f84   f81 + dvol (3)                                   (BTC/ETH options vol)
    f97   f84 + stbl (13)                                  (stable mints: SHIPS)
    f110  f97 + etf (13)                                   (ETF flows: SHIPS)
    f121  f110 + bs (9) + soc (1) + fp (1)                 (full kitchen sink)

V51 T2 chimera additions (post-f133, two new families — ground-truth verified
against btcusdt_v51_chimera_20260522.parquet, 2026-05-24):

    T2_21   fund_ (10) + premium_ (5) + bd_bgf_ (1) + lob_bgf_ (5) = 21 cols
              fund_*:    cross-asset funding rate panel (rate_mean/max/min/abs_mean/
                         z30/extreme_long_count/extreme_short_count/sign_flip/avg_apr/
                         n_settlements)
              premium_*: basis-premium derived signals (vol30/persistence30/
                         extreme_count30/z90/apr)
              bd_bgf_*:  bgf book-depth imbalance at L1
              lob_bgf_*: bgf LOB microstructure (l1_imb_mean/kyle_lambda_mean/
                         spread_bps_mean/top_pressure_mean/count_imb_mean)
              NOTE: xrel_lob_bgf_* prefix has 0 cols in current BTC chimera;
                    omitted until the column is written by the producer.

    XEX_NEW_7   7 new cross-exchange cols beyond the legacy XEX_3:
              _right variants (cb_bn/by_bn/ok_bn at different tape offset),
              spread_dispersion, max_abs_spread, cb_bn_z30, n_venues_active.
              Kept SEPARATE from T2_21 because they extend an existing family
              rather than introducing a new signal domain; a user can include
              XEX_3 (f81+) without XEX_NEW_7, and vice-versa.

New cumulative cuts:

    f154  f133 + T2_21     (21 new T2 cross-asset microstructure cols)
    f161  f154 + XEX_NEW_7 (+ 7 extended cross-exchange spread signals)

API:
    from feature_sets import get_feature_config, FEATURE_LIST_13, FEATURE_LIST_41, ...

    feature_list, input_dim, base_dim = get_feature_config(n_features)
        - feature_list: list[str] of column names in canonical order
        - input_dim: len(feature_list)
        - base_dim: number of "safe" features (0..base_dim-1) for V1.1+
                    base/XD anti-memorization split. For V1.0 / V0 / models
                    that don't use the split, base_dim == input_dim.

V1.0 / V0 callers can ignore base_dim. The 3-tuple is the unified contract.
"""
from __future__ import annotations


# ─── BASE FEATURES (canonical order, V50 schema columns 0-33) ────────────────

FEATURE_LIST_13 = [
    # 0-12 Legacy core
    "norm_deviation",
    "norm_fd_close",
    "norm_vpin",
    "norm_flow_imbalance",
    "norm_vol_cluster",
    "norm_funding",
    "norm_tick_count",
    "norm_log_volume",
    "norm_hl_spread",
    "hurst_regime",
    "norm_oi_change",
    "norm_return_1",
    "norm_spread_bps",
]

EXTENDED_BASE_5 = [
    # 13-17 Extended base
    "norm_ma_distance",
    "norm_whale",
    "norm_efficiency",
    "norm_return_4",
    "norm_return_16",
]

TIER1_3 = [
    # 18-20 Tier 1
    "norm_return_kurtosis",
    "norm_bar_duration",
    "norm_funding_momentum",
]

HAWKES_4 = [
    # 21-24 Hawkes
    "norm_hawkes_intensity",
    "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity",
    "norm_hawkes_imbalance",
]

IC_BOOST_5 = [
    # 25-29 IC-boost dynamics
    "norm_momentum_accel",
    "norm_vol_price_corr",
    "norm_vol_ratio",
    "norm_flow_persistence",
    "norm_oi_price_divergence",
]

SOTA_TIER3_4 = [
    # 30-33 SOTA Tier 3
    "norm_yz_volatility",
    "norm_cs_spread",
    "norm_perm_entropy",
    "norm_kyle_lambda",
]

# Cumulative bases
FEATURE_LIST_18 = FEATURE_LIST_13 + EXTENDED_BASE_5             # 18 cols
FEATURE_LIST_21 = FEATURE_LIST_18 + TIER1_3                     # 21 cols
FEATURE_LIST_25 = FEATURE_LIST_21 + HAWKES_4                    # 25 cols
FEATURE_LIST_30 = FEATURE_LIST_25 + IC_BOOST_5                  # 30 cols
FEATURE_LIST_34 = FEATURE_LIST_30 + SOTA_TIER3_4                # 34 cols

# Cross-asset 7 (V50 columns 34-40)
XD_FEATURES_7 = [
    "xd_btc_return",
    "xd_btc_volatility",
    "xd_funding_spread",
    "xd_cross_return_mean",
    "xd_cross_vol_mean",
    "xd_ma_distance",
    "xd_momentum_rank",
]

# Pattern P (2026-04-14): 5 dead features by raw IC < 0.001
# idx 5  norm_funding,         idx 9  hurst_regime,
# idx 20 norm_funding_momentum, idx 26 norm_vol_price_corr,
# idx 32 norm_perm_entropy
DEAD_FEATURE_INDICES = [5, 9, 20, 26, 32]
FEATURE_LIST_29 = [f for i, f in enumerate(FEATURE_LIST_34) if i not in DEAD_FEATURE_INDICES]

# Backward-compat composite: 30 base + 7 XD (skips SOTA Tier 3)
# Used by old V1.x / V2-V9 checkpoints saved under f37 schema.
FEATURE_LIST_37 = FEATURE_LIST_30 + XD_FEATURES_7               # 37 cols

# Full v50: 34 base + 7 XD
FEATURE_LIST_41 = FEATURE_LIST_34 + XD_FEATURES_7               # 41 cols


# ─── V51 FRONTIER FAMILIES (canonical order, V51 schema columns 41-120) ───────

HBR_5 = [
    "hbr_eta_total", "hbr_eta_buy", "hbr_eta_sell",
    "hbr_eta_imbalance", "hbr_n_trades",
]

S3_14 = [
    "s3_oi_usd", "s3_top_acct_lsr", "s3_top_pos_lsr", "s3_global_lsr", "s3_taker_lsr",
    "s3_top_pos_lsr_z", "s3_top_pos_lsr_xsec_z", "s3_global_lsr_z",
    "s3_smart_vs_retail", "s3_smart_vs_retail_z",
    "s3_smart_bullish", "s3_smart_bearish",
    "s3_smart_extreme_long", "s3_smart_extreme_short",
]

BS_9 = [
    "bs_basis_pct", "bs_basis_z30", "bs_basis_delta_1d", "bs_basis_delta_3d",
    "bs_basis_xsec_z", "bs_basis_bull_shock", "bs_basis_bear_shock",
    "bs_basis_panic", "bs_basis_frenzy",
]

LIQ_13 = [
    # Chimera column names: liq_features panel has prefix="" in feature_registry.yaml
    # because the panel already carries the liq_ prefix. Confirmed against
    # btcusdt_v51_chimera_20260522.parquet (2026-05-25). Do NOT double-prefix.
    "liq_long_usd", "liq_short_usd", "liq_delta_usd", "liq_total_usd",
    "liq_long_z30", "liq_short_z30", "liq_delta_z30",
    "liq_long_xsec_z", "liq_short_xsec_z",
    "liq_long_spike", "liq_short_spike",
    "liq_capitulation", "liq_short_panic",
]

WH_5 = [
    "wh_whale_trade_count", "wh_whale_trade_count_500k",
    "wh_whale_buy_usd", "wh_whale_sell_usd", "wh_whale_net_usd",
]

SOC_1 = ["soc_wiki_views"]

XEX_3 = [
    "xex_cb_bn_spread_bps", "xex_by_bn_spread_bps", "xex_ok_bn_spread_bps",
]

DV_3 = ["dv_dvol_close", "dv_dvol_high", "dv_dvol_low"]

STBL_13 = [
    "stbl_total_zscore_30d", "stbl_total_delta_7d_pct", "stbl_total_delta_30d_pct",
    "stbl_usdt_zscore_30d", "stbl_usdt_delta_7d_pct",
    "stbl_usdc_zscore_30d", "stbl_usde_zscore_30d", "stbl_dai_zscore_30d",
    "stbl_stable_shock", "stbl_stable_crash", "stbl_stable_shock_strong",
    "stbl_usdt_shock", "stbl_compound_shock",
]

ETF_13 = [
    "etf_btc_etf_total_usdm", "etf_btc_etf_total_z30", "etf_btc_etf_total_7d_z",
    "etf_btc_etf_inflow_shock", "etf_btc_etf_outflow_shock",
    "etf_btc_etf_mega_inflow", "etf_btc_etf_mega_outflow",
    "etf_eth_etf_total_usdm", "etf_eth_etf_total_z30",
    "etf_eth_etf_inflow_shock", "etf_eth_etf_outflow_shock",
    "etf_any_inflow_shock", "etf_both_inflow_shock",
]

FP_1 = ["fp_fund_panel"]

# Realized-volatility decomposition (BNS 2004 / Lee-Mykland 2008).
# Source panel: data/processed/panels/daily/rv_jump_panel_<DATE>.parquet
# (built by src/frontier/features/realized_volatility.py, wired as a T2
# build_panels stage). Joined onto per-asset frontier silver by date+asset.
RV_JUMPS_6 = [
    # Chimera column names: rv_jump panel has prefix="rv_" in feature_registry.yaml,
    # so the frontier consolidator renames rv_5m->rv_rv_5m, bpv_5m->rv_bpv_5m, etc.
    # Confirmed against btcusdt_v51_chimera_20260522.parquet (2026-05-25).
    # Source panel (rv_jump_panel_*.parquet) has bare names; chimera has rv_ prefix.
    "rv_rv_5m",             # realized variance (Σ r_i²)
    "rv_bpv_5m",            # bipower variation (jump-robust diffusion proxy)
    "rv_jv_5m",             # jump variation = max(RV - BPV, 0)
    "rv_jump_frac",         # JV / RV ∈ [0,1] — fraction of vol from jumps
    "rv_jump_count",        # Lee-Mykland 5%-FWER significant jumps per day
    "rv_jump_signed_var",   # Σ r_i² · sign(r_i) over jumps (asymmetry)
]

# Transfer Entropy directional info-flow features (Schreiber 2000).
# Source panel: data/processed/panels/daily/te_panel_<DATE>.parquet
# (built by src/frontier/features/transfer_entropy_panel.py).
# Per-asset aggregates of the u10×u10 TE matrix on rolling 90-day windows.
# Genuinely orthogonal to existing xd_* (correlation-only) features.
TE_6 = [
    "te_in",        # max TE INTO this asset from any peer (info-receiving signal)
    "te_out",       # max TE FROM this asset to any peer (info-leading signal)
    "te_in_btc",    # TE(BTC → asset) (BTC's lag info into this asset)
    "te_out_btc",   # TE(asset → BTC) (this asset's lag info into BTC)
    "te_imb",       # te_in - te_out: positive = follower; negative = leader
    "te_btc_imb",   # te_in_btc - te_out_btc: BTC-relative info posture
]

# Frontier in CANONICAL CHIMERA ORDER (matches v51 chimera column write order
# from V19's settings.FEATURE_LIST_121 — this is the order the columns appear
# in data/processed/chimera/dollar/<sym>_v51_chimera_<DATE>.parquet).
# DO NOT change without simultaneously rebuilding all v51 chimeras AND
# all checkpoints trained against this order.
FEATURE_LIST_FRONTIER_80 = (
    HBR_5 + S3_14 + BS_9 + LIQ_13 + WH_5 + SOC_1 + XEX_3 + DV_3 + STBL_13 + ETF_13 + FP_1
)
assert len(FEATURE_LIST_FRONTIER_80) == 80, (
    f"frontier 80 expected, got {len(FEATURE_LIST_FRONTIER_80)}"
)

# Full v51: 41 v50 + 80 frontier in chimera-order
FEATURE_LIST_121 = FEATURE_LIST_41 + FEATURE_LIST_FRONTIER_80
assert len(FEATURE_LIST_121) == 121, f"f121 len = {len(FEATURE_LIST_121)}"


# ─── INTERMEDIATE FRONTIER CUTS (segmentation between f41 and f121) ──────────
# Rationale per memory:
#   - HBR (Hawkes): SHIPS as microstructure feature (Rambaldi 2024)
#   - WH (whale):   SHIPS for institutional flow
#   - STBL (stable): SHIPS (per stable_flow_overlay_v51, +0.20 Sharpe blend)
#   - ETF:          SHIPS (per etf_flow_overlay_v51, blend +0.29 Sharpe)
#   - S3, LIQ, xex, dvol: medium signal, regime-dependent
#   - BS, FP, SOC: conceded as alpha but useful for world-model context
#
# Order: place SHIPPED signals first so cuts add high-signal families before
# low-signal ones. This is independent of the chimera column order — the
# TrainingLoader selects columns BY NAME, so a non-contiguous subset is fine.
# The numerical input_dim of each cut still gives a clean ablation step.

FEATURE_LIST_46 = FEATURE_LIST_41 + HBR_5                       # 46 cols (+Hawkes)
FEATURE_LIST_60 = FEATURE_LIST_46 + S3_14                       # 60 cols (+S3)
FEATURE_LIST_73 = FEATURE_LIST_60 + LIQ_13                      # 73 cols (+liq)
FEATURE_LIST_78 = FEATURE_LIST_73 + WH_5                        # 78 cols (+whale)
FEATURE_LIST_81 = FEATURE_LIST_78 + XEX_3                       # 81 cols (+xex)
FEATURE_LIST_84 = FEATURE_LIST_81 + DV_3                        # 84 cols (+dv)
FEATURE_LIST_97 = FEATURE_LIST_84 + STBL_13                     # 97 cols (+stbl)
FEATURE_LIST_110 = FEATURE_LIST_97 + ETF_13                     # 110 cols (+etf)
# Note: FEATURE_LIST_121 is defined above in chimera order (41 + 80 frontier).
# An "ablation-order f121" would be FEATURE_LIST_110 + BS_9 + SOC_1 + FP_1
# but we keep the canonical chimera order for the f121 entry to preserve
# checkpoint compatibility.

# RV-jump decomposition extension (post-2026-04-28 SOTA addition).
# Adds 6 features per asset (BPV/JV/jump_count/jump_frac/jump_signed_var/intensity).
# Append-only — preserves checkpoint compatibility for f121-trained models.
FEATURE_LIST_127 = FEATURE_LIST_121 + RV_JUMPS_6
assert len(FEATURE_LIST_127) == 127, f"f127 len = {len(FEATURE_LIST_127)}"

# Transfer-entropy extension (G-FRONTIER-003, post-2026-04-28).
# Adds 6 directional info-flow features per asset.
# Append-only — preserves checkpoint compatibility for f127-trained models.
FEATURE_LIST_133 = FEATURE_LIST_127 + TE_6
assert len(FEATURE_LIST_133) == 133, f"f133 len = {len(FEATURE_LIST_133)}"

# T2 chimera cols — new signal domains added to v51 chimera in 2026-05.
# Ground-truth column names verified against btcusdt_v51_chimera_20260522.parquet.
# Grouped by prefix for readability; order within each prefix matches chimera write order.
#
# fund_* (10): cross-asset funding-rate panel aggregates.
#   Captures aggregate cross-exchange funding dynamics beyond fp_fund_panel's scalar.
# premium_* (5): basis-premium derived signals (vol, persistence, z-scores, apr).
# bd_bgf_* (1): bgf-source book-depth imbalance at L1 (complementary to bd_imbalance_l1).
# lob_bgf_* (5): bgf-source LOB microstructure features (kyle lambda, spread, pressure).
#   NOTE: xrel_lob_bgf_kyle_lambda_mean_xratio is NOT yet written to BTC chimera;
#         omit until the producer is confirmed active across all u10 assets.
T2_21 = [
    # fund_ (10) — cross-asset funding rate panel
    "fund_rate_mean",
    "fund_rate_max",
    "fund_rate_min",
    "fund_rate_abs_mean",
    "fund_rate_z30",
    "fund_extreme_long_count",
    "fund_extreme_short_count",
    "fund_sign_flip",
    "fund_avg_apr",
    "fund_n_settlements",
    # premium_ (5) — basis-premium derived signals
    "premium_vol30",
    "premium_persistence30",
    "premium_extreme_count30",
    "premium_z90",
    "premium_apr",
    # bd_bgf_ (1) — bgf book-depth imbalance L1
    "bd_bgf_imbalance_l1",
    # lob_bgf_ (5) — bgf LOB microstructure
    "lob_bgf_l1_imb_mean",
    "lob_bgf_kyle_lambda_mean",
    "lob_bgf_spread_bps_mean",
    "lob_bgf_top_pressure_mean",
    "lob_bgf_count_imb_mean",
]
assert len(T2_21) == 21, f"T2_21 len = {len(T2_21)}"

# XEX_NEW_7: 7 cross-exchange spread cols added to v51 chimera beyond the legacy XEX_3.
# Kept separate so XEX_3-only users (f81+) are unaffected.
# _right variants capture the tape at a different exchange-side offset;
# spread_dispersion/max_abs_spread/z30/n_venues_active are derived summary stats.
XEX_NEW_7 = [
    "xex_cb_bn_spread_bps_right",
    "xex_by_bn_spread_bps_right",
    "xex_ok_bn_spread_bps_right",
    "xex_spread_dispersion",
    "xex_max_abs_spread",
    "xex_cb_bn_z30",
    "xex_n_venues_active",
]
assert len(XEX_NEW_7) == 7, f"XEX_NEW_7 len = {len(XEX_NEW_7)}"

# f154: f133 + T2_21 (21 new T2 cross-asset microstructure cols)
# Append-only — preserves checkpoint compatibility for f133-trained models.
FEATURE_LIST_154 = FEATURE_LIST_133 + T2_21
assert len(FEATURE_LIST_154) == 154, f"f154 len = {len(FEATURE_LIST_154)}"

# f161: f154 + XEX_NEW_7 (extended cross-exchange spread signals)
# Append-only — preserves checkpoint compatibility for f154-trained models.
FEATURE_LIST_161 = FEATURE_LIST_154 + XEX_NEW_7
assert len(FEATURE_LIST_161) == 161, f"f161 len = {len(FEATURE_LIST_161)}"

# Cross-asset relative features (xrel_*) — top-5 by KS separation (2026-05-18).
# These are cross-sectional RANK features computed per-date across the u100 universe.
# They preserve absolute magnitude signal lost in per-asset rolling z-scores.
# KS scores (winner vs non-winner, all dates): hbr_n_trades 0.128, liq_long 0.123,
# rv_rv_5m 0.102, rv_bpv_5m 0.099, rv_bpv_5m_xratio 0.096.
# Available in ALL 435 chimera parquets (dollar + 1d/4h/1h/15m) as of 2026-05-18.
# No-lookahead: cross-section for date D uses only values from date D.
XREL_5 = [
    "xrel_hbr_n_trades_xrank",    # Top KS=0.128; quiet-accumulation (WIN < LOSE rank)
    "xrel_liq_long_usd_xrank",    # KS=0.123; lower relative liq = safer momentum
    "xrel_rv_rv_5m_xrank",        # KS=0.102; realized variance cross-rank (WIN > LOSE)
    "xrel_rv_bpv_5m_xrank",       # KS=0.099; bipower variation cross-rank (jump-robust)
    "xrel_rv_bpv_5m_xratio",      # KS=0.096; ratio to universe median (complementary)
]

# f51: f46 (HBR-Hawkes) + XREL_5 top cross-sectional rank features.
# Recommended next-cohort training cut for all versions supporting f46.
FEATURE_LIST_51 = FEATURE_LIST_46 + XREL_5
assert len(FEATURE_LIST_51) == 51, f"f51 len = {len(FEATURE_LIST_51)}"


# ─── Registry ────────────────────────────────────────────────────────────────

# Maps n_features -> (feature_list, base_dim).
# base_dim: number of "safe" features at the head of the list. For V1.1+
# style models with XD anti-memorization, features [0:base_dim] are
# encoded cleanly while [base_dim:] get heavy dropout + noise.
#
# Convention:
#   - For pure-base lists (no XD/frontier): base_dim == n_features
#   - For lists ending in XD: base_dim = (n_features - 7)  (the 7 xd_*)
#   - For lists ending in XD + frontier: base_dim = 41     (v50 = 34 base + 7 XD)
_REGISTRY = {
    13:  (FEATURE_LIST_13, 13),
    18:  (FEATURE_LIST_18, 18),
    21:  (FEATURE_LIST_21, 21),
    25:  (FEATURE_LIST_25, 25),
    29:  (FEATURE_LIST_29, 29),
    30:  (FEATURE_LIST_30, 30),
    34:  (FEATURE_LIST_34, 34),
    37:  (FEATURE_LIST_37, 30),  # 30 base + 7 XD
    41:  (FEATURE_LIST_41, 34),  # 34 base + 7 XD
    # Frontier cuts: base_dim = 41 (full v50 = 34 base + 7 XD).
    # The frontier tail (cols 41+) gets the XD-style heavy dropout in V1.1+ models.
    46:  (FEATURE_LIST_46, 41),
    51:  (FEATURE_LIST_51, 41),  # f46 + 5 xrel_* cross-sectional rank features (2026-05-18)
    60:  (FEATURE_LIST_60, 41),
    73:  (FEATURE_LIST_73, 41),
    78:  (FEATURE_LIST_78, 41),
    81:  (FEATURE_LIST_81, 41),
    84:  (FEATURE_LIST_84, 41),
    97:  (FEATURE_LIST_97, 41),
    110: (FEATURE_LIST_110, 41),
    121: (FEATURE_LIST_121, 41),
    127: (FEATURE_LIST_127, 41),  # f121 + 6 RV/jump (BPV, JV, jump_count, ...)
    133: (FEATURE_LIST_133, 41),  # f127 + 6 TE features (G-FRONTIER-003)
    154: (FEATURE_LIST_154, 41),  # f133 + 21 T2 cols (fund/premium/bd_bgf/lob_bgf)
    161: (FEATURE_LIST_161, 41),  # f154 + 7 extended cross-exchange spread signals
}

SUPPORTED_COUNTS = sorted(_REGISTRY.keys())


def get_feature_config(n_features: int) -> tuple[list[str], int, int]:
    """Resolve a feature count to (feature_list, input_dim, base_dim).

    Standardized 3-tuple API for ALL versions. V1.0/V0/etc. that don't
    consume base_dim simply ignore the third element.

    Raises ValueError if n_features is not registered.
    """
    if n_features not in _REGISTRY:
        raise ValueError(
            f"unsupported n_features={n_features}; supported: {SUPPORTED_COUNTS}"
        )
    fl, base_dim = _REGISTRY[n_features]
    return list(fl), len(fl), base_dim


def list_supported() -> list[int]:
    """Return all registered feature counts."""
    return list(SUPPORTED_COUNTS)


def describe(n_features: int) -> str:
    """Human-readable description of an n_features bundle."""
    if n_features not in _REGISTRY:
        return f"unsupported f{n_features}"
    fl, base_dim = _REGISTRY[n_features]
    if n_features <= 34:
        return f"f{n_features} = {n_features} base features (no XD, no frontier)"
    if n_features <= 41:
        n_xd = n_features - base_dim
        return (f"f{n_features} = {base_dim} base + {n_xd} XD (v50 schema)")
    n_frontier = n_features - 41
    return (f"f{n_features} = 34 base + 7 XD + {n_frontier} frontier "
            f"(v51 schema; base_dim={base_dim})")


__all__ = [
    "FEATURE_LIST_13", "FEATURE_LIST_18", "FEATURE_LIST_21",
    "FEATURE_LIST_25", "FEATURE_LIST_29", "FEATURE_LIST_30",
    "FEATURE_LIST_34", "FEATURE_LIST_37", "FEATURE_LIST_41",
    "FEATURE_LIST_46", "FEATURE_LIST_51", "FEATURE_LIST_60", "FEATURE_LIST_73",
    "FEATURE_LIST_78", "FEATURE_LIST_81", "FEATURE_LIST_84",
    "FEATURE_LIST_97", "FEATURE_LIST_110", "FEATURE_LIST_121",
    "FEATURE_LIST_127", "FEATURE_LIST_133",
    "FEATURE_LIST_154", "FEATURE_LIST_161",
    "FEATURE_LIST_FRONTIER_80",
    "XD_FEATURES_7",
    "DEAD_FEATURE_INDICES",
    "HBR_5", "XREL_5", "S3_14", "BS_9", "LIQ_13", "WH_5",
    "SOC_1", "XEX_3", "DV_3", "STBL_13", "ETF_13", "FP_1",
    "RV_JUMPS_6", "TE_6",
    "T2_21", "XEX_NEW_7",
    "get_feature_config",
    "list_supported",
    "describe",
    "SUPPORTED_COUNTS",
]
