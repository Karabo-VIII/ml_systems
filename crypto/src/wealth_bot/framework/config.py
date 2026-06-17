"""Config schema + loader for wealth_bot framework.

__contract__:
  kind: config_loader
  inputs: YAML file specifying asset, indicator family, cadence, model params
  outputs: typed dataclass Config; validated.
  invariants:
    - asset is a non-empty symbol
    - cadence in {1h, 4h, 1d}
    - indicator_family in {EMA, SMA, EMA_SMA_mix}
    - n_seeds >= 1
    - windows TRAIN < VAL < OOS < UNSEEN (chronological)
    - dates resolved from src/split_config.py canonical SoT unless windows.use_canonical='custom'
"""
from __future__ import annotations

__contract__ = {
    "kind": "config_loader",
    "owner": "wealth_bot/framework/config",
    "purpose": "Parse + validate bot training config",
    "inputs": {"yaml": "path to YAML file"},
    "outputs": {"config": "BotConfig dataclass"},
    "invariants": [
        "cadence in {1h, 4h, 1d}",
        "n_seeds >= 1",
        "TRAIN < VAL < OOS < UNSEEN",
        "split dates pulled from split_config.py unless windows.use_canonical='custom'",
    ],
}

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Make split_config importable (src/ is parent.parent of this file)
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from split_config import (  # noqa: E402  (import after sys.path injection)
    TRAIN_END_DATE,
    VAL_END_DATE,
    OOS_END_DATE,
    UNSEEN_START_DATE,
    ROLLING_TRAIN_END_DATE,
    ROLLING_VAL_START_DATE,
    ROLLING_VAL_END_DATE,
    ROLLING_TEST_START_DATE,
    ROLLING_TEST_END_DATE,
)


@dataclass
class WindowConfig:
    """Train/Val/OOS/Unseen window boundaries (inclusive start, exclusive end)."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    unseen_start: pd.Timestamp
    unseen_end: pd.Timestamp


@dataclass
class StrategySpec:
    """One static strategy in the picker's action space."""
    ma_type: str         # "EMA_cross" | "SMA_cross" | "SMA_state" | "EMA_dist" | "SMA_dist"
    fast: int            # for *_dist: unused (kept for schema parity)
    slow: int            # for *_cross: slow MA; for *_state/*_dist: MA period
    filter_kind: str     # "whale_net>0" | "whale&hbr_eta_buy" | etc.
    threshold_pct: float = 0.0  # for *_dist: fire when (close/MA - 1) > threshold_pct/100


@dataclass
class ModelConfig:
    """LGBM hyperparams + walk-forward params."""
    n_estimators: int = 100
    max_depth: int = 3
    num_leaves: int = 7
    min_child_samples: int = 10
    learning_rate: float = 0.05
    reg_alpha: float = 0.3
    reg_lambda: float = 0.3
    bagging_fraction: float = 0.8
    feature_fraction: float = 0.8
    wf_train_window: int = 1500
    wf_step: int = 200
    min_signal_count_per_refit: int = 30


@dataclass
class UpgradeConfig:
    """Toggle each architectural upgrade."""
    u1_seed_ensemble: bool = True       # Average predictions across n_seeds at inference
    u2_threshold_calibration: bool = True
    u3_regime_conditional: bool = False  # On if chimera has BTC regime feature
    u4_synthetic_augmentation: bool = False  # Off by default; opt-in
    u1_n_seeds: int = 10
    u2_threshold_grid: list[float] = field(default_factory=lambda: [-0.005, 0.0, 0.005, 0.01, 0.015, 0.02])
    u4_synthetic_multiplier: float = 1.0  # Generate Nx of TRAIN size


@dataclass
class RiskConfig:
    """Trading bot risk parameters.

    Perp-specific (optional, default = spot semantics):
      - leverage: nominal leverage multiplier. 1.0 = spot equivalent.
      - liq_buffer_pct: how far price must move against a position before liq.
    """
    starting_capital_usd: float = 5000.0
    max_position_pct: float = 1.0       # Fraction of capital per trade
    kelly_fraction: float = 0.25         # Quarter Kelly (conservative)
    max_drawdown_pct: float = 25.0       # Halt if rolling DD exceeds
    max_consecutive_losses: int = 10
    cost_per_side_pct: float = 0.22      # 0.22% Binance taker
    whale_freshness_max_hours: float = 28.0
    # Perp extension (2026-05-25, INST fdbdb2bb closeout)
    leverage: float = 1.0                # 1.0 = spot; perp typically 2-5x
    liq_buffer_pct: float | None = None  # liquidation distance; None = no liq sim


@dataclass
class FundingConfig:
    """Perp funding parameters (optional). Use defaults when spot."""
    enabled: bool = False
    settle_hours: int = 8                # most exchanges: 8h
    column_name: str = "fund_rate_mean"  # chimera column
    apply_to_held_positions: bool = True


@dataclass
class BotConfig:
    """Top-level bot configuration."""
    asset: str
    cadence: str
    indicator_family: str
    n_seeds: int = 10
    strategies: list[StrategySpec] = field(default_factory=list)
    chimera_features: list[str] = field(default_factory=list)
    fwd_bars: int = 7
    chimera_lag_bars: int = 6  # 6*4h = 1d
    # Segment-mask purge gap (bars) at start of each non-TRAIN segment.
    # Default 0 preserves backward compat with existing audits; recommended
    # 400 for new bots (covers 200-bar SMA + 200-bar buffer). See
    # `data_loader.segment_masks` docstring.
    purge_bars: int = 0
    windows: WindowConfig | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    upgrades: UpgradeConfig = field(default_factory=UpgradeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    output_dir: str = "runs/audit/wealth_bot_default"


def _resolve_windows(w: dict[str, Any]) -> WindowConfig:
    """Resolve TRAIN/VAL/OOS/UNSEEN boundaries.

    Modes (controlled by ``w['use_canonical']``):

      * ``'legacy'`` (DEFAULT) — pull TRAIN/VAL/OOS/UNSEEN boundaries from
        ``split_config.py`` LEGACY constants. ``train_start`` and ``unseen_end``
        MUST be provided explicitly in YAML (they are the data-range bookends
        and asset-specific, not part of the canonical SoT).
      * ``'rolling'`` — pull boundaries from split_config.py ROLLING_* constants.
        UNSEEN is intentionally NOT defined in rolling mode (rolling mode's
        test window IS the canonical "final unseen" segment, with no further
        reserve). We refuse to construct WindowConfig in rolling mode -- caller
        must use ``rolling_windows()`` helper if rolling semantics are needed.
      * ``'custom'`` — use exactly what the YAML specifies. All 8 boundary
        dates must be present. NO validation against split_config (escape hatch
        for unusual experiments; document why in commit message).

      Default = 'legacy'. If ``use_canonical`` is omitted AND the YAML supplies
      all 8 explicit boundary dates, the loader will VALIDATE that those dates
      match the LEGACY canonical exactly -- mismatch raises ValueError.
    """
    mode = w.get("use_canonical", "legacy")
    if mode not in {"legacy", "rolling", "custom"}:
        raise ValueError(
            f"windows.use_canonical must be one of {{legacy, rolling, custom}}; "
            f"got {mode!r}"
        )

    if mode == "rolling":
        raise NotImplementedError(
            "use_canonical='rolling' is not yet supported by wealth_bot.WindowConfig. "
            "Rolling mode has no UNSEEN reserve segment by design (the test segment "
            "Jan-Apr 2026 IS the canonical final unseen window). Build a dedicated "
            "RollingWindowConfig if rolling semantics are needed."
        )

    if mode == "legacy":
        # Pull canonical boundaries from split_config; YAML supplies only
        # data-range bookends (train_start, unseen_end).
        required = ("train_start", "unseen_end")
        missing = [k for k in required if k not in w]
        if missing:
            raise ValueError(
                f"In use_canonical='legacy' mode the YAML must still provide the "
                f"data-range bookends {missing!r} (asset-specific, not in SoT)."
            )
        # Resolve canonical interior boundaries
        canonical = {
            "train_end":    TRAIN_END_DATE,
            "val_start":    TRAIN_END_DATE,
            "val_end":      VAL_END_DATE,
            "oos_start":    VAL_END_DATE,
            "oos_end":      OOS_END_DATE,
            "unseen_start": UNSEEN_START_DATE,
        }
        # If the YAML ALSO supplied any of these interior dates, validate match.
        for k, expected in canonical.items():
            if k in w and str(w[k])[:10] != expected:
                raise ValueError(
                    f"windows.{k}={w[k]!r} does not match canonical "
                    f"split_config.{k.upper()}={expected!r}. "
                    f"In use_canonical='legacy' mode, either OMIT this field "
                    f"(it will be auto-filled from SoT) or set "
                    f"windows.use_canonical: custom to override."
                )
        return WindowConfig(
            train_start=pd.Timestamp(w["train_start"]),
            train_end=pd.Timestamp(canonical["train_end"]),
            val_start=pd.Timestamp(canonical["val_start"]),
            val_end=pd.Timestamp(canonical["val_end"]),
            oos_start=pd.Timestamp(canonical["oos_start"]),
            oos_end=pd.Timestamp(canonical["oos_end"]),
            unseen_start=pd.Timestamp(canonical["unseen_start"]),
            unseen_end=pd.Timestamp(w["unseen_end"]),
        )

    # mode == "custom": full explicit dates required, no canonical validation.
    required = ("train_start", "train_end", "val_start", "val_end",
                "oos_start", "oos_end", "unseen_start", "unseen_end")
    missing = [k for k in required if k not in w]
    if missing:
        raise ValueError(
            f"In use_canonical='custom' mode the YAML must provide all 8 "
            f"boundary dates; missing {missing!r}."
        )
    return WindowConfig(
        train_start=pd.Timestamp(w["train_start"]),
        train_end=pd.Timestamp(w["train_end"]),
        val_start=pd.Timestamp(w["val_start"]),
        val_end=pd.Timestamp(w["val_end"]),
        oos_start=pd.Timestamp(w["oos_start"]),
        oos_end=pd.Timestamp(w["oos_end"]),
        unseen_start=pd.Timestamp(w["unseen_start"]),
        unseen_end=pd.Timestamp(w["unseen_end"]),
    )


def load_config(yaml_path: str | Path) -> BotConfig:
    """Load + validate config from YAML."""
    with open(yaml_path) as fp:
        raw = yaml.safe_load(fp)

    windows = _resolve_windows(raw["windows"])

    strategies = [
        StrategySpec(
            ma_type=s["ma_type"],
            fast=int(s["fast"]),
            slow=int(s.get("slow", 0)),
            filter_kind=s.get("filter_kind", "none"),
        )
        for s in raw["strategies"]
    ]

    model = ModelConfig(**raw.get("model", {}))
    upgrades = UpgradeConfig(**raw.get("upgrades", {}))
    # Filter risk kwargs to known fields so legacy YAMLs with unknown
    # keys don't blow up. Unknown keys are warned but not fatal.
    risk_raw = raw.get("risk", {}) or {}
    known_risk = {f.name for f in RiskConfig.__dataclass_fields__.values()}
    risk_kwargs = {k: v for k, v in risk_raw.items() if k in known_risk}
    risk = RiskConfig(**risk_kwargs)
    funding_raw = raw.get("funding", {}) or {}
    known_funding = {f.name for f in FundingConfig.__dataclass_fields__.values()}
    funding_kwargs = {k: v for k, v in funding_raw.items() if k in known_funding}
    funding = FundingConfig(**funding_kwargs)

    cfg = BotConfig(
        asset=raw["asset"],
        cadence=raw["cadence"],
        indicator_family=raw["indicator_family"],
        n_seeds=int(raw.get("n_seeds", 10)),
        strategies=strategies,
        chimera_features=list(raw.get("chimera_features", [])),
        fwd_bars=int(raw.get("fwd_bars", 7)),
        chimera_lag_bars=int(raw.get("chimera_lag_bars", 6)),
        purge_bars=int(raw.get("purge_bars", 0)),
        windows=windows,
        model=model,
        upgrades=upgrades,
        risk=risk,
        funding=funding,
        output_dir=raw.get("output_dir", "runs/audit/wealth_bot_default"),
    )

    _validate(cfg)
    return cfg


# Known filter-kind registry — kept in sync with data_loader._SIMPLE_FILTERS,
# _ROLLING_MEDIAN_FILTERS, _whale_filter() AND combos. Any new filter MUST be
# added here AND in data_loader; the config validator hard-fails on unknown
# kinds so silent fall-through to "always-1" can never happen.
KNOWN_FILTER_KINDS = frozenset({
    # Always-1 escape hatches
    "none", "no_filter",
    # Whale family
    "whale_net>0", "whale_net>30d_median", "whale_net>60d_median",
    # Simple lagged single-feature
    "btc_tape>0", "short_liq_z>0", "long_liq_z<0", "basis_z<0", "btc_ret<0",
    "tape_imb>0", "hbr_eta_buy>0",
    # Rolling-median single-feature
    "bd_imb>med", "fund_low", "lob_kyle_low",
    # AND combos
    "whale&btc_tape", "whale&short_liq", "whale30d&btc_weak", "whale&hbr_eta_buy",
    # OR composites (R23c family)
    "whale_OR_pz_neg",
})


def _validate(cfg: BotConfig) -> None:
    assert cfg.asset, "asset must be non-empty"
    assert cfg.cadence in {"1h", "4h", "1d"}, f"unsupported cadence {cfg.cadence}"
    # Accepted families: EMA / SMA / WMA / DEMA / HMA, "cross" or "state" or "dist" suffix allowed,
    # plus EMA_SMA_mix for multi-strat configs that combine families.
    _KNOWN_FAMILIES = {"EMA", "SMA", "WMA", "DEMA", "HMA", "EMA_SMA_mix",
                        "EMA_cross", "SMA_cross", "WMA_cross", "DEMA_cross",
                        "EMA_state", "SMA_state", "HMA_state",
                        "EMA_dist", "SMA_dist", "WMA_dist"}
    assert cfg.indicator_family in _KNOWN_FAMILIES, f"unsupported indicator_family {cfg.indicator_family}; known={sorted(_KNOWN_FAMILIES)}"
    assert cfg.n_seeds >= 1, "n_seeds must be >= 1"
    assert len(cfg.strategies) >= 1, "must define at least 1 strategy"
    # BINDING 2026-05-25: filter_kind must be in the known registry. Pre-fix,
    # a typo'd filter_kind silently fell through _whale_filter to ValueError
    # only at runtime — sometimes after partial sweep work was already lost.
    for i, s in enumerate(cfg.strategies):
        if s.filter_kind not in KNOWN_FILTER_KINDS:
            raise ValueError(
                f"strategies[{i}].filter_kind={s.filter_kind!r} not in known registry. "
                f"Add to data_loader._whale_filter() AND config.KNOWN_FILTER_KINDS. "
                f"Known: {sorted(KNOWN_FILTER_KINDS)}"
            )
    w = cfg.windows
    assert w.train_start < w.train_end <= w.val_start < w.val_end <= w.oos_start < w.oos_end <= w.unseen_start < w.unseen_end, (
        "windows must be chronological"
    )
