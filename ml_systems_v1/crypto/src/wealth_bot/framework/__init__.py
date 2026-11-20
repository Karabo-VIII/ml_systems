"""wealth_bot.framework — reproducible ML training framework.

Modules:
  config         -- Config schema + YAML loader
  data_loader    -- Chimera loader + feature builder + label builder
  signal_picker  -- LGBM signal-picker over indicator strategies
  upgrades       -- U1 (ensemble), U2 (threshold), U3 (regime), U4 (synthetic)
  walk_forward   -- N-seed audit + walk-forward CV + bootstrap CIs
  reports        -- REPORT.md generator
"""
