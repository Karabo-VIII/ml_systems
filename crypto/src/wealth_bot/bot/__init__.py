"""wealth_bot.bot -- paper-trade bot core on top of the framework.

Modules:
  signal_engine    -- consumes ensemble preds + threshold, emits SignalDecision per bar
  position_sizer   -- Kelly-fraction sizing (capped by risk.max_position_pct)
  risk_manager     -- kill switch on DD / consecutive losses / whale freshness
  order_generator  -- Binance-style order dict emitter (paper-trade)
  runner           -- main loop: walk bars, decide, size, order, exit, journal
  telemetry        -- JSONL journal + alert log
"""
