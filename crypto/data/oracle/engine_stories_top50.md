# Top-50 engine stories (passed gate, sorted by composite_score)
_Generated 2026-05-22; based on `engine_catalog.parquet` with 213 engines._

## #1 SHIB -- measure_event -- measure_engines/bd_imbalance_l1 op_abs_gt_thr_1.0

**Story**: SHIB chimera-measure engine on measure_engines/bd_imbalance_l1: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy +10.76% per fire; unit-compound +170.6%; max DD -2.0%. ShIC ratio 0.27 (strong, low memorization). 3-fold: sign-consistent (+0.0/+20.7/+130.7%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=SHIB; indicator_class=measure_engines/bd_imbalance_l1; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=2; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #2 FLOKI -- measure_event -- measure_engines/hbr_eta_buy op_abs_gt_thr_1.0

**Story**: FLOKI chimera-measure engine on measure_engines/hbr_eta_buy: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=11 fires): hit_rate=72.7%, expectancy +6.81% per fire; unit-compound +92.6%; max DD -1.3%. ShIC ratio 0.20 (strong, low memorization). 3-fold: sign-consistent (+55.9/+14.7/+0.0%). Hold 3d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=FLOKI; indicator_class=measure_engines/hbr_eta_buy; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #3 PEPE -- measure_event -- measure_engines/wh_whale_net_usd op_abs_gt_thr_1.0

**Story**: PEPE chimera-measure engine on measure_engines/wh_whale_net_usd: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=12 fires): hit_rate=66.7%, expectancy +7.31% per fire; unit-compound +115.1%; max DD -6.8%. ShIC ratio 0.18 (strong, low memorization). 3-fold: sign-consistent (+1.5/+64.5/+28.8%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=PEPE; indicator_class=measure_engines/wh_whale_net_usd; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #4 SUPER -- ta_event -- VPIN_threshold t_0.5

**Story**: SUPER VPIN_threshold(t_0.5) in bull-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=13 fires): hit_rate=76.9%, expectancy=+5.42% per fire after cost; unit-compound +94.0%; max DD -6.0%. ShIC ratio 0.25 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+1.7/+94.6/+37.7%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=SUPER; indicator_class=VPIN_threshold; indicator_config=t_0.5; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #5 FET -- ta_event -- VPIN_threshold t_0.5

**Story**: FET VPIN_threshold(t_0.5) in bull-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=11 fires): hit_rate=54.5%, expectancy=+7.12% per fire after cost; unit-compound +88.9%; max DD -10.4%. ShIC ratio 0.30 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+0.0/-42.9/-58.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=FET; indicator_class=VPIN_threshold; indicator_config=t_0.5; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #6 NEAR -- measure_event -- measure_engines/wh_whale_trade_count_500k op_abs_gt_thr_1.5

**Story**: NEAR chimera-measure engine on measure_engines/wh_whale_trade_count_500k: op_abs_gt_thr_1.5 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy +5.88% per fire; unit-compound +78.9%; max DD -12.4%. ShIC ratio 0.24 (strong, low memorization). 3-fold: sign-consistent (+13.2/+29.2/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=NEAR; indicator_class=measure_engines/wh_whale_trade_count_500k; indicator_config=op_abs_gt_thr_1.5; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #7 NEAR -- measure_event -- measure_engines/wh_whale_trade_count_500k op_gt_thr_1.5

**Story**: NEAR chimera-measure engine on measure_engines/wh_whale_trade_count_500k: op_gt_thr_1.5 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy +5.88% per fire; unit-compound +78.9%; max DD -12.4%. ShIC ratio 0.24 (strong, low memorization). 3-fold: sign-consistent (+13.2/+29.2/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=NEAR; indicator_class=measure_engines/wh_whale_trade_count_500k; indicator_config=op_gt_thr_1.5; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #8 NEAR -- measure_event -- measure_engines/wh_whale_trade_count_500k op_gt_thr_1.0

**Story**: NEAR chimera-measure engine on measure_engines/wh_whale_trade_count_500k: op_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=15 fires): hit_rate=66.7%, expectancy +5.20% per fire; unit-compound +102.5%; max DD -12.4%. ShIC ratio 0.22 (strong, low memorization). 3-fold: sign-consistent (+13.9/+45.4/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=NEAR; indicator_class=measure_engines/wh_whale_trade_count_500k; indicator_config=op_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #9 ADA -- ta_state -- VWAP_state_above period_20

**Story**: ADA VWAP_state_above(period_20) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=12 bars in state): hit_rate=75.0%, per-fire expectancy +4.59%; unit-compound +62.5%; max DD -11.1%. ShIC ratio 0.27. 3-fold sign: sign-consistent. Hold 3d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=ADA; indicator_class=VWAP_state_above; indicator_config=period_20; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #10 XRP -- measure_event -- measure_engines/hbr_eta_buy op_gt_thr_1.0

**Story**: XRP chimera-measure engine on measure_engines/hbr_eta_buy: op_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=10 fires): hit_rate=90.0%, expectancy +3.75% per fire; unit-compound +42.7%; max DD -0.3%. ShIC ratio 0.21 (strong, low memorization). 3-fold: sign-consistent (+11.0/+8.3/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=XRP; indicator_class=measure_engines/hbr_eta_buy; indicator_config=op_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #11 AR -- ta_state -- MA_state_SMA_above period_20

**Story**: AR MA_state_SMA_above(period_20) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=10 bars in state): hit_rate=70.0%, per-fire expectancy +4.45%; unit-compound +52.3%; max DD -0.4%. ShIC ratio 0.25. 3-fold sign: sign-consistent. Hold 1d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=AR; indicator_class=MA_state_SMA_above; indicator_config=period_20; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=4; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #12 SUI -- measure_event -- measure_engines/bd_imbalance_l5 op_lt_thr_1.0

**Story**: SUI chimera-measure engine on measure_engines/bd_imbalance_l5: op_lt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=11 fires): hit_rate=81.8%, expectancy +3.80% per fire; unit-compound +49.8%; max DD -2.1%. ShIC ratio 0.17 (strong, low memorization). 3-fold: sign-consistent (+15.9/+21.1/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=SUI; indicator_class=measure_engines/bd_imbalance_l5; indicator_config=op_lt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #13 NEAR -- measure_event -- measure_engines/wh_whale_trade_count_500k op_abs_gt_thr_1.0

**Story**: NEAR chimera-measure engine on measure_engines/wh_whale_trade_count_500k: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=17 fires): hit_rate=64.7%, expectancy +4.66% per fire; unit-compound +104.8%; max DD -14.7%. ShIC ratio 0.24 (strong, low memorization). 3-fold: sign-consistent (+13.9/+47.0/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=NEAR; indicator_class=measure_engines/wh_whale_trade_count_500k; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #14 HBAR -- measure_event -- measure_engines/norm_efficiency op_gt_thr_1.0

**Story**: HBAR chimera-measure engine on measure_engines/norm_efficiency: op_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=11 fires): hit_rate=72.7%, expectancy +3.97% per fire; unit-compound +49.7%; max DD -1.8%. ShIC ratio 0.16 (strong, low memorization). 3-fold: sign-consistent (+0.0/+42.5/+7.9%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=HBAR; indicator_class=measure_engines/norm_efficiency; indicator_config=op_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #15 HBAR -- measure_event -- measure_engines/norm_deviation op_gt_thr_1.0

**Story**: HBAR chimera-measure engine on measure_engines/norm_deviation: op_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=10 fires): hit_rate=70.0%, expectancy +3.82% per fire; unit-compound +43.6%; max DD -3.8%. ShIC ratio 0.22 (strong, low memorization). 3-fold: sign-consistent (+25.6/+11.5/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=HBAR; indicator_class=measure_engines/norm_deviation; indicator_config=op_gt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #16 ADA -- ta_event -- VPIN_threshold t_1.0

**Story**: ADA VPIN_threshold(t_1.0) in bull-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=10 fires): hit_rate=60.0%, expectancy=+4.42% per fire after cost; unit-compound +49.3%; max DD -5.2%. ShIC ratio 0.16 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+0.0/+19.2/+15.9%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=ADA; indicator_class=VPIN_threshold; indicator_config=t_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #17 LINK -- ta_event -- VPIN_threshold t_1.0

**Story**: LINK VPIN_threshold(t_1.0) in bull-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=14 fires): hit_rate=78.6%, expectancy=+3.31% per fire after cost; unit-compound +54.7%; max DD -1.9%. ShIC ratio 0.19 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+21.7/+18.1/+4.6%). Hold 1d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=LINK; indicator_class=VPIN_threshold; indicator_config=t_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #18 LINK -- ta_event -- MACD_threshold f_12_s_35_g_9

**Story**: LINK MACD_threshold(f_12_s_35_g_9) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=11 fires): hit_rate=72.7%, expectancy=+3.53% per fire after cost; unit-compound +44.5%; max DD -4.7%. ShIC ratio 0.25 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (-4.9/-1.8/-3.7%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=LINK; indicator_class=MACD_threshold; indicator_config=f_12_s_35_g_9; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #19 ARKM -- ta_event -- VPIN_threshold t_0.5

**Story**: ARKM VPIN_threshold(t_0.5) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=13 fires): hit_rate=69.2%, expectancy=+3.69% per fire after cost; unit-compound +55.2%; max DD -10.5%. ShIC ratio 0.26 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+12.4/+15.9/+0.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=ARKM; indicator_class=VPIN_threshold; indicator_config=t_0.5; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #20 BLUR -- measure_event -- measure_engines/xd_btc_return op_abs_gt_thr_1.0

**Story**: BLUR chimera-measure engine on measure_engines/xd_btc_return: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=12 fires): hit_rate=83.3%, expectancy +2.83% per fire; unit-compound +39.0%; max DD -2.3%. ShIC ratio 0.23 (strong, low memorization). 3-fold: sign-consistent (+0.0/+20.8/+15.1%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=BLUR; indicator_class=measure_engines/xd_btc_return; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #21 SOL -- ta_event -- MACD_threshold f_12_s_35_g_9

**Story**: SOL MACD_threshold(f_12_s_35_g_9) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy=+3.67% per fire after cost; unit-compound +44.3%; max DD -6.6%. ShIC ratio 0.23 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+34.1/+22.2/+15.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=SOL; indicator_class=MACD_threshold; indicator_config=f_12_s_35_g_9; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #22 SOL -- ta_event -- MACD_threshold f_12_s_26_g_9

**Story**: SOL MACD_threshold(f_12_s_26_g_9) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy=+3.67% per fire after cost; unit-compound +44.3%; max DD -6.6%. ShIC ratio 0.25 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+10.1/+22.2/+15.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=SOL; indicator_class=MACD_threshold; indicator_config=f_12_s_26_g_9; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #23 SOL -- ta_event -- MACD_threshold f_12_s_21_g_9

**Story**: SOL MACD_threshold(f_12_s_21_g_9) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy=+3.67% per fire after cost; unit-compound +44.3%; max DD -6.6%. ShIC ratio 0.22 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+10.1/+22.2/+15.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=SOL; indicator_class=MACD_threshold; indicator_config=f_12_s_21_g_9; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #24 SOL -- ta_event -- MACD_threshold f_8_s_35_g_9

**Story**: SOL MACD_threshold(f_8_s_35_g_9) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=11 fires): hit_rate=63.6%, expectancy=+3.67% per fire after cost; unit-compound +44.3%; max DD -6.6%. ShIC ratio 0.20 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+10.1/+22.2/+15.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=SOL; indicator_class=MACD_threshold; indicator_config=f_8_s_35_g_9; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #25 SEI -- measure_event -- measure_engines/norm_deviation op_lt_thr_1.0

**Story**: SEI chimera-measure engine on measure_engines/norm_deviation: op_lt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=10 fires): hit_rate=70.0%, expectancy +3.33% per fire; unit-compound +37.7%; max DD -2.5%. ShIC ratio 0.20 (strong, low memorization). 3-fold: sign-consistent (+0.2/+22.8/+11.9%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=SEI; indicator_class=measure_engines/norm_deviation; indicator_config=op_lt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #26 PEPE -- ta_state -- Donchian_state_above_midline period_55

**Story**: PEPE Donchian_state_above_midline(period_55) STATE engine in chop-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=10 bars in state): hit_rate=90.0%, per-fire expectancy +2.57%; unit-compound +28.3%; max DD -2.1%. ShIC ratio 0.24. 3-fold sign: sign-consistent. Hold 1d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=PEPE; indicator_class=Donchian_state_above_midline; indicator_config=period_55; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #27 PEPE -- ta_state -- MA_state_SMA_above period_50

**Story**: PEPE MA_state_SMA_above(period_50) STATE engine in chop-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=10 bars in state): hit_rate=90.0%, per-fire expectancy +2.57%; unit-compound +28.3%; max DD -2.1%. ShIC ratio 0.24. 3-fold sign: sign-consistent. Hold 1d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=PEPE; indicator_class=MA_state_SMA_above; indicator_config=period_50; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #28 LINK -- ta_event -- VPIN_threshold t_0.5

**Story**: LINK VPIN_threshold(t_0.5) in bull-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=13 fires): hit_rate=76.9%, expectancy=+2.96% per fire after cost; unit-compound +41.6%; max DD -12.8%. ShIC ratio 0.27 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (-3.3/-39.3/-3.0%). Hold 3d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=LINK; indicator_class=VPIN_threshold; indicator_config=t_0.5; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=4; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #29 PEPE -- ta_state -- MA_state_SMA_above period_100

**Story**: PEPE MA_state_SMA_above(period_100) STATE engine in chop-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=12 bars in state): hit_rate=75.0%, per-fire expectancy +3.01%; unit-compound +40.1%; max DD -8.3%. ShIC ratio 0.20. 3-fold sign: sign-consistent. Hold 1d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=PEPE; indicator_class=MA_state_SMA_above; indicator_config=period_100; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #30 LINK -- measure_event -- measure_engines/liq_long_usd op_abs_gt_thr_1.0

**Story**: LINK chimera-measure engine on measure_engines/liq_long_usd: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=10 fires): hit_rate=70.0%, expectancy +3.14% per fire; unit-compound +34.7%; max DD -2.9%. ShIC ratio 0.22 (strong, low memorization). 3-fold: sign-consistent (+7.4/+19.8/+4.7%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=LINK; indicator_class=measure_engines/liq_long_usd; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=2; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #31 HBAR -- measure_event -- measure_engines/norm_efficiency op_abs_gt_thr_1.0

**Story**: HBAR chimera-measure engine on measure_engines/norm_efficiency: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=13 fires): hit_rate=69.2%, expectancy +3.09% per fire; unit-compound +44.3%; max DD -6.2%. ShIC ratio 0.19 (strong, low memorization). 3-fold: sign-consistent (+0.0/+37.3/+7.9%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=HBAR; indicator_class=measure_engines/norm_efficiency; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #32 FIL -- measure_event -- measure_engines/liq_long_usd op_gt_thr_1.0

**Story**: FIL chimera-measure engine on measure_engines/liq_long_usd: op_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=10 fires): hit_rate=60.0%, expectancy +3.56% per fire; unit-compound +38.6%; max DD -4.9%. ShIC ratio 0.21 (strong, low memorization). 3-fold: sign-consistent (+0.0/+1.2/+42.5%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=FIL; indicator_class=measure_engines/liq_long_usd; indicator_config=op_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=2; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #33 FIL -- measure_event -- measure_engines/liq_long_usd op_abs_gt_thr_1.0

**Story**: FIL chimera-measure engine on measure_engines/liq_long_usd: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=10 fires): hit_rate=60.0%, expectancy +3.56% per fire; unit-compound +38.6%; max DD -4.9%. ShIC ratio 0.21 (strong, low memorization). 3-fold: sign-consistent (+0.0/+1.2/+42.5%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=FIL; indicator_class=measure_engines/liq_long_usd; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=2; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #34 FET -- measure_event -- measure_engines/xd_funding_spread op_abs_gt_thr_1.0

**Story**: FET chimera-measure engine on measure_engines/xd_funding_spread: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bear, magnitude: all. Over TRAIN (n=13 fires): hit_rate=84.6%, expectancy +2.49% per fire; unit-compound +36.1%; max DD -6.5%. ShIC ratio 0.22 (strong, low memorization). 3-fold: sign-consistent (+11.1/+0.0/+22.6%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=FET; indicator_class=measure_engines/xd_funding_spread; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bear; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #35 AVAX -- ta_event -- ATR_bands p_14_k_1.5

**Story**: AVAX ATR_bands(p_14_k_1.5) in chop-regime, all magnitude bucket. Fires when the indicator's discrete event triggers. Over TRAIN (n=10 fires): hit_rate=80.0%, expectancy=+2.64% per fire after cost; unit-compound +29.4%; max DD -1.5%. ShIC ratio 0.18 (strong, low memorization). 3-fold sub-TRAIN sign-consistency: sign-consistent (+0.9/+0.1/+5.6%). Hold 1d, taker cost taker_bucket.

**Recipe** (deterministic re-construction):
```
family=ta_event; asset=AVAX; indicator_class=ATR_bands; indicator_config=p_14_k_1.5; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #36 FET -- measure_event -- measure_engines/wh_whale_net_usd op_abs_gt_thr_2.0

**Story**: FET chimera-measure engine on measure_engines/wh_whale_net_usd: op_abs_gt_thr_2.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=11 fires): hit_rate=54.5%, expectancy +3.74% per fire; unit-compound +45.7%; max DD -5.7%. ShIC ratio 0.16 (strong, low memorization). 3-fold: sign-consistent (+11.1/+27.2/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=FET; indicator_class=measure_engines/wh_whale_net_usd; indicator_config=op_abs_gt_thr_2.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #37 ZEC -- measure_event -- measure_engines/norm_efficiency op_abs_gt_thr_1.0

**Story**: ZEC chimera-measure engine on measure_engines/norm_efficiency: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=12 fires): hit_rate=75.0%, expectancy +2.69% per fire; unit-compound +36.7%; max DD -1.2%. ShIC ratio 0.24 (strong, low memorization). 3-fold: sign-consistent (+16.8/+8.6/+0.0%). Hold 3d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=ZEC; indicator_class=measure_engines/norm_efficiency; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #38 JST -- ta_state -- Donchian_state_above_midline period_100

**Story**: JST Donchian_state_above_midline(period_100) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=14 bars in state): hit_rate=71.4%, per-fire expectancy +2.81%; unit-compound +41.7%; max DD -14.6%. ShIC ratio 0.26. 3-fold sign: sign-consistent. Hold 3d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=JST; indicator_class=Donchian_state_above_midline; indicator_config=period_100; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #39 XRP -- ta_state -- MA_state_EMA_above period_50

**Story**: XRP MA_state_EMA_above(period_50) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=12 bars in state): hit_rate=66.7%, per-fire expectancy +2.96%; unit-compound +38.5%; max DD -7.5%. ShIC ratio 0.23. 3-fold sign: sign-consistent. Hold 3d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=XRP; indicator_class=MA_state_EMA_above; indicator_config=period_50; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #40 BLUR -- measure_event -- measure_engines/bd_imbalance_l1 op_abs_gt_thr_1.0

**Story**: BLUR chimera-measure engine on measure_engines/bd_imbalance_l1: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=10 fires): hit_rate=80.0%, expectancy +2.46% per fire; unit-compound +27.0%; max DD -2.6%. ShIC ratio 0.13 (very strong, non-memorized). 3-fold: sign-consistent (+0.0/+13.6/+11.8%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=BLUR; indicator_class=measure_engines/bd_imbalance_l1; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #41 HBAR -- measure_event -- measure_engines/xd_funding_spread op_abs_gt_thr_1.0

**Story**: HBAR chimera-measure engine on measure_engines/xd_funding_spread: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bear, magnitude: all. Over TRAIN (n=10 fires): hit_rate=80.0%, expectancy +2.46% per fire; unit-compound +26.6%; max DD -4.9%. ShIC ratio 0.26 (strong, low memorization). 3-fold: sign-consistent (+2.5/+0.0/+23.4%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=HBAR; indicator_class=measure_engines/xd_funding_spread; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bear; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #42 ARB -- measure_event -- measure_engines/norm_efficiency op_abs_gt_thr_1.0

**Story**: ARB chimera-measure engine on measure_engines/norm_efficiency: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=15 fires): hit_rate=80.0%, expectancy +2.46% per fire; unit-compound +40.5%; max DD -7.6%. ShIC ratio 0.21 (strong, low memorization). 3-fold: sign-consistent (+0.0/+29.8/+6.7%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=ARB; indicator_class=measure_engines/norm_efficiency; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #43 SUI -- ta_state -- MA_state_EMA_above period_200

**Story**: SUI MA_state_EMA_above(period_200) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=21 bars in state): hit_rate=66.7%, per-fire expectancy +2.94%; unit-compound +73.0%; max DD -6.8%. ShIC ratio 0.18. 3-fold sign: sign-consistent. Hold 1d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=SUI; indicator_class=MA_state_EMA_above; indicator_config=period_200; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #44 AAVE -- measure_event -- measure_engines/rv_jump_frac op_abs_gt_thr_1.0

**Story**: AAVE chimera-measure engine on measure_engines/rv_jump_frac: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=10 fires): hit_rate=90.0%, expectancy +2.15% per fire; unit-compound +23.4%; max DD -0.7%. ShIC ratio 0.17 (strong, low memorization). 3-fold: sign-consistent (+15.7/+6.7/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=AAVE; indicator_class=measure_engines/rv_jump_frac; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #45 AAVE -- measure_event -- measure_engines/rv_jump_frac op_gt_thr_1.0

**Story**: AAVE chimera-measure engine on measure_engines/rv_jump_frac: op_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=10 fires): hit_rate=90.0%, expectancy +2.15% per fire; unit-compound +23.4%; max DD -0.7%. ShIC ratio 0.17 (strong, low memorization). 3-fold: sign-consistent (+15.7/+6.7/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=AAVE; indicator_class=measure_engines/rv_jump_frac; indicator_config=op_gt_thr_1.0; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #46 HBAR -- measure_event -- measure_engines/norm_deviation op_abs_gt_thr_1.5

**Story**: HBAR chimera-measure engine on measure_engines/norm_deviation: op_abs_gt_thr_1.5 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: chop, magnitude: all. Over TRAIN (n=11 fires): hit_rate=72.7%, expectancy +2.63% per fire; unit-compound +32.4%; max DD -0.3%. ShIC ratio 0.28 (strong, low memorization). 3-fold: sign-consistent (+13.3/+11.7/+0.0%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=HBAR; indicator_class=measure_engines/norm_deviation; indicator_config=op_abs_gt_thr_1.5; cadence=1d; regime_gate=chop; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #47 CHZ -- measure_event -- measure_engines/xd_btc_return op_abs_gt_thr_1.0

**Story**: CHZ chimera-measure engine on measure_engines/xd_btc_return: op_abs_gt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=12 fires): hit_rate=66.7%, expectancy +2.81% per fire; unit-compound +38.0%; max DD -4.0%. ShIC ratio 0.22 (strong, low memorization). 3-fold: sign-consistent (+7.2/+17.6/+9.5%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=CHZ; indicator_class=measure_engines/xd_btc_return; indicator_config=op_abs_gt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #48 ARB -- measure_event -- measure_engines/hbr_eta_buy op_lt_thr_1.0

**Story**: ARB chimera-measure engine on measure_engines/hbr_eta_buy: op_lt_thr_1.0 (z-score threshold crossing on the raw measure with expanding-window normalization). Regime: bull, magnitude: all. Over TRAIN (n=10 fires): hit_rate=100.0%, expectancy +1.87% per fire; unit-compound +20.2%; max DD +0.0%. ShIC ratio 0.27 (strong, low memorization). 3-fold: sign-consistent (+0.0/+7.1/+9.6%). Hold 1d, microstructure source: chimera v51 raw column.

**Recipe** (deterministic re-construction):
```
family=measure_event; asset=ARB; indicator_class=measure_engines/hbr_eta_buy; indicator_config=op_lt_thr_1.0; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=1; hold_days=1; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #49 XRP -- ta_state -- MA_state_SMA_above period_50

**Story**: XRP MA_state_SMA_above(period_50) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=11 bars in state): hit_rate=63.6%, per-fire expectancy +2.92%; unit-compound +34.1%; max DD -7.5%. ShIC ratio 0.25. 3-fold sign: sign-consistent. Hold 3d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=XRP; indicator_class=MA_state_SMA_above; indicator_config=period_50; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

## #50 LINK -- ta_state -- VWAP_state_above period_20

**Story**: LINK VWAP_state_above(period_20) STATE engine in bull-regime, all. Holds long while price is in the +1 state. Over TRAIN (n=13 bars in state): hit_rate=61.5%, per-fire expectancy +2.94%; unit-compound +42.2%; max DD -9.1%. ShIC ratio 0.14. 3-fold sign: sign-consistent. Hold 3d.

**Recipe** (deterministic re-construction):
```
family=ta_state; asset=LINK; indicator_class=VWAP_state_above; indicator_config=period_20; cadence=1d; regime_gate=bull; magnitude_filter=all; cluster_filter=5; hold_days=3; direction=long_fire; cost_mode=taker_bucket; sizing_rule=fixed_4pct
```
