# Strategy layer — research note

This note says which families are worth building on the gold the pipeline writes, what the previous project already measured, and which libraries not to reimplement. Links are pages opened while writing this note. The backtest core and the first three shortlisted strategies now live in `strategy/`.

Gold for a strategy is `load_gold` (`pipeline/dataset.py`). A bar is labelled at the close of `(t_open, t_close]`. Features on that row are knowable at `ts_ms`. Forward returns are `target_return_{1,4,16,64}` and are labels, not inputs. The embargo is already applied. Rules are in `docs/DATA_CONTRACT.md`.

## What the pipeline produces

Crypto (`--market crypto`), U10 by default: time bars 1m through 1d and dollar bars. Columns include OHLCV, buy and sell volume, trade count, the `norm_*` physics set (deviation, fractional difference, VPIN, flow imbalance, volume, range, Hurst, returns, efficiency, Hawkes, Yang–Zhang volatility, Kyle lambda), funding when the monthly file exists, and cross-asset `xd_*` when at least two assets are in the run. Open-interest features exist only with `--include-oi`.

FX (`--market fx`), seven majors by default: the same time bars, no dollar bars. Mid OHLC, tick count as `volume` and `n_trades`, null notional and null aggressor side, bid/ask/spread, and the FX feature list in the contract (trailing returns, realized / Parkinson / Garman–Klass vol, ATR in pips, spread stats, tick-intensity z-score, session flags, efficiency ratio, weekend gap, USD-strength when two or more USD legs are present). No interest rates, no forwards, no order book.

## Base families

Each family below is a signal definition, not a shipped backtest. "Pipeline" means the columns exist on gold today.

### Trend and moving-average crossovers

Signal: long when a short average of close is above a long average, flat otherwise. The usual pairs in the equity literature are 1/50, 1/150, 5/150, 1/200, and 2/200, sometimes with a band so near-crosses do not trade. Brock, Lakonishok, and LeBaron tested those on the Dow, 1897–1986, and reported that buy signals had higher subsequent returns than sell signals, and that this did not match a random walk, AR(1), GARCH-M, or EGARCH bootstrap ([Journal of Finance, 1992, PDF](https://repec.udesa.edu.ar/pub/Finanzas/Journals/Journal%20of%20Finance/47/5/2328994.pdf)).

Inputs: `close`. The pipeline has it for crypto and FX. Typical crypto use here is the close versus a slow average (a regime gate), not a fast cross. Failure modes: whipsaw in ranges, the lag that makes the entry late, and costs on every cross. A fast cross on one coin is not the diversified result in that paper. The previous project re-derived the null on the entry use; see below.

### Breakout channels (Donchian, Keltner, Bollinger)

Signal: long when close breaks the prior N-bar high, flat when it breaks the prior M-bar low. Donchian is that rule with no volatility term. Brock, Lakonishok, and LeBaron's second rule is the same idea on the Dow (50, 150, and 200 day highs and lows, optional 1% band); that is the evidence opened for this family, on equities, not on crypto or FX. Keltner is the same break of an average-true-range band around an average. Bollinger is a break of an average plus or minus k standard deviations, or the opposite (fade the band) for mean reversion.

Inputs: high, low, close. ATR needs those plus the previous close. The pipeline has OHLC. Crypto also has `norm_hl_spread` and `norm_yz_volatility`. FX has `atr_pips_*`. Typical parameters: Donchian 20/55 or 20 entry and 10 exit; Keltner 20-period, 2 ATR; Bollinger 20-period, 2 standard deviations. Failure modes: a breakout system pays for a long tail and loses in chop; a fade of Bollinger bands is the opposite bet and is run over in a trend. The channel width is a free parameter and is where overfitting shows up.

### Mean reversion

Signal: long when a trailing z-score of close, or the distance from a band, is below a threshold (often −2), flat when it re-enters. Inputs: close, and a trailing mean and variance. Crypto gold has `norm_deviation` and `norm_ma_distance`. FX gold has `close` and the return z-scores are not precomputed; a strategy can form them from `close`. Typical window: 20 to 100 bars. Failure mode: a z-score assumes the series returns. In a trend the signal stays negative and the book holds the loser. It also doubles up if the rule adds size as the z-score gets more extreme. No primary mean-reversion paper was opened for this note.

### Momentum and carry

Time-series momentum: long when the trailing return over h bars is positive, flat otherwise, often scaled by the inverse of trailing volatility. Inputs: `close` or `ret_*`, and a vol column. Crypto has `norm_return_*` and `norm_vol_cluster`. FX has `ret_{1,3,5,10,20,50}` and `rv_*`. Typical horizons in the equity and futures literature are 1, 3, and 12 months. On a 1-day bar those are 21, 63, and 252 bars. Failure mode: the signal is slow, the crash reversal (a positive trailing return into a gap down) is the left tail, and a single-asset book does not get the diversification the futures result is about. The previous project's note on that limit is cited below. No Moskowitz–Ooi–Pedersen paper was opened for this note.

FX carry: long the currency with the higher interest rate. Inputs: a rate or a forward point for each leg. The pipeline does not have them. `usd_strength_ret` is a price index, not a carry. Failure mode: the carry return is the interest differential minus the spot move, and the spot move in a funding crisis is larger than the carry. Do not approximate carry with the price trend.

### Volatility regime

Signal: trade the trend family only when trailing realized vol is in a chosen bucket (for example above its own median), and hold cash otherwise. Or the reverse, if the claim is that trends pay in quiet markets. Inputs: `rv_*` on FX, `norm_vol_cluster` and `norm_yz_volatility` on crypto. Typical window: 20 to 50 bars for the vol estimate, and a longer window for the bucket. Failure mode: the bucket is fit on the same sample as the trend rule, so the "regime" is a second free parameter. A vol estimate that uses the same bar's close is fine; one that uses a forward window is not.

### Session and time of day

Signal: allow a position only while a session flag is 1. The usual claim is that London–New York overlap has more directional flow than the Asian session. Inputs: `is_tokyo`, `is_london`, `is_newyork`, `is_ldn_ny_overlap` on FX gold. They are 1 when the bar overlaps the local session, including daylight-saving time. Crypto is 24/7; an hour-of-day feature is not in the default physics set (the bar timestamp is, so an hour can be derived without a new column). `weekend_gap_pips` is the reopen gap and is null until that bar. Failure mode: a session effect that is a volatility effect will look like a return effect until it is scaled by vol. A flag that is constant on a daily bar (the day overlaps every session) is not a signal.

### Order flow and microstructure

Trade imbalance: long when trailing buy volume exceeds sell volume. Crypto gold has `buy_volume`, `sell_volume`, `signed_usd`, and `norm_flow_imbalance`. FX gold does not: the Dukascopy file has no aggressor, and the side columns are null on purpose.

Cumulative volume delta is the running sum of buy minus sell. The pipeline stores the bar's signed notional, not a precomputed running sum. A strategy can cumulate it. Resetting the sum each day is a choice; not resetting it makes the level non-stationary.

VPIN: a rolling bucketed imbalance of buy and sell volume. Crypto gold has `norm_vpin`. It is a z-scored series from the bar's own buy and sell volume, not a reconstruction of a limit-order book. FX does not have it.

Footprint and absorption need trades at price levels inside the bar, or a book. The pipeline stores one OHLCV row per bar. Those strategies cannot be built from gold. They need a new bronze pass over raw trades, which this note does not add.

Funding, open interest, and basis are crypto-only. Funding is on gold when the monthly Vision file was fetched (`norm_funding`). Open interest is off unless `--include-oi`. A basis column exists on the daily panel path (`basis_daily` in `pipeline/features.py`) when the perp daily kline was fetched. Failure mode, already measured: using funding to tilt a long-only book did not beat equal weight. See below. No Easley–López de Prado–O'Hara paper was opened for this note.

## ML families

These are labels and models on top of a frozen feature row. They are not a reason to skip the base rules.

### Meta-labeling

A base rule emits a side. A second model predicts whether that bet will be profitable, and the side is taken only when the model says yes. The model must not see `target_return_*` or any column built from a future close. The pipeline's targets are the natural labels. The feature row is the natural input. Failure mode: the meta-model learns the base rule's in-sample winners and then vetoes the out-of-sample ones. Train it only on the train split, after the embargo.

### Triple-barrier labels

A label is +1 if the upper barrier (in return or ATR) is touched first, −1 if the lower barrier is touched first, and 0 if a vertical time barrier is hit first. The pipeline's default label is the simpler uncapped forward return, not a barrier label. A barrier label can be built from high, low, and close of the next h bars, and it is still a label: it must not be joined back as a feature. The path touches are future information. Failure mode: the barrier width is tuned until the in-sample hit rate looks good. Freeze the width before looking at the out-of-sample split.

### Linear and gradient-boosting baselines

Ridge (or plain least squares) on the causal feature columns, predicting `target_return_1` or the sign of it, is the baseline a tree model has to beat. A boosted tree on the same columns and the same purged split is the second baseline. Both are fit on train, and the embargo rows are not in either side of a split boundary. Failure mode: a tree will use a feature that is constant on train and informative only because of one regime. The constant gate in `pipeline/validate.py` is there because that happened before. Do not add a feature the validator rejected.

### Regime classifiers

A classifier predicts a discrete state (trend up, trend down, range) from trailing features, and a separate rule trades that state. The state label, if it is built from future returns, is a label. If it is built from trailing returns only, it is just a feature and the "classifier" is a threshold. Failure mode: the state is defined with the same return the strategy is scored on, so the classifier looks perfect in sample.

### Sequence models

A sequence model reads the last L bars and predicts the next return. L is a lookback. The embargo has to cover L as well as the label horizon. The contract's embargo is `max(horizon, feature lookback)` in bars and at least 7 days, with lookbacks capped at 90 days. A sequence model whose window is longer than the cap used at dataset build is not covered by that embargo. Either shorten the window or rebuild gold with the longer lookback. Failure mode: normalization fit on the whole series, including the test period. Fit the scaler on train only.

### Validation rules

Purged cross-validation: a training fold does not contain bars whose label window overlaps the test fold. The pipeline embargo drops `embargo_bars` at each split edge for that reason. Do not turn it off to "recover" samples.

The deflated Sharpe ratio raises the Sharpe a backtest has to clear as the number of tried configurations grows, and it uses the skewness and kurtosis of the return series rather than assuming normality. Bailey and López de Prado, Journal of Portfolio Management 40(5), 2014 ([PDF](https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf)). A single reported Sharpe from a grid search is not the number to ship.

CSCV / PBO (combinatorially symmetric cross-validation, probability of backtest overfitting) splits the returns into blocks, scores every configuration on half the blocks, and asks how often the in-sample winner is below the median out of sample. The previous project treated a high PBO as a failed gate. The paper that defines PBO was not opened for this note; the deflated-Sharpe PDF above cites the 2013 SSRN draft. Use the project's own pre-registered scorecard when comparing to old results, and do not quote a PBO this note did not recompute.

## Libraries

Adopt a library only if it removes code we would otherwise write and the license can sit next to this repo. Versions and status are what the opened pages showed on 2026-11-20. Commit recency was not measured when the page did not show it.

| Library | License, from the file opened | What was opened | Fit | Recommendation |
|---|---|---|---|---|
| TA-Lib | BSD-style 3-clause, copyright 1999–2026, Mario Fortier. [LICENSE](https://raw.githubusercontent.com/TA-Lib/ta-lib/main/LICENSE) | The license file. | C indicator library. We already compute the averages, ATR, and returns the first shortlist needs. | Add it when a strategy needs an indicator we do not want to re-derive. Not before. |
| pandas-ta | The PyPI JSON opened for 0.4.71b0 (uploaded 2025-09-14, requires Python ≥ 3.12) has `license` null. The GitHub license URL tried returned 404. | [PyPI JSON](https://pypi.org/pypi/pandas-ta/json). | Indicator functions on dataframes. | Do not add it until a license file is actually readable. |
| vectorbt | Apache 2.0 plus the Commons Clause. The clause removes the right to sell the software. [LICENSE.md](https://raw.githubusercontent.com/polakowo/vectorbt/master/LICENSE.md) | The license file, which carries a 2026 copyright line. | Fast parameter sweeps over OHLC. | Usable for private research sweeps. Do not depend on it inside a product, because of the Commons Clause. |
| backtesting.py | Not established. `LICENSE.txt` on master returned 404. The README that was opened does not state a license. | [README](https://raw.githubusercontent.com/kernc/backtesting.py/master/README.md), [pyproject.toml](https://raw.githubusercontent.com/kernc/backtesting.py/master/pyproject.toml). | Small event-driven backtest. The README example is a moving-average cross. | Do not adopt until the license file is read. |
| NautilusTrader | GNU LGPL 3.0. [LICENSE](https://raw.githubusercontent.com/nautechsystems/nautilus_trader/develop/LICENSE) | The license file. | Full trading engine. Far beyond a research backtest on gold. | Do not link it into this package. A later execution service can be a separate process. |
| freqtrade | GNU GPL 3.0. [LICENSE](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/LICENSE) | The license file. | Crypto bot and a strategy repository. GPL would cover a derivative of those strategies if we copy the code. | Read ideas. Do not copy strategy source into this repo. |
| Jesse | MIT, copyright 2020 Jesse.Trade. [LICENSE](https://raw.githubusercontent.com/jesse-ai/jesse/master/LICENSE) | The license file. | Another crypto strategy framework. | Not needed for the first backtest. MIT would allow it later. |
| Qlib | MIT, Microsoft. [LICENSE](https://raw.githubusercontent.com/microsoft/qlib/main/LICENSE) | The license file and the README. | ML research stack aimed at its own data format. | Do not adopt the data layer. We already have gold, splits, and an embargo. |
| mlfinlab | All rights reserved. The README says business and enterprise licenses, and that the public repo is for issues. [README](https://raw.githubusercontent.com/hudson-and-thames/mlfinlab/master/README.md) | The README. | Triple barrier, purged CV, PBO. | Do not copy it. The embargo is already in this pipeline. PBO, if we want it, gets implemented from the paper, not from that tree. |

## What the old project already measured

The named live books `s1_ma10x50`, `atrch55x3`, `vwma20x100`, the ml3 ridge and lgbm variants, and `FLEET_DASHBOARD` are not in the committed snapshot. `git ls-files` has no path containing those names, and a search of `ml_systems_v1` for the strings returned no matches. There is no `ml_systems_v1/crypto/runs/strat/ladder/` directory. Nothing below invents a result for a book that is not in the tree. `ml_systems_v1/crypto/src/strat/ladder/merit_scorecard.py` was also not found.

These files were read:

`ml_systems_v1/crypto/runs/strat/MA_FAILURE_DIAGNOSIS_2026-06-10.md`. The entry-trigger use of a moving average (when to get in) was the weak use, and the experiments on that corner were null. A 200-day regime book was reported as beating its phase-shift null (Calmar 19.8 against a null p95 of 10.8). A vol-scaled time-series-momentum ensemble was reported as 41% against 21% for an exposure-matched null. Both were described as the same object: a drawdown-managed trend premium of about 25–48% a year, short of 2x a year. The note says 2x a year appeared in 1 of 7 years (2021, +519%) and was beta. Those figures are the file's, not a new backtest.

`ml_systems_v1/crypto/runs/strat/FIND_LO_WINNER_VERDICT_2026-06-10.md`. Verdict in the file: no robust long-only, no-leverage winner, and 2x a year is not reachable at bar level under spot and leverage 1. On an UNSEEN window described as January–May 2026, families beat buy-and-hold by going to cash, and 4 of 5 lost money. The table in that file: F2 mover-rotation UNSEEN CAGR +1080%/yr but the gate is INCOMPLETE→FAIL; F3 Renko −20%/yr FAIL; F1 Chandelier −24%/yr FAIL; F4 setup-chaser −33%/yr FAIL with PBO 0.79; F5 about 0 or −44%/yr FAIL. The file then says the F2 flicker is not a winner because the firewall and PBO were not run.

`ml_systems_v1/crypto/runs/strat/FAMILY_ENSEMBLE_UNSEEN.md`. Sealed UNSEEN window 2025-12-31 to 2026-06-01, 149 days. Equal-weight buy-hold net −8.1%, max drawdown −40.2%. The family-ensemble book net −0.9%, max drawdown −2.3%, Sharpe −0.83. Scorecard on the full daily stream, 2338 days: selection compound +90.2% (Sharpe 0.98, max drawdown −25.7%, 1898 days), OOS compound −19.4% (Sharpe −1.25, 291 days), UNSEEN −0.9%. Full block-bootstrap p05 −17.9%. Held-out p05 −38.8%. The file says `ship = FALSE`. The preservation came from being in cash, not from a timing edge.

`ml_systems_v1/crypto/runs/strat/FUNDING_TILT_LONGONLY.md`. A long-only tilt toward low or negative funding did not beat equal weight. The file's verdict is ARTIFACT: the tilt-minus-equal-weight difference was negative in every cell of the sweep, in sample and held out, gross and net. Panel stated there: 2338 days by 48 assets, 2020-01-03 to 2026-05-28.

`ml_systems_v1/crypto/runs/strat/fleet_search_20260620.json`. Generated 2026-06-20, 50 assets, hold 7. It ranks single features and a few fleets by a field named `mean_10s`. The producer of that field was not found in `ml_systems_v1/crypto/src/strat/fleet_search.py` (no match), so `mean_10s` is not reported here as a return. The rank order in the file, highest `mean_10s` first, starts mom30, mom14, 4h_funding, rsi14, mom7.

`ml_systems_v1/crypto/STRAT_DISCOVERY_METHODOLOGY.md` and `USAGE_STRATEGY.md` describe the old discovery loop and the chimera dataset. They are not performance reports. The methodology file's own premise is that no strategy works on every asset and regime, and that a prediction setup (yesterday's features to tomorrow's cluster) had already failed three times. That is why this note starts with slow rules on gold, not with a sequence model.

## Backtest core

`strategy/` is a small numpy/polars package, not a second data model. It reads gold through `load_gold` and does not fetch or touch bronze.

vectorbt was considered for parameter sweeps. Its license is Apache 2.0 plus the Commons Clause, which removes the right to sell the software (see the library table above). That is a strong reason not to depend on it inside this package. The engine is therefore self-written and vectorised. It is short enough to check by hand, which is the point of the tests in `tests/test_backtest.py`.

The research sketch below asked the position function to pre-shift the signal. The engine now does that itself, so a strategy function returns the target decided at bar t's close and must not shift it. The fill is on bar t+1, at `open` if that column is present and usable, otherwise at `close`. Positions are in `[-1, 1]`. Flat is zero. The first three shortlisted strategies are still long-or-cash.

Costs are turnover-based: `|delta position| * (fee + slippage + half-spread)` in return units, charged on the fill bar. Values live in `config/pipeline.yaml` under `backtest`. `fee_bps` is a published commission (Binance spot Regular User 10 bps for crypto; Dukascopy Europe JForex self-trader 0.35 bps at the lowest deposit tier for FX). `slippage_bps` is an assumption (2 bps for crypto, carried over from the archived prod config; 0 for FX until measured). `half_spread_bps` is a fallback of 0. FX gold uses `spread_mean` or `bid_close`/`ask_close` when those columns are present.

Walk-forward folds reuse the pipeline's cadence-aware embargo rule: `max(horizon, lookback)` in that file's bars, and at least 7 calendar days, via `load_embargo_policy` and `bar_duration_ms`. The embargo is applied on both sides of a cut, matching `apply_embargo`, and is not shrunk to fit a short remainder.

Usage:

```python
from pipeline.dataset import load_gold
from strategy import regime_gate, donchian_breakout, vol_scaled_momentum
from strategy import run_backtest, buy_and_hold, walk_forward

bars = load_gold("BTCUSDT", "time_1d")
pos = regime_gate(bars)                 # or donchian_breakout, vol_scaled_momentum
bt = run_backtest(bars, pos, market="crypto")
print(bt.metrics)                       # CAGR, vol, Sharpe, Sortino, max drawdown,
                                        # turnover, hit rate, exposure, cost drag
bh = buy_and_hold(bars, market="crypto")
folds = walk_forward(bars, bar_type="time_1d", lookback=200, test_bars=90)
```

`buy_and_hold` is a constant unit weight through the same engine and fill rule, so a constant-long strategy equals that path minus costs. Pass `costs=CostModel(0, 0, 0)` for a gross benchmark.

## A minimal strategy layer

Four functions, in this order. None of them fetch, and none of them read bronze.

1. `signal(gold) -> frame`. Adds one column. It may use only columns whose values at `ts_ms` are knowable then. It must not read `target_return_*`. The shipped functions (`regime_gate`, `donchian_breakout`, `vol_scaled_momentum`) return a target-position array instead of adding a column; the rule is the same.
2. `position(signal_frame) -> frame`. The engine now applies this step: a target decided at the close of bar t is filled on bar t+1. Strategies must not pre-shift. Weights may be long, flat, or short in `[-1, 1]`. The first three shortlisted strategies stay long or cash. Flat is zero.
3. `backtest(weights, gold, cost) -> equity`. This is `run_backtest`. Cost comes from the `backtest` block in `config/pipeline.yaml`, or from a passed `CostModel`. Crypto and FX costs differ.
4. `evaluate(equity, gold) -> report`. `BacktestResult.metrics` is the current report. When gold has a `split` column, later work should still drop rows labelled `embargo` and report a Sharpe together with the number of configurations tried, so a deflated Sharpe can be computed later. Does not refit.

The first implementation calls `load_gold` for one bar type and one asset. It does not grow a second data model.

## First shortlist

In this order. Each one is small, uses columns that exist, and is there because the old measurements said the slow trend use is the one that cleared a null, while the fast entry and the funding tilt did not.

1. **Slow regime gate.** Long the asset when close is above its 200-bar average, otherwise cash. Daily and 4h, crypto U10 and FX majors. This is the use the MA diagnosis said beat a phase-shift null, and it is not the fast cross that came up null.
2. **Donchian breakout.** Long when close breaks the prior 20-bar high, flat when it breaks the prior 10-bar low. Same universes, 4h and 1d. OHLC is on both markets.
3. **Vol-scaled time-series momentum.** Sign of the trailing 63-bar return on the daily bar, weight proportional to the inverse of trailing realized vol, equal risk across the universe, long or cash. FX uses `rv_*`. Crypto uses a vol column already on gold (`norm_yz_volatility` or a trailing std of `close`, computed in the strategy, not a new pipeline feature).
4. **Session gate on the FX trend book.** Take strategy 1 or 2 on FX only while `is_ldn_ny_overlap` is 1. This is a filter, not a new signal, and it is the reason the session flags exist.
5. **Volatility-bucket switch.** Run strategy 1 only when `rv_20` (FX) or `norm_vol_cluster` (crypto) is above its trailing median. One extra parameter. Pre-register the median window before looking at OOS.
6. **Ridge baseline.** Predict `target_return_1` from the causal feature columns with ridge, train split only, embargo respected. The boosted tree is the same design and is second. It has to beat strategy 1 on the OOS split, not on train.
7. **Do not build next:** a fast MA cross as an entry, a long-only funding tilt, or a carry book. The first two were measured and did not clear their gates. Carry has no rate column.

## Pipeline columns still missing

- Interest rates or forward points, for FX carry.
- A traded notional for FX. Tick count is the activity proxy. It is not volume, and it cannot support an Amihud ratio or a dollar bar.
- An aggressor side for FX, so imbalance, CVD, and VPIN stop at crypto.
- Trades at price, or a book, for footprint and absorption. Gold is one row per bar.
- A maintained open-interest series. It is behind `--include-oi` and was not part of the default crypto feature set because of a lookahead failure on the old panel.
- Crosses and a triangle residual. The old FX manifest had them. This fetch does not build them until a cross universe is configured and fetched.
- A liquidation print. The old daily proxy is not in the new feature set.
- An explicit hour-of-week column on crypto. The timestamp is there; the column is not.
