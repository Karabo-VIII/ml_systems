#!/usr/bin/env python
"""
V4 Live Trading Engine
========================

Main loop that connects all production components:
  BarAccumulator -> SignalEngine -> RiskManager -> Exchange

Two modes:
  --paper:  Simulates trades with no exchange connection (default)
  --live:   Real orders on Binance (requires API keys in .env)

The engine:
  1. Connects to Binance WebSocket for aggTrades (or REST fallback)
  2. Accumulates trades into dollar bars
  3. On each bar completion, computes strategy signals
  4. If signal changes (entry/exit), executes via exchange
  5. Persists state to JSON for restart recovery
  6. Enforces risk limits (DD breaker, kill switch, position caps)

Usage:
    # Paper trading (default, safe)
    python src/prod/live_trader.py --paper

    # Paper trading with specific assets
    python src/prod/live_trader.py --paper --assets btcusdt ethusdt

    # Live trading (testnet)
    python src/prod/live_trader.py --live --testnet

    # Live trading (mainnet -- REAL MONEY)
    python src/prod/live_trader.py --live --mainnet

    # Emergency: sell everything and halt
    python src/prod/live_trader.py --liquidate

    # Status check
    python src/prod/live_trader.py --status
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prod.config import (
    ACTIVE_ASSETS, POLL_INTERVAL_SECONDS,
    ccxt_symbol, LOG_DIR,
)
from prod.bar_accumulator import MultiAssetAccumulator
from prod.signal_engine import SignalEngine
from prod.risk_manager import RiskManager
from prod.state_manager import StateManager
from prod.exchange import ExchangeWrapper

# --- Logging Setup ---
def setup_logging(log_level: str = "INFO"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"trader_{ts}.log"

    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )
    return logging.getLogger("prod.trader")


# --- REST-based Data Feed ---
def fetch_recent_trades_rest(exchange: ExchangeWrapper, symbol: str,
                              limit: int = 1000) -> list:
    """Fetch recent aggTrades via REST API.

    Returns list of {price, qty, timestamp, is_buyer_maker} dicts.
    Works in both live mode (authenticated) and paper mode (public API).
    """
    # Use authenticated exchange for live, public exchange for paper
    client = exchange.exchange
    if client is None:
        client = exchange.public_exchange
    if client is None:
        return []

    for attempt in range(3):
        try:
            trades = client.fetch_trades(symbol, limit=limit)
            return [
                {
                    "price": float(t["price"]),
                    "qty": float(t["amount"]),
                    "timestamp": int(t["timestamp"]),
                    "is_buyer_maker": t["side"] == "sell",
                }
                for t in trades
            ]
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate" in err_str or "banned" in err_str:
                wait = (attempt + 1) * 10  # 10s, 20s, 30s backoff
                logging.getLogger("prod.trader").warning(
                    "Rate limited on %s (attempt %d), waiting %ds",
                    symbol, attempt + 1, wait)
                time.sleep(wait)
            else:
                logging.getLogger("prod.trader").error(
                    "Failed to fetch trades for %s: %s", symbol, e)
                return []
    return []


# --- Main Trading Loop ---
def run_trading_loop(assets: List[str], paper: bool = True,
                      testnet: bool = True,
                      strategy_config_path: str = None):
    """Main trading loop.

    Args:
        assets: List of assets to trade (e.g., ['btcusdt', 'ethusdt'])
        paper: If True, simulate orders
        testnet: If True, use Binance testnet (ignored in paper mode)
        strategy_config_path: Path to selected_strategies.json override
    """
    logger = logging.getLogger("prod.trader")

    mode = "PAPER" if paper else ("TESTNET" if testnet else "MAINNET")
    logger.info("=" * 70)
    logger.info("  V4 LIVE TRADING ENGINE")
    logger.info("  Mode: %s | Assets: %d", mode, len(assets))
    logger.info("  Assets: %s", ", ".join(a.upper() for a in assets))
    logger.info("=" * 70)

    if not paper and not testnet:
        logger.warning("!!! MAINNET MODE -- REAL MONEY AT RISK !!!")
        logger.warning("Press Ctrl+C within 10 seconds to abort...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Aborted by user")
            return

    # Initialize components
    exchange = ExchangeWrapper(testnet=testnet, paper=paper)
    state = StateManager(name=mode.lower())
    risk = RiskManager()

    # Load WM model (V1.1 f25 solo -- sweep-validated)
    wm_model = None
    wm_device = None
    try:
        import torch
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "wm" / "v1"))
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "wm" / "v1" / "v1_0_training"))
        from cross_ensemble import CrossModelEnsemble
        from prod.signal_engine import PREFERRED_MODEL_KEY

        wm_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_key = PREFERRED_MODEL_KEY  # "v1_1_f25"
        logger.info("Loading WM model: %s on %s...", model_key, wm_device)
        wm_model = CrossModelEnsemble(model_keys=[model_key], device=wm_device)
        wm_model.eval()
        logger.info("WM model loaded: %d model(s)", len(wm_model.models))
    except Exception as e:
        logger.error("Failed to load WM model: %s", e)
        logger.warning("Continuing without WM -- strategies will be degraded")

    signal_engine = SignalEngine(min_agree=2, wm_available=wm_model is not None)

    # Load strategy config if provided
    if strategy_config_path:
        config_path = Path(strategy_config_path)
        if config_path.exists():
            signal_engine.load_config(config_path)

    # WM prediction generation
    ASSET_TO_IDX = {a.upper(): i for i, a in enumerate([
        "btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
        "dogeusdt", "adausdt", "avaxusdt", "linkusdt", "ltcusdt"])}
    SEQ_LEN = 96
    HORIZONS = [1, 4, 16, 64]

    def generate_wm_predictions(features, asset):
        """Generate WM predictions from feature array [n, n_features]."""
        if wm_model is None or features is None:
            return None
        import torch

        n = len(features)
        if n < SEQ_LEN:
            return None

        asset_idx = ASSET_TO_IDX.get(asset.upper(), 0)
        preds = {h: np.full(n, np.nan) for h in HORIZONS}

        # Use recent windows for prediction (last ~5 sequences)
        indices = list(range(max(0, n - SEQ_LEN * 5), n - SEQ_LEN, SEQ_LEN))
        if not indices:
            indices = [n - SEQ_LEN]

        try:
            with torch.no_grad():
                obs_list = [features[i:i + SEQ_LEN] for i in indices]
                obs = torch.from_numpy(
                    np.stack(obs_list)).float().to(wm_device)
                asset_t = torch.full(
                    (len(indices),), asset_idx,
                    dtype=torch.long, device=wm_device)
                with torch.amp.autocast("cuda", enabled=wm_device == "cuda"):
                    out = wm_model.forward_train(obs, asset_t)
                for h in HORIZONS:
                    dec = wm_model.bucketer.decode(
                        out["return_logits"][h]).cpu().numpy()
                    for si, start in enumerate(indices):
                        end = min(start + SEQ_LEN, n)
                        preds[h][start:end] = dec[si, :end - start]
        except Exception as e:
            logger.error("%s: WM prediction failed: %s", asset.upper(), e)
            return None

        return preds

    # Set up bar accumulators
    def on_bar_complete(asset, bar_dict, buffer):
        """Called when a new dollar bar completes for any asset."""
        acc = accumulators.get(asset)
        if acc is None:
            return

        bar_data = acc.get_buffer_arrays()
        if bar_data is None:
            return

        logger.info("%s: Bar #%d complete (buffer=%d, price=$%.2f)",
                   asset.upper(), bar_dict["bar_id"],
                   len(buffer), bar_dict["close"])

        # Compute features from bar buffer (30 base features)
        from prod.feature_calculator import compute_features_from_buffer
        features = compute_features_from_buffer(bar_data)

        # Generate WM predictions from features
        wm_preds = generate_wm_predictions(features, asset)

        # Compute signal (with WM predictions)
        sig = signal_engine.compute_signal(asset, bar_data, wm_preds=wm_preds)

        if not sig["changed"]:
            return

        logger.info("%s: Signal changed -> %s (vote=%d/%d, price=$%.2f)",
                   asset.upper(), sig["signal"], sig["n_long"],
                   sig["n_total"], sig["price"])

        ccxt_sym = ccxt_symbol(asset)
        current_price = sig["price"]

        # Check risk
        current_equity = state.last_equity or 1000.0

        # EXIT logic
        if state.has_position(asset):
            pos = state.get_position(asset)
            should_exit = risk.should_exit(
                asset, pos.entry_price, current_price, sig)

            if should_exit:
                logger.info("%s: CLOSING position (%.6f @ entry $%.4f)",
                           asset.upper(), pos.amount, pos.entry_price)
                order = exchange.place_market_order(
                    ccxt_sym, "sell", pos.amount,
                    reason=f"signal={sig['signal']}")
                if order:
                    exit_price = order.get("price", current_price) or current_price
                    state.record_exit(asset, exit_price)

        # ENTRY logic
        elif sig["position"] > 0.5:
            if risk.should_enter(asset, sig, current_equity):
                n_active = len(state.get_all_positions()) + 1
                alloc = risk.compute_position_size(
                    current_equity, sig["position"], n_active)

                if alloc >= 10:
                    amount = alloc / current_price
                    logger.info("%s: ENTERING $%.2f (%.6f units @ $%.2f)",
                               asset.upper(), alloc, amount, current_price)
                    strat_names = ",".join(
                        s["name"] for s in sig.get("strategies", [])
                        if s["position"] > 0.5)
                    order = exchange.place_market_order(
                        ccxt_sym, "buy", amount,
                        reason=f"vote={sig['n_long']}/{sig['n_total']}")
                    if order:
                        fill_price = order.get("price", current_price) or current_price
                        fill_amount = order.get("filled", amount) or amount
                        state.record_entry(
                            asset, ccxt_sym, fill_price,
                            fill_amount, strategy=strat_names or "consensus")

    accumulators = MultiAssetAccumulator(assets, on_bar_complete=on_bar_complete)

    # Pre-seed buffers from chimera data (cold-start elimination)
    logger.info("Pre-seeding bar buffers from chimera data...")
    total_seeded = 0
    for asset in assets:
        acc = accumulators.get(asset)
        if acc:
            n_seeded = acc.seed_from_chimera(n_bars=2000)
            total_seeded += n_seeded
    logger.info("Pre-seeded %d total bars across %d assets", total_seeded, len(assets))

    # Initial equity
    if not paper:
        equity = exchange.get_total_equity()
        state.update_equity(equity)
        risk.update_equity(equity)
        logger.info("Starting equity: $%.2f", equity)

    # Reconcile existing positions
    if not paper:
        real_positions = exchange.get_open_positions()
        for asset in assets:
            base = asset[:-4].upper()  # btcusdt -> BTC
            if state.has_position(asset) and base not in real_positions:
                logger.warning("Reconciliation: %s closed on exchange, updating state",
                             asset.upper())
                state.positions.pop(asset, None)
                state.save()

    # Main loop
    logger.info("Starting trading loop (Ctrl+C to stop)...")
    cycle = 0

    try:
        while True:
            cycle += 1

            # Check kill switch
            if risk.check_kill_switch():
                logger.critical("Kill switch active -- selling all positions")
                exchange.sell_all()
                break

            # Fetch trades for each asset (staggered to avoid rate limits)
            for ai, asset in enumerate(assets):
                acc = accumulators.get(asset)
                if acc is None:
                    continue

                ccxt_sym = ccxt_symbol(asset)
                try:
                    trades = fetch_recent_trades_rest(exchange, ccxt_sym)
                    if trades:
                        acc.process_trades_batch(trades)
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        logger.warning("Rate limited on %s, backing off 5s", asset.upper())
                        time.sleep(5)
                    else:
                        logger.error("Trade fetch error %s: %s", asset.upper(), e)

                # Stagger: 0.5s between assets to stay under rate limits
                # 10 assets x 0.5s = 5s per cycle, well under Binance 1200/min
                if ai < len(assets) - 1:
                    time.sleep(0.5)

            # Periodic equity update and status
            if cycle % 6 == 0:  # Every ~60 seconds (cycle is ~10s with stagger)
                if not paper:
                    equity = exchange.get_total_equity()
                    state.update_equity(equity)
                    risk.update_equity(equity)

                # Status
                status = accumulators.status()
                positions = state.get_all_positions()
                logger.info("Cycle %d | Positions: %d | %s",
                           cycle, len(positions),
                           " | ".join(
                               f"{a.upper()}: {s['bars']}bars "
                               f"({s['current_bar_pct']:.0f}%%)"
                               for a, s in status.items()
                               if s["bars"] > 0
                           ) or "warming up...")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")

    # Final state save
    state.save()
    logger.info("State saved. %s", state.status_summary())


# --- Entry Points ---
def run_status():
    """Print current trading state."""
    state = StateManager(name="paper")
    print("\n=== PAPER STATE ===")
    print(state.status_summary())

    for mode in ["testnet", "mainnet"]:
        s = StateManager(name=mode)
        if s.total_trades > 0 or s.last_equity > 0:
            print(f"\n=== {mode.upper()} STATE ===")
            print(s.status_summary())


def run_liquidation(testnet: bool = True):
    """Emergency liquidation."""
    exchange = ExchangeWrapper(testnet=testnet, paper=False)
    print("Liquidating all positions...")
    success = exchange.sell_all()
    if success:
        print("Liquidation complete.")
    else:
        print("WARNING: Some positions may not have been closed!")


def main():
    parser = argparse.ArgumentParser(
        description="V4 Live Trading Engine")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", default=True,
                       help="Paper trading mode (default)")
    group.add_argument("--live", action="store_true",
                       help="Live trading mode")
    group.add_argument("--liquidate", action="store_true",
                       help="Emergency: sell everything")
    group.add_argument("--status", action="store_true",
                       help="Print current state")

    parser.add_argument("--testnet", action="store_true", default=True,
                        help="Use Binance testnet (default for --live)")
    parser.add_argument("--mainnet", action="store_true",
                        help="Use Binance mainnet (REAL MONEY)")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Assets to trade (default: all 10)")
    parser.add_argument("--strategy-config", type=str, default=None,
                        help="Path to selected_strategies.json")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level")
    args = parser.parse_args()

    if args.status:
        run_status()
        return

    logger = setup_logging(args.log_level)

    if args.liquidate:
        testnet = not args.mainnet
        run_liquidation(testnet=testnet)
        return

    assets = args.assets or ACTIVE_ASSETS
    paper = not args.live
    testnet = not args.mainnet

    run_trading_loop(
        assets=assets,
        paper=paper,
        testnet=testnet,
        strategy_config_path=args.strategy_config,
    )


if __name__ == "__main__":
    main()
