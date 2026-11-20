"""G1: Live testnet harness scaffold for champion strategies.

Closes G1 from gap audit 2026-04-25.

What this is:
  Daily-cadence runner that wires the deployed champion blend
  (recommended_2sleeve_v2: xsec K=10+10 + frontier_dib_flow_both) to a
  Binance testnet account. Posts real orders against testnet (no real money),
  collects actual fill statistics (realized p_fill, slippage, latency, partial
  fill rates), and writes a daily reconciliation report.

What this is NOT (yet):
  - Production live trading. Use TESTNET only until 14-day validation passes.
  - A realtime tick handler. This is daily-cadence, matching the strategies'
    native horizon. Sub-day infrastructure (G3 / P5) remains a separate gap.

Usage:
  python -m src.frontier.live.testnet_runner --mode dry    # no network
  python -m src.frontier.live.testnet_runner --mode paper  # market reads, no orders
  python -m src.frontier.live.testnet_runner --mode live --testnet  # actual testnet orders

Required env (live --testnet only):
  BINANCE_TESTNET_API_KEY
  BINANCE_TESTNET_API_SECRET

Output:
  logs/frontier/live_testnet/YYYY-MM-DD/
    signals.json        # daily signal output (picks + sizes)
    fills.json          # per-order outcomes
    reconciliation.json # backtest_pred vs realized
    daily_pnl.csv       # equity curve
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "logs" / "frontier" / "live_testnet"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Binance testnet endpoints (public)
SPOT_TESTNET_BASE = "https://testnet.binance.vision"
FAPI_TESTNET_BASE = "https://testnet.binancefuture.com"


@dataclass
class TradeIntent:
    asset: str
    side: str          # "buy" | "sell"
    weight_target: float  # fraction of book equity (e.g., 0.05 = 5%)
    rationale: str     # which sleeve generated it (xsec / dib_flow)


@dataclass
class FillRecord:
    asset: str
    side: str
    intended_qty: float
    qty_filled: float
    avg_fill_price: float
    p_fill: float                   # qty_filled / intended_qty
    reposts: int
    fell_back_to_taker: bool
    order_ids: list = field(default_factory=list)


@dataclass
class SessionState:
    session_date: str
    book_equity: float
    intents: list[TradeIntent]
    fills: list[FillRecord]
    btc_30d: Optional[float] = None
    regime_blocked: bool = False
    portfolio_dd_halt_triggered: bool = False


def get_book_equity(client, mode: str) -> float:
    if mode in ("dry", "paper"):
        return 10000.0
    try:
        bal = client.get_balance()
        return float(bal.get("USDT", 0.0)) + float(bal.get("BUSD", 0.0))
    except Exception as e:
        print(f"[testnet] balance fetch failed: {e}; falling back to 10k")
        return 10000.0


def generate_signals_xsec_K10_10(today: str) -> list[TradeIntent]:
    """STUB: should call the xsec_xgb_walkforward.py training pipeline.

    Real implementation:
      1. Build panel up to today using build_panel() from xsec_xgb_walkforward.
      2. Train XGB ranker on panel[:today] (or load cached model).
      3. Predict score on today's row per asset.
      4. Pick top-K=10 long, bottom-K=10 short (delta-neutral).
      5. Apply meta-gate (v8) + regime-gate.
    Skipped here to keep scaffold tight; real training run takes ~3 minutes.

    Returns dummy 2 long + 2 short for harness validation.
    """
    return [
        TradeIntent("BTC", "buy", 0.05, "xsec_K10_10_long"),
        TradeIntent("ETH", "buy", 0.05, "xsec_K10_10_long"),
        TradeIntent("DOGE", "sell", 0.05, "xsec_K10_10_short"),
        TradeIntent("AVAX", "sell", 0.05, "xsec_K10_10_short"),
    ]


def generate_signals_dib_flow(today: str) -> list[TradeIntent]:
    """STUB: should call dib_flow_duo.

    Real implementation:
      1. Compute DIB flow for BTC + ETH on today's bars.
      2. If both flow > 0: long BTC + ETH next day.
      3. If both < 0: do nothing (long-only on duo signal).
    """
    return [
        TradeIntent("BTC", "buy", 0.075, "dib_flow_both_long"),
        TradeIntent("ETH", "buy", 0.075, "dib_flow_both_long"),
    ]


def execute_intent(client, intent: TradeIntent, equity: float, price: float, mode: str) -> FillRecord:
    """Convert intent -> order via maker_fill."""
    qty_target = (intent.weight_target * equity) / max(price, 1e-9)
    if mode == "dry":
        return FillRecord(
            asset=intent.asset, side=intent.side,
            intended_qty=qty_target, qty_filled=qty_target,
            avg_fill_price=price, p_fill=1.0, reposts=0,
            fell_back_to_taker=False, order_ids=[f"DRY-{intent.asset}-{intent.side}"],
        )
    if mode == "paper":
        # paper mode in client simulates fill; we read the result
        from ...growth.executors.maker_spot import maker_fill  # type: ignore
        result = maker_fill(
            client, f"{intent.asset}USDT", intent.side, qty_target,
            offset_bps=2.0, max_wait_s=30.0, max_reposts=3,
            allow_taker_fallback=False, venue="spot",
        )
        p_fill = result.qty_filled / max(qty_target, 1e-12)
        return FillRecord(
            asset=intent.asset, side=intent.side,
            intended_qty=qty_target, qty_filled=result.qty_filled,
            avg_fill_price=result.avg_fill_price, p_fill=p_fill,
            reposts=result.reposts, fell_back_to_taker=result.fell_back_to_taker,
            order_ids=result.order_ids,
        )
    # live (testnet) — same path as paper but real network
    from ...growth.executors.maker_spot import maker_fill  # type: ignore
    result = maker_fill(
        client, f"{intent.asset}USDT", intent.side, qty_target,
        offset_bps=2.0, max_wait_s=30.0, max_reposts=3,
        allow_taker_fallback=False, venue="spot",
    )
    p_fill = result.qty_filled / max(qty_target, 1e-12)
    return FillRecord(
        asset=intent.asset, side=intent.side,
        intended_qty=qty_target, qty_filled=result.qty_filled,
        avg_fill_price=result.avg_fill_price, p_fill=p_fill,
        reposts=result.reposts, fell_back_to_taker=result.fell_back_to_taker,
        order_ids=result.order_ids,
    )


def run_session(mode: str, testnet: bool, capital_cap: float = 100.0) -> SessionState:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = LOG_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[testnet] === Session {today} | mode={mode} testnet={testnet} ===")

    # Build client
    if mode == "dry":
        client = None
    else:
        from ...growth.binance_client import BinanceClient  # type: ignore
        from ...growth.config import Config  # type: ignore
        cfg = Config.from_env(prefix="BINANCE_TESTNET_") if testnet else Config.from_env()
        if testnet:
            cfg.spot_base = SPOT_TESTNET_BASE
            cfg.fapi_base = FAPI_TESTNET_BASE
        client = BinanceClient(cfg, mode="paper" if mode == "paper" else "live")

    equity = min(get_book_equity(client, mode), capital_cap)  # safety cap
    print(f"[testnet] book_equity (capped at {capital_cap}): {equity:.2f}")

    # Generate signals
    intents = generate_signals_xsec_K10_10(today) + generate_signals_dib_flow(today)
    # Aggregate weights per (asset, side)
    consolidated = {}
    for it in intents:
        key = (it.asset, it.side)
        consolidated[key] = consolidated.get(key, 0.0) + it.weight_target
    intents_final = [
        TradeIntent(asset=a, side=s, weight_target=w, rationale="consolidated")
        for (a, s), w in consolidated.items() if w > 0.005
    ]
    print(f"[testnet] {len(intents_final)} intents after consolidation")

    # Get prices
    prices = {}
    for it in intents_final:
        if mode in ("dry",):
            prices[it.asset] = 100.0  # dummy
        else:
            try:
                t = client.get_ticker(f"{it.asset}USDT", venue="spot")
                prices[it.asset] = float(t.get("last", 0.0))
            except Exception as e:
                print(f"[testnet] ticker fail {it.asset}: {e}")
                prices[it.asset] = 0.0

    fills = []
    for it in intents_final:
        if prices.get(it.asset, 0.0) <= 0:
            print(f"[testnet] skip {it.asset}: no price")
            continue
        try:
            f = execute_intent(client, it, equity, prices[it.asset], mode)
            fills.append(f)
            print(f"[testnet] {it.asset} {it.side} qty={f.intended_qty:.6f} "
                  f"filled={f.qty_filled:.6f} p_fill={f.p_fill:.2%} "
                  f"reposts={f.reposts}")
        except Exception as e:
            print(f"[testnet] EXEC FAIL {it.asset}: {e}")

    state = SessionState(
        session_date=today, book_equity=equity,
        intents=intents_final, fills=fills,
    )
    # Persist
    (out_dir / "signals.json").write_text(json.dumps(
        [asdict(i) for i in intents_final], indent=2))
    (out_dir / "fills.json").write_text(json.dumps(
        [asdict(f) for f in fills], indent=2))
    p_fills = [f.p_fill for f in fills]
    realized_p_fill = sum(p_fills) / len(p_fills) if p_fills else 0.0
    (out_dir / "reconciliation.json").write_text(json.dumps({
        "session_date": today,
        "n_intents": len(intents_final),
        "n_filled": sum(1 for f in fills if f.qty_filled > 0),
        "mean_p_fill": realized_p_fill,
        "mean_p_fill_assumption_in_backtest": 0.30,
        "delta_vs_assumed": realized_p_fill - 0.30,
        "any_taker_fallback": any(f.fell_back_to_taker for f in fills),
    }, indent=2))
    print(f"[testnet] session done. mean_p_fill={realized_p_fill:.1%} (vs 0.30 assumed).")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry", "paper", "live"], default="dry")
    parser.add_argument("--testnet", action="store_true",
                        help="Use Binance testnet endpoints (live mode only).")
    parser.add_argument("--capital-cap", type=float, default=100.0,
                        help="Hard cap on capital used (USDT). Default 100.")
    args = parser.parse_args()
    if args.mode == "live" and not args.testnet:
        print("[testnet] REFUSING to run live without --testnet flag.")
        print("[testnet] Pass --testnet to use https://testnet.binance.vision (no real money).")
        return
    run_session(mode=args.mode, testnet=args.testnet, capital_cap=args.capital_cap)


if __name__ == "__main__":
    main()
