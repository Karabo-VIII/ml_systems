#!/usr/bin/env python
"""
Universe Screener
===================

Screens the top N crypto assets by volume/volatility to build the
daily trading universe. Uses Binance public REST API (no auth needed).

Process:
  1. Fetch all USDT spot tickers from Binance
  2. Filter by minimum daily volume ($5M+)
  3. Score by: quote volume rank + volatility rank + spread rank
  4. Output top N assets sorted by score
  5. Save to JSON for consumption by live_trader

This runs daily before the trading session to select which assets
to trade. Assets not in the original 10 (BTC, ETH, SOL, etc.)
can still be traded by price-action strategies -- they just won't
have WM predictions (WM models were trained on 10 specific assets).

Usage:
    python src/prod/universe_screener.py                    # Top 20 (default)
    python src/prod/universe_screener.py --top 50           # Top 50
    python src/prod/universe_screener.py --min-volume 10    # Min $10M daily vol
"""
import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prod.config import STATE_DIR, LOG_DIR

logger = logging.getLogger("prod.screener")

# Assets with trained WM models (can use WM-filtered strategies)
WM_ASSETS = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
}

# Stablecoins, wrapped tokens, gold/fiat-backed, and non-crypto to exclude
EXCLUDED = {
    # Stablecoins
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT", "FDUSDUSDT",
    "USDPUSDT", "USDTUSDT", "RLUSDUSDT", "USD1USDT", "PYUSDUSDT",
    # Fiat pairs
    "EURUSDT", "GBPUSDT", "AUDUSDT", "BRLUSDT", "TRYUSDT",
    # Wrapped/staked tokens
    "WBTCUSDT", "WBETHUSDT", "BETHUSDT", "STETHUSDT",
    # Gold/commodity-backed
    "XAUTUSDT", "PAXGUSDT",
    # Leverage tokens and other non-standard
    "ZBTUSDT",
}


def fetch_tickers_public() -> List[Dict]:
    """Fetch all tickers from Binance public API (no auth needed)."""
    try:
        import ccxt
        exchange = ccxt.binance({"options": {"defaultType": "spot"}})
        tickers = exchange.fetch_tickers()
        return [
            {
                "symbol": symbol,
                "base": symbol.split("/")[0] if "/" in symbol else symbol[:-4],
                "last": float(t.get("last", 0) or 0),
                "quoteVolume": float(t.get("quoteVolume", 0) or 0),
                "high": float(t.get("high", 0) or 0),
                "low": float(t.get("low", 0) or 0),
                "bid": float(t.get("bid", 0) or 0),
                "ask": float(t.get("ask", 0) or 0),
            }
            for symbol, t in tickers.items()
            if symbol.endswith("/USDT") and t.get("last")
        ]
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        return []
    except Exception as e:
        logger.error("Failed to fetch tickers: %s", e)
        return []


def score_assets(tickers: List[Dict],
                 min_volume_usd: float = 5_000_000,
                 top_n: int = 20) -> List[Dict]:
    """Score and rank assets for the trading universe.

    Scoring criteria:
      1. Quote volume (24h USDT volume) -- liquidity
      2. Volatility ((high-low)/low * 100) -- opportunity
      3. Spread (ask-bid)/last -- execution cost (lower is better)

    Assets with WM models get a bonus in scoring because they can
    use WM-filtered strategies (higher walk-forward Sharpe).
    """
    # Filter
    valid = []
    excluded_reasons = {}
    for t in tickers:
        sym = t["symbol"].replace("/", "")

        # Hard exclusion list
        if sym in EXCLUDED:
            excluded_reasons[sym] = "excluded_list"
            continue

        # No price
        if t["last"] <= 0:
            excluded_reasons[sym] = "no_price"
            continue

        # Below min volume
        if t["quoteVolume"] < min_volume_usd:
            excluded_reasons[sym] = f"low_vol(${t['quoteVolume']:,.0f})"
            continue

        # Auto-detect stablecoins: 24h volatility < 0.5%
        vol_pct = ((t["high"] - t["low"]) / t["low"] * 100) if t["low"] > 0 else 0
        if vol_pct < 0.5:
            excluded_reasons[sym] = f"stablecoin(vol={vol_pct:.2f}%)"
            continue

        # Auto-detect name patterns for stablecoins/fiat
        base = t.get("base", "").upper()
        if any(s in base for s in ["USD", "EUR", "GBP", "BRL", "TRY", "AUD",
                                     "BUSD", "TUSD", "USDC", "DAI", "FDUSD"]):
            excluded_reasons[sym] = f"fiat_pattern({base})"
            continue

        # Min price sanity ($0.0000001 tokens are untradeable)
        if t["last"] < 0.000001:
            excluded_reasons[sym] = "dust_price"
            continue

        valid.append(t)

    if excluded_reasons:
        n_show = min(5, len(excluded_reasons))
        sample = list(excluded_reasons.items())[:n_show]
        logger.info("Excluded %d assets (sample: %s)",
                   len(excluded_reasons),
                   ", ".join(f"{s}:{r}" for s, r in sample))

    if not valid:
        return []

    # Compute metrics
    for t in valid:
        t["volatility"] = ((t["high"] - t["low"]) / t["low"] * 100
                           if t["low"] > 0 else 0)
        spread = (t["ask"] - t["bid"]) / t["last"] * 10000 if t["last"] > 0 else 999
        t["spread_bps"] = max(spread, 0)  # Negative spread = stale data
        t["has_wm"] = t["symbol"].replace("/", "").upper() in WM_ASSETS

    # Rank
    valid.sort(key=lambda x: x["quoteVolume"], reverse=True)
    for i, t in enumerate(valid):
        t["vol_rank"] = i + 1

    valid.sort(key=lambda x: x["volatility"], reverse=True)
    for i, t in enumerate(valid):
        t["volatility_rank"] = i + 1

    valid.sort(key=lambda x: x["spread_bps"])
    for i, t in enumerate(valid):
        t["spread_rank"] = i + 1

    # Composite score (lower is better)
    # Volume and volatility are opportunity; spread is cost
    # WM assets get 20-rank bonus (prefer tradeable with WM signal)
    for t in valid:
        wm_bonus = -20 if t["has_wm"] else 0
        t["score"] = t["vol_rank"] + t["volatility_rank"] + t["spread_rank"] + wm_bonus

    valid.sort(key=lambda x: x["score"])

    # Return top N
    results = []
    for t in valid[:top_n]:
        sym_clean = t["symbol"].replace("/", "").lower()
        results.append({
            "symbol": t["symbol"],
            "asset": sym_clean,
            "price": t["last"],
            "volume_24h": round(t["quoteVolume"]),
            "volatility_pct": round(t["volatility"], 2),
            "spread_bps": round(t["spread_bps"], 2),
            "has_wm": t["has_wm"],
            "score": t["score"],
            "ranks": {
                "volume": t["vol_rank"],
                "volatility": t["volatility_rank"],
                "spread": t["spread_rank"],
            },
        })

    return results


def run_screener(top_n: int = 20, min_volume: float = 5.0):
    """Run the universe screener and save results."""
    min_vol_usd = min_volume * 1_000_000

    print("=" * 70)
    print("  UNIVERSE SCREENER")
    print(f"  Top {top_n} | Min Volume: ${min_volume:.0f}M")
    print("=" * 70)

    print("\n  Fetching tickers from Binance...")
    tickers = fetch_tickers_public()
    if not tickers:
        print("  [ERROR] No tickers fetched")
        return []

    usdt_count = len(tickers)
    print(f"  {usdt_count} USDT pairs found")

    results = score_assets(tickers, min_vol_usd, top_n)
    n_wm = sum(1 for r in results if r["has_wm"])

    print(f"\n  Top {len(results)} assets ({n_wm} with WM models):")
    print(f"\n  {'#':>3} {'Asset':<12} {'Price':>12} {'Vol24h':>12} "
          f"{'Volat%':>7} {'Sprd':>6} {'WM':>3} {'Score':>6}")
    print("  " + "-" * 65)

    for i, r in enumerate(results):
        wm_tag = "yes" if r["has_wm"] else "-"
        print(f"  {i+1:>3} {r['asset']:<12} ${r['price']:>11,.2f} "
              f"${r['volume_24h']:>11,} {r['volatility_pct']:>6.1f}% "
              f"{r['spread_bps']:>5.1f} {wm_tag:>3} {r['score']:>6}")

    # Save to state dir
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATE_DIR / "universe.json"
    ts = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "top_n": top_n,
            "min_volume_usd": min_vol_usd,
            "n_wm_assets": n_wm,
            "assets": results,
        }, f, indent=2)
    print(f"\n  Saved to {out_path}")

    # Also save asset list for fetch_all consumption
    asset_list_path = STATE_DIR / "universe_assets.txt"
    with open(asset_list_path, "w") as f:
        for r in results:
            f.write(r["asset"] + "\n")
    print(f"  Asset list saved to {asset_list_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Screen top crypto assets for trading universe")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top assets (default: 20)")
    parser.add_argument("--min-volume", type=float, default=5.0,
                        help="Minimum 24h volume in millions USD (default: 5)")
    args = parser.parse_args()

    run_screener(top_n=args.top, min_volume=args.min_volume)


if __name__ == "__main__":
    main()
