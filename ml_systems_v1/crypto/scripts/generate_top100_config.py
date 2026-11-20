"""
Generate data_config.yaml entries for top-100 USDT spot pairs by volume.

Queries Binance public ticker API (no auth needed), filters to USDT spot
pairs, ranks by quoteVolume (24h USD volume), and emits YAML entries with
dollar_bar_size calibrated to ~5min bars (288/day target).

Preserves existing tuned sizes for assets already in config; appends new
ones sorted by volume rank.

Exclusions:
  - Stablecoins (USDC, DAI, TUSD, FDUSD, BUSD, USDP, ...)
  - Pegged / wrapped tokens (WBTC-type already covered by spot BTC)
  - Leveraged tokens (UP/DOWN/BULL/BEAR suffix)

Usage:
    python scripts/generate_top100_config.py --output config/data_config_top100.yaml
    python scripts/generate_top100_config.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import urllib.request
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Known stables/pegged/wrapped — exclude from trading universe.
STABLES = {
    "USDC", "DAI", "TUSD", "FDUSD", "BUSD", "USDP", "USTC", "UST",
    "PYUSD", "EURI", "EUR", "AEUR", "USD1", "USDE", "PAXG",  # paxg = gold
    "FUSD", "USD", "GUSD", "RLUSD", "USDT",  # USDT base, never "X/USDT"
}

# Exclude suffix patterns (leveraged tokens)
BAD_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")


def fetch_binance_24h_tickers() -> List[Dict]:
    """Return list of Binance 24h ticker dicts (spot)."""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def is_valid_usdt_pair(symbol: str) -> bool:
    """BTCUSDT -> True (valid USDT quote pair)."""
    if not symbol.endswith("USDT"):
        return False
    base = symbol[:-4]
    if not base:
        return False
    if base in STABLES:
        return False
    for suf in BAD_SUFFIX:
        if base.endswith(suf) and len(base) > len(suf):
            return False
    return True


def load_existing_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f) or {}
    return data.get("assets", {})


def calibrate_bar_size(quote_volume_24h: float,
                        target_bars_per_day: int = 288) -> int:
    """Dollar-bar size that produces ~target_bars_per_day.

    quote_volume_24h is in USD (USDT). Bar fires when `size` dollars trade.
    """
    if quote_volume_24h <= 0:
        return 50_000
    size = quote_volume_24h / target_bars_per_day
    # Round to sensible buckets
    if size >= 1_000_000:
        return int(round(size / 100_000) * 100_000)
    if size >= 100_000:
        return int(round(size / 10_000) * 10_000)
    if size >= 10_000:
        return int(round(size / 1_000) * 1_000)
    return max(1_000, int(round(size / 500) * 500))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100,
                     help="Top-N USDT pairs by volume")
    ap.add_argument("--output", type=str,
                     default="config/data_config_top100.yaml",
                     help="Output YAML path")
    ap.add_argument("--existing", type=str,
                     default="config/data_config.yaml",
                     help="Existing config to preserve tuned bar sizes from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-existing", action="store_true", default=True,
                     help="Preserve ALL existing assets even if out of top-N (default: True)")
    ap.add_argument("--drop-existing-out-of-top", dest="keep_existing",
                     action="store_false",
                     help="Drop existing assets not in top-N")
    args = ap.parse_args()

    print(f"Fetching Binance 24h tickers...")
    tickers = fetch_binance_24h_tickers()
    print(f"  Got {len(tickers)} total tickers")

    # Filter + rank
    valid = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not is_valid_usdt_pair(sym):
            continue
        try:
            qv = float(t.get("quoteVolume", 0))
            lastp = float(t.get("lastPrice", 0))
            count = int(t.get("count", 0))
        except (TypeError, ValueError):
            continue
        if qv < 1_000_000:  # <$1M/day = illiquid
            continue
        if lastp <= 0 or count < 1000:
            continue
        valid.append({
            "symbol": sym, "base": sym[:-4],
            "quote_volume_24h": qv,
            "last_price": lastp, "trade_count": count,
        })

    valid.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
    top = valid[:args.n]
    print(f"  Filtered to {len(valid)} valid USDT spot pairs; top {args.n} selected")
    print(f"  Volume range: ${top[0]['quote_volume_24h']/1e9:.2f}B -> "
          f"${top[-1]['quote_volume_24h']/1e6:.1f}M")

    # Preserve existing bar sizes
    existing_assets = load_existing_config(PROJECT_ROOT / args.existing)
    print(f"  Loaded {len(existing_assets)} existing tuned configs")

    # Build output
    out_assets = {}
    preserved = 0
    new = 0
    top_keys = {f"{t['base']}/USDT" for t in top}
    # 1. Always preserve existing assets (avoid breaking pipeline/models)
    existing_kept = 0
    existing_dropped = []
    for key, cfg in existing_assets.items():
        if key in top_keys or args.keep_existing:
            out_assets[key] = cfg
            existing_kept += 1
            preserved += 1
        else:
            existing_dropped.append(key)
    # 2. Add top-N that aren't already in output
    for t in top:
        key = f"{t['base']}/USDT"
        if key not in out_assets:
            # Sanitize ticker (exclude any non-ASCII unicode symbols)
            base_ascii = t["base"].encode("ascii", "ignore").decode("ascii")
            if len(base_ascii) != len(t["base"]):
                continue  # skip non-ASCII ticker
            out_assets[key] = {
                "dollar_bar_size": calibrate_bar_size(t["quote_volume_24h"]),
                "is_active": True,
            }
            new += 1

    print(f"  Existing kept: {existing_kept}/{len(existing_assets)} "
          f"(dropped {len(existing_dropped)})")
    if existing_dropped:
        print(f"  Dropped from existing (low vol, not in top {args.n}): "
              f"{existing_dropped}")
    print(f"  New entries added: {new}")
    print(f"  TOTAL assets in output: {len(out_assets)}")

    # Print summary table
    print(f"\n{'Rank':>4} {'Asset':<15} {'Vol24h$':>12} {'BarSize$':>10} {'Status':<10}")
    print("-" * 60)
    for i, t in enumerate(top[:20], 1):
        key = f"{t['base']}/USDT"
        if key not in out_assets:   # non-ASCII skipped
            continue
        size = out_assets[key]["dollar_bar_size"]
        status = "preserved" if key in existing_assets else "new"
        vol_s = f"${t['quote_volume_24h']/1e6:.1f}M" if t['quote_volume_24h'] < 1e9 \
            else f"${t['quote_volume_24h']/1e9:.2f}B"
        key_ascii = key.encode("ascii", "replace").decode("ascii")
        print(f"{i:>4} {key_ascii:<15} {vol_s:>12} {size:>10,} {status:<10}")
    if len(top) > 20:
        print(f"... {len(top)-20} more below")

    if args.dry_run:
        print("\n[DRY-RUN] No file written.")
        return

    # Build full YAML structure — preserve non-assets sections from existing
    existing_full = {}
    existing_path = PROJECT_ROOT / args.existing
    if existing_path.exists():
        with existing_path.open(encoding="utf-8", errors="replace") as f:
            existing_full = yaml.safe_load(f) or {}
    existing_full["assets"] = out_assets

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(existing_full, f, sort_keys=False,
                        default_flow_style=False, allow_unicode=False)
    print(f"\nWrote: {out_path}")
    print(f"Review then rename to {args.existing} if acceptable.")


if __name__ == "__main__":
    main()
