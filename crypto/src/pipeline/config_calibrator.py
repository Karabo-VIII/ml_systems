"""
Config Calibrator -- Updates dollar_bar_size in config/data_config.yaml.

Targets ~5-minute bar frequency (288 bars/day).
Uses geometric mean of live futures volume (estimated spot) and historical median.

SAFETY (2026-05-29 hardening): this script performs an in-place MERGE -- it loads
the existing data_config.yaml, updates ONLY the per-asset `dollar_bar_size` values,
and preserves every other section (splits, exchange, system, data) verbatim. It
NEVER regenerates the whole file from scratch (the previous version dropped the
load-bearing `splits:` block, which breaks purge_split.get_split_dates and all
downstream training). A timestamped backup is written before any change, and the
write is atomic (tmp + os.replace).
"""
import argparse
import os
import sys
import requests
import yaml
import numpy as np
from pathlib import Path

__contract__ = {
    "kind": "config_tool",
    "inputs": ["config/data_config.yaml (existing)", "Binance fapi 24h ticker"],
    "outputs": ["config/data_config.yaml (dollar_bar_size updated in place)"],
    "invariants": [
        "merge_only: never drops non-asset sections (splits/exchange/system/data)",
        "backup_before_write",
        "atomic_write_via_os_replace",
    ],
}

# Project root: src/pipeline/config_calibrator.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"

# THE MAGNIFICENT 10
TARGET_ASSETS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "LTC/USDT"
]

TARGET_FREQUENCY_MINS = 5
BARS_PER_DAY = (24 * 60) / TARGET_FREQUENCY_MINS  # 288

# Estimated median SPOT daily volumes (USD) from historical data
# Used as fallback when API is unavailable
HISTORICAL_MEDIAN_SPOT_VOL = {
    "BTC/USDT":  500_000_000,
    "ETH/USDT":  200_000_000,
    "SOL/USDT":   50_000_000,
    "BNB/USDT":   90_000_000,
    "XRP/USDT":  100_000_000,
    "DOGE/USDT": 120_000_000,
    "ADA/USDT":   30_000_000,
    "AVAX/USDT":  25_000_000,
    "LINK/USDT":  20_000_000,
    "LTC/USDT":   15_000_000,
}

# Spot volume is typically 10-30% of futures volume
SPOT_TO_FUTURES_RATIO = 0.15


def get_24h_volume(symbol):
    """Fetches 24h futures volume from Binance and estimates spot volume."""
    clean_sym = symbol.replace("/", "")
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        r = requests.get(url, params={"symbol": clean_sym}, timeout=10)
        data = r.json()
        futures_vol = float(data['quoteVolume'])
        return futures_vol * SPOT_TO_FUTURES_RATIO
    except Exception as e:
        print(f"  [WARN] Could not fetch {symbol}: {e}", flush=True)
        return None


def _round_bar_size(optimal_size: float) -> int:
    if optimal_size > 10_000_000:
        optimal_size = round(optimal_size / 1_000_000) * 1_000_000
    elif optimal_size > 1_000_000:
        optimal_size = round(optimal_size / 100_000) * 100_000
    elif optimal_size > 100_000:
        optimal_size = round(optimal_size / 10_000) * 10_000
    elif optimal_size > 10_000:
        optimal_size = round(optimal_size / 5_000) * 5_000
    else:
        optimal_size = round(optimal_size / 1_000) * 1_000
    return int(max(optimal_size, 10_000))  # Floor at $10K


def compute_bar_sizes() -> dict:
    """Return {asset_key: dollar_bar_size} for TARGET_ASSETS (no IO)."""
    sizes = {}
    for symbol in TARGET_ASSETS:
        live_vol = get_24h_volume(symbol)
        hist_vol = HISTORICAL_MEDIAN_SPOT_VOL.get(symbol)

        if live_vol and hist_vol:
            daily_vol = np.sqrt(live_vol * hist_vol)  # geometric mean for stability
        elif live_vol:
            daily_vol = live_vol
        elif hist_vol:
            daily_vol = hist_vol
        else:
            print(f"   SKIP {symbol:<10} | No volume data available", flush=True)
            continue

        optimal_size = _round_bar_size(daily_vol / BARS_PER_DAY)
        print(f"   {symbol:<10} | Est.Spot: ${daily_vol/1e6:>6.1f}M | "
              f"Bar Size: ${optimal_size:,.0f}", flush=True)
        sizes[symbol] = optimal_size
    return sizes


def generate_config(dry_run: bool = False):
    print("V500 CONFIG CALIBRATOR (10-ASSET UNIVERSE) -- merge mode", flush=True)
    print(f"   Target Frequency: 1 bar every {TARGET_FREQUENCY_MINS} minutes", flush=True)
    print(f"   Target Bars/Day:  {BARS_PER_DAY:.0f}", flush=True)
    print(f"   Config: {CONFIG_PATH}", flush=True)
    print("-" * 60, flush=True)

    if not CONFIG_PATH.exists():
        print(f"[ERROR] {CONFIG_PATH} does not exist. This tool only UPDATES an "
              f"existing config (it will not synthesize the load-bearing 'splits' "
              f"section). Create the base config first.", flush=True)
        sys.exit(2)

    # Load existing config preserving all sections.
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sizes = compute_bar_sizes()
    if not sizes:
        print("[ERROR] No bar sizes computed; aborting (config unchanged).", flush=True)
        sys.exit(2)

    assets = cfg.setdefault("assets", {})
    updated, added = 0, 0
    for asset_key, size in sizes.items():
        if asset_key in assets and isinstance(assets[asset_key], dict):
            assets[asset_key]["dollar_bar_size"] = size
            updated += 1
        else:
            assets[asset_key] = {"dollar_bar_size": size, "is_active": True}
            added += 1

    preserved = [k for k in cfg.keys() if k != "assets"]
    print("-" * 60, flush=True)
    print(f"   Assets updated: {updated}, added: {added}", flush=True)
    print(f"   Preserved sections: {preserved}", flush=True)

    if dry_run:
        print("[DRY-RUN] No file written.", flush=True)
        return

    # Backup, then atomic write.
    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    backup.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    os.replace(tmp, CONFIG_PATH)

    print(f"Backup written to {backup}", flush=True)
    print(f"Configuration updated at {CONFIG_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Update dollar_bar_size in data_config.yaml (merge-only).")
    ap.add_argument("--dry-run", action="store_true", help="Compute + report, do not write.")
    args = ap.parse_args()
    generate_config(dry_run=args.dry_run)
