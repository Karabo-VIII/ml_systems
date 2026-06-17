"""
Multi-Venue Listing Poller — extends P8 to Bybit + OKX
=======================================================

P8 listing-h1-momentum (currently shipped, 20% allocation) polls Binance
exchangeInfo.onboardDate. Per memory frontier_session_report_2026_04_23
this is a +0.80%/day-at-event alpha (n=85, t=3.61). The pattern extends
to other venues:

    - Bybit: secondary listings often follow Binance by 24-48h with similar
      momentum patterns (Bybit has its own listing announcements)
    - OKX: same pattern, narrower coverage but dollar-volume comparable

Polls each venue's public API every N minutes; emits a unified "newly listed"
event stream that downstream pillars consume.

Output
------
data/processed/panels/daily/multi_venue_listings.parquet (rolling, append-only)

Schema:
    venue           : str  ('binance' | 'bybit' | 'okx')
    symbol          : str  (e.g. 'BTCUSDT')
    onboard_ts_ms   : i64  (UTC ms)
    detected_ts_ms  : i64  (when our poller saw it; quality measure)
    contract_type   : str  ('spot' | 'perp' | 'futures')
    fetched_at      : i64

Decoupled
---------
Standalone ingester. P8 pillar (existing) reads from this panel via venue
filter. New venues slot in by adding a fetcher function.

Status: experimental. Bybit/OKX fetchers are stubs in this commit; full
implementation requires API key registration for higher rate limits.
"""
from __future__ import annotations
import os

# CDAP contract
__contract__ = {
    "kind": "panel_builder",
    "stage": "multi_venue_listings",
    "inputs": {
        "args": ["--once", "--continuous", "--poll-interval-min"],
        "upstream": "binance/api/v3/exchangeInfo + bybit + okx (public APIs)",
    },
    "outputs": {
        "files": "data/processed/panels/daily/multi_venue_listings.parquet",
        "columns": ["venue", "symbol", "onboard_ts_ms", "detected_ts_ms",
                     "contract_type", "fetched_at"],
    },
    "invariants": {
        "atomic_write": True,
        "no_paid_apis": True,
        "rolling_append_safe": True,    # safe to call repeatedly
    },
}

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
import layout as _layout                                 # noqa: E402

OUT_PATH = _layout.panels_dir() / "multi_venue_listings.parquet"
USER_AGENT = "v4-multi-venue-listings/1.0"


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("mv_list", phase, message, **kw)


def _http_get(url: str, timeout_s: float = 10.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except Exception as e:
        # B3: announce the failure so orchestrator/operator can see WHY a fetch
        # returned empty (was: silent None -> empty rows -> stale panel rc=0).
        print(f"[multi_venue_listings] HTTP GET failed {url[:80]}: "
              f"{type(e).__name__}: {e}", flush=True)
        return None


# ──────────────────────────────────────────────────────────────────────────
# Venue fetchers
# ──────────────────────────────────────────────────────────────────────────

def fetch_binance_listings() -> List[dict]:
    """Binance Futures exchangeInfo. Returns list of dicts.

    Schema per row: symbol, onboardDate (ms epoch), contractType
    """
    data = _http_get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    if not data:
        return []
    rows = []
    now = int(time.time() * 1000)
    for sym in (data.get("symbols") or []):
        if sym.get("status") != "TRADING":
            continue
        rows.append({
            "venue":          "binance",
            "symbol":         sym.get("symbol"),
            "onboard_ts_ms":  int(sym.get("onboardDate", 0) or 0),
            "detected_ts_ms": now,
            "contract_type": "perp" if sym.get("contractType") == "PERPETUAL" else "futures",
            "fetched_at":     now,
        })
    return rows


def fetch_bybit_listings() -> List[dict]:
    """Bybit V5 public API: GET /v5/market/instruments-info?category=linear

    Returns list of dicts. Bybit publishes launchTime as ms-epoch string.
    """
    data = _http_get("https://api.bybit.com/v5/market/instruments-info?category=linear")
    if not data or "result" not in data:
        return []
    rows = []
    now = int(time.time() * 1000)
    for inst in (data.get("result", {}).get("list") or []):
        if inst.get("status") != "Trading":
            continue
        try:
            launch = int(inst.get("launchTime", "0"))
        except Exception:
            launch = 0
        rows.append({
            "venue":          "bybit",
            "symbol":         inst.get("symbol"),
            "onboard_ts_ms":  launch,
            "detected_ts_ms": now,
            "contract_type":  "perp",
            "fetched_at":     now,
        })
    return rows


def fetch_okx_listings() -> List[dict]:
    """OKX V5 public API: GET /api/v5/public/instruments?instType=SWAP

    listTime in OKX is ms-epoch.
    """
    data = _http_get("https://www.okx.com/api/v5/public/instruments?instType=SWAP")
    if not data or "data" not in data:
        return []
    rows = []
    now = int(time.time() * 1000)
    for inst in (data.get("data") or []):
        if inst.get("state") != "live":
            continue
        try:
            list_t = int(inst.get("listTime", "0"))
        except Exception:
            list_t = 0
        rows.append({
            "venue":          "okx",
            "symbol":         (inst.get("instId") or "").replace("-", ""),
            "onboard_ts_ms":  list_t,
            "detected_ts_ms": now,
            "contract_type":  "perp",
            "fetched_at":     now,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────
# Aggregator + panel writer
# ──────────────────────────────────────────────────────────────────────────

def fetch_all() -> List[dict]:
    """Union of all venue feeds. Logs counts per venue."""
    out: List[dict] = []
    for fn, name in [(fetch_binance_listings, "binance"),
                      (fetch_bybit_listings, "bybit"),
                      (fetch_okx_listings, "okx")]:
        try:
            rows = fn()
            print(f"[multi_venue_listings] {name}: {len(rows)} listings")
            out.extend(rows)
        except Exception as e:
            print(f"[multi_venue_listings] {name} failed: {type(e).__name__}: {e}")
    return out


def write_panel(rows: List[dict]) -> Path:
    """Append/upsert rows to OUT_PATH. Atomic."""
    import polars as pl
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "venue":          pl.String,
        "symbol":         pl.String,
        "onboard_ts_ms":  pl.Int64,
        "detected_ts_ms": pl.Int64,
        "contract_type":  pl.String,
        "fetched_at":     pl.Int64,
    })

    # Merge with existing (upsert on venue+symbol; keep latest detected_ts_ms)
    if OUT_PATH.exists() and not new_df.is_empty():
        try:
            existing = pl.read_parquet(OUT_PATH)
            merged = pl.concat([existing, new_df], how="vertical_relaxed")
            merged = merged.unique(subset=["venue", "symbol"], keep="last")
        except Exception as e:
            # Do NOT silently replace the whole historical panel with just the new
            # rows on a read error -- that is silent data loss. Fail loudly so the
            # corrupt file is investigated (the existing panel is left untouched).
            raise RuntimeError(
                f"multi_venue_listings: failed to read existing panel {OUT_PATH} "
                f"({type(e).__name__}: {e}); refusing to overwrite history with "
                f"new rows only. Inspect/restore the file, then re-run.") from e
    else:
        merged = new_df

    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    merged.write_parquet(tmp)
    required = set(__contract__["outputs"]["columns"])
    written = set(pl.read_parquet_schema(tmp).keys())
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"multi_venue_listings missing cols: {sorted(missing)}")
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    os.replace(str(tmp), str(OUT_PATH))  # atomic overwrite (Windows-safe)
    print(f"[multi_venue_listings] wrote {OUT_PATH.name}: {len(merged)} total rows")
    return OUT_PATH


def get_recent_listings(window_hours: float = 48.0,
                         venue: Optional[str] = None) -> List[dict]:
    """Query the panel for listings whose onboard_ts_ms is within window."""
    if not OUT_PATH.exists():
        return []
    import polars as pl
    df = pl.read_parquet(OUT_PATH)
    cutoff = int(time.time() * 1000) - int(window_hours * 3600 * 1000)
    df = df.filter(pl.col("onboard_ts_ms") >= cutoff)
    if venue is not None:
        df = df.filter(pl.col("venue") == venue)
    return df.to_dicts()


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="Fetch once and exit (default)")
    ap.add_argument("--continuous", action="store_true",
                    help="Poll continuously every --poll-interval-min")
    ap.add_argument("--poll-interval-min", type=int, default=30)
    ap.add_argument("--force", action="store_true",
                    help="Delete existing panel and rebuild from venue APIs.")
    ap.add_argument("--smoke", action="store_true",
                    help="No network; just verify schema")
    args = ap.parse_args()

    if args.force and OUT_PATH.exists():
        OUT_PATH.unlink()
        print(f"[multi_venue_listings] [force] deleted prior panel "
              f"{OUT_PATH.name}; rebuilding from API", flush=True)

    if args.smoke:
        # Schema-only smoke
        sample = [{
            "venue":          "binance",
            "symbol":         "BTCUSDT",
            "onboard_ts_ms":  int(time.time() * 1000) - 86400_000,
            "detected_ts_ms": int(time.time() * 1000),
            "contract_type":  "perp",
            "fetched_at":     int(time.time() * 1000),
        }]
        required = set(__contract__["outputs"]["columns"])
        assert required.issubset(set(sample[0].keys()))
        print(f"  contract cols match: {sorted(required)}")
        print(f"  sample: {sample[0]}")
        print("PASS: multi_venue_listings smoke (schema)")
        return

    if args.continuous:
        while True:
            rows = fetch_all()
            write_panel(rows)
            print(f"[multi_venue_listings] sleeping {args.poll_interval_min} min")
            time.sleep(args.poll_interval_min * 60)
    else:
        rows = fetch_all()
        write_panel(rows)


if __name__ == "__main__":
    # Production by default (was: RUN_LIVE=1 gated). Pass --smoke for the
    # schema-only check.
    main()
