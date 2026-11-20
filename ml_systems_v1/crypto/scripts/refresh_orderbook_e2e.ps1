# =============================================================================
# refresh_orderbook_e2e.ps1  (2026-05-31)
# Full end-to-end pipeline refresh WITH order-book flow, run under .venv.
#
# Context: the 2026-05-30 rebuild excluded raw_book_depth (order-book) and ran
# the rest. This script fetches the order-book gap and force-rebuilds everything
# downstream so the bd_*/lob_* (order-book) features land FRESH in the chimera
# across the whole u100 roster (77 assets).
#
# Design:
#   - Run with the PROJECT .venv (NOT global Python311 -- it lacks pyarrow/lxml,
#     which silently fails every pandas-based producer).
#   - Phase 1 tops up order-book + ETF INCREMENTALLY (no --force = skip the 1200+
#     existing day-files per asset; fetch only the missing ~5 recent days).
#   - Phase 2 force-rebuilds the whole DAG so order-book propagates into the chimera,
#     while SKIPPING the ~6h raw aggTrades/funding re-fetch (on-disk raw is recent
#     through ~05-28; pass -IncludeRawFetch to also pull the newest Binance bars).
#   - Phase 3 verifies the gate.
#
# Usage:
#   .\scripts\refresh_orderbook_e2e.ps1                 # order-book + downstream (recommended)
#   .\scripts\refresh_orderbook_e2e.ps1 -IncludeRawFetch  # also re-fetch raw trades/funding (+~6h)
#   .\scripts\refresh_orderbook_e2e.ps1 -Universe u10   # smaller scope for a quick test
#   (run `.\.venv\Scripts\python.exe src\pipeline\refresh.py --live` in a 2nd terminal to watch)
# =============================================================================
param(
    [string]$Universe = "u100",
    [switch]$IncludeRawFetch,
    [switch]$NoParallel
)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw ".venv not found at $py -- create/activate the project venv first." }

# Guard: confirm the interpreter actually has the data deps (the 05-30 run failed
# precisely because it used a Python without pyarrow/lxml).
& $py -c "import pyarrow, lxml, polars, pandas, numba; print('[deps] OK', pyarrow.__version__)"
if ($LASTEXITCODE -ne 0) { throw "venv is missing data deps (pyarrow/lxml/...) -- pip install them first." }

Write-Host "`n=== Phase 1: incremental order-book + ETF ingest (top-up only, no --force) ===" -ForegroundColor Cyan
# Order-book (Binance Vision bookDepth) -> bd_* features. --universe now supported
# (cli_universe_support fix). No --force => only the missing recent dates are fetched.
& $py src\pipeline\refresh.py --target raw_book_depth --universe $Universe
# ETF flows: --recheck-missing bypasses the confirmed-missing manifest (Farside has
# transient outages); refreshes btc/eth ETF panels consistently.
& $py src\pipeline\ingest\etf_flows.py --recheck-missing

Write-Host "`n=== Phase 2: full E2E rebuild (force) -- order-book propagates into the chimera ===" -ForegroundColor Cyan
# --all --force rebuilds every downstream stage consistently: book_depth_profile_daily,
# lob_proxy_bars/daily, all daily panels, frontier_silver, chimera_legacy, chimera_v51,
# add_xrel_features. The chimera rebuild (memory_exclusive, 77 assets) is the long pole
# (~hours). Excludes raw_book_depth (done in Phase 1; downstream reads its fresh output).
$excludes = @("raw_book_depth")
if (-not $IncludeRawFetch) {
    # Skip the ~6h Binance raw trade/funding fetch -- on-disk raw is recent (~05-28).
    $excludes += @("raw_aggtrades", "raw_funding")
    Write-Host "  (skipping raw aggTrades/funding fetch; pass -IncludeRawFetch to include)" -ForegroundColor DarkGray
}
# --parallel = wave-schedule independent stages (RAM-safe: memory_exclusive chimera
# stages auto-singleton, so the OOM-prone chimera never runs alongside other heavy
# stages). 20 cores available; cap concurrent stages at 4. Pass -NoParallel to serialize.
$parallelArgs = if ($NoParallel) { @() } else { @("--parallel", "--max-concurrent-stages", "4") }
& $py src\pipeline\refresh.py --all --force --universe $Universe --exclude $excludes $parallelArgs

Write-Host "`n=== Phase 3: verify ===" -ForegroundColor Cyan
& $py src\pipeline\pre_train_gate.py --asset BTC
& $py src\pipeline\refresh.py --status
& $py src\pipeline\refresh.py --failures

Write-Host "`n=== DONE. Spot-check order-book freshness in the chimera: ===" -ForegroundColor Green
& $py -c @"
import polars as pl, glob
f = glob.glob('data/processed/chimera/dollar/btcusdt_v51_chimera*.parquet')[0]
lf = pl.scan_parquet(f)
for c in ['lob_l1_imb_mean','bd_depth_l1pct_mean','bd_imbalance_l1' if 'bd_imbalance_l1' in lf.collect_schema().names() else 'lob_kyle_lambda_mean']:
    tail_null = lf.select(pl.col(c)).tail(50000).select(pl.col(c).is_null().mean()).collect().item()
    print(f'  {c:24s} recent-50k null% = {tail_null*100:.1f}')
"@
