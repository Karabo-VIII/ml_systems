# Phase-1 WM retrain kick-off (2026-05-21 SOTA sweep)
# ==================================================
# Runs V1.1 -> V13 -> V22 in series. Each retrain validates a different
# class of the SOTA fixes:
#   V1.1 (~2.5 GPU-d): Jensen-correct decode + AMP fp32 wraps (cross-cutting)
#   V13  (~2.0 GPU-d): InterpretableAttention RMSNorm (single-change canary)
#   V22  (~3.0 GPU-d): Timer-XL last-bar supervision + stride=1 (HIGH-INFO test)
#
# Total budget: ~7.5 GPU-d. Sequential on RTX 4060.
#
# Each retrain starts FRESH (no resume). Existing ckpts moved to backups/
# so we can A/B compare new vs old if needed.

$ErrorActionPreference = "Stop"
$ROOT = "C:\Users\karab\Documents\coding\v4_crypto_stystem"
Set-Location $ROOT

$ts = Get-Date -Format "yyyyMMdd_HHmm"
$BACKUP = "$ROOT\backups\wm_phase1_pre_retrain_$ts"
New-Item -ItemType Directory -Force -Path $BACKUP | Out-Null

Write-Host "[kickoff] backing up existing ckpts to $BACKUP" -ForegroundColor Cyan

function Backup-Ckpts {
    param([string]$VersionDir, [string]$Label)
    $src = "$ROOT\models\$VersionDir\base"
    if (Test-Path $src) {
        $dst = "$BACKUP\$Label"
        Move-Item -Path $src -Destination $dst -Force
        New-Item -ItemType Directory -Force -Path $src | Out-Null
        Write-Host "  $Label : moved $src -> $dst" -ForegroundColor DarkCyan
    } else {
        Write-Host "  $Label : no existing ckpts at $src (clean start)" -ForegroundColor DarkGray
    }
}

Backup-Ckpts -VersionDir "v1\v1_1" -Label "v1_1"
Backup-Ckpts -VersionDir "v13"     -Label "v13"
Backup-Ckpts -VersionDir "v22"     -Label "v22"

Write-Host "`n[kickoff] Phase-1 cohort: V1.1 -> V13 -> V22 (sequential)" -ForegroundColor Green
Write-Host "[kickoff] pre-train gate runs once at the start (--gate-asset BTC, layer=legacy)" -ForegroundColor Green
Write-Host ""

# Run gate once and let it fail-fast if data is broken
python src\pipeline\pre_train_gate.py --asset BTC --layer legacy
if ($LASTEXITCODE -eq 2) {
    Write-Host "[kickoff] PRE-TRAIN GATE HARD FAIL (rc=2). Aborting." -ForegroundColor Red
    exit 2
}

# V1.1 first (lowest-risk, validates Jensen + AMP cross-cutting). f41 per
# feature signal mining (FEATURE_SIGNAL_MINING_2026_05_21.md): f29 captures
# only 60% of top-signal features; f41 captures 80% at negligible compute cost.
Write-Host "`n[kickoff] === V1.1 (f41) ===" -ForegroundColor Yellow
python src\run_all_training.py --features 41 --model v1_1 --only-base --skip-gate --force
$v11_rc = $LASTEXITCODE

# V13 (single-change RMSNorm canary). Stays at f29 — V13 settings.py doesn't
# expose f41 in its supported counts; adding requires a settings extension.
Write-Host "`n[kickoff] === V13 (f29) ===" -ForegroundColor Yellow
python src\run_all_training.py --features 29 --model v13 --only-base --skip-gate --force
$v13_rc = $LASTEXITCODE

# V22 (Timer-XL last-bar - HIGH-INFO experiment). f41 captures +2 sign-stable
# cross-asset features (xd_momentum_rank, xd_cross_return_mean) vs f29.
Write-Host "`n[kickoff] === V22 (f41) ===" -ForegroundColor Yellow
python src\run_all_training.py --features 41 --model v22 --only-base --skip-gate --force
$v22_rc = $LASTEXITCODE

Write-Host "`n[kickoff] === Phase-1 SUMMARY ===" -ForegroundColor Green
Write-Host "  V1.1 rc=$v11_rc"
Write-Host "  V13  rc=$v13_rc"
Write-Host "  V22  rc=$v22_rc"
Write-Host ""
Write-Host "[kickoff] check logs/v1/v1_1/, logs/v13/, logs/v22/ for IC + ShIC."
Write-Host "[kickoff] backups at: $BACKUP"
