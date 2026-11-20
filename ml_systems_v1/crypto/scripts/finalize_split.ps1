<#
finalize_split.ps1 -- the ONE handoff step for the 3-way repo split that cannot run inside a
live Claude session (parent-dir rename side-effects, venv repath, global-harness reinstall,
Claude memory-dir continuity, and settings.json -- which is deny-listed to Claude's own tools).

RUN ORDER (see crypto/docs/SPLIT_RUNBOOK.md):
  1. CLOSE the Claude Code window/session for this project (so nothing holds the dir).
  2. Rename the parent dir (ONE command, from coding/ -- NOT this script, which lives inside it):
         Move-Item C:\Users\karab\Documents\coding\v4_crypto_stystem C:\Users\karab\Documents\coding\ml_systems
  3. Run THIS script (now at the new path):
         powershell -ExecutionPolicy Bypass -File C:\Users\karab\Documents\coding\ml_systems\crypto\scripts\finalize_split.ps1
  4. Reopen Claude Code at  C:\Users\karab\Documents\coding\ml_systems

It is idempotent and prints every action. No emoji (Windows cp1252 console safety).
#>
param(
  [string]$OldName = "v4_crypto_stystem",
  [string]$NewName = "ml_systems",
  [string]$SystemPython = "C:\Users\karab\AppData\Local\Programs\Python\Python311\python.exe"
)
$ErrorActionPreference = "Stop"

# ROOT = the renamed parent. This script is at <ROOT>\crypto\scripts\finalize_split.ps1
$ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$base = Split-Path $ROOT -Leaf
Write-Host "=== finalize_split: ROOT = $ROOT (basename '$base') ==="
if ($base -eq $OldName) {
  Write-Host "  ABORT: parent dir is still '$OldName'. Do step 2 (the Move-Item rename) FIRST, then re-run." -ForegroundColor Red
  exit 1
}
$venvPy = Join-Path $ROOT ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Write-Host "  WARN: venv python not found at $venvPy" -ForegroundColor Yellow }

# --- 1. venv .pth repath (absolute -> new ROOT) -------------------------------------------------
Write-Host "`n--- 1. venv .pth repath ---"
$sp = Join-Path $ROOT ".venv\Lib\site-packages"
$srcPth = Join-Path $sp "v4_src_root.pth"
$harnPth = Join-Path $sp "harness_root.pth"
Set-Content -Path $srcPth  -Value (Join-Path $ROOT "crypto\src") -Encoding ascii
Write-Host "  wrote $srcPth -> $(Join-Path $ROOT 'crypto\src')"
Set-Content -Path $harnPth -Value $ROOT -Encoding ascii
Write-Host "  wrote $harnPth -> $ROOT"

# --- 2. Claude memory-dir continuity (copy old encoded dir -> new) ------------------------------
Write-Host "`n--- 2. Claude project memory-dir continuity ---"
$proj = "C:\Users\karab\.claude\projects"
$encOld = ("C-Users-karab-Documents-coding-$OldName" -replace '_','-')
$encNew = ("C-Users-karab-Documents-coding-$NewName" -replace '_','-')
$encOld = "c" + $encOld.Substring(1)   # lowercase drive letter -> c--...
$encNew = "c" + $encNew.Substring(1)
$dOld = Join-Path $proj $encOld
$dNew = Join-Path $proj $encNew
if (Test-Path $dNew) {
  Write-Host "  new memory dir already exists: $encNew (skip copy)"
} elseif (Test-Path $dOld) {
  Copy-Item -Path $dOld -Destination $dNew -Recurse
  Write-Host "  copied memory dir: $encOld -> $encNew (old kept as fallback)"
} else {
  Write-Host "  WARN: old memory dir not found ($dOld) -- nothing to carry over" -ForegroundColor Yellow
}

# --- 3. settings.json / settings.local.json repath (JSON-aware python helper) -------------------
Write-Host "`n--- 3. settings.json + settings.local.json repath ---"
& $venvPy (Join-Path $ROOT "crypto\scripts\_finalize_repath.py") --root $ROOT --old $OldName --new $NewName
if ($LASTEXITCODE -ne 0) { Write-Host "  WARN: repath helper exit $LASTEXITCODE" -ForegroundColor Yellow }

# --- 4. global harness reinstall (regenerates the editable finder at the new path) --------------
Write-Host "`n--- 4. global harness reinstall (system user-site editable) ---"
if (Test-Path $SystemPython) {
  & $SystemPython -m pip install --user -e (Join-Path $ROOT "harness") --quiet
  Write-Host "  reinstalled: $SystemPython -m pip install --user -e $ROOT\harness (exit $LASTEXITCODE)"
} else {
  Write-Host "  WARN: system python not at $SystemPython -- run manually: <system-python> -m pip install --user -e $ROOT\harness" -ForegroundColor Yellow
}

# --- 5. verify (RWYB) ---------------------------------------------------------------------------
Write-Host "`n--- 5. verify imports + selftests from the new path ---"
& $venvPy -c "import strat, pipeline, wm, audit, framework; print('  crypto bare imports OK')"
& $venvPy -c "import harness.metaop; print('  harness.metaop OK (via .pth, cwd-independent)')"
Push-Location $ROOT
try {
  & $venvPy -m framework.selftest *>$null; Write-Host "  framework.selftest exit $LASTEXITCODE (0=pass)"
  & $venvPy (Join-Path $ROOT "crypto\src\strat\selftest_all.py") *>$null; Write-Host "  strat selftest exit $LASTEXITCODE (0=pass)"
} finally { Pop-Location }

Write-Host "`n=== DONE. Reopen Claude Code at: $ROOT ===" -ForegroundColor Green
Write-Host "If 'import strat' failed above, the venv may need recreation:"
Write-Host "    $SystemPython -m venv $ROOT\.venv  ;  $ROOT\.venv\Scripts\python -m pip install -r $ROOT\crypto\requirements.txt"
Write-Host "    (then re-add the two .pth files this script wrote in step 1)"
