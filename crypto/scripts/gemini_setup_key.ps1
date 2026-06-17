# gemini_setup_key.ps1 -- SAFE Gemini API key entry helper.
#
# DO NOT paste your API key into the Claude chat. Anything you type into
# Claude becomes part of the conversation history. This script prompts for
# the key locally so it never leaves your terminal.
#
# Usage (PowerShell, from the repo root):
#     .\scripts\gemini_setup_key.ps1
#
# What it does:
#   1. Prompts for the API key as a SECURE (masked) input.
#   2. Sets $env:GEMINI_API_KEY for the CURRENT session only.
#   3. Asks if you want it PERSISTED for your user profile (survives reboots).
#   4. Verifies the key works with a 1-token Gemini call.
#
# To get a Gemini API key:
#   https://aistudio.google.com/apikey
#   (Free tier: 60 req/min, 1500 req/day on Flash. Pro is paid.)
#
# Removal:
#   $env:GEMINI_API_KEY = $null                           # session
#   [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $null, "User")  # persistent

param(
    [switch]$Persist
)

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Gemini API Key Setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "DO NOT paste your key into Claude chat. Anything you" -ForegroundColor Yellow
Write-Host "type to Claude becomes part of the conversation. Use" -ForegroundColor Yellow
Write-Host "this script to set the key locally instead." -ForegroundColor Yellow
Write-Host ""

# Prompt for key as SecureString (masked input)
$secure = Read-Host "Paste your Gemini API key (input hidden)" -AsSecureString
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

if (-not $plain -or $plain.Length -lt 20) {
    Write-Host "ERROR: key looks too short. Aborting without setting." -ForegroundColor Red
    exit 1
}

# Set for current session
$env:GEMINI_API_KEY = $plain
Write-Host ""
Write-Host "[OK] GEMINI_API_KEY set for THIS PowerShell session." -ForegroundColor Green

# Optionally persist
if (-not $Persist) {
    $reply = Read-Host "Persist for your user profile (survives reboots)? [y/N]"
    $Persist = ($reply -eq "y" -or $reply -eq "Y")
}
if ($Persist) {
    [Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $plain, "User")
    Write-Host "[OK] GEMINI_API_KEY persisted to USER environment." -ForegroundColor Green
} else {
    Write-Host "[SKIP] Not persisting; key only lives in this session." -ForegroundColor DarkGray
}

# Verify with a 1-token Gemini call
Write-Host ""
Write-Host "Verifying with a small test call..." -ForegroundColor Cyan
$verify = python scripts/gemini_consult.py --max-tokens 8 "Reply with exactly: OK" 2>&1
Write-Host $verify
Write-Host ""
Write-Host "Setup complete. Claude can now invoke:" -ForegroundColor Cyan
Write-Host "  python scripts/gemini_consult.py 'prompt'" -ForegroundColor Cyan
