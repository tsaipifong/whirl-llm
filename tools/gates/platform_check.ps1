# SPDX-License-Identifier: Apache-2.0
# arch025-S0a: platform headers outside the platform layer (PL0 / PL5 make it block).
# Thin entry point of tools/gates/pending_scan.py rule "platform" (the rule, the pending list platform_pending.txt and the
# selftest live there). Report mode (default) prints the sites and exits 0; -Mode block exits 1 on a new site, a grown
# count or a stale pending entry.
#   pwsh -NoProfile -File tools\gates\platform_check.ps1 [-Mode report|block] [-Root <repo>] [-Pending <file>]
#   pwsh -NoProfile -File tools\gates\platform_check.ps1 -SelfTest
# Python: WHIRL_GATE_PYTHON, else C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe when it exists, else python / py on PATH
# (as release_gate.ps1).
param([ValidateSet("report", "block")][string]$Mode = "report", [string]$Root = "", [string]$Pending = "", [switch]$SelfTest)
$ErrorActionPreference = "Stop"
$py = $env:WHIRL_GATE_PYTHON
if (-not $py) {
    $pyref = "C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
    if (Test-Path $pyref) { $py = $pyref }
    else { $py = @(@(Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1) | ForEach-Object { $_.Source })[0] }
}
if (-not $py) { Write-Host "platform check: FAIL (usage: no python found; set WHIRL_GATE_PYTHON)"; exit 2 }
$scan = Join-Path $PSScriptRoot "pending_scan.py"
if ($SelfTest) { & $py -I $scan --selftest; exit $LASTEXITCODE }
$a = @("platform", "--mode", $Mode)
if ($Root) { $a += @("--root", $Root) }
if ($Pending) { $a += @("--pending", $Pending) }
& $py -I $scan @a
exit $LASTEXITCODE
