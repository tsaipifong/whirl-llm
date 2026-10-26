# SPDX-License-Identifier: Apache-2.0
# arch025-S0a: host Arch::gfx branches outside the device layer. Blocking since arch025-M6: any site outside
# src/hip/, include/whirl/hip.h, include/whirl/device_profile.h and src/model/device.cpp exits 1 in either mode, and
# the pending list (arch_pending.txt) is gone (an entry in a -Pending file is itself a FAIL).
# Thin entry point of tools/gates/pending_scan.py rule "arch" (the rule and the selftest live there;
# pending_scan.py arch --tree-test is the block / pass test on a copy of the real tree).
#   pwsh -NoProfile -File tools\gates\arch_check.ps1 [-Mode report|block] [-Root <repo>] [-Pending <file>]
#   pwsh -NoProfile -File tools\gates\arch_check.ps1 -SelfTest
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
if (-not $py) { Write-Host "arch check: FAIL (usage: no python found; set WHIRL_GATE_PYTHON)"; exit 2 }
$scan = Join-Path $PSScriptRoot "pending_scan.py"
if ($SelfTest) { & $py -I $scan --selftest; exit $LASTEXITCODE }
$a = @("arch", "--mode", $Mode)
if ($Root) { $a += @("--root", $Root) }
if ($Pending) { $a += @("--pending", $Pending) }
& $py -I $scan @a
exit $LASTEXITCODE
