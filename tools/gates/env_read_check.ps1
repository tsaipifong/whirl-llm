# SPDX-License-Identifier: Apache-2.0
# arch025-S0a: process-environment reads. arch025-F7: forward / engine / ops are BLOCKED in every mode.
# Thin entry point of tools/gates/pending_scan.py rule "env" (the rule, the pending list env_pending.txt and the
# selftest live there). Report mode (default) prints the sites and exits 0, except that any read in a blocked scope
# (src/model/qwen35/forward/, forward_internal.h, src/server/engine*, qwen35/server_model.cpp, src/ops/, include/whirl/ops/) or a
# pending entry for one exits 1; -Mode block also exits 1 on a new site, a grown count or a stale pending entry.
# -TreeTest: block / pass on a copy of the real tree (probes in forward/dispatch.cpp and server/engine.cpp).
#   pwsh -NoProfile -File tools\gates\env_read_check.ps1 [-Mode report|block] [-Root <repo>] [-Pending <file>]
#   pwsh -NoProfile -File tools\gates\env_read_check.ps1 -SelfTest
# Python: WHIRL_GATE_PYTHON, else C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe when it exists, else python / py on PATH
# (as release_gate.ps1).
param([ValidateSet("report", "block")][string]$Mode = "report", [string]$Root = "", [string]$Pending = "", [switch]$SelfTest, [switch]$TreeTest)
$ErrorActionPreference = "Stop"
$py = $env:WHIRL_GATE_PYTHON
if (-not $py) {
    $pyref = "C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
    if (Test-Path $pyref) { $py = $pyref }
    else { $py = @(@(Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1) | ForEach-Object { $_.Source })[0] }
}
if (-not $py) { Write-Host "env check: FAIL (usage: no python found; set WHIRL_GATE_PYTHON)"; exit 2 }
$scan = Join-Path $PSScriptRoot "pending_scan.py"
if ($SelfTest) { & $py -I $scan --selftest; exit $LASTEXITCODE }
if ($TreeTest) { $t = @("env", "--tree-test"); if ($Root) { $t += @("--root", $Root) }; & $py -I $scan @t; exit $LASTEXITCODE }
$a = @("env", "--mode", $Mode)
if ($Root) { $a += @("--root", $Root) }
if ($Pending) { $a += @("--pending", $Pending) }
& $py -I $scan @a
exit $LASTEXITCODE
