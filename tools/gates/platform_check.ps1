# SPDX-License-Identifier: Apache-2.0
# arch025-S0a: platform headers outside the platform layer; PL0: also Win32 / Winsock / CRT-platform calls. Blocking since
# arch025-PL5 (whitelist only): any site in src/ or include/ outside src/platform/*_win.cpp, src/hip/, src/release/dll_deps.cpp
# and the vendored src/vision/stb_image.h exits 1 in either mode, and the pending list (platform_pending.txt) is gone (an entry
# in a -Pending file is itself a FAIL). tests/ and tools/ are not scanned; the wmain entry points call whirl::narrowArgs.
# Thin entry point of tools/gates/pending_scan.py rule "platform" (the rule and the selftest live there). Release gate step:
# cpu_platform_check (-Mode block). -SelfTest = the rule selftest + the block / pass test on a copy of the real tree.
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
if ($SelfTest) {
    # arch025-PL0: the rule selftest (synthetic trees) and the block / pass test on a copy of this tree
    & $py -I $scan --selftest
    $rc1 = $LASTEXITCODE
    $t = @("--tree-test", "platform")
    if ($Root) { $t += @("--root", $Root) }
    & $py -I $scan @t
    $rc2 = $LASTEXITCODE
    if ($rc1 -ne 0 -or $rc2 -ne 0) { Write-Host "platform check selftest: FAIL (rule $rc1, tree $rc2)"; exit 1 }
    Write-Host "platform check selftest: ok"
    exit 0
}
$a = @("platform", "--mode", $Mode)
if ($Root) { $a += @("--root", $Root) }
if ($Pending) { $a += @("--pending", $Pending) }
& $py -I $scan @a
exit $LASTEXITCODE
