# SPDX-License-Identifier: Apache-2.0
# Server log scan (arch024 S10, PLAN M8 / C-b1): reads every server log (server_*.log) under a folder and fails when a
# line of the chosen pattern sets is there, or when no server log is found (a check that read nothing must not pass).
# Shared by tools/gates/release_gate.ps1 (srv_* steps), tools/golden/golden.ps1 (suite server) and the GPU runners.
#
#   pwsh tools/gates/server_log_scan.ps1 -Dir <folder> [-Patterns wddm,owner] [-Filter server_*.log]
#   pwsh tools/gates/server_log_scan.ps1 -Dir <folder> -Step <release gate step name>   (sets chosen from the name)
#
# Pattern sets:
#   owner  "E owner check: <func> called on thread <id>": a call into the engine's state from a thread that is not the
#          engine thread (src/server/engine_thread.h, docs/dev/threads.md). Any line fails.
#   wddm   v0.2.2.1 (issues #5 / #6) / v0.2.2.2 (REVIEW-GOV-1 P2-3): the server went over the WDDM budget, GPU memory
#          spilled, or the memory governor acted (gate runs have no other program taking VRAM). The warning lines
#          themselves, not the "vram watch:" line that describes the rule.
# -Step: srv_* -> wddm,owner; golden_* -> owner (golden.ps1 scans each server case itself); other steps: no scan (exit 0,
#        "not a server step").
# Output: "server log scan: PASS|FAIL ..." (one line), then one "  <file>:<line>: <text>" line per match.
# Exit: 0 pass (or not a server step); 1 a line matched; 2 no server log found (or the folder is missing); 3 usage.
param(
    [Parameter(Mandatory)][string]$Dir,
    [string[]]$Patterns = @(),
    [string]$Step = "",
    [string]$Filter = "server_*.log"
)
$ErrorActionPreference = "Stop"
$sets = [ordered]@{
    # the log line itself (src/server/log.cpp "yyyy-mm-dd HH:MM:SS.mmm L message"), not request text quoted in another line
    owner = "^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} E owner check: "
    wddm  = "over the WDDM budget|GPU memory spill(:| continues| is over)|memory governor: .* free under the Windows video memory|memory governor: KV pool re-allocated"
}
$Patterns = @($Patterns | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($Step) {
    if ($Patterns.Count -gt 0) { Write-Host "server log scan: -Step and -Patterns together"; exit 3 }
    if ($Step -like "srv_*") { $Patterns = @("wddm", "owner") }
    elseif ($Step -like "golden_*") { $Patterns = @("owner") }
    else { Write-Host "server log scan: PASS $Step is not a server step (no scan)"; exit 0 }
}
if ($Patterns.Count -eq 0) { $Patterns = @("owner") }
foreach ($p in $Patterns) { if (-not $sets.Contains($p)) { Write-Host "server log scan: unknown pattern set '$p' (known: $($sets.Keys -join ', '))"; exit 3 } }
$rx = (@($Patterns | ForEach-Object { "(?:$($sets[$_]))" }) -join "|")
$slogs = @()
if (Test-Path -LiteralPath $Dir) { $slogs = @(Get-ChildItem -LiteralPath $Dir -Recurse -File -Filter $Filter -ErrorAction SilentlyContinue) }
if ($slogs.Count -eq 0) {
    Write-Host "server log scan: FAIL no server log ($Filter) found under $Dir [$($Patterns -join ',')]"
    exit 2
}
$hits = @($slogs | Select-String -Pattern $rx -ErrorAction SilentlyContinue)
if ($hits.Count -gt 0) {
    $h0 = $hits[0]
    $t0 = "$($h0.Line)"
    Write-Host "server log scan: FAIL $($hits.Count) line(s) matched [$($Patterns -join ',')] in $($slogs.Count) server log(s) under $Dir, first $($h0.Filename):$($h0.LineNumber): $($t0.Substring(0, [Math]::Min(200, $t0.Length)))"
    foreach ($h in $hits) { Write-Host "  $($h.Path):$($h.LineNumber): $($h.Line)" }
    exit 1
}
Write-Host "server log scan: PASS $($slogs.Count) server log(s) under $Dir, 0 line(s) matched [$($Patterns -join ',')]"
exit 0
