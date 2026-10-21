# REVIEW-GOV-2 independent test for REVIEW-GOV-1 P2-3 (reviewer, written from the bug description only).
# Bug: release_gate.ps1 srv_* steps fail only on "over the WDDM budget" / "GPU memory spill"; a governor episode in a
# normal gate run (no other program takes VRAM) silently repairs a budget drift and the gate passes.
# Expected: a srv_* step FAILS when a kept server log has the episode-start line ("memory governor: ... free under the
# Windows video memory") or "memory governor: KV pool re-allocated", and PASSES on the startup "memory governor: on:"
# line and the "HIP resource cache" line.
# Method: take the srv_* WDDM check's pattern ($wpat) and file filter ("server_*.log", -Recurse) from the given gate
# script itself, put each sample log as server_<n>.log in its own work dir (one in a sub dir, as servers may write
# there), and apply the same Get-ChildItem | Select-String the gate uses. CPU only.
#   pwsh -NoProfile -File gate_check_test.ps1 -Gate <tree>\tools\gates\release_gate.ps1
# SPDX-License-Identifier: Apache-2.0
param([Parameter(Mandatory)][string]$Gate, [string]$Work = "C:\ClaudeWorkspace\whirl-work\scratch\gov2review\gate_test_work")
$ErrorActionPreference = "Stop"
$src = Get-Content -Raw $Gate
# the srv_* block: from 'if ($s.name -like "srv_*")' to the first $wpat assignment inside it
$i = $src.IndexOf('if ($s.name -like "srv_*")')
if ($i -lt 0) { "ERROR: srv_* WDDM check block not found in $Gate"; exit 2 }
$blk = $src.Substring($i)
$m = [regex]::Match($blk, '\$wpat\s*=\s*"((?:[^"`]|`.)*)"')
if (-not $m.Success) { "ERROR: `$wpat not found in the srv_* block"; exit 2 }
$wpat = $m.Groups[1].Value
$fm = [regex]::Match($blk, 'Get-ChildItem \$wd -Recurse -File -Filter "([^"]+)"')
if (-not $fm.Success) { "ERROR: server log filter not found"; exit 2 }
$filter = $fm.Groups[1].Value
"gate: $Gate"
"pattern: $wpat"
"filter:  $filter (-Recurse)"
$samples = Join-Path $PSScriptRoot "gate_samples"
# sample -> expected step result
$cases = [ordered]@{
    "clean.log"            = "PASS"
    "startup_on.log"       = "PASS"   # "memory governor: on: ..." + "memory governor: HIP resource cache limited ..."
    "spill.log"            = "FAIL"   # sanity: the 0.2.2.1 check
    "episode_start.log"    = "FAIL"   # P2-3
    "pool_reallocated.log" = "FAIL"   # P2-3
}
$pass = 0; $fail = 0
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
$n = 0
foreach ($k in $cases.Keys) {
    ++$n
    $wd = Join-Path $Work "srv_case$n"
    $dst = if ($n % 2 -eq 0) { Join-Path $wd "sub" } else { $wd }
    New-Item -ItemType Directory -Force $dst | Out-Null
    # every case also carries the clean + startup lines, like a real server log
    $body = (Get-Content -Raw (Join-Path $samples "startup_on.log")) + (Get-Content -Raw (Join-Path $samples "clean.log")) + (Get-Content -Raw (Join-Path $samples $k))
    Set-Content -NoNewline -Encoding utf8 (Join-Path $dst "server_$n.log") $body
    $slogs = @(Get-ChildItem $wd -Recurse -File -Filter $filter -ErrorAction SilentlyContinue)
    $hits = @($slogs | Select-String -Pattern $wpat -ErrorAction SilentlyContinue)
    $got = if ($slogs.Count -eq 0) { "FAIL(no log)" } elseif ($hits.Count -gt 0) { "FAIL" } else { "PASS" }
    $ok = $got -eq $cases[$k]
    if ($ok) { ++$pass } else { ++$fail }
    "{0} {1,-22} expected {2}, step {3} ({4} log(s), {5} hit(s){6})" -f ($(if ($ok) { "ok  " } else { "FAIL" })), $k, $cases[$k], $got, $slogs.Count, $hits.Count, $(if ($hits.Count) { ": " + "$($hits[0].Line)".Substring(0, [Math]::Min(90, "$($hits[0].Line)".Length)) } else { "" })
}
Remove-Item -Recurse -Force $Work
"review_gov2 gate check tests: $pass passed, $fail failed"
exit $(if ($fail -gt 0) { 1 } else { 0 })
