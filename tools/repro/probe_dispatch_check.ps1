# SPDX-License-Identifier: Apache-2.0
# s8-719 follow-up (2026-10-23, root719b case vis_fixed): every executable that calls gpusel::restrictToSelected lists the
# GPUs by starting ITSELF as `<exe> --whirl-gpu-probe`, so its main() must answer that argument. whirl-vis did not: the
# helper printed the usage text and ended with code 2, the log said "could not list the GPUs in a helper process (the
# helper process ended with code 2), so every GPU stays visible", and no GPU was chosen.
# PASS per exe: `<exe> --whirl-gpu-probe` (stdin NUL) ends within 60 s with exit 0 and prints at least one "whirl-gpu<TAB>"
# record. The probe initialises HIP and lists the GPUs (like --list-devices; no kernels): run it under the GPU locks.
# An exe that is missing from the build dir is a FAIL (the list below is the set that calls restrictToSelected).
#   pwsh tools/repro/probe_dispatch_check.ps1 -Build DIR [-Exes whirl.exe,whirl-server.exe,whirl-vis.exe]
# Last line: "probe_dispatch_check: PASS" (exit 0) or "probe_dispatch_check: FAIL (...)" (exit 1).
param([string]$Build, [string[]]$Exes = @("whirl.exe", "whirl-server.exe", "whirl-vis.exe"))
$ErrorActionPreference = "Continue"
$bad = @()
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-probe-dispatch-$PID"
New-Item -ItemType Directory -Force $tmp | Out-Null
foreach ($x in $Exes) {
    $exe = Join-Path $Build $x
    if (-not (Test-Path $exe)) { Write-Host "  ${x}: missing FAIL"; $bad += "$x (missing)"; continue }
    $o = Join-Path $tmp "out.txt"; $e = Join-Path $tmp "err.txt"; $nul = Join-Path $tmp "nul.txt"
    Set-Content -Path $nul -Value "" -NoNewline
    $p = Start-Process -FilePath $exe -ArgumentList "--whirl-gpu-probe" -RedirectStandardOutput $o -RedirectStandardError $e `
        -RedirectStandardInput $nul -NoNewWindow -PassThru
    $done = $p.WaitForExit(60000)
    if (-not $done) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    $code = if ($done) { $p.ExitCode } else { -1 }
    $recs = @(Get-Content $o -ErrorAction SilentlyContinue | Where-Object { $_ -match "^whirl-gpu`t" })
    $first = "$(@(Get-Content $o -ErrorAction SilentlyContinue)[0])"
    $ok = $code -eq 0 -and $recs.Count -ge 1
    Write-Host ("  {0}: exit {1}, {2} GPU record(s) {3} ; first line: {4}" -f $x, $code, $recs.Count, $(if ($ok) { "ok" } else { "FAIL" }),
        $first.Substring(0, [Math]::Min(160, $first.Length)))
    if (-not $ok) { $bad += "$x (exit $code, $($recs.Count) records)" }
}
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
if ($bad.Count) { Write-Host "probe_dispatch_check: FAIL ($($bad -join ', '))"; exit 1 }
Write-Host "probe_dispatch_check: PASS ($($Exes -join ', '))"
exit 0
