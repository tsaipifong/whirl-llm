# SPDX-License-Identifier: Apache-2.0
# s8-719 follow-up (2026-10-23, root719b case vis_fixed): every executable that calls gpusel::restrictToSelected lists the
# GPUs by starting ITSELF as `<exe> --whirl-gpu-probe`, so its main() must answer that argument. whirl-vis did not: the
# helper printed the usage text and ended with code 2, the log said "could not list the GPUs in a helper process (the
# helper process ended with code 2), so every GPU stays visible", and no GPU was chosen.
# PASS per exe: `<exe> --whirl-gpu-probe` (stdin NUL) ends within 60 s with exit 0 and prints at least one "whirl-gpu<TAB>"
# record; every record is well-formed (15 fields, numbers where numbers belong, gfx arch, a name, distinct indices) and
# every exe prints the same GPU list. Release-gate step cpu_probe_dispatch runs it for whirl, whirl-server and whirl-vis
# (review-2 C4: an exe that calls the restriction but forgets the probe dispatch falls back to every GPU visible). The probe initialises HIP and lists the GPUs (like --list-devices; no kernels): run it under the GPU locks.
# An exe that is missing from the build dir is a FAIL (the list below is the set that calls restrictToSelected).
#   pwsh tools/repro/probe_dispatch_check.ps1 -Build DIR [-Exes whirl.exe,whirl-server.exe,whirl-vis.exe]
# Last line: "probe_dispatch_check: PASS" (exit 0) or "probe_dispatch_check: FAIL (...)" (exit 1).
param([string]$Build, [string[]]$Exes = @("whirl.exe", "whirl-server.exe", "whirl-vis.exe"))
$ErrorActionPreference = "Continue"
$bad = @()
$refKey = $null; $refExe = ""
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
    # well-formed (gpu_select.cpp formatRecord): 15 tab fields = tag, index, pci domain / bus / device / function,
    # total MiB, CUs, fit MiB, uuid, luid, integrated, supported, arch, name; indices distinct; total MiB > 0
    $malformed = @($recs | Where-Object {
        $f = $_ -split "`t"
        -not ($f.Count -eq 15 -and $f[1] -match '^\d+$' -and $f[2] -match '^\d+$' -and $f[3] -match '^\d+$' -and $f[4] -match '^\d+$' -and
              $f[5] -match '^\d+$' -and $f[6] -match '^[1-9]\d*$' -and $f[7] -match '^\d+$' -and $f[8] -match '^\d+$' -and $f[9] -ne '' -and
              $f[10] -ne '' -and $f[11] -match '^[01]$' -and $f[12] -match '^[01]$' -and $f[13] -match '^gfx[0-9a-f]+$' -and $f[14].Trim() -ne '')
    })
    $idx = @($recs | ForEach-Object { ($_ -split "`t")[1] })
    $dupIdx = $idx.Count -ne @($idx | Sort-Object -Unique).Count
    $ok = $code -eq 0 -and $recs.Count -ge 1 -and $malformed.Count -eq 0 -and -not $dupIdx
    # every exe lists the same GPUs (same probe code; a different list = a different code path or a broken dispatch)
    $key = (@($recs | ForEach-Object { $f = $_ -split "`t"; "$($f[1])/$($f[3]):$($f[4])/$($f[13])" }) -join ",")
    if ($ok) {
        if ($null -eq $refKey) { $refKey = $key; $refExe = $x }
        elseif ($key -ne $refKey) { $ok = $false; Write-Host "  ${x}: GPU list [$key] differs from $refExe [$refKey]" }
    }
    Write-Host ("  {0}: exit {1}, {2} GPU record(s), {3} malformed{4} {5} ; [{6}] first line: {7}" -f $x, $code, $recs.Count, $malformed.Count,
        $(if ($dupIdx) { ", duplicate index" } else { "" }), $(if ($ok) { "ok" } else { "FAIL" }), $key,
        $first.Substring(0, [Math]::Min(160, $first.Length)))
    if (-not $ok) { $bad += "$x (exit $code, $($recs.Count) records, $($malformed.Count) malformed)" }
}
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
if ($bad.Count) { Write-Host "probe_dispatch_check: FAIL ($($bad -join ', '))"; exit 1 }
Write-Host "probe_dispatch_check: PASS ($($Exes -join ', '))"
exit 0
