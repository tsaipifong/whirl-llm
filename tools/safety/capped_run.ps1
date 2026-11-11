# v0252-b3gate2: the capped run of a safety build (cpu_asan / cpu_analyze / cpu_tsa), dot-sourced by safety_step.ps1.
# Why: build_cfg.bat ran the quiet script (WHIRL_QUIET_WAIT, "wait while a GPU measurement holds the machine") INSIDE the
# step's time cap, so a QUIET_WANTED deferral was counted as build time. cpu_tsa (cap 7 min) failed that way in the b3 gate
# (quiet_wait.log 2026-11-11 07:37 / 07:44, mode incremental) and in its re-run (08:52, mode clean): "TIMEOUT after 7 min (cap)"
# with build.log holding only the DEFER line. Waiting for the machine is not build time (ctest / CI runners likewise start a
# test's timeout when it starts running, not while it waits for a lock or a runner).
# Now: the quiet wait runs here, before the cap clock starts and without a cap of its own (quiet_wait.ps1 has its own stale
# rules), then WHIRL_QUIET_SKIP=1 rides to the child so build_cfg.bat does not wait a second time inside the cap. A build that
# itself overruns the cap still TIMEs OUT and fails the step (test_capped_run.ps1 case C2).
# Usage: . capped_run.ps1; $r = Invoke-CappedRun -Bat <bat> -BatArgs @(...) -CapSec <s> -What <label> -Who <quiet label>
#        $r.ok (false = cap hit, tree killed), $r.exit, $r.quietSec, $r.runSec

function Invoke-QuietBeforeCap([string]$Who) {
    # returns the seconds waited (0 when skipped); never throws: a broken quiet script must not fail the step
    if ($env:WHIRL_QUIET_SKIP -eq "1" -or -not $env:WHIRL_QUIET_WAIT) { return 0.0 }
    $t = Get-Date
    try {
        & pwsh -NoProfile -File $env:WHIRL_QUIET_WAIT -Who $Who 2>&1 | ForEach-Object { Write-Host "   quiet: $_" }
    } catch { Write-Host "   quiet: script failed ($($_.Exception.Message)); running without the wait" }
    return [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
}

function Invoke-CappedRun([string]$Bat, [string[]]$BatArgs, [int]$CapSec, [string]$What, [string]$Who) {
    $qs = Invoke-QuietBeforeCap $Who
    if ($qs -gt 0) { Write-Host "$What`: quiet wait before the cap: $qs s (not counted in the $([math]::Round($CapSec / 60, 1)) min cap)" }
    $prevSkip = $env:WHIRL_QUIET_SKIP
    $env:WHIRL_QUIET_SKIP = "1"
    try {
        # cmd /c "<bat> args": the bat writes its own logs; this only waits with a cap and kills its own tree on overrun
        $q = @($BatArgs | ForEach-Object { '"' + $_ + '"' }) -join " "
        $t = Get-Date
        $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/d /c `"`"$Bat`" $q`"" -NoNewWindow -PassThru
        if (-not $p.WaitForExit([int64]$CapSec * 1000)) {
            & taskkill.exe /T /F /PID $p.Id *> $null
            Write-Host "$What`: TIMEOUT after $([math]::Round($CapSec / 60, 1)) min (cap; quiet wait before it $qs s): process tree $($p.Id) killed"
            return [pscustomobject]@{ ok = $false; exit = $null; quietSec = $qs; runSec = [math]::Round(((Get-Date) - $t).TotalSeconds, 1) }
        }
        $rs = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
        Write-Host "$What`: finished in $([math]::Round($rs / 60, 1)) min (cap clock; quiet wait before it $qs s), exit $($p.ExitCode)"
        return [pscustomobject]@{ ok = $true; exit = $p.ExitCode; quietSec = $qs; runSec = $rs }
    } finally {
        if ($null -eq $prevSkip) { Remove-Item Env:WHIRL_QUIET_SKIP -ErrorAction SilentlyContinue } else { $env:WHIRL_QUIET_SKIP = $prevSkip }
    }
}
