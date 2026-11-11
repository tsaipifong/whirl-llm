# v0252-b3gate2: test of tools/safety/capped_run.ps1 (the capped safety build run). About 15 s, no build, no GPU.
# The fake bat behaves like build_cfg.bat: it runs WHIRL_QUIET_WAIT unless WHIRL_QUIET_SKIP=1, then works for a while.
#   C1 the b3 case: the quiet script defers 5 s, the build takes 1 s, cap 4 s -> must finish (the deferral is not build time)
#   C2 a real overrun still fails: no quiet script, the build takes 7 s, cap 2 s -> TIMEOUT (ok = false)
#   C3 WHIRL_QUIET_SKIP=1 from the caller: the quiet script is not run at all
#   C4 the bat's exit code comes through; WHIRL_QUIET_SKIP is restored afterwards (unset before -> unset after)
#   C5 a broken quiet script (exit 1) does not fail the run
# -Helper: the file to test (default capped_run.ps1 next to this script); RED evidence points it at a shim of the old
# safety_step.ps1 RunCapped (quiet wait inside the cap).
# Prints "capped run test: ok (5/5)" or "capped run test: FAIL (...)"; exit 0 / 1.
param([string]$Helper = "")
$ErrorActionPreference = "Stop"
if (-not $Helper) { $Helper = Join-Path $PSScriptRoot "capped_run.ps1" }
if (-not (Test-Path -LiteralPath $Helper)) { Write-Output "capped run test: FAIL (helper missing: $Helper)"; exit 1 }
. $Helper
$d = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-capped-run-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $d | Out-Null
$mark = Join-Path $d "quiet_ran.txt"
$quiet = Join-Path $d "quiet.ps1"
Set-Content -LiteralPath $quiet -Value @"
param([string]`$Who = "")
Add-Content -LiteralPath "$mark" -Value "ran `$Who"
Start-Sleep -Seconds 5
"@
$badQuiet = Join-Path $d "quiet_bad.ps1"
Set-Content -LiteralPath $badQuiet -Value "param([string]`$Who = ''); exit 1"
$bat = Join-Path $d "fake_build.bat"
Set-Content -LiteralPath $bat -Encoding ascii -Value @"
@echo off
rem like build_cfg.bat: the quiet wait first, unless WHIRL_QUIET_SKIP=1
if "%WHIRL_QUIET_SKIP%"=="1" goto noquiet
if "%WHIRL_QUIET_WAIT%"=="" goto noquiet
pwsh -NoProfile -File "%WHIRL_QUIET_WAIT%" -Who "fake build"
:noquiet
pwsh -NoProfile -Command "Start-Sleep -Seconds %~1"
exit /b %~2
"@
$saved = @{ W = $env:WHIRL_QUIET_WAIT; S = $env:WHIRL_QUIET_SKIP }
$res = @()
function Case([string]$name, [bool]$cond, [string]$detail) {
    $script:res += $cond
    Write-Output "$(if ($cond) { 'ok  ' } else { 'BAD ' }) $name$(if (-not $cond) { ": $detail" })"
}
try {
    Remove-Item Env:WHIRL_QUIET_SKIP -ErrorAction SilentlyContinue
    $env:WHIRL_QUIET_WAIT = $quiet
    $r = Invoke-CappedRun -Bat $bat -BatArgs @("1", "0") -CapSec 4 -What "c1" -Who "test c1"
    Case "C1 quiet deferral (5 s) before a 1 s build, cap 4 s: finishes, not a cap hit" ($r.ok -and $r.exit -eq 0) "ok=$($r.ok) exit=$($r.exit)"

    Remove-Item Env:WHIRL_QUIET_WAIT -ErrorAction SilentlyContinue
    $r = Invoke-CappedRun -Bat $bat -BatArgs @("7", "0") -CapSec 2 -What "c2" -Who "test c2"
    Case "C2 a 7 s build with cap 2 s: TIMEOUT (real overruns still fail)" (-not $r.ok) "ok=$($r.ok)"

    Remove-Item -LiteralPath $mark -ErrorAction SilentlyContinue
    $env:WHIRL_QUIET_WAIT = $quiet; $env:WHIRL_QUIET_SKIP = "1"
    $r = Invoke-CappedRun -Bat $bat -BatArgs @("0", "0") -CapSec 20 -What "c3" -Who "test c3"
    Case "C3 caller WHIRL_QUIET_SKIP=1: quiet script not run" ($r.ok -and -not (Test-Path -LiteralPath $mark)) "ok=$($r.ok) marker=$(Test-Path -LiteralPath $mark)"

    Remove-Item Env:WHIRL_QUIET_SKIP -ErrorAction SilentlyContinue; Remove-Item Env:WHIRL_QUIET_WAIT -ErrorAction SilentlyContinue
    $r = Invoke-CappedRun -Bat $bat -BatArgs @("0", "3") -CapSec 20 -What "c4" -Who "test c4"
    Case "C4 exit 3 passed through, WHIRL_QUIET_SKIP unset again" ($r.ok -and $r.exit -eq 3 -and $null -eq $env:WHIRL_QUIET_SKIP) "ok=$($r.ok) exit=$($r.exit) skip='$env:WHIRL_QUIET_SKIP'"

    $env:WHIRL_QUIET_WAIT = $badQuiet
    $r = Invoke-CappedRun -Bat $bat -BatArgs @("0", "0") -CapSec 20 -What "c5" -Who "test c5"
    Case "C5 broken quiet script: the run still happens" ($r.ok -and $r.exit -eq 0) "ok=$($r.ok) exit=$($r.exit)"
} catch {
    $res += $false
    Write-Output "BAD  exception: $($_.Exception.Message)"
} finally {
    $env:WHIRL_QUIET_WAIT = $saved.W; $env:WHIRL_QUIET_SKIP = $saved.S
    if (-not $saved.W) { Remove-Item Env:WHIRL_QUIET_WAIT -ErrorAction SilentlyContinue }
    if (-not $saved.S) { Remove-Item Env:WHIRL_QUIET_SKIP -ErrorAction SilentlyContinue }
    Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction SilentlyContinue
}
$okN = @($res | Where-Object { $_ }).Count
if ($okN -eq 5 -and $res.Count -eq 5) { Write-Output "capped run test: ok (5/5)"; exit 0 }
Write-Output "capped run test: FAIL ($okN/5 as expected)"
exit 1
