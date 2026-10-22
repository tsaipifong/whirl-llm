# SPDX-License-Identifier: Apache-2.0
# s8-719 review C4 (2026-10-23): whirl refuses a repeated --device ("give --device once"); whirl-server must do the
# same instead of silently taking the last one. CPU only: the arguments are refused before any model or GPU work.
#   pwsh tools/repro/server_device_once.ps1 -Build DIR
# PASS (exit 0): every repeated form exits non-zero within 20 s with "give --device once"; a single --device is not
# refused for that reason. FAIL: exit 1. Setup: exit 2.
param([string]$Build)
$ErrorActionPreference = "Continue"
$srv = Join-Path $Build "whirl-server.exe"
$cli = Join-Path $Build "whirl.exe"
if (-not (Test-Path $srv) -or -not (Test-Path $cli)) { Write-Host "server_device_once: setup (exes missing in $Build)"; exit 2 }
$work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-device-once-$PID"
New-Item -ItemType Directory -Force $work | Out-Null
$nul = Join-Path $work "empty.txt"
Set-Content $nul -Value $null
$model = Join-Path $work "no-such-model.gguf"   # never loaded: the argument check comes first
function Run([string]$exe, [string[]]$a) {
    $e = Join-Path $work "err.txt"; $o = Join-Path $work "out.txt"
    $p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardInput $nul -RedirectStandardError $e -RedirectStandardOutput $o -NoNewWindow -PassThru
    $ex = $p.WaitForExit(20000)
    if (-not $ex) { Stop-Process -Id $p.Id -Force; return @{ code = -1; text = "did not exit in 20 s" } }
    @{ code = $p.ExitCode; text = ((@(Get-Content $e -ErrorAction SilentlyContinue) + @(Get-Content $o -ErrorAction SilentlyContinue)) -join " / ") }
}
$bad = @()
$cases = @(
    @{ exe = $srv; a = @($model, "--device", "0", "--device", "1"); want = $true },
    @{ exe = $srv; a = @($model, "--device", "0", "-dev", "1"); want = $true },
    @{ exe = $srv; a = @($model, "-dev", "pci:71:0", "-dev", "pci:196:0"); want = $true },
    @{ exe = $cli; a = @("chat", $model, "hi", "--device", "0", "--device", "1"); want = $true },
    @{ exe = $srv; a = @($model, "--device", "0"); want = $false }
)
foreach ($c in $cases) {
    $r = Run $c.exe $c.a
    $once = $r.text -match "give --device once"
    $ok = if ($c.want) { $r.code -ne 0 -and $r.code -ne -1 -and $once } else { -not $once }
    Write-Host ("  {0} {1}: exit {2} {3} ; {4}" -f (Split-Path $c.exe -Leaf), ($c.a[1..($c.a.Count - 1)] -join ' '), $r.code, $(if ($ok) { "ok" } else { "FAIL" }), $r.text)
    if (-not $ok) { $bad += "$(Split-Path $c.exe -Leaf) $($c.a -join ' ')" }
}
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($bad.Count) { Write-Host "server_device_once: FAIL ($($bad.Count) case(s))"; exit 1 }
Write-Host "server_device_once: PASS"
exit 0
