# SPDX-License-Identifier: Apache-2.0
# s8-719 follow-up (2026-10-23): `whirl vis-encode` (ships in whirl.exe) and whirl-vis must choose the GPU with the
# same rules as whirl / whirl-server: --device / -dev / WHIRL_DEVICE (pci:<bus>:<device>, index, name), default =
# the GPU with the most GPU memory (the [default] mark of --list-devices), and the startup line naming it. Before:
# vis-encode always took the first R9700 (findDevice("R9700")) and printed no selection line.
# Needs the GPUs (encodes one image per run, --reps 1); run it under the GPU locks, one GPU at a time.
#   pwsh tools/repro/vis_device_select.ps1 -Build DIR -Mmproj MMPROJ.gguf -Image IMG.png
# PASS (exit 0) / FAIL (exit 1) / SKIP or setup (exit 2, e.g. only one GPU).
param([string]$Build, [string]$Mmproj, [string]$Image)
$ErrorActionPreference = "Continue"
$cli = Join-Path $Build "whirl.exe"
$vis = Join-Path $Build "whirl-vis.exe"
if (-not (Test-Path $cli) -or -not (Test-Path $Mmproj) -or -not (Test-Path $Image)) { Write-Host "vis_device_select: setup (missing files)"; exit 2 }
foreach ($v in "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "WHIRL_DEVICE", "WHIRL_HIP_DEVICE") { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
$list = @(& $cli --list-devices 2>&1 | ForEach-Object { "$_" })
$devs = @($list | Where-Object { $_ -match '^\s+device \d+: ' })
if ($devs.Count -lt 2) { Write-Host "vis_device_select: SKIP (needs two GPUs, found $($devs.Count))"; exit 2 }
$def = @($devs | Where-Object { $_ -match '\[default\]' })
if ($def.Count -ne 1 -or $def[0] -notmatch 'pci:(\d+:\d+)') { Write-Host "vis_device_select: setup (no [default] GPU in the list)"; exit 2 }
$pDef = "pci:$($Matches[1])"
$oth = @($devs | Where-Object { $_ -notmatch '\[default\]' })[0]
$null = $oth -match 'pci:(\d+:\d+)'
$pOth = "pci:$($Matches[1])"
$work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-vis-select-$PID"
New-Item -ItemType Directory -Force $work | Out-Null
function Run([string]$exe, [string[]]$a, [string]$envDev) {
    if ($envDev) { $env:WHIRL_DEVICE = $envDev } else { Remove-Item env:WHIRL_DEVICE -ErrorAction SilentlyContinue }
    $e = Join-Path $work "err.txt"; $o = Join-Path $work "out.txt"
    $p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardError $e -RedirectStandardOutput $o -NoNewWindow -PassThru
    $ex = $p.WaitForExit(120000)
    if (-not $ex) { Stop-Process -Id $p.Id -Force }
    Remove-Item env:WHIRL_DEVICE -ErrorAction SilentlyContinue
    @{ code = $(if ($ex) { $p.ExitCode } else { -1 }); text = ((@(Get-Content $e -ErrorAction SilentlyContinue) + @(Get-Content $o -ErrorAction SilentlyContinue)) -join " / ") }
}
$enc = @("vis-encode", $Mmproj, $Image, "--reps", "1")
$cases = @(
    @{ n = "whirl default";        exe = $cli; a = $enc;                              env = "";    want = $pDef },
    @{ n = "whirl --device other"; exe = $cli; a = $enc + @("--device", $pOth);       env = "";    want = $pOth },
    @{ n = "whirl -dev default";   exe = $cli; a = $enc + @("-dev", $pDef);           env = "";    want = $pDef },
    @{ n = "whirl WHIRL_DEVICE";   exe = $cli; a = $enc;                              env = $pOth; want = $pOth },
    @{ n = "whirl list refused";   exe = $cli; a = $enc + @("--device", "$pDef,$pOth"); env = "";  want = "" },
    @{ n = "whirl-vis default";    exe = $vis; a = $enc;                              env = "";    want = $pDef }
)
$bad = @()
foreach ($c in $cases) {
    if (-not (Test-Path $c.exe)) { $bad += "$($c.n) (exe missing)"; continue }
    $r = Run $c.exe $c.a $c.env
    $ok = if ($c.want) { $r.code -eq 0 -and $r.text -match ("GPU " + [regex]::Escape($c.want) + " \(.+?\): ") } else { $r.code -ne 0 -and $r.code -ne -1 -and $r.text -match "only one GPU" }
    Write-Host ("  {0}: exit {1} {2} ; {3}" -f $c.n, $r.code, $(if ($ok) { "ok" } else { "FAIL" }), ($r.text.Substring(0, [Math]::Min(400, $r.text.Length))))
    if (-not $ok) { $bad += $c.n }
}
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($bad.Count) { Write-Host "vis_device_select: FAIL ($($bad -join ', '))"; exit 1 }
Write-Host "vis_device_select: PASS (default $pDef, other $pOth)"
exit 0
