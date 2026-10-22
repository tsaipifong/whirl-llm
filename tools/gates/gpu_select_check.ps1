# SPDX-License-Identifier: Apache-2.0
# Release-gate regression check for s8-719 (hipError 719 with two GPUs) - permanent, owner 2026-10-23.
# On a host with more than one AMD GPU: start whirl-server on THIS gate's GPU with an explicit 8192 MiB RAM tier
# (a) with --device pci:<bus>:<device> and (b) without --device when this GPU is the default; one request each.
# PASS per run (tools\repro\s8_719_root.ps1): HTTP 200, no "(hipError" / "| failed:" line, no adjustment by the
# pinned-room limit (the half-of-available-RAM adjustment is allowed; the size used must then equal it, and the
# run's REPORT line says which rule applied), "kv tier: pinned memory now: 1 HIP GPU(s)", the startup line names
# the expected GPU, and no other adapter's "GPU Adapter Memory Shared Usage" rose by more than 1024 MiB.
# One GPU only: prints SKIP (exit 0).
# Only this gate's GPU is used, one run after the other (the other GPU stays idle; never two GPUs at once).
#   pwsh tools/gates/gpu_select_check.ps1 -Build DIR -Model SWIFT.gguf -Device r9700|8060s [-Port 8096] [-Work DIR]
# Last line: "gpu select check: ok", "gpu select check: SKIP (...)" or "gpu select check: FAIL (...)".
param([string]$Build, [string]$Model, [ValidateSet("r9700", "8060s")][string]$Device = "r9700", [int]$Port = 8096, [string]$Work = "")
$ErrorActionPreference = "Continue"
$cli = Join-Path $Build "whirl.exe"
$srv = Join-Path $Build "whirl-server.exe"
$repro = Join-Path $PSScriptRoot "..\repro\s8_719_root.ps1"
if (-not (Test-Path $cli) -or -not (Test-Path $srv) -or -not (Test-Path $Model)) { Write-Host "gpu select check: FAIL (setup: whirl.exe / whirl-server.exe / model missing)"; exit 2 }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-gpu-select-check" }
New-Item -ItemType Directory -Force $Work | Out-Null
foreach ($v in "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL") {
    if (Test-Path "env:$v") { Write-Host "gpu select check: SKIP ($v is set; the check needs every GPU visible)"; exit 0 }
}
$list = @(& $cli --list-devices 2>&1 | ForEach-Object { "$_" })
$devs = @($list | Where-Object { $_ -match '^\s+device \d+: ' })
$list | ForEach-Object { Write-Host "  $_" }
if ($devs.Count -lt 2) { Write-Host "gpu select check: SKIP (this host has $($devs.Count) GPU; the case needs two or more)"; exit 0 }
$arch = if ($Device -eq "8060s") { "gfx1151" } else { "gfx1201" }
$mine = @($devs | Where-Object { $_ -match "\($arch\)" })
if ($mine.Count -ne 1) { Write-Host "gpu select check: FAIL (expected exactly one $arch GPU in the list, found $($mine.Count))"; exit 1 }
if ($mine[0] -notmatch 'pci:(\d+:\d+) ') { Write-Host "gpu select check: FAIL (no pci:<bus>:<device> in '$($mine[0])')"; exit 1 }
$pci = "pci:$($Matches[1])"
$isDefault = $mine[0] -match '\[default\]'
$runs = @(@{ n = "explicit"; a = @("-Device", $Device, "-DeviceSpec", $pci, "-Select", "option") })
if ($isDefault) { $runs += @{ n = "default"; a = @("-Device", "auto") } }
$bad = @()
foreach ($r in $runs) {
    $o = @(& pwsh -NoProfile -File $repro -Exe $srv -Model $Model -RamMb 8192 -Port $Port -Work (Join-Path $Work $r.n) -ExpectGpu $pci @($r.a) 2>&1 | ForEach-Object { "$_" })
    $rc = $LASTEXITCODE
    $o | ForEach-Object { Write-Host "  [$($r.n)] $_" }
    if ($rc -ne 0) { $bad += "$($r.n) (exit $rc)" }
}
if (-not $isDefault) { Write-Host "  (this GPU is not the default here: the run without --device belongs to the default GPU's gate)" }
if ($bad.Count) { Write-Host "gpu select check: FAIL ($($bad -join ', '))"; exit 1 }
Write-Host "gpu select check: ok ($pci, $($runs.Count) run(s))"
exit 0
