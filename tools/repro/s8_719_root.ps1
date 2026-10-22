# SPDX-License-Identifier: Apache-2.0
# s8-719 root repro: with two AMD GPUs in the machine (Radeon 8060S + R9700) and NO visibility variable set,
# a server on one GPU with an explicit RAM tier must keep that size and answer.
# Before the root fix the HIP runtime saw both GPUs, every pinned host buffer was charged as shared GPU memory
# on BOTH adapters, the first request failed with hipError 719 (0.2.2.2) or the tier was halved by the s8-719
# clamp ("adjusted: --kv-ram-mb 8192 -> 4096 MiB", 0.2.3 802b6e1). Fixed: WHIRL makes HIP see only the GPU it
# uses ("GPU: using device N ...; ... hidden ..."), so the buffer is charged on that adapter only.
#
#   pwsh tools/repro/s8_719_root.ps1 -Exe <whirl-server.exe> -Model <gguf> [-RamMb 8192] [-Device 8060s|r9700|auto]
#        [-Select env|option] [-Port 8198] [-Work DIR] [-Mmproj <gguf>]
# -Select env: WHIRL_DEVICE=<Device>; option: --device <Device> (the command-line option); auto: neither.
# PASS (exit 0): answered, no "(hipError N)" / "| failed:" line, "kv tier: RAM tier size <RamMb> MiB" with no
# "adjusted: --kv-ram-mb" line, the log names the GPU used and the hidden one, and no other adapter's
# "GPU Adapter Memory Shared Usage" rose by more than 1024 MiB while the server was up. FAIL: exit 1. Setup: exit 2.
# Kills only the server it started (by PID); --kv-ssd-mode off (no SSD files).
param([string]$Exe, [string]$Model, [int]$RamMb = 8192, [ValidateSet("8060s", "r9700", "auto")][string]$Device = "8060s",
      [ValidateSet("env", "option")][string]$Select = "env", [int]$Port = 8198, [string]$Work = "", [string]$Mmproj = "")
$ErrorActionPreference = "Stop"
if (-not $Exe -or -not (Test-Path $Exe) -or -not $Model -or -not (Test-Path $Model)) { Write-Host "s8_719_root: -Exe and -Model must exist"; exit 2 }
foreach ($v in "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "WHIRL_HIP_DEVICE", "WHIRL_KV_RAM_MB") {
    if (Test-Path "env:$v") { Remove-Item "env:$v" }
}
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-s8-719-root" }
New-Item -ItemType Directory -Force $Work | Out-Null
function SharedByAdapter {
    $h = @{}
    foreach ($s in (Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples) {
        if ($s.InstanceName -match 'luid_0x([0-9a-f]+)_0x([0-9a-f]+)') { $h["$($Matches[1])_$($Matches[2])"] = [int64]($s.CookedValue / 1MB) }
    }
    $h
}
$s0 = SharedByAdapter
$log = Join-Path $Work "server.log"
Remove-Item $log -ErrorAction SilentlyContinue
$env:AMD_LOG_LEVEL = "1"
if (Test-Path env:WHIRL_DEVICE) { Remove-Item env:WHIRL_DEVICE }
$a = @($Model, "--port", "$Port", "--log-file", $log, "--kv-ssd-mode", "off", "--kv-ram-mb", "$RamMb")
if ($Mmproj) { $a += @("--mmproj", $Mmproj) }
if ($Device -ne "auto") {
    if ($Select -eq "env") { $env:WHIRL_DEVICE = $Device } else { $a += @("--device", $Device) }
}
Write-Host "s8_719_root: $Exe $($a -join ' ')  (WHIRL_DEVICE=$env:WHIRL_DEVICE)"
$p = Start-Process -FilePath $Exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $Work "stdout.txt") -RedirectStandardError (Join-Path $Work "stderr.txt")
$up = $false
for ($i = 0; $i -lt 240 -and -not $p.HasExited; $i++) {
    Start-Sleep 1
    try { if (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2) { $up = $true; break } } catch {}
}
$res = "server did not come up"
if ($up) {
    $body = @{ model = "x"; messages = @(@{ role = "user"; content = "Write one sentence about the sea." }); max_tokens = 32; temperature = 0 } | ConvertTo-Json -Depth 5
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 240
        $res = "answered ($($r.usage.completion_tokens) tokens)"
    } catch { $res = "request failed: $($_.Exception.Message)" }
}
$s1 = SharedByAdapter
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
$rise = @()
foreach ($k in $s1.Keys) { $b = if ($s0.ContainsKey($k)) { $s0[$k] } else { 0 }; $rise += [pscustomobject]@{ luid = $k; before = $b; up = $s1[$k]; rise = $s1[$k] - $b } }
$rise = @($rise | Sort-Object rise -Descending)
$rise | ForEach-Object { Write-Host ("  shared usage luid {0}: {1} -> {2} MiB (rise {3})" -f $_.luid, $_.before, $_.up, $_.rise) }
$txt = @(Get-Content $log -ErrorAction SilentlyContinue)
$bad = @($txt | Select-String -Pattern "\(hipError \d+\)|\| failed:")
$adj = @($txt | Select-String -Pattern "adjusted: .*kv-ram-mb")
$size = @($txt | Select-String -Pattern "kv tier: RAM tier size $RamMb MiB")
$gpu = @($txt | Select-String -Pattern "^.* I GPU: using ")
$hidden = @($gpu | Where-Object { $_.Line -match "hidden" })
@($txt | Select-String -Pattern "I GPU: |I device |kv tier: RAM tier|adjusted: |kv tier: pinned memory now") | ForEach-Object { Write-Host "  log: $($_.Line)" }
$bad | Select-Object -First 3 | ForEach-Object { Write-Host "  log: $($_.Line)" }
$other = if ($rise.Count -ge 2) { $rise[1].rise } else { 0 }
$why = @()
if ($res -notlike "answered*") { $why += $res }
if ($bad.Count) { $why += "HIP error line in the log" }
if ($adj.Count) { $why += "RAM tier adjusted" }
if (-not $size.Count) { $why += "no 'RAM tier size $RamMb MiB' line" }
if (-not $hidden.Count) { $why += "no 'GPU: using ... hidden' line" }
if ($other -gt 1024) { $why += "another adapter's shared usage rose $other MiB" }
$pass = $why.Count -eq 0
Write-Host "s8_719_root: $res; second-largest adapter rise $other MiB"
Write-Host "s8_719_root: $(if ($pass) { 'PASS' } else { 'FAIL (' + ($why -join '; ') + ')' })"
exit $(if ($pass) { 0 } else { 1 })
