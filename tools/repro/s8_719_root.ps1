# SPDX-License-Identifier: Apache-2.0
# s8-719 root repro: with two AMD GPUs in the machine (Radeon 8060S + R9700) and NO visibility variable set,
# a server on one GPU with an explicit RAM tier must answer, and the tier must not be cut by the pinned-room
# (hipError 719) limit. Before the root fix every pinned host buffer was charged as shared GPU memory on BOTH
# adapters: the first request failed with hipError 719 (0.2.2.2) or the tier was cut by the s8-719 clamp
# ("adjusted: --kv-ram-mb 8192 -> 4096 MiB (lowered to ...: the GPUs can keep only so much pinned host memory",
# 0.2.3 802b6e1). Fixed: WHIRL makes HIP see only the GPU it uses, so the buffer is charged on that adapter only.
#
#   pwsh tools/repro/s8_719_root.ps1 -Exe <whirl-server.exe> -Model <gguf> [-RamMb 8192] [-Device 8060s|r9700|auto]
#        [-Select env|option] [-Port 8198] [-Work DIR] [-Mmproj <gguf>] [-DeviceSpec pci:B:D] [-ExpectGpu pci:B:D]
# -Select env: WHIRL_DEVICE=<Device>; option: --device <Device> (the command-line option); -Device auto: neither.
# -DeviceSpec: the value given instead of -Device (e.g. pci:196:0). -ExpectGpu: the startup line must name it.
# PASS (exit 0), all of:
#   - HTTP 200 for one chat request;
#   - no "(hipError" line and no "| failed:" line in the log;
#   - no adjustment by the pinned-room limit ("the GPUs can keep only so much pinned host memory"); the
#     half-of-available-RAM adjustment ("more than half of the N MiB of RAM available now") is allowed and is
#     reported, and then the size used must equal its "-> N MiB"; without any adjustment the size used = -RamMb;
#   - the "kv tier: pinned memory now:" line says "1 HIP GPU(s)";
#   - the startup line "GPU pci:B:D (NAME) selected" names -ExpectGpu (when given);
#   - no other adapter's "GPU Adapter Memory Shared Usage" rose by more than 1024 MiB while the server was up.
# The last line before the verdict is "s8_719_root: REPORT key=value ..." (requested, used, rule, available RAM at
# start, other-adapter rise, 1-HIP-GPU line, HTTP status, hipError count, selected GPU). FAIL: exit 1. Setup: exit 2.
# Kills only the server it started (by PID); --kv-ssd-mode off (no SSD files).
param([string]$Exe, [string]$Model, [int]$RamMb = 8192, [ValidateSet("8060s", "r9700", "auto")][string]$Device = "8060s",
      [ValidateSet("env", "option")][string]$Select = "env", [int]$Port = 8198, [string]$Work = "", [string]$Mmproj = "",
      [string]$DeviceSpec = "", [int]$ExpectMb = 0, [string]$ExpectGpu = "")
# -ExpectMb is accepted for old callers and ignored: the size is checked against the log (see PASS above)
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
$availStart = [int64]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)  # MiB, before the server
$s0 = SharedByAdapter
$log = Join-Path $Work "server.log"
Remove-Item $log -ErrorAction SilentlyContinue
$env:AMD_LOG_LEVEL = "1"
if (Test-Path env:WHIRL_DEVICE) { Remove-Item env:WHIRL_DEVICE }
$a = @($Model, "--port", "$Port", "--log-file", $log, "--kv-ssd-mode", "off", "--kv-ram-mb", "$RamMb")
if ($Mmproj) { $a += @("--mmproj", $Mmproj) }
$spec = if ($DeviceSpec) { $DeviceSpec } else { $Device }
if ($Device -ne "auto" -or $DeviceSpec) {
    if ($Select -eq "env") { $env:WHIRL_DEVICE = $spec } else { $a += @("--device", $spec) }
}
Write-Host "s8_719_root: $Exe $($a -join ' ')  (WHIRL_DEVICE=$env:WHIRL_DEVICE)"
$p = Start-Process -FilePath $Exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $Work "stdout.txt") -RedirectStandardError (Join-Path $Work "stderr.txt")
$up = $false
for ($i = 0; $i -lt 240 -and -not $p.HasExited; $i++) {
    Start-Sleep 1
    try { if (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2) { $up = $true; break } } catch {}
}
$res = "server did not come up"
$http = 0
if ($up) {
    $body = @{ model = "x"; messages = @(@{ role = "user"; content = "Write one sentence about the sea." }); max_tokens = 32; temperature = 0 } | ConvertTo-Json -Depth 5
    try {
        $w = Invoke-WebRequest "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 240 -UseBasicParsing
        $http = [int]$w.StatusCode
        $r = $w.Content | ConvertFrom-Json
        $res = "answered ($($r.usage.completion_tokens) tokens)"
    } catch {
        if ($_.Exception.Response) { $http = [int]$_.Exception.Response.StatusCode }
        $res = "request failed: $($_.Exception.Message)"
    }
}
$s1 = SharedByAdapter
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
$rise = @()
foreach ($k in $s1.Keys) { $b = if ($s0.ContainsKey($k)) { $s0[$k] } else { 0 }; $rise += [pscustomobject]@{ luid = $k; before = $b; up = $s1[$k]; rise = $s1[$k] - $b } }
$rise = @($rise | Sort-Object rise -Descending)
$rise | ForEach-Object { Write-Host ("  shared usage luid {0}: {1} -> {2} MiB (rise {3})" -f $_.luid, $_.before, $_.up, $_.rise) }
$txt = @(Get-Content $log -ErrorAction SilentlyContinue)
$hipErr = @($txt | Select-String -SimpleMatch "(hipError")
$failed = @($txt | Select-String -SimpleMatch "| failed:")
$adjAll = @($txt | Select-String -Pattern "adjusted: .*(kv-ram-mb|WHIRL_KV_RAM_MB)")
$adjPinned = @($adjAll | Where-Object { $_.Line -match "can keep only so much pinned host memory" })
$adjHalf = @($adjAll | Where-Object { $_.Line -match "more than half of the \d+ MiB of RAM available now" })
$sizeLine = @($txt | Select-String -Pattern "kv tier: RAM tier size (\d+) MiB")
$used = if ($sizeLine.Count) { [int]$sizeLine[-1].Matches[0].Groups[1].Value } else { -1 }
$expect = $RamMb
$availLog = ""
if ($adjHalf.Count -and $adjHalf[-1].Line -match "-> (\d+) MiB") { $expect = [int]$Matches[1] }
if ($adjHalf.Count -and $adjHalf[-1].Line -match "half of the (\d+) MiB of RAM available now") { $availLog = $Matches[1] }
$rule = if ($adjPinned.Count) { "pinned-room(719-limit)" } elseif ($adjHalf.Count) { "half-of-available-RAM" } elseif ($adjAll.Count) { "other" } else { "none" }
$pinLine = @($txt | Select-String -Pattern "kv tier: pinned memory now: (\d+) HIP GPU\(s\)")
$hip1 = $pinLine.Count -gt 0 -and $pinLine[-1].Line -match "pinned memory now: 1 HIP GPU\(s\)"
$gpu = @($txt | Select-String -Pattern " I GPU pci:\d+:\d+ \(.+\) selected")
$gpuOk = $gpu.Count -gt 0 -and (-not $ExpectGpu -or $gpu[0].Line -match [regex]::Escape("GPU $ExpectGpu "))
$gpuName = if ($gpu.Count -and $gpu[0].Line -match "GPU (pci:\d+:\d+ \(.+\)) selected") { $Matches[1] } else { "none" }
@($txt | Select-String -Pattern "I GPU |I device |kv tier: RAM tier|adjusted: |kv tier: pinned memory now") | ForEach-Object { Write-Host "  log: $($_.Line)" }
@($hipErr + $failed) | Select-Object -First 3 | ForEach-Object { Write-Host "  log: $($_.Line)" }
$other = if ($rise.Count -ge 2) { $rise[1].rise } else { 0 }
$why = @()
if ($res -notlike "answered*" -or $http -ne 200) { $why += "$res (HTTP $http)" }
if ($hipErr.Count) { $why += "$($hipErr.Count) '(hipError' line(s) in the log" }
if ($failed.Count) { $why += "'| failed:' line in the log" }
if ($adjPinned.Count) { $why += "RAM tier cut by the pinned-room limit" }
if ($adjAll.Count -gt ($adjPinned.Count + $adjHalf.Count)) { $why += "an adjustment of another kind" }
if ($used -ne $expect) { $why += "RAM tier size $used MiB, expected $expect MiB" }
if (-not $hip1) { $why += "no 'pinned memory now: 1 HIP GPU(s)' line" }
if (-not $gpuOk) { $why += "no 'GPU pci:... selected' line$(if ($ExpectGpu) { " for $ExpectGpu" })" }
if ($other -gt 1024) { $why += "another adapter's shared usage rose $other MiB" }
Write-Host ("s8_719_root: REPORT requested={0} used={1} rule={2} avail_start_mib={3} avail_log_mib={4} other_rise_mib={5} one_hip_gpu={6} http={7} hiperror={8} gpu={9}" -f `
    $RamMb, $used, $rule, $availStart, $(if ($availLog) { $availLog } else { "-" }), $other, $(if ($hip1) { "yes" } else { "no" }), $http, $hipErr.Count, ($gpuName -replace ' ', '_'))
$pass = $why.Count -eq 0
Write-Host "s8_719_root: $res; second-largest adapter rise $other MiB"
Write-Host "s8_719_root: $(if ($pass) { 'PASS' } else { 'FAIL (' + ($why -join '; ') + ')' })"
exit $(if ($pass) { 0 } else { 1 })
