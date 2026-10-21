# SPDX-License-Identifier: Apache-2.0
# kvram719 repro: a RAM tier that, with the GPU-pinned memory already in use on the machine, crosses the Windows
# limit for GPU-locked system memory (about half of the physical RAM, all processes and adapters together).
# Before the fix the server pinned it anyway and the first chat request failed with
#   E req 1 | slot 0 | failed: hipMemcpy D2H failed: unspecified launch failure (hipError 719)
# (AMD_LOG_LEVEL=1: "PAL failed to submit CMD! result:-5"). Fixed: the server lowers the RAM tier to what the
# limit leaves ("adjusted: --kv-ram-mb N -> M MiB (... GPU-pinned ...)") and the request is answered.
#
#   pwsh tools/repro/kvram719_tier.ps1 -Exe <whirl-server.exe> -Model <gguf> [-RamMb N | -OverMb 2048] [-Port 8197]
#        [-Device r9700|8060s] [-Work DIR]
# -RamMb: the --kv-ram-mb to request; default = the limit left now (half of the RAM minus the largest adapter's
#         "GPU Adapter Memory Shared Usage") plus -OverMb, so the request is over the limit by design.
# PASS (exit 0): the request is answered, no hipError / "failed:" line in the server log, and the RAM tier in use
# plus what was pinned before is within the limit. FAIL: exit 1. Setup error: exit 2.
# Kills only the server it started (by PID); uses --kv-ssd-mode off (no SSD files).
param([string]$Exe, [string]$Model, [int]$RamMb = 0, [int]$OverMb = 2048, [int]$Port = 8197,
      [ValidateSet("r9700", "8060s")][string]$Device = "r9700", [string]$Work = "")
$ErrorActionPreference = "Stop"
if (-not $Exe -or -not (Test-Path $Exe) -or -not $Model -or -not (Test-Path $Model)) { Write-Host "kvram719: -Exe and -Model must exist"; exit 2 }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-kvram719" }
New-Item -ItemType Directory -Force $Work | Out-Null
function PinnedNowMiB { [int64](((Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Maximum).Maximum / 1MB) }
$totalMiB = [int64]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)
$pinned0 = PinnedNowMiB
$limit = [int64]($totalMiB / 2)
if ($RamMb -le 0) { $RamMb = [int]([math]::Max(1024, $limit - $pinned0 + $OverMb)) }
Write-Host ("kvram719: RAM {0} MiB, GPU-pinned limit ~{1} MiB, pinned now {2} MiB (largest adapter), request --kv-ram-mb {3}" -f $totalMiB, $limit, $pinned0, $RamMb)
$log = Join-Path $Work "kvram719_$RamMb.server.log"
Remove-Item $log -ErrorAction SilentlyContinue
$env:AMD_LOG_LEVEL = "1"
if ($Device -eq "8060s") { $env:WHIRL_DEVICE = "8060s" }
$a = @($Model, "--port", "$Port", "--log-file", $log, "--kv-ssd-mode", "off", "--kv-ram-mb", "$RamMb")
$p = Start-Process -FilePath $Exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $Work "stdout.txt") -RedirectStandardError (Join-Path $Work "stderr.txt")
$up = $false
for ($i = 0; $i -lt 180 -and -not $p.HasExited; $i++) {
    Start-Sleep 1
    try { if (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2) { $up = $true; break } } catch {}
}
$res = "server did not come up"
$pinnedUp = PinnedNowMiB
if ($up) {
    $body = @{ model = "x"; messages = @(@{ role = "user"; content = "Write one sentence about the sea." }); max_tokens = 32; temperature = 0 } | ConvertTo-Json -Depth 5
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180
        $res = "answered ($($r.usage.completion_tokens) tokens)"
    } catch { $res = "request failed: $($_.Exception.Message)" }
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
$txt = @(Get-Content $log -ErrorAction SilentlyContinue)
$bad = @($txt | Select-String -Pattern "hipError|\| failed:")
$tierLine = @($txt | Select-String -Pattern "kv tier: RAM tier size|adjusted: .*kv-ram-mb|kv tier: RAM tier off")
$tierLine | ForEach-Object { Write-Host "  log: $($_.Line)" }
$bad | Select-Object -First 3 | ForEach-Object { Write-Host "  log: $($_.Line)" }
Write-Host ("kvram719: {0}; GPU-pinned with the server up: {1} MiB (limit ~{2})" -f $res, $pinnedUp, $limit)
$pass = $res -like "answered*" -and $bad.Count -eq 0 -and $pinnedUp -le $limit
Write-Host "kvram719: $(if ($pass) { 'PASS' } else { 'FAIL' })"
exit $(if ($pass) { 0 } else { 1 })
