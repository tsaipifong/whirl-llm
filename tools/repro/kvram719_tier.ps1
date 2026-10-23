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
#   pwsh tools/repro/kvram719_tier.ps1 -SelfTest   checks the pass judgement on recorded numbers (no GPU, no server)
param([string]$Exe, [string]$Model, [int]$RamMb = 0, [int]$OverMb = 2048, [int]$Port = 8197,
      [ValidateSet("r9700", "8060s")][string]$Device = "r9700", [string]$Work = "", [switch]$SelfTest)
$ErrorActionPreference = "Stop"
# The verdict from what one run saw: res = the request result text, bad = hipError / "failed:" lines in the server log,
# tierMb = the RAM tier in use (-1: no size line), maxMiB / sumMiB = GPU Adapter Memory Shared Usage with the server up
# (largest adapter / all adapters), limitMiB = half of the RAM. Returns PASS, FAIL or INVALID.
function Get-Kvram719Verdict([string]$res, [int]$bad, [int64]$tierMb, [int64]$maxMiB, [int64]$sumMiB, [int64]$limitMiB) {
    if ($res -like "answered*" -and $bad -eq 0 -and $maxMiB -le $limitMiB) { return "PASS" }
    return "FAIL"
}
if ($SelfTest) {
    # REVIEW-S8-719-2 C-3: recorded numbers (scratch\s8_719 results / fixb, 8060S 96 GB + R9700, 32407 MiB RAM, limit 16203)
    $cases = @(
        # s8-719 failure: 8192 on two HIP GPUs, R9700 Shared Usage 9489 (8060S value not recorded; sum constructed): the request failed
        @{ n = "s8719_fail_request"; res = "request failed: 500"; bad = 1; tier = 8192; max = 9489; sum = 18978; want = "FAIL" },
        # constructed (not measured): the same charge on both adapters and a request that still came back: over the shared GPU memory
        @{ n = "both_adapters_over"; res = "answered (32 tokens)"; bad = 0; tier = 8192; max = 9489; sum = 18978; want = "FAIL" },
        # the tier was off (fixb s23_ram8192: shared 1453): the run proves nothing about s8-719
        @{ n = "tier_off"; res = "answered (32 tokens)"; bad = 0; tier = 0; max = 1453; sum = 1500; want = "INVALID" },
        @{ n = "no_size_line"; res = "answered (32 tokens)"; bad = 0; tier = -1; max = 1453; sum = 1500; want = "INVALID" },
        # fixed build, one HIP GPU: 4096 charged once (fixb s23_ram4096 8060S 5550, R9700 idle)
        @{ n = "one_adapter_ok"; res = "answered (32 tokens)"; bad = 0; tier = 4096; max = 5550; sum = 5600; want = "PASS" }
    )
    $nbad = 0
    foreach ($c in $cases) {
        $v = Get-Kvram719Verdict $c.res $c.bad $c.tier $c.max $c.sum 16203
        $ok = $v -eq $c.want
        if (-not $ok) { $nbad++ }
        Write-Host ("kvram719 selftest {0,-20} got {1,-7} want {2,-7} {3}" -f $c.n, $v, $c.want, $(if ($ok) { "ok" } else { "WRONG" }))
    }
    Write-Host "kvram719 selftest: $(if ($nbad) { "FAIL ($nbad wrong)" } else { 'PASS' })"
    exit $(if ($nbad) { 1 } else { 0 })
}
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
$env:WHIRL_DEVICE = $Device  # explicit: since 0.2.3 the default is the GPU with the most memory
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
# "(hipError N)" is how a HIP failure is logged; the fixed server's "adjusted:" line names the error number too (s8-719)
$bad = @($txt | Select-String -Pattern "\(hipError \d+\)|\| failed:")
$tierLine = @($txt | Select-String -Pattern "kv tier: RAM tier size|adjusted: .*kv-ram-mb|kv tier: RAM tier off")
$tierLine | ForEach-Object { Write-Host "  log: $($_.Line)" }
$bad | Select-Object -First 3 | ForEach-Object { Write-Host "  log: $($_.Line)" }
Write-Host ("kvram719: {0}; GPU-pinned with the server up: {1} MiB (limit ~{2})" -f $res, $pinnedUp, $limit)
$verdict = Get-Kvram719Verdict $res $bad.Count -1 $pinnedUp $pinnedUp $limit
Write-Host "kvram719: $verdict"
exit $(if ($verdict -eq "PASS") { 0 } else { 1 })
