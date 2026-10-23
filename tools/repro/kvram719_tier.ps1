# SPDX-License-Identifier: Apache-2.0
# kvram719 repro: a RAM tier that, with the GPU-pinned memory already in use on the machine, crosses the Windows
# limit for GPU-locked system memory (about half of the physical RAM per adapter). Before the fix the server pinned it
# anyway and the first chat request failed with
#   E req 1 | slot 0 | failed: hipMemcpy D2H failed: unspecified launch failure (hipError 719)
# (AMD_LOG_LEVEL=1: "PAL failed to submit CMD! result:-5"). Fixed: the server lowers the RAM tier to what the
# limit leaves ("adjusted: --kv-ram-mb N -> M MiB (... GPU-pinned ...)") and the request is answered.
#
#   pwsh tools/repro/kvram719_tier.ps1 -Exe <whirl-server.exe> -Model <gguf> [-RamMb N | -OverMb 2048] [-Port 8197]
#        [-Device r9700|8060s] [-Work DIR]
# -RamMb: the --kv-ram-mb to request; default = the limit left now by the server's own machine rule (half of the RAM
#         minus the largest adapter's "GPU Adapter Memory Shared Usage") plus -OverMb, so the request is over it by design.
# Adapters: each PDH instance "GPU Adapter Memory(luid_0x<H>_0x<L>_phys_0)" is named through the DXGI adapter registry
# (HKLM\SOFTWARE\Microsoft\DirectX\{GUID}: AdapterLuid, Description, VendorId, SharedSystemMemory). The HIP GPUs are the
# AMD adapters (VendorId 0x1002); the selected one is the -Device GPU (R9700 / 8060S in its description). Adapters that
# are not HIP GPUs (Microsoft Basic Render Driver, a compute-only adapter without a DXGI entry) are reported, not judged.
# PASS (exit 0), all of (REVIEW-B0 A1):
#   - the request is answered and the server log has no "(hipError N)" / "| failed:" line;
#   - a RAM tier is in use (else INVALID, exit 3: a run with no tier says nothing about the limit);
#   - the selected adapter's Shared Usage with the server up <= its limit: min(half of the RAM, its DXGI
#     SharedSystemMemory, the non-local budget / n the server logged) - the server's own per-adapter rule
#     (src/common/startup_fit.cpp pinnedRoomMb), without its reserves;
#   - no other HIP GPU's Shared Usage rose by more than 1024 MiB while the server was up (s8-719: a pinned buffer mapped
#     to every HIP GPU is charged on each; the rule of tools/repro/s8_719_root.ps1).
# FAIL: exit 1. Setup error: exit 2. The line before the verdict is "kvram719: REPORT verdict=... reason=... hip719=N ..."
# (reason: request = failed request or error line, over_limit, other_hip_rise, tier_off; hip719 = "(hipError 719)" lines).
# Kills only the server it started (by PID); uses --kv-ssd-mode off (no SSD files).
#   pwsh tools/repro/kvram719_tier.ps1 -SelfTest   checks the pass judgement on recorded numbers (no GPU, no server)
param([string]$Exe, [string]$Model, [int]$RamMb = 0, [int]$OverMb = 2048, [int]$Port = 8197,
      [ValidateSet("r9700", "8060s")][string]$Device = "r9700", [string]$Work = "", [switch]$SelfTest)
$ErrorActionPreference = "Stop"
# The verdict from what one run saw: res = the request result text, bad = hipError / "failed:" lines in the server log,
# tierMb = the RAM tier in use (-1: no size line), adapters = one entry per PDH adapter
# @{ luid; name; hip = HIP GPU; sel = the selected GPU; before / up = Shared Usage MiB before the server / with it up },
# limitMiB = the selected adapter's limit. Returns @{ v = PASS|FAIL|INVALID; why = reason }.
function Get-Kvram719Verdict([string]$res, [int]$bad, [int64]$tierMb, [object[]]$adapters, [int64]$limitMiB) {
    if ($res -notlike "answered*" -or $bad -gt 0) { return @{ v = "FAIL"; why = "request" } }
    if ($tierMb -le 0) { return @{ v = "INVALID"; why = "tier_off" } }
    $sel = @($adapters | Where-Object { $_.sel })
    if ($sel.Count -ne 1) { return @{ v = "FAIL"; why = "no_selected_adapter" } }
    if ($sel[0].up -gt $limitMiB) { return @{ v = "FAIL"; why = "over_limit" } }
    $rise = @($adapters | Where-Object { $_.hip -and -not $_.sel } | ForEach-Object { $_.up - $_.before } | Measure-Object -Maximum).Maximum
    if ($rise -gt 1024) { return @{ v = "FAIL"; why = "other_hip_rise" } }
    return @{ v = "PASS"; why = "ok" }
}
function A([string]$luid, [string]$name, [bool]$hip, [bool]$sel, [int64]$before, [int64]$up) {
    [pscustomobject]@{ luid = $luid; name = $name; hip = $hip; sel = $sel; before = $before; up = $up }
}
if ($SelfTest) {
    # This host: 32407 MiB RAM, limit 16203 (half the RAM; SharedSystemMemory 16204). LUIDs of 2026-10-23: 0x182fa 8060S,
    # 0x1aa96 R9700, 0x1aa5e Basic Render, 0x1cb83 compute-only adapter (906 MiB of WorkloadsSessionHost, always there).
    $s8 = "00000000_000182fa"; $r9 = "00000000_0001aa96"; $br = "00000000_0001aa5e"; $np = "00000000_0001cb83"
    $cases = @(
        # MEASURED scratch\s8_719\case_s0222_ram8192 (v0.2.2.2, 8192, both GPUs visible, 2026-10-22 23:39): "req 1 ...
        # failed: ... (hipError 719)"; counters not recorded (the request alone decides)
        @{ n = "v0222_719_measured"; res = "request failed: 500"; bad = 2; tier = 8192
           ad = @((A $s8 "8060S" $true $true 0 0), (A $r9 "R9700" $true $false 0 0)); want = "FAIL"; why = "request" },
        # MEASURED scratch\s8_719\r2_case_old_ram8192_8060s_1 (0.2.3 clamp build, 2 HIP GPUs, on the 8060S): answered,
        # but the R9700 was charged too (14 -> 5404). The old bars (largest <= 16203, all adapters 11857 <= 16203) passed it.
        @{ n = "double_charge_measured"; res = "answered (32 tokens)"; bad = 0; tier = 4096
           ad = @((A $s8 "8060S" $true $true 68 5547), (A $r9 "R9700" $true $false 14 5404), (A $br "Basic" $false $false 0 0), (A $np "-" $false $false 906 906)); want = "FAIL"; why = "other_hip_rise" },
        # MEASURED scratch\s8_719\r2_case_s8s_ram8192_env_1 (root fix, 8060S, 7680 in use): one adapter charged
        @{ n = "s8s_root_measured"; res = "answered (32 tokens)"; bad = 0; tier = 7680
           ad = @((A $s8 "8060S" $true $true 68 9112), (A $r9 "R9700" $true $false 14 14), (A $br "Basic" $false $false 0 0), (A $np "-" $false $false 906 906)); want = "PASS"; why = "ok" },
        # MEASURED scratch\s8_719\r2_case_s97_ram8192_env_1 (root fix, R9700, 8192 in use)
        @{ n = "s97_root_measured"; res = "answered (32 tokens)"; bad = 0; tier = 8192
           ad = @((A $s8 "8060S" $true $false 49 49), (A $r9 "R9700" $true $true 14 9613), (A $br "Basic" $false $false 0 0), (A $np "-" $false $false 906 906)); want = "PASS"; why = "ok" },
        # REVIEW-B0 A1 (constructed from the 18:15 measured background 1357 / 1556 / 906): HEAD near the limit on the R9700,
        # the 8060S holding another program's 1357 MiB (no rise), the compute adapter at 2500: PASS (a13eccb's sum bar failed it)
        @{ n = "background_other_adapters"; res = "answered (32 tokens)"; bad = 0; tier = 12288
           ad = @((A $s8 "8060S" $true $false 1357 1357), (A $r9 "R9700" $true $true 1556 15800), (A $br "Basic" $false $false 0 0), (A $np "-" $false $false 2500 2500)); want = "PASS"; why = "ok" },
        # constructed: the selected adapter itself over its limit
        @{ n = "selected_over_limit"; res = "answered (32 tokens)"; bad = 0; tier = 16384
           ad = @((A $s8 "8060S" $true $true 68 16900), (A $r9 "R9700" $true $false 14 14)); want = "FAIL"; why = "over_limit" },
        # MEASURED fixb s23_ram8192 (the tier was off: shared 1453): the run proves nothing about s8-719
        @{ n = "tier_off"; res = "answered (32 tokens)"; bad = 0; tier = 0
           ad = @((A $s8 "8060S" $true $true 68 1453), (A $r9 "R9700" $true $false 14 14)); want = "INVALID"; why = "tier_off" },
        @{ n = "no_size_line"; res = "answered (32 tokens)"; bad = 0; tier = -1
           ad = @((A $s8 "8060S" $true $true 68 1453), (A $r9 "R9700" $true $false 14 14)); want = "INVALID"; why = "tier_off" }
    )
    $nbad = 0
    foreach ($c in $cases) {
        $v = Get-Kvram719Verdict $c.res $c.bad $c.tier $c.ad 16203
        $ok = $v.v -eq $c.want -and $v.why -eq $c.why
        if (-not $ok) { $nbad++ }
        Write-Host ("kvram719 selftest {0,-26} got {1,-7} {2,-20} want {3,-7} {4,-20} {5}" -f $c.n, $v.v, $v.why, $c.want, $c.why, $(if ($ok) { "ok" } else { "WRONG" }))
    }
    Write-Host "kvram719 selftest: $(if ($nbad) { "FAIL ($nbad wrong)" } else { 'PASS' })"
    exit $(if ($nbad) { 1 } else { 0 })
}
if (-not $Exe -or -not (Test-Path $Exe) -or -not $Model -or -not (Test-Path $Model)) { Write-Host "kvram719: -Exe and -Model must exist"; exit 2 }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-kvram719" }
New-Item -ItemType Directory -Force $Work | Out-Null
# DXGI adapters by LUID ("<high 8 hex>_<low 8 hex>", the PDH instance spelling)
$dx = @{}
foreach ($k in (Get-ChildItem HKLM:\SOFTWARE\Microsoft\DirectX -ErrorAction SilentlyContinue)) {
    $p = Get-ItemProperty $k.PSPath
    if ($null -eq $p.AdapterLuid -or [uint64]$p.AdapterLuid -eq 0) { continue }  # a stale entry of an earlier install has LUID 0
    $l = [uint64]$p.AdapterLuid
    $dx[("{0:x8}_{1:x8}" -f ($l -shr 32), ($l -band 0xffffffff))] = $p
}
function SharedByAdapter {
    $h = @{}
    foreach ($s in (Get-Counter '\GPU Adapter Memory(*)\Shared Usage').CounterSamples) {
        if ($s.InstanceName -match 'luid_0x([0-9a-f]+)_0x([0-9a-f]+)') { $h["$($Matches[1])_$($Matches[2])"] = [int64]($s.CookedValue / 1MB) }
    }
    $h
}
$want = if ($Device -eq "r9700") { "R9700" } else { "8060S" }
$s0 = SharedByAdapter
# only LUIDs of this boot (the registry keeps entries of earlier boots / installs)
$selKey = @($dx.Keys | Where-Object { $s0.ContainsKey($_) -and $dx[$_].VendorId -eq 0x1002 -and $dx[$_].Description -like "*$want*" })
if ($selKey.Count -ne 1) { Write-Host "kvram719: no single DXGI adapter named *$want* (found $($selKey.Count))"; exit 2 }
$selKey = $selKey[0]
$totalMiB = [int64][math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)
$pinned0 = [int64](($s0.Values | Measure-Object -Maximum).Maximum)
$limit = [int64][math]::Floor($totalMiB / 2)
if ($dx[$selKey].SharedSystemMemory) { $limit = [math]::Min($limit, [int64][math]::Floor([uint64]$dx[$selKey].SharedSystemMemory / 1MB)) }
if ($RamMb -le 0) { $RamMb = [int]([math]::Max(1024, [int64]($totalMiB / 2) - $pinned0 + $OverMb)) }
Write-Host ("kvram719: RAM {0} MiB, selected adapter {1} ({2}), its limit ~{3} MiB, largest adapter now {4} MiB, request --kv-ram-mb {5}" -f $totalMiB, $selKey, $dx[$selKey].Description, $limit, $pinned0, $RamMb)
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
if ($up) {
    $body = @{ model = "x"; messages = @(@{ role = "user"; content = "Write one sentence about the sea." }); max_tokens = 32; temperature = 0 } | ConvertTo-Json -Depth 5
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180
        $res = "answered ($($r.usage.completion_tokens) tokens)"
    } catch { $res = "request failed: $($_.Exception.Message)" }
}
$s1 = SharedByAdapter  # with the server up (after the request)
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() }
$txt = @(Get-Content $log -ErrorAction SilentlyContinue) + @(Get-Content (Join-Path $Work "stdout.txt") -ErrorAction SilentlyContinue)
# "(hipError N)" is how a HIP failure is logged; the fixed server's "adjusted:" line names the error number too (s8-719)
$bad = @($txt | Select-String -Pattern "\(hipError \d+\)|\| failed:")
$h719 = @($txt | Select-String -SimpleMatch "(hipError 719)")
$tierLine = @($txt | Select-String -Pattern "kv tier: RAM tier size|adjusted: .*kv-ram-mb|kv tier: RAM tier off|kv tier: pinned memory now")
$tierLine | ForEach-Object { Write-Host "  log: $($_.Line)" }
$bad | Select-Object -First 3 | ForEach-Object { Write-Host "  log: $($_.Line)" }
$sizeM = @($txt | Select-String -Pattern "kv tier: RAM tier size (\d+) MiB") | Select-Object -Last 1
$tierMb = if ($sizeM) { [int64]$sizeM.Matches[0].Groups[1].Value } elseif (@($txt | Select-String -Pattern "kv tier: RAM tier off|kv tier: off").Count) { 0 } else { -1 }
# the server's per-GPU limit for a discrete GPU: its logged DXGI non-local budget / the HIP GPU count (startup_fit.cpp)
$pin = @($txt | Select-String -Pattern "kv tier: pinned memory now: (\d+) HIP GPU\(s\) \[(.*?)\]") | Select-Object -Last 1
if ($pin) {
    $n = [int64]$pin.Matches[0].Groups[1].Value
    foreach ($m in [regex]::Matches($pin.Matches[0].Groups[2].Value, "(\d+) MiB non-local budget")) { $limit = [math]::Min($limit, [int64][math]::Floor([int64]$m.Groups[1].Value / [math]::Max($n, 1))) }
}
$ads = @()
foreach ($k in $s1.Keys) {
    $isHip = $dx.ContainsKey($k) -and $dx[$k].VendorId -eq 0x1002
    $ads += A $k $(if ($dx.ContainsKey($k)) { $dx[$k].Description } else { "(no DXGI adapter)" }) $isHip ($k -eq $selKey) $(if ($s0.ContainsKey($k)) { $s0[$k] } else { 0 }) $s1[$k]
}
$ads | ForEach-Object { Write-Host ("  shared usage luid {0} {1}{2}{3}: {4} -> {5} MiB (rise {6})" -f $_.luid, $_.name, $(if ($_.hip) { " [HIP GPU]" } else { "" }), $(if ($_.sel) { " [selected]" } else { "" }), $_.before, $_.up, ($_.up - $_.before)) }
$v = Get-Kvram719Verdict $res $bad.Count $tierMb $ads $limit
$selA = @($ads | Where-Object { $_.sel })
$oth = @($ads | Where-Object { $_.hip -and -not $_.sel } | ForEach-Object { $_.up - $_.before } | Measure-Object -Maximum).Maximum
Write-Host ("kvram719: REPORT verdict={0} reason={1} hip719={2} errlines={3} tier_mib={4} requested={5} selected={6} selected_up_mib={7} limit_mib={8} other_hip_rise_mib={9} result={10}" -f `
    $v.v, $v.why, $h719.Count, $bad.Count, $tierMb, $RamMb, $selKey, $(if ($selA.Count) { $selA[0].up } else { -1 }), $limit, $(if ($null -ne $oth) { $oth } else { 0 }), ($res -replace ' ', '_'))
if ($v.v -eq "INVALID") { Write-Host "kvram719: INVALID - no RAM tier was in use, so this run says nothing about the 719 limit" }
Write-Host "kvram719: $($v.v)"
exit $(if ($v.v -eq "PASS") { 0 } elseif ($v.v -eq "INVALID") { 3 } else { 1 })
