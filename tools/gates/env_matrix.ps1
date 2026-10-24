# SPDX-License-Identifier: Apache-2.0
# Environment-matrix smoke test (arch024-S14, PLAN c-3 E1-E4, E7) on ONE GPU: the gate's GPU. Every server here runs on
# that GPU, one after the other (never two GPUs at once; the other GPU is only enumerated). The CPU tables of the same
# matrix are tests/device (E2 carve-out) and tests/server --only env_matrix (E1/E3/E4/E5/E6 selection and sizing);
# E6 (the eGPU really unplugged) is the owner's checklist (progress\arch024-S14.E6_CHECKLIST.md), never automated.
#
#   pwsh tools/gates/env_matrix.ps1 -Build DIR -Model SWIFT.gguf -Device r9700|8060s -Cells E1,E3,E4,E7
#        [-Port 8110] [-Work DIR] [-Golden]          (E2 = 8060S only: WHIRL_VRAM_LIMIT_MB x3 + a fake 0.5 GB carve-out)
#
# Cells (each server: --kv-ram-mb 4096, --kv-ssd-mode off, one chat request, killed by PID):
#   E1  visibility: (a) nothing set + --device pci:<this>; (b) nothing set, no --device (only when this GPU is the
#       default); (c) HIP_VISIBLE_DEVICES=<this ordinal>, no --device; (d) the same + --device pci:<this>;
#       (e) HIP_VISIBLE_DEVICES=<the other ordinal> + --device pci:<this>: refused before any model work, the text
#       lists the GPUs ("GPUs in this computer"), no hipError; --list-devices under each setting (2, 2, 1, 1, 1 GPUs).
#       (f) v3 C-b2 "another program holds pinned memory": whirl-mem-hog pins 4096 MiB (it sees only this GPU), then
#       the (d) server again: the startup inputs' pin.g0.adapter_shared grows by >= 3584 MiB, the "room for the RAM
#       tier" drops by >= 3584 MiB (or to 0 / 512-step), and startup_lines.py (P-size replay) passes on both logs.
#   E3  the other GPU hidden: HIP_VISIBLE_DEVICES=<this ordinal>: the default is this GPU (= E1 c) and
#       --device pci:<other> is refused clearly. On the 8060S gate this is "R9700 not there" (PLAN E3).
#   E4  a desktop with other programs: whirl-mem-hog holds 6144 MiB VRAM + 2048 MiB pinned on this GPU, then
#       whirl-server-gate --suite basic (WHIRL_DEVICE=<this>); the step's server logs must show no WDDM-budget /
#       spill / governor line (server_log_scan wddm,owner) and P-size passes.
#   E7  WHIRL_CKPT_HOST=1: golden server (Swift + Ornith, three modes; needs -Golden and WHIRL_GOLDEN_* / WHIRL_GATE_*
#       models) and gpu_select_check.ps1, both with the variable; every server log names "(in pinned host memory)".
#   E2  (8060S) WHIRL_VRAM_LIMIT_MB 49152 / 32768 / 24576: each either answers (HTTP 200) or refuses with a clear text
#       (exit 5, no hipError); the pool / adjusted lines are logged. WHIRL_FAKE_DEDICATED_GB=0.5: refused with exit 5,
#       the text names Armoury Crate and the size to set, no hipError (issue #7).
# Every log records the GPU (arch) and the carve-out (the 8060S line of --list-devices). Server logs are
# <Work>\server_<cell>.log (release_gate's srv_* scan reads them).
# Last line: "env matrix: ok (...)", "env matrix: SKIP (...)" (one GPU, or a visibility variable already set) or
# "env matrix: FAIL (...)". Exit 0 / 0 / 1; setup 2.
param([string]$Build, [string]$Model, [ValidateSet("r9700", "8060s")][string]$Device = "r9700", [string[]]$Cells = @("E1", "E3", "E4"),
      [int]$Port = 8110, [string]$Work = "", [switch]$Golden, [string]$MoeModel = "")
$ErrorActionPreference = "Continue"
$Cells = @($Cells | ForEach-Object { $_ -split "," } | Where-Object { $_ } | ForEach-Object { $_.ToUpper() })
$cli = Join-Path $Build "whirl.exe"
$srv = Join-Path $Build "whirl-server.exe"
$gate = Join-Path $Build "whirl-server-gate.exe"
$hog = Join-Path $Build "whirl-mem-hog.exe"
$replay = Join-Path $Build "whirl-startup-replay.exe"
$sl = Join-Path $PSScriptRoot "startup_lines.py"
$py = if ($env:WHIRL_PYTHON) { $env:WHIRL_PYTHON } else { "C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe" }
if (-not (Test-Path $py)) { $py = "python" }
foreach ($f in $cli, $srv, $Model) { if (-not (Test-Path $f)) { Write-Host "env matrix: FAIL (setup: $f missing)"; exit 2 } }
if (($Cells -contains "E1" -or $Cells -contains "E4") -and -not (Test-Path $hog)) { Write-Host "env matrix: FAIL (setup: $hog missing)"; exit 2 }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-env-matrix-$Device" }
New-Item -ItemType Directory -Force $Work | Out-Null
foreach ($v in "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL") {
    if (Test-Path "env:$v") { Write-Host "env matrix: SKIP ($v is set; the matrix sets it itself)"; exit 0 }
}
foreach ($v in "WHIRL_DEVICE", "WHIRL_HIP_DEVICE", "WHIRL_KV_RAM_MB", "WHIRL_VRAM_LIMIT_MB", "WHIRL_FAKE_DEDICATED_GB", "WHIRL_CKPT_HOST") {
    if (Test-Path "env:$v") { Remove-Item "env:$v" }
}
if (-not $env:AMD_LOG_LEVEL) { $env:AMD_LOG_LEVEL = "1" }

# set or restore one environment variable for the child processes; $null removes it. PowerShell turns a $null
# argument of [Environment]::SetEnvironmentVariable into "", which leaves the variable defined as empty: the 2026-10-24
# S14 GPU run restored HIP_VISIBLE_DEVICES that way and the later "nothing set" cells saw no GPU (env_matrix_selftest.ps1)
function SetEnv([string]$k, $v) {
    if ($null -eq $v) { Remove-Item "env:$k" -ErrorAction SilentlyContinue } else { [Environment]::SetEnvironmentVariable($k, "$v") }
}
function Devs([hashtable]$envs) {
    $saved = @{}
    foreach ($k in $envs.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); SetEnv $k $envs[$k] }
    $o = @(& $cli --list-devices 2>&1 | ForEach-Object { "$_" })
    foreach ($k in $saved.Keys) { SetEnv $k $saved[$k] }
    return $o
}
$list = Devs @{}
$devs = @($list | Where-Object { $_ -match '^\s+device \d+: ' })
$list | ForEach-Object { Write-Host "  $_" }
if ($devs.Count -lt 2) { Write-Host "env matrix: SKIP (this host has $($devs.Count) GPU; the matrix needs two)"; exit 0 }
$arch = if ($Device -eq "8060s") { "gfx1151" } else { "gfx1201" }
$oarch = if ($Device -eq "8060s") { "gfx1201" } else { "gfx1151" }
$mine = @($devs | Where-Object { $_ -match "\($arch\)" })
$oth = @($devs | Where-Object { $_ -match "\($oarch\)" })
if ($mine.Count -ne 1 -or $oth.Count -ne 1) { Write-Host "env matrix: FAIL (expected one $arch and one $oarch GPU, found $($mine.Count) / $($oth.Count))"; exit 1 }
function Ord([string]$l) { if ($l -match '^\s+device (\d+): ') { [int]$Matches[1] } else { -1 } }
function Pci([string]$l) { if ($l -match 'pci:(\d+:\d+) ') { "pci:$($Matches[1])" } else { "" } }
$myOrd = Ord $mine[0]; $myPci = Pci $mine[0]; $oOrd = Ord $oth[0]; $oPci = Pci $oth[0]
$isDefault = $mine[0] -match '\[default\]'
$s8line = if ($Device -eq "8060s") { $mine[0] } else { $oth[0] }
$carve = if ($s8line -match 'dedicated ([\d.]+) GiB') { "$($Matches[1]) GiB dedicated" } else { "unknown" }
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Write-Host "env matrix: GPU $Device ($arch) ordinal $myOrd $myPci$(if ($isDefault) { ' [default]' }); other $oarch ordinal $oOrd $oPci; 8060S carve-out $carve; OS-visible RAM $ram GiB; cells $($Cells -join ',')"

$results = [System.Collections.Generic.List[string]]::new()
function Note([string]$cell, [bool]$ok, [string]$what) {
    $results.Add("$cell $(if ($ok) { 'PASS' } else { 'FAIL' })")
    Write-Host "cell ${cell}: $(if ($ok) { 'PASS' } else { 'FAIL' }) $what"
}
function LogVal([string[]]$txt, [string]$rx) { $m = @($txt | Select-String -Pattern $rx) | Select-Object -Last 1; if ($m) { $m.Matches[0].Groups[1].Value } else { "" } }
# one server: start, wait for /health (or for its exit when a refusal is expected), one request, kill by PID
function Srv([string]$tag, [hashtable]$envs, [string[]]$xargs, [bool]$expectUp = $true) {
    $log = Join-Path $Work "server_$tag.log"
    $so = Join-Path $Work "stdout_$tag.txt"; $se = Join-Path $Work "stderr_$tag.txt"
    Remove-Item $log, $so, $se -ErrorAction SilentlyContinue
    $a = @($Model, "--port", "$Port", "--log-file", $log, "--kv-ssd-mode", "off", "--kv-ram-mb", "4096") + $xargs
    $saved = @{}
    foreach ($k in $envs.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); SetEnv $k $envs[$k] }
    Write-Host "  [$tag] $(($envs.Keys | ForEach-Object { "$_=$($envs[$_])" }) -join ' ') whirl-server $($a -join ' ')"
    $p = Start-Process -FilePath $srv -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $so -RedirectStandardError $se
    $null = $p.Handle  # keeps ExitCode readable after a -PassThru start
    foreach ($k in $saved.Keys) { SetEnv $k $saved[$k] }
    $r = [ordered]@{ tag = $tag; up = $false; http = 0; exit = $null; arch = ""; hip = 0; log = $log; text = ""; room = ""; shared = ""; pinnedHost = $false }
    $limit = if ($expectUp) { 420 } else { 180 }
    for ($i = 0; $i -lt $limit -and -not $p.HasExited; $i++) {
        Start-Sleep 1
        if ($expectUp) { try { if (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2) { $r.up = $true; break } } catch {} }
    }
    if ($r.up) {
        $body = @{ model = "x"; messages = @(@{ role = "user"; content = "用一句話說明 Python 的 list 和 tuple 差在哪。" }); max_tokens = 24; temperature = 0 } | ConvertTo-Json -Depth 5
        try {
            $w = Invoke-WebRequest "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 240 -UseBasicParsing
            $r.http = [int]$w.StatusCode
        } catch { if ($_.Exception.Response) { $r.http = [int]$_.Exception.Response.StatusCode } }
    }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit() } else { $r.exit = $p.ExitCode }
    $txt = @(@($log, $so, $se) | Where-Object { Test-Path $_ } | ForEach-Object { Get-Content $_ })
    $r.text = $txt -join "`n"
    $r.arch = LogVal $txt 'device \d+: .*\((gfx\d+)\)'
    $r.hip = @($txt | Select-String -SimpleMatch "(hipError").Count + @($txt | Select-String -SimpleMatch "| failed:").Count
    $r.room = LogVal $txt '-> room for the RAM tier (\d+) MiB'
    $r.shared = LogVal $txt 'startup inputs:.* pin\.g0\.adapter_shared=(\d+)'
    $r.pinnedHost = [bool](@($txt | Select-String -SimpleMatch "(in pinned host memory)").Count)
    Write-Host "  [$tag] up=$($r.up) http=$($r.http) exit=$($r.exit) arch=$($r.arch) hip/failed lines=$($r.hip) room=$($r.room) MiB pin.g0.adapter_shared=$($r.shared) ckpt_host=$($r.pinnedHost)"
    @($txt | Select-String -Pattern " I GPU|device \d+: |adjusted: |kv tier: RAM tier|pinned memory now|carve-out|VRAM limit|KV pool") | Select-Object -First 12 | ForEach-Object { Write-Host "    log: $($_.Line)" }
    return [pscustomobject]$r
}
function Up([object]$r, [string]$wantArch) { $r.up -and $r.http -eq 200 -and $r.hip -eq 0 -and $r.arch -eq $wantArch }
function Refused([object]$r) { (-not $r.up) -and $null -ne $r.exit -and $r.exit -ne 0 -and $r.hip -eq 0 }
function PSize([string]$log) {
    if (-not (Test-Path $replay)) { return "no replay exe" }
    if (-not (Select-String -Path $log -SimpleMatch "startup inputs:" -Quiet)) { return "no startup inputs line" }
    $o = @(& $py -I $sl check $log --replay $replay 2>&1 | ForEach-Object { "$_" }); $c = $LASTEXITCODE
    $o | Add-Content (Join-Path $Work "psize.txt")
    if ($c -eq 0) { "PASS" } else { "FAIL(exit $c)" }
}
function StartHog([string]$tag, [int]$vram, [int]$pinned) {
    $ready = Join-Path $Work "hog_$tag.ready"
    Remove-Item $ready -ErrorAction SilentlyContinue
    $saved = [Environment]::GetEnvironmentVariable("HIP_VISIBLE_DEVICES")
    SetEnv "HIP_VISIBLE_DEVICES" "$myOrd"  # the hog sees (and pins on) this GPU only
    $h = Start-Process -FilePath $hog -ArgumentList @("--device", "0", "--vram-mb", "$vram", "--pinned-mb", "$pinned", "--seconds", "900", "--ready", $ready) `
        -NoNewWindow -PassThru -RedirectStandardOutput (Join-Path $Work "hog_$tag.txt") -RedirectStandardError (Join-Path $Work "hog_${tag}_err.txt")
    $null = $h.Handle
    SetEnv "HIP_VISIBLE_DEVICES" $saved
    for ($i = 0; $i -lt 120 -and -not (Test-Path $ready) -and -not $h.HasExited; $i++) { Start-Sleep 1 }
    Start-Sleep 3
    Write-Host "  [hog $tag] pid $($h.Id) $(Get-Content (Join-Path $Work "hog_$tag.txt") -ErrorAction SilentlyContinue)"
    if (-not (Test-Path $ready)) { if (-not $h.HasExited) { Stop-Process -Id $h.Id -Force }; return $null }
    return $h
}
function StopHog($h) { if ($h -and -not $h.HasExited) { Stop-Process -Id $h.Id -Force; $h.WaitForExit() } }

if ($Cells -contains "E1" -or $Cells -contains "E3") {
    # --list-devices under each visibility setting
    $nAll = @((Devs @{}) | Where-Object { $_ -match '^\s+device \d+: ' }).Count
    $nMine = @((Devs @{ HIP_VISIBLE_DEVICES = "$myOrd" }) | Where-Object { $_ -match '^\s+device \d+: ' })
    $nOth = @((Devs @{ HIP_VISIBLE_DEVICES = "$oOrd" }) | Where-Object { $_ -match '^\s+device \d+: ' })
    $okList = $nAll -eq 2 -and $nMine.Count -eq 1 -and $nMine[0] -match "\($arch\)" -and $nOth.Count -eq 1 -and $nOth[0] -match "\($oarch\)"
    Note "list" $okList "--list-devices: none set $nAll GPU(s); HIP_VISIBLE_DEVICES=$myOrd -> $($nMine.Count) ($(($nMine -join ' ').Trim())); =$oOrd -> $($nOth.Count)"
    $c = Srv "e1_vis_mine_default" @{ HIP_VISIBLE_DEVICES = "$myOrd" } @()
    $cOk = Up $c $arch
    if ($Cells -contains "E1") {
        $a = Srv "e1_unset_pci" @{} @("--device", $myPci)
        Note "E1a" (Up $a $arch) "nothing set, --device $myPci"
        if ($isDefault) { $b = Srv "e1_unset_default" @{} @(); Note "E1b" (Up $b $arch) "nothing set, no --device (this GPU is the default)" }
        else { Write-Host "cell E1b: n/a (this GPU is not the default; the default GPU's gate runs it)" }
        Note "E1c" $cOk "HIP_VISIBLE_DEVICES=$myOrd, no --device"
        $d = Srv "e1_vis_mine_pci" @{ HIP_VISIBLE_DEVICES = "$myOrd" } @("--device", $myPci)
        Note "E1d" (Up $d $arch) "HIP_VISIBLE_DEVICES=$myOrd, --device $myPci"
        $e = Srv "e1_vis_other_pci" @{ HIP_VISIBLE_DEVICES = "$oOrd" } @("--device", $myPci) $false
        $eOk = (Refused $e) -and $e.text -match "GPUs in this computer" -and $e.text -match [regex]::Escape($myPci)
        Note "E1e" $eOk "HIP_VISIBLE_DEVICES=$oOrd (this GPU hidden), --device ${myPci}: refused (exit $($e.exit)) with the GPU list"
        # (f) another program holds pinned memory
        $base = $d
        $h = StartHog "e1f" 0 4096
        if (-not $h) { Note "E1f" $false "whirl-mem-hog did not get ready" }
        else {
            $f = Srv "e1_pinned_hog" @{ HIP_VISIBLE_DEVICES = "$myOrd" } @("--device", $myPci)
            StopHog $h
            $ps0 = PSize $base.log; $ps1 = PSize $f.log
            $grow = if ($base.shared -and $f.shared) { ([int64]$f.shared - [int64]$base.shared) / 1MB } else { -1 }
            $drop = if ($base.room -and $f.room) { [int64]$base.room - [int64]$f.room } else { -1 }
            $fOk = (Up $f $arch) -and $grow -ge 3584 -and ($drop -ge 3584 -or ($f.room -eq "0")) -and $ps0 -eq "PASS" -and $ps1 -eq "PASS"
            Note "E1f" $fOk "pinned hog 4096 MiB: adapter_shared +$([math]::Round($grow)) MiB, room $($base.room) -> $($f.room) MiB (drop $drop), P-size $ps0 / $ps1"
        }
    }
    if ($Cells -contains "E3") {
        Note "E3a" $cOk "$oarch hidden (HIP_VISIBLE_DEVICES=$myOrd): the default is this GPU ($($c.arch))"
        $g = Srv "e3_other_missing" @{ HIP_VISIBLE_DEVICES = "$myOrd" } @("--device", $oPci) $false
        $gOk = (Refused $g) -and $g.text -match "GPUs in this computer" -and $g.text -match "\($arch\)"
        Note "E3b" $gOk "$oarch hidden, --device ${oPci}: refused (exit $($g.exit)), lists only the $arch GPU"
    }
}
if ($Cells -contains "E4") {
    $h = StartHog "e4" 6144 2048
    if (-not $h -or -not (Test-Path $gate)) { Note "E4" $false "whirl-mem-hog not ready or whirl-server-gate missing"; StopHog $h }
    else {
        $gw = Join-Path $Work "e4_gate"
        $o = @(& $gate --exe $srv --kind whirl --model $Model --suite basic --port "$Port" --work $gw --kv auto --server-env "WHIRL_DEVICE=$Device" --server-env "WHIRL_KV_SSD_MODE=off" 2>&1 | ForEach-Object { "$_" })
        $gc = $LASTEXITCODE
        StopHog $h
        $o | Set-Content (Join-Path $Work "e4_gate.txt")
        $gl = @(Get-ChildItem $gw -Recurse -File -Filter "server_*.log" -ErrorAction SilentlyContinue)
        foreach ($l in $gl) { Copy-Item $l.FullName (Join-Path $Work "server_e4_$($l.Name -replace '^server_','')") -Force }
        $scan = @(& (Join-Path $PSScriptRoot "server_log_scan.ps1") -Dir $gw -Patterns wddm,owner *>&1 | ForEach-Object { "$_" }); $sc = $LASTEXITCODE
        $archOk = $gl.Count -gt 0 -and @($gl | Select-String -SimpleMatch "($arch)" -List).Count -eq $gl.Count
        $ps = @($gl | Where-Object { Select-String -Path $_.FullName -SimpleMatch "startup inputs:" -Quiet } | ForEach-Object { PSize $_.FullName })
        $okG = $gc -eq 0 -and ($o -match "== whirl-server-gate: [1-9]\d* passed, 0 failed \(ALL PASS\)")
        Note "E4" ($okG -and $sc -eq 0 -and $archOk -and @($ps | Where-Object { $_ -ne "PASS" }).Count -eq 0) "6144 MiB VRAM + 2048 MiB pinned held by another program: server gate basic exit $gc; $($scan[0]); arch ok $archOk; P-size $($ps -join ',')"
    }
}
if ($Cells -contains "E2") {
    if ($Device -ne "8060s") { Write-Host "cell E2: n/a (8060S only)" }
    else {
        foreach ($mb in 49152, 32768, 24576) {
            $r = Srv "e2_limit_$mb" @{ WHIRL_VRAM_LIMIT_MB = "$mb" } @("--device", $myPci) $true
            $named = $r.text -match "WHIRL_VRAM_LIMIT_MB|simulated limit"
            Note "E2_$mb" (((Up $r $arch) -or ((Refused $r) -and $r.exit -eq 5)) -and $named) "WHIRL_VRAM_LIMIT_MB=$mb : $(if ($r.up) { "answered HTTP $($r.http)" } else { "refused exit $($r.exit)" }), limit named in the log $named"
        }
        $z = Srv "e2_fake_0_5" @{ WHIRL_FAKE_DEDICATED_GB = "0.5" } @("--device", $myPci) $false
        Note "E2_fake0.5" ((Refused $z) -and $z.exit -eq 5 -and $z.text -match "Armoury Crate" -and $z.text -match "at least \d+ GB") "WHIRL_FAKE_DEDICATED_GB=0.5: exit $($z.exit), names the setting and the size"
    }
}
if ($Cells -contains "E7") {
    $gs = Join-Path $PSScriptRoot "gpu_select_check.ps1"
    SetEnv "WHIRL_CKPT_HOST" "1"
    $o = @(& pwsh -NoProfile -File $gs -Build $Build -Model $Model -Device $Device -Port ($Port + 1) -Work (Join-Path $Work "e7_gpusel") 2>&1 | ForEach-Object { "$_" })
    $gsc = $LASTEXITCODE
    SetEnv "WHIRL_CKPT_HOST" $null
    $o | Set-Content (Join-Path $Work "e7_gpusel.txt")
    $sel = @(Get-ChildItem (Join-Path $Work "e7_gpusel") -Recurse -File -Filter "server.log" -ErrorAction SilentlyContinue)
    foreach ($l in $sel) { Copy-Item $l.FullName (Join-Path $Work "server_e7_gpusel_$($l.Directory.Name).log") -Force }
    $hostOk = $sel.Count -gt 0 -and @($sel | Select-String -SimpleMatch "(in pinned host memory)" -List).Count -eq $sel.Count
    Note "E7_gpu_select" ($gsc -eq 0 -and ($o -match "^gpu select check: ok") -and $hostOk) "WHIRL_CKPT_HOST=1 gpu_select_check exit $gsc ($(@($o | Select-Object -Last 1))); logs naming '(in pinned host memory)' $(@($sel | Select-String -SimpleMatch '(in pinned host memory)' -List).Count)/$($sel.Count)"
    if ($Golden) {
        $goldenPs = Join-Path $PSScriptRoot "..\golden\golden.ps1"
        $gout = Join-Path $Work "e7_golden"
        $o = @(& pwsh -NoProfile -File $goldenPs check -Arch $arch -Suites server -Models swift,moe -Modes precise,balance,fast -Exe $cli -ServerExe $srv -GateExe $gate `
                -Out $gout -KeepServerWork -Port ($Port + 2) -ServerEnv "WHIRL_CKPT_HOST=1" 2>&1 | ForEach-Object { "$_" })
        $gc = $LASTEXITCODE
        $o | Set-Content (Join-Path $Work "e7_golden.txt")
        $gl = @(Get-ChildItem $gout -Recurse -File -Filter "server_*.log" -ErrorAction SilentlyContinue)
        foreach ($l in $gl) { Copy-Item $l.FullName (Join-Path $Work "server_e7_golden_$($l.Directory.Name)_$($l.Name -replace '^server_','')") -Force }
        $nh = @($gl | Select-String -SimpleMatch "(in pinned host memory)" -List).Count
        Note "E7_golden" ($gc -eq 0 -and $gl.Count -gt 0 -and $nh -eq $gl.Count) "WHIRL_CKPT_HOST=1 golden server swift,moe x3 modes: exit $gc ($(@($o | Select-Object -Last 2) -join ' | ')); logs naming '(in pinned host memory)' $nh/$($gl.Count)"
    } else { Write-Host "cell E7_golden: not run (-Golden not given)" }
}
Get-ChildItem $Work -Recurse -Filter *.wkv -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
$bad = @($results | Where-Object { $_ -like "* FAIL" })
if ($bad.Count) { Write-Host "env matrix: FAIL ($($bad -join ', ')) [$Device $arch, carve-out $carve]"; exit 1 }
Write-Host "env matrix: ok ($($results.Count) cell(s): $($results -join ', ')) [$Device $arch, carve-out $carve]"
exit 0
