# SPDX-License-Identifier: Apache-2.0
# env_matrix.ps1 cell E1f judgment (arch025-e1f), dot-sourced by env_matrix.ps1; it reads only two server logs' text, so
# it also runs on saved logs:  . tools/gates/env_matrix_e1f.ps1; E1fJudge (Get-Content base.log) (Get-Content hog.log)
#
# Before: E1f required the "room for the RAM tier" to drop by >= 3584 MiB. That assumed the tested GPU's adapter has the
# largest GPU Shared Usage, but WHIRL's machine term uses the LARGEST adapter (startfit::pinnedRoomMb,
# src/common/startup_fit.cpp: half the RAM - pin.pinned_max - later - 2048, whole 512 MiB). On 2026-11-01 the 8060S
# adapter (display, desktop programs) held 3460 MiB while the R9700 held 1416, so the 4096 MiB hog on the R9700 moved
# the largest figure by 2058 MiB and the room by 2048: a false FAIL.
# Now: (1) the hog is seen on this adapter: pin.g0.adapter_shared grows by >= 3584 MiB; (2) the largest-adapter figure
# covers this adapter in both logs (pin.pinned_max + 64 MiB >= pin.g0.adapter_shared: the two figures come from two
# separate PDH queries, src/hip/wddm_mem.cpp, so another program's pin in between may move one of them; a stale largest
# figure misses a 4096 MiB hog by far more); (3) each log's room equals the rule
# applied to that log's own startup inputs (mirror of pinnedRoomMb below; P-size replays the C++ rule itself); so the
# drop equals what the rule says, whichever adapter is the largest; (4) both logs list one GPU (pin.gpus=1), so pin.g0 is
# this GPU in both (arch025-e1ff, review arch025-e1fr C1 / C4).
function E1fPin([string[]]$txt) {
    $line = @($txt | Where-Object { $_ -match 'startup inputs:.* pin\.measured=' }) | Select-Object -Last 1
    $r = [ordered]@{ ok = $false; room = $null; total = 0; pmax = 0; later = 0; gpus = @() }
    $rm = @($txt | Select-String -Pattern '-> room for the RAM tier (\d+) MiB') | Select-Object -Last 1
    if ($rm) { $r.room = [int64]$rm.Matches[0].Groups[1].Value }
    if (-not $line) { return [pscustomobject]$r }
    $kv = @{}
    foreach ($m in [regex]::Matches($line, '(pin\.[\w.]+)=(\d+)')) { $kv[$m.Groups[1].Value] = [uint64]$m.Groups[2].Value }
    if (-not $kv.ContainsKey("pin.total_phys") -or -not $kv.ContainsKey("pin.pinned_max") -or -not $kv.ContainsKey("pin.gpus")) { return [pscustomobject]$r }
    $r.total = $kv["pin.total_phys"]; $r.pmax = $kv["pin.pinned_max"]; $r.later = [uint64]$kv["pin.later"]
    $r.gpus = @(for ($i = 0; $i -lt [int]$kv["pin.gpus"]; $i++) {
        [pscustomobject]@{ budget = [uint64]$kv["pin.g$i.nonlocal_budget"]; own = [uint64]$kv["pin.g$i.own_nonlocal"]
                           sys = [uint64]$kv["pin.g$i.shared_system"]; shared = [uint64]$kv["pin.g$i.adapter_shared"] } })
    $r.ok = $r.gpus.Count -gt 0
    return [pscustomobject]$r
}
# mirror of startfit::pinnedRoomMb (src/common/startup_fit.cpp, reserves startup_fit.h pinned_reserve_mb /
# pinned_machine_reserve_mb = 2048 / 2048); -1 = unknown (nothing measured)
function E1fRuleRoom([object]$p) {
    $MiB = [uint64]1048576
    $room = [int64]-1
    $take = { param([uint64]$limit, [uint64]$used, [uint64]$res)
        $free = if ($limit -gt $used) { [uint64][math]::Floor(($limit - $used) / $MiB) } else { [uint64]0 }
        $v = if ($free -gt $res) { [int64]([math]::Floor(($free - $res) / 512) * 512) } else { [int64]0 }
        if ($script:e1fRoom -lt 0 -or $v -lt $script:e1fRoom) { $script:e1fRoom = $v } }
    $script:e1fRoom = $room
    $n = [uint64][math]::Max($p.gpus.Count, 1)
    foreach ($g in $p.gpus) {
        $used = [uint64][math]::Max($g.own, $g.shared) + $p.later
        if ($g.budget -gt 0) { & $take ([uint64][math]::Floor($g.budget / $n)) $used 2048 }
        elseif ($g.sys -gt 0) { & $take $g.sys $used 2048 }
    }
    if ($p.total -gt 0) { & $take ([uint64][math]::Floor($p.total / 2)) ($p.pmax + $p.later) 2048 }
    return $script:e1fRoom
}
# returns ok + the cell text; $up / $ps0 / $ps1 = the server came up on this GPU, P-size of the two logs
function E1fJudge([string[]]$baseTxt, [string[]]$hogTxt, [bool]$up = $true, [string]$ps0 = "PASS", [string]$ps1 = "PASS") {
    $b = E1fPin $baseTxt; $f = E1fPin $hogTxt
    if (-not $b.ok -or -not $f.ok -or $null -eq $b.room -or $null -eq $f.room) {
        return [pscustomobject]@{ ok = $false; text = "pinned hog 4096 MiB: no startup inputs / room line in a log (base ok $($b.ok) room $($b.room), hog ok $($f.ok) room $($f.room))" }
    }
    $MiB = 1048576
    $grow = ([double]$f.gpus[0].shared - [double]$b.gpus[0].shared) / $MiB
    $tol = 64 * $MiB   # PDH jitter between the two queries (review arch025-e1fr C1)
    $cover = ($b.pmax + $tol) -ge $b.gpus[0].shared -and ($f.pmax + $tol) -ge $f.gpus[0].shared
    $one = $b.gpus.Count -eq 1 -and $f.gpus.Count -eq 1   # pin.g0 = this GPU in both logs (review arch025-e1fr C4)
    $e0 = E1fRuleRoom $b; $e1 = E1fRuleRoom $f
    $drop = $b.room - $f.room; $want = $e0 - $e1
    $ruleOk = $b.room -eq $e0 -and $f.room -eq $e1
    $ok = $up -and $one -and $grow -ge 3584 -and $cover -and $ruleOk -and $ps0 -eq "PASS" -and $ps1 -eq "PASS"
    $t = "pinned hog 4096 MiB: pin.gpus $($b.gpus.Count) / $($f.gpus.Count), adapter_shared +$([math]::Round($grow)) MiB, largest adapter $([math]::Round($b.pmax / $MiB)) -> $([math]::Round($f.pmax / $MiB)) MiB" +
         " (covers this adapter within 64 MiB: $cover), room $($b.room) -> $($f.room) MiB (drop $drop; the rule on the logged inputs: $e0 -> $e1, drop $want), P-size $ps0 / $ps1"
    return [pscustomobject]@{ ok = $ok; text = $t }
}
