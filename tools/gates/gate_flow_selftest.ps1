# SPDX-License-Identifier: Apache-2.0
# CPU-only self-test of the release gate's step runner (arch025-procopt, PLAN 9.20 / 9.21): the machine-wide heavy-CPU
# lock and the retry-once / FLAKY rule. Runs tools/gates/release_gate.ps1 -TestSteps on a fake build folder (empty *.exe
# files, fake model files; no exe, no model, no GPU), with a TEST lock file and a TEST flaky.tsv in a temp folder (never
# progress\state\cpu_heavy.lock / flaky.tsv). Called by gate_steps_selftest.ps1 (gate step cpu_step_selftest).
#   pwsh tools/gates/gate_flow_selftest.ps1      last line "gate flow selftest: N/M as expected"; exit 0 all, 1 not
# Cases:
#   R1 dev run, a CPU step that fails once then passes: row "FLAKY <step> exit 0 ...| first try: exit 3 ...", the first
#      try's log kept as <step>.try1.txt, verdict "FLAKY (...)" (never "PASS (N steps)"), FLAKY_<tag> marker (no
#      PASS_<tag>), exit 0, one flaky.tsv row (date, tag, step, first-failure marker, log path)
#   R2 -Strict: the same step is FAIL (one attempt, no .try1, no flaky.tsv row), exit 1
#   R3 the default full-gate tag gate_<device> is strict without -Strict
#   R4 dev run, a step that fails twice: FAIL row "retry also failed", exit 1, no flaky.tsv row
#   R5 a GPU step is never retried (a failure is FAIL)
#   L1 a heavy step waits while another process holds the lock; the wait is on the row ("cpu lock: waited N s") and
#      not in the step's own seconds; the step starts after the holder let go; a light step does not wait
#   L2 lock timeout: logged, the step runs without the lock, the holder is not killed
#   L3 a holder that was killed leaves its "held pid N" text, but the lock is free: no wait
#   L4 WHIRL_CPU_LOCK_HELD set (an outer runner holds the lock for this process tree): no wait (no self-deadlock)
#   L5 the gate's held lock reads "held pid <gate pid> ... <tag>/<step>" while the heavy step runs
param([string]$Work = "")
# arch025-c2fu: set or REMOVE a process environment variable (see release_gate.ps1).
function SetEnvOrRemove([string]$k, $v) {
    if ($null -eq $v) { Remove-Item -LiteralPath "env:$k" -ErrorAction SilentlyContinue } else { [Environment]::SetEnvironmentVariable($k, "$v") }
}
$ErrorActionPreference = "Stop"
$gate = Join-Path $PSScriptRoot "release_gate.ps1"
$pwsh = (Get-Process -Id $PID).Path
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-gate-flow-selftest-" + [guid]::NewGuid().ToString("N").Substring(0, 8)) }
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Work | Out-Null
$bin = Join-Path $Work "bin"
New-Item -ItemType Directory -Force $bin | Out-Null
foreach ($n in "whirl", "whirl-server", "whirl-server-gate", "whirl-kernel-test", "whirl-tests", "whirl-model-tests", "whirl-cache-tests",
               "whirl-vision-tests", "whirl-tier-tests", "whirl-server-tests", "whirl-hiprt-tests", "whirl-device-tests", "whirl-ops-tests",
               "whirl-pins", "whirl-startup-replay", "whirl-sampling-tests") { [System.IO.File]::WriteAllText((Join-Path $bin "$n.exe"), "fake $n") }
$sw = Join-Path $Work "Swift-fake.gguf"; [System.IO.File]::WriteAllText($sw, "swift")
$mx = Join-Path $Work "Ornith-fake.gguf"; [System.IO.File]::WriteAllText($mx, "moe")
$lockF = Join-Path $Work "cpu_heavy.lock"
$tsv = Join-Path $Work "flaky.tsv"
$keys = @("WHIRL_GATE_SWIFT", "WHIRL_GATE_MOE", "WHIRL_GATE_TEXT_ROOT", "WHIRL_GATE_DENSE", "WHIRL_GATE_MX", "WHIRL_GATE_LOCK_SCRIPT", "WHIRL_GATE_CPU_LOCK",
          "WHIRL_GATE_CPU_LOCK_TIMEOUT_S", "WHIRL_GATE_CPU_LOCK_POLL_S", "WHIRL_GATE_FLAKY_TSV", "WHIRL_GATE_STRICT", "WHIRL_CPU_LOCK_HELD")
$saved = @{}
foreach ($k in $keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); SetEnvOrRemove $k $null }
$env:WHIRL_GATE_SWIFT = $sw; $env:WHIRL_GATE_MOE = $mx; $env:WHIRL_GATE_TEXT_ROOT = $Work
$env:WHIRL_GATE_CPU_LOCK = $lockF; $env:WHIRL_GATE_FLAKY_TSV = $tsv; $env:WHIRL_GATE_CPU_LOCK_POLL_S = "1"
$ok = 0; $bad = 0
function Check([string]$name, [bool]$cond, [string]$detail) {
    if ($cond) { $script:ok++; Write-Host "ok   $name" } else { $script:bad++; Write-Host "BAD  $name`: $detail" }
}
# a step command: pwsh -Command <body>; @CNT@ = a per-case counter file, @T@ = a per-case timestamp file
function StepCmd([string]$body) { @($pwsh, "-NoProfile", "-Command", $body) }
# fails (exit 3, last line "boom first try") on its first run, passes ("flaky step: ok") on the next
function FlakyBody([string]$cnt) { "`$f = '$cnt'; `$n = if (Test-Path `$f) { [int](Get-Content `$f) } else { 0 }; `$n + 1 | Set-Content `$f; if (`$n -eq 0) { 'boom first try'; exit 3 }; 'flaky step: ok'; exit 0" }
function Steps([string]$name, [object[]]$list) {
    $p = Join-Path $Work "$name.steps.json"
    ConvertTo-Json -InputObject @($list) -Depth 5 | Set-Content $p
    $p
}
function Gate([string]$tag, [string]$stepsFile, [string[]]$extra = @()) {
    $w = Join-Path $Work "work_$tag"
    $o = @(& $pwsh -NoProfile -File $gate -Device r9700 -Build $bin -TestSteps $stepsFile -Work $w -Tag $tag @extra 2>&1 | ForEach-Object { "$_" })
    $c = $LASTEXITCODE
    $sf = Join-Path $w "${tag}_summary.txt"
    $s = if (Test-Path $sf) { @(Get-Content $sf) } else { @() }
    [pscustomobject]@{ rc = $c; out = $o; sum = $s; work = $w; logs = (Join-Path $w $tag) }
}
function Last($r) { (@($r.out) + @($r.sum) | Select-Object -Last 8) -join " | " }
function TsvRows { if (Test-Path $tsv) { @(Get-Content $tsv | Select-Object -Skip 1) } else { @() } }
# a holder of the TEST lock with the gpu_lock.ps1 protocol (handle open with FileShare.Read, text "held pid N <ts> <what>");
# writes <lock>.released_<tag> (ticks) right before it lets go
function Start-Holder([string]$tag, [double]$holdS, [string]$goFile = "") {
    # with -goFile: hold until that file exists (max 120 s), then $holdS more (timing-independent on a loaded machine)
    $goWait = if ($goFile) { "for (`$i = 0; `$i -lt 1200 -and -not (Test-Path '$goFile'); `$i++) { Start-Sleep -Milliseconds 100 }" } else { "" }
    $s = Join-Path $Work "holder_$tag.ps1"
    @(
        "`$fs = [System.IO.File]::Open('$lockF', [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::Read)",
        "`$b = [System.Text.Encoding]::ASCII.GetBytes(`"held pid `$PID `$(Get-Date -Format s) selftest holder $tag``n`")",
        "`$fs.SetLength(0); `$fs.Write(`$b, 0, `$b.Length); `$fs.Flush()",
        "'up' | Set-Content '$Work\holder_$tag.up'",
        $goWait,
        "Start-Sleep -Milliseconds $([int]($holdS * 1000))",
        "[DateTime]::UtcNow.Ticks | Set-Content '$Work\holder_$tag.released'",
        "`$fs.Close()"
    ) | Set-Content $s
    $p = Start-Process $pwsh -ArgumentList "-NoProfile", "-File", $s -PassThru -WindowStyle Hidden
    for ($i = 0; $i -lt 200 -and -not (Test-Path "$Work\holder_$tag.up"); $i++) { Start-Sleep -Milliseconds 100 }
    $p
}
$started = @()
try {
    $dummy = @{ name = "test_light"; heavy = $false; gpu = $false; cmd = (StepCmd "'light step: ok'"); rx = "^light step: ok" }

    # ---- R1 dev run: FLAKY
    $sf = Steps "r1" @($dummy, @{ name = "test_flaky"; heavy = $false; gpu = $false; cmd = (StepCmd (FlakyBody (Join-Path $Work "r1.cnt"))); rx = "^flaky step: ok" })
    $r = Gate "t_r1" $sf
    $row = @($r.sum | Where-Object { $_ -match "^FLAKY test_flaky " })
    Check "R1 dev: exit 0" ($r.rc -eq 0) "exit $($r.rc): $(Last $r)"
    Check "R1 dev: row 'FLAKY test_flaky exit 0 ... | first try: exit 3 ... boom first try'" ($row.Count -eq 1 -and $row[0] -match "^FLAKY test_flaky\s+exit\s+0 .*flaky step: ok.*\| first try: exit\s+3 .*boom first try") "$(Last $r)"
    Check "R1 dev: verdict FLAKY, never 'PASS (N steps)'" ($r.sum.Count -gt 0 -and $r.sum[-1] -match "^release gate t_r1: FLAKY \(2 steps passed, 1 only on a retry: test_flaky\)$" -and -not ($r.sum -match "PASS \(\d+ steps\)")) "$(Last $r)"
    Check "R1 dev: FLAKY_t_r1 marker, no PASS_t_r1" ((Test-Path (Join-Path $r.work "FLAKY_t_r1")) -and -not (Test-Path (Join-Path $r.work "PASS_t_r1"))) "markers: $(@(Get-ChildItem $r.work -File -ErrorAction SilentlyContinue | ForEach-Object Name) -join ',')"
    Check "R1 dev: first try's log kept as test_flaky.try1.txt" ((Test-Path (Join-Path $r.logs "test_flaky.try1.txt")) -and ((Get-Content (Join-Path $r.logs "test_flaky.try1.txt") -ErrorAction SilentlyContinue) -match "boom first try")) "logs: $(@(Get-ChildItem $r.logs -File -ErrorAction SilentlyContinue | ForEach-Object Name) -join ',')"
    $tr = @(TsvRows)
    $hdr = if (Test-Path $tsv) { (Get-Content $tsv -TotalCount 1) } else { "" }
    Check "R1 dev: flaky.tsv header + one row (date, tag, step, first-failure marker, log path)" ($hdr -eq "date`ttag`tstep`tfirst_failure`tlog" -and $tr.Count -eq 1 -and
        ($tr[0] -split "`t").Count -eq 5 -and ($tr[0] -split "`t")[1] -eq "t_r1" -and ($tr[0] -split "`t")[2] -eq "test_flaky" -and ($tr[0] -split "`t")[3] -match "exit 3: boom first try" -and
        ($tr[0] -split "`t")[4] -like "*test_flaky.try1.txt" -and ($tr[0] -split "`t")[0] -match "^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d") "header '$hdr', rows: $($tr -join ' || ')"
    Check "R1 dev: header says the retry policy" ($r.sum.Count -gt 0 -and $r.sum[0] -match "retry: CPU steps once \(dev\)") "$($r.sum | Select-Object -First 1)"

    # ---- R2 -Strict
    $sf = Steps "r2" @(@{ name = "test_flaky"; heavy = $false; gpu = $false; cmd = (StepCmd (FlakyBody (Join-Path $Work "r2.cnt"))); rx = "^flaky step: ok" })
    $r = Gate "t_r2" $sf @("-Strict")
    Check "R2 -Strict: the flaky step is FAIL after one attempt, exit 1, no .try1, no flaky.tsv row" ($r.rc -eq 1 -and ($r.sum -match "^FAIL test_flaky\s+exit\s+3 ") -and
        $r.sum[-1] -match "release gate t_r2: FAIL \(1 of 1 steps failed\)" -and -not (Test-Path (Join-Path $r.logs "test_flaky.try1.txt")) -and @(TsvRows).Count -eq 1 -and
        [int](Get-Content (Join-Path $Work "r2.cnt")) -eq 1 -and $r.sum[0] -match "retry: none \(strict\)") "$(Last $r)"
    # ---- R3 default full-gate tag is strict
    $sf = Steps "r3" @(@{ name = "test_flaky"; heavy = $false; gpu = $false; cmd = (StepCmd (FlakyBody (Join-Path $Work "r3.cnt"))); rx = "^flaky step: ok" })
    $r = Gate "gate_r9700" $sf
    Check "R3 tag gate_r9700 without -Strict: strict (FAIL, one attempt)" ($r.rc -eq 1 -and ($r.sum -match "^FAIL test_flaky ") -and [int](Get-Content (Join-Path $Work "r3.cnt")) -eq 1) "$(Last $r)"
    # ---- R4 fails twice
    $sf = Steps "r4" @(@{ name = "test_bad"; heavy = $false; gpu = $false; cmd = (StepCmd "'always broken'; exit 5"); rx = "^never" })
    $r = Gate "t_r4" $sf
    Check "R4 dev: a step that fails twice is FAIL ('retry also failed'), exit 1, no flaky.tsv row" ($r.rc -eq 1 -and ($r.sum -match "^FAIL test_bad\s+exit\s+5 .*retry also failed") -and @(TsvRows).Count -eq 1) "$(Last $r)"
    # ---- R5 GPU step not retried
    $sf = Steps "r5" @(@{ name = "test_gpu"; heavy = $false; gpu = $true; cmd = (StepCmd (FlakyBody (Join-Path $Work "r5.cnt"))); rx = "^flaky step: ok" })
    $r = Gate "t_r5" $sf
    Check "R5 dev: a GPU step is not retried (FAIL, one attempt)" ($r.rc -eq 1 -and ($r.sum -match "^FAIL test_gpu ") -and [int](Get-Content (Join-Path $Work "r5.cnt")) -eq 1) "$(Last $r)"

    # ---- L1 heavy step waits for a holder; light step does not
    $tf = Join-Path $Work "l1.start"
    $sf = Steps "l1" @(@{ name = "test_light"; heavy = $false; gpu = $false; cmd = (StepCmd "[DateTime]::UtcNow.Ticks | Set-Content '$Work\l1.light'; 'go' | Set-Content '$Work\l1.go'; 'light step: ok'"); rx = "^light step: ok" },
                       @{ name = "test_heavy"; heavy = $true; gpu = $false; cmd = (StepCmd "[DateTime]::UtcNow.Ticks | Set-Content '$tf'; 'heavy step: ok'"); rx = "^heavy step: ok" })
    # releases 10 s after the light step ran: the wait (>= ~8 s) is well above a trivial step's own time on a loaded machine
    $h = Start-Holder "l1" 10 "$Work\l1.go"; $started += $h
    $r = Gate "t_l1" $sf
    $h.WaitForExit(20000) | Out-Null
    $rel = [long](Get-Content "$Work\holder_l1.released" -ErrorAction SilentlyContinue)
    $st = [long](Get-Content $tf -ErrorAction SilentlyContinue)
    $lt = [long](Get-Content "$Work\l1.light" -ErrorAction SilentlyContinue)
    $hrow = @($r.sum | Where-Object { $_ -match "^PASS test_heavy " })
    $wS = if ($hrow.Count -and $hrow[0] -match "cpu lock: waited (\d+) s") { [int]$Matches[1] } else { -1 }
    $sec = if ($hrow.Count -and $hrow[0] -match "^PASS test_heavy\s+exit\s+0\s+(\d+) s") { [int]$Matches[1] } else { -1 }
    Check "L1 heavy step: PASS, row says 'cpu lock: waited N s' (N >= 5)" ($r.rc -eq 0 -and $wS -ge 5) "wait $wS | $(Last $r)"
    Check "L1 heavy step: its own seconds exclude the wait (step s + 3 <= wait s)" ($sec -ge 0 -and $sec + 3 -le $wS) "sec $sec wait $wS"
    Check "L1 heavy step started after the holder let go" ($rel -gt 0 -and $st -gt $rel) "released $rel started $st"
    Check "L1 light step ran while the lock was held (no wait)" ($lt -gt 0 -and $rel -gt 0 -and $lt -lt $rel -and -not (@($r.sum | Where-Object { $_ -match "^PASS test_light " }) -match "cpu lock")) "light $lt released $rel"
    Check "L1 the gate printed a waiting line naming the holder" (@($r.out -match "cpu lock: busy .*selftest holder l1.*waiting").Count -gt 0) "$(Last $r)"

    # ---- L2 timeout: run anyway, holder not killed
    $env:WHIRL_GATE_CPU_LOCK_TIMEOUT_S = "2"
    $sf = Steps "l2" @(@{ name = "test_heavy"; heavy = $true; gpu = $false; cmd = (StepCmd "'heavy step: ok'"); rx = "^heavy step: ok" })
    $h = Start-Holder "l2" 0 "$Work\l2.go"; $started += $h   # holds until l2.go (timing-independent)
    $r = Gate "t_l2" $sf
    $alive = -not $h.HasExited
    "go" | Set-Content "$Work\l2.go"
    SetEnvOrRemove "WHIRL_GATE_CPU_LOCK_TIMEOUT_S" $null
    Check "L2 lock timeout: step runs anyway, row 'cpu lock: timeout after 2 s ... ran without the lock', holder still alive" ($r.rc -eq 0 -and
        ($r.sum -match "^PASS test_heavy .*cpu lock: timeout after \d+ s .*ran without the lock") -and $alive) "alive $alive | $(Last $r)"
    $h.WaitForExit(30000) | Out-Null

    # ---- L3 a killed holder: text stays, lock free
    $h = Start-Holder "l3" 60; $started += $h
    Stop-Process -Id $h.Id -Force; $h.WaitForExit(10000) | Out-Null
    $txt = ""; try { $rd = [System.IO.File]::Open($lockF, "Open", "Read", "ReadWrite"); $sr = New-Object System.IO.StreamReader($rd); $txt = $sr.ReadToEnd(); $sr.Close() } catch { }
    $r = Gate "t_l3" (Steps "l3" @(@{ name = "test_heavy"; heavy = $true; gpu = $false; cmd = (StepCmd "'heavy step: ok'"); rx = "^heavy step: ok" }))
    Check "L3 killed holder ('held pid $($h.Id)' text left): no wait" ($txt -match "^held pid $($h.Id) " -and $r.rc -eq 0 -and ($r.sum -match "^PASS test_heavy ") -and -not ($r.sum -match "cpu lock: (waited|timeout)")) "text '$($txt.Trim())' | $(Last $r)"

    # ---- L4 an outer holder of this process tree
    $h = Start-Holder "l4" 0 "$Work\l4.go"; $started += $h   # holds until l4.go
    $env:WHIRL_CPU_LOCK_HELD = "outer runner (selftest)"
    $env:WHIRL_GATE_CPU_LOCK_TIMEOUT_S = "60"   # a wrong wait would show as 'waited' / 'timeout', not hang the test
    $r = Gate "t_l4" (Steps "l4" @(@{ name = "test_heavy"; heavy = $true; gpu = $false; cmd = (StepCmd "'heavy step: ok'"); rx = "^heavy step: ok" }))
    $alive = -not $h.HasExited
    SetEnvOrRemove "WHIRL_CPU_LOCK_HELD" $null; SetEnvOrRemove "WHIRL_GATE_CPU_LOCK_TIMEOUT_S" $null
    "go" | Set-Content "$Work\l4.go"
    Check "L4 WHIRL_CPU_LOCK_HELD set: no wait while the lock is held (holder alive throughout)" ($alive -and $r.rc -eq 0 -and ($r.sum -match "^PASS test_heavy ") -and -not ($r.sum -match "cpu lock: (waited|timeout)")) "alive $alive | $(Last $r)"
    $h.WaitForExit(30000) | Out-Null

    # ---- L5 lock text while the gate holds it; WHIRL_CPU_LOCK_HELD is passed to the step
    $lf5 = Join-Path $Work "l5.txt"
    $body = "`$rd = [System.IO.File]::Open('$lockF', 'Open', 'Read', 'ReadWrite'); `$sr = New-Object System.IO.StreamReader(`$rd); `$sr.ReadToEnd() + ' | held env: ' + `$env:WHIRL_CPU_LOCK_HELD | Set-Content '$lf5'; `$sr.Close(); 'heavy step: ok'"
    $r = Gate "t_l5" (Steps "l5" @(@{ name = "test_heavy"; heavy = $true; gpu = $false; cmd = (StepCmd $body); rx = "^heavy step: ok" }))
    $t5 = "$(Get-Content $lf5 -Raw -ErrorAction SilentlyContinue)"
    Check "L5 while the heavy step runs the lock reads 'held pid N ... t_l5/test_heavy' and the step sees WHIRL_CPU_LOCK_HELD" ($r.rc -eq 0 -and $t5 -match "^held pid \d+ \S+ release gate t_l5/test_heavy" -and $t5 -match "held env: release gate t_l5/test_heavy") "'$($t5.Trim())' | $(Last $r)"
    Check "L5 after the gate the lock reads 'free ...'" ((Get-Content $lockF -Raw) -match "^free ") "'$(Get-Content $lockF -Raw)'"
} finally {
    foreach ($p in $started) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    foreach ($k in $keys) { SetEnvOrRemove $k $saved[$k] }
    Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
}
Write-Host "gate flow selftest: $ok/$($ok + $bad) as expected"
if ($bad) { exit 1 }
exit 0
