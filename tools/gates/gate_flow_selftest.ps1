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
# 9.24 (arch025-procopt2) parallel sections ("par": true test steps stand for cpu_sampling_iso / cpu_iso_*):
#   P1 -Jobs 4: four par steps overlap in time (each sleeps); the section's wall time < the sum of the sleeps
#   P2 the summary (rows without the seconds column, verdict line), exit code and markers of a -Jobs 4 run equal the
#      -Jobs 1 run's, with par steps finishing in reverse order, a light step before and after, and a par step that
#      fails twice; each step's log has its own output
#   P3 a par step that fails once: FLAKY row as in a serial run; its retry runs after the section, alone
#   P4 heavy par steps take the heavy-CPU lock ONCE for the section (one wait line, the first row says "cpu lock: waited"),
#      no par step starts before the holder lets go, each child sees WHIRL_CPU_LOCK_HELD
#   P5 -Jobs 2: never more than 2 par steps at once (and 2 do overlap)
#   P6 -Strict: par steps run serially; header "parallel: off (strict)"
#   P7 WHIRL_GATE_QUIET_WAIT: the quiet script runs before each par step starts (-Who names the step)
#   P8 each par step gets its own WHIRL_TEST_TMP and TMP / TEMP under <Work>\test_tmp\<step>
#   P9 GPU steps are never run in parallel ("par" ignored)
#   P10 WHIRL_GATE_JOBS sets the default concurrency; the header names the groups
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
          "WHIRL_GATE_CPU_LOCK_TIMEOUT_S", "WHIRL_GATE_CPU_LOCK_POLL_S", "WHIRL_GATE_FLAKY_TSV", "WHIRL_GATE_STRICT", "WHIRL_CPU_LOCK_HELD",
          "WHIRL_GATE_JOBS", "WHIRL_GATE_QUIET_WAIT", "WHIRL_GATE_COSTS")
$saved = @{}
foreach ($k in $keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); SetEnvOrRemove $k $null }
$env:WHIRL_GATE_SWIFT = $sw; $env:WHIRL_GATE_MOE = $mx; $env:WHIRL_GATE_TEXT_ROOT = $Work
$env:WHIRL_GATE_CPU_LOCK = $lockF; $env:WHIRL_GATE_FLAKY_TSV = $tsv; $env:WHIRL_GATE_CPU_LOCK_POLL_S = "1"
# 9.24: never the workspace's real quiet_wait.ps1 / step-cost table in a selftest (a live GPU runner's QUIET_WANTED would stall it)
$env:WHIRL_GATE_QUIET_WAIT = "off"; $env:WHIRL_GATE_COSTS = "off"
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
    # releases 20 s after the light step ran: the wait (>= ~17 s) stays well above a trivial step's own time on a loaded
    # machine (2026-10-31 procopt2 measurement run beside two builds: with 10 s the wait was 7 s and the step's own pwsh
    # start-up 5 s, so 'step s + 3 <= wait s' failed once - a margin of the test, not of the gate)
    $h = Start-Holder "l1" 20 "$Work\l1.go"; $started += $h
    $r = Gate "t_l1" $sf
    $h.WaitForExit(30000) | Out-Null
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

    # ==== 9.24 parallel sections
    # a par test step: sleeps $ms, appends "<start ticks> <end ticks>" to <case>_<name>.iv; $mode: "" ok, "flaky" fails on
    # its first run, "bad" always fails (exit 4); $extra runs before the verdict (e.g. writes what the step saw)
    function ParStep([string]$case, [string]$name, [int]$ms, [string]$mode = "", [string]$extra = "", [bool]$heavy = $false, [bool]$gpu = $false) {
        $iv = Join-Path $Work "${case}_$name.iv"
        $b = "`$a = [DateTime]::UtcNow.Ticks; Start-Sleep -Milliseconds $ms; `$e = [DateTime]::UtcNow.Ticks; `"`$a `$e`" | Add-Content '$iv'; $extra; "
        if ($mode -eq "flaky") { $b += "`$f = '$Work\${case}_$name.cnt'; `$n = if (Test-Path `$f) { [int](Get-Content `$f) } else { 0 }; `$n + 1 | Set-Content `$f; if (`$n -eq 0) { 'par boom first try'; exit 3 }; " }
        if ($mode -eq "bad") { $b += "'par step $name broken'; exit 4; " }
        $b += "'par step ${name}: ok'"
        @{ name = $name; heavy = $heavy; gpu = $gpu; par = $true; cmd = (StepCmd $b); rx = "^par step ${name}: ok" }
    }
    function Ivs([string]$case, [string[]]$names) {
        @(foreach ($n in $names) { foreach ($l in @(Get-Content (Join-Path $Work "${case}_$n.iv") -ErrorAction SilentlyContinue)) { $p = "$l" -split " "; ,@([long]$p[0], [long]$p[1], $n) } })
    }
    # the largest number of intervals open at one moment
    function MaxConc($ivs) {
        $ev = @(foreach ($i in $ivs) { ,@($i[0], 1); ,@($i[1], -1) }) | Sort-Object @{ e = { $_[0] } }, @{ e = { $_[1] } }
        $cur = 0; $mx = 0; foreach ($e in $ev) { $cur += $e[1]; if ($cur -gt $mx) { $mx = $cur } }; $mx
    }
    # rows and verdict without the seconds and the tag (the run-to-run part)
    function Norm($r, [string]$tag) { @($r.sum | Select-Object -Skip 1 | ForEach-Object { ($_ -replace "\b\d+ s\b", "# s") -replace [regex]::Escape($tag), "TAG" }) }
    $light = { param($n) @{ name = $n; heavy = $false; gpu = $false; cmd = (StepCmd "'light $n`: ok'"); rx = "^light $n`: ok" } }

    # ---- P1 overlap with -Jobs 4
    $pn = @("test_pa", "test_pb", "test_pc", "test_pd")
    $r = Gate "t_p1" (Steps "p1" @($pn | ForEach-Object { ParStep "p1" $_ 3000 })) @("-Jobs", "4")
    $iv = @(Ivs "p1" $pn)
    $wall = if ($iv.Count -eq 4) { (($iv | ForEach-Object { $_[1] } | Measure-Object -Maximum).Maximum - ($iv | ForEach-Object { $_[0] } | Measure-Object -Minimum).Minimum) / 1e7 } else { -1 }
    Check "P1 -Jobs 4: 4 par steps PASS and >= 3 ran at once" ($r.rc -eq 0 -and $iv.Count -eq 4 -and (MaxConc $iv) -ge 3 -and @($r.sum -match "^PASS test_p").Count -eq 4) "conc $(MaxConc $iv) | $(Last $r)"
    Check "P1 -Jobs 4: the section's wall time < 0.75 x the sum of the sleeps (12 s)" ($wall -gt 0 -and $wall -lt 9) "wall $wall s"

    # ---- P2 same summary as a serial run (finish order reversed, a failing par step, light steps around)
    function P2Steps([string]$case) {
        @((& $light "test_l0"), (ParStep $case "test_pa" 3000), (ParStep $case "test_pb" 2000), (ParStep $case "test_bad" 500 "bad"), (ParStep $case "test_pc" 200), (& $light "test_l9"))
    }
    $rs = Gate "t_p2s" (Steps "p2s" (P2Steps "p2s")) @("-Jobs", "1")
    $rp = Gate "t_p2p" (Steps "p2p" (P2Steps "p2p")) @("-Jobs", "4")
    $ns = Norm $rs "t_p2s"; $np = Norm $rp "t_p2p"
    $ivs = @(Ivs "p2s" @("test_pa", "test_pb", "test_bad", "test_pc")); $ivp = @(Ivs "p2p" @("test_pa", "test_pb", "test_pc"))
    Check "P2 serial run: no overlap; parallel run: par steps overlapped" ((MaxConc $ivs) -eq 1 -and (MaxConc $ivp) -ge 2) "serial $(MaxConc $ivs) parallel $(MaxConc $ivp)"
    Check "P2 rows (seconds masked) + verdict identical to the serial run, in -List order" ($ns.Count -eq 7 -and (($ns -join "`n") -eq ($np -join "`n")) -and $np[0] -match "^PASS test_l0 " -and $np[1] -match "^PASS test_pa " -and $np[3] -match "^FAIL test_bad .*retry also failed" -and $np[5] -match "^PASS test_l9 ") "serial:`n$($ns -join "`n")`nparallel:`n$($np -join "`n")"
    Check "P2 exit code and markers identical (1, FAIL_ only)" ($rs.rc -eq 1 -and $rp.rc -eq 1 -and (Test-Path (Join-Path $rp.work "FAIL_t_p2p")) -and -not (Test-Path (Join-Path $rp.work "PASS_t_p2p"))) "rc $($rs.rc) / $($rp.rc)"
    $lg = @(foreach ($n in "test_pa", "test_pb", "test_pc") { "$(Get-Content (Join-Path $rp.logs "$n.txt") -Raw -ErrorAction SilentlyContinue)".Trim() })
    Check "P2 each par step's log holds its own output only" ($lg[0] -eq "par step test_pa: ok" -and $lg[1] -eq "par step test_pb: ok" -and $lg[2] -eq "par step test_pc: ok" -and (Test-Path (Join-Path $rp.logs "test_bad.try1.txt"))) "logs: $($lg -join ' | ')"

    # ---- P3 FLAKY inside a section: retry after the section, alone
    $r = Gate "t_p3" (Steps "p3" @((ParStep "p3" "test_pf" 300 "flaky"), (ParStep "p3" "test_pa" 2500), (ParStep "p3" "test_pb" 2500))) @("-Jobs", "4")
    $ivf = @(Ivs "p3" @("test_pf")); $ivo = @(Ivs "p3" @("test_pa", "test_pb"))
    $endO = ($ivo | ForEach-Object { $_[1] } | Measure-Object -Maximum).Maximum
    Check "P3 FLAKY row as in a serial run, verdict FLAKY, exit 0" ($r.rc -eq 0 -and ($r.sum -match "^FLAKY test_pf\s+exit\s+0 .*par step test_pf: ok.*\| first try: exit\s+3 .*par boom first try") -and $r.sum[-1] -match "FLAKY \(3 steps passed, 1 only on a retry: test_pf\)") "$(Last $r)"
    Check "P3 the retry started after every step of the section ended" ($ivf.Count -eq 2 -and $ivo.Count -eq 2 -and $ivf[1][0] -gt $endO) "flaky tries $($ivf.Count), retry start $(if ($ivf.Count -eq 2) { $ivf[1][0] }) section end $endO"

    # ---- P4 one lock for the section
    $hl = Join-Path $Work "p4_held.txt"
    $ex = "`"`$env:WHIRL_CPU_LOCK_HELD`" | Add-Content '$hl'"
    # the holder lets go 8 s after the light step before the section ran (timing-independent, as L1)
    $h = Start-Holder "p4" 8 "$Work\p4.go"; $started += $h
    $go4 = @{ name = "test_l0"; heavy = $false; gpu = $false; cmd = (StepCmd "'go' | Set-Content '$Work\p4.go'; 'light test_l0: ok'"); rx = "^light test_l0: ok" }
    $r = Gate "t_p4" (Steps "p4" @($go4, (ParStep "p4" "test_pa" 1000 "" $ex $true), (ParStep "p4" "test_pb" 1000 "" $ex $true), (ParStep "p4" "test_pc" 1000 "" $ex $true))) @("-Jobs", "4")
    $h.WaitForExit(30000) | Out-Null
    $rel = [long](Get-Content "$Work\holder_p4.released" -ErrorAction SilentlyContinue)
    $iv = @(Ivs "p4" @("test_pa", "test_pb", "test_pc"))
    $minS = ($iv | ForEach-Object { $_[0] } | Measure-Object -Minimum).Minimum
    $hv = @(Get-Content $hl -ErrorAction SilentlyContinue)
    Check "P4 one 'cpu lock: busy' wait for the whole section; only the first row says 'cpu lock: waited'" ($r.rc -eq 0 -and @($r.out -match "cpu lock: busy").Count -eq 1 -and ($r.sum[2] -match "^PASS test_pa .*cpu lock: waited \d+ s") -and -not ($r.sum[1] -match "cpu lock") -and -not ($r.sum[3] -match "cpu lock") -and -not ($r.sum[4] -match "cpu lock")) "$(Last $r)"
    Check "P4 no par step started before the holder let go" ($iv.Count -eq 3 -and $rel -gt 0 -and $minS -gt $rel) "released $rel first start $minS"
    Check "P4 every child saw WHIRL_CPU_LOCK_HELD naming the section" ($hv.Count -eq 3 -and @($hv | Where-Object { $_ -match "^release gate t_p4/parallel test_pa\.\.test_pc" }).Count -eq 3 -and (MaxConc $iv) -ge 2) "seen: $($hv -join ' | ')"

    # ---- P5 -Jobs 2 limit
    $pn = @("test_pa", "test_pb", "test_pc", "test_pd")
    $r = Gate "t_p5" (Steps "p5" @($pn | ForEach-Object { ParStep "p5" $_ 2500 })) @("-Jobs", "2")
    $iv = @(Ivs "p5" $pn)
    Check "P5 -Jobs 2: at most 2 par steps at once, and 2 did overlap" ($r.rc -eq 0 -and $iv.Count -eq 4 -and (MaxConc $iv) -eq 2) "conc $(MaxConc $iv) | $(Last $r)"

    # ---- P6 strict: serial
    $r = Gate "t_p6" (Steps "p6" @("test_pa", "test_pb", "test_pc" | ForEach-Object { ParStep "p6" $_ 1500 })) @("-Strict", "-Jobs", "4")
    $iv = @(Ivs "p6" @("test_pa", "test_pb", "test_pc"))
    Check "P6 -Strict: par steps run one at a time; header 'parallel: off (strict)'" ($r.rc -eq 0 -and $iv.Count -eq 3 -and (MaxConc $iv) -eq 1 -and $r.sum[0] -match "parallel: off \(strict\)") "conc $(MaxConc $iv) | $($r.sum | Select-Object -First 1)"

    # ---- P7 quiet script before each par step
    $qf = Join-Path $Work "p7_quiet.txt"; $qs = Join-Path $Work "p7_quiet.ps1"
    "param([string]`$Who = '', [int]`$MaxWaitMin = 0)`n`"who=`$Who`" | Add-Content '$qf'`nexit 0" | Set-Content $qs
    $env:WHIRL_GATE_QUIET_WAIT = $qs
    $r = Gate "t_p7" (Steps "p7" @((& $light "test_l0"), (ParStep "p7" "test_pa" 200), (ParStep "p7" "test_pb" 200))) @("-Jobs", "4")
    $env:WHIRL_GATE_QUIET_WAIT = "off"
    $q = @(Get-Content $qf -ErrorAction SilentlyContinue)
    Check "P7 WHIRL_GATE_QUIET_WAIT ran once before each par step (-Who names it), not for the light step" ($r.rc -eq 0 -and $q.Count -eq 2 -and @($q -match "who=release gate t_p7/test_pa").Count -eq 1 -and @($q -match "who=release gate t_p7/test_pb").Count -eq 1) "quiet calls: $($q -join ' | ')"

    # ---- P8 own temp folders
    $tf8 = Join-Path $Work "p8_tmp.txt"
    $ex = "`"`$env:WHIRL_TEST_TMP|`$env:TMP|`$env:TEMP`" | Add-Content '$tf8'"
    $r = Gate "t_p8" (Steps "p8" @((ParStep "p8" "test_pa" 200 "" $ex), (ParStep "p8" "test_pb" 200 "" $ex))) @("-Jobs", "4")
    $tl = @(Get-Content $tf8 -ErrorAction SilentlyContinue)
    $okT = $tl.Count -eq 2 -and @($tl | Where-Object { $p = $_ -split "\|"; $n = if ($p[0] -like "*\test_pa") { "test_pa" } else { "test_pb" }; $p[0] -like "*\work_t_p8\test_tmp\$n" -and $p[1] -like "$($p[0])\*" -and $p[2] -eq $p[1] }).Count -eq 2 -and ($tl[0] -split "\|")[0] -ne ($tl[1] -split "\|")[0]
    Check "P8 each par step has its own WHIRL_TEST_TMP (<Work>\test_tmp\<step>) and TMP = TEMP inside it" ($r.rc -eq 0 -and $okT) "seen: $($tl -join ' || ')"

    # ---- P9 GPU steps never parallel
    $r = Gate "t_p9" (Steps "p9" @("test_ga", "test_gb" | ForEach-Object { ParStep "p9" $_ 1500 "" "" $false $true })) @("-Jobs", "4")
    $iv = @(Ivs "p9" @("test_ga", "test_gb"))
    Check "P9 GPU steps marked par still run one at a time" ($r.rc -eq 0 -and $iv.Count -eq 2 -and (MaxConc $iv) -eq 1) "conc $(MaxConc $iv) | $(Last $r)"

    # ---- P10 WHIRL_GATE_JOBS default + header
    $env:WHIRL_GATE_JOBS = "3"
    $pn = @("test_pa", "test_pb", "test_pc", "test_pd")
    $r = Gate "t_p10" (Steps "p10" @($pn | ForEach-Object { ParStep "p10" $_ 2500 }))
    SetEnvOrRemove "WHIRL_GATE_JOBS" $null
    $iv = @(Ivs "p10" $pn)
    Check "P10 WHIRL_GATE_JOBS=3: at most 3 at once (>= 2), header 'parallel: 1 group(s), 4 CPU steps, -Jobs 3'" ($r.rc -eq 0 -and $iv.Count -eq 4 -and (MaxConc $iv) -le 3 -and (MaxConc $iv) -ge 2 -and $r.sum[0] -match "parallel: 1 group\(s\), 4 CPU steps, -Jobs 3") "conc $(MaxConc $iv) | $($r.sum | Select-Object -First 1)"
} finally {
    foreach ($p in $started) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    foreach ($k in $keys) { SetEnvOrRemove $k $saved[$k] }
    Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
}
Write-Host "gate flow selftest: $ok/$($ok + $bad) as expected"
if ($bad) { exit 1 }
exit 0
