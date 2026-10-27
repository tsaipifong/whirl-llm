# SPDX-License-Identifier: Apache-2.0
# FAIL / PASS proofs of the release_gate steps added by arch025-Q6 (ARCH-1 8.7 condition 2): srv_vis, vis_ab, pkg_smoke,
# srv_kvram719, srv_env_e2 and the always-listed Q4_K_M / DFlash2 steps (st_dense_*, golden_dense, seq_dense_*, dfl_*).
# CPU only: no exe, no model, no GPU. release_gate step cpu_step_selftest.
#   1. -List: every new step is listed (GPU), the unavailable ones with "(not in this gate: ...)", on both -Device values
#   2. availability: an unavailable step in -Steps is a setup error (exit 2, "not available here"); with WHIRL_GATE_DENSE
#      set the dense steps are listed as available
#   3. preflight of srv_vis: no WHIRL_GATE_IMAGES / an image missing -> a srv_vis setup line; complete inputs -> none
#   4. rules: planted outputs through each step's own rule (release_gate.ps1 -RuleCheck): a planted failure FAILs, the
#      clean output PASSes
#   5. the scripts' own verdict tests: pkg_smoke.ps1 -SelfTest, vis_ab.ps1 -SelfTest, tools/repro/kvram719_tier.ps1 -SelfTest
#   pwsh tools/gates/gate_steps_selftest.ps1 [-Work DIR]
# Last line "gate steps selftest: N/M as expected"; exit 0 all as expected, 1 not.
param([string]$Work = "")
$ErrorActionPreference = "Stop"
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-gate-steps-selftest" }
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Work | Out-Null
$gate = Join-Path $PSScriptRoot "release_gate.ps1"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pwsh = (Get-Process -Id $PID).Path
$ok = 0; $bad = 0
function Check([string]$name, [bool]$cond, [string]$detail) {
    if ($cond) { $script:ok++; Write-Host "ok   $name" } else { $script:bad++; Write-Host "BAD  $name`: $detail" }
}
function Gate([string[]]$a) { $o = @(& $pwsh -NoProfile -File $gate @a 2>&1 | ForEach-Object { "$_" }); return [pscustomobject]@{ out = $o; code = $LASTEXITCODE } }
$envKeep = @{}
foreach ($v in "WHIRL_GATE_DENSE", "WHIRL_GATE_DFLASH", "WHIRL_GATE_IMAGES", "WHIRL_GATE_MMPROJ", "WHIRL_GATE_SWIFT", "WHIRL_GATE_MOE", "WHIRL_GATE_TEXT_ROOT", "WHIRL_GATE_VIS_BASE", "WHIRL_GATE_PACKAGE") {
    $envKeep[$v] = [Environment]::GetEnvironmentVariable($v); [Environment]::SetEnvironmentVariable($v, $null)
}
try {
    # ---- 1. -List
    $r97 = Gate @("-Device", "r9700", "-List"); $s8 = Gate @("-Device", "8060s", "-List")
    foreach ($n in "srv_vis", "vis_ab", "pkg_smoke", "srv_kvram719") {
        Check "list r9700 $n GPU" (@($r97.out | Where-Object { $_ -match "^$n\s+GPU\s*$" }).Count -eq 1) "not listed as an available GPU step"
        Check "list 8060s $n GPU" (@($s8.out | Where-Object { $_ -match "^$n\s+GPU\s*$" }).Count -eq 1) "not listed as an available GPU step"
    }
    Check "list cpu_platform_check CPU (arch025-PL5)" (@($r97.out | Where-Object { $_ -match "^cpu_platform_check\s+CPU\s*$" }).Count -eq 1 -and @($s8.out | Where-Object { $_ -match "^cpu_platform_check\s+CPU\s*$" }).Count -eq 1) "missing"
    Check "list cpu_step_selftest CPU" (@($r97.out | Where-Object { $_ -match "^cpu_step_selftest\s+CPU\s*$" }).Count -eq 1) "missing"
    Check "list r9700 srv_env_e2 not available" (@($r97.out | Where-Object { $_ -match "^srv_env_e2\s+GPU\s+\(not in this gate: 8060s only" }).Count -eq 1) "want '(not in this gate: 8060s only'"
    Check "list 8060s srv_env_e2 GPU" (@($s8.out | Where-Object { $_ -match "^srv_env_e2\s+GPU\s*$" }).Count -eq 1) "want available on the 8060s"
    foreach ($n in "st_dense_precise", "st_dense_balance", "st_dense_fast", "golden_dense", "seq_dense_precise", "seq_dense_balance", "seq_dense_fast", "dfl_precise", "dfl_balance", "dfl_fast") {
        Check "list r9700 $n named, needs WHIRL_GATE_DENSE" (@($r97.out | Where-Object { $_ -match "^$n\s+GPU\s+\(not in this gate: needs WHIRL_GATE_DENSE" }).Count -eq 1) "not listed with the reason"
    }
    Check "list 8060s dfl_balance r9700 only" (@($s8.out | Where-Object { $_ -match "^dfl_balance\s+GPU\s+\(not in this gate: r9700 only\)" }).Count -eq 1) "want '(not in this gate: r9700 only)'"
    # pcpu.ps1 (scratch\arch025) matches these names; the new lines must not change them
    Check "list srv_env_e1 / e7 / e4 unchanged" (($r97.out -match "srv_env_e1\s+GPU") -and ($r97.out -match "srv_env_e7\s+GPU") -and ($r97.out -match "srv_env_e4\s+GPU") -and ($r97.out -match "cpu_logscan_selftest\s+CPU")) "an existing -List line changed"

    # ---- 2. availability
    $x = Gate @("-Device", "r9700", "-Steps", "st_dense_balance")
    Check "block: -Steps st_dense_balance without WHIRL_GATE_DENSE -> exit 2" ($x.code -eq 2 -and ($x.out -match "step st_dense_balance is not available here: needs WHIRL_GATE_DENSE")) "exit $($x.code): $($x.out | Select-Object -Last 1)"
    $x = Gate @("-Device", "r9700", "-Steps", "srv_env_e2")
    Check "block: -Steps srv_env_e2 on the r9700 -> exit 2" ($x.code -eq 2 -and ($x.out -match "step srv_env_e2 is not available here: 8060s only")) "exit $($x.code): $($x.out | Select-Object -Last 1)"
    $x = Gate @("-Device", "8060s", "-Steps", "dfl_fast")
    Check "block: -Steps dfl_fast on the 8060s -> exit 2" ($x.code -eq 2 -and ($x.out -match "step dfl_fast is not available here: r9700 only")) "exit $($x.code)"
    $dq = Join-Path $Work "q4.gguf"; "x" | Set-Content $dq
    $env:WHIRL_GATE_DENSE = $dq
    $x = Gate @("-Device", "r9700", "-List")
    $env:WHIRL_GATE_DENSE = $null
    # a -Module run drops an unavailable step with a note (the impact map names steps for both GPUs), keeps the rest
    $mf = Join-Path $Work "m.json"
    '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":["cpu_server","srv_env_e2"]}}}' | Set-Content $mf
    $xm = Gate @("-Device", "r9700", "-Module", "m", "-NoMapSteps", "-ModulesFile", $mf, "-Build", (Join-Path $Work "nobuild"))
    Check "module run: srv_env_e2 dropped on the r9700 with a note, cpu_server kept (setup stops at the fake build)" ($xm.code -eq 2 -and ($xm.out -match "step srv_env_e2 not available here \(8060s only") -and -not ($xm.out -match "is not available here:") -and ($xm.out -match "release gate setup: missing")) "exit $($xm.code): $($xm.out | Select-Object -Last 2)"
    Check "pass: with WHIRL_GATE_DENSE the dense and dfl steps are available" (@($x.out | Where-Object { $_ -match "^(st_dense_\w+|golden_dense|seq_dense_\w+|dfl_\w+)\s+GPU\s*$" }).Count -eq 10) "$(@($x.out | Where-Object { $_ -match 'dense|dfl' }) -join ' | ')"

    # ---- 3. srv_vis preflight (fake build folder: the exe lines fail too, only the srv_vis lines are judged)
    $sw = Join-Path $Work "swift.gguf"; "x" | Set-Content $sw; $env:WHIRL_GATE_SWIFT = $sw
    $fb = Join-Path $Work "nobuild"
    $x = Gate @("-Device", "r9700", "-Steps", "srv_vis", "-Build", $fb)
    Check "block: srv_vis without an mmproj and WHIRL_GATE_IMAGES -> setup lines" ($x.code -eq 2 -and ($x.out -match "srv_vis / vis_ab: no mmproj") -and ($x.out -match "srv_vis: WHIRL_GATE_IMAGES not set")) "exit $($x.code)"
    $img = Join-Path $Work "img"; New-Item -ItemType Directory -Force $img | Out-Null; "x" | Set-Content (Join-Path $img "shapes.png")
    "x" | Set-Content (Join-Path $Work "mmproj-Test-F16.gguf")
    $env:WHIRL_GATE_IMAGES = $img
    $x = Gate @("-Device", "r9700", "-Steps", "srv_vis", "-Build", $fb)
    Check "block: srv_vis with dialog.png / s1080.png missing -> setup lines; mmproj found next to the Swift GGUF" ($x.code -eq 2 -and ($x.out -match "dialog.png not found") -and ($x.out -match "s1080.png not found") -and -not ($x.out -match "no mmproj")) "exit $($x.code): $(@($x.out | Where-Object { $_ -match 'srv_vis' }) -join ' | ')"
    "x" | Set-Content (Join-Path $img "dialog.png"); "x" | Set-Content (Join-Path $img "s1080.png")
    $x = Gate @("-Device", "r9700", "-Steps", "srv_vis", "-Build", $fb)
    Check "pass: srv_vis with complete inputs -> no srv_vis setup line (only the fake build's)" ($x.code -eq 2 -and -not ($x.out -match "srv_vis") -and ($x.out -match "missing .*whirl-server-gate.exe")) "$(@($x.out | Where-Object { $_ -match 'srv_vis' }) -join ' | ')"
    $env:WHIRL_GATE_IMAGES = $null; $env:WHIRL_GATE_SWIFT = $null

    # ---- 4. rules on planted outputs
    $allPass = "== whirl-server-gate: 14 passed, 0 failed (ALL PASS)"
    $cases = @(
        @{ step = "srv_vis"; name = "clean"; exit = 0; lines = @("== suite vis", $allPass); want = $true },
        @{ step = "srv_vis"; name = "planted: one vis check failed"; exit = 1; lines = @("FAIL vis: KV pool with --mmproj == without", "== whirl-server-gate: 13 passed, 1 failed"); want = $false },
        @{ step = "srv_vis"; name = "planted: ALL PASS line but exit 1"; exit = 1; lines = @($allPass); want = $false },
        @{ step = "srv_vis"; name = "planted: 0 checks ran"; exit = 0; lines = @("== whirl-server-gate: 0 passed, 0 failed (ALL PASS)"); want = $false },
        @{ step = "vis_ab"; name = "clean"; exit = 0; lines = @("vis ab: new vis-encode resident exit 0 sha256 EDBB0D70E50A6537", "vis ab: ok (resident == stream == chat dump EDBB0D70E50A6537; r9700)"); want = $true },
        @{ step = "vis_ab"; name = "planted: paths differ"; exit = 1; lines = @("vis ab: FAIL (paths_differ)"); want = $false },
        @{ step = "vis_ab"; name = "planted: exit 0 without the ok line"; exit = 0; lines = @("vis ab: new chat --image exit 0"); want = $false },
        @{ step = "pkg_smoke"; name = "clean"; exit = 0; lines = @("pkg smoke: REPORT kpack: ver 0 devices 0 chat 0", "pkg smoke: ok (zip whirl-0.2.5-windows-x64.zip 0123456789abcdef, r9700, with and without rocm_kpack.dll)"); want = $true },
        @{ step = "pkg_smoke"; name = "planted: member missing"; exit = 1; lines = @("pkg smoke: FAIL (member_missing:whirl-server.exe)"); want = $false },
        @{ step = "pkg_smoke"; name = "planted: ok line, exit 1"; exit = 1; lines = @("pkg smoke: ok (x)"); want = $false },
        @{ step = "srv_kvram719"; name = "clean"; exit = 0; lines = @("kvram719: REPORT verdict=PASS reason=ok hip719=0 tier=8192", "kvram719: PASS"); want = $true },
        @{ step = "srv_kvram719"; name = "planted: hipError 719"; exit = 1; lines = @("kvram719: REPORT verdict=FAIL reason=request hip719=1"); want = $false },
        @{ step = "srv_kvram719"; name = "planted: tier off (INVALID)"; exit = 3; lines = @("kvram719: REPORT verdict=INVALID reason=tier_off hip719=0"); want = $false },
        @{ step = "srv_env_e2"; name = "clean (8060s cells)"; exit = 0; lines = @("env matrix: ok (4 cell(s): E2_49152 PASS, E2_32768 PASS, E2_24576 PASS, E2_fake0.5 PASS) [8060s gfx1151, carve-out 96 GB]"); want = $true },
        @{ step = "srv_env_e2"; name = "clean (one GPU: SKIP)"; exit = 0; lines = @("env matrix: SKIP (this host has 1 GPU; the matrix needs two)"); want = $true },
        @{ step = "srv_env_e2"; name = "planted: a refused limit"; exit = 1; lines = @("env matrix: FAIL (E2_32768 FAIL) [8060s gfx1151, carve-out 96 GB]"); want = $false },
        @{ step = "srv_env_e2"; name = "planted: n/a path (no E2 cell ran)"; exit = 0; lines = @("cell E2: n/a (8060S only)", "env matrix: ok (0 cell(s): ) [r9700 gfx1201, carve-out 96 GB]"); want = $false },
        @{ step = "dfl_balance"; name = "clean"; exit = 0; lines = @("plain hash 0123456789abcdef", "dflash hash 0123456789abcdef drafts 40/60", "dflash-ngram hash 0123456789abcdef drafts 44/60"); want = $true },
        @{ step = "dfl_balance"; name = "planted: dflash hash differs"; exit = 0; lines = @("plain hash 0123456789abcdef", "dflash hash 0123456789abcdee drafts 40/60", "dflash-ngram hash 0123456789abcdef drafts 44/60"); want = $false },
        @{ step = "dfl_balance"; name = "planted: no draft accepted"; exit = 0; lines = @("plain hash 0123456789abcdef", "dflash hash 0123456789abcdef drafts 0/60", "dflash-ngram hash 0123456789abcdef drafts 0/60"); want = $false },
        @{ step = "st_dense_balance"; name = "clean"; exit = 0; lines = @("selftest: ok", "leak check: ok (1 units)"); want = $true },
        @{ step = "st_dense_balance"; name = "planted: no leak check line (arch025-Q2 rule, int4 merge)"; exit = 0; lines = @("selftest: ok"); want = $false },
        @{ step = "st_dense_balance"; name = "planted: selftest failed"; exit = 1; lines = @("selftest: FAIL (logits differ)"); want = $false },
        @{ step = "cpu_platform_check"; name = "clean"; exit = 0; lines = @("platform check: ok (0 site(s) in 0 file(s), 0 pending entr(y/ies); 0 new, 0 grown, 0 stale, 0 blocked)"); want = $true },
        @{ step = "cpu_platform_check"; name = "planted: a raw OS call outside the whitelist"; exit = 1; lines = @("  BLOCKED platform header / OS call src/model/x.cpp: 1 site(s) in a blocked scope (any mode)", "platform check: FAIL (1 site(s) in 1 file(s), 0 pending entr(y/ies); 0 new, 0 grown, 0 stale, 1 blocked)"); want = $false },
        @{ step = "cpu_platform_check"; name = "planted: exit 0 but sites reported (report line)"; exit = 0; lines = @("platform check: report (3 site(s) in 1 file(s), 0 pending entr(y/ies); 0 new, 0 grown, 0 stale, 0 blocked)"); want = $false },
        @{ step = "cpu_platform_check"; name = "planted: no verdict line"; exit = 0; lines = @("done"); want = $false },
        @{ step = "cpu_step_selftest"; name = "clean"; exit = 0; lines = @("gate steps selftest: 36/36 as expected"); want = $true },
        @{ step = "cpu_step_selftest"; name = "planted: one proof did not hold"; exit = 1; lines = @("BAD  rules: ...", "gate steps selftest: 35/36 as expected"); want = $false }
    )
    $rc = Join-Path $Work "rules.json"
    ConvertTo-Json -InputObject $cases -Depth 5 | Set-Content $rc
    $x = Gate @("-Device", "r9700", "-RuleCheck", $rc)
    $x.out | Where-Object { $_ -match "^(ok|BAD) " } | ForEach-Object { Write-Host "       rule $_" }
    Check "rules: $($cases.Count) planted / clean outputs through the step rules" ($x.code -eq 0 -and ($x.out -match "^rule check: $($cases.Count)/$($cases.Count) as expected")) "exit $($x.code): $($x.out | Select-Object -Last 1)"
    # the rule check itself must notice a wrong expectation (a check that cannot fail proves nothing)
    $wrong = @(@{ step = "srv_kvram719"; name = "inverted"; exit = 1; lines = @("kvram719: REPORT verdict=FAIL"); want = $true })
    ConvertTo-Json -InputObject $wrong -Depth 5 | Set-Content $rc
    $x = Gate @("-Device", "r9700", "-RuleCheck", $rc)
    Check "rules: an inverted expectation is reported (exit 1)" ($x.code -eq 1 -and ($x.out -match "^rule check: 0/1 as expected")) "exit $($x.code)"

    # ---- 5. the scripts' own verdict tests
    foreach ($t in @(@("tools\gates\pkg_smoke.ps1", "^pkg smoke selftest: (\d+)/\1 as expected"), @("tools\gates\vis_ab.ps1", "^vis ab selftest: (\d+)/\1 as expected"),
                     @("tools\repro\kvram719_tier.ps1", "."),
                     @("tools\gates\platform_check.ps1", "^platform check selftest: ok"))) {
        $o = @(& $pwsh -NoProfile -File (Join-Path $repo $t[0]) -SelfTest 2>&1 | ForEach-Object { "$_" }); $c = $LASTEXITCODE
        Check "$($t[0]) -SelfTest" ($c -eq 0 -and ($o -match $t[1])) "exit $${c}: $($o | Select-Object -Last 1)"
    }
} finally {
    foreach ($v in $envKeep.Keys) { [Environment]::SetEnvironmentVariable($v, $envKeep[$v]) }
    Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
}
Write-Host "gate steps selftest: $ok/$($ok + $bad) as expected"
if ($bad) { exit 1 }
exit 0
