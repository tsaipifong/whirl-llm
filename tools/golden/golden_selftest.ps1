# SPDX-License-Identifier: Apache-2.0
# CPU-only self-test of tools/golden/golden.ps1 (no GPU, no model): runs the gate against
# fake_whirl.cmd and a small fake model file in a temporary golden root and checks its verdicts.
#   pwsh tools/golden/golden_selftest.ps1      (exit 0 = all cases as expected, 1 = a case failed)
# Cases (REVIEW-RC4-C P3-4 / P3-5):
#   - record + check of a normal run: exit 0
#   - a run with only "G header" + "G info" (no step lines): exit 2, against any reference
#   - a reference with only model + header + info lines: exit 2 (it has no step lines)
#   - a record run where one case fails: exit 2 and no reference / models.txt is written
#   - in-run cross-checks (FIX-TEST-A, FR-TEST P2-7): the real gfx1201 dense references pass xcheck; a run
#     whose seg prefill lh differs from solo, or whose MTP stream is not an extension of the plain stream,
#     is an error (exit 2) and record writes nothing; xcheck on such a reference tree exits 2
#   - KL guard (arch025-klguard; KL is a regression guard, not a veto): a report that is not usable evidence (another
#     exe, 1 position, run errors) is refused before any run (exit 2, nothing written); a precise-only record needs no
#     report; a balance KL miss is a WARN line with the numbers and the items, recorded, evidence JSON written, exit 0;
#     a balance change without -TaskAccuracy gets a "TODO task-accuracy:" line, without a report "TODO kl-guard:";
#     a byte-identical re-record needs no report; a precise golden mismatch in check is still exit 1
#   - suite server (ARCH-1, fake_gate.cmd): record + check, a changed hash (exit 1), a failing in-run check or a
#     run without "(ALL PASS)" (exit 2); exe / reference-set hashes printed; refhash follows the reference bytes
#   - suite server log scan (arch024-S10, tools/gates/server_log_scan.ps1 owner set): a PASS line per case; an
#     "E owner check:" line in the case's server log, or no server log at all, fails it (exit 2); a WDDM line does not
#   - default suites (ARCH3, REVIEW-ARCH-D1 P3-7): dense skips suite server while swift runs it; a configured mmproj
#     that is missing fails the swift server case and record writes nothing
$ErrorActionPreference = "Stop"
$gate = Join-Path $PSScriptRoot "golden.ps1"
$fake = Join-Path $PSScriptRoot "fake_whirl.cmd"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-golden-selftest-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$model = Join-Path $tmp "fake-model.gguf"
[System.IO.File]::WriteAllBytes($model, [byte[]](1..200 | ForEach-Object { [byte]($_ % 256) }))
$savedDense = $env:WHIRL_GOLDEN_DENSE
$env:WHIRL_GOLDEN_DENSE = $model
$bad = 0

# a passing KL gate report for the fake exe (tools/klgate/klgate.py --report format)
# -Miss: one dense balance case beyond its guard line (KL 0.0061 > 0.005, top-1 96.80% < 97%), with the numerics line
# of that run (the items); -Errors: a report whose runs failed (not usable evidence)
function KlRep([string]$name, [bool]$pass, [string]$sha, [int]$positions = 3, [switch]$Miss, [switch]$Errors) {
    $p = Join-Path $tmp $name
    $cases = @(
        @{ model = "dense"; prompt = "code"; mode = "balance"; length = 8192; rule = "dense-length"; kl_mean = $(if ($Miss) { 0.0061 } else { 0.0011 })
           kl_limit = 0.005; top1 = $(if ($Miss) { 0.968 } else { 0.985 }); top1_limit = 0.97; zero_rows = 0; pass = -not $Miss },
        @{ model = "dense"; prompt = "code"; mode = "fast"; length = 8192; rule = "dense-length"; kl_mean = 0.012; kl_limit = 0.03
           top1 = 0.95; top1_limit = 0.93; zero_rows = 0; pass = $true })
    @{ exe_sha256 = $sha; device = "default"; pass = $pass; summary = "selftest"; modes = @("balance", "fast"); positions = $positions
       lengths = @(8192, 32768, 98304, 131072); models = @{ dense = "fake-model.gguf" }; cases = $cases; mode_args = @{}
       errors = @(if ($Errors) { "dense_code_balance: exit 3 (log x.log)" })
       numerics = @{ dense_code_balance = "numerics: balance - enabled: fp8, h16, kvq8 (q8h KV (int8 + f16 scale / 32, Hadamard-rotated q/k)), specsample (temperature > 0: race sampler); skipped: moefp8 (dense model); decode: q8dec (int8 activations in the decode / verify GEMV, as llama.cpp's q8_1)" }
    } | ConvertTo-Json -Depth 6 | Set-Content $p
    return $p
}
$fakeSha = (Get-FileHash -Algorithm SHA256 $fake).Hash.ToLower()
$klGood = KlRep "kl_good.json" $true $fakeSha
function RunGate([string]$action, [string]$root, [hashtable]$fakeEnv, [string]$kl = $klGood, [string]$modes = "precise", [string[]]$more = @()) {
    foreach ($k in @("FAKEGOLD_FAIL_SUITE", "FAKEGOLD_NOSTEPS", "FAKEGOLD_DIR")) { [Environment]::SetEnvironmentVariable($k, $null) }
    foreach ($k in $fakeEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $fakeEnv[$k]) }
    $env:WHIRL_GOLDEN_DENSE = $model
    $klArg = if ($kl) { @("-KlReport", $kl) } else { @() }
    $out = & pwsh -NoProfile -File $gate $action -Models dense -Modes $modes -Suites "single,pair" -Exe $fake `
        -GoldenRoot $root -Out (Join-Path $tmp "out") @klArg @more 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
    foreach ($k in @("FAKEGOLD_FAIL_SUITE", "FAKEGOLD_NOSTEPS", "FAKEGOLD_DIR")) { [Environment]::SetEnvironmentVariable($k, $null) }
    return @{ rc = $rc; out = $out }
}
function Expect([string]$name, $r, [int]$want, [scriptblock]$extra = { $true }) {
    $ok = ($r.rc -eq $want) -and (& $extra)
    Write-Host ("[{0}] {1} (exit {2}, want {3})" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $name, $r.rc, $want)
    if (-not $ok) { $r.out | Select-Object -Last 8 | ForEach-Object { Write-Host "      $_" }; $script:bad++ }
}

try {
    # 1) normal record + check
    $g1 = Join-Path $tmp "g1"
    $r = RunGate "record" $g1 @{}
    Expect "record of a normal run" $r 0 { (Test-Path (Join-Path $g1 "gfx1201\dense\precise_single.txt")) -and (Test-Path (Join-Path $g1 "models.txt")) }
    $r = RunGate "check" $g1 @{}
    Expect "check of the same run" $r 0
    # 2) run without step lines vs a normal reference
    $r = RunGate "check" $g1 @{ FAKEGOLD_NOSTEPS = "1" }
    Expect "run with header + info only (no step lines) is an error" $r 2
    # 3) reference without step lines (model + header + info) vs a run without step lines
    foreach ($s in @("single", "pair")) {
        $f = Join-Path $g1 "gfx1201\dense\precise_$s.txt"
        $l = @(Get-Content $f)
        [System.IO.File]::WriteAllLines($f, [string[]]@($l[0..2]))
    }
    $r = RunGate "check" $g1 @{ FAKEGOLD_NOSTEPS = "1" }
    Expect "reference with model + header + info only is an error" $r 2
    # 4) record with one failing case: nothing is written
    $g2 = Join-Path $tmp "g2"
    $r = RunGate "record" $g2 @{ FAKEGOLD_FAIL_SUITE = "pair" }
    Expect "record with a failing case writes no reference and no models.txt" $r 2 {
        -not (Test-Path (Join-Path $g2 "gfx1201\dense\precise_single.txt")) -and -not (Test-Path (Join-Path $g2 "models.txt")) }
    # 5) in-run cross-checks with the real dense precise references as the fake run's output
    $realRoot = Join-Path $PSScriptRoot "..\..\tests\golden"
    # every model that has committed gfx1201 references (the default model list includes swift, whose
    # references may not be recorded yet; the release gate's own xcheck step covers the required set)
    $realModels = (@(Get-ChildItem -Directory (Join-Path $realRoot "gfx1201") | ForEach-Object { $_.Name }) -join ",")
    $r = @{ out = @(& pwsh -NoProfile -File $gate xcheck -Arch gfx1201 -Models $realModels -GoldenRoot $realRoot 2>&1 | ForEach-Object { "$_" }); rc = $LASTEXITCODE }
    Expect "xcheck of the committed gfx1201 references ($realModels)" $r 0
    $fd = Join-Path $tmp "fake_out"
    New-Item -ItemType Directory -Force $fd | Out-Null
    foreach ($s in @("single", "pair")) {
        $l = @(Get-Content (Join-Path $realRoot "gfx1201\dense\precise_$s.txt"))
        [System.IO.File]::WriteAllLines((Join-Path $fd "$s.txt"), [string[]]@($l[1..($l.Count - 1)]))  # drop "G model"
    }
    $g3 = Join-Path $tmp "g3"
    $r = RunGate "record" $g3 @{ FAKEGOLD_DIR = $fd }
    Expect "record of a consistent run (real references as output)" $r 0
    $pf = Join-Path $fd "pair.txt"
    $pl = @(Get-Content $pf)
    $pl = @($pl | ForEach-Object { if ($_ -like "G seg0.prefill *") { $_ -replace "lh=[0-9a-f]+", "lh=0000000000000000" } else { $_ } })
    [System.IO.File]::WriteAllLines($pf, [string[]]$pl)
    $g4 = Join-Path $tmp "g4"
    $r = RunGate "record" $g4 @{ FAKEGOLD_DIR = $fd }
    Expect "record of a run with seg0 lh != solo0 lh is an error and writes nothing" $r 2 {
        -not (Test-Path (Join-Path $g4 "gfx1201\dense\precise_pair.txt")) -and ($r.out -match "solo0.prefill lh != seg0.prefill") }
    $r = RunGate "check" $g3 @{ FAKEGOLD_DIR = $fd }
    Expect "check of a run with seg0 lh != solo0 lh is an error" $r 2
    $sf = Join-Path $fd "single.txt"
    $sl = @(Get-Content $sf)
    $sl = @($sl | ForEach-Object { if ($_ -like "G mtp.stream *") { $_ -replace "ids=(\d+),", 'ids=1$1,' } else { $_ } })
    [System.IO.File]::WriteAllLines($sf, [string[]]$sl)
    $r = RunGate "check" $g3 @{ FAKEGOLD_DIR = $fd }
    Expect "check of a run whose MTP stream differs from the plain stream is an error" $r 2 { $r.out -match "MTP != plain" }
    # xcheck on a reference tree holding the broken pair file
    Copy-Item $pf (Join-Path $g3 "gfx1201\dense\precise_pair.txt") -Force
    $r = @{ out = @(& pwsh -NoProfile -File $gate xcheck -Arch gfx1201 -Models dense -Modes precise -Suites pair -GoldenRoot $g3 2>&1 | ForEach-Object { "$_" }); rc = $LASTEXITCODE }
    Expect "xcheck of a reference with seg0 lh != solo0 lh" $r 2
    # 6) KL guard (arch025-klguard): KL is a regression guard, not a veto; a report that is not usable evidence is
    #    a setup error (exit 2, nothing written); precise stays the strict bitwise golden
    $cases = @(@("a report of another exe", (KlRep "kl_other.json" $true ("0" * 64))),
               @("a single-position report", (KlRep "kl_pos1.json" $true $fakeSha 1)),
               @("a report whose runs failed", (KlRep "kl_err.json" $false $fakeSha -Errors)))
    foreach ($c in $cases) {
        $g6 = Join-Path $tmp ("g6_" + ($c[0] -replace " ", "_"))
        $r = RunGate "record" $g6 @{} $c[1] "precise,balance"
        Expect "record with $($c[0]) is refused before any run and writes nothing" $r 2 {
            -not (Test-Path (Join-Path $g6 "gfx1201\dense\balance_single.txt")) -and ($r.out -match "golden: record refused: kl guard: ERROR") }
    }
    # 6a) precise-only record without a KL report: precise is bitwise, the guard is not needed
    $g6a = Join-Path $tmp "g6a"
    $r = RunGate "record" $g6a @{} ""
    Expect "precise-only record without a KL report: written, KL guard NOT NEEDED" $r 0 {
        (Test-Path (Join-Path $g6a "gfx1201\dense\precise_single.txt")) -and ($r.out -match "^kl guard: NOT NEEDED") -and -not ($r.out -match "^TODO ") }
    # 6b) a KL miss in balance: WARN with the numbers and the items, recorded, evidence written, exit 0, and the
    #     missing task-accuracy reference is a named TODO line
    $g6b = Join-Path $tmp "g6b"
    $ev6b = Join-Path $tmp "ev6b.json"
    $r = RunGate "record" $g6b @{} (KlRep "kl_miss.json" $false $fakeSha -Miss) "precise,balance" @("-KlEvidence", $ev6b)
    $evj = if (Test-Path $ev6b) { Get-Content $ev6b -Raw | ConvertFrom-Json } else { $null }
    Expect "balance KL miss: WARN line with numbers and items, references written, evidence recorded, exit 0" $r 0 {
        (Test-Path (Join-Path $g6b "gfx1201\dense\balance_single.txt")) -and
        ($r.out -match "^WARN kl guard: dense code balance length 8192 \(dense-length\): KL 0\.00610 vs guard line 0\.00500; top1 96\.80% vs guard line 97\.00% \| mode balance, items fp8, h16, kvq8, specsample, q8dec \(allowed balance items: fp8, h16, kvq8, specsample, q8dec\)") -and
        ($r.out -match "^kl guard: WARN \(golden record gfx1201\): 1 of 1 case\(s\) beyond the guard line") -and
        ($r.out -match "^golden update note: gfx1201 balance: 2 reference\(s\); kl guard: WARN") -and
        $evj -and $evj.verdict -eq "WARN" -and @($evj.misses).Count -eq 1 -and $evj.misses[0].kl_mean -eq 0.0061 }
    Expect "balance KL miss without -TaskAccuracy: a named TODO task-accuracy line" $r 0 {
        ($r.out -match "^TODO task-accuracy: balance changed \(golden record gfx1201\)") -and @($evj.todo).Count -eq 1 }
    # 6c) the same with a task-accuracy reference: no TODO, the reference and its hash are in the evidence
    $g6c = Join-Path $tmp "g6c"
    $ta = Join-Path $tmp "taskacc.json"; '{"verdict":"selftest"}' | Set-Content $ta
    $ev6c = Join-Path $tmp "ev6c.json"
    $r = RunGate "record" $g6c @{} (Join-Path $tmp "kl_miss.json") "balance" @("-KlEvidence", $ev6c, "-TaskAccuracy", $ta)
    $evj = if (Test-Path $ev6c) { Get-Content $ev6c -Raw | ConvertFrom-Json } else { $null }
    Expect "balance KL miss with -TaskAccuracy: WARN, no TODO, reference in the evidence" $r 0 {
        ($r.out -match "^WARN kl guard: dense code balance") -and -not ($r.out -match "^TODO ") -and $evj -and "$($evj.task_accuracy.sha256)" -match "^[0-9a-f]{64}$" }
    # 6d) a balance change without any KL report: recorded, TODO kl-guard + TODO task-accuracy, exit 0
    $g6d = Join-Path $tmp "g6d"
    $r = RunGate "record" $g6d @{} "" "balance"
    Expect "balance change without a KL report: written, TODO kl-guard and TODO task-accuracy, NOT RUN" $r 0 {
        (Test-Path (Join-Path $g6d "gfx1201\dense\balance_single.txt")) -and ($r.out -match "^TODO kl-guard: balance changed") -and
        ($r.out -match "^TODO task-accuracy: balance changed") -and ($r.out -match "^kl guard: NOT RUN") }
    # 6e) re-record of unchanged balance references (a bitwise refactor): no KL report needed, no TODO
    $r = RunGate "record" $g6d @{} "" "balance"
    Expect "re-record of byte-identical balance references: KL guard NOT NEEDED, no TODO" $r 0 {
        ($r.out -match "every reference is byte-identical") -and ($r.out -match "^kl guard: NOT NEEDED") -and -not ($r.out -match "^TODO ") }
    # 6f) a balance case within its guard line: PASS (the TODO for the task-accuracy reference still applies)
    $g6f = Join-Path $tmp "g6f"
    $r = RunGate "record" $g6f @{} $klGood "balance"
    Expect "balance within the guard line: kl guard PASS" $r 0 { ($r.out -match "^kl guard: PASS \(golden record gfx1201\): 1 case") -and ($r.out -match "^TODO task-accuracy: balance") }
    # 6g) precise golden mismatch: still a blocking DIFF (exit 1); a balance mismatch in check also stays exit 1 (the
    #     bitwise pin of refactors) and says how to record an intended change
    $f = Join-Path $g6a "gfx1201\dense\precise_single.txt"
    $l = @(Get-Content $f); $l = @($l | ForEach-Object { $_ -replace "lh=aa", "lh=ab" }); [System.IO.File]::WriteAllLines($f, [string[]]$l)
    $r = RunGate "check" $g6a @{} ""
    Expect "precise golden mismatch is still a blocking DIFF (exit 1)" $r 1 { $r.out -match "DIFF  dense/precise_single" }
    $f = Join-Path $g6d "gfx1201\dense\balance_single.txt"
    $l = @(Get-Content $f); $l = @($l | ForEach-Object { $_ -replace "lh=aa", "lh=ab" }); [System.IO.File]::WriteAllLines($f, [string[]]$l)
    $r = RunGate "check" $g6d @{} "" "balance"
    Expect "balance golden mismatch in check: exit 1 with the record hint" $r 1 { $r.out -match "an intended balance output change: record new references with the KL guard report" }
    # 7) suite server (ARCH-1) through a fake whirl-server-gate: record / check, a hash change, a failing in-run check,
    #    a gate run without its ALL PASS line; the exe and reference-set hashes are printed; refhash follows the files
    $fakeGate = Join-Path $PSScriptRoot "fake_gate.cmd"
    function RunSrv([string]$action, [string]$root, [hashtable]$ge) {
        foreach ($k in @("FAKEGATE_FAIL", "FAKEGATE_NOPASS", "FAKEGATE_H", "FAKEGATE_NOLOG", "FAKEGATE_LOGLINE")) { [Environment]::SetEnvironmentVariable($k, $null) }
        foreach ($k in $ge.Keys) { [Environment]::SetEnvironmentVariable($k, $ge[$k]) }
        $env:WHIRL_GOLDEN_DENSE = $model
        $o = & pwsh -NoProfile -File $gate $action -Models dense -Modes precise -Suites server -Exe $fake -ServerExe $fake -GateExe $fakeGate `
            -GoldenRoot $root -Out (Join-Path $tmp "out") -KlReport $klGood 2>&1 | ForEach-Object { "$_" }
        $c = $LASTEXITCODE
        foreach ($k in @("FAKEGATE_FAIL", "FAKEGATE_NOPASS", "FAKEGATE_H", "FAKEGATE_NOLOG", "FAKEGATE_LOGLINE")) { [Environment]::SetEnvironmentVariable($k, $null) }
        return @{ rc = $c; out = $o }
    }
    $g7 = Join-Path $tmp "g7"
    $r = RunSrv "record" $g7 @{}
    Expect "suite server: record" $r 0 { Test-Path (Join-Path $g7 "gfx1201\dense\precise_server.txt") }
    $r = RunSrv "check" $g7 @{}
    Expect "suite server: check of the same run prints the exe and reference-set hashes" $r 0 {
        ($r.out -match "golden: exe .* sha256 [0-9a-f]{64}") -and ($r.out -match "golden: gate exe .* sha256 [0-9a-f]{64}") -and
        ($r.out -match "golden: reference set gfx1201 sha256 [0-9a-f]{64}") }
    $r = RunSrv "check" $g7 @{ FAKEGATE_H = "2222222222222222" }
    Expect "suite server: a changed output hash is a mismatch" $r 1 { $r.out -match "DIFF  dense/precise_server" }
    $r = RunSrv "check" $g7 @{ FAKEGATE_FAIL = "1" }
    Expect "suite server: a failing in-run check of the gate is an error" $r 2
    $r = RunSrv "check" $g7 @{ FAKEGATE_NOPASS = "1" }
    Expect "suite server: a gate run without '(ALL PASS)' is an error" $r 2
    # arch024-S10 (PLAN M8): the server log scan of each server case (tools/gates/server_log_scan.ps1, owner set)
    $r = RunSrv "check" $g7 @{}
    Expect "suite server: the case's server log is scanned (PASS line)" $r 0 { $r.out -match "dense/precise_server server log scan: PASS 1 server log\(s\)" }
    $r = RunSrv "check" $g7 @{ FAKEGATE_LOGLINE = "2026-01-01 00:00:01.000 E owner check: PagePool::take called on thread 4242" }
    Expect "suite server: an owner check line in a server log fails the case" $r 2 {
        ($r.out -match "server log scan: FAIL 1 line\(s\) matched \[owner\]") -and ($r.out -match "whirl exit 10") }
    $r = RunSrv "check" $g7 @{ FAKEGATE_NOLOG = "1" }
    Expect "suite server: a case with no server log fails" $r 2 { $r.out -match "server log scan: FAIL no server log" }
    $r = RunSrv "check" $g7 @{ FAKEGATE_LOGLINE = "2026-01-01 00:00:01.000 W GPU memory spill: 1.0 GiB" }
    Expect "suite server: a WDDM line is not golden's (owner set only)" $r 0 { $r.out -match "server log scan: PASS" }
    $h1 = @(& pwsh -NoProfile -File $gate refhash -Models dense -Modes precise -Suites server -GoldenRoot $g7 2>&1 | ForEach-Object { "$_" })
    $rc1 = $LASTEXITCODE
    Add-Content (Join-Path $g7 "gfx1201\dense\precise_server.txt") "G srv.extra h=0"
    $h2 = @(& pwsh -NoProfile -File $gate refhash -Models dense -Modes precise -Suites server -GoldenRoot $g7 2>&1 | ForEach-Object { "$_" })
    $okh = $rc1 -eq 0 -and $LASTEXITCODE -eq 0 -and ($h1[0] -match "sha256 [0-9a-f]{64} \(2 files") -and $h1[0] -ne $h2[0]
    Expect "refhash: prints the reference-set hash (2 files) and it follows the reference bytes" @{ rc = $(if ($okh) { 0 } else { 1 }); out = $h1 + $h2 } 0
    # 8) default suites (no -Suites; REVIEW-ARCH-D1 P3-7): the dense Q4_K_M model skips suite server, a product model
    #    (swift) runs it; a configured mmproj that is missing fails the server case and record writes nothing
    $swiftModel = Join-Path $tmp "fake-swift.gguf"
    [System.IO.File]::WriteAllBytes($swiftModel, [byte[]](1..300 | ForEach-Object { [byte](($_ * 7) % 256) }))
    $mmproj = Join-Path $tmp "fake-mmproj.gguf"
    [System.IO.File]::WriteAllBytes($mmproj, [byte[]](1..50))
    $imgs = Join-Path $tmp "images"
    New-Item -ItemType Directory -Force $imgs | Out-Null
    [System.IO.File]::WriteAllBytes((Join-Path $imgs "shapes.png"), [byte[]](1..10))
    $klBoth = Join-Path $tmp "kl_both.json"
    @{ exe_sha256 = $fakeSha; device = "default"; pass = $true; summary = "selftest"; modes = @("balance", "fast"); positions = 3
       lengths = @(8192, 32768, 98304, 131072); models = @{ dense = "fake-model.gguf"; swift = "fake-swift.gguf" } } |
        ConvertTo-Json -Depth 4 | Set-Content $klBoth
    function RunDefault([string]$root, [string]$mm) {
        $saved = @{}
        foreach ($k in @("WHIRL_GOLDEN_SWIFT", "WHIRL_GOLDEN_MMPROJ_SWIFT", "WHIRL_GOLDEN_IMAGES")) { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
        $env:WHIRL_GOLDEN_DENSE = $model; $env:WHIRL_GOLDEN_SWIFT = $swiftModel; $env:WHIRL_GOLDEN_MMPROJ_SWIFT = $mm; $env:WHIRL_GOLDEN_IMAGES = $imgs
        $o = & pwsh -NoProfile -File $gate record -Models "dense,swift" -Modes precise -Exe $fake -ServerExe $fake -GateExe $fakeGate `
            -GoldenRoot $root -Out (Join-Path $tmp "out") -KlReport $klBoth 2>&1 | ForEach-Object { "$_" }
        $c = $LASTEXITCODE
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
        return @{ rc = $c; out = $o }
    }
    $g8 = Join-Path $tmp "g8"
    $r = RunDefault $g8 $mmproj
    Expect "default suites: dense skips suite server, swift runs it (record)" $r 0 {
        $d = Join-Path $g8 "gfx1201"
        (Test-Path "$d\dense\precise_single.txt") -and (Test-Path "$d\dense\precise_ngram.txt") -and
        -not (Test-Path "$d\dense\precise_server.txt") -and (Test-Path "$d\swift\precise_server.txt") -and
        (Test-Path "$d\swift\precise_single.txt") -and -not ($r.out -match "dense/precise_server") }
    $g9 = Join-Path $tmp "g9"
    $r = RunDefault $g9 (Join-Path $tmp "no-such-mmproj.gguf")
    Expect "default suites: a missing mmproj fails swift/precise_server and record writes nothing" $r 2 {
        ($r.out -match "FAIL  swift/precise_server suite server needs the mmproj") -and
        -not (Test-Path (Join-Path $g9 "gfx1201\swift\precise_single.txt")) -and -not (Test-Path (Join-Path $g9 "models.txt")) }
} finally {
    $env:WHIRL_GOLDEN_DENSE = $savedDense
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($bad -gt 0) { Write-Host "golden selftest: $bad case(s) failed"; exit 1 }
Write-Host "golden selftest: all cases as expected"
exit 0
