# SPDX-License-Identifier: Apache-2.0
# CPU-only self-test of the KL regression guard (arch025-klguard): tools/klgate/kl_guard.ps1 directly and the release
# gate step kl_guard (tools/gates/release_gate.ps1 -Steps kl_guard on a fake build folder: empty *.exe files, fake model
# files; no GPU, no model run).
#   pwsh tools/gates/kl_guard_selftest.ps1      (exit 0 = "kl guard selftest: N/N as expected", 1 = a case failed)
# Cases:
#   - a balance case beyond its guard line: release gate exit 0, row "PASS kl_guard", a "WARN kl guard:" line with the
#     numbers and the items under the row and in the summary before the verdict "PASS (1 steps)", evidence JSON written
#   - the same without -TaskAccuracy: a "TODO task-accuracy:" line in the summary; with it: no TODO
#   - no -KlReport: PASS with "TODO kl-guard:" (the guard was not run)
#   - a report that is not usable evidence (another exe, run errors, a negative --as test): FAIL row, exit 1
#   - kl_guard.ps1: -Modes empty = NOT NEEDED; a bad -TaskAccuracy = exit 2; fast miss items marked as fast items
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$gate = Join-Path $PSScriptRoot "release_gate.ps1"
$guard = Join-Path $repo "tools\klgate\kl_guard.ps1"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-klguard-selftest-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$bin = Join-Path $tmp "bin"
New-Item -ItemType Directory -Force $bin | Out-Null
foreach ($n in "whirl", "whirl-server", "whirl-server-gate", "whirl-kernel-test", "whirl-tests", "whirl-model-tests", "whirl-cache-tests",
               "whirl-vision-tests", "whirl-tier-tests", "whirl-server-tests", "whirl-hiprt-tests", "whirl-device-tests", "whirl-ops-tests",
               "whirl-pins", "whirl-startup-replay", "whirl-sampling-tests") { [System.IO.File]::WriteAllText((Join-Path $bin "$n.exe"), "fake $n") }
$sw = Join-Path $tmp "Swift-fake.gguf"; [System.IO.File]::WriteAllText($sw, "swift")
$mx = Join-Path $tmp "Ornith-fake.gguf"; [System.IO.File]::WriteAllText($mx, "moe")
$sha = (Get-FileHash -Algorithm SHA256 (Join-Path $bin "whirl.exe")).Hash.ToLower()
$num = @{
    swift_code_balance = "numerics: balance - enabled: fp8, gdnwmma, h16, kvq8 (q8h KV (int8 + f16 scale / 32, Hadamard-rotated q/k)), specsample (temperature > 0: race sampler); skipped: moefp8 (dense model); decode: q8dec (int8 activations in the decode / verify GEMV, as llama.cpp's q8_1)"
    moe_zh_fast = "numerics: fast - enabled: fp8, moefp8, gdnwmma, h16, specsample (x), kvq4 (q4: int4 + f16 scale / 32, Hadamard-rotated q/k); skipped: kvq8 (MoE models keep f16 KV); decode: q8dec (int8 activations)"
}
function Rep([string]$name, [string]$s = $sha, [switch]$Miss, [switch]$FastMiss, [switch]$Errors, [switch]$Neg) {
    $p = Join-Path $tmp $name
    $cases = @(
        @{ model = "swift"; prompt = "code"; mode = "balance"; length = 131072; rule = "dense-length"; kl_mean = $(if ($Miss) { 0.0072 } else { 0.0012 })
           kl_limit = 0.005; top1 = 0.9818; top1_limit = 0.97; zero_rows = 0; pass = -not $Miss },
        @{ model = "moe"; prompt = "zh"; mode = "fast"; length = 0; rule = "moe-mean"; kl_mean = $(if ($FastMiss) { 0.06 } else { 0.02 }); kl_limit = 0.04
           x_noise = $(if ($FastMiss) { 12.0 } else { 4.0 }); pass = -not $FastMiss })
    @{ exe_sha256 = $s; device = "r9700"; pass = -not ($Miss -or $FastMiss); summary = "selftest"; modes = @("balance", "fast"); positions = 3
       lengths = @(8192, 32768, 98304, 131072); models = @{ swift = "Swift-fake.gguf"; moe = "Ornith-fake.gguf" }; cases = $cases
       mode_args = $(if ($Neg) { @{ balance = "--balance=fp8" } } else { @{} }); errors = @(if ($Errors) { "moe_zh_fast: exit 3" }); numerics = $num
    } | ConvertTo-Json -Depth 6 | Set-Content $p
    return $p
}
$ta = Join-Path $tmp "taskacc.json"; '{"selftest": true}' | Set-Content $ta
$saved = @{}
foreach ($k in "WHIRL_GATE_SWIFT", "WHIRL_GATE_MOE", "WHIRL_GATE_TEXT_ROOT", "WHIRL_GATE_DENSE", "WHIRL_GATE_MX", "WHIRL_GATE_LOCK_SCRIPT") { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
$env:WHIRL_GATE_SWIFT = $sw; $env:WHIRL_GATE_MOE = $mx; $env:WHIRL_GATE_TEXT_ROOT = $tmp
foreach ($k in "WHIRL_GATE_DENSE", "WHIRL_GATE_MX", "WHIRL_GATE_LOCK_SCRIPT") { [Environment]::SetEnvironmentVariable($k, $null) }
$n = 0; $okN = 0
function Gate([string]$tag, [string[]]$extra) {
    $w = Join-Path $tmp "work_$tag"
    $o = @(& pwsh -NoProfile -File $gate -Device r9700 -Build $bin -Steps kl_guard -Work $w -Tag $tag @extra 2>&1 | ForEach-Object { "$_" })
    $c = $LASTEXITCODE
    $sf = Join-Path $w "${tag}_summary.txt"
    $s = if (Test-Path $sf) { @(Get-Content $sf) } else { @() }
    return @{ rc = $c; out = $o; sum = $s; ev = (Join-Path $w "$tag\kl_guard_evidence.json") }
}
function Expect([string]$name, $r, [int]$want, [scriptblock]$extra) {
    $script:n++
    $ok = ($r.rc -eq $want) -and [bool](& $extra)
    if ($ok) { $script:okN++ }
    Write-Host ("[{0}] {1} (exit {2}, want {3})" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $name, $r.rc, $want)
    if (-not $ok) { @($r.out) + @($r.sum) | Select-Object -Last 14 | ForEach-Object { Write-Host "      $_" } }
}
try {
    # 1) balance KL miss, no task-accuracy reference: exit 0, WARN + TODO under the row and in the summary, evidence
    $r = Gate "t1" @("-KlReport", (Rep "miss.json" -Miss))
    $ev = if (Test-Path $r.ev) { Get-Content $r.ev -Raw | ConvertFrom-Json } else { $null }
    Expect "balance KL miss: release gate exit 0, PASS row, WARN line with numbers + items, evidence recorded" $r 0 {
        $w = @($r.sum | Where-Object { $_ -match "^kl_guard: WARN kl guard: swift code balance length 131072 \(dense-length\): KL 0\.00720 vs guard line 0\.00500; top1 98\.18% vs guard line 97\.00% \| mode balance, items fp8, gdnwmma, h16, kvq8, specsample, q8dec \(allowed balance items: fp8, gdnwmma, h16, kvq8, specsample, q8dec\)" })
        ($r.sum | Where-Object { $_ -match "^PASS kl_guard .*kl guard: WARN \(release gate r9700\): 1 of 2 case\(s\) beyond the guard line" }) -and $w.Count -eq 1 -and
        $r.sum[-1] -match "release gate t1: PASS \(1 steps\)" -and ($r.out -match "^\s+WARN kl guard: swift code balance") -and
        $ev -and $ev.verdict -eq "WARN" -and @($ev.misses).Count -eq 1 -and $ev.misses[0].mode -eq "balance" }
    Expect "balance KL miss without -TaskAccuracy: a TODO task-accuracy line in the summary" $r 0 {
        @($r.sum | Where-Object { $_ -match "^kl_guard: TODO task-accuracy: balance / fast changed \(release gate r9700\)" }).Count -eq 1 }
    # 2) with a task-accuracy reference: no TODO; the reference hash is in the evidence
    $r = Gate "t2" @("-KlReport", (Join-Path $tmp "miss.json"), "-TaskAccuracy", $ta)
    $ev = if (Test-Path $r.ev) { Get-Content $r.ev -Raw | ConvertFrom-Json } else { $null }
    Expect "balance KL miss with -TaskAccuracy: WARN, no TODO, reference + sha256 in the evidence" $r 0 {
        -not ($r.sum -match "TODO ") -and ($r.sum -match "WARN kl guard: swift") -and "$($ev.task_accuracy.sha256)" -match "^[0-9a-f]{64}$" }
    # 3) no report: PASS with TODO kl-guard (and TODO task-accuracy)
    $r = Gate "t3" @()
    Expect "no -KlReport: PASS row 'kl guard: NOT RUN', TODO kl-guard in the summary" $r 0 {
        ($r.sum -match "^PASS kl_guard .*kl guard: NOT RUN") -and ($r.sum -match "^kl_guard: TODO kl-guard:") -and ($r.sum -match "^kl_guard: TODO task-accuracy:") }
    # 4) every case within its guard line: PASS
    $r = Gate "t4" @("-KlReport", (Rep "ok.json"), "-TaskAccuracy", "feat027:taskacc")
    Expect "within the guard lines + exp_id reference: PASS row, no WARN / TODO" $r 0 {
        ($r.sum -match "^PASS kl_guard .*kl guard: PASS \(release gate r9700\): 2 case") -and -not ($r.sum -match "^kl_guard: (WARN|TODO)") }
    # 5) not usable evidence: FAIL row, exit 1
    foreach ($c in @(@("another exe", (Rep "other.json" ("0" * 64))), @("run errors", (Rep "err.json" -Errors)), @("a negative --as test", (Rep "neg.json" -Neg)))) {
        $r = Gate ("t5" + ($c[0] -replace "\W", "")) @("-KlReport", $c[1])
        Expect "report with $($c[0]): FAIL row (kl guard: ERROR), exit 1" $r 1 { ($r.sum -match "^FAIL kl_guard .*kl guard: ERROR") }
    }
    # 6) kl_guard.ps1 directly: fast items are named as fast items; -Modes empty = NOT NEEDED; a bad -TaskAccuracy = exit 2
    $o = @(& $guard -Report (Rep "fmiss.json" -FastMiss) -Modes fast -TaskAccuracy $ta *>&1 | ForEach-Object { "$_" }); $c = $LASTEXITCODE
    Expect "fast miss: the MoE mean case with x N and the fast items" @{ rc = $c; out = $o } 0 {
        ($o -match "^WARN kl guard: moe zh fast mean over all windows \(moe-mean\): KL 0\.06000 = 12\.00 N vs guard line 0\.04000 \| mode fast, items fp8, moefp8, gdnwmma, h16, specsample, kvq4, q8dec \(allowed balance items: fp8, moefp8, gdnwmma, h16, specsample, q8dec; fast items: kvq4\)") -and $o[-1] -match "^kl guard: WARN" }
    $o = @(& $guard -Report (Join-Path $tmp "miss.json") -Modes @() *>&1 | ForEach-Object { "$_" }); $c = $LASTEXITCODE
    Expect "-Modes empty: NOT NEEDED, no TODO" @{ rc = $c; out = $o } 0 { $o[-1] -match "^kl guard: NOT NEEDED" -and -not ($o -match "^TODO ") }
    $o = @(& $guard -Report (Join-Path $tmp "miss.json") -TaskAccuracy "no such thing" *>&1 | ForEach-Object { "$_" }); $c = $LASTEXITCODE
    Expect "a -TaskAccuracy that is neither a file nor an exp_id: exit 2" @{ rc = $c; out = $o } 2 { $o -match "kl guard: ERROR -TaskAccuracy" }
} finally {
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($okN -ne $n) { Write-Host "kl guard selftest: $okN/$n as expected ($($n - $okN) failed)"; exit 1 }
Write-Host "kl guard selftest: $okN/$n as expected"
exit 0
