# SPDX-License-Identifier: Apache-2.0
# arch025-Q1 / Q3: the release_gate CPU steps of the C++ safety tools (tools/gates/release_gate.ps1 cpu_asan, cpu_analyze,
# cpu_safety_selftest). Each kind builds in its own directory under the source tree (build-asan, build-analyze; /build-*/ is in
# .gitignore), runs under a hard time cap, and prints one verdict line that the gate's rule reads.
#
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind asan     [-Gguf swift.gguf] [-GgufMoe moe.gguf] [-CapMin 60]
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind analyze  [-CapMin 70]
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind selftest
#
#  asan     tools/safety/run_asan.bat (ASan + leak-check build of the 10 CPU test exes and the fault driver; every suite, the CPU
#           pins and the 8 fault modes), then tools/safety/asan_check.py check: every suite clean and passing, every fault mode
#           caught by the checker meant for it, the ASan opt-out whitelist (asan_allow.txt) exact. Verdict "cpu asan: ok (...)".
#  analyze  tools/safety/run_analyze.bat (full /analyze rebuild of the CPU test targets + baseline gate), then the gate must PASS,
#           tools/safety/analyze_check.py reasons (every baseline key has file:line + reason in analyze_whitelist.tsv, at the line cl
#           reports now) and the injected-fault proof tools/safety/analyze_fault.bat must FAIL with its 3 planted warnings NEW.
#           Verdict "cpu analyze: ok (...)".
#  selftest the block / pass unit tests of both verdicts (test_asan_check.py, test_analyze_check.py) and the two whitelist checks
#           without a build (seconds). Verdict "cpu safety selftest: ok".
# Time cap (the 2x rule; measured times in progress\arch025-Q135.HANDOFF.md): over -CapMin the step's own process tree is killed
# and the step FAILS. Environment: WHIRL_PYTHON (python), WHIRL_KERNELS_FROM (HIP code objects for a new build dir, e.g. the gated
# build\Release of this tree), WHIRL_QUIET_WAIT (a local script build_cfg.bat runs before compiling); release_gate sets them.
# Exit 0 only with the ok verdict.
param(
    [Parameter(Mandatory)][ValidateSet("asan", "analyze", "selftest")][string]$Kind,
    [string]$Gguf = "",
    [string]$GgufMoe = "",
    [int]$CapMin = 0
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
$py = if ($env:WHIRL_PYTHON) { $env:WHIRL_PYTHON } else { "python" }
$env:WHIRL_PYTHON = $py
$t0 = Get-Date

function RunCapped([string]$bat, [string[]]$batArgs, [int]$capMin, [string]$what) {
    # cmd /c "<bat> args": the bat writes its own logs; this only waits with a cap and kills its own tree on overrun
    $q = @($batArgs | ForEach-Object { '"' + $_ + '"' }) -join " "
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/d /c `"`"$bat`" $q`"" -NoNewWindow -PassThru
    $ms = [int64]$capMin * 60000
    if (-not $p.WaitForExit($ms)) {
        & taskkill.exe /T /F /PID $p.Id *> $null
        Write-Host "$what`: TIMEOUT after $capMin min (cap): process tree $($p.Id) killed"
        return $false
    }
    Write-Host "$what`: finished in $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min, exit $($p.ExitCode)"
    return $true
}

function Rc([string]$f) {
    if (-not (Test-Path $f)) { return $null }
    $m = [regex]::Match((Get-Content -Raw $f), "rc=(-?\d+)")
    if ($m.Success) { return [int]$m.Groups[1].Value } else { return $null }
}

if ($Kind -eq "selftest") {
    $bad = 0
    foreach ($t in @(@("test_asan_check.py"), @("test_analyze_check.py"), @("asan_check.py", "sites", $repo),
                     @("analyze_check.py", "reasons", (Join-Path $here "analyze_baseline.txt"), (Join-Path $here "analyze_whitelist.tsv")))) {
        $o = @(& $py -I (Join-Path $here $t[0]) @($t | Select-Object -Skip 1) 2>&1 | ForEach-Object { "$_" })
        $c = $LASTEXITCODE
        $o | Select-Object -Last 3 | ForEach-Object { "   $_" }
        Write-Output "$($t -join ' ' -replace [regex]::Escape($repo), '<repo>'): exit $c"
        if ($c -ne 0) { $bad++ }
    }
    Write-Output "cpu safety selftest: $(if ($bad -eq 0) { 'ok' } else { "FAIL ($bad of 4)" })"
    exit $(if ($bad -eq 0) { 0 } else { 1 })
}

if ($Kind -eq "asan") {
    # cap = 2 x measured + 2 for a FRESH build-asan (what a new tree gets): 17.4 min incl. HIP code-object rebuild (arch025-Q135 run 1);
    # an incremental run (build-asan kept, the usual dev case) took 7.2 min (run 3: build 9 s, suites + faults 7.0 min)
    if ($CapMin -le 0) { $CapMin = 37 }
    $B = Join-Path $repo "build-asan"
    $ok = RunCapped (Join-Path $here "run_asan.bat") @($Gguf, $GgufMoe) $CapMin "run_asan.bat"
    if (Test-Path (Join-Path $B "logs\steps.log")) { Get-Content (Join-Path $B "logs\steps.log") | ForEach-Object { "   $_" } }
    $o = @(& $py -I (Join-Path $here "asan_check.py") check $B $repo 2>&1 | ForEach-Object { "$_" })
    $c = $LASTEXITCODE
    $o | ForEach-Object { "$_" }
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($ok -and $c -eq 0) { Write-Output "cpu asan: ok ($min min; $(($o[-1]) -replace '^asan check: ok ', ''))"; exit 0 }
    Write-Output "cpu asan: FAIL ($min min$(if (-not $ok) { '; time cap hit' }))"
    exit 1
}

# analyze
# cap = 2 x measured + 2 for a FRESH build-analyze (what a new tree / release machine gets): run 1 of arch025-Q135 took 16.6 min (configure +
# HIP code objects ~10 min + host TUs, until C1060); runs 2-3 with built HIP objects 1.5 / 1.4 min (60 forced /analyze TUs)
if ($CapMin -le 0) { $CapMin = 35 }
$B = Join-Path $repo "build-analyze"
$ok = RunCapped (Join-Path $here "run_analyze.bat") @($B) $CapMin "run_analyze.bat"
$bad = @()
if (-not $ok) { $bad += "time cap hit" }
$g = Rc (Join-Path $B "GATE")
$gl = Join-Path $B "logs\gate.log"
if (Test-Path $gl) { Get-Content $gl | ForEach-Object { "   gate: $_" } }
if ($g -ne 0) { $bad += "baseline gate GATE rc=$g (0 = PASS)" }
$o = @(& $py -I (Join-Path $here "analyze_check.py") reasons (Join-Path $here "analyze_baseline.txt") (Join-Path $here "analyze_whitelist.tsv") $B 2>&1 | ForEach-Object { "$_" })
if ($LASTEXITCODE -ne 0) { $bad += "whitelist reasons check failed" }
$o | ForEach-Object { "   $_" }
# injected-fault proof: 3 planted defects in tests/safety/analyze_fault.cpp must be NEW and the gate must FAIL
$F = Join-Path $B "fault_proof"
$fo = @(& cmd.exe /d /c "`"$(Join-Path $here 'analyze_fault.bat')`" `"$F`"" 2>&1 | ForEach-Object { "$_" })
$fg = Rc (Join-Path $F "GATE")
$fnew = @($fo | Where-Object { $_ -match "^NEW x\d+: tests/safety/analyze_fault\.cpp\|C\d+\|" }).Count
"   fault proof: GATE rc=$fg, $fnew planted warning(s) NEW" | Write-Output
if ($fg -ne 1 -or $fnew -lt 3) { $bad += "injected-fault proof did not FAIL with 3 NEW (rc=$fg, NEW $fnew)" }
$min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
$gs = @(Get-Content $gl -ErrorAction SilentlyContinue | Where-Object { $_ -match "^analyze: " } | Select-Object -First 1)
if ($bad.Count -eq 0) { Write-Output "cpu analyze: ok ($min min; $($gs -join '') ; whitelist reasons ok; fault proof 3 NEW)"; exit 0 }
Write-Output "cpu analyze: FAIL ($min min; $($bad -join '; '))"
exit 1
