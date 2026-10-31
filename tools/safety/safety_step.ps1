# SPDX-License-Identifier: Apache-2.0
# arch025-Q1 / Q3: the release_gate CPU steps of the C++ safety tools (tools/gates/release_gate.ps1 cpu_asan, cpu_analyze,
# cpu_safety_selftest). Each kind builds in its own directory under the source tree (build-asan, build-analyze; /build-*/ is in
# .gitignore), runs under a hard time cap, and prints one verdict line that the gate's rule reads.
#
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind asan     [-Gguf swift.gguf] [-GgufMoe moe.gguf] [-CapMin 60]
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind analyze  [-CapMin 70]
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind selftest
#   pwsh -NoProfile -File tools/safety/safety_step.ps1 -Kind tsa      [-CapMin 15]
#
#  asan     tools/safety/run_asan.bat (ASan + leak-check build of the 10 CPU test exes and the fault driver; every suite, the CPU
#           pins and the 8 fault modes), then tools/safety/asan_check.py check: every suite clean and passing, every fault mode
#           caught by the checker meant for it, the ASan opt-out whitelist (asan_allow.txt) exact. Verdict "cpu asan: ok (...)".
#  analyze  tools/safety/run_analyze.bat (full /analyze rebuild of the CPU test targets + baseline gate), then the gate must PASS,
#           tools/safety/analyze_check.py reasons (every baseline key has file:line + reason in analyze_whitelist.tsv; keyed on
#           file|rule|function + count, the line is a hint: drift = INFO) and the injected-fault proof tools/safety/analyze_fault.bat must hold (arch025-Q3: baseline-only TU
#           PASS, 3 planted warnings NEW -> FAIL, a 2nd warning of a baselined key NEW x1 -> FAIL).
#           Verdict "cpu analyze: ok (...)".
#  selftest the block / pass unit tests of the verdicts (test_asan_check.py, test_analyze_check.py, tsa/test_tsa_check.py) and the two
#           whitelist checks without a build, plus the real-compiler analyze_fault.bat proof (3 one-TU cl compiles; seconds). Verdict
#           "cpu safety selftest: ok".
#  tsa      (arch025-Q4) tools/safety/tsa/run_tsa_build.bat: clang-cl -Wthread-safety -Werror=thread-safety build of whirl_server_core +
#           whirl_tier_core in build-tsa, then the lock-removal proof (tsa_check.py mutate: each removed lock must FAIL the build);
#           tsa_check.py verdict. Verdict "cpu tsa: ok (...)".
# Time cap (the 2x rule; measured times in progress\arch025-Q135.HANDOFF.md): over -CapMin the step's own process tree is killed
# and the step FAILS. Environment: WHIRL_PYTHON (python), WHIRL_KERNELS_FROM (HIP code objects for a new build dir, e.g. the gated
# build\Release of this tree), WHIRL_QUIET_WAIT (a local script build_cfg.bat runs before compiling); release_gate sets them.
# Exit 0 only with the ok verdict.
#
# -Mode (arch025-procopt3, PLAN 9.33; default WHIRL_SAFETY_MODE, else clean). Every verdict line says which mode ran:
#   clean        the build dirs under this tree (build-asan / build-analyze / build-tsa) as before; release_gate uses it for
#                strict / release runs (-Strict, tags gate_r9700 / gate_8060s)
#   cache        tools/safety/safety_cache.py: the tree is synced by CONTENT into one stable mirror (<parent of the tree>\safety_cache\src,
#                or WHIRL_SAFETY_CACHE\src) whose build-<kind> dirs persist across worktrees; only changed files get a new mtime, so
#                ninja (mtime + recorded header deps + command-line hash) rebuilds what a change reaches. A new or changed key (build
#                scripts, cmake / ninja / MSVC / clang-cl / HIP SDK versions, toolchain env) wipes the dir: "mode clean-cache (why)";
#                same key: "mode incremental". After the build safety_cache.py depscheck must pass (msvc_deps_prefix present, no
#                unparsed /showIncludes note, every TU with a project include has recorded deps); if it fails, the key is dropped
#                (next run clean-cache) and an incremental run FAILS. Only asan and tsa use the cache (PLAN 9.33b): there the
#                build is the cost and the findings come from running the build. cpu_analyze always builds clean in the tree
#                ("mode clean (cpu_analyze never uses the cache, PLAN 9.33b ...)"): its findings are the per-TU SARIF files, which an
#                incremental build keeps from older compiles, and the incremental analyze lost 6 of 22 warning keys in every round
#                (scratch\a25_procopt3\DONE_measure3). The cache is used by one step at a time (an
#                exclusive handle on <root>\LOCK, released when this process ends); busy -> this run builds in the tree ("mode clean
#                (cache busy)"). WHIRL_SAFETY_CACHE_CLEAN=1 forces clean-cache.
param(
    [Parameter(Mandatory)][ValidateSet("asan", "analyze", "selftest", "tsa")][string]$Kind,
    [string]$Gguf = "",
    [string]$GgufMoe = "",
    [int]$CapMin = 0,
    [ValidateSet("", "clean", "cache")][string]$Mode = ""
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
$py = if ($env:WHIRL_PYTHON) { $env:WHIRL_PYTHON } else { "python" }
$env:WHIRL_PYTHON = $py
$t0 = Get-Date
if (-not $Mode) { $Mode = if ($env:WHIRL_SAFETY_MODE -in @("clean", "cache")) { $env:WHIRL_SAFETY_MODE } else { "clean" } }
$modeNote = "mode clean (in-tree build-$Kind)"
$cacheMode = ""
$cacheLock = $null
if ($Mode -eq "cache" -and $Kind -eq "analyze") {
    $modeNote = "mode clean (cpu_analyze never uses the cache, PLAN 9.33b: an incremental /analyze lost 6 of 22 warning keys; in-tree build-analyze)"
}
if ($Mode -eq "cache" -and $Kind -in @("asan", "tsa")) {
    $cacheRoot = if ($env:WHIRL_SAFETY_CACHE) { $env:WHIRL_SAFETY_CACHE } else { Join-Path (Split-Path $repo -Parent) "safety_cache" }
    New-Item -ItemType Directory -Force $cacheRoot | Out-Null
    try { $cacheLock = [System.IO.File]::Open((Join-Path $cacheRoot "LOCK"), "OpenOrCreate", "ReadWrite", "None") } catch { $cacheLock = $null }
    if (-not $cacheLock) {
        $modeNote = "mode clean (cache busy: $cacheRoot\LOCK held by another safety step; in-tree build-$Kind)"
    } else {
        $mirror = Join-Path $cacheRoot "src"
        # WHIRL_SAFETY_SYNC_FROM (measurement / test only): sync another tree (e.g. a copy with an older commit's sources) instead
        $syncFrom = if ($env:WHIRL_SAFETY_SYNC_FROM) { $env:WHIRL_SAFETY_SYNC_FROM } else { $repo }
        if ($env:WHIRL_SAFETY_SYNC_FROM) { Write-Host "   cache: WHIRL_SAFETY_SYNC_FROM: syncing $syncFrom (not this tree)" }
        $so = @(& $py -I (Join-Path $here "safety_cache.py") sync $syncFrom $mirror 2>&1 | ForEach-Object { "$_" })
        $sc = $LASTEXITCODE
        $so | ForEach-Object { "   cache: $_" }
        if ($sc -ne 0) {
            $modeNote = "mode clean (cache sync failed rc=$sc; in-tree build-$Kind)"
        } else {
            $pa = @("prepare", $Kind, $mirror) + @(if ($env:WHIRL_SAFETY_CACHE_CLEAN -eq "1") { "--clean" })
            $po = @(& $py -I (Join-Path $mirror "tools\safety\safety_cache.py") @pa 2>&1 | ForEach-Object { "$_" })
            $pc = $LASTEXITCODE
            $po | ForEach-Object { "   cache: $_" }
            $pm = [regex]::Match(($po -join "`n"), "(?m)^mode (clean-cache|incremental) \((.*)\)$")
            if ($pc -ne 0 -or -not $pm.Success) {
                $modeNote = "mode clean (cache prepare failed rc=$pc; in-tree build-$Kind)"
            } else {
                $cacheMode = $pm.Groups[1].Value
                $copied = [regex]::Match(($so -join "`n"), "(\d+) copied").Groups[1].Value
                $modeNote = "mode $cacheMode ($($pm.Groups[2].Value -replace '; key \w+$', ''); $copied file(s) changed; $mirror\build-$Kind)"
                # from here on the mirror is the tree: its own copies of these scripts build in its build-<kind>
                $repo = $mirror
                $here = Join-Path $mirror "tools\safety"
            }
        }
    }
}
Write-Host "safety step $Kind`: $modeNote"

function DepsCheck([string]$b) {
    # cache mode only: are the header dependencies ninja recorded usable? (memory whirl-build-header-deps-broken)
    if (-not $cacheMode) { return "" }
    $do = @(& $py -I (Join-Path $here "safety_cache.py") depscheck $b $repo (Join-Path $b "logs\build.log") 2>&1 | ForEach-Object { "$_" })
    $dc = $LASTEXITCODE
    $do | ForEach-Object { "   cache: $_" } | Write-Host
    if ($dc -eq 0) { return "" }
    & $py -I (Join-Path $here "safety_cache.py") forget $Kind $repo | Out-Null
    if ($cacheMode -eq "incremental") { return "header deps not usable in an incremental run (cache key dropped: the next run is clean-cache)" }
    Write-Host "   cache: deps check failed after a clean-cache build: findings stand (full build), key dropped so the next run is clean again"
    return ""
}

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
    $tests = @(@("test_asan_check.py"), @("test_analyze_check.py"), @("tsa\test_tsa_check.py"), @("test_safety_cache.py"),@("asan_check.py", "sites", $repo),
               @("analyze_check.py", "reasons", (Join-Path $here "analyze_baseline.txt"), (Join-Path $here "analyze_whitelist.tsv")))
    foreach ($t in $tests) {
        $o = @(& $py -I (Join-Path $here $t[0]) @($t | Select-Object -Skip 1) 2>&1 | ForEach-Object { "$_" })
        $c = $LASTEXITCODE
        $o | Select-Object -Last 3 | ForEach-Object { "   $_" }
        Write-Output "$($t -join ' ' -replace [regex]::Escape($repo), '<repo>'): exit $c"
        if ($c -ne 0) { $bad++ }
    }
    # arch025-Q3: the real-compiler block AND pass proof of the /analyze baseline gate (3 one-TU cl /analyze compiles, seconds):
    # baseline-only TU PASS, 3 planted keys NEW -> FAIL, a 2nd warning of a baselined key NEW x1 -> FAIL. Output in build-analyze
    # (/build-*/ is ignored by git); needs only cl.
    $F = Join-Path $repo "build-analyze\fault_selftest"
    $fo = @(& cmd.exe /d /c "`"$(Join-Path $here 'analyze_fault.bat')`" `"$F`"" 2>&1 | ForEach-Object { "$_" })
    $fc = Rc (Join-Path $F "PROOF")
    $fo | Select-Object -Last 3 | ForEach-Object { "   $_" }
    Write-Output "analyze_fault.bat (clean PASS / new FAIL / same key FAIL): PROOF rc=$fc"
    if ($fc -ne 0) { $bad++ }
    Write-Output "cpu safety selftest: $(if ($bad -eq 0) { 'ok' } else { "FAIL ($bad of $($tests.Count + 1))" })"
    exit $(if ($bad -eq 0) { 0 } else { 1 })
}

if ($Kind -eq "asan") {
    # cap = 2 x measured + 2 for a FRESH build-asan (what a new tree gets): 17.4 min incl. HIP code-object rebuild (arch025-Q135 run 1);
    # an incremental run (build-asan kept, the usual dev case) took 7.2 min (run 3: build 9 s, suites + faults 7.0 min)
    if ($CapMin -le 0) { $CapMin = 37 }
    $B = Join-Path $repo "build-asan"
    $ok = RunCapped (Join-Path $here "run_asan.bat") @($Gguf, $GgufMoe) $CapMin "run_asan.bat"
    $dp = DepsCheck $B
    if (Test-Path (Join-Path $B "logs\steps.log")) { Get-Content (Join-Path $B "logs\steps.log") | ForEach-Object { "   $_" } }
    $o = @(& $py -I (Join-Path $here "asan_check.py") check $B $repo 2>&1 | ForEach-Object { "$_" })
    $c = $LASTEXITCODE
    $o | ForEach-Object { "$_" }
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($ok -and $c -eq 0 -and -not $dp) { Write-Output "cpu asan: ok ($modeNote; $min min; $(($o[-1]) -replace '^asan check: ok ', ''))"; exit 0 }
    Write-Output "cpu asan: FAIL ($modeNote; $min min$(if (-not $ok) { '; time cap hit' })$(if ($dp) { "; $dp" }))"
    exit 1
}

if ($Kind -eq "tsa") {
    # cap = 2 x measured + 2 for a FRESH build-tsa (configure + 39 clang-cl steps + 7 mutant / control compiles): 2.3 min (arch025-Q4
    # 2026-10-26 04:05-04:08, tsa_fresh.log); incremental 1.8 min
    if ($CapMin -le 0) { $CapMin = 7 }
    $B = Join-Path $repo "build-tsa"
    $ok = RunCapped (Join-Path $here "tsa\run_tsa_build.bat") @($B) $CapMin "run_tsa_build.bat"
    $dp = DepsCheck $B
    $o = @(& $py -I (Join-Path $here "tsa\tsa_check.py") verdict $B 2>&1 | ForEach-Object { "$_" })
    $c = $LASTEXITCODE
    $ml = Join-Path $B "logs\mutate.log"
    if (Test-Path $ml) { Get-Content $ml | ForEach-Object { "   $_" } }
    $bl = Join-Path $B "logs\build.log"
    if (Test-Path $bl) { Get-Content $bl | Where-Object { $_ -match "Wthread-safety|FAILED:" } | Select-Object -First 20 | ForEach-Object { "   build: $_" } }
    $o | ForEach-Object { "$_" }
    $min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($ok -and $c -eq 0 -and -not $dp) { Write-Output "cpu tsa: ok ($modeNote; $min min; $(($o[-1]) -replace '^tsa verdict: ok ', ''))"; exit 0 }
    Write-Output "cpu tsa: FAIL ($modeNote; $min min$(if (-not $ok) { '; time cap hit' })$(if ($dp) { "; $dp" }); $(($o[-1]) -replace '^tsa verdict: ', ''))"
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
# injected-fault proof (arch025-Q3, real cl, tests/safety/analyze_fault.cpp x3): the baseline-only TU must PASS, the 3 planted
# defects must be NEW (FAIL), a 2nd warning of the baselined key must be NEW x1 (FAIL); verdict analyze_check.py faultproof
$F = Join-Path $B "fault_proof"
$fo = @(& cmd.exe /d /c "`"$(Join-Path $here 'analyze_fault.bat')`" `"$F`"" 2>&1 | ForEach-Object { "$_" })
$fc = Rc (Join-Path $F "PROOF")
$fo | ForEach-Object { "   fault proof: $_" }
if ($fc -ne 0) { $bad += "injected-fault proof failed (PROOF rc=$fc): see $F\proof.log" }
$min = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
$gs = @(Get-Content $gl -ErrorAction SilentlyContinue | Where-Object { $_ -match "^analyze: " } | Select-Object -First 1)
if ($bad.Count -eq 0) { Write-Output "cpu analyze: ok ($modeNote; $min min; $($gs -join '') ; whitelist reasons ok; fault proof clean PASS / 3 NEW / same key NEW x1)"; exit 0 }
Write-Output "cpu analyze: FAIL ($modeNote; $min min; $($bad -join '; '))"
exit 1
