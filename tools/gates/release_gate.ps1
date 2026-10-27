# SPDX-License-Identifier: Apache-2.0
# Release / rc gate: the canonical set of checks that must all pass before an rc merge or a release
# (docs/dev/testing.md). Versioned here (FIX-TEST-A, FR-TEST P2-1 / P2-4 / P2-5 / P2-6 / P3-12): the
# per-rc copies outside the repo only logged exit codes, wrote DONE markers unconditionally and left the
# verdict to whoever read the logs.
#
#   pwsh tools/gates/release_gate.ps1 -Device r9700|8060s [-Build DIR] [-ExpectCommit SHA] [-Steps a,b,...]
#                                     [-Work DIR] [-Tag NAME] [-InjectFail] [-List] [-LaunchLog DIR]
#                                     [-Module name[,name]] [-ModulesFile PATH] (-ImpactRange A..B|worktree | -NoMapSteps)
#                                     [-KlReport klgate_report.json] [-TaskAccuracy FILE|exp_id]
#
# -KlReport / -TaskAccuracy (arch025-klguard): inputs of step kl_guard, the KL regression guard (not a veto; see the
# step's comment). Its "WARN kl guard:" / "TODO task-accuracy:" / "TODO kl-guard:" lines are printed under the step's
# row and written into the summary before the verdict line. Self-test: tools/gates/kl_guard_selftest.ps1 (CPU).
#
# -Module (arch024-S12): select the steps of one or more modules from tools/gates/modules.json (PLAN c-1);
# -Steps given as well adds those steps (e.g. a module's `when` steps that impact.py prints). DEV-TIME FAST
# FEEDBACK ONLY: not for a release, an rc merge or a bug-fix release until the owner signs off O1 / O2 (arch024
# PLAN); those run the full gate on both GPUs. The -Tag of a module run is module_<names>_<device> (an explicit -Tag
# must start with module_: exit 2 otherwise), so a module run never writes the PASS_gate_<device> marker of a full
# gate, and its summary header says "module=<names>" (release/release_check.py R1 rejects such summaries).
# A module with "callers": "rdeps" (M1 handles) also runs the modules of every src/include file that includes its
# files (transitive); an includer in no module turns the run into the full gate. "callers": [names] (M8) runs the
# named modules too. Steps the impact map selects beyond the table (PLAN c-2): `python tools/impact/impact.py` prints
# them on its mod-gate line as -Steps. Exit 2 (setup error): unknown module, a module step that the
# gate does not have and modules.json does not mark pending, a pending step that already exists (stale list), a
# module with paths but none present in this tree and no lands_on, an empty selection, a -Tag not starting with
# module_. A module whose files have not landed on this branch (lands_on set) is printed as such and its steps still
# run. With -List only the selected steps are listed (and the tag).
# -Module needs -ImpactRange or -NoMapSteps (review K2: exit 2 otherwise). -ImpactRange A..B (or "worktree") runs
# tools/impact/impact.py on that diff and adds what its mod-gate line says: the `when` steps and the steps the map
# selects beyond the module table (PLAN c-2); impact.py's "full gate" turns the run into the full gate; a module it
# names that -Module does not list is exit 2. -NoMapSteps runs the table only and says so in the summary header.
# Steps added beyond the module table (by -Steps or -ImpactRange) are printed as "run on both GPUs" (review K1).
# Python (impact.py, cpu_modules_check): WHIRL_GATE_PYTHON, else C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe when it
# exists, else python / py on PATH.
#
# Verdict: every step has an exit code AND an output rule (e.g. "(ALL PASS)", "seqtest: ok", three equal
# DFlash2 hashes with drafts accepted); a step that does not run counts as failed. At the end a table of
# all steps is printed and written to <Work>\<Tag>_summary.txt, plus PASS_<Tag> or FAIL_<Tag> (never
# both, never a plain DONE).
# Exit: 0 every selected step passed; 1 a step failed; 2 setup error (missing exe / model / text root,
#       wrong commit, unknown step name, 0 steps selected).
#
# Machine paths come from the environment (a local wrapper sets them; nothing here is machine specific):
#   WHIRL_GATE_SWIFT   Swift-1.5 27B MXFP4-A GGUF (required: the dense model of every dense step)
#   WHIRL_GATE_MOE     Ornith-1.5-35B-A3B MXFP4 GGUF (required)
#   WHIRL_GATE_DENSE   Qwen3.8-27B Q4_K_M GGUF (optional, SWIFT-GATES: adds the st/golden/seq "dense" steps,
#                      kernel test --q4 and the DFlash2 steps; unset = those steps are not in the gate)
#   WHIRL_GATE_MX      old name of WHIRL_GATE_SWIFT (used when WHIRL_GATE_SWIFT is unset)
#   WHIRL_GATE_DFLASH  DFlash2 drafter GGUF (needed only with WHIRL_GATE_DENSE)
#   WHIRL_GATE_TEXT_ROOT  llama.cpp checkout (server gate texts)
#   WHIRL_GATE_LOCK_SCRIPT  optional: a script defining Enter-R9700Lock; each GPU step then holds
#                           that lock for its own duration only (other agents interleave)
#                           The 8060s gate never takes the R9700 lock: its GPU steps run with
#                           WHIRL_R9700_LOCK_HELD set, else whirl-server-gate's lock (which waits for any
#                           whirl.exe, also one on the R9700) stalls each server step 10 min (REVIEW-ARCH-D1 P3-8)
# Steps (-List prints them): CPU test binaries, gate / golden self-tests, golden xcheck, the C++ safety steps
# (cpu_safety_selftest, cpu_asan, cpu_analyze, cpu_tsa: arch025-Q1 / Q3 / Q4, own build dirs, ~tens of minutes of CPU); kernel test
# --require-all; selftest Swift + MoE x 3 modes; golden check Swift + MoE; seqtest Swift x 3 modes and MoE
# precise + balance; with WHIRL_GATE_DENSE also the Q4_K_M selftest / golden / seqtest x 3 modes and bench
# DFlash2 x 3 modes (hashes equal, drafts > 0; target Q4_K_M, Swift + DFlash2 not yet gated: inventory G10);
# server gate (model Swift) balance + fast basic,mt_cache,
# precise basic with --cli, precise mt_cache_precise (multi-turn cache == cold, bitwise), fast_stream dense + MoE, tier,sys,restore_conc,pool in balance and precise
# (--kv auto: the formats the modes ship, not a pinned q8v).
# -InjectFail: the golden steps compare against a copy of the references with one hash changed (a real
# check that must turn the gate red; used to prove the gate fails).
# -LaunchLog DIR: every GPU step runs with WHIRL_LAUNCH_LOG=DIR\<step>: each whirl process writes its kernel launch
# table there (ARCH-1 K-0; tools/impact/launch_table.py builds and compares the tables); server-gate steps pass the
# folder to their servers with --server-env (arch024-S0b).
param(
    [ValidateSet("r9700", "8060s")][string]$Device = "r9700",
    [string]$Build = "",
    [string]$ExpectCommit = "",
    [string[]]$Steps = @(),
    [string]$Work = "",
    [string]$Tag = "",
    [switch]$InjectFail,
    [switch]$List,
    [string]$LaunchLog = "",
    [string[]]$Module = @(),
    [string]$ModulesFile = "",
    [string]$ImpactRange = "",
    [switch]$NoMapSteps,
    [string]$KlReport = "",      # step kl_guard: the tools/klgate/klgate.py --report of this exe on this GPU (regression guard)
    [string]$TaskAccuracy = ""   # step kl_guard: the task-accuracy result reference (file or KB exp_id) of this build's balance / fast
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Build) { $Build = Join-Path $repo "build\Release" }
if (-not $Tag) { $Tag = "gate_$Device" }
# one folder per GPU: two gates at once must not share the server steps' SSD tier folders (ARCH3 d5 srv_cache_balance)
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-release-gate-$Device" }
$Steps = @($Steps | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Module = @($Module | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($Module.Count -gt 0) {
    if (-not $PSBoundParameters.ContainsKey("Tag")) { $Tag = "module_$($Module -join '+')_$Device" }
    # review A5: a module run must never write the marker / summary of a full gate (PASS_gate_<device>)
    elseif ($Tag -notlike "module_*") { Write-Host "release gate -Module: -Tag '$Tag' refused: a module run's tag must start with module_ (dev-time only, never PASS_gate_*)"; exit 2 }
}
$arch = if ($Device -eq "8060s") { "gfx1151" } else { "gfx1201" }
$port = if ($Device -eq "8060s") { 8095 } else { 8093 }
$Q = $env:WHIRL_GATE_DENSE; $MX = $env:WHIRL_GATE_MOE; $DF = $env:WHIRL_GATE_DFLASH
$SW = if ($env:WHIRL_GATE_SWIFT) { $env:WHIRL_GATE_SWIFT } else { $env:WHIRL_GATE_MX }
$hasQ = [bool]$Q  # Q4_K_M is optional (SWIFT-GATES): Swift + Ornith are the required models
# python for impact.py and cpu_modules_check: WHIRL_GATE_PYTHON (authoritative when set), else the workspace's pyref
# interpreter, else python / py on PATH (a release machine's gate shell often has no python on PATH)
$py = $env:WHIRL_GATE_PYTHON
if (-not $py) {
    $pyref = "C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
    if (Test-Path $pyref) { $py = $pyref }
    else { $py = @(@(Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1) | ForEach-Object { $_.Source })[0] }
}
if (-not $py) { $py = "python-not-found" }
$exe = @{}
foreach ($n in "whirl", "whirl-server", "whirl-server-gate", "whirl-kernel-test", "whirl-tests", "whirl-model-tests", "whirl-cache-tests",
               "whirl-vision-tests", "whirl-tier-tests", "whirl-server-tests", "whirl-hiprt-tests", "whirl-device-tests", "whirl-ops-tests",
               "whirl-pins", "whirl-startup-replay") { $exe[$n] = Join-Path $Build "$n.exe" }

# ---------------------------------------------------------------------------------------------------
# step table: name, gpu, command (array: exe + args), rule (scriptblock over the output lines + exit code)
$S = [System.Collections.Generic.List[object]]::new()
function AddStep([string]$name, [bool]$gpu, [string[]]$cmd, [scriptblock]$rule, [hashtable]$envv = @{}) {
    $S.Add([pscustomobject]@{ name = $name; gpu = $gpu; cmd = $cmd; rule = $rule; env = $envv })
}
$exit0 = { param($o, $c) $c -eq 0 }
# dense model keys: swift always, dense (Q4_K_M) only when WHIRL_GATE_DENSE is set
$denseKeys = @("swift") + @(if ($hasQ) { "dense" })
$modelOf = @{ swift = $SW; dense = $Q; moe = $MX }
$gmodels = (@("swift", "moe") + @(if ($hasQ) { "dense" })) -join ","
$pwsh = (Get-Process -Id $PID).Path
foreach ($t in "whirl-tests", "whirl-model-tests", "whirl-cache-tests", "whirl-vision-tests", "whirl-tier-tests", "whirl-device-tests", "whirl-ops-tests") {
    AddStep "cpu_$($t -replace '^whirl-','' -replace '-tests$','')" $false @($exe[$t]) { param($o, $c) $c -eq 0 -and ($o -match "\b[1-9]\d* passed, 0 failed") }
}
# HIP runtime loader (ARCH-1 D-1): fake / absent runtimes, then this machine's runtime through load()
# (a runtime that does not import rocm_kpack.dll must never load it)
AddStep "cpu_hiprt" $false @($exe["whirl-hiprt-tests"]) { param($o, $c) $c -eq 0 -and ($o -match "hiprt tests: [1-9]\d* passed, 0 failed") -and ($o -match "^real runtime: ok=1") } @{ WHIRL_TEST_REAL_HIPRT = "1" }
# arch6b: without hiprt_fakes\ next to the exe the test must fail with a message, not crash (0xC0000005 in gate M_r97)
AddStep "cpu_hiprt_nofakes" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\gates\hiprt_nofakes_check.ps1"), "-Build", $Build, "-Work", "@WORK@") { param($o, $c) $c -eq 0 -and ($o -match "^hiprt nofakes check: ok") }
# REVIEW-ARCH3 P3-3: a new hip::Arch enumerator without a device::Profile row must not compile (MSVC)
AddStep "cpu_profile_switch" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\impact\profile_switch_check.ps1"), "-Work", "@WORK@") { param($o, $c) $c -eq 0 -and ($o -match "^profile switch check: ok") }
# arch024-S11c (F-3): the CPU pins (tests/pins/check.ps1: prompt token hash of tests/pins/requests.jsonl for Swift and
# Ornith, engine page-copy / tier / restore / governor pins on the mock model, startup_replay decision table), byte for
# byte against tests/pins/expected. The S11b mutants P1 / P4 (template) and Q1 / N3 were caught by them and by no step.
AddStep "cpu_pins" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tests\pins\check.ps1"), "-Bin", $Build, "-Swift", $SW, "-Moe", $MX, "-Out", "@WORK@") { param($o, $c) $c -eq 0 -and ($o -match "(?m)^pins: ok") }
AddStep "cpu_server" $false @($exe["whirl-server-tests"], "--gguf", $SW) { param($o, $c) $c -eq 0 -and ($o -match "server tests: [1-9]\d* passed, 0 failed") }
AddStep "cpu_gate_selftest" $false @($exe["whirl-server-gate"], "--selftest") { param($o, $c) $c -eq 0 -and ($o -match ", 0 failed") }
AddStep "cpu_gate_nosuite" $false @($exe["whirl-server-gate"], "--exe", $exe["whirl-server"], "--model", $SW) { param($o, $c) $c -eq 2 }
# s8-719 review C4: whirl-server refuses a repeated --device like whirl (CPU only: refused before any model or GPU work)
AddStep "cpu_device_once" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\repro\server_device_once.ps1"), "-Build", $Build) { param($o, $c) $c -eq 0 -and ($o -match "(?m)^server_device_once: PASS") }
# s8-719 review-2 C4: every exe that restricts HIP to the chosen GPU (whirl, whirl-server, whirl-vis) answers
# --whirl-gpu-probe with a well-formed GPU list, the same list from each (an exe that forgets the dispatch silently keeps
# every GPU visible). No kernels, no model, no GPU memory: the probe only lists the GPUs, like --list-devices.
AddStep "cpu_probe_dispatch" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\repro\probe_dispatch_check.ps1"), "-Build", $Build) { param($o, $c) $c -eq 0 -and ($o -match "(?m)^probe_dispatch_check: PASS") }
AddStep "cpu_server_only_typo" $false @($exe["whirl-server-tests"], "--gguf", $SW, "--only", "no_such_test") { param($o, $c) $c -eq 2 }
AddStep "cpu_golden_selftest" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\golden\golden_selftest.ps1")) { param($o, $c) $c -eq 0 -and ($o -match "all cases as expected") }
AddStep "cpu_kl_guard_selftest" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\gates\kl_guard_selftest.ps1")) { param($o, $c) $c -eq 0 -and ($o -match "^kl guard selftest: ([1-9]\d*)/\1 as expected") }
AddStep "cpu_logscan_selftest" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\gates\server_log_scan_selftest.ps1")) { param($o, $c) $c -eq 0 -and ($o -match "server log scan selftest: \d+/\d+ as expected") }
AddStep "cpu_golden_xcheck" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\golden\golden.ps1"), "xcheck", "-Arch", $arch, "-Models", $gmodels) { param($o, $c) $c -eq 0 -and ($o -match "file\(s\) ok") }
# arch024-S12: the -Module step-selection table (tools/gates/module_gate_cases.tsv; CPU, runs this script with -List)
AddStep "cpu_module_gate" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\gates\module_gate_test.ps1"), "-Work", "@WORK@") { param($o, $c) $c -eq 0 -and ($o -match "^module gate test: ([1-9]\d*)/\1 cases ok") }
# review C7: modules.json against this tree and this gate, and the block / pass tests of that check and of impact.py's
# module mapping (python: WHIRL_GATE_PYTHON, else python on PATH; none found = the step fails)
AddStep "cpu_modules_check" $false @($pwsh, "-NoProfile", "-Command", "& '$py' -I '$(Join-Path $repo "tools\gates\modules_check_test.py")'; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; & '$py' -I '$(Join-Path $repo "tools\impact\validate_test.py")'; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; & '$py' -I '$(Join-Path $repo "tools\impact\impact_steps_test.py")'; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; & '$py' -I '$(Join-Path $repo "tools\impact\mutations_check.py")' --selftest; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; & '$py' -I '$(Join-Path $repo "tools\impact\mutations_check.py")'; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; & '$py' -I '$(Join-Path $repo "tools\gates\modules_check.py")'; exit `$LASTEXITCODE") `
    { param($o, $c) $c -eq 0 -and ($o -match "^modules check test: ([1-9]\d*)/\1 ok") -and ($o -match "^validate test: ([1-9]\d*)/\1 ok") -and ($o -match "^impact steps test: ([1-9]\d*)/\1 ok") -and ($o -match "^modules check: ok") }
# arch025-Q1 / Q3 (PLAN (e)): the C++ safety tools, CPU, each in its own build directory under the tree (build-asan,
# build-analyze), with a time cap and a block / pass proof inside the step (tools/safety/safety_step.ps1):
#   cpu_safety_selftest  block / pass unit tests of both verdicts + the two whitelists + the real-cl /analyze baseline proof
#                (analyze_fault.bat: baseline-only TU PASS, new keys FAIL, 2nd warning of a baselined key FAIL; seconds)
#   cpu_asan     ASan + leak-check build of the 10 CPU test exes; every suite and the CPU pins clean, the 8 injected faults
#                (use-after-free, overflow, double free, 3 leaks) each caught by its checker, opt-out whitelist exact
#   cpu_analyze  full MSVC /analyze rebuild; no warning beyond tools/safety/analyze_baseline.txt, every baseline key with file:line
#                + reason (analyze_whitelist.tsv), the analyze_fault.bat proof (clean PASS, planted NEW, same key NEW x1)
#   cpu_tsa      (Q4) clang-cl -Wthread-safety -Werror=thread-safety build of whirl_server_core + whirl_tier_core (build-tsa,
#                include/whirl/thread_annotations.h) clean, and each of the 7 lock removals of tools/safety/tsa/tsa_check.py must
#                fail it (a few minutes; needs the HIP SDK clang-cl, no GPU)
# They reuse the HIP code objects of -Build (same tree) for a new build directory. WHIRL_GATE_QUIET_WAIT: a local script that
# waits while a GPU measurement holds the machine (passed on as WHIRL_QUIET_WAIT; build_cfg.bat runs it before compiling).
$safetyEnv = @{ WHIRL_PYTHON = $py; WHIRL_KERNELS_FROM = $Build; WHIRL_QUIET_WAIT = $env:WHIRL_GATE_QUIET_WAIT }
$safetyPs = Join-Path $repo "tools\safety\safety_step.ps1"
AddStep "cpu_safety_selftest" $false @($pwsh, "-NoProfile", "-File", $safetyPs, "-Kind", "selftest") { param($o, $c) $c -eq 0 -and ($o -match "(?m)^cpu safety selftest: ok") } $safetyEnv
AddStep "cpu_asan" $false @($pwsh, "-NoProfile", "-File", $safetyPs, "-Kind", "asan", "-Gguf", $SW, "-GgufMoe", $MX) { param($o, $c) $c -eq 0 -and ($o -match "(?m)^cpu asan: ok") } $safetyEnv
AddStep "cpu_analyze" $false @($pwsh, "-NoProfile", "-File", $safetyPs, "-Kind", "analyze") { param($o, $c) $c -eq 0 -and ($o -match "(?m)^cpu analyze: ok") } $safetyEnv
AddStep "cpu_tsa" $false @($pwsh, "-NoProfile", "-File", $safetyPs, "-Kind", "tsa") { param($o, $c) $c -eq 0 -and ($o -match "(?m)^cpu tsa: ok") } $safetyEnv
$ktArgs = @($exe["whirl-kernel-test"], "--device", $Device, "--require-all", "--mx", $SW, "--moemx", $MX) + @(if ($hasQ) { "--q4"; $Q })
# arch025-Q2 (X-13 (a) / O8): kt_full and st_* run with the GPU-side leak count (WHIRL_LEAK_CHECK=1, include/whirl/leak_count.h):
# every kernel-test family / the whole selftest must free every hip allocation, stream and event it made. The rule also
# needs the "leak check: ok" line, so a build or a shell that drops the variable cannot pass silently. The count adds
# no HIP call (launch sequence unchanged).
$leakEnv = @{ WHIRL_LEAK_CHECK = "1" }
AddStep "kt_full" $true $ktArgs { param($o, $c) $c -eq 0 -and ($o -match "^PASS: 0 failing") -and ($o -match "--require-all: 0 required kernel") -and ($o -match "(?m)^leak check: ok \([1-9]\d* units\)") } $leakEnv
foreach ($m in @($denseKeys) + @("moe")) {
    foreach ($mode in "precise", "balance", "fast") {
        $mp = $modelOf[$m]
        AddStep "st_${m}_$mode" $true @($exe["whirl"], "selftest", $mp, "--$mode", "--device", $Device) { param($o, $c) $c -eq 0 -and ($o -match "^selftest: ok") -and ($o -match "(?m)^leak check: ok \(1 units\)") } $leakEnv
    }
}
foreach ($m in @($denseKeys) + @("moe")) {
    AddStep "golden_$m" $true @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\golden\golden.ps1"), "check", "-Arch", $arch, "-Models", $m, "-Exe", $exe["whirl"], "-GoldenRoot", "@GOLDEN@",
                         "-ServerExe", $exe["whirl-server"], "-GateExe", $exe["whirl-server-gate"], "-Port", "$($port + 4)") `
        { param($o, $c) $c -eq 0 -and ($o -match "golden check $arch`: ok") } @{ WHIRL_GOLDEN_SWIFT = $SW; WHIRL_GOLDEN_DENSE = $Q; WHIRL_GOLDEN_MOE = $MX }
}
$seqSet = @(foreach ($m in $denseKeys) { foreach ($mode in "precise", "balance", "fast") { ,@($m, $mode) } }) + @(,@("moe", "precise")) + @(,@("moe", "balance"))
foreach ($p in $seqSet) {
    $mp = $modelOf[$p[0]]
    AddStep "seq_$($p[0])_$($p[1])" $true @($exe["whirl"], "seqtest", $mp, "--decode", "24", "--$($p[1])", "--device", $Device) { param($o, $c) $c -eq 0 -and ($o -match "^seqtest: ok") }
}
$dflRule = {
    param($o, $c)
    $h = @($o | ForEach-Object { if ($_ -match "hash ([0-9a-f]{16})") { $Matches[1] } })
    $dr = @($o | ForEach-Object { if ($_ -match "drafts (\d+)/(\d+)") { [int]$Matches[1] } })
    $c -eq 0 -and $h.Count -eq 3 -and @($h | Select-Object -Unique).Count -eq 1 -and $dr.Count -ge 1 -and ($dr | Measure-Object -Minimum).Minimum -gt 0
}
if ($Device -eq "r9700" -and $hasQ) {
    foreach ($mode in "precise", "balance", "fast") {
        AddStep "dfl_$mode" $true @($exe["whirl"], "bench", $Q, "--$mode", "--draft", "dflash", "--draft-model", $DF, "--prefill", "512", "--decode", "160",
                                "--modes", "plain,dflash,dflash-ngram", "--device", $Device) $dflRule
    }
}
$gateRule = { param($o, $c) $c -eq 0 -and ($o -match "== whirl-server-gate: [1-9]\d* passed, 0 failed \(ALL PASS\)") }
function GateCmd([string]$model, [string]$suites, [string]$mode, [string[]]$extra) {
    $a = @($exe["whirl-server-gate"], "--exe", $exe["whirl-server"], "--kind", "whirl", "--model", $model, "--suite", $suites,
           "--port", "$port", "--work", "@WORK@", "--kv", "auto")
    if ($mode) { $a += @("--server-env", "WHIRL_MODE=$mode") }
    # explicit on every GPU: since 0.2.3 the default is the GPU with the most memory (the 8060S on the dev box)
    $a += @("--server-env", "WHIRL_DEVICE=$Device")
    return @($a + @($extra | Where-Object { $_ }))
}
$ssdOff = @("--server-env", "WHIRL_KV_SSD_MODE=off")
$ram8s = @(if ($Device -eq "8060s") { "--server-env"; "WHIRL_KV_RAM_MB=8192" })
AddStep "srv_balance" $true (GateCmd $SW "basic,mt_cache" "balance" $ssdOff) $gateRule
AddStep "srv_fast" $true (GateCmd $SW "basic,mt_cache" "fast" $ssdOff) $gateRule
AddStep "srv_precise_cli" $true (GateCmd $SW "basic" "precise" ($ssdOff + @("--cli", $exe["whirl"]))) $gateRule
# precise multi-turn sessions with the prefix cache == cold prefill, bitwise (COLD-RESUME; the suite sets WHIRL_MODE=precise)
AddStep "srv_mt_precise" $true (GateCmd $SW "mt_cache_precise" "" $ssdOff) $gateRule
# the same for the MoE product model (REVIEW-022 F5: the precise cache-hit guarantee is documented for every model)
AddStep "srv_mt_precise_moe" $true (GateCmd $MX "mt_cache_precise" "" $ssdOff) $gateRule
AddStep "srv_fs_swift" $true (GateCmd $SW "fast_stream" "" $ssdOff) $gateRule
AddStep "srv_fs_moe" $true (GateCmd $MX "fast_stream" "" $ssdOff) $gateRule
# the cache suites run their own SSD dirs under --work (deleted after the step)
AddStep "srv_cache_balance" $true (GateCmd $SW "tier,sys,restore_conc,pool" "balance" $ram8s) $gateRule
AddStep "srv_cache_precise" $true (GateCmd $SW "tier,sys,restore_conc,pool" "precise" $ram8s) $gateRule
# REVIEW-ARCH3 P2-2: the tier suite with WHIRL_TIER_VERIFY=1: the server reads every spill back from the device and
# compares it with the host copy, and every restore with the entry. The step passes only when the gate passes, the
# server logs show >= 1 byte-identical spill and >= 1 byte-identical restore, and no verify line says "differ"
# (a device copy short by 1/64 survived golden server: both restore paths went through the same broken copy).
# REVIEW-ARCH4 P3-4: a verify that throws now logs "kv tier verify: ... error"; the step also fails on any such
# line, and every spill ("kv tier: spill N tok to RAM") and restore ("kv tier: restored N tok") must have exactly one
# verify line (attempts == spills / restores, passes == attempts), so a verify that is skipped or swallowed fails.
$verifyRule = {
    param($o, $c, $wd, $log)
    $all = @(Get-ChildItem $wd -Filter "server_*.log" -File -ErrorAction SilentlyContinue | ForEach-Object { Get-Content $_.FullName })
    $lines = @($all | Where-Object { $_ -match "kv tier verify:" })
    $vsp = @($lines | Where-Object { $_ -match "kv tier verify: spill of \d+ tok" }).Count
    $vrs = @($lines | Where-Object { $_ -match "kv tier verify: restore of \d+ tok" }).Count
    $sp = @($lines | Where-Object { $_ -match "spill of \d+ tok byte-identical" }).Count
    $rs = @($lines | Where-Object { $_ -match "restore of \d+ tok byte-identical" }).Count
    $df = @($lines | Where-Object { $_ -match "differ" }).Count
    $er = @($lines | Where-Object { $_ -match "kv tier verify: .*error" }).Count
    $nsp = @($all | Where-Object { $_ -match "kv tier: spill \d+ tok to RAM" }).Count
    $nrs = @($all | Where-Object { $_ -match "kv tier: restored \d+ tok from" }).Count
    "tier verify: spills $nsp, verify attempts $vsp, byte-identical $sp; restores $nrs, verify attempts $vrs, byte-identical $rs; $df line(s) differ, $er error(s)" | Add-Content $log
    (& $gateRule $o $c) -and $sp -ge 1 -and $rs -ge 1 -and $df -eq 0 -and $er -eq 0 -and $vsp -eq $nsp -and $sp -eq $vsp -and $vrs -eq $nrs -and $rs -eq $vrs
}
AddStep "srv_tier_verify" $true (GateCmd $SW "tier" "balance" ($ram8s + @("--server-env", "WHIRL_TIER_VERIFY=1"))) $verifyRule
# s8-719 (hipError 719 with two GPUs), permanent: with more than one GPU, an 8192 MiB RAM tier on this GPU with
# --device pci:<bus>:<device> (and without --device when it is the default): HTTP 200, no hipError, not cut by the
# pinned-room limit (the half-of-available-RAM rule may lower it and is reported), "1 HIP GPU(s)", the startup
# line names this GPU, the other adapter's shared usage rises <= 1024 MiB. Only this gate's GPU runs (the runs are
# one after the other); the other GPU's gate is a separate release_gate invocation. One GPU: SKIP.
AddStep "srv_gpu_select" $true @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\gates\gpu_select_check.ps1"), "-Build", $Build, "-Model", $SW,
                                 "-Device", $Device, "-Port", "$($port + 3)", "-Work", "@WORK@") { param($o, $c) $c -eq 0 -and ($o -match "(?m)^gpu select check: (ok|SKIP)") }
# arch025-kvram719tag: the kvram719 repro (tools/repro/kvram719_tier.ps1) as a gate step: a RAM tier over what the
# Windows limit for GPU-locked memory leaves is lowered by the server, the request is answered, no hipError 719, the
# selected adapter stays under its limit and no other HIP adapter is charged (REVIEW-B0 A1). Exit 0 + "kvram719: PASS"
# (INVALID = no RAM tier in use = exit 3 = not a pass). Only this gate's GPU runs. Replaces the free-text "P-719" manual items.
AddStep "srv_kvram719" $true @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\repro\kvram719_tier.ps1"), "-Exe", $exe["whirl-server"], "-Model", $SW,
                               "-Device", $Device, "-Port", "$($port + 6)", "-Work", "@WORK@") { param($o, $c) $c -eq 0 -and ($o -match "(?m)^kvram719: PASS") }
# arch024-S14 (PLAN c-3) environment matrix on this gate's GPU (tools/gates/env_matrix.ps1; one server at a time, the
# other GPU only enumerated): E1 one vs two GPUs visible x default / --device + another program's pinned memory;
# E3 the other GPU hidden; E4 VRAM + pinned held by another program during server gate basic; E7 WHIRL_CKPT_HOST=1
# (golden server + gpu_select_check). The CPU tables (E2 / E5 and the selection rows) are in cpu_device / cpu_server.
# srv_* names: release_gate scans their server_*.log (wddm, owner). One GPU on the host: SKIP.
$envRule = { param($o, $c) $c -eq 0 -and ($o -match "(?m)^env matrix: (ok|SKIP)") }
function EnvCmd([string]$cell) {
    return @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\gates\env_matrix.ps1"), "-Build", $Build, "-Model", $SW, "-Device", $Device,
             "-Cells", $cell, "-Port", "$($port + 10)", "-Work", "@WORK@")
}
AddStep "srv_env_e1" $true (EnvCmd "E1") $envRule
AddStep "srv_env_e3" $true (EnvCmd "E3") $envRule
AddStep "srv_env_e4" $true (EnvCmd "E4") $envRule
AddStep "srv_env_e7" $true @((EnvCmd "E7") + "-Golden") $envRule @{ WHIRL_GOLDEN_SWIFT = $SW; WHIRL_GOLDEN_MOE = $MX }
# arch025-klguard (owner mode standard 2026-10-26): the KL gate is a REGRESSION GUARD, not a veto. kl_guard (CPU) judges
# -KlReport (klgate.py --report of this whirl.exe on this GPU) with tools/klgate/kl_guard.ps1: a balance / fast case beyond
# its guard line is a "WARN kl guard:" line with the numbers and the mode's items, no report is "TODO kl-guard:", no
# -TaskAccuracy is "TODO task-accuracy:"; all of them PASS the step (the WARN / TODO lines are copied into the summary,
# before the verdict) and the evidence JSON stays in the step's log folder. A report that is not usable evidence (another
# exe / GPU, run errors, < 3 positions, a length / model missing, a negative --as test) fails the step (exit 2). precise is
# judged by the golden / seqtest / mt_cache_precise steps (bitwise), never here.
$klArgs = @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\klgate\kl_guard.ps1"), "-Exe", $exe["whirl"], "-Arch", $arch,
            "-Modes", "balance,fast", "-Context", "release gate $Device", "-Evidence", (Join-Path (Join-Path $Work $Tag) "kl_guard_evidence.json"))
$klM = @(@($SW, $MX) | Where-Object { $_ } | ForEach-Object { Split-Path -Leaf $_ })
if ($klM.Count) { $klArgs += @("-Models", ($klM -join ",")) }
if ($KlReport) { $klArgs += @("-Report", $KlReport) }
if ($TaskAccuracy) { $klArgs += @("-TaskAccuracy", $TaskAccuracy) }
AddStep "kl_guard" $false $klArgs { param($o, $c) $c -eq 0 -and ($o -match "^kl guard: (PASS|WARN|NOT RUN) ") }

# ---------------------------------------------------------------------------------------------------
# -Module: steps from tools/gates/modules.json (arch024-S12; dev-time fast feedback only, see the header)
if ($Module.Count -gt 0) {
    if (-not $ModulesFile) { $ModulesFile = Join-Path $PSScriptRoot "modules.json" }
    $mj = $null
    try { $mj = Get-Content -Raw $ModulesFile -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { Write-Host "release gate -Module: cannot read $ModulesFile`: $_"; exit 2 }
    Write-Host "release gate -Module $($Module -join ','): DEV-TIME FAST FEEDBACK ONLY, not for releases / rc merges / bug-fix releases until owner sign-off O1/O2 (arch024 PLAN); those run the full gate on both GPUs"
    $have = @($S | ForEach-Object { $_.name })
    $treeFiles = @(& git -C $repo ls-files 2>$null)
    $msel = [System.Collections.Generic.List[string]]::new()
    $merr = @()
    $full = $false
    if ($Module.Count -gt 1) { Write-Host "release gate -Module: WARNING: $($Module.Count) modules: impact.py recommends the full gate for a change in two modules; this union is dev-time feedback only" }
    # review A4: transitive includers (src/include/kernels) of a set of files
    function Includers([string[]]$files) {
        $seen = @{}; $todo = [System.Collections.Generic.Queue[string]]::new()
        foreach ($f in $files) { $todo.Enqueue($f) }
        while ($todo.Count) {
            $f = $todo.Dequeue(); $b = [regex]::Escape([System.IO.Path]::GetFileName($f))
            $pat = '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]([^>"]*/)?' + $b + '[>"]'
            foreach ($x in @(& git -C $repo grep -l -E $pat -- src include kernels 2>$null)) {
                if (-not $seen.ContainsKey($x) -and $files -notcontains $x) { $seen[$x] = 1; if ($x -like "*.h" -or $x -like "*.hpp" -or $x -like "*.inc") { $todo.Enqueue($x) } }
            }
        }
        return @($seen.Keys | Sort-Object)
    }
    $ownerOf = { param($f) @($mj.modules.PSObject.Properties | Where-Object { $g = @($_.Value.paths | Where-Object { $_ }); @($g | Where-Object { $f -like $_ }).Count -gt 0 } | ForEach-Object { $_.Name }) }
    $expand = [System.Collections.Generic.List[string]]::new()
    foreach ($mn in $Module) {
        if (-not $expand.Contains($mn)) { $expand.Add($mn) }
        $md = $mj.modules.PSObject.Properties[$mn]
        if (-not $md) { continue }
        $md = $md.Value
        if ($md.callers -is [string] -and $md.callers -eq "rdeps") {
            $own = @($treeFiles | Where-Object { $f = $_; @(@($md.paths | Where-Object { $_ }) | Where-Object { $f -like $_ }).Count -gt 0 })
            if ($own.Count -eq 0) { Write-Host "release gate -Module: module $mn`: callers by includers, but its files are not in this tree: full gate"; $full = $true; continue }
            $inc = Includers $own
            $lone = @($inc | Where-Object { (& $ownerOf $_).Count -eq 0 })
            $cm = @($inc | ForEach-Object { & $ownerOf $_ } | Where-Object { $_ -ne $mn } | Sort-Object -Unique)
            if ($lone.Count -gt 0) { Write-Host "release gate -Module: module $mn`: $($lone.Count) includer(s) in no module ($(@($lone | Select-Object -First 4) -join ', ')): full gate"; $full = $true }
            if ($cm.Count) { Write-Host "release gate -Module: module $mn`: callers' modules (includers): $($cm -join ', ')" }
            foreach ($k in $cm) { if (-not $expand.Contains($k)) { $expand.Add($k) } }
        } elseif ($md.callers) {
            Write-Host "release gate -Module: module $mn`: callers' modules: $(@($md.callers) -join ', ')"
            foreach ($k in @($md.callers)) { if (-not $expand.Contains($k)) { $expand.Add($k) } }
        }
    }
    foreach ($mn in $expand) {
        $md = $mj.modules.PSObject.Properties[$mn]
        if (-not $md) { $merr += "unknown module '$mn' (modules: $(@($mj.modules.PSObject.Properties.Name) -join ', '))"; continue }
        $md = $md.Value
        $pend = @{}
        foreach ($p in @($md.pending | Where-Object { $_ })) {
            if ($have -contains $p.step) { $merr += "module $mn`: pending step '$($p.step)' exists in the gate now: move it to steps" } else { $pend[$p.step] = $p.lands_in }
        }
        $globs = @($md.paths | Where-Object { $_ })
        $present = @($treeFiles | Where-Object { $f = $_; @($globs | Where-Object { $f -like $_ }).Count -gt 0 })
        if ($globs.Count -gt 0 -and $present.Count -eq 0) {
            if ($md.lands_on) { Write-Host "release gate -Module: module $mn ($($md.id)) is declared, its files are not on this branch (lands on $($md.lands_on)); its steps run anyway" }
            else { $merr += "module $mn`: none of its paths ($($globs -join ', ')) is in this tree and it has no lands_on" }
        } elseif ($globs.Count -eq 0) {
            Write-Host "release gate -Module: module $mn ($($md.id)) has no paths yet (lands on $($md.lands_on))"
        }
        if ($md.full) { $full = $true; Write-Host "release gate -Module: module $mn ($($md.id)): full gate (cross-module contract)" }
        foreach ($st in @($md.steps | Where-Object { $_ })) {
            if ($have -contains $st) { if (-not $msel.Contains($st)) { $msel.Add($st) } }
            elseif ($pend.ContainsKey($st)) { }
            else { $merr += "module $mn`: step '$st' is not in this gate (-List) and not marked pending" }
        }
        foreach ($k in $pend.Keys) { Write-Host "release gate -Module: module $mn`: pending step $k (lands in $($pend[$k])): not run" }
        if ($md.when) { foreach ($w in $md.when.PSObject.Properties) { Write-Host "release gate -Module: module $mn`: when $($w.Name) changes add -Steps $(@($w.Value) -join ',')"
                foreach ($st in @($w.Value)) { if ($have -notcontains $st) { $merr += "module $mn`: when-step '$st' is not in this gate" } } } }
        if ($md.gpus) { $two = @($md.gpus.PSObject.Properties | Where-Object { $_.Value -ge 2 } | ForEach-Object { $_.Name })
            if ($two.Count) { Write-Host "release gate -Module: module $mn`: run on both GPUs, one after the other: $($two -join ', ')" } }
        foreach ($x in @($md.manual | Where-Object { $_ })) { Write-Host "release gate -Module: module $mn`: also (not in this gate): $x" }
    }
    if ($merr.Count -gt 0) { $merr | ForEach-Object { Write-Host "release gate -Module: $_" }; exit 2 }
    # review K2: the conditional steps of PLAN c-2 come from the impact map of the change, not from memory
    if (-not $ImpactRange -and -not $NoMapSteps) { Write-Host "release gate -Module: give -ImpactRange A..B (or worktree) so the impact map's steps are added (PLAN c-2), or -NoMapSteps to run the table only"; exit 2 }
    if ($ImpactRange -and $NoMapSteps) { Write-Host "release gate -Module: -ImpactRange and -NoMapSteps together"; exit 2 }
    if ($NoMapSteps) { Write-Host "release gate -Module: -NoMapSteps: the impact map's extra steps (PLAN c-2) are NOT added" }
    if ($ImpactRange) {
        $ia = if ($ImpactRange -eq "worktree") { @("--worktree") } else { @("--range", $ImpactRange) }
        $ij = $null
        try { $ij = (& $py -I (Join-Path $repo "tools\impact\impact.py") @ia --json 2>$null | Out-String) | ConvertFrom-Json -ErrorAction Stop } catch { $ij = $null }
        if (-not $ij -or $LASTEXITCODE -ne 0) { Write-Host "release gate -Module: impact.py $($ia -join ' ') failed (python $py)"; exit 2 }
        foreach ($jp in $ij.PSObject.Properties) {
            $mg = "$($jp.Value.module_gate)"
            Write-Host "release gate -Module: impact $($jp.Name): $mg"
            if ($mg -like "release_gate.ps1 -Module *") {
                $im = @(($mg -split " ")[2] -split ",")
                $miss = @($im | Where-Object { $expand -notcontains $_ })
                if ($miss.Count) { Write-Host "release gate -Module: impact.py maps the change to $($im -join ','), -Module covers $($expand -join ',') (missing: $($miss -join ','))"; exit 2 }
                if ($mg -match "-Steps (\S+)") { foreach ($st in @($Matches[1] -split ",")) { if ($Steps -notcontains $st) { $Steps += $st } } }
            } elseif ($mg -like "full gate*") { Write-Host "release gate -Module: impact.py says full gate: running the full gate"; $full = $true }
        }
    }
    $added = @($Steps | Where-Object { $msel -notcontains $_ })
    $addedGpu = @($added | Where-Object { $n = $_; @($S | Where-Object { $_.name -eq $n -and $_.gpu }).Count -gt 0 })
    if ($addedGpu.Count) { Write-Host "release gate -Module: steps added beyond the module table: run on both GPUs, one after the other: $($addedGpu -join ', ')" }
    if (-not $full -and $msel.Count -eq 0 -and $Steps.Count -eq 0) { Write-Host "release gate -Module: 0 steps selected (an empty selection must not mean the full gate)"; exit 2 }
    if ($full) { $Steps = @() } else { $Steps = @($msel) + @($Steps | Where-Object { $msel -notcontains $_ }) }
}
if ($List -and $Module.Count -gt 0) {
    foreach ($n in $Steps) { if (-not ($S | Where-Object { $_.name -eq $n })) { Write-Host "release gate: unknown step '$n' (-List shows them)"; exit 2 } }
    Write-Host "release gate -Module: tag $Tag"
    $S | Where-Object { $Steps.Count -eq 0 -or $Steps -contains $_.name } | ForEach-Object { "{0,-22} {1}" -f $_.name, ($(if ($_.gpu) { "GPU" } else { "CPU" })) }; exit 0
}
if ($List) { $S | ForEach-Object { "{0,-22} {1}" -f $_.name, ($(if ($_.gpu) { "GPU" } else { "CPU" })) }; exit 0 }

# ---------------------------------------------------------------------------------------------------
# selection and preflight (setup errors exit 2 before any step runs)
foreach ($n in $Steps) { if (-not ($S | Where-Object { $_.name -eq $n })) { Write-Host "release gate: unknown step '$n' (-List shows them)"; exit 2 } }
$sel = @($S | Where-Object { $Steps.Count -eq 0 -or $Steps -contains $_.name })
if ($sel.Count -eq 0) { Write-Host "release gate: 0 steps selected"; exit 2 }
$setup = @()
foreach ($k in $exe.Keys) { if (-not (Test-Path $exe[$k])) { $setup += "missing $($exe[$k])" } }
$need = @(@("WHIRL_GATE_SWIFT", $SW), @("WHIRL_GATE_MOE", $MX), @("WHIRL_GATE_TEXT_ROOT", $env:WHIRL_GATE_TEXT_ROOT))
if ($hasQ) { $need += @(,@("WHIRL_GATE_DENSE", $Q)); if ($Device -eq "r9700") { $need += @(,@("WHIRL_GATE_DFLASH", $DF)) } }
foreach ($p in $need) {
    if (-not $p[1] -or -not (Test-Path $p[1])) { $setup += "$($p[0]) not set or not found ($($p[1]))" }
}
$head = (& git -C $repo rev-parse HEAD 2>$null)
$dirty = @(& git -C $repo status --porcelain --untracked-files=no 2>$null).Count
if ($ExpectCommit -and -not ($head -and $head.StartsWith($ExpectCommit))) { $setup += "tree $repo is at $head, expected $ExpectCommit" }
if ($setup.Count -gt 0) { $setup | ForEach-Object { Write-Host "release gate setup: $_" }; exit 2 }
New-Item -ItemType Directory -Force $Work | Out-Null
$logDir = Join-Path $Work $Tag
New-Item -ItemType Directory -Force $logDir | Out-Null
Remove-Item (Join-Path $Work "PASS_$Tag"), (Join-Path $Work "FAIL_$Tag") -ErrorAction SilentlyContinue
$whirlTime = (Get-Item $exe["whirl"]).LastWriteTime.ToString("s")
$qnote = if ($hasQ) { "optional Q4_K_M steps on" } else { "optional Q4_K_M steps off (WHIRL_GATE_DENSE unset)" }
$mnote = if ($Module.Count -gt 0) { " module=$($Module -join ',') (dev-time module gate, not a release gate; steps $($sel.Count); map steps $(if ($NoMapSteps) { 'NOT added (-NoMapSteps)' } else { "from impact $ImpactRange" }))," } else { "" }
$hdr = "release gate $Tag on $Device ($arch),$mnote models Swift + Ornith, $qnote`: tree $repo commit $head (dirty tracked files: $dirty), exe $Build (whirl.exe $whirlTime), $(Get-Date -Format s)"
Write-Host $hdr
$golden = Join-Path $repo "tests\golden"
if ($InjectFail) {
    # a copy of the references with one step hash changed: the golden steps must fail
    $golden = Join-Path $Work "golden_injected"
    Remove-Item -Recurse -Force $golden -ErrorAction SilentlyContinue
    Copy-Item -Recurse (Join-Path $repo "tests\golden") $golden
    $f = Join-Path $golden "$arch\swift\precise_single.txt"
    if (-not (Test-Path $f)) { $f = Join-Path $golden "$arch\moe\precise_single.txt" }
    $l = @(Get-Content $f)
    $l = @($l | ForEach-Object { if ($_ -like "G short.step01 *") { $_ -replace "lh=[0-9a-f]+", "lh=0123456789abcdef" } else { $_ } })
    [System.IO.File]::WriteAllLines($f, [string[]]$l)
    Write-Host "release gate: -InjectFail: golden references copied to $golden with $f short.step01 lh changed"
}

# clean environment for the steps: no stray WHIRL_* (a WHIRL_MODE / WHIRL_KV in the shell would change
# what is tested); restored at the end
$saved = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -like "WHIRL_*" -and $_.Name -notlike "WHIRL_GATE_*" -and $_.Name -ne "WHIRL_R9700_LOCK_HELD" })) {
    $saved[$v.Name] = $v.Value; [Environment]::SetEnvironmentVariable($v.Name, $null)
}
if (-not $env:AMD_LOG_LEVEL) { $env:AMD_LOG_LEVEL = "1" }
$env:PYTHONIOENCODING = "utf-8"
# the tier / server unit tests keep their SSD folders under WHIRL_TEST_TMP, else %TEMP%\whirl-tests: two gates at
# once (one per GPU, or two agents) shared that folder and broke each other's cpu_tier / cpu_server (rel0221 rc3,
# 2026-10-21 03:56: tier_tests.cpp:600 crash + close_reused). One folder per gate Work, removed afterwards.
$testTmp = Join-Path $Work "test_tmp"
Remove-Item -Recurse -Force $testTmp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $testTmp | Out-Null
$env:WHIRL_TEST_TMP = $testTmp
$lock = $env:WHIRL_GATE_LOCK_SCRIPT
$rows = @()
$notes = @()
try {
    foreach ($s in $sel) {
        $t0 = Get-Date
        $wd = Join-Path $Work "work_$($s.name)"
        $cmd = @($s.cmd | ForEach-Object { $_ -replace "@WORK@", $wd -replace "@GOLDEN@", $golden })
        $log = Join-Path $logDir "$($s.name).txt"
        $c = -1
        foreach ($k in $s.env.Keys) { [Environment]::SetEnvironmentVariable($k, $s.env[$k]) }
        if ($LaunchLog -and $s.gpu) {
            $ld = Join-Path $LaunchLog $s.name
            New-Item -ItemType Directory -Force $ld | Out-Null
            $env:WHIRL_LAUNCH_LOG = $ld
            # arch024-S0b: whirl-server-gate gives its servers no WHIRL_* variable of its own environment (except
            # WHIRL_DEVICE / WHIRL_VRAM_LIMIT_MB / WHIRL_MODE), so the servers get the folder with --server-env
            if ($cmd[0] -eq $exe["whirl-server-gate"]) { $cmd += @("--server-env", "WHIRL_LAUNCH_LOG=$ld") }
        }
        try {
            # hold the R9700 lock for this step only, in this process (a child pwsh in between lost the
            # server gate's output after its first Ctrl+C server stop)
            $held = $null
            $s8held = $false
            if ($s.gpu -and $lock -and -not $env:WHIRL_R9700_LOCK_HELD -and $Device -eq "r9700") {
                . $lock
                $held = Enter-R9700Lock
            }
            # 8060s: the R9700 lock is not this GPU's; mark it held so the server gate does not wait for the R9700
            if ($s.gpu -and $Device -eq "8060s" -and -not $env:WHIRL_R9700_LOCK_HELD) {
                $env:WHIRL_R9700_LOCK_HELD = "release_gate-8060s"; $s8held = $true
            }
            try {
                & $cmd[0] @($cmd | Select-Object -Skip 1) *> $log
                $c = $LASTEXITCODE
            } finally {
                if ($held) { $held.Close(); Remove-Item Env:WHIRL_R9700_LOCK_HELD -ErrorAction SilentlyContinue }
                if ($s8held) { Remove-Item Env:WHIRL_R9700_LOCK_HELD -ErrorAction SilentlyContinue }
            }
        } catch {
            "release gate: step did not run: $_" | Add-Content $log
            $c = -1
        }
        foreach ($k in $s.env.Keys) { [Environment]::SetEnvironmentVariable($k, $null) }
        Remove-Item Env:WHIRL_LAUNCH_LOG -ErrorAction SilentlyContinue
        $out = @(Get-Content $log -ErrorAction SilentlyContinue)
        $ok = $false
        try { $ok = [bool](& $s.rule $out $c $wd $log) } catch { $ok = $false }
        $out = @(Get-Content $log -ErrorAction SilentlyContinue)  # a rule may append its own summary line
        # v0.2.2.1 (issues #5 / #6): a normal server run (no other program taking VRAM) must stay inside the WDDM
        # budget: any server log of a server step that shows "over the WDDM budget" or a GPU memory spill
        # warning fails the step. A server step with no server log to read fails too (the check must not pass
        # vacuously). v0.2.2.2 (REVIEW-GOV-1 P2-3): the memory governor repairs a drift under the margin before it
        # becomes a spill, which would hide a budget regression, so a governor episode start ("memory governor: N GiB
        # free under the Windows video memory") or a KV pool re-allocation fails a server step too: gate runs have no
        # other program taking VRAM.
        # arch024-S10 (PLAN M8, C-b1): the scan is tools/gates/server_log_scan.ps1, shared with golden.ps1 and the GPU
        # runners. srv_* steps: the WDDM lines above and owner check violations ("E owner check:", a call into the engine
        # state from another thread; docs/dev/threads.md). golden_swift / golden_moe: golden.ps1 scans each server case
        # itself (its passing cases delete their server logs), so the step needs its "server log scan: PASS" lines.
        $wddmNote = ""
        if ($s.name -like "srv_*") {
            $scan = @(& (Join-Path $PSScriptRoot "server_log_scan.ps1") -Dir $wd -Step $s.name *>&1 | ForEach-Object { "$_" })
            if ($LASTEXITCODE -ne 0) { $ok = $false }
            $wddmNote = " | $("$($scan[0])" -replace '^server log scan: ', 'log scan: ')"
            $scan | ForEach-Object { "release gate: $_" } | Add-Content $log
        } elseif ($s.name -in @("golden_swift", "golden_moe")) {
            $gp = @($out | Where-Object { $_ -match "server log scan: PASS" }).Count
            $gf = @($out | Where-Object { $_ -match "server log scan: FAIL" }).Count
            if ($gp -lt 1 -or $gf -gt 0) { $ok = $false }
            $wddmNote = " | log scan: golden server cases $gp PASS, $gf FAIL"
        }
        # CB-FLAKE: a step keeps its server logs (server_*.log / *.stdout.txt, small) next to the step log, so a
        # failure can be explained afterwards and a pass can be audited (REVIEW-GOV-1 P2-3: whether a governor episode
        # fired in a gate run could not be checked once the logs were gone); the SSD tier files are deleted either way
        if (Test-Path $wd) {
            $keep = Join-Path $logDir "$($s.name)_server_logs"
            $kl = @(Get-ChildItem $wd -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "server_*.log" -or $_.Name -like "*.stdout.txt" -or $_.Name -like "*.json" })
            if ($kl.Count -gt 0) {
                New-Item -ItemType Directory -Force $keep | Out-Null
                $kl | Copy-Item -Destination $keep -ErrorAction SilentlyContinue
            }
        }
        Remove-Item -Recurse -Force $wd -ErrorAction SilentlyContinue  # server gate work dirs (SSD tier files)
        $sec = [int]((Get-Date) - $t0).TotalSeconds
        $last = ($out | Where-Object { $_ -match "\S" -and $_ -notmatch "^:1:" } | Select-Object -Last 1)
        $row = "{0,-4} {1,-22} exit {2,3}  {3,5} s  {4}{5}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $s.name, $c, $sec, "$last".Trim(), $wddmNote
        Write-Host $row
        $rows += [pscustomobject]@{ ok = $ok; line = $row }
        # arch025-klguard: the regression-guard warnings and the open TODOs (task accuracy, a guard not run) are printed
        # and kept in the summary (before the verdict line; release_check reads only the PASS / FAIL rows and the verdict)
        foreach ($wl in @($out | Where-Object { $_ -match "^(WARN kl guard|TODO task-accuracy|TODO kl-guard):" })) {
            Write-Host "     $wl"
            $notes += "$($s.name): $wl"
        }
    }
} finally {
    Remove-Item Env:WHIRL_TEST_TMP -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $testTmp -ErrorAction SilentlyContinue
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
}
$nf = @($rows | Where-Object { -not $_.ok }).Count
$verdict = if ($rows.Count -eq 0) { "FAIL (no step ran)" } elseif ($nf -gt 0) { "FAIL ($nf of $($rows.Count) steps failed)" } else { "PASS ($($rows.Count) steps)" }
$summary = @($hdr) + @($rows | ForEach-Object { $_.line }) + @($notes) + @("release gate $Tag`: $verdict")
[System.IO.File]::WriteAllLines((Join-Path $Work "${Tag}_summary.txt"), [string[]]$summary)
Write-Host "release gate $Tag`: $verdict  (logs: $logDir)"
if ($rows.Count -gt 0 -and $nf -eq 0) { "PASS $(Get-Date -Format s) $head" | Set-Content (Join-Path $Work "PASS_$Tag"); exit 0 }
"FAIL $(Get-Date -Format s) $head $verdict" | Set-Content (Join-Path $Work "FAIL_$Tag")
exit 1
