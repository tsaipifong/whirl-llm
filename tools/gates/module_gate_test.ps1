# SPDX-License-Identifier: Apache-2.0
# Step-selection table test for release_gate.ps1 -Module (arch024-S12). CPU only, no exe, no model, no GPU:
# every case runs `release_gate.ps1 ... -List` (which exits before the preflight) and compares the exit code, the
# listed steps and one output line with tools/gates/module_gate_cases.tsv (hand-written from arch024 PLAN c-1).
# Block cases use small modules files written here; pass cases use tools/gates/modules.json.
#   pwsh tools/gates/module_gate_test.ps1 [-Work DIR] [-Gate PATH] [-Cases PATH]
# Exit 0 = every case as expected ("module gate test: N/N cases ok"), 1 = a case differs, 2 = setup error.
param([string]$Work = "", [string]$Gate = "", [string]$Cases = "")
# arch025-c2fu: set or REMOVE a process environment variable. pwsh 7.5+ (.NET 9+) keeps a variable DEFINED AS EMPTY for
# [Environment]::SetEnvironmentVariable(k, $null) (PowerShell passes $null as "") and for (k, ""): a restore of an unset
# variable then leaves k= for every later child (arch024-S14 / arch025-c2ctx: HIP_VISIBLE_DEVICES= hid every GPU).
# $null = remove; any other value (also "") = set. Checked by tools/gates/env_restore_check.ps1.
function SetEnvOrRemove([string]$k, $v) {
    if ($null -eq $v) { Remove-Item -LiteralPath "env:$k" -ErrorAction SilentlyContinue } else { [Environment]::SetEnvironmentVariable($k, "$v") }
}
$ErrorActionPreference = "Stop"
if (-not $Gate) { $Gate = Join-Path $PSScriptRoot "release_gate.ps1" }
if (-not $Cases) { $Cases = Join-Path $PSScriptRoot "module_gate_cases.tsv" }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-module-gate-test" }
if (-not (Test-Path $Gate) -or -not (Test-Path $Cases)) { Write-Host "module gate test: missing $Gate or $Cases"; exit 2 }
New-Item -ItemType Directory -Force $Work | Out-Null
$pwsh = (Get-Process -Id $PID).Path

# fixtures: one module "m" each; paths that exist in every WHIRL tree (src/server/engine.cpp) unless the case is about missing paths
$fx = @{
    bad_step        = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":["cpu_server","no_such_step"]}}}'
    stale_pending   = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":["cpu_server"],"pending":[{"step":"cpu_server","lands_in":"S99"}]}}}'
    missing_nolands = '{"modules":{"m":{"id":"T","paths":["src/no_such_dir/x.*"],"steps":["cpu_server"]}}}'
    missing_lands   = '{"modules":{"m":{"id":"T","paths":["src/no_such_dir/x.*"],"lands_on":"arch024-s99","steps":["cpu_server"]}}}'
    zero_steps      = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":[],"pending":[{"step":"cpu_future_check","lands_in":"S99"}]}}}'
    pending_ok      = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":["cpu_server","cpu_future_check"],"pending":[{"step":"cpu_future_check","lands_in":"S99"}]}}}'
    bad_when        = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":["cpu_server"],"when":{"include/whirl/prefill_schedule.h":["no_such_step"]}}}}'
    bad_json        = '{"modules": {'
    rdeps_ok        = '{"modules":{"h":{"id":"T","paths":["src/device/handles.h"],"callers":"rdeps","steps":["cpu_device"]},"rest":{"id":"R","paths":["src/server/*","src/cache/*","src/tier/*","src/device/null_backend.*","src/device/backend*","include/whirl/*","src/model/*","src/vision/*","src/cli/*","src/release/*","src/config/*"],"steps":["cpu_server"]}}}'
    callers_list    = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"callers":["c"],"steps":["cpu_server"]},"c":{"id":"C","paths":["src/cache/*"],"steps":["cpu_cache"]}}}'
    callers_bad     = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"callers":["nope"],"steps":["cpu_server"]}}}'
    one_cpu         = '{"modules":{"m":{"id":"T","paths":["src/server/engine.cpp"],"steps":["cpu_server"]}}}'
}
foreach ($k in $fx.Keys) { [System.IO.File]::WriteAllText((Join-Path $Work "$k.json"), $fx[$k]) }

function RunGate([string[]]$a) {
    $o = @(& $pwsh -NoProfile -File $Gate -Device r9700 @a 2>&1 | ForEach-Object { "$_" })
    return [pscustomobject]@{ out = $o; code = $LASTEXITCODE }
}
function Listed($out) { @($out | Where-Object { $_ -match "^(\S+)\s+(CPU|GPU)\s*$" } | ForEach-Object { ($_ -split "\s+")[0] }) }

$lst = (RunGate @("-List")).out
$all = Listed $lst
if ($all.Count -lt 10) { Write-Host "module gate test: plain -List gave $($all.Count) steps"; exit 2 }
# "@cpu" in a steps cell = every cpu_* step of the gate in gate order (arch025-srctext: impact always selects them all), so a
# new CPU step does not break the hand-written GPU part of a case (2026-10-28: K2 pass went stale after srctext)
$cpu = @($lst | Where-Object { $_ -match "^(cpu_\S+)\s+CPU\s*$" } | ForEach-Object { ($_ -split "\s+")[0] })
$rows = @(Get-Content $Cases | Where-Object { $_ -and -not $_.StartsWith("#") } | Select-Object -Skip 1)
if ($rows.Count -eq 0) { Write-Host "module gate test: no cases in $Cases"; exit 2 }
$ok = 0; $bad = 0
foreach ($r in $rows) {
    $c = $r -split "`t"
    if ($c.Count -lt 6) { Write-Host "module gate test: bad row: $r"; exit 2 }
    $name, $mf, $argl, $wantExit, $wantSteps, $wantLine = $c[0..5]
    $a = @($argl -split " " | Where-Object { $_ })
    if ($mf -ne "repo") { $a += @("-ModulesFile", (Join-Path $Work "$mf.json")) }
    if ($a -notcontains "-List") { $a += "-List" }
    $res = RunGate $a
    $why = @()
    if ("$($res.code)" -ne $wantExit) { $why += "exit $($res.code), want $wantExit" }
    $got = Listed $res.out
    if ($wantSteps -ne "-") {
        $want = if ($wantSteps -eq "*") { $all } else { @($wantSteps -split "," | ForEach-Object { if ($_ -eq "@cpu") { $cpu } else { $_ } }) }
        if (($got -join ",") -ne ($want -join ",")) { $why += "steps [$($got -join ',')], want [$($want -join ',')]" }
    }
    if ($wantLine -ne "-" -and -not ($res.out | Where-Object { $_ -match $wantLine })) { $why += "no line matching '$wantLine'" }
    if ($why.Count -eq 0) { $ok++; Write-Host "ok   $name" }
    else { $bad++; Write-Host "FAIL $name`: $($why -join '; ')"; $res.out | Select-Object -First 12 | ForEach-Object { Write-Host "       | $_" } }
}

# review A5: a real (not -List) module run: summary header names the module, markers are module_*, never *gate_<device>.
# Fake build: every exe is a copy of whoami.exe (exits non-zero at once on unknown options): the step fails, fast, CPU.
$fb = Join-Path $Work "fakebuild"; New-Item -ItemType Directory -Force $fb | Out-Null
$who = Join-Path $env:SystemRoot "System32\whoami.exe"
foreach ($n in "whirl", "whirl-server", "whirl-server-gate", "whirl-kernel-test", "whirl-tests", "whirl-model-tests", "whirl-cache-tests",
               "whirl-vision-tests", "whirl-tier-tests", "whirl-server-tests", "whirl-hiprt-tests", "whirl-device-tests", "whirl-ops-tests",
               "whirl-pins", "whirl-startup-replay", "whirl-sampling-tests") {
    Copy-Item $who (Join-Path $fb "$n.exe") -Force
}
$dm = Join-Path $Work "dummy.gguf"; "x" | Set-Content $dm
$gw = Join-Path $Work "gw"
$keep = @{}; foreach ($v in "WHIRL_GATE_SWIFT", "WHIRL_GATE_MOE", "WHIRL_GATE_TEXT_ROOT", "WHIRL_GATE_DENSE") { $keep[$v] = [Environment]::GetEnvironmentVariable($v) }
$env:WHIRL_GATE_SWIFT = $dm; $env:WHIRL_GATE_MOE = $dm; $env:WHIRL_GATE_TEXT_ROOT = $Work; Remove-Item Env:WHIRL_GATE_DENSE -ErrorAction SilentlyContinue
try {
    $o = @(& $pwsh -NoProfile -File $Gate -Device r9700 -Build $fb -Work $gw -Module m -NoMapSteps -ModulesFile (Join-Path $Work "one_cpu.json") 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
} finally { foreach ($v in $keep.Keys) { SetEnvOrRemove $v $keep[$v] } }
$sum = Join-Path $gw "module_m_r9700_summary.txt"
$why = @()
if ($code -ne 1) { $why += "exit $code, want 1 (the fake step fails)" }
if (-not (Test-Path $sum)) { $why += "no $sum" }
elseif (-not ((Get-Content $sum -TotalCount 1) -match "module=m \(dev-time module gate.*map steps NOT added \(-NoMapSteps\)")) { $why += "summary header does not say module=m ... map steps NOT added" }
if (-not (Test-Path (Join-Path $gw "FAIL_module_m_r9700"))) { $why += "no FAIL_module_m_r9700 marker" }
$gm = @(Get-ChildItem $gw -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*gate_r9700*" })
if ($gm.Count) { $why += "full-gate marker / summary written: $($gm.Name -join ', ')" }
if ($why.Count -eq 0) { $ok++; Write-Host "ok   real module run: header module=m, marker FAIL_module_m_r9700, no *gate_r9700* file" }
else { $bad++; Write-Host "FAIL real module run: $($why -join '; ')"; $o | Select-Object -Last 8 | ForEach-Object { Write-Host "       | $_" } }

# review item 3: with WHIRL_GATE_PYTHON unset and no python on PATH the gate still finds the workspace interpreter
# (pass); an explicit WHIRL_GATE_PYTHON that does not exist is authoritative and fails the impact call (block)
$pyref = "C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
if (Test-Path $pyref) {
    $keepP = $env:PATH; $keepW = $env:WHIRL_GATE_PYTHON
    try {
        Remove-Item Env:WHIRL_GATE_PYTHON -ErrorAction SilentlyContinue
        $env:PATH = (@($env:PATH -split ";" | Where-Object { $_ -and -not (Test-Path (Join-Path $_ "python.exe")) -and -not (Test-Path (Join-Path $_ "py.exe")) }) -join ";")
        $o1 = @(& $pwsh -NoProfile -File $Gate -Device r9700 -Module tier.kv -ImpactRange "bf31527^..bf31527" -List 2>&1 | ForEach-Object { "$_" }); $c1 = $LASTEXITCODE
        $env:WHIRL_GATE_PYTHON = Join-Path $Work "no_python.exe"
        $o2 = @(& $pwsh -NoProfile -File $Gate -Device r9700 -Module tier.kv -ImpactRange "bf31527^..bf31527" -List 2>&1 | ForEach-Object { "$_" }); $c2 = $LASTEXITCODE
    } finally { $env:PATH = $keepP; if ($keepW) { $env:WHIRL_GATE_PYTHON = $keepW } else { Remove-Item Env:WHIRL_GATE_PYTHON -ErrorAction SilentlyContinue } }
    if ($c1 -eq 0 -and ($o1 | Where-Object { $_ -match "^srv_balance\s+GPU" })) { $ok++; Write-Host "ok   python: no python on PATH -> the pyref interpreter runs impact.py" }
    else { $bad++; Write-Host "FAIL python: no python on PATH: exit $c1"; $o1 | Select-Object -Last 4 | ForEach-Object { Write-Host "       | $_" } }
    if ($c2 -eq 2 -and ($o2 | Where-Object { $_ -match "impact.py .* failed" })) { $ok++; Write-Host "ok   python: WHIRL_GATE_PYTHON set to a missing file -> exit 2" }
    else { $bad++; Write-Host "FAIL python: missing WHIRL_GATE_PYTHON: exit $c2" }
} else { Write-Host "skip python default test: $pyref not on this machine" }

Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
Write-Host "module gate test: $ok/$($ok + $bad) cases ok"
if ($bad -gt 0) { exit 1 }
exit 0
