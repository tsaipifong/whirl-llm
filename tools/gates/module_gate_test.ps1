# SPDX-License-Identifier: Apache-2.0
# Step-selection table test for release_gate.ps1 -Module (arch024-S12). CPU only, no exe, no model, no GPU:
# every case runs `release_gate.ps1 ... -List` (which exits before the preflight) and compares the exit code, the
# listed steps and one output line with tools/gates/module_gate_cases.tsv (hand-written from arch024 PLAN c-1).
# Block cases use small modules files written here; pass cases use tools/gates/modules.json.
#   pwsh tools/gates/module_gate_test.ps1 [-Work DIR] [-Gate PATH] [-Cases PATH]
# Exit 0 = every case as expected ("module gate test: N/N cases ok"), 1 = a case differs, 2 = setup error.
param([string]$Work = "", [string]$Gate = "", [string]$Cases = "")
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
}
foreach ($k in $fx.Keys) { [System.IO.File]::WriteAllText((Join-Path $Work "$k.json"), $fx[$k]) }

function RunGate([string[]]$a) {
    $o = @(& $pwsh -NoProfile -File $Gate -Device r9700 @a 2>&1 | ForEach-Object { "$_" })
    return [pscustomobject]@{ out = $o; code = $LASTEXITCODE }
}
function Listed($out) { @($out | Where-Object { $_ -match "^(\S+)\s+(CPU|GPU)\s*$" } | ForEach-Object { ($_ -split "\s+")[0] }) }

$all = Listed (RunGate @("-List")).out
if ($all.Count -lt 10) { Write-Host "module gate test: plain -List gave $($all.Count) steps"; exit 2 }
$rows = @(Get-Content $Cases | Where-Object { $_ -and -not $_.StartsWith("#") } | Select-Object -Skip 1)
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
        $want = if ($wantSteps -eq "*") { $all } else { @($wantSteps -split ",") }
        if (($got -join ",") -ne ($want -join ",")) { $why += "steps [$($got -join ',')], want [$($want -join ',')]" }
    }
    if ($wantLine -ne "-" -and -not ($res.out | Where-Object { $_ -match $wantLine })) { $why += "no line matching '$wantLine'" }
    if ($why.Count -eq 0) { $ok++; Write-Host "ok   $name" }
    else { $bad++; Write-Host "FAIL $name`: $($why -join '; ')"; $res.out | Select-Object -First 12 | ForEach-Object { Write-Host "       | $_" } }
}
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
Write-Host "module gate test: $ok/$($ok + $bad) cases ok"
if ($bad -gt 0) { exit 1 }
exit 0
