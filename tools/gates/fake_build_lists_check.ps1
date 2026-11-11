# fake_build_lists_check.ps1 -- v0252-b3gate: every exe that tools/gates/release_gate.ps1 registers ($exe[$n], a missing
# one is "release gate setup: missing ...") must also be in the fake build of each selftest that runs release_gate.ps1
# against a fake build directory. v0252-ktrng (204a2dc) registered whirl-kt-seed-tests and broke three selftests at once
# (batch b3 gate: cpu_step_selftest flow case, cpu_kl_guard_selftest 8 cases, cpu_module_gate real-module case).
#   pwsh tools/gates/fake_build_lists_check.ps1 [-Repo DIR] [-SelfTest]
# Last line "fake build lists check: ok (N exes, M selftests)" exit 0, or "fake build lists check: FAIL (...)" exit 1;
# -SelfTest: "fake build lists selftest: N/M as expected".
param([string]$Repo = (Join-Path $PSScriptRoot "..\.."), [switch]$SelfTest)
$ErrorActionPreference = "Stop"
$fakes = @("tools\gates\gate_flow_selftest.ps1", "tools\gates\kl_guard_selftest.ps1", "tools\gates\module_gate_test.ps1")

function Names([string]$list) { @([regex]::Matches($list, '"([A-Za-z0-9_.-]+)"') | ForEach-Object { $_.Groups[1].Value }) }
# release_gate.ps1: every  foreach ($n in "a", "b", ...) { $exe[$n] = ... }
function GateExes([string]$text) {
    $r = @()
    foreach ($m in [regex]::Matches($text, 'foreach\s*\(\s*\$n\s+in\s+([^)]*)\)\s*\{\s*\$exe\[\$n\]\s*=')) { $r += Names $m.Groups[1].Value }
    return @($r | Select-Object -Unique)
}
# a selftest's fake build: the foreach ($n in ...) list that names "whirl-pins" (the registered-exe list)
function FakeExes([string]$text) {
    $r = @()
    foreach ($m in [regex]::Matches($text, 'foreach\s*\(\s*\$n\s+in\s+([^)]*)\)\s*\{')) {
        $n = Names $m.Groups[1].Value
        if ($n -contains "whirl-pins") { $r += $n }
    }
    return @($r | Select-Object -Unique)
}
function Judge([string]$gateText, [hashtable]$fakeTexts) {
    $want = GateExes $gateText
    $bad = @()
    if ($want.Count -lt 5) { $bad += "release_gate.ps1: only $($want.Count) registered exe(s) parsed (parser broken?)" }
    foreach ($k in $fakeTexts.Keys) {
        $have = FakeExes $fakeTexts[$k]
        if ($have.Count -eq 0) { $bad += "${k}: no fake build list (foreach naming whirl-pins) found"; continue }
        foreach ($w in $want) { if ($have -notcontains $w) { $bad += "${k}: fake build lacks $w.exe (release_gate.ps1 registers it: setup error 'missing ...$w.exe')" } }
    }
    return [pscustomobject]@{ want = $want; bad = $bad }
}

if ($SelfTest) {
    $g = 'foreach ($n in "whirl", "whirl-pins", "a1", "a2",' + "`n" + '   "a3") { $exe[$n] = Join-Path $Build "$n.exe" }' + "`n" +
         'foreach ($n in "b1", "b2") { $exe[$n] = Join-Path $Build "$n.exe" }'
    $full = 'foreach ($n in "whirl", "whirl-pins", "a1", "a2", "a3",' + "`n" + '  "b1", "b2") { [IO.File]::WriteAllText("x", "y") }'
    $lack = 'foreach ($n in "whirl", "whirl-pins", "a1", "a2", "a3", "b1") { Copy-Item $who "x" }'
    $other = 'foreach ($n in "q") { }'
    $ok = 0; $all = 0
    foreach ($c in @(
        @("complete fake builds -> ok", $g, @{ x = $full; y = $full }, 0),
        @("one fake build lacks the 2nd list's last exe (the b3 ktrng case) -> FAIL", $g, @{ x = $full; y = $lack }, 1),
        @("a selftest with no fake build list -> FAIL", $g, @{ x = $full; y = $other }, 1),
        @("release_gate parse finds nothing -> FAIL", "nothing", @{ x = $full }, 1))) {
        $all++; $j = Judge $c[1] $c[2]; $got = if ($j.bad.Count) { 1 } else { 0 }
        if ($got -eq $c[3]) { $ok++; Write-Host "ok   $($c[0])" } else { Write-Host "BAD  $($c[0]): got $got ($($j.bad -join '; '))" }
    }
    Write-Host "fake build lists selftest: $ok/$all as expected"
    exit $(if ($ok -eq $all) { 0 } else { 1 })
}

$gateFile = Join-Path $Repo "tools\gates\release_gate.ps1"
$texts = @{}
foreach ($f in $fakes) { $p = Join-Path $Repo $f; $texts[$f] = if (Test-Path $p) { [IO.File]::ReadAllText($p) } else { "" } }
$j = Judge ([IO.File]::ReadAllText($gateFile)) $texts
foreach ($b in $j.bad) { Write-Host "FAIL $b" }
if ($j.bad.Count) { Write-Host "fake build lists check: FAIL ($($j.bad.Count) problem(s); $($j.want.Count) registered exes, $($fakes.Count) selftests)"; exit 1 }
Write-Host "fake build lists check: ok ($($j.want.Count) exes, $($fakes.Count) selftests)"
exit 0
