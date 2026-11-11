# fake_build_lists_check.ps1 -- the selftests that run tools/gates/release_gate.ps1 against a fake build directory must
# make that fake build from the gate's own registered exe list (release_gate.ps1 -ListExes), never from a hand copy.
# History: v0252-b3gate (V1-59): ktrng (204a2dc) registered whirl-kt-seed-tests and broke three selftests at once
# (cpu_step_selftest flow case, cpu_kl_guard_selftest 8 cases, cpu_module_gate real-module case); fixed by adding the name
# to the three hand lists and this check compared the lists. v0252-b4fix (V1-98): moedf16b (0a6309c) registered
# whirl-kt-f16tie-tests and broke the same three again (this check was in cpu_step_selftest, which batch b4's gate never
# reached). Now there is no list to keep in step: the three selftests read -ListExes, and this check fails if
#   - release_gate.ps1 -ListExes fails, prints < 5 names, or misses a name the gate's  foreach ($n in ...) { $exe[$n] = ... }
#     tables register (the setup check "release gate setup: missing ..." iterates $exe),
#   - a fake-build selftest (gate_flow_selftest, kl_guard_selftest, module_gate_test) does not call -ListExes and loop over it,
#   - any .ps1 under tools\ or tests\ (outside release_gate.ps1 and this file) holds a hand-written exe list again
#     (a foreach ($n in "...", ...) list that names "whirl-pins").
#   pwsh tools/gates/fake_build_lists_check.ps1 [-Repo DIR] [-SelfTest]
# Last line "fake build lists check: ok (N exes, M selftests, K scripts scanned)" exit 0, or
# "fake build lists check: FAIL (...)" exit 1; -SelfTest: "fake build lists selftest: N/M as expected".
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
# a hand-written fake exe list: a foreach ($n in "...", ...) list that names "whirl-pins"
function HandList([string]$text) {
    foreach ($m in [regex]::Matches($text, 'foreach\s*\(\s*\$n\s+in\s+("[^)]*)\)\s*\{')) {
        if ((Names $m.Groups[1].Value) -contains "whirl-pins") { return $true }
    }
    return $false
}
# a fake-build selftest that reads the gate's list: calls release_gate -ListExes into $gateExes and loops over it
function UsesListExes([string]$text) { ($text -match '-File\s+\$[Gg]ate\s+-ListExes') -and ($text -match 'foreach\s*\(\s*\$n\s+in\s+\$gateExes\s*\)') }
# $listed: release_gate.ps1 -ListExes output lines; $listCode: its exit code; $fakeTexts / $otherTexts: file -> text
function Judge([string]$gateText, [string[]]$listed, [int]$listCode, [hashtable]$fakeTexts, [hashtable]$otherTexts) {
    $want = GateExes $gateText
    $bad = @()
    if ($want.Count -lt 5) { $bad += "release_gate.ps1: only $($want.Count) registered exe(s) parsed (parser broken?)" }
    if ($listCode -ne 0) { $bad += "release_gate.ps1 -ListExes: exit $listCode" }
    elseif ($listed.Count -lt 5) { $bad += "release_gate.ps1 -ListExes: only $($listed.Count) name(s)" }
    foreach ($w in $want) { if ($listed -notcontains $w) { $bad += "release_gate.ps1 -ListExes lacks $w (registered in an `$exe table: the setup check requires $w.exe)" } }
    foreach ($k in $fakeTexts.Keys) {
        if (-not (UsesListExes $fakeTexts[$k])) { $bad += "${k}: does not build its fake exes from release_gate.ps1 -ListExes (foreach (`$n in `$gateExes))" }
        if (HandList $fakeTexts[$k]) { $bad += "${k}: holds a hand-written fake exe list (names whirl-pins): use -ListExes" }
    }
    foreach ($k in $otherTexts.Keys) { if (HandList $otherTexts[$k]) { $bad += "${k}: holds a hand-written exe list (names whirl-pins): a fake build must use release_gate.ps1 -ListExes" } }
    return [pscustomobject]@{ want = $want; bad = $bad }
}

if ($SelfTest) {
    $g = 'foreach ($n in "whirl", "whirl-pins", "a1", "a2",' + "`n" + '   "a3") { $exe[$n] = Join-Path $Build "$n.exe" }' + "`n" +
         'foreach ($n in "b1", "b2") { $exe[$n] = Join-Path $Build "$n.exe" }'
    $all7 = @("a1", "a2", "a3", "b1", "b2", "whirl", "whirl-pins")
    $good = '$gateExes = @(& $pwsh -NoProfile -File $gate -ListExes 2>&1)' + "`n" + 'foreach ($n in $gateExes) { [IO.File]::WriteAllText("x", "y") }'
    $goodG = '$gateExes = @(& $pwsh -NoProfile -File $Gate -ListExes 2>&1)' + "`n" + 'foreach ($n in $gateExes) {' + "`n" + '  Copy-Item $who "x" }'
    $hand = 'foreach ($n in "whirl", "whirl-pins", "a1", "a2", "a3", "b1") { Copy-Item $who "x" }'
    $both = $good + "`n" + $hand
    $none = 'foreach ($n in "q") { }'
    $ok = 0; $all = 0
    foreach ($c in @(
        @("fake builds from -ListExes, -ListExes complete -> ok", $g, $all7, 0, @{ x = $good; y = $goodG }, @{ z = $none }, 0),
        @("a selftest with a hand list that lacks the 2nd table's last exe (the b3 kt-seed / b4 kt-f16tie case) -> FAIL", $g, $all7, 0, @{ x = $good; y = $hand }, @{}, 1),
        @("a selftest that calls -ListExes but also keeps a hand list -> FAIL", $g, $all7, 0, @{ x = $good; y = $both }, @{}, 1),
        @("a selftest with no fake build list at all -> FAIL", $g, $all7, 0, @{ x = $good; y = $none }, @{}, 1),
        @("-ListExes misses a registered exe -> FAIL", $g, @("a1", "a2", "a3", "b1", "whirl", "whirl-pins"), 0, @{ x = $good }, @{}, 1),
        @("-ListExes exits non-zero -> FAIL", $g, $all7, 2, @{ x = $good }, @{}, 1),
        @("another script brings a hand-written exe list back -> FAIL", $g, $all7, 0, @{ x = $good }, @{ z = $hand }, 1),
        @("release_gate parse finds nothing -> FAIL", "nothing", $all7, 0, @{ x = $good }, @{}, 1))) {
        $all++; $j = Judge $c[1] $c[2] $c[3] $c[4] $c[5]; $got = if ($j.bad.Count) { 1 } else { 0 }
        if ($got -eq $c[6]) { $ok++; Write-Host "ok   $($c[0])" } else { Write-Host "BAD  $($c[0]): got $got ($($j.bad -join '; '))" }
    }
    Write-Host "fake build lists selftest: $ok/$all as expected"
    exit $(if ($ok -eq $all) { 0 } else { 1 })
}

$Repo = (Resolve-Path $Repo).Path
$gateFile = Join-Path $Repo "tools\gates\release_gate.ps1"
$pwsh = (Get-Process -Id $PID).Path
$listed = @(& $pwsh -NoProfile -File $gateFile -ListExes 2>&1 | ForEach-Object { "$_" }); $listCode = $LASTEXITCODE
$texts = @{}
foreach ($f in $fakes) { $p = Join-Path $Repo $f; $texts[$f] = if (Test-Path $p) { [IO.File]::ReadAllText($p) } else { "" } }
$others = @{}
$skip = @($fakes + @("tools\gates\release_gate.ps1", "tools\gates\fake_build_lists_check.ps1"))
foreach ($d in "tools", "tests") {
    foreach ($fi in @(Get-ChildItem -Path (Join-Path $Repo $d) -Recurse -File -Filter "*.ps1" -ErrorAction SilentlyContinue)) {
        $rel = $fi.FullName.Substring($Repo.Length).TrimStart('\', '/')
        if ($skip -contains $rel) { continue }
        $others[$rel] = [IO.File]::ReadAllText($fi.FullName)
    }
}
$j = Judge ([IO.File]::ReadAllText($gateFile)) $listed $listCode $texts $others
foreach ($b in $j.bad) { Write-Host "FAIL $b" }
if ($j.bad.Count) { Write-Host "fake build lists check: FAIL ($($j.bad.Count) problem(s); $($j.want.Count) registered exes, $($fakes.Count) selftests)"; exit 1 }
Write-Host "fake build lists check: ok ($($listed.Count) exes, $($fakes.Count) selftests, $($others.Count) scripts scanned)"
exit 0
