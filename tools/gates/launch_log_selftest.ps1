# SPDX-License-Identifier: Apache-2.0
# CPU-only self-test of tools/gates/launch_log.ps1 (measure-tools P7: launch logs by default + lt_import at the end)
# and of its wiring into release_gate.ps1 / golden.ps1 (the golden end-to-end cases are in tools/golden/golden_selftest.ps1).
#   pwsh tools/gates/launch_log_selftest.ps1     (exit 0 = all cases as expected, 1 = a case failed)
# The importer is a fake python script (WHIRL_LT_IMPORT): nothing is written to any results.db.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "launch_log.ps1")
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-ll-selftest-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$bad = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    Write-Host ("[{0}] {1}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $name)
    if (-not $ok) { if ($detail) { Write-Host "      $detail" }; $script:bad++ }
}
$savedLti = $env:WHIRL_LT_IMPORT
try {
    # 1) folder selection
    Check "explicit -LaunchLog wins" ((Resolve-LaunchLogDir "X:\a" "Y:\b" $false) -eq "X:\a")
    Check "default when not given" ((Resolve-LaunchLogDir "" "Y:\b" $false) -eq "Y:\b")
    Check "-NoLaunchLog turns it off (also over an explicit folder)" ((Resolve-LaunchLogDir "X:\a" "Y:\b" $true) -eq "")

    # 2) import
    $py = Get-GatePython
    Check "a python is found" ([bool]$py) "Get-GatePython returned nothing"
    $fake = Join-Path $tmp "fake_lt_import.py"
    $argsOut = Join-Path $tmp "args.json"
    @(
        "import json, sys",
        "json.dump(sys.argv[1:], open(r'$argsOut', 'w'))",
        "print('lt_import: 1 unit(s), 2 log(s), 2 row(s) written')",
        "sys.exit(int(__import__('os').environ.get('FAKE_LTI_EXIT', '0')))"
    ) | Set-Content -Encoding utf8 $fake
    $dir = Join-Path $tmp "lt"
    New-Item -ItemType Directory -Force (Join-Path $dir "golden_swift") | Out-Null
    $env:WHIRL_LT_IMPORT = $fake
    $m = Invoke-LaunchLogImport $dir "" @{} "" $py
    Check "no launch logs: no import" ($m -like "launch log import: no launch logs*" -and -not (Test-Path $argsOut)) $m
    "launch log format 1" | Set-Content (Join-Path $dir "golden_swift\launch_11.txt")
    "launch log format 1" | Set-Content (Join-Path $dir "launch_12.txt")
    $m = Invoke-LaunchLogImport $dir "G:\gate" @{ "whirl.exe" = "aa"; "whirl-server.exe" = "bb"; "x.exe" = "" } "release_gate t" $py
    $a = if (Test-Path $argsOut) { @(Get-Content $argsOut -Raw | ConvertFrom-Json) } else { @() }
    $want = @($dir, "--gate-dir", "G:\gate", "--exe-sha", "whirl-server.exe=bb", "--exe-sha", "whirl.exe=aa", "--note", "release_gate t")
    Check "importer called with the folder, gate dir, exe sha256 and note" ((($a -join "|") -eq ($want -join "|")) -and $m -like "launch log import (exit 0, 2 log(s)*2 row(s) written") "$m | args: $($a -join ' ')"
    Remove-Item $argsOut
    $env:FAKE_LTI_EXIT = "1"
    $m = Invoke-LaunchLogImport $dir "" @{} "" $py
    Remove-Item Env:FAKE_LTI_EXIT
    Check "importer failure is reported, not thrown" ($m -like "launch log import (exit 1*") $m
    Remove-Item $argsOut -ErrorAction SilentlyContinue
    $env:WHIRL_LT_IMPORT = "off"
    $m = Invoke-LaunchLogImport $dir "" @{} "" $py
    Check "WHIRL_LT_IMPORT=off: logs kept, no import" ($m -like "launch log import: off*" -and -not (Test-Path $argsOut)) $m
    $env:WHIRL_LT_IMPORT = (Join-Path $tmp "no_such_lt_import.py")
    $m = Invoke-LaunchLogImport $dir "" @{} "" $py
    Check "importer missing: skipped with the log count" ($m -like "launch log import: skipped (no importer*2 log(s) kept*") $m
    $env:WHIRL_LT_IMPORT = $fake
    $m = Invoke-LaunchLogImport $dir "" @{} "" (Join-Path $tmp "no_python.exe")
    Check "python missing: skipped" ($m -like "launch log import: skipped (python not found)*") $m
    $h = Get-ExeSha @($fake, (Join-Path $tmp "none.exe"))
    Check "Get-ExeSha hashes existing files only, keyed by lower-case file name" ($h.Count -eq 1 -and $h["fake_lt_import.py"] -eq (Get-FileHash -Algorithm SHA256 $fake).Hash.ToLower())

    # 3) wiring (static: release_gate needs the built exes and models to run; golden end to end is in golden_selftest)
    foreach ($f in @((Join-Path $PSScriptRoot "release_gate.ps1"), (Join-Path $PSScriptRoot "..\golden\golden.ps1"))) {
        $t = Get-Content -Raw $f
        $n = Split-Path $f -Leaf
        Check "$n dot-sources launch_log.ps1" ($t -match '\. \(Join-Path \$PSScriptRoot "(\.\.\\gates\\)?launch_log\.ps1"\)')
        Check "$n has -NoLaunchLog and resolves the default folder" ($t -match '\[switch\]\$NoLaunchLog' -and $t -match 'Resolve-LaunchLogDir')
        Check "$n imports at the end" ($t -match 'Invoke-LaunchLogImport')
    }
    $t = Get-Content -Raw (Join-Path $PSScriptRoot "release_gate.ps1")
    Check "release_gate: default folder is per run under -Work (lt_<Tag>_<time>)" ($t -match 'Join-Path \$Work \("lt_" \+ \$Tag \+ "_" \+ \(Get-Date')
    Check "release_gate: the summary header names the launch-log folder" ($t -match 'launch log \$')
} finally {
    $env:WHIRL_LT_IMPORT = $savedLti
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($bad -gt 0) { Write-Host "launch log selftest: $bad case(s) failed"; exit 1 }
Write-Host "launch log selftest: all cases as expected"
exit 0
