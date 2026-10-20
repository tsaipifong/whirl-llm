# REVIEW-ARCH3 P3-3: device::profile() must not compile when hip::Arch gains an enumerator without a profile row.
# Compiles include/whirl/device_profile.h (MSVC, C++20) against a stand-in whirl/hip.h twice: with the real two
# enumerators (must compile) and with a third one (must fail with C4062 raised to an error). Exit 0 = both as expected.
#   pwsh -NoProfile -File tools\impact\profile_switch_check.ps1 [-Work <scratch dir>]
# SPDX-License-Identifier: Apache-2.0
param([string]$Work = "")
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Work) { $Work = Join-Path $env:TEMP "whirl_profile_switch_check" }
$vc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$ok = $true
foreach ($case in @(@{ name = "two"; enum = "gfx1201, gfx1151"; want = 0 }, @{ name = "three"; enum = "gfx1201, gfx1151, gfx1200"; want = 1 })) {
    $d = Join-Path $Work $case.name
    Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force (Join-Path $d "whirl") | Out-Null
    Copy-Item (Join-Path $repo "include\whirl\device_profile.h") (Join-Path $d "whirl\device_profile.h")
    Set-Content (Join-Path $d "whirl\hip.h") "#pragma once`nnamespace whirl::hip { enum class Arch { $($case.enum) }; }"
    Set-Content (Join-Path $d "t.cpp") "#include `"whirl/device_profile.h`"`nint main() { return whirl::device::profile(whirl::hip::Arch::gfx1201).tune_fast ? 1 : 0; }"
    $out = & cmd /c "call `"$vc`" >nul && cl /nologo /std:c++20 /EHsc /c /I`"$d`" `"$d\t.cpp`" /Fo`"$d\t.obj`" 2>&1"
    $rc = if ($LASTEXITCODE -eq 0) { 0 } else { 1 }
    $c4062 = [bool]($out | Select-String -SimpleMatch "C4062")
    $pass = ($rc -eq $case.want) -and ($case.want -eq 0 -or $c4062)
    "{0}: {1} enumerators -> compile exit {2}{3}: {4}" -f $(if ($pass) { "ok  " } else { "FAIL" }), $case.name, $LASTEXITCODE, $(if ($c4062) { " (C4062)" } else { "" }), $(if ($case.want -eq 0) { "must compile" } else { "must fail" })
    if (-not $pass) { $ok = $false; $out | Select-Object -Last 5 }
}
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
if ($ok) { "profile switch check: ok"; exit 0 } else { "profile switch check: FAIL"; exit 1 }
