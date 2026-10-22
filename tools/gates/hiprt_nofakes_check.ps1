# arch6b: whirl-hiprt-tests run from a folder WITHOUT hiprt_fakes\ (a saved bin that copied only the *.exe, as the
# arch6 release gate M_r97 did) must fail cleanly - exit 1 with a FAIL line naming hiprt_fakes - and never crash
# (the gate saw exit -1073741819 = 0xC0000005 and an empty log, because the test called a null HIP slot).
#   pwsh -NoProfile -File tools\gates\hiprt_nofakes_check.ps1 -Build <dir with whirl-hiprt-tests.exe> [-Work <scratch>]
# Exit 0 = as expected.
# SPDX-License-Identifier: Apache-2.0
param([string]$Build, [string]$Work = "")
if (-not $Work) { $Work = Join-Path $env:TEMP "whirl_hiprt_nofakes_check" }
$d = Join-Path $Work "nofakes"
Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $d | Out-Null
Copy-Item (Join-Path $Build "whirl-hiprt-tests.exe") $d
$out = @(& (Join-Path $d "whirl-hiprt-tests.exe") 2>&1 | ForEach-Object { "$_" })
$c = $LASTEXITCODE
$named = [bool]($out | Select-String -SimpleMatch "hiprt_fakes")
$summary = [bool]($out | Select-String -Pattern "^hiprt tests: \d+ passed, [1-9]\d* failed")
$out | Select-Object -Last 6
Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
"exit $c, names hiprt_fakes=$named, summary with failures=$summary"
if ($c -eq 1 -and $named -and $summary) { "hiprt nofakes check: ok"; exit 0 }
"hiprt nofakes check: FAIL (want exit 1, a FAIL line naming hiprt_fakes and the summary line)"; exit 1
