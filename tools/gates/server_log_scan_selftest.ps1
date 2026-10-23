# SPDX-License-Identifier: Apache-2.0
# CPU self-test of tools/gates/server_log_scan.ps1 (arch024-S10): pass, block (owner / WDDM lines), no server log, the
# pattern sets per release gate step, lines that only mention the rule, sub-folders. No GPU, no model.
#   pwsh tools/gates/server_log_scan_selftest.ps1      (exit 0 = all cases as expected, 1 = a case failed)
$ErrorActionPreference = "Stop"
$scan = Join-Path $PSScriptRoot "server_log_scan.ps1"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-logscan-selftest-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$bad = 0
$n = 0
$ts = "2026-10-24 04:30:00.123"
function MkLogDir([string]$name, [string[]]$lines, [string]$file = "server_a.log") {
    $d = Join-Path $tmp $name
    $f = Join-Path $d $file
    New-Item -ItemType Directory -Force (Split-Path $f) | Out-Null
    if ($null -ne $lines) { [System.IO.File]::WriteAllLines($f, [string[]]$lines) }
    return $d
}
function Case([string]$what, [int]$want, [string]$rx, [string[]]$a) {
    $o = @(& pwsh -NoProfile -File $scan @a 2>&1 | ForEach-Object { "$_" })
    $c = $LASTEXITCODE
    $ok = ($c -eq $want) -and (($o -join "`n") -match $rx)
    $script:n++
    Write-Host ("[{0}] {1} (exit {2}, want {3})" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $what, $c, $want)
    if (-not $ok) { $o | Select-Object -First 6 | ForEach-Object { Write-Host "      $_" }; $script:bad++ }
}
try {
    New-Item -ItemType Directory -Force $tmp | Out-Null
    $clean = MkLogDir "clean" @("$ts I whirl-server: listening on 127.0.0.1:8093", "$ts I batch | 1 slot")
    $own = MkLogDir "owner" @("$ts I whirl-server: listening", "$ts E owner check: PagePool::take called on thread 4242")
    $wddm = MkLogDir "wddm" @("$ts I ok", "$ts W GPU memory spill: 1.2 GiB of the KV pool is in shared memory")
    $rule = MkLogDir "rule" @("$ts I vram watch: warns while GPU memory spills", "$ts I request text: E owner check: PagePool::take called on thread 1")
    $empty = Join-Path $tmp "empty"; New-Item -ItemType Directory -Force $empty | Out-Null
    $other = MkLogDir "other" @("$ts E owner check: x called on thread 1") "server_a.stdout.txt"
    $sub = MkLogDir "sub" @("$ts E owner check: TierBridge::tick called on thread 7") "case1\server_b.log"
    Case "clean log passes (owner)" 0 "server log scan: PASS 1 server log\(s\).*0 line\(s\) matched \[owner\]" @("-Dir", $clean, "-Patterns", "owner")
    Case "default set is owner" 0 "\[owner\]" @("-Dir", $clean)
    Case "owner check line blocks" 1 "FAIL 1 line\(s\) matched \[owner\].*PagePool::take" @("-Dir", $own, "-Patterns", "owner")
    Case "no server log fails" 2 "FAIL no server log" @("-Dir", $empty, "-Patterns", "owner")
    Case "missing folder fails" 2 "FAIL no server log" @("-Dir", (Join-Path $tmp "nope"), "-Patterns", "owner")
    Case "only *.stdout.txt is no server log" 2 "FAIL no server log" @("-Dir", $other, "-Patterns", "owner")
    Case "server log in a sub-folder is read" 1 "TierBridge::tick" @("-Dir", $sub, "-Patterns", "owner")
    Case "WDDM line passes the owner set" 0 "PASS" @("-Dir", $wddm, "-Patterns", "owner")
    Case "WDDM line blocks wddm,owner" 1 "FAIL 1 line\(s\) matched \[wddm,owner\]" @("-Dir", $wddm, "-Patterns", "wddm,owner")
    Case "owner line blocks owner,wddm" 1 "FAIL 1 line\(s\) matched \[owner,wddm\]" @("-Dir", $own, "-Patterns", "owner,wddm")
    Case "lines that only mention the rule pass" 0 "PASS" @("-Dir", $rule, "-Patterns", "wddm,owner")
    Case "-Step srv_*: WDDM blocks" 1 "\[wddm,owner\]" @("-Dir", $wddm, "-Step", "srv_balance")
    Case "-Step srv_*: owner blocks" 1 "\[wddm,owner\]" @("-Dir", $own, "-Step", "srv_tier_verify")
    Case "-Step srv_*: no log fails" 2 "no server log" @("-Dir", $empty, "-Step", "srv_mt_precise")
    Case "-Step golden_*: owner only" 0 "\[owner\]" @("-Dir", $wddm, "-Step", "golden_swift")
    Case "-Step of a CPU step: no scan" 0 "not a server step" @("-Dir", $empty, "-Step", "cpu_server")
    Case "unknown pattern set is a usage error" 3 "unknown pattern set" @("-Dir", $clean, "-Patterns", "nosuch")
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($bad -eq 0) { Write-Host "server log scan selftest: $n/$n as expected"; exit 0 }
Write-Host "server log scan selftest: $bad of $n case(s) failed"
exit 1
