# SPDX-License-Identifier: Apache-2.0
# bug-emptyrootpaths repro (CPU only, no model load, no GPU work): an EMPTY folder value joined with "\whirl\..." or
# "\<file>" is a path at the root of the current drive. whirl-server.exe runs as a child with a FIXED environment
# (SystemRoot, PATH, plus the variables of the case) and a fake model file, so it stops at the model read after the
# argument checks and the log open.
#   b_arg / b_eq / b_env: --kv-ssd-dir "" , --kv-ssd-dir= , WHIRL_KV_SSD_DIR= must be refused at startup with a message
#                         naming the option (before: accepted -> \whirl.lock and \<fp>.wkv at the drive root once a
#                         model loads). Safe on the unfixed build: the log goes to the case's private LOCALAPPDATA.
#   a_log (only with -LogCase): LOCALAPPDATA= (set but empty) -> the log is whirl-server.log in the working folder
#                         (before: \whirl\server.log at the drive root, created by the server). NOT safe on an unfixed
#                         build, so it runs only when asked (after the fix).
# Every case: the names directly under the drive root of the work folder are listed before and after (must be equal)
# and <root>\whirl must not exist after. The script itself never writes outside -Work.
#
#   pwsh -NoProfile -File tools/repro/emptyroot_paths.ps1 -Exe <whirl-server.exe> -Work <scratch dir> [-LogCase] [-Port 18931]
# Exit 0 = every case as expected, 1 = a case failed, 2 = setup problem.
param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$Work,
    [switch]$LogCase,
    [int]$Port = 18931
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Exe)) { Write-Host "emptyroot_paths: setup: $Exe not found"; exit 2 }
New-Item -ItemType Directory -Force $Work | Out-Null
$Work = (Resolve-Path $Work).Path
$root = [System.IO.Path]::GetPathRoot($Work)
$model = Join-Path $Work "fake_model.gguf"
[System.IO.File]::WriteAllBytes($model, [byte[]](0x4E, 0x4F, 0x54, 0x47, 0x47, 0x55, 0x46, 0x00))  # "NOTGGUF": the model read fails

function RootNames { @(Get-ChildItem -LiteralPath $root -Force -ErrorAction SilentlyContinue | ForEach-Object { $_.Name } | Sort-Object) }

function RunChild([string]$file, [string[]]$argv, [hashtable]$vars, [string]$cwd) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new($file)
    foreach ($a in $argv) { $psi.ArgumentList.Add($a) }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = $cwd
    $psi.Environment.Clear()
    $psi.Environment["SystemRoot"] = $env:SystemRoot
    $psi.Environment["PATH"] = $env:PATH
    foreach ($k in $vars.Keys) { $psi.Environment[$k] = $vars[$k] }
    $p = [System.Diagnostics.Process]::Start($psi)
    $so = $p.StandardOutput.ReadToEndAsync()
    $se = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit(60000)) { $p.Kill(); $p.WaitForExit() }
    return [pscustomobject]@{ code = $p.ExitCode; out = $so.Result; err = $se.Result; pid = $p.Id }
}

$bad = 0
$n = 0
function Case([string]$name, [string[]]$argv, [hashtable]$vars, [string]$cwd, [scriptblock]$ok) {
    $script:n++
    New-Item -ItemType Directory -Force $cwd | Out-Null
    $before = RootNames
    $r = RunChild $Exe $argv $vars $cwd
    $after = RootNames
    $rootSame = (($before -join "|") -eq ($after -join "|"))
    $noWhirl = -not (Test-Path -LiteralPath (Join-Path $root "whirl"))
    $pass = (& $ok $r) -and $rootSame -and $noWhirl
    $first = (($r.err + $r.out) -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 1)
    Write-Host ("{0} {1}: exit {2}, root entries {3} -> {4}{5}, {6}\whirl {7} | {8}" -f $(if ($pass) { "ok  " } else { "FAIL" }), $name, $r.code,
        $before.Count, $after.Count, $(if ($rootSame) { "" } else { " (CHANGED: " + ((Compare-Object $before $after | ForEach-Object { $_.InputObject }) -join ",") + ")" }),
        $root.TrimEnd("\"), $(if ($noWhirl) { "absent" } else { "PRESENT" }), $first)
    if (-not $pass) { $script:bad++; Write-Host "  stderr: $($r.err.Trim())" }
}

# the fixed environment really carries a set-but-empty variable (else a_log / b_env would test "unset")
$chk = RunChild "$env:SystemRoot\System32\cmd.exe" @("/d", "/c", "set", "WHIRL_KV_SSD_DIR") @{ "WHIRL_KV_SSD_DIR" = "" } $Work
$envOk = ($chk.out.Trim() -eq "WHIRL_KV_SSD_DIR=")
Write-Host ("{0} env_block: child sees '{1}'" -f $(if ($envOk) { "ok  " } else { "FAIL" }), $chk.out.Trim())
if (-not $envOk) { Write-Host "emptyroot_paths: setup: the empty variable did not reach the child"; exit 2 }

$lad = Join-Path $Work "lad"
$base = @($model, "--host", "127.0.0.1", "--port", "$Port")
$refused = { param($r) $r.code -eq 2 -and $r.err -match "--kv-ssd-dir" -and $r.err -match "empty" }
Case "b_arg" ($base + @("--kv-ssd-dir", "")) @{ "LOCALAPPDATA" = $lad } (Join-Path $Work "b_arg") $refused
Case "b_eq" ($base + @("--kv-ssd-dir=")) @{ "LOCALAPPDATA" = $lad } (Join-Path $Work "b_eq") $refused
Case "b_env" $base @{ "LOCALAPPDATA" = $lad; "WHIRL_KV_SSD_DIR" = "" } (Join-Path $Work "b_env") {
    param($r) $r.code -eq 2 -and $r.err -match "WHIRL_KV_SSD_DIR" -and $r.err -match "empty" }
# a folder that is given still starts normally (stops at the fake model, exit 3 = model)
Case "b_given" ($base + @("--kv-ssd-dir", (Join-Path $Work "ssd"))) @{ "LOCALAPPDATA" = $lad } (Join-Path $Work "b_given") {
    param($r) $r.code -ne 2 -and $r.code -ne 0 }
if ($LogCase) {
    $acwd = Join-Path $Work "a_cwd"
    if (Test-Path (Join-Path $acwd "whirl-server.log")) { Remove-Item (Join-Path $acwd "whirl-server.log") }
    Case "a_log" $base @{ "LOCALAPPDATA" = "" } $acwd {
        param($r) $r.code -ne 0 -and (Test-Path (Join-Path $acwd "whirl-server.log")) }
} else {
    Write-Host "skip a_log: run with -LogCase on a fixed build only (an unfixed server creates $($root)whirl\server.log)"
}
Write-Host "emptyroot_paths: $($n - $bad)/$n as expected"
exit $(if ($bad -eq 0) { 0 } else { 1 })
