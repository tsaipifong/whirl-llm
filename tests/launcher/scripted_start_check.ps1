# SPDX-License-Identifier: Apache-2.0
# feat026 launcher: "never prompt, never the tray when started by a script" on the real whirl.exe (CPU only: these
# starts return before the GPU runtime is loaded). Every case must finish at once (a prompt would hang until the
# timeout) with today's behaviour: no arguments -> the usage text on stderr, exit code 2.
#   pwsh -NoProfile -File tests\launcher\scripted_start_check.ps1 [-Exe <path to whirl.exe>]
# Exit 0 = all cases pass. A run from this script is itself "a script": the console has >= 2 processes.
param([string]$Exe = "$PSScriptRoot\..\..\build\Release\whirl.exe", [int]$TimeoutS = 20)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Exe)) { Write-Host "scripted_start_check: FAIL (no $Exe)"; exit 2 }
$Exe = (Resolve-Path $Exe).Path
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("whirl-launcher-check-" + $PID)
New-Item -ItemType Directory -Force $tmp | Out-Null
$bad = 0
function Case([string]$name, [string[]]$argList, [hashtable]$envAdd, [string]$stdin, [int]$wantCode, [string]$wantErr) {
    $out = Join-Path $tmp "$name.out"; $err = Join-Path $tmp "$name.err"
    $saved = @{}
    foreach ($k in $envAdd.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $envAdd[$k]) }
    try {
        $p = @{ FilePath = $Exe; RedirectStandardOutput = $out; RedirectStandardError = $err; NoNewWindow = $true; PassThru = $true }
        if ($argList.Count -gt 0) { $p.ArgumentList = $argList }
        if ($stdin) { $p.RedirectStandardInput = $stdin }
        $proc = Start-Process @p
        if (-not $proc.WaitForExit($TimeoutS * 1000)) {
            $proc.Kill()
            $script:bad++
            Write-Host "FAIL $name : still running after $TimeoutS s (it waited for input or started a server)"
            return
        }
        $code = $proc.ExitCode
        $e = "" + $(if (Test-Path $err) { Get-Content $err -Raw } else { "" })
        $ok = ($code -eq $wantCode) -and ($e -match $wantErr)
        if (-not $ok) { $script:bad++ }
        Write-Host ("{0} {1} : exit {2} (want {3}); stderr {4}" -f ($(if ($ok) { "ok  " } else { "FAIL" }), $name, $code, $wantCode, ($e -split "`n")[0].Trim()))
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}
$nul = Join-Path $tmp "empty.txt"; Set-Content -Path $nul -Value "" -NoNewline
$enter = Join-Path $tmp "enter.txt"; Set-Content -Path $enter -Value "`r`n`r`n`r`n`r`n`r`n`r`n"
# no arguments, stdin / stdout / stderr redirected (a script, a pipe)
Case "noargs_redirected" @() @{} $nul 2 "usage: whirl <command>"
# no arguments with Enter keys on stdin: must still not take them as setup answers
Case "noargs_enter_on_stdin" @() @{} $enter 2 "usage: whirl <command>"
# stdout redirected only (stdin inherited from this console)
Case "noargs_stdout_redirected" @() @{} "" 2 "usage: whirl <command>"
# WHIRL_NO_LAUNCHER=1
Case "noargs_no_launcher_env" @() @{ WHIRL_NO_LAUNCHER = "1" } $nul 2 "usage: whirl <command>"
# --setup from a script: refused, no prompt
Case "setup_redirected" @("--setup") @{} $enter 2 "--setup asks questions"
# existing commands keep their text / exit codes
Case "help" @("--help") @{} $nul 0 "^$|.*"
Case "unknown_command" @("frobnicate") @{} $nul 2 "unknown command 'frobnicate'"
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
if ($bad -eq 0) { Write-Host "scripted_start_check: all cases passed"; exit 0 }
Write-Host "scripted_start_check: $bad case(s) FAILED"; exit 1
