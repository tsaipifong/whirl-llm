# SPDX-License-Identifier: Apache-2.0
# arch025-c2fu: no PowerShell script of this tree may leave an environment variable DEFINED AS EMPTY when it means to
# remove or restore it. pwsh 7.5+ (.NET 9+) keeps "k=" for [Environment]::SetEnvironmentVariable(k, $null) (PowerShell
# binds $null to the string parameter as "") and for (k, "") and for $env:k = "" (measured on pwsh 7.6.8 / .NET 10:
# scratch\a25_c2fu\envforms.ps1; Windows PowerShell 5.1 removed the variable in all three). Twice a runner restored an
# unset HIP_VISIBLE_DEVICES that way and every later child saw no GPU (arch024-S14 env_matrix.ps1, arch025-c2ctx
# survgpu.ps1). release_gate.ps1 and golden.ps1 cleared WHIRL_* the same way (a WHIRL_KV_SSD_DIR= left behind is refused
# at startup since bug-emptyrootpaths).
#
# Rules, on the PowerShell AST of every *.ps1 under tools\ and tests\ (or -Root / -Files):
#   R1 a call [Environment]::SetEnvironmentVariable(...) only inside a function whose name starts with SetEnv (the helper:
#      $null -> Remove-Item env:, else set; tools/gates/env_matrix.ps1 SetEnv, the SetEnvOrRemove copies)
#   R2 no $env:NAME = "" / '' (an assignment of an empty string literal to an env: variable)
#   R3 the helper itself must remove for $null: its body contains Remove-Item and "env:"
# A script that really needs NAME= set to an empty value for a child passes it in the child's own environment
# (ProcessStartInfo / Start-Process -Environment), not through this process.
# arch025-X8bfix (c2fu REVIEW C3): both incidents were in runners outside the tree. In the WHIRL workspace layout (the
# tree is a folder of whirl-work, next to progress\ and scratch\), the default run also checks
#   - the live runner tools <Root>\..\progress\tools\*.ps1 (the gate GPU runners start from): same rules, a violation FAILS;
#   - the one-off runners <Root>\..\scratch\**\*.ps1: violations are listed as WARN only (scratch is never shipped and
#     old runners stay as records), with a count line. A scratch runner MUST be fixed (SetEnvOrRemove) before it is
#     copied as a template for a new runner.
# Folders that do not exist are skipped (a clone of the public repo has neither). -NoWorkspace checks the tree only;
# -Live / -Scratch give other folders.
#   pwsh tools/gates/env_restore_check.ps1 [-Root DIR] [-Files a.ps1,b.ps1] [-NoWorkspace] [-Live DIR] [-Scratch DIR]
#        -> "env restore check: ok (N file(s))" / FAIL lines, exit 1; "WARN scratch ..." lines + "env restore check: scratch
#           N file(s), M with the old restore (WARN, not shipped)" never change the exit code
#   pwsh tools/gates/env_restore_check.ps1 -SelfTest                             -> "env restore check selftest: N/N as expected"
param([string]$Root = "", [string[]]$Files = @(), [switch]$SelfTest, [switch]$NoWorkspace, [string]$Live = "", [string]$Scratch = "")
$ErrorActionPreference = "Stop"
if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
if (-not $NoWorkspace -and $Files.Count -eq 0) {
    if (-not $Live) { $Live = Join-Path $Root "..\progress\tools" }
    if (-not $Scratch) { $Scratch = Join-Path $Root "..\scratch" }
} else { $Live = ""; $Scratch = "" }

function Violations([System.Management.Automation.Language.Ast]$ast) {
    $v = @()
    $calls = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.InvokeMemberExpressionAst] -and "$($n.Member)" -eq "SetEnvironmentVariable" }, $true))
    foreach ($c in $calls) {
        $p = $c.Parent; $fn = $null
        while ($p) { if ($p -is [System.Management.Automation.Language.FunctionDefinitionAst]) { $fn = $p; break }; $p = $p.Parent }
        if (-not $fn -or $fn.Name -notlike "SetEnv*") { $v += "R1 line $($c.Extent.StartLineNumber): SetEnvironmentVariable outside a SetEnv* helper: $($c.Extent.Text)" }
        elseif (-not ($fn.Body.Extent.Text -match "Remove-Item" -and $fn.Body.Extent.Text -match "env:")) {
            $v += "R3 line $($fn.Extent.StartLineNumber): helper $($fn.Name) does not remove the variable for `$null (Remove-Item env:)"
        }
    }
    $asg = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true))
    foreach ($a in $asg) {
        $l = $a.Left
        if ($l -is [System.Management.Automation.Language.VariableExpressionAst] -and $l.VariablePath.DriveName -eq "env") {
            $r = $a.Right
            if ($r -is [System.Management.Automation.Language.CommandExpressionAst]) { $r = $r.Expression }
            if ($r -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $r.Value -eq "") {
                $v += "R2 line $($a.Extent.StartLineNumber): $($a.Extent.Text) leaves the variable defined as empty (use Remove-Item env:NAME)"
            }
        }
    }
    return $v
}

function CheckText([string]$text) {
    $tok = $null; $err = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tok, [ref]$err)
    if ($err.Count) { return @("PARSE $($err[0].Message)") }
    return @(Violations $ast)
}

if ($SelfTest) {
    $helper = "function SetEnvOrRemove([string]`$k, `$v) { if (`$null -eq `$v) { Remove-Item -LiteralPath `"env:`$k`" -ErrorAction SilentlyContinue } else { [Environment]::SetEnvironmentVariable(`$k, `"`$v`") } }"
    $cases = @(
        @{ n = "restore with `$null (survgpu With-Env)"; t = "[Environment]::SetEnvironmentVariable(`$k, `$null)"; want = "R1" },
        @{ n = "restore of a saved value outside a helper"; t = "foreach (`$k in `$saved.Keys) { [Environment]::SetEnvironmentVariable(`$k, `$saved[`$k]) }"; want = "R1" },
        @{ n = "a helper with another name"; t = "function Restore([string]`$k, `$v) { [Environment]::SetEnvironmentVariable(`$k, `$v) }"; want = "R1" },
        @{ n = "a SetEnv helper that never removes"; t = "function SetEnvX([string]`$k, `$v) { [Environment]::SetEnvironmentVariable(`$k, `$v) }"; want = "R3" },
        @{ n = "`$env:X = empty string"; t = "`$env:HIP_VISIBLE_DEVICES = `"`""; want = "R2" },
        @{ n = "`$env:X = empty single-quoted"; t = "`$env:WHIRL_MODE = ''"; want = "R2" },
        @{ n = "the helper and its calls"; t = "$helper`nSetEnvOrRemove HIP_VISIBLE_DEVICES `$null`nSetEnvOrRemove WHIRL_MODE precise"; want = "" },
        @{ n = "env_matrix.ps1 SetEnv form"; t = "function SetEnv([string]`$k, `$v) {`n    if (`$null -eq `$v) { Remove-Item `"env:`$k`" -ErrorAction SilentlyContinue } else { [Environment]::SetEnvironmentVariable(`$k, `"`$v`") }`n}"; want = "" },
        @{ n = "Remove-Item env: and a non-empty assignment"; t = "Remove-Item Env:\WHIRL_MODE -ErrorAction SilentlyContinue`n`$env:WHIRL_MODE = 'precise'`n`$env:X = `$null"; want = "" },
        @{ n = "GetEnvironmentVariable is fine"; t = "`$o = [Environment]::GetEnvironmentVariable('X')"; want = "" }
    )
    $ok = 0
    foreach ($c in $cases) {
        $v = @(CheckText $c.t)
        $got = if ($v.Count -eq 0) { "" } else { ($v[0] -split " ")[0] }
        if ($got -eq $c.want) { $ok++; Write-Host "  ok    $($c.n): $(if ($got) { $got } else { 'clean' })" }
        else { Write-Host "  WRONG $($c.n): want '$($c.want)', got '$got' $($v -join ' | ')" }
    }
    # the tree itself must be clean (the live check below, on this tree)
    Write-Host "env restore check selftest: $ok/$($cases.Count) as expected"
    exit $(if ($ok -eq $cases.Count) { 0 } else { 1 })
}

$list = @($Files | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($list.Count -eq 0) {
    $list = @(foreach ($d in "tools", "tests") { $p = Join-Path $Root $d; if (Test-Path $p) { Get-ChildItem $p -Recurse -File -Filter *.ps1 | ForEach-Object { $_.FullName } } })
    # the live runner tools of the workspace (not recursive: progress\tools holds the runners themselves)
    if ($Live -and (Test-Path -LiteralPath $Live -PathType Container)) {
        $lv = @(Get-ChildItem -LiteralPath $Live -File -Filter *.ps1 | ForEach-Object { $_.FullName })
        Write-Host "env restore check: live runner folder $((Resolve-Path -LiteralPath $Live).Path) ($($lv.Count) file(s), FAIL rules)"
        $list += $lv
    }
}
$bad = 0
foreach ($f in $list) {
    $tok = $null; $err = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$tok, [ref]$err)
    if ($err.Count) { Write-Host "FAIL $f`: parse error line $($err[0].Extent.StartLineNumber): $($err[0].Message)"; $bad++; continue }
    foreach ($x in @(Violations $ast)) { Write-Host "FAIL $f`: $x"; $bad++ }
}
# scratch runners: WARN only (never shipped); listed so a template is fixed before it is copied
if ($Scratch -and (Test-Path -LiteralPath $Scratch -PathType Container)) {
    $sl = @(Get-ChildItem -LiteralPath $Scratch -Recurse -File -Filter *.ps1 -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $sw = 0
    foreach ($f in $sl) {
        $tok = $null; $err = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$tok, [ref]$err)
        if ($err.Count) { continue }  # a broken one-off script is not this check's business
        $vs = @(Violations $ast)
        if ($vs.Count) { $sw++; Write-Host "WARN scratch $f`: $($vs.Count) violation(s), first: $($vs[0])" }
    }
    Write-Host "env restore check: scratch $($sl.Count) file(s), $sw with the old restore (WARN, not shipped; fix a runner with SetEnvOrRemove before copying it as a template)"
}
if ($bad) { Write-Host "env restore check: FAIL ($bad violation(s) in $($list.Count) file(s))"; exit 1 }
Write-Host "env restore check: ok ($($list.Count) file(s))"
exit 0
