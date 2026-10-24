# env_matrix.ps1 self-test (CPU only, no GPU, no model).
# 2026-10-24 S14 GPU run: env_matrix restored HIP_VISIBLE_DEVICES with [Environment]::SetEnvironmentVariable(k, $saved);
# PowerShell turns a $null $saved into "" and the variable stays defined as empty, so every later "nothing set" cell
# (E1a, E1b, E4) ran with HIP_VISIBLE_DEVICES= and HIP saw no GPU (server log: NoTargetGpu).
# Checks: (1) env_matrix.ps1 sets or restores environment variables only through its SetEnv helper;
#         (2) SetEnv with $null removes the variable for child processes; SetEnv with a value sets it.
# Usage: pwsh -NoProfile -File tools\gates\env_matrix_selftest.ps1 [-Script <env_matrix.ps1>]   exit 0 = PASS
param([string]$Script = (Join-Path $PSScriptRoot "env_matrix.ps1"))
$ErrorActionPreference = "Stop"
$fail = 0
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { Write-Output "PASS $name $detail" } else { $script:fail++; Write-Output "FAIL $name $detail" }
}
$tokens = $null; $errs = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$tokens, [ref]$errs)
Check "parse" ($errs.Count -eq 0) "$($errs.Count) error(s)"
$helper = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "SetEnv" }, $true)
Check "SetEnv helper exists" ($null -ne $helper)
$calls = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.InvokeMemberExpressionAst] -and "$($n.Member)" -eq "SetEnvironmentVariable" }, $true))
$outside = @($calls | Where-Object { -not $helper -or $_.Extent.StartOffset -lt $helper.Extent.StartOffset -or $_.Extent.EndOffset -gt $helper.Extent.EndOffset })
Check "no SetEnvironmentVariable outside SetEnv" ($outside.Count -eq 0) (($outside | ForEach-Object { "line $($_.Extent.StartLineNumber)" }) -join ', ')
if ($helper) {
    Invoke-Expression $helper.Extent.Text
    $k = "WHIRL_ENVMATRIX_SELFTEST"
    Remove-Item "env:$k" -ErrorAction SilentlyContinue
    $saved = [Environment]::GetEnvironmentVariable($k)
    SetEnv $k "1"
    $mid = (cmd /c "if defined $k (echo set=%$k%) else (echo unset)") | Select-Object -First 1
    SetEnv $k $saved
    $after = (cmd /c "if defined $k (echo set) else (echo unset)") | Select-Object -First 1
    Check "SetEnv value reaches a child" ("$mid".Trim() -eq "set=1") "$mid"
    Check "SetEnv `$null removes the variable for a child" ("$after".Trim() -eq "unset") "$after"
    Remove-Item "env:$k" -ErrorAction SilentlyContinue
}
if ($fail -eq 0) { Write-Output "env_matrix selftest: PASS"; exit 0 } else { Write-Output "env_matrix selftest: FAIL ($fail)"; exit 1 }
