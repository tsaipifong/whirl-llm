# SPDX-License-Identifier: Apache-2.0
# One step of a release_gate.ps1 parallel section (arch025-procopt2, PLAN 9.24), run in its own pwsh by the gate:
#   pwsh -NoProfile -NonInteractive -File tools/gates/gate_step_child.ps1 -Spec <json>
# <json> = { cmd: [exe, args...] (@WORK@ / @GOLDEN@ already replaced), log: <step>.txt, env: { NAME: value } }. Sets env, then
# runs the command exactly as the gate's serial loop does ("& exe args *> log"; a command that cannot start appends
# "release gate: step did not run: ..." to the log and is exit -1) and exits with the command's exit code. The gate
# reads the log and applies the step's rule itself, in -List order, after the section.
param([Parameter(Mandatory = $true)][string]$Spec)
$ErrorActionPreference = "Stop"
$j = Get-Content -Raw -LiteralPath $Spec | ConvertFrom-Json
foreach ($p in @($j.env.PSObject.Properties)) { [Environment]::SetEnvironmentVariable($p.Name, "$($p.Value)") }
$cmd = @($j.cmd | ForEach-Object { "$_" })
$log = "$($j.log)"
$c = -1
try {
    & $cmd[0] @($cmd | Select-Object -Skip 1) *> $log
    $c = $LASTEXITCODE
} catch {
    "release gate: step did not run: $_" | Add-Content $log
    $c = -1
}
exit $c
