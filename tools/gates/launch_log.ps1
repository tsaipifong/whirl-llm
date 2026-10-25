# SPDX-License-Identifier: Apache-2.0
# Launch logs by default (measure-tools P7, ADR-0003 decision 3): dot-sourced by tools/gates/release_gate.ps1 and
# tools/golden/golden.ps1. A gate / golden run records the kernel launch table of every whirl process (WHIRL_LAUNCH_LOG,
# src/hip/launch_log.cpp; it records and changes nothing) into a folder of its own work dir, and at the end imports
# the folder into the records' results.db (work_traces) with lt_import.py, so the next release pair can be compared by
# tools/impact/work_equal.py instead of ending INCOMPLETE (0.2.4: S15 / S16 ran without launch logs).
#   Resolve-LaunchLogDir EXPLICIT DEFAULT OFF   -> the folder to use ("" = off): an explicit -LaunchLog wins, -NoLaunchLog
#                                                  turns recording off, else DEFAULT (a new folder per run: lt_<tag>_<time>;
#                                                  lt_import replaces the rows of a re-imported folder, so never reuse one)
#   Invoke-LaunchLogImport DIR GATEDIR EXESHA NOTE PY -> one line saying what happened; never throws, never changes a verdict
# Importer: WHIRL_LT_IMPORT (a lt_import.py path, or "off"), else the workspace's progress\tools\lt_import.py when it
# exists (records repo; a machine without it only keeps the logs). Python: Get-GatePython (WHIRL_GATE_PYTHON, else the
# workspace's pyref interpreter, else python / py on PATH). Self-test: tools/gates/launch_log_selftest.ps1.

function Resolve-LaunchLogDir([string]$Explicit, [string]$Default, [bool]$Off) {
    if ($Off) { return "" }
    if ($Explicit) { return $Explicit }
    return $Default
}

function Get-GatePython {
    $p = $env:WHIRL_GATE_PYTHON
    if ($p) { return $p }
    $pyref = "C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe"
    if (Test-Path $pyref) { return $pyref }
    $c = @(Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($c.Count -gt 0) { return $c[0].Source }
    return ""
}

function Get-LtImporter {
    if ($env:WHIRL_LT_IMPORT) { return $env:WHIRL_LT_IMPORT }
    return "C:\ClaudeWorkspace\whirl-work\progress\tools\lt_import.py"
}

function Invoke-LaunchLogImport([string]$Dir, [string]$GateDir, [hashtable]$ExeSha, [string]$Note, [string]$Py) {
    $lti = Get-LtImporter
    if ($lti -eq "off") { return "launch log import: off (WHIRL_LT_IMPORT=off); logs kept in $Dir" }
    $logs = @(Get-ChildItem -LiteralPath $Dir -Recurse -File -Filter "launch_*.txt" -ErrorAction SilentlyContinue)
    if ($logs.Count -eq 0) { return "launch log import: no launch logs under $Dir" }
    if (-not (Test-Path -LiteralPath $lti)) { return "launch log import: skipped (no importer at $lti); $($logs.Count) log(s) kept in $Dir" }
    if (-not $Py -or -not ((Test-Path -LiteralPath $Py) -or (Get-Command $Py -ErrorAction SilentlyContinue))) {
        return "launch log import: skipped (python not found); $($logs.Count) log(s) kept in $Dir"
    }
    $a = @("-I", $lti, $Dir)
    if ($GateDir) { $a += @("--gate-dir", $GateDir) }
    if ($ExeSha) { foreach ($k in @($ExeSha.Keys | Sort-Object)) { if ($ExeSha[$k]) { $a += @("--exe-sha", "$k=$($ExeSha[$k])") } } }
    if ($Note) { $a += @("--note", $Note) }
    try {
        $o = @(& $Py @a 2>&1 | ForEach-Object { "$_" })
        $c = $LASTEXITCODE
    } catch {
        return "launch log import: failed to run ($_); $($logs.Count) log(s) kept in $Dir"
    }
    $last = @($o | Where-Object { $_ -match "\S" } | Select-Object -Last 1)
    return "launch log import (exit $c, $($logs.Count) log(s) in $Dir): $(if ($last.Count) { $last[0] } else { '(no output)' })"
}

function Get-ExeSha([string[]]$Paths) {
    $h = @{}
    foreach ($p in $Paths) {
        if ($p -and (Test-Path -LiteralPath $p)) { $h[(Split-Path $p -Leaf).ToLower()] = (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower() }
    }
    return $h
}
