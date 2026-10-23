# arch024-S0a CPU pins (PLAN v3 (b) P-cpu): run whirl-pins and whirl-startup-replay and compare their output with the
# files recorded on the B0 base (06a1540) in tests/pins/expected, byte for byte. CPU only.
# engine_*_tier (arch024-S6b: the spill span sequence through a RAM tier) were recorded on 906e367, before the TierBridge move.
# SPDX-License-Identifier: Apache-2.0
#   pwsh -NoProfile -File tests/pins/check.ps1 -Bin <build\Release> -Swift <gguf> -Moe <gguf> [-Out <dir>] [-Record]
# -Record writes the outputs into tests/pins/expected instead of comparing (approval style: only on the base, by hand).
param(
    [Parameter(Mandatory)][string]$Bin,
    [Parameter(Mandatory)][string]$Swift,
    [Parameter(Mandatory)][string]$Moe,
    [string]$Out = "",
    [switch]$Record
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$exp = Join-Path $here "expected"
if (-not $Out) { $Out = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl_pins_" + [guid]::NewGuid().ToString("N")) }
New-Item -ItemType Directory -Force $Out | Out-Null
$runs = @(
    @{ name = "tokens_swift"; exe = "whirl-pins"; args = @("tokens", "--gguf", $Swift, "--requests", (Join-Path $here "requests.jsonl")) },
    @{ name = "tokens_moe"; exe = "whirl-pins"; args = @("tokens", "--gguf", $Moe, "--requests", (Join-Path $here "requests.jsonl")) },
    @{ name = "engine_f16"; exe = "whirl-pins"; args = @("engine", "--gguf", $Swift, "--kv", "f16") },
    @{ name = "engine_q8h"; exe = "whirl-pins"; args = @("engine", "--gguf", $Swift, "--kv", "q8h") },
    @{ name = "engine_q4"; exe = "whirl-pins"; args = @("engine", "--gguf", $Swift, "--kv", "q4") },
    @{ name = "engine_f16_tier"; exe = "whirl-pins"; args = @("engine", "--gguf", $Swift, "--kv", "f16", "--tier") },
    @{ name = "engine_q8h_tier"; exe = "whirl-pins"; args = @("engine", "--gguf", $Swift, "--kv", "q8h", "--tier") },
    @{ name = "engine_q4_tier"; exe = "whirl-pins"; args = @("engine", "--gguf", $Swift, "--kv", "q4", "--tier") },
    @{ name = "startup_table"; exe = "whirl-startup-replay"; args = @("--table", (Join-Path $here "startup_table.txt")) }
)
$bad = 0
foreach ($r in $runs) {
    $f = Join-Path $Out "$($r.name).txt"
    $o = @(& (Join-Path $Bin "$($r.exe).exe") @($r.args) 2>&1 | ForEach-Object { "$_" })
    $rc = $LASTEXITCODE
    [System.IO.File]::WriteAllText($f, (($o -join "`n") + "`n"))
    if ($rc -ne 0) { Write-Output "FAIL $($r.name): exit $rc"; $bad++; continue }
    $e = Join-Path $exp "$($r.name).txt"
    if ($Record) {
        New-Item -ItemType Directory -Force $exp | Out-Null
        Copy-Item $f $e -Force
        Write-Output "RECORDED $($r.name) ($($o.Count) lines)"
        continue
    }
    if (-not (Test-Path $e)) { Write-Output "FAIL $($r.name): no recorded file $e"; $bad++; continue }
    $a = [System.IO.File]::ReadAllBytes($e)
    $b = [System.IO.File]::ReadAllBytes($f)
    $same = $a.Length -eq $b.Length
    if ($same) { for ($i = 0; $i -lt $a.Length; $i++) { if ($a[$i] -ne $b[$i]) { $same = $false; break } } }
    if ($same) { Write-Output "SAME $($r.name) ($($o.Count) lines)" }
    else {
        $bad++
        Write-Output "DIFF $($r.name): $e vs $f"
        $ea = Get-Content $e; $fb = Get-Content $f
        Compare-Object $ea $fb | Select-Object -First 10 | ForEach-Object { Write-Output "   $($_.SideIndicator) $($_.InputObject)" }
    }
}
Write-Output "pins: $(if ($bad -eq 0) { 'ok' } else { "$bad failed" }) (outputs in $Out)"
exit $(if ($bad -eq 0) { 0 } else { 1 })
