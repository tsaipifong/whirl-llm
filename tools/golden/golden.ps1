# SPDX-License-Identifier: Apache-2.0
# Golden-hash regression gate (developers). Runs `whirl golden` for every
# {model} x {mode} x {suite} and records the "G ..." lines under
# tests/golden/<arch>/<model>/<mode>_<suite>.txt, or diffs a new run against them.
#
#   pwsh tools/golden/golden.ps1 check  [-Arch gfx1201|gfx1151] [-Models dense,moe] [-Modes precise,balance,fast] [-Suites single,pair]
#   pwsh tools/golden/golden.ps1 record [-Arch ...] ...
#
# Exit code: 0 all match, 1 mismatch (first differing step per file printed), 2 a run failed / setup error.
# Model files: $env:WHIRL_GOLDEN_DENSE / $env:WHIRL_GOLDEN_MOE, else the default LM Studio paths below.
# The executable: -Exe, else build\Release\whirl.exe of this checkout.
# A golden file stores the model file name and size; check refuses a different model.
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet("check", "record")][string]$Action,
    [ValidateSet("gfx1201", "gfx1151")][string]$Arch = "gfx1201",
    [string[]]$Models = @("dense", "moe"),
    [string[]]$Modes = @("precise", "balance", "fast"),
    [string[]]$Suites = @("single", "pair"),
    [string]$Exe = "",
    [string]$Out = ""   # check: also keep the new outputs here (default: a temp folder)
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Exe) { $Exe = Join-Path $root "build\Release\whirl.exe" }
if (-not (Test-Path $Exe)) { Write-Host "golden: no executable at $Exe (build first or pass -Exe)"; exit 2 }
# accept comma lists given as one string (pwsh -File passes "a,b" as one element)
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Modes = @($Modes | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Suites = @($Suites | ForEach-Object { $_ -split "," } | Where-Object { $_ })

$lms = Join-Path $env:USERPROFILE ".lmstudio\models"
$modelPath = @{
    dense = if ($env:WHIRL_GOLDEN_DENSE) { $env:WHIRL_GOLDEN_DENSE } else { Join-Path $lms "unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_M.gguf" }
    moe   = if ($env:WHIRL_GOLDEN_MOE) { $env:WHIRL_GOLDEN_MOE } else { Join-Path $lms "ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf" }
}
if (-not $env:AMD_LOG_LEVEL) { $env:AMD_LOG_LEVEL = "1" }
$goldDir = Join-Path $root "tests\golden\$Arch"
if (-not $Out) { $Out = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-golden-$Arch" }
New-Item -ItemType Directory -Force $Out | Out-Null

# first differing field of two "G ..." lines, e.g. "lh 1a2b... -> 3c4d..."
function FieldDiff([string]$a, [string]$b) {
    $fa = $a -split " "; $fb = $b -split " "
    $d = @()
    for ($i = 0; $i -lt [Math]::Max($fa.Count, $fb.Count); $i++) {
        $x = if ($i -lt $fa.Count) { $fa[$i] } else { "" }
        $y = if ($i -lt $fb.Count) { $fb[$i] } else { "" }
        if ($x -ne $y) {
            $kx = ($x -split "=", 2); $ky = ($y -split "=", 2)
            if ($kx[0] -eq $ky[0] -and $kx.Count -eq 2 -and $kx[0] -eq "ids") {
                # token lists: first differing position
                $ta = $kx[1] -split ","; $tb = $ky[1] -split ","
                $p = 0
                while ($p -lt $ta.Count -and $p -lt $tb.Count -and $ta[$p] -eq $tb[$p]) { $p++ }
                $ea = if ($p -lt $ta.Count) { $ta[$p] } else { "(end)" }
                $eb = if ($p -lt $tb.Count) { $tb[$p] } else { "(end)" }
                $d += "ids differ at token $p ($ea -> $eb)"
            } else {
                $d += "$x -> $y"
            }
        }
    }
    return ($d -join "; ")
}

$fail = 0; $err = 0; $t0 = Get-Date
foreach ($mk in $Models) {
    $mp = $modelPath[$mk]
    if (-not $mp -or -not (Test-Path $mp)) { Write-Host "golden: model '$mk' not found ($mp)"; $err++; continue }
    $mi = Get-Item $mp
    $mline = "G model file=$($mi.Name) bytes=$($mi.Length)"
    foreach ($mode in $Modes) {
        foreach ($suite in $Suites) {
            $name = "$mk/${mode}_$suite"
            $ts = Get-Date
            $raw = & $Exe golden $mp --suite $suite "--$mode" --device $Arch 2>&1 | ForEach-Object { "$_" }
            $rc = $LASTEXITCODE
            $sec = ((Get-Date) - $ts).TotalSeconds
            $lines = @($mline) + @($raw | Where-Object { $_ -like "G *" })
            $new = Join-Path $Out "$mk`_${mode}_$suite.txt"
            [System.IO.File]::WriteAllLines($new, [string[]]$lines)
            $diag = @($raw | Where-Object { $_ -notlike "G *" })
            if ($rc -ne 0 -or ($diag -match "hipError")) {
                Write-Host ("FAIL  {0,-24} whirl exit {1} after {2:N0} s" -f $name, $rc, $sec)
                $diag | Select-Object -Last 15 | ForEach-Object { Write-Host "      $_" }
                $err++
                if ($diag -match "hipError.*719|719.*hip|hip.*719") { Write-Host "golden: hipError 719, stopping"; exit 2 }
                continue
            }
            $gold = Join-Path $goldDir "$mk\${mode}_$suite.txt"
            if ($Action -eq "record") {
                New-Item -ItemType Directory -Force (Split-Path $gold) | Out-Null
                [System.IO.File]::WriteAllLines($gold, [string[]]$lines)
                Write-Host ("REC   {0,-24} {1} lines, {2:N0} s" -f $name, $lines.Count, $sec)
                continue
            }
            if (-not (Test-Path $gold)) { Write-Host "MISS  $name (no golden file $gold; run record)"; $err++; continue }
            $ref = @(Get-Content $gold)
            $bad = $null
            for ($i = 0; $i -lt [Math]::Max($ref.Count, $lines.Count); $i++) {
                $x = if ($i -lt $ref.Count) { $ref[$i] } else { "(missing)" }
                $y = if ($i -lt $lines.Count) { $lines[$i] } else { "(missing)" }
                if ($x -ne $y) { $bad = $i; break }
            }
            if ($null -eq $bad) {
                Write-Host ("OK    {0,-24} {1} lines, {2:N0} s" -f $name, $lines.Count, $sec)
            } else {
                $fail++
                $x = if ($bad -lt $ref.Count) { $ref[$bad] } else { "(missing)" }
                $y = if ($bad -lt $lines.Count) { $lines[$bad] } else { "(missing)" }
                $step = (($x -split " ")[1])
                $ndiff = 0
                for ($i = 0; $i -lt [Math]::Max($ref.Count, $lines.Count); $i++) {
                    if (($i -lt $ref.Count) -and ($i -lt $lines.Count) -and $ref[$i] -eq $lines[$i]) { continue }
                    $ndiff++
                }
                Write-Host ("DIFF  {0,-24} first at line {1} ({2}); {3} of {4} lines differ" -f $name, ($bad + 1), $step, $ndiff, $ref.Count)
                Write-Host "      golden: $x"
                Write-Host "      now:    $y"
                Write-Host "      change: $(FieldDiff $x $y)"
                Write-Host "      (full new output: $new)"
            }
        }
    }
}
$tot = ((Get-Date) - $t0).TotalSeconds
if ($err -gt 0) { Write-Host ("golden {0} {1}: {2} run(s) failed, {3} mismatch(es), {4:N0} s" -f $Action, $Arch, $err, $fail, $tot); exit 2 }
if ($fail -gt 0) { Write-Host ("golden check {0}: {1} mismatch(es), {2:N0} s" -f $Arch, $fail, $tot); exit 1 }
Write-Host ("golden {0} {1}: ok, {2:N0} s" -f $Action, $Arch, $tot)
exit 0
