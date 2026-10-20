# SPDX-License-Identifier: Apache-2.0
# Golden-hash regression gate (developers). Runs `whirl golden` for every
# {model} x {mode} x {suite} and records the "G ..." lines under
# tests/golden/<arch>/<model>/<mode>_<suite>.txt, or diffs a new run against them.
#
#   pwsh tools/golden/golden.ps1 check  [-Arch gfx1201|gfx1151] [-Models swift,moe[,dense]] [-Modes precise,balance,fast] [-Suites single,pair]
#   pwsh tools/golden/golden.ps1 record [-Arch ...] ...
#   pwsh tools/golden/golden.ps1 refhash [-Arch ...] [-Models ...] [-Modes ...] [-Suites ...]   (CPU: reference-set hash only)
#   pwsh tools/golden/golden.ps1 check -Impact A..B|worktree [-Arch ...]   only the cases tools/impact/impact.py lists
#
# Exit code: 0 all match, 1 mismatch (first differing step per file printed), 2 a run failed / setup error.
# Models (SWIFT-GATES): swift = Swift-1.5 27B MXFP4-A (dense, the headline dense model) and moe =
# Ornith-1.5-35B-A3B MXFP4 are the default and required set; dense = Qwen3.8-27B UD-Q4_K_M is optional
# (-Models dense, or swift,moe,dense).
# Model files: $env:WHIRL_GOLDEN_SWIFT / $env:WHIRL_GOLDEN_MOE / $env:WHIRL_GOLDEN_DENSE, else the default LM Studio paths below.
# The executable: -Exe, else build\Release\whirl.exe of this checkout.
# A golden file stores the model file name and size; check refuses a different model. The model is
# also identified by a hash of its size + first and last 4 MiB (tests/golden/models.txt, written by record).
# The gate cannot pass vacuously: an empty model / mode / suite list, a run without "G" lines, an
# empty reference file, or a suite ngram run that never drafted from the n-gram table (ng_cycles=0)
# is an error (exit 2); a step line is a "G" line other than G model / G header / G info. record writes the
# references and models.txt only when every case ran. CPU self-test: tools/golden/golden_selftest.ps1. The runner clears every WHIRL_* environment variable (a stray WHIRL_HIP_DEVICE
# or WHIRL_MODE would change what is measured; --device selects the GPU) and restores them on exit.
# Suites: single, pair, ngram (forced n-gram MTP drafts on a copy-the-text prompt) and server; all four are the
# default for both architectures (server only for the product models swift / moe: the optional Q4_K_M dense
# model has no server references; -Suites ...,server asks for it). server (ARCH-1): the server paths through
# whirl-server-gate --suite golden
# (whirl-server.exe on the engine schedule: prefill, streaming, prefix cache, RAM restore, SSD restore after a
# restart, and in precise a no-prefix-cache server + two concurrent requests, all cross-checked in the gate);
# -ServerExe / -GateExe (default build\Release\whirl-server.exe / whirl-server-gate.exe of this checkout). Vision
# (one image request, tests/golden/images/shapes.png) is part of the server case: the mmproj is
# WHIRL_GOLDEN_MMPROJ_<MODEL> (e.g. WHIRL_GOLDEN_MMPROJ_SWIFT), else the mmproj-*.gguf next to the default
# swift / moe model file (dense: none, no vision case); the image folder WHIRL_GOLDEN_IMAGES, else
# tests/golden/images. A configured mmproj that is missing is an error for suite server (the references hold
# the srv.vis lines). References recorded 2026-10-20 on both GPUs with the
# v0.2.2 exes (progress/arch2.decisions.tsv).
# Every check / record prints the SHA-256 of the executables and of the reference set it compares against
# ("golden: reference set <arch> sha256 ..."): the hash over the sorted relative paths and bytes of the selected
# reference files plus models.txt; refhash prints it without running anything (release_check can derive it).
# In-run cross-checks (FIX-TEST-A, FR-TEST P2-7), on every run before it is compared or recorded, so a
# state where the paths already disagree can never become the reference (exit 2, record writes nothing):
#   every mode:          stream ids == prefill next + step next (single: short / long4k)
#   precise and balance: pair soloK.prefill == segK.prefill == segmtpK.prefill (next, top5, lh);
#                        batchK.rows and mtpcycleK ids continue soloK.stream (prefix / contiguous run);
#                        single short.stream is a prefix of mtp.stream, mtp_ngram.stream == mtp.stream
#   (fast: relaxed acceptance and the fast batch kernels change MTP / batched outputs by design.)
#   pwsh tools/golden/golden.ps1 xcheck [-Arch ...] ...   cross-checks the reference files only (CPU)
# record needs -KlReport: the JSON report of a passing KL gate run (tools/klgate/klgate.py: balance and fast
# vs precise at 8k / 32k / 96k / 128k) made with the same whirl.exe (SHA-256), on the same GPU family, covering
# every model being recorded. Without it record writes nothing and exits 2 (docs/dev/testing.md, FIX-TEST-B).
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet("check", "record", "xcheck", "refhash")][string]$Action,
    [ValidateSet("gfx1201", "gfx1151")][string]$Arch = "gfx1201",
    [string[]]$Models = @("swift", "moe"),
    [string[]]$Modes = @("precise", "balance", "fast"),
    [string[]]$Suites = @(),
    [string]$Exe = "",
    [string]$Out = "",  # check: also keep the new outputs here (default: a temp folder)
    [string]$GoldenRoot = "",  # reference tree (default tests\golden of this checkout; golden_selftest.ps1 uses a temp one)
    [string]$KlReport = "",  # record: a passing tools/klgate report (--report) of this exe on this arch, for these models
    [string]$ServerExe = "",  # suite server: whirl-server.exe (default build\Release\whirl-server.exe)
    [string]$GateExe = "",    # suite server: whirl-server-gate.exe (default build\Release\whirl-server-gate.exe)
    [int]$Port = 8097,        # suite server: the port of the servers the gate starts
    [string]$Impact = ""      # check: run only the golden cases tools/impact/impact.py lists for a diff (A..B, or worktree)
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Exe) { $Exe = Join-Path $root "build\Release\whirl.exe" }
if (-not $ServerExe) { $ServerExe = Join-Path $root "build\Release\whirl-server.exe" }
if (-not $GateExe) { $GateExe = Join-Path $root "build\Release\whirl-server-gate.exe" }
if ($Action -notin @("xcheck", "refhash") -and -not (Test-Path $Exe)) { Write-Host "golden: no executable at $Exe (build first or pass -Exe)"; exit 2 }
# accept comma lists given as one string (pwsh -File passes "a,b" as one element)
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Modes = @($Modes | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Suites = @($Suites | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$defaultSuites = $Suites.Count -eq 0 -and -not $PSBoundParameters.ContainsKey("Suites")
if ($defaultSuites) {
    $Suites = @("single", "pair", "ngram", "server")
}
# default suites: no server case for the optional Q4_K_M dense model (no server references; product models only)
function SkipCase([string]$mk, [string]$su) { return $defaultSuites -and $mk -eq "dense" -and $su -eq "server" }
if ($Models.Count -eq 0 -or $Modes.Count -eq 0 -or $Suites.Count -eq 0) {
    Write-Host "golden: empty model / mode / suite list (models=$($Models.Count) modes=$($Modes.Count) suites=$($Suites.Count)): nothing would be checked"
    exit 2
}
# -Impact: the impact map decides which models / suites this change needs on this arch (tools/impact/impact.py)
if ($Impact) {
    if ($Action -ne "check") { Write-Host "golden: -Impact is for check only"; exit 2 }
    $ia = if ($Impact -eq "worktree") { @("--worktree") } else { @("--range", $Impact) }
    $ij = & python (Join-Path $root "tools\impact\impact.py") @ia --json 2>$null | Out-String
    if ($LASTEXITCODE -ne 0 -or -not $ij) { Write-Host "golden: tools/impact/impact.py failed for '$Impact'"; exit 2 }
    $irep = ($ij | ConvertFrom-Json).PSObject.Properties | Select-Object -First 1 -ExpandProperty Value
    $iline = @($irep.golden | Where-Object { $_ -like "$Arch`:*" })
    if ($iline.Count -eq 0) {
        Write-Host "golden: SKIPPED: the impact map lists no golden case on $Arch for '$Impact' (paths: $(@($irep.paths.PSObject.Properties.Name) -join ', '))"
        exit 0
    }
    $mm = [regex]::Match($iline[0], "models (\S+) x modes (\S+) x suites (\S+)")
    $Models = @($mm.Groups[1].Value -split "," | ForEach-Object { @{ dense = "swift"; moe = "moe" }[$_] })
    $Suites = @($mm.Groups[3].Value -split ",")
    Write-Host "golden: impact map for '$Impact' on $Arch`: models $($Models -join ','), suites $($Suites -join ',')"
}
foreach ($s in $Suites) { if ($s -notin @("single", "pair", "ngram", "server")) { Write-Host "golden: unknown suite '$s'"; exit 2 } }
if ($Action -in @("check", "record") -and @($Models | Where-Object { -not (SkipCase $_ "server") }).Count -gt 0 -and $Suites -contains "server") {
    foreach ($x in @($ServerExe, $GateExe)) { if (-not (Test-Path $x)) { Write-Host "golden: suite server needs $x (build first or pass -ServerExe / -GateExe)"; exit 2 } }
}

$lms = Join-Path $env:USERPROFILE ".lmstudio\models"
$modelPath = @{
    dense = if ($env:WHIRL_GOLDEN_DENSE) { $env:WHIRL_GOLDEN_DENSE } else { Join-Path $lms "unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_M.gguf" }
    swift = if ($env:WHIRL_GOLDEN_SWIFT) { $env:WHIRL_GOLDEN_SWIFT } else { Join-Path $lms "ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf" }
    moe   = if ($env:WHIRL_GOLDEN_MOE) { $env:WHIRL_GOLDEN_MOE } else { Join-Path $lms "ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf" }
}
# suite server, vision (optional): mmproj per model + the image folder (read before the WHIRL_* variables are cleared)
$mmprojPath = @{
    dense = $env:WHIRL_GOLDEN_MMPROJ_DENSE
    swift = if ($env:WHIRL_GOLDEN_MMPROJ_SWIFT) { $env:WHIRL_GOLDEN_MMPROJ_SWIFT } else { Join-Path $lms "ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf" }
    moe   = if ($env:WHIRL_GOLDEN_MMPROJ_MOE) { $env:WHIRL_GOLDEN_MMPROJ_MOE } else { Join-Path $lms "ornith-ai\Ornith-1.5-35B-A3B-MXFP4\mmproj-Ornith-1.5-35B-A3B-F16.gguf" }
}
$goldImages = if ($env:WHIRL_GOLDEN_IMAGES) { $env:WHIRL_GOLDEN_IMAGES } else { Join-Path $root "tests\golden\images" }
# SHA-256 over the selected reference files (sorted relative path + bytes) and models.txt
function RefSetHash([string]$groot, [string]$arch, [string[]]$models, [string[]]$modes, [string[]]$suites, [bool]$all = $false) {
    $files = @()
    foreach ($mk in $models) { foreach ($md in $modes) { foreach ($su in $suites) {
        if (-not $all -and (SkipCase $mk $su)) { continue }
        $f = Join-Path $groot "$arch\$mk\${md}_$su.txt"
        if (Test-Path $f) { $files += (Resolve-Path $f).Path }
    } } }
    $mt = Join-Path $groot "models.txt"
    if (Test-Path $mt) { $files += (Resolve-Path $mt).Path }
    $groot = [System.IO.Path]::GetFullPath($groot)
    $ms = New-Object System.IO.MemoryStream
    foreach ($f in ($files | Sort-Object)) {
        $relp = [System.IO.Path]::GetRelativePath($groot, $f).Replace("\", "/")
        $b = [System.Text.Encoding]::UTF8.GetBytes($relp + "`n")
        $ms.Write($b, 0, $b.Length)
        $c = [System.IO.File]::ReadAllBytes($f)
        $ms.Write($c, 0, $c.Length)
    }
    $h = ([BitConverter]::ToString([System.Security.Cryptography.SHA256]::HashData($ms.ToArray())) -replace "-", "").ToLower()
    return @{ hash = $h; n = $files.Count }
}
# FIX-TEST-B: no reference is recorded without a passing KL gate report for this exe / arch / models
function KlReportError([string]$path) {
    if (-not $path) { return "no -KlReport (run tools/klgate/klgate.py --report on this exe first)" }
    if (-not (Test-Path $path)) { return "KL report $path not found" }
    try { $r = Get-Content $path -Raw | ConvertFrom-Json } catch { return "KL report $path is not JSON" }
    if ($r.pass -ne $true) { return "KL report $path did not pass ($($r.summary))" }
    $h = (Get-FileHash -Algorithm SHA256 $Exe).Hash.ToLower()
    if ("$($r.exe_sha256)".ToLower() -ne $h) { return "KL report $path is for another exe (sha256 $($r.exe_sha256), this exe $h)" }
    $is8060 = "$($r.device)" -eq "8060s"
    if (($Arch -eq "gfx1151") -ne $is8060) { return "KL report $path was measured on device '$($r.device)', not on $Arch" }
    foreach ($md in @("balance", "fast")) { if ($md -notin @($r.modes)) { return "KL report $path has no $md cases" } }
    # KL-ROBUST: 3 window positions per length (klgate.py default); a 1-position report carries chunk-schedule jitter
    if ([int]"$($r.positions)" -lt 3) { return "KL report $path has $([int]"$($r.positions)") window position(s) per length, not 3 (KL-ROBUST: rerun klgate.py without --positions)" }
    foreach ($L in @(8192, 32768, 98304, 131072)) { if ($L -notin @($r.lengths)) { return "KL report $path does not cover length $L" } }
    $names = @($r.models.PSObject.Properties | ForEach-Object { "$($_.Value)" })
    foreach ($m in $Models) {
        $b = Split-Path -Leaf $modelPath[$m]
        if ($b -notin $names) { return "KL report $path does not cover model $b" }
    }
    return ""
}
if ($Action -eq "record") {
    $klErr = KlReportError $KlReport
    if ($klErr) { Write-Host "golden: record refused: $klErr"; exit 2 }
    Write-Host "golden: KL gate report $KlReport passed for this exe ($Arch)"
}

# (model paths above were read from WHIRL_GOLDEN_*; now drop every WHIRL_* variable, restored in the finally below)
$savedEnv = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -like "WHIRL_*" -and $_.Name -ne "WHIRL_R9700_LOCK_HELD" })) {
    $savedEnv[$v.Name] = $v.Value
    [Environment]::SetEnvironmentVariable($v.Name, $null)
}
if (-not $env:AMD_LOG_LEVEL) { $env:AMD_LOG_LEVEL = "1" }
if (-not $GoldenRoot) { $GoldenRoot = Join-Path $root "tests\golden" }
$goldDir = Join-Path $GoldenRoot $Arch
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

# size + first and last 4 MiB, SHA-256 (a renamed or re-quantized file of the same size still differs)
function ModelHash([string]$path) {
    $fs = [System.IO.File]::OpenRead($path)
    try {
        $len = $fs.Length
        $n = [int][Math]::Min([long]4194304, $len)
        $head = New-Object byte[] $n
        [void]$fs.Read($head, 0, $n)
        $tail = New-Object byte[] $n
        [void]$fs.Seek($len - $n, [System.IO.SeekOrigin]::Begin)
        [void]$fs.Read($tail, 0, $n)
        $all = New-Object byte[] (2 * $n + 8)
        [Array]::Copy($head, 0, $all, 0, $n)
        [Array]::Copy($tail, 0, $all, $n, $n)
        [Array]::Copy([BitConverter]::GetBytes([long]$len), 0, $all, 2 * $n, 8)
        return ([BitConverter]::ToString([System.Security.Cryptography.SHA256]::HashData($all)) -replace "-", "").ToLower()
    } finally { $fs.Dispose() }
}
$hashFile = Join-Path $GoldenRoot "models.txt"
$hashes = @{}
if (Test-Path $hashFile) { foreach ($l in (Get-Content $hashFile)) { $p = $l -split " "; if ($p.Count -ge 4) { $hashes[$p[0]] = $p } } }
$hashesChanged = $false
# step lines: every "G ..." line except the model / header / info lines (a file or run without one proves nothing)
function StepCount([string[]]$l) { return @($l | Where-Object { $_ -like "G *" -and $_ -notmatch "^G (model|header|info)( |$)" }).Count }
# record: references are staged here and written with models.txt only when every case ran
$staged = [ordered]@{}

# "G name k=v k=v ..." -> @{ name = @{ k = v } }
function ParseG([string[]]$l) {
    $m = @{}
    foreach ($x in $l) {
        if ($x -notlike "G *") { continue }
        $p = $x -split " "
        $f = @{}
        foreach ($kv in $p[2..($p.Count - 1)]) { $q = $kv -split "=", 2; if ($q.Count -eq 2) { $f[$q[0]] = $q[1] } }
        $m[$p[1]] = $f
    }
    return $m
}
function IsPrefix([string]$a, [string]$b) { return ($b -eq $a) -or $b.StartsWith("$a,") }  # id list a is a prefix of b
function IsRun([string]$a, [string]$b) { return (",$b,").Contains(",$a,") }                 # a contiguous in b
# one MTP draft / verify cycle after the prefill: verify row r is the token at stream position r + 1; rows
# after the first rejected draft are not meaningful (row r counts only when drafts 0..r-1 were accepted)
function CycleOk($c, [string]$stream) {
    $st = $stream -split ","; $ids = $c["ids"] -split ","; $dr = @(if ($c["drafts"]) { $c["drafts"] -split "," })
    for ($r = 0; $r -lt $ids.Count; $r++) {
        if ($r + 1 -ge $st.Count) { return $true }
        if ($ids[$r] -ne $st[$r + 1]) { return $false }
        if ($r -ge $dr.Count -or $dr[$r] -ne $ids[$r]) { return $true }  # draft r rejected: later rows unchecked
    }
    return $true
}
# In-run relations of one golden output (see the header). Returns the violated relations; empty: ok.
function CrossCheck([string]$suite, [string]$mode, [string[]]$l) {
    $g = ParseG $l
    $bad = @()
    $exact = $mode -in @("precise", "balance")
    foreach ($s in @("short", "long4k")) {
        if (-not ($g.ContainsKey("$s.prefill") -and $g.ContainsKey("$s.stream"))) { continue }
        $ids = $g["$s.stream"]["ids"] -split ","
        $nx = @($g["$s.prefill"]["next"])
        for ($i = 1; $g.ContainsKey(("{0}.step{1:D2}" -f $s, $i)); $i++) { $nx += $g[("{0}.step{1:D2}" -f $s, $i)]["next"] }
        $n = [Math]::Min($nx.Count, $ids.Count)
        if ($n -lt 1 -or (($nx[0..($n - 1)] -join ",") -ne ($ids[0..($n - 1)] -join ","))) { $bad += "$s.stream != prefill/step next" }
    }
    if ($exact) {
        for ($k = 0; $g.ContainsKey("solo$k.prefill"); $k++) {
            $so = $g["solo$k.prefill"]
            foreach ($o in @("seg$k.prefill", "segmtp$k.prefill")) {
                if (-not $g.ContainsKey($o)) { continue }
                foreach ($f in @("next", "top5", "lh")) {
                    if ($so[$f] -ne $g[$o][$f]) { $bad += "solo$k.prefill $f != $o $f ($($so[$f]) vs $($g[$o][$f]))" }
                }
            }
            if ($g.ContainsKey("solo$k.stream")) {
                $st = $g["solo$k.stream"]["ids"]
                if ($g.ContainsKey("batch$k.rows") -and -not (IsPrefix $g["batch$k.rows"]["ids"] $st)) { $bad += "batch$k.rows is not a prefix of solo$k.stream" }
                if ($g.ContainsKey("litverify$k") -and -not (IsRun $g["litverify$k"]["ids"] $st)) { $bad += "litverify$k ids are not a run of solo$k.stream" }
                if ($g.ContainsKey("mtpcycle$k") -and -not (CycleOk $g["mtpcycle$k"] $st)) { $bad += "mtpcycle$k verify rows do not continue solo$k.stream" }
            }
        }
        if ($g.ContainsKey("short.stream") -and $g.ContainsKey("mtp.stream") -and -not (IsPrefix $g["short.stream"]["ids"] $g["mtp.stream"]["ids"])) {
            $bad += "short.stream is not a prefix of mtp.stream (MTP != plain)"
        }
        if ($g.ContainsKey("mtp.stream") -and $g.ContainsKey("mtp_ngram.stream") -and $g["mtp.stream"]["ids"] -ne $g["mtp_ngram.stream"]["ids"]) {
            $bad += "mtp_ngram.stream != mtp.stream (n-gram drafts changed the output)"
        }
    }
    return ,$bad
}
$rs = RefSetHash $GoldenRoot $Arch $Models $Modes $Suites
$rsAll = RefSetHash $GoldenRoot $Arch @("dense", "swift", "moe") @("precise", "balance", "fast") @("single", "pair", "ngram", "server") $true
if ($Action -eq "refhash") {
    Write-Host "golden: reference set $Arch sha256 $($rs.hash) ($($rs.n) files: models $($Models -join ','), modes $($Modes -join ','), suites $($Suites -join ','))"
    Write-Host "golden: reference set $Arch (all files) sha256 $($rsAll.hash) ($($rsAll.n) files)"
    foreach ($k in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $savedEnv[$k]) }
    exit 0
}
if ($Action -eq "xcheck") {
    $nf = 0; $nx = 0
    foreach ($mk in $Models) { foreach ($mode in $Modes) { foreach ($suite in $Suites) {
        if (SkipCase $mk $suite) { continue }
        $gold = Join-Path $goldDir "$mk\${mode}_$suite.txt"
        if (-not (Test-Path $gold)) { Write-Host "MISS  $mk/${mode}_$suite ($gold)"; $nx++; continue }
        $nf++
        $v = CrossCheck $suite $mode @(Get-Content $gold)
        if ($v.Count -gt 0) { $nx++; Write-Host "XBAD  $mk/${mode}_$suite"; $v | ForEach-Object { Write-Host "      $_" } }
        else { Write-Host "XOK   $mk/${mode}_$suite" }
    } } }
    foreach ($k in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $savedEnv[$k]) }
    if ($nf -eq 0) { Write-Host "golden xcheck: no reference file found"; exit 2 }
    if ($nx -gt 0) { Write-Host "golden xcheck $Arch`: $nx file(s) violate in-run relations or are missing"; exit 2 }
    Write-Host "golden xcheck $Arch`: $nf file(s) ok"
    exit 0
}

$fail = 0; $err = 0; $ran = 0; $t0 = Get-Date
Write-Host "golden: exe $Exe sha256 $((Get-FileHash -Algorithm SHA256 $Exe).Hash.ToLower())"
if ($Suites -contains "server") {
    Write-Host "golden: server exe $ServerExe sha256 $((Get-FileHash -Algorithm SHA256 $ServerExe).Hash.ToLower())"
    Write-Host "golden: gate exe $GateExe sha256 $((Get-FileHash -Algorithm SHA256 $GateExe).Hash.ToLower())"
}
Write-Host "golden: reference set $Arch sha256 $($rs.hash) ($($rs.n) files; all files of $Arch`: $($rsAll.hash))"
try {
foreach ($mk in $Models) {
    $mp = $modelPath[$mk]
    if (-not $mp -or -not (Test-Path $mp)) { Write-Host "golden: model '$mk' not found ($mp)"; $err++; continue }
    $mi = Get-Item $mp
    $mline = "G model file=$($mi.Name) bytes=$($mi.Length)"
    $mh = ModelHash $mp
    if ($Action -eq "record") {
        $hashes[$mk] = @($mk, $mi.Name, "$($mi.Length)", $mh); $hashesChanged = $true
    } elseif (-not $hashes.ContainsKey($mk)) {
        Write-Host "golden: no recorded hash for model '$mk' in $hashFile (run record)"; $err++; continue
    } elseif ($hashes[$mk][1] -ne $mi.Name -or $hashes[$mk][2] -ne "$($mi.Length)" -or $hashes[$mk][3] -ne $mh) {
        Write-Host "golden: model '$mk' ($mp) is not the recorded file (name/size/hash differ from $hashFile)"; $err++; continue
    }
    foreach ($mode in $Modes) {
        foreach ($suite in $Suites) {
            if (SkipCase $mk $suite) { continue }
            $name = "$mk/${mode}_$suite"
            $ts = Get-Date
            if ($suite -eq "server") {
                # whirl-server-gate --suite golden: G srv.* lines; its own in-run checks must all pass
                $gw = Join-Path $Out "srvwork_$mk`_$mode"
                $gargs = @("--exe", $ServerExe, "--kind", "whirl", "--model", $mp, "--suite", "golden", "--port", "$Port",
                           "--work", $gw, "--kv", "auto", "--server-env", "WHIRL_MODE=$mode")
                if ($Arch -eq "gfx1151") { $gargs += @("--server-env", "WHIRL_DEVICE=8060s", "--server-env", "WHIRL_KV_RAM_MB=8192") }
                if ($mmprojPath[$mk]) {
                    if (-not ((Test-Path $mmprojPath[$mk]) -and (Test-Path (Join-Path $goldImages "shapes.png")))) {
                        Write-Host "FAIL  $name suite server needs the mmproj ($($mmprojPath[$mk])) and $goldImages\shapes.png (vision case)"
                        $err++
                        continue
                    }
                    $gargs += @("--mmproj", $mmprojPath[$mk], "--images", $goldImages)
                }
                $raw = & $GateExe @gargs 2>&1 | ForEach-Object { "$_" }
                $rc = $LASTEXITCODE
                if ($rc -eq 0 -and -not ($raw -match "\(ALL PASS\)")) { $rc = 9 }
                Remove-Item -Recurse -Force $gw -ErrorAction SilentlyContinue
            } else {
                $raw = & $Exe golden $mp --suite $suite "--$mode" --device $Arch 2>&1 | ForEach-Object { "$_" }
                $rc = $LASTEXITCODE
            }
            $sec = ((Get-Date) - $ts).TotalSeconds
            $glines = @($raw | Where-Object { $_ -like "G *" })
            $lines = @($mline) + $glines
            $new = Join-Path $Out "$mk`_${mode}_$suite.txt"
            [System.IO.File]::WriteAllLines($new, [string[]]$lines)
            $diag = @($raw | Where-Object { $_ -notlike "G *" })
            if ($rc -ne 0 -or ($diag -match "hipError")) {
                Write-Host ("FAIL  {0,-24} whirl exit {1} after {2:N0} s" -f $name, $rc, $sec)
                $diag | Select-Object -Last 15 | ForEach-Object { Write-Host "      $_" }
                $err++
                if ($diag -match "hipError.*719|\b719\b.*hip|hip.*\b719\b") { Write-Host "golden: hipError 719, stopping"; exit 2 }
                continue
            }
            # a run that printed (almost) nothing must not become, or be compared with, a reference
            if ($glines.Count -lt 2 -or -not ($glines[0] -like "G header *") -or (StepCount $glines) -lt 1) {
                Write-Host ("FAIL  {0,-24} no golden output ({1} G lines, {2} step lines)" -f $name, $glines.Count, (StepCount $glines))
                $diag | Select-Object -Last 10 | ForEach-Object { Write-Host "      $_" }
                $err++
                continue
            }
            if ($suite -eq "ngram") {
                $ng = @($glines | Where-Object { $_ -match "ng_cycles=(\d+)" } | ForEach-Object { [int]$Matches[1] })
                if ($ng.Count -eq 0 -or ($ng | Measure-Object -Maximum).Maximum -le 0) {
                    Write-Host ("FAIL  {0,-24} the n-gram path was not exercised (ng_cycles=0): the case proves nothing" -f $name)
                    $err++
                    continue
                }
            }
            # in-run relations (FR-TEST P2-7): checked before compare / record
            $xv = CrossCheck $suite $mode $lines
            if ($xv.Count -gt 0) {
                Write-Host ("FAIL  {0,-24} in-run cross-check: {1} relation(s) violated" -f $name, $xv.Count)
                $xv | ForEach-Object { Write-Host "      $_" }
                $err++
                continue
            }
            $ran++
            $gold = Join-Path $goldDir "$mk\${mode}_$suite.txt"
            if ($Action -eq "record") {
                $staged[$gold] = [string[]]$lines
                Write-Host ("REC   {0,-24} {1} lines, {2:N0} s (staged)" -f $name, $lines.Count, $sec)
                continue
            }
            if (-not (Test-Path $gold)) { Write-Host "MISS  $name (no golden file $gold; run record)"; $err++; continue }
            $ref = @(Get-Content $gold)
            if ((StepCount $ref) -lt 1) { Write-Host "FAIL  $name golden file $gold is empty or has no step lines"; $err++; continue }
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
} finally {
    foreach ($k in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $savedEnv[$k]) }
}
# record: the references and models.txt are written together, and only when no case failed (a partial
# record would leave new references next to a stale model hash)
if ($Action -eq "record" -and $err -eq 0) {
    foreach ($gold in $staged.Keys) {
        New-Item -ItemType Directory -Force (Split-Path $gold) | Out-Null
        [System.IO.File]::WriteAllLines($gold, $staged[$gold])
    }
    if ($hashesChanged) {
        New-Item -ItemType Directory -Force (Split-Path $hashFile) | Out-Null
        [System.IO.File]::WriteAllLines($hashFile, [string[]]@($hashes.Keys | Sort-Object | ForEach-Object { $hashes[$_] -join " " }))
    }
} elseif ($Action -eq "record" -and $staged.Count -gt 0) {
    Write-Host "golden record: $err case(s) failed: nothing written ($($staged.Count) staged reference(s) discarded, models.txt unchanged)"
}
$tot = ((Get-Date) - $t0).TotalSeconds
if ($ran -eq 0) { Write-Host "golden: no case ran: refusing to report success"; exit 2 }
if ($err -gt 0) { Write-Host ("golden {0} {1}: {2} run(s) failed, {3} mismatch(es), {4:N0} s" -f $Action, $Arch, $err, $fail, $tot); exit 2 }
if ($fail -gt 0) { Write-Host ("golden check {0}: {1} mismatch(es), {2:N0} s" -f $Arch, $fail, $tot); exit 1 }
Write-Host ("golden {0} {1}: ok, {2:N0} s" -f $Action, $Arch, $tot)
exit 0
