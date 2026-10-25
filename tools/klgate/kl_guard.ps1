# SPDX-License-Identifier: Apache-2.0
# KL regression guard (arch025-klguard; owner mode standard 2026-10-26, audit research/mode-audit A4): reads a
# tools/klgate/klgate.py --report JSON and reports it as a REGRESSION GUARD, never as a veto. Shared by
# tools/golden/golden.ps1 record and the release gate step kl_guard (tools/gates/release_gate.ps1).
#
#   pwsh tools/klgate/kl_guard.ps1 [-Report kl.json] [-Exe whirl.exe] [-Arch gfx1201|gfx1151] [-Models a.gguf,b.gguf]
#                                  [-Modes balance,fast] [-TaskAccuracy FILE|exp_id] [-Evidence out.json]
#                                  [-Context "golden record"] [-ValidateOnly]
#
# The three modes (owner standard): precise = strictly lossless (golden bitwise, blocking; not judged here);
# balance = near-lossless by industry standard, defined by its allowed items (fp8, moefp8, q8dec, h16, gdnwmma,
# int8 KV), acceptance = task accuracy; fast = 4-bit-level items, acceptance = task-accuracy recovery >= 95% of
# precise. KL / top-1 limits (klgate.py LIMITS / MOE_LIMITS) are each mode's guard line.
#
# Verdicts (the LAST line printed is always one of these "kl guard: ..." lines):
#   kl guard: PASS      every case of -Modes within its guard line                                     exit 0
#   kl guard: WARN      a case beyond its guard line: one "WARN kl guard: ..." line per case with the numbers,
#                       the limit, and the mode's item list (the numerics line of that run, each item marked
#                       allowed balance item / fast item); the next steps are printed                    exit 0
#   kl guard: NOT RUN   no -Report: the guard was not run ("TODO kl-guard: ..." line)                     exit 0
#   kl guard: NOT NEEDED  -Modes empty (no balance / fast change; precise is bitwise golden)             exit 0
#   kl guard: ERROR     the report is not usable evidence (unreadable, run errors / 0 cases, a negative --as
#                       test, another exe, another GPU family, < 3 positions, a length / mode / model missing):
#                       a broken measurement, kept apart from a KL miss                                   exit 2
# Every balance / fast mode in -Modes needs a task-accuracy result reference (-TaskAccuracy: a result file, whose
# SHA-256 is recorded, or a KB exp_id such as feat027:taskacc); without it a "TODO task-accuracy: ..." line
# names the modes. -Evidence writes everything above as JSON (format whirl-kl-guard/1).
# -ValidateOnly: only the ERROR checks (golden record runs them before the GPU work), prints nothing on success.
param(
    [string]$Report = "",
    [string]$Exe = "",
    [string]$Arch = "",
    [string[]]$Models = @(),
    [string[]]$Modes = @("balance", "fast"),
    [string]$TaskAccuracy = "",
    [string]$Evidence = "",
    [string]$Context = "kl guard",
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Modes = @($Modes | ForEach-Object { $_ -split "," } | Where-Object { $_ -in @("balance", "fast") })
# allowed balance items (owner standard; numerics.cpp item names) and the fast-only items
$balanceItems = @("fp8", "moefp8", "q8dec", "h16", "gdnwmma", "kvq8", "specsample")
$fastItems = @("kvq4", "relaxacc", "headq", "moeskip", "a8", "a4")

function Fail([string]$why) { Write-Host "kl guard: ERROR $why (not usable as evidence; a KL miss is a WARN, this is a broken or wrong measurement)"; exit 2 }

# "numerics: balance - enabled: fp8, kvq8 (q8h KV (int8 ...)), specsample (...); skipped: ...; decode: q8dec (...)"
function Items([string]$num) {
    if (-not $num) { return @() }
    $s = $num
    do { $p = $s; $s = $s -replace "\([^()]*\)", "" } while ($s -ne $p)
    $out = @()
    if ($s -match "enabled:([^;]*)") { $out += @($Matches[1] -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
    if ($s -match "decode:\s*(\S+)") { $out += $Matches[1].Trim() }
    return @($out | ForEach-Object { ($_ -split "\s+")[0] } | Where-Object { $_ })
}
function ItemText([string[]]$it) {
    if ($it.Count -eq 0) { return "items unknown (no numerics line in the report)" }
    $b = @($it | Where-Object { $_ -in $balanceItems }); $f = @($it | Where-Object { $_ -in $fastItems })
    $o = @($it | Where-Object { $_ -notin $balanceItems -and $_ -notin $fastItems })
    $t = "items $($it -join ', ')"
    $t += " (allowed balance items: $(if ($b.Count) { $b -join ', ' } else { 'none' })"
    if ($f.Count) { $t += "; fast items: $($f -join ', ')" }
    if ($o.Count) { $t += "; not in the allowed lists: $($o -join ', ')" }
    return "$t)"
}
function Pct($x) { return ("{0:N2}%" -f ([double]$x * 100)) }

# ---- the report as evidence (ERROR checks) ----
$r = $null
if ($Report) {
    if (-not (Test-Path $Report)) { Fail "KL report $Report not found" }
    try { $r = Get-Content $Report -Raw | ConvertFrom-Json } catch { Fail "KL report $Report is not JSON" }
    $errs = @($r.errors | Where-Object { $_ })
    if ($errs.Count -gt 0) { Fail "KL report $Report has $($errs.Count) run error(s) (first: $($errs[0]))" }
    if (@($r.cases).Count -eq 0) { Fail "KL report $Report has 0 judged cases" }
    if ($r.mode_args -and @($r.mode_args.PSObject.Properties).Count -gt 0) { Fail "KL report $Report is a negative test (--as)" }
    if ($Exe) {
        $h = (Get-FileHash -Algorithm SHA256 $Exe).Hash.ToLower()
        if ("$($r.exe_sha256)".ToLower() -ne $h) { Fail "KL report $Report is for another exe (sha256 $($r.exe_sha256), this exe $h)" }
    }
    if ($Arch) {
        $is8060 = "$($r.device)" -eq "8060s"
        if (($Arch -eq "gfx1151") -ne $is8060) { Fail "KL report $Report was measured on device '$($r.device)', not on $Arch" }
    }
    foreach ($md in $Modes) { if ($md -notin @($r.modes)) { Fail "KL report $Report has no $md cases" } }
    # KL-ROBUST: 3 window positions per length (klgate.py default); a 1-position report carries chunk-schedule jitter
    if ([int]"$($r.positions)" -lt 3) { Fail "KL report $Report has $([int]"$($r.positions)") window position(s) per length, not 3 (KL-ROBUST: rerun klgate.py without --positions)" }
    foreach ($L in @(8192, 32768, 98304, 131072)) { if ($L -notin @($r.lengths)) { Fail "KL report $Report does not cover length $L" } }
    $names = @($r.models.PSObject.Properties | ForEach-Object { "$($_.Value)" })
    foreach ($b in $Models) { if ($b -notin $names) { Fail "KL report $Report does not cover model $b" } }
}
if ($ValidateOnly) { exit 0 }

# ---- the guard ----
$ev = [ordered]@{
    format = "whirl-kl-guard/1"; context = $Context; date = (Get-Date -Format s); arch = $Arch; modes = @($Modes)
    report = $(if ($Report) { (Resolve-Path $Report).Path } else { $null })
    report_sha256 = $(if ($Report) { (Get-FileHash -Algorithm SHA256 $Report).Hash.ToLower() } else { $null })
    exe_sha256 = $(if ($r) { "$($r.exe_sha256)" } else { $null }); device = $(if ($r) { "$($r.device)" } else { $null })
    verdict = ""; cases = 0; misses = @(); task_accuracy = $null; todo = @()
}
$lines = @()
$todo = @()
$verdict = ""
if ($Modes.Count -eq 0) {
    $verdict = "kl guard: NOT NEEDED ($Context): no balance / fast change (precise is the bitwise golden; a KL report is diagnostic only)"
} elseif (-not $r) {
    $todo += "TODO kl-guard: $($Modes -join ' / ') changed ($Context) but no KL report was given: run tools/klgate/klgate.py --report on this exe and GPU family (the regression guard; not a veto)"
    $verdict = "kl guard: NOT RUN ($Context): no KL report for $($Modes -join ' / ')"
} else {
    $cs = @($r.cases | Where-Object { "$($_.mode)" -in $Modes })
    $ev.cases = $cs.Count
    $miss = @($cs | Where-Object { $_.pass -ne $true })
    foreach ($c in $miss) {
        $md = "$($c.mode)"
        $num = $null
        if ($r.numerics) { $p = $r.numerics.PSObject.Properties["$($c.model)_$($c.prompt)_$md"]; if ($p) { $num = "$($p.Value)" } }
        $it = @(Items $num)
        $where = if ("$($c.rule)" -eq "moe-mean") { "mean over all windows" } else { "length $($c.length)" }
        $xn = if ($null -ne $c.x_noise) { " = {0:N2} N" -f [double]$c.x_noise } else { "" }
        $t = "KL {0:N5}{1} vs guard line {2:N5}" -f [double]$c.kl_mean, $xn, [double]$c.kl_limit
        if ($null -ne $c.top1) { $t += "; top1 $(Pct $c.top1) vs guard line $(Pct $c.top1_limit)" }
        if ($c.zero_rows) { $t += "; zero rows $($c.zero_rows)" }
        $lines += "WARN kl guard: $($c.model) $($c.prompt) $md $where ($($c.rule)): $t | mode $md, $(ItemText $it)"
        $ev.misses += [ordered]@{ model = "$($c.model)"; prompt = "$($c.prompt)"; mode = $md; rule = "$($c.rule)"; length = $c.length
            kl_mean = $c.kl_mean; kl_limit = $c.kl_limit; x_noise = $c.x_noise; top1 = $c.top1; top1_limit = $c.top1_limit; items = @($it); numerics = $num }
    }
    if ($miss.Count -gt 0) {
        $mm = @($miss | ForEach-Object { "$($_.mode)" } | Sort-Object -Unique) -join " / "
        $lines += "WARN kl guard: next: find which item moved it, check the task-accuracy result, then re-baseline the guard line (klgate.py --rejudge after a limit change); KL is a regression guard, not a veto (docs/dev/testing.md)"
        $verdict = "kl guard: WARN ($Context): $($miss.Count) of $($cs.Count) case(s) beyond the guard line ($mm; report $Report)"
    } else {
        $verdict = "kl guard: PASS ($Context): $($cs.Count) case(s) of $($Modes -join ' / ') within the guard line (report $Report)"
    }
}
if ($Modes.Count -gt 0) {
    if ($TaskAccuracy) {
        if (Test-Path $TaskAccuracy) {
            $ev.task_accuracy = [ordered]@{ ref = (Resolve-Path $TaskAccuracy).Path; sha256 = (Get-FileHash -Algorithm SHA256 $TaskAccuracy).Hash.ToLower() }
        } elseif ($TaskAccuracy -match "^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$") {
            $ev.task_accuracy = [ordered]@{ ref = $TaskAccuracy; kind = "kb exp_id" }
        } else { Fail "-TaskAccuracy '$TaskAccuracy' is neither a file nor a KB exp_id (item:slug)" }
        $lines += "kl guard: task-accuracy reference $($ev.task_accuracy.ref)$(if ($ev.task_accuracy.sha256) { " sha256 $($ev.task_accuracy.sha256)" })"
    } else {
        $todo += "TODO task-accuracy: $($Modes -join ' / ') changed ($Context) without a task-accuracy result reference (-TaskAccuracy FILE|exp_id): acceptance is task accuracy (balance: no loss vs precise by industry standard; fast: >= 95% of precise), KL is only the guard"
    }
}
$ev.todo = @($todo)
$ev.verdict = ($verdict -replace "^kl guard: (\S+( NEEDED| RUN)?).*$", '$1')
if ($Evidence) {
    New-Item -ItemType Directory -Force (Split-Path -Parent ([System.IO.Path]::GetFullPath($Evidence))) | Out-Null
    $ev | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $Evidence
    $lines += "kl guard: evidence $([System.IO.Path]::GetFullPath($Evidence))"
}
$lines + $todo + @($verdict) | ForEach-Object { Write-Host $_ }
exit 0
