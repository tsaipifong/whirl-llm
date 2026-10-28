# SPDX-License-Identifier: Apache-2.0
# arch025-c2fu: release_gate step st_ctx_refuse. Precise mode keeps the KV cache in f16 and a context the user did not
# shorten must NOT be shrunk silently: when the f16 KV for the default 4096 tokens does not fit, `whirl selftest MODEL
# --precise` refuses (exit 5, "... (room for about N tokens). Use a shorter --ctx ..."). Mutant c2-ctx-default (cli.h:
# loadModel's ctx_explicit default flipped to false) loads with a shrunk context instead and survived every other step
# (arch025-survgpu / arch025-c2ctx: killed only by this squeeze on the 8060S, 2026-10-28).
#
# Squeeze: WHIRL_VRAM_LIMIT_MB (include/whirl/vram_limit.h: total' = min(total, L), free' = free - (total - total')), set
# for the selftest child only. No memory hog: on the UMA 8060S another process's allocation does not lower this
# process's free memory (arch025-c2ctx), on the R9700 other programs' VRAM still counts against L, so L is calibrated
# per run: start at -StartMb (20480: the 8060S control refuses there with room ~1280 tokens), then from the refusal text
# (per-token bytes from "KV for 4096 tokens needs X GiB", room N) aim at a room of 2048 tokens; a run that loads past the
# KV check ("numerics at load" / "context lowered") lowers L, a run that fails before it (weights do not fit) raises L
# (bisection). PASS = one run refused with the KV text, exit 5, a room in 512..3584 tokens (the room a shrinking mutant
# would have used: numerics::cliCtxFit rounds the room down to whole KV pages and shrinks instead of refusing whenever
# that is >= max(min_ctx, one page), so a shrinking selftest build (min_ctx 0) refuses only with room 0), the GPU line
# naming the -Device card. FAIL = no such refusal in -MaxTries runs. A loading run is
# stopped at once (6-10 s per run on the 8060S).
#   pwsh tools/gates/ctx_refuse_check.ps1 -Exe build\Release\whirl.exe -Model <Swift 27B MXFP4-A> -Device r9700|8060s -Work DIR
#   pwsh tools/gates/ctx_refuse_check.ps1 -SelfTest      (the outcome classifier on planted outputs; CPU)
# Output: "ctx refuse check: REPORT ..." per run, then "ctx refuse check: PASS limit=L room=N" or "ctx refuse check: FAIL ...".
param([string]$Exe = "", [string]$Model = "", [ValidateSet("r9700", "8060s")][string]$Device = "8060s", [string]$Work = "",
      [int]$StartMb = 20480, [int]$MaxTries = 6, [int]$RunCapS = 300, [switch]$SelfTest)
$ErrorActionPreference = "Stop"
function SetEnvOrRemove([string]$k, $v) {
    if ($null -eq $v) { Remove-Item -LiteralPath "env:$k" -ErrorAction SilentlyContinue } else { [Environment]::SetEnvironmentVariable($k, "$v") }
}
$cardRe = if ($Device -eq "8060s") { "8060S" } else { "R9700" }

# outcome of one run from its exit code (or "stopped" when it loaded past the KV check and was stopped) and its text
function Classify($exit, [string]$t) {
    $r = @{ outcome = "unknown"; room = -1; pt = 0.0 }
    if ($t -match 'KV for 4096 tokens needs ([0-9.]+) GiB') { $r.pt = [double]$Matches[1] * 1073741824 / 4096 }
    if ($t -match 'room for about (\d+) tokens') { $r.room = [int]$Matches[1] }
    if ($exit -eq "stopped" -or $exit -eq 0 -or $t -match 'numerics at load|context lowered') { $r.outcome = "proceeded" }
    elseif ($exit -eq 5 -and $t -match 'Use a shorter --ctx' -and $r.room -ge 0) { $r.outcome = "refused" }
    elseif ($exit -is [int] -and $exit -ne 0) { $r.outcome = "loadfail" }
    return $r
}

if ($SelfTest) {
    $refuse = "whirl: GPU pci:196:0 (AMD Radeon(TM) 8060S Graphics): chosen with --device`nwhirl: error: precise mode keeps the KV cache in f16: f16 KV for 4096 tokens needs 0.27 GiB, about 0.09 GiB of GPU memory is left for it (room for about 1280 tokens). Use a shorter --ctx (--balance allows int8 KV)"
    $cases = @(
        @{ n = "control refusal (8060S, 2026-10-28)"; e = 5; t = $refuse; want = "refused"; room = 1280 },
        @{ n = "mutant loads past the KV check"; e = "stopped"; t = "  numerics at load: precise (precise checks run with fp8 / MoE fp8 / h16 off)"; want = "proceeded"; room = -1 },
        @{ n = "selftest passes (KV fits)"; e = 0; t = "selftest: ok"; want = "proceeded"; room = -1 },
        @{ n = "context lowered note"; e = 0; t = "whirl: context lowered to 1024 tokens"; want = "proceeded"; room = -1 },
        @{ n = "weights do not fit"; e = 5; t = "whirl: error: the model does not fit in the GPU's memory (VramLimit)"; want = "loadfail"; room = -1 },
        @{ n = "refusal text with another exit code"; e = 1; t = $refuse; want = "loadfail"; room = 1280 },
        @{ n = "no GPU"; e = 3; t = "whirl: error: no AMD GPU is visible"; want = "loadfail"; room = -1 }
    )
    $ok = 0
    foreach ($c in $cases) {
        $r = Classify $c.e $c.t
        if ($r.outcome -eq $c.want -and $r.room -eq $c.room) { $ok++; Write-Host "  ok    $($c.n): $($r.outcome) room=$($r.room)" }
        else { Write-Host "  WRONG $($c.n): want $($c.want) room=$($c.room), got $($r.outcome) room=$($r.room)" }
    }
    Write-Host "ctx refuse check selftest: $ok/$($cases.Count) as expected"
    exit $(if ($ok -eq $cases.Count) { 0 } else { 1 })
}

if (-not (Test-Path -LiteralPath $Exe) -or -not (Test-Path -LiteralPath $Model)) { Write-Host "ctx refuse check: FAIL (setup: exe '$Exe' or model '$Model' not found)"; exit 2 }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-ctx-refuse-$Device" }
New-Item -ItemType Directory -Force $Work | Out-Null

function RunOnce([int]$mb, [int]$k) {
    $out = Join-Path $Work "selftest_$k.out"; $err = Join-Path $Work "selftest_$k.err"
    $old = [Environment]::GetEnvironmentVariable("WHIRL_VRAM_LIMIT_MB")
    SetEnvOrRemove "WHIRL_VRAM_LIMIT_MB" "$mb"
    try {
        $p = Start-Process -FilePath $Exe -ArgumentList @("selftest", "`"$Model`"", "--precise", "--device", $Device) -PassThru -NoNewWindow `
            -RedirectStandardOutput $out -RedirectStandardError $err
    } finally { SetEnvOrRemove "WHIRL_VRAM_LIMIT_MB" $old }
    $t0 = Get-Date; $stopped = $false; $capped = $false
    while (-not $p.HasExited -and ((Get-Date) - $t0).TotalSeconds -lt $RunCapS) {
        $t = (Get-Content $out -Raw -ErrorAction SilentlyContinue) + (Get-Content $err -Raw -ErrorAction SilentlyContinue)
        if ($t -match 'numerics at load|context lowered') { $stopped = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; $p.WaitForExit(30000) | Out-Null; if (-not $stopped) { $capped = $true } }
    $exit = if ($capped) { "cap" } elseif ($stopped) { "stopped" } else { $p.ExitCode }
    $t = (Get-Content $out -Raw -ErrorAction SilentlyContinue) + "`n" + (Get-Content $err -Raw -ErrorAction SilentlyContinue)
    $r = Classify $exit $t
    if ($exit -eq "cap") { $r.outcome = "timeout" }
    $r.gpu = (($t -split "`n") | Where-Object { $_ -match '^whirl: GPU ' } | Select-Object -First 1)
    $r.why = (($t -split "`n") | Where-Object { $_ -match 'error|KV|context|numerics at load' } | Select-Object -First 1)
    Write-Host ("ctx refuse check: REPORT run {0} limit={1} MiB exit={2} outcome={3} room={4} {5:N0} s | {6} | {7}" -f $k, $mb, $exit, $r.outcome, $r.room,
        ((Get-Date) - $t0).TotalSeconds, "$($r.gpu)".Trim(), "$($r.why)".Trim())
    return $r
}

$x = $StartMb; $lo = $null; $hi = $null; $pt = 69632.0  # bytes per token (0.27 GiB / 4096, Swift 27B f16 KV), refined from the text
for ($k = 1; $k -le $MaxTries; $k++) {
    $r = RunOnce $x $k
    if ($r.pt -gt 0) { $pt = $r.pt }
    if ($r.outcome -eq "refused" -and $r.room -ge 512 -and $r.room -le 3584) {
        if ("$($r.gpu)" -notmatch $cardRe) { Write-Host "ctx refuse check: FAIL (the run's GPU line does not name the ${cardRe}: '$($r.gpu)')"; exit 1 }
        Write-Host "ctx refuse check: PASS limit=$x room=$($r.room) (selftest --precise refused the 4096-token f16 KV instead of shrinking the context)"
        exit 0
    }
    # room > 3584: aim at 2048 from the per-token size; room < 512 (0 = the KV check is already short by an unknown amount):
    # a lower bound like a load failure (bisect towards the last limit that loaded past the check)
    if ($r.outcome -eq "refused" -and $r.room -gt 3584) { $x = [int]($x - ($r.room - 2048) * $pt / 1048576) }
    elseif ($r.outcome -eq "refused" -or $r.outcome -eq "loadfail") { $lo = $x; $x = if ($null -ne $hi) { [int](($x + $hi) / 2) } else { $x + 2048 } }
    elseif ($r.outcome -eq "proceeded") { $hi = $x; $x = if ($null -ne $lo) { [int](($lo + $x) / 2) } else { $x - 2048 } }
    else { Write-Host "ctx refuse check: FAIL (run $k $($r.outcome))"; exit 1 }
    if ($x -lt 1024) { break }
}
Write-Host "ctx refuse check: FAIL (no f16-KV refusal with room 512..3584 tokens in $MaxTries run(s); last limit $x MiB; loads past the KV check at $hi MiB, fails before it at $lo MiB)"
exit 1
