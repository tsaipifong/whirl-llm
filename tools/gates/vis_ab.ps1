# SPDX-License-Identifier: Apache-2.0
# CLI vision check (arch025-Q6, release_gate step vis_ab; s15-blast "visab", arch024-S11 runner scratch\s11\s11gpu.ps1).
# The three CLI vision paths of one build must give byte-identical embeddings for the same image:
#   whirl vis-encode MMPROJ IMAGE OUT --mode resident     (weights in VRAM)
#   whirl vis-encode MMPROJ IMAGE OUT --mode stream       (layer by layer)
#   whirl chat MODEL "..." --image IMAGE --mmproj MMPROJ --precise --max-tokens 48 with WHIRL_VIS_DUMP (the embeddings
#        the chat used, written by the CLI)
# and chat must answer. With -BaseBuild (a folder with a base whirl.exe) the same three runs of the base build must give
# the same embeddings and the same answer text (the A/B of S11). The answer text is compared without the lines that
# change run to run: the HIP runtime log (":1:..." with its microsecond stamps: the S11 false "TEXT DIFFERENT",
# main_session.todo:187) and the lines with timings / sizes / load rates.
#
#   pwsh tools/gates/vis_ab.ps1 -Build <dir> -Model <gguf> -Mmproj <gguf> -Image <png> -Device r9700|8060s [-BaseBuild <dir>] [-Work DIR]
#   pwsh tools/gates/vis_ab.ps1 -SelfTest            the verdict on recorded / planted cases (CPU, no exe)
# Last line "vis ab: ok (...)" exit 0, "vis ab: FAIL (<reason>)" exit 1, setup error exit 2.
param([string]$Build = "", [string]$Model = "", [string]$Mmproj = "", [string]$Image = "", [ValidateSet("r9700", "8060s")][string]$Device = "r9700",
      [string]$BaseBuild = "", [string]$Work = "", [switch]$SelfTest)
$ErrorActionPreference = "Stop"

# the answer part of a chat output: HIP log lines and lines with timings / sizes dropped
function Get-AnswerText([string[]]$lines) {
    (@($lines | Where-Object { $_ -notmatch '^:\d+:' -and $_ -notmatch '(\d+(\.\d+)?\s*(us|ms|s|tok/s|MiB|GiB|MB|GB|GB/s)\b)|sha256|loaded in|load ' }) -join "`n").Trim()
}
# side: @{ res = sha|""; str = sha|""; chat = sha|""; exit = chat exit; text = answer text }; base: same or $null
function Get-VisAbVerdict($new, $base) {
    foreach ($k in "res", "str", "chat") { if (-not $new[$k]) { return @{ v = "FAIL"; why = "missing:$k" } } }
    if ($new.exit -ne 0) { return @{ v = "FAIL"; why = "chat_exit:$($new.exit)" } }
    if ($new.res -ne $new.str -or $new.res -ne $new.chat) { return @{ v = "FAIL"; why = "paths_differ" } }
    if (-not $new.text) { return @{ v = "FAIL"; why = "no_answer" } }
    if ($base) {
        if ($base.res -ne $new.res -or $base.str -ne $new.str -or $base.chat -ne $new.chat) { return @{ v = "FAIL"; why = "ab_embeddings" } }
        if ($base.text -ne $new.text) { return @{ v = "FAIL"; why = "ab_text" } }
    }
    return @{ v = "PASS"; why = "ok" }
}

if ($SelfTest) {
    $h = "EDBB0D70E50A6537F877EBBEFC6ED5BF077FB7595701163EBAACFA19F60BC7CF"  # MEASURED scratch\s11\case_visab_r9700 (all six files)
    $hdr1 = @(":1:paldevice.cpp            :1145: 72999463529 us:  Couldn't register debug state from the loader!",
              "  mmproj: C:\models\mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf (888 MiB pinned, 544 ms)", "The picture shows a red circle and a blue square.",
              "  prefill: 239 tok in 222 ms (1078.5 tok/s)")
    $hdr2 = @(":1:paldevice.cpp            :1145: 73012674142 us:  Couldn't register debug state from the loader!",
              "  mmproj: C:\models\mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf (888 MiB pinned, 587 ms)", "The picture shows a red circle and a blue square.",
              "  prefill: 239 tok in 225 ms (1062.1 tok/s)")
    $t1 = Get-AnswerText $hdr1; $t2 = Get-AnswerText $hdr2
    $good = @{ res = $h; str = $h; chat = $h; exit = 0; text = $t1 }
    function W($over) { $x = $good.Clone(); foreach ($k in $over.Keys) { $x[$k] = $over[$k] }; $x }
    $cases = @(
        @{ n = "s11_r9700_same_build"; new = $good; base = $null; want = "PASS"; why = "ok" },
        # MEASURED S11 case: the two outputs differ only in HIP log stamps and timings -> must PASS now (was a false FAIL)
        @{ n = "s11_hip_log_stamps_only"; new = $good; base = (W @{ text = $t2 }); want = "PASS"; why = "ok" },
        @{ n = "stream_differs"; new = (W @{ str = "0" * 64 }); base = $null; want = "FAIL"; why = "paths_differ" },
        @{ n = "chat_dump_differs"; new = (W @{ chat = "1" * 64 }); base = $null; want = "FAIL"; why = "paths_differ" },
        @{ n = "no_chat_dump"; new = (W @{ chat = "" }); base = $null; want = "FAIL"; why = "missing:chat" },
        @{ n = "chat_crash"; new = (W @{ exit = -1073741819 }); base = $null; want = "FAIL"; why = "chat_exit:-1073741819" },
        @{ n = "empty_answer"; new = (W @{ text = "" }); base = $null; want = "FAIL"; why = "no_answer" },
        @{ n = "ab_embedding_changed"; new = $good; base = (W @{ res = "2" * 64; str = "2" * 64; chat = "2" * 64 }); want = "FAIL"; why = "ab_embeddings" },
        @{ n = "ab_answer_changed"; new = $good; base = (W @{ text = "The picture shows a green triangle." }); want = "FAIL"; why = "ab_text" }
    )
    $ok = 0; $bad = 0
    if ($t1 -eq $t2 -and $t1 -eq "The picture shows a red circle and a blue square.") { $ok++; Write-Host "ok   answer filter: HIP log, mmproj load and prefill timing lines dropped" }
    else { $bad++; Write-Host "BAD  answer filter: '$t1' vs '$t2'" }
    foreach ($c in $cases) {
        $r = Get-VisAbVerdict $c.new $c.base
        if ($r.v -eq $c.want -and $r.why -eq $c.why) { $ok++; Write-Host "ok   $($c.n): $($r.v) $($r.why)" }
        else { $bad++; Write-Host "BAD  $($c.n): got $($r.v) $($r.why), want $($c.want) $($c.why)" }
    }
    Write-Host "vis ab selftest: $ok/$($ok + $bad) as expected"
    if ($bad) { exit 1 }; exit 0
}

if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-vis-ab-$Device" }
New-Item -ItemType Directory -Force $Work | Out-Null
if (-not $Build) { $Build = Join-Path $PSScriptRoot "..\..\build\Release" }
foreach ($p in @((Join-Path $Build "whirl.exe"), $Model, $Mmproj, $Image) + @(if ($BaseBuild) { Join-Path $BaseBuild "whirl.exe" })) {
    if (-not $p -or -not (Test-Path -LiteralPath $p)) { Write-Host "vis ab: FAIL (setup: '$p' not found)"; exit 2 }
}
function Run-Side([string]$bin, [string]$tag) {
    $r = @{ res = ""; str = ""; chat = ""; exit = -1; text = "" }
    foreach ($mode in "resident", "stream") {
        $of = Join-Path $Work "vis_${mode}_$tag.f32"
        Remove-Item $of -ErrorAction SilentlyContinue
        $ErrorActionPreference = "Continue"
        $o = @(& (Join-Path $bin "whirl.exe") vis-encode $Mmproj $Image $of --reps 2 --mode $mode --device $Device 2>&1 | ForEach-Object { "$_" })
        $rc = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        $o | Set-Content (Join-Path $Work "vis_${mode}_$tag.txt")
        $hs = if ($rc -eq 0 -and (Test-Path $of)) { (Get-FileHash $of -Algorithm SHA256).Hash } else { "" }
        if ($mode -eq "resident") { $r.res = $hs } else { $r.str = $hs }
        Write-Host "vis ab: $tag vis-encode $mode exit $rc sha256 $(if ($hs) { $hs.Substring(0, 16) } else { '-' })"
    }
    $dump = Join-Path $Work "chat_$tag.emb"
    Get-ChildItem $Work -Filter "chat_$tag.emb*" -ErrorAction SilentlyContinue | Remove-Item -Force
    $env:WHIRL_VIS_DUMP = $dump
    $ErrorActionPreference = "Continue"
    $o = @(& (Join-Path $bin "whirl.exe") chat $Model "Describe the picture in one sentence." --image $Image --mmproj $Mmproj --precise --max-tokens 48 --device $Device 2>&1 | ForEach-Object { "$_" })
    $r.exit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Remove-Item Env:WHIRL_VIS_DUMP -ErrorAction SilentlyContinue
    $o | Set-Content (Join-Path $Work "chat_$tag.txt")
    $r.text = Get-AnswerText $o
    $df = @(Get-ChildItem $Work -Filter "chat_$tag.emb*" -File -ErrorAction SilentlyContinue | Sort-Object Name)
    if ($df.Count -eq 1) { $r.chat = (Get-FileHash $df[0].FullName -Algorithm SHA256).Hash }
    Write-Host "vis ab: $tag chat --image exit $($r.exit), dump files $($df.Count) sha256 $(if ($r.chat) { $r.chat.Substring(0, 16) } else { '-' })"
    return $r
}
$new = Run-Side $Build "new"
$base = if ($BaseBuild) { Run-Side $BaseBuild "base" } else { $null }
Get-ChildItem $Work -Filter "*.f32" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue  # 4.5 MB each
Get-ChildItem $Work -Filter "chat_*.emb*" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
$v = Get-VisAbVerdict $new $base
if ($v.v -eq "PASS") { Write-Host "vis ab: ok (resident == stream == chat dump $($new.res.Substring(0, 16))$(if ($base) { ', == base build, same answer' } else { '' }); $Device)"; exit 0 }
Write-Host "vis ab: FAIL ($($v.why))"
exit 1
