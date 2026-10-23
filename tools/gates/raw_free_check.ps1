# SPDX-License-Identifier: Apache-2.0
# arch024-S11: no hand-written GPU release outside the RAII layer. Fails when a source file under src/ calls
# hip::free / hip::hostFree / hip::eventDestroy / hip::streamDestroy, the same four through a Backend
# (x.free( / x->hostFree( ...), or the raw HIP entry points (hipFree( / hipHostFree( / hipEventDestroy( /
# hipStreamDestroy(), outside the files that implement the release:
#   src/hip/hip_runtime.cpp        the HIP wrapper itself
#   src/device/handles.h           the RAII handles (DevBuf / PinnedBuf / EventH / StreamH)
#   src/device/backend_hip.cpp     Backend over HIP (forwards to hip::)
#   src/device/null_backend.*      Backend for the CPU tests
# Comments are stripped before matching. Sites that other arch024 steps remove on other branches are listed one
# per line in tools/gates/raw_free_pending.txt (path <TAB> exact text of the line, trimmed <TAB> count <TAB>
# step). Each entry must match exactly `count` hits: once the step is merged its entry no longer matches, the
# check FAILs as STALE, and the entry has to be deleted - so the list can only shrink.
#   pwsh -NoProfile -File tools\gates\raw_free_check.ps1 [-Root <repo>] [-Pending <file>]
#   pwsh -NoProfile -File tools\gates\raw_free_check.ps1 -SelfTest
# Last line: "raw free check: ok (...)" (exit 0) or "raw free check: FAIL (...)" (exit 1); exit 2 = usage.
param([string]$Root = "", [string]$Pending = "", [switch]$SelfTest)
$ErrorActionPreference = "Stop"

$Allowed = @('src/hip/hip_runtime.cpp', 'src/device/handles.h', 'src/device/backend_hip.cpp', 'src/device/null_backend.cpp',
    'src/device/null_backend.h')
$Patterns = @(
    '\bhip::(free|hostFree|eventDestroy|streamDestroy)\s*\(',
    '(\.|->)\s*(free|hostFree|eventDestroy|streamDestroy)\s*\(',
    '\bhip(Free|HostFree|FreeHost|EventDestroy|StreamDestroy)\s*\('
)
$Exts = @('.cpp', '.h', '.hpp', '.cc', '.inl', '.hip', '.cu', '.cuh')

# Code text of each line with // and /* */ comments removed (string literals are kept as they are; a "//" inside
# a string cuts the rest of that line, which can only hide a match, never invent one).
function Get-CodeLines([string]$path) {
    $raw = [System.IO.File]::ReadAllText($path)
    $lines = $raw -split "`r?`n"
    $out = New-Object System.Collections.Generic.List[string]
    $inBlock = $false
    foreach ($l in $lines) {
        $s = $l
        $code = ""
        while ($true) {
            if ($inBlock) {
                $e = $s.IndexOf('*/')
                if ($e -lt 0) { $s = ""; break }
                $s = $s.Substring($e + 2)
                $inBlock = $false
                continue
            }
            $lc = $s.IndexOf('//')
            $bc = $s.IndexOf('/*')
            if ($bc -ge 0 -and ($lc -lt 0 -or $bc -lt $lc)) {
                $code += $s.Substring(0, $bc)
                $s = $s.Substring($bc + 2)
                $inBlock = $true
                continue
            }
            if ($lc -ge 0) { $code += $s.Substring(0, $lc) } else { $code += $s }
            break
        }
        $out.Add($code)
    }
    return , $out
}

function Invoke-Check([string]$treeRoot, [string]$pendingFile) {
    $src = Join-Path $treeRoot "src"
    if (-not (Test-Path $src)) { return @{ rc = 2; lines = @("raw free check: FAIL (usage: no src/ under $treeRoot)") } }
    $hits = New-Object System.Collections.Generic.List[object]
    $files = Get-ChildItem -Path $src -Recurse -File | Where-Object { $Exts -contains $_.Extension.ToLowerInvariant() }
    $rootFull = (Resolve-Path $treeRoot).Path.TrimEnd('\', '/')
    foreach ($f in $files) {
        $rel = $f.FullName.Substring($rootFull.Length + 1).Replace('\', '/')
        if ($Allowed -contains $rel) { continue }
        $code = Get-CodeLines $f.FullName
        $rawLines = [System.IO.File]::ReadAllText($f.FullName) -split "`r?`n"
        for ($i = 0; $i -lt $code.Count; $i++) {
            foreach ($p in $Patterns) {
                if ($code[$i] -match $p) {
                    $hits.Add([pscustomobject]@{ path = $rel; line = $i + 1; text = $rawLines[$i].Trim(); used = $false })
                    break
                }
            }
        }
    }
    $entries = @()
    if ($pendingFile -and (Test-Path $pendingFile)) {
        $n = 0
        foreach ($l in [System.IO.File]::ReadAllLines($pendingFile)) {
            $n += 1
            if ($l.Trim() -eq "" -or $l.TrimStart().StartsWith('#')) { continue }
            $c = $l -split "`t"
            if ($c.Count -lt 4 -or -not ($c[2] -match '^\d+$')) {
                return @{ rc = 2; lines = @("raw free check: FAIL (usage: ${pendingFile}:$n is not path<TAB>text<TAB>count<TAB>step)") }
            }
            $entries += [pscustomobject]@{ path = $c[0].Trim(); text = $c[1].Trim(); count = [int]$c[2]; step = $c[3].Trim(); n = $n }
        }
    }
    $msgs = New-Object System.Collections.Generic.List[string]
    $stale = 0
    foreach ($e in $entries) {
        $m = @($hits | Where-Object { -not $_.used -and $_.path -eq $e.path -and $_.text -eq $e.text })
        if ($m.Count -ne $e.count) {
            $stale += 1
            $msgs.Add("  STALE pending entry line $($e.n): $($e.path) '$($e.text)' x$($e.count) ($($e.step)) matches $($m.Count) site(s): delete or fix the entry")
            continue
        }
        foreach ($h in $m) {
            $h.used = $true
            $msgs.Add("  pending $($e.step): $($h.path):$($h.line): $($h.text)")
        }
    }
    $bad = @($hits | Where-Object { -not $_.used })
    foreach ($h in $bad) { $msgs.Add("  RAW FREE $($h.path):$($h.line): $($h.text)  (use device::DevBuf / PinnedBuf / EventH / StreamH, src/device/handles.h)") }
    $np = @($hits | Where-Object { $_.used }).Count
    if ($bad.Count -gt 0 -or $stale -gt 0) {
        $msgs.Add("raw free check: FAIL ($($bad.Count) raw release site(s), $stale stale pending entr(y/ies), $np pending other steps, $($files.Count) files)")
        return @{ rc = 1; lines = $msgs }
    }
    $msgs.Add("raw free check: ok (0 raw release sites, $np pending other steps, $($files.Count) files)")
    return @{ rc = 0; lines = $msgs }
}

if ($SelfTest) {
    $base = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl_raw_free_selftest_" + [System.Diagnostics.Process]::GetCurrentProcess().Id)
    $fails = 0
    function New-Tree([hashtable]$files) {
        if (Test-Path $base) { Remove-Item -Recurse -Force $base }
        foreach ($k in $files.Keys) {
            $p = Join-Path $base $k
            New-Item -ItemType Directory -Force (Split-Path $p) | Out-Null
            [System.IO.File]::WriteAllText($p, $files[$k])
        }
    }
    $T = "`t"
    $cases = @(
        @{ name = "clean tree passes"; want = 0; pend = $null
            files = @{ 'src/a.cpp' = "void f() {`n  auto b = DevBuf::alloc(be, 16);`n  std::free(p);`n  CkptStore::free(c);`n  free(c);`n}`n" } },
        @{ name = "hip::free blocked"; want = 1; pend = $null
            files = @{ 'src/a.cpp' = "void f() { hip::free(p); }`n" } },
        @{ name = "Backend member release blocked"; want = 1; pend = $null
            files = @{ 'src/b.cpp' = "void f(Backend& b) {`n  b.hostFree(h);`n}`n" } },
        @{ name = "pointer member release blocked"; want = 1; pend = $null
            files = @{ 'src/b.h' = "inline void f(Backend* b) { b->eventDestroy(e); }`n" } },
        @{ name = "raw hipStreamDestroy blocked"; want = 1; pend = $null
            files = @{ 'src/c.cpp' = "void f() { (void)hipStreamDestroy(s); }`n" } },
        @{ name = "comments are not code"; want = 0; pend = $null
            files = @{ 'src/d.cpp' = "// hip::free(p) was here`nint x; /* hip::hostFree(h); */ int y;`n/*`n hip::streamDestroy(s);`n*/`n" } },
        @{ name = "whitelisted files pass"; want = 0; pend = $null
            files = @{ 'src/hip/hip_runtime.cpp' = "void free(DevPtr p) { (void)rt().hipFree(ptr(p)); }`n"; 'src/device/handles.h' = "inline void freeDev(Backend& b, DevPtr p) { b.free(p); }`n" } },
        @{ name = "a whitelisted name elsewhere is not whitelisted"; want = 1; pend = $null
            files = @{ 'src/x/hip_runtime.cpp' = "void f() { hip::free(p); }`n" } },
        @{ name = "pending entry covers its site"; want = 0; pend = "src/t.cpp${T}ops_.eventDestroy(ev);${T}2${T}S6a`n"
            files = @{ 'src/t.cpp' = "void a() {`n    ops_.eventDestroy(ev);`n}`nvoid b() {`n        ops_.eventDestroy(ev);`n}`n" } },
        @{ name = "pending count too high is stale"; want = 1; pend = "src/t.cpp${T}ops_.eventDestroy(ev);${T}2${T}S6a`n"
            files = @{ 'src/t.cpp' = "void a() { }`nvoid b() {`n    ops_.eventDestroy(ev);`n}`n" } },
        @{ name = "pending entry after its step merged is stale"; want = 1; pend = "src/t.cpp${T}ops_.hostFree(c);${T}1${T}S6a`n"
            files = @{ 'src/t.cpp' = "void a() { arena.reset(); }`n" } },
        @{ name = "pending does not cover other sites"; want = 1; pend = "src/t.cpp${T}ops_.hostFree(c);${T}1${T}S6a`n"
            files = @{ 'src/t.cpp' = "void a() {`n  ops_.hostFree(c);`n  hip::free(d);`n}`n" } },
        @{ name = "malformed pending file is a usage error"; want = 2; pend = "src/t.cpp ops_.hostFree(c); one S6a`n"
            files = @{ 'src/t.cpp' = "void a() { }`n" } }
    )
    foreach ($c in $cases) {
        New-Tree $c.files
        $pf = ""
        if ($c.pend) { $pf = Join-Path $base "pending.txt"; [System.IO.File]::WriteAllText($pf, $c.pend) }
        $r = Invoke-Check $base $pf
        $ok = ($r.rc -eq $c.want)
        if (-not $ok) { $fails += 1 }
        Write-Host ("  {0} {1}: rc {2} (want {3}) | {4}" -f ($(if ($ok) { "ok  " } else { "FAIL" })), $c.name, $r.rc, $c.want, @($r.lines)[-1])
    }
    if (Test-Path $base) { Remove-Item -Recurse -Force $base }
    if ($fails -eq 0) { Write-Host "raw free check selftest: ok ($($cases.Count) cases)"; exit 0 }
    Write-Host "raw free check selftest: FAIL ($fails of $($cases.Count) cases)"; exit 1
}

if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
if (-not $Pending) { $Pending = Join-Path $PSScriptRoot "raw_free_pending.txt" }
$r = Invoke-Check $Root $Pending
$r.lines | ForEach-Object { Write-Host $_ }
exit $r.rc
