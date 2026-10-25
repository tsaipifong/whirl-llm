# SPDX-License-Identifier: Apache-2.0
# Package smoke test (arch025-Q6, release_gate step pkg_smoke; ARCH-1 8.7 condition 2 "package smoke automated"):
# the release zip unpacked into a clean folder runs with only the AMD driver, the way a user starts it. Automates the
# R7 clean-folder run of 0.2.3 / 0.2.4 (scratch\release024\clean24.ps1, PACKAGE_VERIFY.json clean_folder_run) and adds
# the run WITHOUT rocm_kpack.dll (issue #2: the ROCm 10.1 driver's runtime imports it and does not install it).
#
#   pwsh tools/gates/pkg_smoke.ps1 -Build <dir> -Model <gguf> -Device r9700|8060s [-Zip <zip>] [-Port 8114] [-Work DIR]
#   pwsh tools/gates/pkg_smoke.ps1 -SelfTest        the verdict on recorded / planted cases + a real member check (CPU)
#
# -Zip: the zip to test; default: tools/package_release.ps1 -BuildDir <Build> -OutDir <Work>\pkg (the same packer the
#       release uses, with its version and import checks). Nothing is uploaded.
# Each variant is a new folder under -Work: Expand-Archive, PATH = the Windows folders only, HIP_PATH* / ROCM* / WHIRL_* /
# GPU visibility variables removed (restored afterwards), every exe started from the unpacked folder:
#   kpack   (as shipped): whirl --version, whirl devices, whirl chat <model> --max-tokens 32 --device <dev>,
#           whirl-server --list-devices, one whirl-server reply (--kv-ssd-mode off, -np 1); the modules the server
#           process loaded (amdhip64 / comgr / kpack / Visual C++ runtime) are logged
#   nokpack (rocm_kpack.dll deleted): whirl --version, whirl chat ...
# Verdict (Get-PkgSmokeVerdict):
#   members  the unpacked folder has whirl.exe, whirl-server.exe, rocm_kpack.dll, msvcp140.dll, vcruntime140.dll,
#            vcruntime140_1.dll, README_FIRST.txt, LICENSE, NOTICE and no other DLL;
#   kpack    every command exit 0, the server answered (>= 1 completion token); when the loaded HIP runtime does not
#            import rocm_kpack.dll, the server must not have loaded it (it would come from the zip for nothing);
#   nokpack  loaded runtime imports rocm_kpack.dll (ROCm 10.1 driver): chat must refuse (exit != 0) with a message naming
#            rocm_kpack.dll (src/release/dll_deps.cpp), not crash silently; else: --version and chat exit 0.
# The loaded runtime is the amdhip64_7.dll the kpack server process had loaded (else System32\amdhip64_7.dll); its
# imports are read from the PE import table of that file (no dumpbin).
# Output: "pkg smoke: REPORT ..." lines, then "pkg smoke: ok (...)" exit 0 / "pkg smoke: FAIL (<reason>)" exit 1;
# setup error exit 2. Kills only the server it started (by PID). Unpacked folders are deleted at the end (logs kept).
param([string]$Build = "", [string]$Model = "", [ValidateSet("r9700", "8060s")][string]$Device = "r9700", [string]$Zip = "",
      [int]$Port = 8114, [string]$Work = "", [switch]$SelfTest)
$ErrorActionPreference = "Stop"

$required = @("whirl.exe", "whirl-server.exe", "rocm_kpack.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "README_FIRST.txt", "LICENSE", "NOTICE")
$allowedDll = @("rocm_kpack.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")

# members: file names of the unpacked folder (top level). Returns "" when fine, else the reason.
function Test-PkgMembers([string[]]$members) {
    $low = @($members | ForEach-Object { $_.ToLowerInvariant() })
    $miss = @($required | Where-Object { $low -notcontains $_.ToLowerInvariant() })
    if ($miss.Count) { return "member_missing:$($miss -join ',')" }
    $extra = @($low | Where-Object { $_ -like "*.dll" -and $allowedDll -notcontains $_ })
    if ($extra.Count) { return "extra_dll:$($extra -join ',')" }
    return ""
}

# o: @{ members = [string[]]; rtKpack = bool (the loaded HIP runtime imports rocm_kpack.dll);
#       k = @{ ver; dev; chat; list = exit codes; up = bool; tokens = int; kpackLoaded = bool };
#       n = @{ ver; chat = exit codes; text = chat output } }  ->  @{ v = PASS|FAIL; why }
function Get-PkgSmokeVerdict($o) {
    $m = Test-PkgMembers $o.members
    if ($m) { return @{ v = "FAIL"; why = $m } }
    foreach ($k in "ver", "dev", "chat", "list") { if ($o.k[$k] -ne 0) { return @{ v = "FAIL"; why = "with_kpack:$k exit $($o.k[$k])" } } }
    if (-not $o.k.up -or $o.k.tokens -lt 1) { return @{ v = "FAIL"; why = "with_kpack:server (up $($o.k.up), tokens $($o.k.tokens))" } }
    if (-not $o.rtKpack -and $o.k.kpackLoaded) { return @{ v = "FAIL"; why = "kpack_loaded_unneeded" } }
    if ($o.rtKpack) {
        if ($o.n.chat -eq 0) { return @{ v = "FAIL"; why = "no_kpack:chat ran without rocm_kpack.dll on a runtime that imports it" } }
        if ("$($o.n.text)" -notmatch "rocm_kpack\.dll") { return @{ v = "FAIL"; why = "no_kpack:refused without naming rocm_kpack.dll (exit $($o.n.chat))" } }
    } else {
        if ($o.n.ver -ne 0 -or $o.n.chat -ne 0) { return @{ v = "FAIL"; why = "no_kpack:ver exit $($o.n.ver), chat exit $($o.n.chat)" } }
    }
    return @{ v = "PASS"; why = "ok" }
}

# PE import table (data directory 1 imports, 13 delay imports) of a 64-bit image: DLL names, lower case
function Get-PeImportNames([string]$path) {
    $b = [IO.File]::ReadAllBytes($path)
    $pe = [BitConverter]::ToInt32($b, 0x3c)
    $nsec = [BitConverter]::ToUInt16($b, $pe + 6)
    $optSize = [BitConverter]::ToUInt16($b, $pe + 20)
    $opt = $pe + 24
    if ([BitConverter]::ToUInt16($b, $opt) -ne 0x20b) { throw "$path is not a PE32+ image" }
    $secs = @(for ($i = 0; $i -lt $nsec; $i++) { $s = $opt + $optSize + 40 * $i
        [pscustomobject]@{ va = [BitConverter]::ToUInt32($b, $s + 12); vs = [BitConverter]::ToUInt32($b, $s + 8); raw = [BitConverter]::ToUInt32($b, $s + 20) } })
    $off = { param([uint32]$rva) foreach ($s in $secs) { if ($rva -ge $s.va -and $rva -lt $s.va + [Math]::Max($s.vs, 1)) { return [int]($rva - $s.va + $s.raw) } }; throw "rva $rva outside sections" }
    $names = @()
    foreach ($d in @(@{ idx = 1; size = 20; at = 12 }, @{ idx = 13; size = 32; at = 4 })) {
        $rva = [BitConverter]::ToUInt32($b, $opt + 112 + 8 * $d.idx)
        if ($rva -eq 0) { continue }
        $o = & $off $rva
        while ($true) {
            $n = [BitConverter]::ToUInt32($b, $o + $d.at)
            if ($n -eq 0) { break }
            $p = & $off $n; $e = $p; while ($b[$e] -ne 0) { $e++ }
            $names += [Text.Encoding]::ASCII.GetString($b, $p, $e - $p).ToLowerInvariant()
            $o += $d.size
        }
    }
    return $names
}

if ($SelfTest) {
    $okK = @{ ver = 0; dev = 0; chat = 0; list = 0; up = $true; tokens = 12; kpackLoaded = $false }
    $mem = @($required + @("THIRD_PARTY_NOTICES.md", "PROVENANCE.md", "README.md", "docs"))
    $refuse = "whirl: the GPU runtime C:\Windows\System32\amdhip64_7.dll needs rocm_kpack.dll ... put rocm_kpack.dll from the WHIRL zip next to whirl.exe"
    function C([string]$n, $o, [string]$want, [string]$why) { @{ n = $n; o = $o; want = $want; why = $why } }
    function K($over) { $h = $okK.Clone(); foreach ($k in $over.Keys) { $h[$k] = $over[$k] }; $h }
    $cases = @(
        # MEASURED shape: 0.2.4 clean_r9700 / clean_8060s (System32 3679 runtime: no kpack import, kpack not loaded)
        (C "clean_3679_pass" @{ members = $mem; rtKpack = $false; k = $okK; n = @{ ver = 0; chat = 0; text = "" } } "PASS" "ok"),
        # MEASURED shape: hotfix021 ROCm 10.1 simulation without kpack: the runtime imports it, whirl refuses with the message
        (C "rocm101_nokpack_refuses_pass" @{ members = $mem; rtKpack = $true; k = (K @{ kpackLoaded = $true }); n = @{ ver = 0; chat = 1; text = $refuse } } "PASS" "ok"),
        # planted failures
        (C "member_missing_server" @{ members = @($mem | Where-Object { $_ -ne "whirl-server.exe" }); rtKpack = $false; k = $okK; n = @{ ver = 0; chat = 0 } } "FAIL" "member_missing:whirl-server.exe"),
        (C "member_missing_kpack" @{ members = @($mem | Where-Object { $_ -ne "rocm_kpack.dll" }); rtKpack = $false; k = $okK; n = @{ ver = 0; chat = 0 } } "FAIL" "member_missing:rocm_kpack.dll"),
        (C "extra_hip_sdk_dll" @{ members = @($mem + "amdhip64_7.dll"); rtKpack = $false; k = $okK; n = @{ ver = 0; chat = 0 } } "FAIL" "extra_dll:amdhip64_7.dll"),
        (C "chat_dll_not_found" @{ members = $mem; rtKpack = $false; k = (K @{ chat = -1073741515 }); n = @{ ver = 0; chat = 0 } } "FAIL" "with_kpack:chat exit -1073741515"),
        (C "server_no_reply" @{ members = $mem; rtKpack = $false; k = (K @{ tokens = 0 }); n = @{ ver = 0; chat = 0 } } "FAIL" "with_kpack:server (up True, tokens 0)"),
        (C "kpack_loaded_on_3679" @{ members = $mem; rtKpack = $false; k = (K @{ kpackLoaded = $true }); n = @{ ver = 0; chat = 0 } } "FAIL" "kpack_loaded_unneeded"),
        (C "nokpack_breaks_3679" @{ members = $mem; rtKpack = $false; k = $okK; n = @{ ver = 0; chat = -1073741515 } } "FAIL" "no_kpack:ver exit 0, chat exit -1073741515"),
        (C "rocm101_nokpack_silent_crash" @{ members = $mem; rtKpack = $true; k = (K @{ kpackLoaded = $true }); n = @{ ver = 0; chat = -1073741515; text = "" } } "FAIL" "no_kpack:refused without naming rocm_kpack.dll (exit -1073741515)"),
        (C "rocm101_nokpack_runs" @{ members = $mem; rtKpack = $true; k = (K @{ kpackLoaded = $true }); n = @{ ver = 0; chat = 0; text = "" } } "FAIL" "no_kpack:chat ran without rocm_kpack.dll on a runtime that imports it")
    )
    $ok = 0; $bad = 0
    foreach ($c in $cases) {
        $r = Get-PkgSmokeVerdict $c.o
        if ($r.v -eq $c.want -and $r.why -eq $c.why) { $ok++; Write-Host "ok   $($c.n): $($r.v) $($r.why)" }
        else { $bad++; Write-Host "BAD  $($c.n): got $($r.v) $($r.why), want $($c.want) $($c.why)" }
    }
    # a real zip: members read from an unpacked folder (the code path of a real run), clean and with whirl-server.exe planted out
    $t = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-pkg-smoke-selftest-$PID"
    try {
        Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue
        $stg = Join-Path $t "stage\whirl-0.0.0-windows-x64"; New-Item -ItemType Directory -Force $stg | Out-Null
        foreach ($f in $mem) { if ($f -eq "docs") { New-Item -ItemType Directory -Force (Join-Path $stg "docs") | Out-Null; "x" | Set-Content (Join-Path $stg "docs\a.md") } else { "x" | Set-Content (Join-Path $stg $f) } }
        Compress-Archive -Path $stg -DestinationPath (Join-Path $t "good.zip")
        Remove-Item (Join-Path $stg "whirl-server.exe")
        Compress-Archive -Path $stg -DestinationPath (Join-Path $t "bad.zip")
        foreach ($z in @(@("good.zip", ""), @("bad.zip", "member_missing:whirl-server.exe"))) {
            $u = Join-Path $t "u_$($z[0])"; Expand-Archive (Join-Path $t $z[0]) -DestinationPath $u
            $dir = (Get-ChildItem $u -Recurse -Filter "whirl.exe" | Select-Object -First 1).DirectoryName
            $got = Test-PkgMembers @(Get-ChildItem $dir | ForEach-Object { $_.Name })
            if ($got -eq $z[1]) { $ok++; Write-Host "ok   real zip $($z[0]): '$got'" } else { $bad++; Write-Host "BAD  real zip $($z[0]): '$got', want '$($z[1])'" }
        }
        # the PE reader on a real Windows image: kernel32.dll imports no rocm_kpack.dll, and ntdll.dll is read without error
        $k32 = Get-PeImportNames (Join-Path $env:SystemRoot "System32\kernel32.dll")
        if ($k32.Count -gt 0 -and $k32 -notcontains "rocm_kpack.dll") { $ok++; Write-Host "ok   PE imports of kernel32.dll: $($k32.Count) DLLs, no rocm_kpack.dll" } else { $bad++; Write-Host "BAD  PE imports of kernel32.dll: $($k32 -join ',')" }
    } finally { Remove-Item -Recurse -Force $t -ErrorAction SilentlyContinue }
    Write-Host "pkg smoke selftest: $ok/$($ok + $bad) as expected"
    if ($bad) { exit 1 }; exit 0
}

# ------------------------------------------------------------------------------------------------- real run (GPU)
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-pkg-smoke-$Device" }
New-Item -ItemType Directory -Force $Work | Out-Null
if (-not $Model -or -not (Test-Path -LiteralPath $Model)) { Write-Host "pkg smoke: FAIL (setup: model '$Model' not found)"; exit 2 }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Zip) {
    if (-not $Build) { $Build = Join-Path $repo "build\Release" }
    $pk = @(& (Get-Process -Id $PID).Path -NoProfile -File (Join-Path $repo "tools\package_release.ps1") -BuildDir $Build -OutDir (Join-Path $Work "pkg") 2>&1 | ForEach-Object { "$_" })
    $pkc = $LASTEXITCODE
    $pk | Set-Content (Join-Path $Work "package_release.txt")
    $Zip = @(Get-ChildItem (Join-Path $Work "pkg") -Filter "whirl-*-windows-x64.zip" -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })[0]
    if ($pkc -ne 0 -or -not $Zip) { Write-Host "pkg smoke: FAIL (package_release.ps1 exit $pkc, see $Work\package_release.txt: $(@($pk | Select-Object -Last 1)))"; exit 1 }
}
if (-not (Test-Path -LiteralPath $Zip)) { Write-Host "pkg smoke: FAIL (setup: zip $Zip not found)"; exit 2 }
$zipSha = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "pkg smoke: REPORT zip $Zip sha256 $zipSha device $Device"

$saved = @{}
$clear = @(Get-ChildItem Env: | Where-Object { $_.Name -like "HIP_PATH*" -or $_.Name -like "ROCM*" -or $_.Name -like "WHIRL_*" -or
                                              $_.Name -in @("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "HIP_DEVICE") })
foreach ($v in $clear) { $saved[$v.Name] = $v.Value; [Environment]::SetEnvironmentVariable($v.Name, $null) }
$savedPath = $env:PATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
$obs = @{ members = @(); rtKpack = $false; k = @{ ver = -1; dev = -1; chat = -1; list = -1; up = $false; tokens = 0; kpackLoaded = $false }; n = @{ ver = -1; chat = -1; text = "" } }
$folders = @()
$rt = ""
try {
    foreach ($variant in "kpack", "nokpack") {
        $f = Join-Path $Work "clean_${variant}_$(Get-Date -Format 'HHmmss')"
        $folders += $f
        Expand-Archive $Zip -DestinationPath $f
        $dir = (Get-ChildItem $f -Recurse -Filter "whirl.exe" | Select-Object -First 1).DirectoryName
        if (-not $dir) { Write-Host "pkg smoke: FAIL (no whirl.exe in $Zip)"; exit 1 }
        $log = Join-Path $Work "$variant.txt"
        "variant $variant folder $dir PATH $env:PATH" | Set-Content $log
        if ($variant -eq "kpack") { $obs.members = @(Get-ChildItem $dir | ForEach-Object { $_.Name }) }
        else { Remove-Item (Join-Path $dir "rocm_kpack.dll") -Force -ErrorAction SilentlyContinue }
        "files: $((Get-ChildItem $dir -File | ForEach-Object { $_.Name }) -join ', ')" | Add-Content $log
        $runs = if ($variant -eq "kpack") { [ordered]@{ ver = @("--version"); dev = @("devices"); chat = @("chat", $Model, "Write a Python function that reverses a string.", "--max-tokens", "32", "--device", $Device) } }
                else { [ordered]@{ ver = @("--version"); chat = @("chat", $Model, "Write a Python function that reverses a string.", "--max-tokens", "32", "--device", $Device) } }
        foreach ($k in $runs.Keys) {
            "=== whirl $($runs[$k] -join ' ')" | Add-Content $log
            $ErrorActionPreference = "Continue"
            $out = @(& (Join-Path $dir "whirl.exe") @($runs[$k]) 2>&1 | ForEach-Object { "$_" })
            $rc = $LASTEXITCODE
            $ErrorActionPreference = "Stop"
            $out | Add-Content $log
            "exit=$rc" | Add-Content $log
            if ($variant -eq "kpack") { $obs.k[$k] = $rc } else { $obs.n[$k] = $rc; if ($k -eq "chat") { $obs.n.text = ($out -join "`n") } }
        }
        if ($variant -ne "kpack") { continue }
        "=== whirl-server --list-devices" | Add-Content $log
        $ErrorActionPreference = "Continue"
        & (Join-Path $dir "whirl-server.exe") --list-devices *>> $log
        $obs.k.list = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        $slog = Join-Path $Work "server_pkg_smoke.log"
        $p = Start-Process (Join-Path $dir "whirl-server.exe") -ArgumentList @("`"$Model`"", "--port", "$Port", "--device", $Device, "-np", "1", "--kv-ssd-mode", "off", "--log-file", "`"$slog`"") -PassThru -WindowStyle Hidden
        try {
            for ($i = 0; $i -lt 300 -and -not $obs.k.up; $i++) { try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null; $obs.k.up = $true } catch { if ($p.HasExited) { break }; Start-Sleep 1 } }
            if ($obs.k.up) {
                $body = @{ messages = @(@{ role = "user"; content = "Say hello in one short sentence." }); max_tokens = 32; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5
                $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
                $obs.k.tokens = [int]$r.usage.completion_tokens
                "=== server reply: [$($r.choices[0].message.content)] (finish $($r.choices[0].finish_reason), completion tokens $($r.usage.completion_tokens))" | Add-Content $log
                $mods = @((Get-Process -Id $p.Id).Modules | Where-Object { $_.ModuleName -match "amdhip|comgr|kpack|msvcp140|vcruntime140" })
                "=== server modules:" | Add-Content $log
                $mods | ForEach-Object { "$($_.ModuleName) $($_.FileVersionInfo.FileVersion) $($_.FileName)" } | Add-Content $log
                $obs.k.kpackLoaded = @($mods | Where-Object { $_.ModuleName -ieq "rocm_kpack.dll" }).Count -gt 0
                $rt = @($mods | Where-Object { $_.ModuleName -ieq "amdhip64_7.dll" } | ForEach-Object { $_.FileName })[0]
            } else { "=== server not up (exited $($p.HasExited))" | Add-Content $log }
        } catch { "server error: $_" | Add-Content $log }
        finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
} finally {
    $env:PATH = $savedPath
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    foreach ($f in $folders) { Remove-Item -Recurse -Force $f -ErrorAction SilentlyContinue }
}
if (-not $rt) { $rt = Join-Path $env:SystemRoot "System32\amdhip64_7.dll" }
try { $obs.rtKpack = (Get-PeImportNames $rt) -contains "rocm_kpack.dll" } catch { Write-Host "pkg smoke: FAIL (cannot read the imports of $rt`: $_)"; exit 1 }
Write-Host "pkg smoke: REPORT runtime $rt imports rocm_kpack.dll: $($obs.rtKpack)"
Write-Host "pkg smoke: REPORT kpack: ver $($obs.k.ver) devices $($obs.k.dev) chat $($obs.k.chat) list $($obs.k.list) server up $($obs.k.up) tokens $($obs.k.tokens) kpack loaded $($obs.k.kpackLoaded)"
Write-Host "pkg smoke: REPORT nokpack: ver $($obs.n.ver) chat $($obs.n.chat)"
$v = Get-PkgSmokeVerdict $obs
if ($v.v -eq "PASS") { Write-Host "pkg smoke: ok (zip $(Split-Path $Zip -Leaf) $($zipSha.Substring(0, 16)), $Device, with and without rocm_kpack.dll)"; exit 0 }
Write-Host "pkg smoke: FAIL ($($v.why))"
exit 1
