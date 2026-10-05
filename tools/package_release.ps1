# WHIRL release packaging: builds the Windows x64 zip from an existing Release build.
# SPDX-License-Identifier: Apache-2.0
#
#   powershell -ExecutionPolicy Bypass -File tools\package_release.ps1 [-BuildDir DIR] [-OutDir DIR] [-RepoUrl URL]
#   (defaults: -BuildDir build\Release, -OutDir out\release, -RepoUrl https://github.com/tsaipifong/whirl-llm)
#
# Packs whirl.exe, whirl-server.exe, LICENSE, NOTICE, THIRD_PARTY_NOTICES.md,
# PROVENANCE.md, README.md, the user documentation under docs\ (same layout as the
# repository, so the README's relative links work offline) and a README_FIRST.txt
# (short pointer to the READMEs) into <OutDir>\whirl-<version>-windows-x64.zip and
# writes its SHA-256 next to it.
# The version comes from project(whirl VERSION ...) in CMakeLists.txt.
# Nothing is uploaded or published.
#
# Before packing it checks that each executable reports that version and that its
# imports are only Windows system DLLs plus the AMD driver's amdhip64_7.dll (no
# Visual C++ runtime, no HIP SDK DLL), so the zip runs with just the driver.

param(
    [string]$BuildDir = "",
    [string]$OutDir = "",
    [string]$RepoUrl = "https://github.com/tsaipifong/whirl-llm"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if ($BuildDir -eq "") { $BuildDir = Join-Path $repo "build\Release" }
if ($OutDir -eq "") { $OutDir = Join-Path $repo "out\release" }

# ---- version (single source: CMakeLists.txt)
$cm = Get-Content (Join-Path $repo "CMakeLists.txt") -Raw
if ($cm -notmatch 'project\(whirl\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)') { throw "no project(whirl VERSION x.y.z) in CMakeLists.txt" }
$version = $Matches[1]
$name = "whirl-$version-windows-x64"
Write-Host "WHIRL $version from $BuildDir"

$exes = @("whirl.exe", "whirl-server.exe")
foreach ($e in $exes) {
    $p = Join-Path $BuildDir $e
    if (-not (Test-Path $p)) { throw "missing $p (build the Release configuration first: build.bat Release)" }
    $v = (& $p --version | Select-Object -First 1)
    if ($v -notmatch [regex]::Escape($version)) { throw "$e reports '$v', expected version $version (stale build?)" }
}

# ---- import check: PE import directory, read directly (no dumpbin needed)
function Get-PeImports([string]$path) {
    $b = [IO.File]::ReadAllBytes($path)
    $pe = [BitConverter]::ToInt32($b, 0x3c)
    $nsec = [BitConverter]::ToUInt16($b, $pe + 6)
    $optSize = [BitConverter]::ToUInt16($b, $pe + 20)
    $opt = $pe + 24
    if ([BitConverter]::ToUInt16($b, $opt) -ne 0x20b) { throw "$path is not a PE32+ image" }
    $secs = @()
    for ($i = 0; $i -lt $nsec; $i++) {
        $s = $opt + $optSize + 40 * $i
        $secs += [pscustomobject]@{ va = [BitConverter]::ToUInt32($b, $s + 12); vs = [BitConverter]::ToUInt32($b, $s + 8); raw = [BitConverter]::ToUInt32($b, $s + 20) }
    }
    function rva2off([uint32]$rva) {
        foreach ($s in $secs) { if ($rva -ge $s.va -and $rva -lt $s.va + [Math]::Max($s.vs, 1)) { return [int]($rva - $s.va + $s.raw) } }
        throw "rva $rva outside sections"
    }
    function cstr([int]$o) { $e = $o; while ($b[$e] -ne 0) { $e++ }; return [Text.Encoding]::ASCII.GetString($b, $o, $e - $o) }
    $names = @()
    # data directory 1 = imports (20-byte descriptors), 13 = delay imports (32-byte descriptors)
    foreach ($dir in @(@{ idx = 1; size = 20; nameAt = 12; delay = $false }, @{ idx = 13; size = 32; nameAt = 4; delay = $true })) {
        $rva = [BitConverter]::ToUInt32($b, $opt + 112 + 8 * $dir.idx)
        if ($rva -eq 0) { continue }
        $o = rva2off $rva
        while ($true) {
            $nrva = [BitConverter]::ToUInt32($b, $o + $dir.nameAt)
            if ($nrva -eq 0) { break }
            $names += [pscustomobject]@{ dll = (cstr (rva2off $nrva)); delay = $dir.delay }
            $o += $dir.size
        }
    }
    return $names
}
$system = @("kernel32.dll", "ws2_32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "bcrypt.dll", "ntdll.dll", "shlwapi.dll", "dbghelp.dll", "dxgi.dll", "winmm.dll")
foreach ($e in $exes) {
    $imports = Get-PeImports (Join-Path $BuildDir $e)
    foreach ($i in $imports) {
        $d = $i.dll.ToLowerInvariant()
        $ok = ($system -contains $d) -or ($d -eq "amdhip64_7.dll")
        $tag = if ($i.delay) { " (delay-loaded)" } else { "" }
        Write-Host ("  {0}: {1}{2}{3}" -f $e, $i.dll, $tag, $(if ($ok) { "" } else { "   <-- not a system / driver DLL" }))
        if (-not $ok) { throw "$e imports $($i.dll): the zip would need it next to the exe (static CRT build expected)" }
    }
}

# ---- stage
$stage = Join-Path $OutDir $name
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $stage | Out-Null
foreach ($e in $exes) { Copy-Item (Join-Path $BuildDir $e) $stage }
Copy-Item (Join-Path $repo "LICENSE") $stage
Copy-Item (Join-Path $repo "NOTICE") $stage
Copy-Item (Join-Path $repo "THIRD_PARTY_NOTICES.md") $stage
Copy-Item (Join-Path $repo "PROVENANCE.md") $stage
Copy-Item (Join-Path $repo "README.md") $stage

# user documentation, same relative layout as the repository (README links resolve offline);
# developer notes (docs\server_model_interface.md) are left out
$docsSrc = Join-Path $repo "docs"
$docsDst = Join-Path $stage "docs"
New-Item -ItemType Directory -Force $docsDst | Out-Null
$userDocs = @("README_zh-TW.md", "quickstart.md", "quickstart_zh-TW.md", "usage.md", "cli.md",
              "windows_security.md", "windows_security_zh-TW.md", "building.md", "building_zh-TW.md",
              "benchmarks.md", "phase1_parity.md", "recipes.md", "recipes_zh-TW.md")
foreach ($d in $userDocs) {
    $p = Join-Path $docsSrc $d
    if (-not (Test-Path $p)) { throw "missing documentation file $p" }
    Copy-Item $p $docsDst
}
Copy-Item -Recurse (Join-Path $docsSrc "guide") $docsDst
if (Test-Path (Join-Path $docsSrc "images")) { Copy-Item -Recurse (Join-Path $docsSrc "images") $docsDst }

$first = @"
WHIRL $version for Windows x64 - LLM inference for the AMD Radeon AI PRO R9700 (RDNA 4)

Needs: Windows 11 (64-bit), an AMD Radeon AI PRO R9700 (gfx1201) and the graphics driver
AMD Software: Adrenalin Edition 26.8.1 or newer. Nothing else (no HIP SDK, no ROCm,
no Visual C++ runtime).

  .\whirl.exe devices
  .\whirl.exe chat MODEL.gguf "Explain TCP slow start in two sentences." --max-tokens 400
  .\whirl-server.exe MODEL.gguf --port 8080     (OpenAI API at http://127.0.0.1:8080/v1)

Read next:
  README.md                      overview, supported models, performance
  docs\quickstart.md             step-by-step first run
  docs\usage.md                  every option, environment variable and exit code
  docs\windows_security.md       SmartScreen / Smart App Control (the programs are unsigned)
  docs\README_zh-TW.md           Traditional Chinese README (also docs\quickstart_zh-TW.md,
                                 docs\guide\zh-TW\usage.md)

Help from the programs: .\whirl.exe --help, .\whirl.exe help env, .\whirl-server.exe --help
Project page: $RepoUrl
License: Apache-2.0 (LICENSE, NOTICE, THIRD_PARTY_NOTICES.md)
"@
[IO.File]::WriteAllText((Join-Path $stage "README_FIRST.txt"), $first.Replace("`n", "`r`n"), [Text.UTF8Encoding]::new($true))

# ---- zip + checksum
$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $name.zip" | Out-File -Encoding ascii (Join-Path $OutDir "$name.zip.sha256")
Write-Host ("{0} ({1:N1} MiB)" -f $zip, ((Get-Item $zip).Length / 1MB))
Write-Host "sha256 $hash"
Get-ChildItem $stage | ForEach-Object { Write-Host ("  {0,-26} {1,12:N0} bytes" -f $_.Name, $_.Length) }
