# Microsoft Visual C++ 2015-2022 runtime (x64), app-local (redistributed, unmodified)

Why it is here: `rocm_kpack.dll` (redist/rocm_kpack, shipped since 0.2.1 for the ROCm 10.1
driver) imports `MSVCP140.dll`, `VCRUNTIME140.dll` and `VCRUNTIME140_1.dll`; so does that
driver's `amd_comgr_3.dll` 3.5. WHIRL itself is linked with the static CRT and imports none of
them. On a PC without the Visual C++ Redistributable, the GPU runtime of the ROCm 10.1 driver
would then not load. `tools/package_release.ps1` puts these three files next to `whirl.exe` and
`whirl-server.exe` (local deployment: the application folder is searched first for a DLL's
dependencies), so the zip needs no Visual C++ install with either driver branch. The
Adrenalin 26.8.x runtime (amdhip64_7.dll 10.0.3679, amd_comgr_3.dll 3.0) imports none of them.
The Universal CRT (`api-ms-win-crt-*`, `ucrtbase.dll`) is part of Windows 10 and 11 and is not
shipped.

| File | Bytes | Version | SHA-256 |
|---|---|---|---|
| `msvcp140.dll` | 557,728 | 14.44.35211.0 | `0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9` |
| `vcruntime140.dll` | 124,544 | 14.44.35211.0 | `d5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066` |
| `vcruntime140_1.dll` | 49,792 | 14.44.35211.0 | `1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f` |

| | |
|---|---|
| Taken from | Visual Studio Build Tools 2022 17.14.39, `VC\Redist\MSVC\14.44.35112\x64\Microsoft.VC143.CRT\` (copied 2026-10-20) |
| Signature | Authenticode, `CN=Microsoft Windows, O=Microsoft Corporation` (valid) |
| Imports | KERNEL32 and the Universal CRT only (`vcruntime140_1.dll` and `msvcp140.dll` also import `vcruntime140.dll`) |
| Why this version | `rocm_kpack.dll`, `amd_comgr_3.dll` 3.5 and `amdhip64_7.dll` 10.0.3684 were linked with linker 14.44 (PE header); the runtime must be at least as new as the toolset that built the DLL, and 14.44 is the newest Visual Studio 2022 runtime |
| License | Microsoft Visual Studio 2022 license terms, "Distributable Code": the files under `[VisualStudioFolder]\VC\redist` may be copied and distributed unmodified with a program (https://learn.microsoft.com/visualstudio/releases/2022/redistribution). Local deployment is a supported deployment model (https://learn.microsoft.com/cpp/windows/deployment-in-visual-cpp) |
| Updating | Windows Update does not service app-local copies. To update, copy the newer files from a newer Visual Studio redist folder, then change the table above and `$vcrt` in `tools/package_release.ps1` |

Shadowing note: a DLL in the application folder is used even when System32 holds a newer copy.
If a future AMD driver ships DLLs built with a newer toolset (14.5x), these 14.44 copies may be
too old for them (load error 127). The GPU pre-flight message then says to rename them (and
`rocm_kpack.dll`) so the system copies are used (src/release/dll_deps.cpp, `message127`).

Verified (2026-10-20, CPU-only load test, no HIP calls, PATH = System32 only): a static-CRT test
exe in a folder with these three files, `rocm_kpack.dll` and the ROCm 10.1 driver's
`amdhip64_7.dll` 10.0.3684 + `amd_comgr_3.dll` 3.5 loads both driver DLLs, and `msvcp140.dll`,
`vcruntime140.dll` and `vcruntime140_1.dll` resolve to the folder copies (System32 holds 14.51
copies on the test PC, so the folder copies take precedence).
