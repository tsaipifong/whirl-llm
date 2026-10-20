# rocm_kpack.dll (redistributed, unmodified)

Why it is here: the GPU runtime of "AMD Software: Adrenalin Edition for ROCm 10.1"
(driver 26.10.41.05), `amdhip64_7.dll` 10.0.3684.0, imports `rocm_kpack.dll`, but the driver
does not install that DLL. Windows then refuses to load the runtime (error 126) and WHIRL cannot
use the GPU (GitHub issue #2). `tools/package_release.ps1` puts this file next to `whirl.exe` and
`whirl-server.exe`; the application folder is searched first for a DLL's dependencies, so the
driver's runtime finds it there. The older runtime 10.0.3679 (Adrenalin 26.8.x) does not import
it and never loads it.

| | |
|---|---|
| File | `rocm_kpack.dll`, 181,248 bytes, SHA-256 `2e9734ffe6f6eefe1ff660a55dc85fffbcd2d66f794fce4878c6d95a55787b48` |
| Built by | AMD (TheRock, ROCm Core SDK 10.1.0 stable release, tag `therock-10.1`, 2026-10-05) |
| Taken from | `https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/rocm_sdk_core-10.1.0-py3-none-win_amd64.whl` (776,978,384 bytes, SHA-256 `b12f1c4cde14ed1cd8864006d17586c6b1e6a4e4695b9597cfe04ed7715e952f`), path `_rocm_sdk_core/bin/rocm_kpack.dll`; the wheel's RECORD lists the same hash |
| Matching runtime | the same wheel ships `amdhip64_7.dll` 10.0.3684.0, the version the ROCm 10.1 driver installs |
| Signature | none (AMD does not Authenticode-sign the TheRock wheel binaries; the wheel's `amdhip64_7.dll` is unsigned too). No signed copy exists: the driver package does not contain the file |
| Imports | KERNEL32, MSVCP140, VCRUNTIME140, VCRUNTIME140_1 and the Universal CRT. Since 0.2.2 the zip ships the three Visual C++ runtime DLLs next to it (redist/vcruntime/SOURCE.md); the Universal CRT is part of Windows 10/11 |
| Source code | ROCm/rocm-systems `shared/kpack` (https://github.com/ROCm/rocm-systems/tree/develop/shared/kpack) |
| License | MIT, Copyright Advanced Micro Devices, Inc. (LICENSE in this folder). Statically linked: Zstandard (BSD-3-Clause, Meta Platforms) and msgpack-c C++ (header-only, Boost Software License 1.0); texts in THIRD_PARTY_NOTICES.md |

Verified (2026-10-20, CPU-only load test, no driver installed): the driver package's
`amdhip64_7.dll` 10.0.3684.0 (Authenticode: Advanced Micro Devices) fails `LoadLibraryExW` with
error 126 alone and loads with this file next to it.
