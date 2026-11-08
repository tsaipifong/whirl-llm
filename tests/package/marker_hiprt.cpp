// A marker "AMD runtime" for the PACKAGE_VERIFY path cases (tools/gates/pkg_pathlen_check.py; PATH-256 P5, PLAN B26-6 /
// V1-22, progress/pathlen256.DISCOVERY.md v3.1 section 2d-3). Built as amdhip64_7.dll into <build>/pkg_marker/ (never next
// to the build's own exes). Test only, not shipped.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. The check copies it next to whirl.exe / whirl-server.exe: the exe folder's amdhip64_7.dll is the
// one src/hip/runtime_loader.cpp loads (step 1, no fall-back to System32), so the driver's runtime is never loaded and
// no GPU is touched. On DLL_PROCESS_ATTACH it writes whirl_marker_loaded.txt next to itself: the file exists exactly
// when an exe loaded "the AMD runtime". It exports no HIP function, so a load ends in WHIRL's missing-symbol error.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern "C" BOOL WINAPI DllMain(HINSTANCE self, DWORD reason, LPVOID) {
    if (reason != DLL_PROCESS_ATTACH) return TRUE;
    wchar_t path[1024];
    const DWORD n = GetModuleFileNameW(self, path, 1024);
    if (n == 0 || n >= 1024) return TRUE;
    DWORD s = n;
    while (s > 0 && path[s - 1] != L'\\' && path[s - 1] != L'/') --s;
    const wchar_t name[] = L"whirl_marker_loaded.txt";
    if (s + sizeof(name) / sizeof(name[0]) > 1024) return TRUE;
    for (DWORD i = 0; i < sizeof(name) / sizeof(name[0]); ++i) path[s + i] = name[i];
    const HANDLE h = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        const char text[] = "amdhip64_7.dll marker loaded\r\n";
        DWORD w = 0;
        WriteFile(h, text, sizeof(text) - 1, &w, nullptr);
        CloseHandle(h);
    }
    return TRUE;
}
