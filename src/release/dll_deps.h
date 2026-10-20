// Why a DLL fails to load with ERROR_MOD_NOT_FOUND (126): the DLL itself is
// missing, or it is present and one of the DLLs it imports is missing.
// Used by the GPU runtime pre-flight check (release.cpp): the ROCm 10.1
// driver's amdhip64_7.dll (10.0.3684) imports rocm_kpack.dll, which that driver
// does not install (GitHub issue #2), and LoadLibrary reports both cases as 126.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Windows only; no HIP.
#pragma once

#include <string>
#include <vector>

namespace whirl::app::dlldeps {

// DLL names imported by the PE32+ image at `path` (import directory, then the
// delay-import directory), as written in the file. Empty when the file cannot
// be read or is not a PE32+ image.
std::vector<std::string> peImports(const std::wstring& path);

// Full path of `dll` in the standard search order (SearchPathW: the exe's
// folder, the current folder, System32, Windows, PATH), or empty.
std::wstring findDll(const std::wstring& dll);

struct Diagnosis {
    std::wstring found_at;                // where the DLL itself is ("" = not found)
    std::vector<std::string> missing;     // imports (direct or one level down) that are not found
    std::vector<std::string> unloadable;  // imports that exist but do not load for another reason
};

// Diagnoses a load failure of `dll` (a name, searched with findDll, or a full
// path). For each import that does not load, looks one level further down,
// so "rocm_kpack.dll is present but vcruntime140.dll is missing" is told apart
// from "rocm_kpack.dll is missing". Loads the imports that are present (the
// caller is about to exit anyway). API-set names (api-ms-*, ext-ms-*) are skipped.
Diagnosis diagnose(const std::wstring& dll);

// User-facing message for a 126 failure of `dll_name`, from a diagnosis.
// `driver_advice` is appended when the DLL itself is not found.
std::string message126(const std::string& dll_name, const Diagnosis& d, const char* driver_advice);

// ERROR_PROC_NOT_FOUND (127): the DLL and all its imports were found, but one
// import lacks a function the DLL needs. For WHIRL the likely cause is a DLL
// shipped next to whirl.exe shadowing a different version the driver expects:
// a future driver may ship its own rocm_kpack.dll (ours is the ROCm 10.1 one),
// or DLLs built with a newer Visual C++ toolset than our app-local runtime.
// The application folder is searched before System32, so ours wins.
struct Shadow127 {
    std::wstring found_at;              // the DLL that failed to load ("" = not located)
    std::wstring app_dir;               // the WHIRL folder (no trailing separator)
    std::vector<std::string> bundled;   // WHIRL-shipped DLLs present in app_dir (k_bundled_dlls order)
    std::wstring other_kpack;           // a rocm_kpack.dll outside app_dir (next to the runtime, or System32), or ""
};

// DLLs the WHIRL zip places next to the executables (tools/package_release.ps1).
inline constexpr const char* k_bundled_dlls[] = {"rocm_kpack.dll", "msvcp140.dll", "vcruntime140.dll",
                                                 "vcruntime140_1.dll"};

// Folder of the running executable, or "".
std::wstring exeDir();

// Collects the facts for a 127 failure of `dll` (a name, searched with findDll,
// or a full path) for a program in `app_dir`. Only checks which files exist;
// loads nothing.
Shadow127 inspect127(const std::wstring& dll, const std::wstring& app_dir);

// User-facing message for a 127 failure of `dll_name`. Says which shipped DLL to
// rename or remove, or to use the WHIRL release that matches the driver; it does
// not tell the user to update the driver (a newer driver is what causes this).
std::string message127(const std::string& dll_name, const Shadow127& s);

}  // namespace whirl::app::dlldeps
