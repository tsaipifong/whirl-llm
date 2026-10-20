// HIP runtime loader (ARCH-1 D-1): loads the AMD driver's GPU runtime
// (amdhip64_7.dll) explicitly and resolves the HIP functions WHIRL calls into
// one function-pointer table (src/hip/hip_api.h). Replaces linking against
// amdhip64.lib and the MSVC delay-load of whirl.exe / whirl-server.exe.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Free of HIP SDK includes (the release front-end and the
// unit tests include it with plain MSVC).
//
// Load order (fixed; the first file found is the result, a file that is found
// but fails to load is NOT skipped in favour of a later one):
//   1. <folder of the executable>\amdhip64_7.dll, when that file exists
//      (LoadLibraryExW with LOAD_WITH_ALTERED_SEARCH_PATH: its own imports, e.g.
//      the ROCm 10.1 runtime's rocm_kpack.dll, are searched in that folder first);
//   2. otherwise the name in the standard DLL search order (System32, the
//      Windows folders, the current folder, PATH), where the driver installs it.
// This is the order the delay-load import used before (the executable's folder
// comes first in the standard search order too), now written down and logged:
// describe() gives the exact file that was loaded.
#pragma once

#include <cstddef>
#include <string>

namespace whirl::hip::runtime {

inline constexpr const wchar_t* k_dll = L"amdhip64_7.dll";
inline constexpr const char* k_dll_name = "amdhip64_7.dll";

struct LoadResult {
    bool ok = false;
    unsigned long error = 0;     // GetLastError of the failed load: 126 the DLL or one of its imports is missing,
                                 // 127 an import lacks a function; 0 when loaded (or only a HIP function is missing)
    std::wstring path;           // the file loaded / attempted: step 1's full path, or the search result ("" = not found)
    bool app_local = false;      // step 1 (the executable's folder)
    std::string missing_symbol;  // first HIP function the loaded DLL does not export ("" = all resolved)
    std::string version;         // file version of the loaded DLL, e.g. "10.0.3679.0" ("" = unknown / not loaded)
};

// Loads the runtime once for the process (thread-safe; later calls return the
// same result) and installs the function table. The release executables call it
// from gpuPreflight() before any HIP call; every other program loads lazily on
// the first HIP call (a failure there throws std::runtime_error with
// failureText()).
const LoadResult& load();

// The same procedure for another folder / DLL name, without touching the
// process-wide table (unit tests: fake and absent DLLs). With `keep` false the
// module is unloaded again before returning; on failure it is always unloaded.
LoadResult probe(const std::wstring& app_dir, const std::wstring& dll, bool keep = false);

// Folder of the running executable (no trailing separator), or "".
std::wstring exeDir();

// "C:\Windows\System32\amdhip64_7.dll (10.0.3679.0)" for a loaded runtime.
std::string describe(const LoadResult& r);

// One-paragraph English reason for a failed load (developer tools; the release
// executables print the dll_deps diagnosis instead, see release.cpp).
std::string failureText(const LoadResult& r);

// The HIP function names resolved from the DLL, in table order.
std::size_t symbolCount();
const char* symbolName(std::size_t i);

}  // namespace whirl::hip::runtime
