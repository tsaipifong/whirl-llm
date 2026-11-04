// The prefill-GEMM tune cache on disk (PATH-256 P2, v0.2.5.1 PLAN V1-22; progress/pathlen256.DISCOVERY.md v3.1 2b-1):
// which table loadOrTune reads, where it writes the tuned table, and the copy-only import of the tables older WHIRL
// versions wrote. Declared here, defined in src/model/tune.cpp (module model.core); the tests drive it with their own
// Io (tests/model/model_tests.cpp testTuneCache*), so no GPU is needed.
// SPDX-License-Identifier: Apache-2.0
//
// The file of v0.2.5.1 is <LOCALAPPDATA>\whirl\tune5-<p24>-<h16><suffix>.txt (platform::paths::tune5Name: at most 61
// characters, so the path stays under 256 for any model name). Its key is the model file's name AS THE FILE SYSTEM
// STORES IT (FindFirstFileW cFileName: -m swift.gguf, SWIFT.GGUF and the 8.3 name SWIFT~1.GGU give the same file) and
// its size; when FindFirstFileW fails (a folder with traverse but no list permission) the typed name is used, as v0.2.5.
// Older tables (tune4 in \whirl\ and \whirl-r9700\, then tune3 / tune- with buckets 0 / 1 only, the v0.2.5 read order)
// are imported by COPY, never moved or changed: v0.2.5 on the same machine still reads its own tune4. A full tune4 table
// is copied to the tune5 name (CopyFileExW with COPY_FILE_FAIL_IF_EXISTS: an existing tune5 is never overwritten); a
// tune3 / tune- table gives buckets 0 / 1 and the rest is tuned and written in the tune5 format. The first import of a
// model appends one line to <LOCALAPPDATA>\whirl\tune5-imported.txt (tune5 name, then the old file's name on disk, size
// and FILETIME); once the model's tune5 name is in that list, no old file of the model is read again, so deleting the
// tune5 file always tunes again (test (h): a model with tune4 AND tune3). An old path longer than path_budget::kPathMax
// UTF-16 units is skipped (2c #7). Tune only picks kernels ("every choice computes the same bits", tune.cpp
// autotuneGemm): which table is read never changes any output.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace whirl::qwen35::tune_cache {

// What a table file is to the ledger: its name on disk, size and last-write time (FILETIME, 100 ns since 1601)
struct Stamp {
    std::string name;
    std::uint64_t size = 0;
    std::uint64_t last_write = 0;
};

// The file operations of the resolver (realIo(): the Windows calls through platform::fs; the tests replace one or
// more of them). Paths are UTF-8.
struct Io {
    std::function<std::optional<std::string>(const std::string& path)> read;            // the whole file; nullopt = absent / unreadable
    std::function<std::optional<std::string>(const std::string& model_path)> disk_name;  // the name the file system stores; nullopt = unknown
    std::function<bool(const std::string& from, const std::string& to)> copy_no_replace;  // CopyFileExW(.., COPY_FILE_FAIL_IF_EXISTS)
    std::function<bool(const std::string& path, const std::string& line)> append;       // append one line (creates the file)
    std::function<std::optional<Stamp>(const std::string& path)> stamp;                  // of an old table, for the ledger line
};
Io realIo();

enum class Found {
    none,          // nothing usable: tune every bucket
    full,          // a whole table (tune5, or an imported tune4): no tuning
    old_buckets01  // a tune3 / tune- table: buckets 0 / 1 from it, the others tuned
};

struct Plan {
    Found found = Found::none;
    std::string read_path;   // the file the table came from (full / old_buckets01)
    std::string write_path;  // the tune5 file loadOrTune writes after tuning ("" = no cache: no LOCALAPPDATA)
    std::string name;        // the model's name used for the key (on disk, else typed), UTF-8
    std::uint64_t size = 0;  // the model file's size
    std::vector<std::string> notes;  // warnings for the load log (a ledger line that could not be written)
};

// accept(text, n_buckets): apply buckets [0, n_buckets) of a table to the Model (Model::readTune); false = it does not
// fit (stale), the next candidate is tried.
using Accept = std::function<bool(const std::string& text, std::size_t n_buckets)>;

// base: %LOCALAPPDATA% (platform::paths::localAppData(); "" = no cache); model_path: as given (UTF-8); size: the model
// file's size; suffix: device::profile().tune_cache_suffix; n_tune: the number of buckets of a full table.
Plan resolve(const std::string& base, const std::string& model_path, std::uint64_t size, const std::string& suffix,
             std::size_t n_tune, const Accept& accept, const Io& io);

// The tune5 file text: the header line "# model <name> <size>" then the table (Model::writeTune). applyTuneText skips a
// line whose first token is not a number, so a name such as 7b-model.gguf is never read as a table row.
std::string fileText(const std::string& name, std::uint64_t size, const std::string& table);

}  // namespace whirl::qwen35::tune_cache
