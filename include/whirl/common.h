// Small host-side utilities: memory-mapped files, whole-file reads, UTF-8
// helpers, timing.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace whirl {

// Read-only memory mapping of a whole file (Windows file mapping; the
// pages of a 16+ GB GGUF are only touched when read).
class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept;
    MappedFile& operator=(MappedFile&& o) noexcept;
    ~MappedFile();

    const std::uint8_t* data() const { return data_; }
    std::size_t size() const { return size_; }
    std::span<const std::uint8_t> bytes() const { return {data_, size_}; }
    const std::string& path() const { return path_; }

private:
    void close();
    std::string path_;
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    void* file_ = nullptr;     // HANDLE
    void* mapping_ = nullptr;  // HANDLE
};

// Whole file as bytes; throws std::runtime_error on failure.
std::string readFile(const std::string& path);
void writeFile(const std::string& path, std::string_view data);

// Converts a UTF-8 path/argument to the native wide form (Windows).
std::wstring widen(std::string_view utf8);
std::string narrow(std::wstring_view wide);

// Appends the UTF-8 encoding of cp (cp <= 0x10FFFF; surrogates are encoded
// as-is, like the generic 3-byte form).
void appendUtf8(std::string& out, std::uint32_t cp);

// Byte length of a UTF-8 sequence from its lead byte, by the high nibble
// (1 for ASCII and stray continuation bytes, 2 for C*/D*, 3 for E*, 4 for F*).
inline std::size_t utf8LenFromLead(std::uint8_t c) {
    static constexpr std::uint8_t k[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return k[c >> 4];
}

// True if s is well-formed UTF-8 (no overlongs, surrogates or > U+10FFFF).
bool isValidUtf8(std::string_view s);

// Monotonic clock in seconds.
double nowSeconds();

}  // namespace whirl
