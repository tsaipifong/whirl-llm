// Host-side utilities (see include/whirl/common.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/common.h"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <utility>

namespace whirl {

std::wstring widen(std::string_view utf8) {
    if (utf8.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), w.data(), n);
    return w;
}

std::string narrow(std::wstring_view wide) {
    if (wide.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), s.data(), n, nullptr, nullptr);
    return s;
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    file_ = f;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz)) {
        close();
        throw std::runtime_error("cannot stat " + path);
    }
    size_ = static_cast<std::size_t>(sz.QuadPart);
    if (size_ == 0) return;
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        close();
        throw std::runtime_error("cannot map " + path);
    }
    mapping_ = m;
    data_ = static_cast<const std::uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    if (!data_) {
        close();
        throw std::runtime_error("cannot map view of " + path);
    }
}

MappedFile::MappedFile(MappedFile&& o) noexcept
    : path_(std::move(o.path_)),
      data_(std::exchange(o.data_, nullptr)),
      size_(std::exchange(o.size_, 0)),
      file_(std::exchange(o.file_, nullptr)),
      mapping_(std::exchange(o.mapping_, nullptr)) {}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
        data_ = std::exchange(o.data_, nullptr);
        size_ = std::exchange(o.size_, 0);
        file_ = std::exchange(o.file_, nullptr);
        mapping_ = std::exchange(o.mapping_, nullptr);
    }
    return *this;
}

MappedFile::~MappedFile() { close(); }

void MappedFile::close() {
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
    if (file_) CloseHandle(static_cast<HANDLE>(file_));
    data_ = nullptr;
    mapping_ = file_ = nullptr;
    size_ = 0;
}

std::string readFile(const std::string& path) {
    std::FILE* f = _wfopen(widen(path).c_str(), L"rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string data;
    char buf[1 << 16];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) data.append(buf, n);
    std::fclose(f);
    return data;
}

void writeFile(const std::string& path, std::string_view data) {
    std::FILE* f = _wfopen(widen(path).c_str(), L"wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

void appendUtf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool isValidUtf8(std::string_view s) {
    std::size_t i = 0;
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    while (i < s.size()) {
        const unsigned c = p[i];
        if (c < 0x80) {
            ++i;
            continue;
        }
        std::size_t n;
        std::uint32_t cp, min;
        if ((c & 0xE0) == 0xC0) {
            n = 2, cp = c & 0x1F, min = 0x80;
        } else if ((c & 0xF0) == 0xE0) {
            n = 3, cp = c & 0x0F, min = 0x800;
        } else if ((c & 0xF8) == 0xF0) {
            n = 4, cp = c & 0x07, min = 0x10000;
        } else {
            return false;
        }
        if (i + n > s.size()) return false;
        for (std::size_t k = 1; k < n; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += n;
    }
    return true;
}

double nowSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace whirl
