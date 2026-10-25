// Shared JSONL writer for WHIRL's measurement tools (measure-tools P0; progress\measure-tools.DESIGN.md section (a)).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Header-only, C++20, standard library only (no HIP, no Windows headers), so every test / bench
// executable (whirl-kernel-test --bench, whirl-host-bench, whirl-stage-bench, whirl-page-bench) can include it.
//
// One output contract for all tools: one JSON object per line per measured cell, written with --jsonl FILE (console
// text stays for humans). The records side imports it with `db_record.py jsonl FILE` into results.db (schema 2:
// measurement component columns + work_traces). Fields (all optional except tool, metric, unit, value):
//   tool, tool_ver, component, cell, metric, unit, value, n, median, p10, p90, mad, stop_reason,
//   work {launches, seq_hash, arg_hash, copies_h2d, bytes_h2d, copies_d2h, bytes_d2h, copies_d2d, bytes_d2d, syncs,
//         allocs, pages},
//   code_hash, shape, settings ("k=v;k=v"), mode, anchor {ref, min, max, drift_pct, cooldowns},
//   gpu_arch, gpu_pci, clock_note, started, ended (local time, "YYYY-MM-DDTHH:MM:SS"), plus extra keys (kept, not dropped).
// Differences from llama-bench's jsonl_printer (tools/llama-bench/llama-bench.cpp, see progress\mt-p0.PRIOR_ART.md):
//   - bytes >= 0x80 are compared as unsigned char, so UTF-8 text passes through unchanged (llama-bench's signed `char`
//     test `c <= 0x1f` escapes every UTF-8 byte);
//   - a non-finite number is never printed as nan / inf (invalid JSON): an optional field is left out, a non-finite
//     `value` is written as null, which the importer refuses (a timing that measured nothing must not become a row);
//   - doubles are printed shortest-round-trip (std::to_chars), so the value read back is the same double.
// Every line is flushed when written: a tool that crashes later keeps the cells it already measured.
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#ifdef _WIN32
#include <share.h>  // _SH_DENYNO
#endif
#include <ctime>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::bench {

inline constexpr double kUnset = std::numeric_limits<double>::quiet_NaN();

// deterministic work done by one cell: compared exactly (DESIGN principle 1, "work first, time second")
struct Work {
    int64_t launches = 0;
    std::string seq_hash;  // launch-sequence hash (src/hip/launch_log.cpp)
    std::string arg_hash;  // argument digest
    int64_t copies_h2d = 0, bytes_h2d = 0;
    int64_t copies_d2h = 0, bytes_d2h = 0;
    int64_t copies_d2d = 0, bytes_d2d = 0;
    int64_t syncs = 0, allocs = 0, pages = 0;
};

// in-run drift control (DESIGN section (d)): the fixed anchor kernel's range and the cooldowns taken
struct Anchor {
    std::string ref;
    double min = kUnset, max = kUnset, drift_pct = kUnset;
    int64_t cooldowns = 0;
};

struct Stats {
    int64_t n = 0;
    double median = kUnset, p10 = kUnset, p90 = kUnset, mad = kUnset;
};

struct Row {
    std::string tool, tool_ver, component, cell, metric, unit;
    double value = kUnset;  // the judged number (usually the median)
    int64_t n = -1;         // samples; < 0 = not written
    double median = kUnset, p10 = kUnset, p90 = kUnset, mad = kUnset;
    std::string stop_reason;  // why sampling stopped ("rel<=1%", "time", "n", "noisy")
    std::optional<Work> work;
    std::string code_hash, shape, settings, mode;
    std::optional<Anchor> anchor;
    std::string gpu_arch, gpu_pci, clock_note, started, ended;
    // extra keys: (name, already-encoded JSON value), e.g. {"rows", "4"} or {"note", json_string("...")}
    std::vector<std::pair<std::string, std::string>> extra;

    void set_stats(const Stats & s) {
        n = s.n;
        median = s.median;
        p10 = s.p10;
        p90 = s.p90;
        mad = s.mad;
        if (std::isnan(value)) value = s.median;
    }
};

// ---------------------------------------------------------------- statistics
// quantile with linear interpolation between closest ranks (numpy's default "linear" / Hyndman-Fan type 7)
inline double quantile_sorted(const std::vector<double> & v, double q) {
    if (v.empty()) return kUnset;
    if (v.size() == 1) return v[0];
    const double h = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(h));
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (h - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

// n, median, p10, p90 and MAD (median absolute deviation from the median, unscaled) of the finite samples
inline Stats summarize(std::vector<double> samples) {
    samples.erase(std::remove_if(samples.begin(), samples.end(), [](double x) { return !std::isfinite(x); }), samples.end());
    Stats s;
    s.n = static_cast<int64_t>(samples.size());
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.median = quantile_sorted(samples, 0.5);
    s.p10 = quantile_sorted(samples, 0.1);
    s.p90 = quantile_sorted(samples, 0.9);
    std::vector<double> dev;
    dev.reserve(samples.size());
    for (double x : samples) dev.push_back(std::fabs(x - s.median));
    std::sort(dev.begin(), dev.end());
    s.mad = quantile_sorted(dev, 0.5);
    return s;
}

// ---------------------------------------------------------------- JSON encoding
inline std::string json_string(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 2);
    o += '"';
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    o += "\\u00";
                    o += hex[c >> 4];
                    o += hex[c & 15];
                } else {
                    o += ch;  // >= 0x80: UTF-8 byte, unchanged
                }
        }
    }
    o += '"';
    return o;
}

// shortest text that reads back as the same double; non-finite -> "null"
inline std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

inline std::string json_number(int64_t v) {
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

namespace detail {
struct Obj {
    std::string s = "{";
    bool first = true;
    void key(std::string_view k) {
        if (!first) s += ", ";
        first = false;
        s += json_string(k);
        s += ": ";
    }
    void raw(std::string_view k, std::string_view v) { key(k); s += v; }
    void str(std::string_view k, const std::string & v, bool always = false) {
        if (always || !v.empty()) raw(k, json_string(v));
    }
    void num(std::string_view k, double v) {
        if (std::isfinite(v)) raw(k, json_number(v));
    }
    void i64(std::string_view k, int64_t v) { raw(k, json_number(v)); }
    std::string done() { return s + "}"; }
};
}  // namespace detail

inline std::string to_json(const Work & w) {
    detail::Obj o;
    o.i64("launches", w.launches);
    o.str("seq_hash", w.seq_hash, true);
    o.str("arg_hash", w.arg_hash, true);
    o.i64("copies_h2d", w.copies_h2d);
    o.i64("bytes_h2d", w.bytes_h2d);
    o.i64("copies_d2h", w.copies_d2h);
    o.i64("bytes_d2h", w.bytes_d2h);
    o.i64("copies_d2d", w.copies_d2d);
    o.i64("bytes_d2d", w.bytes_d2d);
    o.i64("syncs", w.syncs);
    o.i64("allocs", w.allocs);
    o.i64("pages", w.pages);
    return o.done();
}

inline std::string to_json(const Anchor & a) {
    detail::Obj o;
    o.str("ref", a.ref);
    o.num("min", a.min);
    o.num("max", a.max);
    o.num("drift_pct", a.drift_pct);
    o.i64("cooldowns", a.cooldowns);
    return o.done();
}

// one line (without the newline). Required keys are always written; empty strings / unset numbers are left out.
inline std::string to_json(const Row & r) {
    detail::Obj o;
    o.str("tool", r.tool, true);
    o.str("tool_ver", r.tool_ver);
    o.str("component", r.component);
    o.str("cell", r.cell);
    o.str("metric", r.metric, true);
    o.str("unit", r.unit, true);
    o.raw("value", json_number(r.value));  // null when unset: the importer refuses the line
    if (r.n >= 0) o.i64("n", r.n);
    o.num("median", r.median);
    o.num("p10", r.p10);
    o.num("p90", r.p90);
    o.num("mad", r.mad);
    o.str("stop_reason", r.stop_reason);
    if (r.work) o.raw("work", to_json(*r.work));
    o.str("code_hash", r.code_hash);
    o.str("shape", r.shape);
    o.str("settings", r.settings);
    o.str("mode", r.mode);
    if (r.anchor) o.raw("anchor", to_json(*r.anchor));
    o.str("gpu_arch", r.gpu_arch);
    o.str("gpu_pci", r.gpu_pci);
    o.str("clock_note", r.clock_note);
    o.str("started", r.started);
    o.str("ended", r.ended);
    for (const auto & [k, v] : r.extra) o.raw(k, v);
    return o.done();
}

// local time as results.db stores it: "YYYY-MM-DDTHH:MM:SS"
inline std::string iso_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

// ---------------------------------------------------------------- the file
// Appends lines to FILE (UTF-8 path). RAII: closed by the destructor; not copyable. An empty path = disabled
// (write() returns true and does nothing), so tools can pass their --jsonl option through unconditionally.
class JsonlWriter {
public:
    JsonlWriter() = default;
    explicit JsonlWriter(const std::string & path, bool append = true) { open(path, append); }
    JsonlWriter(const JsonlWriter &) = delete;
    JsonlWriter & operator=(const JsonlWriter &) = delete;
    ~JsonlWriter() { close(); }

    bool open(const std::string & path, bool append = true) {
        close();
        path_ = path;
        if (path.empty()) return true;
        const std::filesystem::path p(reinterpret_cast<const char8_t *>(path.c_str()));
#ifdef _WIN32
        // shared read / write (not _wfopen_s, which denies sharing): a runner can tail the file while the tool runs
        f_ = _wfsopen(p.c_str(), append ? L"ab" : L"wb", _SH_DENYNO);
#else
        f_ = std::fopen(p.c_str(), append ? "ab" : "wb");
#endif
        return f_ != nullptr;
    }
    void close() {
        if (f_) std::fclose(f_);
        f_ = nullptr;
    }
    bool enabled() const { return !path_.empty(); }
    bool ok() const { return path_.empty() || f_ != nullptr; }
    const std::string & path() const { return path_; }
    int64_t lines() const { return lines_; }

    // one row = one line, flushed; false on an I/O error (the caller reports it: a lost row is a lost number)
    bool write(const Row & r) {
        if (path_.empty()) return true;
        if (!f_) return false;
        const std::string s = to_json(r) + "\n";
        const bool good = std::fwrite(s.data(), 1, s.size(), f_) == s.size() && std::fflush(f_) == 0;
        if (good) ++lines_;
        return good;
    }

private:
    std::string path_;
    std::FILE * f_ = nullptr;
    int64_t lines_ = 0;
};

}  // namespace whirl::bench
