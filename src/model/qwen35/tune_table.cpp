// qwen35::TuneTable (src/model/qwen35/tune_table.h): the tune cache text (tune4-*.txt) as pure functions over the
// matrices, and Model::writeTune / readTune over every layer's tuneMats (arch025-M3, R-8c).
// SPDX-License-Identifier: Apache-2.0
// Moved from src/model/tune.cpp: the loops of writeTune / readTune are the same text over the flattened matrix list
// (the layers' tuneMats in order), so the lines, their order and the stale-cache partial apply are unchanged.

#include "model/qwen35/tune_table.h"

#include "whirl/model.h"

#include <cstdarg>
#include <cstdio>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

}  // namespace

// Tune table as text: one line per distinct (bucket, type, rows, cols).
std::string tuneText(std::span<Mat* const> mats) {
    std::string out;
    std::map<std::tuple<u32, u32, u32, u32>, bool> seen;
    for (Mat* m : mats)
        for (u32 b = 0; b < n_tune; ++b) {
            const auto key = std::make_tuple(b, static_cast<u32>(m->ty), m->nrows, m->ncols);
            if (seen.count(key)) continue;
            seen[key] = true;
            out += fmt("%u %u %u %u %u\n", b, static_cast<u32>(m->ty), m->nrows, m->ncols, static_cast<u32>(m->tune[b]));
        }
    return out;
}

// Apply buckets [0, n_buckets) of a table written by tuneText. Returns false if
// any matrix shape is missing (stale cache).
bool applyTuneText(std::string_view text, std::size_t n_buckets, std::span<Mat* const> mats) {
    std::map<std::tuple<u32, u32, u32, u32>, std::uint8_t> map;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t e = text.find('\n', pos);
        if (e == std::string_view::npos) e = text.size();
        std::string line(text.substr(pos, e - pos));
        pos = e + 1;
        for (char& c : line)
            if (c == '\r') c = ' ';
        std::istringstream is(line);
        u64 v[5];
        int got = 0;
        for (; got < 5 && (is >> v[got]); ++got) {
        }
        if (got == 0) continue;
        if (got != 5) return false;
        if (v[0] >= n_buckets) continue;
        if (v[4] >= n_choices) return false;
        map[std::make_tuple(static_cast<u32>(v[0]), static_cast<u32>(v[1]), static_cast<u32>(v[2]), static_cast<u32>(v[3]))] =
            static_cast<std::uint8_t>(v[4]);
    }
    for (Mat* m : mats)
        for (u32 b = 0; b < n_buckets; ++b) {
            auto it = map.find(std::make_tuple(b, static_cast<u32>(m->ty), m->nrows, m->ncols));
            if (it == map.end()) return false;
            m->tune[b] = it->second;
        }
    return true;
}

std::string Model::writeTune() const {
    std::vector<Mat*> mats;
    for (const Layer& L : layers) {
        Layer& LL = const_cast<Layer&>(L);
        for (Mat* m : const_cast<Model*>(this)->tuneMats(LL)) mats.push_back(m);
    }
    return tuneText(mats);
}

bool Model::readTune(std::string_view text, std::size_t n_buckets) {
    std::vector<Mat*> mats;
    for (Layer& L : layers)
        for (Mat* m : tuneMats(L)) mats.push_back(m);
    return applyTuneText(text, n_buckets, mats);
}

}  // namespace whirl::qwen35
