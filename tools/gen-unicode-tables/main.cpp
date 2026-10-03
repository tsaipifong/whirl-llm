// whirl-gen-unicode: generates src/tokenizer/unicode_tables.inc for the
// WHIRL tokenizer from the Unicode Character Database text files.
// SPDX-License-Identifier: Apache-2.0
//
// Written from the UCD specification (UAX #44):
//   - UnicodeData.txt field 2 (General_Category) -> one category flag per code
//     point (code points not listed are Cn -> UNDEFINED; L* -> LETTER,
//     M* -> MARK, N* -> NUMBER, P* -> PUNCT, S* -> SYMBOL, Z* -> SEPARATOR,
//     Cc/Cf/Co/Cs -> CONTROL); flag values are those of include/whirl/unicode.h.
//     "<..., First>" / "<..., Last>" line pairs denote ranges.
//   - UnicodeData.txt field 13 (Simple_Lowercase_Mapping).
//   - White_Space (PropList.txt): a fixed list of 25 code points, stable since
//     Unicode 6.x; when PropList.txt is present it is parsed and must agree.
//
// llama.cpp's tables (the tokenizer parity reference) are at Unicode 15.1.
// With a 16.0 UCD the code points first assigned in Unicode 16.0
// (DerivedAge.txt, Age=16.0: 5,185 code points in the ranges below) are
// emitted as unassigned by default, giving Unicode 15.1 tables;
// --unicode 16.0 emits the full UCD.
//
// usage: whirl-gen-unicode --ucd DIR [--ucd-version 16.0.0] [--unicode 15.1|16.0] [--out FILE]
//   DIR must contain UnicodeData.txt; PropList.txt is optional (its header
//   line also gives the UCD version). Output goes to stdout without --out.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr std::uint32_t kMaxCp = 0x110000;

enum : std::uint8_t {
    UNDEFINED = 0x01,
    NUMBER = 0x02,
    LETTER = 0x04,
    SEPARATOR = 0x08,
    MARK = 0x10,
    PUNCT = 0x20,
    SYMBOL = 0x40,
    CONTROL = 0x80,
};

// Age=16.0 (code points new in Unicode 16.0), inclusive ranges (UCD data).
constexpr std::pair<std::uint32_t, std::uint32_t> kAge16[] = {
    {0x0897, 0x0897}, {0x1B4E, 0x1B4F}, {0x1B7F, 0x1B7F}, {0x1C89, 0x1C8A}, {0x2427, 0x2429},
    {0x31E4, 0x31E5}, {0xA7CB, 0xA7CD}, {0xA7DA, 0xA7DC}, {0x105C0, 0x105F3}, {0x10D40, 0x10D65},
    {0x10D69, 0x10D85}, {0x10D8E, 0x10D8F}, {0x10EC2, 0x10EC4}, {0x10EFC, 0x10EFC}, {0x11380, 0x11389},
    {0x1138B, 0x1138B}, {0x1138E, 0x1138E}, {0x11390, 0x113B5}, {0x113B7, 0x113C0}, {0x113C2, 0x113C2},
    {0x113C5, 0x113C5}, {0x113C7, 0x113CA}, {0x113CC, 0x113D5}, {0x113D7, 0x113D8}, {0x113E1, 0x113E2},
    {0x116D0, 0x116E3}, {0x11BC0, 0x11BE1}, {0x11BF0, 0x11BF9}, {0x11F5A, 0x11F5A}, {0x13460, 0x143FA},
    {0x16100, 0x16139}, {0x16D40, 0x16D79}, {0x18CFF, 0x18CFF}, {0x1CC00, 0x1CCF9}, {0x1CD00, 0x1CEB3},
    {0x1E5D0, 0x1E5FA}, {0x1E5FF, 0x1E5FF}, {0x1F8B2, 0x1F8BB}, {0x1F8C0, 0x1F8C1}, {0x1FA89, 0x1FA89},
    {0x1FA8F, 0x1FA8F}, {0x1FABE, 0x1FABE}, {0x1FAC6, 0x1FAC6}, {0x1FADC, 0x1FADC}, {0x1FADF, 0x1FADF},
    {0x1FAE9, 0x1FAE9}, {0x1FBCB, 0x1FBEF},
};

// PropList.txt, White_Space.
std::vector<std::uint32_t> builtinWhiteSpace() {
    std::vector<std::uint32_t> v;
    for (std::uint32_t c = 0x09; c <= 0x0D; ++c) v.push_back(c);
    v.push_back(0x20);
    v.push_back(0x85);
    v.push_back(0xA0);
    v.push_back(0x1680);
    for (std::uint32_t c = 0x2000; c <= 0x200A; ++c) v.push_back(c);
    for (std::uint32_t c : {0x2028u, 0x2029u, 0x202Fu, 0x205Fu, 0x3000u}) v.push_back(c);
    return v;
}

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "whirl-gen-unicode: " << msg << "\n";
    std::exit(1);
}

bool readText(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> lines;
    std::size_t p = 0;
    while (p < s.size()) {
        std::size_t e = s.find('\n', p);
        if (e == std::string::npos) e = s.size();
        std::string l = s.substr(p, e - p);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        lines.push_back(std::move(l));
        p = e + 1;
    }
    return lines;
}

std::vector<std::string> splitFields(const std::string& l, char sep) {
    std::vector<std::string> f;
    std::size_t p = 0;
    for (;;) {
        std::size_t e = l.find(sep, p);
        if (e == std::string::npos) {
            f.push_back(l.substr(p));
            break;
        }
        f.push_back(l.substr(p, e - p));
        p = e + 1;
    }
    return f;
}

std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}

std::uint32_t parseHex(const std::string& s) {
    std::string t = trim(s);
    if (t.empty()) die("empty code point field");
    char* end = nullptr;
    unsigned long v = std::strtoul(t.c_str(), &end, 16);
    if (*end != '\0' || v >= kMaxCp) die("bad code point '" + t + "'");
    return static_cast<std::uint32_t>(v);
}

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::uint8_t categoryFlag(const std::string& gc) {
    if (gc.size() != 2) die("bad general category '" + gc + "'");
    if (gc == "Cn") return UNDEFINED;
    switch (gc[0]) {
        case 'C': return CONTROL;  // Cc, Cf, Co, Cs
        case 'L': return LETTER;
        case 'M': return MARK;
        case 'N': return NUMBER;
        case 'P': return PUNCT;
        case 'S': return SYMBOL;
        case 'Z': return SEPARATOR;
        default: die("bad general category '" + gc + "'");
    }
}

// "# PropList-16.0.0.txt" -> "16.0.0"
std::string versionFromHeader(const std::string& text) {
    std::size_t nl = text.find('\n');
    std::string first = text.substr(0, nl);
    std::size_t dash = first.find('-');
    std::size_t dot = first.rfind(".txt");
    if (first.rfind("#", 0) != 0 || dash == std::string::npos || dot == std::string::npos || dot <= dash) return "";
    return first.substr(dash + 1, dot - dash - 1);
}

char hexBuf[64];

const char* fmt(const char* f, std::uint32_t a, std::uint32_t b) {
    std::snprintf(hexBuf, sizeof hexBuf, f, a, b);
    return hexBuf;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ucd_dir, ucd_version, level = "15.1", out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) die(a + " needs a value");
            return argv[++i];
        };
        if (a == "--ucd") ucd_dir = need();
        else if (a == "--ucd-version") ucd_version = need();
        else if (a == "--unicode") level = need();
        else if (a == "--out") out_path = need();
        else die("usage: whirl-gen-unicode --ucd DIR [--ucd-version V] [--unicode 15.1|16.0] [--out FILE]");
    }
    if (ucd_dir.empty()) die("--ucd DIR is required (directory with UnicodeData.txt)");
    if (level != "15.1" && level != "16.0") die("--unicode must be 15.1 or 16.0");

    std::string unicode_data;
    if (!readText(ucd_dir + "/UnicodeData.txt", unicode_data)) die("cannot read " + ucd_dir + "/UnicodeData.txt");

    // White_Space
    std::vector<std::uint32_t> white = builtinWhiteSpace();
    std::string proplist;
    if (readText(ucd_dir + "/PropList.txt", proplist)) {
        if (ucd_version.empty()) ucd_version = versionFromHeader(proplist);
        std::vector<std::uint32_t> ws;
        for (const std::string& raw : splitLines(proplist)) {
            std::string l = raw.substr(0, raw.find('#'));
            std::vector<std::string> f = splitFields(l, ';');
            if (f.size() < 2 || trim(f[1]) != "White_Space") continue;
            std::string r = trim(f[0]);
            std::size_t dd = r.find("..");
            std::uint32_t lo = parseHex(r.substr(0, dd));
            std::uint32_t hi = dd == std::string::npos ? lo : parseHex(r.substr(dd + 2));
            for (std::uint32_t c = lo; c <= hi; ++c) ws.push_back(c);
        }
        std::sort(ws.begin(), ws.end());
        if (ws != white) die("PropList.txt White_Space differs from the built-in list");
    }
    if (ucd_version.empty()) die("UCD version unknown: pass --ucd-version (or provide PropList.txt)");
    if (level == "15.1" && ucd_version.rfind("16.", 0) != 0)
        die("--unicode 15.1 expects a 16.x UCD (found " + ucd_version + "); use --unicode 16.0 for other versions");

    std::vector<std::uint8_t> flags(kMaxCp, UNDEFINED);
    std::vector<std::uint32_t> lower(kMaxCp, 0xFFFFFFFFu);
    std::vector<std::string> lines = splitLines(unicode_data);
    std::size_t n_entries = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].empty()) continue;
        std::vector<std::string> f = splitFields(lines[i], ';');
        if (f.size() != 15) die("UnicodeData.txt line " + std::to_string(i + 1) + ": expected 15 fields");
        std::uint32_t cp = parseHex(f[0]);
        std::uint8_t flag = categoryFlag(f[2]);
        ++n_entries;
        if (endsWith(f[1], ", First>")) {
            if (i + 1 >= lines.size()) die("range without Last line");
            std::vector<std::string> g = splitFields(lines[i + 1], ';');
            if (g.size() != 15 || !endsWith(g[1], ", Last>") || g[2] != f[2]) die("bad range at line " + std::to_string(i + 1));
            std::uint32_t last = parseHex(g[0]);
            for (std::uint32_t c = cp; c <= last; ++c) flags[c] = flag;
            ++i;
            continue;
        }
        flags[cp] = flag;
        std::string lo = trim(f[13]);
        if (!lo.empty()) lower[cp] = parseHex(lo);
    }
    if (level == "15.1") {
        std::size_t n = 0;
        for (auto [a, b] : kAge16) {
            for (std::uint32_t c = a; c <= b; ++c) {
                flags[c] = UNDEFINED;
                lower[c] = 0xFFFFFFFFu;
            }
            n += b - a + 1;
        }
        if (n != 5185) die("internal: Age=16.0 table size");
    }

    std::vector<std::pair<std::uint32_t, std::uint8_t>> ranges{{0, flags[0]}};
    for (std::uint32_t cp = 1; cp < kMaxCp; ++cp)
        if (flags[cp] != ranges.back().second) ranges.push_back({cp, flags[cp]});
    std::vector<std::pair<std::uint32_t, std::uint32_t>> low;
    for (std::uint32_t cp = 0; cp < kMaxCp; ++cp)
        if (lower[cp] != 0xFFFFFFFFu && lower[cp] != cp) low.push_back({cp, lower[cp]});

    std::string o;
    o += "// Generated by tools/gen-unicode-tables (whirl-gen-unicode) from the Unicode\n";
    o += "// Character Database " + ucd_version + " (UnicodeData.txt), emitted at Unicode " + level +
         " level. Do not edit.\n";
    o += "// SPDX-License-Identifier: Apache-2.0\n\n";
    o += "// clang-format off\n";
    o += "static constexpr const char k_unicode_version[] = \"" + level + "\";\n\n";
    o += "// {first code point, category flag}; a range ends where the next begins.\n";
    o += "static constexpr FlagRange k_flag_ranges[] = {\n";
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        o += fmt("{0x%06X,0x%02X},", ranges[i].first, ranges[i].second);
        if (i % 6 == 5) o += "\n";
    }
    o += "\n{0x110000,0x00}};\n\n";
    o += "static constexpr std::uint32_t k_whitespace[] = {\n";
    for (std::size_t i = 0; i < white.size(); ++i) {
        if (i) o += ",";
        o += fmt("0x%04X", white[i], 0);
    }
    o += "};\n\n";
    o += "// {code point, simple lowercase}, sorted by code point.\n";
    o += "static constexpr CasePair k_lowercase[] = {\n";
    for (std::size_t i = 0; i < low.size(); ++i) {
        o += fmt("{0x%05X,0x%05X},", low[i].first, low[i].second);
        if (i % 6 == 5) o += "\n";
    }
    o += "\n};\n// clang-format on\n";

    if (out_path.empty()) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        std::fwrite(o.data(), 1, o.size(), stdout);
    } else {
        std::ofstream f(out_path, std::ios::binary);
        if (!f) die("cannot write " + out_path);
        f.write(o.data(), static_cast<std::streamsize>(o.size()));
    }
    std::cerr << "entries=" << n_entries << " ranges=" << ranges.size() << " whitespace=" << white.size()
              << " lowercase=" << low.size() << " ucd=" << ucd_version << " unicode=" << level << "\n";
    return 0;
}
