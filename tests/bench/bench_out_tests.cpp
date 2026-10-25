// Unit tests of tests/bench/bench_out.h, the shared JSONL writer of the measurement tools (measure-tools P0).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. CPU only. Usage: whirl-bench-out-tests [--emit FILE]
//   --emit FILE  also writes a sample JSONL file (three rows) for the records-side importer check
//                (`db_record.py jsonl FILE --dry-run`); the tests themselves use a temp file and delete it.
// Exit 0 = all pass.

#include "bench_out.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace whirl::bench;

static int g_fail = 0;

static void check(const char * name, bool ok, const std::string & info = "") {
    std::printf("%s  %s", ok ? "PASS" : "FAIL", name);
    if (!ok && !info.empty()) std::printf("  (%s)", info.c_str());
    std::printf("\n");
    if (!ok) ++g_fail;
}

static Row sample_row() {
    Row r;
    r.tool = "whirl-kernel-test";
    r.tool_ver = "0.2.5-dev";
    r.component = "K-gemv";
    r.cell = "swift.ffn_down.n1";
    r.metric = "kern.gemv.mxfp4";
    r.unit = "us";
    r.set_stats(summarize({81.0, 82.0, 81.5, 80.5, 83.0}));
    r.stop_reason = "rel<=1%";
    Work w;
    w.launches = 12;
    w.seq_hash = "ab12";
    w.arg_hash = "cd34";
    w.copies_h2d = 1;
    w.bytes_h2d = 4096;
    w.syncs = 2;
    w.allocs = 3;
    r.work = w;
    r.code_hash = "c0de";
    r.shape = "5120x17408";
    r.settings = "rot_bytes=536870912;state=burst";
    Anchor a;
    a.ref = "gemv.anchor";
    a.min = 10.25;
    a.max = 10.5;
    a.drift_pct = 2.4390243902439024;
    r.anchor = a;
    r.gpu_arch = "gfx1151";
    r.gpu_pci = "pci:196:0";
    r.started = "2026-10-25T10:00:00";
    r.ended = "2026-10-25T10:00:05";
    return r;
}

static void test_escape() {
    check("escape: quote, backslash, newline, tab, CR",
          json_string("a\"b\\c\nd\te\rf") == "\"a\\\"b\\\\c\\nd\\te\\rf\"", json_string("a\"b\\c\nd\te\rf"));
    check("escape: other control bytes as \\u00XX", json_string(std::string("x\x01y\x1fz", 5)) == "\"x\\u0001y\\u001fz\"",
          json_string(std::string("x\x01y\x1fz", 5)));
    const std::string zh = "\xe9\x87\x8f\xe6\xb8\xac";  // UTF-8 for two CJK characters
    check("escape: UTF-8 bytes >= 0x80 pass through unchanged (llama-bench escapes them)", json_string(zh) == "\"" + zh + "\"",
          json_string(zh));
    check("escape: DEL (0x7f) and '/' unchanged", json_string("\x7f/") == "\"\x7f/\"");
}

static void test_numbers() {
    check("number: shortest round trip (81.5, 0.1, 1e-07, 1e+300)",
          json_number(81.5) == "81.5" && json_number(0.1) == "0.1" && json_number(1e-7) == "1e-07" && json_number(1e300) == "1e+300",
          json_number(0.1) + " " + json_number(1e-7));
    check("number: integral double has no exponent / dot noise (42)", json_number(42.0) == "42", json_number(42.0));
    check("number: NaN / inf -> null", json_number(kUnset) == "null" && json_number(std::numeric_limits<double>::infinity()) == "null");
    check("number: int64 max", json_number(static_cast<int64_t>(9223372036854775807LL)) == "9223372036854775807");
    double back = 0;
    const double v = 2.4390243902439024;
    std::sscanf(json_number(v).c_str(), "%lf", &back);
    check("number: reads back as the same double", back == v, json_number(v));
}

static void test_stats() {
    Stats s = summarize({5, 1, 4, 2, 3});
    check("stats: odd n median / p10 / p90 (type 7) / MAD", s.n == 5 && s.median == 3 && std::fabs(s.p10 - 1.4) < 1e-12 &&
          std::fabs(s.p90 - 4.6) < 1e-12 && s.mad == 1, json_number(s.p10) + " " + json_number(s.p90) + " " + json_number(s.mad));
    s = summarize({1, 2, 3, 10});
    check("stats: even n median 2.5, MAD 1 (|dev| 1.5 .5 .5 7.5)", s.median == 2.5 && s.mad == 1.0, json_number(s.median) + " " + json_number(s.mad));
    s = summarize({7.0, kUnset, std::numeric_limits<double>::infinity()});
    check("stats: non-finite samples dropped", s.n == 1 && s.median == 7 && s.p10 == 7 && s.mad == 0);
    s = summarize({});
    check("stats: empty -> n 0, unset stats", s.n == 0 && std::isnan(s.median));
    Row r;
    r.value = 5.0;
    r.set_stats(summarize({1, 2, 3}));
    check("stats: set_stats keeps an explicit value, fills an unset one", r.value == 5.0 && r.median == 2);
}

static void test_row_json() {
    const std::string got = to_json(sample_row());
    const std::string want =
        "{\"tool\": \"whirl-kernel-test\", \"tool_ver\": \"0.2.5-dev\", \"component\": \"K-gemv\", \"cell\": \"swift.ffn_down.n1\", "
        "\"metric\": \"kern.gemv.mxfp4\", \"unit\": \"us\", \"value\": 81.5, \"n\": 5, \"median\": 81.5, \"p10\": 80.7, \"p90\": 82.6, "
        "\"mad\": 0.5, \"stop_reason\": \"rel<=1%\", \"work\": {\"launches\": 12, \"seq_hash\": \"ab12\", \"arg_hash\": \"cd34\", "
        "\"copies_h2d\": 1, \"bytes_h2d\": 4096, \"copies_d2h\": 0, \"bytes_d2h\": 0, \"copies_d2d\": 0, \"bytes_d2d\": 0, \"syncs\": 2, "
        "\"allocs\": 3, \"pages\": 0}, \"code_hash\": \"c0de\", \"shape\": \"5120x17408\", \"settings\": \"rot_bytes=536870912;state=burst\", "
        "\"anchor\": {\"ref\": \"gemv.anchor\", \"min\": 10.25, \"max\": 10.5, \"drift_pct\": 2.4390243902439024, \"cooldowns\": 0}, "
        "\"gpu_arch\": \"gfx1151\", \"gpu_pci\": \"pci:196:0\", \"started\": \"2026-10-25T10:00:00\", \"ended\": \"2026-10-25T10:00:05\"}";
    check("row: full line, fixed key order", got == want, got);
    Row m;
    m.tool = "t";
    m.metric = "m";
    m.unit = "ns";
    m.value = 3;
    check("row: minimal line (only required keys; empty / unset left out)",
          to_json(m) == "{\"tool\": \"t\", \"metric\": \"m\", \"unit\": \"ns\", \"value\": 3}", to_json(m));
    m.value = kUnset;
    check("row: unset value -> null (importer refuses it)", to_json(m).find("\"value\": null") != std::string::npos, to_json(m));
    m.value = 1;
    m.extra = {{"rows", "4"}, {"note", json_string("a \"b\"")}};
    check("row: extra keys appended as given", to_json(m).find("\"rows\": 4, \"note\": \"a \\\"b\\\"\"}") != std::string::npos, to_json(m));
}

static std::vector<std::string> read_lines(const std::filesystem::path & p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<std::string> out;
    std::string l;
    while (std::getline(f, l)) out.push_back(l);
    return out;
}

static void test_writer() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "whirl-bench-out-tests";
    fs::create_directories(dir);
    // non-ASCII file name given as UTF-8 (a narrow char* path would be read in the ANSI code page)
    const fs::path p = dir / fs::path(reinterpret_cast<const char8_t *>("\xe9\x87\x8f\xe6\xb8\xac_rows.jsonl"));
    fs::remove(p);
    const auto pu8 = p.u8string();
    const std::string ps(reinterpret_cast<const char *>(pu8.c_str()), pu8.size());
    {
        JsonlWriter w(ps, false);
        check("writer: opens a UTF-8 path", w.ok() && w.enabled());
        check("writer: write two rows", w.write(sample_row()) && w.write(sample_row()) && w.lines() == 2);
        // flushed per line: readable before the writer is closed
        check("writer: each line flushed", read_lines(p).size() == 2);
    }
    {
        JsonlWriter w(ps);  // append (default)
        Row r = sample_row();
        r.cell = "swift.ffn_down.n2";
        check("writer: append", w.write(r));
    }
    const auto lines = read_lines(p);
    check("writer: 3 lines, LF only, each the row's JSON", lines.size() == 3 && lines[0] == to_json(sample_row()) &&
          lines[2].find("\"cell\": \"swift.ffn_down.n2\"") != std::string::npos && lines[0].find('\r') == std::string::npos);
    JsonlWriter off("");
    check("writer: empty path = disabled, write() is a no-op success", off.ok() && !off.enabled() && off.write(sample_row()));
    const auto bad_u8 = (dir / "no_such_dir" / "x.jsonl").u8string();
    JsonlWriter bad(std::string(reinterpret_cast<const char *>(bad_u8.c_str()), bad_u8.size()));
    check("writer: unopenable path -> ok() false, write() false", !bad.ok() && !bad.write(sample_row()));
    fs::remove(p);
    fs::remove(dir);
    const std::string t = iso_now();
    check("iso_now: YYYY-MM-DDTHH:MM:SS", t.size() == 19 && t[4] == '-' && t[10] == 'T' && t[13] == ':', t);
}

static bool emit(const std::string & path) {
    JsonlWriter w(path, false);
    Row a = sample_row();
    Row b = sample_row();
    b.cell = "swift.ffn_down.n2";
    b.work.reset();
    b.anchor.reset();
    Row h;
    h.tool = "whirl-host-bench";
    h.tool_ver = "0.2.5-dev";
    h.component = "M3";
    h.cell = "ensure.128k";
    h.metric = "host.pagepool.ensure";
    h.unit = "ns";
    h.set_stats(summarize({812, 790, 840, 805, 820}));
    h.stop_reason = "n";
    h.work = Work{};
    h.settings = "backend=null";
    h.started = h.ended = iso_now();
    h.extra = {{"note", json_string("\xe4\xb8\xad\xe6\x96\x87 note")}};
    return w.ok() && w.write(a) && w.write(b) && w.write(h);
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash keeps the lines already printed
    for (auto t : {test_escape, test_numbers, test_stats, test_row_json, test_writer}) {
        try {
            t();
        } catch (const std::exception & e) {  // an exception is a failure, never a crash without a report
            check("exception", false, e.what());
        }
    }
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--emit") == 0) check("emit sample JSONL", emit(argv[i + 1]), argv[i + 1]);
    }
    std::printf("whirl-bench-out-tests: %d failed\n", g_fail);
    return g_fail ? 1 : 0;
}
