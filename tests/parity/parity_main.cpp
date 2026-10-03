// whirl-parity: C++ drivers for the Phase 1 parity checks against external
// references (llama.cpp's llama-tokenize and unicode-data.cpp, the test-only
// Jinja template oracle, reference GGUF dumps). Nothing external is copied
// into WHIRL; the references are only read or executed at test time.
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-parity unicode   --inc src/tokenizer/unicode_tables.inc --llama LLAMA_CPP_SRC
//   whirl-parity gguf      --whirl-tool T --ref-dir DIR MODEL.gguf...
//   whirl-parity tokenizer --whirl-tool T --llama-tokenize L --model M.gguf [--model ...]
//                          --work DIR [--jobs 4] [--modes special,nospecial] [--report F] CORPUS...
//   whirl-parity template  --whirl-tool T --oracle O --work DIR --model-a A.gguf --model-b B.gguf
//                          [--cases tests/corpus/template_cases.json] [--show]
//   whirl-parity make-corpus [--out tests/corpus/synthetic]
//
// Every child process runs at IDLE priority.

#include "support.h"

#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace whirl::parity {
int cmdMakeCorpus(const std::string& out_dir);  // make_corpus.cpp
}

using namespace whirl;
using namespace whirl::parity;

namespace {

struct Args {
    std::vector<std::string> pos;
    std::multimap<std::string, std::string> kv;
    std::set<std::string> flags;

    const std::string* value(const std::string& k) const {
        auto it = kv.find(k);
        return it == kv.end() ? nullptr : &it->second;
    }
    std::string req(const std::string& k) const {
        const std::string* v = value(k);
        if (!v) throw std::runtime_error("missing " + k);
        return *v;
    }
    std::vector<std::string> all(const std::string& k) const {
        std::vector<std::string> r;
        auto [a, b] = kv.equal_range(k);
        for (auto it = a; it != b; ++it) r.push_back(it->second);
        return r;
    }
};

Args parseArgs(const std::vector<std::string>& v, std::size_t start, const std::set<std::string>& bool_flags) {
    Args a;
    for (std::size_t i = start; i < v.size(); ++i) {
        const std::string& s = v[i];
        if (s.rfind("--", 0) == 0) {
            if (bool_flags.count(s)) {
                a.flags.insert(s);
            } else {
                if (i + 1 >= v.size()) throw std::runtime_error(s + " needs a value");
                a.kv.emplace(s, v[++i]);
            }
        } else {
            a.pos.push_back(s);
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// unicode: generated tables vs llama.cpp src/unicode-data.cpp

constexpr std::uint32_t kMaxCp = 0x110000;

std::string block(const std::string& text, const std::string& start) {
    const std::size_t i = text.find(start);
    if (i == std::string::npos) throw std::runtime_error("block not found: " + start);
    const std::size_t j = text.find("};", i);
    return text.substr(i, j - i);
}

// all {0xA, 0xB} pairs (whitespace allowed after the comma)
std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs(const std::string& s) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> r;
    std::size_t p = 0;
    while ((p = s.find("{0x", p)) != std::string::npos) {
        char* e = nullptr;
        const unsigned long a = std::strtoul(s.c_str() + p + 3, &e, 16);
        const char* q = e;
        if (*q != ',') {
            ++p;
            continue;
        }
        ++q;
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') ++q;
        if (q[0] != '0' || (q[1] != 'x' && q[1] != 'X')) {
            ++p;
            continue;
        }
        const unsigned long b = std::strtoul(q + 2, &e, 16);
        if (*e == '}') r.push_back({static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)});
        p = static_cast<std::size_t>(e - s.c_str());
    }
    return r;
}

std::vector<std::uint32_t> hexNumbers(const std::string& s) {
    std::vector<std::uint32_t> r;
    std::size_t p = 0;
    while ((p = s.find("0x", p)) != std::string::npos) {
        char* e = nullptr;
        r.push_back(static_cast<std::uint32_t>(std::strtoul(s.c_str() + p + 2, &e, 16)));
        p = static_cast<std::size_t>(e - s.c_str());
    }
    return r;
}

std::vector<std::uint8_t> expand(const std::vector<std::pair<std::uint32_t, std::uint32_t>>& ranges) {
    std::vector<std::uint8_t> out(kMaxCp, 0);
    for (std::size_t i = 0; i + 1 < ranges.size(); ++i)
        for (std::uint32_t c = ranges[i].first; c < ranges[i + 1].first && c < kMaxCp; ++c)
            out[c] = static_cast<std::uint8_t>(ranges[i].second & 0xFF);
    return out;
}

int cmdUnicode(const Args& a) {
    const std::string ref = readFile(joinPath(joinPath(a.req("--llama"), "src"), "unicode-data.cpp"));
    const std::string ours = readFile(a.req("--inc"));
    const auto ref_ranges = pairs(block(ref, "unicode_ranges_flags"));
    const auto our_ranges = pairs(block(ours, "k_flag_ranges"));
    const auto rf = expand(ref_ranges), of = expand(our_ranges);
    std::vector<std::uint32_t> diff;
    for (std::uint32_t c = 0; c < kMaxCp; ++c)
        if (rf[c] != of[c]) diff.push_back(c);

    auto ref_ws = hexNumbers(block(ref, "unicode_set_whitespace"));
    auto our_ws = hexNumbers(block(ours, "k_whitespace"));
    std::sort(ref_ws.begin(), ref_ws.end());
    std::sort(our_ws.begin(), our_ws.end());

    std::map<std::uint32_t, std::uint32_t> ref_low, our_low;
    for (auto [x, y] : pairs(block(ref, "unicode_map_lowercase")))
        if (x != y) ref_low[x] = y;
    for (auto [x, y] : pairs(block(ours, "k_lowercase"))) our_low[x] = y;
    std::vector<std::string> low_diff;
    for (auto [x, y] : ref_low) {
        auto it = our_low.find(x);
        if (it == our_low.end() || it->second != y) {
            char b[80];
            std::snprintf(b, sizeof b, "U+%04X->U+%04X only in llama.cpp", x, y);
            low_diff.push_back(b);
        }
    }
    for (auto [x, y] : our_low) {
        auto it = ref_low.find(x);
        if (it == ref_low.end() || it->second != y) {
            char b[80];
            std::snprintf(b, sizeof b, "U+%04X->U+%04X only in whirl", x, y);
            low_diff.push_back(b);
        }
    }

    std::printf("category flags: %zu ranges (llama.cpp) vs %zu (whirl); %zu code points differ\n", ref_ranges.size(),
                our_ranges.size(), diff.size());
    for (std::size_t i = 0; i < diff.size() && i < 20; ++i)
        std::printf("  U+%04X llama=0x%02X whirl=0x%02X\n", diff[i], rf[diff[i]], of[diff[i]]);
    std::printf("whitespace: %s (%zu vs %zu)\n", ref_ws == our_ws ? "identical" : "DIFFERENT", ref_ws.size(),
                our_ws.size());
    std::printf("lowercase map: %zu vs %zu entries; %zu differing entries\n", ref_low.size(), our_low.size(),
                low_diff.size());
    for (std::size_t i = 0; i < low_diff.size() && i < 20; ++i) std::printf("   %s\n", low_diff[i].c_str());
    const bool ok = diff.empty() && ref_ws == our_ws && low_diff.empty();
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// gguf: whirl-tool gguf-info --dump vs a reference dump (<basename>.dump in
// --ref-dir, produced outside the repository with gguf-py)

int cmdGguf(const Args& a) {
    const std::string tool = a.req("--whirl-tool");
    const std::string ref_dir = a.req("--ref-dir");
    int bad = 0;
    for (const std::string& m : a.pos) {
        const ProcResult r = run({tool, "gguf-info", m, "--dump"});
        const auto ours = splitLines(r.out);
        const std::string ref_path = joinPath(ref_dir, basename(m) + ".dump");
        if (!fileExists(ref_path)) {
            ++bad;
            std::printf("FAIL %s: no reference dump %s\n", m.c_str(), ref_path.c_str());
            continue;
        }
        const auto ref = splitLines(readFile(ref_path));
        std::size_t n_kv = 0, n_t = 0;
        for (const auto& l : ref) {
            if (l.rfind("kv ", 0) == 0) ++n_kv;
            if (l.rfind("tensor ", 0) == 0) ++n_t;
        }
        if (r.exit_code == 0 && ours == ref) {
            std::printf("PASS %s: %zu keys, %zu tensors identical\n", m.c_str(), n_kv, n_t);
            continue;
        }
        ++bad;
        std::printf("FAIL %s (exit %d)\n", m.c_str(), r.exit_code);
        for (std::size_t i = 0; i < ours.size() && i < ref.size(); ++i) {
            if (ours[i] != ref[i]) {
                std::printf("  whirl: %.200s\n  ref  : %.200s\n", ours[i].c_str(), ref[i].c_str());
                break;
            }
        }
        if (ours.size() != ref.size()) std::printf("  line count whirl %zu vs ref %zu\n", ours.size(), ref.size());
    }
    return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// tokenizer: whirl-tool tokenize --batch vs llama-tokenize, per file

std::optional<std::vector<long long>> parseIds(const std::string& text) {
    for (const std::string& raw : splitLines(text)) {
        std::size_t b = raw.find_first_not_of(" \t");
        std::size_t e = raw.find_last_not_of(" \t");
        if (b == std::string::npos) continue;
        const std::string l = raw.substr(b, e - b + 1);
        if (l.front() != '[' || l.back() != ']') continue;
        std::vector<long long> ids;
        const std::string body = l.substr(1, l.size() - 2);
        std::size_t p = 0;
        if (body.find_first_not_of(" \t") == std::string::npos) return ids;
        while (p <= body.size()) {
            std::size_t c = body.find(',', p);
            if (c == std::string::npos) c = body.size();
            ids.push_back(std::stoll(body.substr(p, c - p)));
            p = c + 1;
        }
        return ids;
    }
    return std::nullopt;
}

// Python json.dumps(list_of_ints)
std::string idsJson(const std::vector<long long>& ids) {
    std::string s = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(ids[i]);
    }
    s += "]";
    return s;
}

std::vector<std::string> expandCorpus(const std::vector<std::string>& corpus, const std::string& work) {
    std::vector<std::string> files;
    for (const std::string& c : corpus) {
        if (isDirectory(c)) {
            for (auto& f : listFiles(c)) files.push_back(f);
        } else if (c.size() > 5 && c.compare(c.size() - 5, 5, ".json") == 0) {
            std::string parent = c.substr(0, c.find_last_of("/\\"));
            const std::string d = joinPath(work, "prompts_" + basename(parent));
            makeDirs(d);
            const json::Value items = json::parse(readFile(c));
            for (const json::Value& it : items.asArray()) {
                const std::string p = joinPath(d, it.get("name")->asString() + ".txt");
                writeFile(p, it.get("prompt")->asString());
                files.push_back(p);
            }
        } else {
            files.push_back(c);
        }
    }
    return files;
}

std::string refTokenize(const std::string& llama, const std::string& model, const std::string& path, bool special,
                        const std::string& cache_dir) {
    const std::string data = readFile(path);
    const std::string key = sha256Hex(model + (special ? "S" : "N") + data).substr(0, 32);
    const std::string cp = joinPath(cache_dir, key + ".ref");
    if (fileExists(cp)) return readFile(cp);
    std::vector<std::string> cmd = {llama, "-m", model, "-f", path, "--ids", "--log-disable", "--no-escape", "--no-bos"};
    if (!special) cmd.push_back("--no-parse-special");
    const ProcResult r = run(cmd);
    std::optional<std::vector<long long>> ids;
    if (r.exit_code == 0) ids = parseIds(r.out);
    const std::string res = ids ? idsJson(*ids) : "ERROR";
    writeFile(cp, res);
    return res;
}

int cmdTokenizer(const Args& a) {
    const std::string tool = a.req("--whirl-tool");
    const std::string llama = a.req("--llama-tokenize");
    const std::string work = a.req("--work");
    const std::vector<std::string> models = a.all("--model");
    if (models.empty()) throw std::runtime_error("missing --model");
    const int jobs = a.value("--jobs") ? std::max(1, std::stoi(*a.value("--jobs"))) : 4;
    const std::string modes = a.value("--modes") ? *a.value("--modes") : "special,nospecial";
    makeDirs(work);
    const std::string cache = joinPath(work, "refcache");
    makeDirs(cache);
    const std::vector<std::string> files = expandCorpus(a.pos, work);
    std::map<std::string, std::string> names;
    for (const auto& f : files) {
        const std::string b = basename(f);
        auto it = names.find(b);
        if (it != names.end() && it->second != f) throw std::runtime_error("duplicate corpus basename: " + b);
        names[b] = f;
    }

    std::vector<std::string> report;
    std::size_t total_bad = 0;
    for (const std::string& model : models) {
        std::size_t mstart = 0;
        while (mstart <= modes.size()) {
            std::size_t mend = modes.find(',', mstart);
            if (mend == std::string::npos) mend = modes.size();
            const std::string mode = modes.substr(mstart, mend - mstart);
            mstart = mend + 1;
            const bool special = mode == "special";
            const std::string outdir = joinPath(work, "whirl_" + sha256Hex(model).substr(0, 8) + "_" + mode);
            makeDirs(outdir);
            const std::string listfile = joinPath(work, "list.txt");
            std::string list;
            for (const auto& f : files) list += f + "\n";
            writeFile(listfile, list);
            std::vector<std::string> cmd = {tool, "tokenize", model, "--batch", listfile, "--out-dir", outdir};
            if (!special) cmd.push_back("--no-parse-special");
            const ProcResult r = run(cmd);
            std::fputs(r.err.c_str(), stderr);

            std::vector<std::string> refs(files.size());
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> pool;
            for (int j = 0; j < jobs; ++j)
                pool.emplace_back([&] {
                    for (std::size_t i; (i = next++) < files.size();)
                        refs[i] = refTokenize(llama, model, files[i], special, cache);
                });
            for (auto& t : pool) t.join();

            std::size_t n_ok = 0, n_tok = 0;
            std::vector<std::string> bad;
            for (std::size_t i = 0; i < files.size(); ++i) {
                const std::string ours_txt = readFile(joinPath(outdir, basename(files[i]) + ".ids"));
                std::string ours;
                if (ours_txt.rfind("ERROR", 0) == 0) {
                    ours = "ERROR";
                } else {
                    auto ids = parseIds(ours_txt);
                    ours = ids ? idsJson(*ids) : "UNPARSEABLE";
                }
                const std::string& ref = refs[i];
                if (ours == ref) {
                    ++n_ok;
                    if (ref != "ERROR") n_tok += parseIds(ref)->size();
                } else {
                    bad.push_back(files[i]);
                }
            }
            total_bad += bad.size();
            char line[512];
            std::snprintf(line, sizeof line, "%s [%s]: %zu/%zu files identical (%zu tokens compared)",
                          basename(model).c_str(), mode.c_str(), n_ok, files.size(), n_tok);
            std::printf("%s\n", line);
            report.push_back(line);
            for (std::size_t i = 0; i < bad.size() && i < 20; ++i) {
                std::printf("  MISMATCH %s\n", bad[i].c_str());
                report.push_back("  MISMATCH " + bad[i]);
            }
            std::fflush(stdout);
        }
    }
    if (const std::string* rep = a.value("--report")) {
        std::string s;
        for (const auto& l : report) s += l + "\n";
        writeFile(*rep, s);
    }
    return total_bad ? 1 : 0;
}

// ---------------------------------------------------------------------------
// template: whirl-tool template-test --cases vs the Jinja oracle

std::vector<json::Value> runJsonLines(const std::vector<std::string>& cmd) {
    const ProcResult r = run(cmd);
    if (r.exit_code != 0) throw std::runtime_error("command failed: " + cmd[0] + "\n" + r.err);
    std::vector<json::Value> out;
    for (const auto& l : splitLines(r.out))
        if (l.find_first_not_of(" \t") != std::string::npos) out.push_back(json::parse(l));
    return out;
}

std::string escaped(const std::string& s, std::size_t from, std::size_t len) {
    // keep the cut on a UTF-8 boundary
    while (from > 0 && from < s.size() && (static_cast<unsigned char>(s[from]) & 0xC0) == 0x80) --from;
    std::size_t end = std::min(s.size(), from + len);
    while (end < s.size() && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) ++end;
    std::string o = "\"";
    json::appendEscaped(o, s.substr(from, end - from));
    o += "\"";
    return o;
}

std::string resultText(const json::Value& v) {
    if (v.get("ok") && v.get("ok")->asBool()) return v.get("prompt")->asString();
    const json::Value* e = v.get("error");
    return "ERROR: " + (e && e->isString() ? e->asString() : std::string());
}

int cmdTemplate(const Args& a) {
    const std::string tool = a.req("--whirl-tool");
    const std::string oracle = a.req("--oracle");
    const std::string work = a.req("--work");
    std::string cases_path;
    if (const std::string* c = a.value("--cases")) {
        cases_path = *c;
    } else {
        // default: tests/corpus/template_cases.json next to this source tree
        cases_path = WHIRL_SOURCE_DIR "/tests/corpus/template_cases.json";
    }
    const bool show = a.flags.count("--show") > 0;
    makeDirs(work);
    const json::Value cases = json::parse(readFile(cases_path));
    std::map<std::string, std::set<std::string>> expect;
    for (const json::Value& c : cases.asArray()) {
        std::set<std::string> s;
        if (const json::Value* e = c.get("expect_diverge"))
            for (const json::Value& k : e->asArray()) s.insert(k.asString());
        expect[c.get("name")->asString()] = s;
    }

    std::size_t failed = 0;
    for (auto [kind, model] : {std::pair<std::string, std::string>{"a", a.req("--model-a")},
                               std::pair<std::string, std::string>{"b", a.req("--model-b")}}) {
        const std::string tmpl = joinPath(work, "template_" + kind + ".jinja");
        {
            const gguf::File f = gguf::File::open(model);
            writeFile(tmpl, f.getString("tokenizer.chat_template"));
        }
        const auto ours = runJsonLines({tool, "template-test", "--kind", kind, "--cases", cases_path});
        const auto ref = runJsonLines({oracle, tmpl, cases_path});
        std::size_t n_same = 0, n_both_err = 0, n_expected = 0;
        std::vector<std::tuple<std::string, json::Value, json::Value>> bad;
        for (std::size_t i = 0; i < ours.size() && i < ref.size(); ++i) {
            const json::Value& o = ours[i];
            const json::Value& r = ref[i];
            const std::string name = o.get("name")->asString();
            const bool ook = o.get("ok")->asBool(), rok = r.get("ok")->asBool();
            const bool same = (ook && rok && o.get("prompt")->asString() == r.get("prompt")->asString()) ||
                              (!ook && !rok);
            if (same) {
                (ook ? n_same : n_both_err)++;
                continue;
            }
            if (expect[name].count(kind)) {
                ++n_expected;
                if (show)
                    std::printf("  expected divergence %s/%s: whirl=%s oracle=%s\n", kind.c_str(), name.c_str(),
                                escaped(resultText(o), 0, 160).c_str(), escaped(resultText(r), 0, 160).c_str());
                continue;
            }
            bad.emplace_back(name, o, r);
        }
        if (ours.size() != ref.size()) {
            std::printf("  result count differs: whirl %zu vs oracle %zu\n", ours.size(), ref.size());
            ++failed;
        }
        failed += bad.size();
        std::printf(
            "variant %s (%s): %zu identical prompts, %zu both rejected, %zu documented divergences, %zu MISMATCHES "
            "(of %zu cases)\n",
            kind.c_str(), basename(model).c_str(), n_same, n_both_err, n_expected, bad.size(), cases.asArray().size());
        for (const auto& [name, o, r] : bad) {
            const std::string po = resultText(o), pr = resultText(r);
            std::size_t k = 0;
            while (k < po.size() && k < pr.size() && po[k] == pr[k]) ++k;
            const std::size_t from = k > 60 ? k - 60 : 0;
            std::printf("  MISMATCH %s: first difference at byte %zu\n", name.c_str(), k);
            std::printf("    whirl : %s\n", escaped(po, from, 140).c_str());
            std::printf("    oracle: %s\n", escaped(pr, from, 140).c_str());
        }
    }
    return failed ? 1 : 0;
}

const char* kUsage =
    "usage:\n"
    "  whirl-parity unicode   --inc UNICODE_TABLES.inc --llama LLAMA_CPP_SRC\n"
    "  whirl-parity gguf      --whirl-tool T --ref-dir DIR MODEL.gguf...\n"
    "  whirl-parity tokenizer --whirl-tool T --llama-tokenize L --model M.gguf [--model ...] --work DIR\n"
    "                         [--jobs 4] [--modes special,nospecial] [--report F] CORPUS...\n"
    "  whirl-parity template  --whirl-tool T --oracle O --work DIR --model-a A.gguf --model-b B.gguf\n"
    "                         [--cases CASES.json] [--show]\n"
    "  whirl-parity make-corpus [--out DIR]\n";

}  // namespace

int wmain(int argc, wchar_t** wargv) {
    std::vector<std::string> v;
    for (int i = 0; i < argc; ++i) v.push_back(narrow(wargv[i]));
    if (v.size() < 2) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    lowerOwnPriority();
    try {
        const std::string cmd = v[1];
        const Args a = parseArgs(v, 2, {"--show"});
        if (cmd == "unicode") return cmdUnicode(a);
        if (cmd == "gguf") return cmdGguf(a);
        if (cmd == "tokenizer") return cmdTokenizer(a);
        if (cmd == "template") return cmdTemplate(a);
        if (cmd == "make-corpus")
            return cmdMakeCorpus(a.value("--out") ? *a.value("--out")
                                                   : std::string(WHIRL_SOURCE_DIR "/tests/corpus/synthetic"));
        std::fputs(kUsage, stderr);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "whirl-parity: %s\n", e.what());
        return 1;
    }
}
