// whirl-tool: host-side inspection and parity commands.
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-tool devices [--smoke] [--device SPEC]
//   whirl-tool gguf-info MODEL.gguf [--tensors] [--dump]
//   whirl-tool tokenize MODEL.gguf (--text T | --file F | --batch LIST --out-dir D) [--no-parse-special] [--time]
//   whirl-tool detokenize MODEL.gguf --ids "1,2,3" [--no-special]
//   whirl-tool template-test (MODEL.gguf | --kind a|b) (--request REQ.json | --cases CASES.json) [--out F]

#include "whirl/chat.h"
#include "whirl/common.h"
#include "whirl/gguf.h"
#include "whirl/json.h"
#include "whirl/tokenizer.h"
#include "whirl/unicode.h"
#if WHIRL_WITH_HIP
#include "whirl/hip.h"
#endif

#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace whirl;

namespace {

const char* k_usage =
    "usage:\n"
    "  whirl-tool devices [--smoke] [--device SPEC]\n"
    "  whirl-tool gguf-info MODEL.gguf [--tensors] [--dump]\n"
    "  whirl-tool tokenize MODEL.gguf (--text T | --file F | --batch LIST --out-dir D)\n"
    "                      [--no-parse-special] [--time]\n"
    "  whirl-tool detokenize MODEL.gguf --ids \"1,2,3\" [--no-special]\n"
    "  whirl-tool template-test (MODEL.gguf | --kind a|b) (--request REQ.json | --cases CASES.json) [--out F]\n";

struct Args {
    std::vector<std::string> pos;
    std::vector<std::pair<std::string, std::string>> opts;  // "--x" -> value ("" for flags)

    bool flag(const char* name) const {
        for (const auto& o : opts)
            if (o.first == name) return true;
        return false;
    }
    const std::string* value(const char* name) const {
        for (const auto& o : opts)
            if (o.first == name) return &o.second;
        return nullptr;
    }
};

// options that take a value
bool takesValue(const std::string& o) {
    static const char* v[] = {"--device", "--text", "--file", "--batch", "--out-dir", "--ids", "--kind", "--request", "--cases", "--out"};
    for (const char* s : v)
        if (o == s) return true;
    return false;
}

Args parseArgs(const std::vector<std::string>& argv, std::size_t from) {
    Args a;
    for (std::size_t i = from; i < argv.size(); ++i) {
        const std::string& s = argv[i];
        if (s.rfind("--", 0) == 0) {
            if (takesValue(s)) {
                if (i + 1 >= argv.size()) throw std::runtime_error("missing value for " + s);
                a.opts.emplace_back(s, argv[++i]);
            } else {
                a.opts.emplace_back(s, "");
            }
        } else {
            a.pos.push_back(s);
        }
    }
    return a;
}

void out(std::string_view s) { std::fwrite(s.data(), 1, s.size(), stdout); }

std::string idsToString(const std::vector<TokenId>& ids) {
    std::string s = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(ids[i]);
    }
    s += "]\n";
    return s;
}

// ---------------------------------------------------------------------------

int cmdDevices(const Args& a) {
#if WHIRL_WITH_HIP
    const auto devs = hip::listDevices();
    if (devs.empty()) {
        std::printf("no HIP devices\n");
        return 1;
    }
    for (const auto& d : devs) {
        std::printf("dev%d: %s  arch=%s  code-object=%s  mem=%.1f GiB  CUs=%d  wave=%d  clock=%d MHz  pci=%02x:%02x%s\n",
                    d.index, d.name.c_str(), d.gcn_arch.c_str(), hip::archName(hip::archFor(d.gcn_arch)),
                    static_cast<double>(d.total_mem) / (1024.0 * 1024 * 1024), d.compute_units, d.warp_size,
                    d.clock_khz / 1000, d.pci_bus, d.pci_device, d.integrated ? "  integrated" : "");
    }
    std::printf("embedded code objects:");
    for (const auto& o : hip::embeddedObjects()) std::printf(" %s (%llu bytes)", o.arch, *o.size);
    std::printf("\n");
    if (!a.flag("--smoke")) return 0;
    std::vector<int> which;
    if (const std::string* spec = a.value("--device")) {
        const int i = hip::findDevice(*spec);
        if (i < 0) {
            std::fprintf(stderr, "no device matches '%s'\n", spec->c_str());
            return 1;
        }
        which.push_back(i);
    } else {
        for (const auto& d : devs) which.push_back(d.index);
    }
    int fails = 0;
    for (int i : which) {
        hip::setDevice(i);
        const double t0 = nowSeconds();
        const std::string err = hip::smokeTest();
        const double ms = (nowSeconds() - t0) * 1e3;
        if (err.empty()) {
            std::printf("smoke dev%d: PASS (%.0f ms, %llu launches total)\n", i, ms,
                        static_cast<unsigned long long>(hip::launchCount()));
        } else {
            std::printf("smoke dev%d: FAIL: %s\n", i, err.c_str());
            ++fails;
        }
    }
    return fails ? 1 : 0;
#else
    (void)a;
    std::fprintf(stderr, "built without HIP\n");
    return 1;
#endif
}

// Canonical dump used by `whirl-parity gguf` (one line per entry).
void dumpCanonical(const gguf::File& f) {
    std::string s;
    s += "version " + std::to_string(f.version()) + "\n";
    s += "alignment " + std::to_string(f.alignment()) + "\n";
    s += "data_offset " + std::to_string(f.dataOffset()) + "\n";
    for (const auto& kv : f.kv()) {
        const auto& v = kv.value;
        s += "kv ";
        s += kv.key;
        s += ' ';
        s += gguf::valueTypeName(v.type);
        s += ' ';
        if (v.type == gguf::ValueType::array) {
            s += gguf::valueTypeName(v.arr.elem);
            s += ' ';
            s += std::to_string(v.arr.len);
            // a checksum of the element values keeps the dump short
            std::uint64_t h = 1469598103934665603ull;
            auto mix = [&](std::string_view bytes) {
                for (unsigned char c : bytes) h = (h ^ c) * 1099511628211ull;
            };
            const std::string key(kv.key);
            if (v.arr.elem == gguf::ValueType::string) {
                for (auto e : f.stringArray(key)) {
                    mix(e);
                    mix(std::string_view("\0", 1));
                }
            } else if (v.arr.elem == gguf::ValueType::f32 || v.arr.elem == gguf::ValueType::f64) {
                for (double d : f.floatArray(key)) mix(json::pythonFloat(d) + ",");
            } else if (v.arr.elem != gguf::ValueType::array) {
                for (auto i : f.intArray(key)) mix(std::to_string(i) + ",");
            }
            char hb[24];
            std::snprintf(hb, sizeof hb, " %016llx", static_cast<unsigned long long>(h));
            s += hb;
        } else if (v.type == gguf::ValueType::string) {
            json::Value js(std::string(v.str));
            s += json::dumpPython(js);
        } else if (v.isFloat()) {
            s += json::pythonFloat(v.type == gguf::ValueType::f32 ? static_cast<double>(static_cast<float>(v.f)) : v.f);
        } else if (v.type == gguf::ValueType::boolean) {
            s += v.u ? "true" : "false";
        } else if (v.isSigned()) {
            s += std::to_string(v.i);
        } else {
            s += std::to_string(v.u);
        }
        s += '\n';
    }
    for (const auto& t : f.tensors()) {
        s += "tensor ";
        s += t.name;
        std::string tn = gguf::typeName(t.type);
        for (auto& ch : tn) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        s += " " + tn + " " + std::to_string(static_cast<unsigned>(t.type)) + " [";
        for (std::uint32_t d = 0; d < t.n_dims; ++d) {
            if (d) s += ',';
            s += std::to_string(t.ne[d]);
        }
        s += "] " + std::to_string(t.offset) + ' ' + std::to_string(t.nbytes()) + '\n';
    }
    out(s);
}

int cmdGgufInfo(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("gguf-info needs a file");
    const double t0 = nowSeconds();
    const gguf::File f = gguf::File::open(a.pos[0]);
    const double ms = (nowSeconds() - t0) * 1e3;
    if (a.flag("--dump")) {
        dumpCanonical(f);
        return 0;
    }
    std::printf("file: %s\nGGUF v%u, %zu metadata keys, %zu tensors, alignment %llu, data at %llu, size %.2f GiB (parsed in %.1f ms)\n",
                a.pos[0].c_str(), f.version(), f.kv().size(), f.tensors().size(),
                static_cast<unsigned long long>(f.alignment()), static_cast<unsigned long long>(f.dataOffset()),
                static_cast<double>(f.fileSize()) / (1024.0 * 1024 * 1024), ms);
    for (const auto& kv : f.kv()) {
        std::printf("  %-48.*s %-4s %s\n", static_cast<int>(kv.key.size()), kv.key.data(), gguf::valueTypeName(kv.value.type),
                    gguf::File::formatValue(kv.value).c_str());
    }
    // tensor type histogram
    std::vector<std::pair<std::string, std::pair<std::size_t, std::uint64_t>>> hist;
    for (const auto& t : f.tensors()) {
        const std::string n = gguf::typeName(t.type);
        auto it = std::find_if(hist.begin(), hist.end(), [&](const auto& e) { return e.first == n; });
        if (it == hist.end()) {
            hist.push_back({n, {0, 0}});
            it = hist.end() - 1;
        }
        it->second.first += 1;
        it->second.second += t.nbytes();
    }
    std::printf("tensor types:");
    for (const auto& h : hist)
        std::printf(" %s x%zu (%.2f GiB)", h.first.c_str(), h.second.first,
                    static_cast<double>(h.second.second) / (1024.0 * 1024 * 1024));
    std::printf("\n");
    if (a.flag("--tensors")) {
        for (const auto& t : f.tensors()) {
            std::printf("  %-48.*s %-8s [", static_cast<int>(t.name.size()), t.name.data(), gguf::typeName(t.type).c_str());
            for (std::uint32_t d = 0; d < t.n_dims; ++d) std::printf(d ? ", %llu" : "%llu", static_cast<unsigned long long>(t.ne[d]));
            std::printf("]  offset %llu  %llu bytes\n", static_cast<unsigned long long>(t.offset),
                        static_cast<unsigned long long>(t.nbytes()));
        }
    }
    return 0;
}

int cmdTokenize(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("tokenize needs a model file");
    const double t0 = nowSeconds();
    const gguf::File f = gguf::File::open(a.pos[0]);
    const Tokenizer tok = Tokenizer::fromGguf(f);
    const double t_load = nowSeconds() - t0;
    const bool parse_special = !a.flag("--no-parse-special");

    if (const std::string* list = a.value("--batch")) {
        const std::string* dir = a.value("--out-dir");
        if (!dir) throw std::runtime_error("--batch needs --out-dir");
        const std::string names = readFile(*list);
        std::size_t start = 0, n = 0, n_tok = 0;
        const double t1 = nowSeconds();
        while (start < names.size()) {
            std::size_t e = names.find('\n', start);
            if (e == std::string::npos) e = names.size();
            std::string path = names.substr(start, e - start);
            start = e + 1;
            if (!path.empty() && path.back() == '\r') path.pop_back();
            if (path.empty()) continue;
            const std::string text = readFile(path);
            std::string result;
            try {
                const auto ids = tok.encode(text, parse_special);
                n_tok += ids.size();
                result = idsToString(ids);
            } catch (const TokenizerError& ex) {
                result = std::string("ERROR ") + ex.what() + "\n";
            }
            const std::size_t slash = path.find_last_of("/\\");
            writeFile(*dir + "/" + path.substr(slash == std::string::npos ? 0 : slash + 1) + ".ids", result);
            ++n;
        }
        std::fprintf(stderr, "tokenized %zu files, %zu tokens, load %.2f s, encode %.3f s\n", n, n_tok, t_load,
                     nowSeconds() - t1);
        return 0;
    }

    std::string text;
    if (const std::string* t = a.value("--text"))
        text = *t;
    else if (const std::string* p = a.value("--file"))
        text = readFile(*p);
    else
        throw std::runtime_error("tokenize needs --text, --file or --batch");
    const double t1 = nowSeconds();
    const auto ids = tok.encode(text, parse_special);
    const double t_enc = nowSeconds() - t1;
    out(idsToString(ids));
    if (a.flag("--time"))
        std::fprintf(stderr, "%zu bytes -> %zu tokens; load %.2f s, encode %.2f ms (%.0f tokens/s)\n", text.size(),
                     ids.size(), t_load, t_enc * 1e3, static_cast<double>(ids.size()) / (t_enc > 0 ? t_enc : 1e-9));
    return 0;
}

int cmdDetokenize(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("detokenize needs a model file");
    const std::string* s = a.value("--ids");
    if (!s) throw std::runtime_error("detokenize needs --ids");
    const gguf::File f = gguf::File::open(a.pos[0]);
    const Tokenizer tok = Tokenizer::fromGguf(f);
    std::vector<TokenId> ids;
    std::string cur;
    for (char c : *s + ",") {
        if (c >= '0' && c <= '9') {
            cur.push_back(c);
        } else if (!cur.empty()) {
            ids.push_back(std::stoi(cur));
            cur.clear();
        }
    }
    out(tok.decode(ids, !a.flag("--no-special")));
    return 0;
}

int cmdTemplateTest(const Args& a) {
    chat::TemplateKind kind = chat::TemplateKind::a;
    if (const std::string* k = a.value("--kind")) {
        if (*k == "a")
            kind = chat::TemplateKind::a;
        else if (*k == "b")
            kind = chat::TemplateKind::b;
        else
            throw std::runtime_error("--kind must be a or b");
    } else if (!a.pos.empty()) {
        const gguf::File f = gguf::File::open(a.pos[0]);
        kind = chat::detectTemplate(f.getStringOr("tokenizer.chat_template", ""));
    } else {
        throw std::runtime_error("template-test needs MODEL.gguf or --kind");
    }

    auto render = [&](const json::Value& req) -> std::string {
        const chat::ChatRequest r = chat::parseChatRequest(req);
        return chat::renderChat(kind, r.messages, r.tools, r.options);
    };

    std::string result;
    if (const std::string* p = a.value("--request")) {
        const json::Value req = json::parse(readFile(*p));
        try {
            result = render(req);
        } catch (const chat::RequestError& ex) {
            std::fprintf(stderr, "request error: %s\n", ex.what());
            return 2;
        }
    } else if (const std::string* c = a.value("--cases")) {
        // [{"name": ..., "request": {...}}, ...] -> JSON lines {"name", "ok", "prompt" | "error"}
        const json::Value cases = json::parse(readFile(*c));
        if (!cases.isArray()) throw std::runtime_error("--cases must be a JSON array");
        for (const json::Value& cs : cases.asArray()) {
            json::Object line;
            const json::Value* name = cs.get("name");
            line.set("name", name ? *name : json::Value("?"));
            line.set("kind", json::Value(chat::templateName(kind)));
            try {
                const json::Value* req = cs.get("request");
                if (!req) throw chat::RequestError("case has no request");
                line.set("ok", json::Value(true));
                line.set("prompt", json::Value(render(*req)));
            } catch (const chat::RequestError& ex) {
                line.set("ok", json::Value(false));
                line.set("error", json::Value(std::string(ex.what())));
            }
            result += json::dumpPython(json::Value(std::move(line)), true);
            result += '\n';
        }
    } else {
        throw std::runtime_error("template-test needs --request or --cases");
    }
    if (const std::string* o = a.value("--out"))
        writeFile(*o, result);
    else
        out(result);
    return 0;
}

}  // namespace

int wmain(int argc, wchar_t** wargv) {
    _setmode(_fileno(stdout), _O_BINARY);
    std::vector<std::string> argv;
    for (int i = 0; i < argc; ++i) argv.push_back(narrow(wargv[i]));
    if (argv.size() < 2) {
        std::fputs(k_usage, stderr);
        return 2;
    }
    const std::string& cmd = argv[1];
    try {
        const Args a = parseArgs(argv, 2);
        if (cmd == "devices") return cmdDevices(a);
        if (cmd == "gguf-info") return cmdGgufInfo(a);
        if (cmd == "tokenize") return cmdTokenize(a);
        if (cmd == "detokenize") return cmdDetokenize(a);
        if (cmd == "template-test") return cmdTemplateTest(a);
        std::fputs(k_usage, stderr);
        return 2;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
