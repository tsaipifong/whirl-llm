// Test-only oracle: renders chat requests with llama.cpp's Jinja engine
// (built from an external llama.cpp checkout, MIT) so `whirl-parity template`
// can compare WHIRL's hand-coded templates against the real GGUF template.
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-template-oracle TEMPLATE.jinja CASES.json > out.jsonl
// CASES.json: [{"name": ..., "request": {OpenAI chat request}}, ...]
// Output: one JSON object per line {"name", "ok", "prompt" | "error"}.

#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "json.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// json.cpp asserts through GGML_ASSERT; the oracle does not link ggml.
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "abort at %s:%d: ", file, line);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
    std::abort();
}

static std::string slurp(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: whirl-template-oracle TEMPLATE.jinja CASES.json\n");
        return 2;
    }
    const std::string tmpl = slurp(argv[1]);
    const common_json cases = common_json::parse(slurp(argv[2]));

    jinja::lexer lexer;
    auto toks = lexer.tokenize(tmpl);
    jinja::program prog = jinja::parse_from_tokens(toks);

    std::string out;
    for (size_t i = 0; i < cases.size(); ++i) {
        const common_json& cs = cases.at(i);
        common_json line = common_json::object();
        line["name"] = cs.at("name");
        try {
            const common_json& req = cs.at("request");
            common_json inp = common_json::object();
            inp["messages"] = req.at("messages");
            bool tools_on = req.contains("tools") && req.at("tools").is_array();
            if (req.contains("tool_choice") && req.at("tool_choice").is_string() &&
                std::string(req.at("tool_choice")) == "none")
                tools_on = false;
            if (tools_on) inp["tools"] = req.at("tools");
            if (req.contains("chat_template_kwargs") && req.at("chat_template_kwargs").is_object()) {
                const common_json& kw = req.at("chat_template_kwargs");
                for (const char* k : {"enable_thinking", "reasoning_effort", "preserve_thinking"})
                    if (kw.contains(k) && !kw.at(k).is_null()) inp[k] = kw.at(k);
            }
            for (const char* k : {"enable_thinking", "reasoning_effort"})
                if (req.contains(k) && !req.at(k).is_null()) inp[k] = req.at(k);
            inp["add_generation_prompt"] = true;

            jinja::context ctx(tmpl);
            jinja::global_from_json(ctx, inp, false);
            jinja::runtime rt(ctx);
            const jinja::value res = rt.execute(prog);
            auto parts = jinja::runtime::gather_string_parts(res);
            line["ok"] = true;
            line["prompt"] = parts->as_string().str();
        } catch (const std::exception& ex) {
            line["ok"] = false;
            line["error"] = std::string(ex.what());
        }
        out += line.dump();
        out += '\n';
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
