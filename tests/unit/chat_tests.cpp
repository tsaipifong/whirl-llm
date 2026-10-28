// WHIRL unit tests of module chat (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-chat-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/json.h"
#include "whirl/chat.h"
#include "whirl/json_fields.h"  // arch025-N6 lane 1 leg 7 (testJsonFields)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// ---------------------------------------------------------------------------
void testChat() {
    auto render = [](chat::TemplateKind k, const char* req) {
        const auto r = chat::parseChatRequest(json::parse(req));
        return chat::renderChat(k, r.messages, r.tools, r.options);
    };
    CHECK_EQ(render(chat::TemplateKind::b, R"({"messages":[{"role":"user","content":" hi "}]})"),
             std::string("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    CHECK_EQ(render(chat::TemplateKind::a, R"({"messages":[{"role":"user","content":"hi"}],"enable_thinking":false})"),
             std::string("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    CHECK_EQ(render(chat::TemplateKind::a,
                    R"({"messages":[{"role":"system","content":"S"},{"role":"user","content":"q"}],"reasoning_effort":"medium"})"),
             std::string("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    // arch024-S11b P3: leading system messages are merged into one system block with one newline between them (the
    // mutation that dropped the newline survived every CPU test and the prompt token pin: no pinned request had two)
    CHECK_EQ(render(chat::TemplateKind::a,
                    R"({"messages":[{"role":"system","content":"A"},{"role":"system","content":"B"},{"role":"user","content":"q"}],"reasoning_effort":"medium"})"),
             std::string("<|im_start|>system\nA\nB<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    CHECK(throws([&] { render(chat::TemplateKind::a, R"({"messages":[]})"); }));
    // reasoning effort: aliases, the OpenRouter-style "reasoning" object, unknown values
    {
        const auto ra = [&](const std::string& extra) {
            return render(chat::TemplateKind::a, (R"({"messages":[{"role":"user","content":"q"}])" + extra + "}").c_str());
        };
        const std::string def = ra(""), xhigh = ra(R"(,"reasoning_effort":"xhigh")"), low = ra(R"(,"reasoning_effort":"low")"),
                          medium = ra(R"(,"reasoning_effort":"medium")"), off = ra(R"(,"enable_thinking":false)");
        CHECK(def == xhigh && low != xhigh && medium != xhigh && medium != low);
        CHECK(xhigh.find("Reasoning effort is set to xhigh") != std::string::npos);
        CHECK(low.find("Reasoning effort is set to low") != std::string::npos);
        for (const char* v : {"high", "max", "ultra", "XHigh", " Max "}) {
            const std::string extra = std::string(",\"reasoning_effort\":\"") + v + "\"";
            CHECK_EQ(ra(extra), xhigh);
        }
        CHECK_EQ(ra(R"(,"reasoning_effort":"minimal")"), low);
        CHECK_EQ(ra(R"(,"reasoning_effort":"MEDIUM")"), medium);
        CHECK_EQ(ra(R"(,"reasoning_effort":"none")"), off);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"low"})"), low);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"ultra"})"), xhigh);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"none"})"), off);
        CHECK_EQ(ra(R"(,"reasoning":{"enabled":false})"), off);
        CHECK_EQ(ra(R"(,"reasoning":{"enabled":true,"effort":"medium"})"), medium);
        CHECK_EQ(ra(R"(,"reasoning":"minimal")"), low);
        CHECK_EQ(ra(R"(,"chat_template_kwargs":{"reasoning_effort":"max"})"), xhigh);
        // later sources override earlier ones: kwargs < reasoning object < top level
        CHECK_EQ(ra(R"(,"chat_template_kwargs":{"reasoning_effort":"low"},"reasoning":{"effort":"medium"})"), medium);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"low"},"reasoning_effort":"medium")"), medium);
        CHECK_EQ(ra(R"(,"reasoning":{"enabled":false},"enable_thinking":true)"), def);
        // a valid effort does not turn thinking back on
        CHECK_EQ(ra(R"(,"enable_thinking":false,"reasoning_effort":"low")"), off);
        // unknown values: no error, default effort, one warning
        CHECK_EQ(ra(R"(,"reasoning_effort":"turbo")"), def);
        CHECK_EQ(ra(R"(,"reasoning":{"effort":"turbo"})"), def);
        CHECK_EQ(ra(R"(,"reasoning_effort":"low","reasoning":{"effort":"turbo"})"), low);
        const auto r1 = chat::parseChatRequest(json::parse(R"({"messages":[{"role":"user","content":"q"}],"reasoning_effort":"turbo"})"));
        CHECK(r1.warnings.size() == 1 && r1.warnings[0].find("turbo") != std::string::npos && !r1.options.effort);
        const auto r2 = chat::parseChatRequest(json::parse(R"({"messages":[{"role":"user","content":"q"}],"reasoning_effort":"Max"})"));
        CHECK(r2.warnings.empty() && r2.options.effort && *r2.options.effort == "xhigh");
        CHECK(chat::canonicalEffort("minimal") == std::optional<std::string>("low"));
        CHECK(!chat::canonicalEffort("") && !chat::canonicalEffort("hi"));
        // direct renderChat callers still get an error for an unknown value
        chat::ChatOptions bad_opt;
        bad_opt.effort = "turbo";
        const json::Array no_tools;
        CHECK(throws([&] { (void)chat::renderChat(chat::TemplateKind::a, r1.messages, no_tools, bad_opt); }));
    }
    CHECK(throws([&] { render(chat::TemplateKind::b, R"({"messages":[{"role":"robot","content":"q"}]})"); }));
    // bug-toolargbool: tool-call argument values. The product GGUFs (Swift-1.5 MXFP4 = a, Ornith-1.5-35B-A3B MXFP4 = b)
    // render every non-string argument with `args_value | tojson | safe`, so bool / null are JSON (true / false /
    // null), also at the top level, and nested values are json.dumps. Same case as tests/corpus/template_cases.json
    // tool_call_scalar_args.
    {
        const char* req =
            R"({"messages":[{"role":"user","content":"edit the file"},{"role":"assistant","content":"","tool_calls":[)"
            R"({"type":"function","function":{"name":"edit","arguments":{"path":"a.py","replace_all":true,"dry_run":false,)"
            R"("encoding":null,"count":3,"ratio":1.5,"opts":{"force":true,"backup":null,"lines":[false,2,"x"]},)"
            R"("flags":[true,null]}}}]},{"role":"tool","content":"ok"}]})";
        const std::string want =
            "<function=edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=replace_all>\ntrue\n</parameter>\n"
            "<parameter=dry_run>\nfalse\n</parameter>\n<parameter=encoding>\nnull\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n"
            "<parameter=opts>\n{\"force\": true, \"backup\": null, \"lines\": [false, 2, \"x\"]}\n</parameter>\n"
            "<parameter=flags>\n[true, null]\n</parameter>\n</function>";
        for (chat::TemplateKind k : {chat::TemplateKind::a, chat::TemplateKind::b}) {
            const std::string p = render(k, req);
            const bool ok = p.find(want) != std::string::npos;
            if (!ok) std::printf("  tool_call_scalar_args kind %s rendered:\n%s\n", chat::templateName(k), p.c_str());
            CHECK(ok);
        }
        // b_pyargs (original ornith-ai Ornith-1.5 Q4_K_M / 9B, Qwen3-Coder, Qwen3.5 templates): scalars via
        // `args_value | string` (Python str), containers via tojson
        const std::string want_py =
            "<function=edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=replace_all>\nTrue\n</parameter>\n"
            "<parameter=dry_run>\nFalse\n</parameter>\n<parameter=encoding>\nNone\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n"
            "<parameter=opts>\n{\"force\": true, \"backup\": null, \"lines\": [false, 2, \"x\"]}\n</parameter>\n"
            "<parameter=flags>\n[true, null]\n</parameter>\n</function>";
        const std::string p = render(chat::TemplateKind::b_pyargs, req);
        const bool ok = p.find(want_py) != std::string::npos;
        if (!ok) std::printf("  tool_call_scalar_args kind b-pyargs rendered:\n%s\n", p.c_str());
        CHECK(ok);
        // apart from the argument values b and b_pyargs render the same prompt
        const char* req2 = R"({"messages":[{"role":"system","content":"S"},{"role":"user","content":"q"},)"
                           R"({"role":"assistant","content":"<think>r</think>a","tool_calls":[{"function":{"name":"t",)"
                           R"("arguments":{"k":"v","n":2}}}]},{"role":"tool","content":"x"},{"role":"user","content":"q2"}]})";
        CHECK_EQ(render(chat::TemplateKind::b, req2), render(chat::TemplateKind::b_pyargs, req2));
    }
    // bug-emptysystem: an empty / whitespace / null leading system message. The product Ornith MXFP4 template (b)
    // renders `messages[0].role == 'system'` without checking the content, so it emits an empty system turn (as
    // the Qwen3.5 / Qwen3 templates do); the Swift template (a, `if content`) and the original Ornith-1.5 template
    // (b_pyargs, `if merged_system`) emit none. With tools all three append the system text only when non-empty.
    // Same cases as tests/corpus/template_cases.json empty_system / null_system / empty_string_system /
    // empty_system_parts / empty_system_tools / empty_system_no_think.
    {
        const std::string user_only = "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n";
        for (const char* sys : {R"("   ")", R"("")", "null", R"([{"type":"text","text":" "}])"}) {
            const std::string req =
                std::string(R"({"messages":[{"role":"system","content":)") + sys + R"(},{"role":"user","content":"x"}]})";
            const std::string pb = render(chat::TemplateKind::b, req.c_str());
            const bool ok = pb == "<|im_start|>system\n<|im_end|>\n" + user_only;
            if (!ok) std::printf("  empty system %s kind b rendered:\n%s\n", sys, pb.c_str());
            CHECK(ok);
            CHECK_EQ(render(chat::TemplateKind::b_pyargs, req.c_str()), user_only);
            const std::string ra = std::string(R"({"reasoning_effort":"medium",)") + req.substr(1);
            CHECK_EQ(render(chat::TemplateKind::a, ra.c_str()), user_only);
        }
        // with tools the empty system text adds nothing (all variants)
        const char* tools_empty = R"({"tools":[{"type":"function","function":{"name":"f"}}],"messages":[)"
                                  R"({"role":"system","content":""},{"role":"user","content":"go"}]})";
        const char* tools_none = R"({"tools":[{"type":"function","function":{"name":"f"}}],"messages":[)"
                                 R"({"role":"user","content":"go"}]})";
        for (chat::TemplateKind k : {chat::TemplateKind::a, chat::TemplateKind::b, chat::TemplateKind::b_pyargs})
            CHECK_EQ(render(k, tools_empty), render(k, tools_none));
        // a non-empty system message renders the same as before
        CHECK_EQ(render(chat::TemplateKind::b, R"({"messages":[{"role":"system","content":" S "},{"role":"user","content":"x"}]})"),
                 "<|im_start|>system\nS<|im_end|>\n" + user_only);
    }
    // the two Ornith-1.5 argument lines as they appear in the GGUFs
    CHECK(chat::detectTemplate("{%- set args_value = args_value | string if args_value is string else args_value | "
                               "tojson | safe %}") == chat::TemplateKind::b);
    CHECK(chat::detectTemplate("{%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value "
                               "is sequence and args_value is not string) else args_value | string %}") ==
          chat::TemplateKind::b_pyargs);
    CHECK(std::string(chat::templateName(chat::TemplateKind::b_pyargs)) == "b-pyargs");
    CHECK(chat::detectTemplate("{% if reasoning_effort %}") == chat::TemplateKind::a);
    CHECK(chat::detectTemplate("{% if x %}") == chat::TemplateKind::b);
}

// arch025-N6 (lane 1 leg 7): the JSON request field helpers chat_template.cpp (json::Value, Value::get) and protocol.cpp
// (json::Object, Object::find) each wrote out -> one copy, whirl/json_fields.h. References: verbatim copies of both
// former versions (f7e11d7 src/chat/chat_template.cpp:15-53, src/server/protocol.cpp:18-50 / :74-85). Pointers are compared
// for identity (the same member of the same document), errors by their message.
namespace oldjf {
constexpr std::string_view k_ws = " \t\r\n\x0b\x0c";
std::string_view c_trimWs(std::string_view s) {
    const std::size_t b = s.find_first_not_of(k_ws);
    if (b == std::string_view::npos) return {};
    const std::size_t e = s.find_last_not_of(k_ws);
    return s.substr(b, e - b + 1);
}
[[noreturn]] void bad(const std::string& msg) { throw chat::RequestError(msg); }
const json::Value* c_getField(const json::Value& obj, std::string_view key) {
    const json::Value* v = obj.get(key);
    return (v && !v->isNull()) ? v : nullptr;
}
const std::string* c_getStr(const json::Value& obj, std::string_view key) {
    const json::Value* v = c_getField(obj, key);
    return (v && v->isString()) ? &v->asString() : nullptr;
}
std::optional<bool> c_getBool(const json::Value& obj, std::string_view key, const char* err) {
    const json::Value* v = c_getField(obj, key);
    if (!v) return std::nullopt;
    if (!v->isBool()) bad(err);
    return v->asBool();
}
const json::Value* p_getField(const json::Object& obj, std::string_view key) {
    const json::Value* v = obj.find(key);
    if (!v || v->isNull()) return nullptr;
    return v;
}
std::optional<double> p_getNum(const json::Object& obj, std::string_view key, const char* msg) {
    const json::Value* v = p_getField(obj, key);
    if (!v) return std::nullopt;
    switch (v->kind()) {
        case json::Value::Kind::integer: return static_cast<double>(v->asInt());
        case json::Value::Kind::number: return v->asDouble();
        case json::Value::Kind::big_integer: return v->asDouble();
        default: throw chat::RequestError(msg);
    }
}
std::optional<bool> p_getBool(const json::Object& obj, std::string_view key, const char* msg) {
    const json::Value* v = p_getField(obj, key);
    if (!v) return std::nullopt;
    if (!v->isBool()) throw chat::RequestError(msg);
    return v->asBool();
}
const std::string* p_getStr(const json::Object& obj, std::string_view key) {
    const json::Value* v = p_getField(obj, key);
    if (!v || !v->isString()) return nullptr;
    return &v->asString();
}
std::string_view p_trimLeftWs(std::string_view s) {
    const std::size_t a = s.find_first_not_of(k_ws);
    return a == std::string_view::npos ? std::string_view() : s.substr(a);
}
std::string_view p_trimRightWs(std::string_view s) {
    const std::size_t b = s.find_last_not_of(k_ws);
    return b == std::string_view::npos ? std::string_view() : s.substr(0, b + 1);
}
std::string_view p_trimWs(std::string_view s) { return p_trimRightWs(p_trimLeftWs(s)); }

// "v:<value>" / "e:<message>" / "none"
template <class F>
std::string outcome(F f) {
    try {
        const auto r = f();
        if (!r) return "none";
        return "v:" + std::format("{}", *r);
    } catch (const chat::RequestError& e) {
        return std::string("e:") + e.what();
    }
}
bool sameView(std::string_view a, std::string_view b) { return a.data() == b.data() && a.size() == b.size(); }
}  // namespace oldjf

void testJsonFields() {
    const char* const docs[] = {
        R"({"a":1,"b":null,"c":"s","d":true,"e":2.5,"f":[1],"g":{"x":1},"h":123456789012345678901234567890,"i":false,"j":"","c":"dup"})",
        R"({})", R"({"":"empty key","a":null})", R"([1,2])", R"("str")", R"(null)", R"(5)", R"(true)"};
    const char* const keys[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "zz", "", "x"};
    std::size_t n = 0, objs = 0;
    for (const char* d : docs) {
        const json::Value doc = json::parse(d);
        for (const char* k : keys) {
            // json::Value: == chat_template's former code on every value (an object or not)
            CHECK(chat::getField(doc, k) == oldjf::c_getField(doc, k));
            CHECK(chat::getStr(doc, k) == oldjf::c_getStr(doc, k));
            CHECK_EQ(oldjf::outcome([&] { return chat::getBool(doc, k, "bad bool"); }),
                     oldjf::outcome([&] { return oldjf::c_getBool(doc, k, "bad bool"); }));
            ++n;
            if (!doc.isObject()) continue;
            // json::Object: == protocol's former code; and the Value overload == the Object one on an object
            const json::Object& o = doc.asObject();
            CHECK(chat::getField(o, k) == oldjf::p_getField(o, k));
            CHECK(chat::getField(o, k) == chat::getField(doc, k));
            CHECK(chat::getStr(o, k) == oldjf::p_getStr(o, k));
            CHECK_EQ(oldjf::outcome([&] { return chat::getBool(o, k, "bad bool"); }),
                     oldjf::outcome([&] { return oldjf::p_getBool(o, k, "bad bool"); }));
            CHECK_EQ(oldjf::outcome([&] { return chat::getNum(o, k, "bad num"); }),
                     oldjf::outcome([&] { return oldjf::p_getNum(o, k, "bad num"); }));
            ++objs;
        }
    }
    const std::string texts[] = {"",        " ",      " \t\r\n\x0b\x0c", "a",       " a ",  "\ta b\n",    "x\x0b",
                                 "\x0c" "y", "  in  ner  ", "\xc2\xa0z\xc2\xa0", "\r\n\r\n", "no-ws", std::string("a\0b", 3)};
    std::size_t nt = 0;
    for (const std::string& t : texts) {
        const std::string_view s(t);
        CHECK(oldjf::sameView(chat::trimWs(s), oldjf::c_trimWs(s)));
        CHECK(oldjf::sameView(chat::trimWs(s), oldjf::p_trimWs(s)));
        CHECK(oldjf::sameView(chat::trimLeftWs(s), oldjf::p_trimLeftWs(s)));
        CHECK(oldjf::sameView(chat::trimRightWs(s), oldjf::p_trimRightWs(s)));
        ++nt;
    }
    // the image placeholder: one copy (chat.h) for the template and `whirl chat --image` (the CLI's former literal)
    CHECK(chat::k_image_placeholder == std::string_view("<|vision_start|><|image_pad|><|vision_end|>"));
    std::printf("json fields: %zu value cases, %zu object cases, %zu trim cases vs the chat_template / protocol copies\n", n, objs, nt);
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_chat() {
    static const whirl::test::Suite s{"chat",
                                      {
                                          {"testChat", [] { testChat(); }},
                                          {"testJsonFields", [] { testJsonFields(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_chat()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
