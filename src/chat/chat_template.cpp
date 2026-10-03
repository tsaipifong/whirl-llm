// Chat template rendering (see include/whirl/chat.h).
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the project's own prototype (server.zig: renderChat,
// contentText, writeArgValue and the request option handling of buildJob).

#include "whirl/chat.h"

#include <string_view>

namespace whirl::chat {

namespace {

constexpr std::string_view k_ws = " \t\r\n\x0b\x0c";

std::string_view trimWs(std::string_view s) {
    const std::size_t b = s.find_first_not_of(k_ws);
    if (b == std::string_view::npos) return {};
    const std::size_t e = s.find_last_not_of(k_ws);
    return s.substr(b, e - b + 1);
}

std::string_view trimLeftChars(std::string_view s, std::string_view chars) {
    const std::size_t b = s.find_first_not_of(chars);
    return b == std::string_view::npos ? std::string_view{} : s.substr(b);
}

std::string_view trimRightChars(std::string_view s, std::string_view chars) {
    const std::size_t e = s.find_last_not_of(chars);
    return e == std::string_view::npos ? std::string_view{} : s.substr(0, e + 1);
}

bool startsWith(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool endsWith(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.substr(s.size() - p.size()) == p; }

[[noreturn]] void bad(const std::string& msg) { throw RequestError(msg); }

// member that is present and not JSON null
const json::Value* getField(const json::Value& obj, std::string_view key) {
    const json::Value* v = obj.get(key);
    return (v && !v->isNull()) ? v : nullptr;
}

const std::string* getStr(const json::Value& obj, std::string_view key) {
    const json::Value* v = getField(obj, key);
    return (v && v->isString()) ? &v->asString() : nullptr;
}

std::optional<bool> getBool(const json::Value& obj, std::string_view key, const char* err) {
    const json::Value* v = getField(obj, key);
    if (!v) return std::nullopt;
    if (!v->isBool()) bad(err);
    return v->asBool();
}

const std::string& roleOf(const json::Value& m) {
    if (!m.isObject()) bad("each message must be an object");
    const std::string* r = getStr(m, "role");
    if (!r) bad("message.role is required");
    return *r;
}

bool isSystemRole(std::string_view r) { return r == "system" || r == "developer"; }

constexpr std::string_view k_image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>";
constexpr std::string_view k_video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>";

// render_content(): string, list of parts, or null/absent.
std::string contentText(const json::Value& m, const ChatOptions& opt, bool is_system) {
    const json::Value* v = getField(m, "content");
    if (!v) return {};
    if (v->isString()) return v->asString();
    if (!v->isArray()) bad("message.content must be a string, a list of parts, or null");
    std::string out;
    for (const json::Value& it : v->asArray()) {
        if (!it.isObject()) bad("content parts must be objects");
        if (opt.allow_media) {
            // Jinja order: image, video, text
            const std::string* type = getStr(it, "type");
            const bool image = it.get("image") || it.get("image_url") || (type && *type == "image");
            const bool video = !image && (it.get("video") || (type && *type == "video"));
            if (image || video) {
                if (is_system) bad(image ? "System message cannot contain images." : "System message cannot contain videos.");
                out += image ? k_image_placeholder : k_video_placeholder;
            } else if (const std::string* t = getStr(it, "text")) {
                out += *t;
            } else {
                bad("unexpected content part");
            }
            continue;
        }
        if (const std::string* t = getStr(it, "text")) {
            out += *t;
        } else if (it.get("image_url") || it.get("image") || it.get("video")) {
            bad("image / video content is not supported by this server");
        } else {
            bad("unexpected content part");
        }
    }
    return out;
}

void writeArgValue(std::string& w, const json::Value& v, TemplateKind kind) {
    if (v.isString()) {
        w += v.asString();
        return;
    }
    if (kind == TemplateKind::b) {
        if (v.isBool()) {
            w += v.asBool() ? "True" : "False";
            return;
        }
        if (v.isNull()) {
            w += "None";
            return;
        }
    }
    json::dumpPython(w, v);
}

constexpr std::string_view k_tools_preamble = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view k_tools_instructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";
constexpr std::string_view k_effort_xhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
constexpr std::string_view k_effort_low =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
    "without unnecessary elaboration.";

}  // namespace

TemplateKind detectTemplate(std::string_view chat_template) {
    return chat_template.find("reasoning_effort") != std::string_view::npos ? TemplateKind::a : TemplateKind::b;
}

const char* templateName(TemplateKind k) { return k == TemplateKind::a ? "a" : "b"; }

std::string renderChat(TemplateKind kind, const json::Array& msgs, const json::Array& tools, const ChatOptions& opt) {
    if (msgs.empty()) bad("messages must be a non-empty array");
    std::string w;

    // leading system / developer messages
    std::size_t num_sys = 0;
    std::string merged;
    if (kind == TemplateKind::a) {
        for (const json::Value& m : msgs) {
            if (!isSystemRole(roleOf(m))) break;
            const std::string text = contentText(m, opt, true);
            const std::string_view t = trimWs(text);
            if (!t.empty()) {
                if (!merged.empty()) merged.push_back('\n');
                merged += t;
            }
            ++num_sys;
        }
    } else if (isSystemRole(roleOf(msgs[0]))) {
        merged = trimWs(contentText(msgs[0], opt, true));
        num_sys = 1;
        if (msgs.size() > 1 && isSystemRole(roleOf(msgs[1]))) {
            merged.push_back('\n');
            merged += trimWs(contentText(msgs[1], opt, true));
            num_sys = 2;
        }
    }

    std::string_view instr;
    if (kind == TemplateKind::a && opt.think) {
        std::string eff = opt.effort.value_or("xhigh");
        if (eff == "high") eff = "xhigh";
        if (eff == "xhigh")
            instr = k_effort_xhigh;
        else if (eff == "low")
            instr = k_effort_low;
        else if (eff != "medium")
            bad("unexpected reasoning_effort (supported: xhigh / high, medium, low)");
    }

    if (!tools.empty()) {
        w += "<|im_start|>system\n";
        if (!instr.empty()) {
            w += instr;
            w += "\n\n";
        }
        w += k_tools_preamble;
        for (const json::Value& t : tools) {
            w.push_back('\n');
            json::dumpPython(w, t);
        }
        w += "\n</tools>";
        w += k_tools_instructions;
        if (!merged.empty()) {
            w += "\n\n";
            w += merged;
        }
        w += "<|im_end|>\n";
    } else if (!merged.empty()) {
        w += "<|im_start|>system\n";
        if (!instr.empty()) {
            w += instr;
            w += "\n\n";
        }
        w += merged;
        w += "<|im_end|>\n";
    } else if (!instr.empty()) {
        w += "<|im_start|>system\n";
        w += instr;
        w += "<|im_end|>\n";
    }

    // index of the last real user query (tool responses wrapped in a user
    // message do not count)
    bool multi_step = true;
    std::size_t last_query = msgs.size() - 1;
    for (std::size_t i = msgs.size(); i-- > 0;) {
        const std::string& r = roleOf(msgs[i]);
        if (multi_step && r == "user") {
            const std::string text = contentText(msgs[i], opt, false);
            const std::string_view c = trimWs(text);
            if (!(startsWith(c, "<tool_response>") && endsWith(c, "</tool_response>"))) {
                multi_step = false;
                last_query = i;
            }
        }
    }

    for (std::size_t i = 0; i < msgs.size(); ++i) {
        if (i < num_sys) continue;
        const json::Value& m = msgs[i];
        const std::string& role = roleOf(m);
        const std::string text = contentText(m, opt, false);
        std::string_view c = trimWs(text);
        if (isSystemRole(role)) {
            // template a raises here and b skips; a keeps the text instead of failing
            if (kind == TemplateKind::b) continue;
            w += "<|im_start|>system\n";
            w += c;
            w += "<|im_end|>\n";
        } else if (role == "user") {
            w += "<|im_start|>user\n";
            w += c;
            w += "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string_view reasoning;
            if (const std::string* rc = getStr(m, "reasoning_content")) {
                reasoning = *rc;
            } else if (kind == TemplateKind::b) {
                const std::size_t close = c.find("</think>");
                if (close != std::string_view::npos) {
                    const std::string_view before = trimRightChars(c.substr(0, close), "\n");
                    const std::size_t open = before.rfind("<think>");
                    reasoning = trimLeftChars(open != std::string_view::npos ? before.substr(open + 7) : before, "\n");
                    const std::size_t last_close = c.rfind("</think>");
                    c = trimLeftChars(c.substr(last_close + 8), "\n");
                }
            }
            reasoning = trimWs(reasoning);
            const bool keep = kind == TemplateKind::b || !opt.preserve_thinking.has_value() ||
                              *opt.preserve_thinking || i > last_query;
            w += "<|im_start|>assistant\n";
            if (keep) {
                w += "<think>\n";
                w += reasoning;
                w += "\n</think>\n\n";
            }
            w += c;
            if (const json::Value* tcv = getField(m, "tool_calls")) {
                if (!tcv->isArray()) bad("assistant.tool_calls must be an array");
                std::size_t j = 0;
                for (const json::Value& tc0 : tcv->asArray()) {
                    if (!tc0.isObject()) bad("tool_calls entries must be objects");
                    const json::Value* tc = &tc0;
                    if (const json::Value* fn = getField(tc0, "function")) {
                        if (!fn->isObject()) bad("tool_call.function must be an object");
                        tc = fn;
                    }
                    const std::string* name = getStr(*tc, "name");
                    if (!name) bad("Tool call is missing a function name.");
                    if (j == 0)
                        w += !trimWs(c).empty() ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=";
                    else
                        w += "\n<tool_call>\n<function=";
                    w += *name;
                    w += ">\n";
                    json::Value parsed;
                    const json::Object* args = nullptr;
                    if (const json::Value* av = getField(*tc, "arguments")) {
                        if (av->isObject()) {
                            args = &av->asObject();
                        } else if (av->isString()) {
                            if (!trimWs(av->asString()).empty()) {
                                try {
                                    parsed = json::parse(av->asString());
                                } catch (const json::ParseError&) {
                                    bad("tool_call.function.arguments is not valid JSON");
                                }
                                if (!parsed.isObject()) bad("tool_call.function.arguments must be a JSON object");
                                args = &parsed.asObject();
                            }
                        } else {
                            bad("tool_call.function.arguments must be an object or a JSON string");
                        }
                    }
                    if (args) {
                        for (const json::Member& kv : args->members()) {
                            w += "<parameter=";
                            w += kv.first;
                            w += ">\n";
                            writeArgValue(w, kv.second, kind);
                            w += "\n</parameter>\n";
                        }
                    }
                    w += "</function>\n</tool_call>";
                    ++j;
                }
            }
            w += "<|im_end|>\n";
        } else if (role == "tool") {
            if (i > 0 && roleOf(msgs[i - 1]) != "tool") w += "<|im_start|>user";
            w += "\n<tool_response>\n";
            w += c;
            w += "\n</tool_response>";
            if (i + 1 == msgs.size() || roleOf(msgs[i + 1]) != "tool") w += "<|im_end|>\n";
        } else {
            bad("Unexpected message role.");
        }
    }
    w += "<|im_start|>assistant\n";
    w += opt.think ? "<think>\n" : "<think>\n\n</think>\n\n";
    return w;
}

ChatRequest parseChatRequest(const json::Value& root) {
    if (!root.isObject()) bad("request body must be a JSON object");
    ChatRequest req;
    const json::Value* mv = getField(root, "messages");
    if (!mv) bad("'messages' is required");
    if (!mv->isArray()) bad("'messages' must be an array");
    req.messages = mv->asArray();

    if (const json::Value* kw = getField(root, "chat_template_kwargs")) {
        if (!kw->isObject()) bad("chat_template_kwargs must be an object");
        if (auto b = getBool(*kw, "enable_thinking", "enable_thinking must be a boolean")) req.options.think = *b;
        if (const std::string* s = getStr(*kw, "reasoning_effort")) req.options.effort = *s;
        if (auto b = getBool(*kw, "preserve_thinking", "preserve_thinking must be a boolean"))
            req.options.preserve_thinking = *b;
    }
    if (auto b = getBool(root, "enable_thinking", "enable_thinking must be a boolean")) req.options.think = *b;
    if (const std::string* s = getStr(root, "reasoning_effort")) req.options.effort = *s;

    if (const json::Value* tv = getField(root, "tools")) {
        if (!tv->isArray()) bad("'tools' must be an array");
        for (const json::Value& t : tv->asArray())
            if (!t.isObject()) bad("each tool must be an object");
        req.tools = tv->asArray();
    }
    if (const json::Value* tc = getField(root, "tool_choice")) {
        if (tc->isString())
            req.tool_choice = tc->asString();
        else if (tc->isObject())
            req.tool_choice = "function";
        else
            bad("tool_choice must be a string or an object");
    }
    if (req.tool_choice == "none") req.tools.clear();
    return req;
}

}  // namespace whirl::chat
