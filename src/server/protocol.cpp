// Reimplements the WHIRL Zig research prototype's server.zig (helpers, parseParams, Out.view,
// parseToolCalls, Timings, writeUsage, HTTP response text).
// SPDX-License-Identifier: Apache-2.0

#include "protocol.h"

#include "whirl/common.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>

namespace whirl::server {

namespace {

constexpr std::string_view kWs = " \t\r\n\x0b\x0c";

const json::Value* getField(const json::Object& obj, std::string_view key) {
    const json::Value* v = obj.find(key);
    if (!v || v->isNull()) return nullptr;
    return v;
}

// number field: nullopt when absent / null; throws RequestError(msg) when not a number
std::optional<double> getNum(const json::Object& obj, std::string_view key, const char* msg) {
    const json::Value* v = getField(obj, key);
    if (!v) return std::nullopt;
    switch (v->kind()) {
        case json::Value::Kind::integer: return static_cast<double>(v->asInt());
        case json::Value::Kind::number: return v->asDouble();
        case json::Value::Kind::big_integer: return v->asDouble();
        default: throw RequestError(msg);
    }
}

std::optional<bool> getBool(const json::Object& obj, std::string_view key, const char* msg) {
    const json::Value* v = getField(obj, key);
    if (!v) return std::nullopt;
    if (!v->isBool()) throw RequestError(msg);
    return v->asBool();
}

const std::string* getStr(const json::Object& obj, std::string_view key) {
    const json::Value* v = getField(obj, key);
    if (!v || !v->isString()) return nullptr;
    return &v->asString();
}

bool startsWith(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool endsWith(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

std::optional<std::string_view> toolParamType(const json::Array& tools, std::string_view fname, std::string_view pname) {
    for (const json::Value& t : tools) {
        if (!t.isObject()) continue;
        const json::Value* f = getField(t.asObject(), "function");
        if (!f || !f->isObject()) continue;
        const std::string* n = getStr(f->asObject(), "name");
        if (!n || *n != fname) continue;
        const json::Value* ps = getField(f->asObject(), "parameters");
        if (!ps || !ps->isObject()) return std::nullopt;
        const json::Value* props = getField(ps->asObject(), "properties");
        if (!props || !props->isObject()) return std::nullopt;
        const json::Value* pv = getField(props->asObject(), pname);
        if (!pv || !pv->isObject()) return std::nullopt;
        const std::string* ty = getStr(pv->asObject(), "type");
        if (!ty) return std::nullopt;
        return std::string_view(*ty);
    }
    return std::nullopt;
}

}  // namespace

std::string_view trimWs(std::string_view s) { return trimRightWs(trimLeftWs(s)); }

std::string_view trimLeftWs(std::string_view s) {
    const std::size_t a = s.find_first_not_of(kWs);
    return a == std::string_view::npos ? std::string_view() : s.substr(a);
}

std::string_view trimRightWs(std::string_view s) {
    const std::size_t b = s.find_last_not_of(kWs);
    return b == std::string_view::npos ? std::string_view() : s.substr(0, b + 1);
}

std::size_t partialHold(std::string_view text, std::string_view pat) {
    if (pat.empty()) return 0;
    std::size_t k = std::min(pat.size() - 1, text.size());
    for (; k > 0; --k)
        if (endsWith(text, pat.substr(0, k))) return k;
    return 0;
}

std::size_t validUtf8Prefix(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t n;
        if (c < 0x80) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        else {
            ++i;  // stray byte: passes through
            continue;
        }
        if (i + n > s.size()) return i;
        i += n;
    }
    return i;
}

std::string sanitizeUtf8(std::string_view s) {
    if (isValidUtf8(s)) return std::string(s);
    std::string out;
    out.reserve(s.size() + 8);
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t n;
        if (c < 0x80) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        else {
            out += "\xEF\xBF\xBD";
            ++i;
            continue;
        }
        if (i + n <= s.size() && isValidUtf8(s.substr(i, n))) {
            out.append(s.substr(i, n));
            i += n;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}

void appendJsonStr(std::string& out, std::string_view s) {
    out.push_back('"');
    json::appendEscaped(out, s);
    out.push_back('"');
}

void appendFloat(std::string& out, double v) {
    if (std::isnan(v) || std::isinf(v)) {
        out += "null";
        return;
    }
    char buf[400];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::fixed);
    out.append(buf, r.ptr);
}

// ---------------------------------------------------------------------------

Params parseParams(const json::Object& obj, bool think, bool chat) {
    Params p;
    if (chat && !think) {
        // model-card defaults for non-thinking mode
        p.temperature = 0.7f;
        p.top_p = 0.8f;
    }
    if (auto v = getNum(obj, "temperature", "temperature must be a number")) {
        if (*v < 0 || *v > 100) throw RequestError("temperature must be >= 0");
        p.temperature = static_cast<float>(*v);
    }
    if (auto v = getNum(obj, "top_p", "top_p must be a number")) {
        if (*v <= 0 || *v > 1) throw RequestError("top_p must be in (0, 1]");
        p.top_p = static_cast<float>(*v);
    }
    if (auto v = getNum(obj, "top_k", "top_k must be a number")) {
        if (*v < 0) throw RequestError("top_k must be >= 0");
        p.top_k = static_cast<std::uint32_t>(std::min(*v, 1e9));
    }
    if (auto v = getNum(obj, "min_p", "min_p must be a number")) {
        if (*v < 0 || *v > 1) throw RequestError("min_p must be in [0, 1]");
        p.min_p = static_cast<float>(*v);
    }
    if (auto v = getNum(obj, "seed", "seed must be an integer")) {
        const double c = std::max(std::min(*v, 9.2e18), -9.2e18);
        p.seed = static_cast<std::uint64_t>(static_cast<std::int64_t>(c));
        p.seed_given = true;
    }
    std::optional<double> mt = getNum(obj, "max_completion_tokens", "max_completion_tokens must be an integer");
    if (!mt) mt = getNum(obj, "max_tokens", "max_tokens must be an integer");
    if (mt) {
        if (*mt < 1) throw RequestError("max_tokens must be >= 1");
        p.max_tokens = static_cast<std::uint32_t>(std::min(*mt, 4e9));
    }
    if (auto v = getNum(obj, "presence_penalty", "presence_penalty must be a number"))
        p.presence_penalty = static_cast<float>(*v);
    if (auto v = getNum(obj, "frequency_penalty", "frequency_penalty must be a number"))
        p.frequency_penalty = static_cast<float>(*v);
    if (auto v = getNum(obj, "n", "n must be an integer")) {
        if (*v != 1) throw RequestError("only n = 1 is supported");
    }
    if (const json::Value* sv = getField(obj, "stop")) {
        if (sv->isString()) {
            if (!sv->asString().empty()) p.stop.push_back(sv->asString());
        } else if (sv->isArray()) {
            for (const json::Value& it : sv->asArray()) {
                if (!it.isString()) throw RequestError("stop must be a string or a list of strings");
                if (!it.asString().empty()) p.stop.push_back(it.asString());
            }
        } else {
            throw RequestError("stop must be a string or a list of strings");
        }
    }
    p.stream = getBool(obj, "stream", "stream must be a boolean").value_or(false);
    p.ignore_eos = getBool(obj, "ignore_eos", "ignore_eos must be a boolean").value_or(false);
    p.cache_prompt = getBool(obj, "cache_prompt", "cache_prompt must be a boolean").value_or(true);
    if (const json::Value* so = getField(obj, "stream_options")) {
        if (so->isObject()) {
            try {
                p.include_usage = getBool(so->asObject(), "include_usage", "").value_or(false);
            } catch (const RequestError&) {
                p.include_usage = false;
            }
        }
    }
    return p;
}

// ---------------------------------------------------------------------------

Out::View Out::view(bool final_in) const {
    std::string_view t(text.data(), valid);
    bool final = final_in;
    View r;
    if (!stops.empty()) {
        std::optional<std::size_t> best;
        for (const std::string& s : stops) {
            const std::size_t i = t.find(s);
            if (i != std::string_view::npos && (!best || i < *best)) best = i;
        }
        if (best) {
            t = t.substr(0, *best);
            final = true;
            r.stopped = true;
        } else if (!final) {
            std::size_t hold = 0;
            for (const std::string& s : stops) hold = std::max(hold, partialHold(t, s));
            t = t.substr(0, t.size() - hold);
        }
    }
    if (!chat) {
        r.content = t;
        return r;
    }
    bool in_reason = think_open;
    std::optional<std::string_view> rest = t;
    if (!in_reason) {
        const std::string_view lt = trimLeftWs(t);
        if (startsWith(lt, "<think>")) {
            in_reason = true;
            t = lt.substr(7);
        } else if (!final && !lt.empty() && lt.size() < 7 && startsWith("<think>", lt)) {
            return r;  // maybe the start of "<think>": hold everything
        }
    }
    if (in_reason) {
        const std::size_t i = t.find("</think>");
        if (i != std::string_view::npos) {
            r.reason = trimWs(t.substr(0, i));
            rest = t.substr(i + 8);
        } else {
            const std::size_t hold = final ? 0 : partialHold(t, "</think>");
            r.reason = trimWs(t.substr(0, t.size() - hold));
            rest = std::nullopt;
        }
    }
    if (rest) {
        std::string_view c = trimLeftWs(*rest);
        if (tools_on) {
            const std::size_t i = c.find("<tool_call>");
            if (i != std::string_view::npos) {
                r.tool = c.substr(i);
                c = c.substr(0, i);
            } else if (!final) {
                c = c.substr(0, c.size() - partialHold(c, "<tool_call>"));
            }
        }
        r.content = trimRightWs(c);
    }
    return r;
}

std::optional<std::vector<ToolCall>> parseToolCalls(std::string_view region, const json::Array& tools) {
    std::vector<ToolCall> calls;
    std::string_view rest = region;
    for (;;) {
        const std::size_t s = rest.find("<tool_call>");
        if (s == std::string_view::npos) break;
        const std::string_view after = rest.substr(s + 11);
        const std::size_t e = after.find("</tool_call>");
        const std::string_view inner = trimWs(e != std::string_view::npos ? after.substr(0, e) : after);
        rest = e != std::string_view::npos ? after.substr(e + 12) : std::string_view();
        if (startsWith(inner, "<function=")) {
            const std::size_t gt = inner.find('>');
            if (gt == std::string_view::npos) return std::nullopt;
            const std::string_view name = trimWs(inner.substr(10, gt - 10));
            if (name.empty()) return std::nullopt;
            std::string_view body = inner.substr(gt + 1);
            if (const std::size_t k = body.find("</function>"); k != std::string_view::npos) body = body.substr(0, k);
            json::Object obj;
            for (;;) {
                const std::size_t ps = body.find("<parameter=");
                if (ps == std::string_view::npos) break;
                const std::string_view b2 = body.substr(ps + 11);
                const std::size_t pg = b2.find('>');
                if (pg == std::string_view::npos) return std::nullopt;
                const std::string_view pname = trimWs(b2.substr(0, pg));
                std::string_view vtext = b2.substr(pg + 1);
                const std::size_t pe = vtext.find("</parameter>");
                body = pe != std::string_view::npos ? vtext.substr(pe + 12) : std::string_view();
                if (pe != std::string_view::npos) vtext = vtext.substr(0, pe);
                if (startsWith(vtext, "\n")) vtext.remove_prefix(1);
                if (endsWith(vtext, "\n")) vtext.remove_suffix(1);
                const auto ty = toolParamType(tools, name, pname);
                json::Value val{std::string(vtext)};
                if (!ty || *ty != "string") {
                    try {
                        val = json::parse(vtext);
                    } catch (const std::exception&) {
                    }
                }
                obj.set(std::string(pname), std::move(val));
            }
            calls.push_back({std::string(name), json::dumpPython(json::Value(std::move(obj)), true)});
        } else if (startsWith(inner, "{")) {
            json::Value v;
            try {
                v = json::parse(inner);
            } catch (const std::exception&) {
                return std::nullopt;
            }
            if (!v.isObject()) return std::nullopt;
            const std::string* name = getStr(v.asObject(), "name");
            if (!name) return std::nullopt;
            std::string args = "{}";
            if (const json::Value* av = getField(v.asObject(), "arguments"))
                args = av->isString() ? av->asString() : json::dumpPython(*av, true);
            calls.push_back({*name, std::move(args)});
        } else {
            return std::nullopt;
        }
    }
    if (calls.empty()) return std::nullopt;
    return calls;
}

// ---------------------------------------------------------------------------

void Timings::write(std::string& out) const {
    const double pn = prompt_n, gn = predicted_n;
    out += std::format("{{\"cache_n\":{},\"prompt_n\":{},\"prompt_ms\":", cache_n, prompt_n);
    appendFloat(out, prompt_ms);
    out += ",\"prompt_per_token_ms\":";
    appendFloat(out, prompt_n > 0 ? prompt_ms / pn : 0);
    out += ",\"prompt_per_second\":";
    appendFloat(out, prompt_ms > 0 ? pn * 1000.0 / prompt_ms : 0);
    out += std::format(",\"predicted_n\":{},\"predicted_ms\":", predicted_n);
    appendFloat(out, predicted_ms);
    out += ",\"predicted_per_token_ms\":";
    appendFloat(out, predicted_n > 0 ? predicted_ms / gn : 0);
    out += ",\"predicted_per_second\":";
    appendFloat(out, predicted_ms > 0 ? gn * 1000.0 / predicted_ms : 0);
    out += std::format(",\"draft_n\":{},\"draft_n_accepted\":{}}}", draft_n, draft_n_accepted);
}

void writeUsage(std::string& out, std::uint32_t n_prompt, const Timings& t) {
    out += std::format(
        "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{},\"prompt_tokens_details\":{{\"cached_tokens\":{}}}}}",
        n_prompt, t.predicted_n, n_prompt + t.predicted_n, t.cache_n);
}

const char* statusText(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Error";
    }
}

std::string httpResponse(int status, std::string_view ctype, std::string_view body) {
    std::string r = std::format(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: "
        "close\r\n\r\n",
        status, statusText(status), ctype, body.size());
    r += body;
    return r;
}

std::string errorResponse(int status, std::string_view msg) {
    std::string b = "{\"error\":{\"message\":";
    std::string_view m = msg.substr(0, std::min<std::size_t>(msg.size(), 3000));
    // keep the cut on a UTF-8 boundary
    while (!m.empty() && m.size() < msg.size() && (static_cast<unsigned char>(msg[m.size()]) & 0xC0) == 0x80)
        m.remove_suffix(1);
    appendJsonStr(b, m);
    b += std::format(",\"type\":\"{}\",\"code\":{}}}}}",
                     status >= 500 ? "server_error" : (status == 404 ? "not_found_error" : "invalid_request_error"), status);
    return httpResponse(status, "application/json; charset=utf-8", b);
}

std::string randomId(std::string_view prefix, std::size_t n, std::uint64_t (*next)(void*), void* ctx) {
    static constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string s(prefix);
    for (std::size_t i = 0; i < n; ++i) s.push_back(alphabet[next(ctx) % 62]);
    return s;
}

}  // namespace whirl::server
