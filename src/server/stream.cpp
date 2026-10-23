// Reimplements the WHIRL Zig research prototype's server.zig (StreamState: the response writer); moved out of
// engine.cpp in arch024-S8 (M6). The engine is reached only through ResponseCtx.
// SPDX-License-Identifier: Apache-2.0

#include "stream.h"

#include "log.h"

#include <array>
#include <ctime>
#include <format>

namespace whirl::server {

namespace {

std::uint64_t prngNext(void* ctx) { return (*static_cast<std::mt19937_64*>(ctx))(); }

std::int64_t unixNow() { return static_cast<std::int64_t>(std::time(nullptr)); }

}  // namespace

std::string IdGen::id(std::string_view prefix, std::size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    return randomId(prefix, n, prngNext, &prng_);
}

std::uint64_t IdGen::next() {
    std::lock_guard<std::mutex> lk(mu_);
    return prng_();
}

// ---------------------------------------------------------------------------
// response writing (stream: SSE chunks as tokens arrive; else one JSON body)

// FR-SRV P3-8: a "length" that came from the KV pool, not max_tokens: an extra choice field
// (absent otherwise; OpenAI clients ignore unknown fields)
static void appendCause(std::string& b, const Final& f) {
    if (f.cause.empty()) return;
    b += ",\"finish_cause\":";
    appendJsonStr(b, f.cause);
}

void StreamState::send(const std::string& b) {
    if (gone) return;
    if (!job->w || !job->w->write(b)) fail();
}

void StreamState::begin() {
    const char* prefix = job->kind == JobKind::chat ? "chatcmpl-" : "cmpl-";
    id = cx->ids.id(prefix, 24);
    created = unixNow();
    if (!job->params.stream) return;
    std::string b =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\n"
        "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";  // CORS headers and the chunk framing
                                                                    // (HTTP-DROP): added by the connection (http.cpp)
    if (job->kind == JobKind::chat) {
        chunkBegin(b);
        b += "{\"role\":\"assistant\",\"content\":null}";
        chunkEnd(b, nullptr);
    }
    send(b);
    if (!gone && !job->w->flush()) fail();
}

void StreamState::fail() {
    if (!gone) logW("req {} | client connection lost; stopping generation", job->id);
    gone = true;
}

void StreamState::chunkBegin(std::string& b) const {
    b += "data: {\"id\":";
    appendJsonStr(b, id);
    if (job->kind == JobKind::chat) {
        b += std::format(",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":", created);
        appendJsonStr(b, cx->model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"delta\":";
    } else {
        b += std::format(",\"object\":\"text_completion\",\"created\":{},\"model\":", created);
        appendJsonStr(b, cx->model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"text\":";
    }
}

void StreamState::chunkEnd(std::string& b, const char* fin_reason) {
    b += ",\"logprobs\":null,\"finish_reason\":";
    if (fin_reason) appendJsonStr(b, fin_reason);
    else b += "null";
    b += "}]}\n\n";
}

bool StreamState::push(Out& out, std::span<const std::uint32_t> toks, std::uint32_t* n_after) {
    std::array<std::size_t, 64> starts{};  // text offset where each token's piece starts
    for (std::size_t k = 0; k < toks.size(); ++k) {
        if (k < starts.size()) starts[k] = out.text.size();
        out.text += cx->tok.piece(static_cast<TokenId>(toks[k]), true);
    }
    out.valid = sanitizeUtf8Prefix(out.text, out.valid);
    const Out::View v = out.view(false);
    if (v.stopped && n_after) {
        std::uint32_t n = 0;
        for (std::size_t k = 0; k < toks.size() && k < starts.size(); ++k)
            if (starts[k] >= v.stop_end) ++n;
        *n_after = n;
    }
    if (job->params.stream && !gone) {
        sendDelta(out, v);
        if (!gone && !job->w->flush()) fail();
    }
    return v.stopped;
}

void StreamState::sendDelta(Out& out, const Out::View& v) {
    std::string b;
    if (job->kind == JobKind::chat) {
        if (v.reason.size() > out.sent_reason) {
            chunkBegin(b);
            b += "{\"reasoning_content\":";
            appendJsonStr(b, v.reason.substr(out.sent_reason));
            b += "}";
            chunkEnd(b, nullptr);
            out.sent_reason = v.reason.size();
        }
        if (v.content.size() > out.sent_content) {
            chunkBegin(b);
            b += "{\"content\":";
            appendJsonStr(b, v.content.substr(out.sent_content));
            b += "}";
            chunkEnd(b, nullptr);
            out.sent_content = v.content.size();
        }
    } else if (v.content.size() > out.sent_content) {
        chunkBegin(b);
        appendJsonStr(b, v.content.substr(out.sent_content));
        chunkEnd(b, nullptr);
        out.sent_content = v.content.size();
    }
    if (!b.empty()) send(b);
}

Final StreamState::finish(Out& out, const std::string& reason_in) {
    out.text = sanitizeUtf8(out.text);
    out.valid = out.text.size();
    Out::View v = out.view(true);
    Final f;
    ToolResolution tr = resolveTools(v, reason_in, job->tools);
    f.reason = std::move(tr.reason);
    const std::string content = std::move(tr.content);
    if (tr.calls) {
        f.n_tools = tr.calls->size();
        f.calls = std::move(tr.calls);
    }
    f.content = content;
    f.reasoning = std::string(v.reason);
    f.reason_len = v.reason.size();
    f.content_len = content.size();
    if (job->params.stream && !gone) {
        // remaining text (e.g. an unparsable tool block) then the tool calls
        if (content.size() >= out.sent_content) {
            Out::View v2 = v;
            v2.content = f.content;
            sendDelta(out, v2);
        }
        if (f.calls) {
            std::string b;
            chunkBegin(b);
            b += "{\"tool_calls\":[";
            for (std::size_t i = 0; i < f.calls->size(); ++i) {
                const ToolCall& c = (*f.calls)[i];
                const std::string cid = cx->ids.id("call_", 24);
                if (i > 0) b += ",";
                b += std::format("{{\"index\":{},\"id\":", i);
                appendJsonStr(b, cid);
                b += ",\"type\":\"function\",\"function\":{\"name\":";
                appendJsonStr(b, c.name);
                b += ",\"arguments\":";
                appendJsonStr(b, c.args);
                b += "}}";
            }
            b += "]}";
            chunkEnd(b, nullptr);
            send(b);
        }
    }
    return f;
}

void StreamState::end(Out& out, const Final& f, const Timings& tim, std::uint32_t n_prompt) {
    (void)out;
    if (job->params.stream) {
        if (gone) return;
        // final chunk: empty delta + finish_reason (+ usage / timings as llama.cpp)
        std::string b;
        chunkBegin(b);
        b += job->kind == JobKind::chat ? "{}" : "\"\"";
        b += ",\"logprobs\":null,\"finish_reason\":";
        appendJsonStr(b, f.reason);
        appendCause(b, f);
        b += "}],\"usage\":";
        writeUsage(b, n_prompt, tim);
        b += ",\"timings\":";
        tim.write(b);
        b += "}\n\n";
        if (job->params.include_usage) {
            b += "data: {\"id\":";
            appendJsonStr(b, id);
            b += std::format(",\"object\":\"{}\",\"created\":{},\"model\":",
                             job->kind == JobKind::chat ? "chat.completion.chunk" : "text_completion", created);
            appendJsonStr(b, cx->model_name);
            b += ",\"choices\":[],\"usage\":";
            writeUsage(b, n_prompt, tim);
            b += "}\n\n";
        }
        b += "data: [DONE]\n\n";
        send(b);
        if (!gone && !job->w->flush()) fail();
        return;
    }
    std::string b = "{\"id\":";
    appendJsonStr(b, id);
    if (job->kind == JobKind::chat) {
        b += std::format(",\"object\":\"chat.completion\",\"created\":{},\"model\":", created);
        appendJsonStr(b, cx->model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":";
        if (f.calls && f.content.empty()) b += "null";
        else appendJsonStr(b, f.content);
        if (!f.reasoning.empty()) {
            b += ",\"reasoning_content\":";
            appendJsonStr(b, f.reasoning);
        }
        if (f.calls) {
            b += ",\"tool_calls\":[";
            for (std::size_t i = 0; i < f.calls->size(); ++i) {
                const ToolCall& c = (*f.calls)[i];
                const std::string cid = cx->ids.id("call_", 24);
                if (i > 0) b += ",";
                b += "{\"id\":";
                appendJsonStr(b, cid);
                b += ",\"type\":\"function\",\"function\":{\"name\":";
                appendJsonStr(b, c.name);
                b += ",\"arguments\":";
                appendJsonStr(b, c.args);
                b += "}}";
            }
            b += "]";
        }
        b += "},\"logprobs\":null,\"finish_reason\":";
    } else {
        b += std::format(",\"object\":\"text_completion\",\"created\":{},\"model\":", created);
        appendJsonStr(b, cx->model_name);
        b += ",\"system_fingerprint\":\"whirl\",\"choices\":[{\"index\":0,\"text\":";
        appendJsonStr(b, f.content);
        b += ",\"logprobs\":null,\"finish_reason\":";
    }
    appendJsonStr(b, f.reason);
    appendCause(b, f);
    b += "}],\"usage\":";
    writeUsage(b, n_prompt, tim);
    b += ",\"timings\":";
    tim.write(b);
    b += "}";
    const std::string r = httpResponse(200, "application/json; charset=utf-8", b);
    if (!job->w || !job->w->write(r) || !job->w->flush()) fail();
}

}  // namespace whirl::server
