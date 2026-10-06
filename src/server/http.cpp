// Reimplements the WHIRL Zig research prototype's server.zig (connThread, handleConn, buildJob,
// acceptLoop); sockets written new on Winsock.
// SPDX-License-Identifier: Apache-2.0

#include "http.h"

#include "log.h"
#include "whirl/version.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <optional>
#include <format>
#include <mutex>
#include <set>
#include <string>
#include <string_view>

#pragma comment(lib, "ws2_32.lib")

namespace whirl::server {

namespace {

constexpr std::size_t kReadBuf = 1 << 16;
constexpr std::size_t kMaxBody = 64u << 20;
// unsent response bytes of one connection while the engine writes (non-blocking)
constexpr std::size_t kMaxPending = 64u << 20;

using SteadyClock = std::chrono::steady_clock;

void setTimeout(SOCKET s, int opt, std::uint32_t ms) {
    const DWORD v = ms;
    setsockopt(s, SOL_SOCKET, opt, reinterpret_cast<const char*>(&v), sizeof v);
}

// After shutdown(SD_SEND): read and discard what the client still sends (an
// unread request body, the rest of an oversized header) until it closes, for at
// most budget_ms / 1 MiB. Closing with unread received data makes Windows reset
// the connection, and the client may then lose the response it was sent.
void lingerDrain(SOCKET s, std::uint32_t budget_ms) {
    const auto end = SteadyClock::now() + std::chrono::milliseconds(budget_ms);
    char buf[4096];
    std::size_t total = 0;
    for (;;) {
        const auto left = std::chrono::ceil<std::chrono::milliseconds>(end - SteadyClock::now()).count();
        if (left <= 0 || total > (1u << 20)) return;
        setTimeout(s, SO_RCVTIMEO, static_cast<std::uint32_t>(left));
        const int n = ::recv(s, buf, sizeof buf, 0);
        if (n <= 0) return;
        total += static_cast<std::size_t>(n);
    }
}

// Response writer of a connection. While a request runs, the engine's main
// thread writes through it in non-blocking mode: bytes the client does not
// accept stay buffered and are retried on the next flush, so a client that stops
// reading never stalls the engine (and so the other slots); after send_timeout
// without progress (or kMaxPending unsent bytes) the connection counts as gone and
// the engine drops the request. When the job is done, the connection thread sends
// what is left (blocking, bounded by SO_SNDTIMEO).
// The CORS headers of the request are inserted after the status line of the
// response.
class SocketConn final : public Conn {
public:
    SocketConn(SOCKET s, std::uint32_t send_timeout_ms) : s_(s), stall_ms_(send_timeout_ms) { buf_.reserve(16384); }
    void setCors(std::string h) { cors_ = std::move(h); }
    bool write(std::string_view data) override {
        if (bad_) return false;
        if (!started_) {
            started_ = true;
            const std::size_t eol = data.find("\r\n");
            if (!cors_.empty() && data.starts_with("HTTP/") && eol != std::string_view::npos) {
                buf_.append(data.substr(0, eol + 2));
                buf_.append(cors_);
                data.remove_prefix(eol + 2);
            }
        }
        buf_.append(data);
        if (buf_.size() - off_ >= 16384) return flush();
        return true;
    }
    bool flush() override {
        if (bad_) return false;
        while (off_ < buf_.size()) {
            const int n = ::send(s_, buf_.data() + off_, static_cast<int>(std::min<std::size_t>(buf_.size() - off_, 1 << 20)), 0);
            if (n > 0) {
                off_ += static_cast<std::size_t>(n);
                stalled_ = false;
                continue;
            }
            if (nonblocking_ && n == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK) {
                const auto now = SteadyClock::now();
                if (!stalled_) {
                    stalled_ = true;
                    stall_t0_ = now;
                } else if (now - stall_t0_ >= std::chrono::milliseconds(stall_ms_)) {
                    return setBad();
                }
                if (buf_.size() - off_ > kMaxPending) return setBad();
                if (off_ >= (1u << 20)) {  // drop the sent prefix now and then
                    buf_.erase(0, off_);
                    off_ = 0;
                }
                return true;  // kept for the next flush
            }
            return setBad();
        }
        buf_.clear();
        off_ = 0;
        return true;
    }
    // Non-blocking while the engine writes; back to blocking for the final drain.
    void setNonBlocking(bool on) {
        u_long v = on ? 1 : 0;
        if (ioctlsocket(s_, FIONBIO, &v) == 0) nonblocking_ = on;
        stalled_ = false;
    }
    bool bad() const { return bad_; }

private:
    bool setBad() {
        bad_ = true;
        buf_.clear();
        off_ = 0;
        return false;
    }
    SOCKET s_;
    std::uint32_t stall_ms_;
    std::string buf_;
    std::size_t off_ = 0;
    std::string cors_;
    bool started_ = false;
    bool bad_ = false;
    bool nonblocking_ = false;
    bool stalled_ = false;
    SteadyClock::time_point stall_t0_{};
};

// Buffered reader over a socket. Each recv waits at most idle_ms (SO_RCVTIMEO);
// with a deadline set (the header phase), also no longer than the deadline.
class Reader {
public:
    Reader(SOCKET s, std::uint32_t idle_ms) : s_(s), idle_ms_(idle_ms), cur_to_(idle_ms) { buf_.resize(kReadBuf); }
    enum class Res { ok, eof, too_long, timeout, error };
    void setDeadline(std::optional<SteadyClock::time_point> d) { deadline_ = d; }
    // the header deadline passed (answer 408); a read that was idle for
    // recv_timeout is only closed (nothing useful to tell an idle client)
    bool deadlineHit() const { return deadline_hit_; }
    // One line including '\n' (or up to EOF); the line must fit kReadBuf.
    Res line(std::string& out) {
        out.clear();
        for (;;) {
            const char* b = buf_.data() + pos_;
            const void* nl = std::memchr(b, '\n', end_ - pos_);
            if (nl) {
                const std::size_t n = static_cast<const char*>(nl) - b + 1;
                out.assign(b, n);
                pos_ += n;
                return Res::ok;
            }
            if (end_ - pos_ >= kReadBuf) return Res::too_long;
            if (!fill()) {
                if (end_ > pos_) {
                    out.assign(buf_.data() + pos_, end_ - pos_);
                    pos_ = end_;
                    return Res::ok;
                }
                return eof_ ? Res::eof : timed_out_ ? Res::timeout : Res::error;
            }
        }
    }
    // n bytes; the string grows as data arrives (no up-front reservation of a
    // claimed Content-Length)
    bool read(std::string& out, std::size_t n) {
        out.clear();
        out.reserve(std::min<std::size_t>(n, kReadBuf));
        while (out.size() < n) {
            if (pos_ == end_ && !fill()) return false;
            const std::size_t take = std::min(n - out.size(), end_ - pos_);
            out.append(buf_.data() + pos_, take);
            pos_ += take;
        }
        return true;
    }

private:
    bool fill() {
        if (pos_ > 0) {
            std::memmove(buf_.data(), buf_.data() + pos_, end_ - pos_);
            end_ -= pos_;
            pos_ = 0;
        }
        if (end_ == buf_.size()) return false;
        std::uint32_t to = idle_ms_;
        if (deadline_) {
            const auto now = SteadyClock::now();
            if (now >= *deadline_) {
                timed_out_ = deadline_hit_ = true;
                return false;
            }
            const auto left = std::chrono::ceil<std::chrono::milliseconds>(*deadline_ - now).count();
            to = static_cast<std::uint32_t>(std::min<long long>(to, std::max<long long>(left, 1)));
        }
        if (to != cur_to_) {
            setTimeout(s_, SO_RCVTIMEO, to);
            cur_to_ = to;
        }
        const int n = ::recv(s_, buf_.data() + end_, static_cast<int>(buf_.size() - end_), 0);
        if (n == 0) eof_ = true;
        if (n == SOCKET_ERROR && WSAGetLastError() == WSAETIMEDOUT) {
            timed_out_ = true;
            // cut short by the deadline (not idle): still answer 408 (best effort)
            if (deadline_ && to < idle_ms_) deadline_hit_ = true;
        }
        if (n <= 0) return false;
        end_ += static_cast<std::size_t>(n);
        return true;
    }
    SOCKET s_;
    std::uint32_t idle_ms_, cur_to_;
    std::optional<SteadyClock::time_point> deadline_;
    std::string buf_;
    std::size_t pos_ = 0, end_ = 0;
    bool eof_ = false;
    bool timed_out_ = false;
    bool deadline_hit_ = false;
};

bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    return true;
}

bool icontains(std::string_view h, std::string_view n) {
    if (n.size() > h.size()) return false;
    for (std::size_t i = 0; i + n.size() <= h.size(); ++i)
        if (ieq(h.substr(i, n.size()), n)) return true;
    return false;
}

void sendAll(Conn& w, const std::string& s) {
    w.write(s);
    w.flush();
}

// ---- vision (image_url content parts; ported from the prototype's server.zig, item VIS)

const std::string* partStr(const json::Value& o, std::string_view k) {
    const json::Value* v = o.get(k);
    return v && v->isString() ? &v->asString() : nullptr;
}

// Image / video parts the server cannot take (checked before rendering, with the
// prototype's messages): images without an mmproj, any video part.
void checkMediaParts(const json::Array& msgs, bool vis_on) {
    for (const json::Value& m : msgs) {
        if (!m.isObject()) continue;
        const json::Value* v = m.get("content");
        if (!v || !v->isArray()) continue;
        for (const json::Value& it : v->asArray()) {
            if (!it.isObject()) continue;
            if (partStr(it, "text")) continue;
            const bool img = it.get("image_url") != nullptr || it.get("image") != nullptr;
            if (img && !vis_on) throw RequestError("image content needs an mmproj (start the server with --mmproj MMPROJ.gguf)");
            if (img) continue;
            if (it.get("video") != nullptr) throw RequestError("video content is not supported by this server");
            throw RequestError("unexpected content part");
        }
    }
}

// The image sources of a chat request in prompt order (empty: none).
std::vector<std::string> collectImages(const json::Array& msgs) {
    std::vector<std::string> out;
    for (const json::Value& m : msgs) {
        if (!m.isObject()) continue;
        const json::Value* v = m.get("content");
        if (!v || !v->isArray()) continue;
        const std::string* role = partStr(m, "role");
        for (const json::Value& it : v->asArray()) {
            if (!it.isObject()) continue;
            const json::Value* iv = it.get("image_url");
            if (!iv) iv = it.get("image");
            if (!iv) continue;
            if (role && *role == "system") throw RequestError("System message cannot contain images.");
            if (iv->isString()) {
                out.push_back(iv->asString());
            } else if (iv->isObject()) {
                const std::string* u = partStr(*iv, "url");
                if (!u) throw RequestError("image_url.url is required");
                out.push_back(*u);
            } else {
                throw RequestError("image_url must be an object {url} or a string");
            }
        }
    }
    return out;
}

// Bytes of an image source: a data: URL (base64), or with --allow-local-images a
// local path / file:// URL. No network fetches.
std::vector<std::uint8_t> imageBytes(const std::string& src, bool allow_files) {
    std::vector<std::uint8_t> out;
    if (src.starts_with("data:")) {
        const std::size_t comma = src.find(',');
        if (comma == std::string::npos) throw RequestError("malformed data: URL");
        if (std::string_view(src).substr(0, comma).find(";base64") == std::string_view::npos) throw RequestError("data: URLs must be base64");
        std::string_view b64 = std::string_view(src).substr(comma + 1);
        while (!b64.empty() && (b64.back() == '\n' || b64.back() == '\r' || b64.back() == ' ')) b64.remove_suffix(1);
        if (!base64Decode(b64, out)) throw RequestError("invalid base64 in data: URL");
        return out;
    }
    if (src.starts_with("http://") || src.starts_with("https://"))
        throw RequestError("remote image URLs are not fetched; send the image as a data: URL (base64)");
    if (!allow_files)
        throw RequestError("image sources must be data: URLs (start the server with --allow-local-images to accept local file paths)");
    const std::string p = src.starts_with("file:///") ? src.substr(8) : src.starts_with("file://") ? src.substr(7) : src;
    try {
        const std::string s = readFile(p);
        if (s.size() > (256ull << 20)) throw std::runtime_error("too large");
        out.assign(s.begin(), s.end());
    } catch (const std::exception&) {
        throw RequestError("cannot read the local image file");
    }
    return out;
}

// Decode + preprocess the request's images, expand their placeholders.
std::shared_ptr<JobVis> buildVis(const VisionConfig& vc, const std::vector<std::string>& srcs, std::vector<std::uint32_t>& tokens) {
    auto jv = std::make_shared<JobVis>();
    jv->preps.reserve(srcs.size());
    for (const std::string& src : srcs) {
        const std::vector<std::uint8_t> bytes = imageBytes(src, vc.allow_files);
        try {
            jv->preps.push_back(vc.vis->prepare(bytes));
        } catch (const vision::VisionError& ex) {
            if (ex.code() == "ImageDecode") throw RequestError("cannot decode the image (PNG, JPEG, BMP, GIF, TGA, PSD, PNM are supported)");
            if (ex.code() == "ImageTooLarge") throw RequestError("image larger than 16384 pixels on a side");
            throw;
        }
    }
    std::vector<const vision::Prepared*> ptrs;
    for (const vision::Prepared& p : jv->preps) ptrs.push_back(&p);
    jv->spans.resize(srcs.size());
    try {
        tokens = vision::expand(tokens, vc.image_pad_id, ptrs, jv->spans);
    } catch (const vision::VisionError& ex) {
        if (ex.code() == "ImagePlaceholderMismatch")
            throw RequestError("image placeholders do not match the images (is <|image_pad|> written in the text?)");
        throw;
    }
    jv->vmap.spans = jv->spans;
    jv->ents.assign(srcs.size(), nullptr);
    return jv;
}

// Builds a job from the request body (throws RequestError for HTTP 400).
void buildJob(Engine& e, Job& job, std::string_view path, std::string_view body) {
    json::Value root;
    try {
        root = json::parse(body);
    } catch (const std::exception&) {
        throw RequestError("request body is not valid JSON");
    }
    if (!root.isObject()) throw RequestError("request body must be a JSON object");
    const json::Object& obj = root.asObject();
    if (job.kind == JobKind::chat) {
        chat::ChatRequest req = chat::parseChatRequest(root);
        for (const std::string& warn : req.warnings) logW("req {} | POST {} | {}", job.id, path, warn);
        const VisionConfig& vc = e.visionConfig();
        // vision: image parts render as the Qwen placeholder when an mmproj is loaded
        checkMediaParts(req.messages, vc.vis != nullptr);
        req.options.allow_media = vc.vis != nullptr;
        const std::string prompt = chat::renderChat(e.options().tmpl, req.messages, req.tools, req.options);
        const auto ids = e.tokenizer().encode(prompt, true);
        job.tokens.assign(ids.begin(), ids.end());
        // vision: image parts -> per-image content ids (decoding here, on the connection thread)
        if (vc.vis != nullptr) {
            const std::vector<std::string> srcs = collectImages(req.messages);
            if (!srcs.empty()) job.vis = buildVis(vc, srcs, job.tokens);
        }
        job.think = req.options.think;
        job.tools_on = !req.tools.empty();
        const std::size_t n_tools = req.tools.size();
        job.tools = std::move(req.tools);
        job.params = parseParams(obj, req.options.think, true);
        logI("req {} | POST {} | stream {} | {} messages, {} tools (tool_choice {}), thinking {}{}{}", job.id, path,
             job.params.stream ? "on" : "off", req.messages.size(), n_tools, req.tool_choice,
             req.options.think ? "on" : "off", req.options.effort ? ", reasoning_effort " : "",
             req.options.effort ? *req.options.effort : std::string());
    } else {
        const json::Value* pv = obj.find("prompt");
        if (!pv || pv->isNull()) throw RequestError("'prompt' is required");
        const char* shape = "'prompt' must be a string, a list with one string, or a list of token ids";
        if (pv->isString()) {
            const auto ids = e.tokenizer().encode(pv->asString(), true);
            job.tokens.assign(ids.begin(), ids.end());
        } else if (pv->isArray()) {
            const json::Array& arr = pv->asArray();
            if (arr.empty()) throw RequestError("'prompt' must not be empty");
            if (arr[0].isString()) {
                if (arr.size() > 1) throw RequestError("only one prompt per request is supported");
                const auto ids = e.tokenizer().encode(arr[0].asString(), true);
                job.tokens.assign(ids.begin(), ids.end());
            } else if (arr[0].kind() == json::Value::Kind::integer) {
                const std::int64_t V = e.model().cfg().n_vocab;
                for (const json::Value& it : arr) {
                    if (it.kind() != json::Value::Kind::integer || it.asInt() < 0 || it.asInt() >= V)
                        throw RequestError("prompt token ids must be integers in [0, n_vocab)");
                    job.tokens.push_back(static_cast<std::uint32_t>(it.asInt()));
                }
            } else {
                throw RequestError(shape);
            }
        } else {
            throw RequestError(shape);
        }
        job.params = parseParams(obj, true, false);
        logI("req {} | POST {} | stream {}", job.id, path, job.params.stream ? "on" : "off");
    }
    if (job.tokens.empty()) throw RequestError("the prompt is empty");
}

// The model's file name for GET /props: never a directory (the option is set to
// the bare file name; anything before a path separator is dropped regardless).
std::string_view modelFileName(const EngineOptions& o) {
    std::string_view f = o.model_file.empty() ? std::string_view(o.model_name) : std::string_view(o.model_file);
    if (const std::size_t sep = f.find_last_of("/\\:"); sep != std::string_view::npos) f = f.substr(sep + 1);
    return f;
}

// GET /props: the subset of llama.cpp server's /props that clients read to
// auto-detect the context window and slot count (written from the documented
// JSON shape). No chat_template: WHIRL renders built-in templates, not Jinja.
std::string propsJson(Engine& e) {
    const EngineOptions& o = e.options();
    const Engine::Health hs = e.health();
    std::string b = "{\"default_generation_settings\":{\"id\":0,\"n_ctx\":";
    b += std::to_string(o.ctx);
    b += ",\"model\":";
    appendJsonStr(b, o.model_name);
    b += std::format(",\"speculative\":{},\"is_processing\":{}}}", o.use_mtp ? "true" : "false",
                     hs.active > 0 ? "true" : "false");
    b += std::format(",\"total_slots\":{},\"model_path\":", hs.slots);
    appendJsonStr(b, modelFileName(o));
    b += ",\"model_alias\":";
    appendJsonStr(b, o.model_name);
    b += std::format(",\"modalities\":{{\"vision\":{},\"audio\":false}},\"build_info\":\"whirl {}\"",
                     e.visionConfig().vis ? "true" : "false", WHIRL_VERSION_STRING);
    if (!o.numerics_json.empty()) b += ",\"numerics\":" + o.numerics_json;
    b += "}";
    return b;
}

// Paths that other servers' clients probe to detect the server type (LM Studio,
// Ollama). WHIRL does not emulate those native APIs: answering them would make a
// client switch to an API WHIRL lacks. 404, logged once per path at 'I' level
// instead of a warning per request.
bool isKnownProbe(std::string_view path) {
    return path == "/api/v1/models" || path == "/api/tags" || path == "/api/version" || path == "/api/show" ||
           path == "/api/v0/models";
}

bool firstProbe(std::string_view path) {
    static std::mutex mu;
    static std::set<std::string, std::less<>> seen;
    std::lock_guard<std::mutex> lk(mu);
    return seen.emplace(path).second;
}

// Per-connection context from the server.
struct ConnCtx {
    Engine& e;
    const HttpOptions& o;
    std::function<void()> on_read;  // the request has been read (stop() no longer cuts it)
};

void handleConn(const ConnCtx& cx, SOCKET s) {
    Engine& e = cx.e;
    e.beginBuilding();
    bool building = true;
    struct Guard {
        Engine& e;
        bool& b;
        ~Guard() {
            if (b) e.endBuilding();
        }
    } guard{e, building};
    Reader r(s, cx.o.recv_timeout_ms);
    r.setDeadline(SteadyClock::now() + std::chrono::milliseconds(cx.o.header_timeout_ms));
    SocketConn w(s, cx.o.send_timeout_ms);
    auto timedOut = [&] {
        if (r.deadlineHit()) sendAll(w, errorResponse(408, "request headers not received in time"));
    };
    std::string line0;
    if (const Reader::Res res = r.line(line0); res != Reader::Res::ok) {
        if (res == Reader::Res::too_long) return sendAll(w, errorResponse(413, "request line too long"));
        if (res == Reader::Res::timeout) timedOut();
        return;
    }
    const std::string req_line(trimWs(line0));
    std::string_view rl = req_line;
    auto nextTok = [&]() -> std::string_view {
        while (!rl.empty() && rl.front() == ' ') rl.remove_prefix(1);
        const std::size_t sp = rl.find(' ');
        std::string_view t = rl.substr(0, sp);
        rl = sp == std::string_view::npos ? std::string_view() : rl.substr(sp);
        return t;
    };
    const std::string method(nextTok());
    const std::string_view target = nextTok();
    if (method.empty() || target.empty()) return sendAll(w, errorResponse(400, "malformed request line"));
    std::string path(target.substr(0, target.find('?')));
    std::size_t content_len = 0;
    bool have_len = false;
    bool chunked = false;
    std::string origin;
    std::size_t header_bytes = line0.size();
    std::uint32_t header_lines = 0;
    std::string hl;
    for (;;) {
        const Reader::Res res = r.line(hl);
        if (res == Reader::Res::too_long) return sendAll(w, errorResponse(431, "header line too long"));
        if (res == Reader::Res::timeout) return timedOut();
        if (res != Reader::Res::ok) return;
        header_bytes += hl.size();
        if (header_bytes > cx.o.max_header_bytes || ++header_lines > cx.o.max_header_lines)
            return sendAll(w, errorResponse(431, "request headers too large"));
        const std::string_view h = trimWs(hl);
        if (h.empty()) break;
        const std::size_t colon = h.find(':');
        if (colon == std::string_view::npos) continue;
        const std::string_view name = trimWs(h.substr(0, colon));
        const std::string_view val = trimWs(h.substr(colon + 1));
        if (ieq(name, "content-length")) {
            // digits only; anything above the body limit is 413 (no overflow)
            std::size_t v = 0;
            bool ok = !val.empty();
            bool big = false;
            for (char c : val) {
                if (c < '0' || c > '9') {
                    ok = false;
                    break;
                }
                if (!big) {
                    v = v * 10 + static_cast<std::size_t>(c - '0');
                    if (v > kMaxBody) big = true;
                }
            }
            if (!ok || (have_len && !big && v != content_len)) return sendAll(w, errorResponse(400, "bad Content-Length"));
            if (big) return sendAll(w, errorResponse(413, "request body too large"));
            content_len = v;
            have_len = true;
        } else if (ieq(name, "transfer-encoding")) {
            if (icontains(val, "chunked")) chunked = true;
        } else if (ieq(name, "origin")) {
            origin.assign(val);
        }
    }
    r.setDeadline(std::nullopt);
    const std::string cors = corsHeaders(origin, cx.o.cors_origins);
    w.setCors(cors);
    if (method == "OPTIONS") {
        // preflight: allowed Origin -> the CORS headers (Allow-Origin from setCors);
        // other Origins get a bare 204 and the browser blocks the request
        std::string resp = "HTTP/1.1 204 No Content\r\n";
        if (!cors.empty())
            resp += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n"
                    "Access-Control-Max-Age: 86400\r\n";
        resp += "Content-Length: 0\r\nConnection: close\r\n\r\n";
        return sendAll(w, resp);
    }
    if (chunked) return sendAll(w, errorResponse(411, "chunked request bodies are not supported; send Content-Length"));
    if (content_len > kMaxBody) return sendAll(w, errorResponse(413, "request body too large"));
    std::string body;
    if (content_len > 0 && !r.read(body, content_len)) return;
    if (cx.on_read) cx.on_read();

    const bool is_get = method == "GET";
    const bool is_post = method == "POST";
    if (path == "/health" || path == "/v1/health") {
        if (!is_get) return sendAll(w, errorResponse(405, "use GET"));
        const Engine::Health hs = e.health();
        const std::string b = std::format("{{\"status\":\"ok\",\"busy\":{},\"slots_busy\":{},\"slots\":{},\"queued\":{}}}",
                                          hs.active > 0 ? "true" : "false", hs.active, hs.slots, hs.queued);
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8", b));
    }
    if (path == "/v1/models" || path == "/models") {
        if (!is_get) return sendAll(w, errorResponse(405, "use GET"));
        const auto& cfg = e.model().cfg();
        std::string b = "{\"object\":\"list\",\"data\":[{\"id\":";
        appendJsonStr(b, e.options().model_name);
        b += std::format(
            ",\"object\":\"model\",\"created\":{},\"owned_by\":\"whirl\",\"meta\":{{\"arch\":\"{}\",\"n_ctx\":{},\"n_vocab\":{},"
            "\"n_embd\":{},\"n_layer\":{},\"mtp\":{}}}}}]}}",
            static_cast<long long>(std::time(nullptr)), cfg.archName(), e.options().ctx, cfg.n_vocab, cfg.n_embd,
            cfg.n_layer, e.options().use_mtp ? "true" : "false");
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8", b));
    }
    if (path == "/props" || path == "/v1/props") {
        if (!is_get) return sendAll(w, errorResponse(405, "use GET"));
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8", propsJson(e)));
    }
    if (path == "/version" || path == "/v1/version") {
        if (!is_get) return sendAll(w, errorResponse(405, "use GET"));
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8",
                                       std::format("{{\"version\":\"{}\",\"name\":\"whirl\"}}", WHIRL_VERSION_STRING)));
    }
    const bool is_chat = path == "/v1/chat/completions" || path == "/chat/completions";
    const bool is_cmpl = path == "/v1/completions" || path == "/completions";
    if (!is_chat && !is_cmpl) {
        if (isKnownProbe(path)) {
            if (firstProbe(path))
                logI("{} {} -> 404 (probe for another server type; further probes of this path are not logged)", method,
                     path);
            return sendAll(w, errorResponse(404, "not found"));
        }
        logW("{} {} -> 404", method, path);
        return sendAll(w, errorResponse(404, "not found"));
    }
    if (!is_post) return sendAll(w, errorResponse(405, "use POST"));

    Job job;
    job.id = e.nextJobId();
    job.t_arrive = Clock::now();
    job.kind = is_chat ? JobKind::chat : JobKind::completion;
    job.endpoint = path;
    job.w = &w;
    try {
        buildJob(e, job, path, body);
    } catch (const RequestError& ex) {
        logW("req {} | POST {} -> 400: {}", job.id, path, ex.what());
        return sendAll(w, errorResponse(400, ex.what()));
    } catch (const std::exception& ex) {
        logE("req {} | POST {} -> 500: {}", job.id, path, ex.what());
        return sendAll(w, errorResponse(500, ex.what()));
    }
    // enqueue and wait (the main thread runs the job and writes the response,
    // without blocking on this client; what the client has not taken yet is sent
    // here afterwards)
    building = false;
    w.setNonBlocking(true);
    e.submitAndWait(job);
    w.setNonBlocking(false);
    w.flush();
}

std::once_flag g_net_once;

std::string lower(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

}  // namespace

bool isLoopbackOrigin(std::string_view origin) {
    const std::string o = lower(origin);
    std::string_view v = o;
    if (v.starts_with("http://")) v.remove_prefix(7);
    else if (v.starts_with("https://")) v.remove_prefix(8);
    else return false;
    bool host = false;
    for (std::string_view h : {"localhost", "127.0.0.1", "[::1]"})
        if (v.starts_with(h)) {
            v.remove_prefix(h.size());
            host = true;
            break;
        }
    if (!host) return false;
    if (v.empty()) return true;
    if (v.front() != ':' || v.size() < 2 || v.size() > 6) return false;
    for (char c : v.substr(1))
        if (c < '0' || c > '9') return false;
    return true;
}

std::string corsHeaders(std::string_view origin, const std::vector<std::string>& extra) {
    for (const std::string& x : extra)
        if (x == "*") return "Access-Control-Allow-Origin: *\r\n";
    // an Origin echoed back: only printable ASCII (it goes into a header line)
    if (origin.empty() || origin.size() > 256) return {};
    for (char c : origin)
        if (static_cast<unsigned char>(c) < 0x21 || static_cast<unsigned char>(c) > 0x7e) return {};
    bool ok = isLoopbackOrigin(origin);
    if (!ok) {
        const std::string lo = lower(origin);
        for (const std::string& x : extra) {
            std::string_view xv = x;
            while (!xv.empty() && xv.back() == '/') xv.remove_suffix(1);
            if (lower(xv) == lo) {
                ok = true;
                break;
            }
        }
    }
    if (!ok) return {};
    std::string h = "Access-Control-Allow-Origin: ";
    h += origin;
    h += "\r\nVary: Origin\r\n";
    return h;
}

void netInit() {
    std::call_once(g_net_once, [] {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("WSAStartup failed");
    });
}

HttpServer::HttpServer(Engine& e, HttpOptions o) : e_(e), o_(std::move(o)) { netInit(); }

HttpServer::~HttpServer() { stop(); }

void HttpServer::start(const std::string& host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICHOST;
    addrinfo* res = nullptr;
    const std::string ps = std::to_string(port);
    if (getaddrinfo(host.c_str(), ps.c_str(), &hints, &res) != 0 || !res)
        throw std::runtime_error("invalid --host " + host);
    SOCKET s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) {
        freeaddrinfo(res);
        throw std::runtime_error("socket() failed");
    }
    // exclusive: with SO_REUSEADDR a second server could bind the same port and
    // silently share (steal) its connections; restarting on a port whose old
    // connections are in TIME_WAIT still works
    BOOL on = TRUE;
    setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&on), sizeof on);
    if (bind(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0 || listen(s, SOMAXCONN) != 0) {
        const int err = WSAGetLastError();
        freeaddrinfo(res);
        closesocket(s);
        throw std::runtime_error(std::format("cannot listen on {}:{} (WSA error {})", host, port, err));
    }
    freeaddrinfo(res);
    sockaddr_storage ss{};
    int sl = sizeof ss;
    getsockname(s, reinterpret_cast<sockaddr*>(&ss), &sl);
    port_ = ntohs(ss.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port
                                           : reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    listen_ = static_cast<std::uintptr_t>(s);
    accept_ = std::thread([this] { acceptLoop(); });
}

void HttpServer::stop() {
    if (listen_ == ~std::uintptr_t(0)) return;
    stopping_ = true;
    closesocket(static_cast<SOCKET>(listen_));
    if (accept_.joinable()) accept_.join();
    listen_ = ~std::uintptr_t(0);
    auto shutAll = [this](bool reading_only) {
        // shutdown (not closesocket) from this thread: the connection thread still
        // owns the handle and closes it; it leaves the sets before closing
        std::lock_guard<std::mutex> lk(conns_mu_);
        // (shutdown alone does not end a recv that is already waiting: cancel it)
        for (std::uintptr_t c : reading_only ? reading_ : conns_) {
            shutdown(static_cast<SOCKET>(c), SD_BOTH);
            CancelIoEx(reinterpret_cast<HANDLE>(c), nullptr);
        }
    };
    auto waitConns = [this](int ms) {
        for (int i = 0; i < ms / 5 && n_conn_.load() > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    };
    // connections still reading a request (idle, slow or stuck clients) end now;
    // the others finish on their own (their jobs are answered) or are shut down
    // after a few seconds (a client that does not take its response)
    shutAll(true);
    waitConns(5000);
    if (n_conn_.load() > 0) {
        shutAll(false);
        waitConns(5000);
    }
}

void HttpServer::setReading(std::uintptr_t sock, bool on) {
    std::lock_guard<std::mutex> lk(conns_mu_);
    if (on) reading_.insert(sock);
    else reading_.erase(sock);
}

void HttpServer::acceptLoop() {
    for (;;) {
        const SOCKET c = accept(static_cast<SOCKET>(listen_), nullptr, nullptr);
        if (c == INVALID_SOCKET) {
            if (stopping_) return;
            logE("accept failed: WSA error {}", WSAGetLastError());
            continue;
        }
        setTimeout(c, SO_RCVTIMEO, o_.recv_timeout_ms);
        setTimeout(c, SO_SNDTIMEO, o_.send_timeout_ms);
        if (o_.sndbuf > 0) setsockopt(c, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&o_.sndbuf), sizeof o_.sndbuf);
        if (n_conn_.load() >= static_cast<int>(o_.max_conn)) {
            static std::atomic<std::uint64_t> n_rejected{0};
            if (n_rejected.fetch_add(1) % 100 == 0)
                logW("connection limit ({} open connections) reached: answering 503", o_.max_conn);
            setTimeout(c, SO_SNDTIMEO, 1000);
            const std::string r = errorResponse(503, "too many open connections; retry later");
            ::send(c, r.data(), static_cast<int>(r.size()), 0);
            shutdown(c, SD_SEND);
            lingerDrain(c, 50);
            closesocket(c);
            continue;
        }
        n_conn_.fetch_add(1);
        {
            std::lock_guard<std::mutex> lk(conns_mu_);
            conns_.insert(static_cast<std::uintptr_t>(c));
            reading_.insert(static_cast<std::uintptr_t>(c));
        }
        try {
            std::thread([this, c] { connThread(static_cast<std::uintptr_t>(c)); }).detach();
        } catch (const std::exception& ex) {
            logE("cannot spawn connection thread: {}", ex.what());
            {
                std::lock_guard<std::mutex> lk(conns_mu_);
                conns_.erase(static_cast<std::uintptr_t>(c));
                reading_.erase(static_cast<std::uintptr_t>(c));
            }
            closesocket(c);
            n_conn_.fetch_sub(1);
        }
    }
}

void HttpServer::connThread(std::uintptr_t sock) {
    const SOCKET s = static_cast<SOCKET>(sock);
    try {
        const ConnCtx cx{e_, o_, [this, sock] { setReading(sock, false); }};
        handleConn(cx, s);
    } catch (const std::exception& ex) {
        logE("connection error: {}", ex.what());
    }
    {
        std::lock_guard<std::mutex> lk(conns_mu_);
        conns_.erase(sock);
        reading_.erase(sock);
    }
    shutdown(s, SD_SEND);
    lingerDrain(s, 500);
    closesocket(s);
    n_conn_.fetch_sub(1);
}

}  // namespace whirl::server
