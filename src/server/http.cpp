// Reimplements the WHIRL Zig research prototype's server.zig (connThread, handleConn, buildJob,
// acceptLoop); sockets written new on Winsock (arch025 PL3: the Winsock calls are in src/platform/net_win.cpp).
// SPDX-License-Identifier: Apache-2.0

#include "http.h"
#include "whirl/vision.h"  // buildVis: Prepared, expand, VisionError (engine_api.h no longer brings it: arch025-lane2 leg 5 S6)

#include "log.h"
#include "whirl/common.h"
#include "whirl/small_helpers.h"  // lowerAscii (arch025-N6, lane 1 leg 11)
#include "platform/net.h"
#include "whirl/version.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <functional>
#include <optional>
#include <format>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace whirl::server {

namespace {

namespace net = platform::net;

constexpr std::size_t kReadBuf = 1 << 16;
constexpr std::size_t kMaxBody = 64u << 20;
// unsent response bytes of one connection while the engine writes (non-blocking)
constexpr std::size_t kMaxPending = 64u << 20;

using SteadyClock = std::chrono::steady_clock;

// After shutdown(SD_SEND): read and discard what the client still sends (an
// unread request body, the rest of an oversized header) until it closes, for at
// most budget_ms / 1 MiB. Closing with unread received data makes Windows reset
// the connection, and the client may then lose the response it was sent.
// Also used before our own FIN, to let the client close first (connThread).
// Returns true when the client closed (its FIN arrived).
bool lingerDrain(net::Socket s, std::uint32_t budget_ms) {
    const auto end = SteadyClock::now() + std::chrono::milliseconds(budget_ms);
    char buf[4096];
    std::size_t total = 0;
    for (;;) {
        const auto left = std::chrono::ceil<std::chrono::milliseconds>(end - SteadyClock::now()).count();
        if (left <= 0 || total > (1u << 20)) return false;
        net::setRecvTimeout(s, static_cast<std::uint32_t>(left));
        const int n = net::recv(s, buf, sizeof buf, 0);
        if (n == 0) return true;
        if (n < 0) return false;
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
    // framed_out (optional): set at destruction to whether a complete response the client can tell
    // the end of (Content-Length, or chunked and finished) was handed to the socket, so the client can
    // close first (HTTP-DROP)
    SocketConn(net::Socket s, std::uint32_t send_timeout_ms, bool* framed_out = nullptr)
        : s_(s), stall_ms_(send_timeout_ms), framed_out_(framed_out) {
        buf_.reserve(16384);
    }
    ~SocketConn() override {
        if (framed_out_) *framed_out_ = (framed_ || (chunked_ && finished_)) && !bad_;
    }
    void setCors(std::string h) { cors_ = std::move(h); }
    // the request was HTTP/1.0: no Transfer-Encoding may be sent (RFC 9112 section 6.1), so a stream goes
    // out unframed and ends at the server's close, as before HTTP-DROP
    void setHttp10(bool v) { http10_ = v; }
    bool write(std::string_view data) override {
        if (bad_) return false;
        std::string head10;  // HTTP/1.0: the first write without its Transfer-Encoding line
        if (!started_) {
            started_ = true;
            // the first write carries the whole response header
            const std::size_t he = data.find("\r\n\r\n");
            if (data.starts_with("HTTP/") && he != std::string_view::npos) {
                std::string h(data.substr(0, he + 2));
                for (auto& ch : h) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                framed_ = h.find("\r\ncontent-length:") != std::string::npos;
                const std::size_t te = h.find("\r\ntransfer-encoding: chunked\r\n");
                chunked_ = te != std::string::npos;
                if (chunked_ && http10_) {
                    head10.assign(data.substr(0, te + 2));
                    head10.append(data.substr(te + 2 + std::string_view("transfer-encoding: chunked\r\n").size()));
                    data = head10;
                    chunked_ = false;
                }
            }
            const std::size_t eol = data.find("\r\n");
            if (!cors_.empty() && data.starts_with("HTTP/") && eol != std::string_view::npos) {
                buf_.append(data.substr(0, eol + 2));
                buf_.append(cors_);
                data.remove_prefix(eol + 2);
            }
            if (chunked_) {  // the header as it is; what follows it is body
                const std::size_t e = data.find("\r\n\r\n");
                buf_.append(data.substr(0, e + 4));
                data.remove_prefix(e + 4);
            }
        }
        if (chunked_) {
            // HTTP-DROP: an SSE stream is sent chunked, so the client sees where it ends without
            // waiting for the server to close (and can close first); one chunk per write
            if (data.empty() || finished_) return true;
            buf_ += std::format("{:x}\r\n", data.size());
            buf_.append(data);
            buf_ += "\r\n";
        } else {
            buf_.append(data);
        }
        if (buf_.size() - off_ >= 16384) return flush();
        return true;
    }
    // the response is complete: a chunked one gets its last (empty) chunk
    void finish() {
        if (chunked_ && !finished_ && !bad_) buf_ += "0\r\n\r\n";
        finished_ = true;
    }
    bool flush() override {
        if (bad_) return false;
        while (off_ < buf_.size()) {
            const int n = net::send(s_, buf_.data() + off_, static_cast<int>(std::min<std::size_t>(buf_.size() - off_, 1 << 20)));
            if (n > 0) {
                off_ += static_cast<std::size_t>(n);
                stalled_ = false;
                continue;
            }
            if (nonblocking_ && n == net::kError && net::lastError() == net::kErrWouldBlock) {
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
        if (net::setNonBlocking(s_, on) == 0) nonblocking_ = on;
        stalled_ = false;
    }
    bool bad() const { return bad_; }
    // FIN / reset from the client: readable with nothing to read (or an error). Bytes waiting
    // (a pipelined request) count as alive. Non-blocking; the engine calls it while the
    // connection thread waits for the job.
    bool alive() override {
        if (bad_) return false;
        const int n = net::pollReadable(s_);
        if (n <= 0) return true;  // nothing to read (or select failed: assume alive)
        char c;
        const int r = net::recv(s_, &c, 1, net::kPeek);
        if (r > 0) return true;
        if (r == 0) return false;
        return net::lastError() == net::kErrWouldBlock;
    }

private:
    bool setBad() {
        bad_ = true;
        buf_.clear();
        off_ = 0;
        return false;
    }
    net::Socket s_;
    std::uint32_t stall_ms_;
    std::string buf_;
    std::size_t off_ = 0;
    std::string cors_;
    bool started_ = false;
    bool bad_ = false;
    bool nonblocking_ = false;
    bool stalled_ = false;
    bool* framed_out_ = nullptr;
    bool http10_ = false;
    bool framed_ = false;
    bool chunked_ = false;
    bool finished_ = false;
    SteadyClock::time_point stall_t0_{};
};

// Buffered reader over a socket. Each recv waits at most idle_ms (SO_RCVTIMEO);
// with a deadline set (the header phase), also no longer than the deadline.
class Reader {
public:
    Reader(net::Socket s, std::uint32_t idle_ms) : s_(s), idle_ms_(idle_ms), cur_to_(idle_ms) { buf_.resize(kReadBuf); }
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
            net::setRecvTimeout(s_, to);
            cur_to_ = to;
        }
        const int n = net::recv(s_, buf_.data() + end_, static_cast<int>(buf_.size() - end_), 0);
        if (n == 0) eof_ = true;
        if (n == net::kError && net::lastError() == net::kErrTimedOut) {
            timed_out_ = true;
            // cut short by the deadline (not idle): still answer 408 (best effort)
            if (deadline_ && to < idle_ms_) deadline_hit_ = true;
        }
        if (n <= 0) return false;
        end_ += static_cast<std::size_t>(n);
        return true;
    }
    net::Socket s_;
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

// Client-supplied text for a log line (FR-SRV P3-11): control bytes (ESC sequences, CR/LF that
// would forge log lines) as \xNN, at most 256 bytes.
std::string logSafe(std::string_view s) {
    std::string out;
    const std::size_t n = std::min<std::size_t>(s.size(), 256);
    out.reserve(n + 8);
    for (std::size_t i = 0; i < n; ++i) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x20 || c == 0x7f) out += std::format("\\x{:02x}", c);
        else out.push_back(static_cast<char>(c));
    }
    if (s.size() > n) out += "...";
    return out;
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
std::vector<std::uint8_t> imageBytes(const std::string& src, bool allow_files, bool web) {
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
    // FR-SRV P2-4: a web page (a request with an Origin other than a localhost page) never reads
    // local files, whatever --cors-origin allows
    if (web) throw RequestError("local image files are not accepted from web pages (send the image as a data: URL)");
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
std::shared_ptr<JobVis> buildVis(const VisionConfig& vc, const std::vector<std::string>& srcs, std::vector<std::uint32_t>& tokens,
                                 bool web) {
    auto jv = std::make_shared<JobVis>();
    jv->preps.reserve(srcs.size());
    for (const std::string& src : srcs) {
        const std::vector<std::uint8_t> bytes = imageBytes(src, vc.allow_files, web);
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

}  // namespace

// (exported, http.h)
ChatPrompt chatPrompt(const Tokenizer& tok, chat::TemplateKind tmpl, chat::ChatRequest& req, bool media) {
    checkMediaParts(req.messages, media);
    req.options.allow_media = media;
    ChatPrompt cp;
    cp.text = chat::renderChat(tmpl, req.messages, req.tools, req.options);
    cp.ids = tok.encode(cp.text, true);
    return cp;
}

// (exported, http.h)
std::vector<std::uint32_t> completionTokens(const Tokenizer& tok, std::uint32_t n_vocab, const json::Object& obj) {
    std::vector<std::uint32_t> out;
    const json::Value* pv = obj.find("prompt");
    if (!pv || pv->isNull()) throw RequestError("'prompt' is required");
    const char* shape = "'prompt' must be a string, a list with one string, or a list of token ids";
    if (pv->isString()) {
        const auto ids = tok.encode(pv->asString(), true);
        out.assign(ids.begin(), ids.end());
    } else if (pv->isArray()) {
        const json::Array& arr = pv->asArray();
        if (arr.empty()) throw RequestError("'prompt' must not be empty");
        if (arr[0].isString()) {
            if (arr.size() > 1) throw RequestError("only one prompt per request is supported");
            const auto ids = tok.encode(arr[0].asString(), true);
            out.assign(ids.begin(), ids.end());
        } else if (arr[0].kind() == json::Value::Kind::integer) {
            const std::int64_t V = n_vocab;
            for (const json::Value& it : arr) {
                if (it.kind() != json::Value::Kind::integer || it.asInt() < 0 || it.asInt() >= V)
                    throw RequestError("prompt token ids must be integers in [0, n_vocab)");
                out.push_back(static_cast<std::uint32_t>(it.asInt()));
            }
        } else {
            throw RequestError(shape);
        }
    } else {
        throw RequestError(shape);
    }
    return out;
}

namespace {

// Builds a job from the request body (throws RequestError for HTTP 400).
void buildJob(const EngineFacts& f, Job& job, std::string_view path, std::string_view body, bool web) {
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
        const VisionConfig& vc = f.vision;
        // vision: image parts render as the Qwen placeholder when an mmproj is loaded
        const ChatPrompt cp = chatPrompt(*f.tok, f.tmpl, req, vc.vis != nullptr);
        job.tokens.assign(cp.ids.begin(), cp.ids.end());
        // vision: image parts -> per-image content ids (decoding here, on the connection thread)
        if (vc.vis != nullptr) {
            const std::vector<std::string> srcs = collectImages(req.messages);
            if (!srcs.empty()) job.vis = buildVis(vc, srcs, job.tokens, web);
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
        job.tokens = completionTokens(*f.tok, f.n_vocab, obj);
        job.params = parseParams(obj, true, false);
        logI("req {} | POST {} | stream {}", job.id, path, job.params.stream ? "on" : "off");
    }
    if (job.tokens.empty()) throw RequestError("the prompt is empty");
}

// The model's file name for GET /props: never a directory (the option is set to
// the bare file name; anything before a path separator is dropped regardless).
std::string_view modelFileName(const EngineFacts& o) {
    std::string_view f = o.model_file.empty() ? std::string_view(o.model_name) : std::string_view(o.model_file);
    if (const std::size_t sep = f.find_last_of("/\\:"); sep != std::string_view::npos) f = f.substr(sep + 1);
    return f;
}

// GET /props: the subset of llama.cpp server's /props that clients read to
// auto-detect the context window and slot count (written from the documented
// JSON shape). No chat_template: WHIRL renders built-in templates, not Jinja.
std::string propsJson(EngineApi& e, const EngineFacts& o) {
    const EngineApi::Health hs = e.health();
    std::string b = "{\"default_generation_settings\":{\"id\":0,\"n_ctx\":";
    // (v0.2.2.2: the context per request for new requests; the memory governor lowers it while the WDDM budget is low)
    b += std::to_string(e.ctxPublished());
    b += ",\"model\":";
    appendJsonStr(b, o.model_name);
    b += std::format(",\"speculative\":{},\"is_processing\":{}}}", o.use_mtp ? "true" : "false",
                     hs.active > 0 ? "true" : "false");
    b += std::format(",\"total_slots\":{},\"model_path\":", hs.slots);
    appendJsonStr(b, modelFileName(o));
    b += ",\"model_alias\":";
    appendJsonStr(b, o.model_name);
    b += std::format(",\"modalities\":{{\"vision\":{},\"audio\":false}},\"build_info\":\"whirl {}\"",
                     o.vision.vis ? "true" : "false", WHIRL_VERSION_STRING);
    if (!o.numerics_json.empty()) b += ",\"numerics\":" + o.numerics_json;
    if (!o.startup_json.empty()) b += ",\"startup\":" + o.startup_json;
    if (const std::string gj = e.governorJson(); !gj.empty()) b += ",\"governor\":" + gj;
    const EngineApi::Stats st = e.statsShared();
    b += ",\"kv_ssd\":{\"mode\":";
    appendJsonStr(b, o.kv_ssd_mode);
    b += std::format(",\"bytes_written\":{},\"bytes_shared\":{},\"bytes_used\":{},\"entries\":{},\"superseded\":{}}}",
                     st.ssd_written, st.ssd_dedup, st.ssd_used, st.ssd_entries, st.superseded);
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
    static whirl::Mutex mu;
    static std::set<std::string, std::less<>> seen;
    whirl::MutexLock lk(mu);
    return seen.emplace(path).second;
}

// Per-connection context from the server.
struct ConnCtx {
    EngineApi& e;
    const EngineFacts& f;
    const HttpOptions& o;
    std::string_view bound;  // the listening address
    std::function<void()> on_read;  // the request has been read (stop() no longer cuts it)
    bool* framed = nullptr;         // out: the response had a Content-Length (SocketConn framed_out)
};

void handleConn(const ConnCtx& cx, net::Socket s) {
    EngineApi& e = cx.e;
    const EngineFacts& f = cx.f;
    e.beginBuilding();
    bool building = true;
    struct Guard {
        EngineApi& e;
        bool& b;
        ~Guard() {
            if (b) e.endBuilding();
        }
    } guard{e, building};
    Reader r(s, cx.o.recv_timeout_ms);
    r.setDeadline(SteadyClock::now() + std::chrono::milliseconds(cx.o.header_timeout_ms));
    SocketConn w(s, cx.o.send_timeout_ms, cx.framed);
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
    w.setHttp10(nextTok() == "HTTP/1.0");
    std::string path(target.substr(0, target.find('?')));
    std::size_t content_len = 0;
    bool have_len = false;
    bool chunked = false;
    std::string origin;
    std::string host;
    bool have_origin = false;
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
            have_origin = true;
        } else if (ieq(name, "host")) {
            host.assign(val);
        }
    }
    r.setDeadline(std::nullopt);
    // FR-SRV P2-4: DNS rebinding (a page on evil.example whose name now resolves to 127.0.0.1 is
    // same-origin to itself): only the Host names this server answers to
    if (!hostAllowed(host, cx.o.allowed_hosts, cx.bound)) {
        logW("{} {} -> 403 (Host header not allowed: {})", method, logSafe(path), logSafe(host));
        return sendAll(w, errorResponse(403, "Host header not allowed (start the server with --allow-host NAME to answer this name)"));
    }
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
    // FR-SRV P2-4: a state-changing request from a web page whose Origin is not allowed is refused,
    // not just left without CORS headers (a "simple" text/plain POST needs no preflight and would
    // still run on the GPU). Requests without Origin (SDKs, curl, IDE agents) are not affected.
    if (method != "GET" && method != "HEAD" && have_origin && cors.empty()) {
        logW("{} {} -> 403 (cross-origin request from {} refused; see --cors-origin)", method, logSafe(path), logSafe(origin));
        return sendAll(w, errorResponse(403, "cross-origin request refused: this web page's Origin is not allowed (see --cors-origin)"));
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
        const EngineApi::Health hs = e.health();
        // WDDM-LOG: this process's video memory at the last sample (Windows; absent without WDDM numbers)
        const std::string vram =
            hs.vram_ok ? std::format(",\"vram\":{{\"local_mib\":{},\"budget_mib\":{},\"shared_mib\":{},\"spill\":{}}}",
                                     hs.vram_local >> 20, hs.vram_budget >> 20, hs.vram_shared >> 20, hs.vram_spill ? "true" : "false")
                       : std::string();
        // v0.2.2.1: effective values after the startup fit (ctx per request, pool, slots, RAM tier, margin, adjustments)
        const std::string& su = f.startup_json;
        // v0.2.2.2: the memory governor's effective values (absent when it is off)
        const std::string gj = e.governorJson();
        const std::string b = std::format("{{\"status\":\"ok\",\"busy\":{},\"slots_busy\":{},\"slots\":{},\"queued\":{}{}{}{}}}",
                                          hs.active > 0 ? "true" : "false", hs.active, hs.slots, hs.queued, vram,
                                          su.empty() ? std::string() : ",\"startup\":" + su,
                                          gj.empty() ? std::string() : ",\"governor\":" + gj);
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8", b));
    }
    if (path == "/v1/models" || path == "/models") {
        if (!is_get) return sendAll(w, errorResponse(405, "use GET"));
        std::string b = "{\"object\":\"list\",\"data\":[{\"id\":";
        appendJsonStr(b, f.model_name);
        b += std::format(
            ",\"object\":\"model\",\"created\":{},\"owned_by\":\"whirl\",\"meta\":{{\"arch\":\"{}\",\"n_ctx\":{},\"n_vocab\":{},"
            "\"n_embd\":{},\"n_layer\":{},\"mtp\":{}}}}}]}}",
            static_cast<long long>(std::time(nullptr)), f.arch, e.ctxPublished(), f.n_vocab, f.n_embd,
            f.n_layer, f.use_mtp ? "true" : "false");
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8", b));
    }
    if (path == "/props" || path == "/v1/props") {
        if (!is_get) return sendAll(w, errorResponse(405, "use GET"));
        return sendAll(w, httpResponse(200, "application/json; charset=utf-8", propsJson(e, f)));
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
        logW("{} {} -> 404", logSafe(method), logSafe(path));
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
        buildJob(f, job, path, body, have_origin && !isLoopbackOrigin(origin));
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
    w.finish();
    w.flush();
}

std::once_flag g_net_once;

}  // namespace

bool isLoopbackOrigin(std::string_view origin) {
    const std::string o = lowerAscii(origin);
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

bool hostAllowed(std::string_view host_in, const std::vector<std::string>& extra, std::string_view bound_in) {
    if (host_in.empty()) return true;
    for (const std::string& x : extra)
        if (x == "*") return true;
    const std::string h = lowerAscii(host_in);
    std::string_view name = h, port;
    if (name.starts_with("[")) {
        const std::size_t e = name.find(']');
        if (e == std::string_view::npos) return false;
        port = name.substr(e + 1);
        name = name.substr(0, e + 1);
    } else if (const std::size_t c = name.rfind(':'); c != std::string_view::npos) {
        port = name.substr(c);
        name = name.substr(0, c);
    }
    if (!port.empty()) {
        if (port.front() != ':' || port.size() < 2 || port.size() > 6) return false;
        for (char ch : port.substr(1))
            if (ch < '0' || ch > '9') return false;
    }
    if (name.ends_with(".")) name.remove_suffix(1);  // "localhost." is the same name
    if (name.empty()) return false;
    if (name == "localhost" || name == "127.0.0.1" || name == "[::1]") return true;
    std::string bound = lowerAscii(bound_in);
    if (bound.find(':') != std::string::npos && !bound.starts_with("[")) bound = "[" + bound + "]";
    if (name == bound) return true;
    if (bound == "0.0.0.0" || bound == "[::]") {
        // every interface: any IP literal (a DNS name is what rebinding needs)
        if (name.starts_with("[")) return true;
        int dots = 0;
        bool digits = true;
        for (char ch : name) {
            if (ch == '.') ++dots;
            else if (ch < '0' || ch > '9') digits = false;
        }
        if (digits && dots == 3) return true;
    }
    for (const std::string& x : extra) {
        std::string_view xv = x;
        while (!xv.empty() && xv.back() == '.') xv.remove_suffix(1);
        if (lowerAscii(xv) == name) return true;
    }
    return false;
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
        const std::string lo = lowerAscii(origin);
        for (const std::string& x : extra) {
            std::string_view xv = x;
            while (!xv.empty() && xv.back() == '/') xv.remove_suffix(1);
            if (lowerAscii(xv) == lo) {
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
        if (!net::startup()) throw std::runtime_error("WSAStartup failed");
    });
}

HttpServer::HttpServer(EngineApi& e, HttpOptions o) : e_(e), facts_(e.facts()), o_(std::move(o)) { netInit(); }

HttpServer::~HttpServer() { stop(); }

void HttpServer::start(const std::string& host, std::uint16_t port) {
    bound_ = host;
    const std::string ps = std::to_string(port);
    const net::Addr* res = net::resolve(host.c_str(), ps.c_str(), net::Lookup::passive_numeric);
    if (!res) throw std::runtime_error("invalid --host " + host);
    const net::Socket s = net::open(res);
    if (s == net::kInvalid) {
        net::freeAddr(res);
        throw std::runtime_error("socket() failed");
    }
    // exclusive: with SO_REUSEADDR a second server could bind the same port and
    // silently share (steal) its connections; restarting on a port whose old
    // connections are in TIME_WAIT still works
    net::setExclusiveAddrUse(s);
    if (net::bind(s, res) != 0 || net::listen(s, net::kMaxBacklog) != 0) {
        const int err = net::lastError();
        net::freeAddr(res);
        net::close(s);
        throw std::runtime_error(std::format("cannot listen on {}:{} (WSA error {})", host, port, err));
    }
    net::freeAddr(res);
    net::localPort(s, &port_);
    listen_ = s;
    accept_ = std::thread([this] { acceptLoop(); });
}

void HttpServer::stop() {
    if (listen_ == net::kInvalid) return;
    stopping_ = true;
    net::close(listen_);
    if (accept_.joinable()) accept_.join();
    listen_ = net::kInvalid;
    auto shutAll = [this](bool reading_only) {
        // shutdown (not closesocket) from this thread: the connection thread still
        // owns the handle and closes it; it leaves the sets before closing
        whirl::MutexLock lk(conns_mu_);
        // (shutdown alone does not end a recv that is already waiting: cancel it)
        for (net::Socket c : reading_only ? reading_ : conns_) {
            net::shutdown(c, net::Shut::both);
            net::cancelIo(c);
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
    // FR-SRV P3-1: connection threads use the Engine (submitAndWait answers 503 once it stopped):
    // the owner destroys it after stop() returns, so wait for them (shut down sockets end soon)
    if (n_conn_.load() > 0) {
        shutAll(false);
        waitConns(20000);
        if (n_conn_.load() > 0) logW("http: {} connection thread(s) still running at shutdown", n_conn_.load());
    }
}

void HttpServer::setReading(platform::net::Socket sock, bool on) {
    whirl::MutexLock lk(conns_mu_);
    if (on) reading_.insert(sock);
    else reading_.erase(sock);
}

void HttpServer::acceptLoop() {
    for (;;) {
        const net::Socket c = net::accept(listen_);
        if (c == net::kInvalid) {
            if (stopping_) return;
            logE("accept failed: WSA error {}", net::lastError());
            continue;
        }
        net::setRecvTimeout(c, o_.recv_timeout_ms);
        net::setSendTimeout(c, o_.send_timeout_ms);
        if (o_.sndbuf > 0) net::setSendBuf(c, o_.sndbuf);
        if (n_conn_.load() >= static_cast<int>(o_.max_conn)) {
            static std::atomic<std::uint64_t> n_rejected{0};
            if (n_rejected.fetch_add(1) % 100 == 0)
                logW("connection limit ({} open connections) reached: answering 503", o_.max_conn);
            net::setSendTimeout(c, 1000);
            const std::string r = errorResponse(503, "too many open connections; retry later");
            net::send(c, r.data(), static_cast<int>(r.size()));
            net::shutdown(c, net::Shut::send);
            lingerDrain(c, 50);
            net::close(c);
            continue;
        }
        n_conn_.fetch_add(1);
        {
            whirl::MutexLock lk(conns_mu_);
            conns_.insert(c);
            reading_.insert(c);
        }
        try {
            std::thread([this, c] { connThread(c); }).detach();
        } catch (const std::exception& ex) {
            logE("cannot spawn connection thread: {}", ex.what());
            {
                whirl::MutexLock lk(conns_mu_);
                conns_.erase(c);
                reading_.erase(c);
            }
            net::close(c);
            n_conn_.fetch_sub(1);
        }
    }
}

void HttpServer::connThread(platform::net::Socket sock) {
    const net::Socket s = sock;
    bool framed = false;
    try {
        const ConnCtx cx{e_, facts_, o_, bound_, [this, sock] { setReading(sock, false); }, &framed};
        handleConn(cx, s);
    } catch (const std::exception& ex) {
        logE("connection error: {}", ex.what());
    }
    // HTTP-DROP: the response is complete and the client can tell its end (Content-Length, or a
    // chunked SSE stream with its last chunk); give the client close_wait_ms to close first. The side that
    // closes first keeps the connection's TIME_WAIT (120 s). On the server side it broke the client's
    // next connect: Windows hands the same client port out again, and the server's TIME_WAIT silently
    // drops that SYN when the earlier request was large (tens of KB), so connect() hung for 21 s
    // (WSAETIMEDOUT). Kept on the client side, the TIME_WAIT keeps the OS from reusing that port for
    // this server. Clients that read up to the end of the response close at once (curl, browsers,
    // Python requests / httpx, the OpenAI SDKs); one that waits for our FIN gets it after close_wait_ms.
    bool peer_closed = false;
    if (framed && o_.close_wait_ms > 0 && !stopping_) {
        setReading(sock, true);  // an idle wait: stop() ends it at once
        peer_closed = lingerDrain(s, o_.close_wait_ms);
    }
    {
        whirl::MutexLock lk(conns_mu_);
        conns_.erase(sock);
        reading_.erase(sock);
    }
    net::shutdown(s, net::Shut::send);
    if (!peer_closed) lingerDrain(s, 500);
    net::close(s);
    n_conn_.fetch_sub(1);
}

}  // namespace whirl::server
