// Reimplements the WHIRL Zig research prototype's server.zig (connThread, handleConn, buildJob,
// acceptLoop); sockets written new on Winsock.
// SPDX-License-Identifier: Apache-2.0

#include "http.h"

#include "log.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <chrono>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <string_view>

#pragma comment(lib, "ws2_32.lib")

namespace whirl::server {

namespace {

constexpr std::size_t kReadBuf = 1 << 16;

class SocketConn final : public Conn {
public:
    explicit SocketConn(SOCKET s) : s_(s) { buf_.reserve(16384); }
    bool write(std::string_view data) override {
        if (bad_) return false;
        buf_.append(data);
        if (buf_.size() >= 16384) return flush();
        return true;
    }
    bool flush() override {
        if (bad_) return false;
        std::size_t off = 0;
        while (off < buf_.size()) {
            const int n = ::send(s_, buf_.data() + off, static_cast<int>(std::min<std::size_t>(buf_.size() - off, 1 << 20)), 0);
            if (n <= 0) {
                bad_ = true;
                buf_.clear();
                return false;
            }
            off += static_cast<std::size_t>(n);
        }
        buf_.clear();
        return true;
    }

private:
    SOCKET s_;
    std::string buf_;
    bool bad_ = false;
};

// Buffered reader over a socket.
class Reader {
public:
    explicit Reader(SOCKET s) : s_(s) { buf_.resize(kReadBuf); }
    enum class Res { ok, eof, too_long, error };
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
                return eof_ ? Res::eof : Res::error;
            }
        }
    }
    bool read(std::string& out, std::size_t n) {
        out.clear();
        out.reserve(n);
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
        const int n = ::recv(s_, buf_.data() + end_, static_cast<int>(buf_.size() - end_), 0);
        if (n == 0) eof_ = true;
        if (n <= 0) return false;
        end_ += static_cast<std::size_t>(n);
        return true;
    }
    SOCKET s_;
    std::string buf_;
    std::size_t pos_ = 0, end_ = 0;
    bool eof_ = false;
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

void handleConn(Engine& e, SOCKET s) {
    e.beginBuilding();
    bool building = true;
    struct Guard {
        Engine& e;
        bool& b;
        ~Guard() {
            if (b) e.endBuilding();
        }
    } guard{e, building};
    Reader r(s);
    SocketConn w(s);
    std::string line0;
    if (r.line(line0) != Reader::Res::ok) return;
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
    bool chunked = false;
    std::string hl;
    for (;;) {
        const Reader::Res res = r.line(hl);
        if (res == Reader::Res::too_long) return sendAll(w, errorResponse(413, "header line too long"));
        if (res != Reader::Res::ok) return;
        const std::string_view h = trimWs(hl);
        if (h.empty()) break;
        const std::size_t colon = h.find(':');
        if (colon == std::string_view::npos) continue;
        const std::string_view name = trimWs(h.substr(0, colon));
        const std::string_view val = trimWs(h.substr(colon + 1));
        if (ieq(name, "content-length")) {
            std::size_t v = 0;
            bool ok = !val.empty();
            for (char c : val) {
                if (c < '0' || c > '9') {
                    ok = false;
                    break;
                }
                v = v * 10 + static_cast<std::size_t>(c - '0');
            }
            if (!ok) return sendAll(w, errorResponse(400, "bad Content-Length"));
            content_len = v;
        } else if (ieq(name, "transfer-encoding")) {
            if (icontains(val, "chunked")) chunked = true;
        }
    }
    if (method == "OPTIONS") {
        sendAll(w,
                "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, "
                "OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\nContent-Length: "
                "0\r\nConnection: close\r\n\r\n");
        return;
    }
    if (chunked) return sendAll(w, errorResponse(411, "chunked request bodies are not supported; send Content-Length"));
    if (content_len > (64u << 20)) return sendAll(w, errorResponse(413, "request body too large"));
    std::string body;
    if (content_len > 0 && !r.read(body, content_len)) return;

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
    const bool is_chat = path == "/v1/chat/completions" || path == "/chat/completions";
    const bool is_cmpl = path == "/v1/completions" || path == "/completions";
    if (!is_chat && !is_cmpl) {
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
    // enqueue and wait (the main thread runs the job and writes the response)
    building = false;
    e.submitAndWait(job);
}

std::once_flag g_net_once;

}  // namespace

void netInit() {
    std::call_once(g_net_once, [] {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("WSAStartup failed");
    });
}

HttpServer::HttpServer(Engine& e) : e_(e) { netInit(); }

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
    // connection threads finish on their own (their jobs are answered)
    for (int i = 0; i < 6000 && n_conn_.load() > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void HttpServer::acceptLoop() {
    for (;;) {
        const SOCKET c = accept(static_cast<SOCKET>(listen_), nullptr, nullptr);
        if (c == INVALID_SOCKET) {
            if (stopping_) return;
            logE("accept failed: WSA error {}", WSAGetLastError());
            continue;
        }
        n_conn_.fetch_add(1);
        try {
            std::thread([this, c] { connThread(static_cast<std::uintptr_t>(c)); }).detach();
        } catch (const std::exception& ex) {
            logE("cannot spawn connection thread: {}", ex.what());
            closesocket(c);
            n_conn_.fetch_sub(1);
        }
    }
}

void HttpServer::connThread(std::uintptr_t sock) {
    const SOCKET s = static_cast<SOCKET>(sock);
    try {
        handleConn(e_, s);
    } catch (const std::exception& ex) {
        logE("connection error: {}", ex.what());
    }
    shutdown(s, SD_SEND);
    closesocket(s);
    n_conn_.fetch_sub(1);
}

}  // namespace whirl::server
