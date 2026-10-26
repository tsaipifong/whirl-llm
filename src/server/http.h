// HTTP/1.1 front end of the server (one thread per connection, Connection:
// close): /health, /v1/models, /v1/chat/completions, /v1/completions, the
// read-only compatibility endpoints /props and /version, and CORS preflight. Requests are parsed, rendered and tokenized here; the
// engine's main thread runs them and writes the responses.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_api.h"
#include "platform/net.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace whirl::server {

// Limits of the HTTP front end (the defaults are the server's; tests use small
// timeouts).
struct HttpOptions {
    std::uint32_t recv_timeout_ms = 30000;    // a request read stalls (no bytes) this long -> closed
    std::uint32_t header_timeout_ms = 60000;  // request line + headers must arrive within this -> else 408
    // a response the client does not read: bytes not accepted for this long -> the
    // request is dropped (its slot freed); the engine thread never blocks on a client
    std::uint32_t send_timeout_ms = 30000;
    std::uint32_t max_conn = 64;              // open connections; more -> 503 and closed
    std::size_t max_header_bytes = 64 << 10;  // request line + headers in total -> else 431
    std::uint32_t max_header_lines = 100;     // header lines -> else 431
    // CORS: browser Origins allowed besides http(s)://localhost / 127.0.0.1 / [::1]
    // (any port); "*" = every Origin (Access-Control-Allow-Origin: *)
    std::vector<std::string> cors_origins;
    // Host header names allowed besides localhost / 127.0.0.1 / [::1] and the bound address
    // (DNS rebinding guard; "*" = any Host)
    std::vector<std::string> allowed_hosts;
    int sndbuf = 0;                           // test hook: SO_SNDBUF of accepted sockets (0 = system default)
    // after a response: wait this long for the client to close first, then close (HTTP-DROP: a
    // server-side TIME_WAIT can silently drop the client's next SYN from the same port; 0 = close at once)
    std::uint32_t close_wait_ms = 200;
};

// CORS headers for a request's Origin header value ("" = no CORS headers);
// each line ends in CRLF.
std::string corsHeaders(std::string_view origin, const std::vector<std::string>& extra);
// http(s)://localhost, 127.0.0.1 or [::1] with an optional port.
bool isLoopbackOrigin(std::string_view origin);
// FR-SRV P2-4 (DNS rebinding): a Host header value the server answers. Allowed: localhost,
// 127.0.0.1, [::1], the bound address (any IP literal when bound to 0.0.0.0 / ::), the names in
// `extra` ("*" = any); an optional :port; no Host header (non-browser HTTP/1.0 clients).
bool hostAllowed(std::string_view host, const std::vector<std::string>& extra, std::string_view bound);

// A chat request's prompt, as the connection thread builds it (arch024 A7: the prompt token pin calls this):
// image / video parts checked (media: an mmproj is loaded), the template rendered (image parts as the Qwen
// placeholder when media), then encoded with special tokens. Throws RequestError.
struct ChatPrompt {
    std::string text;
    std::vector<TokenId> ids;
};
ChatPrompt chatPrompt(const Tokenizer& tok, chat::TemplateKind tmpl, chat::ChatRequest& req, bool media);
// The token ids of a /v1/completions request's "prompt": a string, a list with one string, or a list of token ids
// (each in [0, n_vocab)). Throws RequestError.
std::vector<std::uint32_t> completionTokens(const Tokenizer& tok, std::uint32_t n_vocab, const json::Object& obj);

class HttpServer {
public:
    // Takes e.facts() once (every startup setter of the engine has run before).
    explicit HttpServer(EngineApi& e, HttpOptions o = {});
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Binds and listens (throws std::runtime_error), then accepts on a thread.
    // port 0 picks a free port (see port()).
    void start(const std::string& host, std::uint16_t port);
    std::uint16_t port() const { return port_; }
    // Stops accepting, closes connections that are still reading a request and
    // waits for the connection threads to finish (connections whose request is
    // running get a few seconds before they are shut down too).
    void stop();
    const HttpOptions& options() const { return o_; }

private:
    void acceptLoop();
    void connThread(platform::net::Socket sock);
    void setReading(platform::net::Socket sock, bool on);

    EngineApi& e_;
    const EngineFacts facts_;
    HttpOptions o_;
    std::string bound_;  // the --host address (Host header allow-list)
    std::mutex conns_mu_;
    std::set<platform::net::Socket> conns_;    // open connection sockets
    std::set<platform::net::Socket> reading_;  // ... of which still reading their request
    platform::net::Socket listen_ = platform::net::kInvalid;
    std::uint16_t port_ = 0;
    std::thread accept_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> n_conn_{0};
};

// Starts Winsock once (platform::net::startup; throws if it fails). Idempotent.
void netInit();

}  // namespace whirl::server
