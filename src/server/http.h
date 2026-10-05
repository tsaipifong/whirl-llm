// HTTP/1.1 front end of the server (one thread per connection, Connection:
// close): /health, /v1/models, /v1/chat/completions, /v1/completions, the
// read-only compatibility endpoints /props and /version, and CORS preflight. Requests are parsed, rendered and tokenized here; the
// engine's main thread runs them and writes the responses.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
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
    int sndbuf = 0;                           // test hook: SO_SNDBUF of accepted sockets (0 = system default)
};

// CORS headers for a request's Origin header value ("" = no CORS headers);
// each line ends in CRLF.
std::string corsHeaders(std::string_view origin, const std::vector<std::string>& extra);
// http(s)://localhost, 127.0.0.1 or [::1] with an optional port.
bool isLoopbackOrigin(std::string_view origin);

class HttpServer {
public:
    explicit HttpServer(Engine& e, HttpOptions o = {});
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
    void connThread(std::uintptr_t sock);
    void setReading(std::uintptr_t sock, bool on);

    Engine& e_;
    HttpOptions o_;
    std::mutex conns_mu_;
    std::set<std::uintptr_t> conns_;    // open connection sockets
    std::set<std::uintptr_t> reading_;  // ... of which still reading their request
    std::uintptr_t listen_ = ~std::uintptr_t(0);
    std::uint16_t port_ = 0;
    std::thread accept_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> n_conn_{0};
};

// WSAStartup / WSACleanup (idempotent).
void netInit();

}  // namespace whirl::server
