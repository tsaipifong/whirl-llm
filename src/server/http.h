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
#include <string>
#include <thread>

namespace whirl::server {

class HttpServer {
public:
    explicit HttpServer(Engine& e);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Binds and listens (throws std::runtime_error), then accepts on a thread.
    // port 0 picks a free port (see port()).
    void start(const std::string& host, std::uint16_t port);
    std::uint16_t port() const { return port_; }
    // Stops accepting and waits for the connection threads to finish.
    void stop();

private:
    void acceptLoop();
    void connThread(std::uintptr_t sock);

    Engine& e_;
    std::uintptr_t listen_ = ~std::uintptr_t(0);
    std::uint16_t port_ = 0;
    std::thread accept_;
    std::atomic<bool> stopping_{false};
    std::atomic<int> n_conn_{0};
};

// WSAStartup / WSACleanup (idempotent).
void netInit();

}  // namespace whirl::server
