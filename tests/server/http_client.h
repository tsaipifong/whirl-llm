// Minimal blocking HTTP/1.1 client for the server tests (Connection: close;
// the whole response is read until the server closes the socket).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::test {

struct HttpResult {
    int status = 0;          // 0: connection failed
    std::string headers;
    std::string body;        // de-chunked if needed (the server never chunks)
    double ms = 0;           // wall time of the request
    double first_byte_ms = 0;
};

HttpResult httpRequest(const std::string& host, std::uint16_t port, std::string_view method, std::string_view path,
                       std::string_view body = {}, double timeout_s = 600, std::string_view extra_headers = {});

// Raw sockets for the connection-handling tests (idle / slow / non-reading
// clients). rawConnect returns ~0 on failure; rcvbuf > 0 sets SO_RCVBUF before
// connecting.
constexpr std::uintptr_t kNoSock = ~std::uintptr_t(0);
std::uintptr_t rawConnect(const std::string& host, std::uint16_t port, int rcvbuf = 0);
bool rawSend(std::uintptr_t s, std::string_view data);
// Reads until the peer closes, an error, or timeout_s; *closed: the server
// closed / reset the connection (false: the client-side timeout expired).
std::string rawRecvAll(std::uintptr_t s, double timeout_s, bool* closed = nullptr);
void rawClose(std::uintptr_t s);

// "data: {...}" payloads of an SSE body, in order ("[DONE]" included).
std::vector<std::string> sseEvents(std::string_view body);

}  // namespace whirl::test
