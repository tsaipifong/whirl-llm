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
                       std::string_view body = {}, double timeout_s = 600);

// "data: {...}" payloads of an SSE body, in order ("[DONE]" included).
std::vector<std::string> sseEvents(std::string_view body);

}  // namespace whirl::test
