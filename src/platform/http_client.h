// Platform layer: a minimal HTTP client for the launcher's chat window (feat026-launcher, 0.2.6). See
// docs/dev/platform.md and docs/dev/launcher.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): platform-free declarations; the Windows implementation
// is http_client_win.cpp (WinHTTP), chosen by cmake/platform.cmake. Only the launcher uses it (a request to the
// WHIRL server this launcher started, on this computer); the server's own Winsock code is net (PL3).
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace whirl::platform::http_client {

struct Result {
    bool ok = false;      // a response arrived (any status)
    int status = 0;       // HTTP status
    std::string body;     // the response body when on_data is not given
    std::string why;      // "" or what failed (connect, send, receive)
};

// GET http://host:port/path. timeout_ms bounds connect, send and every receive.
Result get(const std::string& host, std::uint16_t port, const std::string& path, unsigned timeout_ms);
// POST a JSON body; on_data receives the response body as it arrives (a streamed answer) and returns false to stop.
Result postJson(const std::string& host, std::uint16_t port, const std::string& path, const std::string& body,
                unsigned timeout_ms, const std::function<bool(const std::string&)>& on_data);

}  // namespace whirl::platform::http_client
