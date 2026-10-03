// SPDX-License-Identifier: Apache-2.0

#include "http_client.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <chrono>
#include <format>
#include <mutex>

#pragma comment(lib, "ws2_32.lib")

namespace whirl::test {

namespace {
std::once_flag g_once;
}

HttpResult httpRequest(const std::string& host, std::uint16_t port, std::string_view method, std::string_view path,
                       std::string_view body, double timeout_s) {
    std::call_once(g_once, [] {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    });
    HttpResult r;
    const auto t0 = std::chrono::steady_clock::now();
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) return r;
    SOCKET s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) {
        freeaddrinfo(res);
        return r;
    }
    const DWORD to = static_cast<DWORD>(timeout_s * 1000);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&to), sizeof to);
    if (connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0) {
        freeaddrinfo(res);
        closesocket(s);
        return r;
    }
    freeaddrinfo(res);
    std::string req = std::format("{} {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: "
                                  "{}\r\nConnection: close\r\n\r\n",
                                  method, path, host, port, body.size());
    req += body;
    std::size_t off = 0;
    while (off < req.size()) {
        const int n = send(s, req.data() + off, static_cast<int>(req.size() - off), 0);
        if (n <= 0) break;
        off += static_cast<std::size_t>(n);
    }
    std::string resp;
    char buf[65536];
    bool first = true;
    for (;;) {
        const int n = recv(s, buf, sizeof buf, 0);
        if (n <= 0) break;
        if (first) {
            r.first_byte_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            first = false;
        }
        resp.append(buf, static_cast<std::size_t>(n));
    }
    closesocket(s);
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::size_t he = resp.find("\r\n\r\n");
    if (he == std::string::npos) return r;
    r.headers = resp.substr(0, he);
    r.body = resp.substr(he + 4);
    if (r.headers.size() > 12 && r.headers.compare(0, 5, "HTTP/") == 0) r.status = std::atoi(r.headers.c_str() + 9);
    return r;
}

std::vector<std::string> sseEvents(std::string_view body) {
    std::vector<std::string> out;
    std::size_t p = 0;
    while (p < body.size()) {
        std::size_t e = body.find("\n\n", p);
        if (e == std::string_view::npos) e = body.size();
        std::string_view ev = body.substr(p, e - p);
        if (ev.substr(0, 6) == "data: ") out.emplace_back(ev.substr(6));
        p = e + 2;
    }
    return out;
}

}  // namespace whirl::test
