// SPDX-License-Identifier: Apache-2.0

#include "http_client.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <mutex>
#include <string_view>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

namespace whirl::test {

namespace {
std::once_flag g_once;
// local port of the last connection this process made (any thread): a failed connect says whether the OS
// gave it the port of the connection just before (HTTP-DROP)
std::atomic<unsigned> g_last_local_port{0};

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// netstat rows (TCP) containing any of the keys, each on its own "netstat:" line
std::string netstatRows(const std::vector<std::string>& keys, bool skip_time_wait = false) {
    std::string out;
    int tw = 0;
    if (FILE* p = _popen("netstat -ano -p tcp", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, p)) {
            const std::string_view lv(line);
            bool hit = false;
            for (const auto& k : keys) hit = hit || lv.find(k) != std::string_view::npos;
            if (!hit) continue;
            if (skip_time_wait && lv.find("TIME_WAIT") != std::string_view::npos) {
                ++tw;
                continue;
            }
            std::string l(line);
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            out += "\n      netstat: " + l;
        }
        _pclose(p);
    }
    if (skip_time_wait) out += std::format("\n      netstat: {} TIME_WAIT rows not shown", tw);
    return out;
}

// HTTP-DROP: connect(), but non-blocking with a wake-up every 500 ms. If the handshake is still
// pending after 4 s (a loopback SYN-ACK normally takes well under 1 ms; a refused connect ends
// within ~2 s; at 4 s the first SYN and its retransmission went unanswered), it records once what
// the TCP table shows for this local port and the server port, and whether a second, fresh
// connection to the same server is accepted meanwhile. Returns 0 or the WSA error of the attempt.
// Chunked transfer coding (RFC 9112 7.1): complete (last chunk and final CRLF seen), partial, or bad.
enum class Dechunk { complete, partial, bad };
Dechunk dechunk(std::string_view b, std::string* out) {
    std::size_t p = 0;
    for (;;) {
        const std::size_t eol = b.find("\r\n", p);
        if (eol == std::string_view::npos) return Dechunk::partial;
        std::size_t len = 0, i = p;
        for (; i < eol && std::isxdigit(static_cast<unsigned char>(b[i])); ++i)
            len = len * 16 + static_cast<std::size_t>(std::isdigit(static_cast<unsigned char>(b[i])) ? b[i] - '0' : (b[i] | 0x20) - 'a' + 10);
        if (i == p || (i < eol && b[i] != ';')) return Dechunk::bad;
        p = eol + 2;
        if (len == 0) {  // last chunk; the server sends no trailers
            if (b.size() < p + 2) return Dechunk::partial;
            return b.substr(p, 2) == "\r\n" ? Dechunk::complete : Dechunk::bad;
        }
        if (b.size() < p + len + 2) return Dechunk::partial;
        if (b.substr(p + len, 2) != "\r\n") return Dechunk::bad;
        if (out) out->append(b.substr(p, len));
        p += len + 2;
    }
}

int watchedConnect(SOCKET s, const sockaddr* addr, int addrlen, std::uint16_t port, std::string& diag) {
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    const auto t0 = std::chrono::steady_clock::now();
    int err = 0;
    if (connect(s, addr, addrlen) != 0) {
        err = WSAGetLastError();
        if (err == WSAEWOULDBLOCK) {
            err = 0;
            bool looked = false;
            for (;;) {
                fd_set w, e;
                FD_ZERO(&w);
                FD_ZERO(&e);
                FD_SET(s, &w);
                FD_SET(s, &e);
                timeval tv{0, 500000};
                const int n = select(0, nullptr, &w, &e, &tv);
                if (n == SOCKET_ERROR) {
                    err = WSAGetLastError();
                    break;
                }
                if (n > 0) {
                    if (FD_ISSET(s, &e)) {
                        int so = 0, sl = sizeof so;
                        getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so), &sl);
                        err = so != 0 ? so : WSAECONNREFUSED;
                    }
                    break;
                }
                if (!looked && msSince(t0) >= 4000) {
                    looked = true;
                    sockaddr_in la{};
                    int ll = sizeof la;
                    getsockname(s, reinterpret_cast<sockaddr*>(&la), &ll);
                    diag += std::format("\n      pending {:.0f} ms (local port {}): TCP rows of this port and of server port {}:",
                                        msSince(t0), ntohs(la.sin_port), port);
                    diag += netstatRows({std::format(":{} ", ntohs(la.sin_port)), std::format(":{} ", port)}, true);
                    // a second connection while the first one waits: does the listener answer at all?
                    const auto t1 = std::chrono::steady_clock::now();
                    SOCKET p2 = socket(addr->sa_family, SOCK_STREAM, IPPROTO_TCP);
                    u_long nb2 = 1;
                    ioctlsocket(p2, FIONBIO, &nb2);
                    connect(p2, addr, addrlen);
                    fd_set w2, e2;
                    FD_ZERO(&w2);
                    FD_ZERO(&e2);
                    FD_SET(p2, &w2);
                    FD_SET(p2, &e2);
                    timeval tv2{3, 0};
                    const int n2 = select(0, nullptr, &w2, &e2, &tv2);
                    sockaddr_in la2{};
                    int ll2 = sizeof la2;
                    getsockname(p2, reinterpret_cast<sockaddr*>(&la2), &ll2);
                    diag += std::format("\n      second connection from port {}: {} after {:.1f} ms", ntohs(la2.sin_port),
                                        n2 > 0 && FD_ISSET(p2, &w2) ? "accepted" : n2 > 0 ? "refused" : "no answer in 3 s",
                                        msSince(t1));
                    closesocket(p2);  // never sends a request: the server sees a client that closed at once
                }
            }
            if (looked) diag += std::format("\n      first connection {} after {:.0f} ms", err == 0 ? "completed" : "failed", msSince(t0));
        }
    }
    nb = 0;
    ioctlsocket(s, FIONBIO, &nb);
    return err;
}
}  // namespace

HttpResult httpRequest(const std::string& host, std::uint16_t port, std::string_view method, std::string_view path,
                       std::string_view body, double timeout_s, std::string_view extra_headers) {
    return httpRequestFrom(0, host, port, method, path, body, timeout_s, extra_headers);
}

HttpResult httpRequestFrom(std::uint16_t local_port, const std::string& host, std::uint16_t port, std::string_view method,
                           std::string_view path, std::string_view body, double timeout_s, std::string_view extra_headers) {
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
    auto fail = [&](const char* step) {
        r.err = std::format("{} failed, WSA error {}", step, WSAGetLastError());
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        fail("getaddrinfo");
        return r;
    }
    SOCKET s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) {
        fail("socket");
        freeaddrinfo(res);
        return r;
    }
    const DWORD to = static_cast<DWORD>(timeout_s * 1000);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&to), sizeof to);
    if (local_port != 0) {
        // HTTP-DROP: a fixed local port, as the OS picks it again for a later connection (Windows reuses
        // ephemeral ports whose old connection is in TIME_WAIT on the server side)
        sockaddr_in la{};
        la.sin_family = AF_INET;
        la.sin_port = htons(local_port);
        inet_pton(AF_INET, "127.0.0.1", &la.sin_addr);
        if (bind(s, reinterpret_cast<const sockaddr*>(&la), sizeof la) != 0) {
            fail("bind");
            freeaddrinfo(res);
            closesocket(s);
            return r;
        }
    }
    // HTTP-DROP: connect without blocking so a stalled handshake can be looked at while it is
    // pending (the outcome is the same: the OS still decides when it fails, 21 s by default)
    std::string pending_diag;
    const int cerr = watchedConnect(s, res->ai_addr, static_cast<int>(res->ai_addrlen), port, pending_diag);
    if (cerr != 0) {
        WSASetLastError(cerr);
        fail("connect");
        // CB-FLAKE: the local port of the failed attempt and every TCP row that names it (a TIME_WAIT
        // of an earlier connection on the same 4-tuple would explain an unanswered SYN)
        sockaddr_in la{};
        int ll = sizeof la;
        if (getsockname(s, reinterpret_cast<sockaddr*>(&la), &ll) == 0 && ntohs(la.sin_port) != 0) {
            const unsigned lp = ntohs(la.sin_port);
            r.err += std::format(", local port {} (previous connection's local port {})", lp, g_last_local_port.load());
            r.err += netstatRows({std::format(":{} ", lp)});
        }
        r.err += pending_diag;
        freeaddrinfo(res);
        closesocket(s);
        return r;
    }
    {
        sockaddr_in la{};
        int ll = sizeof la;
        if (getsockname(s, reinterpret_cast<sockaddr*>(&la), &ll) == 0) g_last_local_port = ntohs(la.sin_port);
    }
    if (!pending_diag.empty()) std::fprintf(stderr, "[http_client] slow connect to port %u:%s\n", port, pending_diag.c_str());
    freeaddrinfo(res);
    // extra_headers: complete lines, each ending in CRLF
    std::string req = std::format("{} {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: "
                                  "{}\r\n{}Connection: close\r\n\r\n",
                                  method, path, host, port, body.size(), extra_headers);
    req += body;
    std::size_t off = 0;
    while (off < req.size()) {
        const int n = send(s, req.data() + off, static_cast<int>(req.size() - off), 0);
        if (n <= 0) {
            r.err = std::format("send failed after {} of {} bytes, WSA error {}", off, req.size(), WSAGetLastError());
            break;
        }
        off += static_cast<std::size_t>(n);
    }
    std::string resp;
    char buf[65536];
    bool first = true;
    // HTTP-DROP: like real clients (curl, browsers, Python requests / httpx), a response with a
    // Content-Length ends after that many body bytes and the client closes the connection then;
    // without one (an SSE stream) it ends when the server closes
    std::size_t want = std::string::npos;  // header + body bytes, once the header is complete
    bool header_seen = false, chunked = false;
    std::size_t body_at = 0;
    for (;;) {
        const int n = recv(s, buf, sizeof buf, 0);
        if (n < 0 && r.err.empty()) r.err = std::format("recv failed after {} bytes, WSA error {}", resp.size(), WSAGetLastError());
        if (n <= 0) break;
        if (first) {
            r.first_byte_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            first = false;
        }
        resp.append(buf, static_cast<std::size_t>(n));
        if (!header_seen) {
            const std::size_t he = resp.find("\r\n\r\n");
            if (he != std::string::npos) {
                header_seen = true;
                std::string h = resp.substr(0, he + 2);
                for (auto& ch : h) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                const std::size_t cl = h.find("\r\ncontent-length:");
                chunked = h.find("\r\ntransfer-encoding: chunked\r\n") != std::string::npos;
                body_at = he + 4;
                if (cl != std::string::npos && !chunked)
                    want = he + 4 + static_cast<std::size_t>(std::strtoull(h.c_str() + cl + 17, nullptr, 10));
            }
        }
        if (resp.size() >= want) break;
        if (chunked && dechunk(std::string_view(resp).substr(body_at), nullptr) == Dechunk::complete) break;
    }
    closesocket(s);
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::size_t he = resp.find("\r\n\r\n");
    if (he == std::string::npos) {
        if (r.err.empty()) r.err = std::format("no response header ({} bytes received)", resp.size());
        return r;
    }
    r.headers = resp.substr(0, he);
    r.body = resp.substr(he + 4);
    if (chunked) {  // the body without its chunk framing
        std::string plain;
        const Dechunk d = dechunk(r.body, &plain);
        if (d != Dechunk::complete && r.err.empty())
            r.err = d == Dechunk::bad ? "malformed chunked body" : "chunked body ended without its last chunk";
        r.body = std::move(plain);
    }
    if (r.headers.size() > 12 && r.headers.compare(0, 5, "HTTP/") == 0) r.status = std::atoi(r.headers.c_str() + 9);
    return r;
}

std::uint16_t freeLocalPort() {
    std::call_once(g_once, [] {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    });
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    std::uint16_t p = 0;
    int al = sizeof a;
    if (bind(s, reinterpret_cast<const sockaddr*>(&a), sizeof a) == 0 && getsockname(s, reinterpret_cast<sockaddr*>(&a), &al) == 0)
        p = ntohs(a.sin_port);
    closesocket(s);
    return p;
}

int tcpRowCount(std::uint16_t local_port, std::uint16_t remote_port, std::string_view state) {
    int n = 0;
    const std::string a = std::format("127.0.0.1:{} ", local_port), b = std::format("127.0.0.1:{} ", remote_port);
    if (FILE* p = _popen("netstat -ano -p tcp", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, p)) {
            const std::string_view lv(line);
            const std::size_t pa = lv.find(a);
            if (pa != std::string_view::npos && lv.find(b, pa + a.size()) != std::string_view::npos && lv.find(state) != std::string_view::npos)
                ++n;
        }
        _pclose(p);
    }
    return n;
}

std::uintptr_t rawConnect(const std::string& host, std::uint16_t port, int rcvbuf) {
    std::call_once(g_once, [] {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    });
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) return kNoSock;
    SOCKET s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) {
        freeaddrinfo(res);
        return kNoSock;
    }
    if (rcvbuf > 0) setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&rcvbuf), sizeof rcvbuf);
    const int rc = connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen));
    freeaddrinfo(res);
    if (rc != 0) {
        closesocket(s);
        return kNoSock;
    }
    return static_cast<std::uintptr_t>(s);
}

bool rawSend(std::uintptr_t s, std::string_view data) {
    std::size_t off = 0;
    while (off < data.size()) {
        const int n = send(static_cast<SOCKET>(s), data.data() + off, static_cast<int>(data.size() - off), 0);
        if (n <= 0) return false;
        off += static_cast<std::size_t>(n);
    }
    return true;
}

std::string rawRecvAll(std::uintptr_t s, double timeout_s, bool* closed) {
    const DWORD to = static_cast<DWORD>(timeout_s * 1000);
    setsockopt(static_cast<SOCKET>(s), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&to), sizeof to);
    std::string out;
    char buf[65536];
    if (closed) *closed = false;
    for (;;) {
        const int n = recv(static_cast<SOCKET>(s), buf, sizeof buf, 0);
        if (n > 0) {
            out.append(buf, static_cast<std::size_t>(n));
            continue;
        }
        if (closed) *closed = n == 0 || WSAGetLastError() != WSAETIMEDOUT;
        return out;
    }
}

void rawClose(std::uintptr_t s) {
    if (s != kNoSock) closesocket(static_cast<SOCKET>(s));
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
