// SPDX-License-Identifier: Apache-2.0

#include "http_client.h"

#include "platform/net.h"

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <mutex>
#include <string_view>
#include <vector>

namespace whirl::test {

namespace {
namespace net = whirl::platform::net;
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

int watchedConnect(net::Socket s, const net::Addr* addr, std::uint16_t port, std::string& diag) {
    net::setNonBlocking(s, true);
    const auto t0 = std::chrono::steady_clock::now();
    int err = 0;
    if (net::connect(s, addr) != 0) {
        err = net::lastError();
        if (err == net::kErrWouldBlock) {
            err = 0;
            bool looked = false;
            for (;;) {
                bool wr = false, ex = false;
                const int n = net::waitConnect(s, 0, 500000, &wr, &ex);
                if (n == net::kError) {
                    err = net::lastError();
                    break;
                }
                if (n > 0) {
                    if (ex) {
                        const int so = net::socketError(s);
                        err = so != 0 ? so : net::kErrConnRefused;
                    }
                    break;
                }
                if (!looked && msSince(t0) >= 4000) {
                    looked = true;
                    std::uint16_t lp = 0;
                    net::localPort(s, &lp);
                    diag += std::format("\n      pending {:.0f} ms (local port {}): TCP rows of this port and of server port {}:",
                                        msSince(t0), lp, port);
                    diag += netstatRows({std::format(":{} ", lp), std::format(":{} ", port)}, true);
                    // a second connection while the first one waits: does the listener answer at all?
                    const auto t1 = std::chrono::steady_clock::now();
                    const net::Socket p2 = net::openTcp(addr);
                    net::setNonBlocking(p2, true);
                    net::connect(p2, addr);
                    bool wr2 = false, ex2 = false;
                    const int n2 = net::waitConnect(p2, 3, 0, &wr2, &ex2);
                    std::uint16_t lp2 = 0;
                    net::localPort(p2, &lp2);
                    diag += std::format("\n      second connection from port {}: {} after {:.1f} ms", lp2,
                                        n2 > 0 && wr2 ? "accepted" : n2 > 0 ? "refused" : "no answer in 3 s",
                                        msSince(t1));
                    net::close(p2);  // never sends a request: the server sees a client that closed at once
                }
            }
            if (looked) diag += std::format("\n      first connection {} after {:.0f} ms", err == 0 ? "completed" : "failed", msSince(t0));
        }
    }
    net::setNonBlocking(s, false);
    return err;
}
}  // namespace

HttpResult httpRequest(const std::string& host, std::uint16_t port, std::string_view method, std::string_view path,
                       std::string_view body, double timeout_s, std::string_view extra_headers) {
    return httpRequestFrom(0, host, port, method, path, body, timeout_s, extra_headers);
}

HttpResult httpRequestFrom(std::uint16_t local_port, const std::string& host, std::uint16_t port, std::string_view method,
                           std::string_view path, std::string_view body, double timeout_s, std::string_view extra_headers) {
    std::call_once(g_once, [] { net::startup(); });
    HttpResult r;
    const auto t0 = std::chrono::steady_clock::now();
    auto fail = [&](const char* step) {
        r.err = std::format("{} failed, WSA error {}", step, net::lastError());
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    const net::Addr* res = net::resolve(host.c_str(), std::to_string(port).c_str(), net::Lookup::client_v4);
    if (!res) {
        fail("getaddrinfo");
        return r;
    }
    const net::Socket s = net::open(res);
    if (s == net::kInvalid) {
        fail("socket");
        net::freeAddr(res);
        return r;
    }
    net::setRecvTimeout(s, static_cast<std::uint32_t>(timeout_s * 1000));
    if (local_port != 0) {
        // HTTP-DROP: a fixed local port, as the OS picks it again for a later connection (Windows reuses
        // ephemeral ports whose old connection is in TIME_WAIT on the server side)
        if (net::bindLoopback4(s, local_port) != 0) {
            fail("bind");
            net::freeAddr(res);
            net::close(s);
            return r;
        }
    }
    // HTTP-DROP: connect without blocking so a stalled handshake can be looked at while it is
    // pending (the outcome is the same: the OS still decides when it fails, 21 s by default)
    std::string pending_diag;
    const int cerr = watchedConnect(s, res, port, pending_diag);
    if (cerr != 0) {
        net::setLastError(cerr);
        fail("connect");
        // CB-FLAKE: the local port of the failed attempt and every TCP row that names it (a TIME_WAIT
        // of an earlier connection on the same 4-tuple would explain an unanswered SYN)
        std::uint16_t lp16 = 0;
        if (net::localPort(s, &lp16) == 0 && lp16 != 0) {
            const unsigned lp = lp16;
            r.err += std::format(", local port {} (previous connection's local port {})", lp, g_last_local_port.load());
            r.err += netstatRows({std::format(":{} ", lp)});
        }
        r.err += pending_diag;
        net::freeAddr(res);
        net::close(s);
        return r;
    }
    {
        std::uint16_t lp16 = 0;
        if (net::localPort(s, &lp16) == 0) g_last_local_port = lp16;
    }
    if (!pending_diag.empty()) std::fprintf(stderr, "[http_client] slow connect to port %u:%s\n", port, pending_diag.c_str());
    net::freeAddr(res);
    // extra_headers: complete lines, each ending in CRLF
    std::string req = std::format("{} {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: "
                                  "{}\r\n{}Connection: close\r\n\r\n",
                                  method, path, host, port, body.size(), extra_headers);
    req += body;
    std::size_t off = 0;
    while (off < req.size()) {
        const int n = net::send(s, req.data() + off, static_cast<int>(req.size() - off));
        if (n <= 0) {
            r.err = std::format("send failed after {} of {} bytes, WSA error {}", off, req.size(), net::lastError());
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
        const int n = net::recv(s, buf, sizeof buf, 0);
        if (n < 0 && r.err.empty()) r.err = std::format("recv failed after {} bytes, WSA error {}", resp.size(), net::lastError());
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
    net::close(s);
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
    std::call_once(g_once, [] { net::startup(); });
    const net::Socket s = net::openTcp4();
    std::uint16_t p = 0, got = 0;
    if (net::bindLoopback4(s, 0) == 0 && net::localPort(s, &got) == 0) p = got;
    net::close(s);
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
    std::call_once(g_once, [] { net::startup(); });
    const net::Addr* res = net::resolve(host.c_str(), std::to_string(port).c_str(), net::Lookup::client_v4);
    if (!res) return kNoSock;
    const net::Socket s = net::open(res);
    if (s == net::kInvalid) {
        net::freeAddr(res);
        return kNoSock;
    }
    if (rcvbuf > 0) net::setRecvBuf(s, rcvbuf);
    const int rc = net::connect(s, res);
    net::freeAddr(res);
    if (rc != 0) {
        net::close(s);
        return kNoSock;
    }
    return s;
}

bool rawSend(std::uintptr_t s, std::string_view data) {
    std::size_t off = 0;
    while (off < data.size()) {
        const int n = net::send(s, data.data() + off, static_cast<int>(data.size() - off));
        if (n <= 0) return false;
        off += static_cast<std::size_t>(n);
    }
    return true;
}

std::string rawRecvAll(std::uintptr_t s, double timeout_s, bool* closed) {
    net::setRecvTimeout(s, static_cast<std::uint32_t>(timeout_s * 1000));
    std::string out;
    char buf[65536];
    if (closed) *closed = false;
    for (;;) {
        const int n = net::recv(s, buf, sizeof buf, 0);
        if (n > 0) {
            out.append(buf, static_cast<std::size_t>(n));
            continue;
        }
        if (closed) *closed = n == 0 || net::lastError() != net::kErrTimedOut;
        return out;
    }
}

void rawClose(std::uintptr_t s) {
    if (s != kNoSock) net::close(s);
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
