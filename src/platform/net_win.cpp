// Platform layer, sockets: the Windows (Winsock 2) implementation of platform/net.h and FakeNet (net_fake.h).
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL3: the Winsock calls of src/server/http.cpp, src/release/release.cpp (port pre-check) and
// tests/server/http_client.cpp moved here one function per call, each with the arguments it had in v0.2.4 0e35407.

#include "platform/net.h"
#include "platform/net_fake.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <format>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

static_assert(sizeof(SOCKET) == sizeof(whirl::platform::net::Socket));
static_assert(INVALID_SOCKET == whirl::platform::net::kInvalid);
static_assert(SOCKET_ERROR == whirl::platform::net::kError);
static_assert(WSAEACCES == whirl::platform::net::kErrAccess);
static_assert(WSAEWOULDBLOCK == whirl::platform::net::kErrWouldBlock);
static_assert(WSAEADDRINUSE == whirl::platform::net::kErrAddrInUse);
static_assert(WSAEADDRNOTAVAIL == whirl::platform::net::kErrAddrNotAvail);
static_assert(WSAECONNRESET == whirl::platform::net::kErrConnReset);
static_assert(WSAENOBUFS == whirl::platform::net::kErrNoBufs);
static_assert(WSAETIMEDOUT == whirl::platform::net::kErrTimedOut);
static_assert(WSAECONNREFUSED == whirl::platform::net::kErrConnRefused);
static_assert(SOMAXCONN == whirl::platform::net::kMaxBacklog);
static_assert(MSG_PEEK == whirl::platform::net::kPeek);
static_assert(SD_SEND == static_cast<int>(whirl::platform::net::Shut::send));
static_assert(SD_BOTH == static_cast<int>(whirl::platform::net::Shut::both));

namespace whirl::platform::net {

namespace {

const addrinfo* ai(const Addr* a) { return reinterpret_cast<const addrinfo*>(a); }
SOCKET so(Socket s) { return static_cast<SOCKET>(s); }

// ---- FakeNet state (off unless a test called fake::begin)

std::atomic<bool> g_on{false};
thread_local bool t_last_tracked = false;  // the last net call of this thread was a tracked one

struct FakeState {
    std::mutex mu;
    std::set<const Addr*> passive;      // live passive_numeric lookups
    std::map<Socket, std::string> tag;  // tracked sockets
    int n_listen = 0, n_conn = 0;
    std::vector<fake::Call> calls;
    int fail_accept = 0, fail_accept_err = 0;
    int fail_recv = 0, fail_recv_err = 0;
};
FakeState& fstate() {
    static FakeState s;
    return s;
}

// Records a call on socket s if it is tracked (lock held by the caller). Keeps the thread's Winsock error.
bool recLocked(FakeState& f, Socket s, std::string text) {
    const auto it = f.tag.find(s);
    t_last_tracked = it != f.tag.end();
    if (!t_last_tracked) return false;
    f.calls.push_back({it->second, std::move(text)});
    return true;
}
void rec(Socket s, std::string text) {
    if (!g_on.load(std::memory_order_relaxed)) return;
    const int e = WSAGetLastError();
    FakeState& f = fstate();
    {
        std::lock_guard<std::mutex> lk(f.mu);
        recLocked(f, s, std::move(text));
    }
    WSASetLastError(e);
}
void recPlain(std::string text) {  // a call on no socket ("-")
    const int e = WSAGetLastError();
    FakeState& f = fstate();
    {
        std::lock_guard<std::mutex> lk(f.mu);
        f.calls.push_back({"-", std::move(text)});
        t_last_tracked = true;
    }
    WSASetLastError(e);
}

int setOpt(Socket s, int name, const void* v, int len, long long shown) {
    rec(s, std::format("setsockopt({},{},{},{})", SOL_SOCKET, name, len, shown));
    return setsockopt(so(s), SOL_SOCKET, name, static_cast<const char*>(v), len);
}

}  // namespace

bool startup() {
    if (g_on.load(std::memory_order_relaxed)) recPlain("WSAStartup(514)");
    WSADATA d;
    return WSAStartup(MAKEWORD(2, 2), &d) == 0;
}

void cleanup() {
    if (g_on.load(std::memory_order_relaxed)) recPlain("WSACleanup()");
    WSACleanup();
}

int lastError() {
    const int e = WSAGetLastError();
    if (g_on.load(std::memory_order_relaxed) && t_last_tracked) recPlain(std::format("WSAGetLastError()={}", e));
    return e;
}

void setLastError(int e) { WSASetLastError(e); }

const Addr* resolve(const char* host, const char* port, Lookup how) {
    addrinfo hints{};
    if (how == Lookup::passive_numeric) {
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE | AI_NUMERICHOST;
    } else {
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
    }
    const bool passive = how == Lookup::passive_numeric && g_on.load(std::memory_order_relaxed);
    if (passive) recPlain(std::format("getaddrinfo({},{},{})", hints.ai_family, hints.ai_socktype, hints.ai_flags));
    else t_last_tracked = false;
    addrinfo* res = nullptr;
    if (getaddrinfo(host, port, &hints, &res) != 0 || !res) return nullptr;
    const Addr* a = reinterpret_cast<const Addr*>(res);
    if (passive) {
        const int e = WSAGetLastError();
        FakeState& f = fstate();
        {
            std::lock_guard<std::mutex> lk(f.mu);
            f.passive.insert(a);
        }
        WSASetLastError(e);
    }
    return a;
}

void freeAddr(const Addr* a) {
    if (g_on.load(std::memory_order_relaxed)) {
        FakeState& f = fstate();
        bool was = false;
        {
            std::lock_guard<std::mutex> lk(f.mu);
            was = f.passive.erase(a) > 0;
        }
        if (was) recPlain("freeaddrinfo()");
        else t_last_tracked = false;
    }
    freeaddrinfo(const_cast<addrinfo*>(ai(a)));
}

Socket open(const Addr* a) {
    const addrinfo* r = ai(a);
    const SOCKET s = socket(r->ai_family, r->ai_socktype, r->ai_protocol);
    if (g_on.load(std::memory_order_relaxed)) {
        const int e = WSAGetLastError();
        FakeState& f = fstate();
        {
            std::lock_guard<std::mutex> lk(f.mu);
            t_last_tracked = false;
            if (f.passive.count(a) && s != INVALID_SOCKET) {
                f.tag[static_cast<Socket>(s)] = std::format("L{}", ++f.n_listen);
                recLocked(f, static_cast<Socket>(s), std::format("socket({},{},{})", r->ai_family, r->ai_socktype, r->ai_protocol));
            }
        }
        WSASetLastError(e);
    }
    return static_cast<Socket>(s);
}

Socket openTcp(const Addr* a) {
    t_last_tracked = false;
    return static_cast<Socket>(socket(ai(a)->ai_addr->sa_family, SOCK_STREAM, IPPROTO_TCP));
}

Socket openTcp4() {
    t_last_tracked = false;
    return static_cast<Socket>(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
}

int bind(Socket s, const Addr* a) {
    rec(s, "bind()");
    return ::bind(so(s), ai(a)->ai_addr, static_cast<int>(ai(a)->ai_addrlen));
}

int bindLoopback4(Socket s, std::uint16_t port) {
    sockaddr_in la{};
    la.sin_family = AF_INET;
    la.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &la.sin_addr);
    rec(s, std::format("bind(127.0.0.1:{})", port));
    return ::bind(so(s), reinterpret_cast<const sockaddr*>(&la), sizeof la);
}

int listen(Socket s, int backlog) {
    rec(s, std::format("listen({})", backlog));
    return ::listen(so(s), backlog);
}

int connect(Socket s, const Addr* a) {
    rec(s, "connect()");
    return ::connect(so(s), ai(a)->ai_addr, static_cast<int>(ai(a)->ai_addrlen));
}

Socket accept(Socket listener) {
    if (g_on.load(std::memory_order_relaxed)) {
        FakeState& f = fstate();
        std::unique_lock<std::mutex> lk(f.mu);
        const auto it = f.tag.find(listener);
        if (it != f.tag.end() && f.fail_accept > 0) {
            --f.fail_accept;
            const int err = f.fail_accept_err;
            recLocked(f, listener, std::format("accept()=fail({})", err));
            lk.unlock();
            WSASetLastError(err);
            return kInvalid;
        }
    }
    const SOCKET c = ::accept(so(listener), nullptr, nullptr);
    if (g_on.load(std::memory_order_relaxed)) {
        const int e = WSAGetLastError();
        FakeState& f = fstate();
        {
            std::lock_guard<std::mutex> lk(f.mu);
            if (f.tag.count(listener)) {
                if (c != INVALID_SOCKET) {
                    const std::string t = std::format("C{}", ++f.n_conn);
                    f.tag[static_cast<Socket>(c)] = t;
                    recLocked(f, listener, "accept()=" + t);
                } else {
                    recLocked(f, listener, "accept()=invalid");
                }
            } else {
                t_last_tracked = false;
            }
        }
        WSASetLastError(e);
    }
    return static_cast<Socket>(c);
}

int localPort(Socket s, std::uint16_t* port) {
    rec(s, "getsockname()");
    sockaddr_storage ss{};
    int sl = sizeof ss;
    const int rc = getsockname(so(s), reinterpret_cast<sockaddr*>(&ss), &sl);
    *port = ntohs(ss.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port
                                           : reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    return rc;
}

int recv(Socket s, char* buf, int len, int flags) {
    if (g_on.load(std::memory_order_relaxed)) {
        FakeState& f = fstate();
        std::unique_lock<std::mutex> lk(f.mu);
        const auto it = f.tag.find(s);
        if (it != f.tag.end() && it->second[0] == 'C' && f.fail_recv > 0) {
            --f.fail_recv;
            const int err = f.fail_recv_err;
            recLocked(f, s, std::format("recv({},{})=fail({})", len, flags, err));
            lk.unlock();
            WSASetLastError(err);
            return kError;
        }
        lk.unlock();
        rec(s, std::format("recv({},{})", len, flags));
    }
    return ::recv(so(s), buf, len, flags);
}

int send(Socket s, const char* buf, int len) {
    rec(s, std::format("send({},0)", len));
    return ::send(so(s), buf, len, 0);
}

int shutdown(Socket s, Shut how) {
    rec(s, std::format("shutdown({})", static_cast<int>(how)));
    return ::shutdown(so(s), static_cast<int>(how));
}

int close(Socket s) {
    if (g_on.load(std::memory_order_relaxed)) {
        // untracked before the handle is released: Winsock may hand the same value to the next socket at once
        FakeState& f = fstate();
        std::lock_guard<std::mutex> lk(f.mu);
        if (recLocked(f, s, "closesocket()")) f.tag.erase(s);
    }
    return closesocket(so(s));
}

void cancelIo(Socket s) {
    rec(s, "CancelIoEx()");
    CancelIoEx(reinterpret_cast<HANDLE>(s), nullptr);
}

int setExclusiveAddrUse(Socket s) {
    BOOL on = TRUE;
    return setOpt(s, SO_EXCLUSIVEADDRUSE, &on, sizeof on, on);
}

int setRecvTimeout(Socket s, std::uint32_t ms) {
    const DWORD v = ms;
    return setOpt(s, SO_RCVTIMEO, &v, sizeof v, v);
}

int setSendTimeout(Socket s, std::uint32_t ms) {
    const DWORD v = ms;
    return setOpt(s, SO_SNDTIMEO, &v, sizeof v, v);
}

int setSendBuf(Socket s, int bytes) { return setOpt(s, SO_SNDBUF, &bytes, sizeof bytes, bytes); }

int setRecvBuf(Socket s, int bytes) { return setOpt(s, SO_RCVBUF, &bytes, sizeof bytes, bytes); }

int socketError(Socket s) {
    rec(s, "getsockopt(SO_ERROR)");
    int v = 0, sl = sizeof v;
    getsockopt(so(s), SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&v), &sl);
    return v;
}

int setNonBlocking(Socket s, bool on) {
    rec(s, std::format("ioctlsocket(FIONBIO,{})", on ? 1 : 0));
    u_long v = on ? 1 : 0;
    return ioctlsocket(so(s), FIONBIO, &v);
}

int pollReadable(Socket s) {
    rec(s, "select(read,0)");
    fd_set rs;
    FD_ZERO(&rs);
    FD_SET(so(s), &rs);
    timeval tv{0, 0};
    return ::select(0, &rs, nullptr, nullptr, &tv);
}

int waitConnect(Socket s, long sec, long usec, bool* writable, bool* failed) {
    rec(s, std::format("select(write+except,{},{})", sec, usec));
    fd_set w, e;
    FD_ZERO(&w);
    FD_ZERO(&e);
    FD_SET(so(s), &w);
    FD_SET(so(s), &e);
    timeval tv{sec, usec};
    const int n = ::select(0, nullptr, &w, &e, &tv);
    *writable = FD_ISSET(so(s), &w) != 0;
    *failed = FD_ISSET(so(s), &e) != 0;
    return n;
}

ProbeResult probeListen(const char* host, const char* port) {
    ProbeResult r;
    if (!startup()) {
        r.stage = ProbeResult::Stage::no_winsock;
        return r;
    }
    const Addr* res = resolve(host, port, Lookup::passive_numeric);
    if (res == nullptr) {
        cleanup();
        r.stage = ProbeResult::Stage::bad_host;
        return r;
    }
    int err = 0;
    const Socket s = open(res);
    if (s != kInvalid) {
        setExclusiveAddrUse(s);
        if (net::bind(s, res) != 0 || net::listen(s, 1) != 0) err = lastError();
        net::close(s);
    }
    freeAddr(res);
    cleanup();
    r.err = err;
    return r;
}

// ---- FakeNet control

namespace fake {

void begin() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    f.passive.clear();
    f.tag.clear();
    f.n_listen = f.n_conn = 0;
    f.calls.clear();
    f.fail_accept = f.fail_accept_err = f.fail_recv = f.fail_recv_err = 0;
    g_on.store(true);
}

std::vector<Call> end() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    g_on.store(false);
    std::vector<Call> out = std::move(f.calls);
    f.calls.clear();
    f.passive.clear();
    f.tag.clear();
    f.fail_accept = f.fail_recv = 0;
    return out;
}

std::vector<Call> snapshot() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    return f.calls;
}

void failAccept(int n, int err) {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    f.fail_accept = n;
    f.fail_accept_err = err;
}

void failRecv(int n, int err) {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    f.fail_recv = n;
    f.fail_recv_err = err;
}

int injectedLeft() {
    FakeState& f = fstate();
    std::lock_guard<std::mutex> lk(f.mu);
    return f.fail_accept + f.fail_recv;
}

}  // namespace fake

}  // namespace whirl::platform::net
