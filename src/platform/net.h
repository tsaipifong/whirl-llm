// Platform layer: sockets (arch025 PL3). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only (the
// socket handle is an opaque integer type, not SOCKET); the Windows implementation is net_win.cpp, chosen by
// cmake/platform.cmake (which links ws2_32). No virtual interface.
//
// One function per Winsock call (as cpp-httplib's read_socket / send_socket / close_socket): the callers keep their
// logic and so their call order; every socket option keeps its level / name / value type / length of v0.2.4
// 0e35407 (HTTP-DROP fix fc80ed4 included), pinned by the server test net_call_table. Return values and error codes
// are Winsock's (0 / -1 = kError, WSAGetLastError values as plain ints). The callers:
//   - the HTTP server (src/server/http.cpp, the socket of src/server/http.h)
//   - the release front-end's port pre-check (probeListen, the whole sequence of 0e35407 release.cpp:148-178)
//   - the test client (tests/server/http_client.cpp)
// Tests inject accept / recv errors and record the call sequence with FakeNet (platform/net_fake.h).
#pragma once

#include <cstdint>

namespace whirl::platform::net {

// A socket handle (the bits of a Winsock SOCKET, which is UINT_PTR).
using Socket = std::uintptr_t;
inline constexpr Socket kInvalid = ~Socket(0);  // INVALID_SOCKET
inline constexpr int kError = -1;               // SOCKET_ERROR

// Winsock error codes the callers test for (net_win.cpp static_asserts them against <winsock2.h>)
inline constexpr int kErrAccess = 10013;        // WSAEACCES
inline constexpr int kErrWouldBlock = 10035;    // WSAEWOULDBLOCK
inline constexpr int kErrAddrInUse = 10048;     // WSAEADDRINUSE
inline constexpr int kErrAddrNotAvail = 10049;  // WSAEADDRNOTAVAIL
inline constexpr int kErrConnReset = 10054;     // WSAECONNRESET
inline constexpr int kErrNoBufs = 10055;        // WSAENOBUFS
inline constexpr int kErrTimedOut = 10060;      // WSAETIMEDOUT
inline constexpr int kErrConnRefused = 10061;   // WSAECONNREFUSED

inline constexpr int kMaxBacklog = 0x7fffffff;  // SOMAXCONN
inline constexpr int kPeek = 0x2;               // MSG_PEEK

// WSAStartup(MAKEWORD(2, 2)); true when it returned 0. WSACleanup.
bool startup();
void cleanup();
int lastError();          // WSAGetLastError
void setLastError(int e);  // WSASetLastError

// A getaddrinfo result list (opaque addrinfo). The callers use its first entry only, as before.
struct Addr;
enum class Lookup {
    passive_numeric,  // AF_UNSPEC, SOCK_STREAM, AI_PASSIVE | AI_NUMERICHOST (the server's and the pre-check's bind)
    client_v4,        // AF_INET, SOCK_STREAM, no flags (the test client's connect)
};
// getaddrinfo(host, port, hints of `how`); nullptr when it failed or returned no entry.
const Addr* resolve(const char* host, const char* port, Lookup how);
void freeAddr(const Addr* a);  // freeaddrinfo

Socket open(const Addr* a);       // socket(ai_family, ai_socktype, ai_protocol) of the first entry
Socket openTcp(const Addr* a);    // socket(ai_addr->sa_family, SOCK_STREAM, IPPROTO_TCP)
Socket openTcp4();                // socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
int bind(Socket s, const Addr* a);           // bind(s, ai_addr, ai_addrlen)
int bindLoopback4(Socket s, std::uint16_t port);  // bind(s, 127.0.0.1:port as sockaddr_in)
int listen(Socket s, int backlog);
int connect(Socket s, const Addr* a);        // connect(s, ai_addr, ai_addrlen)
Socket accept(Socket listener);              // accept(listener, nullptr, nullptr)
// getsockname (sockaddr_storage); *port = the IPv4 / IPv6 port in host order. Returns getsockname's result.
int localPort(Socket s, std::uint16_t* port);
int recv(Socket s, char* buf, int len, int flags = 0);
int send(Socket s, const char* buf, int len);  // flags 0
enum class Shut { send = 1, both = 2 };        // SD_SEND, SD_BOTH
int shutdown(Socket s, Shut how);
int close(Socket s);                           // closesocket
void cancelIo(Socket s);                       // CancelIoEx(s, nullptr): ends a recv already waiting

// setsockopt(s, SOL_SOCKET, ...), each with the value type of 0e35407:
int setExclusiveAddrUse(Socket s);              // SO_EXCLUSIVEADDRUSE, BOOL TRUE (4 bytes)
int setRecvTimeout(Socket s, std::uint32_t ms);  // SO_RCVTIMEO, DWORD ms
int setSendTimeout(Socket s, std::uint32_t ms);  // SO_SNDTIMEO, DWORD ms
int setSendBuf(Socket s, int bytes);             // SO_SNDBUF, int
int setRecvBuf(Socket s, int bytes);             // SO_RCVBUF, int
int socketError(Socket s);                       // getsockopt(SO_ERROR) value (0 if getsockopt failed)
int setNonBlocking(Socket s, bool on);           // ioctlsocket(FIONBIO, u_long 1 / 0)

// select(0, {s}, nullptr, nullptr, {0, 0}): > 0 readable, 0 not, kError on failure.
int pollReadable(Socket s);
// select(0, nullptr, {s}, {s}, {sec, usec}); *writable / *failed: FD_ISSET of the write / except set.
int waitConnect(Socket s, long sec, long usec, bool* writable, bool* failed);

// The release front-end's port pre-check, the call sequence of 0e35407 release.cpp:148-178 (WSAStartup;
// getaddrinfo passive_numeric [fail: WSACleanup]; socket; SO_EXCLUSIVEADDRUSE; bind; listen(1) [fail:
// WSAGetLastError]; closesocket; freeaddrinfo; WSACleanup). The caller maps the result to its messages.
struct ProbeResult {
    enum class Stage { no_winsock, bad_host, done } stage = Stage::done;
    int err = 0;  // done: the bind / listen error (0: the port is free, or socket() failed)
};
ProbeResult probeListen(const char* host, const char* port);

}  // namespace whirl::platform::net
