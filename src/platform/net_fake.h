// FakeNet (arch025 PL3): test-only control of the socket layer (platform/net.h). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// A pass-through fake inside net_win.cpp (prior art: Boost.Corosio's mocket "fuse", HAWQ's simulated errors; MSVC
// has no ld --wrap, ADR-0004 #5 has no virtual interface): off unless a test calls begin(), then
//  - it tracks the sockets a server makes: every socket opened on a Lookup::passive_numeric address (the HTTP
//    listener, the pre-check's socket) and every socket accept() returns on a tracked listener. Sockets of the test
//    client in the same process (Lookup::client_v4) are never tracked, so the client is never affected;
//  - it records every call on a tracked socket (and the passive getaddrinfo / freeaddrinfo, WSAStartup /
//    WSACleanup, and a WSAGetLastError right after a tracked call on the same thread) with its arguments;
//  - it injects errors: the next n accept() calls on a tracked listener / recv() calls on a tracked accepted
//    socket return kInvalid / kError with lastError() = err, without calling Winsock.
// Off (the product), each net call costs one relaxed atomic load more.
#pragma once

#include <string>
#include <vector>

namespace whirl::platform::net::fake {

// One recorded call: tag "L1", "L2" (tracked passive sockets in opening order), "C1", ... (accepted sockets), or
// "-" (no socket); text "fn(arg,arg,...)" with the arguments as plain integers, e.g.
// "setsockopt(65535,4102,4,300)" (level, name, length, value) or "recv(65536,0)" (length, flags).
struct Call {
    std::string tag;
    std::string text;
};

void begin();                    // start tracking / recording (clears everything recorded before)
std::vector<Call> end();         // stop; the calls recorded since begin(), in order
std::vector<Call> snapshot();    // the calls so far (recording goes on)
void failAccept(int n, int err);  // the next n accept() calls on a tracked listener fail with err
void failRecv(int n, int err);    // the next n recv() calls on a tracked accepted socket fail with err
int injectedLeft();               // accept + recv injections not used yet

}  // namespace whirl::platform::net::fake
