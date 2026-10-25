// Platform layer: sockets (arch025 PL3). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only (the
// socket handle becomes an opaque integer type, not SOCKET); the Windows implementation is net_win.cpp, chosen by
// cmake/platform.cmake (ws2_32 moves to whirl_platform then). No virtual interface.
#pragma once

namespace whirl::platform::net {

// PL3 moves here (Winsock call order and every socket option kept as in 0e35407, HTTP-DROP fix fc80ed4 included):
//   - the HTTP server: WSAStartup, getaddrinfo / freeaddrinfo, socket, setsockopt, bind, listen, accept, select,
//     recv / send, ioctlsocket, shutdown, closesocket, CancelIoEx, WSAGetLastError (src/server/http.cpp, SOCKET in
//     src/server/http.h)
//   - the port pre-check of the release front-end: WSAStartup / WSACleanup, getaddrinfo, socket, setsockopt, bind,
//     listen, closesocket (src/release/release.cpp)
//   - the test client (tests/server/http_client.cpp)
// The only other place with a fake (FakeNet: injected accept / recv errors).

}  // namespace whirl::platform::net
