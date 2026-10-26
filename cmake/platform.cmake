# WHIRL - platform layer (src/platform, arch025 PL0; ADR-0004 Decision 5)
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt before the core library. One header per concern
# (src/platform/<concern>.h, platform-free) and one Windows implementation per concern (<concern>_win.cpp), chosen
# here at link time; no virtual interface. whirl_platform is the bottom layer: it links no other WHIRL target.
# PL0 is the skeleton (headers_win.cpp: the compile-time "headers stay platform-free" check); PL3 added net
# (net_win.cpp + FakeNet, ws2_32); PL1 / PL2 / PL4 add env / paths / console / time, fs, proc / sys. See docs/dev/platform.md.

add_library(whirl_platform STATIC
  ${CMAKE_SOURCE_DIR}/src/platform/headers_win.cpp
  ${CMAKE_SOURCE_DIR}/src/platform/net_win.cpp)
# PL3: Winsock is linked here only (net_win.cpp is its one user; whirl_server_core / whirl_release no longer name it)
target_link_libraries(whirl_platform PRIVATE ws2_32)
# PL1: env / paths / console / time (added as a separate target_sources so PL2 / PL4 add theirs without touching
# these lines); winmm for timeBeginPeriod (was #pragma comment(lib) in src/server/server_main.cpp)
target_sources(whirl_platform PRIVATE
  ${CMAKE_SOURCE_DIR}/src/platform/env_win.cpp
  ${CMAKE_SOURCE_DIR}/src/platform/paths_win.cpp
  ${CMAKE_SOURCE_DIR}/src/platform/console_win.cpp
  ${CMAKE_SOURCE_DIR}/src/platform/time_win.cpp)
target_link_libraries(whirl_platform PRIVATE winmm)
# PL2: fs (MappedFile, the GGUF reader, the server log, the SSD tier files; FakeFs in fs_fake.h)
target_sources(whirl_platform PRIVATE
  ${CMAKE_SOURCE_DIR}/src/platform/fs_win.cpp)
# callers include "platform/<concern>.h"; never put src/platform itself on an include path (its time.h would shadow
# the C library's <time.h>)
target_include_directories(whirl_platform PUBLIC ${CMAKE_SOURCE_DIR}/src)
