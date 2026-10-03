# WHIRL - server / tiered cache / repository tools and parity drivers
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt after the core library.

# ---------------------------------------------------------------------------
# repository tools (pure C++; nothing needs Python to build or test)

# UCD text files -> src/tokenizer/unicode_tables.inc
add_executable(whirl-gen-unicode tools/gen-unicode-tables/main.cpp)

# parity drivers against external references (llama-tokenize, llama.cpp
# unicode-data.cpp, the template oracle, reference GGUF dumps)
add_executable(whirl-parity
  tests/parity/parity_main.cpp
  tests/parity/make_corpus.cpp
  tests/parity/support.cpp)
target_link_libraries(whirl-parity PRIVATE whirl)
target_compile_definitions(whirl-parity PRIVATE WHIRL_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(MSVC)
  # wmain entry point (UTF-16 command line)
  target_link_options(whirl-parity PRIVATE /ENTRY:wmainCRTStartup)
endif()

# ---------------------------------------------------------------------------
# host prefix-cache tiers (src/tier): own fragment
include(${CMAKE_CURRENT_LIST_DIR}/tier.cmake OPTIONAL)

# ---------------------------------------------------------------------------
# OpenAI-compatible server (src/server)

# engine + HTTP front end; HIP-free (the GPU comes in through the ServerModel /
# DeviceOps interfaces), so the tests can run it against a host-memory mock
add_library(whirl_server_core STATIC
  ${CMAKE_SOURCE_DIR}/src/server/log.cpp
  ${CMAKE_SOURCE_DIR}/src/server/protocol.cpp
  ${CMAKE_SOURCE_DIR}/src/server/tokens.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_vision.cpp
  ${CMAKE_SOURCE_DIR}/src/server/http.cpp)
target_include_directories(whirl_server_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
# whirl/version.h (configured by cmake/release.cmake) for GET /version and /props
target_include_directories(whirl_server_core PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(whirl_server_core PUBLIC whirl whirl_tier_core ws2_32)

if(WHIRL_WITH_HIP)
  # whirl-server.exe: the production server (qwen35::Model on HIP)
  add_executable(whirl-server
    ${CMAKE_SOURCE_DIR}/src/server/whirl_server.cpp
    ${CMAKE_SOURCE_DIR}/src/server/server_main.cpp
    ${CMAKE_SOURCE_DIR}/src/server/backend_qwen35.cpp
    ${CMAKE_SOURCE_DIR}/src/tier/device_ops_hip.cpp)
  target_link_libraries(whirl-server PRIVATE whirl_server_core whirl_model)
  if(MSVC)
    target_link_options(whirl-server PRIVATE /ENTRY:wmainCRTStartup)
  endif()
endif()

# server tests: unit tests + end-to-end over HTTP against the mock model
# (the MTP / n-gram policy code of the model library is host-only: spec.cpp)
add_executable(whirl-server-tests
  ${CMAKE_SOURCE_DIR}/tests/server/server_tests.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/mock_backend.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp
  ${CMAKE_SOURCE_DIR}/src/model/spec.cpp)
target_include_directories(whirl-server-tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(whirl-server-tests PRIVATE whirl_server_core)

# end-to-end gates on a real model (GPU): drives whirl-server.exe or, as a
# black box, the prototype exe; records outputs for cross-run comparison
add_executable(whirl-server-gate
  ${CMAKE_SOURCE_DIR}/tests/server/server_gate.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
target_link_libraries(whirl-server-gate PRIVATE whirl)
if(MSVC)
  target_link_options(whirl-server-gate PRIVATE /ENTRY:wmainCRTStartup)
endif()
