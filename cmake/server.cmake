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
# device module (src/device): backends, own fragment (before the tiers, whose tests link it)
include(${CMAKE_CURRENT_LIST_DIR}/device.cmake)
# host prefix-cache tiers (src/tier): own fragment
include(${CMAKE_CURRENT_LIST_DIR}/tier.cmake OPTIONAL)

# ---------------------------------------------------------------------------
# prefix cache (src/cache): lookup / insertion / eviction decisions and the tier
# fingerprint, host-only; engine -> PrefixCache -> cache::Tier (adapter over tier::Tier)
add_library(whirl_cache STATIC
  ${CMAKE_SOURCE_DIR}/src/cache/prefix_cache.cpp
  ${CMAKE_SOURCE_DIR}/src/cache/tier_adapter.cpp
  ${CMAKE_SOURCE_DIR}/src/cache/fingerprint.cpp)
target_include_directories(whirl_cache PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whirl_cache PUBLIC whirl whirl_tier_core)

add_executable(whirl-cache-tests ${CMAKE_SOURCE_DIR}/tests/cache/cache_tests.cpp)
target_include_directories(whirl-cache-tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/cache)
target_link_libraries(whirl-cache-tests PRIVATE whirl_cache)
# arch025-i6vis: testFingerprintEnvClassified scans src/ and include/ for the WHIRL_* names the source reads
target_compile_definitions(whirl-cache-tests PRIVATE WHIRL_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(TARGET whirl-tier-tests)
  target_link_libraries(whirl-tier-tests PRIVATE whirl_cache)
endif()

# ---------------------------------------------------------------------------
# token sampling and draft acceptance (src/sampling, arch025-E1a): host-only; the device side of a verify
# row comes in through sampling::RowSource (the engine; the server tests use a host fake)
add_library(whirl_sampling STATIC
  ${CMAKE_SOURCE_DIR}/src/sampling/sampler.cpp
  ${CMAKE_SOURCE_DIR}/src/sampling/acceptor.cpp)
target_include_directories(whirl_sampling PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whirl_sampling PUBLIC whirl)
# isolation check (ARCH-1 8.5, arch025-X3 / arch025-isoall): every isolated module (tests/sampling/isolation.txt,
# tests/iso/*/isolation.txt) built from a copy that sees only its declared headers, its test program linked against
# those objects alone (no whirl, no server, no GGUF)
include(${CMAKE_CURRENT_LIST_DIR}/iso.cmake)

# ---------------------------------------------------------------------------
# OpenAI-compatible server (src/server)

# engine + HTTP front end; HIP-free (the GPU comes in through the ServerModel /
# device::Backend interfaces), so the tests can run it against a host-memory backend
add_library(whirl_server_core STATIC
  ${CMAKE_SOURCE_DIR}/src/server/log.cpp
  ${CMAKE_SOURCE_DIR}/src/server/server_args.cpp
  ${CMAKE_SOURCE_DIR}/src/server/protocol.cpp
  ${CMAKE_SOURCE_DIR}/src/server/stream.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_decode.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_pages.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_prefill.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_queue.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_tier.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_sampling.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_thread.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_vision.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_gov.cpp
  ${CMAKE_SOURCE_DIR}/src/server/engine_vram.cpp
  ${CMAKE_SOURCE_DIR}/src/cache/ckpt_store.cpp
  ${CMAKE_SOURCE_DIR}/src/cache/page_pool.cpp
  ${CMAKE_SOURCE_DIR}/src/server/prefix_store.cpp
  ${CMAKE_SOURCE_DIR}/src/server/tier_bridge.cpp
  ${CMAKE_SOURCE_DIR}/src/server/http.cpp)
target_include_directories(whirl_server_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
# whirl/version.h (configured by cmake/release.cmake) for GET /version and /props
target_include_directories(whirl_server_core PRIVATE ${CMAKE_BINARY_DIR}/generated)
# whirl_spec: the MTP / n-gram draft policy the engine runs (src/spec, arch025-M4)
# whirl_arch: the architecture registry (server_main's config + pre-load sizing, the tests' fake table; arch025-M5)
# whirl_sampling: Sampler / Acceptor (src/sampling, arch025-E1a)
# ws2_32 is linked by whirl_platform only (arch025 PL3: net_win.cpp is Winsock's one user)
target_link_libraries(whirl_server_core PUBLIC whirl whirl_spec whirl_arch whirl_cache whirl_tier_core whirl_sampling whirl_settings)

if(WHIRL_WITH_HIP)
  # whirl-server.exe: the production server (qwen35::Model on HIP)
  add_executable(whirl-server
    ${CMAKE_SOURCE_DIR}/src/server/whirl_server.cpp
    ${CMAKE_SOURCE_DIR}/src/server/server_main.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/server_model.cpp)
  target_link_libraries(whirl-server PRIVATE whirl_server_core whirl_model whirl_device)
  if(MSVC)
    target_link_options(whirl-server PRIVATE /ENTRY:wmainCRTStartup)
  endif()
endif()

# server tests: unit tests + end-to-end over HTTP against the mock model
# (the MTP / n-gram policy comes with whirl_server_core: whirl_spec, cmake/spec.cmake, arch025-M4)
add_executable(whirl-server-tests
  ${CMAKE_SOURCE_DIR}/tests/server/server_tests.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/mock_backend.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
target_include_directories(whirl-server-tests PRIVATE ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(whirl-server-tests PRIVATE whirl_server_core whirl_device_null)

# arch024 S2: host micro-benchmark of the KV page path at 128k (mock model, no GPU; S4's P-speed check)
add_executable(whirl-page-bench
  ${CMAKE_SOURCE_DIR}/tests/bench/page_bench.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/mock_backend.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
target_include_directories(whirl-page-bench PRIVATE ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(whirl-page-bench PRIVATE whirl_server_core whirl_device_null)

# measure-tools P0: the shared JSONL writer of the measurement tools (tests/bench/bench_out.h, header-only), CPU-only
# unit tests. `whirl-bench-out-tests --emit FILE` also writes a sample file for the records-side importer check.
add_executable(whirl-bench-out-tests ${CMAKE_SOURCE_DIR}/tests/bench/bench_out_tests.cpp)

# measure-tools P2: kt::Bench (the shared kernel timer of whirl-kernel-test --bench) has no HIP in it; its stop rule,
# calibration, rotation, anchor / cooldown, ABBA, burst rule, sha256 and JSONL rows are unit-tested on the CPU.
add_executable(whirl-kt-bench-tests
  ${CMAKE_SOURCE_DIR}/tests/kernels/unit/kt_bench_tests.cpp
  ${CMAKE_SOURCE_DIR}/tests/kernels/kt_bench.cpp)

# measure-tools P4: whirl-stage-bench's HIP-free part (options, stage table, KV tile plan, per-stage golden, validation
# judge, JSONL rows, the restore / preload / hash driver) on the CPU. `--emit FILE` writes sample rows for db_record.py.
add_executable(whirl-stage-bench-tests
  ${CMAKE_SOURCE_DIR}/tests/bench/unit/stage_bench_tests.cpp
  ${CMAKE_SOURCE_DIR}/tests/bench/stage_bench.cpp
  ${CMAKE_SOURCE_DIR}/tests/kernels/kt_bench.cpp)

# measure-tools P3: host components one at a time (tests/bench/host_bench.cpp). whirl-host-bench: --backend null
# only (CPU, never loads HIP); whirl-host-bench-hip: the same source with the HIP backend for the copy suite.
add_executable(whirl-host-bench
  ${CMAKE_SOURCE_DIR}/tests/bench/host_bench.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/mock_backend.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
target_include_directories(whirl-host-bench PRIVATE ${CMAKE_SOURCE_DIR}/tests/server ${CMAKE_SOURCE_DIR}/tests/bench)
target_link_libraries(whirl-host-bench PRIVATE whirl_server_core whirl_device_null)
if(WHIRL_WITH_HIP)
  add_executable(whirl-host-bench-hip
    ${CMAKE_SOURCE_DIR}/tests/bench/host_bench.cpp
    ${CMAKE_SOURCE_DIR}/tests/server/mock_backend.cpp
    ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
  target_include_directories(whirl-host-bench-hip PRIVATE ${CMAKE_SOURCE_DIR}/tests/server ${CMAKE_SOURCE_DIR}/tests/bench)
  target_compile_definitions(whirl-host-bench-hip PRIVATE WHIRL_HOST_BENCH_HIP=1)
  target_link_libraries(whirl-host-bench-hip PRIVATE whirl_server_core whirl_model whirl_device whirl_device_null)
endif()

# arch024 S0a: CPU pins (prompt tokens, engine memory / copy sequence), recorded on the B0 base (tests/pins)
add_executable(whirl-pins
  ${CMAKE_SOURCE_DIR}/tests/pins/pins.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/mock_backend.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
target_include_directories(whirl-pins PRIVATE ${CMAKE_SOURCE_DIR}/tests/server)
target_link_libraries(whirl-pins PRIVATE whirl_server_core whirl_device_null)
# arch024 S0a (PLAN v3 N3): startup sizing replay from a "startup inputs:" line (pure functions only)
add_executable(whirl-startup-replay ${CMAKE_SOURCE_DIR}/tests/pins/startup_replay.cpp)
target_link_libraries(whirl-startup-replay PRIVATE whirl_server_core)

# end-to-end gates on a real model (GPU): drives whirl-server.exe or, as a
# black box, the prototype exe; records outputs for cross-run comparison
add_executable(whirl-server-gate
  ${CMAKE_SOURCE_DIR}/tests/server/server_gate.cpp
  ${CMAKE_SOURCE_DIR}/tests/server/http_client.cpp)
target_link_libraries(whirl-server-gate PRIVATE whirl)
if(MSVC)
  target_link_options(whirl-server-gate PRIVATE /ENTRY:wmainCRTStartup)
endif()
