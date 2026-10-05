# WHIRL model: weight loader, qwen35 forward, MTP / n-gram drafting,
# autotune, and the `whirl` CLI (chat / bench / selftest).
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt after the core library.

if(WHIRL_WITH_HIP)
  # default MTP draft-head vocabulary subset (64k token ids, uint32 LE), embedded
  # so the release does not depend on a file next to the exe (WHIRL_DRAFT_VOCAB)
  set(WHIRL_DRAFT_VOCAB_BIN "${CMAKE_SOURCE_DIR}/data/draft_vocab/subset_64k.bin")
  set(WHIRL_DRAFT_VOCAB_CC "${CMAKE_BINARY_DIR}/generated/draft_vocab_64k.cpp")
  add_custom_command(
    OUTPUT "${WHIRL_DRAFT_VOCAB_CC}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/generated"
    COMMAND whirl-bin2c "${WHIRL_DRAFT_VOCAB_BIN}" "${WHIRL_DRAFT_VOCAB_CC}" whirl_draft_vocab_64k
    DEPENDS whirl-bin2c "${WHIRL_DRAFT_VOCAB_BIN}"
    COMMENT "Embedding draft vocabulary subset 64k"
    VERBATIM)

  add_library(whirl_model STATIC
    ${CMAKE_SOURCE_DIR}/src/model/config.cpp
    ${CMAKE_SOURCE_DIR}/src/model/kernels.cpp
    ${CMAKE_SOURCE_DIR}/src/model/loader.cpp
    ${CMAKE_SOURCE_DIR}/src/model/forward.cpp
    ${CMAKE_SOURCE_DIR}/src/model/tune.cpp
    ${CMAKE_SOURCE_DIR}/src/model/device.cpp
    ${CMAKE_SOURCE_DIR}/src/model/spec.cpp
    ${CMAKE_SOURCE_DIR}/src/model/draft_vocab_embed.cpp
    "${WHIRL_DRAFT_VOCAB_CC}")
  target_include_directories(whirl_model PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(whirl_model PUBLIC whirl_kernels_host whirl)

  # the CLI executable is whirl.exe (target name differs from the core library)
  add_executable(whirl-cli ${CMAKE_SOURCE_DIR}/src/cli/main.cpp)
  set_target_properties(whirl-cli PROPERTIES OUTPUT_NAME whirl)
  target_link_libraries(whirl-cli PRIVATE whirl_model)
  # `whirl serve` forwards to the server's serveMain (cmake/server.cmake)
  if(TARGET whirl_server_core)
    target_sources(whirl-cli PRIVATE
      ${CMAKE_SOURCE_DIR}/src/server/server_main.cpp
      ${CMAKE_SOURCE_DIR}/src/server/backend_qwen35.cpp
      ${CMAKE_SOURCE_DIR}/src/tier/device_ops_hip.cpp)
    target_link_libraries(whirl-cli PRIVATE whirl_server_core)
    target_compile_definitions(whirl-cli PRIVATE WHIRL_HAVE_SERVER=1)
  endif()

  # host-only unit tests of the model library (no GPU work)
  add_executable(whirl-model-tests ${CMAKE_SOURCE_DIR}/tests/model/model_tests.cpp)
  target_link_libraries(whirl-model-tests PRIVATE whirl_model)
endif()
