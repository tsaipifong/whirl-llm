# WHIRL model: weight loader, qwen35 forward, MTP / n-gram drafting,
# autotune, and the `whirl` CLI (chat / bench / selftest).
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt after the core library.

if(WHIRL_WITH_HIP)
  add_library(whirl_model STATIC
    ${CMAKE_SOURCE_DIR}/src/model/config.cpp
    ${CMAKE_SOURCE_DIR}/src/model/kernels.cpp
    ${CMAKE_SOURCE_DIR}/src/model/loader.cpp
    ${CMAKE_SOURCE_DIR}/src/model/forward.cpp
    ${CMAKE_SOURCE_DIR}/src/model/tune.cpp
    ${CMAKE_SOURCE_DIR}/src/model/device.cpp
    ${CMAKE_SOURCE_DIR}/src/model/spec.cpp)
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
