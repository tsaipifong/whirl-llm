# WHIRL - run settings shared by the CLI and the server (src/config/run_settings.*, arch025-N15): the WHIRL_* names both
# front ends read, one parse rule each (the shape of llama.cpp common_params / common_arg).
# SPDX-License-Identifier: Apache-2.0
#
# Host-only, links whirl_platform only (no Model / HIP / log). Included from the top-level CMakeLists.txt after
# platform.cmake and before server.cmake / model.cmake; whirl_server_core links it PUBLIC (server_main.cpp, also built
# into whirl.exe for `whirl serve`) and whirl-cli links it for src/cli. Isolation: tests/iso/config.settings.

add_library(whirl_settings STATIC ${CMAKE_SOURCE_DIR}/src/config/run_settings.cpp)
target_include_directories(whirl_settings PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(whirl_settings PUBLIC whirl_platform)
