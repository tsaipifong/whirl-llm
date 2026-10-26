# WHIRL - architecture registry (src/model/arch_registry.*, arch025-M5): the GGUF architecture names, the family module
# serving each (config reader, tensor check, pre-load sizing), the UnsupportedArch text and the server's per-slot state
# bytes.
# SPDX-License-Identifier: Apache-2.0
#
# Host-only (no HIP): the server core (cmake/server.cmake) and the model library (cmake/model.cmake) both link it; the
# product table builtinArchs() is the model library's (src/model/qwen35/module.cpp), tests pass their own table.
# Included from the top-level CMakeLists.txt after spec.cmake, before server.cmake and model.cmake.

add_library(whirl_arch STATIC ${CMAKE_SOURCE_DIR}/src/model/arch_registry.cpp)
target_include_directories(whirl_arch PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whirl_arch PUBLIC whirl)
