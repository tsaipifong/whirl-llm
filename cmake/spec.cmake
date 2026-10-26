# WHIRL - speculative-decoding policy (src/spec, arch025-M4): MTP draft-count control, per-device MTP defaults, the
# draft-head vocabulary choice, the per-slot draft allocation and prompt-lookup (n-gram) drafting.
# SPDX-License-Identifier: Apache-2.0
#
# Host-only (no HIP): the server core (cmake/server.cmake) and the model library (cmake/model.cmake) both link it, so
# the CPU test executables no longer compile the policy source themselves. Included from the top-level CMakeLists.txt
# before server.cmake and model.cmake.

add_library(whirl_spec STATIC ${CMAKE_SOURCE_DIR}/src/spec/policy.cpp)
target_include_directories(whirl_spec PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whirl_spec PUBLIC whirl)
