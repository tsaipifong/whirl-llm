# WHIRL - host prefix-cache tiers (pinned RAM / SSD), src/tier
# SPDX-License-Identifier: Apache-2.0
#
# Included from cmake/server.cmake.
#   whirl_tier_core : kv_tier.cpp only, no HIP (tests run it against a
#                     host-memory device::Backend)
# (the HIP device::Backend is whirl_device, cmake/server.cmake; ARCH-1 D-2)

add_library(whirl_tier_core STATIC ${CMAKE_SOURCE_DIR}/src/tier/kv_tier.cpp)
target_include_directories(whirl_tier_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_executable(whirl-tier-tests ${CMAKE_SOURCE_DIR}/tests/server/tier_tests.cpp)
target_link_libraries(whirl-tier-tests PRIVATE whirl_tier_core)
