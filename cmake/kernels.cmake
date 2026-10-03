# WHIRL kernels: device sources, host-side kernel registry and the
# kernel test runner. Included from CMakeLists.txt right after
# WHIRL_KERNEL_SOURCES is initialized (before the per-arch code objects are
# built), so the list below feeds the code-object dependencies.
# SPDX-License-Identifier: Apache-2.0
#
# Only the gfx1201 (RDNA 4) kernel set is ported so far; the gfx1151 code
# object still carries just the smoke kernel (see kernels/whirl_kernels.hip).
# Configure with -DWHIRL_GPU_ARCHS=gfx1201 to skip the gfx1151 object.

file(GLOB WHIRL_KERNEL_FAMILY_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/kernels/*.hip
  ${CMAKE_SOURCE_DIR}/kernels/*.h
  ${CMAKE_SOURCE_DIR}/kernels/gfx1151/*.hip)
list(APPEND WHIRL_KERNEL_SOURCES ${WHIRL_KERNEL_FAMILY_SOURCES})

if(WHIRL_WITH_HIP)
  # Host-side kernel registry (loads every kernel by name from the embedded
  # code object). Links against whirl / whirl_hip, which are defined later in
  # CMakeLists.txt; CMake resolves target names at generate time.
  add_library(whirl_kernels_host STATIC
    ${CMAKE_SOURCE_DIR}/src/kernels_host/kernels_abi.cpp)
  target_include_directories(whirl_kernels_host PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(whirl_kernels_host PUBLIC whirl)

  # whirl-kernel-test: GPU kernels vs C++ CPU references.
  file(GLOB WHIRL_KERNEL_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/kernels/*.cpp)
  add_executable(whirl-kernel-test ${WHIRL_KERNEL_TEST_SOURCES})
  target_include_directories(whirl-kernel-test PRIVATE ${CMAKE_SOURCE_DIR}/kernels)  # iq_tables.h (host copy of the grid)
  target_link_libraries(whirl-kernel-test PRIVATE whirl_kernels_host whirl)
endif()
