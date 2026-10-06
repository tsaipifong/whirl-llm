# WHIRL kernels: device sources, host-side kernel registry and the
# kernel test runner. Included from CMakeLists.txt right after
# WHIRL_KERNEL_SOURCES is initialized (before the per-arch code objects are
# built), so the list below feeds the code-object dependencies.
# SPDX-License-Identifier: Apache-2.0
#
# Two kernel sets: gfx1201 (RDNA 4, kernels/*.hip) and gfx1151 (RDNA 3.5,
# kernels/gfx1151/*.hip), one code object each (WHIRL_GPU_ARCHS, default both;
# -DWHIRL_GPU_ARCHS=gfx1201 or =gfx1151 builds one). Each code object depends
# only on its own sources (WHIRL_KERNEL_SOURCES_<arch>), so editing one set
# never recompiles the other.

file(GLOB WHIRL_KERNEL_FAMILY_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/kernels/*.hip
  ${CMAKE_SOURCE_DIR}/kernels/*.h)
file(GLOB WHIRL_KERNEL_GFX1151_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/kernels/gfx1151/*.hip)
set(WHIRL_KERNEL_SOURCES_gfx1201 ${WHIRL_KERNEL_SOURCES} ${WHIRL_KERNEL_FAMILY_SOURCES})
set(WHIRL_KERNEL_SOURCES_gfx1151 ${WHIRL_KERNEL_SOURCES} ${CMAKE_SOURCE_DIR}/kernels/iq_tables.h
    ${CMAKE_SOURCE_DIR}/kernels/spec_sample.hip ${CMAKE_SOURCE_DIR}/kernels/whirl_kernels.hip
    ${WHIRL_KERNEL_GFX1151_SOURCES})
list(APPEND WHIRL_KERNEL_SOURCES ${WHIRL_KERNEL_FAMILY_SOURCES} ${WHIRL_KERNEL_GFX1151_SOURCES})

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
