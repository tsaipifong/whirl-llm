# WHIRL ops: kernel choice per op (ARCH-1 K-1..K-5): pure functions of the problem, the numerics flags, the
# code object's kernel table and the tuned tables; src/model/forward.cpp launches what they return.
# SPDX-License-Identifier: Apache-2.0
#
#   whirl_ops        : src/ops (no HIP calls; kernels_abi.h types only)
#   whirl-ops-tests  : the module's own unit tests (CPU, fake kernel tables)

add_library(whirl_ops STATIC ${CMAKE_SOURCE_DIR}/src/ops/matmul.cpp ${CMAKE_SOURCE_DIR}/src/ops/attention.cpp
            ${CMAKE_SOURCE_DIR}/src/ops/moe.cpp)
target_include_directories(whirl_ops PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(whirl-ops-tests ${CMAKE_SOURCE_DIR}/tests/ops/ops_tests.cpp)
# feat026-moep: CPU replay of the multi-fragment MoE prefill GEMM schedule (kernels/moep_map.h, kMoepGu / kMoepDn)
add_executable(whirl-moep-sched-tests ${CMAKE_SOURCE_DIR}/tests/kernels/unit/moep_sched_tests.cpp)
target_include_directories(whirl-moep-sched-tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whirl-ops-tests PRIVATE whirl_ops)
