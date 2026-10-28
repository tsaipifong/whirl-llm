# WHIRL host-side unit tests, one file per module (arch025 N7; pattern: llama.cpp tests/CMakeLists.txt
# llama_build_and_test, LLVM add_llvm_unittest - progress\arch025-lane3.PRIOR_ART.md section 5).
#   tests/unit/<module>_tests.cpp   one module's tests + its Suite (tests/test_check.h); module name with '.' -> '_'
#   whirl-unit-<module>-tests       that file alone (without WHIRL_UNIT_ALL the file has its own main); release_gate cpu_unit_<module>
#   whirl-tests                     every tests/unit/*_tests.cpp: the module files + unit_tests.cpp (runs them all in the
#                                   old order and fails setup when a case is missing from its ORDER table) +
#                                   numerics_plan_tests.cpp (contract's plan table, no main); release_gate cpu_tests
# Adding a module: add tests/unit/<module>_tests.cpp and its cases to unit_tests.cpp ORDER (whirl-tests refuses to run
# otherwise). tools/impact/impact_build.py reads the GLOB below for whirl-tests' sources.
# SPDX-License-Identifier: Apache-2.0

file(GLOB WHIRL_UNIT_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/unit/*_tests.cpp)
list(SORT WHIRL_UNIT_SOURCES)

add_executable(whirl-tests ${WHIRL_UNIT_SOURCES})
target_link_libraries(whirl-tests PRIVATE whirl)
# GPU runtime load-failure diagnosis (release front-end, issue #2), CPU-only
target_sources(whirl-tests PRIVATE src/release/dll_deps.cpp)
target_include_directories(whirl-tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(whirl-tests PRIVATE WHIRL_TEST_DLL_DEPS=1 WHIRL_UNIT_ALL=1)  # WHIRL_UNIT_ALL: the module files without their own main

# extra sources / definitions a module's own program needs (whirl-tests has them all)
set(WHIRL_UNIT_EXTRA_contract ${CMAKE_SOURCE_DIR}/tests/unit/numerics_plan_tests.cpp)
set(WHIRL_UNIT_EXTRA_release ${CMAKE_SOURCE_DIR}/src/release/dll_deps.cpp)
set(WHIRL_UNIT_DEFS_release WHIRL_TEST_DLL_DEPS=1)

foreach(_src IN LISTS WHIRL_UNIT_SOURCES)
  get_filename_component(_stem ${_src} NAME_WE)
  string(REGEX REPLACE "_tests$" "" _m ${_stem})
  if(_m STREQUAL "unit" OR _m STREQUAL "numerics_plan")
    continue()  # the whole-set main / a part of contract's tests
  endif()
  string(REPLACE "_" "-" _dash ${_m})
  set(_t whirl-unit-${_dash}-tests)
  add_executable(${_t} ${_src} ${WHIRL_UNIT_EXTRA_${_m}})
  target_link_libraries(${_t} PRIVATE whirl)
  target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(${_t} PRIVATE ${WHIRL_UNIT_DEFS_${_m}})
endforeach()
