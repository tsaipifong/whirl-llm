# WHIRL - C++ safety build options (safe025 prototype of the 0.2.5 exit criteria; PLAN_0.2.1.md "C++ safety")
# SPDX-License-Identifier: Apache-2.0
#
# All options default OFF: a normal configure is unchanged. Included from the top-level CMakeLists.txt right after the
# global compile options, before any target exists. Use with the separate build directories build-asan / build-analyze
# (tools/safety/*.bat; /build-*/ is in .gitignore). Written for WHIRL.
option(WHIRL_SANITIZE_ADDRESS "MSVC /fsanitize=address on every target (use a separate build directory)" OFF)
option(WHIRL_LEAKCHECK "NullBackend counts leaked allocations / streams / events when it is destroyed (test programs fail at exit)" OFF)
option(WHIRL_ANALYZE "MSVC /analyze (code analysis) on every target; warnings are compared with tools/safety/analyze_baseline.txt" OFF)

if(WHIRL_SANITIZE_ADDRESS)
  if(NOT MSVC)
    message(FATAL_ERROR "WHIRL_SANITIZE_ADDRESS: only the MSVC /fsanitize=address build is wired up")
  endif()
  # /fsanitize=address is incompatible with /RTC*, /ZI (edit and continue) and incremental linking
  string(REGEX REPLACE "/RTC[1csu]+" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REGEX REPLACE "/ZI" "/Zi" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  add_compile_options(/fsanitize=address /Zi)
  add_link_options(/INCREMENTAL:NO /DEBUG)
  add_compile_definitions(WHIRL_ASAN=1)
  message(STATUS "WHIRL: AddressSanitizer build (/fsanitize=address)")
endif()

if(WHIRL_LEAKCHECK)
  add_compile_definitions(WHIRL_LEAKCHECK=1)
  message(STATUS "WHIRL: NullBackend leak check on")
endif()

if(WHIRL_ANALYZE)
  if(NOT MSVC)
    message(FATAL_ERROR "WHIRL_ANALYZE: MSVC /analyze only")
  endif()
  # /analyze:external- keeps the standard library and the SDK out of the report. /analyze:log:format:sarif writes
  # <obj>.nativecodeanalysis.sarif next to every object: ruleId + function, independent of the MSVC language pack and of
  # the tree path, which is what tools/safety/analyze_check.py keys on (safe025 review A2)
  add_compile_options(/analyze /analyze:external- /analyze:log:format:sarif /external:W0 /external:anglebrackets)
  # arch025-Q3: the bin2c byte arrays of the HIP code objects (build/kernels/whirl_kernels_<arch>.cpp, whirl_vision_<arch>.cpp;
  # 47-131 MB of generated initializers, no code) are not analyzed: cl /analyze ran out of heap on them (C1060, 2026-10-25
  # build-analyze) once the CPU test targets link whirl_hip. The later /analyze- wins; analyze_check.py prepare skips such TUs.
  foreach(_arch gfx1151 gfx1201)
    set_source_files_properties("${CMAKE_BINARY_DIR}/kernels/whirl_kernels_${_arch}.cpp" "${CMAKE_BINARY_DIR}/kernels/whirl_vision_${_arch}.cpp"
                                PROPERTIES COMPILE_OPTIONS "/analyze-")
  endforeach()
  message(STATUS "WHIRL: MSVC /analyze build")
endif()

# arch025-Q4: Clang thread-safety analysis (release_gate step cpu_tsa). Configure a separate build directory (build-tsa) with
# clang-cl as the C++ compiler and WHIRL_WITH_HIP=OFF, and build whirl_server_core / whirl_tier_core
# (tools/safety/tsa/run_tsa_build.bat). Only the thread-safety diagnostics are on, and they are errors; the annotations are in
# include/whirl/thread_annotations.h (empty under MSVC, so the product build does not change).
option(WHIRL_TSA "clang-cl -Wthread-safety -Werror=thread-safety analysis build (separate build directory)" OFF)
if(WHIRL_TSA)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "WHIRL_TSA: needs clang-cl as CMAKE_CXX_COMPILER (found ${CMAKE_CXX_COMPILER_ID})")
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/clang:-Wno-everything;/clang:-Wthread-safety;/clang:-Werror=thread-safety>")
  message(STATUS "WHIRL: clang thread-safety analysis build (-Wthread-safety -Werror=thread-safety)")
endif()
