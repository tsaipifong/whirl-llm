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
  message(STATUS "WHIRL: MSVC /analyze build")
endif()
