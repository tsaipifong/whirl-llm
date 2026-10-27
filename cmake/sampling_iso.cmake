# WHIRL - isolation check of src/sampling (ARCH-1 8.5, arch025-X3)
# SPDX-License-Identifier: Apache-2.0
#
# Included from cmake/server.cmake after whirl_sampling. The files listed in tests/sampling/isolation.txt are copied
# into <build>/iso/sampling/{mod,test}/ (the leading src/ include/ tests/ dropped) and compiled from there:
#   whirl_sampling_iso    the module sources; include path = iso/sampling/mod ONLY (its interface headers and the
#                         declared header-only dependencies). An undeclared include, also a relative "../server/x.h",
#                         is C1083.
#   whirl-sampling-tests  the sampling unit tests (tests/sampling/sampling_unit.inc, the same bodies whirl-server-tests
#                         runs); include path = iso/sampling/mod + iso/sampling/test; links whirl_sampling_iso and
#                         NOTHING else, so a module symbol from outside the listed sources is LNK2019.
# The product build is unchanged: whirl_server_core still links whirl_sampling (cmake/server.cmake).
# Negative check (mutants that reach across the boundary must fail here and pass the product flags):
#   tools/gates/iso_check.py --mutants   (from a VS developer prompt; scratch\a25_x3\iso_run.bat)

set(WHIRL_ISO_SAMPLING_LIST ${CMAKE_SOURCE_DIR}/tests/sampling/isolation.txt)
set(_iso ${CMAKE_BINARY_DIR}/iso/sampling)
# start from an empty tree at every configure: a file dropped from the list must not stay visible
file(REMOVE_RECURSE ${_iso})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${WHIRL_ISO_SAMPLING_LIST})
file(STRINGS ${WHIRL_ISO_SAMPLING_LIST} _iso_lines REGEX "^[a-z]+[ \t]+[^ \t#]+")
set(_iso_mod_src)
set(_iso_test_src)
set(_iso_exe)
foreach(_l IN LISTS _iso_lines)
  string(REGEX MATCH "^([a-z]+)[ \t]+([^ \t#]+)" _m "${_l}")
  set(_kind ${CMAKE_MATCH_1})
  set(_path ${CMAKE_MATCH_2})
  if(_kind STREQUAL "exe")
    set(_iso_exe ${_path})
    continue()
  endif()
  if(NOT _kind MATCHES "^(src|hdr|testsrc|testhdr)$")
    message(FATAL_ERROR "${WHIRL_ISO_SAMPLING_LIST}: unknown kind '${_kind}' in '${_l}'")
  endif()
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${_path})
    message(FATAL_ERROR "${WHIRL_ISO_SAMPLING_LIST}: no such file '${_path}'")
  endif()
  string(REGEX REPLACE "^(src|include|tests)/" "" _rel "${_path}")
  if(_kind MATCHES "^test")
    set(_dst ${_iso}/test/${_rel})
  else()
    set(_dst ${_iso}/mod/${_rel})
  endif()
  configure_file(${CMAKE_SOURCE_DIR}/${_path} ${_dst} COPYONLY)
  if(_kind STREQUAL "src")
    list(APPEND _iso_mod_src ${_dst})
  elseif(_kind STREQUAL "testsrc")
    list(APPEND _iso_test_src ${_dst})
  endif()
endforeach()

if(NOT _iso_exe STREQUAL "whirl-sampling-tests" OR NOT _iso_mod_src OR NOT _iso_test_src)
  message(FATAL_ERROR "${WHIRL_ISO_SAMPLING_LIST}: needs 'exe whirl-sampling-tests', src and testsrc lines")
endif()

add_library(whirl_sampling_iso STATIC ${_iso_mod_src})
# set (not append) the include path: nothing inherited, nothing from linked targets
set_property(TARGET whirl_sampling_iso PROPERTY INCLUDE_DIRECTORIES ${_iso}/mod)

add_executable(whirl-sampling-tests ${_iso_test_src})
set_property(TARGET whirl-sampling-tests PROPERTY INCLUDE_DIRECTORIES ${_iso}/mod ${_iso}/test)
target_link_libraries(whirl-sampling-tests PRIVATE whirl_sampling_iso)
