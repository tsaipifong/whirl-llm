# WHIRL - isolation check of every isolated module (ARCH-1 8.5; arch025-X3 for src/sampling, arch025-isoall generic)
# SPDX-License-Identifier: Apache-2.0
#
# Included from cmake/server.cmake after whirl_sampling. One manifest per module: tests/sampling/isolation.txt and
# tests/iso/<module>/isolation.txt (kinds: module, exe, lib, mutate, mutinc, mutsym, syslib, gen, src, hdr, testsrc, testhdr; see
# tools/gates/iso_check.py). The listed files are copied into <build>/iso/<module>/{mod,test}/ (the leading src/
# include/ tests/ dropped) and compiled from there:
#   <lib>   OBJECT library of the module sources + the declared dependency sources; include path = iso/<module>/mod
#           ONLY. An undeclared include, also a relative "../server/x.h", is C1083.
#   <exe>   the module's test program; include path = mod + test; links the <lib> OBJECTS directly (every object, not
#           a static-library member selection) with /OPT:NOREF and only the manifest's syslib lines, so a symbol from
#           outside the listed sources is LNK2019 even when no test calls it.
# The product build is unchanged: these targets are extra, nothing links them.
# Negative check (mutants that reach across the boundary must fail here and pass the product flags):
#   tools/gates/iso_check.py --mutants   (release_gate cpu_sampling_iso, cpu_iso_<module>)

function(whirl_iso_module _list)
  # start from an empty tree at every configure: a file dropped from the list must not stay visible
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_list})
  file(STRINGS ${_list} _lines REGEX "^[a-z]+[ \t]+[^ \t#]+")
  set(_module)
  set(_exe)
  set(_lib)
  set(_syslibs)
  set(_entries)
  set(_gens)
  foreach(_l IN LISTS _lines)
    string(REGEX MATCH "^([a-z]+)[ \t]+([^ \t#]+)" _m "${_l}")
    set(_kind ${CMAKE_MATCH_1})
    set(_path ${CMAKE_MATCH_2})
    if(_kind STREQUAL "module")
      set(_module ${_path})
    elseif(_kind STREQUAL "exe")
      set(_exe ${_path})
    elseif(_kind STREQUAL "lib")
      set(_lib ${_path})
    elseif(_kind STREQUAL "syslib")
      list(APPEND _syslibs ${_path})
    elseif(_kind MATCHES "^(mutate|mutinc|mutsym)$")
      # iso_check.py only
    elseif(_kind STREQUAL "gen")
      # arch025-lane2 leg 5: a header the build generates (${CMAKE_BINARY_DIR}/generated/<path>, e.g. whirl/version.h,
      # configured by cmake/release.cmake after this file): copied into mod/ at build time
      list(APPEND _gens ${_path})
    elseif(_kind MATCHES "^(src|hdr|testsrc|testhdr)$")
      if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${_path})
        message(FATAL_ERROR "${_list}: no such file '${_path}'")
      endif()
      list(APPEND _entries "${_kind}|${_path}")
    else()
      message(FATAL_ERROR "${_list}: unknown kind '${_kind}' in '${_l}'")
    endif()
  endforeach()
  if(NOT _module AND _list STREQUAL "${CMAKE_SOURCE_DIR}/tests/sampling/isolation.txt")
    set(_module sampling)
  endif()
  if(NOT _module OR NOT _exe)
    message(FATAL_ERROR "${_list}: needs 'module' and 'exe' lines")
  endif()
  if(NOT _lib)
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _lib "whirl_iso_${_module}")
  endif()
  set(_iso ${CMAKE_BINARY_DIR}/iso/${_module})
  file(REMOVE_RECURSE ${_iso})
  set(_mod_src)
  set(_test_src)
  foreach(_e IN LISTS _entries)
    string(REPLACE "|" ";" _kp "${_e}")
    list(GET _kp 0 _kind)
    list(GET _kp 1 _path)
    string(REGEX REPLACE "^(src|include|tests)/" "" _rel "${_path}")
    if(NOT _path MATCHES "^(src|include|tests)/")
      # arch025-lane2 leg 3: a file outside src/ include/ tests/ (kernels/gfx1151/attn_ws4_index.h, which
      # src/ops/attention.cpp includes as "../../kernels/...") sits next to mod/ as it sits next to src/ in the tree
      set(_dst ${_iso}/${_path})
    elseif(_kind MATCHES "^test")
      set(_dst ${_iso}/test/${_rel})
    else()
      set(_dst ${_iso}/mod/${_rel})
    endif()
    configure_file(${CMAKE_SOURCE_DIR}/${_path} ${_dst} COPYONLY)
    if(_kind STREQUAL "src")
      list(APPEND _mod_src ${_dst})
    elseif(_kind STREQUAL "testsrc")
      list(APPEND _test_src ${_dst})
    endif()
  endforeach()
  if(NOT _mod_src OR NOT _test_src)
    message(FATAL_ERROR "${_list}: needs src and testsrc lines")
  endif()

  set(_gen_dst)
  foreach(_g IN LISTS _gens)
    add_custom_command(OUTPUT ${_iso}/mod/${_g}
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/generated/${_g} ${_iso}/mod/${_g}
                       DEPENDS ${CMAKE_BINARY_DIR}/generated/${_g} VERBATIM)
    list(APPEND _gen_dst ${_iso}/mod/${_g})
  endforeach()
  add_library(${_lib} OBJECT ${_mod_src} ${_gen_dst})
  # set (not append) the include path: nothing inherited, nothing from linked targets
  set_property(TARGET ${_lib} PROPERTY INCLUDE_DIRECTORIES ${_iso}/mod)

  add_executable(${_exe} ${_test_src})
  set_property(TARGET ${_exe} PROPERTY INCLUDE_DIRECTORIES ${_iso}/mod ${_iso}/test)
  target_link_libraries(${_exe} PRIVATE ${_lib} ${_syslibs})
  if(MSVC)
    target_link_options(${_exe} PRIVATE /OPT:NOREF)
  endif()
endfunction()

whirl_iso_module(${CMAKE_SOURCE_DIR}/tests/sampling/isolation.txt)
file(GLOB _whirl_iso_lists CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/iso/*/isolation.txt)
list(SORT _whirl_iso_lists)
foreach(_iso_list IN LISTS _whirl_iso_lists)
  whirl_iso_module(${_iso_list})
endforeach()
