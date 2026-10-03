# WHIRL release front-end: version header, Windows resources
# (VERSIONINFO, icon, application manifest), the shared error-message /
# help module, and delay-loading of the driver's GPU runtime.
# SPDX-License-Identifier: Apache-2.0
#
# Included last from the top-level CMakeLists.txt, so every executable target
# of the other fragments exists here.

# ---------------------------------------------------------------------------
# version header (single source: project(whirl VERSION ...))

string(REPLACE ";" ", " WHIRL_GPU_ARCHS_TEXT "${WHIRL_GPU_ARCHS}")
configure_file(${CMAKE_SOURCE_DIR}/src/release/version.h.in
               ${CMAKE_BINARY_DIR}/generated/whirl/version.h @ONLY)

# ---------------------------------------------------------------------------
# application icon: rendered at build time by a small C++ tool (no binary
# asset in the repository)

add_executable(whirl-gen-icon ${CMAKE_SOURCE_DIR}/tools/gen-icon/main.cpp)
set(WHIRL_ICON "${CMAKE_BINARY_DIR}/res/whirl.ico")
add_custom_command(
  OUTPUT "${WHIRL_ICON}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/res"
  COMMAND whirl-gen-icon "${WHIRL_ICON}"
  DEPENDS whirl-gen-icon
  COMMENT "Rendering the WHIRL icon"
  VERBATIM)
add_custom_target(whirl-icon DEPENDS "${WHIRL_ICON}")

# whirl_add_resources(TARGET "File description" [ICON])
#   VERSIONINFO (+ icon) resource and the application manifest for TARGET.
function(whirl_add_resources target description)
  if(NOT MSVC OR NOT TARGET ${target})
    return()
  endif()
  get_target_property(out_name ${target} OUTPUT_NAME)
  if(NOT out_name)
    set(out_name ${target})
  endif()
  set(WHIRL_RC_DESCRIPTION "${description}")
  set(WHIRL_RC_INTERNAL "${out_name}")
  set(WHIRL_RC_ORIGINAL "${out_name}.exe")
  set(WHIRL_RC_ICON_LINE "")
  set(rc "${CMAKE_BINARY_DIR}/res/${target}.rc")
  if("ICON" IN_LIST ARGN)
    file(TO_CMAKE_PATH "${WHIRL_ICON}" ico_path)
    set(WHIRL_RC_ICON_LINE "1 ICON \"${ico_path}\"")
  endif()
  configure_file(${CMAKE_SOURCE_DIR}/res/whirl.rc.in "${rc}" @ONLY)
  target_sources(${target} PRIVATE "${rc}" ${CMAKE_SOURCE_DIR}/res/whirl.manifest)
  if("ICON" IN_LIST ARGN)
    set_source_files_properties("${rc}" PROPERTIES OBJECT_DEPENDS "${WHIRL_ICON}")
    add_dependencies(${target} whirl-icon)
  endif()
  # the manifest above carries trustInfo (asInvoker); no second one from the linker
  target_link_options(${target} PRIVATE /MANIFESTUAC:NO)
endfunction()

# ---------------------------------------------------------------------------
# shared front-end of the shipped executables

if(WHIRL_WITH_HIP)
  add_library(whirl_release STATIC ${CMAKE_SOURCE_DIR}/src/release/release.cpp)
  target_include_directories(whirl_release PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
                                                  ${CMAKE_BINARY_DIR}/generated)
  target_link_libraries(whirl_release PUBLIC whirl ws2_32)

  foreach(t whirl-cli whirl-server)
    if(TARGET ${t})
      target_link_libraries(${t} PRIVATE whirl_release)
      target_compile_definitions(${t} PRIVATE WHIRL_HAVE_RELEASE=1)
      if(MSVC)
        # the driver's GPU runtime is loaded on first use, after a check that it
        # exists (plain-language message instead of the loader's dialog)
        target_link_options(${t} PRIVATE /DELAYLOAD:amdhip64_7.dll)
        target_link_libraries(${t} PRIVATE delayimp)
      endif()
    endif()
  endforeach()

  whirl_add_resources(whirl-cli "WHIRL LLM inference (command line)" ICON)
  whirl_add_resources(whirl-server "WHIRL OpenAI-compatible LLM server" ICON)
endif()
