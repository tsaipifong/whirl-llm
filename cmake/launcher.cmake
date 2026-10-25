# WHIRL - Windows launcher (src/launcher, feat026-launcher, 0.2.6). See docs/dev/launcher.md.
# SPDX-License-Identifier: Apache-2.0
#
# whirl_launcher: the decision logic (no OS call, no HIP): which start runs the launcher, the settings file, the
# recommendations, the server command line, the chat stream parsing. The interactive flow (launcher_main.cpp) is a
# source of whirl.exe only; its OS calls are in src/platform/shell_win.cpp and http_client_win.cpp
# (cmake/platform.cmake). No other executable links it.

add_library(whirl_launcher STATIC ${CMAKE_SOURCE_DIR}/src/launcher/launcher_logic.cpp)
target_include_directories(whirl_launcher PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(whirl_launcher PUBLIC whirl)

if(TARGET whirl-cli)
  target_sources(whirl-cli PRIVATE ${CMAKE_SOURCE_DIR}/src/launcher/launcher_main.cpp)
  target_link_libraries(whirl-cli PRIVATE whirl_launcher)
  target_compile_definitions(whirl-cli PRIVATE WHIRL_HAVE_LAUNCHER=1)
  # the launcher's system DLLs load only when the launcher runs: every other whirl.exe command keeps the DLL set of
  # 0.2.5 (dxgi, pdh, ws2_32, kernel32, winmm; checked with dumpbin /dependents, docs/dev/launcher.md)
  if(MSVC)
    target_link_options(whirl-cli PRIVATE /DELAYLOAD:WINHTTP.dll /DELAYLOAD:SHELL32.dll /DELAYLOAD:ole32.dll /DELAYLOAD:USER32.dll)
    target_link_libraries(whirl-cli PRIVATE delayimp)
  endif()
endif()

# CPU tests of the decision logic (no GPU, no console needed)
add_executable(whirl-launcher-tests ${CMAKE_SOURCE_DIR}/tests/launcher/launcher_tests.cpp)
target_link_libraries(whirl-launcher-tests PRIVATE whirl_launcher)
