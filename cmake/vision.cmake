# WHIRL vision (Phase 3): image input for qwen35 models - mmproj
# encoder (own code object, loaded on demand), image decode / preprocessing,
# the `whirl-vis` tool (vis-encode) and host tests.
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt after the core library.

if(WHIRL_WITH_HIP)
  # the encoder kernels: a separate gfx1201 code object, embedded but only
  # loaded (hipModuleLoadData) when the first image is encoded
  set(WHIRL_VISION_CO "${CMAKE_BINARY_DIR}/kernels/whirl_vision.gfx1201.hsaco")
  set(WHIRL_VISION_CC "${CMAKE_BINARY_DIR}/kernels/whirl_vision_gfx1201.cpp")
  add_custom_command(
    OUTPUT "${WHIRL_VISION_CO}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/kernels"
    COMMAND "${WHIRL_HIP_CLANG}" -x hip --offload-arch=gfx1201 --cuda-device-only
            --no-gpu-bundle-output -O3 -std=c++17
            -o "${WHIRL_VISION_CO}" "${CMAKE_SOURCE_DIR}/kernels/vision/vision.hip"
    DEPENDS ${CMAKE_SOURCE_DIR}/kernels/vision/vision.hip
    COMMENT "HIP vision code object gfx1201"
    VERBATIM)
  add_custom_command(
    OUTPUT "${WHIRL_VISION_CC}"
    COMMAND whirl-bin2c "${WHIRL_VISION_CO}" "${WHIRL_VISION_CC}" whirl_vision_gfx1201
    DEPENDS whirl-bin2c "${WHIRL_VISION_CO}"
    COMMENT "Embedding vision code object gfx1201"
    VERBATIM)

  add_library(whirl_vision STATIC
    ${CMAKE_SOURCE_DIR}/src/vision/vision.cpp
    ${CMAKE_SOURCE_DIR}/src/vision/image.cpp
    ${CMAKE_SOURCE_DIR}/src/vision/hash.cpp
    ${CMAKE_SOURCE_DIR}/src/vision/kernel_image.cpp
    ${CMAKE_SOURCE_DIR}/src/vision/vis_cli.cpp
    ${CMAKE_SOURCE_DIR}/src/vision/stb_image_impl.cpp
    "${WHIRL_VISION_CC}")
  target_include_directories(whirl_vision PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(whirl_vision PUBLIC whirl)
  if(MSVC)
    # third-party single-header library: keep its warnings out of the /W4 build
    set_source_files_properties(${CMAKE_SOURCE_DIR}/src/vision/stb_image_impl.cpp PROPERTIES COMPILE_OPTIONS "/wd4244;/wd4245;/wd4100;/wd4456;/wd4457;/wd4701;/wd4127;/wd4310;/wd4296;/wd4146;/wd4204;/wd4505")
  endif()

  # whirl-vis vis-encode MMPROJ IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]
  add_executable(whirl-vis ${CMAKE_SOURCE_DIR}/src/vision/vis_main.cpp)
  target_link_libraries(whirl-vis PRIVATE whirl_vision)

  # host tests (no GPU): preprocessing, hashes, placeholder expansion, M-RoPE map
  add_executable(whirl-vision-tests ${CMAKE_SOURCE_DIR}/tests/vision/vision_tests.cpp)
  target_link_libraries(whirl-vision-tests PRIVATE whirl_vision)

  # image input in the CLI (whirl chat --mmproj --image, whirl vis-encode) and the
  # server (image_url parts): the executables of the model / server areas link the
  # vision library when this fragment is present
  # the server engine takes image_url parts (placeholder expansion, embedding cache)
  if(TARGET whirl_server_core)
    target_link_libraries(whirl_server_core PUBLIC whirl_vision)
  endif()
  foreach(t whirl-cli whirl-server)
    if(TARGET ${t})
      target_link_libraries(${t} PRIVATE whirl_vision)
      target_compile_definitions(${t} PRIVATE WHIRL_HAVE_VISION=1)
    endif()
  endforeach()
endif()
