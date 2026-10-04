# WHIRL - Windows HIP Inference for RDNA LLMs
# SPDX-License-Identifier: Apache-2.0
#
# Host code: MSVC (C++20). Device code: HIP SDK clang, one code object per
# GPU architecture, embedded into the host binary via the bin2c tool.

cmake_minimum_required(VERSION 3.24)
# The project version is the single version source: --version texts, the
# Windows VERSIONINFO resources and the release package name derive from it.
project(whirl VERSION 0.1.2 LANGUAGES CXX)
if(MSVC)
  enable_language(RC)
  # static C++ runtime (/MT): the executables need no Visual C++ redistributable
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WHIRL_WITH_HIP "Build the HIP runtime wrapper and embedded kernels" ON)
set(WHIRL_GPU_ARCHS "gfx1201;gfx1151" CACHE STRING "GPU architectures to build code objects for")
set(WHIRL_LLAMACPP_SRC "" CACHE PATH
    "Optional: path to an external llama.cpp source tree; enables the test-only chat-template oracle (never copied into WHIRL)")

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# HIP SDK

if(WHIRL_WITH_HIP)
  if(DEFINED ENV{HIP_PATH})
    file(TO_CMAKE_PATH "$ENV{HIP_PATH}" WHIRL_HIP_ROOT_DEFAULT)
  else()
    set(WHIRL_HIP_ROOT_DEFAULT "C:/Program Files/AMD/ROCm/7.2")
  endif()
  set(WHIRL_HIP_ROOT "${WHIRL_HIP_ROOT_DEFAULT}" CACHE PATH "HIP SDK root")
  string(REGEX REPLACE "/$" "" WHIRL_HIP_ROOT "${WHIRL_HIP_ROOT}")
  find_program(WHIRL_HIP_CLANG clang PATHS "${WHIRL_HIP_ROOT}/bin" NO_DEFAULT_PATH REQUIRED)
  find_library(WHIRL_AMDHIP_LIB amdhip64 PATHS "${WHIRL_HIP_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  message(STATUS "WHIRL: HIP SDK ${WHIRL_HIP_ROOT}, device compiler ${WHIRL_HIP_CLANG}")
endif()

# ---------------------------------------------------------------------------
# bin2c: embeds a binary file as a C++ byte array

add_executable(whirl-bin2c tools/bin2c/bin2c.cpp)

# ---------------------------------------------------------------------------
# device code objects

set(WHIRL_KERNEL_SOURCES ${CMAKE_SOURCE_DIR}/kernels/smoke.hip)
set(WHIRL_EMBED_SOURCES)
include(${CMAKE_SOURCE_DIR}/cmake/kernels.cmake OPTIONAL)  # kernels area

if(WHIRL_WITH_HIP)
  foreach(arch IN LISTS WHIRL_GPU_ARCHS)
    set(co "${CMAKE_BINARY_DIR}/kernels/whirl_kernels.${arch}.hsaco")
    set(cc "${CMAKE_BINARY_DIR}/kernels/whirl_kernels_${arch}.cpp")
    add_custom_command(
      OUTPUT "${co}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/kernels"
      COMMAND "${WHIRL_HIP_CLANG}" -x hip --offload-arch=${arch} --cuda-device-only
              --no-gpu-bundle-output -O3 -std=c++17
              -I "${CMAKE_SOURCE_DIR}/kernels"
              -o "${co}" "${CMAKE_SOURCE_DIR}/kernels/whirl_kernels.hip"
      DEPENDS ${CMAKE_SOURCE_DIR}/kernels/whirl_kernels.hip ${WHIRL_KERNEL_SOURCES}
      COMMENT "HIP device code object ${arch}"
      VERBATIM)
    add_custom_command(
      OUTPUT "${cc}"
      COMMAND whirl-bin2c "${co}" "${cc}" whirl_kernels_${arch}
      DEPENDS whirl-bin2c "${co}"
      COMMENT "Embedding code object ${arch}"
      VERBATIM)
    list(APPEND WHIRL_EMBED_SOURCES "${cc}")
  endforeach()
endif()

# ---------------------------------------------------------------------------
# core library

add_library(whirl STATIC
  src/common/util.cpp
  src/gguf/gguf.cpp
  src/tokenizer/unicode.cpp
  src/tokenizer/tokenizer.cpp
  src/chat/json.cpp
  src/chat/chat_template.cpp
)
target_include_directories(whirl PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(WHIRL_WITH_HIP)
  # the arch list is baked into the generated table of embedded code objects
  set(WHIRL_CO_DECLS "")
  set(WHIRL_CO_ENTRIES "")
  foreach(arch IN LISTS WHIRL_GPU_ARCHS)
    string(APPEND WHIRL_CO_DECLS
      "extern const unsigned char whirl_kernels_${arch}[];\nextern const unsigned long long whirl_kernels_${arch}_size;\n")
    string(APPEND WHIRL_CO_ENTRIES
      "    {\"${arch}\", whirl_kernels_${arch}, &whirl_kernels_${arch}_size},\n")
  endforeach()
  configure_file(src/hip/embedded_objects.cpp.in ${CMAKE_BINARY_DIR}/kernels/embedded_objects.cpp @ONLY)

  add_library(whirl_hip STATIC
    src/hip/hip_runtime.cpp
    src/hip/busy.cpp
    ${CMAKE_BINARY_DIR}/kernels/embedded_objects.cpp
    ${WHIRL_EMBED_SOURCES})
  target_include_directories(whirl_hip PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(whirl_hip SYSTEM PRIVATE "${WHIRL_HIP_ROOT}/include")
  target_compile_definitions(whirl_hip PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(whirl_hip PUBLIC "${WHIRL_AMDHIP_LIB}")
  target_compile_definitions(whirl PUBLIC WHIRL_WITH_HIP=1)
  target_link_libraries(whirl PUBLIC whirl_hip)
endif()

# ---------------------------------------------------------------------------
# tools and tests

add_executable(whirl-tool tools/whirl-tool/main.cpp)
target_link_libraries(whirl-tool PRIVATE whirl)

add_executable(whirl-tests tests/unit/unit_tests.cpp)
target_link_libraries(whirl-tests PRIVATE whirl)

if(WHIRL_LLAMACPP_SRC)
  add_subdirectory(tests/template_oracle)
endif()

# ---------------------------------------------------------------------------
# area fragments: each source area has its own cmake/<area>.cmake

include(cmake/server.cmake OPTIONAL)
include(cmake/model.cmake OPTIONAL)
include(cmake/vision.cmake OPTIONAL)  # Phase 3 (vision)
include(cmake/release.cmake OPTIONAL)  # release front-end: version, resources, icon, error messages
