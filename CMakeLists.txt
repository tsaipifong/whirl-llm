# WHIRL - Windows HIP Inference for RDNA LLMs
# SPDX-License-Identifier: Apache-2.0
#
# Host code: MSVC (C++20). Device code: HIP SDK clang, one code object per
# GPU architecture, embedded into the host binary via the bin2c tool.

cmake_minimum_required(VERSION 3.24)
# The project version is the single version source: --version texts, the
# Windows VERSIONINFO resources and the release package name derive from it.
project(whirl VERSION 0.2.4 LANGUAGES CXX)
if(MSVC)
  enable_language(RC)
  # static C++ runtime (/MT): the executables need no Visual C++ redistributable
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WHIRL_WITH_HIP "Build the HIP runtime wrapper and embedded kernels" ON)
set(WHIRL_GPU_ARCHS "gfx1201;gfx1151" CACHE STRING "GPU architectures to build code objects for (gfx1201 = R9700, gfx1151 = Radeon 8060S)")
set(WHIRL_LLAMACPP_SRC "" CACHE PATH
    "Optional: path to an external llama.cpp source tree; enables the test-only chat-template oracle (never copied into WHIRL)")

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

include(cmake/safety.cmake)  # safe025: sanitizer / leak check / static analysis options (default OFF)

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
  # no amdhip64.lib: the driver's runtime is loaded at run time (src/hip/runtime_loader.cpp, ARCH-1 D-1)
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
    if(NOT arch MATCHES "^(gfx1201|gfx1151)$")
      message(FATAL_ERROR "WHIRL_GPU_ARCHS: unsupported architecture '${arch}' (gfx1201, gfx1151)")
    endif()
    set(co "${CMAKE_BINARY_DIR}/kernels/whirl_kernels.${arch}.hsaco")
    # per-arch dependency list from cmake/kernels.cmake (falls back to all sources)
    set(co_deps ${WHIRL_KERNEL_SOURCES})
    if(DEFINED WHIRL_KERNEL_SOURCES_${arch})
      set(co_deps ${WHIRL_KERNEL_SOURCES_${arch}})
    endif()
    set(cc "${CMAKE_BINARY_DIR}/kernels/whirl_kernels_${arch}.cpp")
    add_custom_command(
      OUTPUT "${co}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/kernels"
      COMMAND "${WHIRL_HIP_CLANG}" -x hip --offload-arch=${arch} --cuda-device-only
              --no-gpu-bundle-output -O3 -std=c++17
              -I "${CMAKE_SOURCE_DIR}/kernels"
              -Xclang -dependency-file -Xclang "${co}.d" -Xclang -MT -Xclang "${co}"
              -o "${co}" "${CMAKE_SOURCE_DIR}/kernels/whirl_kernels.hip"
      # DEPFILE: clang lists every fragment the translation unit really
      # includes (kernels/*.hip, gfx1151/*.hip, *.h), so an edit to any of them
      # rebuilds this code object; DEPENDS stays as the first-build fallback.
      # The cc1 options are used because the driver's -MD/-MF are ignored in
      # --cuda-device-only mode (clang warns "argument unused").
      DEPFILE "${co}.d"
      DEPENDS ${CMAKE_SOURCE_DIR}/kernels/whirl_kernels.hip ${co_deps}
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
  src/common/vram_limit.cpp
  src/common/wddm_watch.cpp
  src/common/pool_budget.cpp
  src/common/startup_fit.cpp
  src/common/gpu_select.cpp
  src/common/mem_governor.cpp
  src/common/numerics.cpp
  src/common/dump.cpp
  src/common/spec_sample.cpp
  src/gguf/gguf.cpp
  src/tokenizer/unicode.cpp
  src/tokenizer/tokenizer.cpp
  src/chat/json.cpp
  src/chat/chat_template.cpp
)
target_include_directories(whirl PUBLIC ${CMAKE_SOURCE_DIR}/include)
# platform layer (arch025 PL0): linked now so PL1-PL4 only add sources to cmake/platform.cmake
include(cmake/platform.cmake)
target_link_libraries(whirl PUBLIC whirl_platform)

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

  # HIP runtime loader (ARCH-1 D-1): LoadLibrary + one function table, no import of amdhip64_7.dll.
  # Its own library so its unit tests (whirl-hiprt-tests) link nothing else.
  add_library(whirl_hiprt STATIC src/hip/runtime_loader.cpp)
  target_include_directories(whirl_hiprt PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(whirl_hiprt SYSTEM PRIVATE "${WHIRL_HIP_ROOT}/include")
  target_compile_definitions(whirl_hiprt PRIVATE __HIP_PLATFORM_AMD__)

  add_library(whirl_hip STATIC
    src/hip/hip_runtime.cpp
    src/hip/alloc_log.cpp
    src/hip/launch_log.cpp
    src/hip/wddm_mem.cpp
    src/hip/device_mem.cpp
    src/hip/uma_mem.cpp
    src/hip/gpu_restrict.cpp
    ${CMAKE_BINARY_DIR}/kernels/embedded_objects.cpp
    ${WHIRL_EMBED_SOURCES})
  target_include_directories(whirl_hip PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(whirl_hip SYSTEM PRIVATE "${WHIRL_HIP_ROOT}/include")
  target_compile_definitions(whirl_hip PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(whirl_hip PUBLIC whirl_hiprt)
  # arch025 PL1b: env reads of alloc_log / launch_log / uma_mem / gpu_restrict go through platform::env (bottom layer)
  target_link_libraries(whirl_hip PRIVATE whirl_platform)
  if(WIN32)
    target_link_libraries(whirl_hip PUBLIC dxgi pdh)
  endif()
  target_compile_definitions(whirl PUBLIC WHIRL_WITH_HIP=1)
  target_link_libraries(whirl PUBLIC whirl_hip)
endif()

# ---------------------------------------------------------------------------
# tools and tests

add_executable(whirl-tool tools/whirl-tool/main.cpp)
target_link_libraries(whirl-tool PRIVATE whirl)
if(WHIRL_WITH_HIP)
  # arch024-S14: another program holding VRAM / pinned memory (tools/gates/env_matrix.ps1 E1 / E4)
  add_executable(whirl-mem-hog tools/mem-hog/main.cpp)
  target_link_libraries(whirl-mem-hog PRIVATE whirl)
endif()

add_executable(whirl-tests tests/unit/unit_tests.cpp)
target_link_libraries(whirl-tests PRIVATE whirl)
# GPU runtime load-failure diagnosis (release front-end, issue #2), CPU-only
target_sources(whirl-tests PRIVATE src/release/dll_deps.cpp)
target_include_directories(whirl-tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(whirl-tests PRIVATE WHIRL_TEST_DLL_DEPS=1)

if(WHIRL_WITH_HIP)
  # HIP runtime loader tests (ARCH-1 D-1): load order, absent / fake DLLs, 126 / 127, a missing HIP function.
  # The fake runtimes go to a subfolder so neither the test's own folder nor the current folder holds their imports.
  set(WHIRL_FAKE_HIPRT_DIR "${CMAKE_BINARY_DIR}/hiprt_fakes")
  add_library(whirl-fake-dep SHARED tests/hiprt/fake_dep.cpp)
  add_library(whirl-fake-hiprt SHARED tests/hiprt/fake_hiprt.cpp)
  add_library(whirl-fake-hiprt-drop SHARED tests/hiprt/fake_hiprt.cpp)
  target_compile_definitions(whirl-fake-hiprt-drop PRIVATE WHIRL_FAKE_DROP_LAST=1)
  add_library(whirl-fake-hiprt-dep SHARED tests/hiprt/fake_hiprt.cpp)
  target_compile_definitions(whirl-fake-hiprt-dep PRIVATE WHIRL_FAKE_WITH_DEP=1)
  target_link_libraries(whirl-fake-hiprt-dep PRIVATE whirl-fake-dep)
  foreach(t whirl-fake-dep whirl-fake-hiprt whirl-fake-hiprt-drop whirl-fake-hiprt-dep)
    string(REPLACE "-" "_" n ${t})
    set_target_properties(${t} PROPERTIES OUTPUT_NAME ${n} RUNTIME_OUTPUT_DIRECTORY "${WHIRL_FAKE_HIPRT_DIR}")
  endforeach()
  add_executable(whirl-hiprt-tests tests/hiprt/hiprt_tests.cpp src/release/dll_deps.cpp)
  target_link_libraries(whirl-hiprt-tests PRIVATE whirl_hiprt)
  target_include_directories(whirl-hiprt-tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_include_directories(whirl-hiprt-tests SYSTEM PRIVATE "${WHIRL_HIP_ROOT}/include")
  target_compile_definitions(whirl-hiprt-tests PRIVATE __HIP_PLATFORM_AMD__)
  add_dependencies(whirl-hiprt-tests whirl-fake-dep whirl-fake-hiprt whirl-fake-hiprt-drop whirl-fake-hiprt-dep)
endif()

if(WHIRL_LLAMACPP_SRC)
  add_subdirectory(tests/template_oracle)
endif()

# ---------------------------------------------------------------------------
# area fragments: each source area has its own cmake/<area>.cmake

include(cmake/ops.cmake)  # kernel choice per op (ARCH-1 K-1..K-5)
include(cmake/server.cmake OPTIONAL)
include(cmake/safety_targets.cmake)  # safe025: test-only fault driver (default OFF)
include(cmake/model.cmake OPTIONAL)
include(cmake/vision.cmake OPTIONAL)  # Phase 3 (vision)
include(cmake/release.cmake OPTIONAL)  # release front-end: version, resources, icon, error messages
