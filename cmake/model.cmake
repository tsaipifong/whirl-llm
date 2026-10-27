# WHIRL model: weight loader, qwen35 family module (src/model/qwen35/: forward, weights, KV, scratch, the ServerModel,
# the architecture table row; the registry itself: cmake/arch.cmake), MTP drafting (the draft policy: cmake/spec.cmake),
# autotune, and the `whirl` CLI (chat / bench / selftest).
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt after the core library.

if(WHIRL_WITH_HIP)
  # default MTP draft-head vocabulary subset (64k token ids, uint32 LE), embedded
  # so the release does not depend on a file next to the exe (WHIRL_DRAFT_VOCAB)
  set(WHIRL_DRAFT_VOCAB_BIN "${CMAKE_SOURCE_DIR}/data/draft_vocab/subset_64k.bin")
  set(WHIRL_DRAFT_VOCAB_CC "${CMAKE_BINARY_DIR}/generated/draft_vocab_64k.cpp")
  add_custom_command(
    OUTPUT "${WHIRL_DRAFT_VOCAB_CC}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/generated"
    COMMAND whirl-bin2c "${WHIRL_DRAFT_VOCAB_BIN}" "${WHIRL_DRAFT_VOCAB_CC}" whirl_draft_vocab_64k
    DEPENDS whirl-bin2c "${WHIRL_DRAFT_VOCAB_BIN}"
    COMMENT "Embedding draft vocabulary subset 64k"
    VERBATIM)

  add_library(whirl_model STATIC
    ${CMAKE_SOURCE_DIR}/src/config/env_snapshot.cpp
    ${CMAKE_SOURCE_DIR}/src/model/config.cpp
    ${CMAKE_SOURCE_DIR}/src/model/kernels.cpp
    ${CMAKE_SOURCE_DIR}/src/model/loader.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/weights.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/kv.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/scratch.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/tune_table.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/layer_subset.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/module.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/run.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/mtp_verify.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/prefill.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/vision_rows.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/dispatch.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/attn.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/ffn.cpp
    ${CMAKE_SOURCE_DIR}/src/model/qwen35/forward/gdn.cpp
    ${CMAKE_SOURCE_DIR}/src/model/tune.cpp
    ${CMAKE_SOURCE_DIR}/src/model/exec_dump.cpp
    ${CMAKE_SOURCE_DIR}/src/model/device.cpp
    ${CMAKE_SOURCE_DIR}/src/model/dflash.cpp
    ${CMAKE_SOURCE_DIR}/src/model/draft_vocab_embed.cpp
    "${WHIRL_DRAFT_VOCAB_CC}")
  target_include_directories(whirl_model PUBLIC ${CMAKE_SOURCE_DIR}/include)
  # whirl_device: device::hipBackend() and the src/device RAII handles that own the model buffers (arch024 S11)
  # whirl_spec: the MTP / n-gram draft policy and mtpDefaults (src/spec, arch025-M4; was src/model/spec.cpp)
  # whirl_arch: the architecture registry (findArch / findArchName, arch025-M5; the table itself: src/model/qwen35/module.cpp)
  target_link_libraries(whirl_model PUBLIC whirl_kernels_host whirl whirl_ops whirl_device whirl_spec whirl_arch)

  # the CLI executable is whirl.exe (target name differs from the core library)
  # arch025-C2: one file per command (src/cli/cmd_*.cpp), the shared helpers, the CLI's model building (src/cli/cli.h)
  add_executable(whirl-cli
    ${CMAKE_SOURCE_DIR}/src/cli/main.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/common.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/build_model.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/decode.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_chat.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_bench.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_selftest.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_seqtest.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_golden.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_dflash_check.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_devices.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_serve.cpp
    ${CMAKE_SOURCE_DIR}/src/cli/cmd_vis.cpp)
  set_target_properties(whirl-cli PROPERTIES OUTPUT_NAME whirl)
  target_link_libraries(whirl-cli PRIVATE whirl_model)
  # `whirl serve` forwards to the server's serveMain (cmake/server.cmake)
  if(TARGET whirl_server_core)
    target_sources(whirl-cli PRIVATE
      ${CMAKE_SOURCE_DIR}/src/server/server_main.cpp
      ${CMAKE_SOURCE_DIR}/src/model/qwen35/server_model.cpp)
    target_link_libraries(whirl-cli PRIVATE whirl_server_core whirl_device)
    target_compile_definitions(whirl-cli PRIVATE WHIRL_HAVE_SERVER=1)
  endif()

  # host-only unit tests of the model library (no GPU work)
  add_executable(whirl-model-tests ${CMAKE_SOURCE_DIR}/tests/model/model_tests.cpp)
  # arch025-M1: whirl_device_null (NullBackend: host memory behind WeightLoader in testWeightBinding)
  target_link_libraries(whirl-model-tests PRIVATE whirl_model whirl_device_null)
  # OVH-C1 test reads src/model/qwen35/forward/*.cpp (both MTP call sites use Model::draftAttn; forward.cpp split by arch025-F1..F4,
  # moved under src/model/qwen35/ by M5)
  target_compile_definitions(whirl-model-tests PRIVATE WHIRL_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

  # measure-tools P4: whirl-stage-bench (forward stages at a synthetic KV / state depth; test executable, GPU).
  # Its HIP-free part (stage_bench.cpp) is unit-tested by whirl-stage-bench-tests (cmake/server.cmake).
  add_executable(whirl-stage-bench
    ${CMAKE_SOURCE_DIR}/tests/bench/stage_bench_main.cpp
    ${CMAKE_SOURCE_DIR}/tests/bench/stage_bench.cpp
    ${CMAKE_SOURCE_DIR}/tests/kernels/kt_bench.cpp)
  target_link_libraries(whirl-stage-bench PRIVATE whirl_model)
  target_compile_definitions(whirl-stage-bench PRIVATE WHIRL_SB_VERSION="${PROJECT_VERSION}")
endif()
