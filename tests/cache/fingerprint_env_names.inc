// arch025-i6vis (rule I6, X4 K4): every WHIRL_* name the source reads that the SSD tier fingerprint DOES hash.
// tests/cache/cache_tests.cpp testFingerprintEnvClassified() scans src/ and include/ for WHIRL_* names; each one must be
// either skipped by fingerprintEnvKept() (src/cache/fingerprint.cpp: test / debug / dump / size / path switches that
// change no kernel and no KV format) or listed here (it can change what is computed or stored, so a cache written
// without it must not be restored with it). A new name in neither place fails the test: decide which, in the same
// commit (I6). A name here that the source no longer reads also fails (the list cannot go stale).
// SPDX-License-Identifier: Apache-2.0
#pragma once

static const char* const k_fp_hashed[] = {
    "WHIRL_ACT_FUSE",  // src/common/numerics.cpp
    "WHIRL_ATTN_DQ4",  // src/common/numerics.cpp
    "WHIRL_ATTN_DQF",  // src/common/numerics.cpp
    "WHIRL_ATTN_KG",  // src/common/numerics.cpp
    "WHIRL_ATTN_KX",  // src/common/numerics.cpp
    "WHIRL_ATTN_SEG",  // src/common/numerics.cpp
    "WHIRL_ATTN_WIDE",  // src/common/numerics.cpp
    "WHIRL_ATTN_WS4",  // src/common/numerics.cpp
    "WHIRL_BENCH_MAX_BATCH",  // src/cli/cmd_bench.cpp
    "WHIRL_CODE_OBJECT",  // src/config/env_snapshot.cpp
    "WHIRL_DBG",  // src/common/numerics.cpp
    "WHIRL_DEVICE",  // src/config/env_snapshot.cpp
    "WHIRL_DFLASH_CAPTURE_ROWS",  // include/whirl/dflash_policy.h
    "WHIRL_DFLASH_NMIN",  // include/whirl/dflash_policy.h
    "WHIRL_DFLASH_PMIN",  // include/whirl/dflash_policy.h
    "WHIRL_DFLASH_WARM",  // include/whirl/dflash_policy.h
    "WHIRL_DQ4_DYN",  // src/common/numerics.cpp
    "WHIRL_DRAFT_HEAD",  // src/cli/cmd_bench.cpp
    "WHIRL_DRAFT_MODEL",  // src/cli/build_model.cpp
    "WHIRL_DRAFT_VOCAB",  // src/cli/build_model.cpp
    "WHIRL_DRAFT_WINDOW",  // src/common/numerics.cpp
    "WHIRL_DRAFT_WINDOW_MIN",  // src/common/numerics.cpp
    "WHIRL_FD_SPLITS",  // src/config/env_snapshot.cpp
    "WHIRL_FFN_H16",  // src/common/numerics.cpp
    "WHIRL_FLOAT_GEMV",  // src/common/numerics.cpp
    "WHIRL_FP8",  // src/common/numerics.cpp
    "WHIRL_FP8_MASK",  // src/common/numerics.cpp
    "WHIRL_G8T",  // src/common/numerics.cpp
    "WHIRL_GDN_BA",  // src/common/numerics.cpp
    "WHIRL_GDN_IN2",  // src/common/numerics.cpp
    "WHIRL_GDN_REPLAY",  // src/cli/cmd_golden.cpp
    "WHIRL_GDN_SEQ",  // src/common/numerics.cpp
    "WHIRL_GDN_STRIX",  // src/common/numerics.cpp
    "WHIRL_GDN_STRIX_MIN",  // src/common/numerics.cpp
    "WHIRL_GDN_V0",  // src/common/numerics.cpp
    "WHIRL_GDN_WMMA",  // src/common/numerics.cpp
    "WHIRL_GEMMH",  // src/common/numerics.cpp
    "WHIRL_GEMMHQ",  // src/common/numerics.cpp
    "WHIRL_GEMVH_WAVES",  // src/kernels_host/kernels_abi.cpp
    "WHIRL_GEMVX_VARIANT",  // src/kernels_host/kernels_abi.cpp
    "WHIRL_GEMV_BITWISE_FLOAT",  // src/config/env_snapshot.cpp
    "WHIRL_GEMV_MAX",  // src/common/numerics.cpp
    "WHIRL_GEMV_R",  // src/config/env_snapshot.cpp
    "WHIRL_GEMV_W",  // src/config/env_snapshot.cpp
    "WHIRL_GEMV_WH",  // src/config/env_snapshot.cpp
    "WHIRL_GOV_HOLD_S",  // src/server/server_main.cpp
    "WHIRL_GOV_RECOVER_S",  // src/server/server_main.cpp
    "WHIRL_GPU_RESTRICT",  // src/hip/gpu_restrict.cpp
    "WHIRL_GV_GROUP",  // src/common/numerics.cpp
    "WHIRL_GV_NMAX",  // src/common/numerics.cpp
    "WHIRL_HEAD_CHUNK",  // src/common/numerics.cpp
    "WHIRL_HEAD_GEMVX",  // src/config/env_snapshot.cpp
    "WHIRL_HIP_DEVICE",  // src/cli/build_model.cpp
    "WHIRL_KV",  // src/common/numerics.cpp
    "WHIRL_KV_IDLE_FLUSH_MS",  // src/server/server_main.cpp
    "WHIRL_KV_SSD_DEDUP",  // src/server/server_main.cpp
    "WHIRL_KV_SSD_EXPIRE_S",  // src/server/server_main.cpp
    "WHIRL_KV_SSD_MODE",  // src/server/server_main.cpp
    "WHIRL_KV_SUPERSEDE",  // src/server/server_main.cpp
    "WHIRL_MAX_CTX",  // src/cli/cmd_chat.cpp
    "WHIRL_MMPROJ",  // src/server/server_main.cpp
    "WHIRL_MODE",  // src/cli/build_model.cpp
    "WHIRL_MOE_BN",  // src/common/numerics.cpp
    "WHIRL_MOE_FP8",  // src/common/numerics.cpp
    "WHIRL_MOE_MXW",  // src/common/numerics.cpp
    "WHIRL_MOE_RBF",  // src/common/numerics.cpp
    "WHIRL_MOE_WIDE",  // src/common/numerics.cpp
    "WHIRL_MTP",  // src/cli/cmd_bench.cpp
    "WHIRL_MTP_ADAPT",  // src/cli/decode.cpp
    "WHIRL_MTP_BATCH_DRAFTS",  // src/server/server_main.cpp
    "WHIRL_MTP_DRAFTS",  // src/cli/decode.cpp
    "WHIRL_MTP_FULLHEAD",  // src/cli/cmd_bench.cpp
    "WHIRL_MTP_NMIN",  // src/cli/decode.cpp
    "WHIRL_MTP_PMIN",  // src/cli/decode.cpp
    "WHIRL_MTP_Q4",  // src/cli/cmd_bench.cpp
    "WHIRL_NAIVE_ATTN",  // src/common/numerics.cpp
    "WHIRL_NGRAM",  // src/cli/decode.cpp
    "WHIRL_NGRAM_FORCE",  // src/cli/decode.cpp
    "WHIRL_NGRAM_MAX",  // src/cli/decode.cpp
    "WHIRL_NGRAM_MIN",  // src/cli/decode.cpp
    "WHIRL_NGRAM_SLOPE",  // src/cli/decode.cpp
    "WHIRL_NO_FUSE",  // src/common/numerics.cpp
    "WHIRL_NO_GRAPH",  // src/common/numerics.cpp
    "WHIRL_POOL_BUDGET_MARGIN_MB",  // src/server/server_main.cpp
    "WHIRL_POOL_RESERVE_MB",  // src/server/server_main.cpp
    "WHIRL_PREC_GEMVH",  // src/config/env_snapshot.cpp
    "WHIRL_PREC_GS",  // src/config/env_snapshot.cpp
    "WHIRL_PREFILL_BATCH",  // src/cli/build_model.cpp
    "WHIRL_PREFILL_CHUNK",  // src/cli/common.cpp
    "WHIRL_Q3K_U16",  // src/kernels_host/kernels_abi.cpp
    "WHIRL_Q4_RELAXED",  // src/common/numerics.cpp
    "WHIRL_Q8DEC",  // src/common/numerics.cpp
    "WHIRL_RELAX",  // src/common/numerics.cpp
    "WHIRL_RELAX_ALPHA",  // src/common/numerics.cpp
    "WHIRL_RELAX_DELTA",  // src/common/numerics.cpp
    "WHIRL_RELAX_EPS",  // src/common/numerics.cpp
    "WHIRL_RELAX_K",  // src/common/numerics.cpp
    "WHIRL_SERVE_CKPTS",  // src/server/server_main.cpp
    "WHIRL_SLOT_DRAFTS",  // src/server/server_main.cpp
    "WHIRL_SNAP_SETS",  // src/server/server_main.cpp
    "WHIRL_SPEC_SAMPLE",  // src/common/numerics.cpp
    "WHIRL_SYS_CKPTS",  // src/server/server_main.cpp
    "WHIRL_SYS_MIN",  // src/cli/common.cpp
    "WHIRL_TAIL_CK",  // src/cli/common.cpp
    "WHIRL_THINK",  // src/cli/common.cpp
    "WHIRL_TUNE_COLD",  // src/cli/build_model.cpp
    "WHIRL_TUNE_MASK",  // src/cli/build_model.cpp
    "WHIRL_VIS_CACHE_MB",  // src/server/server_main.cpp
    "WHIRL_VIS_CKPT_MIN",  // src/cli/common.cpp
    "WHIRL_VIS_IDLE_S",  // src/server/server_main.cpp
    "WHIRL_VIS_MODE",  // src/cli/cmd_chat.cpp
    "WHIRL_VIS_PRE",  // src/vision/vis_cli.cpp
    "WHIRL_VRAM_HEADROOM_MB",  // src/server/server_main.cpp
    "WHIRL_VRAM_LIMIT_MB",  // src/common/vram_limit.cpp
    "WHIRL_WIDE_VERIFY",  // src/common/numerics.cpp
    // pre-0.2.5 diagnostics that are still hashed, as before this change (moving one to the skip list changes the
    // fingerprint of a user who sets it: one cache miss). Follow-up candidates for the skip list, not changed here.
    "WHIRL_DFLASH_DUMP",  // src/cli/cmd_chat.cpp
    "WHIRL_DFLASH_PROF",  // src/config/env_snapshot.cpp
    "WHIRL_DUMP_AT",  // src/cli/cmd_chat.cpp
    "WHIRL_DUMP_FROM",  // src/cli/cmd_chat.cpp
    "WHIRL_DUMP_LOGITS",  // src/cli/cmd_chat.cpp
    "WHIRL_DUMP_MOE",  // src/cli/cmd_chat.cpp
    "WHIRL_DUMP_ROWS",  // src/cli/cmd_chat.cpp
    "WHIRL_FAKE_DEDICATED_GB",  // src/hip/uma_mem.cpp
    "WHIRL_LOAD_DEBUG",  // src/cli/build_model.cpp
    "WHIRL_NGRAM_DEBUG",  // src/cli/decode.cpp
    "WHIRL_PRINT_IDS",  // src/cli/cmd_chat.cpp
    "WHIRL_TOKENIZE_ONLY",  // src/cli/cmd_chat.cpp
    "WHIRL_TRACE_TPS",  // src/cli/cmd_chat.cpp
    "WHIRL_UMA_CHECK",  // src/hip/uma_mem.cpp
    "WHIRL_VIS_DUMP",  // src/cli/cmd_chat.cpp
    "WHIRL_VIS_DUMP_RGB",  // src/vision/vis_cli.cpp
    "WHIRL_VIS_PROF",  // src/vision/vis_cli.cpp
    // arch025-lane1 leg 5 (N10): server_main read these through its local envOn() wrapper, a call form the scanner
    // does not match, so the check never saw them. fingerprintEnv() hashed every one of them before and after N10
    // (it hashes every set WHIRL_* outside the skip list); listed here as hashed, i.e. the fingerprint is unchanged.
    "WHIRL_CKPT_HOST",  // src/server/server_main.cpp
    "WHIRL_COLD_RESUME",  // src/server/server_main.cpp
    "WHIRL_EMBD_HOST",  // src/server/server_main.cpp
    "WHIRL_FIT_CHECK",  // src/server/server_main.cpp
    "WHIRL_GOVERNOR",  // src/server/server_main.cpp
    "WHIRL_HIP_CACHE_LIMIT",  // src/server/server_main.cpp
    "WHIRL_NO_PREFIX_CACHE",  // src/server/server_main.cpp
    "WHIRL_SEG_PREFILL",  // src/server/server_main.cpp
    "WHIRL_SYS_LCP",  // src/server/server_main.cpp
    "WHIRL_TIMING_RESET",  // src/server/server_main.cpp
};
