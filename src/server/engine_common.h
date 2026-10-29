// Server engine, the names every engine file shares (arch025-engine S3, module server.engine_types; moved verbatim
// out of engine.h and engine_internal.h): the aliases of the model / prefill-schedule / checkpoint constants and the
// sampler types, and KvPoolFull (arch025-N4f: one definition, so a KvPoolFull thrown in one file is the type another
// file's handler sees). Includes are spelled from src/ (server/...): a header-only isolation mutant shadows this
// file, and engine.h / engine_options.h / engine_state.h must reach the same copy (tests/iso/server.engine_types).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cache/prefix_cache.h"       // cache::ckpt_keep_prefix_min
#include "sampling/sampler.h"         // sampling::Cand, sampling::Sampler
#include "whirl/model_types.h"        // qwen35::kv_page, qwen35::gdn_max_seg
#include "whirl/prefill_schedule.h"   // sched::prefill_chunk .. prefillExecDefault
#include "whirl/small_helpers.h"      // whirl::msSince

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace whirl::server {

using whirl::msSince;  // whirl/small_helpers.h (arch025-N6)

constexpr std::uint32_t kv_page = qwen35::kv_page;
constexpr std::uint32_t gdn_max_seg = qwen35::gdn_max_seg;
// the cold prefill schedule (shared with whirl seqtest's gate: include/whirl/prefill_schedule.h)
using sched::prefill_chunk;
using sched::prefill_merge;
// TAIL-CK: per-slot checkpoints needed for the tail checkpoint (tail + prompt-end + gen-end); with
// fewer, a tail would evict what a regenerate or the next chat turn resumes from, so none is taken
constexpr std::size_t tail_ck_slots = cache::ckpt_keep_prefix_min;
using sched::prefillChunkLen;
using sched::prefillExecFor;
using sched::prefill_exec_default;
using sched::prefillExecDefault;

// sampling (arch025-E1a): the sampler and its candidates live in src/sampling
using sampling::Cand;
using sampling::Sampler;

struct KvPoolFull : std::runtime_error {
    KvPoolFull() : std::runtime_error("KvPoolFull") {}
};

}  // namespace whirl::server
