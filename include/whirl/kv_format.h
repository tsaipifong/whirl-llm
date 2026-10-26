// The KV cache format (arch025-N3, R-6): one enum from the user's request (WHIRL_KV, LoadOptions::kv_req) through the
// choice (numerics::chooseKv) to the kernel table (kernels::KernelTable::load) and the pool layout (kv::layerRows).
// Before N3 the same five values were three enums (KvMode with an extra "automatic", numerics::KvKind,
// kernels::KvFormat) plus four bools on the Model (kv_q8, kv_rot, kv_kf16, kv_q4) that loadKernels turned back into
// the enum. A request is std::optional<KvFormat>: nullopt = the numerics mode decides (was KvMode::automatic).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Same shape as llama.cpp's one ggml_type for cache_type_k / _v (common/arg.cpp
// kv_cache_type_from_str, common.h cache_type_k, llama.h type_k) with ggml_is_quantized() as a predicate of the type.

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace whirl {

// f16; q8 = int8 + f16 scale per 32 values (K and V); q8h = q8 with Hadamard-rotated q / k; q8v = K f16, V q8;
// q4 = 4-bit K and V (+ f16 scale per 32 values) with Hadamard-rotated q / k.
enum class KvFormat : std::uint8_t { f16, q8, q8h, q8v, q4 };

// "f16" | "q8" | "q8h" | "q8v" | "q4"
constexpr const char* kvFormatName(KvFormat k) {
    switch (k) {
        case KvFormat::f16: return "f16";
        case KvFormat::q8: return "q8";
        case KvFormat::q8h: return "q8h";
        case KvFormat::q8v: return "q8v";
        case KvFormat::q4: return "q4";
    }
    return "f16";
}

// the inverse of kvFormatName (nullopt for anything else, including "auto")
constexpr std::optional<KvFormat> kvFormatFromName(std::string_view s) {
    if (s == "f16") return KvFormat::f16;
    if (s == "q8") return KvFormat::q8;
    if (s == "q8h") return KvFormat::q8h;
    if (s == "q8v") return KvFormat::q8v;
    if (s == "q4") return KvFormat::q4;
    return std::nullopt;
}

// Properties of a format (the Model's former bools; like ggml_is_quantized):
// not f16: V (and K except q8v) are int8 / 4-bit with an f16 scale per 32 values (was Model::kv_q8)
constexpr bool kvQuant(KvFormat k) { return k != KvFormat::f16; }
// q / k Hadamard-rotated (q8h, q4; was Model::kv_rot)
constexpr bool kvRotated(KvFormat k) { return k == KvFormat::q8h || k == KvFormat::q4; }
// K stays f16 (q8v; was Model::kv_kf16)
constexpr bool kvKf16(KvFormat k) { return k == KvFormat::q8v; }
// 4-bit K and V (was Model::kv_q4)
constexpr bool kvIs4bit(KvFormat k) { return k == KvFormat::q4; }

}  // namespace whirl
