// Image input for qwen35 models: the Qwen3-VL style vision encoder
// ("qwen3vl_merger" mmproj GGUF, F16 / BF16 / F32 weights) on HIP, on demand.
// SPDX-License-Identifier: Apache-2.0
//
// Design (ported from the research prototype, item VIS):
// - The mmproj weights are read once into pinned host memory (repacked: the two
//   temporal patch kernels side by side, ffn_down padded to K = n_ff_pad).
// - Nothing of the encoder lives in VRAM until an image is encoded: the kernels
//   are a separate code object (loaded on first use) and the weights are either
//   uploaded and kept resident (when enough VRAM is free) or streamed layer by
//   layer into a double buffer (default when the KV pool fills VRAM).
// - Activations use device regions the caller lends (the LM's prefill scratch,
//   idle between forwards on the same stream), so an image needs no extra VRAM.
// - After idle_s seconds without images the module and resident weights are
//   released again (VRAM back to the text-only baseline).
//
// Graph and preprocessing follow llama.cpp tools/mtmd (MIT): clip.cpp (qwen3vl),
// models/qwen3vl.cpp, mtmd-image.cpp. Image decoding: stb_image (public domain /
// MIT). See THIRD_PARTY_NOTICES.md.

#pragma once

#include "whirl/hip.h"
#include "whirl/vismap.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::vision {

using hip::DevPtr;

// Errors carry a code: "ImageDecode", "ImageTooLarge", "NotAnMmproj",
// "UnsupportedProjector", "UnsupportedVisionShape", "DeepstackUnsupported",
// "MissingTensor", "UnsupportedTensorType", "OutputTooSmall",
// "ImagePlaceholderMismatch", "TooManyImages".
class VisionError : public std::runtime_error {
public:
    VisionError(std::string code, const std::string& detail = {})
        : std::runtime_error(detail.empty() ? code : code + ": " + detail), code_(std::move(code)) {}
    const std::string& code() const { return code_; }

private:
    std::string code_;
};

constexpr std::uint32_t E = 1152;
constexpr std::uint32_t n_head = 16;
constexpr std::uint32_t head_dim = 72;
constexpr std::uint32_t head_pad = 80;
constexpr std::uint32_t QKV = 3 * E;
constexpr std::uint32_t patch_k = 2 * 3 * 16 * 16;  // two temporal kernels side by side

// ---------------------------------------------------------------------------
// image decoding and preprocessing (src/vision/image.cpp)

struct Rgb {
    std::uint32_t w = 0, h = 0;
    std::vector<std::uint8_t> px;  // w * h * 3 bytes, row-major RGB
};

// PNG / JPEG / BMP / GIF (first frame) / TGA / PSD / PNM bytes to RGB.
// Throws VisionError("ImageDecode") / ("ImageTooLarge": > 16384 on a side).
Rgb decodeImage(std::span<const std::uint8_t> bytes);
// stb_image's reason for the last decode failure.
const char* decodeFailureReason();

struct SizeOpt {
    std::int32_t align_size = 0;
    std::int32_t min_pixels = 0;
    std::int32_t max_pixels = 0;
};
// calc_size_preserved_ratio (llama.cpp mtmd-image.cpp; same float / int mix): {w, h}
std::array<std::uint32_t, 2> targetSize(std::uint32_t w, std::uint32_t h, const SizeOpt& o);
// img_tool::resize with PAD_CEIL, black padding, bicubic (Pillow, a = -0.5,
// 22-bit fixed-point weights).
Rgb resizePadCeil(const Rgb& src, std::uint32_t tw, std::uint32_t th);

// ---------------------------------------------------------------------------
// hashes (src/vision/hash.cpp)

std::array<std::uint8_t, 32> sha256(std::span<const std::span<const std::uint8_t>> parts);
// (tests) the same with the SHA-NI path allowed or not
std::array<std::uint8_t, 32> sha256Impl(std::span<const std::span<const std::uint8_t>> parts, bool allow_ni);
// Wyhash (final v4.2, the variant of Zig's std.hash.Wyhash) of `data` with `seed`.
std::uint64_t wyhash(std::uint64_t seed, std::span<const std::uint8_t> data);

// ---------------------------------------------------------------------------
// encoder

struct Hparams {
    std::uint32_t n_layer = 0;
    std::uint32_t n_ff = 0;
    std::uint32_t n_ff_pad = 0;
    std::uint32_t proj_dim = 0;
    std::uint32_t patch = 0;
    std::uint32_t merge = 0;
    std::uint32_t n_side = 0;
    float eps = 1e-6f;
    std::array<float, 3> mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> sd = {0.5f, 0.5f, 0.5f};
    std::uint32_t min_tokens = 8;
    std::uint32_t max_tokens = 4096;

    SizeOpt sizeOpt() const;
};

// A device region lent to the encoder for the duration of one encode.
struct Region {
    DevPtr ptr = 0;
    std::uint64_t len = 0;
};

enum class Mode { automatic, resident, stream };
std::optional<Mode> parseMode(std::string_view s);  // "auto" | "resident" | "stream"
const char* modeName(Mode m);

struct Prepared {
    Rgb rgb;  // preprocessed RGB (tw x th)
    std::uint32_t nx = 0, ny = 0;  // merged tokens per row / column
    // SHA-256 over the decoded pixels, their size, the preprocessing limits and the mmproj identity
    std::array<std::uint8_t, 32> hash{};
    std::uint32_t nTokens() const { return nx * ny; }
};

struct EncodeStats {
    double ms_total = 0;
    double ms_upload = 0;  // resident-mode weight upload (first use)
    bool resident = false;
    std::uint32_t n_patches = 0;
    std::uint64_t borrowed_extra = 0;  // bytes hipMalloc'ed because the lent regions were short
};

class Vision {
public:
    // Reads an mmproj GGUF into pinned host memory (no GPU work besides the
    // pinned allocation). Throws VisionError.
    static std::unique_ptr<Vision> load(const std::string& path);
    ~Vision();
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    Hparams hp;
    std::uint64_t id = 0;  // Wyhash of the repacked weights
    std::string path;
    Mode mode = Mode::automatic;
    // resident mode: VRAM that must stay free next to the weights
    std::uint64_t resident_margin = 1024ull << 20;
    std::uint32_t idle_s = 60;
    std::uint64_t n_encodes = 0;
    // diagnostics (WHIRL_VIS_PROF): per-op host-timed totals in ms (synchronizes)
    double* prof = nullptr;

    std::uint64_t pinnedBytes() const { return total_; }
    std::uint64_t totalBytes() const { return total_; }
    bool loaded() const { return module_.loaded(); }

    // Decode + preprocess one image (bytes of a PNG / JPEG / ... file).
    Prepared prepare(std::span<const std::uint8_t> bytes) const;
    Prepared prepareRgb(const Rgb& src) const;

    // Bytes of lent device memory an encode of n_tok tokens uses (stream mode).
    std::uint64_t arenaBytes(std::uint32_t n_tok) const;

    // Encode one preprocessed image: out (host, n_tok * proj_dim f32) gets the
    // projected embeddings in token order (row-major over the nx x ny grid).
    // `lend`: device regions free for the duration of the call (work on
    // `stream` is ordered after whatever used them before). Synchronous.
    EncodeStats encode(const Prepared& p, std::span<const Region> lend, hip::Stream stream, float* out, std::size_t out_len);

    // Free resident weights and unload the kernels (idle timeout / shutdown).
    void release();
    // Release after idle_s seconds without an encode; true when it released.
    bool idleTick();

private:
    Vision() = default;
    struct LayerOff {
        std::uint64_t ln1_g = 0, ln1_b = 0, qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0, ln2_g = 0, ln2_b = 0, up_w = 0, up_b = 0,
                      down_w = 0, down_b = 0, bytes = 0;
    };
    struct MiscOff {
        std::uint64_t patch_w = 0, patch_b = 0, pos = 0, bytes = 0;
    };
    struct MergerOff {
        std::uint64_t post_g = 0, post_b = 0, mm0_w = 0, mm0_b = 0, mm2_w = 0, mm2_b = 0, bytes = 0;
    };
    struct Funcs {
        hip::Function patchify = nullptr, pos_ln = nullptr, resid_ln = nullptr, qkv_prep = nullptr, attn = nullptr, gelu = nullptr,
                      bias = nullptr, gemm = nullptr;
    };
    std::array<std::uint64_t, 7> needs(std::uint32_t n_tok, bool stream_w) const;
    void loadModule();
    void launchGemm(hip::Stream s, DevPtr w, DevPtr x, DevPtr y, std::uint32_t ncols, std::uint32_t nrows, std::uint32_t n);
    void pm(hip::Stream s, int c);

    std::uint8_t* host_ = nullptr;
    LayerOff lo_;
    MiscOff so_;
    MergerOff mo_;
    std::uint64_t layer0_ = 0, merger_ = 0, total_ = 0;
    hip::Module module_;
    Funcs f_;
    DevPtr resident_ = 0;
    hip::Stream copy_stream_ = nullptr;
    std::array<hip::Event, 2> ev_copy_{}, ev_free_{};
    hip::Event ev_start_ = nullptr;
    double last_use_ = -1;  // nowSeconds() of the last encode, < 0: none
    double prof_t_ = -1;
};

// ---------------------------------------------------------------------------
// prompt side: image placeholders -> per-image content ids + RoPE spans

// Token id of image token i of the image with content hash h: >= image_id_base,
// 31 bits of a per-position hash of the SHA-256, so the same image always gets
// the same ids (prefix cache / tier reuse) and two different images collide
// only if all of their (>= 8) positions do.
std::uint32_t tokenId(const std::array<std::uint8_t, 32>& h, std::uint32_t i);

// Replace the k-th `pad_id` token of `tokens` with image k's n tokens (in
// order). spans[k] gets the cache positions (start .. start + n) and RoPE
// deltas; emb stays null. Throws VisionError("ImagePlaceholderMismatch").
std::vector<std::uint32_t> expand(std::span<const std::uint32_t> tokens, std::uint32_t pad_id,
                                  std::span<const Prepared* const> imgs, std::span<qwen35::VisSpan> spans);

// `vis-encode MMPROJ IMAGE [OUT.f32] [--reps N] [--mode M]` (args after the command
// name): standalone encoder test; prints the prototype-style report. Returns the exit code.
int cliVisEncode(std::span<const std::string> args);

// The vision code object embedded for gfx1201 (empty span if not built).
std::span<const std::uint8_t> kernelImage();

}  // namespace whirl::vision
