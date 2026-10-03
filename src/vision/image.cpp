// Image decoding (stb_image) and the Qwen-VL preprocessing.
// SPDX-License-Identifier: Apache-2.0
//
// Preprocessing adapted from llama.cpp tools/mtmd/mtmd-image.cpp (MIT, see
// THIRD_PARTY_NOTICES.md): "smart resize" (calc_size_preserved_ratio:
// aspect-preserving size aligned to patch * merge = 32, clamped to [min, max]
// tokens), Pillow-compatible bicubic resampling with 22-bit fixed-point weights
// (resize_pillow), and PAD_CEIL letterboxing with black. Normalization happens
// on the GPU (kernels/vision/vision.hip vis_patchify). Ported from the research
// prototype (vision/image.zig, item VIS); bit-exact to PIL.

#include "whirl/vision.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

extern "C" {
unsigned char* stbi_load_from_memory(const unsigned char* buffer, int len, int* x, int* y, int* comp, int req_comp);
void stbi_image_free(void* retval_from_stbi_load);
const char* stbi_failure_reason(void);
}

namespace whirl::vision {

Rgb decodeImage(std::span<const std::uint8_t> bytes) {
    if (bytes.empty() || bytes.size() > static_cast<std::size_t>(INT_MAX)) throw VisionError("ImageDecode");
    int w = 0, h = 0, comp = 0;
    unsigned char* p = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &comp, 3);
    if (p == nullptr) throw VisionError("ImageDecode", decodeFailureReason());
    struct Free {
        unsigned char* p;
        ~Free() { stbi_image_free(p); }
    } guard{p};
    if (w <= 0 || h <= 0) throw VisionError("ImageDecode");
    // 16k x 16k cap (decoded size 768 MiB)
    if (w > 16384 || h > 16384) throw VisionError("ImageTooLarge");
    Rgb out;
    out.w = static_cast<std::uint32_t>(w);
    out.h = static_cast<std::uint32_t>(h);
    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
    out.px.assign(p, p + n);
    return out;
}

const char* decodeFailureReason() {
    const char* r = stbi_failure_reason();
    return r != nullptr ? r : "unknown";
}

std::array<std::uint32_t, 2> targetSize(std::uint32_t w_in, std::uint32_t h_in, const SizeOpt& o) {
    const std::int32_t width = static_cast<std::int32_t>(w_in);
    const std::int32_t height = static_cast<std::int32_t>(h_in);
    const float f = static_cast<float>(o.align_size);
    const std::int32_t fi = o.align_size;
    auto roundBy = [&](float x) { return static_cast<std::int32_t>(std::round(x / f)) * fi; };
    auto ceilBy = [&](float x) { return static_cast<std::int32_t>(std::ceil(x / f)) * fi; };
    auto floorBy = [&](float x) { return static_cast<std::int32_t>(std::floor(x / f)) * fi; };
    std::int32_t w_bar = std::max(o.align_size, roundBy(static_cast<float>(width)));
    std::int32_t h_bar = std::max(o.align_size, roundBy(static_cast<float>(height)));
    if (o.max_pixels > 0 && h_bar * w_bar > o.max_pixels) {
        const float beta = std::sqrt(static_cast<float>(height) * static_cast<float>(width) / static_cast<float>(o.max_pixels));
        h_bar = std::max(o.align_size, floorBy(static_cast<float>(height) / beta));
        w_bar = std::max(o.align_size, floorBy(static_cast<float>(width) / beta));
    } else if (o.min_pixels > 0 && h_bar * w_bar < o.min_pixels) {
        const float beta = std::sqrt(static_cast<float>(o.min_pixels) / (static_cast<float>(height) * static_cast<float>(width)));
        h_bar = ceilBy(static_cast<float>(height) * beta);
        w_bar = ceilBy(static_cast<float>(width) * beta);
    }
    return {static_cast<std::uint32_t>(w_bar), static_cast<std::uint32_t>(h_bar)};
}

namespace {

constexpr int precision_bits = 32 - 8 - 2;

double filterBicubic(double x) {
    if (x < 0.0) x = -x;
    const double a = -0.5;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

struct Coeffs {
    std::size_t ksize = 0;
    std::vector<std::int32_t> bounds;   // [out][2] = xmin, xcnt
    std::vector<std::int32_t> weights;  // [out][ksize]
};

Coeffs precompute(std::uint32_t in_size, std::uint32_t out_size) {
    const double filter_support = 2.0;
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    double filterscale = scale;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = filter_support * filterscale;
    Coeffs co;
    co.ksize = static_cast<std::size_t>(std::ceil(support)) * 2 + 1;
    std::vector<double> pre(static_cast<std::size_t>(out_size) * co.ksize);
    co.bounds.resize(static_cast<std::size_t>(out_size) * 2);
    for (std::size_t xx = 0; xx < out_size; ++xx) {
        const double center = (static_cast<double>(xx) + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        std::int32_t xmin = static_cast<std::int32_t>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        std::int32_t xmax = static_cast<std::int32_t>(center + support + 0.5);
        if (xmax > static_cast<std::int32_t>(in_size)) xmax = static_cast<std::int32_t>(in_size);
        xmax -= xmin;
        std::size_t x = 0;
        for (; x < static_cast<std::size_t>(xmax); ++x) {
            const double w = filterBicubic((static_cast<double>(x) + static_cast<double>(xmin) - center + 0.5) * ss);
            pre[xx * co.ksize + x] = w;
            ww += w;
        }
        for (x = 0; x < static_cast<std::size_t>(xmax); ++x)
            if (ww != 0.0) pre[xx * co.ksize + x] /= ww;
        for (; x < co.ksize; ++x) pre[xx * co.ksize + x] = 0;
        co.bounds[xx * 2] = xmin;
        co.bounds[xx * 2 + 1] = xmax;
    }
    co.weights.resize(pre.size());
    const double fxp = std::ldexp(1.0, precision_bits);
    for (std::size_t i = 0; i < pre.size(); ++i) {
        const double p = pre[i];
        const double r = p * fxp + (p < 0 ? -0.5 : 0.5);
        co.weights[i] = static_cast<std::int32_t>(r);  // truncation toward zero
    }
    return co;
}

inline std::uint8_t clip8(std::int32_t v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<std::uint8_t>(v);
}

// i32 wrapping multiply-add (as the prototype's +% / *%)
inline std::uint32_t mad(std::uint32_t acc, std::uint8_t s, std::int32_t w) {
    return acc + static_cast<std::uint32_t>(s) * static_cast<std::uint32_t>(w);
}
inline std::int32_t shr(std::uint32_t acc) { return static_cast<std::int32_t>(acc) >> precision_bits; }

std::vector<std::uint8_t> resizePillow(const Rgb& src, std::uint32_t tw, std::uint32_t th) {
    const bool need_h = tw != src.w;
    const bool need_v = th != src.h;
    std::vector<std::uint8_t> cur = src.px;
    std::uint32_t cw = src.w;
    const std::uint32_t init = 1u << (precision_bits - 1);
    if (need_h) {
        const Coeffs co = precompute(src.w, tw);
        std::vector<std::uint8_t> out(static_cast<std::size_t>(tw) * src.h * 3);
        for (std::size_t yy = 0; yy < src.h; ++yy) {
            const std::uint8_t* row = cur.data() + yy * cw * 3;
            std::uint8_t* drow = out.data() + yy * tw * 3;
            for (std::size_t xx = 0; xx < tw; ++xx) {
                const std::size_t xmin = static_cast<std::size_t>(co.bounds[xx * 2]);
                const std::size_t xcnt = static_cast<std::size_t>(co.bounds[xx * 2 + 1]);
                const std::int32_t* k = co.weights.data() + xx * co.ksize;
                std::uint32_t s0 = init, s1 = init, s2 = init;
                for (std::size_t x = 0; x < xcnt; ++x) {
                    const std::uint8_t* p = row + (xmin + x) * 3;
                    s0 = mad(s0, p[0], k[x]);
                    s1 = mad(s1, p[1], k[x]);
                    s2 = mad(s2, p[2], k[x]);
                }
                drow[xx * 3] = clip8(shr(s0));
                drow[xx * 3 + 1] = clip8(shr(s1));
                drow[xx * 3 + 2] = clip8(shr(s2));
            }
        }
        cur = std::move(out);
        cw = tw;
    }
    if (need_v) {
        const Coeffs co = precompute(src.h, th);
        const std::size_t row_elems = static_cast<std::size_t>(cw) * 3;
        std::vector<std::uint8_t> out(row_elems * th);
        std::vector<std::uint32_t> acc(row_elems);
        for (std::size_t yy = 0; yy < th; ++yy) {
            const std::size_t ymin = static_cast<std::size_t>(co.bounds[yy * 2]);
            const std::size_t ycnt = static_cast<std::size_t>(co.bounds[yy * 2 + 1]);
            const std::int32_t* k = co.weights.data() + yy * co.ksize;
            std::fill(acc.begin(), acc.end(), init);
            for (std::size_t y = 0; y < ycnt; ++y) {
                const std::uint8_t* srow = cur.data() + (ymin + y) * row_elems;
                const std::int32_t w = k[y];
                for (std::size_t i = 0; i < row_elems; ++i) acc[i] = mad(acc[i], srow[i], w);
            }
            std::uint8_t* drow = out.data() + yy * row_elems;
            for (std::size_t i = 0; i < row_elems; ++i) drow[i] = clip8(shr(acc[i]));
        }
        cur = std::move(out);
    }
    return cur;
}

}  // namespace

Rgb resizePadCeil(const Rgb& src, std::uint32_t tw, std::uint32_t th) {
    Rgb dst;
    dst.w = tw;
    dst.h = th;
    if (src.w == tw && src.h == th) {
        dst.px = src.px;
        return dst;
    }
    const float scale_w = static_cast<float>(tw) / static_cast<float>(src.w);
    const float scale_h = static_cast<float>(th) / static_cast<float>(src.h);
    const float scale = std::min(scale_w, scale_h);
    const std::uint32_t nw = std::min(static_cast<std::uint32_t>(std::ceil(static_cast<float>(src.w) * scale)), tw);
    const std::uint32_t nh = std::min(static_cast<std::uint32_t>(std::ceil(static_cast<float>(src.h) * scale)), th);
    const std::vector<std::uint8_t> resized = resizePillow(src, nw, nh);
    dst.px.assign(static_cast<std::size_t>(tw) * th * 3, 0);
    const std::uint32_t ox = (tw - nw) / 2;
    const std::uint32_t oy = (th - nh) / 2;
    for (std::uint32_t y = 0; y < nh; ++y) {
        const std::uint32_t dy = y + oy;
        if (dy >= th) continue;
        const std::size_t n = static_cast<std::size_t>(std::min(nw, tw - ox)) * 3;
        std::memcpy(dst.px.data() + (static_cast<std::size_t>(dy) * tw + ox) * 3, resized.data() + static_cast<std::size_t>(y) * nw * 3, n);
    }
    return dst;
}

}  // namespace whirl::vision
