// Vision encoder host side: mmproj loading (pinned, repacked), the on-demand
// code object, resident / streamed weights and the encode schedule.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the research prototype (vision/vision.zig, item VIS). The graph
// follows llama.cpp tools/mtmd (MIT): clip.cpp (qwen3vl), models/qwen3vl.cpp.

#include "whirl/vision.h"

#include "whirl/common.h"
#include "whirl/gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>

namespace whirl::vision {

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using hip::Dim3;

u64 alignUp(u64 x, u64 a) { return (x + a - 1) / a * a; }

// f32 -> f16 bits, round to nearest even (as a hardware / Zig @floatCast conversion)
std::uint16_t f32ToF16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {  // inf / nan
        if (ax > 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7e00u | ((ax >> 13) & 0x3ffu));
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (ax >= 0x477ff000u) return static_cast<std::uint16_t>(sign | 0x7c00u);  // rounds to >= 65520: inf
    if (ax < 0x38800000u) {                                                   // f16 subnormal / zero
        if (ax < 0x33000000u) return static_cast<std::uint16_t>(sign);        // < 2^-25: rounds to 0
        const std::uint32_t e = ax >> 23;
        const std::uint32_t m = (ax & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126 - e;  // value = m * 2^(e-150); f16 unit 2^-24 -> shift by (126 - e)
        const std::uint32_t q = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1);
        const std::uint32_t half = 1u << (shift - 1);
        std::uint32_t r = q;
        if (rem > half || (rem == half && (q & 1u))) r += 1;
        return static_cast<std::uint16_t>(sign | r);
    }
    std::uint32_t r = ((ax - 0x38000000u) >> 13);
    const std::uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r += 1;
    return static_cast<std::uint16_t>(sign | r);
}

float f16ToF32(std::uint16_t h) {
    const std::uint32_t sign = (std::uint32_t(h) & 0x8000u) << 16;
    const std::uint32_t e = (h >> 10) & 0x1fu;
    std::uint32_t m = h & 0x3ffu;
    std::uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = sign;
        } else {
            int ee = -1;
            do {
                ++ee;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            x = sign | (static_cast<std::uint32_t>(127 - 15 - ee) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// tensor `name` as f16 (want_f16) or f32, n bytes, into dst; BF16 / F32 / F16 sources are
// converted (BF16 -> f16 is exact within the f16 range; BF16 mmproj files, e.g. Ornith)
void readTensor(const gguf::File& f, std::string_view name, bool want_f16, u64 n, std::uint8_t* dst) {
    const gguf::TensorInfo* t = f.tensor(name);
    if (t == nullptr) throw VisionError("MissingTensor", std::string(name));
    const u64 want_es = want_f16 ? 2 : 4;
    const u64 elems = n / want_es;
    const gguf::GgmlType want_ty = want_f16 ? gguf::GgmlType::f16 : gguf::GgmlType::f32;
    const auto src = f.tensorData(*t);
    if (t->type == want_ty) {
        if (t->nbytes() != n) throw VisionError("UnsupportedVisionShape", std::string(name));
        std::memcpy(dst, src.data(), n);
        return;
    }
    u64 src_es = 0;
    switch (t->type) {
    case gguf::GgmlType::f32: src_es = 4; break;
    case gguf::GgmlType::f16:
    case gguf::GgmlType::bf16: src_es = 2; break;
    default: throw VisionError("UnsupportedTensorType", std::string(name));
    }
    if (t->nbytes() != elems * src_es) throw VisionError("UnsupportedVisionShape", std::string(name));
    const std::uint8_t* s = src.data();
    for (u64 i = 0; i < elems; ++i) {
        float x;
        if (t->type == gguf::GgmlType::f32) {
            std::memcpy(&x, s + i * 4, 4);
        } else if (t->type == gguf::GgmlType::f16) {
            std::uint16_t h;
            std::memcpy(&h, s + i * 2, 2);
            x = f16ToF32(h);
        } else {
            std::uint16_t h;
            std::memcpy(&h, s + i * 2, 2);
            const std::uint32_t b = std::uint32_t(h) << 16;
            std::memcpy(&x, &b, 4);
        }
        if (want_f16) {
            const std::uint16_t h = f32ToF16(x);
            std::memcpy(dst + i * 2, &h, 2);
        } else {
            std::memcpy(dst + i * 4, &x, 4);
        }
    }
}

std::optional<std::vector<float>> floatArray3(const gguf::File& f, std::string_view key) {
    const gguf::Value* v = f.find(key);
    if (v == nullptr || v->type != gguf::ValueType::array) return std::nullopt;
    if (v->arr.elem != gguf::ValueType::f32 || v->arr.len != 3) return std::nullopt;
    std::vector<float> out(3);
    std::memcpy(out.data(), v->arr.raw.data(), 12);
    return out;
}

double msSince(double t0) { return (nowSeconds() - t0) * 1000.0; }

}  // namespace

SizeOpt Hparams::sizeOpt() const {
    const i32 area = static_cast<i32>(patch * patch * merge * merge);
    return {static_cast<i32>(patch * merge), static_cast<i32>(min_tokens) * area, static_cast<i32>(max_tokens) * area};
}

std::optional<Mode> parseMode(std::string_view s) {
    if (s == "auto") return Mode::automatic;
    if (s == "resident") return Mode::resident;
    if (s == "stream") return Mode::stream;
    return std::nullopt;
}

const char* modeName(Mode m) {
    switch (m) {
    case Mode::automatic: return "auto";
    case Mode::resident: return "resident";
    case Mode::stream: return "stream";
    }
    return "?";
}

std::unique_ptr<Vision> Vision::load(const std::string& path) {
    gguf::File f = [&] {
        try {
            return gguf::File::open(path);
        } catch (const std::exception& e) {
            throw VisionError("NotAnMmproj", e.what());
        }
    }();
    if (f.getStringOr("general.architecture", "") != "clip") throw VisionError("NotAnMmproj", path);
    if (f.getStringOr("clip.projector_type", "") != "qwen3vl_merger") throw VisionError("UnsupportedProjector", std::string(f.getStringOr("clip.projector_type", "")));
    Hparams hp;
    try {
        hp.n_layer = static_cast<u32>(f.getUint("clip.vision.block_count"));
        hp.n_ff = static_cast<u32>(f.getUint("clip.vision.feed_forward_length"));
        hp.proj_dim = static_cast<u32>(f.getUint("clip.vision.projection_dim"));
        hp.patch = static_cast<u32>(f.getUint("clip.vision.patch_size"));
        hp.merge = static_cast<u32>(f.getUintOr("clip.vision.spatial_merge_size", 2));
        hp.eps = f.has("clip.vision.attention.layer_norm_epsilon") ? static_cast<float>(f.getFloat("clip.vision.attention.layer_norm_epsilon")) : 1e-6f;
        if (f.getUint("clip.vision.embedding_length") != E || f.getUint("clip.vision.attention.head_count") != n_head || hp.patch != 16 ||
            hp.merge != 2 || hp.n_ff % 16 != 0)
            throw VisionError("UnsupportedVisionShape");
    } catch (const gguf::Error& e) {
        throw VisionError("UnsupportedVisionShape", e.what());
    }
    hp.n_ff_pad = static_cast<u32>(alignUp(hp.n_ff, 128));
    if (auto a = floatArray3(f, "clip.vision.image_mean")) std::copy(a->begin(), a->end(), hp.mean.begin());
    if (auto a = floatArray3(f, "clip.vision.image_std")) std::copy(a->begin(), a->end(), hp.sd.begin());
    if (const gguf::Value* v = f.find("clip.vision.is_deepstack_layers"); v != nullptr && v->type == gguf::ValueType::array &&
                                                                        v->arr.elem == gguf::ValueType::boolean) {
        for (u64 i = 0; i < v->arr.len; ++i)
            if (v->arr.raw[i] != 0) throw VisionError("DeepstackUnsupported");
    }
    if (hp.n_layer % 2 == 0) throw VisionError("UnsupportedVisionShape", "even layer count");  // stream schedule assumes an odd count
    const gguf::TensorInfo* pos_t = f.tensor("v.position_embd.weight");
    if (pos_t == nullptr) throw VisionError("MissingTensor", "v.position_embd.weight");
    const u64 n_pos = pos_t->ne[1];
    hp.n_side = static_cast<u32>(std::sqrt(static_cast<double>(n_pos)));
    if (static_cast<u64>(hp.n_side) * hp.n_side != n_pos || pos_t->ne[0] != E) throw VisionError("UnsupportedVisionShape");

    std::unique_ptr<Vision> v(new Vision());
    v->hp = hp;
    const u64 A = 256;
    u64 o = 0;
    v->so_.patch_w = o;
    o = alignUp(o + static_cast<u64>(E) * patch_k * 2, A);
    v->so_.patch_b = o;
    o = alignUp(o + E * 4, A);
    v->so_.pos = o;
    o = alignUp(o + n_pos * E * 4, A);
    v->so_.bytes = alignUp(o, 4096);
    LayerOff lo;
    o = 0;
    const std::pair<u64*, u64> lparts[] = {
        {&lo.ln1_g, E * 4},       {&lo.ln1_b, E * 4}, {&lo.qkv_w, static_cast<u64>(QKV) * E * 2}, {&lo.qkv_b, QKV * 4},
        {&lo.out_w, static_cast<u64>(E) * E * 2}, {&lo.out_b, E * 4}, {&lo.ln2_g, E * 4},       {&lo.ln2_b, E * 4},
    };
    for (const auto& p : lparts) {
        *p.first = o;
        o = alignUp(o + p.second, A);
    }
    lo.up_w = o;
    o = alignUp(o + static_cast<u64>(hp.n_ff) * E * 2, A);
    lo.up_b = o;
    o = alignUp(o + static_cast<u64>(hp.n_ff) * 4, A);
    lo.down_w = o;
    o = alignUp(o + static_cast<u64>(E) * hp.n_ff_pad * 2, A);
    lo.down_b = o;
    o = alignUp(o + E * 4, A);
    lo.bytes = alignUp(o, 4096);
    v->lo_ = lo;
    const u64 MI = 4 * E;
    MergerOff mo;
    o = 0;
    mo.post_g = o;
    o = alignUp(o + E * 4, A);
    mo.post_b = o;
    o = alignUp(o + E * 4, A);
    mo.mm0_w = o;
    o = alignUp(o + MI * MI * 2, A);
    mo.mm0_b = o;
    o = alignUp(o + MI * 4, A);
    mo.mm2_w = o;
    o = alignUp(o + static_cast<u64>(hp.proj_dim) * MI * 2, A);
    mo.mm2_b = o;
    o = alignUp(o + static_cast<u64>(hp.proj_dim) * 4, A);
    mo.bytes = alignUp(o, 4096);
    v->mo_ = mo;
    v->layer0_ = v->so_.bytes;
    v->merger_ = v->layer0_ + static_cast<u64>(hp.n_layer) * lo.bytes;
    v->total_ = v->merger_ + mo.bytes;

    v->host_ = static_cast<std::uint8_t*>(hip::hostMalloc(v->total_));
    std::memset(v->host_, 0, v->total_);
    std::uint8_t* const H = v->host_;
    // patch embedding: [E][2 * 768] = [W0 row | W1 row]
    {
        std::vector<std::uint8_t> tmp(static_cast<std::size_t>(E) * 768 * 2);
        const char* names[2] = {"v.patch_embd.weight", "v.patch_embd.weight.1"};
        for (int k = 0; k < 2; ++k) {
            readTensor(f, names[k], true, tmp.size(), tmp.data());
            for (u64 r = 0; r < E; ++r)
                std::memcpy(H + v->so_.patch_w + (r * patch_k + static_cast<u64>(k) * 768) * 2, tmp.data() + r * 768 * 2, 768 * 2);
        }
    }
    readTensor(f, "v.patch_embd.bias", false, E * 4, H + v->so_.patch_b);
    readTensor(f, "v.position_embd.weight", false, n_pos * E * 4, H + v->so_.pos);
    std::vector<std::uint8_t> tmpd(static_cast<std::size_t>(E) * hp.n_ff * 2);
    for (u32 l = 0; l < hp.n_layer; ++l) {
        std::uint8_t* h = H + v->layer0_ + static_cast<u64>(l) * lo.bytes;
        const std::string pre = "v.blk." + std::to_string(l) + ".";
        struct P {
            const char* name;
            u64 off;
            bool f16;
            u64 n;
        };
        const P parts[] = {
            {"ln1.weight", lo.ln1_g, false, E * 4},
            {"ln1.bias", lo.ln1_b, false, E * 4},
            {"attn_qkv.weight", lo.qkv_w, true, static_cast<u64>(QKV) * E * 2},
            {"attn_qkv.bias", lo.qkv_b, false, QKV * 4},
            {"attn_out.weight", lo.out_w, true, static_cast<u64>(E) * E * 2},
            {"attn_out.bias", lo.out_b, false, E * 4},
            {"ln2.weight", lo.ln2_g, false, E * 4},
            {"ln2.bias", lo.ln2_b, false, E * 4},
        };
        for (const P& p : parts) readTensor(f, pre + p.name, p.f16, p.n, h + p.off);
        readTensor(f, pre + "ffn_up.weight", true, static_cast<u64>(hp.n_ff) * E * 2, h + lo.up_w);
        readTensor(f, pre + "ffn_up.bias", false, static_cast<u64>(hp.n_ff) * 4, h + lo.up_b);
        // ffn_down [E][n_ff] -> rows padded to n_ff_pad (zeros)
        readTensor(f, pre + "ffn_down.weight", true, tmpd.size(), tmpd.data());
        for (u64 r = 0; r < E; ++r) std::memcpy(h + lo.down_w + r * hp.n_ff_pad * 2, tmpd.data() + r * hp.n_ff * 2, static_cast<std::size_t>(hp.n_ff) * 2);
        readTensor(f, pre + "ffn_down.bias", false, E * 4, h + lo.down_b);
    }
    {
        std::uint8_t* h = H + v->merger_;
        readTensor(f, "v.post_ln.weight", false, E * 4, h + mo.post_g);
        readTensor(f, "v.post_ln.bias", false, E * 4, h + mo.post_b);
        readTensor(f, "mm.0.weight", true, MI * MI * 2, h + mo.mm0_w);
        readTensor(f, "mm.0.bias", false, MI * 4, h + mo.mm0_b);
        readTensor(f, "mm.2.weight", true, static_cast<u64>(hp.proj_dim) * MI * 2, h + mo.mm2_w);
        readTensor(f, "mm.2.bias", false, static_cast<u64>(hp.proj_dim) * 4, h + mo.mm2_b);
    }
    v->id = wyhash(0x56495331ull, std::span<const std::uint8_t>(H, v->total_));
    v->path = path;
    return v;
}

Vision::~Vision() {
    release();
    if (host_ != nullptr) hip::hostFree(host_);
}

Prepared Vision::prepare(std::span<const std::uint8_t> bytes) const {
    const Rgb src = decodeImage(bytes);
    return prepareRgb(src);
}

Prepared Vision::prepareRgb(const Rgb& src) const {
    const auto ts = targetSize(src.w, src.h, hp.sizeOpt());
    std::uint8_t hdr[32];
    auto put32 = [&](int at, u32 x) { std::memcpy(hdr + at, &x, 4); };
    std::memcpy(hdr, &id, 8);
    put32(8, src.w);
    put32(12, src.h);
    put32(16, ts[0]);
    put32(20, ts[1]);
    put32(24, hp.min_tokens);
    put32(28, hp.max_tokens);
    static const char tag[] = "whirl-vis-1";
    const std::span<const std::uint8_t> parts[3] = {
        {reinterpret_cast<const std::uint8_t*>(tag), sizeof(tag) - 1}, {hdr, 32}, {src.px.data(), src.px.size()}};
    Prepared p;
    p.nx = ts[0] / 32;
    p.ny = ts[1] / 32;
    p.hash = sha256(parts);
    p.rgb = resizePadCeil(src, ts[0], ts[1]);
    return p;
}

std::array<u64, 7> Vision::needs(u32 n_tok, bool stream_w) const {
    const u64 N = static_cast<u64>(n_tok) * 4;
    const u64 t_cols = std::max<u64>(std::max<u64>(QKV, hp.n_ff), std::max<u64>(hp.proj_dim / 4 + 1, patch_k / 2));
    return {
        N * E * 4,
        N * E * 2,
        N * t_cols * 4,
        3ull * n_head * alignUp(N, 128) * head_pad * 2,
        N * E * 2,
        N * std::max<u64>(hp.n_ff_pad, 4 * 3 * 16 * 16 / 4) * 2,
        stream_w ? lo_.bytes + mo_.bytes : 0,
    };
}

u64 Vision::arenaBytes(u32 n_tok) const {
    u64 s = 0;
    for (u64 b : needs(n_tok, true)) s += alignUp(b, 256);
    return s;
}

void Vision::loadModule() {
    if (module_.loaded()) return;
    const auto img = kernelImage();
    if (img.empty()) throw VisionError("NoVisionKernels", "this build has no gfx1201 vision code object");
    module_ = hip::Module::loadData(img.data(), hip::Arch::gfx1201);
    f_.patchify = module_.getFunction("vis_patchify");
    f_.pos_ln = module_.getFunction("vis_pos_ln");
    f_.resid_ln = module_.getFunction("vis_resid_ln");
    f_.qkv_prep = module_.getFunction("vis_qkv_prep2");
    f_.attn = module_.getFunction("vis_attn2");
    f_.gelu = module_.getFunction("vis_gelu");
    f_.bias = module_.getFunction("vis_bias");
    f_.gemm = module_.getFunction("vis_gemm");
    if (copy_stream_ == nullptr) {
        copy_stream_ = hip::streamCreate(true);
        for (auto& e : ev_copy_) e = hip::eventCreate(false);
        for (auto& e : ev_free_) e = hip::eventCreate(false);
        ev_start_ = hip::eventCreate(false);
    }
}

void Vision::release() {
    try {
        if (copy_stream_ != nullptr) hip::streamSync(copy_stream_);
    } catch (...) {
    }
    if (resident_ != 0) {
        hip::free(resident_);
        resident_ = 0;
    }
    if (module_.loaded()) module_ = hip::Module();
    if (copy_stream_ != nullptr) {
        for (auto& e : ev_copy_) {
            hip::eventDestroy(e);
            e = nullptr;
        }
        for (auto& e : ev_free_) {
            hip::eventDestroy(e);
            e = nullptr;
        }
        hip::eventDestroy(ev_start_);
        ev_start_ = nullptr;
        hip::streamDestroy(copy_stream_);
        copy_stream_ = nullptr;
    }
}

bool Vision::idleTick() {
    if (!module_.loaded()) return false;
    if (last_use_ < 0) return false;
    if (nowSeconds() - last_use_ < static_cast<double>(idle_s)) return false;
    release();
    return true;
}

void Vision::pm(hip::Stream s, int c) {
    if (prof == nullptr) return;
    try {
        hip::streamSync(s);
    } catch (...) {
    }
    const double now = nowSeconds();
    if (prof_t_ >= 0) prof[c] += (now - prof_t_) * 1000.0;
    prof_t_ = now;
}

void Vision::launchGemm(hip::Stream s, DevPtr w, DevPtr x, DevPtr y, u32 ncols, u32 nrows, u32 n) {
    hip::launch(f_.gemm, Dim3{(n + 255) / 256, (nrows + 127) / 128, 1}, Dim3{256, 1, 1}, 0, s, w, x, y, static_cast<i32>(ncols),
                static_cast<i32>(nrows), static_cast<i32>(n), i32(0));
}

EncodeStats Vision::encode(const Prepared& p, std::span<const Region> lend, hip::Stream stream, float* out, std::size_t out_len) {
    const u32 n_tok = p.nTokens();
    const u32 N = n_tok * 4;
    if (out_len < static_cast<std::size_t>(n_tok) * hp.proj_dim) throw VisionError("OutputTooSmall");
    const double t0 = nowSeconds();
    EncodeStats st;
    st.n_patches = N;
    loadModule();
    // resident weights when they fit (auto) / forced
    if (resident_ == 0 && mode != Mode::stream) {
        const hip::MemInfo mi = hip::memInfo();
        if (mode == Mode::resident || mi.free > total_ + resident_margin) {
            const double tu = nowSeconds();
            try {
                resident_ = hip::malloc(total_);
            } catch (const hip::Error&) {
                resident_ = 0;
            }
            if (resident_ != 0) {
                hip::copyAnyAsync(resident_, reinterpret_cast<DevPtr>(host_), total_, copy_stream_);
                hip::streamSync(copy_stream_);
                st.ms_upload = msSince(tu);
            }
        }
    }
    const bool res = resident_ != 0;
    st.resident = res;
    // carve the lent regions (first fit, largest first); hipMalloc what does not fit
    const std::array<u64, 7> need = needs(n_tok, !res);
    std::array<DevPtr, 7> ptr{};
    struct Extra {
        std::array<DevPtr, 7> p{};
        ~Extra() {
            for (DevPtr e : p)
                if (e != 0) hip::free(e);
        }
    } extra;
    {
        std::array<u64, 16> used{};
        std::array<int, 7> order = {0, 1, 2, 3, 4, 5, 6};
        // (the prototype sorts with an unstable pdq sort on strictly greater; ties keep
        // any order, and equal sizes are interchangeable for first fit)
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return need[a] > need[b]; });
        for (int k : order) {
            if (need[k] == 0) continue;
            const u64 sz = alignUp(need[k], 256);
            for (std::size_t ri = 0; ri < lend.size(); ++ri) {
                if (ri >= used.size()) break;
                if (lend[ri].len - used[ri] >= sz) {
                    ptr[k] = lend[ri].ptr + used[ri];
                    used[ri] += sz;
                    break;
                }
            }
            if (ptr[k] == 0) {
                extra.p[k] = hip::malloc(sz);
                ptr[k] = extra.p[k];
                st.borrowed_extra += sz;
            }
        }
    }
    const DevPtr X = ptr[0], Hb = ptr[1], T = ptr[2], Qb = ptr[3], Ab = ptr[4], M = ptr[5], W = ptr[6];
    const u32 Np = static_cast<u32>(alignUp(N, 128));
    const u64 qkv_bytes = static_cast<u64>(n_head) * Np * head_pad * 2;
    const DevPtr Kb = Qb + qkv_bytes;
    const DevPtr Vb = Kb + qkv_bytes;
    const u32 pw = p.nx * 2;
    const u32 ph = p.ny * 2;
    const u64 lb = lo_.bytes;
    const DevPtr host0 = reinterpret_cast<DevPtr>(host_);
    // weight pointers: resident buffer, or the stream slots
    //   slot 0 = W, slot 1 = W + lb (misc first, then odd layers); merger at W + lb
    const DevPtr slot[2] = {W, W + lb};
    hip::Stream cs = copy_stream_;
    if (!res) {
        // the lent regions may still be read by queued work on `stream`
        hip::eventRecord(ev_start_, stream);
        hip::streamWaitEvent(cs, ev_start_);
        hip::copyAnyAsync(slot[1], host0 + 0, so_.bytes, cs);
        hip::eventRecord(ev_copy_[1], cs);
        hip::copyAnyAsync(slot[0], host0 + layer0_, lb, cs);
        hip::eventRecord(ev_copy_[0], cs);
    }
    const DevPtr misc = res ? resident_ : slot[1];
    // image -> M (u8), patch rows -> T (f16), patch GEMM -> X
    prof_t_ = -1;
    pm(stream, 7);
    // pageable source: drain the stream first (queued LM work may still use the lent regions)
    hip::streamSync(stream);
    hip::uploadAsync(M, p.rgb.px.data(), p.rgb.px.size(), stream);
    if (!res) hip::streamWaitEvent(stream, ev_copy_[1]);
    hip::launch(f_.patchify, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, M, T, static_cast<i32>(p.rgb.w), static_cast<i32>(pw), hp.mean[0],
                hp.mean[1], hp.mean[2], hp.sd[0], hp.sd[1], hp.sd[2]);
    launchGemm(stream, misc + so_.patch_w, T, X, patch_k, E, N);
    const DevPtr lw0 = res ? resident_ + layer0_ : slot[0];
    hip::launch(f_.pos_ln, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, X, misc + so_.pos, misc + so_.patch_b, static_cast<i32>(pw),
                static_cast<i32>(ph), static_cast<i32>(hp.n_side), lw0 + lo_.ln1_g, lw0 + lo_.ln1_b, Hb, hp.eps);
    pm(stream, 0);
    if (!res) {
        hip::eventRecord(ev_free_[1], stream);
        if (hp.n_layer > 1) {
            hip::streamWaitEvent(cs, ev_free_[1]);
            hip::copyAnyAsync(slot[1], host0 + layer0_ + lb, lb, cs);
            hip::eventRecord(ev_copy_[1], cs);
        }
    }
    const float theta_scale = static_cast<float>(std::pow(10000.0, -2.0 / 36.0));
    const float scale_log2 = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)) * std::numbers::log2e);
    const DevPtr mbase = res ? resident_ + merger_ : W + lb;
    const LayerOff lo = lo_;
    for (u32 l = 0; l < hp.n_layer; ++l) {
        const u32 s = l % 2;
        const DevPtr L = res ? resident_ + layer0_ + static_cast<u64>(l) * lb : slot[s];
        if (!res) hip::streamWaitEvent(stream, ev_copy_[s]);
        pm(stream, 7);
        launchGemm(stream, L + lo.qkv_w, Hb, T, E, QKV, N);
        pm(stream, 1);
        hip::launch(f_.qkv_prep, Dim3{Np / 64, n_head, 1}, Dim3{256, 1, 1}, 0, stream, T, L + lo.qkv_b, Qb, Kb, Vb, static_cast<i32>(N),
                    static_cast<i32>(Np), static_cast<i32>(pw), theta_scale);
        pm(stream, 2);
        hip::launch(f_.attn, Dim3{Np / 128, n_head, 1}, Dim3{128, 1, 1}, 0, stream, Qb, Kb, Vb, Ab, static_cast<i32>(N), static_cast<i32>(Np),
                    scale_log2);
        pm(stream, 3);
        launchGemm(stream, L + lo.out_w, Ab, T, E, E, N);
        pm(stream, 1);
        hip::launch(f_.resid_ln, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, X, T, L + lo.out_b, L + lo.ln2_g, L + lo.ln2_b, Hb, hp.eps);
        pm(stream, 4);
        launchGemm(stream, L + lo.up_w, Hb, T, E, hp.n_ff, N);
        pm(stream, 1);
        hip::launch(f_.gelu, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, T, L + lo.up_b, M, static_cast<i32>(hp.n_ff), static_cast<i32>(hp.n_ff_pad));
        pm(stream, 5);
        launchGemm(stream, L + lo.down_w, M, T, hp.n_ff_pad, E, N);
        pm(stream, 1);
        // residual, then the next layer's ln1 (or post_ln after the last layer)
        const bool last = l + 1 == hp.n_layer;
        // the next layer's ln1 lives in the other slot (already waited for below)
        if (!last && !res) hip::streamWaitEvent(stream, ev_copy_[1 - s]);
        if (last && !res) {
            // merger weights were queued after layer n_layer - 2 freed slot 1
            hip::streamWaitEvent(stream, ev_copy_[1]);
        }
        const DevPtr Ln = res ? resident_ + layer0_ + static_cast<u64>(l + 1) * lb : slot[1 - s];
        const DevPtr g = last ? mbase + mo_.post_g : Ln + lo.ln1_g;
        const DevPtr b = last ? mbase + mo_.post_b : Ln + lo.ln1_b;
        hip::launch(f_.resid_ln, Dim3{N, 1, 1}, Dim3{256, 1, 1}, 0, stream, X, T, L + lo.down_b, g, b, Hb, hp.eps);
        pm(stream, 4);
        if (!res) {
            hip::eventRecord(ev_free_[s], stream);
            if (l + 2 < hp.n_layer) {
                hip::streamWaitEvent(cs, ev_free_[s]);
                hip::copyAnyAsync(slot[s], host0 + layer0_ + static_cast<u64>(l + 2) * lb, lb, cs);
                hip::eventRecord(ev_copy_[s], cs);
            } else if (l + 2 == hp.n_layer) {
                // slot 1 free for good: merger weights into W + lb .. (overlaps the last layer)
                hip::streamWaitEvent(cs, ev_free_[1]);
                hip::copyAnyAsync(W + lb, host0 + merger_, mo_.bytes, cs);
                hip::eventRecord(ev_copy_[1], cs);
            }
        }
    }
    // merger: H viewed as [N/4][4E] -> mm0 -> GELU -> mm2 (+b) -> T
    const u32 MI = 4 * E;
    launchGemm(stream, mbase + mo_.mm0_w, Hb, T, MI, MI, n_tok);
    hip::launch(f_.gelu, Dim3{n_tok, 1, 1}, Dim3{256, 1, 1}, 0, stream, T, mbase + mo_.mm0_b, M, static_cast<i32>(MI), static_cast<i32>(MI));
    launchGemm(stream, mbase + mo_.mm2_w, M, T, MI, hp.proj_dim, n_tok);
    const u32 n_out = n_tok * hp.proj_dim;
    hip::launch(f_.bias, Dim3{(n_out + 255) / 256, 1, 1}, Dim3{256, 1, 1}, 0, stream, T, mbase + mo_.mm2_b, T, static_cast<i32>(hp.proj_dim),
                static_cast<i32>(n_out));
    pm(stream, 6);
    hip::streamSync(stream);  // pageable destination: copy only after the merger finished
    hip::downloadAsync(out, T, static_cast<std::size_t>(n_out) * 4, stream);
    hip::streamSync(stream);
    if (!res) hip::streamSync(cs);
    st.ms_total = msSince(t0);
    last_use_ = nowSeconds();
    n_encodes += 1;
    return st;
}

// ---------------------------------------------------------------------------

std::uint32_t tokenId(const std::array<std::uint8_t, 32>& h, std::uint32_t i) {
    return qwen35::image_id_base | static_cast<std::uint32_t>(wyhash(0x69760000ull + i, h) & 0x7fffffffull);
}

std::vector<std::uint32_t> expand(std::span<const std::uint32_t> tokens, std::uint32_t pad_id, std::span<const Prepared* const> imgs,
                                  std::span<qwen35::VisSpan> spans) {
    std::vector<std::uint32_t> out;
    out.reserve(tokens.size());
    std::size_t k = 0;
    i32 delta = 0;
    for (std::uint32_t t : tokens) {
        if (t != pad_id) {
            out.push_back(t);
            continue;
        }
        if (k >= imgs.size() || k >= spans.size()) throw VisionError("ImagePlaceholderMismatch");
        const Prepared* p = imgs[k];
        const u32 n = p->nTokens();
        spans[k] = qwen35::VisSpan{static_cast<u32>(out.size()), n, p->nx, p->ny, delta, nullptr};
        for (u32 i = 0; i < n; ++i) out.push_back(tokenId(p->hash, i));
        delta += static_cast<i32>(n) - static_cast<i32>(std::max(p->nx, p->ny));
        k += 1;
    }
    if (k != imgs.size()) throw VisionError("ImagePlaceholderMismatch");
    return out;
}

}  // namespace whirl::vision
