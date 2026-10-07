// Weight loader and buffer / KV-pool / sequence management.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Model.load, Loader, allocKvPool,
// setupSeqs, snapshots, requantMtpQ4, buildDraftHeadEx).

#include "whirl/model.h"
#include "whirl/vram_limit.h"

#include "whirl/common.h"

#include <algorithm>
#include <format>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
inline i32 I(u64 v) { return static_cast<i32>(v); }

// Positional reads of the model file (plain buffered reads beat page-faulting
// a 16 GB mapping into pinned staging).
class Reader {
public:
    explicit Reader(const std::string& path) {
        f_ = _wfopen(widen(path).c_str(), L"rb");
        if (!f_) throw ModelError("OpenFailed", path);
        setvbuf(f_, nullptr, _IONBF, 0);
    }
    ~Reader() {
        if (f_) std::fclose(f_);
    }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    void read(void* dst, std::size_t n, u64 off) {
        if (_fseeki64(f_, static_cast<long long>(off), SEEK_SET) != 0) throw ModelError("Truncated", "seek");
        if (std::fread(dst, 1, n, f_) != n) throw ModelError("Truncated", "short read");
    }

private:
    std::FILE* f_ = nullptr;
};

}  // namespace

// Reads tensors into device memory (and validates their types).
struct Loader {
    Model& m;
    const gguf::File& f;
    Reader file;
    std::uint8_t* staging;
    std::size_t staging_len;
    LoadStats& stats;

    DevPtr upload(const gguf::TensorInfo& t) {
        const u64 nbytes = t.nbytes();
        if (nbytes == 0) throw ModelError("UnsupportedTensorType", std::string(t.name));
        const DevPtr dst = m.alloc(nbytes);
        uploadTo(t, dst);
        return dst;
    }

    // Reads tensor t into the existing device buffer dst (t.nbytes() bytes).
    void uploadTo(const gguf::TensorInfo& t, DevPtr dst) {
        const u64 nbytes = t.nbytes();
        u64 off = 0;
        const u64 base = f.absOffset(t);
        while (off < nbytes) {
            const std::size_t n = static_cast<std::size_t>(std::min<u64>(nbytes - off, staging_len));
            file.read(staging, n, base + off);
            hip::upload(dst + off, staging, n);
            off += n;
        }
        stats.bytes += nbytes;
        stats.tensors += 1;
        m.type_bytes[ti(t.type) % n_types] += nbytes;
    }

    // Like mat(), but the data stays in pinned host memory (device-accessible).
    Mat matHost(const char* name, void** keep) {
        const gguf::TensorInfo* t = f.tensor(name);
        if (!t) throw ModelError("MissingTensor", name);
        const u64 nbytes = t->nbytes();
        void* buf = hip::hostMalloc(static_cast<std::size_t>(nbytes));
        *keep = buf;
        file.read(buf, static_cast<std::size_t>(nbytes), f.absOffset(*t));
        stats.bytes += nbytes;
        stats.tensors += 1;
        Mat r;
        r.ptr = reinterpret_cast<std::uintptr_t>(buf);
        r.ty = t->type;
        r.ncols = static_cast<u32>(t->ne[0]);
        r.nrows = static_cast<u32>(t->rows());
        r.row_bytes = t->rowBytes();
        return r;
    }

    // MXFP4: file blocks (e, qs[16]) x 32 values -> per 256 values e[8] then
    // qs[8][16] (same byte count), plus each row's reference exponent (its
    // largest block exponent) for the fp8 prefill GEMM.
    DevPtr uploadMx(const gguf::TensorInfo& t, DevPtr* ref_out) {
        const u64 ncols = t.ne[0];
        if (ncols % 256 != 0) throw ModelError("UnsupportedTensorType", std::string(t.name) + ": MXFP4 row not a multiple of 256");
        if (ncols > 0x7FFFFFFF) throw ModelError("UnsupportedTensorType", std::string(t.name) + ": MXFP4 row too long");
        const u64 rb = t.rowBytes();
        const u64 nrows = t.rows();
        const u64 nbytes = rb * nrows;
        const DevPtr dst = m.alloc(nbytes);
        std::vector<std::uint8_t> refs(static_cast<std::size_t>(nrows));
        std::vector<std::uint8_t> tmp(static_cast<std::size_t>(rb));
        const u64 rows_per = std::max<u64>(1, staging_len / rb);
        const u64 base = f.absOffset(t);
        u64 lossy = 0;
        for (u64 r0 = 0; r0 < nrows;) {
            const u64 nr = std::min(rows_per, nrows - r0);
            const std::size_t n = static_cast<std::size_t>(nr * rb);
            file.read(staging, n, base + r0 * rb);
            for (u64 ri = 0; ri < nr; ++ri) {
                std::uint8_t* row = staging + ri * rb;
                const std::uint8_t emax = kernels::repackMxfp4Row(row, row, static_cast<int>(ncols), &lossy, tmp.data());
                refs[static_cast<std::size_t>(r0 + ri)] = emax;
            }
            hip::upload(dst + r0 * rb, staging, n);
            r0 += nr;
        }
        const DevPtr rdev = m.alloc(nrows);
        hip::upload(rdev, refs.data(), refs.size());
        *ref_out = rdev;
        m.mx_fold_lossy += lossy;
        stats.bytes += nbytes;
        stats.tensors += 1;
        m.type_bytes[ti(t.type) % n_types] += nbytes;
        return dst;
    }

    Mat mat(const std::string& name) {
        const gguf::TensorInfo* t = f.tensor(name);
        if (!t) throw ModelError("MissingTensor", name);
        if (m.k.gemv1[ti(t->type) % n_types] == nullptr)
            throw ModelError("UnsupportedTensorType", name + ": no kernel for type " + gguf::typeName(t->type));
        Mat r;
        r.ty = t->type;
        r.ncols = static_cast<u32>(t->ne[0]);
        r.nrows = static_cast<u32>(t->rows());
        r.row_bytes = t->rowBytes();
        if (t->type == GgmlType::mxfp4) {
            r.ptr = uploadMx(*t, &r.ref);
        } else {
            r.ptr = upload(*t);
        }
        return r;
    }

    // Two matrices with the same row layout in one allocation: a's rows, then
    // b's (a.ptr / b.ptr point at the two parts; bytes are unchanged, so a GEMV
    // on either reads exactly what mat() would give). Falls back to two mat()
    // loads unless the types / ncols / row_bytes match, the type is not MXFP4
    // (repacked + per-row refs) and b's start stays 256-byte aligned.
    std::pair<Mat, Mat> matPair(const std::string& na, const std::string& nb, bool* contig) {
        *contig = false;
        const gguf::TensorInfo* ta = f.tensor(na);
        const gguf::TensorInfo* tb = f.tensor(nb);
        if (!ta || !tb || ta->type != tb->type || ta->type == GgmlType::mxfp4 || ta->ne[0] != tb->ne[0] ||
            ta->rowBytes() != tb->rowBytes() || ta->nbytes() == 0 || tb->nbytes() == 0 || ta->nbytes() % 256 != 0 ||
            ta->nbytes() != ta->rowBytes() * ta->rows() || tb->nbytes() != tb->rowBytes() * tb->rows() ||
            m.k.gemv1[ti(ta->type) % n_types] == nullptr)
            return {mat(na), mat(nb)};
        const DevPtr base = m.alloc(ta->nbytes() + tb->nbytes());
        uploadTo(*ta, base);
        uploadTo(*tb, base + ta->nbytes());
        auto mk = [](const gguf::TensorInfo& t, DevPtr p) {
            Mat r;
            r.ptr = p;
            r.ty = t.type;
            r.ncols = static_cast<u32>(t.ne[0]);
            r.nrows = static_cast<u32>(t.rows());
            r.row_bytes = t.rowBytes();
            return r;
        };
        *contig = true;
        return {mk(*ta, base), mk(*tb, base + ta->nbytes())};
    }

    DevPtr vec(const std::string& name) {
        const gguf::TensorInfo* t = f.tensor(name);
        if (!t) throw ModelError("MissingTensor", name);
        if (t->type != GgmlType::f32) throw ModelError("UnsupportedTensorType", name + ": expected f32, got " + gguf::typeName(t->type));
        return upload(*t);
    }

    static std::string bn(u32 l, const char* suffix) { return "blk." + std::to_string(l) + "." + suffix; }

    // Routed experts of block il (qwen35moe).
    MoeW moe(u32 il) {
        const Config& cfg = m.cfg;
        const gguf::TensorInfo* rt = f.tensor(bn(il, "ffn_gate_inp.weight"));
        if (!rt) throw ModelError("MissingTensor", bn(il, "ffn_gate_inp.weight"));
        if (rt->type != GgmlType::f32 || rt->ne[0] != cfg.n_embd || rt->ne[1] != cfg.n_expert)
            throw ModelError("UnsupportedTensorType", bn(il, "ffn_gate_inp.weight"));
        MoeW w;
        const DevPtr router_ptr = vec(bn(il, "ffn_gate_inp.weight"));
        w.sh_gate = f.tensor(bn(il, "ffn_gate_inp_shexp.weight")) ? vec(bn(il, "ffn_gate_inp_shexp.weight")) : 0;
        w.router.ptr = router_ptr;
        w.router.ty = GgmlType::f32;
        w.router.ncols = cfg.n_embd;
        w.router.nrows = cfg.n_expert;
        w.router.row_bytes = static_cast<u64>(cfg.n_embd) * 4;
        w.gate = mat(bn(il, "ffn_gate_exps.weight"));
        w.up = mat(bn(il, "ffn_up_exps.weight"));
        w.down = mat(bn(il, "ffn_down_exps.weight"));
        const std::pair<const char*, const Mat*> parts[] = {{"gate", &w.gate}, {"up", &w.up}, {"down", &w.down}};
        for (const auto& [fld, e] : parts) {
            if (m.k.moe_gu[ti(e->ty) % n_types] == nullptr)
                throw ModelError("UnsupportedTensorType", "blk." + std::to_string(il) + " ffn_" + fld + "_exps: no MoE kernel for type " +
                                                              gguf::typeName(e->ty));
        }
        if (w.gate.nrows != cfg.n_ff_exp * cfg.n_expert || w.up.nrows != w.gate.nrows || w.gate.ncols != cfg.n_embd ||
            w.down.ncols != cfg.n_ff_exp || w.down.nrows != cfg.n_embd * cfg.n_expert || cfg.n_expert > 1024 ||
            cfg.n_expert_used > 64 || cfg.n_ff_exp % 256 != 0 || cfg.n_embd % 256 != 0)
            throw ModelError("UnsupportedMoeShape", "blk." + std::to_string(il));
        return w;
    }
};

// ---------------------------------------------------------------------------

Model::~Model() {
    decode_graph.reset();
    if (rpos_buf != 0) hip::free(rpos_buf);
    for (DevPtr p : allocations) hip::free(p);
    allocations.clear();
    if (embd_host) hip::hostFree(embd_host);
    if (stream) hip::streamDestroy(stream);
}

DevPtr Model::alloc(u64 bytes) {
    const DevPtr p = hip::malloc(static_cast<std::size_t>(bytes));
    allocations.push_back(p);
    return p;
}

DevPtr Model::allocZero(u64 bytes) {
    const DevPtr p = alloc(bytes);
    hip::memset(p, 0, static_cast<std::size_t>(bytes));
    return p;
}

void Model::freeAlloc(DevPtr p) {
    for (std::size_t i = 0; i < allocations.size(); ++i)
        if (allocations[i] == p) {
            allocations[i] = allocations.back();
            allocations.pop_back();
            hip::free(p);
            return;
        }
}

u64 Model::convBytes() const { return static_cast<u64>(cfg.d_conv - 1) * cfg.convCh() * 4; }
u64 Model::ssmBytes() const { return static_cast<u64>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4; }

void Model::ensureSnapshots(u32 n) {
    const u32 want = std::min(n, gdn_max_snap);
    while (snap_sets < want) {
        const u32 kk = snap_sets;
        std::vector<DevPtr> conv(cfg.n_layer, 0), ssm(cfg.n_layer, 0);
        for (u32 i = 0; i < cfg.n_layer; ++i) {
            if (ssm_state[i] == 0) continue;
            conv[i] = allocZero(convBytes());
            ssm[i] = allocZero(ssmBytes());
        }
        snap_conv[kk] = std::move(conv);
        snap_ssm[kk] = std::move(ssm);
        snap_sets += 1;
    }
}

namespace {

// LoadOptions / WHIRL_KV -> the forced format for numerics::chooseKv (automatic: the mode decides)
std::optional<numerics::KvKind> kvKindOf(KvMode m) {
    switch (m) {
        case KvMode::f16: return numerics::KvKind::f16;
        case KvMode::q8: return numerics::KvKind::q8;
        case KvMode::q8h: return numerics::KvKind::q8h;
        case KvMode::q8v: return numerics::KvKind::q8v;
        case KvMode::q4: return numerics::KvKind::q4;
        case KvMode::automatic: break;
    }
    return std::nullopt;
}

std::string readWhole(const std::string& path) { return readFile(path); }

}  // namespace

std::optional<KvMode> kvModeFromEnv() {
    const auto v = envGet("KV");
    if (!v) return std::nullopt;
    if (*v == "auto") return KvMode::automatic;
    if (*v == "f16") return KvMode::f16;
    if (*v == "q8") return KvMode::q8;
    if (*v == "q8h") return KvMode::q8h;
    if (*v == "q8v") return KvMode::q8v;
    if (*v == "q4") return KvMode::q4;
    throw std::invalid_argument("WHIRL_KV must be auto, f16, q8, q8h, q8v or q4 (got '" + *v + "')");
}

u64 bufferBytes(const Config& cfg, u32 Bq, u64 ffs, u64 max_elems) {
    // mirrors the buffer allocations of Model::load below (keep in step)
    const u64 f4 = 4;
    const u64 B = Bq;
    u64 n = 0;
    n += B * cfg.n_embd * f4 * 2;                         // x, h
    n += B * cfg.n_head * cfg.head_dim * 2 * f4;          // qf
    n += B * cfg.n_head_kv * cfg.head_dim * f4 * 2;       // kv_k, kv_v
    n += B * cfg.n_head * cfg.head_dim * f4;              // attn_out
    n += B * cfg.convCh() * f4 * 2;                       // qkv, conv_out
    n += B * cfg.d_inner * f4;                            // z
    n += B * cfg.n_v_heads * f4 * 2;                      // beta, alpha
    n += B * 2 * cfg.n_v_heads * f4;                      // ba_buf
    n += B * cfg.d_inner * f4;                            // gdn_out
    n += B * ffs * f4 * 2;                                // ffn_g, ffn_u
    n += static_cast<u64>(max_verify_rows) * cfg.n_vocab * f4;  // logits
    n += B * cfg.n_embd * f4;                             // hn
    n += B * 2 * cfg.n_embd * f4;                         // mtp_cat
    n += static_cast<u64>(max_small_batch) * cfg.n_embd * f4;  // mtp_h
    n += B * 4 * 3;                                       // mtp_ids, ids, pos_buf
    n += ctl_words * 4;                                   // out_tok
    const u64 n_split = 64;
    n += max_verify_rows * n_split * cfg.n_head * 2 * f4;  // part_ml
    n += max_verify_rows * n_split * cfg.n_head * cfg.head_dim * f4;  // part_acc
    n += max_elems * 2;                                   // w16
    {
        const u64 n_chunks = (B + 63) / 64;
        const u64 heads = cfg.n_v_heads;
        n += n_chunks * heads * (64 * 128 * 4 * 2 + 64 * 64 * 4 + 64 * 4);  // gc_w, gc_u, gc_m, gc_g
    }
    const u64 in_max = std::max<u64>(std::max<u64>(cfg.n_ff, 2ull * cfg.n_embd), std::max<u64>(cfg.d_inner, static_cast<u64>(cfg.n_head) * cfg.head_dim));
    n += max_verify_rows * in_max + 256;                  // xq
    n += (max_verify_rows * in_max / 32 + 8) * 4;         // xd
    n += B * std::max<u64>(in_max, cfg.n_embd) * 2;       // x16
    n += B * 4;                                           // sx8
    if (cfg.moe) {
        const u64 Kx = cfg.n_expert_used;
        const u64 F = cfg.n_ff_exp;
        const u64 E = cfg.n_embd;
        n += B * cfg.n_expert * f4;                       // moe_logits
        n += B * Kx * 4 + B * Kx * f4 + B * f4;           // moe_ids, moe_w, moe_sg
        n += max_small_batch * Kx * F * f4 * 2;           // moe_g, moe_u
        n += max_small_batch * Kx * F + 256;              // moe_xq
        n += (max_small_batch * Kx * F / 32 + 8) * 4;     // moe_xd
        n += B * E * f4;                                  // moe_ysh
        n += B * Kx * 4 * 2;                              // moe_perm, moe_inv
        n += (B * Kx / 32 + cfg.n_expert + 1) * 16;       // moe_tiles
        n += 16;                                          // moe_ntiles
        n += B * Kx * std::max(E, F) * 2;                 // moe_x16
        n += B * Kx * F * f4 * 2;                         // moe_yg, moe_yu
        n += B * Kx * E * f4;                             // moe_yd
        n += B * Kx * f4;                                 // moe_sx
    }
    return n;
}

u64 kvBytesPerTokenCfg(const Config& cfg, bool has_mtp, bool q8, bool kf16, bool q4) {
    u64 layers = has_mtp ? 1 : 0;
    for (u32 i = 0; i < cfg.n_layer; ++i)
        if (cfg.isAttn(i)) layers += 1;
    const u64 e = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
    if (q4) return layers * 2 * (e / 2 + e / 32 * 2);
    const u64 q = e + e / 32 * 2;
    return layers * ((q8 && !kf16 ? q : e * 2) + (q8 ? q : e * 2));
}

LoadEstimate estimateLoad(const gguf::File& f, const Config& cfg, bool embd_on_host) {
    LoadEstimate r;
    const gguf::TensorInfo* emb = f.tensor("token_embd.weight");
    r.embd_on_host = embd_on_host && emb != nullptr && f.tensor("output.weight") != nullptr && emb->type != GgmlType::mxfp4;
    r.has_mtp = cfg.n_nextn > 0 && f.tensor(Loader::bn(cfg.n_layer, "nextn.eh_proj.weight")) != nullptr;
    static constexpr std::string_view mats[] = {"attn_q.weight",  "attn_k.weight",    "attn_v.weight",   "attn_output.weight",
                                                "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
                                                "ssm_out.weight", "ffn_gate.weight",  "ffn_up.weight",   "ffn_down.weight",
                                                "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    r.ffs = cfg.n_ff;
    for (const gguf::TensorInfo& t : f.tensors()) {
        if (r.embd_on_host && t.name == "token_embd.weight") continue;
        r.weights += t.nbytes();
        if (t.type == GgmlType::mxfp4) r.weights += t.rows();  // per-row reference exponents
        // layer matrices of the main blocks (not the nextn block): scratch sizes
        if (t.n_dims != 2 || t.name.rfind("blk.", 0) != 0) continue;
        const std::size_t dot = t.name.find('.', 4);
        if (dot == std::string_view::npos || dot == 4) continue;
        u32 il = 0;
        bool num = true;
        for (std::size_t i = 4; i < dot; ++i) {
            num = num && t.name[i] >= '0' && t.name[i] <= '9';
            il = il * 10 + static_cast<u32>(t.name[i] - '0');
        }
        if (!num || il >= cfg.n_layer) continue;
        const std::string_view suffix = t.name.substr(dot + 1);
        bool is_mat = false;
        for (std::string_view mn : mats) is_mat = is_mat || suffix == mn;
        if (!is_mat) continue;
        r.ffs = std::max<u64>(r.ffs, std::max(t.ne[0], t.ne[1]));
        r.max_elems = std::max<u64>(r.max_elems, t.ne[0] * t.ne[1]);
    }
    for (u32 il = 0; il < cfg.n_layer; ++il)
        if (!cfg.isAttn(il))
            r.state += static_cast<u64>(cfg.d_conv - 1) * cfg.convCh() * 4 + static_cast<u64>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4;
    r.kv_f16 = kvBytesPerTokenCfg(cfg, r.has_mtp, false, false);
    r.kv_q8v = kvBytesPerTokenCfg(cfg, r.has_mtp, true, true);
    r.kv_q8h = kvBytesPerTokenCfg(cfg, r.has_mtp, true, false);
    return r;
}

std::unique_ptr<Model> Model::load(const gguf::File& f, u32 max_ctx_req, LoadStats& stats, const LoadOptions& opt) {
    auto mp = std::make_unique<Model>();
    Model& m = *mp;
    m.cfg = Config::fromGguf(f);
    const Config& cfg = m.cfg;
    // every tensor's type and shape against cfg, before any GPU allocation
    cfg.validateTensors(f);
    std::string co_path = opt.code_object;
    if (co_path.empty()) {
        if (auto v = envGet("CODE_OBJECT")) co_path = *v;
    }
    if (!co_path.empty()) {
        // development override: an external code object built from the kernel sources
        static std::string image;  // must outlive the module
        image = readWhole(co_path);
        const hip::DeviceInfo d = hip::describeDevice(hip::getDevice());
        m.module = hip::Module::loadData(image.data(), hip::archFor(d.gcn_arch));
    } else {
        m.module = hip::Module::loadEmbedded();
    }
    m.arch = m.module.arch();
    m.stream = hip::streamCreate();
    m.max_ctx = max_ctx_req;
    m.max_batch = std::clamp<u32>(opt.max_batch, 1, max_batch_limit);
    KvMode kvm = opt.kv_mode;
    if (kvm == KvMode::automatic) kvm = kvModeFromEnv().value_or(KvMode::automatic);
    m.kv_mode = kvm;
    {
        // KV format: fixed per mode x model type (numerics::chooseKv, shared by CLI and server;
        // WHIRL_KV is a debug override). Decided once, here, before the kernel table is loaded,
        // and never switched afterwards (no fallback to a lower-precision format when it does
        // not fit: the context shrinks or the load stops, below / in the server).
        const kernels::Caps caps = kernels::Caps::probe(m.module);
        m.kv_choice = numerics::chooseKv(opt.numerics, cfg.moe, numerics::KvCaps{caps.kv_q8h, caps.kv_q8v, caps.kv_q4}, kvKindOf(kvm));
        const numerics::KvKind kv = m.kv_choice.kv;
        m.kv_q8 = kv != numerics::KvKind::f16;
        m.kv_rot = kv == numerics::KvKind::q8h || kv == numerics::KvKind::q4;
        m.kv_kf16 = kv == numerics::KvKind::q8v;
        m.kv_q4 = kv == numerics::KvKind::q4;
        // an explicit KV format the code object has no kernels for (gfx1201: q4)
        if ((m.kv_kf16 && !caps.kv_q8v) || (m.kv_q4 && !caps.kv_q4) || (m.kv_rot && !m.kv_q4 && !caps.kv_q8h))
            throw ModelError("UnsupportedKvFormat", std::string(m.kv_kf16 ? "q8v" : m.kv_q4 ? "q4" : "q8h") +
                                                        " KV needs kernels this GPU does not have; use WHIRL_KV=f16, q8 or auto");
    }
    m.k = loadKernels(m.module, m.kv_q8, m.kv_rot, m.kv_kf16, m.kv_q4);
    // MXFP4 routed experts: process-level switches (as WHIRL_KV), read by CLI and server alike
    m.num_req = opt.numerics;
    // MoE expert fp8: a balance item (WHIRL_MOE_FP8 overrides the mode)
    m.moe_fp8 = envFlag("MOE_FP8", opt.numerics.has(numerics::Item::moefp8));
    m.moe_mxw = envFlag("MOE_MXW", true);
    m.moe_rbf = envFlag("MOE_RBF", true);
    m.layers.resize(cfg.n_layer);
    m.kcache.assign(cfg.n_layer, 0);
    m.vcache.assign(cfg.n_layer, 0);
    m.kscale.assign(cfg.n_layer, 0);
    m.vscale.assign(cfg.n_layer, 0);
    m.state_tables.emplace_back(cfg.n_layer, 0);
    m.conv_state = std::span<DevPtr>(m.state_tables.back());
    m.state_tables.emplace_back(cfg.n_layer, 0);
    m.ssm_state = std::span<DevPtr>(m.state_tables.back());
    if (m.arch == hip::Arch::gfx1151) m.gemv_r = gfx1151GemvR();

    const std::size_t staging_len = 64u << 20;
    auto* staging = static_cast<std::uint8_t*>(hip::hostMalloc(staging_len));
    struct StagingFree {
        void* p;
        ~StagingFree() { hip::hostFree(p); }
    } staging_free{staging};
    Loader ld{m, f, Reader(f.path()), staging, staging_len, stats};

    // fit check before any weight is uploaded: weights + state + buffers against the
    // free VRAM (the simulated card's under WHIRL_VRAM_LIMIT_MB); a load that cannot fit
    // fails here with the sizes instead of an out-of-memory error part way through
    // (not on the UMA iGPU, gfx1151, unless a limit is set: its free VRAM does not bound what it can allocate)
    if (m.arch != hip::Arch::gfx1151 || vram::limitBytes() != 0) {
        const LoadEstimate est = estimateLoad(f, cfg, opt.embd_on_host);
        const u64 buf = bufferBytes(cfg, m.max_batch, est.ffs, est.max_elems);
        const u64 need = est.weights + est.state + buf;
        const hip::MemInfo mi = hip::memInfo();
        if (need > mi.free) {
            const double g = 1024.0 * 1024.0 * 1024.0;
            const std::string lim = vram::limitNote();
            throw ModelError("VramLimit",
                             std::format("needs about {:.2f} GiB of VRAM (weights {:.2f} GiB + buffers for prefill batch {} {:.2f} GiB + state "
                                         "{:.2f} GiB), {:.2f} GiB free of {:.2f} GiB{}",
                                         need / g, est.weights / g, m.max_batch, buf / g, est.state / g, mi.free / g, mi.total / g,
                                         lim.empty() ? std::string() : " (" + lim + ")"));
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const gguf::TensorInfo* emb = f.tensor("token_embd.weight");
    if (!emb) throw ModelError("MissingTensor", "token_embd.weight");
    // (MXFP4 is repacked at load, so it cannot be read straight into host memory)
    if (opt.embd_on_host && f.tensor("output.weight") != nullptr && emb->type != GgmlType::mxfp4) {
        m.tok_embd = ld.matHost("token_embd.weight", &m.embd_host);
    } else {
        m.tok_embd = ld.mat("token_embd.weight");
    }
    m.output = f.tensor("output.weight") ? ld.mat("output.weight") : m.tok_embd;
    m.output_norm = ld.vec("output_norm.weight");

    for (u32 il = 0; il < cfg.n_layer; ++il) {
        Layer& L = m.layers[il];
        auto n = [&](const char* s) { return Loader::bn(il, s); };
        L.attn_norm = ld.vec(n("attn_norm.weight"));
        L.post_norm = ld.vec(n("post_attention_norm.weight"));
        L.moe.reset();
        if (cfg.moe) {
            L.ffn_gate = ld.mat(n("ffn_gate_shexp.weight"));
            L.ffn_up = ld.mat(n("ffn_up_shexp.weight"));
            L.ffn_down = ld.mat(n("ffn_down_shexp.weight"));
            L.moe = ld.moe(il);
        } else {
            L.ffn_gate = ld.mat(n("ffn_gate.weight"));
            L.ffn_up = ld.mat(n("ffn_up.weight"));
            L.ffn_down = ld.mat(n("ffn_down.weight"));
        }
        if (cfg.isAttn(il)) {
            L.kind = LayerKind::attn;
            L.attn.q = ld.mat(n("attn_q.weight"));
            L.attn.k = ld.mat(n("attn_k.weight"));
            L.attn.v = ld.mat(n("attn_v.weight"));
            L.attn.o = ld.mat(n("attn_output.weight"));
            L.attn.q_norm = ld.vec(n("attn_q_norm.weight"));
            L.attn.k_norm = ld.vec(n("attn_k_norm.weight"));
        } else {
            L.kind = LayerKind::gdn;
            L.gdn.qkv = ld.mat(n("attn_qkv.weight"));
            L.gdn.gate = ld.mat(n("attn_gate.weight"));
            bool ba_contig = false;
            std::tie(L.gdn.beta, L.gdn.alpha) = ld.matPair(n("ssm_beta.weight"), n("ssm_alpha.weight"), &ba_contig);
            if (ba_contig && rowsContiguous(L.gdn.beta, L.gdn.alpha)) stats.gdn_ba_contig += 1;
            L.gdn.out = ld.mat(n("ssm_out.weight"));
            L.gdn.conv = ld.vec(n("ssm_conv1d.weight"));
            L.gdn.dt = ld.vec(n("ssm_dt.bias"));
            L.gdn.a = ld.vec(n("ssm_a"));
            L.gdn.norm = ld.vec(n("ssm_norm.weight"));
            m.conv_state[il] = m.allocZero(static_cast<u64>(cfg.d_conv - 1) * cfg.convCh() * 4);
            m.ssm_state[il] = m.allocZero(static_cast<u64>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4);
        }
    }
    if (cfg.n_nextn > 0 && f.tensor(Loader::bn(cfg.n_layer, "nextn.eh_proj.weight")) != nullptr) {
        const u32 il = cfg.n_layer;
        auto n = [&](const char* s) { return Loader::bn(il, s); };
        MtpW w;
        w.attn.q = ld.mat(n("attn_q.weight"));
        w.attn.k = ld.mat(n("attn_k.weight"));
        w.attn.v = ld.mat(n("attn_v.weight"));
        w.attn.o = ld.mat(n("attn_output.weight"));
        w.attn.q_norm = ld.vec(n("attn_q_norm.weight"));
        w.attn.k_norm = ld.vec(n("attn_k_norm.weight"));
        w.attn_norm = ld.vec(n("attn_norm.weight"));
        w.post_norm = ld.vec(n("post_attention_norm.weight"));
        w.ffn_gate = ld.mat(n(cfg.moe ? "ffn_gate_shexp.weight" : "ffn_gate.weight"));
        w.ffn_up = ld.mat(n(cfg.moe ? "ffn_up_shexp.weight" : "ffn_up.weight"));
        w.ffn_down = ld.mat(n(cfg.moe ? "ffn_down_shexp.weight" : "ffn_down.weight"));
        if (cfg.moe) w.moe = ld.moe(il);
        w.eh_proj = ld.mat(n("nextn.eh_proj.weight"));
        w.enorm = ld.vec(n("nextn.enorm.weight"));
        w.hnorm = ld.vec(n("nextn.hnorm.weight"));
        w.head_norm = f.tensor(n("nextn.shared_head_norm.weight")) ? ld.vec(n("nextn.shared_head_norm.weight")) : m.output_norm;
        m.mtp = std::move(w);
    }
    const auto t1 = std::chrono::steady_clock::now();
    stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const u64 f4 = 4;
    const u64 B = m.max_batch;
    m.x = m.alloc(B * cfg.n_embd * f4);
    m.h = m.alloc(B * cfg.n_embd * f4);
    m.qf = m.alloc(B * cfg.n_head * cfg.head_dim * 2 * f4);
    m.kv_k = m.alloc(B * cfg.n_head_kv * cfg.head_dim * f4);
    m.kv_v = m.alloc(B * cfg.n_head_kv * cfg.head_dim * f4);
    m.attn_out = m.alloc(B * cfg.n_head * cfg.head_dim * f4);
    m.qkv = m.alloc(B * cfg.convCh() * f4);
    m.conv_out = m.alloc(B * cfg.convCh() * f4);
    m.z = m.alloc(B * cfg.d_inner * f4);
    m.beta = m.alloc(B * cfg.n_v_heads * f4);
    m.alpha = m.alloc(B * cfg.n_v_heads * f4);
    m.ba_buf = m.alloc(B * 2 * cfg.n_v_heads * f4);
    m.gdn_out = m.alloc(B * cfg.d_inner * f4);
    // ffn_g / ffn_u double as autotune / bench scratch for every matrix
    u32 ffs = cfg.n_ff;
    for (const Layer& L : m.layers)
        for (const Mat& w : m.layerMats(L).slice()) ffs = std::max(ffs, std::max(w.nrows, w.ncols));
    m.ff_scratch = ffs;
    m.ffn_g = m.alloc(B * ffs * f4);
    m.ffn_u = m.alloc(B * ffs * f4);
    m.logits = m.alloc(static_cast<u64>(max_verify_rows) * cfg.n_vocab * f4);
    m.hn = m.alloc(B * cfg.n_embd * f4);
    m.mtp_cat = m.alloc(B * 2 * cfg.n_embd * f4);
    m.mtp_h = m.alloc(static_cast<u64>(max_small_batch) * cfg.n_embd * f4);
    m.mtp_ids = m.alloc(B * 4);
    m.ids = m.alloc(B * 4);
    m.host_ids.assign(static_cast<std::size_t>(2 * B), 0);
    m.pos_buf = m.alloc(B * 4);
    m.out_tok = m.allocZero(ctl_words * 4);
    m.fd_splits = m.kv_q4 ? 256 : 64;
    if (const auto v = envGet("FD_SPLITS")) {
        const unsigned long s = std::strtoul(v->c_str(), nullptr, 10);
        if (s >= 64 && s <= 256 && s % 64 == 0) m.fd_splits = static_cast<u32>(s);
    }
    const u64 n_split = m.fd_splits;
    m.part_ml = m.alloc(max_verify_rows * n_split * cfg.n_head * 2 * f4);
    m.part_acc = m.alloc(max_verify_rows * n_split * cfg.n_head * cfg.head_dim * f4);
    {
        u64 max_elems = 0;
        for (const Layer& L : m.layers)
            for (const Mat& w : m.layerMats(L).slice()) max_elems = std::max(max_elems, static_cast<u64>(w.ncols) * w.nrows);
        m.w16 = m.alloc(max_elems * 2);
    }
    {
        const u64 n_chunks = (B + 63) / 64;
        const u64 heads = cfg.n_v_heads;
        m.gc_w = m.alloc(n_chunks * heads * 64 * 128 * 4);
        m.gc_u = m.alloc(n_chunks * heads * 64 * 128 * 4);
        m.gc_m = m.alloc(n_chunks * heads * 64 * 64 * 4);
        m.gc_g = m.alloc(n_chunks * heads * 64 * 4);
    }
    // widest decode matmul input: FFN, DeltaNet / attention output, MTP eh_proj
    const u64 in_max = std::max<u64>(std::max<u64>(cfg.n_ff, 2ull * cfg.n_embd), std::max<u64>(cfg.d_inner, static_cast<u64>(cfg.n_head) * cfg.head_dim));
    m.xq = m.alloc(max_verify_rows * in_max + 256);
    m.xd = m.alloc((max_verify_rows * in_max / 32 + 8) * 4);
    m.x16_bytes = B * std::max<u64>(in_max, cfg.n_embd) * 2;
    m.x16 = m.alloc(m.x16_bytes);
    // GDN prefill x16b (fp8 rows + f16 rows from one norm) must fit in x16 for any batch > 16
    if (B > 16 && m.x16bOff() + B * cfg.n_embd * 2 > m.x16_bytes) throw ModelError("X16bOverflow");
    m.sx8 = m.alloc(B * 4);
    m.fp8_prefill = opt.numerics.has(numerics::Item::fp8);  // balance item (WHIRL_FP8 overrides in loadOrTune)
    if (cfg.moe) {
        const u64 Kx = cfg.n_expert_used;
        const u64 F = cfg.n_ff_exp;
        const u64 E = cfg.n_embd;
        m.moe_logits = m.alloc(B * cfg.n_expert * f4);
        m.moe_ids = m.allocZero(B * Kx * 4);
        m.moe_w = m.allocZero(B * Kx * f4);
        m.moe_sg = m.allocZero(B * f4);
        // precise mode runs verify batches up to max_verify_rows rows on the decode experts
        m.moe_g = m.alloc(max_verify_rows * Kx * F * f4);
        m.moe_u = m.alloc(max_verify_rows * Kx * F * f4);
        m.moe_xq = m.alloc(max_small_batch * Kx * F + 256);
        m.moe_xd = m.alloc((max_small_batch * Kx * F / 32 + 8) * 4);
        m.moe_ysh = m.allocZero(B * E * f4);
        m.moe_perm = m.alloc(B * Kx * 4);
        m.moe_inv = m.alloc(B * Kx * 4);
        m.moe_tiles = m.alloc((B * Kx / 32 + cfg.n_expert + 1) * 16);
        m.moe_ntiles = m.allocZero(16);
        m.moe_tiles64 = m.alloc((B * Kx / 64 + cfg.n_expert + 1) * 16);
        m.moe_ntiles64 = m.allocZero(16);
        m.moe_x16 = m.alloc(B * Kx * std::max(E, F) * 2);
        m.moe_yg = m.alloc(B * Kx * F * f4);
        m.moe_yu = m.alloc(B * Kx * F * f4);
        m.moe_yd = m.alloc(B * Kx * E * f4);
        m.moe_sx = m.alloc(B * Kx * f4);
    }
    // KV pool for max_ctx tokens, one sequence with an identity page table (the
    // server loads with max_ctx 0 and allocates its shared pool later). The format chosen above
    // must fit next to what the CLI still allocates (MTP snapshot sets for 8 drafts, the 2-bit
    // draft head) plus a 768 MiB margin: else the context shrinks when it was not given, or the
    // load stops (never a lower-precision format). WHIRL_KV (debug): no check, as before.
    if (max_ctx_req > 0) {
        if (!m.kv_choice.debug) {
            const hip::MemInfo mi = hip::memInfo();
            u64 n_gdn = 0;
            for (u32 i = 0; i < cfg.n_layer; ++i)
                if (m.ssm_state[i] != 0) n_gdn += 1;
            const u64 later = 8 * n_gdn * (m.convBytes() + m.ssmBytes()) + static_cast<u64>(cfg.n_vocab) * cfg.n_embd * 5 / 16 + (768ull << 20);
            const numerics::CtxFit cf = numerics::cliCtxFit(max_ctx_req, opt.ctx_explicit, opt.min_ctx, mi.free, later, m.kvBytesPerToken(), kv_page,
                                                            numerics::kvFitLabel(opt.numerics, cfg.moe, m.kv_choice.kv));
            if (cf.refuse) throw ModelError("KvDoesNotFit", cf.msg);
            if (cf.shrunk) {
                max_ctx_req = cf.ctx;
                m.max_ctx = cf.ctx;
                m.load_note = "warning: " + cf.msg + "\n";
            }
        }
        m.allocKvPool(max_ctx_req);
    }
    hip::sync();
    return mp;
}

u64 Model::kvBytesPerTokenFmt(bool q8, bool kf16, bool q4) const {
    const u64 e = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
    if (q4) return kvLayers() * 2 * (e / 2 + e / 32 * 2);
    const u64 q = e + e / 32 * 2;
    return kvLayers() * ((q8 && !kf16 ? q : e * 2) + (q8 ? q : e * 2));
}

const char* Model::kvName() const {
    if (!kv_q8) return "f16";
    if (kv_q4) return "q4 (4-bit + f16 scale / 32, Hadamard-rotated q/k)";
    if (kv_kf16) return "q8v (K f16, V int8 + f16 scale / 32)";
    if (kv_rot) return "q8h (int8 + f16 scale / 32, Hadamard-rotated q/k)";
    return "q8 (int8 + f16 scale / 32)";
}

u32 Model::kvLayers() const {
    u32 n = 0;
    for (u32 i = 0; i < cfg.n_layer; ++i)
        if (cfg.isAttn(i)) n += 1;
    if (mtp) n += 1;
    return n;
}

void Model::allocKvPool(u32 tokens) {
    if (pool_pages != 0) throw ModelError("KvPoolAllocated");
    const u32 pages = (tokens + kv_page - 1) / kv_page;
    const u64 rows = static_cast<u64>(pages) * kv_page;
    const u64 elems = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
    const bool kq8 = kv_q8 && !kv_kf16;  // K format (q8v: f16)
    // element bytes x2 (q4: one byte per two values)
    const u64 ebk = kv_q4 ? 1 : kq8 ? 2 : 4;
    const u64 ebv = kv_q4 ? 1 : kv_q8 ? 2 : 4;
    // MTP KV first: the draft head reads it every step, so if the process ever ends up over
    // its WDDM budget it should not be the last (most likely demoted) allocation
    if (mtp) {
        mtp_kc = allocZero(rows * elems * ebk / 2);
        mtp_vc = allocZero(rows * elems * ebv / 2);
        if (kq8) mtp_ks = allocZero(rows * elems / 32 * 2);
        if (kv_q8) mtp_vs = allocZero(rows * elems / 32 * 2);
    }
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (!cfg.isAttn(i)) continue;
        kcache[i] = allocZero(rows * elems * ebk / 2);
        vcache[i] = allocZero(rows * elems * ebv / 2);
        if (kq8) kscale[i] = allocZero(rows * elems / 32 * 2);
        if (kv_q8) vscale[i] = allocZero(rows * elems / 32 * 2);
    }
    pool_pages = pages;
    if (seqs.empty()) {
        seq_pages = pages;
        ptab = alloc(static_cast<u64>(pages) * 4);
        std::vector<i32> idv(pages);
        for (u32 j = 0; j < pages; ++j) idv[j] = static_cast<i32>(j);
        hip::upload(ptab, idv.data(), idv.size() * 4);
        max_ctx = static_cast<u32>(rows);
    }
}

void Model::mapPages(u32 s, u32 first, std::span<const i32> phys) {
    if (phys.empty()) return;
    if (first + phys.size() > seq_pages) throw ModelError("ContextTooLong");
    const u64 off = (static_cast<u64>(seqs[s].kv_base) + first) * 4;
    hip::upload(ptab + off, phys.data(), phys.size() * 4);
}

KvArgs Model::kvArgs(const KvLayer& kv, DevPtr kvbase, u32 tab0) const {
    KvArgs a;
    a.k = kv.k;
    a.v = kv.v;
    a.ks = kv.ks;
    a.vs = kv.vs;
    a.ptab = ptab;
    a.kvbase = kvbase;
    a.tab0 = static_cast<i32>(tab0);
    return a;
}

void Model::reset() {
    dropPending(cur_seq);
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (conv_state[i] != 0) hip::memset(conv_state[i], 0, static_cast<std::size_t>(convBytes()));
        if (ssm_state[i] != 0) hip::memset(ssm_state[i], 0, static_cast<std::size_t>(ssmBytes()));
    }
}

void Model::setupSeqs(u32 n, u32 slot_ctx) {
    if (n == 0 || n > gdn_max_seg || pool_pages != 0 || slot_ctx < kv_page) throw ModelError("BadSlots");
    const u64 E = cfg.n_embd;
    std::vector<Seq> ss(n);
    for (u32 si = 0; si < n; ++si) {
        Seq& sq = ss[si];
        if (si == 0) {
            sq.conv = conv_state;
            sq.ssm = ssm_state;
            sq.hid = mtp_h;
            sq.kv_base = 0;
            continue;
        }
        state_tables.emplace_back(cfg.n_layer, 0);
        sq.conv = std::span<DevPtr>(state_tables.back());
        state_tables.emplace_back(cfg.n_layer, 0);
        sq.ssm = std::span<DevPtr>(state_tables.back());
        for (u32 i = 0; i < cfg.n_layer; ++i) {
            if (ssm_state[i] == 0) continue;
            sq.conv[i] = allocZero(convBytes());
            sq.ssm[i] = allocZero(ssmBytes());
        }
        sq.hid = allocZero(static_cast<u64>(max_small_batch) * E * 4);
        sq.kv_base = si * ((slot_ctx + kv_page - 1) / kv_page);
    }
    seqs = std::move(ss);
    // replay needs the replay-mode fused segment kernels (gfx1201 code object)
    if (gdn_replay && !(fusedDecode() && k.caps.gdn_replay)) gdn_replay = false;
    if (gdn_replay) {
        gdn_ord.assign(cfg.n_layer, 0);
        u32 n_gdn = 0;
        for (u32 i = 0; i < cfg.n_layer; ++i) {
            gdn_ord[i] = n_gdn;
            if (ssm_state[i] != 0) n_gdn += 1;
        }
        for (Seq& sq : seqs) sq.pend = alloc(static_cast<u64>(n_gdn) * pendLayerBytes());
    }
    seq_pages = (slot_ctx + kv_page - 1) / kv_page;
    ptab = allocZero(static_cast<u64>(n) * seq_pages * 4);
    out_tok = allocZero(static_cast<u64>(n) * ctl_words * 4);
    draft_samp.assign(n, spec::DraftSample{});
    draft_qbuf = mtp ? allocZero(static_cast<u64>(n) * max_drafts * spec::q_words * 4) : DevPtr{0};
    kvbase_buf = allocZero(static_cast<u64>(max_verify_rows) * 4);
    mtp_in = alloc(static_cast<u64>(max_verify_rows) * E * 4);
    max_ctx = slot_ctx;
    selectSeq(0);
}

u64 Model::pendLayerBytes() const {
    return pendSsmOff() + static_cast<u64>(max_small_batch) * cfg.n_v_heads * (cfg.d_state + cfg.headV() + 2) * 4;
}
u64 Model::pendSsmOff() const { return static_cast<u64>(max_small_batch) * cfg.convCh() * 4; }
u64 Model::pendPtr(const Seq& sq, std::size_t i, bool conv) const {
    const u64 base = sq.pend + static_cast<u64>(gdn_ord[i]) * pendLayerBytes();
    return conv ? base : base + pendSsmOff();
}

void Model::selectSeq(u32 s) {
    const Seq& sq = seqs[s];
    cur_seq = s;
    conv_state = sq.conv;
    ssm_state = sq.ssm;
    mtp_h = sq.hid;
    kv_off = sq.kv_base;
    ctl_off = s * ctl_words;
}

DevPtr Model::seqCtl(u32 s, u32 word) const { return out_tok + 4 * (static_cast<u64>(s) * ctl_words + word); }

void Model::restoreSeqSnapshot(u32 s, u32 set) {
    const Seq& sq = seqs[s];
    for (u32 i = 0; i < cfg.n_layer; ++i) {
        if (sq.ssm[i] == 0) continue;
        std::swap(sq.ssm[i], snap_ssm[set][i]);
        std::swap(sq.conv[i], snap_conv[set][i]);
    }
}

Mat Model::requantQ4k(const Mat& w) {
    // Q6_K (Q4_K_M files) or Q8_0 (e.g. the Swift MXFP4 files' MTP block)
    const hip::Function f = w.ty == GgmlType::q6_k ? k.requant_q6k_q4k : w.ty == GgmlType::q8_0 ? k.requant_q80_q4k : nullptr;
    if (f == nullptr || w.ncols % 256 != 0) return w;
    const u64 rb = w.ncols / 256 * 144;
    const DevPtr p = alloc(rb * w.nrows);
    hip::launch(f, {w.nrows, w.ncols / 256, 1}, {256, 1, 1}, 0, stream, w.ptr, w.row_bytes, p, rb);
    Mat r;
    r.ptr = p;
    r.ty = GgmlType::q4_k;
    r.ncols = w.ncols;
    r.nrows = w.nrows;
    r.row_bytes = rb;
    return r;
}

void Model::requantMtpQ4() {
    if (!mtp) return;
    MtpW& mw = *mtp;
    if (mw.moe) return;
    Mat* mats[] = {&mw.attn.q, &mw.attn.o, &mw.ffn_gate, &mw.ffn_up, &mw.ffn_down, &mw.eh_proj};
    for (Mat* w : mats) {
        const Mat nw = requantQ4k(*w);
        if (nw.ptr == w->ptr) continue;
        Mat tuned = nw;
        for (const Layer& L : layers)
            for (const Mat& lw : layerMats(L).slice())
                if (lw.ty == GgmlType::q4_k && lw.nrows == nw.nrows && lw.ncols == nw.ncols) tuned.tune = lw.tune;
        const DevPtr old = w->ptr;
        *w = tuned;
        hip::sync();
        freeAlloc(old);
    }
    hip::sync();
}

void Model::buildDraftHeadEx(DraftHeadKind kind) {
    const Mat w = output;
    if (w.ty != GgmlType::q6_k || w.ncols % 256 != 0 || draft_head || draft_d2) return;
    if (kind == DraftHeadKind::d2 && k.requant_q6k_d2 != nullptr && k.gemv_d2[0] != nullptr) {
        const DevPtr p = alloc(static_cast<u64>(w.ncols / 4 + w.ncols / 16) * w.nrows);
        hip::launch(k.requant_q6k_d2, {w.nrows, w.ncols / 256, 1}, {256, 1, 1}, 0, stream, w.ptr, w.row_bytes, p, I(w.ncols));
        hip::sync();
        draft_d2 = p;
        return;
    }
    draft_head = requantQ4k(w);
    hip::sync();
}

bool Model::setDraftVocab(std::span<const u32> vocab_ids) {
    if (!draft_d2 || k.copy_rows_map == nullptr || vocab_ids.empty() || vocab_ids.size() >= output.nrows) return false;
    const u32 n = static_cast<u32>(vocab_ids.size());
    const u64 rb = static_cast<u64>(output.ncols / 4 + output.ncols / 16);  // 2-bit codes + f16 scales per row
    const std::vector<i32> m(vocab_ids.begin(), vocab_ids.end());
    const DevPtr map = alloc(static_cast<u64>(n) * 4);
    hip::upload(map, m.data(), m.size() * 4);
    const DevPtr p = alloc(rb * n);
    hip::launch(k.copy_rows_map, {n, 1, 1}, {256, 1, 1}, 0, stream, *draft_d2, rb, p, map);
    hip::sync();
    freeAlloc(*draft_d2);
    draft_d2 = p;
    draft_rows = n;
    draft_map = map;
    return true;
}

MatList Model::layerMats(const Layer& L) const {
    MatList out;
    if (L.kind == LayerKind::attn) {
        out.add(L.attn.q);
        out.add(L.attn.k);
        out.add(L.attn.v);
        out.add(L.attn.o);
    } else {
        out.add(L.gdn.qkv);
        out.add(L.gdn.gate);
        out.add(L.gdn.beta);
        out.add(L.gdn.alpha);
        out.add(L.gdn.out);
    }
    out.add(L.ffn_gate);
    out.add(L.ffn_up);
    out.add(L.ffn_down);
    return out;
}

std::vector<Mat*> Model::tuneMats(Layer& L) {
    std::vector<Mat*> out;
    if (L.kind == LayerKind::attn) {
        out = {&L.attn.q, &L.attn.k, &L.attn.v, &L.attn.o};
    } else {
        out = {&L.gdn.qkv, &L.gdn.gate, &L.gdn.beta, &L.gdn.alpha, &L.gdn.out};
    }
    out.push_back(&L.ffn_gate);
    out.push_back(&L.ffn_up);
    out.push_back(&L.ffn_down);
    return out;
}

// ---------------------------------------------------------------------------
// profiling

Profile::Profile() : events(cap, nullptr), classes(cap, OpClass::misc) {}
Profile::~Profile() {
    for (hip::Event e : events)
        if (e) hip::eventDestroy(e);
}
void Profile::record(hip::Stream s, OpClass cls) {
    if (count == cap) flush();
    if (events[count] == nullptr) events[count] = hip::eventCreate();
    hip::eventRecord(events[count], s);
    classes[count] = cls;
    count += 1;
}
void Profile::flush() {
    if (count == 0) return;
    hip::sync();
    for (std::size_t i = 1; i < count; ++i) ms[static_cast<std::size_t>(classes[i])] += hip::eventElapsedMs(events[i - 1], events[i]);
    std::swap(events[count - 1], events[0]);
    classes[0] = classes[count - 1];
    count = 1;
}
void Profile::reset() {
    ms.fill(0);
    matmul_bytes = 0;
    count = 0;
}

}  // namespace whirl::qwen35
