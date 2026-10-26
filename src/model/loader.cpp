// Model::load (weights through WeightLoader, src/model/qwen35/weights.cpp since arch025-M1) and buffer / KV-pool /
// sequence management.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Model.load, Loader, allocKvPool,
// setupSeqs, snapshots, requantMtpQ4, buildDraftHeadEx).

#include "config/env_snapshot.h"
#include "whirl/dump.h"
#include "whirl/model.h"
#include "whirl/kv_layout.h"
#include "model/qwen35/weight_loader.h"
#include "device/backend.h"
#include "whirl/vram_limit.h"
#include "whirl/uma_mem.h"

#include "whirl/common.h"

#include <algorithm>
#include <format>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

inline i32 I(u64 v) { return static_cast<i32>(v); }

}  // namespace

// ---------------------------------------------------------------------------

Model::~Model() {
    // the release order of before arch024 S11 (graph, rpos, allocations in order, pinned embedding, stream);
    // the members would otherwise be destroyed in reverse declaration order
    decode_graph.reset();
    rpos_own.reset();
    for (device::DevBuf& b : allocations) b.reset();
    allocations.clear();
    embd_host.reset();
    stream_own.reset();
}

DevPtr Model::alloc(u64 bytes) {
    device::DevBuf b = device::DevBuf::alloc(device::hipBackend(), bytes);
    const DevPtr p = b.get();
    allocations.push_back(std::move(b));
    alloc_total += bytes;
    return p;
}

DevPtr Model::allocZero(u64 bytes) {
    const DevPtr p = alloc(bytes);
    hip::memset(p, 0, static_cast<std::size_t>(bytes));
    return p;
}

void Model::freeAlloc(DevPtr p) {
    for (std::size_t i = 0; i < allocations.size(); ++i)
        if (allocations[i].get() == p) {
            device::DevBuf gone = std::move(allocations[i]);
            if (i + 1 != allocations.size()) allocations[i] = std::move(allocations.back());
            allocations.pop_back();
            alloc_total -= gone.bytes();  // WDDM-LOG review P3-8: alloc_total stays the live sum
            gone.reset();
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

std::string readWhole(const std::string& path) { return readFile(path); }

}  // namespace

namespace {
// arch025-N1: the reads of kvRequestFromEnv / resolveFdSplits on a snapshot (Model::load passes the load's)
// WHIRL_KV: nullopt when unset or "auto" (the mode decides), else the forced format
std::optional<KvFormat> kvRequestOf(const whirl::config::EnvSnapshot& env) {
    const auto& v = env.kv;
    if (!v || *v == "auto") return std::nullopt;
    if (const std::optional<KvFormat> k = kvFormatFromName(*v)) return k;
    throw std::invalid_argument("WHIRL_KV must be auto, f16, q8, q8h, q8v or q4 (got '" + *v + "')");
}

u32 fdSplitsOf(KvFormat kv, const whirl::config::EnvSnapshot& env) {
    u32 splits = kvIs4bit(kv) ? 256 : 64;
    if (const auto& v = env.fd_splits) {
        const unsigned long s = std::strtoul(v->c_str(), nullptr, 10);
        if (s >= 64 && s <= 256 && s % 64 == 0) splits = static_cast<u32>(s);
    }
    return splits;
}

}  // namespace

std::optional<KvFormat> kvRequestFromEnv() { return kvRequestOf(whirl::config::EnvSnapshot::capture()); }

u32 resolveFdSplits(KvFormat kv) { return fdSplitsOf(kv, whirl::config::EnvSnapshot::capture()); }

u64 bufferBytes(const Config& cfg, u32 Bq, u64 ffs, u64 max_elems, u32 fd_splits) {
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
    const u64 n_split = fd_splits;
    n += max_verify_rows * n_split * cfg.n_head * 2 * f4;  // part_ml
    n += max_verify_rows * n_split * cfg.n_head * cfg.head_dim * f4;  // part_acc
    n += B * cfg.n_head * 2 * f4;                         // dq_st_ml (q8 / q4 / KG-3 prefill attention softmax state)
    n += (static_cast<u64>(Model::kDqMaxKeys) / 256 + 1) * 4;  // dq_ptab (identity pages for up to kDqMaxKeys keys)
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
        n += max_verify_rows * Kx * F * f4 * 2;           // moe_g, moe_u
        n += max_verify_rows * Kx * F + 256;              // moe_xq
        n += (max_verify_rows * Kx * F / 32 + 8) * 4;     // moe_xd
        n += B * E * f4;                                  // moe_ysh
        n += B * Kx * 4 * 2;                              // moe_perm, moe_inv
        n += (B * Kx / 32 + cfg.n_expert + 1) * 16;       // moe_tiles
        n += 16;                                          // moe_ntiles
        n += (B * Kx / 64 + cfg.n_expert + 1) * 16;       // moe_tiles64
        n += 16;                                          // moe_ntiles64
        n += B * Kx * std::max(E, F) * 2;                 // moe_x16
        n += B * Kx * F * f4 * 2;                         // moe_yg, moe_yu
        n += B * Kx * E * f4;                             // moe_yd
        n += B * Kx * f4;                                 // moe_sx
    }
    return n;
}

u64 kvBytesPerTokenCfg(const Config& cfg, bool has_mtp, KvFormat kv) {
    u64 layers = has_mtp ? 1 : 0;
    for (u32 i = 0; i < cfg.n_layer; ++i)
        if (cfg.isAttn(i)) layers += 1;
    const u64 e = static_cast<u64>(cfg.n_head_kv) * cfg.head_dim;
    return layers * kv::layerBytes(e, kv);  // whirl/kv_layout.h (arch024-S11c F-6)
}

LoadEstimate estimateLoad(const gguf::File& f, const Config& cfg, bool embd_on_host, bool no_mtp) {
    LoadEstimate r;
    const gguf::TensorInfo* emb = f.tensor("token_embd.weight");
    r.embd_on_host = embd_on_host && emb != nullptr && f.tensor("output.weight") != nullptr && emb->type != GgmlType::mxfp4;
    r.has_mtp = !no_mtp && cfg.n_nextn > 0 && f.tensor(WeightLoader::bn(cfg.n_layer, "nextn.eh_proj.weight")) != nullptr;
    const std::string nextn_prefix = "blk." + std::to_string(cfg.n_layer) + ".";  // the MTP block (skipped with no_mtp)
    static constexpr std::string_view mats[] = {"attn_q.weight",  "attn_k.weight",    "attn_v.weight",   "attn_output.weight",
                                                "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
                                                "ssm_out.weight", "ffn_gate.weight",  "ffn_up.weight",   "ffn_down.weight",
                                                "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    r.ffs = cfg.n_ff;
    for (const gguf::TensorInfo& t : f.tensors()) {
        if (r.embd_on_host && t.name == "token_embd.weight") continue;
        if (no_mtp && cfg.n_nextn > 0 && t.name.starts_with(nextn_prefix)) continue;
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
    r.kv_f16 = kvBytesPerTokenCfg(cfg, r.has_mtp, KvFormat::f16);
    r.kv_q8v = kvBytesPerTokenCfg(cfg, r.has_mtp, KvFormat::q8v);
    r.kv_q8h = kvBytesPerTokenCfg(cfg, r.has_mtp, KvFormat::q8h);
    r.kv_q4 = kvBytesPerTokenCfg(cfg, r.has_mtp, KvFormat::q4);
    return r;
}

u64 dflashWeightBytes(const gguf::File& draft) {
    u64 b = 0;
    for (const gguf::TensorInfo& t : draft.tensors()) b += t.nbytes() + (t.type == GgmlType::mxfp4 ? t.rows() : 0);
    return b;
}

std::unique_ptr<Model> Model::load(const gguf::File& f, u32 max_ctx_req, LoadStats& stats, const LoadOptions& opt) {
    auto mp = std::make_unique<Model>();
    Model& m = *mp;
    // arch025-N1: this load's WHIRL_* environment, the caller's (LoadOptions::env) or read here, before the first
    // read below; kept on the Model for loadOrTune / applyGemvEnv / the selftest checks
    m.env_snap = opt.env ? opt.env : std::make_shared<const whirl::config::EnvSnapshot>(whirl::config::EnvSnapshot::capture());
    const whirl::config::EnvSnapshot& env = *m.env_snap;
    // arch025-F7: the forward's research switches, once per load (were read by forward at every call / first call)
    {
        const whirl::config::ForwardEnv fe = whirl::config::forwardEnv(env);
        m.head_gemvx = fe.head_gemvx;
        m.prec_gemvh = fe.prec_gemvh;
        m.prec_gs = fe.prec_gs;
    }
    m.cfg = Config::fromGguf(f);
    const Config& cfg = m.cfg;
    // every tensor's type and shape against cfg, before any GPU allocation
    cfg.validateTensors(f);
    // integrated GPU (Radeon 8060S): the weights, buffers and a minimal KV cache must fit the BIOS
    // carve-out, checked before the first device-memory call (a tiny carve-out makes the runtime fail
    // with hipError 719 in hipModuleLoadData; UMA-1). Discrete GPUs: umaMemory reads nothing else.
    if (hip::umaCheckOn()) {
        const hip::UmaMem u = hip::umaMemory(hip::getDevice());
        if (u.integrated && u.known) {
            const LoadEstimate est = estimateLoad(f, cfg, opt.embd_on_host, opt.no_mtp);
            // the KV format this load will choose below (numerics::chooseKv, same inputs; the 8060S code
            // object has every KV kernel), not f16: a balance / fast load with an explicit --ctx was refused
            // although its q8 / q4 cache fits (REVIEW-022 F6)
            const std::optional<KvFormat> kvm_est = opt.kv_req ? opt.kv_req : kvRequestOf(env);
            // (WHIRL_DUMP_KV: one "D kv: site=loader.uma_est" line, arch025-S0a)
            const KvFormat kv_est = numerics::kvEstimate(numerics::KvEstSite::loader_uma, opt.numerics, cfg.moe, kvm_est);
            hip::UmaNeed n;
            n.weights = est.weights;
            n.fixed = est.weights + est.state +
                      bufferBytes(cfg, std::clamp<u32>(opt.max_batch, 1, max_batch_limit), est.ffs, est.max_elems, fdSplitsOf(KvFormat::q4, env)) +
                      (256ull << 20);
            n.kv_per_token = est.kvPerToken(kv_est);
            n.min_tokens = std::max<u32>(max_ctx_req, 4096);
            n.want_tokens = n.min_tokens;
            if (const std::string why = hip::umaRefusal(u, n, hip::describeDevice(hip::getDevice()).name); !why.empty())
                throw ModelError("UmaTooSmall", why);
        }
    }
    std::string co_path = opt.code_object;
    if (co_path.empty()) {
        if (const auto& v = env.code_object) co_path = *v;
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
    m.stream_own = device::StreamH::adopt(device::hipBackend(), hip::streamCreate());  // blocking, as before
    m.stream = m.stream_own.get();
    m.max_ctx = max_ctx_req;
    m.max_batch = std::clamp<u32>(opt.max_batch, 1, max_batch_limit);
    const std::optional<KvFormat> kvm = opt.kv_req ? opt.kv_req : kvRequestOf(env);
    m.kv_req = kvm;
    {
        // KV format: fixed per mode x model type (numerics::chooseKv, shared by CLI and server;
        // WHIRL_KV is a debug override). Decided once, here, before the kernel table is loaded,
        // and never switched afterwards (no fallback to a lower-precision format when it does
        // not fit: the context shrinks or the load stops, below / in the server).
        const kernels::Caps caps = kernels::Caps::probe(m.module);
        const numerics::KvCaps kv_caps{caps.kv_q8h, caps.kv_q8v, caps.kv_q4, device::profile(m.arch).numerics_8060s};
        m.kv_choice = dump::kv("loader.load", opt.numerics, cfg.moe, kv_caps, kvm,
                               numerics::chooseKv(opt.numerics, cfg.moe, kv_caps, kvm));  // (WHIRL_DUMP_KV)
        m.fixFormat(m.kv_choice.kv);  // the KV pool's format, fixed here once (arch025-M2: KvPool, read-only after)
        // an explicit KV format the code object has no kernels for (gfx1201: q4)
        if (!caps.supports(m.kv_fmt))
            throw ModelError("UnsupportedKvFormat", std::string(kvFormatName(m.kv_fmt)) +
                                                        " KV needs kernels this GPU does not have; use WHIRL_KV=f16, q8 or auto");
    }
    m.k = loadKernels(m.module, m.kv_fmt);
    // MXFP4 routed experts: process-level switches (as WHIRL_KV), read by CLI and server alike
    // per-model-type request (numerics::forModel; MoE balance = dense balance items since MODES-MXFP4)
    m.num_req = numerics::forModel(opt.numerics, cfg.moe);
    // (arch025-N2b: moe_fp8 / moe_mxw / moe_wide / moe_rbf are numerics::plan's, in loadOrTune: WHIRL_MOE_FP8 / _MXW /
    // _WIDE / _RBF from this load's snapshot; nothing reads them before loadOrTune)
    m.layers.resize(cfg.n_layer);
    m.kcache.assign(cfg.n_layer, 0);
    m.vcache.assign(cfg.n_layer, 0);
    m.kscale.assign(cfg.n_layer, 0);
    m.vscale.assign(cfg.n_layer, 0);
    m.state_tables.emplace_back(cfg.n_layer, 0);
    m.conv_state = std::span<DevPtr>(m.state_tables.back());
    m.state_tables.emplace_back(cfg.n_layer, 0);
    m.ssm_state = std::span<DevPtr>(m.state_tables.back());
    m.gemv_r = device::profile(m.arch).gemv_r;

    const std::size_t staging_len = 64u << 20;
    const device::PinnedBuf staging_buf = device::PinnedBuf::alloc(device::hipBackend(), staging_len);
    auto* staging = static_cast<std::uint8_t*>(staging_buf.get());
    // the weights' device side: this Model's allocation ledger (Model::alloc, freed in order by ~Model), hip::upload
    WeightLoader ld(f, cfg, m.k, m, stats,
                    WeightLoader::Device{[&m](u64 n) { return m.alloc(n); },
                                         [](DevPtr dst, const void* src, std::size_t n) { hip::upload(dst, src, n); },
                                         &device::hipBackend()},
                    staging, staging_len);

    // fit check before any weight is uploaded: weights + state + buffers against the
    // free VRAM (the simulated card's under WHIRL_VRAM_LIMIT_MB); a load that cannot fit
    // fails here with the sizes instead of an out-of-memory error part way through
    // (not on the UMA iGPU, gfx1151, unless a limit is set: its free VRAM does not bound what it can allocate)
    if (!device::profile(m.arch).uma_no_vram_fit_check || vram::limitBytes() != 0) {
        const LoadEstimate est = estimateLoad(f, cfg, opt.embd_on_host);
        const u64 buf = bufferBytes(cfg, m.max_batch, est.ffs, est.max_elems, fdSplitsOf(m.kv_fmt, env));
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
    ld.embeddings(opt.embd_on_host);

    for (u32 il = 0; il < cfg.n_layer; ++il) {
        ld.layer(il);
        // a DeltaNet layer's recurrent state, right after its weights (the allocation order of before arch025-M1)
        if (!cfg.isAttn(il)) {
            m.conv_state[il] = m.allocZero(static_cast<u64>(cfg.d_conv - 1) * cfg.convCh() * 4);
            m.ssm_state[il] = m.allocZero(static_cast<u64>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4);
        }
    }
    // --draft dflash: no MTP block, unless the drafter could not run (no dflash_* kernels)
    m.mtp_skipped = opt.no_mtp && m.dflashMissingKernel().empty();
    if (!m.mtp_skipped && cfg.n_nextn > 0 && f.tensor(WeightLoader::bn(cfg.n_layer, "nextn.eh_proj.weight")) != nullptr)
        m.mtp = ld.mtpBlock();
    const auto t1 = std::chrono::steady_clock::now();
    stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const u64 f4 = 4;
    const u64 B = m.max_batch;
    const u64 buffers_from = m.alloc_total;
    u64 max_elems = 0;
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
    m.fd_splits = fdSplitsOf(m.kv_fmt, env);
    const u64 n_split = m.fd_splits;
    m.part_ml = m.alloc(max_verify_rows * n_split * cfg.n_head * 2 * f4);
    m.part_acc = m.alloc(max_verify_rows * n_split * cfg.n_head * cfg.head_dim * f4);
    if (kvQuant(m.kv_fmt) && !kvKf16(m.kv_fmt) && (kvIs4bit(m.kv_fmt) ? m.k.kv_dq_rows_r != nullptr : m.k.kv_dq_rows != nullptr)) {
        // q8 / q8h / q4 prefill attention: f16 copies of a sequence's K / V in ffn_g / ffn_u (Model::prefillAttn)
        m.dq_rows = static_cast<u32>(std::min<u64>(B * ffs * f4 / (static_cast<u64>(cfg.n_head_kv) * cfg.head_dim * 2), 1u << 30)) / 256 * 256;
        if (m.k.kv_dq_rows_r != nullptr) m.dq_st_ml = m.alloc(B * cfg.n_head * 2 * f4);
        // identity pages for every key position (the key ranges of attn_kgs index it by sequence position)
        std::vector<std::int32_t> id(std::max<u64>(m.dq_rows, kDqMaxKeys) / 256 + 1);
        for (std::size_t i = 0; i < id.size(); ++i) id[i] = static_cast<std::int32_t>(i);
        m.dq_ptab = m.alloc(id.size() * 4);
        hip::upload(m.dq_ptab, id.data(), id.size() * 4);
    }
    // KG-3: attn_prefill_wmma_s softmax state (also f16 KV, for WHIRL_ATTN_SEG key ranges)
    if (m.dq_st_ml == 0 && m.k.attn_pws != nullptr) m.dq_st_ml = m.alloc(B * cfg.n_head * 2 * f4);
    {
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
        m.gc_w_bytes = n_chunks * heads * 64 * 128 * 4;
        m.gc_u_bytes = m.gc_w_bytes;
        m.gc_m_bytes = n_chunks * heads * 64 * 64 * 4;
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
    // (arch025-N2b: fp8_prefill is numerics::plan's, in loadOrTune: the item fp8, WHIRL_FP8 overrides)
    if (cfg.moe) {
        const u64 Kx = cfg.n_expert_used;
        const u64 F = cfg.n_ff_exp;
        const u64 E = cfg.n_embd;
        m.moe_logits = m.alloc(B * cfg.n_expert * f4);
        m.moe_ids = m.allocZero(B * Kx * 4);
        m.moe_w = m.allocZero(B * Kx * f4);
        m.moe_sg = m.allocZero(B * f4);
        // verify batches up to max_verify_rows rows run on the decode experts (precise, and the
        // balance / fast wide verify)
        m.moe_g = m.alloc(max_verify_rows * Kx * F * f4);
        m.moe_u = m.alloc(max_verify_rows * Kx * F * f4);
        m.moe_xq = m.alloc(max_verify_rows * Kx * F + 256);
        m.moe_xd = m.alloc((max_verify_rows * Kx * F / 32 + 8) * 4);
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
    // sizing self-check: the pre-load estimate (bufferBytes, the server's auto-shrink) must cover what was just
    // allocated; a shortfall means an allocation was added without updating the estimate
    if (const u64 used = m.alloc_total - buffers_from, est = bufferBytes(cfg, m.max_batch, ffs, max_elems, m.fd_splits); used > est)
        m.load_note += "warning: the VRAM estimate for the prefill buffers (" + std::to_string(est >> 20) + " MiB) is below what was allocated (" +
                       std::to_string(used >> 20) + " MiB); please report\n";
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

// (arch025-M2: kvBytesPerTokenFmt, kvName, kvLayers, allocKvPool, freeKvPool, mapPages, kvArgs, reset, setupSeqs,
// pendLayerBytes / pendSsmOff / pendPtr, selectSeq, restoreSeqSnapshot: src/model/qwen35/kv.cpp)

u64 Model::loadDrafter(const gguf::File& f, LoadStats& stats) {
    DFlashConfig dc = DFlashConfig::fromGguf(f);
    dc.validate(cfg, f);
    const std::size_t staging_len = 64u << 20;
    const device::PinnedBuf staging_buf = device::PinnedBuf::alloc(device::hipBackend(), staging_len);
    auto* staging = static_cast<std::uint8_t*>(staging_buf.get());
    const u64 bytes0 = stats.bytes;
    WeightLoader ld(f, cfg, k, *this, stats,
                    WeightLoader::Device{[this](u64 n) { return alloc(n); },
                                         [](DevPtr dst, const void* src, std::size_t n) { hip::upload(dst, src, n); },
                                         &device::hipBackend()},
                    staging, staging_len);
    auto d = ld.dflash(dc);
    hip::sync();
    d->bytes = stats.bytes - bytes0;
    dflash = std::move(d);
    return dflash->bytes;
}

void Model::enableCapture(u32 rows) {
    if (dflash) enableCaptureLayers(dflash->cfg.target_layers, rows);
}

void Model::enableCaptureLayers(std::span<const u32> layer_ids, u32 rows) {
    disableCapture();
    if (layer_ids.empty() || rows == 0) return;
    cap_slot.assign(cfg.n_layer + 1, -1);
    for (std::size_t j = 0; j < layer_ids.size(); ++j)
        if (layer_ids[j] <= cfg.n_layer) cap_slot[layer_ids[j]] = static_cast<i32>(j);
    cap_rows = std::min(rows, max_batch);
    cap_buf = allocZero(static_cast<u64>(layer_ids.size()) * cap_rows * cfg.n_embd * 4);
}

void Model::disableCapture() {
    if (cap_buf) freeAlloc(cap_buf);
    cap_buf = 0;
    cap_rows = 0;
    cap_n = 0;
    cap_slot.clear();
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

Profile::Profile() : events(cap), classes(cap, OpClass::misc) {}
Profile::~Profile() {
    for (device::EventH& e : events) e.reset();  // in index order, as before arch024 S11
}
void Profile::record(hip::Stream s, OpClass cls) {
    if (count == cap) flush();
    if (!events[count]) events[count] = device::EventH::adopt(device::hipBackend(), hip::eventCreate());  // timing event
    hip::eventRecord(events[count].get(), s);
    classes[count] = cls;
    count += 1;
}
void Profile::flush() {
    if (count == 0) return;
    hip::sync();
    for (std::size_t i = 1; i < count; ++i) ms[static_cast<std::size_t>(classes[i])] += hip::eventElapsedMs(events[i - 1].get(), events[i].get());
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
