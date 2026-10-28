// qwen35 weight binding (arch025-M1, R-8a), see weight_loader.h. Moved from src/model/loader.cpp (Reader, struct
// Loader, the weight part of Model::load and Model::loadDrafter); the bodies are unchanged apart from the device calls
// going through WeightLoader::Device and the counters through the Weights / LoadStats references.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (Loader).

#include "model/qwen35/weight_loader.h"

#include "platform/fs.h"
#include "whirl/common.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }

}  // namespace

Reader::Reader(const std::string& path) {
    f_ = platform::fs::openStdio(widen(path).c_str(), L"rb");  // arch025 PL2 (re-pointed from src/model/loader.cpp at the int3 merge)
    if (!f_) throw ModelError("OpenFailed", path);
    setvbuf(f_, nullptr, _IONBF, 0);
}

Reader::~Reader() {
    if (f_) std::fclose(f_);
}

void Reader::read(void* dst, std::size_t n, u64 off) {
    if (_fseeki64(f_, static_cast<long long>(off), SEEK_SET) != 0) throw ModelError("Truncated", "seek");
    if (std::fread(dst, 1, n, f_) != n) throw ModelError("Truncated", "short read");
}

WeightLoader::WeightLoader(const gguf::File& f, const Config& cfg, const kernels::KernelTable& k, Weights& w, LoadStats& stats, Device dev,
                           std::uint8_t* staging, std::size_t staging_len)
    : f_(f), cfg_(cfg), k_(k), w_(w), stats_(stats), dev_(std::move(dev)), file_(f.path()), staging_(staging), staging_len_(staging_len) {}

DevPtr WeightLoader::upload(const gguf::TensorInfo& t) {
    const u64 nbytes = t.nbytes();
    if (nbytes == 0) throw ModelError("UnsupportedTensorType", std::string(t.name));
    const DevPtr dst = dev_.alloc(nbytes);
    uploadTo(t, dst);
    return dst;
}

void WeightLoader::uploadTo(const gguf::TensorInfo& t, DevPtr dst) {
    const u64 nbytes = t.nbytes();
    u64 off = 0;
    const u64 base = f_.absOffset(t);
    while (off < nbytes) {
        const std::size_t n = static_cast<std::size_t>(std::min<u64>(nbytes - off, staging_len_));
        file_.read(staging_, n, base + off);
        dev_.upload(dst + off, staging_, n);
        off += n;
    }
    stats_.bytes += nbytes;
    stats_.tensors += 1;
    w_.type_bytes[ti(t.type) % n_types] += nbytes;
}

Mat WeightLoader::matHost(const char* name, device::PinnedBuf* keep) {
    const gguf::TensorInfo* t = f_.tensor(name);
    if (!t) throw ModelError("MissingTensor", name);
    const u64 nbytes = t->nbytes();
    *keep = device::PinnedBuf::alloc(*dev_.pinned, nbytes);
    void* buf = keep->get();
    file_.read(buf, static_cast<std::size_t>(nbytes), f_.absOffset(*t));
    stats_.bytes += nbytes;
    stats_.tensors += 1;
    Mat r;
    r.ptr = reinterpret_cast<std::uintptr_t>(buf);
    r.ty = t->type;
    r.ncols = static_cast<u32>(t->ne[0]);
    r.nrows = static_cast<u32>(t->rows());
    r.row_bytes = t->rowBytes();
    return r;
}

DevPtr WeightLoader::uploadMx(const gguf::TensorInfo& t, DevPtr* ref_out) {
    const u64 ncols = t.ne[0];
    if (ncols % 256 != 0) throw ModelError("UnsupportedTensorType", std::string(t.name) + ": MXFP4 row not a multiple of 256");
    if (ncols > 0x7FFFFFFF) throw ModelError("UnsupportedTensorType", std::string(t.name) + ": MXFP4 row too long");
    const u64 rb = t.rowBytes();
    const u64 nrows = t.rows();
    const u64 nbytes = rb * nrows;
    const DevPtr dst = dev_.alloc(nbytes);
    std::vector<std::uint8_t> refs(static_cast<std::size_t>(nrows));
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(rb));
    const u64 rows_per = std::max<u64>(1, staging_len_ / rb);
    const u64 base = f_.absOffset(t);
    u64 lossy = 0;
    for (u64 r0 = 0; r0 < nrows;) {
        const u64 nr = std::min(rows_per, nrows - r0);
        const std::size_t n = static_cast<std::size_t>(nr * rb);
        file_.read(staging_, n, base + r0 * rb);
        for (u64 ri = 0; ri < nr; ++ri) {
            std::uint8_t* row = staging_ + ri * rb;
            const std::uint8_t emax = kernels::repackMxfp4Row(row, row, static_cast<int>(ncols), &lossy, tmp.data());
            refs[static_cast<std::size_t>(r0 + ri)] = emax;
        }
        dev_.upload(dst + r0 * rb, staging_, n);
        r0 += nr;
    }
    const DevPtr rdev = dev_.alloc(nrows);
    dev_.upload(rdev, refs.data(), refs.size());
    *ref_out = rdev;
    w_.mx_fold_lossy += lossy;
    stats_.bytes += nbytes;
    stats_.tensors += 1;
    w_.type_bytes[ti(t.type) % n_types] += nbytes;
    return dst;
}

Mat WeightLoader::mat(const std::string& name) {
    const gguf::TensorInfo* t = f_.tensor(name);
    if (!t) throw ModelError("MissingTensor", name);
    if (k_.gemv1[ti(t->type) % n_types] == nullptr)
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

std::pair<Mat, Mat> WeightLoader::matPair(const std::string& na, const std::string& nb, bool* contig) {
    *contig = false;
    const gguf::TensorInfo* ta = f_.tensor(na);
    const gguf::TensorInfo* tb = f_.tensor(nb);
    if (!ta || !tb || ta->type != tb->type || ta->type == GgmlType::mxfp4 || ta->ne[0] != tb->ne[0] ||
        ta->rowBytes() != tb->rowBytes() || ta->nbytes() == 0 || tb->nbytes() == 0 || ta->nbytes() % 256 != 0 ||
        ta->nbytes() != ta->rowBytes() * ta->rows() || tb->nbytes() != tb->rowBytes() * tb->rows() ||
        k_.gemv1[ti(ta->type) % n_types] == nullptr)
        return {mat(na), mat(nb)};
    const DevPtr base = dev_.alloc(ta->nbytes() + tb->nbytes());
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

DevPtr WeightLoader::vec(const std::string& name) {
    const gguf::TensorInfo* t = f_.tensor(name);
    if (!t) throw ModelError("MissingTensor", name);
    if (t->type != GgmlType::f32) throw ModelError("UnsupportedTensorType", name + ": expected f32, got " + gguf::typeName(t->type));
    return upload(*t);
}

MoeW WeightLoader::moe(u32 il) {
    const Config& cfg = cfg_;
    const gguf::TensorInfo* rt = f_.tensor(bn(il, "ffn_gate_inp.weight"));
    if (!rt) throw ModelError("MissingTensor", bn(il, "ffn_gate_inp.weight"));
    if (rt->type != GgmlType::f32 || rt->ne[0] != cfg.n_embd || rt->ne[1] != cfg.n_expert)
        throw ModelError("UnsupportedTensorType", bn(il, "ffn_gate_inp.weight"));
    MoeW w;
    const DevPtr router_ptr = vec(bn(il, "ffn_gate_inp.weight"));
    w.sh_gate = f_.tensor(bn(il, "ffn_gate_inp_shexp.weight")) ? vec(bn(il, "ffn_gate_inp_shexp.weight")) : 0;
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
        if (k_.moe_gu[ti(e->ty) % n_types] == nullptr)
            throw ModelError("UnsupportedTensorType", "blk." + std::to_string(il) + " ffn_" + fld + "_exps: no MoE kernel for type " +
                                                          gguf::typeName(e->ty));
    }
    if (w.gate.nrows != cfg.n_ff_exp * cfg.n_expert || w.up.nrows != w.gate.nrows || w.gate.ncols != cfg.n_embd ||
        w.down.ncols != cfg.n_ff_exp || w.down.nrows != cfg.n_embd * cfg.n_expert || cfg.n_expert > 1024 ||
        cfg.n_expert_used > 64 || cfg.n_ff_exp % 256 != 0 || cfg.n_embd % 256 != 0)
        throw ModelError("UnsupportedMoeShape", "blk." + std::to_string(il));
    return w;
}

void WeightLoader::embeddings(bool embd_on_host) {
    const gguf::TensorInfo* emb = f_.tensor("token_embd.weight");
    if (!emb) throw ModelError("MissingTensor", "token_embd.weight");
    // (MXFP4 is repacked at load, so it cannot be read straight into host memory)
    if (embd_on_host && f_.tensor("output.weight") != nullptr && emb->type != GgmlType::mxfp4) {
        w_.tok_embd = matHost("token_embd.weight", &w_.embd_host);
    } else {
        w_.tok_embd = mat("token_embd.weight");
    }
    w_.output = f_.tensor("output.weight") ? mat("output.weight") : w_.tok_embd;
    w_.output_norm = vec("output_norm.weight");
}

void WeightLoader::layer(u32 il) {
    const Config& cfg = cfg_;
    Layer& L = w_.layers[il];
    auto n = [&](const char* s) { return bn(il, s); };
    L.attn_norm = vec(n("attn_norm.weight"));
    L.post_norm = vec(n("post_attention_norm.weight"));
    L.moe.reset();
    if (cfg.moe) {
        L.ffn_gate = mat(n("ffn_gate_shexp.weight"));
        L.ffn_up = mat(n("ffn_up_shexp.weight"));
        L.ffn_down = mat(n("ffn_down_shexp.weight"));
        L.moe = moe(il);
    } else {
        L.ffn_gate = mat(n("ffn_gate.weight"));
        L.ffn_up = mat(n("ffn_up.weight"));
        L.ffn_down = mat(n("ffn_down.weight"));
    }
    if (cfg.isAttn(il)) {
        L.kind = LayerKind::attn;
        L.attn.q = mat(n("attn_q.weight"));
        L.attn.k = mat(n("attn_k.weight"));
        L.attn.v = mat(n("attn_v.weight"));
        L.attn.o = mat(n("attn_output.weight"));
        L.attn.q_norm = vec(n("attn_q_norm.weight"));
        L.attn.k_norm = vec(n("attn_k_norm.weight"));
    } else {
        L.kind = LayerKind::gdn;
        L.gdn.qkv = mat(n("attn_qkv.weight"));
        L.gdn.gate = mat(n("attn_gate.weight"));
        bool ba_contig = false;
        std::tie(L.gdn.beta, L.gdn.alpha) = matPair(n("ssm_beta.weight"), n("ssm_alpha.weight"), &ba_contig);
        if (ba_contig && rowsContiguous(L.gdn.beta, L.gdn.alpha)) stats_.gdn_ba_contig += 1;
        L.gdn.out = mat(n("ssm_out.weight"));
        L.gdn.conv = vec(n("ssm_conv1d.weight"));
        L.gdn.dt = vec(n("ssm_dt.bias"));
        L.gdn.a = vec(n("ssm_a"));
        L.gdn.norm = vec(n("ssm_norm.weight"));
    }
}

std::optional<MtpW> WeightLoader::mtpBlock() {
    const Config& cfg = cfg_;
    const u32 il = cfg.n_layer;
    auto n = [&](const char* s) { return bn(il, s); };
    MtpW w;
    w.attn.q = mat(n("attn_q.weight"));
    w.attn.k = mat(n("attn_k.weight"));
    w.attn.v = mat(n("attn_v.weight"));
    w.attn.o = mat(n("attn_output.weight"));
    w.attn.q_norm = vec(n("attn_q_norm.weight"));
    w.attn.k_norm = vec(n("attn_k_norm.weight"));
    w.attn_norm = vec(n("attn_norm.weight"));
    w.post_norm = vec(n("post_attention_norm.weight"));
    w.ffn_gate = mat(n(cfg.moe ? "ffn_gate_shexp.weight" : "ffn_gate.weight"));
    w.ffn_up = mat(n(cfg.moe ? "ffn_up_shexp.weight" : "ffn_up.weight"));
    w.ffn_down = mat(n(cfg.moe ? "ffn_down_shexp.weight" : "ffn_down.weight"));
    if (cfg.moe) w.moe = moe(il);
    w.eh_proj = mat(n("nextn.eh_proj.weight"));
    w.enorm = vec(n("nextn.enorm.weight"));
    w.hnorm = vec(n("nextn.hnorm.weight"));
    w.head_norm = f_.tensor(n("nextn.shared_head_norm.weight")) ? vec(n("nextn.shared_head_norm.weight")) : w_.output_norm;
    return w;
}

std::unique_ptr<DFlashW> WeightLoader::dflash(const DFlashConfig& dc) {
    auto d = std::make_unique<DFlashW>();
    d->cfg = dc;
    d->fc = mat("fc.weight");
    d->enc_norm = vec("enc.output_norm.weight");
    d->out_norm = vec("output_norm.weight");
    d->sel_prev = mat("selector_predecessor.weight");
    d->sel_next = mat("selector_successor.weight");
    d->sel_hidden = mat("selector_hidden.weight");
    d->layers.resize(dc.n_layer);
    for (u32 il = 0; il < dc.n_layer; ++il) {
        DFlashLayerW& w = d->layers[il];
        auto n = [&](const char* s) { return bn(il, s); };
        w.attn_norm = vec(n("attn_norm.weight"));
        w.ffn_norm = vec(n("ffn_norm.weight"));
        w.q_norm = vec(n("attn_q_norm.weight"));
        w.k_norm = vec(n("attn_k_norm.weight"));
        w.q = mat(n("attn_q.weight"));
        w.k = mat(n("attn_k.weight"));
        w.v = mat(n("attn_v.weight"));
        w.o = mat(n("attn_output.weight"));
        w.gate = mat(n("ffn_gate.weight"));
        w.up = mat(n("ffn_up.weight"));
        w.down = mat(n("ffn_down.weight"));
        w.attn_conv_proj = mat(n("attn_conv_proj.weight"));
        w.ffn_conv_proj = mat(n("ffn_conv_proj.weight"));
        w.attn_conv_base = vec(n("attn_conv_base"));
        w.ffn_conv_base = vec(n("ffn_conv_base"));
    }
    return d;
}

}  // namespace whirl::qwen35
