// qwen35 weight binding (arch025-M1, R-8a): reads GGUF tensors into device memory and binds them to the fields of
// qwen35::Weights / DFlashW, with the type / kernel / shape checks and their error messages. Moved from
// src/model/loader.cpp (Reader, struct Loader and the weight part of Model::load / Model::loadDrafter).
// SPDX-License-Identifier: Apache-2.0
//
// The device side is injected (Device): Model::load passes its allocation ledger (Model::alloc), hip::upload and
// device::hipBackend() for pinned host memory, so the allocation calls keep their order and sizes (invariant I2);
// tests/model passes host memory and a NullBackend, so the whole binding runs on the CPU.

#pragma once

#include "whirl/model.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace whirl::qwen35 {

// Positional reads of the model file (plain buffered reads beat page-faulting
// a 16 GB mapping into pinned staging).
class Reader {
public:
    explicit Reader(const std::string& path);
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    void read(void* dst, std::size_t n, std::uint64_t off);

private:
    std::FILE* f_ = nullptr;
};

class WeightLoader {
public:
    struct Device {
        std::function<DevPtr(std::uint64_t)> alloc;                      // one device allocation of exactly these bytes
        std::function<void(DevPtr, const void*, std::size_t)> upload;    // host -> device copy
        device::Backend* pinned = nullptr;                                // owner of matHost's pinned buffers
    };

    // Opens f.path() (ModelError("OpenFailed")). staging: a pinned host buffer of staging_len bytes the reads go
    // through. Counts into w.type_bytes / w.mx_fold_lossy and stats (bytes, tensors, gdn_ba_contig).
    WeightLoader(const gguf::File& f, const Config& cfg, const Kernels& k, Weights& w, LoadStats& stats, Device dev,
                 std::uint8_t* staging, std::size_t staging_len);

    static std::string bn(std::uint32_t l, const char* suffix) { return "blk." + std::to_string(l) + "." + suffix; }

    // ---- one tensor
    DevPtr upload(const gguf::TensorInfo& t);
    // Reads tensor t into the existing device buffer dst (t.nbytes() bytes).
    void uploadTo(const gguf::TensorInfo& t, DevPtr dst);
    // Like mat(), but the data stays in pinned host memory (device-accessible).
    Mat matHost(const char* name, device::PinnedBuf* keep);
    // MXFP4: file blocks (e, qs[16]) x 32 values -> per 256 values e[8] then
    // qs[8][16] (same byte count), plus each row's reference exponent (its
    // largest block exponent) for the fp8 prefill GEMM.
    DevPtr uploadMx(const gguf::TensorInfo& t, DevPtr* ref_out);
    Mat mat(const std::string& name);
    // Two matrices with the same row layout in one allocation: a's rows, then
    // b's (a.ptr / b.ptr point at the two parts; bytes are unchanged, so a GEMV
    // on either reads exactly what mat() would give). Falls back to two mat()
    // loads unless the types / ncols / row_bytes match, the type is not MXFP4
    // (repacked + per-row refs) and b's start stays 256-byte aligned.
    std::pair<Mat, Mat> matPair(const std::string& na, const std::string& nb, bool* contig);
    DevPtr vec(const std::string& name);
    // Routed experts of block il (qwen35moe).
    MoeW moe(std::uint32_t il);

    // ---- the bindings of Model::load, in its upload order
    // token_embd (pinned host memory when embd_on_host and allowed), output (else tied to token_embd), output_norm
    void embeddings(bool embd_on_host);
    // trunk layer il's weights into w.layers[il] (the caller allocates a DeltaNet layer's state afterwards)
    void layer(std::uint32_t il);
    // the nextn block (block n_layer); the caller decides whether the model has one
    std::optional<MtpW> mtpBlock();
    // ---- Model::loadDrafter: the DFlash2 drafter's tensors (cfg = the drafter's config, validated)
    std::unique_ptr<DFlashW> dflash(const DFlashConfig& dc);

private:
    const gguf::File& f_;
    const Config& cfg_;
    const Kernels& k_;
    Weights& w_;
    LoadStats& stats_;
    Device dev_;
    Reader file_;
    std::uint8_t* staging_;
    std::size_t staging_len_;
};

}  // namespace whirl::qwen35
