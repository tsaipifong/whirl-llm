// whirl-kernel-test: shared harness (device buffers, comparisons, report).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kt_seed.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"

namespace kt {

namespace hip = whirl::hip;
namespace wk = whirl::kernels;
using wk::QType;
using hip::DevPtr;

// ---------------------------------------------------------------------------
// f16 helpers (IEEE binary16, round to nearest even)

float h2f(std::uint16_t h);
std::uint16_t f2h(float f);         // single rounding from f32
std::uint16_t d2h(double d);        // single rounding from an (exact) double
inline float roundh(float f) { return h2f(f2h(f)); }

// ---------------------------------------------------------------------------
// Device buffer (RAII)

class Buf {
public:
    Buf() = default;
    explicit Buf(std::size_t bytes) : bytes_(bytes) { p_ = hip::malloc(bytes > 0 ? bytes : 1); }
    template <class T>
    explicit Buf(const std::vector<T>& v) : Buf(v.size() * sizeof(T)) {
        if (!v.empty()) hip::upload(p_, v.data(), bytes_);
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p_(o.p_), bytes_(o.bytes_) { o.p_ = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = o.p_;
            bytes_ = o.bytes_;
            o.p_ = 0;
        }
        return *this;
    }
    ~Buf() { reset(); }
    DevPtr p() const { return p_; }
    operator DevPtr() const { return p_; }
    std::size_t bytes() const { return bytes_; }
    template <class T>
    void up(const std::vector<T>& v) { hip::upload(p_, v.data(), v.size() * sizeof(T)); }
    template <class T>
    std::vector<T> down(std::size_t n) const {
        std::vector<T> v(n);
        if (n) hip::download(v.data(), p_, n * sizeof(T));
        return v;
    }
    // hipMemset runs on the legacy null stream: wait for it before kernels on other streams.
    void zero() {
        hip::memset(p_, 0, bytes_);
        hip::sync();
    }
    void fill(int byte) {
        hip::memset(p_, byte, bytes_);
        hip::sync();
    }

private:
    void reset() {
        if (p_) hip::free(p_);
        p_ = 0;
    }
    DevPtr p_ = 0;
    std::size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Results

enum class Kind { exact, tol, invariant };

struct Result {
    std::string family, name;
    Kind kind = Kind::exact;
    bool pass = false;
    std::size_t n = 0, mismatches = 0;
    double max_err = 0, metric = 0;  // tol: worst |err| / bound (<= 1 passes)
    std::string note;
};

class Report {
public:
    void add(Result r);
    void skip(const std::string& family, const std::string& name, const std::string& why);
    int summary() const;  // prints, returns failure count
    std::size_t checks() const { return results_.size(); }
    struct Skip {
        std::string family, name, why;
        std::string key() const { return family + " " + name; }
    };
    const std::vector<Skip>& skips() const { return skips_; }
    bool verbose = false;
    std::string family;   // current family label

private:
    std::vector<Result> results_;
    std::vector<std::string> skipped_;
    std::vector<Skip> skips_;
};

// FIX-TEST-A (FR-TEST P3-4): a skip whose reason is a kernel absent from the code object (as opposed to
// a model / tensor that was not given): with --require-all such a skip fails unless the device's arch
// lists it as absent by design (requiredKernelAbsentOk).
bool skipIsMissingKernel(const std::string& why);
// arch: hip::DeviceInfo::gcn_arch prefix ("gfx1201", "gfx1151"); key: "family name" of the skip.
// nullopt: no list for this arch (--require-all then refuses to run: exit 2).
std::optional<bool> requiredKernelAbsentOk(const std::string& arch, const std::string& key);

// Exact comparison of raw bytes (float arrays compared bitwise).
Result cmpExactBits(const std::string& name, const void* got, const void* ref, std::size_t n_elems,
                    std::size_t elem_bytes, Kind kind = Kind::exact);
template <class T>
Result cmpExact(const std::string& name, const std::vector<T>& got, const std::vector<T>& ref,
                Kind kind = Kind::exact) {
    const std::size_t n = std::min(got.size(), ref.size());
    Result r = cmpExactBits(name, got.data(), ref.data(), n, sizeof(T), kind);
    if (got.size() != ref.size()) {
        r.pass = false;
        r.note += " (size mismatch)";
    }
    return r;
}
// Tolerance: |got - ref| <= atol + rtol * scale[i] (scale = sum |terms| or |ref|).
Result cmpTol(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref,
              const std::vector<double>& scale, double rtol, double atol);
Result cmpTolRel(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref,
                 double rtol, double atol);

// ---------------------------------------------------------------------------
// Test context

struct Models {
    std::string q4;   // Qwen3.8-27B UD-Q4_K_M (dense, many K-quant / IQ types)
    std::string mx;   // Qwen3.8-27B MXFP4
    std::string moe;  // Ornith-1.5-35B-A3B Q4_K_M (qwen35moe)
    std::string moemx;  // Ornith-1.5-35B-A3B MXFP4 (qwen35moe, MXFP4 experts)
};

struct Ctx {
    hip::Module mod;
    wk::KernelTable k;
    hip::Stream s = nullptr;
    Report rep;
    Models models;
    std::mt19937_64 rng{12345};  // v0252-ktrng: reseeded per family by beginFamily (kt_seed.h); nothing is shared across families
    void beginFamily(const std::string& name) { rng.seed(familySeed(name)); }
    bool quick = false;  // smaller shapes

    hip::Function fn(const std::string& name) const { return mod.getFunction(name.c_str()); }
    hip::Function fnOpt(const std::string& name) const { return mod.getFunctionOpt(name.c_str()); }
    void sync() const { hip::streamSync(s); }

    std::vector<float> randn(std::size_t n, float sigma = 1.0f);
    std::vector<float> randu(std::size_t n, float lo, float hi);
    std::vector<int> randi(std::size_t n, int lo, int hi);

    // GGUF files stay mapped for the whole run; nullptr if the path is empty
    // or the file cannot be opened.
    const whirl::gguf::File* open(const std::string& path);
    std::map<std::string, std::unique_ptr<whirl::gguf::File>> files;
};

struct HostMat;
// One real weight matrix per supported type from the configured models (rows
// [0, max_rows) of a representative tensor), plus synthetic f32 / f16 (and
// synthetic blocks for types no model provides).
std::vector<HostMat> sampleMats(Ctx& c, int max_rows);

// A weight matrix in kernel layout (MXFP4 already repacked), host copy.
struct HostMat {
    QType type = QType::f32;
    int ncols = 0, nrows = 0;
    std::uint64_t row_bytes = 0;
    std::vector<std::uint8_t> data;    // nrows * row_bytes
    std::vector<std::uint8_t> ref;     // MXFP4: per-row max block exponent (gemm8 wref)
    std::string name;
};

// Loads rows [r0, r0 + nrows) of a GGUF tensor (all rows if nrows <= 0) of
// a 2-D (or expert 3-D, flattened) tensor. MXFP4 rows are repacked.
HostMat loadMat(const whirl::gguf::File& f, const std::string& tensor, int r0 = 0, int nrows = -1);
// Synthetic matrix with random but valid blocks of the given type.
HostMat randomMat(Ctx& c, QType t, int nrows, int ncols);

// Test families (each in its own .cpp)
void testQuant(Ctx& c);
void testGemv(Ctx& c);
void testGemvk(Ctx& c);   // "gemvk": gfx1151 K-balanced MXFP4 multi-token GEMV bitwise == 1-token (feat026-ggemv)
void testGemm(Ctx& c);
void testAttn(Ctx& c);
void testAttnDq4(Ctx& c);  // "attndq4": attn_dq4 at 256 splits / contexts up to 256k / random draft windows
// v0252-c1acc (AO-2): "attnkgl" (on request): attn_kg / kgf / kgv / kgk <np> == attn_prefill_wmma[_q8] bit for bit at
// L 32k / 128k, Ornith 16 / 2 (NP 4) and Swift 24 / 4 (NP 6), f16 paged cache and Q8P rows (test_attn.cpp); v0252-m0a: Q4P rows
// (kv_dq_rows_r_q4) attn_kgk / attn_kg <np> == one-range attn_kgs <np> (fast prefill, candidate A)
void testAttnKgLong(Ctx& c);
void testGemvh(Ctx& c);   // "gemvh": bitwise sweep of precise-decode gemvh/gemvh2
void benchGemvh(Ctx& c);  // "gemvhbench": timing only
void testGemvhv(Ctx& c);  // "gemvhv": bitwise sweep of gfx1201 gemvh candidates from WHIRL_KT_GEMVH_HSACO (on request)
void benchAttn(Ctx& c);  // "attnbench": timing only, run with --only attnbench
void testAttnWs4(Ctx& c);   // "ws4": gfx1151 attn_ws4 bitwise == attn_wsplit1
void benchAttnWs4(Ctx& c);  // "ws4bench": timing only, on request
void testAttnWs4g(Ctx& c);   // "ws4g": gfx1151 attn_ws4g multi-row verify bitwise == single-row
void benchAttnWs4g(Ctx& c);  // "ws4gbench": timing only, on request
void testAttnWs4r(Ctx& c);   // "ws4r": gfx1201 attn_ws4r* bitwise == attn_wsplit1 / attn_wsplit2 (v0252-ws4r97)
void testGdn(Ctx& c);
void testVision(Ctx& c);  // "vision": encoder kernels (own code object) vs CPU
void testMoe(Ctx& c);
void testMoeMx(Ctx& c);
void testMoeF(Ctx& c);  // "moef": precise MoE decode experts moe_gu_f_* / moe_down_f_*
// v0252-c3port "moefuse" (test_moe_fused.cpp): gfx1201 fused precise MoE decode moe_gus*_f_mxfp4 == moe_gu_f + silu_mul and
// f32_to_f16 + gemvh x2 + silu_mul + f32_to_f16, gemvhf_<T> == f32_to_f16 + gemvh_<T> (bitwise); presence rows per arch
void testMoeFused(Ctx& c);
// v0251-a11port (called by testMoeMx, family "moemx"): fused router moe_router_fused_i* == moe_logits_f32 + moe_topk
// (bitwise) and vs CPU, presence rows (every code object + host table; v0252-c5port adds gfx1201) (test_moe_router.cpp)
void testMoeRouter(Ctx& c);
// gfx1201 precise MoE prefill gemmr_moe* == gemm_moe + moe_act_f16 (test_moe_gu.cpp), called by testMoeMx
void gemmrChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K);
// 8060S grouped expert GEMMs (test_moe_gu.cpp), called by testMoe / testMoeMx
void moeGuChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K);
// feat026-moep: the multi-fragment 8060S expert GEMM variants vs the shipped path (test_moep.cpp); v0251-a7port: the
// default kernels (kernels::kMoepDefault) must be in the gfx1151 object and the host table, absent elsewhere
void moepChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K);
void testMisc(Ctx& c);
void testDflash(Ctx& c);

inline unsigned cdiv(std::uint64_t a, std::uint64_t b) { return static_cast<unsigned>((a + b - 1) / b); }

// v0252-8sbal (PLAN V1-53): grouped-raster prototypes of the gfx1151 fused-dequant GEMM, gemmfq_<name>_mxfp4 =
// gemmfq_<base>_mxfp4 with the blocks walked in groups of <g> row tiles (Gufo prefill_quant_gemm.hpp:207-223). Same
// tile, same WMMA chain per output -> bitwise == the base slot (kt gemm checks it); the same ABI and grid as the base
// slot (x = token blocks, y = row blocks). Not in the product tables (kernels::kGemmFqNames): kt only, until the
// screen picks one. base = index into kernels::kGemmFqCfgs (0 c2, 1 t2).
// t2g8 / c2g8 are also the product's grouped-raster twins (kernels::gemmFqNames grp, v0252-8sbadopt); t2g4 / t2g16 /
// c2g4 stay reachable through WHIRL_FQ_RASTER (development A/B).
struct FqProto {
    const char* name;
    int base;
};
inline constexpr FqProto kFqProto[] = {{"t2g4", 1}, {"t2g8", 1}, {"t2g16", 1}, {"c2g4", 0}, {"c2g8", 0}};
// v0252-valuout (PLAN V1-53): the seven v0252-8sbvalu VALU-lever prototypes (d185546; all REVERT on the 8060S, q7 951b-955,
// KB v0252-8sbvalu:gemmfq-valu-levers) are NOT in the shipped gfx1151 object: unused kernels move the object layout
// (KB v0252-8sbal:e2e-raster-g8). Their source stays in git history (d185546) and research\b8s. kt gemm checks absence.
inline constexpr const char* kFqRetired[] = {"t2l", "t2s", "t2ls", "t2lsp", "t2g8ls", "c2g8ls", "w5ls"};

}  // namespace kt
