// whirl-kernel-test: shared harness (device buffers, comparisons, report).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
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
    bool verbose = false;
    std::string family;   // current family label

private:
    std::vector<Result> results_;
    std::vector<std::string> skipped_;
};

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
    std::mt19937_64 rng{12345};
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
void testGemm(Ctx& c);
void testAttn(Ctx& c);
void testGdn(Ctx& c);
void testMoe(Ctx& c);
void testMoeMx(Ctx& c);
void testMisc(Ctx& c);

inline unsigned cdiv(std::uint64_t a, std::uint64_t b) { return static_cast<unsigned>((a + b - 1) / b); }

}  // namespace kt
