// whirl-kernel-test: harness implementation.
// SPDX-License-Identifier: Apache-2.0

#include "kt.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace kt {

float h2f(std::uint16_t h) {
    const std::uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(static_cast<float>(m | 1024u), static_cast<int>(e) - 25);
    return s ? -v : v;
}

std::uint16_t d2h(double d) {
    if (std::isnan(d)) return 0x7e00;
    const std::uint16_t sign = std::signbit(d) ? 0x8000 : 0;
    const double a = std::fabs(d);
    if (a == 0) return sign;
    if (a < std::ldexp(1.0, -14)) {
        const double r = std::nearbyint(a * std::ldexp(1.0, 24));  // ties to even
        return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(r));
    }
    int E = 0;
    (void)std::frexp(a, &E);
    int e = E - 1;  // a in [2^e, 2^(e+1))
    double r = std::nearbyint(std::ldexp(a, 10 - e));  // [1024, 2048]
    if (r >= 2048) {
        r = 1024;
        ++e;
    }
    if (e + 15 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00);
    return static_cast<std::uint16_t>(sign | ((e + 15) << 10) | (static_cast<int>(r) - 1024));
}

std::uint16_t f2h(float f) { return d2h(static_cast<double>(f)); }

// ---------------------------------------------------------------------------

static const char* kindName(Kind k) {
    switch (k) {
        case Kind::exact: return "exact";
        case Kind::tol: return "tol";
        case Kind::invariant: return "bitwise-invariance";
    }
    return "?";
}

void Report::add(Result r) {
    if (r.family.empty()) r.family = family;
    if (verbose || !r.pass) {
        std::printf("  [%s] %-5s %-14s %s", r.pass ? "ok" : "FAIL", r.family.c_str(), kindName(r.kind), r.name.c_str());
        if (r.kind == Kind::tol)
            std::printf("  n=%zu worst=%.3g of bound, max|err|=%.3g", r.n, r.metric, r.max_err);
        else
            std::printf("  n=%zu mismatches=%zu max|d|=%.3g", r.n, r.mismatches, r.max_err);
        if (!r.note.empty()) std::printf("  (%s)", r.note.c_str());
        std::printf("\n");
        std::fflush(stdout);
    }
    results_.push_back(std::move(r));
}

void Report::skip(const std::string& fam, const std::string& name, const std::string& why) {
    skipped_.push_back(fam + " " + name + ": " + why);
    if (verbose) std::printf("  [skip] %s %s: %s\n", fam.c_str(), name.c_str(), why.c_str());
}

int Report::summary() const {
    std::map<std::string, std::array<int, 6>> by;  // fam -> exact ok/fail, tol ok/fail, inv ok/fail
    int fails = 0;
    for (const Result& r : results_) {
        auto& a = by[r.family];
        const int base = r.kind == Kind::exact ? 0 : (r.kind == Kind::tol ? 2 : 4);
        ++a[base + (r.pass ? 0 : 1)];
        if (!r.pass) ++fails;
    }
    std::printf("\nsummary (checks per family; exact = bit-exact vs CPU reference, tol = within tolerance of a\n"
                "double-precision CPU reference, invariance = bitwise equality between GPU variants)\n");
    std::printf("  %-6s %12s %12s %16s\n", "family", "exact ok/n", "tol ok/n", "invariance ok/n");
    std::array<int, 6> tot{};
    for (const auto& [fam, a] : by) {
        std::printf("  %-6s %7d/%-4d %7d/%-4d %10d/%-4d\n", fam.c_str(), a[0], a[0] + a[1], a[2], a[2] + a[3], a[4],
                    a[4] + a[5]);
        for (int i = 0; i < 6; ++i) tot[i] += a[i];
    }
    std::printf("  %-6s %7d/%-4d %7d/%-4d %10d/%-4d\n", "total", tot[0], tot[0] + tot[1], tot[2], tot[2] + tot[3],
                tot[4], tot[4] + tot[5]);
    if (!skipped_.empty()) {
        std::printf("  skipped: %zu\n", skipped_.size());
        for (const auto& s : skipped_) std::printf("    %s\n", s.c_str());
    }
    std::printf("%s: %d failing check(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails;
}

Result cmpExactBits(const std::string& name, const void* got, const void* ref, std::size_t n, std::size_t eb,
                    Kind kind) {
    Result r;
    r.name = name;
    r.kind = kind;
    r.n = n;
    const auto* g = static_cast<const std::uint8_t*>(got);
    const auto* f = static_cast<const std::uint8_t*>(ref);
    std::size_t first = n;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(g + i * eb, f + i * eb, eb) != 0) {
            if (first == n) first = i;
            ++r.mismatches;
            if (eb == 4) {
                float a, b;
                std::memcpy(&a, g + i * eb, 4);
                std::memcpy(&b, f + i * eb, 4);
                const double d = std::fabs(static_cast<double>(a) - b);
                if (!(d <= r.max_err)) r.max_err = std::isnan(d) ? INFINITY : std::max(r.max_err, d);
            } else if (eb == 2) {
                std::uint16_t a, b;
                std::memcpy(&a, g + i * eb, 2);
                std::memcpy(&b, f + i * eb, 2);
                r.max_err = std::max(r.max_err, static_cast<double>(std::fabs(h2f(a) - h2f(b))));
            } else {
                r.max_err = std::max(r.max_err, 1.0);
            }
        }
    }
    r.pass = r.mismatches == 0;
    if (!r.pass) r.note = "first mismatch at " + std::to_string(first);
    return r;
}

Result cmpTol(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref,
              const std::vector<double>& scale, double rtol, double atol) {
    Result r;
    r.name = name;
    r.kind = Kind::tol;
    r.n = std::min(got.size(), ref.size());
    std::size_t worst = 0;
    for (std::size_t i = 0; i < r.n; ++i) {
        const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
        const double bound = atol + rtol * (scale.empty() ? std::fabs(ref[i]) : scale[i]);
        const double m = std::isfinite(err) ? err / bound : INFINITY;
        if (!std::isfinite(static_cast<double>(got[i]))) ++r.mismatches;
        if (m > 1) ++r.mismatches;
        if (m > r.metric || std::isnan(m)) {
            r.metric = std::isnan(m) ? INFINITY : m;
            worst = i;
        }
        r.max_err = std::max(r.max_err, std::isfinite(err) ? err : INFINITY);
    }
    r.pass = r.mismatches == 0 && got.size() == ref.size();
    if (!r.pass && r.n) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "worst at %zu: got %.7g ref %.7g; %zu outside", worst, got[worst], ref[worst],
                      r.mismatches);
        r.note = buf;
    }
    return r;
}

Result cmpTolRel(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref, double rtol,
                 double atol) {
    return cmpTol(name, got, ref, {}, rtol, atol);
}

// ---------------------------------------------------------------------------

std::vector<float> Ctx::randn(std::size_t n, float sigma) {
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
std::vector<float> Ctx::randu(std::size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
std::vector<int> Ctx::randi(std::size_t n, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    std::vector<int> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

const whirl::gguf::File* Ctx::open(const std::string& path) {
    if (path.empty()) return nullptr;
    auto it = files.find(path);
    if (it != files.end()) return it->second.get();
    try {
        auto f = std::make_unique<whirl::gguf::File>(whirl::gguf::File::open(path));
        return files.emplace(path, std::move(f)).first->second.get();
    } catch (const std::exception& e) {
        std::printf("  (cannot open %s: %s)\n", path.c_str(), e.what());
        files.emplace(path, nullptr);
        return nullptr;
    }
}

// ---------------------------------------------------------------------------

static QType qtypeOf(whirl::gguf::GgmlType t) {
    using G = whirl::gguf::GgmlType;
    switch (t) {
        case G::f32: return QType::f32;
        case G::f16: return QType::f16;
        case G::q8_0: return QType::q8_0;
        case G::q3_k: return QType::q3_k;
        case G::q4_k: return QType::q4_k;
        case G::q5_k: return QType::q5_k;
        case G::q6_k: return QType::q6_k;
        case G::iq4_nl: return QType::iq4_nl;
        case G::iq3_s: return QType::iq3_s;
        case G::iq4_xs: return QType::iq4_xs;
        case G::mxfp4: return QType::mxfp4;
        default: throw std::runtime_error("unsupported tensor type " + whirl::gguf::typeName(t));
    }
}

HostMat loadMat(const whirl::gguf::File& f, const std::string& tensor, int r0, int nrows) {
    const auto* t = f.tensor(tensor);
    if (t == nullptr) throw std::runtime_error("missing tensor " + tensor);
    HostMat m;
    m.name = tensor;
    m.type = qtypeOf(t->type);
    m.ncols = static_cast<int>(t->ne[0]);
    const int total = static_cast<int>(t->rows());
    if (nrows <= 0 || r0 + nrows > total) nrows = total - r0;
    m.nrows = nrows;
    m.row_bytes = t->rowBytes();
    if (m.row_bytes != wk::rowBytes(m.type, m.ncols)) throw std::runtime_error("row size mismatch " + tensor);
    const auto all = f.tensorData(*t);
    m.data.assign(all.begin() + static_cast<std::ptrdiff_t>(static_cast<std::uint64_t>(r0) * m.row_bytes),
                  all.begin() + static_cast<std::ptrdiff_t>(static_cast<std::uint64_t>(r0 + nrows) * m.row_bytes));
    if (m.type == QType::mxfp4) {
        m.ref.resize(static_cast<std::size_t>(nrows));
        for (int r = 0; r < nrows; ++r) {
            std::uint8_t* row = m.data.data() + static_cast<std::size_t>(r) * m.row_bytes;
            m.ref[static_cast<std::size_t>(r)] = wk::repackMxfp4Row(row, row, m.ncols, nullptr);
        }
    }
    return m;
}

static void putHalf(std::uint8_t* p, float v) {
    const std::uint16_t h = f2h(v);
    std::memcpy(p, &h, 2);
}

HostMat randomMat(Ctx& c, QType t, int nrows, int ncols) {
    HostMat m;
    m.type = t;
    m.nrows = nrows;
    m.ncols = ncols;
    m.row_bytes = wk::rowBytes(t, ncols);
    m.name = std::string("random_") + wk::typeSuffix(t);
    m.data.resize(static_cast<std::size_t>(nrows) * m.row_bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : m.data) b = static_cast<std::uint8_t>(byte(c.rng));
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    const wk::BlockInfo bi = wk::blockInfo(t);
    const std::size_t nblk = m.data.size() / static_cast<std::size_t>(bi.bytes);
    for (std::size_t b = 0; b < nblk; ++b) {
        std::uint8_t* p = m.data.data() + b * static_cast<std::size_t>(bi.bytes);
        switch (t) {
            case QType::f32: {
                const float v = std::normal_distribution<float>(0.f, 0.05f)(c.rng);
                std::memcpy(p, &v, 4);
                break;
            }
            case QType::f16: putHalf(p, std::normal_distribution<float>(0.f, 0.05f)(c.rng)); break;
            case QType::q8_0:
            case QType::iq4_nl:
            case QType::iq4_xs:
            case QType::iq3_s: putHalf(p, sc(c.rng)); break;
            case QType::q4_k:
            case QType::q5_k:
                putHalf(p, sc(c.rng));
                putHalf(p + 2, sc(c.rng));
                break;
            case QType::q6_k: putHalf(p + 208, sc(c.rng)); break;
            case QType::q3_k: putHalf(p + 108, sc(c.rng)); break;
            case QType::mxfp4: {
                std::uniform_int_distribution<int> e(118, 126);
                for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(e(c.rng));
                break;
            }
        }
    }
    if (t == QType::mxfp4) {
        m.ref.resize(static_cast<std::size_t>(nrows));
        for (int r = 0; r < nrows; ++r) {
            const std::uint8_t* row = m.data.data() + static_cast<std::size_t>(r) * m.row_bytes;
            std::uint8_t e = 0;
            for (int sb = 0; sb < ncols / 256; ++sb)
                for (int i = 0; i < 8; ++i) e = std::max(e, row[sb * 136 + i]);
            m.ref[static_cast<std::size_t>(r)] = e;
        }
    }
    return m;
}

std::vector<HostMat> sampleMats(Ctx& c, int max_rows) {
    struct Pick {
        QType t;
        const std::string* model;
        const char* tensor;
    };
    const Pick picks[] = {
        {QType::q4_k, &c.models.q4, "blk.3.ffn_gate.weight"},   // 5120 x 17408
        {QType::q5_k, &c.models.q4, "blk.0.attn_qkv.weight"},   // 5120 x 10240
        {QType::q6_k, &c.models.q4, "blk.3.attn_output.weight"},// 6144 x 5120
        {QType::q3_k, &c.models.q4, "blk.0.ffn_up.weight"},     // 5120 x 17408
        {QType::q8_0, &c.models.q4, "blk.0.ssm_alpha.weight"},  // 5120 x 48
        {QType::iq4_nl, &c.models.q4, "blk.3.ffn_down.weight"}, // 17408 x 5120
        {QType::iq4_xs, &c.models.q4, "blk.0.ffn_down.weight"}, // 17408 x 5120
        {QType::iq3_s, &c.models.q4, "blk.11.ffn_gate.weight"}, // 5120 x 17408
        {QType::mxfp4, &c.models.mx, "blk.0.ffn_down.weight"},  // 17408 x 5120
    };
    std::vector<HostMat> out;
    for (const Pick& p : picks) {
        const whirl::gguf::File* f = c.open(*p.model);
        const auto* t = f ? f->tensor(p.tensor) : nullptr;
        if (t != nullptr && qtypeOf(t->type) == p.t) {
            out.push_back(loadMat(*f, p.tensor, 0, max_rows));
            out.back().name = std::string(wk::typeSuffix(p.t)) + ":" + p.tensor;
        } else {
            out.push_back(randomMat(c, p.t, std::min(max_rows, 512), 4096));
            c.rep.skip("data", wk::typeSuffix(p.t), "model tensor not available, synthetic blocks used");
        }
    }
    out.push_back(randomMat(c, QType::f16, std::min(max_rows, 512), 4096));
    out.push_back(randomMat(c, QType::f32, std::min(max_rows, 512), 4096));
    return out;
}

}  // namespace kt
