// GGUF reader (see include/whirl/gguf.h).
// SPDX-License-Identifier: Apache-2.0
//
// Ported from the project's own prototype reader (gguf.zig) and extended
// to the full ggml type table and to memory-mapped payloads. Format facts
// follow the public GGUF specification (ggml docs/gguf.md); block sizes
// are the documented ggml block layouts.

#include "whirl/gguf.h"

#include <cstring>
#include <limits>
#include <sstream>

namespace whirl::gguf {

namespace {

struct TypeEntry {
    GgmlType type;
    TypeTraits traits;
};

constexpr TypeEntry k_types[] = {
    {GgmlType::f32, {"f32", 1, 4}},
    {GgmlType::f16, {"f16", 1, 2}},
    {GgmlType::q4_0, {"q4_0", 32, 18}},
    {GgmlType::q4_1, {"q4_1", 32, 20}},
    {GgmlType::q5_0, {"q5_0", 32, 22}},
    {GgmlType::q5_1, {"q5_1", 32, 24}},
    {GgmlType::q8_0, {"q8_0", 32, 34}},
    {GgmlType::q8_1, {"q8_1", 32, 40}},
    {GgmlType::q2_k, {"q2_K", 256, 84}},
    {GgmlType::q3_k, {"q3_K", 256, 110}},
    {GgmlType::q4_k, {"q4_K", 256, 144}},
    {GgmlType::q5_k, {"q5_K", 256, 176}},
    {GgmlType::q6_k, {"q6_K", 256, 210}},
    {GgmlType::q8_k, {"q8_K", 256, 292}},
    {GgmlType::iq2_xxs, {"iq2_xxs", 256, 66}},
    {GgmlType::iq2_xs, {"iq2_xs", 256, 74}},
    {GgmlType::iq3_xxs, {"iq3_xxs", 256, 98}},
    {GgmlType::iq1_s, {"iq1_s", 256, 50}},
    {GgmlType::iq4_nl, {"iq4_nl", 32, 18}},
    {GgmlType::iq3_s, {"iq3_s", 256, 110}},
    {GgmlType::iq2_s, {"iq2_s", 256, 82}},
    {GgmlType::iq4_xs, {"iq4_xs", 256, 136}},
    {GgmlType::i8, {"i8", 1, 1}},
    {GgmlType::i16, {"i16", 1, 2}},
    {GgmlType::i32, {"i32", 1, 4}},
    {GgmlType::i64, {"i64", 1, 8}},
    {GgmlType::f64, {"f64", 1, 8}},
    {GgmlType::iq1_m, {"iq1_m", 256, 56}},
    {GgmlType::bf16, {"bf16", 1, 2}},
    {GgmlType::tq1_0, {"tq1_0", 256, 54}},
    {GgmlType::tq2_0, {"tq2_0", 256, 66}},
    {GgmlType::mxfp4, {"mxfp4", 32, 17}},
    {GgmlType::nvfp4, {"nvfp4", 64, 36}},
    {GgmlType::q1_0, {"q1_0", 128, 18}},
    {GgmlType::q2_0, {"q2_0", 64, 18}},
};

constexpr std::uint32_t k_magic = 0x46554747;  // "GGUF" little-endian
constexpr std::uint32_t k_max_dims = 4;

std::size_t fixedWidth(ValueType t) {
    switch (t) {
        case ValueType::u8:
        case ValueType::i8:
        case ValueType::boolean: return 1;
        case ValueType::u16:
        case ValueType::i16: return 2;
        case ValueType::u32:
        case ValueType::i32:
        case ValueType::f32: return 4;
        case ValueType::u64:
        case ValueType::i64:
        case ValueType::f64: return 8;
        default: return 0;
    }
}

bool knownValueType(std::uint32_t t) { return t <= 12; }

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> b) : buf_(b) {}

    std::size_t pos() const { return pos_; }

    void need(std::uint64_t n) const {
        if (n > buf_.size() - pos_) throw Error("gguf: truncated file");
    }

    template <class T>
    T scalar() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, buf_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }

    std::string_view str() {
        const std::uint64_t n = scalar<std::uint64_t>();
        need(n);
        std::string_view s(reinterpret_cast<const char*>(buf_.data() + pos_), static_cast<std::size_t>(n));
        pos_ += static_cast<std::size_t>(n);
        return s;
    }

    void skip(ValueType t, int depth = 0) {
        if (const std::size_t w = fixedWidth(t)) {
            need(w);
            pos_ += w;
            return;
        }
        if (t == ValueType::string) {
            (void)str();
            return;
        }
        if (t == ValueType::array) {
            if (depth > 8) throw Error("gguf: arrays nested too deeply");
            const std::uint32_t et = scalar<std::uint32_t>();
            if (!knownValueType(et)) throw Error("gguf: bad array element type " + std::to_string(et));
            const std::uint64_t n = scalar<std::uint64_t>();
            skipN(static_cast<ValueType>(et), n, depth + 1);
            return;
        }
        throw Error("gguf: bad value type");
    }

    void skipN(ValueType et, std::uint64_t n, int depth) {
        if (const std::size_t w = fixedWidth(et)) {
            if (n > (std::numeric_limits<std::uint64_t>::max)() / w) throw Error("gguf: array too large");
            need(n * w);
            pos_ += static_cast<std::size_t>(n * w);
            return;
        }
        for (std::uint64_t k = 0; k < n; ++k) skip(et, depth);
    }

    Value value(ValueType t) {
        Value v;
        v.type = t;
        switch (t) {
            case ValueType::u8: v.u = scalar<std::uint8_t>(); break;
            case ValueType::u16: v.u = scalar<std::uint16_t>(); break;
            case ValueType::u32: v.u = scalar<std::uint32_t>(); break;
            case ValueType::u64: v.u = scalar<std::uint64_t>(); break;
            case ValueType::i8: v.i = scalar<std::int8_t>(); break;
            case ValueType::i16: v.i = scalar<std::int16_t>(); break;
            case ValueType::i32: v.i = scalar<std::int32_t>(); break;
            case ValueType::i64: v.i = scalar<std::int64_t>(); break;
            case ValueType::f32: v.f = scalar<float>(); break;
            case ValueType::f64: v.f = scalar<double>(); break;
            case ValueType::boolean: {
                const std::uint8_t b = scalar<std::uint8_t>();
                if (b > 1) throw Error("gguf: bad bool value");
                v.u = b;
                break;
            }
            case ValueType::string: v.str = str(); break;
            case ValueType::array: {
                const std::uint32_t et = scalar<std::uint32_t>();
                if (!knownValueType(et)) throw Error("gguf: bad array element type " + std::to_string(et));
                v.arr.elem = static_cast<ValueType>(et);
                v.arr.len = scalar<std::uint64_t>();
                const std::size_t start = pos_;
                skipN(v.arr.elem, v.arr.len, 1);
                v.arr.raw = buf_.subspan(start, pos_ - start);
                break;
            }
            default: throw Error("gguf: bad value type");
        }
        return v;
    }

private:
    std::span<const std::uint8_t> buf_;
    std::size_t pos_ = 0;
};

}  // namespace

const char* valueTypeName(ValueType t) {
    switch (t) {
        case ValueType::u8: return "u8";
        case ValueType::i8: return "i8";
        case ValueType::u16: return "u16";
        case ValueType::i16: return "i16";
        case ValueType::u32: return "u32";
        case ValueType::i32: return "i32";
        case ValueType::f32: return "f32";
        case ValueType::boolean: return "bool";
        case ValueType::string: return "str";
        case ValueType::array: return "arr";
        case ValueType::u64: return "u64";
        case ValueType::i64: return "i64";
        case ValueType::f64: return "f64";
    }
    return "?";
}

const TypeTraits* typeTraits(GgmlType t) {
    for (const auto& e : k_types)
        if (e.type == t) return &e.traits;
    return nullptr;
}

std::string typeName(GgmlType t) {
    if (const auto* tr = typeTraits(t)) return tr->name;
    return "type" + std::to_string(static_cast<std::uint32_t>(t));
}

bool Value::isUnsigned() const {
    return type == ValueType::u8 || type == ValueType::u16 || type == ValueType::u32 || type == ValueType::u64;
}

bool Value::isSigned() const {
    return type == ValueType::i8 || type == ValueType::i16 || type == ValueType::i32 || type == ValueType::i64;
}

// a * b without wrapping; false (r untouched) on overflow
static bool mulOk(std::uint64_t a, std::uint64_t b, std::uint64_t& r) {
    if (b != 0 && a > (std::numeric_limits<std::uint64_t>::max)() / b) return false;
    r = a * b;
    return true;
}

std::uint64_t TensorInfo::rowBytes() const {
    const TypeTraits* tr = typeTraits(type);
    if (!tr || ne[0] % tr->block_size != 0) return 0;
    std::uint64_t rb = 0;
    if (!mulOk(ne[0] / tr->block_size, tr->type_size, rb)) return 0;
    return rb;
}

std::uint64_t TensorInfo::nbytes() const {
    std::uint64_t rows = 1, nb = 0;
    for (int d = 1; d < 4; ++d)
        if (!mulOk(rows, ne[d], rows)) return 0;
    if (!mulOk(rowBytes(), rows, nb)) return 0;
    return nb;
}

File File::open(const std::string& path) {
    File f;
    f.map_ = MappedFile(path);
    f.bytes_ = f.map_.bytes();
    f.parseImpl();
    return f;
}

File File::parse(std::span<const std::uint8_t> bytes) {
    File f;
    f.bytes_ = bytes;
    f.parseImpl();
    return f;
}

void File::parseImpl() {
    Reader r(bytes_);
    if (r.scalar<std::uint32_t>() != k_magic) throw Error("gguf: bad magic (not a GGUF file)");
    version_ = r.scalar<std::uint32_t>();
    if (version_ < 2 || version_ > 3) throw Error("gguf: unsupported version " + std::to_string(version_));
    const std::uint64_t n_tensors = r.scalar<std::uint64_t>();
    const std::uint64_t n_kv = r.scalar<std::uint64_t>();
    // every entry takes at least 8 bytes, so a count beyond the file size is corrupt
    if (n_tensors > bytes_.size() / 8 || n_kv > bytes_.size() / 8) throw Error("gguf: implausible header counts");
    // a key/value entry takes >= 13 bytes (key length, type, 1-byte value), a
    // tensor entry >= 24 (name length, n_dims, type, offset): the counts must
    // fit in what is left of the file. (Both terms are <= the file size here,
    // so the sum cannot wrap.)
    constexpr std::uint64_t k_min_kv = 13, k_min_tensor = 24;
    const std::uint64_t rest = bytes_.size() - r.pos();
    if (n_kv > rest / k_min_kv || n_tensors > rest / k_min_tensor || n_kv * k_min_kv + n_tensors * k_min_tensor > rest)
        throw Error("gguf: implausible header counts (more entries than the file can hold)");

    kv_.reserve(static_cast<std::size_t>(n_kv));
    for (std::uint64_t k = 0; k < n_kv; ++k) {
        KeyValue e;
        e.key = r.str();
        const std::uint32_t t = r.scalar<std::uint32_t>();
        if (!knownValueType(t)) throw Error("gguf: bad value type " + std::to_string(t) + " for key " + std::string(e.key));
        e.value = r.value(static_cast<ValueType>(t));
        if (!kv_index_.emplace(e.key, kv_.size()).second) throw Error("gguf: duplicate key " + std::string(e.key));
        kv_.push_back(e);
    }

    if (const Value* a = find("general.alignment")) {
        if (a->type != ValueType::u32) throw Error("gguf: general.alignment must be u32");
        alignment_ = a->u;
        if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) throw Error("gguf: alignment is not a power of two");
    }

    tensors_.reserve(static_cast<std::size_t>(n_tensors));
    for (std::uint64_t k = 0; k < n_tensors; ++k) {
        TensorInfo t;
        t.name = r.str();
        t.n_dims = r.scalar<std::uint32_t>();
        if (t.n_dims > k_max_dims) throw Error("gguf: tensor " + std::string(t.name) + " has too many dimensions");
        for (std::uint32_t d = 0; d < t.n_dims; ++d) {
            t.ne[d] = r.scalar<std::uint64_t>();
            // ggml dimensions are int64
            if (t.ne[d] > static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
                throw Error("gguf: tensor " + std::string(t.name) + " has a dimension that is too large");
        }
        {
            // elements(), rows() and the byte size must not wrap
            std::uint64_t rows = 1, elems = 0, nb = 0;
            for (std::uint32_t d = 1; d < 4; ++d)
                if (!mulOk(rows, t.ne[d], rows)) throw Error("gguf: tensor " + std::string(t.name) + " shape overflows");
            if (!mulOk(t.ne[0], rows, elems)) throw Error("gguf: tensor " + std::string(t.name) + " shape overflows");
            t.type = static_cast<GgmlType>(r.scalar<std::uint32_t>());
            if (const TypeTraits* tr = typeTraits(t.type); tr && t.ne[0] % tr->block_size == 0) {
                std::uint64_t rb = 0;
                if (!mulOk(t.ne[0] / tr->block_size, tr->type_size, rb) || !mulOk(rb, rows, nb))
                    throw Error("gguf: tensor " + std::string(t.name) + " size overflows");
            }
        }
        t.offset = r.scalar<std::uint64_t>();
        if (t.offset % alignment_ != 0) throw Error("gguf: tensor " + std::string(t.name) + " is misaligned");
        if (!tensor_index_.emplace(t.name, tensors_.size()).second)
            throw Error("gguf: duplicate tensor " + std::string(t.name));
        tensors_.push_back(t);
    }

    data_offset_ = (r.pos() + alignment_ - 1) / alignment_ * alignment_;
    // tensor payloads must lie inside the file (unknown types are listed but not checked)
    for (const auto& t : tensors_) {
        const std::uint64_t nb = t.nbytes();
        if (nb == 0) continue;
        if (!inBounds(t, nb)) throw Error("gguf: tensor " + std::string(t.name) + " extends past the end of the file");
    }
}

bool File::inBounds(const TensorInfo& t, std::uint64_t nb) const {
    if (nb == 0) return true;
    const std::uint64_t size = bytes_.size();
    // subtraction only: data_offset_ + t.offset + nb may not fit in 64 bits
    if (data_offset_ > size) return false;
    const std::uint64_t avail = size - data_offset_;
    return t.offset <= avail && nb <= avail - t.offset;
}

const Value* File::find(std::string_view key) const {
    const auto it = kv_index_.find(key);
    return it == kv_index_.end() ? nullptr : &kv_[it->second].value;
}

static const Value& need(const File& f, std::string_view key) {
    const Value* v = f.find(key);
    if (!v) throw Error("gguf: key not found: " + std::string(key));
    return *v;
}

std::uint64_t File::getUint(std::string_view key) const {
    const Value& v = need(*this, key);
    if (v.isUnsigned()) return v.u;
    if (v.isSigned() && v.i >= 0) return static_cast<std::uint64_t>(v.i);
    throw Error("gguf: key " + std::string(key) + " is not a non-negative integer");
}

std::uint64_t File::getUintOr(std::string_view key, std::uint64_t def) const {
    return has(key) ? getUint(key) : def;
}

std::int64_t File::getInt(std::string_view key) const {
    const Value& v = need(*this, key);
    if (v.isSigned()) return v.i;
    if (v.isUnsigned() && v.u <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
        return static_cast<std::int64_t>(v.u);
    throw Error("gguf: key " + std::string(key) + " is not an integer");
}

double File::getFloat(std::string_view key) const {
    const Value& v = need(*this, key);
    if (v.isFloat()) return v.f;
    if (v.isUnsigned()) return static_cast<double>(v.u);
    if (v.isSigned()) return static_cast<double>(v.i);
    throw Error("gguf: key " + std::string(key) + " is not a number");
}

bool File::getBool(std::string_view key) const {
    const Value& v = need(*this, key);
    if (v.type != ValueType::boolean) throw Error("gguf: key " + std::string(key) + " is not a bool");
    return v.u != 0;
}

std::optional<bool> File::getBoolOpt(std::string_view key) const {
    if (!has(key)) return std::nullopt;
    return getBool(key);
}

std::string_view File::getString(std::string_view key) const {
    const Value& v = need(*this, key);
    if (v.type != ValueType::string) throw Error("gguf: key " + std::string(key) + " is not a string");
    return v.str;
}

std::string_view File::getStringOr(std::string_view key, std::string_view def) const {
    return has(key) ? getString(key) : def;
}

const Array& File::getArray(std::string_view key) const {
    const Value& v = need(*this, key);
    if (v.type != ValueType::array) throw Error("gguf: key " + std::string(key) + " is not an array");
    return v.arr;
}

std::vector<std::string_view> File::stringArray(std::string_view key) const {
    const Array& a = getArray(key);
    if (a.elem != ValueType::string) throw Error("gguf: key " + std::string(key) + " is not a string array");
    std::vector<std::string_view> out;
    out.reserve(static_cast<std::size_t>(a.len));
    Reader r(a.raw);
    for (std::uint64_t k = 0; k < a.len; ++k) out.push_back(r.str());
    return out;
}

std::vector<std::int64_t> File::intArray(std::string_view key) const {
    const Array& a = getArray(key);
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(a.len));
    Reader r(a.raw);
    for (std::uint64_t k = 0; k < a.len; ++k) {
        switch (a.elem) {
            case ValueType::u8: out.push_back(r.scalar<std::uint8_t>()); break;
            case ValueType::i8: out.push_back(r.scalar<std::int8_t>()); break;
            case ValueType::u16: out.push_back(r.scalar<std::uint16_t>()); break;
            case ValueType::i16: out.push_back(r.scalar<std::int16_t>()); break;
            case ValueType::u32: out.push_back(r.scalar<std::uint32_t>()); break;
            case ValueType::i32: out.push_back(r.scalar<std::int32_t>()); break;
            case ValueType::u64: out.push_back(static_cast<std::int64_t>(r.scalar<std::uint64_t>())); break;
            case ValueType::i64: out.push_back(r.scalar<std::int64_t>()); break;
            case ValueType::boolean: out.push_back(r.scalar<std::uint8_t>()); break;
            default: throw Error("gguf: key " + std::string(key) + " is not an integer array");
        }
    }
    return out;
}

std::vector<double> File::floatArray(std::string_view key) const {
    const Array& a = getArray(key);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(a.len));
    Reader r(a.raw);
    for (std::uint64_t k = 0; k < a.len; ++k) {
        if (a.elem == ValueType::f32)
            out.push_back(r.scalar<float>());
        else if (a.elem == ValueType::f64)
            out.push_back(r.scalar<double>());
        else
            throw Error("gguf: key " + std::string(key) + " is not a float array");
    }
    return out;
}

const TensorInfo* File::tensor(std::string_view name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

std::span<const std::uint8_t> File::tensorData(const TensorInfo& t) const {
    const std::uint64_t nb = t.nbytes();
    if (nb == 0) throw Error("gguf: tensor " + std::string(t.name) + " has an unsupported type or shape");
    if (!inBounds(t, nb)) throw Error("gguf: tensor " + std::string(t.name) + " is out of bounds");
    const std::uint64_t off = data_offset_ + t.offset;  // cannot wrap: inBounds
    return bytes_.subspan(static_cast<std::size_t>(off), static_cast<std::size_t>(nb));
}

static void formatScalar(std::ostringstream& os, ValueType t, Reader& r) {
    switch (t) {
        case ValueType::u8: os << static_cast<unsigned>(r.scalar<std::uint8_t>()); break;
        case ValueType::i8: os << static_cast<int>(r.scalar<std::int8_t>()); break;
        case ValueType::u16: os << r.scalar<std::uint16_t>(); break;
        case ValueType::i16: os << r.scalar<std::int16_t>(); break;
        case ValueType::u32: os << r.scalar<std::uint32_t>(); break;
        case ValueType::i32: os << r.scalar<std::int32_t>(); break;
        case ValueType::u64: os << r.scalar<std::uint64_t>(); break;
        case ValueType::i64: os << r.scalar<std::int64_t>(); break;
        case ValueType::f32: os << r.scalar<float>(); break;
        case ValueType::f64: os << r.scalar<double>(); break;
        case ValueType::boolean: os << (r.scalar<std::uint8_t>() ? "true" : "false"); break;
        case ValueType::string: {
            const auto s = r.str();
            os << '"' << (s.size() > 60 ? std::string(s.substr(0, 60)) + "..." : std::string(s)) << '"';
            break;
        }
        case ValueType::array: os << "[...]"; r.skip(ValueType::array); break;
    }
}

std::string File::formatValue(const Value& v, std::size_t max_array_items) {
    std::ostringstream os;
    os.precision(9);
    switch (v.type) {
        case ValueType::string:
            if (v.str.size() > 120)
                os << '"' << v.str.substr(0, 120) << "...\" (" << v.str.size() << " bytes)";
            else
                os << '"' << v.str << '"';
            break;
        case ValueType::boolean: os << (v.u ? "true" : "false"); break;
        case ValueType::f32:
        case ValueType::f64: os << v.f; break;
        case ValueType::array: {
            os << "arr[" << valueTypeName(v.arr.elem) << ", " << v.arr.len << "] [";
            Reader r(v.arr.raw);
            for (std::uint64_t k = 0; k < v.arr.len && k < max_array_items; ++k) {
                if (k) os << ", ";
                formatScalar(os, v.arr.elem, r);
            }
            if (v.arr.len > max_array_items) os << ", ...";
            os << "]";
            break;
        }
        default:
            if (v.isSigned())
                os << v.i;
            else
                os << v.u;
    }
    return os.str();
}

}  // namespace whirl::gguf
