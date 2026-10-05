// GGUF reader: header, metadata key/values, tensor directory, memory-mapped
// tensor payloads. Supports GGUF v2 and v3 (little-endian).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/common.h"

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace whirl::gguf {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class ValueType : std::uint32_t {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};
const char* valueTypeName(ValueType t);

// ggml tensor types (numbering of ggml.h; removed types are absent).
enum class GgmlType : std::uint32_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    q1_0 = 41,
    q2_0 = 42,
};

struct TypeTraits {
    const char* name;          // llama.cpp spelling, e.g. "q4_K"
    std::uint32_t block_size;  // values per block
    std::uint32_t type_size;   // bytes per block
};
// nullptr for an unknown type id.
const TypeTraits* typeTraits(GgmlType t);
std::string typeName(GgmlType t);  // "type<N>" for unknown ids

struct Array {
    ValueType elem = ValueType::u8;
    std::uint64_t len = 0;
    std::span<const std::uint8_t> raw;  // payload bytes (strings length-prefixed)
};

struct Value {
    ValueType type = ValueType::u8;
    std::uint64_t u = 0;  // u8/u16/u32/u64, bool
    std::int64_t i = 0;   // i8/i16/i32/i64
    double f = 0;         // f32/f64
    std::string_view str;
    Array arr;

    bool isUnsigned() const;
    bool isSigned() const;
    bool isFloat() const { return type == ValueType::f32 || type == ValueType::f64; }
};

struct KeyValue {
    std::string_view key;
    Value value;
};

struct TensorInfo {
    std::string_view name;
    std::uint32_t n_dims = 0;
    std::uint64_t ne[4] = {1, 1, 1, 1};
    GgmlType type = GgmlType::f32;
    std::uint64_t offset = 0;  // relative to the data section

    // File::parse rejects shapes whose element count or byte size overflows,
    // so elements() / rows() cannot wrap for tensors of a parsed file.
    std::uint64_t elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    // 0 if the type is unknown or ne[0] is not a multiple of the block size.
    std::uint64_t rowBytes() const;
    std::uint64_t rows() const { return ne[1] * ne[2] * ne[3]; }
    // rowBytes() * rows(); 0 (like an unknown type) if that would overflow.
    std::uint64_t nbytes() const;
};

class File {
public:
    // Maps and parses the file; throws gguf::Error on malformed input.
    static File open(const std::string& path);
    // Parses an in-memory image (tests); `bytes` must outlive the File.
    static File parse(std::span<const std::uint8_t> bytes);

    std::uint32_t version() const { return version_; }
    std::uint64_t alignment() const { return alignment_; }
    std::uint64_t dataOffset() const { return data_offset_; }
    std::uint64_t fileSize() const { return bytes_.size(); }
    const std::string& path() const { return map_.path(); }

    const std::vector<KeyValue>& kv() const { return kv_; }  // file order
    const Value* find(std::string_view key) const;
    bool has(std::string_view key) const { return find(key) != nullptr; }

    std::uint64_t getUint(std::string_view key) const;  // also accepts non-negative signed
    std::uint64_t getUintOr(std::string_view key, std::uint64_t def) const;
    std::int64_t getInt(std::string_view key) const;
    double getFloat(std::string_view key) const;  // also accepts integers
    bool getBool(std::string_view key) const;
    std::optional<bool> getBoolOpt(std::string_view key) const;
    std::string_view getString(std::string_view key) const;
    std::string_view getStringOr(std::string_view key, std::string_view def) const;
    const Array& getArray(std::string_view key) const;
    std::vector<std::string_view> stringArray(std::string_view key) const;
    std::vector<std::int64_t> intArray(std::string_view key) const;
    std::vector<double> floatArray(std::string_view key) const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* tensor(std::string_view name) const;
    std::uint64_t absOffset(const TensorInfo& t) const { return data_offset_ + t.offset; }
    // Payload bytes of a tensor inside the mapping (bounds-checked).
    std::span<const std::uint8_t> tensorData(const TensorInfo& t) const;
    // True when [dataOffset() + t.offset, + nbytes) lies inside the file
    // (overflow-safe; nbytes 0 counts as inside).
    bool inBounds(const TensorInfo& t, std::uint64_t nbytes) const;

    // Human-readable value, e.g. for gguf-info (arrays summarized).
    static std::string formatValue(const Value& v, std::size_t max_array_items = 8);

private:
    void parseImpl();

    MappedFile map_;
    std::span<const std::uint8_t> bytes_;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = 32;
    std::uint64_t data_offset_ = 0;
    std::vector<KeyValue> kv_;
    std::unordered_map<std::string_view, std::size_t> kv_index_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string_view, std::size_t> tensor_index_;
};

}  // namespace whirl::gguf
