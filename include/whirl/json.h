// Minimal JSON value, parser (RFC 8259) and Python-compatible serializer.
// SPDX-License-Identifier: Apache-2.0
//
// Objects keep insertion order. A repeated key keeps its first position and
// takes the last value (Python json.loads semantics). Integers that fit in
// int64 are integers; larger integer literals are kept verbatim.

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::json {

class ParseError : public std::runtime_error {
public:
    ParseError(const std::string& msg, std::size_t offset)
        : std::runtime_error(msg + " at byte " + std::to_string(offset)), offset_(offset) {}
    std::size_t offset() const { return offset_; }

private:
    std::size_t offset_;
};

class Value;
using Array = std::vector<Value>;
using Member = std::pair<std::string, Value>;

class Object {
public:
    const Value* find(std::string_view key) const;
    Value* find(std::string_view key);
    // Inserts or replaces (keeping the original position).
    Value& set(std::string key, Value v);
    const std::vector<Member>& members() const { return members_; }
    std::size_t size() const { return members_.size(); }
    bool empty() const { return members_.empty(); }

private:
    std::vector<Member> members_;
};

class Value {
public:
    enum class Kind { null, boolean, integer, number, big_integer, string, array, object };

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : kind_(Kind::boolean), b_(b) {}
    Value(std::int64_t i) : kind_(Kind::integer), i_(i) {}
    Value(int i) : kind_(Kind::integer), i_(i) {}
    Value(double d) : kind_(Kind::number), d_(d) {}
    Value(std::string s) : kind_(Kind::string), s_(std::move(s)) {}
    Value(const char* s) : kind_(Kind::string), s_(s) {}
    Value(Array a);
    Value(Object o);
    static Value bigInteger(std::string digits);

    Value(const Value& o);
    Value& operator=(const Value& o);
    Value(Value&&) noexcept = default;
    Value& operator=(Value&&) noexcept = default;
    ~Value();

    Kind kind() const { return kind_; }
    bool isNull() const { return kind_ == Kind::null; }
    bool isBool() const { return kind_ == Kind::boolean; }
    bool isString() const { return kind_ == Kind::string; }
    bool isArray() const { return kind_ == Kind::array; }
    bool isObject() const { return kind_ == Kind::object; }
    bool isNumber() const { return kind_ == Kind::integer || kind_ == Kind::number || kind_ == Kind::big_integer; }

    bool asBool() const { return b_; }
    std::int64_t asInt() const { return i_; }
    double asDouble() const;
    const std::string& asString() const { return s_; }  // also the digits of a big integer
    const Array& asArray() const { return *a_; }
    Array& asArray() { return *a_; }
    const Object& asObject() const { return *o_; }
    Object& asObject() { return *o_; }

    // Object member lookup; nullptr if not an object or absent.
    const Value* get(std::string_view key) const;

private:
    Kind kind_ = Kind::null;
    bool b_ = false;
    std::int64_t i_ = 0;
    double d_ = 0;
    std::string s_;
    std::unique_ptr<Array> a_;
    std::unique_ptr<Object> o_;
};

// Parses one JSON document (surrounding whitespace allowed). Strings must
// be valid UTF-8; \u escapes are decoded and lone surrogates become U+FFFD.
Value parse(std::string_view text, std::size_t max_depth = 256);

// Python json.dumps(v, ensure_ascii=False): ", " / ": " separators, floats
// as Python repr (1.0, 1e-05, 1e+20). compact = true gives "," / ":".
std::string dumpPython(const Value& v, bool compact = false);
void dumpPython(std::string& out, const Value& v, bool compact = false);
// Python repr of a float (shortest round-trip digits).
std::string pythonFloat(double d);
// String body escaping as Python json.dumps(ensure_ascii=False).
void appendEscaped(std::string& out, std::string_view s);

}  // namespace whirl::json
