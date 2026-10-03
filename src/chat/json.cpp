// JSON parser and Python-compatible serializer (see include/whirl/json.h).
// SPDX-License-Identifier: Apache-2.0
//
// New implementation from RFC 8259; the serializer follows the prototype's
// writePyJson (Python json.dumps(ensure_ascii=False) output).

#include "whirl/json.h"

#include "whirl/common.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace whirl::json {

// ---------------------------------------------------------------------------
// Object / Value

const Value* Object::find(std::string_view key) const {
    for (const auto& m : members_)
        if (m.first == key) return &m.second;
    return nullptr;
}

Value* Object::find(std::string_view key) {
    for (auto& m : members_)
        if (m.first == key) return &m.second;
    return nullptr;
}

Value& Object::set(std::string key, Value v) {
    if (Value* old = find(key)) {
        *old = std::move(v);
        return *old;
    }
    members_.emplace_back(std::move(key), std::move(v));
    return members_.back().second;
}

Value::Value(Array a) : kind_(Kind::array), a_(std::make_unique<Array>(std::move(a))) {}
Value::Value(Object o) : kind_(Kind::object), o_(std::make_unique<Object>(std::move(o))) {}
Value::~Value() = default;

Value Value::bigInteger(std::string digits) {
    Value v;
    v.kind_ = Kind::big_integer;
    v.s_ = std::move(digits);
    return v;
}

Value::Value(const Value& o) : kind_(o.kind_), b_(o.b_), i_(o.i_), d_(o.d_), s_(o.s_) {
    if (o.a_) a_ = std::make_unique<Array>(*o.a_);
    if (o.o_) o_ = std::make_unique<Object>(*o.o_);
}

Value& Value::operator=(const Value& o) {
    if (this != &o) {
        Value tmp(o);
        *this = std::move(tmp);
    }
    return *this;
}

double Value::asDouble() const {
    switch (kind_) {
        case Kind::integer: return static_cast<double>(i_);
        case Kind::number: return d_;
        case Kind::big_integer: return std::strtod(s_.c_str(), nullptr);
        default: return 0;
    }
}

const Value* Value::get(std::string_view key) const { return kind_ == Kind::object ? o_->find(key) : nullptr; }

// ---------------------------------------------------------------------------
// parser

namespace {

class Parser {
public:
    Parser(std::string_view t, std::size_t max_depth) : t_(t), max_depth_(max_depth) {}

    Value document() {
        ws();
        Value v = value(0);
        ws();
        if (p_ != t_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* msg) const { throw ParseError(msg, p_); }

    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
    }

    bool lit(const char* s) {
        const std::size_t n = std::strlen(s);
        if (t_.substr(p_, n) == s) {
            p_ += n;
            return true;
        }
        return false;
    }

    Value value(std::size_t depth) {
        if (depth > max_depth_) fail("nesting too deep");
        if (p_ >= t_.size()) fail("unexpected end of input");
        const char c = t_[p_];
        switch (c) {
            case '{': return object(depth);
            case '[': return array(depth);
            case '"': return Value(string());
            case 't':
                if (lit("true")) return Value(true);
                break;
            case 'f':
                if (lit("false")) return Value(false);
                break;
            case 'n':
                if (lit("null")) return Value(nullptr);
                break;
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return number();
        }
        fail("unexpected character");
    }

    Value object(std::size_t depth) {
        ++p_;  // {
        Object o;
        ws();
        if (p_ < t_.size() && t_[p_] == '}') {
            ++p_;
            return Value(std::move(o));
        }
        while (true) {
            ws();
            if (p_ >= t_.size() || t_[p_] != '"') fail("expected object key");
            std::string key = string();
            ws();
            if (p_ >= t_.size() || t_[p_] != ':') fail("expected ':'");
            ++p_;
            ws();
            Value v = value(depth + 1);
            o.set(std::move(key), std::move(v));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') {
                ++p_;
                continue;
            }
            if (p_ < t_.size() && t_[p_] == '}') {
                ++p_;
                return Value(std::move(o));
            }
            fail("expected ',' or '}'");
        }
    }

    Value array(std::size_t depth) {
        ++p_;  // [
        Array a;
        ws();
        if (p_ < t_.size() && t_[p_] == ']') {
            ++p_;
            return Value(std::move(a));
        }
        while (true) {
            ws();
            a.push_back(value(depth + 1));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') {
                ++p_;
                continue;
            }
            if (p_ < t_.size() && t_[p_] == ']') {
                ++p_;
                return Value(std::move(a));
            }
            fail("expected ',' or ']'");
        }
    }

    unsigned hex4() {
        if (p_ + 4 > t_.size()) fail("truncated \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = t_[p_++];
            v <<= 4;
            if (c >= '0' && c <= '9')
                v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f')
                v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                v |= static_cast<unsigned>(c - 'A' + 10);
            else
                fail("bad \\u escape");
        }
        return v;
    }

    std::string string() {
        ++p_;  // opening quote
        std::string out;
        while (true) {
            if (p_ >= t_.size()) fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(t_[p_]);
            if (c == '"') {
                ++p_;
                break;
            }
            if (c < 0x20) fail("control character in string");
            if (c == '\\') {
                ++p_;
                if (p_ >= t_.size()) fail("unterminated escape");
                const char e = t_[p_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        std::uint32_t cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            // high surrogate: combine with a following low surrogate
                            if (p_ + 6 <= t_.size() && t_[p_] == '\\' && t_[p_ + 1] == 'u') {
                                const std::size_t save = p_;
                                p_ += 2;
                                const unsigned lo = hex4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    p_ = save;
                                    cp = 0xFFFD;
                                }
                            } else {
                                cp = 0xFFFD;
                            }
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            cp = 0xFFFD;
                        }
                        appendUtf8(out, cp);
                        break;
                    }
                    default: --p_; fail("bad escape");
                }
                continue;
            }
            // raw run up to the next quote / backslash / control byte
            std::size_t q = p_;
            while (q < t_.size()) {
                const unsigned char d = static_cast<unsigned char>(t_[q]);
                if (d == '"' || d == '\\' || d < 0x20) break;
                ++q;
            }
            const std::string_view run = t_.substr(p_, q - p_);
            if (!isValidUtf8(run)) fail("invalid UTF-8 in string");
            out.append(run);
            p_ = q;
        }
        return out;
    }

    Value number() {
        const std::size_t start = p_;
        if (t_[p_] == '-') ++p_;
        if (p_ >= t_.size()) fail("bad number");
        if (t_[p_] == '0') {
            ++p_;
        } else if (t_[p_] >= '1' && t_[p_] <= '9') {
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        } else {
            fail("bad number");
        }
        bool is_float = false;
        if (p_ < t_.size() && t_[p_] == '.') {
            is_float = true;
            ++p_;
            if (p_ >= t_.size() || t_[p_] < '0' || t_[p_] > '9') fail("bad number");
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
            is_float = true;
            ++p_;
            if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
            if (p_ >= t_.size() || t_[p_] < '0' || t_[p_] > '9') fail("bad number");
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        const std::string_view s = t_.substr(start, p_ - start);
        if (!is_float) {
            std::int64_t i = 0;
            const auto r = std::from_chars(s.data(), s.data() + s.size(), i);
            if (r.ec == std::errc() && r.ptr == s.data() + s.size()) return Value(i);
            return Value::bigInteger(std::string(s));
        }
        double d = 0;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), d);
        if (r.ec == std::errc::result_out_of_range) {
            // overflow -> +-inf like Python float(); underflow -> 0
            d = std::strtod(std::string(s).c_str(), nullptr);
        } else if (r.ec != std::errc()) {
            fail("bad number");
        }
        return Value(d);
    }

    std::string_view t_;
    std::size_t p_ = 0;
    std::size_t max_depth_;
};

}  // namespace

Value parse(std::string_view text, std::size_t max_depth) { return Parser(text, max_depth).document(); }

// ---------------------------------------------------------------------------
// serializer

void appendEscaped(std::string& out, std::string_view s) {
    static const char* hex = "0123456789abcdef";
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case 0x08: out += "\\b"; break;
            case 0x0C: out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 15]);
                } else {
                    out.push_back(ch);
                }
        }
    }
}

std::string pythonFloat(double d) {
    if (std::isnan(d)) return "NaN";
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
    // shortest round-trip digits and decimal exponent
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    const bool neg = sci[0] == '-';
    if (neg) sci.erase(0, 1);
    const std::size_t epos = sci.find('e');
    std::string digits = sci.substr(0, epos);
    const int exp10 = std::atoi(sci.c_str() + epos + 1);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    // Python repr: positional for 1e-4 <= |d| < 1e16, otherwise scientific
    std::string out = neg ? "-" : "";
    const int nd = static_cast<int>(digits.size());
    if (exp10 >= -4 && exp10 < 16) {
        if (exp10 >= 0) {
            if (nd <= exp10 + 1) {
                out += digits + std::string(static_cast<std::size_t>(exp10 + 1 - nd), '0') + ".0";
            } else {
                out += digits.substr(0, static_cast<std::size_t>(exp10 + 1)) + "." +
                       digits.substr(static_cast<std::size_t>(exp10 + 1));
            }
        } else {
            out += "0." + std::string(static_cast<std::size_t>(-exp10 - 1), '0') + digits;
        }
    } else {
        out += digits.substr(0, 1);
        if (nd > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
        out += eb;
    }
    return out;
}

void dumpPython(std::string& out, const Value& v, bool compact) {
    switch (v.kind()) {
        case Value::Kind::null: out += "null"; break;
        case Value::Kind::boolean: out += v.asBool() ? "true" : "false"; break;
        case Value::Kind::integer: out += std::to_string(v.asInt()); break;
        case Value::Kind::big_integer: out += v.asString(); break;
        case Value::Kind::number: out += pythonFloat(v.asDouble()); break;
        case Value::Kind::string:
            out.push_back('"');
            appendEscaped(out, v.asString());
            out.push_back('"');
            break;
        case Value::Kind::array: {
            out.push_back('[');
            bool first = true;
            for (const auto& it : v.asArray()) {
                if (!first) out += compact ? "," : ", ";
                first = false;
                dumpPython(out, it, compact);
            }
            out.push_back(']');
            break;
        }
        case Value::Kind::object: {
            out.push_back('{');
            bool first = true;
            for (const auto& m : v.asObject().members()) {
                if (!first) out += compact ? "," : ", ";
                first = false;
                out.push_back('"');
                appendEscaped(out, m.first);
                out += compact ? "\":" : "\": ";
                dumpPython(out, m.second, compact);
            }
            out.push_back('}');
            break;
        }
    }
}

std::string dumpPython(const Value& v, bool compact) {
    std::string out;
    dumpPython(out, v, compact);
    return out;
}

}  // namespace whirl::json
