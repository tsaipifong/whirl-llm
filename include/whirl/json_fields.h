// JSON request field helpers shared by the chat template rendering (src/chat/chat_template.cpp) and whirl-server's
// request parsing (src/server/protocol.cpp) (arch025-N6, lane 1 leg 7: both files wrote the same functions out; the shape
// of llama.cpp tools/server/server-common.h json_value, one helper for every request parser). A field is a member that
// is present and not JSON null; a field of the wrong type is a RequestError (the HTTP 400 text). The json::Value
// overloads equal the json::Object ones on an object and give nullptr on anything else (Value::get).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "whirl/chat.h"
#include "whirl/json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace whirl::chat {

// the JSON / Python whitespace set both files trimmed with
inline constexpr std::string_view k_json_ws = " \t\r\n\x0b\x0c";

inline std::string_view trimLeftWs(std::string_view s) {
    const std::size_t a = s.find_first_not_of(k_json_ws);
    return a == std::string_view::npos ? std::string_view() : s.substr(a);
}

inline std::string_view trimRightWs(std::string_view s) {
    const std::size_t b = s.find_last_not_of(k_json_ws);
    return b == std::string_view::npos ? std::string_view() : s.substr(0, b + 1);
}

inline std::string_view trimWs(std::string_view s) { return trimRightWs(trimLeftWs(s)); }

// member that is present and not JSON null
inline const json::Value* getField(const json::Object& obj, std::string_view key) {
    const json::Value* v = obj.find(key);
    if (!v || v->isNull()) return nullptr;
    return v;
}

inline const json::Value* getField(const json::Value& obj, std::string_view key) {
    const json::Value* v = obj.get(key);
    return (v && !v->isNull()) ? v : nullptr;
}

inline const std::string* getStr(const json::Object& obj, std::string_view key) {
    const json::Value* v = getField(obj, key);
    if (!v || !v->isString()) return nullptr;
    return &v->asString();
}

inline const std::string* getStr(const json::Value& obj, std::string_view key) {
    const json::Value* v = getField(obj, key);
    return (v && v->isString()) ? &v->asString() : nullptr;
}

// nullopt when absent / null; throws RequestError(msg) when not a boolean
inline std::optional<bool> getBool(const json::Object& obj, std::string_view key, const char* msg) {
    const json::Value* v = getField(obj, key);
    if (!v) return std::nullopt;
    if (!v->isBool()) throw RequestError(msg);
    return v->asBool();
}

inline std::optional<bool> getBool(const json::Value& obj, std::string_view key, const char* msg) {
    const json::Value* v = getField(obj, key);
    if (!v) return std::nullopt;
    if (!v->isBool()) throw RequestError(msg);
    return v->asBool();
}

// number field: nullopt when absent / null; throws RequestError(msg) when not a number
inline std::optional<double> getNum(const json::Object& obj, std::string_view key, const char* msg) {
    const json::Value* v = getField(obj, key);
    if (!v) return std::nullopt;
    switch (v->kind()) {
        case json::Value::Kind::integer: return static_cast<double>(v->asInt());
        case json::Value::Kind::number: return v->asDouble();
        case json::Value::Kind::big_integer: return v->asDouble();
        default: throw RequestError(msg);
    }
}

}  // namespace whirl::chat
