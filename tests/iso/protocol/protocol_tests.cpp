// whirl-iso-protocol-tests: the HTTP / OpenAI protocol / streaming module (src/server/protocol.cpp, stream.cpp, http.cpp,
// base64.cpp) built alone (ARCH-1 8.5, arch025-lane2 leg 5 S6) with its declared dependencies: the chat template and JSON
// (src/chat), the tokenizer and the GGUF reader it loads from, the vision prompt side (src/vision/prompt.cpp + hash.cpp:
// placeholder expansion, no encoder), the server log, the platform layer (sockets, time, env) and lib whirl's util. What
// the module sees of the engine is engine_api.h (EngineApi, Job, Conn) and of vision only vision_iface.h's names: no
// engine, no ServerModel, no model, no HIP layer, no image encoder.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/http.h"
#include "server/protocol.h"
#include "server/vision_iface.h"
#include "whirl/json.h"
#include "whirl/vision.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace sv = whirl::server;
namespace vi = whirl::vision;

void testTextHelpers() {
    ISO_CHECK(sv::trimWs("  a b \t") == "a b");
    ISO_CHECK(sv::trimLeftWs("  a ") == "a ");
    ISO_CHECK(sv::trimRightWs("  a ") == "  a");
    // the held-back tail of a streamed text that may still become a stop string
    ISO_CHECK(sv::partialHold("hello </", "</think>") == 2);
    ISO_CHECK(sv::partialHold("hello", "</think>") == 0);
    // UTF-8: a cut multi-byte sequence at the end is not part of the valid prefix
    const std::string cut = std::string("ab") + "\xE4\xB8";
    ISO_CHECK(sv::validUtf8Prefix(cut) == 2);
    std::string js;
    sv::appendJsonStr(js, "a\"b\\\n");
    ISO_CHECK(js == "\"a\\\"b\\\\\\n\"");
}

void testParams() {
    const whirl::json::Value v = whirl::json::parse(R"({"temperature": 0.25, "top_k": 5, "max_tokens": 7, "stop": ["x", "yz"], "stream": true})");
    const sv::Params p = sv::parseParams(v.asObject(), true, true);
    ISO_CHECK(p.temperature == 0.25f && p.top_k == 5 && p.max_tokens && *p.max_tokens == 7 && p.stream);
    ISO_CHECK(p.stop.size() == 2 && p.stop[0] == "x" && p.stop[1] == "yz");
    // non-thinking chat: the model-card defaults (temperature 0.7, top_p 0.8)
    const whirl::json::Value e = whirl::json::parse("{}");
    const sv::Params d = sv::parseParams(e.asObject(), false, true);
    ISO_CHECK(d.temperature == 0.7f && d.top_p == 0.8f);
    ISO_THROWS(sv::parseParams(whirl::json::parse(R"({"top_p": "x"})").asObject(), true, true), whirl::chat::RequestError);
    ISO_THROWS(sv::parseParams(whirl::json::parse(R"({"top_p": 1.5})").asObject(), true, true), whirl::chat::RequestError);
}

void testResponses() {
    const std::string r = sv::httpResponse(200, "application/json", "{}");
    ISO_CHECK(r.rfind("HTTP/1.1 200 OK\r\n", 0) == 0 && r.find("Content-Length: 2\r\n") != std::string::npos &&
              r.size() >= 2 && r.substr(r.size() - 2) == "{}");
    ISO_CHECK(std::string(sv::statusText(404)) == "Not Found");
    ISO_CHECK(sv::errorResponse(400, "bad").find("\"bad\"") != std::string::npos);
    ISO_CHECK(sv::isLoopbackOrigin("http://localhost:8080") && sv::isLoopbackOrigin("http://127.0.0.1") &&
              !sv::isLoopbackOrigin("http://example.com"));
    ISO_CHECK(sv::hostAllowed("localhost:8080", {}, "127.0.0.1") && !sv::hostAllowed("evil.example", {}, "127.0.0.1"));
    ISO_CHECK(sv::corsHeaders("", {}).empty());
}

void testBase64() {
    std::vector<std::uint8_t> o;
    ISO_CHECK(sv::base64Decode("aGVsbG8=", o) && std::string(o.begin(), o.end()) == "hello");
    ISO_CHECK(sv::base64Decode("aGVsbG8", o) && std::string(o.begin(), o.end()) == "hello");
    ISO_CHECK(sv::base64Decode("", o) && o.empty());
    ISO_CHECK(!sv::base64Decode("aGVsbG9", o));  // non-zero leftover bits
    ISO_CHECK(!sv::base64Decode("aGVsbG8*", o));
}

void testExpand() {
    // the prompt side of an image request (buildVis): one placeholder -> the image's nx * ny content token ids
    vi::Prepared p;
    p.nx = 2;
    p.ny = 3;
    p.hash[0] = 7;
    const std::vector<std::uint32_t> toks = {10, 99, 11};
    const vi::Prepared* imgs[] = {&p};
    std::array<whirl::qwen35::VisSpan, 1> spans{};
    const std::vector<std::uint32_t> out = vi::expand(toks, 99, imgs, spans);
    ISO_CHECK(out.size() == 2 + 6 && out.front() == 10 && out.back() == 11);
    ISO_CHECK(spans[0].start == 1 && spans[0].n == 6);
    for (std::uint32_t i = 0; i < 6; ++i) ISO_CHECK(out[1 + i] == vi::tokenId(p.hash, i) && out[1 + i] >= whirl::qwen35::image_id_base);
    // a placeholder without an image
    std::array<whirl::qwen35::VisSpan, 1> s2{};
    ISO_THROWS(vi::expand(std::vector<std::uint32_t>{99, 99}, 99, imgs, s2), vi::VisionError);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("protocol",
                    {{"text_helpers", testTextHelpers},
                     {"params", testParams},
                     {"responses", testResponses},
                     {"base64", testBase64},
                     {"vision_expand", testExpand}},
                    argc, argv);
}
