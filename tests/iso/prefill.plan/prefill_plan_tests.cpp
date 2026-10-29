// whirl-iso-prefill-plan-tests: the module prefill.plan (include/whirl/prefill_schedule.h, include/whirl/chat_tokens.h, src/chat/chat_tokens.cpp)
// built with its declared dependencies (the tokenizer and the GGUF reader) and run (ARCH-1 8.5 isolation, arch025-iso4). No unit test file
// existed (the cases lived in tests/server/server_tests.cpp behind a TestServer, and in tests/unit/mem_fit_tests.cpp among other modules). The
// checks pin the cold-prefill schedule (chunks, merged tail, forced splits, checkpoints, forward ends), the split rules as functions of the
// tokens, and the chat token facts of a vocabulary (ids by name, the CR LF -> LF table).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/chat_tokens.h"
#include "whirl/device_profile.h"
#include "whirl/gguf.h"
#include "whirl/prefill_schedule.h"
#include "whirl/tokenizer.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace s = whirl::sched;
using Ids = std::vector<std::uint32_t>;
using Sizes = std::vector<std::size_t>;
const std::span<const std::uint32_t> k_noimg;

void testChunks() {
    ISO_CHECK(s::prefillChunkLen(0) == 0 && s::prefillChunkLen(100) == 100 && s::prefillChunkLen(1280) == 1280);
    ISO_CHECK(s::prefillChunkLen(1281) == 1024);  // a remainder over chunk + merge starts a new chunk
    ISO_CHECK(s::schedNext(3000, 500, 2999, k_noimg, 0) == 500 && s::schedNext(3000, 500, 2999, k_noimg, 500) == 2999 &&
              s::schedNext(3000, 500, 2999, k_noimg, 2999) == 3000);
    const Ids imgs = {700, 900};
    ISO_CHECK(s::schedNext(3000, 0, 0, imgs, 0) == 700 && s::schedNext(3000, 0, 0, imgs, 700) == 900 && s::schedNext(3000, 0, 0, imgs, 900) == 3000);
    ISO_CHECK(s::schedNext(3000, 800, 0, imgs, 0) == 700);  // the first image end before the system split wins
}

void testSchedulePoints() {
    ISO_CHECK(s::schedFloor(3000, 0, 0, k_noimg, 2000) == 1024 && s::schedFloor(3000, 0, 0, k_noimg, 1023) == 0 &&
              s::schedFloor(3000, 0, 0, k_noimg, 2048) == 2048);
    ISO_CHECK(s::onSchedule(3000, 0, 0, k_noimg, 0) && s::onSchedule(3000, 0, 0, k_noimg, 1024) && !s::onSchedule(3000, 0, 0, k_noimg, 1000));
    ISO_CHECK(s::tailPoint(3000, 0, 0, k_noimg) == 2048 && s::tailPoint(0, 0, 0, k_noimg) == 0 && s::tailPoint(1000, 0, 0, k_noimg) == 0);
    // a forced split restarts the chunking there
    ISO_CHECK(s::onSchedule(3000, 500, 0, k_noimg, 500) && s::onSchedule(3000, 500, 0, k_noimg, 1524) && !s::onSchedule(3000, 500, 0, k_noimg, 1024));
}

void testForwards() {
    const Sizes none;
    ISO_CHECK(s::execLen(3000, 0, 0, k_noimg, 0, 2048, none) == 2048);
    ISO_CHECK(s::execLen(3000, 0, 0, k_noimg, 0, 1024, none) == 1024);  // exec <= one chunk: one chunk per forward
    ISO_CHECK(s::forwardEnds(3000, 0, 0, k_noimg, 0, 2048, none) == (Sizes{2048, 3000}));
    ISO_CHECK(s::forwardEnds(3000, 0, 0, k_noimg, 0, 1024, none) == (Sizes{1024, 2048, 3000}));
    const Sizes stops = {1024};  // a kept checkpoint is a forward end
    ISO_CHECK(s::forwardEnds(3000, 0, 0, k_noimg, 0, 2048, stops) == (Sizes{1024, 3000}));
    ISO_CHECK(s::forwardEnds(3000, 0, 0, k_noimg, 1024, 2048, none) == (Sizes{3000}));  // resumed from a schedule point
    ISO_CHECK(s::forwardEnds(0, 0, 0, k_noimg, 0, 2048, none).empty());
    ISO_CHECK(s::execLen(3000, 0, 0, k_noimg, 0, 2048, none, 1024) == 1024);  // max_rows caps the forward
    // the solo prefill (CLI-SCHED): the TAIL-CK stop is a forward end for a prompt of >= tail_min tokens
    ISO_CHECK(s::soloForwardEnds(5000, 0, 0, k_noimg, 0, 2048, 4096) == (Sizes{2048, 4096, 5000}));
    ISO_CHECK(s::soloForwardEnds(5000, 0, 0, k_noimg, 0, 2048, 0) == s::forwardEnds(5000, 0, 0, k_noimg, 0, 2048, none));
}

void testExecSize() {
    ISO_CHECK(s::prefillExecFor(2048, 4096) == 2048);
    ISO_CHECK(s::prefillExecFor(8192, 4096) == 3072);              // a forward + the merged tail must fit the batch
    ISO_CHECK(s::prefillExecFor(8192, 4096, true) == 4096);        // fill: whole chunks up to the batch itself
    ISO_CHECK(s::prefillExecFor(100, 4096) == 1024);               // never below one chunk
    namespace d = whirl::device;
    const d::Profile& r9700 = d::profile(whirl::hip::Arch::gfx1201);
    const d::Profile& strix = d::profile(whirl::hip::Arch::gfx1151);
    ISO_CHECK(s::prefillExecDefault(r9700, true) == 2048 && s::prefillExecDefault(strix, true) == 4096 && s::prefillExecDefault(strix, false) == 2048);
    ISO_CHECK(s::prefillExec(std::nullopt, strix, true, 4096) == 4096 && s::prefillExec(std::nullopt, strix, false, 4096) == 2048);
    ISO_CHECK(s::prefillExec(std::size_t(1024), strix, true, 4096) == 1024 && s::prefillExec(std::nullopt, r9700, true, 4096) == 2048);
}

void testSplitsFromTokens() {
    const Ids a = {1, 2, 9, 9, 3, 4, 1, 7, 3, 4, 1};  // <start> system x x <end> \n <start> ...
    ISO_CHECK(s::sysBoundary(a, 1, 2, 3, 4) == 6);
    const Ids b = {1, 2, 3, 9, 3, 4, 1, 7};  // an <end> the system text mentions is not the end
    ISO_CHECK(s::sysBoundary(b, 1, 2, 3, 4) == 6);
    ISO_CHECK(s::sysBoundary(Ids{1, 5, 3, 4, 1}, 1, 2, 3, 4) == 0 && s::sysBoundary(Ids{1, 2}, 1, 2, 3, 4) == 0);
    ISO_CHECK(s::sysBoundary(a, 0xffffffffu, 2, 3, 4) == 0);  // a none id never matches
    ISO_CHECK(s::sysSplit(6, 5000, s::sys_min_default) == 0 && s::sysSplit(3000, 5000, s::sys_min_default) == 3000);
    ISO_CHECK(s::sysSplit(3000, 3000, s::sys_min_default) == 0 && s::sysSplit(3000, 5000, 0) == 0);
    const Ids t = {1, 2, 100, 101, 102, 50, 5, 6, 100, 101, 50, 7, 8, 9};
    ISO_CHECK(s::imageSplits(t, 100, 50, 3) == (Ids{6, 11}) && s::imageSplits(t, 100, 50, 8) == (Ids{11}) && s::imageSplits(t, 100, 50, 0).empty());
    ISO_CHECK(s::imageSplits(Ids{1, 100, 101, 50}, 100, 50, 1).empty());  // an image end at the end of the prompt is no split
}

// a minimal gpt2-style tokenizer (no merges): the chat control tokens, "\n" (U+010A) and "\r\n" (U+010D U+010A) as byte-level pieces
std::vector<std::uint8_t> chatVocabGguf() {
    std::vector<std::uint8_t> b;
    const auto put = [&](auto v) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof v);
    };
    const auto str = [&](std::string_view x) {
        put(static_cast<std::uint64_t>(x.size()));
        b.insert(b.end(), x.begin(), x.end());
    };
    const std::vector<std::pair<std::string, std::int32_t>> v = {
        {"<|im_start|>", 3}, {"<|im_end|>", 3}, {"<|endoftext|>", 3}, {"<think>", 3}, {"<|vision_end|>", 3},
        {"\xC4\x8A", 1},          // "\n"
        {"\xC4\x8D\xC4\x8A", 1},  // "\r\n"
        {"s", 1}, {"y", 1}, {"t", 1}, {"e", 1}, {"m", 1}};
    put(std::uint32_t(0x46554747));
    put(std::uint32_t(3));
    put(std::uint64_t(0));
    put(std::uint64_t(4));
    const auto key = [&](std::string_view k, whirl::gguf::ValueType t) {
        str(k);
        put(static_cast<std::uint32_t>(t));
    };
    key("tokenizer.ggml.model", whirl::gguf::ValueType::string), str("gpt2");
    key("tokenizer.ggml.pre", whirl::gguf::ValueType::string), str("qwen35");
    key("tokenizer.ggml.tokens", whirl::gguf::ValueType::array), put(std::uint32_t(8)), put(static_cast<std::uint64_t>(v.size()));
    for (const auto& e : v) str(e.first);
    key("tokenizer.ggml.token_type", whirl::gguf::ValueType::array), put(std::uint32_t(5)), put(static_cast<std::uint64_t>(v.size()));
    for (const auto& e : v) put(e.second);
    while (b.size() % 32) b.push_back(0);
    return b;
}

void testChatTokenFacts() {
    namespace c = whirl::chat;
    const auto bytes = chatVocabGguf();
    const whirl::gguf::File f = whirl::gguf::File::parse(bytes);
    const whirl::Tokenizer tok = whirl::Tokenizer::fromGguf(f);
    const c::ChatTokens ct(tok);
    ISO_CHECK(ct.im_start == 0 && ct.im_end == 1 && ct.endoftext == 2 && ct.think == 3 && ct.vision_end == 4);
    ISO_CHECK(ct.nl == 5);                       // "\n" is one token
    ISO_CHECK(ct.system == c::ChatTokens::none);  // "system" is six tokens in this vocabulary (no merges)
    const std::vector<std::uint32_t> norm = c::ChatTokens::crlfToLf(tok);
    ISO_CHECK(norm.size() == tok.size() && norm[6] == 5 && norm[5] == 5 && norm[0] == 0);  // "\r\n" -> "\n"; control tokens never map
    const c::ChatTokens empty;  // a default ChatTokens knows nothing: no split
    ISO_CHECK(empty.thinkOpenSplit(Ids{1, 2, 3}) == 0 && empty.sysBoundary(Ids{1, 2, 3}) == 0 && empty.nextTurn(Ids{1, 2, 3, 4}, 0) == 0);
}

void testChatTokenSplits() {
    namespace c = whirl::chat;
    c::ChatTokens ct;
    ct.im_start = 10, ct.im_end = 11, ct.think = 12, ct.nl = 13, ct.system = 14;
    ISO_CHECK(ct.thinkOpenSplit(Ids{1, 2, 12, 13}) == 3);  // the prefill stops before the last "\n"
    ISO_CHECK(ct.thinkOpenSplit(Ids{12, 13, 5}) == 0 && ct.thinkOpenSplit(Ids{13}) == 0 && ct.thinkOpenSplit(Ids{}) == 0);
    const Ids p = {10, 14, 5, 5, 11, 13, 10, 6, 5, 11, 13, 10, 7, 5, 11, 13, 10};  // system, user, assistant turns
    ISO_CHECK(ct.sysBoundary(p) == 6);
    ISO_CHECK(ct.nextTurn(p, 0) == 6 && ct.nextTurn(p, 6) == 11 && ct.nextTurn(p, 11) == 16 && ct.nextTurn(p, 16) == 0);
    ISO_CHECK(ct.sysBoundary(Ids{10, 6, 5, 11, 13, 10}) == 0);  // no system message
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("prefill.plan",
                    {{"chunks", testChunks},
                     {"schedule_points", testSchedulePoints},
                     {"forwards", testForwards},
                     {"exec_size", testExecSize},
                     {"splits_from_tokens", testSplitsFromTokens},
                     {"chat_token_facts", testChatTokenFacts},
                     {"chat_token_splits", testChatTokenSplits}},
                    argc, argv);
}
