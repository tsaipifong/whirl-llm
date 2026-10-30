// whirl-iso-server-slot-output-tests: the module server.slot_output (src/server/engine_output.{h,cpp}: emit, flushOut,
// writeError; arch025-engine S5, the bodies moved verbatim out of engine.cpp:223-246 / 596-603 at 6d004b6, member
// functions of Engine -> free functions; emit takes the chat tokens ct_, the one core.shared member it reads, as its
// first parameter under the Engine name) built with only its declared dependencies (ARCH-1 8.5 isolation). The checks
// pin what the move must keep, with expected values written out: emit's end-of-turn / end-of-text stop (and
// ignore_eos), the batch append and the max_tokens limit; flushOut's call of the slot's response writer with exactly
// the batch, the stop-string end (done, stopped_str, n_after_stop) and the batch reset; writeError's answer
// (errorResponse text, then one flush) and a job without a connection.
// StreamState::push is the response writer of module protocol (src/server/stream.cpp, which needs a GGUF tokenizer): this
// program defines it as a recording double (link seam), so the test sees exactly what flushOut hands to it.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_output.h"

#include "server/protocol.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace s = whirl::server;

// ---- the response writer double: StreamState::push records its arguments and answers what the test set
namespace {
struct PushRec {
    int calls = 0;
    const s::Out* out = nullptr;
    const std::uint32_t* data = nullptr;
    std::vector<std::uint32_t> toks;
    bool had_n_after = false;
    bool ret = false;          // what push answers (true = a stop string matched)
    std::uint32_t n_after = 0; // written through n_after when ret
};
PushRec g_push;
}  // namespace

bool whirl::server::StreamState::push(Out& out, std::span<const std::uint32_t> toks, std::uint32_t* n_after) {
    g_push.calls += 1;
    g_push.out = &out;
    g_push.data = toks.data();
    g_push.toks.assign(toks.begin(), toks.end());
    g_push.had_n_after = n_after != nullptr;
    if (g_push.ret && n_after) *n_after = g_push.n_after;
    return g_push.ret;
}

namespace {

// the signatures of DESIGN v3 2.3 / 3.1: free functions, ct_ is a parameter (const: emit only reads it)
static_assert(std::is_same_v<decltype(&s::emit), bool (*)(const whirl::chat::ChatTokens&, s::Slot&, std::uint32_t)>);
static_assert(std::is_same_v<decltype(&s::flushOut), void (*)(s::Slot&)>);
static_assert(std::is_same_v<decltype(&s::writeError), void (*)(s::Job&, int, const std::string&)>);

struct FakeConn : s::Conn {
    std::string data;
    std::vector<std::string> log;  // "w" / "f" in call order
    bool write(std::string_view d) override {
        data += d;
        log.push_back("w");
        return true;
    }
    bool flush() override {
        log.push_back("f");
        return true;
    }
};

whirl::chat::ChatTokens tokens() {
    whirl::chat::ChatTokens ct;
    ct.im_end = 151645;
    ct.endoftext = 151643;
    return ct;
}

void testEmitStop() {
    const auto ct = tokens();
    s::Slot sl;
    sl.max_tokens = 8;
    sl.finish = "length";
    ISO_CHECK(s::emit(ct, sl, 151645));  // <|im_end|> ends the request: not added, not counted
    ISO_CHECK(sl.finish == "stop" && sl.n_batch == 0 && sl.n_gen == 0);
    s::Slot s2;
    s2.max_tokens = 8;
    ISO_CHECK(s::emit(ct, s2, 151643));  // <|endoftext|> as well
    ISO_CHECK(s2.finish == "stop" && s2.n_batch == 0 && s2.n_gen == 0);
    // ignore_eos: both are ordinary tokens
    s::Slot s3;
    s3.max_tokens = 8;
    s3.ignore_eos = true;
    ISO_CHECK(!s::emit(ct, s3, 151645));
    ISO_CHECK(!s::emit(ct, s3, 151643));
    ISO_CHECK(s3.finish == "length" && s3.n_batch == 2 && s3.n_gen == 2 && s3.batch[0] == 151645 && s3.batch[1] == 151643);
    // the default ChatTokens (none = 0xffffffff): no id stops
    const whirl::chat::ChatTokens none;
    s::Slot s4;
    s4.max_tokens = 8;
    ISO_CHECK(!s::emit(none, s4, 151645) && s4.n_batch == 1 && s4.n_gen == 1);
}

void testEmitAppendAndLength() {
    const auto ct = tokens();
    s::Slot sl;
    sl.max_tokens = 3;
    sl.n_gen = 0;
    ISO_CHECK(!s::emit(ct, sl, 7));
    ISO_CHECK(sl.n_batch == 1 && sl.n_gen == 1 && sl.batch[0] == 7 && sl.finish == "length");
    ISO_CHECK(!s::emit(ct, sl, 9));
    ISO_CHECK(sl.n_batch == 2 && sl.n_gen == 2 && sl.batch[1] == 9);
    sl.finish = "x";
    ISO_CHECK(s::emit(ct, sl, 11));  // the third token reaches max_tokens: added, counted, finish "length"
    ISO_CHECK(sl.n_batch == 3 && sl.n_gen == 3 && sl.batch[2] == 11 && sl.finish == "length");
    // n_gen counts the whole request, n_batch only this batch (flushOut resets it)
    s::Slot s2;
    s2.max_tokens = 10;
    s2.n_gen = 9;
    s2.n_batch = 4;
    ISO_CHECK(s::emit(ct, s2, 5));
    ISO_CHECK(s2.n_batch == 5 && s2.batch[4] == 5 && s2.n_gen == 10 && s2.finish == "length");
    // a stop token at the limit is still a stop (checked first)
    s::Slot s3;
    s3.max_tokens = 1;
    ISO_CHECK(s::emit(ct, s3, 151645) && s3.finish == "stop" && s3.n_gen == 0);
}

void testFlushOut() {
    s::Slot sl;
    sl.batch[0] = 4;
    sl.batch[1] = 5;
    sl.batch[2] = 6;
    sl.n_batch = 3;
    g_push = {};
    g_push.ret = false;
    s::flushOut(sl);
    ISO_CHECK(g_push.calls == 1 && g_push.out == &sl.out && g_push.data == sl.batch.data() && g_push.had_n_after);
    ISO_CHECK((g_push.toks == std::vector<std::uint32_t>{4, 5, 6}));
    ISO_CHECK(sl.n_batch == 0 && !sl.done && !sl.stopped_str && sl.n_after_stop == 0);
    // a stop string matched: the request ends, the tokens after it are reported
    sl.batch[0] = 8;
    sl.batch[1] = 9;
    sl.n_batch = 2;
    g_push.ret = true;
    g_push.n_after = 1;
    s::flushOut(sl);
    ISO_CHECK(g_push.calls == 2 && (g_push.toks == std::vector<std::uint32_t>{8, 9}));
    ISO_CHECK(sl.n_batch == 0 && sl.done && sl.stopped_str && sl.n_after_stop == 1);
    // an empty batch still goes through the writer (the caller decides when to flush)
    s::Slot s2;
    s2.n_after_stop = 7;
    g_push = {};
    s::flushOut(s2);
    ISO_CHECK(g_push.calls == 1 && g_push.toks.empty() && s2.n_batch == 0 && !s2.done && s2.n_after_stop == 7);
    // n_after 0 with a stop: n_after_stop is set to 0
    s::Slot s3;
    s3.n_batch = 1;
    s3.n_after_stop = 5;
    g_push = {};
    g_push.ret = true;
    g_push.n_after = 0;
    s::flushOut(s3);
    ISO_CHECK(s3.done && s3.stopped_str && s3.n_after_stop == 0 && s3.n_batch == 0);
}

void testWriteError() {
    FakeConn c;
    s::Job job;
    job.id = 42;
    job.w = &c;
    s::writeError(job, 400, "prompt is too long for the context size");
    ISO_CHECK(c.data == s::errorResponse(400, "prompt is too long for the context size"));
    ISO_CHECK((c.log == std::vector<std::string>{"w", "f"}));
    ISO_CHECK(c.data.rfind("HTTP/1.1 400 ", 0) == 0 &&
              c.data.find("\"prompt is too long for the context size\"") != std::string::npos &&
              c.data.find("\"invalid_request_error\"") != std::string::npos);
    FakeConn c5;
    job.w = &c5;
    s::writeError(job, 503, "busy");
    ISO_CHECK(c5.data == s::errorResponse(503, "busy") && c5.data.find("\"server_error\"") != std::string::npos);
    ISO_CHECK((c5.log == std::vector<std::string>{"w", "f"}));
    // no connection: nothing written, no crash (the log line only)
    s::Job none;
    none.id = 43;
    s::writeError(none, 400, "x");
    ISO_CHECK(none.w == nullptr && c.log.size() == 2 && c5.log.size() == 2);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.slot_output",
                    {{"emit_stop", testEmitStop},
                     {"emit_append_length", testEmitAppendAndLength},
                     {"flush_out", testFlushOut},
                     {"write_error", testWriteError}},
                    argc, argv);
}
