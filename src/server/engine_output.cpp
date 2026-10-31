// Server engine, a slot's output: emit, flushOut, writeError (arch025-engine S5, module server.slot_output; moved
// verbatim out of engine.cpp:223-246 and :596-603 at 6d004b6: only the definition lines changed, `Engine::f(` -> `f(`,
// and emit takes the chat tokens as its first parameter `const chat::ChatTokens& ct_`). Declarations: engine_output.h.
// SPDX-License-Identifier: Apache-2.0

#include "server/engine_output.h"

#include "server/log.h"       // logW
#include "server/protocol.h"  // errorResponse

#include <span>

namespace whirl::server {

bool emit(const chat::ChatTokens& ct_, Slot& sl, std::uint32_t t) {
    if (!sl.ignore_eos && (t == ct_.im_end || t == ct_.endoftext)) {
        sl.finish = "stop";
        return true;
    }
    sl.batch[sl.n_batch] = t;
    sl.n_batch += 1;
    sl.n_gen += 1;
    if (sl.n_gen >= sl.max_tokens) {
        sl.finish = "length";
        return true;
    }
    return false;
}

void flushOut(Slot& sl) {
    std::uint32_t n_after = 0;
    if (sl.st.push(sl.out, std::span<const std::uint32_t>(sl.batch.data(), sl.n_batch), &n_after)) {
        sl.done = true;
        sl.stopped_str = true;
        sl.n_after_stop = n_after;
    }
    sl.n_batch = 0;
}

void writeError(Job& job, int status, const std::string& msg) {
    logW("req {} | {} {}", job.id, status, msg);
    const std::string r = errorResponse(status, msg);
    if (job.w) {
        job.w->write(r);
        job.w->flush();
    }
}

}  // namespace whirl::server
