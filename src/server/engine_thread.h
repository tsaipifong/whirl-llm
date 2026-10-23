// Thread ownership of the engine state (arch024 M8): which thread may call the mutating entry points of the engine's
// modules (CkptStore, PagePool, PrefixStore, TierBridge, the memory governor glue). The table of owners is
// docs/dev/threads.md.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Prior art: progress/arch024-S10.PRIOR_ART.md (llama.cpp server_queue: one loop thread, no
// runtime check; Chromium THREAD_CHECKER / DETACH_FROM_THREAD: bind to a thread, compiled out in release).
//
// The owner is the thread inside Engine::runLoop. Before runLoop (startup: allocState, initPool, attachTier ...) and
// after it returned there is no owner and every thread may call (nothing runs the loop at the same time). Other
// threads reach the engine through a Job (submitAndWait) or, in tests, runOnLoop, both of which run on the owner.
//
// A call from another thread while the loop runs is a violation: it is counted, and
//   - release (the default): one line "E owner check: <func> called on thread <id>" per call site (the first time),
//     the call goes on as before (no new 500 or crash on an existing path; the gates scan the server logs for it,
//     tools/gates/server_log_scan.ps1);
//   - throwing mode (setThrowing(true): the test programs, PLAN M8 "test build"): std::logic_error after counting.
// Cost on the owner: one relaxed atomic load, and std::this_thread::get_id() while the loop runs. The checks sit at
// module entry points, not in the per-token path (the decode cycle calls ServerModel directly).

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

namespace whirl::server {

class EngineThread {
public:
    // runLoop: the calling thread owns the engine state until leave()
    void enter() { id_.store(std::this_thread::get_id(), std::memory_order_relaxed); }
    void leave() { id_.store(std::thread::id(), std::memory_order_relaxed); }
    bool running() const { return id_.load(std::memory_order_relaxed) != std::thread::id(); }
    std::thread::id owner() const { return id_.load(std::memory_order_relaxed); }

    // func: a string literal naming the entry point ("PagePool::take"); one per call site
    void check(const char* func) const {
        const std::thread::id o = id_.load(std::memory_order_relaxed);
        if (o != std::thread::id() && o != std::this_thread::get_id()) violation(func);
    }

    // process-wide policy and counter (all engines)
    static void setThrowing(bool on);
    static bool throwing();
    static std::uint64_t violations();
    // tests: the call sites logged so far are forgotten (the next violation of each logs again; the counter stays)
    static void resetLoggedForTests();

private:
    static void violation(const char* func);
    std::atomic<std::thread::id> id_{};
};

// a module's view of the owner (nullptr: a module used on its own, e.g. a unit test; nothing is checked)
inline void ownerCheck(const EngineThread* o, const char* func) {
    if (o) o->check(func);
}

}  // namespace whirl::server
