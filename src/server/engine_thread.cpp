// Thread ownership of the engine state (arch024 M8): the violation path (log once per call site, count, throw in
// throwing mode). See engine_thread.h.
// SPDX-License-Identifier: Apache-2.0

#include "server/engine_thread.h"

#include "server/log.h"
#include "whirl/thread_annotations.h"

#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace whirl::server {

namespace {
std::atomic<bool> g_throw{false};
std::atomic<std::uint64_t> g_violations{0};
whirl::Mutex g_mu;
std::set<std::string> g_logged WH_GUARDED_BY(g_mu);  // call sites already logged
}  // namespace

void EngineThread::setThrowing(bool on) { g_throw.store(on, std::memory_order_relaxed); }
bool EngineThread::throwing() { return g_throw.load(std::memory_order_relaxed); }
std::uint64_t EngineThread::violations() { return g_violations.load(std::memory_order_relaxed); }

void EngineThread::resetLoggedForTests() {
    whirl::MutexLock lk(g_mu);
    g_logged.clear();
}

void EngineThread::violation(const char* func) {
    g_violations.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream id;
    id << std::this_thread::get_id();
    const std::string msg = std::string("owner check: ") + func + " called on thread " + id.str();
    bool first = false;
    {
        whirl::MutexLock lk(g_mu);
        first = g_logged.insert(func).second;
    }
    if (first) Log::line('E', msg);
    if (throwing()) throw std::logic_error(msg);
}

}  // namespace whirl::server
