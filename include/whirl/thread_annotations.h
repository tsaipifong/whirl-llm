// Clang thread-safety annotations (arch025-Q4, release_gate step cpu_tsa).
// SPDX-License-Identifier: Apache-2.0
//
// The product is built with MSVC: there every macro below is empty and Mutex / MutexLock / UniqueLock are plain aliases of
// std::mutex / std::lock_guard<std::mutex> / std::unique_lock<std::mutex>, so the product code is the same code as before.
//
// The cpu_tsa build (clang-cl -Wthread-safety -Werror=thread-safety, tools/safety/tsa/run_tsa_build.bat) compiles the same
// sources with clang. The MSVC STL's std::mutex has no capability attribute, so under clang Mutex is a thin annotated wrapper
// of std::mutex (same pattern as the macro header of the Clang Thread Safety Analysis docs, Abseil absl::Mutex / MutexLock,
// Chromium base::Lock / AutoLock, Marl marl::mutex). UniqueLock converts to std::unique_lock<std::mutex>& so
// std::condition_variable::wait(lk, pred) is unchanged; a wait predicate that reads guarded state is a separate function for
// the analysis and says so with WH_REQUIRES(mu) after its parameter list.
//
// Rules for annotated state: a member written by more than one thread names its mutex with WH_GUARDED_BY(mu); a function that
// must be called with the lock held says WH_REQUIRES(mu). Written for WHIRL.
#pragma once

#include <mutex>

#if defined(__clang__)
#define WH_TSA_ATTR(x) __attribute__((x))
#else
#define WH_TSA_ATTR(x)
#endif

#define WH_CAPABILITY(x) WH_TSA_ATTR(capability(x))
#define WH_SCOPED_CAPABILITY WH_TSA_ATTR(scoped_lockable)
#define WH_GUARDED_BY(x) WH_TSA_ATTR(guarded_by(x))
#define WH_PT_GUARDED_BY(x) WH_TSA_ATTR(pt_guarded_by(x))
#define WH_REQUIRES(...) WH_TSA_ATTR(requires_capability(__VA_ARGS__))
#define WH_ACQUIRE(...) WH_TSA_ATTR(acquire_capability(__VA_ARGS__))
#define WH_RELEASE(...) WH_TSA_ATTR(release_capability(__VA_ARGS__))
#define WH_TRY_ACQUIRE(...) WH_TSA_ATTR(try_acquire_capability(__VA_ARGS__))
#define WH_EXCLUDES(...) WH_TSA_ATTR(locks_excluded(__VA_ARGS__))
#define WH_RETURN_CAPABILITY(x) WH_TSA_ATTR(lock_returned(x))
#define WH_NO_THREAD_SAFETY_ANALYSIS WH_TSA_ATTR(no_thread_safety_analysis)

namespace whirl {

#if defined(__clang__)

class WH_CAPABILITY("mutex") Mutex {
public:
    Mutex() = default;
    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;
    void lock() WH_ACQUIRE() { m_.lock(); }
    void unlock() WH_RELEASE() { m_.unlock(); }
    bool try_lock() WH_TRY_ACQUIRE(true) { return m_.try_lock(); }
    std::mutex& native() { return m_; }

private:
    std::mutex m_;
};

class WH_SCOPED_CAPABILITY MutexLock {
public:
    explicit MutexLock(Mutex& m) WH_ACQUIRE(m) : m_(m) { m_.lock(); }
    ~MutexLock() WH_RELEASE() { m_.unlock(); }
    MutexLock(const MutexLock&) = delete;
    MutexLock& operator=(const MutexLock&) = delete;

private:
    Mutex& m_;
};

class WH_SCOPED_CAPABILITY UniqueLock {
public:
    explicit UniqueLock(Mutex& m) WH_ACQUIRE(m) : lk_(m.native()) {}
    ~UniqueLock() WH_RELEASE() {}
    UniqueLock(const UniqueLock&) = delete;
    UniqueLock& operator=(const UniqueLock&) = delete;
    void lock() WH_ACQUIRE() { lk_.lock(); }
    void unlock() WH_RELEASE() { lk_.unlock(); }
    // for std::condition_variable::wait / wait_for / wait_until
    operator std::unique_lock<std::mutex>&() { return lk_; }

private:
    std::unique_lock<std::mutex> lk_;
};

#else

using Mutex = std::mutex;
using MutexLock = std::lock_guard<std::mutex>;
using UniqueLock = std::unique_lock<std::mutex>;

#endif

// arch025-tsaidle: a function that is handed the caller's UniqueLock and unlocks / relocks it says WH_REQUIRES(mu) and
// does both through these two, which name the mutex: the analysis cannot tie a UniqueLock& parameter to its mutex (the
// parameter form `UniqueLock& lk WH_REQUIRES(mu)` of the Clang docs does not bind a member mutex in clang 21), so lk.unlock()
// there would need WH_NO_THREAD_SAFETY_ANALYSIS on the whole function, its wait predicates included. With these the body
// stays checked (a guarded read between them, a missing relock, a predicate without WH_REQUIRES all fail cpu_tsa).
// lk must be the lock of mu (lk was made from mu). MSVC: the same lk.unlock() / lk.lock() as before.
inline void unlockHeld(UniqueLock& lk, Mutex& mu) WH_RELEASE(mu) WH_NO_THREAD_SAFETY_ANALYSIS {
    (void)mu;
    lk.unlock();
}
inline void relockHeld(UniqueLock& lk, Mutex& mu) WH_ACQUIRE(mu) WH_NO_THREAD_SAFETY_ANALYSIS {
    (void)mu;
    lk.lock();
}

}  // namespace whirl
