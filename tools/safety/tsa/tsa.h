// safe025 prototype: Clang thread-safety annotations over MSVC STL (std::mutex has no capability attribute there, so a
// thin annotated wrapper is needed; same idea as Abseil Mutex / Chromium base::Lock). Test-only prototype, not in the product.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <mutex>

#if defined(__clang__)
#define WH_TSA(x) __attribute__((x))
#else
#define WH_TSA(x)
#endif
#define WH_CAPABILITY(x) WH_TSA(capability(x))
#define WH_SCOPED_CAPABILITY WH_TSA(scoped_lockable)
#define WH_GUARDED_BY(x) WH_TSA(guarded_by(x))
#define WH_ACQUIRE(...) WH_TSA(acquire_capability(__VA_ARGS__))
#define WH_RELEASE(...) WH_TSA(release_capability(__VA_ARGS__))
#define WH_REQUIRES(...) WH_TSA(requires_capability(__VA_ARGS__))
#define WH_EXCLUDES(...) WH_TSA(locks_excluded(__VA_ARGS__))

namespace whirl {
class WH_CAPABILITY("mutex") Mutex {
public:
    void lock() WH_ACQUIRE() { m_.lock(); }
    void unlock() WH_RELEASE() { m_.unlock(); }
private:
    std::mutex m_;
};
class WH_SCOPED_CAPABILITY MutexLock {
public:
    explicit MutexLock(Mutex& m) WH_ACQUIRE(m) : m_(m) { m_.lock(); }
    ~MutexLock() WH_RELEASE() { m_.unlock(); }
private:
    Mutex& m_;
};
}  // namespace whirl
