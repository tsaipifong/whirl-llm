// whirl-iso-common-tsa-tests: the header-only module common.tsa (include/whirl/thread_annotations.h) compiled with nothing but its own
// header and linked to nothing (ARCH-1 8.5 isolation, arch025-iso4). MSVC (the product, and this check) sees every macro empty and
// Mutex / MutexLock / UniqueLock as the std types, so the checks pin exactly that: the product code is the plain std code. The
// annotated clang wrappers are exercised by the cpu_tsa build (clang-cl -Wthread-safety), not here.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/thread_annotations.h"

#include <condition_variable>
#include <thread>
#include <type_traits>

namespace {

struct Counter {
    whirl::Mutex mu;
    int n WH_GUARDED_BY(mu) = 0;
    int* p WH_PT_GUARDED_BY(mu) = nullptr;
    void bump() WH_EXCLUDES(mu) {
        whirl::MutexLock lk(mu);
        ++n;
    }
    int readLocked() WH_REQUIRES(mu) { return n; }
};

void testMsvcAliases() {
#if !defined(__clang__)
    static_assert(std::is_same_v<whirl::Mutex, std::mutex>);
    static_assert(std::is_same_v<whirl::MutexLock, std::lock_guard<std::mutex>>);
    static_assert(std::is_same_v<whirl::UniqueLock, std::unique_lock<std::mutex>>);
#endif
    ISO_CHECK(true);
}

void testLocking() {
    Counter c;
    std::thread a([&] { for (int i = 0; i < 1000; ++i) c.bump(); });
    std::thread b([&] { for (int i = 0; i < 1000; ++i) c.bump(); });
    a.join();
    b.join();
    whirl::MutexLock lk(c.mu);
    ISO_CHECK(c.readLocked() == 2000);
}

void testUniqueLockAndWait() {
    whirl::Mutex mu;
    std::condition_variable cv;
    bool ready = false;
    std::thread t([&] {
        whirl::UniqueLock lk(mu);
        ready = true;
        cv.notify_one();
    });
    whirl::UniqueLock lk(mu);
    cv.wait(lk, [&] { return ready; });  // UniqueLock converts to std::unique_lock<std::mutex>&
    ISO_CHECK(ready);
    WH_UNLOCK_HELD(lk, mu);
    ISO_CHECK(!lk.owns_lock());
    WH_RELOCK_HELD(lk, mu);
    ISO_CHECK(lk.owns_lock());
    lk.unlock();
    t.join();
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("common.tsa", {{"msvc_aliases", testMsvcAliases}, {"locking", testLocking}, {"unique_lock_wait", testUniqueLockAndWait}}, argc, argv);
}
