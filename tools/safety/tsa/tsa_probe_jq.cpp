// arch025-engine S0 (DESIGN 4 S10, review2 R11): can Engine's methods name the lock of a component whose mutex is PRIVATE?
// Shape of S10: class JobQueue owns `mu` (private, the queue data guarded by it); Engine holds `JobQueue jq_` and a method
// of Engine (later: of Scheduler) runs with that lock held. Variants (-D):
//   V_PRIVATE_MEMBER   WH_REQUIRES(jq_.mu) with mu private                 -> expected: access error (not usable)
//   V_PUBLIC_MEMBER    WH_REQUIRES(jq_.mu) with mu public                  -> compiles; but anyone can lock / write
//   V_ACCESSOR         mu private + `Mutex& mutex() WH_RETURN_CAPABILITY(mu)`, WH_REQUIRES(jq_.mutex())  -> compiles
//   + FAULT            the caller does not hold the lock                   -> must fail -Werror=thread-safety
// tools/safety/tsa/tsa_probe.py compiles each variant with clang-cl -fsyntax-only and prints the verdicts.
// SPDX-License-Identifier: Apache-2.0
#include "whirl/thread_annotations.h"
#include <deque>

namespace probe {
class JobQueue {
#if defined(V_PUBLIC_MEMBER)
public:
    whirl::Mutex mu;
#else
    whirl::Mutex mu;
public:
#endif
#if defined(V_ACCESSOR)
    whirl::Mutex& mutex() WH_RETURN_CAPABILITY(mu) { return mu; }
#endif
    // the owned data is private and changed only through methods (llama.cpp server-queue.h:15-46)
    void push(int v) WH_REQUIRES(mu) { q_.push_back(v); }
    int size() const WH_REQUIRES(mu) { return static_cast<int>(q_.size()); }
private:
    std::deque<int> q_ WH_GUARDED_BY(mu);
};

class Engine {
public:
#if defined(V_ACCESSOR)
    void admit() WH_REQUIRES(jq_.mutex()) { jq_.push(1); }
    int run() {
#if defined(FAULT)
        admit();                                   // lock not held
        return 0;
#else
        whirl::MutexLock lk(jq_.mutex());
        admit();
        return jq_.size();
#endif
    }
#else
    void admit() WH_REQUIRES(jq_.mu) { jq_.push(1); }
    int run() {
#if defined(FAULT)
        admit();
        return 0;
#else
        whirl::MutexLock lk(jq_.mu);
        admit();
        return jq_.size();
#endif
    }
#endif
private:
    JobQueue jq_;
};
}  // namespace probe

int main() {
    probe::Engine e;
    return e.run() == 1 ? 0 : 1;
}
