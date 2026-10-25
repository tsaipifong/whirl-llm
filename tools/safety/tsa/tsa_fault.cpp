// safe025: shared state accessed with / without its lock. Default: correct. -DINJECT_UNLOCKED: the fault.
#include "tsa.h"
#include <vector>
namespace {
struct Shared {
    whirl::Mutex mu;
    std::vector<int> items WH_GUARDED_BY(mu);
    int count WH_GUARDED_BY(mu) = 0;
    void add(int v) {
        whirl::MutexLock lk(mu);
        items.push_back(v);
        ++count;
    }
    int size() {
        whirl::MutexLock lk(mu);
        return count;
    }
    void addUnlocked(int v) WH_REQUIRES(mu) { items.push_back(v); ++count; }
#ifdef INJECT_UNLOCKED
    void faulty(int v) { items.push_back(v); ++count; }       // no lock
    void faulty2() { addUnlocked(1); }                         // REQUIRES(mu) caller does not hold it
#endif
};
}  // namespace
int main() {
    Shared s;
    s.add(1);
#ifdef INJECT_UNLOCKED
    s.faulty(2);
    s.faulty2();
#endif
    return s.size() == 1 ? 0 : 1;
}
