// whirl-host-bench: speed / latency of the server's host components, one component at a time (measure-tools P3,
// progress/measure-tools.DESIGN.md §A3-A5). Test executable only: no product code is changed or added.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Prior art: progress/mt-p3.PRIOR_ART.md (llama.cpp test-backend-ops perf mode: batch sized to a
// time target, work-normalised units, re-init outside the timed region; Google Benchmark: grow iterations to a minimum
// time, repetitions with median / spread, consume every result; vLLM benchmarks/benchmark_block_pool.py: the block
// pool timed apart from the scheduler).
//
//   whirl-host-bench --suite S[,S...] [--backend null|hip] [--device SPEC] [--gguf TOKENIZER.gguf] [--jsonl FILE]
//                    [--quick] [--time-cap S] [--selftest]
// Suites (backend null = CPU only, no GPU; real components over device::NullBackend + tests/server MockModel):
//   pagepool  M3 PagePool: ensure + unref, cow of a shared tail, take with eviction, quarantine + reclaim (1k..128k tok)
//   prefix    M4 PrefixStore: pickSlot / lcpTarget / cacheMatch over 1 / 4 / 16 slots at 1k..128k tok
//   ckpt      M2 CkptStore: init + release, save, load of one slot checkpoint
//   handles   M1 RAII handles: DevBuf / PinnedBuf alloc + free, EventH create + destroy
//   tier      M5 TierBridge + kv_tier: spill of an idle slot (new entry, then removed), restoreMatch over 1 / 8 entries
//   sched     A4 Engine + HttpServer + MockModel (free forward): us per decode cycle at 1 / 4 / 8 slots, request
//             latency, 128k prefix reuse request, stream on / off
//   copy      A5 copy bandwidth h2d / d2h / d2d by size (64 KiB..64 MiB) and queue depth 1 / 4. On --backend null it is
//             host memmove (a control); on --backend hip (whirl-host-bench-hip only, --device required) the GPU copies.
// The GGUF is read for its tokenizer only (MockModel needs one); default WHIRL_TEST_GGUF. Suites prefix / handles /
// copy need none.
//
// Method (per cell): one probe execution of the op records the work counters (copies / bytes by direction, syncs,
// allocations, pages; exact, the contract part); then the batch size k is doubled until one timed batch takes
// >= 200 us; 2 warm-up batches; samples (one batch each, ns per op = wall / k) until >= 15 samples and
// (IQR / median <= 2 % or the time cap) -- stop_reason says which. Untimed prep / clean steps per batch keep reset
// work out of the timed region. Thread cycles (QueryThreadCycleTime) are recorded next to wall time.
// NullBackend copies are memmove of host memory: tier / ckpt / pagepool-cow times include memmove of the mock's
// buffers (bytes in the work row); they are host logic + memmove cost, not device copy cost.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef near
#undef far

#include "bench_out.h"
#include "http_client.h"
#include "mock_backend.h"

#include "cache/ckpt_store.h"
#include "cache/page_pool.h"
#include "device/backend.h"
#include "device/handles.h"
#include "device/null_backend.h"
#include "server/engine.h"
#include "server/http.h"
#include "server/log.h"
#include "server/prefix_store.h"
#include "server/tier_bridge.h"
#include "server/tokens.h"
#include "tier/kv_tier.h"
#include "whirl/gguf.h"
#include "whirl/model.h"
#include "whirl/tokenizer.h"

#ifdef WHIRL_HOST_BENCH_HIP
#include "whirl/gpu_select.h"
#include "whirl/hip.h"
#endif


#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace whirl;
using namespace whirl::server;
namespace wb = whirl::bench;

namespace {

constexpr const char* kTool = "whirl-host-bench";
constexpr const char* kToolVer = "0.1";

// ------------------------------------------------------------------ counting decorator (exact work counters)
// Forwards every call to the real backend and counts it; copy direction from the pinned host blocks it handed out.
class CountingBackend final : public device::Backend {
public:
    explicit CountingBackend(device::Backend& b) : b_(b) {}
    device::Backend& inner() { return b_; }
    struct Counts {
        std::int64_t c_h2d = 0, b_h2d = 0, c_d2h = 0, b_d2h = 0, c_d2d = 0, b_d2d = 0;
        std::int64_t syncs = 0, allocs = 0, frees = 0, events = 0, records = 0;
    };
    Counts counts() const {
        Counts c;
        c.c_h2d = c_h2d_.load();
        c.b_h2d = b_h2d_.load();
        c.c_d2h = c_d2h_.load();
        c.b_d2h = b_d2h_.load();
        c.c_d2d = c_d2d_.load();
        c.b_d2d = b_d2d_.load();
        c.syncs = syncs_.load();
        c.allocs = allocs_.load();
        c.frees = frees_.load();
        c.events = events_.load();
        c.records = records_.load();
        return c;
    }

    void copyAsync(device::DevPtr dst, device::DevPtr src, std::uint64_t bytes, device::Stream s) override {
        const bool fh = pinned(src), th = pinned(dst);
        if (fh && !th) add(c_h2d_, b_h2d_, bytes);
        else if (th && !fh) add(c_d2h_, b_d2h_, bytes);
        else add(c_d2d_, b_d2d_, bytes);
        b_.copyAsync(dst, src, bytes, s);
    }
    void download(void* dst, device::DevPtr src, std::uint64_t bytes) override {
        add(c_d2h_, b_d2h_, bytes);
        b_.download(dst, src, bytes);
    }
    device::Stream streamCreateNonBlocking() override { return b_.streamCreateNonBlocking(); }
    void streamDestroy(device::Stream s) override { b_.streamDestroy(s); }
    void streamSync(device::Stream s) override {
        syncs_.fetch_add(1, std::memory_order_relaxed);
        b_.streamSync(s);
    }
    void streamWaitEvent(device::Stream s, device::Event e) override { b_.streamWaitEvent(s, e); }
    device::Event eventCreate() override {
        events_.fetch_add(1, std::memory_order_relaxed);
        return b_.eventCreate();
    }
    void eventDestroy(device::Event e) override { b_.eventDestroy(e); }
    void eventRecord(device::Event e, device::Stream s) override {
        records_.fetch_add(1, std::memory_order_relaxed);
        b_.eventRecord(e, s);
    }
    bool eventDone(device::Event e) override { return b_.eventDone(e); }
    bool eventDoneChecked(device::Event e) override { return b_.eventDoneChecked(e); }
    void eventSync(device::Event e) override {
        syncs_.fetch_add(1, std::memory_order_relaxed);
        b_.eventSync(e);
    }
    void* hostMalloc(std::uint64_t bytes) override {
        void* p = b_.hostMalloc(bytes);
        allocs_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mu_);
        host_[reinterpret_cast<std::uintptr_t>(p)] = bytes;
        return p;
    }
    void hostFree(void* p) override {
        frees_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            host_.erase(reinterpret_cast<std::uintptr_t>(p));
        }
        b_.hostFree(p);
    }
    device::DevPtr malloc(std::uint64_t bytes) override {
        allocs_.fetch_add(1, std::memory_order_relaxed);
        return b_.malloc(bytes);
    }
    void free(device::DevPtr p) override {
        frees_.fetch_add(1, std::memory_order_relaxed);
        b_.free(p);
    }
    void setDevice(int d) override { b_.setDevice(d); }
    const char* lastErrorString() const override { return b_.lastErrorString(); }
    hip::MemInfo memInfo() override { return b_.memInfo(); }
    hip::WddmMemInfo wddmMemInfo() override { return b_.wddmMemInfo(); }
    hip::LocalMemSnap localMemSnap() override { return b_.localMemSnap(); }
    hip::UmaMem umaMemory(int d) override { return b_.umaMemory(d); }
    hip::GpuOthers gpuOthers(int d) override { return b_.gpuOthers(d); }
    std::uint64_t pinnedHostBytes() override { return b_.pinnedHostBytes(); }

private:
    static void add(std::atomic<std::int64_t>& c, std::atomic<std::int64_t>& b, std::uint64_t n) {
        c.fetch_add(1, std::memory_order_relaxed);
        b.fetch_add(static_cast<std::int64_t>(n), std::memory_order_relaxed);
    }
    bool pinned(device::DevPtr p) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = host_.upper_bound(static_cast<std::uintptr_t>(p));
        if (it == host_.begin()) return false;
        --it;
        return static_cast<std::uintptr_t>(p) < it->first + it->second;
    }
    device::Backend& b_;
    std::mutex mu_;
    std::map<std::uintptr_t, std::uint64_t> host_;
    std::atomic<std::int64_t> c_h2d_{0}, b_h2d_{0}, c_d2h_{0}, b_d2h_{0}, c_d2d_{0}, b_d2d_{0};
    std::atomic<std::int64_t> syncs_{0}, allocs_{0}, frees_{0}, events_{0}, records_{0};
};

wb::Work diff(const CountingBackend::Counts& a, const CountingBackend::Counts& b, std::int64_t pages) {
    wb::Work w;
    w.copies_h2d = b.c_h2d - a.c_h2d;
    w.bytes_h2d = b.b_h2d - a.b_h2d;
    w.copies_d2h = b.c_d2h - a.c_d2h;
    w.bytes_d2h = b.b_d2h - a.b_d2h;
    w.copies_d2d = b.c_d2d - a.c_d2d;
    w.bytes_d2d = b.b_d2d - a.b_d2d;
    w.syncs = b.syncs - a.syncs;
    w.allocs = b.allocs - a.allocs;
    w.pages = pages;
    return w;
}

bool sameWork(const wb::Work& a, const wb::Work& b) {
    return a.copies_h2d == b.copies_h2d && a.bytes_h2d == b.bytes_h2d && a.copies_d2h == b.copies_d2h && a.bytes_d2h == b.bytes_d2h &&
           a.copies_d2d == b.copies_d2d && a.bytes_d2d == b.bytes_d2d && a.syncs == b.syncs && a.allocs == b.allocs && a.pages == b.pages;
}

std::string workText(const wb::Work& w) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "h2d %lld/%lld d2h %lld/%lld d2d %lld/%lld syncs %lld allocs %lld pages %lld",
                  static_cast<long long>(w.copies_h2d), static_cast<long long>(w.bytes_h2d), static_cast<long long>(w.copies_d2h),
                  static_cast<long long>(w.bytes_d2h), static_cast<long long>(w.copies_d2d), static_cast<long long>(w.bytes_d2d),
                  static_cast<long long>(w.syncs), static_cast<long long>(w.allocs), static_cast<long long>(w.pages));
    return buf;
}

// ------------------------------------------------------------------ sampler
volatile std::uint64_t g_sink = 0;  // every op's result goes here (no dead-code elimination)
inline void sink(std::uint64_t v) { g_sink = g_sink + v; }

std::uint64_t threadCycles() {
    ULONG64 c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return static_cast<std::uint64_t>(c);
}

struct Opts {
    double time_cap_s = 1.5;   // per cell (sampling part)
    int min_n = 15, max_n = 101;
    double min_batch_us = 200;
    double rel_stop = 0.02;    // IQR / median
    bool quick = false;        // selftest: 5 samples, 0.05 s
};

struct Sampled {
    wb::Stats ns, cyc;
    std::string stop;
    std::int64_t k = 0;
    double iqr_rel = 0;
};

// prep(k) / clean(k) untimed; run(k) timed: k ops. kmax: the most ops one batch can hold (pool capacity etc.)
Sampled sampleOps(const Opts& o, std::int64_t kmax, const std::function<void(std::int64_t)>& prep,
                  const std::function<void(std::int64_t)>& run, const std::function<void(std::int64_t)>& clean) {
    using clk = std::chrono::steady_clock;
    auto batch = [&](std::int64_t k, double& wall_ns, double& cyc) {
        if (prep) prep(k);
        const std::uint64_t c0 = threadCycles();
        const auto t0 = clk::now();
        run(k);
        const auto t1 = clk::now();
        const std::uint64_t c1 = threadCycles();
        if (clean) clean(k);
        wall_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        cyc = static_cast<double>(c1 - c0);
    };
    Sampled s;
    std::int64_t k = 1;
    double w = 0, c = 0;
    for (;;) {
        batch(k, w, c);
        if (w >= o.min_batch_us * 1000.0 || k >= kmax) break;
        k = std::min<std::int64_t>(kmax, k * 2);
    }
    s.k = k;
    const int warm = o.quick ? 1 : 2;
    for (int i = 0; i < warm; ++i) batch(k, w, c);
    std::vector<double> ns, cy;
    const auto t_start = clk::now();
    const int min_n = o.quick ? 5 : o.min_n;
    const double cap = o.quick ? 0.05 : o.time_cap_s;
    for (;;) {
        batch(k, w, c);
        ns.push_back(w / static_cast<double>(k));
        cy.push_back(c / static_cast<double>(k));
        const double el = std::chrono::duration<double>(clk::now() - t_start).count();
        if (static_cast<int>(ns.size()) >= min_n) {
            std::vector<double> v = ns;
            std::sort(v.begin(), v.end());
            const double med = wb::quantile_sorted(v, 0.5);
            const double iqr = wb::quantile_sorted(v, 0.75) - wb::quantile_sorted(v, 0.25);
            s.iqr_rel = med > 0 ? iqr / med : 0;
            if (!o.quick && s.iqr_rel <= o.rel_stop) {
                s.stop = "rel<=2%";
                break;
            }
            if (el >= cap) {
                s.stop = "time";
                break;
            }
            if (static_cast<int>(ns.size()) >= o.max_n) {
                s.stop = "n";
                break;
            }
        }
        if (el >= cap * 4 && static_cast<int>(ns.size()) >= 3) {  // very slow cells: fewer samples, said so
            s.stop = "time-min";
            break;
        }
    }
    s.ns = wb::summarize(ns);
    s.cyc = wb::summarize(cy);
    return s;
}

// ------------------------------------------------------------------ output
struct BenchOut {
    wb::JsonlWriter* jsonl = nullptr;
    std::string backend = "null", gpu_arch, gpu_pci, settings_extra;
    int rows = 0, lost = 0;
    // selftest: work of every (metric, cell) seen, to compare two passes
    std::map<std::string, wb::Work>* seen = nullptr;
    int work_mismatch = 0;
};

void emit(BenchOut& out, const char* component, const std::string& metric, const std::string& cell, const Sampled& s,
          const wb::Work& work, std::vector<std::pair<std::string, std::string>> extra = {}, const char* unit = "ns",
          double scale = 1.0) {
    wb::Row r;
    r.tool = kTool;
    r.tool_ver = kToolVer;
    r.component = component;
    r.cell = cell;
    r.metric = metric;
    r.unit = unit;
    wb::Stats st = s.ns;
    if (scale != 1.0) {
        st.median *= scale;
        st.p10 *= scale;
        st.p90 *= scale;
        st.mad *= scale;
    }
    r.set_stats(st);
    r.stop_reason = s.stop;
    r.work = work;
    r.settings = "backend=" + out.backend + ";timer=wall;batch_us>=200;sampler=1" + out.settings_extra;  // stable key (k is in extra)
    r.gpu_arch = out.gpu_arch;
    r.gpu_pci = out.gpu_pci;
    r.clock_note = out.backend == "null" ? "cpu-only" : "";
    r.ended = wb::iso_now();
    extra.push_back({"cyc_per_op_median", wb::json_number(s.cyc.median)});
    extra.push_back({"iqr_rel", wb::json_number(s.iqr_rel)});
    extra.push_back({"batch_k", wb::json_number(s.k)});
    r.extra = std::move(extra);
    std::printf("%-8s %-34s %-28s %12.1f %s  p10 %10.1f p90 %10.1f  n %3lld k %6lld %-8s | %s\n", component, metric.c_str(),
                cell.c_str(), st.median, unit, st.p10, st.p90, static_cast<long long>(st.n), static_cast<long long>(s.k),
                s.stop.c_str(), workText(work).c_str());
    std::fflush(stdout);
    ++out.rows;
    if (out.jsonl && out.jsonl->enabled() && !out.jsonl->write(r)) ++out.lost;
    if (out.seen) {
        const std::string key = metric + "|" + cell;
        auto it = out.seen->find(key);
        if (it == out.seen->end()) (*out.seen)[key] = work;
        else if (!sameWork(it->second, work)) {
            ++out.work_mismatch;
            std::printf("SELFTEST work mismatch %s: %s vs %s\n", key.c_str(), workText(it->second).c_str(), workText(work).c_str());
        }
    }
}

// selftest expectations
int g_fail = 0;
#define EXPECT(c)                                                                     \
    do {                                                                              \
        if (!(c)) {                                                                   \
            ++g_fail;                                                                 \
            std::printf("SELFTEST FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);         \
        }                                                                             \
    } while (0)

const std::uint32_t kToks[] = {1024, 8192, 32768, 131072};

std::vector<std::uint32_t> filler(std::size_t n, std::uint32_t salt, std::size_t from = 0, const std::vector<std::uint32_t>* prefix = nullptr) {
    std::vector<std::uint32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (prefix && i < from) v[i] = (*prefix)[i];
        else v[i] = static_cast<std::uint32_t>(1000 + (i * 7 + salt * 7919) % 20000);
    }
    return v;
}

// ------------------------------------------------------------------ M3 PagePool
struct OneEvictor final : PagePool::Evictor {
    PagePool& pool;
    std::vector<std::int32_t> victims;  // one page each
    std::size_t next = 0;
    std::uint64_t calls = 0;
    explicit OneEvictor(PagePool& p) : pool(p) {}
    bool evictOne() override {
        ++calls;
        if (next >= victims.size()) return false;
        const std::int32_t one[] = {victims[next++]};
        pool.unref(one);
        return true;
    }
};

void suitePagePool(const Opts& o, BenchOut& out, const Tokenizer& tok) {
    device::NullBackend nb;
    CountingBackend cb(nb);
    test::MockConfig mc;
    mc.parallel = 2;
    mc.slot_ctx = 131072;
    mc.pool_tokens = 3 * 131072;  // 1536 pages of 256 tokens
    mc.mtp = false;
    auto model = std::make_unique<test::MockModel>(cb, tok, mc);
    {
        PagePool pool(cb, *model);
        const auto arrs = model->kvArrays();
        pool.init(model->poolPages(), arrs);
        device::Stream ts = cb.streamCreateNonBlocking();
        pool.setQuarantineStream(ts);
        const std::uint32_t pages = pool.pages();
        OneEvictor none(pool);
        std::uint64_t page_bytes = 0;
        for (const KvArr& a : arrs) page_bytes += static_cast<std::uint64_t>(a.row) * qwen35::kv_page;
        for (const std::uint32_t t : kToks) {
            const std::string cell = "tok=" + std::to_string(t);
            const std::int64_t np = static_cast<std::int64_t>(PagePool::pagesFor(t));
            // ensure + unref (a request's pages taken page by page, then the slot dropped)
            {
                std::vector<std::int32_t> l;
                auto c0 = cb.counts();
                pool.ensure(l, 0, t, none);
                const std::int64_t got = static_cast<std::int64_t>(l.size());
                pool.unref(l);
                l.clear();
                const wb::Work w = diff(c0, cb.counts(), got);
                EXPECT(got == np);
                EXPECT(pool.freeCount() == pages);
                const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i) {
                        pool.ensure(l, 0, t, none);
                        sink(static_cast<std::uint64_t>(l.back()));
                        pool.unref(l);
                        l.clear();
                    }
                }, nullptr);
                emit(out, "M3", "host.pagepool.ensure_unref", cell, s, w, {{"ns_per_page", wb::json_number(s.ns.median / static_cast<double>(np))}});
            }
            // cow of a shared tail: the list's pages are shared with another holder; reuse = half the tokens -> the
            // page at (reuse - 1) / kv_page copied to a new page (every pool array), the rest released
            {
                std::vector<std::int32_t> base;
                pool.ensure(base, 0, t, none);
                const std::uint32_t reuse = t / 2 + 7;
                std::vector<std::vector<std::int32_t>> lists;
                auto prep = [&](std::int64_t k) {
                    lists.assign(static_cast<std::size_t>(k), base);
                    for (auto& l : lists)
                        for (std::int32_t p : l) pool.ref(p);
                };
                auto clean = [&](std::int64_t) {
                    for (auto& l : lists) pool.unref(l);
                    lists.clear();
                };
                prep(1);
                auto c0 = cb.counts();
                const PagePool::Cow r = pool.cow(lists[0], 1, reuse, false, none);
                const wb::Work w = diff(c0, cb.counts(), 1);
                EXPECT(r == PagePool::Cow::done);
                EXPECT(w.copies_d2d == static_cast<std::int64_t>(arrs.size()) && w.bytes_d2d == static_cast<std::int64_t>(page_bytes));
                clean(1);
                const std::int64_t kmax = std::min<std::int64_t>(4096, (pages - np) / 2);
                const Sampled s = sampleOps(o, kmax, prep, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i) sink(static_cast<std::uint64_t>(pool.cow(lists[static_cast<std::size_t>(i)], 1, reuse, false, none)));
                }, clean);
                emit(out, "M3", "host.pagepool.cow", cell, s, w, {{"page_bytes", wb::json_number(static_cast<std::int64_t>(page_bytes))}});
                pool.unref(base);
                EXPECT(pool.freeCount() == pages);
            }
            // quarantine + reclaim (a spill's pages held until its event, then given back)
            {
                std::vector<std::vector<std::int32_t>> lists;
                auto prep = [&](std::int64_t k) {
                    lists.assign(static_cast<std::size_t>(k), {});
                    for (auto& l : lists) pool.ensure(l, 0, t, none);
                };
                prep(1);
                auto c0 = cb.counts();
                pool.quarantine(lists[0]);
                pool.reclaim(false);
                const wb::Work w = diff(c0, cb.counts(), np);
                EXPECT(pool.quarantineEmpty());
                EXPECT(pool.freeCount() == pages);
                const std::int64_t kmax = std::max<std::int64_t>(1, pages / np);
                const Sampled s = sampleOps(o, kmax, prep, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i) {
                        pool.quarantine(lists[static_cast<std::size_t>(i)]);
                        pool.reclaim(false);
                    }
                    sink(pool.freeCount());
                }, nullptr);
                emit(out, "M3", "host.pagepool.quarantine_reclaim", cell, s, w);
                EXPECT(pool.freeCount() == pages);
            }
        }
        // take with eviction: the pool full of one-page holders; every take evicts one (the evictor itself is O(1),
        // so the number is PagePool's part)
        {
            OneEvictor ev(pool);
            std::vector<std::int32_t> taken;
            auto prep = [&](std::int64_t) {
                ev.victims.clear();
                ev.next = 0;
                while (auto p = pool.takeFree()) ev.victims.push_back(*p);
                taken.clear();
            };
            auto clean = [&](std::int64_t) {
                pool.unref(taken);
                taken.clear();
                for (std::size_t i = ev.next; i < ev.victims.size(); ++i) {
                    const std::int32_t one[] = {ev.victims[i]};
                    pool.unref(one);
                }
                ev.victims.clear();
                ev.next = 0;
            };
            prep(1);
            auto c0 = cb.counts();
            const auto p = pool.take(ev);
            if (p) taken.push_back(*p);
            const wb::Work w = diff(c0, cb.counts(), 1);
            EXPECT(p.has_value() && ev.next == 1);
            clean(1);
            const Sampled s = sampleOps(o, pages, prep, [&](std::int64_t k) {
                for (std::int64_t i = 0; i < k; ++i) taken.push_back(*pool.take(ev));
            }, clean);
            emit(out, "M3", "host.pagepool.take_evict", "pool_pages=" + std::to_string(pages), s, w);
            EXPECT(pool.freeCount() == pages);
        }
        cb.streamDestroy(ts);
    }
    model.reset();
    EXPECT(nb.clean());
}

// ------------------------------------------------------------------ M4 PrefixStore
ChatTokens benchChatIds() {
    ChatTokens ct;
    ct.im_start = 1;
    ct.im_end = 2;
    ct.nl = 3;
    ct.system = 4;
    ct.think = 5;
    return ct;
}

void suitePrefix(const Opts& o, BenchOut& out) {
    const ChatTokens ct = benchChatIds();
    PrefixStore::Config cfg;
    cfg.sys_min = sched::sys_min_default;
    cfg.kv_page = qwen35::kv_page;
    cfg.lcp_on = true;  // the server's default (WHIRL_SYS_LCP on)
    PrefixStore ps(ct, cfg);
    ps.spes().resize(2);  // n_spe_default 2 shared prefix checkpoints, none valid (lcpTarget scans the slots)
    for (const std::uint32_t t : kToks) {
        const std::vector<std::uint32_t> query = filler(t + 500, 1);
        for (const std::size_t nslots : {std::size_t{1}, std::size_t{4}, std::size_t{16}}) {
            const std::string cell = "tok=" + std::to_string(t) + ";slots=" + std::to_string(nslots);
            // slot i shares the query's first (i + 1) / (nslots + 1) of t tokens, then diverges; checkpoints at
            // a quarter, half and the end of its own tokens (prompt-end), LRU order = slot order
            std::vector<SlotPrefix> slots(nslots);
            for (std::size_t i = 0; i < nslots; ++i) {
                const std::size_t share = static_cast<std::size_t>(t) * (i + 1) / (nslots + 1);
                slots[i].cache_tokens = filler(t, 100 + static_cast<std::uint32_t>(i), share, &query);
                slots[i].last_used = i + 1;
                slots[i].ckpts.resize(3);
                const std::uint32_t pos[3] = {t / 4, t / 2, t};
                for (int j = 0; j < 3; ++j) {
                    Ckpt& c = slots[i].ckpts[j];
                    c.valid = true;
                    c.pos = pos[j];
                    c.kind = "prompt-end";
                    c.seq = static_cast<std::uint64_t>(j + 1);
                    c.has_logits = true;
                }
            }
            const wb::Work w{};  // host only: no device work
            {
                const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i) {
                        SlotPrefix* b = ps.pickSlot(std::span<SlotPrefix>(slots), query, [](const SlotPrefix&) { return true; });
                        sink(reinterpret_cast<std::uintptr_t>(b));
                    }
                }, nullptr);
                emit(out, "M4", "host.prefix.pick_slot", cell, s, w);
            }
            {
                const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i)
                        sink(ps.lcpTarget(cache::View<const cache::SlotCache>(slots), query, 0, 0));
                }, nullptr);
                emit(out, "M4", "host.prefix.lcp_target", cell, s, w);
            }
            if (nslots == 1) {
                const PrefixStore::Match m = ps.cacheMatch(slots[0], query);
                EXPECT(m.reuse >= t / 4 && m.reuse <= t / 2);  // slot 0 shares t / 2 tokens: a checkpoint at t / 4 or t / 2
                const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i) sink(ps.cacheMatch(slots[0], query).reuse);
                }, nullptr);
                emit(out, "M4", "host.prefix.cache_match", cell, s, w);
            }
        }
    }
}

// ------------------------------------------------------------------ M2 CkptStore, M1 handles
void suiteCkpt(const Opts& o, BenchOut& out, const Tokenizer& tok) {
    device::NullBackend nb;
    CountingBackend cb(nb);
    test::MockConfig mc;
    mc.parallel = 2;
    mc.pool_tokens = 4096;
    mc.mtp = false;
    auto model = std::make_unique<test::MockModel>(cb, tok, mc);
    {
        CkptStore cs(cb, *model, CkptStore::Layout{model->convBytes(), model->ssmBytes(), false});
        const std::string cell = "slot_ckpt;bytes=" + std::to_string(cs.bytes(true));
        {
            Ckpt c;
            auto c0 = cb.counts();
            cs.init(c, true);
            cs.release(c, {});
            const wb::Work w = diff(c0, cb.counts(), 0);
            EXPECT(w.allocs == 1);
            const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                for (std::int64_t i = 0; i < k; ++i) {
                    cs.init(c, true);
                    sink(c.dev.get());
                    cs.release(c, {});
                }
            }, nullptr);
            emit(out, "M2", "host.ckpt.init_release", cell, s, w);
        }
        {
            Ckpt c;
            cs.init(c, true);
            auto c0 = cb.counts();
            cs.save(c, std::nullopt, model->logits());
            const wb::Work ws = diff(c0, cb.counts(), 0);
            c0 = cb.counts();
            cs.load(c);
            const wb::Work wl = diff(c0, cb.counts(), 0);
            EXPECT(ws.copies_d2d > 0 && wl.copies_d2d > 0);
            const Sampled s1 = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                for (std::int64_t i = 0; i < k; ++i) cs.save(c, std::nullopt, model->logits());
                sink(c.pos);
            }, nullptr);
            emit(out, "M2", "host.ckpt.save", cell, s1, ws);
            const Sampled s2 = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                for (std::int64_t i = 0; i < k; ++i) cs.load(c);
                sink(c.pos);
            }, nullptr);
            emit(out, "M2", "host.ckpt.load", cell, s2, wl);
            cs.release(c, {});
        }
    }
    model.reset();
    EXPECT(nb.clean());
}

void suiteHandles(const Opts& o, BenchOut& out) {
    device::NullBackend nb;
    CountingBackend cb(nb);
    for (const std::uint64_t bytes : {std::uint64_t{4096}, std::uint64_t{1} << 20}) {
        const std::string cell = "bytes=" + std::to_string(bytes);
        auto c0 = cb.counts();
        { auto b = device::DevBuf::alloc(cb, bytes); }
        const wb::Work w = diff(c0, cb.counts(), 0);
        EXPECT(w.allocs == 1);
        const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
            for (std::int64_t i = 0; i < k; ++i) {
                auto b = device::DevBuf::alloc(cb, bytes);
                sink(b.get());
            }
        }, nullptr);
        emit(out, "M1", "host.handles.devbuf_alloc_free", cell, s, w);
        const Sampled s2 = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
            for (std::int64_t i = 0; i < k; ++i) {
                auto b = device::PinnedBuf::alloc(cb, bytes);
                sink(b.addr());
            }
        }, nullptr);
        emit(out, "M1", "host.handles.pinned_alloc_free", cell, s2, w);
    }
    {
        auto c0 = cb.counts();
        { auto e = device::EventH::create(cb); }
        const wb::Work w = diff(c0, cb.counts(), 0);
        const Sampled s = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
            for (std::int64_t i = 0; i < k; ++i) {
                auto e = device::EventH::create(cb);
                sink(reinterpret_cast<std::uintptr_t>(e.get()));
            }
        }, nullptr);
        emit(out, "M1", "host.handles.event_create_destroy", "-", s, w);
    }
    EXPECT(nb.clean());
}

// ------------------------------------------------------------------ M5 TierBridge + kv_tier
struct TierRig {
    device::NullBackend nb;
    CountingBackend cb{nb};
    std::unique_ptr<test::MockModel> model;
    std::vector<KvArr> kv;
    std::unique_ptr<ChatTokens> ct;
    std::unique_ptr<PrefixStore> ps;
    std::unique_ptr<CkptStore> cs;
    std::unique_ptr<tier::Tier> tier;
    std::unique_ptr<TierBridge> tb;
    explicit TierRig(const Tokenizer& tok, std::uint64_t ram_mb) {
        test::MockConfig mc;
        mc.parallel = 2;
        mc.slot_ctx = 131072;
        mc.pool_tokens = 2 * 131072;
        mc.mtp = false;
        model = std::make_unique<test::MockModel>(cb, tok, mc);
        kv = model->kvArrays();
        ct = std::make_unique<ChatTokens>(tok);
        ps = std::make_unique<PrefixStore>(*ct, PrefixStore::Config{});
        cs = std::make_unique<CkptStore>(cb, *model, CkptStore::Layout{model->convBytes(), model->ssmBytes(), false});
        tier = tier::Tier::create(cb, ram_mb << 20, 42, 0);
        tier::Config c;
        c.ram_bytes = ram_mb << 20;
        c.min_tokens = 512;
        c.ssd_delay_ms = 0;
        tier::Layout lay;
        const auto& cfg = model->cfg();
        std::uint64_t n_gdn = 0;
        for (auto p : model->ssmState())
            if (p) ++n_gdn;
        lay.ck_bytes = n_gdn * (model->convBytes() + model->ssmBytes()) + static_cast<std::uint64_t>(cfg.n_embd) * 4 +
                       static_cast<std::uint64_t>(cfg.n_vocab) * 4;
        std::uint64_t per_tok = 0;
        for (const KvArr& a : kv) per_tok += a.row;
        lay.page_bytes = per_tok * qwen35::kv_page;
        lay.page_tokens = qwen35::kv_page;
        lay.tok_cap = mc.slot_ctx;
        tb = std::make_unique<TierBridge>(cb, *model, *ct, *ps, kv, TierBridge::Config{});
        tb->attach(tier.get());
        ps->attachTier(tier.get());
        tier->start(c, lay, 7);
    }
    ~TierRig() {
        tb->attach(nullptr);
        ps->attachTier(nullptr);
        tier.reset();
        tb.reset();
        cs.reset();
        ps.reset();
        model.reset();
    }
    struct S {
        SlotPrefix sp;
        SlotTier st;
    };
    // toks with checkpoints at a quarter / half / end, pages 0..
    std::unique_ptr<S> slot(const std::vector<std::uint32_t>& toks) {
        auto s = std::make_unique<S>();
        s->sp.cache_tokens = toks;
        const std::uint32_t n = static_cast<std::uint32_t>(toks.size());
        const std::uint32_t pos[3] = {n / 4, n / 2, n};
        s->sp.ckpts.resize(3);
        for (int j = 0; j < 3; ++j) {
            Ckpt& c = s->sp.ckpts[j];
            cs->init(c, true);
            c.valid = true;
            c.pos = pos[j];
            c.seq = static_cast<std::uint64_t>(j + 1);
            c.kind = "prompt-end";
        }
        const std::size_t np = PagePool::pagesFor(n);
        for (std::size_t p = 0; p < np; ++p) s->sp.pages.push_back(static_cast<std::int32_t>(p));
        s->sp.ckpt_seq = 10;
        return s;
    }
    void release(S& s) {
        for (Ckpt& c : s.sp.ckpts) cs->release(c, {});  // resetTier synced the spill fence first
    }
    void resetTier(S& s) {
        if (s.st.spill_ev) nb.eventSync(s.st.spill_ev.get());
        if (tier::Entry* e = tier->find(s.st.tier_id)) {
            tier->gpuIdle(e);
            tier->removeEntry(e);
        }
        s.st.tier_id = 0;
        s.st.tier_pages = 0;
        s.st.spill_busy = false;
        s.st.spill_upto = 0;
        s.st.spill_wanted = false;
    }
};

void suiteTier(const Opts& o, BenchOut& out, const Tokenizer& tok) {
    for (const std::uint32_t t : kToks) {
        const std::string cell = "tok=" + std::to_string(t);
        TierRig r(tok, 512);
        auto s = r.slot(filler(t, 3));
        // spill of an idle slot to a new RAM entry (span list, copyOut, fence, dropContained / supersede scan over
        // the entries), then the entry removed (untimed reset)
        auto c0 = r.cb.counts();
        r.tb->spillSlot(0, true, s->sp, s->st);
        const wb::Work w = diff(c0, r.cb.counts(), static_cast<std::int64_t>(s->st.tier_pages));
        EXPECT(s->st.tier_id != 0);
        r.resetTier(*s);
        const Sampled sm = sampleOps(o, 1, nullptr, [&](std::int64_t) { r.tb->spillSlot(0, true, s->sp, s->st); },
                                     [&](std::int64_t) { r.resetTier(*s); });
        const std::int64_t bytes = w.bytes_d2d + w.bytes_d2h + w.bytes_h2d;
        emit(out, "M5", "host.tier.spill_new", cell, sm, w,
             {{"gbps_memmove", wb::json_number(sm.ns.median > 0 ? static_cast<double>(bytes) / sm.ns.median : 0.0)}});
        r.release(*s);
        // restoreMatch (which tier entry is worth restoring into an empty slot) over n entries of t tokens; entry i
        // shares the first (i + 1) / (n + 1) of the query
        for (const std::size_t n : {std::size_t{1}, std::size_t{8}}) {
            if (t == 131072 && n == 8 && o.quick) continue;
            const std::vector<std::uint32_t> query = filler(t + 500, 9);
            std::vector<std::unique_ptr<TierRig::S>> ss;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t share = static_cast<std::size_t>(t) * (i + 1) / (n + 1);
                ss.push_back(r.slot(filler(t, 200 + static_cast<std::uint32_t>(i), share, &query)));
                r.tb->spillSlot(static_cast<std::uint32_t>(i % 2), true, ss.back()->sp, ss.back()->st);
                r.nb.eventSync(ss.back()->st.spill_ev.get());
            }
            TierRig::S empty;
            const std::size_t entries = r.tier->entries.size();
            auto c1 = r.cb.counts();
            const auto m = r.tb->restoreMatch(empty.sp, query);
            const wb::Work wm = diff(c1, r.cb.counts(), 0);
            const Sampled sr = sampleOps(o, 1 << 20, nullptr, [&](std::int64_t k) {
                for (std::int64_t i = 0; i < k; ++i) {
                    auto mm = r.tb->restoreMatch(empty.sp, query);
                    sink(mm ? mm->reuse : 0);
                }
            }, nullptr);
            emit(out, "M5", "host.tier.restore_match", cell + ";entries=" + std::to_string(entries), sr, wm,
                 {{"reuse", wb::json_number(static_cast<std::int64_t>(m ? m->reuse : 0))}});
            for (auto& x : ss) {
                r.resetTier(*x);
                r.release(*x);
            }
        }
    }
}

// ------------------------------------------------------------------ A4 server scheduling (Engine over MockModel)
std::string reqBody(const std::vector<std::uint32_t>& prompt, std::uint32_t max_tokens, bool stream) {
    std::string s = "{\"prompt\":[";
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (i) s += ',';
        s += std::to_string(prompt[i]);
    }
    s += "],\"max_tokens\":" + std::to_string(max_tokens) + ",\"temperature\":0,\"ignore_eos\":true,\"stream\":" +
         (stream ? "true" : "false") + "}";
    return s;
}

struct EngineRig {
    device::NullBackend nb;
    CountingBackend cb{nb};
    std::unique_ptr<test::MockModel> model;
    std::unique_ptr<Engine> engine;
    std::unique_ptr<HttpServer> http;
    std::thread loop;
    EngineRig(const Tokenizer& tok, std::uint32_t parallel, std::uint32_t slot_ctx) {
        test::MockConfig mc;
        mc.parallel = parallel;
        mc.slot_ctx = slot_ctx;
        mc.pool_tokens = parallel * slot_ctx + slot_ctx;
        mc.max_batch = 4096;
        mc.mtp = false;
        model = std::make_unique<test::MockModel>(cb, tok, mc);
        EngineOptions eo;
        eo.parallel = mc.parallel;
        eo.ctx = mc.slot_ctx;
        eo.model_name = "mock";
        eo.use_mtp = false;
        engine = std::make_unique<Engine>(*model, cb, tok, eo);
        engine->allocState();
        engine->initPool();
        http = std::make_unique<HttpServer>(*engine);
        http->start("127.0.0.1", 0);
        loop = std::thread([this] { engine->runLoop(); });
    }
    ~EngineRig() {
        engine->stop();
        loop.join();
        http->stop();
        http.reset();
        engine.reset();
        model.reset();
    }
    test::HttpResult post(const std::string& b) {
        return test::httpRequest("127.0.0.1", http->port(), "POST", "/v1/completions", b);
    }
};

void suiteSched(const Opts& o, BenchOut& out, const Tokenizer& tok) {
    constexpr std::uint32_t kGen = 64, kPrompt = 512;
    for (const std::uint32_t par : {1u, 4u, 8u}) {
        for (const bool stream : {false, true}) {
            if (stream && par != 1) continue;
            const std::string cell = "slots=" + std::to_string(par) + ";prompt=" + std::to_string(kPrompt) + ";gen=" +
                                     std::to_string(kGen) + ";stream=" + (stream ? "1" : "0");
            EngineRig r(tok, par, 16384);
            std::uint64_t round = 0;
            int bad = 0;
            std::vector<double> lat;
            // one round: par concurrent requests with fresh prompts (cold 512-row prefill + 64 decode cycles each)
            auto oneRound = [&] {
                std::vector<std::thread> th;
                std::vector<test::HttpResult> res(par);
                for (std::uint32_t i = 0; i < par; ++i) {
                    const std::string b = reqBody(filler(kPrompt, static_cast<std::uint32_t>(round * 16 + i + 1)), kGen, stream);
                    th.emplace_back([&, i, b] { res[i] = r.post(b); });
                }
                for (auto& x : th) x.join();
                for (const auto& x : res) {
                    if (x.status != 200) ++bad;
                    lat.push_back(x.ms);
                }
                ++round;
            };
            auto c0 = r.cb.counts();
            oneRound();
            const wb::Work w = diff(c0, r.cb.counts(), 0);
            const auto st0 = r.engine->stats();
            lat.clear();
            Opts oo = o;
            oo.min_batch_us = 0;  // one round per sample (a round is ms long)
            const Sampled s = sampleOps(oo, 1, nullptr, [&](std::int64_t) { oneRound(); }, nullptr);
            // per decode cycle: a round runs kGen cycles (all slots together) plus one prefill
            emit(out, "A4", "host.sched.round_us_per_cycle", cell, s, w, {{"evictions", wb::json_number(static_cast<std::int64_t>(st0.evictions))}},
                 "us", 1e-3 / kGen);
            const wb::Stats ls = wb::summarize(lat);
            Sampled sl;
            sl.ns = ls;
            sl.cyc = wb::Stats{};
            sl.stop = s.stop;
            sl.k = 1;
            emit(out, "A4", "host.sched.request_ms", cell, sl, w, {}, "ms");
            EXPECT(bad == 0);
        }
    }
    // 128k prefix reuse: prompt A (ctx - 64) cached in the slot, then A + 32 tokens: the request that resumes from the
    // slot's prompt-end checkpoint (JSON parse of 131k numbers, lookup, cow, 32-row prefill, 8 decode cycles)
    {
        const std::uint32_t ctx = 131072, n = ctx - 64;
        EngineRig r(tok, 1, ctx);
        const auto A = filler(n, 5);
        const auto A2 = filler(n + 32, 6, n, &A);
        const std::string bA = reqBody(A, 8, false), bA2 = reqBody(A2, 8, false);
        int bad = 0;
        if (r.post(bA).status != 200) ++bad;
        auto c0 = r.cb.counts();
        if (r.post(bA2).status != 200) ++bad;
        const wb::Work w = diff(c0, r.cb.counts(), 0);
        Opts oo = o;
        oo.min_batch_us = 0;
        oo.min_n = 9;
        const Sampled s = sampleOps(oo, 1, [&](std::int64_t) {
            if (r.post(bA).status != 200) ++bad;
        }, [&](std::int64_t) {
            if (r.post(bA2).status != 200) ++bad;
        }, nullptr);
        emit(out, "A4", "host.sched.prefix_reuse_ms", "ctx=131072;reuse=" + std::to_string(n) + ";new=32;gen=8", s, w, {}, "ms", 1e-6);
        EXPECT(bad == 0);
    }
}

// ------------------------------------------------------------------ A5 copy bandwidth
void suiteCopy(const Opts& o, BenchOut& out, device::Backend& be) {
    CountingBackend cb(be);
    const std::uint64_t max_bytes = 64ull << 20;
    const int qd_max = 4;
    std::vector<device::DevBuf> dev;
    std::vector<device::PinnedBuf> host;
    for (int i = 0; i < qd_max; ++i) {
        dev.push_back(device::DevBuf::alloc(cb, max_bytes));
        host.push_back(device::PinnedBuf::alloc(cb, max_bytes));
    }
    auto dst2 = device::DevBuf::alloc(cb, max_bytes * qd_max);
    device::Stream s = cb.streamCreateNonBlocking();
    for (const std::uint64_t bytes : {64ull << 10, 1ull << 20, 2ull << 20, 16ull << 20, 64ull << 20}) {
        for (const int qd : {1, 4}) {
            for (const char* dir : {"h2d", "d2h", "d2d"}) {
                auto op = [&] {
                    for (int q = 0; q < qd; ++q) {
                        if (dir[0] == 'h') cb.copyAsync(dev[q].get(), host[q].addr(), bytes, s);
                        else if (dir[1] == '2' && dir[2] == 'h') cb.copyAsync(host[q].addr(), dev[q].get(), bytes, s);
                        else cb.copyAsync(dst2.get() + static_cast<std::uint64_t>(q) * max_bytes, dev[q].get(), bytes, s);
                    }
                    cb.streamSync(s);
                };
                auto c0 = cb.counts();
                op();
                const wb::Work w = diff(c0, cb.counts(), 0);
                const Sampled sm = sampleOps(o, 64, nullptr, [&](std::int64_t k) {
                    for (std::int64_t i = 0; i < k; ++i) op();
                }, nullptr);
                const double total = static_cast<double>(bytes) * qd;
                // GB/s from the median ns per op (one op = qd copies + one sync)
                Sampled g = sm;
                auto toG = [&](double ns) { return ns > 0 ? total / ns : 0.0; };
                g.ns.median = toG(sm.ns.median);
                g.ns.p10 = toG(sm.ns.p90);  // the slow end is the low bandwidth end
                g.ns.p90 = toG(sm.ns.p10);
                g.ns.mad = sm.ns.median > 0 ? g.ns.median * sm.ns.mad / sm.ns.median : 0;
                emit(out, "A5", std::string("tier.copy.") + dir + "_gbps",
                     "bytes=" + std::to_string(bytes) + ";qd=" + std::to_string(qd), g, w,
                     {{"us_per_op_median", wb::json_number(sm.ns.median / 1000.0)}}, "GB/s");
            }
        }
    }
    cb.streamDestroy(s);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef WHIRL_HOST_BENCH_HIP
    if (argc == 2 && std::strcmp(argv[1], gpusel::k_probe_arg) == 0) return gpusel::probeMain();
#endif
    std::string suites, backend = "null", device_spec, gguf_path, jsonl_path;
    Opts o;
    bool selftest = false;
    auto usage = [] {
        std::fprintf(stderr,
                     "usage: whirl-host-bench --suite pagepool,prefix,ckpt,handles,tier,sched,copy|all [--backend null|hip]\n"
                     "                        [--device SPEC] [--gguf TOKENIZER.gguf] [--jsonl FILE] [--quick] [--time-cap S]\n"
                     "       whirl-host-bench --selftest [--gguf TOKENIZER.gguf]\n");
        return 2;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--suite" && i + 1 < argc) suites = argv[++i];
        else if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--device" && i + 1 < argc) device_spec = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--jsonl" && i + 1 < argc) jsonl_path = argv[++i];
        else if (a == "--time-cap" && i + 1 < argc) o.time_cap_s = std::atof(argv[++i]);
        else if (a == "--quick") o.quick = true;
        else if (a == "--selftest") selftest = true;
        else return usage();
    }
    if (selftest) {
        suites = "all";
        o.quick = true;
    }
    if (suites == "all") suites = "pagepool,prefix,ckpt,handles,tier,sched,copy";
    std::set<std::string> want;
    for (std::size_t p = 0; p <= suites.size();) {
        std::size_t e = suites.find(',', p);
        if (e == std::string::npos) e = suites.size();
        if (e > p) want.insert(suites.substr(p, e - p));
        p = e + 1;
    }
    const std::set<std::string> known = {"pagepool", "prefix", "ckpt", "handles", "tier", "sched", "copy"};
    for (const auto& s : want)
        if (!known.count(s)) {
            std::fprintf(stderr, "whirl-host-bench: unknown suite \"%s\"\n", s.c_str());
            return usage();
        }
    if (want.empty() || (backend != "null" && backend != "hip")) return usage();
    BenchOut out;
    out.backend = backend;
    {
        // the process priority class is part of the key: quiet_wait.sh runs commands at nice 10 (below normal), and the
        // sched suite (thread wake-ups) is sensitive to it
        const DWORD pc = GetPriorityClass(GetCurrentProcess());
        const char* pn = pc == IDLE_PRIORITY_CLASS ? "idle" : pc == BELOW_NORMAL_PRIORITY_CLASS ? "below_normal" :
                         pc == NORMAL_PRIORITY_CLASS ? "normal" : pc == ABOVE_NORMAL_PRIORITY_CLASS ? "above_normal" :
                         pc == HIGH_PRIORITY_CLASS ? "high" : "other";
        out.settings_extra = std::string(";prio=") + pn;
    }
    std::unique_ptr<device::Backend> hipb;
    if (backend == "hip") {
#ifdef WHIRL_HOST_BENCH_HIP
        if (device_spec.empty()) {
            std::fprintf(stderr, "whirl-host-bench: --backend hip needs --device (pci:BUS:DEV); the GPU is never guessed\n");
            return 2;
        }
        for (const auto& s : want)
            if (s != "copy") {
                std::fprintf(stderr, "whirl-host-bench: --backend hip runs the copy suite only (got \"%s\")\n", s.c_str());
                return 2;
            }
        const gpusel::Restriction& gr = qwen35::restrictDevice(device_spec);
        if (!gr.note.empty()) std::printf("gpu: %s\n", gr.note.c_str());
        const int dev = qwen35::pickDevice(device_spec);
        const hip::DeviceInfo info = hip::describeDevice(dev);
        hip::setDevice(dev);
        hipb = device::makeHipBackend();
        out.gpu_arch = info.gcn_arch;
        out.gpu_pci = device_spec;
        std::printf("whirl-host-bench: device %d %s (%s), --device %s\n", dev, info.name.c_str(), info.gcn_arch.c_str(), device_spec.c_str());
#else
        std::fprintf(stderr, "whirl-host-bench: this build has no HIP backend; use whirl-host-bench-hip.exe\n");
        return 2;
#endif
    }
    const bool need_tok = want.count("pagepool") || want.count("ckpt") || want.count("tier") || want.count("sched");
    if (gguf_path.empty())
        if (const char* e = std::getenv("WHIRL_TEST_GGUF")) gguf_path = e;
    std::unique_ptr<gguf::File> file;
    std::unique_ptr<Tokenizer> tok;
    if (need_tok && backend == "null") {
        if (gguf_path.empty()) {
            std::fprintf(stderr, "whirl-host-bench: suites pagepool / ckpt / tier / sched need --gguf (tokenizer only) or WHIRL_TEST_GGUF\n");
            return 2;
        }
        try {
            file = std::make_unique<gguf::File>(gguf::File::open(gguf_path));
            tok = std::make_unique<Tokenizer>(Tokenizer::fromGguf(*file));
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "cannot load the tokenizer from %s: %s\n", gguf_path.c_str(), ex.what());
            return 2;
        }
    }
    Log::setConsole(false);
    wb::JsonlWriter jw(jsonl_path);
    out.jsonl = &jw;
    std::map<std::string, wb::Work> seen;
    if (selftest) out.seen = &seen;
    std::printf("whirl-host-bench %s: backend %s, suites %s%s\n", kToolVer, backend.c_str(), suites.c_str(), o.quick ? " (quick)" : "");
    const int passes = selftest ? 2 : 1;
    try {
        for (int pass = 0; pass < passes; ++pass) {
            if (backend == "null") {
                if (want.count("handles")) suiteHandles(o, out);
                if (want.count("prefix")) suitePrefix(o, out);
                if (want.count("pagepool")) suitePagePool(o, out, *tok);
                if (want.count("ckpt")) suiteCkpt(o, out, *tok);
                if (want.count("tier")) suiteTier(o, out, *tok);
                if (want.count("sched")) suiteSched(o, out, *tok);
                if (want.count("copy")) {
                    device::NullBackend nb;
                    suiteCopy(o, out, nb);
                    EXPECT(nb.clean());
                }
            } else {
                suiteCopy(o, out, *hipb);
            }
        }
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "whirl-host-bench: exception %s\n", ex.what());
        return 1;
    }
    if (out.lost) std::printf("whirl-host-bench: %d JSONL row(s) could not be written\n", out.lost);
    if (selftest) {
        EXPECT(out.work_mismatch == 0);
        EXPECT(out.rows > 0);
        std::printf("whirl-host-bench selftest: %d rows, %d work mismatches, %d failed\n", out.rows, out.work_mismatch, g_fail);
        return g_fail ? 1 : 0;
    }
    std::printf("whirl-host-bench: %d rows%s\n", out.rows, g_fail ? ", CHECK FAILED" : "");
    return g_fail || out.lost ? 1 : 0;
}
