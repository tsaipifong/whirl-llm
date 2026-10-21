// REVIEW-GOV-4 independent test for REVIEW-GOV-3 question 4 (reviewer; written from the description only, before reading
// eb2bb25's test):
//   "with --mmproj the startup KV pool did not count the vision encoder, leaving ~30 MiB above the trigger after a 1920x1080
//    image on Ornith; text-only runs must keep exactly the same pool".
// CPU only. Drives the tree's poolbudget::visionReserve / kvAvail the way serveMain sizes the pool on a discrete GPU
// (server_main.cpp at cb6c9b4 :1356-1382: room = budget - used - margin, then minus the vision reserve) and checks the VRAM left
// after the encoder's real use, measured on the R9700 (raw: gov3 vis_*_summary / gov4 vis_*_g4a_summary health local_mib):
//   Swift  +900 MiB  (first image, 1920x1080 adds 0; gov3 29699 -> 30599)
//   Ornith +1007 MiB (first image +873, 1920x1080 +134; gov3 29696 -> 30569 -> 30703)
// Inputs to the reserve: encoder weights (Vision::totalBytes, from the file: Swift 887 MiB, Ornith 860 MiB) and the activation
// bytes of a max_tokens image beyond the lent prefill scratch as the startup log reports them (Swift 0, Ornith 405 MiB).
// fbdbe38 has no reserve at all (server_main.cpp:1305-1340 at fbdbe38 subtracts nothing for --mmproj): build with
// /DREVIEW_FBDBE38, which supplies that behaviour (reserve 0) because the API does not exist there.
// Checks: a. with --mmproj, after the measured encoder use at least margin/2 + 512 MiB stays free (not near the trigger);
//         b. the reserve covers the measured use (the full margin stays free, 16 MiB tolerance);
//         c. text-only (no --mmproj): the pool bytes are exactly the old ones for many inputs; no underflow.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/pool_budget.h"

#include <cstdint>
#include <cstdio>

#ifdef REVIEW_FBDBE38
namespace whirl::poolbudget {
struct VisionVram {
    std::uint64_t weights = 0, act_extra = 0;
};
inline std::uint64_t visionReserve(const VisionVram&) { return 0; }  // fbdbe38: the encoder is not counted
inline std::uint64_t kvAvail(std::uint64_t avail, std::uint64_t) { return avail; }
}  // namespace whirl::poolbudget
#endif

using namespace whirl::poolbudget;

namespace {
int g_pass = 0, g_fail = 0;
#define CHECK(cond, ...)                                           \
    do {                                                           \
        if (cond) {                                                \
            ++g_pass;                                              \
        } else {                                                   \
            ++g_fail;                                              \
            std::printf("  FAIL line %d: %s | ", __LINE__, #cond); \
            std::printf(__VA_ARGS__);                              \
            std::printf("\n");                                     \
        }                                                          \
    } while (0)
constexpr std::uint64_t MiB = 1ull << 20;
constexpr std::uint64_t budget = 31767 * MiB, margin = 2068 * MiB;

void model(const char* name, std::uint64_t used0_mib, std::uint64_t weights_mib, std::uint64_t act_mib, std::uint64_t vis_use_mib) {
    VisionVram v;
    v.weights = weights_mib * MiB;
    v.act_extra = act_mib * MiB;
    const std::uint64_t used0 = used0_mib * MiB;  // WDDM local before the pool
    const std::uint64_t room = budget - used0 - margin;
    const std::uint64_t res = visionReserve(v);
    const std::uint64_t pool = kvAvail(room, res);
    const std::uint64_t local_after = used0 + pool + vis_use_mib * MiB;  // pool fills its room (page rounding only adds free)
    const std::int64_t free_after = static_cast<std::int64_t>(budget) - static_cast<std::int64_t>(local_after);
    const std::int64_t to_trigger = free_after - static_cast<std::int64_t>(margin / 2);
    std::printf("%s --mmproj: reserve %llu MiB, pool room %llu MiB, after the encoder's measured %llu MiB: free %lld MiB, %lld MiB above the trigger\n",
                name, static_cast<unsigned long long>(res >> 20), static_cast<unsigned long long>(pool >> 20),
                static_cast<unsigned long long>(vis_use_mib), static_cast<long long>(free_after / static_cast<std::int64_t>(MiB)),
                static_cast<long long>(to_trigger / static_cast<std::int64_t>(MiB)));
    CHECK(to_trigger >= static_cast<std::int64_t>(512 * MiB), "%s: at least 512 MiB above the trigger after the image (%lld MiB)", name,
          static_cast<long long>(to_trigger / static_cast<std::int64_t>(MiB)));
    CHECK(free_after + static_cast<std::int64_t>(16 * MiB) >= static_cast<std::int64_t>(margin),
          "%s: the reserve covers the measured use (free %lld MiB vs margin 2068)", name,
          static_cast<long long>(free_after / static_cast<std::int64_t>(MiB)));
    // text-only: same inputs, no --mmproj
    const std::uint64_t text = kvAvail(room, visionReserve(VisionVram{}));
    CHECK(text == room, "%s text-only: pool room unchanged (%llu vs %llu)", name, static_cast<unsigned long long>(text),
          static_cast<unsigned long long>(room));
}

void textOnly() {
    std::printf("text-only: kvAvail(a, visionReserve({})) == a\n");
    int bad = 0;
    for (std::uint64_t a : {0ull, 1ull, 255ull, 4096ull, 1ull << 30, 9876543210ull, 13ull << 30, ~0ull >> 1})
        if (kvAvail(a, visionReserve(VisionVram{})) != a) ++bad;
    CHECK(bad == 0, "%d inputs changed", bad);
    CHECK(visionReserve(VisionVram{}) == 0, "reserve 0 without --mmproj");
#ifndef REVIEW_FBDBE38
    CHECK(kvAvail(100 * MiB, 200 * MiB) == 0, "no underflow when the reserve is larger than the room");
#endif
}
}  // namespace

int main() {
    // used before the pool: gov4 startup logs, "kv pool sizing: ... used X GiB" (Swift 17.27, Ornith 20.73 GiB)
    model("Swift", 17685, 887, 0, 900);
    model("Ornith", 21228, 860, 405, 1007);
    textOnly();
    std::printf("review_gov4 pool tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
