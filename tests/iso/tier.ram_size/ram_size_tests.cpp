// whirl-iso-tier-ram-size-tests: the header-only module tier.ram_size (src/tier/ram_size.h, the pinned-RAM tier's default size) compiled with
// nothing but its own header and linked to nothing (ARCH-1 8.5 isolation, arch025-iso4). The integrated-GPU table is the
// RAM-tier column of testEnvCarveOutTable (tests/device/device_tests.cpp), which also drives the null backend and so cannot be built alone;
// the discrete rule, the explicit size and the availability caps are pinned here.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "tier/ram_size.h"

#include <string>

namespace {

namespace t = whirl::tier;
constexpr std::uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
constexpr std::uint64_t k_os_ram_96 = 33981464576ull;  // GlobalMemoryStatusEx ullTotalPhys at the 96 GB carve-out (measured)

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

void testExplicit() {
    t::RamTierInput in;
    in.explicit_mb = 0;
    t::RamTierSize r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 0 && r.reason == "--kv-ram-mb 0: host tiers off");
    in.explicit_mb = 6144;
    in.explicit_src = "WHIRL_KV_RAM_MB";
    in.total_phys = 8 * GiB;  // the explicit size wins over everything else
    in.integrated = true;
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 6144 && r.reason == "WHIRL_KV_RAM_MB 6144" && !r.avail_limited);
}

void testIntegratedTable() {
    // OS-visible RAM of a Strix Halo with the BIOS carve-out 0.5 / 16 / 32 / 64 / 96 GB (the measured table of testEnvCarveOutTable)
    const auto osram = [](double gb) { return static_cast<std::uint64_t>((128 - gb) * 1024) * MiB - 378273792ull; };
    const struct {
        double gb;
        std::uint64_t mb;
    } rows[] = {{0.5, 16384}, {16, 14336}, {32, 12288}, {64, 8192}, {96, 4096}};
    for (const auto& r : rows) {
        t::RamTierInput in;
        in.total_phys = osram(r.gb);
        in.avail_phys = in.total_phys / 4 * 3;
        in.integrated = true;
        const t::RamTierSize s = t::ramTierSize(in);
        ISO_CHECK(s.mb == r.mb && !s.avail_limited);
        ISO_CHECK(s.reason.find("integrated GPU: 1/8 of") == 0);
    }
    ISO_CHECK(osram(96) == k_os_ram_96);
    t::RamTierInput big;  // 1/8 of 192 GB is over the 16 GiB cap
    big.total_phys = 192 * GiB;
    big.integrated = true;
    const t::RamTierSize b = t::ramTierSize(big);
    ISO_CHECK(b.mb == 16384 && has(b.reason, "capped at 16 GiB"));
    t::RamTierInput lo;  // under 32 GB visible to Windows: off, with a reason
    lo.total_phys = 31000000000ull;
    lo.integrated = true;
    const t::RamTierSize l = t::ramTierSize(lo);
    ISO_CHECK(l.mb == 0 && has(l.reason, "under 32 GB): off; --kv-ram-mb N turns it on"));
    t::RamTierInput unk;
    unk.integrated = true;  // total unknown
    ISO_CHECK(t::ramTierSize(unk).mb == 0);
}

void testIntegratedEntryAndAvail() {
    t::RamTierInput in;
    in.integrated = true;
    in.total_phys = k_os_ram_96;  // 1/8 = 4 GiB
    in.entry_mb = 10240;          // one max-context entry larger than that: raised to it
    t::RamTierSize r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 10240 && has(r.reason, "raised to one max-context entry 10 GiB"));
    in.entry_mb = 1024;
    in.avail_phys = 8 * GiB;  // 1/4 of 8 GiB = 2 GiB < the 4 GiB wanted, and not below one entry: limited
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 2048 && r.avail_limited && has(r.reason, "limited to 2 GiB"));
    in.entry_mb = 4096;
    in.avail_phys = 6 * GiB;  // 1/4 = 1 GiB < one entry (4 GiB), and the entry is more than half (3 GiB): off
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 0 && r.avail_limited && has(r.reason, "would be more than half of it"));
    in.entry_mb = 2048;
    in.avail_phys = 6 * GiB;  // 1/4 = 1 GiB < the entry (2 GiB) <= half (3 GiB): one entry
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 2048 && r.avail_limited && has(r.reason, "at most half of the available RAM"));
}

void testDiscrete() {
    t::RamTierInput in;
    in.total_phys = 64 * GiB;  // 1/4 = 16 GiB
    ISO_CHECK(t::ramTierSize(in).mb == 16384);
    in.total_phys = 256 * GiB;  // 1/4 = 64 GiB, capped at 32 GiB
    t::RamTierSize r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 32768 && has(r.reason, "capped at 32 GiB"));
    in.total_phys = 16 * GiB;  // 1/4 = 4 GiB, raised to the 8 GiB floor
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 8192 && has(r.reason, "raised to the minimum 8 GiB"));
    in.floor_mb = 12288;  // one full-length session is bigger than 8 GiB: the floor wins
    ISO_CHECK(t::ramTierSize(in).mb == 12288);
    in.floor_mb = t::ram_auto_min_mb;
    in.total_phys = 0;  // unknown
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 8192 && r.reason.find("physical RAM unknown") == 0);
    in.total_phys = 64 * GiB;
    in.avail_phys = 20 * GiB;  // never more than half of the available RAM
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 10240 && r.avail_limited && has(r.reason, "half of the 20 GiB RAM available"));
    in.avail_phys = 1 * GiB;  // half is under 1 GiB: off
    r = t::ramTierSize(in);
    ISO_CHECK(r.mb == 0 && r.avail_limited && has(r.reason, "half of it would be under 1 GiB"));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("tier.ram_size",
                    {{"explicit", testExplicit},
                     {"integrated_table", testIntegratedTable},
                     {"integrated_entry_avail", testIntegratedEntryAndAvail},
                     {"discrete", testDiscrete}},
                    argc, argv);
}
