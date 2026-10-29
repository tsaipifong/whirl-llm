// whirl-iso-device-profile-tests: the header-only module device.profile (include/whirl/device_profile.h, the per-architecture profile rows)
// compiled with its one declared dependency header (whirl/hip.h: hip::Arch) and linked to nothing (ARCH-1 8.5 isolation, arch025-iso4).
// testProfiles is the case of tests/device/device_tests.cpp (verbatim), which also drives the null backend and cannot be built alone.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/device_profile.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

#define CHECK(c) ISO_CHECK(c)

void testProfiles() {
    using whirl::hip::Arch;
    namespace d = whirl::device;
    const d::Profile& a = d::profile(Arch::gfx1201);
    const d::Profile& b = d::profile(Arch::gfx1151);
    CHECK(&a == &d::k_profile_gfx1201 && &b == &d::k_profile_gfx1151);
    CHECK(a.arch == Arch::gfx1201 && b.arch == Arch::gfx1151);
    // R9700 (gfx1201): discrete, the reference behaviour
    CHECK(!a.uma_no_vram_fit_check && !a.gemvh_grid_by_rows && !a.h16_from_candidates && !a.gemms_tie_larger_bm);
    CHECK(!a.attn_dq4_dynamic_splits && !a.tune_fast && !a.gdn_strix && !a.numerics_8060s && !a.mtp_costly_verify_rows);
    CHECK(std::string_view(a.tune_cache_suffix).empty());
    CHECK(std::string_view(a.tune_time_hint) == "2 minutes (a few seconds for MXFP4 models)");
    CHECK(a.prefill_exec_moe == 2048 && !a.prefill_fill_batch && a.vision_attn_threads == 128);
    const std::array<std::uint8_t, 17> ra = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    CHECK(a.gemv_r == ra);
    // Radeon 8060S (gfx1151): UMA iGPU
    CHECK(b.uma_no_vram_fit_check && b.gemvh_grid_by_rows && b.h16_from_candidates && b.gemms_tie_larger_bm);
    CHECK(b.attn_dq4_dynamic_splits && b.tune_fast && b.gdn_strix && b.numerics_8060s && b.mtp_costly_verify_rows);
    CHECK(std::string_view(b.tune_cache_suffix) == "-gfx1151r2");
    CHECK(std::string_view(b.tune_time_hint) == "8 minutes on the Radeon 8060S (less for MXFP4 models)");
    CHECK(b.prefill_exec_moe == 4096 && b.prefill_fill_batch && b.vision_attn_threads == 256);
    const std::array<std::uint8_t, 17> rb = {1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    CHECK(b.gemv_r == rb);
    // constexpr: the field reads fold like the arch comparisons they replaced
    static_assert(d::profile(Arch::gfx1151).prefill_exec_moe == 4096 && !d::profile(Arch::gfx1201).tune_fast);
    // REVIEW-ARCH3 P3-3: a value with no row throws instead of getting the gfx1201 row (a new enumerator without a
    // case does not compile: tools/impact/profile_switch_check.ps1)
    bool threw = false;
    try {
        (void)d::profile(static_cast<Arch>(7));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    // arch025-M6: the two named host defaults are rows of this table (Model::arch / defaultGemvR / mtpDefaults' default
    // = the discrete reference row; the server's pre-load UMA estimate = the UMA row)
    static_assert(&d::profile(d::k_default_arch) == &d::k_profile_gfx1201);
    static_assert(&d::profile(d::k_uma_estimate_arch) == &d::k_profile_gfx1151);
    CHECK(d::k_default_arch == Arch::gfx1201 && d::k_uma_estimate_arch == Arch::gfx1151);
    CHECK(!d::profile(d::k_default_arch).uma_no_vram_fit_check && d::profile(d::k_uma_estimate_arch).uma_no_vram_fit_check);
    // every row is found again by its own arch (a row copied without changing .arch would fail here)
    for (const d::Profile* p : {&d::k_profile_gfx1201, &d::k_profile_gfx1151}) CHECK(&d::profile(p->arch) == p);
}

}  // namespace

int main(int argc, char** argv) { return iso::run("device.profile", {{"profiles", testProfiles}}, argc, argv); }
