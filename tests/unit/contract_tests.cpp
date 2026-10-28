// WHIRL unit tests of module contract (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-contract-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/json.h"
#include "whirl/numerics.h"
#include "whirl/vram_limit.h"
#include "whirl/kernels_abi.h"
#include "whirl/kv_format.h"
#include "whirl/device_profile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// numerics modes: parsing, capability table, overrides, precise KV decisions (whirl/numerics.h)
void testNumerics() {
    namespace nu = whirl::numerics;
    using nu::Item;
    using nu::Mode;
    // parsing and resolution
    CHECK(nu::resolve(std::nullopt, std::nullopt).mode == Mode::balance);  // default: balance
    CHECK(nu::resolve(std::nullopt, std::nullopt).items == nu::balance_items);
    CHECK(nu::resolve(std::nullopt, std::nullopt).source == "default");
    // FIX-PS: selftest loads in the default mode (balance) and Model::checkPreciseDecode turns the
    // kernel-switching balance items off itself (fp8_prefill, moe_fp8, out_h16). A new balance
    // item fails here: add its switch to checkPreciseDecode (or the precise selftest measures it).
    CHECK(nu::balance_items ==
          (nu::bit(Item::fp8) | nu::bit(Item::moefp8) | nu::bit(Item::gdnwmma) | nu::bit(Item::h16) | nu::bit(Item::kvq8) | nu::bit(Item::specsample)));
    CHECK(nu::resolve(std::nullopt, std::string("")).mode == Mode::balance);
    CHECK(nu::resolve(std::string("precise"), std::nullopt).mode == Mode::precise);
    CHECK(nu::resolve(std::nullopt, std::string("balance")).mode == Mode::balance);
    CHECK(nu::resolve(std::nullopt, std::string("Balanced")).mode == Mode::balance);
    CHECK(nu::resolve(std::string("precise"), std::string("fast")).mode == Mode::precise);  // command line first
    {
        const nu::Request r = nu::parseMode("balance", "t");
        CHECK(r.items == nu::balance_items && !r.custom && r.has(Item::fp8) && r.has(Item::kvq8) && !r.has(Item::kvq4));
        const nu::Request f = nu::parseMode("fast", "t");
        CHECK(f.items == (nu::balance_items | nu::fast_only_items));
        const nu::Request c = nu::parseMode("balance:fp8, kvq8", "t");
        CHECK(c.custom && c.items == (nu::bit(Item::fp8) | nu::bit(Item::kvq8)));
        const nu::Request c2 = nu::parseMode("fast:kvq4", "t");
        CHECK(c2.mode == Mode::fast && c2.items == nu::bit(Item::kvq4));
        CHECK(nu::parseMode("precise", "t").items == 0);
    }
    auto throws = [](const char* s) {
        try {
            (void)nu::parseMode(s, "t");
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(throws("turbo"));
    CHECK(throws("balance:fp9"));
    CHECK(throws("precise:fp8"));
    CHECK(throws("balance:kvq4"));  // a fast-only item in balance
    // command-line flags
    {
        std::string sp;
        bool nx = false;
        CHECK(nu::modeFlag("--precise", &sp, &nx) && sp == "precise" && !nx);
        CHECK(nu::modeFlag("--balance", &sp, &nx) && sp == "balance");
        CHECK(nu::modeFlag("--balance=fp8,h16", &sp, &nx) && sp == "balance:fp8,h16");
        CHECK(nu::modeFlag("--fast", &sp, &nx) && sp == "fast");
        CHECK(nu::modeFlag("--mode", &sp, &nx) && nx);
        CHECK(nu::modeFlag("--mode=balanced", &sp, &nx) && sp == "balanced" && !nx);
        CHECK(!nu::modeFlag("--fastest", &sp, &nx) && !nu::modeFlag("--ctx", &sp, &nx));
        CHECK(nu::parseMode(sp, "t").mode == Mode::balance);
    }
    // capability table: R9700 (gfx1201) MXFP4 dense / MXFP4 MoE / Q4_K_M dense, 8060S
    nu::Target r97;
    r97.k_gemm8 = r97.k_gemm8_moe = r97.k_gdn_wmma = r97.k_kv_q8v = r97.k_kv_q8h = true;
    nu::Target swift = r97;
    swift.has_mxfp4 = true;
    nu::Target ornith = swift;
    ornith.moe = ornith.moe_mxfp4 = true;
    nu::Target q4 = r97;
    nu::Target s8060;
    s8060.gfx1151 = true;
    s8060.has_mxfp4 = true;
    const nu::Request bal = nu::parseMode("balance", "t"), pre{}, fast = nu::parseMode("fast", "t");
    {
        const nu::Plan p = nu::plan(pre, swift);
        for (const nu::ItemState& st : p.items) CHECK(!st.enabled && !st.requested);
        CHECK(nu::logLine(p).find("numerics: precise - f16/f32") == 0);
        CHECK(nu::logLine(p).find("q8dec") == std::string::npos && nu::logLine(p).find("f16 decode / verify activations") != std::string::npos);
        CHECK(!p.q8dec && nu::propsJson(p, "f16").find("\"decode\":\"f16\",\"always_on\":[]") != std::string::npos);
        CHECK(nu::modeLabel(p) == "precise");
        nu::Plan po = p;
        nu::setQ8dec(po, true, "1");  // user-requested int8 decode in precise: lossy override
        CHECK(po.q8dec && nu::modeLabel(po) == "precise+overrides" && nu::logLine(po).find("WHIRL_Q8DEC=1: q8dec on") != std::string::npos);
        nu::Plan pn = p;
        nu::setQ8dec(pn, false, "0");  // same as the default: no override
        CHECK(pn.overrides.empty() && nu::modeLabel(pn) == "precise");
        const nu::Plan pb = nu::plan(nu::parseMode("balance", "t"), swift);
        CHECK(pb.q8dec && nu::logLine(pb).find("decode: q8dec") != std::string::npos);
    }
    {
        const nu::Plan p = nu::plan(bal, swift);
        CHECK(p.on(Item::fp8) && !p.on(Item::moefp8) && p.on(Item::gdnwmma) && p.on(Item::h16) && p.on(Item::kvq8));
        CHECK(p.items[static_cast<std::size_t>(Item::moefp8)].note == "dense model");
        const nu::Plan po = nu::plan(bal, ornith);  // MODES-MXFP4: MoE balance = fp8 + moefp8 + gdnwmma + h16 + q8dec, f16 KV
        CHECK(po.on(Item::fp8) && po.on(Item::moefp8) && po.on(Item::gdnwmma) && po.on(Item::h16) && !po.on(Item::kvq8));
        CHECK(po.on(Item::specsample) && po.q8dec && po.q8dec_default);
        const nu::Plan pq = nu::plan(bal, q4);
        CHECK(!pq.on(Item::fp8) && pq.on(Item::gdnwmma) && pq.on(Item::h16) && pq.on(Item::kvq8));  // KG-1: dense Q4_K_M on the R9700
        CHECK(pq.items[static_cast<std::size_t>(Item::fp8)].note.find("no MXFP4") != std::string::npos);
        CHECK(!nu::plan(pre, q4).on(Item::gdnwmma) && !nu::plan(pre, q4).on(Item::h16));
        nu::Target q4moe = q4;  // MoE with non-MXFP4 experts: unchanged (f32 paths)
        q4moe.moe = true;
        const nu::Plan pqm = nu::plan(bal, q4moe);
        CHECK(!pqm.on(Item::gdnwmma) && !pqm.on(Item::h16) && pqm.items[static_cast<std::size_t>(Item::h16)].note.find("MoE") != std::string::npos);
        nu::Target q4s = q4;  // Radeon 8060S Q4_K_M: unchanged (no WMMA DeltaNet, no h16 for quantized weights)
        q4s.gfx1151 = true;
        q4s.k_gemm8 = q4s.k_gemm8_moe = q4s.k_gdn_wmma = false;
        const nu::Plan pqs = nu::plan(bal, q4s);
        CHECK(!pqs.on(Item::gdnwmma) && !pqs.on(Item::h16));
        const nu::Plan p8 = nu::plan(bal, s8060);  // never an error: skipped with a reason
        CHECK(!p8.on(Item::fp8) && !p8.on(Item::moefp8) && !p8.on(Item::gdnwmma) && p8.on(Item::h16) && p8.on(Item::kvq8));
        CHECK(p8.items[static_cast<std::size_t>(Item::fp8)].note.find("gfx1151") != std::string::npos);
        CHECK(p8.items[static_cast<std::size_t>(Item::kvq8)].note.rfind("q8", 0) == 0);
        const std::string l = nu::logLine(p8);
        CHECK(l.find("numerics: balance - enabled: h16") == 0 && l.find("skipped: fp8 (") != std::string::npos);
    }
    {
        const nu::Plan p = nu::plan(fast, swift);  // R9700: no q4 KV kernels, the rest not implemented
        CHECK(p.on(Item::fp8) && !p.on(Item::kvq4) && !p.on(Item::a4));
        CHECK(p.items[static_cast<std::size_t>(Item::kvq4)].note.find("8060S only") != std::string::npos);
        CHECK(p.items[static_cast<std::size_t>(Item::a4)].note.find("not yet implemented") != std::string::npos);
        CHECK(p.on(Item::relaxacc) && nu::logLine(p).find("fast runs as balance") == std::string::npos);
        CHECK(!nu::plan(bal, swift).on(Item::relaxacc) && !nu::plan(pre, swift).on(Item::relaxacc));
        CHECK(!nu::plan(fast, ornith).on(Item::relaxacc));  // MoE: skipped (no measured gain)
        {
            // FIX-NUM P3-7: relaxacc needs an MTP head; a run that cannot use it reports it skipped
            nu::Target nomtp = swift;
            nomtp.mtp = false;
            const nu::Plan pn = nu::plan(fast, nomtp);
            CHECK(!pn.on(Item::relaxacc));
            CHECK(pn.items[static_cast<std::size_t>(Item::relaxacc)].note.find("no MTP head") != std::string::npos);
            CHECK(nu::logLine(pn).find("fast runs as balance") != std::string::npos);
            nu::Plan pd = nu::plan(fast, swift);
            nu::skipItem(pd, Item::relaxacc, "--draft dflash");
            CHECK(!pd.on(Item::relaxacc) && nu::logLine(pd).find("relaxacc (--draft dflash)") != std::string::npos);
            CHECK(nu::logLine(pd).find("fast runs as balance") != std::string::npos);
            // FIX-NUM P3-5: kernel-path A/B switches are recorded as overrides
            nu::Plan ps = nu::plan(bal, swift);
            nu::noteSwitches(ps, [](const char*) -> std::optional<std::string> { return std::nullopt; });
            CHECK(ps.overrides.empty() && nu::modeLabel(ps) == "balance");
            nu::noteSwitches(ps, [](const char* n) -> std::optional<std::string> {
                if (std::string(n) == "GEMV_MAX") return std::string("8");
                if (std::string(n) == "FD_SPLITS") return std::string("128");
                if (std::string(n) == "NO_FUSE") return std::string("");
                return std::nullopt;
            });
            CHECK(ps.overrides.size() == 2 && nu::modeLabel(ps) == "balance+overrides");
            CHECK(nu::logLine(ps).find("WHIRL_GEMV_MAX=8") != std::string::npos && nu::logLine(ps).find("WHIRL_FD_SPLITS=128") != std::string::npos);
        }
        nu::Request fk = nu::parseMode("fast:kvq8", "t");  // custom fast list without relaxacc / kvq4
        CHECK(nu::logLine(nu::plan(fk, swift)).find("fast runs as balance") != std::string::npos);
        nu::Target t8 = s8060;
        t8.k_kv_q4 = true;
        const nu::Plan p8 = nu::plan(fast, t8);  // 8060S: kvq4 on
        CHECK(p8.on(Item::kvq4) && p8.on(Item::kvq8) && nu::logLine(p8).find("fast runs as balance") == std::string::npos);
        t8.kv_explicit = true;
        CHECK(!nu::plan(fast, t8).on(Item::kvq4));
        t8.kv_explicit = false;
        CHECK(!nu::plan(bal, t8).on(Item::kvq4));
    }
    // specsample: balance / fast item, off in precise, skipped without an MTP head or the kernel
    {
        CHECK(nu::plan(bal, swift).on(Item::specsample) && nu::plan(bal, ornith).on(Item::specsample));
        CHECK(nu::plan(fast, s8060).on(Item::specsample) && !nu::plan(pre, swift).on(Item::specsample));
        nu::Target nomtp = swift;
        nomtp.mtp = false;
        CHECK(!nu::plan(bal, nomtp).on(Item::specsample) &&
              nu::plan(bal, nomtp).items[static_cast<std::size_t>(Item::specsample)].note == "no MTP head");
        nu::Target nok = swift;
        nok.k_spec_sample = false;
        CHECK(!nu::plan(bal, nok).on(Item::specsample));
        CHECK(nu::parseMode("balance:specsample", "t").items == nu::bit(Item::specsample));
        nu::Plan pp = nu::plan(pre, swift);
        nu::applyOverride(pp, Item::specsample, "SPEC_SAMPLE", "1", true, true);
        CHECK(pp.on(Item::specsample) && pp.lossy_override);
        nu::Plan pb = nu::plan(bal, swift);
        nu::applyOverride(pb, Item::specsample, "SPEC_SAMPLE", "0", false, true);
        CHECK(!pb.on(Item::specsample) && !pb.lossy_override);
    }
    // MODES-MXFP4 mode table (owner 2026-10-16, supersedes MOE-BAL 2026-10-13): MoE balance has the
    // dense balance items (fp8 + moefp8 back); forModel moves nothing; dense unchanged
    {
        CHECK(nu::moe_fast_items == 0);
        const nu::Request rb = nu::forModel(bal, true);
        CHECK(rb.mode == Mode::balance && rb.items == nu::balance_items && rb.moved == 0);
        CHECK(rb.has(Item::fp8) && rb.has(Item::moefp8) && rb.has(Item::gdnwmma) && rb.has(Item::h16) && rb.has(Item::specsample));
        const nu::Request rb2 = nu::forModel(rb, true);  // idempotent
        CHECK(rb2.items == rb.items && rb2.moved == rb.moved);
        CHECK(nu::forModel(bal, false).items == nu::balance_items && nu::forModel(bal, false).moved == 0);  // dense: unchanged
        CHECK(nu::forModel(fast, true).items == fast.items && nu::forModel(fast, true).moved == 0);       // fast: unchanged
        CHECK(nu::forModel(pre, true).items == 0 && nu::forModel(pre, true).moved == 0);
        const nu::Request cust = nu::parseMode("balance:gdnwmma,h16", "t");  // an explicit list is honoured (the old MOE-BAL table)
        CHECK(nu::forModel(cust, true).items == cust.items && nu::forModel(cust, true).moved == 0);
        // R9700 Ornith (MXFP4 MoE): balance enables fp8 / moefp8, nothing is "MoE: fast only"
        const nu::Plan pb = nu::plan(bal, ornith);
        const nu::ItemState& f8 = pb.items[static_cast<std::size_t>(Item::fp8)];
        CHECK(f8.enabled && f8.requested && !f8.moved && pb.on(Item::moefp8));
        for (const nu::ItemState& st : pb.items) CHECK(!st.moved && st.note.find("fast only") == std::string::npos);
        const std::string l = nu::logLine(pb);
        CHECK(l.find("numerics: balance - enabled: fp8, moefp8, gdnwmma, h16") == 0 && l.find("fast only") == std::string::npos &&
              l.find("decode: q8dec") != std::string::npos && l.find("f16 decode / verify") == std::string::npos);
        const std::string j = nu::propsJson(pb, "f16");
        CHECK(j.find("\"enabled\":[\"fp8\",\"moefp8\",\"gdnwmma\",\"h16\",\"specsample\"]") != std::string::npos &&
              j.find("fast only") == std::string::npos);
        // MoE balance vs fast on the R9700: the same items except q4 KV
        nu::Target ornith_q4 = ornith;
        ornith_q4.k_kv_q4 = true;  // R9700 since FAST-1c
        const nu::Plan pf = nu::plan(fast, ornith_q4);
        CHECK(pf.on(Item::fp8) && pf.on(Item::moefp8) && pf.on(Item::gdnwmma) && pf.on(Item::h16) && pf.on(Item::kvq4));
        CHECK(!nu::plan(cust, ornith).on(Item::fp8) && !nu::plan(cust, ornith).on(Item::moefp8));
        // KV: MoE balance f16, MoE fast q4 (where the kernels exist), dense balance q8h
        CHECK(nu::chooseKv(rb, true, nu::KvCaps{true, true, true}, std::nullopt).kv == whirl::KvFormat::f16);
        CHECK(nu::chooseKv(nu::forModel(fast, true), true, nu::KvCaps{true, true, true}, std::nullopt).kv == whirl::KvFormat::q4);
        CHECK(nu::chooseKv(nu::forModel(bal, false), false, nu::KvCaps{true, true, true}, std::nullopt).kv == whirl::KvFormat::q8h);
        // a user override turning fp8 off in MoE balance is a (non-lossy) override; turning it on adds nothing
        nu::Plan po = nu::plan(bal, ornith);
        nu::applyOverride(po, Item::fp8, "FP8", "0", false, true);
        CHECK(!po.on(Item::fp8) && !po.lossy_override);
        // dense: unchanged (MXFP4 dense keeps fp8 in balance; Q4_K_M has no fp8)
        CHECK(nu::plan(bal, swift).on(Item::fp8) && !nu::plan(bal, swift).items[static_cast<std::size_t>(Item::fp8)].moved);
        const nu::Plan pq = nu::plan(bal, q4);
        CHECK(!pq.on(Item::fp8) && pq.on(Item::gdnwmma) && pq.on(Item::h16) && pq.on(Item::kvq8) && !pq.items[0].moved);
        // Radeon 8060S MoE: no fp8 WMMA there, so fp8 / moefp8 stay skipped with the hardware reason and
        // the enabled set is the same as under MOE-BAL (h16, specsample; f16 KV): 8060S MoE balance unchanged
        nu::Target s8m = s8060;
        s8m.moe = s8m.moe_mxfp4 = true;
        const nu::Plan p8 = nu::plan(bal, s8m);
        CHECK(!p8.on(Item::fp8) && !p8.on(Item::moefp8) && !p8.on(Item::gdnwmma) && p8.on(Item::h16) && !p8.on(Item::kvq8) && p8.on(Item::specsample));
        CHECK(p8.items[static_cast<std::size_t>(Item::fp8)].note.find("gfx1151") != std::string::npos);
        CHECK(p8.items[static_cast<std::size_t>(Item::moefp8)].note.find("gfx1151") != std::string::npos);
        std::uint32_t en_old = 0, en_new = 0;  // old = the MOE-BAL table (gdnwmma, h16, specsample)
        const nu::Plan p8o = nu::plan(nu::parseMode("balance:gdnwmma,h16,specsample", "t"), s8m);
        for (std::size_t i = 0; i < nu::n_items; ++i) {
            if (p8o.items[i].enabled) en_old |= 1u << i;
            if (p8.items[i].enabled) en_new |= 1u << i;
        }
        CHECK(en_old == en_new);
        CHECK(p8.q8dec && p8o.q8dec);  // MOE-BAL-3: q8dec on in 8060S MoE balance too
    }
    // MOE-BAL-3 (2026-10-15): q8dec back in MoE balance (MOE-BAL-2 had made it fast-only); the
    // decode activation form depends only on the mode, never on the model type
    {
        CHECK(!nu::q8decDefault(pre, false) && !nu::q8decDefault(pre, true));
        CHECK(nu::q8decDefault(bal, false) && nu::q8decDefault(bal, true));
        CHECK(nu::q8decDefault(fast, false) && nu::q8decDefault(fast, true));
        CHECK(nu::q8decDefault(nu::forModel(bal, true), true));  // after the loader's forModel too
        const nu::Request cust = nu::parseMode("balance:fp8,moefp8,gdnwmma,h16,specsample", "t");
        CHECK(nu::q8decDefault(cust, true));
        nu::Target s8m = s8060;
        s8m.moe = s8m.moe_mxfp4 = true;
        // dense and MoE, R9700 and 8060S, balance and fast: int8 decode, nothing about q8dec skipped
        for (const nu::Target& t : {swift, q4, s8060, ornith, s8m}) {
            for (const nu::Request& r : {bal, fast}) {
                const nu::Plan p = nu::plan(r, t);
                CHECK(p.q8dec && p.q8dec_default && nu::logLine(p).find("decode: q8dec") != std::string::npos);
                CHECK(nu::logLine(p).find("q8dec (MoE") == std::string::npos && nu::logLine(p).find("f16 decode / verify") == std::string::npos);
                const std::string j = nu::propsJson(p, t.moe ? "f16" : "q8h");
                CHECK(j.find("\"decode\":\"q8dec\",\"always_on\":[{\"item\":\"q8dec\"") != std::string::npos &&
                      j.find("{\"item\":\"q8dec\",\"reason\"") == std::string::npos);
                CHECK(json::parse(j).isObject());
                CHECK(nu::modeLabel(p) == (r.mode == nu::Mode::balance ? "balance" : "fast"));
            }
            const nu::Plan pp = nu::plan(pre, t);
            CHECK(!pp.q8dec && !pp.q8dec_default);
        }
        for (const nu::Target& t : {ornith, s8m}) {
            const nu::Plan pb = nu::plan(bal, t);
            // WHIRL_Q8DEC=0 in MoE balance: f16 decode, a (non-lossy) override
            nu::Plan p0 = pb;
            nu::setQ8dec(p0, false, "0");
            CHECK(!p0.q8dec && !p0.lossy_override && p0.overrides.size() == 1 && nu::modeLabel(p0) == "balance");
            CHECK(nu::logLine(p0).find("WHIRL_Q8DEC=0") != std::string::npos);
            nu::Plan p1 = pb;
            nu::setQ8dec(p1, true, "1");  // same as the default: no override
            CHECK(p1.overrides.empty() && nu::modeLabel(p1) == "balance");
        }
        // dense balance WHIRL_Q8DEC=0: off, not lossy (unchanged)
        nu::Plan pd = nu::plan(bal, swift);
        nu::setQ8dec(pd, false, "0");
        CHECK(!pd.q8dec && !pd.lossy_override && pd.overrides.size() == 1 && nu::logLine(pd).find("q8dec (MoE") == std::string::npos);
    }
    // per-item environment overrides
    {
        nu::Plan p = nu::plan(pre, swift);
        nu::applyOverride(p, Item::fp8, "FP8", "1", true, true);
        CHECK(p.on(Item::fp8) && p.lossy_override && p.overrides.size() == 1 && nu::modeLabel(p) == "precise+overrides");
        CHECK(p.overrides[0].find("lossy, user-requested") != std::string::npos);
        nu::Plan pq = nu::plan(pre, q4);
        nu::applyOverride(pq, Item::fp8, "FP8", "1", true, false);  // not applicable: no change
        CHECK(!pq.on(Item::fp8) && pq.overrides.empty() && !pq.lossy_override);
        nu::Plan pb = nu::plan(bal, swift);
        nu::applyOverride(pb, Item::fp8, "FP8", "0", false, true);
        CHECK(!pb.on(Item::fp8) && !pb.lossy_override && pb.overrides.size() == 1 && nu::modeLabel(pb) == "balance");
        nu::applyOverride(pb, Item::gdnwmma, "GDN_WMMA", "1", true, true);  // already on: nothing
        CHECK(pb.overrides.size() == 1);
        const std::string j = nu::propsJson(pb, "q8v");
        CHECK(j.find("\"mode\":\"balance\"") != std::string::npos && j.find("\"kv\":\"q8v\"") != std::string::npos);
        CHECK(j.find("\"enabled\":[\"gdnwmma\",\"h16\",\"kvq8\",\"specsample\"]") != std::string::npos);
        CHECK(j.find("{\"item\":\"fp8\",\"reason\":\"off by WHIRL_FP8=0\"}") != std::string::npos && j.find("{\"item\":\"moefp8\",\"reason\":\"dense model\"}") != std::string::npos);
        const json::Value v = json::parse(nu::propsJson(nu::plan(fast, s8060), "f16"));
        CHECK(v.isObject());
    }
    // precise KV, CLI: fits / shrinks (implicit) / refuses (explicit or below the minimum)
    {
        const std::uint64_t GiB = 1ull << 30, pt = 65536;  // 64 KiB/token
        nu::CtxFit a = nu::cliCtxFit(8192, false, 100, 10 * GiB, GiB, pt, 256);
        CHECK(!a.shrunk && !a.refuse && a.ctx == 8192);
        nu::CtxFit b = nu::cliCtxFit(131072, false, 1000, 5 * GiB, GiB, pt, 256);  // 4 GiB -> 65536 tokens
        CHECK(b.shrunk && !b.refuse && b.ctx == 65536 && b.msg.find("--balance") != std::string::npos);
        nu::CtxFit c = nu::cliCtxFit(131072, true, 1000, 5 * GiB, GiB, pt, 256);
        CHECK(c.refuse && c.msg.find("--balance") != std::string::npos && c.msg.find("--ctx") != std::string::npos);
        nu::CtxFit d = nu::cliCtxFit(131072, false, 70000, 5 * GiB, GiB, pt, 256);  // shrink would cut the prompt
        CHECK(d.refuse);
    }
    // precise KV, server: f16 pool, context shrink / refusal
    {
        const std::uint64_t GiB = 1ull << 30, pt = 69632;  // ~68 KiB/token (27B f16)
        // R9700 32 GB Swift: ~11.24 GiB -> ~173k tokens >= one 131072 request (no shrink)
        nu::PoolFit a = nu::serverPoolFit(11 * GiB + GiB / 4, pt, std::nullopt, false, 131072, false, 256, 8 * 262144);
        CHECK(!a.refuse && !a.shrunk_ctx && a.slot_ctx == 131072 && a.pool >= 131072 && a.pool % 256 == 0 && a.msg.empty());
        // 16 GB card: 4 GiB -> 61k tokens: per-request context lowered (not given) / refused (given)
        nu::PoolFit b = nu::serverPoolFit(4 * GiB, pt, std::nullopt, false, 131072, false, 256, 8 * 262144);
        CHECK(!b.refuse && b.shrunk_ctx && b.slot_ctx == b.pool && b.pool == 61440 && b.msg.find("--balance") != std::string::npos);
        nu::PoolFit c = nu::serverPoolFit(4 * GiB, pt, std::nullopt, false, 131072, true, 256, 8 * 262144);
        CHECK(c.refuse && c.msg.find("--ctx-per-slot") != std::string::npos);
        // explicit --ctx that does not fit: refused; the UMA default pool: shrunk
        nu::PoolFit d = nu::serverPoolFit(4 * GiB, pt, 131072u, true, 131072, false, 256, 8 * 262144);
        CHECK(d.refuse && d.msg.find("--ctx") != std::string::npos);
        nu::PoolFit e = nu::serverPoolFit(4 * GiB, pt, 262144u, false, 131072, false, 256, 8 * 262144);
        CHECK(!e.refuse && e.shrunk_pool && e.shrunk_ctx && e.pool == 61440 && e.slot_ctx == 61440);
        nu::PoolFit f = nu::serverPoolFit(4 * GiB, pt, 32768u, true, 32768, false, 256, 8 * 262144);  // fits
        CHECK(!f.refuse && f.pool == 32768 && f.slot_ctx == 32768 && f.msg.empty());
        nu::PoolFit g = nu::serverPoolFit(GiB / 8, pt, std::nullopt, false, 131072, false, 256, 8 * 262144);  // < 4096 tokens
        CHECK(g.refuse && g.msg.find("--balance") != std::string::npos);
    }
}

// arch025-N3 pin (PLAN N3, R15): the pre-load VRAM estimates choose the KV format with numerics::kvCapsEstimate (no
// code object loaded yet: Model::load's UMA check and the server's UMA estimate with integrated = true, the server's
// dGPU estimate with false); Model::load then chooses with the real caps of the code object (KvCaps{caps.kv_q8h,
// caps.kv_q8v, caps.kv_q4, profile.numerics_8060s}, loader.cpp). Both must agree in every DeviceProfile x mode x model
// x WHIRL_KV cell, or the estimate sizes a different KV than the load allocates. The code-object caps per GPU are the
// S0b-2 "D kv: site=loader.load" lines (scratch/arch025/s0b/s0b2: r9700 caps=q8h:1,q8v:1,q4:1,fast_q8h:0; 8060s
// caps=q8h:1,q8v:1,q4:1,fast_q8h:1). A disagreeing cell is an existing bug candidate (bug-fix line), not a refactor.
void testKvEstimateCaps() {
    namespace nu = whirl::numerics;
    using whirl::KvFormat;
    struct Gpu {
        const whirl::device::Profile& prof;
        bool integrated;  // which estimate reaches it (server_main.cpp: info.integrated; loader: umaMemory().integrated)
        nu::KvCaps code;  // the code object's KV kernels (Caps::probe), measured: S0b-2 dumps
    };
    const Gpu gpus[] = {
        {whirl::device::k_profile_gfx1201, false, nu::KvCaps{true, true, true, false}},
        {whirl::device::k_profile_gfx1151, true, nu::KvCaps{true, true, true, false}},
    };
    // (parseMode refuses "precise:<items>" and fast-only items in a balance list)
    const char* modes[] = {"precise", "balance", "fast", "balance:fp8", "balance:kvq8", "fast:kvq8", "fast:kvq4",
                           "fast:fp8", "balance:fp8,moefp8,gdnwmma,h16,specsample"};
    const std::optional<KvFormat> forced[] = {std::nullopt, KvFormat::f16, KvFormat::q8, KvFormat::q8h, KvFormat::q8v, KvFormat::q4};
    int cells = 0, differ = 0;
    for (const Gpu& g : gpus) {
        nu::KvCaps real = g.code;
        real.fast_q8h = g.prof.numerics_8060s;  // loader.cpp: device::profile(m.arch).numerics_8060s
        const nu::KvCaps est = nu::kvCapsEstimate(g.integrated);
        for (const char* ms : modes)
            for (bool moe : {false, true})
                for (const auto& f : forced) {
                    const nu::Request r = nu::parseMode(ms, "t");
                    const nu::KvChoice a = nu::chooseKv(r, moe, est, f), b = nu::chooseKv(r, moe, real, f);
                    ++cells;
                    if (a.kv != b.kv || a.debug != b.debug || a.why != b.why) {
                        ++differ;
                        std::fprintf(stderr, "  kv estimate != load: %s %s moe=%d forced=%s: est %s, load %s\n",
                                     g.integrated ? "uma" : "dgpu", ms, moe ? 1 : 0, f ? whirl::kvFormatName(*f) : "none",
                                     whirl::kvFormatName(a.kv), whirl::kvFormatName(b.kv));
                    }
                }
    }
    CHECK(cells == 2 * 9 * 2 * 6);
    CHECK(differ == 0);
    // arch025-N3-r1 (review C1, mutant n3-srv-est-caps): the three estimate sites call numerics::kvEstimate; each
    // site's format equals the load's choice with the real caps of the GPU that site runs on (server dGPU branch:
    // R9700; Model::load's UMA check and the server UMA branch: 8060S), and the "D kv:" site names are the old ones
    {
        struct Site {
            nu::KvEstSite site;
            const char* name;
            const Gpu& gpu;
        };
        const Site sites[] = {{nu::KvEstSite::loader_uma, "loader.uma_est", gpus[1]},
                              {nu::KvEstSite::server_dgpu, "server.est_dgpu", gpus[0]},
                              {nu::KvEstSite::server_uma, "server.est_uma", gpus[1]}};
        int site_cells = 0, site_differ = 0;
        for (const Site& s : sites) {
            CHECK(nu::kvEstSiteName(s.site) == std::string_view(s.name));
            CHECK(nu::kvEstSiteIntegrated(s.site) == s.gpu.integrated);
            nu::KvCaps real = s.gpu.code;
            real.fast_q8h = s.gpu.prof.numerics_8060s;
            for (const char* ms : modes)
                for (bool moe : {false, true})
                    for (const auto& f : forced) {
                        const nu::Request r = nu::parseMode(ms, "t");
                        ++site_cells;
                        if (nu::kvEstimate(s.site, r, moe, f) != nu::chooseKv(r, moe, real, f).kv) {
                            ++site_differ;
                            std::fprintf(stderr, "  kvEstimate(%s) != load: %s moe=%d forced=%s\n", s.name, ms, moe ? 1 : 0,
                                         f ? whirl::kvFormatName(*f) : "none");
                        }
                    }
        }
        CHECK(site_cells == 3 * 9 * 2 * 6);
        CHECK(site_differ == 0);
        // the server estimates' WHIRL_KV reading: unset / "auto" / unknown -> no request (Model::load refuses the
        // unknown name later), a known name -> that format
        CHECK(!nu::kvEstimateRequest(std::nullopt) && !nu::kvEstimateRequest(std::string("auto")));
        CHECK(!nu::kvEstimateRequest(std::string("q5")) && !nu::kvEstimateRequest(std::string("")) && !nu::kvEstimateRequest(std::string("AUTO")));
        for (const KvFormat k : {KvFormat::f16, KvFormat::q8, KvFormat::q8h, KvFormat::q8v, KvFormat::q4})
            CHECK(nu::kvEstimateRequest(std::string(whirl::kvFormatName(k))) == k);
    }
    // the estimate caps themselves (the literals they replaced at loader.cpp:492, server_main.cpp:810 / :922)
    CHECK(nu::kvCapsEstimate(false).q8h && nu::kvCapsEstimate(false).q8v && nu::kvCapsEstimate(false).q4 && !nu::kvCapsEstimate(false).fast_q8h);
    CHECK(nu::kvCapsEstimate(true).q8h && nu::kvCapsEstimate(true).q8v && nu::kvCapsEstimate(true).q4 && nu::kvCapsEstimate(true).fast_q8h);
    // the KvFormat names and properties (the former Model bools: loader.cpp:543-546 before N3)
    for (const KvFormat k : {KvFormat::f16, KvFormat::q8, KvFormat::q8h, KvFormat::q8v, KvFormat::q4}) {
        CHECK(whirl::kvFormatFromName(whirl::kvFormatName(k)) == k);
        CHECK(whirl::kvQuant(k) == (k != KvFormat::f16));
        CHECK(whirl::kvRotated(k) == (k == KvFormat::q8h || k == KvFormat::q4));
        CHECK(whirl::kvKf16(k) == (k == KvFormat::q8v));
        CHECK(whirl::kvIs4bit(k) == (k == KvFormat::q4));
        // kernels::Caps::supports == the UnsupportedKvFormat rule of loader.cpp before N3
        for (int m = 0; m < 8; ++m) {
            whirl::kernels::Caps c;
            c.kv_q8h = (m & 1) != 0;
            c.kv_q8v = (m & 2) != 0;
            c.kv_q4 = (m & 4) != 0;
            const bool q8 = k != KvFormat::f16, rot = k == KvFormat::q8h || k == KvFormat::q4, kf16 = k == KvFormat::q8v, q4 = k == KvFormat::q4;
            const bool refused = (kf16 && !c.kv_q8v) || (q4 && !c.kv_q4) || (rot && !q4 && !c.kv_q8h);
            (void)q8;
            CHECK(c.supports(k) == !refused);
        }
    }
    static_assert(static_cast<int>(KvFormat::f16) == 0 && static_cast<int>(KvFormat::q4) == 4);  // env table kvModeFromEnv numbering
}

// KV format (BAL-Q8): one fixed format per mode x model type, never by what fits
void testKvChoice() {
    namespace nu = whirl::numerics;
    using K = whirl::KvFormat;
    const nu::Request pre = nu::parseMode("precise", "t"), bal = nu::parseMode("balance", "t"), fast = nu::parseMode("fast", "t");
    const nu::KvCaps noq4{true, true, false};   // a code object without q4 (gfx1201 before FAST-1c)
    const nu::KvCaps r9700{true, true, true};   // gfx1201: q8h / q8v / q4 (FAST-1c)
    const nu::KvCaps s8060{true, true, true};   // gfx1151: q4 too
    for (const nu::KvCaps& c : {noq4, r9700, s8060}) {
        for (bool moe : {false, true}) {
            CHECK(nu::chooseKv(pre, moe, c, std::nullopt).kv == K::f16);
            CHECK(nu::chooseKv(bal, moe, c, std::nullopt).kv == (moe ? K::f16 : K::q8h));
            // fast: q4 where the kernels exist; else f16 (no int8 stand-in)
            CHECK(nu::chooseKv(fast, moe, c, std::nullopt).kv == (c.q4 ? K::q4 : K::f16));
            CHECK(!nu::chooseKv(bal, moe, c, std::nullopt).debug);
        }
    }
    CHECK(nu::chooseKv(fast, false, noq4, std::nullopt).why.find("no q4 KV kernels") != std::string::npos);
    CHECK(nu::chooseKv(fast, false, r9700, std::nullopt).kv == K::q4);
    CHECK(nu::chooseKv(bal, false, r9700, std::nullopt).why == "balance, dense model: q8h");
    // a code object without q8h: plain q8 (never q8v)
    CHECK(nu::chooseKv(bal, false, nu::KvCaps{false, true, false}, std::nullopt).kv == K::q8);
    // custom item lists: kvq8 decides for balance, kvq4 for fast
    CHECK(nu::chooseKv(nu::parseMode("balance:fp8", "t"), false, r9700, std::nullopt).kv == K::f16);
    CHECK(nu::chooseKv(nu::parseMode("fast:kvq8", "t"), false, s8060, std::nullopt).kv == K::q8h);
    // RC6: gfx1151 (fast_q8h) fast on a dense model -> q8h; MoE keeps q4; the R9700 keeps q4; WHIRL_KV still wins
    {
        const nu::KvCaps s8060f{true, true, true, true};
        CHECK(nu::chooseKv(fast, false, s8060f, std::nullopt).kv == K::q8h);
        CHECK(nu::chooseKv(fast, false, s8060f, std::nullopt).why == "fast, dense model: q8h (Radeon 8060S: faster decode than q4)");
        CHECK(nu::chooseKv(fast, true, s8060f, std::nullopt).kv == K::q4);
        CHECK(nu::chooseKv(bal, false, s8060f, std::nullopt).kv == K::q8h);
        CHECK(nu::chooseKv(pre, false, s8060f, std::nullopt).kv == K::f16);
        CHECK(nu::chooseKv(fast, false, s8060f, K::q4).kv == K::q4);
        CHECK(nu::chooseKv(fast, false, nu::KvCaps{false, true, true, true}, std::nullopt).kv == K::q4);  // no q8h kernels: q4
        nu::Target t8;
        t8.gfx1151 = true;
        t8.k_kv_q8h = true;
        t8.k_kv_q4 = true;
        t8.fast_kv_q8h = true;
        const nu::Plan pd = nu::plan(fast, t8);
        CHECK(!pd.on(nu::Item::kvq4));
        CHECK(nu::logLine(pd).find("kvq4 (Radeon 8060S dense models: q8h KV") != std::string::npos);
        t8.moe = true;
        CHECK(nu::plan(nu::forModel(fast, true), t8).on(nu::Item::kvq4));
        // GDN-8S default: gdnwmma reported enabled when the strix prefill runs
        nu::Target tg;
        tg.gfx1151 = true;
        // (arch025-N2b: plan decides the strix prefill itself from the device row, kernels and shape; was Target::gdn_strix)
        tg.strix_profile = tg.k_gdn_sx = tg.gdn_strix_shape = true;
        CHECK(nu::plan(bal, tg).on(nu::Item::gdnwmma) && nu::plan(bal, tg).flags.gdn_strix);
        tg.strix_profile = false;
        CHECK(!nu::plan(bal, tg).on(nu::Item::gdnwmma));
    }
    // WHIRL_KV: debug override, any mode
    const nu::KvChoice o = nu::chooseKv(pre, false, r9700, K::q8v);
    CHECK(o.kv == K::q8v && o.debug && o.why.find("WHIRL_KV=q8v") != std::string::npos);
    CHECK(whirl::kvFormatFromName("q8h") == K::q8h && !whirl::kvFormatFromName("auto") && !whirl::kvFormatFromName("q5"));

    // does not fit: the chosen format only, shrink (implicit) or refuse (explicit); 32 GB vs 16 GB (simulated)
    const std::uint64_t GiB = 1ull << 30;
    const std::uint64_t f16_pt = 65536, q8h_pt = 34816;  // ~64 / 34 KiB/token
    const nu::KvFitLabel lb_bal = nu::kvFitLabel(bal, false, K::q8h);
    CHECK(lb_bal.mode == "balance" && lb_bal.fmt == "q8h" && lb_bal.alt.empty());
    CHECK(nu::kvFitLabel(pre, false, K::f16).alt.find("--balance") != std::string::npos);
    CHECK(nu::kvFitLabel(pre, true, K::f16).alt.empty());  // MoE: balance is f16 as well
    {
        // CLI: 32 GB card (~12 GiB left): 128k q8h fits, unchanged
        const nu::CtxFit a = nu::cliCtxFit(131072, false, 1000, 13 * GiB, GiB, q8h_pt, 256, lb_bal);
        CHECK(!a.shrunk && !a.refuse && a.ctx == 131072);
        // 16 GB card (~4 GiB left): shrunk in q8h (not given) / refused (given), never a lower format
        const nu::CtxFit b = nu::cliCtxFit(131072, false, 1000, 5 * GiB, GiB, q8h_pt, 256, lb_bal);
        CHECK(b.shrunk && b.ctx == 4 * GiB / q8h_pt / 256 * 256 && b.msg.find("balance mode keeps the KV cache in q8h") == 0);
        CHECK(b.msg.find("--balance") == std::string::npos);
        const nu::CtxFit c = nu::cliCtxFit(131072, true, 1000, 5 * GiB, GiB, q8h_pt, 256, lb_bal);
        CHECK(c.refuse && c.msg.find("q8h") != std::string::npos && c.msg.find("--ctx") != std::string::npos);
        // MoE balance (f16) on 16 GB
        const nu::CtxFit d = nu::cliCtxFit(131072, false, 1000, 5 * GiB, GiB, f16_pt, 256, nu::kvFitLabel(bal, true, K::f16));
        CHECK(d.shrunk && d.ctx == 65536 && d.msg.find("balance mode keeps the KV cache in f16") == 0);
    }
    {
        // server: 32 GB (~11 GiB) holds a 131072 request in q8h; 16 GB (~4 GiB): per-request ctx lowered / refused
        const nu::PoolFit a = nu::serverPoolFit(11 * GiB, q8h_pt, std::nullopt, false, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(!a.refuse && !a.shrunk_ctx && a.pool >= 131072 && a.msg.empty());
        const nu::PoolFit b = nu::serverPoolFit(4 * GiB, q8h_pt, std::nullopt, false, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(!b.refuse && b.shrunk_ctx && b.slot_ctx == b.pool && b.pool == 4 * GiB / q8h_pt / 256 * 256);
        CHECK(b.msg.find("balance mode keeps the KV cache in q8h") == 0);
        const nu::PoolFit c = nu::serverPoolFit(4 * GiB, q8h_pt, std::nullopt, false, 131072, true, 256, 8 * 262144, lb_bal);
        CHECK(c.refuse && c.msg.find("--ctx-per-slot") != std::string::npos);
        const nu::PoolFit d = nu::serverPoolFit(4 * GiB, q8h_pt, 262144u, true, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(d.refuse && d.msg.find("q8h") != std::string::npos);
        const nu::PoolFit e = nu::serverPoolFit(GiB / 16, q8h_pt, std::nullopt, false, 131072, false, 256, 8 * 262144, lb_bal);
        CHECK(e.refuse && e.msg.find("q8h KV cache (balance mode)") != std::string::npos);
    }
    // the card's defaults carry no KV choice: 16 GB and 32 GB pick the same format
    {
        vram::CardInput in;
        in.total = 16 * GiB;
        CHECK(vram::cardDefaults(in).note.find("KV") == std::string::npos);
    }
}

}  // namespace

void testNumericsPlan(int& pass, int& fail);  // tests/unit/numerics_plan_tests.cpp (arch025-N2a)

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_contract() {
    static const whirl::test::Suite s{"contract",
                                      {
                                          {"testNumerics", [] { testNumerics(); }},
                                          {"testNumericsPlan", [] { testNumericsPlan(whirl::test::g_pass, whirl::test::g_fail); }},
                                          {"testKvChoice", [] { testKvChoice(); }},
                                          {"testKvEstimateCaps", [] { testKvEstimateCaps(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_contract()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
