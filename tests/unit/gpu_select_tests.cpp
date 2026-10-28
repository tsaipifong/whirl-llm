// WHIRL unit tests of module gpu_select (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-gpu_select-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/gpu_select.h"

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
// ---------------------------------------------------------------------------
// s8-719 root fix: GPU selection (--device / -dev list), the HIP_VISIBLE_DEVICES value, user variables,
// the identity check, the probe protocol, the stable lock name, the device list (written from the bug
// description and the owner / lead requirements of 2026-10-23: a process that uses one GPU must make HIP
// see only that GPU; the user's own visibility variable wins; the option is a list; identical cards and
// same-architecture cards are never resolved by "first match")
void testGpuSelect() {
    using namespace whirl::gpusel;
    auto rec = [](int i, const char* name, const char* arch, int bus, bool integ, bool sup, std::uint64_t mib) {
        GpuRecord r;
        r.index = i;
        r.name = name;
        r.arch = arch;
        r.pci_bus = bus;
        r.luid = std::format("{:016x}", 0x182fa + 0x1000 * i);
        r.integrated = integ;
        r.supported = sup;
        r.total_mib = mib;
        r.compute_units = integ ? 20 : 32;
        r.uuid = std::format("{:08x}-0000-0000-0000-{:012x}", 0xa0 + i, 0xbeef00 + bus);  // per card
        return r;
    };
    const char* k8060 = "AMD Radeon(TM) 8060S Graphics";
    const char* kR97 = "AMD Radeon AI PRO R9700";
    const char* k9070 = "AMD Radeon RX 9070 XT";
    // this machine (scratch\s8_719\devices_all.txt): 0 = 8060S, 1 = R9700
    const std::vector<GpuRecord> two = {rec(0, k8060, "gfx1151", 0xc3, true, true, 110490), rec(1, kR97, "gfx1201", 0x63, false, true, 32624)};
    const std::vector<GpuRecord> igpu_only = {rec(0, k8060, "gfx1151", 0xc3, true, true, 110490)};
    const std::vector<GpuRecord> r9700_only = {rec(0, kR97, "gfx1201", 0x03, false, true, 32624)};
    const std::vector<GpuRecord> r9700_first = {rec(0, kR97, "gfx1201", 0x63, false, true, 32624), rec(1, k8060, "gfx1151", 0xc3, true, true, 110490)};
    const std::vector<GpuRecord> unsupported_first = {rec(0, "AMD Radeon RX 6800", "gfx1030", 0x10, false, false, 16368),
                                                      rec(1, k8060, "gfx1151", 0xc3, true, true, 110490)};
    // owner's 4-card host: 2x R9700 + 2x RX 9070 XT (all gfx1201), distinct PCI buses
    const std::vector<GpuRecord> four = {rec(0, kR97, "gfx1201", 0x03, false, true, 32624), rec(1, kR97, "gfx1201", 0x04, false, true, 32624),
                                         rec(2, k9070, "gfx1201", 0x07, false, true, 16304), rec(3, k9070, "gfx1201", 0x0a, false, true, 16304)};
    auto errOf = [](const std::vector<GpuRecord>& g, const char* spec) {
        try {
            (void)selectGpus(g, spec);
        } catch (const SelectError& e) {
            return std::string(e.what());
        } catch (...) {
            return std::string("other exception");
        }
        return std::string("no error");
    };
    auto has = [](const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; };

    // list parsing (llama.cpp -dev <dev1,dev2,..>)
    CHECK(parseDeviceList("").empty());
    CHECK(parseDeviceList("   ").empty());
    CHECK(parseDeviceList("8060s") == std::vector<std::string>{"8060s"});
    CHECK(parseDeviceList(" r9700 , 8060s ") == (std::vector<std::string>{"r9700", "8060s"}));
    CHECK(throws([] { parseDeviceList("r9700,"); }));
    CHECK(throws([] { parseDeviceList(",r9700"); }));
    CHECK(throws([] { parseDeviceList("0,,1"); }));

    // single choices: alias, index, llama.cpp-style name, arch, name part, the full name the device list prints
    CHECK(selectGpus(two, "8060s") == std::vector<int>{0});
    CHECK(selectGpus(two, "8060S") == std::vector<int>{0});
    CHECK(selectGpus(two, "igpu") == std::vector<int>{0});
    CHECK(selectGpus(two, "strix") == std::vector<int>{0});
    CHECK(selectGpus(two, "r9700") == std::vector<int>{1});
    CHECK(selectGpus(two, "R9700") == std::vector<int>{1});
    CHECK(selectGpus(two, "1") == std::vector<int>{1});
    CHECK(selectGpus(two, "0") == std::vector<int>{0});
    CHECK(selectGpus(two, "ROCm1") == std::vector<int>{1});
    CHECK(selectGpus(two, "rocm0") == std::vector<int>{0});
    CHECK(selectGpus(two, "HIP1") == std::vector<int>{1});
    CHECK(selectGpus(two, "gfx1201") == std::vector<int>{1});
    CHECK(selectGpus(two, "gfx1151") == std::vector<int>{0});
    CHECK(selectGpus(two, kR97) == std::vector<int>{1});
    CHECK(selectGpus(two, k8060) == std::vector<int>{0});
    CHECK(selectGpus(two, " r9700 ") == std::vector<int>{1});
    // PCI address forms (hex), stable when cards are added / removed
    // (decimal, as Task Manager > Performance > GPU > Physical location and Device Manager show it:
    // "PCI bus 99, device 0, function 0" -> pci:99:0.0; never hex)
    CHECK(selectGpus(two, "pci:99:0") == std::vector<int>{1});
    CHECK(selectGpus(two, "pci:99:0.0") == std::vector<int>{1});  // a pasted full form works too
    CHECK(selectGpus(two, "PCI:195:0") == std::vector<int>{0});
    CHECK(has(errOf(two, "pci:99"), "pci:99:0"));  // bus alone: error that says how to write it
    CHECK(has(errOf(two, "pci:100:0"), "pci:100:0"));
    CHECK(has(errOf(two, "pci:63:0"), "no GPU matches"));  // hex 0x63 = 99 is not accepted as 63
    CHECK_EQ(pciText(two[1]), std::string("99:0.0"));
    CHECK_EQ(pciLongText(two[1]), std::string("PCI bus 99, device 0, function 0"));
    {
        GpuRecord fn = two[1];
        fn.pci_device = 1;
        fn.pci_function = 2;
        CHECK_EQ(pciText(fn), std::string("99:1.2"));
        CHECK_EQ(pciLongText(fn), std::string("PCI bus 99, device 1, function 2"));
        const std::vector<GpuRecord> same_bus = {two[1], fn};
        CHECK(selectGpus(same_bus, "pci:99:1.2") == std::vector<int>{1});
        // the bus number is what users see and type; the full form only where two GPUs share a bus
        CHECK_EQ(pciId(two, two[1]), std::string("99:0"));
        CHECK_EQ(pciIdLong(two, two[1]), std::string("PCI bus 99, device 0"));
        CHECK_EQ(pciId(same_bus, same_bus[0]), std::string("99:0"));
        CHECK_EQ(pciId(same_bus, same_bus[1]), std::string("99:1"));
        CHECK_EQ(pciIdLong(same_bus, same_bus[1]), std::string("PCI bus 99, device 1"));
        CHECK(has(listingText(same_bus, 0), "pci:99:1 (PCI bus 99, device 1)"));
        CHECK(selectGpus(same_bus, "pci:99:1") == std::vector<int>{1});  // bus:device, as the list prints it
        CHECK(selectGpus(same_bus, "pci:99:0") == std::vector<int>{0});
        GpuRecord dom = two[1];
        dom.pci_domain = 1;
        CHECK_EQ(pciText(dom), std::string("1:99:0.0"));  // a non-zero PCI domain is shown in front
    }

    // default (no option; owner 2026-10-23 「預設用vram最大的卡」「不管是不是獨顯」): among the GPUs this build
    // has kernels for, the most GPU memory (an integrated GPU: its dedicated carve-out), discrete or
    // integrated alike; a tie -> the lowest index. On this host that is now the 8060S (was the R9700).
    CHECK(selectGpus(two, "") == std::vector<int>{0});
    CHECK(selectGpus(r9700_first, "") == std::vector<int>{1});
    CHECK(selectGpus(igpu_only, "") == std::vector<int>{0});
    CHECK(selectGpus(r9700_only, "") == std::vector<int>{0});
    CHECK(selectGpus(unsupported_first, "") == std::vector<int>{1});
    CHECK(selectGpus(four, "") == std::vector<int>{0});
    {
        const std::vector<GpuRecord> small_first = {rec(0, k9070, "gfx1201", 0x07, false, true, 16304), rec(1, kR97, "gfx1201", 0x03, false, true, 32624)};
        CHECK(selectGpus(small_first, "") == std::vector<int>{1});  // most memory, not index 0
        const std::vector<GpuRecord> twins = {rec(0, kR97, "gfx1201", 0x09, false, true, 32624), rec(1, kR97, "gfx1201", 0x03, false, true, 32624)};
        CHECK(selectGpus(twins, "") == std::vector<int>{0});  // identical cards: lowest index
    }
    CHECK_EQ(autoPick(two), 0);
    CHECK_EQ(autoPick(std::vector<GpuRecord>{rec(0, "AMD Radeon RX 6800", "gfx1030", 1, false, false, 16368)}), -1);
    CHECK(throws([&] { selectGpus(std::vector<GpuRecord>{rec(0, "AMD Radeon RX 6800", "gfx1030", 1, false, false, 16368)}, ""); }));
    CHECK(throws([&] { selectGpus(std::vector<GpuRecord>{}, ""); }));

    // the alias r9700 is the card's NAME, not gfx1201: an RX 9070 XT is gfx1201 too
    {
        const std::vector<GpuRecord> r97_and_9070 = {rec(0, k9070, "gfx1201", 0x07, false, true, 16304), rec(1, kR97, "gfx1201", 0x03, false, true, 32624)};
        CHECK(selectGpus(r97_and_9070, "r9700") == std::vector<int>{1});
        CHECK(selectGpus(r97_and_9070, "9070") == std::vector<int>{0});
        const std::string e = errOf(r97_and_9070, "gfx1201");  // a raw architecture is ambiguous here
        CHECK(has(e, "device 0") && has(e, "device 1") && has(e, "more than one GPU"));
        const std::vector<GpuRecord> only9070 = {rec(0, k9070, "gfx1201", 0x07, false, true, 16304)};
        CHECK(has(errOf(only9070, "r9700"), "no GPU matches"));
    }
    // the 4-card host
    {
        const std::string e = errOf(four, "r9700");
        CHECK(has(e, "more than one GPU"));
        CHECK(has(e, "device 0: AMD Radeon AI PRO R9700 (gfx1201), pci:3:0, 31.9 GiB"));
        CHECK(has(e, "device 1: AMD Radeon AI PRO R9700 (gfx1201), pci:4:0, 31.9 GiB"));
        CHECK(!has(e, "device 2:") && !has(e, "device 3:"));
        CHECK(has(e, "--device 0") && has(e, "--device pci:3:0"));
        const std::string e2 = errOf(four, "9070");
        CHECK(has(e2, "more than one GPU"));
        CHECK(has(e2, "device 2: AMD Radeon RX 9070 XT (gfx1201), pci:7:0, 15.9 GiB"));
        CHECK(has(e2, "device 3: AMD Radeon RX 9070 XT (gfx1201), pci:10:0, 15.9 GiB"));
        CHECK(!has(e2, "device 0:") && !has(e2, "device 1:"));
        CHECK(selectGpus(four, "2") == std::vector<int>{2});
        CHECK(selectGpus(four, "pci:10:0.0") == std::vector<int>{3});
        CHECK(selectGpus(four, "ROCm3") == std::vector<int>{3});
        CHECK(has(errOf(four, "gfx1201"), "more than one GPU"));
        // the list
        const std::string l = listingText(four, autoPick(four));
        CHECK(has(l, std::string("device 0: AMD Radeon AI PRO R9700 (gfx1201), 31.9 GiB, 32 CUs, pci:3:0 (PCI bus 3, device 0), uuid " + four[0].uuid + "  [supported] [default]")));
        CHECK(has(l, std::string("device 3: AMD Radeon RX 9070 XT (gfx1201), 15.9 GiB, 32 CUs, pci:10:0 (PCI bus 10, device 0), uuid " + four[3].uuid + "  [supported]\n")));
        CHECK(has(l, "can change") && has(l, "pci:") && has(l, "Physical location"));
        const std::string l2 = listingText(unsupported_first, autoPick(unsupported_first));
        CHECK(has(l2, std::string("device 0: AMD Radeon RX 6800 (gfx1030), 16.0 GiB, 32 CUs, pci:16:0 (PCI bus 16, device 0), uuid " + unsupported_first[0].uuid + "  [no GPU kernels for this architecture in this build]")));
        CHECK(has(l2, "[integrated: memory shared with the CPU] [default]"));
        // the PCI function number is never part of what users read (one card = one bus number)
        CHECK(!has(l, "function") && !has(errOf(four, "r9700"), "function") && !has(restrictionNote(four, {2}, ""), "function"));
        CHECK(!has(errOf(four, "rtx4090"), "function") && !has(l, ":0.0"));
    }

    // daisy-chained USB4 eGPUs: two identical R9700 (bus numbers given at connect time) + the 8060S
    {
        const std::vector<GpuRecord> chain = {rec(0, k8060, "gfx1151", 196, true, true, 110490), rec(1, kR97, "gfx1201", 71, false, true, 32624),
                                              rec(2, kR97, "gfx1201", 74, false, true, 32624)};
        const std::string e = errOf(chain, "r9700");
        CHECK(has(e, "more than one GPU") && has(e, "pci:71:0") && has(e, "pci:74:0") && !has(e, "pci:196:0"));
        CHECK(has(errOf(chain, "R9700"), "Physical location"));
        CHECK(selectGpus(chain, "pci:71:0") == std::vector<int>{1});
        CHECK(selectGpus(chain, "pci:74:0") == std::vector<int>{2});
        CHECK(selectGpus(chain, "pci:196:0") == std::vector<int>{0});
        CHECK(selectGpus(chain, "8060s") == std::vector<int>{0});
        CHECK(selectGpus(chain, "") == std::vector<int>{0});  // the 8060S: most memory
        CHECK(has(errOf(chain, "pci:74"), "pci:74:0"));
        const std::string l = listingText(chain, autoPick(chain));
        CHECK(has(l, "pci:71:0 (PCI bus 71, device 0)") && has(l, "pci:74:0 (PCI bus 74, device 0)") && has(l, "pci:196:0 (PCI bus 196, device 0)"));
    }

    // the default rule in one function (defaultGpu): most GPU memory, tie -> lowest index, kernels required
    {
        auto fit = [](std::vector<GpuRecord> g, int i, std::uint64_t mib) {
            g[i].fit_mib = mib;
            return g;
        };
        const std::vector<GpuRecord> host = fit(two, 0, 98099);  // this machine: 8060S dedicated 95.8 GiB, R9700 31.9 GiB
        CHECK_EQ(defaultGpu(host).index, 0);
        CHECK_EQ(defaultGpu(fit(two, 0, 16 * 1024)).index, 1);  // an 8060S with a 16 GiB carve-out: the R9700 has more
        CHECK_EQ(defaultGpu(r9700_only).index, 0);
        CHECK_EQ(defaultGpu(igpu_only).index, 0);
        CHECK_EQ(defaultGpu(four).index, 0);
        {
            const std::vector<GpuRecord> eq = {rec(0, k9070, "gfx1201", 9, false, true, 16304), rec(1, k9070, "gfx1201", 3, false, true, 16304)};
            CHECK_EQ(defaultGpu(eq).index, 0);  // equal memory: lowest index
            const std::vector<GpuRecord> big_unsup = {rec(0, "AMD Instinct MI300X", "gfx942", 5, false, false, 196608), rec(1, kR97, "gfx1201", 3, false, true, 32624)};
            CHECK_EQ(defaultGpu(big_unsup).index, 1);  // a larger card without kernels in this build is ignored
            CHECK_EQ(defaultGpu(std::vector<GpuRecord>{rec(0, "AMD Instinct MI300X", "gfx942", 5, false, false, 196608)}).index, -1);
        }
        // the startup line (owner 2026-10-23: usage only, no cause): "GPU pci:196:0 (NAME): most GPU memory (95.8 GiB);
        // use --device pci:<bus>:<device> to choose another (see --list-devices)"
        const std::vector<GpuRecord> h2 = fit({rec(0, k8060, "gfx1151", 196, true, true, 110490), rec(1, kR97, "gfx1201", 71, false, true, 32624)}, 0, 98099);
        const AutoChoice c = defaultGpu(h2);
        CHECK_EQ(c.index, 0);
        CHECK(has(c.reason, "most GPU memory (95.8 GiB)"));
        const std::string n = restrictionNote(h2, {0}, c.reason);
        CHECK_EQ(n, std::string("GPU pci:196:0 (AMD Radeon(TM) 8060S Graphics): most GPU memory (95.8 GiB); use --device "
                                "pci:<bus>:<device> to choose another (see --list-devices)"));
        CHECK(!has(n, "function"));
        const std::string ne = restrictionNote(h2, {1}, "");  // chosen with --device: no automatic reason
        CHECK_EQ(ne, std::string("GPU pci:71:0 (AMD Radeon AI PRO R9700): chosen with --device or WHIRL_DEVICE; use --device "
                                 "pci:<bus>:<device> to choose another (see --list-devices)"));
        // the list marks the same default and shows the dedicated memory of the iGPU
        CHECK(has(listingText(h2, defaultGpu(h2).index), "[integrated: memory shared with the CPU] [default]"));
        // the probe carries the usable memory
        const std::vector<GpuRecord> back = parseRecords(formatRecord(host[0]));
        CHECK(back.size() == 1 && back[0].fit_mib == host[0].fit_mib);
    }
    // a list of two: a clear error, never a silent fallback to one of them
    // owner 2026-10-23: WHIRL always used one GPU, so the error says only that (no "in this version", no "planned")
    for (const char* lst : {"r9700,8060s", "0,1"}) {
        const std::string msg = errOf(two, lst);
        CHECK(has(msg, "only one GPU can be given with --device"));
        CHECK(!has(msg, "in this version"));
        CHECK(!has(msg, "planned"));
        CHECK(!has(msg, "splitting"));
    }
    CHECK(throws([&] { selectGpus(two, "1,1"); }));
    // unknown name / index out of range: the error names the value and lists the GPUs present
    for (const char* bad : {"rtx4090", "7", "ROCm5", "gfx90a"}) {
        const std::string msg = errOf(two, bad);
        CHECK(has(msg, bad));
        CHECK(has(msg, "device 0: AMD Radeon(TM) 8060S Graphics (gfx1151)"));
        CHECK(has(msg, "device 1: AMD Radeon AI PRO R9700 (gfx1201)"));
    }

    // HIP_VISIBLE_DEVICES value (a list later)
    CHECK_EQ(visibleDevicesValue({1}), std::string("1"));
    CHECK_EQ(visibleDevicesValue({0}), std::string("0"));
    CHECK_EQ(visibleDevicesValue({0, 1}), std::string("0,1"));

    // a visibility variable the user set wins (never overridden); an empty value counts as set
    {
        std::map<std::string, std::string> envm;
        auto ge = [&](const char* k) -> const char* {
            auto it = envm.find(k);
            return it == envm.end() ? nullptr : it->second.c_str();
        };
        CHECK_EQ(userVisibilityVar(ge), std::string());
        envm["WHIRL_DEVICE"] = "8060s";
        CHECK_EQ(userVisibilityVar(ge), std::string());
        for (const char* k : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
            envm.clear();
            envm[k] = "0";
            CHECK_EQ(userVisibilityVar(ge), std::string(k));
        }
        envm.clear();
        envm["HIP_VISIBLE_DEVICES"] = "";
        CHECK_EQ(userVisibilityVar(ge), std::string("HIP_VISIBLE_DEVICES"));

        // arch025-c2fu: zero GPUs. No visibility variable = the real no-driver / no-GPU case, text unchanged.
        envm.clear();
        envm["WHIRL_DEVICE"] = "8060s";
        CHECK_EQ(noGpuText(ge), std::string("the AMD graphics driver reports no GPU"));
        // HIP_VISIBLE_DEVICES set but EMPTY (a script restore that left HIP_VISIBLE_DEVICES=, arch024-S14 / arch025-c2ctx):
        // the variable is the cause, so the text names it, says it is empty and how to remove it, and never sends the
        // user to a driver update
        envm.clear();
        envm["HIP_VISIBLE_DEVICES"] = "";
        {
            const std::string t = noGpuText(ge);
            CHECK(t.find("HIP_VISIBLE_DEVICES") != std::string::npos);
            CHECK(t.find("empty") != std::string::npos);
            CHECK(t.find("Remove-Item Env:HIP_VISIBLE_DEVICES") != std::string::npos);
            CHECK(t.find("driver reports no GPU") == std::string::npos);
            CHECK(t.find("update") == std::string::npos);
            CHECK(t.find("driver") == std::string::npos);  // arch025-X8bfix: the empty-value text stays driver-free
            CHECK(!t.empty() && t.back() != '.' && t.back() != '\n');
        }
        // blank counts as empty
        envm["HIP_VISIBLE_DEVICES"] = "  ";
        CHECK(noGpuText(ge).find("empty") != std::string::npos);
        // a value that names no GPU of this computer: the variable and its value are named
        envm["HIP_VISIBLE_DEVICES"] = "7";
        {
            const std::string t = noGpuText(ge);
            CHECK(t.find("HIP_VISIBLE_DEVICES=7") != std::string::npos);
            CHECK(t.find("empty") == std::string::npos);
            CHECK(t.find("driver reports no GPU") == std::string::npos);
        }
        // arch025-X8bfix (c2fu REVIEW C2): a NON-empty value may be a valid index (HIP_VISIBLE_DEVICES=0 from a ROCm
        // guide) while the driver is missing or broken; the runtime shows zero GPUs either way. The text says the
        // variable may hide the GPU (named, with how to remove it) AND that a driver problem is also possible (with the
        // AMD support page), so the user is not sent through two rounds to find a driver cause.
        for (const char* v : {"0", "7", "0,1"}) {
            envm.clear();
            envm["HIP_VISIBLE_DEVICES"] = v;
            const std::string t = noGpuText(ge);
            CHECK(t.find(std::string("HIP_VISIBLE_DEVICES=") + v) != std::string::npos);
            CHECK(t.find("may hide") != std::string::npos);
            CHECK(t.find("Remove-Item Env:HIP_VISIBLE_DEVICES") != std::string::npos);
            CHECK(t.find("driver") != std::string::npos);
            CHECK(t.find("https://www.amd.com/en/support") != std::string::npos);
            CHECK(t.find("Remove-Item") < t.find("driver"));  // the variable first (cheap to check), then the driver
            CHECK(!t.empty() && t.back() != '.' && t.back() != '\n');
        }
        envm.clear();
        envm["ROCR_VISIBLE_DEVICES"] = "1";
        CHECK(noGpuText(ge).find("ROCR_VISIBLE_DEVICES=1") != std::string::npos);
        CHECK(noGpuText(ge).find("driver") != std::string::npos);
        // the other visibility variables WHIRL honours are named the same way
        envm.clear();
        envm["CUDA_VISIBLE_DEVICES"] = "";
        CHECK(noGpuText(ge).find("CUDA_VISIBLE_DEVICES") != std::string::npos);
        CHECK(noGpuText(ge).find("Remove-Item Env:CUDA_VISIBLE_DEVICES") != std::string::npos);
    }

    // identity: the GPU left after the restriction must be the one chosen (never trust the ordinal alone)
    {
        GpuRecord after = two[1];
        after.index = 0;  // re-numbered by the restriction
        CHECK(sameGpu(two[1], after));
        CHECK(!sameGpu(two[0], after));
        GpuRecord bus = two[1];
        bus.pci_bus = 0x64;
        CHECK(!sameGpu(two[1], bus));
        GpuRecord luid = two[1];
        luid.luid = "0000000000000001";
        CHECK(!sameGpu(two[1], luid));
        GpuRecord noluid = two[1];
        noluid.luid = "";
        CHECK(sameGpu(two[1], noluid));  // LUID unknown on one side: the PCI address + name + arch decide
        GpuRecord nm = two[1];
        nm.name = k9070;
        CHECK(!sameGpu(two[1], nm));
        CHECK(!sameGpu(four[0], four[1]));  // identical cards differ by PCI bus
    }
    // uuid (stable per card when the runtime reports one): selection by full id or a unique prefix
    {
        CHECK(selectGpus(four, std::string("uuid:" + four[3].uuid).c_str()) == std::vector<int>{3});
        CHECK(selectGpus(four, "uuid:000000a2") == std::vector<int>{2});
        CHECK(selectGpus(four, "UUID:000000A1") == std::vector<int>{1});
        CHECK(has(errOf(four, "uuid:000000a"), "more than one GPU"));  // prefix of all four
        CHECK(has(errOf(four, "uuid:ffff"), "no GPU matches"));
        std::vector<GpuRecord> nouuid = two;
        for (auto& g : nouuid) g.uuid.clear();
        CHECK(has(errOf(nouuid, "uuid:000000a0"), "no GPU matches"));
        const std::string l = listingText(four, 0);
        CHECK(has(l, "uuid " + four[0].uuid));
        CHECK(!has(listingText(nouuid, 1), "uuid"));
        CHECK(!has(restrictionNote(four, {1}, ""), "uuid"));  // the startup line stays short (owner 2026-10-23)
    }
    // the GPU lock is keyed by the PCI address: index 0 after the restriction is not "GPU 0"
    {
        GpuRecord a0 = four[1];
        a0.index = 0;  // the same card as seen by a restricted process
        CHECK_EQ(lockName(a0), lockName(four[1]));
        CHECK(lockName(four[0]) != lockName(four[1]));
        GpuRecord b0 = four[2];
        b0.index = 0;
        CHECK(lockName(a0) != lockName(b0));  // two restricted processes on two GPUs do not block each other
        CHECK(has(lockName(four[1]), four[1].uuid));  // uuid first when reported
        GpuRecord np = four[1];
        np.uuid.clear();
        CHECK(has(lockName(np), "pci-4-0-0"));  // else the PCI address
        GpuRecord np0 = np;
        np0.index = 0;
        CHECK_EQ(lockName(np0), lockName(np));
    }

    // probe protocol round trip (names with spaces and parentheses), other lines ignored
    {
        const std::string text =
            "some runtime chatter\n" + formatRecord(two[0]) + "\n" + formatRecord(two[1]) + "\r\nwhirl-gpu\tbad\n";
        const std::vector<GpuRecord> back = parseRecords(text);
        CHECK_EQ(back.size(), std::size_t(2));
        if (back.size() == 2) {
            for (std::size_t i = 0; i < 2; ++i) {
                CHECK_EQ(back[i].index, two[i].index);
                CHECK_EQ(back[i].name, two[i].name);
                CHECK_EQ(back[i].arch, two[i].arch);
                CHECK_EQ(back[i].pci_bus, two[i].pci_bus);
                CHECK_EQ(back[i].luid, two[i].luid);
                CHECK_EQ(back[i].integrated, two[i].integrated);
                CHECK_EQ(back[i].supported, two[i].supported);
                CHECK_EQ(back[i].total_mib, two[i].total_mib);
                CHECK_EQ(back[i].compute_units, two[i].compute_units);
                CHECK_EQ(back[i].uuid, two[i].uuid);
                CHECK(sameGpu(back[i], two[i]));
            }
        }
        CHECK(parseRecords("").empty());
    }

    // the log line names the GPU used (name, pci) and how to choose another; owner 2026-10-23: usage only, the cause
    // is not public (no "hidden from HIP", no "pinned host memory", no HIP_VISIBLE_DEVICES)
    {
        const std::string n = restrictionNote(two, {1}, "");
        CHECK(has(n, kR97) && has(n, "GPU pci:99:0 (") && !has(n, "pci:99:0."));
        CHECK(has(n, "use --device pci:<bus>:<device> to choose another (see --list-devices)"));
        const std::string n4 = restrictionNote(four, {2}, "");
        for (const std::string& t : {n, n4}) {
            CHECK(!has(t, "hidden") && !has(t, "HIP") && !has(t, "pinned") && !has(t, "other GPU"));
        }
        // must not look like a HIP error to the log scanners (golden.ps1 / kvram719_tier.ps1 patterns)
        CHECK(!std::regex_search(
            n + n4, std::regex("hipError|hip.*\\b719\\b|\\b719\\b.*hip|\\| failed:|PAL failed|Failed PAL|result:-5|unspecified launch",
                               std::regex::icase)));
        const std::string one = restrictionNote(igpu_only, {0}, "");
        CHECK(!has(one, "hidden"));  // nothing to hide
        CHECK(has(one, k8060) && has(one, "the only GPU"));
        const std::string u = userVarNote("HIP_VISIBLE_DEVICES", "1");
        CHECK(has(u, "HIP_VISIBLE_DEVICES=1") && has(u, "visible"));
    }
}

// s8-719 review C1 (written from the finding, 2026-10-23): the GPU probe is a hidden helper process, so
// (a) a crash in it must never show a Windows error dialog, (b) the parent must not hand it every inheritable
// handle (log file, listening socket), only the pipe and NUL, and (c) it must give up after 10 s, not 60 s.
// The child is this test program itself (helper modes in main()).
#if WHIRL_WITH_HIP && defined(_WIN32)
void testProbeChild(const wchar_t* self) {
    using namespace whirl::gpusel;
    CHECK_EQ(k_probe_timeout_ms, 10000u);
    const std::wstring exe = self;
    const std::wstring q = L"\"" + exe + L"\" ";
    // (a) error mode of the child: SEM_NOGPFAULTERRORBOX even when the parent has none
    const UINT old = SetErrorMode(0);
    ChildRun em = runChild(exe, q + L"--child-error-mode", 10000);
    SetErrorMode(old);
    CHECK(em.started && em.exit_code == 0);
    const unsigned long mode = em.out.empty() ? 0 : std::stoul(em.out);
    CHECK((mode & SEM_NOGPFAULTERRORBOX) != 0);
    // (b) an inheritable handle of the parent must not reach the child
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    static unsigned ev_seq = 0;  // unique per call: GetTickCount64 has ~16 ms steps and --probe-child-loop calls this repeatedly
    const std::wstring ev_name = L"whirl-probechild-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64()) +
                                 L"-" + std::to_wstring(++ev_seq);
    HANDLE ev = CreateEventW(&sa, TRUE, TRUE, ev_name.c_str());  // inheritable, named so the child can check identity
    CHECK(ev != nullptr && GetLastError() != ERROR_ALREADY_EXISTS);
    const std::wstring ev_arg = ev_name + L" " + std::to_wstring(reinterpret_cast<std::uintptr_t>(ev));
    ChildRun hi = runChild(exe, q + L"--child-has-handle " + ev_arg, 10000);
    CHECK(hi.started && hi.exit_code == 0);
    CHECK(hi.out.rfind("0", 0) == 0);  // "0" = not inherited
    if (hi.out.rfind("0", 0) != 0) std::printf("  (b) child said '%s' for handle %llu\n", hi.out.c_str(), (unsigned long long)reinterpret_cast<std::uintptr_t>(ev));
    {  // control: the oracle is not vacuous - without a handle list the event IS inherited and the child must say so
        STARTUPINFOW csi{sizeof(csi)};
        PROCESS_INFORMATION cpi{};
        std::wstring ccmd = q + L"--child-has-handle-rc " + ev_arg;
        const BOOL cok = CreateProcessW(exe.c_str(), ccmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &csi, &cpi);
        CHECK(cok);
        DWORD crc = 99;
        if (cok) {
            CHECK(WaitForSingleObject(cpi.hProcess, 10000) == WAIT_OBJECT_0);
            GetExitCodeProcess(cpi.hProcess, &crc);
            CloseHandle(cpi.hThread);
            CloseHandle(cpi.hProcess);
        }
        CHECK_EQ(crc, 1ul);  // 1 = the parent's event reached this child
    }
    CloseHandle(ev);
    // (b2) bug-probechild-flaky 2026-10-26: none of 256 inheritable parent events may reach the child. Handle values are
    // small and reused per process, so the child can hold an unrelated live object at the parent's value: the check must
    // be exact, not "this value is a live signaled handle here" (that flaked 1 in ~4 runs in N2a and Q2).
    {
        const std::wstring name2 = ev_name + L"-b2";
        HANDLE base = CreateEventW(nullptr, TRUE, TRUE, name2.c_str());
        CHECK(base != nullptr);
        std::vector<HANDLE> evs;
        std::wstring args = L" " + name2;
        for (int i = 0; i < 256 && base; ++i) {  // inheritable duplicates: 256 parent handle values, one object
            HANDLE e = nullptr;
            if (!DuplicateHandle(GetCurrentProcess(), base, GetCurrentProcess(), &e, 0, TRUE, DUPLICATE_SAME_ACCESS)) break;
            evs.push_back(e);
            args += L" " + std::to_wstring(reinterpret_cast<std::uintptr_t>(e));
        }
        CHECK_EQ(evs.size(), size_t{256});
        ChildRun many = runChild(exe, q + L"--child-has-handle" + args, 10000);
        CHECK(many.started && many.exit_code == 0);
        CHECK(many.out == "0");
        if (many.out != "0") std::printf("  (b2) child said %s of %zu parent events are inherited\n", many.out.c_str(), evs.size());
        for (HANDLE e : evs) CloseHandle(e);
        if (base) CloseHandle(base);
    }
    // (c) a child that hangs is killed after the bound
    const ULONGLONG t0 = GetTickCount64();
    ChildRun hang = runChild(exe, q + L"--child-sleep-ms 30000", 1000);
    const ULONGLONG dt = GetTickCount64() - t0;
    CHECK(hang.timed_out);
    CHECK(hang.out.empty());
    CHECK(dt < 8000);
    // a normal child: output captured
    ChildRun ok = runChild(exe, q + L"--child-echo", 10000);
    CHECK(ok.started && ok.exit_code == 0 && ok.why.empty());
    CHECK(ok.out.rfind("echo-ok", 0) == 0);
}
#endif

}  // namespace

namespace {
// the old single main()'s helper modes of these tests (the test runs this program as the child)
static int childModes(int argc, char** argv) {
#if WHIRL_WITH_HIP && defined(_WIN32)
    if (argc >= 2) {  // helper modes for testProbeChild (the test runs this program as the child)
        const std::string m = argv[1];
        if (m == "--child-error-mode") { std::printf("%u", GetErrorMode()); return 0; }
        // --child-has-handle[-rc] <event name> <value>...: how many of the values are handles to THAT event (the parent's
        // object, opened here by name). Exact: a value that is merely some live handle of this process does not count
        // (bug-probechild-flaky: handle values are reused per process). -rc reports the count as the exit code.
        if ((m == "--child-has-handle" || m == "--child-has-handle-rc") && argc >= 4) {
            using CompareFn = BOOL(WINAPI*)(HANDLE, HANDLE);
            HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
            auto same = kb ? reinterpret_cast<CompareFn>(GetProcAddress(kb, "CompareObjectHandles")) : nullptr;
            HANDLE mine = OpenEventW(SYNCHRONIZE, FALSE, std::wstring(argv[2], argv[2] + std::strlen(argv[2])).c_str());
            if (!same || !mine) { std::printf("no-oracle"); return 4; }
            int live = 0;
            for (int i = 3; i < argc; ++i) {
                HANDLE h = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(std::stoull(argv[i])));
                live += h != mine && same(h, mine);  // `mine` took a free slot, so its own value was not inherited
            }
            if (m == "--child-has-handle-rc") return live;
            std::printf("%d", live);
            return 0;
        }
        if (m == "--child-sleep-ms" && argc == 3) { Sleep(static_cast<DWORD>(std::stoul(argv[2]))); return 0; }
        if (m == "--child-echo") { std::printf("echo-ok"); return 0; }
        if (m == "--probe-child-loop" && argc == 3) {  // stress: only testProbeChild, N times (bug-probechild-flaky)
            wchar_t self[1024];
            const DWORD n = GetModuleFileNameW(nullptr, self, 1024);
            if (n == 0 || n >= 1024) return 2;
            const int reps = std::stoi(argv[2]);
            int bad_iters = 0;
            for (int i = 0; i < reps; ++i) {
                const int before = g_fail;
                testProbeChild(self);
                if (g_fail != before) std::printf("iteration %d failed\n", i), ++bad_iters;
            }
            std::printf("probe-child loop: %d iterations, %d failed iterations, %d passed, %d failed checks\n", reps, bad_iters, g_pass, g_fail);
            return g_fail ? 1 : 0;
        }
    }
#endif
    (void)argc;
    (void)argv;
    return -1;
}
}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_gpu_select() {
    static const whirl::test::Suite s{"gpu_select",
                                      {
                                          {"testGpuSelect", [] { testGpuSelect(); }},
#if WHIRL_WITH_HIP && defined(_WIN32)
                                          {"testProbeChild", [] { wchar_t self[1024]; const DWORD n = GetModuleFileNameW(nullptr, self, 1024); if (n > 0 && n < 1024) testProbeChild(self); else CHECK(false); }},
#endif
                                      },
                                      childModes};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_gpu_select()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
