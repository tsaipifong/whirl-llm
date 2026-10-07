// whirl-kernel-test: runs the WHIRL HIP kernels on a GPU and compares them
// with C++ CPU references and with each other (bitwise invariances).
// SPDX-License-Identifier: Apache-2.0
//
// usage: whirl-kernel-test [--device SPEC] [--only fam,fam] [--quick] [-v]
//                          [--q4 GGUF] [--mx GGUF] [--moe GGUF] [--moemx GGUF]
// families: quant gemv gemm attn gdn moe moemx misc

#include <chrono>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

#include "cpu_ref.h"

namespace {

// default model paths come from the environment (unset: synthetic data is used)
std::string envOr(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

int usage() {
    std::printf(
        "usage: whirl-kernel-test [--device SPEC] [--only quant,gemv,gemm,attn,gdn,moe,moemx,misc] [--quick] [-v]\n"
        "                         [--q4 GGUF] [--mx GGUF] [--moe GGUF] [--moemx GGUF]\n"
        "  --device  index, gfx arch, name substring, r9700 or 8060s (default: WHIRL_DEVICE, else the\n"
        "            first R9700, else the first GPU this build has kernels for)\n"
        "  models default to WHIRL_TEST_Q4 (else WHIRL_TEST_GGUF) / WHIRL_TEST_MX / WHIRL_TEST_MOE /\n"
        "  WHIRL_TEST_MOEMX; an unset model or '-' means synthetic data is used\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string device = envOr("WHIRL_DEVICE");
    std::set<std::string> only;
    kt::Ctx c;
    c.models = {envOr("WHIRL_TEST_Q4"), envOr("WHIRL_TEST_MX"), envOr("WHIRL_TEST_MOE"), envOr("WHIRL_TEST_MOEMX")};
    if (c.models.q4.empty()) c.models.q4 = envOr("WHIRL_TEST_GGUF");
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        try {
            if (a == "--device") device = next();
            else if (a == "--only") {
                std::stringstream ss(next());
                std::string f;
                while (std::getline(ss, f, ',')) only.insert(f);
            } else if (a == "--quick") c.quick = true;
            else if (a == "-v" || a == "--verbose") c.rep.verbose = true;
            else if (a == "--q4") c.models.q4 = next();
            else if (a == "--mx") c.models.mx = next();
            else if (a == "--moe") c.models.moe = next();
            else if (a == "--moemx") c.models.moemx = next();
            else return usage();
        } catch (const std::exception& e) {
            std::printf("%s\n", e.what());
            return usage();
        }
    }
    for (std::string* p : {&c.models.q4, &c.models.mx, &c.models.moe, &c.models.moemx})
        if (*p == "-") p->clear();

    try {
        namespace hip = whirl::hip;
        auto have = [](const hip::DeviceInfo& d) {
            for (const hip::EmbeddedObject& o : hip::embeddedObjects())
                if (d.gcn_arch.rfind(o.arch, 0) == 0) return true;
            return false;
        };
        int dev = -1;
        if (!device.empty()) {
            dev = hip::findDevice(device == "8060s" ? "gfx1151" : device == "r9700" ? "gfx1201" : device);
            if (dev < 0) throw std::runtime_error("no device matches '" + device + "'");
        } else {
            for (const hip::DeviceInfo& d : hip::listDevices())
                if (dev < 0 && d.gcn_arch.rfind("gfx1201", 0) == 0 && have(d)) dev = d.index;
            for (const hip::DeviceInfo& d : hip::listDevices())
                if (dev < 0 && have(d)) dev = d.index;
            if (dev < 0) dev = 0;
        }
        hip::setDevice(dev);
        const auto info = hip::describeDevice(dev);
        std::printf("device %d: %s (%s)\n", dev, info.name.c_str(), info.gcn_arch.c_str());
        c.mod = hip::Module::loadEmbedded();
        c.k = whirl::kernels::KernelTable::load(c.mod, whirl::kernels::KvFormat::f16);
        // gfx1151 stores packed int8 scale words; the CPU references follow the code object
        kt::ref::setXdSum(c.k.caps.xd_sum);
        const auto& cp = c.k.caps;
        std::printf("caps: fp8_gemm %d, kv_q8v %d, kv_q8h %d, gemvw %d, gdn_replay %d, mrope %d, xd_sum %d, attn_group1 %d, draft_window %d\n",
                    cp.fp8_gemm, cp.kv_q8v, cp.kv_q8h, cp.gemvw, cp.gdn_replay, cp.mrope, cp.xd_sum, cp.attn_group1, cp.draft_window);
        c.s = whirl::hip::streamCreate(false);  // blocking: ordered with the null-stream copies / memsets
    } catch (const std::exception& e) {
        std::printf("setup failed: %s\n", e.what());
        return 1;
    }

    struct Fam {
        const char* name;
        void (*run)(kt::Ctx&);
    };
    const Fam fams[] = {{"quant", kt::testQuant}, {"gemv", kt::testGemv}, {"gemm", kt::testGemm},
                        {"attn", kt::testAttn},   {"gdn", kt::testGdn},   {"moe", kt::testMoe},
                        {"moemx", kt::testMoeMx}, {"misc", kt::testMisc},   {"attnbench", kt::benchAttn}};
    for (const Fam& f : fams) {
        if (!only.empty() && !only.count(f.name)) continue;
        if (only.empty() && std::string(f.name) == "attnbench") continue;  // timing only, on request
        const auto t0 = std::chrono::steady_clock::now();
        std::printf("== %s\n", f.name);
        std::fflush(stdout);
        try {
            f.run(c);
        } catch (const std::exception& e) {
            kt::Result r;
            r.family = f.name;
            r.name = std::string("exception: ") + e.what();
            r.pass = false;
            c.rep.add(r);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("   (%s: %.1f s)\n", f.name, s);
    }
    const int fails = c.rep.summary();
    whirl::hip::streamDestroy(c.s);
    return fails ? 1 : 0;
}
