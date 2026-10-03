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

#include "kt.h"

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
        "  --device  index, gfx arch or name substring (default: R9700, else device 0)\n"
        "  models default to WHIRL_TEST_Q4 (else WHIRL_TEST_GGUF) / WHIRL_TEST_MX / WHIRL_TEST_MOE /\n"
        "  WHIRL_TEST_MOEMX; an unset model or '-' means synthetic data is used\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string device = "R9700";
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
        int dev = whirl::hip::findDevice(device);
        if (dev < 0) dev = 0;
        whirl::hip::setDevice(dev);
        const auto info = whirl::hip::describeDevice(dev);
        std::printf("device %d: %s (%s)\n", dev, info.name.c_str(), info.gcn_arch.c_str());
        if (whirl::hip::archFor(info.gcn_arch) != whirl::hip::Arch::gfx1201)
            std::printf("note: only the gfx1201 kernel set is ported so far; this device's code object lacks them\n");
        c.mod = whirl::hip::Module::loadEmbedded();
        c.k = whirl::kernels::KernelTable::load(c.mod, whirl::kernels::KvFormat::f16);
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
                        {"moemx", kt::testMoeMx}, {"misc", kt::testMisc}};
    for (const Fam& f : fams) {
        if (!only.empty() && !only.count(f.name)) continue;
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
