// whirl-mem-hog (arch024-S14, PLAN c-3 E1 / E4): another program that holds GPU memory while a gate starts a server.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL (scratch\wddmlog\vram_hog.cpp made permanent, plus pinned host memory).
//   whirl-mem-hog --device N [--vram-mb M] [--pinned-mb P] [--seconds S] [--ready FILE]
// N: the HIP ordinal this process sees (set HIP_VISIBLE_DEVICES to limit what it sees; WHIRL's own GPU
// restriction is not used here: a pinned buffer of this process is mapped to every GPU it sees, like any other
// program's). Allocates in 512 MiB pieces (one huge allocation can be refused), touches every piece, writes
// "ready vram_mb=M pinned_mb=P" to stdout and to --ready FILE, holds for S seconds (default 600), frees and exits 0.
// A gate stops it earlier by killing it by PID (Windows frees the memory with the process). Exit 1: an allocation
// failed (the line names it and what was held).

#include "whirl/hip.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    using namespace whirl;
    int dev = -1;
    long long vram_mb = 0, pinned_mb = 0, secs = 600;
    std::string ready;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--device" && more) dev = std::atoi(argv[++i]);
        else if (a == "--vram-mb" && more) vram_mb = std::atoll(argv[++i]);
        else if (a == "--pinned-mb" && more) pinned_mb = std::atoll(argv[++i]);
        else if (a == "--seconds" && more) secs = std::atoll(argv[++i]);
        else if (a == "--ready" && more) ready = argv[++i];
        else {
            std::fprintf(stderr, "usage: whirl-mem-hog --device N [--vram-mb M] [--pinned-mb P] [--seconds S] [--ready FILE]\n");
            return 2;
        }
    }
    if (dev < 0 || vram_mb < 0 || pinned_mb < 0 || (vram_mb == 0 && pinned_mb == 0)) {
        std::fprintf(stderr, "whirl-mem-hog: --device and a size (--vram-mb / --pinned-mb) are required\n");
        return 2;
    }
    std::vector<hip::DevPtr> dbufs;
    std::vector<void*> hbufs;
    long long got_v = 0, got_p = 0;
    try {
        const hip::DeviceInfo info = hip::describeDevice(dev);
        hip::setDevice(dev);
        std::printf("whirl-mem-hog: device %d: %s (%s), %d HIP GPU(s) visible\n", dev, info.name.c_str(), info.gcn_arch.c_str(),
                    hip::deviceCount());
        while (got_v < vram_mb) {
            const long long n = vram_mb - got_v < 512 ? vram_mb - got_v : 512;
            const hip::DevPtr p = hip::malloc(static_cast<std::size_t>(n) << 20);
            hip::memset(p, 1, static_cast<std::size_t>(n) << 20);
            dbufs.push_back(p);
            got_v += n;
        }
        hip::sync();
        while (got_p < pinned_mb) {
            const long long n = pinned_mb - got_p < 512 ? pinned_mb - got_p : 512;
            void* p = hip::hostMalloc(static_cast<std::size_t>(n) << 20);
            std::memset(p, 1, static_cast<std::size_t>(n) << 20);
            hbufs.push_back(p);
            got_p += n;
        }
    } catch (const std::exception& e) {
        std::printf("whirl-mem-hog: allocation failed at vram %lld / %lld MiB, pinned %lld / %lld MiB: %s\n", got_v, vram_mb, got_p,
                    pinned_mb, e.what());
        return 1;
    }
    std::printf("ready vram_mb=%lld pinned_mb=%lld\n", got_v, got_p);
    std::fflush(stdout);
    if (!ready.empty()) {
        if (std::FILE* f = std::fopen(ready.c_str(), "w")) {
            std::fprintf(f, "ready vram_mb=%lld pinned_mb=%lld\n", got_v, got_p);
            std::fclose(f);
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(secs));
    for (void* p : hbufs) hip::hostFree(p);
    for (hip::DevPtr p : dbufs) hip::free(p);
    std::printf("whirl-mem-hog: freed\n");
    return 0;
}
