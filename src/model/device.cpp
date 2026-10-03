// Device selection and the one-process-per-GPU lock.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's gguf_cli.zig (pickDevice, lockGpu, matchDevice).

#include "whirl/model.h"

#include <cctype>
#include <cstdio>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::qwen35 {

namespace {

bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
    return true;
}

std::string lower(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

int matchDevice(int n, std::string_view spec) {
    bool all_digits = !spec.empty();
    for (char c : spec) all_digits = all_digits && c >= '0' && c <= '9';
    if (all_digits) {
        const int idx = std::stoi(std::string(spec));
        if (idx >= n) throw ModelError("NoTargetGpu", "device index " + std::string(spec));
        return idx;
    }
    // aliases: marketing short names -> LLVM target
    std::string key(spec);
    if (ieq(spec, "8060s") || ieq(spec, "igpu") || ieq(spec, "strix"))
        key = "gfx1151";
    else if (ieq(spec, "r9700"))
        key = "gfx1201";
    const std::string lk = lower(key);
    for (int i = 0; i < n; ++i) {
        const hip::DeviceInfo info = hip::describeDevice(i);
        if (lower(info.gcn_arch).rfind(lk, 0) == 0 || lower(info.name).find(lk) != std::string::npos) return i;
    }
    throw ModelError("NoTargetGpu", "no HIP device matches --device / WHIRL_DEVICE '" + std::string(spec) + "'");
}

// One model process per GPU: a named mutex per device index, held until the
// process exits. A second WHIRL process on the same GPU waits (reporting every
// 30 s, up to WHIRL_GPU_WAIT seconds, default 1800). WHIRL_GPU_SHARE=1 skips it.
void lockGpu(int dev) {
    if (auto v = envGet("GPU_SHARE"); v && !v->empty() && (*v)[0] != '0') return;
    const std::wstring name = L"Local\\whirl-gpu-" + std::to_wstring(dev);
    HANDLE h = CreateMutexW(nullptr, FALSE, name.c_str());
    if (!h) return;
    unsigned limit_s = 1800;
    if (auto v = envGet("GPU_WAIT")) {
        try {
            limit_s = static_cast<unsigned>(std::stoul(*v));
        } catch (...) {
        }
    }
    unsigned waited = 0;
    for (;;) {
        const DWORD r = WaitForSingleObject(h, 30000);
        if (r == WAIT_OBJECT_0 || r == WAIT_ABANDONED) return;  // held until exit (never closed)
        waited += 30;
        std::fprintf(stderr, "whirl: another WHIRL process is using GPU %d; waiting (%u s)\n", dev, waited);
        if (waited >= limit_s) throw ModelError("GpuBusy");
    }
}

}  // namespace

int pickDevice(std::string_view spec_arg) {
    const int n = hip::deviceCount();
    std::string spec(spec_arg);
    if (spec.empty()) {
        if (auto v = envGet("DEVICE")) spec = *v;
    }
    int dev = -1;
    if (!spec.empty()) {
        dev = matchDevice(n, spec);
    } else {
        // default: the first R9700, else the first device this build has kernels for
        // (e.g. a Radeon 8060S on its own)
        auto have = [](const std::string& a) {
            for (const hip::EmbeddedObject& o : hip::embeddedObjects())
                if (a.rfind(o.arch, 0) == 0) return true;
            return false;
        };
        for (int i = 0; i < n && dev < 0; ++i) {
            const std::string a = hip::describeDevice(i).gcn_arch;
            if (a.rfind("gfx1201", 0) == 0 && have(a)) dev = i;
        }
        for (int i = 0; i < n && dev < 0; ++i)
            if (have(hip::describeDevice(i).gcn_arch)) dev = i;
        if (dev < 0) throw ModelError("NoTargetGpu", "no R9700 or Radeon 8060S found (set --device / WHIRL_DEVICE)");
    }
    lockGpu(dev);
    return dev;
}

}  // namespace whirl::qwen35
