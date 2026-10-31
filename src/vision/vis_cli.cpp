// `vis-encode`: standalone vision encoder test (own device buffers, no language
// model), shared by whirl-vis.exe and `whirl vis-encode`.
// SPDX-License-Identifier: Apache-2.0
//
//   vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream] [--device D]
//
// GPU: --device / -dev / WHIRL_DEVICE, default = the most GPU memory (same rules as whirl and
// whirl-server, gpusel). Environment: WHIRL_VIS_DUMP_RGB=FILE
// (preprocessed RGB bytes), WHIRL_VIS_PRE=IMAGE (encode another image first in
// the same arena), WHIRL_VIS_PROF=1 (synchronized per-op profile of the last rep).
// Output format follows the prototype's `whirl vis-encode` (item VIS).

#include "whirl/common.h"
#include "whirl/gpu_select.h"
#include "whirl/hip.h"
#include "whirl/vision.h"

#include "platform/env.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace whirl;

namespace {

// arch025 PL1b + bug-envpaths: through platform::env::getUtf8 (each read at its old place, once per command;
// UTF-8 of the wide value, was the ANSI code page); both path values go through readFile / writeFile (widen)
std::optional<std::string> envRaw(const char* name) { return platform::env::getUtf8(name); }

std::vector<std::uint8_t> readBytes(const std::string& path) {
    const std::string s = readFile(path);
    return std::vector<std::uint8_t>(s.begin(), s.end());
}

// test only (bug-visstream-race): WHIRL_TEST_VIS_COPY_DELAY_MB=N queues N MiB of dummy host->device copies on the
// encoder's copy stream right before its layer-0 weight copy (Vision::copy_test_hook), so that copy lands well after
// the compute stream's first kernels; off when unset. Own pinned / device buffers, never the encoder's.
struct CopyDelay {
    static constexpr std::uint64_t chunk = 64ull << 20;
    std::uint64_t n_chunks = 0;
    device::PinnedBuf src;
    device::DevBuf dst;
};

void copyDelayHook(void* ctx, hip::Stream cs) {
    const CopyDelay& d = *static_cast<const CopyDelay*>(ctx);
    for (std::uint64_t i = 0; i < d.n_chunks; ++i) hip::copyAnyAsync(d.dst.get(), d.src.addr(), CopyDelay::chunk, cs);
}

}  // namespace

namespace whirl::vision {

int cliVisEncode(std::span<const std::string> args) {
    if (args.size() < 2) {
        std::printf("usage: vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]\n");
        return 2;
    }
    const std::string mm = args[0], img = args[1];
    std::uint32_t reps = 2;
    vision::Mode mode = vision::Mode::automatic;
    std::string out_path;
    std::string device_spec;
    bool device_given = false;
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--reps" && i + 1 < args.size()) {
            reps = static_cast<std::uint32_t>(std::strtoul(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--mode" && i + 1 < args.size()) {
            mode = vision::parseMode(args[++i]).value_or(vision::Mode::automatic);
        } else if ((args[i] == "--device" || args[i] == "-dev") && i + 1 < args.size()) {
            if (device_given) throw std::runtime_error("give --device only once");
            device_given = true;
            device_spec = args[++i];
        } else {
            out_path = args[i];
        }
    }
    // the same choice and restriction as whirl / whirl-server, before the first HIP call
    const gpusel::Restriction& r = gpusel::restrictToSelected(device_spec);  // throws SelectError (a bad --device)
    if (!r.note.empty()) std::fprintf(stderr, "whirl: %s\n", r.note.c_str());
    const std::vector<gpusel::GpuRecord> recs = gpusel::enumerate();
    if (recs.empty()) {
        if (!gpusel::userVisibilityVarHere().empty()) std::printf("no GPU: %s\n", gpusel::noGpuTextHere().c_str());  // arch025-c2fu
        else std::printf("no GPU: the AMD graphics driver reports none\n");
        return 1;
    }
    const int dev = recs[static_cast<std::size_t>(gpusel::chosenPosition(recs, device_spec))].index;
    hip::setDevice(dev);
    const double tl = nowSeconds();
    auto v = vision::Vision::load(mm);
    v->mode = mode;
    std::printf("mmproj %s: %u layers, ff %u (pad %u), proj %u, pos grid %u^2; pinned %.1f MiB, loaded in %.0f ms, id %llx\n", mm.c_str(),
                v->hp.n_layer, v->hp.n_ff, v->hp.n_ff_pad, v->hp.proj_dim, v->hp.n_side, static_cast<double>(v->totalBytes()) / 1048576.0,
                (nowSeconds() - tl) * 1000.0, static_cast<unsigned long long>(v->id));
    const std::vector<std::uint8_t> bytes = readBytes(img);
    const double tp = nowSeconds();
    vision::Prepared p;
    try {
        p = v->prepare(bytes);
    } catch (const vision::VisionError& e) {
        std::printf("image decode failed: %s (%s)\n", e.code().c_str(), vision::decodeFailureReason());
        return 1;
    }
    std::printf("image %s: -> %ux%u (%ux%u = %u tokens, %u patches), preprocess %.1f ms\n", img.c_str(), p.rgb.w, p.rgb.h, p.nx, p.ny, p.nTokens(),
                p.nTokens() * 4, (nowSeconds() - tp) * 1000.0);
    if (const std::optional<std::string> rp = envRaw("WHIRL_VIS_DUMP_RGB"))
        writeFile(*rp, std::string_view(reinterpret_cast<const char*>(p.rgb.px.data()), p.rgb.px.size()));
    {
        std::string hx;
        char b[3];
        for (std::uint8_t c : p.hash) {
            std::snprintf(b, sizeof(b), "%02x", c);
            hx += b;
        }
        std::printf("content hash %s; token ids %08x %08x ...\n", hx.c_str(), vision::tokenId(p.hash, 0), vision::tokenId(p.hash, 1));
    }
    std::fflush(stdout);
    const device::StreamH stream_own = device::StreamH::adopt(device::hipBackend(), hip::streamCreate());  // blocking
    hip::Stream stream = stream_own.get();
    std::optional<vision::Prepared> pre;
    if (const std::optional<std::string> pp = envRaw("WHIRL_VIS_PRE")) pre = v->prepare(readBytes(*pp));
    const std::uint64_t need = std::max(v->arenaBytes(p.nTokens()), pre ? v->arenaBytes(pre->nTokens()) : 0);
    device::DevBuf arena_own = device::DevBuf::alloc(device::hipBackend(), need);
    const hip::DevPtr arena = arena_own.get();
    const vision::Region lend[1] = {{arena, need}};
    if (pre) {
        std::vector<float> pout(static_cast<std::size_t>(pre->nTokens()) * v->hp.proj_dim);
        (void)v->encode(*pre, lend, stream, pout.data(), pout.size());
        std::printf("pre-encoded %u tokens\n", pre->nTokens());
    }
    CopyDelay delay;
    if (const std::optional<std::string> dm = envRaw("WHIRL_TEST_VIS_COPY_DELAY_MB")) {
        const std::uint64_t mb = std::strtoull(dm->c_str(), nullptr, 10);
        delay.n_chunks = (mb * (1ull << 20) + CopyDelay::chunk - 1) / CopyDelay::chunk;
        if (delay.n_chunks > 0) {
            delay.src = device::PinnedBuf::alloc(device::hipBackend(), CopyDelay::chunk);
            delay.dst = device::DevBuf::alloc(device::hipBackend(), CopyDelay::chunk);
            v->copy_test_ctx = &delay;
            v->copy_test_hook = copyDelayHook;
            std::printf("test hook: %llu MiB of dummy copies before the layer-0 weight copy (streamed weights only)\n",
                        static_cast<unsigned long long>(delay.n_chunks * (CopyDelay::chunk >> 20)));
        }
    }
    std::vector<float> out(static_cast<std::size_t>(p.nTokens()) * v->hp.proj_dim);
    double prof[8] = {};
    const bool want_prof = envRaw("WHIRL_VIS_PROF").has_value();
    for (std::uint32_t r = 0; r < reps; ++r) {
        if (want_prof && r + 1 == reps) v->prof = prof;
        const vision::EncodeStats st = v->encode(p, lend, stream, out.data(), out.size());
        std::printf("encode %u: %.1f ms (%s%s), arena %.1f MiB\n", r, st.ms_total, st.resident ? "resident" : "streamed weights",
                    st.ms_upload > 0 ? " incl. upload" : "", static_cast<double>(need) / 1048576.0);
        std::fflush(stdout);
    }
    v->prof = nullptr;
    v->copy_test_hook = nullptr;
    v->copy_test_ctx = nullptr;
    if (want_prof)
        std::printf("profile (synchronized, last rep): patch+pos %.1f | gemm %.1f | qkv_prep %.1f | attn %.1f | ln/resid %.1f | gelu %.1f | merger %.1f | wait/copy %.1f ms\n",
                    prof[0], prof[1], prof[2], prof[3], prof[4], prof[5], prof[6], prof[7]);
    double s = 0, s2 = 0;
    for (float x : out) {
        s += x;
        s2 += static_cast<double>(x) * x;
    }
    const double n = static_cast<double>(out.size());
    std::printf("embeddings [%u, %u]: mean %.6f std %.6f; token 0: %.6f %.6f %.6f %.6f\n", p.nTokens(), v->hp.proj_dim, s / n,
                std::sqrt(s2 / n - (s / n) * (s / n)), out[0], out[1], out[2], out[3]);
    if (!out_path.empty()) {
        writeFile(out_path, std::string_view(reinterpret_cast<const char*>(out.data()), out.size() * 4));
        std::printf("wrote %s\n", out_path.c_str());
    }
    arena_own.reset();  // arena, encoder, then the stream (the order of before arch024 S11)
    v.reset();
    return 0;
}

// The whirl-vis executable's main() (arch025-n8b2: moved verbatim from vis_main.cpp, which keeps a one-line main that calls this,
// the llama.cpp tools/server/main.cpp pattern; so a test can run it: tests/vision/vision_tests.cpp testVisMain).
int visMain(int argc, char** argv) {
    // restrictToSelected lists the GPUs by starting this exe as `<exe> --whirl-gpu-probe` (s8-719 root719b vis_fixed)
    if (argc == 2 && std::strcmp(argv[1], whirl::gpusel::k_probe_arg) == 0) return whirl::gpusel::probeMain();
    if (argc < 2 || std::strcmp(argv[1], "vis-encode") != 0) {
        std::printf("usage: whirl-vis vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]\n");
        return 2;
    }
    std::vector<std::string> args(argv + 2, argv + argc);
    try {
        return whirl::vision::cliVisEncode(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "whirl-vis: %s\n", e.what());
        return 1;
    }
}

}  // namespace whirl::vision
