// `vis-encode`: standalone vision encoder test (own device buffers, no language
// model), shared by whirl-vis.exe and `whirl vis-encode`.
// SPDX-License-Identifier: Apache-2.0
//
//   vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]
//
// Environment: WHIRL_DEVICE (default: the first R9700), WHIRL_VIS_DUMP_RGB=FILE
// (preprocessed RGB bytes), WHIRL_VIS_PRE=IMAGE (encode another image first in
// the same arena), WHIRL_VIS_PROF=1 (synchronized per-op profile of the last rep).
// Output format follows the prototype's `whirl vis-encode` (item VIS).

#include "whirl/common.h"
#include "whirl/hip.h"
#include "whirl/vision.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace whirl;

namespace {

const char* envRaw(const char* name) { return std::getenv(name); }

std::vector<std::uint8_t> readBytes(const std::string& path) {
    const std::string s = readFile(path);
    return std::vector<std::uint8_t>(s.begin(), s.end());
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
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--reps" && i + 1 < args.size()) {
            reps = static_cast<std::uint32_t>(std::strtoul(args[++i].c_str(), nullptr, 10));
        } else if (args[i] == "--mode" && i + 1 < args.size()) {
            mode = vision::parseMode(args[++i]).value_or(vision::Mode::automatic);
        } else {
            out_path = args[i];
        }
    }
    const char* dev_spec = envRaw("WHIRL_DEVICE");
    const int dev = hip::findDevice(dev_spec != nullptr ? dev_spec : "R9700");
    if (dev < 0) {
        std::printf("no such device: %s\n", dev_spec != nullptr ? dev_spec : "R9700");
        return 1;
    }
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
    if (const char* rp = envRaw("WHIRL_VIS_DUMP_RGB"))
        writeFile(rp, std::string_view(reinterpret_cast<const char*>(p.rgb.px.data()), p.rgb.px.size()));
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
    hip::Stream stream = hip::streamCreate();
    std::optional<vision::Prepared> pre;
    if (const char* pp = envRaw("WHIRL_VIS_PRE")) pre = v->prepare(readBytes(pp));
    const std::uint64_t need = std::max(v->arenaBytes(p.nTokens()), pre ? v->arenaBytes(pre->nTokens()) : 0);
    const hip::DevPtr arena = hip::malloc(need);
    const vision::Region lend[1] = {{arena, need}};
    if (pre) {
        std::vector<float> pout(static_cast<std::size_t>(pre->nTokens()) * v->hp.proj_dim);
        (void)v->encode(*pre, lend, stream, pout.data(), pout.size());
        std::printf("pre-encoded %u tokens\n", pre->nTokens());
    }
    std::vector<float> out(static_cast<std::size_t>(p.nTokens()) * v->hp.proj_dim);
    double prof[8] = {};
    const bool want_prof = envRaw("WHIRL_VIS_PROF") != nullptr;
    for (std::uint32_t r = 0; r < reps; ++r) {
        if (want_prof && r + 1 == reps) v->prof = prof;
        const vision::EncodeStats st = v->encode(p, lend, stream, out.data(), out.size());
        std::printf("encode %u: %.1f ms (%s%s), arena %.1f MiB\n", r, st.ms_total, st.resident ? "resident" : "streamed weights",
                    st.ms_upload > 0 ? " incl. upload" : "", static_cast<double>(need) / 1048576.0);
        std::fflush(stdout);
    }
    v->prof = nullptr;
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
    hip::free(arena);
    v.reset();
    hip::streamDestroy(stream);
    return 0;
}

}  // namespace whirl::vision
