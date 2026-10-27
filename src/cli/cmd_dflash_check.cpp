// whirl dflash-check: the DFlash2 drafter forward vs reference values (arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

const char* k_help_dflash_check =
    "usage: whirl dflash-check MODEL.gguf DRAFT.gguf REFDIR [--device SPEC]\n"
    "\n"
    "Runs the DFlash2 drafter (DRAFT.gguf) of MODEL.gguf on reference inputs and compares each layer's\n"
    "block hidden state, the final norm and the drafts with reference values (whirl-cloud\n"
    "tools/dflash_ref: ref_drafter.py, then export_check.py writes REFDIR = meta.txt, feats.f32 in\n"
    "layer-major order, hidden_l<i>.f32, final_norm.f32, drafts.i32). Prints relative errors; exit 1\n"
    "if a layer differs by more than 3e-2 (Q8_0 weights, int8 GEMV activations).\n"
    "\n"
    "options:\n"
    "  --device SPEC, -dev SPEC   GPU: pci:BUS:DEVICE (Task Manager > Performance > GPU > Physical location: PCI bus 71,\n"
    "                             device 0 -> pci:71:0), an index from 'whirl devices', r9700, 8060s, or a part of the\n"
    "                             name. Default: the GPU with the most GPU memory\n"
    "  -h, --help                 this text\n";

namespace {

// whirl dflash-check MODEL DRAFT REFDIR: drafter forward vs reference values
std::string readBinFile(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("dflash-check: cannot open " + path);
    std::string s;
    char buf[1 << 16];
    for (std::size_t n; (n = std::fread(buf, 1, sizeof buf, fp)) > 0;) s.append(buf, n);
    std::fclose(fp);
    return s;
}

}  // namespace

int cmdDflashCheck(const Args& a) {
    if (a.pos.size() < 3) throw std::runtime_error("dflash-check: usage: whirl dflash-check MODEL.gguf DRAFT.gguf REFDIR");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Loaded L = loadModel(a, f, 4096);
    q::Model& m = *L.model;
    const gguf::File df = gguf::File::open(a.pos[1]);
    q::LoadStats ds;
    m.loadDrafter(df, ds);
    const std::string dir = a.pos[2];
    std::map<std::string, long long> meta;
    {
        const std::string t = readBinFile(dir + "/meta.txt");
        std::size_t p = 0;
        while (p < t.size()) {
            std::size_t e = t.find('\n', p);
            if (e == std::string::npos) e = t.size();
            const std::string line = t.substr(p, e - p);
            const std::size_t sp = line.find(' ');
            if (sp != std::string::npos) meta[line.substr(0, sp)] = std::stoll(line.substr(sp + 1));
            p = e + 1;
        }
    }
    for (const char* key : {"n_ctx", "ctx_pos0", "anchor", "anchor_pos"})
        if (!meta.count(key)) throw std::runtime_error(std::string("dflash-check: meta.txt has no ") + key);
    const auto& dc = m.dflash->cfg;
    const u64 E = dc.n_embd;
    const u32 n_ctx = static_cast<u32>(meta["n_ctx"]), B = dc.block;
    const std::string feats = readBinFile(dir + "/feats.f32");
    if (feats.size() != dc.target_layers.size() * n_ctx * E * 4)
        throw std::runtime_error("dflash-check: feats.f32 is not [target layers][n_ctx][n_embd] f32");
    m.dflashSetup(1, std::max<u32>(1, n_ctx), true);
    const hip::DevPtr fd = m.alloc(feats.size());
    hip::upload(fd, feats.data(), feats.size());
    m.dflashIngest(0, fd, n_ctx, 0, n_ctx, static_cast<u32>(meta["ctx_pos0"]));
    const std::vector<u32> drafts = m.dflashDraft(0, static_cast<u32>(meta["anchor"]), static_cast<u32>(meta["anchor_pos"]));
    bool ok = true;
    std::vector<float> got(static_cast<std::size_t>(B) * E);
    auto compare = [&](const std::string& name, hip::DevPtr src, const std::string& file) {
        hip::download(got.data(), src, got.size() * 4);
        const std::string rs = readBinFile(dir + "/" + file);
        if (rs.size() != got.size() * 4) throw std::runtime_error("dflash-check: " + file + " is not [block][n_embd] f32");
        const float* ref = reinterpret_cast<const float*>(rs.data());
        double num = 0, den = 0, mx = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double d = static_cast<double>(got[i]) - ref[i];
            num += d * d;
            den += static_cast<double>(ref[i]) * ref[i];
            mx = std::max(mx, std::fabs(d));
        }
        const double rel = std::sqrt(num / std::max(den, 1e-30));
        const bool pass = rel < 3e-2;
        ok = ok && pass;
        out(fmt("  %-12s rel. error %.3e, max abs %.3e  %s\n", name.c_str(), rel, mx, pass ? "ok" : "FAIL"));
    };
    for (u32 l = 0; l < dc.n_layer; ++l)
        compare("layer " + std::to_string(l), m.dfr->dbg + static_cast<u64>(l) * B * E * 4, "hidden_l" + std::to_string(l) + ".f32");
    compare("final norm", m.dfr->e, "final_norm.f32");
    const std::string dr = readBinFile(dir + "/drafts.i32");
    const auto* rd = reinterpret_cast<const std::int32_t*>(dr.data());
    const std::size_t n_ref = dr.size() / 4;
    std::size_t same = 0;
    while (same < n_ref && same < drafts.size() && static_cast<u32>(rd[same]) == drafts[same]) ++same;
    std::string gs, rs2;
    for (u32 t : drafts) gs += " " + std::to_string(t);
    for (std::size_t i = 0; i < n_ref; ++i) rs2 += " " + std::to_string(rd[i]);
    out(fmt("  drafts: whirl%s | reference%s | common prefix %zu of %zu\n", gs.c_str(), rs2.c_str(), same, n_ref));
    out(ok ? "dflash-check: ok\n" : "dflash-check: FAIL\n");
    return ok ? 0 : 1;
}

}  // namespace cli
