// whirl selftest: bitwise kernel self-checks on the model's own weights (arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

const char* k_help_selftest =
    "usage: whirl selftest MODEL.gguf [--device SPEC]\n"
    "\n"
    "Bitwise self-checks on the model's own matrices: int8 vs f32 GEMV error, multi-token GEMV\n"
    "== one-token GEMV, prefill GEMM invariance over all configurations, MoE token-tile\n"
    "invariance, grouped vs per-query decode attention. Prints 'selftest: ok' or FAIL (exit 1).\n"
    "\n"
    "options:\n"
    "  --device SPEC, -dev SPEC   GPU: pci:BUS:DEVICE (Task Manager > Performance > GPU > Physical location: PCI bus 71,\n"
    "                             device 0 -> pci:71:0), an index from 'whirl devices', r9700, 8060s, or a part of the\n"
    "                             name. Default: the GPU with the most GPU memory\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

int cmdSelftest(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("selftest: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Loaded L = loadModel(a, f, 4096);
    q::Model& m = *L.model;
    hip::memset(m.scratch().h, 0, 4ull * m.max_batch * m.cfg.n_embd);
    hip::memset(m.scratch().ffn_g, 0, 4ull * m.max_batch * m.scratch().ff_scratch);
    bool ok = true;
    std::string log;
    // the precise checks force the precise items themselves (fp8 / MoE fp8 / h16 off), whatever
    // the load mode; the int8 and balance checks use the load mode's items
    out(fmt("  numerics at load: %s (precise checks run with fp8 / MoE fp8 / h16 off)\n", whirl::numerics::modeName(m.num_req.mode)));
    out("  int8 GEMV vs f32 GEMV (random x, one matrix per type):\n");
    ok = m.checkGemvq(log) && ok;
    ok = m.checkGemvBitwise(log) && ok;
    ok = m.checkPrefillInvariance(log) && ok;
    ok = m.checkPreciseDecode(log) && ok;
    ok = m.checkCapture(log) && ok;
    const u32 grp = std::max<u32>(1, 16 / (m.cfg.n_head / m.cfg.n_head_kv));
    ok = m.checkAttnGroups(log, 1000, grp) && ok;
    ok = m.checkAttnGroups(log, 1023, grp) && ok;
    const u32 grp2 = std::max<u32>(1, 32 / (m.cfg.n_head / m.cfg.n_head_kv));
    ok = m.checkAttnGroups(log, 1000, grp2, true) && ok;
    ok = m.checkAttnGroups(log, 1023, grp2, true) && ok;
    out(log);
    out(ok ? "selftest: ok\n" : "selftest: FAIL\n");
    return ok ? 0 : 1;
}

}  // namespace cli
