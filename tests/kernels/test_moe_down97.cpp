// whirl-kernel-test family "moedown97" (feat026-moedown97): the gfx1201 MXFP4 decode expert load-schedule
// candidates of a development code object (WHIRL_KT_MD97_HSACO = a gfx1201 kernels build with -DWHIRL_MD97_DEV=1;
// exports md97dn<id>_mxfp4w, md97df<id>_mxfp4, md97gu<id>_mxfp4w) bit for bit against the SHIPPED kernels of the
// main code object (moe_down_mxfp4w, moe_down_f_mxfp4, moe_gu_mxfp4w), same launches, same inputs.
//   down (int8 and f32): Ornith 2048 x 512 K 8 (n 1, 2, 4, 5, 16), no shared expert, skip flag (int8 only),
//     F 256 / 768 / 1024 / 1536 / 2560 (nb 8 / 24 / 32 / 48 / 80: every dk / db branch of the k / b stepping),
//     K 1 / 3 / 6, E 2000 (partial last block).
//   gate / up: Ornith C 2048 ff 512 K 8 (n 1, 3, 16), C 768 / 2304 / 4096 (1 partial / 3 partial / 4 steps),
//     ff 100 (partial last block), skip flag.
// Synthetic MXFP4 weights (random valid blocks), int8 activations from quantize_q8. WHIRL_KT_MD97_IDS=dn3,gu1 limits
// the candidates (default: every exported one; a listed id missing from the object FAILS). One invariant per candidate
// plus a liveness check per reference.
// feat026-moe-skipsafe (review moedown97-kcand-r1 A1 / A2):
//   * moeskip id -1 tails (feat027: a token's ids are a kept prefix, then -1 with weight 0): per-token random kept
//     count, every token K - 1, every token 1 kept; Ornith (nb 16: one KB batch covers 2 KB slots, so KB batch tails,
//     last step, NG full batches at kt >= 8 for KB 4 / kt >= 4 for KB 2), F 256 (a kept boundary inside a lane step),
//     F 1024 (NG at kt >= KB), K 3 / 6, E 2000. Fixed NG-full-batch-then-dropped-tail cases (review
//     skipsafe-gemvports C2, independent of the RNG): every token K - 1 = 7 kept at F 1024 (224 units: KB 2 3 full
//     batches + 32, KB 4 one full 128 batch + 96) and at F 1536 (336 units: KB 8 one full 256 batch + 80).
//     Dropped slots carry NaN activation scales (int8) / NaN activations (f32), so reading one is never 0 * finite.
//     Reference: the SHIPPED kernel on the compacted problem (per token K' = kt, the kept slots only): the same
//     lane -> unit map and order as feat027's loop over kt * nb, valid before feat027 (whose pre-moeskip shipped
//     kernel would read before the weight tensor on id -1) and after it. GU: the shipped kernel with the dropped ids
//     replaced by valid ones, then the dropped slots' rows reset to the poison (an id -1 block must write nothing).
//     When the main object has moeskip (exports moe_topk_skip), the shipped kernels also run the -1 problem directly
//     and must equal the compacted reference.
//   * candidate main object (exports <name>_shipped, i.e. a WHIRL_MD97_* tune id; WHIRL_KT_CODE_OBJECT / a promoted
//     default): the md97 comparisons of that kernel FAIL instead of comparing a candidate with a candidate.
//   * permanent invariant (no dev object needed): when the main object exports <name>_shipped, <name> == _shipped
//     bitwise on every case (the -1 cases against the compacted _shipped).
//   * independent reference: the shipped reference output vs a CPU double reference (exact MXFP4 weights, kept slots
//     only), tolerance 1e-4 (int8) / 1e-5 (f32) of sum |terms| (moemx / moe_f bounds); no other GPU kernel computes
//     the fused down + combine with the same bits, so this is the gemm_c0 counterpart here.
//   * skip-safety marker: a candidate object (and a tune-id main object) must export md97_skipsafe_v1 (built from
//     this branch); an object without it FAILS and its kernels do not run the -1 cases (they would read out of bounds).
//   * id -1 address check (feat026-skipsafe-r2, review skipsafe-gemvports C1): a value check cannot see a load from
//     an id -1 row whose value is masked out (e.g. a step clamped to slot K - 1 with ok = k < kt). A candidate object
//     built with -DWHIRL_MD97_DEV=1 -DWHIRL_MD97_NEGCHK=1 exports md97_negchk_v1 and the counter md97_negid_hits
//     (every weight address formed from an id < 0 counts and uses expert 0 instead, so nothing is read out of the
//     tensor); per candidate the counter must stay 0 over all runs (and the id -1 runs must exist). NEGCHK=2 (exports
//     md97_negchk_fault_v1) plants that fault (positive control): then every candidate must count hits, and its value
//     results go only into the note (DN / DF values stay equal: the blind spot this check closes). Without an
//     instrumented object the check is a skip line.
//   * DF direct id -1 run (review C3): feat027-moeskip-r1 (27bfb54) patches moe_down_f_mxfp4, b8a041a does not, and
//     moe_topk_skip exists in both, so it cannot tell them apart. The direct DF check runs when the main object
//     exports moe_f_skipsafe_v1 (proposed marker for moe_f.hip on feat027-moeskip-r1, see the merge note) or when
//     WHIRL_KT_MD97_DF_DIRECT=1 (merged-state assumption; never with a b8a041a-only object: it would read before the
//     weight tensor).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_ref.h"
#include "kt.h"

namespace kt {

namespace {

struct Cand {
    std::string name;
    hip::Function f;
    std::size_t checked = 0, bad = 0;
    std::string first_bad;
    std::size_t neg_runs = 0, neg_hits = 0;  // id -1 address check (instrumented object)
    std::string first_neg;
};

// Counter of one bitwise invariant (twin, direct-vs-compacted).
struct Inv {
    std::size_t runs = 0, bad = 0;
    std::string first;
};

// Routing: K distinct experts per token, weights normalized. drop (moeskip tails): 0 none, 1 random kept count
// kt in [1, K] per token, 2 every token kt = K - 1, 3 every token kt = 1. Dropped slots: id -1, weight 0, the kept
// weights renormalized; hvalid keeps the drawn ids (GU reference), kts the kept count per token.
void routing(Ctx& c, int n, int R, int K, std::vector<int>& hids, std::vector<float>& hw, int drop = 0, std::vector<int>* hvalid = nullptr,
             std::vector<int>* kts = nullptr) {
    hids.assign(static_cast<std::size_t>(n) * K, 0);
    hw.assign(hids.size(), 0.f);
    std::uniform_int_distribution<int> ur(0, R - 1);
    std::uniform_real_distribution<float> uw(0.05f, 1.f);
    for (int t = 0; t < n; ++t) {
        float s = 0;
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = ur(c.rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup = dup || hids[static_cast<std::size_t>(t) * K + j] == e;
            } while (dup);
            hids[static_cast<std::size_t>(t) * K + k] = e;
            hw[static_cast<std::size_t>(t) * K + k] = uw(c.rng);
            s += hw[static_cast<std::size_t>(t) * K + k];
        }
        for (int k = 0; k < K; ++k) hw[static_cast<std::size_t>(t) * K + k] /= s;
    }
    if (hvalid) *hvalid = hids;
    if (kts) kts->assign(static_cast<std::size_t>(n), K);
    if (drop == 0) return;
    std::uniform_int_distribution<int> uk(1, K);
    for (int t = 0; t < n; ++t) {
        const int kt = drop == 1 ? uk(c.rng) : drop == 2 ? (K > 1 ? K - 1 : 1) : 1;
        float s = 0;
        for (int k = 0; k < kt; ++k) s += hw[static_cast<std::size_t>(t) * K + k];
        for (int k = 0; k < K; ++k) {
            const std::size_t p = static_cast<std::size_t>(t) * K + k;
            if (k < kt) {
                hw[p] /= s;
            } else {
                hids[p] = -1;
                hw[p] = 0.f;
            }
        }
        if (kts) (*kts)[static_cast<std::size_t>(t)] = kt;
    }
}

bool sameBits(const std::vector<float>& y, const std::vector<float>& ref) {
    return y.size() == ref.size() && std::memcmp(y.data(), ref.data(), ref.size() * 4) == 0;
}

std::string firstDiff(const std::vector<float>& y, const std::vector<float>& ref, const char* what, int n, const char* a, const char* b) {
    std::size_t j = 0;
    while (j < y.size() && j < ref.size() && std::memcmp(&y[j], &ref[j], 4) == 0) ++j;
    char s[240];
    std::snprintf(s, sizeof(s), "%s n %d: [%zu] %s %.9g vs %s %.9g", what, n, j, a, j < y.size() ? y[j] : 0.f, b, j < ref.size() ? ref[j] : 0.f);
    return s;
}

void compare(Cand& cd, const std::vector<float>& y, const std::vector<float>& ref, const char* what, int n) {
    ++cd.checked;
    if (sameBits(y, ref)) return;
    ++cd.bad;
    if (cd.first_bad.empty()) cd.first_bad = firstDiff(y, ref, what, n, "got", "shipped");
}

void check(Inv& v, const std::vector<float>& y, const std::vector<float>& ref, const char* what, int n, const char* a, const char* b) {
    ++v.runs;
    if (sameBits(y, ref)) return;
    ++v.bad;
    if (v.first.empty()) v.first = firstDiff(y, ref, what, n, a, b);
}

void report(Ctx& c, const std::vector<Cand>& cs, const std::string& shipped, const char* sweep) {
    for (const Cand& cd : cs) {
        Result r;
        r.name = cd.name + " == shipped " + shipped + ", bitwise (" + std::to_string(cd.checked) + " runs: " + sweep + ")";
        r.kind = Kind::invariant;
        r.n = cd.checked;
        r.mismatches = cd.bad;
        r.pass = cd.bad == 0 && cd.checked > 0;
        if (!cd.first_bad.empty()) r.note = "first " + cd.first_bad;
        c.rep.add(r);
    }
}

// id -1 address check per candidate (instrumented object): normal = no hit; planted fault = hits for every candidate
void reportNeg(Ctx& c, const std::vector<Cand>& cs, bool fault) {
    for (const Cand& cd : cs) {
        Result r;
        r.kind = Kind::invariant;
        r.n = cd.neg_runs;
        if (!fault) {
            r.name = cd.name + ": no weight address formed from an id -1 (instrumented object; " + std::to_string(cd.neg_runs) + " id -1 runs)";
            r.mismatches = cd.neg_hits;
            r.pass = cd.neg_hits == 0 && cd.neg_runs > 0;
            if (!cd.first_neg.empty()) r.note = "first " + cd.first_neg;
        } else {
            r.name = cd.name + ": planted fault (NEGCHK=2) seen by the id -1 address check (" + std::to_string(cd.neg_runs) + " id -1 runs)";
            r.mismatches = cd.neg_hits > 0 ? 0 : 1;
            r.pass = cd.neg_hits > 0 && cd.neg_runs > 0;
            r.note = "hits " + std::to_string(cd.neg_hits) + "; value mismatches " + std::to_string(cd.bad) + " of " + std::to_string(cd.checked) +
                     " runs (informational: 0 for DN / DF = the value checks cannot see this fault)";
        }
        c.rep.add(r);
    }
}

void reportInv(Ctx& c, const std::string& name, const Inv& v) {
    Result r;
    r.name = name + " (" + std::to_string(v.runs) + " runs)";
    r.kind = Kind::invariant;
    r.n = v.runs;
    r.mismatches = v.bad;
    r.pass = v.bad == 0 && v.runs > 0;
    if (!v.first.empty()) r.note = "first " + v.first;
    c.rep.add(r);
}

void fail(Ctx& c, const std::string& name) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.pass = false;
    c.rep.add(r);
}

void live(Ctx& c, const std::string& shipped, std::size_t runs, std::size_t dead) {
    Result r;
    r.name = "moedown97: shipped " + shipped + " writes every run and honours skip (" + std::to_string(runs) + " runs; the reference is live)";
    r.kind = Kind::invariant;
    r.n = runs;
    r.mismatches = dead;
    r.pass = dead == 0 && runs > 0;
    c.rep.add(r);
}

// CPU tolerance samples of one kernel (independent reference), collected over all cases
struct CpuAcc {
    std::vector<float> got;
    std::vector<double> ref, sc;
};

void reportCpu(Ctx& c, const std::string& name, const CpuAcc& a, double rtol, double atol) {
    Result r = cmpTol(name + " (" + std::to_string(a.got.size()) + " samples)", a.got, a.ref, a.sc, rtol, atol);
    if (a.got.empty()) {
        r.pass = false;
        r.note = "no samples";
    }
    c.rep.add(r);
}

}  // namespace

void testMoeDown97(Ctx& c) {
    c.rep.family = "moedown97";
    const char* hs = std::getenv("WHIRL_KT_MD97_HSACO");
    const int ti = static_cast<int>(QType::mxfp4);
    if (c.mod.arch() != hip::Arch::gfx1201 || !c.k.moe_down_mxw || !c.k.moe_gu_mxw || !c.k.moe_down_f[ti] || !c.k.quantize_q8) {
        c.rep.skip("moedown97", "gfx1201 MXFP4 decode expert candidates", "needs the gfx1201 code object (moe_*_mxfp4w, moe_down_f_mxfp4)");
        return;
    }
    // whirl_kernels.hip renames a shipped entry to <name>_shipped exactly when its WHIRL_MD97_* tune id is non-zero
    const hip::Function twin_dn = c.fnOpt("moe_down_mxfp4w_shipped"), twin_df = c.fnOpt("moe_down_f_mxfp4_shipped"),
                        twin_gu = c.fnOpt("moe_gu_mxfp4w_shipped");
    const bool any_twin = twin_dn || twin_df || twin_gu;
    const bool skipaware = static_cast<bool>(c.fnOpt("moe_topk_skip"));  // feat027-moeskip merged: shipped kernels take id -1
    // feat027-moeskip-r1 (27bfb54) also patches moe_down_f_mxfp4; moe_topk_skip cannot tell it from b8a041a (review C3)
    const char* dfd = std::getenv("WHIRL_KT_MD97_DF_DIRECT");
    const bool df_skipaware = skipaware && (static_cast<bool>(c.fnOpt("moe_f_skipsafe_v1")) || (dfd && std::strcmp(dfd, "1") == 0));
    if ((!hs || !*hs) && !any_twin) {
        c.rep.skip("moedown97", "gfx1201 MXFP4 decode expert candidates", "WHIRL_KT_MD97_HSACO not set (main object is the shipped build)");
        return;
    }
    // a tune-id main object's md97 bodies must be the skip-safe ones before they see id -1
    const bool main_safe = !any_twin || static_cast<bool>(c.fnOpt("md97_skipsafe_v1"));
    if (!main_safe)
        fail(c, "moedown97: the main code object has a WHIRL_MD97_* tune id but no md97_skipsafe_v1 (built before feat026-moe-skipsafe: its "
                "candidate reads before the weight tensor on a moeskip id -1); rebuild it; its -1 cases are not run");
    std::vector<Cand> dn, df, gu;
    hip::Module xm;  // owns the candidate functions for the whole test
    bool cand_safe = true;
    DevPtr negp = 0;         // md97_negid_hits of an instrumented candidate object (review C1)
    bool neg_fault = false;  // the object carries the planted fault (NEGCHK=2)
    if (hs && *hs) {
        std::vector<char> img;
        if (std::FILE* fp = std::fopen(hs, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            img.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(img.data(), 1, img.size(), fp) != img.size()) img.clear();
            std::fclose(fp);
        }
        if (img.empty()) {
            fail(c, std::string("moedown97: cannot read ") + hs);
            return;
        }
        xm = hip::Module::loadData(img.data(), c.mod.arch());
        cand_safe = static_cast<bool>(xm.getFunctionOpt("md97_skipsafe_v1"));
        if (!cand_safe)
            fail(c, std::string("moedown97: candidate object ") + hs +
                        " has no md97_skipsafe_v1 (built before feat026-moe-skipsafe); rebuild with -DWHIRL_MD97_DEV=1 from this branch; "
                        "its candidates do not run the id -1 cases");
        if (xm.getFunctionOpt("md97_negchk_v1")) {
            negp = xm.getGlobal("md97_negid_hits");
            neg_fault = static_cast<bool>(xm.getFunctionOpt("md97_negchk_fault_v1"));
        } else {
            c.rep.skip("moedown97", "id -1 address check",
                       "candidate object not instrumented (build it with -DWHIRL_MD97_DEV=1 -DWHIRL_MD97_NEGCHK=1; runner step kt_negchk)");
        }
        // candidate list: dn1..dn10, df1..df6, gu1..gu4 (WHIRL_KT_MD97_IDS=dn3,df2,gu1 limits it; a listed id must exist)
        std::vector<std::string> want;
        const char* e = std::getenv("WHIRL_KT_MD97_IDS");
        const bool listed = e && *e;
        if (listed) {
            for (const char* q = e; *q;) {
                const char* s = q;
                while (*q && *q != ',') ++q;
                want.emplace_back(s, q);
                if (*q == ',') ++q;
            }
        } else {
            for (int i = 1; i <= 10; ++i) want.push_back("dn" + std::to_string(i));
            for (int i = 1; i <= 6; ++i) want.push_back("df" + std::to_string(i));
            for (int i = 1; i <= 4; ++i) want.push_back("gu" + std::to_string(i));
        }
        for (const std::string& id : want) {
            const std::string fam = id.substr(0, 2);
            const std::string nm = "md97" + id + (fam == "df" ? "_mxfp4" : "_mxfp4w");
            const hip::Function f = xm.getFunctionOpt(nm.c_str());
            if (!f) {
                if (listed) fail(c, "moedown97: WHIRL_KT_MD97_IDS lists " + id + " but the candidate object has no " + nm);
                continue;
            }
            (fam == "dn" ? dn : fam == "df" ? df : gu).push_back({nm, f});
        }
        // candidate main object: a candidate vs a candidate proves nothing
        auto refuse = [&](std::vector<Cand>& v, bool twin, const char* name) {
            if (!twin || v.empty()) return;
            fail(c, std::string("moedown97: the main code object is a candidate build for ") + name + " (exports " + name +
                        "_shipped, a WHIRL_MD97_* tune id): md97 vs its " + name +
                        " would compare a candidate with a candidate; run with the shipped main object (WHIRL_CODE_OBJECT cleared)");
            v.clear();
        };
        refuse(dn, static_cast<bool>(twin_dn), "moe_down_mxfp4w");
        refuse(df, static_cast<bool>(twin_df), "moe_down_f_mxfp4");
        refuse(gu, static_cast<bool>(twin_gu), "moe_gu_mxfp4w");
        if (dn.empty() && df.empty() && gu.empty() && !any_twin) {
            c.rep.skip("moedown97", "md97*", "not in the candidate code object");
            return;
        }
    }
    // the shipped bodies (the twins when the main object is a tune-id build)
    const hip::Function ship_dn = twin_dn ? twin_dn : c.k.moe_down_mxw, ship_df = twin_df ? twin_df : c.k.moe_down_f[ti],
                        ship_gu = twin_gu ? twin_gu : c.k.moe_gu_mxw;
    const bool do_dn = !dn.empty() || twin_dn, do_df = !df.empty() || twin_df, do_gu = !gu.empty() || twin_gu;
    const float qnan = [] {
        const std::uint32_t b = 0xFFC00000u;  // NaN as an f32 scale and after the gfx1151 pk_d decode
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }();

    // ---------------- down + combine (int8: moe_down_mxfp4w, f32: moe_down_f_mxfp4)
    struct DCase {
        int E, F, K, R;
        std::vector<int> ns;
        bool shared, skip;
        const char* what;
        int drop = 0;  // moeskip tails (routing drop mode)
    };
    std::vector<DCase> dcases = {
        {2048, 512, 8, 32, {1, 2, 4, 5, 16}, true, false, "Ornith 2048x512 K8"},
        {2048, 512, 8, 32, {1, 4}, false, false, "Ornith, no shared expert"},
        {2048, 512, 8, 32, {2}, true, true, "Ornith, skip flag set"},
        {2048, 1024, 8, 16, {1, 3}, true, false, "F 1024 (nb 32)"},
        {2048, 512, 8, 32, {1, 2, 5, 16}, true, false, "Ornith, id -1 tails (random kept count)", 1},
        {2048, 256, 8, 16, {1, 3}, true, false, "F 256 (nb 8: 4 experts per lane step)"},
        {2048, 768, 8, 12, {1, 2}, true, false, "F 768 (nb 24)"},
        {2048, 1536, 8, 12, {1, 2}, true, false, "F 1536 (nb 48)"},
        {1024, 2560, 4, 8, {1, 2}, true, false, "F 2560 (nb 80)"},
        {2048, 512, 1, 8, {1, 4}, true, false, "K 1"},
        {2048, 512, 3, 8, {1, 5}, true, false, "K 3"},
        {2048, 512, 6, 16, {1, 2}, true, false, "K 6"},
        {2000, 512, 8, 32, {1, 3}, true, false, "E 2000 (partial last block)"},
        {2048, 512, 8, 32, {1, 4}, false, false, "Ornith, id -1 tails, no shared expert", 1},
        {2048, 512, 8, 32, {2}, true, false, "Ornith, every token K-1 kept (last step)", 2},
        {2048, 512, 8, 32, {3}, true, false, "Ornith, every token 1 kept", 3},
        {2048, 512, 8, 32, {2}, true, true, "Ornith, id -1 tails + skip flag", 1},
        {2048, 256, 8, 16, {3, 16}, true, false, "F 256, id -1 tails (kept boundary inside a lane step)", 1},
        {2048, 1024, 8, 16, {2, 16}, true, false, "F 1024, id -1 tails (NG full batches at kt >= 4)", 1},
        {2048, 1536, 8, 12, {2}, true, false, "F 1536, id -1 tails", 1},
        {2048, 512, 3, 8, {5}, true, false, "K 3, id -1 tails", 1},
        {2048, 512, 6, 16, {4}, true, false, "K 6, id -1 tails", 1},
        {2000, 512, 8, 32, {3}, true, false, "E 2000, id -1 tails", 1},
        // review C2: NG full batch(es) followed by a dropped tail, fixed (kt 7 for every token)
        {2048, 1024, 8, 16, {2}, true, false, "F 1024, every token K-1 kept (NG full batches + dropped tail)", 2},
        {2048, 1536, 8, 12, {2}, true, false, "F 1536, every token K-1 kept (KB 8 full batch + dropped tail)", 2},
    };
    if (c.quick) dcases.resize(5);  // incl. the first id -1 case
    std::size_t runs_q = 0, dead_q = 0, runs_f = 0, dead_f = 0;
    Inv twin_q, twin_f, direct_q;
    Inv direct_f;
    CpuAcc cpu_q, cpu_f;
    // id -1 address check around one candidate run (no-op without an instrumented object)
    auto negBegin = [&] {
        if (!negp) return;
        const std::uint32_t z = 0;
        hip::upload(negp, &z, 4);
    };
    auto negEnd = [&](Cand& cd, bool drop, const char* what, int n) {
        if (!negp) return;
        std::uint32_t h = 0;
        hip::download(&h, negp, 4);
        if (drop) ++cd.neg_runs;
        if (h == 0) return;
        cd.neg_hits += h;
        if (cd.first_neg.empty()) cd.first_neg = std::string(what) + " n " + std::to_string(n) + ": " + std::to_string(h) + " id -1 addresses";
    };
    for (const DCase& cs : dcases) {
        const int E = cs.E, F = cs.F, K = cs.K, R = cs.R, nb = F / 32;
        const HostMat down = randomMat(c, QType::mxfp4, R * E, F);
        Buf dW(down.data);
        for (int n : cs.ns) {
            std::vector<int> hids, kts;
            std::vector<float> hw;
            routing(c, n, R, K, hids, hw, cs.drop, nullptr, &kts);
            const std::size_t na = static_cast<std::size_t>(n) * K * F;
            std::vector<float> act = c.randn(na, 2.0f);
            Buf dact(act), aq(na), ad(na / 32 * 4);
            hip::launch(c.k.quantize_q8, {cdiv(na, 256), 1, 1}, {256, 1, 1}, 0, c.s, dact.p(), aq.p(), ad.p(), static_cast<int>(na));
            c.sync();
            const std::vector<std::int8_t> aqh = aq.down<std::int8_t>(na);
            std::vector<float> adh = ad.down<float>(na / 32);
            if (cs.drop) {  // poison the dropped slots: reading one gives NaN (0 * NaN), never a silent 0 * finite
                for (std::size_t p = 0; p < hids.size(); ++p)
                    if (hids[p] < 0) {
                        for (int b = 0; b < nb; ++b) adh[p * nb + b] = qnan;
                        for (int i = 0; i < F; ++i) act[p * F + i] = std::nanf("");
                    }
                ad.up(adh);
                dact.up(act);
            }
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
            const std::vector<float> hsg = c.randu(static_cast<std::size_t>(n), 0.1f, 0.9f);
            Buf ids(hids), w(hw), sg(hsg), dysh(ysh), dskip(std::vector<int>{1});
            c.sync();
            const DevPtr psg = cs.shared ? sg.p() : DevPtr{0}, pysh = cs.shared ? dysh.p() : DevPtr{0};
            auto runq = [&](hip::Function f) {
                Buf dx(x0);
                hip::launch(f, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, aq.p(), ad.p(), ids.p(),
                            w.p(), psg, pysh, dx.p(), F, K, cs.skip ? dskip.p() : DevPtr{0});
                c.sync();
                return dx.down<float>(x0.size());
            };
            auto runf = [&](hip::Function f) {  // precise: no skip argument
                Buf dx(x0);
                hip::launch(f, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, dact.p(), ids.p(), w.p(),
                            psg, pysh, dx.p(), F, K);
                c.sync();
                return dx.down<float>(x0.size());
            };
            // compacted problem: token t alone with K' = kt (its kept slots), so no id -1 reaches the kernel
            auto compact = [&](hip::Function f, bool q8) {
                std::vector<float> out = x0;
                for (int t = 0; t < n; ++t) {
                    const int kt = kts[static_cast<std::size_t>(t)];
                    const std::size_t p0 = static_cast<std::size_t>(t) * K;
                    Buf id1(std::vector<int>(hids.begin() + static_cast<std::ptrdiff_t>(p0), hids.begin() + static_cast<std::ptrdiff_t>(p0 + kt)));
                    Buf w1(std::vector<float>(hw.begin() + static_cast<std::ptrdiff_t>(p0), hw.begin() + static_cast<std::ptrdiff_t>(p0 + kt)));
                    Buf x1(std::vector<float>(x0.begin() + static_cast<std::ptrdiff_t>(t) * E, x0.begin() + static_cast<std::ptrdiff_t>(t + 1) * E));
                    Buf y1(std::vector<float>(ysh.begin() + static_cast<std::ptrdiff_t>(t) * E, ysh.begin() + static_cast<std::ptrdiff_t>(t + 1) * E));
                    Buf s1(std::vector<float>{hsg[static_cast<std::size_t>(t)]});
                    const DevPtr ps = cs.shared ? s1.p() : DevPtr{0}, py = cs.shared ? y1.p() : DevPtr{0};
                    if (q8) {
                        Buf q1(std::vector<std::int8_t>(aqh.begin() + static_cast<std::ptrdiff_t>(p0 * F), aqh.begin() + static_cast<std::ptrdiff_t>((p0 + kt) * F)));
                        Buf d1(std::vector<float>(adh.begin() + static_cast<std::ptrdiff_t>(p0 * nb), adh.begin() + static_cast<std::ptrdiff_t>((p0 + kt) * nb)));
                        hip::launch(f, {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, q1.p(), d1.p(), id1.p(), w1.p(), ps, py, x1.p(), F, kt,
                                    cs.skip ? dskip.p() : DevPtr{0});
                    } else {
                        Buf a1(std::vector<float>(act.begin() + static_cast<std::ptrdiff_t>(p0 * F), act.begin() + static_cast<std::ptrdiff_t>((p0 + kt) * F)));
                        hip::launch(f, {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, a1.p(), id1.p(), w1.p(), ps, py, x1.p(), F, kt);
                    }
                    c.sync();
                    const std::vector<float> r = x1.down<float>(static_cast<std::size_t>(E));
                    std::copy(r.begin(), r.end(), out.begin() + static_cast<std::ptrdiff_t>(t) * E);
                }
                return out;
            };
            // independent CPU reference (kept slots, exact weights, double accumulation) on a row subset
            auto cpu = [&](CpuAcc& acc, const std::vector<float>& got, bool q8) {
                if (cs.skip) return;
                std::vector<float> wr(static_cast<std::size_t>(F));
                for (int t = 0; t < n; ++t)
                    for (int r = 0; r < E; r = r + 97 < E ? r + 97 : (r < E - 1 ? E - 1 : E)) {  // every 97th row + the last
                        double s = 0, a = 0;
                        for (int k = 0; k < kts[static_cast<std::size_t>(t)]; ++k) {
                            const std::size_t p = static_cast<std::size_t>(t) * K + k;
                            ref::dequantRow(QType::mxfp4, down.data.data() + (static_cast<std::size_t>(hids[p]) * E + r) * down.row_bytes, F, wr.data());
                            double y = 0, ya = 0;
                            for (int i = 0; i < F; ++i) {
                                const double xv = q8 ? static_cast<double>(aqh[p * F + i]) * ref::xdScale(adh[(p * F + i) / 32])
                                                     : static_cast<double>(act[p * F + i]);
                                y += wr[static_cast<std::size_t>(i)] * xv;
                                ya += std::fabs(wr[static_cast<std::size_t>(i)] * xv);
                            }
                            s += hw[p] * y;
                            a += hw[p] * ya;
                        }
                        const std::size_t o = static_cast<std::size_t>(t) * E + r;
                        const double sh = cs.shared ? static_cast<double>(hsg[static_cast<std::size_t>(t)]) * ysh[o] : 0.0;
                        acc.got.push_back(got[o]);
                        acc.ref.push_back(x0[o] + s + sh);
                        acc.sc.push_back(std::fabs(x0[o]) + a + std::fabs(sh));
                    }
            };
            if (do_dn) {
                const std::vector<float> ship = cs.drop ? compact(ship_dn, true) : runq(ship_dn);
                ++runs_q;
                const bool same = sameBits(ship, x0);
                if (cs.skip ? !same : same) ++dead_q;
                if (cs.drop && skipaware) check(direct_q, runq(ship_dn), ship, cs.what, n, "direct", "compacted");
                if (twin_dn && (!cs.drop || main_safe)) check(twin_q, runq(c.k.moe_down_mxw), ship, cs.what, n, "moe_down_mxfp4w", "_shipped");
                if (!cs.drop || cand_safe)
                    for (Cand& cd : dn) {
                        negBegin();
                        const std::vector<float> y = runq(cd.f);
                        negEnd(cd, cs.drop != 0, cs.what, n);
                        compare(cd, y, ship, cs.what, n);
                    }
                cpu(cpu_q, ship, true);
            }
            if (do_df && !cs.skip) {
                const std::vector<float> ship = cs.drop ? compact(ship_df, false) : runf(ship_df);
                ++runs_f;
                if (sameBits(ship, x0)) ++dead_f;
                // direct id -1 run of the shipped moe_down_f_mxfp4 only when it is the feat027-moeskip-r1 one (27bfb54 patches
                // it, b8a041a does not; see df_skipaware). Precise / balance never produce -1 (r1 ignores the override there).
                if (cs.drop && df_skipaware) check(direct_f, runf(ship_df), ship, cs.what, n, "direct", "compacted");
                if (twin_df && (!cs.drop || main_safe)) check(twin_f, runf(c.k.moe_down_f[ti]), ship, cs.what, n, "moe_down_f_mxfp4", "_shipped");
                if (!cs.drop || cand_safe)
                    for (Cand& cd : df) {
                        negBegin();
                        const std::vector<float> y = runf(cd.f);
                        negEnd(cd, cs.drop != 0, cs.what, n);
                        compare(cd, y, ship, cs.what, n);
                    }
                cpu(cpu_f, ship, false);
            }
        }
    }
    const char* dsweep = "Ornith n 1..16, F 256/768/1024/1536/2560, K 1/3/6, E 2000, no shared, skip, id -1 tails, NG full + dropped tail";
    if (do_dn) {
        live(c, "moe_down_mxfp4w", runs_q, dead_q);
        if (!neg_fault) report(c, dn, "moe_down_mxfp4w", dsweep);
        if (negp) reportNeg(c, dn, neg_fault);
        reportCpu(c, "moedown97: reference moe_down_mxfp4w" + std::string(twin_dn ? "_shipped" : "") + " vs CPU (independent; kept slots, tol 1e-4)",
                  cpu_q, 1e-4, 1e-6);
        if (twin_dn) reportInv(c, "moedown97: moe_down_mxfp4w (tune-id build) == moe_down_mxfp4w_shipped, bitwise (id -1 cases vs the compacted _shipped)", twin_q);
        if (skipaware) reportInv(c, "moedown97: shipped moe_down_mxfp4w on id -1 tails == the compacted problem (moeskip merged)", direct_q);
    }
    if (do_df) {
        live(c, "moe_down_f_mxfp4", runs_f, dead_f);
        if (!neg_fault) report(c, df, "moe_down_f_mxfp4", dsweep);
        if (negp) reportNeg(c, df, neg_fault);
        reportCpu(c, "moedown97: reference moe_down_f_mxfp4" + std::string(twin_df ? "_shipped" : "") + " vs CPU (independent; kept slots, tol 1e-5)",
                  cpu_f, 1e-5, 1e-7);
        if (twin_df) reportInv(c, "moedown97: moe_down_f_mxfp4 (tune-id build) == moe_down_f_mxfp4_shipped, bitwise (id -1 cases vs the compacted _shipped)", twin_f);
        if (df_skipaware)
            reportInv(c, "moedown97: shipped moe_down_f_mxfp4 on id -1 tails == the compacted problem (moeskip-r1 merged)", direct_f);
        else if (skipaware)
            c.rep.skip("moedown97", "direct id -1 moe_down_f_mxfp4",
                       "moeskip present but no moe_f_skipsafe_v1 (b8a041a does not patch moe_f.hip); WHIRL_KT_MD97_DF_DIRECT=1 once r1 is merged");
    }

    // ---------------- gate / up (moe_gu_mxfp4w)
    if (do_gu) {
        struct GCase {
            int C, ff, K, R;
            std::vector<int> ns;
            bool skip;
            const char* what;
            int drop = 0;
        };
        std::vector<GCase> gcases = {
            {2048, 512, 8, 32, {1, 3, 16}, false, "Ornith C 2048 ff 512 K8"},
            {2048, 512, 8, 32, {2}, true, "Ornith, skip flag set"},
            {2048, 512, 8, 32, {1, 3, 16}, false, "Ornith, id -1 tails (blocks of dropped slots write nothing)", 1},
            {768, 256, 4, 8, {1, 2}, false, "C 768 (1 partial step)"},
            {2304, 256, 4, 8, {1, 2}, false, "C 2304 (3 steps, last partial)"},
            {4096, 128, 4, 8, {1, 2}, false, "C 4096 (4 steps)"},
            {2048, 100, 8, 16, {1, 2}, false, "ff 100 (partial last block)"},
            {2048, 512, 8, 32, {2}, false, "Ornith, every token 1 kept", 3},
            {2304, 256, 4, 8, {3}, false, "C 2304, id -1 tails", 1},
            {2048, 100, 8, 16, {2}, false, "ff 100, id -1 tails", 1},
        };
        if (c.quick) gcases.resize(3);
        std::size_t runs = 0, dead = 0;
        Inv twin_g, direct_g;
        CpuAcc cpu_g;
        for (const GCase& cs : gcases) {
            const int C = cs.C, ff = cs.ff, K = cs.K, R = cs.R;
            const HostMat gate = randomMat(c, QType::mxfp4, R * ff, C), up = randomMat(c, QType::mxfp4, R * ff, C);
            Buf dg(gate.data), du(up.data);
            for (int n : cs.ns) {
                std::vector<int> hids, hvalid;
                std::vector<float> hw;
                routing(c, n, R, K, hids, hw, cs.drop, &hvalid, nullptr);
                const std::size_t nh = static_cast<std::size_t>(n) * C;
                const std::vector<float> h = c.randn(nh, 2.0f);
                Buf dh(h), xq(nh), xd(nh / 32 * 4), ids(hids), idv(hvalid), dskip(std::vector<int>{1});
                hip::launch(c.k.quantize_q8, {cdiv(nh, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq.p(), xd.p(), static_cast<int>(nh));
                const std::size_t ny = static_cast<std::size_t>(n) * K * ff;
                const std::vector<float> y0 = c.randn(2 * ny);  // poison: both halves must be written (kept slots) / kept (dropped)
                c.sync();
                auto run = [&](hip::Function f, const Buf& pid) {
                    Buf y1(std::vector<float>(y0.begin(), y0.begin() + static_cast<std::ptrdiff_t>(ny)));
                    Buf y2(std::vector<float>(y0.begin() + static_cast<std::ptrdiff_t>(ny), y0.end()));
                    hip::launch(f, {cdiv(2 * ff, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, ff, 2 * ff,
                                xq.p(), xd.p(), pid.p(), y1.p(), y2.p(), C, K, cs.skip ? dskip.p() : DevPtr{0});
                    c.sync();
                    std::vector<float> a = y1.down<float>(ny);
                    const std::vector<float> b = y2.down<float>(ny);
                    a.insert(a.end(), b.begin(), b.end());
                    return a;
                };
                // reference: the shipped kernel with valid ids, then the dropped slots' rows back to the poison
                std::vector<float> ship = run(ship_gu, cs.drop ? idv : ids);
                if (cs.drop)
                    for (std::size_t p = 0; p < hids.size(); ++p)
                        if (hids[p] < 0)
                            for (int half = 0; half < 2; ++half)
                                for (int r = 0; r < ff; ++r) {
                                    const std::size_t o = static_cast<std::size_t>(half) * ny + p * ff + r;
                                    ship[o] = y0[o];
                                }
                ++runs;
                const bool same = sameBits(ship, y0);
                if (cs.skip ? !same : same) ++dead;
                if (cs.drop && skipaware) check(direct_g, run(ship_gu, ids), ship, cs.what, n, "direct", "reference");
                if (twin_gu && (!cs.drop || main_safe)) check(twin_g, run(c.k.moe_gu_mxw, ids), ship, cs.what, n, "moe_gu_mxfp4w", "_shipped");
                if (!cs.drop || cand_safe)
                    for (Cand& cd : gu) {
                        negBegin();
                        const std::vector<float> y = run(cd.f, ids);
                        negEnd(cd, cs.drop != 0, cs.what, n);
                        compare(cd, y, ship, cs.what, n);
                    }
                if (!cs.skip) {  // independent CPU reference on a row subset of the kept slots
                    const std::vector<std::int8_t> xqh = xq.down<std::int8_t>(nh);
                    const std::vector<float> xdh = xd.down<float>(nh / 32);
                    std::vector<float> wr(static_cast<std::size_t>(C));
                    for (std::size_t p = 0; p < hids.size(); ++p) {
                        if (hids[p] < 0) continue;
                        const std::size_t t = p / static_cast<std::size_t>(K);
                        for (int half = 0; half < 2; ++half)
                            for (int r = static_cast<int>(p % 7); r < ff; r += 37) {
                                const HostMat& m = half ? up : gate;
                                ref::dequantRow(QType::mxfp4, m.data.data() + (static_cast<std::size_t>(hids[p]) * ff + r) * m.row_bytes, C, wr.data());
                                double y = 0, a = 0;
                                for (int i = 0; i < C; ++i) {
                                    const double xv = static_cast<double>(xqh[t * C + i]) * ref::xdScale(xdh[(t * C + i) / 32]);
                                    y += wr[static_cast<std::size_t>(i)] * xv;
                                    a += std::fabs(wr[static_cast<std::size_t>(i)] * xv);
                                }
                                cpu_g.got.push_back(ship[static_cast<std::size_t>(half) * ny + p * ff + r]);
                                cpu_g.ref.push_back(y);
                                cpu_g.sc.push_back(a);
                            }
                    }
                }
            }
        }
        live(c, "moe_gu_mxfp4w", runs, dead);
        if (!neg_fault) report(c, gu, "moe_gu_mxfp4w", "Ornith n 1..16, C 768/2304/4096, ff 100, skip, id -1 tails");
        if (negp) reportNeg(c, gu, neg_fault);
        reportCpu(c, "moedown97: reference moe_gu_mxfp4w" + std::string(twin_gu ? "_shipped" : "") + " vs CPU (independent; kept slots, tol 1e-4)", cpu_g,
                  1e-4, 1e-6);
        if (twin_gu) reportInv(c, "moedown97: moe_gu_mxfp4w (tune-id build) == moe_gu_mxfp4w_shipped, bitwise (id -1 cases vs the masked _shipped)", twin_g);
        if (skipaware) reportInv(c, "moedown97: shipped moe_gu_mxfp4w on id -1 slots writes nothing for them (moeskip merged)", direct_g);
    }
}

}  // namespace kt
