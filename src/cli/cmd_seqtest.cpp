// whirl seqtest: the multi-sequence (server) paths against single-sequence runs (a gate tool; arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

const char* k_help_seqtest =
    "usage: whirl seqtest MODEL.gguf [--decode N] [--slot-ctx N] [--device SPEC]\n"
    "\n"
    "Runs the multi-sequence (server) paths against the single-sequence reference on two\n"
    "prompts: segmented prefill, batched decode rows, batched verify with literal drafts.\n"
    "Every token must match; prints 'seqtest: ok' or FAIL (exit 1).\n"
    "\n"
    "options:\n"
    "  --decode N                 tokens generated per sequence (default 24)\n"
    "  --device SPEC, -dev SPEC   GPU: pci:BUS:DEVICE (Task Manager > Performance > GPU > Physical location: PCI bus 71,\n"
    "                             device 0 -> pci:71:0), an index from 'whirl devices', r9700, 8060s, or a part of the\n"
    "                             name. Default: the GPU with the most GPU memory\n"
    WHIRL_HELP_MODES
    "  -h, --help                 this text\n";

// Multi-sequence (server) paths against the single-sequence reference:
// 1. solo per sequence: chunked prefill + greedy one-token forwards;
// 2. both sequences: segmented prefill (prefillSegs) + batched decode rows
//    (verifyBatchEnqueue, no drafts), then a batched verify with literal
//    drafts taken from the reference continuation (all must be accepted).
// Every token must match the solo run.
namespace {

// logits hash of row r of the last forward
u64 rowHash(q::Model& m, u32 r) {
    std::vector<float> lg(m.cfg.n_vocab);
    hip::download(lg.data(), m.scratch().logits + static_cast<u64>(r) * m.cfg.n_vocab * 4, lg.size() * 4);
    return fnvF32(lg);
}

}  // namespace

int cmdSeqtest(const Args& a) {
    if (a.pos.empty()) throw std::runtime_error("seqtest: missing MODEL.gguf");
    gguf::File f = gguf::File::open(a.pos[0]);
    selectDevice(a, nullptr);
    Tok tok(f);
    // --slot-ctx N: the context per sequence (P3-6: the solo digest must not depend on it)
    // (12288: room for the chunk schedule gate's long prompts of long_n / long_n2 tokens)
    const u32 long_n = 9500, long_n2 = 8232;
    // CLI-SCHED gate: prompt lengths whose old CLI chunking (max_batch forwards) was off the server's
    // schedule; --sched-lens N,N,... adds lengths (e.g. 32800; the context per sequence grows to fit)
    std::vector<u32> sched_lens = {4000, 4100, long_n2};  // 4000: the golden suite single long4k prompt
    if (auto v = a.get("--sched-lens"))
        for (u32 x : parseIds(*v)) sched_lens.push_back(x);
    u32 slot_ctx = a.get("--slot-ctx") ? static_cast<u32>(std::stoul(*a.get("--slot-ctx"))) : 12288;
    const u32 n_gen = a.get("--decode") ? static_cast<u32>(std::stoul(*a.get("--decode"))) : 24;
    for (u32 x : sched_lens) slot_ctx = std::max<u32>(slot_ctx, (x + 64 + q::kv_page - 1) / q::kv_page * q::kv_page);
    const std::vector<u32> pa = tok.encode(chatWrap(std::string("Write a C function that reverses a singly linked list, and explain it.") + k_bench_text, false));
    const std::vector<u32> pb = tok.encode(chatWrap("請用繁體中文說明 TCP 三向交握的過程，並用 Python 寫一個簡單的 socket 伺服器。", false));
    q::LoadOptions lo;
    lo.kv_req = q::kvRequestFromEnv();
    lo.numerics = modeRequest(a);
    q::LoadStats stats;
    auto mp = q::Model::load(f, 0, stats, lo);
    q::Model& m = *mp;
    {
        std::string log;
        q::loadOrTune(m, a.pos[0], log);
        out(log);
    }
    m.setupSeqs(2, slot_ctx);
    m.allocKvPool(2 * slot_ctx);
    for (u32 s = 0; s < 2; ++s) {
        std::vector<std::int32_t> phys(slot_ctx / q::kv_page);
        for (std::size_t j = 0; j < phys.size(); ++j) phys[j] = static_cast<std::int32_t>(s * phys.size() + j);
        m.mapPages(s, 0, phys);
    }
    const std::vector<u32>* prompts[2] = {&pa, &pb};
    // 1. solo reference
    // lh[s][i]: logits hash of the solo row that produced ref[s][i] (the CLI plain decode head)
    std::vector<u32> ref[2];
    std::vector<u64> lh[2];
    for (u32 s = 0; s < 2; ++s) {
        m.selectSeq(s);
        m.reset();
        m.prefill(*prompts[s], 0);
        u32 t = m.argmax();
        lh[s].push_back(rowHash(m, 0));
        u32 pos = static_cast<u32>(prompts[s]->size());
        for (u32 i = 0; i < n_gen; ++i) {
            ref[s].push_back(t);
            m.step(t, pos++);
            t = m.argmax();
            lh[s].push_back(rowHash(m, 0));
        }
        ref[s].push_back(t);
    }
    {
        u64 dg = 0xcbf29ce484222325ull;
        for (u32 s = 0; s < 2; ++s)
            for (u64 v : lh[s]) dg = (dg ^ v) * 0x100000001b3ull;
        out(fmt("  solo logits digest (slot ctx %u, max_ctx %u): %016llx\n", slot_ctx, m.max_ctx, static_cast<unsigned long long>(dg)));
    }
    // P2-1: every head input (plain decode, CLI verify rows, server batched rows) must give the
    // same logits bits as the solo row; lh_bad counts rows whose logits hash differs
    u32 lh_bad = 0, lh_total = 0;
    bool ok = true;
    // 2. segmented prefill + batched rows
    for (u32 s = 0; s < 2; ++s) {
        m.selectSeq(s);
        m.reset();
    }
    const q::PSeg segs[2] = {{0, pa, 0, 0, pa.size(), std::nullopt}, {1, pb, 0, 0, pb.size(), std::nullopt}};
    const bool segmented = m.canSegment() && pa.size() + pb.size() <= m.max_batch;
    u32 next[2];
    if (segmented) {
        m.prefillSegs(segs, false);
        for (u32 s = 0; s < 2; ++s) {
            m.selectSeq(0);
            m.segLogitsToFront(s);
            next[s] = m.argmax();
        }
    } else {
        for (u32 s = 0; s < 2; ++s) {
            m.selectSeq(s);
            m.prefill(*prompts[s], 0);
            next[s] = m.argmax();
        }
    }
    out(fmt("  prefill: %s; first tokens %s\n", segmented ? "segmented (prefillSegs)" : "solo (cannot segment)",
            next[0] == ref[0][0] && next[1] == ref[1][0] ? "match" : "DIFFER"));
    ok = ok && next[0] == ref[0][0] && next[1] == ref[1][0];
    u32 pos[2] = {static_cast<u32>(pa.size()), static_cast<u32>(pb.size())};
    const bool fused = m.fusedDecode();
    u32 matched = 0, total = 0;
    std::vector<std::int32_t> ctl(2 * q::ctl_words);
    const u32 half = n_gen / 2;
    for (u32 i = 0; i < half; ++i) {
        std::vector<q::VSeg> vs;
        for (u32 s = 0; s < (fused ? 2u : 1u); ++s) vs.push_back({s, next[s], 0, pos[s], {}});
        m.verifyBatchEnqueue(vs);
        m.readCtl(ctl);
        for (const q::VSeg& v : vs) {
            lh_total += 1;
            lh_bad += rowHash(m, v.seq) != lh[v.seq][i + 1] ? 1 : 0;
            next[v.seq] = static_cast<u32>(ctl[v.seq * q::ctl_words + q::ctl_rows]);
            pos[v.seq] += 1;
            total += 1;
            matched += next[v.seq] == ref[v.seq][i + 1] ? 1 : 0;
        }
    }
    out(fmt("  batched decode rows (%s): %u/%u tokens match the solo run\n", fused ? "2 sequences" : "1 sequence, unfused path", matched, total));
    ok = ok && matched == total;
    if (fused) {
        // verify with literal drafts = the reference continuation (all accepted)
        const u32 nd = std::min<u32>(4, n_gen - half - 1);
        std::vector<u32> dr[2];
        std::vector<q::VSeg> vs;
        for (u32 s = 0; s < 2; ++s) {
            dr[s].assign(ref[s].begin() + half + 1, ref[s].begin() + half + 1 + nd);
            vs.push_back({s, next[s], nd, pos[s], dr[s]});
        }
        m.verifyBatchEnqueue(vs);
        m.readCtl(ctl);
        u32 good = 0;
        for (u32 s = 0; s < 2; ++s)
            for (u32 r = 0; r <= nd; ++r) {
                good += static_cast<u32>(ctl[s * q::ctl_words + q::ctl_rows + r]) == ref[s][half + 1 + r] ? 1 : 0;
                lh_total += 1;
                lh_bad += rowHash(m, s * (nd + 1) + r) != lh[s][half + 1 + r] ? 1 : 0;
            }
        out(fmt("  batched verify with %u literal drafts per sequence: %u/%u rows match the solo run\n", nd, good, 2 * (nd + 1)));
        ok = ok && good == 2 * (nd + 1);
    }
    // the CLI's MTP / n-gram verify (Model::verify) over the reference continuation of sequence 0
    {
        m.selectSeq(0);
        m.reset();
        m.prefill(pa, 0);
        const u32 nv = std::min<u32>(8, n_gen);
        std::vector<u32> vo(nv);
        m.verify(std::span<const u32>(ref[0]).subspan(0, nv), static_cast<u32>(pa.size()), vo);
        u32 vbad = 0;
        for (u32 r = 0; r < nv; ++r) vbad += rowHash(m, r) != lh[0][r + 1] ? 1 : 0;
        lh_total += nv;
        lh_bad += vbad;
        out(fmt("  CLI verify rows (%u): %u/%u logits hashes differ from the solo rows\n", nv, vbad, nv));
    }
    // FR-DFL: the literal-draft verify of whirl chat's DFlash2 / n-gram decode (verifyEnqueueEx, the
    // dflashDecode path) with a DFlash2 block of 8 (7 drafts) and 16 (15 drafts): every row's logits
    // bits must equal the solo decode row, in every numerics mode (fast: 4-bit KV)
    for (const u32 nd : {7u, 15u}) {
        if (nd + 1 > n_gen || nd + 1 > q::max_small_batch) continue;
        m.selectSeq(0);
        m.reset();
        m.prefill(pa, 0);
        const u32 first = m.argmax();
        m.verifyEnqueueEx(first, nd, std::span<const u32>(ref[0]).subspan(1, nd), static_cast<u32>(pa.size()));
        std::array<u32, q::max_small_batch> vo{};
        (void)m.readCycle(std::span<u32>(vo.data(), nd + 1), std::span<u32>(vo.data(), 0));
        u32 vbad = 0, tbad = 0;
        for (u32 r = 0; r <= nd; ++r) {
            vbad += rowHash(m, r) != lh[0][r + 1] ? 1 : 0;
            tbad += vo[r] != ref[0][r + 1] ? 1 : 0;
        }
        lh_total += nd + 1;
        lh_bad += vbad;
        ok = ok && tbad == 0 && first == ref[0][0];
        out(fmt("  literal-draft verify (verifyEnqueueEx, %u drafts = %u rows): %u/%u logits hashes and %u tokens differ from the solo rows\n",
                nd, nd + 1, vbad, nd + 1, tbad));
    }
    // FIX-TEST-B gate (part of ok): the server's cold chunk schedule (include/whirl/prefill_schedule.h,
    // the functions the engine runs) on a golden long prompt of > 4096 tokens. Every execution the
    // server can choose for the same prompt must give the same last-row logits bits: forwards of 1, 2
    // or more chunks (decode floor / WHIRL_PREFILL_CHUNK / WHIRL_PREFILL_BATCH), with or without the
    // TAIL-CK tail checkpoint, and a resume from a checkpoint at a schedule point (a prefix checkpoint
    // inside the prompt, the tail checkpoint). Splits off the schedule are printed for information only;
    // the CLI's prefill is checked by the CLI-SCHED gate below.
    u32 sched_bad = 0, sched_total = 0;
    if (slot_ctx < long_n + 64) {
        out(fmt("  chunk schedule: needs --slot-ctx >= %u (got %u)\n", long_n + 64, slot_ctx));
        ok = false;
    } else {
        m.selectSeq(0);
        const std::vector<u32> lp = goldenLongIds(tok, long_n);
        const std::span<const u32> ls(lp);
        const std::size_t N = lp.size(), th = tok.thinkOpenSplit(lp);
        const std::span<const std::uint32_t> noimg;
        const std::size_t tail = sched::tailPoint(N, 0, th, noimg);
        // run forwards ending at `ends` (the first starts at 0); the last row's logits hash
        auto runEnds = [&](std::span<const u32> ids, const std::vector<std::size_t>& ends) {
            m.reset();
            std::size_t cur = 0;
            for (std::size_t e : ends) {
                m.forward(ids.subspan(cur, e - cur), static_cast<u32>(cur));
                cur = e;
            }
            return rowHash(m, 0);
        };
        auto cold = [&](std::size_t exec, std::span<const std::size_t> stops) { return sched::forwardEnds(N, 0, th, noimg, 0, exec, stops, m.max_batch); };
        // resume at a schedule point r: the run that kept the checkpoint there (no forward crosses r),
        // then the resumed request's own forwards from r (the state at r is what the checkpoint holds)
        auto resumed = [&](std::size_t r, std::size_t exec) {
            const std::array<std::size_t, 1> st = {r};
            std::vector<std::size_t> e = sched::forwardEnds(N, 0, th, noimg, 0, exec, st, m.max_batch);
            e.erase(std::upper_bound(e.begin(), e.end(), r), e.end());
            if (e.empty() || e.back() != r) e.push_back(r);  // the checkpoint holds the state after a forward that ended at r
            const std::vector<std::size_t> e2 = sched::forwardEnds(N, 0, th, noimg, r, exec, {}, m.max_batch);
            e.insert(e.end(), e2.begin(), e2.end());
            return e;
        };
        auto endsStr = [](const std::vector<std::size_t>& e) {
            std::string r;
            for (std::size_t x : e) r += (r.empty() ? "" : ",") + std::to_string(x);
            return r;
        };
        const device::Profile& prof = device::profile(m.arch);
        const std::size_t ex = sched::prefillExecFor(sched::prefillExecDefault(prof, m.cfg.moe), m.max_batch, prof.prefill_fill_batch),
                          ex_big = sched::prefillExecFor(16384, m.max_batch, prof.prefill_fill_batch), ex2 = sched::prefillExecFor(2048, m.max_batch);
        const std::array<std::size_t, 1> tail_stop = {tail};
        const std::size_t mid = sched::schedFloor(N, 0, th, noimg, N / 3);  // a prefix checkpoint inside the prompt
        const std::vector<std::size_t> ref_ends = cold(ex, tail_stop);
        const u64 ref = runEnds(ls, ref_ends);
        out(fmt("  chunk schedule (N %zu, tail checkpoint @%zu, exec %zu): reference = cold server forwards [%s]\n", N, tail, ex,
                endsStr(ref_ends).c_str()));
        struct Var {
            const char* name;
            std::vector<std::size_t> ends;
        };
        const Var vars[] = {
            {"cold, no tail checkpoint (parallel > 4)", cold(ex, {})},
            {"cold, 1-chunk forwards (decode floor)", cold(sched::prefill_chunk, tail_stop)},
            {"cold, 2-chunk forwards (WHIRL_PREFILL_CHUNK=2048)", cold(ex2, tail_stop)},
            {"cold, largest forwards (WHIRL_PREFILL_CHUNK)", cold(ex_big, tail_stop)},
            {"resume at the tail checkpoint", resumed(tail, ex)},
            {"resume at a prefix checkpoint", resumed(mid, ex)},
            {"resume at a prefix checkpoint, 1-chunk forwards", resumed(mid, sched::prefill_chunk)},
        };
        for (const Var& v : vars) {
            bool on = !v.ends.empty() && v.ends.back() == N;
            for (std::size_t e : v.ends) on = on && sched::onSchedule(N, 0, th, noimg, e);
            const bool same = on && runEnds(ls, v.ends) == ref;
            sched_bad += same ? 0 : 1;
            sched_total += 1;
            out(fmt("    %-48s [%s]: %s\n", v.name, endsStr(v.ends).c_str(), !on ? "FAIL (a forward ends off the schedule)" : same ? "==" : "DIFFER (FAIL)"));
        }
        // information only: one extra split the server never makes (added to the reference's
        // forwards)
        const std::pair<const char*, std::size_t> info[] = {
            {"split 64 rows after a schedule point", mid + 64},
            {"split 40 rows after a schedule point", mid + 40},
            {"split at 1000 (not 64-aligned)", 1000},
            {"last 40 rows as their own forward", N - 40},
            {"last 9 rows as their own forward (decode kernels)", N - 9},
        };
        for (const auto& [name, sp] : info) {
            std::vector<std::size_t> e = ref_ends;  // the reference's forwards with one more split at sp
            if (std::find(e.begin(), e.end(), sp) == e.end()) e.insert(std::upper_bound(e.begin(), e.end(), sp), sp);
            out(fmt("    info: %-50s: %s\n", name, runEnds(ls, e) == ref ? "==" : "DIFFER"));
        }
    }
    out(fmt("  chunk schedule: %u/%u server executions differ from the cold reference\n", sched_bad, sched_total));
    ok = ok && sched_bad == 0;
    // CLI-SCHED gate (part of ok): the CLI's prefill (prefillLikeServer, the code `whirl chat` and the golden
    // suite run) must give the same last-row logits bits as the server's forwards for the same prompt
    // (the engine's cold solo request: sched::forwardEnds with the TAIL-CK stop, exec from
    // WHIRL_PREFILL_CHUNK / max_batch), without and with the MTP block. Prompts: no-think chat prompts of
    // sched_lens tokens (not 64-aligned / short tails past a max_batch boundary), a thinking-mode prompt
    // (think-open split) and one with a long system message (system split). "old CLI" (max_batch
    // forwards, the CLI before CLI-SCHED) is printed for information.
    u32 cli_bad = 0, cli_total = 0;
    {
        m.selectSeq(0);
        const std::span<const std::uint32_t> noimg;
        const device::Profile& prof = device::profile(m.arch);
        const std::size_t ex = sched::prefillExecFor(sched::prefillExecDefault(prof, m.cfg.moe), m.max_batch, prof.prefill_fill_batch);
        auto runEndsMtp = [&](std::span<const u32> ids, const std::vector<std::size_t>& ends, bool mtp) {
            m.reset();
            std::size_t cur = 0;
            for (std::size_t e : ends) {
                if (mtp)
                    m.prefillMtpChunk(ids, 0, cur, e - cur, std::nullopt);
                else
                    m.forward(ids.subspan(cur, e - cur), static_cast<u32>(cur));
                cur = e;
            }
            return rowHash(m, 0);
        };
        struct P {
            std::string name;
            std::vector<u32> ids;
        };
        std::vector<P> ps;
        for (u32 n : sched_lens) ps.push_back({fmt("no-think, %u tokens", n), goldenLongIds(tok, n)});
        {
            // (k sections of the bench text, ~140 tokens each: thinking ~5k tokens; system message ~2.7k
            // tokens (>= sys_min, so the server splits there) + a ~1.7k-token user turn)
            auto doc = [](int k) {
                std::string d;
                for (int i = 1; i <= k; ++i) d += fmt("Section %d.\n", i) + std::string(k_bench_text) + "\n";
                return d;
            };
            ps.push_back({"thinking mode (think-open split)", tok.encode(chatWrap("Summarize these notes.\n\n" + doc(36), true))});
            ps.push_back({"long system message (system split)",
                          tok.encode("<|im_start|>system\nYou are a careful assistant. Reference notes:\n" + doc(19) + "<|im_end|>\n<|im_start|>user\n" +
                                     doc(12) + "List the three most important points.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")});
        }
        const bool has_mtp = m.mtp.has_value();
        for (const P& pp : ps) {
            const std::span<const u32> ids(pp.ids);
            const std::size_t n = ids.size();
            if (n + 64 > slot_ctx) {
                out(fmt("    CLI-SCHED %-36s: %zu tokens do not fit --slot-ctx %u (FAIL)\n", pp.name.c_str(), n, slot_ctx));
                ++cli_bad;
                ++cli_total;
                continue;
            }
            const std::size_t th = tok.thinkOpenSplit(pp.ids);
            const std::size_t sy = sched::sysSplit(sched::sysBoundary(ids, tok.im_start, tok.system, tok.im_end, tok.newline), n, sched::sys_min_default);
            // the split prompts must really split (else the case is vacuous)
            if ((pp.name.rfind("thinking", 0) == 0 && th == 0) || (pp.name.rfind("long system", 0) == 0 && sy == 0)) {
                out(fmt("    CLI-SCHED %-36s: no forced split in the prompt (N %zu) (FAIL)\n", pp.name.c_str(), n));
                ++cli_bad;
                ++cli_total;
                continue;
            }
            std::vector<std::size_t> st;
            if (n >= sched::tail_ck_min_default) {
                const std::size_t t = sched::tailPoint(n, sy, th, noimg);
                if (t > 0 && t != sy && t != th) st.push_back(t);
            }
            const std::vector<std::size_t> srv_ends = sched::forwardEnds(n, sy, th, noimg, 0, ex, st, m.max_batch);
            // the CLI before CLI-SCHED: max_batch forwards, the think-open token as its own step
            std::vector<std::size_t> old_ends;
            for (std::size_t off = 0; off < n;) {
                const std::size_t lim = (th > 0 && off < th) ? th : n;
                off += std::min<std::size_t>(lim - off, m.max_batch);
                old_ends.push_back(off);
            }
            for (int mt = 0; mt < (has_mtp ? 2 : 1); ++mt) {
                const bool mtp = mt == 1;
                if (mtp && pp.name.rfind("no-think", 0) == 0 && n != long_n2) continue;  // MTP: one length + the split prompts
                const u64 srv = runEndsMtp(ids, srv_ends, mtp);
                m.reset();
                prefillLikeServer(m, tok, ids, mtp);
                const u64 cli = rowHash(m, 0);
                const u64 old = runEndsMtp(ids, old_ends, mtp);
                cli_bad += cli == srv ? 0 : 1;
                cli_total += 1;
                std::string se;
                for (std::size_t e : srv_ends) se += (se.empty() ? "" : ",") + std::to_string(e);
                out(fmt("    CLI-SCHED %-36s%s (N %zu, sys %zu, think %zu, server forwards [%s]): CLI %s server; old CLI %s\n", pp.name.c_str(),
                        mtp ? " +MTP" : "", n, sy, th, se.c_str(), cli == srv ? "==" : "DIFFER (FAIL)", old == srv ? "==" : "differed"));
            }
        }
    }
    out(fmt("  CLI-SCHED: %u/%u CLI prefills differ from the server's forwards\n", cli_bad, cli_total));
    ok = ok && cli_bad == 0;
    out(fmt("  head logits bits: %u/%u verify / batched rows differ from the solo decode row\n", lh_bad, lh_total));
    ok = ok && lh_bad == 0;
    out(ok ? "seqtest: ok\n" : "seqtest: FAIL\n");
    return ok ? 0 : 1;
}

}  // namespace cli
