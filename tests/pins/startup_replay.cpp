// startup_replay (arch024-S0a; PLAN v3 N3, P-size): recompute the server's startup sizing decisions from the
// "startup inputs:" line (server_main.cpp) with the same pure functions the server calls. CPU only.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Calls only pure functions: startfit::fitParallel / fitFixed / slotsThatFit / pinnedRoomMb /
// pinnedWhy / ramTierFit / ramTierText, tier::ramTierSize, numerics::serverPoolFit. It never re-implements the glue
// in server_main.cpp (the order of the calls, the "off" conditions, the line formats); what only the glue decides is
// marked "glue" in tools/gates/startup_lines.py and covered by the cross-run compare and the B0 tests. Builds on the B0
// base (06a1540) too: the CPU table (tests/pins/startup_table.txt -> startup_table.out) is recorded there.
//
//   whirl-startup-replay --line "<text after 'startup inputs:'>"   one input line -> decision lines
//   whirl-startup-replay --log SERVER.log                         the last "startup inputs:" line of a server log
//   whirl-startup-replay --table FILE                             "<name>\t<inputs>" per line (fixed CPU table)
// Output: one line per decision group, "<group> key=value ...", values percent-encoded like the input line.

#include "tier/ram_size.h"
#include "whirl/numerics.h"
#include "whirl/startup_fit.h"
#include "whirl/vram_limit.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace whirl;

namespace {

std::string dec(std::string_view s) {
    if (s == "%") return {};
    std::string o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const std::string h(s.substr(i + 1, 2));
            o += static_cast<char>(std::stoi(h, nullptr, 16));
            i += 2;
        } else {
            o += s[i];
        }
    }
    return o;
}
std::string enc(std::string_view s) {
    std::string o;
    for (const char c : s) {
        if (c == ' ') o += "%20";
        else if (c == '%') o += "%25";
        else if (c == '=') o += "%3D";
        else if (c == '\n') o += "%0A";
        else o += c;
    }
    return o.empty() ? std::string("%") : o;
}

struct In {
    std::map<std::string, std::string> kv;
    bool has(const std::string& k) const { return kv.count(k) != 0; }
    std::string str(const std::string& k) const {
        auto it = kv.find(k);
        if (it == kv.end()) throw std::runtime_error("missing input " + k);
        return dec(it->second);
    }
    std::uint64_t u(const std::string& k) const { return std::stoull(str(k)); }
    bool b(const std::string& k) const { return u(k) != 0; }
    std::optional<std::uint64_t> opt(const std::string& k) const {
        const std::string v = str(k);
        if (v == "-") return std::nullopt;
        return std::stoull(v);
    }
};

In parse(std::string_view line) {
    In in;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        const std::size_t j = line.find(' ', i);
        const std::string_view tok = line.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i);
        if (!tok.empty()) {
            const auto eq = tok.find('=');
            if (eq == std::string_view::npos) throw std::runtime_error("bad token " + std::string(tok));
            const std::string key(tok.substr(0, eq));
            if (in.kv.count(key)) throw std::runtime_error("duplicate input " + key);
            in.kv[key] = std::string(tok.substr(eq + 1));
        }
        if (j == std::string_view::npos) break;
        i = j + 1;
    }
    return in;
}

vram::FitInput fitOf(const In& in, const std::string& p) {
    vram::FitInput fi;
    fi.avail = in.u(p + ".avail");
    fi.weights = in.u(p + ".weights");
    fi.fixed = in.u(p + ".fixed");
    fi.buf_per_row = in.u(p + ".buf_per_row");
    fi.buf_fixed = in.u(p + ".buf_fixed");
    fi.ckpt_bytes = in.u(p + ".ckpt_bytes");
    fi.parallel = static_cast<std::uint32_t>(in.u(p + ".parallel"));
    fi.kv_per_token = in.u(p + ".kv_per_token");
    fi.pool_min_tokens = static_cast<std::uint32_t>(in.u(p + ".pool_min_tokens"));
    fi.pool_hard_min = static_cast<std::uint32_t>(in.u(p + ".pool_hard_min"));
    fi.batch = static_cast<std::uint32_t>(in.u(p + ".batch"));
    fi.n_ck = static_cast<std::uint32_t>(in.u(p + ".n_ck"));
    fi.n_spe = static_cast<std::uint32_t>(in.u(p + ".n_spe"));
    fi.batch_fixed = in.b(p + ".batch_fixed");
    fi.ck_fixed = in.b(p + ".ck_fixed");
    return fi;
}

std::string plan(const char* p, const vram::FitPlan& pl) {
    return std::string(" ") + p + ".batch=" + std::to_string(pl.batch) + " " + p + ".n_ck=" + std::to_string(pl.n_ck) + " " + p +
           ".n_spe=" + std::to_string(pl.n_spe) + " " + p + ".pool_tokens=" + std::to_string(pl.pool_tokens) + " " + p +
           ".fits=" + (pl.fits ? "1" : "0");
}

std::vector<std::string> replay(const In& in) {
    std::vector<std::string> out;
    if (in.has("fitp.avail")) {
        const vram::FitInput fi = fitOf(in, "fitp");
        const startfit::ParallelFit pf = startfit::fitParallel(fi, in.u("fitp.per_slot"), in.b("fitp.given"));
        out.push_back("fitp parallel=" + std::to_string(pf.parallel) + " pool_requested=" + std::to_string(pf.pool_requested) +
                      plan("plan", pf.plan));
    }
    if (in.has("fitf.avail")) {
        const vram::FitInput fi = fitOf(in, "fitf");
        vram::FitPlan pl;
        pl.batch = static_cast<std::uint32_t>(in.u("fitf.plan.batch"));
        pl.n_ck = static_cast<std::uint32_t>(in.u("fitf.plan.n_ck"));
        pl.n_spe = static_cast<std::uint32_t>(in.u("fitf.plan.n_spe"));
        pl.pool_tokens = in.u("fitf.plan.pool_tokens");
        const startfit::FixedFit ff = startfit::fitFixed(fi, pl);
        out.push_back(std::string("fitf changed=") + (ff.changed ? "1" : "0") + " batch_changed=" + (ff.batch_changed ? "1" : "0") +
                      " ck_changed=" + (ff.ck_changed ? "1" : "0") + plan("plan", ff.plan));
    }
    if (in.has("slots.parallel")) {
        const std::uint32_t r = startfit::slotsThatFit(static_cast<std::uint32_t>(in.u("slots.parallel")), in.u("slots.avail"),
                                                        in.u("slots.per_slot"), in.u("slots.shared"), in.u("slots.per_token"),
                                                        in.u("slots.target"));
        out.push_back("slots result=" + std::to_string(r));
    }
    if (in.has("ram.total_phys")) {
        tier::RamTierInput rin;
        rin.explicit_mb = in.opt("ram.explicit_mb");
        const std::string src = in.str("ram.explicit_src");
        rin.explicit_src = src.c_str();
        rin.total_phys = in.u("ram.total_phys");
        rin.avail_phys = in.u("ram.avail_phys");
        rin.floor_mb = in.u("ram.floor_mb");
        rin.entry_mb = in.u("ram.entry_mb");
        rin.integrated = in.b("ram.integrated");
        const tier::RamTierSize rsz = tier::ramTierSize(rin);
        tier::RamTierInput a = rin;  // the automatic size (glue: server_main computes it the same way)
        a.explicit_mb.reset();
        const std::uint64_t auto_mb = tier::ramTierSize(a).mb;
        std::uint64_t room = startfit::pinned_room_unknown;
        std::string why;
        std::size_t n_gpus = 0;
        if (in.has("pin.measured") && in.b("pin.measured")) {
            startfit::PinnedState pst;
            pst.total_phys = in.u("pin.total_phys");
            pst.pinned_max = in.u("pin.pinned_max");
            pst.later = in.u("pin.later");
            n_gpus = in.u("pin.gpus");
            for (std::size_t g = 0; g < n_gpus; ++g) {
                const std::string k = "pin.g" + std::to_string(g);
                pst.gpus.push_back({in.u(k + ".nonlocal_budget"), in.u(k + ".own_nonlocal"), in.u(k + ".shared_system"),
                                    in.u(k + ".adapter_shared")});
            }
            room = startfit::pinnedRoomMb(pst);
            why = startfit::pinnedWhy(room == startfit::pinned_room_unknown ? 0 : room, pst.gpus.size());
        }
        std::optional<startfit::RamFit> rf =
            startfit::ramTierFit(rsz.mb, rin.avail_phys, auto_mb, rin.explicit_mb.has_value(), room, rin.entry_mb);
        if (in.u("ram.n_ck") == 0 || in.b("ram.no_pc")) rf.reset();  // glue (server_main): no host tiers
        const startfit::RamTierText t =
            startfit::ramTierText(rf, rsz.mb, rin.explicit_mb.has_value(), rin.explicit_src, rsz.reason, why, rin.entry_mb, rin.avail_phys >> 20);
        std::string l = "ram size_mb=" + std::to_string(rsz.mb) + " avail_limited=" + (rsz.avail_limited ? "1" : "0") +
                        " auto_mb=" + std::to_string(auto_mb) + " room=" +
                        (room == startfit::pinned_room_unknown ? std::string("unknown") : std::to_string(room) + "%20MiB");
        if (rf)
            l += " fit=1 fit.mb=" + std::to_string(rf->mb) + " fit.half=" + (rf->half ? "1" : "0") + " fit.safe=" + (rf->safe ? "1" : "0") +
                 " fit.pinned=" + (rf->pinned ? "1" : "0") + " fit.below_entry=" + (rf->below_entry ? "1" : "0");
        else
            l += " fit=0";
        l += " final_mb=" + std::to_string(rf ? rf->mb : rsz.mb);
        l += " pinned_why=" + enc(t.pinned_why) + " adjusted_why=" + enc(t.adjusted_why) + " off_reason=" + enc(t.off_reason) +
             " size_reason=" + enc(t.size_reason);
        out.push_back(l);
    }
    if (in.has("pool.avail")) {
        numerics::KvFitLabel lb;
        lb.mode = in.str("pool.mode");
        lb.fmt = in.str("pool.fmt");
        lb.alt = in.str("pool.alt");
        const auto req = in.opt("pool.req");
        const numerics::PoolFit pf = numerics::serverPoolFit(
            in.u("pool.avail"), in.u("pool.per_token"), req ? std::optional<std::uint32_t>(static_cast<std::uint32_t>(*req)) : std::nullopt,
            in.b("pool.pool_explicit"), static_cast<std::uint32_t>(in.u("pool.slot_ctx")), in.b("pool.slot_explicit"),
            static_cast<std::uint32_t>(in.u("pool.page")), static_cast<std::uint32_t>(in.u("pool.cap")), lb);
        out.push_back("pool pool=" + std::to_string(pf.pool) + " slot_ctx=" + std::to_string(pf.slot_ctx) + " shrunk_pool=" +
                      (pf.shrunk_pool ? "1" : "0") + " shrunk_ctx=" + (pf.shrunk_ctx ? "1" : "0") + " refuse=" + (pf.refuse ? "1" : "0") +
                      " msg=" + enc(pf.msg));
    }
    return out;
}

int run(std::string_view line, const char* name) {
    try {
        for (const std::string& l : replay(parse(line))) std::printf("%s%s%s\n", name ? name : "", name ? " " : "", l.c_str());
        return 0;
    } catch (const std::exception& ex) {
        std::printf("%s%serror %s\n", name ? name : "", name ? " " : "", ex.what());
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: whirl-startup-replay --line TEXT | --log SERVER.log | --table FILE\n");
        return 2;
    }
    const std::string mode = argv[1], arg = argv[2];
    if (mode == "--line") return run(arg, nullptr);
    std::ifstream f(arg, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", arg.c_str());
        return 2;
    }
    std::string line, last;
    int bad = 0, n = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (mode == "--log") {
            const auto p = line.find("startup inputs:");
            if (p != std::string::npos) last = line.substr(p + 15);
        } else if (mode == "--table") {
            if (line.empty() || line[0] == '#') continue;
            const auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            const std::string name = line.substr(0, tab);
            ++n;
            bad += run(std::string_view(line).substr(tab + 1), name.c_str());
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
    }
    if (mode == "--log") {
        if (last.empty()) {
            std::printf("error no 'startup inputs:' line in %s\n", arg.c_str());
            return 1;
        }
        return run(last, nullptr);
    }
    std::printf("table %d rows, %d errors\n", n, bad);
    return bad == 0 && n > 0 ? 0 : 1;
}
