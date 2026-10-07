// Numerics modes: parsing, capability table, precise-mode KV decisions (see include/whirl/numerics.h).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/numerics.h"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace whirl::numerics {

namespace {

struct ItemInfo {
    const char* name;
    const char* desc;
};
constexpr std::array<ItemInfo, n_items> k_items = {{
    {"fp8", "MXFP4 dense prefill GEMMs with fp8 activations"},
    {"moefp8", "MoE expert prefill GEMMs with fp8 activations"},
    {"gdnwmma", "f16-WMMA DeltaNet prefill chunks"},
    {"h16", "f16 FFN / DeltaNet GEMM outputs"},
    {"kvq8", "int8 KV when f16 does not fit"},
    {"specsample", "race-coupled sampled MTP drafts (temperature > 0)"},
    {"kvq4", "4-bit KV"},
    {"relaxacc", "relaxed speculative acceptance"},
    {"headq", "low-bit output head in decode"},
    {"moeskip", "skip low-weight MoE experts"},
    {"a8", "int8 activations for K-quant prefill (W4A8)"},
    {"a4", "int4 activations for prefill (W4A4)"},
}};

constexpr const char* k_q8dec = "q8dec (int8 activations in the decode / verify GEMV, as llama.cpp's q8_1)";
constexpr const char* k_f16dec = "f16 decode / verify activations into the f16 GEMM, f32 accumulation (no int8)";

std::string lower(std::string_view s) {
    std::string o(s);
    for (char& c : o)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return o;
}

std::string trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return std::string(s);
}

std::uint32_t parseItems(std::string_view list) {
    std::uint32_t m = 0;
    std::size_t p = 0;
    while (p <= list.size()) {
        std::size_t e = list.find(',', p);
        if (e == std::string_view::npos) e = list.size();
        const std::string t = lower(trim(list.substr(p, e - p)));
        if (!t.empty()) {
            if (t == "all") m |= balance_items | fast_only_items;
            else if (t == "balance" || t == "balanced") m |= balance_items;
            else if (t == "q8dec") {
                // always on (listed for completeness)
            } else if (auto it = itemFromName(t)) m |= bit(*it);
            else {
                std::string known;
                for (const ItemInfo& ii : k_items) known += (known.empty() ? "" : ", ") + std::string(ii.name);
                throw std::invalid_argument(std::format("unknown numerics item '{}' (items: {})", t, known));
            }
        }
        p = e + 1;
    }
    return m;
}

void jsonStr(std::string& b, std::string_view s) {
    b += '"';
    for (char c : s) {
        if (c == '"' || c == '\\') {
            b += '\\';
            b += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            b += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
        } else {
            b += c;
        }
    }
    b += '"';
}

double gib(std::uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

const char* modeName(Mode m) {
    switch (m) {
        case Mode::precise: return "precise";
        case Mode::balance: return "balance";
        case Mode::fast: return "fast";
    }
    return "precise";
}

const char* itemName(Item it) { return k_items[static_cast<std::size_t>(it)].name; }
const char* itemDesc(Item it) { return k_items[static_cast<std::size_t>(it)].desc; }

std::optional<Item> itemFromName(std::string_view s) {
    for (std::size_t i = 0; i < n_items; ++i)
        if (s == k_items[i].name) return static_cast<Item>(i);
    return std::nullopt;
}

Request parseMode(std::string_view spec, std::string source) {
    const std::string s = trim(spec);
    std::string name = s, items;
    if (const std::size_t c = s.find_first_of(":="); c != std::string::npos) {
        name = s.substr(0, c);
        items = s.substr(c + 1);
    }
    name = lower(trim(name));
    Request r;
    r.source = std::move(source);
    if (name == "precise" || name.empty() || name == "default") r.mode = Mode::precise;
    else if (name == "balance" || name == "balanced") r.mode = Mode::balance;
    else if (name == "fast") r.mode = Mode::fast;
    else throw std::invalid_argument(std::format("unknown numerics mode '{}' (precise, balance or fast)", name));
    if (r.mode == Mode::precise) {
        if (!trim(items).empty()) throw std::invalid_argument("precise mode takes no item list (precise has no lossy items)");
        return r;
    }
    if (trim(items).empty()) {
        r.items = r.mode == Mode::balance ? balance_items : (balance_items | fast_only_items);
    } else {
        r.items = parseItems(items);
        r.custom = true;
        if (r.mode == Mode::balance && (r.items & fast_only_items) != 0)
            throw std::invalid_argument(
                "balance mode takes balance items only (fp8, moefp8, gdnwmma, h16, kvq8, specsample); use --fast=... for the others");
    }
    return r;
}

bool modeFlag(std::string_view arg, std::string* spec, bool* takes_next) {
    *takes_next = false;
    if (arg == "--precise") {
        *spec = "precise";
        return true;
    }
    for (std::string_view f : {std::string_view("--balance"), std::string_view("--balanced"), std::string_view("--fast")}) {
        if (arg == f) {
            *spec = std::string(f.substr(2));
            return true;
        }
        if (arg.size() > f.size() && arg.substr(0, f.size()) == f && arg[f.size()] == '=') {
            *spec = std::string(f.substr(2)) + ":" + std::string(arg.substr(f.size() + 1));
            return true;
        }
    }
    if (arg == "--mode") {
        *takes_next = true;
        return true;
    }
    if (arg.size() > 7 && arg.substr(0, 7) == "--mode=") {
        *spec = std::string(arg.substr(7));
        return true;
    }
    return false;
}

Request resolve(std::optional<std::string> cli_spec, std::optional<std::string> env_mode) {
    if (cli_spec) return parseMode(*cli_spec, "command line");
    if (env_mode && !trim(*env_mode).empty()) return parseMode(*env_mode, "WHIRL_MODE");
    return Request{};
}

Plan plan(const Request& r, const Target& t) {
    Plan p;
    p.mode = r.mode;
    p.q8dec = r.mode != Mode::precise;  // the loader confirms (setQ8dec: WHIRL_Q8DEC, kernels)
    for (std::size_t i = 0; i < n_items; ++i) {
        const Item it = static_cast<Item>(i);
        ItemState& s = p.items[i];
        s.item = it;
        s.requested = r.has(it);
        if (!s.requested) continue;
        std::string why;  // non-empty: skipped
        std::string note;
        switch (it) {
            case Item::fp8:
                if (!t.k_gemm8) why = t.gfx1151 ? "no fp8 WMMA on this GPU (gfx1151)" : "no fp8 GEMM kernels on this GPU";
                else if (!t.has_mxfp4) why = "no MXFP4 tensors in this model";
                break;
            case Item::moefp8:
                if (!t.moe) why = "dense model";
                else if (!t.moe_mxfp4) why = "experts are not MXFP4";
                else if (!t.k_gemm8_moe) why = t.gfx1151 ? "no fp8 WMMA on this GPU (gfx1151)" : "no fp8 expert GEMM kernels on this GPU";
                break;
            case Item::gdnwmma:
                if (!t.k_gdn_wmma) why = "no WMMA DeltaNet kernels on this GPU";
                else if (!t.has_mxfp4) why = "MXFP4 models only for now (WHIRL_Q4_RELAXED=1 forces it)";
                break;
            case Item::h16:
                if (!t.has_mxfp4) why = "MXFP4 models only for now (WHIRL_Q4_RELAXED=1 forces it)";
                else if (t.gfx1151) note = "f16 weights only on this GPU";
                break;
            case Item::kvq8:
                if (t.moe) why = "MoE models keep f16 KV";
                else if (t.k_kv_q8v && t.k_kv_q8h) note = "auto: q8v, then q8h, when f16 does not fit";
                else if (t.k_kv_q8v) note = "auto: q8v, then q8, when f16 does not fit";
                else note = "auto: q8 when f16 does not fit";
                break;
            case Item::specsample:
                if (!t.mtp) why = "no MTP head";
                else if (!t.k_spec_sample) why = "no draft sampling kernel";
                else note = "temperature > 0: race sampler (other tokens for a seed than precise), text independent of drafts";
                break;
            case Item::kvq4:
                if (!t.k_kv_q4) why = t.gfx1151 ? "no q4 KV kernels in this code object" : "no 4-bit KV kernels on this GPU (Radeon 8060S only)";
                else if (t.kv_explicit) why = "WHIRL_KV picks the KV format";
                else note = "q4: int4 + f16 scale / 32, Hadamard-rotated q/k";
                break;
            case Item::relaxacc:
                if (t.moe) {
                    // FAST-2 (8060S): MoE runs one MTP draft per cycle there; relaxed acceptance moved
                    // Ornith 1.75 -> 1.85 tok/cycle with no decode gain (short +2%, 32k -12%, n=1)
                    why = "MoE: one MTP draft per cycle, no measured gain (WHIRL_RELAX=1 forces it)";
                    break;
                }
                note = "drafts accepted when close to the target (greedy: top-k + p ratio; sampling: typical acceptance)";
                break;
            default:
                why = "not yet implemented (balance behaviour)";
                break;
        }
        s.enabled = why.empty();
        s.note = s.enabled ? note : why;
    }
    return p;
}

void setQ8dec(Plan& p, bool q8, const std::string& env) {
    p.q8dec = q8;
    const bool def = p.mode != Mode::precise;
    if (env.empty() || q8 == def) return;
    if (q8) {
        p.overrides.push_back(std::format("WHIRL_Q8DEC={}: q8dec on (lossy, user-requested)", env));
        p.lossy_override = true;
    } else {
        p.overrides.push_back(std::format("WHIRL_Q8DEC={}: q8dec off (f16 decode activations)", env));
    }
}

void applyOverride(Plan& p, Item it, const char* var, const std::string& value, bool forced, bool applicable) {
    ItemState& s = p.items[static_cast<std::size_t>(it)];
    const bool eff = forced && applicable;
    if (eff == s.enabled) return;
    if (eff) {
        s.enabled = true;
        s.note = std::format("{}={}, user-requested", var, value);
        p.overrides.push_back(std::format("WHIRL_{}={}: {} on (lossy, user-requested)", var, value, itemName(it)));
        if (p.mode == Mode::precise || !s.requested) p.lossy_override = true;
    } else {
        s.enabled = false;
        s.note = std::format("off by WHIRL_{}={}", var, value);
        p.overrides.push_back(std::format("WHIRL_{}={}: {} off", var, value, itemName(it)));
    }
}

std::string modeLabel(const Plan& p) {
    std::string l = modeName(p.mode);
    if (p.lossy_override) l += "+overrides";
    return l;
}

std::string logLine(const Plan& p) {
    std::string en, sk;
    for (const ItemState& s : p.items) {
        if (s.enabled) en += std::format("{}{}{}", en.empty() ? "" : ", ", itemName(s.item), s.note.empty() ? "" : " (" + s.note + ")");
        else if (s.requested) sk += std::format("{}{} ({})", sk.empty() ? "" : ", ", itemName(s.item), s.note);
    }
    std::string l = "numerics: " + modeLabel(p);
    if (p.mode == Mode::precise && en.empty())
        l += std::string(" - f16/f32 prefill activations, ") + (p.q8dec ? "" : "f16 decode / verify activations, ") + "f32 DeltaNet, f16 KV";
    else l += " - enabled: " + (en.empty() ? std::string("none") : en);
    if (p.mode != Mode::precise || !sk.empty()) l += "; skipped: " + (sk.empty() ? std::string("none") : sk);
    if (p.mode == Mode::fast && !p.on(Item::kvq4) && !p.on(Item::relaxacc))
        l += "; no fast item applies here: fast runs as balance";
    for (const std::string& o : p.overrides) l += "; " + o;
    l += std::string("; decode: ") + (p.q8dec ? k_q8dec : k_f16dec);
    return l;
}

std::string propsJson(const Plan& p, std::string_view kv) {
    std::string b = "{\"mode\":";
    jsonStr(b, modeName(p.mode));
    b += ",\"label\":";
    jsonStr(b, modeLabel(p));
    b += ",\"enabled\":[";
    bool first = true;
    for (const ItemState& s : p.items) {
        if (!s.enabled) continue;
        if (!first) b += ',';
        first = false;
        jsonStr(b, itemName(s.item));
    }
    b += "],\"skipped\":[";
    first = true;
    for (const ItemState& s : p.items) {
        if (s.enabled || !s.requested) continue;
        if (!first) b += ',';
        first = false;
        b += "{\"item\":";
        jsonStr(b, itemName(s.item));
        b += ",\"reason\":";
        jsonStr(b, s.note);
        b += '}';
    }
    b += "],\"overrides\":[";
    for (std::size_t i = 0; i < p.overrides.size(); ++i) {
        if (i) b += ',';
        jsonStr(b, p.overrides[i]);
    }
    b += "],\"kv\":";
    jsonStr(b, kv);
    b += ",\"decode\":";
    jsonStr(b, p.q8dec ? "q8dec" : "f16");
    b += ",\"always_on\":[";
    if (p.q8dec) b += "{\"item\":\"q8dec\",\"note\":\"int8 activations in the decode / verify GEMV (llama.cpp q8_1 class)\"}";
    b += "]}";
    return b;
}

CtxFit cliCtxFit(std::uint32_t ctx, bool ctx_explicit, std::uint32_t min_ctx, std::uint64_t free, std::uint64_t later,
                 std::uint64_t f16_per_token, std::uint32_t page) {
    CtxFit r;
    r.ctx = ctx;
    const std::uint64_t toks = (static_cast<std::uint64_t>(ctx) + page - 1) / page * page;
    if (f16_per_token == 0 || toks * f16_per_token + later <= free) return r;
    const std::uint64_t room = free > later ? free - later : 0;
    const std::uint64_t fit = room / f16_per_token / page * page;
    const std::string what = std::format("f16 KV for {} tokens needs {:.2f} GiB, about {:.2f} GiB of GPU memory is left for it", ctx,
                                         gib(toks * f16_per_token), gib(room));
    if (!ctx_explicit && fit >= std::max<std::uint64_t>(min_ctx, page)) {
        r.ctx = static_cast<std::uint32_t>(std::min<std::uint64_t>(fit, ctx));
        r.shrunk = true;
        r.msg = std::format("precise mode keeps the KV cache in f16: {}; context lowered to {} tokens (--balance allows int8 KV; "
                            "--ctx N sets the context)",
                            what, r.ctx);
        return r;
    }
    r.refuse = true;
    r.msg = std::format("precise mode keeps the KV cache in f16: {} (room for about {} tokens). Use a shorter --ctx, or --balance "
                        "(int8 KV when f16 does not fit)",
                        what, fit);
    return r;
}

PoolFit serverPoolFit(std::uint64_t avail, std::uint64_t f16_per_token, std::optional<std::uint32_t> pool_req, bool pool_explicit,
                      std::uint32_t slot_ctx, bool slot_explicit, std::uint32_t page, std::uint32_t pool_cap) {
    PoolFit r;
    r.slot_ctx = slot_ctx;
    const std::uint64_t by_mem = f16_per_token ? std::min<std::uint64_t>(avail / f16_per_token, pool_cap) / page * page : 0;
    bool from_mem = !pool_req;
    if (pool_req) {
        if (static_cast<std::uint64_t>(*pool_req) * f16_per_token <= avail) {
            r.pool = *pool_req;
        } else if (pool_explicit) {
            r.refuse = true;
            r.msg = std::format("precise mode keeps the KV cache in f16: a pool of {} tokens (--ctx) needs {:.2f} GiB, about {:.2f} GiB of "
                                "GPU memory is usable (room for about {} tokens). Use a smaller --ctx, or --balance (int8 KV when f16 "
                                "does not fit)",
                                *pool_req, gib(static_cast<std::uint64_t>(*pool_req) * f16_per_token), gib(avail), by_mem);
            return r;
        } else {
            r.pool = static_cast<std::uint32_t>(by_mem);
            r.shrunk_pool = true;
            from_mem = true;
        }
    } else {
        r.pool = static_cast<std::uint32_t>(by_mem);
    }
    if (r.pool < 4096) {
        r.refuse = true;
        r.msg = std::format("not enough GPU memory left for an f16 KV cache: a pool of {} tokens (at least 4096 needed).\n"
                            "  Use a smaller model or quantization, --balance (int8 KV), a shorter context, or close other programs "
                            "that use the GPU.",
                            r.pool);
        return r;
    }
    if (from_mem && r.pool < r.slot_ctx) {
        if (slot_explicit) {
            r.refuse = true;
            r.msg = std::format("precise mode keeps the KV cache in f16: --ctx-per-slot {} does not fit, the f16 pool holds about {} "
                                "tokens. Use a smaller --ctx-per-slot, or --balance (int8 KV when f16 does not fit)",
                                r.slot_ctx, r.pool);
            return r;
        }
        r.slot_ctx = r.pool;
        r.shrunk_ctx = true;
    }
    if (r.shrunk_pool || r.shrunk_ctx)
        r.msg = std::format("precise mode keeps the KV cache in f16: pool {} tokens{}{} (--balance allows int8 KV: a longer pool; "
                            "--ctx / --ctx-per-slot choose explicitly)",
                            r.pool, r.shrunk_pool ? std::format(" (wanted {})", pool_req.value_or(0)) : std::string(),
                            r.shrunk_ctx ? std::format(", context per request lowered to {} tokens (wanted {})", r.slot_ctx, slot_ctx)
                                         : std::string());
    return r;
}

}  // namespace whirl::numerics
