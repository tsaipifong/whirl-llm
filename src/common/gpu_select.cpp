// GPU selection (see include/whirl/gpu_select.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. The matching kinds follow pickDevice before 0.2.3 (src/model/device.cpp: index,
// the r9700 / 8060s aliases, architecture prefix, name part) with two changes: aliases match the card's
// NAME (r9700 used to mean "gfx1201" and so also took an RX 9070 XT), and a name that matches several
// GPUs is an error instead of the first match. Added: llama.cpp's "ROCmN" names, "uuid:", the decimal
// PCI location Windows shows, and the comma list of llama.cpp's -dev <dev1,dev2,..>
// (refsrc llama.cpp common/arg.cpp parse_device_list).

#include "whirl/gpu_select.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <format>

namespace whirl::gpusel {

namespace {

std::string lower(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

std::string trim(std::string_view s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return std::string(s.substr(b, e - b));
}

bool allDigits(std::string_view s) {
    if (s.empty() || s.size() > 6) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}

// the memory a model can use (an integrated GPU: its dedicated carve-out), as the default rule counts it
std::string gib(const GpuRecord& g) {
    return std::format("{:.1f} GiB", static_cast<double>(g.fit_mib ? g.fit_mib : g.total_mib) / 1024.0);
}

std::string oneLine(const GpuRecord& g) {
    return std::format("    device {}: {} ({}), pci:{}, {}{}{}\n", g.index, g.name, g.arch, pciId({}, g), gib(g),
                       g.uuid.empty() ? "" : ", uuid " + g.uuid, g.supported ? "" : " [no GPU kernels in this build]");
}

const char* k_hint =
    "  Choose one with --device (or -dev / WHIRL_DEVICE): pci:BUS:DEVICE, an index, r9700, 8060s, or a part of the name.\n"
    "  To tell identical cards apart see Task Manager > Performance > GPU > Physical location (PCI bus B, device D = pci:B:D).";

[[noreturn]] void fail(const std::vector<GpuRecord>& gpus, const std::string& what) {
    throw SelectError(what + "\n  GPUs in this computer:\n" +
                      (gpus.empty() ? std::string("    (the AMD graphics driver reports no GPU)\n") : listText(gpus)) + k_hint);
}

[[noreturn]] void ambiguous(const std::vector<GpuRecord>& gpus, const std::string& item, const std::vector<int>& hits) {
    std::string s = std::format("--device {} matches more than one GPU:\n", item);
    for (int h : hits) s += oneLine(gpus[h]);
    const GpuRecord& f = gpus[hits[0]];
    s += std::format("  Say which one: --device pci:{} (stays the same while the card stays connected the same way) or --device {}\n"
                     "  (the index; it can change when GPUs are added or removed){}.\n"
                     "  See Task Manager > Performance > GPU > Physical location (PCI bus B, device D = pci:B:D).",
                     pciId({}, f), f.index, f.uuid.empty() ? "" : ", or --device uuid:" + f.uuid);
    throw SelectError(s);
}

// "71", "71:0", "71:0.0", "1:71:0.0" (decimal) -> domain/bus/device/function; device / function -1 = any
bool parsePci(std::string_view t, int& dom, int& bus, int& dev, int& fn) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : t) {
        if (c == ':' || c == '.') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    parts.push_back(cur);
    for (const std::string& p : parts)
        if (!allDigits(p)) return false;
    const auto dots = std::count(t.begin(), t.end(), '.');
    if (dots > 1) return false;
    dom = 0;
    dev = fn = -1;
    std::vector<int> v;
    for (const std::string& p : parts) v.push_back(std::atoi(p.c_str()));
    if (dots == 1) {  // ...dev.fn at the end
        if (v.size() == 3) { bus = v[0]; dev = v[1]; fn = v[2]; return true; }
        if (v.size() == 4) { dom = v[0]; bus = v[1]; dev = v[2]; fn = v[3]; return true; }
        return false;
    }
    if (v.size() == 2) { bus = v[0]; dev = v[1]; return true; }
    if (v.size() == 3) { dom = v[0]; bus = v[1]; dev = v[2]; return true; }
    return false;
}

std::vector<int> matchOne(const std::vector<GpuRecord>& gpus, const std::string& item) {
    const int n = static_cast<int>(gpus.size());
    std::string key = lower(item);
    // llama.cpp names its HIP devices ROCm0, ROCm1, ...; "HIP0" also accepted
    for (const char* pre : {"rocm", "hip"}) {
        const std::string p(pre);
        if (key.size() > p.size() && key.rfind(p, 0) == 0 && allDigits(key.substr(p.size()))) {
            key = key.substr(p.size());
            break;
        }
    }
    if (allDigits(key)) {
        const int idx = std::atoi(key.c_str());
        for (int i = 0; i < n; ++i)
            if (gpus[i].index == idx) return {i};
        // not an index here: a number can still be part of a name ("9070"), but only for a plain number
        bool named = false;
        for (int i = 0; i < n && key == lower(item) && key.size() >= 3; ++i) named = named || lower(gpus[i].name).find(key) != std::string::npos;
        if (!named) fail(gpus, std::format("--device {}: there is no GPU with that index", item));
    }
    std::vector<int> hits;
    if (key.rfind("pci:", 0) == 0) {
        int dom = 0, bus = 0, dev = -1, fn = -1;
        if (!parsePci(key.substr(4), dom, bus, dev, fn)) {
            const std::string b = key.substr(4);
            fail(gpus, allDigits(b) ? std::format("--device {}: give the bus and the device number: pci:{}:0 (Task Manager > Performance > "
                                                  "GPU > Physical location: PCI bus {}, device 0)", item, b, b)
                                    : std::format("--device {}: write the PCI location as pci:BUS:DEVICE in decimal, e.g. pci:71:0", item));
        }
        for (int i = 0; i < n; ++i) {
            const GpuRecord& g = gpus[i];
            if (g.pci_domain == dom && g.pci_bus == bus && (dev < 0 || g.pci_device == dev) && (fn < 0 || g.pci_function == fn))
                hits.push_back(i);
        }
    } else if (key.rfind("uuid:", 0) == 0) {
        const std::string want = key.substr(5);
        if (!want.empty())
            for (int i = 0; i < n; ++i)
                if (!gpus[i].uuid.empty() && lower(gpus[i].uuid).rfind(want, 0) == 0) hits.push_back(i);
    } else if (key == "igpu") {
        for (int i = 0; i < n; ++i)
            if (gpus[i].integrated) hits.push_back(i);
    } else {
        // strix = gfx1151; gfxNNNN = an architecture prefix; anything else (r9700, 8060s, 9070, a full name)
        // is a part of the marketing name
        const bool arch_key = key == "strix" || key.rfind("gfx", 0) == 0;
        const std::string a = key == "strix" ? std::string("gfx1151") : key;
        for (int i = 0; i < n; ++i) {
            const bool m = arch_key ? lower(gpus[i].arch).rfind(a, 0) == 0 : lower(gpus[i].name).find(key) != std::string::npos;
            if (m) hits.push_back(i);
        }
    }
    if (hits.empty()) fail(gpus, std::format("--device {}: no GPU matches that name", item));
    if (hits.size() > 1) ambiguous(gpus, item, hits);
    return hits;
}

}  // namespace

std::vector<std::string> parseDeviceList(std::string_view spec) {
    std::vector<std::string> out;
    if (trim(spec).empty()) return out;
    std::size_t start = 0;
    for (;;) {
        const std::size_t comma = spec.find(',', start);
        const std::string item = trim(spec.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start));
        if (item.empty())
            throw SelectError(std::format("--device '{}': an empty entry in the list (give names separated by commas)", spec));
        out.push_back(item);
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return out;
}

namespace {

std::uint64_t memOf(const GpuRecord& g) { return g.fit_mib ? g.fit_mib : g.total_mib; }

std::string gibOf(std::uint64_t mib) { return std::format("{:.1f} GiB", static_cast<double>(mib) / 1024.0); }

}  // namespace

AutoChoice defaultGpu(const std::vector<GpuRecord>& gpus) {
    // owner 2026-10-23: the most GPU memory, discrete or integrated alike; a tie -> the lowest index
    int best = -1;
    for (int i = 0; i < static_cast<int>(gpus.size()); ++i)
        if (gpus[i].supported && (best < 0 || memOf(gpus[i]) > memOf(gpus[best]))) best = i;
    AutoChoice c;
    if (best < 0) return c;
    c.index = gpus[best].index;
    c.reason = std::format("most GPU memory ({})", gibOf(memOf(gpus[best])));
    return c;
}

int autoPick(const std::vector<GpuRecord>& gpus) { return defaultGpu(gpus).index; }

std::vector<int> selectGpus(const std::vector<GpuRecord>& gpus, std::string_view spec) {
    const std::vector<std::string> items = parseDeviceList(spec);
    if (items.empty()) {
        const int d = autoPick(gpus);
        if (d < 0) fail(gpus, "no GPU this build has kernels for was found (WHIRL runs on the R9700 and the Radeon 8060S)");
        for (int i = 0; i < static_cast<int>(gpus.size()); ++i)
            if (gpus[i].index == d) return {i};
        return {d};
    }
    if (items.size() > 1)
        throw SelectError(std::format(
            "--device {}: only one GPU can be given with --device",
            spec));
    return matchOne(gpus, items[0]);
}

std::string pciText(const GpuRecord& r) {
    return r.pci_domain ? std::format("{}:{}:{}.{}", r.pci_domain, r.pci_bus, r.pci_device, r.pci_function)
                        : std::format("{}:{}.{}", r.pci_bus, r.pci_device, r.pci_function);
}

std::string pciLongText(const GpuRecord& r) {
    return std::format("{}PCI bus {}, device {}, function {}", r.pci_domain ? std::format("PCI domain {}, ", r.pci_domain) : "",
                       r.pci_bus, r.pci_device, r.pci_function);
}

std::string pciId(const std::vector<GpuRecord>&, const GpuRecord& r) {
    return r.pci_domain ? std::format("{}:{}:{}", r.pci_domain, r.pci_bus, r.pci_device) : std::format("{}:{}", r.pci_bus, r.pci_device);
}

std::string pciIdLong(const std::vector<GpuRecord>&, const GpuRecord& r) {
    return std::format("{}PCI bus {}, device {}", r.pci_domain ? std::format("PCI domain {}, ", r.pci_domain) : "", r.pci_bus, r.pci_device);
}

std::string lockName(const GpuRecord& r) {
    if (!r.uuid.empty()) return "Local\\whirl-gpu-uuid-" + lower(r.uuid);
    return std::format("Local\\whirl-gpu-pci-{}-{}-{}{}", r.pci_bus, r.pci_device, r.pci_function,
                       r.pci_domain ? std::format("-d{}", r.pci_domain) : std::string());
}

std::wstring lockMutexName(const GpuRecord& r) {
    const std::string n = lockName(r);
    return std::wstring(n.begin(), n.end());
}

std::string listText(const std::vector<GpuRecord>& gpus) {
    std::string s;
    for (const GpuRecord& g : gpus) s += oneLine(g);
    return s;
}

std::string listingText(const std::vector<GpuRecord>& gpus, int def) {
    std::string s;
    bool any_uuid = false;
    for (const GpuRecord& g : gpus) {
        any_uuid = any_uuid || !g.uuid.empty();
        std::string tags = g.supported ? "  [supported]" : "  [no GPU kernels for this architecture in this build]";
        if (g.integrated) tags += " [integrated: memory shared with the CPU]";
        if (g.index == def) tags += " [default]";
        s += std::format("  device {}: {} ({}), {}, {} CUs, pci:{} ({}){}{}\n", g.index, g.name, g.arch,
                         g.mem_text.empty() ? gib(g) : g.mem_text, g.compute_units, pciId(gpus, g), pciIdLong(gpus, g),
                         g.uuid.empty() ? "" : ", uuid " + g.uuid, tags);
    }
    s += std::format(
        "  The index is the GPU runtime's order: it can change when a GPU is added or removed, the driver changes, or\n"
        "  HIP_VISIBLE_DEVICES is set. For a fixed choice use --device pci:BUS:DEVICE{} (the numbers Task Manager >\n"
        "  Performance > GPU > Physical location shows; an external USB4 / Thunderbolt GPU gets its bus number when it is\n"
        "  connected, so check again after replugging it or changing the port or the chain order).\n",
        any_uuid ? " or --device uuid:..." : "");
    return s;
}

std::string visibleDevicesValue(const std::vector<int>& ordinals) {
    std::string s;
    for (int o : ordinals) {
        if (!s.empty()) s += ',';
        s += std::to_string(o);
    }
    return s;
}

std::string userVisibilityVar(const std::function<const char*(const char*)>& getenv_fn) {
    for (const char* k : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"})
        if (getenv_fn(k) != nullptr) return k;
    return {};
}

std::string noGpuText(const std::function<const char*(const char*)>& getenv_fn) {
    // arch025-c2fu (PRIOR_ART progress/arch025-c2fu.PRIOR_ART.md): an EMPTY HIP_VISIBLE_DEVICES hides every GPU from the
    // runtime (measured: whirl --list-devices lists none; CUDA documents the same for CUDA_VISIBLE_DEVICES=""), and a
    // PowerShell restore with SetEnvironmentVariable(k, $null) leaves exactly that behind on pwsh 7.5+. llama.cpp prints
    // only the runtime's "no ROCm-capable device"; WHIRL names the variable, since it reads the same list anyway.
    const std::string var = userVisibilityVar(getenv_fn);
    if (var.empty()) return "the AMD graphics driver reports no GPU";
    const char* raw = getenv_fn(var.c_str());
    const std::string raw_s = raw ? raw : "";  // copied once: /analyze C6387 flagged std::string(raw) below (int7 cpu_analyze)
    const std::string val = trim(raw_s);
    const std::string why = val.empty()
        ? std::format("{} is set to an empty value, which hides every GPU from the GPU runtime", var)
        : std::format("{}={} is set and may hide the GPU (it lists the GPUs to show, by index)", var, raw_s);
    std::string text = why + std::format("; remove the variable and run the command again (PowerShell: Remove-Item Env:{0}; "
                                         "Command Prompt: set {0}=; or delete it under System Properties > Environment Variables)",
                                         var);
    // arch025-X8bfix (c2fu REVIEW C2, progress/arch025-X8bfix.PRIOR_ART.md): a non-empty value can be a valid index
    // (HIP_VISIBLE_DEVICES=0 from a ROCm guide) while the driver is missing or broken; the runtime shows zero GPUs either
    // way and cannot tell which. So the driver stays on the list, after the variable (the cheaper check). The empty value
    // alone hides every GPU (measured, arch025-c2ctx), so that text keeps no driver hint.
    if (!val.empty())
        text += "; if no GPU shows after that, a driver problem is also possible: install or update AMD Software: Adrenalin "
                "Edition (https://www.amd.com/en/support) and restart Windows";
    return text;
}

bool sameGpu(const GpuRecord& a, const GpuRecord& b) {
    if (a.pci_domain != b.pci_domain || a.pci_bus != b.pci_bus || a.pci_device != b.pci_device || a.pci_function != b.pci_function)
        return false;
    if (!a.uuid.empty() && !b.uuid.empty() && lower(a.uuid) != lower(b.uuid)) return false;
    if (!a.luid.empty() && !b.luid.empty() && a.luid != b.luid) return false;
    return a.name == b.name && a.arch == b.arch;
}

std::string formatRecord(const GpuRecord& r) {
    std::string name = r.name;
    for (char& c : name)
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return std::format("whirl-gpu\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", r.index, r.pci_domain, r.pci_bus,
                       r.pci_device, r.pci_function, r.total_mib, r.compute_units, r.fit_mib, r.uuid.empty() ? "-" : r.uuid,
                       r.luid.empty() ? "-" : r.luid,
                       r.integrated ? 1 : 0, r.supported ? 1 : 0, r.arch, name);
}

std::vector<GpuRecord> parseRecords(std::string_view text) {
    std::vector<GpuRecord> out;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t nl = text.find('\n', pos);
        if (nl == std::string_view::npos) nl = text.size();
        std::string_view line = text.substr(pos, nl - pos);
        pos = nl + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.rfind("whirl-gpu\t", 0) != 0) continue;
        std::vector<std::string> f;
        std::size_t s = 0;
        for (;;) {
            const std::size_t t = line.find('\t', s);
            f.emplace_back(line.substr(s, t == std::string_view::npos ? std::string_view::npos : t - s));
            if (t == std::string_view::npos) break;
            s = t + 1;
        }
        if (f.size() != 15) continue;
        auto num = [](const std::string& x) {
            if (x.empty() || x.size() > 15) return false;
            for (char c : x)
                if (c < '0' || c > '9') return false;
            return true;
        };
        bool ok = true;
        for (int i : {1, 2, 3, 4, 5, 6, 7, 8}) ok = ok && num(f[i]);
        ok = ok && (f[11] == "0" || f[11] == "1") && (f[12] == "0" || f[12] == "1") && !f[13].empty();
        if (!ok) continue;
        GpuRecord r;
        r.index = std::atoi(f[1].c_str());
        r.pci_domain = std::atoi(f[2].c_str());
        r.pci_bus = std::atoi(f[3].c_str());
        r.pci_device = std::atoi(f[4].c_str());
        r.pci_function = std::atoi(f[5].c_str());
        r.total_mib = std::strtoull(f[6].c_str(), nullptr, 10);
        r.compute_units = std::atoi(f[7].c_str());
        r.fit_mib = std::strtoull(f[8].c_str(), nullptr, 10);
        r.uuid = f[9] == "-" ? std::string() : f[9];
        r.luid = f[10] == "-" ? std::string() : f[10];
        r.integrated = f[11] == "1";
        r.supported = f[12] == "1";
        r.arch = f[13];
        r.name = f[14];
        out.push_back(r);
    }
    return out;
}

// the startup line: which GPU and how to choose another - usage only (owner 2026-10-23: the cause is not public)
std::string restrictionNote(const std::vector<GpuRecord>& all, const std::vector<int>& selected, const std::string& reason) {
    std::string used;
    for (int s : selected) used += std::format("{}GPU pci:{} ({})", used.empty() ? "" : ", ", pciId(all, all[s]), all[s].name);
    if (all.size() <= selected.size()) return used + ": the only GPU";
    return std::format("{}: {}; use --device pci:<bus>:<device> to choose another (see --list-devices)", used,
                       reason.empty() ? std::string("chosen with --device or WHIRL_DEVICE") : reason);
}

std::string userVarNote(const std::string& var, const std::string& value) {
    return std::format(
        "GPU: {}={} is set, so WHIRL leaves the GPU visibility as given: device indices (--device N) count only the visible "
        "GPUs",
        var, value);
}

}  // namespace whirl::gpusel
