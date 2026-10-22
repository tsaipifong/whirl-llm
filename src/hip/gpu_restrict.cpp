// GPU restriction at startup (s8-719 root fix; see include/whirl/gpu_select.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL from the Win32 documentation (CreateProcessW, anonymous pipes) and the HIP runtime
// API (hipDeviceProp_t). Never reads stdin and never waits for the user: every outcome is a log line, a
// result, or an error with the list of GPUs.

#include "whirl/gpu_select.h"
#include "whirl/hip.h"
#include "whirl/hip_runtime_loader.h"
#include "whirl/uma_mem.h"

#include "hip_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::gpusel {

using hip::runtime::rt;

namespace {

Restriction g_r;
std::once_flag g_once;

// hipDeviceProp_t::uuid. Measured 2026-10-23 on this machine (driver 10.0.3679): the Windows runtime fills it
// with the ASCII text of the PCI location ("0000019600000000" for PCI bus 196 = bytes 30 30 30 30 30 31 39 36 ...),
// not a per-card serial, so it says nothing the PCI address does not. Such an id (all bytes ASCII hex digits)
// is dropped: WHIRL then shows and locks by the PCI address. A real binary uuid (another driver) is kept.
std::string hexUuid(const hipUUID& u) {
    const auto* b = reinterpret_cast<const unsigned char*>(u.bytes);
    bool any = false, text = true;
    for (int i = 0; i < 16; ++i) {
        any = any || b[i] != 0;
        text = text && ((b[i] >= '0' && b[i] <= '9') || (b[i] >= 'a' && b[i] <= 'f') || (b[i] >= 'A' && b[i] <= 'F'));
    }
    if (!any || text) return {};
    std::string s;
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) s += '-';
        s += std::format("{:02x}", b[i]);
    }
    return s;
}

std::string hexLuid(const char* l) {
    const auto* b = reinterpret_cast<const unsigned char*>(l);
    bool any = false;
    for (int i = 0; i < 8; ++i) any = any || b[i] != 0;
    if (!any) return {};
    std::string s;
    for (int i = 7; i >= 0; --i) s += std::format("{:02x}", b[i]);  // little-endian LUID {LowPart, HighPart} as one 64-bit value
    return s;
}

// runs `<this exe> --whirl-gpu-probe` and returns its stdout ("" on any failure; why = reason)
std::string runProbe(std::string* why) {
    std::wstring exe(1024, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
    if (n == 0 || n >= exe.size()) {
        *why = "the program path is unknown";
        return {};
    }
    exe.resize(n);
    const std::wstring cmd = L"\"" + exe + L"\" " + std::wstring(k_probe_arg, k_probe_arg + std::strlen(k_probe_arg));
    ChildRun r = runChild(exe, cmd, k_probe_timeout_ms);
    *why = r.why;
    return r.out;
}

}  // namespace

ChildRun runChild(const std::wstring& exe, const std::wstring& cmdline, unsigned timeout_ms) {
    ChildRun r;
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) {
        r.why = "no pipe";
        return r;
    }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof(si);
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = nul;   // never interactive
    si.StartupInfo.hStdOutput = wr;
    si.StartupInfo.hStdError = nul;   // AMD_LOG_LEVEL output of the child is not needed
    // the child inherits exactly the pipe and NUL, not every inheritable handle of the parent (log file, socket)
    HANDLE inherit[2] = {wr, nul};
    const DWORD n_inherit = nul != INVALID_HANDLE_VALUE ? 2 : 1;
    SIZE_T attr_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
    std::string attr_buf(attr_size, '\0');
    auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
    bool have_list = attr_size > 0 && InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size) &&
                     UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, n_inherit * sizeof(HANDLE), nullptr, nullptr);
    si.lpAttributeList = have_list ? attrs : nullptr;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = cmdline;
    // no Windows error dialog for a crash in the hidden helper: the child inherits this error mode
    const UINT old_mode = GetErrorMode();
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    const BOOL ok = have_list ? CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                               CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &si.StartupInfo, &pi)
                              : FALSE;
    const DWORD create_err = GetLastError();
    SetErrorMode(old_mode);
    if (have_list) DeleteProcThreadAttributeList(attrs);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        r.why = have_list ? std::format("the helper process did not start (error {})", create_err)
                          : std::string("the helper process could not be limited to its own handles");
        return r;
    }
    r.started = true;
    std::string out;
    char buf[4096];
    DWORD got = 0;
    // the child exits after enumerating; the pipe closes with it. Bounded by timeout_ms
    const ULONGLONG t0 = GetTickCount64();
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) break;  // closed: child done
        if (avail > 0) {
            if (!ReadFile(rd, buf, sizeof buf, &got, nullptr) || got == 0) break;
            out.append(buf, got);
            continue;
        }
        if (WaitForSingleObject(pi.hProcess, 20) == WAIT_OBJECT_0) {
            while (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail > 0 && ReadFile(rd, buf, sizeof buf, &got, nullptr) && got > 0)
                out.append(buf, got);
            break;
        }
        if (GetTickCount64() - t0 > timeout_ms) {
            TerminateProcess(pi.hProcess, 1);
            r.timed_out = true;
            r.why = std::format("the helper process did not finish in {} s", (timeout_ms + 999) / 1000);
            out.clear();
            break;
        }
    }
    DWORD code = 1;
    WaitForSingleObject(pi.hProcess, 5000);
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(rd);
    r.exit_code = code;
    if (code != 0 && r.why.empty()) r.why = std::format("the helper process ended with code {}", code);
    if (code == 0) r.out = std::move(out);
    return r;
}

namespace {

void setEnv(const char* k, const std::string& v) {
    _putenv_s(k, v.c_str());           // this module's CRT and the process environment
    SetEnvironmentVariableA(k, v.c_str());
    // the HIP runtime may use the shared UCRT (ucrtbase.dll): update its copy too
    if (HMODULE u = GetModuleHandleW(L"ucrtbase.dll"))
        if (auto f = reinterpret_cast<int(__cdecl*)(const char*, const char*)>(GetProcAddress(u, "_putenv_s"))) f(k, v.c_str());
}

}  // namespace

std::vector<GpuRecord> enumerate() {
    std::vector<GpuRecord> out;
    const int n = hip::deviceCount();
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p{};
        if (rt().hipGetDevicePropertiesR0600(&p, i) != hipSuccess) continue;
        GpuRecord r;
        r.index = i;
        r.name = p.name;
        r.arch = p.gcnArchName;
        r.pci_domain = p.pciDomainID;
        r.pci_bus = p.pciBusID;
        r.pci_device = p.pciDeviceID;
        r.pci_function = 0;  // not reported by HIP; one GPU per PCI device
        const hip::DeviceInfo d = hip::describeDevice(i);
        r.total_mib = d.total_mem >> 20;
        r.compute_units = p.multiProcessorCount;
        r.uuid = hexUuid(p.uuid);
        r.luid = hexLuid(p.luid);
        r.integrated = p.integrated != 0;
        for (const hip::EmbeddedObject& o : hip::embeddedObjects())
            r.supported = r.supported || r.arch.rfind(o.arch, 0) == 0;
        r.fit_mib = r.total_mib;
        if (r.integrated) {
            const hip::UmaMem u = hip::umaMemory(i);
            if (u.known) {
                r.fit_mib = u.dedicated >> 20;
                r.mem_text = hip::umaSizeText(u);
            }
        }
        out.push_back(r);
    }
    return out;
}

int probeMain() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);  // a crash here never shows a dialog
    if (!hip::runtime::load().ok) return 3;
    try {
        for (const GpuRecord& r : enumerate()) std::printf("%s\n", formatRecord(r).c_str());
    } catch (const std::exception&) {
        return 4;
    }
    std::fflush(stdout);
    return 0;
}

const Restriction& current() { return g_r; }

int chosenPosition(const std::vector<GpuRecord>& recs, std::string_view spec_arg) {
    if (g_r.active) return 0;  // identity proven in restrictToSelected
    std::string spec(spec_arg);
    if (spec.empty())
        if (const char* v = std::getenv("WHIRL_DEVICE")) spec = v;
    return selectGpus(recs, spec)[0];
}

const Restriction& restrictToSelected(std::string_view spec_arg) {
    std::call_once(g_once, [&] {
        g_r.done = true;
        std::string spec(spec_arg);
        if (spec.empty())
            if (const char* v = std::getenv("WHIRL_DEVICE")) spec = v;
        (void)parseDeviceList(spec);  // a malformed list is an error before any GPU work
        const std::string var = userVisibilityVar([](const char* k) { return std::getenv(k); });
        if (!var.empty()) {
            const char* v = std::getenv(var.c_str());
            g_r.note = userVarNote(var, v ? v : "");
            return;
        }
        if (const char* off = std::getenv("WHIRL_GPU_RESTRICT"); off && off[0] == '0') {
            g_r.note = "GPU: WHIRL_GPU_RESTRICT=0: every GPU stays visible to the GPU runtime (pinned host memory is counted on each)";
            return;
        }
        std::string why;
        const std::string text = runProbe(&why);
        g_r.all = parseRecords(text);
        if (g_r.all.empty()) {
            g_r.note = "GPU: could not list the GPUs in a helper process (" + (why.empty() ? std::string("no GPU reported") : why) +
                       "), so every GPU stays visible to the GPU runtime";
            return;
        }
        g_r.selected = selectGpus(g_r.all, spec);  // throws SelectError (shown to the user, exit non-zero)
        const std::string reason = spec.empty() ? defaultGpu(g_r.all).reason : std::string();
        if (g_r.all.size() == 1) {
            g_r.note = restrictionNote(g_r.all, g_r.selected, reason);
            return;
        }
        std::vector<int> ordinals;
        for (int s : g_r.selected) ordinals.push_back(g_r.all[s].index);
        setEnv("HIP_VISIBLE_DEVICES", visibleDevicesValue(ordinals));
        // prove the mapping: the runtime must now see exactly the chosen GPU(s), identified by PCI address,
        // uuid, LUID, name and architecture (never by the ordinal alone)
        const std::vector<GpuRecord> now = enumerate();
        bool same = now.size() == g_r.selected.size();
        for (std::size_t i = 0; same && i < now.size(); ++i) same = sameGpu(now[i], g_r.all[g_r.selected[i]]);
        if (same) {
            g_r.active = true;
            g_r.note = restrictionNote(g_r.all, g_r.selected, reason);
            return;
        }
        if (now.size() == g_r.all.size()) {
            // the runtime ignored the variable: carry on with every GPU visible (the pinned-memory clamp still applies)
            g_r.note = restrictionNote(g_r.all, g_r.selected, reason) +
                       " - BUT the GPU runtime did not apply HIP_VISIBLE_DEVICES, so every GPU stays visible";
            return;
        }
        throw SelectError(std::format(
            "the GPU runtime shows a different GPU after HIP_VISIBLE_DEVICES={} (expected {} {}, pci:{}). Set HIP_VISIBLE_DEVICES "
            "yourself to choose the GPU, or WHIRL_GPU_RESTRICT=0 to keep every GPU visible.",
            visibleDevicesValue(ordinals), g_r.all[g_r.selected[0]].index, g_r.all[g_r.selected[0]].name,
            pciId(g_r.all, g_r.all[g_r.selected[0]])));
    });
    return g_r;
}

}  // namespace whirl::gpusel
