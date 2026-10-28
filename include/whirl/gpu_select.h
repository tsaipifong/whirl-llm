// GPU selection and startup restriction (s8-719 root fix, 0.2.3).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. The pure part (records, --device list parsing, the
// automatic pick, the HIP_VISIBLE_DEVICES value, the identity check) has no
// HIP dependency so the CPU unit tests cover it; restrictToSelected() in
// src/hip/gpu_restrict.cpp does the process work.
//
// Why: a pinned host buffer (hipHostMalloc) is charged as shared GPU memory
// on EVERY GPU the HIP runtime sees (measured, progress\s8_719.HANDOFF.md).
// With the R9700 attached, an 8 GiB RAM tier of a server on the 8060S made the
// first request fail with hipError 719. A process that uses one GPU therefore
// makes HIP see only that GPU, by setting HIP_VISIBLE_DEVICES before the HIP
// runtime initialises (ROCclr reads its flags at the first API call; the same
// is done for GPU_RESOURCE_CACHE_SIZE in server_main.cpp, measured).
//
// The ordinal HIP_VISIBLE_DEVICES takes is the runtime's own enumeration order.
// WHIRL gets it from a child process of itself (`<exe> --whirl-gpu-probe`)
// that enumerates every GPU with the unrestricted runtime, and afterwards
// proves in-process that the one GPU left is the chosen one (PCI domain / bus /
// device, LUID, name and architecture must all match).
//
// Multi-GPU (later): selectGpus() already returns a list and
// visibleDevicesValue() writes several ordinals; only the "one GPU" check in
// selectGpus() has to go.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::gpusel {

// One GPU as the HIP runtime enumerates it.
struct GpuRecord {
    int index = -1;               // HIP ordinal in the enumeration this process sees
    std::string name;             // e.g. "AMD Radeon AI PRO R9700"
    std::string arch;             // gcnArchName, e.g. "gfx1201"
    int pci_domain = 0;
    int pci_bus = 0;
    int pci_device = 0;
    int pci_function = 0;
    std::uint64_t total_mib = 0;  // memory the runtime reports (an integrated GPU: carve-out + shared RAM)
    int compute_units = 0;
    std::uint64_t fit_mib = 0;    // memory a model can use: VRAM of a discrete GPU, the BIOS carve-out of an integrated one
                                  // (DXGI); 0 = unknown (total_mib is used)
    std::string uuid;             // hipDeviceProp_t::uuid as 8-4-4-4-12 hex, "" if the runtime reports none (all zero)
    std::string luid;             // 16 hex digits (Windows adapter LUID: changes every boot), "" if unknown
    std::string mem_text;         // the memory as the device list prints it ("" = total_mib in GiB)
    bool integrated = false;
    bool supported = false;       // this build has GPU kernels for its architecture
};

// A selection error: the text is meant for the user (it names the GPUs present).
struct SelectError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// "a,b" -> {"a","b"}; items are trimmed. An empty item (",a", "a,,b") is an error.
// An empty / blank spec -> {} (= automatic pick).
std::vector<std::string> parseDeviceList(std::string_view spec);

// The GPU(s) to use, as indices into `gpus` order (= HIP ordinals).
// spec: --device / -dev / WHIRL_DEVICE value, a comma-separated list. Each item is one of
//   - an index as the device list prints it ("1"), or llama.cpp's name for it ("ROCm1"; "HIP1" also
//     accepted). The order is the GPU runtime's enumeration order: stable only while the hardware, the
//     driver and the environment stay the same (HIP_VISIBLE_DEVICES renumbers);
//   - "uuid:<id>" (the full id the device list prints, or a unique prefix; case ignored), when the
//     runtime reports one;
//   - the PCI bus and device numbers, decimal as Task Manager / Device Manager show them: "pci:71:0"
//     ("PCI bus 71, device 0"; a pasted "pci:71:0.0" works too).
//     Stable while the card stays in the same slot; a USB4 / Thunderbolt eGPU can get another bus number on
//     another port or after a replug;
//   - an alias, matched on the marketing NAME: r9700 ("R9700" in the name), 8060s; igpu = an integrated
//     GPU; strix = gfx1151; an architecture prefix ("gfx1201"); or a part of the name ("9070", "AMD Radeon
//     AI PRO R9700"); case is ignored. A name / alias / architecture that matches MORE than one GPU is an
//     error that lists the matching GPUs (index, name, PCI address, memory) - never the first match.
// Empty spec: defaultGpu(). This version uses one GPU: a list of two or more throws
// "only one GPU is supported in this version". Throws SelectError (listing the GPUs) when nothing matches.
std::vector<int> selectGpus(const std::vector<GpuRecord>& gpus, std::string_view spec);

// The default GPU (no --device): among the GPUs this build has kernels for, the one with the most GPU
// memory (fit_mib: VRAM of a discrete GPU, the dedicated carve-out of an integrated one, as Task Manager
// shows "Dedicated GPU memory"), discrete or integrated alike; a tie -> the lowest index (owner
// 2026-10-23). index -1 = none. reason: one clause for the startup line.
struct AutoChoice {
    int index = -1;
    std::string reason;
};
AutoChoice defaultGpu(const std::vector<GpuRecord>& gpus);

// The automatic pick alone (-1 when no GPU has kernels in this build).
int autoPick(const std::vector<GpuRecord>& gpus);  // defaultGpu(gpus).index

// The PCI location in DECIMAL, as Windows shows it (Task Manager > Performance > GPU > Physical location,
// Device Manager > Location: "PCI bus 71, device 0, function 0"). pciText / pciLongText are the full
// internal form ("71:0.0"; a non-zero PCI domain in front) for the lock key and logs of the probe.
// Users read and type bus + device, always: pciId "71:0", pciIdLong "PCI bus 71, device 0" (the function
// number is never shown). --device takes pci:71:0 and (silently) a pasted pci:71:0.0; pci:71 alone is an
// error that says to write pci:71:0.
std::string pciText(const GpuRecord& r);
std::string pciLongText(const GpuRecord& r);
// (see above; `all` is kept for a later shared-bus rule)
std::string pciId(const std::vector<GpuRecord>& all, const GpuRecord& r);
std::string pciIdLong(const std::vector<GpuRecord>& all, const GpuRecord& r);

// Name of the one-process-per-GPU mutex: keyed by the uuid when the runtime reports one, else the PCI
// address; never the index (after the restriction every process sees its GPU as index 0).
std::string lockName(const GpuRecord& r);
// The name passed to CreateMutexW (src/model/device.cpp lockGpu): lockName widened byte by byte (ASCII). Every WHIRL
// version on one GPU must use exactly this name, or two servers stop waiting for each other (GPU_WAIT); pinned by
// tests/unit/unit_tests.cpp testPlatformProc (arch025 PL4).
std::wstring lockMutexName(const GpuRecord& r);

// "    device 0: NAME (ARCH), pci:3:0, 31.9 GiB" lines, for error texts.
std::string listText(const std::vector<GpuRecord>& gpus);

// The device list (whirl devices / --list-devices): one line per GPU
// "  device 0: NAME (ARCH), 31.9 GiB, 32 CUs, pci:3:0 (PCI bus 3, device 0)[, uuid U]  [supported] [default]"
// and a closing note on index vs pci / uuid stability. def = autoPick (-1 = none).
std::string listingText(const std::vector<GpuRecord>& gpus, int def);

// HIP_VISIBLE_DEVICES value for these ordinals: "1", "0,1".
std::string visibleDevicesValue(const std::vector<int>& ordinals);

// The first GPU-visibility variable the user set (HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES,
// ROCR_VISIBLE_DEVICES, GPU_DEVICE_ORDINAL), "" if none. WHIRL never overrides one of them.
std::string userVisibilityVar(const std::function<const char*(const char*)>& getenv_fn);

// arch025-c2fu: the reason text when the GPU runtime shows NO GPU at all. With no visibility variable set it is
// "the AMD graphics driver reports no GPU" (the real no-driver / no-GPU case). When the user set one
// (userVisibilityVar), the text names it and its value and how to remove it. An empty value ("set to an empty value,
// which hides every GPU" for HIP_VISIBLE_DEVICES=) is the cause, so nothing is said about the driver; a non-empty value
// "may hide the GPU", and a driver problem is also possible, so the driver hint follows (arch025-X8bfix). No trailing
// period / newline. getenv_fn as for userVisibilityVar (CPU-tested).
std::string noGpuText(const std::function<const char*(const char*)>& getenv_fn);

// Same physical GPU (PCI domain/bus/device/function, uuid and LUID when both sides know them, name, arch).
bool sameGpu(const GpuRecord& a, const GpuRecord& b);

// Probe protocol (child -> parent, one line per GPU, tab separated, "-" = empty):
// "whirl-gpu\t<index>\t<domain>\t<bus>\t<device>\t<function>\t<mib>\t<cus>\t<uuid>\t<luid>\t<integrated 0|1>\t
//  <supported 0|1>\t<arch>\t<name>"
std::string formatRecord(const GpuRecord& r);
std::vector<GpuRecord> parseRecords(std::string_view text);  // ignores other lines

// The log line for a restriction: the GPU used (name, index, pci, uuid), how many others are hidden and
// how to choose another.
// reason: defaultGpu().reason for the automatic pick, "" for a --device choice.
std::string restrictionNote(const std::vector<GpuRecord>& all, const std::vector<int>& selected, const std::string& reason);
// The log line when the user set a visibility variable (WHIRL leaves it; indices count the visible GPUs).
std::string userVarNote(const std::string& var, const std::string& value);

// ---------------------------------------------------------------------------
// process side (src/hip/gpu_restrict.cpp, needs the HIP runtime)

// What restrictToSelected() decided (one per process).
struct Restriction {
    bool done = false;              // restrictToSelected() ran
    bool active = false;            // WHIRL set HIP_VISIBLE_DEVICES; the chosen GPU is now HIP device 0
    std::vector<GpuRecord> all;     // every GPU (probe), empty when not probed
    std::vector<int> selected;      // ordinals into `all`
    std::string note;               // the one log line (may be several sentences)
};

// Decides the GPU and makes the HIP runtime see only it. Must run before the first HIP API call of the
// process (the runtime DLL may already be loaded: ROCclr reads HIP_VISIBLE_DEVICES when it initialises at
// the first call). spec: --device / -dev value, else WHIRL_DEVICE. Nothing is changed when the user set a
// visibility variable, WHIRL_GPU_RESTRICT=0, or only one GPU exists. Throws SelectError for a bad spec.
// Runs once per process; later calls return the first result.
const Restriction& restrictToSelected(std::string_view spec);
const Restriction& current();

// `<exe> --whirl-gpu-probe`: prints formatRecord() lines for every GPU and returns the exit code.
inline constexpr const char* k_probe_arg = "--whirl-gpu-probe";
int probeMain();

// How long the parent waits for the probe child before it kills it and keeps every GPU visible.
inline constexpr unsigned k_probe_timeout_ms = 10000;  // the probe itself takes about 0.1 s

// Runs one helper process (the probe) and captures its stdout; stdin and stderr are NUL. Never waits longer
// than timeout_ms (then the child is terminated). out is empty unless the child exited with code 0.
// The child inherits only the pipe and NUL, and runs with SEM_NOGPFAULTERRORBOX (no crash dialog).
struct ChildRun {
    bool started = false;
    bool timed_out = false;
    unsigned long exit_code = 1;
    std::string out;
    std::string why;  // "" on success
};
ChildRun runChild(const std::wstring& exe, const std::wstring& cmdline, unsigned timeout_ms);

// Every GPU this process sees, as records (HIP calls).
std::vector<GpuRecord> enumerate();

// userVisibilityVar / noGpuText with this process's environment (arch025-c2fu): for the "no GPU" messages of whirl,
// whirl-server and vis-encode when enumerate() is empty.
std::string userVisibilityVarHere();
std::string noGpuTextHere();

// Position in `recs` (= enumerate() after restrictToSelected) of the GPU to use: 0 when the restriction is
// active (the runtime shows only the chosen GPU), else selectGpus(recs, spec or WHIRL_DEVICE) (default rule
// when both are empty). The one place whirl, whirl-server and vis-encode decide it. Throws SelectError.
int chosenPosition(const std::vector<GpuRecord>& recs, std::string_view spec);

}  // namespace whirl::gpusel
