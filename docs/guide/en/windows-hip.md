**English** | [繁體中文](../zh-TW/windows-hip.md)

# HIP on Windows

**Who this helps:** anyone writing or running HIP code on Windows — kernel authors, llama.cpp
users and contributors on the ROCm/HIP backend, people porting Linux ROCm tooling, and anyone
whose AMD GPU "runs out of memory" with plenty of memory free. Most of this is not in AMD's
documentation; every item below was measured on our machine and states the conditions under
which it was observed.

## <a id="env"></a>0. The environment these findings come from

| Component | Version / detail |
|---|---|
| OS | Windows 11 Home, build 26200 (measurements 2026-09-25 → 2026-10-02) |
| HIP SDK | 7.2 (`C:\Program Files\AMD\ROCm\7.2`): clang for device code, `amdhip64.lib`, headers |
| HIP runtime | `C:\Windows\System32\amdhip64_7.dll` 10.0.3679.0, installed by the Adrenalin driver |
| Host compiler | MSVC 19.44 (Visual Studio 2022 Build Tools 17.14), C++20 |
| GPU 0 | AMD Radeon 8060S (Ryzen AI Max+ 395), gfx1151, unified memory (HIP reports 99.7 GiB) |
| GPU 1 | AMD Radeon AI PRO R9700, gfx1201, 32 GB (HIP reports 31.9 GiB), **USB4 eGPU** |
| Reference | llama.cpp b10686 → b11214 (ROCm and Vulkan builds), used only for comparison |

The R9700 is attached through USB4. Findings marked **[eGPU]** depend on that link and may not
reproduce on a direct PCIe slot. Findings marked **[8060S]** were observed on the APU.

## <a id="pal"></a>1. The driver stack is PAL, not ROCr/KFD

On Linux, HIP sits on ROCr and the KFD kernel driver. On Windows, HIP sits on AMD's **PAL**
(Platform Abstraction Library) under WDDM. Consequences we hit:

- **The Linux tooling does not exist.** `rocprof`, `rocminfo`, `omniperf`/`rocprof-compute`
  are not available. There is no hardware-counter profiler you can attach to a HIP process.
  You build your own measurement (section 3).
- **Error messages come from PAL.** With `AMD_LOG_LEVEL=1` the runtime prints its own errors to
  stderr, and they reference `palvirtual.cpp`. Without it, applications often show only a
  generic failure (`hipErrorLaunchFailure`, llama.cpp's `ROCm error: unspecified launch failure`,
  WHIRL's `HipFailed`). **Set `AMD_LOG_LEVEL=1` first whenever anything fails.**
- **Linux-specific advice does not transfer.** Example: on Linux/ROCm a decode loop on this
  card was reported to alternate between ~28 and ~36 ms per step, fixed by
  `GPU_MAX_HW_QUEUES=1` (a KFD hardware-queue assignment issue). We measured it on Windows with
  a fresh process per run, interleaved with and without the variable:

  | Run (R9700 unless noted) | without | with `GPU_MAX_HW_QUEUES=1` |
  |---|---|---|
  | 27B plain decode, 8 pairs | 28.0 ms/token in all 16 runs (35.67–35.74 tok/s, max–min 0.2%) | same |
  | Ornith plain, 5 pairs | 176.2–176.4 tok/s | 176.2–176.9 |
  | 27B MTP, 4 pairs | 116.11–116.63 | 116.27–116.61 |
  | Server restarted per run, MTP on | 115.96–116.13 | 115.95–116.36 |
  | 8060S 27B plain, 3 pairs, 60 s idle before each | 13.47–13.49 | 13.48–13.49 |

  No bimodality exists on Windows, and the variable has no measurable effect. Do not cargo-cult
  ROCm environment variables from Linux guides.

## <a id="build"></a>2. Building and loading device code

### 2.1 One code object per architecture, embedded

Device code is compiled separately from host code:

```
clang -x hip --offload-arch=gfx1201 --cuda-device-only --no-gpu-bundle-output -O3 kernels.hip -o kernels_gfx1201.co
```

(the HIP SDK's `clang.exe`; one invocation per architecture). The `.co` files are turned into
byte arrays and linked into the executable; at run time `hipModuleLoadData` loads the one for
the selected device. Advantages on Windows: no fat-binary/bundle tooling, nothing to install
next to the exe, and the host code can be built by MSVC while device code uses the SDK clang.

### 2.2 Optional kernels and the sticky last error

When the host probes for a kernel that is absent in the loaded code object (for example an
RDNA 4-only kernel on gfx1151), `hipModuleGetFunction` fails **and leaves an error that a
later `hipGetLastError` returns**. Code that checks the last error after an unrelated call then
reports a bogus failure. WHIRL's `getFunctionOpt` returns null for a missing kernel and clears
the last error; required kernels use `getFunction`, which fails loudly.

### 2.3 `hipModuleLaunchKernel` does not check arguments

Module launches pass an array of pointers to argument values. Nothing checks count or types.
A call site that passed one argument too few (a KV base pointer added to a kernel signature
later) made the kernel read a garbage pointer and crash the process with `0xC0000005`
(access violation) on both GPUs — not a HIP error. Rule: generate launch argument packs from
one definition shared by the kernel signature and the host, and test every launch path
(including benchmark-only paths) after changing a signature.

### 2.4 Resource-usage reports must use the exact build flags

`-Rpass-analysis=kernel-resource-usage` reports VGPRs, SGPRs, scratch and spills per kernel.
Run it with **exactly** the flags of the real build. With missing flags we saw a fake
"Dynamic Stack", symbolic VGPR counts and spills that do not exist in the real build. The other
reliable source is the code object's metadata (VGPR/SGPR/scratch per kernel), which we read
from the built `.co` to confirm that a refactor did not change code generation.

### 2.5 Kernel argument size

Kernel arguments are limited to 4 KB. Our segmented DeltaNet kernel passes per-segment state
pointers by value; at 16 segments that struct is about 2.2 KB. Passing small tables by value
(token ids, segment descriptors, KV page-table arguments) is a deliberate design choice — see
the next section for why.

## <a id="timing"></a>3. Timing and profiling without a profiler

- **`hipEvent` timing on the null stream reads low.** Our first kernel timings came from
  events recorded on the null stream and were consistently too small. Use host wall-clock time
  around a synchronized region, or events on an explicit stream with enough work between them.
- **Per-op events distort what they measure.** An event-per-operation profiler mode dropped
  decode from 34 to 14 tok/s and stretched a server decode cycle from 44 ms to 74 ms; it also
  attributed MTP drafting time to the embedding op. A coarse mode with three events per cycle
  (MTP start, verify start, verify end) still costs about 2.5% on the MoE model. Use profiling
  only to find *proportions*, never to report speed, and switch it off for A/B runs.
- **Knockouts.** For decode, where events distort too much, we build variants that skip one
  kernel class and measure the difference (for example, a 16-row verify of 41.65 ms lost
  4.6 ms without the DeltaNet step + snapshots, 1.2 ms without attention, 1.0 ms without
  RMSNorm+quantize).
- **Standalone probes.** Kernel experiments run in a small host program that loads a code
  object, uses real weights extracted from the GGUF, interleaves variants for 6–12 rounds,
  reports min/median/max, and compares every variant's output byte for byte with the
  production kernel.
- **Defeat the caches in probes.** The R9700 has a 64 MB Infinity Cache (the 8060S a 32 MB
  MALL). A probe that re-reads one weight matrix measures cache bandwidth. Rotate through copies
  totalling more than 512 MB (we use ≥ 512 MB on the R9700, > 2 GB on the 8060S).
- **Task Manager's "Compute" graph does not show HIP work.** It stays flat while the GPU is
  saturated. The VRAM graph's sawtooth during a benchmark was the model being reloaded per
  prompt, not a leak. Use AMD Software's utilization and power readout, or tok/s.

## <a id="memory"></a>4. Memory

### <a id="big-alloc"></a>4.1 A single huge allocation is refused even when memory is free **[8060S]**

On the 8060S (HIP SDK 7.2), PAL refused **one** allocation of about 56 GB while 110 GB were
reported free. Total capacity was never the problem: six 15.3 GB models in six processes
reached 90.46 GB together. The signature:

```
Failed PAL memory allocation!
PAL failed to submit CMD! result:-5
ggml-cuda.cu:107: ROCm error
```

Two things hide the cause in llama.cpp: its logger is an asynchronous queue, so `GGML_ABORT`
can discard the lines naming the failing call (`AMD_LOG_LEVEL=1` prints HIP's errors directly
to stderr and survives the abort); and the HIP backend reports no maximum buffer size, so the
allocator packs the whole model into one buffer. **Fix:** cap individual buffer sizes so the
loader splits them (we used a local cap of 8192 MiB per buffer). Four earlier diagnoses on this
machine (BIOS carve-out, faulty memory, ROCm-only bug, KV quantization) were wrong because they
all assumed a capacity problem. WHIRL allocates weights per tensor group and never asks for one
giant block.

### <a id="mmap"></a>4.2 Loading from a memory-mapped file fails

llama.cpp's ROCm build loads with mmap by default; on this stack that crashes during load with
`ROCm error: unspecified launch failure`, and `AMD_LOG_LEVEL=1` shows PAL `result: -28`.
Workaround: `--no-mmap` on older builds (b10686); from b11214 that flag was removed and
replaced by `--load-mode none` (`-lm none`), and passing the old flag makes `llama-server` fail
to start. WHIRL reads the file and uploads with explicit copies, so it never had the problem.
(WHIRL still memory-maps the GGUF on the **host** for parsing — that is fine; the failure is in
using mapped pages as a GPU upload source.)

### <a id="vmm"></a>4.3 HIP virtual memory management is not usable on WDDM

We wanted per-sequence contiguous virtual KV ranges with physical pages mapped on demand
(`hipMemAddressReserve` / `hipMemCreate` / `hipMemMap`), so attention kernels would not need a
page table. A probe on both GPUs:

- the runtime reports VMM supported, granularity 64 KiB;
- `hipMemcpy` into the mapped range works;
- **kernels only see the first physical allocation mapped into the range.** Writes into pages
  backed by the second physical handle are lost, and in some runs the GPU faults.

Our interpretation: WDDM residency makes only the allocation referenced by the kernel argument
resident. Running this probe on the R9700 while a benchmark was timing on the same card also
faulted the benchmark process — never run probes on a GPU that is being measured. We
implemented a real page table instead (256-token pages; [kv-and-caching.md](kv-and-caching.md)).
The lookup costs 0.3–0.5% of decode time.

### <a id="wddm-demote"></a>4.4 Two large processes on one GPU: WDDM demotes both to shared memory

Twice, a second model process (each ~15+ GiB) was started on the R9700 while one was running.
WDDM did not fail either allocation; it moved parts of **both** processes to shared system
memory (one showed 7.2 of 11.6 GiB as shared), and both slowed to a crawl for about 20 minutes
until killed. Every measurement from that window was invalid. Defenses we now use, in layers:

1. **A named mutex in the executable.** After selecting a device, WHIRL takes
   `Local\whirl-gpu-<device index>`. Windows releases it when the process exits or crashes. A
   second instance waits (printing a message every 30 s) up to a timeout (default 1800 s) and
   then fails. An explicit override exists for deliberate sharing; do not use it for
   measurements.
2. **A file lock in every launcher script.** The mutex only exists in new binaries, and the
   second incident involved an older binary. All scripts open a lock file with share mode 0
   (`FileShare.None`) and hold it for their lifetime; a second launcher waits. After acquiring
   the lock, the launcher also waits until no engine process is alive (leftovers), up to
   10 minutes. Child processes inherit an environment marker so a locked script tree does not
   deadlock on itself. PowerShell and Python launchers interlock (tested both ways).
3. **Confirm your previous background job ended before starting the next one.** Both incidents
   came from assuming a background job had finished after reading part of its log.
4. **Only kill PIDs you recorded yourself.** Killing waiters by process name took down another
   user's unrelated waiting jobs once.

### <a id="shared-usage"></a>4.5 Pinned host memory is reported as "Shared Usage"

Windows exposes per-process, per-adapter GPU memory counters (Dedicated Usage, Shared Usage).
We sample them every 2 s for every engine process and reject any measurement whose shared usage
exceeds 256 MiB (the idle baseline is about 89 MiB), because shared usage normally means VRAM
overflow into system memory and invalidates speed numbers.

But **`hipHostMalloc` (pinned) memory is counted as Shared Usage** even though nothing
overflowed. Experiments that put the token embedding (0.67 GiB) or prefix checkpoints
(1.56 GiB) into pinned host memory raised shared usage by exactly those amounts while dedicated
usage stayed at 31.28–31.31 GiB. Our RAM KV tier (9 GiB pinned by default) does the same. We
now have the server write its pinned allocation size to a declaration file at startup, and the
monitor subtracts the declared size before applying the 256 MiB rule. Observed: shared =
declared tier (8192 or 9216 MiB) + 35–55 MiB.

The same effect appears in llama.cpp: without `GGML_CUDA_NO_PINNED=1`, its CPU-side token
embedding / host compute buffers are pinned and add about 1 GB of shared usage.

### 4.6 Pinned allocation is slow; allocate once

Allocating 16 × 512 MiB of pinned memory took 1.7–1.8 s. CPU reads/writes to pinned memory ran
at 32.7 GB/s. WHIRL allocates its pinned arena once at server start (about +1.6 s startup) and
manages it in 2 MiB blocks; nothing is pinned or unpinned on the request path.

### <a id="zero-copy"></a>4.7 Zero-copy kernel access to pinned memory did not work **[eGPU]**

Kernels that wrote directly to `hipHostMalloc` memory (zero-copy) never landed their data in
host memory on this machine. All host↔device traffic in WHIRL therefore uses `hipMemcpyAsync`
on a non-blocking stream. We have not verified this on a direct-PCIe R9700; treat it as
unverified there rather than as broken.

This is about kernel **writes** to host memory. Kernel **reads** of pinned host memory are used for
the token embedding (`WHIRL_EMBD_HOST`, on by default in the server since 0.1.3); outputs are bitwise
identical to keeping it in VRAM.

### 4.8 `hipMemGetInfo` does not see everything WDDM counts

- Creating a non-blocking stream and the pinned arena reduced `hipMemGetInfo`'s free memory by
  12.8 MiB but raised WDDM's dedicated counter by about 60 MiB. Size VRAM pools *after* creating
  every stream and auxiliary buffer, and keep a reserve.
- The MoE model allocated about 0.5 GiB lazily after the KV pool had been sized to "all
  remaining memory", pushing the process into shared memory (368–399 MiB shared). We now keep a
  per-model reserve: 768 MiB for the dense model, 1.5 GiB for the MoE model. (The source of the
  late allocation was not identified.)
- Code objects count too: a 1.8 MB larger code object cost one 256-token KV page.

### 4.9 GPU identity in monitoring changes across reboots

The monitor keys GPUs by adapter LUID. **The LUID changes after a reboot.** Our report script,
still filtering for the old LUID, found zero samples and printed "all runs in dedicated
VRAM". It now fails when it sees no samples and prints the LUIDs it did see. Rule: a monitoring
check must fail on empty input.

## <a id="streams"></a>5. Streams, copies and synchronization

- **A synchronous `hipMemcpy` drains the queue.** Uploading a few token ids with `hipMemcpy`
  before each MTP draft step waited for all queued GPU work, so the GPU went idle four times per
  cycle. Passing the ids as kernel arguments (a small by-value struct) instead of copying them
  gave +5% end to end. Rule: small per-step data goes in kernel arguments; the device writes
  results it needs next (argmax writes the next token and position into device memory).
- **`hipDeviceSynchronize` waits for every stream, including background copies.** Copies on a
  non-blocking stream did not block `hipStreamSynchronize(main)` (0.00 ms) or a small
  null-stream `hipMemcpy` (0.57 ms with 6000 copies queued), but `hipDeviceSynchronize` waited
  for all of them. Once WHIRL had a background tier stream, every device-wide sync in the server
  became a stall and was replaced by a stream sync.
- **HIP graphs gave nothing on Windows.** Capturing the decode step: 28.87 vs 28.81 ms/token
  early on, and 27.81 vs 27.79 ms later. Host enqueue (0.4–0.7 ms per step for ~600–740
  launches) is already hidden behind GPU execution. The graph path is kept as an option.
- **Kernel launch overhead is real but small.** Each launch has a head/tail of about 2.8 µs
  (a pure-read kernel reaches 605 GB/s on a 50 MB matrix but 626 GB/s on the 1 GB output head).
  At ~740 launches per token that is worth fusing same-input matrices into one launch
  ([kernels.md](kernels.md#grouped)), but it is not worth graphs.

## <a id="limits"></a>6. Launch limits

- **Grid y and z are limited to 65,536 blocks.** A kernel that put the 248,320 output-head rows
  on the y axis returned `HipFailed`. Put the large dimension on x.
- **LDS above ~41 KB per workgroup cut occupancy from 16 to 6 waves per SIMD** in a multi-query
  attention variant, and it was slower. On gfx12 a workgroup can use up to 64 KiB of the WGP's
  128 KiB; blocks of 36–64 KiB still run two per WGP. **Do not size kernels from
  `hipOccupancyMaxActiveBlocksPerMultiprocessor`** — it assumes 64 KiB per WGP and halves the
  LDS-limited occupancy ([pitfalls KERN-20b](pitfalls.md#kern-20b)).

## <a id="devices"></a>7. Selecting the right GPU in a two-GPU machine

| Runtime | Order on this machine |
|---|---|
| HIP / ROCm (llama.cpp ROCm) | `ROCm0` = 8060S, `ROCm1` = R9700 |
| HIP with `HIP_VISIBLE_DEVICES=1` | R9700 becomes `ROCm0` |
| Vulkan (llama.cpp Vulkan) | `Vulkan0` = 8060S, `Vulkan1` = R9700 |

Every tool must select the GPU explicitly; defaults land on the integrated GPU. WHIRL selects
by name/architecture (each build has a default device; `WHIRL_DEVICE` overrides). Note that
Vulkan reports a warp size of 64 for RDNA, while HIP waves on RDNA are 32 lanes; all WHIRL
reductions and shuffles are written for wave32.

**[8060S] The two GPUs are not independent.** The 8060S shares power and cooling with the CPU.
A build or a benchmark on the R9700 (which loads the CPU) slowed 8060S decode by 11–18% in one
run. Never time the 8060S while anything else runs.

## <a id="egpu"></a>8. eGPU (USB4) effects **[eGPU]**

These are environment limitations of the development machine. We measured them so that nobody
mistakes them for engine behavior, and we did not optimize for them.

| Measurement | Result |
|---|---|
| Host link, `hipMemcpyAsync`, any piece size 16 KiB–256 MiB | D2H 3.78–3.81 GB/s, H2D 3.84–3.86 GB/s |
| 6018 scattered KV-page pieces | 3.8 GB/s, enqueue 0.45 µs per copy |
| H2D staging into VRAM + D2D scatter vs direct piece copies | 3.51 vs 3.69–3.83 GB/s (staging is slower) |
| Kernel dispatch while H2D saturates the link | 1500 small kernels: 15 → 64 ms; 20 large kernels: +4%; small sync copy 0.11 → 0.25 ms |
| `GPU_BLIT_ENGINE_TYPE` | no effect |
| `PAL_DISABLE_SDMA=1` | moves the cost elsewhere; copies get slower |
| Effect on serving | other slots' decode cycle 43.0 → 50.4 ms (+17%) for the 0.5–2.0 s a KV restore runs |

The dispatch slowdown is proportional to bytes transferred; pacing the copies or changing the
piece size did not reduce it. On a direct PCIe x16 link we expect neither the low bandwidth nor
the dispatch interference, but we have not measured that.

## <a id="storage"></a>9. Storage I/O for the SSD tier

| NVMe access | Throughput |
|---|---|
| Buffered write | 1.8 GB/s |
| Unbuffered write / read | 5.1–5.3 / 5.1 GB/s |
| Unbuffered read, 2 MiB, queue depth 1 | 4.1 GB/s |
| Unbuffered read, 8 MiB, QD1 | 5.06 GB/s |
| Unbuffered read, 2 MiB, QD4 (4 overlapped reads, completed in order) | 7.0 GB/s |

Use `FILE_FLAG_NO_BUFFERING` (aligned buffers) and keep several overlapped reads in flight.

## <a id="shell"></a>10. Shell and tooling traps on Windows

| Trap | Symptom | Rule |
|---|---|---|
| Windows PowerShell 5.1 reads UTF-8 files without BOM as the ANSI code page | a `.ps1` containing Chinese text fails to parse | run scripts with PowerShell 7 (`pwsh`) |
| Console code page cp950 (zh-TW) | Python test harness dies with `UnicodeEncodeError` when printing ☔ or é | set `PYTHONIOENCODING=utf-8` in every launcher |
| Bash heredocs and inline strings | backslashes eaten; `\x..` and `\u....` rewritten; generated source fails to compile | write patch/edit scripts as files, then run them |
| Non-raw Python triple-quoted strings in code generators | `\n` becomes a real newline inside a generated string literal; raw strings double backslashes | generate source from files, or test the generator's output |
| Git Bash `sed -i` on a CRLF file | the whole file is rewritten with LF line endings | edit CRLF files with a tool that preserves line endings |
| Calling `bash` from Python | runs WSL's `bash.exe`, not Git Bash | never call `bash` from Python; use pure Python or explicit paths |
| Git Bash `$!` | is an MSYS PID; `taskkill` cannot use it | start background processes with `Start-Process -PassThru` to get the Windows PID |
| `Invoke-WebRequest` errors are non-terminating | a failed download was followed by the script's "clear target folder" step, emptying a working install | verify the archive before touching the destination |
| GitHub release downloads | 130–176 KB/s and dropped connections | `curl -C -` with retries |
| Stale output files | a check script compared outputs left by a previous run after the binary failed to start, and printed "ok" | delete expected outputs before each run; check exit codes |
| Long builds on a shared machine | interactive work and GPU timing disturbed | build at low priority (`start /low`, idle priority class) and never build while timing |

## 11. Checklist

1. `AMD_LOG_LEVEL=1` before debugging anything.
2. Select the GPU explicitly; print its name and architecture at startup.
3. One model process per GPU — enforce it with a named mutex *and* a file lock.
4. Watch per-process Dedicated/Shared Usage; subtract declared pinned memory; fail on empty data.
5. No mmap-backed GPU uploads; no single giant allocations.
6. No synchronous copies or device-wide syncs on hot paths; small data in kernel arguments.
7. Time with wall clock around synchronized regions; rotate > 512 MB of data in probes.
8. Treat eGPU host-link effects as environment limits; confirm on PCIe before optimizing for them.

All Windows/HIP pitfalls, with symptom → cause → fix → gate, are also listed in
[pitfalls.md](pitfalls.md#hip).
