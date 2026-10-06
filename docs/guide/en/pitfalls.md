**English** | [繁體中文](../zh-TW/pitfalls.md)

# Pitfalls: everything that bit us, and how we catch it now

**Who this helps:** HIP developers on Windows; RDNA 3.5 / RDNA 4 kernel authors; llama.cpp / ggml
contributors (tokenizer, ROCm/HIP backend, MTP, quantization); people building inference servers
for hybrid attention + recurrent models; anyone benchmarking AMD GPUs on laptops or eGPUs.

Many of these traps are not documented anywhere else, because few people run a from-scratch HIP
engine natively on Windows. Each entry gives the **conditions** under which we saw it (GPU, HIP SDK
7.2, Windows 11 build 26200, eGPU vs direct PCIe where it matters), the **symptom** with exact
error strings, the **root cause** and the evidence that proved it, the **fix** stated as a general
rule, and the **gate** — the check that now catches it automatically.

Conventions: R9700 = Radeon AI PRO R9700 (gfx1201, attached via USB4). 8060S = Radeon 8060S
(gfx1151, Ryzen AI Max+ 395 in an ASUS ROG Flow Z13). Numbers are from the research prototype.
Categories: [HIP/Windows](#hip) · [Toolchain and shell](#tool) · [Kernels and numerics](#kern) ·
[Speculative decoding exactness](#mtp) · [Server and caching](#srv) · [Tokenizer and
templates](#tok) · [Model files and quantization](#quant) · [Vision](#vis) · [Measurement](#meas) ·
[eGPU](#egpu).

## <a id="lessons"></a>Generalizable lessons

1. **On Windows/PAL, "out of memory" may mean "one allocation too large".** Set `AMD_LOG_LEVEL=1`
   before forming any theory; four wrong diagnoses preceded the right one ([HIP-2](#hip-2)).
2. **Bit-exactness is a design property, not a test result.** Batched and unbatched paths must share
   the same arithmetic (same per-unit expression, same reduction tree, explicit fma). Once that holds,
   "speculative output == plain output" becomes a one-line gate that catches whole classes of bugs
   ([MTP-1](#mtp-1), [KERN-1](#kern-1)).
3. **Test the default path.** Every gate pinned an environment variable, so a bug that existed only
   when *nothing* was set shipped once ([SRV-6](#srv-defaultenv)). Keep one test with zero
   configuration.
4. **Checks must fail on empty input.** A memory monitor that found no samples reported success; a
   script that compared stale files reported "ok" ([HIP-21](#hip-21), [TOOL-11](#tool-11)).
5. **Measure the ceiling before optimizing.** A pure-read or register-only probe showed that most
   formats were already at 97–100% of bandwidth and pointed at the two that were not
   ([KERN-13](#kern-13)); a memory-pattern probe showed a "memory problem" was really an instruction
   scheduling problem.
6. **On RDNA, registers decide speed.** Read VGPR/scratch/spill for every kernel change; many of our
   biggest wins were removing spills the compiler introduced for reasons unrelated to the math
   ([KERN-4](#kern-4)…[KERN-12](#kern-12)).
7. **Never compare speculative speed across different outputs.** Content changes acceptance; compare
   ms per cycle, interleave A/B, report min–max, and keep small but reproducible gains
   ([MEAS-2](#meas-2), [MEAS-7](#meas-7)).
8. **Recurrent state changes caching.** A DeltaNet/Mamba-style state cannot be truncated like KV;
   reuse needs checkpoints, and checkpoint boundaries must be part of the prefill schedule for cold and
   resumed runs alike ([SRV-1](#srv-1), [SRV-12](#srv-12)).
9. **Tokenizer bugs are performance bugs.** A whitespace pre-split bug made prompts out-of-distribution,
   broke n-gram drafting and lowered prefix-cache hits — while every output still looked fine
   ([TOK-1](#tok-1)).
10. **Anything that outlives a request must be reset, re-initialized or returned at the request
    boundary:** timing tables ([MTP-4](#mtp-4)), borrowed scratch with padding ([VIS-1](#vis-1)),
    KV pages ([SRV-5](#srv-5)), tier-entry links ([SRV-9](#srv-9)).
11. **WDDM does not fail a second large process — it demotes both.** Enforce one model process per GPU
    with a named mutex *and* launcher file locks ([HIP-5](#hip-5)).
12. **Label environment limits; don't optimize them away.** eGPU link effects and laptop thermals were
    measured and documented, not worked around ([EGPU-1](#egpu-1), [MEAS-3](#meas-3)).
13. **Linux ROCm advice does not transfer to Windows** ([HIP-17](#hip-17)), and reference engines can
    be broken too — check the reference before comparing ([MEAS-10](#meas-10)).

---

## <a id="hip"></a>1. HIP on Windows

### <a id="hip-1"></a>HIP-1 · The Linux tooling and error messages are not there
- **Conditions:** any HIP program on Windows (HIP SDK 7.2, Adrenalin `amdhip64_7.dll` 10.0.3679.0).
- **Symptom:** `rocprof`, `rocminfo`, `omniperf` do not exist; failures surface as generic codes
  (`hipErrorLaunchFailure`, llama.cpp's `ROCm error: unspecified launch failure`, our `HipFailed`).
- **Root cause:** HIP on Windows runs on AMD's PAL under WDDM, not ROCr/KFD.
- **Fix:** run with `AMD_LOG_LEVEL=1`; the runtime then prints its own errors (they reference
  `palvirtual.cpp`) to stderr. Build your own timing ([windows-hip.md](windows-hip.md#timing)).
- **Detection now:** every launcher can set `AMD_LOG_LEVEL=1`; unexplained failures are re-run with it
  before anything else.

### <a id="hip-2"></a>HIP-2 · A single huge allocation is refused while memory is free
- **Conditions:** 8060S, unified memory, HIP SDK 7.2, llama.cpp ROCm loading a model of ~56 GB with
  110 GB reported free.
- **Symptom:** `Failed PAL memory allocation!` then `PAL failed to submit CMD! result:-5`, then
  `ggml-cuda.cu:107: ROCm error` with no cause.
- **Root cause:** PAL rejects one allocation of that size; total capacity is fine (six 15.3 GB models
  in six processes reached 90.46 GB). llama.cpp's HIP backend reports no maximum buffer size, so the
  whole model goes into one buffer; its asynchronous logger can drop the lines naming the failing call
  when `GGML_ABORT` fires.
- **Fix:** cap individual buffer sizes so the loader splits (8192 MiB worked). In your own code,
  allocate per tensor group. Four earlier theories (BIOS carve-out, faulty RAM, ROCm-only bug, KV
  quantization) were wrong because they assumed a capacity problem.
- **Detection now:** `AMD_LOG_LEVEL=1` on load failures; WHIRL never requests one giant block.

### <a id="hip-3"></a>HIP-3 · Loading from memory-mapped files crashes (llama.cpp)
- **Conditions:** llama.cpp ROCm b10686 → b11214, R9700.
- **Symptom:** crash during load with `ROCm error: unspecified launch failure`; `AMD_LOG_LEVEL=1` shows
  PAL `result: -28`.
- **Root cause:** using mmap-backed pages as the upload source fails on this stack.
- **Fix:** `--no-mmap` (b10686); in b11214 that flag was **removed** and replaced by `--load-mode none`
  (`-lm none`) — the old flag makes `llama-server` refuse to start. In your own engine, read the file and
  upload with explicit copies (host-side mmap for parsing is fine).
- **Detection now:** all llama.cpp comparison scripts pass `--load-mode none`.

### <a id="hip-4"></a>HIP-4 · HIP virtual memory: kernels only see the first physical allocation
- **Conditions:** both GPUs, WDDM, `hipMemAddressReserve` / `hipMemCreate` / `hipMemMap`.
- **Symptom:** VMM reported supported (granularity 64 KiB), `hipMemcpy` into the mapped range worked,
  but kernel writes into pages backed by the second physical handle were lost; some runs faulted the GPU.
- **Root cause (our reading):** WDDM residency makes resident only the allocation referenced by the
  kernel arguments.
- **Fix:** do not rely on HIP VMM on Windows; use an explicit page table (WHIRL: 256-token KV pages,
  one lookup per key tile, 0.3–0.5% decode cost).
- **Detection now:** the paged path is verified bit-identical to a contiguous cache.

### <a id="hip-5"></a>HIP-5 · Two big processes on one GPU: WDDM demotes both to shared memory
- **Conditions:** R9700 32 GB, two model processes of 15+ GiB each (twice: once a test server started
  during a needle test, once a check started while a previous benchmark was still running).
- **Symptom:** no allocation failure; both processes became extremely slow for ~20 minutes; per-process
  counters showed large Shared Usage (one: 7.2 of 11.6 GiB shared). All numbers from that window were
  invalid.
- **Root cause:** WDDM over-commits VRAM and pages parts of both processes to system memory.
- **Fix:** (1) a named mutex in the executable, `Local\whirl-gpu-<device>`, released by Windows on exit or
  crash; a second instance waits with a message every 30 s and gives up after a timeout. (2) A file lock
  (share mode 0) in every launcher, because old binaries lack the mutex; after acquiring it, also wait
  until no engine process remains. (3) Confirm the previous background job has ended before starting the
  next. (4) Kill only PIDs you recorded.
- **Detection now:** the memory monitor flags any two engine processes overlapping on one adapter.

### <a id="hip-6"></a>HIP-6 · Pinned host memory is counted as "Shared Usage"
- **Conditions:** Windows per-process GPU memory counters; `hipHostMalloc`.
- **Symptom:** our VRAM-overflow alarm fired (1,916 / 771 / 1,234 MiB shared) for experiments that put the
  token embedding or checkpoints in pinned host memory; dedicated usage was unchanged (31.28–31.31 GiB).
- **Root cause:** pinned allocations are charged to the GPU's shared segment.
- **Fix:** declare pinned sizes (the server writes them to a file) and subtract them before applying the
  256 MiB rule. In llama.cpp set `GGML_CUDA_NO_PINNED=1` to keep ~1 GB of host buffers out of shared
  usage.
- **Detection now:** monitor reports shared = declared tier + 35–55 MiB for tier-enabled servers.

### <a id="hip-7"></a>HIP-7 · `hipEvent` timing on the null stream reads low
- **Conditions:** early kernel timing with events on the null stream.
- **Symptom:** kernel times consistently too small.
- **Fix:** host wall clock around synchronized regions, or events on an explicit stream with enough work
  between them.
- **Detection now:** speed claims come from wall-clock end-to-end runs only.

### <a id="hip-8"></a>HIP-8 · Event-per-op profiling destroys what it measures
- **Symptom:** decode 34 → 14 tok/s with the profiler on; a server decode cycle 44 → 74 ms; MTP drafting
  time attributed to the embedding op. Even 3 events per cycle cost ~2.5% on the MoE model.
- **Fix:** use profiling for proportions in prefill only; use knockout builds for decode; never profile
  during A/B.
- **Detection now:** A/B scripts refuse profiling flags.

### <a id="hip-9"></a>HIP-9 · Task Manager shows no compute activity
- **Symptom:** the "Compute" graph is flat while the GPU is saturated; a VRAM sawtooth looked like a leak.
- **Root cause:** HIP work is not shown there; the sawtooth was the model reloading per prompt.
- **Fix:** read AMD Software's utilization/power or tok/s.

### <a id="hip-10"></a>HIP-10 · A synchronous `hipMemcpy` drains the whole queue
- **Conditions:** uploading a few token ids before each MTP draft step.
- **Symptom:** GPU idle four times per cycle.
- **Root cause:** a synchronous H2D copy waits for all queued work.
- **Fix:** pass small per-step data as kernel arguments (by-value structs); let kernels write what the next
  step needs (argmax writes the next token/position on the device). +5% end to end.
- **Detection now:** decode cycles have exactly one host synchronization.

### <a id="hip-11"></a>HIP-11 · `hipDeviceSynchronize` waits for background copies
- **Conditions:** non-blocking tier stream copying KV to host while the main stream decodes.
- **Measured:** background copies do not block `hipStreamSynchronize(main)` (0.00 ms) or a small null-stream
  `hipMemcpy` (0.57 ms with 6000 copies queued); `hipDeviceSynchronize` waits for all of them.
- **Fix:** no device-wide syncs in a server that has background transfers; synchronize streams.

### <a id="hip-12"></a>HIP-12 · Zero-copy writes to pinned memory never land **[eGPU]**
- **Conditions:** R9700 over USB4.
- **Symptom:** kernels writing directly to `hipHostMalloc` memory produced no data on the host.
- **Fix:** `hipMemcpyAsync` on a non-blocking stream. Unverified on direct PCIe.
- **Scope:** this is about kernel **writes** to host memory. Kernel **reads** of pinned host memory are
  used for the token embedding (`WHIRL_EMBD_HOST`, on by default in the server since 0.1.3); outputs are
  bitwise identical to keeping it in VRAM.

### <a id="hip-13"></a>HIP-13 · Grid y/z limit → `HipFailed`
- **Symptom:** a requantization kernel over the 248,320 output-head rows returned `HipFailed`.
- **Root cause:** grid y and z are limited to 65,536 blocks; the rows were on y.
- **Fix:** put the large dimension on x.

### <a id="hip-14"></a>HIP-14 · Probing for an optional kernel leaves a sticky error
- **Symptom:** an unrelated later call "fails" when the code checks `hipGetLastError`.
- **Root cause:** `hipModuleGetFunction` for a kernel absent from the loaded code object sets the last
  error.
- **Fix:** an optional lookup that returns null and clears the last error; required lookups fail loudly.
- **Detection now:** smoke test calls the optional lookup on a missing name and checks the error state.

### <a id="hip-15"></a>HIP-15 · Module launches do not check arguments
- **Symptom:** a benchmark mode crashed both GPUs' processes with `0xC0000005`; a self-check printed
  `-nan`.
- **Root cause:** a call site passed one argument too few after a kernel gained a KV-base parameter (the
  kernel read garbage as a pointer); separately, a self-check allocated an input of the MoE expert width
  (512) for a 2048-column matrix.
- **Fix:** generate launch argument packs from one definition; size test buffers from the maximum matrix
  dimension; exercise benchmark-only paths after signature changes.

### <a id="hip-16"></a>HIP-16 · Resource-usage reports lie unless flags match the build
- **Symptom:** `-Rpass-analysis=kernel-resource-usage` showed a "Dynamic Stack", symbolic VGPR counts and
  spills that the real build did not have.
- **Fix:** run it with exactly the production flags, or read VGPR/SGPR/scratch from the built code object's
  metadata.

### <a id="hip-17"></a>HIP-17 · `GPU_MAX_HW_QUEUES=1` does nothing on Windows
- **Conditions:** advice from Linux/ROCm, where decode steps alternated ~28 ↔ 36 ms on this card.
- **Measured:** fresh process per run, interleaved, 8 pairs: 28.0 ms/token in all 16 runs with and without
  it (spread 0.2%); MTP, server-restart and 8060S runs equally unchanged.
- **Root cause:** the bimodality is a KFD hardware-queue issue; Windows schedules through PAL/WDDM.
- **Fix:** don't port Linux ROCm tuning variables without measuring.

### <a id="hip-18"></a>HIP-18 · HIP graphs give nothing here
- **Measured:** 28.87 vs 28.81 ms/token; later 27.81 vs 27.79 ms. Host enqueue of 600–740 launches
  (0.4–0.7 ms) is already hidden behind GPU execution.
- **Fix:** keep the graph path optional; reduce launches by fusing same-input GEMVs instead (−1.2…−2.0%).

### <a id="hip-19"></a>HIP-19 · Device numbering in a two-GPU machine
- **Facts:** ROCm/HIP: `ROCm0` = 8060S, `ROCm1` = R9700; with `HIP_VISIBLE_DEVICES=1` the R9700 becomes
  `ROCm0`. Vulkan: `Vulkan0` = 8060S, `Vulkan1` = R9700. Vulkan reports warp size 64; HIP waves on RDNA
  are 32.
- **Fix:** select explicitly by name/architecture and print the selection at startup.

### <a id="hip-20"></a>HIP-20 · `hipMemGetInfo` is not WDDM's accounting; late allocations overflow
- **Symptom:** MoE server processes showed 368–399 MiB shared after sizing the KV pool to "all remaining
  memory"; creating a stream + pinned arena cost 12.8 MiB in `hipMemGetInfo` but ~60 MiB in WDDM's
  dedicated counter.
- **Root cause:** ~0.5 GiB allocated lazily after pool sizing (source not identified); driver-side
  allocations invisible to `hipMemGetInfo`.
- **Fix:** create every stream and auxiliary buffer before sizing pools; keep a per-model reserve (dense
  768 MiB, MoE 1.5 GiB).
- **Detection now:** worst-case VRAM tests (4 concurrent long requests filling 95% of the pool) under the
  memory monitor.

### <a id="hip-21"></a>HIP-21 · The GPU's LUID changes after a reboot
- **Symptom:** the memory report printed "all runs in dedicated VRAM" — for zero samples.
- **Root cause:** it filtered for the old adapter LUID.
- **Fix:** fail when no samples match and print the LUIDs seen.

### <a id="hip-22"></a>HIP-22 · PID reuse creates false memory alarms
- **Symptom:** a server flagged with 9,256 MiB shared although it declared 9,216 MiB of pinned tier.
- **Root cause:** declaration files named by PID were overwritten by a later process with the same PID.
- **Fix:** key by PID + start time; split monitor samples when a PID disappears for > 20 s.

### <a id="hip-23"></a>HIP-23 · Pinned allocation is slow
- **Measured:** 16 × 512 MiB pinned in 1.7–1.8 s.
- **Fix:** allocate one arena at startup (+1.6 s server start) and manage 2 MiB blocks; never pin on the
  request path.

### <a id="hip-24"></a><a id="hip-256k"></a>HIP-24 · `HipFailed` during a 256k prefill, with display re-enumeration (unresolved)
- **Conditions:** R9700, 27B, q8h KV, 262,400-token pool, one prefill of ~262k tokens.
- **Symptom:** `HipFailed` mid-prefill; Windows logged a Win32k display re-enumeration at the same time; the
  next load also failed; later the same prompt succeeded (prefill 398 tok/s, needle found).
- **Root cause:** unknown (suspected GPU reset).
- **Mitigation:** prefill attention beyond 128k context is split into launches by head range (≈2% cost,
  identical results; never applied ≤ 128k). If it recurs: rerun with `AMD_LOG_LEVEL=1`.

### <a id="hip-25"></a>HIP-25 · Running a probe on the GPU being measured
- **Symptom:** a benchmark run produced no decode line; at the same time a VMM probe on the same card had
  faulted the GPU.
- **Fix:** nothing else on a GPU while it is timed — including "tiny" probes.

### <a id="hip-26"></a>HIP-26 · The two GPUs are not independent (APU) **[8060S]**
- **Symptom:** 8060S 27B plain decode dropped 11–18% and prefill ~20% during a benchmark.
- **Root cause:** long-context jobs on the R9700 loaded the CPU, which shares power and cooling with the
  8060S.
- **Fix:** time the 8060S only when the machine is otherwise idle. Confirmed by a clean interleaved A/B
  (+0.7%, not a regression).

---

## <a id="tool"></a>2. Toolchain and shell

### <a id="tool-1"></a>TOOL-1 · Windows PowerShell 5.1 misreads UTF-8 scripts
- **Symptom:** a `.ps1` containing Chinese text fails to parse.
- **Root cause:** without a BOM, 5.1 reads the file in the ANSI code page.
- **Fix:** run scripts with PowerShell 7 (`pwsh`), or save with a BOM.

### <a id="tool-2"></a>TOOL-2 · cp950 console kills Python test harnesses
- **Symptom:** `UnicodeEncodeError` when a test prints ☔ or é; a whole gate run aborted.
- **Fix:** `PYTHONIOENCODING=utf-8` in every launcher.

### <a id="tool-3"></a>TOOL-3 · Shell heredocs and inline strings rewrite escapes
- **Symptom:** generated source with `\x..`, `\u....` or `\\n` fails to compile; backslashes disappear;
  paths lose separators (`C:\a\b` → `C:ab`) when passed through Bash.
- **Fix:** write patch and generator scripts as files and run them; run Windows-path-heavy pipelines from
  PowerShell.

### <a id="tool-4"></a>TOOL-4 · Code generators with non-raw Python strings
- **Symptom:** a `\n` inside a generated string literal became a real newline; in raw strings `\\` became
  two backslashes.
- **Fix:** generate from template files; compile-test generator output.

### <a id="tool-5"></a>TOOL-5 · Git Bash `sed -i` converts CRLF files to LF
- **Symptom:** one-line edits rewrote every line ending of a CRLF source file.
- **Fix:** edit with tools that preserve line endings (read/write with `newline=''`).

### <a id="tool-6"></a>TOOL-6 · `bash` from Python is WSL's bash
- **Symptom:** a Python harness calling `bash` ran WSL's `bash.exe` in a different environment.
- **Fix:** never shell out to `bash` from Python on Windows; write the harness in Python.

### <a id="tool-7"></a>TOOL-7 · `$!` in Git Bash is not a Windows PID
- **Fix:** start background processes with `Start-Process -PassThru` and record the Windows PID.

### <a id="tool-8"></a>TOOL-8 · Non-terminating download errors deleted an install
- **Symptom:** a working llama.cpp install directory was emptied.
- **Root cause:** `Invoke-WebRequest` failed without stopping the script; the next step cleared the target.
- **Fix:** verify the archive (size/hash/unzip test) before touching the destination.

### <a id="tool-9"></a>TOOL-9 · Slow, dropping release downloads
- **Measured:** 130–176 KB/s with disconnects for a 257 MB package.
- **Fix:** `curl -C -` with retries.

### <a id="tool-10"></a>TOOL-10 · A half-applied patch
- **Symptom:** kernels changed but host code did not (the patch script failed mid-way).
- **Fix:** patch scripts check every anchor string before writing anything and are re-runnable.

### <a id="tool-11"></a>TOOL-11 · Stale outputs read as success
- **Symptom:** a numeric check printed "ok" although the binary never ran (wrong path).
- **Root cause:** the check read output files left by the previous run.
- **Fix:** delete expected outputs before each run; check exit codes.

### <a id="tool-12"></a>TOOL-12 · A failed build did not stop the pipeline
- **Symptom:** a background test suite ran entirely on the previous binary after a compile error
  (a variable-shadowing error); every number was void.
- **Fix:** abort on build failure; record and verify the binary's hash in every result file.

### <a id="tool-13"></a>TOOL-13 · Renamed settings are silently ignored by old binaries
- **Symptom:** A/B runs against older candidate binaries ignored the new environment-variable names.
- **Fix:** baselines for A/B must be rebuilt (or aliased) whenever setting names change.

### <a id="tool-14"></a>TOOL-14 · Raw control bytes in source literals
- **Symptom:** a string literal containing a raw NUL byte did not compile.
- **Fix:** emit escapes (`\x00`) in generated tables.

### <a id="tool-15"></a>TOOL-15 · A source edited during a device compile left a stale code object
- **Conditions:** Ninja, a HIP device compile that takes ~2 minutes, a kernel source edited while it
  was running.
- **Symptom:** a marker kernel added to the gfx1151 set was missing from the embedded code object:
  the capability probe reported it absent and dozens of kernel-test checks failed with nonsense
  values (the CPU reference decoded the wrong scale-word format).
- **Root cause:** the compiler read the old source; the output it wrote is newer than the edit, so
  Ninja considers it up to date and never rebuilds it.
- **Fix:** do not edit sources a running build reads; after doing so, touch the file again. Check
  the symbol table of the code object (`llvm-objdump -t`) when a new kernel "is not found".
- **Detection now:** `whirl-kernel-test` prints the probed capability flags first.

---

## <a id="kern"></a>3. Kernels and numerics

### <a id="kern-1"></a>KERN-1 · Refactoring changed fma contraction and the logits
- **Conditions:** fusing two decode kernels; clang for gfx1201/gfx1151.
- **Symptom:** after moving `h0*w0 + h1*w1 + h2*w2 + x*w3` into an inline function, last-token logits
  differed by up to 0.17 (not bit-identical).
- **Root cause:** the compiler chose a different fma contraction order.
- **Fix:** write the canonical order as an explicit `__builtin_fmaf` chain (the order the original
  compiled to) or use `#pragma clang fp contract(off)`.
- **Detection now:** fused vs unfused outputs compared bit for bit (batched, step-1, step-2, 1,107-token
  prefill) on both GPUs; end-to-end logits compared across builds.

### <a id="kern-2"></a>KERN-2 · `x*x` contracted into a reduction; multiply merged with conversion
- **Symptom:** 1-ulp differences in fused RMSNorm / gated-norm producers.
- **Root cause:** clang contracted the square into the first butterfly add (fma) and merged a multiply
  with the f16 conversion into `v_fma_mix`.
- **Fix:** `asm volatile("" : "+v"(v))` after squares and final products; reproduce the reference
  reduction's lane permutation.
- **Detection now:** a probe compares fused kernels byte for byte with the originals; logits on/off in the
  same binary.

### <a id="kern-3"></a>KERN-3 · The same contraction in a recurrent-step rewrite
- **Symptom:** the rewritten DeltaNet step kernel was not bit-identical.
- **Root cause:** `y*y` contracted into the shuffle-add fma.
- **Fix/Detection:** asm barrier; 16-row teacher-forced logits dumps compared bit for bit.

### <a id="kern-4"></a>KERN-4 · Faster dequantization made prefill much slower
- **Symptom:** the most common tile configuration spilled 552 bytes/lane to scratch.
- **Fix:** compile every candidate configuration, parse resource usage, keep only spill-free ones (24
  survived).
- **Detection now:** resource usage checked whenever a kernel template changes.

### <a id="kern-5"></a>KERN-5 · Device lambdas capturing arrays → scratch
- **Fix:** macros instead of lambdas for fetch helpers.

### <a id="kern-6"></a>KERN-6 · HIP `int4` arrays live in scratch
- **Symptom:** an int8 GEMM probe at 52.5 TOPS; its prefetch buffer was in scratch.
- **Root cause:** HIP's `int4` is a struct; arrays of it were not promoted to registers.
- **Fix:** `ext_vector_type` vectors → 95.0 TOPS.

### <a id="kern-7"></a>KERN-7 · Hoisted loop-invariant addresses: 700+ spills
- **Conditions:** the f16-WMMA DeltaNet scan.
- **Root cause:** the compiler hoisted 64 64-bit addresses (32 reads, 32 writes) out of the loop.
- **Fix:** make the lane offset opaque per iteration (`asm volatile("" : "+v"(off))`), split addresses
  into uniform base + 32-bit offset → 20 spills.

### <a id="kern-8"></a>KERN-8 · The scheduler preloaded all V fragments
- **Conditions:** gfx1151 prefill attention.
- **Symptom:** 256 VGPRs + 13 spilled, 56 B scratch; two restructurings still spilled 11–18.
- **Fix:** pin each accumulator after its WMMA → 253 VGPRs, 0 spill, +1.1% prefill, bit-identical.

### <a id="kern-9"></a>KERN-9 · Interleaved WMMA chains blew the register file
- **Conditions:** int8 WMMA mid-batch GEMV.
- **Symptom:** 256 VGPRs plus hundreds of bytes of spill.
- **Fix:** `readfirstlane` for wave-uniform indices, branch-free 16-byte header decode, `asm volatile`
  pins per unit accumulator.

### <a id="kern-10"></a>KERN-10 · A computed weight pointer changed codegen
- **Symptom:** after adding a grouped-launch ABI, multi-token GEMV VGPRs grew 30–90% and some variants
  spilled (q6_k v7 T=8: 0.92 → 1.97 ms).
- **Root cause:** whenever the weight pointer did not come directly from a kernel argument (`select`, or
  pointer + offset). `readfirstlane`, `__builtin_assume`, integer offsets did not help.
- **Fix:** separate "twin" entries for grouped launches; single launches keep the original entries.
- **Detection now:** code-object metadata compared (VGPR/SGPR/scratch) for every entry after GEMV changes.

### <a id="kern-11"></a>KERN-11 · A second caller stopped inlining
- **Symptom:** 248 VGPRs + scratch; verify n=3 30 → 32 ms.
- **Fix:** `__forceinline__` on shared implementations.

### <a id="kern-12"></a>KERN-12 · `break` in a pipelined loop disabled prefetching
- **Symptom:** prefetch depth 2–3 had no effect; ISA showed the wait counter zeroed every step.
- **Fix:** a main loop without `break` plus a separate tail.

### <a id="kern-13"></a>KERN-13 · Divergent scale branches serialized memory latency
- **Conditions:** Q3_K/IQ3_S 1-token GEMV (491/466 and 502/482 GB/s vs a 593–598 ceiling); the mid-batch
  GEMV's `j < 2` scale branch.
- **Root cause:** `kk < 8 ? sb8[kk] & 0xF : sb8[kk-8] >> 4` became a wave-divergent branch: load,
  branch, load, wait, load — two serialized latencies per super-block.
- **Fix:** branch-free byte index and shift; issue loads for 4 super-blocks before computing → +16…+25%.
- **Detection now:** per-type GB/s vs pure-read ceiling in the benchmark self-report.

### <a id="kern-14"></a>KERN-14 · The compiler would not select `v_mad_i32_i24`
- **Symptom:** `mul_lo_u32` or `mul24 + add3` in the int8 epilogue.
- **Fix:** inline asm (118.2 TOPS from 95.0 together with header pre-decode).

### <a id="kern-15"></a>KERN-15 · A wrong constant in the `v_perm` lookup table
- **Symptom:** IQ4 decode wrong for some codes; one table constant was `0xCD` instead of `0x98`.
- **Detection:** caught by the comparison against the numpy reference before release.

### <a id="kern-16"></a>KERN-16 · A cached int8 activation was not invalidated
- **Symptom:** stale data after the source buffer was rewritten.
- **Fix:** reset the cache key at every write site.

### <a id="kern-17"></a>KERN-17 · A 32-bit candidate mask overflowed at 48 candidates
- **Root cause:** shifting a 32-bit mask by ≥ 32 is undefined.
- **Fix:** 64-bit masks.

### <a id="kern-18"></a>KERN-18 · Autotuning measured launch overhead
- **Symptom:** the tuner favored the fused path.
- **Root cause:** a sync after every repetition.
- **Fix:** time a batch of repetitions with one final sync. Cold-cache tuning did not help.

### <a id="kern-19"></a>KERN-19 · Batch buckets tuned at the wrong size
- **Symptom:** 85-token prompts padded into 256-token tiles (FFN-up 0.791 ms vs 0.271 ms with a smaller
  tile); the server's 1024-row chunks used configurations tuned at 512.
- **Fix:** more buckets (11), each tuned at its own size; 85–89-token prefill −29%; server 8k cold prefill
  +9.3% from the 1024 bucket alone.
- **Detection now:** the invariance check covers every option (tile choice never changes bits).

### <a id="kern-20"></a>KERN-20 · LDS above ~41 KB per workgroup cut occupancy
- **Measured:** 16 → 6 waves per SIMD, slower than the smaller-LDS version.

### <a id="kern-20b"></a>KERN-20b · HIP's occupancy numbers assume half of RDNA 4's LDS
- **Symptom:** `hipDeviceProp_t` (`sharedMemPerMultiprocessor`) and
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` predicted one block per WGP for every block using
  more than 32 KiB of LDS, so kernels were sized to stay under 32 KiB.
- **Root cause:** a gfx12 WGP has **128 KiB** of LDS (one workgroup can still use at most 64 KiB);
  the HIP runtime reports and computes with 64 KiB, so its LDS-limited occupancy is half the real one.
- **Measured:** blocks of 36–64 KiB LDS run **two per WGP** at the same time. Registers decide the rest:
  up to 236 VGPRs a SIMD still holds 6 waves; at 256 VGPRs it drops to 5.
- **Fix:** do not trust the occupancy API for LDS on gfx12; measure (two blocks in flight per WGP) or
  compute from 128 KiB per WGP / 64 KiB per workgroup. KERN-20's "~41 KB cliff" was one kernel's
  register-plus-LDS combination, not a general LDS limit.

### <a id="kern-21"></a>KERN-21 · "Read KV once for all queries" was slower
- **Measured:** 24k MTP 39.5 → 35.6 tok/s.
- **Root cause:** the bottleneck was the reductions, not KV reads; L2/Infinity Cache already served the
  repeated reads.

### <a id="kern-22"></a>KERN-22 · gfx11 WMMA is inexact in one case
- **Conditions:** gfx1151 grouped verify attention.
- **Symptom:** grouped query columns not bit-identical to single columns.
- **Root cause:** a P = 0 entry multiplying another query's real V row did not yield an exact zero
  contribution.
- **Fix:** one query per group on gfx1151.

### <a id="kern-23"></a>KERN-23 · gfx11 WMMA duplicates B, so we decoded twice
- **Fix:** each half-wave decodes its half, exchange with `v_permlanex16` → IQ4_XS T=3 −8…−9%,
  bit-identical.

### <a id="kern-24"></a>KERN-24 · Exact int8 prefill cannot beat f16 on gfx1201
- **Measured:** W8A8 exact 96.5 TOPS, lossy 108.8, exact W4A8 best 126.1 vs f16 108–116 TFLOPS; only
  +6.6…+10.4% on FFN-up, worse on down-projection; 35× larger GEMM error.
- **Root cause:** a per-sub-block integer scale (`v_mad_i32_i24` per element) caps iu8 WMMA at ~205 TOPS;
  only 4–5 VALU per WMMA overlap for free.
- **Rule:** int8 GEMM pays off only without sub-block scales (or with power-of-two scales folded into fp8,
  as for MXFP4). llama.cpp's Vulkan int8 path reached the same conclusion on this card.

### <a id="kern-25"></a>KERN-25 · Fusing dequant into the GEMM made it slower
- **Measured:** bit-identical but −10…−13% on Q4_K_M prefill: decode VALU on the GEMM's critical path.
- **Rule:** dequantize-then-GEMM wins at large batch; measure before fusing.

### <a id="kern-26"></a>KERN-26 · Double-buffered LDS and direct-to-register weights lost
- **Measured:** fp8 GEMM: direct A fragments 183–189 TOPS, double-buffered 158–173, vs single buffer +
  register prefetch 194–210 (before the fragment-tiled rewrite). Same result for int8 (v4 110 vs v3 126).

### <a id="kern-27"></a>KERN-27 · Prefill attention is sensitive to code layout
- **Measured:** skipping the causal mask on off-diagonal tiles: −4% at 30k, +1.3% at 120k; skipping only
  the mask: +8% (worse); pinning the PV loop cut scratch 252 → 28 B/lane but was 5% slower.
- **Rule:** measure every change at several context lengths.

### <a id="kern-28"></a>KERN-28 · Half the KV bytes, slower decode
- **Measured:** q8h decode +1.0% / +3.6% / +5.2% ms/token at 16k / 64k / 128k; prefill attention +49%.
- **Root cause:** dequantizing K in the split-K kernel costs more than the bandwidth saved.
- **Fix:** q8v (K f16, V int8) decodes as fast as f16 ([kv-and-caching.md](kv-and-caching.md#formats)).

### <a id="kern-29"></a>KERN-29 · MoE routing near-ties make KL thresholds meaningless unless measured
- **Symptom:** a long-context MoE check failed (KL 3.84e-3 vs threshold 1e-3) on the 8060S.
- **Evidence:** expert ids dumped per (token, layer): between any two valid paths, 17–28% of pairs chose a
  different expert set from layers 0–4 on; KL 1.2e-4 … 3.4e-2 across 45 path pairs, top-1 equal in all.
  Dense 27B: KL ~1e-7.
- **Fix:** MoE long-context threshold 1e-2 with top-1 equality and "every token with p ≥ 1e-3 in both
  top-10s"; reasons written in the script. A real bug shows as a top-1 change or KL ≫ 0.1.

### <a id="kern-30"></a>KERN-30 · Output precision chosen by batch bucket broke solo == batched
- **Conditions:** opt-in f16-output paths.
- **Root cause:** whether a GEMM wrote f16 depended on the tune bucket of n; a solo chunk and a segmented
  forward fell into different buckets.
- **Fix:** enable f16 output only if every bucket's configuration supports it.
- **Detection now:** segmented-prefill gate in the relaxed mode too.

### <a id="kern-31"></a>KERN-31 · Two forked kernel sets drifted apart in one argument **[8060S]**
- **Conditions:** gfx1201 and gfx1151 kernel sets kept as separate sources with the same kernel
  names; the host shares one launch path.
- **Symptom:** none in short tests. The kernel test's "head-split launches == one launch" check
  failed on gfx1151 (half the heads wrong).
- **Root cause:** the gfx1201 `attn_prefill_wmma*` had gained an `h0` (first head) argument for
  long prefills split over head ranges; the gfx1151 copy had not. The host passed `h0`, the kernel
  ignored it, so every range after the first recomputed heads 0.. — only for prompts long enough to
  split (n x context > 4096 x 128k).
- **Fix:** added `h0` to the gfx1151 kernel. Kernel argument layouts of both code objects are now
  compared from the AMDGPU metadata (`.args`: offset, size, kind) for every kernel present in both.
- **Detection now:** the ABI comparison (0 differing kernels) and the head-split invariance check in
  `whirl-kernel-test`, which runs on each GPU.

### <a id="kern-32"></a>KERN-32 · The int8 scale words have a per-architecture format **[8060S]**
- **Symptom:** on gfx1151 every int8-activation check failed: `quantize_q8 xd` differed in all
  words, the GEMV references were off by 1e37.
- **Root cause:** the gfx1151 kernels store each 32-value block's scale as (f32 scale rounded to an
  11-bit mantissa, block sum + 4096) in one 32-bit word, so the dot kernels get the block sum for
  free; the CPU references assumed a plain f32 scale.
- **Fix:** the code object announces the format with a marker kernel (`kernels::Caps::xd_sum`); the
  references pack and unpack accordingly (`ref::setXdSum`, `ref::xdScale`).
- **Detection now:** exact `quantize_q8` / fused-quantization checks on both GPUs.

---

## <a id="mtp"></a>4. Speculative decoding exactness

### <a id="mtp-1"></a>MTP-1 · One ulp in the multi-token GEMV broke MTP == greedy
- **Symptom:** after a multi-token dot-product rewrite, MTP output no longer matched plain greedy output.
- **Root cause:** verify logits differed from decode logits in the last bits; near-tied tokens flipped.
- **Fix:** the 1-token kernel and all multi-row kernels share the canonical per-unit expression and the
  same xor reduction tree.
- **Detection now:** `checkGemvBitwise` (all types × 2–16 rows × all variants × the output head);
  MTP == plain greedy for draft counts 1, 2, 5, 10, with n-gram forced, in CLI, server and concurrency
  tests.

### <a id="mtp-2"></a>MTP-2 · F32 small matrices take different kernels at n = 1 and n ≥ 2
- **Conditions:** a GGUF storing `ssm_alpha` / `ssm_beta` as F32.
- **Symptom:** MTP ≠ plain greedy on all three quantized variants; drafts capped at 1.
- **Root cause:** F32 weights: f32-activation GEMV for one row, f16 GEMM for verify.
- **Fix:** Q8_0 for those tensors (fused exact path; still recommended). Engine fix landed in 0.1.3:
  F32 / F16 small matrices now use one GEMV family for all n, so output is exact, but drafts stay
  capped at 1 because the fused DeltaNet decode (`gdn_ab`) exists only for Q8_0 / MXFP4 α/β
  ([speculative-decoding.md](speculative-decoding.md#exact)).
- **Detection now:** MTP smoke test (plain == MTP == MTP + n-gram == forced n-gram) on every new model file.

### <a id="mtp-3"></a>MTP-3 · Drafts accepted past end-of-sequence
- **Symptom:** next-turn cache misses (`cache now 42`, next common prefix 41).
- **Root cause:** drafts after `<|im_end|>` were consumed into state and cached tokens.
- **Fix:** cut accepted drafts before EOS; same for `max_tokens`. Output unchanged.
- **Detection now:** multi-turn cache-hit test requires each turn to cache ≥ previous prompt − 1.

### <a id="mtp-4"></a>MTP-4 · A timing table carried across requests
- **Symptom:** second and later requests from one user drafted too much (~4.5 vs ~3.3 per cycle on the
  8060S); server 3.6% slower than the CLI on the R9700.
- **Fix:** reset the cost model's timing table when a request starts on an idle engine (+4.1% for
  requests 2–4; server vs CLI −0.5%).
- **Detection now:** server-vs-CLI speed check with before/after CLI references.

### <a id="mtp-5"></a>MTP-5 · Buffers sized for the MTP draft limit overflowed with n-gram drafts
- **Root cause:** arrays sized for 10 drafts; n-gram allowed more.
- **Fix:** size everything for the maximum verify batch (16 rows).

### <a id="mtp-6"></a>MTP-6 · A naive n-gram drafter only made things slower
- **Measured:** edit task 128.4 → 111.5 tok/s (−13%).
- **Root cause:** used whenever a match existed; only the latest occurrence of a 3-gram (common code
  3-grams point to the wrong place); capped at 8.
- **Fix:** 3- and 12-token keys, longest backward match, counterfactual scoring, cost-model decision,
  15 drafts → edits +48…+99%.

### <a id="mtp-7"></a>MTP-7 · A one-off first-use cost poisoned the timing model
- **Root cause:** the first 16-row verify costs ~50 ms extra once.
- **Fix:** ignore the first sample of each row count.

### <a id="mtp-8"></a>MTP-8 · Truncating the draft vocabulary killed Chinese acceptance
- **Measured:** 82% → 28–33% (Chinese tokens have high ids); later 2.58 → 2.43 tokens/cycle at 150k/100k.
- **Rule:** draft heads may be quantized (2-bit works) but not truncated.

### <a id="mtp-9"></a>MTP-9 · More drafts hurt the MoE model
- **Measured:** per-position acceptance 92 / 31 / 4 / 0%; 3 drafts slower than none at 24k.
- **Fix:** 1 draft for MoE by default.

### <a id="mtp-10"></a>MTP-10 · p-min with a host sync per draft cost 3%
- **Fix:** decide on the GPU (stop flag read by later draft kernels); one sync per cycle.

---

## <a id="srv"></a>5. Server and caching

### <a id="srv-1"></a>SRV-1 · `<think>\n` vs `<think>\n\n</think>`: every turn re-prefilled
- **Conditions:** thinking mode; clients that do not return `reasoning_content` (most agent clients).
- **Symptom:** multi-turn cache hit ratio 0.000; whole conversation prefilled every turn.
- **Root cause:** the prompt ended with `<think>\n`; the re-rendered history contains `\n\n`, a single
  different token — the `prompt-end` checkpoint missed by one token.
- **Fix:** split such prompts at N−1 and save a `think-open` checkpoint after `<think>` (CLI splits too,
  keeping server == CLI). Hit ratio 0.000 → 0.928 (agent) / 0.515 (chat).
- **Detection now:** multi-turn cache test with five session types, both models.

### <a id="srv-2"></a>SRV-2 · Generated tokens are not always the canonical tokenization
- **Symptom:** occasional mid-turn cache divergence (e.g. around `"""'` in code).
- **Decision:** accepted (one reply's prefill lost); the alternative makes cached and uncached inputs differ.

### <a id="srv-3"></a>SRV-3 · Cached vs uncached is not bit-identical
- **Root cause:** reused history KV came from decode kernels, re-prefilled history from prefill GEMMs.
- **Fix in tests:** when texts differ, a verify server re-prefills each cached turn and requires KL ≤ 1e-2
  and equal top-1.

### <a id="srv-4"></a><a id="srv-ngram-pages"></a>SRV-4 · Verify rows wrote into unmapped KV pages
- **Conditions:** n-gram drafts > 8 with extra snapshot capacity.
- **Symptom:** a multi-turn conversation's second turn differed.
- **Root cause:** slots guaranteed pages only to `pos + n_draft + 2`; extra verify rows hit page-table
  entry 0 → page 0 of the pool, another sequence's data.
- **Fix:** cap drafts by the slot's mapped pages.
- **Detection now:** agent benchmark (LF and CRLF) with off / on / on-with-more-snapshots, all six turns
  identical.

### <a id="srv-5"></a>SRV-5 · A running slot kept a dead session's pages (restore impossible)
- **Symptom:** a 126k-token restore never happened while other slots decoded; the request fell back to a
  117k-token prefill (~75 s; others dropped to 9–15 tok/s) and failed with `KvPoolFull` (HTTP 500).
- **Root cause:** a slot that got a short request reusing nothing kept its old 459 pages; running slots
  are not evictable; only 310 pages were free.
- **Fix:** return pages beyond `N + drafts + 2` at job start (quarantine / refcount aware).
- **Detection now:** `restore_conc_gate` (restore beside 3 decoding slots == never evicted).

### <a id="srv-6"></a><a id="srv-defaultenv"></a>SRV-6 · Garbage output only with the default configuration
- **Symptom:** one candidate passed every gate, yet a benchmark with the server's default settings
  produced 61 garbage outputs.
- **Root cause:** with automatic KV format, the server reloaded its kernel table after loading; that reset
  swapped function pointers (fp8 producers back to row-major) while a "tiled" flag stayed on — producers
  wrote one layout, the GEMM read another. Every gate set the KV format explicitly, so the reload never
  happened in tests.
- **Fix:** never mutate kernel tables for load-time decisions; choose per launch from state. The broken
  candidate was uninstalled immediately.
- **Detection now:** a permanent "default environment" phase: a server with no variables and no options
  must pick the same KV format as the CLI reference and produce identical greedy and seeded outputs.

### <a id="srv-7"></a>SRV-7 · Concurrent ≠ solo for MXFP4 (a path condition)
- **Symptom:** MXFP4 with 2–4 concurrent requests: greedy ≠ solo (first divergence at character 4 for one
  request — inside prefill).
- **Bisection:** off with segmented prefill disabled; off with two speed-mode features disabled; still
  present with only one disabled.
- **Root cause:** an f16-output DeltaNet path was conditioned on "no segments", so segmented prefill took
  the f32 path, solo the f16 one.
- **Fix:** segmented prefill uses the same path per segment.
- **Detection now:** segmented-prefill gate for MXFP4 + a C = 4 concurrency == solo check.

### <a id="srv-8"></a>SRV-8 · Waiting for a background spill stalled decoding
- **Symptom:** ~150 ms main-stream stalls when a new request hit a slot mid-spill (692 MiB spill ≈ 180 ms).
- **Fix:** copy-on-write of the boundary page + quarantine of old pages until the copy fence; 0 waits.

### <a id="srv-9"></a>SRV-9 · A slot stayed linked to the wrong tier entry
- **Symptom:** an unrelated session overwrote another session's RAM entry in place.
- **Fix:** unlink when a new request reuses nothing.
- **Detection now:** `tier_gate`.

### <a id="srv-10"></a>SRV-10 · Cross-stream ordering: spilled a checkpoint before it was written
- **Symptom:** RAM/SSD copies of a shared checkpoint held the pre-copy contents.
- **Fix:** the tier stream waits on an event recorded after the main-stream copies.
- **Related race fixed:** a spill-completion check skipped after eviction let a later checkpoint save
  overwrite a buffer the spill was still reading.

### <a id="srv-11"></a>SRV-11 · A deferred SSD write was never scheduled
- **Symptom:** after a restart, an older entry (28,356 instead of 28,414 tokens) was restored.
- **Root cause:** a spill deferred while the entry was being written to SSD left the idle loop asleep.
- **Fix:** report "busy" after deferring.

### <a id="srv-12"></a>SRV-12 · A literal `<|im_end|>` in a system prompt moved the cache boundary
- **Symptom:** a 30k-token system prompt was not reused; the server fell back to coarser `prefix`
  checkpoints.
- **Root cause:** the system text (tokenizer docs) contained `<|im_end|>`, parsed as the special token; the
  boundary search used the first `<|im_end|>`.
- **Fix:** use the three-token pattern `<|im_end|>\n<|im_start|>`; the gate's system prompt now contains a
  literal `<|im_end|>`.

### <a id="srv-13"></a>SRV-13 · Bigger prefill chunks made follow-ups slower
- **Symptom:** chunk 2048: cold prefill +18…+21%, but cross-slot follow-ups reused a checkpoint at 27,418
  instead of 28,442 (30k + 1000 tokens: 1548 → 2230 ms).
- **Fix:** schedule in 1024-token chunks, execute merged forwards of up to 2048 rows.

### <a id="srv-14"></a>SRV-14 · Checkpoints in pinned host memory: 96 small copies
- **Measured:** ~40 ms per save/load at ~3.75 GB/s; agent turns −25%.
- **Rule:** gather to contiguous buffers before host copies; keep hot checkpoints in VRAM.

### <a id="srv-15"></a>SRV-15 · Evicted session ≠ solo — by policy, not by bug
- **Symptom:** a multi-session test's "tier == solo" check failed after shared system checkpoints arrived.
- **Root cause:** entries gaining < 512 tokens over the system checkpoint are not restored; ~60
  decode-produced tokens are re-prefilled (numerically equivalent).
- **Evidence:** forced restores 28/28 == solo; a byte-verify mode found all 115 spills/restores identical.

### <a id="srv-16"></a>SRV-16 · Test servers shared one SSD cache directory
- **Symptom:** later tests restored entries written by earlier tests instead of prefilling.
- **Fix:** a fresh SSD directory per test server (the default-environment test keeps the real default).

### <a id="srv-17"></a>SRV-17 · Reference servers picked a different KV format
- **Symptom:** every comparison in a gate "failed", including requests with 0 cached tokens.
- **Root cause:** a reference server without caches had more free VRAM, so the auto rule chose f16; the
  tested server chose q8v. Happened in the system-prompt gate and again in the vision gate.
- **Fix:** pin the KV format in gates that compare servers.

### <a id="srv-18"></a>SRV-18 · Random tool-call ids broke equality checks
- **Fix:** compare tool calls by name and arguments.

### <a id="srv-19"></a>SRV-19 · The first request of a burst prefilled alone, then stalled
- **Symptom:** in a 4-request burst the first request decoded, then stopped for 2.4 s while the others
  prefilled together.
- **Fix:** 30 ms burst gathering for requests still being received or tokenized.

### <a id="srv-20"></a>SRV-20 · Parallel sub-agents recomputed the same system prompt
- **Fix:** queued requests wait for a system checkpoint that another request is producing.

---

## <a id="tok"></a>6. Tokenizer and templates

### <a id="tok-1"></a>TOK-1 · The Qwen whitespace rule `\s+(?!\S)` was not implemented
- **Conditions:** our own BPE pre-tokenizer for the `qwen35` pre-tokenizer type (also used by the MoE
  model).
- **Symptom:** nothing looked wrong — outputs were fluent. It surfaced while investigating why n-gram
  drafts copying a file were accepted only 25% of the time: prompt token counts differed from the
  reference (one coding prompt 152 → 156 tokens; a 4k-token source prompt 4068 → 4128).
- **Root cause:** the split regex is `…|\s*[\r\n]+|\s+(?!\S)|\s+`. The negative lookahead means a run of
  spaces followed by a word gives its *last* space to the word: `"    return"` must split as
  `"   " + " return"`, not `"    " + "return"`. Also `\s*[\r\n]+` must extend to the last newline of a run
  (`"\n  \n"` was split wrong), and `\v` counts as whitespace. Every indented code prompt was tokenized
  into a sequence the model had never seen in training. The model itself writes canonical tokens — so
  MTP drafts (from the model) copied fine, but n-gram drafts (from the prompt) did not match.
- **Fix:** implement the lookahead (back off one whitespace before a non-space) and the newline-run rule.
  Behavior change: indented prompts now produce different (correct) outputs. Agent-session cache hit
  ratio rose (0.972 → 0.978 dense, 0.963 → 0.975 MoE) because re-tokenized history now matches generated
  tokens.
- **Verification:** a reference tokenizer built from the GGUF vocabulary on tiktoken's fancy-regex with
  llama.cpp's `qwen35` pattern agreed token for token on source files, Chinese-heavy documents (33k
  tokens), a 44k-token source file, all benchmark prompts and whitespace edge strings.
- **Detection now:** the C++ tokenizer is checked against `llama-tokenize` on 440 file/model/mode
  combinations (~1.5 M tokens), with indentation edge cases (spaces, tabs, mixed, blank lines) in the
  corpus.

### <a id="tok-2"></a>TOK-2 · CRLF files in prompts vs LF output
- **Symptom:** n-gram matching failed on Windows files.
- **Root cause:** `"\r\n"` (id 317) never matches the model's `"\n"` (198).
- **Fix:** token-level CRLF→LF normalization table for the n-gram index (653 CR tokens in 5 files, all
  1:1; normalized sequence == LF file's tokens); drafts emit LF unless the model writes CR.

### <a id="tok-3"></a>TOK-3 · Unicode 15.1 vs 16.0
- **Symptom:** 5,185 code points classified differently from llama.cpp.
- **Root cause:** llama.cpp's tables are Unicode 15.1; Python 3.14's `unicodedata` is 16.0. All 5,185 are
  characters new in 16.0 (UNDEFINED in llama.cpp).
- **Fix:** generate tables at the 15.1 level by default (16.0 by option).
- **Detection now:** table verification against llama.cpp's `unicode-data.cpp` for all 0x110000 code
  points; a corpus case with 16.0-only characters.

### <a id="tok-4"></a>TOK-4 · Lead bytes F5–F7 decode above U+10FFFF
- **Symptom:** `llama-tokenize` aborts with an unhandled C++ exception on such input.
- **Fix (WHIRL):** raise a tokenizer error; the parity test counts both as the same error outcome.

### <a id="tok-5"></a>TOK-5 · Jinja `tojson` floats differ between engines
- **Facts:** llama.cpp's Jinja prints `2` for `2.0` and 6-digit precision; Python `json.dumps` prints `2.0`.
  WHIRL follows Python; documented as a known divergence.

### <a id="tok-6"></a>TOK-6 · `|trim` differs between Python Jinja and C++ engines
- **Facts:** Python's `str.strip()` also removes U+3000, U+00A0, U+2028, U+0085, U+001C–U+001F; llama.cpp
  and WHIRL trim ASCII whitespace only. Affects only content starting/ending with such characters.

---

## <a id="quant"></a>7. Model files and quantization

### <a id="quant-1"></a>QUANT-1 · The imatrix is ignored for MXFP4 and Q8_0
- **Fact:** mainline `llama-quantize` does not use quant weights for MXFP4 (`GGML_UNUSED(quant_weights)`)
  or Q8_0; only K-quant tensors (here, the output head) use it.

### <a id="quant-2"></a>QUANT-2 · No dense MXFP4 file type in llama.cpp b11214
- **Fix:** ftype `MXFP4_MOE` + `--tensor-type-file` (first match wins) + `--output-tensor-type`.
  Verify the result by dumping tensor types.

### <a id="quant-3"></a>QUANT-3 · Output-head type changes MTP speed more than plain speed
- **Measured:** Q8_0 head: highest acceptance (73.2%) but MTP 98.0 vs 117.0 tok/s (Q6_K head). Q4_K head:
  +1.9% without MTP but −4.5% with MTP (no Q6_K-specific draft/verify paths).
- **Rule:** prefer Q6_K heads for WHIRL.

### <a id="quant-4"></a>QUANT-4 · fp8 activations amplify every other perturbation
- **Measured:** the same DeltaNet change: KL 4.5e-8 … 1.5e-4 under f16 prefill, 1.2e-4 … 7.0e-2 under fp8.
- **Rule:** evaluate precision-sensitive changes in precision mode; record, don't gate, speed-mode KL.

(See also [MTP-2](#mtp-2): F32 `ssm_alpha` / `ssm_beta`.)

---

## <a id="vis"></a>8. Vision

### <a id="vis-1"></a>VIS-1 · Unwritten padding lanes made embeddings depend on server history
- **Symptom:** image embeddings in the server differed from the standalone encoder and varied with earlier
  traffic.
- **Root cause:** the Q/K prep kernel left padded head-dimension lanes 76–79 unwritten; inside the server,
  activations borrow the prefill scratch, which held old data. The standalone tool's fresh arena hid it.
- **Fix:** write the padding.
- **Detection now:** server embeddings == standalone, bit for bit, after arbitrary history. General rule:
  test with dirty buffers.

---

## <a id="meas"></a>9. Measurement

### <a id="meas-1"></a>MEAS-1 · Caches inflate microbenchmarks
- **Fix:** rotate > 512 MB (R9700, 64 MB Infinity Cache) or > 2 GB (8060S, 32 MB MALL) of real weights.

### <a id="meas-2"></a>MEAS-2 · Machine drift looks like gains and losses
- **Measured:** benchmark sessions differed by 2–6% with no code change; the MoE model showed a ±3.5%
  bimodal shift between time slots.
- **Fix:** interleaved A/B in one session, 2+ rounds, min–max.

### <a id="meas-3"></a>MEAS-3 · Laptop thermals: boost vs sustained **[8060S]**
- **Measured (ROG Flow Z13, ~80 W sustained):** prefill −15% after ~5–7 s; back-to-back processes: plain
  decode 13.4 → 5.4–5.8 tok/s (−58%) after 3–5 min, identically for old and new binaries.
- **Fix:** 60–90 s idle before each run, interleave, report boost vs sustained; mini-PC numbers are not
  comparable.

### <a id="meas-4"></a>MEAS-4 · A speed check compared a cold CLI with a hot server
- **Symptom:** server "−7…−14% slower" than the CLI on the 8060S.
- **Fix:** CLI reference measured before and after the server (median of 3 each), tolerance 5% + drift,
  15 s between server requests. The rewrite then exposed a real server issue ([MTP-4](#mtp-4)).

### <a id="meas-5"></a>MEAS-5 · Too-strict top-k agreement thresholds
- **Symptom:** the int8 two-token path occasionally swapped the 10th-ranked token.
- **Fix:** top-10 overlap ≥ 9 (with KL and top-1 unchanged); for MoE, a probability-floor rule
  ([KERN-29](#kern-29)).

### <a id="meas-6"></a>MEAS-6 · An insensitive QA benchmark
- **Symptom:** answer-only QA: 5–7% correct for both models — no power to detect anything.
- **Fix:** reasoning before `ANSWER:`, 350 items, McNemar on discordant pairs.

### <a id="meas-7"></a>MEAS-7 · Speculative speed across different outputs
- **Example:** a numerics change made an English summary 138 instead of 105 tokens; acceptance fell 2.76 →
  2.42 tokens/cycle while each cycle got faster (40.6 → 39.9 ms).
- **Fix:** compare ms/cycle and acceptance separately; only compare tok/s on identical outputs or
  multi-prompt means.

### <a id="meas-8"></a>MEAS-8 · Tools that break the VRAM rule
- **Facts:** `llama-perplexity` peaked at 306 MiB shared (speed invalid, values fine); llama.cpp without
  `GGML_CUDA_NO_PINNED=1` adds ~1 GB shared.

### <a id="meas-9"></a>MEAS-9 · Buffered I/O and queue depth hide SSD speed
- **Measured:** buffered write 1.8 GB/s vs unbuffered 5.1–5.3; 2 MiB reads QD1 4.1 vs QD4 7.0 GB/s.

### <a id="meas-10"></a>MEAS-10 · The reference engine can be broken
- **Facts:** llama.cpp b10686 Vulkan decoded at 5.6 tok/s on the R9700 (prefill normal); fixed in b11214.
  b11214 Vulkan with MTP at 14k/24k context dropped prefill from ~800 to ~97 tok/s. The ROCm build needed
  `--load-mode none` ([HIP-3](#hip-3)).
- **Rule:** sanity-check the reference before publishing a comparison.

### <a id="meas-11"></a>MEAS-11 · Agent benchmarks need bounded tool loops
- **Symptom:** in a configuration whose text diverged, the model looped `read_file` until the 128k limit
  and the benchmark crashed.
- **Fix:** cap tool calls per question in harnesses.

---

## <a id="egpu"></a>10. eGPU (USB4) — environment limitations

### <a id="egpu-1"></a>EGPU-1 · ~3.8 GB/s host link
- **Measured:** D2H 3.78–3.81, H2D 3.84–3.86 GB/s for any piece size; KV restore ~14–16 µs per token;
  128k restore ~1.9–2.0 s.
- **Status:** environment limit; direct PCIe expected to be much faster (not measured).

### <a id="egpu-2"></a>EGPU-2 · H2D traffic slows kernel dispatch
- **Measured:** 1500 small kernels 15 → 64 ms while H2D saturates the link (20 large kernels +4%); other
  slots' decode cycle +17% during a restore. `GPU_BLIT_ENGINE_TYPE` no effect; `PAL_DISABLE_SDMA=1` only
  moves the cost; VRAM staging + D2D scatter is slower (3.51 vs 3.69–3.83 GB/s).
- **Status:** cost proportional to bytes; pacing does not help; not optimized.

### <a id="egpu-3"></a>EGPU-3 · Vision weights stream per image
- **Measured:** ~260 ms H2D per image when the encoder weights are not resident (default 27B config).
- **Status:** environment-dependent; resident mode avoids it when VRAM allows.

---

Entry count: HIP 26 · TOOL 14 · KERN 30 · MTP 10 · SRV 20 · TOK 6 · QUANT 4 · VIS 1 · MEAS 11 ·
EGPU 3 — **125 entries**.
