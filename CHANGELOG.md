# Changelog

All notable changes to WHIRL are listed here. Versions follow `project(whirl VERSION ...)` in
`CMakeLists.txt`.

## 0.2.2.2 — 2026-10-21 (bug fixes)

### Fixed

- Server, R9700 and other discrete GPUs: when another program took more VRAM while the server ran, the server stayed
  over the Windows (WDDM) video memory budget until it was restarted, and decode stayed several times slower (0.2.2.1
  only warned; issues #5 and #6). A memory governor now acts on the samples the server already takes (each `batch |`
  status line, each request end, an idle sample; never in the decode loop). When less than half of the margin is free
  under the budget (about 1-1.5 GiB) it acts in this order, one `memory governor:` line per action with numbers: frees
  reusable cache (idle slots' prefix cache, which went to the RAM tier at request end; while a request still runs, also
  the checkpoint buffers of idle slots and shared prefix checkpoints, VRAM back to Windows at once); queues new requests
  while a smaller KV pool waits; lowers the context for new requests if the smaller pool holds less; re-allocates the KV
  pool smaller as soon as every slot is idle, which is what returns the VRAM to Windows; and only then warns. It only
  shrinks WHIRL's own VRAM: the RAM tier is pinned system RAM, not VRAM (17 GiB pinned left the R9700's budget unchanged
  in our test), and is never shrunk for the VRAM budget. When the budget has room again for 30 s, at least 60 s after
  the last drop (longer if the budget keeps going up and down), the pool grows back while idle: straight to its startup
  size once that fits with 3/4 of the margin free, which also gives new requests the startup context again; otherwise
  part way. A running request's KV is never moved, so a request that was running when the budget dropped stays slower
  until it finishes. `GET /health` and `GET /props` show the effective values in a `governor` object; `/props` `n_ctx`
  is the context for new requests. Output is unchanged: only how many tokens the pool holds changes. Not on the Radeon
  8060S (its pool is a fixed carve-out) and not with `WHIRL_POOL_RESERVE_MB`; `WHIRL_GOVERNOR=0` turns it off.
  The HIP runtime keeps freed GPU memory in a cache for reuse instead of giving it back to Windows (up to 4 GiB on the
  R9700), so a smaller KV pool still counted much of the old one. whirl-server now sets `GPU_RESOURCE_CACHE_SIZE=256`
  (MiB) for its own process unless you set it (`WHIRL_HIP_CACHE_LIMIT=0` keeps the runtime default): a 3.93 GiB shrink
  then lowered WHIRL's VRAM by 3.90 GiB, against 2.55 GiB with the default cache. Whether it also speeds up the recovery
  of decode is not established (see below). Because memory freed while loading now goes back too, the default KV pool on
  the R9700 is about 3% larger (Swift-1.5 27B, `--parallel 1`: 329728 -> 340480 tokens).
  Measured on the R9700 (another process takes 4 GiB while a long request runs, `--parallel 1`, balance; same prompt
  before and after, identical output every time; build before the 0.2.2.2 review fixes, whose pool grew back only to
  364032 of 393984 tokens): Ornith-1.5-35B-A3B MXFP4, 4 runs: 259-261 tok/s before, about 19 tok/s while the running
  request was over the budget. After the pool was re-allocated (393984 -> 200960 tokens) with the other process still
  holding its 4 GiB, decode came back to about 260 tok/s in 2 of 4 runs and stayed at 57.6 and 95-135 tok/s in the
  other two. After the other process exited and the pool grew back, 3 of 4 runs were back at 259-261 tok/s; one stayed
  at 160 tok/s (38% slower) for at least 5 minutes with no warning, cause unknown and under investigation. Windows moves
  WHIRL's paged-out buffers back on its own schedule. With the HIP runtime's default cache, 2 of 2 Ornith runs stayed at
  57 tok/s after the re-allocation. Swift-1.5 27B MXFP4, one run: 90 tok/s before, 16 tok/s during, 18.9 tok/s for the
  first request after the re-allocation and 89-91 tok/s from about 60 s later (with the default cache the first request
  was already fast in 2 of 2 runs).
- Server: the RAM tier cap of 0.2.2.1 ran after the half-of-available-RAM rule, so asking for more could give less
  (R9700, 47348 MiB available: `--kv-ram-mb 22528` -> 20480 MiB, `--kv-ram-mb 24064` -> 16384 MiB). The 20480 MiB cap
  now comes first, and a size above half of the available RAM goes to half of it (not to the default size, which was
  not monotonic either). The automatic size, which the user never chose, is capped with an info line instead of an
  `adjusted:` line, and the wording no longer quotes the HIP error: "capped at 20480 MiB while a crash with larger RAM
  tiers is investigated" (the cause is still open).
- Server, `--draft dflash` with `--parallel`: the re-fit of `--parallel` after loading did not count the DFlash2
  drafter's per-slot context ring, so it could keep a slot too many (the `-c` adjustment caught it later).
- CHANGELOG 0.2.2.1: the other process held 3.2 GiB (3308 MiB), not 3.3 GiB, and the "4 GiB test" is now described
  where it is cited.

### Known

- The close_reused server test still waits for log lines instead of an engine-thread hook; the hook (746a596) lands
  with the engine split (ARCH).
- Root cause of the crash with RAM tiers above about 20 GiB: still open (the cap stays).

## 0.2.2.1 — 2026-10-21 (bug fixes)

### Fixed

- Server, R9700 and other discrete GPUs: decode could get several times slower in a long session because the server
  ran over the Windows (WDDM) video memory budget (issue #5, reported by @CJure and @DanGTG; issue #6, reported by
  @Triazic). @CJure's startup log found it: the KV pool was sized against a budget of 31.02 GiB, Windows reported
  29.55 GiB about 0.2 s later, and the server started 0.37 GiB over it. Why the budget fell in that case is not known
  for certain. On our R9700 the same thing happens when another process holds VRAM: with 3.2 GiB taken by another
  process, the budget read 31.02 GiB when the pool was sized and 28.08 GiB right after the pool was allocated, so the
  budget seen at startup does not always count other programs yet. Over the budget, decode slowed from about 100-119
  to 10-43 tok/s in those tests, with "Shared GPU memory" unchanged (as in @CJure's report), and recovered when the
  other process exited. Changes:
  - The server reads, once at startup and before it loads anything, how much VRAM other processes use on its GPU and
    whether a monitor is attached to it (`other processes use X GiB of VRAM on this GPU (display attached: yes/no)`;
    about 0.2-0.3 s). The room kept free under the budget was 768 MiB (MoE 1.5 GiB). It is now what other processes
    use that the budget does not show yet, plus room for them to grow: the larger of 2 GiB (3 GiB with a monitor on
    the GPU) and 6% of the budget. `WHIRL_POOL_BUDGET_MARGIN_MB` sets the room. Cost on an R9700 with nothing else on
    it, balance: the KV pool of Swift-1.5 27B MXFP4 goes from 366592 to 330240 tokens with `--parallel 1` and from
    308736 to 272384 with the default 4 slots; Ornith-1.5-35B-A3B MXFP4 from 418816 to 395008 and from 381952 to
    358144. On the R9700 every request still gets its full 131072-token context. On cards with less than 20 GiB the extra
    1536 MiB desktop headroom is dropped, since the margin now covers it.
  - Right after the KV pool is allocated the server reads the budget again and, when its usage plus the margin no
    longer fits, frees the pool and allocates a smaller one (whole pages, before any request), logging
    `kv pool check: ...`; it waits for the WDDM numbers to settle after each free. In the 3.2 GiB test the pool was
    sized at 236544 tokens and fit the lowered budget with 2.5 GiB to spare.
  - Settings given on the command line that do not fit what the server measured are lowered to what fits instead of
    refusing to start or running over the budget, never silently: one `adjusted:` line each with the reason, a
    `startup adjustments` summary, and the effective values in a `startup` object in `GET /props` and `GET /health`.
    In order: `--parallel` (when one full request no longer fits next to every slot's state and checkpoints: from an
    estimate before loading, then again with the VRAM measured after the weights are loaded), `WHIRL_PREFILL_BATCH` /
    checkpoint counts, the RAM tier (`--kv-ram-mb` above half of the available RAM goes to the default size), `-c`,
    then `--ctx-per-slot`; `target not reached` in the summary when one full request still does not fit. The server
    refuses to start only when even one slot with a 4096-token pool does not fit. The Radeon 8060S keeps its
    carve-out check.
  - Not covered yet: when other programs take more VRAM than the room while the server runs, Windows can still move
    WHIRL's memory and decode slows down (on our R9700, when another process took 4 GiB more while a request
    ran, decode fell from about 100 to 10-43 tok/s). The server now says so
    in the log (below), and a restart sizes the pool for the VRAM free at that time. Adjusting the pool while running
    comes in 0.2.2.2.
  - Output is unchanged; only the pool, and so how much context can be cached at once, is smaller.
- Server: a RAM tier of the prefix cache above about 20 GiB could fail the first request with hipError 719 (R9700:
  `--kv-ram-mb 24064` failed, 20480 worked; 0.2.2 too). The automatic size is above 20 GiB on PCs with more than about 80 GB of
  RAM. Until the cause is found, every RAM tier, given or automatic, is capped at 20480 MiB, with an `adjusted:` line.
  Measured on the R9700 with 0.2.2.1: `--kv-ram-mb 22528` started with 20480 MiB (the cap), `--kv-ram-mb 24064` with
  16384 MiB (it was more than half of the RAM available then, so it went to the default size); both served requests.

### Added

- Server log: WHIRL's video memory and RAM while it runs, so a slowdown can be traced to the budget (issues #5, #6).
  Each `batch |` status line and each request end show
  `vram local X / budget Y GiB, shared Z MiB (+D MiB since start), ram R GiB (pinned P GiB)`. A plain-language
  `GPU memory spill` warning is printed when local usage goes over the budget or shared GPU memory grows by more than
  256 MiB since the pool was sized (not counting pinned host memory WHIRL allocated since): what happened, what it
  does to generation, what to do (close other programs that use the GPU, or restart with a larger
  `WHIRL_POOL_BUDGET_MARGIN_MB`), the KV pool size, how much VRAM other programs use now and at startup, and the
  numbers. It is also printed at startup when the server already starts over the budget, repeats at most once a
  minute while the spill lasts, and an info line says when it is over. While idle the server takes one sample a
  minute. A request whose decode gets more than 40% slower than its first 200 tokens gets one `decode slowed from X
  to Y tok/s` line; it names the spill only when one is seen and each decode cycle got slower, and with
  `--parallel 2` or more it says when another request's prefill shared the GPU. `GET /health` has a `vram` object
  with the last sample (`local_mib`, `budget_mib`, `shared_mib`, `spill`). One sample takes about 0.3 ms on the R9700
  (measured in the server: 262-321 us, including the HIP device query); it is taken every 5 s while a request runs,
  at request end and once a minute while idle, never per decode cycle. Model output is unchanged. See
  docs/guide/en/server.md, "Video memory while it runs".

### Changed

- The startup warning for a server over the budget, and the docs, recommended `WHIRL_POOL_RESERVE_MB`, which turns
  the budget-based sizing off. They now recommend `WHIRL_POOL_BUDGET_MARGIN_MB`, which keeps it and only leaves more
  room.

## 0.2.2 — 2026-10-20 (bug fixes)

### Fixed

- Radeon 8060S, fast mode (MoE models, q4 KV): decode was no faster than balance, in a coding chat 3.5% slower
  (Ornith-1.5-35B-A3B MXFP4), because `attn_dq4` always launched its fixed 256-split grid and the split combine walked
  256 partials per head even when only a few splits held keys (8 at 2k). It now launches only the splits the rows'
  keys use (the same key partition, so the output is bitwise unchanged). Measured on the Radeon 8060S at max power (80 W),
  ASUS ROG Flow Z13, Ornith fast decode, 0.2.1 vs 0.2.2, mean of 5 interleaved rounds: QA document 2k 94.8 -> 99.8
  tok/s (+5.2%), 96k 63.8 -> 66.7 (+4.6%), 128k 60.6 -> 62.7 (+3.4%), 32k 81.6 -> 82.9 (within the round-to-round
  range); coding chat 96.7 -> 99.6 (+3.0%, same text). Against balance (same rounds), fast is no longer behind:
  level in the coding chat (99.6 vs 100.2 tok/s; was 3.5% behind) and at 2k / 32k (within the round-to-round
  range), ahead at 96k / 128k (+12%; was +7% / +9%); it is not yet clearly faster at short lengths. Swift-1.5 27B MXFP4 (q8h KV in fast
  on this GPU) and the R9700 are unchanged. `WHIRL_DQ4_DYN=0` restores the fixed grid. FAST-8S-1.
- Server: a client's next request could hang for 21 s and fail to connect (Windows error 10060), rarely (seen in our server
  stress tests). The server closed every connection first, so each finished connection stayed in
  TIME_WAIT on the server side for minutes; when Windows handed the client the same local port again and the earlier
  request had been large (tens of KB, as agent prompts are), that TIME_WAIT silently dropped the new connection
  attempt. The server now lets the client close first: after a response whose end the client can see it waits up to
  200 ms for the client to close, and streamed (SSE) responses to HTTP/1.1 requests are now sent with `Transfer-Encoding: chunked` (as
  llama.cpp's server does) so clients see where a stream ends without waiting for the server to close. Clients that
  only stop at the server's close get it 200 ms later. Response bodies and model output are unchanged. HTTP/1.0 requests get their streams unframed, ended by the server's close, as before. HTTP-DROP.
- Radeon 8060S: `whirl devices` and the server log showed the HIP runtime's total (99.7 GiB on a 64 GB carve-out,
  which adds a share of system RAM) as the GPU's memory. They now show the dedicated memory set in Armoury Crate / the
  BIOS (the UMA carve-out) and the shared size, e.g. "dedicated 63.8 GiB (+ shared 47.6 GiB)". With a carve-out too
  small for the model, the server and the CLI stop before loading with a message that names the setting and the size
  to set, instead of failing with hipError 719 (GitHub issue #7); a carve-out that holds the model but not the default
  262,144-token KV cache gets a smaller cache. A hipError 719 on the 8060S now prints the same hint. Output is
  unchanged; R9700 output and memory sizing are unchanged. UMA-1.
- KV tier, SSD block dedup: when one entry's SSD write had just finished but the tier had not processed it yet,
  another entry sharing its prefix (a common system prompt) could start its write in that moment and store the
  shared blocks again instead of referencing them (more SSD space and write time; restored output was never
  affected). It now waits for the finished write like for one still running. Seen as the intermittent server gate
  failure "sys: SSD entries share the system prompt's KV blocks: 0 of 2" (1 of 24 R9700 runs). CB-FLAKE.
- Server: `WHIRL_PREFIX_CACHE_VERIFY=1` (the cache-hit diagnostic) no longer changes the KV tier fingerprint, so a
  diagnostic run restores the RAM / SSD entries a normal server wrote instead of ignoring them as another build's.
  Output unchanged. CACHEBAL.
- Release gate (`srv_cache_balance`): the tier check "turn 2 after a restart == no prefix cache" asserted equal greedy
  text for a gen-end reuse, which is numerically equivalent to a cold prefill but not bit-identical (decode-produced
  history; kv-and-caching.md section 5). With the 2026-10-19 Swift-1.5 27B MXFP4-A requant it failed on the R9700 in
  balance (also with the 0.2.0 exe; the SSD restore itself was bit-exact). The gate now checks the restored state
  against a from-scratch prefill (balance: KL <= 1e-2 and the same top-1, measured KL ~2e-4; precise: bitwise) and
  keeps the text check as a gate in precise. CACHEBAL.
- `--precise`: a conversation's next turn (and any cache hit) is now bitwise identical to a cold run, as the README
  promises. It used to resume at the end of the previous reply, whose history the decode kernels produced (R9700,
  Swift-1.5 27B MXFP4-A: max abs logit difference 0.016 to 0.029, KL 1e-8 to 7e-8 against a fresh prefill; also in
  0.2.0). precise now resumes only at checkpoints on the new prompt's chunk schedule that prefill produced, normally
  the previous prompt's tail checkpoint (kept for every prompt in precise), and re-prefills the rest (up to ~1,280
  tokens plus the previous reply). Cost, Swift-1.5 27B MXFP4-A, a 7k-token file and 3 follow-up questions (3
  interleaved rounds, the 0.2.2 exe, `WHIRL_COLD_RESUME=0` as the old behaviour): next-turn TTFT R9700 0.82-0.86 s
  instead of 0.06-0.30 s, Radeon 8060S at max power (80 W), ASUS ROG Flow Z13, 2.8-3.1 s instead of 0.17-0.19 s; first
  turn unchanged; texts identical. Balance and
  fast are unchanged. `WHIRL_COLD_RESUME=0/1` overrides. COLD-RESUME.
- Windows zip: it now also carries Microsoft's Visual C++ 2015-2022 x64 runtime (`msvcp140.dll`, `vcruntime140.dll`,
  `vcruntime140_1.dll`, 14.44.35211.0, unmodified, Visual Studio redistributable terms). `rocm_kpack.dll` (shipped
  since 0.2.1 for the Adrenalin Edition for ROCm 10.1 driver) and that driver's `amd_comgr_3.dll` import them, so on a
  PC without the Visual C++ Redistributable the GPU runtime would not load. WHIRL itself still links the C runtime
  statically; nothing needs to be installed besides the driver, with either driver branch.
- GPU pre-flight: when the AMD GPU runtime fails with Windows error 127 (a DLL it uses lacks a function), the message
  now names the likely cause, a DLL shipped next to `whirl.exe` that does not match the driver (for example a future
  driver that brings its own `rocm_kpack.dll`), and says to rename or remove that file in the WHIRL folder or use the
  WHIRL release that matches the driver. It used to tell the user to install the driver. The missing-DLL (126) message
  for the Visual C++ runtime now points at the copies in the zip first.
- Docs: kv-and-caching.md (en / zh-TW) gave "27B KL <= 9e-5 in practice" for a cache hit; it now gives the measured
  balance gen-end range per GPU (R9700 9e-5 to 2e-4, Radeon 8060S 9e-7 to 2.1e-5; precise bitwise). The tier gate
  description said `--parallel 1` runs f16 KV and `--parallel 4` q8v; the suite pins no format (`--kv auto`, the
  format of the server's mode).
- Server gate: new suite `mt_cache_precise` (release gate steps `srv_mt_precise` for Swift-1.5 27B MXFP4-A and
  `srv_mt_precise_moe` for Ornith-1.5-35B-A3B MXFP4): the mt_cache multi-turn sessions in precise must give, request by
  request, the same replies as a server without the prefix cache, and every resumed state must equal a from-scratch
  prefill bit for bit (`WHIRL_PREFIX_CACHE_VERIFY=1`: 0 logits whose bits differ, KL 0; the verify log line now counts
  the bitwise-differing logits).

## 0.2.1 — 2026-10-20 (bug fixes)

- With the "AMD Software: Adrenalin Edition for ROCm 10.1" driver, WHIRL stopped with "amdhip64_7.dll was not found": that driver's GPU runtime needs `rocm_kpack.dll`, which the driver does not install. The zip now ships AMD's `rocm_kpack.dll` next to `whirl.exe`, and if a DLL the runtime needs is still missing, the error names it instead of saying "not found". Root cause found by @vvsvvt, reported by @novelsk ([#2](https://github.com/tsaipifong/whirl-llm/issues/2)).

## 0.2.0 — 2026-10-17

### Highlights

<!-- gen:changelog.lead.en -->
- **Faster than the official llama.cpp b11214 on both GPUs (Radeon 8060S and R9700), in every mode and at every measured length (2k to ~250k tokens)**, ROCm and Vulkan builds compared separately, on our two MXFP4 models. WHIRL balance (default) leads, prefill / decode:
  - Radeon 8060S (max power (80 W), ASUS ROG Flow Z13), Swift-1.5 27B MXFP4: prefill +53% to +65% over ROCm and +93% to +183% over Vulkan; decode +28% to +46% / +24% to +105%.
  - Radeon 8060S, Ornith-1.5-35B-A3B MXFP4: prefill +64% to +91% / +142% to +199%; decode +41% to +54% / +43% to +57% (as shipped: WHIRL MTP vs llama.cpp plain).
  - R9700, Swift: prefill +139% to +191% / +208% to +295%; decode +56% to +80% / +38% to +84%.
  - R9700, Ornith: prefill +75% to +188% / +119% to +225%; decode +74% to +107% / +78% to +106%.
  - Coding-agent session (Swift, cold): 8060S 95.7 s vs 145.6 s / 264.4 s; R9700 17.0 s vs 81.0 s / 69.4 s.
  Decode = document QA (explain and rewrite a function quoted in the prompt). Ranges leave out three footnoted kinds of llama.cpp cells (8060S Vulkan 2k prefill, R9700 Swift 8k cold request and ~250k memory spill); tables: [README](README.md#performance), [benchmarks.md](docs/benchmarks.md).
<!-- /gen -->
- **Radeon 8060S (Ryzen AI Max+ 395, gfx1151) support, tuned**, with image input and the RAM / SSD cache tiers on by default.
- **Our MXFP4 models, quantized for WHIRL**: Swift-1.5 27B MXFP4 (dense) and Ornith-1.5-35B-A3B MXFP4 (MoE). fp8 prefill on the R9700, Q8_0 `ssm_alpha` / `ssm_beta` so MTP output stays bitwise identical to plain greedy decoding, and a Q6_K head for fast drafts ([README](README.md#why-whirl-is-fast-with-these-mxfp4-files)).
- **Three numerics modes**: `--precise` (bitwise lossless), `--balance` (default, near-lossless), `--fast` (opt-in, gated loss; not fully optimized yet, see below). In precise and balance, speculative decoding, batching and cache restores give the same tokens as plain greedy decoding of that mode. Details below and in [usage.md](docs/usage.md#modes).
- **DFlash2 drafts, opt-in** (`--draft dflash`), **SSD tier that writes only reused sessions and never exceeds its cap**, **second question about the same long document served from a tail checkpoint**, **CLI and server give the same output bits** for the same prompt.

### Radeon 8060S

- New GPU: the Radeon 8060S (gfx1151, the Ryzen AI Max+ 395 iGPU) with its own tuned kernel set, for the same models as the R9700 (`qwen35`, `qwen35moe`). `--device 8060s` (or `WHIRL_DEVICE=8060s`) selects it when both GPUs are present; the R9700 stays the default.
- Image input (`--mmproj`, `image_url` parts) works on the 8060S (precise-mode answers matched the R9700's in our checks).
- The RAM and SSD tiers of the prefix cache are **on by default** on integrated GPUs: on Windows the BIOS VRAM carve-out and the OS RAM are separate pools, so the RAM tier is a real second copy. RAM tier size: OS-visible RAM / 8, at least one max-context entry, at most 16 GiB and 1/4 of the RAM available at startup; off under 32 GB of OS-visible RAM (8 GiB on a 64 GB machine). `--kv-ram-mb` overrides, `--kv-ram-mb 0` turns both host tiers off. Discrete GPUs are unchanged.

### Numerics modes

- `--precise` / `--balance` / `--fast` (or `--mode M`, env `WHIRL_MODE`) for `whirl` and `whirl-server`; one mode per process, shown in the start-up log and in `GET /props` (`"numerics"`). With no flag WHIRL runs **balance**.
- **precise**: bitwise lossless; f16 KV on every model and GPU. The same bits with or without drafts, alone or batched, and from a cache restore.
- **balance** (default): near-lossless; q8h KV on dense models, f16 KV on MoE models. MTP / n-gram drafts, batching and cache restores give the same tokens as plain greedy decoding in balance.
- **fast** (opt-in, not fully optimized yet; it will improve in later 0.2.x releases): 4-bit KV (dense models on the Radeon 8060S keep q8h KV, which decodes faster there) and relaxed MTP acceptance on dense models with an MTP head, within the KL limits checked by `tools/klgate/klgate.py`. The text can depend on the drafts and on other users.
- The KV format is fixed per mode, the same on every card, CLI and server. The automatic fallback to a lower-precision KV format is gone: when the format does not fit, the context is lowered with a warning, or the program stops with a message if you gave `--ctx` / `--ctx-per-slot`. `WHIRL_KV` remains a debug override.
- Sampling (temperature > 0) in balance and fast uses a new exact speculative sampler: the text for a seed depends only on the seed and the model, not on the drafts, the prefix cache or earlier requests. `WHIRL_SPEC_SAMPLE=0` turns it off.
- The per-item variables (`WHIRL_FP8`, `WHIRL_KV`, ...) still work and are reported as overrides of the mode.

### DFlash2 drafts (opt-in)

- `--draft dflash --draft-model FILE` (or `WHIRL_DRAFT_MODEL`) for `whirl chat`, `whirl bench` and `whirl-server`, dense 27B models; drafter `Qwen3.8-27B-DFlash2-Q8_0.gguf` from Hugging Face `incoai/Qwen3.8-27B-DFlash2-GGUF` (Apache-2.0, not bundled). Drafts are verified with exact acceptance, so greedy output equals plain greedy decoding of the same mode.
- Compared with MTP: faster for one user with short or medium context and predictable output; slower with 4 users, at long context and on mostly-Chinese output. Costs about 1.9 GiB of VRAM. Off by default.
- Without a fitting drafter (MoE models, an unreadable or malformed file) WHIRL warns and keeps the MTP drafts.

### Prefix cache and SSD tier

- **`--kv-ssd-mode shutdown` is the new default**: only reused sessions are written, when the server stops (Ctrl+C waits up to 60 s for it; closing the console, logoff and OS shutdown allow only about 5 s), after about 5 minutes idle, or when RAM must evict them. A one-shot prompt is never written.
- `--kv-ssd-mode always` now writes a session after `--kv-ssd-delay` seconds unchanged (default 300), and only reused sessions.
- The cache directory never exceeds `--kv-ssd-gb` (default 64 GiB): the cap is also enforced at start-up, files of other models or builds count against it and go first, unusable files are deleted, and files unused for 7 days are deleted.
- The cache fingerprint includes the numerics mode and the GPU, so a precise server never restores state saved in balance, and the R9700 and the 8060S never share files.
- **Long documents**: prompts of 4096+ tokens keep a tail checkpoint, so another question about the same document prefills only about 0.3k-1.3k tokens. `-np 2` to `4` default to 3 checkpoints per slot (counted in the VRAM fit).
- `whirl chat` prefills on the server's schedule, so the CLI and the server give the same output bits for the same prompt (the two documented exceptions are in [kv-and-caching.md, section 7](docs/guide/en/kv-and-caching.md#merge)).

### Important fixes

- Server: a client that disconnects frees its slot; parallel prefills that fill the KV pool are queued instead of failing with 500 (the remaining case is 503 with `Retry-After`); a decode stopped by a full pool adds `"finish_cause": "kv_pool_exhausted"`.
- A tool call cut off by `max_tokens` or a stop string is no longer returned as a complete call.
- SSD tier: short or corrupt files are deleted and failed reads fall back to a cold prefill instead of restoring stale data; a session can no longer be deleted while it is being restored; rewriting a file is crash-safe.
- Models with DeltaNet shapes the kernels do not implement are refused (`UnsupportedConfig`) before any GPU work; a malformed DFlash2 drafter is refused instead of overrunning GPU buffers.
- `WHIRL_PREFILL_BATCH` below 1280 is raised to 1280 (smaller values failed long prompts).
- Smaller: requests during shutdown get 503; an invalid `--kv-ssd-mode` is a usage error; temperature below 1e-3 is greedy; at most 16 stop strings of up to 256 bytes; `completion_tokens` no longer counts tokens after a stop string.

### Breaking changes and changed defaults

- **Cross-origin requests are refused**: the `Host` header must be `localhost`, `127.0.0.1`, `[::1]`, the `--host` address or a name given with `--allow-host NAME`; a non-GET request with a disallowed Origin gets 403. Clients that reach the server by a host name need `--allow-host`. `--allow-local-images` reads local files only for requests from a localhost page.
- Default `--kv-ssd-mode` is `shutdown` (was `always`); `always` now waits 5 minutes and writes only reused sessions.
- The automatic KV format fallback is gone (see Numerics modes); use `--ctx` / `--ctx-per-slot` to size the context.
- Use `--precise` when you need bitwise-lossless output (identical to plain greedy decoding). MXFP4 prefill is slower in precise mode than in 0.1.x.
- Sampled text for a given seed can differ from earlier versions in balance and fast (new sampler).

## 0.1.4 — 2026-10-06

Routine bug-fix release: no new features, no speed changes. The changes harden input handling (GGUF
files, HTTP requests, tokenizer vocabularies); inference code paths are unchanged, so outputs are
bit-identical to 0.1.3.

### Fixes

- Server (HTTP): a streaming client that stops reading no longer stalls the other slots. While a
  request runs, the engine writes its response through a non-blocking socket; bytes the client does
  not accept stay buffered and are retried on the next flush. With no progress for 30 s, or more than
  64 MiB unsent, the connection counts as gone and the request is dropped (its slot is freed). The
  rest of a finished response is sent by the connection thread (a10ac5b).
- Server: shutdown no longer waits up to 30 s for connections still reading a request; `stop()`
  shuts them down and cancels the pending reads. Connections close with a linger so clients receive
  error responses instead of a reset (a10ac5b).
- Tokenizer: `piece()` decodes `<0xNN>` byte tokens with a small hex parser instead of `std::stoi`,
  which threw on `<0xZZ>` and silently mis-decoded `<0x-1>` / `<0x1Z>` (24475c4).
- zh-TW guide: bold runs broken by CJK punctuation render correctly again (no text change, 8b2c58d).

### Security hardening

- GGUF parser: tensor extent checks are overflow-safe (subtraction-based `File::inBounds`, at parse
  and in `tensorData()`), so `data_offset + offset + nbytes` can no longer wrap past 2^64; dimensions
  above INT64_MAX and shapes whose row count, element count or byte size overflow are rejected;
  `TensorInfo::nbytes()` / `rowBytes()` are overflow-checked. The header's KV and tensor counts must
  fit in the remaining file bytes, so their `reserve()` is bounded by the file size (b6af30a).
- Model load: `Config::fromGguf` rejects inconsistent or out-of-range hyper-parameters (for example
  `nextn >= block_count`, zero dimensions, `head_dim > 256`, GQA group > 8,
  `value_length != key_length`, interval 0) with `UnsupportedConfig`; `Config::validateTensors`
  checks the type and shape of every tensor `Model::load` reads (including MoE experts and the MTP
  block) before any GPU allocation, with the new error codes `MissingTensor`, `UnsupportedTensorType`
  and `UnsupportedTensorShape` (b6af30a).
- HTTP: 30 s idle timeout per read and a 60 s deadline for the request line and headers (`408`); at
  most 64 open connections (`503`, then close); headers capped at 64 KiB / 100 lines (`431`);
  `Content-Length` above 64 MiB is refused (`413`) instead of wrapping, conflicting duplicate
  `Content-Length` headers get `400`; the body buffer grows as data arrives instead of reserving the
  claimed length (a10ac5b).
- CORS: `Access-Control-Allow-Origin` is sent only to pages from `http(s)://localhost`, `127.0.0.1`
  or `[::1]` (any port), echoing the `Origin` with `Vary: Origin`; other origins (including `null`,
  i.e. `file://` pages) get no CORS headers, and preflight answers the same way. New option
  `--cors-origin ORIGIN` (repeatable; `*` restores the previous allow-all behaviour) (a10ac5b,
  docs 765f49a).
- Tokenizer: a byte-type token whose text is not exactly `<0xNN>` (two hex digits) fails loading with
  `TokenizerError("malformed byte token at id N")` (24475c4).

### Behaviour changes

- CORS default tightened (see above). Browser pages on other origins, or opened from `file://`, can no
  longer read the server's responses unless started with `--cors-origin <origin>` (or
  `--cors-origin "*"`). SDKs, curl, IDE agents and other non-browser clients are not affected.
- HTTP limits: 64 connections, 30 s read idle / 60 s header deadline, 30 s send stall or 64 MiB
  unsent on a streaming response, 64 KiB / 100 header lines, 64 MiB request body.
- Malformed GGUF files that 0.1.3 might have loaded (or crashed on) are now refused at load with one
  of the error codes above; the release front-end reports them as "cannot be loaded".

### Internal

- Version 0.1.4 (`project(whirl VERSION 0.1.4)`).
- MXFP4: the loader and the tests share one row-repack implementation (`kernels::repackMxfp4Row`,
  new scratch-buffer overload; scratch still allocated once per tensor); model tests pin the bytes
  against frozen copies of both previous implementations (ee87049).
- `build.bat` finds `vcvars64.bat` with `vswhere`, restricted to Visual Studio 2022
  (`-version [17.0,18.0)`; HIP clang with the VS 2026 STL is unverified), any edition with the C++ x64
  tools; it falls back to the default Build Tools path and prints a clear error if neither exists
  (f54e2da).
- Tests: crafted malformed GGUFs, synthetic tiny models with mutations (optional
  `WHIRL_TEST_GGUF="a.gguf;b.gguf"` runs the load validation on real files without a GPU), HTTP limits,
  connection cap, stuck reader, CORS matrix, byte-token validation.

## 0.1.3 — 2026-10-05

Every change below keeps the output bit-identical to plain greedy decoding, unless it says otherwise.

### Performance

- Prefill attention: new GQA-grouped kernel (`attn_kg`) for f16, q8v, q8 and q8h KV. One block holds the
  query heads of one KV head over 16 queries; K fragments are loaded straight into WMMA registers,
  f16 Vᵀ fragments with RDNA 4's transposing load (`global_load_tr_b128`), q8v V is dequantized once
  per block into a double-buffered LDS stage, and there is one barrier per 32-key tile. Outputs are
  bit-identical to the previous kernel (kernel-test invariants; last-token logits, MTP == plain and
  server outputs identical to 0.1.2). Kernel throughput at long context: f16 ~65 → ~93 TFLOPS, q8v
  ~61 → ~84 TFLOPS. `whirl bench` prefill, Swift-1.5 27B MXFP4-A (f16 KV, R9700): 32k +10%, 64k
  +17%, 96k +21%, 128k +24%; Ornith MXFP4 64k +11%, 128k +14%; Qwen3.8-27B unsloth UD 4-bit K-quant (f16 KV) 32k
  +5%, 128k +16%. Long prompts on the server (q8v KV for the dense models): Swift 96k +19%, 128k +21%.
  q8 and q8h KV use `attn_kg` too (`attn_kg6_q8` / `attn_kg4_q8`; K dequantized from int8 with a
  magic-number f16 conversion, bit-identical to `attn_kx_q8`; probe at 61k 57.8 → 72.1 TFLOPS). Server
  prefill, Swift-1.5 27B MXFP4-A, one user, `--ctx-per-slot 262144`, q8h KV: 128k 1,342 → 1,522
  (+13.4%), 192k 1,021 → 1,177 (+15.3%), 256k 824 → 961 (+16.6%). `WHIRL_ATTN_KG=0` restores the
  previous kernel. See [kernels.md](docs/guide/en/kernels.md#flash).
- Radeon 8060S: q8 / q8h / q4 KV prefill attention through dequantized f16 rows (KG-3, the gfx1151
  port of Q8P / FC-1c). `kv_dq_rows_r` / `kv_dq_rows_r_q4` dequantize the sequence's keys once per
  layer and chunk into `ffn_g` / `ffn_u` and the f16 `attn_prefill_wmma` runs on them; keys past the
  scratch (139k at batch 4096 on the 27B) go in ranges through the new `attn_prefill_wmma_s` with the
  softmax state carried. Bit-identical to `attn_prefill_wmma_q8` / `_q4` (kernel test: 180 new
  invariants; 16 greedy tokens after 64k / 128k prompts identical; bench hash identical), so goldens
  are unchanged. Speed, Qwen3.8-27B unsloth UD 4-bit K-quant, end to end: within the run-to-run drift of the 8060S
  (about ±3% between consecutive 4-minute runs). balance q8h vs `WHIRL_KV=f16`: prefill 8k 374.2 vs
  377.5 tok/s (−0.9%; old loader 369.5, −2.1%); 64k, two runs in opposite orders, 279.6 / 279.7 vs
  288.3 / 277.0 (mean −1.1%; old loader 272.8 / 280.2, −2.2%); 128k, two runs in opposite orders,
  212.1 / 221.9 vs 221.0 / 213.3 (mean −0.1%; whichever ran first was ~4% faster). fast q4 at 128k 211.9 vs the old loader 214.6 (−1.3%, noise). The kernel
  bench (4096 queries, 24 / 4 heads, random K / V) disagrees with the end-to-end runs: there the f16
  kernel on f16 data was slower than the int8 loader (64k 490–523 vs 388–425 ms) and key ranges of
  16k–32k keys cut f16 by 26–35%, but `WHIRL_ATTN_SEG=32768` end to end changed nothing (64k q8h
  279.9, f16 279.4, same text), so the bench is not representative on this GPU. `WHIRL_ATTN_DQF=0`
  restores the old loaders; `WHIRL_ATTN_SEG=N` (new, test knob) forces key ranges, f16 KV included.
- **q8 / q8h KV prefill attention at f16 speed (Q8P)**, so balance can default to q8h on dense
  models: per layer and prefill chunk, `kv_dq_rows` dequantizes the sequence's keys once into f16 rows
  (in `ffn_g` / `ffn_u`, idle during attention; up to 139k keys at batch 4096) and the f16 `attn_kg`
  runs on them, bit-identical to `attn_kg_q8` (which converts K once per query head and was 29%
  slower than f16 at 128k keys). R9700, balance, q8h vs `WHIRL_KV=f16`: Swift-1.5 27B MXFP4-A prefill
  8k −1.5%, 32k −0.9%, 128k −1.0% (before −3.3 / −7.7 / −15.1%), decode_7p +1.7%; Qwen3.8-27B
  unsloth UD 4-bit K-quant 8k −0.5%, 32k −0.4%, 128k −0.7%, decode_7p +0.4%. Radeon 8060S (unchanged kernels):
  same 4-bit K-quant model, prefill 8k −0.8%, 64k −2.7%, decode at 64k −1.9%. `WHIRL_ATTN_DQF=0` restores the int8
  kernel. Goldens unchanged on gfx1201; gfx1151 dense balance re-recorded for the q8h default.
- Speculative verify attention: a sequence's verify rows share one K/V pass in groups of up to 32
  query columns (`attn_wsplit2`; 27B: 5 rows × 6 GQA heads) instead of 16 (2 rows). The second column
  group runs on the block's otherwise idle second wave over the same staged Vᵀ tile; each column's
  arithmetic is unchanged (checked by `checkAttnGroups`, the kernel test and server A/B against
  0.1.2). Attention kernel time per layer at 16k context: one user with 4 drafts 0.286 → 0.141 ms,
  four users with 3 drafts each 0.853 → 0.607 ms; at 32k, 1.784 → 1.162 ms. Single-row decode keeps
  `attn_wsplit1`. `WHIRL_ATTN_WIDE=0` restores the old grouping.
- Wide verify: verify forwards of 17–32 rows use dual-token WMMA GEMV kernels (`gemvx_v6` for MXFP4,
  IQ4_XS, Q4_K, Q5_K, Q6_K) and the output head covers the whole vocabulary per chunk
  (`head_chunk = 248320`; `WHIRL_WIDE_VERIFY=0`, `WHIRL_HEAD_CHUNK=N`). Only several users (or n-gram
  drafts adding rows) take this path; one user's verify has at most 16 rows. 4 users, prose:
  247.1 → 267.8 tok/s. Default draft caps stay 8 / 7 / 4 / 3 for 1–4 decoding slots.
- MTP block stored as Q8_0 (e.g. the Swift-1.5 MXFP4 files) is requantized to Q4_K for drafting, like
  Q6_K blocks already were (`WHIRL_MTP_Q4=0` keeps the file's types). Drafts only.
- Server: Windows timer resolution set to 1 ms while serving (`timeBeginPeriod`), and the main loop's
  idle / burst-gather / restore waits wake on a new request instead of sleeping: a 1 ms sleep took
  11.5 ms before, 1.9 ms after (`WHIRL_TIMER_PROBE=1`). A short request arriving while the
  prefix-cache tier is still writing the previous one: TTFT median 86 → 69 ms. C = 4 burst of ~1.1k
  prompts: mean TTFT 1207 → 1187 ms.
- Server: with the decode floor off (`--decode-min-tps 0`), a prefilling request's chunks merge into
  2048-row forwards even while other slots decode. With the floor on they still do not while any slot
  decodes (merging next to a long prompt made its prefill only ~4% faster but doubled the decoders'
  stalls, 96k: 1.2 → 3.3 s).

### Fixes

- Server KV pool sized under the WDDM budget, MTP KV allocated first (FIX-1). The automatic pool kept
  only 768 MiB of free VRAM, which put the process over its WDDM local budget (31.29 / 31.02 GiB on
  the R9700); Windows then demoted the last allocation, the MTP KV, to system memory and the draft
  head read it across PCIe: a ~128k-token agent request decoded at 29.0 tok/s. The pool is now
  min(free VRAM − reserve, WDDM budget − usage − margin) (`hip::wddmMemInfo()`, DXGI
  `IDXGIAdapter3::QueryVideoMemoryInfo`, adapter matched by LUID; margin 768 MiB, MoE 1.5 GiB,
  `WHIRL_POOL_BUDGET_MARGIN_MB`; `WHIRL_POOL_RESERVE_MB` keeps the old free-VRAM-only rule), and
  `allocKvPool` allocates the MTP KV before the trunk layers. After startup the server logs WDDM local
  usage / budget and non-local growth, and warns when over budget. Swift-1.5 27B MXFP4-A, default 4
  slots: pool 226,304 tokens (v0.1.2: 210,688; +7.4%), local 30.06 / 31.02 GiB; the 128k agent request
  decodes at 67.2 tok/s, output identical. Server tests 340/340; gates basic / tier / restore_conc
  identical to v0.1.2.
- Decode floor (`--decode-min-tps N`): every decoding slot is protected except one that arrived in
  the same burst (within the 30 ms gathering window) as a request that is still prefilling. Requests
  arriving together therefore merge their prefill without false floor throttling (Scenario A, C = 4
  concurrent ~1.1k prompts: 0 floor forwards, batch wall 4.65 s vs 4.91 s in v0.1.2, median of 3),
  while a stream is protected regardless of arrival order. An earlier version of this change protected
  only slots that arrived before every prefilling request, which left a conversation's next turn
  unprotected when it arrived after its subagents' prompts ("reverse order", 25.7k: 8.6 tok/s while
  they prefilled; now 22.8–23.0, v0.1.2 22.4); the server test `decode_floor` now covers that order.
  Scenario B (Swift 27B MXFP4-A, 3 concurrent ~17k subagents) at 25.7k context is unchanged
  (22.7 tok/s, longest pause 0.47 s; v0.1.2 23.1 / 0.47). Outputs are identical to v0.1.2.
- CLI: n-gram drafts are capped at 1 on the unfused DeltaNet decode path (for example F32 / F16
  `ssm_alpha` / `ssm_beta` without `gdn_ab`), like the MTP draft count and the server already were.
  That path writes only snapshot 0, so accepting 1..nd−1 drafts restored a never-written snapshot and
  diverged from greedy. `restoreSnapshot` now throws `SnapshotNotWritten` for out-of-range or
  unwritten sets (42b4cc9).

### Defaults

- The token embedding table is kept in pinned host memory (`WHIRL_EMBD_HOST`, now on by default;
  `WHIRL_EMBD_HOST=0` keeps it in VRAM). The KV pool grows by ~11.9% on the R9700, prefill −0.4–0.9%.
  Models with tied embeddings or an MXFP4 embedding keep it in VRAM. The server adds the embedding
  size to its pinned-memory declaration (`%LOCALAPPDATA%\whirl\pinned\<pid>.txt`), so Shared Usage
  monitoring does not count it as spill.
- The MTP draft head uses a 64k-token vocabulary subset (`WHIRL_DRAFT_VOCAB`, default `64k`),
  embedded in the executable (`data/draft_vocab/subset_64k.bin` via whirl-bin2c), for dense qwen35
  models with the 2-bit draft head and a 248,320-token vocabulary; MoE models and other vocabularies
  keep the full head. Main scenario +3.4–5.0% tok/s, same acceptance. `WHIRL_DRAFT_VOCAB=off|48k|<file>`
  overrides. The subset is only a list of token ids (uint32 LE, ascending), ranked by token frequency
  over permissively licensed code and documentation (llama.cpp, ROCm aiter, hipfire, dflash,
  PaddleNLP / PaddleOCR, ECharts, WHIRL docs) and Wikipedia samples (CC BY-SA 4.0), tokenized with the
  Qwen3.x tokenizer; no corpus text is included. Generated by whirl-cloud `tools/vocab_subset/build.py`
  (sources and licenses in its `sources.tsv`).
- MTP draft attention window on by default (DEF-2, 2ba0713): from a 64k-token context on, the draft
  attention sees only the first 256 and the last 16,384 positions. Trunk and verification always see
  the whole context, so output hashes are identical. Window W = 16,384 against off: typical case
  (real agent sessions, server) 110k +1.1%, 128k +7.6%, 200k +3.7%; worst case (long non-repeating
  text) 128k −0.5%, 256k +17% (single run); best case (repeated editing) 128k +3.1%, 256k +4.2%.
  `WHIRL_DRAFT_WINDOW=0` turns it off; `WHIRL_DRAFT_WINDOW=W` / `WHIRL_DRAFT_WINDOW_MIN=N` change the
  window and the threshold.

### Internal

- Version 0.1.3 (`project(whirl VERSION 0.1.3)`, 8fbee5f).
- `package_release`: `dxgi.dll` (WDDM budget query, FIX-1) and `winmm.dll` (`timeBeginPeriod`) added
  to the allowed system DLLs; both ship with every Windows 10/11 install, so requirements are
  unchanged (3b1bebc).
- `whirl bench`: `WHIRL_PROFILE=1` prints the per-op-class GPU time of each prefill size.

## 0.1.2 — 2026-10-03

### Changed

- Server: the pinned-RAM tier of the prefix cache (`--kv-ram-mb`) is now sized from the machine
  instead of a fixed max(8 GiB, one full-length session + checkpoints) ≈ 9 GiB: **1/4 of physical
  RAM** (nearest GiB), at least that old minimum, at most 32 GiB (the minimum wins over the cap), and
  never more than half of the RAM available at startup (so the pinned arena does not push the machine
  into paging; logged as a warning when it limits the size; under 1 GiB the tier stays off). A 64 GB
  PC now gets 16 GiB. In real agent use (Hermes, 30–40k-token sessions with subagents) the old 9 GiB
  filled within an hour, after which restores came from the SSD. `--kv-ram-mb N` /
  `WHIRL_KV_RAM_MB=N` still override (used as given), `0` still turns the host tiers off. On an
  integrated GPU (shared system memory) the RAM tier is off unless a size is given. The startup log
  prints the chosen size and the reason, and the pinning time.

## 0.1.1 — 2026-10-03

### Added

- Server: read-only compatibility endpoints for clients that auto-detect the server type and context
  length (agent frameworks, chat front ends, IDE plugins):
  - `GET /props` and `GET /v1/props` — llama.cpp-server-style subset: `default_generation_settings`
    (`n_ctx` = context per slot, `model`), `total_slots`, `model_path` (file name only), `model_alias`,
    `modalities`, `build_info`.
  - `GET /version` — `{"version":"0.1.1","name":"whirl"}`.

- Chat requests: the OpenRouter / OpenAI Responses-style `"reasoning": {"effort": "...", "enabled": bool}`
  object, and effort aliases: `max` / `ultra` / `xhigh` / `high` → xhigh, `medium` → medium,
  `low` / `minimal` → low, `none` → thinking off (values are case-insensitive).

- Server: decode floor, `--decode-min-tps N` (env `WHIRL_DECODE_MIN_TPS`, default 20, `0` = off).
  While other requests prefill long prompts, every streaming request keeps at least N tok/s: prefill
  forwards are limited to a row budget (whole chunks; the oldest request always progresses) and decode
  cycles are interleaved, adapted every cycle from the measured decode-cycle time, tokens per cycle and
  prefill rate. Swift-1.5 27B MXFP4 on the R9700, one stream at 25.7k context + three ~17k-token prompts:
  the stream goes from 3.3 to 23.0 tok/s (longest pause 1.22 → 0.47 s), prefill throughput meanwhile
  2791 → 1862 tok/s, mean TTFT of the three 17.8 → 18.6 s. With no slot decoding, prefill is unchanged;
  outputs are bit-identical for any N; short-prompt C = 1 / C = 4 throughput unchanged.

### Changed

- Chat requests: an unknown reasoning effort no longer fails the request with 400; the server logs a
  warning and uses the default effort. Prompts for the values accepted before are unchanged.
- Server: probes of LM Studio / Ollama native paths (`/api/v1/models`, `/api/tags`, `/api/show`,
  `/api/version`, `/api/v0/models`) still answer 404 but are logged once per path at `I` level
  instead of a `W` line per request.

## 0.1.0 — 2026-10-03

- First public release: native Windows C++/HIP inference engine for qwen35 / qwen35moe GGUF models on
  the Radeon AI PRO R9700 (gfx1201) — `whirl.exe` (CLI) and `whirl-server.exe` (OpenAI-compatible
  server with continuous batching, prefix cache, RAM / SSD KV tiers, MTP and n-gram speculative
  decoding, image input).
