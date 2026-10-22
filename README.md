**English** | [繁體中文](docs/README_zh-TW.md)

# WHIRL

**Windows HIP Inference for RDNA LLMs**

> ### New in 0.2.0: AMD Ryzen AI Max+ 395 (Radeon 8060S) support
> WHIRL now runs on the **Radeon 8060S** integrated GPU of the **AMD Ryzen AI Max+ 395**, with kernels tuned for it, all three numerics modes, and prompts up to ~250k tokens. On the 8060S it is faster than the official llama.cpp ROCm and Vulkan builds at every prompt length we measured (see the [8060S tables below](#whirl-vs-llamacpp-at-a-glance); max power (80 W), ASUS ROG Flow Z13).

WHIRL is an open-source (Apache-2.0) LLM inference engine for AMD GPUs on Windows: native C++/HIP, every kernel included, no WSL, Docker or ROCm install. **Download the zip, unzip, run; the AMD driver is all it needs.** Version 0.2.0 runs on the **Radeon 8060S** of the **AMD Ryzen AI Max+ 395** (new in 0.2.0) and the **AMD Radeon AI PRO R9700** (RDNA 4, 32 GB), with our two MXFP4 models, **Swift-1.5 27B MXFP4** (dense) and **Ornith-1.5-35B-A3B MXFP4** (MoE). On both GPUs it is **faster than llama.cpp**, ROCm and Vulkan builds alike, in every mode and at every prompt length we measured, from 2k to ~250k tokens.

## WHIRL vs llama.cpp at a glance

WHIRL 0.2.0 in its default mode (balance) against the official llama.cpp release **b11214** (ggml-org), shown separately for its **ROCm** and its **Vulkan** build. Same GGUF files, same prompts, greedy decoding, one user.

![Prefill speed on the Radeon 8060S (Ryzen AI Max+ 395), WHIRL 0.2.0 vs llama.cpp ROCm and Vulkan](docs/images/bench020_prefill_s8.png)

![Decode speed on the Radeon 8060S (Ryzen AI Max+ 395), WHIRL 0.2.0 vs llama.cpp ROCm and Vulkan](docs/images/bench020_decode_s8.png)

![Prefill speed on the Radeon AI PRO R9700, WHIRL 0.2.0 vs llama.cpp ROCm and Vulkan](docs/images/bench020_prefill_r97.png)

![Decode speed on the Radeon AI PRO R9700, WHIRL 0.2.0 vs llama.cpp ROCm and Vulkan](docs/images/bench020_decode_r97.png)

![Coding-agent session wall time, WHIRL 0.2.0 vs llama.cpp ROCm and Vulkan](docs/images/bench020_agent.png)

![WHIRL 0.2.0 prefill at 32k in precise, balance and fast mode](docs/images/bench020_modes_pp.png)

![WHIRL 0.2.0 decode after a 32k document in precise, balance and fast mode](docs/images/bench020_modes_tg.png)

<!-- gen:lead.en -->
- **Radeon 8060S** (max power (80 W), ASUS ROG Flow Z13): prefill at 8k Ornith 2,244 tok/s (+75% / +199%), Swift 470.8 (+53% / +101%); decode after a 128k-token document Ornith 58.4 (+54% / +57%), Swift 23.8 (+46% / +105%).
- **Prefill, R9700**: Ornith 35B-A3B reads a 2k-token prompt at **11,550 tok/s** (llama.cpp ROCm 4,014, Vulkan 3,551: WHIRL +188% / +225%) and a ~250k-token prompt at 2,664 tok/s (1,519 / 1,218: +75% / +119%). Swift 27B: 2,805 tok/s at 32k (+163% / +256%) and 1,715 at 128k (+139% / +208%).
- **Decode, R9700**: Ornith 238.4 tok/s answering about a 2k-token document (+107% / +106%); Swift 79.2 tok/s after a 32k-token document (+80% / +84%).
- **A real coding-agent session** (Swift 27B, cold): 95.7 s on the 8060S vs 145.6 s (ROCm) and 264.4 s (Vulkan); 17.0 s on the R9700 vs 81.0 s and 69.4 s.
<!-- /gen -->

**Radeon 8060S** (gfx1151, Ryzen AI Max+ 395), **max power (80 W), ASUS ROG Flow Z13**. The percentage in each llama.cpp cell is WHIRL's lead over that cell.

<!-- gen:glance.s8.en -->
| tok/s, higher is better | Swift 27B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan | Ornith 35B-A3B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill, 2k-token prompt | **498.4** | 305.7 (WHIRL +63%) | 162.8ᶜ (WHIRL +206%) | **2,411** | 1,259 (WHIRL +91%) | 447.7ᶜ (WHIRL +439%) |
| Prefill, 32k-token prompt | **426.7** | 266.2 (WHIRL +60%) | 221.6 (WHIRL +93%) | **1,829** | 1,049 (WHIRL +74%) | 756.2 (WHIRL +142%) |
| Prefill, 128k-token prompt | **298.9** | 180.8 (WHIRL +65%) | 127.8 (WHIRL +134%) | **1,055** | 636.6 (WHIRL +66%) | 428.4 (WHIRL +146%) |
| Prefill, ~250k-token prompt | **221.3** | 135.1 (WHIRL +64%) | 78.1 (WHIRL +183%) | **685.0** | 418.3 (WHIRL +64%) | 278.9 (WHIRL +146%) |
| Decode, QA about a 2k-token document | **31.2** | 24.4 (WHIRL +28%) | 25.1 (WHIRL +24%) | **101.4** | 70.0 (WHIRL +45%) | 70.7 (WHIRL +43%) |
| Decode, QA about a 32k-token document | **29.5** | 22.7 (WHIRL +30%) | 17.0 (WHIRL +74%) | **83.8** | 57.9 (WHIRL +45%) | 57.1 (WHIRL +47%) |
| Decode, QA about a 128k-token document | **23.8** | 16.2 (WHIRL +46%) | 11.6 (WHIRL +105%) | **58.4** | 38.0 (WHIRL +54%) | 37.3 (WHIRL +57%) |
| Decode, coding chat (new text) | **34.2** | 27.6 (WHIRL +24%) | 28.3 (WHIRL +21%) | **106.0** | 70.6 (WHIRL +50%) | 72.9 (WHIRL +45%) |
<!-- /gen -->

**Radeon AI PRO R9700** (gfx1201, 32 GB).

<!-- gen:glance.r97.en -->
| tok/s, higher is better | Swift 27B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan | Ornith 35B-A3B: WHIRL | llama.cpp ROCm | llama.cpp Vulkan |
|---|---|---|---|---|---|---|
| Prefill, 2k-token prompt | **3,273** | 1,124 (WHIRL +191%) | 829.4 (WHIRL +295%) | **11,550** | 4,014 (WHIRL +188%) | 3,551 (WHIRL +225%) |
| Prefill, 32k-token prompt | **2,805** | 1,065 (WHIRL +163%) | 787.5 (WHIRL +256%) | **8,469** | 3,872 (WHIRL +119%) | 3,395 (WHIRL +149%) |
| Prefill, 128k-token prompt | **1,715** | 718.7 (WHIRL +139%) | 557.2 (WHIRL +208%) | **4,275** | 2,307 (WHIRL +85%) | 1,922 (WHIRL +122%) |
| Prefill, ~250k-token prompt | **1,153** | 342.0ᵃ (WHIRL +237%) | 259.9ᵃ (WHIRL +344%) | **2,664** | 1,519 (WHIRL +75%) | 1,218 (WHIRL +119%) |
| Decode, QA about a 2k-token document | **87.6** | 50.5 (WHIRL +74%) | 50.4 (WHIRL +74%) | **238.4** | 115.4 (WHIRL +107%) | 115.9 (WHIRL +106%) |
| Decode, QA about a 32k-token document | **79.2** | 44.0 (WHIRL +80%) | 43.1 (WHIRL +84%) | **199.1** | 101.4 (WHIRL +96%) | 100.6 (WHIRL +98%) |
| Decode, QA about a 128k-token document | **53.9** | 33.2 (WHIRL +62%) | 34.8 (WHIRL +55%) | **131.3** | 75.3 (WHIRL +74%) | 73.8 (WHIRL +78%) |
| Decode, coding chat (new text) | **109.6** | 57.8 (WHIRL +90%) | 57.6 (WHIRL +90%) | **241.5** | 118.9 (WHIRL +103%) | 115.1 (WHIRL +110%) |
<!-- /gen -->

<!-- gen:foot.glance.en -->
ᵃ ~250k on the R9700 with Swift 27B: official llama.cpp with f16 KV does not fit in 32 GB at this length (its log: "failed to fit params to free device memory") and spills into shared memory, which collapses its decode. The numbers are shown as measured, but this cell is not used as a lead anywhere.

ᶜ 2k prefill on the 8060S, llama.cpp Vulkan: Vulkan prefill is much slower on short prompts on this GPU (well below its own trend at 8k+), so the 2k lead over Vulkan is not used as a headline; headline leads over Vulkan use 8k and longer.
<!-- /gen -->

What the rows measure (exact prompts and commands: [benchmarks.md](docs/benchmarks.md#prompts)):

- **Prefill**: prompt processing speed for a long document of that many tokens. "~250k" is a prompt of 249,510 tokens.
- **Decode, document QA**: the same question at every length (explain one function quoted near the start of the document and write a corrected version, 256 tokens). Part of the answer can copy code from the prompt, which helps speculative drafts in every engine that uses them.
- **Decode, coding chat (new text)**: a coding question whose answer (explanation and code) is all newly written, with nothing in the prompt to copy: a hard case for speculative drafts. 512 tokens.
- **How each engine decodes**: WHIRL uses its MTP (+ n-gram) drafts. llama.cpp decodes Swift 27B with MTP + n-gram (`--spec-type draft-mtp,ngram-mod`), its fastest mode on this model. On Ornith it decodes **without** speculation, because its MTP was slower than plain decoding on this MoE model in our measurements; the Ornith decode rows therefore compare both engines **as shipped**, not like for like.
- Every R9700 cell is a single run, except WHIRL's coding-chat cells (median of 3); treat differences under about 5% as noise. Full tables, with all three WHIRL modes: [Performance](#performance).

### A real coding-agent session

Swift 27B, WHIRL balance. One session = a ~20k-token system + tools prompt (coding-agent instructions plus skills / MCP tool manuals), a `read_file` tool call that returns a ~3.6k-token C++ source file, then four user turns (explain the file, find its most complex function, rewrite that function, review an edited copy of the file), with the full history resent on every call, at most 384 tokens per answer. The fixture is in the repository ([benchmarks.md §6](docs/benchmarks.md#agent-session)).

**Radeon 8060S** (max power (80 W), ASUS ROG Flow Z13):

<!-- gen:agent.s8.en -->
| Session (wall time) | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL time saved vs ROCm / Vulkan | model calls WHIRL / ROCm / Vulkan |
|---|---|---|---|---|---|
| cold: the 20k-token system + tools prompt is not cached yet | **95.7 s** | 145.6 s | 264.4 s | 34% / 64% | 6/6/6 |
| second session: shares that prefix, works on a different file | **44.9 s** | 66.2 s | 196.8 s | 32% / 77% | 5/5/6 |
<!-- /gen -->

**R9700:**

<!-- gen:agent.r97.en -->
| Session (wall time) | WHIRL balance | llama.cpp ROCm | llama.cpp Vulkan | WHIRL time saved vs ROCm / Vulkan | model calls WHIRL / ROCm / Vulkan |
|---|---|---|---|---|---|
| cold: the 20k-token system + tools prompt is not cached yet | **17.0 s** | 81.0 s | 69.4 s | 79% / 76% | 5/6/6 |
| second session: shares that prefix, works on a different file | **11.8 s** | 60.9 s | 38.0 s | 81% / 69% | 6/5/6 |
<!-- /gen -->

Each engine writes its own answers, so the number of model calls differs (last column: an engine that re-reads the file makes one more call). Single run per cell.

## Why WHIRL is fast with these MXFP4 files

Our MXFP4 files were quantized for WHIRL, and WHIRL has kernels written for them:

<!-- gen:why.en -->
- **fp8 prefill on the R9700** (balance and fast): the MXFP4 matrix multiplies of prefill run on RDNA 4's fp8 matrix units (Swift 27B at 2k: 3,273 tok/s vs 1,702 without fp8 in precise). The 8060S has no fp8 units and uses its regular path.
- **`ssm_alpha` / `ssm_beta` in Q8_0** (instead of F32): single-token decode and multi-token MTP verification then run through the same fused kernel, so MTP output stays bitwise identical to plain greedy decoding and the drafts stay long.
- **Q6_K output head**: the fastest head for WHIRL's draft step (every draft token goes through the 248k-row head).
<!-- /gen -->

Every kernel is open source in this repository; nothing is closed or downloaded at run time.

## Requirements

| | |
|---|---|
| **Download and run** | Download **`whirl-0.2.3-windows-x64.zip`** from [Releases](https://github.com/tsaipifong/whirl-llm/releases), unzip, run. **Only the AMD driver is needed**: no HIP SDK, no ROCm, no Visual C++ Redistributable to install (the few runtime DLLs the ROCm 10.1 driver needs are in the zip) |
| Driver | **AMD Software: Adrenalin Edition 26.8.1 (driver 32.0.31041.1004) or newer**, for both GPUs. Older drivers are untested. With the Adrenalin Edition for ROCm 10.1 driver, keep the `rocm_kpack.dll` from the zip next to `whirl.exe` (that driver's GPU runtime needs it but does not install it), together with the Visual C++ runtime DLLs it uses (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`, also in the zip) |
| GPU | **Radeon 8060S** (RDNA 3.5, gfx1151, the AMD Ryzen AI Max+ 395 iGPU) or **AMD Radeon AI PRO R9700** (RDNA 4, gfx1201, 32 GB); one GPU per process |
| Radeon 8060S: GPU memory | **Set the GPU memory (UMA frame buffer size) in Armoury Crate or the BIOS first** (other laptops: *Variable Graphics Memory* in AMD Software), then restart. WHIRL keeps the model, its buffers and the KV cache in this dedicated memory; Windows' "shared GPU memory" is not used for them (the token-embedding table and the RAM cache tier sit in pinned system RAM, which Task Manager shows as shared GPU memory). Set **at least 40 GB for Swift-1.5 27B MXFP4** (48 GB for its precise mode with the default KV cache) and **32 GB for Ornith-1.5-35B-A3B MXFP4** (64 GB on a 128 GB machine runs either model in every mode). Measured with the server defaults (4 slots, 262,144-token KV cache): Swift uses 29.9 GiB in balance, 31.0 GiB in fast, 38.2 GiB in precise; Ornith 27.3 GiB in balance and precise, 23.4 GiB in fast. With less, the server shrinks the KV cache, or stops with a message naming the size to set. `whirl devices` shows the current size as "dedicated" |
| OS | Windows 11, 64-bit |
| Model | [Swift-1.5 27B MXFP4](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) or [Ornith-1.5-35B-A3B MXFP4](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) from Hugging Face ([supported models](#supported-models)) |
| Memory / disk | the server pins host RAM for its RAM cache tier (Radeon 8060S: 1/8 of the OS-visible RAM, 8 GiB on a 64 GB machine, off under 32 GB; R9700 host: about 1/4 of RAM, 16 GiB on a 64 GB PC) and may use up to 64 GiB of SSD for the SSD tier, written only when the server stops; both adjustable or off (`--kv-ram-mb`, `--kv-ssd-gb`) |
| Advanced: build from source | for users who want to modify and tune WHIRL themselves: HIP SDK 7.2 + Visual Studio 2022 Build Tools (MSVC) + CMake + Ninja; see [building.md](docs/building.md) |

Support scope: WHIRL officially supports the Radeon 8060S and the R9700 and targets cards with **16 GB or more**. Other AMD GPUs are untested (the RX 9070 series has the R9700's chip with 16 GB; untested). Cards under 16 GB are not officially supported; you are welcome to fork the source and adapt it.

Our R9700 is connected as a USB4 / Thunderbolt eGPU; that works. A card in a direct PCIe slot may differ slightly where data crosses the host link (model loading, RAM / SSD tier restores), not in prefill or decode.

## Quick start

1. Download `whirl-0.2.3-windows-x64.zip` from [Releases](https://github.com/tsaipifong/whirl-llm/releases) and a model, for example [`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) saved in `C:\models\`.
2. Unzip and run, in PowerShell:

```powershell
Unblock-File .\whirl-0.2.3-windows-x64.zip
Expand-Archive .\whirl-0.2.3-windows-x64.zip -DestinationPath C:\whirl
cd C:\whirl\whirl-0.2.3-windows-x64
.\whirl.exe devices                                    # lists the Radeon 8060S (gfx1151) and/or the R9700 (gfx1201)
.\whirl.exe chat C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf "Explain TCP slow start in two sentences." --max-tokens 400
.\whirl-server.exe C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf --port 8080      # default mode: balance
```

The server speaks the OpenAI API at `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`), so any OpenAI-compatible client works. Add `--precise` for bitwise-lossless output or `--fast` for maximum speed. Stop the server with Ctrl+C; reused sessions are written to the SSD tier first.

For long documents give a slot a longer context, for example `--ctx-per-slot 262144`; see [recipes.md](docs/recipes.md#long-context).

### Choosing the GPU

Without `--device`, WHIRL uses the GPU with the most GPU memory (integrated GPUs count their dedicated memory; if
equal, the first one in `--list-devices`); the startup log shows which. To choose another, give `--device
pci:<bus>:<device>` (also `-dev`, or `WHIRL_DEVICE`) to `whirl` or `whirl-server`, with the numbers Task Manager >
Performance > GPU > Physical location shows.

Example: Task Manager shows Physical location: PCI bus 71, device 0, function 0 → start with
`whirl-server.exe <model.gguf> --device pci:71:0`

`whirl devices` (or `--list-devices`) prints the same `pci:` numbers for every GPU. An index, `r9700`, `8060s` or a
part of the name also work when only one GPU matches. An external USB4 / Thunderbolt GPU gets its bus number when it
is connected, so check Task Manager again after replugging it.

The first run with a model tunes the GPU kernels once and caches the result in `%LOCALAPPDATA%\whirl` (a few seconds for our MXFP4 files on the R9700; longer on the 8060S). Full walkthrough: [quick start](docs/quickstart.md). Every option: [usage reference](docs/usage.md).

**Advanced: build from source** — for users who want to modify and tune WHIRL themselves: [building.md](docs/building.md).

## Three modes: speed and precision

WHIRL 0.2.0 has three numerics modes. You pick one per process (`--precise`, `--balance`, `--fast`); with no flag it runs **balance**.

| Mode | What it is | Output guarantee |
|---|---|---|
| **precise** | **bitwise lossless**: nothing is quantized beyond the GGUF's own weights; f16 KV cache, f16 / f32 activations everywhere | output **bitwise identical** to plain greedy decoding: the same bits with or without speculative drafts, alone or batched with other users, and from a prefix-cache hit or a restored session |
| **balance** (default) | near-lossless, default | MTP / n-gram drafts and batching give the same tokens as plain greedy decoding in balance mode, and a restored session the same tokens as that session kept in VRAM; a conversation's next turn, continued from the cached end of the previous reply, is numerically equivalent to a cold run, not bitwise, so its reply can differ |
| **fast** (opt-in) | maximum speed within an accepted, gated loss: balance plus a 4-bit KV cache and relaxed MTP acceptance (on the Radeon 8060S the dense model keeps an int8 KV cache, which decodes faster there) | lossy within published KL limits; the text can depend on the drafts and on other users |

What this means in practice:

- **precise** is the mode with the bitwise guarantee: bitwise lossless, output identical to plain greedy decoding, on both GPUs.
- In **precise and balance**, every acceleration gives the same tokens as plain greedy decoding of that mode: MTP and n-gram speculative decoding, several users batched together, prefix-cache hits on shared system prompts, and a second question about the same long document asked in a fresh conversation (without the first answer); a session restored from the RAM or SSD tier gives the same tokens as that session kept in VRAM. Standing tests enforce it (`whirl selftest`, kernel tests, server gates, golden outputs on both GPUs). One difference between the two: in balance, a conversation's next turn continues from the cached end of the previous reply, whose history the decode kernels produced, so its reply can differ from what a fresh server gives for the whole conversation (numerically equivalent, KL up to about 2e-4; a restored session still equals the same session that never left VRAM, bit for bit). precise re-prefills from the previous prompt's last chunk start instead and stays bitwise identical to the cold run.
- **precise and balance never use 4-bit KV, 3-bit weights or fp8 attention.** fast is the only mode that can use 4-bit KV, and only when you ask for it.
- **Weights** are whatever GGUF you choose; WHIRL computes with them as stored. In balance and fast, continuing a conversation reuses the state computed for the previous turn, which is numerically equivalent but not bit-identical to re-processing the whole history as one prompt; in precise it is bit-identical ([kv-and-caching.md](docs/guide/en/kv-and-caching.md)).

<!-- gen:precise.en -->
precise costs speed. On the R9700 the biggest cost is prefill on MXFP4 files, which runs without fp8 in precise (Swift 27B at 2k: 1,702 vs 3,273 tok/s in balance; Ornith: 7,575 vs 11,550); in the coding chat Swift decodes 83.4 vs 109.6 tok/s. The 8060S has no fp8 matrix units, so there precise and balance prefill are close (479.1 vs 498.4, Swift at 2k). precise still leads both llama.cpp builds in every measured cell.
<!-- /gen -->

<!-- gen:fast.en -->
**fast mode is not fully optimized yet; it will improve in later 0.2.x point releases.** What it does today, measured: it mainly speeds up decode on the dense Swift 27B (R9700: 115.5 vs 87.6 tok/s in balance after a 2k-token document, 130.4 vs 109.6 in the coding chat; 8060S (max power (80 W), ASUS ROG Flow Z13): 37.7 vs 32.5 at 8k) and long-context decode on the Ornith MoE (R9700 at ~250k: 125.3 vs 101.7). On Ornith it is **slower than balance below about 64k tokens** (R9700: 208.3 vs 238.4 at 2k, 183.5 vs 199.1 at 32k; at 64k 174.9 vs 169.1; 8060S at 2k: 93.8 vs 101.4), because its 4-bit KV cache adds a fixed cost per token that only pays off at long context. These are 0.2.0 measurements; 0.2.2 makes Ornith fast on the Radeon 8060S 3-5% faster (QA document at 2k, 96k and 128k, and the coding chat; 32k within noise), level with balance at 2k, at 32k and in the coding chat (CHANGELOG). fast does not speed up prefill yet (2,790 vs 2,805 tok/s, Swift at 32k on the R9700).
<!-- /gen -->

Every mode in detail: [usage.md, numerics modes](docs/usage.md#modes).

## Supported models

> **Our own MXFP4 quantizations, made for WHIRL** (all published numbers use these files):
> - Dense 27B: [**Swift-1.5-Qwen3.8-27b-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF), variant A (`Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf`). Swift-1.5 also thinks much less than the base model (see the model card).
> - MoE 35B (~3B active): [**Ornith-1.5-35B-A3B-MXFP4-GGUF**](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF).

| Model file (Hugging Face) | Type | Arch | Quantization |
|---|---|---|---|
| [tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF) `…-A-outQ6_K.gguf` (recommended), `…-B-outQ8_0.gguf`, `…-C-outQ4_K.gguf` | dense 27B | qwen35 | MXFP4 (+ Q8_0 MTP / embeddings / `ssm_alpha` / `ssm_beta`); quantized by us for WHIRL |
| [tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) `Ornith-1.5-35B-A3B-MXFP4.gguf` | MoE 35B, ~3B active | qwen35moe | MXFP4 experts and dense (+ Q8_0 MTP / embeddings / `ssm_alpha` / `ssm_beta`, Q6_K head); quantized by us for WHIRL |

Both were loaded and checked on both GPUs (greedy output, MTP output identical to plain greedy in precise and balance). WHIRL reads `general.architecture` from the GGUF and accepts `qwen35` and `qwen35moe` (the Gated DeltaNet + gated attention hybrid, with an optional MTP head), so other GGUFs of these architectures load too, but they are not what we measure or tune for. Other architectures (Llama, Gemma, Mistral, DeepSeek, `qwen3`, `qwen2`, …) are refused at load time with a clear message, and so are tensor types WHIRL has no kernels for (for example NVFP4); there is no slow generic fallback.

Image input works with a Qwen3-VL-style mmproj (F16 / BF16) via `--mmproj`, on both GPUs; both model repositories include one. An opt-in DFlash2 drafter is available for the dense 27B model (`--draft dflash`, [below](#whats-new-in-020)). Choosing and making GGUFs: [quantization.md](docs/guide/en/quantization.md).

## Performance

WHIRL 0.2.0 against the official llama.cpp b11214 (ggml-org Windows binaries), ROCm and Vulkan builds side by side. Same GGUF files, same prompts (a document-QA prompt with a unique first line, so no prefix-cache hits; 256 generated tokens), greedy decoding, thinking off, one user, tok/s from each server's own timings. All three WHIRL modes are shown; the lead columns are for balance, the default. **precise and fast lead both llama.cpp builds in every cell as well.** Methods, flags and prompts: [benchmarks.md](docs/benchmarks.md).

<!-- gen:ranges.en -->
| WHIRL balance lead, every measured length | Prefill vs ROCm | Prefill vs Vulkan | Decode vs ROCm | Decode vs Vulkan |
|---|---|---|---|---|
| 8060S · Swift 27B | +53% to +65% | +93% to +183% | +28% to +46% | +24% to +105% |
| 8060S · Ornith 35B-A3B | +64% to +91% | +142% to +199% | +41% to +54% | +43% to +57% |
| R9700 · Swift 27B | +139% to +191% | +208% to +295% | +56% to +80% | +38% to +84% |
| R9700 · Ornith 35B-A3B | +75% to +188% | +119% to +225% | +74% to +107% | +78% to +106% |

Footnoted cells (ᵃ ᵇ ᶜ) are left out of these ranges; WHIRL leads in those cells too.
<!-- /gen -->

### <a id="radeon-8060s"></a>Radeon 8060S, max power (80 W), ASUS ROG Flow Z13

Ryzen AI Max+ 395. WHIRL one run per cell (2026-10-17); llama.cpp median of 1 to 3 runs.

<!-- gen:s8.swift.pp.en -->
#### Swift-1.5 27B MXFP4 (dense): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 479.1 | **498.4** | 502.6 | 305.7 | 162.8ᶜ | +63% | +206%ᶜ |
| 8k | 446.5 | **470.8** | 471.8 | 307.5 | 234.4 | +53% | +101% |
| 32k | 401.4 | **426.7** | 428.9 | 266.2 | 221.6 | +60% | +93% |
| 64k | 353.4 | **374.4** | 375.9 | 230.9 | 177.3 | +62% | +111% |
| 96k | 315.7 | **332.9** | 334.1 | 203.3 | 149.0 | +64% | +123% |
| 128k | 282.9 | **298.9** | 297.4 | 180.8 | 127.8 | +65% | +134% |
| ~250k | 197.0 | **221.3** | 222.2 | 135.1 | 78.1 | +64% | +183% |
<!-- /gen -->

<!-- gen:s8.swift.tg.en -->
#### Swift-1.5 27B MXFP4 (dense): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 30.0 | **31.2** | 37.1 | 24.4 (MTP+n-gram) | 25.1 (MTP+n-gram) | +28% | +24% |
| 8k | 30.8 | **32.5** | 37.7 | 24.6 (MTP+n-gram) | 25.2 (MTP+n-gram) | +32% | +29% |
| 32k | 25.5 | **29.5** | 34.7 | 22.7 (MTP+n-gram) | 17.0 (MTP+n-gram) | +30% | +74% |
| 64k | 23.9 | **26.0** | 31.2 | 18.4 (MTP+n-gram) | 18.0 (MTP+n-gram) | +41% | +44% |
| 96k | 21.8 | **24.1** | 29.6 | 17.9 (MTP+n-gram) | 16.6 (MTP+n-gram) | +35% | +45% |
| 128k | 22.4 | **23.8** | 26.1 | 16.2 (MTP+n-gram) | 11.6 (MTP+n-gram) | +46% | +105% |
| ~250k | 14.6 | **16.8** | 20.7 | 12.2 (MTP+n-gram) | 10.9 (MTP+n-gram) | +38% | +54% |
<!-- /gen -->

<!-- gen:s8.ornith.pp.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 2,044 | **2,411** | 2,401 | 1,259 | 447.7ᶜ | +91% | +439%ᶜ |
| 8k | 1,867 | **2,244** | 2,232 | 1,282 | 749.2 | +75% | +199% |
| 32k | 1,577 | **1,829** | 1,830 | 1,049 | 756.2 | +74% | +142% |
| 64k | 1,299 | **1,463** | 1,466 | 848.9 | 603.6 | +72% | +142% |
| 96k | 1,108 | **1,227** | 1,229 | 710.4 | 500.5 | +73% | +145% |
| 128k | 966.3 | **1,055** | 1,059 | 636.6 | 428.4 | +66% | +146% |
| ~250k | 644.7 | **685.0** | 697.9 | 418.3 | 278.9 | +64% | +146% |
<!-- /gen -->

<!-- gen:s8.ornith.tg.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 91.6 | **101.4** | 93.8 | 70.0 (plain) | 70.7 (plain) | +45% | +43% |
| 8k | 87.5 | **94.9** | 93.6 | 64.3 (plain) | 65.2 (plain) | +48% | +46% |
| 32k | 76.3 | **83.8** | 86.1 | 57.9 (plain) | 57.1 (plain) | +45% | +47% |
| 64k | 67.9 | **69.2** | 76.2 | 49.0 (plain) | 48.4 (plain) | +41% | +43% |
| 96k | 58.5 | **63.9** | 67.0 | 42.7 (plain) | 42.2 (plain) | +50% | +51% |
| 128k | 51.1 | **58.4** | 62.7 | 38.0 (plain) | 37.3 (plain) | +54% | +57% |
| ~250k | 38.9 | **40.4** | 48.9 | 26.3 (plain) | 25.8 (plain) | +54% | +57% |
<!-- /gen -->

<!-- gen:foot.s8.en -->
ᶜ 2k prefill on the 8060S, llama.cpp Vulkan: Vulkan prefill is much slower on short prompts on this GPU (well below its own trend at 8k+), so the 2k lead over Vulkan is not used as a headline; headline leads over Vulkan use 8k and longer.
<!-- /gen -->

### Radeon AI PRO R9700

One run per cell, WHIRL measured on 2026-10-17; treat differences under about 5% as noise. Decode: WHIRL MTP (+ n-gram); llama.cpp Swift 27B MTP + n-gram, Ornith plain (as shipped).

<!-- gen:r97.swift.pp.en -->
#### Swift-1.5 27B MXFP4 (dense): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 1,702 | **3,273** | 3,180 | 1,124 | 829.4 | +191% | +295% |
| 8k | 1,764 | **3,296** | 3,269 | 867.5ᵇ | 705.5ᵇ | +280%ᵇ | +367%ᵇ |
| 32k | 1,626 | **2,805** | 2,790 | 1,065 | 787.5 | +163% | +256% |
| 64k | 1,444 | **2,332** | 2,304 | 913.6 | 691.9 | +155% | +237% |
| 96k | 1,301 | **1,971** | 1,952 | 805.2 | 618.1 | +145% | +219% |
| 128k | 1,185 | **1,715** | 1,697 | 718.7 | 557.2 | +139% | +208% |
| ~250k | not measured ᵈ | **1,153** | 1,153 | 342.0ᵃ | 259.9ᵃ | +237%ᵃ | +344%ᵃ |
<!-- /gen -->

<!-- gen:r97.swift.tg.en -->
#### Swift-1.5 27B MXFP4 (dense): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 70.1 | **87.6** | 115.5 | 50.5 (MTP+n-gram) | 50.4 (MTP+n-gram) | +74% | +74% |
| 8k | 74.3 | **90.7** | 102.3 | 28.2ᵇ (MTP+n-gram) | 44.4ᵇ (MTP+n-gram) | +222%ᵇ | +104%ᵇ |
| 32k | 56.4 | **79.2** | 79.3 | 44.0 (MTP+n-gram) | 43.1 (MTP+n-gram) | +80% | +84% |
| 64k | 56.7 | **60.4** | 67.8 | 38.6 (MTP+n-gram) | 39.5 (MTP+n-gram) | +56% | +53% |
| 96k | 49.8 | **56.3** | 63.4 | 36.1 (MTP+n-gram) | 40.7 (MTP+n-gram) | +56% | +38% |
| 128k | 49.4 | **53.9** | 67.0 | 33.2 (MTP+n-gram) | 34.8 (MTP+n-gram) | +62% | +55% |
| ~250k | not measured ᵈ | **37.9** | 38.8 | 2.3ᵃ (MTP+n-gram) | 6.4ᵃ (MTP+n-gram) | +1548%ᵃ | +492%ᵃ |
<!-- /gen -->

<!-- gen:r97.ornith.pp.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Prefill tok/s

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 7,575 | **11,550** | 11,801 | 4,014 | 3,551 | +188% | +225% |
| 8k | 6,846 | **10,601** | 10,501 | 4,487 | 3,912 | +136% | +171% |
| 32k | 5,959 | **8,469** | 8,450 | 3,872 | 3,395 | +119% | +149% |
| 64k | 4,873 | **6,451** | 6,420 | 3,173 | 2,735 | +103% | +136% |
| 96k | 4,098 | **5,156** | 5,143 | 2,667 | 2,280 | +93% | +126% |
| 128k | 3,520 | **4,275** | 4,270 | 2,307 | 1,922 | +85% | +122% |
| ~250k | 2,351 | **2,664** | 2,666 | 1,519 | 1,218 | +75% | +119% |
<!-- /gen -->

<!-- gen:r97.ornith.tg.en -->
#### Ornith-1.5-35B-A3B MXFP4 (MoE, ~3B active): Decode tok/s, document QA (256-token answer: explain a function quoted in the document and rewrite it, so part of the answer can be copied)

| Prompt | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 2k | 181.8 | **238.4** | 208.3 | 115.4 (plain) | 115.9 (plain) | +107% | +106% |
| 8k | 174.6 | **212.9** | 191.3 | 112.3 (plain) | 111.9 (plain) | +90% | +90% |
| 32k | 154.2 | **199.1** | 183.5 | 101.4 (plain) | 100.6 (plain) | +96% | +98% |
| 64k | 138.6 | **169.1** | 174.9 | 90.7 (plain) | 89.6 (plain) | +86% | +89% |
| 96k | 126.4 | **148.1** | 156.2 | 81.9 (plain) | 81.1 (plain) | +81% | +83% |
| 128k | 121.3 | **131.3** | 152.4 | 75.3 (plain) | 73.8 (plain) | +74% | +78% |
| ~250k | 90.0 | **101.7** | 125.3 | 54.1 (plain) | 53.6 (plain) | +88% | +90% |
<!-- /gen -->

<!-- gen:foot.r97.en -->
ᵃ ~250k on the R9700 with Swift 27B: official llama.cpp with f16 KV does not fit in 32 GB at this length (its log: "failed to fit params to free device memory") and spills into shared memory, which collapses its decode. The numbers are shown as measured, but this cell is not used as a lead anywhere.

ᵇ 8k on the R9700 with Swift 27B: the llama.cpp 8k request was the first long request on a fresh server (cold), and it came out below its own 2k and 32k values. This cell is not used as a lead anywhere.

ᵈ Swift 27B precise at ~250k on the R9700: not measured. precise keeps the KV cache in f16, and f16 KV for this length does not fit next to the weights in 32 GB (the f16 pool holds about 190k tokens).
<!-- /gen -->

### Coding chat (new text)

512 tokens; the answer (explanation and code) is all newly written and the prompt holds nothing to copy beyond a few option names, so this is a hard case for speculative drafts. WHIRL: one run on the 8060S, median of 3 runs on the R9700. llama.cpp: first run only, because its n-gram drafts learn a repeated prompt.

<!-- gen:chat.en -->
| GPU · model | WHIRL precise | WHIRL balance (default) | WHIRL fast | llama.cpp ROCm | llama.cpp Vulkan | balance vs ROCm | balance vs Vulkan |
|---|---|---|---|---|---|---|---|
| 8060S · Swift 27B | 36.3 | **34.2** | 42.1 | 27.6 (MTP+n-gram) | 28.3 (MTP+n-gram) | +24% | +21% |
| 8060S · Ornith 35B-A3B | 94.5 | **106.0** | 102.8 | 70.6 (plain) | 72.9 (plain) | +50% | +45% |
| R9700 · Swift 27B | 83.4 | **109.6** | 130.4 | 57.8 (MTP+n-gram) | 57.6 (MTP+n-gram) | +90% | +90% |
| R9700 · Ornith 35B-A3B | 191.2 | **241.5** | 219.0 | 118.9 (plain) | 115.1 (plain) | +103% | +110% |

8060S coding chat: one run per WHIRL mode. How many drafts are accepted depends on the text each mode writes, so in this run Swift 27B precise (36.3) came out above balance (34.2).
<!-- /gen -->

### Where WHIRL does not lead by much

<!-- gen:small.en -->
- **Smallest leads: dense decode on the Radeon 8060S.** Swift 27B in the coding chat (new text): 34.2 vs 28.3 tok/s on llama.cpp Vulkan (+21%) and 34.2 vs 27.6 on ROCm (+24%); after a 2k-token document: 31.2 vs 25.1 tok/s on llama.cpp Vulkan (+24%); 31.2 vs 24.4 tok/s on llama.cpp ROCm (+28%). Dense decode there is limited by memory bandwidth, and the lead comes from speculative decoding.
<!-- /gen -->
- **Ornith decode is compared as shipped**: WHIRL with MTP drafts, llama.cpp without speculation (its MTP was slower on this model).
- **Plain decoding without speculation** is memory-bandwidth bound in every engine; most of WHIRL's decode lead comes from MTP + n-gram speculative decoding.
- **fast is not always faster than balance** (see [Three modes](#three-modes-speed-and-precision)), and **precise costs speed**, most of all in R9700 prefill.

## What's new in 0.2.0

- **Radeon 8060S (Ryzen AI Max+ 395) support**, tuned: faster than llama.cpp ROCm and Vulkan at every measured prompt length on both models (tables above). Image input works on the 8060S, and the RAM and SSD cache tiers are on by default there (on Windows the BIOS VRAM carve-out and the OS RAM are separate pools, so a 128k-token session is restored from RAM or SSD instead of being prefilled again).
- **Three numerics modes**: precise (bitwise lossless), balance (default, near-lossless), fast (opt-in, gated loss). See [above](#three-modes-speed-and-precision).
- **Our MXFP4 models**: Swift-1.5 27B MXFP4 and Ornith-1.5-35B-A3B MXFP4, quantized for WHIRL ([why](#why-whirl-is-fast-with-these-mxfp4-files)).
- **DFlash2 drafts, opt-in** (`--draft dflash --draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf`, drafter from Hugging Face `incoai/Qwen3.8-27B-DFlash2-GGUF`, not bundled): faster than MTP for one user with short or medium context and predictable output; slower with 4 users, at long context and on mostly-Chinese output; costs about 1.9 GiB of VRAM. Greedy output equals plain greedy decoding of the same mode.
- **SSD cache writes far less**: by default it writes only reused sessions, when the server stops (or after about 5 minutes idle, or when RAM must evict them); a one-shot prompt is never written. The cache never exceeds `--kv-ssd-gb` (default 64 GiB) on disk, and files unused for 7 days are deleted.
- **A second question about the same long document reuses it**: the server keeps a tail checkpoint, so it prefills only the new question (about 0.3k–1.3k tokens) instead of the whole document.
- **CLI and server give bit-identical output** for the same prompt (same prefill schedule; two documented exceptions in [kv-and-caching.md §7](docs/guide/en/kv-and-caching.md)).

Full list: [CHANGELOG.md](CHANGELOG.md).

## Windows security prompts

`whirl.exe` and `whirl-server.exe` are not code-signed. Windows SmartScreen may show "Windows protected your PC" for a freshly downloaded copy: click **More info → Run anyway**, or run `Unblock-File` on the zip before extracting it (only for a zip from the official releases page, after checking its SHA-256). With **Smart App Control** set to *On*, Windows may block the programs without a "Run anyway" option. Details: [windows_security.md](docs/windows_security.md).

## Documentation

| | English | 繁體中文 |
|---|---|---|
| Quick start | [quickstart.md](docs/quickstart.md) | [quickstart_zh-TW.md](docs/quickstart_zh-TW.md) |
| Recipes: connect a client, long agent sessions, long context, several users, images, reading the log, troubleshooting | [recipes.md](docs/recipes.md) | [recipes_zh-TW.md](docs/recipes_zh-TW.md) |
| Usage reference (all commands, options, numerics modes, environment variables, exit codes) | [usage.md](docs/usage.md) | [usage.md](docs/guide/zh-TW/usage.md) |
| Server (API, sampling, tool calls, batching, decode floor, logs) | [server.md](docs/guide/en/server.md) | [server.md](docs/guide/zh-TW/server.md) |
| Benchmarks vs llama.cpp | [benchmarks.md](docs/benchmarks.md) | [benchmarks.md](docs/guide/zh-TW/benchmarks.md) |
| Windows security prompts | [windows_security.md](docs/windows_security.md) | [windows_security_zh-TW.md](docs/windows_security_zh-TW.md) |
| Advanced: building from source | [building.md](docs/building.md) | [building_zh-TW.md](docs/building_zh-TW.md) |
| Guide index | [index.md](docs/guide/en/index.md) | [index.md](docs/guide/zh-TW/index.md) |
| Architecture | [architecture.md](docs/guide/en/architecture.md) | [architecture.md](docs/guide/zh-TW/architecture.md) |
| Kernels | [kernels.md](docs/guide/en/kernels.md) | [kernels.md](docs/guide/zh-TW/kernels.md) |
| Speculative decoding (MTP + n-gram, DFlash2) | [speculative-decoding.md](docs/guide/en/speculative-decoding.md) | [speculative-decoding.md](docs/guide/zh-TW/speculative-decoding.md) |
| KV cache, prefix caching, tiers | [kv-and-caching.md](docs/guide/en/kv-and-caching.md) | [kv-and-caching.md](docs/guide/zh-TW/kv-and-caching.md) |
| Quantization and model files | [quantization.md](docs/guide/en/quantization.md) | [quantization.md](docs/guide/zh-TW/quantization.md) |
| Image input | [vision.md](docs/guide/en/vision.md) | [vision.md](docs/guide/zh-TW/vision.md) |
| HIP on Windows | [windows-hip.md](docs/guide/en/windows-hip.md) | [windows-hip.md](docs/guide/zh-TW/windows-hip.md) |
| Benchmarking methodology | [benchmarking.md](docs/guide/en/benchmarking.md) | [benchmarking.md](docs/guide/zh-TW/benchmarking.md) |
| **Pitfalls** (each with symptom → root cause → fix → check) | [pitfalls.md](docs/guide/en/pitfalls.md) | [pitfalls.md](docs/guide/zh-TW/pitfalls.md) |

## Design principles

1. **Built for AMD GPUs on Windows.** A native Windows C++/HIP inference engine with hand-written, hand-tuned kernels for RDNA. No WSL, no Docker, no Linux VM: install the AMD driver and run.
2. **Fully open, kernels included.** Everything from host-side scheduling to every GPU kernel is open source under Apache-2.0, with no closed binary components. Bug reports, benchmark results and code contributions are all welcome.
3. **Accuracy before speed.** In the precise and balance modes every speedup (speculative decoding, batching, prefix caching, cache restore) gives the same tokens as plain greedy decoding of that mode (in balance, a conversation's next turn is numerically equivalent to a cold run, not bitwise, so its reply can differ), enforced by standing tests. Anything lossier is opt-in (`--fast`) and gated by KL and accuracy limits.
4. **Few models, pushed to the limit.** WHIRL targets specific models on specific hardware and pushes them close to the hardware limit, adding architectures one at a time. The architectures supported today are `qwen35` (dense) and `qwen35moe` (mixture of experts).
5. **Transparent, verifiable numbers.** Every figure is measured, with methods, commands and prompts published, including where WHIRL does not lead by much.
6. **Designed for real workflows.** Coding agents and long contexts come first: fast long-prompt processing, a responsive main conversation while others share the GPU, and a VRAM → RAM → SSD cache that survives server restarts.

WHIRL follows the design philosophy of [NInfer](https://github.com/Neroued/ninfer) (Apache-2.0) — build for selected models and selected GPUs, as close to the hardware limit as possible; [zynfer](https://github.com/thanos/zynfer) carried the same idea to RDNA 4.

## Development and AI assistance

WHIRL was designed, implemented, optimized and benchmarked with the participation of **Claude Opus 5.5**, an AI model by Anthropic, working under the direction of the project owner ([@tsaipifong](https://github.com/tsaipifong)). Anthropic is not affiliated with and does not endorse this project.

## License and acknowledgements

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), the origin of every source file in [PROVENANCE.md](PROVENANCE.md).

- [NInfer](https://github.com/Neroued/ninfer) — the design philosophy WHIRL follows (ideas only).
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — used as the comparison baseline in our benchmarks and as the reference for GGUF and quantization-format compatibility; the few adapted pieces (tokenizer pre-split / BPE loop, lookup tables, image preprocessing) are MIT-licensed and listed in THIRD_PARTY_NOTICES.md.
- [gufo](https://github.com/gufo-org/gufo) and [r9700-stack](https://github.com/bkvargyas/r9700-stack) — measurement methodology and optimization ideas.
- Model authors: the Qwen team (Qwen3.8), [ornith-ai](https://huggingface.co/ornith-ai) (Ornith-1.5), and [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) (Swift-1.5, the fine-tune our MXFP4 files are quantized from).
