**English** | [繁體中文](../zh-TW/kv-and-caching.md)

# KV cache, checkpoints and tiered caching

> **Status.** Implemented in the C++ server (`src/server/`, `src/tier/`). Every flag and `WHIRL_*`
> variable is listed in [usage.md](../../usage.md). All numbers: Qwen3.8-27B on the R9700 (32 GB)
> unless noted.

**Who this helps:** anyone serving hybrid attention + recurrent models (DeltaNet, Mamba-like),
where prefix caching needs recurrent-state checkpoints and not just KV blocks; anyone budgeting
KV formats on a 32 GB card; anyone building a RAM/SSD KV tier on Windows.

## 1. The memory budget

| Item | Size |
|---|---|
| Weights (UD-Q4_K_M) | 15.32 GiB |
| KV per token, f16 (16 attention layers + MTP layer = 17 × K/V × 4 KV heads × 256 × 2 B) | 68 KiB |
| KV per token, `q8v` (K f16, V int8) | 52.1 KiB |
| KV per token, `q8` / `q8h` (K and V int8) | 36.1 KiB |
| KV per token, MoE model (10 attention layers + MTP, 2 KV heads), f16 | 22 KiB |
| One DeltaNet checkpoint (48 layers of conv + recurrent state, plus an MTP hidden row and a logits row) | 150.6 MiB (MoE: 63.8 MiB) |
| 131,072 tokens of f16 KV | 8.5 GiB |

The DeltaNet state itself is a fixed size per sequence, independent of context length. The server
prints its VRAM split at startup; one example (four slots, before the RAM tier existed):
weights + prefill buffers 17.85 GiB (the batch-4096 per-row scratch is ~538 KB/row, ~2.1 GiB),
slot recurrent state + replay buffers 0.82 GiB, other 0.31 GiB, prefix checkpoints 1.56 GiB
(4 slots × 2), KV pool 10.41 GiB, reserve 0.75 GiB.

## <a id="paged"></a>2. A paged KV pool shared by all slots

**Why not virtual memory.** The first plan was to give each slot a contiguous virtual KV range and
map physical pages on demand, so kernels would not change. On Windows/WDDM, kernels only saw the
first physical allocation in a mapped range ([windows-hip.md](windows-hip.md#vmm)), so WHIRL uses a
real page table.

**Layout.** Each layer's K and V live in a pool of pages of **256 tokens**. Position p of a
sequence lives at pool row `ptab[tab + p / 256] * 256 + p % 256`. Kernels receive one by-value
`KvArgs { k, v, ks, vs, ptab, kvbase, tab0 }`: batched rows use a per-row page-table offset
(`kvbase[row]`), single-sequence launches use `tab0`. Every key tile (32 or 64 keys) is aligned
inside one page, so the page table is read once per tile. Decode, verify, prefill and the MTP
layer all use the same scheme.

- The f16 paged version is **bit-identical** to the old contiguous cache (both models × 4 prompt
  types: prefill logits, MTP text, plain text).
- The lookup costs **0.3–0.5%** of decode time (e.g. 64k plain decode 35.8 vs 35.7 ms/token).

**Server pool management.** A free-page stack; each slot owns a page list (kept while idle, for
prefix reuse). Before every prefill chunk and every decode cycle a slot ensures pages up to
`pos + drafts + 2`. When the pool is empty the least recently used *idle* slot is evicted (its
prefix cache is dropped and its pages reclaimed — or, with tiers enabled, spilled first, §8).
If every page is held by running requests, the request ends with `finish_reason: length` (decode)
or HTTP 500 (prefill). Pages carry reference counts so that shared prefixes (§5) can be mapped
read-only into several slots; pages still being read by a background copy go to a quarantine
list and return to the free list only after the copy's fence.

- `--ctx N` = pool size in tokens; `--ctx-per-slot N` = per-request maximum.
- Default pool = all VRAM left after weights, buffers, slot state and checkpoints, minus a reserve:
  **768 MiB for dense models, 1.5 GiB for MoE** (the MoE model allocated ~0.5 GiB late and
  overflowed into shared memory with a smaller reserve).

## <a id="formats"></a>3. KV formats: f16, q8, q8h, q8v

| Format | K | V | Bytes/token (27B) | Notes |
|---|---|---|---|---|
| `f16` | f16 | f16 | 68 KiB | exact |
| `q8` | int8 + f16 scale per 32 | same | 36.1 KiB | |
| `q8h` | as q8, after a 256-dim Walsh–Hadamard rotation of q and k | int8 | 36.1 KiB | rotation spreads K's outlier channels |
| `q8v` | f16 | int8 + f16 scale per 32 | 52.1 KiB | **current default for the dense model** |

**Quantization happens once, at write.** In `kv_st32`, one wave of 32 lanes is exactly one scale
group: wave max → `scale = amax / 127` stored as f16 → `q = round(x / scale)`. Every read
dequantizes with one f16 multiply; WMMA still consumes f16. Kernels are templated on the format
(f16 / q8 / q8v), so f16 stays available.

**Hadamard (`q8h`).** After RoPE, `attn_prep` applies an orthonormal 256-point fast Walsh–Hadamard
transform to every q and k head. q·k is invariant under the rotation in exact arithmetic; K's
outlier channels get spread across all dimensions, which suits a per-32 scale. Cost: 8 butterfly
stages inside the fused prep kernel. 27B KL dropped another 2–3×.

### 3.1 Quality

KL of the last-token distribution vs f16 KV, same binary. "Path noise" is the KL between two
equally correct f16 paths (chunked vs sequential DeltaNet prefill); prefill chunk 4096 vs 512 is
0 (bit-identical) for both models.

| Prompt | 27B path noise | 27B q8 | 27B q8h | MoE path noise | MoE q8 | MoE q8h |
|---|---|---|---|---|---|---|
| arch 1,107 tok | 6.9e-8 | 2.1e-7 | 2.3e-7 | 7.3e-4 | 5.7e-4 | 9.4e-4 |
| p4k | 1.8e-5 | 6.4e-4 | 3.1e-4 | 1.5e-2 | 6.1e-2 (top-1 flip 0.48→0.39) | 2.0e-2 |
| p12k | 1.7e-6 | 3.4e-6 | 1.1e-6 | 3.1e-3 | 2.4e-1 (p 0.73→0.41) | 1.6e-1 |
| p24k | 2.5e-7 | 1.7e-6 | 5.4e-7 | 1.1e-3 | 4.1e-4 | 1.1e-3 |

Later comparison of q8v and q8h (27B, vs f16): q8v had lower KL on 8 of 10 prompts (mostly 2–10×
lower), e.g. code32k 2.0e-4 vs 3.8e-4, long64k 4.7e-4 vs 1.0e-3, code64k 7.4e-6 vs 9.3e-5;
top-1 identical and top-10 10/10 on all.

**Paired QA** (350 items: 150 "what does this Python print" checked by executing it, 100 multi-step
integer word problems, 100 key/value retrievals from 17k–70k-token C++ sources with `_OLD` decoys;
greedy, reasoning before `ANSWER:`; McNemar exact test):

| | f16 | q8h | q8 | q8v |
|---|---|---|---|---|
| 27B total | 265/350 | 263 (p = 0.63) | 266 (p = 1.0) | 264 (p = 1.0, separate run) |
| 27B code / arithmetic / retrieval | 67 / 98 / 100 | 65 / 98 / 100 | 67 / 99 / 100 | 66 / 98 / 100 |
| MoE total | 333/350 | 334 (p = 1.0) | — | — |

No format showed a measurable task-level difference (retrieval 100% everywhere, including 70k
contexts with decoys). An earlier QA version that asked for the answer only was too insensitive
(5–7% correct for both models) and was replaced.

**Needles:** 27B with q8h, 128k at 10%/50%/90% depth and 256k at 10%/90% depth all found
(256k: 261,909-token prompt, prefill 398 tok/s, MTP decode 30 tok/s, dedicated VRAM 29.1 GiB).

### 3.2 Speed — the surprise

| Plain decode, ms/token (CLI, same binary) | 16k | 64k | 128k |
|---|---|---|---|
| f16 | 30.0 | 35.8–35.9 | 42.0–42.1 |
| q8h | 30.3 (+1.0%) | 37.1–37.2 (+3.6%) | 44.2–44.3 (+5.2%) |
| q8v | — | 35.7 (−0.3%) | 41.7 (−1.0%) |

| Prefill, tok/s | 64k | 128k |
|---|---|---|
| f16 | 1125.7–1134.5 | 926.6–927.0 |
| q8v | 1102.9–1105.4 (−2.3%) | 891.0–891.1 (−3.9%) |
| q8h | 1100.2–1100.8 (−2.6%) | 890.2–891.3 (−3.9%) |

Halving KV bytes made decode **slower**: the int8 dequantization of K inside the split-K decode
kernel costs more than the bandwidth it saves (decode attention is not purely bandwidth-bound once
dequant is added). The decode cost is entirely in K; quantizing only V (`q8v`) decodes as fast as
f16 or slightly faster. The prefill cost is in V (the PV product), where q8v and q8h pay the same.

### <a id="kv-auto"></a>3.3 The automatic policy

- **MoE model: always f16.** Its single-prompt KL under q8 was far above its own path noise (77× at
  p12k; 50× even with q8h), so precision mode keeps f16, even though QA showed no difference.
  `q8h` remains a QA-validated option for a larger MoE pool. (With 22 KiB/token the MoE f16 pool is
  already ~340k tokens.)
- **Dense model:** the first of **f16 → q8v → q8h** whose *floor pool* fits (§4). With the default
  four slots that is q8v; `--parallel 1` gets f16; `--ctx-per-slot 262144` gets q8h.
- The CLI decides from its maximum context the same way (16k and 128k CLI runs use f16).
- Forcing a format (`WHIRL_KV=f16|q8|q8h|q8v`) prints a warning if the pool ends up below the floor.

## <a id="floor"></a>4. The 128k + 64k floor rule

Project policy: with the default four slots, one session must be able to reach **128k** tokens
while another running session can still reach **64k**. Idle sessions can be evicted. Floor pool =
131,072 + 65,536 + 2 decode pages per request = **197,120 tokens**.

| KV format | Pool with default `--parallel 4` | Meets floor |
|---|---|---|
| f16 | 160,512 | no (2.37 GiB short) |
| **q8v** | 209,664 (at introduction) → **199,936** today | yes |
| q8h | 302,080 | yes, but decode +3.6% (64k) / +5.2% (128k) |

The pool shrank since q8v was introduced because later features reserve VRAM before the pool is
sized: the tier stream and pinned arena (12.8 MiB visible to HIP), two shared system-prompt
checkpoints (2 × 150.6 MiB, −7,424 tokens), and a larger code object (−256 tokens). Headroom over
the floor is now **2,816 tokens**; any new resident buffer must be paid for elsewhere or the
default falls back to q8h.

Lossless VRAM savers considered for keeping f16: prefill batch 4096 → 3072 saves 0.52 GiB at no
speed cost (3072 vs 4096: 4k +1.2%, 16k −0.7%, 64k 0.0%; 2048 costs 1.3–2.0%); aliasing scratch
buffers (estimated ≤ 0.75 GiB, not done); halving checkpoints (0.78 GiB, hurts prefix caching).
Together not enough for f16, so q8v. Also measured and rejected: prefill batch 8192 (no faster on
long prompts, which are attention-bound; +2.1 GiB), embedding or checkpoints in pinned host memory
(checkpoint save/load took ~40 ms as 96 small copies at ~3.75 GB/s, agent turns −25%; and pinned
memory shows up as Shared Usage, [windows-hip.md](windows-hip.md#shared-usage)).

Server defaults that follow: dense 27B → `--parallel 4`, ≤ 131,072 tokens per request
(`--ctx-per-slot`, max 262,144), q8v. MoE → f16, 131,072 per request, pool ~341k tokens.

## <a id="checkpoints"></a>5. Prefix caching for a hybrid model: checkpoints

A transformer can reuse any prefix whose KV blocks are cached. A DeltaNet layer cannot: its state
at position p cannot be reconstructed from KV pages. Reuse is only possible at positions where the
engine **saved a checkpoint**: the conv and recurrent state of every DeltaNet layer, plus the MTP
hidden row (needed to continue drafting) and, where meaningful, the logits row (`has_logits`:
needed if the new prompt ends exactly there).

| Checkpoint kind | Saved at | Purpose |
|---|---|---|
| `prompt-end` | end of the prompt | same prompt again / next turn if generation is discarded |
| `gen-end` | after the last generated token | next turn of a conversation |
| `think-open` | just after `<think>\n` at the end of a thinking prompt (no logits) | next turn when the client drops reasoning |
| `system` | end of the system message (≥ 2048 tokens) | new sessions with the same system prompt |
| `prefix` | a chunk start inside a long common prefix | system prompts that end with per-session text |

Each slot holds 2 checkpoints when `--parallel` > 1, 4 otherwise. A new request is placed on the
slot with the longest reusable prefix; KV is reused by truncation and the slot's state is loaded
from the best checkpoint at or before that point.

### <a id="think-open"></a>5.1 Why multi-turn caching missed, and the fixes

Server logs showed two root causes, not the single one first suspected:

1. **MTP accepted drafts past EOS** ([speculative-decoding.md](speculative-decoding.md#eos)): the
   state ran 1–3 tokens past `<|im_end|>`, so `gen-end` lay outside the next turn's common prefix
   (example: `cache now 42`, next turn's common prefix 41).
2. **Thinking prompts end with `<think>\n`.** When the client does not send back
   `reasoning_content` (most agent clients don't), the history is re-rendered as
   `<think>\n\n</think>\n\n…`, and `\n\n` is a single token that differs from the prompt's final
   `\n`. The `prompt-end` checkpoint missed by one token, so **every turn re-prefilled the whole
   conversation** (hit ratio 0.000). Fix: split the prefill of such prompts at N−1 and save a
   `think-open` checkpoint after `<think>`. The CLI splits identically, so server == CLI stays
   bit-exact; cost: one extra 1-token step on thinking prompts (~28 ms on the 27B).

| Multi-turn hit ratio (later turns: cached / prompt) | 27B before → after | MoE before → after |
|---|---|---|
| agent (4k system prompt + 2 tools) | 0.961 → 0.972 | 0.968 → 0.963 |
| thinking, reasoning sent back | 0.681 / 0.447 → 0.862 | 0.923 → 0.848 |
| no thinking | 0.742 → 0.759 | 0.754 → 0.747 |
| agent, thinking, reasoning dropped | **0.000 → 0.928** | **0.000 → 0.914** |
| thinking, reasoning dropped | **0.000 → 0.515** | **0.000 → 0.523** |

(Short conversations have low ratios by nature; what matters is that mismatches went to zero.)
After the tokenizer whitespace fix, the agent case rose to 0.978 / 0.975.

A remaining, accepted limitation: the model sometimes generates a token sequence that is not the
tokenizer's canonical split of the same text (e.g. around `"""'` in code). Re-tokenizing that
reply diverges mid-turn and the next turn falls back to the previous `prompt-end` (one reply's
prefill lost). Fixing it would mean feeding generated tokens instead of re-tokenized text, which
makes cached and uncached inputs differ — not worth it. llama.cpp has the same behavior.

**Cached vs uncached is not bit-identical by design:** reused history KV was computed by decode
kernels, re-prefilled history by prefill GEMMs. The diagnostic mode re-prefills after each cache
hit and requires KL ≤ 1e-2 and identical top-1 (27B KL ≤ 9e-5 in practice).

## <a id="system-ckpt"></a>6. System-prompt checkpoints: new sessions skip the prefill

Agent clients send 10–30k tokens of tool schemas and skills text as the system prompt, and every
new session repeats it. Without a checkpoint at the end of the system message, each new session
re-prefilled all of it.

- **Boundary.** If the prompt starts with `<|im_start|>system`, B = the position after the first
  `<|im_end|>\n<|im_start|>` (the tools block is inside the system message, so it is included). B
  depends only on preceding tokens, so every prompt with the same system message + tools gets the
  same B. The first version searched for the first `<|im_end|>` alone; a 30k system prompt that
  quoted tokenizer documentation contained a literal `<|im_end|>`, parsed as the special token, and
  the boundary landed in the wrong place. The three-token pattern fixed it (a system text that
  literally contains `<|im_end|>\n<|im_start|>` still puts B inside the text — exact, just a few
  hundred tokens less reuse).
- **Always split at B.** When B ≥ 2048, prefill is *always* run as two chunk runs, [0, B) and
  [B, N), cold requests included. So a cold run and a run resumed from B execute identical chunks →
  bit-identical. (A 1024-aligned checkpoint that left cold runs unchanged would have recomputed
  ~512 tokens on average, ~0.35 s — worse than the +0.5% cost of the split.)
- **Shared checkpoint.** When a prefill chunk ends at B, the server saves a `system` checkpoint
  (state + MTP hidden row, two kept in VRAM by default) and holds references to the KV pages of
  [0, B): pages before the one containing B−1 are shared via refcounts; that last page is copied
  (the slot will write after B). A new request whose best reuse is the shared checkpoint maps the
  shared pages read-only into its page table, copies the last page, loads the state and starts
  prefilling at B. Any slot about to write at or beyond its reuse point first replaces shared pages
  with private ones (copy-on-write).
- **`prefix` checkpoints** (a simplified automatic prefix caching). Some clients append the date,
  working directory or git status to the system prompt, so the system message differs per session.
  For a new request, the server computes the longest common prefix L with every known token
  sequence (slots, shared checkpoints, tier entries); if L ≥ 2048 it saves a `prefix` checkpoint at
  the last of its **own** chunk starts ≤ L (already a chunk boundary, no extra split). A later
  session uses it only if that position is also a chunk start in its own schedule, so it stays
  bit-identical to a cold run. The second session creates it, the third onward benefits. Per-page
  hashing as in vLLM's APC is impossible here: every reuse point needs a 150 MiB state.
- **Bursts.** A queued request whose system message is currently being prefilled by another request
  waits for that checkpoint instead of recomputing it in parallel (parallel sub-agents).
- **Tiers.** Shared checkpoints are spilled to the RAM/SSD tiers (§8) and survive restarts. A bug
  found here: the spill read the checkpoint on the tier stream before the main-stream copy that
  produced it had finished — fixed with an event wait.

Cost: two VRAM checkpoints shrink the default pool by 7,424 tokens; cold prefill +0.5% (MXFP4 30k
+0.5…2%).

| New session's first-turn TTFT | Before (always cold) | After: later sessions | After restart (SSD restore) |
|---|---|---|---|
| Q4_K_M, 13.1k-token system prompt | 10.22–10.30 s | **0.153–0.185 s** | 0.45–0.48 s |
| Q4_K_M, 30.5k | 25.51–25.72 s | **0.567–0.608 s** | 1.27–1.28 s |
| MXFP4, 13.1k | 4.31–4.40 s | **0.085–0.110 s** | 0.40 s |
| MXFP4, 30.5k | 11.79–11.98 s | **0.361–0.379 s** | 1.06–1.07 s |

Agent scenario (8 sessions, each "ask about a file → `read_file` → answer", shared 14k-token system
prompt + 12 tools): sequential wall time 150.6 → 65.3 s (−56.6%), 4 concurrent 123.8 → 54.1 s
(−56.3%); first-turn TTFT mean 11.00 → 1.50 s and 27.68 → 1.90 s.

Trade-off: server requests with a ≥ 2048-token system message are no longer bit-identical to
binaries that did not split at B (numerically equivalent, like `think-open`); the CLI is unaffected.

## <a id="merge"></a>7. Forward merging versus checkpoint granularity

Bigger prefill forwards are faster, but checkpoints (`prefix`, `system`, `think-open`) can only sit
at chunk boundaries. Making the server's chunk 2048 rows gave +18…+21% cold prefill, but cross-slot
follow-ups then reused a `prefix` checkpoint at position 27,418 instead of 28,442, and a 30k
context + 300 / + 1000 new tokens went from 980 → 1681 ms and 1548 → 2230 ms TTFT.

The adopted design keeps the **schedule** at 1024-token chunks (checkpoint positions unchanged) and
**executes** consecutive whole chunks in one forward of up to 2048 rows (+256 tail) when no other
slot is decoding, never across a checkpoint boundary. This relies on logits being independent of
forward size — verified: last-token logits identical for batch 1024 / 2048 / 4096, both models,
MTP on/off.

| Server, MTP + n-gram | Q4_K_M before → after | MXFP4 before → after |
|---|---|---|
| Cold prefill 8k / 30k (tok/s) | 1304 → 1560 (+19.6%) / 1196 → 1413 (+18.1%) | 3135 → 3235 / 2595 → 2637 |
| 30k cached + 1000 new tokens, cross-slot | 1540 → 1376 ms | 849 → 833 ms |

## <a id="tiers"></a>8. Tiered prefix cache: VRAM → pinned RAM → SSD

When a session's slot is taken by other sessions, or the server restarts, its context used to be
re-prefilled from scratch (126k tokens: 147 s). The tiers turn that into a restore.

### 8.1 What is stored

An entry is a snapshot of one idle slot: token ids up to the last checkpoint position P, that slot's
checkpoints (DeltaNet conv/recurrent state, MTP hidden row, logits row), and the KV pages for
positions 0..P−1 (256-token pages, every attention layer and the MTP layer, every array of the KV
format: K, V, K scales, V scales). **A restore copies the same bytes back**, so a restored slot is
bit-identical to one that never left VRAM.

Layout (same in RAM and in the file): checkpoint k at `k × ck_stride` (150.6 MiB rounded to 2 MiB),
then KV page j at `nck × ck_stride + j × page_bytes` (13.0 MiB per page for q8v, 17.0 MiB for f16).

### 8.2 RAM tier

- A pinned arena allocated once at startup in 512 MiB pieces and managed in 2 MiB blocks; entries
  map to block lists; LRU eviction (an entry already on SSD falls back to its SSD copy, otherwise it
  is dropped).
- All GPU↔host copies run on one **non-blocking tier stream** (FIFO) in pieces ≤ 1 MiB, so decode's
  small readbacks are never queued behind a long transfer. Kernels writing host memory directly
  (zero-copy) did not work on this machine ([windows-hip.md](windows-hip.md#zero-copy)).
- Device-wide synchronization had to go: `hipDeviceSynchronize` waits for tier copies.
- Default size (`src/tier/ram_size.h`): **1/4 of physical RAM** (nearest GiB), at least the old
  minimum max(8 GiB, one full-length f16 session + checkpoints) = 9 GiB for 27B at 128k, at most
  32 GiB (the minimum wins over the cap), and at most half of the RAM available at startup (whole GiB;
  logged as a warning when it bites; under 1 GiB the tier is off) so pinning does not push the
  machine into paging. **16 GiB on a 64 GB PC.** `--kv-ram-mb N` / `WHIRL_KV_RAM_MB` are used as
  given; on an integrated GPU (shared system memory) the tier is off unless a size is given. Why: in
  agent use (Hermes, 30–40k-token sessions, subagents) the 9 GiB default was 8.3 GiB full within an
  hour, after which LRU entries fell back to their slower SSD copies. The startup log line
  `kv tier: RAM tier size N MiB (reason)` shows the choice; the summary line shows the pinning time
  (R9700 host, 64 GB: 16 GiB pinned in 3.1–3.4 s; "model ready" 13.6–14.1 s vs 11.8–12.3 s with the old 9 GiB and 10.2 s with the tier off, so the arena is still allocated up front).
  It appears as Shared Usage in GPU counters; the server declares its pinned size so monitoring can
  subtract it.

### 8.3 SSD tier

- One file per entry (4 KiB header, token ids, data region). An I/O thread writes only dirty blocks
  with unbuffered positional writes: header marked invalid → data → flush → valid header. An entry
  is written after it has been unchanged for 2 s.
- At startup the directory is scanned and indexed; magic, fingerprint and token checksum are
  checked and bad files deleted. Over the size cap (default 64 GiB) the LRU file is deleted.
- **Fingerprint:** executable (size + modification time), model file name + tensor count + bytes,
  KV format, MTP on/off, layout, prefill batch, and every engine setting that changes numerics
  (scheduling/logging/tier settings excluded). An entry written by a different build or numerical
  configuration is never restored.
- Reads: up to 4 overlapped reads in flight, completed in order (NVMe 2 MiB QD4 7.0 GB/s vs QD1
  4.1 GB/s); the restore is pipelined — as soon as a block is read, its pieces are copied H2D.

### 8.4 Spill and restore policy

- **Spill:** after every request, an idle slot with P ≥ 2048 tokens is copied to RAM asynchronously.
  The same session's next turn updates the entry **in place incrementally** (only changed
  checkpoints and pages from the first modified page; an agent turn ≈ 2 checkpoints + 2 pages ≈
  318 MiB). A request that reuses nothing from a slot unlinks the slot from its old entry (bug
  found: otherwise an unrelated session overwrote another session's entry).
- **Never wait for a spill.** If a new request must overwrite positions that a spill is still
  reading, the slot keeps the pages before the reuse point, duplicates the page containing it
  (device-to-device) and takes fresh pages after it; the old pages go to quarantine until the tier
  stream passes a fence. (The first version waited: ~150 ms stalls of the main stream when a request
  hit a slot mid-spill — a 692 MiB spill takes 180 ms at the link speed.)
- **Restore:** the server first picks the slot with the longest VRAM prefix as before; if the tiers
  hold a clearly longer usable prefix (≥ 512 more tokens and ≥ reuse/40), the slot is filled from the
  entry (H2D on the tier stream while other slots keep decoding), then the normal job start runs
  (loading a checkpoint discards pending replay rows), so the following prefill chunks are exactly
  those of a never-evicted slot. TTFT includes the restore.
- **Main/tail split:** the hit checkpoint and KV pages ("main") are copied first and the request
  starts when they land; the slot's other checkpoints follow ("tail"). A checkpoint save that meets
  a still-moving tail drops the not-yet-enqueued part rather than waiting.
- **Delay hit:** queued requests whose best prefix is an entry currently being restored wait for
  that restore instead of starting a second one; a shared entry becomes a VRAM shared checkpoint
  after restore and the others attach to it.
- **SSD → RAM prefetch:** a queued request whose best prefix is an SSD-only entry starts reading it
  into RAM while it waits.
- The idea of delay hits and prefetch-on-hit comes from the Strata paper (arXiv 2508.18572); no code
  was used.

### <a id="page-return"></a>8.5 The page-return bug

Under load, a 126k-token restore never happened. A slot that received a short request reusing
nothing kept the old session's pages (459 of them) — running slots cannot be evicted — leaving 310
free pages. The restore did not fit, the request fell back to a full 117k-token prefill (~75 s,
during which the other three slots decoded at 9–15 tok/s), and finally failed with `KvPoolFull`
(HTTP 500). Fix: when a job starts and truncates its cache to the reuse point, every page beyond
`N + drafts + 2` is returned to the pool immediately (via quarantine if a spill is reading it;
shared pages via refcount). This bug existed since pages were introduced; it only shows when long
sessions and short requests mix.

### 8.6 Results

| Q4_K_M, context | Full prefill TTFT | Restored from RAM | From SSD after restart | Never evicted (VRAM hit) |
|---|---|---|---|---|
| 28.3k | 23.06 s | 0.590 s | 0.654 s | 0.18 s |
| 67.2k | ~59 s (62.5k: 58.83 s) | 1.221 s | 1.270 s | 0.23 s |
| 125.1k | ~147 s (126.2k: 147.14 s) | 2.178 s | 2.207 s | 0.35 s |
| 125.1k while 3 other slots decode | (before the page fix: restore impossible, KvPoolFull) | **2.225 s** | 2.280 s | |

A restore costs ~14–16 µs per token — the host-link limit of this eGPU (~3.65 GB/s achieved) — plus
~80 ms for two checkpoints; 35–65× faster than re-prefilling. SSD is only 2–10% slower than RAM
because reads and H2D copies are pipelined. MXFP4 28.3k: full 10.70 s, RAM 0.62 s, SSD 0.71 s.

| Burst: 30.6k-token system prompt evicted from VRAM, 3 new sessions at once | Before | After |
|---|---|---|
| Shared entry in RAM | 2.78 / 23.57 / 3.68 s | 0.64 / 0.83 / 0.82 s |
| After restart (entry on SSD) | 42.1 / 42.1 / 41.9 s | 0.89 / 0.71 / 0.89 s |

Multi-session agent under eviction pressure (6 sessions sharing a ~12k-token system prompt, each
reading a ~3.5k-token file and asking 3 questions, 4 slots, requests interleaved so every session
finds its slot taken): tiers off 378.1 s wall / 435,697 new prefill tokens; tiers on **146.4 s** /
129,968 tokens — 2.58× faster, the same new-prefill count as a never-evicted solo run, identical
texts. The normal path (no eviction) did not get slower (CLI mean 182.43 vs 182.43 tok/s).

Why an evicted session can differ from a solo run in one case: if its own entry is less than 512
tokens longer than the shared system checkpoint, the server deliberately resumes from the system
checkpoint and re-prefills those few tokens (originally produced by decode). The result is a correct
cold computation of those tokens — numerically equivalent, not bit-identical to the decode-produced
KV. Forcing restores always (`WHIRL_TIER_MIN_GAIN=1`) made 28/28 calls identical to solo, and a
byte-verify mode confirmed every spill/restore byte-identical (115 of them).

**[eGPU] environment limitation:** while a restore saturates the USB4 link, kernel dispatch slows and
the other slots' decode cycles stretch from 43.0 to 50.4 ms (+17%) for the 0.5–2.0 s the restore
runs. Not optimized ([windows-hip.md](windows-hip.md#egpu)); not expected on direct PCIe.

### 8.7 Gates

- `tier_gate`: a reference server without tiers runs turns 1 and 2 of each session without
  eviction; the tier server runs all first turns, is forced to evict, runs the second turns
  (restored from RAM: text and cached-token count must equal the reference), restarts and runs them
  again (restored from SSD). With `--parallel 1` (f16 KV, thinking and non-thinking sessions,
  including `think-open`) and `--parallel 4` (q8v, five sessions). Both models.
- `sys_gate`: shared system checkpoint reuse (cached == B, text == cold), 4 concurrent new sessions,
  eviction to RAM and restore, SSD after restart, tiers off, bursts, and `prefix` checkpoints.
- `restore_conc_gate`: RAM restore beside 3 decoding slots == never evicted; burst on a shared entry
  in RAM / SSD; SSD prefetch for a queued request.
- Every test server gets its own SSD directory; otherwise entries written by a previous test turn the
  next test's prefill into a restore (same fingerprint) and change what is being tested.

## 9. Open items

- Each session's tier entry still copies the shared system prompt's KV; the 9 GiB RAM tier fills
  quickly with 30k-token system prompts. Entries could reference the shared entry's blocks.
- Only two system checkpoints fit in VRAM; several alternating system prompts (main agent + different
  sub-agents) evict each other (restores take ~0.3 s for 13k tokens).
- A second request for the same session entry restores it again from RAM; it could share the first
  slot's VRAM pages instead.
- Pinned-size declaration files are keyed by PID; PID reuse caused false alarms
  (use PID + start time).
- q8v prefill is 2–4% slower than f16 (V dequant in the PV loop); dequantizing V tiles into LDS once
  could recover it.
