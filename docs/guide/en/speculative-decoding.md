**English** | [繁體中文](../zh-TW/speculative-decoding.md)

# Speculative decoding: MTP and n-gram co-drafting

> **Status.** Implemented in `src/spec/policy.cpp` (draft policy) and `src/model/forward.cpp`
> (draft / verify passes). The environment variables are listed in [usage.md](../../usage.md#env).

**Who this helps:** anyone implementing speculative decoding on a bandwidth-bound GPU, in
particular for hybrid models with recurrent state (DeltaNet, Mamba-style layers) where a cache
cannot simply be truncated; and anyone who needs speculative output to be *identical* to plain
greedy output rather than merely "equivalent".

## 1. Why speculation is the only real lever

A 27B 4-bit model reads all its weights for every token. WHIRL's 1-token GEMV already runs at
97–100% of the measured 604–626 GB/s streaming ceiling ([kernels.md](kernels.md#gemv)), and a plain
decode step of the 27B MXFP4 model takes 25.15 ms (about 40 tok/s). But verifying
several tokens costs almost the same as decoding one, because the weights are read once for all rows:

| Rows through the model (27B MXFP4, R9700) | Time |
|---|---|
| 1 (plain decode step) | 25.15 ms |
| 3 | 27.26 ms |
| 5 | 28.44 ms |

So every accepted draft is nearly free. The engineering problem is to (a) produce drafts that
are likely to be accepted, (b) make the multi-row verify as cheap as possible, and (c) keep the
output exactly what plain decoding would produce.

## 2. MTP: the model's own draft layer

Qwen3.8 ships one multi-token-prediction layer (`blk.64`, "nextn"). Given the trunk's hidden
state h for position t and the token at t+1, it predicts the token at t+2:

```
x = eh_proj([ enorm(embed(token)) ; hnorm(h) ])
x = one gated-attention layer + FFN (MoE in Ornith), with its own KV cache
draft = argmax(output_head(norm(x)))            # output head shared with the trunk
```

Chaining: the MTP layer's own head-normed output is used as the hidden state for the next draft
step, so k drafts cost k MTP steps.

### 2.1 A draft cycle

1. MTP drafts k tokens; each draft token is passed to the next step **as a kernel argument** and
   stays on the GPU (a by-value token struct; no `hipMemcpy` — see
   [windows-hip.md](windows-hip.md#streams) for why a synchronous copy here was costly).
2. The trunk verifies k+1 rows in one forward pass (k drafts + the token they follow), keeping
   every row's logits and hidden state.
3. A pick kernel compares drafts with the trunk's argmax row by row and writes the accepted
   count, the next token and a control word.
4. The host synchronizes **once per cycle**, reads the control word, emits accepted tokens and
   enqueues the next cycle.

How the first version was built up (3 drafts max at the time), each step faster than the one
before:

| Step | Change |
|---|---|
| no MTP | int8 GEMV + fusion + device-resident state |
| 1 draft | draft 1, verify 2 |
| 2 chained drafts | MTP's own head-normed output as next hidden |
| multi-token GEMV decodes each weight unit once | the 4-row matmul got much cheaper |
| batched verify attention | 2 launches per query → 2 launches for all queries |
| 3 drafts + 4-bit draft head | third snapshot; draft head from the Q6_K output head |
| whole cycle chained on the GPU | token ids as kernel args, one sync per cycle |

## <a id="exact"></a>3. The exactness requirement

WHIRL's MTP output is **bit-identical** to plain greedy output, for every draft count 1–10, with
and without n-gram drafts, single-user and concurrent. This is a gate, not an aspiration. It
requires every row of a verify pass to be computed exactly as a 1-token decode would compute it:

| Component | What makes the verify row equal to the decode row |
|---|---|
| GEMV | multi-row kernels share the canonical per-unit float order and reduction tree with the 1-row kernel; int8 WMMA keeps the integer part exact ([kernels.md](kernels.md#multirow)) |
| Attention | grouped verify computes each query column exactly as a single column (contraction off, explicit fma) ([kernels.md](kernels.md#decode-attn)) |
| DeltaNet state | per-row snapshots or replay with identical formulas (§4) |
| Small matrices | every weight type must use the *same* kernel family for n = 1 and n ≥ 2 (a counter-example is below) |
| EOS handling | accepted drafts are cut at end-of-sequence so the state stops exactly where plain decoding stops (§8) |

Why insist? Because "equivalent" is not testable, and near-ties are common. When an early
refactor changed the multi-token dot-product order by one ulp, verify logits differed from
decode logits in the last bits, near-tied tokens flipped, and MTP output stopped matching
greedy. With bit equality, any difference between MTP and plain output is a bug, and the gate
`MTP greedy == plain greedy` catches it immediately.

**Counter-example that slipped through a model file, not the engine:** a quantization of
Swift-1.5 stored the tiny DeltaNet `ssm_alpha`/`ssm_beta` matrices as F32. WHIRL has no int8
GEMV for F32 weights: n = 1 used an f32-activation GEMV, n ≥ 2 used an f16 GEMM — different
numerics, so MTP output differed from plain greedy on all three variants. Requantizing those
matrices to Q8_0 (which has a fused exact path) fixed it.

**Engine fix (0.1.3).** F32 / F16 weights at n = 2..16 now run the same f32-activation GEMV
instances as n = 1 (`gemv_<T>_8 / _4 / _1`), so every token is bit-identical to the single-row
result and speculative output equals plain greedy on such files (for example the Cyber-Tiel
series, and some Q4_0 / Q4_K_M quantizations load and run this way). **They still draft only 1 token:** the fused DeltaNet
decode path (`gdn_ab`) exists only for Q8_0 and MXFP4 α/β, and without it (`fusedDecode()` false)
only one recurrent-state snapshot is written per verify. This is a **hard cap**, not a choice of the
cost model: the CLI (MTP and n-gram) and the server both clamp drafts to 1 (the server's startup
line says "fused decode off: 1 used"). Restoring an unwritten snapshot now throws `SnapshotNotWritten`
instead of silently diverging. For full draft counts, requantize α/β to Q8_0
([quantization.md](quantization.md#f32-alpha-beta)).

Gates: CLI MTP vs plain on Chinese and English prompts for both models; draft counts 1, 2, 5, 10
all identical; n-gram forced on every cycle == plain; server and concurrent variants
([benchmarking.md](benchmarking.md#gates)).

## <a id="replay"></a>4. Rolling back DeltaNet state: snapshots, then replay

An attention KV cache can be "rolled back" by ignoring positions past the accepted length. A
recurrent DeltaNet state cannot: after verifying k+1 rows it already contains all of them.

**Snapshots (first design).** During verify, the state after each row was written to a snapshot
buffer (ping-pong `*_alt` buffers, later an array of up to 15 sets). On rejection at row k the
engine swaps the state pointer to snapshot k — a pointer swap, no copy. State and snapshot
pointers are passed by value as kernel arguments. But each snapshot set is 150 MiB for the 27B
model (48 layers), a 16-row verify spent noticeable time writing them, and the server had to
reserve 8 sets (1.2 GiB).

**Replay (current design, server default).** Verify no longer writes per-row snapshots or even
the final state. It records each row's *inputs* per slot and layer: the conv input rows
`[16][convCh]` and the delta-rule inputs `[16][head][k 128 | v 128 | decay | beta]` (66.3 MiB per
slot for 27B). After the host decides that a rows were accepted, it marks `a + 1` rows pending
for that slot. The next verify's conv and step kernels first apply the pending rows to the state
— with exactly the same formulas as inside verify, so the result is bit-for-bit what a snapshot
would have held — write one new base state, and then process the new rows. Every other path that
reads or writes the state commits pending rows first (prefill, single-sequence forward,
checkpoint save) or discards them (checkpoint load, reset).

| Effect of replay | |
|---|---|
| 12-row verify at four concurrent users | slightly faster (no snapshot writes) |
| Server VRAM for snapshot sets | 1.2 GiB → 0 (the KV pool grew by 39,424 tokens at the time) |
| n-gram drafts per slot in the server | no longer capped by the 8 snapshot sets (up to 15) |
| Cost | the next verify replays ~1.3 extra rows (a small cost for one user on the MoE model) |

Possible next step: apply the pending rows for all layers and slots in one kernel on a second
stream, overlapped with the next MTP draft steps.

## <a id="auto"></a>5. Choosing the number of drafts automatically

The best fixed draft count depends on content. A max-draft sweep (1–10 drafts; coding with and
without thinking, Chinese, English, 14k and 24k context) found that code wants 6–8 drafts, prose 3–4,
long contexts 2–3, and that too many drafts are clearly slower than the best count. The auto policy
is a cost model:

- For each draft position i, track the **conditional acceptance** p_i (probability draft i is
  accepted given drafts 1..i−1 were) as an EMA. Expected tokens per cycle for k drafts:
  E(k) = 1 + Σ_{j=1..k} Π_{i≤j} p_i.
- Track the **measured cycle time** T(k) for each k (EMA).
- Each cycle pick k maximizing E(k) / T(k), with 3% hysteresis against timing noise, and probe a
  neighboring k every 32 cycles.

It lands near each content type's best (in the sweep and on the 8060S, checked in one process with
interleaved prompts to avoid thermal drift, it matched the best fixed count).
Variants that showed no measurable gain and were removed: expiring the timing table after 64
cycles, shrinking deep-position acceptance toward the previous position, keeping acceptance
across requests.

**MoE model: 1 draft.** Ornith's per-position acceptance falls off sharply after the first draft;
with 3 drafts it was slower than with 1, and at 24k slower than no MTP. Fixed at 1 (overridable).

**Server specifics:**

- Draft caps by number of decoding slots (dense): `{8, 7, 4, 3, 2, 1, 1, 1}` for 1–8 slots (0 beyond),
  which keeps an MTP verify at ≤ 16 rows; n-gram drafts may use up to 32 rows in total with wide verify
  ([server.md](server.md#batching)). The cost model chooses within the cap. Raising the 4-slot cap from 2 to 3 (16 rows)
  made four users faster.
- With zero drafts (too many users), the MTP layer still processes accepted tokens (without the
  draft head) so its KV cache stays complete and drafting can resume immediately.
- **Reset the timing table when a request starts on an idle engine.** The server kept the
  cycle-time table across requests; a single user sending requests back to back got too many
  drafts from the second request on (8060S: ~4.5 drafts per cycle vs ~3.3 in the CLI). Resetting
  on idle-start brought the server back to CLI speed for requests 2–4.
- Per-slot draft counts (moving drafts from the slot least likely to accept to the one most
  likely) helped two mixed users but slowed low-acceptance requests and fixed-batch wall time;
  kept off for fairness.

**p-min / n-min (optional, off).** The pick kernel can also compute the draft's softmax
probability; drafts after the first n-min stop when the probability falls below p-min. The stop
decision runs on the GPU (a stop flag makes later draft kernels exit immediately), so there is
still one sync per cycle. With the auto policy, p-min 0.3 made no measurable difference — not a default.

## <a id="draft-head"></a>6. Making drafts cheaper: draft-head quantization

The output head (248,320 × 5120, Q6_K, 1.04 GB) runs once per draft step, so it dominates draft
cost. Measured first: **acceptance barely depends on draft-head precision** (only the argmax
matters): a full Q6_K head and 4-bit and 3-bit requantized heads gave the same acceptance at a
fixed 6 drafts.

**D2 (2.5 bits per weight).** One f16 scale per 32 values and a 2-bit code per value,
`w = s · (2c − 3)`, stored structure-of-arrays (codes K/4 bytes + scales K/16 bytes per row).
Built at load time on the GPU from the Q6_K head, choosing for each block the scale with the
smallest squared error among 6 candidates. Its own GEMV (int8 activations, `v_dot4`, 1–16 rows).
Size 715 MB (4-bit head) → 397 MB, which also saves 318 MB of VRAM.

| Draft step | Full Q6_K head | 4-bit (Q4_K) head | D2 head |
|---|---|---|---|
| R9700 | 2.72 ms | 2.17 ms | **1.64 ms** |
| 8060S | 6.88 ms | 5.44 ms | **4.05 ms** |

Interleaved A/B with the auto policy: faster on both models and both GPUs. (Per-draft acceptance
fell because the policy sends more drafts; tokens per second rose.)

**The MTP block in Q4_K.** Requantizing the MTP layer's Q6_K matrices (~330 MB) to Q4_K
(~225 MB) was first rejected (slightly lower acceptance, no gain). Re-measured after the other
speedups it was faster for one user and at four users, so it is now the default for dense models — an example of why small rejected ideas are
re-tested after the core gets faster. Files whose MTP block is stored as **Q8_0** (the Swift-1.5
MXFP4 files: ~450 MB per draft step plus the 2-bit head) get the same Q4_K copies (`requant_q80_q4k`,
~240 MB); outputs are unchanged. Swift MXFP4-A on the R9700, ~16k-token prompts, steady-state decode:
one user 78.3 → 85.4 tok/s, four users 213.7 → 221.2 tok/s aggregate (both with `attn_wsplit2`).

**Rejected:** a Q3_K draft head (acceptance unchanged but its GEMV ran at only ~488 GB/s, no real
gain); truncating the draft vocabulary to the first N token ids (Chinese tokens have high ids, so
acceptance collapsed; even with 150,000/100,000 ids, fewer tokens per cycle).

Model-file note: an output head stored as Q8_0 makes every draft step more expensive. Of three
Swift-1.5 MXFP4 variants, the Q8_0-head file had the highest acceptance (73.2%) but the slowest
MTP (98.0 vs 117.0 tok/s for the Q6_K-head file). See [quantization.md](quantization.md#swift).

## <a id="ngram"></a>7. n-gram co-drafting

Editing tasks (rename a variable and write the file back, translate comments, refactor) copy
long spans of the prompt. MTP is already 100% accepted on such spans within its 8-draft cap; the
win comes from **longer drafts** (15) and **cheaper cycles** (an n-gram cycle runs no MTP draft
steps, so a 16-row n-gram verify costs less than an 8-draft MTP cycle).

### 7.1 What the first version did wrong

An untested first version made every editing prompt slower, and other prompts too:
it used an n-gram match whenever one existed regardless of its track record, took only the most
recent occurrence of the 3-gram (common code 3-grams like `;\n    ` point to the wrong place),
and was capped at the MTP limit of 8. Investigating why copying a file accepted only 25% of
n-gram drafts uncovered two unrelated problems that matter far beyond n-gram drafting:

- **A tokenizer bug** split indented code into non-canonical tokens; the model writes canonical
  tokens, so MTP copied fine and n-gram lookup did not ([pitfalls.md](pitfalls.md#tok-1)).
- **CRLF:** in a Windows file inside a prompt, `"\r\n"` is token 317; the model writes back LF,
  token 198. The n-gram index now uses a token-level CRLF→LF normalization table (every token
  containing `\r\n` maps to the token for the same text with LF; on 5 source files all 653 CR
  tokens mapped 1:1 and the normalized sequence equalled tokenizing the LF file). Matching uses the
  normalized sequence; drafts emit LF tokens, unless the model has itself been writing CR tokens.

### 7.2 Design

- **Index.** Every position of the history (prompt + output) is inserted under a 3-token key and
  a 12-token key (hash maps; each key chains all earlier positions).
- **Candidates:** the continuation of the previous proposal's source position (usually right when
  copying), the first 16 entries of the 12-token key's chain, and — if the best match is shorter
  than 12 — the first 16 of the 3-token chain. Choose the candidate with the **longest backward
  match** (capped at 256, ties → most recent), at least 3 tokens. Proposal length up to 15 drafts
  (16 verify rows).
- **Scoring every proposal counterfactually.** Whether or not a proposal is used, it is scored
  afterwards against the tokens actually produced, so the n-gram's per-position conditional
  acceptance is always fresh. Acceptance is tracked in three buckets by match length (< 8, 8–23,
  ≥ 24; priors 0.35 / 0.7 / 0.85; EMA 0.2).
- **Decision.** Each cycle compares `max_k E_ng(k) / T_ng(k)` with MTP's chosen E/T and uses
  n-gram only if it is better, with the best k. n-gram cycle time is measured separately (prior:
  the 1-draft MTP cycle × (1 + 1.5%·(k−1)) until measured; the first occurrence of a row count is
  not recorded, because the first 16-row verify carries a one-off cost). n-gram cycles do
  not update MTP's acceptance or timing model.
- **An n-gram cycle** runs the MTP layer only on the pending rows (to keep MTP's KV cache
  complete) and verifies tokens supplied directly by the host.
- **No MTP chained after n-gram drafts:** MTP chaining needs each draft position's hidden state,
  which n-gram drafts do not have.
- **Limits.** CLI: if VRAM allows, extra snapshot capacity is allocated up front so n-gram can
  propose 15 drafts; otherwise the cap equals what exists (nothing is allocated lazily mid-decode,
  nothing spills to shared memory). Server: per-slot cap = min(15, 16 / active slots − 1, the KV
  pages already mapped for that slot, context).

### 7.3 Results (interleaved A/B, same binary, n-gram off vs on, all outputs == plain greedy)

On the editing prompts (rename / refactor / comment translation of 1.6–2.2k-token files, written
back whole, including a CRLF file) n-gram co-drafting is much faster than MTP alone, on the dense
and the MoE model; on the coding benchmark it is neutral (small per-prompt swings come from the MTP
policy trajectory changing after a single n-gram cycle, not from n-gram cost: the host-side lookup
takes ~10 µs per cycle). In the server agent benchmark (tool calls read_file / write_file, two
"rename and write back" turns) the write turns gained, more so after replay lifted the snapshot cap.
At 2–4 concurrent users n-gram was neutral. Current decode numbers: [benchmarks.md](../../benchmarks.md).

Bugs found while building it: arrays sized for 10 MTP drafts overflowed when n-gram allowed more
than 10; and verify rows beyond the slot's mapped KV pages wrote into page 0 of the shared pool
(another sequence's data) — see [pitfalls.md](pitfalls.md#srv-ngram-pages).

## <a id="eos"></a>8. End-of-sequence truncation of accepted drafts

When a verify accepts several drafts and one of them is `<|im_end|>` (or `<|endoftext|>`), the
drafts after it (`\n`, `<|im_start|>`, …) used to be consumed too: the visible output stopped at
EOS, but the DeltaNet state and the cached token list had advanced 1–3 tokens further. The next
chat turn re-renders the history as `content<|im_end|>\n<|im_start|>user…`, so the gen-end
checkpoint sat beyond the common prefix and could not be used.

Fix: the accepted count is cut just before the EOS token, and EOS becomes the chosen token (the
verify produced exactly that token at that position, so the output is unchanged and the state
stops where plain decoding stops). `max_tokens` is truncated the same way (the last emitted
token is not fed back). Effect on multi-turn cache hits is in
[kv-and-caching.md](kv-and-caching.md#think-open).

## <a id="sampling"></a>9. Speculative sampling (temperature > 0)

Drafts are the draft head's argmax (deterministic). The trunk accepts a draft with probability
p(draft) under the sampling distribution; on rejection it samples from that distribution with
the draft token removed and renormalized. For a deterministic draft this reproduces the target
distribution exactly. Random numbers are derived from (seed, token index), so a seeded request
produces the same text alone, concurrently, with MTP on or off.

## <a id="vs-llamacpp"></a>10. Why llama.cpp's MTP is slower on this card

Same GGUF, same R9700, greedy. Unified server protocol (19 prompts; llama.cpp b11214 ROCm with
`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 --spec-draft-p-min 0.3`):

| | llama.cpp | WHIRL |
|---|---|---|
| MTP output == plain output | 6 of 19 prompts | 19 of 19 |

Current speed numbers against the official llama.cpp builds (Swift-1.5 27B MXFP4 and Ornith MXFP4)
are in [benchmarks.md](../../benchmarks.md). In an earlier head-to-head with both engines capped at
3 drafts on five coding tasks, the acceptance was nearly identical (Java task: 441/483 vs 437/483)
and WHIRL was still faster. What we can attribute from measurements:

1. **Per-cycle cost.** In WHIRL, verifying 4 tokens costs 13% more than decoding 1 (27B MXFP4,
   §1), the whole cycle is chained on the GPU with one synchronization, and drafting uses a 2-bit
   head. With the same acceptance, WHIRL produced more tokens per second at the same draft cap.
2. **Draft count.** llama.cpp is fixed at a maximum of 3 drafts here; WHIRL's policy uses up to 8
   (15 with n-gram), which matters most for code. Lower per-draft acceptance in WHIRL is the
   expected consequence of sending more drafts — tokens per cycle are higher.
3. **Exactness.** llama.cpp's batched verify is not bit-identical to its single-token path (13 of
   19 outputs changed with MTP on). WHIRL's adaptive policy depends on bit-exact verify to stay
   correct with long draft runs.
4. **MoE.** On Ornith, llama.cpp's MTP was roughly neutral short and slower at long context;
   WHIRL's MTP was faster than its plain decode.

At long context the gap narrows because attention takes a larger share of each cycle for both
engines.

## 11. Open items

- Each cycle still has a host gap of about a millisecond (enqueue overlaps the GPU; the gap is around
  completion and the next start). Removing it requires computing the accepted count and building
  the next MTP/verify row tables on the device.
- Replay commit on a side stream (§4).
- Q4_K output heads have no multi-row tuning or draft path comparable to the Q6_K head's yet.
