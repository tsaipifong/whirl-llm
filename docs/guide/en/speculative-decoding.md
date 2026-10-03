**English** | [繁體中文](../zh-TW/speculative-decoding.md)

# Speculative decoding: MTP and n-gram co-drafting

> **Status.** Implemented in `src/model/spec.cpp` (draft policy) and `src/model/forward.cpp`
> (draft / verify passes). The environment variables are listed in [usage.md](../../usage.md#env).

**Who this helps:** anyone implementing speculative decoding on a bandwidth-bound GPU, in
particular for hybrid models with recurrent state (DeltaNet, Mamba-style layers) where a cache
cannot simply be truncated; and anyone who needs speculative output to be *identical* to plain
greedy output rather than merely "equivalent".

## 1. Why speculation is the only real lever

A 27B 4-bit model reads 14.33 GB of weights per token. At the measured 604–626 GB/s streaming
ceiling that caps plain decode at about 38 tok/s, and WHIRL's 1-token GEMV is already at
97–100% of that ceiling ([kernels.md](kernels.md#gemv)). But verifying several tokens costs
almost the same as decoding one, because the weights are read once for all rows:

| Rows through the model (Q4_K_M, R9700) | Time |
|---|---|
| 1 (plain decode step) | 27.74 ms |
| 2 | 29.21 ms |
| 3 | 30.22 ms |
| 4 | 30.93 ms |
| 5 | 31.71 ms |
| 8 | 34.79 ms |
| 16 | 39.50 ms |

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
step, so k drafts cost k MTP steps. Conditional acceptance of the second chained draft was
68–77% on our prompts.

### 2.1 A draft cycle

1. MTP drafts k tokens; each draft token is passed to the next step **as a kernel argument** and
   stays on the GPU (a by-value token struct; no `hipMemcpy` — see
   [windows-hip.md](windows-hip.md#streams) for why a synchronous copy here cost 5%).
2. The trunk verifies k+1 rows in one forward pass (k drafts + the token they follow), keeping
   every row's logits and hidden state.
3. A pick kernel compares drafts with the trunk's argmax row by row and writes the accepted
   count, the next token and a control word.
4. The host synchronizes **once per cycle**, reads the control word, emits accepted tokens and
   enqueues the next cycle.

How the first version got from 34 to 68.8 tok/s (27B, Chinese thinking prompt, 3 drafts max at
the time):

| Step | Change | tok/s |
|---|---|---|
| no MTP | int8 GEMV + fusion + device-resident state | 34.2 |
| 1 draft | draft 1, verify 2; acceptance 75–82% | 50.7 |
| 2 chained drafts | MTP's own head-normed output as next hidden | 55.9 |
| multi-token GEMV decodes each weight unit once | T=4 matmul 35.7 → 28.6 ms | 58.4 |
| batched verify attention | 2 launches per query → 2 launches for all queries | 61.1 |
| 3 drafts + Q4_K draft head | third snapshot; draft head from the Q6_K output head | 64.3 |
| whole cycle chained on the GPU | token ids as kernel args, one sync per cycle | 68.8 |

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
numerics. MTP output differed from plain greedy on all three variants, and the auto draft
policy collapsed to 1 draft. Requantizing those matrices to Q8_0 (which has a fused exact path,
as in unsloth's Q4_K_M) fixed it. Engine-side fix (route F32 small matrices through one kernel
for all n) is on the to-do list. See [quantization.md](quantization.md#f32-alpha-beta).

Gates: CLI MTP vs plain on Chinese and English prompts for both models; draft counts 1, 2, 5, 10
all identical; n-gram forced on every cycle == plain; server and concurrent variants
([benchmarking.md](benchmarking.md#gates)).

## <a id="replay"></a>4. Rolling back DeltaNet state: snapshots, then replay

An attention KV cache can be "rolled back" by ignoring positions past the accepted length. A
recurrent DeltaNet state cannot: after verifying k+1 rows it already contains all of them.

**Snapshots (first design).** During verify, the state after each row was written to a snapshot
buffer (ping-pong `*_alt` buffers, later an array of up to 15 sets). On rejection at row k the
engine swaps the state pointer to snapshot k — a pointer swap, no copy. State and snapshot
pointers are passed by value as kernel arguments. Cost: about 0.4 ms per cycle at 3 drafts, but
each snapshot set is 150 MiB for the 27B model (48 layers), and a 16-row verify spent about
2.6 ms writing them. The server had to reserve 8 sets (1.2 GiB).

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
| 12-row verify at four concurrent users | 40.42 → 39.47 ms |
| Server VRAM for snapshot sets | 1.2 GiB → 0 (the KV pool grew by 39,424 tokens at the time) |
| n-gram drafts per slot in the server | no longer capped by the 8 snapshot sets (up to 15) |
| Cost | MoE model, 1 user: −0.5% (the next verify replays ~1.3 extra rows) |

Possible next step: apply the pending rows for all layers and slots in one kernel on a second
stream, overlapped with the next MTP draft steps.

## <a id="auto"></a>5. Choosing the number of drafts automatically

The best fixed draft count depends on content. Max-draft sweep, 27B, tok/s (coding = harmonic
mean of 5 tasks × 600 tokens):

| Max drafts | Coding, no thinking | Coding, thinking | Chinese | English | 14k context | 24k context |
|---|---|---|---|---|---|---|
| 1 | 61.1 | 59.1 | 55.1 | 54.3 | 47.8 | 40.6 |
| 2 | 81.7 | 75.7 | 65.8 | 61.4 | 52.8 | 44.8 |
| 3 | 94.3 | 85.4 | 67.3 | 60.2 | 52.5 | 41.9 |
| 4 | 103.2 | 90.1 | 67.3 | 58.7 | 50.7 | 36.5 |
| 6 | 109.9 | 90.3 | 59.4 | 53.6 | 43.9 | 30.1 |
| 8 | 111.1 | 83.7 | 51.7 | 44.8 | 36.0 | 24.4 |
| 10 | 95.6 | 71.2 | 42.7 | 37.2 | 30.1 | 20.4 |
| **auto (cap 8)** | **105.2** | **90.1** | **67.6** | **60.4** | **52.4** | **44.2** |

Code wants 6–8, prose 3–4, long contexts 2–3. The auto policy is a cost model:

- For each draft position i, track the **conditional acceptance** p_i (probability draft i is
  accepted given drafts 1..i−1 were) as an EMA. Expected tokens per cycle for k drafts:
  E(k) = 1 + Σ_{j=1..k} Π_{i≤j} p_i.
- Track the **measured cycle time** T(k) for each k (EMA).
- Each cycle pick k maximizing E(k) / T(k), with 3% hysteresis against timing noise, and probe a
  neighboring k every 32 cycles.

It lands near each content type's best. Checked on the 8060S in one process with interleaved
prompts (to avoid thermal drift): auto 31.36 tok/s vs the best fixed count (4) 31.28.
Variants that showed no measurable gain and were removed: expiring the timing table after 64
cycles, shrinking deep-position acceptance toward the previous position, keeping acceptance
across requests.

**MoE model: 1 draft.** Ornith's per-position acceptance was 92 / 31 / 4 / 0%; with 3 drafts it
ran 171–172 tok/s and at 24k was slower than no MTP. Fixed at 1 (overridable).

**Server specifics:**

- Draft caps by number of decoding slots (dense): `{8, 7, 4, 3}` for 1–4 slots, i.e. at most 16
  verify rows; the cost model chooses within the cap. Raising the 4-slot cap from 2 to 3 (16 rows)
  gave +10% at four users.
- With zero drafts (too many users), the MTP layer still processes accepted tokens (without the
  draft head) so its KV cache stays complete and drafting can resume immediately.
- **Reset the timing table when a request starts on an idle engine.** The server kept the
  cycle-time table across requests; a single user sending requests back to back got too many
  drafts from the second request on (8060S: ~4.5 drafts per cycle vs ~3.3 in the CLI). Resetting
  on idle-start gave +4.1% for requests 2–4, and server-vs-CLI speed went from −3.6% to −0.5%.
- Per-slot draft counts (moving drafts from the slot least likely to accept to the one most
  likely) gave +13% at two mixed users but slowed low-acceptance requests by ~10% and fixed-batch
  wall time by 5%; kept off for fairness.

**p-min / n-min (optional, off).** The pick kernel can also compute the draft's softmax
probability; drafts after the first n-min stop when the probability falls below p-min. The stop
decision runs on the GPU (a stop flag makes later draft kernels exit immediately), so there is
still one sync per cycle. With the auto policy, p-min 0.3 changed speed by ±1% — not a default.

## <a id="draft-head"></a>6. Making drafts cheaper: draft-head quantization

The output head (248,320 × 5120, Q6_K, 1.04 GB) runs once per draft step, so it dominates draft
cost. Measured first: **acceptance barely depends on draft-head precision** (only the argmax
matters): full Q6_K head 58.8%, Q4_K 59.1%, Q3_K 59.2% at a fixed 6 drafts.

**D2 (2.5 bits per weight).** One f16 scale per 32 values and a 2-bit code per value,
`w = s · (2c − 3)`, stored structure-of-arrays (codes K/4 bytes + scales K/16 bytes per row).
Built at load time on the GPU from the Q6_K head, choosing for each block the scale with the
smallest squared error among 6 candidates. Its own GEMV (int8 activations, `v_dot4`, 1–16 rows).
Size 715 MB (Q4_K head) → 397 MB, which also saves 318 MB of VRAM.

| Draft step | Full Q6_K head | Q4_K head | D2 head |
|---|---|---|---|
| R9700 | 2.72 ms | 2.17 ms | **1.64 ms** |
| 8060S | 6.88 ms | 5.44 ms | **4.05 ms** |

Interleaved A/B with the auto policy: R9700 27B +2.4%, MoE +2.0%; 8060S +1.5% / +2.5%. (Per-draft
acceptance fell from 63.2% to 59.4% because the policy sends more drafts; tokens per second rose.)

**The MTP block in Q4_K.** Requantizing the MTP layer's Q6_K matrices (~330 MB) to Q4_K
(~225 MB) was first rejected (acceptance −1.5 points, speed −1.6%). Re-measured after the other
speedups it gave +0.8% single-user and cut the per-cycle MTP time at four users from 5.00 to
4.35 ms, so it is now the default for dense models — an example of why small rejected ideas are
re-tested after the core gets faster.

**Rejected:** a Q3_K draft head (acceptance unchanged but its GEMV ran at ~488 GB/s: +0.3%);
truncating the draft vocabulary to the first N token ids (Chinese tokens have high ids:
acceptance fell from 82% to 28–33%; with 150,000/100,000 ids later, 2.58 → 2.43 tokens/cycle).

Model-file note: an output head stored as Q8_0 makes every draft step more expensive. Of three
Swift-1.5 MXFP4 variants, the Q8_0-head file had the highest acceptance (73.2%) but the slowest
MTP (98.0 vs 117.0 tok/s for the Q6_K-head file). See [quantization.md](quantization.md#swift).

## <a id="ngram"></a>7. n-gram co-drafting

Editing tasks (rename a variable and write the file back, translate comments, refactor) copy
long spans of the prompt. MTP is already 100% accepted on such spans within its 8-draft cap; the
win comes from **longer drafts** (15) and **cheaper cycles** (an n-gram cycle runs no MTP draft
steps: a 16-row verify is ~43–44 ms vs ~49 ms for an 8-draft MTP cycle).

### 7.1 What the first version did wrong

An untested first version made every editing prompt slower (−13% on edits, −5…−7% elsewhere):
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
  not recorded, because the first 16-row verify carries a one-off ~50 ms cost). n-gram cycles do
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

| 27B prompt (think off) | MTP | MTP + n-gram | Δ |
|---|---|---|---|
| Coding benchmark, 7-prompt mean | 94.38 | 94.20 | −0.2% (other runs +0.2%, +0.1%) |
| py-rename-zh (1.6–2.2k-token file, write it back) | 157.9–158.1 | 283.8–284.3 | +79.8% |
| zig-rename-en | 166.1–166.3 | 315.2–315.8 | +89.8% |
| py-refactor-en | 160.2 | 268.1–269.3 | +67.8% |
| py-rename-crlf | 170.8 | 338.7–339.2 | +98.5% |
| zig-comment-zh | 143.7 | 213.0–213.1 | +48.3% |

Per-prompt ±2% swings on the coding benchmark come from the MTP policy trajectory changing after
a single n-gram cycle, not from n-gram cost (host-side lookup ~10 µs per cycle, 0.13%). MoE model:
coding benchmark +2.0%, edits +52…+151%. Server agent benchmark (tool calls read_file /
write_file, two "rename and write back" turns): write turns +28% / +21% at first, **+56% / +50%**
after replay lifted the snapshot cap (327–329 and 284 tok/s). At 2–4 concurrent users n-gram
was neutral (±1%).

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

| Qwen3.8-27B Q4_K_M | llama.cpp | WHIRL |
|---|---|---|
| Decode, no MTP | 30.9 tok/s | 33.9 |
| Decode, MTP | 56.9 | 105.9 |
| Draft acceptance | 82.4% (30,100 / 36,528) | 64.8% (33,353 / 51,461) |
| MTP output == plain output | 6 of 19 prompts | 19 of 19 |

(WHIRL numbers from the same run; later kernel work raised WHIRL MTP to 110.4.) An earlier
head-to-head with both engines capped at 3 drafts on five coding tasks: WHIRL 87–94 tok/s vs
llama.cpp 61–68, with nearly identical acceptance (Java task: 441/483 vs 437/483). What we can
attribute from measurements:

1. **Per-cycle cost.** In WHIRL, verifying 4 tokens costs ~15% more than decoding 1, the whole
   cycle is chained on the GPU with one synchronization, and drafting uses a 2-bit head. With the
   same acceptance, WHIRL still produced ~40% more tokens per second at the same draft cap.
2. **Draft count.** llama.cpp is fixed at a maximum of 3 drafts here; WHIRL's policy uses up to 8
   (15 with n-gram), which matters most for code. Lower per-draft acceptance in WHIRL is the
   expected consequence of sending more drafts — tokens per cycle are higher.
3. **Exactness.** llama.cpp's batched verify is not bit-identical to its single-token path (13 of
   19 outputs changed with MTP on). WHIRL's adaptive policy depends on bit-exact verify to stay
   correct with long draft runs.
4. **MoE.** On Ornith, llama.cpp's MTP was roughly neutral and slower at long context
   (decode 99.5 → 101.7 tok/s short, 89.7 → 84.1 at 24k); WHIRL: 158.9 → 188.1 short.

At long context the gap narrows (24k: WHIRL 45.1 vs llama.cpp 39.8 in the early comparison)
because attention takes a larger share of each cycle for both engines.

## 11. Open items

- Each cycle still has ~0.95 ms of host gap (enqueue overlaps the GPU; the gap is around
  completion and the next start). Removing it requires computing the accepted count and building
  the next MTP/verify row tables on the device.
- Replay commit on a side stream (§4).
- Q4_K output heads have no multi-row tuning or draft path comparable to the Q6_K head's yet.
