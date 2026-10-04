**English** | [繁體中文](../zh-TW/server.md)

# The OpenAI-compatible server

> **Status.** Describes the C++ server in this repository (`whirl-server.exe`, also `whirl serve`;
> source in `src/server/`). Every option, environment variable and endpoint is listed in
> [usage.md](../../usage.md#serve). Numbers: R9700 (32 GB, USB4 eGPU).

**Who this helps:** users who want to point an OpenAI client (Open WebUI, Cline, Continue, the
`openai` Python package) at WHIRL; and server authors who want to see how continuous batching,
speculative decoding and prefix caching interact in a hybrid-model server.

## 1. Starting it

```
whirl serve MODEL.gguf --port 8080
```

Base URL `http://127.0.0.1:8080/v1`; any API key is accepted. The model is loaded once and stays
resident. First load of a new model file autotunes the prefill GEMMs (tens of seconds; cached per
model and GPU on disk).

| Flag | Meaning | Default |
|---|---|---|
| `--host` / `--port` | bind address / port | `127.0.0.1` / `8080` |
| `--parallel N` | slots served concurrently (continuous batching), max 16 | 4 |
| `--ctx N` | size of the shared KV pool, tokens | all VRAM left after other buffers, minus a reserve |
| `--ctx-per-slot N` | per-request context limit | 131,072 (max 262,144) |
| `--mtp-drafts N` | MTP draft cap 1–10 | dense: automatic, cap 8; MoE: 1 |
| `--decode-min-tps N` | decode floor per streaming request while others prefill ([§6](#batching)); 0 = off | 20 |
| `--kv-ram-mb N` | pinned RAM KV tier size; 0 disables RAM and SSD tiers | 1/4 of physical RAM within [max(8 GiB, one full f16 session + checkpoints), 32 GiB], at most half of the RAM available at startup; 16 GiB on a 64 GB PC; off on integrated GPUs |
| `--kv-ssd-dir` / `--kv-ssd-gb N` | SSD tier directory / size cap; 0 GB disables SSD | per-user local app-data directory / 64 |
| `--log-file` | log file (also printed to the console) | per-user local app-data directory |
| `--alias` | model id reported by `/v1/models` | GGUF file name |
| `--mmproj FILE`, `--vis-idle-s`, `--vis-mode`, `--vis-cache-mb` | image input ([vision.md](vision.md)) | off / 60 / auto / 1024 |

Useful environment variables (all of them: [usage.md](../../usage.md#env)):

| Variable | Effect |
|---|---|
| `WHIRL_KV=auto\|f16\|q8\|q8h\|q8v` | KV format; `auto` applies the floor rule ([kv-and-caching.md](kv-and-caching.md#kv-auto)) |
| `WHIRL_POOL_RESERVE_MB` | VRAM reserve after the pool (default 768 dense / 1536 MoE) |
| `WHIRL_NO_PREFIX_CACHE=1` | disable prefix caching |
| `WHIRL_PREFIX_CACHE_VERIFY=1` | after every cache hit, re-prefill and compare logits (diagnostic) |
| `WHIRL_SERVE_CKPTS` | checkpoints per slot (default 2 when `--parallel` > 1, else 4) |
| `WHIRL_MTP=0`, `WHIRL_NGRAM=0` | disable MTP / n-gram co-drafting |
| `WHIRL_MTP_BATCH_DRAFTS=d1,d2,…` | draft caps by number of decoding slots |
| `WHIRL_GATHER_MS` | burst-gather window for new requests (default 30, 0 = off) |
| `WHIRL_DECODE_MIN_TPS` | decode floor (= `--decode-min-tps`, default 20, 0 = off) |
| `WHIRL_PREFILL_CHUNK` | max rows per merged prefill forward (multiple of 1024, default 2048) |
| `WHIRL_SYS_MIN`, `WHIRL_SYS_CKPTS`, `WHIRL_SYS_LCP=0` | system-prompt checkpoint threshold (2048), VRAM count (2), disable `prefix` checkpoints |
| `WHIRL_KV_TIER_MIN` | minimum entry size for the tiers (2048 tokens) |
| `WHIRL_TIMING_RESET=0` | keep the MTP timing table across requests (not recommended) |
| `WHIRL_LOOP_LOG=1`, `WHIRL_TIER_VERIFY=1`, `WHIRL_TIER_MIN_GAIN=n` | diagnostics: per-loop phase timing, byte-verify every spill/restore, restore threshold (default 512) |
| `WHIRL_PROFILE=1\|2` | per-cycle GPU breakdown; **distorts timing** — never use it for speed numbers |

Only one engine process may use a GPU at a time: the executable takes a named mutex per device and a
second instance waits ([windows-hip.md](windows-hip.md#wddm-demote)).

**Stopping.** Ctrl+C, Ctrl+Break or closing the console window stops the server gracefully: queued
requests get 503, running ones finish, finished sessions spill to the RAM tier and every RAM-tier
entry not yet on the SSD tier (normally written 2 s after its last change) is written now, for at
most 10 s (`shutdown: tier writes done …` in the log), so the next start restores the last turns
from the SSD. A second Ctrl+C exits at once; for a console close Windows ends the process after
about 5 s whatever is left. Killing the process (Task Manager, `taskkill /F`) skips all of this.

## 2. API surface

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming (SSE) or not; tools; thinking; images (with `--mmproj`) |
| `POST /v1/completions` | raw prompt |
| `GET /v1/models` | one model |
| `GET /health` | answers immediately even while busy; reports busy state and queue length |
| CORS preflight | supported |

Responses are standard OpenAI JSON: `finish_reason` `stop` / `length` / `tool_calls`; `usage`
includes `cached_tokens`; a llama.cpp-style `timings` object is added (prompt and predicted tokens
per second, draft counts), which lets existing llama.cpp benchmark clients read WHIRL's numbers.

```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+3?"}],"temperature":0,"max_tokens":200}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = c.chat.completions.create(
    model="x",
    messages=[{"role": "user", "content": "Write a Python quicksort"}],
    temperature=0.6, seed=1,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
print(r.choices[0].message.content)
print(getattr(r.choices[0].message, "reasoning_content", None))
```

**Greedy consistency.** With temperature 0 the server's output equals the CLI's output token for
token (chat, completions, streaming, thinking on and off). A permanent gate checks this for a server
started with **no environment variables and no options**, because one bug produced garbage only on
that default path (it re-loaded kernel tables after the KV format was chosen at load time;
[pitfalls.md](pitfalls.md#srv-defaultenv)).

## <a id="sampling"></a>3. Sampling

- Parameters: `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens` (or
  `max_completion_tokens`), `stop`.
- Without sampling parameters, the model card's recommendations apply: thinking mode
  temperature 0.6 / top_p 0.95 / top_k 20; non-thinking 0.7 / 0.8 / 20.
- The GPU selects the top 256 candidates; the CPU applies the filters and draws.
- Random numbers come from (seed, token index): a seeded request produces the same text alone or
  concurrently, with MTP on or off.
- Speculative sampling: a draft is accepted with probability p(draft); on rejection the token is
  drawn from the distribution with the draft removed and renormalized — the output distribution is
  the same as without MTP ([speculative-decoding.md](speculative-decoding.md#sampling)).
- `presence_penalty` / `frequency_penalty` are accepted but **ignored** (a `W` log line says so);
  `n` must be 1; no `logprobs`; no `response_format`.

## <a id="thinking"></a>4. Thinking and `reasoning_content`

- Thinking is controlled by `chat_template_kwargs.enable_thinking` or a top-level `enable_thinking`;
  `reasoning_effort` and `preserve_thinking` are supported as in the GGUF chat template. The
  OpenRouter-style `"reasoning": {"effort", "enabled"}` object and effort aliases (`max` / `ultra`,
  `minimal`, `none`) are accepted too; see [usage.md](../../usage.md#endpoints).
- Thinking text is returned in `message.reasoning_content` (streaming: `delta.reasoning_content`),
  the answer in `content`.
- Most clients do not send `reasoning_content` back in the next turn. The template then renders an
  empty `<think>\n\n</think>` block, which tokenizes differently from the `<think>\n` that ended the
  previous prompt. The server saves a `think-open` checkpoint so such turns still hit the prefix
  cache ([kv-and-caching.md](kv-and-caching.md#think-open)).

## <a id="tools"></a>5. Tool calls

- The supported models emit tool calls in an XML-style format:
  `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>`. The server
  converts them into OpenAI `tool_calls`; JSON-style calls are parsed too. Parameter values are
  converted to the types declared in the tool's JSON schema.
- Tool results in the history (`role: tool`) and assistant tool calls are rendered by the chat
  template. Deliberate differences from the reference Jinja rendering (documented in
  [phase1_parity.md](../../phase1_parity.md)): `arguments` given as a JSON string are parsed into
  parameters; a tool call without a function name and unknown roles are rejected instead of being
  rendered or skipped.
- `tool_choice: "none"` removes the tools from the prompt. `"required"` or a named function is
  **not enforced** — there is no grammar-constrained decoding.
- Tool-call ids are random; tests compare name + arguments only.

## <a id="batching"></a>6. Continuous batching

**Slots.** `--parallel N` (default 4, max 16) slots, each with its own KV page list in the shared
pool, DeltaNet state, MTP hidden state, sampler state and prefix checkpoints. A new request goes to
the slot that can reuse the longest prefix (or restore one from the tiers).

**Decode cycle for all decoding slots together:**

1. MTP drafts for all slots in one multi-sequence batch (or n-gram drafts per slot).
2. One verify forward pass for all slots: GEMV rows span all sequences; attention rows use their own
   KV pages and positions; DeltaNet uses segmented kernels, one state per segment.
3. One synchronization per cycle.

**Draft limits by load.** All decoding rows fit in one verify of at most 16 rows, so drafts compete
with users. Dense caps by number of decoding slots: 8 / 7 / 4 / 3 for 1–4 slots; the cost model
picks within the cap. MoE: 1 draft up to 8 users. With zero drafts the MTP layer still processes
accepted tokens to keep its KV complete.

**Prefill scheduling.** Prefill runs in chunks of ≤ 1024 tokens (the last chunk may absorb ≤ 256
extra), interleaved with decode cycles. Several requests' chunks are combined into one segmented
forward: row-wise work (embeddings, norms, GEMMs, MoE) runs once over all rows; per-sequence work
(attention, DeltaNet conv and chunk scan) runs per segment with the solo kernels. The oldest
prefilling request always runs; others join if their next chunk has more than 16 rows (to avoid
the small-batch path) and the total stays ≤ 4096. Because every GEMM configuration and MoE tile is
row-invariant, a batched prefill row equals the solo row bit for bit. When no slot is decoding,
consecutive chunks of one request merge into forwards of up to 2048 rows
([kv-and-caching.md](kv-and-caching.md#merge)). While slots decode, the decode floor below limits
the forward size.

**Decode floor (`--decode-min-tps N`, default 20).** Without it, the loop runs one prefill forward
(up to 4096 rows of combined chunks, ~1.5 s on the 27B model) per decode cycle, so a streaming request
advances one cycle per forward while other requests prefill long prompts — about 3 tok/s, which looks
frozen.

The decode floor protects **every decoding slot except those of the same burst**: a decoding slot is
not protected only if it arrived within the burst-gathering window (30 ms) of a request that is still
prefilling. Requests that arrive together (Scenario A: C = 4 at once) therefore do not hold one
another back and merge their prefill at full speed, while a stream that started earlier — or a
conversation's next turn that arrives after the subagent prompts, while they still prefill — is
protected regardless of arrival order.

With a floor, and only while protected slots are decoding:
- a prefill forward carries at most a row budget: whole schedule chunks (the oldest request's next
  chunk always runs, so prefill always progresses), sized from the measured prefill rate so that one
  forward's stall is about 10 tokens' worth at the floor rate (≈ 1 chunk at N = 20);
- after each prefill forward, decode cycles run alone until every protected decoding slot has produced
  ≥ N tokens per second over the period that began with that forward. The cycle count adapts every cycle
  to the actual tokens (MTP acceptance, number of decoding slots) and to the measured prefill time
  (EMAs of decode-cycle time, tokens per cycle and prefill rows/s);
- if the floor is out of reach even by pure decoding, decode-only time per period is capped at 4 × the
  period's prefill time, so prefill keeps about 20% of the GPU.

When no protected slot is decoding, prefill runs exactly as without the floor (full-size merged forwards),
and burst gathering is unchanged. Only the grouping of rows into forwards and the timing of decode cycles
change; every GEMM / MoE tile is row-invariant, so outputs are bit-identical for any N (checked: N = 0 vs 20 at 25.7k, 97.4k
and 123.7k context and in reverse order, N = 0 / 10 / 20 / 30 / 40 at 25.7k in v0.1.1, the
short-prompt runs and Ornith). A side effect: with a small row budget the waiting
prompts prefill oldest-first instead of side by side, so the first one is answered much earlier and
the last one later.

| Scenario B: Swift-1.5 27B MXFP4-A, R9700: main stream at various contexts, then 3 subagent prompts (15.8k / 17.1k / 18.9k tokens) arrive | 25.7k context (N = 0) | 25.7k context (**N = 20**) | 97.4k context (**N = 20**) | 123.7k context (**N = 20**) |
|---|---|---|---|---|
| Main stream rate while they prefill, tok/s | 3.2 | **23.7** | **23.0** | **23.1** |
| Main stream longest pause, s | 1.217 | **0.499** | **0.482** | **0.479** |
| The 3 subagents' TTFT, s | 17.0 / 17.8 / 18.5 | **8.5 / 17.0 / 27.3** | **10.1 / 18.8 / 28.6** | **9.7 / 22.2 / 31.8** |
| Subagents prefill throughput while streaming, tok/s | 2797 | **1897** | **1813** | **1629** |

Ornith-1.5-35B-A3B MXFP4 (MoE), same scenario: N = 0 → 20 raises the streaming request from 6.7 to
31.5 tok/s (alone ~230; longest pause 0.39 → 0.43 s) and moves the last TTFT from 5.8 to 6.8 s
(prefill 8986 → 7653 tok/s). Short prompts are unaffected: Swift MXFP4-A, ~1.1k-token prompts, 256
generated, Scenario A (C = 4 concurrent arrival) finishes in wall 4.67 s with 288.8 tok/s decode
in steady state without triggering the floor, merging 3 requests into one prefill forward. 20 is
the default: the stream stays readable (> 20 tok/s, pauses < 0.5 s) for about a third of the prefill
throughput, while the mean TTFT of the waiting prompts stays the same. Use 0 for batch jobs where
only total throughput matters, or a higher N for a smoother stream at the cost of slower prefill.

**Burst gathering.** Requests that are still being received, parsed or tokenized are counted; while
any exist, a new request's first prefill chunk waits up to 30 ms so a burst of requests enters the
same segmented forward. A single user never waits (their own request is already queued). Without
it, the first request of a burst prefilled alone and then stalled 2.4 s while the others prefilled.

**Stalls.** When another user submits a 14k-token prompt, a streaming request's longest pause was
0.90 s (MoE: 0.25 s) before the decode floor; with three long prompts at once it was 1.22 s and the
stream fell to 3 tok/s (table above), 0.48–0.50 s and 23 tok/s with the default floor.

| Concurrency (27B Q4_K_M / MXFP4, ~1.1k-token prompts, 256 generated, greedy) | C = 1 | C = 2 | C = 4 |
|---|---|---|---|
| Wall-clock aggregate tok/s (incl. prefill), MTP + n-gram | 61.6–62.4 | 98.0–99.6 | 138.7–139.6 |
| MXFP4, same | 68.5–70.7 | 118.9–119.3 | 190.3–190.7 |
| Steady state inside decode loop, Q4_K_M / MXFP4 (Scenario A) | 109.5–110.1 | — | 252.9 / **288.8** |
| llama.cpp ROCm `-np 4`, MTP on (older measurement) | 39.8 | 43.3 | 64.1 |

Every concurrent output equals the same request run alone (gated). Earlier 8/16-user measurements
(before several later optimizations; 4096 context per user): 27B no MTP 100.0 / 115.5 tok/s
aggregate; at 16 users MTP no longer helps (113.1 vs 115.5) because 16 rows leave no room for
drafts, and 16-row GEMV is already compute-bound.

## <a id="logging"></a>7. Logging

Each line starts with local time `yyyy-mm-dd HH:MM:SS.mmm` and a level `I` / `W` / `E`. Content
follows llama.cpp's server where it makes sense: startup information, each request's parameters,
reused prefix, prefill speed, periodic generation speed (about every 100 tokens), a final summary,
MTP acceptance and the finish reason. Examples (timestamps omitted):

```
I req 6 | POST /v1/chat/completions | stream off | 1 messages, 0 tools (tool_choice auto), thinking on, reasoning_effort medium
I req 6 | prompt 35 tok, common prefix 0, reused 0 tok (no usable checkpoint), prefill 35 new
I req 6 | gen: n_gen 101, tg 70.00 t/s, draft acceptance 59.3% (2.78 tok/cycle)
I req 6 | MTP: drafted 216, accepted 128, acceptance rate 59.3%, 2.78 tokens per cycle (72 verify cycles)
I req 6 | finish_reason length, completion 200 tok, reasoning 954 B, content 0 B, tool_calls 0, cache now 235 tok
I req 23 | prompt 31 tok, common prefix 31, reused 31 tok (checkpoint 'prompt-end' @31), prefill 0 new
I req 45 | queued (2 request(s) ahead)
I batch | slots busy 0/4 (decode 0, prefill 0) | 80 cycles, 3.79 slots/cycle, 11.44 rows/cycle, 2.09 drafts/cycle | aggregate tg 158.7 tok/s, prefill 0 tok/s over 4.9 s
```

Other lines worth knowing: the startup `VRAM use:` line (weights + buffers, slots, other,
checkpoints, KV pool, left) — the quickest way to see what a change costs in VRAM; the KV format
choice with its reason (e.g. `(auto: the needed pool does not fit as f16)`); `shared prefix:` and
`checkpoint 'system' @B` for system-prompt reuse; one line per tier spill / restore / SSD write and
an idle `kv tier |` summary; `vision:` lines for image input.

## <a id="limits"></a>8. Known limitations

- Only `qwen35` / `qwen35moe` GGUFs ([architecture.md](architecture.md)).
- No grammar-constrained decoding: `tool_choice: "required"` and `response_format` are not enforced.
- Penalties accepted but ignored; `n` = 1 only; no `logprobs`.
- No HTTP keep-alive (every response is `Connection: close`); no shutdown endpoint (stop the
  server with Ctrl+C, see section 1).
- Streaming output is written on the engine thread; a very slow client can delay the batch.
- Cached vs uncached runs are numerically equivalent, not bit-identical (history KV computed by
  decode kernels vs prefill GEMMs; [kv-and-caching.md](kv-and-caching.md#think-open)). Requests with
  a ≥ 2048-token system message are split at the system boundary and so are not bit-identical to a
  binary that does not split.
- With the MoE model, near-tied expert routing can make a multi-turn cached conversation diverge from
  an uncached one (seen once in a thinking-mode test); both are valid computations.
- The auto draft count depends on measured timing, so speed varies by about ±2–4% between runs;
  output does not.
- Image input: Qwen3-VL style mmproj without deepstack only; no video; `http(s)` image URLs are not
  fetched ([vision.md](vision.md)).

## 9. How the server is tested

Every change runs the full server suite against both the dense and the MoE model (50 checks each),
including: endpoints; CLI MTP == no MTP; server == CLI for chat, completions and streaming; three
concurrent requests == sequential; multi-turn with cache == without (KL criterion when texts
differ); the **default-environment** phase (no variables, no options); segmented prefill (8 requests
of 160–2,489 tokens at once, each == solo); pool eviction and re-computation; multi-turn cache hit
ratios; tier, system-checkpoint and restore-under-load gates; and MXFP4-specific concurrency (C = 4
greedy == solo). See [benchmarking.md](benchmarking.md#gates).
