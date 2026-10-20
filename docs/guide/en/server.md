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
| `--kv-ram-mb N` | pinned RAM KV tier size; 0 disables RAM and SSD tiers | 1/4 of physical RAM within [max(8 GiB, one full f16 session + checkpoints), 32 GiB], at most half of the RAM available at startup; 16 GiB on a 64 GB PC; integrated GPU (Radeon 8060S): OS-visible RAM / 8 within [one max-context entry (about 6 GiB for the 27B model at 262k), 16 GiB], at most 1/4 of the RAM available at startup, off under 32 GB of OS-visible RAM (the BIOS VRAM carve-out does not count; 8 GiB on a 64 GB machine) |
| `--kv-ssd-dir` / `--kv-ssd-gb N` | SSD tier directory / size cap; 0 GB disables SSD | per-user local app-data directory / 64 |
| `--kv-ssd-mode shutdown\|always\|off` | when the SSD tier writes (shutdown = only reused entries, ever: when WHIRL is closed — Ctrl+C, console window closed, logoff, OS shutdown; the last three allow only about 5 s — when RAM must evict them, and after about 5 minutes idle; a one-shot prompt is never written; always = idle sessions after `--kv-ssd-delay`; off = no SSD files) | shutdown |
| `--kv-ssd-delay SEC` | `always` mode: write a session once unchanged for SEC seconds | 300 |
| `--log-file` | log file (also printed to the console) | per-user local app-data directory |
| `--alias` | model id reported by `/v1/models` | GGUF file name |
| `--mmproj FILE`, `--vis-idle-s`, `--vis-mode`, `--vis-cache-mb` | image input ([vision.md](vision.md)) | off / 60 / auto / 1024 |
| `--cors-origin ORIGIN` | extra web-page origin allowed to call the server from a browser (repeatable; `*` = any page, the pre-v0.1.4 behaviour); a `POST` from any other page is refused (403) | only `http(s)://localhost`, `127.0.0.1`, `[::1]` pages |
| `--allow-host NAME` | extra `Host` header name the server answers (repeatable; `*` = any); other names get 403 (DNS rebinding guard) | `localhost`, `127.0.0.1`, `[::1]`, the `--host` address |

Useful environment variables (all of them: [usage.md](../../usage.md#env)):

| Variable | Effect |
|---|---|
| `WHIRL_KV=auto\|f16\|q8\|q8h\|q8v` | KV format; `auto` applies the floor rule ([kv-and-caching.md](kv-and-caching.md#kv-auto)) |
| `WHIRL_POOL_BUDGET_MARGIN_MB` | room kept free under the WDDM budget for other programs to grow (default: the larger of 2 GiB, 3 GiB with a monitor on the GPU, and 6% of the budget; other programs' VRAM that the budget does not show yet is kept free on top; [details](#vram-log)) |
| `WHIRL_POOL_RESERVE_MB` | old free-VRAM-only rule: VRAM reserve after the pool, WDDM budget not applied (default reserve 768 dense / 1536 MoE under the free VRAM) |
| `WHIRL_NO_PREFIX_CACHE=1` | disable prefix caching |
| `WHIRL_PREFIX_CACHE_VERIFY=1` | after every cache hit, re-prefill and compare logits (diagnostic) |
| `WHIRL_SERVE_CKPTS` | checkpoints per slot (default 4 with one slot, 3 up to 4 slots, else 1; at most 8) |
| `WHIRL_TAIL_CK=0` | debug: no tail checkpoint near the end of prompts >= 4096 tokens |
| `WHIRL_MTP=0`, `WHIRL_NGRAM=0` | disable MTP / n-gram co-drafting |
| `WHIRL_MTP_BATCH_DRAFTS=d1,d2,…` | draft caps by number of decoding slots |
| `WHIRL_GATHER_MS` | burst-gather window for new requests (default 30, 0 = off) |
| `WHIRL_DECODE_MIN_TPS` | decode floor (= `--decode-min-tps`, default 20, 0 = off) |
| `WHIRL_PREFILL_CHUNK` | max rows per merged prefill forward (multiple of 1024, default 2048; Radeon 8060S MoE models 4096 = the prefill batch; a forward never exceeds the batch) |
| `WHIRL_SYS_MIN`, `WHIRL_SYS_CKPTS`, `WHIRL_SYS_LCP=0` | system-prompt checkpoint threshold (2048), VRAM count (2), disable `prefix` checkpoints |
| `WHIRL_KV_TIER_MIN` | minimum entry size for the tiers (2048 tokens) |
| `WHIRL_TIMING_RESET=0` | keep the MTP timing table across requests (not recommended) |
| `WHIRL_LOOP_LOG=1`, `WHIRL_TIER_VERIFY=1`, `WHIRL_TIER_MIN_GAIN=n`, `WHIRL_TIMER_PROBE=1` | diagnostics: per-loop phase timing, byte-verify every spill/restore, restore threshold (default 512), actual sleep length before / after the 1 ms timer resolution the server sets |
| `WHIRL_PROFILE=1\|2` | per-cycle GPU breakdown; **distorts timing** — never use it for speed numbers |

Only one engine process may use a GPU at a time: the executable takes a named mutex per device and a
second instance waits ([windows-hip.md](windows-hip.md#wddm-demote)).

**Stopping.** Ctrl+C, Ctrl+Break, closing the console window, logoff or OS shutdown stops the server gracefully: queued
requests get 503, running ones finish, finished sessions spill to the RAM tier and the RAM-tier
entries not yet current on the SSD tier are written now: by default (`--kv-ssd-mode shutdown`) every
reused entry — a one-shot prompt is not written when WHIRL is closed, nor on RAM eviction or after
about 5 minutes idle, the other two times reused entries are written; in `always` mode the entries its
seen-once filter keeps (normally written 5 minutes after their last change). At most 60 s (10 s in `always` mode;
`shutdown: tier writes done …` in the log), so the next start restores the last turns
from the SSD. A second Ctrl+C exits at once; for a console close, logoff or OS shutdown Windows ends the process after
about 5 s whatever is left. Killing the process (Task Manager, `taskkill /F`) skips all of this.

## 2. API surface

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | streaming (SSE) or not; tools; thinking; images (with `--mmproj`) |
| `POST /v1/completions` | raw prompt |
| `GET /v1/models` | one model |
| `GET /health` | answers immediately even while busy; reports busy state and queue length |
| CORS preflight | supported; `Access-Control-Allow-Origin` only for allowed origins (`--cors-origin`) |

Responses are standard OpenAI JSON: `finish_reason` `stop` / `length` / `tool_calls`; `usage`
includes `cached_tokens`; a llama.cpp-style `timings` object is added (prompt and predicted tokens
per second, draft counts), which lets existing llama.cpp benchmark clients read WHIRL's numbers.
`completion_tokens` counts the tokens up to the one that completes a stop string (drafts accepted
after it in the same cycle are not counted). A generation cut by `max_tokens` while it writes a tool
call ends with `finish_reason: "length"` and the call as text, never as a half `tool_calls` entry.
A `"finish_cause": "kv_pool_exhausted"` next to `finish_reason: "length"` means the KV pool, not
`max_tokens`, ended it. A non-streaming request whose client disconnects stops (queued, prefilling
or generating) and frees its slot; a client that half-closes its socket after sending counts as
disconnected. During shutdown new requests get 503.

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

**Draft limits by load.** All decoding rows fit in one verify forward, so drafts compete with users.
The verify holds at most 16 rows per sequence and, with wide verify (`WHIRL_WIDE_VERIFY`, default on),
at most 32 rows in total (16 otherwise). MoE models stay at 16 rows by default; `WHIRL_MOE_WIDE=1` (balance / fast)
allows 32, the 17–32-row batches on the same int8 decode experts as a 1-token decode (same output, but
each row reads its own experts: about 2% slower at four users on Ornith-1.5-35B-A3B).
Default dense caps by number of decoding slots: 8 / 7 / 4 / 3 / 2 / 1 / 1 / 1 for 1–8 slots, 0 beyond
(`--batch-drafts` overrides; when DeltaNet verify uses snapshot sets instead of replay, the cap for
k slots is also ≤ 8 / k); with these caps an MTP verify stays at ≤ 16 rows, and only n-gram drafts
or raised caps with several users use rows 17–32. The cost model picks within the cap. MoE: 1 draft
up to 8 users. Models whose DeltaNet decode is not fused (F32 / F16 α/β) are capped at 1 draft.
With zero drafts the MTP layer still processes
accepted tokens to keep its KV complete.

**Prefill scheduling.** Prefill runs in chunks of ≤ 1024 tokens (the last chunk may absorb ≤ 256
extra), interleaved with decode cycles. Several requests' chunks are combined into one segmented
forward: row-wise work (embeddings, norms, GEMMs, MoE) runs once over all rows; per-sequence work
(attention, DeltaNet conv and chunk scan) runs per segment with the solo kernels. The oldest
prefilling request always runs; others join if their next chunk has more than 16 rows (to avoid
the small-batch path) and the total stays ≤ 4096. Because every GEMM configuration and MoE tile is
row-invariant, a batched prefill row equals the solo row bit for bit. When no other slot is
decoding, or with the decode floor off (`--decode-min-tps 0`, throughput first), consecutive chunks
of one request merge into forwards of up to 2048 rows ([kv-and-caching.md](kv-and-caching.md#merge)).
With the floor on, any decoding slot (protected or of the same burst) keeps forwards at whole chunks:
measured on the 27B model with 3 short requests decoding next to a 32k / 96k / 128k prompt of the
same burst, merging raised that prompt's prefill throughput by only 3.7–4.2% but doubled the decoders' longest
pause (0.54 → 1.03 s, 1.17 → 3.27 s, 1.85 → 2.37 s).

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

When no protected slot is decoding, the floor holds nothing back (no row budget, no decode-only
cycles), and burst gathering is unchanged. Only the grouping of rows into forwards and the timing of
decode cycles change; every GEMM / MoE tile is row-invariant, so outputs are bit-identical for any N
(checked for every N in the tables below, in both arrival orders, against v0.1.2 at N = 20 for all
three contexts, the short-prompt runs and Ornith). A side effect: with a small row budget the waiting
prompts prefill oldest-first instead of side by side, so the first one is answered much earlier and
the last one later.

Measured on Swift-1.5 27B MXFP4-A, R9700, 3 subagent prompts of 15.8k / 17.1k / 18.9k tokens.
**Forward order**: the main stream decodes, then the 3 prompts arrive. **Reverse order**: the 3
prompts arrive first and the conversation's next turn (a short delta on its cached context) 1.5 s
later; "while they prefill" then runs from the main stream's first token to the last subagent's first
token.

| 25.7k context | N = 0 (off) | 10 | **20** | 30 | 40 |
|---|---|---|---|---|---|
| Forward: main stream while they prefill, tok/s (alone ~84) | 2.8 | 11.9 | **22.7** | 31.0 | 40.3 |
| Forward: its longest pause, s | 1.57 | 1.12 | **0.47** | 0.47 | 0.46 |
| Forward: the 3 prompts' TTFT, s | 11.1 / 12.0 / 18.5 | 16.7 / 20.5 / 21.9 | **7.7 / 16.3 / 26.1** | 9.2 / 19.7 / 31.1 | 11.2 / 23.3 / 38.6 |
| Forward: mean TTFT, s | 13.9 | 19.7 | **16.7** | 20.0 | 24.4 |
| Forward: prefill throughput while streaming, tok/s | 2799 | 2368 | **1981** | 1663 | 1342 |
| Reverse: main stream while they prefill, tok/s | 4.1 | 14.6 | **22.8** | 33.3 | 42.3 |
| Reverse: its longest pause, s | 1.07 | 0.52 | **0.47** | 0.47 | 0.47 |
| Reverse: the 3 prompts' TTFT, s | 11.8 / 12.6 / 18.3 | 12.2 / 12.5 / 19.6 | **11.0 / 12.4 / 21.9** | 11.6 / 11.9 / 24.2 | 11.5 / 11.9 / 28.2 |

| N = 20, by context | 25.7k | 97.4k | 123.7k |
|---|---|---|---|
| Forward: main stream while they prefill, tok/s | 22.7 | 23.0 | 23.1 |
| Forward: its longest pause, s | 0.47 | 0.47 | 0.52 |
| Forward: the 3 prompts' TTFT, s | 7.7 / 16.3 / 26.1 | 8.4 / 19.1 / 29.0 | 11.8 / 21.7 / 32.2 |
| Forward: prefill throughput while streaming, tok/s | 1981 | 1789 | 1609 |
| Reverse: main stream while they prefill, tok/s | 22.8 | 21.1 | 22.9 |
| Reverse: its longest pause, s | 0.47 | 0.48 | 0.48 |
| Reverse: the 3 prompts' TTFT, s | 11.0 / 12.4 / 21.9 | 11.4 / 11.9 / 25.1 | 11.3 / 11.7 / 23.6 |

v0.1.2 measured 23.1 / 22.7 / 22.1 tok/s forward and 22.4 / 21.5 / 22.4 tok/s reverse at the same
three contexts (longest pause 0.47–0.52 s); at 123.7k the pause is 0.52 s in both versions. An earlier
build of this release that protected only streams arriving before every prefilling prompt fell to
8.6 tok/s (pause 0.92 s) in reverse order at 25.7k.

Ornith-1.5-35B-A3B MXFP4 (MoE), same scenario (v0.1.1): N = 0 → 20 raises the streaming request from
6.7 to 31.5 tok/s (alone ~230; longest pause 0.39 → 0.43 s) and moves the last TTFT from 5.8 to 6.8 s
(prefill 8986 → 7653 tok/s). Short prompts are unaffected: Swift MXFP4-A, ~1.1k-token prompts, 256
generated, Scenario A (C = 4 concurrent arrival) finishes in 4.65 s wall (median of 3; v0.1.2 4.91 s)
without triggering the floor, merging 3 requests into one prefill forward. 20 is the default: the
stream stays readable (> 20 tok/s, pauses ≤ 0.5 s) for about 30% less prefill throughput. Use 0 for
batch jobs where only total throughput matters, or a higher N for a smoother stream at the cost of
slower prefill.

**Burst gathering.** Requests that are still being received, parsed or tokenized are counted; while
any exist, a new request's first prefill chunk waits up to 30 ms so a burst of requests enters the
same segmented forward. A single user never waits (their own request is already queued). Without
it, the first request of a burst prefilled alone and then stalled 2.4 s while the others prefilled.

**Stalls.** When another user submits a 14k-token prompt, a streaming request's longest pause was
0.90 s (MoE: 0.25 s) before the decode floor; with three long prompts at once it was 1.57 s and the
stream fell to 3 tok/s (table above), 0.47–0.52 s and 21–23 tok/s with the default floor.

| Concurrency (27B MXFP4, ~1.1k-token prompts, 256 generated, greedy) | C = 1 | C = 2 | C = 4 |
|---|---|---|---|
| Wall-clock aggregate tok/s (incl. prefill), MTP + n-gram | 68.5–70.7 | 118.9–119.3 | 190.3–190.7 |
| Steady state inside decode loop (Scenario A) | — | — | **288.8** |

Every concurrent output equals the same request run alone (gated). Earlier 8/16-user measurements
(before several later optimizations; 4096 context per user): at 16 users MTP no longer helps
because 16 rows leave no room for drafts, and 16-row GEMV is already compute-bound. (These predate wide verify; with the default
caps, 9–16 slots still get 0 drafts, so the conclusion holds. Rows 17–32 have no measured per-row
cost yet.)

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

### <a id="vram-log"></a>Video memory while it runs (`vram`, "GPU memory spill")

Decode that gets several times slower in a long session (GPU usage still 100%, GPU power down) is
often Windows moving part of WHIRL's GPU memory to system RAM. WHIRL sizes its KV pool to stay inside
the Windows (WDDM) video memory budget, and these lines show whether it still does:

- `other processes use X GiB of VRAM on this GPU (display attached: yes/no)` at startup: what other
  programs (desktop, browser, other AI tools, Docker) hold on the same GPU when the server starts.
- `kv pool sizing: ... margin N MiB (...)`: what is kept free under the WDDM budget. The budget Windows
  reports when the server starts does not always count other programs yet: on an R9700 with another
  process holding 3.3 GiB, it read 31.02 GiB before the pool was allocated and 28.08 GiB right after.
  So the margin is what other processes used at startup that the budget does not show yet, plus room
  for them to grow: the larger of 2 GiB (3 GiB when a monitor is attached to this GPU) and 6% of the
  budget. `WHIRL_POOL_BUDGET_MARGIN_MB=N` sets the room.
- `adjusted: ...` at startup: a setting you gave that does not fit what the server measured (RAM, the
  WDDM budget with the margin, other programs' VRAM) is lowered to what fits, one line each with the
  reason, in this order: the RAM tier (`--kv-ram-mb`, more than half of the available RAM: lowered to
  the default size), `--parallel` (each slot's state and checkpoints are fixed VRAM; lowered when one
  full request no longer fits next to them), `WHIRL_PREFILL_BATCH` / checkpoint counts, `-c` (the
  pool), then `--ctx-per-slot` (it reserves no memory by itself, so it is capped to the pool last).
  `startup adjustments (N): ...` sums them up (`none` when everything fits), and `GET /props` and
  `GET /health` show the effective values in a `startup` object. A prompt longer than an adjusted
  context per request is rejected as too long for the context, and the log says it was adjusted. The
  server refuses to start only when even one slot with a 4096-token pool does not fit. Not on the
  Radeon 8060S, which keeps its carve-out check.
- `kv pool check:` right after the KV pool is allocated: the budget is read again; when WHIRL's usage
  plus the margin no longer fits, the pool is made smaller (whole pages, before any request) so WHIRL
  starts inside VRAM. It waits for the WDDM numbers to settle after each free (at most 1 s), so it does
  not shrink more than needed. Output is unchanged; only fewer tokens fit in the pool. With an explicit
  `WHIRL_POOL_RESERVE_MB` the pool is not resized; a warning says how many tokens would fit. An
  explicit `-c` or `--ctx-per-slot` that no longer fits is lowered with an `adjusted:` line.
- `vram watch:` after everything is allocated: the numbers, and the cost of one sample (about 0.3 ms
  on the R9700).
- Every `batch |` status line (about every 5 s while a request runs) ends with
  `| vram local X / budget Y GiB, shared Z MiB (+D MiB since start), ram R GiB (pinned P GiB)`, and
  each request end adds a `req N | slot S | vram ...` line next to its `finish_reason` line. While
  idle the server takes one sample a minute (no line unless a spill starts or ends), so `/health`
  stays current.
  - `local` is WHIRL's GPU memory in VRAM, `budget` the share of VRAM Windows lets WHIRL keep there
    right now (it drops when other programs use the GPU).
  - `shared` is WHIRL's "Shared GPU memory" (Task Manager > GPU): GPU buffers that live in system RAM.
    Part of it is there from the start on purpose (pinned host memory such as the RAM tier of the
    prefix cache), so what matters is the growth since the KV pool was sized (`+D MiB`). Growth beyond
    the pinned memory WHIRL itself allocated later likely means Windows moved GPU buffers out of VRAM:
    every access to them crosses PCIe, and decode slows down when they are used at every step.
  - `ram` is the process working set, `pinned` the pinned host memory WHIRL allocated.
- `W GPU memory spill: ...` when local usage goes over the budget or shared grows by more than
  256 MiB since the pool was sized (not counting pinned memory WHIRL allocated since): what happened,
  what it does to generation, what to do (close other programs that use the GPU; or restart the
  server, which sizes the pool from the budget at startup, with `WHIRL_POOL_BUDGET_MARGIN_MB=4096` or
  more to keep more room for other programs), the pool size, how much VRAM other programs use now and
  at startup, and the numbers. The same text is printed at startup when the spill is already there.
  While it lasts it repeats at most once a minute (`GPU memory spill continues (N so far)`);
  `I GPU memory spill is over` when it ends.
- `req N | slot S | decode slowed from X to Y tok/s (...)` once per request when a window of its
  decode is more than 40% slower than its first 200 tokens. It names the spill as the likely cause
  (a warning) only when a spill is seen and each decode cycle got slower; when each cycle is not
  slower it says fewer tokens per cycle (fewer drafts accepted); with `--parallel 2` or more it says
  when another request's prefill shared the GPU in that window.
- `GET /health` carries the last sample: `"vram":{"local_mib":..,"budget_mib":..,"shared_mib":..,"spill":false}`.

`WHIRL_POOL_RESERVE_MB` is the old free-VRAM-only rule: it turns the WDDM budget, the margin and the
pool check off. To keep more room for other programs, use `WHIRL_POOL_BUDGET_MARGIN_MB` instead.

On the Radeon 8060S (integrated, "shared" is the same RAM) the numbers are logged without the spill
warning; the pool there has a fixed default size (262144 tokens, checked against the carve-out at
startup), so the margin and the pool check do not apply.

## <a id="limits"></a>8. Known limitations

- Only `qwen35` / `qwen35moe` GGUFs ([architecture.md](architecture.md)).
- No grammar-constrained decoding: `tool_choice: "required"` and `response_format` are not enforced.
- Penalties accepted but ignored; `n` = 1 only; no `logprobs`.
- No HTTP keep-alive (every response is `Connection: close`); no shutdown endpoint (stop the
  server with Ctrl+C, see section 1).
- Connection limits: at most 64 open connections (more get `503`); a request must arrive within
  60 s (headers; otherwise `408`) and with no more than 30 s between reads; headers are capped at
  64 KiB / 100 lines (`431`). The engine never waits for a client: a client that stops reading its
  response for 30 s is disconnected and its request stopped, without slowing the other slots.
- Cache hits on `system` / `prefix` checkpoints, on the same prompt's `prompt-end`, and tier restores
  are bit-identical to a cold run. Continuing a conversation (`gen-end`, or a prompt checkpoint followed
  by more tokens) is numerically equivalent but not bit-identical in balance / fast: the history's KV and
  DeltaNet state came from decode kernels ([kv-and-caching.md](kv-and-caching.md#checkpoints)). `--precise`
  never resumes there (COLD-RESUME, 0.2.2): only at checkpoints on the new prompt's chunk schedule that prefill
  produced, so every precise cache hit is bit-identical to a cold run. Requests with
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
