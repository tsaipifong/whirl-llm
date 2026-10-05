**English** | [繁體中文](recipes_zh-TW.md)

# Recipes: how to do common things with WHIRL

Short, task-oriented answers: "I want to do X — here is exactly how." Every option mentioned here
exists in WHIRL 0.1.1; the full reference is [usage.md](usage.md), the background is in
[server.md](guide/en/server.md) and [kv-and-caching.md](guide/en/kv-and-caching.md). Commands are for
PowerShell, run from the folder that contains `whirl-server.exe`. New here? Do the
[quick start](quickstart.md) first.

The examples use these file names (put yours in their place):

```powershell
$model  = "C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf"   # recommended dense 27B
$mmproj = "C:\models\mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf"        # only for image input
```

- [1. Connect an agent or chat front end](#connect)
- [2. Long agent sessions (agent + subagents)](#agents)
- [3. Long context (up to 256k tokens)](#long-context)
- [4. Several users or slots](#multi-user)
- [5. Images](#images)
- [6. Reading the server log](#log)
- [7. Troubleshooting](#troubleshooting)

## <a id="connect"></a>1. Connect an agent or chat front end

Start the server with a model id that is easy to type:

```powershell
.\whirl-server.exe $model --alias swift-27b --port 8080
```

Then enter these values in the client's "OpenAI-compatible" / "custom OpenAI" provider settings.
The field names differ from product to product; the values do not:

| Field (typical names) | Value |
|---|---|
| Base URL / API base / endpoint | `http://127.0.0.1:8080/v1` |
| API key | any non-empty text (WHIRL has no authentication), e.g. `none` |
| Model / model id | the `--alias` value (`swift-27b`); `GET /v1/models` lists it |
| Context window / max context | `--ctx-per-slot` (default 131072); see below |
| Provider type | "OpenAI-compatible", "OpenAI (custom base URL)", or similar — not an Ollama or LM Studio provider |

This works the same way for agent frameworks (for example Hermes agent), chat front ends (for example
Open WebUI) and editor assistants (Continue, Cline and other VS Code-style tools): pick the generic
OpenAI-compatible provider and fill in the table. If the client runs on another computer, start the
server with `--host 0.0.0.0` and use this PC's address — anyone who can reach it can then use the
server, as there is no API key check.

Check it from PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8080/props    | ConvertTo-Json -Depth 5   # context per slot, slots, alias
```

**Context length auto-detection (0.1.1).** Clients that probe llama.cpp-style `GET /props` read the
context per request from `default_generation_settings.n_ctx`; `GET /v1/models` reports it as
`meta.n_ctx`. Clients that do neither need the context window entered by hand: use the
`--ctx-per-slot` value. A client that probes Ollama / LM Studio paths (`/api/tags`, `/api/v1/models`,
…) gets 404 for them — that is expected; point it at the `/v1` base URL above.

**Thinking.** Thinking is on by default; the thinking text arrives in `reasoning_content`
(streaming: `delta.reasoning_content`) and the answer in `content`. To change it per request:

| You want | Send any one of |
|---|---|
| thinking off | `"reasoning_effort": "none"`, `"reasoning": {"enabled": false}`, `"chat_template_kwargs": {"enable_thinking": false}`, or `"enable_thinking": false` |
| shorter thinking | `"reasoning_effort": "low"` (also `minimal`) or `"medium"`, or `"reasoning": {"effort": "low"}` |
| full thinking (the default) | `"reasoning_effort": "high"` (also `xhigh`, `max`, `ultra`) |

An unknown effort value is logged as a warning and the default is used. Clients that cannot send
extra fields simply get thinking on.

**Tool calling** works with the standard OpenAI `tools` / `tool_calls` / `role: "tool"` messages,
streaming or not; argument values are converted to the types in each tool's JSON schema. Limits:
`tool_choice: "none"` removes the tools from the prompt, but `"required"` or a named function is
**not enforced** (no grammar-constrained decoding — the model decides); `response_format` (JSON
mode / JSON schema) is not supported; `n` must be 1; no `logprobs`; `presence_penalty` /
`frequency_penalty` are accepted but ignored. Details: [server.md](guide/en/server.md#tools).

## <a id="agents"></a>2. Long agent sessions (agent + subagents)

This is the scenario WHIRL's server was built for: an agent that re-sends a long, growing conversation
(system prompt, tool definitions, history) on every turn, sometimes with several subagents at once.

```powershell
.\whirl-server.exe $model --mmproj $mmproj --alias swift-27b --port 8080
```

Defaults that matter here: 4 slots (`-np 4`), up to 131,072 tokens per request, a KV pool that fills
the free VRAM, MTP + n-gram speculative decoding, `--decode-min-tps 20`, a pinned-RAM tier and a
64 GiB SSD tier. `--mmproj` is optional (it adds ~0.9 GiB of pinned RAM and no VRAM until an image
arrives, [§5](#images)).

**Why later turns start in under a second.** Each turn the server compares the new prompt with what
each slot already holds and resumes from the latest saved checkpoint inside the common prefix, so only
the new tail (the last tool result and the new user message) is prefilled. A hybrid model like
Qwen3.8 cannot resume at an arbitrary token — the DeltaNet state exists only at checkpoints (end of
prompt, end of generation, `think-open`, a long system message, a prefix shared with another session) —
which is why the reused count is a bit below the common prefix. Keep the start of the prompt stable:
a system prompt with a changing timestamp or a reordered tool list forces a full prefill.

**A real session** (Swift-1.5 27B MXFP4-A on the R9700, Hermes agent with 25 tools, sampling at
temperature 0.6, default server options):

| What | Measured |
|---|---|
| Context of a turn | ~31k tokens: 29,460 of 31,067 tokens reused from the prefix cache, 1,607 prefilled |
| Time to first token | 0.89 s |
| Decode, that turn | 78.6 tok/s (sampling, MTP + n-gram) |
| Decode, another turn at ~29k context | 98 tok/s, peaking at 112 tok/s (MTP + n-gram drafts accepted well) |
| 4 subagents at once | ~40–48 tok/s each, 180–195 tok/s aggregate |

**Subagents.** Each subagent is its own conversation in its own slot. Subagents usually share the
system prompt and tool definitions; a system message of ≥ 2048 tokens gets a shared `system`
checkpoint and prefixes common to several sessions get a shared `prefix` checkpoint, so a new
subagent skips that part of the prefill (a new conversation reusing a 26k-token system prompt:
0.12 s TTFT on this model, [benchmarks.md](benchmarks.md#8-cold-prefill-cached-prefixes-and-kv-restore-ttft)).
With more than 4 concurrent requests, the rest wait in a queue; raise `-np` (up to 16) if you run
the main agent plus 4 or more subagents at the same time. More slots means fewer drafts per slot per
cycle and, above 4 slots, one instead of two prefix checkpoints per slot.

**Smooth main stream while subagents send long prompts: `--decode-min-tps`.** While other requests
prefill long prompts, every streaming request is kept at ≥ N tok/s (default 20). Measured with one
request streaming at a 25.7k-token context while three 16–19k-token prompts arrive (Swift 27B MXFP4-A):

| `--decode-min-tps` | 0 (off) | 10 | **20 (default)** | 30 | 40 |
|---|---|---|---|---|---|
| Streaming request during their prefill, tok/s | 3.3 | 12.0 | **23.0** | 30.7 | 40.5 |
| Its longest pause, s | 1.22 | 1.13 | **0.47** | 0.46 | 0.48 |
| Mean TTFT of the 3 prompts, s | 17.8 | 19.3 | **18.6** | 21.3 | 29.0 |
| Prefill throughput meanwhile, tok/s | 2791 | 2341 | **1862** | 1551 | 1137 |

Use `0` for batch jobs where only total throughput matters, a higher value (30–40) if you watch the
main agent's stream and the subagents can wait. Outputs are identical for any value
([server.md](guide/en/server.md#batching)).

**RAM tier sizing.** Idle sessions are copied to pinned RAM (and from there to the SSD) so that a
session pushed out of VRAM, or a session from before a restart, is restored instead of prefilled
again. The default RAM tier is about 1/4 of system RAM (8–32 GB) from v0.1.2; v0.1.1 uses ~9 GB — set
`--kv-ram-mb` to override. One entry costs about 13 MiB per 256 tokens (q8v KV, the dense default;
17 MiB with f16) plus ~150 MiB per DeltaNet checkpoint (the startup `kv tier:` line prints both for
your model), so a 30k-token session is roughly 1.8–2.6 GiB. Give the tier enough for the sessions you
switch between; `--kv-ram-mb 0` turns both host tiers off.

```powershell
.\whirl-server.exe $model --alias swift-27b --kv-ram-mb 16384 --kv-ssd-gb 128   # bigger tiers
```

**Stop and restart without losing the cache.** Press **Ctrl+C** once: queued requests get 503,
running ones finish, and every RAM-tier entry that is not on the SSD yet is written (at most 10 s;
the log says `shutdown: tier writes done …`). A second Ctrl+C or `taskkill /F` skips this; closing
the console window also stops gracefully, but Windows ends the process after about 5 s. Start the server again with **the same model, executable and options that
affect numerics** (KV format, MTP on/off): the SSD index is scanned at startup and the next turn of a
recent session is restored (`kv tier: restoring … from SSD`) — 0.73 s TTFT for a 25.7k-token session
on this model. Entries written by a different build or numerical configuration are ignored.

## <a id="long-context"></a>3. Long context (up to 256k tokens)

```powershell
.\whirl-server.exe $model -np 1 --ctx-per-slot 262144
```

- `--ctx-per-slot` is the longest single request (default 131,072, maximum 262,144). With one user,
  `-np 1` leaves the whole KV pool to that user.
- The KV format is chosen automatically and printed at startup with its reason. Dense models use the
  first of **f16 → q8v → q8h** whose required pool fits: `-np 1` at the default context gets f16, the
  default 4 slots get q8v, `--ctx-per-slot 262144` gets q8h. In our paired QA (350 items, up to 70k
  contexts) no format differed measurably from f16; q8h decodes 3.6–5.2% slower at 64k–128k
  ([kv-and-caching.md](guide/en/kv-and-caching.md#formats)). Force one with `$env:WHIRL_KV = "q8v"`
  (or `f16`, `q8`, `q8h`) before starting the server.
- The MoE model (Ornith-1.5-35B-A3B) always uses f16 KV at 22 KiB per token (the 27B dense model:
  68 KiB in f16), so its default pool already holds ~414k tokens, and it decodes far faster at long
  context (301.5 vs 69.5 tok/s after 16k tokens, [benchmarks.md](benchmarks.md#6-decode-after-a-16k-token-context)).
- A cold long prompt takes time: Swift 27B MXFP4-A prefills 32k tokens at ~2,600 tok/s (~13 s); at
  256k the 27B model ran at ~400 tok/s (about 11 minutes) in our needle test. After that, the prefix
  cache and tiers make follow-up turns fast.
- **VRAM.** The server sizes its KV pool to fill the free VRAM, so the process always shows about
  30–31 GiB dedicated ([benchmarks.md §9](benchmarks.md#9-vram)) — that is expected, not a leak. To
  leave VRAM for other programs, cap the pool with `-c N` (tokens). Task Manager's "shared GPU memory"
  of 8–16 GiB is the pinned RAM tier, not spilled VRAM.

## <a id="multi-user"></a>4. Several users or slots

```powershell
.\whirl-server.exe $model --host 0.0.0.0 -np 8
```

- `-np N` (1–16, default 4) is the number of requests served at the same time; later ones queue
  (`queued (…)` in the log). All slots share one KV pool: a slot takes pages as its context grows, and
  when the pool is full the least recently used *idle* slot is evicted — spilled to the RAM / SSD
  tiers first, so it is restored when that user returns.
- Aggregate throughput, 4 concurrent users (~1.1k-token prompts, 256 tokens each, prefill included):
  Swift 27B MXFP4-A **198.6 tok/s**, Ornith MoE MXFP4 **396.7 tok/s**, Qwen3.8-27B Q4_K_M
  **134.1 tok/s** ([benchmarks.md §7](benchmarks.md#7-server-concurrency)). Every concurrent output
  equals the same request run alone.
- Speculative drafts compete with users inside one verify pass, so the gain per user shrinks as more
  users decode at once; at 16 users MTP no longer helps.
- `--host 0.0.0.0` listens on every network with no authentication: use it only on a network you
  trust (Windows Firewall may ask once).

## <a id="images"></a>5. Images

```powershell
.\whirl-server.exe $model --mmproj $mmproj --alias swift-27b
```

Send images as OpenAI `image_url` parts with a `data:` URL (base64 PNG / JPEG / …):

```powershell
$b64  = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\pics\chart.png"))
$body = @{ messages = @(@{ role = "user"; content = @(
          @{ type = "text"; text = "What does this chart show?" },
          @{ type = "image_url"; image_url = @{ url = "data:image/png;base64,$b64" } }) });
        max_tokens = 800 } | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body |
  ForEach-Object { $_.choices[0].message.content }
```

- `http(s)` image URLs are never fetched. Local paths / `file://` URLs work only with
  `--allow-local-images`.
- Cost: the encoder weights live in pinned RAM (~0.9 GiB) and nothing goes to VRAM until an image
  arrives. With the default 27B configuration VRAM is tight, so the weights are streamed per image:
  about 0.26–0.29 s per image on our USB4 eGPU, 0.30–0.54 s for the first encode
  ([benchmarks.md §10](benchmarks.md#10-vision-swift-mxfp4-a-with-mmproj-f16)). The encoder is
  released after `--vis-idle-s` (default 60) seconds without images.
- The same image again (in a later turn or another request) is not re-encoded: embeddings are cached
  by content hash (`--vis-cache-mb`, default 1024), and the prefix cache reuses the KV after it.
- Supported: Qwen3-VL-style mmproj, F16 / BF16; several images per request; no video
  ([vision.md](guide/en/vision.md)).

## <a id="log"></a>6. Reading the server log

The log goes to the console and to `%LOCALAPPDATA%\whirl\server.log` (`--log-file`). Each line starts
with the local time and a level `I` / `W` / `E`; request lines carry `req <id> | slot <n>`.
An abridged example of one agent turn (illustrative values, timestamps removed):

```
I req 412 | POST /v1/chat/completions | stream on | 58 messages, 25 tools (tool_choice auto), thinking on
I req 412 | slot 0 | prompt 31067 tok, common prefix 31040, reused 29460 tok (checkpoint 'think-open' @29460), prefill 1607 new
I req 412 | slot 0 | prefill: 1607 tok in 870.0 ms (1847.1 tok/s)
I req 412 | slot 0 | gen: n_gen 100, tg 96.50 t/s, draft acceptance 71.2% (3.85 tok/cycle)
I req 412 | slot 0 | MTP: drafted 610, accepted 434, acceptance rate 71.1%, 3.84 tokens per cycle (154 verify cycles)
I req 412 | slot 0 | n-gram drafts in 41 of 154 cycles
I req 412 | slot 0 | finish_reason tool_calls, completion 592 tok, reasoning 1210 B, content 0 B, tool_calls 1, cache now 31659 tok
I slot 0 | kv tier: spill 31488 tok to RAM (in place, pages 115..123, checkpoints 301 MiB, 418.0 MiB in all; RAM tier 6.10 / 9.00 GiB)
I batch | slots busy 1/4 (decode 1, prefill 0) | 154 cycles, 1.00 slots/cycle, 4.96 rows/cycle, 3.96 drafts/cycle | aggregate tg 92.4 tok/s, prefill 321 tok/s over 6.4 s
I kv tier: entry 3f2a9c0d81b7e655 (31488 tok) written to SSD: 418 MiB in 160 ms (file 1802 MiB)
I kv tier | RAM 6.10 / 9.00 GiB in 4 entries, SSD 12.40 GiB in 9 entries | spills 37 (9120 MiB), restores RAM 2 / SSD 1 (3410 MiB), …
```

| Line | What it tells you |
|---|---|
| `POST … \| stream on \| 58 messages, 25 tools (tool_choice auto), thinking on` | what the client sent; `reasoning_effort …` is appended when one was given |
| `prompt N tok, common prefix L, reused R tok (checkpoint 'K' @P), prefill M new` | N prompt tokens; L of them match what the slot held; it resumed from the saved checkpoint K at position P, so only M = N − R tokens are computed. `reused 0 tok (no usable checkpoint)` means a full prefill (new conversation or changed prompt start) |
| checkpoint kinds | `prompt-end` / `gen-end` (end of the last prompt / reply), `think-open` (lets a turn hit the cache when the client does not send the previous thinking back), `system` / `prefix` (shared with other sessions) |
| `prefill: M tok in X ms (Y tok/s)` | time of the new part; together with queueing this is about the time to first token |
| `gen: n_gen …, tg … t/s, draft acceptance …% (x tok/cycle)` | progress every ~100 tokens or 5 s: tokens so far, decode speed, share of drafts accepted, tokens produced per verify cycle (1.0 = no gain from speculation) |
| `MTP: drafted D, accepted A, acceptance rate …, x tokens per cycle (C verify cycles)` | the final speculation summary of the request |
| `n-gram drafts in X of Y cycles` | cycles that used prompt-lookup (n-gram) drafts — high when the model copies text from the context (code edits, quoting tool output) |
| `finish_reason …, completion … tok, reasoning … B, content … B, tool_calls …, cache now … tok` | how the request ended; `cache now` is what the slot keeps for the next turn |
| `batch \| slots busy b/n (decode d, prefill p) \| …` | every ~5 s while busy: slots in use, average slots / rows / drafts per cycle, aggregate decode and prefill tok/s |
| `batch \| decode floor: …` | how often the decode floor ([§2](#agents)) interleaved decode with prefill |
| `kv tier: spill … tok to RAM (…; RAM tier a / b GiB)` | an idle session copied to the RAM tier (`in place` = only the changed part) |
| `kv tier: entry … written to SSD` | that entry now also survives a restart |
| `kv tier: restoring … from RAM\|SSD` / `restored … in X ms (Y GB/s)` | a session brought back from a tier instead of prefilled |
| `kv tier \| RAM a/b GiB in n entries, SSD …` | summary printed when the server goes idle |
| `VRAM use: …` and `context … tokens per request, pool … tokens` | startup: where the VRAM went, the per-request context and pool size; the KV format line gives the reason for its choice |
| `GET /api/tags -> 404 (probe for another server type; …)` | a client looked for an Ollama / LM Studio API. Since 0.1.1 each such path is logged once at `I` level (0.1.0 printed a `W` line per request); harmless — point the client at `/v1` |

The response JSON also carries `usage.cached_tokens` and a llama.cpp-style `timings` object with the
same numbers ([server.md](guide/en/server.md#logging)).

## <a id="troubleshooting"></a>7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "Windows protected your PC", or the program is blocked | the executables are not code-signed: **More info → Run anyway**, or `Unblock-File` the official zip before extracting; Smart App Control may block outright — see [windows_security.md](windows_security.md) |
| first start with a new model sits for a while | one-time kernel autotune: a few seconds for MXFP4 files, about 1.5–2 minutes for a 27B Q4_K_M file (a message says so); cached in `%LOCALAPPDATA%\whirl` |
| start waits without loading | another WHIRL process holds the GPU (one process per GPU; the second waits up to 30 minutes, `WHIRL_GPU_WAIT`). Stop the other one |
| exit code **2** | bad command line (unknown option, missing model path, a `--host` that is not this computer's): `.\whirl-server.exe --help` |
| exit code **3** | GPU / driver problem: install AMD Software: Adrenalin Edition 26.8.1 or newer; check `.\whirl.exe devices` |
| exit code **4** | model file problem: file missing, not a GGUF, architecture not `qwen35` / `qwen35moe`, unsupported tensor type, truncated download |
| exit code **5** (out of GPU memory) | close other programs using the GPU (two large GPU processes make Windows move both to slow shared memory); cap the pool with `-c 131072`; shorten `--ctx-per-slot`; use fewer slots (`-np 2`); `$env:WHIRL_KV = "q8v"` (or `q8h`) |
| exit code **6** (port in use) | another program listens on that port; use `--port 8081`, or find it with `Get-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess` |
| client says "model not found" or lists nothing | use the `/v1` base URL, the OpenAI-compatible provider type, and the `--alias` as model id |
| client cuts long conversations early | it does not know the context length: enter `--ctx-per-slot` (default 131072) as its context window |
| every turn is slow (no `reused` in the log) | the prompt start changes between turns (timestamp in the system prompt, reordered tools), or `WHIRL_NO_PREFIX_CACHE` is set |

All exit codes and messages: [usage.md](usage.md#exit-codes).
