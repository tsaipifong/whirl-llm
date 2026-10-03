# Phase 1 parity report

Date: 2026-10-02. Machine: Windows 11, MSVC 19.44, HIP SDK 7.2, Radeon 8060S
(gfx1151, dev0) + Radeon AI PRO R9700 (gfx1201, dev1). All builds and tests
ran at low / idle CPU priority while benchmark jobs had the GPUs.

References used (external, never copied into WHIRL):
llama.cpp b11214 `llama-tokenize.exe` (vocab-only, CPU), llama.cpp source
5cf3a35 (`unicode-data.cpp`, `common/jinja`), gguf-py 0.19.0.

## Summary

| Area | Test | Result |
|---|---|---|
| Unit tests | `whirl-tests.exe` (JSON, Python float repr, UTF-8 decoding, Unicode flags, qwen35 split, synthetic GGUF v2/v3 with every value type and alignment 32/64, malformed GGUF, chat rendering) | 164 / 164 pass |
| Unicode tables | `whirl-parity unicode` (Phase 1: a Python script): category flags for all 0x110000 code points, White_Space set, lowercase map vs llama.cpp `unicode-data.cpp` | 0 differences (at Unicode 15.1 level, see below) |
| GGUF reader | `whirl-parity gguf` vs reference dumps made with gguf-py outside the repo (Phase 1: a Python script): every metadata key, value type and value (arrays by element checksum), alignment, data offset, full tensor table (name, type, shape, offset, byte size) | 13 / 13 files identical |
| Tokenizer | `whirl-parity tokenizer` (Phase 1: a Python script) vs `llama-tokenize --ids --no-escape --no-bos` | 440 / 440 file-mode combinations identical (110 files x 2 models x parse_special on/off; 1,497,742 tokens compared) + 28 / 28 rendered chat prompts |
| Chat template | `whirl-parity template` (Phase 1: a Python script) vs llama.cpp's Jinja engine running each GGUF `tokenizer.chat_template` | 45 cases x 2 variants: 0 unexpected differences (details below) |
| HIP smoke | `whirl-tool devices --smoke` (embedded code object load, `getFunction`, `getFunctionOpt` on a missing name, `getGlobal`, H2D/D2H copies, stream + events, kernel launch, graph capture + replay) | PASS on dev0 (8060S, gfx1151) and dev1 (R9700, gfx1201) |

## GGUF reader

Files (all in a local model directory, `<models>` below): Qwen3.8-27B UD-Q4_K_M and
UD-IQ3_XXS, Qwen3.8 mmproj F16, Ornith-1.5-35B-A3B Q4_K_M, Ornith mmproj BF16,
Qwen3.8-27B MXFP4 (FreedomAISVR), Swift-1.5 MXFP4 A/B/C, Ornith CyberTiel
MTPv2 ICE, bge-m3 F16 (bert), DeepSeek-V4 drafter Q2_K, Qwen-Image 2.1 Q8_0
(no metadata keys). Tensor types covered: f32, f16, bf16, q8_0, q2_K, q3_K,
q4_K, q5_K, q6_K, iq3_xxs, iq3_s, iq4_nl, iq4_xs, mxfp4 (among others).
The Qwen3.8 file (15.3 GiB, 866 tensors, 248k-token vocab) is parsed from
the memory mapping in about 7 ms.

## Tokenizer

Corpus (110 files):

- 82 synthetic cases from `whirl-parity make-corpus` (tests/parity/make_corpus.cpp): CJK / Japanese /
  Korean, emoji with ZWJ and skin tones, combining marks (Latin, Devanagari,
  Thai), RTL, Cyrillic/Greek case pairs, Unicode spaces (U+00A0, U+3000,
  U+2028, U+0085, ...), zero-width characters and BOM, code points new in
  Unicode 16.0, unassigned and private-use code points, full-width and math
  symbols, digit runs, contractions (including `'S`, U+2019, U+017F, U+0130),
  punctuation runs with trailing newlines, every newline style, indentation
  edge cases (the prototype's former `"    return"` bug, tabs, mixed
  space/tab, blank lines), long runs (3000 letters, 2500 spaces, 1200 digits,
  repeated CJK and emoji), URLs, Windows paths, JSON, Markdown, code in
  Python / C++ / Rust / JS / Go / SQL / Zig / Bash and Python with Chinese
  comments, every special token (alone, spaced, in chat and tool-call
  layouts, partial and nested look-alikes, `[PAD...]`), and invalid UTF-8
  (stray continuation bytes, truncated 2/3/4-byte sequences, overlong forms,
  encoded surrogates, F8..FF, bad continuation, NUL and control bytes, lone
  CR, BOM, and an F5 lead byte).
- 7 mixed Chinese/English coding prompts (`bench_zhcode/prompts.json`).
- `prompt_4k/12k/24k.txt` (long English+code prompts).
- 8 third-party Markdown documents (read in place as test data only; replaced
  in the later re-run, see below).
- Large code files: the prototype's own `kernels.hip` (551 KB) and
  `gguf.zig`, llama.cpp `llama-vocab.cpp` and `common/chat.cpp`, and several
  WHIRL sources and documents.

Results (files identical / files):

| Model | parse_special = true | parse_special = false |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_M | 110 / 110 (374,212 tokens) | 110 / 110 (374,659 tokens) |
| Ornith-1.5-35B-A3B Q4_K_M | 110 / 110 (374,212 tokens) | 110 / 110 (374,659 tokens) |

The `invalid_f5_f7_lead` case decodes to a code point above U+10FFFF:
llama.cpp throws from `unicode_cpt_to_utf8` and `llama-tokenize` aborts with
an unhandled C++ exception; WHIRL raises `TokenizerError`. Both are counted
as an (identical) error outcome.

Additionally, 28 prompts rendered by WHIRL's chat templates (the 7 coding
prompts, with system message and with tools + tool-call history, variants a
and b) tokenize identically with `parse_special = true`.

Speed: loading the vocabulary and 247,587 merges from the mapped GGUF takes
0.10-0.15 s; encoding the 85 KB `prompt_24k.txt` (24,552 tokens) takes about
13-15 ms (~1.7-1.9 M tokens/s, single thread). The whole 110-file corpus
(374k tokens) encodes in about 0.15 s.

### Unicode 15.1 vs 16.0

llama.cpp's `unicode-data.cpp` was generated from Unicode 15.1, while
Python 3.14 (the generator's UCD source) ships Unicode 16.0. Comparing a
plain 16.0 table gave exactly 5,185 differing code points, all of them
characters first assigned in Unicode 16.0 (llama.cpp: UNDEFINED). The
generator therefore emits Unicode 15.1-level tables by default (code points
with Age=16.0 are left unassigned); `--unicode 16.0` produces the full table.
The synthetic case `new_in_unicode16` exercises these characters.

## Chat template

The oracle (`tests/template_oracle`, test-only) compiles llama.cpp's Jinja
engine from the external checkout and renders the GGUF templates with
`add_generation_prompt = true` and the request's `tools`,
`enable_thinking`, `reasoning_effort`, `preserve_thinking`.

| Variant (template) | identical prompt | both raise | documented divergence | mismatch |
|---|---|---|---|---|
| a (Qwen3.8-27B) | 37 | 4 | 4 | 0 |
| b (Ornith-1.5-35B) | 39 | 1 | 5 | 0 |

Covered: single/multiple leading system and developer messages, empty and
null system content, content part lists, reasoning_content, `<think>` blocks
inside assistant content (variant b), preserve_thinking true/false with the
last-query rule, multi-step tool turns, enable_thinking off (top level and
kwargs), every reasoning effort including invalid ones, tools with nested
schemas / quotes / backslashes / control characters / emoji, tool_choice
none, tool calls with object arguments of every JSON type, flat tool calls,
empty / null / whitespace arguments, consecutive tool responses, tool
message first or last, assistant prefill, Unicode content with U+3000.

Documented divergences (deliberate, inherited from the prototype):

| Case | Variant | WHIRL | Template / oracle |
|---|---|---|---|
| `tool_call_args_json_string` | a | parses the JSON string into parameters | raises |
| `tool_call_args_json_string` | b | parses the JSON string into parameters | renders the call without parameters |
| `mid_system_message` | a | renders a system turn | raises |
| `unknown_role` | b | rejects the request | silently skips the message |
| `tool_call_missing_name` | b | rejects the request | renders `<function=>` (Python Jinja would raise) |
| `image_part` | a, b | rejects (images unsupported; `allow_media` renders the placeholder) | vision placeholder |
| `float_values` | a, b | Python `json.dumps` floats (`2.0`, `0.7`) | llama.cpp `tojson` prints `2` |

### Known differences from the Hugging Face (Python Jinja2) rendering

Not testable here (jinja2 is not installed), noted for completeness:
Python's `str.strip()` used by `|trim` also removes Unicode whitespace
(U+3000, U+00A0, U+2028, U+0085, U+001C..U+001F). WHIRL, the prototype and
llama.cpp trim only ASCII whitespace, so content that starts or ends with,
for example, an ideographic space keeps it. This only affects such edge
whitespace.

## HIP smoke test

Run once while holding the machine's GPU lock (so no other GPU job ran at the same time; lock held 1 s):

```
dev0: AMD Radeon(TM) 8060S Graphics  arch=gfx1151  code-object=gfx1151  mem=99.7 GiB  CUs=20  wave=32  clock=2900 MHz  pci=c4:00  integrated
dev1: AMD Radeon AI PRO R9700  arch=gfx1201  code-object=gfx1201  mem=31.9 GiB  CUs=32  wave=32  clock=2350 MHz  pci=47:00
embedded code objects: gfx1201 (5208 bytes) gfx1151 (5208 bytes)
smoke dev0: PASS (656 ms, 2 launches total)
smoke dev1: PASS (340 ms, 4 launches total)
exit=0
```

Both embedded code objects load and run on their device (dev0 gets the
gfx1151 build, dev1 the gfx1201 build); the axpy kernel result, the device
global read back through `hipModuleGetGlobal` and the graph-captured replay
are all correct.

## Phase 2 re-run with the C++ drivers (2026-10-02)

The repository is pure C++ since Phase 2: the Python generator and parity
scripts were rewritten in C++ (`tools/gen-unicode-tables`, `tests/parity`,
executables `whirl-gen-unicode` and `whirl-parity`). The only check that
still needs Python is producing the gguf-py reference dumps; that script
lives outside the repository (a short gguf-py script in `<work-dir>`), and
`whirl-parity gguf` compares against its output files. Results after the
conversion (build in a separate build directory, all at idle priority):

| Check | Command | Result |
|---|---|---|
| Unicode generator | `whirl-gen-unicode --ucd DIR` on UnicodeData.txt / PropList.txt exported from the same UCD 16.0.0 (by a small export script in `<work-dir>`, outside the repository) | tables byte-identical to the Phase 1 file (only the two header comment lines differ); `--unicode 16.0` again gives exactly 5,185 differing code points vs llama.cpp |
| Unicode tables | `whirl-parity unicode --inc src/tokenizer/unicode_tables.inc --llama <llama.cpp>` | 2273 vs 2273 ranges, 0 code points differ, whitespace identical, lowercase 1433 vs 1433, PASS |
| Synthetic corpus | `whirl-parity make-corpus --out DIR` | 82 files, byte-identical to `tests/corpus/synthetic` |
| GGUF reader | `whirl-parity gguf --whirl-tool T --ref-dir DIR <13 GGUF>` | 13 / 13 identical |
| Tokenizer | `whirl-parity tokenizer ...` (same reference cache, SHA-256 keys) | 440 / 440 identical (507,841 tokens per model with parse_special, 508,340 without) |
| Chat template | `whirl-parity template ...` | a: 37 identical / 4 both rejected / 4 documented / 0 mismatches; b: 39 / 1 / 5 / 0 |

The tokenizer corpus differs from Phase 1 in 8 files: the eight third-party
Markdown documents were dropped from the test inputs under the stricter Phase 2
provenance rules and were replaced by eight of this project's own Markdown
files (research notes and documentation), so the token totals differ; the
other 102 files are the same.

Commands (paths are machine-local):

```
whirl-parity unicode   --inc src\tokenizer\unicode_tables.inc --llama <llama.cpp source checkout>
whirl-parity make-corpus --out <DIR>          (default: tests\corpus\synthetic)
whirl-parity gguf      --whirl-tool <whirl-tool.exe> --ref-dir <DIR of *.gguf.dump> <GGUF...>
whirl-parity tokenizer --whirl-tool <whirl-tool.exe> --llama-tokenize <llama-tokenize.exe>
                       --model <GGUF> [--model <GGUF>] --work <DIR> [--jobs 4] [--report F] <CORPUS...>
whirl-parity template  --whirl-tool <whirl-tool.exe> --oracle <whirl-template-oracle.exe> --work <DIR>
                       --model-a <Qwen3.8 GGUF> --model-b <Ornith GGUF> [--show]
whirl-gen-unicode --ucd <DIR with UnicodeData.txt [PropList.txt]> [--unicode 15.1|16.0] --out src\tokenizer\unicode_tables.inc
```
