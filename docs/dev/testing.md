# Testing for developers

This page is for people changing WHIRL itself. It covers the golden-hash gate (G0), a fast
check that a change did not alter any output, and the table of which tests a change needs.
Build and unit-test instructions are in [building.md](../building.md#test).

## Golden-hash gate (G0)

`whirl golden` runs fixed prompts and prints one `G ...` line per step. Each line holds the
generated token, the top-5 token ids and an FNV-1a 64-bit hash of the full f32 logits row.
Token streams are printed as id lists with their hash. `tools/golden/golden.ps1` runs it for
each {model} x {mode} x {suite} and either records the lines as reference files or diffs a new
run against them:

```powershell
pwsh tools/golden/golden.ps1 check                      # R9700 (gfx1201), both models, all modes
pwsh tools/golden/golden.ps1 check -Arch gfx1151        # Radeon 8060S
pwsh tools/golden/golden.ps1 check -Models dense -Modes precise -Suites single   # a subset
pwsh tools/golden/golden.ps1 record [-Arch ...]         # rewrite the reference files
```

Exit codes: 0 means every line matches. 1 means at least one file differs. 2 means a run failed
(a crash, a hipError, a missing model or a missing reference file). For each differing file the
script prints the first differing line, which tells you the case and step, both versions of the
line and the field that changed. For a token list it gives the first differing position. Example:

```
DIFF  dense/fast_single        first at line 9 (short.step05); 12 of 70 lines differ
      golden: G short.step05 next=1234 top5=1234,88,... lh=0f3a...
      now:    G short.step05 next=1234 top5=1234,88,... lh=77c1...
      change: lh=0f3a... -> lh=77c1...
```

When only `lh` changes and `next`/`top5` stay the same, the logits changed at the bit level but
the chosen token did not. A bit-exact refactor must not change even that.

### What it covers

| Suite | Cases |
|---|---|
| `single` | `short`: a Chinese coding prompt, chunked prefill, then 24 decode steps on the pipelined decode path that chat uses (logits hash on every step). `long4k`: a 4000-token prompt (the prefill is split into `max_batch` chunks), then 8 decode steps. `mtp`: MTP speculative decode of 64 tokens with the model's default draft count held fixed (no timing-based adaptation). `mtp_ngram`: the same with n-gram drafts forced whenever they match. |
| `pair` | Two sequences on the server's multi-sequence paths: a solo reference for each, segmented prefill (`prefillSegs`), batched decode rows, a batched verify with literal drafts, and one batched MTP draft + verify cycle after an MTP segmented prefill (DeltaNet replay on, as in the server). |

Models: one dense and one MoE model. `WHIRL_GOLDEN_DENSE` defaults to Qwen3.8-27B UD-Q4_K_M and
`WHIRL_GOLDEN_MOE` defaults to Ornith-1.5-35B-A3B MXFP4, both at their LM Studio download paths.
Each reference file stores the model's file name and size, so a check against a different file
fails on the first line. Modes: `--precise`, `--balance`, `--fast`. The fast mode loses precision
(quantized KV, relaxed acceptance) but is still deterministic, so it can be hashed too.

Reference files are kept per architecture, at `tests/golden/<arch>/<model>/<mode>_<suite>.txt`.
gfx1201 and gfx1151 use different kernels and tuning tables, so their outputs differ.

The script needs `build\Release\whirl.exe`, or pass `-Exe`. It sets `AMD_LOG_LEVEL=1` and stops
at the first hipError 719. It does not take the workstation GPU locks, so wrap it in your own
lock if other jobs share the GPU. A full check is 12 runs, and each one loads its model. On the
R9700 a run takes 12-19 s with the model files in the OS file cache, and a full check takes
about 3 minutes. The Radeon 8060S takes about the same (8-30 s a run). A subset (one model, one
mode) takes about 30 s.

In this version `--fast` has no items of its own yet and runs as balance, so the fast files
match the balance files. They are kept separate so the gate already covers fast once its items
land.

### Rules

- A refactor or other change that must not alter output passes `golden.ps1 check` with no diff
  on every architecture it touches.
- A commit that changes output on purpose (a lossy kernel, a different default, a new numerics
  item) records the new references in the same commit, and the commit message says
  `golden update: <reason>`. Otherwise the gate stops meaning anything.
- Run the check three times after recording on a new machine or driver to confirm the run is
  deterministic there.

## Change → required tests

R9700 = gfx1201 and 8060S = gfx1151. G0 = the golden-hash gate above. Some tests in the table
are planned but not written yet: the resolve table tests, kv_policy table tests, prefix_cache
unit tests and block tests. Until they exist, run G0 for those changes.

| What changed | Required | GPU? | Both GPUs? |
|---|---|---|---|
| tokenizer / chat template / json | whirl-tests, whirl-parity | no | no |
| protocol / http / StreamState | whirl-server-tests | no | no |
| numerics / config resolve (pure logic) | whirl-tests (numerics), resolve table tests | no | no |
| **default ExecFlags change** (the actual flags of any mode) | the row above + G0 (affected modes) + check.ps1 | yes | **yes** (different Caps give different flags) |
| kv_policy | kv_policy table tests, whirl-server-tests (pool pressure) | no | no; if the default format changes add G0 + server-gate pool |
| prefix_cache / PagePool | whirl-server-tests (prefix, shared sys, pool, restore), prefix_cache unit tests | no | no; on a behaviour change (not just a move) add server-gate mt_cache, sys (R9700) |
| tier (RAM/SSD) | whirl-tier-tests, whirl-server-tests testTiers/testRestoreConcurrent | no | no; on an IO path change add server-gate tier (R9700) |
| sampling / spec policy (CPU side) | whirl-model-tests, whirl-tests specSample, whirl-server-tests sampler/sampling/slotDraft | no | no |
| spec acceptance behaviour (specsample/relaxacc) | the row above + G0 (fast, balance) + cmp_mtp + server-gate fast_stream (dense and MoE) | yes | yes |
| engine scheduling (prefill grouping, decode loop) | all whirl-server-tests + G0 (pair suite) + server-gate basic (+ fast_stream when MTP prefill rows move) | yes | R9700 only; 8060S at merge time |
| one forward block (host dispatch) | block tests (after R-7) + G0 | yes | shared caps paths: both; gfx1151-only paths: 8060S |
| one kernel (*.hip, gfx1201) | whirl-kernel-test --family X (R9700) + G0 | yes | no |
| one kernel (gfx1151/*.hip) | whirl-kernel-test --family X (8060S) + G0 (8060S) | yes | no |
| **kernels_abi.h structs/arguments (KvArgs, GvArgs, DSampArgs, ...)** | kernel test, all families, both archs + G0 on both archs | yes | **yes** |
| weights/loader (quant formats, repack) | whirl-model-tests, kernel test quant/moemx, G0 | yes | yes |
| tune (tuning tables/candidates) | G0 (tuning only picks bit-identical candidates, so the hashes must not change) + check.ps1 speed | yes | only the affected arch |
| platform layer | whirl-server-tests, tier-tests, server-gate basic+tier | yes (basic) | no |
| before a release / rc merge | run_gates.ps1, the full set on both GPUs | yes | **yes** |

**Cases that still need the full R9700 + 8060S set:** a kernels_abi change, a change to the
default ExecFlags or the default KV format, a change to what a precision or numerics mode means,
an rc merge and a release. A change that only affects the 8060S does not rerun the full R9700
set. The R9700 set is checked once, when the change is merged.
