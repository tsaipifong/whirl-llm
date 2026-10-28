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

Models: the required set is `swift` (Swift-1.5 27B MXFP4-A, the headline dense model, `WHIRL_GOLDEN_SWIFT`)
and `moe` (Ornith-1.5-35B-A3B MXFP4, `WHIRL_GOLDEN_MOE`), the default `-Models swift,moe`. `dense`
(Qwen3.8-27B UD-Q4_K_M, `WHIRL_GOLDEN_DENSE`) is optional (`-Models dense`). Each defaults to its LM Studio
download path.
Each reference file stores the model's file name and size, so a check against a different file
fails on the first line. Modes: `--precise`, `--balance`, `--fast`. The fast mode loses precision
(quantized KV, relaxed acceptance) but is still deterministic, so it can be hashed too.

The gate cannot pass vacuously (FIX-REVIEW2). An empty model, mode or suite list, a run that printed no
`G` lines, a reference file with no step lines, and a model file that differs from the recorded one (name,
size and a hash of its size plus first and last 4 MiB, kept in `tests/golden/models.txt` and rewritten by
`record`) all exit 2. A step line is any `G` line other than `G model`, `G header` and `G info`; a run or a
reference without one is an error. `record` stages the references and writes them together with
`tests/golden/models.txt` only when every case ran (a partial record writes nothing, so references and model
hashes never disagree). `pwsh tools/golden/golden_selftest.ps1` checks these verdicts on the CPU with a fake
`whirl.exe` (`tools/golden/fake_whirl.cmd`). The script clears every `WHIRL_*` environment variable before the runs (a stray
`WHIRL_HIP_DEVICE` or `WHIRL_MODE` would change what is measured; `--device` picks the GPU) and restores
them afterwards. Suites: `single`, `pair`, and `ngram` (the forced n-gram MTP drafts on a copy-the-text
prompt; the runner fails the case when `ng_cycles` is 0, because the n-gram case of `single` never drafts on
its prompt). `ngram` needs a model with an MTP head and is in the default list for both architectures
(gfx1151 references recorded on the Radeon 8060S with `golden.ps1 record -Arch gfx1151 -Suites ngram`).

Reference files are kept per architecture, at `tests/golden/<arch>/<model>/<mode>_<suite>.txt`.
gfx1201 and gfx1151 use different kernels and tuning tables, so their outputs differ.

The script needs `build\Release\whirl.exe`, or pass `-Exe`. It sets `AMD_LOG_LEVEL=1` and stops
at the first hipError 719. It does not take the workstation GPU locks, so wrap it in your own
lock if other jobs share the GPU. A full check is 12 runs, and each one loads its model. On the
R9700 a run takes 12-19 s with the model files in the OS file cache, and a full check takes
about 3 minutes. The Radeon 8060S takes about the same (8-30 s a run). A subset (one model, one
mode) takes about 30 s.

Suite `server` (ARCH-1, opt-in until its references are recorded on both GPUs): runs
`whirl-server-gate --suite golden` against `build\Release\whirl-server.exe` (`-ServerExe`, `-GateExe`) and
records the `G srv.*` lines: prefill on the engine schedule, streaming, thinking, a ~5k-token document, the
in-VRAM prefix cache, a RAM-tier restore and an SSD-tier restore after a restart; in precise also a server
with no prefix cache (the restored turn must equal it bitwise) and two concurrent requests (each must equal
the request alone). The gate checks these relations itself and the runner rejects a run without
`(ALL PASS)`. Vision joins with `WHIRL_GOLDEN_MMPROJ_<MODEL>` and `WHIRL_GOLDEN_IMAGES`.

Every check and record prints the SHA-256 of the executables and of the reference set (selected
reference files plus `models.txt`); `golden.ps1 refhash` prints the reference-set hash without a GPU.
`golden.ps1 check -Impact A..B` (or `-Impact worktree`) runs only the models and suites that
`tools/impact/impact.py` lists for that diff (see [Impact map](#impact-map)).

`--fast` has its own items (q4 KV, relaxacc on dense models with an MTP head), so the fast
files differ from the balance files wherever those items apply. `golden.ps1 record` reports the
changed balance / fast references through the KL regression guard (`-KlReport`, `-TaskAccuracy`, see
[KL gate](#kl-gate) below); the guard warns, it does not refuse the record.

### Rules

- A refactor or other change that must not alter output passes `golden.ps1 check` with no diff
  on every architecture it touches.
- A commit that changes output on purpose (a lossy kernel, a different default, a new numerics
  item) records the new references in the same commit, and the commit message says
  `golden update: <reason>`. Otherwise the gate stops meaning anything.
- Run the check three times after recording on a new machine or driver to confirm the run is
  deterministic there.
- **KL guard on every balance / fast re-record (arch025-klguard, owner mode standard 2026-10-26;
  replaces the FIX-TEST-B rule "no re-record without a passing KL gate").** precise references are
  the strict bitwise golden: `golden.ps1 check` fails (exit 1) on any precise difference, and a
  record that changes them prints that the `golden update:` note must name why. When a record
  changes balance or fast references, run `tools/klgate/klgate.py` with the same `whirl.exe` on the
  same GPU family for every recorded model and pass its report as `-KlReport`, plus the
  task-accuracy result (`-TaskAccuracy FILE` or a KB exp_id), which is the acceptance of balance and
  fast. `tools/klgate/kl_guard.ps1` judges the report as a guard: a case beyond its guard line prints
  `WARN kl guard: <model> <prompt> <mode> <length>: KL ... vs guard line ...; top1 ... | mode <mode>,
  items ...` (the run's numerics items, marked allowed balance item / fast item) and the next steps
  (find the item, check task accuracy, re-baseline the guard line); no report prints
  `TODO kl-guard: ...`, no task-accuracy reference `TODO task-accuracy: ...`. The record is written
  and exits 0 in all of these cases; the evidence goes to `-KlEvidence` (default
  `<Out>\kl_guard_<arch>.json`) and `golden update note:` lines are printed for the commit message.
  A record whose balance / fast references are byte-identical (a bitwise refactor) needs no report.
  A `-KlReport` that is not usable evidence (another exe SHA-256 or GPU family, run errors, 0 cases,
  fewer than 3 positions, a missing length 8k / 32k / 96k / 128k, model or mode, a negative `--as`
  test) is a setup error: exit 2 before any run, nothing written. The guard takes each case's
  pass / limit from the report as klgate judged it; after a limit change use `klgate.py --rejudge`.

## <a id="impact-map"></a>Impact map

`tools/impact_map.json` maps source functions and kernel files to product paths, tests, gates and
release-matrix cells. It is built from the code (`python tools/impact/impact_build.py`, a few seconds):
CMake link sets, the include graph, the call graph, and the kernel dispatch table (`kernels_abi.cpp` names
-> `.hip` files, arch from the `#if __gfx1151__` guards). Path roots, server-gate suites and matrix
metrics are declared in `impact_build.py` and marked `declared` in the JSON.

```
python tools/impact/impact.py --worktree           # what to re-run for uncommitted changes
python tools/impact/impact.py --commit SHA --why   # with the reason for every path
python tools/impact/impact_build.py --check        # is the committed map current?
python tools/impact/validate.py                    # replay real fix commits: does the map flag their tests?
```

`validate.py` is the map's acceptance test (arch024-S12): every named case (a real bug, the fix commit
and the gate that caught it) must be hit by the map built at that fix commit, and the history replay
(`--history 200`: every commit that changed product code and a registered test together) must list that
test for >= 95 % of the pairs. It prints the selectivity next to the hit rate (a map that lists
everything would hit everything) and every miss with its commit. The history rate counts **CPU tests
only**; it is not evidence for the GPU steps. For those, `--baseline REV` replays REV's `tools/impact`
on the same commits (`tools/impact/replay.py`) and fails on every commit for which the current tools
select fewer release steps or paths. `--skip N` reports on older commits the tools were not tuned on
(hold-out). A touched test that no executable of the map registers counts as missed (UNPAIRED), so a test
executable missing from the map cannot shrink the denominator. "Not narrower than the old tools" is a
regression check only, not GPU recall: what the old map missed, the new one may miss too. Exit 0 = every
named case hit, every rate >= 95 %, nothing narrowed; 1 otherwise. A changed host function that dispatches
kernels (a host user of a `.hip` file) lists those kernels' tests (kernel-test functions, `kt_full`).

### <a id="module-gate"></a>Module gate (dev-time only)

`tools/gates/modules.json` maps each engine module (arch024 M1-M8, plus the cross-module contracts) to
its source globs and the `release_gate.ps1` steps of its row in the arch024 plan, table c-1.

```
pwsh tools/gates/release_gate.ps1 -Device r9700 -Module tier_bridge -ImpactRange main..HEAD -List   # table + the map's steps
pwsh tools/gates/release_gate.ps1 -Device r9700 -Module ckpt_store -ImpactRange worktree           # uncommitted changes
pwsh tools/gates/release_gate.ps1 -Device r9700 -Module ckpt_store -NoMapSteps -List   # the table only (said in the summary)
python tools/gates/modules_check.py [--rev arch024-s6b]   # modules.json against a tree and the gate's steps
python tools/impact/impact.py --worktree                  # "module" / "mod-gate" lines: which module gate fits
pwsh tools/gates/module_gate_test.ps1                     # -Module step-selection table test (CPU)
python tools/gates/modules_check_test.py                  # modules_check / impact module mapping block + pass tests
```

**This is fast feedback while developing, not a release rule.** Until the owner signs off O1 and O2
(arch024 plan), a release, an rc merge and a bug-fix release run the full gate on both GPUs under the
rules above. A `-Module` run writes `PASS_module_<names>_<device>`, never the `PASS_gate_<device>` of a
full gate (a `-Tag` that does not start with `module_` is refused), its summary header says
`module=<names>`, and `release/release_check.py` (R1) rejects such a summary for a bug-fix release.
The gate prints the module's checks that are not release_gate steps (P-719, P-size, P-sess, flake
loops) and which steps need both GPUs. M1 handles runs with the modules of every file that includes
it (an includer in no module, e.g. `engine.h`, means the full gate); M8 threads runs with M2-M5 and M7.
`-Module` needs `-ImpactRange A..B` (or `worktree`) or `-NoMapSteps`. With `-ImpactRange` the gate runs
impact.py on that diff and adds the steps the map selects beyond the module's row (PLAN c-2, e.g.
`server.mtp` -> `srv_balance`, `srv_fast`) and the module's `when` steps; impact.py's "full gate" runs the full
gate, and a change that impact.py maps to a module `-Module` does not list is refused. Steps added beyond the
table are printed as "run on both GPUs, one after the other". Python for impact.py and `cpu_modules_check`:
`WHIRL_GATE_PYTHON`, else `C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe` when it exists, else `python` /
`py` on PATH.

- A module whose files are not on the current branch yet (`lands_on`) is reported, and its steps still run;
  a module with paths of which none exists and no `lands_on` is a setup error (exit 2), as are an
  unknown module, a step name that the gate does not have and that is not listed as `pending`, a
  `pending` step that already exists in the gate, and an empty selection.
- `impact.py` suggests `-Module X` only when every changed product file belongs to one module (plus
  that module's `when` files, e.g. `prefill_schedule.h` for M3 / M4); product code in no module, two
  modules, a contract file (ServerModel, `device::Backend`, kernels_abi, numerics) or a build-wide file
  (CMakeLists.txt, cmake/*, build.bat, redist/*, tools/package_release.ps1) means the full gate.
  `release_gate.ps1 -Module a,b` accepts two modules with a warning (dev-time union only).
  During the arch024 moves every step also edits `engine.cpp` / `engine.h`, so the suggestion is the
  full gate there; the module gate is for later changes that stay inside one module.

The map lists all three modes for every change: a change is shown to be mode-confined only by the
golden hashes (precise unchanged). Host code that branches on the GPU inside a shared function is listed
for both GPUs.

### <a id="isolation"></a>Isolation check (ARCH-1 8.5: src/sampling, then every isolatable module)

One manifest per isolated module: `tests/sampling/isolation.txt` (sampling, arch025-X3) and
`tests/iso/<module>/isolation.txt` (arch025-isoall). A manifest names the module (`module`), its test program
(`exe`), the source the boundary mutants edit (`mutate`), the link mutants (`mutsym hip|log|now|engine`), extra
Windows libraries (`syslib`), and the files: `src` / `hdr` = the module and its DECLARED dependencies (for example
spec.policy declares `src/common/util.cpp` and the platform layer it calls), `testsrc` / `testhdr` = the test
program only. `cmake/iso.cmake` copies those files into `build/Release/iso/<module>/{mod,test}/` and builds them from
there with that copy as the only include path; the test program links the module OBJECTS directly (no static-library
member selection) with `/OPT:NOREF` and nothing else (no `whirl`, no server, no tokenizer GGUF, no GPU). So an
undeclared include (also `../server/engine.h`) fails to compile and a call into anything not listed fails to link, even
in a function no test calls. The product build is unchanged.

- `cpu_sampling`: runs `whirl-sampling-tests`.
- `cpu_sampling_iso`, `cpu_iso_<module>` (`.` -> `_`; one per `tests/iso/*/isolation.txt`):
  `tools/gates/iso_check.py --mutants --module <m>` builds the same thing by hand, runs the test program (control) and
  proves the boundary: an engine include and a relative engine include (C1083), and one call per `mutsym` into the
  device runtime / the server / lib whirl (LNK2019), each of which must also compile with the product flags.
  `iso_check.py --list` lists the isolated modules; without `--module` it checks all of them.

To declare a new dependency of a module, add its header (and, if it is not header-only, its source) to the
module's list. The change is then visible in review. Which modules are not isolated yet, and the include / call that
blocks each one with the smallest fix: [isolation_blockers.md](isolation_blockers.md).

## <a id="vis-order"></a>Vision stream order (`cpu_vis_order`)

`tools/gates/vis_stream_order_check.py` is a static happens-before check of the stream schedule in
`Vision::encode` (the `!res` path that streams the encoder weights through two slots on a copy stream). It reads
the source and runs nothing on a GPU. It models per-stream program order, event record / wait edges and slot
reads / copies, the same model as the PyTorch CUDA Stream Sanitizer, but over the source in host order.

- R1: every read of a slot waits for the copy event that covers the slot's latest copy (RAW).
- R2: every copy into a slot waits for a free event recorded after the slot's last read (WAR).
- R3: the merger reads after the loop are ordered.
- R4: a schedule call (a copy on `cs`, a record or wait of `ev_copy_` / `ev_free_` / `ev_start_`) may only be
  guarded by `!res` and the known loop conditions (`last`, `!last`, `l + 2 < hp.n_layer`,
  `l + 2 == hp.n_layer`, `hp.n_layer > 1`). An inverted guard such as `if (res)` is a violation. The model also
  leaves that call out, so the read it should order fails R1 too. An unknown condition is a violation.
- R5: inside the layer loop a copy goes into `slot[s]` or is the merger copy into `W + lb`.
- R0: the parse finds exactly 17 reads and 5 copies (`EXPECTED_READS` / `EXPECTED_COPIES`). A launch the parser
  stops seeing, such as a new wrapper or a moved stream argument, fails the check and is not skipped. Update the two
  numbers only after checking a real change by hand.

`--selftest` plants the 12011b4 revert and nine other changes. For each one it requires the exact set of
(rule, line) results. The line comes from an anchor text, so it follows edits of `vision.cpp`.

Limits:
- The loop body is checked as one symbolic iteration (s, 1 - s). The loop conditions are treated as true, so a
  swapped `last` / `!last` is not detected.
- The merger copy into `W + lb` is not WAR-checked. Whether slot 1 is free then depends on the parity of `n_layer`.
- The loop boundary is covered by construction. A copy into `slot[s]` must wait for `ev_free_[s]` recorded in the
  loop body. Prologue events are keyed 0 / 1 / start and never match s. On `stream` that record comes after every
  read of the prologue and of earlier iterations. A loop copy into a concrete slot would break this argument, so R5
  forbids it.
- Timing is not modelled. Only the GPU repro (vis-encode `WHIRL_TEST_VIS_COPY_DELAY_MB`, vis_ab) shows that the
  device really reorders.

## <a id="ctx-refuse"></a>Precise KV refusal (`st_ctx_refuse`) and environment restores (`cpu_env_restore`)

`st_ctx_refuse` (GPU, both cards) runs `tools/gates/ctx_refuse_check.ps1`. Precise mode keeps the KV cache in f16.
When the f16 KV for the default 4096 tokens does not fit, `whirl selftest --precise` must refuse with exit 5 and
"room for about N tokens. Use a shorter --ctx". It must not shrink a context the user did not ask for. The script
squeezes the card with `WHIRL_VRAM_LIMIT_MB`, set only for the selftest child; it needs no memory hog. It calibrates
the limit for each run. The first run uses 20480 MiB, and the room in the refusal text sets the next limit, aiming at
2048 tokens. A run that loads past the KV check is stopped and lowers the limit. A run that fails before the check
raises it. The step passes on a refusal with a room of 512 to 3584 tokens. A build that shrinks the context only
refuses with room 0, because `numerics::cliCtxFit` rounds the room down to whole KV pages. Mutant `c2-ctx-default`
fails the step.

`cpu_env_restore` runs `tools/gates/env_restore_check.ps1` on every `.ps1` file under `tools\` and `tests\`.
pwsh 7.5 and later (.NET 9 and later) keep a variable defined but empty after any of
`[Environment]::SetEnvironmentVariable(k, $null)`, `(k, "")` or `$env:k = ""`. A restore of an unset
`HIP_VISIBLE_DEVICES` written that way hid every GPU from the later child processes (arch024-S14, arch025-c2ctx).
The script fails on three things: a `SetEnvironmentVariable` call outside a `SetEnv*` helper, an empty-string
assignment to `$env:`, and a helper that does not remove the variable when it is given `$null`.
Both incidents were in runner scripts outside the tree. When the tree sits in the WHIRL workspace (a folder next to
`progress\` and `scratch\`), the script also checks the live runner tools in `..\progress\tools\*.ps1` with the same
rules, and a violation there fails the step. It lists the one-off runners in `..\scratch\**\*.ps1` that break a rule
as `WARN scratch` lines and a count line. Those never fail the step, since scratch is never shipped and old runners
stay as records. Fix a scratch runner (use a `SetEnvOrRemove` helper) before you copy it as a template for a new
runner. Folders that do not exist are skipped, and `-NoWorkspace` checks the tree only.

## <a id="kl-gate"></a>KL gate

`tools/klgate/klgate.py` measures what the golden hashes cannot: how far balance and fast are from
precise. For every model x prompt x mode one `whirl chat` run prefills a fixed prompt (raw, no MTP)
and dumps the teacher-forced logits of 3 windows of 512 positions (`--rows`; 128 before MOE-BAL)
near each of 8,192 / 32,768 / 98,304 / 131,072 tokens (`WHIRL_DUMP_LOGITS` + `WHIRL_DUMP_AT`, 16-row
forwards; the zh prompt has 129,297 tokens, so its last windows end there). The 3 windows per length
L end at L - 1216, L - 592 and L (`--positions`, default 3; KL-ROBUST): each follows a different
forward split and DeltaNet chunk phase (0 / 48 / 16 rows into a 64-row chunk). Per window it takes
the mean KL(precise || mode) and the top-1 agreement with precise.

**Judging (KL-ROBUST-2, owner decision 2026-10-14): per length the 3 positions are averaged first,
and the per-window limits apply to that per-length average.** A single window's KL moves by
+-20-35% and its top-1 by about 1.6 points just from how the prompt before it was split into
forwards (the same 512 tokens after an 80-row instead of a 4096-row forward: audit/KL-ROBUST.md), and
the noise floor jitters the same way, while the mean over positions is stable. The report prints
every position (not judged alone), each length's average, and the number each limit was applied to.
**The limits are a regression guard per mode (MODES-MXFP4, owner policy 2026-10-16), not a
near-lossless proof and not a veto on an allowed item.** Balance is defined by the items it may use
(fp8, moefp8, q8dec, h16, gdnwmma, int8 KV); near-lossless is shown separately with task accuracy.
The limits are set from the measured baseline of each mode on the two product models
(Swift-1.5 27B MXFP4-A, Ornith-1.5-35B-A3B MXFP4) with a margin. They depend on the model type (the
`arch` line of the precise run's log):

Dense models (`qwen35`), absolute, on each length's 3-position average:

| Mode | KL mean (3 x 512-row windows) | top-1 agreement |
|---|---|---|
| balance | <= 0.005 | >= 97% |
| fast | <= 0.03 | >= 93% |

MoE models (`qwen35moe`), relative to the noise floor **N** measured in the same gate run: one more
run per prompt, precise with the sequential DeltaNet prefill (`WHIRL_GDN_SEQ=1`, an equally exact
f32 path), compared with precise on the same windows. N = the mean of its window KLs (all lengths x
positions); N_top1 = its top-1 agreement averaged over all its windows (KL-ROBUST-2: not the same
window, since both sides carry the split jitter). Two equally exact paths through a 256-expert top-8 router already
differ by KL 0.002-0.006 on these prompts (near-tied 8th experts flip), more than the dense balance
limit, so an absolute limit cannot tell a real regression from the model's own sensitivity:

| Mode | mean KL over all windows (4 lengths x 3) | KL of each length's average | top-1 of each length's average |
|---|---|---|---|
| balance | <= 4.5 N | <= 7 N | >= N_top1 - 4.5 points |
| fast | <= 8 N | <= 12 N | >= N_top1 - 8 points |

MODES-MXFP4 baselines (R9700, both GPUs in audit/MODES-MXFP4.md): MoE balance with fp8 + moefp8,
mean 3.48 N (code) / 2.92 N (zh), per-length average up to 4.96 N, top-1 drop up to 2.91 points;
Swift balance (fp8) KL 0.0007-0.0015, top-1 down to 98.18% (zh 128k), hence the dense balance
top-1 limit 97% (98% before, set for Q4_K_M without fp8). The previous MoE balance limits (2 N / 3 N /
3.0 points; MOE-BAL, MOE-BAL-3) belonged to the table without fp8 / moefp8.
The fast multipliers were set the same way from the measured fast numbers. 512 rows per window: with 128 rows one row is 0.78 points of top-1 (dense fast zh 8k once
missed 93% by a single row); the limits were not relaxed for it. A 256-row run still failed (dense zh
8k balance 97.27% / fast 92.97%, MoE zh balance 2.04 N): the window start also moves where the
prefill is split into forwards, so the bits of a window - and its KL / top-1 - change with the window
size as well as by sampling (audit/MOE-BAL.md). That is why the gate now measures 3 positions per
length and judges their average (KL-ROBUST, KL-ROBUST-2); judged window by window, the 3-position run
failed MoE balance on single windows that the average passes. The tool keeps one precise dump in
memory (3 GB at 3 x 512 rows) and compares every other run right after it finishes.
`tools/klgate/test_klgate_judge.py` unit-tests the judging rule (no GPU). `--noise-dense` also runs
the noise floor on dense models (reported, not judged). `--rejudge REPORT --report OUT` judges an
earlier report's per-window numbers with the current limits without running anything (exe hash,
device and corpus are carried over and `rejudged_from` is recorded), e.g. after a limit change. `--as MODE=ARG` runs MODE
with another CLI argument and judges it by MODE's limits (a negative test: its report never passes).
`golden.ps1 record` and the release gate step `kl_guard` refuse a report with fewer than 3 positions as
evidence (exit 2: a measurement problem, not a KL miss).

Exit 0 only when every case ran and passed; 1 when a limit failed; 2 on a failed run, a missing or
changed corpus file (`tools/klgate/corpus.sha256`), a too-short prompt or 0 cases. Exit 1 is the
measurement tool's "beyond the guard line", not a verdict on the build: golden record and the release
gate read the report through `tools/klgate/kl_guard.ps1` (WARN + evidence + the task-accuracy TODO,
exit 0; self-test `tools/gates/kl_guard_selftest.ps1`). The corpus is two
fixed texts: `code.txt` (C++ / HIP source with Chinese design notes, 131,478 tokens) and `zh.txt`
(Chinese prose with some code); the workstation copy is `C:\ClaudeWorkspace\whirl-work\klgate\corpus`.

```
python tools/klgate/klgate.py --exe build\Release\whirl.exe --corpus <corpus dir> ^
    --model dense=<Qwen3.8-27B-UD-Q4_K_M.gguf> --model moe=<Ornith-1.5-35B-A3B-MXFP4.gguf> ^
    [--device 8060s] --report kl_<arch>.json
```

R9700: 14 runs (2 models x 2 prompts x precise / balance / fast, + the MoE noise run per prompt), about 30 minutes; take the R9700
lock around the whole run. 8060S: pass `--wrap "python <gpu_lock.py> --max-min 30 --"` so each run
takes the shared lock on its own. Run it (as the regression guard): before a golden re-record that
changes balance / fast references (above), after a change to a numerics item or a default KV format,
and before an rc merge / release on both GPUs (pass the report to `release_gate.ps1 -KlReport`, with
`-TaskAccuracy`, so step `kl_guard` puts its WARN / TODO lines into the gate summary).

## <a id="chunks"></a>Chunk schedule gate (`whirl seqtest`)

`whirl seqtest` (part of its `ok`) runs a golden long prompt of 9,500 tokens through the server's
cold prefill schedule (`include/whirl/prefill_schedule.h`, the same functions the engine uses) and
requires the same last-row logits bits from every execution the server can choose: forwards of 1 or
more whole 1024-row chunks (decode floor, `WHIRL_PREFILL_CHUNK`, `WHIRL_PREFILL_BATCH`), with or
without the TAIL-CK tail checkpoint, and resumed from a checkpoint at a schedule point (the tail, a
prefix checkpoint). Splits off the schedule and the CLI's own chunking are printed as `info:` lines
and do not fail the gate (they are not guaranteed; see kv-and-caching.md section 7). Run it in each
mode (`--precise`, default balance, `--fast`) on dense and MoE; it needs the default `--slot-ctx`
(12288) or larger. CPU side: whirl-server-tests `prefill_sched` checks the schedule functions.

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
| spec acceptance behaviour (specsample/relaxacc) | the row above + G0 (fast, balance) + seqtest (balance, fast) + server-gate fast_stream (dense and MoE) | yes | yes |
| engine scheduling (prefill grouping, decode loop) | all whirl-server-tests + G0 (pair suite) + server-gate basic (+ fast_stream when MTP prefill rows move) | yes | R9700 only; 8060S at merge time |
| one forward block (host dispatch) | block tests (after R-7) + G0 | yes | shared caps paths: both; gfx1151-only paths: 8060S |
| one kernel (*.hip, gfx1201) | whirl-kernel-test --family X (R9700) + G0 | yes | no |
| one kernel (gfx1151/*.hip) | whirl-kernel-test --family X (8060S) + G0 (8060S) | yes | no |
| **kernels_abi.h structs/arguments (KvArgs, GvArgs, DSampArgs, ...)** | kernel test, all families, both archs + G0 on both archs | yes | **yes** |
| weights/loader (quant formats, repack) | whirl-model-tests, kernel test quant/moemx, G0 | yes | yes |
| tune (tuning tables/candidates) | G0 (tuning only picks bit-identical candidates, so the hashes must not change) + check.ps1 speed | yes | only the affected arch |
| platform layer | whirl-server-tests, tier-tests, server-gate basic+tier | yes (basic) | no |
| numerics item / default KV format / anything that changes balance or fast output | the rows above + KL gate (tools/klgate) | yes | **yes** |
| prefill schedule (`prefill_schedule.h`, engine chunking, checkpoint positions) | whirl-server-tests (prefill_sched, tail_ck_*) + seqtest in precise / balance / fast, dense and MoE | yes | R9700; 8060S at merge time |
| before a release / rc merge | `tools/gates/release_gate.ps1 -Device r9700` and `-Device 8060s` (every step must pass; exit 0 and a PASS_ marker; includes seqtest precise / balance / fast dense + MoE precise / balance) + KL gate (`tools/klgate/klgate.py`, both GPUs) | yes | **yes** |

**Cases that still need the full R9700 + 8060S set:** a kernels_abi change, a change to the
default ExecFlags or the default KV format, a change to what a precision or numerics mode means,
an rc merge and a release. A change that only affects the 8060S does not rerun the full R9700
set. The R9700 set is checked once, when the change is merged.

**The release gate** (`tools/gates/release_gate.ps1`, FIX-TEST-A) is the versioned list of those
checks: the CPU test binaries and self-tests, golden xcheck, `whirl-kernel-test --require-all`,
selftest, golden check (with its in-run cross-checks), seqtest (Swift precise / balance / fast, MoE
precise / balance), and the server gate with Swift as the dense model (balance and fast
basic,mt_cache; precise basic with `--cli`; precise `mt_cache_precise` (the mt_cache sessions with the prefix cache, replayed with `WHIRL_PREFIX_CACHE_VERIFY=1` (every resumed state == a from-scratch prefill, max abs logit difference 0) and on a server without the prefix cache (every reply equal); 0.2.2); fast_stream dense and MoE; tier,sys,restore_conc,pool in
balance and precise with `--kv auto`). Each step has an exit code and an output rule; the gate prints a
table, writes `PASS_<tag>` or `FAIL_<tag>` and exits 0 only when every selected step passed (1 a step
failed, 2 setup error such as a missing model or the wrong `-ExpectCommit`). Model and text paths come
from `WHIRL_GATE_*` variables (see the script header): `WHIRL_GATE_SWIFT` and `WHIRL_GATE_MOE` are required;
`WHIRL_GATE_DENSE` (Q4_K_M) is optional and adds the Q4_K_M selftest / golden / seqtest steps and the
bench DFlash2 hash steps (target Q4_K_M; Swift with DFlash2 is not gated yet). `-InjectFail` turns one golden reference into a
mismatch to prove the gate goes red. `whirl-server-tests --only`, `WHIRL_TIER_ONLY` and
`whirl-kernel-test --only` exit 2 for a name that matches nothing or a run with 0 checks.
