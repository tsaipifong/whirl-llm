# Threads and ownership of the server state

This page is for people changing the server (`src/server`, `src/cache`, `src/tier`). It says which thread may touch
which state, how other threads reach it, and how the rule is checked. The check is in
`src/server/engine_thread.h`; the module headers point here.

## Threads

| Thread | Started by | Does |
|---|---|---|
| engine thread | `Engine::runLoop` (whirl-server: the main thread after startup; tests: a `std::thread`) | scheduling, prefill / decode, the prefix cache, the page pool, spills and restores, the memory governor |
| HTTP accept + connection threads | `HttpServer::start` (`http.cpp`) | parse a request, render the chat template, tokenize, build a `Job`, `submitAndWait`, stream the reply; they reach the engine only through `EngineApi` (`src/server/engine_api.h`, arch024 S8) |
| Tier IO thread | `tier::Tier::start` (`kv_tier.cpp`) | SSD reads / writes of the host tier (`WriteFile` / `ReadFile`), wakes the engine |
| startup (before `runLoop`) | `serveMain` | sizing, `allocState`, `initPool`, `attachTier`, `attachVision`, `setVramProbe`, `setGovernor` |

The HTTP threads hold an `EngineApi&`, not the `Engine`: every `EngineApi` method reads atomics, takes the queue or
VRAM mutex, or returns facts taken once at startup. A new `EngineApi` method must not reach a function with an
`owner.check` (the module entry points below); work for the engine thread goes through a `Job`.

## Who owns what

| State | Owner | How other threads reach it |
|---|---|---|
| slots, `CkptStore` (checkpoint buffers), `PagePool` (KV pages), `PrefixStore` (shared prefix checkpoints, prefix cache), `TierBridge` (spill / restore), the memory governor glue (`engine_gov.cpp`), sampling buffers | the engine thread (checked: `owner.check` at the module entry points) | only through a `Job` (`submitAndWait`) or, in tests, `Engine::runOnLoop`; both run on the engine thread |
| `ServerModel` | the engine thread; called by the modules above and by the decode cycle | none. Not checked: it is on the per-token path |
| `device::Backend` (`ops_`) | **shared**: the same `*ops` is passed to `tier::Tier::create` | engine thread: every method. Tier IO thread: only `setDevice(dev_)` and `eventSync(j->fence)` (`kv_tier.cpp` ioThread / runJob; proven by the NullBackend thread record, arch024-S6a). `gpuOthers()` is stateless and may be called from any thread (the vram probe's "others"). HTTP threads: none. Not checked |
| the tier's pinned RAM blocks | `tier::Tier` | the engine thread writes / reads them with `copyAsync`; the IO thread reads / writes them on the host after `eventSync(j->fence)`; freed only in `~Tier`, after the IO thread is joined |
| tier wake (`q_mutex_`, `tier_wake_`, `wake_seq_`) and `log_fn` | the engine | the IO thread calls the wake callback (takes `q_mutex_`) and `log_fn` (the log is thread-safe) |
| request queue, `health()`, `governorJson()`, `statsShared()`, `ctxPublished()` | the engine | any thread: under `q_mutex_` / `vram_mu_`, or atomics published by the engine thread |
| `EngineOptions`, tokenizer | set at startup, read-only afterwards | any thread |

## The check

`EngineThread` (`src/server/engine_thread.h`) records the thread inside `runLoop` (`enter` at the start, `leave` when it
returns). Before `runLoop` and after it returned nothing runs the loop, so every thread may call (startup:
`allocState`, `initPool`, `attachTier` ...; tests that inspect an engine after `stopEngine`).

Each mutating public entry point of the modules calls `owner.check("Class::function")` first, before it changes
anything:

- `CkptStore`: `init`, `save`, `load`, `park`, `release`, `unpark`
- `PagePool`: `init`, `takeFree`, `take`, `ref`, `unref`, `copy`, `setQuarantineStream`, `quarantine`, `reclaim`, `cow`, `ensure`
- `PrefixStore`: `attachTier`, `setVision`, `speMatch`, `pickSlot`, `waitSpeFences`, `parkSpes`, `unparkSpes`
- `TierBridge`: `attach`, `waitSpill`, `countSpillWait`, `ckSpans`, `pageSpans`, `spillSlot`, `spillSpe`, `supersedeMark`,
  `supersedeApply`, `restoreMatch`, `startRestore`, `pumpRestore`, `restoreFailed`, `restoreDone`, `finishTail`, `cutTail`,
  `prefetchCandidate`, `prefetch`, `tick`, `flushOnStop`, `idlePoll`, `idleActivity`
- governor glue (`Engine`): `setVramProbe`, `setGovernor`, `vramIdleTick`, `govTick`, `govDrain`, `govFreeCache`,
  `govRestoreCache`, `govResize`, `govApply`

Not checked: `const` readers, accessors that return a reference (`spes()`, `cache()`, `spans()`: the change happens at
an entry point above), `device::Backend`, `ServerModel`. A module used on its own (a unit test) has no owner and is not
checked.

What a violation does:

| Build | Behaviour |
|---|---|
| test programs (`whirl-server-tests`: `EngineThread::setThrowing(true)` in `main`) | counted, then `std::logic_error("owner check: <func> called on thread <id>")`. Each test must end with no violation other than the ones it causes on purpose (a violation on an HTTP thread may be turned into a 500 by the handler, so the count is checked, not only the throw) |
| the product (`whirl-server`) | counted; the first violation of each call site logs `E owner check: <func> called on thread <id>`; the call goes on as before (no new 500 or crash on an existing path) |

The gates read the server logs with `tools/gates/server_log_scan.ps1`: `release_gate.ps1` (srv_* steps: the WDDM lines
and `E owner check:`; golden_* steps: golden's own scan), `golden.ps1` suite server (`E owner check:` per case) and the
GPU runners. A step or case with no server log fails.

Cost: one relaxed atomic load per entry point, plus `std::this_thread::get_id()` while the loop runs. The entry points
are not on the per-token path (the decode cycle calls `ServerModel` directly).

Prior art: llama.cpp's server keeps all model state on the `server_queue` loop thread and has no runtime check; Chromium's
`THREAD_CHECKER` binds to a thread and is compiled out of release builds. WHIRL keeps a log line in release so the
gates can see a violation.
