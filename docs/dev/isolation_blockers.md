# Isolation check: modules not isolated yet

ARCH-1 8.5 (arch025-isoall). The isolation check (docs/dev/testing.md#isolation) builds a module from a copy that sees
only its declared headers and links its test program against those objects alone. A module of
`tools/gates/modules.json` gets a manifest (`tests/iso/<module>/isolation.txt`) only when that build works WITHOUT a
change to product code. This file lists every module that does not, the first include or call that blocks it, and the
smallest fix. A fix that only moves code must be proven by identical object code (objeq) before a manifest is added.

Isolated (manifest + release_gate step): sampling (`cpu_sampling_iso`), handles, platform.args, platform.fs,
platform.proc, platform.sys, spec.policy, spec.engine, model.arch_registry, server.args, threads, server.kv_fit,
server.sample_buffers, server.image_embeddings (`cpu_iso_<module>`; the last three arch025-lane1 leg 8: no product
change, the two engine components with the test double `tests/iso/fake_server_model.h`);
server.slot and server.cycle_prof (arch025-engine leg 1: Phase / SlotData / Slot / parkSlotCkpts moved verbatim to
src/server/engine_slot.{h,cpp} on CkptStore + NullBackend with tests/iso/fake_cache_model.h; CycleProf and its print() to
src/server/engine_prof.{h,cpp} with the server log and tests/iso/fake_server_model.h; objeq: scratch/a25_engine/objeq_leg1.txt);
env_snapshot and model.tune (arch025-lane2: envGet / envFlag declared in src/config/env_get.h; Model::writeTune / readTune moved
to src/model/qwen35/model_tune.cpp and n_choices to tune_table.h); model.kv and model.scratch (arch025-lane2 N9 S2c: Model's KV /
sequence functions moved to src/model/qwen35/model_kv.cpp, Model::allocScratch to model_scratch.cpp, bufferBytes declared in
scratch.h, ModelError in src/model/qwen35/model_error.h)); forward (one manifest for forward.dispatch .. forward.vision_rows + forward.state: the forward
sources call each other's qwen35::Context members; arch025-lane2 leg 3: tuneBucket / opClassName / Profile, declared in
context.h, moved from src/model/config.cpp and loader.cpp to context.cpp; the HIP layer is the link double
tests/iso/fake_hip.cpp; link mutant `model` = qwen35::loadOrTune), model.weights (GGUF reader + kernels_abi.cpp + util /
platform, fake_hip.cpp), ckpt_store, page_pool (CacheModel + NullBackend + threads; tests/iso/fake_cache_model.h) and
tier.kv (NullBackend + platform); arch025-lane2 leg 4: prefix_store (tokens.cpp + the tokenizer and GGUF reader,
prefix_cache, tier_adapter over tier.kv, ckpt_store, NullBackend, threads), tier_bridge (prefix_store and its declared
dependencies + tier.kv; manifest `mutinc end`: tier_bridge.cpp's anonymous-namespace kv_page / msSince share their
names with engine.h's, so an engine include at the top of the file does not compile with the product flags either)
and model.core (every model module below it, fake_hip.cpp and tests/iso/model.core/fake_gpu_restrict.cpp; link
mutants engine / log / `tier` = tier::blockHash / `archtable` = builtinArchs). arch025-lane2 leg 5: the link-level
cycle model.core <-> model.qwen35 (config.cpp's Config::fromGguf -> findArchName(name, builtinArchs()) -> module.cpp's
table -> Config::fromGguf; cycles.py sees includes only) is gone: the qwen35 family's GGUF names are data in
src/model/qwen35/arch_names.h (model.core), Config::fromGguf checks them, module.cpp's row lists the same array, so
model.core no longer declares module.cpp / spec.policy. Also leg 5: protocol (S6: vision_iface.h names the vision types
only, base64Decode moved from engine_vision.cpp to src/server/base64.cpp, vision::tokenId / expand from vision.cpp to
src/vision/prompt.cpp; declared: src/chat, the tokenizer + GGUF reader, the vision prompt side, the server log, util,
platform; manifest kind `gen` for the generated whirl/version.h) and model.qwen35.arch (module.cpp over the model core
+ spec.policy; model.qwen35 split into model.qwen35.arch and model.qwen35.server).

arch025-iso3 added, each built with its own `tests/unit/<module>_tests.cpp`: gguf, tokenizer, json, mem.governor, mem.wddm_watch,
gpu_select, spec.sample, hip.leak, platform.env, platform.paths, platform.console.

| module | blocked by (first reach outside the declared dependencies) | smallest fix |
|---|---|---|
| hip.mem, hip.runtime | `src/hip/hip_api.h:12` includes the SDK header `hip/hip_runtime_api.h` (isolated build of hip.mem: C1083); `hip.runtime` also loads the driver (`runtime_loader.cpp`) | a HIP fake (`tests/iso/fake_hip.cpp`, arrives with arch025 lane 2) or the SDK include path as a manifest feature; then the manifests are 3 lines each |
| hip.restrict | no unit test file (`tests/unit/` has none for `src/hip/gpu_restrict.cpp`); also `hip_api.h` as above | write the unit test after the HIP fake exists |
| release | `tests/unit/release_tests.cpp:7,49,210` compile its only case under `WHIRL_TEST_DLL_DEPS` (`cmake/unit.cmake:20,25` defines it); iso_check.py has no compile-definition line, so the isolated program runs 0 cases and the control (needs >= 1 passed) fails. `src/release/release.cpp:16` includes `whirl/model.h` (composition root, not isolatable) | a `define` manifest kind in iso_check.py / cmake/iso.cmake; then isolate `src/release/dll_deps.cpp` only (release.cpp stays out, list it as the module's composition part) |
| accept_policy, kv_format, tier.ram_size | header-only modules (`include/whirl/relax_accept.h`, `kv_format.h`, `src/tier/ram_size.h`; kv_format and tier.ram_size have no tests/unit file either): a manifest needs at least one `src` line (`iso_check.py` Manifest: needs module, exe, src and testsrc), there is no object to link | a header-only manifest form (no `src`: compile the test against the copied headers only) in iso_check.py + cmake/iso.cmake |
| kernels.gemm (host side) | `tests/unit/kernels_gemm_tests.cpp:7` includes `../../kernels/gemm_small_addr.h` by a relative path outside the staged tree (header-only, no `src`; C1083 in the isolated copy) | header-only manifest form as above plus a staging rule for the `kernels/` header (or a test include `whirl/...`) |
| contract (partial; written before the lane 2 merge split contract into contract.abi / .device / .server / .cache) | module = `backend_hip.cpp` (HIP) + `kernels_abi.cpp` + `numerics.cpp`; `src/common/numerics.cpp` includes `whirl/dump.h` (lane 1 changes both); `tests/unit/contract_tests.cpp:7-12` also needs json, vram_limit, kv_format, device_profile | after the lane-1 merge: a manifest of numerics + kernels_abi + their declared dependencies (json, vram_limit, util), `backend_hip.cpp` left to the HIP fake; do not isolate part of the module under its name before that |
| mem.fit | `tests/unit/mem_fit_tests.cpp:7-15` mixes the tests of vram_limit with pool_budget, `startup_fit.h` (lane 1), numerics, prefill_schedule, model_types, device_profile (modules server.kv_fit, contract, prefill.plan) | split the file per module (tests/unit) after lane 1 merges, then isolate vram_limit.cpp (+ platform env) alone |
| engine_gov | `src/server/engine_gov.cpp:13` `engine.h`: the file is Engine member functions | by design part of the engine (isolate after Engine is split) |
| server.startup | `src/server/server_main.cpp:13` engine.h, `:18` device/backend.h, Model::load: the program's composition root | not isolatable by design (covered by the gate's end-to-end steps). It is the one place that measures (memInfo / wddmMemInfo / GpuOthers at fixed points), calls Model::load and wires the engine, tiers and HTTP; the rules it used to write inline go to modules that ARE isolated: arch025-lane1 leg 8 moved the context plan, the checkpoint default / cap / bytes and the RAM-tier floor (server.kv_fit), SLOT_DRAFTS / GATHER_MS / KV_RAM_MB / SNAP_SETS (config.settings), the MTP auto-mode rule (spec.policy, shared with the CLI) and the served model name (server.args); legs 5-7 the KV room, preload / UMA fit, draft vocabulary and batch drafts; leg 9 the DeltaNet layer count (one rule for planSlots and makeEngine), the pinned-memory, pool-refusal, pool-floor, target-not-reached and /health note texts, the default pool and the checkpoint MiB (server.kv_fit), the PREFILL_BATCH floor warning (mem.fit), the chat template description (chat), the CLI-only knob list (numerics Entry::cli_only) and the --draft dflash relaxacc / MTP-only rules (spec.policy). What is still inline, with the reason each stays or is a candidate: progress/arch025-lane1.HANDOFF.md section 49 |
| server.engine | `src/server/engine.h:12-21` the whole cache stack (cache/ckpt_store.h, page_pool.h, prefix_cache.h, prefix_store.h, tier_bridge.h; isolated only on lane 2's branch so far), `:29` tier/kv_tier.h, `:41` whirl/tokenizer.h (lib whirl, GGUF reader), `:36` mem_governor.h / `:42` wddm_watch.h; class Engine has 71 data members + 112 functions over 9 files (census 2026-10-29, arch025-engine; the slot types and CycleProf left it in leg 1: server.slot, server.cycle_prof) | after the cache stack's manifests (lane 2: CacheModel, prefix_store, tier_bridge, tier.kv) are merged and the D11 N4 Engine split leaves a scheduler core: declare those modules + tokenizer (tokenizer.cpp, gguf.cpp) as dependencies, FakeServerModel / NullBackend as doubles; until then whirl-server-tests (MockModel) is the CPU coverage. The split plan (members -> components, layers, steps, exit commands): progress/arch025-engine.DESIGN.md |
| server.vram_watch | `src/server/engine_vram.cpp:9` engine.h: the file is Engine member functions (setVramProbe, vramNote, govTick's VRAM side; moved verbatim from engine_gov.cpp in N4d) | by design part of the engine, like engine_gov: isolate when N4 gives the watch its own class (probe + baseline + spill-warning state behind a small callback interface to the engine) |
| cli | `src/cli/*`: whirl.exe composition root (build_model.cpp: Model::load, HIP device) | not isolatable by design |
| contract.abi, contract.device, contract.server | (arch025-lane2 N3: `contract` split in three; the old module mixed the kernel ABI / numerics (low) with the ServerModel interface (high) and closed a 15-node include cycle.) contract.abi: `src/kernels_host/kernels_abi.cpp` loads the kernel table through the HIP runtime header; contract.device: `src/device/backend_hip.cpp` is the HIP backend (whirl_hip); contract.server: `src/server/backend.h:13` `model/qwen35/kv.h`, `:14` vision_iface.h -> device/handles.h | contract.abi: numerics.cpp + dump.cpp are already isolated as declared dependencies of server.args; a manifest needs a HIP-free kernels_abi.cpp test double. contract.server: after the model split (N9) |
| model.qwen35.server | `src/model/qwen35/server_model.cpp:4` server/backend.h + `:8` whirl/model.h: the adapter between the server contract and the Model (model.qwen35.arch, module.cpp, is isolated since arch025-lane2 leg 5) | not isolatable by design (it is the seam) |
