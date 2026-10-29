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
+ spec.policy; model.qwen35 split into model.qwen35.arch and model.qwen35.server). arch025-lane2 leg 6: model.types
(header-only: model_types.h + model_error.h over whirl/hip.h and whirl/kv_format.h; iso_check.py header-only mode,
`mutinc end` because server/engine.h includes model_types.h back).

arch025-iso3 added, each built with its own `tests/unit/<module>_tests.cpp`: gguf, tokenizer, json, mem.governor, mem.wddm_watch,
gpu_select, spec.sample, hip.leak, platform.env, platform.paths, platform.console.

arch025-iso4 added, 29 manifests (every one `iso_check --mutants` green, evidence progress/arch025-iso4.HANDOFF.md): accept_policy, kv_format,
tier.ram_size, kernels.gemm, model.arch_open (header-only mode: `mutate` names an hdr line; kernels.gemm's header sits outside src/ include/ tests/ and
is staged next to mod/), hip.mem, hip.runtime, hip.restrict (manifest kinds `sdkinc hip` = the HIP SDK include/ after the staged tree, `define NAME[=VALUE]`;
the SDK headers are the platform), release (`define WHIRL_TEST_DLL_DEPS=1`), platform.time / platform.net / platform.headers (own tests; `syslib`
winmm / ws2_32), device.null, device.profile, common.base, common.tsa, config.device (link double tests/iso/config.device/fake_device_pick.cpp), prefill.plan,
spec.rules, chat, mem.fit and server.kv_fit's unit file (tests/unit/mem_fit_tests.cpp split: vram_limit stays, the rest is tests/unit/server_kv_fit_tests.cpp, 327 +
640 = the former 967 checks), contract.abi (numerics + kernel table loader, HIP double), contract.device (backend_hip.cpp over fake_hip.cpp),
contract.server / contract.cache (the interfaces over the existing fakes), vision, and two manifests that cover several modules because their test is one program:
`ops` (ops.act, ops.attention, ops.gdn, ops.matmul, ops.moe, ops.mtp: tests/ops/ops_tests.cpp, whose cases mix the six) and `cache` (cache.fingerprint,
cache.prefix, cache.tier_adapter + contract.cache's header: tests/cache/cache_tests.cpp). cli is isolated for its one header-only part only.
Not inside a manifest although the module's name is (each a row below): release.cpp, the cli composition files, vision's vis_main.cpp, and the test
halves that need a tree or an Engine (marked "compiled out" below).

| module | blocked by (first reach outside the declared dependencies) | smallest fix |
|---|---|---|
| release (release.cpp) | `src/release/release.cpp:16` includes `whirl/model.h` (the composition root), `:10` platform/net, `:18` the generated version.h; the manifest holds `dll_deps.cpp` only (its unit case needs `define WHIRL_TEST_DLL_DEPS=1`) | move the port pre-check and the message texts that use no Model into their own file (dll_deps-like), leave `release.cpp` as the assembly |
| cli (everything but `dflash_dump.h`) | `src/cli/build_model.cpp` Model::load, the HIP device, every `cmd_*.cpp`: the whirl.exe composition root; the manifest holds the header-only `dflash_dump.h` with tests/unit/cli_tests.cpp | not isolatable by design (like server.startup); its rules go to the isolated modules (arch025-lane1) |
| vision (vis_main.cpp) | `src/vision/vis_main.cpp` is the whirl-vis executable's `main()` (a second `main` in the test program); `vis_cli.cpp:13-18` reaches `device::hipBackend()` and `gpusel::*` (declared: hip.restrict, gpu_select, the null backend behind `tests/iso/vision/fake_hip_backend.cpp`); `kernel_image.cpp` embeds hipcc output (`whirl_vision_<arch>[]`), the manifest links `tests/iso/vision/fake_vision_objects.cpp` (distinct ELF headers) instead, so the real objects are pinned by whirl-vision-tests only | none needed; a vision-encoder GPU step covers the real objects and the real backend |
| model.arch_open (include closure) | `src/model/arch_open.h:12` includes `whirl/model.h` (the whole Model class, its closure is 39 headers) for `qwen35::ModelError` and `qwen35::Config` only; no Model code is linked (the program passes its own table) | include `model/qwen35/model_error.h` and `whirl/model_types.h` (module model.types, both isolated) instead of `whirl/model.h`: the closure drops to 4 headers; product change, so left for the owner of N15d |
| cache.fingerprint (one case compiled out) | `tests/cache/cache_tests.cpp:687` testFingerprintEnvClassified scans the whole `src/` + `include/` tree (WHIRL_SOURCE_DIR): a copy that holds only the declared files has no tree; the manifest defines `WHIRL_ISO_NO_SOURCE_TREE`, the case stays in whirl-cache-tests | move the tree scan to a tools/gates check (like env_read_check.ps1) and keep only the fingerprint rules in cache_tests.cpp |
| spec.rules (one half compiled out) | `tests/server/spec_rules_tests.inc:202-241` testRulesD4's second half runs a real Engine behind a TestServer (`g_log`, `completionText`): compiled out by `define WHIRL_ISO_NO_SERVER`, it stays in whirl-server-tests | split D4 into the pickDrafts counter pin (pure) and the Engine trace pin (server test) |
| ops.act / .attention / .gdn / .matmul / .moe / .mtp (one manifest) | `tests/ops/ops_tests.cpp` testActChoice uses the matmul and gdn rules, testGdnChoice the attention / act / mtp rules, `moe.cpp` includes the matmul header: one test program, so one manifest `ops`; a call from one op module into another can not fail a link here | split ops_tests.cpp per module (the shared fake kernel table moves to a header), then one manifest each |
| cache.fingerprint / .prefix / .tier_adapter / contract.cache (one manifest) | `tests/cache/cache_tests.cpp` shares the mock tier and the fingerprint inputs between the four: one manifest `cache` | split cache_tests.cpp per module (mock_tier.h is already a header) |
| hip.restrict, hip.mem, hip.runtime (what runs) | the isolated programs run on the HIP SDK headers and the runtime loader but make no GPU call; hip.restrict tests only the paths that decide without a GPU (the user's visibility variable, the two environment readers): probing the GPUs in a helper process and `HIP_VISIBLE_DEVICES` writes are the srv_gpu_select / srv_kvram719 GPU steps | a probe seam (`runProbe` and `enumerate` behind a function table) would let a fake GPU list drive restrictToSelected; product change |
| contract.device / contract.server / contract.cache (what the test proves) | the interfaces are pinned through fakes (`fake_hip.cpp`, `fake_server_model.h`, `fake_cache_model.h`): every Backend call reaches the HIP-layer function of its name, the ServerModel hook defaults, the CacheModel shape. The behaviour of the real HIP backend is the GPU steps | none |
| engine_gov | `src/server/engine_gov.cpp:13` `engine.h`: the file is Engine member functions | by design part of the engine (isolate after Engine is split) |
| server.startup | `src/server/server_main.cpp:13` engine.h, `:18` device/backend.h, Model::load: the program's composition root | not isolatable by design (covered by the gate's end-to-end steps). It is the one place that measures (memInfo / wddmMemInfo / GpuOthers at fixed points), calls Model::load and wires the engine, tiers and HTTP; the rules it used to write inline go to modules that ARE isolated: arch025-lane1 leg 8 moved the context plan, the checkpoint default / cap / bytes and the RAM-tier floor (server.kv_fit), SLOT_DRAFTS / GATHER_MS / KV_RAM_MB / SNAP_SETS (config.settings), the MTP auto-mode rule (spec.policy, shared with the CLI) and the served model name (server.args); legs 5-7 the KV room, preload / UMA fit, draft vocabulary and batch drafts; leg 9 the DeltaNet layer count (one rule for planSlots and makeEngine), the pinned-memory, pool-refusal, pool-floor, target-not-reached and /health note texts, the default pool and the checkpoint MiB (server.kv_fit), the PREFILL_BATCH floor warning (mem.fit), the chat template description (chat), the CLI-only knob list (numerics Entry::cli_only) and the --draft dflash relaxacc / MTP-only rules (spec.policy). What is still inline, with the reason each stays or is a candidate: progress/arch025-lane1.HANDOFF.md section 49 |
| server.engine | `src/server/engine.h:12-21` the whole cache stack (cache/ckpt_store.h, page_pool.h, prefix_cache.h, prefix_store.h, tier_bridge.h; isolated only on lane 2's branch so far), `:29` tier/kv_tier.h, `:41` whirl/tokenizer.h (lib whirl, GGUF reader), `:36` mem_governor.h / `:42` wddm_watch.h; class Engine has ~229 members over 7 files (engine*.cpp) | after the cache stack's manifests (lane 2: CacheModel, prefix_store, tier_bridge, tier.kv) are merged and the D11 N4 Engine split leaves a scheduler core: declare those modules + tokenizer (tokenizer.cpp, gguf.cpp) as dependencies, FakeServerModel / NullBackend as doubles; until then whirl-server-tests (MockModel) is the CPU coverage |
| server.vram_watch | `src/server/engine_vram.cpp:9` engine.h: the file is Engine member functions (setVramProbe, vramNote, govTick's VRAM side; moved verbatim from engine_gov.cpp in N4d) | by design part of the engine, like engine_gov: isolate when N4 gives the watch its own class (probe + baseline + spill-warning state behind a small callback interface to the engine) |
| model.qwen35.server | `src/model/qwen35/server_model.cpp:4` server/backend.h + `:8` whirl/model.h: the adapter between the server contract and the Model (model.qwen35.arch, module.cpp, is isolated since arch025-lane2 leg 5) | not isolatable by design (it is the seam) |
