# Isolation check: modules not isolated yet

ARCH-1 8.5 (arch025-isoall). The isolation check (docs/dev/testing.md#isolation) builds a module from a copy that sees
only its declared headers and links its test program against those objects alone. A module of
`tools/gates/modules.json` gets a manifest (`tests/iso/<module>/isolation.txt`) only when that build works WITHOUT a
change to product code. This file lists every module that does not, the first include or call that blocks it, and the
smallest fix. A fix that only moves code must be proven by identical object code (objeq) before a manifest is added.

Isolated (manifest + release_gate step): sampling (`cpu_sampling_iso`), handles, platform.args, platform.fs,
platform.proc, platform.sys, spec.policy, spec.engine, model.arch_registry, server.args, threads (`cpu_iso_<module>`);
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
and model.core (every model module below it + module.cpp's product table + spec.policy, fake_hip.cpp and
tests/iso/model.core/fake_gpu_restrict.cpp; link mutants engine / log / `tier` = tier::blockHash). model.core's
declared module.cpp is a link-level cycle (config.cpp's Config::fromGguf -> findArchName(name, builtinArchs()) ->
module.cpp's table -> Config::fromGguf) that cycles.py (includes only) does not see; fix: the caller passes the table
(or checks the name through findArch) so config.cpp no longer names builtinArchs().


| module | blocked by (first reach outside the declared dependencies) | smallest fix |
|---|---|---|
| protocol | `src/server/engine_api.h:11` server/vision_iface.h -> `include/whirl/vision.h:23` device/handles.h (device) and vision::Vision; `engine_api.h:14` whirl/tokenizer.h (lib whirl, GGUF) | forward-declare the vision types engine_api / vision_iface use by pointer (header change, objeq); tokenizer (+ gguf) as declared dependencies |
| engine_gov | `src/server/engine_gov.cpp:13` `engine.h`: the file is Engine member functions | by design part of the engine (isolate after Engine is split) |
| server.startup | `src/server/server_main.cpp:13` engine.h, `:18` device/backend.h, Model::load: the program's composition root | not isolatable by design (covered by the gate's end-to-end steps) |
| cli | `src/cli/*`: whirl.exe composition root (build_model.cpp: Model::load, HIP device) | not isolatable by design |
| contract.abi, contract.device, contract.server | (arch025-lane2 N3: `contract` split in three; the old module mixed the kernel ABI / numerics (low) with the ServerModel interface (high) and closed a 15-node include cycle.) contract.abi: `src/kernels_host/kernels_abi.cpp` loads the kernel table through the HIP runtime header; contract.device: `src/device/backend_hip.cpp` is the HIP backend (whirl_hip); contract.server: `src/server/backend.h:13` `model/qwen35/kv.h`, `:14` vision_iface.h -> device/handles.h | contract.abi: numerics.cpp + dump.cpp are already isolated as declared dependencies of server.args; a manifest needs a HIP-free kernels_abi.cpp test double. contract.server: after the model split (N9) |
| model.qwen35 | `src/model/qwen35/module.cpp:10` whirl/model.h; `server_model.cpp:4` server/backend.h + `:8` whirl/model.h: the adapter between the server contract and the Model | not isolatable by design (it is the seam) |
