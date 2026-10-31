# SPDX-License-Identifier: Apache-2.0
"""Unit tests of tools/gates/test_change.py plan logic (CPU, synthetic gate table / modules / impact reports; no git, no GPU).

    python -I tools/gates/test_change_test.py        exit 0 = all pass
"""
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_change as tc  # noqa: E402

GATE = [{"name": n, "gpu": g, "na": na} for n, g, na in [
    ("cpu_tests", False, ""), ("cpu_server", False, ""), ("cpu_golden_xcheck", False, ""), ("kl_guard", False, ""),
    ("kt_full", True, ""), ("st_swift_precise", True, ""), ("golden_swift", True, ""), ("golden_moe", True, ""),
    ("seq_swift_precise", True, ""), ("srv_balance", True, ""), ("srv_fast", True, ""), ("srv_cache_balance", True, ""),
    ("srv_gpu_select", True, ""), ("st_dense_precise", True, "needs WHIRL_GATE_DENSE"), ("srv_env_e2", True, "")]]
GATES = {"r9700": [dict(x, na="8060s only" if x["name"] == "srv_env_e2" else x["na"]) for x in GATE], "8060s": GATE}
MODS = {
    "protocol": {"paths": ["src/server/stream.*"], "steps": ["cpu_server", "srv_balance", "golden_swift"], "gpus": {"golden_swift": 2}},
    "tier_bridge": {"paths": ["src/server/tier_bridge.*"], "steps": ["srv_cache_balance", "srv_gpu_select"], "gpus": {"srv_gpu_select": 2}},
    "forward.attn": {"paths": ["src/model/qwen35/forward/attn.cpp"], "steps": ["st_swift_precise", "golden_swift"]},
    "page_pool": {"paths": ["src/cache/page_pool.*"], "steps": ["cpu_tests"], "when": {"include/whirl/prefill_schedule.h": ["seq_swift_precise"]}},
}
TIMES = {("r9700", "srv_balance"): 60, ("8060s", "srv_balance"): 120, ("r9700", "golden_swift"): 300, ("8060s", "golden_swift"): 360,
         ("r9700", "cpu_tests"): 6, ("r9700", "cpu_server"): 60}
fails = 0
n = 0


def check(name, cond, info=""):
    global fails, n
    n += 1
    if not cond:
        fails += 1
        print(f"FAIL {name} {info}")


def rep(modules=None, gate="", steps=(), gpus=("gfx1151", "gfx1201"), callers=None):
    return {"modules": modules or {}, "module_gate": gate, "release_gate_steps": list(steps), "gpus": list(gpus),
            "module_callers": callers or {}}


def plan(files, r, **kw):
    ch = {f: {} for f in files}
    return tc.make_plan("t", "x", ch, r, MODS, {"includers": kw.pop("includers", {})}, GATES, TIMES, **kw)


# 1. no product code -> CPU only, all CPU steps
p = plan(["docs/dev/testing.md"], rep(gate="none (no product code)"))
check("doc only level", p["level"] == "cpu", p["level"])
check("doc only no gpu", not any(p["gpu_steps"].values()))
check("cpu all", p["cpu_steps"] == ["cpu_tests", "cpu_server"], p["cpu_steps"])
# 2. one non-core module: its GPU steps; gpus 2 -> both, else --one-gpu
p = plan(["src/server/stream.cpp"], rep({"protocol": ["src/server/stream.cpp"]}, "release_gate.ps1 -Module protocol"))
check("protocol level", p["level"] == "module", p["level"])
check("protocol r97", p["gpu_steps"]["r9700"] == ["golden_swift", "srv_balance"], p["gpu_steps"])
check("protocol s8 only 2-GPU step", p["gpu_steps"]["8060s"] == ["golden_swift"], p["gpu_steps"])
p = plan(["src/server/stream.cpp"], rep({"protocol": ["src/server/stream.cpp"]}, "release_gate.ps1 -Module protocol"), one_gpu="8060s")
check("one-gpu 8060s", p["gpu_steps"]["8060s"] == ["golden_swift", "srv_balance"] and p["gpu_steps"]["r9700"] == ["golden_swift"], p["gpu_steps"])
check("estimate", p["est_gpu_min"]["8060s"] == round((360 + 120) / 60, 1), p["est_gpu_min"])
# 3. two non-core modules: union, NOT the full matrix (impact.py's cross-module full gate is overridden)
p = plan(["src/server/stream.cpp", "src/server/tier_bridge.cpp"],
         rep({"protocol": ["src/server/stream.cpp"], "tier_bridge": ["src/server/tier_bridge.cpp"]}, "full gate, both GPUs (cross-module: protocol, tier_bridge)"))
check("union level", p["level"] == "module", p)
check("union r97", set(p["gpu_steps"]["r9700"]) == {"golden_swift", "srv_balance", "srv_cache_balance", "srv_gpu_select"}, p["gpu_steps"])
check("union s8", set(p["gpu_steps"]["8060s"]) == {"golden_swift", "srv_gpu_select"}, p["gpu_steps"])
# 4. map GPU steps added on the map's GPUs; <mode> expanded; unavailable (na) dropped
p = plan(["src/server/stream.cpp"], rep({"protocol": ["src/server/stream.cpp"]}, "x", steps=["srv_fast", "st_dense_<mode>", "srv_env_e2"], gpus=("gfx1151",)))
check("map step 8060s", "srv_fast" in p["gpu_steps"]["8060s"] and "srv_fast" not in p["gpu_steps"]["r9700"], p["gpu_steps"])
check("map na dropped", "st_dense_precise" not in p["gpu_steps"]["8060s"] and "srv_env_e2" in p["gpu_steps"]["8060s"], p["gpu_steps"])
# 5. when-file adds steps on both GPUs
p = plan(["src/cache/page_pool.cpp", "include/whirl/prefill_schedule.h"], rep({"page_pool": ["src/cache/page_pool.cpp"]}, "release_gate.ps1 -Module page_pool -Steps seq_swift_precise"))
check("when both", all("seq_swift_precise" in p["gpu_steps"][d] for d in tc.DEVICES), p["gpu_steps"])
# 6. core module -> full matrix on both GPUs (all available GPU steps)
p = plan(["src/model/qwen35/forward/attn.cpp"], rep({"forward.attn": ["src/model/qwen35/forward/attn.cpp"]}, "release_gate.ps1 -Module forward.attn"))
check("core full", p["level"] == "full" and any("core module forward.attn" in r for r in p["full_reasons"]), p["full_reasons"])
check("core full steps r97", len(p["gpu_steps"]["r9700"]) == 9 and "srv_env_e2" not in p["gpu_steps"]["r9700"], p["gpu_steps"]["r9700"])
check("core full steps s8", len(p["gpu_steps"]["8060s"]) == 10, p["gpu_steps"]["8060s"])
p = plan(["src/server/stream.h"], rep({"protocol": ["src/server/stream.h"]}, "release_gate.ps1 -Module protocol,forward.attn", callers={"protocol": ["forward.attn"]}))
check("core caller full", p["level"] == "full" and any("includes a changed header of protocol" in r for r in p["full_reasons"]), p["full_reasons"])
# 7. core by path for an old path no module glob matches
p = plan(["src/model/forward/attn.cpp"], rep({}, "full gate or the release steps above: 1 file(s) in no module (src/model/forward/attn.cpp)"))
check("core path", p["level"] == "full" and any(r.startswith("core file src/model/forward/attn.cpp") for r in p["full_reasons"]), p["full_reasons"])
# 8. build-wide / unmapped product file -> full (impact.py's reason kept)
p = plan(["cmake/server.cmake", "src/server/stream.cpp"], rep({"protocol": ["src/server/stream.cpp"]}, "full gate or the release steps above: 1 file(s) in no module (cmake/server.cmake); cmake/server.cmake is build-wide"))
check("build-wide full", p["level"] == "full", p["level"])
check("reason class bw", {tc.reason_class(r) for r in p["full_reasons"]} == {"impact: build-wide file"}, p["full_reasons"])
p = plan(["src/server/engine.cpp"], rep({}, "full gate or the release steps above: 1 file(s) in no module (src/server/engine.cpp)"))
check("unmapped full", p["level"] == "full" and tc.reason_class(p["full_reasons"][0]) == "impact: product file in no module", p["full_reasons"])
# 9. contract module (modules.json full) -> full via impact text and CORE
p = plan(["src/device/backend.h"], rep({"contract.device": ["src/device/backend.h"]}, "full gate, both GPUs (contract module contract.device)"))
check("contract full", p["level"] == "full" and any("core module contract.device" in r for r in p["full_reasons"]), p["full_reasons"])
# 10. arch-specific kernel -> full matrix on that GPU only
p = plan(["kernels/gfx1151/attn.hip"], rep({}, "full gate or the release steps above: 1 file(s) in no module (kernels/gfx1151/attn.hip)", gpus=("gfx1151",)))
check("kernel 1151 only", p["level"] == "full" and p["gpu_steps"]["r9700"] == [] and len(p["gpu_steps"]["8060s"]) == 10, p["gpu_steps"])
# 10b. the kernel ABI layout table (host + device, arch025 N13) -> full matrix on both GPUs, counted as a kernel source
p = plan(["include/whirl/kernels_abi_layout.h"], rep({"kernels.abi": ["include/whirl/kernels_abi_layout.h"]}, "full gate, both GPUs (contract module kernels.abi)"))
check("abi layout full both", p["level"] == "full" and all(p["gpu_steps"][d] for d in tc.DEVICES)
      and any(r.startswith("core file include/whirl/kernels_abi_layout.h") for r in p["full_reasons"]), p["full_reasons"])
check("abi layout is kernel source", "kernels_abi_layout.h" in p["work_equivalence"]["static"], p.get("work_equivalence"))
# 11. work-equivalence proof: skips the covered steps when bound to --build; not when a kernel / .hip-included header changed
with tempfile.TemporaryDirectory() as td:
    b = os.path.join(td, "build")
    os.makedirs(b)
    open(os.path.join(b, "whirl.exe"), "wb").write(b"exe-b")
    sha = hashlib.sha256(b"exe-b").hexdigest()

    def proof(name, **over):
        r = {"format": "whirl-work-equal/1", "verdict": "EQUIVALENT", "checks": {"code_objects": "identical-except-cuid", "launch_sequence": "identical", "golden": "identical"},
             "b": {"exes": {"whirl.exe": sha}}, "launch": [{"runs": [{"key": "st_swift_precise/whirl.exe selftest x.gguf --device gfx1201"}]}]}
        r.update(over)
        f = os.path.join(td, name)
        json.dump(r, open(f, "w"))
        return f

    core = rep({"forward.attn": ["src/model/qwen35/forward/attn.cpp"]}, "release_gate.ps1 -Module forward.attn")
    p = plan(["src/model/qwen35/forward/attn.cpp"], core, build=b, proofs={"r9700": proof("ok.json")})
    sk = p["skipped_by_proof"]["r9700"]
    check("proof skips covered", set(sk) == {"kt_full", "st_swift_precise", "golden_swift", "golden_moe"}, sk)
    check("proof keeps server", "srv_balance" in p["gpu_steps"]["r9700"] and "st_swift_precise" not in p["gpu_steps"]["r9700"], p["gpu_steps"]["r9700"])
    check("proof other gpu untouched", p["skipped_by_proof"]["8060s"] == [] and "kt_full" in p["gpu_steps"]["8060s"])
    p = plan(["src/model/qwen35/forward/attn.cpp"], core, build=b, proofs={"8060s": proof("wrongdev.json")})
    check("proof wrong gpu", p["skipped_by_proof"]["8060s"] == [] and "another GPU" in p["proofs"]["8060s"], p["proofs"])
    p = plan(["src/model/qwen35/forward/attn.cpp"], core, build=b, proofs={"r9700": proof("bad.json", verdict="INCOMPLETE")})
    check("proof incomplete", p["skipped_by_proof"]["r9700"] == [] and "INCOMPLETE" in p["proofs"]["r9700"], p["proofs"])
    p = plan(["src/model/qwen35/forward/attn.cpp"], core, build=b, proofs={"r9700": proof("sha.json", b={"exes": {"whirl.exe": "0" * 64}})})
    check("proof other exe", p["skipped_by_proof"]["r9700"] == [] and "not the B exe" in p["proofs"]["r9700"], p["proofs"])
    p = plan(["src/model/qwen35/forward/attn.cpp"], core, build=None, proofs={"r9700": proof("ok2.json")})
    check("proof no build", p["skipped_by_proof"]["r9700"] == [], p["proofs"])
    p = plan(["kernels/attn.hip"], rep({}, "full gate"), build=b, proofs={"r9700": proof("ok3.json")})
    check("proof kernel touched", p["skipped_by_proof"]["r9700"] == [] and "not used" in p["proofs"]["r9700"], p["proofs"])
    p = plan(["include/whirl/x.h"], rep({}, "full gate"), build=b, proofs={"r9700": proof("ok4.json")}, includers={"include/whirl/x.h": ["kernels/attn.hip"]})
    check("proof hip header", p["skipped_by_proof"]["r9700"] == [] and "x.h" in p["work_equivalence"]["static"], p["work_equivalence"])
    p = plan(["cmake/model.cmake", "src/model/qwen35/forward/attn.cpp"], core, build=b, proofs={"r9700": proof("ok5.json")})
    check("proof with build file", "kt_full" in p["skipped_by_proof"]["r9700"] and "build files" in p["work_equivalence"]["static"], p["work_equivalence"])
    # 12. ledger row
    class A:
        ledger = os.path.join(td, "ledger.tsv")
    p = plan(["src/server/stream.cpp"], rep({"protocol": ["src/server/stream.cpp"]}, "x"))
    tc.ledger(A, p, [{"cpu_s": 120, "gpu_s": 600}], "PASS", "s.json")
    tc.ledger(A, p, None, "PLAN", "")
    rows = open(A.ledger, encoding="utf-8").read().splitlines()
    check("ledger header + 2 rows", len(rows) == 3 and rows[0].startswith("ts\tkind\tchange"), rows)
    check("ledger run minutes", rows[1].split("\t")[12:15] == ["2.0", "10.0", "PASS"], rows[1])
# 13. summary parser
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "x_summary.txt")
    open(f, "w").write("release gate x on 8060s (gfx1151), models ...\nPASS cpu_tests              exit   0      3 s  ok\n"
                       "FAIL srv_balance            exit   1    120 s  bad\nrelease gate x: FAIL\n")
    s = tc.parse_summary(f)
    check("summary parse", s == ("x", "8060s", [("PASS", "cpu_tests", 3), ("FAIL", "srv_balance", 120)]), s)
    out = os.path.join(td, "times.tsv")
    tc.refresh_times([td], out)
    t = tc.load_times(out)
    check("times PASS only", t.get(("8060s", "cpu_tests")) == 3 and ("8060s", "srv_balance") not in t, t)
# 13b. arch025-procopt2 (PLAN 9.28): FLAKY rows of release_gate 9.21 ("FLAKY <step> exit <rc> <s> s <marker> | first try:
# exit <rc> <s> s <marker>") are parsed; the step's cost is the PASSING retry's seconds (the row's own column: one clean run,
# as a PASS row), never the failed first try's (a crash or a cap, not the step's length) nor their sum
with tempfile.TemporaryDirectory() as td:
    f = os.path.join(td, "y_summary.txt")
    open(f, "w").write("release gate y on r9700 (gfx1201), models ...; retry: CPU steps once (dev); heavy-CPU lock off; parallel: off (-Jobs 1)\n"
                       "PASS cpu_tests              exit   0      5 s  ok\n"
                       "FLAKY cpu_iso_tier_kv        exit   0     40 s  iso check: ok (tier.kv, 6/6 cases as expected) | first try: exit   1    300 s  iso check: FAIL\n"
                       "FAIL cpu_asan               exit   1    900 s  bad | retry also failed (first try: exit   1    800 s  bad)\n"
                       "release gate y: FAIL (1 of 3 steps failed)\n")
    s = tc.parse_summary(f)
    check("summary parse FLAKY row (retry seconds)", s == ("y", "r9700", [("PASS", "cpu_tests", 5), ("FLAKY", "cpu_iso_tier_kv", 40), ("FAIL", "cpu_asan", 900)]), s)
    out = os.path.join(td, "times.tsv")
    tc.refresh_times([td], out)
    t = tc.load_times(out)
    check("times: a FLAKY step counts with its passing retry's seconds", t.get(("r9700", "cpu_iso_tier_kv")) == 40 and t.get(("r9700", "cpu_tests")) == 5
          and t.get(("r9700", "cpu_asan")) == tc.TIMING_DEFAULTS["cpu_asan"][0], t)
    # the machine time of the run (--run's cpu_s / gpu_s) is both tries: the first try's seconds of FLAKY and retried FAIL rows
    ft = tc.first_try_seconds(f)
    check("first-try seconds of FLAKY and 'retry also failed' rows", ft == {"cpu_iso_tier_kv": 300, "cpu_asan": 800}, ft)
# 14. --run: one gate call per GPU, R9700 then 8060S, CPU steps once (first call), per-GPU CPU checks on both, tag
# change_<id>_<dev>, summary + ledger; a running whirl process refuses (exit 2). release_gate is faked (no GPU).
with tempfile.TemporaryDirectory() as td:
    calls = []
    real_run = tc.subprocess.run

    def fake_run(cmd, stdout=None, stderr=None, **kw):
        calls.append(cmd)
        dev = cmd[cmd.index("-Device") + 1]
        steps = cmd[cmd.index("-Steps") + 1].split(",")
        tag = cmd[cmd.index("-Tag") + 1]
        w = cmd[cmd.index("-Work") + 1]
        os.makedirs(w, exist_ok=True)
        with open(os.path.join(w, f"{tag}_summary.txt"), "w") as fh:
            fh.write(f"release gate {tag} on {dev} (x)\n")
            for s in steps:
                fh.write(f"{'FAIL' if s == 'srv_balance' and dev == '8060s' else 'PASS'} {s:<22} exit   0     60 s  ok\n")

        class R:
            returncode = 1 if dev == "8060s" and "srv_balance" in steps else 0
        return R()

    class A:
        work = os.path.join(td, "w")
        force = False
        build = None
        ledger = os.path.join(td, "ledger.tsv")
    p = plan(["src/server/stream.cpp"], rep({"protocol": ["src/server/stream.cpp"]}, "x"), one_gpu="8060s")
    tc.subprocess.run = fake_run
    tc.whirl_running = lambda: []
    try:
        rc = tc.run_plan(p, A)
    finally:
        tc.subprocess.run = real_run
    check("run order", [c[c.index("-Device") + 1] for c in calls] == ["r9700", "8060s"], calls)
    s1 = calls[0][calls[0].index("-Steps") + 1].split(",")
    s2 = calls[1][calls[1].index("-Steps") + 1].split(",")
    check("run cpu once", "cpu_tests" in s1 and "cpu_tests" not in s2 and "cpu_golden_xcheck" in s2 and "kl_guard" in s1, (s1, s2))
    check("run tags", all(c[c.index("-Tag") + 1].startswith("change_") for c in calls))
    check("run launch log", all("-LaunchLog" in c for c in calls))
    check("run fail rc", rc == 1, rc)
    res = json.load(open(os.path.join(A.work, "test_change_summary.json")))
    check("run summary", res["result"] == "FAIL" and res["runs"][1]["failed"] == ["srv_balance"], res["runs"])
    row = open(A.ledger, encoding="utf-8").read().splitlines()[1].split("\t")
    check("run ledger", row[1] == "run" and row[14] == "FAIL" and float(row[13]) > 0, row)
    tc.whirl_running = lambda: ["whirl.exe"]
    calls.clear()
    tc.subprocess.run = fake_run
    try:
        rc = tc.run_plan(p, A)
    finally:
        tc.subprocess.run = real_run
    check("run refuses busy GPU", rc == 2 and not calls, (rc, calls))
# 15. arch025-modsel2 (PLAN 3.3): a list-only CMake edit (impact.py cmake_lists) is the module of its sources: module
# level, not a build file for work equivalence; a listed source on a core path still forces the full matrix
r15 = dict(rep({"protocol": ["cmake/server.cmake", "src/server/stream.cpp"]}, "release_gate.ps1 -Module protocol"),
           cmake_lists={"cmake/server.cmake": ["src/server/stream.cpp"]})
p = plan(["cmake/server.cmake", "src/server/stream.cpp"], r15)
check("cmake list-only = module", p["level"] == "module" and p["gpu_steps"]["r9700"] == ["golden_swift", "srv_balance"], (p["level"], p["full_reasons"]))
check("cmake list-only not a build file", "build files" not in p["work_equivalence"]["static"], p["work_equivalence"])
r15c = dict(rep({"protocol": ["cmake/server.cmake"]}, "release_gate.ps1 -Module protocol"), cmake_lists={"cmake/server.cmake": ["src/ops/x.cpp"]})
p = plan(["cmake/server.cmake"], r15c)
check("cmake lists a core path = full", p["level"] == "full" and any("src/ops/x.cpp" in r and "cmake/server.cmake" in r for r in p["full_reasons"]), p["full_reasons"])
p = plan(["cmake/server.cmake", "src/server/stream.cpp"], dict(r15, cmake_lists={}))
check("cmake not list-only stays build file", "build files" in p["work_equivalence"]["static"], p["work_equivalence"])
# 16. arch025-modsel2 (PLAN 3.5): dev-time plans skip cpu_asan / cpu_analyze / cpu_tsa / cpu_safety_selftest only when
# every changed file is outside their inputs (allow-list: docs, *.md, python gate / impact tools, golden refs); any
# C / C++ / HIP / CMake / build / safety-tool / isolation-manifest change runs them (the release gate always does)
SG = [dict(x) for x in GATE] + [{"name": s, "gpu": False, "na": ""} for s in ("cpu_safety_selftest", "cpu_asan", "cpu_analyze", "cpu_tsa")]
SGATES = {"r9700": SG, "8060s": SG}


def splan(files, r=None, **kw):
    return tc.make_plan("t", "x", {f: {} for f in files}, r or rep(gate="none (no product code)"), MODS, {"includers": {}}, SGATES, TIMES, **kw)


SAF = set(tc.SAFETY_STEPS)
for fs in (["docs/dev/testing.md"], ["README.md", "tools/gates/test_change.py"], ["tools/impact/impact.py", "tools/impact_map.json"],
           ["tests/golden/swift.json"], ["tools/gates/module_gate_cases.tsv", "tools/gates/modules.json"]):
    p = splan(fs)
    check(f"safety skipped {fs[0]}", not SAF & set(p["cpu_steps"]) and set(p["cpu_skipped"]["steps"]) == SAF, (p["cpu_steps"], p.get("cpu_skipped")))
for fs in (["src/server/stream.cpp"], ["tests/server/server_tests.cpp"], ["tools/safety/run_asan.bat"], ["CMakeLists.txt"],
           ["cmake/server.cmake"], ["tests/iso/forward/isolation.txt"], ["kernels/gemv.hip"], ["src/hip/embedded_objects.cpp.in"],
           ["docs/x.md", "include/whirl/model.h"], ["tools/gates/release_gate.ps1"], ["build.bat"], ["tools/gen-unicode-tables/main.cpp"]):
    p = splan(fs)
    check(f"safety runs {fs[-1]}", SAF <= set(p["cpu_steps"]) and not p["cpu_skipped"]["steps"], (p["cpu_steps"], p.get("cpu_skipped")))
p = splan(["docs/dev/testing.md"], all_cpu=True)
check("--all-cpu keeps the safety steps", SAF <= set(p["cpu_steps"]), p["cpu_steps"])
p = splan([])
check("empty change skips nothing", SAF <= set(p["cpu_steps"]), p["cpu_steps"])
# 17. arch025-modsel2 (PLAN 3.6, X12 Findings K8 1-7): pins on THIS tree's modules.json for the impact-map gaps whose
# GPU selection is closed by the module table (no map edge needed): K8-2 ~Model / member destruction order
# (src/model/loader.cpp, include/whirl/model.h: model.core, full) and K8-5 kernel-ABI struct member deletion
# (include/whirl/kernels_abi.h: contract.abi + CORE_GLOBS) plan the full matrix on both GPUs; a file moved out of
# those modules breaks the pin here, not silently in a dev run.
RMODS = tc.load_mods(tc.REPO)
sys.path.insert(0, os.path.join(tc.REPO, "tools", "impact"))
import impact as _imp  # noqa: E402
for f, why in (("src/model/loader.cpp", "K8-2 ~Model"), ("include/whirl/model.h", "K8-2 Model member order"),
               ("include/whirl/kernels_abi.h", "K8-5 ABI struct member deletion")):
    g = _imp.module_gate({f: {"old_path": f, "new": [1], "old": [1]}}, RMODS)
    r = rep(g["modules"], g["module_gate"], callers=g.get("module_callers"))
    p = tc.make_plan("t", "x", {f: {}}, r, RMODS, {"includers": {}}, GATES, TIMES, core_sets={})  # module table pin (N11 sets: below)
    check(f"{why} ({f}) = full matrix both GPUs", p["level"] == "full" and all(len(p["gpu_steps"][d]) >= 9 for d in tc.DEVICES),
          (g["module_gate"], p["level"], p["full_reasons"][:2]))
# 18. arch025-modsel2 (PLAN 3.6, K8-3 GPU half): a gate script a GPU step runs selects that step (on --one-gpu); a script
# release_gate.ps1 uses outside any AddStep statement, or a release_gate.ps1 line outside every AddStep, is common code
# -> every GPU step on one GPU; a script the gate never reaches selects no GPU step (CPU all covers the CPU steps)
PS = "\n".join([
    '$vs = Join-Path $repo "tools\\gates\\vis_ab.ps1"',                                   # 1
    'AddStep "cpu_tests" $false @($py, "x_check.py") `',                                  # 2
    '    { param($o, $c) $c -eq 0 }',                                                     # 3
    'foreach ($m in "swift", "moe") {',                                                   # 4
    '    AddStep "golden_$m" $true @($pwsh, "-File", (Join-Path $PSScriptRoot "golden.ps1"))',  # 5
    '}',                                                                                  # 6
    'AddStep "srv_balance" $true @($pwsh, "-File", $vs)',                                 # 7
    '$scan = & (Join-Path $PSScriptRoot "server_log_scan.ps1") -Dir $wd',                  # 8
    'AddStep "kt_full" $true @($exe)'])                                                   # 9
gm = tc.gate_script_map(PS, {"golden.ps1": "& (Join-Path $PSScriptRoot 'helper_lines.py')", "helper_lines.py": ""})
check("map direct", gm["refs"].get("x_check.py") == {"cpu_tests"}, gm["refs"])
check("map loop name", gm["refs"].get("golden.ps1") == {"golden_*"}, gm["refs"])
check("map variable", gm["refs"].get("vis_ab.ps1") == {"srv_balance"}, gm["refs"])
check("map common ref", "server_log_scan.ps1" in gm["common"], gm["common"])
check("map indirect", gm["refs"].get("helper_lines.py") == {"golden_*"}, gm["refs"])
check("map lines", tc.gate_line_steps(gm, [3]) == {"cpu_tests"} and tc.gate_line_steps(gm, [8]) is None and tc.gate_line_steps(gm, [5]) == {"golden_*"},
      (tc.gate_line_steps(gm, [3]), tc.gate_line_steps(gm, [8])))


def gplan(files, changes=None, **kw):
    ch = changes or {f: {} for f in files}
    return tc.make_plan("t", "x", ch, rep(gate="none (no product code)"), MODS, {"includers": {}}, GATES, TIMES, gate_map=gm, **kw)


p = gplan(["tools/gates/vis_ab.ps1"])
check("gate script -> its GPU step on one GPU", p["gpu_steps"]["r9700"] == ["srv_balance"] and p["gpu_steps"]["8060s"] == [] and p["level"] == "module", (p["level"], p["gpu_steps"]))
p = gplan(["tools/golden/helper_lines.py"])
check("indirect script -> loop steps", p["gpu_steps"]["r9700"] == ["golden_swift", "golden_moe"], p["gpu_steps"])
p = gplan(["tools/gates/server_log_scan.ps1"])
check("common script -> every GPU step, one GPU", len(p["gpu_steps"]["r9700"]) == 9 and p["gpu_steps"]["8060s"] == [], p["gpu_steps"])
p = gplan(["tools/gates/release_gate.ps1"], {"tools/gates/release_gate.ps1": {"new": [3], "old": [3]}})
check("release_gate.ps1 AddStep line of a CPU step -> no GPU step", not any(p["gpu_steps"].values()), p["gpu_steps"])
p = gplan(["tools/gates/release_gate.ps1"], {"tools/gates/release_gate.ps1": {"new": [8], "old": []}})
check("release_gate.ps1 common line -> every GPU step, one GPU", len(p["gpu_steps"]["r9700"]) == 9, p["gpu_steps"])
p = gplan(["tools/gates/module_gate_mutate.py"])
check("script the gate never runs -> no GPU step", not any(p["gpu_steps"].values()) and p["level"] == "cpu", (p["level"], p["gpu_steps"]))
p = gplan(["tools/gates/vis_ab.ps1"], one_gpu="8060s")
check("gate script on --one-gpu 8060s", p["gpu_steps"]["8060s"] == ["srv_balance"] and p["gpu_steps"]["r9700"] == [], p["gpu_steps"])
# 19. arch025-modsel2 (PLAN 3.6, K8-7: product commits that selected no GPU step): product code whose modules select
# no GPU step is the full matrix unless every such module is in NO_GPU_MODULES (compile-only / no-code modules, each
# with its reason); this tree's modules.json has no other non-full module without a GPU step or callers
p = plan(["src/x/y.cpp"], rep({"m0": ["src/x/y.cpp"]}, "release_gate.ps1 -Module m0"))
check("product code, module without GPU steps = full", p["level"] == "full" and any("no GPU step" in r for r in p["full_reasons"]), (p["level"], p["full_reasons"]))
p = tc.make_plan("t", "x", {"src/platform/headers_win.cpp": {}}, rep({"platform.headers": ["src/platform/headers_win.cpp"]}, "release_gate.ps1 -Module platform.headers"),
                 dict(MODS, **{"platform.headers": {"paths": ["src/platform/headers_win.cpp"], "steps": ["cpu_tests"]}}), {"includers": {}}, GATES, TIMES)
check("NO_GPU_MODULES member stays CPU", p["level"].startswith("cpu"), (p["level"], p["full_reasons"]))
lone = [k for k, d in RMODS.items() if not d.get("full") and not d.get("callers") and k not in tc.NO_GPU_MODULES
        and not any(not s.startswith(("cpu_", "kl_")) for s in d.get("steps") or [])]
check("every non-full module has a GPU step, callers, or a NO_GPU_MODULES reason", not lone, lone)
# 20. arch025-modsel3 (PLAN 3.7): a changed source of a test executable selects the GPU steps that run that executable
# (release_gate.ps1: AddStep text, the variables / functions it uses, the gate scripts it calls); headers under tests/
# through the map's includers; whirl-kernel-test on both GPUs (each GPU's own kernels), the others on --one-gpu; a test
# source no GPU step's executable is built from selects none (CPU all runs it)
PS2 = "\n".join([
    '$ktArgs = @($exe["whirl-kernel-test"], "--device", $Device)',
    'function GateCmd([string]$m) {',
    '    $a = @($exe["whirl-server-gate"], "--exe", $exe["whirl-server"], "--model", $m)',
    '    $a',
    '}',
    'AddStep "kt_full" $true $ktArgs { param($o, $c) $c -eq 0 }',
    'AddStep "srv_balance" $true (GateCmd $SW) $gateRule',
    'AddStep "golden_swift" $true @($pwsh, "-File", (Join-Path $repo "tools\\golden\\golden.ps1"))',
    'AddStep "st_swift_precise" $true @($exe["whirl"], "selftest")',
    'AddStep "cpu_server" $false @($exe["whirl-server-tests"])'])
gm2 = tc.gate_script_map(PS2, {"golden.ps1": "$g = Join-Path $Bin 'whirl-server-gate.exe'\n& $g --suite golden"})
ex = gm2.get("exes", {})
check("exe map: variable", ex.get("whirl-kernel-test") == {"kt_full"}, ex)
check("exe map: function + script", ex.get("whirl-server-gate") == {"srv_balance", "golden_swift"}, ex)
check("exe map: direct", ex.get("whirl") == {"st_swift_precise"} and ex.get("whirl-server-tests") == {"cpu_server"}, ex)
M7 = {"includers": {"tests/server/http_client.h": ["tests/server/http_client.cpp"], "tests/server/deep.h": ["tests/server/http_client.h"]},
      "executables": {"whirl-kernel-test": ["tests/kernels/test_gemv.cpp"], "whirl-server-gate": ["tests/server/server_gate.cpp", "tests/server/http_client.cpp"],
                      "whirl-server-tests": ["tests/server/http_client.cpp", "tests/server/server_tests.cpp"]}}
check("test source exes", tc.test_source_exes(["tests/server/deep.h"], M7) == {"tests/server/deep.h": ["whirl-server-gate", "whirl-server-tests"]},
      tc.test_source_exes(["tests/server/deep.h"], M7))


def tplan(files, **kw):
    return tc.make_plan("t", "x", {f: {} for f in files}, rep(gate="none (no product code)"), MODS, M7, GATES, TIMES, gate_map=gm2, **kw)


p = tplan(["tests/kernels/test_gemv.cpp"])
check("kernel test source -> kt_full on both GPUs", p["level"] == "module" and p["gpu_steps"]["r9700"] == ["kt_full"] and p["gpu_steps"]["8060s"] == ["kt_full"], (p["level"], p["gpu_steps"]))
p = tplan(["tests/server/server_gate.cpp"])
check("server gate source -> its GPU steps, one GPU", p["gpu_steps"]["r9700"] == ["golden_swift", "srv_balance"] and p["gpu_steps"]["8060s"] == [], p["gpu_steps"])
check("why names the exe", "whirl-server-gate" in p["gpu_why"].get("r9700:srv_balance", ""), p["gpu_why"])
p = tplan(["tests/server/deep.h"], one_gpu="8060s")
check("test header -> includer's exe, --one-gpu 8060s", p["gpu_steps"]["8060s"] == ["golden_swift", "srv_balance"] and p["gpu_steps"]["r9700"] == [], p["gpu_steps"])
p = tplan(["tests/server/server_tests.cpp"])
check("CPU test exe source -> no GPU step", p["level"] == "cpu" and not any(p["gpu_steps"].values()), (p["level"], p["gpu_steps"]))
p = tplan(["tests/nowhere/x.cpp"])
check("test source in no exe -> no GPU step", p["level"] == "cpu", p["level"])
p = tplan(["tests/golden/swift_gfx1201.json"])
check("golden refs are not a test exe source", not any(p["gpu_steps"].values()), p["gpu_steps"])
# pins on THIS tree: the gate's exe map and the impact map's executables (a renamed exe / step breaks it here)
RGM = tc.load_gate_map(tc.REPO)
check("tree: whirl-kernel-test runs in kt_full", "kt_full" in RGM["exes"].get("whirl-kernel-test", set()), RGM["exes"].get("whirl-kernel-test"))
check("tree: whirl-server-gate runs in srv_* and golden_*", {"srv_balance", "golden_*"} <= RGM["exes"].get("whirl-server-gate", set()), RGM["exes"].get("whirl-server-gate"))
RMAP = json.load(open(os.path.join(tc.REPO, "tools", "impact_map.json"), encoding="utf-8"))
check("tree: test_gemv.cpp is a whirl-kernel-test source", tc.test_source_exes(["tests/kernels/test_gemv.cpp"], RMAP).get("tests/kernels/test_gemv.cpp") == ["whirl-kernel-test"],
      tc.test_source_exes(["tests/kernels/test_gemv.cpp"], RMAP))
ARTS = {tc.ARTIFACT.get(t, t) for t in RMAP["executables"]}
check("tree: every exe a gate step runs is an impact-map executable", not (set(RGM["exes"]) - ARTS), sorted(set(RGM["exes"]) - ARTS))
# 20b. the gate map reads a multi-line AddStep (an argument list still open at the line end: golden_<m>) and a ${m}
# braced variable in a step name (st_${m}_$mode -> st_*_*)
PS3 = "\n".join([
    'foreach ($m in "swift") {',
    '    AddStep "golden_$m" $true @($pwsh, "-File", "golden.ps1", "-Exe", $exe["whirl"],',
    '                         "-GateExe", $exe["whirl-server-gate"]) `',
    '        { param($o, $c) $c -eq 0 }',
    '    AddStep "st_${m}_$mode" $true @($exe["whirl"], "selftest")',
    '}'])
gm3 = tc.gate_script_map(PS3, {})
check("multi-line AddStep: exe on the 2nd line", gm3["exes"].get("whirl-server-gate") == {"golden_*"}, gm3["exes"])
check("multi-line AddStep: lines 2-4 are one statement", tc.gate_line_steps(gm3, [3]) == {"golden_*"}, tc.gate_line_steps(gm3, [3]))
check("braced variable in a step name", "st_*_*" in gm3["exes"].get("whirl", set()), gm3["exes"])
# 21. arch025-modsel3 (PLAN 3.8): a non-list CMake edit is attributed through the CMake build graph (tools/impact/
# cmake_graph.py): the sources it builds differently count as changed (module / core rules), a GPU test exe it relinks
# selects that exe's steps, anything the graph cannot attribute (or a failed configure) stays the full matrix
CM = "cmake/server.cmake"


def cgplan(graph, r=None, files=(CM,), **kw):
    return tc.make_plan("t", "x", {f: {} for f in files}, r or rep(gate="none (no product code)"), MODS, M7, GATES, TIMES, gate_map=gm2,
                        cmake_graph=graph, **kw)


G0 = {"ok": True, "full": [], "sources": {}, "exe_changes": {}, "notes": []}
p = cgplan(dict(G0, sources={"whirl": ["src/server/stream.cpp"]}), rep({"protocol": ["src/server/stream.cpp"]}, "release_gate.ps1 -Module protocol"))
check("graph source -> its module, not full", p["level"] == "module" and p["gpu_steps"]["r9700"] == ["golden_swift", "srv_balance"], (p["level"], p["full_reasons"], p["gpu_steps"]))
check("graph shown in the plan", p.get("cmake_graph", {}).get("sources") == {"whirl": ["src/server/stream.cpp"]}, p.get("cmake_graph"))
p = cgplan(dict(G0, sources={"whirl": ["src/model/qwen35/forward/attn.cpp"]}), rep({"forward.attn": ["x"]}, "release_gate.ps1 -Module forward.attn"))
check("graph source on a core path -> full", p["level"] == "full" and any("CMake graph" in x for x in p["full_reasons"]), p["full_reasons"])
p = cgplan(dict(G0, full=["product executable whirl: link rule / flags / external libraries changed"]))
check("graph full reason -> full", p["level"] == "full" and any("link rule" in x for x in p["full_reasons"]), p["full_reasons"])
p = cgplan({"ok": False, "why": "configure failed", "full": [], "sources": {}, "exe_changes": {}, "notes": []})
check("graph not ok -> full", p["level"] == "full" and any("configure failed" in x for x in p["full_reasons"]), p["full_reasons"])
p = cgplan(dict(G0, exe_changes={"whirl-server-gate": ["link"]}, sources={"whirl-server-gate": ["tests/server/server_gate.cpp"]}))
check("graph relinks a GPU test exe -> its steps", p["level"] == "module" and p["gpu_steps"]["r9700"] == ["golden_swift", "srv_balance"], (p["level"], p["gpu_steps"]))
p = cgplan(G0)
check("graph: nothing for the GPU -> CPU only", p["level"] == "cpu" and not p["full_reasons"], (p["level"], p["full_reasons"]))
# graph_rep: when it runs and what it hands to impact.analyze
calls = []


def fake_runner(root, old, new, work, gpu_exes):
    calls.append((old, new, sorted(gpu_exes)))
    return dict(G0, sources={"whirl": ["src/server/stream.cpp"], "whirl-server-gate": ["tests/server/server_gate.cpp"]})


def fake_analyze(ch, m, new, old):
    return {"modules": {}, "module_gate": "seen " + ",".join(sorted(ch)), "release_gate_steps": [], "gpus": [], "seen": ch}


BW = {"modules": {}, "module_gate": "full gate or the release steps above: 1 file(s) in no module (cmake/server.cmake); cmake/server.cmake is build-wide"}
r2, g2 = tc.graph_rep({CM: {"new": [3], "old": [3]}}, BW, {}, "R", "a1", "b2", "W", gm2, runner=fake_runner, analyze_fn=fake_analyze, lines_fn=lambda rev, f: 7)
check("graph_rep runs the graph with the revisions and the GPU exes", calls and calls[0][:2] == ("a1", "b2") and "whirl-kernel-test" in calls[0][2], calls)
check("graph_rep: CMake file replaced by whole-file changes of the graph's sources",
      r2 is not None and CM not in r2["seen"] and r2["seen"].get("src/server/stream.cpp", {}).get("new") == list(range(1, 8))
      and "tests/server/server_gate.cpp" in r2["seen"], r2 and r2.get("seen"))
calls.clear()
r2, g2 = tc.graph_rep({CM: {}}, dict(BW, cmake_lists={CM: ["src/a.cpp"]}), {}, "R", "a1", "b2", "W", gm2, runner=fake_runner, analyze_fn=fake_analyze)
check("graph_rep: list-only CMake edit -> no graph", r2 is None and not calls, calls)
r2, g2 = tc.graph_rep({CM: {}, "build.bat": {}}, BW, {}, "R", "a1", "b2", "W", gm2, runner=fake_runner, analyze_fn=fake_analyze)
check("graph_rep: build.bat changed too -> no graph (full anyway)", r2 is None and not calls, calls)
r2, g2 = tc.graph_rep({"src/a.cpp": {}}, BW, {}, "R", "a1", "b2", "W", gm2, runner=fake_runner, analyze_fn=fake_analyze)
check("graph_rep: no CMake file -> no graph", r2 is None and not calls, calls)
# ---------------------------------------------------------------------------------------------------------------------
# arch025-n11 (PLAN N11): a core module with GPU mutation evidence plans its own GPU step set (tools/gates/core_steps.json)
CAT = {"fa-1": "forward.attn", "fa-2": "forward.attn", "fa-3": "forward.attn", "kv-1": "model.kv", "kv-2": "model.kv",
       "cd-1": "contract.device", "cd-2": "contract.device", "ff-1": "forward.ffn", "ff-2": "forward.ffn"}


def ev(i, card, verdict="KILLED", by="st_swift_precise", date="2026-10-29T10:00"):
    return {"id": i, "card": card, "verdict": verdict, "killed_by": by if verdict == "KILLED" else "", "date": date, "source": "t"}


def both(i, **kw):
    return [ev(i, "r9700", **kw), ev(i, "8060s", **kw)]


def doc(rows):
    return {"always": ["golden_swift", "golden_moe"], "min_mutants": 2, "evidence": rows}


CAT2 = {k: v for k, v in CAT.items() if k != "fa-3"}  # forward.attn with exactly two mutants
ok, why = tc.core_step_sets(doc(both("fa-1") + both("fa-2", by="seq_swift_precise")), CAT2)
check("n11: 2 mutants killed on both cards -> set", "forward.attn" in ok, why)
check("n11: set = killers + golden on both cards", ok.get("forward.attn", {}).get("r9700") == ["golden_moe", "golden_swift", "seq_swift_precise", "st_swift_precise"]
      and ok["forward.attn"]["8060s"] == ok["forward.attn"]["r9700"], ok)
check("n11: the set names its mutants", sorted(ok.get("forward.attn", {}).get("mutants", [])) == ["fa-1", "fa-2"], ok)
ok, why = tc.core_step_sets(doc(both("fa-1") + both("fa-2")), CAT)
check("n11: a catalog mutant never run on the GPU -> full", "forward.attn" not in ok and "fa-3" in why.get("forward.attn", ""), why)
ok, why = tc.core_step_sets(doc(both("fa-1") + [ev("fa-2", "r9700")]), CAT2)
check("n11: one card without evidence -> full", "forward.attn" not in ok and "8060s" in why.get("forward.attn", ""), why)
ok, why = tc.core_step_sets(doc(both("fa-1") + [ev("fa-2", "r9700"), ev("fa-2", "8060s", verdict="SURVIVED")]), CAT2)
check("n11: a GPU survivor -> full", "forward.attn" not in ok and "SURVIVED" in why.get("forward.attn", ""), why)
for v in ("CAP", "GATE-ERROR", "NOT-RUN", "DEVICE-MISMATCH", "CONTROL-FAIL"):
    ok, why = tc.core_step_sets(doc(both("fa-1") + [ev("fa-2", "r9700"), ev("fa-2", "8060s", verdict=v)]), CAT2)
    check(f"n11: verdict {v} is not a kill -> full", "forward.attn" not in ok, why)
ok, why = tc.core_step_sets(doc(both("ff-1")), {"ff-1": "forward.ffn"})
check("n11: one mutant is not enough (min 2)", "forward.ffn" not in ok and "1 killed" in why.get("forward.ffn", ""), why)
ok, why = tc.core_step_sets(doc(both("ff-1") + both("ff-2", verdict="BUILD-FAIL")), {"ff-1": "forward.ffn", "ff-2": "forward.ffn"})
check("n11: BUILD-FAIL is caught but does not count toward 2", "forward.ffn" not in ok, why)
ok, why = tc.core_step_sets(doc(both("fa-1") + both("fa-2") + both("fa-3", verdict="BUILD-FAIL")), CAT)
check("n11: BUILD-FAIL + 2 kills -> set", "forward.attn" in ok, why)
ok, why = tc.core_step_sets(doc(both("fa-1") + both("fa-2", verdict="SURVIVED", date="2026-10-28T10:00")
                                + both("fa-2", date="2026-10-29T12:00")), CAT2)
check("n11: the newest row per mutant and card counts", "forward.attn" in ok, why)
ok, why = tc.core_step_sets(doc(both("fa-1") + both("fa-2", date="2026-10-28T10:00")
                                + both("fa-2", verdict="SURVIVED", date="2026-10-29T12:00")), CAT2)
check("n11: a newer survivor overrides an older kill", "forward.attn" not in ok, why)
ok, why = tc.core_step_sets(doc(both("cd-1") + both("cd-2")), CAT)
check("n11: contract modules always full", "contract.device" not in ok and "contract" in why.get("contract.device", ""), why)
ok, why = tc.core_step_sets(doc([ev("fa-1", "r9700", by="")] + [ev("fa-1", "8060s")] + both("fa-2")), CAT2)
check("n11: KILLED without a killing step is not evidence", "forward.attn" not in ok, why)
ok, why = tc.core_step_sets(doc([]), CAT)
check("n11: no evidence -> every core module full with a reason", not ok and all(m in why for m in tc.CORE), sorted(set(tc.CORE) - set(why)))
# plans
FA = "src/model/qwen35/forward/attn.cpp"
SETS = {"forward.attn": {"r9700": ["golden_moe", "golden_swift", "seq_swift_precise"], "8060s": ["golden_moe", "golden_swift", "kt_full"],
                         "mutants": ["fa-1", "fa-2"], "proof": "fa-1, fa-2 killed on both cards"}}
p = plan([FA], rep({"forward.attn": [FA]}, "release_gate.ps1 -Module forward.attn"), core_sets=SETS)
check("n11 plan: evidenced core module -> module level", p["level"] == "module" and not p["full_reasons"], p["full_reasons"])
check("n11 plan: the evidence set per card (not modules.json steps)", p["gpu_steps"]["r9700"] == ["golden_swift", "golden_moe", "seq_swift_precise"]
      and p["gpu_steps"]["8060s"] == ["kt_full", "golden_swift", "golden_moe"], p["gpu_steps"])
check("n11 plan: why names the evidence", p["gpu_why"].get("r9700:seq_swift_precise", "").startswith("core module forward.attn: GPU mutation evidence"), p["gpu_why"])
check("n11 plan: core set recorded", "forward.attn" in (p.get("core_sets") or {}), p.get("core_sets"))
p = plan([FA], rep({"forward.attn": [FA]}, "release_gate.ps1 -Module forward.attn", steps=["srv_balance"]), core_sets=SETS)
check("n11 plan: impact map steps still added", "srv_balance" in p["gpu_steps"]["r9700"], p["gpu_steps"])
p = plan([FA], rep({"forward.attn": [FA]}, "release_gate.ps1 -Module forward.attn"), core_sets={})
check("n11 plan: no evidence -> full (unchanged)", p["level"] == "full" and any("core module forward.attn" in r for r in p["full_reasons"]), p["full_reasons"])
p = plan([FA, "src/model/kv.cpp"], rep({"forward.attn": [FA], "model.kv": ["src/model/kv.cpp"]}, "x"), core_sets=SETS)
check("n11 plan: a second core module without evidence -> full", p["level"] == "full" and any("model.kv" in r for r in p["full_reasons"])
      and not any("forward.attn" in r for r in p["full_reasons"] if r.startswith("core module")), p["full_reasons"])
p = plan(["src/model/forward/attn.cpp"], rep({}, "full gate or the release steps above: 1 file(s) in no module (src/model/forward/attn.cpp)"), core_sets=SETS)
check("n11 plan: core path in no module stays full", p["level"] == "full" and any(r.startswith("core file src/model/forward/attn.cpp") for r in p["full_reasons"]), p["full_reasons"])
p = plan([FA, "kernels/attn.hip"], rep({"forward.attn": [FA]}, "x"), core_sets=SETS)
check("n11 plan: a kernel file with it stays full", p["level"] == "full" and any(r.startswith("core file kernels/attn.hip") for r in p["full_reasons"]), p["full_reasons"])
p = plan(["src/device/backend.h", FA], rep({"contract.device": ["src/device/backend.h"], "forward.attn": [FA]}, "full gate, both GPUs (contract module contract.device)"),
         core_sets=dict(SETS, **{"contract.device": SETS["forward.attn"]}))
check("n11 plan: contract module full even if a set is passed for it", p["level"] == "full" and any("contract.device" in r for r in p["full_reasons"]), p["full_reasons"])
p = plan(["src/server/stream.h"], rep({"protocol": ["src/server/stream.h"]}, "release_gate.ps1 -Module protocol,forward.attn", callers={"protocol": ["forward.attn"]}), core_sets=SETS)
check("n11 plan: evidenced core caller -> its set", p["level"] == "module" and "seq_swift_precise" in p["gpu_steps"]["r9700"]
      and "srv_balance" in p["gpu_steps"]["r9700"], (p["level"], p["full_reasons"], p["gpu_steps"]))
# modules.json full: true modules that are not contracts (model.core, forward.state, ...) can earn a set the same way;
# contract.* and kernels.abi never
MODS2 = dict(MODS, **{"model.core": {"paths": ["include/whirl/model.h"], "full": True}, "contract.server": {"paths": ["src/server/backend.h"], "full": True},
                      "ops.matmul": {"paths": ["src/ops/matmul.cpp"], "steps": ["srv_fast"]}})
CAT3 = dict(CAT, **{"mc-1": "model.core", "mc-2": "model.core", "om-1": "ops.matmul", "om-2": "ops.matmul", "ka-1": "kernels.abi", "ka-2": "kernels.abi"})
ok, why = tc.core_step_sets(doc(both("mc-1") + both("mc-2") + both("om-1") + both("om-2") + both("ka-1") + both("ka-2")), CAT3,
                            modules=tc.evidence_modules(MODS2))
check("n11: a full:true module (model.core) can earn a set", "model.core" in ok, why)
check("n11: a non-core module owning a core path (ops.matmul) can earn a set", "ops.matmul" in ok, why)
check("n11: kernels.abi is a contract: never", "kernels.abi" not in ok, why)
check("n11: evidence_modules lists CORE + full:true + mutant modules, no contracts",
      {"forward.attn", "model.core", "handles"} <= set(tc.evidence_modules(MODS2)) and not any(m.startswith("contract.") or m == "kernels.abi" for m in tc.evidence_modules(MODS2)),
      tc.evidence_modules(MODS2))
SET_MC = {"r9700": ["golden_moe", "golden_swift", "srv_balance"], "8060s": ["golden_moe", "golden_swift"], "mutants": ["mc-1", "mc-2"], "proof": "p"}
MH = "include/whirl/model.h"


def plan2(files, r, **kw):
    return tc.make_plan("t", "x", {f: {} for f in files}, r, MODS2, {"includers": {}}, GATES, TIMES, **kw)


p = plan2([MH], rep({"model.core": [MH]}, "full gate, both GPUs (contract module model.core)"), core_sets={"model.core": SET_MC})
check("n11 plan: evidenced full:true module -> its set, impact's full gate overridden", p["level"] == "module" and not p["full_reasons"]
      and p["gpu_steps"]["r9700"] == ["golden_swift", "golden_moe", "srv_balance"] and p["gpu_steps"]["8060s"] == ["golden_swift", "golden_moe"], (p["full_reasons"], p["gpu_steps"]))
p = plan2([MH], rep({"model.core": [MH]}, "full gate, both GPUs (contract module model.core)"), core_sets={})
check("n11 plan: full:true module without evidence -> full", p["level"] == "full" and any(r.startswith("impact.py: full gate") for r in p["full_reasons"]), p["full_reasons"])
p = plan2([MH, "src/server/backend.h"], rep({"model.core": [MH], "contract.server": ["src/server/backend.h"]}, "full gate, both GPUs (contract module contract.server, model.core)"),
          core_sets={"model.core": SET_MC, "contract.server": SET_MC})
check("n11 plan: with a contract module the impact full gate stays", p["level"] == "full" and any(r.startswith("impact.py: full gate") for r in p["full_reasons"]), p["full_reasons"])
p = plan2([MH, "src/x.cpp"], rep({"model.core": [MH]}, "full gate, both GPUs (contract module model.core)"), core_sets={"model.core": SET_MC})
check("n11 plan: a file in no module keeps the full gate", p["level"] == "full", p["full_reasons"])
OM = "src/ops/matmul.cpp"
p = plan2([OM], rep({"ops.matmul": [OM]}, "release_gate.ps1 -Module ops.matmul"), core_sets={"ops.matmul": dict(SET_MC, r9700=["golden_swift", "kt_full"])})
check("n11 plan: evidenced owner of a core path -> module level, modules.json steps + evidence set", p["level"] == "module"
      and set(p["gpu_steps"]["r9700"]) == {"srv_fast", "golden_swift", "kt_full"}, (p["full_reasons"], p["gpu_steps"]))
p = plan2([OM], rep({"ops.matmul": [OM]}, "release_gate.ps1 -Module ops.matmul"), core_sets={})
check("n11 plan: owner of a core path without evidence -> full", p["level"] == "full" and any(r.startswith("core file src/ops/matmul.cpp") for r in p["full_reasons"]), p["full_reasons"])
# a bug class named in "require" (e.g. K8-2 Model member order) must be represented by a KILLED mutant on both cards
R = dict(doc(both("fa-1") + both("fa-2")), require={"forward.attn": {"fa-2": "K-x class", "fa-9": "K-y class"}})
ok, why = tc.core_step_sets(R, CAT2)
check("n11: a required mutant missing from the catalog -> full", "forward.attn" not in ok and "fa-9" in why.get("forward.attn", "") and "K-y" in why["forward.attn"], why)
R = dict(doc(both("fa-1") + both("fa-2") + both("fa-3", verdict="BUILD-FAIL")), require={"forward.attn": {"fa-3": "K-z class"}})
ok, why = tc.core_step_sets(R, CAT)
check("n11: a required mutant must be KILLED by a GPU step (BUILD-FAIL is not enough)", "forward.attn" not in ok and "fa-3" in why.get("forward.attn", ""), why)
R = dict(doc(both("fa-1") + both("fa-2")), require={"forward.attn": {"fa-2": "K-x class"}})
ok, why = tc.core_step_sets(R, CAT2)
check("n11: required mutant killed -> set", "forward.attn" in ok, why)
real_req = tc.load_core_doc(tc.REPO).get("require") or {}
check("n11 shipped: K8-2 pins model.core (~Model release order; arch025-n11b: x12-f8-model-dtor is a model.core file's mutant)",
      "x12-f8-model-dtor" in (real_req.get("model.core") or {}), real_req)
# kill records -> evidence rows (gpumut.ps1 kill_<card>.tsv)
with tempfile.TemporaryDirectory() as td:
    kt = os.path.join(td, "kill_r9700.tsv")
    open(kt, "w", encoding="utf-8").write("id\tmodule\tcard\tverdict\tkilled_by\tsteps_run\tfailed_steps\tminutes\n"
                                          "fa-1\tforward.attn\tr9700\tKILLED\tst_swift_precise\tst_swift_precise\tst_swift_precise\t4.2\n"
                                          "fa-2\tforward.attn\tr9700\tSURVIVED\t\tst_swift_precise,golden_swift\t\t9.0\n"
                                          "zz-9\tforward.attn\tr9700\tKILLED\tgolden_swift\tgolden_swift\tgolden_swift\t3\n")
    rows, notes = tc.kill_rows([kt], CAT, "2026-10-29T13:00", "pilot")
    check("n11 kill rows: one row per catalog mutant", [(r["id"], r["card"], r["verdict"], r["killed_by"]) for r in rows]
          == [("fa-1", "r9700", "KILLED", "st_swift_precise"), ("fa-2", "r9700", "SURVIVED", "")], rows)
    check("n11 kill rows: id not in the catalog reported", any("zz-9" in x for x in notes), notes)
    check("n11 kill rows: date / source / file kept", rows[0]["date"] == "2026-10-29T13:00" and rows[0]["source"].startswith("pilot") and "kill_r9700.tsv" in rows[0]["source"], rows[0])
# the shipped data file: loads, every row names a catalog mutant and a card, and no core module claims a set it cannot prove
real = tc.load_core_doc(tc.REPO)
cat_real = tc.load_catalog(tc.REPO)
check("n11 shipped core_steps.json loads", isinstance(real.get("evidence"), list) and real.get("min_mutants", 0) >= 2, real.keys())
check("n11 shipped rows name catalog mutants and cards", all(r["id"] in cat_real and r["card"] in tc.DEVICES for r in real["evidence"]),
      [r["id"] for r in real["evidence"] if r["id"] not in cat_real])
ok_real, why_real = tc.core_step_sets(real, cat_real)
check("n11 shipped: every core module has a set or a reason", all((m in ok_real) != (m in why_real) for m in tc.CORE), (ok_real, why_real))
# arch025-n11b: (a) a mutant valid on one card only (catalog "cards": a kernel file of one code object) is evidence on
# that card only: it is not "not run" / a survivor on the other card, and the >= 2 kills are counted per card over the
# mutants of that card (llama.cpp test-backend-ops NOT_SUPPORTED, progress/arch025-n11b.PRIOR_ART.md)
CATK = {"kc-a": "kernels.common", "kc-b": "kernels.common", "kc-r": "kernels.common", "kc-s": "kernels.common"}
METAK = {"kc-r": {"cards": ["r9700"]}, "kc-s": {"cards": ["8060s"]}}
ok, why = tc.core_step_sets(doc(both("kc-a") + both("kc-b") + [ev("kc-r", "r9700", by="kt_full"), ev("kc-s", "8060s", by="kt_full")]), CATK,
                            modules=["kernels.common"], meta=METAK)
check("n11b cards: a one-card mutant run on its card only -> set", "kernels.common" in ok, why)
check("n11b cards: its killer joins that card's set only", "kt_full" in ok.get("kernels.common", {}).get("r9700", [])
      and "kt_full" in ok.get("kernels.common", {}).get("8060s", []), ok)
ok, why = tc.core_step_sets(doc(both("kc-a") + both("kc-b") + [ev("kc-r", "r9700", by="kt_full"), ev("kc-s", "8060s", by="kt_full")]), CATK,
                            modules=["kernels.common"])
check("n11b cards: without the cards field the same rows stay full (both-cards rule)", "kernels.common" not in ok, why)
ok, why = tc.core_step_sets(doc([ev("kc-r", "r9700", by="kt_full"), ev("kc-s", "8060s", by="kt_full")] + both("kc-a")), CATK, modules=["kernels.common"],
                            meta=dict(METAK, **{"kc-b": {"cards": ["r9700"]}}))
check("n11b cards: kills are counted per card over that card's mutants (8060s 2: kc-a + kc-s; r9700 kc-b not run)",
      "kernels.common" not in ok and "kc-b not run on r9700" in why.get("kernels.common", ""), why)
ok, why = tc.core_step_sets(doc(both("kc-a") + both("kc-b") + [ev("kc-r", "r9700", by="kt_full")]), CATK, modules=["kernels.common"], meta=METAK)
check("n11b cards: a one-card mutant not run on its card -> full", "kernels.common" not in ok and "kc-s not run on 8060s" in why.get("kernels.common", ""), why)
ok, why = tc.core_step_sets(doc(both("kc-a") + both("kc-b") + [ev("kc-r", "r9700", by="kt_full"), ev("kc-s", "8060s", by="kt_full"),
                                                                 ev("kc-r", "8060s", verdict="SURVIVED")]), CATK, modules=["kernels.common"], meta=METAK)
check("n11b cards: a row on a card the mutant does not apply to is ignored", "kernels.common" in ok, why)
# (b) an equivalent mutant (catalog "equivalent": the argument) is not evidence either way: left out of the rule
CATE = {"dp-1": "device.profile", "dp-2": "device.profile", "dp-eq": "device.profile"}
ok, why = tc.core_step_sets(doc(both("dp-1") + both("dp-2")), CATE, modules=["device.profile"], meta={"dp-eq": {"equivalent": "static_assert deleted: no code"}})
check("n11b equivalent: an argued equivalent mutant does not block the module", "device.profile" in ok and "dp-eq" not in ok["device.profile"]["mutants"], why)
ok, why = tc.core_step_sets(doc(both("dp-1") + both("dp-2")), CATE, modules=["device.profile"])
check("n11b equivalent: without the field it blocks (not run)", "device.profile" not in ok and "dp-eq" in why.get("device.profile", ""), why)
R = dict(doc(both("dp-1") + both("dp-2")), require={"device.profile": {"dp-eq": "K-q"}})
ok, why = tc.core_step_sets(R, CATE, modules=["device.profile"], meta={"dp-eq": {"equivalent": "x"}})
check("n11b equivalent: a required mutant cannot be equivalent", "device.profile" not in ok and "dp-eq" in why.get("device.profile", ""), why)
with tempfile.TemporaryDirectory() as td:
    kt = os.path.join(td, "kill_8060s.tsv")
    open(kt, "w", encoding="utf-8").write("id\tmodule\tcard\tverdict\tkilled_by\tsteps_run\tfailed_steps\tminutes\n"
                                          "kc-r\tkernels.common\t8060s\tSURVIVED\t\tkt_full\t\t9\n"
                                          "kc-s\tkernels.common\t8060s\tKILLED\tkt_full\tkt_full\tkt_full\t9\n")
    rows, notes = tc.kill_rows([kt], CATK, "2026-10-29T18:00", "t", meta=METAK)
    check("n11b kill rows: a row on a card the mutant does not apply to is skipped and reported",
          [r["id"] for r in rows] == ["kc-s"] and any("kc-r" in x and "r9700" in x for x in notes), (rows, notes))
# (c) device.null (CPU null backend, NO_GPU_MODULES) is not core just because its files are under src/device/*
NB = "src/device/null_backend.cpp"
MODSN = dict(MODS, **{"device.null": {"paths": ["src/device/null_backend.*"], "steps": ["cpu_tests"]},
                      "contract.device": {"paths": ["src/device/backend.h"], "full": True}})
p = tc.make_plan("t", "x", {NB: {}}, rep({"device.null": [NB]}, "release_gate.ps1 -Module device.null"), MODSN, {"includers": {}}, GATES, TIMES, core_sets={})
check("n11b device.null: a core path owned only by NO_GPU_MODULES -> CPU", p["level"].startswith("cpu") and not p["full_reasons"], (p["level"], p["full_reasons"]))
p = tc.make_plan("t", "x", {"src/device/zz.cpp": {}}, rep({}, "full gate or the release steps above: 1 file(s) in no module (src/device/zz.cpp)"), MODSN,
                 {"includers": {}}, GATES, TIMES, core_sets={})
check("n11b device.null: a src/device file in no module stays full", p["level"] == "full" and any(r.startswith("core file src/device/zz.cpp") for r in p["full_reasons"]), p["full_reasons"])
p = tc.make_plan("t", "x", {NB: {}, "src/device/backend.h": {}}, rep({"device.null": [NB], "contract.device": ["src/device/backend.h"]}, "full gate, both GPUs (contract module contract.device)"),
                 MODSN, {"includers": {}}, GATES, TIMES, core_sets={})
check("n11b device.null: with a GPU device file it stays full", p["level"] == "full" and any("src/device/backend.h" in r for r in p["full_reasons"]), p["full_reasons"])
p = tc.make_plan("t", "x", {"cmake/device.cmake": {}}, dict(rep({}, "x"), cmake_lists={"cmake/device.cmake": [NB]}), MODSN, {"includers": {}}, GATES, TIMES, core_sets={})
check("n11b device.null: a CMake list naming only its source is not a core change", not any("core file" in r for r in p["full_reasons"]), p["full_reasons"])
# (d) the shipped catalog: labels, cards, the missing mutants, the declared equivalent
meta_real = tc.load_catalog_meta(tc.REPO)
check("n11b shipped: hiprt-* name hip.runtime", all(cat_real.get(i) == "hip.runtime" for i in ("hiprt-slot-wrong-symbol", "hiprt-order-reversed",
      "hiprt-fallback-on-broken-local", "hiprt-missing-symbol-ignored")), [cat_real.get(i) for i in cat_real if i.startswith("hiprt-")])
check("n11b shipped: kernels.common one-card mutants carry cards", meta_real.get("n8-kt-common-wq8-gfx1201", {}).get("cards") == ["r9700"]
      and meta_real.get("n8-kt-common-wq8-gfx1151", {}).get("cards") == ["8060s"], (meta_real.get("n8-kt-common-wq8-gfx1201"), meta_real.get("n8-kt-common-wq8-gfx1151")))
check("n11b shipped: k8-2-model-member-order is a model.core mutant", cat_real.get("k8-2-model-member-order") == "model.core", cat_real.get("k8-2-model-member-order"))
for mname in ("forward.state", "model.types"):
    k = [i for i, mm in cat_real.items() if mm == mname and not (meta_real.get(i) or {}).get("equivalent")]
    check(f"n11b shipped: {mname} has >= 2 non-equivalent mutants", len(k) >= 2, k)
check("n11b shipped: m6-uma-assert-drop is a declared, argued equivalent", len((meta_real.get("m6-uma-assert-drop") or {}).get("equivalent") or "") >= 40,
      meta_real.get("m6-uma-assert-drop"))
for mname in ("model.core", "hip.runtime", "kernels.common", "forward.state", "model.types"):
    for card in tc.DEVICES:
        k = [i for i, mm in cat_real.items() if mm == mname and not (meta_real.get(i) or {}).get("equivalent") and card in tc.mutant_cards(meta_real, i)]
        check(f"n11b shipped: {mname} has >= 2 killable mutants on {card} (else no GPU run can earn it a set)", len(k) >= 2, k)
bad_req = [(mm, i) for mm, d in (real.get("require") or {}).items() for i in d
           if cat_real.get(i) != mm or (meta_real.get(i) or {}).get("equivalent")]
check("n11b shipped: every required mutant is in the catalog under its module and killable (not equivalent)", not bad_req, bad_req)
check("n11b shipped: the K8-2 class of model.core is required through ~Model's explicit release order (x12-f8-model-dtor, src/model/loader.cpp = model.core)",
      "x12-f8-model-dtor" in ((real.get("require") or {}).get("model.core") or {}) and cat_real.get("x12-f8-model-dtor") == "model.core",
      (real.get("require"), cat_real.get("x12-f8-model-dtor")))
ok_real, why_real = tc.core_step_sets(real, cat_real, tc.evidence_modules(RMODS), meta=meta_real)
check("n11b shipped: device.profile is blocked only by missing GPU runs, not by m6-uma-assert-drop", "m6-uma-assert-drop" not in why_real.get("device.profile", "x"), why_real.get("device.profile"))
# arch025-n11c: a mutant counts under the module that OWNS the file it mutates (modules.json paths); the old label is fault_area
_cat_rows = json.load(open(os.path.join(tc.REPO, "tools", "impact", "mutations.json"), encoding="utf-8"))["mutations"]
_own = lambda f: [k for k, d in RMODS.items() if tc.matches(f, (d or {}).get("paths") or [])]
bad_own = [(x["id"], x["module"], _own(x["file"])) for x in _cat_rows if _own(x["file"]) and x["module"] not in _own(x["file"])]
check("n11c shipped: every catalog mutant is counted under the module that owns its mutated file", not bad_own, bad_own[:5])
check("n11c shipped: m1-* (loader.cpp) count under model.core and keep model.weights as fault_area",
      all(x["module"] == "model.core" and x.get("fault_area") == "model.weights" for x in _cat_rows if x["id"].startswith("m1-") and x["file"] == "src/model/loader.cpp")
      and any(x["id"].startswith("m1-") and x["file"] == "src/model/loader.cpp" for x in _cat_rows), "m1-*")
import subprocess  # noqa: E402
rr = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_change.py"), "--core-steps-report"],
                    capture_output=True, text=True)
lines = {ln.split()[0]: ln for ln in rr.stdout.splitlines() if ln.split()}
check("n11b report: modules with catalog mutants are listed (hip.runtime, kernels.common)", "hip.runtime" in lines and "kernels.common" in lines, rr.stdout[-800:])
check("n11b report: the report honours cards (kernels.common: n8-kt-common-wq8-gfx1201 is not 'not run on 8060s')",
      "n8-kt-common-wq8-gfx1201 not run on 8060s" not in lines.get("kernels.common", ""), lines.get("kernels.common"))
check("n11b report: the report honours equivalent (device.profile: m6-uma-assert-drop not listed)", "m6-uma-assert-drop" not in lines.get("device.profile", "x"), lines.get("device.profile"))
print(f"test_change_test: {n - fails}/{n} passed")
sys.exit(1 if fails else 0)
