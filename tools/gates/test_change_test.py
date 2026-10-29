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
    p = tc.make_plan("t", "x", {f: {}}, r, RMODS, {"includers": {}}, GATES, TIMES)
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
print(f"test_change_test: {n - fails}/{n} passed")
sys.exit(1 if fails else 0)
