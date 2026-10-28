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
print(f"test_change_test: {n - fails}/{n} passed")
sys.exit(1 if fails else 0)
