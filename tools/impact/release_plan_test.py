# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/impact/release_plan.py (measure-tools P7). Run: python -I tools/impact/release_plan_test.py
CPU only: synthetic impact output, step list and modules.json standards; plus the real tools/gates/modules.json
standards against the real release_gate.ps1 -List (skipped when pwsh is missing)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import release_plan as RP  # noqa: E402

FAILS = []


def expect(name, cond, detail=""):
    print(("ok    " if cond else "FAIL  ") + name)
    if not cond:
        FAILS.append(name)
        if detail:
            print("      " + str(detail)[:2000])


STD = {"kinds": {"exit_rule": {"desc": "exit code + output rule", "compare": "rule"},
                 "golden_exact": {"desc": "golden hashes identical", "compare": "exact"},
                 "work_trace": {"desc": "launch sequence identical", "compare": "exact"}},
       "steps": [{"match": "cpu_*", "kinds": ["exit_rule"], "host_only": True},
                 {"match": "golden_*", "kinds": ["golden_exact", "work_trace"]},
                 {"match": "srv_*", "kinds": ["exit_rule", "work_trace"]},
                 {"match": "seq_*", "kinds": ["exit_rule"]}]}
GATE = ["cpu_tests", "cpu_server", "golden_swift", "golden_moe", "srv_balance", "srv_fs_swift", "seq_moe_precise", "seq_moe_balance"]
IMP = {"range 1111111..2222222 (from merge-base 1111111aa)": {
    "release_gate_steps": ["cpu_server", "golden_swift", "seq_moe_<mode>", "dfl_<mode> (R9700, Q4_K_M compatibility only)"],
    "paths": ["server.loop"], "modules": {"protocol": ["src/server/http.cpp"]}}}


def plan(**kw):
    a = dict(impact=IMP, gate_steps=GATE, standards=STD, arch="gfx1151")
    a.update(kw)
    return RP.plan(**a)


def main():
    p = plan()
    by = {r["step"]: r for r in p["steps"]}
    expect("every gate step gets one line", [r["step"] for r in p["steps"]] == GATE, p["steps"])
    expect("selected steps RUN, with the range in the reason", by["golden_swift"]["action"] == "RUN" and "1111111..2222222" in by["golden_swift"]["why"], by["golden_swift"])
    expect("<mode> expanded: seq_moe_precise / balance RUN", by["seq_moe_precise"]["action"] == "RUN" and by["seq_moe_balance"]["action"] == "RUN")
    expect("not selected -> SKIP with the reason", by["srv_balance"]["action"] == "SKIP" and "not selected" in by["srv_balance"]["why"], by["srv_balance"])
    expect("each line names the step's standard (kinds)", by["golden_swift"]["kinds"] == ["golden_exact", "work_trace"] and by["cpu_server"]["kinds"] == ["exit_rule"])
    expect("required = the RUN steps", p["required"] == sorted(["cpu_server", "golden_swift", "seq_moe_precise", "seq_moe_balance"]), p["required"])
    expect("range is reported", p["range"] == "1111111..2222222")

    # host-only steps: on the second GPU they are SKIP when they already ran on the other GPU
    p = plan(host_done=["cpu_server"])
    by = {r["step"]: r for r in p["steps"]}
    expect("host-only step done on the other GPU -> SKIP (other_arch)", by["cpu_server"]["action"] == "SKIP" and by["cpu_server"]["exempt"] == "other_arch", by["cpu_server"])
    p = plan(host_done=["srv_balance"])
    expect("host_done ignored for a GPU step", {r["step"]: r for r in p["steps"]}["srv_balance"]["action"] == "SKIP" and
           not {r["step"]: r for r in p["steps"]}["srv_balance"].get("exempt"))

    # verified on the identical exe (stored evidence) -> SKIP with the evidence
    ver = [{"step": "golden_swift", "arch": "gfx1151", "exe_sha256": "a" * 64, "summary": "S16_s8_summary.txt"}]
    p = plan(verified=ver, exe_sha="a" * 64)
    r = {x["step"]: x for x in p["steps"]}["golden_swift"]
    expect("selected but verified on the identical exe -> SKIP with the summary", r["action"] == "SKIP" and "S16_s8_summary.txt" in r["why"] and r["exempt"] == "verified", r)
    p = plan(verified=ver, exe_sha="b" * 64)
    expect("verified on another exe -> still RUN", {x["step"]: x for x in p["steps"]}["golden_swift"]["action"] == "RUN")
    p = plan(verified=[dict(ver[0], arch="gfx1201")], exe_sha="a" * 64)
    expect("verified on the other GPU -> still RUN (GPU step)", {x["step"]: x for x in p["steps"]}["golden_swift"]["action"] == "RUN")

    # setup errors
    for name, kw in (("a gate step without a standard -> error", dict(gate_steps=GATE + ["kt_full"])),
                     ("a standard naming an unknown kind -> error", dict(standards=dict(STD, steps=STD["steps"] + [{"match": "kt_*", "kinds": ["nope"]}]))),
                     ("impact selects a step the gate does not have -> error", dict(impact={"range 1..2": {"release_gate_steps": ["srv_nope"]}})),
                     ("impact output with two sections -> error", dict(impact=dict(IMP, **{"worktree": {"release_gate_steps": []}}))),
                     ("impact section that is not a range -> error", dict(impact={"worktree": {"release_gate_steps": []}}))):
        try:
            plan(**kw)
            expect(name, False, "no error")
        except RP.PlanError as e:
            expect(name, True)
            print("      why: " + str(e))

    # text output of impact.py is accepted too
    txt = ("impact: map stale: rebuilding\n== range 1111111..2222222 (from merge-base 1111111aa)\n   touched  src/x.cpp: f\n"
           "   release  cpu_server, golden_swift, dfl_<mode> (R9700, Q4_K_M compatibility only), seq_moe_<mode>\n")
    t, j = RP.parse_impact_text(txt), RP.parse_impact_json(IMP)
    expect("impact text output parses like --json (range, expanded steps)", t[0] == j[0] and RP.expand_steps(t[1]) == RP.expand_steps(j[1]), t)

    # CLI: prints one line per step, --json writes the plan, exit 0
    tmp = tempfile.mkdtemp()
    try:
        ip, gp, mp, op = (os.path.join(tmp, n) for n in ("imp.json", "gate.txt", "modules.json", "plan.json"))
        json.dump(IMP, open(ip, "w", encoding="utf-8"))
        open(gp, "w", encoding="utf-8").write("".join(f"{s:<22} {'CPU' if s.startswith('cpu_') else 'GPU'}\n" for s in GATE))
        json.dump({"modules": {}, "standards": STD}, open(mp, "w", encoding="utf-8"))
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "release_plan.py"), "--impact", ip, "--gate-list", gp, "--modules", mp,
                            "--arch", "gfx1151", "--json", op], capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith(("RUN ", "SKIP "))]
        expect("CLI: exit 0, one RUN/SKIP line per gate step, plan JSON written", r.returncode == 0 and len(lines) == len(GATE) and os.path.isfile(op), r.stdout + r.stderr)
        j = json.load(open(op, encoding="utf-8")) if os.path.isfile(op) else {}
        expect("CLI JSON: format, range, arch, required", j.get("format") == "whirl-release-plan/1" and j.get("range") == "1111111..2222222" and j.get("arch") == "gfx1151"
               and j.get("required") == sorted(["cpu_server", "golden_swift", "seq_moe_precise", "seq_moe_balance"]), j)
        json.dump({"modules": {}}, open(mp, "w", encoding="utf-8"))
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "release_plan.py"), "--impact", ip, "--gate-list", gp, "--modules", mp,
                            "--arch", "gfx1151"], capture_output=True, text=True)
        expect("CLI: modules.json without standards -> exit 2", r.returncode == 2, r.stdout + r.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # the real tools/gates/modules.json standards cover every step of the real release_gate.ps1 -List
    mj = json.load(open(os.path.join(HERE, "..", "gates", "modules.json"), encoding="utf-8"))
    if shutil.which("pwsh"):
        r = subprocess.run(["pwsh", "-NoProfile", "-File", os.path.join(HERE, "..", "gates", "release_gate.ps1"), "-List"], capture_output=True, text=True)
        steps = RP.parse_gate_list(r.stdout)
        try:
            miss = [s for s in steps if RP.standard_of(mj.get("standards") or {}, s) is None]
            RP.check_standards(mj.get("standards") or {})
            expect(f"real modules.json standards cover all {len(steps)} release_gate.ps1 steps", len(steps) > 10 and not miss, miss)
        except RP.PlanError as e:
            expect("real modules.json standards are valid", False, e)
    else:
        print("skip  real release_gate.ps1 -List (no pwsh)")
    print("RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
