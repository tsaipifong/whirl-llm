# SPDX-License-Identifier: Apache-2.0
"""Release / validation plan for one GPU (measure-tools P7, ADR-0003 decision 1: "every step logs why it runs or not").

Reads the tools/impact/impact.py output of ONE commit range (--json or text), the gate's step list
(release_gate.ps1 -List, or --gate-list FILE with its output), and tools/gates/modules.json "standards" (which pass
standard / measurement kinds each gate step uses; no numbers). Prints one line per gate step:
  RUN   <step>  [kinds]  why: selected by impact.py for A..B
  SKIP  <step>  [kinds]  why: not selected by impact.py for A..B | host-only, already PASSed on the other GPU |
                              verified on the identical exe <sha8> (<summary>)
and with --json FILE writes {"format": "whirl-release-plan/1", "range", "arch", "steps": [...], "required": [RUN steps],
"exempt": [{step, covered_by, why}]}: "required" is what CURRENT.json required_steps_by_arch[arch] must hold
(release_check.py scope_check; the impact output file is its scope_proof, the range its scope_range).

  python tools/impact/release_plan.py (--impact FILE | --range A..B [--root REPO]) --arch gfx1151|gfx1201
         [--gate-list FILE] [--modules FILE] [--host-done a,b] [--verified FILE --exe-sha SHA] [--json OUT]
--host-done: host-only steps (standards entry "host_only": true) that PASSed in the other GPU's accepted gate.
--verified: JSON list [{"step", "arch", "exe_sha256", "summary"}] of PASSes already on record; a selected step whose
record names this arch and --exe-sha is SKIP (the evidence is named). Exit 0 plan printed, 2 setup error (no standards,
a gate step without a standard, an unknown kind, impact output not one range, impact selecting a step the gate lacks).
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from impact import expand_steps  # noqa: E402  (one rule for <mode> expansion: impact.py)

FORMAT = "whirl-release-plan/1"


class PlanError(Exception):
    pass


def parse_impact_json(js):
    """{title: report} of impact.py --json -> (range "A..B", raw release step list); exactly one section, a range."""
    if not isinstance(js, dict) or not js:
        raise PlanError("impact output is empty or not impact.py --json")
    if len(js) != 1:
        raise PlanError(f"impact output has {len(js)} diff sections (one impact.py --range run expected)")
    title, rep = next(iter(js.items()))
    m = re.match(r"range (\S+?)\.\.(\S+)", title)
    if not m:
        raise PlanError(f"impact section '{title[:40]}' is not a commit range")
    return f"{m.group(1)}..{m.group(2)}", list((rep or {}).get("release_gate_steps") or [])


def parse_impact_text(txt):
    secs, cur = {}, None
    for line in txt.splitlines():
        if line.startswith("== "):
            cur = line[3:]
            secs[cur] = {"release_gate_steps": []}
        elif cur is not None and re.match(r"^\s+release\s{2}", line):
            secs[cur]["release_gate_steps"] = [s.strip() for s in re.split(r",\s*(?![^()]*\))", line.split("release", 1)[1].strip()) if s.strip()]
    if not secs:
        raise PlanError("not impact.py output (no '== range A..B' section)")
    return parse_impact_json(secs)


def load_impact(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    try:
        return parse_impact_json(json.loads(txt))
    except ValueError:
        return parse_impact_text(txt)


def parse_gate_list(out):
    return [l.split()[0] for l in out.splitlines() if len(l.split()) == 2 and l.split()[1] in ("CPU", "GPU")]


def check_standards(std):
    kinds = std.get("kinds") or {}
    if not kinds or not std.get("steps"):
        raise PlanError("modules.json has no standards (kinds + steps)")
    for e in std["steps"]:
        if not e.get("match") or not e.get("kinds"):
            raise PlanError(f"standards step entry {e} needs match and kinds")
        bad = [k for k in e["kinds"] if k not in kinds]
        if bad:
            raise PlanError(f"standards step entry {e['match']}: unknown kind(s) {bad}")


def standard_of(std, step):
    """first standards.steps entry whose match (fnmatch) fits the step, else None"""
    for e in std.get("steps") or []:
        if fnmatch.fnmatchcase(step, e.get("match", "")):
            return e
    return None


def plan(impact, gate_steps, standards, arch, host_done=(), verified=(), exe_sha=None):
    """impact: impact.py --json dict (or a (range, raw steps) tuple); returns the plan dict (raises PlanError)."""
    check_standards(standards)
    rng, raw = impact if isinstance(impact, tuple) else parse_impact_json(impact)
    sel = expand_steps(raw)
    gate = list(gate_steps)
    missing_std = [s for s in gate if standard_of(standards, s) is None]
    if missing_std:
        raise PlanError(f"gate steps without a standard in modules.json: {missing_std}")
    unknown = sorted(sel - set(gate))
    if unknown:
        raise PlanError(f"impact.py selected steps the gate does not have: {unknown}")
    ver = {}
    for v in verified or []:
        if v.get("arch") == arch and exe_sha and v.get("exe_sha256") == exe_sha and v.get("step"):
            ver.setdefault(v["step"], v)
    rows, req, exempt = [], [], []
    for s in gate:
        e = standard_of(standards, s)
        r = {"step": s, "kinds": list(e["kinds"]), "host_only": bool(e.get("host_only"))}
        if s not in sel:
            r.update(action="SKIP", why=f"not selected by impact.py for {rng}")
        elif r["host_only"] and s in set(host_done or ()):
            r.update(action="SKIP", exempt="other_arch", why="host-only step, already PASSed in the other GPU's accepted gate")
        elif s in ver:
            r.update(action="SKIP", exempt="verified",
                     why=f"selected for {rng}, verified on the identical exe {exe_sha[:8]} ({os.path.basename(str(ver[s].get('summary')))})")
        else:
            r.update(action="RUN", why=f"selected by impact.py for {rng}")
        if r["action"] == "RUN":
            req.append(s)
        elif r.get("exempt"):
            exempt.append({"step": s, "covered_by": r["exempt"], "why": r["why"]})
        rows.append(r)
    return {"format": FORMAT, "range": rng, "arch": arch, "steps": rows, "required": sorted(req), "exempt": exempt}


def main():
    ap = argparse.ArgumentParser(description="per-step release plan: why each gate step runs or not")
    ap.add_argument("--impact")
    ap.add_argument("--range")
    ap.add_argument("--root", default=os.path.normpath(os.path.join(HERE, "..", "..")))
    ap.add_argument("--arch", required=True, choices=["gfx1151", "gfx1201"])
    ap.add_argument("--gate-list")
    ap.add_argument("--modules", default=os.path.join(HERE, "..", "gates", "modules.json"))
    ap.add_argument("--host-done", default="")
    ap.add_argument("--verified")
    ap.add_argument("--exe-sha")
    ap.add_argument("--json")
    a = ap.parse_args()
    try:
        if a.impact:
            imp = load_impact(a.impact)
        elif a.range:
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "impact.py"), "--range", a.range, "--json", "--root", a.root],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode != 0:
                raise PlanError(f"impact.py --range {a.range} failed: {p.stderr.strip()[-300:]}")
            imp = parse_impact_json(json.loads(p.stdout))
        else:
            raise PlanError("give --impact FILE or --range A..B")
        if a.gate_list:
            gate = parse_gate_list(open(a.gate_list, encoding="utf-8", errors="replace").read())
        else:
            p = subprocess.run(["pwsh", "-NoProfile", "-File", os.path.join(HERE, "..", "gates", "release_gate.ps1"), "-List"],
                               capture_output=True, text=True)
            gate = parse_gate_list(p.stdout)
        if len(gate) < 1:
            raise PlanError("no gate steps (release_gate.ps1 -List / --gate-list)")
        std = (json.load(open(a.modules, encoding="utf-8")) or {}).get("standards") or {}
        ver = json.load(open(a.verified, encoding="utf-8")) if a.verified else []
        pl = plan(imp, gate, std, a.arch, [s for s in a.host_done.split(",") if s], ver, a.exe_sha)
    except (PlanError, OSError, ValueError) as e:
        print(f"release plan: {e}")
        return 2
    print(f"release plan {pl['arch']} for {pl['range']} (impact.py selection; standards from {os.path.basename(a.modules)})")
    for r in pl["steps"]:
        print(f"{r['action']:<5} {r['step']:<22} [{','.join(r['kinds'])}]  why: {r['why']}")
    print(f"release plan: {len(pl['required'])} RUN, {len(pl['steps']) - len(pl['required'])} SKIP ({len(pl['exempt'])} by evidence)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(pl, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
