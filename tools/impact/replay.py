# SPDX-License-Identifier: Apache-2.0
"""Replays one version of tools/impact over a commit range and dumps what it selects per commit (arch024-S12, from the
S12 review's replay). validate.py --baseline runs it twice (the baseline's tools and these tools, on the same tree)
and lists every commit for which the new tools select fewer release steps or paths ("narrowed").

    python -I tools/impact/replay.py --tools DIR --root REPO --rev REV --n N [--skip K] --out OUT.json

--tools: a folder with impact.py / impact_build.py / cxxscan.py (any version). The map is built once from REV by that
version's impact_build, so both runs see the same tree. Per commit (product code only): release steps, paths, CPU tests,
kernel tests.
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--tools", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--rev", default="HEAD")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--exclude", default="", help="comma-separated commits to leave out (validate.py --exclude-tuned)")
a = ap.parse_args()
sys.path.insert(0, a.tools)
import impact  # noqa: E402
import impact_build  # noqa: E402

impact.ROOT = a.root
if getattr(impact.git, "__kwdefaults__", None) is not None:
    impact.git.__kwdefaults__ = {"root": a.root}


def git(*x):
    r = subprocess.run(["git", "-C", a.root, *x], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"replay: git {' '.join(x)}: {r.stderr.strip()}")
    return r.stdout


t0 = time.time()
data = subprocess.run(["git", "-C", a.root, "archive", "--format=tar", a.rev, "src", "include", "kernels", "tests", "tools", "cmake",
                       "CMakeLists.txt"], capture_output=True).stdout
d = tempfile.mkdtemp(prefix="replay_")
with tarfile.open(fileobj=io.BytesIO(data)) as t:
    t.extractall(d, filter="data")
m = impact_build.build(d)
shutil.rmtree(d, ignore_errors=True)
res = {}
roots = set(git("rev-list", "--max-parents=0", a.rev).split())
for s in git("log", "--format=%h", "--no-merges", f"--skip={a.skip}", f"-{a.n}", a.rev).split():
    if any(r.startswith(s) for r in roots):
        continue  # root commit: no parent to diff against
    if s[:7] in set(x[:7] for x in a.exclude.split(",") if x):
        continue
    ch = impact.parse_diff(git("diff", "-U0", "--no-color", "-M", f"{s}^", s))
    prod = {p: c for p, c in ch.items() if p.startswith(("src/", "include/", "kernels/"))}
    if not prod:
        continue
    r = impact.analyze(prod, m, s, f"{s}^")
    res[s] = {"steps": sorted(r["release_gate_steps"]), "paths": sorted(r["paths"]),
              "cpu": sorted(f"{e}:{x}" for e, v in r["cpu_tests"].items() for x in v), "kt": sorted(r["kernel_tests"]),
              # measure-tools P1: what the declared-name rule saw (absent in older tools): declared names, moved notes
              "decl": sorted({d["name"] for d in r.get("declarations", [])}),
              "moved": sum(1 for n in r["notes"] if "moved / re-indented" in n), "defects": r.get("map_defects", []),
              # mt-p1 pass 5 (review4 C1): the evidence validate --pass-json needs for R-fallback-resolved (users found per
              # changed file; the changed product files) - absent with older tools, which then cannot pass that rule
              "users": r.get("decl_users", {}), "files": sorted(prod),
              # mt-p1 pass 6 (review5 A-2): changed file -> the paths its own lines selected (absent with older tools)
              "owned": r.get("paths_by_file")}
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=0)
print(f"replay: {len(res)} commits, {round(time.time() - t0)} s ({a.tools})", file=sys.stderr)
