# SPDX-License-Identifier: Apache-2.0
"""Engine split check (arch025 N4, PLAN: class Engine split; progress/arch025-engine.DESIGN.md S0, E2, E3).

Runs, on a source tree (default: this repo):
  1. the cut pin (review3 C2): every architecture section of tools/gates/engine_cut.json must match
     tests/pins/engine_cut_arch.sha256, so a step cannot edit the cut and pass its own check. Only new_funcs /
     new_data (planned rows a step deletes when it writes them) and "_doc" keys are outside the pin. A cut change is
     a separate, reviewed change of the pin file (--pin-write prints / writes the new pin).
  2. engine_split_census.py (Engine + the cut's component classes; writes, nested types, aliases, holder types)
  3. engine_split_graph.py (unassigned, layering, cycles, writers, reference lists, N3, [--core] leg-end shape)

  python -I tools/gates/engine_split_check.py [--root R] [--cut C] [--work W] [--writers now|leg_end] [--core]
  python -I tools/gates/engine_split_check.py --pin-write          # rewrite the pin after a reviewed cut change
  python -I tools/gates/engine_split_check.py --selftest [--work W] [--base R2 ...]   (engine_split_selftest.py)

Exit 0 = "engine split check: ok", 1 = a finding (the report lines name it), 3 = the tools crashed.
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PIN = os.path.join(REPO, "tests", "pins", "engine_cut_arch.sha256")
UNPINNED = {"new_funcs", "new_data"}


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a else default


def sections(cut):
    out = {}
    for k in sorted(cut):
        if k.startswith("_") or k in UNPINNED: continue
        v = cut[k]
        if k == "modules":
            for m in sorted(v):
                out[f"modules.{m}"] = v[m]
            continue
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        out[k] = v
    return {k: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for k, v in out.items()}


def pin_check(cutp, pinp):
    have = sections(json.load(open(cutp, encoding="utf-8")))
    if not os.path.exists(pinp):
        return [f"engine cut pin: MISSING {pinp}"]
    want = {}
    for l in open(pinp, encoding="utf-8"):
        l = l.strip()
        if l and not l.startswith("#"):
            k, h = l.split()
            want[k] = h
    bad = []
    for k in sorted(set(have) | set(want)):
        if have.get(k) != want.get(k):
            bad.append(f"engine cut pin: MISMATCH section {k} (cut {(have.get(k) or 'absent')[:12]}, pin "
                       f"{(want.get(k) or 'absent')[:12]})")
    return bad


def main():
    if "--selftest" in sys.argv:
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "engine_split_selftest.py")] + sys.argv[1:])
        return r.returncode
    cutp = arg("--cut", os.path.join(HERE, "engine_cut.json"))
    pinp = arg("--pin", PIN)
    if "--pin-write" in sys.argv:
        txt = ("# arch025-engine: sha256 of each architecture section of tools/gates/engine_cut.json (review3 C2).\n"
               "# A change here is a cut change: it needs its own review (DESIGN S0). new_funcs / new_data are not pinned.\n"
               + "".join(f"{k} {h}\n" for k, h in sections(json.load(open(cutp, encoding="utf-8"))).items()))
        open(pinp, "w", encoding="utf-8", newline="\n").write(txt)
        print(f"engine cut pin: wrote {pinp}")
        return 0
    root = arg("--root", REPO)
    work = arg("--work", os.path.join(REPO, "build", "engine_split_check"))
    os.makedirs(work, exist_ok=True)
    lines = [] if "--no-pin" in sys.argv else pin_check(cutp, pinp)
    for l in lines: print(l)
    c = os.path.join(work, "census")
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "engine_split_census.py"), root, c, "--cut", cutp],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode not in (0, 2):
        print(r.stderr.strip()); print("engine split check: CRASH (census)"); return 3
    g = [sys.executable, "-I", os.path.join(HERE, "engine_split_graph.py"), os.path.join(c, "funcs.tsv"), cutp, c,
         "--inline", arg("--inline", os.path.join(HERE, "engine_split_inline_c1ca7ce.tsv")),
         "--writers", arg("--writers", "now")] + (["--core"] if "--core" in sys.argv else [])
    r2 = subprocess.run(g, capture_output=True, text=True)
    if r2.returncode not in (0, 1):
        print(r2.stdout.strip(), r2.stderr.strip()); print("engine split check: CRASH (graph)"); return 3
    rep = os.path.join(c, f"graph_{os.path.splitext(os.path.basename(cutp))[0]}.txt")
    print(r2.stdout.strip() + f"  (report: {rep})")
    if r2.returncode or lines or r.returncode:
        txt = open(rep, encoding="utf-8").read().splitlines()
        for l in txt:
            if ("VIOLATION" in l or "INLINING LOST" in l or (l.startswith(("unassigned", "v3:", "S0:", "   not in", "== writes to core"))
                and not l.rstrip().endswith(("-", ": 0", "-; cut data missing: -")))):
                print("  " + l[:400])
        print("engine split check: FAIL")
        return 1
    print(f"engine split check: ok ({root}, writers {arg('--writers', 'now')}{', --core' if '--core' in sys.argv else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
