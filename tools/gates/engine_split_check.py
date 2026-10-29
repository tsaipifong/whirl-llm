# SPDX-License-Identifier: Apache-2.0
"""Engine split check (arch025 N4, PLAN: class Engine split; progress/arch025-engine.DESIGN.md S0, E2, E3).

Runs, on a source tree (default: this repo):
  1. the cut pin (review3 C2; S0b = S0 review act-on 3): every architecture section of tools/gates/engine_cut.json
     must match tests/pins/engine_cut_arch.sha256, so a step cannot edit the cut and pass its own check. new_funcs /
     new_data (planned rows) are pinned as key -> module maps from which rows may only be DELETED (a step deletes a row
     when it writes the entry); a new key or a changed module is a MISMATCH. The pin also holds the sha256 of the
     checker tools (TOOLS below, line endings normalised), and the pin file's own sha256 must have a row in
     tests/pins/engine_cut_reviews.tsv (sha256 <TAB> review <TAB> what changed): rewriting the pin with --pin-write in
     the same commit as a cut / tool edit leaves it UNREVIEWED until a review row is added (visible in the diff).
     "_doc" keys are outside the pin.
  2. engine_split_census.py (Engine + the cut's component classes; writes, nested types, aliases, holder types)
  3. engine_split_graph.py (unassigned, layering, cycles, writers, reference lists, N3, [--core] leg-end shape)

  python -I tools/gates/engine_split_check.py [--root R] [--cut C] [--work W] [--writers now|leg_end] [--core]
  python -I tools/gates/engine_split_check.py --pin-write          # rewrite the pin (then add the review row)
  --reviews F: the review ledger (default tests/pins/engine_cut_reviews.tsv); --no-pin: skip the pin (self-test only)
  python -I tools/gates/engine_split_check.py --selftest [--work W] [--base R2 ...]   (engine_split_selftest.py)

Exit 0 = "engine split check: ok", 1 = a finding (the report lines name it), 3 = the tools crashed.
"""
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PIN = os.path.join(REPO, "tests", "pins", "engine_cut_arch.sha256")
REVIEWS = os.path.join(REPO, "tests", "pins", "engine_cut_reviews.tsv")
UNPINNED = {"new_funcs", "new_data"}   # pinned separately: key -> module, rows may only be deleted (S0b)
# S0b: the checker tools themselves (in this script's folder); a change of any of them is a pin change
TOOLS = ["engine_split_census.py", "engine_split_graph.py", "engine_split_check.py", "engine_split_selftest.py",
         "engine_split_inline_c1ca7ce.tsv", "member_order_check.py"]


def sha_text(p):
    """sha256 of a text file with CRLF -> LF (a git autocrlf checkout hashes the same)"""
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def planned(cut):
    return {f"{sec} {k}": v for sec in sorted(UNPINNED) for k, v in sorted(cut.get(sec, {}).items()) if not k.startswith("_")}


def tools_sha(here):
    return {f"tools {t}": (sha_text(os.path.join(here, t)) if os.path.exists(os.path.join(here, t)) else "absent") for t in TOOLS}


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


def pin_check(cutp, pinp, reviews=REVIEWS):
    cut = json.load(open(cutp, encoding="utf-8"))
    have = sections(cut)
    if not os.path.exists(pinp):
        return [f"engine cut pin: MISSING {pinp}"]
    want, wplan, wtools = {}, {}, {}
    for l in open(pinp, encoding="utf-8"):
        l = l.strip()
        if not l or l.startswith("#"): continue
        q = l.split()
        if q[0] in UNPINNED and len(q) == 3: wplan[f"{q[0]} {q[1]}"] = q[2]
        elif q[0] == "tools" and len(q) == 3: wtools[f"tools {q[1]}"] = q[2]
        elif len(q) == 2: want[q[0]] = q[1]
        else: return [f"engine cut pin: UNREADABLE line in {pinp}: {l[:80]}"]
    bad = []
    for k in sorted(set(have) | set(want)):
        if have.get(k) != want.get(k):
            bad.append(f"engine cut pin: MISMATCH section {k} (cut {(have.get(k) or 'absent')[:12]}, pin "
                       f"{(want.get(k) or 'absent')[:12]})")
    # S0b: planned rows may only be deleted
    for k, m in planned(cut).items():
        if wplan.get(k) != m:
            bad.append(f"engine cut pin: MISMATCH {k} (cut {m}, pin {wplan.get(k, 'absent: a new planned row')}) "
                       f"[new_funcs / new_data rows may only be deleted]")
    # S0b: the checker tools
    ht = tools_sha(HERE)
    for k in sorted(set(ht) | set(wtools)):
        if ht.get(k) != wtools.get(k):
            bad.append(f"engine cut pin: MISMATCH {k} (tree {ht.get(k, 'absent')[:12]}, pin {wtools.get(k, 'absent')[:12]})")
    # S0b: the pin file itself must be reviewed
    ps = sha_text(pinp)
    rows = set()
    if os.path.exists(reviews):
        for l in open(reviews, encoding="utf-8"):
            q = l.rstrip("\n").split("\t")
            if len(q) >= 3 and not l.startswith("#") and q[1].strip() and q[2].strip(): rows.add(q[0].strip())
    if ps not in rows:
        bad.append(f"engine cut pin: UNREVIEWED (pin {os.path.basename(pinp)} sha256 {ps} has no row in {reviews}; "
                   f"a cut or tool change needs a review row: sha256<TAB>review<TAB>what changed)")
    return bad


def main():
    if "--selftest" in sys.argv:
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "engine_split_selftest.py")] + sys.argv[1:])
        return r.returncode
    cutp = arg("--cut", os.path.join(HERE, "engine_cut.json"))
    pinp = arg("--pin", PIN)
    if "--pin-write" in sys.argv:
        cut = json.load(open(cutp, encoding="utf-8"))
        txt = ("# arch025-engine: sha256 of each architecture section of tools/gates/engine_cut.json (review3 C2).\n"
               "# A change here is a cut change: it needs its own review (DESIGN S0) and a row in tests/pins/engine_cut_reviews.tsv\n"
               "# with this file's sha256 (S0b). new_funcs / new_data: key -> module, rows may only be deleted. tools: the checkers.\n"
               + "".join(f"{k} {h}\n" for k, h in sections(cut).items())
               + "".join(f"{k} {m}\n" for k, m in planned(cut).items())
               + "".join(f"{k} {h}\n" for k, h in tools_sha(HERE).items()))
        open(pinp, "w", encoding="utf-8", newline="\n").write(txt)
        print(f"engine cut pin: wrote {pinp}; sha256 {sha_text(pinp)} (add a row to {REVIEWS} after the review)")
        return 0
    root = arg("--root", REPO)
    work = arg("--work", os.path.join(REPO, "build", "engine_split_check"))
    os.makedirs(work, exist_ok=True)
    lines = [] if "--no-pin" in sys.argv else pin_check(cutp, pinp, arg("--reviews", REVIEWS))
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
