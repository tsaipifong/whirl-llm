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
import hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PIN = os.path.join(REPO, "tests", "pins", "engine_cut_arch.sha256")
REVIEWS = os.path.join(REPO, "tests", "pins", "engine_cut_reviews.tsv")
UNPINNED = {"new_funcs", "new_data"}   # pinned separately: key -> module, rows may only be deleted (S0b)
# S0b: the checker tools themselves; a change of any of them is a pin change. S0c (S0b review act-on 2): revtext.py and
# e8_judge.py (+ abjudge.py, which e8_judge imports) joined. pin key "tools <name>" -> path under tools/.
TOOLS = {"engine_split_census.py": "gates/engine_split_census.py", "engine_split_graph.py": "gates/engine_split_graph.py",
         "engine_split_check.py": "gates/engine_split_check.py", "engine_split_selftest.py": "gates/engine_split_selftest.py",
         "engine_split_inline_c1ca7ce.tsv": "gates/engine_split_inline_c1ca7ce.tsv", "member_order_check.py": "gates/member_order_check.py",
         "revtext.py": "impact/revtext.py", "e8_judge.py": "bench/e8_judge.py", "abjudge.py": "bench/abjudge.py"}
BASE_BRANCHES = ["arch025-engine", "origin/arch025-engine"]   # S0c: the range is merge-base(HEAD, first that exists)..
REL_PIN, REL_CUT, REL_REV = "tests/pins/engine_cut_arch.sha256", "tools/gates/engine_cut.json", "tests/pins/engine_cut_reviews.tsv"


def sha_text(p):
    """sha256 of a text file with CRLF -> LF (a git autocrlf checkout hashes the same)"""
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def planned(cut):
    return {f"{sec} {k}": v for sec in sorted(UNPINNED) for k, v in sorted(cut.get(sec, {}).items()) if not k.startswith("_")}


def tools_sha(here):
    """here = the tools/gates folder; the tools are found relative to its parent (tools/)"""
    root = os.path.dirname(here)
    return {f"tools {t}": (sha_text(os.path.join(root, r)) if os.path.exists(os.path.join(root, r)) else "absent")
            for t, r in TOOLS.items()}


def _lf(t):
    return None if t is None else t.replace("\r\n", "\n")


def _git(*a):
    try:
        r = subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def base_state():
    """S0c (S0b review act-on 2): what the range is compared against.
    --base-dir D: files D/engine_cut.json, D/engine_cut_arch.sha256, D/engine_cut_reviews.tsv (self-test; the commit
    order is not checked). Else git: --base REF, env WHIRL_ENGINE_CUT_BASE, or merge-base(HEAD, arch025-engine |
    origin/arch025-engine). None = no base: fail closed (the pin then needs a valid review row whatever changed)."""
    bd = arg("--base-dir")
    if bd:
        rd = lambda n: (_lf(open(os.path.join(bd, n), encoding="utf-8").read()) if os.path.exists(os.path.join(bd, n)) else None)
        return {"ref": f"files {bd}", "cut": rd("engine_cut.json"), "pin": rd("engine_cut_arch.sha256"),
                "reviews": rd("engine_cut_reviews.tsv"), "git": None}
    ref = arg("--base") or os.environ.get("WHIRL_ENGINE_CUT_BASE")
    if not ref:
        for b in BASE_BRANCHES:
            mb = _git("merge-base", "HEAD", b)
            if mb and mb.strip():
                ref = mb.strip(); break
    if not ref or _git("rev-parse", "--verify", "--quiet", ref + "^{commit}") is None:
        return None
    show = lambda rev, rel: _lf(_git("show", f"{rev}:{rel}"))
    g = {"head_pin": show("HEAD", REL_PIN), "pin_commit_reviews": None, "pin_commit": None}
    log = _git("log", "--format=%H", f"{ref}..HEAD", "--", REL_PIN)
    if log and log.split():
        g["pin_commit"] = log.split()[0]
        g["pin_commit_reviews"] = show(g["pin_commit"], REL_REV) or ""
    return {"ref": ref, "cut": show(ref, REL_CUT), "pin": show(ref, REL_PIN), "reviews": show(ref, REL_REV), "git": g}


def _report_path(p):
    for c in (p, os.path.join(REPO, p), os.path.join(os.path.dirname(REPO), p)):
        if os.path.isabs(c) and os.path.isfile(c): return c
    return None


def row_problem(q, sha):
    """S0c: why a ledger row cannot review pin sha256 `sha`, or None. Columns: sha256, review report path (+ note),
    what changed, author (who made the cut / tool change). The report must exist (absolute, or relative to the repo or
    to its parent folder), name the full sha256 and have a `reviewer: <name>` line whose name is not the author; the
    review column must not say pending."""
    if len(q) < 4 or not all(x.strip() for x in q[:4]):
        return "needs 4 non-empty columns: sha256, review report, what changed, author"
    rev, author = q[1].strip(), q[3].strip()
    if re.search(r"\bpending\b", rev, re.I): return "review column says pending"
    rp = _report_path(rev.split()[0])
    if not rp: return f"review report {rev.split()[0]} not found"
    txt = open(rp, encoding="utf-8", errors="replace").read()
    if sha not in txt: return f"review report {os.path.basename(rp)} does not name the pin sha256"
    m = re.search(r"(?mi)^[\s>*-]*reviewer:\s*([^\s,;(]+)", txt)
    if not m: return f"review report {os.path.basename(rp)} has no 'reviewer: <name>' line"
    if m.group(1).strip().lower() == author.lower():
        return f"reviewer {m.group(1)} is the author (no self-review)"
    return None


def _rows(text):
    return [l.split("\t") for l in (text or "").split("\n") if l.strip() and not l.startswith("#")]


def _pinkeys(text):
    out = {}
    for l in (text or "").split("\n"):
        q = l.split()
        if not q or l.startswith("#"): continue
        out[q[0] + ((" " + q[1]) if q[0] in UNPINNED | {"tools"} and len(q) > 1 else "")] = l
    return out


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


def pin_check(cutp, pinp, reviews=REVIEWS, base="auto"):
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
    # S0b: the pin file itself must be reviewed. S0c (S0b review act-on 2): judged against the base of the range
    ps = sha_text(pinp)
    rtext = _lf(open(reviews, encoding="utf-8").read()) if os.path.exists(reviews) else ""
    rows = _rows(rtext)
    if base == "auto": base = base_state()
    ptext = _lf(open(pinp, encoding="utf-8").read())
    if base is None:
        changed = True
        print("engine cut pin: NO BASE (no --base / merge-base with " + " | ".join(BASE_BRANCHES) + "): the pin needs a valid "
              "review row whatever changed")
    else:
        changed = base["pin"] != ptext
        # (b) the ledger is append-only: the base ledger's lines stay at the top, byte for byte, in order
        bl = [l for l in (base["reviews"] or "").split("\n") if l.strip()]
        cl = [l for l in rtext.split("\n") if l.strip()]
        if cl[:len(bl)] != bl:
            k = next((i for i in range(len(bl)) if i >= len(cl) or cl[i] != bl[i]), len(bl))
            bad.append(f"engine cut pin: LEDGER EDITED (line {k + 1} of the base ledger ({base['ref'][:24]}) changed or "
                       f"removed; rows are only appended)")
        # (a) planned rows: the key set stays a subset of the base cut's, modules unchanged
        if base["cut"]:
            bc = json.loads(base["cut"])
            for sec in sorted(UNPINNED):
                bs = {k: v for k, v in bc.get(sec, {}).items() if not k.startswith("_")}
                for k, m in sorted(cut.get(sec, {}).items()):
                    if k.startswith("_"): continue
                    if k not in bs:
                        bad.append(f"engine cut pin: NEW PLANNED ROW {sec} {k} (not in the base cut {base['ref'][:24]}; "
                                   f"rows may only be deleted)")
                    elif bs[k] != m:
                        bad.append(f"engine cut pin: PLANNED ROW MOVED {sec} {k} ({bs[k]} -> {m} against the base cut)")
        if changed:
            bp, cp = _pinkeys(base["pin"]), _pinkeys(ptext)
            diff = sorted(k for k in set(bp) | set(cp) if bp.get(k) != cp.get(k))
            print(f"engine cut pin: CHANGED IN THIS RANGE (base {base['ref'][:24]}): {len(diff)} pin lines: "
                  + ", ".join(diff[:12]) + (" ..." if len(diff) > 12 else "") + " - needs a valid review row in a later commit")
    mine = [q for q in rows if q[0].strip() == ps]
    if not changed:
        # S0c: an empty range (e.g. on the integration branch itself, where merge-base == HEAD) still needs a VALID row
        # for today's pin - a row that only exists is not enough (an unreviewed pin merged in must not pass there)
        if not mine:
            bad.append(f"engine cut pin: UNREVIEWED (pin {os.path.basename(pinp)} sha256 {ps} has no row in {reviews})")
        elif all(row_problem(q, ps) for q in mine):
            bad.append(f"engine cut pin: REVIEW ROW INVALID (pin sha256 {ps[:12]}: "
                       + " | ".join(row_problem(q, ps) for q in mine) + ")")
        return bad
    if not mine:
        bad.append(f"engine cut pin: UNREVIEWED (pin {os.path.basename(pinp)} sha256 {ps} has no row in {reviews}; a cut or "
                   f"tool change needs a row sha256<TAB>review report<TAB>what changed<TAB>author, added after the review)")
        return bad
    g = base["git"] if base else None
    probs = []
    for q in mine:
        pr = row_problem(q, ps)
        if pr is None and g is not None:
            if g["head_pin"] != ptext:
                pr = "SELF-REVIEWED: the pin change is not committed; its review row must land in a later commit than the pin"
            elif g["pin_commit"] and any(x.split("\t")[0].strip() == ps for x in (g["pin_commit_reviews"] or "").split("\n")):
                pr = f"SELF-REVIEWED: the row was added in the same commit as the pin change ({g['pin_commit'][:12]})"
        if pr is None:
            return bad
        probs.append(pr)
    bad.append(f"engine cut pin: REVIEW ROW INVALID (pin sha256 {ps[:12]}: " + " | ".join(probs) + ")")
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
