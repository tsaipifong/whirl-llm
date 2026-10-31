# SPDX-License-Identifier: Apache-2.0
# arch025-engine S0 (from scratch/a25_engine/v3/graph3.py): module graph of the Engine cut, measured from
# engine_split_census.py's funcs.tsv. Run through tools/gates/engine_split_check.py.
#   python -I engine_split_graph.py <funcs.tsv> <cut.json> <out_dir> [--inline inline.tsv] [--writers now|leg_end] [--core]
# S0 additions (third review): A2 a type used by an entry but assigned to no module fails ("used but unassigned
# types"); planned rewrites apply to writes too; N3 every core.shared member is used by >= 3 component modules
# ("core.shared below 3 components"). Every failing check prints its own line prefix (the self-test greps it, A5).
# Reports (out_dir/graph_<name>.txt, exit 1 on any finding):
#   - unassigned entries / data members (every census entry must be in exactly one module);
#   - module edges (call, data, type) with the functions behind each, the cross-module call pairs;
#   - layering violations: an edge must point to a strictly lower layer (core.shared = layer 0);
#   - strongly connected components of the module graph (Tarjan): must be none of size > 1;
#   - per-token path: a hot caller whose callee was inlined in the c1ca7ce object (inline.tsv) and now sits in another
#     module (another TU) unless the callee is header-inline in its new module (cut 'header_inline');
#   - lines per module (census body lines).
# --v1 builds the v1 cut from member_table.md (design v1 layers) to reproduce the review's F2 / F3.
import json, os, re, sys, collections

FUNCS, CUT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]
inline_tsv = args[args.index("--inline") + 1] if "--inline" in args else None
v1 = args[args.index("--v1") + 1] if "--v1" in args else None
WSET = args[args.index("--writers") + 1] if "--writers" in args else "now"
os.makedirs(OUT, exist_ok=True)

rows = {}
for l in list(open(FUNCS, encoding="utf-8"))[1:]:
    p = l.rstrip("\n").split("\t")
    k = p[0].split("#")[0]
    r = rows.setdefault(k, {"file": p[1], "lines": 0, "data": set(), "calls": set(), "types": set()})
    r["lines"] += int(p[3])
    r["data"].update(x for x in p[4].split(",") if x)
    r["calls"].update(x for x in p[6].split(",") if x)
    r["types"].update("type:" + x for x in p[7].split(",") if x)
    r.setdefault("defs", set()).add(p[8] if len(p) > 8 else "")
all_data = set()
for l in list(open(os.path.join(os.path.dirname(FUNCS), "members.tsv"), encoding="utf-8"))[1:]:
    p = l.split("\t")
    if p[0] == "data":
        all_data.add(p[1])

if v1:
    name = "v1"
    layer_v1 = {"server.vram_watch": 1, "server.governor": 1, "server.job_queue": 1, "server.engine_stats": 1,
                "server.slot_pages": 2, "server.spe_glue": 2, "server.tier_glue": 2, "server.request": 2,
                "server.prefill": 2, "server.spec_cycle": 2, "server.engine": 3}
    mods = {m: {"layer": L, "funcs": [], "data": []} for m, L in layer_v1.items()}
    for l in open(v1, encoding="utf-8"):
        m = re.match(r"\| \d+ \| (\w+) \| `([^`]+)` \| \w+ \| ([^|]+)\|", l)
        if not m:
            continue
        kind, nm, mod = m.group(1), m.group(2), m.group(3).strip().split(" (")[0]
        if kind == "data":
            mods[mod]["data"].append(nm)
        elif kind == "func":
            mods[mod]["funcs"].append(nm)
        elif kind == "type":
            mods[mod]["funcs"].append("type:" + nm)
    cut = {"modules": mods, "hot": json.load(open(CUT, encoding="utf-8"))["hot"], "header_inline": []}
    same_layer_ok = True  # v1 allowed L2 <-> L2 through SlotOps: reported, counted as violations below
else:
    name = os.path.splitext(os.path.basename(CUT))[0]
    cut = json.load(open(CUT, encoding="utf-8"))
mods = cut["modules"]
rw = {k: v for k, v in cut.get("rewrites", {}).items() if not k.startswith("_")}
rw_log = []
for f, m in rw.items():
    for old, new in m.items():
        if f in rows and old in rows[f]["data"]:
            rows[f]["data"].discard(old); rows[f]["data"].add(new); rw_log.append(f"{f}: {old} -> {new}")
for d, m in cut.get("new_data", {}).items():
    if not d.startswith("_"): mods[m]["data"].append(d)
for f, m in cut.get("new_funcs", {}).items():
    if not f.startswith("_"): mods[m]["funcs"].append(f)
f2m, d2m = {}, {}
dups = []
for m, v in mods.items():
    for f in v["funcs"]:
        if f in f2m: dups.append(f)
        f2m[f] = m
    for d in v["data"]:
        if d in d2m: dups.append(d)
        d2m[d] = m
L = {m: v["layer"] for m, v in mods.items()}
out = []
bad = 0
un_f = sorted(k for k in rows if k not in f2m)
un_d = sorted(d for d in all_data if d not in d2m)
extra = sorted(f for f in f2m if f not in rows)
out.append(f"== {name}: {len(rows)} census entries, {len(all_data)} data members")
out.append(f"unassigned entries: {','.join(un_f) or '-'}; unassigned data: {','.join(un_d) or '-'}; "
           f"in cut but not in census: {','.join(extra) or '-'}; duplicates: {','.join(dups) or '-'}")
bad += len(un_f) + len(un_d) + len(dups)
# S0 A2: a type that an entry uses must belong to a module (type:X rows come from nested / helper types)
un_t = sorted({t for r in rows.values() for t in r["types"] if t not in f2m})
out.append(f"S0: used but unassigned types: {','.join(un_t) or '-'}")
bad += len(un_t)
# v3 R1: an entry of the cut that the census no longer finds is a failure (it moved where the census cannot see it),
# unless the cut lists it as planned (new_funcs: written by a later step); same for data (new_data).
planned_f = set(cut.get("new_funcs", {}))
miss_f = [f for f in extra if f not in planned_f]
miss_d = sorted(d for d in d2m if d not in all_data and d not in cut.get("new_data", {}))
out.append(f"v3: cut entries missing from the census (not planned): {','.join(miss_f) or '-'}; cut data missing: {','.join(miss_d) or '-'}")
bad += len(miss_f) + len(miss_d)
CLS = {k: v for k, v in cut.get("classes", {}).items() if not k.startswith("_")}
mism = []
for f, r in rows.items():
    for dfn in r.get("defs", ()):
        c = dfn.split()[-1].split("::")[0] if "::" in dfn else None
        if c in CLS and f in f2m and CLS[c] != f2m[f]:
            mism.append(f"{dfn} is in class {c} ({CLS[c]}) but the cut puts {f} in {f2m[f]}")
out.append(f"v3: class / module mismatches: {'; '.join(mism) or '-'}")
bad += len(mism)
_fp = os.path.join(os.path.dirname(FUNCS), "findings.txt")
fnd = [l.strip() for l in open(_fp, encoding="utf-8") if l.strip()] if os.path.exists(_fp) else []
out.append(f"v3: census findings (macros, friend, unlisted engine-header users, declared without body): {len(fnd)}")
for x in fnd: out.append("   " + x)
bad += len(fnd)
out.append("planned rewrites applied: " + ("; ".join(rw_log) or "-"))

edges = collections.defaultdict(lambda: collections.defaultdict(set))  # (a,b) -> kind -> {why}
for f, r in rows.items():
    a = f2m.get(f)
    if not a: continue
    for g in r["calls"] | r["types"]:
        b = f2m.get(g)
        if b and b != a:
            edges[(a, b)]["call" if not g.startswith("type:") else "type"].add(f"{f}->{g}")
    for d in r["data"]:
        b = d2m.get(d)
        if b and b != a:
            edges[(a, b)]["data"].add(f"{f}:{d}")
out.append("\n== module edges (caller module -> used module [kind n]: why)")
viol = []
for (a, b), kinds in sorted(edges.items()):
    ok = L[b] < L[a]
    tag = "" if ok else "  <-- VIOLATION (not to a lower layer)"
    if not ok: viol.append((a, b))
    for k, why in sorted(kinds.items()):
        out.append(f"{a} -> {b} [{k} {len(why)}]{tag}: {', '.join(sorted(why))}")
bad += len(viol)

# Tarjan SCC
graph = collections.defaultdict(set)
for (a, b) in edges: graph[a].add(b)
idx, low, st, on, sccs, c = {}, {}, [], set(), [], [0]
sys.setrecursionlimit(10000)
def sc(v):
    idx[v] = low[v] = c[0]; c[0] += 1; st.append(v); on.add(v)
    for w in graph[v]:
        if w not in idx:
            sc(w); low[v] = min(low[v], low[w])
        elif w in on:
            low[v] = min(low[v], idx[w])
    if low[v] == idx[v]:
        comp = []
        while True:
            w = st.pop(); on.discard(w); comp.append(w)
            if w == v: break
        sccs.append(comp)
for v in list(mods):
    if v not in idx: sc(v)
cyc = [s for s in sccs if len(s) > 1]
out.append(f"\n== cycles (strongly connected module groups of size > 1): {len(cyc)}")
for s in cyc: out.append("  " + " <-> ".join(sorted(s)))
bad += len(cyc)

pairs = sorted({(f, g) for f, r in rows.items() for g in r["calls"] if f2m.get(f) and f2m.get(g) and f2m[f] != f2m[g]})
callees = sorted({g for _, g in pairs})
out.append(f"\n== cross-module call pairs: {len(pairs)} (distinct callees {len(callees)}: {', '.join(callees)})")
nonroot = [(f, g) for f, g in pairs if f2m[f] != "server.engine"]
out.append(f"   of them not from server.engine (component -> component): {len(nonroot)}")

if inline_tsv:
    inl = {}
    for l in list(open(inline_tsv, encoding="utf-8"))[1:]:
        p = l.rstrip("\n").split("\t")
        inl[(p[0], p[1])] = (p[4], p[5])
    hot = set(cut.get("hot", []))
    hi = set(cut.get("header_inline", []))
    out.append("\n== per-token / per-chunk path (hot callers): cross-module calls and their c1ca7ce codegen")
    for f, g in pairs:
        if f not in hot: continue
        tu, st_ = inl.get((f, g), ("?", "?"))
        lost = st_ == "inlined" and g not in hi
        if lost: bad += 1
        out.append(f"  {f} -> {g} ({f2m[f]} -> {f2m[g]}): today {tu} {st_}"
                   + ("  <-- INLINING LOST on the hot path" if lost else (" (header-inline in its module)" if g in hi else "")))
    out.append("== cross-module calls inlined today (not hot; codegen changes, covered by the speed check)")
    for f, g in pairs:
        if f in hot: continue
        tu, st_ = inl.get((f, g), ("?", "?"))
        if st_ == "inlined":
            out.append(f"  {f} -> {g} today {tu} inlined" + (" (header-inline in its module)" if g in hi else ""))

# writers outside the owning module (F1: a new writer must be reported)
wr = {}
for l in list(open(FUNCS, encoding="utf-8"))[1:]:
    p = l.rstrip("\n").split("\t")
    wr.setdefault(p[0].split("#")[0], set()).update(x for x in p[5].split(",") if x)
for f, m in rw.items():   # S0: planned rewrites change writes too (govPublish: vram_mu_ -> gov_mu_)
    for old, new in m.items():
        if f in wr and old in wr[f]:
            wr[f].discard(old); wr[f].add(new)
ow = sorted(f"{f}:{d}" for f, ds in wr.items() for d in ds
            if f in f2m and d in d2m and d2m[d] not in (f2m[f], "core.shared"))
# v3 R3: every write to a core.shared member must be by an entry the cut lists for that member
sw = cut.get("shared_writers", {}).get(WSET, {})
sw_new = sorted(f"{f}:{d}" for f, ds in wr.items() for d in ds
                if d in d2m and d2m[d] == "core.shared" and f not in sw.get(d, []))
out.append(f"\n== writes to core.shared not in shared_writers '{WSET}' ({len(sw_new)}): {', '.join(sw_new) or '-'}")
bad += len(sw_new)
out.append(f"\n== writes to another module's data ({len(ow)}): {', '.join(ow) or '-'}")
if "outside_writers" in cut:
    exp = set(cut["outside_writers"].get(WSET, []))
    new = [x for x in ow if x not in exp]
    out.append(f"   not in the expected list '{WSET}': {', '.join(new) or '-'}")
    bad += len(new)

# v3 R4: each component's constructor reference list (cut modules[m]["refs"]) = the core.shared members it may use
ref_bad, ref_unused = [], []
for m, v in mods.items():
    if m in ("core.shared", "server.engine") or "refs" not in v: continue
    used = {d.split(":")[1] for (a, b), kinds in edges.items() if a == m and b == "core.shared" for d in kinds.get("data", ())}
    for d in sorted(used - set(v["refs"])): ref_bad.append(f"{m} uses {d} (not in its reference list)")
    for d in sorted(set(v["refs"]) - used): ref_unused.append(f"{m}:{d}")
noref = [m for m, v in mods.items() if m not in ("core.shared", "server.engine") and "refs" not in v and v["funcs"]]
out.append(f"\n== reference lists (R4): uses outside the list: {'; '.join(ref_bad) or '-'}; modules without a list: "
           f"{','.join(noref) or '-'}; listed but unused (info): {','.join(ref_unused) or '-'}")
bad += len(ref_bad) + len(noref)
# S0 N3: the ownership rule (DESIGN 2.1): a core.shared member is used by >= 3 component modules
cs_users = collections.defaultdict(set)
for (a, b), kinds in edges.items():
    if b == "core.shared" and a not in ("server.engine",):
        for d in kinds.get("data", ()): cs_users[d.split(":")[1]].add(a)
low = sorted(f"{d} ({len(cs_users[d])}: {','.join(sorted(cs_users[d])) or '-'})"
             for d in mods["core.shared"]["data"] if len(cs_users[d]) < 3 and d not in cut.get("new_data", {}))
out.append(f"S0: core.shared below 3 components: {'; '.join(low) or '-'}")
bad += len(low)
# v3 R6: core categories are mutually exclusive (always); --core: the leg-end shape of server.engine (R5)
cats = {k: v for k, v in cut.get("core_categories", {}).items() if not k.startswith("_")}
seen_c, overlap = {}, []
for c, fs in cats.items():
    for f in fs:
        if f in seen_c: overlap.append(f"{f} in {seen_c[f]} and {c}")
        seen_c.setdefault(f, c)
out.append(f"== core categories overlap (R6): {'; '.join(overlap) or '-'}")
bad += len(overlap)
if "--core" in args:
    bodies = json.load(open(os.path.join(os.path.dirname(FUNCS), "bodies.json"), encoding="utf-8"))
    core_f = sorted(f for f in rows if f2m.get(f) == "server.engine")
    cb = [f"{f}: in no category" for f in core_f if f not in seen_c]
    SHAPE = {k: v for k, v in cut.get("core_shapes", {}).items() if not k.startswith("_")}
    for f in core_f:
        c = seen_c.get(f)
        if not c: continue
        txt = "".join(bodies[k] for k in bodies if k.split("#")[0] == f)
        body = txt[txt.find("{") + 1: txt.rfind("}")] if "{" in txt else ""
        stm = [x for x in body.split(";") if x.strip() and x.strip() != "}"]
        rule = SHAPE.get(c, {})
        if rule.get("max_statements") is not None and len(stm) > rule["max_statements"]:
            cb.append(f"{f} ({c}): {len(stm)} statements > {rule['max_statements']}")
        for bad_re in rule.get("forbid", []):
            if re.search(bad_re, body): cb.append(f"{f} ({c}): body matches forbidden /{bad_re}/")
        if rule.get("max_lines") is not None and rows[f]["lines"] > rule["max_lines"]:
            cb.append(f"{f} ({c}): {rows[f]['lines']} lines > {rule['max_lines']}")
    # core reads / writes no component-owned data (the constructor / destructor are the assembly point: exempt)
    for (a, b), kinds in sorted(edges.items()):
        if a != "server.engine" or b == "core.shared": continue
        for w in sorted(kinds.get("data", ())):
            if w.split(":")[0] not in ("Engine", "~Engine") and w not in cut.get("core_exceptions", {}):
                cb.append(f"core uses {b} data: {w}")
    # E3: Engine's data members are exactly engine_members_final, in that order
    em = [l.split("\t")[1] for l in list(open(os.path.join(os.path.dirname(FUNCS), "members.tsv"), encoding="utf-8"))[1:]
          if l.split("\t")[0] == "data" and l.rstrip("\n").split("\t")[7] == "Engine"]
    want = cut.get("engine_members_final", [])
    if em != want:
        cb.append(f"Engine data members differ from engine_members_final: extra {sorted(set(em) - set(want))}, "
                  f"missing {sorted(set(want) - set(em))}" + (", order differs" if set(em) == set(want) else ""))
    out.append(f"\n== --core (leg-end shape of server.engine, R5 / R6): {len(cb)} problems")
    for x in cb: out.append("   " + x)
    bad += len(cb)

out.append("\n== size per module (census body lines / entries / data members owned)")
for m in sorted(mods, key=lambda x: L[x]):
    fl = [f for f in mods[m]["funcs"] if f in rows]
    out.append(f"  L{L[m]} {m}: {sum(rows[f]['lines'] for f in fl)} lines, {len(fl)} entries, {len(mods[m]['data'])} data")

out.append("\n== allowed edges (manifest dependencies) per module: the modules it uses")
for m in sorted(mods, key=lambda x: L[x]):
    deps = sorted({b for (a, b) in edges if a == m})
    out.append(f"  {m}: {', '.join(deps) or '-'}")
out.append(f"\nRESULT {name}: {'FAIL' if bad else 'OK'} ({len(viol)} layering violations, {len(cyc)} cycles, "
           f"{len(un_f) + len(un_d)} unassigned, {len(pairs)} cross-module call pairs)")
txt = "\n".join(out)
open(os.path.join(OUT, f"graph_{name}.txt"), "w", encoding="utf-8").write(txt + "\n")
print(txt.split("\n")[-1])
sys.exit(1 if bad else 0)
