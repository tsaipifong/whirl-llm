# SPDX-License-Identifier: Apache-2.0
"""Test what you changed (arch025-modsel, owner 2026-10-28 13:00: 「我已經受夠改一壞三，以及一改測一大串這種浪費時間的爛事了」).

One command takes a change and prints (and with --run executes) exactly:
  * ALL CPU steps of release_gate.ps1 (owner 2026-10-28 「乾脆都跑了吧」), once; the two per-GPU CPU checks
    (cpu_golden_xcheck, kl_guard) on both GPUs; the C++ safety steps (SAFETY_STEPS) are left out only when no changed
    file is one of their inputs (NO_SAFETY_INPUT allow-list: docs, python gate / impact tools, golden refs; --all-cpu),
  * the GPU steps of the touched modules (tools/gates/modules.json, with their callers) plus the GPU steps the impact
    map selects for the diff (tools/impact/impact.py), on the GPUs modules.json / the map name,
  * the FULL two-GPU matrix only when the change touches a CORE module or file (CORE / CORE_GLOBS below, with the reason
    for each), a contract module, a build-wide file, product code in no module, or when work equivalence cannot be shown
    (code objects of --base-build and --build differ although no kernel source changed),
  * minus the GPU steps a work-equivalence proof covers (--proof DEV=report.json: an EQUIVALENT tools/impact/work_equal.py
    report bound to this build's whirl.exe sha256), only when the diff touches no kernel source.
Every --run appends one row to the test-cost ledger (internal_docs/TEST_COST_LEDGER.tsv of the workspace, or --ledger).
DEV-TIME ONLY: a release / rc merge / bug-fix release still runs the full release gate on both GPUs (docs/dev/testing.md);
the summaries are tagged change_<id>_<device> and never write PASS_gate_<device>.

    python tools/gates/test_change.py --worktree                     plan for uncommitted changes (vs HEAD)
    python tools/gates/test_change.py --range A..B                   plan for what B adds since it left A
    python tools/gates/test_change.py --commit SHA                   plan for one commit
        [--build DIR] [--base-build DIR] [--proof r9700=FILE] [--proof 8060s=FILE] [--one-gpu r9700|8060s] [--json] [--all-cpu]
        [--run [--work DIR] [--force]] [--ledger FILE] [--record-plan]
    python tools/gates/test_change.py --refresh-times [ROOT ...]      step durations from release-gate summaries -> STEP_TIMES.tsv
    python tools/gates/test_change.py --backtest N --rev REV [--out DIR]   plans for REV's last N non-merge product commits
Exit: 0 plan printed / run passed; 1 a step failed; 2 setup error (bad args, another whirl process running, gate exit 2).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fnmatch
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "impact"))
import impact  # noqa: E402
import impact_build  # noqa: E402

DEVICES = ("r9700", "8060s")  # run order: one GPU after the other, never both at once
ARCH = {"r9700": "gfx1201", "8060s": "gfx1151"}
DEV_OF_ARCH = {v: k for k, v in ARCH.items()}
PER_DEVICE_CPU = ("cpu_golden_xcheck", "kl_guard")  # CPU steps that check one GPU's references: run with each GPU

# Core modules (tools/gates/modules.json names): a change here runs the full matrix on both GPUs. The brief's list is
# forward / kernels / kv / numerics / device; decided per module from its modules.json row:
CORE = {
    "forward.dispatch": "forward: host dispatch of every forward kernel (all models, modes, GPUs)",
    "forward.attn": "forward: attention blocks of every layer",
    "forward.ffn": "forward: FFN / MoE blocks of every layer",
    "forward.gdn": "forward: GDN (DeltaNet) layers",
    "forward.run": "forward: the decode / prefill driver",
    "forward.mtp_verify": "forward: MTP draft verify (spec decode of every mode)",
    "forward.prefill": "forward: prefill path (chunks, checkpoints)",
    "forward.vision_rows": "forward: vision rows enter the same forward (image tokens of every vision request)",
    "model.kv": "kv: KV cache layout / formats the attention kernels read",
    "model.weights": "kernels: weight layout / repack the GEMV / GEMM / MoE kernels read (testing.md: both GPUs)",
    "model.scratch": "forward: activation buffers every forward kernel writes",
    "model.tune": "kernels: the per-GPU kernel / tile choice tables",
    "handles": "device: RAII device buffers / streams / events under every GPU call",
    "contract.abi": "numerics / kernels_abi contracts (modules.json full: true; arch025-lane2 N3 split of contract)",
    "contract.device": "device::Backend contract (modules.json full: true)",
    "contract.server": "ServerModel contract (modules.json full: true)",
}
# Not core (their own steps are enough; modules.json rows already carry their GPU steps): env_snapshot, ckpt_store,
# page_pool, prefix_store, tier_bridge, protocol, engine_gov, threads, sampling, platform.*, server.*, cli, spec.*,
# model.arch_registry, model.qwen35.arch / model.qwen35.server (registration / glue: golden + MTP server steps in their rows).

# Core by path, also for paths of older commits that no module glob matches any more (e.g. src/model/forward/*.cpp
# before the qwen35 move) and for files no module owns:
CORE_GLOBS = [
    ("kernels/*", "kernels: GPU kernel source"),
    ("*.hip", "kernels: GPU kernel source"),
    ("src/kernels_host/*", "kernels: host side of the kernel ABI / loaders"),
    ("include/whirl/kernels_abi.h", "kernels: kernel argument structs (testing.md: both GPUs)"),
    ("include/whirl/kernels_abi_layout.h", "kernels: the one layout table of the kernel argument structs (host + device, arch025 N13)"),
    ("include/whirl/numerics.h", "numerics: mode / precision definitions"),
    ("src/common/numerics.cpp", "numerics: mode / precision resolution"),
    ("src/numerics/*", "numerics"),
    ("include/whirl/ops/*", "kernels: op tables"),
    ("src/ops/*", "kernels: op dispatch"),
    ("src/hip/*", "device: HIP runtime layer"),
    ("src/device/*", "device: device backend"),
    ("src/model/forward*", "forward"),
    ("src/model/*/forward*", "forward"),
    ("src/model/kv.*", "kv"),
    ("src/model/*/kv.*", "kv"),
]
# A change to one of these (or to a header that a .hip file includes) can change the code objects: no work-equivalence
# short-cut.
KERNEL_GLOBS = ["kernels/*", "*.hip", "include/whirl/kernels_abi.h", "include/whirl/kernels_abi_layout.h", "src/kernels_host/*"]
# build files can change code objects too (flags, offload archs): no static "cannot change" claim, but a binary proof
# (work_equal code objects identical) still counts for them
BUILD_GLOBS = ["cmake/*", "CMakeLists.txt", "build.bat", "redist/*"]
# GPU steps whose work a work_equal proof covers: launch runs keyed "<step>/<command>" in the report, plus kt_full when
# the code objects are identical and golden_* when golden is identical.
PROOF_EXES = ("whirl.exe", "whirl-server.exe")
# arch025-modsel2 (PLAN 3.5): the C++ safety steps build and run the C / C++ tree with their own tools; a change that
# touches none of their inputs gives the parent's result again (same inputs -> same result, like a build-cache hit), so a
# dev-time plan skips them ONLY when every changed file is on this allow-list (fail closed: anything not listed runs
# them). Measured cost (stored summaries, PASS medians): cpu_asan 662 s (n 10, max 1328), cpu_analyze 139 s (n 28, full
# rebuild 618-903), cpu_tsa 149 s (n 28, max 403), cpu_safety_selftest 18 s: ~16 min a change, up to ~44 min.
# The release gate (release_gate.ps1 without -Steps / -Module) always runs them; --all-cpu keeps them here.
# arch025-modsel2 (PLAN 3.6, X12 Finding K8-7): product code whose modules select no GPU step plans the full matrix
# (fail closed: never miss a module), except for these modules, which generate no GPU-visible code:
NO_GPU_MODULES = {
    "platform.headers": "compile-time proof that the platform headers stay platform-free (headers_win.cpp; cpu_platform_check)",
    "common.tsa": "clang thread-safety annotation macros, no code generated (cpu_tsa checks them)",
    "device.null": "the CPU null backend of the device tests (whirl-device-tests); src/device/* is a CORE_GLOBS path anyway",
}
SAFETY_STEPS = ("cpu_safety_selftest", "cpu_asan", "cpu_analyze", "cpu_tsa")
NO_SAFETY_INPUT = ["docs/*", "*.md", "LICENSE", "NOTICE", "tools/impact/*.py", "tools/impact/*.json", "tools/impact_map.json",
                   "tools/gates/*.py", "tools/gates/*.json", "tools/gates/*.tsv", "tools/gates/*.txt", "tools/klgate/*.py",
                   "tools/golden/*.py", "tests/golden/*"]

# arch025-n11 (PLAN N11): a core module plans its OWN GPU step set instead of the full matrix only when recorded GPU
# mutation evidence supports it (progress/arch025-n11.PRIOR_ART.md: fault-detection evidence, not dependency safety).
# The evidence rows live in tools/gates/core_steps.json (one row per mutant x card: verdict and the GPU step that killed
# it, from scratch gpumut.ps1 kill_<card>.tsv files via --add-kill-records); the step sets are COMPUTED here from those
# rows, never written by hand. Rule (fail closed; core_step_sets): every mutant of the module in tools/impact/mutations.json
# has a newest row on BOTH cards, each KILLED by a named GPU step or BUILD-FAIL (compiler), and >= 2 of them are KILLED
# on each card. Then the set of a card = the steps that killed them on that card + golden_swift + golden_moe ("always").
# Anything else (a survivor, CAP / GATE-ERROR / NOT-RUN / DEVICE-MISMATCH, a mutant not run on one card, a new mutant
# added to the catalog after the run) keeps the module on the full matrix. contract.* modules are always full.
CORE_STEPS_FILE = os.path.join("tools", "gates", "core_steps.json")
CORE_ALWAYS = ["golden_swift", "golden_moe"]
CORE_MIN_MUTANTS = 2  # floor: a data file asking for fewer is ignored (Shin et al. STVR 2018: one mutant is no evidence)


def load_core_doc(root=REPO):
    try:
        return json.load(open(os.path.join(root, CORE_STEPS_FILE), encoding="utf-8"))
    except (OSError, ValueError):
        return {"always": list(CORE_ALWAYS), "min_mutants": CORE_MIN_MUTANTS, "evidence": []}


def load_catalog(root=REPO):
    """{mutant id: module} of tools/impact/mutations.json"""
    try:
        d = json.load(open(os.path.join(root, "tools", "impact", "mutations.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {x["id"]: x.get("module", "") for x in d.get("mutations", [])}


def load_catalog_meta(root=REPO):
    """arch025-n11b: {mutant id: {"cards": [...] or None, "equivalent": text or None}} of tools/impact/mutations.json.
    cards = the cards a one-code-object mutant applies to (kernels/common.hip: gfx1201 only); equivalent = the written
    argument why no test can kill it (mutations_check enforces both fields)"""
    try:
        d = json.load(open(os.path.join(root, "tools", "impact", "mutations.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {x["id"]: {"cards": x.get("cards"), "equivalent": x.get("equivalent")} for x in d.get("mutations", [])}


def mutant_cards(meta, i):
    c = ((meta or {}).get(i) or {}).get("cards")
    return [x for x in DEVICES if x in c] if c else list(DEVICES)


def is_contract(n):
    """contract modules never get a step set: contract.* (modules.json full: true by design) and the kernel ABI"""
    return n.startswith("contract.") or n == "kernels.abi"


def evidence_modules(mods):
    """modules that may earn a GPU step set from evidence: every CORE / modules.json module except the contracts (a
    modules.json full: true module such as model.core / forward.state, and a module owning a CORE_GLOBS path such as
    ops.matmul, forces the full matrix today exactly like a CORE module)"""
    return sorted(n for n in set(CORE) | set(mods or {}) if not is_contract(n))


def core_step_sets(doc, catalog, modules=None, meta=None):
    """({module: {"r9700": [steps], "8060s": [steps], "mutants": [ids], "proof": text}}, {module: why full}) over
    `modules` (default: the CORE table). meta (load_catalog_meta): a mutant with "cards" is evidence on those cards
    only (a row on another card is ignored, like llama.cpp's NOT_SUPPORTED); a mutant with an "equivalent" argument
    is left out of the rule (it can never be killed: PRIOR_ART n11b, Trivial Compiler Equivalence)."""
    meta = meta or {}
    always = sorted(set(doc.get("always") or CORE_ALWAYS) | set(CORE_ALWAYS))
    need = max(CORE_MIN_MUTANTS, int(doc.get("min_mutants") or 0))
    newest = {}
    for r in doc.get("evidence") or []:
        k = (r.get("id"), r.get("card"))
        if k[0] in catalog and k[1] in mutant_cards(meta, k[0]) and (k not in newest or str(r.get("date", "")) >= str(newest[k].get("date", ""))):
            newest[k] = r
    ok, why = {}, {}
    for mname in (list(CORE) if modules is None else modules):
        if is_contract(mname):
            why[mname] = "contract module: always the full matrix (modules.json full: true)"
            continue
        muts = sorted(i for i, mod in catalog.items() if mod == mname and not (meta.get(i) or {}).get("equivalent"))
        if not muts:
            why[mname] = "no mutant of this module in tools/impact/mutations.json"
            continue
        problems, sets, dates = [], {}, []
        # a bug class the module must be shown to catch (doc "require": {module: {mutant id: class}}): that mutant is
        # in the catalog and KILLED by a GPU step on both cards (a compile error proves nothing about the GPU steps)
        for i, cls in sorted(((doc.get("require") or {}).get(mname) or {}).items()):
            if catalog.get(i) != mname:
                problems.append(f"required mutant {i} ({cls}) not in tools/impact/mutations.json for {mname}: write it")
                continue
            if (meta.get(i) or {}).get("equivalent"):
                problems.append(f"required mutant {i} ({cls}) is declared equivalent: a required class needs a killable mutant")
                continue
            for card in mutant_cards(meta, i):
                r = newest.get((i, card))
                if not (r and r.get("verdict") == "KILLED" and r.get("killed_by")):
                    problems.append(f"required mutant {i} ({cls}) not KILLED by a GPU step on {card}")
        for card in DEVICES:
            killers, nk = set(), 0
            for i in muts:
                if card not in mutant_cards(meta, i):
                    continue
                r = newest.get((i, card))
                if r is None:
                    problems.append(f"{i} not run on {card}")
                    continue
                v = r.get("verdict") or "?"
                if v == "KILLED" and r.get("killed_by"):
                    nk += 1
                    killers.add(r["killed_by"])
                    dates.append(str(r.get("date", "")))
                elif v != "BUILD-FAIL":
                    problems.append(f"{i} {v} on {card}" + (" (no killing step)" if v == "KILLED" else ""))
            if nk < need:
                problems.append(f"{card}: {nk} killed on the GPU (need >= {need})")
            sets[card] = sorted(killers | set(always))
        if problems:
            why[mname] = "; ".join(problems[:6]) + (f" (+{len(problems) - 6} more)" if len(problems) > 6 else "")
        else:
            ok[mname] = dict(sets, mutants=muts, proof=f"{len(muts)} mutant(s) caught on both cards ({min(dates)[:10]}..{max(dates)[:10]})")
    return ok, why


_CORE_SETS = {}


def load_core_sets(root=REPO):
    if root not in _CORE_SETS:
        try:
            mods = load_mods(root)
        except (OSError, ValueError, KeyError):
            mods = {}
        _CORE_SETS[root] = core_step_sets(load_core_doc(root), load_catalog(root), evidence_modules(mods), load_catalog_meta(root))[0]
    return _CORE_SETS[root]


def kill_rows(paths, catalog, date, source, meta=None):
    """evidence rows from gpumut.ps1 kill_<card>.tsv files (id, module, card, verdict, killed_by, ...) -> (rows, notes)"""
    rows, notes = [], []
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            i, card = (r.get("id") or "").strip(), (r.get("card") or "").strip()
            if i not in catalog:
                notes.append(f"{i}: not in tools/impact/mutations.json (row of {p} skipped)")
                continue
            if card not in DEVICES:
                notes.append(f"{i}: card {card!r} unknown (row of {p} skipped)")
                continue
            if card not in mutant_cards(meta, i):
                notes.append(f"{i}: applies to {', '.join(mutant_cards(meta, i))} only (catalog cards), row on {card} of {p} skipped")
                continue
            if (r.get("module") or catalog[i]) != catalog[i]:
                notes.append(f"{i}: module {r.get('module')} in {p}, {catalog[i]} in the catalog (catalog used)")
            rows.append({"id": i, "module": catalog[i], "card": card, "verdict": (r.get("verdict") or "").strip(),
                         "killed_by": (r.get("killed_by") or "").strip(), "date": date, "source": f"{source} ({p})"})
    return rows, notes


def now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def workspace_file(name, env):
    """a workspace data file (outside the repo: memory data-never-to-github): $env, else <workspace>/internal_docs/name"""
    if os.environ.get(env):
        return os.environ[env]
    for up in (os.path.join(REPO, ".."), os.path.join(REPO, "..", "..")):
        d = os.path.abspath(os.path.join(up, "internal_docs"))
        if os.path.isdir(d):
            return os.path.join(d, name)
    return None


def py_exe():
    return os.environ.get("WHIRL_GATE_PYTHON") or sys.executable


def pwsh():
    return os.environ.get("WHIRL_PWSH") or "pwsh"


# ---------------------------------------------------------------------------------------------------------------------
# release_gate step list and durations

_STEPS = {}


def gate_steps(device, repo=REPO):
    """[{name, gpu, na}] of release_gate.ps1 -List on this device (the gate's own table, nothing copied here)"""
    if device in _STEPS:
        return _STEPS[device]
    out = subprocess.run([pwsh(), "-NoProfile", "-File", os.path.join(repo, "tools", "gates", "release_gate.ps1"),
                          "-Device", device, "-List"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = []
    for ln in out.stdout.splitlines():
        m = re.match(r"^(\S+)\s+(CPU|GPU)\s*(?:\(not in this gate: (.*)\))?\s*$", ln)
        if m:
            rows.append({"name": m.group(1), "gpu": m.group(2) == "GPU", "na": m.group(3) or ""})
    if not rows:
        raise SystemExit(f"test_change: release_gate.ps1 -Device {device} -List printed no steps (exit {out.returncode})")
    _STEPS[device] = rows
    return rows


# arch025-procopt2 (PLAN 9.28): FLAKY rows (release_gate 9.21 retry) too; their seconds column is the passing retry's (the
# first try's seconds follow "| first try:" and are not read: a failed try is a crash or a cap, not the step's length)
ROW = re.compile(r"^(PASS|FAIL|FLAKY)\s+(\S+)\s+exit\s+-?\d+\s+(\d+) s\b")
HDR = re.compile(r"^release gate (\S+) on (r9700|8060s) ")


def parse_summary(path):
    """(tag, device, [(verdict, step, seconds)]) of a release_gate summary"""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    h = HDR.match(lines[0])
    if not h:
        return None
    rows = [(m.group(1), m.group(2), int(m.group(3))) for m in map(ROW.match, lines[1:]) if m]
    return h.group(1), h.group(2), rows


FIRST_TRY = re.compile(r"\bfirst try: exit\s+-?\d+\s+(\d+) s\b")


def first_try_seconds(path):
    """{step: seconds of its failed first try} of a release_gate summary (9.21 retry rows: FLAKY "| first try: ..." and FAIL
    "| retry also failed (first try: ...)"). A run's machine time is the row's seconds plus these (9.28)."""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return {}
    out = {}
    for l in lines[1:]:
        m, f = ROW.match(l), FIRST_TRY.search(l)
        if m and f:
            out[m.group(2)] = int(f.group(1))
    return out


# TIMING_REFERENCE.md rows for steps that no stored summary has (incremental builds; cited there with their source)
TIMING_DEFAULTS = {
    "cpu_asan": (432, "TIMING_REFERENCE: cpu_asan incremental 7.2 min (2026-10-25)"),
    "cpu_analyze": (83, "TIMING_REFERENCE: cpu_analyze with HIP objects built 1.4 min (2026-10-25)"),
    "cpu_tsa": (108, "TIMING_REFERENCE: cpu_tsa incremental 1.8 min (2026-10-26)"),
}


# arch025-procopt3 (PLAN 9.32): the steps a summary ran inside a 9.24 parallel section. release_gate.ps1 writes no per-row marker
# (rows stay identical to a serial run), only the header note "parallel: N group(s), M CPU steps, -Jobs K"; the sections are the
# runs of >= 2 consecutive par steps in row order (= -List order), par = release_gate.ps1 $parRx (keep the two in step).
PAR_RX = re.compile(r"^cpu_(sampling_iso|iso_(?!includes$)\w+)$")
PAR_NOTE = re.compile(r"parallel: (\d+) group\(s\), (\d+) CPU steps, -Jobs (\d+)")


def parallel_steps(path):
    """({steps run in a parallel section}, jobs) of a release_gate summary; (set(), 0) for a serial run or an old summary"""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return set(), 0
    m = PAR_NOTE.search(lines[0]) if lines else None
    if not m:
        return set(), 0
    out, cur = set(), []
    for st in [r.group(2) for r in map(ROW.match, lines[1:]) if r] + [None]:
        if st and PAR_RX.match(st):
            cur.append(st)
            continue
        if len(cur) >= 2:
            out.update(cur)
        cur = []
    if len(out) != int(m.group(2)):
        print(f"refresh_times: {path}: header says {m.group(2)} parallel CPU steps, rows give {len(out)} (counted as parallel anyway)")
    return out, int(m.group(3))


def refresh_times(roots, out_path, keep=5):
    """median of the newest `keep` PASS durations per (device, step) over every release-gate summary under roots; seconds from a
    parallel section (9.32) go to median_par_s / n_par / jobs, never into median_s (a step with no serial sample falls back to its
    parallel median, n = 0)"""
    files = []
    for r in roots:
        files += glob.glob(os.path.join(r, "**", "*_summary.txt"), recursive=True)
    seen, par = {}, {}
    for f in sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True):
        s = parse_summary(f)
        if not s:
            continue
        _, dev, rows = s
        pset, jobs = parallel_steps(f)
        for v, st, sec in rows:
            if v not in ("PASS", "FLAKY"):  # FLAKY: the passing retry = one clean run of the step (9.28)
                continue
            if st in pset:
                par.setdefault((dev, st), []).append((sec, f, jobs))
            else:
                seen.setdefault((dev, st), []).append((sec, f))
    rows = []
    for key in sorted(set(seen) | set(par)):
        dev, st = key
        xs, ps = seen.get(key, [])[:keep], par.get(key, [])[:keep]
        pmed = int(statistics.median([x[0] for x in ps])) if ps else ""
        pjobs = ps[0][2] if ps else ""
        if xs:
            rows.append([dev, st, int(statistics.median([x[0] for x in xs])), len(xs), xs[0][1], pmed, len(ps), pjobs])
        else:
            rows.append([dev, st, pmed, 0, ps[0][1] + " (parallel section only)", pmed, len(ps), pjobs])
    have = set(seen) | set(par)
    for dev in DEVICES:
        for st, (sec, src) in TIMING_DEFAULTS.items():
            if (dev, st) not in have:
                rows.append([dev, st, sec, 0, src, "", 0, ""])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["device", "step", "median_s", "n", "newest_source", "median_par_s", "n_par", "jobs"])
        w.writerows(rows)
    return len(rows), len(files)


def load_times(path):
    t = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
            t[(r["device"], r["step"])] = int(r["median_s"])
    return t


def step_seconds(times, dev, step):
    if (dev, step) in times:
        return times[(dev, step)]
    if not step.startswith(("cpu_", "kl_")):
        return None
    o = [v for (d, s), v in times.items() if s == step]  # a CPU step costs about the same with either -Device
    return o[0] if o else None


# ---------------------------------------------------------------------------------------------------------------------
# the change

def diff_job(a, root):
    impact.ROOT = root
    g = impact.git
    if a.commit:
        c = a.commit
        return f"commit {g('log', '-1', '--format=%h %s', c).strip()}", g("diff", "-U0", "--no-color", "-M", f"{c}^", c), c, f"{c}^", g("rev-parse", "--short", c).strip()
    if a.range:
        x, y = a.range.split("..")
        base = g("merge-base", x, y).strip() or x
        return f"range {a.range} (from merge-base {base[:9]})", g("diff", "-U0", "--no-color", "-M", base, y), y, base, f"{base[:7]}..{g('rev-parse', '--short', y).strip()}"
    head = g("rev-parse", "--short", "HEAD").strip()
    d = g("diff", "-U0", "--no-color", "-M", "HEAD")
    # untracked files are part of the change too (git diff HEAD does not list them): one hunk over the whole file
    for f in g("ls-files", "--others", "--exclude-standard").splitlines():
        try:
            nl = sum(1 for _ in open(os.path.join(root, f), "rb"))
        except OSError:
            continue
        d += f"\ndiff --git a/{f} b/{f}\nnew file\n@@ -0,0 +1,{max(nl, 1)} @@\n"
    return "worktree (tracked changes + untracked files)", d, None, "HEAD", f"{head}+worktree"


def load_map(root):
    p = os.path.join(root, "tools", "impact_map.json")
    try:
        m = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        m = None
    if m is None or m.get("tree_hash") != impact_build.tree_hash(root, impact_build.list_files(root)):
        print("test_change: tools/impact_map.json missing or stale: rebuilding it in memory", file=sys.stderr)
        m = impact_build.build(root)
    return m


def load_mods(root):
    return json.load(open(os.path.join(root, "tools", "gates", "modules.json"), encoding="utf-8"))["modules"]


def hip_closure(m, files):
    """files whose change can alter a code object: a header a .hip file includes (transitively, impact map includers)"""
    inc = (m or {}).get("includers", {})
    out = []
    for f in files:
        if f.endswith((".h", ".hpp", ".inc")) and any(x.endswith(".hip") for x in inc.get(f, [])):
            out.append(f)
    return out


def matches(path, globs):
    return any(fnmatch.fnmatch(path, g) for g in globs)


# ---------------------------------------------------------------------------------------------------------------------
# arch025-modsel2 (PLAN 3.6, X12 Finding K8-3, GPU half): which release_gate steps run a gate script. A changed script
# that a GPU step runs selects that step (on --one-gpu: impact.py FILE_RULES "the changed step on one GPU"); a script used
# by release_gate.ps1 outside every AddStep statement (the runner loop, e.g. server_log_scan.ps1; a helper function, e.g.
# env_matrix.ps1), or a release_gate.ps1 line outside every AddStep statement, is common code: every GPU step on one GPU.
# A script the gate never reaches (directly, by a variable, or through another gate script) selects no GPU step.

SCRIPT_RX = re.compile(r"([A-Za-z0-9_.-]+\.(?:ps1|py))\b")
# used by release_gate.ps1 only to CHOOSE steps (-Module / -ImpactRange), never inside a step; its own tests run in
# cpu_modules_check, so a change does not alter how any GPU step runs
SELECTION_ONLY = {"impact.py"}
DOCQ = '"' * 3


def code_text(txt):
    """script text without # / <# comment lines and python docstring lines (names mentioned there are not calls)"""
    out, doc = [], False
    for l in txt.splitlines():
        s = l.strip()
        if s.startswith((DOCQ, "'''")):
            if not (len(s) > 3 and s.endswith(s[:3])):
                doc = not doc
            continue
        if doc or s.startswith(("#", "<#")):
            continue
        out.append(l)
    return "\n".join(out)


EXE_RX = (re.compile(r'\$exe\[\s*"(whirl[\w-]*)"\s*\]'), re.compile(r"(?<![\w-])(whirl(?:-[a-z0-9]+)*)\.exe(?![\w-])", re.I))


def paren_open(txt):
    """True while txt (PowerShell, quoted strings ignored) opens more parentheses than it closes"""
    t = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', "", txt)
    return t.count("(") > t.count(")")


def exe_names(txt):
    """WHIRL executables a piece of gate text starts: $exe["name"] (release_gate.ps1's table) or name.exe"""
    return {x.lower() for rx in EXE_RX for x in rx.findall(txt)}


def gate_script_map(ps_text, others=None):
    """release_gate.ps1 text (+ {basename: text} of the other gate scripts) -> {"refs": {script: {step name globs}},
    "common": {script}, "stmts": [(first line, last line, name glob)], "neutral": {blank / comment line numbers},
    "exes": {executable: {step name globs that start it}}}"""
    lines = ps_text.splitlines()
    var, evar, fexe = {}, {}, {}
    for l in lines:
        m = re.match(r"^\s*\$(\w+)\s*=(.*)$", l)
        if m and SCRIPT_RX.search(m.group(2)):
            var[m.group(1)] = set(SCRIPT_RX.findall(m.group(2)))
        if m and exe_names(m.group(2)):
            evar[m.group(1)] = exe_names(m.group(2))
    # arch025-modsel3 (PLAN 3.7): the executables a function of the gate starts (GateCmd, EnvCmd): its body's $exe["..."]
    k = 0
    while k < len(lines):
        m = re.match(r"^\s*function\s+(\w+)", lines[k])
        if not m:
            k += 1
            continue
        depth, j, body = 0, k, []
        while j < len(lines):
            body.append(lines[j])
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0 and "{" in "".join(body):
                break
            j += 1
        btxt = " ".join(body)
        fexe[m.group(1)] = (exe_names(btxt), set(SCRIPT_RX.findall(btxt)) - {"release_gate.ps1"})  # (exes, scripts)
        k = j + 1
    stmts, in_stmt, used_vars = [], set(), set()
    i = 0
    while i < len(lines):
        m = re.match(r'^\s*AddStep\s+"([^"]+)"', lines[i])
        if not m:
            i += 1
            continue
        j = i
        # a statement continues on the next line after a trailing backtick or while a "(" is still open (golden_<m>:
        # its argument list spans two lines)
        while j + 1 < len(lines) and (lines[j].rstrip().endswith("`") or paren_open("\n".join(lines[i:j + 1]))):
            j += 1
        text = "\n".join(lines[i:j + 1])
        name = re.sub(r"\$\((?:[^()]|\([^()]*\))*\)|\$\{\w+\}|\$\w+", "*", m.group(1))
        refs = set(SCRIPT_RX.findall(text))
        for v in re.findall(r"\$(\w+)", text):
            if v in var:
                refs |= var[v]
                used_vars.add(v)
        exes = exe_names(text)
        for v in re.findall(r"\$(\w+)", text):
            exes |= evar.get(v, set())
        fsc = set()  # scripts the functions this step calls start (EnvCmd -> env_matrix.ps1): their exes count below
        for f, (es, sc) in fexe.items():
            if re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", text):
                exes |= es
                fsc |= sc
        stmts.append((i + 1, j + 1, name, refs, (exes, fsc)))
        in_stmt.update(range(i + 1, j + 2))
        i = j + 1
    refs = {}
    for _, _, name, rs, _ in stmts:
        for b in rs:
            refs.setdefault(b, set()).add(name)
    common, neutral = set(), set()
    for k, l in enumerate(lines, 1):
        s = l.strip()
        if not s or s.startswith("#"):
            neutral.add(k)
            continue
        if k in in_stmt:
            continue
        m = re.match(r"^\s*\$(\w+)\s*=", l)
        if m and m.group(1) in used_vars and m.group(1) in var:
            continue  # the assignment of a variable an AddStep uses: counted with that step
        common |= set(SCRIPT_RX.findall(l))
        for v in re.findall(r"\$(\w+)", l):
            if v in var and not (m and m.group(1) == v):
                common |= var[v]
    common.discard("release_gate.ps1")
    common -= SELECTION_ONLY
    for _ in range(2):  # through other gate scripts (golden.ps1 -> tools/golden/*, server_log_scan.ps1 ...)
        for b, txt in (others or {}).items():
            for c in set(SCRIPT_RX.findall(code_text(txt))) - {b} - SELECTION_ONLY:
                if b in refs:
                    refs.setdefault(c, set()).update(refs[b])
                if b in common:
                    common.add(c)
    # executables started by a gate script a step calls (golden.ps1 -> whirl-server-gate.exe, env_matrix.ps1 -> ...),
    # through other gate scripts too (two levels, as refs above)
    sexe = {b: exe_names(code_text(t)) for b, t in (others or {}).items()}
    for _ in range(2):
        for b, t in (others or {}).items():
            for c in set(SCRIPT_RX.findall(code_text(t))) - {b}:
                sexe[b] = sexe[b] | sexe.get(c, set())
    exes = {}
    for _, _, name, rs, (es, fsc) in stmts:
        for e in es | {x for b in rs | fsc for x in sexe.get(b, set())}:
            exes.setdefault(e, set()).add(name)
    return {"refs": refs, "common": common, "stmts": [(a, b, n) for a, b, n, _, _ in stmts], "neutral": neutral, "exes": exes}


def gate_line_steps(gm, line_nos):
    """step name globs of the AddStep statements that hold these release_gate.ps1 lines; None = a line is common code"""
    out = set()
    for k in line_nos:
        if k in gm["neutral"]:
            continue
        hit = [n for a, b, n in gm["stmts"] if a <= k <= b]
        if not hit:
            return None
        out |= set(hit)
    return out


_GATE_MAP = {}


def load_gate_map(root=REPO):
    if root not in _GATE_MAP:
        def rd(p):
            try:
                return open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                return ""
        others = {}
        for d in ("tools/gates", "tools/golden", "tools/safety", "tools/klgate"):
            for p in glob.glob(os.path.join(root, d, "*.ps1")) + glob.glob(os.path.join(root, d, "*.py")):
                if os.path.basename(p) != "release_gate.ps1":
                    others[os.path.basename(p)] = rd(p)
        _GATE_MAP[root] = gate_script_map(rd(os.path.join(root, "tools", "gates", "release_gate.ps1")), others)
    return _GATE_MAP[root]


def gate_script_steps(files, changes, gm):
    """{changed script: step name globs, or "ALL" (common code)} for the changed gate scripts"""
    out = {}
    for f in files:
        if not f.startswith("tools/"):
            continue
        b = f.rsplit("/", 1)[-1]
        if f == "tools/gates/release_gate.ps1":
            ch = changes.get(f) or {}
            new, old = ch.get("new") or [], ch.get("old") or []
            st = None if (not new and not old) or len(old) > len(new) else gate_line_steps(gm, new)
            out[f] = "ALL" if st is None else st
        elif b in gm["common"]:
            out[f] = "ALL"
        elif b in gm["refs"]:
            out[f] = set(gm["refs"][b])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# arch025-modsel3 (PLAN 3.7): a changed tests/ source selects the GPU steps that run the executable(s) built from it. The
# impact map's `executables` (CMake target -> its sources) names the executables; a header under tests/ counts for the
# executables of every file that includes it (map includers, transitively). Same class as K8-3 (gate scripts).
ARTIFACT = {"whirl-cli": "whirl"}  # CMake target -> the exe name release_gate.ps1 uses (OUTPUT_NAME, cmake/model.cmake)
# executables whose GPU steps run on both GPUs when only a test source changed: the kernel test checks each GPU's own
# kernels (gfx1151-only kernels such as attn_ws4 have their tests in the same sources), so one GPU would leave the other
# GPU's kernels untested; every other test executable (whirl-server-gate suites) runs on --one-gpu (impact FILE_RULES
# "the changed suite on one GPU")
EXE_BOTH_GPUS = {"whirl-kernel-test": "kernel tests check each GPU's own kernels"}
TEST_SRC_SKIP = ["tests/golden/*"]  # golden references: impact FILE_RULES (golden on both GPUs via the map)


def test_source_exes(files, m):
    """{changed tests/ file: [exe names (release_gate.ps1 names) built from it]} (headers through includers)"""
    exes = (m or {}).get("executables") or {}
    inc = (m or {}).get("includers") or {}
    out = {}
    for f in files:
        if not f.startswith("tests/") or matches(f, TEST_SRC_SKIP):
            continue
        seen, todo, srcs = {f}, [f], set()
        while todo:
            x = todo.pop()
            srcs.add(x)
            for y in inc.get(x, []):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        es = sorted({ARTIFACT.get(t, t) for t, fl in exes.items() if srcs & set(fl)})
        if es:
            out[f] = es
    return out


# ---------------------------------------------------------------------------------------------------------------------
# arch025-modsel3 (PLAN 3.8): a non-list CMake edit (impact.py: build-wide) is attributed through the CMake build graph
# of both revisions (tools/impact/cmake_graph.py: configure each, diff the Ninja action graphs). The graph's changed
# sources replace the CMake file as whole-file changes for impact.analyze (module / core / map rules as for an edit of
# that source); the GPU test executables it relinks select their steps; whatever the graph cannot attribute is full.
GRAPH_FILES = ["CMakeLists.txt", "cmake/*.cmake"]
NOT_GRAPHED = ["build.bat", "redist/*", "tools/package_release.ps1"]  # build-wide files the CMake graph does not cover


def file_lines(root, rev, path):
    """line count of path at rev (None = the working tree of root); 0 when it does not exist there"""
    if rev is None:
        try:
            return sum(1 for _ in open(os.path.join(root, path), "rb"))
        except OSError:
            return 0
    t = impact.show(rev, path)
    return 0 if t is None else len(t.splitlines())


def graph_rep(changes, rep, m, root, old_rev, new_rev, work, gm, runner=None, analyze_fn=None, lines_fn=None):
    """(new impact report or None, graph result or None): None, None when no non-list CMake file changed or a build-wide
    file the graph does not cover changed too (full anyway); the report is recomputed only when the graph is ok"""
    lists = rep.get("cmake_lists") or {}
    cm = [f for f in changes if matches(f, GRAPH_FILES) and f not in lists]
    if not cm or any(matches(f, NOT_GRAPHED) for f in changes):
        return None, None
    if runner is None:
        import cmake_graph
        runner = cmake_graph.analyze_revs
    g = runner(root, old_rev, new_rev, work, sorted(gm.get("exes") or {}))
    if not g.get("ok"):
        return None, g
    lines_fn = lines_fn or (lambda rev, f: file_lines(root, rev, f))
    ch2 = {k: v for k, v in changes.items() if k not in cm}
    for srcs in g.get("sources", {}).values():
        for f in srcs:
            if f not in ch2:
                ch2[f] = {"new": list(range(1, lines_fn(new_rev, f) + 1)), "old": list(range(1, lines_fn(old_rev, f) + 1)),
                          "old_path": f, "cmake_graph": True}
    r2 = (analyze_fn or impact.analyze)(ch2, m, new_rev, old_rev)
    r2["cmake_graph_files"] = cm
    return r2, g


def graph_needed(p):
    """a first plan (without the CMake graph) is full ONLY because of a build-wide CMake file: the graph can narrow it.
    Full for any other reason too (core / contract / product file in no module): the graph would not change the plan."""
    rs = p["full_reasons"]
    return bool(rs) and all(reason_class(x) == "impact: build-wide file" for x in rs)


def plan_with_graph(title, cid, changes, rep, mods, m, gates, times, root, old_rev, new_rev, work, use_graph=True, **kw):
    """make_plan, and when the plan is full only because of a non-list CMake edit, again with the CMake build graph
    (graph_rep); the second plan replaces the first (its full reasons say what the graph could not attribute)"""
    p = make_plan(title, cid, changes, rep, mods, m, gates, times, **kw)
    if not use_graph or not graph_needed(p):
        return p
    gm = kw.get("gate_map") if kw.get("gate_map") is not None else load_gate_map(root)
    r2, g = graph_rep(changes, rep, m, root, old_rev, new_rev, work, gm)
    if g is None:
        return p
    return make_plan(title, cid, changes, r2 if r2 is not None else rep, mods, m, gates, times, cmake_graph=g, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# plan

def expand_gpu(names, gates):
    """map / module step names -> GPU step names that exist (not na) per device"""
    out = {d: [] for d in DEVICES}
    for d in DEVICES:
        ok = {s["name"] for s in gates[d] if s["gpu"] and not s["na"]}
        for n in names:
            if n in ok and n not in out[d]:
                out[d].append(n)
    return out


def check_proof(path, dev, build):
    """(ok, why, covered steps) of a work_equal report for this device and build"""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"cannot read: {e}", set()
    if r.get("format") != "whirl-work-equal/1":
        return False, "not a work_equal report (format)", set()
    if r.get("verdict") != "EQUIVALENT":
        return False, f"verdict {r.get('verdict')} (only EQUIVALENT skips GPU steps)", set()
    chk = r.get("checks", {})
    if chk.get("code_objects") not in ("identical", "identical-except-cuid") or chk.get("launch_sequence") != "identical":
        return False, f"checks {chk}", set()
    if not build:
        return False, "no --build: the proof cannot be bound to the exe under test", set()
    for exe in PROOF_EXES:
        p = os.path.join(build, exe)
        want = r.get("b", {}).get("exes", {}).get(exe)
        if os.path.exists(p) and want:
            if hashlib.sha256(open(p, "rb").read()).hexdigest() != want:
                return False, f"{exe} of --build is not the B exe of the proof", set()
        elif exe == "whirl.exe":
            return False, f"{exe}: missing in --build or not in the proof", set()
    arch = ARCH[dev]
    cov = set()
    for L in r.get("launch") or []:
        for run in L.get("runs") or []:
            k = run.get("key", "") if isinstance(run, dict) else ""
            if f"--device {arch}" in k or not re.search(r"--device gfx\d+", k):
                cov.add(k.split("/", 1)[0])
            if any(f"--device {a}" in k for a in ARCH.values() if a != arch):
                return False, f"launch run for another GPU in the proof: {k[:80]}", set()
    if chk.get("golden") == "identical":
        cov |= {"golden_swift", "golden_moe"}
    cov.add("kt_full")  # identical code objects: the kernel test runs the same kernels
    return True, "EQUIVALENT, bound to --build", cov


def compare_code_objects(base, build, out_json):
    """work_equal.py compare of the exes (code objects only, CPU): 'identical' / 'identical-except-cuid' / other / None"""
    args = [py_exe(), "-I", os.path.join(REPO, "tools", "impact", "work_equal.py"), "compare", "--allow-cuid-length", "-o", out_json]
    n = 0
    for exe in PROOF_EXES:
        if os.path.exists(os.path.join(base, exe)) and os.path.exists(os.path.join(build, exe)):
            args += ["--a-exe", os.path.join(base, exe), "--b-exe", os.path.join(build, exe)]
            n += 1
    if not n:
        return None, "no exe pair found"
    subprocess.run(args, capture_output=True, text=True)
    try:
        r = json.load(open(out_json, encoding="utf-8"))
    except (OSError, ValueError):
        return None, "work_equal.py wrote no report"
    return r.get("checks", {}).get("code_objects"), out_json


def no_gpu_file(f, mods):
    """arch025-n11b: f is owned (modules.json paths) only by NO_GPU_MODULES modules"""
    owners = [k for k, d in (mods or {}).items() if matches(f, (d or {}).get("paths") or [])]
    return bool(owners) and all(k in NO_GPU_MODULES for k in owners)


def make_plan(title, cid, changes, rep, mods, m, gates, times, one_gpu="r9700", build=None, base_build=None, proofs=None,
              scratch=None, all_cpu=False, gate_map=None, cmake_graph=None, core_sets=None):
    files = sorted(changes)
    gm = gate_map if gate_map is not None else load_gate_map()
    graph_src = sorted({f for v in ((cmake_graph or {}).get("sources") or {}).values() for f in v})
    product = [f for f in files if f.startswith(("src/", "include/", "kernels/"))]
    modules = dict(rep.get("modules") or {})
    callers = {k: list(v) for k, v in (rep.get("module_callers") or {}).items()}
    gate_txt = rep.get("module_gate") or ""
    # arch025-n11: core modules with a GPU mutation evidence set (core_step_sets); contract.* never
    cset = {k: v for k, v in (load_core_sets() if core_sets is None else core_sets).items() if not is_contract(k)}
    core_used = {}
    full = []  # reasons
    for n in sorted(modules):
        if n in CORE:
            if n in cset:
                core_used[n] = cset[n]
            else:
                full.append(f"core module {n}: {CORE[n]}")
    for n, cs in sorted(callers.items()):  # rdeps: a changed header of n is compiled into these modules
        for k in cs:
            if k in CORE:
                if k in cset:
                    core_used[k] = cset[k]
                else:
                    full.append(f"core module {k} includes a changed header of {n} (callers): {CORE[k]}")
    for f in files:
        for g, why in CORE_GLOBS:
            if fnmatch.fnmatch(f, g):
                owners = [k for k, fl in modules.items() if f in (fl or [])]
                if owners and all(k in NO_GPU_MODULES for k in owners):
                    break  # arch025-n11b: a core path owned only by no-GPU modules (device.null: the CPU null backend) is not core
                if owners and all(k in cset for k in owners):
                    for k in owners:
                        core_used[k] = cset[k]
                    break  # a core path inside an evidenced module: that module's set covers it
                full.append(f"core file {f}: {why}")
                break
    # arch025-modsel2 (PLAN 3.3): a list-only CMake edit counts as the modules of the sources it names, so a named
    # source on a core path is a core change too
    lists = rep.get("cmake_lists") or {}
    for cf, srcs in sorted(lists.items()):
        for f in srcs:
            if no_gpu_file(f, mods):
                continue  # arch025-n11b: a source of a no-GPU module only (device.null)
            for g, why in CORE_GLOBS:
                if fnmatch.fnmatch(f, g):
                    full.append(f"core file {f} listed in {cf}: {why}")
                    break
    if cmake_graph is not None:
        if not cmake_graph.get("ok"):
            full.append(f"CMake graph: cannot attribute the CMake edit ({cmake_graph.get('why', '?')})")
        for x in cmake_graph.get("full") or []:
            full.append(f"CMake graph: {x}")
        for f in graph_src:
            for g, why in CORE_GLOBS:
                if fnmatch.fnmatch(f, g):
                    full.append(f"core file {f} built differently (CMake graph): {why}")
                    break
    if gate_txt.startswith("full gate") and "cross-module" not in gate_txt:
        # arch025-n11: full only because of modules.json full: true modules that all have an evidence set -> impact.py's
        # module gate again without their full flag (a file / includer in no module or a build-wide file keeps it full)
        touched = set(modules) | {k for v in callers.values() for k in v}
        fl = sorted(n for n in touched if (mods.get(n) or {}).get("full"))
        g2 = gate_txt
        if fl and "contract module" in gate_txt and all(n in cset for n in fl):
            mods2 = {k: (dict(v, full=False) if k in fl else v) for k, v in mods.items()}
            g2 = impact.module_gate(changes, mods2, rep, m).get("module_gate") or ""
        if g2.startswith("full gate") and "cross-module" not in g2:
            full.append(f"impact.py: {g2}")  # contract / build-wide / product file in no module / includer in no module
        else:
            for n in fl:
                core_used[n] = cset[n]
    kernel = [f for f in files if matches(f, KERNEL_GLOBS)] + hip_closure(m, files)
    bfiles = [f for f in files if matches(f, BUILD_GLOBS) and f not in lists]  # a source-list edit changes no flags
    weq = {"code_objects": "not compared (no --base-build)", "static": (
        f"kernel sources touched: {', '.join(kernel[:4])}" if kernel else
        f"build files touched (code objects may change; a binary proof decides): {', '.join(bfiles[:4])}" if bfiles else
        "no kernel source, no header a .hip file includes and no build file changed: code objects cannot change (source closure)")}
    if base_build and build and scratch:
        co, src = compare_code_objects(base_build, build, os.path.join(scratch, f"code_objects_{cid.replace('+', '_').replace('..', '_')}.json"))
        weq["code_objects"] = f"{co} ({src})"
        if co not in ("identical", "identical-except-cuid") and not kernel:
            full.append(f"work equivalence cannot be shown: code objects {co} although no kernel source changed ({src})")
    # GPU selection
    names = set()
    for n in list(modules) + [k for v in callers.values() for k in v]:
        d = mods.get(n, {})
        names |= set(d.get("steps") or [])
    for n in modules:
        for f, sts in (mods.get(n, {}).get("when") or {}).items():
            if f in changes:
                names |= set(sts)
    map_gpu = impact.expand_steps(rep.get("release_gate_steps", []))
    map_devs = [DEV_OF_ARCH[a] for a in rep.get("gpus") or ARCH.values() if a in DEV_OF_ARCH] or list(DEVICES)
    gpu = {d: [] for d in DEVICES}
    why = {}
    if full:
        devs = list(DEVICES)
        kfiles = [f for f in product if matches(f, ["kernels/*", "*.hip"])]
        if (kfiles and kfiles == product and len(map_devs) == 1 and not modules
                and not any(matches(f, ["cmake/*", "CMakeLists.txt", "build.bat", "redist/*"]) for f in files)
                and not any(r.startswith("work equivalence") for r in full)):
            devs = map_devs  # every core reason is an arch-specific kernel (testing.md: one kernel -> its GPU)
        for d in devs:
            gpu[d] = [s["name"] for s in gates[d] if s["gpu"] and not s["na"]]
            for s in gpu[d]:
                why[(d, s)] = "full matrix"
        level = "full"
    else:
        for n in sorted(set(modules) | {k for v in callers.values() for k in v}):
            ev = core_used.get(n) or cset.get(n)
            if ev:  # arch025-n11: the module's GPU mutation evidence set, both cards
                for dv in DEVICES:
                    for st in ev.get(dv) or []:
                        if st in expand_gpu([st], gates)[dv] and st not in gpu[dv]:
                            gpu[dv].append(st)
                            why[(dv, st)] = f"core module {n}: GPU mutation evidence ({', '.join(ev.get('mutants') or [])})"
                if n in CORE or (mods.get(n) or {}).get("full"):
                    continue  # a core / full: true module: the evidence set replaces its modules.json GPU steps
            d = mods.get(n, {})
            g2 = d.get("gpus") or {}
            for st in d.get("steps") or []:
                for dv in (DEVICES if g2.get(st, 1) >= 2 else (one_gpu,)):
                    if st in expand_gpu([st], gates)[dv] and st not in gpu[dv]:
                        gpu[dv].append(st)
                        why[(dv, st)] = f"module {n}"
        for n in modules:
            for f, sts in (mods.get(n, {}).get("when") or {}).items():
                if f in changes:
                    for dv in DEVICES:
                        for st in expand_gpu(sts, gates)[dv]:
                            if st not in gpu[dv]:
                                gpu[dv].append(st)
                                why[(dv, st)] = f"module {n} when {f}"
        for dv in map_devs:
            for st in expand_gpu(sorted(map_gpu), gates)[dv]:
                if st not in gpu[dv]:
                    gpu[dv].append(st)
                    why[(dv, st)] = "impact map"
        ok1 = [s["name"] for s in gates[one_gpu] if s["gpu"] and not s["na"]]
        # arch025-modsel3 (PLAN 3.7 / 3.8): test sources -> the GPU steps of the executables built from them; GPU test
        # executables the CMake graph relinks -> their GPU steps
        exe_sel = {}
        for f, es in sorted(test_source_exes(files + [x for x in graph_src if x.startswith("tests/")], m).items()):
            for e in es:
                exe_sel.setdefault(e, f"test source {f} of {e}")
        for e, ws in sorted(((cmake_graph or {}).get("exe_changes") or {}).items()):
            exe_sel.setdefault(e, f"{e} relinked (CMake graph: {'; '.join(ws)[:60]})")
        for e, w in sorted(exe_sel.items()):
            globs = (gm.get("exes") or {}).get(e) or set()
            for dv in (DEVICES if e in EXE_BOTH_GPUS else (one_gpu,)):
                okd = [s["name"] for s in gates[dv] if s["gpu"] and not s["na"]]
                for st in okd:
                    if any(fnmatch.fnmatch(st, g) for g in globs) and st not in gpu[dv]:
                        gpu[dv].append(st)
                        why[(dv, st)] = w
        for f, sel in sorted(gate_script_steps(files, changes, gm).items()):
            for st in ok1:
                if (sel == "ALL" or any(fnmatch.fnmatch(st, g) for g in sel)) and st not in gpu[one_gpu]:
                    gpu[one_gpu].append(st)
                    why[(one_gpu, st)] = f"gate script {f}" + (" (common code: every GPU step)" if sel == "ALL" else "")
        level = "module" if any(gpu.values()) else ("cpu" if not product else "cpu (product code, no GPU step selected)")
        touched = set(modules) | {k for v in callers.values() for k in v}
        if level.startswith("cpu (product") and not (touched and touched <= set(NO_GPU_MODULES)):
            full.append(f"product code selected no GPU step (modules {', '.join(sorted(touched)) or '-'} not in NO_GPU_MODULES; K8-7)")
            for d in DEVICES:
                gpu[d] = [s["name"] for s in gates[d] if s["gpu"] and not s["na"]]
                for s in gpu[d]:
                    why[(d, s)] = "full matrix"
            level = "full"
    # work-equivalence short-cut
    skipped = {d: [] for d in DEVICES}
    proof_notes = {}
    for dv, p in (proofs or {}).items():
        if kernel:
            proof_notes[dv] = f"{p}: not used (kernel sources touched: {', '.join(kernel[:3])})"
            continue
        ok, w, cov = check_proof(p, dv, build)
        proof_notes[dv] = f"{p}: {w}"
        if ok:
            keep = []
            for st in gpu[dv]:
                (skipped[dv] if st in cov else keep).append(st)
            gpu[dv] = keep
            proof_notes[dv] += f"; skips {len(skipped[dv])} GPU step(s) on {dv}"
    # order steps as the gate lists them
    for dv in DEVICES:
        order = [s["name"] for s in gates[dv]]
        gpu[dv].sort(key=order.index)
    cpu = [s["name"] for s in gates[DEVICES[0]] if not s["gpu"] and not s["na"] and s["name"] not in PER_DEVICE_CPU]
    no_input = bool(files) and not all_cpu and all(matches(f, NO_SAFETY_INPUT) for f in files)
    skipped_cpu = [s for s in cpu if s in SAFETY_STEPS] if no_input else []
    cpu = [s for s in cpu if s not in skipped_cpu]
    est = {}
    unknown = []
    cpu_s = sum(step_seconds(times, DEVICES[0], s) or 0 for s in cpu)
    unknown += [s for s in cpu if step_seconds(times, DEVICES[0], s) is None]
    for dv in DEVICES:
        cpu_s += sum(step_seconds(times, dv, s) or 0 for s in PER_DEVICE_CPU)
        g = 0
        for s in gpu[dv]:
            v = step_seconds(times, dv, s)
            if v is None:
                unknown.append(f"{s}@{dv}")
            g += v or 0
        est[dv] = g
    full_s = {dv: sum(step_seconds(times, dv, s["name"]) or 0 for s in gates[dv] if s["gpu"] and not s["na"]) for dv in DEVICES}
    return {
        "change": title, "id": cid, "created": now(), "files": files, "product_files": product,
        "modules": modules, "callers": callers, "impact_module_gate": gate_txt, "level": level, "full_reasons": full,
        "cpu_steps": cpu, "per_device_cpu": list(PER_DEVICE_CPU),
        "cpu_skipped": {"steps": skipped_cpu, "why": ("no changed file is an input of the C++ safety steps (NO_SAFETY_INPUT); "
                                                      "the release gate runs them") if skipped_cpu else ""}, "gpu_steps": gpu, "gpu_why": {f"{d}:{s}": w for (d, s), w in why.items()},
        "skipped_by_proof": skipped, "proofs": proof_notes, "work_equivalence": weq,
        "est_cpu_min": round(cpu_s / 60, 1), "est_gpu_min": {d: round(v / 60, 1) for d, v in est.items()},
        "est_full_gpu_min": {d: round(v / 60, 1) for d, v in full_s.items()}, "est_unknown": unknown,
        "one_gpu": one_gpu, "cmake_graph": cmake_graph,
        "core_sets": {k: {"mutants": v.get("mutants"), "proof": v.get("proof")} for k, v in sorted(core_used.items())},
    }


def print_plan(p):
    print(f"== test what you changed: {p['change']}")
    print(f"   files    {len(p['files'])} changed, {len(p['product_files'])} product")
    print(f"   modules  {', '.join(sorted(p['modules'])) or '(none)'}" + (f"; callers {p['callers']}" if p["callers"] else ""))
    print(f"   level    {p['level'].upper()}" + ("".join(f"\n            - {r}" for r in p["full_reasons"][:8]) if p["full_reasons"] else "")
          + (f"\n            ... {len(p['full_reasons']) - 8} more" if len(p["full_reasons"]) > 8 else ""))
    for k, v in (p.get("core_sets") or {}).items():
        print(f"   core     {k}: own GPU step set from mutation evidence ({v.get('proof')}; {', '.join(v.get('mutants') or [])})")
    print(f"   CPU      all {len(p['cpu_steps'])} CPU steps once + {', '.join(p['per_device_cpu'])} on both GPUs: ~{p['est_cpu_min']} min [est]")
    if p.get("cpu_skipped", {}).get("steps"):
        print(f"            not run: {', '.join(p['cpu_skipped']['steps'])} ({p['cpu_skipped']['why']}; --all-cpu runs them)")
    for d in DEVICES:
        st = p["gpu_steps"][d]
        print(f"   GPU {d:<5} {len(st)} step(s), ~{p['est_gpu_min'][d]} min [est] (full matrix ~{p['est_full_gpu_min'][d]} min)"
              + (f": {', '.join(st)}" if st and p["level"] != "full" else ""))
        if p["skipped_by_proof"][d]:
            print(f"            skipped by work-equivalence proof: {', '.join(p['skipped_by_proof'][d])}")
    for d, n in p["proofs"].items():
        print(f"   proof    {d}: {n}")
    print(f"   work-eq  {p['work_equivalence']['static']}; code objects: {p['work_equivalence']['code_objects']}")
    g = p.get("cmake_graph")
    if g:
        srcs = sorted({f for v in (g.get("sources") or {}).values() for f in v})
        print(f"   cmake    build graph ({' vs '.join(map(str, g.get('configured', [])))}): " + (
            f"cannot attribute ({g.get('why')})" if not g.get("ok") else
            f"{len(srcs)} source(s) built differently{': ' + ', '.join(srcs[:6]) if srcs else ''}"
            f"{'; relinked ' + ', '.join(sorted(g.get('exe_changes') or {})) if g.get('exe_changes') else ''}"
            f"{'; ' + str(len(g.get('full') or [])) + ' unattributable' if g.get('full') else ''}"))
        for x in (g.get("notes") or [])[:3]:
            print(f"            note: {x}")
    if p["est_unknown"]:
        print(f"   note     no stored duration for: {', '.join(p['est_unknown'][:8])} (counted 0; --refresh-times)")
    print("   note     dev-time only: releases / rc merges / bug-fix releases run the full release gate on both GPUs")


# ---------------------------------------------------------------------------------------------------------------------
# run

def whirl_running():
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    return sorted({ln.split(",")[0].strip('"') for ln in out.splitlines() if ln.lower().startswith('"whirl')})


def run_plan(p, a):
    work = a.work or os.path.join(REPO, "..", "scratch", "test_change", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    work = os.path.abspath(work)
    os.makedirs(work, exist_ok=True)
    busy = whirl_running()
    if busy and not a.force:
        print(f"test_change: refused: whirl processes running ({', '.join(busy)}); one GPU job at a time (--force overrides)")
        return 2
    tagid = re.sub(r"[^0-9A-Za-z]+", "_", p["id"])[:40]
    runs = []
    first = True
    for dev in DEVICES:  # one GPU after the other
        steps = (p["cpu_steps"] if first else []) + list(PER_DEVICE_CPU) + p["gpu_steps"][dev]
        first = False
        tag = f"change_{tagid}_{dev}"
        cmd = [pwsh(), "-NoProfile", "-File", os.path.join(REPO, "tools", "gates", "release_gate.ps1"), "-Device", dev,
               "-Steps", ",".join(steps), "-Tag", tag, "-Work", os.path.join(work, dev), "-LaunchLog", os.path.join(work, f"launch_{dev}")]
        if a.build:
            cmd += ["-Build", a.build]
        print(f"test_change: {dev}: {len(steps)} step(s) -> {tag}", flush=True)
        t0 = time.time()
        with open(os.path.join(work, f"{tag}.out.txt"), "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        summ = os.path.join(work, dev, f"{tag}_summary.txt")
        s = parse_summary(summ)
        rows = s[2] if s else []
        ft = first_try_seconds(summ) if s else {}  # 9.28: a retried step cost both tries
        runs.append({"device": dev, "tag": tag, "exit": rc, "wall_s": round(time.time() - t0), "summary": summ if s else None,
                     "cpu_s": sum(x[2] + ft.get(x[1], 0) for x in rows if x[1].startswith(("cpu_", "kl_"))),
                     "gpu_s": sum(x[2] + ft.get(x[1], 0) for x in rows if not x[1].startswith(("cpu_", "kl_"))),
                     "failed": [x[1] for x in rows if x[0] == "FAIL"],
                     "flaky": [x[1] for x in rows if x[0] == "FLAKY"]})
        print(f"test_change: {dev}: exit {rc}, {len(rows)} row(s), failed {runs[-1]['failed']}", flush=True)
        if rc == 2:
            break
    rc = 2 if any(r["exit"] == 2 for r in runs) else (1 if any(r["exit"] for r in runs) else 0)
    res = {"plan": p, "runs": runs, "exit": rc, "result": {0: "PASS", 1: "FAIL", 2: "SETUP"}[rc], "work": work, "ended": now()}
    with open(os.path.join(work, "test_change_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(work, "test_change_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"test_change {p['change']} level {p['level']} result {res['result']}\n")
        for r in runs:
            fh.write(f"{r['device']} {r['tag']} exit {r['exit']} cpu {r['cpu_s']} s gpu {r['gpu_s']} s failed {','.join(r['failed']) or '-'} summary {r['summary']}\n")
    ledger(a, p, runs, res["result"], os.path.join(work, "test_change_summary.json"))
    print(f"test_change: {res['result']} (summary {work}\\test_change_summary.txt)")
    return rc


LEDGER_COLS = ["ts", "kind", "change", "id", "modules", "level", "cpu_steps", "gpu_steps_r9700", "gpu_steps_8060s",
               "est_cpu_min", "est_gpu_min", "est_full_gpu_min", "cpu_min", "gpu_min", "result", "summary"]


def ledger(a, p, runs, result, summary):
    path = a.ledger or workspace_file("TEST_COST_LEDGER.tsv", "WHIRL_TEST_COST_LEDGER")
    if not path:
        print("test_change: no ledger (give --ledger or WHIRL_TEST_COST_LEDGER)")
        return
    new = not os.path.exists(path)
    row = {
        "ts": now(), "kind": "run" if runs is not None else "plan", "change": p["change"][:120], "id": p["id"],
        "modules": ",".join(sorted(p["modules"])) or "-", "level": p["level"], "cpu_steps": len(p["cpu_steps"]) + 2 * len(PER_DEVICE_CPU),
        "gpu_steps_r9700": len(p["gpu_steps"]["r9700"]), "gpu_steps_8060s": len(p["gpu_steps"]["8060s"]),
        "est_cpu_min": p["est_cpu_min"], "est_gpu_min": round(sum(p["est_gpu_min"].values()), 1),
        "est_full_gpu_min": round(sum(p["est_full_gpu_min"].values()), 1),
        "cpu_min": round(sum(r["cpu_s"] for r in runs) / 60, 1) if runs else "",
        "gpu_min": round(sum(r["gpu_s"] for r in runs) / 60, 1) if runs else "",
        "result": result, "summary": summary or "",
    }
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, LEDGER_COLS, delimiter="\t", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"test_change: ledger row -> {path}")


# ---------------------------------------------------------------------------------------------------------------------
# back-test

def reason_class(x):
    """class of a full-matrix reason (back-test tally)"""
    if x.startswith("core module"):
        return "core module"
    if x.startswith("core file"):
        return "core file"
    if x.startswith("work equivalence"):
        return "work equivalence not shown"
    if "contract module" in x:
        return "impact: contract module"
    if " in no module (" in x:
        fl = x.split(" in no module (", 1)[1].split(")", 1)[0].replace(" ...", "").split(", ")
        if all(f.startswith("cmake/") or f.endswith("CMakeLists.txt") or f in ("build.bat",) for f in fl):
            return "impact: build-wide file"
        return "impact: product file in no module"
    if "includes" in x:
        return "impact: includer in no module"
    return "impact: other"


def backtest(a):
    root = os.path.abspath(a.root or REPO)
    impact.ROOT = root
    g = impact.git
    m = load_map(root)
    mods = load_mods(root)
    gates = {d: gate_steps(d, root) for d in DEVICES}
    times = load_times(a.times or workspace_file("STEP_TIMES.tsv", "WHIRL_STEP_TIMES"))
    shas = g("log", "--format=%h", "--no-merges", f"-{a.backtest}", a.rev).split()
    out = a.out or os.path.join(REPO, "..", "scratch", "a25_modsel", "backtest")
    os.makedirs(out, exist_ok=True)
    rows = []
    for s in reversed(shas):
        changes = impact.parse_diff(g("diff", "-U0", "--no-color", "-M", f"{s}^", s))
        if not any(p.startswith(("src/", "include/", "kernels/")) for p in changes):
            continue
        rep = impact.analyze(changes, m, s, f"{s}^")
        title = g("log", "-1", "--format=%h %s", s).strip()
        p = plan_with_graph(f"commit {title}", s, changes, rep, mods, m, gates, times, root, f"{s}^", s,
                            os.path.join(out, "cmake_graph"), use_graph=a.cmake_graph_backtest, one_gpu=a.one_gpu)
        rows.append(p)
        print(f"{s} {p['level']:<6} r97 {len(p['gpu_steps']['r9700']):>2} ({p['est_gpu_min']['r9700']:>5} min) "
              f"s8 {len(p['gpu_steps']['8060s']):>2} ({p['est_gpu_min']['8060s']:>5} min)  {title[8:70]}", flush=True)
    json.dump(rows, open(os.path.join(out, "backtest.json"), "w", encoding="utf-8"), indent=1)
    full1 = {d: rows[0]["est_full_gpu_min"][d] if rows else 0 for d in DEVICES}
    tot = {d: round(sum(r["est_gpu_min"][d] for r in rows), 1) for d in DEVICES}
    lv = {}
    for r in rows:
        lv[r["level"].split(" ")[0]] = lv.get(r["level"].split(" ")[0], 0) + 1
    why = {}
    for r in rows:
        for c in sorted({reason_class(x) for x in r["full_reasons"]}):
            why[c] = why.get(c, 0) + 1
    only_bw = [r["id"] for r in rows if r["full_reasons"] and {reason_class(x) for x in r["full_reasons"]} <= {"impact: build-wide file"}]
    with open(os.path.join(out, "backtest.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# test_change back-test: {len(rows)} product commits of the last {a.backtest} non-merge commits of {a.rev}\n\n")
        fh.write(f"Tools: {root} (impact map built from this tree, modules.json and release_gate -List of this tree, CORE of "
                 f"test_change.py). Minutes are ESTIMATES from STEP_TIMES.tsv (median of stored gate summaries), not runs.\n"
                 f"No builds of the old commits exist here: the work-equivalence code-object compare is not done; the "
                 f"static source-closure rule is applied (no kernel source changed -> code objects unchanged).\n\n")
        fh.write("| commit | level | modules | GPU steps R9700 | GPU steps 8060S | GPU min R9700 [est] | GPU min 8060S [est] | full-matrix reason (first) | subject |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            subj = r["change"].split(" ", 2)[2][:60].replace("|", "/")
            fr = (r["full_reasons"][0][:70] if r["full_reasons"] else "").replace("|", "/")
            fh.write(f"| {r['id']} | {r['level'].split(' ')[0]} | {','.join(sorted(r['modules']))[:40] or '-'} | {len(r['gpu_steps']['r9700'])} | "
                     f"{len(r['gpu_steps']['8060s'])} | {r['est_gpu_min']['r9700']} | {r['est_gpu_min']['8060s']} | {fr} | {subj} |\n")
        fh.write(f"\nLevels: {lv}\n\nGPU minutes [est], test_change plans: R9700 {tot['r9700']}, 8060S {tot['8060s']}, total "
                 f"{round(tot['r9700'] + tot['8060s'], 1)}.\nFull gate every commit [est]: {len(rows)} x (R9700 {full1['r9700']} + 8060S "
                 f"{full1['8060s']}) = {round(len(rows) * (full1['r9700'] + full1['8060s']), 1)} min.\n"
                 f"\nFull-matrix reasons (commits with at least one reason of the class): {why}\n"
                 f"Full ONLY because of a build-wide file (cmake / CMakeLists): {len(only_bw)} ({', '.join(only_bw)})\n")
    print(f"backtest: {len(rows)} product commits; levels {lv}; GPU min [est] plans {tot}, full gate each "
          f"{round(len(rows) * (full1['r9700'] + full1['8060s']), 1)}; -> {out}\\backtest.md")
    return 0


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worktree", action="store_true")
    ap.add_argument("--range")
    ap.add_argument("--commit")
    ap.add_argument("--root")
    ap.add_argument("--build", help="build folder under test (default <repo>/build/Release when it exists)")
    ap.add_argument("--base-build", help="build of the base revision: code objects compared (work equivalence, CPU)")
    ap.add_argument("--proof", action="append", default=[], help="DEV=work_equal report (EQUIVALENT) for this build")
    ap.add_argument("--one-gpu", choices=DEVICES, default="r9700", help="GPU of a step modules.json runs on one GPU (testing.md: R9700; 8060S at merge)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all-cpu", action="store_true", help="keep the C++ safety steps even when no changed file is their input")
    ap.add_argument("--no-cmake-graph", action="store_true", help="do not configure the two revisions for a non-list CMake edit (full matrix)")
    ap.add_argument("--cmake-graph-backtest", action="store_true", help="--backtest: configure both sides of each commit whose plan is full only because of a CMake edit")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--work")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ledger")
    ap.add_argument("--times")
    ap.add_argument("--record-plan", action="store_true", help="append a 'plan' row to the ledger without running")
    ap.add_argument("--refresh-times", nargs="*")
    ap.add_argument("--backtest", type=int)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out")
    ap.add_argument("--core-steps-report", action="store_true", help="per core module: its GPU step set from mutation evidence, or why it stays full")
    ap.add_argument("--add-kill-records", nargs="+", metavar="KILL_TSV", help="append gpumut.ps1 kill_<card>.tsv rows to tools/gates/core_steps.json")
    ap.add_argument("--source", default="", help="--add-kill-records: what produced the rows (runner, tree, commit)")
    a = ap.parse_args()
    if a.add_kill_records:
        root = os.path.abspath(a.root or REPO)
        d = load_core_doc(root)
        rows, notes = kill_rows(a.add_kill_records, load_catalog(root), now(), a.source or "gpumut", meta=load_catalog_meta(root))
        d.setdefault("evidence", []).extend(rows)
        with open(os.path.join(root, CORE_STEPS_FILE), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(d, fh, indent=1)
            fh.write("\n")
        for x in notes:
            print(f"note: {x}")
        print(f"add-kill-records: {len(rows)} row(s) -> {os.path.join(root, CORE_STEPS_FILE)}")
        a.core_steps_report = True
    if a.core_steps_report:
        root = os.path.abspath(a.root or REPO)
        mods = load_mods(root)
        doc = load_core_doc(root)
        cat = load_catalog(root)
        ok, why = core_step_sets(doc, cat, evidence_modules(mods), meta=load_catalog_meta(root))
        seen = {r.get("module") for r in doc.get("evidence") or []} | set(cat.values())  # arch025-n11b: every module with a mutant
        show = [k for k in list(CORE) + sorted(mods) if k in CORE or (mods.get(k) or {}).get("full") or k in ok or k in seen]
        for k in dict.fromkeys(show):
            if k in ok:
                print(f"{k:<20} SET   r9700 {','.join(ok[k]['r9700'])} | 8060s {','.join(ok[k]['8060s'])} | {ok[k]['proof']}: {','.join(ok[k]['mutants'])}")
            else:
                print(f"{k:<20} FULL  {why.get(k) or ('contract module: always the full matrix' if is_contract(k) else '?')}")
        print(f"core-steps: {len(ok)} module(s) plan their own GPU step set from mutation evidence "
              f"({len([k for k in CORE if k in ok])} of {len(CORE)} CORE modules)")
        return 0
    if a.refresh_times is not None:
        ws = os.path.abspath(os.path.join(REPO, ".."))
        roots = a.refresh_times or [ws]
        out = a.times or workspace_file("STEP_TIMES.tsv", "WHIRL_STEP_TIMES")
        n, nf = refresh_times(roots, out)
        print(f"refresh-times: {n} (device, step) rows from {nf} summary file(s) -> {out}")
        return 0
    if a.backtest:
        return backtest(a)
    if sum(bool(x) for x in (a.worktree, a.range, a.commit)) != 1:
        ap.error("give exactly one of --worktree, --range A..B, --commit SHA")
    root = os.path.abspath(a.root or REPO)
    if not a.build and os.path.isdir(os.path.join(root, "build", "Release")):
        a.build = os.path.join(root, "build", "Release")
    proofs = {}
    for x in a.proof:
        d, _, f = x.partition("=")
        if d not in DEVICES or not f:
            ap.error(f"--proof {x}: want r9700=FILE or 8060s=FILE")
        proofs[d] = f
    title, diff, new_rev, old_rev, cid = diff_job(a, root)
    changes = impact.parse_diff(diff)
    m = load_map(root)
    rep = impact.analyze(changes, m, new_rev, old_rev)
    gates = {d: gate_steps(d, root) for d in DEVICES}
    times = load_times(a.times or workspace_file("STEP_TIMES.tsv", "WHIRL_STEP_TIMES"))
    scratch = a.work or os.path.join(root, "..", "scratch", "test_change")
    if a.base_build:
        os.makedirs(scratch, exist_ok=True)
    p = plan_with_graph(title, cid, changes, rep, load_mods(root), m, gates, times, root, old_rev, new_rev,
                        os.path.join(scratch, "cmake_graph"), use_graph=not a.no_cmake_graph, one_gpu=a.one_gpu, build=a.build,
                        base_build=a.base_build, proofs=proofs, scratch=scratch, all_cpu=a.all_cpu)
    if a.json:
        print(json.dumps(p, indent=1))
    else:
        print_plan(p)
    if a.run:
        return run_plan(p, a)
    if a.record_plan:
        ledger(a, p, None, "PLAN", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
