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
    report bound to this build's whirl.exe sha256), only when the diff touches no kernel source,
  * v0251-d1b: for a diff whose product files all compile into ONE architecture's code object (kernels/gfx1151/, registered
    per module in modules.json "arch", included by nothing outside that set but tests; see arch_scope) plus registered
    kernel tests: that module's arch steps on that GPU only (the full matrix there when no module owns a file), and NO
    GPU step on the other GPU, which is carried by a work_equal [--arch] proof instead: every gfx1201 code object of
    --build equal to those of --base-build, a build stamped (--stamp-build) at the plan's base revision; --build itself
    the planned version's (stamped at the --range end / --commit, or for --worktree configured from this tree with no
    source newer than its exes; v0251-d1b3). The impact map and the selection data (modules.json) must be the planned
    version's (--worktree, or a root checked out clean at the --range / --commit end; a diff that edits modules.json or a
    selection tool gets the full matrix). --run refuses
    while the carry is not proven; a refused proof plans the full matrix on both GPUs.
Every --run appends one row to the test-cost ledger (internal_docs/TEST_COST_LEDGER.tsv of the workspace, or --ledger).
DEV-TIME ONLY: a release / rc merge / bug-fix release still runs the full release gate on both GPUs (docs/dev/testing.md);
the summaries are tagged change_<id>_<device> and never write PASS_gate_<device>.

    python tools/gates/test_change.py --worktree                     plan for uncommitted changes (vs HEAD)
    python tools/gates/test_change.py --range A..B                   plan for what B adds since it left A
    python tools/gates/test_change.py --commit SHA                   plan for one commit
        [--build DIR] [--base-build DIR] [--proof r9700=FILE] [--proof 8060s=FILE] [--one-gpu r9700|8060s] [--json] [--all-cpu]
        [--run [--work DIR] [--force]] [--ledger FILE] [--record-plan]
    python tools/gates/test_change.py --stamp-build DIR              record the commit DIR's exes were built from (carry base)
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
# the names a gate / runner uses for a card (= tools/impact/work_equal.py GPU_ALIASES; test_change_test.py keeps them equal)
GPU_ALIASES = {"r9700": ("r9700", "r97", "gfx1201", "pci:71:0"), "8060s": ("8060s", "s8", "8s", "gfx1151", "pci:196:0")}
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
# (v0251-d1b: a kernels/gfx1151/ file of an arch-scoped diff, arch_scope below, is planned per architecture instead.)
# A change to one of these (or to a header that a .hip file includes) can change the code objects: no work-equivalence
# short-cut.
KERNEL_GLOBS = ["kernels/*", "*.hip", "include/whirl/kernels_abi.h", "include/whirl/kernels_abi_layout.h", "src/kernels_host/*"]
# v0251-d1b (0.2.5.1 D1(b), PLAN V1-03): per-architecture kernel scope. A file under ARCH_DIRS[arch] compiles into that
# architecture's code object only: ARCH_UNITY (the one translation unit CMakeLists.txt compiles per --offload-arch)
# includes kernels/gfx1151/whirl_kernels_gfx1151.hip only inside #if defined(__gfx1151__) (unity_guard_errors, pinned on
# this tree by test_change_test.py), and the impact map's includers name anything else that includes the file (a host
# includer, e.g. src/ops/attention.cpp -> kernels/gfx1151/attn_ws4_index.h, puts it into whirl.exe for both GPUs: never
# arch-scoped). The other GPU's side is carried by a code-object proof (carry_proof) instead of GPU steps.
ARCH_DIRS = {"gfx1151": "kernels/gfx1151/"}
ARCH_UNITY = "kernels/whirl_kernels.hip"
# GPU steps of the other card that the carry covers when only registered kernel tests (modules.json arch.<a>.tests) select
# them: the kernel test of identical code objects runs the same kernels (check_proof below counts kt_full the same way)
CARRY_TEST_STEPS = ("kt_full",)
CARRY_BY = ("tools/impact/work_equal.py compare --arch {arch} (v0251-d1a): every {arch} code object of --build equal to the "
            "base build's, the report bound to --build's exes")
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
    "server.engine_split_tools": "Python checker tools of the Engine split (cut, census, pins, revtext, e8_judge), no product "
                                 "code; their CPU steps cpu_engine_split / cpu_member_order run them (arch025-engine S0b / S0c)",
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
    # v0251-d1b (v0251-d1a interface): a work_equal --arch report ("arch": gfx1151 / gfx1201) judges only that
    # architecture's code objects and counts only that card's runs, so it proves nothing about the other card even with
    # untagged runs: null (a whole-exe report) or this card's own arch only
    if r.get("arch") not in (None, ARCH[dev]):
        return False, f"a work_equal --arch {r.get('arch')} report is no proof for {dev} ({ARCH[dev]})", set()
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
    # v0251-d1b review AO-1: every run is attributed to its card in every form the gates write (release_gate passes
    # --device r9700 / 8060s, pci ids and --device=X occur, the server gate picks the card with WHIRL_DEVICE); a run of the
    # other card refuses the proof; a run that names no card is compared by work_equal but proves nothing for a card
    # (ADR-0003 addendum 1), so its step is not covered and a report without a run of this card is no proof
    mine = 0
    for L in r.get("launch") or []:
        for run in L.get("runs") or []:
            k = run.get("key", "") if isinstance(run, dict) else ""
            d = run_device(k)
            if d is None:
                continue
            if d != dev:
                return False, f"launch run for another GPU in the proof: {k[:80]}", set()
            mine += 1
            cov.add(k.split("/", 1)[0])
    if not mine:
        return False, f"no launch run of {dev} in the proof (a run that names no GPU is never proof for a card)", set()
    # golden: only cases of this card's reference set ("<arch>:<model>/<mode>_<suite>") that are OK on both sides
    gc = [c for c in ((r.get("golden") or {}).get("cases") or []) if isinstance(c, dict)]
    if (chk.get("golden") == "identical" and any(str(c.get("case", "")).startswith(arch + ":") and c.get("same") for c in gc)
            and not any(re.match(r"gfx\d+:", str(c.get("case", ""))) and not str(c.get("case", "")).startswith(arch + ":") for c in gc)):
        cov |= {"golden_swift", "golden_moe"}
    cov.add("kt_full")  # identical code objects: the kernel test runs the same kernels
    return True, f"EQUIVALENT, bound to --build, {mine} run(s) of {dev}", cov


def run_device(key):
    """the card a launch run belongs to ("r9700" / "8060s" / another gfx name), from its key / command line: --device X,
    --device=X or WHIRL_DEVICE=X with X a card name, an arch or a pci id (work_equal.norm_gpu aliases); None when the run
    names no card (the last one wins, as for a repeated option)"""
    found = re.findall(r"(?:--device[ =]|WHIRL_DEVICE=)\s*([A-Za-z0-9:._-]+)", key or "")
    if not found:
        return None
    g = found[-1].lower()
    for dev, aliases in GPU_ALIASES.items():
        if g in aliases:
            return dev
    return g  # an unknown card / arch: never this card's evidence


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


# ---------------------------------------------------------------------------------------------------------------------
# v0251-d1b: per-architecture scope (modules.json "arch": {"<arch>": {"paths": [globs under kernels/<arch>/, a subset of
# the module's paths], "tests": [tests/ globs: the registered kernel tests of that part], "steps": [GPU steps run on that
# architecture's GPU]}})

def arch_entry_errors(mods, gpu_steps):
    """problems of the modules.json "arch" entries (gpu_steps: the GPU step names of release_gate.ps1 -List)"""
    errs = []
    for n, d in sorted((mods or {}).items()):
        ar = (d or {}).get("arch")
        if ar is None:
            continue
        if not isinstance(ar, dict) or not ar:
            errs.append(f"{n}: arch must be a non-empty object {{arch: {{paths, tests, steps}}}}")
            continue
        for a, e in sorted(ar.items()):
            root = ARCH_DIRS.get(a)
            if root is None:
                errs.append(f"{n}: arch {a}: unknown architecture (known: {', '.join(sorted(ARCH_DIRS))}, test_change.py ARCH_DIRS)")
                continue
            if not isinstance(e, dict):
                errs.append(f"{n}: arch {a}: must be an object")
                continue
            if not e.get("paths"):
                errs.append(f"{n}: arch {a}: no paths")
            for g in e.get("paths") or []:
                if not str(g).startswith(root):
                    errs.append(f"{n}: arch {a}: path {g} is not under {root} (only that folder compiles into the {a} code object alone)")
                elif not any(g == mg or fnmatch.fnmatch(g, mg) for mg in (d.get("paths") or [])):
                    errs.append(f"{n}: arch {a}: path {g} is not one of the module's paths")
            for t in e.get("tests") or []:
                if not str(t).startswith("tests/"):
                    errs.append(f"{n}: arch {a}: test {t} is not under tests/")
            if not e.get("steps"):
                errs.append(f"{n}: arch {a}: no steps")
            for s in e.get("steps") or []:
                if s not in gpu_steps:
                    errs.append(f"{n}: arch {a}: step {s} is not a GPU step of release_gate.ps1")
    return errs


def unity_guard_errors(text, arch="gfx1151"):
    """ARCH_UNITY text -> problems: every #include of an "<arch>/..." file must sit in a branch whose condition is exactly
    defined(__<arch>__) (#if / #elif) or #ifdef __<arch>__ (not its #else), so another architecture's code object never
    compiles a kernels/<arch>/ file (comments stripped; a stricter shape than the preprocessor needs, on purpose)"""
    src = re.sub(r"/\*.*?\*/", lambda mm: "\n" * mm.group(0).count("\n"), text, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    exact = re.compile(r"^(?:defined\s*\(\s*__%s__\s*\)|defined\s+__%s__)$" % (arch, arch))
    stack, errs = [], []
    for no, ln in enumerate(src.splitlines(), 1):
        s = ln.strip()
        d = re.match(r"#\s*(if|ifdef|ifndef|elif|else|endif)\b\s*(.*)$", s)
        if d:
            kw, rest = d.group(1), re.sub(r"\s+", " ", d.group(2).strip())
            if kw == "if":
                stack.append([rest, False])
            elif kw == "ifdef":
                stack.append([f"defined({rest})", False])
            elif kw == "ifndef":
                stack.append([f"!defined({rest})", False])
            elif kw == "elif" and stack:
                stack[-1] = [rest, False]
            elif kw == "else" and stack:
                stack[-1][1] = True
            elif kw == "endif" and stack:
                stack.pop()
            continue
        inc = re.match(r'#\s*include\s*"([^"]+)"', s)
        if inc and (inc.group(1).startswith(f"{arch}/") or f"/{arch}/" in inc.group(1)):
            if not any(exact.match(c) and not in_else for c, in_else in stack):
                errs.append(f"line {no}: #include \"{inc.group(1)}\" outside #if defined(__{arch}__)")
    return errs


def arch_file(f, arch, m):
    """(ok, why): f compiles into arch's code object only, by the impact map OF THE PLANNED VERSION (arch_scope checks
    that): under ARCH_DIRS[arch]; a type the map scans (review C-2: another extension has no includer list); a .hip whose
    map arch is exactly [arch]; a header the map lists includers for; every includer in that folder, the unity source
    (which includes the set only for arch: unity_guard_errors) or a test. A file the map does not know is not arch-only
    (review AO-4: unknown is never a pass)"""
    root = ARCH_DIRS.get(arch)
    if not root or not f.startswith(root):
        return False, f"{f} is not under {root}"
    if not f.endswith(impact_build.SRC_EXT):
        return False, f"{f}: the impact map does not scan {os.path.splitext(f)[1] or 'extension-less'} files (who includes it is unknown)"
    inc = ((m or {}).get("includers") or {})
    if f.endswith(".hip"):
        h = ((m or {}).get("hip") or {}).get(f)
        if h is None:
            return False, f"{f} is not in the impact map (its architecture is unknown)"
        if sorted(h.get("arch") or []) != [arch]:
            return False, f"{f}: impact map arch {'/'.join(h.get('arch') or []) or '?'} (not {arch} only)"
    elif f not in inc:
        return False, f"{f} has no includer in the impact map (who includes it is unknown)"
    bad = [x for x in inc.get(f, []) if not (x.startswith((root, "tests/")) or x == ARCH_UNITY)]
    if bad:
        return False, f"{f} is included by {bad[0]} (not {arch} device code: it reaches both code objects / the host)"
    return True, ""


# v0251-d1b review C-1: the tools that select the steps and judge the carry; a diff that changes them would judge itself.
# v0251-d1b3 (review C-10): and the selection data they read from the working tree: tools/gates/modules.json (the
# modules' arch steps; an edit in the same diff narrowed its own 8060S plan 15 -> 13 steps in the reviewer's probe)
SELF_TOOLS = ["tools/gates/test_change.py", "tools/impact/work_equal.py", "tools/impact/impact.py", "tools/impact/impact_build.py",
              "tools/impact/cxxscan.py", "tools/impact/launch_table.py", "tools/gates/modules.json"]
# v0251-d1b review AO-6: kernel family manifest under kernels/: read by tools/gates/kernel_iso_check.py only (a CPU check),
# compiled into no code object; a port must add its new fragments there
FAMILIES = "kernels/families.json"


def families_arch_change(old_text, new_text):
    """(arch, why) of a kernels/families.json change: the architecture whose folder holds every path added to or removed
    from a family ("" when no path changed); None when anything else changed (a family, desc, deps, _doc), a path outside
    one architecture's folder changed, or a version cannot be parsed"""
    try:
        o, n = json.loads(old_text), json.loads(new_text)
    except (TypeError, ValueError) as e:
        return None, f"{FAMILIES}: cannot read both versions ({e})"
    if not (isinstance(o, dict) and isinstance(n, dict)):
        return None, f"{FAMILIES}: not a JSON object"
    if {k: v for k, v in o.items() if k != "families"} != {k: v for k, v in n.items() if k != "families"}:
        return None, f"{FAMILIES}: a key besides families changed"
    fo, fn = o.get("families") or {}, n.get("families") or {}
    if set(fo) != set(fn):
        return None, f"{FAMILIES}: a family was added or removed"
    changed = set()
    for k in sorted(fo):
        x, y = fo[k] or {}, fn[k] or {}
        if {a: v for a, v in x.items() if a != "paths"} != {a: v for a, v in y.items() if a != "paths"}:
            return None, f"{FAMILIES}: family {k}: a field besides paths changed"
        changed |= set(x.get("paths") or []) ^ set(y.get("paths") or [])
    if not changed:
        return "", f"{FAMILIES}: no path changed"
    a = next((x for x, d in sorted(ARCH_DIRS.items()) if all(p.startswith(d) for p in changed)), None)
    if a is None:
        return None, f"{FAMILIES}: paths outside one architecture's folder changed ({', '.join(sorted(changed)[:3])})"
    return a, f"{FAMILIES}: {len(changed)} {a} path(s) only"


def map_matches(root, new_rev):
    """(ok, why): the impact map test_change uses (load_map: root's working tree) describes the planned version: always
    for --worktree (new_rev None: the working tree is what is planned); for --range / --commit only when root's HEAD is
    new_rev and nothing the map scans differs from it (v0251-d1b review AO-4)"""
    if new_rev is None:
        return True, ""
    try:
        head = impact.git("rev-parse", "HEAD", root=root).strip()
        want = impact.git("rev-parse", f"{new_rev}^{{commit}}", root=root).strip()
    except SystemExit as e:
        return False, f"the planned revision {new_rev} cannot be resolved in {root} ({str(e)[:80]})"
    if head != want:
        return False, (f"the impact map is of {root} at {head[:9]}, not of the planned {new_rev} ({want[:9]}): plan from a tree checked "
                       f"out at {new_rev} for the per-architecture plan")
    dirty = impact.git("status", "--porcelain", "--untracked-files=all", "--", *impact_build.SCAN_DIRS, "cmake", "CMakeLists.txt", root=root)
    if dirty.strip():
        return False, f"the impact map is of {root}'s working tree, which differs from {new_rev} ({dirty.strip().splitlines()[0][:60]})"
    return True, ""


def arch_scope(files, product, modules, callers, mods, m, rep, lists, bfiles, changes=None, map_rev=(True, ""), read_file=None):
    """(scope, why not): scope when every changed product file compiles into ONE architecture's code object only
    (arch_file, by the planned version's impact map: map_rev) and no build file / build-wide input / selection or proof
    tool changed; every module the diff touches and every caller module must register that part (modules.json
    arch.<a>); a product file no module owns is "unregistered" (the full matrix on that GPU). kernels/families.json may
    ride along when its change names that architecture's files only (families_arch_change on the two versions read_file
    gives). scope = {arch, device, other, files, registered {module: [files]}, unregistered, test_includers}"""
    if not product:
        return None, "no product file"
    if not map_rev[0]:
        return None, map_rev[1]
    own_tools = [f for f in files if matches(f, SELF_TOOLS)]
    if own_tools:
        return None, f"{own_tools[0]} changes the tools / data that select the steps and judge the carry (they would judge their own change)"
    if lists or bfiles or any(matches(f, impact.BUILD_WIDE) for f in files) or rep.get("build_wide"):
        return None, "a build file or a build-wide input changed"
    code = [f for f in product if f != FAMILIES]
    if not code:
        return None, f"only {FAMILIES} changed: no kernel file names the architecture"
    found = {next((a for a, d in ARCH_DIRS.items() if f.startswith(d)), None) for f in code}
    if None in found:
        f = next(f for f in code if not f.startswith(tuple(ARCH_DIRS.values())))
        return None, f"{f} is not under {', '.join(sorted(ARCH_DIRS.values()))}"
    if len(found) != 1:
        return None, "files of more than one architecture"
    a = found.pop()
    for f in code:
        op = ((changes or {}).get(f) or {}).get("old_path")
        if op and op != f and not op.startswith(ARCH_DIRS[a]):  # review N-2: renamed in from a place both sets compile
            return None, f"{f} was {op}, which is not under {ARCH_DIRS[a]}"
        ok, w = arch_file(f, a, m)
        if not ok:
            return None, w
    if FAMILIES in product:
        old, new = (read_file("old", FAMILIES), read_file("new", FAMILIES)) if read_file else (None, None)
        if old is None or new is None:
            return None, f"{FAMILIES} changed and its two versions cannot be read"
        fa, w = families_arch_change(old, new)
        if fa is None or fa not in ("", a):
            return None, w if fa is None else f"{w} (not {a})"
    # impact.py lists both GPUs for any header (its generic host-code rule): only arch_file's includer check can narrow a
    # header; anything else that adds a GPU to the map's list is not explained by these files (fail closed)
    if set(rep.get("gpus") or []) - {a} and all(f.endswith(".hip") for f in code):
        return None, f"impact map lists {', '.join(rep.get('gpus') or [])} for the diff"
    reg, unreg = {}, []
    for f in code:
        owners = [k for k, fl in (modules or {}).items() if f in (fl or [])]
        if not owners:
            unreg.append(f)
        for k in owners:
            e = (((mods or {}).get(k) or {}).get("arch") or {}).get(a)
            if not e or not matches(f, e.get("paths") or []):
                return None, f"{f}: module {k} does not register it for {a} (modules.json {k} arch.{a}.paths)"
            reg.setdefault(k, []).append(f)
    for k in sorted(set(modules or {}) | {x for v in (callers or {}).values() for x in v}):
        if not ((((mods or {}).get(k) or {}).get("arch") or {}).get(a)):
            return None, f"module {k} has no {a} part (modules.json {k} arch.{a})"
    tinc = sorted({x for f in code for x in ((m or {}).get("includers") or {}).get(f, []) if x.startswith("tests/")} - set(files))
    dev = DEV_OF_ARCH[a]
    return {"arch": a, "device": dev, "other": [d for d in DEVICES if d != dev], "files": sorted(product), "registered": reg,
            "unregistered": sorted(unreg), "test_includers": tinc}, ""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


MANIFEST = "whirl_build_manifest.json"  # v0251-d1b review AO-5: written by --stamp-build into a build folder


# ninja's Windows .ninja_log mtime = FILETIME (100 ns since 1601) - 12622770400 s (ninja disk_interface.cc); unix ns / 100 +
# 11644473600 s - 12622770400 s = unix ns / 100 - this constant (checked on wt_a25_rel/build/Release: 0.43 s apart)
NINJA_EPOCH_100NS = 9782968000000000
# v0251-d1b3 (review C-12): a .ninja_log v7 line is start_ms, end_ms, mtime, output, command hash (ninja build_log.cc), and
# for a normal rule the logged mtime is the COMMAND START (build.cc FinishCommand: record_mtime = command_start_time_; restat
# / generator rules log the output's own mtime). So the output's mtime lies between the logged value and the logged value
# plus the command's run time (end - start of the same line): on wt_a25_rel/build/Release 1701 of 1720 outputs sit within
# 0.2 s of their run time, whirl.exe 0.43 s of a 0.50 s link, three > 10 s (19.4 s the longest). The slack covers the file
# system's timestamp write after the command returns, not the link time.
NINJA_SLACK_100NS = 2 * 10 ** 7


def ninja_mtimes(build):
    """{output base name (lower case): (logged mtime, run time)} of the newest .ninja_log entry per output of build (100 ns
    units; mtime since 2001, run time = end - start of that line)"""
    out = {}
    try:
        lines = open(os.path.join(build, ".ninja_log"), encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return out
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) >= 4 and p[0].isdigit() and p[1].isdigit() and p[2].isdigit():
            out[p[3].replace("\\", "/").rsplit("/", 1)[-1].lower()] = (int(p[2]), max(0, int(p[1]) - int(p[0])) * 10 ** 4)
    return out


def ninja_unlogged(build, names):
    """the outputs of build that are not what its last ninja run wrote: the file's mtime must lie in [logged start,
    logged start + run time + NINJA_SLACK] (copied / replaced after the build: the copy time, hours or days later; an
    older file: not that run's)"""
    logged = ninja_mtimes(build)
    bad = []
    for x in names:
        t = os.stat(os.path.join(build, x)).st_mtime_ns // 100 - NINJA_EPOCH_100NS
        k = x.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if k not in logged or not 0 <= t - logged[k][0] <= logged[k][1] + NINJA_SLACK_100NS:
            bad.append(x)
    return bad


def stamp_problems(head, dirty, src_newest, exe_oldest, unlogged=()):
    """why a build folder cannot be stamped as the build of `head` (empty = it can): a git HEAD, no uncommitted product
    file, no product source newer than the oldest exe (git checkout / an edit after the build makes the tree newer than
    the exes: the build is not of this tree), and every exe the output ninja logged for this folder (unlogged: a copied
    or replaced exe)"""
    out = []
    if unlogged:
        out.append(f"{', '.join(unlogged)}: not the output ninja logged in this build folder (.ninja_log): copied or replaced after the build")
    if not head:
        out.append("the source tree has no git HEAD")
    if dirty:
        out.append(f"{len(dirty)} uncommitted product file(s) in the source tree ({', '.join(dirty[:3])})")
    if src_newest is not None and exe_oldest is not None and src_newest > exe_oldest:
        out.append("a product source is newer than the exes: the build is older than the tree (rebuild, then stamp)")
    if exe_oldest is None:
        out.append("no whirl.exe in the build folder")
    return out


PRODUCT_DIRS = ["src", "include", "kernels", "cmake", "CMakeLists.txt", "build.bat", "redist"]


class BuildFactsError(Exception):
    pass


def build_facts(build, untracked=False):
    """(source tree, HEAD, dirty product files, newest product source mtime, oldest exe mtime, exes ninja did not log) of
    a build folder: the source tree is DIR/CMakeCache.txt's CMAKE_HOME_DIRECTORY; untracked=True also times the untracked
    product files (a --worktree build compiles them). Raises BuildFactsError (no CMakeCache, not a git tree)."""
    try:
        cache = open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8", errors="replace").read()
    except OSError as e:
        raise BuildFactsError(f"no CMakeCache.txt in {build} ({e})")
    mh = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache, re.M)
    src = mh.group(1).strip() if mh else ""
    try:
        head = impact.git("rev-parse", "HEAD", root=src).strip() if src else ""
        dirty = [ln[3:] for ln in impact.git("status", "--porcelain", "--untracked-files=all", "--", *PRODUCT_DIRS, root=src).splitlines() if ln.strip()]
        tracked = impact.git("ls-files", "--", *PRODUCT_DIRS, root=src).splitlines()
        if untracked:
            tracked += impact.git("ls-files", "--others", "--exclude-standard", "--", *PRODUCT_DIRS, root=src).splitlines()
    except SystemExit as e:
        raise BuildFactsError(f"{src or '(no source dir)'} is not a git tree ({e})")
    mt = [os.path.getmtime(os.path.join(src, f)) for f in tracked if os.path.isfile(os.path.join(src, f))]
    exes = {x: os.path.join(build, x) for x in PROOF_EXES if os.path.isfile(os.path.join(build, x))}
    return (src, head, dirty, max(mt) if mt else None, min(os.path.getmtime(p) for p in exes.values()) if "whirl.exe" in exes else None,
            ninja_unlogged(build, sorted(exes)))


def same_dir(a, b):
    return bool(a) and bool(b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def tree_build_problems(src, tree, src_newest, exe_oldest, unlogged=()):
    """why a build folder is not the build of the working tree `tree` (--worktree: no commit to stamp; empty = it is): it
    was configured from that tree (src: CMakeCache CMAKE_HOME_DIRECTORY), no product source of the tree (tracked or
    untracked) is newer than the oldest exe, every exe is the output ninja logged (v0251-d1b3, review C-11)"""
    out = []
    if not same_dir(src, tree):
        out.append(f"--build was configured from {src or '(no source tree recorded)'}, not the planned tree {tree}")
    if unlogged:
        out.append(f"{', '.join(unlogged)} of --build: not the output ninja logged (.ninja_log): copied or replaced after the build")
    if exe_oldest is None:
        out.append("no whirl.exe in --build")
    elif src_newest is not None and src_newest > exe_oldest:
        out.append("a product source of the planned tree is newer than --build's exes: rebuild after the last edit")
    return out


def head_build_problems(build, head_rev=None, head_tree=None):
    """why --build is not the planned version's build (empty = it is; v0251-d1b3, review C-11): --range / --commit
    (head_rev, the planned revision's full sha): build stamped by --stamp-build at head_rev, clean, its exes unchanged
    since; --worktree (head_tree, the planned working tree): tree_build_problems on build_facts; neither known: refused"""
    if head_rev:
        try:
            man = json.load(open(os.path.join(build, MANIFEST), encoding="utf-8"))
        except (OSError, ValueError):
            return [f"--build {build} has no stamp ({MANIFEST}): run test_change.py --stamp-build {build} right after building the planned version"]
        if man.get("format") != "whirl-build-manifest/1" or not man.get("clean"):
            return [f"{MANIFEST} of --build is not a clean stamp"]
        if str(man.get("commit", "")).lower() != str(head_rev).lower():
            return [f"--build is stamped at {str(man.get('commit', '?'))[:9]}, the planned revision is {str(head_rev)[:9]}"]
        out = []
        for exe in PROOF_EXES:
            st, p = (man.get("exes") or {}).get(exe), os.path.join(build, exe)
            if not st or not os.path.exists(p) or file_sha256(p) != st:
                out.append(f"{exe} of --build is not the stamped one (rebuilt or replaced after the stamp)")
        return out
    if head_tree:
        try:
            src, _head, _dirty, newest, oldest, unlogged = build_facts(build, untracked=True)
        except BuildFactsError as e:
            return [f"--build: {e}"]
        return tree_build_problems(src, head_tree, newest, oldest, unlogged)
    return ["the planned version is not known (no head revision, no working tree): --build cannot be bound to it"]


def stamp_build(build):
    """--stamp-build DIR: record which commit the exes of DIR were built from (DIR/whirl_build_manifest.json): the source
    tree comes from DIR/CMakeCache.txt (CMAKE_HOME_DIRECTORY); refused (exit 2) on stamp_problems"""
    try:
        src, head, dirty, newest, oldest, unlogged = build_facts(build)
    except BuildFactsError as e:
        print(f"test_change: --stamp-build: {e}")
        return 2
    exes = {x: os.path.join(build, x) for x in PROOF_EXES if os.path.isfile(os.path.join(build, x))}
    probs = stamp_problems(head, dirty, newest, oldest, unlogged)
    if probs:
        print(f"test_change: --stamp-build {build}: refused: " + "; ".join(probs))
        return 2
    doc = {"format": "whirl-build-manifest/1", "commit": head, "source": src, "clean": True, "stamped": now(),
           "exes": {x: file_sha256(p) for x, p in sorted(exes.items())}, "tool": "tools/gates/test_change.py --stamp-build"}
    with open(os.path.join(build, MANIFEST), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"test_change: stamped {build}: commit {head[:9]}, {', '.join(f'{k} {v[:12]}' for k, v in doc['exes'].items())}")
    return 0


def carry_proof(path, arch, build, base=None, base_rev=None, head_rev=None, head_tree=None):
    """(ok, why): the work_equal report at path shows every `arch` code object of --build's exes (whirl.exe and
    whirl-server.exe, both required: the carried steps run both) equal to the BASE build's. Read per object
    (exes[].code_objects.objects[]: target, same / cuid_only, only), so a plain `compare` report and a v0251-d1a
    `compare --arch` report count the same way; the report's b exes must be --build's (sha256) and (review AO-5) its a exes
    the base build's: --base-build DIR stamped by --stamp-build at the plan's base revision (base_rev), unchanged since, and
    not the B exes (a self-compare proves nothing); (v0251-d1b3, review C-11) --build itself the planned version's
    (head_build_problems: stamped at head_rev, or for --worktree a build of head_tree). Refused: another arch named by the report, --arch verdict
    NOT_EQUIVALENT, a launch sequence or golden output that differs, an object of that arch that differs or sits on one
    side only, no object of that arch. Launch / golden evidence is not required: the diff changed no host source
    (arch_scope), so the host work on that GPU is the base build's."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"cannot read the proof: {e}"
    if not isinstance(r, dict) or r.get("format") != "whirl-work-equal/1":
        return False, "not a work_equal report (format)"
    ra = r.get("arch")  # v0251-d1a: top-level "arch" of a compare --arch report, null for a whole-exe report
    if isinstance(ra, str) and ra:
        if ra != arch:
            return False, f"the report is for {ra}, not {arch}"
        if r.get("verdict") == "NOT_EQUIVALENT":
            return False, f"--arch {arch} verdict NOT_EQUIVALENT"
    chk = r.get("checks") or {}
    for k in ("launch_sequence", "golden"):
        if chk.get(k) == "different":
            return False, f"{k.replace('_', ' ')} different"
    if not build:
        return False, "no --build: the proof cannot be bound to the exe under test"
    bex = (r.get("b") or {}).get("exes") or {}
    aex = (r.get("a") or {}).get("exes") or {}
    names = []
    for exe in PROOF_EXES:
        p = os.path.join(build, exe)
        if not os.path.exists(p):  # review N-6: the carried steps run whirl.exe and whirl-server.exe: both must be proven
            return False, f"{exe} missing in --build (the carried steps run it: it must be proven too)"
        if not bex.get(exe):
            return False, f"{exe} of --build is not in the proof"
        if file_sha256(p) != bex[exe]:
            return False, f"{exe} of --build is not the B exe of the proof"
        if aex.get(exe) == bex[exe]:
            return False, f"{exe}: A and B are the same file (a self-compare proves nothing)"
        names.append(exe)
    # v0251-d1b3 (review C-11): the B side is the planned version's build (a stamp at the planned revision, or for
    # --worktree a build of that tree newer than its sources), not any folder whose exes the report happens to name
    hp = head_build_problems(build, head_rev=head_rev, head_tree=head_tree)
    if hp:
        return False, "the B side is not the planned version's build: " + "; ".join(hp)
    # review AO-5: the A side is the base build of the plan's base revision
    if not base:
        return False, "the A side is not bound: give --base-build DIR (a base build stamped with test_change.py --stamp-build)"
    try:
        man = json.load(open(os.path.join(base, MANIFEST), encoding="utf-8"))
    except (OSError, ValueError):
        return False, f"--base-build {base} has no stamp ({MANIFEST}): run test_change.py --stamp-build {base} right after building the base"
    if man.get("format") != "whirl-build-manifest/1" or not man.get("clean"):
        return False, f"{MANIFEST} of --base-build is not a clean stamp"
    if not base_rev or str(man.get("commit", "")).lower() != str(base_rev).lower():
        return False, f"--base-build is stamped at {str(man.get('commit', '?'))[:9]}, the plan's base revision is {str(base_rev or '?')[:9]}"
    for exe in names:
        st, p = (man.get("exes") or {}).get(exe), os.path.join(base, exe)
        if not st or not os.path.exists(p) or file_sha256(p) != st:
            return False, f"{exe} of --base-build is not the stamped one (rebuilt or replaced after the stamp)"
        if aex.get(exe) != st:
            return False, f"{exe}: the proof's A exe is not the stamped base build's"
    by = {e.get("name"): e for e in r.get("exes") or [] if isinstance(e, dict)}
    n = 0
    for exe in names:
        e = by.get(exe)
        if not e or (e.get("b") or {}).get("sha256") not in (None, bex[exe]) or (e.get("a") or {}).get("sha256") not in (None, aex.get(exe)):
            return False, f"{exe}: no code-object comparison for these A / B exes in the proof"
        objs = [o for o in (e.get("code_objects") or {}).get("objects") or [] if arch in str(o.get("target", "")).split("/")]
        if not objs:
            return False, f"{exe}: no {arch} code object in the proof"
        for o in objs:
            if o.get("target") != arch or o.get("only"):
                return False, f"{exe}: {arch} code object #{o.get('index')} is not on both sides ({o.get('target')}, only {o.get('only')})"
            if not (o.get("same") or o.get("cuid_only")):
                return False, f"{exe}: {arch} code object #{o.get('index')} differs ({', '.join(o.get('differing_sections') or []) or 'header'})"
        n += len(objs)
    return True, (f"{n} {arch} code object(s) of {', '.join(names)} equal to the base build's (stamped at {str(base_rev)[:9]}), "
                  f"bound to --build")


def make_plan(title, cid, changes, rep, mods, m, gates, times, one_gpu="r9700", build=None, base_build=None, proofs=None,
              scratch=None, all_cpu=False, gate_map=None, cmake_graph=None, core_sets=None, map_rev=(True, ""), base_rev=None,
              read_file=None, head_rev=None, head_tree=None):
    """map_rev (v0251-d1b review AO-4): (ok, why) whether m is the impact map of the planned version (map_matches);
    base_rev: the full sha of the plan's base revision (the carry's A side is bound to it, review AO-5); head_rev / head_tree
    (v0251-d1b3, review C-11): the planned revision's full sha (--range / --commit) or the planned working tree
    (--worktree); the carry's B side (--build) is bound to it; read_file(side,
    path) -> text of a file at the base ("old") / planned ("new") version or None (kernels/families.json, review AO-6)"""
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
    co_json = None
    if base_build and build and scratch:
        co, src = compare_code_objects(base_build, build, os.path.join(scratch, f"code_objects_{cid.replace('+', '_').replace('..', '_')}.json"))
        weq["code_objects"] = f"{co} ({src})"
        if co not in ("identical", "identical-except-cuid") and not kernel:
            full.append(f"work equivalence cannot be shown: code objects {co} although no kernel source changed ({src})")
        if co is not None and src and os.path.isfile(str(src)):
            co_json = src  # v0251-d1b: the same report carries the other GPU of an arch-scoped diff (carry_proof)
    # v0251-d1b: product files of one architecture's code object only (arch_scope): the full-matrix reasons those files
    # raise (CORE_GLOBS kernels/*, impact.py's file in no module) are replaced by the per-architecture plan below; any
    # other reason keeps today's plan
    scope, scope_why = arch_scope(files, product, modules, callers, mods, m, rep, lists, bfiles, changes=changes, map_rev=map_rev,
                                  read_file=read_file)
    if scope:
        own = set(scope["files"])
        other = [r for r in full if not (any(r.startswith(f"core file {f}:") for f in own)
                                         or (scope["unregistered"] and r.startswith("impact.py: full gate or the release steps above")))]
        if other:
            scope, scope_why = None, f"another full-matrix reason: {other[0][:120]}"
        elif not scope["unregistered"]:
            scope["replaced"], full = full, []
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
    carry = {}
    avail = {d: [s["name"] for s in gates[d] if s["gpu"] and not s["na"]] for d in DEVICES}
    if scope:
        # v0251-d1b (replaces the arch025-modsel shortcut "full matrix on the kernel's GPU", which needed `not modules` and
        # so never applied once every kernel file had a module): the architecture's GPU runs the arch steps of the touched
        # modules and their callers (modules.json arch.<a>.steps; the full matrix when a file is in no module), the map's
        # steps and the test-source / gate-script steps; the other GPU runs only what the carry cannot cover
        a, dev = scope["arch"], scope["device"]

        def add(dv, st, w):
            if st in avail[dv] and st not in gpu[dv]:
                gpu[dv].append(st)
                why[(dv, st)] = w
        if scope["unregistered"]:
            for st in avail[dev]:
                add(dev, st, f"full matrix ({a} kernel file(s) in no module: {', '.join(scope['unregistered'][:3])})")
            level = "full"
        else:
            for n in sorted(set(scope["registered"]) | {k for v in callers.values() for k in v}):
                for st in mods[n]["arch"][a].get("steps") or []:
                    add(dev, st, f"module {n} ({a})")
                ev = core_used.get(n) or cset.get(n)
                for st in (ev or {}).get(dev) or []:
                    add(dev, st, f"core module {n}: GPU mutation evidence ({', '.join(ev.get('mutants') or [])})")
                for f, sts in ((mods.get(n) or {}).get("when") or {}).items():
                    if f in changes:
                        for st in expand_gpu(sts, gates)[dev]:
                            add(dev, st, f"module {n} when {f}")
            for st in sorted(map_gpu):
                add(dev, st, "impact map")  # the map's GPUs: this architecture's only (arch_scope checked the files)
            level = "module"
        reg_tests = [g for d in (mods or {}).values() for g in ((((d or {}).get("arch") or {}).get(a) or {}).get("tests") or [])]
        exe_files = {}
        for f, es in sorted(test_source_exes(files + scope["test_includers"] + [x for x in graph_src if x.startswith("tests/")], m).items()):
            for e in es:
                exe_files.setdefault(e, []).append(f)
        carried_tests = set()
        for e, fl in sorted(exe_files.items()):
            globs = (gm.get("exes") or {}).get(e) or set()
            for dv in (DEVICES if e in EXE_BOTH_GPUS else (one_gpu,)):
                for st in avail[dv]:
                    if not any(fnmatch.fnmatch(st, g) for g in globs):
                        continue
                    if dv != dev and st in CARRY_TEST_STEPS and all(matches(f, reg_tests) for f in fl):
                        carried_tests |= set(fl)  # registered kernel tests: the carry's code-object proof covers kt_full
                        continue
                    add(dv, st, f"test source {fl[0]} of {e}")
        for f, sel in sorted(gate_script_steps(files, changes, gm).items()):
            for st in avail[one_gpu]:
                if sel == "ALL" or any(fnmatch.fnmatch(st, g) for g in sel):
                    add(one_gpu, st, f"gate script {f}" + (" (common code: every GPU step)" if sel == "ALL" else ""))
        scope["tests_carried"] = sorted(carried_tests)
        for dv in scope["other"]:
            covers = [s for s in avail[dv] if s not in gpu[dv]]
            c = {"by": CARRY_BY.format(arch=ARCH[dv]), "status": "required", "proof": "", "covers": covers,
                 "covers_min": round(sum(step_seconds(times, dv, s) or 0 for s in covers) / 60, 1),
                 "why": (f"no proof yet: give --base-build DIR (the base revision's build, stamped with --stamp-build; code objects compared "
                         f"on the CPU), optionally with --proof {dv}=FILE (a work_equal [--arch {ARCH[dv]}] report of that base build); "
                         f"--build must be the planned version's (stamped with --stamp-build at the --range end / --commit; for "
                         f"--worktree a build of this tree)")}
            src = (proofs or {}).get(dv) or co_json
            if src:
                ok, w = carry_proof(src, ARCH[dv], build, base=base_build, base_rev=base_rev, head_rev=head_rev, head_tree=head_tree)
                c.update(status="proven" if ok else "refused", proof=src, why=w)
            carry[dv] = c
        refused = {dv: c for dv, c in carry.items() if c["status"] == "refused"}
        if refused:  # fail closed: the binary evidence contradicts the per-architecture scope -> today's full matrix
            full = list(scope.get("replaced") or full) + [
                f"{dv} carry refused ({c['why']}): the {ARCH[dv]} code objects are not shown equal to the base build's"
                for dv, c in sorted(refused.items())]
            gpu = {d: list(avail[d]) for d in DEVICES}
            why = {(d, s): "full matrix" for d in DEVICES for s in gpu[d]}
            scope["tests_carried"] = []
            level = "full"
    elif full:
        for d in DEVICES:
            gpu[d] = list(avail[d])
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
        if dv in carry:  # v0251-d1b: the proof of the carried GPU (its kernels did not change; carry_proof read it above)
            proof_notes[dv] = f"{p}: {dv} carry {carry[dv]['status']}: {carry[dv]['why']}"
            continue
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
        "arch_scope": scope or None, "carry": carry,
        "arch_scope_note": "" if scope or not any(f.startswith(tuple(ARCH_DIRS.values())) for f in product) else scope_why,
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
    sc = p.get("arch_scope")
    if sc:
        reg = "; ".join(f"{k} {len(v)} file(s)" for k, v in sorted((sc.get("registered") or {}).items()))
        print(f"   arch     {sc['arch']} only ({sc['device']}): " + (f"registered {reg}: their {sc['arch']} steps" if not sc.get("unregistered") else
              f"in no module: {', '.join(sc['unregistered'][:3])}: the full matrix on {sc['device']}")
              + (f"; registered kernel tests carried: {', '.join(sc.get('tests_carried') or [])}" if sc.get("tests_carried") else ""))
    elif p.get("arch_scope_note"):
        print(f"   arch     not one architecture's kernels only: {p['arch_scope_note']}")
    for d, c in sorted((p.get("carry") or {}).items()):
        print(f"   carry    {d}: {c['status'].upper()}: {c['by']}; replaces {len(c['covers'])} GPU step(s), ~{c['covers_min']} min [est]"
              + (f"; proof {c['proof']}" if c.get("proof") else "") + f" ({c['why']})")
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
    need = sorted(d for d, c in (p.get("carry") or {}).items() if c.get("status") == "required")
    if need:  # v0251-d1b: a GPU whose steps the plan drops runs nothing only with its proof (fail closed, before any step)
        print(f"test_change: refused: {', '.join(need)} carried by a work_equal --arch proof that is not given: "
              f"{p['carry'][need[0]]['why']}")
        return 2
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
    # v0251-d1b review C-4: a carried GPU without GPU steps runs last with its per-GPU CPU checks only, so a setup error
    # there (exit 2, e.g. the eGPU not attached) never stops the other GPU's steps
    carried = [d for d in DEVICES if ((p.get("carry") or {}).get(d) or {}).get("status") == "proven" and not p["gpu_steps"][d]]
    for dev in [d for d in DEVICES if d not in carried] + carried:  # one GPU after the other
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
                            os.path.join(out, "cmake_graph"), use_graph=a.cmake_graph_backtest, one_gpu=a.one_gpu,
                            map_rev=(False, "back-test: the impact map is the root tree's, not each commit's"))
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
    ap.add_argument("--stamp-build", metavar="DIR", help="record the commit DIR's exes were built from (whirl_build_manifest.json; "
                                                         "the base build of a carried GPU must be stamped, v0251-d1b review AO-5)")
    a = ap.parse_args()
    if a.stamp_build:
        return stamp_build(os.path.abspath(a.stamp_build))
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
    # v0251-d1b review AO-4 / AO-5 / AO-6: is m the planned version's map; the base revision the carry's A side is bound
    # to; the two versions of a data file (kernels/families.json)
    map_rev = map_matches(root, new_rev)
    try:
        base_rev = impact.git("rev-parse", f"{old_rev}^{{commit}}", root=root).strip()
    except SystemExit:
        base_rev = None
    # v0251-d1b3 (review C-11): the planned version the carry's B side (--build) is bound to
    head_tree = root if new_rev is None else None
    try:
        head_rev = impact.git("rev-parse", f"{new_rev}^{{commit}}", root=root).strip() if new_rev else None
    except SystemExit:
        head_rev = None

    def read_file(side, path):
        if side == "old":
            return impact.show(old_rev, path)
        if new_rev:
            return impact.show(new_rev, path)
        try:
            return open(os.path.join(root, path), encoding="utf-8").read()
        except OSError:
            return None
    p = plan_with_graph(title, cid, changes, rep, load_mods(root), m, gates, times, root, old_rev, new_rev,
                        os.path.join(scratch, "cmake_graph"), use_graph=not a.no_cmake_graph, one_gpu=a.one_gpu, build=a.build,
                        base_build=a.base_build, proofs=proofs, scratch=scratch, all_cpu=a.all_cpu, map_rev=map_rev,
                        base_rev=base_rev, read_file=read_file, head_rev=head_rev, head_tree=head_tree)
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
