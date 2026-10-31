#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""arch025-Q4 cpu_tsa: the lock-removal proof of the clang thread-safety build, and the step verdict.

  tsa_check.py mutate <build-tsa> <repo>   for every mutation in MUTATIONS: copy the source next to the build (build-tsa/mutants),
                                           apply the edit (the text must occur exactly once), compile the copy with the TU's own
                                           command from compile_commands.json (same flags, -Werror=thread-safety) and require:
                                           control copy (no edit) compiles, mutant FAILS with its expected diagnostic.
                                           Prints one "mutant <name>: ok|FAIL ..." line each and "tsa mutate: ok (N of N)".
  tsa_check.py verdict <build-tsa>         the build DONE "rc=0", MUTATE "rc=0", and the build log has no thread-safety error.
                                           Prints "tsa verdict: ok (...)" or "tsa verdict: FAIL (...)".
Written for WHIRL. No third-party code.
"""
import json
import os
import re
import shlex
import subprocess
import sys

# name, source (repo-relative), exact text to find, its replacement, a regex the compiler output of the mutant must match
MUTATIONS = [
    ("engine-stop-unlocked", "src/server/engine_queue.cpp",
     "void Engine::stop() {\n    whirl::MutexLock lk(q_mutex_);\n    stop_ = true;",
     "void Engine::stop() {\n    stop_ = true;",
     r"writing variable 'stop_' requires holding mutex 'q_mutex_'"),
    ("engine-wait-predicate-unannotated", "src/server/engine.cpp",
     "[&]() WH_REQUIRES(q_mutex_) { return wake_seq_ != wake_seq0; }",
     "[&] { return wake_seq_ != wake_seq0; }",
     r"reading variable 'wake_seq_' requires holding mutex 'q_mutex_'"),
    # arch025-engine S9: health's vram_mu_ lock and reads are VramWatch::vramHealth (engine_vram.cpp, verbatim); vram_last_ /
    # vram_spill_ are VramWatch references guarded by its vram_mu_ reference: the annotation must still bind there
    ("engine-health-vram-unlocked", "src/server/engine_vram.cpp",
     "    whirl::MutexLock lk(vram_mu_);\n    h.vram_ok = vram_last_.ok;",
     "    h.vram_ok = vram_last_.ok;",
     r"reading variable 'vram_last_' requires holding mutex 'vram_mu_'"),
    # arch025-tsaidle: idleWait / runLoopCalls unlock and relock the caller's lock through WH_UNLOCK_HELD / WH_RELOCK_HELD,
    # so their bodies are analysed (they were WH_NO_THREAD_SAFETY_ANALYSIS after arch025-E2, where a dropped relock passed;
    # the predicate lambdas are checked either way): a dropped predicate annotation, a dropped relock and an unbalanced
    # runLoopCalls must fail
    ("engine-idlewait-predicate-unannotated", "src/server/engine.cpp",
     "[&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });\n        } else {",
     "[&] { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });\n        } else {",
     r"reading variable 'queue_' requires holding mutex 'q_mutex_'"),
    ("engine-idlewait-relock-dropped", "src/server/engine.cpp",
     "            vram_due = vw_.vramIdleTick();\n            gv_.govApply();\n            WH_RELOCK_HELD(lk, q_mutex_);\n",
     "            vram_due = vw_.vramIdleTick();\n            gv_.govApply();\n",
     r"reading variable 'queue_' requires holding mutex 'q_mutex_'"),
    ("engine-runloopcalls-relock-dropped", "src/server/engine_queue.cpp",
     "    for (auto& f : calls) f();\n    WH_RELOCK_HELD(lk, q_mutex_);\n",
     "    for (auto& f : calls) f();\n",
     r"mutex 'q_mutex_' (is not held on every path|to be held at the end of function)"),
    # arch025-engine S8 (F14): gov_json_ is guarded by GovState's own gov_mu_ (was vram_mu_); GovernorGlue reaches both through
    # references under the same names (engine_gov.h), and the annotation must still bind there: removing either lock fails
    ("gov-json-unlocked", "src/server/engine_gov.cpp",
     "    whirl::MutexLock lk(gov_mu_);\n    return gov_json_;\n",
     "    return gov_json_;\n",
     r"reading variable 'gov_json_' requires holding mutex 'gov_mu_'"),
    ("gov-publish-unlocked", "src/server/engine_gov.cpp",
     "    whirl::MutexLock lk(gov_mu_);\n    gov_json_ = memgov::statusJson(st);\n",
     "    gov_json_ = memgov::statusJson(st);\n",
     r"writing variable 'gov_json_' requires holding mutex 'gov_mu_'"),
    ("vram-note-unlocked", "src/server/engine_vram.cpp",
     "        whirl::MutexLock lk(vram_mu_);\n        vram_last_ = s;\n        vram_spill_ = vram_warn_ && vram_watch_.spilling();\n",
     "        vram_last_ = s;\n        vram_spill_ = vram_warn_ && vram_watch_.spilling();\n",
     r"writing variable 'vram_last_' requires holding mutex 'vram_mu_'"),
    # not Tier::submit: clang's analysis does not see the access in `(c ? q_reads_ : q_writes_).push_back(j)` (a guarded member
    # reached through ?: is not checked; arch025-Q4 HANDOFF "known gaps"), so removing that lock is not caught
    ("tier-pendingread-unlocked", "src/tier/kv_tier.cpp",
     "Job* Tier::pendingRead() {\n    whirl::MutexLock lk(q_mutex_);\n",
     "Job* Tier::pendingRead() {\n",
     r"variable 'q_reads_' requires holding mutex 'q_mutex_'"),
    ("tier-io-predicate-unannotated", "src/tier/kv_tier.cpp",
     "[&]() WH_REQUIRES(q_mutex_) { return stop_ || !q_reads_.empty() || !q_writes_.empty(); }",
     "[&] { return stop_ || !q_reads_.empty() || !q_writes_.empty(); }",
     r"reading variable 'stop_' requires holding mutex 'q_mutex_'"),
    ("http-setreading-unlocked", "src/server/http.cpp",
     "    whirl::MutexLock lk(conns_mu_);\n    if (on) reading_.insert(sock);",
     "    if (on) reading_.insert(sock);",
     r"variable 'reading_' requires holding mutex 'conns_mu_'"),
    ("log-setconsole-unlocked", "src/server/log.cpp",
     "    whirl::MutexLock lk(g_mutex);\n    g_console = on;",
     "    g_console = on;",
     r"writing variable 'g_console' requires holding mutex 'g_mutex'"),
]

TSA_ERR = re.compile(r"\[-Werror,-Wthread-safety[-a-z]*\]")


def apply_edit(text, old, new):
    """Returns the edited text; raises ValueError unless old occurs exactly once (a stale mutation must not pass silently)."""
    n = text.count(old)
    if n != 1:
        raise ValueError(f"mutation text found {n} times (must be 1)")
    return text.replace(old, new)


def mutant_ok(control_rc, control_out, rc, out, expect):
    """(ok, why) for one mutation: the control compiles, the mutant fails with a thread-safety error matching expect."""
    if control_rc != 0:
        return False, f"control copy does not compile (rc={control_rc})"
    if rc == 0:
        return False, "mutant compiled: the removed lock was not detected"
    if not TSA_ERR.search(out):
        return False, f"mutant failed (rc={rc}) without a thread-safety error"
    if not re.search(expect, out):
        return False, f"mutant failed but not with /{expect}/"
    return True, "removed lock detected"


def read_rc(path):
    try:
        m = re.search(r"rc=(-?\d+)", open(path, encoding="utf-8", errors="replace").read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def verdict(done_rc, mutate_rc, build_log):
    """(ok, why) of the cpu_tsa step from the two rc files and the build log text."""
    bad = []
    if done_rc != 0:
        bad.append(f"build DONE rc={done_rc}")
    if TSA_ERR.search(build_log or ""):
        n = len(TSA_ERR.findall(build_log))
        bad.append(f"{n} thread-safety error(s) in the build log")
    if mutate_rc != 0:
        bad.append(f"lock-removal proof MUTATE rc={mutate_rc}")
    return (not bad), ("; ".join(bad) if bad else "clean build, every removed lock detected")


def norm(p):
    return os.path.normcase(os.path.normpath(p))


def compile_cmd(db, src):
    for e in db:
        f = e["file"] if os.path.isabs(e["file"]) else os.path.join(e["directory"], e["file"])
        if norm(f) == norm(src):
            return e
    return None


def run_compile(entry, src, copy, obj):
    cmd = entry.get("command")
    args = entry["arguments"] if "arguments" in entry else shlex.split(cmd, posix=False)
    out = []
    for a in args:
        u = a.strip('"')
        if norm(os.path.join(entry["directory"], u)) == norm(src):
            out.append(copy)
        elif u.startswith(("/Fo", "-Fo")):
            out.append("/Fo" + obj)
        else:
            out.append(u)
    if copy not in out:
        out.append(copy)
    # the copy lives elsewhere: its "..." includes must still resolve next to the original
    out.insert(1, "/I" + os.path.dirname(src))
    p = subprocess.run(out, cwd=entry["directory"], capture_output=True, text=True, errors="replace")
    return p.returncode, p.stdout + p.stderr


def cmd_mutate(build, repo):
    dbp = os.path.join(build, "compile_commands.json")
    try:
        db = json.load(open(dbp, encoding="utf-8"))
    except (OSError, ValueError) as ex:
        print(f"tsa mutate: FAIL (no compile_commands.json in {build}: {ex})")
        return 1
    mdir = os.path.join(build, "mutants")
    os.makedirs(mdir, exist_ok=True)
    good = 0
    for name, rel, old, new, expect in MUTATIONS:
        src = os.path.join(repo, rel.replace("/", os.sep))
        entry = compile_cmd(db, src)
        if entry is None:
            print(f"mutant {name}: FAIL ({rel} not in compile_commands.json)")
            continue
        text = open(src, encoding="utf-8", newline="").read()
        try:
            mtext = apply_edit(text, old, new)
        except ValueError as ex:
            print(f"mutant {name}: FAIL ({rel}: {ex})")
            continue
        base = os.path.splitext(os.path.basename(rel))[0]
        ctl = os.path.join(mdir, f"{base}.control.cpp")
        mut = os.path.join(mdir, f"{base}.{name}.cpp")
        open(ctl, "w", encoding="utf-8", newline="").write(text)
        open(mut, "w", encoding="utf-8", newline="").write(mtext)
        crc, cout = run_compile(entry, src, ctl, os.path.join(mdir, f"{base}.control.obj"))
        rc, out = run_compile(entry, src, mut, os.path.join(mdir, f"{base}.{name}.obj"))
        open(mut + ".log", "w", encoding="utf-8").write(out)
        ok, why = mutant_ok(crc, cout, rc, out, expect)
        first = next((ln.strip() for ln in out.splitlines() if TSA_ERR.search(ln)), "")
        print(f"mutant {name}: {'ok' if ok else 'FAIL'} ({rel}: {why}{'; ' + first[-160:] if first else ''})")
        if not ok and crc != 0:
            print("   control output: " + " | ".join(cout.splitlines()[:6]))
        good += ok
    n = len(MUTATIONS)
    print(f"tsa mutate: {'ok' if good == n else 'FAIL'} ({good} of {n} removed locks detected)")
    return 0 if good == n else 1


def cmd_verdict(build):
    log = ""
    try:
        log = open(os.path.join(build, "logs", "build.log"), encoding="utf-8", errors="replace").read()
    except OSError:
        pass
    ok, why = verdict(read_rc(os.path.join(build, "DONE")), read_rc(os.path.join(build, "MUTATE")), log)
    print(f"tsa verdict: {'ok' if ok else 'FAIL'} ({why})")
    return 0 if ok else 1


def main(argv):
    if len(argv) == 3 and argv[0] == "mutate":
        return cmd_mutate(argv[1], argv[2])
    if len(argv) == 2 and argv[0] == "verdict":
        return cmd_verdict(argv[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
