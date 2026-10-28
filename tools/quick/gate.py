#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1b: automatic gate cascade for one candidate (FASTLANE.md FL-1 item 3; PROTOCOL "固定的自動驗證關卡").
# Internal experiment tool (arch025 N16): the dev-time "test what you changed" entry point is tools/gates/test_change.py
# (docs/dev/testing.md#module-gate); this cascade judges one kernel / mode candidate (bitwise, NMSE, KL, perf).
# Stages, in order; the first failure stops the run (the baseline stays the default: this tool never changes one):
#   build    (only with --worktree --build) build.bat Release in the candidate worktree (--touch-cpp first when a
#            header changed: the MSVC /showIncludes dependency scan is unreliable here)
#   bitwise  (default for any candidate whose output must not change)
#            a. whirl-kernel-test --only FAMILIES on real model matrices (shape sweep of the harness: every type's real
#               rows, several token counts / seeds inside each family) -> "PASS: 0 failing"
#            b. whirl golden for model x mode x suite (plain prefill short + 4000 tokens, decode steps, MTP, n-gram drafts,
#               segmented prefill / batched rows / verify / MTP cycle) with the candidate env, every step line ('G'
#               lines: top-5 ids + FNV-1a hash of the logits) compared with tests/golden/<arch>/... -> identical
#   nmse     (instead of b., --check nmse: for a fast-mode kernel that may change numerics by design)
#            kernel test as in a. (its own tolerances vs CPU references), then the teacher-forced logits of the last
#            --nmse-rows rows of a fixed 2k code prompt, candidate vs the baseline exe in the same mode:
#            NMSE = sum (a - b)^2 / sum a^2 (llama.cpp tests/test-backend-ops.cpp nmse(), there per op vs the CPU
#            backend, default limit 1e-7; here per model output, limit --nmse-max) and top-1 agreement >= --top1-min
#   kl       tools/quick/kl_quick.py (2k + 8k, one 512-row window each, code + zh, the gate's dense limits)
#   perf     tools/quick/toks.py (decode 2k / 32k, prefill 8k) vs a recorded reference of the baseline exe
#            (--record-perf-ref writes it); a drop of more than --perf-tol % at any length fails
#
#   python gate.py --device r97|8s (--exe DIR|whirl.exe | --worktree DIR [--build] [--touch-cpp]) [--mode balance]
#                  [--env K=V;K2=V2] [--check bitwise|nmse] [--kt FAMILIES|none] [--models dense[,moe]]
#                  [--suites single,pair,ngram] [--stages build,bitwise,kl,perf] [--tag NAME]
#   python gate.py --device 8s --record-perf-ref [--mode balance]       (baseline exe -> fl1b\perf_ref_<dev>_<mode>.json)
# Exit: 0 every selected stage passed, 1 a stage failed (named in the summary), 2 setup / run error.
# Summary: whirl-work\fl1b\gate\<time>_<dev>_<tag>\summary.json + the stage logs next to it.
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ARCH = {"r97": "gfx1201", "8s": "gfx1151"}
MOE = r"C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf"
MODELS = {"dense": qlib.MODEL, "moe": MOE}
DEF_KT = "quant,gemv,gemvh,gemm,gdn,ws4,ws4g"  # ~55 s on the R9700; attn (305 s) and misc (67 s) via --kt all or by name
ALL_KT = "quant,gemv,gemm,attn,attndq4,gdn,moe,moemx,moef,misc,dflash,gemvh,ws4,ws4g"
OUT = os.path.join(qlib.WORK, "fl1b", "gate")


def envd(s):
    return dict(kv.split("=", 1) for kv in s.split(";") if kv)


def run(cmd, logf, env=None, cwd=None):
    t = time.time()
    with open(logf, "w", encoding="utf-8", errors="replace") as fo:
        fo.write(" ".join(cmd) + "\n")
        fo.flush()
        rc = subprocess.run(cmd, env=env, cwd=cwd, stdout=fo, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t, open(logf, encoding="utf-8", errors="replace").read()


def step_lines(text):
    return [l for l in text.splitlines() if l.startswith("G ") and not l.startswith(("G model", "G header", "G info"))]


def stage_build(a, d):
    if a.touch_cpp:
        now = time.time()
        for root in ("src", "tests", "tools", "kernels"):
            for dp, _, fs in os.walk(os.path.join(a.worktree, root)):
                for f in fs:
                    if f.endswith((".cpp", ".hip")):
                        os.utime(os.path.join(dp, f), (now, now))
    rc, dt, out = run(["cmd", "/c", os.path.join(a.worktree, "build.bat"), "Release"], os.path.join(d, "build.log"), cwd=a.worktree)
    return rc == 0, dt, f"build.bat exit {rc}"


def stage_kt(a, d, bdir, env):
    kt = os.path.join(bdir, "whirl-kernel-test.exe")
    if a.kt == "all":
        a.kt = ALL_KT
    cmd = [kt, "--device", qlib.DEVICES[a.device], "--only", a.kt, "--q4", qlib.MODEL, "--moemx", MOE]
    rc, dt, out = run(cmd, os.path.join(d, "kt.log"), env=qlib.clean_env(**env))
    m = re.search(r"^(PASS|FAIL): (\d+) failing.*$", out, re.M)
    checks = re.search(r"(\d+) checks", out)
    ok = rc == 0 and m is not None and m.group(1) == "PASS"
    return ok, dt, f"kernel test --only {a.kt}: exit {rc}, {m.group(0) if m else 'no PASS/FAIL line'}"


def stage_golden(a, d, exe, env):
    arch = ARCH[a.device]
    gold = os.path.join(a.golden_root, arch)
    res, ok, t0 = [], True, time.time()
    for mdl in a.models.split(","):
        for s in a.suites.split(","):
            ref = os.path.join(gold, mdl, f"{a.mode}_{s}.txt")
            if not os.path.isfile(ref):
                return False, time.time() - t0, f"no reference {ref}"
            cmd = [exe, "golden", MODELS[mdl], "--suite", s, "--" + a.mode, "--device", arch]
            e = qlib.clean_env(**env)
            e.setdefault("AMD_LOG_LEVEL", "1")
            r = subprocess.run(cmd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
            txt = r.stdout + "\n" + r.stderr
            open(os.path.join(d, f"golden_{mdl}_{a.mode}_{s}.txt"), "w", encoding="utf-8").write(txt)
            got = step_lines(txt)
            want = step_lines(open(ref, encoding="utf-8").read())
            info_g = [l for l in txt.splitlines() if l.startswith("G info")]
            info_r = [l for l in open(ref, encoding="utf-8").read().splitlines() if l.startswith("G info")]
            if r.returncode != 0 or not got:
                res.append(f"{mdl}/{a.mode}_{s}: run error exit {r.returncode} ({len(got)} step lines)")
                ok = False
                break
            if got != want:
                i = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
                res.append(f"{mdl}/{a.mode}_{s}: DIFFERENT at step line {i + 1} of {len(want)}: ref '{want[i] if i < len(want) else '<end>'}' "
                           f"got '{got[i] if i < len(got) else '<end>'}'" + (f" (G info differs: {info_g[:1]})" if info_g != info_r else ""))
                ok = False
                break
            res.append(f"{mdl}/{a.mode}_{s}: {len(got)} step lines identical")
        if not ok:
            break
    return ok, time.time() - t0, "; ".join(res)


def dump_logits(exe, prompt, n, rows, mode, dev, env, path, logf):
    e = qlib.clean_env(**env)
    e.update({"WHIRL_MTP": "0", "WHIRL_DUMP_LOGITS": path, "WHIRL_DUMP_AT": str(n), "WHIRL_DUMP_ROWS": str(rows), "AMD_LOG_LEVEL": "1"})
    ctx = (n + 1 + 255) // 256 * 256 + 256
    cmd = [exe, "chat", qlib.MODEL, "@" + prompt, "--raw", "--max-tokens", "1", "--ctx", str(ctx), "--" + mode, "--device", qlib.DEVICES[dev]]
    rc, dt, _ = run(cmd, logf, env=e)
    return rc


def stage_nmse(a, d, exe, env):
    t0 = time.time()
    prompt = qlib.prompt_file(qlib.DEF_EXE, a.nmse_len, "code")
    n = qlib.ntokens(qlib.DEF_EXE, prompt)
    pb, pc = os.path.join(d, "base.f16"), os.path.join(d, "cand.f16")
    if dump_logits(a.base_exe, prompt, n, a.nmse_rows, a.mode, a.device, {}, pb, os.path.join(d, "nmse_base.log")) != 0:
        return False, time.time() - t0, "baseline logits dump failed"
    if dump_logits(exe, prompt, n, a.nmse_rows, a.mode, a.device, env, pc, os.path.join(d, "nmse_cand.log")) != 0:
        return False, time.time() - t0, "candidate logits dump failed"
    A = np.fromfile(pb, dtype=np.float16).astype(np.float64)
    B = np.fromfile(pc, dtype=np.float16).astype(np.float64)
    if A.size == 0 or A.size != B.size or A.size % a.nmse_rows:
        return False, time.time() - t0, f"dump sizes {A.size} / {B.size}"
    A, B = A.reshape(a.nmse_rows, -1), B.reshape(a.nmse_rows, -1)
    nmse = float(((A - B) ** 2).sum() / (A ** 2).sum())
    row = ((A - B) ** 2).sum(1) / (A ** 2).sum(1)
    top1 = float((A.argmax(1) == B.argmax(1)).mean())
    for p in (pb, pc):
        os.remove(p)
    ok = nmse <= a.nmse_max and top1 >= a.top1_min
    return ok, time.time() - t0, (f"logits NMSE {nmse:.3e} (<= {a.nmse_max:.1e}), worst row {row.max():.3e}, top-1 agreement "
                                  f"{100 * top1:.2f}% (>= {100 * a.top1_min:.1f}%) over {a.nmse_rows} rows of a {n}-token prompt")


def stage_kl(a, d, exe, env):
    cmd = [sys.executable, os.path.join(HERE, "kl_quick.py"), "--device", a.device, "--modes", a.mode, "--exe", exe, "--tag", "gate_" + a.tag]
    if env:
        cmd += ["--env", a.env]
    rc, dt, out = run(cmd, os.path.join(d, "kl.log"), env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    lines = [l.strip() for l in out.splitlines() if l.strip().startswith(("PASS", "FAIL", "ERROR"))]
    return rc == 0, dt, f"kl_quick exit {rc}: " + " | ".join(lines)


def toks_rows(out):
    r = {}
    for l in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:.*hash ([0-9a-f]+))?", l)
        if m:
            r[int(m.group(1))] = dict(prefill=float(m.group(3)), decode=float(m.group(5)), hash=m.group(6))
    return r


def perf_ref_path(a):
    return os.path.join(qlib.WORK, "fl1b", f"perf_ref_{a.device}_{a.mode}.json")


def toks_cmd(a, exe, env, d, i):
    cmd = [sys.executable, os.path.join(HERE, "toks.py"), "--device", a.device, "--mode", a.mode, "--exe", exe, "--tag", "gate",
           "--fixed-nonce", "--out", os.path.join(d, f"toks{i}.tsv")]
    if env:
        cmd += ["--env", a.env]
    return cmd


def stage_perf(a, d, exe, env):
    """toks.py with a fixed prompt nonce vs the recorded reference; a length that drops more than --perf-tol is run
    once more and the better of the two runs counts (a real regression survives a rerun, one-off jitter does not)."""
    ref_p = perf_ref_path(a)
    if not os.path.isfile(ref_p):
        return False, 0.0, f"no perf reference {ref_p} (run gate.py --record-perf-ref --device {a.device} --mode {a.mode})"
    refj = json.load(open(ref_p, encoding="utf-8"))
    ref = {int(k): v for k, v in refj["rows"].items()}
    t0 = time.time()
    best, hashes = {}, {}
    for i in range(2):
        rc, dt, out = run(toks_cmd(a, exe, env, d, i), os.path.join(d, f"perf{i}.log"))
        got = toks_rows(out)
        if rc != 0 or not got:
            return False, time.time() - t0, f"toks exit {rc}, {len(got)} rows"
        for L, g in got.items():
            b = best.setdefault(L, dict(g))
            b["decode"], b["prefill"] = max(b["decode"], g["decode"]), max(b["prefill"], g["prefill"])
            hashes.setdefault(L, g.get("hash"))
        bad = [L for L, rv in ref.items() if L not in best or any(best[L][k] < rv[k] * (1 - a.perf_tol / 100) for k in ("decode", "prefill"))]
        if not bad:
            break
    msgs, ok = [], True
    for L, rv in sorted(ref.items()):
        g = best.get(L)
        if not g:
            ok = False
            msgs.append(f"{L}: missing")
            continue
        for k in ("decode", "prefill"):
            ch = 100.0 * (g[k] / rv[k] - 1.0)
            bad = ch < -a.perf_tol
            ok = ok and not bad
            msgs.append(f"{L} {k} {g[k]:.1f} vs {rv[k]:.1f} ({ch:+.1f}%){' FAIL' if bad else ''}")
        if rv.get("hash") and hashes.get(L) and rv["hash"] != hashes[L]:
            msgs.append(f"{L} output hash {hashes[L]} != reference {rv['hash']} (output changed: acceptance differs)")
    return ok, time.time() - t0, f"runs {i + 1}: " + "; ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], required=True)
    ap.add_argument("--exe", default="", help="candidate whirl.exe (or its build directory); default: the baseline exe")
    ap.add_argument("--worktree", default="", help="candidate worktree (with --build: build it first)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--touch-cpp", action="store_true", help="touch every .cpp/.hip before the build (header changes)")
    ap.add_argument("--base-exe", default=qlib.DEF_EXE)
    ap.add_argument("--mode", default="balance", choices=["precise", "balance", "fast"])
    ap.add_argument("--env", default="", help="candidate WHIRL_* settings K=V;K2=V2 (env-switch candidates)")
    ap.add_argument("--check", choices=["bitwise", "nmse"], default="bitwise")
    ap.add_argument("--kt", default=DEF_KT, help="kernel test families, or 'none'")
    ap.add_argument("--models", default="dense")
    ap.add_argument("--suites", default="single,pair,ngram")
    ap.add_argument("--golden-root", default=os.path.join(REPO, "tests", "golden"))
    ap.add_argument("--stages", default="build,bitwise,kl,perf")
    ap.add_argument("--nmse-len", type=int, default=2048)
    ap.add_argument("--nmse-rows", type=int, default=256)
    ap.add_argument("--nmse-max", type=float, default=1e-3)
    ap.add_argument("--top1-min", type=float, default=0.95)
    ap.add_argument("--perf-tol", type=float, default=4.0, help="max drop in percent (n=1 runs jitter about +-2%%)")
    ap.add_argument("--record-perf-ref", action="store_true")
    ap.add_argument("--tag", default="cand")
    a = ap.parse_args()

    if a.record_perf_ref:
        d = os.path.join(OUT, f"{time.strftime('%Y%m%d-%H%M%S')}_{a.device}_perfref")
        os.makedirs(d, exist_ok=True)
        rc, dt, out = run(toks_cmd(a, a.base_exe, {}, d, 0), os.path.join(d, "perf.log"))
        rows = toks_rows(out)
        if rc != 0 or not rows:
            print(out[-2000:])
            return 2
        json.dump(dict(exe=a.base_exe, mode=a.mode, device=a.device, when=time.strftime("%Y-%m-%dT%H:%M:%S"), rows=rows),
                  open(perf_ref_path(a), "w", encoding="utf-8"), indent=1)
        print(f"perf reference {perf_ref_path(a)}: {rows} ({dt:.0f} s)")
        return 0

    if a.worktree:
        bdir = os.path.join(a.worktree, "build", "Release")
        exe = os.path.join(bdir, "whirl.exe")
        if a.golden_root == os.path.join(REPO, "tests", "golden") and os.path.isdir(os.path.join(a.worktree, "tests", "golden")):
            a.golden_root = os.path.join(a.worktree, "tests", "golden")
    else:
        exe = a.exe or a.base_exe
        if os.path.isdir(exe):
            exe = os.path.join(exe, "whirl.exe")
        bdir = os.path.dirname(exe)
    stages = [s for s in a.stages.split(",") if s]
    if "build" in stages and not (a.worktree and a.build):
        stages.remove("build")
    env = envd(a.env)
    d = os.path.join(OUT, f"{time.strftime('%Y%m%d-%H%M%S')}_{a.device}_{a.tag}")
    os.makedirs(d, exist_ok=True)
    summary = dict(device=a.device, mode=a.mode, exe=exe, env=a.env, check=a.check, stages=[], verdict=None)
    tw0 = time.time()
    qlib.log(f"gate {a.tag}: {exe} {a.device} {a.mode} env '{a.env}' check {a.check}; stages {','.join(stages)}; logs {d}")

    def record(name, ok, dt, msg):
        summary["stages"].append(dict(stage=name, ok=ok, seconds=round(dt, 1), detail=msg))
        qlib.log(f"  {'PASS' if ok else 'FAIL'} {name} ({dt:.0f} s): {msg}")
        return ok

    ok = True
    if "build" in stages:
        ok = record("build", *stage_build(a, d))
    if ok and not os.path.isfile(exe):
        record("setup", False, 0.0, f"no executable {exe}")
        summary["verdict"] = "ERROR"
        json.dump(summary, open(os.path.join(d, "summary.json"), "w", encoding="utf-8"), indent=1)
        return 2
    gpu = [s for s in stages if s != "build"]
    if ok and gpu:
        with qlib.GpuLock():
            for s in gpu:
                if s == "bitwise":
                    if a.kt != "none":
                        ok = record("kernel_test", *stage_kt(a, d, bdir, env))
                        if not ok:
                            break
                    ok = record("golden_bitwise" if a.check == "bitwise" else "nmse",
                                *(stage_golden(a, d, exe, env) if a.check == "bitwise" else stage_nmse(a, d, exe, env)))
                elif s == "kl":
                    ok = record("kl_quick", *stage_kl(a, d, exe, env))
                elif s == "perf":
                    ok = record("perf", *stage_perf(a, d, exe, env))
                else:
                    record(s, False, 0.0, "unknown stage")
                    ok = False
                if not ok:
                    break
    summary["verdict"] = "PASS" if ok else "FAIL at " + summary["stages"][-1]["stage"]
    summary["wall_s"] = round(time.time() - tw0, 1)
    json.dump(summary, open(os.path.join(d, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(f"\n== gate {a.tag} ({a.device} {a.mode}, check {a.check}, env '{a.env}'): {summary['verdict']}  wall {summary['wall_s']:.0f} s")
    for s in summary["stages"]:
        print(f"   {'PASS' if s['ok'] else 'FAIL'}  {s['stage']:15s} {s['seconds']:6.0f} s  {s['detail'][:300]}")
    print(f"   summary {os.path.join(d, 'summary.json')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
