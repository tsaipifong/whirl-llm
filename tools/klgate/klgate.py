#!/usr/bin/env python3
# KL gate (FIX-TEST-B, MOE-BAL, KL-ROBUST, KL-ROBUST-2, MODES-MXFP4): balance and fast against precise, on a fixed corpus, at fixed context lengths.
# SPDX-License-Identifier: Apache-2.0
#
# For every model x prompt x mode, one `whirl chat` run prefills the whole prompt (raw, no MTP) and
# dumps the teacher-forced next-token logits of --positions (default 3) windows of --rows (default 512)
# positions near each length in --lengths (WHIRL_DUMP_LOGITS + WHIRL_DUMP_AT; 16-row forwards, every
# row computed: d61f6df). Per window it computes KL(precise || mode) (mean over the rows) and the top-1
# agreement with precise.
# Window positions (KL-ROBUST, audit/KL-ROBUST.md): a window's bits depend on how the tokens before it
# were split into forwards (the chunked DeltaNet prefill runs in 64-row chunks counted from each
# forward's start, and the tool prefills in max_batch = 4096-row forwards from the previous window's
# end), so one window per length mixed sampling error with chunk-schedule jitter (MoE balance passed
# at 512 rows and failed at 256: audit/MOE-BAL.md). Each length L now has windows ending at L - o for
# o in OFFSETS, all in the same run (ascending):
#   o = 1216: window [L-1728, L-1216), right after the long 4096-row prefill, which ends on a 64-row
#             chunk boundary (the old single window's situation);
#   o =  592: window [L-1104, L-592), after an 112-row forward (one 64-row chunk + a 48-row partial);
#   o =    0: window [L-512, L) (the old window's rows), after an 80-row forward (64 + a 16-row partial).
# So the three windows follow three different chunk phases (0 / 48 / 16 rows into a DeltaNet chunk)
# and three different last-forward sizes, with 3 x 512 distinct rows per length. The MoE noise run
# uses the same windows. --positions 1 is the old single-window layout.
# Judging (KL-ROBUST-2, owner decision 2026-10-14, audit/KL-ROBUST-2.md): a single window's numbers
# jitter by +-20-35% KL / +-1.6 points top-1 with the forward split in front of it, while the mean over
# positions is stable. So per length the positions are AVERAGED FIRST (KL mean and top-1), and the
# per-window limits apply to that per-length average; the MoE mean limit stays on the mean over all
# windows. The report prints every position (not judged alone), the per-length average and the
# number each limit was applied to.
# MODES-MXFP4 (owner policy 2026-10-16): the limits are a REGRESSION GUARD per mode, not a near-lossless
# proof (that is the task-accuracy check) and never a veto on an allowed balance item (fp8, moefp8, q8dec,
# h16, gdnwmma, int8 KV). They are set from the measured baseline of each mode's definition on the two
# product models (Swift-1.5 27B MXFP4-A, Ornith-1.5-35B-A3B MXFP4; audit/MODES-MXFP4.md) with a margin, as
# the fast limits were. Limit values:
#   dense models (absolute, per-length average):
#     balance: KL <= 0.005 and top-1 >= 97%      fast: KL <= 0.03 and top-1 >= 93%
#     (balance top-1 98% -> 97% in MODES-MXFP4: Swift balance with fp8 measured 98.18% at zh 128k on the
#     R9700, a 3-row margin; KL 0.0015 max against 0.005)
#     (512 rows per window since MOE-BAL: with 128 rows one row is 0.78 points of top-1)
#   MoE models (arch *moe): relative to the noise floor N measured in the same gate run = precise vs
#     precise with the sequential DeltaNet prefill (WHIRL_GDN_SEQ=1, an equally exact f32 path), same
#     windows; N = the mean of the noise run's window KLs (all lengths x positions), N_top1 = the noise
#     run's top-1 averaged over ALL its windows (KL-ROBUST-2: not the same window, both sides jitter):
#     balance: mean KL over all windows <= 4.5 N, each length's average KL <= 7 N, its average top-1 >= N_top1 - 4.5 points
#     (MODES-MXFP4: fp8 + moefp8 back in MoE balance, R9700 baseline mean 3.48 N (code) / 2.92 N (zh), per length
#     max 4.96 N, top-1 drop max 2.91 points; MOE-BAL / MOE-BAL-3 had 2 N / 3 N / 3.0 points for the table without fp8)
#     fast:    mean KL <= 8 N, each length's average KL <= 12 N, average top-1 >= N_top1 - 8 points
#     (two equally exact paths of a 256-expert top-8 router already differ by KL 0.002-0.006 on
#     these prompts, above the dense balance limit; see audit/MOE-KL.md). The 3.0-point top-1 drop
#     admits int8 decode activations (q8dec: the q8_1 scheme llama.cpp itself uses for decode, the
#     same numeric level as llama.cpp); the KL limits are unchanged (audit/MOE-BAL-3.md).
# Exit 0 only when every case ran and passed; 1 = a limit failed; 2 = a run / input error or 0 cases.
# The JSON report (--report) holds the exe's SHA-256: tools/golden/golden.ps1 record requires a
# passing report for the same exe before it writes fast / balance references.
#
# Corpus: --corpus DIR with the files listed in corpus.sha256 next to this script (fixed texts; a
# changed file is an error, so numbers stay comparable across releases). Defaults: code = C++/HIP
# source with Chinese design notes, zh = Chinese prose docs with some code, each >= 129k tokens.
#
# usage: python klgate.py --exe build\Release\whirl.exe --model dense=Q.gguf --model moe=M.gguf
#            --corpus DIR [--device r9700|8060s] [--lengths 8192,32768,98304,131072] [--modes balance,fast]
#            [--report out.json] [--work DIR] [--wrap "python gpu_lock.py --"] [--only dense:code] [--rows 512]
#            [--positions 3]

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time

import numpy as np

LIMITS = {"balance": (0.005, 0.97), "fast": (0.03, 0.93)}  # dense: absolute (KL max, top-1 min) per length average
# MoE: (mean KL over all windows <= k_mean * N, per-length average KL <= k_win * N, its top-1 >= N_top1(all windows) - drop)
MOE_LIMITS = {"balance": (4.5, 7.0, 0.045), "fast": (8.0, 12.0, 0.08)}
ROWS = 512
# window end offsets before each length, earliest first (see the header): chunk phases 0 / 48 / 16
OFFSETS = (1216, 592, 0)
POSITIONS = 3
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def load_corpus(d):
    want = {}
    with open(os.path.join(HERE, "corpus.sha256"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split()
                want[name] = h
    out = {}
    for name, h in want.items():
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            raise SystemExit(f"klgate: corpus file missing: {p}")
        got = sha256(p)
        if got != h:
            raise SystemExit(f"klgate: corpus file changed: {p} sha256 {got}, want {h}")
        out[os.path.splitext(name)[0]] = p
    return out


def clean_env():
    # a stray WHIRL_* in the shell (WHIRL_KV, WHIRL_MODE, A/B switches) would test another config
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("WHIRL_")}


def ntokens(exe, model, prompt):
    env = clean_env()
    env["WHIRL_TOKENIZE_ONLY"] = "1"
    args = [exe, "chat", model, "@" + prompt, "--raw", "--max-tokens", "1"]  # tokenizer only: no GPU, no lock
    r = subprocess.run(args, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ids = [l for l in r.stdout.splitlines() if l and l[0].isdigit()]
    if r.returncode != 0 or not ids:
        raise RuntimeError(f"tokenize failed (exit {r.returncode}): {r.stderr[-400:]}")
    return ids[-1].count(",") + 1


def truncated_prompt(exe, model, prompt, n, want, work):
    """A copy of the prompt cut at a line end after at least want tokens (whose first want tokens are the
    original's, checked on the token ids); returns (path, tokens)."""
    text = open(prompt, encoding="utf-8").read()
    cut = int(len(text) * want / n * 1.02)
    full_ids = None
    for _ in range(8):
        end = text.find("\n", cut)
        end = len(text) if end < 0 else end + 1
        out = os.path.join(work, "trunc_" + os.path.basename(prompt))
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(text[:end])
        m = ntokens(exe, model, out)
        if m >= want:
            if full_ids is None:
                full_ids = token_ids(exe, model, prompt)
            if token_ids(exe, model, out)[:want] != full_ids[:want]:
                raise RuntimeError(f"truncated prompt {out}: first {want} tokens differ from the full file")
            return out, m
        cut = int(cut * 1.05) + 1000
    return prompt, n


def token_ids(exe, model, prompt):
    env = clean_env()
    env["WHIRL_TOKENIZE_ONLY"] = "1"
    r = subprocess.run([exe, "chat", model, "@" + prompt, "--raw", "--max-tokens", "1"], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    ids = [l for l in r.stdout.splitlines() if l and l[0].isdigit()]
    return ids[-1].split(",")


def run_dump(exe, model, prompt, mode, ends, n, dump, logf, wrap, device, extra=None, mode_arg=None):
    env = clean_env()
    env.update(extra or {})
    env.update({"WHIRL_MTP": "0", "WHIRL_DUMP_LOGITS": dump, "WHIRL_DUMP_AT": ",".join(map(str, ends)), "WHIRL_DUMP_ROWS": str(ROWS),
                "AMD_LOG_LEVEL": "1"})
    ctx = (n + 1 + 255) // 256 * 256 + 256
    args = wrap + [exe, "chat", model, "@" + prompt, "--raw", "--max-tokens", "1", "--ctx", str(ctx), mode_arg or "--" + mode]
    if device:
        args += ["--device", device]
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as fo:
        fo.write(" ".join(args) + (" [" + " ".join(f"{k}={v}" for k, v in extra.items()) + "]" if extra else "") + "\n")
        fo.flush()
        r = subprocess.run(args, env=env, stdout=fo, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0


def load_dump(p, nwin):
    # kept in f16 (512 rows x 4 windows x a 248k vocab = 1 GB); compare() widens one window at a time
    a = np.fromfile(p, dtype=np.float16)
    rows = nwin * ROWS
    if a.size == 0 or a.size % rows:
        raise RuntimeError(f"dump {p}: {a.size} halfs, not a multiple of {rows} rows")
    return a.reshape(rows, a.size // rows)


def log_info(logf):
    # the model arch ("  arch qwen35moe  layers ...") and the numerics line of a run's log
    arch, num = "", ""
    try:
        with open(logf, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not arch:
                    m = re.match(r"\s*arch (\S+)", line)
                    if m:
                        arch = m.group(1)
                if not num and line.strip().startswith("numerics:"):
                    num = line.strip()
    except OSError:
        pass
    return arch, num


def logsm(x):
    z = x - x.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def compare(ref, test):
    ref = ref.astype(np.float32)
    test = test.astype(np.float32)
    lr = logsm(ref)
    kl = (np.exp(lr) * (lr - logsm(test))).sum(1)
    top1 = (ref.argmax(1) == test.argmax(1)).mean()
    return float(kl.mean()), float(np.quantile(kl, 0.99)), float(kl.max()), float(top1)


def length_groups(groups):
    """The distinct length groups in order of first appearance, each with its window indices."""
    out = {}
    for w, g in enumerate(groups):
        out.setdefault(g, []).append(w)
    return list(out.items())


def judge(res, groups, poss, md, moe, noise=None):
    """Judge one model x prompt x mode (KL-ROBUST-2, owner decision 2026-10-14).

    res = [(end, zero_rows, kl, p99, max, top1)] per window; groups[w] = the length group of window w;
    poss[w] = its position; noise = {"n_mean": N, "windows": [{"top1": ...}, ...]} (MoE only).
    Per length the positions are averaged FIRST (KL mean and top-1), and the per-window limits apply to
    that per-length average: dense absolute (KL <= kl_max, top-1 >= top1_min); MoE KL <= k_win * N and
    top-1 >= N_top1 - drop, where N_top1 = the noise run's top-1 averaged over ALL its windows (not the
    same window: both sides carry chunk-schedule jitter). The MoE mean limit stays on the mean over all
    windows (k_mean * N). A window with all-zero rows fails its length. Returns the judged cases: one
    per length (rule dense-length / moe-length, with the per-position values under "positions") and,
    for MoE, one moe-mean case."""
    cases = []
    if moe:
        k_mean, k_win, drop = MOE_LIMITS[md]
        nm = noise["n_mean"]
        nt1 = float(np.mean([x["top1"] for x in noise["windows"]]))
        kl_lim, t1_lim = k_win * nm, nt1 - drop
    else:
        kl_lim, t1_lim = LIMITS[md]
    for g, ix in length_groups(groups):
        kl = float(np.mean([res[w][2] for w in ix]))
        top1 = float(np.mean([res[w][5] for w in ix]))
        zt = sum(res[w][1] for w in ix)
        ok = zt == 0 and kl <= kl_lim + 1e-12 and top1 >= t1_lim - 1e-9
        pos = [{"pos": poss[w], "end": res[w][0], "kl_mean": res[w][2], "kl_p99": res[w][3], "kl_max": res[w][4], "top1": res[w][5],
                "zero_rows": res[w][1]} for w in ix]
        c = {"length": g, "group": g, "rule": "moe-length" if moe else "dense-length", "kl_mean": kl, "top1": top1, "zero_rows": zt,
             "kl_limit": kl_lim, "top1_limit": t1_lim, "kl_p99": max(p["kl_p99"] for p in pos), "kl_max": max(p["kl_max"] for p in pos),
             "positions": pos, "pass": ok}
        if moe:
            c.update(x_noise=kl / nm, noise_top1=nt1)
        cases.append(c)
    if moe:
        mean = float(np.mean([x[2] for x in res]))
        cases.append({"length": 0, "rule": "moe-mean", "kl_mean": mean, "kl_limit": k_mean * nm, "x_noise": mean / nm,
                      "pass": mean <= k_mean * nm + 1e-12})
    return cases


def case_lines(mname, pname, md, c, noise=None):
    """Report lines for one judged case: every position's values, the per-length average and the
    number each limit was applied to."""
    hd = f"{mname:6s} {pname:5s} {md:8s}"
    verdict = "PASS" if c["pass"] else "FAIL"
    if c["rule"] == "moe-mean":
        return [f"  {verdict} {hd}    mean over all windows: KL {c['kl_mean']:.5f} = {c['x_noise']:.2f} N "
                f"<= {c['kl_limit']:.5f} ({c['kl_limit'] / noise['n_mean']:g} N)?"]
    out = []
    nm = noise["n_mean"] if noise else None
    for p in c["positions"]:
        out.append(f"    pos  {hd} {c['length']:7d}/p{p['pos']} (end {p['end']}): KL {p['kl_mean']:.5f}"
                   + (f" = {p['kl_mean'] / nm:.2f} N" if nm else "") + f" p99 {p['kl_p99']:.4f} max {p['kl_max']:.4f}"
                   f"  top1 {p['top1'] * 100:.2f}%" + (f"  zero rows {p['zero_rows']}" if p["zero_rows"] else "") + "  (not judged alone)")
    kl_s = f"KL {c['kl_mean']:.5f}" + (f" = {c['x_noise']:.2f} N" if nm else "")
    kl_l = f"{c['kl_limit']:.5f}" + (f" ({c['kl_limit'] / nm:g} N)" if nm else "")
    t1_l = f"{c['top1_limit'] * 100:.2f}%" + (f" (noise top1 all-window mean {c['noise_top1'] * 100:.2f}% - "
                                              f"{(c['noise_top1'] - c['top1_limit']) * 100:.1f}pt)" if nm else "")
    out.append(f"  {verdict} {hd} {c['length']:7d} avg of {len(c['positions'])} positions: {kl_s} <= {kl_l}?  "
               f"top1 {c['top1'] * 100:.2f}% >= {t1_l}?" + (f"  zero rows {c['zero_rows']}" if c["zero_rows"] else ""))
    return out


def rejudge(src, out):
    """Judge an earlier report's per-window numbers with the current limits (no runs). The numbers come
    from src's runs of src's exe, so exe_sha256 / device / corpus / lengths / positions are carried over."""
    if not out:
        print("klgate: --rejudge needs --report")
        return 2
    with open(src, encoding="utf-8") as f:
        old = json.load(f)
    if old.get("mode_args"):
        print("klgate: --rejudge of a negative test (--as) report")
        return 2
    rep = {k: v for k, v in old.items() if k not in ("cases", "pass", "summary", "limits", "moe_limits")}
    rep.update(limits=LIMITS, moe_limits=MOE_LIMITS, cases=[], rejudged_from=os.path.abspath(src),
               rejudged=time.strftime("%Y-%m-%d %H:%M:%S"), errors=list(old.get("errors", [])))
    noise_by = {(n["model"], n["prompt"]): n for n in old.get("noise", [])}
    seen = []
    for c in old["cases"]:
        key = (c["model"], c["prompt"], c["mode"])
        if c["rule"] == "moe-mean" or key in seen:
            continue
        seen.append(key)
        res, groups, poss = [], [], []
        for x in old["cases"]:
            if (x["model"], x["prompt"], x["mode"]) != key or x["rule"] == "moe-mean":
                continue
            for p in x["positions"]:
                res.append((p["end"], p["zero_rows"], p["kl_mean"], p["kl_p99"], p["kl_max"], p["top1"]))
                groups.append(x["group"])
                poss.append(p["pos"])
        moe = c["rule"].startswith("moe")
        noise = noise_by.get(key[:2]) if moe else None
        if moe and noise is None:
            rep["errors"].append(f"{key[0]}:{key[1]}: MoE noise floor missing in {src}")
            continue
        base = {"model": key[0], "prompt": key[1], "mode": key[2]}
        for j in judge(res, groups, poss, key[2], moe, noise):
            rep["cases"].append({**base, **j})
            for line in case_lines(key[0], key[1], key[2], j, noise):
                log(line)
    npass = sum(1 for c in rep["cases"] if c["pass"])
    nfail = len(rep["cases"]) - npass
    rc = 2 if rep["errors"] or not rep["cases"] else (1 if nfail else 0)
    rep["pass"] = rc == 0
    rep["summary"] = f"{npass} passed, {nfail} failed, {len(rep['errors'])} errors (rejudged)"
    log(f"klgate: rejudged {src}: {rep['summary']} -> {'PASS' if rc == 0 else 'FAIL'} (exit {rc})")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rc


def main():
    global ROWS, POSITIONS
    ap = argparse.ArgumentParser(description="KL gate: balance / fast vs precise")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH (repeat)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default="r9700")  # explicit: since 0.2.3 the default GPU is the one with the most memory
    ap.add_argument("--lengths", default="8192,32768,98304,131072")
    ap.add_argument("--modes", default="balance,fast")
    ap.add_argument("--report", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--wrap", default="", help="command prefix for every run (a GPU lock)")
    ap.add_argument("--only", default="", help="MODEL:PROMPT[,...] subset")
    ap.add_argument("--keep", action="store_true", help="keep the logits dumps")
    ap.add_argument("--rows", type=int, default=ROWS, help=f"rows per window (default {ROWS})")
    ap.add_argument("--as", dest="as_", action="append", default=[], metavar="MODE=ARG",
                    help="negative test: run MODE with the CLI argument ARG (e.g. balance=--balance=fp8,moefp8,gdnwmma,h16,specsample), "
                         "judged by MODE's limits; the report never passes (pass false), so it cannot feed golden record")
    ap.add_argument("--mode-env", dest="mode_env", action="append", default=[], metavar="MODE=VAR=VALUE",
                    help="candidate test: set environment VAR=VALUE (a WHIRL_* A/B switch, e.g. WHIRL_ATTN_PF2=q8pz) for MODE's runs "
                         "only (the precise reference keeps the defaults); judged by MODE's limits; the report never passes")
    ap.add_argument("--truncate", action="store_true",
                    help="prefill each prompt only up to the longest length (quick checks; same window numbers)")
    ap.add_argument("--noise-dense", action="store_true",
                    help="also run the noise floor (precise + WHIRL_GDN_SEQ=1) on dense models: reported, not judged")
    ap.add_argument("--rejudge", default="", metavar="REPORT",
                    help="no runs: judge the per-window numbers of an earlier report with the current limits and write "
                         "--report (exe_sha256 / device / corpus carried over, rejudged_from recorded)")
    ap.add_argument("--positions", type=int, default=POSITIONS, choices=[1, 2, 3],
                    help=f"windows per length (default {POSITIONS}; 1 = the single window before each length)")
    a = ap.parse_args()
    if a.rejudge:
        return rejudge(a.rejudge, a.report)
    ROWS = a.rows
    POSITIONS = a.positions
    offsets = OFFSETS[len(OFFSETS) - POSITIONS:]
    if POSITIONS > 1 and ROWS > 512:
        print("klgate: --rows > 512 with several positions: the windows would overlap (the offsets assume 512)")
        return 2

    exe = os.path.abspath(a.exe)
    if not os.path.isfile(exe):
        print(f"klgate: no exe {exe}")
        return 2
    models = {}
    for m in a.model:
        name, _, path = m.partition("=")
        if not path or not os.path.isfile(path):
            print(f"klgate: bad --model {m}")
            return 2
        models[name] = path
    modes = [x for x in a.modes.split(",") if x]
    for md in modes:
        if md not in LIMITS:
            print(f"klgate: unknown mode {md}")
            return 2
    mode_args = {}
    for x in a.as_:
        md, _, arg = x.partition("=")
        if md not in modes or not arg.startswith("--"):
            print(f"klgate: bad --as {x} (MODE must be in --modes, ARG a CLI flag)")
            return 2
        mode_args[md] = arg
    mode_env = {}
    for x in a.mode_env:
        md, _, kv = x.partition("=")
        var, _, val = kv.partition("=")
        if md not in modes or not var:
            print(f"klgate: bad --mode-env {x} (MODE=VAR=VALUE, MODE in --modes)")
            return 2
        mode_env.setdefault(md, {})[var] = val
    lengths = sorted(int(x) for x in a.lengths.split(","))
    corpus = load_corpus(a.corpus)
    only = set(x for x in a.only.split(",") if x)
    wrap = shlex.split(a.wrap, posix=False) if a.wrap else []
    work = os.path.abspath(a.work or os.path.join(os.path.dirname(exe), "klgate_work"))
    os.makedirs(work, exist_ok=True)
    rep = {"exe": exe, "exe_sha256": sha256(exe), "device": a.device or "default", "limits": LIMITS, "moe_limits": MOE_LIMITS,
           "rows": ROWS, "positions": POSITIONS, "offsets": list(offsets), "mode_args": mode_args, "mode_env": mode_env, "lengths": lengths, "modes": modes, "models": {k: os.path.basename(v) for k, v in models.items()},
           "corpus": {k: sha256(v) for k, v in corpus.items()}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": [],
           "judge": "length-average", "noise": [], "numerics": {}, "errors": []}
    log(f"klgate: exe {exe} sha256 {rep['exe_sha256'][:16]} device {rep['device']} rows {ROWS} positions {POSITIONS} (offsets {list(offsets)})")

    for mname, mpath in models.items():
        for pname, ppath in corpus.items():
            if only and f"{mname}:{pname}" not in only:
                continue
            try:
                n = ntokens(exe, mpath, ppath)
                if a.truncate and n > lengths[-1] + 64:
                    # FASTLANE kl_quick: prefill only up to the longest length (the whole corpus file is 129k+
                    # tokens). The windows' bits do not depend on the tokens after the last window (the forward
                    # split before each window is counted from the previous window's end), so the numbers are
                    # those of the full-file run.
                    ppath, n = truncated_prompt(exe, mpath, ppath, n, lengths[-1] + 64, work)
            except Exception as e:
                rep["errors"].append(f"{mname}:{pname}: {e}")
                log(f"ERROR {mname}:{pname}: {e}")
                continue
            # the windows end at each length minus the offsets (the longest length: the prompt end if the
            # prompt is shorter; it must be within 4% of the length, so the corpus really reaches it)
            ends, groups, poss = [], [], []  # window end, its length group (the group's last end), position
            for L in lengths:
                e = min(L, n)
                if e < L * 0.96:
                    rep["errors"].append(f"{mname}:{pname}: prompt has {n} tokens, too short for {L}")
                    continue
                if e - offsets[0] - ROWS < (ends[-1] if ends else 0):
                    continue  # would overlap the previous length's windows
                for p, o in enumerate(offsets):
                    ends.append(e - o)
                    groups.append(e)
                    poss.append(p)
            if not ends:
                continue
            log(f"{mname}:{pname}: {n} tokens, windows end at {ends}")
            ref = None
            res_by = {}  # mode -> per-window results (each dump is compared right after its run; only precise stays in memory)
            moe = False
            ok_runs = True
            runs = ["precise"]
            ri = 0

            def windows(t):
                out = []
                for w, e in enumerate(ends):
                    s = slice(w * ROWS, (w + 1) * ROWS)
                    zt = int((np.abs(t[s]).max(1) == 0).sum())
                    out.append((e, zt) + compare(ref[s], t[s]))  # (end, zero rows, kl, p99, max, top1)
                return out

            while ri < len(runs):
                md = runs[ri]
                ri += 1
                tag = f"{mname}_{pname}_{md}"
                dump = os.path.join(work, tag + ".f16")
                logf = os.path.join(work, tag + ".log")
                # the MoE noise floor: precise with the sequential f32 DeltaNet prefill
                extra = {"WHIRL_GDN_SEQ": "1"} if md == "noise" else None
                if md in mode_env:
                    extra = dict(extra or {}, **mode_env[md])
                rc, wall = run_dump(exe, mpath, ppath, "precise" if md == "noise" else md, ends, n, dump, logf, wrap, a.device, extra, mode_args.get(md))
                log(f"  run {tag}: exit {rc}, {wall:.0f} s")
                if rc != 0 or not os.path.isfile(dump):
                    rep["errors"].append(f"{tag}: exit {rc} (log {logf})")
                    ok_runs = False
                    break
                arch, num = log_info(logf)
                rep["numerics"][tag] = num
                try:
                    t = load_dump(dump, len(ends))
                except Exception as e:
                    rep["errors"].append(f"{tag}: {e}")
                    ok_runs = False
                    break
                if not a.keep:
                    os.remove(dump)
                if md == "precise":
                    ref = t
                    zero = int((np.abs(ref).max(1) == 0).sum())
                    if zero:
                        rep["errors"].append(f"{mname}:{pname}: {zero} all-zero rows in the precise dump")
                        ok_runs = False
                        break
                    moe = arch.endswith("moe")
                    runs += (["noise"] if moe or a.noise_dense else []) + modes
                    log(f"  {mname}: arch {arch or '?'} -> {'MoE: limits relative to the noise floor' if moe else 'dense: absolute limits'}")
                    continue
                if t.shape != ref.shape:
                    rep["errors"].append(f"{mname}:{pname}:{md}: dump shape {t.shape} != precise {ref.shape}")
                    continue
                res_by[md] = windows(t)
                del t
            ref = None
            if not ok_runs:
                continue

            noise = None
            if moe or "noise" in res_by:
                if "noise" not in res_by:
                    rep["errors"].append(f"{mname}:{pname}: MoE noise floor run missing")
                    continue
                nw = res_by["noise"]
                n_mean = float(np.mean([x[2] for x in nw]))
                noise = {"model": mname, "prompt": pname, "n_mean": n_mean, "judged": moe,
                         "windows": [{"length": x[0], "group": groups[w], "pos": poss[w], "kl_mean": x[2], "kl_p99": x[3], "kl_max": x[4],
                                      "top1": x[5]} for w, x in enumerate(nw)]}
                rep["noise"].append(noise)
                log(f"  noise  {mname:6s} {pname:5s}: N = {n_mean:.5f}, top1 all-window mean "
                    f"{np.mean([x[5] for x in nw]) * 100:.2f}%  (" +
                    ", ".join(f"{x[0]}: KL {x[2]:.5f} top1 {x[5] * 100:.2f}%" for x in nw) + ")")
                if n_mean <= 0 and moe:
                    rep["errors"].append(f"{mname}:{pname}: noise floor N = 0 (WHIRL_GDN_SEQ=1 did not change the bits)")
                    continue
                if not moe:
                    noise = None  # dense: reported only, absolute limits
            for md in modes:
                if md not in res_by:
                    continue
                res = res_by[md]
                base = {"model": mname, "prompt": pname, "mode": md}
                for c in judge(res, groups, poss, md, moe, noise):
                    rep["cases"].append({**base, **c})
                    for line in case_lines(mname, pname, md, c, noise):
                        log(line)

    npass = sum(1 for c in rep["cases"] if c["pass"])
    nfail = len(rep["cases"]) - npass
    rc = 2 if rep["errors"] or not rep["cases"] else (1 if nfail else 0)
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rep["pass"] = rc == 0 and not mode_args and not mode_env  # a negative test (--as) / candidate (--mode-env) never passes
    rep["summary"] = f"{npass} passed, {nfail} failed, {len(rep['errors'])} errors"
    for e in rep["errors"]:
        log("ERROR " + e)
    log(f"klgate: {rep['summary']} -> {'PASS' if rc == 0 else 'FAIL'} (exit {rc})" + (f"  [negative test, modes run as {mode_args}]" if mode_args else "")
        + (f"  [candidate env {mode_env}]" if mode_env else ""))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
