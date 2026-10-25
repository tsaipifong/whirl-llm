"""WHIRL task-accuracy eval (feat027-taskacc): execution-graded coding tasks through whirl-server,
paired comparison of two configurations with McNemar and the recovery ratio vs precise.

Run with the isolated interpreter:  python -I tools\\taskacc\\taskacc.py <command> ...

  build    --set core200|full --out DIR                 write DIR\\tasks_<set>.jsonl (+ .meta.json: sources, sha256, licenses)
  build    --set long --gguf MODEL.gguf --out DIR [--lengths 32k,64k,96k,128k] [--contexts 6] [--n-code 4] [--n-kv 6]
           long-context suite (longctx.py, feat027-longacc): repo-snapshot contexts, execution-graded coding needles
           (two hops + deprecated-section decoy) and key/value needles with near-duplicate decoys
  validate --tasks FILE [--workers 4]                   reference solutions must pass, stub solutions must fail (grader check)
  run      --tasks FILE --url http://127.0.0.1:PORT --out RES.jsonl --label NAME
           [--samples 1] [--temperature 0|card|T] [--seed0 0] [--think off|on] [--effort low|medium|high]
           [--max-tokens 2048] [--concurrency 4] [--workers 4] [--timeout 10] [--limit N] [--max-wall-min M]
           resumable: rows already in RES.jsonl (same key) are skipped
  regrade  RES.jsonl --tasks FILE --out RES2.jsonl      re-run the grader on stored replies (grader fixes)
  report   RES.jsonl                                    accuracy of one run, by language / source, status counts
  compare  A.jsonl B.jsonl [--json OUT] [--boot 10000]  paired: McNemar, recovery vs A, bootstrap CI, verdicts
  plan     --n 200 --acc 0.85                           planning: minimum drop McNemar can flag [estimate]
  selftest                                              CPU selftests (test_taskacc.py, test_longacc.py)
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client as C          # noqa: E402
import extract as X         # noqa: E402
import sandbox as S         # noqa: E402
import stats as T           # noqa: E402
import tasks as K           # noqa: E402

VERSION = "taskacc 1 + longacc 1 (2026-10-26)"


def _out(*a):
    print(*a, flush=True)


def _subsets(rec):
    s = ["all", "lang=" + rec.get("lang", "?"), "src=" + rec.get("source", "?")]
    if rec.get("ctx"):           # long-context suite
        s += ["ctx=" + rec["ctx"], "ctx=" + rec["ctx"] + "/" + rec.get("source", "?")]
    return s


def _grade(t, content, timeout):
    """Grade one reply: kv = exact retrieval (no code), everything else = extract + sandbox."""
    if t["kind"] == "kv":
        import longctx as LC  # noqa: PLC0415
        return LC.grade_kv(t, content), None
    code = X.extract(content, t["entry_point"])
    return S.grade(t, code, timeout=timeout), code


# ------------------------------------------------------------------ build / validate
def cmd_build(a):
    if a.set == "long":
        import bpe as B  # noqa: PLC0415
        import longctx as LC  # noqa: PLC0415
        if not a.gguf:
            _out("ERROR build --set long needs --gguf (the model whose tokenizer sizes the contexts)")
            return 2
        t0 = time.time()
        tok = B.BPE.from_gguf(a.gguf)
        tasks, meta = LC.build(tok, lengths=a.lengths.split(","), contexts=a.contexts, n_code=a.n_code, n_kv=a.n_kv,
                               seed=a.seed)
        meta["tokenizer_gguf"] = a.gguf
        p, meta = LC.write(tasks, meta, a.out, name=a.name or "tasks_long")
        _out(f"wrote {p}: {meta['n']} tasks {meta['by_kind']} {meta['by_lang']} in {time.time() - t0:.0f} s; "
             f"taskset_sha256 {meta['taskset_sha256'][:16]}")
        for c in meta["contexts"]:
            _out(f"  {c['cluster']:10s} target {c['target']:6d} budget {c['prompt_budget']:6d} "
                 f"context ~{c['context_tokens_est']:6d} tok [E]")
        return 0
    tasks, meta = K.build(a.set)
    p = K.write(tasks, meta, a.out)
    _out(f"wrote {p}: {meta['n']} tasks {meta['by_lang']} {meta['by_source']}  taskset_sha256 {meta['taskset_sha256'][:16]}")
    return 0


STUBS = {"he": "def {ep}(*args, **kwargs):\n    return None\n", "mbpp": "def {ep}(*args, **kwargs):\n    return None\n"}


def cmd_validate(a):
    tasks = K.load(a.tasks)
    if a.limit:
        tasks = tasks[:a.limit]
    bad_ref, stub_pass = [], []

    bad_decoy = []

    def one(t):
        if t["kind"] == "kv":       # longacc retrieval: answer passes; decoy, empty, answer + another value fail
            import longctx as LC  # noqa: PLC0415
            r = LC.grade_kv(t, t["answer"])
            s = {"passed": any(LC.grade_kv(t, x)["passed"]
                               for x in (t["decoy_answer"], "", "amber-cobalt-1234 " + t["answer"]))}
            return t, r, s
        r = S.grade(t, t["canonical"], timeout=a.timeout)
        stub = "" if t["kind"] == "lc" else STUBS[t["kind"]].format(ep=t["entry_point"])  # lc: the module's own stub
        s = S.grade(t, stub, timeout=a.timeout)
        if t["kind"] == "lc":       # the deprecated (decoy) section implemented exactly must fail
            d = S.grade(t, t["decoy"], timeout=a.timeout)
            if d["passed"]:
                bad_decoy.append(t["id"])
        return t, r, s
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for t, r, s in ex.map(one, tasks):
            if not r["passed"]:
                bad_ref.append((t["id"], r["status"], r["stderr"].strip().splitlines()[-1:] if r["stderr"] else ""))
            if s["passed"]:
                stub_pass.append(t["id"])
            if not r.get("jobbed") and os.name == "nt" and t["kind"] != "kv":
                _out(f"WARNING {t['id']}: process not in the job object")
    _out(f"validate {a.tasks}: {len(tasks)} tasks in {time.time() - t0:.1f} s; reference fails {len(bad_ref)}, stub passes {len(stub_pass)}")
    for x in bad_ref:
        _out("  REF FAIL", *x)
    for x in stub_pass:
        _out("  STUB PASS", x)
    if any(t["kind"] == "lc" for t in tasks):
        _out(f"  decoy (deprecated-section) implementations passing: {len(bad_decoy)}")
    for x in bad_decoy:
        _out("  DECOY PASS", x)
    return 0 if not bad_ref and not stub_pass and not bad_decoy else 1


# ------------------------------------------------------------------ run
def _load_rows(path):
    meta, rows = None, []
    if not os.path.exists(path):
        return meta, rows
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            j = json.loads(ln)
            if "meta" in j:
                meta = meta or j["meta"]
            else:
                rows.append(j)
    return meta, rows


def cmd_run(a):
    tasks = K.load(a.tasks)
    if a.ctx:   # long suite: only these length buckets (e.g. "32k,64k")
        tasks = [t for t in tasks if t.get("ctx") in a.ctx.split(",")]
    if a.limit:
        tasks = tasks[:a.limit]
    cl = C.Client(a.url, timeout=a.request_timeout)
    if not cl.health():
        _out(f"ERROR server {a.url} not healthy")
        return 3
    models = cl.models()
    model = a.model or (models[0] if models else "whirl")
    props = cl.props()
    temp = a.temperature if a.temperature == "card" else float(a.temperature)
    meta = {"tool": VERSION, "label": a.label, "url": a.url, "model": model, "models": models,
            "numerics": props.get("numerics") if isinstance(props, dict) else None,
            "props_keys": sorted(props.keys()) if isinstance(props, dict) else None,
            "tasks": os.path.abspath(a.tasks), "tasks_sha256": K.file_sha256(a.tasks),
            "samples": a.samples, "temperature": temp, "seed0": a.seed0, "think": a.think, "effort": a.effort,
            "max_tokens": a.max_tokens, "concurrency": a.concurrency, "timeout_s": a.timeout,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "extra": a.extra}
    old_meta, old = _load_rows(a.out)
    done = {(r["task_id"], r.get("seed", 0)) for r in old if not r.get("invalid")}
    if old_meta:
        for k in ("tasks_sha256", "temperature", "think", "max_tokens", "model"):
            if old_meta.get(k) != meta.get(k):
                _out(f"ERROR resume: {k} differs ({old_meta.get(k)!r} in file vs {meta.get(k)!r}); use a new --out")
                return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    lock = threading.Lock()
    fh = open(a.out, "a", encoding="utf-8", newline="\n")
    if not old_meta:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        fh.flush()
    jobs = [(t, a.seed0 + s) for t in tasks for s in range(a.samples) if (t["id"], a.seed0 + s) not in done]
    _out(f"run {a.label}: model {model}; numerics {json.dumps(meta['numerics'], ensure_ascii=False)[:300]}")
    _out(f"run {a.label}: {len(jobs)} samples to do ({len(done)} already in {a.out}); concurrency {a.concurrency}")
    t0 = time.time()
    stop = threading.Event()
    cnt = {"n": 0, "pass": 0, "invalid": 0}

    def one(job):
        t, seed = job
        if stop.is_set():
            return None
        mt = a.max_tokens if (a.force_max_tokens or not t.get("max_tokens")) else t["max_tokens"]
        r = cl.chat(model, t["prompt"], temperature=temp, max_tokens=mt,
                    seed=seed if (temp == "card" or temp > 0) else None, think=(a.think == "on"), effort=a.effort)
        rec = {"task_id": t["id"], "seed": seed, "lang": t["lang"], "source": t["source"], "label": a.label,
               "finish": r["finish"], "usage": r["usage"], "seconds": r["seconds"], "content": r["content"],
               "reasoning_chars": len(r.get("reasoning") or "")}
        if t.get("ctx"):   # long-context suite: bucket, cluster (shared context), depth, measured prompt tokens
            u = r.get("usage") or {}
            rec.update({"ctx": t["ctx"], "cluster": t.get("cluster"), "depth": t.get("depth"), "kind": t["kind"],
                        "max_tokens": mt, "prompt_tokens": u.get("prompt_tokens"),
                        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens")})
        if not r["ok"]:
            rec.update({"invalid": True, "error": r["error"], "passed": False, "status": "request_error"})
        else:
            g, code = _grade(t, r["content"], a.timeout)
            rec.update({"passed": g["passed"], "status": g["status"], "grade_seconds": g["seconds"],
                        "stderr_tail": g["stderr"][-400:], "code_found": code is not None or t["kind"] == "kv"})
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            cnt["n"] += 1
            cnt["pass"] += bool(rec["passed"])
            cnt["invalid"] += bool(rec.get("invalid"))
            if cnt["n"] % 10 == 0 or cnt["n"] == len(jobs):
                el = time.time() - t0
                _out(f"  {cnt['n']}/{len(jobs)} done, pass {cnt['pass']}, invalid {cnt['invalid']}, "
                     f"{el / 60:.1f} min, eta {el / cnt['n'] * (len(jobs) - cnt['n']) / 60:.1f} min")
            if a.max_wall_min and time.time() - t0 > a.max_wall_min * 60:
                if not stop.is_set():
                    _out(f"STOP: wall cap {a.max_wall_min} min reached; the file is resumable")
                stop.set()
        return rec
    try:
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(one, jobs))
    finally:
        fh.close()
    _, rows = _load_rows(a.out)
    rows = [r for r in rows if not r.get("invalid")]
    n_ok = sum(r["passed"] for r in rows)
    complete = len({(r["task_id"], r.get("seed", 0)) for r in rows}) >= len(tasks) * a.samples
    _out(f"== {a.label}: {n_ok}/{len(rows)} passed ({100 * n_ok / max(1, len(rows)):.1f}%) "
         f"{'complete' if complete else 'INCOMPLETE'}; {(time.time() - t0) / 60:.1f} min")
    if any(t.get("ctx") for t in tasks):   # long suite: per bucket accuracy, invalid rows (e.g. slot too small), cache use
        _, allrows = _load_rows(a.out)
        for b in sorted({t["ctx"] for t in tasks}, key=lambda x: int(x.rstrip("k"))):
            rb = [r for r in allrows if r.get("ctx") == b]
            ok = [r for r in rb if not r.get("invalid")]
            pt = [r.get("prompt_tokens") or 0 for r in ok]
            ct = [r.get("cached_tokens") or 0 for r in ok]
            _out(f"   ctx {b:5s} pass {sum(r['passed'] for r in ok)}/{len(ok)}  invalid {len(rb) - len(ok)}  "
                 f"prompt_tokens max {max(pt or [0])}  cached share {sum(ct) / max(1, sum(pt)):.2f}")
            errs = sorted({str(r.get("error"))[:120] for r in rb if r.get("invalid")})
            for e in errs[:3]:
                _out(f"     invalid: {e}")
    return 0 if complete else 4


def cmd_regrade(a):
    tasks = {t["id"]: t for t in K.load(a.tasks)}
    meta, rows = _load_rows(a.res)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        meta = dict(meta or {}, regraded_by=VERSION, regraded_from=os.path.abspath(a.res))
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")

        def one(r):
            if r.get("invalid") or r["task_id"] not in tasks:
                return r
            t = tasks[r["task_id"]]
            g, code = _grade(t, r.get("content") or "", a.timeout)
            r = dict(r, passed=g["passed"], status=g["status"], stderr_tail=g["stderr"][-400:], code_found=code is not None)
            return r
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(one, rows):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _out(f"regraded {len(rows)} rows -> {a.out}")
    return 0


def cmd_report(a):
    meta, rows = _load_rows(a.res)
    rows = [r for r in rows if not r.get("invalid")]
    agg = {}
    for r in rows:
        for s in _subsets(r):
            x = agg.setdefault(s, [0, 0])
            x[0] += 1
            x[1] += bool(r["passed"])
    _out(f"{a.res}: label {meta and meta.get('label')}, model {meta and meta.get('model')}, think {meta and meta.get('think')}")
    for s, (n, p) in sorted(agg.items()):
        _out(f"  {s:18s} {p:4d}/{n:4d}  {100 * p / n:5.1f}%")
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    fin = {}
    for r in rows:
        fin[str(r.get("finish"))] = fin.get(str(r.get("finish")), 0) + 1
    toks = [((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows]
    _out(f"  status {st}; finish {fin}; completion tokens mean {sum(toks) / max(1, len(toks)):.0f}, max {max(toks or [0])}")
    return 0


def _fmt(s):
    lo, hi = s.get("recovery_ci95", [float("nan")] * 2)
    return (f"n {s['n']:4d}  A {s['pass_a']:4d} ({100 * s['acc_a']:5.1f}%)  B {s['pass_b']:4d} ({100 * s['acc_b']:5.1f}%)  "
            f"recovery {s['recovery']:.3f} [{lo:.3f}, {hi:.3f}]  A-only {s['a_only']:3d}  B-only {s['b_only']:3d}  "
            f"McNemar p2 {s['mcnemar_p_two_sided']:.3f} p(B worse) {s['mcnemar_p_b_worse']:.3f}  identical {s['identical_replies']}")


def cmd_compare(a):
    ma, ra = _load_rows(a.a)
    mb, rb = _load_rows(a.b)
    inv_a = sum(1 for r in ra if r.get("invalid"))
    inv_b = sum(1 for r in rb if r.get("invalid"))
    ra = [r for r in ra if not r.get("invalid")]
    rb = [r for r in rb if not r.get("invalid")]
    pairs, only_a, only_b = T.pair(ra, rb)
    warn = []
    for k in ("tasks_sha256", "temperature", "think", "effort", "max_tokens", "samples"):
        if (ma or {}).get(k) != (mb or {}).get(k):
            warn.append(f"{k}: A {(ma or {}).get(k)!r} vs B {(mb or {}).get(k)!r}")
    out = {"A": {"file": os.path.abspath(a.a), "label": (ma or {}).get("label"), "numerics": (ma or {}).get("numerics")},
           "B": {"file": os.path.abspath(a.b), "label": (mb or {}).get("label"), "numerics": (mb or {}).get("numerics")},
           "unpaired": {"only_a": len(only_a), "only_b": len(only_b), "invalid_a": inv_a, "invalid_b": inv_b},
           "warnings": warn, "subsets": {}}
    _out(f"A = {out['A']['label']} ({a.a})\nB = {out['B']['label']} ({a.b})")
    for side in ("A", "B"):   # confirm the server really ran the intended mode / KV format (from GET /props)
        nm = out[side]["numerics"] or {}
        _out(f"{side} numerics: mode {nm.get('mode')!r} label {nm.get('label')!r} kv {nm.get('kv')!r} "
             f"decode {nm.get('decode')!r} overrides {nm.get('overrides')!r}")
    if (out["A"]["numerics"] or {}) == (out["B"]["numerics"] or {}) and out["A"]["numerics"]:
        warn.append("A and B report the same numerics: is B really a different configuration?")
    for w in warn:
        _out("WARNING settings differ:", w)
    if only_a or only_b or inv_a or inv_b:
        _out(f"WARNING unpaired rows: only A {len(only_a)}, only B {len(only_b)}, invalid A {inv_a}, B {inv_b} (excluded)")
    groups = {}
    for p in pairs:
        for s in _subsets(p[3]):
            groups.setdefault(s, []).append(p)
    for s in sorted(groups, key=lambda x: (x != "all", x)):
        sm = T.summarize(groups[s], n_boot=a.boot if s == "all" or a.boot_subsets else 0)
        out["subsets"][s] = sm
        _out(f"{s:16s} " + (_fmt(sm) if "recovery_ci95" in sm else _fmt(dict(sm, recovery_ci95=[float('nan')] * 2))))
    al = out["subsets"].get("all", {})
    if al:
        _out("fast verdict   : " + al.get("fast_verdict", "n/a"))
        _out("balance note   : " + al.get("balance_note", "n/a"))
        disc = [(k, ra_["status"], rb_["status"]) for k, x, y, ra_, rb_ in pairs if x != y]
        out["discordant"] = [{"task_id": k[0], "seed": k[1], "a": sa, "b": sb} for k, sa, sb in disc]
        for d in out["discordant"][:40]:
            _out(f"  discordant {d['task_id']} seed {d['seed']}: A {d['a']} / B {d['b']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


def cmd_plan(a):
    d, b, drop = T.min_detectable_drop(a.n, a.acc)
    _out(f"[estimate] n {a.n}, acc_A {a.acc:.2f}: about {d} discordant pairs if errors flip at rate acc(1-acc); "
         f"McNemar one-sided 5% needs b >= {b} of {d} -> detectable drop ~{100 * drop:.1f} points "
         f"(recovery ~{(a.acc - drop) / a.acc:.3f})" if b is not None else f"[estimate] d={d}: no significant split possible")
    return 0


def cmd_selftest(a):
    import unittest
    here = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(here, pattern="test_*.py", top_level_dir=here)
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if r.wasSuccessful() else 1


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build"); p.add_argument("--set", default="core200"); p.add_argument("--out", required=True)
    p.add_argument("--gguf", default=None, help="long: model GGUF whose tokenizer sizes the contexts")
    p.add_argument("--lengths", default="32k,64k,96k,128k"); p.add_argument("--contexts", type=int, default=6)
    p.add_argument("--n-code", type=int, default=4); p.add_argument("--n-kv", type=int, default=6)
    p.add_argument("--seed", type=int, default=20261026); p.add_argument("--name", default=None)
    p = sp.add_parser("validate"); p.add_argument("--tasks", required=True); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=10.0); p.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("run")
    p.add_argument("--tasks", required=True); p.add_argument("--url", required=True); p.add_argument("--out", required=True)
    p.add_argument("--label", required=True); p.add_argument("--model", default=None)
    p.add_argument("--samples", type=int, default=1); p.add_argument("--temperature", default="0")
    p.add_argument("--seed0", type=int, default=0); p.add_argument("--think", choices=["off", "on"], default="off")
    p.add_argument("--effort", default=None); p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=4); p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--request-timeout", type=float, default=900.0); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-wall-min", type=float, default=0); p.add_argument("--extra", default="")
    p.add_argument("--ctx", default="", help="long suite: only these buckets, e.g. 32k,64k")
    p.add_argument("--force-max-tokens", action="store_true",
                   help="use --max-tokens even for tasks that set their own max_tokens (long suite)")
    p = sp.add_parser("regrade"); p.add_argument("res"); p.add_argument("--tasks", required=True); p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--timeout", type=float, default=10.0)
    p = sp.add_parser("report"); p.add_argument("res")
    p = sp.add_parser("compare"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--json", default=None)
    p.add_argument("--boot", type=int, default=10000); p.add_argument("--boot-subsets", action="store_true")
    p = sp.add_parser("plan"); p.add_argument("--n", type=int, default=200); p.add_argument("--acc", type=float, default=0.85)
    sp.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"build": cmd_build, "validate": cmd_validate, "run": cmd_run, "regrade": cmd_regrade, "report": cmd_report,
            "compare": cmd_compare, "plan": cmd_plan, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
