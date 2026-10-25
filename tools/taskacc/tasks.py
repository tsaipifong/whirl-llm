"""Task set builder for the WHIRL task-accuracy eval (feat027-taskacc).

The task data is NOT part of the repository (owner rule: data stays local). It is read from
the downloaded upstream files (each in its own directory) and from a local translation file:

  HumanEval     openai/human-eval data/HumanEval.jsonl.gz         MIT
  HumanEval-XL  FloatAI/HumanEval-XL data/python/Chinese.jsonl     MIT (Chinese docstrings of 80 HumanEval tasks)
  MBPP          google-research mbpp/sanitized-mbpp.json           CC-BY-4.0
  MBPP-zh       mbpp_zh.json (our Traditional Chinese translation of 60 MBPP descriptions; CC-BY-4.0 adaptation)

Sets:
  core200  80 XL-zh + 60 MBPP-zh (Chinese prompts, English code)  +  60 HumanEval-en (not in XL)  = 200
  full     HumanEval 164 en + XL-zh 80 + MBPP test split en (257) + MBPP-zh 60                    = 561

Each task: id, set, source, license, lang, kind ("he" = function completion with HumanEval
check(candidate), "mbpp" = asserts), entry_point, prompt (the user message), prompt_code (HE only,
the signature + docstring), test, test_imports, canonical (reference solution, for the grader selftest).
"""
import gzip
import hashlib
import json
import os
import random

DEFAULT_SRC = {
    "humaneval": r"C:\ClaudeWorkspace\downloads\feat027_humaneval\HumanEval.jsonl.gz",
    "xl_zh": r"C:\ClaudeWorkspace\downloads\feat027_humanevalxl\Chinese.jsonl",
    "mbpp": r"C:\ClaudeWorkspace\downloads\feat027_mbpp\sanitized-mbpp.json",
    "mbpp_zh": r"C:\ClaudeWorkspace\whirl-work\scratch\feat027\data\mbpp_zh.json",
}
LICENSES = {
    "humaneval": "MIT (openai/human-eval)",
    "xl_zh": "MIT (FloatAI/HumanEval-XL)",
    "mbpp": "CC-BY-4.0 (google-research MBPP, Austin et al. 2021)",
    "mbpp_zh": "CC-BY-4.0 adaptation (MBPP, translated to Traditional Chinese by WHIRL feat027)",
}
SEL_SEED_MBPP_ZH = 27     # random.Random(27).sample(test split, 60): the ids in mbpp_zh.json
SEL_SEED_HE_EN = 1027     # random.Random(1027).sample(HumanEval not in XL, 60)

INSTR_HE = {
    "en": ("Complete the following Python function. Reply with the complete function (signature, "
           "body and any imports it needs) in a single ```python code block."),
    "zh": "請完成下面的 Python 函式。請把完整的函式（包含函式簽名、函式本體和需要的 import）放在一個 ```python 程式碼區塊中回覆。",
}
INSTR_MBPP = {
    "en": ("{desc}\nYour code should pass this test:\n```python\n{test}\n```\n"
           "Reply with the complete Python code in a single ```python code block."),
    "zh": "{desc}\n你的程式必須通過這個測試：\n```python\n{test}\n```\n請把完整的 Python 程式碼放在一個 ```python 程式碼區塊中回覆。",
}


def _read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _he_task(src, lang, rec, tid, prompt_code):
    return {
        "id": tid, "source": src, "license": LICENSES[src], "lang": lang, "kind": "he",
        "entry_point": rec["entry_point"], "prompt_code": prompt_code,
        "prompt": INSTR_HE[lang] + "\n\n```python\n" + prompt_code.rstrip() + "\n```",
        "test": rec["test"], "test_imports": [], "canonical": prompt_code + rec["canonical_solution"],
        "upstream_id": rec.get("he_task_id") or rec["task_id"],
    }


def _mbpp_task(src, lang, rec, desc):
    import re
    # entry point = the reference solution's function that the first test calls
    defs = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", rec["code"], flags=re.M)
    called = [d for d in defs if re.search(r"\b" + re.escape(d) + r"\s*\(", rec["test_list"][0])]
    ep = called[0] if called else (defs[-1] if defs else None)
    test = "\n".join(rec["test_list"])
    return {
        "id": f"{'mbppzh' if lang == 'zh' else 'mbpp'}/{rec['task_id']}", "source": src, "license": LICENSES[src],
        "lang": lang, "kind": "mbpp", "entry_point": ep, "prompt_code": "",
        "prompt": INSTR_MBPP[lang].format(desc=desc.strip(), test=rec["test_list"][0]),
        "test": test, "test_imports": list(rec.get("test_imports") or []), "canonical": rec["code"],
        "upstream_id": rec["task_id"],
    }


def build(name="core200", src=None):
    """Return (tasks, meta). meta holds source paths, sha256 and licenses."""
    src = dict(DEFAULT_SRC, **(src or {}))
    he = _read_jsonl(src["humaneval"])
    xl = _read_jsonl(src["xl_zh"])
    mb = json.load(open(src["mbpp"], encoding="utf-8"))
    zh = {k: v for k, v in json.load(open(src["mbpp_zh"], encoding="utf-8")).items() if not k.startswith("_")}
    # which HumanEval task each XL task is: same entry point AND the same test (entry points repeat in
    # HumanEval: add = /53 and /85, solve = /84 and /161, sum_squares = /133 and /142)
    norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines())  # noqa: E731
    xl_he = set()
    for x in xl:
        m = [h["task_id"] for h in he if h["entry_point"] == x["entry_point"] and norm(h["test"]) == norm(x["test"])]
        if len(m) != 1:
            raise ValueError(f"HumanEval-XL {x['task_id']} ({x['entry_point']}) matches {len(m)} HumanEval tasks")
        x["he_task_id"] = m[0]
        xl_he.add(m[0])
    mb_test = sorted([m for m in mb if 11 <= m["task_id"] <= 510], key=lambda m: m["task_id"])
    mb_by_id = {m["task_id"]: m for m in mb_test}

    sel_zh = sorted(random.Random(SEL_SEED_MBPP_ZH).sample(mb_test, 60), key=lambda m: m["task_id"])
    if sorted(int(k) for k in zh) != [m["task_id"] for m in sel_zh]:
        raise ValueError("mbpp_zh.json ids do not match the seeded MBPP selection")

    tasks = []
    # Chinese HumanEval-XL: the test is the upstream HumanEval test (XL copies it; checked in selftest)
    for x in xl:
        tasks.append(_he_task("xl_zh", "zh", x, "xlzh/" + x["task_id"].split("/")[1] + "_" + x["entry_point"], x["prompt"]))
    for m in sel_zh:
        tasks.append(_mbpp_task("mbpp_zh", "zh", m, zh[str(m["task_id"])]))
    if name == "core200":
        rest = sorted([h for h in he if h["task_id"] not in xl_he], key=lambda h: int(h["task_id"].split("/")[1]))
        sel_en = sorted(random.Random(SEL_SEED_HE_EN).sample(rest, 60), key=lambda h: int(h["task_id"].split("/")[1]))
        for h in sel_en:
            tasks.append(_he_task("humaneval", "en", h, "he/" + h["task_id"].split("/")[1], h["prompt"]))
    elif name == "full":
        for h in he:
            tasks.append(_he_task("humaneval", "en", h, "he/" + h["task_id"].split("/")[1], h["prompt"]))
        for m in mb_test:
            tasks.append(_mbpp_task("mbpp", "en", m, m["prompt"]))
    else:
        raise ValueError(f"unknown task set {name!r} (core200 | full)")
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    meta = {
        "set": name, "n": len(tasks),
        "by_lang": {l: sum(1 for t in tasks if t["lang"] == l) for l in ("zh", "en")},
        "by_source": {s: sum(1 for t in tasks if t["source"] == s) for s in LICENSES},
        "sources": {k: {"path": p, "sha256": file_sha256(p), "license": LICENSES[k]} for k, p in src.items()},
        "selection": {"mbpp_zh_seed": SEL_SEED_MBPP_ZH, "he_en_seed": SEL_SEED_HE_EN, "mbpp_test_split": "task_id 11-510"},
    }
    h = hashlib.sha256()
    for t in tasks:
        h.update(json.dumps([t["id"], t["prompt"], t["test"], t["test_imports"]], ensure_ascii=False).encode("utf-8"))
    meta["taskset_sha256"] = h.hexdigest()
    return tasks, meta


def write(tasks, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"tasks_{meta['set']}.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"tasks_{meta['set']}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return p


def load(path):
    return _read_jsonl(path)
