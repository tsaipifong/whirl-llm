"""CPU selftests for the long-context suite (feat027-longacc). Run: python -I tools\\taskacc\\taskacc.py selftest

Data-free tests use a small synthetic filler and a character-based token counter; the tests on the real
suite (scratch\\feat027\\longacc\\data\\tasks_long.jsonl) and on the GGUF tokenizer are skipped when the
local files are missing (data never goes into the repository).
"""
import http.server
import json
import os
import random
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe as B        # noqa: E402
import longctx as LC   # noqa: E402
import sandbox as S    # noqa: E402
import stats as T      # noqa: E402
import taskacc as A    # noqa: E402

SUITE = r"C:\ClaudeWorkspace\whirl-work\scratch\feat027\longacc\data\tasks_long.jsonl"
GGUF = r"C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf"


class CharTok:
    """Stand-in tokenizer for data-free tests: 1 token per 4 characters."""
    def count(self, text):
        return (len(text) + 3) // 4


def fake_files(n=40, seed=5):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = "\n".join(f"def helper_{i}_{j}(x):\n    return x * {rng.randint(2, 99)} + {j}\n" for j in range(rng.randint(20, 60)))
        out.append((f"vendor/mod{i}.py", f'"""module {i}"""\nimport os\n\n' + body))
    return out


def small_suite(contexts=2, L=8192, n_code=3, n_kv=3, seed=7):
    old = dict(LC.LENGTHS)
    try:
        LC.LENGTHS["8k"] = L
        return LC.build(CharTok(), lengths=("8k",), contexts=contexts, n_code=n_code, n_kv=n_kv, seed=seed, files=fake_files())
    finally:
        LC.LENGTHS.clear()
        LC.LENGTHS.update(old)


class Tokenizer(unittest.TestCase):
    def test_pretokenizer_splits(self):
        self.assertEqual(B.pretokenize("def hello_world(x):\n"), ["def", " hello", "_world", "(x", "):\n"])
        self.assertEqual(B.pretokenize("x = 12345"), ["x", " =", " ", "1", "2", "3", "4", "5"])
        self.assertEqual(B.pretokenize("it's  中文字"), ["it", "'s", " ", " 中文字"])
        self.assertEqual("".join(B.pretokenize("a\n\n  b_c\t!!\r\n")), "a\n\n  b_c\t!!\r\n")   # lossless split

    @unittest.skipUnless(os.path.exists(GGUF), "model GGUF missing")
    def test_gguf_bpe_known_ids(self):
        tok = B.BPE.from_gguf(GGUF)
        ids = [tok.vocab[p] for p in tok.tokenize("def hello_world(x):\n    return x + 1  # 中文註解 12345\n")]
        # ids checked against the model's own tokenizer.json (HF tokenizers, CPU; tools/taskacc/tokcheck_hf.py)
        self.assertEqual(ids[:6], [727, 23066, 30749, 2007, 1590, 198])
        self.assertEqual(ids[-6:], [220, 16, 17, 18, 19, 20][-5:] + [198])


class Templates(unittest.TestCase):
    def test_reference_passes_stub_and_decoy_fail(self):
        rng = random.Random(11)
        for T_ in LC.TEMPLATES:
            for k in range(2):
                nd = LC.make_coding_needle(T_, rng, "pkgx", 5 + k)
                task = {"kind": "lc", "entry_point": nd["entry_point"], "module_name": nd["module_name"],
                        "module_src": nd["module_src"], "test": nd["test"], "test_imports": []}
                self.assertTrue(S.grade(task, nd["reference"], timeout=10)["passed"], (T_.name, "reference"))
                self.assertFalse(S.grade(task, "", timeout=10)["passed"], (T_.name, "stub"))
                self.assertFalse(S.grade(task, nd["decoy"], timeout=10)["passed"], (T_.name, "decoy"))
                self.assertGreaterEqual(nd["decoy_asserts_differ"], 2)

    def test_module_is_importable_by_package_path(self):
        nd = LC.make_coding_needle(LC.Key, random.Random(3), "pkgy", 6)
        task = {"kind": "lc", "entry_point": "make_key", "module_name": nd["module_name"], "module_src": nd["module_src"],
                "test": nd["test"], "test_imports": []}
        code = "from pkgy.storage.keys import _short_hash as _h\n" + nd["reference"].replace("_short_hash(", "_h(")
        self.assertTrue(S.grade(task, code, timeout=10)["passed"])
        # a reply that starts with `from __future__ import annotations` still runs after the module prelude
        self.assertTrue(S.grade(task, "from __future__ import annotations\n" + nd["reference"], timeout=10)["passed"])

    def test_doc_has_current_and_deprecated_sections(self):
        nd = LC.make_coding_needle(LC.Fee, random.Random(4), "pkgz", 7)
        self.assertIn("v7", nd["doc"])
        self.assertIn("v6 (deprecated)", nd["doc"])
        self.assertIn("CURRENT_VERSION = 7", nd["module_src"])
        self.assertNotIn("v7", nd["module_src"].replace("CURRENT_VERSION = 7", ""))   # the version is only via the constant


class Context(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.meta = small_suite()

    def test_counts_budget_and_order(self):
        self.assertEqual(self.meta["by_kind"], {"lc": 6, "kv": 6})
        tok = CharTok()
        for c in self.meta["contexts"]:
            self.assertLessEqual(c["context_tokens_est"], c["prompt_budget"])
            self.assertGreater(c["context_tokens_est"], 0.85 * c["prompt_budget"])
        for t in self.tasks:
            self.assertLessEqual(tok.count(t["prompt"]) + t["max_tokens"] + LC.TEMPLATE_RESERVE, t["ctx_tokens"])
        # tasks of one context are adjacent and share the exact same prefix (prefix cache)
        seen = []
        for t in self.tasks:
            if not seen or seen[-1] != t["cluster"]:
                self.assertNotIn(t["cluster"], seen)
                seen.append(t["cluster"])
        for cl in seen:
            ps = [t["prompt"][:-len(t["question"])] for t in self.tasks if t["cluster"] == cl]
            self.assertEqual(len(set(ps)), 1)

    def test_needles_unique_and_depths(self):
        for t in self.tasks:
            ctx = t["prompt"][:-len(t["question"])]
            if t["kind"] == "lc":
                self.assertEqual(ctx.count(f"def {t['entry_point']}("), 1)
                self.assertNotAlmostEqual(t["depth"], t["doc_depth"], places=2)
                self.assertTrue(0 <= t["depth"] <= 1 and 0 <= t["doc_depth"] <= 1)
            else:
                self.assertEqual(ctx.count(t["answer"]), 1)
                self.assertEqual(ctx.count(t["decoy_answer"]), 1)
                self.assertEqual(ctx.count(t["key"] + " = "), 1)
                self.assertLess(ctx.index(t["key"] + " = "), ctx.index(t["key"] + "_OLD = "))

    def test_deterministic(self):
        t2, _ = small_suite()
        self.assertEqual([t["prompt"] for t in t2], [t["prompt"] for t in self.tasks])
        t3, _ = small_suite(seed=8)
        self.assertNotEqual([t["prompt"] for t in t3], [t["prompt"] for t in self.tasks])

    def test_validate_command(self):
        with tempfile.TemporaryDirectory(dir=S.DEFAULT_ROOT if os.path.isdir(S.DEFAULT_ROOT) else None) as d:
            p, _ = LC.write(self.tasks, self.meta, d, name="t")
            self.assertEqual(A.main(["validate", "--tasks", p]), 0)


class Retrieval(unittest.TestCase):
    T = {"kind": "kv", "answer": "amber-otter-1234", "decoy_answer": "cedar-nova-9876"}

    def test_grade_kv(self):
        g = lambda s: LC.grade_kv(self.T, s)["status"]  # noqa: E731
        self.assertEqual(g("amber-otter-1234"), "pass")
        self.assertEqual(g('`"amber-otter-1234"`\n'), "pass")
        self.assertEqual(g("CFG_X=amber-otter-1234"), "pass")
        self.assertEqual(g("<think>maybe cedar-nova-9876</think>amber-otter-1234"), "pass")
        self.assertEqual(g("cedar-nova-9876"), "decoy")
        self.assertEqual(g("amber-otter-1234 or cedar-nova-9876"), "decoy")
        self.assertEqual(g("amber-otter-1235"), "wrong")
        self.assertEqual(g("I could not find it"), "no_answer")
        self.assertEqual(g(""), "no_answer")


class ClusterBootstrap(unittest.TestCase):
    def test_clusters_are_contexts(self):
        pairs = []
        for c in range(5):
            for i in range(8):
                r = {"task_id": f"c{c}/t{i}", "seed": 0, "passed": True, "cluster": f"c{c}"}
                pairs.append(((r["task_id"], 0), True, (i + c) % 3 != 0, r, dict(r)))
        self.assertEqual(T.bootstrap(pairs, n_boot=200)["clusters"], 5)
        for p in pairs:
            del p[3]["cluster"]
        self.assertEqual(T.bootstrap(pairs, n_boot=200)["clusters"], 40)


# ------------------------------------------------------------------ end to end with a mock OpenAI server
class _MockL(http.server.BaseHTTPRequestHandler):
    tasks = {}
    mode = "good"
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"status": "ok"})
        if self.path == "/v1/models":
            return self._send({"data": [{"id": "mock-model"}]})
        if self.path == "/props":
            return self._send({"numerics": {"mode": _MockL.mode, "kv": "f16" if _MockL.mode == "good" else "q4"}})
        self._send({}, 404)

    def do_POST(self):
        j = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        _MockL.seen.append({k: v for k, v in j.items() if k != "messages"})
        t = _MockL.tasks[j["messages"][0]["content"]]
        bad = _MockL.mode == "bad" and t["id"].endswith(("0_frame", "0_fee", "0_key", "0_backoff", "0_classify", "0_check", "kv1"))
        if t["kind"] == "lc":
            content = "```python\n" + (t["decoy"] if bad else t["reference"]) + "```"
        else:
            content = t["decoy_answer"] if bad else t["answer"]
        self._send({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(j["messages"][0]["content"]) // 4, "completion_tokens": 9,
                              "prompt_tokens_details": {"cached_tokens": 100}}})


class EndToEndLong(unittest.TestCase):
    def test_run_and_compare_long(self):
        tasks, meta = small_suite(contexts=2, n_code=2, n_kv=2)
        with tempfile.TemporaryDirectory(dir=S.DEFAULT_ROOT if os.path.isdir(S.DEFAULT_ROOT) else None) as d:
            tp, _ = LC.write(tasks, meta, d, name="t")
            _MockL.tasks = {t["prompt"]: t for t in tasks}
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockL)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                _MockL.mode = "good"
                ra = os.path.join(d, "a.jsonl")
                self.assertEqual(A.main(["run", "--tasks", tp, "--url", url, "--out", ra, "--label", "good", "--concurrency", "1"]), 0)
                self.assertEqual(sorted({s["max_tokens"] for s in _MockL.seen}), [LC.MAX_TOKENS_KV, LC.MAX_TOKENS_LC])
                _MockL.mode = "bad"
                rb = os.path.join(d, "b.jsonl")
                self.assertEqual(A.main(["run", "--tasks", tp, "--url", url, "--out", rb, "--label", "bad", "--concurrency", "1"]), 0)
                rows_a = [json.loads(l) for l in open(ra, encoding="utf-8")][1:]
                rows_b = [json.loads(l) for l in open(rb, encoding="utf-8")][1:]
                self.assertTrue(all(r["passed"] for r in rows_a), [(r["task_id"], r["status"]) for r in rows_a if not r["passed"]])
                self.assertTrue(all(r["ctx"] == "8k" and r["cluster"] and r["prompt_tokens"] for r in rows_a))
                n_bad = sum(1 for r in rows_b if not r["passed"])
                self.assertEqual(n_bad, 2 + 2)          # c0: one coding (lc0) + kv1; c1: the same
                self.assertEqual(sorted(r["status"] for r in rows_b if not r["passed"]), ["decoy", "decoy", "fail", "fail"])
                js = os.path.join(d, "cmp.json")
                self.assertEqual(A.main(["compare", ra, rb, "--json", js, "--boot", "300"]), 0)
                c = json.load(open(js, encoding="utf-8"))["subsets"]
                self.assertEqual(c["all"]["a_only"], 4)
                self.assertEqual(c["all"]["clusters"], 2)
                self.assertIn("ctx=8k/lc_code", c)
                self.assertEqual(c["ctx=8k/lc_kv"]["a_only"], 2)
                rg = os.path.join(d, "b2.jsonl")
                self.assertEqual(A.main(["regrade", rb, "--tasks", tp, "--out", rg]), 0)
                rows_g = [json.loads(l) for l in open(rg, encoding="utf-8")][1:]
                self.assertEqual([r["passed"] for r in rows_g], [r["passed"] for r in rows_b])
            finally:
                srv.shutdown()


@unittest.skipUnless(os.path.exists(SUITE), "local long suite missing")
class RealSuite(unittest.TestCase):
    def test_suite_shape_and_sample(self):
        meta = json.load(open(SUITE.replace(".jsonl", ".meta.json"), encoding="utf-8"))
        self.assertEqual(meta["by_kind"], {"lc": 96, "kv": 144})
        self.assertEqual(meta["lengths"], ["32k", "64k", "96k", "128k"])
        seen = set()
        with open(SUITE, encoding="utf-8") as f:
            for i, ln in enumerate(f):
                t = json.loads(ln)
                seen.add(t["cluster"])
                self.assertLessEqual(len(t["prompt"]) / 5.0, t["ctx_tokens"])     # loose guard; exact check: tokcheck_hf.py
                if i % 23 == 0 and t["kind"] == "lc":
                    self.assertTrue(S.grade(t, t["reference"], timeout=10)["passed"], t["id"])
                    self.assertFalse(S.grade(t, t["decoy"], timeout=10)["passed"], t["id"])
        self.assertEqual(len(seen), 24)


if __name__ == "__main__":
    unittest.main()
