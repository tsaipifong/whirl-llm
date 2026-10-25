"""CPU selftests for tools/taskacc (feat027-taskacc). Run: python -I tools\\taskacc\\taskacc.py selftest

Needs the local task data (scratch\\feat027\\data, downloads\\feat027_*); the data-dependent tests are
skipped with a message when it is missing (the data is never in the repository).
"""
import gzip
import http.server
import json
import math
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as X   # noqa: E402
import sandbox as S   # noqa: E402
import stats as T     # noqa: E402
import tasks as K     # noqa: E402
import taskacc as A   # noqa: E402

HAVE_DATA = all(os.path.exists(p) for p in K.DEFAULT_SRC.values())
TASK_HE = {"id": "t/he", "kind": "he", "entry_point": "add", "lang": "en", "source": "selftest",
           "prompt_code": "from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    \"\"\"sum\"\"\"\n",
           "test": "def check(candidate):\n    assert candidate([1, 2]) == 3\n    assert candidate([]) == 0\n",
           "test_imports": []}
TASK_MB = {"id": "t/mb", "kind": "mbpp", "entry_point": "sq", "lang": "zh", "source": "selftest", "prompt_code": "",
           "test": "assert sq(3) == 9\nassert math.isclose(sq(0.5), 0.25)", "test_imports": ["import math"]}


class Sandbox(unittest.TestCase):
    def g(self, task, code, **kw):
        kw.setdefault("timeout", 8)
        return S.grade(task, code, **kw)

    def test_pass_and_fail(self):
        self.assertEqual(self.g(TASK_HE, "def add(xs: List[int]) -> int:\n    return sum(xs)\n")["status"], "pass")
        self.assertEqual(self.g(TASK_HE, "def add(xs):\n    return 1\n")["status"], "fail")
        self.assertEqual(self.g(TASK_MB, "def sq(x):\n    return x * x\n")["status"], "pass")
        self.assertEqual(self.g(TASK_MB, "def sq(x):\n    return x + x\n")["status"], "fail")

    def test_body_only_completion(self):
        # a reply that is only the body is put under the prompt's signature
        self.assertTrue(self.g(TASK_HE, "    return sum(xs)\n")["passed"])
        self.assertTrue(self.g(TASK_HE, "return sum(xs)\n")["passed"])

    def test_prompt_imports_added(self):
        # the model forgot `from typing import List`; the prompt's import lines are prepended
        self.assertTrue(self.g(TASK_HE, "def add(xs: List[int]) -> int:\n    return sum(xs)\n")["passed"])

    def test_timeout(self):
        r = self.g(TASK_HE, "def add(xs):\n    while True:\n        pass\n", timeout=3)
        self.assertEqual(r["status"], "timeout")
        self.assertFalse(r["passed"])

    def test_no_network(self):
        code = ("import socket\ndef add(xs):\n    s = socket.create_connection(('example.com', 80), timeout=2)\n    return sum(xs)\n")
        r = self.g(TASK_HE, code)
        self.assertFalse(r["passed"])
        self.assertIn("network is disabled", r["stderr"])
        code2 = "import socket\ndef add(xs):\n    socket.socket()\n    return sum(xs)\n"
        self.assertIn("network is disabled", self.g(TASK_HE, code2)["stderr"])
        code3 = "import urllib.request\ndef add(xs):\n    urllib.request.urlopen('http://127.0.0.1:9/', timeout=2)\n    return sum(xs)\n"
        r3 = self.g(TASK_HE, code3)
        self.assertFalse(r3["passed"])

    def test_no_subprocess(self):
        r = self.g(TASK_HE, "import subprocess\ndef add(xs):\n    subprocess.run(['cmd', '/c', 'echo hi'])\n    return sum(xs)\n")
        self.assertFalse(r["passed"])
        self.assertIn("process creation is disabled", r["stderr"])
        r = self.g(TASK_HE, "import os\ndef add(xs):\n    os.system('echo hi')\n    return sum(xs)\n")
        self.assertFalse(r["passed"])

    @unittest.skipUnless(os.name == "nt", "job object is Windows only")
    def test_job_blocks_children_even_without_the_python_guard(self):
        # _winapi.CreateProcess bypasses the Python-level guard; the job's active-process limit stops it
        code = ("import _winapi, sys\ndef add(xs):\n    _winapi.CreateProcess(None, 'cmd /c exit 0', None, None, 0, 0, None, None, None)\n"
                "    return sum(xs)\n")
        r = self.g(TASK_HE, code)
        self.assertTrue(r["jobbed"])
        self.assertFalse(r["passed"])

    @unittest.skipUnless(os.name == "nt", "job object is Windows only")
    def test_memory_limit(self):
        r = self.g(TASK_HE, "def add(xs):\n    b = bytearray(3 * 1024 ** 3)\n    return sum(xs)\n", mem_mb=512)
        self.assertFalse(r["passed"])
        self.assertIn(r["status"], ("memory", "error"))

    def test_no_writes_outside(self):
        target = os.path.join(tempfile.gettempdir(), "taskacc_should_not_exist.txt")
        code = f"def add(xs):\n    open({target!r}, 'w').write('x')\n    return sum(xs)\n"
        r = self.g(TASK_HE, code)
        self.assertFalse(r["passed"])
        self.assertFalse(os.path.exists(target))
        # writing inside the sandbox directory is fine
        self.assertTrue(self.g(TASK_HE, "def add(xs):\n    open('scratch.txt', 'w').write('x')\n    return sum(xs)\n")["passed"])

    def test_no_delete(self):
        keep = tempfile.NamedTemporaryFile(delete=False, dir=S.DEFAULT_ROOT if os.path.isdir(S.DEFAULT_ROOT) else None)
        keep.close()
        try:
            r = self.g(TASK_HE, f"import os\ndef add(xs):\n    os.remove({keep.name!r})\n    return sum(xs)\n")
            self.assertFalse(r["passed"])
            self.assertTrue(os.path.exists(keep.name))
        finally:
            os.unlink(keep.name)

    def test_early_exit_is_not_a_pass(self):
        for code in ("import sys\ndef add(xs):\n    return sum(xs)\nsys.exit(0)\n",
                     "import os\ndef add(xs):\n    return sum(xs)\nos._exit(0)\n",
                     "def add(xs):\n    raise SystemExit(0)\n"):
            self.assertFalse(self.g(TASK_HE, code)["passed"], code)

    def test_fake_nonce_is_not_a_pass(self):
        code = "import sys\nprint('TASKACC_OK_' + '0' * 32)\ndef add(xs):\n    return 5\n"
        self.assertFalse(self.g(TASK_HE, code)["passed"])
        code2 = "import sys\ndef add(xs):\n    print(sys.stdin.readline())\n    return sum(xs)\n"
        r = self.g(TASK_HE, code2)  # stdin is already consumed: the model code cannot read the nonce
        self.assertNotIn("TASKACC_OK_", r["stdout"].replace("<nonce>", ""))

    def test_main_block_not_run(self):
        code = "def add(xs):\n    return sum(xs)\nif __name__ == '__main__':\n    assert add([1]) == 99\n"
        self.assertTrue(self.g(TASK_HE, code)["passed"])

    def test_no_code(self):
        self.assertEqual(S.grade(TASK_HE, None)["status"], "no_code")

    def test_cleanup(self):
        root = tempfile.mkdtemp(prefix="taskacc_root_", dir=S.DEFAULT_ROOT if os.path.isdir(S.DEFAULT_ROOT) else None)
        try:
            S.grade(TASK_HE, "def add(xs):\n    return sum(xs)\n", root=root)
            self.assertEqual(os.listdir(root), [])
        finally:
            os.rmdir(root)


class Extract(unittest.TestCase):
    def test_prefers_block_with_entry_point(self):
        t = "Here:\n```python\ndef helper():\n    pass\n```\nand\n```python\ndef add(xs):\n    return sum(xs)\n```\nexample:\n```python\nprint(add([1]))\n```"
        self.assertEqual(X.extract(t, "add"), "def add(xs):\n    return sum(xs)")

    def test_think_removed(self):
        t = "<think>\n```python\ndef add(xs):\n    return 0\n```\n</think>\n```python\ndef add(xs):\n    return sum(xs)\n```"
        self.assertIn("sum", X.extract(t, "add"))
        self.assertIn("sum", X.extract("reasoning...</think>```python\ndef add(xs):\n    return sum(xs)\n```", "add"))

    def test_unfenced_and_unterminated(self):
        self.assertTrue(X.extract("Sure.\nimport math\ndef add(xs):\n    return sum(xs)\n", "add").startswith("import math"))
        self.assertIn("def add", X.extract("```python\ndef add(xs):\n    return sum(xs)\n", "add"))
        self.assertIsNone(X.extract("I cannot do that.", "add"))

    def test_py_tag_and_plain_fence(self):
        self.assertIn("x * x", X.extract("```\ndef sq(x):\n    return x * x\n```", "sq"))
        self.assertIn("x * x", X.extract("```py\ndef sq(x):\n    return x * x\n```", "sq"))


class Stats(unittest.TestCase):
    def test_mcnemar_known_values(self):
        self.assertEqual(T.mcnemar(0, 0), (1.0, 1.0))
        p2, _ = T.mcnemar(3, 2)          # itemD q8v: 3 vs 2 -> p = 1.0
        self.assertAlmostEqual(p2, 1.0)
        p2, _ = T.mcnemar(5, 8)          # itemD fp8: 5 vs 8 -> p = 0.58
        self.assertAlmostEqual(p2, 0.5811, places=3)
        p2, p1 = T.mcnemar(10, 0)        # 2 * 0.5^10
        self.assertAlmostEqual(p2, 2 / 1024)
        self.assertAlmostEqual(p1, 1 / 1024)
        _, p1 = T.mcnemar(0, 10)
        self.assertAlmostEqual(p1, 1.0)
        # one-sided equals scipy.stats.binomtest(b, b + c, 0.5, alternative='greater'): P(X >= 7 | n=9) = 46/512
        self.assertAlmostEqual(T.mcnemar(7, 2)[1], 46 / 512)

    def _pairs(self, a, b, n_tasks, seeds=1):
        res_a, res_b = [], []
        i = 0
        for t in range(n_tasks):
            for s in range(seeds):
                res_a.append({"task_id": f"x{t}", "seed": s, "passed": a(i), "lang": "zh", "source": "s"})
                res_b.append({"task_id": f"x{t}", "seed": s, "passed": b(i), "lang": "zh", "source": "s"})
                i += 1
        return T.pair(res_a, res_b)[0]

    def test_identical_configs(self):
        pr = self._pairs(lambda i: i % 5 != 0, lambda i: i % 5 != 0, 200)
        s = T.summarize(pr, n_boot=500)
        self.assertEqual((s["a_only"], s["b_only"]), (0, 0))
        self.assertAlmostEqual(s["recovery"], 1.0)
        self.assertTrue(s["fast_verdict"].startswith("PASS"))
        self.assertTrue(s["balance_note"].startswith("no significant loss"))

    def test_big_drop_fails(self):
        # A 80%, B 60%: recovery 0.75 -> FAIL, significant
        pr = self._pairs(lambda i: i % 5 != 0, lambda i: i % 5 not in (0, 1), 200)
        s = T.summarize(pr, n_boot=2000)
        self.assertAlmostEqual(s["recovery"], 0.75)
        self.assertTrue(s["fast_verdict"].startswith("FAIL"), s["fast_verdict"])
        self.assertTrue(s["balance_note"].startswith("SIGNIFICANT"))
        self.assertLess(s["mcnemar_p_b_worse"], 1e-6)

    def test_small_n_inconclusive(self):
        pr = self._pairs(lambda i: i % 5 != 0, lambda i: i % 5 != 0 and i != 1, 20)
        s = T.summarize(pr, n_boot=2000)
        self.assertTrue(s["fast_verdict"].startswith("INCONCLUSIVE"), s["fast_verdict"])

    def test_bootstrap_deterministic_and_clustered(self):
        pr = self._pairs(lambda i: i % 3 != 0, lambda i: i % 4 != 0, 50, seeds=4)
        b1, b2 = T.bootstrap(pr, 300), T.bootstrap(pr, 300)
        self.assertEqual(b1, b2)
        self.assertEqual(b1["clusters"], 50)

    def test_pair_reports_unpaired(self):
        _, oa, ob = T.pair([{"task_id": "a", "passed": 1}, {"task_id": "b", "passed": 1}], [{"task_id": "a", "passed": 0}])
        self.assertEqual((oa, ob), ([("b", 0)], []))

    def test_plan(self):
        d, b, drop = T.min_detectable_drop(200, 0.85)
        self.assertGreater(d, 0)
        self.assertTrue(b is None or b <= d)


@unittest.skipUnless(HAVE_DATA, "local task data missing (scratch\\feat027\\data, downloads\\feat027_*)")
class TaskSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core, cls.meta = K.build("core200")

    def test_counts_and_ids(self):
        self.assertEqual(len(self.core), 200)
        self.assertEqual(self.meta["by_lang"], {"zh": 140, "en": 60})
        self.assertEqual(len({t["id"] for t in self.core}), 200)
        for t in self.core:
            self.assertTrue(t["entry_point"], t["id"])
            self.assertIn(t["kind"], ("he", "mbpp"))
            self.assertTrue(t["license"].startswith(("MIT", "CC-BY-4.0")))

    def test_deterministic(self):
        again, meta2 = K.build("core200")
        self.assertEqual(meta2["taskset_sha256"], self.meta["taskset_sha256"])

    def test_xl_tests_are_the_humaneval_tests(self):
        he = {}
        with gzip.open(K.DEFAULT_SRC["humaneval"], "rt", encoding="utf-8") as f:
            for ln in f:
                h = json.loads(ln)
                he[h["task_id"]] = h
        norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines())  # noqa: E731
        xl = [t for t in self.core if t["source"] == "xl_zh"]
        self.assertEqual(len(xl), 80)
        for t in xl:
            self.assertEqual(norm(t["test"]), norm(he[t["upstream_id"]]["test"]), t["id"])
        # no English HumanEval task duplicates a Chinese XL task
        en = {t["upstream_id"] for t in self.core if t["source"] == "humaneval"}
        self.assertFalse(en & {t["upstream_id"] for t in xl})

    def test_zh_prompts_are_chinese(self):
        for t in self.core:
            if t["lang"] == "zh":
                self.assertTrue(any("一" <= ch <= "鿿" for ch in t["prompt"]), t["id"])

    def test_reference_and_stub_on_a_sample(self):
        for t in self.core[::17]:
            self.assertTrue(S.grade(t, t["canonical"], timeout=10)["passed"], t["id"])
            self.assertFalse(S.grade(t, A.STUBS[t["kind"]].format(ep=t["entry_point"]), timeout=10)["passed"], t["id"])


# ------------------------------------------------------------------ end to end with a mock OpenAI server
class _Mock(http.server.BaseHTTPRequestHandler):
    answers = {}
    eps = {}
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
            return self._send({"numerics": {"mode": _Mock.mode, "label": "mock " + _Mock.mode}})
        self._send({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        j = json.loads(self.rfile.read(n))
        _Mock.seen.append(j)
        prompt = j["messages"][0]["content"]
        tid, code = _Mock.answers[prompt]
        if _Mock.mode == "bad" and hash(tid) % 4 == 0:
            code = A.STUBS["he"].format(ep=_Mock.eps[tid])
        content = "好的，程式如下：\n```python\n" + code + "\n```\n"
        self._send({"choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": ""},
                                 "finish_reason": "stop"}], "usage": {"completion_tokens": 42}})


@unittest.skipUnless(HAVE_DATA, "local task data missing")
class EndToEnd(unittest.TestCase):
    def test_run_and_compare(self):
        tasks, meta = K.build("core200")
        tasks = tasks[::10]   # 20 tasks: zh XL, zh MBPP, en HE
        with tempfile.TemporaryDirectory(dir=S.DEFAULT_ROOT if os.path.isdir(S.DEFAULT_ROOT) else None) as d:
            tp = os.path.join(d, "tasks.jsonl")
            with open(tp, "w", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            _Mock.answers = {t["prompt"]: (t["id"], t["canonical"]) for t in tasks}
            _Mock.eps = {t["id"]: t["entry_point"] for t in tasks}
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                _Mock.mode = "good"
                ra = os.path.join(d, "a.jsonl")
                self.assertEqual(A.main(["run", "--tasks", tp, "--url", url, "--out", ra, "--label", "good", "--concurrency", "3"]), 0)
                self.assertFalse(_Mock.seen[-1]["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(_Mock.seen[-1]["temperature"], 0.0)
                # resume: nothing left to do, same file
                self.assertEqual(A.main(["run", "--tasks", tp, "--url", url, "--out", ra, "--label", "good"]), 0)
                _Mock.mode = "bad"
                rb = os.path.join(d, "b.jsonl")
                self.assertEqual(A.main(["run", "--tasks", tp, "--url", url, "--out", rb, "--label", "bad"]), 0)
                rows_a = [json.loads(l) for l in open(ra, encoding="utf-8")][1:]
                self.assertEqual(len(rows_a), 20)
                self.assertTrue(all(r["passed"] for r in rows_a), [r["task_id"] for r in rows_a if not r["passed"]])
                rows_b = [json.loads(l) for l in open(rb, encoding="utf-8")][1:]
                n_bad = sum(1 for t in tasks if hash(t["id"]) % 4 == 0)
                self.assertEqual(sum(not r["passed"] for r in rows_b), n_bad)
                js = os.path.join(d, "cmp.json")
                self.assertEqual(A.main(["compare", ra, rb, "--json", js, "--boot", "500"]), 0)
                c = json.load(open(js, encoding="utf-8"))["subsets"]["all"]
                self.assertEqual(c["a_only"], n_bad)
                self.assertEqual(c["b_only"], 0)
                self.assertEqual(c["n"], 20)
                self.assertEqual(A.main(["report", rb]), 0)
                rg = os.path.join(d, "b2.jsonl")
                self.assertEqual(A.main(["regrade", rb, "--tasks", tp, "--out", rg]), 0)
                rows_g = [json.loads(l) for l in open(rg, encoding="utf-8")][1:]
                self.assertEqual([r["passed"] for r in rows_g], [r["passed"] for r in rows_b])
            finally:
                srv.shutdown()


if __name__ == "__main__":
    unittest.main()
