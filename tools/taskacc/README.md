# tools/taskacc — task-accuracy eval

Execution-graded coding tasks sent to `whirl-server`'s OpenAI endpoint, one server per numerics
configuration, then a paired comparison of two configurations (normally precise vs balance or fast):
pass@1, recovery = acc(B) / acc(A), exact McNemar on the discordant pairs, cluster-bootstrap CI.

This implements the acceptance yardstick of the modes: balance is accepted on task accuracy (KL and
top-1 are only regression guards), fast on task-accuracy recovery >= 95% of precise.

Only code lives here. The task data (HumanEval MIT, HumanEval-XL MIT, MBPP CC-BY-4.0 and our Chinese
translation of 60 MBPP descriptions) is read from local files (paths in `tasks.py: DEFAULT_SRC`) and is
not part of the repository.

```
python -I tools\taskacc\taskacc.py build --set core200 --out <data dir>      # 200 tasks: zh 140, en 60
python -I tools\taskacc\taskacc.py validate --tasks <data dir>\tasks_core200.jsonl
python -I tools\taskacc\taskacc.py selftest
python -I tools\taskacc\taskacc.py run --tasks T.jsonl --url http://127.0.0.1:18427 --out precise.jsonl --label precise
python -I tools\taskacc\taskacc.py compare precise.jsonl balance.jsonl --json cmp.json
python -I tools\taskacc\taskacc.py plan --n 200 --acc 0.85
```

Defaults: greedy (temperature 0), thinking off, max_tokens 2048, 4 concurrent requests. With
`--think on` raise `--max-tokens` (8192 or more), or replies are cut inside the reasoning.

Model code runs in a child process (`python -I -S`, temp directory, timeout, Windows Job Object with
memory and process limits, no network / subprocess / destructive file calls at the Python level); see
the docstring of `sandbox.py` for what this does and does not stop.

## Long-context suite (feat027-longacc): 32k / 64k / 96k / 128k

KV-cache formats (int8 KV in balance, 4-bit KV in fast) lose most at long context, and the core200
tasks are short. `longctx.py` builds "repository snapshot" contexts: real Python files of llama.cpp
(MIT, read from a local checkout) as filler, plus synthetic needles:

- **coding needles** (`kind: lc`, execution-graded): a module with helpers, a `CURRENT_VERSION`
  constant and a stub function whose docstring points to a section of a `docs/*.md` file placed at
  another depth (two hops). The document also holds a deprecated section with other parameters (the
  decoy). The model implements the stub; the grader makes the module importable under its package
  path and runs asserts computed from the reference. `validate` proves reference pass / stub fail /
  decoy-section implementation fail for every task.
- **key/value needles** (`kind: kv`, exact match): `CFG_<NAME> = "<value>"` with a near-duplicate
  `CFG_<NAME>_OLD` a few lines later; pass iff the reply holds the value and no other value.

All questions follow the shared context, so a prefix cache can reuse it (run with one slot, in file
order). Context sizes are counted with the model's tokenizer read from its GGUF (`bpe.py`, the qwen35
pre-tokenizer; cross-check with `tokcheck_hf.py` and the model's `tokenizer.json`); the run records
`usage.prompt_tokens` and `cached_tokens`. The bootstrap resamples contexts (field `cluster`).

```
python -I tools\taskacc\taskacc.py build --set long --gguf <model.gguf> --out <data dir>   # 24 contexts, 240 tasks
python -I tools\taskacc\taskacc.py validate --tasks <data dir>\tasks_long.jsonl
python -I tools\taskacc\taskacc.py run --tasks <data dir>\tasks_long.jsonl --url http://127.0.0.1:18428 --out p.jsonl --label precise --concurrency 1 [--ctx 32k,64k]
python -I tools\taskacc\taskacc.py compare p.jsonl b.jsonl --boot-subsets      # all, ctx=<bucket>, ctx=<bucket>/lc_code|lc_kv
```

The server needs one slot of at least 131072 tokens (+ margin): `-np 1 --ctx-per-slot 133120`.
