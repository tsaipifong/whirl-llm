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
