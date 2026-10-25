# exact token count with the model's own HF tokenizer (CPU) vs bpe.py's estimate; feat027-longacc check
import json, sys, time
sys.path.insert(0, r"C:\ClaudeWorkspace\whirl-work\wt_027_longacc\tools\taskacc")
from tokenizers import Tokenizer
import bpe
hf = Tokenizer.from_file(sys.argv[2])
B = bpe.BPE.from_gguf(sys.argv[3])
tpl = None
try:
    from transformers import AutoTokenizer
    tpl = AutoTokenizer.from_pretrained(sys.argv[4])
except Exception as e:
    print("no chat template:", type(e).__name__, e)
seen, worst = set(), 0
for l in open(sys.argv[1], encoding="utf-8"):
    t = json.loads(l)
    key = (t["cluster"], t["kind"])
    if key in seen:
        continue
    seen.add(key)
    a = len(hf.encode(t["prompt"], add_special_tokens=False).ids)
    b = B.count(t["prompt"])
    full = None
    if tpl is not None:
        ids = tpl.apply_chat_template([{"role": "user", "content": t["prompt"]}], add_generation_prompt=True,
                                      tokenize=True, enable_thinking=False)
        full = len(ids["input_ids"] if isinstance(ids, dict) else ids)
    lim = t["ctx_tokens"] - t["max_tokens"]
    worst = max(worst, (full or a) - lim)
    print(f"{t['id']:22s} hf {a:6d} bpe.py {b:6d} diff {b - a:+5d}  chat-template prompt {full}  room {lim - (full or a):6d}")
print("worst (prompt - (ctx - max_tokens)):", worst)
