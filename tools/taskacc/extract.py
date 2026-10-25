"""Pull the Python code out of a chat reply (feat027-taskacc).

Rules (EvalPlus-style sanitising, kept simple and deterministic):
  1. Drop any <think>...</think> part (servers normally return reasoning separately; this is a fallback).
  2. Fenced blocks (```python / ```py / ``` ): prefer the LAST block that defines the entry point;
     else the longest python-tagged block; else the longest block.
  3. No fence: if the text contains `def <entry_point>(`, take from the first top-level import/def
     to the end; else return None (no code -> graded "no_code").
"""
import re

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in text:          # an opening tag that the template already emitted
        text = text.split("</think>", 1)[1]
    return text


def extract(text, entry_point=None):
    t = strip_think(text)
    blocks = [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(t)]
    if not blocks and "```" in t:    # unterminated last fence (reply cut at max_tokens)
        m = re.search(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*)$", t, re.S)
        if m:
            blocks = [(m.group(1).lower(), m.group(2))]
    if blocks:
        if entry_point:
            pat = re.compile(r"^\s*def\s+" + re.escape(entry_point) + r"\s*\(", re.M)
            hit = [b for lang, b in blocks if pat.search(b)]
            if hit:
                return hit[-1].strip("\n")
        py = [b for lang, b in blocks if lang in ("python", "py", "python3")]
        pool = py or [b for _, b in blocks]
        return max(pool, key=len).strip("\n")
    if entry_point and re.search(r"def\s+" + re.escape(entry_point) + r"\s*\(", t):
        m = re.search(r"^(?:import |from |def |class |@)", t, re.M)
        return t[m.start():].strip("\n") if m else None
    return None
