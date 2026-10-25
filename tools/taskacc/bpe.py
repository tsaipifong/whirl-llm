"""Token counter for sizing the long-context suite (feat027-longacc). Pure Python, CPU only.

Reads the byte-level BPE vocabulary and merges from a model GGUF (gguf package, header only, memory
mapped) and counts tokens the way llama.cpp / WHIRL do for tokenizer.ggml.pre = "qwen35":
  pre-tokenizer (llama.cpp src/llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35, WHIRL src/tokenizer/unicode.cpp)
    (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}
    | ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
  then GPT-2 byte-to-unicode mapping and rank-ordered merges.
Python's `re` has no \\p{..}; letters+marks are approximated by [^\\W\\d_] and \\p{N} by \\d. That is exact
for ASCII and for CJK text (letters) and approximate only for rare scripts / non-decimal numerals, so
the count is labelled an estimate [E]; the GPU runner records the server's usage.prompt_tokens per row,
which is the measured value [M]. Special tokens (chat template) are not counted: callers add a margin.
"""
import re

_L = r"[^\W\d_]"        # letters (+ most marks)
_N = r"\d"
_LN = r"[^\W_]"
PAT = re.compile(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|(?:(?![\r\n])(?!" + _LN + r").)?" + _L + r"+"
    r"|" + _N +
    r"| ?(?:(?!\s)(?!" + _LN + r").)+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+", re.S)


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = _bytes_to_unicode()


class BPE:
    def __init__(self, vocab, merges):
        self.vocab = vocab                      # token string -> id
        self.ranks = {}
        for i, m in enumerate(merges):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = i
        self.cache = {}

    @classmethod
    def from_gguf(cls, path):
        import gguf  # noqa: PLC0415  (pyref venv has gguf 0.19)
        r = gguf.GGUFReader(path)
        f = r.fields["tokenizer.ggml.pre"]
        pre = bytes(f.parts[f.data[0]]).decode()
        if pre != "qwen35":
            raise ValueError(f"pre-tokenizer {pre!r}: only qwen35 is implemented")
        t = r.fields["tokenizer.ggml.tokens"]
        m = r.fields["tokenizer.ggml.merges"]
        toks = [bytes(t.parts[i]).decode("utf-8") for i in t.data]
        merges = [bytes(m.parts[i]).decode("utf-8") for i in m.data]
        return cls({s: i for i, s in enumerate(toks)}, merges)

    def _word(self, w):
        hit = self.cache.get(w)
        if hit is not None:
            return hit
        parts = [B2U[b] for b in w.encode("utf-8")]
        ranks = self.ranks
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            a, b = parts[bi], parts[bi + 1]
            out, i = [], 0
            while i < len(parts):   # merge every occurrence of the best pair (GPT-2 semantics)
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        if len(self.cache) < 500000:
            self.cache[w] = parts
        return parts

    def tokenize(self, text):
        out = []
        for w in PAT.findall(text):
            out.extend(self._word(w))
        return out

    def count(self, text):
        return sum(len(self._word(w)) for w in PAT.findall(text))


def pretokenize(text):
    return PAT.findall(text)
