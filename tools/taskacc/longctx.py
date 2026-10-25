"""Long-context suite for the WHIRL task-accuracy eval (feat027-longacc): 32k / 64k / 96k / 128k tokens.

Why: KV-cache formats (Ornith balance int8 KV q8h, fast 4-bit KV) lose most at long context, and the
core200 coding tasks are short. Method notes and prior art: progress\\feat027-longacc.PRIOR_ART.md.

Each CONTEXT is one "repository snapshot" of about L tokens:
  * filler = real Python files of llama.cpp (refsrc, MIT License, Copyright the ggml authors), shuffled per
    context, each file cut at MAX_FILE_TOKENS on a line boundary;
  * Q synthetic CODING needles (kind "lc", execution-graded): a module `<pkg>/<area>/<mod>.py` with
    helpers, a CURRENT_VERSION constant and a stub function whose docstring points to a section of
    `docs/<topic>.md`; the doc file sits at ANOTHER depth (two hops) and also holds a deprecated
    section (the decoy: same structure, other parameters). The task: implement the stub. Grading runs
    the module source (importable as its package path) + the model's function in the taskacc sandbox
    against asserts computed from our reference implementation. Validation proves: reference passes,
    the stub fails, and an implementation of the DECOY section fails.
  * K KEY/VALUE needles (kind "kv", exact match; itemD/qa_kv.py design): `CFG_<NAME> = "<value>"` lines
    with a `CFG_<NAME>_OLD` near-duplicate decoy a few lines later; one question per key. Pass iff the
    reply contains the value and not the decoy value.
All questions come AFTER the shared context (same prefix for every task of a context, so a prefix
cache can reuse it). Lengths are counted with the model's tokenizer (bpe.py) [E]; the runner records
usage.prompt_tokens [M].

Task fields (on top of the taskacc ones): ctx ("32k".."128k"), ctx_tokens (target), cluster (context id;
the bootstrap resamples contexts), depth (needle position 0..1), doc_depth, max_tokens, kind "lc"/"kv",
module_name / module_src (lc), reference / decoy (lc, for validation), answer / decoy_answer (kv).
"""
import hashlib
import json
import os
import random
import re

LENGTHS = {"32k": 32768, "64k": 65536, "96k": 98304, "128k": 131072}
OUTPUT_RESERVE = 2048          # prompt <= L - OUTPUT_RESERVE, so prompt + reply fits in a slot of L tokens
TEMPLATE_RESERVE = 64          # chat template tokens (not counted by bpe.py)
MAX_FILE_TOKENS = 6000         # filler files are cut to this size (finer depth placement, more variety)
FILLER_ROOT = r"C:\ClaudeWorkspace\refsrc\llama.cpp"
FILLER_LICENSE = "MIT (llama.cpp, Copyright (c) 2023-2026 The ggml authors; refsrc\\llama.cpp\\LICENSE)"
MAX_TOKENS_LC = 1536
MAX_TOKENS_KV = 64
SUITE_VERSION = "longacc 1 (2026-10-26)"

PKGS = ["tidewater", "lumenkit", "quarrylane", "harborline", "copperleaf", "nimbusflow", "basaltdb", "kestrelio",
        "foxglove", "marrowstone", "ironbark", "sablecore", "wrenmesh", "tamarack", "obsidianq", "pebblefs"]
WORDS = ["amber", "cobalt", "violet", "copper", "falcon", "heron", "otter", "maple", "cedar", "quartz", "delta",
         "sigma", "nova", "orbit", "pixel", "raven", "willow", "ember", "lotus", "cinder", "glacier", "saffron"]


# ====================================================================== coding templates
# Each template: gen(rng) -> dict(entry, area, mod, topic, title, cur (params), old (params)),
#   src(P) -> reference implementation source (uses only the stdlib and the module's helpers),
#   doc(P, ver, deprecated) -> markdown section, module(entry-stub, helpers) text, inputs(rng) -> arg tuples.
def _fmt_hex2(b):
    return " ".join(f"0x{x:02X}" for x in b)


class Frame:
    name, entry, area, mod, topic = "frame", "encode_frame", "wire", "frames", "wire-format"
    sig = "def encode_frame(flags: int, payload: bytes) -> bytes:"
    helpers = '''import struct


def _pack_uint(value: int, width: int, byteorder: str) -> bytes:
    """Pack an unsigned integer of `width` bytes; byteorder is "little" or "big"."""
    return int(value).to_bytes(width, byteorder)


def _crc_unused(data: bytes) -> int:
    # kept for the v1 tooling; not part of any current frame layout
    return sum(data) & 0xFF
'''

    @staticmethod
    def params(rng):
        return {"magic": bytes(rng.sample(range(0x21, 0xFF), 2)), "order": rng.sample(["version", "flags", "length"], 3),
                "lenw": rng.choice([2, 4]), "endian": rng.choice(["little", "big"]),
                "mul": rng.choice([3, 5, 7, 11, 13, 17, 31]), "add": rng.randint(1, 60000)}

    @staticmethod
    def src(P, ver):
        w = {"version": 1, "flags": 2, "length": P["lenw"]}
        val = {"version": str(ver), "flags": "flags", "length": "len(payload)"}
        parts = " + ".join(f"_pack_uint({val[f]}, {w[f]}, {P['endian']!r})" for f in P["order"])
        return (f"def encode_frame(flags: int, payload: bytes) -> bytes:\n"
                f"    head = {P['magic']!r} + {parts}\n"
                f"    ck = (sum(payload) * {P['mul']} + {P['add']}) % 65536\n"
                f"    return head + bytes(payload) + _pack_uint(ck, 2, {P['endian']!r})\n")

    @staticmethod
    def doc(P, ver, dep):
        wn = {"version": "version (u8, always {v})".format(v=ver), "flags": "flags (u16)",
              "length": f"payload length in bytes (u{8 * P['lenw']})"}
        lines = [f"## Frame layout v{ver}" + (" (deprecated, do not produce)" if dep else ""), "",
                 f"1. Magic: two bytes {_fmt_hex2(P['magic'])}.",
                 "2. Header fields, in this order: " + ", then ".join(wn[f] for f in P["order"]) + ".",
                 f"   All integers in this layout are {P['endian']}-endian.",
                 "3. The payload bytes, unchanged.",
                 f"4. Checksum (u16): (sum of all payload bytes * {P['mul']} + {P['add']}) mod 65536.", ""]
        return "\n".join(lines)

    @staticmethod
    def inputs(rng):
        out = [(0, b""), (1, b"A"), (0xFFFF, bytes(range(16)))]
        for _ in range(5):
            out.append((rng.randint(0, 0xFFFF), bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 40)))))
        return out


class Fee:
    name, entry, area, mod, topic = "fee", "compute_fee", "billing", "fees", "billing"
    sig = "def compute_fee(amount_cents: int) -> int:"
    helpers = '''from decimal import Decimal


def _cents(amount) -> int:
    """Convert a Decimal / str amount in dollars to integer cents (helper for callers)."""
    return int((Decimal(str(amount)) * 100).to_integral_value())
'''

    @staticmethod
    def params(rng):
        b = sorted(rng.sample(range(5, 400), 3))
        bounds = [x * 100 for x in b]
        rates = sorted(rng.sample(range(20, 600), 4), reverse=rng.random() < 0.5)
        return {"bounds": bounds, "rates": rates, "mode": rng.choice(["marginal", "flat"]),
                "minfee": rng.choice([0, 25, 30, 50, 99, 100]), "round": rng.choice([1, 5, 10, 25]),
                "cap": rng.choice([2500, 4999, 7500, 10000, 15000])}

    @staticmethod
    def src(P, ver):
        return (f"def compute_fee(amount_cents: int) -> int:\n"
                f"    bounds = {P['bounds']!r}\n"
                f"    rates = {P['rates']!r}  # basis points\n"
                f"    if {P['mode']!r} == 'marginal':\n"
                f"        bp, lo = 0, 0\n"
                f"        for i, r in enumerate(rates):\n"
                f"            hi = bounds[i] if i < len(bounds) else None\n"
                f"            top = amount_cents if hi is None else min(amount_cents, hi)\n"
                f"            if top > lo:\n"
                f"                bp += (top - lo) * r\n"
                f"            if hi is None or amount_cents <= hi:\n"
                f"                break\n"
                f"            lo = hi\n"
                f"    else:\n"
                f"        i = 0\n"
                f"        while i < len(bounds) and amount_cents > bounds[i]:\n"
                f"            i += 1\n"
                f"        bp = amount_cents * rates[i]\n"
                f"    fee = -(-bp // (10000 * {P['round']})) * {P['round']}\n"
                f"    return max({P['minfee']}, min({P['cap']}, fee))\n")

    @staticmethod
    def doc(P, ver, dep):
        b = P["bounds"]
        rows = [f"| up to {b[0]} cents | {P['rates'][0]} bp |",
                f"| above {b[0]} up to {b[1]} cents | {P['rates'][1]} bp |",
                f"| above {b[1]} up to {b[2]} cents | {P['rates'][2]} bp |",
                f"| above {b[2]} cents | {P['rates'][3]} bp |"]
        how = ("Marginal: each part of the amount is charged at the rate of the band it falls in, and the parts are added."
               if P["mode"] == "marginal" else
               "Flat: the WHOLE amount is charged at the rate of the band that contains the amount.")
        return "\n".join([f"## Fee schedule v{ver}" + (" (deprecated)" if dep else ""), "",
                          "| band | rate (basis points, 1 bp = 0.01%) |", "|---|---|"] + rows + ["",
                          how,
                          f"Round the exact fee UP to the next multiple of {P['round']} cents.",
                          f"Then apply the limits: at least {P['minfee']} cents, at most {P['cap']} cents.", ""])

    @staticmethod
    def inputs(rng):
        out = [(0,), (1,), (99,)]
        for _ in range(7):
            out.append((rng.randint(100, 60000),))
        return out

    @staticmethod
    def edges(P):
        return [(x,) for x in P["bounds"]] + [(x + 1,) for x in P["bounds"]]


class Key:
    name, entry, area, mod, topic = "key", "make_key", "storage", "keys", "storage-keys"
    sig = "def make_key(name: str) -> str:"
    helpers = '''import zlib


def _short_hash(text: str) -> str:
    """Six lowercase hex digits of the CRC-32 of the UTF-8 text (low 24 bits)."""
    return format(zlib.crc32(text.encode("utf-8")) & 0xFFFFFF, "06x")
'''

    @staticmethod
    def params(rng):
        return {"ns": rng.choice(["kv", "obj", "blob", "idx", "meta", "tw", "lk"]) + str(rng.randint(1, 9)),
                "sep": rng.choice(["-", "_", "."]), "maxlen": rng.randint(20, 36), "case": rng.choice(["lower", "upper"])}

    @staticmethod
    def src(P, ver):
        return (f"def make_key(name: str) -> str:\n"
                f"    sep = {P['sep']!r}\n"
                f"    out = []\n"
                f"    for ch in name:\n"
                f"        if ch.isascii() and ch.isalnum():\n"
                f"            out.append(ch)\n"
                f"        elif not out or out[-1] != sep:\n"
                f"            out.append(sep)\n"
                f"    body = ''.join(out).strip(sep).{P['case']}()\n"
                f"    key = {P['ns']!r} + sep + body\n"
                f"    if len(key) > {P['maxlen']}:\n"
                f"        key = key[:{P['maxlen']} - 7] + '~' + _short_hash(name)\n"
                f"    return key\n")

    @staticmethod
    def doc(P, ver, dep):
        sepname = {"-": "a hyphen `-`", "_": "an underscore `_`", ".": "a dot `.`"}[P["sep"]]
        return "\n".join([f"## Key format v{ver}" + (" (deprecated)" if dep else ""), "",
                          f"- The separator is {sepname}.",
                          "- Every character that is not an ASCII letter or digit becomes the separator; runs of separators "
                          "collapse into one, and separators at the start and the end of the body are removed.",
                          f"- The body is converted to {P['case']} case.",
                          f"- The key is the namespace `{P['ns']}`, the separator, then the body.",
                          f"- If the key is longer than {P['maxlen']} characters, keep its first {P['maxlen']} - 7 characters and "
                          "append `~` and `_short_hash(name)` of the ORIGINAL name (6 hex digits), so the result is exactly "
                          f"{P['maxlen']} characters.", ""])

    @staticmethod
    def inputs(rng):
        out = [("Hello World",), ("  a--b__c  ",), ("Q3 report (final).pdf",)]
        for _ in range(5):
            n = rng.randint(4, 60)
            out.append(("".join(rng.choice("abcdefghijKLMNOP0123456789 -_./#") for _ in range(n)),))
        return out


class Backoff:
    name, entry, area, mod, topic = "backoff", "retry_delays", "net", "retry", "retry-policy"
    sig = "def retry_delays(n: int) -> list[int]:"
    helpers = '''import math


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))
'''

    @staticmethod
    def params(rng):
        num, den = rng.choice([(2, 1), (3, 2), (5, 2), (3, 1), (5, 4)])
        return {"base": rng.choice([50, 80, 100, 120, 200, 250]), "num": num, "den": den,
                "cap": rng.choice([3000, 5000, 8000, 12000, 20000]), "j": rng.randint(7, 97), "m": rng.choice([37, 50, 64, 101, 128])}

    @staticmethod
    def src(P, ver):
        return (f"def retry_delays(n: int) -> list[int]:\n"
                f"    out = []\n"
                f"    for i in range(n):\n"
                f"        d = {P['base']} * {P['num']} ** i // {P['den']} ** i\n"
                f"        out.append(min({P['cap']}, d) + (i * {P['j']}) % {P['m']})\n"
                f"    return out\n")

    @staticmethod
    def doc(P, ver, dep):
        return "\n".join([f"## Retry schedule v{ver}" + (" (deprecated)" if dep else ""), "",
                          "Attempt i (i = 0, 1, ..., n-1) waits, in milliseconds:", "",
                          f"    d_i = min({P['cap']}, floor({P['base']} * ({P['num']}/{P['den']})^i)) + ((i * {P['j']}) mod {P['m']})",
                          "", "Compute the floor exactly with integers (no floating point).", ""])

    @staticmethod
    def inputs(rng):
        return [(0,), (1,), (2,), (5,), (9,), (14,)]


class Classify:
    name, entry, area, mod, topic = "classify", "classify", "api", "errors", "error-codes"
    sig = "def classify(status: int, retryable: bool) -> str:"
    helpers = '''from enum import Enum


class Severity(Enum):
    INFO = 0
    WARN = 1
    ERROR = 2
'''
    CODES = ["E_TIMEOUT", "E_THROTTLED", "E_AUTH", "E_NOT_FOUND", "E_CONFLICT", "E_UPSTREAM", "E_BAD_INPUT", "E_UNAVAILABLE",
             "E_QUOTA", "E_INTERNAL"]

    @staticmethod
    def params(rng):
        codes = rng.sample(Classify.CODES, 6)
        r1 = rng.choice([(400, 499), (400, 449), (401, 403)])
        rows = [(r1[0], r1[1], codes[0]), (500, 599, codes[1]), (rng.choice([404, 408, 409]), rng.choice([410, 420, 430]), codes[2]),
                (300, 399, codes[3])]
        rng.shuffle(rows)
        exact = {rng.choice([429, 503, 418]): codes[4]}
        retry = sorted(rng.sample([r[2] for r in rows], 2))
        return {"rows": rows, "exact": exact, "retry": retry, "default": codes[5], "suffix": rng.choice(["_RETRY", "_TRANSIENT", "_AGAIN"])}

    @staticmethod
    def src(P, ver):
        return (f"def classify(status: int, retryable: bool) -> str:\n"
                f"    exact = {P['exact']!r}\n"
                f"    if status in exact:\n"
                f"        code = exact[status]\n"
                f"    else:\n"
                f"        code = {P['default']!r}\n"
                f"        for lo, hi, c in {P['rows']!r}:\n"
                f"            if lo <= status <= hi:\n"
                f"                code = c\n"
                f"                break\n"
                f"    if retryable and code in {P['retry']!r}:\n"
                f"        code += {P['suffix']!r}\n"
                f"    return code\n")

    @staticmethod
    def doc(P, ver, dep):
        (st, c), = P["exact"].items()
        rows = [f"| {lo}..{hi} (inclusive) | {cc} |" for lo, hi, cc in P["rows"]]
        return "\n".join([f"## Error code mapping v{ver}" + (" (deprecated)" if dep else ""), "",
                          f"1. Status {st} always maps to `{c}` (checked before the table).",
                          "2. Otherwise take the FIRST row of this table (top to bottom) whose range contains the status:", "",
                          "| status range | code |", "|---|---|"] + rows +
                         ["", f"3. If no row matches, the code is `{P['default']}`.",
                          f"4. If `retryable` is true and the code is one of {', '.join('`' + x + '`' for x in P['retry'])}, "
                          f"append `{P['suffix']}` to it.", ""])

    @staticmethod
    def inputs(rng):
        out = []
        for st in [200, 302, 400, 401, 404, 408, 409, 418, 429, 450, 499, 500, 503, 599, 600]:
            out.append((st, rng.random() < 0.5))
        out += [(429, True), (500, True), (404, True), (301, True)]
        return out


class Check:
    name, entry, area, mod, topic = "check", "check_char", "ids", "checksum", "id-checksum"
    sig = "def check_char(code: str) -> str:"
    helpers = '''VALUE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _value(ch: str) -> int:
    """Value of one ID character: its index in VALUE_ALPHABET (case-insensitive)."""
    return VALUE_ALPHABET.index(ch.upper())
'''

    @staticmethod
    def params(rng):
        m = rng.choice([11, 13, 17, 19, 23, 29, 31])
        return {"w": [rng.randint(2, 9) for _ in range(3)], "m": m, "off": rng.randint(1, m - 1),
                "out": "".join(rng.sample("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ", m))}

    @staticmethod
    def src(P, ver):
        return (f"def check_char(code: str) -> str:\n"
                f"    w = {P['w']!r}\n"
                f"    s = 0\n"
                f"    for k, ch in enumerate(reversed(code)):\n"
                f"        s += w[k % 3] * _value(ch)\n"
                f"    return {P['out']!r}[(s + {P['off']}) % {P['m']}]\n")

    @staticmethod
    def doc(P, ver, dep):
        return "\n".join([f"## Check character v{ver}" + (" (deprecated)" if dep else ""), "",
                          f"- Weights {P['w'][0]}, {P['w'][1]}, {P['w'][2]} are applied cyclically starting from the RIGHTMOST "
                          "character (rightmost * first weight, next to the left * second weight, ...).",
                          "- s = sum of weight * _value(character).",
                          f"- The check character is OUTPUT_ALPHABET[(s + {P['off']}) mod {P['m']}] with "
                          f"OUTPUT_ALPHABET = \"{P['out']}\" ({P['m']} characters).", ""])

    @staticmethod
    def inputs(rng):
        out = [("0",), ("A1",), ("ZZZZ",)]
        for _ in range(6):
            out.append(("".join(rng.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(3, 14))),))
        return out


TEMPLATES = [Frame, Fee, Key, Backoff, Classify, Check]


def _call(src, prelude, entry, args):
    g = {"__name__": "longacc_ref"}
    exec(compile(prelude + "\n" + src, "<ref>", "exec"), g)     # our own reference code (generator side, trusted)
    return g[entry](*args)


def make_coding_needle(T, rng, pkg, ver):
    """Return a needle dict for template T, or raise if the decoy could not be made distinguishable."""
    for _ in range(50):
        cur, old = T.params(rng), T.params(rng)
        if cur == old:
            continue
        inputs = T.inputs(rng) + (T.edges(cur) if hasattr(T, "edges") else [])
        ref, dec = T.src(cur, ver), T.src(old, ver - 1)
        exp = [_call(ref, T.helpers, T.entry, a) for a in inputs]
        dexp = [_call(dec, T.helpers, T.entry, a) for a in inputs]
        diff = sum(1 for x, y in zip(exp, dexp) if x != y)
        if diff >= 2:
            break
    else:
        raise RuntimeError(f"{T.name}: decoy not distinguishable")
    modname = f"{pkg}.{T.area}.{T.mod}"
    path = f"{pkg}/{T.area}/{T.mod}.py"
    docpath = f"docs/{T.topic}.md"
    title = {"frame": "Frame layout", "fee": "Fee schedule", "key": "Key format", "backoff": "Retry schedule",
             "classify": "Error code mapping", "check": "Check character"}[T.name]
    stub = (f'{T.sig}\n'
            f'    """Implements the CURRENT_VERSION rules of {docpath}, section "{title} v<CURRENT_VERSION>".\n\n'
            f'    Older sections of that document are kept for reference only and must not be used.\n'
            f'    """\n'
            f'    raise NotImplementedError("{T.entry}")\n')
    module_src = (f'"""{pkg}.{T.area}.{T.mod}: {title.lower()} (see {docpath})."""\n' + T.helpers +
                  f"\n\nCURRENT_VERSION = {ver}\n\n\n" + stub)
    secs = [(ver, cur, False), (ver - 1, old, True)]
    rng.shuffle(secs)
    doc = (f"# {title}\n\nThis document defines the {title.lower()} used by `{path}`. "
           f"The module constant CURRENT_VERSION selects the section that is in force.\n\n" +
           "\n".join(T.doc(P, v, dep) for v, P, dep in secs))
    tests = "\n".join(f"assert {T.entry}(*{a!r}) == {e!r}, {T.entry + ' case ' + str(i)!r}" for i, (a, e) in enumerate(zip(inputs, exp)))
    return {"template": T.name, "entry_point": T.entry, "module_name": modname, "path": path, "doc_path": docpath,
            "module_src": module_src, "doc": doc, "reference": ref, "decoy": dec, "test": tests, "version": ver,
            "n_asserts": len(inputs), "decoy_asserts_differ": diff}


# ====================================================================== filler
def filler_files(root=FILLER_ROOT):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in (".git", "node_modules", "__pycache__"))
        for f in sorted(fn):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    t = fh.read()
                if len(t) > 400:
                    out.append((os.path.relpath(p, root).replace("\\", "/"), t))
    return out


def _render(path, text, lang="python"):
    return f"\n### File: {path}\n```{lang}\n{text.rstrip()}\n```\n"


def _cut(bpe, text, max_tokens):
    if bpe.count(text) <= max_tokens:
        return text
    lines, acc, out = text.split("\n"), 0, []
    for ln in lines:
        n = bpe.count(ln + "\n")
        if acc + n > max_tokens:
            break
        out.append(ln)
        acc += n
    return "\n".join(out) + "\n# ... (file truncated in this snapshot)"


# ====================================================================== build
INSTR_LC = {
    "zh": ("以上是一個專案的原始碼快照。請實作 `{path}` 裡的 `{entry}`：完全依照它的 docstring 所指的規格文件中、"
           "目前版本（CURRENT_VERSION）的那一節，不要用已淘汰的舊版。該檔案中已定義的名稱（helper、常數）都可以直接使用。"
           "請只回覆完整的函式（含需要的 import），放在一個 ```python 程式碼區塊中。"),
    "en": ("Above is a snapshot of a project's source code. Implement `{entry}` in `{path}`: follow exactly the section of the "
           "specification its docstring points to for the CURRENT_VERSION of that module, not a deprecated one. Names "
           "already defined in that file (helpers, constants) are available. Reply with only the complete function "
           "(and any imports it needs) in a single ```python code block."),
}
INSTR_KV = {
    "zh": "在上面的程式碼中，常數 `CFG_{name}` 被設成哪個字串值？只回覆那個值（不含引號），不要其他文字。",
    "en": "In the code above, which string value is assigned to the constant `CFG_{name}`? Reply with the value only (no quotes), nothing else.",
}


def build_context(bpe, files, rng, L, n_code, n_kv, cid, zh_frac=0.75):
    budget = L - OUTPUT_RESERVE - TEMPLATE_RESERVE - 400        # 400: the question
    pkg = rng.choice(PKGS)
    temps = rng.sample(TEMPLATES, n_code)
    needles = [make_coding_needle(T, rng, pkg, rng.randint(3, 9)) for T in temps]
    # needle files: module and doc at different depths (stratified)
    segs = []   # (depth, rendered text, tag)
    k = len(needles)
    for i, nd in enumerate(needles):
        d_mod = (i + rng.uniform(0.1, 0.9)) / k * 0.94 + 0.03
        d_doc = (d_mod + rng.uniform(0.25, 0.75)) % 1.0 * 0.94 + 0.03
        segs.append((d_mod, _render(nd["path"], nd["module_src"]), ("mod", i)))
        segs.append((d_doc, _render(nd["doc_path"], nd["doc"], "markdown"), ("doc", i)))
    need = sum(bpe.count(s[1]) for s in segs)
    order = list(range(len(files)))
    rng.shuffle(order)
    fill, tot = [], need + 40
    for j in order:
        p, t = files[j]
        t = _cut(bpe, t, MAX_FILE_TOKENS)
        r = _render(p, t)
        n = bpe.count(r)
        if tot + n > budget:
            if budget - tot > 800:       # fill the rest with a cut piece of this file
                t2 = _cut(bpe, t, budget - tot - 120)
                r = _render(p, t2)
                fill.append([r, bpe.count(r)])
            break
        fill.append([r, n])
        tot += n
    # kv needles inside filler files (top-level line boundary), stratified depths
    kvs = []
    used_vals = set()
    for i in range(n_kv):
        d = (i + rng.uniform(0.1, 0.9)) / max(1, n_kv) * 0.94 + 0.03
        while True:
            name = f"{rng.choice(WORDS).upper()}_{rng.randint(100, 999)}_{cid.split('_')[-1]}{i}"
            val = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
            dv = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
            if val not in used_vals and dv not in used_vals and val != dv:
                break
        used_vals |= {val, dv}
        kvs.append({"name": name, "value": val, "decoy": dv, "depth_target": d})
    # place kv lines into the filler file covering their depth
    tot_f = sum(n for _, n in fill) or 1
    cum = 0
    bounds = []
    for r, n in fill:
        bounds.append((cum / tot_f, (cum + n) / tot_f))
        cum += n
    for kv in kvs:
        fi = next((ix for ix, (a, b) in enumerate(bounds) if a <= kv["depth_target"] < b), len(fill) - 1)
        r = fill[fi][0]
        lines = r.split("\n")
        top = [ix for ix in range(3, len(lines) - 2) if lines[ix] and not lines[ix][0].isspace() and not lines[ix - 1].rstrip().endswith(("\\", ",", "(", "[", "{"))] or [3]
        at = top[len(top) // 2] if len(top) > 0 else 3
        lines.insert(at, f'CFG_{kv["name"]} = "{kv["value"]}"')
        lines.insert(min(len(lines) - 2, at + 1 + rng.randint(2, 30)), f'CFG_{kv["name"]}_OLD = "{kv["decoy"]}"  # previous value')
        fill[fi][0] = "\n".join(lines)
    # interleave needle files into the filler by depth
    pieces = [(i, r) for i, (r, _) in enumerate(fill)]
    cum, pos = 0, []
    for i, (r, n) in enumerate(fill):
        pos.append(cum / tot_f)
        cum += n
    out_parts, tags = [], []
    segs.sort(key=lambda s: s[0])
    si = 0
    for i, r in pieces:
        while si < len(segs) and segs[si][0] <= pos[i]:
            out_parts.append(segs[si][1])
            tags.append(segs[si][2])
            si += 1
        out_parts.append(r)
        tags.append(("fill", i))
    while si < len(segs):
        out_parts.append(segs[si][1])
        tags.append(segs[si][2])
        si += 1
    head = (f"The following is a snapshot of a source repository ({pkg} and vendored third-party code). "
            "Each file starts with a '### File:' line.\n")
    counts = [bpe.count(p) for p in out_parts]
    total = bpe.count(head) + sum(counts)
    context = head + "".join(out_parts)
    # actual depths (token offsets)
    offs, acc = {}, bpe.count(head)
    for tag, n, part in zip(tags, counts, out_parts):
        offs[tag] = acc / total
        if tag[0] == "fill":
            for kv in kvs:
                if f'CFG_{kv["name"]} = "' in part:
                    kv["depth"] = (acc + bpe.count(part[:part.index(f'CFG_{kv["name"]} = "')])) / total
        acc += n
    # sanity: each needle unique in the context
    for nd in needles:
        assert context.count(f"def {nd['entry_point']}(") == 1, nd["entry_point"]
        assert context.count(nd["path"]) >= 2  # file header + docstring of the doc
    for kv in kvs:
        assert context.count(kv["value"]) == 1 and context.count(kv["decoy"]) == 1, kv
        assert context.count(f'CFG_{kv["name"]} = ') == 1
    tasks = []
    ctx_name = [k for k, v in LENGTHS.items() if v == L][0] if L in LENGTHS.values() else str(L)
    common = {"set": "long", "license": "WHIRL synthetic needles + " + FILLER_LICENSE, "ctx": ctx_name, "ctx_tokens": L,
              "cluster": cid, "context_tokens_est": total, "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest()}
    for i, nd in enumerate(needles):
        lang = "zh" if rng.random() < zh_frac else "en"
        q = INSTR_LC[lang].format(path=nd["path"], entry=nd["entry_point"])
        tasks.append(dict(common, id=f"{cid}/lc{i}_{nd['template']}", source="lc_code", lang=lang, kind="lc",
                          entry_point=nd["entry_point"], prompt=context + "\n\n" + q, question=q, max_tokens=MAX_TOKENS_LC,
                          module_name=nd["module_name"], module_src=nd["module_src"], test=nd["test"], test_imports=[],
                          canonical=nd["reference"], reference=nd["reference"], decoy=nd["decoy"], template=nd["template"],
                          version=nd["version"], depth=round(offs[("mod", i)], 4), doc_depth=round(offs[("doc", i)], 4),
                          n_asserts=nd["n_asserts"], decoy_asserts_differ=nd["decoy_asserts_differ"]))
    for i, kv in enumerate(kvs):
        lang = "zh" if i % 2 == 0 else "en"
        q = INSTR_KV[lang].format(name=kv["name"])
        tasks.append(dict(common, id=f"{cid}/kv{i}", source="lc_kv", lang=lang, kind="kv", entry_point="",
                          prompt=context + "\n\n" + q, question=q, max_tokens=MAX_TOKENS_KV, answer=kv["value"],
                          decoy_answer=kv["decoy"], key=f"CFG_{kv['name']}", depth=round(kv.get("depth", kv["depth_target"]), 4)))
    return tasks, total


def build(bpe, lengths=("32k", "64k", "96k", "128k"), contexts=6, n_code=4, n_kv=6, seed=20261026, files=None):
    files = files if files is not None else filler_files()
    rng = random.Random(seed)
    tasks, ctxs = [], []
    for ln in lengths:
        L = LENGTHS[ln]
        for c in range(contexts):
            cid = f"{ln}_c{c}"
            t, total = build_context(bpe, files, random.Random(rng.randrange(1 << 30)), L, n_code, n_kv, cid)
            tasks += t
            ctxs.append({"cluster": cid, "ctx": ln, "target": L, "prompt_budget": L - OUTPUT_RESERVE,
                         "context_tokens_est": total, "n_tasks": len(t)})
    meta = {"suite": SUITE_VERSION, "seed": seed, "lengths": list(lengths), "contexts_per_length": contexts,
            "coding_per_context": n_code, "kv_per_context": n_kv, "n": len(tasks),
            "by_kind": {k: sum(1 for t in tasks if t["kind"] == k) for k in ("lc", "kv")},
            "by_lang": {k: sum(1 for t in tasks if t["lang"] == k) for k in ("zh", "en")},
            "contexts": ctxs, "filler": {"root": FILLER_ROOT, "license": FILLER_LICENSE, "files": len(files)},
            "token_count": "bpe.py (qwen35 pre-tokenizer, GGUF vocab) [E]; usage.prompt_tokens from the server [M]",
            "output_reserve": OUTPUT_RESERVE}
    return tasks, meta


def write(tasks, meta, out_dir, name="tasks_long"):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name + ".jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    meta = dict(meta, taskset_sha256=h, file=p)
    with open(os.path.join(out_dir, name + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return p, meta


# ====================================================================== grading
VAL_RE = re.compile(r"[a-z]+-[a-z]+-\d{4}")


def grade_kv(task, content):
    """Exact retrieval: pass iff the reply contains the right value and no other value-shaped string."""
    from extract import strip_think  # noqa: PLC0415
    t = strip_think(content or "")
    found = set(VAL_RE.findall(t))
    if task["answer"] in found and len(found) == 1:
        st = "pass"
    elif task.get("decoy_answer") in found:
        st = "decoy"
    elif not found:
        st = "no_answer"
    else:
        st = "wrong"
    return {"status": st, "passed": st == "pass", "rc": None, "seconds": 0.0, "jobbed": True, "stderr": "", "stdout": ""}


def lc_prelude(task):
    """Source that registers the needle module (and its parent packages) in sys.modules, so the model's
    `from <pkg>.<area>.<mod> import helper` works, then defines the module's names at top level."""
    return ("import sys as _lc_sys, types as _lc_types\n"
            "def _lc_register(_name, _src):\n"
            "    _parts = _name.split('.')\n"
            "    for _i in range(1, len(_parts)):\n"
            "        _p = '.'.join(_parts[:_i])\n"
            "        if _p not in _lc_sys.modules:\n"
            "            _m = _lc_types.ModuleType(_p); _m.__path__ = []; _lc_sys.modules[_p] = _m\n"
            "    _m = _lc_types.ModuleType(_name)\n"
            "    exec(compile(_src, _name.replace('.', '/') + '.py', 'exec'), _m.__dict__)\n"
            "    _lc_sys.modules[_name] = _m\n"
            f"_lc_register({task['module_name']!r}, {task['module_src']!r})\n"
            "del _lc_register\n"
            + task["module_src"] + "\n\n# ---- model code ----\n")
