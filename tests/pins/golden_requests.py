# arch024-S8 (PLAN A7): the golden server suite's chat requests (tests/server/server_gate.cpp suiteGolden) as
# requests.jsonl rows for the prompt token pin, plus the /v1/completions prompt shapes (http.cpp completionTokens).
# goldenDoc / fnv64 are replicas of server_gate.cpp:2272-2305, checked against the gate selftest's constants
# (server_gate.cpp:2500-2504) before anything is written.
# Turn 2 of a session carries the model's turn-1 reply as the assistant message (GPU output); here it is a fixed
# sentence, so the row pins the multi-turn document template path, not that exact reply.
# The vision row renders the image placeholder only (x_allow_media); the image token expansion (buildVis, needs an
# mmproj) is covered by the server-tests vision cases (MockVision).
# Usage: python -I golden_requests.py > rows.txt   (rows are appended to requests.jsonl by hand, then recorded on the base)
# SPDX-License-Identifier: Apache-2.0
import json
import sys

W = ["cache", "kernel", "token", "prefill", "decode", "buffer", "stream", "thread", "slot", "page",
     "matrix", "vector", "scale", "block", "layer", "expert", "router", "state", "queue", "batch",
     "returns", "updates", "reads", "writes", "checks", "stores", "loads", "maps", "splits", "merges",
     "the", "a", "each", "every", "one", "two", "first", "last", "next", "previous"]
M32 = 0xFFFFFFFF


def golden_doc(seed, words):
    x = (seed * 2654435761 + 12345) & M32
    o = []
    for i in range(words):
        x = (x * 1664525 + 1013904223) & M32
        if i % 12 == 0:
            o.append("\nSection {}.{}: function f_{}_{}() ".format(seed, i // 12, seed, i // 12))
        o.append(W[(x >> 16) % len(W)])
        o.append(". " if i % 12 == 11 else " ")
    return "".join(o)


def fnv64(s):
    h = 1469598103934665603
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


assert fnv64("") == 1469598103934665603
assert fnv64(golden_doc(1, 2400)) == 0xc1d46b50c56aaea0, hex(fnv64(golden_doc(1, 2400)))
assert fnv64(golden_doc(2, 1500)) == 0x2b9d4e7c2c35cb70
assert fnv64(golden_doc(3, 1500)) == 0x2efb3273d0550c17


def msg(role, content):
    return {"role": role, "content": content}


def think_kw(on):
    return {"enable_thinking": True, "reasoning_effort": "medium"} if on else {"enable_thinking": False}


def chat(messages, think, max_tokens):
    return {"model": "m", "messages": messages, "chat_template_kwargs": think_kw(think), "temperature": 0, "max_tokens": max_tokens}


P_SHORT = "請用繁體中文說明快速排序的原理，並用 C++ 寫一個泛型的 quicksort 函式，程式碼註解用中文。"
P_PAIR_A = "Write a C function that reverses a singly linked list, and explain it."
P_PAIR_B = "請用繁體中文說明 TCP 三向交握的過程，並用 Python 寫一個簡單的 socket 伺服器。"
doc_long = golden_doc(1, 2400) + "\n\nIn one sentence: what does function f_1_7 do?"
doc_a, doc_b = golden_doc(2, 1500), golden_doc(3, 1500)
q1 = "Name the first function above and what it returns, in one sentence."
q2 = "Now name the last function above, in one sentence."
A1 = "The first function above is f_2_0(), which returns the cache state."  # fixed stand-in for the turn-1 reply


def t1(d):
    return [msg("user", d + "\n\n" + q1)]


def t2(d):
    return [msg("user", d + "\n\n" + q1), msg("assistant", A1), msg("user", q2)]


rows = [
    ("golden_short", chat([msg("user", P_SHORT)], False, 48)),
    ("golden_short_stream", dict(chat([msg("user", P_SHORT)], False, 48), stream=True, stream_options={"include_usage": True})),
    ("golden_think", chat([msg("user", P_PAIR_A)], True, 64)),
    ("golden_long", chat([msg("user", doc_long)], False, 32)),
    ("golden_vis", dict(chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "text", "text": "Describe this image: the shapes and their colors."}]}], False, 32), x_allow_media=True)),
    ("golden_sessA_t1", chat(t1(doc_a), False, 32)),
    ("golden_sessB_t1", chat(t1(doc_b), False, 32)),
    ("golden_sessA_t2", chat(t2(doc_a), False, 32)),
    ("golden_sessB_t2", chat(t2(doc_b), False, 32)),
    ("golden_solo0", chat([msg("user", P_PAIR_A)], False, 48)),
    ("golden_solo1", chat([msg("user", P_PAIR_B)], False, 48)),
    # /v1/completions prompt shapes (no chat template): a string, a list with one string, token ids
    ("cmpl_string", {"prompt": "def quicksort(xs):\n    ", "max_tokens": 16}),
    ("cmpl_list1", {"prompt": ["請用繁體中文說明 " + P_PAIR_A]}),
    ("cmpl_ids", {"prompt": [9707, 11, 1879, 0, 13]}),
]
out = sys.stdout.buffer
for name, body in rows:
    out.write((name + "\t" + json.dumps(body, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
