# SPDX-License-Identifier: Apache-2.0
r"""alloc_seq: the device / pinned-host allocation sequence of a CLI model load, against a pinned seqhash per card.

arch025-allocseq (PLAN N11-alloc). The allocation log (src/hip/alloc_log.h, WHIRL_ALLOC_LOG) hashes every allocation
event (kind, bytes, ordinal) of a process; invariant I2 of arch025.PLAN.md says the Model's allocations keep their
sizes and order. Outputs do not show an order change (mutant m1-state-order: the DeltaNet state allocated before the
layer's weights survived every golden / selftest step), so this step reads the log itself.

    python tools/gates/alloc_seq_check.py run    --exe whirl.exe --device r9700 --arch gfx1201
                                                 --model swift=<gguf> --model moe=<gguf> --pins tests/pins/alloc_seq
                                                 --work DIR [--record]
    python tools/gates/alloc_seq_check.py compare A.txt B.txt     (two alloc logs, e.g. a mutant against its control)
    python tools/gates/alloc_seq_check.py --selftest              (CPU: a fake whirl.exe writes planted logs)

Per model (a "cell"): `whirl chat <gguf> x --precise --ctx 4096 --max-tokens 1 --device <card> --dump-only` with
WHIRL_ALLOC_LOG=<work>\<key> and no other WHIRL_* variable, then the alloc log of the chat process (whirl.exe also
starts a --whirl-gpu-probe child that writes its own, empty log: ignored) is compared with the pin line
"<key> precise ctx4096 <model file> <size> events N seqhash H" of tests/pins/alloc_seq/<arch>.txt.
Why this command (progress/arch025-allocseq.PRIOR_ART.md): an explicit --ctx is never shrunk to the free VRAM
(src/model/loader.cpp cliCtxFit), and --dump-only exits after load + tune, before the decode loop sizes its snapshot sets
from hip::memInfo() (src/cli/decode.cpp); the server's budget-sized KV pool changed with a HIP runtime update
(arch025:n2b-golden), the CLI load with a given ctx does not read the budget. The saved S0b-2 / M1 matrices gave one
seqhash per (card, model, mode) over 14 runs of three exes. A cell whose output says the prefill GEMMs were tuned now
(first run with that model on that GPU: "prefill GEMM autotune:") is run once more with the cache written.
Why a pin and not two runs: m1-state-order is deterministic, so two runs of one exe give the same (wrong) hash; only a
reference recorded from a known-good build kills it. gpumut needs no change: its control passes, a mutant fails.
--record runs every cell twice, refuses when the two hashes differ (not deterministic), and writes the pin file and
the event lists (<arch>_<key>.events.txt: the first differing event is printed on a mismatch).
Exit: 0 every cell matches its pin (or --record wrote them); 1 a cell failed; 2 usage / setup error.
Last line: "alloc seq check: ok (<n>/<n> cells, <arch>)" or "alloc seq check: FAIL (...)".
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

MODE = "precise"
CTX = 4096
TIMEOUT_S = 900
SUMMARY_RE = re.compile(r"^# events (\d+) seqhash ([0-9a-f]{16})\s*$")
PIN_RE = re.compile(r"^(\S+) (\S+) ctx(\d+) (\S+) (\d+) events (\d+) seqhash ([0-9a-f]{16})\s*$")
EVENT_RE = re.compile(r"^\d+ (dev|host|free|hostfree) ")


def cell_args(gguf: str, device: str) -> list[str]:
    return ["chat", gguf, "x", f"--{MODE}", "--ctx", str(CTX), "--max-tokens", "1", "--device", device, "--dump-only"]


def parse_log(path: str) -> dict:
    """the summary and event list of one alloc log file"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\r\n") for l in f]
    d = {"path": path, "cmd": "", "events": None, "seqhash": None, "list": None, "dropped": False, "ev": []}
    for l in lines:
        if l.startswith("# cmd:"):
            d["cmd"] = l[6:].strip()
        elif (m := SUMMARY_RE.match(l)):
            d["events"], d["seqhash"] = int(m.group(1)), m.group(2)
        elif l.startswith("# list "):
            d["list"] = l[7:].strip()
        elif l.startswith("# lines dropped"):
            d["dropped"] = True
        elif EVENT_RE.match(l):
            d["ev"].append(l)
    return d


def is_chat_log(d: dict) -> bool:
    # the cmd line is the quoted exe path then the arguments: the chat process, not the --whirl-gpu-probe child
    return re.search(r"(^|\s)chat(\s|$)", d["cmd"]) is not None and "--whirl-gpu-probe" not in d["cmd"]


def first_diff(a: list[str], b: list[str]) -> str:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return f"first differing event #{i}: pinned '{a[i]}' got '{b[i]}'"
    if len(a) != len(b):
        return f"event lists equal for {min(len(a), len(b))} events, then lengths differ (pinned {len(a)}, got {len(b)})"
    return "event lists identical (only the summary differs)"


def child_env(log_dir: str) -> dict:
    # no WHIRL_* of the caller (a WHIRL_KV / WHIRL_MODE in the shell would change what is pinned; WHIRL_GATE_* are the
    # gate's own inputs, not read by whirl.exe), except the R9700 lock marker
    env = {k: v for k, v in os.environ.items() if not (k.upper().startswith("WHIRL_") and k.upper() != "WHIRL_R9700_LOCK_HELD")}
    env["WHIRL_ALLOC_LOG"] = log_dir
    return env


def run_once(prefix: list[str], gguf: str, device: str, arch: str, log_dir: str, out) -> tuple[dict | None, str, str]:
    """run one cell once -> (parsed chat log or None, why-failed, output text)"""
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    cmd = prefix + cell_args(gguf, device)
    try:
        p = subprocess.run(cmd, env=child_env(log_dir), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {TIMEOUT_S} s", ""
    except OSError as e:
        return None, f"could not start {cmd[0]}: {e}", ""
    text = (p.stdout or "") + (p.stderr or "")
    with open(os.path.join(os.path.dirname(log_dir), os.path.basename(log_dir) + ".out.txt"), "a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n{text}\n[exit {p.returncode}]\n")
    if p.returncode != 0:
        tail = " | ".join([l for l in text.splitlines() if l.strip()][-3:])
        return None, f"exit {p.returncode}: {tail}", text
    if arch not in text:
        return None, f"the output never names {arch} (wrong GPU or no device line)", text
    logs = []
    for fn in sorted(os.listdir(log_dir)):
        if fn.startswith("alloc_") and fn.endswith(".txt"):
            logs.append(parse_log(os.path.join(log_dir, fn)))
    chat = [d for d in logs if is_chat_log(d)]
    if len(chat) != 1:
        return None, f"{len(chat)} alloc logs of the chat process in {log_dir} (of {len(logs)} logs; want exactly 1)", text
    d = chat[0]
    if d["seqhash"] is None:
        return None, f"no '# events N seqhash H' line in {d['path']}", text
    if d["list"] != "on" or d["dropped"] or len(d["ev"]) != d["events"]:
        return None, f"event list incomplete in {d['path']} (list {d['list']}, {len(d['ev'])} lines for {d['events']} events)", text
    return d, "", text


def run_cell(prefix, gguf, device, arch, log_dir, out):
    d, why, text = run_once(prefix, gguf, device, arch, log_dir, out)
    if d is not None and "prefill GEMM autotune:" in text:
        out(f"alloc seq: {os.path.basename(log_dir)}: the prefill GEMMs were tuned in this run (no tune cache): run again with the cache")
        d, why, text = run_once(prefix, gguf, device, arch, log_dir, out)
        if d is not None and "prefill GEMM autotune:" in text:
            return None, "tuned again on the second run (the tune cache is not written)"
    return d, why


def read_pins(path: str) -> dict:
    pins = {}
    if not os.path.exists(path):
        return pins
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            m = PIN_RE.match(l)
            if not m:
                raise ValueError(f"{path}: bad pin line '{l}'")
            pins[m.group(1)] = {"mode": m.group(2), "ctx": int(m.group(3)), "file": m.group(4), "size": int(m.group(5)),
                                "events": int(m.group(6)), "seqhash": m.group(7)}
    return pins


def check(prefix: list[str], device: str, arch: str, models: list[tuple[str, str]], pins_dir: str, work: str, record: bool, out) -> int:
    pin_file = os.path.join(pins_dir, f"{arch}.txt")
    try:
        pins = read_pins(pin_file)
    except ValueError as e:
        out(f"alloc seq check: FAIL ({e})")
        return 1
    os.makedirs(work, exist_ok=True)
    bad = []
    got = {}
    for key, gguf in models:
        if not os.path.isfile(gguf):
            out(f"alloc seq check: FAIL (model {key}: {gguf} not found)")
            return 2
        fname, size = os.path.basename(gguf), os.path.getsize(gguf)
        d, why = run_cell(prefix, gguf, device, arch, os.path.join(work, key), out)
        if d is None:
            out(f"alloc seq: {key} FAIL {why}")
            bad.append(key)
            continue
        if record:
            d2, why2 = run_cell(prefix, gguf, device, arch, os.path.join(work, key + "_2"), out)
            if d2 is None:
                out(f"alloc seq: {key} FAIL second record run: {why2}")
                bad.append(key)
                continue
            if (d2["seqhash"], d2["events"]) != (d["seqhash"], d["events"]):
                out(f"alloc seq: {key} FAIL not deterministic: run 1 events {d['events']} seqhash {d['seqhash']}, run 2 events {d2['events']} "
                    f"seqhash {d2['seqhash']} ({first_diff(d['ev'], d2['ev'])}): nothing recorded")
                bad.append(key)
                continue
            got[key] = (fname, size, d)
            out(f"alloc seq: {key} recorded events {d['events']} seqhash {d['seqhash']} (2 runs equal)")
            continue
        p = pins.get(key)
        if p is None:
            out(f"alloc seq: {key} FAIL no pin for {key} in {pin_file} (record it on this GPU with: alloc_seq_check.py run ... --record); "
                f"got events {d['events']} seqhash {d['seqhash']}")
            bad.append(key)
            continue
        if (p["mode"], p["ctx"], p["file"], p["size"]) != (MODE, CTX, fname, size):
            out(f"alloc seq: {key} FAIL the pin was recorded for {p['file']} ({p['size']} bytes) {p['mode']} ctx{p['ctx']}, "
                f"this run is {fname} ({size} bytes) {MODE} ctx{CTX}")
            bad.append(key)
            continue
        if (p["seqhash"], p["events"]) != (d["seqhash"], d["events"]):
            ef = os.path.join(pins_dir, f"{arch}_{key}.events.txt")
            diag = "no pinned event list" if not os.path.exists(ef) else first_diff(
                [l.rstrip("\r\n") for l in open(ef, encoding="utf-8") if EVENT_RE.match(l)], d["ev"])
            out(f"alloc seq: {key} FAIL pinned events {p['events']} seqhash {p['seqhash']}, got events {d['events']} seqhash {d['seqhash']}; {diag}")
            out(f"alloc seq: {key}   log {d['path']}")
            bad.append(key)
            continue
        out(f"alloc seq: {key} ok events {d['events']} seqhash {d['seqhash']}")
    n = len(models)
    if record:
        if bad:
            out(f"alloc seq check: FAIL (record: {len(bad)} of {n} cells failed: {', '.join(bad)}; nothing written)")
            return 1
        os.makedirs(pins_dir, exist_ok=True)
        keep = {k: v for k, v in pins.items() if k not in got}
        hdr = [f"# alloc_seq pins for {arch} (tools/gates/alloc_seq_check.py; arch025-allocseq). One line per model:",
               "# <key> <mode> ctx<ctx> <model file> <bytes> events <N> seqhash <H>. Recorded "
               f"{datetime.datetime.now().isoformat(timespec='seconds')} on --device {device} by {prefix[-1]}.",
               "# Re-record (run ... --record) only when an allocation change is intended or the model file changed, and say why in the commit."]
        lines = hdr + [f"{k} {v['mode']} ctx{v['ctx']} {v['file']} {v['size']} events {v['events']} seqhash {v['seqhash']}" for k, v in sorted(keep.items())]
        for k, (fname, size, d) in sorted(got.items()):
            lines.append(f"{k} {MODE} ctx{CTX} {fname} {size} events {d['events']} seqhash {d['seqhash']}")
            with open(os.path.join(pins_dir, f"{arch}_{k}.events.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write(f"# alloc_seq event list {arch} {k} {MODE} ctx{CTX} {fname}: events {d['events']} seqhash {d['seqhash']}\n")
                f.write("\n".join(d["ev"]) + "\n")
        with open(pin_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        out(f"alloc seq check: ok (recorded {n}/{n} cells, {arch}: {pin_file})")
        return 0
    if bad:
        out(f"alloc seq check: FAIL ({len(bad)} of {n} cells, {arch}: {', '.join(bad)})")
        return 1
    out(f"alloc seq check: ok ({n}/{n} cells, {arch})")
    return 0


def compare(a: str, b: str, out) -> int:
    da, db = parse_log(a), parse_log(b)
    if da["seqhash"] is None or db["seqhash"] is None:
        out("alloc seq compare: FAIL (a log without a summary line)")
        return 2
    if (da["seqhash"], da["events"]) == (db["seqhash"], db["events"]):
        out(f"alloc seq compare: same (events {da['events']} seqhash {da['seqhash']})")
        return 0
    out(f"alloc seq compare: DIFFERENT (A events {da['events']} seqhash {da['seqhash']}, B events {db['events']} seqhash {db['seqhash']}; "
        f"{first_diff(da['ev'], db['ev'])})")
    return 1


# ------------------------------------------------------------------------------------------------- selftest
FAKE = r'''
import json, os, sys
c = json.load(open(os.environ["ALLOCSEQ_FAKE"], encoding="utf-8"))
n_file = os.environ["ALLOCSEQ_FAKE"] + ".n"
n = int(open(n_file).read()) if os.path.exists(n_file) else 0
open(n_file, "w").write(str(n + 1))
a = sys.argv[1:]
if any(k.startswith("WHIRL_") and k not in ("WHIRL_ALLOC_LOG", "WHIRL_R9700_LOCK_HELD") for k in os.environ):
    print("fake: stray WHIRL_ variable"); sys.exit(7)
if a[0] != "chat" or "--dump-only" not in a or a[a.index("--ctx") + 1] != "4096" or "--precise" not in a:
    print("fake: unexpected arguments", a); sys.exit(8)
model = os.path.basename(a[1])
cell = c["cells"][model]
runs = cell.get("runs")
ev = runs[min(n, len(runs) - 1)] if runs else cell["events"]
d = os.environ["WHIRL_ALLOC_LOG"]
def write(name, cmd, events, h):
    with open(os.path.join(d, name), "w") as f:
        f.write("# whirl alloc log v1\n# cmd: " + cmd + "\n")
        f.write("# events %d seqhash %s\n# dev 0 allocs 0 bytes, 0 frees (0 live, 0 bytes; peak 0 bytes)\n" % (len(events), h))
        f.write("# unknown frees 0\n# list on\n")
        for i, e in enumerate(events):
            f.write("%d %s\n" % (i, e))
if c.get("probe_log"):
    write("alloc_1_a.txt", '"whirl.exe" --whirl-gpu-probe', [], "14650fb0739d0383")
for i in range(c.get("chat_logs", 1)):
    import hashlib
    h = hashlib.sha256("|".join(ev).encode()).hexdigest()[:16]
    write("alloc_2_%d.txt" % i, '"whirl.exe" ' + " ".join(a), ev, h)
print("  device: fake " + c.get("arch", "gfx1201"))
if c.get("tune_first") and n == 0:
    print("  prefill GEMM autotune: 1.0 s")
else:
    print("  prefill GEMM tune: cached (x)")
sys.exit(c.get("exit", 0))
'''

EV_SW = ["dev 100 #0", "dev 200 #1", "dev 64 #2", "dev 64 #3", "host 8 #0", "hostfree #0 8"]
EV_SW_M1 = ["dev 64 #0", "dev 64 #1", "dev 100 #2", "dev 200 #3", "host 8 #0", "hostfree #0 8"]  # state before weights
EV_MOE = ["dev 300 #0", "dev 32 #1", "dev 32 #2"]


def selftest() -> int:
    root = tempfile.mkdtemp(prefix="allocseq_selftest_")
    fake = os.path.join(root, "fake_whirl.py")
    with open(fake, "w", encoding="utf-8") as f:
        f.write(FAKE)
    models = []
    for k, name in (("swift", "Swift-A.gguf"), ("moe", "Ornith.gguf")):
        p = os.path.join(root, name)
        with open(p, "wb") as f:
            f.write(b"x" * (10 if k == "swift" else 20))
        models.append((k, p))
    prefix = [sys.executable, "-I", fake]
    results = []

    def case(name, cfg, want_rc, want_lines=(), record=False, pins=None, arch="gfx1201", only=None, env_extra=None):
        cdir = os.path.join(root, name)
        os.makedirs(cdir, exist_ok=True)
        cf = os.path.join(cdir, "fake.json")
        with open(cf, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        pins_dir = pins if pins is not None else os.path.join(cdir, "pins")
        old = dict(os.environ)
        os.environ["ALLOCSEQ_FAKE"] = cf
        for k, v in (env_extra or {}).items():
            os.environ[k] = v
        lines = []
        try:
            rc = check(prefix, "r9700", arch, only or models, pins_dir, os.path.join(cdir, "work"), record, lines.append)
        finally:
            os.environ.clear()
            os.environ.update(old)
        text = "\n".join(lines)
        ok = rc == want_rc and all(re.search(w, text, re.M) for w in want_lines)
        results.append((name, ok, rc, text))
        return pins_dir

    base = {"cells": {"Swift-A.gguf": {"events": EV_SW}, "Ornith.gguf": {"events": EV_MOE}}}
    # 1 record (2 runs equal) writes the pins and the event lists; 2 the same build then passes
    pins = case("record", dict(base, probe_log=True), 0, [r"^alloc seq check: ok \(recorded 2/2 cells, gfx1201", r"^alloc seq: swift recorded events 6 "], record=True)
    case("pass", dict(base, probe_log=True), 0, [r"^alloc seq: swift ok events 6 ", r"^alloc seq: moe ok events 3 ", r"^alloc seq check: ok \(2/2 cells, gfx1201\)$"], pins=pins)
    # 3 m1-state-order: the same allocations, the state moved before the weights -> other seqhash, first diff #0
    m1 = {"cells": {"Swift-A.gguf": {"events": EV_SW_M1}, "Ornith.gguf": {"events": EV_MOE}}}
    case("m1_order", m1, 1, [r"^alloc seq: swift FAIL pinned events 6 seqhash \w+, got events 6 seqhash \w+; first differing event #0: pinned '0 dev 100 #0' got '0 dev 64 #0'",
                             r"^alloc seq: moe ok", r"^alloc seq check: FAIL \(1 of 2 cells, gfx1201: swift\)$"], pins=pins)
    # 4 one allocation more (a new buffer): count differs
    case("extra_alloc", {"cells": {"Swift-A.gguf": {"events": EV_SW + ["dev 4 #4"]}, "Ornith.gguf": {"events": EV_MOE}}}, 1,
         [r"^alloc seq: swift FAIL pinned events 6 .* got events 7 .*then lengths differ"], pins=pins)
    # 5 no pin file at all: FAIL, never a vacuous pass
    case("no_pins", base, 1, [r"^alloc seq: swift FAIL no pin for swift", r"^alloc seq check: FAIL \(2 of 2 cells"])
    # 6 pins of the other card (the gfx1151 file does not exist)
    case("other_card", dict(base, arch="gfx1151"), 1, [r"FAIL no pin for swift in .*gfx1151\.txt", r"check: FAIL \(2 of 2 cells, gfx1151"], pins=pins, arch="gfx1151")
    # 7 the run went to another GPU: the output names gfx1151, the gate is the R9700's
    case("wrong_gpu", dict(base, arch="gfx1151"), 1, [r"^alloc seq: swift FAIL the output never names gfx1201"], pins=pins)
    # 8 only the probe child's log: the chat process wrote none
    case("no_chat_log", dict(base, probe_log=True, chat_logs=0), 1, [r"FAIL 0 alloc logs of the chat process .* \(of 1 logs; want exactly 1\)"], pins=pins)
    # 9 two chat logs in the folder (a stale one): refused
    case("two_chat_logs", dict(base, chat_logs=2), 1, [r"FAIL 2 alloc logs of the chat process"], pins=pins)
    # 10 the exe fails
    case("exit_nonzero", dict(base, exit=3), 1, [r"^alloc seq: swift FAIL exit 3"], pins=pins)
    # 11 cold tune cache: the first run tunes, the second (cached) is the one compared
    case("tune_first", dict(base, tune_first=True), 0, [r"were tuned in this run .* run again", r"^alloc seq check: ok \(2/2 cells"], pins=pins)
    # 12 record refuses a non-deterministic sequence (two runs differ)
    case("record_nondet", {"cells": {"Swift-A.gguf": {"runs": [EV_SW, EV_SW_M1]}, "Ornith.gguf": {"events": EV_MOE}}}, 1,
         [r"^alloc seq: swift FAIL not deterministic: .*first differing event #0", r"nothing written"], record=True)
    # 13 a stray WHIRL_* variable of the caller is not passed to whirl.exe (the fake exits 7 when it sees one)
    case("env_clean", base, 0, [r"^alloc seq check: ok \(2/2 cells"], pins=pins, env_extra={"WHIRL_KV": "q8v", "WHIRL_ALLOC_LOG_RAW": "server", "WHIRL_GATE_SWIFT": "x.gguf"})
    # 14 the pin was recorded for another model file (size differs)
    other = os.path.join(root, "other")
    os.makedirs(other, exist_ok=True)
    big = os.path.join(other, "Swift-A.gguf")
    with open(big, "wb") as f:
        f.write(b"x" * 11)
    case("other_file", base, 1, [r"FAIL the pin was recorded for Swift-A\.gguf \(10 bytes\) precise ctx4096, this run is Swift-A\.gguf \(11 bytes\)"],
         pins=pins, only=[("swift", big)])
    # 15 compare: equal / different logs
    la, lb = os.path.join(root, "pass", "work", "swift"), os.path.join(root, "m1_order", "work", "swift")
    fa = [os.path.join(la, x) for x in (os.listdir(la) if os.path.isdir(la) else []) if x.startswith("alloc_2_")]
    fb = [os.path.join(lb, x) for x in (os.listdir(lb) if os.path.isdir(lb) else []) if x.startswith("alloc_2_")]
    ln = []
    if fa and fb:
        rc_same, rc_diff = compare(fa[0], fa[0], ln.append), compare(fa[0], fb[0], ln.append)
        results.append(("compare", rc_same == 0 and rc_diff == 1 and "first differing event #0" in ln[-1], rc_diff, "\n".join(ln)))
    else:
        results.append(("compare", False, -1, f"no planted chat log in {la} / {lb}"))
    n_ok = sum(1 for r in results if r[1])
    for name, ok, rc, text in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name} (rc {rc})")
        if not ok:
            print("    " + text.replace("\n", "\n    "))
    shutil.rmtree(root, ignore_errors=True)
    print(f"alloc seq selftest: {n_ok}/{len(results)} as expected")
    return 0 if n_ok == len(results) else 1


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exe", required=True)
    r.add_argument("--device", required=True)
    r.add_argument("--arch", required=True, choices=["gfx1201", "gfx1151"])
    r.add_argument("--model", action="append", required=True, metavar="KEY=GGUF")
    r.add_argument("--pins", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--record", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        return compare(a.a, a.b, print)
    models = []
    for mv in a.model:
        k, sep, p = mv.partition("=")
        if not sep or not k or not p:
            print(f"alloc seq check: FAIL (--model wants KEY=GGUF, got '{mv}')")
            return 2
        models.append((k, p))
    if not os.path.isfile(a.exe):
        print(f"alloc seq check: FAIL ({a.exe} not found)")
        return 2
    return check([a.exe], a.device, a.arch, models, a.pins, a.work, a.record, lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main())
