#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1b: smoke test of the tiny random-weight models (gen_tiny.py) and their use in place of the 27B.
#   1. CLI: whirl chat TINY (greedy, 48 tokens) twice -> identical text; wall time per run
#   2. server: whirl-server TINY, load time, /v1/completions greedy twice -> identical text (and once streamed)
#   3. --gate SUITES: whirl-server-gate --suite SUITES on the tiny model (and with --compare-27b also on the 27B), wall times
#
#   python tiny_check.py --device r97|8s [--models dense,moe] [--exe whirl.exe] [--gate pool] [--compare-27b] [--mode balance]
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib  # noqa: E402

TINY = os.path.join(qlib.WORK, "fl1b", "tiny")
TINYM = {"dense": os.path.join(TINY, "qwen35-tiny-L2-shuffle1.gguf"), "moe": os.path.join(TINY, "qwen35moe-tiny-L2-shuffle1.gguf")}
REAL = {"dense": qlib.MODEL, "moe": r"C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf"}
PROMPT = "Write a C++ function that returns the n-th Fibonacci number.\n"
OUT = os.path.join(qlib.WORK, "fl1b", "tiny_runs")


def cli_run(exe, model, dev, mode, pfile):
    t = time.time()
    r = subprocess.run([exe, "chat", model, "@" + pfile, "--raw", "--max-tokens", "48", "--" + mode, "--device", qlib.DEVICES[dev]],
                       env=qlib.clean_env(AMD_LOG_LEVEL=1), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, time.time() - t, r.stdout, r.stderr


def gen_text(stdout):
    # the generated text: stdout without the engine's info lines (they start with two spaces or a known prefix)
    return "\n".join(l for l in stdout.splitlines() if l.strip() and not l.startswith("  ") and "tok/s" not in l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["r97", "8s"], default="r97")
    ap.add_argument("--models", default="dense,moe")
    ap.add_argument("--exe", default=qlib.DEF_EXE)
    ap.add_argument("--mode", default="balance")
    ap.add_argument("--gate", default="", help="whirl-server-gate suites to run on the tiny model")
    ap.add_argument("--compare-27b", action="store_true")
    ap.add_argument("--ssd", action="store_true", help="leave the SSD tier on (suites tier / sys / restore_conc test it; "
                    "the suites use their own SSD dirs under the work dir, deleted afterwards)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--only-27b", action="store_true", help="--gate on the real model only")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pfile = os.path.join(OUT, "prompt.txt")
    open(pfile, "w", encoding="utf-8").write(PROMPT)
    res = {}
    ok_all = True
    with qlib.GpuLock():
        for m in ([] if a.skip_smoke else a.models.split(",")):
            r = res.setdefault(m, {})
            outs = []
            for i in range(2):
                rc, dt, so, se = cli_run(a.exe, TINYM[m], a.device, a.mode, pfile)
                open(os.path.join(OUT, f"cli_{m}_{i}.txt"), "w", encoding="utf-8").write(so + "\n--- stderr\n" + se)
                outs.append(gen_text(so))
                r[f"cli_run{i}_s"] = round(dt, 1)
                r[f"cli_run{i}_rc"] = rc
            r["cli_identical"] = outs[0] == outs[1] and bool(outs[0])
            r["cli_text"] = outs[0][:160]
            log = os.path.join(OUT, f"server_{m}.log")
            with qlib.Server(a.exe, a.device, a.mode, 8192, log, task="fl1b", model=TINYM[m]) as s:
                r["server_load_s"] = round(s.t_load, 1)
                t = time.time()
                a1 = s.complete(PROMPT, 48)["choices"][0]["text"]
                r["server_req_s"] = round(time.time() - t, 2)
                a2 = s.complete(PROMPT, 48)["choices"][0]["text"]
                st = s.complete(PROMPT, 48, stream=True)
                txt = ""
                for line in st:
                    line = line.decode("utf-8", "replace").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        ch = json.loads(line[6:]).get("choices") or []
                        if ch:
                            txt += ch[0].get("text", "")
                r["server_identical"] = a1 == a2 == txt and bool(a1)
                r["server_text"] = a1[:160]
            ok_all = ok_all and r["cli_identical"] and r["server_identical"] and r["cli_run0_rc"] == 0
            qlib.log(f"{m}: {r}")
        if a.gate:
            gexe = os.path.join(os.path.dirname(a.exe), "whirl-server-gate.exe")
            sexe = os.path.join(os.path.dirname(a.exe), "whirl-server.exe")
            runs = ([] if a.only_27b else [("tiny", TINYM[m]) for m in a.models.split(",")[:1]]) +                 ([("27b", REAL[a.models.split(",")[0]])] if a.compare_27b or a.only_27b else [])
            for tag, model in runs:
                work = os.path.join(OUT, f"gate_{tag}")
                cmd = [gexe, "--exe", sexe, "--kind", "whirl", "--model", model, "--suite", a.gate, "--port", "8197", "--work", work,
                       "--kv", "auto", "--server-env", f"WHIRL_MODE={a.mode}"] + ([] if a.ssd else ["--server-env", "WHIRL_KV_SSD_MODE=off"])
                if a.device == "8s":
                    cmd += ["--server-env", "WHIRL_DEVICE=8060s", "--server-env", "WHIRL_KV_RAM_MB=8192"]
                else:  # explicit: since 0.2.3 the default is the GPU with the most memory (the 8060S on the dev box)
                    cmd += ["--server-env", "WHIRL_DEVICE=r9700"]
                env = dict(os.environ, WHIRL_GATE_TEXT_ROOT=r"C:\ClaudeWorkspace\refsrc\llama.cpp")
                t = time.time()
                with open(os.path.join(OUT, f"gate_{tag}.log"), "w", encoding="utf-8", errors="replace") as fo:
                    rc = subprocess.run(cmd, env=env, stdout=fo, stderr=subprocess.STDOUT).returncode
                dt = time.time() - t
                tail = [l for l in open(os.path.join(OUT, f"gate_{tag}.log"), encoding="utf-8", errors="replace") if "== whirl-server-gate" in l]
                res[f"gate_{tag}"] = dict(rc=rc, seconds=round(dt, 1), summary=tail[-1].strip() if tail else "")
                qlib.log(f"gate {a.gate} on {tag}: exit {rc}, {dt:.0f} s, {tail[-1].strip() if tail else ''}")
                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", work], capture_output=True)
    json.dump(res, open(os.path.join(OUT, f"result_{a.device}_{time.strftime('%H%M%S')}.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
