# SPDX-License-Identifier: Apache-2.0
"""arch024-S12: proves module_gate_test.ps1 goes red when release_gate.ps1's -Module code or modules.json is wrong
(block test of the table test). Edits each file in place and always restores it (finally). CPU only, about 3 min.

    python tools/gates/module_gate_mutate.py [--work DIR]      exit 0 = "mutations: all caught"
"""
import os
import subprocess
import sys
import tempfile

G = os.path.dirname(os.path.abspath(__file__))
WORK = sys.argv[sys.argv.index("--work") + 1] if "--work" in sys.argv else os.path.join(tempfile.gettempdir(), "whirl-module-gate-mut")
gate, mods = os.path.join(G, "release_gate.ps1"), os.path.join(G, "modules.json")
muts = [
    ("gate ignores an unknown module step", gate,
     "            else { $merr += \"module $mn`: step '$st' is not in this gate (-List) and not marked pending\" }",
     "            else { }"),
    ("gate: empty selection runs the full gate", gate,
     "    if (-not $full -and $msel.Count -eq 0 -and $Steps.Count -eq 0) {", "    if ($false) {"),
    ("gate: missing paths always tolerated", gate,
     "            else { $merr += \"module $mn`: none of its paths ($($globs -join ', ')) is in this tree and it has no lands_on\" }",
     "            else { }"),
    ("modules.json: ckpt_store loses srv_cache_balance", mods,
     '"srv_cache_balance", "srv_cache_precise", "srv_gpu_select"],\n   "gpus": {"golden_swift": 2, "golden_moe": 2, "srv_mt_precise": 2, "srv_mt_precise_moe": 2, "srv_cache_balance": 1,\n            "srv_cache_precise": 1, "srv_gpu_select": 2}',
     '"srv_cache_precise", "srv_gpu_select"],\n   "gpus": {"golden_swift": 2, "golden_moe": 2, "srv_mt_precise": 2, "srv_mt_precise_moe": 2, "srv_cache_balance": 1,\n            "srv_cache_precise": 1, "srv_gpu_select": 2}'),
    ("gate: explicit -Tag gate_* accepted with -Module (review A5)", gate,
     '    elseif ($Tag -notlike "module_*") {', '    elseif ($false) {'),
    ("gate: summary header without module= (review A5)", gate,
     '$mnote = if ($Module.Count -gt 0) {', '$mnote = if ($false) {'),
    ("gate: rdeps includer in no module does not force the full gate (review A4)", gate,
     '            if ($lone.Count -gt 0) {', '            if ($false) {'),
    ("gate: list callers ignored (review A4)", gate,
     '            foreach ($k in @($md.callers)) { if', '            foreach ($k in @()) { if'),
    ("gate: -Module without -ImpactRange / -NoMapSteps accepted (review K2)", gate,
     '    if (-not $ImpactRange -and -not $NoMapSteps) {', '    if ($false) {'),
    ("gate: impact.py's full gate ignored (review K2)", gate,
     '            } elseif ($mg -like "full gate*") {', '            } elseif ($false) {'),
    ("gate: impact.py's extra steps not added (review K2)", gate,
     '                if ($mg -match "-Steps (\\S+)") {', '                if ($false) {'),
    ("gate: no both-GPU hint for added steps (review K1)", gate,
     '    if ($addedGpu.Count) {', '    if ($false) {'),
    ("modules.json: protocol drops golden_swift", mods,
     '"steps": ["cpu_pins", "cpu_server", "srv_fs_swift", "srv_fs_moe", "golden_swift"]', '"steps": ["cpu_pins", "cpu_server", "srv_fs_swift", "srv_fs_moe"]'),
    ("modules.json: protocol drops cpu_pins (arch024-S11c F-3)", mods,
     '"steps": ["cpu_pins", "cpu_server", "srv_fs_swift", "srv_fs_moe", "golden_swift"]', '"steps": ["cpu_server", "srv_fs_swift", "srv_fs_moe", "golden_swift"]'),
]
fails = 0
for name, f, old, new in muts:
    raw = open(f, "rb").read()
    crlf = b"\r\n" in raw
    txt = raw.decode("utf-8")
    o, n = (old.replace("\n", "\r\n"), new.replace("\n", "\r\n")) if crlf else (old, new)
    if txt.count(o) != 1:
        print(f"SETUP {name}: pattern found {txt.count(o)} times")
        fails += 1
        continue
    try:
        open(f, "wb").write(txt.replace(o, n).encode("utf-8"))
        r = subprocess.run(["pwsh", "-NoProfile", "-File", os.path.join(G, "module_gate_test.ps1"), "-Work", WORK],
                           capture_output=True, text=True)
        last = [l for l in r.stdout.splitlines() if l.startswith(("FAIL", "module gate test"))]
        caught = r.returncode == 1
        print(f"{'CAUGHT ' if caught else 'SURVIVED'} {name}: exit {r.returncode}; {' | '.join(last[:3])}")
        fails += 0 if caught else 1
    finally:
        open(f, "wb").write(raw)
print("mutations: all caught" if fails == 0 else f"mutations: {fails} not caught")
sys.exit(1 if fails else 0)
