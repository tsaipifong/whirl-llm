#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# WHIRL v0252-rgpfix (2026-11-10): new output, not in the vendored hip-ep parser.
"""idle.csv - one row per GPU idle window: no wave of any dispatch alive on any shader engine.

The windows come from the raw occupancy start / end events of every engine (no wave -> dispatch attribution is
needed for the window itself); ``prev`` is the dispatch whose wave ended at the window start, ``next`` the one whose
wave started at its end (v2 attribution). ``prev_idx`` / ``next_idx`` are the ``idx`` of dispatches.csv.
``next_launch_v2_us`` is the earliest initiator time of the next dispatch over all engines: the part of the window
before it is "not initiated yet" (host / queue side), the rest "initiated, started late". Header only when the source
had no v2 decode.
"""

from __future__ import annotations

import csv

from ..model import Trace

FIELDS = [
    "start_us",
    "end_us",
    "dur_us",
    "prev_idx",
    "prev_kernel",
    "prev_pc",
    "next_idx",
    "next_kernel",
    "next_pc",
    "next_launch_v2_us",
]


def write(trace: Trace, path: str) -> None:
    idx = {id(d): i for i, d in enumerate(sorted(trace.dispatches, key=lambda x: x.ts_us))}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for x in trace.idle_windows or []:
            p, n = x.get("prev"), x.get("next")
            w.writerow(
                {
                    "start_us": round(x["start_us"], 4),
                    "end_us": round(x["end_us"], 4),
                    "dur_us": round(x["dur_us"], 4),
                    "prev_idx": idx[id(p)] if p is not None else "",
                    "prev_kernel": x["prev_kernel"],
                    "prev_pc": hex(x["prev_pc"]),
                    "next_idx": idx[id(n)] if n is not None else "",
                    "next_kernel": x["next_kernel"],
                    "next_pc": hex(x["next_pc"]),
                    "next_launch_v2_us": ""
                    if n is None or n.launch_v2_us is None
                    else round(n.launch_v2_us, 4),
                }
            )
