#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""arch025-S0a include check: whirl_server_core (cmake/server.cmake) must not reach include/whirl/model.h (M3 makes it
block). Thin entry point of tools/gates/pending_scan.py rule "include" (pending list include_pending.txt).

  python -I tools/gates/include_check.py [--mode report|block] [--root R] [--pending F]
  python -I tools/gates/include_check.py --selftest
"""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location("pending_scan", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pending_scan.py"))
_ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ps)

if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(_ps.main(args if "--selftest" in args else ["include"] + args))
