#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""arch025-Q4: block / pass tests of tsa_check.py (no compiler needed). Run: python -I test_tsa_check.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsa_check as t  # noqa: E402

ERR = ("engine.cpp(219,5): error: writing variable 'stop_' requires holding mutex 'q_mutex_' exclusively "
       "[-Werror,-Wthread-safety-analysis]")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

CASES = []


def case(f):
    CASES.append(f)
    return f


@case
def pass_mutant_detected():
    ok, why = t.mutant_ok(0, "", 1, ERR, r"writing variable 'stop_' requires holding mutex 'q_mutex_'")
    assert ok, why


@case
def block_mutant_compiles():
    ok, why = t.mutant_ok(0, "", 0, "", r"stop_")
    assert not ok and "not detected" in why, why


@case
def block_control_broken():
    ok, why = t.mutant_ok(1, ERR, 1, ERR, r"stop_")
    assert not ok and "control" in why, why


@case
def block_other_error():
    # the mutant fails for a reason that is not the analysis (syntax error): not a detection
    ok, why = t.mutant_ok(0, "", 1, "engine.cpp(1,1): error: expected ';'", r"stop_")
    assert not ok and "without a thread-safety error" in why, why


@case
def block_wrong_diagnostic():
    ok, why = t.mutant_ok(0, "", 1, ERR, r"reading variable 'queue_'")
    assert not ok and "not with" in why, why


@case
def block_stale_mutation_text():
    for text in ("", "a lock a lock"):
        try:
            t.apply_edit(text, "a lock", "")
        except ValueError:
            continue
        raise AssertionError("apply_edit accepted text found 0 / 2 times")


@case
def pass_apply_edit_once():
    assert t.apply_edit("x lock y", "lock ", "") == "x y"


@case
def pass_verdict_clean():
    ok, why = t.verdict(0, 0, "[42/42] Linking CXX static library whirl_server_core.lib")
    assert ok, why


@case
def block_verdict_build_failed():
    ok, why = t.verdict(1, 0, ERR)
    assert not ok and "DONE rc=1" in why and "thread-safety error" in why, why


@case
def block_verdict_missing_files():
    ok, why = t.verdict(None, None, "")
    assert not ok and "DONE rc=None" in why and "MUTATE rc=None" in why, why


@case
def block_verdict_proof_failed():
    ok, why = t.verdict(0, 1, "")
    assert not ok and "MUTATE rc=1" in why, why


@case
def pass_mutations_apply_to_tree():
    # every mutation's text is in its source exactly once (a refactor that moves a lock must update MUTATIONS, not skip it)
    for name, rel, old, new, expect in t.MUTATIONS:
        text = open(os.path.join(REPO, rel), encoding="utf-8", newline="").read()
        t.apply_edit(text, old, new)


def main():
    bad = 0
    for f in CASES:
        try:
            f()
            print(f"ok   {f.__name__}")
        except Exception as ex:  # noqa: BLE001
            bad += 1
            print(f"FAIL {f.__name__}: {ex}")
    print(f"test_tsa_check: {len(CASES) - bad}/{len(CASES)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
