# SPDX-License-Identifier: Apache-2.0
# Copied unchanged (code below) from progress/tools/anchor/abjudge.py (2026-10-24) by arch025-engine S0 so tools/bench/e8_judge.py
# (DESIGN 7 E8) does not depend on the progress folder. Keep the two in step (the progress copy is the anchor runners').
#!/usr/bin/env python3
# abjudge-order (2026-10-24): order-robust judge for interleaved A/B rounds (shared by asum25.py and ssum_paired.py).
# Prior art: progress\abjudge-order.PRIOR_ART.md (2x2 crossover AB/BA analysis; paired per-round differences).
#
# Design (anchor25.py / ab<step>s.py run each round as one PAIR, order reversed on even rounds: AB, BA, AB, ...):
#   per pair i:   d_i = 100 * (fin_i / base_i - 1)            (paired difference, %, same round)
#   sequence groups: A = pairs with base first, B = pairs with fin first. With a first-in-pair advantage theta (%):
#                 d(A) = delta - theta,  d(B) = delta + theta
#   delta_paired = (median d(A) + median d(B)) / 2            (order-adjusted change, %)
#   order_effect = (median d(B) - median d(A)) / 2            (first-in-pair minus second-in-pair, % of base)
#   noise        = max |d_i - median of its group|            (spread left after removing the order effect)
# Verdicts (SMALL = 2 % band, as asum25):
#   no pair                                   -> MISSING
#   output / cache checks differ              -> OUT     (unchanged rule: correctness, not speed)
#   only ONE sequence present (1 round, or rounds dropped by the overlap filter): delta and theta are aliased, so
#       |median d| < SMALL                    -> IN-small (no OUT is possible from one sequence)
#       else                                  -> INCONCLUSIVE (needs counterbalanced rounds), EXTEND if not final
#   both sequences:
#       |delta_paired| < SMALL                -> IN
#       |order_effect| >= |delta_paired|      -> INCONCLUSIVE (the sign of d flips with order: order effect dominates)
#       noise >= |delta_paired|               -> INCONCLUSIVE (spread as large as the change)
#       one pair per order group (2 pairs)    -> INCONCLUSIVE (spread not estimable; S7 8s ornith 32k tg: 2 pairs
#                                                read +2.7 % "OUT", all 5 rounds read +1.2 % IN)
#       else                                  -> OUT     (same sign in both orders, beyond the spread); EXTEND if not final
# Never OUT from a single round.  Symmetric (an improvement >= 2 % is also flagged OUT, as the old asum25 rule: FLAG =
# investigate, e.g. a wrong GPU picked).
import statistics

SMALL = 2.0


def first_of(times_base, times_fin, rnd):
    """which build ran first in a round: from the earliest timestamp of each build, else the anchor25 rule
    (odd round: base first, even round: fin first; variants given as base;fin)"""
    tb, tf = times_base.get(rnd), times_fin.get(rnd)
    if tb and tf and tb != tf:
        return "base" if tb < tf else "fin"
    return "base" if rnd % 2 else "fin"


def paired(base, fin, first, small=SMALL, final=True, checks_ok=True):
    """base, fin: {round: value}; first: {round: 'base'|'fin'}. Returns a dict (verdict + statistics)."""
    rounds = sorted(set(base) & set(fin))
    o = {"rounds": rounds, "n_pairs": len(rounds), "pair_d_pct": [], "first": [first.get(r) for r in rounds],
         "n_base_first": 0, "n_fin_first": 0, "delta_paired_pct": None, "order_effect_pct": None, "median_d_pct": None,
         "spread_d_pct": None, "noise_pct": None, "order_estimable": False, "verdict": "MISSING", "reason": "no pair"}
    if not rounds:
        return o
    d = {r: 100.0 * (fin[r] / base[r] - 1.0) for r in rounds}
    A = [d[r] for r in rounds if first.get(r) == "base"]
    B = [d[r] for r in rounds if first.get(r) == "fin"]
    allv = list(d.values())
    o.update(pair_d_pct=[round(d[r], 2) for r in rounds], n_base_first=len(A), n_fin_first=len(B),
             median_d_pct=round(statistics.median(allv), 2), spread_d_pct=[round(min(allv), 2), round(max(allv), 2)])
    if not checks_ok:
        o.update(verdict="OUT", reason="output / cache check differs between builds")
        return o
    if not A or not B:
        md = statistics.median(allv)
        o["noise_pct"] = round(max(abs(x - md) for x in allv), 2)
        if abs(md) < small:
            o.update(verdict="IN-small", reason=f"one sequence only ({len(allv)} pair(s)): |median d| < {small}%, order effect not estimable")
        else:
            o.update(verdict="INCONCLUSIVE" if final else "EXTEND",
                     reason=f"one sequence only ({len(allv)} pair(s)): change and order effect are aliased; needs counterbalanced rounds")
        return o
    mA, mB = statistics.median(A), statistics.median(B)
    dp, th = (mA + mB) / 2.0, (mB - mA) / 2.0
    noise = max([abs(x - mA) for x in A] + [abs(x - mB) for x in B])
    o.update(delta_paired_pct=round(dp, 2), order_effect_pct=round(th, 2), noise_pct=round(noise, 2), order_estimable=True)
    if abs(dp) < small:
        o.update(verdict="IN", reason=f"order-adjusted change {dp:+.2f}% within {small}%")
    elif abs(th) >= abs(dp):
        o.update(verdict="INCONCLUSIVE" if final else "EXTEND",
                 reason=f"order effect {th:+.2f}% >= change {dp:+.2f}%: sign flips with order; needs more counterbalanced rounds")
    elif len(A) < 2 and len(B) < 2:
        o.update(verdict="INCONCLUSIVE" if final else "EXTEND",
                 reason=f"change {dp:+.2f}% from one pair per order: spread not estimable; needs more counterbalanced rounds")
    elif noise >= abs(dp):
        o.update(verdict="INCONCLUSIVE" if final else "EXTEND",
                 reason=f"spread {noise:.2f}% >= change {dp:+.2f}%; needs more counterbalanced rounds")
    else:
        o.update(verdict="OUT" if final else "EXTEND",
                 reason=f"change {dp:+.2f}% in both orders (order effect {th:+.2f}%, spread {noise:.2f}%)")
    return o


def line(o):
    """the extra fields printed after the old ANCHOR / speed line (added, never replacing the old ones)"""
    f = lambda x: "-" if x is None else f"{x:+.2f}%"
    return (f"paired {f(o['delta_paired_pct'])} order_effect {f(o['order_effect_pct'])} median_d {f(o['median_d_pct'])} "
            f"spread_d {o['spread_d_pct']} noise {'-' if o['noise_pct'] is None else o['noise_pct']} "
            f"seq base-first {o['n_base_first']} fin-first {o['n_fin_first']} d {o['pair_d_pct']}")


def selftest():
    ok = True
    # pure order effect (first in pair +3 %), 4 rounds AB BA AB BA -> IN (never OUT)
    b = {1: 515, 2: 500, 3: 515, 4: 500}; f = {1: 500, 2: 515, 3: 500, 4: 515}
    fs = {1: "base", 2: "fin", 3: "base", 4: "fin"}
    ok &= paired(b, f, fs)["verdict"] == "IN"
    # S7 pattern, 3 rounds -> not OUT
    ok &= paired({1: 507.9, 2: 494.1, 3: 507.2}, {1: 491.9, 2: 506.7, 3: 496.4}, {1: "base", 2: "fin", 3: "base"})["verdict"] == "IN"
    # real -5 % plus a 3 % order effect -> OUT
    b = {1: 1030, 2: 1000, 3: 1030, 4: 1000}; f = {1: 950 * 1.0, 2: 950 * 1.03, 3: 950 * 1.0 + 1, 4: 950 * 1.03 - 1}
    r = paired(b, f, fs); ok &= r["verdict"] == "OUT" and abs(r["delta_paired_pct"] + 5) < 0.3 and abs(r["order_effect_pct"] - 3) < 0.3
    # order effect larger than the change -> INCONCLUSIVE
    ok &= paired({1: 1000, 2: 1000}, {1: 980, 2: 1080}, {1: "base", 2: "fin"})["verdict"] == "INCONCLUSIVE"  # d -2 / +8
    ok &= paired({1: 1000, 2: 1000}, {1: 920, 2: 980}, {1: "base", 2: "fin"})["verdict"] == "INCONCLUSIVE"  # 2 pairs: no OUT
    ok &= paired({1: 1000, 2: 1000}, {1: 930, 2: 1070}, {1: "base", 2: "fin"})["verdict"] == "IN"  # d -7 / +7: order only
    # spread larger than the change -> INCONCLUSIVE (d base-first 0 / -12, fin-first -3 / -5: change -5, spread 6)
    ok &= paired({r: 1000 for r in (1, 2, 3, 4)}, {1: 1000, 3: 880, 2: 970, 4: 950}, fs)["verdict"] == "INCONCLUSIVE"
    # single round -> never OUT
    ok &= paired({3: 507.2}, {3: 496.4}, {3: "base"})["verdict"] == "INCONCLUSIVE"
    ok &= paired({3: 1000}, {3: 995}, {3: "base"})["verdict"] == "IN-small"
    ok &= paired({3: 1000}, {3: 800}, {3: "base"}, final=False)["verdict"] == "EXTEND"
    ok &= paired({}, {}, {})["verdict"] == "MISSING"
    print("abjudge selftest", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
