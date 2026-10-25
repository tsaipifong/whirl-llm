"""Paired statistics for the task-accuracy eval (feat027-taskacc). Pure stdlib.

A = the reference configuration (normally precise), B = the candidate (balance, fast, an item on/off).
Pairs are keyed by (task_id, seed); with several seeds per task the bootstrap resamples TASKS
(cluster bootstrap), so correlated samples of one task are not counted as independent.

  acc            pass@1 = mean(passed) over all samples (= the unbiased pass@1 estimator for n samples)
  recovery       acc_B / acc_A (Kurtic et al. 2024, Neural Magic model cards)
  McNemar        exact binomial test on the discordant pairs b = A pass & B fail, c = A fail & B pass:
                 two-sided p = min(1, 2 * P(X <= min(b, c))), one-sided "B worse" p = P(X >= b), X ~ Bin(b + c, 1/2)
                 (as itemD/qa_kv.py and hipfire certify/coherence.py mcnemar_worse)
  CI             paired percentile bootstrap (default 10000 resamples, fixed seed) of recovery and of acc_B - acc_A
  verdicts       fast  : recovery >= 0.95 is the owner bar (task-accuracy recovery >= 95% of precise).
                         PASS when the bootstrap lower bound >= 0.95, FAIL when the upper bound < 0.95,
                         else INCONCLUSIVE (point estimate given; more tasks or seeds needed).
                 balance: there is no owner number; the report gives the one-sided McNemar p and the recovery CI.
                         "no significant loss" = one-sided p >= alpha; the numeric balance bar is owner policy.
"""
import math
import random


def binom_tail_ge(k, n):
    """P(X >= k), X ~ Bin(n, 1/2)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def mcnemar(b, c):
    """b = A pass / B fail, c = A fail / B pass. Returns (p_two_sided, p_one_sided_B_worse)."""
    n = b + c
    if n == 0:
        return 1.0, 1.0
    p2 = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(b, c) + 1)) / 2 ** n)
    return p2, binom_tail_ge(b, n)


def _key(r):
    return (r["task_id"], r.get("seed", 0))


def pair(res_a, res_b):
    """res_*: lists of sample records {task_id, seed, passed, ...}. Returns list of (key, a_pass, b_pass, a_rec, b_rec)."""
    ma = {_key(r): r for r in res_a}
    mb = {_key(r): r for r in res_b}
    keys = sorted(set(ma) & set(mb))
    return [(k, bool(ma[k]["passed"]), bool(mb[k]["passed"]), ma[k], mb[k]) for k in keys], \
        sorted(set(ma) - set(mb)), sorted(set(mb) - set(ma))


def bootstrap(pairs, n_boot=10000, seed=12345):
    """Cluster bootstrap over task ids. Returns dict of percentile CIs (2.5 / 97.5) for recovery and diff."""
    by_task = {}
    for k, a, b, *_ in pairs:
        t = by_task.setdefault(k[0], [0, 0, 0])
        t[0] += a
        t[1] += b
        t[2] += 1
    tasks = list(by_task.values())
    if not tasks:
        return {}
    rng = random.Random(seed)
    rec, dif = [], []
    nt = len(tasks)
    for _ in range(n_boot):
        sa = sb = sn = 0
        for _ in range(nt):
            t = tasks[rng.randrange(nt)]
            sa += t[0]
            sb += t[1]
            sn += t[2]
        dif.append((sb - sa) / sn)
        rec.append(sb / sa if sa else float("nan"))
    rec = sorted(x for x in rec if x == x)
    dif.sort()

    def q(v, p):
        if not v:
            return float("nan")
        i = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
        return v[i]
    return {"recovery_ci95": [q(rec, 0.025), q(rec, 0.975)], "diff_ci95": [q(dif, 0.025), q(dif, 0.975)],
            "n_boot": n_boot, "boot_seed": seed, "clusters": nt}


def summarize(pairs, n_boot=10000, alpha=0.05, fast_bar=0.95):
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    pa = sum(a for _, a, _, *_ in pairs)
    pb = sum(b for _, _, b, *_ in pairs)
    b = sum(1 for _, a, bb, *_ in pairs if a and not bb)
    c = sum(1 for _, a, bb, *_ in pairs if (not a) and bb)
    p2, p1 = mcnemar(b, c)
    acc_a, acc_b = pa / n, pb / n
    out = {"n": n, "pass_a": pa, "pass_b": pb, "acc_a": acc_a, "acc_b": acc_b, "diff": acc_b - acc_a,
           "recovery": (acc_b / acc_a) if acc_a else float("nan"),
           "a_only": b, "b_only": c, "mcnemar_p_two_sided": p2, "mcnemar_p_b_worse": p1}
    ident = [1 for _, _, _, ra, rb in pairs if ra.get("content") is not None and ra.get("content") == rb.get("content")]
    out["identical_replies"] = len(ident)
    if n_boot:
        out.update(bootstrap(pairs, n_boot=n_boot))
        lo, hi = out["recovery_ci95"]
        if lo >= fast_bar:
            v = "PASS"
        elif hi < fast_bar:
            v = "FAIL"
        else:
            v = "INCONCLUSIVE"
        out["fast_verdict"] = f"{v} (bar recovery >= {fast_bar:.2f}; CI95 [{lo:.3f}, {hi:.3f}], point {out['recovery']:.3f})"
    out["balance_note"] = ("no significant loss" if p1 >= alpha else "SIGNIFICANT LOSS") + \
        f" (one-sided McNemar p = {p1:.4f}, alpha {alpha}); numeric balance bar = owner policy"
    return out


def min_detectable_drop(n, acc_a, discordant_rate=None, alpha=0.05):
    """Rough planning number [estimate]: the smallest one-sided-significant b - c with d discordant pairs.
    d defaults to 2 * acc_a * (1 - acc_a) * n * 0.5 (half of the independent-errors flip rate).
    Returns (d, b_needed, drop_in_points) — the accuracy drop (b - c) / n that McNemar would flag."""
    if discordant_rate is None:
        discordant_rate = acc_a * (1 - acc_a)
    d = max(1, int(round(discordant_rate * n)))
    for b in range(d + 1):
        if binom_tail_ge(b, d) < alpha:
            c = d - b
            return d, b, (b - c) / n
    return d, None, None
