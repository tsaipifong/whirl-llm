r"""feat026-attn8s: kernel resource table from a clang -Rpass-analysis=kernel-resource-usage log.

Usage: python -I pf2_resources.py <cc.log> [name-regex]   (default regex: attn_prefill_wmma|attn_pf2_|fa_probe)
Prints one TSV row per kernel: name, VGPRs, VGPR spills, SGPRs, scratch bytes, occupancy (waves/SIMD), LDS bytes.
"""
import re
import sys


def parse(path):
    rows, cur = {}, None
    pat = re.compile(r"remark:\s+([A-Za-z ]+?)(?: \[[^\]]*\])?: (\S+) \[-Rpass")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            k, v = m.group(1), m.group(2)
            if k == "Function Name":
                cur = v
                rows[cur] = {}
            elif cur is not None:
                rows[cur][k] = v
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rx = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"attn_prefill_wmma|attn_pf2_|fa_probe")
    rows = parse(sys.argv[1])
    print("kernel\tVGPRs\tVGPR_spill\tSGPRs\tscratch_B_per_lane\twaves_per_SIMD\tLDS_B")
    for name in sorted(rows):
        if not rx.search(name):
            continue
        r = rows[name]
        print("\t".join([name] + [r.get(k, "?") for k in ("VGPRs", "VGPRs Spill", "TotalSGPRs", "ScratchSize", "Occupancy", "LDS Size")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
