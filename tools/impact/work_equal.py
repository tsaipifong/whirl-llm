# SPDX-License-Identifier: Apache-2.0
"""Work-equivalence check A vs B (measure-tools P5, DESIGN §B3; ADR-0003 decision 3, owner sign-off 「核准!」).

A stored speed cell may be carried over from build A to build B only when, for the same GPU:
  1. code objects: every AMDGPU code object embedded in each exe (whirl.exe, whirl-server.exe, ...) is byte-identical
     (per object: sha256 of the whole object and of each ELF section); an object that differs only in the per-TU
     __hip_cuid_<hex> symbol clang derives from the source path, and in that symbol's exact consequences, counts as
     equal: code_objects "identical-except-cuid" (cuid_equivalent; a cuid of another length too unless --strict-cuid-length),
  2. launch sequence: the launch tables of the deterministic CLI runs (golden single / pair / ngram, seqtest, selftest;
     server runs only with --server and the ignore list) are identical: launch count, sequence hash, kernel/geometry rows
     and, when both logs carry them, the argument digests and argument-sequence hash (tools/impact/launch_table.py),
  3. golden: the golden outputs are bitwise identical (both sides OK against the same reference set, and the output files'
     sha256 equal when both output folders are given).
Missing evidence never passes: the verdict is EQUIVALENT (exit 0), NOT_EQUIVALENT (1) or INCOMPLETE (2); usage error 3.

Every piece of evidence is bound to the exe it claims to be about (REVIEW-R5-024 C1): both exe sha256 are in the report;
launch logs are bound through the exe path in their command line (file sha256 == the side's exe and the exe unchanged since
the log was written) or through results.db work_traces.exe_sha256 ("db:<sha>"); golden summaries through their
"golden: exe ... sha256" lines. The host code is compared too (PE sections, build-stamp fields located and separated, the
code-object ranges excluded) and reported as identical / stamp_only / differs with the differing strings: ADR-0003 does not
make host equality a condition, so it is information for release_check (P7), which may require it.

    python tools/impact/work_equal.py compare --a-exe A\\whirl.exe [--a-exe A\\whirl-server.exe ...] --b-exe B\\whirl.exe ...
        [--a-launch DIR|TABLE|db:<sha> --b-launch ...] [--db results.db] [--server] [--ignore RX] [--allow-unbound-launch]
        [--a-golden FILE ... --b-golden FILE ...] [--a-golden-out DIR --b-golden-out DIR]
        [--repo R --a-commit C --b-commit C] [-o report.json]
    python tools/impact/work_equal.py scan EXE          # code objects + PE sections of one exe (JSON)

A launch DIR may hold launch_<pid>.txt / launch_<pid>_<start>[_<n>].txt files directly or in step sub-folders (release_gate -LaunchLog layout
<tag>\\<step>\\launch_*.txt); runs are keyed "<step>/<normalized command>".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_table  # noqa: E402

FORMAT = "whirl-work-equal/1"
EM_AMDGPU = 224
# EF_AMDGPU_MACH (low byte of e_flags), the targets WHIRL builds and their neighbours (llvm/BinaryFormat/ELF.h)
MACH = {0x41: "gfx1100", 0x42: "gfx1101", 0x43: "gfx1102", 0x44: "gfx1103", 0x48: "gfx1200", 0x4A: "gfx1151",
        0x4C: "gfx1150", 0x4E: "gfx1201", 0x30: "gfx1030"}


GPU_ALIASES = {"r9700": ("r9700", "r97", "gfx1201", "pci:71:0"), "8060s": ("8060s", "s8", "8s", "gfx1151", "pci:196:0")}


def norm_gpu(g):
    """r9700 / gfx1201 / pci:71:0 / R9700 -> "r9700"; 8060s / gfx1151 / pci:196:0 -> "8060s" (as resultsdb.norm_gpu, lower case)"""
    s = (g or "").strip().lower()
    for canon, al in GPU_ALIASES.items():
        if s and (s == canon or s in al):
            return canon
    return s


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- exe scan
def amdgpu_objects(d):
    """Every embedded 64-bit AMDGPU ELF (as scratch\\s16\\cosect.py: e_machine 224, object = header .. end of the section
    header table). Returns [{index, offset, size, sha256, target, sections {name: sha256}}]."""
    res = []
    k = d.find(b"\x7fELF")
    while k >= 0:
        try:
            if d[k + 4] == 2 and struct.unpack_from("<H", d, k + 18)[0] == EM_AMDGPU:
                shoff = struct.unpack_from("<Q", d, k + 0x28)[0]
                flags = struct.unpack_from("<I", d, k + 0x30)[0]
                shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, k + 0x3A)
                end = shoff + shentsize * shnum
                if 0 < end < 1 << 31 and k + end <= len(d) and shentsize == 64 and shstrndx < shnum:
                    e = d[k:k + end]
                    secs = [struct.unpack_from("<IIQQQQ", e, shoff + s * shentsize) for s in range(shnum)]
                    so = secs[shstrndx][4]
                    named = {}
                    for nm, typ, _fl, _addr, off, sz in secs:
                        n = e[so + nm:e.index(b"\0", so + nm)].decode("ascii", "replace")
                        body = b"" if typ == 8 else e[off:off + sz]  # SHT_NOBITS has no bytes
                        named[n] = sha(body) if n not in named else sha(bytes.fromhex(named[n]) + body)
                    res.append({"index": len(res), "offset": k, "size": end, "sha256": sha(e),
                                "target": MACH.get(flags & 0xFF, f"mach_0x{flags & 0xFF:02x}"), "sections": named})
                    k = d.find(b"\x7fELF", k + end)
                    continue
        except (struct.error, ValueError, IndexError):
            pass
        k = d.find(b"\x7fELF", k + 1)
    return res


# ---------------------------------------------------------------- __hip_cuid (arch025-weq)
# clang emits one external i8 global "__hip_cuid_<hex>" per HIP TU (CodeGenModule::Release); with the default
# -fuse-cuid=hash the hex is utohexstr(MD5(input path + driver args).low()), so two worktrees always differ there, and
# utohexstr drops leading zeros (15 hex digits about once in 16 TUs). Consequences inside one code object: the name in
# .dynstr / .strtab, the .dynsym order (lld sorts .dynsym by .gnu.hash bucket), .gnu.hash / .hash contents, relocation
# symbol indices; with a length change also .dynstr / .strtab sh_size, DT_STRSZ and later st_name offsets.
# cuid_equivalent() accepts exactly those consequences and nothing else (rule: s15-blast §2 / PLAN I1, arch025:S0b-runners).
CUID_NAME = re.compile(rb"__hip_cuid_[0-9a-f]{1,16}")
CUID_STR = {".dynstr", ".strtab"}
CUID_SYM = {".dynsym", ".symtab"}
CUID_HASH = {".hash", ".gnu.hash"}
CUID_SECTIONS = CUID_STR | CUID_SYM | CUID_HASH | {".dynamic", ".rela.dyn"}
DT_STRSZ = 10


def _elf_sections(x):
    """[(name, (sh_name, type, flags, addr, offset, size, link, info, align, entsize))] of one ELF64 object"""
    shoff = struct.unpack_from("<Q", x, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", x, 0x3A)
    if shentsize != 64 or shstrndx >= shnum:
        raise ValueError("section header table")
    hs = [struct.unpack_from("<IIQQQQIIQQ", x, shoff + i * 64) for i in range(shnum)]
    so = hs[shstrndx][4]
    return [(x[so + h[0]:x.index(b"\0", so + h[0])].decode("ascii", "replace"), h) for h in hs]


def _gnu_hash(n):
    h = 5381
    for c in n:
        h = (h * 33 + c) & 0xFFFFFFFF
    return h


def _sysv_hash(n):
    h = 0
    for c in n:
        h = ((h << 4) + c) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


def _hash_ok(x, kind, off, size, names):
    """Is this .hash / .gnu.hash section exactly the table of its own .dynsym (names[i] = name of index i)? Strict, so
    the table is fully determined by the symbol names: .hash every bucket chain holds exactly the symbols of that bucket,
    each once (only the order inside a chain is free); .gnu.hash the bloom words are exactly the bits of the hashed
    symbols, .dynsym from symoffset on is sorted by bucket, each bucket starts at its first symbol (0 when empty) and the
    chain words are (hash & ~1) | end-of-bucket. The section has no bytes beyond the table."""
    b = x[off:off + size]
    if kind == ".hash":
        nb, nc = struct.unpack_from("<II", b, 0)
        if nc != len(names) or 8 + 4 * (nb + nc) != len(b) or not nb:
            return False
        bk = struct.unpack_from(f"<{nb}I", b, 8)
        ch = struct.unpack_from(f"<{nc}I", b, 8 + 4 * nb)
        want = {}
        for i in range(1, nc):
            want.setdefault(_sysv_hash(names[i]) % nb, set()).add(i)
        for q in range(nb):
            seen, j = [], bk[q]
            while j:
                if j >= nc or j in seen:
                    return False
                seen.append(j); j = ch[j]
            if set(seen) != want.get(q, set()) or len(seen) != len(want.get(q, ())):
                return False
        return ch[0] == 0
    nb, symoff, bsz, shift = struct.unpack_from("<IIII", b, 0)
    n = len(names)
    if not nb or not bsz or not 1 <= symoff <= n or 16 + 8 * bsz + 4 * nb + 4 * (n - symoff) != len(b):
        return False
    bloom = struct.unpack_from(f"<{bsz}Q", b, 16)
    bk = struct.unpack_from(f"<{nb}I", b, 16 + 8 * bsz)
    ch = struct.unpack_from(f"<{n - symoff}I", b, 16 + 8 * bsz + 4 * nb)
    hs = [_gnu_hash(names[i]) for i in range(symoff, n)]
    wb = [0] * bsz
    for h in hs:
        wb[(h // 64) % bsz] |= (1 << (h % 64)) | (1 << ((h >> shift) % 64))
    if tuple(wb) != bloom:
        return False
    qs = [h % nb for h in hs]
    if qs != sorted(qs):
        return False
    first = {}
    for k, q in enumerate(qs):
        first.setdefault(q, symoff + k)
    if any(bk[q] != first.get(q, 0) for q in range(nb)):
        return False
    return all(ch[k] == (hs[k] & ~1) | (1 if k == len(qs) - 1 or qs[k + 1] != qs[k] else 0) for k in range(len(qs)))


def cuid_equivalent(xa, xb, allow_length=True):
    """Are two AMDGPU code objects identical except for the __hip_cuid_<hex> symbol of their TU and its consequences?
    -> (ok, detail). Required: same size, ELF and program headers and every byte outside the symbol sections identical,
    every other section (.text, .rodata, .note, .shstrtab, ...) byte-identical with the same header; the string tables
    identical string by string except whole strings __hip_cuid_<1-16 hex> that map one-to-one; .symtab entry by entry
    and .dynsym as a set equal after that renaming; .rela.dyn equal by symbol name; .dynamic equal except DT_STRSZ;
    .hash / .gnu.hash each valid for its own .dynsym. A cuid of another length (sh_size / DT_STRSZ / st_name shifts)
    passes only with allow_length (default: different)."""
    try:
        if len(xa) != len(xb):
            return False, "object size differs"
        sa, sb = _elf_sections(xa), _elf_sections(xb)
        if [n for n, _ in sa] != [n for n, _ in sb]:
            return False, "section names / order differ"
        phoff, phnum = struct.unpack_from("<Q", xa, 0x20)[0], struct.unpack_from("<H", xa, 0x38)[0]
        if xa[:64] != xb[:64] or xa[phoff:phoff + 56 * phnum] != xb[phoff:phoff + 56 * phnum]:
            return False, "ELF or program header differs"
        if len({n for n, _ in sa}) != len(sa):
            return False, "duplicate section names"
        shoff = struct.unpack_from("<Q", xa, 0x28)[0]
        cov = [(0, 64), (phoff, 56 * phnum), (shoff, 64 * len(sa))]
        diff_secs = []
        for (n, ha), (_, hb) in zip(sa, sb):
            if ha[1] != 8:  # SHT_NOBITS occupies no bytes
                cov.append((ha[4], max(ha[5], hb[5])))
            same_hdr = ha == hb
            same_body = ha[1] == 8 or xa[ha[4]:ha[4] + ha[5]] == xb[hb[4]:hb[4] + hb[5]]
            if same_hdr and same_body:
                continue
            if n not in CUID_SECTIONS:
                return False, f"section {n or '(null)'} differs"
            if ha[:5] + ha[6:] != hb[:5] + hb[6:] or (ha[5] != hb[5] and n not in CUID_STR):
                return False, f"section header of {n} differs"
            diff_secs.append(n)
        pos = 0  # the bytes no header or section covers (alignment padding) must be identical too
        for o, n in sorted(cov) + [(len(xa), 0)]:
            if o > pos and xa[pos:o] != xb[pos:o]:
                return False, "bytes outside the sections differ"
            pos = max(pos, o + n)
        if not diff_secs:
            return True, "identical"
        hdr = {n: (ha, hb) for (n, ha), (_, hb) in zip(sa, sb)}
        # string tables: string by string; the only differing strings are whole cuid names, mapped one-to-one
        cmap, delta = {}, {}
        for n in CUID_STR & set(hdr):
            ha, hb = hdr[n]
            la = xa[ha[4]:ha[4] + ha[5]].split(b"\0")
            lb = xb[hb[4]:hb[4] + hb[5]].split(b"\0")
            if len(la) != len(lb):
                return False, f"{n}: string count differs"
            for p, q in zip(la, lb):
                if p == q:
                    continue
                if not (CUID_NAME.fullmatch(p) and CUID_NAME.fullmatch(q)):
                    return False, f"{n}: non-cuid string differs ({p[:40]!r} vs {q[:40]!r})"
                if cmap.setdefault(p, q) != q:
                    return False, f"{n}: cuid {p!r} maps to two names"
                if len(p) != len(q) and not allow_length:
                    return False, f"cuid length differs ({p.decode()} {len(p)} vs {q.decode()} {len(q)} bytes)"
            delta[n] = hb[5] - ha[5]
            if delta[n] != sum(len(q) - len(p) for p, q in zip(la, lb)):
                return False, f"{n}: size change not explained by the cuid"
        if len(set(cmap.values())) != len(cmap):
            return False, "two cuids map to one name"
        if not cmap:
            return False, "symbol sections differ but no cuid string differs"
        ren = lambda s: cmap.get(s, s)
        idx = {i: n for i, (n, _) in enumerate(sa)}

        def syms(x, h, sec):
            st = sec[h[6]][1]
            if st[1] != 3 or h[9] != 24:
                raise ValueError("symbol table link / entsize")
            out = []
            for k in range(h[5] // 24):
                nm, info, oth, shn, val, sz = struct.unpack_from("<IBBHQQ", x, h[4] + 24 * k)
                out.append((x[st[4] + nm:x.index(b"\0", st[4] + nm)], info, oth, shn, val, sz))
            return out
        dyn_names = {}
        for n in CUID_SYM & set(hdr):
            ha, hb = hdr[n]
            ya = [(ren(e[0]),) + e[1:] for e in syms(xa, ha, sa)]
            yb = syms(xb, hb, sb)
            if (ya != yb) if n == ".symtab" else (ya[:1] != yb[:1] or sorted(ya) != sorted(yb)):
                return False, f"{n}: symbols differ beyond the cuid"
            if n == ".dynsym":
                dyn_names = {"a": [e[0] for e in syms(xa, ha, sa)], "b": [e[0] for e in yb]}
        for n in CUID_HASH & set(hdr):
            if n in diff_secs:
                if not dyn_names:
                    return False, f"{n} differs without a .dynsym"
                ha, hb = hdr[n]
                if idx.get(ha[6]) != ".dynsym":
                    return False, f"{n}: not linked to .dynsym"
                if not (_hash_ok(xa, n, ha[4], ha[5], dyn_names["a"]) and _hash_ok(xb, n, hb[4], hb[5], dyn_names["b"])):
                    return False, f"{n}: does not index its .dynsym"
        if ".rela.dyn" in diff_secs:
            ha, hb = hdr[".rela.dyn"]
            if ha[9] != 24 or idx.get(ha[6]) != ".dynsym" or not dyn_names:
                return False, ".rela.dyn: entsize / link"
            ra = [struct.unpack_from("<QQq", xa, ha[4] + 24 * k) for k in range(ha[5] // 24)]
            rb = [struct.unpack_from("<QQq", xb, hb[4] + 24 * k) for k in range(hb[5] // 24)]
            na, nb_ = dyn_names["a"], dyn_names["b"]
            if any(len(na) <= i >> 32 for _, i, _ in ra) or any(len(nb_) <= i >> 32 for _, i, _ in rb):
                return False, ".rela.dyn: symbol index out of range"
            if [(o, i & 0xFFFFFFFF, ren(na[i >> 32]), ad) for o, i, ad in ra] != [(o, i & 0xFFFFFFFF, nb_[i >> 32], ad) for o, i, ad in rb]:
                return False, ".rela.dyn: relocations differ beyond the cuid symbol index"
        if delta.get(".dynstr") and ".dynamic" not in diff_secs:
            return False, ".dynstr size changed but DT_STRSZ did not"
        if ".dynamic" in diff_secs:
            ha, hb = hdr[".dynamic"]
            da_ = [struct.unpack_from("<qQ", xa, ha[4] + 16 * k) for k in range(ha[5] // 16)]
            db_ = [struct.unpack_from("<qQ", xb, hb[4] + 16 * k) for k in range(hb[5] // 16)]
            for p, q in zip(da_, db_):
                if p != q and not (p[0] == q[0] == DT_STRSZ and q[1] - p[1] == delta.get(".dynstr", 0)):
                    return False, f".dynamic: tag {p[0]} differs"
        return True, "cuid " + ", ".join(f"{p.decode()}->{q.decode()}" for p, q in sorted(cmap.items()))
    except (struct.error, ValueError, IndexError) as e:
        return False, f"unparsable ({type(e).__name__}: {e})"


def pe_layout(d):
    """PE sections [{name, offset, size, vaddr, sha256}] and the build-stamp byte ranges [(offset, length, what)]:
    COFF TimeDateStamp, optional-header CheckSum, each debug directory entry's TimeDateStamp, the CodeView RSDS GUID + age,
    and the data of REPRO entries (msvcpp-normalize-pe: the fields MSVC varies per link). Not a PE -> None."""
    try:
        if d[:2] != b"MZ":
            return None
        pe = struct.unpack_from("<I", d, 0x3C)[0]
        if d[pe:pe + 4] != b"PE\0\0":
            return None
        nsec = struct.unpack_from("<H", d, pe + 6)[0]
        optsz = struct.unpack_from("<H", d, pe + 20)[0]
        opt = pe + 24
        magic = struct.unpack_from("<H", d, opt)[0]
        dd = opt + (112 if magic == 0x20B else 96)
        secs = []
        so = opt + optsz
        for i in range(nsec):
            nm, vs, va, rs, rp = struct.unpack_from("<8sIIII", d, so + 40 * i)
            secs.append({"name": nm.rstrip(b"\0").decode("ascii", "replace"), "offset": rp, "size": rs, "vaddr": va, "vsize": vs})
        first = min([s["offset"] for s in secs if s["size"]] or [len(d)])
        secs.insert(0, {"name": "(headers)", "offset": 0, "size": first, "vaddr": 0, "vsize": first})
        for s in secs:
            s["sha256"] = sha(d[s["offset"]:s["offset"] + s["size"]])
        stamps = [(pe + 8, 4, "coff_timestamp"), (opt + 64, 4, "checksum")]

        def rva2off(rva):
            for s in secs[1:]:
                if s["vaddr"] <= rva < s["vaddr"] + max(s["vsize"], s["size"]):
                    return s["offset"] + rva - s["vaddr"]
            return None
        nrva = struct.unpack_from("<I", d, dd - 4)[0]
        if nrva > 6:
            drva, dsz = struct.unpack_from("<II", d, dd + 6 * 8)
            o = rva2off(drva) if drva else None
            if o is not None:
                for i in range(dsz // 28):
                    e = o + 28 * i
                    _ch, _ts, _maj, _min, typ, sz, _addr, ptr = struct.unpack_from("<IIHHIIII", d, e)
                    stamps.append((e + 4, 4, "debug_timestamp"))
                    if typ == 2 and d[ptr:ptr + 4] == b"RSDS":
                        stamps.append((ptr + 4, 20, "codeview_guid_age"))
                    elif typ == 16 and sz:
                        stamps.append((ptr, sz, "repro_hash"))
        return {"sections": secs, "stamps": stamps}
    except (struct.error, IndexError):
        return None


def scan_exe(path):
    with open(path, "rb") as f:
        d = f.read()
    return {"path": os.path.abspath(path), "sha256": sha(d), "size": len(d), "objects": amdgpu_objects(d), "pe": pe_layout(d)}


def _strings(b, minlen=6):
    """printable ASCII runs and UTF-16LE runs (version resources are UTF-16)"""
    out = set(re.findall(rb"[ -~]{%d,}" % minlen, b))
    out |= {x.decode("utf-16-le").encode("ascii", "replace") for x in re.findall(rb"(?:[ -~]\x00){%d,}" % minlen, b)}
    return out


CHUNK = 4096


def _cut(x, ranges):
    """x with the byte ranges [(off, len)] replaced by NUL (code objects: compared on their own)."""
    b = bytearray(x)
    for o, n in ranges:
        lo, hi = max(o, 0), min(o + n, len(b))
        if lo < hi:
            b[lo:hi] = bytes(hi - lo)
    return bytes(b)


def _section_diff(xa, xb, code, stamp):
    """Differing bytes of one PE section pair outside the code-object ranges, split into build-stamp bytes and the rest,
    plus the printable strings around the differing bytes (only in A / only in B). Chunked: equal chunks are skipped."""
    ca, cb = _cut(xa, code), _cut(xb, code)
    if len(ca) != len(cb):
        sa, sb = _strings(ca), _strings(cb)
        return max(len(ca), len(cb)), 0, sa, sb, "size differs"
    st = set()
    for o, n in stamp:
        st.update(range(max(o, 0), min(o + n, len(ca))))
    nd = ns = 0
    sa, sb = set(), set()
    for k in range(0, len(ca), CHUNK):
        x, y = ca[k:k + CHUNK], cb[k:k + CHUNK]
        if x == y:
            continue
        hit = False
        for i in range(len(x)):
            if x[i] != y[i]:
                if k + i in st:
                    ns += 1
                else:
                    nd += 1; hit = True
        if hit:
            lo, hi = max(k - 256, 0), k + CHUNK + 256
            sa |= _strings(ca[lo:hi], 4); sb |= _strings(cb[lo:hi], 4)
    return nd, ns, sa, sb, ""


def compare_exe(a_path, b_path, data=None, allow_cuid_length=True):
    """Code objects and host code of one exe pair. code_objects.result: identical / identical-except-cuid (every differing
    object passes cuid_equivalent: work-equivalent for ADR-0003) / different / missing."""
    da = open(a_path, "rb").read()
    db_ = open(b_path, "rb").read()
    oa, ob = amdgpu_objects(da), amdgpu_objects(db_)
    objs = []
    for i in range(max(len(oa), len(ob))):
        x, y = (oa[i] if i < len(oa) else None), (ob[i] if i < len(ob) else None)
        if x is None or y is None:
            objs.append({"index": i, "same": False, "only": "B" if x is None else "A",
                         "target": (x or y)["target"], "size_a": x and x["size"], "size_b": y and y["size"]})
            continue
        dif = sorted(n for n in set(x["sections"]) | set(y["sections"]) if x["sections"].get(n) != y["sections"].get(n))
        objs.append({"index": i, "target": x["target"] if x["target"] == y["target"] else f"{x['target']}/{y['target']}",
                     "same": x["sha256"] == y["sha256"], "sha_a": x["sha256"], "sha_b": y["sha256"], "size_a": x["size"],
                     "size_b": y["size"], "sections": len(x["sections"]), "differing_sections": dif,
                     "header_only": x["sha256"] != y["sha256"] and not dif})
        if objs[-1]["same"] is False and set(dif) <= CUID_SECTIONS:
            okc, why = cuid_equivalent(da[x["offset"]:x["offset"] + x["size"]], db_[y["offset"]:y["offset"] + y["size"]], allow_cuid_length)
            objs[-1].update(cuid_only=okc and why != "identical", cuid=why)  # sha differs: "identical" would be a parser gap
    if not oa and not ob:
        co = "missing"  # nothing to compare: never "identical" by default
    elif len(oa) == len(ob) and all(o["same"] for o in objs):
        co = "identical"
    elif len(oa) == len(ob) and all(o["same"] or o.get("cuid_only") for o in objs):
        co = "identical-except-cuid"
    else:
        co = "different"
    # host: PE sections with the code-object byte ranges and the build-stamp fields left out of the byte count
    host = {"result": "identical" if da == db_ else None, "sections": [], "stamp_bytes": 0, "unexplained_bytes": 0,
            "strings_only_a": [], "strings_only_b": [], "text_same": None}
    pa, pb = pe_layout(da), pe_layout(db_)
    if da != db_:
        if pa is None or pb is None:
            host.update(result="differs", note="not a PE file on one side: whole-file comparison only")
        else:
            skip_a = [(o["offset"], o["size"]) for o in oa]
            skip_b = [(o["offset"], o["size"]) for o in ob]
            st_a = [(o, n) for o, n, _ in pa["stamps"]]
            sb_names = {s["name"]: s for s in pb["sections"]}
            sa_only, sb_only = set(), set()
            for s in pa["sections"]:
                t = sb_names.get(s["name"])
                e = {"name": s["name"], "a_sha": s["sha256"], "b_sha": t and t["sha256"], "size_a": s["size"], "size_b": t and t["size"]}
                if t is None:
                    e.update(same=False, only="A"); host["sections"].append(e); host["unexplained_bytes"] += s["size"]; continue
                e["same"] = s["sha256"] == t["sha256"]
                if not e["same"]:
                    xa = da[s["offset"]:s["offset"] + s["size"]]
                    xb = db_[t["offset"]:t["offset"] + t["size"]]
                    code = [(o - s["offset"], n) for o, n in skip_a] + [(o - t["offset"], n) for o, n in skip_b]
                    stamp = [(o - s["offset"], n) for o, n in st_a]
                    nd, ns, ssa, ssb, note = _section_diff(xa, xb, code, stamp)
                    e.update(differing_bytes=nd, stamp_bytes=ns)
                    if note:
                        e["note"] = note
                    host["stamp_bytes"] += ns
                    host["unexplained_bytes"] += nd
                    sa_only |= ssa - ssb; sb_only |= ssb - ssa
                else:
                    e.update(differing_bytes=0, stamp_bytes=0)
                host["sections"].append(e)
            for t in pb["sections"]:
                if t["name"] not in {s["name"] for s in pa["sections"]}:
                    host["sections"].append({"name": t["name"], "same": False, "only": "B", "b_sha": t["sha256"], "size_b": t["size"]})
                    host["unexplained_bytes"] += t["size"]
            dec = lambda ss: sorted(x.decode("ascii") for x in ss)[:12]
            host["strings_only_a"], host["strings_only_b"] = dec(sa_only), dec(sb_only)
            host["result"] = "differs" if host["unexplained_bytes"] else "stamp_only"
    txt = [s for s in host["sections"] if s["name"] == ".text"]
    host["text_same"] = True if da == db_ else (bool(txt) and txt[0]["same"]) if txt else None
    return {"name": os.path.basename(b_path).lower(), "a": {"path": os.path.abspath(a_path), "sha256": sha(da), "size": len(da)},
            "b": {"path": os.path.abspath(b_path), "sha256": sha(db_), "size": len(db_)},
            "code_objects": {"result": co, "n_a": len(oa), "n_b": len(ob), "objects": objs, "allow_cuid_length": allow_cuid_length},
            "host": host}


# ---------------------------------------------------------------- launch tables
def _log_files(folder):
    out = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if launch_table.is_launch_log(fn):  # bug:log-pid-names: launch_<pid>.txt or launch_<pid>_<start>[_<n>].txt
                out.append(os.path.join(root, fn))
    return out


def _exe_of(cmd):
    t = re.findall(r'"[^"]*"|\S+', cmd)
    return t[0].strip('"') if t else ""


def load_side(src, include_all=False, server_seq=False, ignore=(), db=None, scope="test"):
    """-> (runs {key: run}, info {kind, logs: [(path, exe, mtime_ns)] }). src: launch folder (flat or step sub-folders),
    a saved table file, or 'db:<exe sha>' (results.db work_traces rows of that exe)."""
    if src.startswith("db:"):
        h, _, gpu = src[3:].lower().partition("@")
        runs = {}
        if db and os.path.isfile(db):
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            # gpu: the log's --device (lt_import work_json.gpu) or the run's gpu; "db:<sha>@gfx1201" keeps one GPU
            q = ("SELECT w.component, w.cell, w.work_hash, w.seq_hash, w.arg_hash, w.launches, "
                 "COALESCE(NULLIF(json_extract(w.work_json, '$.gpu'), ''), r.gpu, '') FROM work_traces w "
                 "LEFT JOIN runs r ON r.run_id = w.run_id WHERE lower(w.exe_sha256) = ? AND w.scope = ?")
            for comp, cell, wh, sq, ah, n, g in con.execute(q, (h, scope)):
                if gpu and norm_gpu(gpu) != norm_gpu(g):
                    continue
                key = f"{comp}/{cell}" if comp else cell
                # rows = the stored work hash (it covers launches, sequence, kernel rows and argument digests);
                # args {} when the log had argument digests, so the compare counts them as compared
                r = runs.setdefault(key, {"n": n, "seq": sq or "", "aseq": ah or "", "rows": {"work_hash": wh},
                                          "args": {} if ah else None, "hashes": set()})
                r["hashes"].add((wh, sq, ah, n))
            con.close()
        for k, r in runs.items():
            if len(r["hashes"]) > 1:  # repeated runs of one exe disagree: the stored evidence is not deterministic
                r["rows"] = {"work_hash": "inconsistent:" + ",".join(sorted(str(x[0]) for x in r["hashes"]))}
            del r["hashes"]
        return runs, {"kind": "db", "exe": h, "gpu": gpu, "logs": []}
    if os.path.isfile(src):
        return launch_table.apply_ignore(launch_table.read_table(src), ignore), {"kind": "table", "logs": [(src, "", 0)]}
    if not os.path.isdir(src):
        return {}, {"kind": "none", "logs": []}
    runs, logs = {}, []
    dirs = sorted({os.path.dirname(p) for p in _log_files(src)})
    for d in dirs:
        rel = os.path.relpath(d, src).replace("\\", "/")
        t = launch_table.apply_ignore(launch_table.table_of(d, include_all, server_seq), ignore)
        for k, v in t.items():
            runs[k if rel == "." else f"{rel}/{k}"] = v
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if launch_table.is_launch_log(fn):  # bug:log-pid-names: launch_<pid>.txt or launch_<pid>_<start>[_<n>].txt
                cmd = launch_table.read_log(p)[0]
                srv = launch_table.norm_cmd(cmd).startswith("whirl-server")
                if srv and not (include_all or server_seq):
                    continue
                logs.append((p, _exe_of(cmd), os.stat(p).st_mtime_ns))
    return runs, {"kind": "folder", "logs": logs}


def bind(info, exes, cache):
    """Is the launch evidence about this side's exes? -> {state, detail}. sha: every log's exe file (matched by base name)
    hashes to the side's sha and was not modified after the log was written; db: rows selected by exe sha; table:
    a saved table names no exe; mismatch: some log's exe hashes to another sha; unverified: exe gone / rebuilt since."""
    if info["kind"] == "db":
        ok = info["exe"] in {v.lower() for v in exes.values()} if exes else True
        return {"state": "db" if ok else "mismatch", "detail": f"work_traces.exe_sha256 {info['exe'][:16]}"}
    if info["kind"] == "table":
        return {"state": "table", "detail": "a saved launch table names no exe"}
    if not info["logs"]:
        return {"state": "none", "detail": "no launch logs"}
    states, det = set(), {}
    for p, exe, mt in info["logs"]:
        name = os.path.basename(exe).lower()
        want = (exes or {}).get(name)
        if not want:
            states.add("unverified"); det[name] = "no expected sha for this exe"; continue
        if not os.path.isfile(exe):
            states.add("unverified"); det[name] = f"{exe} no longer exists"; continue
        st = os.stat(exe)
        if st.st_mtime_ns > mt:
            states.add("unverified"); det[name] = f"{exe} modified after the log {os.path.basename(p)}"; continue
        if exe not in cache:
            cache[exe] = file_sha(exe)
        if cache[exe] != want.lower():
            states.add("mismatch"); det[name] = f"{exe} sha {cache[exe][:16]} != expected {want[:16]}"; continue
        states.add("sha"); det.setdefault(name, f"{exe} sha {want[:16]}")
    state = "mismatch" if "mismatch" in states else "unverified" if "unverified" in states else "sha"
    return {"state": state, "detail": det}


def compare_launch(a_src, b_src, server=False, ignore=(), a_exes=None, b_exes=None, db=None, allow_unbound=False, scope="test"):
    ra, ia = load_side(a_src, False, server, ignore, db, scope)
    rb, ib = load_side(b_src, False, server, ignore, db, scope)
    cache = {}
    ba, bb = bind(ia, a_exes, cache), bind(ib, b_exes, cache)
    res = launch_table.compare(ra, rb) if ra and rb else []
    runs = [{k: v for k, v in e.items() if k not in ("rows_diff", "args_diff")} | ({"rows_diff": e["rows_diff"][:20], "args_diff": e["args_diff"][:20]} if e["status"] == "DIFF" else {})
            for e in res]
    reasons = []
    if not ra or not rb:
        result = "missing"; reasons.append(f"launch: no runs for {'A' if not ra else 'B'} ({(a_src if not ra else b_src)})")
    elif any(e["status"] == "DIFF" for e in res):
        result = "different"; reasons.append(f"launch: {sum(e['status'] == 'DIFF' for e in res)} run(s) differ")
    elif any(e["status"] != "SAME" for e in res):
        result = "missing"; reasons.append(f"launch: {sum(e['status'] != 'SAME' for e in res)} run(s) on one side only")
    else:
        result = "identical"
    if result == "identical":
        for side, b in (("A", ba), ("B", bb)):
            if b["state"] == "mismatch" or (b["state"] not in ("sha", "db") and not allow_unbound):
                result = "missing"; reasons.append(f"launch: evidence of {side} not bound to its exe ({b['state']}: {b['detail']})")
    return {"result": result, "a": a_src, "b": b_src, "server": server, "ignore": list(ignore), "runs": runs,
            "n_same": sum(e["status"] == "SAME" for e in res), "n_runs": len(res),
            "args_compared": sum(1 for e in res if e.get("args_compared")),
            "binding": {"a": ba, "b": bb, "allow_unbound": allow_unbound}, "reasons": reasons, "_logs": (ia["logs"], ib["logs"])}


# ---------------------------------------------------------------- golden
GOLD_EXE = re.compile(r"^golden: (?:(server|gate) )?exe (.+?) sha256 ([0-9a-f]{64})")
GOLD_REF = re.compile(r"^golden: reference set (\S+) .*all files of \S+?:? ([0-9a-f]{64})")
GOLD_CASE = re.compile(r"^(OK|DIFF|FAIL|MISS|ERR|SKIP)\s+(\S+/\S+)\s")


def parse_golden(paths):
    out = {"exes": {}, "ref_all": {}, "cases": {}}
    for p in paths:
        arch = ""  # cases are keyed "<arch>:<model>/<mode>_<suite>": one report may hold both GPUs' golden runs
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                m = GOLD_EXE.match(line)
                if m:
                    n = os.path.basename(m.group(2)).lower()
                    out["exes"][n] = m.group(3) if out["exes"].get(n, m.group(3)) == m.group(3) else "conflict"
                    continue
                m = GOLD_REF.match(line)
                if m:
                    arch = m.group(1)
                    if out["ref_all"].get(arch, m.group(2)) != m.group(2):
                        out["ref_all"][arch] = "conflict"  # two files of one side compared against different references
                    else:
                        out["ref_all"][arch] = m.group(2)
                    continue
                m = GOLD_CASE.match(line)
                if m:
                    k = f"{arch}:{m.group(2)}" if arch else m.group(2)
                    if out["cases"].get(k, "OK") != "OK":
                        continue  # a case that failed once on this side stays failed
                    out["cases"][k] = m.group(1)
    return out


def _out_hashes(d):
    if not d or not os.path.isdir(d):
        return {}
    return {fn: file_sha(os.path.join(d, fn)) for fn in sorted(os.listdir(d))
            if fn.endswith(".txt") and not fn.endswith(".diag.txt") and os.path.isfile(os.path.join(d, fn))}


def compare_golden(a_files, b_files, a_out=None, b_out=None, a_exes=None, b_exes=None):
    ga, gb = parse_golden(a_files or []), parse_golden(b_files or [])
    reasons = []

    def binding(g, exes):
        """the exes that produced the golden outputs: whirl.exe always, whirl-server.exe when a server case ran
        (whirl-server-gate.exe is the test client: not part of the product work)"""
        if not g["exes"]:
            return "none"
        need = {"whirl.exe"} | ({"whirl-server.exe"} if any(c.endswith("_server") for c in g["cases"]) else set())
        st = "sha"
        for n in sorted(need):
            h, want = g["exes"].get(n), (exes or {}).get(n)
            if h is None or want is None:
                st = "unverified"
            elif want.lower() != h:
                return "mismatch"
        return st
    bnd = {"a": binding(ga, a_exes), "b": binding(gb, b_exes)}
    cases = []
    for c in sorted(set(ga["cases"]) | set(gb["cases"])):
        x, y = ga["cases"].get(c), gb["cases"].get(c)
        cases.append({"case": c, "a": x, "b": y, "same": x == "OK" and y == "OK"})
    oa, ob = _out_hashes(a_out), _out_hashes(b_out)
    outs = {"compared": 0, "differing": [], "only_a": sorted(set(oa) - set(ob)), "only_b": sorted(set(ob) - set(oa))}
    for fn in sorted(set(oa) & set(ob)):
        outs["compared"] += 1
        if oa[fn] != ob[fn]:
            outs["differing"].append(fn)
    if not cases:
        result = "missing"; reasons.append("golden: no golden summary for " + ("A and B" if not ga["cases"] and not gb["cases"] else "A" if not ga["cases"] else "B"))
    elif any(c["a"] not in ("OK", None) or c["b"] not in ("OK", None) for c in cases) or outs["differing"]:
        result = "different"; reasons.append(f"golden: {sum(1 for c in cases if c['a'] not in ('OK', None) or c['b'] not in ('OK', None))} case(s) not OK, "
                                             f"{len(outs['differing'])} output file(s) differ")
    elif any(c["b"] is None for c in cases) or not any(c["same"] for c in cases):
        # B (the build that inherits A's numbers) must cover every case A ran; extra B cases are reported, not blocking
        result = "missing"; reasons.append(f"golden: {sum(1 for c in cases if c['b'] is None)} case(s) of A missing in B, "
                                           f"{sum(c['same'] for c in cases)} compared")
    elif any(gb["ref_all"].get(k) != v or v == "conflict" for k, v in ga["ref_all"].items()) or not ga["ref_all"]:
        result = "missing"; reasons.append(f"golden: reference sets differ or unnamed (A {ga['ref_all']} vs B {gb['ref_all']}): OK on both sides does not prove equal outputs")
    else:
        result = "identical"
    if result == "identical":
        for side in ("a", "b"):
            if bnd[side] != "sha":
                result = "missing"; reasons.append(f"golden: evidence of {side.upper()} not bound to its exe ({bnd[side]}; named {(ga if side == 'a' else gb)['exes']})")
    return {"result": result, "cases": cases, "only_a": [c["case"] for c in cases if c["b"] is None],
            "only_b": [c["case"] for c in cases if c["a"] is None], "ref_all": {"a": ga["ref_all"], "b": gb["ref_all"]},
            "exes_named": {"a": ga["exes"], "b": gb["exes"]}, "binding": bnd, "outputs": outs, "reasons": reasons,
            "files": {"a": list(a_files or []), "b": list(b_files or [])}}


# ---------------------------------------------------------------- git (host-code diff summary, read only)
def git_summary(repo, a, b):
    p = subprocess.run(["git", "-C", repo, "diff", "--name-only", a, b], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip()[:300], "a": a, "b": b}
    files = [f for f in p.stdout.split("\n") if f]
    kind = lambda f: ("kernel" if f.startswith("kernels/") or f.endswith((".hip", ".cuh")) else
                      "host" if f.startswith(("src/", "include/")) else "build" if f.startswith(("cmake/", "CMakeLists")) else
                      "test" if f.startswith("tests/") else "tool" if f.startswith("tools/") else "doc" if f.endswith(".md") else "other")
    by = {}
    for f in files:
        by.setdefault(kind(f), []).append(f)
    return {"a": a, "b": b, "files": len(files), "by_kind": {k: v[:50] for k, v in sorted(by.items())},
            "counts": {k: len(v) for k, v in sorted(by.items())}}


# ---------------------------------------------------------------- report
def _evidence(paths):
    out = []
    for p in paths:
        if p and os.path.isfile(p):
            out.append({"path": os.path.abspath(p), "sha256": file_sha(p), "size": os.path.getsize(p)})
    return out


def report(a_exes, b_exes, launch=None, golden_a=None, golden_b=None, golden_out=(None, None), server=False, ignore=(),
           db=None, allow_unbound=False, git=None, extra_launch=(), allow_cuid_length=True):
    """a_exes / b_exes: exe paths, paired by base name. launch: (a_src, b_src) or None; extra_launch: more pairs.
    checks.code_objects: identical / identical-except-cuid (both count as equal for the verdict) / different / missing."""
    pa = {os.path.basename(p).lower(): p for p in a_exes}
    pb = {os.path.basename(p).lower(): p for p in b_exes}
    reasons, exes = [], []
    for n in sorted(set(pa) | set(pb)):
        gone = [p for p in (pa.get(n), pb.get(n)) if p and not os.path.isfile(p)]
        if n not in pa or n not in pb or gone:
            reasons.append(f"code objects: {n} " + (f"file not found: {gone}" if gone else f"only on side {'A' if n in pa else 'B'}"))
            exes.append({"name": n, "code_objects": {"result": "missing"}, "host": {"result": "missing"}})
            continue
        exes.append(compare_exe(pa[n], pb[n], allow_cuid_length=allow_cuid_length))
    sha_a = {e["name"]: e["a"]["sha256"] for e in exes if "a" in e}
    sha_b = {e["name"]: e["b"]["sha256"] for e in exes if "b" in e}
    cos = [e["code_objects"]["result"] for e in exes]
    co = ("missing" if not exes or "missing" in cos and "different" not in cos else "different" if "different" in cos else
          "identical-except-cuid" if "identical-except-cuid" in cos else "identical")
    for e in exes:
        if e["code_objects"]["result"] == "different":
            bad = [o for o in e["code_objects"]["objects"] if not o["same"] and not o.get("cuid_only")]
            reasons.append(f"code objects: {e['name']} {len(bad)} object(s) differ (n {e['code_objects']['n_a']} vs {e['code_objects']['n_b']}): "
                           + ", ".join(f"#{o['index']} {o['target']} {o.get('differing_sections') or o.get('only') or 'header'}"
                                       + (f" (not cuid-only: {o['cuid']})" if o.get("cuid") else "") for o in bad[:4]))
        elif e["code_objects"]["result"] == "missing" and "a" in e:
            reasons.append(f"code objects: {e['name']} has no embedded AMDGPU code object")
    pairs = ([launch] if launch else []) + list(extra_launch)
    ls = [compare_launch(x, y, server, ignore, sha_a, sha_b, db, allow_unbound) for x, y in pairs]
    if not ls:
        lr = "missing"; reasons.append("launch: no launch-table evidence given (--a-launch / --b-launch)")
    else:
        rs = [x["result"] for x in ls]
        lr = "different" if "different" in rs else "missing" if "missing" in rs else "identical"
        for x in ls:
            reasons += x["reasons"]
    gl = compare_golden(golden_a, golden_b, golden_out[0], golden_out[1], sha_a, sha_b)
    reasons += gl["reasons"]
    checks = {"code_objects": co, "launch_sequence": lr, "golden": gl["result"]}
    hosts = [e["host"]["result"] for e in exes]
    checks["host_code"] = ("missing" if not hosts or "missing" in hosts else "differs" if "differs" in hosts else
                           "stamp_only" if "stamp_only" in hosts else "identical")
    req = [checks["code_objects"], checks["launch_sequence"], checks["golden"]]  # identical-except-cuid passes (arch025-weq)
    verdict = "NOT_EQUIVALENT" if "different" in req else "INCOMPLETE" if "missing" in req else "EQUIVALENT"
    ev = []
    for x in ls:
        for logs in x.pop("_logs"):
            ev += [p for p, _e, _m in logs]
    evidence = _evidence(list(golden_a or []) + list(golden_b or []) + ev)
    rep = {"format": FORMAT, "tool": "tools/impact/work_equal.py", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "rule": "ADR-0003 decision 3: code objects + launch sequence + golden bitwise identical (host code: information); "
                   "code objects that differ only in the per-TU __hip_cuid_<hex> symbol count as identical (identical-except-cuid"
                   + (", cuid length changes allowed" if allow_cuid_length else "") + ")",
           "verdict": verdict, "exit": {"EQUIVALENT": 0, "NOT_EQUIVALENT": 1, "INCOMPLETE": 2}[verdict], "checks": checks,
           "reasons": reasons, "a": {"exes": sha_a}, "b": {"exes": sha_b}, "exes": exes, "launch": ls, "golden": gl,
           "evidence": evidence}
    if git:
        rep["git"] = git_summary(*git)
    return rep


def summary_lines(rep):
    yield f"work_equal: {rep['verdict']}  (code objects {rep['checks']['code_objects']}, launch {rep['checks']['launch_sequence']}, " \
          f"golden {rep['checks']['golden']}; host code {rep['checks']['host_code']})"
    for e in rep["exes"]:
        if "a" not in e:
            yield f"  {e['name']}: only on one side"; continue
        co, h = e["code_objects"], e["host"]
        yield (f"  {e['name']}: A {e['a']['sha256'][:16]} B {e['b']['sha256'][:16]}; code objects {co['result']} "
               f"({co['n_a']}/{co['n_b']}: {', '.join(o['target'] + ('=' if o['same'] else '=~cuid' if o.get('cuid_only') else '!=') for o in co['objects'])}); "
               f"host {h['result']} (.text {'same' if h['text_same'] else 'differs' if h['text_same'] is False else '?'}, "
               f"unexplained bytes {h['unexplained_bytes']}, stamp bytes {h['stamp_bytes']})")
        if h.get("strings_only_a") or h.get("strings_only_b"):
            yield f"      strings only in A: {h['strings_only_a'][:4]}; only in B: {h['strings_only_b'][:4]}"
        for s in h["sections"]:
            if not s.get("same"):
                yield f"      section {s['name']}: {s.get('differing_bytes', '?')} differing bytes outside code objects + {s.get('stamp_bytes', 0)} stamp bytes"
    for x in rep["launch"]:
        yield (f"  launch {x['a']} vs {x['b']}: {x['result']}, {x['n_same']}/{x['n_runs']} runs same, arguments compared in "
               f"{x['args_compared']}; binding A {x['binding']['a']['state']} B {x['binding']['b']['state']}")
    g = rep["golden"]
    yield (f"  golden: {g['result']}, {sum(c['same'] for c in g['cases'])}/{len(g['cases'])} cases OK on both (only in B {len(g['only_b'])}), outputs compared "
           f"{g['outputs']['compared']} ({len(g['outputs']['differing'])} differ); binding A {g['binding']['a']} B {g['binding']['b']}")
    if "git" in rep:
        yield f"  git {rep['git'].get('a')}..{rep['git'].get('b')}: {rep['git'].get('counts', rep['git'].get('error'))}"
    for r in rep["reasons"]:
        yield f"  reason: {r}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("compare")
    c.add_argument("--a-exe", action="append", default=[])
    c.add_argument("--b-exe", action="append", default=[])
    c.add_argument("--a-launch", action="append", default=[])
    c.add_argument("--b-launch", action="append", default=[])
    c.add_argument("--db")
    c.add_argument("--server", action="store_true")
    c.add_argument("--ignore", action="append", default=[])
    c.add_argument("--ignore-file")
    c.add_argument("--allow-unbound-launch", action="store_true")
    c.add_argument("--a-golden", action="append", default=[])
    c.add_argument("--b-golden", action="append", default=[])
    c.add_argument("--a-golden-out")
    c.add_argument("--b-golden-out")
    c.add_argument("--repo")
    c.add_argument("--a-commit")
    c.add_argument("--b-commit")
    # main session 2026-10-25: on by default. clang prints the cuid with utohexstr, which drops leading zeros, so about
    # 1 TU in 16 gets a 15-digit cuid and cross-worktree pairs differ in length at random (real pair: obj #2, 27 vs 26 B);
    # the length-aware check still requires sh_size, DT_STRSZ and every st_name shift to equal the length difference.
    c.add_argument("--strict-cuid-length", dest="allow_cuid_length", action="store_false",
                   help="treat a __hip_cuid_<hex> of another length as a difference (default: accepted)")
    c.add_argument("--allow-cuid-length", dest="allow_cuid_length", action="store_true", help=argparse.SUPPRESS)
    c.add_argument("-o")
    s = sub.add_parser("scan")
    s.add_argument("exe")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 3 if e.code else 0
    if a.cmd == "scan":
        r = scan_exe(a.exe)
        r["pe"] = r["pe"] and {"sections": r["pe"]["sections"], "stamps": r["pe"]["stamps"]}
        print(json.dumps(r, indent=1))
        return 0
    if a.cmd != "compare" or not a.a_exe or not a.b_exe or len(a.a_launch) != len(a.b_launch):
        print("work_equal: compare needs --a-exe and --b-exe (paired by name) and as many --a-launch as --b-launch", file=sys.stderr)
        return 3
    ign = list(a.ignore) + (launch_table.read_ignore_file(a.ignore_file) if a.ignore_file else [])
    pairs = list(zip(a.a_launch, a.b_launch))
    git = (a.repo, a.a_commit, a.b_commit) if a.repo and a.a_commit and a.b_commit else None
    rep = report(a.a_exe, a.b_exe, pairs[0] if pairs else None, a.a_golden, a.b_golden, (a.a_golden_out, a.b_golden_out),
                 a.server, ign, a.db, a.allow_unbound_launch, git, pairs[1:], a.allow_cuid_length)
    for line in summary_lines(rep):
        print(line)
    if a.o:
        with open(a.o, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return rep["exit"]


if __name__ == "__main__":
    sys.exit(main())
