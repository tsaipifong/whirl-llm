#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel ISA + kernel descriptor comparison of two AMDGPU code objects (arch025-sc11k, ADR-0005; CPU only).

Proves that a source restructuring (moving kernels between files, merging duplicated code into helpers) changed no
kernel, even when the kernels' order in the code object changed - which whole-file byte comparison
(tools/impact/work_equal.py) cannot tell apart from a real change. Used by SC-11c / SC-11d / SC-15 and any kernel
refactor (prior art: progress/arch025-sc11k.PRIOR_ART.md).

Per code object it lists every symbol and compares the two builds BY NAME:
  code        every FUNC symbol in .text (kernels and non-inlined device functions): the official ROCm llvm-objdump -d
              disassembly of [value, value + size) with the address / encoding comments dropped (branch operands are
              relative) and the PC-relative literals after s_getpc_b64 (s_add[_co]_u32 lo / s_addc_u32 | s_add_co_ci_u32
              hi, gfx11 and gfx12) replaced by the symbol they reach (<tab+0x8>)
  descriptor every <kernel>.kd (64 bytes): raw bytes with kernel_code_entry_byte_offset (bytes 16-23, relative to the
              descriptor) replaced by the symbol it points at
  metadata    the per-kernel map of the NT_AMDGPU_METADATA note (msgpack; .vgpr_count, .sgpr_count, LDS, args ...),
              matched by .name, and the note's top-level fields (amdhsa.target, amdhsa.version)
  object      every other OBJECT symbol (__constant__ / __device__ data): bytes (.bss: size), dynamic relocations
              symbolized; plus the .got section
  abs         the ABS resource symbols (<kernel>.num_vgpr, .private_seg_size, ...)
__hip_cuid_<hash> (hash of the input path) is compared as __hip_cuid_. A symbol present on one side only is a
difference. Same content at other addresses / in another order is IDENTICAL (reordered).

    python -I tools/gates/kernel_isa_cmp.py A B [--max-diff N] [--json OUT] [--objdump EXE]
A and B are two code-object files (ELF, e.g. build/kernels/whirl_kernels.gfx1201.hsaco) or two directories (files
*.hsaco / *.co matched by relative path). Last line "kernel_isa_cmp: IDENTICAL (...)" or "kernel_isa_cmp: DIFFERENT
(...)". Exit 0 identical, 1 any difference, 2 setup error. Test: python -I tools/gates/kernel_isa_cmp_test.py.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import difflib
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

OBJDUMP_DEFAULT = r"C:\Program Files\AMD\ROCm\7.2\bin\llvm-objdump.exe"
NT_AMDGPU_METADATA = 32
SHT_NOBITS, SHT_RELA, SHT_SYMTAB = 8, 4, 2
STT_OBJECT, STT_FUNC = 1, 2
SHN_ABS = 0xFFF1
R_AMDGPU_RELATIVE64 = 13
Sym = collections.namedtuple("Sym", "name value size kind")


class SetupError(Exception):
    pass


def norm_name(n):
    return re.sub(r"^__hip_cuid_[0-9a-f]+$", "__hip_cuid_", n)


# ---- msgpack (the subset the AMDGPU metadata note uses) ------------------------------------------------------------
def msgpack_decode(buf):
    v, _ = _mp(buf, 0)
    return v


def _mp(b, p):
    c = b[p]
    p += 1
    if c <= 0x7F:
        return c, p
    if c >= 0xE0:
        return c - 0x100, p
    if 0x80 <= c <= 0x8F:
        return _map(b, p, c & 0x0F)
    if 0x90 <= c <= 0x9F:
        return _arr(b, p, c & 0x0F)
    if 0xA0 <= c <= 0xBF:
        n = c & 0x1F
        return b[p:p + n].decode("utf-8", "replace"), p + n
    if c == 0xC0:
        return None, p
    if c in (0xC2, 0xC3):
        return c == 0xC3, p
    if c in (0xC4, 0xC5, 0xC6):
        w = {0xC4: 1, 0xC5: 2, 0xC6: 4}[c]
        n = int.from_bytes(b[p:p + w], "big")
        return bytes(b[p + w:p + w + n]), p + w + n
    if c == 0xCA:
        return struct.unpack(">f", b[p:p + 4])[0], p + 4
    if c == 0xCB:
        return struct.unpack(">d", b[p:p + 8])[0], p + 8
    if 0xCC <= c <= 0xCF:
        w = 1 << (c - 0xCC)
        return int.from_bytes(b[p:p + w], "big"), p + w
    if 0xD0 <= c <= 0xD3:
        w = 1 << (c - 0xD0)
        return int.from_bytes(b[p:p + w], "big", signed=True), p + w
    if c in (0xD9, 0xDA, 0xDB):
        w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[c]
        n = int.from_bytes(b[p:p + w], "big")
        return bytes(b[p + w:p + w + n]).decode("utf-8", "replace"), p + w + n
    if c in (0xDC, 0xDD):
        w = 2 if c == 0xDC else 4
        return _arr(b, p + w, int.from_bytes(b[p:p + w], "big"))
    if c in (0xDE, 0xDF):
        w = 2 if c == 0xDE else 4
        return _map(b, p + w, int.from_bytes(b[p:p + w], "big"))
    raise SetupError("msgpack: unsupported type byte 0x%02x" % c)


def _arr(b, p, n):
    out = []
    for _ in range(n):
        v, p = _mp(b, p)
        out.append(v)
    return out, p


def _map(b, p, n):
    out = {}
    for _ in range(n):
        k, p = _mp(b, p)
        v, p = _mp(b, p)
        out[k] = v
    return out, p


# ---- ELF ------------------------------------------------------------------------------------------------------------
class Elf:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = d = f.read()
        self.path = path
        if d[:4] != b"\x7fELF":
            if d.startswith(b"__CLANG_OFFLOAD_BUNDLE__") or d.startswith(b"CCOB"):
                raise SetupError("%s: an offload bundle, not a code object (unbundle it with clang-offload-bundler, "
                                 "or build with --no-gpu-bundle-output)" % path)
            raise SetupError("%s: not an ELF file" % path)
        if d[4] != 2 or d[5] != 1:
            raise SetupError("%s: not ELF64 little-endian" % path)
        self.e_flags = struct.unpack_from("<I", d, 48)[0]
        shoff, = struct.unpack_from("<Q", d, 40)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 58)
        self.sections = []
        for i in range(shnum):
            name, typ, flags, addr, off, size, link, info, align, entsize = struct.unpack_from(
                "<IIQQQQIIQQ", d, shoff + i * shentsize)
            self.sections.append(dict(name_off=name, type=typ, flags=flags, addr=addr, off=off, size=size, link=link,
                                      entsize=entsize, idx=i))
        sst = self.sections[shstrndx]
        for s in self.sections:
            s["name"] = self._cstr(sst["off"] + s["name_off"])
        self.by_name = {s["name"]: s for s in self.sections}
        self.syms = self._symbols()
        self.relocs = self._relocs()

    def _cstr(self, p):
        e = self.data.index(b"\0", p)
        return self.data[p:e].decode("utf-8", "replace")

    def _symbols(self):
        st = next((s for s in self.sections if s["type"] == SHT_SYMTAB), None)
        if st is None:
            raise SetupError("%s: no .symtab (stripped code object?)" % self.path)
        strs = self.sections[st["link"]]
        out = []
        for i in range(st["size"] // 24):
            name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", self.data, st["off"] + i * 24)
            out.append(dict(name=self._cstr(strs["off"] + name) if name else "", type=info & 0xF, shndx=shndx,
                            value=value, size=size))
        return out

    def _relocs(self):
        out = {}
        dyn = self.by_name.get(".dynsym")
        dsyms = []
        if dyn:
            strs = self.sections[dyn["link"]]
            for i in range(dyn["size"] // 24):
                name, info, other, shndx, value, size = struct.unpack_from("<IBBHQQ", self.data, dyn["off"] + i * 24)
                dsyms.append((self._cstr(strs["off"] + name) if name else "", value))
        for s in self.sections:
            if s["type"] != SHT_RELA:
                continue
            for i in range(s["size"] // 24):
                off, info, addend = struct.unpack_from("<QQq", self.data, s["off"] + i * 24)
                out[off] = (info & 0xFFFFFFFF, info >> 32, addend, dsyms)
        return out

    def section_of(self, idx):
        return self.sections[idx] if 0 < idx < len(self.sections) else None

    def read(self, sec, addr, size):
        if sec["type"] == SHT_NOBITS:
            return None
        o = sec["off"] + addr - sec["addr"]
        return self.data[o:o + size]

    def notes_metadata(self):
        for s in self.sections:
            if s["type"] != 7:  # SHT_NOTE
                continue
            p, end = s["off"], s["off"] + s["size"]
            while p + 12 <= end:
                nsz, dsz, typ = struct.unpack_from("<III", self.data, p)
                name = self.data[p + 12:p + 12 + nsz].rstrip(b"\0")
                q = p + 12 + ((nsz + 3) & ~3)
                if typ == NT_AMDGPU_METADATA and name == b"AMDGPU":
                    return msgpack_decode(self.data[q:q + dsz])
                p = q + ((dsz + 3) & ~3)
        return {}


# ---- symbolization ----------------------------------------------------------------------------------------------------
class Resolver:
    def __init__(self, syms, sections=()):
        self.ranges = sorted((s.value, s.size, norm_name(s.name)) for s in syms if s.size > 0)
        self.starts = [r[0] for r in self.ranges]
        self.exact = {}
        for s in syms:
            self.exact.setdefault(s.value, norm_name(s.name))
        self.sections = sorted((s["addr"], s["size"], s["name"]) for s in sections if s.get("addr"))

    def name(self, a):
        i = bisect.bisect_right(self.starts, a) - 1
        # sized symbols can nest (none do in practice); take the closest start that covers a
        while i >= 0:
            v, sz, n = self.ranges[i]
            if v <= a < v + sz:
                return n if a == v else "%s+0x%x" % (n, a - v)
            if a - v > 1 << 32:
                break
            i -= 1
        if a in self.exact:
            return self.exact[a]
        for v, sz, n in self.sections:
            if v <= a < v + sz:
                return "%s+0x%x" % (n, a - v)
        return "?0x%x" % a


GETPC = re.compile(r"^s_getpc_b64 s\[(\d+):(\d+)\]$")
ADDLO = re.compile(r"^(s_add_u32|s_add_co_u32) s(\d+), s(\d+), (\S+)$")
ADDHI = re.compile(r"^(s_addc_u32|s_add_co_ci_u32) s(\d+), s(\d+), (\S+)$")


def _imm(t):
    try:
        return int(t, 0) & 0xFFFFFFFF
    except ValueError:
        return None


def symbolize(ins, syms_or_resolver):
    """ins: [(addr, normalized text)] of one function -> [text] with PC-relative literals replaced by <symbol>."""
    res = syms_or_resolver if isinstance(syms_or_resolver, Resolver) else Resolver(syms_or_resolver)
    out = []
    pend = {}  # lo register -> [base, lo value, index of the lo line in out, opcode, operand]
    for addr, t in ins:
        m = GETPC.match(t)
        if m:
            pend[int(m.group(1))] = [addr + 4, None, None]
            out.append(t)
            continue
        m = ADDLO.match(t)
        if m and m.group(2) == m.group(3) and int(m.group(2)) in pend and pend[int(m.group(2))][1] is None:
            v = _imm(m.group(4))
            if v is not None:
                pend[int(m.group(2))][1:] = [v, len(out)]
                out.append(t)
                continue
        m = ADDHI.match(t)
        if m and m.group(2) == m.group(3) and int(m.group(2)) - 1 in pend and pend[int(m.group(2)) - 1][1] is not None:
            hi = _imm(m.group(4))
            if hi is not None:
                base, lo, li = pend.pop(int(m.group(2)) - 1)
                target = (base + ((hi << 32) | lo)) & 0xFFFFFFFFFFFF
                n = res.name(target)
                lo_line = out[li]
                out[li] = lo_line[:lo_line.rindex(" ") + 1] + "<%s>" % n
                out.append(t[:t.rindex(" ") + 1] + "<%s:hi>" % n.split("+")[0])
                continue
        out.append(t)
    return out


# ---- extraction -------------------------------------------------------------------------------------------------------
LINE = re.compile(r"^\s+(\S.*?)\s*//\s*([0-9A-Fa-f]{6,16}):")


def disassemble(objdump, path, funcs):
    """funcs: [(start, end, name)] -> {name: [(addr, text)]}; streams llvm-objdump -d (official ROCm tool)."""
    funcs = sorted(funcs)
    starts = [f[0] for f in funcs]
    out = {f[2]: [] for f in funcs}
    try:
        p = subprocess.Popen([objdump, "-d", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", bufsize=1 << 20)
    except OSError as e:
        raise SetupError("cannot run %s: %s" % (objdump, e))
    cur = None
    for line in p.stdout:
        m = LINE.match(line)
        if not m:
            continue
        a = int(m.group(2), 16)
        if cur is None or not (cur[0] <= a < cur[1]):
            i = bisect.bisect_right(starts, a) - 1
            cur = funcs[i] if i >= 0 and funcs[i][0] <= a < funcs[i][1] else None
            if cur is None:
                continue  # padding between functions / after the last one
        out[cur[2]].append((a, " ".join(m.group(1).split())))
    err = p.stderr.read()
    if p.wait() != 0:
        raise SetupError("%s -d %s failed: %s" % (objdump, path, err[-500:]))
    return out


def reloc_view(elf, res, addr, size, raw):
    """hex of raw with every 8-byte dynamic relocation slot replaced by its symbolized target"""
    parts, i = [], 0
    while i < size:
        r = elf.relocs.get(addr + i)
        if r is not None and i + 8 <= size:
            typ, symi, addend, dsyms = r
            if typ == R_AMDGPU_RELATIVE64:
                parts.append("<reloc %s>" % res.name(addend))
            else:
                sname = dsyms[symi][0] if symi < len(dsyms) else "?"
                parts.append("<reloc %d %s%+d>" % (typ, norm_name(sname), addend))
            i += 8
        else:
            j = i
            while j < size and (addr + j) not in elf.relocs:
                j += 1
            parts.append(raw[i:j].hex())
            i = j
    return " ".join(parts)


def extract(path, objdump):
    elf = Elf(path)
    syms, funcs, kds, objs, abss = [], [], {}, {}, {}
    for s in elf.syms:
        if not s["name"]:
            continue
        if s["shndx"] == SHN_ABS:
            abss[norm_name(s["name"])] = s["value"]
            continue
        sec = elf.section_of(s["shndx"])
        if sec is None:
            continue
        if s["type"] in (STT_FUNC, STT_OBJECT) or s["size"]:
            syms.append(Sym(s["name"], s["value"], s["size"], s["type"]))
        if s["type"] == STT_FUNC and sec["flags"] & 4 and s["size"]:  # SHF_EXECINSTR
            funcs.append((s["value"], s["value"] + s["size"], norm_name(s["name"])))
        elif s["type"] == STT_OBJECT:
            if s["name"].endswith(".kd") and s["size"] == 64:
                kds[s["name"][:-3]] = (s["value"], elf.read(sec, s["value"], 64))
            else:
                objs[norm_name(s["name"])] = (sec, s["value"], s["size"])
    res = Resolver(syms, elf.sections)
    order = [f[2] for f in sorted(funcs)]
    dis = disassemble(objdump, path, funcs)
    code = {n: symbolize(ins, res) for n, ins in dis.items()}
    desc = {}
    for k, (v, raw) in kds.items():
        entry = struct.unpack_from("<q", raw, 16)[0]
        f = {"group_segment_fixed_size": struct.unpack_from("<I", raw, 0)[0],
             "private_segment_fixed_size": struct.unpack_from("<I", raw, 4)[0],
             "kernarg_size": struct.unpack_from("<I", raw, 8)[0],
             "compute_pgm_rsrc3": "0x%08x" % struct.unpack_from("<I", raw, 44)[0],
             "compute_pgm_rsrc1": "0x%08x" % struct.unpack_from("<I", raw, 48)[0],
             "compute_pgm_rsrc2": "0x%08x" % struct.unpack_from("<I", raw, 52)[0],
             "kernel_code_properties": "0x%04x" % struct.unpack_from("<H", raw, 56)[0],
             "entry": res.name(v + entry),
             "bytes": (raw[:16] + b"\0" * 8 + raw[24:]).hex()}
        desc[norm_name(k)] = f
    objects = {}
    for n, (sec, v, sz) in objs.items():
        raw = elf.read(sec, v, sz)
        objects[n] = "%s nobits size %d" % (sec["name"], sz) if raw is None else \
            "%s size %d: %s" % (sec["name"], sz, reloc_view(elf, res, v, sz, raw))
    got = elf.by_name.get(".got")
    if got and got["size"]:
        objects["section:.got"] = reloc_view(elf, res, got["addr"], got["size"], elf.read(got, got["addr"], got["size"]))
    meta = elf.notes_metadata()
    kmeta = {}
    for k in meta.get("amdhsa.kernels", []) or []:
        kmeta[norm_name(str(k.get(".name")))] = k
    top = {k: v for k, v in meta.items() if k != "amdhsa.kernels"}
    return dict(code=code, order=order, desc=desc, meta=kmeta, top=top, objects=objects, abs=abss,
                e_flags=elf.e_flags, nkern=len(kds))


# ---- comparison -------------------------------------------------------------------------------------------------------
def _short(v, n=300):
    s = json.dumps(v, sort_keys=True, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "...(%d chars)" % len(s)


def compare_one(a, b, objdump, max_diff):
    """returns (lines, n_differences, summary dict)"""
    lines = []
    ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
    hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
    if ha == hb:
        return ["  byte-identical (sha256 %s)" % ha[:16]], 0, {"result": "byte-identical"}
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        fa, fb = ex.submit(extract, a, objdump), ex.submit(extract, b, objdump)
        A, B = fa.result(), fb.result()
    nd = 0
    if A["e_flags"] != B["e_flags"]:
        lines.append("  DIFFERENT e_flags 0x%x vs 0x%x (another architecture / target features)" % (A["e_flags"], B["e_flags"]))
        nd += 1
    for cat in ("code", "desc", "meta", "objects", "abs"):
        label = {"code": "code", "desc": "descriptor", "meta": "metadata", "objects": "object", "abs": "abs"}[cat]
        da, db = A[cat], B[cat]
        only_a, only_b = sorted(set(da) - set(db)), sorted(set(db) - set(da))
        for n in only_a:
            lines.append("  only in base: %s %s" % (label, n))
        for n in only_b:
            lines.append("  only in new: %s %s" % (label, n))
        nd += len(only_a) + len(only_b)
        for n in sorted(set(da) & set(db)):
            if da[n] == db[n]:
                continue
            nd += 1
            lines.append("  DIFFERENT %s %s" % (label, n))
            if cat == "code":
                d = list(difflib.unified_diff(da[n], db[n], lineterm="", n=2))[2:]
                for l in d[:max_diff]:
                    lines.append("    " + (l if l.startswith("@@") else l[0] + " " + l[1:]))
                if len(d) > max_diff:
                    lines.append("    ... %d more diff lines" % (len(d) - max_diff))
                lines.append("    (%d vs %d instructions)" % (len(da[n]), len(db[n])))
            elif isinstance(da[n], dict):
                for k in sorted(set(da[n]) | set(db[n])):
                    if da[n].get(k) != db[n].get(k):
                        lines.append("    %s: %s -> %s" % (k, _short(da[n].get(k)), _short(db[n].get(k))))
            else:
                lines.append("    base: %s" % _short(da[n]))
                lines.append("    new:  %s" % _short(db[n]))
    if A["top"] != B["top"]:
        nd += 1
        lines.append("  DIFFERENT metadata (top level): %s -> %s" % (_short(A["top"]), _short(B["top"])))
    common = [n for n in A["order"] if n in set(B["order"])]
    pos_b = {n: i for i, n in enumerate(n for n in B["order"] if n in set(A["order"]))}
    moved = sum(1 for i, n in enumerate(common) if pos_b[n] != i)
    summ = {"functions": len(A["code"]), "kernels": A["nkern"], "descriptors": len(A["desc"]),
            "objects": len(A["objects"]), "abs": len(A["abs"]), "reordered": moved, "differences": nd}
    head = "  %d functions (%d kernels), %d descriptors, %d metadata entries, %d data objects, %d abs symbols" % (
        len(A["code"]), A["nkern"], len(A["desc"]), len(A["meta"]), len(A["objects"]), len(A["abs"]))
    if nd == 0:
        summ["result"] = "identical-reordered" if moved else "identical"
        res = "  IDENTICAL" + (" (reordered: %d function(s) at another position in the code object)" % moved if moved
                               else " (same order; files differ only in addresses / symbol names such as __hip_cuid_)")
        return [head, res], 0, summ
    summ["result"] = "different"
    return [head] + lines, nd, summ


def pairs(a, b):
    if os.path.isfile(a) and os.path.isfile(b):
        return [(os.path.basename(a), a, b)], []
    if os.path.isdir(a) and os.path.isdir(b):
        def ls(d):
            out = {}
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith((".hsaco", ".co")):
                        p = os.path.join(root, f)
                        out[os.path.relpath(p, d).replace("\\", "/")] = p
            return out
        la, lb = ls(a), ls(b)
        if not la and not lb:
            raise SetupError("no *.hsaco / *.co files under %s or %s" % (a, b))
        common = sorted(set(la) & set(lb))
        missing = ["only in base: code object %s" % r for r in sorted(set(la) - set(lb))] + \
                  ["only in new: code object %s" % r for r in sorted(set(lb) - set(la))]
        return [(r, la[r], lb[r]) for r in common], missing
    raise SetupError("give two files or two directories (%s, %s)" % (a, b))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--max-diff", type=int, default=60, help="instruction diff lines shown per function (default 60)")
    ap.add_argument("--json", help="write the per-code-object summary as JSON here")
    ap.add_argument("--objdump", default=os.environ.get("WHIRL_LLVM_OBJDUMP", OBJDUMP_DEFAULT))
    args = ap.parse_args(argv)
    try:
        if not os.path.exists(args.objdump):
            raise SetupError("llvm-objdump not found: %s (ROCm 7.2 HIP SDK; or --objdump / WHIRL_LLVM_OBJDUMP)" % args.objdump)
        todo, missing = pairs(args.base, args.new)
        total, report = len(missing), {}
        print("kernel_isa_cmp: %s vs %s" % (args.base, args.new))
        for m in missing:
            print("  " + m)
        for rel, a, b in todo:
            print("[%s]" % rel)
            lines, nd, summ = compare_one(a, b, args.objdump, args.max_diff)
            for l in lines:
                print(l)
            total += nd
            report[rel] = summ
    except SetupError as e:
        print("kernel_isa_cmp: setup error: %s" % e)
        return 2
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"base": args.base, "new": args.new, "code_objects": report, "differences": total}, f, indent=1)
    n = len(todo)
    if total == 0:
        print("kernel_isa_cmp: IDENTICAL (%d code object(s), every function / descriptor / metadata / data object equal)" % n)
        return 0
    print("kernel_isa_cmp: DIFFERENT (%d difference(s); %d code object(s) compared)" % (total, n))
    return 1


if __name__ == "__main__":
    sys.exit(main())
