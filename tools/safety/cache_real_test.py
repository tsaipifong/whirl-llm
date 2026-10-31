#!/usr/bin/env python
"""arch025-procopt3 (PLAN 9.33): real-toolchain proof of the persistent safety build cache (safety_cache.py) - the same cl.exe /
CMake / ninja and the same console code page as tools/safety/build_cfg.bat, on a 2-TU project in a temp dir (about a minute):

  1 first build                    both TUs compiled; depscheck ok (headers recorded through the localized /showIncludes prefix)
  2 fresh checkout (new mtimes,    sync copies nothing, the next build does no work
    same content)
  3 header edit (h1.h, only a.cpp  sync copies h1.h only; the build recompiles a.cpp and NOT b.cpp (brief 9.33 (a))
    includes it)
  4 compile flag change            CMakeLists.txt edit: CMake re-runs and both TUs are recompiled (brief 9.33 (b); a toolchain /
                                   build-script change wipes the dir through the key: test_safety_cache.py)
  5 clean rebuild of the same tree the objects of a fresh dir match the incremental ones byte for byte except the COFF timestamp,
                                   i.e. incremental == clean (brief 9.33 (c), on this project)
Prints "cache real test: ok (...)" and exits 0, or "cache real test: FAIL (...)" and exits 1.
Usage: python -I tools/safety/cache_real_test.py [work_dir]
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import safety_cache as sc  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CML = "cmake_minimum_required(VERSION 3.20)\nproject(cachetest CXX)\n{flags}add_library(t STATIC a.cpp b.cpp)\n"


def w(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def oem_cp():
    r = subprocess.run(["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\Nls\CodePage", "/v", "OEMCP"], capture_output=True, text=True,
                       errors="replace")
    m = re.search(r"OEMCP\s+REG_SZ\s+(\d+)", r.stdout)
    return m.group(1) if m else None


def build(work, mirror, b, configure):
    """vcenv + the build_cfg.bat code page + (configure) + build; returns (rc, output)"""
    bat = os.path.join(work, "b.bat")
    cp = oem_cp()
    lines = ["@echo off", f'call "{os.path.join(HERE, "vcenv.bat")}"', "if errorlevel 1 exit /b 1"]
    if cp:
        lines.append(f"chcp {cp} >nul")
    if configure:
        lines.append(f'cmake -S "{mirror}" -B "{b}" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1')
    lines.append(f'cmake --build "{b}"')
    with open(bat, "w", encoding="ascii", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run(["cmd.exe", "/d", "/c", bat], capture_output=True)
    out = r.stdout + r.stderr
    with open(os.path.join(work, "build.log"), "wb") as f:
        f.write(out)
    return r.returncode, out.decode("cp" + cp if cp else "utf-8", errors="replace")


def compiled(out):
    return sorted(set(re.findall(r"Building CXX object CMakeFiles[\\/]t\.dir[\\/](\w+)\.cpp\.obj", out)))


def obj_body(p):
    """a COFF object without its 4-byte TimeDateStamp (offset 4) - the only field that differs between two compiles"""
    with open(p, "rb") as f:
        d = bytearray(f.read())
    d[4:8] = b"\0\0\0\0"
    return bytes(d)


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="cache_real_")
    if os.path.exists(work):
        shutil.rmtree(work)
    src, mirror = os.path.join(work, "tree"), os.path.join(work, "cache", "src")
    b = os.path.join(mirror, "build-asan")
    files = {"CMakeLists.txt": CML.format(flags=""), "h1.h": "inline int h1() { return 1; }\n", "h2.h": "inline int h2() { return 2; }\n",
             "a.cpp": '#include "h1.h"\nint a() { return h1(); }\n', "b.cpp": '#include "h2.h"\nint b() { return h2(); }\n'}
    for k, v in files.items():
        w(os.path.join(src, k), v)
    sc.write_manifest(src, list(files))
    res, bad = [], []

    def step(name, ok, detail):
        res.append(f"{name}: {'ok' if ok else 'FAIL'} ({detail})")
        if not ok:
            bad.append(name)

    sc.sync(src, mirror)
    rc, out = build(work, mirror, b, True)
    c = compiled(out)
    step("1 first build", rc == 0 and c == ["a", "b"], f"rc {rc}, compiled {c}")
    dp = sc.depscheck(b, mirror, os.path.join(work, "build.log"))
    step("1 depscheck", not dp, "; ".join(dp) or "headers recorded")
    r = subprocess.run(["ninja", "-C", b, "-t", "deps"], capture_output=True, text=True, errors="replace")
    step("1 h1.h recorded for a.cpp", re.search(r"a\.cpp\.obj: #deps [1-9].*\n(?:\s+.*\n)*?\s+.*h1\.h", r.stdout) is not None,
         "ninja -t deps")

    for k in files:  # a fresh checkout: new mtimes, same content
        os.utime(os.path.join(src, k), None)
    s = sc.sync(src, mirror)
    rc, out = build(work, mirror, b, False)
    step("2 fresh checkout", s["copied"] == 0 and rc == 0 and compiled(out) == [], f"copied {s['copied']}, compiled {compiled(out)}")

    w(os.path.join(src, "h1.h"), "inline int h1() { return 11; }\n")
    s = sc.sync(src, mirror)
    rc, out = build(work, mirror, b, False)
    step("3 header edit", s["changed"] == ["h1.h"] and rc == 0 and compiled(out) == ["a"], f"copied {s['changed']}, compiled {compiled(out)}")

    w(os.path.join(src, "CMakeLists.txt"), CML.format(flags="add_compile_definitions(CACHETEST_FLAG=2)\n"))
    s = sc.sync(src, mirror)
    rc, out = build(work, mirror, b, False)
    step("4 compile flag change", rc == 0 and compiled(out) == ["a", "b"], f"compiled {compiled(out)}")

    # clean rebuild in the SAME dir (cl writes the object path and the command line into .debug$S, so another dir differs anyway)
    inc = {t: obj_body(os.path.join(b, "CMakeFiles", "t.dir", t + ".cpp.obj")) for t in ("a", "b")}
    shutil.rmtree(b)
    rc2, out2 = build(work, mirror, b, True)
    same = []
    for t in ("a", "b"):
        p2 = os.path.join(b, "CMakeFiles", "t.dir", t + ".cpp.obj")
        same.append(os.path.exists(p2) and obj_body(p2) == inc[t])
    step("5 incremental == clean", rc2 == 0 and all(same), f"objects equal {same}")

    for x in res:
        print(x)
    print(f"cache real test: {'ok' if not bad else 'FAIL (' + ', '.join(bad) + ')'} ({len(res) - len(bad)}/{len(res)} checks; work {work})")
    if not bad and len(sys.argv) <= 1:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
