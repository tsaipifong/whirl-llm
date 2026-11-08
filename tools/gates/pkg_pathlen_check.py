"""pkg_pathlen_check.py - the PACKAGE_VERIFY CPU cases of PATH-256 (P5, PLAN B26-6 / V1-22; progress/pathlen256.DISCOVERY.md
v3.1 section 2c #2 and 2d-3 A-3; prior art progress/v0252-path5.PRIOR_ART.md). Release gate step cpu_pkg_pathlen.

Rule (Sir Pifeng 2026-11-03 17:44 「安裝資料夾 ≤150」): whirl.exe and whirl-server.exe refuse an install folder over 150 UTF-8
bytes (exit 2, a message naming "bytes in UTF-8" and 150) as their first action, before anything can load the AMD runtime
(amdhip64_7.dll); a folder of 150 bytes works. The check copies the two exes (hard links when possible) and a MARKER
amdhip64_7.dll (tests/package/marker_hiprt.cpp, CMake target whirl-test-marker-hiprt -> <build>/pkg_marker/) into folders of
an exact UTF-8 length and runs them. The marker writes whirl_marker_loaded.txt next to itself when loaded and exports no HIP
function. The exe folder's amdhip64_7.dll is the only one src/hip/runtime_loader.cpp loads, so the driver's runtime is never
loaded and no GPU is touched; every case first checks that the marker is the file next to the exe (else "not run").

  folder                         exe and arguments                                        expected
  150 bytes ASCII                whirl --version, whirl-server --version                  exit 0, no marker
  150 bytes ASCII                whirl devices, whirl-server -m x.gguf (positive control) marker written, exit not 0
  151 bytes ASCII                whirl / whirl-server --version                           exit 2 + message, no marker
  151 bytes ASCII                whirl devices, whirl-server -m x.gguf,
                                 whirl / whirl-server --whirl-gpu-probe                   exit 2, no marker
  153 bytes, Chinese (< 150 ch)  whirl / whirl-server --version                           exit 2 (unit is bytes, not characters)
  150 bytes, Chinese             whirl / whirl-server --version                           exit 0, no marker

Usage: python -I pkg_pathlen_check.py --exe-dir <folder with whirl.exe + whirl-server.exe> --marker <amdhip64_7.dll>
       [--work <short folder>]   (--exe-dir: a build's Release folder or the unpacked release zip)
Exit 0 = every case as expected ("pkg pathlen check: ok (n/n cases)"), 1 = a case failed, 2 = not run (setup: missing
file, the work folder too long to make a 150-byte folder, the marker not in place)."""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INSTALL_MAX = 150   # src/platform/path_budget.h kInstallDirMax (UTF-8 bytes)
MARK = "whirl_marker_loaded.txt"
EXES = ("whirl.exe", "whirl-server.exe")


def u8(s):
    return len(s.encode("utf-8"))


def make_folder(base, nbytes, fill):
    """base + '\\' + name with u8(path) == nbytes; fill is 'a' (1 byte) or a Chinese character (3 bytes) + 'a' padding"""
    room = nbytes - u8(str(base)) - 1
    if room < 1:
        return None
    if fill == "a":
        name = "a" * room
    else:
        k = room // 3
        name = fill * k + "a" * (room - 3 * k)
    p = Path(str(base) + "\\" + name)
    assert u8(str(p)) == nbytes, (p, u8(str(p)))
    return p


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe-dir", required=True)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    exe_dir, marker = Path(a.exe_dir), Path(a.marker)
    for f in [exe_dir / e for e in EXES] + [marker]:
        if not f.is_file():
            print(f"pkg pathlen check: NOT RUN (missing {f})")
            return 2
    msha = hashlib.sha256(marker.read_bytes()).hexdigest()
    base = Path(a.work) if a.work else Path(tempfile.gettempdir()) / "whirl-pkgpath"
    base = Path(os.path.abspath(base))
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    zh = "路"   # 路 (3 UTF-8 bytes, 1 UTF-16 unit)
    folders = {"a150": make_folder(base, INSTALL_MAX, "a"), "a151": make_folder(base, INSTALL_MAX + 1, "a"),
               "z153": make_folder(base, INSTALL_MAX + 3, zh), "z150": make_folder(base, INSTALL_MAX, zh)}
    if any(v is None for v in folders.values()):
        print(f"pkg pathlen check: NOT RUN (work folder {base} is {u8(str(base))} bytes: no room for a {INSTALL_MAX}-byte folder)")
        return 2
    for k, f in folders.items():
        f.mkdir()
        for e in EXES:
            place(exe_dir / e, f / e)
        shutil.copy2(marker, f / "amdhip64_7.dll")
        print(f"folder {k}: {u8(str(f))} UTF-8 bytes, {len(str(f))} characters")
    if u8(str(folders["z153"])) <= INSTALL_MAX or len(str(folders["z153"])) > INSTALL_MAX:
        print("pkg pathlen check: NOT RUN (the Chinese folder must be over 150 bytes but at most 150 characters)")
        return 2
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("WHIRL_", "HIP_", "ROCR_", "HSA_"))}
    env["LOCALAPPDATA"] = str(base / "lad")
    (base / "lad").mkdir()
    fails, n = [], 0

    def run(key, exe, args, want_rc, want_marker, words=()):
        nonlocal n
        n += 1
        f = folders[key]
        dll = f / "amdhip64_7.dll"
        if not dll.is_file() or hashlib.sha256(dll.read_bytes()).hexdigest() != msha:
            print(f"pkg pathlen check: NOT RUN (the marker amdhip64_7.dll is not in {f}: the driver runtime could load)")
            sys.exit(2)
        (f / MARK).unlink(missing_ok=True)
        try:
            p = subprocess.run([str(f / exe), *args], cwd=str(base), env=env, capture_output=True, timeout=120,
                               stdin=subprocess.DEVNULL)
            rc, out = p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")
        except subprocess.TimeoutExpired:
            rc, out = None, "timeout 120 s"
        marked = (f / MARK).exists()
        rc_ok = (rc == want_rc) if want_rc != "nonzero" else (rc not in (None, 0))
        ok = rc_ok and marked == want_marker and all(w in out for w in words)
        name = f"{key} {exe} {' '.join(args)}: exit {rc} (want {want_rc}), marker {int(marked)} (want {int(want_marker)})"
        print(("ok    " if ok else "FAIL  ") + name)
        if not ok:
            fails.append(name)
            print("      " + out.strip().replace("\n", "\n      ")[:800])

    msg = ("bytes in UTF-8", "150")
    for e in EXES:
        run("a150", e, ["--version"], 0, False)
    run("a150", "whirl.exe", ["devices"], "nonzero", True)
    run("a150", "whirl-server.exe", ["-m", "x.gguf"], "nonzero", True)
    for e in EXES:
        run("a151", e, ["--version"], 2, False, msg)
    run("a151", "whirl.exe", ["devices"], 2, False, msg)
    run("a151", "whirl-server.exe", ["-m", "x.gguf"], 2, False, msg)
    for e in EXES:
        run("a151", e, ["--whirl-gpu-probe"], 2, False, msg)
    for e in EXES:
        run("z153", e, ["--version"], 2, False, msg)
    for e in EXES:
        run("z150", e, ["--version"], 0, False)
    shutil.rmtree(base, ignore_errors=True)
    print(f"pkg pathlen check: {'ok' if not fails else 'FAIL'} ({n - len(fails)}/{n} cases)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
