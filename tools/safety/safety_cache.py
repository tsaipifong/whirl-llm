#!/usr/bin/env python
"""arch025-procopt3 (PLAN 9.33 / 9.33b): persistent build cache of the cpu_asan / cpu_tsa steps (tools/safety/safety_step.ps1
-Mode cache). Not cpu_analyze (PLAN 9.33b): its gate reads the per-TU SARIF files, a by-product that an incremental build keeps
from older compiles; the incremental analyze run lost 6 of 22 warning keys in every round (scratch\\a25_procopt3\\DONE_measure3:
its "SARIF not older than its object" rule rejected 4 large TUs whose SARIF cl writes seconds before the object), so cpu_analyze
keeps its clean in-tree build (safety_step.ps1 "mode clean (cpu_analyze never uses the cache, PLAN 9.33b ...)"). Every integration verify runs in a NEW worktree (wt_a25_int5b, int6, int7 ...), so the in-tree build-asan /
build-analyze / build-tsa were always cold (int7: asan build 708 s of 1369 s, analyze 1186 s, tsa 389 s). CMake build dirs are
bound to one source path, so the cache keeps ONE stable mirror of the source tree and the build dirs inside it:

  <cache root>\\src                 the mirror (default <parent of the tree>\\safety_cache\\src; WHIRL_SAFETY_CACHE = the root)
  <cache root>\\src\\build-<kind>    the persistent build dir; <kind>\\.safety_key = the key it was configured with

  safety_cache.py sync <tree> <mirror>          copy into the mirror every file of the tree's list (git ls-files -co
                                                --exclude-standard, or the tree's .safety_mirror_manifest when it is not a git
                                                checkout) whose CONTENT differs; unchanged files keep their mtime, so ninja (mtime +
                                                recorded header deps + command-line hash) rebuilds exactly what sees a change, as a
                                                long-lived checkout would after "git checkout" (git rewrites only changed files).
                                                Files of the previous list that are gone are deleted; nothing else is touched.
  safety_cache.py prepare <kind> <mirror> [--clean]
                                                key = build scripts of the kind + cmake / ninja / MSVC tools / clang-cl / HIP SDK
                                                versions + toolchain env; new or changed key (or --clean) -> wipe build-<kind>
                                                ("mode clean-cache (...)"), same key -> "mode incremental". build_cfg.bat never
                                                re-runs cmake in an existing dir, so a changed -D flag needs the key.
  safety_cache.py forget <kind> <mirror>        drop the key (the next run is clean-cache): after a failed deps check
  safety_cache.py depscheck <build_dir> <src_root> <build.log>
                                                the recorded header deps are usable: every deps = msvc rule file has a
                                                msvc_deps_prefix, the build log has no unparsed /showIncludes note (a localized
                                                prefix that does not match: memory whirl-build-header-deps-broken), and every TU whose
                                                source includes a project header has >= 1 recorded dependency (ninja -t deps).
Exit 0 = ok. Tests: tools/safety/test_safety_cache.py; real cl + ninja proof: tools/safety/cache_real_test.ps1.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANIFEST = ".safety_mirror_manifest"
KEYFILE = ".safety_key"
KINDS = ("asan", "tsa")  # PLAN 9.33b: no "analyze" (it always builds clean, see the header)
COMMON = ["tools/safety/build_cfg.bat", "tools/safety/vcenv.bat", "tools/safety/safety_cache.py"]
KEY_FILES = {
    "asan": COMMON + ["tools/safety/run_asan.bat"],
    "tsa": COMMON + ["tools/safety/tsa/run_tsa_build.bat"],
}
ENV_KEYS = ("WHIRL_VCVARS", "WHIRL_CLANG_CL", "HIP_PATH", "VSLANG", "CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CL", "_CL_", "LINK")


def readb(p):
    with open(p, "rb") as f:
        return f.read()


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def write_manifest(root, files):
    with open(os.path.join(root, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(files)) + "\n")


def read_manifest(root):
    p = os.path.join(root, MANIFEST)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def list_files(tree):
    """the tree's source files (relative, '/'): tracked + untracked-not-ignored for a git checkout, else its manifest"""
    if os.path.exists(os.path.join(tree, ".git")):
        r = subprocess.run(["git", "-C", tree, "ls-files", "-co", "--exclude-standard", "-z"], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"sync: git ls-files failed in {tree}: {r.stderr.decode(errors='replace')[:300]}")
        files = [x for x in r.stdout.decode("utf-8", errors="replace").split("\0") if x]
        return sorted(set(f for f in files if os.path.isfile(os.path.join(tree, f))))
    m = read_manifest(tree)
    if m is None:
        raise SystemExit(f"sync: {tree} is neither a git checkout nor a tree with {MANIFEST}")
    return sorted(set(f for f in m if f != MANIFEST))


def same_content(a, b):
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
    except OSError:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(1 << 20), fb.read(1 << 20)
            if x != y:
                return False
            if not x:
                return True


def sync(tree, mirror):
    t, m = norm(tree), norm(mirror)
    if t == m or m.startswith(t + os.sep) or t.startswith(m + os.sep):
        raise SystemExit(f"sync: refused: the mirror {mirror} and the tree {tree} are the same or nested")
    files = list_files(tree)
    for f in files:
        if f.startswith(("/", "\\")) or ".." in re.split(r"[\\/]", f) or ":" in f:
            raise SystemExit(f"sync: refused: list entry outside the tree: {f!r}")
    os.makedirs(mirror, exist_ok=True)
    old = read_manifest(mirror) or []
    changed = []
    for f in files:
        s, d = os.path.join(tree, f), os.path.join(mirror, f)
        if same_content(s, d):
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d):
            os.remove(d)  # a new file: its mtime is now even when the copy keeps attributes
        shutil.copyfile(s, d)
        changed.append(f)
    keep = set(files)
    deleted = 0
    for f in old:
        if f in keep or ".." in re.split(r"[\\/]", f):
            continue
        d = os.path.join(mirror, f)
        if os.path.isfile(d):
            os.remove(d)
            deleted += 1
    write_manifest(mirror, files)
    return {"files": len(files), "copied": len(changed), "deleted": deleted, "changed": changed}


def _out(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60)
        return (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else f"rc={r.returncode}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({type(e).__name__})"


def msvc_version():
    vsw = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft Visual Studio", "Installer", "vswhere.exe")
    vc = os.environ.get("WHIRL_VCVARS", "")
    inst = ""
    if os.path.exists(vsw):
        inst = _out([vsw, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                     "-property", "installationPath"])
    base = os.path.dirname(os.path.dirname(os.path.dirname(vc))) if vc else inst  # ...\VC\Auxiliary\Build\vcvars64.bat -> install
    for p in (os.path.join(base, "VC", "Auxiliary", "Build", "Microsoft.VCToolsVersion.default.txt"),):
        if os.path.exists(p):
            return inst + " tools " + readb(p).decode("utf-8", errors="replace").strip()
    return inst + " tools unknown"


def probe_tools():
    hip = os.environ.get("HIP_PATH", "")
    clang = os.environ.get("WHIRL_CLANG_CL") or (os.path.join(hip, "bin", "clang-cl.exe") if hip else "clang-cl")
    return {
        "cmake": _out(["cmake", "--version"]),
        "ninja": _out(["ninja", "--version"]),
        "msvc": msvc_version(),
        "clang-cl": _out([clang, "--version"]),
        "hip": hip + " | " + (_out([os.path.join(hip, "bin", "clang.exe"), "--version"]) if hip else "none"),
        "env": ";".join(f"{k}={os.environ.get(k, '')}" for k in ENV_KEYS),
    }


def tool_key(kind, repo, probe=None):
    if kind not in KINDS:
        raise SystemExit(f"key: unknown kind {kind!r}")
    p = probe if probe is not None else probe_tools()
    lines = [f"kind {kind}"]
    for rel in KEY_FILES[kind]:
        fp = os.path.join(repo, rel)
        h = hashlib.sha256(readb(fp)).hexdigest() if os.path.exists(fp) else "missing"
        lines.append(f"file {rel} {h}")
    for k in sorted(p):
        lines.append(f"tool {k} {p[k]}")
    text = "\n".join(lines) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), text


def build_dir(kind, mirror):
    if kind not in KINDS:
        raise SystemExit(f"prepare: refused: unknown kind {kind!r}")
    return os.path.join(mirror, "build-" + kind)


def prepare(kind, mirror, key, force_clean=False, key_text=""):
    b = build_dir(kind, mirror)
    kf = os.path.join(b, KEYFILE)
    old = readb(kf).decode("utf-8", errors="replace").split("\n", 1)[0].strip() if os.path.exists(kf) else None
    if force_clean:
        why = "forced clean"
    elif old is None:
        why = "new cache" if not os.path.exists(b) else "no key (new, or the last deps check failed)"
    elif old != key:
        why = "key changed (build script or toolchain)"
    else:
        return "incremental", "key unchanged"
    if os.path.exists(b):
        if norm(os.path.dirname(b)) != norm(mirror):
            raise SystemExit(f"prepare: refused to wipe {b}")
        shutil.rmtree(b)
    os.makedirs(b)
    with open(kf, "w", encoding="utf-8", newline="\n") as f:
        f.write(key + "\n" + key_text)
    return "clean-cache", why


def forget(kind, mirror):
    kf = os.path.join(build_dir(kind, mirror), KEYFILE)
    if os.path.exists(kf):
        os.remove(kf)


DEPS_REC = re.compile(r"^(\S.*?): #deps (\d+), deps mtime \d+ \((\w+)\)")
PROJECT_INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"|<whirl/)', re.M)
EN_NOTE = "Note: including file:"
ZH_NOTE = "注意: 包含檔案:"


def obj_source(obj, src_root):
    """CMakeFiles/<target>.dir/<path relative to the source dir>.obj -> that source file (None for files outside the tree)"""
    m = re.match(r"^(?:.*/)?CMakeFiles/[^/]+\.dir/(.+)\.(?:obj|o)$", obj.replace("\\", "/"))
    if not m:
        return None
    rel = m.group(1)
    if rel.startswith("__/") or re.match(r"^[A-Za-z]_/", rel):
        return None
    p = os.path.join(src_root, rel)
    return p if os.path.isfile(p) else None


def depscheck(bdir, src_root, build_log, deps_text=None):
    probs = []
    rules = os.path.join(bdir, "CMakeFiles", "rules.ninja")
    raw = readb(rules) if os.path.exists(rules) else b""
    prefixes = []
    if b"deps = msvc" in raw:
        m = re.search(rb"^msvc_deps_prefix = (.*?)\r?$", raw, re.M)
        if not m or not m.group(1).strip():
            probs.append("rules.ninja has deps = msvc rules but no msvc_deps_prefix (cl /showIncludes notes are not parsed)")
        else:
            prefixes.append(m.group(1).strip())
    if os.path.exists(build_log):
        lb = readb(build_log)
        notes = [EN_NOTE.encode(), ZH_NOTE.encode("utf-8"), ZH_NOTE.encode("cp950")] + prefixes
        for ln in lb.splitlines():
            s = ln.lstrip()
            if any(s.startswith(n) for n in notes if n):
                probs.append("unparsed /showIncludes note in the build log (the msvc_deps_prefix does not match cl's output, so no "
                             f"header dependency was recorded): {s[:80].decode('utf-8', errors='replace')}")
                break
    if deps_text is None:
        r = subprocess.run(["ninja", "-C", bdir, "-t", "deps"], capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            probs.append(f"ninja -t deps failed ({r.returncode}): {r.stderr.strip()[:200]}")
            return probs
        deps_text = r.stdout
    bad, n = [], 0
    for ln in deps_text.splitlines():
        m = DEPS_REC.match(ln)
        if not m:
            continue
        obj, cnt = m.group(1), int(m.group(2))
        src = obj_source(obj, src_root)
        if src is None:
            continue
        n += 1
        if cnt == 0:
            try:
                txt = readb(src).decode("utf-8", errors="replace")
            except OSError:
                continue
            if PROJECT_INCLUDE.search(txt):
                bad.append(os.path.relpath(src, src_root).replace("\\", "/"))
    if bad:
        probs.append(f"{len(bad)} TU(s) include a project header but have no recorded dependency: " + ", ".join(bad[:5])
                     + (" ..." if len(bad) > 5 else ""))
    return probs


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "sync" and len(argv) == 4:
        r = sync(argv[2], argv[3])
        print(f"sync: {r['files']} files, {r['copied']} copied, {r['deleted']} deleted"
              + (": " + ", ".join(r["changed"][:8]) + (" ..." if len(r["changed"]) > 8 else "") if 0 < r["copied"] < 200 else ""))
        return 0
    if cmd == "prepare" and len(argv) in (4, 5):
        key, text = tool_key(argv[2], argv[3])
        mode, why = prepare(argv[2], argv[3], key, force_clean=(len(argv) == 5 and argv[4] == "--clean"), key_text=text)
        print(f"mode {mode} ({why}; key {key[:12]})")
        return 0
    if cmd == "forget" and len(argv) == 4:
        forget(argv[2], argv[3])
        print(f"forget: {argv[2]} key removed (next run clean-cache)")
        return 0
    if cmd == "key" and len(argv) == 4:
        key, text = tool_key(argv[2], argv[3])
        print(text + key)
        return 0
    if cmd == "depscheck" and len(argv) == 5:
        p = depscheck(argv[2], argv[3], argv[4])
        for x in p:
            print("DEPS:", x)
        print("depscheck: " + ("ok" if not p else f"FAIL ({len(p)} problem(s))"))
        return 1 if p else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
