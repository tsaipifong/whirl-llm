#!/usr/bin/env python
"""arch025-procopt3 (PLAN 9.33): block / pass tests of tools/safety/safety_cache.py, the persistent build cache of the cpu_asan /
cpu_tsa steps (cpu_analyze: always clean, PLAN 9.33b). CPU only, no compiler: temp source trees, a temp git repo, synthetic rules.ninja / build logs /
"ninja -t deps" text. Run: python tools/safety/test_safety_cache.py   (exit 0 = all pass)

What must hold (a skipped rebuild that misses a changed header is a false PASS):
  sync      only files whose CONTENT changed get a new mtime in the mirror (unchanged files keep theirs, so ninja's mtime + deps
            tracking rebuilds exactly the TUs that see a change); a removed file is removed; build-* directories and other
            files the sync did not write are never touched; untracked-but-not-ignored files are included, ignored files are not.
  key       any change of a build script, of a tool version (cmake, ninja, MSVC tools, clang-cl, HIP SDK) or of the
            toolchain environment changes the key (CMake never re-configures an existing build dir: build_cfg.bat skips
            cmake when build.ninja exists, so new -D flags would be ignored without the key).
  prepare   new / changed key or --clean -> the build dir is wiped ("clean-cache"); same key -> "incremental".
  depscheck the header dependencies ninja recorded are usable: deps = msvc rules have msvc_deps_prefix, the build log has no
            unparsed /showIncludes line (a localized prefix that does not match), every TU whose source includes a project
            header has >= 1 recorded dependency.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safety_cache as sc  # noqa: E402


def write(p, text, mtime=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if mtime is not None:
        os.utime(p, (mtime, mtime))


def run(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sc.main(["safety_cache.py", *argv])
    return rc, buf.getvalue()


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sccache_")
        self.src = os.path.join(self.tmp, "tree")
        self.mir = os.path.join(self.tmp, "cache", "src")
        write(os.path.join(self.src, "include", "a.h"), "int a();\n")
        write(os.path.join(self.src, "include", "b.h"), "int b();\n")
        write(os.path.join(self.src, "src", "a.cpp"), '#include "a.h"\n')
        write(os.path.join(self.src, "src", "b.cpp"), '#include "b.h"\n')
        write(os.path.join(self.src, "CMakeLists.txt"), "project(x)\n")
        # a manifest-listed source (no git): the file list of a tree that is not a git checkout
        sc.write_manifest(self.src, ["CMakeLists.txt", "include/a.h", "include/b.h", "src/a.cpp", "src/b.cpp"])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_sync_copies_everything(self):
        r = sc.sync(self.src, self.mir)
        self.assertEqual(r["copied"], 5, r)
        self.assertEqual(open(os.path.join(self.mir, "src", "a.cpp")).read(), '#include "a.h"\n')

    def test_unchanged_files_keep_mtime_changed_header_gets_new(self):
        sc.sync(self.src, self.mir)
        old = time.time() - 3600
        for rel in ("include/a.h", "include/b.h", "src/a.cpp", "src/b.cpp", "CMakeLists.txt"):
            os.utime(os.path.join(self.mir, rel), (old, old))
        # a fresh checkout: every source file gets a new mtime, only b.h has new content
        for rel in ("include/a.h", "src/a.cpp", "src/b.cpp", "CMakeLists.txt"):
            os.utime(os.path.join(self.src, rel), None)
        write(os.path.join(self.src, "include", "b.h"), "int b(int);\n")
        r = sc.sync(self.src, self.mir)
        self.assertEqual(r["copied"], 1, r)
        self.assertEqual(r["changed"], ["include/b.h"])
        self.assertLess(os.path.getmtime(os.path.join(self.mir, "include", "a.h")), old + 1)
        self.assertGreater(os.path.getmtime(os.path.join(self.mir, "include", "b.h")), old + 1000)
        self.assertEqual(open(os.path.join(self.mir, "include", "b.h")).read(), "int b(int);\n")

    def test_same_size_content_change_is_seen(self):
        sc.sync(self.src, self.mir)
        write(os.path.join(self.src, "include", "a.h"), "int z();\n")  # same length as "int a();"
        r = sc.sync(self.src, self.mir)
        self.assertEqual(r["changed"], ["include/a.h"])

    def test_removed_file_removed_build_dirs_untouched(self):
        sc.sync(self.src, self.mir)
        write(os.path.join(self.mir, "build-asan", "x.obj"), "obj")
        write(os.path.join(self.mir, "stray.txt"), "not ours")
        os.remove(os.path.join(self.src, "src", "b.cpp"))
        sc.write_manifest(self.src, ["CMakeLists.txt", "include/a.h", "include/b.h", "src/a.cpp"])
        r = sc.sync(self.src, self.mir)
        self.assertEqual(r["deleted"], 1, r)
        self.assertFalse(os.path.exists(os.path.join(self.mir, "src", "b.cpp")))
        self.assertTrue(os.path.exists(os.path.join(self.mir, "build-asan", "x.obj")))
        self.assertTrue(os.path.exists(os.path.join(self.mir, "stray.txt")))

    def test_refuses_nested_paths(self):
        with self.assertRaises(SystemExit):
            sc.sync(self.src, os.path.join(self.src, "cache"))
        with self.assertRaises(SystemExit):
            sc.sync(self.src, self.src)

    def test_refuses_list_entry_outside_tree(self):
        sc.write_manifest(self.src, ["CMakeLists.txt", "../evil.txt"])
        with self.assertRaises(SystemExit):
            sc.sync(self.src, self.mir)

    def test_git_tree_lists_tracked_and_untracked_not_ignored(self):
        g = os.path.join(self.tmp, "gtree")
        write(os.path.join(g, ".gitignore"), "/build-*/\n*.obj\n")
        write(os.path.join(g, "a.cpp"), "x\n")
        write(os.path.join(g, "new_untracked.h"), "y\n")
        write(os.path.join(g, "build-asan", "a.obj"), "o\n")
        write(os.path.join(g, "z.obj"), "o\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        subprocess.run(["git", "init", "-q", g], check=True, env=env)
        subprocess.run(["git", "-C", g, "add", ".gitignore", "a.cpp"], check=True, env=env)
        files = sc.list_files(g)
        self.assertEqual(sorted(files), [".gitignore", "a.cpp", "new_untracked.h"])

    def test_cli_sync_line(self):
        rc, out = run("sync", self.src, self.mir)
        self.assertEqual(rc, 0, out)
        self.assertIn("sync: 5 files, 5 copied, 0 deleted", out)


class KeyPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sckey_")
        self.repo = os.path.join(self.tmp, "src")
        for rel in sc.KEY_FILES["asan"] + sc.KEY_FILES["tsa"]:
            write(os.path.join(self.repo, rel), "rem " + rel + "\n")
        self.probe = {"cmake": "cmake version 4.4.2", "ninja": "1.13.2", "msvc": "14.44.35207", "clang-cl": "clang version 21.0.0",
                      "hip": "C:/Program Files/AMD/ROCm/7.2", "env": ""}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def key(self, kind="asan", **over):
        p = dict(self.probe, **over)
        return sc.tool_key(kind, self.repo, probe=p)[0]

    def test_key_stable(self):
        self.assertEqual(self.key(), self.key())

    def test_key_changes_with_build_script(self):
        k0 = self.key()
        write(os.path.join(self.repo, "tools", "safety", "run_asan.bat"), "rem new -D flag\n")
        self.assertNotEqual(k0, self.key())

    def test_key_changes_with_each_tool_version(self):
        k0 = self.key()
        for name, val in (("cmake", "cmake version 4.5.0"), ("ninja", "1.14.0"), ("msvc", "14.45.1"), ("clang-cl", "clang version 22"),
                          ("hip", "C:/Program Files/AMD/ROCm/7.3"), ("env", "WHIRL_VCVARS=x")):
            self.assertNotEqual(k0, self.key(**{name: val}), name)

    def test_key_differs_per_kind(self):
        self.assertNotEqual(self.key("asan"), self.key("tsa"))

    def test_prepare_new_then_incremental_then_changed_key(self):
        k = self.key()
        mode, why = sc.prepare("asan", self.repo, k)
        self.assertEqual(mode, "clean-cache", why)
        b = os.path.join(self.repo, "build-asan")
        write(os.path.join(b, "x.obj"), "o")
        mode, why = sc.prepare("asan", self.repo, k)
        self.assertEqual(mode, "incremental", why)
        self.assertTrue(os.path.exists(os.path.join(b, "x.obj")))
        mode, why = sc.prepare("asan", self.repo, self.key(cmake="cmake version 9"))
        self.assertEqual(mode, "clean-cache")
        self.assertIn("key changed", why)
        self.assertFalse(os.path.exists(os.path.join(b, "x.obj")))

    def test_prepare_force_clean_and_forget(self):
        k = self.key()
        sc.prepare("asan", self.repo, k)
        write(os.path.join(self.repo, "build-asan", "x.obj"), "o")
        mode, why = sc.prepare("asan", self.repo, k, force_clean=True)
        self.assertEqual(mode, "clean-cache")
        self.assertFalse(os.path.exists(os.path.join(self.repo, "build-asan", "x.obj")))
        sc.forget("asan", self.repo)  # a failed deps check: the next run must be clean
        mode, why = sc.prepare("asan", self.repo, k)
        self.assertEqual(mode, "clean-cache")

    def test_prepare_refuses_build_dir_outside_mirror(self):
        with self.assertRaises(SystemExit):
            sc.prepare("../x", self.repo, "k")

    def test_analyze_is_not_a_cache_kind(self):  # PLAN 9.33b: cpu_analyze always builds clean (incremental lost 6 of 22 keys)
        self.assertEqual(sc.KINDS, ("asan", "tsa"))
        self.assertNotIn("analyze", sc.KEY_FILES)
        with self.assertRaises(SystemExit):
            sc.prepare("analyze", self.repo, "k")
        with self.assertRaises(SystemExit):
            sc.tool_key("analyze", self.repo, probe=self.probe)


RULES_OK = ("# CMAKE generated file\nmsvc_deps_prefix = Note: including file: \n\nrule CXX_COMPILER__whirl_tests_unscanned_Release\n"
            "  deps = msvc\n  command = cl.exe /nologo /TP $DEFINES $INCLUDES $FLAGS /showIncludes /Fo$out /Fd$TARGET_COMPILE_PDB /FS -c $in\n")
RULES_NOPREFIX = RULES_OK.replace("msvc_deps_prefix = Note: including file: \n", "")


class DepsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="scdeps_")
        self.src = os.path.join(self.tmp, "src")
        self.b = os.path.join(self.src, "build-asan")
        write(os.path.join(self.src, "src", "a.cpp"), '#include "whirl/a.h"\nint f(){return 0;}\n')
        write(os.path.join(self.src, "src", "sys.cpp"), "#include <vector>\n")  # system headers only: 0 recorded deps is right
        write(os.path.join(self.b, "CMakeFiles", "rules.ninja"), RULES_OK)
        write(os.path.join(self.b, "logs", "build.log"), "[1/2] Building CXX object CMakeFiles/t.dir/src/a.cpp.obj\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def deps(self, a_n):
        txt = (f"CMakeFiles/t.dir/src/a.cpp.obj: #deps {a_n}, deps mtime 1 (VALID)\n" + ("    ../include/whirl/a.h\n" if a_n else "") + "\n"
               "CMakeFiles/t.dir/src/sys.cpp.obj: #deps 0, deps mtime 1 (VALID)\n\n")
        return txt

    def check(self, deps_text):
        return sc.depscheck(self.b, self.src, os.path.join(self.b, "logs", "build.log"), deps_text=deps_text)

    def test_good_deps_pass(self):
        self.assertEqual(self.check(self.deps(1)), [])

    def test_tu_with_project_include_and_no_deps_fails(self):
        p = self.check(self.deps(0))
        self.assertTrue(any("src/a.cpp" in x for x in p), p)

    def test_missing_prefix_fails(self):
        write(os.path.join(self.b, "CMakeFiles", "rules.ninja"), RULES_NOPREFIX)
        p = self.check(self.deps(1))
        self.assertTrue(any("msvc_deps_prefix" in x for x in p), p)

    def test_unparsed_include_note_in_log_fails(self):
        for note in ("Note: including file:  C:\\x\\include\\whirl\\a.h\n", "注意: 包含檔案:  C:\\x\\a.h\n"):
            write(os.path.join(self.b, "logs", "build.log"), "[1/2] Building CXX object x.obj\n" + note)
            p = self.check(self.deps(1))
            self.assertTrue(any("unparsed" in x for x in p), (note, p))

    def test_unparsed_note_in_cp950_bytes_fails(self):
        with open(os.path.join(self.b, "logs", "build.log"), "wb") as f:
            f.write(b"[1/2] Building CXX object x.obj\r\n" + "注意: 包含檔案:  C:\\x\\a.h\r\n".encode("cp950"))
        p = self.check(self.deps(1))
        self.assertTrue(any("unparsed" in x for x in p), p)

    def test_object_outside_tree_ignored(self):
        txt = self.deps(1) + "CMakeFiles/t.dir/C_/sdk/x.cpp.obj: #deps 0, deps mtime 1 (VALID)\n\n"
        self.assertEqual(self.check(txt), [])


class GateWiringTests(unittest.TestCase):
    """release_gate.ps1: strict / release runs keep the clean in-tree build; the strictness test is the same as the retry rule's"""

    def test_strict_runs_clean(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gates", "release_gate.ps1")
        txt = open(p, encoding="utf-8").read()
        import re
        a = re.search(r"^\$safetyStrict = (.+)$", txt, re.M)
        b = re.search(r"^\$strictRun = (.+)$", txt, re.M)
        self.assertTrue(a and b, "both strictness lines present")
        self.assertEqual(a.group(1).strip(), b.group(1).strip())
        self.assertRegex(txt, r'\$safetyMode = if \(\$safetyStrict\) \{ "clean" \}')
        self.assertRegex(txt, r"\$safetyEnv = @\{[^}]*WHIRL_SAFETY_MODE = \$safetyMode")

    def test_step_default_is_clean(self):  # safety_step.ps1 alone (no gate, no env) builds in the tree as before
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safety_step.ps1")
        txt = open(p, encoding="utf-8").read()
        self.assertIn('{ $env:WHIRL_SAFETY_MODE } else { "clean" }', txt)
        for k in ("asan", "tsa", "analyze"):
            self.assertIn(f'"cpu {k}: ok ($modeNote; ', txt)
            self.assertIn(f'"cpu {k}: FAIL ($modeNote; ', txt)


    def test_gate_passes_cache_env_through(self):  # 9.33b: release_gate removes every WHIRL_* but WHIRL_GATE_* before a step, so
        # WHIRL_SAFETY_CACHE_CLEAN / WHIRL_SAFETY_CACHE / WHIRL_SAFETY_SYNC_FROM must ride in $safetyEnv (measure3 p3_delta6 synced 0 files)
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gates", "release_gate.ps1")
        txt = open(p, encoding="utf-8").read()
        import re
        m = re.search(r"^\$safetyEnv = @\{(.*)\}$", txt, re.M)
        self.assertTrue(m)
        for k in ("WHIRL_SAFETY_CACHE_CLEAN", "WHIRL_SAFETY_CACHE", "WHIRL_SAFETY_SYNC_FROM"):
            self.assertIn(f"{k} = $env:{k}", m.group(1))
        self.assertLess(txt.index("$safetyEnv = @{"), txt.index('$_.Name -like "WHIRL_*"'))  # read before the scrub

    def test_analyze_never_uses_the_cache(self):  # PLAN 9.33b: the cache serves cpu_asan / cpu_tsa only
        here = os.path.dirname(os.path.abspath(__file__))
        txt = open(os.path.join(here, "safety_step.ps1"), encoding="utf-8").read()
        self.assertIn('if ($Mode -eq "cache" -and $Kind -in @("asan", "tsa")) {', txt)
        self.assertIn('mode clean (cpu_analyze never uses the cache, PLAN 9.33b', txt)
        self.assertNotIn("WHIRL_ANALYZE_INCREMENTAL", txt)
        bat = open(os.path.join(here, "run_analyze.bat"), encoding="utf-8").read()
        self.assertNotIn("--incremental", bat)
        self.assertNotIn("analyze_uptodate", bat)
        ac = open(os.path.join(here, "analyze_check.py"), encoding="utf-8").read()
        self.assertNotIn("--incremental", ac)


if __name__ == "__main__":
    r = unittest.main(exit=False, verbosity=1).result
    n = r.testsRun
    bad = len(r.failures) + len(r.errors)
    print(f"safety cache test: {n - bad}/{n} ok")
    sys.exit(1 if bad else 0)
