"""Execution sandbox for model-written code (feat027-taskacc).

Every program runs in its own child process:
  * `python -I -S` (no user site, no environment variables, no site-packages), below-normal priority,
    no console window;
  * cwd = a fresh temp directory under the sandbox root (default scratch\\feat027\\sandbox), deleted after;
  * a minimal environment (SYSTEMROOT, TEMP/TMP = the temp directory, nothing else);
  * a wall-clock timeout; on Windows a Job Object with a per-process memory limit, an active-process
    limit of 1 (the program cannot start children) and kill-on-close (nothing survives the job);
  * inside the child, before the model code runs (as in openai/human-eval reliability_guard):
    sockets / DNS raise, subprocess / os.system / os.popen / os.startfile / os.kill are disabled,
    destructive file functions (remove, unlink, rmdir, rename, replace, chmod, shutil.rmtree / move)
    are disabled, open() for writing is allowed only inside the temp directory, exit() / quit() are
    removed, and sys.exit before the end means "not passed".
A program passes only when it exits with code 0 AND prints a per-run random nonce that the model code
never sees (it comes through stdin and is read before the model code runs).
Limits of this sandbox (documented, not hidden): the network block is at the Python level, not an OS
firewall rule (changing the firewall is an owner-only system setting); ctypes / _winapi are still
importable by a deliberately hostile program. The eval runs our own product models on public coding
tasks; the guard is against accidents (infinite loops, memory bombs, stray deletes, network calls).
"""
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_ROOT = r"C:\ClaudeWorkspace\whirl-work\scratch\feat027\sandbox"
# the real interpreter, not a venv redirector (a redirector would start a child process, which the job forbids)
PY = getattr(sys, "_base_executable", None) or sys.executable

RUNNER = r'''
import sys, os, builtins, io
_nonce = sys.stdin.readline().strip()
_kind, _ep = sys.argv[1], sys.argv[2]
del sys.argv[1:]
_here = os.path.realpath(os.getcwd())
_src = open("solution.py", encoding="utf-8").read()
_tst = open("test.py", encoding="utf-8").read()
_write, _flush, _exit = sys.stdout.write, sys.stdout.flush, os._exit

def _guard():
    import socket, subprocess, shutil, faulthandler
    faulthandler.disable()
    def _no_net(*a, **k):
        raise OSError("network is disabled in the taskacc sandbox")
    class _NoSocket(socket.socket):
        def __init__(self, *a, **k):
            raise OSError("network is disabled in the taskacc sandbox")
    socket.socket = _NoSocket
    for n in ("create_connection", "create_server", "getaddrinfo", "gethostbyname", "gethostbyname_ex",
              "gethostbyaddr", "socketpair", "fromfd"):
        if hasattr(socket, n):
            setattr(socket, n, _no_net)
    def _no_proc(*a, **k):
        raise PermissionError("process creation is disabled in the taskacc sandbox")
    subprocess.Popen = _no_proc
    for n in ("run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"):
        setattr(subprocess, n, _no_proc)
    def _no_fs(*a, **k):
        raise PermissionError("this file operation is disabled in the taskacc sandbox")
    for n in ("system", "popen", "startfile", "kill", "killpg", "spawnl", "spawnv", "spawnve", "execv", "execve",
              "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace", "truncate", "chmod",
              "chown", "lchown", "fchmod", "fchown", "chroot", "fork", "forkpty", "putenv", "unsetenv",
              "chdir", "fchdir", "setuid", "symlink", "link"):
        if hasattr(os, n):
            setattr(os, n, _no_fs)
    for n in ("rmtree", "move", "chown", "copytree"):
        setattr(shutil, n, _no_fs)
    _open = builtins.open
    def _safe_open(file, mode="r", *a, **k):
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            if not isinstance(file, int):
                p = os.path.realpath(os.fspath(file))
                if os.path.commonpath([p, _here]) != _here:
                    raise PermissionError("writing outside the sandbox directory is disabled")
        return _open(file, mode, *a, **k)
    builtins.open = _safe_open
    io.open = _safe_open
    builtins.exit = None
    builtins.quit = None
    for m in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[m] = None

_guard()
_g = {"__name__": "taskacc_solution", "__builtins__": builtins}
exec(compile(_src, "solution.py", "exec"), _g)
exec(compile(_tst, "test.py", "exec"), _g)
if _kind == "he":
    _g["check"](_g[_ep])
_write("\n" + _nonce + "\n")
_flush()
_exit(0)
'''

# ------------------------------------------------------------------ Windows Job Object (ctypes, kernel32)
_job_api = None


def _job():
    global _job_api
    if _job_api is not None or os.name != "nt":
        return _job_api
    import ctypes
    from ctypes import wintypes as wt

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                      "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
    k32.SetInformationJobObject.restype = wt.BOOL
    k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    k32.AssignProcessToJobObject.restype = wt.BOOL
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.TerminateJobObject.restype = wt.BOOL
    k32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
    k32.CloseHandle.restype = wt.BOOL
    k32.CloseHandle.argtypes = [wt.HANDLE]
    _job_api = (ctypes, k32, EXT)
    return _job_api


def _make_job(mem_mb):
    api = _job()
    if api is None:
        return None
    ctypes, k32, EXT = api
    h = k32.CreateJobObjectW(None, None)
    if not h:
        return None
    info = EXT()
    # PROCESS_MEMORY | ACTIVE_PROCESS | DIE_ON_UNHANDLED_EXCEPTION | KILL_ON_JOB_CLOSE
    info.BasicLimitInformation.LimitFlags = 0x100 | 0x8 | 0x400 | 0x2000
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = int(mem_mb) << 20
    if not k32.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(h)
        return None
    return h


def _assign(h, proc):
    if h is None:
        return False
    _, k32, _ = _job()
    return bool(k32.AssignProcessToJobObject(h, int(proc._handle)))


def _close_job(h, kill):
    if h is None:
        return
    _, k32, _ = _job()
    if kill:
        k32.TerminateJobObject(h, 1)
    k32.CloseHandle(h)


# ------------------------------------------------------------------ program assembly
def import_lines(code):
    out = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("import ") or (s.startswith("from ") and " import " in s):
            if ln == s:  # top level only
                out.append(ln)
    return out


def assemble(task, code):
    """Return (solution_source, test_source) for a task and the extracted model code."""
    import re
    ep = task.get("entry_point") or ""
    if task["kind"] == "he":
        has_def = bool(re.search(r"^\s*def\s+" + re.escape(ep) + r"\s*\(", code, flags=re.M))
        if has_def:
            sol = "\n".join(import_lines(task["prompt_code"])) + "\n" + code
        else:  # a bare body / continuation: put it under the prompt's signature + docstring
            body = code if code.startswith((" ", "\t")) else "\n".join("    " + l for l in code.splitlines())
            sol = task["prompt_code"].rstrip() + "\n" + body
        return sol, task["test"]
    test = "\n".join(task.get("test_imports") or []) + "\n" + task["test"]
    if task["kind"] == "lc":   # feat027-longacc: the repository module is importable and in scope
        import longctx  # noqa: PLC0415
        fut = [l for l in code.splitlines() if l.strip().startswith("from __future__ import")]
        body = "\n".join(l for l in code.splitlines() if not l.strip().startswith("from __future__ import"))
        return "\n".join(fut) + ("\n" if fut else "") + longctx.lc_prelude(task) + body, test
    return code, test


def run_program(solution, test, kind, entry_point, timeout=10.0, mem_mb=1024, root=DEFAULT_ROOT, keep=False):
    """Run one program; return {"status", "passed", "rc", "seconds", "stderr", "stdout"}."""
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix="t_", dir=root)
    nonce = "TASKACC_OK_" + secrets.token_hex(16)
    try:
        with open(os.path.join(d, "solution.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(solution)
        with open(os.path.join(d, "test.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(test)
        with open(os.path.join(d, "runner.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(RUNNER)
        env = {"TEMP": d, "TMP": d, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        for k in ("SYSTEMROOT", "SystemRoot", "WINDIR"):
            if k in os.environ:
                env[k] = os.environ[k]
        flags = 0
        if os.name == "nt":
            flags = 0x00004000 | 0x08000000  # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
        job = _make_job(mem_mb)
        t0 = time.time()
        p = subprocess.Popen([PY, "-I", "-S", "-X", "utf8", "runner.py", kind, entry_point or ""],
                             cwd=d, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             creationflags=flags)
        jobbed = _assign(job, p)
        timed_out = False
        try:
            out, err = p.communicate(input=(nonce + "\n").encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _close_job(job, kill=True)
            job = None
            p.kill()
            out, err = p.communicate()
        finally:
            _close_job(job, kill=True)
        sec = time.time() - t0
        out = out.decode("utf-8", "replace")
        err = err.decode("utf-8", "replace")
        ok = (not timed_out) and p.returncode == 0 and (nonce in out.splitlines())
        if ok:
            status = "pass"
        elif timed_out:
            status = "timeout"
        else:
            last = [l for l in err.strip().splitlines() if l.strip()]
            last = last[-1] if last else ""
            if last.startswith("AssertionError") or "AssertionError" in last:
                status = "fail"
            elif "MemoryError" in last:
                status = "memory"
            elif p.returncode == 0 or not last:
                status = "no_result"   # exited without running the tests to the end (sys.exit, os._exit, ...)
            else:
                status = "error"
        return {"status": status, "passed": ok, "rc": p.returncode, "seconds": round(sec, 3),
                "jobbed": jobbed, "stderr": err[-1500:], "stdout": out.replace(nonce, "<nonce>")[-500:]}
    finally:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


def grade(task, code, **kw):
    if code is None:
        return {"status": "no_code", "passed": False, "rc": None, "seconds": 0.0, "jobbed": False, "stderr": "", "stdout": ""}
    sol, test = assemble(task, code)
    return run_program(sol, test, task["kind"], task.get("entry_point") or "", **kw)
