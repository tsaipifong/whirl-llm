@echo off
rem Stand-in for whirl-server-gate.exe used by golden_selftest.ps1 (no GPU). SPDX-License-Identifier: Apache-2.0
rem Called as: --exe SERVER --kind whirl --model MODEL --suite golden ... --server-env WHIRL_MODE=MODE ...
rem FAKEGATE_FAIL=1: one in-run check fails (exit 1); FAKEGATE_NOPASS=1: exit 0 without the ALL PASS line;
rem FAKEGATE_H=HEX: the hash of srv.sessA.t2.ssd (default 1111111111111111).
rem arch024-S10: like the real gate, writes a server log into --work DIR (server_fake.log); FAKEGATE_NOLOG=1: none;
rem FAKEGATE_LOGLINE=TEXT: one more line in it (an "E owner check:" or a WDDM line for the server log scan cases).
if not defined FAKEGATE_H set FAKEGATE_H=1111111111111111
set "FG_WORK="
:fg_args
if "%~1"=="" goto fg_args_done
if "%~1"=="--work" set "FG_WORK=%~2"
shift
goto fg_args
:fg_args_done
if not defined FG_WORK goto fg_log_done
if not exist "%FG_WORK%" mkdir "%FG_WORK%"
if "%FAKEGATE_NOLOG%"=="1" goto fg_log_done
echo 2026-01-01 00:00:00.000 I fake server: listening> "%FG_WORK%\server_fake.log"
if defined FAKEGATE_LOGLINE echo %FAKEGATE_LOGLINE%>> "%FG_WORK%\server_fake.log"
:fg_log_done
echo G header format=1 suite=server
echo G srv.short status=200 n=48 prompt=40 cached=0 finish=length h=aaaaaaaaaaaaaaaa
echo G srv.sessA.t2.ram status=200 n=32 prompt=3100 cached=3050 finish=length h=1111111111111111
echo G srv.sessA.t2.ssd status=200 n=32 prompt=3100 cached=3050 finish=length h=%FAKEGATE_H%
if "%FAKEGATE_FAIL%"=="1" (
  echo   [FAIL] golden: turn 2 after the SSD restore == after the RAM restore
  echo == whirl-server-gate: 3 passed, 1 failed
  exit /b 1
)
echo   [PASS] golden: turn 2 after the SSD restore == after the RAM restore
if "%FAKEGATE_NOPASS%"=="1" exit /b 0
echo == whirl-server-gate: 4 passed, 0 failed (ALL PASS)
exit /b 0
