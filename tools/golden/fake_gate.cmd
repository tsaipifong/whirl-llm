@echo off
rem Stand-in for whirl-server-gate.exe used by golden_selftest.ps1 (no GPU). SPDX-License-Identifier: Apache-2.0
rem Called as: --exe SERVER --kind whirl --model MODEL --suite golden ... --work DIR --server-env WHIRL_MODE=MODE ...
rem FAKEGATE_FAIL=1: one in-run check fails (exit 1); FAKEGATE_NOPASS=1: exit 0 without the ALL PASS line;
rem FAKEGATE_H=HEX: the hash of srv.sessA.t2.ssd (default 1111111111111111).
rem FAKEGATE_TIER (arch024-S0b2): the two tier-restore checks fail (exit 1) and the work dir gets a server log:
rem   off = its "adjusted: ... host tiers are off" line (the server started with too little RAM available);
rem   nolog = a log without that line; mixed = tiers off, and a check that is not a tier check fails too.
if not defined FAKEGATE_H set FAKEGATE_H=1111111111111111
set "FG_WORK="
:args
if "%~1"=="" goto argsdone
if "%~1"=="--work" set "FG_WORK=%~2"
shift
goto args
:argsdone
echo G header format=1 suite=server
echo G srv.short status=200 n=48 prompt=40 cached=0 finish=length h=aaaaaaaaaaaaaaaa
echo G srv.sessA.t2.ram status=200 n=32 prompt=3100 cached=3050 finish=length h=1111111111111111
echo G srv.sessA.t2.ssd status=200 n=32 prompt=3100 cached=3050 finish=length h=%FAKEGATE_H%
if defined FAKEGATE_TIER goto tier
if "%FAKEGATE_FAIL%"=="1" goto fail
echo   [PASS] golden: turn 2 after the SSD restore == after the RAM restore
if "%FAKEGATE_NOPASS%"=="1" exit /b 0
echo == whirl-server-gate: 4 passed, 0 failed (ALL PASS)
exit /b 0
:fail
echo   [FAIL] golden: turn 2 after the SSD restore == after the RAM restore
echo == whirl-server-gate: 3 passed, 1 failed
exit /b 1
:tier
if not exist "%FG_WORK%" mkdir "%FG_WORK%"
echo 2026-10-23 23:08:31.400 I kv tier: pinned memory now: 1 HIP GPU(s)> "%FG_WORK%\server_golden_a.log"
if "%FAKEGATE_TIER%"=="nolog" goto tierchecks
echo 2026-10-23 23:08:31.488 W adjusted: WHIRL_KV_RAM_MB 8192 -^> 0 MiB (the RAM tier is pinned host memory; more than half of the 587 MiB of RAM available now would make the system page, so it is lowered to half; that leaves 0 MiB, so the host tiers are off (RAM and SSD))>> "%FG_WORK%\server_golden_a.log"
:tierchecks
echo   [FAIL] golden: session A turn 2 restored from the RAM tier (cached ^> 0, restore logged)  -- cached 0, RAM restores 0
echo   [FAIL] golden: session A turn 2 after a restart restored from the SSD tier (cached ^> 0, restore logged)  -- cached 0, SSD restores 0
if "%FAKEGATE_TIER%"=="mixed" goto tiermixed
echo == whirl-server-gate: 2 passed, 2 failed
echo    FAILED: golden: session A turn 2 restored from the RAM tier (cached ^> 0, restore logged)
echo    FAILED: golden: session A turn 2 after a restart restored from the SSD tier (cached ^> 0, restore logged)
exit /b 1
:tiermixed
echo   [FAIL] golden: precise: two concurrent requests == each alone (batched decode, no prefix cache)
echo == whirl-server-gate: 1 passed, 3 failed
echo    FAILED: golden: session A turn 2 restored from the RAM tier (cached ^> 0, restore logged)
echo    FAILED: golden: session A turn 2 after a restart restored from the SSD tier (cached ^> 0, restore logged)
echo    FAILED: golden: precise: two concurrent requests == each alone (batched decode, no prefix cache)
exit /b 1
