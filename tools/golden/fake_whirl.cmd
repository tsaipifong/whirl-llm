@echo off
rem Stand-in for whirl.exe used by golden_selftest.ps1 (no GPU). SPDX-License-Identifier: Apache-2.0
rem Called as: golden MODEL --suite SUITE --MODE --device ARCH
rem FAKEGOLD_FAIL_SUITE=SUITE: exit 3 for that suite; FAKEGOLD_NOSTEPS=1: header + info lines only; FAKEGOLD_NODEV=1: no device line.
rem Suite ngram also prints an n-gram cycle line (ng_cycles=2) so the default-suite selftest case can pass.
rem FAKEGOLD_DIR=DIR: print DIR\SUITE.txt instead (golden lines without the G model line; FIX-TEST-A).
if defined FAKEGOLD_DIR (
  type "%FAKEGOLD_DIR%\%4.txt"
  exit /b 0
)
rem the device line whirl golden prints (selectDevice); FAKEGOLD_NODEV=1 leaves it out (REVIEW-ARCH6 C-M3 selftest)
if not "%FAKEGOLD_NODEV%"=="1" echo   whirl build, device 0: Fake GPU (%7)
if "%FAKEGOLD_FAIL_SUITE%"=="%4" (
  echo fake: failing suite %4
  exit /b 3
)
echo G header arch=%7 suite=%4 mode=%5
echo G info fake=1
if "%FAKEGOLD_NOSTEPS%"=="1" exit /b 0
echo G step0 ids=1,2,3 lh=aa
echo G step1 ids=4,5 lh=bb
if "%4"=="ngram" echo G mtp_ngram.cycles drafts=8 cycles=5 accepted=10 ng_cycles=2 ng_accepted=4
exit /b 0
