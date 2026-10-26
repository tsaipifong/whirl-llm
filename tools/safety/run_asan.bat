@echo off
setlocal enabledelayedexpansion
rem safe025 / arch025-Q1: AddressSanitizer + NullBackend leak-check build (build-asan) of the 10 CPU test executables of
rem arch025 PLAN Q1 (unit, model, cache, tier, device, ops, server, pins, startup-replay, vision) and the fault driver, then run
rem every suite and every fault mode. Run it detached, or through tools/safety/safety_step.ps1 -Kind asan (release_gate cpu_asan),
rem which adds the time cap and the verdict (tools/safety/asan_check.py).
rem Usage: run_asan.bat [swift.gguf] [moe.gguf]
rem   swift.gguf default WHIRL_TEST_GGUF (whirl-server-tests and the pins read only the GGUF header / vocabulary)
rem   moe.gguf   default WHIRL_TEST_GGUF_MOE (tests/pins/check.ps1 tokens_moe)
rem Output: build-asan\logs\<step>.log, build-asan\logs\steps.log (one line per step: exit code, expected code, ok/UNEXPECTED, start
rem time), build-asan\DONE (build rc), build-asan\RUN_DONE ("rc=<number of UNEXPECTED steps>").
rem WHIRL_KERNELS_FROM=<build dir> reuses its HIP code objects for a new build-asan (see build_cfg.bat).
set "R=%~dp0..\.."
set "B=%R%\build-asan"
set "L=%B%\logs"
set "GGUF=%~1"
if "%GGUF%"=="" set "GGUF=%WHIRL_TEST_GGUF%"
set "GMOE=%~2"
if "%GMOE%"=="" set "GMOE=%WHIRL_TEST_GGUF_MOE%"
set /a BAD=0
if not exist "%L%" mkdir "%L%"
if exist "%B%\RUN_DONE" del "%B%\RUN_DONE"
rem arch025-Q135: no log of an earlier run may be read as this run's (run 2 of 2026-10-25 passed a stale vision log to the checker);
rem asan_check.py requires every log to be newer than RUN_START
del /q "%L%\*.log" 2>nul
if exist "%L%\pins_out" rmdir /s /q "%L%\pins_out"
>"%B%\RUN_START" echo start %DATE% %TIME%
>"%L%\steps.log" echo [%TIME%] build start
set "T0=%TIME%"
call "%~dp0build_cfg.bat" "%B%" "-DWHIRL_SANITIZE_ADDRESS=ON -DWHIRL_LEAKCHECK=ON -DWHIRL_SAFETY_FAULTS=ON -DWHIRL_WITH_HIP=ON" whirl-tests whirl-model-tests whirl-server-tests whirl-tier-tests whirl-cache-tests whirl-device-tests whirl-ops-tests whirl-vision-tests whirl-pins whirl-startup-replay whirl-safety-faults > "%L%\build.log" 2>&1
call :rec build %ERRORLEVEL% 0
if not "!LAST!"=="ok" goto end
call :suite whirl-device-tests
call :suite whirl-cache-tests
call :suite whirl-tier-tests
call :suite whirl-tests
call :suite whirl-model-tests
call :suite whirl-ops-tests
call :suite whirl-vision-tests
set "T0=%TIME%"
if "%GGUF%"=="" goto nogguf
"%B%\whirl-server-tests.exe" --gguf "%GGUF%" > "%L%\whirl-server-tests.log" 2>&1
call :rec whirl-server-tests %ERRORLEVEL% 0
set "T0=%TIME%"
if "%GMOE%"=="" goto nomoe
rem whirl-pins + whirl-startup-replay under ASan, outputs compared byte for byte with tests/pins/expected (the cpu_pins check).
rem WHIRL_LEAKCHECK_QUIET=1: the leak check's exit summary line ("LEAKCHECK: N NullBackend instances checked, 0 leaked", stderr)
rem would land in the compared output (arch025-Q135 run 2026-10-25: 14 of 15 pins DIFF by that one line only); a leak still ends
rem the program with exit 97, which check.ps1 reports as FAIL.
set "WHIRL_LEAKCHECK_QUIET=1"
pwsh -NoProfile -File "%R%\tests\pins\check.ps1" -Bin "%B%" -Swift "%GGUF%" -Moe "%GMOE%" -Out "%L%\pins_out" > "%L%\pins.log" 2>&1
call :rec pins %ERRORLEVEL% 0
set "WHIRL_LEAKCHECK_QUIET="
goto faults
:nogguf
>"%L%\whirl-server-tests.log" echo SKIPPED: no tokenizer GGUF (argument 1 or WHIRL_TEST_GGUF)
call :rec whirl-server-tests skipped 0
:nomoe
>"%L%\pins.log" echo SKIPPED: no Swift and MoE GGUF (arguments 1 and 2, or WHIRL_TEST_GGUF / WHIRL_TEST_GGUF_MOE)
call :rec pins skipped 0
:faults
rem expected: clean 0; a leak -> 97 (leak check); a memory error -> non-zero from ASan ("nz")
call :fault clean 0
call :fault missing_free 97
call :fault missing_host 97
call :fault missing_stream 97
call :fault uaf nz
call :fault uaf_vector nz
call :fault overflow nz
call :fault double_free nz
:end
>"%B%\RUN_DONE" echo rc=!BAD!
exit /b !BAD!

:suite
set "T0=%TIME%"
"%B%\%1.exe" > "%L%\%1.log" 2>&1
call :rec %1 %ERRORLEVEL% 0
exit /b 0

:fault
set "T0=%TIME%"
"%B%\whirl-safety-faults.exe" %1 > "%L%\fault_%1.log" 2>&1
call :rec fault_%1 %ERRORLEVEL% %2
exit /b 0

rem :rec <step> <exit code> <expected: number, or nz = any non-zero>
:rec
set "LAST=ok"
if "%3"=="nz" (
  if "%2"=="0" set "LAST=UNEXPECTED"
) else (
  if not "%2"=="%3" set "LAST=UNEXPECTED"
)
if "!LAST!"=="UNEXPECTED" set /a BAD+=1
>>"%L%\steps.log" echo [%TIME%] %1 exit %2 expect %3 !LAST! start=!T0!
exit /b 0
