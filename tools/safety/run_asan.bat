@echo off
setlocal enabledelayedexpansion
rem safe025: AddressSanitizer + NullBackend leak-check build (build-asan) of the 5 CPU test suites and the fault driver, then run
rem every suite and every fault mode. Run it detached.
rem Usage: run_asan.bat [tokenizer.gguf]   (default: WHIRL_TEST_GGUF; whirl-server-tests reads only the GGUF header / vocabulary)
rem Output: build-asan\logs\<step>.log, build-asan\logs\steps.log (one line per step: exit code, expected code, ok/UNEXPECTED, start
rem time), build-asan\DONE (build rc), build-asan\RUN_DONE ("rc=<number of UNEXPECTED steps>").
rem WHIRL_KERNELS_FROM=<build dir> reuses its HIP code objects for a new build-asan (see build_cfg.bat).
set "R=%~dp0..\.."
set "B=%R%\build-asan"
set "L=%B%\logs"
set "GGUF=%~1"
if "%GGUF%"=="" set "GGUF=%WHIRL_TEST_GGUF%"
set /a BAD=0
if not exist "%L%" mkdir "%L%"
if exist "%B%\RUN_DONE" del "%B%\RUN_DONE"
>"%L%\steps.log" echo [%TIME%] build start
set "T0=%TIME%"
call "%~dp0build_cfg.bat" "%B%" "-DWHIRL_SANITIZE_ADDRESS=ON -DWHIRL_LEAKCHECK=ON -DWHIRL_SAFETY_FAULTS=ON -DWHIRL_WITH_HIP=ON" whirl-tests whirl-server-tests whirl-tier-tests whirl-cache-tests whirl-device-tests whirl-safety-faults > "%L%\build.log" 2>&1
call :rec build %ERRORLEVEL% 0
if not "!LAST!"=="ok" goto end
set "T0=%TIME%"
"%B%\whirl-device-tests.exe" > "%L%\whirl-device-tests.log" 2>&1
call :rec whirl-device-tests %ERRORLEVEL% 0
set "T0=%TIME%"
"%B%\whirl-cache-tests.exe" > "%L%\whirl-cache-tests.log" 2>&1
call :rec whirl-cache-tests %ERRORLEVEL% 0
set "T0=%TIME%"
"%B%\whirl-tier-tests.exe" > "%L%\whirl-tier-tests.log" 2>&1
call :rec whirl-tier-tests %ERRORLEVEL% 0
set "T0=%TIME%"
"%B%\whirl-tests.exe" > "%L%\whirl-tests.log" 2>&1
call :rec whirl-tests %ERRORLEVEL% 0
set "T0=%TIME%"
if "%GGUF%"=="" goto nogguf
"%B%\whirl-server-tests.exe" --gguf "%GGUF%" > "%L%\whirl-server-tests.log" 2>&1
call :rec whirl-server-tests %ERRORLEVEL% 0
goto faults
:nogguf
>"%L%\whirl-server-tests.log" echo SKIPPED: no tokenizer GGUF (argument 1 or WHIRL_TEST_GGUF)
call :rec whirl-server-tests skipped 0
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
