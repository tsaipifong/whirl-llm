@echo off
setlocal
rem safe025 / arch025-Q3: real-compiler block AND pass proof of the /analyze baseline gate. Usage: analyze_fault.bat <out_dir> [baseline.txt]
rem Compiles tests\safety\analyze_fault.cpp with /analyze three times (AF_MODE 0 clean, 1 new, 2 same key), each into a one-TU
rem "build directory" laid out like run_analyze.bat leaves it (start marker, expected list, DONE, SARIF), runs the baseline gate
rem (analyze_check.py check) on each against tests\safety\analyze_fault_baseline.txt (one accepted key), then the verdict
rem analyze_check.py faultproof <out_dir>: clean must PASS, new must FAIL with the 3 planted keys NEW, same key must FAIL with
rem "NEW x1" on the baselined key. Output: <out_dir>\<case>\gate.log + GATE, <out_dir>\proof.log, exit code in <out_dir>\PROOF.
rem Exit 0 only when the proof holds. Needs cl (vcenv.bat) and WHIRL_PYTHON (default python); a few seconds.
set "R=%~dp0..\.."
set "O=%~1"
set "BL=%~2"
if "%O%"=="" goto usage
if "%BL%"=="" set "BL=%R%\tests\safety\analyze_fault_baseline.txt"
set "PY=%WHIRL_PYTHON%"
if "%PY%"=="" set "PY=python"
if not exist "%O%" mkdir "%O%"
if exist "%O%\PROOF" del "%O%\PROOF"
call "%~dp0vcenv.bat" || goto novc
for %%f in ("%R%") do set "RA=%%~ff"
call :one clean 0
call :one new 1
call :one samekey 2
"%PY%" -I "%~dp0analyze_check.py" faultproof "%O%" > "%O%\proof.log" 2>&1
set "PRC=%ERRORLEVEL%"
>"%O%\PROOF" echo rc=%PRC%
type "%O%\proof.log"
exit /b %PRC%

:one
set "C=%O%\%~1"
if exist "%C%" rmdir /s /q "%C%"
mkdir "%C%"
>"%C%\CMakeCache.txt" echo CMAKE_HOME_DIRECTORY:INTERNAL=%RA%
>"%C%\analyze_expected.txt" echo af.cpp.obj
>"%C%\analyze_start" echo targets analyze_fault AF_MODE=%~2
cl /nologo /c /W4 /permissive- /EHsc /utf-8 /DAF_MODE=%~2 /analyze /analyze:external- /analyze:log:format:sarif /Fo"%C%\af.cpp.obj" "%R%\tests\safety\analyze_fault.cpp" > "%C%\analyze_fault.log" 2>&1
>"%C%\DONE" echo rc=%ERRORLEVEL%
"%PY%" -I "%~dp0analyze_check.py" check "%C%" "%BL%" > "%C%\gate.log" 2>&1
>"%C%\GATE" echo rc=%ERRORLEVEL%
exit /b 0

:novc
>"%O%\PROOF" echo rc=1
echo analyze fault proof: FAIL (cl not available: vcenv.bat)
exit /b 1

:usage
echo usage: analyze_fault.bat ^<out_dir^> [baseline.txt]
exit /b 2
