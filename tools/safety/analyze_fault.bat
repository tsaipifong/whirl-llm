@echo off
setlocal
rem safe025: injected static-analysis fault proof. Usage: analyze_fault.bat <out_dir> [baseline.txt]
rem Compiles tests\safety\analyze_fault.cpp (3 planted defects) with /analyze into a one-TU "build directory" laid out like
rem run_analyze.bat leaves it (start marker, expected list, DONE, SARIF), then runs the baseline gate on it: it must FAIL with the
rem planted warnings as NEW. Gate output: <out_dir>\gate.log, exit code in <out_dir>\GATE.
set "R=%~dp0..\.."
set "O=%~1"
set "BL=%~2"
if "%BL%"=="" set "BL=%~dp0analyze_baseline.txt"
set "PY=%WHIRL_PYTHON%"
if "%PY%"=="" set "PY=python"
if not exist "%O%" mkdir "%O%"
call "%~dp0vcenv.bat" || exit /b 1
for %%f in ("%R%") do set "RA=%%~ff"
>"%O%\CMakeCache.txt" echo CMAKE_HOME_DIRECTORY:INTERNAL=%RA%
>"%O%\analyze_expected.txt" echo af.cpp.obj
>"%O%\analyze_start" echo targets analyze_fault
if exist "%O%\af.cpp.obj" del "%O%\af.cpp.obj"
cl /nologo /c /W4 /permissive- /EHsc /utf-8 /analyze /analyze:external- /analyze:log:format:sarif /Fo"%O%\af.cpp.obj" "%R%\tests\safety\analyze_fault.cpp" > "%O%\analyze_fault.log" 2>&1
>"%O%\DONE" echo rc=%ERRORLEVEL%
"%PY%" "%~dp0analyze_check.py" check "%O%" "%BL%" > "%O%\gate.log" 2>&1
>"%O%\GATE" echo rc=%ERRORLEVEL%
type "%O%\gate.log"
