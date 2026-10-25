@echo off
setlocal
rem safe025: MSVC /analyze full host rebuild of the CPU test targets, then the baseline gate. Run it detached.
rem Usage: run_analyze.bat [build_dir]     (default build-analyze in the source tree)
rem "analyze_check.py prepare" deletes the object of every /analyze TU first, so every TU is compiled again and the gate sees all
rem warnings (review A3: an incremental build reports only the TUs it recompiled). The HIP kernels are not rebuilt.
rem Output: <build_dir>\DONE (build "rc=N"), <build_dir>\GATE (gate "rc=N": 0 PASS, 1 FAIL), logs in <build_dir>\logs.
rem WHIRL_PYTHON=<python.exe> (default: python on PATH); WHIRL_KERNELS_FROM=<build dir> for a new build dir (see build_cfg.bat).
set "R=%~dp0..\.."
set "B=%~1"
if "%B%"=="" set "B=%R%\build-analyze"
set "TG=whirl-tests whirl-server-tests whirl-tier-tests whirl-cache-tests whirl-device-tests"
set "PY=%WHIRL_PYTHON%"
if "%PY%"=="" set "PY=python"
if not exist "%B%\logs" mkdir "%B%\logs"
if exist "%B%\GATE" del "%B%\GATE"
set "WHIRL_CFG_ONLY=1"
call "%~dp0build_cfg.bat" "%B%" "-DWHIRL_ANALYZE=ON -DWHIRL_WITH_HIP=ON" > "%B%\logs\configure.log" 2>&1
set "CRC=%ERRORLEVEL%"
set "WHIRL_CFG_ONLY="
if not "%CRC%"=="0" goto cfgfail
call "%~dp0vcenv.bat"
"%PY%" "%~dp0analyze_check.py" prepare "%B%" %TG% > "%B%\logs\prepare.log" 2>&1
call "%~dp0build_cfg.bat" "%B%" "" %TG% > "%B%\logs\build.log" 2>&1
"%PY%" "%~dp0analyze_check.py" check "%B%" "%~dp0analyze_baseline.txt" > "%B%\logs\gate.log" 2>&1
>"%B%\GATE" echo rc=%ERRORLEVEL%
exit /b 0
:cfgfail
>"%B%\DONE" echo rc=%CRC%
>"%B%\GATE" echo rc=1
exit /b 1
