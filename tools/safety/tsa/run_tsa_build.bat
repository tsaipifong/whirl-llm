@echo off
rem arch025-Q4: the cpu_tsa build. clang-cl (the AMD clang of the HIP SDK) over the MSVC STL compiles whirl_server_core and
rem whirl_tier_core (and the libraries they link) with -Wno-everything -Wthread-safety -Werror=thread-safety (cmake/safety.cmake
rem WHIRL_TSA), then tsa_check.py mutate removes one lock at a time from copies of the sources and checks that every such build
rem FAILS with the expected diagnostic (and the unchanged copy passes).
rem Usage: run_tsa_build.bat [build_dir]   (default <repo>\build-tsa; /build-*/ is in .gitignore)
rem Writes <build_dir>\DONE "rc=N" (the build, by build_cfg.bat) and <build_dir>\MUTATE "rc=N" (the proof); logs in <build_dir>\logs\.
rem clang-cl: WHIRL_CLANG_CL (full path of clang-cl.exe), else %HIP_PATH%bin\clang-cl.exe, else clang-cl on PATH.
rem Python: WHIRL_PYTHON, else python. WHIRL_QUIET_WAIT / WHIRL_QUIET_SKIP as in build_cfg.bat.
setlocal
rem own variable names: build_cfg.bat (called below, no setlocal) sets H, R, B, XA, BRC, TG
set "TSAH=%~dp0"
set "TSAR=%~dp0..\..\.."
set "TSAB=%~1"
if "%TSAB%"=="" set "TSAB=%TSAR%\build-tsa"
set "TSAPY=%WHIRL_PYTHON%"
if "%TSAPY%"=="" set "TSAPY=python"
if not exist "%TSAB%\logs" mkdir "%TSAB%\logs"
if exist "%TSAB%\MUTATE" del "%TSAB%\MUTATE"
set "CCDIR="
if not "%WHIRL_CLANG_CL%"=="" for %%i in ("%WHIRL_CLANG_CL%") do set "CCDIR=%%~dpi"
if "%CCDIR%"=="" if not "%HIP_PATH%"=="" set "CCDIR=%HIP_PATH%\bin\"
rem CXX (read by the first configure) instead of -DCMAKE_CXX_COMPILER: the HIP SDK path has spaces
set "CXX=clang-cl"
if not "%CCDIR%"=="" set "CXX=%CCDIR%clang-cl.exe"
"%CXX%" --version >"%TSAB%\logs\clang.log" 2>&1
call "%TSAH%..\build_cfg.bat" "%TSAB%" "-DWHIRL_WITH_HIP=OFF -DWHIRL_TSA=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" whirl_server_core whirl_tier_core >"%TSAB%\logs\build.log" 2>&1
set "TSABR=%ERRORLEVEL%"
rem the proof needs compile_commands.json from a configured tree; it runs after a failed build too (its control compile then
rem shows the same failure)
"%TSAPY%" -I "%TSAH%tsa_check.py" mutate "%TSAB%" "%TSAR%" >"%TSAB%\logs\mutate.log" 2>&1
set "TSAMR=%ERRORLEVEL%"
>"%TSAB%\MUTATE" echo rc=%TSAMR%
type "%TSAB%\logs\mutate.log"
if not "%TSABR%"=="0" exit /b %TSABR%
exit /b %TSAMR%
