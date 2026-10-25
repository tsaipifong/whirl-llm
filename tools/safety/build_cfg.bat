@echo off
rem safe025: configure (if needed) + build the CPU test targets in a separate build directory (low priority, -j 8).
rem Usage: build_cfg.bat <build_dir> "<extra cmake -D args>" [targets...]
rem Writes <build_dir>\DONE = "rc=<exit code>" when it ends (a configure failure is rc=1). Run it detached.
rem Optional environment:
rem   WHIRL_CFG_ONLY=1        configure only (no build, no DONE)
rem   WHIRL_KERNELS_FROM=<d>  copy <d>\kernels and <d>\res (the HIP device code objects) into a NEW build directory before its first
rem                           build, so a CPU check does not compile the kernels again; only valid for the same kernel sources
rem   WHIRL_QUIET_WAIT=<ps1>  a local script run first (e.g. "wait while a GPU measurement holds the machine"); WHIRL_QUIET_SKIP=1 skips it
rem read %~dp0 before "shift": shift moves %0 too
set "H=%~dp0"
set "R=%~dp0..\.."
set "B=%~1"
set "XA=%~2"
set "BRC=1"
shift
shift
set "TG="
:loop
if "%~1"=="" goto args_done
set "TG=%TG% %~1"
shift
goto loop
:args_done
if "%TG%"=="" set "TG=whirl-tests whirl-server-tests whirl-tier-tests whirl-cache-tests whirl-device-tests"
if not exist "%B%" mkdir "%B%"
if not "%WHIRL_CFG_ONLY%"=="1" if exist "%B%\DONE" del "%B%\DONE"
if "%WHIRL_QUIET_SKIP%"=="1" goto noquiet
if "%WHIRL_QUIET_WAIT%"=="" goto noquiet
pwsh -NoProfile -File "%WHIRL_QUIET_WAIT%" -Who "safe025 build %B%"
:noquiet
call "%H%vcenv.bat"
if errorlevel 1 goto fin
rem cl prints in the console code page; use the system OEM page so the logs are readable
for /f "tokens=3" %%c in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Nls\CodePage" /v OEMCP 2^>nul ^| findstr /i OEMCP') do chcp %%c >nul
if exist "%B%\build.ninja" goto build
cmake -S "%R%" -B "%B%" -G Ninja -DCMAKE_BUILD_TYPE=Release %XA%
if errorlevel 1 goto fin
if "%WHIRL_KERNELS_FROM%"=="" goto build
robocopy "%WHIRL_KERNELS_FROM%\kernels" "%B%\kernels" /E /COPY:DAT /NFL /NDL /NJH /NJS /NP >nul
robocopy "%WHIRL_KERNELS_FROM%\res" "%B%\res" /E /COPY:DAT /NFL /NDL /NJH /NJS /NP >nul
:build
set "BRC=0"
if "%WHIRL_CFG_ONLY%"=="1" exit /b 0
start "" /b /wait /low cmake --build "%B%" -j 8 --target %TG%
set "BRC=%ERRORLEVEL%"
:fin
if "%WHIRL_CFG_ONLY%"=="1" exit /b %BRC%
rem the variable is BRC, not RC: CMake reads RC as the resource compiler. Redirection first: "echo %BRC%> f" with 0 is parsed as "echo" + "0>" (handle 0) and leaves an empty file (review C4)
>"%B%\DONE" echo rc=%BRC%
exit /b %BRC%
