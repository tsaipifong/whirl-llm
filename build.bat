@echo off
rem WHIRL build entry point (SPDX-License-Identifier: Apache-2.0)
rem Usage: build.bat [Release|Debug] [extra cmake configure args...]
rem Runs at low CPU priority so it never competes with benchmark jobs.
rem
rem Header dependencies: Ninja learns a .obj's headers from cl /showIncludes,
rem matching each line against the prefix CMake detected (msvc_deps_prefix in
rem rules.ninja). A localized cl prints that prefix translated (zh-TW: the
rem Chinese for "Note: including file:") in the console code page, and CMake
rem writes it into rules.ninja in the code page of the console that generated
rem the build. If the build later runs under another code page (chcp 65001, a
rem different shell) no line matches and Ninja silently records 0 headers, so
rem header edits stop rebuilding dependents. VSLANG=1033 is no cure: cl falls
rem back to the installed language when the English pack is missing. So this
rem script pins the console to the system OEM code page (what a fresh cmd or
rem PowerShell uses) for configure and build, restores it afterwards, and
rem stamps the build directory with that code page; a directory generated
rem under another code page is regenerated once and its dependency log reset
rem (one full rebuild).
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release
set BUILD_DIR=%~dp0build\%CFG%
set "WHIRL_OLDCP="
for /f "tokens=2 delims=:." %%c in ('chcp') do set /a WHIRL_OLDCP=%%c >nul 2>&1
set "WHIRL_CP="
for /f "tokens=3" %%c in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Nls\CodePage" /v OEMCP 2^>nul ^| findstr /i OEMCP') do set "WHIRL_CP=%%c"
if not defined WHIRL_CP set "WHIRL_CP=%WHIRL_OLDCP%"
if defined WHIRL_CP chcp %WHIRL_CP% >nul
call :main %*
set WHIRL_RC=%ERRORLEVEL%
if defined WHIRL_OLDCP chcp %WHIRL_OLDCP% >nul
exit /b %WHIRL_RC%

:main
if defined VSCMD_VER goto :have_vs
rem find vcvars64.bat via vswhere (any edition with the C++ x64 tools:
rem Community / Professional / Enterprise / BuildTools; VS 2022 only, HIP clang with
rem the VS 2026 STL is unverified), else the old fixed path
set "VCVARS="
set "VSINST="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :vs_fallback
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINST=%%i"
if not defined VSINST goto :vs_fallback
if exist "%VSINST%\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%VSINST%\VC\Auxiliary\Build\vcvars64.bat"
:vs_fallback
if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :call_vs
echo build.bat: no Visual Studio with the C++ x64 tools found ^(checked vswhere and the default Build Tools path^). 1>&2
echo build.bat: install Visual Studio 2022 with "Desktop development with C++" or run from a Developer Command Prompt. 1>&2
exit /b 1
:call_vs
call "%VCVARS%" >nul || exit /b 1
:have_vs
set "WHIRL_STAMP=%BUILD_DIR%\whirl_codepage.txt"
if not exist "%BUILD_DIR%\build.ninja" goto :configure
set "WHIRL_STAMPCP="
if exist "%WHIRL_STAMP%" set /p WHIRL_STAMPCP=<"%WHIRL_STAMP%"
if "%WHIRL_STAMPCP%"=="%WHIRL_CP%" goto :build
echo build.bat: build files not generated under code page %WHIRL_CP% ^(stamp: "%WHIRL_STAMPCP%"^); regenerating and rescanning header dependencies ^(one full rebuild^)
start "" /b /wait /low cmake -S "%~dp0." -B "%BUILD_DIR%" || exit /b 1
if exist "%BUILD_DIR%\.ninja_deps" del /q "%BUILD_DIR%\.ninja_deps"
> "%WHIRL_STAMP%" echo %WHIRL_CP%
goto :build
:configure
start "" /b /wait /low cmake -S "%~dp0." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CFG% %2 %3 %4 %5 || exit /b 1
> "%WHIRL_STAMP%" echo %WHIRL_CP%
:build
start "" /b /wait /low cmake --build "%BUILD_DIR%" -j 8
exit /b %ERRORLEVEL%
