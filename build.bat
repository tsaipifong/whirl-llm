@echo off
rem WHIRL build entry point (SPDX-License-Identifier: Apache-2.0)
rem Usage: build.bat [Release|Debug] [extra cmake configure args...]
rem Runs at low CPU priority so it never competes with benchmark jobs.
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release
set BUILD_DIR=%~dp0build\%CFG%
if defined VSCMD_VER goto :have_vs
rem find vcvars64.bat via vswhere (any edition with the C++ x64 tools:
rem Community / Professional / Enterprise / BuildTools), else the old fixed path
set "VCVARS="
set "VSINST="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :vs_fallback
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINST=%%i"
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
if not exist "%BUILD_DIR%\build.ninja" (
  start "" /b /wait /low cmake -S "%~dp0." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CFG% %2 %3 %4 %5 || exit /b 1
)
start "" /b /wait /low cmake --build "%BUILD_DIR%" -j 8
exit /b %ERRORLEVEL%
