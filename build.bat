@echo off
rem WHIRL build entry point (SPDX-License-Identifier: Apache-2.0)
rem Usage: build.bat [Release|Debug] [extra cmake configure args...]
rem Runs at low CPU priority so it never competes with benchmark jobs.
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release
set BUILD_DIR=%~dp0build\%CFG%
if not defined VSCMD_VER call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist "%BUILD_DIR%\build.ninja" (
  start "" /b /wait /low cmake -S "%~dp0." -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=%CFG% %2 %3 %4 %5 || exit /b 1
)
start "" /b /wait /low cmake --build "%BUILD_DIR%" -j 8
exit /b %ERRORLEVEL%
