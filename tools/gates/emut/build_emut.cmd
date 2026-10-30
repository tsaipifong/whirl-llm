@echo off
rem SPDX-License-Identifier: Apache-2.0
rem arch025-emut: build emut.exe with MSVC + CMake + Ninja against the official LLVM release archive.
rem   build_emut.cmd <build dir> [<LLVM folder, default %WHIRL_LLVM% or C:\ClaudeWorkspace\tools\llvm\21.1.8>]
setlocal
set "B=%~1"
if "%B%"=="" (echo usage: build_emut.cmd ^<build dir^> [llvm dir] & exit /b 2)
set "L=%~2"
if "%L%"=="" set "L=%WHIRL_LLVM%"
if "%L%"=="" set "L=C:\ClaudeWorkspace\tools\llvm\21.1.8"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 3
cmake -G Ninja -S "%~dp0." -B "%B%" -DWHIRL_LLVM="%L%" || exit /b 4
cmake --build "%B%" || exit /b 5
echo emut built: %B%\emut.exe
