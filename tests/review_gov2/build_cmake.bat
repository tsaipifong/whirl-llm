@echo off
rem REVIEW-GOV-2: CPU-only clean build of the reviewer's tests against a WHIRL tree (no HIP).
rem Usage: build_cmake.bat TREE BUILDDIR   (BUILDDIR is deleted first: clean build, header deps are unreliable)
setlocal
set TREE=%~1
set BD=%~2
if exist "%BD%" rmdir /s /q "%BD%"
if not defined VSCMD_VER call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not defined VSCMD_VER call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cmake -S "%TREE%" -B "%BD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DWHIRL_WITH_HIP=OFF "-DCMAKE_PROJECT_INCLUDE=%~dp0inject.cmake" || exit /b 1
cmake --build "%BD%" -j 8 --target review-gov2-unit review-gov2-restore
exit /b %ERRORLEVEL%
