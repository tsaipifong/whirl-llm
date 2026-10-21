@echo off
rem REVIEW-GOV-2: re-run configure + build in an existing build dir (after build_cmake.bat made it clean once)
setlocal
if not defined VSCMD_VER call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cmake "%~1" >nul || exit /b 1
cmake --build "%~1" -j 8 --target review-gov2-unit review-gov2-restore
exit /b %ERRORLEVEL%
