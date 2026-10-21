@echo off
rem REVIEW-GOV-2: compile gov_unit_tests.cpp against a WHIRL tree's mem_governor.cpp (CPU only).
rem Usage: build_unit.bat TREE OUTDIR
setlocal
set TREE=%~1
set OUT=%~2
if not exist "%OUT%" mkdir "%OUT%"
if not defined VSCMD_VER call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not defined VSCMD_VER call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /std:c++20 /EHsc /O2 /utf-8 /DNOMINMAX /I "%TREE%\include" "%~dp0gov_unit_tests.cpp" "%TREE%\src\common\mem_governor.cpp" /Fo"%OUT%\\" /Fe"%OUT%\gov_unit_tests.exe" >"%OUT%\build.txt" 2>&1
set RC=%ERRORLEVEL%
if %RC% neq 0 type "%OUT%\build.txt"
exit /b %RC%
