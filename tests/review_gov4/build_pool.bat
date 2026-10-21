@echo off
rem REVIEW-GOV-4: compile pool_test.cpp against a tree's pool_budget.cpp (CPU only). Usage: build_pool.bat TREE OUTDIR [FB]
rem FB: the tree is fbdbe38 (no vision reserve API): define REVIEW_FBDBE38.
setlocal
set TREE=%~1
set OUT=%~2
set DEF=
if /I "%~3"=="FB" set DEF=/DREVIEW_FBDBE38
if not exist "%OUT%" mkdir "%OUT%"
if not defined VSCMD_VER call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not defined VSCMD_VER call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /std:c++20 /EHsc /O2 /utf-8 /DNOMINMAX %DEF% /I "%TREE%\include" "%~dp0pool_test.cpp" "%TREE%\src\common\pool_budget.cpp" /Fo"%OUT%\\" /Fe"%OUT%\pool_test.exe" >"%OUT%\build.txt" 2>&1
set RC=%ERRORLEVEL%
if %RC% neq 0 type "%OUT%\build.txt"
exit /b %RC%
