@echo off
rem safe025: clang-cl -Wthread-safety proof on the toy file tsa_fault.cpp (clean must compile, -DINJECT_UNLOCKED must fail).
rem Usage: run_tsa.bat [out_dir]   (default build-tsa in the source tree). clang-cl: WHIRL_CLANG_CL, else %HIP_PATH%bin\clang-cl.exe
rem (the AMD clang 21 of the HIP SDK), else clang-cl on PATH.
rem SCOPE: this proves the analysis on tsa_fault.cpp only. 6 of 17 WHIRL host sources do not compile with clang-cl over the MSVC STL
rem (include/whirl/json.h:30/41 uses an incomplete type in std::pair; see syntax_check.bat and the safe025 HANDOFF).
call "%~dp0..\vcenv.bat" || exit /b 1
set "CC=%WHIRL_CLANG_CL%"
if "%CC%"=="" if not "%HIP_PATH%"=="" set "CC=%HIP_PATH%\bin\clang-cl.exe"
if "%CC%"=="" set "CC=clang-cl"
set "O=%~1"
if "%O%"=="" set "O=%~dp0..\..\..\build-tsa"
if not exist "%O%" mkdir "%O%"
set "F=/std:c++20 /EHsc /MT /clang:-Wthread-safety /clang:-Wthread-safety-beta /clang:-Werror=thread-safety-analysis"
echo === clean source
"%CC%" %F% /c "%~dp0tsa_fault.cpp" /Fo"%O%\clean.obj"
echo exit=%ERRORLEVEL%
echo === injected fault
"%CC%" %F% /DINJECT_UNLOCKED /c "%~dp0tsa_fault.cpp" /Fo"%O%\fault.obj"
echo exit=%ERRORLEVEL%
