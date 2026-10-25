@echo off
rem feat026-attn8s: build + run the CPU emulation of attn_prefill_wmma vs attn_pf2 (no GPU).
rem Usage: build_emu.bat <out_dir>   writes <out_dir>\attn_pf2_emu.exe, emu_build.log, emu_run.log, DONE_emu (rc=...)
set "O=%~1"
if "%O%"=="" set "O=C:\ClaudeWorkspace\whirl-work\scratch\feat026_attn8s"
del /q "%O%\DONE_emu" 2>nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul || (>"%O%\DONE_emu" echo rc=vcvars & exit /b 1)
cl /nologo /O2 /std:c++20 /EHsc /fp:precise /Fe"%O%\attn_pf2_emu.exe" /Fo"%O%\attn_pf2_emu.obj" "%~dp0attn_pf2_emu.cpp" > "%O%\emu_build.log" 2>&1 || (>"%O%\DONE_emu" echo rc=build & exit /b 1)
"%O%\attn_pf2_emu.exe" > "%O%\emu_run.log" 2>&1
>"%O%\DONE_emu" echo rc=%ERRORLEVEL%
