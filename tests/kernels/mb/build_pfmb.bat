@echo off
rem WHIRL pfmb microbench (feat026-prefetch): device-only gfx1201 compile of tests\kernels\mb\pfmb.hip + ISA / notes dumps.
rem CPU only (no GPU). Not part of the CMake build: the engine code object (kernels\whirl_kernels.hip) never includes pfmb.hip.
rem usage: build_pfmb.bat <out_dir>   -> <out_dir>\pfmb.gfx1201.hsaco, pfmb.dis.s (llvm-objdump -d), pfmb.notes.txt
rem        (llvm-readobj --notes: vgpr / sgpr / spill per kernel), build.log, DONE (rc=...)
rem run: set WHIRL_KT_PFMB_HSACO=<out_dir>\pfmb.gfx1201.hsaco and whirl-kernel-test --device pci:71:0 --only pfmb
setlocal
set "SRC=%~dp0pfmb.hip"
set "O=%~1"
if "%O%"=="" (
  echo usage: build_pfmb.bat ^<out_dir^>
  exit /b 2
)
set "ROCM=C:\Program Files\AMD\ROCm\7.2\bin"
if not exist "%O%" mkdir "%O%"
del /q "%O%\DONE" 2>nul
"%ROCM%\clang.exe" -x hip --offload-arch=gfx1201 --cuda-device-only --no-gpu-bundle-output -O3 -std=c++17 -o "%O%\pfmb.gfx1201.hsaco" "%SRC%" > "%O%\build.log" 2>&1
set "RC=%ERRORLEVEL%"
if exist "%O%\pfmb.gfx1201.hsaco" (
  "%ROCM%\llvm-objdump.exe" -d "%O%\pfmb.gfx1201.hsaco" > "%O%\pfmb.dis.s" 2>nul
  "%ROCM%\llvm-readobj.exe" --notes "%O%\pfmb.gfx1201.hsaco" > "%O%\pfmb.notes.txt" 2>nul
)
>"%O%\DONE" echo rc=%RC%
exit /b %RC%
