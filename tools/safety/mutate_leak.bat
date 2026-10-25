@echo off
setlocal enabledelayedexpansion
rem safe025: "missing VRAM free" mutation proof in a SCRATCH COPY: device::freeDev (src/device/handles.h, what every RAII device
rem buffer calls) is turned into a no-op, the copy is built with WHIRL_LEAKCHECK=ON and whirl-server-tests is run.
rem Expected: LEAKCHECK lines, the test summary line still in the log, and exit code 97 (the control, build-asan, gives
rem "N NullBackend instances checked, 0 leaked").
rem Usage: mutate_leak.bat <scratch_dir> [tokenizer.gguf] [kernels_from_build_dir]
rem   gguf default WHIRL_TEST_GGUF; kernels default build-asan of this tree.
rem Detached; <scratch_dir>\LEAK_DONE; logs in <scratch_dir>\logs_leak (steps.log has the exit codes).
set "R=%~dp0..\.."
set "S=%~1"
set "GGUF=%~2"
if "%GGUF%"=="" set "GGUF=%WHIRL_TEST_GGUF%"
set "KF=%~3"
if "%KF%"=="" set "KF=%R%\build-asan"
set "L=%S%\logs_leak"
if not exist "%L%" mkdir "%L%"
if exist "%S%\LEAK_DONE" del "%S%\LEAK_DONE"
robocopy "%R%" "%S%\src_leak" /MIR /XD build build-* .git /XF *.log /NFL /NDL /NJH /NJS /NP > "%L%\robocopy.log"
call "%~dp0vcenv.bat" || exit /b 1
rem the new line must sit where the old one was: rebuild the file with the replacement in place
powershell -NoProfile -Command "(Get-Content -Raw '%R%\src\device\handles.h').Replace('inline void freeDev(Backend& b, DevPtr p) { b.free(p); }','inline void freeDev(Backend&, DevPtr) {}  // MUTANT: the free is missing') | Set-Content -NoNewline '%S%\src_leak\src\device\handles.h'"
fc "%R%\src\device\handles.h" "%S%\src_leak\src\device\handles.h" > "%L%\mutation.diff"
cmake -S "%S%\src_leak" -B "%S%\bleak" -G Ninja -DCMAKE_BUILD_TYPE=Release -DWHIRL_WITH_HIP=ON -DWHIRL_LEAKCHECK=ON > "%L%\configure.log" 2>&1
robocopy "%KF%\kernels" "%S%\bleak\kernels" /E /COPY:DAT /NFL /NDL /NJH /NJS /NP >nul
robocopy "%KF%\res" "%S%\bleak\res" /E /COPY:DAT /NFL /NDL /NJH /NJS /NP >nul
start "" /b /wait /low cmake --build "%S%\bleak" -j 8 --target whirl-server-tests > "%L%\build.log" 2>&1
>"%L%\steps.log" echo [!TIME!] build exit !ERRORLEVEL!
"%S%\bleak\whirl-server-tests.exe" --gguf "%GGUF%" > "%L%\run.log" 2>&1
>>"%L%\steps.log" echo [!TIME!] run exit !ERRORLEVEL!
>"%S%\LEAK_DONE" echo done
