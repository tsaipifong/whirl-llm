@echo off
setlocal enabledelayedexpansion
rem safe025: ownership-check mutation proof, in a SCRATCH COPY (the source tree is not touched).
rem Copies the sources to <scratch_dir>\src_copy, builds whirl-server-tests (control) and runs it, then deletes the ownerCheck of
rem PagePool::takeFree ("engine state changed from another thread, bypassing runOnLoop"), builds and runs again.
rem Expected: the control passes, the mutant FAILS in testOwnerCheck.
rem Usage: mutate_owner.bat <scratch_dir> [tokenizer.gguf] [kernels_from_build_dir]
rem   gguf default WHIRL_TEST_GGUF; kernels default build-asan of this tree (its HIP code objects are reused; same sources).
rem Detached; <scratch_dir>\MUT_DONE when finished; logs in <scratch_dir>\logs (steps.log has every exit code).
set "R=%~dp0..\.."
set "S=%~1"
set "GGUF=%~2"
if "%GGUF%"=="" set "GGUF=%WHIRL_TEST_GGUF%"
set "KF=%~3"
if "%KF%"=="" set "KF=%R%\build-asan"
if not exist "%S%\logs" mkdir "%S%\logs"
if exist "%S%\MUT_DONE" del "%S%\MUT_DONE"
robocopy "%R%" "%S%\src_copy" /MIR /XD build build-* .git /XF *.log /NFL /NDL /NJH /NJS /NP > "%S%\logs\robocopy.log"
call "%~dp0vcenv.bat" || exit /b 1
cmake -S "%S%\src_copy" -B "%S%\b" -G Ninja -DCMAKE_BUILD_TYPE=Release -DWHIRL_WITH_HIP=ON > "%S%\logs\configure.log" 2>&1
robocopy "%KF%\kernels" "%S%\b\kernels" /E /COPY:DAT /NFL /NDL /NJH /NJS /NP >nul
robocopy "%KF%\res" "%S%\b\res" /E /COPY:DAT /NFL /NDL /NJH /NJS /NP >nul
start "" /b /wait /low cmake --build "%S%\b" -j 8 --target whirl-server-tests > "%S%\logs\build_control.log" 2>&1
>"%S%\logs\steps.log" echo [!TIME!] build control exit !ERRORLEVEL!
"%S%\b\whirl-server-tests.exe" --gguf "%GGUF%" > "%S%\logs\run_control.log" 2>&1
>>"%S%\logs\steps.log" echo [!TIME!] run control exit !ERRORLEVEL!
rem the mutation: one ownerCheck line removed
findstr /v /c:"ownerCheck(owner_, \"PagePool::takeFree\");" "%S%\src_copy\src\cache\page_pool.cpp" > "%S%\page_pool.mut.cpp"
fc "%S%\src_copy\src\cache\page_pool.cpp" "%S%\page_pool.mut.cpp" > "%S%\logs\mutation.diff"
copy /y "%S%\page_pool.mut.cpp" "%S%\src_copy\src\cache\page_pool.cpp" >nul
start "" /b /wait /low cmake --build "%S%\b" -j 8 --target whirl-server-tests > "%S%\logs\build_mutant.log" 2>&1
>>"%S%\logs\steps.log" echo [!TIME!] build mutant exit !ERRORLEVEL!
"%S%\b\whirl-server-tests.exe" --gguf "%GGUF%" > "%S%\logs\run_mutant.log" 2>&1
>>"%S%\logs\steps.log" echo [!TIME!] run mutant exit !ERRORLEVEL!
>"%S%\MUT_DONE" echo done
