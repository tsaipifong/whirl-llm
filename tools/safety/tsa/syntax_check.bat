@echo off
setlocal enabledelayedexpansion
rem safe025: can the host sources be compiled by clang-cl (AMD clang 21 of the HIP SDK) over the MSVC STL? Syntax only, with
rem -Wthread-safety, the same defines as the MSVC build. Usage: syntax_check.bat <generated_include_dir>   (prints exit=N per file)
rem Result 2026-10-25: 11 of 17 pass; src\server\{protocol,stream,engine,engine_vision,engine_gov,http} fail at include\whirl\json.h:30/41
rem (std::pair<std::string, json::Value> inside Value: incomplete type; MSVC accepts it, clang does not). Fixing json.h is a separate
rem portability bug fix (arch025 Q4), not part of safe025.
call "%~dp0..\vcenv.bat" || exit /b 1
set "CC=%WHIRL_CLANG_CL%"
if "%CC%"=="" if not "%HIP_PATH%"=="" set "CC=%HIP_PATH%\bin\clang-cl.exe"
if "%CC%"=="" set "CC=clang-cl"
set "S=%~dp0..\..\..\"
set "G=%~1"
set "FL=/std:c++20 /EHsc /MT /utf-8 /permissive- /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN /I%S%src /I%S%include /I%G% /clang:-fsyntax-only /clang:-Wthread-safety /clang:-Wno-unknown-pragmas"
for %%f in (src\server\log src\server\protocol src\server\tokens src\server\stream src\server\engine src\server\engine_thread src\server\engine_vision src\server\engine_gov src\cache\ckpt_store src\cache\page_pool src\server\prefix_store src\server\tier_bridge src\server\http src\cache\prefix_cache src\cache\tier_adapter src\cache\fingerprint src\device\null_backend) do (
  echo === %%f
  "!CC!" %FL% "%S%%%f.cpp" 2>&1
  echo exit=!ERRORLEVEL!
)
