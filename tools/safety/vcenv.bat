@echo off
rem safe025: put MSVC (cl, link) on PATH unless it is there already. Called by the other tools\safety scripts (no setlocal: the
rem environment must reach the caller). Finds Visual Studio / Build Tools with vswhere, which every VS 2017+ installs at this fixed
rem place; WHIRL_VCVARS=<full path of vcvars64.bat> overrides the search.
where cl >nul 2>nul && exit /b 0
if defined WHIRL_VCVARS goto have
set "WHIRL_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%WHIRL_VSWHERE%" goto novswhere
for /f "usebackq delims=" %%i in (`"%WHIRL_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "WHIRL_VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
:have
if not exist "%WHIRL_VCVARS%" goto novcvars
call "%WHIRL_VCVARS%" >nul 2>nul
where cl >nul 2>nul && exit /b 0
echo vcenv: cl is not on PATH after "%WHIRL_VCVARS%"
exit /b 1
:novswhere
echo vcenv: vswhere.exe not found; set WHIRL_VCVARS to vcvars64.bat
exit /b 1
:novcvars
echo vcenv: vcvars64.bat not found (WHIRL_VCVARS=%WHIRL_VCVARS%)
exit /b 1
