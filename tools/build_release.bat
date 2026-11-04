@echo off
REM ===================================================================
REM build_release.bat  --  Undistortion Standalone V0 (CAPP) Release build
REM
REM 注意（本机环境坑点）：
REM   系统环境变量里同时存在 NO_PROXY 与 no_proxy（仅大小写不同），
REM   MSBuild 的 ToolTask 用区分大小写不敏感的 Hashtable 组装子进程环境，
REM   会抛 System.ArgumentException 并导致 "No CMAKE_CXX_COMPILER could be found"。
REM   因此这里先清空所有 *_proxy 变量再进 vcvars。
REM ===================================================================
setlocal

set "no_proxy="
set "NO_PROXY="
set "http_proxy="
set "HTTP_PROXY="
set "https_proxy="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "all_proxy="

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

set SRC=%~dp0..\CAPP
cd /d "%SRC%"

if /I "%~1"=="clean" (
    rmdir /s /q build 2>nul
)

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DOpenCV_DIR="E:/3rdparty/opencv/build"
if errorlevel 1 exit /b 1

cmake --build build --config Release --parallel
if errorlevel 1 exit /b 1

echo.
echo === build artifacts ===
dir /b build\bin 2>nul
endlocal
exit /b 0
