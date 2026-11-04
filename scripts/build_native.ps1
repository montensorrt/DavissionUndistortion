# ===================================================================
# 构建原生库（undist_api.dll）+ C++ 验收程序，并把 DLL 同步到 csharp\thirdpart
#   依赖：VS2022 (MSVC) / CUDA 12.8 / TensorRT 10.14 / OpenCV 4.12
#
#   为什么同步：thirdpart 是“交付快照”——客户端没有 CMake/VS，靠这一份 DLL 直接跑；
#   每次构建后覆盖它，避免 thirdpart 与源码脱节（双源真相）。
# ===================================================================
param(
    [string]$Config = "Release",
    [string]$TensorRtRoot = "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v10.14.1.48"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$ninjaDir = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

if (-not (Test-Path $vcvars)) { throw "vcvars64.bat 不存在: $vcvars" }
if (-not (Test-Path "$TensorRtRoot/include/NvInfer.h")) { throw "TensorRT 头文件不存在: $TensorRtRoot" }

$env:PATH = "$ninjaDir;$env:PATH"
$cmd = "call `"$vcvars`" >nul 2>&1 && cmake -S `"$root/trt`" -B `"$root/trt/build`" -G Ninja " +
       "-DCMAKE_BUILD_TYPE=$Config -DTENSORRT_ROOT=`"$TensorRtRoot`" && cmake --build `"$root/trt/build`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "native 构建失败 (exit=$LASTEXITCODE)" }

# ---- 同步交付快照到 csharp\thirdpart ----
$built = Join-Path $root "trt\build\undist_api.dll"
$thirdpart = Join-Path $root "csharp\thirdpart"
New-Item -ItemType Directory -Path $thirdpart -Force | Out-Null
Copy-Item $built (Join-Path $thirdpart "undist_api.dll") -Force
$hash = (Get-FileHash $built -Algorithm SHA256).Hash

Write-Host "[ok]   $built" -ForegroundColor Green
Write-Host "[sync] $thirdpart\undist_api.dll" -ForegroundColor Green
Write-Host "[hash] SHA256 $hash" -ForegroundColor DarkGray
