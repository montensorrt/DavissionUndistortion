# ===================================================================
# 最小调用测试：构建 → 加载指定图像 → 初始化 → 执行 → 释放
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1 -Image D:\a.bmp
#   powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1 -Matrix D:\botong25
#
# 离线参数（图像 / 矩阵目录）来自本脚本，代码里没有配置文件。
# ===================================================================
param(
    [string]$Image = "",
    [string]$Matrix = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $root
$v0 = Join-Path $workspace "undistortion_standaloneV0"

if ([string]::IsNullOrWhiteSpace($Image)) { $Image = Join-Path $v0 "demo\raw\PIC_1_1.bmp" }
if ([string]::IsNullOrWhiteSpace($Matrix)) { $Matrix = Join-Path $v0 "botong25" }
if (-not (Test-Path $Image)) { throw "图像不存在: $Image" }
if (-not (Test-Path $Matrix)) { throw "矩阵目录不存在: $Matrix" }

dotnet build (Join-Path $root "csharp\Undistortion.sln") -c Release --nologo | Out-Null
if ($LASTEXITCODE -ne 0) { throw "C# 构建失败" }

. (Join-Path $PSScriptRoot "_env.ps1")

$exe = Join-Path $root "csharp\Undistortion.Test\bin\Release\net8.0\Undistortion.Test.exe"
$out = Join-Path (Join-Path $root "csharp\Undistortion.Test\output") ([System.IO.Path]::GetFileName($Image))

& $exe --image $Image --matrix $Matrix --out $out
exit $LASTEXITCODE
